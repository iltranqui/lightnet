//! `[ppyoloe]` training: Task-Aligned assignment, per-cell deltas, loss terms
//! and the inference decode, ported from the raptor-fp4 reference
//! `src-lib/modern_yolo_layer.cpp` (kind `PPYOLOE`).
//!
//! Reference map (all in `modern_yolo_layer.cpp` unless noted):
//!
//! * `assign_tal_like` (`nms_aware = false`)      -> [`assign_ppyoloe`]
//! * `collect_candidates_for_truth` (TAL branch)  -> `collect_candidates`
//! * `train_batch_modern` (PPYOLOE branch)        -> `Head::train_image`
//! * `add_objectness_delta`                       -> `Head::add_objectness`
//! * `add_class_delta_multi`                      -> `Head::add_class_targets`
//! * `add_dfl_box_delta` / `add_dfl_delta`        -> `Head::add_dfl_box` (+ cpu `dfl_delta`)
//! * `add_dfl_giou_delta_unprojected`             -> `Head::add_dfl_giou`
//! * `assigned_class_loss_value`                  -> `Head::assigned_class_loss`
//! * `dfl_box_loss_value` / `dfl_side_loss_value` -> `Head::dfl_box_loss`
//! * `best_truth_iou_for_prediction`              -> `Head::best_truth_iou`
//! * `forward_modern_yolo_layer_impl` (cost)      -> [`ppyoloe_training_delta`]
//! * `get_modern_yolo_detections_raw_batch`       -> [`decode_ppyoloe`]
//! * `box_giou` (`box.cpp`)                       -> [`box_giou`]
//! * `dx_box_iou(..., GIOU)` (`box.cpp`)          -> [`giou_gradient`]
//! * `set_default_training_params`                -> [`PpyoloeWeights::default`]
//!
//! Layout, decode of the DFL box, activation and the `ModernBox` / `ModernLayout`
//! types are shared with `cpu.rs` (the YOLOX head and the existing TAL helpers).
//! This file does not call `cpu::assign_tal_like`: that port predates the
//! reference's conflict and quality rules, so the assignment is written out here.
//!
//! `output` must already be activated (`cpu::activate_modern_output`): the
//! objectness and class entries are sigmoids, the DFL logits stay raw.
//!
//! Not ported: the YACT mask path (`mask_coefs > 0`), the labels/track-id
//! output, the YOLO-NAS duplicate decay, and the `MODERN_YOLO_GRAD_CHECK` probes.
//! The ATSS warm-up flag is an input: the caller computes the schedule
//! (`make_modern_yolo_train_schedule` in C++).

use std::cmp::Ordering;

use super::cpu::{
    box_iou, decode_dfl_box, dfl_delta, dfl_expectation, dfl_probabilities, modern_class_score,
    modern_objectness, target_ltrb_distance, truth_box, truth_class, ModernBox, ModernLayout,
    ModernYoloError, ModernYoloSettings, CONFLICT_EPS, EPS, FLT_MAX, TAL_TOPK, VFL_ALPHA,
};
use crate::model::{ModernYoloKind, ModernYoloSpec};

/// Head-level knobs the PP-YOLOE path reads from the layer, with the C++ defaults
/// of `set_default_training_params` for `PPYOLOE`.
#[derive(Clone, Debug, PartialEq)]
pub struct PpyoloeWeights {
    /// C++ `box_loss_weight` (2.5 for PP-YOLOE).
    pub box_loss_weight: f32,
    /// C++ `dfl_loss_weight` (0.5 for PP-YOLOE).
    pub dfl_loss_weight: f32,
    /// C++ `center_radius`, in feature cells (2.5).
    pub center_radius: f32,
    /// C++ `assign_topk` (13 for the TAL heads).
    pub assign_topk: usize,
    /// C++ `tal_alpha`: exponent on the aligned class*objectness score (1.0).
    pub tal_alpha: f32,
    /// C++ `tal_beta`: exponent on the aligned IoU (6.0).
    pub tal_beta: f32,
    /// C++ `schedule.atss_warmup`: when true the candidate score is the IoU and
    /// the quality is `clamp(iou, 0.05, 1)`.
    pub atss_warmup: bool,
}

impl Default for PpyoloeWeights {
    /// C++ `set_default_training_params` for `PPYOLOE`.
    fn default() -> Self {
        Self {
            box_loss_weight: 2.5,
            dfl_loss_weight: 0.5,
            center_radius: 2.5,
            assign_topk: TAL_TOPK,
            tal_alpha: 1.0,
            tal_beta: 6.0,
            atss_warmup: false,
        }
    }
}

/// One truth's claim on one cell (C++ `Assignment`). `dfl_truth` of
/// `assign_tal_like` equals `truth` for PP-YOLOE, so only `truth` is kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PpyoloeAssignment {
    pub truth_index: usize,
    pub class_id: usize,
    /// Conflict key during assignment, then the clamped quality in `[0, 1]`.
    pub quality: f32,
    pub iou: f32,
    pub truth: ModernBox,
}

/// Output of [`assign_ppyoloe`] for one image.
#[derive(Clone, Debug, PartialEq)]
pub struct PpyoloeAssignments {
    /// One entry per cell, `n * spatial` long (C++ `slot = n * spatial + loc`).
    pub assignments: Vec<Option<PpyoloeAssignment>>,
    /// Row-major `[slot * classes + class]`: the best quality of any truth of
    /// that class at that cell, kept even when the truth loses the box conflict.
    pub class_quality: Vec<f32>,
}

/// Loss and count statistics for one batch (C++ `ModernYoloStats`, PP-YOLOE subset).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PpyoloeStats {
    pub sum_iou: f32,
    pub sum_iou_loss: f32,
    pub cls_loss: f32,
    pub box_loss: f32,
    /// Already multiplied by `dfl_loss_weight`, as `dfl_box_loss_value` returns it.
    pub dfl_loss: f32,
    /// Assigned cells.
    pub count: usize,
}

/// Output of [`ppyoloe_training_delta`] for a whole batch.
#[derive(Clone, Debug, PartialEq)]
pub struct PpyoloeTrainOutput {
    /// Whole-batch delta in `ModernLayout::entry_index` order, non-finite entries zeroed.
    pub delta: Vec<f32>,
    /// `||delta||^2` over the batch (C++ `*l.cost`).
    pub cost: f32,
    /// Assigned class per cell, `-1` for background, `batch * n * spatial` long.
    pub class_ids: Vec<i32>,
    pub stats: PpyoloeStats,
}

/// One decoded PP-YOLOE cell (C++ `dets[count]` before box correction).
#[derive(Clone, Debug, PartialEq)]
pub struct PpyoloeDetection {
    pub anchor: usize,
    pub loc: usize,
    /// C++ `dets[count].objectness`: objectness, or max class score when `cls_only`.
    pub objectness: f32,
    /// Normalised box from the DFL expectations (`decode_dfl_box`).
    pub bbox: ModernBox,
    /// Per-class probability, zeroed where it is not above the threshold.
    pub probs: Vec<f32>,
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn safe_log(value: f32) -> f32 {
    value.max(EPS).ln()
}

/// C++ `valid_truth`.
fn valid_truth(t: &ModernBox) -> bool {
    t.x.is_finite()
        && t.y.is_finite()
        && t.w.is_finite()
        && t.h.is_finite()
        && t.x > 0.0
        && t.x <= 1.0
        && t.y >= 0.0
        && t.y <= 1.0
        && t.w > 0.0
        && t.h > 0.0
}

/// C++ `point_inside_box_grid`: the pixel centre lies inside the truth box.
fn point_inside_box_grid(l: &ModernLayout, loc: usize, truth: &ModernBox) -> bool {
    let cx = ((loc % l.w) as f32 + 0.5) / l.w as f32;
    let cy = ((loc / l.w) as f32 + 0.5) / l.h as f32;
    cx >= truth.left() && cx <= truth.right() && cy >= truth.top() && cy <= truth.bottom()
}

/// C++ `point_inside_center_radius`, in feature cells.
fn point_inside_center_radius(l: &ModernLayout, loc: usize, truth: &ModernBox, radius: f32) -> bool {
    let gx = (loc % l.w) as f32 + 0.5;
    let gy = (loc / l.w) as f32 + 0.5;
    let tx = truth.x * l.w as f32;
    let ty = truth.y * l.h as f32;
    (gx - tx).abs() <= radius && (gy - ty).abs() <= radius
}

/// Rejects anything but a `[ppyoloe]` head with matching layout.
fn check_ppyoloe(spec: &ModernYoloSpec, l: &ModernLayout) -> Result<(), ModernYoloError> {
    if spec.kind != ModernYoloKind::Ppyoloe || l.kind != ModernYoloKind::Ppyoloe {
        return Err(ModernYoloError(format!(
            "ppyoloe_train needs a [ppyoloe] head, got {:?}",
            spec.kind
        )));
    }
    Ok(())
}

/// One candidate cell for one truth (C++ `Candidate`, TAL branch).
#[derive(Clone, Copy, Debug)]
struct PpyoloeCandidate {
    anchor: usize,
    loc: usize,
    class_id: usize,
    iou: f32,
    score: f32,
}

/// C++ `collect_candidates_for_truth` for the TAL score. The `min_object_cells`
/// gate is applied by the caller, as in `assign_tal_like`.
fn collect_candidates(
    x: &[f32],
    l: &ModernLayout,
    weights: &PpyoloeWeights,
    b: usize,
    class_id: usize,
    truth: &ModernBox,
) -> Vec<PpyoloeCandidate> {
    let mut out = Vec::new();
    for n in 0..l.n {
        for loc in 0..l.spatial() {
            let inside_box = point_inside_box_grid(l, loc, truth);
            let geometry_ok = inside_box
                || (weights.center_radius > 0.0
                    && point_inside_center_radius(l, loc, truth, weights.center_radius));
            if !geometry_ok {
                continue;
            }
            let pred = decode_dfl_box(x, l, b, n, loc);
            let iou = finite_or_zero(box_iou(&pred, truth)).clamp(0.0, 1.0);
            let obj = modern_objectness(x, l, b, n, loc);
            let cls = modern_class_score(x, l, b, n, loc, class_id);
            let score = if weights.atss_warmup {
                iou
            } else {
                let aligned_cls = (cls * obj).max(EPS).powf(weights.tal_alpha);
                let aligned_iou = iou.max(EPS).powf(weights.tal_beta);
                aligned_cls * aligned_iou
            };
            let cost = -safe_log(score + EPS);
            if score.is_finite() && cost.is_finite() {
                out.push(PpyoloeCandidate { anchor: n, loc, class_id, iou, score });
            }
        }
    }
    out
}

/// C++ `assign_tal_like` with `nms_aware = false` (PP-YOLOE).
///
/// Per truth: collect candidates, sort by score (descending), keep the top
/// `assign_topk`, and give each kept cell a quality
/// `clamp(score / max(max_score, eps) * max_iou, 0, 1)`. A cell goes to the
/// smallest truth that claims it (the LegoGears nested-object rule); the score
/// breaks ties between truths of equal area.
#[allow(clippy::too_many_arguments)]
pub fn assign_ppyoloe(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    weights: &PpyoloeWeights,
    b: usize,
) -> Result<PpyoloeAssignments, ModernYoloError> {
    check_ppyoloe(spec, l)?;
    let truths = spec.max_boxes * settings.truth_size;
    if truth.len() < truths * l.batch {
        return Err(ModernYoloError(format!(
            "anchor-free truth needs {} entries for {} images x {} boxes x {} fields, got {}",
            truths * l.batch,
            l.batch,
            spec.max_boxes,
            settings.truth_size,
            truth.len()
        )));
    }
    let slots = l.n * l.spatial();
    let mut assignments: Vec<Option<PpyoloeAssignment>> = vec![None; slots];
    let mut conflict_score = vec![-1.0f32; slots];
    let mut conflict_area = vec![f32::INFINITY; slots];
    let mut class_quality = vec![0.0f32; slots * l.classes];

    for t in 0..spec.max_boxes {
        let box_ = truth_box(truth, settings.truth_size, truths, t, b);
        if box_.x == 0.0 {
            break;
        }
        if !valid_truth(&box_) {
            return Err(ModernYoloError(format!(
                "invalid anchor-free truth box (x={}, y={}, w={}, h={})",
                box_.x, box_.y, box_.w, box_.h
            )));
        }
        let mut class_id = truth_class(truth, settings.truth_size, truths, t, b);
        if class_id >= l.classes {
            return Err(ModernYoloError(format!("invalid class ID #{class_id}")));
        }
        if let Some(map) = &settings.class_map {
            class_id = map[class_id];
        }
        // C++ `collect_candidates_for_truth` returns no candidates for a truth
        // narrower than `min_object_cells` grid cells.
        if spec.min_object_cells > 0.0
            && ((box_.w * l.w as f32) < spec.min_object_cells
                || (box_.h * l.h as f32) < spec.min_object_cells)
        {
            continue;
        }
        let mut candidates = collect_candidates(x, l, weights, b, class_id, &box_);
        if candidates.is_empty() {
            continue;
        }
        // TODO(verify): C++ uses std::sort (unstable); tie order between equal
        // scores may differ. This is a stable sort.
        candidates.sort_by(|lhs, rhs| rhs.score.partial_cmp(&lhs.score).unwrap_or(Ordering::Equal));
        let keep = weights.assign_topk.min(candidates.len());
        let mut max_score = 0.0f32;
        let mut max_iou = 0.0f32;
        for c in &candidates[..keep] {
            max_score = max_score.max(c.score);
            max_iou = max_iou.max(c.iou);
        }
        let truth_area = box_.w * box_.h;
        for c in &candidates[..keep] {
            let slot = c.anchor * l.spatial() + c.loc;
            let quality = if weights.atss_warmup {
                c.iou.clamp(0.05, 1.0)
            } else {
                (c.score / max_score.max(EPS) * max_iou).clamp(0.0, 1.0)
            };
            // Recorded even if this truth loses the box conflict below.
            let cq = &mut class_quality[slot * l.classes + c.class_id];
            *cq = cq.max(quality);

            let smaller_truth_wins = truth_area < conflict_area[slot] - CONFLICT_EPS;
            let tied_area_better_score = (truth_area - conflict_area[slot]).abs() <= CONFLICT_EPS
                && c.score > conflict_score[slot];
            if smaller_truth_wins || tied_area_better_score {
                conflict_area[slot] = truth_area;
                conflict_score[slot] = c.score;
                assignments[slot] = Some(PpyoloeAssignment {
                    truth_index: t,
                    class_id: c.class_id,
                    quality,
                    iou: c.iou,
                    truth: box_,
                });
            }
        }
    }
    Ok(PpyoloeAssignments { assignments, class_quality })
}

/// C++ `dx_box_iou(pred, truth, GIOU)` in `box.cpp`, GIOU branch.
///
/// Returns `[dt, db, dl, dr]`, the gradient of IoU - enclosing penalty with
/// respect to `(x, y, w, h)` of `pred`, in the C++ convention:
/// `dt = d/dx`, `db = d/dy`, `dl = d/dw`, `dr = d/dh`, with `dw` and `dh` not
/// halved (`dl = p_dr - p_dl`, `dr = p_db - p_dt`).
pub fn giou_gradient(pred: &ModernBox, truth: &ModernBox) -> [f32; 4] {
    let pt0 = pred.top();
    let pb0 = pred.bottom();
    let pl0 = pred.left();
    let pr0 = pred.right();
    let pred_t = pt0.min(pb0);
    let pred_b = pt0.max(pb0);
    let pred_l = pl0.min(pr0);
    let pred_r = pl0.max(pr0);
    let (tt, tb, tl, tr) = (truth.top(), truth.bottom(), truth.left(), truth.right());

    let x_area = (pred_b - pred_t) * (pred_r - pred_l);
    let xhat = (tb - tt) * (tr - tl);
    let ih = pred_b.min(tb) - pred_t.max(tt);
    let iw = pred_r.min(tr) - pred_l.max(tl);
    let inter = iw * ih;
    let union = x_area + xhat - inter;
    let cw = pred_r.max(tr) - pred_l.min(tl);
    let ch = pred_b.max(tb) - pred_t.min(tt);
    let c_area = cw * ch;

    let dx_t = -(pred_r - pred_l);
    let dx_b = pred_r - pred_l;
    let dx_l = -(pred_b - pred_t);
    let dx_r = pred_b - pred_t;
    let di_t = if pred_t > tt { -iw } else { 0.0 };
    let di_b = if pred_b < tb { iw } else { 0.0 };
    let di_l = if pred_l > tl { -ih } else { 0.0 };
    let di_r = if pred_r < tr { ih } else { 0.0 };
    let du_t = dx_t - di_t;
    let du_b = dx_b - di_b;
    let du_l = dx_l - di_l;
    let du_r = dx_r - di_r;
    let dc_t = if pred_t < tt { -cw } else { 0.0 };
    let dc_b = if pred_b > tb { cw } else { 0.0 };
    let dc_l = if pred_l < tl { -ch } else { 0.0 };
    let dc_r = if pred_r > tr { ch } else { 0.0 };

    let (mut p_dt, mut p_db, mut p_dl, mut p_dr) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    if union > 0.0 {
        let u2 = union * union;
        p_dt = (union * di_t - inter * du_t) / u2;
        p_db = (union * di_b - inter * du_b) / u2;
        p_dl = (union * di_l - inter * du_l) / u2;
        p_dr = (union * di_r - inter * du_r) / u2;
    }
    // TODO(verify): literal port of the C++ sequential corner-selection assignments
    // (`p_dt = top<bot ? p_dt : p_db; p_db = top<bot ? p_db : p_dt;`). Since h > 0
    // for every sanitised box these branches never fire.
    let (p_dt, p_db) = if pt0 < pb0 { (p_dt, p_db) } else { (p_db, p_db) };
    let (p_dl, p_dr) = if pl0 < pr0 { (p_dl, p_dr) } else { (p_dr, p_dr) };
    let (mut p_dt, mut p_db, mut p_dl, mut p_dr) = (p_dt, p_db, p_dl, p_dr);

    // GIoU enclosing-box term.
    if c_area > 0.0 {
        let c2 = c_area * c_area;
        p_dt += (c_area * du_t - union * dc_t) / c2;
        p_db += (c_area * du_b - union * dc_b) / c2;
        p_dl += (c_area * du_l - union * dc_l) / c2;
        p_dr += (c_area * du_r - union * dc_r) / c2;
    }
    // Degenerate overlap: C++ overwrites with the enclosing-term gradient only.
    if iw <= 0.0 || ih <= 0.0 {
        let c2 = c_area * c_area;
        p_dt = (c_area * du_t - union * dc_t) / c2;
        p_db = (c_area * du_b - union * dc_b) / c2;
        p_dl = (c_area * du_l - union * dc_l) / c2;
        p_dr = (c_area * du_r - union * dc_r) / c2;
    }

    [p_dl + p_dr, p_dt + p_db, p_dr - p_dl, p_db - p_dt]
}

/// C++ `box_giou` in `box.cpp`: IoU minus `(C - U) / C`, with `C` the enclosing box.
pub fn box_giou(a: &ModernBox, b: &ModernBox) -> f32 {
    let iou = box_iou(a, b);
    let c_left = a.left().min(b.left());
    let c_right = a.right().max(b.right());
    let c_top = a.top().min(b.top());
    let c_bottom = a.bottom().max(b.bottom());
    let c = (c_right - c_left) * (c_bottom - c_top);
    if c == 0.0 {
        return iou;
    }
    let ow = a.right().min(b.right()) - a.left().max(b.left());
    let oh = a.bottom().min(b.bottom()) - a.top().max(b.top());
    let inter = if ow <= 0.0 || oh <= 0.0 { 0.0 } else { ow * oh };
    let union = a.w * a.h + b.w * b.h - inter;
    iou - (c - union) / c
}

/// Per-head state for one call: the activated output, the raw truth rows and
/// the layer, spec, settings and weights.
struct Head<'a> {
    x: &'a [f32],
    truth: &'a [f32],
    l: &'a ModernLayout,
    spec: &'a ModernYoloSpec,
    settings: &'a ModernYoloSettings,
    weights: &'a PpyoloeWeights,
}

impl Head<'_> {
    /// C++ `add_objectness_delta`: `obj_normalizer * (clamp(target) - obj)`.
    fn add_objectness(&self, delta: &mut [f32], b: usize, n: usize, loc: usize, target: f32) {
        let idx = self.l.obj_index(b, n, loc);
        let d = self.settings.obj_normalizer
            * (target.clamp(0.0, 1.0) - finite_or_zero(self.x[idx]));
        if d.is_finite() {
            delta[idx] += d;
        }
    }

    /// C++ `add_class_delta_multi` (`enable_vfl = true`): one target per class
    /// from `targets`, with the Varifocal weight and class multipliers.
    fn add_class_targets(&self, delta: &mut [f32], b: usize, n: usize, loc: usize, targets: &[f32]) {
        let use_vfl = self.spec.vfl_gamma > 0.0;
        let eps = self.settings.label_smooth_eps;
        for c in 0..self.l.classes {
            let mut target = targets[c].clamp(0.0, 1.0);
            if eps != 0.0 && !use_vfl {
                target = target * (1.0 - eps) + 0.5 * eps;
            }
            let idx = self.l.class_index(b, n, loc, c);
            let p = finite_or_zero(self.x[idx]).clamp(0.0, 1.0);
            let mut d = self.settings.cls_normalizer * (target - p);
            if use_vfl {
                let weight = if target > 0.0 {
                    target
                } else {
                    VFL_ALPHA * p.powf(self.spec.vfl_gamma)
                };
                d *= weight;
            }
            if target > 0.0 {
                if let Some(mult) = &self.settings.classes_multipliers {
                    d *= mult[c];
                }
            }
            if d.is_finite() {
                delta[idx] += d;
            }
        }
    }

    /// C++ `add_dfl_box_delta`: the two-bin distribution target on each side.
    fn add_dfl_box(&self, delta: &mut [f32], b: usize, n: usize, loc: usize, truth: &ModernBox, normalizer: f32) {
        for side in 0..4 {
            let distance = target_ltrb_distance(self.l, loc, truth, side);
            dfl_delta(delta, self.x, self.l, b, n, loc, side, distance, normalizer);
        }
    }

    /// C++ `add_dfl_giou_delta_unprojected`: pushes `weight * dIoU/d(corner)`
    /// through the DFL expectation into each bin logit.
    fn add_dfl_giou(&self, delta: &mut [f32], b: usize, n: usize, loc: usize, truth: &ModernBox, weight: f32) {
        let l = self.l;
        let bins = l.dfl_bins();
        let stride = l.spatial();
        let location = n * stride + loc;
        let pred = decode_dfl_box(self.x, l, b, n, loc);
        let g = giou_gradient(&pred, truth);
        let inv_w = 1.0 / l.w as f32;
        let inv_h = 1.0 / l.h as f32;
        // `g` carries the w/h gradients without the 0.5 edge factor, so the corner
        // gradients are dIoU/dx1 = 0.5 (dt - dl), dIoU/dy1 = 0.5 (db - dr), and
        // dIoU/dx2 = 0.5 (dt + dl), dIoU/dy2 = 0.5 (db + dr).
        let corner_grad = [
            finite_or_zero(0.5 * (g[0] - g[2])),
            finite_or_zero(0.5 * (g[1] - g[3])),
            finite_or_zero(0.5 * (g[0] + g[2])),
            finite_or_zero(0.5 * (g[1] + g[3])),
        ];
        let expectation_scale = [-inv_w, -inv_h, inv_w, inv_h];
        for side in 0..4 {
            let probs = dfl_probabilities(self.x, l, b, location, side);
            let expected = finite_or_zero(dfl_expectation(self.x, l, b, location, side));
            let start = l.entry_index(b, location, side * bins);
            let chain = finite_or_zero(weight) * corner_grad[side] * expectation_scale[side];
            for (k, prob) in probs.iter().enumerate() {
                let mut d = chain * finite_or_zero(*prob) * (k as f32 - expected);
                if self.settings.max_delta != FLT_MAX {
                    d = d.clamp(-self.settings.max_delta, self.settings.max_delta);
                }
                if d.is_finite() {
                    delta[start + k * stride] += d;
                }
            }
        }
    }

    /// C++ `assigned_class_loss_value` (`enable_vfl = true`).
    fn assigned_class_loss(&self, b: usize, n: usize, loc: usize, class_id: usize, quality: f32) -> f32 {
        let use_vfl = self.spec.vfl_gamma > 0.0;
        let eps = self.settings.label_smooth_eps;
        let mut target = quality.clamp(0.0, 1.0);
        if eps != 0.0 && !use_vfl {
            target = target * (1.0 - eps) + 0.5 * eps;
        }
        let p = modern_class_score(self.x, self.l, b, n, loc, class_id);
        let mut loss = -target * safe_log(p) - (1.0 - target) * safe_log(1.0 - p);
        if use_vfl {
            loss *= target;
        }
        finite_or_zero(loss)
    }

    /// C++ `dfl_box_loss_value`: `dfl_loss_weight` times the four two-bin
    /// cross-entropies against the truth distances.
    fn dfl_box_loss(&self, b: usize, n: usize, loc: usize, truth: &ModernBox) -> f32 {
        let l = self.l;
        let bins = l.dfl_bins();
        let max_bin = bins - 1;
        let location = n * l.spatial() + loc;
        let mut loss = 0.0f32;
        for side in 0..4 {
            let distance = target_ltrb_distance(l, loc, truth, side);
            let td = finite_or_zero(distance).clamp(0.0, max_bin as f32 - 1e-4);
            let left = (td.floor() as usize).min(max_bin);
            let right = (left + 1).min(max_bin);
            let right_weight = td - left as f32;
            let left_weight = 1.0 - right_weight;
            let probs = dfl_probabilities(self.x, l, b, location, side);
            loss += finite_or_zero(
                left_weight * -safe_log(probs[left]) + right_weight * -safe_log(probs[right]),
            );
        }
        finite_or_zero(self.weights.dfl_loss_weight * loss)
    }

    /// C++ `best_truth_iou_for_prediction`: best IoU of the decoded box against
    /// every truth of the image, used by the ignore band of background cells.
    fn best_truth_iou(&self, b: usize, n: usize, loc: usize) -> f32 {
        let pred = decode_dfl_box(self.x, self.l, b, n, loc);
        let truths = self.spec.max_boxes * self.settings.truth_size;
        let mut best = 0.0f32;
        for t in 0..self.spec.max_boxes {
            let tb = truth_box(self.truth, self.settings.truth_size, truths, t, b);
            if tb.x == 0.0 {
                break;
            }
            best = best.max(finite_or_zero(box_iou(&pred, &tb)).clamp(0.0, 1.0));
        }
        best
    }

    /// C++ `train_batch_modern`, PP-YOLOE branch, for one image. Writes into the
    /// whole-batch `delta` at image `b` and the image's `class_ids` slice.
    fn train_image(
        &self,
        b: usize,
        delta: &mut [f32],
        class_ids: &mut [i32],
        stats: &mut PpyoloeStats,
    ) -> Result<(), ModernYoloError> {
        let l = self.l;
        let spatial = l.spatial();
        let classes = l.classes;
        let asg = assign_ppyoloe(self.x, self.truth, l, self.spec, self.settings, self.weights, b)?;
        let zeros = vec![0.0f32; classes];
        for n in 0..l.n {
            for loc in 0..spatial {
                let slot = n * spatial + loc;
                match asg.assignments[slot] {
                    None => {
                        if self.spec.vfl_gamma > 0.0 {
                            self.add_class_targets(delta, b, n, loc, &zeros);
                        }
                        let best = self.best_truth_iou(b, n, loc);
                        if best > self.settings.ignore_thresh {
                            if self.settings.objectness_smooth {
                                self.add_objectness(delta, b, n, loc, best);
                            }
                            continue;
                        }
                        self.add_objectness(delta, b, n, loc, 0.0);
                    }
                    Some(a) => {
                        let class_multiplier = self
                            .settings
                            .classes_multipliers
                            .as_ref()
                            .map_or(1.0, |m| m[a.class_id]);
                        let quality = a.quality.clamp(0.0, 1.0);
                        self.add_objectness(delta, b, n, loc, quality);
                        // The winner's own channel takes the quality actually used
                        // for box loss; other classes keep their per-class targets.
                        let mut row = asg.class_quality[slot * classes..(slot + 1) * classes].to_vec();
                        row[a.class_id] = quality;
                        self.add_class_targets(delta, b, n, loc, &row);
                        self.add_dfl_box(
                            delta,
                            b,
                            n,
                            loc,
                            &a.truth,
                            self.settings.iou_normalizer * class_multiplier * self.weights.dfl_loss_weight,
                        );
                        let giou_weight = self.settings.iou_normalizer
                            * class_multiplier
                            * self.weights.box_loss_weight
                            * quality;
                        self.add_dfl_giou(delta, b, n, loc, &a.truth, giou_weight);

                        class_ids[slot] = a.class_id as i32;
                        stats.sum_iou += a.iou;
                        stats.sum_iou_loss += 1.0 - a.iou;
                        stats.cls_loss += self.assigned_class_loss(b, n, loc, a.class_id, quality);
                        let pred = decode_dfl_box(self.x, l, b, n, loc);
                        let giou = finite_or_zero(box_giou(&pred, &a.truth));
                        stats.box_loss += self.weights.box_loss_weight * quality * (1.0 - giou);
                        stats.dfl_loss += self.dfl_box_loss(b, n, loc, &a.truth);
                        stats.count += 1;
                    }
                }
            }
        }
        Ok(())
    }
}

/// C++ `forward_modern_yolo_layer_impl` (PP-YOLOE, no mask): the whole-batch
/// training delta and its cost. `output` must be activated.
pub fn ppyoloe_training_delta(
    output: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    weights: &PpyoloeWeights,
) -> Result<PpyoloeTrainOutput, ModernYoloError> {
    check_ppyoloe(spec, l)?;
    if output.len() < l.len() {
        return Err(ModernYoloError(format!(
            "ppyoloe head output needs {} entries, got {}",
            l.len(),
            output.len()
        )));
    }
    let head = Head { x: output, truth, l, spec, settings, weights };
    let slots = l.n * l.spatial();
    let mut delta = vec![0.0f32; l.len()];
    let mut class_ids = vec![-1i32; l.batch * slots];
    let mut stats = PpyoloeStats::default();
    for b in 0..l.batch {
        head.train_image(b, &mut delta, &mut class_ids[b * slots..(b + 1) * slots], &mut stats)?;
    }
    for v in delta.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    let cost = delta.iter().map(|v| v * v).sum::<f32>();
    Ok(PpyoloeTrainOutput { delta, cost, class_ids, stats })
}

/// C++ `get_modern_yolo_detections_raw_batch` for PP-YOLOE (before box correction).
///
/// Cells are visited location-major, then anchor, as in C++. `cls_only` is
/// `score_mode == 1`: the detection score becomes the max class score and the
/// per-class probability is the class score alone.
pub fn decode_ppyoloe(
    x: &[f32],
    l: &ModernLayout,
    b: usize,
    thresh: f32,
    cls_only: bool,
) -> Vec<PpyoloeDetection> {
    let mut out = Vec::new();
    for loc in 0..l.spatial() {
        for n in 0..l.n {
            let objectness = modern_objectness(x, l, b, n, loc);
            let detection_score = if cls_only {
                let mut best = 0.0f32;
                for c in 0..l.classes {
                    best = best.max(modern_class_score(x, l, b, n, loc, c));
                }
                best
            } else {
                objectness
            };
            if detection_score <= thresh {
                continue;
            }
            let probs = (0..l.classes)
                .map(|c| {
                    let class_score = modern_class_score(x, l, b, n, loc, c);
                    let prob = if cls_only { class_score } else { objectness * class_score };
                    if prob > thresh { prob } else { 0.0 }
                })
                .collect();
            out.push(PpyoloeDetection {
                anchor: n,
                loc,
                objectness: detection_score,
                bbox: decode_dfl_box(x, l, b, n, loc),
                probs,
            });
        }
    }
    out
}

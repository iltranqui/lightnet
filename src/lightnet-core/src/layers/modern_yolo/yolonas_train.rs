//! `[yolonas]` training path: assignment, loss terms, per-cell deltas and decode.
//!
//! Ported from `raptor-fp4:src-lib/modern_yolo_layer.cpp` (anonymous-namespace
//! helpers `assign_tal_like`, `train_batch_modern`, `add_dfl_delta`,
//! `add_dfl_giou_delta_unprojected`, `add_class_delta_multi`,
//! `get_modern_yolo_detections_raw_batch`, `correct_modern_yolo_boxes`) and the
//! IoU helpers of `raptor-fp4:src-lib/box.cpp` (`box_iou`, `box_giou`,
//! `dx_box_iou` with `GIOU`).  The GPU twin of the per-cell delta step is
//! `src/lightnet-cuda-oxide/src/kernels/yolonas_train.rs`.
//!
//! The head input is the post-activation tensor (`activate_modern_output` in
//! `cpu.rs` has run): objectness and class channels are sigmoided, the four
//! distribution blocks of `reg_max + 1` logits are raw.  Deltas are descent
//! directions, as in C++.  The per-cell order is the C++ order:
//!
//! 1. [`yolonas_assign`]: candidates per truth, top-k by score, conflict
//!    tie-break (smaller truth wins), per-class quality row, duplicate decay.
//! 2. [`yolonas_train_delta`]: per cell, objectness, class (VFL), DFL box and
//!    DFL-GIoU, or the ignore band / background target.
//!
//! Decode is [`yolonas_decode`] followed by [`yolonas_correct_boxes`].
//!
//! Deliberate differences from `cpu.rs`: the `[yolonas]` path here follows the
//! raptor reference exactly (its `assign_tal_like` and `train_batch_modern`),
//! not the simplified `cpu.rs::assign_tal_like`.

use super::cpu::{
    EPS, FLT_MAX, ModernBox, ModernLayout, ModernYoloError, VFL_ALPHA, decode_dfl_box, dfl_delta,
    dfl_expectation, dfl_probabilities, target_ltrb_distance, truth_box,
};
use crate::model::{ModernYoloKind, ModernYoloSpec};

/// Net-level knobs of the `[yolonas]` loss.  `Default` holds the C++
/// `set_default_training_params` values for the YOLO-NAS head; the
/// spec-carried fields come from [`NasTrainParams::for_spec`].
#[derive(Clone, Debug, PartialEq)]
pub struct NasTrainParams {
    /// Floats per truth row (C++ `l.truth_size`; class at index 4).
    /// TODO(verify): default 5 here, `cpu.rs` uses 6 (track id).
    pub truth_size: usize,
    /// C++ `ignore_thresh`, default 0.7.
    pub ignore_thresh: f32,
    /// C++ `iou_normalizer`, default 1.
    pub iou_normalizer: f32,
    /// C++ `obj_normalizer`, default 1.
    pub obj_normalizer: f32,
    /// C++ `cls_normalizer`, default 1.
    pub cls_normalizer: f32,
    /// C++ `max_delta`; `FLT_MAX` disables the clip (only the DFL-GIoU delta clips).
    pub max_delta: f32,
    /// C++ `label_smooth_eps`, default 0 (ignored when `vfl_gamma > 0`).
    pub label_smooth_eps: f32,
    /// C++ `objectness_smooth`: ignore-band cells get `best_iou` as target.
    pub objectness_smooth: bool,
    /// C++ `center_radius` in cells, default 2.5.
    pub center_radius: f32,
    /// C++ `assign_topk`, default 13 for the TAL-like heads.
    pub assign_topk: usize,
    /// C++ `tal_alpha`, default 1.
    pub tal_alpha: f32,
    /// C++ `tal_beta`, default 6.
    pub tal_beta: f32,
    /// C++ `box_loss_weight`, default 2.5 for the TAL-like heads.
    pub box_loss_weight: f32,
    /// C++ `dfl_loss_weight`, default 0.25 for YOLO-NAS.
    pub dfl_loss_weight: f32,
    /// C++ `nas_duplicate_decay`, default 0.35; `0` disables the decay.
    pub nas_duplicate_decay: f32,
    /// From the spec: truths narrower or shorter than this (cells) are skipped.
    pub min_object_cells: f32,
    /// From the spec: Varifocal exponent; `<= 0` disables VFL weighting.
    pub vfl_gamma: f32,
    /// C++ `atss_warmup_iters` already resolved for this iteration.
    pub atss_warmup: bool,
    /// C++ `score_mode == 1`: detection score is the best class score only.
    pub score_cls_only: bool,
    /// C++ `classes_multipliers`, one per class when present.
    pub classes_multipliers: Option<Vec<f32>>,
    /// C++ `map`: remaps the truth class id before assignment.
    pub class_map: Option<Vec<usize>>,
}

impl Default for NasTrainParams {
    fn default() -> Self {
        Self {
            truth_size: 5,
            ignore_thresh: 0.7,
            iou_normalizer: 1.0,
            obj_normalizer: 1.0,
            cls_normalizer: 1.0,
            max_delta: FLT_MAX,
            label_smooth_eps: 0.0,
            objectness_smooth: false,
            center_radius: 2.5,
            assign_topk: 13,
            tal_alpha: 1.0,
            tal_beta: 6.0,
            box_loss_weight: 2.5,
            dfl_loss_weight: 0.25,
            nas_duplicate_decay: 0.35,
            min_object_cells: 0.0,
            vfl_gamma: 0.0,
            atss_warmup: false,
            score_cls_only: false,
            classes_multipliers: None,
            class_map: None,
        }
    }
}

impl NasTrainParams {
    /// Defaults plus the spec-carried gates (`min_object_cells`, `vfl_gamma`).
    pub fn for_spec(spec: &ModernYoloSpec) -> Self {
        Self {
            min_object_cells: spec.min_object_cells,
            vfl_gamma: spec.vfl_gamma,
            ..Self::default()
        }
    }
}

/// Loss sums of one batch, as C++ `ModernYoloStats` (YOLO-NAS subset).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NasStats {
    pub sum_iou: f32,
    pub sum_iou_loss: f32,
    pub cls_loss: f32,
    pub box_loss: f32,
    pub dfl_loss: f32,
    /// Assigned cells only (C++ `stats.count`).
    pub count: usize,
    /// C++ `loss = pow(mag_array(delta), 2)` over the whole batch delta.
    pub cost: f32,
}

/// One cell's claim by one truth, as C++ `Assignment`.
/// C++ also keeps `dfl_truth`; on the TAL path it is the same box as `truth`,
/// so only `truth` is stored.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NasAssignment {
    /// Truth row within the image, `None` for background cells.
    pub truth_index: Option<usize>,
    pub class_id: usize,
    /// Quality after the duplicate decay; clamped to `[0, 1]` at delta time.
    pub quality: f32,
    pub iou: f32,
    pub truth: ModernBox,
}

/// Assignment of one image: one entry per cell plus the per-class rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NasAssignments {
    /// `n * H * W` entries, indexed by `slot = n * H * W + loc`.
    pub slots: Vec<NasAssignment>,
    /// Row-major `[slot * classes + class]`, C++ `class_quality`.
    pub class_quality: Vec<f32>,
}

/// One candidate cell for one truth, as C++ `Candidate` (TAL branch).
#[derive(Clone, Copy, Debug)]
struct NasCandidate {
    anchor: usize,
    loc: usize,
    truth_index: usize,
    class_id: usize,
    iou: f32,
    score: f32,
    truth: ModernBox,
}

/// One decoded detection, as C++ `Detection` (box, objectness, class probs).
#[derive(Clone, Debug, PartialEq)]
pub struct NasDetection {
    pub bbox: ModernBox,
    pub objectness: f32,
    pub prob: Vec<f32>,
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn clamp01(value: f32) -> f32 {
    finite_or_zero(value).clamp(0.0, 1.0)
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

/// C++ `overlap` (box.cpp, anonymous namespace).
fn overlap(x1: f32, w1: f32, x2: f32, w2: f32) -> f32 {
    let left = (x1 - w1 / 2.0).max(x2 - w2 / 2.0);
    let right = (x1 + w1 / 2.0).min(x2 + w2 / 2.0);
    right - left
}

/// C++ `box_intersection`.
fn box_intersection(a: &ModernBox, b: &ModernBox) -> f32 {
    let w = overlap(a.x, a.w, b.x, b.w);
    if w <= 0.0 {
        return 0.0;
    }
    let h = overlap(a.y, a.h, b.y, b.h);
    if h <= 0.0 {
        return 0.0;
    }
    w * h
}

/// C++ `box_union(a, b, intersection)`.
fn box_union_with(a: &ModernBox, b: &ModernBox, intersection: f32) -> f32 {
    a.w * a.h + b.w * b.h - intersection
}

/// C++ `box_iou` (box.cpp).  Not clamped; callers clamp as C++ does.
pub fn box_iou_ref(a: &ModernBox, b: &ModernBox) -> f32 {
    let i = box_intersection(a, b);
    if i == 0.0 {
        return 0.0;
    }
    let u = box_union_with(a, b, i);
    i / u
}

/// C++ `box_giou` (box.cpp).
fn box_giou_ref(a: &ModernBox, b: &ModernBox) -> f32 {
    let top = (a.y - a.h / 2.0).min(b.y - b.h / 2.0);
    let bot = (a.y + a.h / 2.0).max(b.y + b.h / 2.0);
    let left = (a.x - a.w / 2.0).min(b.x - b.w / 2.0);
    let right = (a.x + a.w / 2.0).max(b.x + b.w / 2.0);
    let c = (right - left) * (bot - top);
    let iou = box_iou_ref(a, b);
    if c == 0.0 {
        return iou;
    }
    let i = box_intersection(a, b);
    let u = box_union_with(a, b, i);
    let giou_term = (c - u) / c;
    iou - giou_term
}

/// C++ `dx_box_iou(pred, truth, GIOU)` (box.cpp), GIOU branch only.
/// Returns `[dt, db, dl, dr]` in the C++ field order; `dt+db` and `dl+dr`
/// are the edge-pair sums, `dl`/`dr` carry the `w`/`h` gradients.
fn dx_box_giou(pred: &ModernBox, truth: &ModernBox) -> [f32; 4] {
    // to_tblr(pred), then the sorted edges as dx_box_iou() takes them.
    let p_top0 = pred.y - pred.h / 2.0;
    let p_bot0 = pred.y + pred.h / 2.0;
    let p_left0 = pred.x - pred.w / 2.0;
    let p_right0 = pred.x + pred.w / 2.0;
    let pred_t = p_top0.min(p_bot0);
    let pred_b = p_top0.max(p_bot0);
    let pred_l = p_left0.min(p_right0);
    let pred_r = p_left0.max(p_right0);
    let t_top = truth.y - truth.h / 2.0;
    let t_bot = truth.y + truth.h / 2.0;
    let t_left = truth.x - truth.w / 2.0;
    let t_right = truth.x + truth.w / 2.0;

    let x_area = (pred_b - pred_t) * (pred_r - pred_l);
    let x_hat = (t_bot - t_top) * (t_right - t_left);
    let ih = pred_b.min(t_bot) - pred_t.max(t_top);
    let iw = pred_r.min(t_right) - pred_l.max(t_left);
    let i = iw * ih;
    let u = x_area + x_hat - i;
    let giou_cw = pred_r.max(t_right) - pred_l.min(t_left);
    let giou_ch = pred_b.max(t_bot) - pred_t.min(t_top);
    let giou_c = giou_cw * giou_ch;

    let dx_t = -1.0 * (pred_r - pred_l);
    let dx_b = pred_r - pred_l;
    let dx_l = -1.0 * (pred_b - pred_t);
    let dx_r = pred_b - pred_t;

    let di_t = if pred_t > t_top { -1.0 * iw } else { 0.0 };
    let di_b = if pred_b < t_bot { iw } else { 0.0 };
    let di_l = if pred_l > t_left { -1.0 * ih } else { 0.0 };
    let di_r = if pred_r < t_right { ih } else { 0.0 };
    let du_t = dx_t - di_t;
    let du_b = dx_b - di_b;
    let du_l = dx_l - di_l;
    let du_r = dx_r - di_r;

    let dc_t = if pred_t < t_top { -1.0 * giou_cw } else { 0.0 };
    let dc_b = if pred_b > t_bot { giou_cw } else { 0.0 };
    let dc_l = if pred_l < t_left { -1.0 * giou_ch } else { 0.0 };
    let dc_r = if pred_r > t_right { giou_ch } else { 0.0 };

    let (mut p_dt, mut p_db, mut p_dl, mut p_dr) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    if u > 0.0 {
        p_dt = (u * di_t - i * du_t) / (u * u);
        p_db = (u * di_b - i * du_b) / (u * u);
        p_dl = (u * di_l - i * du_l) / (u * u);
        p_dr = (u * di_r - i * du_r) / (u * u);
    }

    // Corner selection for inverted prediction boxes.  Sequential, as in C++:
    // the second assignment reads the value the first one just wrote.
    p_dt = if p_top0 < p_bot0 { p_dt } else { p_db };
    p_db = if p_top0 < p_bot0 { p_db } else { p_dt };
    p_dl = if p_left0 < p_right0 { p_dl } else { p_dr };
    p_dr = if p_left0 < p_right0 { p_dr } else { p_dl };

    // GIoU "C" term.
    if giou_c > 0.0 {
        p_dt += ((giou_c * du_t) - (u * dc_t)) / (giou_c * giou_c);
        p_db += ((giou_c * du_b) - (u * dc_b)) / (giou_c * giou_c);
        p_dl += ((giou_c * du_l) - (u * dc_l)) / (giou_c * giou_c);
        p_dr += ((giou_c * du_r) - (u * dc_r)) / (giou_c * giou_c);
    }
    if iw <= 0.0 || ih <= 0.0 {
        p_dt = ((giou_c * du_t) - (u * dc_t)) / (giou_c * giou_c);
        p_db = ((giou_c * du_b) - (u * dc_b)) / (giou_c * giou_c);
        p_dl = ((giou_c * du_l) - (u * dc_l)) / (giou_c * giou_c);
        p_dr = ((giou_c * du_r) - (u * dc_r)) / (giou_c * giou_c);
    }

    let p_dx = p_dl + p_dr;
    let p_dy = p_dt + p_db;
    let p_dw = p_dr - p_dl;
    let p_dh = p_db - p_dt;
    [p_dx, p_dy, p_dw, p_dh]
}

/// C++ `point_inside_box_grid`: the cell centre lies inside the truth box.
fn point_inside_box_grid(l: &ModernLayout, loc: usize, truth: &ModernBox) -> bool {
    let col = loc % l.w;
    let row = loc / l.w;
    let cx = (col as f32 + 0.5) / l.w as f32;
    let cy = (row as f32 + 0.5) / l.h as f32;
    cx >= truth.left() && cx <= truth.right() && cy >= truth.top() && cy <= truth.bottom()
}

/// C++ `point_inside_center_radius`, in cells.
fn point_inside_center_radius(l: &ModernLayout, loc: usize, truth: &ModernBox, radius: f32) -> bool {
    let col = loc % l.w;
    let row = loc / l.w;
    let gx = col as f32 + 0.5;
    let gy = row as f32 + 0.5;
    let tx = truth.x * l.w as f32;
    let ty = truth.y * l.h as f32;
    (gx - tx).abs() <= radius && (gy - ty).abs() <= radius
}

fn objectness(x: &[f32], l: &ModernLayout, batch: usize, n: usize, loc: usize) -> f32 {
    clamp01(x[l.obj_index(batch, n, loc)])
}

/// C++ `class_score_for`: 0 for an out-of-range class id.
fn class_score(x: &[f32], l: &ModernLayout, batch: usize, n: usize, loc: usize, class_id: usize) -> f32 {
    if class_id >= l.classes {
        return 0.0;
    }
    clamp01(x[l.class_index(batch, n, loc, class_id)])
}

/// C++ `decode_dfl_box` for the YOLO-NAS head (`decode_modern_box` without YOLOX).
fn decode_box(x: &[f32], l: &ModernLayout, batch: usize, n: usize, loc: usize) -> ModernBox {
    decode_dfl_box(x, l, batch, n, loc)
}

/// C++ `best_truth_iou_for_prediction`: best IoU of the decoded box against
/// every truth of the image (stops at the first truth with `x == 0`).
fn best_truth_iou(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    p: &NasTrainParams,
    batch: usize,
    truths: usize,
    n: usize,
    loc: usize,
) -> f32 {
    let pred = decode_box(x, l, batch, n, loc);
    let mut best = 0.0f32;
    for t in 0..truths / p.truth_size {
        let bx = truth_box(truth, p.truth_size, truths, t, batch);
        if bx.x == 0.0 {
            break;
        }
        best = best.max(clamp01(box_iou_ref(&pred, &bx)));
    }
    best
}

/// C++ `assign_tal_like` with `nms_aware = true` (the YOLO-NAS branch).
///
/// Produces the box-regression owner of each cell and the per-class quality
/// row for one image.  Errors match C++'s `darknet_fatal_error` checks.
pub fn yolonas_assign(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    p: &NasTrainParams,
    batch: usize,
) -> Result<NasAssignments, ModernYoloError> {
    if spec.kind != ModernYoloKind::Yolonas {
        return Err(ModernYoloError("yolonas_assign needs a [yolonas] spec".into()));
    }
    if p.truth_size < 5 {
        return Err(ModernYoloError(format!("truth_size {} < 5", p.truth_size)));
    }
    let truths = spec.max_boxes * p.truth_size;
    if truth.len() < truths * l.batch {
        return Err(ModernYoloError(format!(
            "anchor-free truth needs {} entries for {} images x {} boxes x {} fields, got {}",
            truths * l.batch,
            l.batch,
            spec.max_boxes,
            p.truth_size,
            truth.len()
        )));
    }

    let slots = l.n * l.spatial();
    let mut assignments = vec![NasAssignment::default(); slots];
    // C++ conflict_score (init -1) and conflict_area (init +inf) per slot.
    let mut conflict_score = vec![-1.0f32; slots];
    let mut conflict_area = vec![f32::INFINITY; slots];
    let mut class_quality = vec![0.0f32; slots * l.classes];

    for t in 0..spec.max_boxes {
        let bx = truth_box(truth, p.truth_size, truths, t, batch);
        if bx.x == 0.0 {
            break;
        }
        if !valid_truth(&bx) {
            return Err(ModernYoloError(format!(
                "invalid anchor-free truth box (x={}, y={}, w={}, h={})",
                bx.x, bx.y, bx.w, bx.h
            )));
        }
        let raw_class = truth[t * p.truth_size + batch * truths + 4];
        if raw_class < 0.0 || raw_class as usize >= l.classes {
            return Err(ModernYoloError(format!("invalid class ID #{}", raw_class as i64)));
        }
        let mut class_id = raw_class as usize;
        if let Some(map) = &p.class_map {
            class_id = map[class_id];
        }

        // C++ collect_candidates_for_truth: min_object_cells gate first.
        if p.min_object_cells > 0.0
            && ((bx.w * l.w as f32) < p.min_object_cells || (bx.h * l.h as f32) < p.min_object_cells)
        {
            continue;
        }
        let mut candidates = Vec::new();
        for n in 0..l.n {
            for loc in 0..l.spatial() {
                // TAL/NAS geometry: inside the box, or within the centre radius.
                let geometry_ok = point_inside_box_grid(l, loc, &bx)
                    || (p.center_radius > 0.0 && point_inside_center_radius(l, loc, &bx, p.center_radius));
                if !geometry_ok {
                    continue;
                }
                let pred = decode_box(x, l, batch, n, loc);
                let iou = clamp01(box_iou_ref(&pred, &bx));
                let obj = objectness(x, l, batch, n, loc);
                let cls = class_score(x, l, batch, n, loc, class_id);
                let score = if p.atss_warmup {
                    // TODO(verify): ATSS-lite warmup copied from C++ (its own TODO
                    // says to replace it with full ATSS mean+std thresholding).
                    iou
                } else {
                    let aligned_cls = (cls * obj).max(EPS).powf(p.tal_alpha);
                    let aligned_iou = iou.max(EPS).powf(p.tal_beta);
                    aligned_cls * aligned_iou
                };
                let cost = -safe_log(score + EPS);
                if score.is_finite() && cost.is_finite() {
                    candidates.push(NasCandidate {
                        anchor: n,
                        loc,
                        truth_index: t,
                        class_id,
                        iou,
                        score,
                        truth: bx,
                    });
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }
        // TODO(verify): C++ uses std::sort (unstable); ties here keep input order.
        candidates.sort_by(|lhs, rhs| rhs.score.partial_cmp(&lhs.score).unwrap_or(std::cmp::Ordering::Equal));
        let keep = p.assign_topk.min(candidates.len());
        let mut max_score = 0.0f32;
        let mut max_iou = 0.0f32;
        for c in &candidates[..keep] {
            max_score = max_score.max(c.score);
            max_iou = max_iou.max(c.iou);
        }
        let truth_area = bx.w * bx.h;
        for c in &candidates[..keep] {
            let slot = c.anchor * l.spatial() + c.loc;
            let quality = if p.atss_warmup {
                c.iou.clamp(0.05, 1.0)
            } else {
                (c.score / max_score.max(EPS) * max_iou).clamp(0.0, 1.0)
            };
            // Record this truth's class quality even if it loses the box conflict.
            let cq = &mut class_quality[slot * l.classes + c.class_id];
            *cq = cq.max(quality);

            let smaller_truth_wins = truth_area < conflict_area[slot] - EPS;
            let tied_area_better_score =
                (truth_area - conflict_area[slot]).abs() <= EPS && c.score > conflict_score[slot];
            if smaller_truth_wins || tied_area_better_score {
                conflict_area[slot] = truth_area;
                conflict_score[slot] = c.score;
                assignments[slot] = NasAssignment {
                    truth_index: Some(c.truth_index),
                    class_id: c.class_id,
                    quality,
                    iou: c.iou,
                    truth: c.truth,
                };
            }
        }
    }

    // NMS-aware duplicate decay (C++ `nas_duplicate_decay` block).
    if p.nas_duplicate_decay != 0.0 {
        let mut per_truth: Vec<Vec<usize>> = vec![Vec::new(); spec.max_boxes];
        for (slot, a) in assignments.iter().enumerate() {
            if let Some(t) = a.truth_index {
                if t < per_truth.len() {
                    per_truth[t].push(slot);
                }
            }
        }
        for list in per_truth.iter_mut() {
            // TODO(verify): C++ std::sort (unstable) on IoU; ties keep slot order here.
            list.sort_by(|&lhs, &rhs| {
                assignments[rhs].iou.partial_cmp(&assignments[lhs].iou).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (rank, &slot) in list.iter().enumerate() {
                let duplicate_weight = (-p.nas_duplicate_decay * rank as f32).exp();
                let a = &mut assignments[slot];
                a.quality = (a.quality * duplicate_weight).clamp(0.02, 1.0);
            }
        }
    }

    Ok(NasAssignments { slots: assignments, class_quality })
}

/// C++ `dfl_side_loss_value`: the two-bin cross-entropy of one side.
/// `target_ltrb_distance` (C++ of the same name) lives in `cpu.rs`.
fn dfl_side_loss(x: &[f32], l: &ModernLayout, batch: usize, n: usize, loc: usize, side: usize, target_distance: f32) -> f32 {
    let bins = l.dfl_bins();
    let max_bin = bins - 1;
    let location = n * l.spatial() + loc;
    let target = finite_or_zero(target_distance).clamp(0.0, max_bin as f32 - 1e-4);
    let left_bin = (target.floor() as usize).min(max_bin);
    let right_bin = (left_bin + 1).min(max_bin);
    let right_weight = target - left_bin as f32;
    let left_weight = 1.0 - right_weight;
    let probs = dfl_probabilities(x, l, batch, location, side);
    finite_or_zero(left_weight * -safe_log(probs[left_bin]) + right_weight * -safe_log(probs[right_bin]))
}

/// C++ `dfl_box_loss_value`: `dfl_loss_weight` times the four side losses.
fn dfl_box_loss(x: &[f32], l: &ModernLayout, p: &NasTrainParams, batch: usize, n: usize, loc: usize, truth: &ModernBox) -> f32 {
    let mut loss = 0.0f32;
    for side in 0..4 {
        loss += dfl_side_loss(x, l, batch, n, loc, side, target_ltrb_distance(l, loc, truth, side));
    }
    finite_or_zero(p.dfl_loss_weight * loss)
}

/// C++ `add_objectness_delta`.
fn add_objectness_delta(
    delta: &mut [f32],
    x: &[f32],
    l: &ModernLayout,
    p: &NasTrainParams,
    batch: usize,
    n: usize,
    loc: usize,
    target: f32,
) {
    let idx = l.obj_index(batch, n, loc);
    let d = p.obj_normalizer * (target.clamp(0.0, 1.0) - finite_or_zero(x[idx]));
    if d.is_finite() {
        delta[idx] += d;
    }
}

/// C++ `add_class_delta_multi(..., enable_vfl = true)`.  The NAS path always
/// passes `true`, so VFL is on whenever `vfl_gamma > 0`.  Background cells
/// use the all-zero row, which matches C++ `add_class_delta(..., -1, 0, true)`.
fn add_class_delta_multi(
    delta: &mut [f32],
    x: &[f32],
    l: &ModernLayout,
    p: &NasTrainParams,
    batch: usize,
    n: usize,
    loc: usize,
    row: &[f32],
) {
    let use_vfl = p.vfl_gamma > 0.0;
    for c in 0..l.classes {
        let mut target = row[c].clamp(0.0, 1.0);
        if p.label_smooth_eps != 0.0 && !use_vfl {
            target = target * (1.0 - p.label_smooth_eps) + 0.5 * p.label_smooth_eps;
        }
        let idx = l.class_index(batch, n, loc, c);
        let prob = clamp01(x[idx]);
        let mut d = p.cls_normalizer * (target - prob);
        if use_vfl {
            let weight = if target > 0.0 { target } else { VFL_ALPHA * prob.powf(p.vfl_gamma) };
            d *= weight;
        }
        if let Some(m) = &p.classes_multipliers {
            if target > 0.0 {
                d *= m[c];
            }
        }
        if d.is_finite() {
            delta[idx] += d;
        }
    }
}

/// C++ `assigned_class_loss_value` with `enable_vfl = true`.
fn assigned_class_loss(
    x: &[f32],
    l: &ModernLayout,
    p: &NasTrainParams,
    batch: usize,
    n: usize,
    loc: usize,
    class_id: usize,
    quality: f32,
) -> f32 {
    let use_vfl = p.vfl_gamma > 0.0;
    let mut target = quality.clamp(0.0, 1.0);
    if p.label_smooth_eps != 0.0 && !use_vfl {
        target = target * (1.0 - p.label_smooth_eps) + 0.5 * p.label_smooth_eps;
    }
    let prob = class_score(x, l, batch, n, loc, class_id);
    let mut loss = -target * safe_log(prob) - (1.0 - target) * safe_log(1.0 - prob);
    if use_vfl {
        loss *= target;
    }
    finite_or_zero(loss)
}

/// C++ `add_dfl_delta`: the two-bin distribution target for one side.
/// Delegates to `cpu.rs::dfl_delta`, which is the same routine.
fn add_dfl_box_delta(
    delta: &mut [f32],
    x: &[f32],
    l: &ModernLayout,
    batch: usize,
    n: usize,
    loc: usize,
    truth: &ModernBox,
    normalizer: f32,
) {
    for side in 0..4 {
        let target = target_ltrb_distance(l, loc, truth, side);
        dfl_delta(delta, x, l, batch, n, loc, side, target, normalizer);
    }
}

/// C++ `add_dfl_giou_delta_unprojected`: the GIoU gradient pushed through the
/// expectation of each side's distribution, clipped by `max_delta`.
fn add_dfl_giou_delta(
    delta: &mut [f32],
    x: &[f32],
    l: &ModernLayout,
    batch: usize,
    n: usize,
    loc: usize,
    truth: &ModernBox,
    weight: f32,
    max_delta: f32,
) {
    let bins = l.dfl_bins();
    let stride = l.spatial();
    let location = n * stride + loc;
    let pred = decode_box(x, l, batch, n, loc);
    let g = dx_box_giou(&pred, truth);
    let inv_w = 1.0 / l.w as f32;
    let inv_h = 1.0 / l.h as f32;
    // dx_box_iou() returns w/h gradients without the 0.5 edge factor, so the
    // corner gradients are 0.5 * (g.dt - g.dl) etc. (C++ comment).
    let corner_grad = [
        finite_or_zero(0.5 * (g[0] - g[2])),
        finite_or_zero(0.5 * (g[1] - g[3])),
        finite_or_zero(0.5 * (g[0] + g[2])),
        finite_or_zero(0.5 * (g[1] + g[3])),
    ];
    let expectation_scale = [-inv_w, -inv_h, inv_w, inv_h];
    for side in 0..4 {
        let probs = dfl_probabilities(x, l, batch, location, side);
        let expected = finite_or_zero(dfl_expectation(x, l, batch, location, side));
        let start = l.entry_index(batch, location, side * bins);
        let chain = finite_or_zero(weight) * corner_grad[side] * expectation_scale[side];
        for k in 0..bins {
            let mut d = chain * finite_or_zero(probs[k]) * (k as f32 - expected);
            if max_delta != FLT_MAX {
                d = d.clamp(-max_delta, max_delta);
            }
            if d.is_finite() {
                delta[start + k * stride] += d;
            }
        }
    }
}

/// One image of C++ `train_batch_modern` (the YOLO-NAS branch).
fn train_image(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    p: &NasTrainParams,
    batch: usize,
    delta: &mut [f32],
    stats: &mut NasStats,
) -> Result<(), ModernYoloError> {
    let asg = yolonas_assign(x, truth, l, spec, p, batch)?;
    let truths = spec.max_boxes * p.truth_size;
    let spatial = l.spatial();
    let zero_row = vec![0.0f32; l.classes];
    let mut row = vec![0.0f32; l.classes];

    for n in 0..l.n {
        for loc in 0..spatial {
            let slot = n * spatial + loc;
            let a = asg.slots[slot];
            match a.truth_index {
                None => {
                    if p.vfl_gamma > 0.0 {
                        add_class_delta_multi(delta, x, l, p, batch, n, loc, &zero_row);
                    }
                    let best = best_truth_iou(x, truth, l, p, batch, truths, n, loc);
                    if best > p.ignore_thresh {
                        if p.objectness_smooth {
                            add_objectness_delta(delta, x, l, p, batch, n, loc, best);
                        }
                        continue;
                    }
                    add_objectness_delta(delta, x, l, p, batch, n, loc, 0.0);
                }
                Some(_) => {
                    let class_multiplier = p.classes_multipliers.as_ref().map_or(1.0, |m| m[a.class_id]);
                    let quality = a.quality.clamp(0.0, 1.0);

                    add_objectness_delta(delta, x, l, p, batch, n, loc, quality);
                    // The box winner's own channel takes its (decayed) quality; other
                    // classes keep their own nested-candidate quality.
                    row.copy_from_slice(&asg.class_quality[slot * l.classes..(slot + 1) * l.classes]);
                    row[a.class_id] = quality;
                    add_class_delta_multi(delta, x, l, p, batch, n, loc, &row);

                    add_dfl_box_delta(
                        delta,
                        x,
                        l,
                        batch,
                        n,
                        loc,
                        &a.truth,
                        p.iou_normalizer * class_multiplier * p.dfl_loss_weight,
                    );
                    let dfl_giou_weight = p.iou_normalizer * class_multiplier * p.box_loss_weight * quality;
                    add_dfl_giou_delta(delta, x, l, batch, n, loc, &a.truth, dfl_giou_weight, p.max_delta);

                    stats.sum_iou += a.iou;
                    stats.sum_iou_loss += 1.0 - a.iou;
                    stats.cls_loss += assigned_class_loss(x, l, p, batch, n, loc, a.class_id, quality);
                    let pred = decode_box(x, l, batch, n, loc);
                    let giou = finite_or_zero(box_giou_ref(&pred, &a.truth));
                    stats.box_loss += p.box_loss_weight * quality * (1.0 - giou);
                    stats.dfl_loss += dfl_box_loss(x, l, p, batch, n, loc, &a.truth);
                    stats.count += 1;
                }
            }
        }
    }
    Ok(())
}

/// Whole-batch YOLO-NAS training delta, C++ `forward_modern_yolo_layer_impl`
/// for `kind == YOLONAS` (training branch).
///
/// `x` is the activated head output (`activate_modern_output` already applied),
/// `truth` the packed truth rows.  `delta` is zeroed, filled per cell, has
/// non-finite entries reset to 0, and its `pow(mag_array(delta), 2)` becomes
/// `stats.cost`.  The C++ `state.train` / `onlyforward` gating is the caller's job.
pub fn yolonas_train_delta(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    p: &NasTrainParams,
    delta: &mut [f32],
) -> Result<NasStats, ModernYoloError> {
    let len = l.len();
    if x.len() < len || delta.len() < len {
        return Err(ModernYoloError(format!(
            "yolonas needs {len} output and delta entries, got {} and {}",
            x.len(),
            delta.len()
        )));
    }
    let delta = &mut delta[..len];
    for v in delta.iter_mut() {
        *v = 0.0;
    }
    let mut stats = NasStats::default();
    for batch in 0..l.batch {
        train_image(x, truth, l, spec, p, batch, delta, &mut stats)?;
    }
    for v in delta.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    // C++ mag_array: float sum of squares, then sqrt; cost is its square.
    let mut sum = 0.0f32;
    for v in delta.iter() {
        sum += v * v;
    }
    let mag = sum.sqrt();
    stats.cost = mag.powf(2.0);
    Ok(stats)
}

/// C++ `get_modern_yolo_detections_raw_batch` for YOLO-NAS.  Cells are walked
/// `loc` outer, `n` inner, as in C++.  `x` must be activated.
pub fn yolonas_decode(x: &[f32], l: &ModernLayout, batch: usize, thresh: f32, score_cls_only: bool) -> Vec<NasDetection> {
    let mut out = Vec::new();
    for loc in 0..l.spatial() {
        for n in 0..l.n {
            let obj = objectness(x, l, batch, n, loc);
            let max_cls = (0..l.classes).fold(0.0f32, |m, c| m.max(class_score(x, l, batch, n, loc, c)));
            let detection_score = if score_cls_only { max_cls } else { obj };
            if detection_score <= thresh {
                continue;
            }
            let bbox = decode_box(x, l, batch, n, loc);
            let prob = (0..l.classes)
                .map(|c| {
                    let cs = class_score(x, l, batch, n, loc, c);
                    let p = if score_cls_only { cs } else { obj * cs };
                    if p > thresh { p } else { 0.0 }
                })
                .collect();
            out.push(NasDetection { bbox, objectness: detection_score, prob });
        }
    }
    out
}

/// C++ `correct_modern_yolo_boxes`: undo letterbox padding and scale to the
/// source image (`relative == false` multiplies by `w`/`h`).  Integer
/// arithmetic for the letterbox sizes matches C++.
pub fn yolonas_correct_boxes(
    dets: &mut [NasDetection],
    w: i32,
    h: i32,
    netw: i32,
    neth: i32,
    relative: bool,
    letter: bool,
) {
    let (mut new_w, mut new_h) = (netw, neth);
    if letter {
        if (netw as f32 / w as f32) < (neth as f32 / h as f32) {
            new_h = (h * netw) / w;
        } else {
            new_w = (w * neth) / h;
        }
    }
    let deltaw = (netw - new_w) as f32;
    let deltah = (neth - new_h) as f32;
    let ratiow = new_w as f32 / netw as f32;
    let ratioh = new_h as f32 / neth as f32;
    for d in dets.iter_mut() {
        let b = &mut d.bbox;
        b.x = (b.x - deltaw * 0.5 / netw as f32) / ratiow;
        b.y = (b.y - deltah * 0.5 / neth as f32) / ratioh;
        b.w *= 1.0 / ratiow;
        b.h *= 1.0 / ratioh;
        if !relative {
            b.x *= w as f32;
            b.w *= w as f32;
            b.y *= h as f32;
            b.h *= h as f32;
        }
    }
}

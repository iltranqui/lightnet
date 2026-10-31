//! `[yolox]` training loss and decode, ported from the Darknet reference
//! `raptor-fp4:src-lib/modern_yolo_layer.cpp`:
//!
//! * `assign_yolox`: SimOTA assignment (C++ `collect_candidates_for_truth` YOLOX
//!   geometry, `simota_dynamic_k`, cost-sorted take).  See [`assign_simota`].
//! * `train_batch_modern`, YOLOX branch: per-cell objectness / class / box deltas.
//!   See [`train_yolox_image`] and [`train_yolox_delta`].
//! * `add_yolox_iou_box_delta`, `add_yolox_l1_box_delta`, `add_class_delta`,
//!   `add_objectness_delta`, `assigned_class_loss_value`, `best_truth_iou_for_prediction`.
//! * `get_modern_yolo_detections_raw_batch`: decode scores, see [`decode_yolox_scores`].
//! * `raptor-fp4:src-lib/box.cpp::dx_box_iou`, IOU branch: see [`yolox_iou_grad`].
//! * `raptor-fp4:src-lib/darknet_cfg.cpp::parse_yolox_section`: the `[yolox]` keys
//!   that are not in [`ModernYoloSettings`] live in [`YoloxTrainOptions`].
//!
//! Forward, box decode (`decode_yolox_box`), `box_iou` and the shared types are in
//! `cpu.rs`.  `output` must already be activated (`activate_modern_output`).
//! Every buffer is whole-batch, indexed by `ModernLayout::entry_index`, like C++.
//!
//! Function names are prefixed to avoid clashing with `cpu.rs` (`assign_yolox`,
//! `yolox_training_delta`, `decode_yolox` already exist there).

use crate::model::{ModernYoloKind, ModernYoloSpec};

use super::{
    Assignment, EPS, FLT_MAX, ModernBox, ModernLayout, ModernYoloError, ModernYoloSettings,
    box_iou, decode_yolox_box, truth_box, truth_class,
};

/// The `[yolox]` keys that `ModernYoloSettings` does not carry.
///
/// Defaults follow C++ `parse_yolox_section` and `set_default_training_params`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YoloxTrainOptions {
    /// `assign_topk` (default 10): how many best IoUs feed the dynamic `k`.
    pub assign_topk: usize,
    /// `center_radius` (default 2.5): radius in feature cells.
    pub center_radius: f32,
    /// `box_loss_weight` (default 5.0).
    pub box_loss_weight: f32,
    /// `yolox_soft_label` (default 0): class target is the matched IoU, not `1.0`.
    pub soft_label: bool,
    /// `yolox_ignore_neg` (default 0): unassigned cells above `ignore_thresh` are skipped.
    pub ignore_neg: bool,
    /// The L1 box term is active this iteration; see [`yolox_final_l1_active`].
    pub final_l1: bool,
}

impl Default for YoloxTrainOptions {
    fn default() -> Self {
        Self {
            assign_topk: 10,
            center_radius: 2.5,
            box_loss_weight: 5.0,
            soft_label: false,
            ignore_neg: false,
            final_l1: false,
        }
    }
}

/// C++ `make_modern_yolo_train_schedule`: the L1 term runs in the last
/// `l1_final_iters` iterations.
pub fn yolox_final_l1_active(iteration: i64, max_batches: i64, l1_final_iters: i64) -> bool {
    l1_final_iters > 0 && max_batches > 0 && iteration > max_batches - l1_final_iters
}

/// Sums over one batch, as C++ `ModernYoloStats` restricted to YOLOX.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct YoloxTrainStats {
    pub sum_iou: f32,
    pub sum_iou_loss: f32,
    pub cls_loss: f32,
    pub box_loss: f32,
    /// Assigned cells.
    pub count: usize,
    /// `||delta||^2` over the whole batch (C++ `pow(mag_array(delta), 2)`).
    pub cost: f32,
}

/// One cell of a decoded `[yolox]` output, as C++ `Darknet::Detection`
/// before the image-space correction.
#[derive(Clone, Debug, PartialEq)]
pub struct YoloxScoredBox {
    pub bbox: ModernBox,
    /// `objectness`, or the max class score when `cls_only`.
    pub objectness: f32,
    /// Per-class `obj * cls` (or `cls`), zeroed at or below the threshold.
    pub probs: Vec<f32>,
}

/// C++ `finite_or_zero`.
fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// C++ `safe_log`: `log(max(value, k_eps))`.
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

/// Rejects a head that is not `[yolox]`.
fn require_yolox(l: &ModernLayout) -> Result<(), ModernYoloError> {
    if l.kind == ModernYoloKind::Yolox {
        Ok(())
    } else {
        Err(ModernYoloError(
            "yolox training called on a non-YOLOX head".to_string(),
        ))
    }
}

/// C++ `objectness_for`, clamped to `[0, 1]`.
fn objectness_at(x: &[f32], l: &ModernLayout, b: usize, n: usize, loc: usize) -> f32 {
    finite_or_zero(x[l.obj_index(b, n, loc)]).clamp(0.0, 1.0)
}

/// C++ `class_score_for`: `0` for an out-of-range class, else clamped to `[0, 1]`.
fn class_at(x: &[f32], l: &ModernLayout, b: usize, n: usize, loc: usize, class_id: usize) -> f32 {
    if class_id >= l.classes {
        return 0.0;
    }
    finite_or_zero(x[l.class_index(b, n, loc, class_id)]).clamp(0.0, 1.0)
}

/// C++ `point_inside_box_grid`: the pixel centre lies inside the truth box.
fn point_inside_box_grid(l: &ModernLayout, loc: usize, truth: &ModernBox) -> bool {
    let col = loc % l.w;
    let row = loc / l.w;
    let cx = (col as f32 + 0.5) / l.w as f32;
    let cy = (row as f32 + 0.5) / l.h as f32;
    cx >= truth.left() && cx <= truth.right() && cy >= truth.top() && cy <= truth.bottom()
}

/// C++ `point_inside_center_radius`, in feature cells.
fn point_inside_center_radius(l: &ModernLayout, loc: usize, truth: &ModernBox, radius: f32) -> bool {
    let col = loc % l.w;
    let row = loc / l.w;
    let gx = col as f32 + 0.5;
    let gy = row as f32 + 0.5;
    let tx = truth.x * l.w as f32;
    let ty = truth.y * l.h as f32;
    (gx - tx).abs() <= radius && (gy - ty).abs() <= radius
}

/// One candidate cell for one truth (C++ `Candidate`, YOLOX fields only).
#[derive(Clone, Copy, Debug)]
struct YoloxCandidate {
    anchor: usize,
    loc: usize,
    iou: f32,
    cost: f32,
}

/// C++ `collect_candidates_for_truth`, YOLOX branch.
///
/// Geometry is `inside_box || center_radius` (the reference, not the centre-only
/// test). Cost is `-log(max(sqrt(cls * obj), eps)) + 3 * -log(iou + eps)`.
fn collect_yolox_candidates(
    x: &[f32],
    l: &ModernLayout,
    opts: &YoloxTrainOptions,
    min_object_cells: f32,
    b: usize,
    class_id: usize,
    truth: &ModernBox,
) -> Vec<YoloxCandidate> {
    let mut out = Vec::new();
    // C++ `min_object_cells` gate: 0 disables it (the default).
    if min_object_cells > 0.0
        && ((truth.w * l.w as f32) < min_object_cells || (truth.h * l.h as f32) < min_object_cells)
    {
        return out;
    }
    for n in 0..l.n {
        for loc in 0..l.spatial() {
            let geometry_ok = point_inside_box_grid(l, loc, truth)
                || point_inside_center_radius(l, loc, truth, opts.center_radius);
            if !geometry_ok {
                continue;
            }
            let pred = decode_yolox_box(x, l, b, n, loc);
            let iou = finite_or_zero(box_iou(&pred, truth)).clamp(0.0, 1.0);
            let obj = objectness_at(x, l, b, n, loc);
            let cls = class_at(x, l, b, n, loc, class_id);
            let cls_obj = (cls * obj).max(EPS).sqrt();
            let cls_loss = -safe_log(cls_obj);
            let iou_loss = -safe_log(iou + EPS);
            let cost = cls_loss + 3.0 * iou_loss;
            let score = 1.0 / (1.0 + cost);
            if score.is_finite() && cost.is_finite() {
                out.push(YoloxCandidate {
                    anchor: n,
                    loc,
                    iou,
                    cost,
                });
            }
        }
    }
    out
}

/// C++ `simota_dynamic_k`: `max(1, int(sum of the top assign_topk IoUs))`.
fn simota_dynamic_k(candidates: &[YoloxCandidate], topk: usize) -> usize {
    if candidates.is_empty() {
        return 0;
    }
    let mut ious: Vec<f32> = candidates.iter().map(|c| c.iou).collect();
    ious.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let k = topk.min(ious.len());
    let sum: f32 = ious[..k].iter().sum();
    (sum as usize).max(1)
}

/// C++ `assign_yolox`: SimOTA over one image's cells.
///
/// Returns one [`Assignment`] per cell (`n * H * W`), with `truth_index == usize::MAX`
/// for unassigned cells. `quality` ends as the hard `1.0` unless `soft_label`, then
/// the clamped IoU.
pub fn assign_simota(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    opts: &YoloxTrainOptions,
    b: usize,
) -> Result<Vec<Assignment>, ModernYoloError> {
    require_yolox(l)?;
    let truths = spec.max_boxes * settings.truth_size;
    let mut assignments = vec![
        Assignment {
            truth_index: usize::MAX,
            class_id: usize::MAX,
            quality: 0.0,
            iou: 0.0,
            truth: ModernBox::default(),
        };
        l.n * l.spatial()
    ];
    for t in 0..spec.max_boxes {
        if truth.len() < truths * l.batch {
            return Err(ModernYoloError(format!(
                "yolox truth needs {} entries, got {}",
                truths * l.batch,
                truth.len()
            )));
        }
        // C++ `float_to_box_stride` + `if (!truth.x) break;`
        let box_ = truth_box(truth, settings.truth_size, truths, t, b);
        if box_.x == 0.0 {
            break;
        }
        // C++ `invalid YOLOX truth box` fatal.
        if !valid_truth(&box_) {
            return Err(ModernYoloError(format!(
                "invalid YOLOX truth box (x={}, y={}, w={}, h={})",
                box_.x, box_.y, box_.w, box_.h
            )));
        }
        let mut class_id = truth_class(truth, settings.truth_size, truths, t, b);
        // C++ `validate_class_or_continue` (fatal on bad id).
        if class_id >= l.classes {
            return Err(ModernYoloError(format!("invalid class ID #{class_id}")));
        }
        if let Some(map) = &settings.class_map {
            class_id = map[class_id];
        }

        let mut candidates = collect_yolox_candidates(
            x,
            l,
            opts,
            spec.min_object_cells,
            b,
            class_id,
            &box_,
        );
        if candidates.is_empty() {
            continue;
        }
        let dynamic_k = simota_dynamic_k(&candidates, opts.assign_topk).min(candidates.len());
        // TODO(verify): C++ uses std::sort (unstable); tie order between equal costs may differ.
        candidates.sort_by(|a, c| a.cost.partial_cmp(&c.cost).unwrap_or(std::cmp::Ordering::Equal));

        for c in candidates.iter().take(dynamic_k) {
            let slot = c.anchor * l.spatial() + c.loc;
            let a = &mut assignments[slot];
            // C++: replace = truth_index < 0 || c.cost < -a.quality, where quality holds -cost.
            if a.truth_index == usize::MAX || c.cost < -a.quality {
                *a = Assignment {
                    truth_index: t,
                    class_id,
                    quality: -c.cost,
                    iou: c.iou,
                    truth: box_,
                };
            }
        }
    }
    for a in assignments.iter_mut() {
        if a.truth_index != usize::MAX {
            a.quality = if opts.soft_label {
                a.iou.clamp(0.0, 1.0)
            } else {
                1.0
            };
        }
    }
    Ok(assignments)
}

/// Packs one image's assignments for the GPU kernel: four floats per cell,
/// `[truth_index or -1, class_id, quality, iou]`.
pub fn pack_simota_assignments(assignments: &[Assignment]) -> Vec<f32> {
    let mut packed = Vec::with_capacity(assignments.len() * 4);
    for a in assignments {
        if a.truth_index == usize::MAX {
            packed.extend_from_slice(&[-1.0, 0.0, 0.0, 0.0]);
        } else {
            packed.extend_from_slice(&[a.truth_index as f32, a.class_id as f32, a.quality, a.iou]);
        }
    }
    packed
}

/// C++ `best_truth_iou_for_prediction`: the best IoU of this cell's prediction
/// against any truth box of the image.
fn best_truth_iou(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    b: usize,
    n: usize,
    loc: usize,
) -> f32 {
    let pred = decode_yolox_box(x, l, b, n, loc);
    let truths = spec.max_boxes * settings.truth_size;
    let mut best = 0.0f32;
    for t in 0..spec.max_boxes {
        let box_ = truth_box(truth, settings.truth_size, truths, t, b);
        if box_.x == 0.0 {
            break;
        }
        best = best.max(finite_or_zero(box_iou(&pred, &box_)).clamp(0.0, 1.0));
    }
    best
}

/// Gradient of IoU with respect to the four edge terms of `pred`, C++
/// `dx_box_iou(pred, truth, IOU)`.
///
/// Returns `(dt, db, dl, dr)` in the reference packing, i.e. `dt = dIoU/dx`,
/// `db = dIoU/dy`, `dl = dIoU/dw` (no 0.5 factor), `dr = dIoU/dh`.
pub fn yolox_iou_grad(pred: &ModernBox, truth: &ModernBox) -> (f32, f32, f32, f32) {
    let (pl0, pr0, pt0, pb0) = (pred.left(), pred.right(), pred.top(), pred.bottom());
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

    let (mut p_t, mut p_b, mut p_l, mut p_r) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    if union > 0.0 {
        let u2 = union * union;
        p_t = (union * di_t - inter * du_t) / u2;
        p_b = (union * di_b - inter * du_b) / u2;
        p_l = (union * di_l - inter * du_l) / u2;
        p_r = (union * di_r - inter * du_r) / u2;
    }
    // C++ sequential corner swap, kept literal: `p_db = c ? p_db : p_dt` reads the
    // already-updated `p_dt`, so the else branch yields `(p_b, p_b)` and `(p_r, p_r)`.
    // Unreachable for sanitized boxes with positive size.
    // TODO(verify): confirm the else branch is never hit for YOLOX preds.
    let (p_t, p_b) = if pt0 < pb0 { (p_t, p_b) } else { (p_b, p_b) };
    let (p_l, p_r) = if pl0 < pr0 { (p_l, p_r) } else { (p_r, p_r) };

    (p_l + p_r, p_t + p_b, p_r - p_l, p_b - p_t)
}

/// C++ `add_yolox_iou_box_delta`: IoU box loss, `box_loss_weight * (1 - iou^2)`,
/// gradient on the raw cxcywh outputs.
fn add_iou_box_delta(
    x: &[f32],
    l: &ModernLayout,
    b: usize,
    n: usize,
    loc: usize,
    truth: &ModernBox,
    normalizer: f32,
    box_loss_weight: f32,
    max_delta: f32,
    delta: &mut [f32],
) {
    let stride = l.spatial();
    let index = l.box_index(b, n, loc);
    let pred = decode_yolox_box(x, l, b, n, loc);
    let iou = finite_or_zero(box_iou(&pred, truth)).clamp(0.0, 1.0);
    let (dt, db, dl, dr) = yolox_iou_grad(&pred, truth);
    // IOU branch: factor = 2 * iou * box_loss_weight.
    let total = normalizer * (2.0 * iou * box_loss_weight);
    let inv_w = 1.0 / l.w as f32;
    let inv_h = 1.0 / l.h as f32;
    let deltas = [
        total * dt * inv_w,
        total * db * inv_h,
        total * dl * pred.w,
        total * dr * pred.h,
    ];
    for (c, d) in deltas.iter().enumerate() {
        add_clipped(delta, index + c * stride, *d, max_delta);
    }
}

/// C++ `add_yolox_l1_box_delta`: L1 on the raw `(tx, ty, log tw, log th)` targets.
fn add_l1_box_delta(
    x: &[f32],
    l: &ModernLayout,
    b: usize,
    n: usize,
    loc: usize,
    truth: &ModernBox,
    normalizer: f32,
    max_delta: f32,
    delta: &mut [f32],
) {
    let col = loc % l.w;
    let row = loc / l.w;
    let stride = l.spatial();
    let index = l.box_index(b, n, loc);
    let targets = [
        truth.x * l.w as f32 - col as f32,
        truth.y * l.h as f32 - row as f32,
        safe_log(truth.w * l.w as f32),
        safe_log(truth.h * l.h as f32),
    ];
    for (c, target) in targets.iter().enumerate() {
        let d = normalizer * (target - finite_or_zero(x[index + c * stride]));
        add_clipped(delta, index + c * stride, d, max_delta);
    }
}

/// C++ `add_objectness_delta`: `obj_normalizer * (clamp(target) - output)`.
fn add_objectness(
    x: &[f32],
    l: &ModernLayout,
    b: usize,
    n: usize,
    loc: usize,
    normalizer: f32,
    target: f32,
    delta: &mut [f32],
) {
    let idx = l.obj_index(b, n, loc);
    let d = normalizer * (target.clamp(0.0, 1.0) - finite_or_zero(x[idx]));
    if d.is_finite() {
        delta[idx] += d;
    }
}

/// C++ `add_*_delta` clip: `max_delta` clamps only when it is not `FLT_MAX`.
/// Non-finite values are dropped, as in C++.
fn add_clipped(delta: &mut [f32], index: usize, value: f32, max_delta: f32) {
    if !value.is_finite() {
        return;
    }
    let d = if max_delta != FLT_MAX {
        value.clamp(-max_delta, max_delta)
    } else {
        value
    };
    if d.is_finite() {
        delta[index] += d;
    }
}

/// C++ `assigned_class_loss_value` with `enable_vfl = false`: binary cross-entropy
/// of the assigned class against `quality` (label-smoothed when `label_smooth_eps != 0`).
fn assigned_class_loss(
    x: &[f32],
    l: &ModernLayout,
    b: usize,
    n: usize,
    loc: usize,
    class_id: usize,
    quality: f32,
    label_smooth_eps: f32,
) -> f32 {
    let mut target = quality.clamp(0.0, 1.0);
    if label_smooth_eps != 0.0 {
        target = target * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps;
    }
    let p = class_at(x, l, b, n, loc, class_id);
    let loss = -target * safe_log(p) - (1.0 - target) * safe_log(1.0 - p);
    finite_or_zero(loss)
}

/// C++ `train_batch_modern`, YOLOX branch, for one image `b`.
///
/// Adds this image's deltas into `delta` (whole-batch layout) and its sums into
/// `stats`. Per cell, in C++ order:
///
/// * unassigned: if `ignore_neg` and the best IoU is above `ignore_thresh`, the
///   objectness target is the IoU when `objectness_smooth` and the cell is skipped;
///   otherwise the target is `0`. No class or box delta.
/// * assigned: objectness target `1`; class target `quality` for the assigned
///   class and `0` for the rest, times `cls_normalizer` (times the class multiplier
///   on the assigned channel); IoU box delta with `iou_normalizer * class_multiplier`;
///   L1 delta when `final_l1`.
pub fn train_yolox_image(
    x: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    opts: &YoloxTrainOptions,
    b: usize,
    delta: &mut [f32],
    stats: &mut YoloxTrainStats,
) -> Result<(), ModernYoloError> {
    require_yolox(l)?;
    let assignments = assign_simota(x, truth, l, spec, settings, opts, b)?;
    let stride = l.spatial();
    for n in 0..l.n {
        for loc in 0..stride {
            let slot = n * stride + loc;
            let a = &assignments[slot];

            if a.truth_index == usize::MAX {
                if opts.ignore_neg {
                    let best = best_truth_iou(x, truth, l, spec, settings, b, n, loc);
                    if best > settings.ignore_thresh {
                        if settings.objectness_smooth {
                            add_objectness(x, l, b, n, loc, settings.obj_normalizer, best, delta);
                        }
                        continue;
                    }
                }
                add_objectness(x, l, b, n, loc, settings.obj_normalizer, 0.0, delta);
                continue;
            }

            let class_multiplier = settings
                .classes_multipliers
                .as_ref()
                .map(|m| m[a.class_id])
                .unwrap_or(1.0);
            let quality = a.quality.clamp(0.0, 1.0);

            add_objectness(x, l, b, n, loc, settings.obj_normalizer, 1.0, delta);

            // Class targets: one-hot on the assigned class (C++ `add_class_delta`, enable_vfl = false).
            for c in 0..l.classes {
                let mut target = if c == a.class_id { quality } else { 0.0 };
                if settings.label_smooth_eps != 0.0 {
                    target = target * (1.0 - settings.label_smooth_eps)
                        + 0.5 * settings.label_smooth_eps;
                }
                let idx = l.class_index(b, n, loc, c);
                let p = finite_or_zero(x[idx]).clamp(0.0, 1.0);
                let mut d = settings.cls_normalizer * (target - p);
                if c == a.class_id {
                    if let Some(m) = &settings.classes_multipliers {
                        d *= m[a.class_id];
                    }
                }
                if d.is_finite() {
                    delta[idx] += d;
                }
            }

            let normalizer = settings.iou_normalizer * class_multiplier;
            add_iou_box_delta(
                x,
                l,
                b,
                n,
                loc,
                &a.truth,
                normalizer,
                opts.box_loss_weight,
                settings.max_delta,
                delta,
            );
            if opts.final_l1 {
                add_l1_box_delta(x, l, b, n, loc, &a.truth, normalizer, settings.max_delta, delta);
            }

            stats.sum_iou += a.iou;
            stats.sum_iou_loss += 1.0 - a.iou;
            stats.cls_loss += assigned_class_loss(
                x,
                l,
                b,
                n,
                loc,
                a.class_id,
                quality,
                settings.label_smooth_eps,
            );
            stats.box_loss += opts.box_loss_weight * (1.0 - a.iou * a.iou);
            stats.count += 1;
        }
    }
    Ok(())
}

/// C++ `train_batch_modern` for YOLOX over the whole batch.
///
/// Returns the whole-batch delta (same layout as `output`) and the summed
/// stats. `cost` is `||delta||^2`, as C++ reports it. `output` must be activated.
pub fn train_yolox_delta(
    output: &[f32],
    truth: &[f32],
    l: &ModernLayout,
    spec: &ModernYoloSpec,
    settings: &ModernYoloSettings,
    opts: &YoloxTrainOptions,
) -> Result<(Vec<f32>, YoloxTrainStats), ModernYoloError> {
    require_yolox(l)?;
    let mut delta = vec![0.0f32; l.len()];
    let mut stats = YoloxTrainStats::default();
    for b in 0..l.batch {
        train_yolox_image(output, truth, l, spec, settings, opts, b, &mut delta, &mut stats)?;
    }
    for d in delta.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
    }
    stats.cost = delta.iter().map(|d| d * d).sum::<f32>();
    Ok((delta, stats))
}

/// C++ `get_modern_yolo_detections_raw_batch` for YOLOX (`score_mode` is passed
/// as `cls_only`; `score_mode == 1` means the max class score).
///
/// Cells are visited in C++ order (location outer, anchor inner). A cell is kept
/// when its detection score is above `thresh`. The box is `decode_yolox_box`. The
/// image-space correction (`correct_modern_yolo_boxes`) is the caller's job.
pub fn decode_yolox_scores(
    output: &[f32],
    l: &ModernLayout,
    b: usize,
    thresh: f32,
    cls_only: bool,
) -> Vec<YoloxScoredBox> {
    let mut out = Vec::new();
    for loc in 0..l.spatial() {
        for n in 0..l.n {
            let objectness = objectness_at(output, l, b, n, loc);
            let max_cls = (0..l.classes)
                .map(|c| class_at(output, l, b, n, loc, c))
                .fold(0.0f32, f32::max);
            let detection_score = if cls_only { max_cls } else { objectness };
            if detection_score <= thresh {
                continue;
            }
            let probs = (0..l.classes)
                .map(|c| {
                    let cls = class_at(output, l, b, n, loc, c);
                    let prob = if cls_only { cls } else { objectness * cls };
                    if prob > thresh { prob } else { 0.0 }
                })
                .collect();
            out.push(YoloxScoredBox {
                bbox: decode_yolox_box(output, l, b, n, loc),
                objectness: detection_score,
                probs,
            });
        }
    }
    out
}

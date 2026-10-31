//! PP-YOLOE per-cell training delta kernel. Port of the PPYOLOE branch of
//! raptor-fp4 `src-lib/modern_yolo_layer.cpp::train_batch_modern`, with the
//! Task-Aligned assignment already applied on the host. The CPU reference is
//! `src/lightnet-core/src/layers/modern_yolo/ppyoloe_train.rs` (`Head::train_image`).
//!
//! Packing (all `f32` unless noted):
//! * `d: &[u32]`: `[width, height, anchors, classes, coords, max_boxes, truth_size,
//!   truths, outputs, batches, objectness_smooth, vfl_on]`. `coords = 4 * (reg_max + 1)`,
//!   `bins = max(2, coords / 4)`, `truths = max_boxes * truth_size`, `outputs` per image.
//! * `p: &[f32]`: `[obj_normalizer, cls_normalizer, iou_normalizer, label_smooth_eps,
//!   ignore_thresh, max_delta (f32::MAX = off), box_loss_weight, dfl_loss_weight, vfl_gamma]`.
//! * `assign: &[f32]`, 8 per cell, `cell = batch * slots + anchor * spatial + loc`:
//!   `[assigned (1 or 0), quality, class_id, truth x, truth y, truth w, truth h,
//!   class_multiplier]`.
//! * `class_targets: &[f32]`, `classes` per cell. The host already wrote the winner's
//!   quality into its own channel, as C++ `class_quality_row[a.class_id] = quality`.
//! * `class_mult: &[f32]`, `classes` entries (1.0 when the cfg has no multipliers).
//! * `truth: &[f32]`, the raw detection truth layout `batch * truths + t * truth_size + field`.
//!
//! `delta` must be zero on entry (the host clears it, as C++ memsets `l.delta`).
//! One thread owns one cell and writes only that cell's entries, so no two
//! threads touch the same slot.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

type PBox = (f32, f32, f32, f32);

const EPS: f32 = 1.0e-8;
const VFL_ALPHA: f32 = 0.75;

#[device]
fn ppyoloe_finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// C++ `dfl_bins`: `max(2, coords / 4)`.
#[device]
fn ppyoloe_bins(d: &[u32]) -> usize {
    let bins = (d[4] / 4) as usize;
    if bins < 2 { 2 } else { bins }
}

/// C++ `modern_entry_index`: flat index of `entry` of cell `location`
/// (`location = anchor * spatial + loc`) for image `batch`.
#[device]
fn ppyoloe_entry(d: &[u32], batch: usize, location: usize, entry: usize) -> usize {
    let spatial = (d[0] * d[1]) as usize;
    let n = location / spatial;
    let loc = location % spatial;
    batch * d[8] as usize + n * spatial * (d[4] as usize + d[3] as usize + 1) + entry * spatial + loc
}

/// One DFL logit, sanitised as C++ `dfl_probabilities` reads it.
#[device]
fn ppyoloe_logit(output: &[f32], d: &[u32], batch: usize, location: usize, side: usize, k: usize) -> f32 {
    ppyoloe_finite(output[ppyoloe_entry(d, batch, location, side * ppyoloe_bins(d) + k)])
}

/// Max logit and softmax denominator of one side (C++ `dfl_probabilities`).
#[device]
fn ppyoloe_side_softmax(output: &[f32], d: &[u32], batch: usize, location: usize, side: usize) -> (f32, f32) {
    let bins = ppyoloe_bins(d);
    let mut max_logit = f32::NEG_INFINITY;
    let mut k = 0;
    while k < bins {
        max_logit = max_logit.max(ppyoloe_logit(output, d, batch, location, side, k));
        k += 1;
    }
    let mut denom = 0.0f32;
    k = 0;
    while k < bins {
        denom += (ppyoloe_logit(output, d, batch, location, side, k) - max_logit)
            .clamp(-60.0, 60.0)
            .exp();
        k += 1;
    }
    (max_logit, denom)
}

/// Softmax probability of bin `k` (C++ `dfl_probabilities`, denominator floored at eps).
#[device]
fn ppyoloe_prob(
    output: &[f32], d: &[u32], batch: usize, location: usize, side: usize, k: usize,
    max_logit: f32, denom: f32,
) -> f32 {
    let e = (ppyoloe_logit(output, d, batch, location, side, k) - max_logit)
        .clamp(-60.0, 60.0)
        .exp();
    e / denom.max(EPS)
}

/// C++ `dfl_expectation`: softmax-weighted bin index of one side.
#[device]
fn ppyoloe_expectation(output: &[f32], d: &[u32], batch: usize, location: usize, side: usize) -> f32 {
    let bins = ppyoloe_bins(d);
    let (max_logit, denom) = ppyoloe_side_softmax(output, d, batch, location, side);
    let mut expected = 0.0f32;
    let mut k = 0;
    while k < bins {
        let e = (ppyoloe_logit(output, d, batch, location, side, k) - max_logit)
            .clamp(-60.0, 60.0)
            .exp();
        expected += k as f32 * e;
        k += 1;
    }
    expected / denom.max(EPS)
}

/// C++ `decode_dfl_box` (through `box_from_corners`): normalised cxcywh.
#[device]
fn ppyoloe_decode(output: &[f32], d: &[u32], batch: usize, slot: usize) -> PBox {
    let spatial = (d[0] * d[1]) as usize;
    let w = d[0] as usize;
    let loc = slot % spatial;
    let cx = (loc % w) as f32 + 0.5;
    let cy = (loc / w) as f32 + 0.5;
    let left = ppyoloe_expectation(output, d, batch, slot, 0);
    let top = ppyoloe_expectation(output, d, batch, slot, 1);
    let right = ppyoloe_expectation(output, d, batch, slot, 2);
    let bottom = ppyoloe_expectation(output, d, batch, slot, 3);
    let (fw, fh) = (d[0] as f32, d[1] as f32);
    let mut x0 = ppyoloe_finite((cx - left) / fw);
    let mut y0 = ppyoloe_finite((cy - top) / fh);
    let mut x1 = ppyoloe_finite((cx + right) / fw);
    let mut y1 = ppyoloe_finite((cy + bottom) / fh);
    if x1 < x0 {
        let tmp = x0;
        x0 = x1;
        x1 = tmp;
    }
    if y1 < y0 {
        let tmp = y0;
        y0 = y1;
        y1 = tmp;
    }
    let bw = (x1 - x0).max(1.0e-6);
    let bh = (y1 - y0).max(1.0e-6);
    (ppyoloe_finite(0.5 * (x0 + x1)), ppyoloe_finite(0.5 * (y0 + y1)), bw, bh)
}

/// C++ `box_iou` (box.cpp): intersection over union of two cxcywh boxes.
#[device]
fn ppyoloe_iou(a: PBox, b: PBox) -> f32 {
    let width = (a.0 + a.2 * 0.5).min(b.0 + b.2 * 0.5) - (a.0 - a.2 * 0.5).max(b.0 - b.2 * 0.5);
    let height = (a.1 + a.3 * 0.5).min(b.1 + b.3 * 0.5) - (a.1 - a.3 * 0.5).max(b.1 - b.3 * 0.5);
    if width <= 0.0 || height <= 0.0 {
        return 0.0;
    }
    let inter = width * height;
    inter / (a.2 * a.3 + b.2 * b.3 - inter)
}

/// Port of `dx_box_iou(pred, truth, GIOU)` (box.cpp, GIOU branch). Returns
/// `(dt, db, dl, dr)` with the same convention as the C++ `dxrep`.
#[device]
fn ppyoloe_giou_grad(pred: PBox, truth: PBox) -> (f32, f32, f32, f32) {
    let pt0 = pred.1 - pred.3 * 0.5;
    let pb0 = pred.1 + pred.3 * 0.5;
    let pl0 = pred.0 - pred.2 * 0.5;
    let pr0 = pred.0 + pred.2 * 0.5;
    let pred_t = pt0.min(pb0);
    let pred_b = pt0.max(pb0);
    let pred_l = pl0.min(pr0);
    let pred_r = pl0.max(pr0);
    let tt = truth.1 - truth.3 * 0.5;
    let tb = truth.1 + truth.3 * 0.5;
    let tl = truth.0 - truth.2 * 0.5;
    let tr = truth.0 + truth.2 * 0.5;

    let x_area = (pred_b - pred_t) * (pred_r - pred_l);
    let xhat = (tb - tt) * (tr - tl);
    let ih = pred_b.min(tb) - pred_t.max(tt);
    let iw = pred_r.min(tr) - pred_l.max(tl);
    let inter = iw * ih;
    let union_area = x_area + xhat - inter;
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

    let mut p_dt = 0.0f32;
    let mut p_db = 0.0f32;
    let mut p_dl = 0.0f32;
    let mut p_dr = 0.0f32;
    if union_area > 0.0 {
        let u2 = union_area * union_area;
        p_dt = (union_area * di_t - inter * du_t) / u2;
        p_db = (union_area * di_b - inter * du_b) / u2;
        p_dl = (union_area * di_l - inter * du_l) / u2;
        p_dr = (union_area * di_r - inter * du_r) / u2;
    }
    // Literal port of the C++ corner selection (`h > 0` and `w > 0` always hold here).
    let (p_dt, p_db) = if pt0 < pb0 { (p_dt, p_db) } else { (p_db, p_db) };
    let (p_dl, p_dr) = if pl0 < pr0 { (p_dl, p_dr) } else { (p_dr, p_dr) };

    if c_area > 0.0 {
        let c2 = c_area * c_area;
        p_dt += (c_area * du_t - union_area * dc_t) / c2;
        p_db += (c_area * du_b - union_area * dc_b) / c2;
        p_dl += (c_area * du_l - union_area * dc_l) / c2;
        p_dr += (c_area * du_r - union_area * dc_r) / c2;
    }
    if iw <= 0.0 || ih <= 0.0 {
        let c2 = c_area * c_area;
        p_dt = (c_area * du_t - union_area * dc_t) / c2;
        p_db = (c_area * du_b - union_area * dc_b) / c2;
        p_dl = (c_area * du_l - union_area * dc_l) / c2;
        p_dr = (c_area * du_r - union_area * dc_r) / c2;
    }

    (p_dl + p_dr, p_dt + p_db, p_dr - p_dl, p_db - p_dt)
}

/// C++ `target_ltrb_distance`: the truth's distance to this cell on one side,
/// in feature cells (`side`: 0 left, 1 top, 2 right, 3 bottom).
#[device]
fn ppyoloe_target_ltrb(loc: usize, w: usize, fw: f32, fh: f32, truth: PBox, side: usize) -> f32 {
    let cx = (loc % w) as f32 + 0.5;
    let cy = (loc / w) as f32 + 0.5;
    let left = (truth.0 - truth.2 * 0.5) * fw;
    let top = (truth.1 - truth.3 * 0.5) * fh;
    let right = (truth.0 + truth.2 * 0.5) * fw;
    let bottom = (truth.1 + truth.3 * 0.5) * fh;
    if side == 0 {
        cx - left
    } else if side == 1 {
        cy - top
    } else if side == 2 {
        right - cx
    } else {
        bottom - cy
    }
}

/// C++ `add_objectness_delta`.
#[device]
fn ppyoloe_add_objectness(delta: *mut f32, output: &[f32], d: &[u32], p: &[f32], batch: usize, slot: usize, target: f32) {
    let idx = ppyoloe_entry(d, batch, slot, d[4] as usize);
    let value = p[0] * (target.clamp(0.0, 1.0) - ppyoloe_finite(output[idx]));
    if value.is_finite() {
        // SAFETY: this thread owns the cell's objectness entry.
        unsafe {
            *delta.add(idx) += value;
        }
    }
}

/// C++ `add_class_delta_multi` (`enable_vfl = true`). `use_targets == false`
/// is the background path, where every target is zero (C++ `add_class_delta(-1)`).
#[device]
fn ppyoloe_add_classes(
    delta: *mut f32, output: &[f32], d: &[u32], p: &[f32], class_mult: &[f32],
    batch: usize, slot: usize, targets: &[f32], row: usize, use_targets: bool,
) {
    let classes = d[3] as usize;
    let use_vfl = d[11] != 0;
    let eps = p[3];
    let mut c = 0;
    while c < classes {
        let mut target = if use_targets { targets[row + c].clamp(0.0, 1.0) } else { 0.0 };
        if eps != 0.0 && !use_vfl {
            target = target * (1.0 - eps) + 0.5 * eps;
        }
        let idx = ppyoloe_entry(d, batch, slot, d[4] as usize + 1 + c);
        let prob = ppyoloe_finite(output[idx]).clamp(0.0, 1.0);
        let mut value = p[1] * (target - prob);
        if use_vfl {
            // TODO(verify): C++ uses std::pow(p, gamma); written as exp(gamma * ln p)
            // with the p = 0 case set to 0 (same value for gamma > 0).
            let neg = if prob > 0.0 { VFL_ALPHA * (p[8] * prob.ln()).exp() } else { 0.0 };
            let weight = if target > 0.0 { target } else { neg };
            value *= weight;
        }
        if target > 0.0 {
            value *= class_mult[c];
        }
        if value.is_finite() {
            // SAFETY: this thread owns the cell's class entries.
            unsafe {
                *delta.add(idx) += value;
            }
        }
        c += 1;
    }
}

/// C++ `add_dfl_delta`: two-bin target around `target` for one side.
#[device]
fn ppyoloe_add_dfl_side(
    delta: *mut f32, output: &[f32], d: &[u32], batch: usize, slot: usize,
    side: usize, target: f32, normalizer: f32,
) {
    let bins = ppyoloe_bins(d);
    let max_bin = bins - 1;
    let stride = (d[0] * d[1]) as usize;
    let start = ppyoloe_entry(d, batch, slot, side * bins);
    let td = ppyoloe_finite(target).clamp(0.0, max_bin as f32 - 1.0e-4);
    let left = (td as usize).min(max_bin);
    let right = (left + 1).min(max_bin);
    let right_weight = td - left as f32;
    let left_weight = 1.0 - right_weight;
    let (max_logit, denom) = ppyoloe_side_softmax(output, d, batch, slot, side);
    let mut k = 0;
    while k < bins {
        let mut t = 0.0f32;
        if k == left {
            t += left_weight;
        }
        if k == right {
            t += right_weight;
        }
        let prob = ppyoloe_prob(output, d, batch, slot, side, k, max_logit, denom);
        let value = normalizer * (t - prob);
        if value.is_finite() {
            // SAFETY: this thread owns the cell's four distribution blocks.
            unsafe {
                *delta.add(start + k * stride) += value;
            }
        }
        k += 1;
    }
}

/// C++ `add_dfl_giou_delta_unprojected`: pushes `weight * dIoU/d(corner)` through
/// the DFL expectation into each bin logit, optionally clipped by `max_delta`.
#[device]
fn ppyoloe_add_giou(
    delta: *mut f32, output: &[f32], d: &[u32], p: &[f32], batch: usize, slot: usize,
    truth: PBox, weight: f32,
) {
    let bins = ppyoloe_bins(d);
    let stride = (d[0] * d[1]) as usize;
    let pred = ppyoloe_decode(output, d, batch, slot);
    let g = ppyoloe_giou_grad(pred, truth);
    let inv_w = 1.0 / d[0] as f32;
    let inv_h = 1.0 / d[1] as f32;
    let corner = [
        ppyoloe_finite(0.5 * (g.0 - g.2)),
        ppyoloe_finite(0.5 * (g.1 - g.3)),
        ppyoloe_finite(0.5 * (g.0 + g.2)),
        ppyoloe_finite(0.5 * (g.1 + g.3)),
    ];
    let scale = [-inv_w, -inv_h, inv_w, inv_h];
    let max_delta = p[5];
    let mut side = 0;
    while side < 4 {
        let expected = ppyoloe_finite(ppyoloe_expectation(output, d, batch, slot, side));
        let (max_logit, denom) = ppyoloe_side_softmax(output, d, batch, slot, side);
        let start = ppyoloe_entry(d, batch, slot, side * bins);
        let chain = ppyoloe_finite(weight) * corner[side] * scale[side];
        let mut k = 0;
        while k < bins {
            let prob = ppyoloe_finite(ppyoloe_prob(output, d, batch, slot, side, k, max_logit, denom));
            let mut value = chain * prob * (k as f32 - expected);
            if max_delta != f32::MAX {
                value = value.clamp(-max_delta, max_delta);
            }
            if value.is_finite() {
                // SAFETY: this thread owns the cell's distribution blocks.
                unsafe {
                    *delta.add(start + k * stride) += value;
                }
            }
            k += 1;
        }
        side += 1;
    }
}

/// C++ `best_truth_iou_for_prediction`: best IoU of the decoded box against the
/// image's truths, stopping at the first row with `x == 0`.
#[device]
fn ppyoloe_best_truth_iou(output: &[f32], truth: &[f32], d: &[u32], batch: usize, slot: usize) -> f32 {
    let pred = ppyoloe_decode(output, d, batch, slot);
    let size = d[6] as usize;
    let base = batch * d[7] as usize;
    let max_boxes = d[5] as usize;
    let mut best = 0.0f32;
    let mut t = 0usize;
    while t < max_boxes {
        let at = base + t * size;
        if truth[at] == 0.0 {
            break;
        }
        let tb = (truth[at], truth[at + 1], truth[at + 2], truth[at + 3]);
        best = best.max(ppyoloe_finite(ppyoloe_iou(pred, tb)).clamp(0.0, 1.0));
        t += 1;
    }
    best
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// PP-YOLOE loss deltas, one thread per (image, anchor, location) cell.
    /// Writes the objectness, class, distribution and GIoU deltas of that cell,
    /// or the background objectness and VFL negatives when the cell is unassigned.
    #[kernel]
    pub fn ppyoloe_loss_delta(
        output: &[f32],
        truth: &[f32],
        assign: &[f32],
        class_targets: &[f32],
        class_mult: &[f32],
        mut delta: DisjointSlice<f32>,
        d: &[u32],
        p: &[f32],
    ) {
        let g = thread::index_1d().get();
        let spatial = (d[0] * d[1]) as usize;
        let slots = d[2] as usize * spatial;
        let batches = d[9] as usize;
        if g >= batches * slots {
            return;
        }
        let batch = g / slots;
        let slot = g % slots;
        let classes = d[3] as usize;
        let row = g * classes;
        let del = delta.as_mut_ptr();
        let rec = g * 8;

        if assign[rec] == 0.0 {
            if d[11] != 0 {
                ppyoloe_add_classes(del, output, d, p, class_mult, batch, slot, class_targets, row, false);
            }
            let best = ppyoloe_best_truth_iou(output, truth, d, batch, slot);
            if best > p[4] {
                if d[10] != 0 {
                    ppyoloe_add_objectness(del, output, d, p, batch, slot, best);
                }
                return;
            }
            ppyoloe_add_objectness(del, output, d, p, batch, slot, 0.0);
            return;
        }

        let quality = assign[rec + 1];
        let mult = assign[rec + 7];
        let truth_box = (assign[rec + 3], assign[rec + 4], assign[rec + 5], assign[rec + 6]);
        let fw = d[0] as f32;
        let fh = d[1] as f32;
        let loc = slot % spatial;

        ppyoloe_add_objectness(del, output, d, p, batch, slot, quality);
        ppyoloe_add_classes(del, output, d, p, class_mult, batch, slot, class_targets, row, true);

        // C++ `add_dfl_box_delta` with normalizer `iou_normalizer * class_mult * dfl_loss_weight`.
        let dfl_norm = p[2] * mult * p[7];
        let mut side = 0;
        while side < 4 {
            let td = ppyoloe_target_ltrb(loc, d[0] as usize, fw, fh, truth_box, side);
            ppyoloe_add_dfl_side(del, output, d, batch, slot, side, td, dfl_norm);
            side += 1;
        }

        // C++ `dfl_giou_weight = iou_normalizer * class_mult * box_loss_weight * quality`.
        ppyoloe_add_giou(del, output, d, p, batch, slot, truth_box, p[2] * mult * p[6] * quality);
    }
}

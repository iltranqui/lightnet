//! `[yolox]` training: per-cell delta kernel. Port of raptor-fp4
//! `modern_yolo_layer.cpp::train_batch_modern` (YOLOX branch) and
//! `add_yolox_iou_box_delta` / `add_yolox_l1_box_delta`. Host side and the
//! SimOTA assignment are in `lightnet-core/src/layers/modern_yolo/yolox_train.rs`
//! (`assign_simota`, `pack_simota_assignments`).
//!
//! Layout (`n = 1` for every YOLOX head, cells indexed `b * spatial + loc`):
//! - `output`: activated head output, whole batch, channel-major. Entry `e` of cell
//!   `(b, loc)` is at `b * (5 + classes) * spatial + e * spatial + loc`, with
//!   `e = 0..4` the raw cxcywh box, `e = 4` objectness, `e = 5 + c` class `c`.
//! - `truth`: `t * truth_size + b * truths` holds `[x, y, w, h, class, ..]`,
//!   `truths = max_boxes * truth_size`. A zero `x` ends the list.
//! - `assign`: 4 floats per cell, `[truth_index or -1, class_id, quality, iou]`.
//! - `class_mult`: `classes` floats, `1.0` when the cfg has no `classes_multipliers`.
//! - `cfg`: `[obj_normalizer, cls_normalizer, iou_normalizer, box_loss_weight,
//!   ignore_thresh, max_delta, label_smooth_eps, objectness_smooth, ignore_neg,
//!   final_l1]`. Flags are `0.0` or `1.0`. `max_delta` of `f32::MAX` means no clip.
//! - `delta`: whole-batch buffer, same size as `output`. Must be zeroed by the caller.
//! - `stats`: 5 floats per cell, `[sum_iou, sum_iou_loss, cls_loss, box_loss, count]`.
//!   Written for every cell, so the host reduces it without a zero fill.
//!
//! One thread per cell. Each thread writes only its own cell's entries in `delta`
//! and its own `stats` slot, so the raw-pointer writes are disjoint.
//! Kernels are kept simple: no shared memory, no atomics.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

const YOLOX_EPS: f32 = 1e-8;
const YOLOX_FLT_MAX: f32 = 3.402_823_5e38;

#[device]
fn yx_finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[device]
fn yx_clamp01(value: f32) -> f32 {
    value.max(0.0).min(1.0)
}

/// C++ `safe_log`.
#[device]
fn yx_log(value: f32) -> f32 {
    value.max(YOLOX_EPS).ln()
}

/// Decode of one cxcywh box, C++ `decode_yolox_box`. Returns `(x, y, w, h)`.
#[device]
fn yx_decode(
    output: &[f32],
    box_index: usize,
    stride: usize,
    col: usize,
    row: usize,
    width: usize,
    height: usize,
) -> (f32, f32, f32, f32) {
    let x = (col as f32 + yx_finite(output[box_index])) / width as f32;
    let y = (row as f32 + yx_finite(output[box_index + stride])) / height as f32;
    let tw = yx_finite(output[box_index + 2 * stride]).max(-10.0).min(10.0);
    let th = yx_finite(output[box_index + 3 * stride]).max(-10.0).min(10.0);
    let w = (tw.exp() / width as f32).max(1e-6);
    let h = (th.exp() / height as f32).max(1e-6);
    (x, y, w, h)
}

/// C++ `box_iou`: intersection over union of two centre-size boxes.
#[device]
fn yx_iou(px: f32, py: f32, pw: f32, ph: f32, tx: f32, ty: f32, tw: f32, th: f32) -> f32 {
    let iw = (px + pw * 0.5).min(tx + tw * 0.5) - (px - pw * 0.5).max(tx - tw * 0.5);
    let ih = (py + ph * 0.5).min(ty + th * 0.5) - (py - ph * 0.5).max(ty - th * 0.5);
    if iw <= 0.0 || ih <= 0.0 {
        return 0.0;
    }
    let inter = iw * ih;
    let union = pw * ph + tw * th - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

/// C++ `dx_box_iou(pred, truth, IOU)`, same packing as the CPU `yolox_iou_grad`:
/// returns `(dt, db, dl, dr)`.
#[device]
fn yx_iou_grad(
    px: f32, py: f32, pw: f32, ph: f32, tx: f32, ty: f32, tw: f32, th: f32,
) -> (f32, f32, f32, f32) {
    let pl0 = px - pw * 0.5;
    let pr0 = px + pw * 0.5;
    let pt0 = py - ph * 0.5;
    let pb0 = py + ph * 0.5;
    let pred_t = pt0.min(pb0);
    let pred_b = pt0.max(pb0);
    let pred_l = pl0.min(pr0);
    let pred_r = pl0.max(pr0);
    let tt = ty - th * 0.5;
    let tb = ty + th * 0.5;
    let tl = tx - tw * 0.5;
    let tr = tx + tw * 0.5;

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

    let mut p_t = 0.0f32;
    let mut p_b = 0.0f32;
    let mut p_l = 0.0f32;
    let mut p_r = 0.0f32;
    if union > 0.0 {
        let u2 = union * union;
        p_t = (union * di_t - inter * du_t) / u2;
        p_b = (union * di_b - inter * du_b) / u2;
        p_l = (union * di_l - inter * du_l) / u2;
        p_r = (union * di_r - inter * du_r) / u2;
    }
    // Literal C++ sequential corner swap (see CPU `yolox_iou_grad`).
    let (p_t, p_b) = if pt0 < pb0 { (p_t, p_b) } else { (p_b, p_b) };
    let (p_l, p_r) = if pl0 < pr0 { (p_l, p_r) } else { (p_r, p_r) };

    (p_l + p_r, p_t + p_b, p_r - p_l, p_b - p_t)
}

/// Adds a finite value to `delta[index]`. The caller owns `index`.
#[device]
fn yx_add(delta: *mut f32, index: usize, value: f32) {
    if value.is_finite() {
        // SAFETY: the calling thread owns this cell; no other thread writes `index`.
        unsafe {
            let p = delta.add(index);
            *p = *p + value;
        }
    }
}

/// C++ `add_*_delta` clip: non-finite values are dropped, and the clamp applies only
/// when `max_delta` is not `FLT_MAX`.
#[device]
fn yx_add_clipped(delta: *mut f32, index: usize, value: f32, max_delta: f32) {
    if !value.is_finite() {
        return;
    }
    let mut v = value;
    if max_delta != YOLOX_FLT_MAX {
        v = v.max(-max_delta).min(max_delta);
    }
    yx_add(delta, index, v);
}

/// C++ `add_objectness_delta`: `obj_normalizer * (clamp(target) - output)`.
#[device]
fn yx_add_objectness(delta: *mut f32, index: usize, normalizer: f32, target: f32, out: f32) {
    let d = normalizer * (target.max(0.0).min(1.0) - out);
    yx_add(delta, index, d);
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// YOLOX training deltas, one thread per cell (`slot = b * spatial + loc`).
    ///
    /// Mirrors the CPU `train_yolox_image`: unassigned cells get an objectness
    /// target (0, or the best IoU in the ignore band), and assigned cells get
    /// objectness 1, one-hot class targets, the IoU box delta and optionally the
    /// L1 delta. Writes its 5 stats slots in every path.
    #[kernel]
    pub fn yolox_train_delta(
        output: &[f32],
        truth: &[f32],
        assign: &[f32],
        class_mult: &[f32],
        cfg: &[f32],
        mut delta: DisjointSlice<f32>,
        mut stats: DisjointSlice<f32>,
        batch: usize,
        width: usize,
        height: usize,
        classes: usize,
        max_boxes: usize,
        truth_size: usize,
    ) {
        let slot = thread::index_1d().get();
        let spatial = width * height;
        if slot >= batch * spatial {
            return;
        }
        let b = slot / spatial;
        let loc = slot % spatial;
        let col = loc % width;
        let row = loc / width;
        let stride = spatial;
        let entries = 5 + classes;
        let base = b * entries * stride;
        let box_index = base + loc;
        let obj_index = base + 4 * stride + loc;
        let truths = max_boxes * truth_size;

        let obj_norm = cfg[0];
        let cls_norm = cfg[1];
        let iou_norm = cfg[2];
        let box_w = cfg[3];
        let ignore_thresh = cfg[4];
        let max_delta = cfg[5];
        let eps_smooth = cfg[6];
        let obj_smooth = cfg[7] != 0.0;
        let ignore_neg = cfg[8] != 0.0;
        let final_l1 = cfg[9] != 0.0;

        let delta_ptr = delta.as_mut_ptr();
        let stats_ptr = stats.as_mut_ptr();
        // SAFETY: this thread owns stats slot `slot` (5 floats).
        unsafe {
            for k in 0..5 {
                *stats_ptr.add(slot * 5 + k) = 0.0;
            }
        }

        let meta = slot * 4;
        let truth_index = assign[meta];
        let out_obj = yx_finite(output[obj_index]);

        // Unassigned: objectness target only (C++ `a.truth_index < 0` branch).
        if truth_index < 0.0 {
            if ignore_neg {
                let (px, py, pw, ph) = yx_decode(output, box_index, stride, col, row, width, height);
                let mut best = 0.0f32;
                for t in 0..max_boxes {
                    let tb = t * truth_size + b * truths;
                    if truth[tb] == 0.0 {
                        break;
                    }
                    let iou = yx_clamp01(yx_finite(yx_iou(
                        px, py, pw, ph, truth[tb], truth[tb + 1], truth[tb + 2], truth[tb + 3],
                    )));
                    best = best.max(iou);
                }
                if best > ignore_thresh {
                    if obj_smooth {
                        yx_add_objectness(delta_ptr, obj_index, obj_norm, best, out_obj);
                    }
                    return;
                }
            }
            yx_add_objectness(delta_ptr, obj_index, obj_norm, 0.0, out_obj);
            return;
        }

        // Assigned.
        let class_id = assign[meta + 1] as usize;
        let quality = yx_clamp01(assign[meta + 2]);
        let a_iou = assign[meta + 3];
        let mult = class_mult[class_id];
        let tb = (truth_index as usize) * truth_size + b * truths;
        let (tx, ty, tw, th) = (truth[tb], truth[tb + 1], truth[tb + 2], truth[tb + 3]);

        yx_add_objectness(delta_ptr, obj_index, obj_norm, 1.0, out_obj);

        for c in 0..classes {
            let mut target = if c == class_id { quality } else { 0.0 };
            if eps_smooth != 0.0 {
                target = target * (1.0 - eps_smooth) + 0.5 * eps_smooth;
            }
            let idx = base + (5 + c) * stride + loc;
            let p = yx_clamp01(yx_finite(output[idx]));
            let mut d = cls_norm * (target - p);
            if c == class_id {
                d = d * mult;
            }
            yx_add(delta_ptr, idx, d);
        }

        let normalizer = iou_norm * mult;
        let (px, py, pw, ph) = yx_decode(output, box_index, stride, col, row, width, height);
        let iou = yx_clamp01(yx_finite(yx_iou(px, py, pw, ph, tx, ty, tw, th)));
        let (dt, db, dl, dr) = yx_iou_grad(px, py, pw, ph, tx, ty, tw, th);
        let total = normalizer * (2.0 * iou * box_w);
        let inv_w = 1.0 / width as f32;
        let inv_h = 1.0 / height as f32;
        yx_add_clipped(delta_ptr, box_index, total * dt * inv_w, max_delta);
        yx_add_clipped(delta_ptr, box_index + stride, total * db * inv_h, max_delta);
        yx_add_clipped(delta_ptr, box_index + 2 * stride, total * dl * pw, max_delta);
        yx_add_clipped(delta_ptr, box_index + 3 * stride, total * dr * ph, max_delta);

        if final_l1 {
            let l1_targets = [
                tx * width as f32 - col as f32,
                ty * height as f32 - row as f32,
                yx_log(tw * width as f32),
                yx_log(th * height as f32),
            ];
            for c in 0..4 {
                let idx = box_index + c * stride;
                let d = normalizer * (l1_targets[c] - yx_finite(output[idx]));
                yx_add_clipped(delta_ptr, idx, d, max_delta);
            }
        }

        // Stats: C++ `assigned_class_loss_value` (BCE, enable_vfl = false) and the
        // box-loss / IoU sums of `train_batch_modern`.
        let mut t_cls = quality;
        if eps_smooth != 0.0 {
            t_cls = t_cls * (1.0 - eps_smooth) + 0.5 * eps_smooth;
        }
        let p_cls = yx_clamp01(yx_finite(output[base + (5 + class_id) * stride + loc]));
        let cls_loss = yx_finite(-t_cls * yx_log(p_cls) - (1.0 - t_cls) * yx_log(1.0 - p_cls));
        let box_loss = box_w * (1.0 - a_iou * a_iou);
        // SAFETY: this thread owns stats slot `slot`.
        unsafe {
            *stats_ptr.add(slot * 5) = a_iou;
            *stats_ptr.add(slot * 5 + 1) = 1.0 - a_iou;
            *stats_ptr.add(slot * 5 + 2) = cls_loss;
            *stats_ptr.add(slot * 5 + 3) = box_loss;
            *stats_ptr.add(slot * 5 + 4) = 1.0;
        }
    }
}

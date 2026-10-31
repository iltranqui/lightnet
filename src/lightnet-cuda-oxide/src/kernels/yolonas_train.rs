//! `[yolonas]` per-cell training delta: GPU twin of `yolonas_train_delta` in
//! `src/lightnet-core/src/layers/modern_yolo/yolonas_train.rs`.
//!
//! Ported from `raptor-fp4:src-lib/modern_yolo_layer.cpp`: the per-cell body of
//! `train_batch_modern` (YOLO-NAS branch), `add_dfl_delta`,
//! `add_dfl_giou_delta_unprojected`, `add_class_delta_multi`, and from
//! `raptor-fp4:src-lib/box.cpp` the GIOU branch of `dx_box_iou`.
//!
//! Assignment (`yolonas_assign`) runs on the host.  Its result is passed in as
//! flat arrays indexed by `g = batch * slots + slot`, `slots = anchors * W * H`.
//! The kernel only writes deltas.  The caller zeroes `delta` first (C++ memset)
//! and resets non-finite entries afterwards, as the CPU path does.
//!
//! One thread per (image, cell).  Each thread writes only the delta entries of
//! its own cell, so the writes are disjoint and need no atomics.
//!
//! Buffer layout:
//! - `d = [width, height, anchors, classes, coords, max_boxes, truth_size, batches]`
//! - `p = [iou_normalizer, cls_normalizer, obj_normalizer, box_loss_weight,
//!         dfl_loss_weight, ignore_thresh, max_delta, label_smooth_eps, vfl_gamma,
//!         objectness_smooth]`; `max_delta == f32::MAX` disables the clip,
//!   `objectness_smooth` is 0 or 1.
//! - `slot_truth[g]`: truth row of the assignment, `-1` for background.
//! - `slot_class[g]`, `slot_quality[g]` (after duplicate decay), and
//!   `slot_box[4 * g ..]` (the assigned truth as `x, y, w, h`).
//! - `class_quality[g * classes + c]`: the per-class quality row.
//! - `class_multipliers`: `classes` entries, or empty for none.
//! - `truth`: packed truth rows, `batch * max_boxes * truth_size + t * truth_size + field`.
//! - `output`: the activated head output (`batch * outputs` entries).

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// C++ `k_eps` (`EPS` in `cpu.rs`).
const NAS_EPS: f32 = 1e-8;
/// C++ `k_vfl_alpha`.
const NAS_VFL_ALPHA: f32 = 0.75;

/// Entry index of `entry` for cell `slot` of image `batch` (C++ `modern_entry_index`).
#[device]
fn nas_entry(d: &[u32], batch: usize, slot: usize, entry: usize) -> usize {
    let spatial = (d[0] * d[1]) as usize;
    let channels = d[4] as usize + d[3] as usize + 1;
    let per_image = d[2] as usize * spatial * channels;
    batch * per_image + (slot / spatial) * spatial * channels + entry * spatial + slot % spatial
}

/// C++ `std::clamp(finite_or_zero(x), 0, 1)`.
#[device]
fn nas_clamp01(value: f32) -> f32 {
    let v = if value.is_finite() { value } else { 0.0 };
    v.max(0.0).min(1.0)
}

#[device]
fn nas_fin(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

/// Max over the finite side logits (C++ `dfl_expectation` / `dfl_probabilities`).
#[device]
fn nas_dfl_max(out: &[f32], start: usize, stride: usize, bins: usize) -> f32 {
    let mut m = f32::NEG_INFINITY;
    let mut k = 0;
    while k < bins {
        m = m.max(nas_fin(out[start + k * stride]));
        k += 1;
    }
    m
}

/// `exp(clamp(x_k - max, -60, 60))` for one bin.
#[device]
fn nas_dfl_exp(out: &[f32], start: usize, stride: usize, k: usize, max_logit: f32) -> f32 {
    (nas_fin(out[start + k * stride]) - max_logit).clamp(-60.0, 60.0).exp()
}

/// Softmax probability of bin `k` (C++ `dfl_probabilities`).  Recomputes the
/// max and the denominator per call: O(bins^2) per side, kept simple on purpose.
#[device]
fn nas_dfl_prob(out: &[f32], start: usize, stride: usize, bins: usize, k: usize) -> f32 {
    let m = nas_dfl_max(out, start, stride, bins);
    let mut den = 0.0f32;
    let mut j = 0;
    while j < bins {
        den += nas_dfl_exp(out, start, stride, j, m);
        j += 1;
    }
    nas_dfl_exp(out, start, stride, k, m) / den.max(NAS_EPS)
}

/// Softmax expectation of one side (C++ `dfl_expectation`).
#[device]
fn nas_dfl_expect(out: &[f32], start: usize, stride: usize, bins: usize) -> f32 {
    let m = nas_dfl_max(out, start, stride, bins);
    let mut den = 0.0f32;
    let mut ex = 0.0f32;
    let mut k = 0;
    while k < bins {
        let e = nas_dfl_exp(out, start, stride, k, m);
        den += e;
        ex += (k as f32) * e;
        k += 1;
    }
    ex / den.max(NAS_EPS)
}

/// C++ `decode_dfl_box`: returns `(x, y, w, h)` normalised.
#[device]
fn nas_decode(out: &[f32], d: &[u32], batch: usize, slot: usize) -> (f32, f32, f32, f32) {
    let width = d[0] as usize;
    let spatial = (d[0] * d[1]) as usize;
    let bins = (d[4] as usize / 4).max(2);
    let loc = slot % spatial;
    let col = loc % width;
    let row = loc / width;
    let cx = col as f32 + 0.5;
    let cy = row as f32 + 0.5;
    let w = d[0] as f32;
    let h = d[1] as f32;
    let left = nas_dfl_expect(out, nas_entry(d, batch, slot, 0), spatial, bins);
    let top = nas_dfl_expect(out, nas_entry(d, batch, slot, bins), spatial, bins);
    let right = nas_dfl_expect(out, nas_entry(d, batch, slot, 2 * bins), spatial, bins);
    let bottom = nas_dfl_expect(out, nas_entry(d, batch, slot, 3 * bins), spatial, bins);
    // C++ box_from_corners.
    let (mut l, mut t, mut r, mut b) = (
        nas_fin((cx - left) / w),
        nas_fin((cy - top) / h),
        nas_fin((cx + right) / w),
        nas_fin((cy + bottom) / h),
    );
    if r < l {
        let s = l;
        l = r;
        r = s;
    }
    if b < t {
        let s = t;
        t = b;
        b = s;
    }
    let bw = (r - l).max(1e-6);
    let bh = (b - t).max(1e-6);
    let bx = nas_fin(0.5 * (l + r));
    let by = nas_fin(0.5 * (t + b));
    (bx, by, nas_fin(bw).max(1e-6), nas_fin(bh).max(1e-6))
}

#[device]
fn nas_overlap(x1: f32, w1: f32, x2: f32, w2: f32) -> f32 {
    let left = (x1 - w1 / 2.0).max(x2 - w2 / 2.0);
    let right = (x1 + w1 / 2.0).min(x2 + w2 / 2.0);
    right - left
}

/// C++ `box_intersection`.
#[device]
fn nas_intersection(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let w = nas_overlap(a.0, a.2, b.0, b.2);
    if w <= 0.0 {
        return 0.0;
    }
    let h = nas_overlap(a.1, a.3, b.1, b.3);
    if h <= 0.0 {
        return 0.0;
    }
    w * h
}

/// C++ `box_iou` (not clamped; the caller clamps).
#[device]
fn nas_iou(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> f32 {
    let i = nas_intersection(a, b);
    if i == 0.0 {
        return 0.0;
    }
    let u = a.2 * a.3 + b.2 * b.3 - i;
    i / u
}

/// C++ `dx_box_iou(pred, truth, GIOU)`, returns `(dt, db, dl, dr)`.
#[device]
fn nas_dx_giou(pred: (f32, f32, f32, f32), truth: (f32, f32, f32, f32)) -> (f32, f32, f32, f32) {
    let p_top0 = pred.1 - pred.3 / 2.0;
    let p_bot0 = pred.1 + pred.3 / 2.0;
    let p_left0 = pred.0 - pred.2 / 2.0;
    let p_right0 = pred.0 + pred.2 / 2.0;
    let pred_t = p_top0.min(p_bot0);
    let pred_b = p_top0.max(p_bot0);
    let pred_l = p_left0.min(p_right0);
    let pred_r = p_left0.max(p_right0);
    let t_top = truth.1 - truth.3 / 2.0;
    let t_bot = truth.1 + truth.3 / 2.0;
    let t_left = truth.0 - truth.2 / 2.0;
    let t_right = truth.0 + truth.2 / 2.0;

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

    let mut p_dt = 0.0f32;
    let mut p_db = 0.0f32;
    let mut p_dl = 0.0f32;
    let mut p_dr = 0.0f32;
    if u > 0.0 {
        p_dt = (u * di_t - i * du_t) / (u * u);
        p_db = (u * di_b - i * du_b) / (u * u);
        p_dl = (u * di_l - i * du_l) / (u * u);
        p_dr = (u * di_r - i * du_r) / (u * u);
    }

    // Corner selection for inverted boxes; sequential as in C++.
    p_dt = if p_top0 < p_bot0 { p_dt } else { p_db };
    p_db = if p_top0 < p_bot0 { p_db } else { p_dt };
    p_dl = if p_left0 < p_right0 { p_dl } else { p_dr };
    p_dr = if p_left0 < p_right0 { p_dr } else { p_dl };

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

    (p_dl + p_dr, p_dt + p_db, p_dr - p_dl, p_db - p_dt)
}

/// C++ `train_batch_modern`, objectness write: `obj_normalizer * (clamp(target) - obj)`.
#[device]
fn nas_add_objectness(out: &[f32], del: *mut f32, idx: usize, target: f32, obj_norm: f32) {
    let t = target.max(0.0).min(1.0);
    let dd = obj_norm * (t - nas_fin(out[idx]));
    if dd.is_finite() {
        // SAFETY: this thread owns the cell's objectness entry.
        unsafe {
            *del.add(idx) += dd;
        }
    }
}

/// C++ `add_class_delta_multi` (enable_vfl = true).  The row is the per-class
/// quality, with the box winner's own channel set to `quality`.
#[device]
fn nas_class_multi(
    out: &[f32], del: *mut f32, d: &[u32], batch: usize, slot: usize,
    class_quality: &[f32], g: usize, multipliers: &[f32], class_id: usize,
    quality: f32, cls_norm: f32, label_smooth_eps: f32, vfl_gamma: f32,
) {
    let classes = d[3] as usize;
    let coords = d[4] as usize;
    let use_vfl = vfl_gamma > 0.0;
    let mut c = 0;
    while c < classes {
        let raw = if c == class_id { quality } else { class_quality[g * classes + c] };
        let mut target = raw.max(0.0).min(1.0);
        if label_smooth_eps != 0.0 && !use_vfl {
            target = target * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps;
        }
        let idx = nas_entry(d, batch, slot, coords + 1 + c);
        let prob = nas_clamp01(out[idx]);
        let mut dd = cls_norm * (target - prob);
        if use_vfl {
            let weight = if target > 0.0 { target } else { NAS_VFL_ALPHA * prob.powf(vfl_gamma) };
            dd *= weight;
        }
        if multipliers.len() > 0 && target > 0.0 {
            dd *= multipliers[c];
        }
        if dd.is_finite() {
            // SAFETY: this thread owns the cell's class entries.
            unsafe {
                *del.add(idx) += dd;
            }
        }
        c += 1;
    }
}

/// C++ `add_class_delta(..., -1, 0, true)`: every class target 0, VFL weighted.
#[device]
fn nas_negative_class(out: &[f32], del: *mut f32, d: &[u32], batch: usize, slot: usize, cls_norm: f32, vfl_gamma: f32) {
    let classes = d[3] as usize;
    let coords = d[4] as usize;
    let mut c = 0;
    while c < classes {
        let idx = nas_entry(d, batch, slot, coords + 1 + c);
        let prob = nas_clamp01(out[idx]);
        let mut dd = cls_norm * (0.0 - prob);
        dd *= NAS_VFL_ALPHA * prob.powf(vfl_gamma);
        if dd.is_finite() {
            // SAFETY: this thread owns the cell's class entries.
            unsafe {
                *del.add(idx) += dd;
            }
        }
        c += 1;
    }
}

/// C++ `add_dfl_delta`: two-bin distribution target for one side.
#[device]
fn nas_dfl_delta(
    out: &[f32], del: *mut f32, d: &[u32], batch: usize, slot: usize,
    side: usize, target_distance: f32, normalizer: f32,
) {
    let bins = (d[4] as usize / 4).max(2);
    let max_bin = bins - 1;
    let spatial = (d[0] * d[1]) as usize;
    let start = nas_entry(d, batch, slot, side * bins);
    let target = nas_fin(target_distance).max(0.0).min(max_bin as f32 - 1e-4);
    let left_bin = (target.floor() as usize).min(max_bin);
    let right_bin = (left_bin + 1).min(max_bin);
    let right_weight = target - left_bin as f32;
    let left_weight = 1.0 - right_weight;
    let mut k = 0;
    while k < bins {
        let mut tgt = 0.0f32;
        if k == left_bin {
            tgt += left_weight;
        }
        if k == right_bin {
            tgt += right_weight;
        }
        let dd = normalizer * (tgt - nas_dfl_prob(out, start, spatial, bins, k));
        if dd.is_finite() {
            // SAFETY: this thread owns the cell's distribution entries.
            unsafe {
                *del.add(start + k * spatial) += dd;
            }
        }
        k += 1;
    }
}

/// C++ `add_dfl_box_delta`: target distances from the assigned truth.
#[device]
fn nas_add_dfl_box(
    out: &[f32], del: *mut f32, d: &[u32], batch: usize, slot: usize,
    bx: (f32, f32, f32, f32), normalizer: f32,
) {
    let w = d[0] as f32;
    let h = d[1] as f32;
    let spatial = (d[0] * d[1]) as usize;
    let loc = slot % spatial;
    let col = loc % d[0] as usize;
    let row = loc / d[0] as usize;
    let cx = col as f32 + 0.5;
    let cy = row as f32 + 0.5;
    // C++ target_ltrb_distance: truth.left()*W etc.
    let left = (bx.0 - 0.5 * bx.2) * w;
    let top = (bx.1 - 0.5 * bx.3) * h;
    let right = (bx.0 + 0.5 * bx.2) * w;
    let bottom = (bx.1 + 0.5 * bx.3) * h;
    nas_dfl_delta(out, del, d, batch, slot, 0, cx - left, normalizer);
    nas_dfl_delta(out, del, d, batch, slot, 1, cy - top, normalizer);
    nas_dfl_delta(out, del, d, batch, slot, 2, right - cx, normalizer);
    nas_dfl_delta(out, del, d, batch, slot, 3, bottom - cy, normalizer);
}

/// C++ `add_dfl_giou_delta_unprojected`: GIoU gradient through each side's expectation.
#[device]
fn nas_add_dfl_giou(
    out: &[f32], del: *mut f32, d: &[u32], batch: usize, slot: usize,
    truth: (f32, f32, f32, f32), weight: f32, max_delta: f32,
) {
    let bins = (d[4] as usize / 4).max(2);
    let spatial = (d[0] * d[1]) as usize;
    let pred = nas_decode(out, d, batch, slot);
    let g = nas_dx_giou(pred, truth);
    let inv_w = 1.0 / d[0] as f32;
    let inv_h = 1.0 / d[1] as f32;
    let corner: [f32; 4] = [
        nas_fin(0.5 * (g.0 - g.2)),
        nas_fin(0.5 * (g.1 - g.3)),
        nas_fin(0.5 * (g.0 + g.2)),
        nas_fin(0.5 * (g.1 + g.3)),
    ];
    let scale: [f32; 4] = [-inv_w, -inv_h, inv_w, inv_h];
    let mut side = 0;
    while side < 4 {
        let start = nas_entry(d, batch, slot, side * bins);
        let expected = nas_fin(nas_dfl_expect(out, start, spatial, bins));
        let chain = nas_fin(weight) * corner[side] * scale[side];
        let mut k = 0;
        while k < bins {
            let mut dd = chain * nas_fin(nas_dfl_prob(out, start, spatial, bins, k)) * (k as f32 - expected);
            if max_delta != f32::MAX {
                dd = dd.max(-max_delta).min(max_delta);
            }
            if dd.is_finite() {
                // SAFETY: this thread owns the cell's distribution entries.
                unsafe {
                    *del.add(start + k * spatial) += dd;
                }
            }
            k += 1;
        }
        side += 1;
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// YOLO-NAS per-cell delta: one thread per (image, cell).  Background cells
    /// get the ignore-band or background objectness target; assigned cells get
    /// objectness, class (VFL), DFL box and DFL-GIoU deltas.  Writes only the
    /// cell's own entries, so the caller must zero `delta` beforehand.
    #[kernel]
    pub fn yolonas_cell_delta(
        output: &[f32],
        truth: &[f32],
        slot_truth: &[i32],
        slot_class: &[i32],
        slot_quality: &[f32],
        slot_box: &[f32],
        class_quality: &[f32],
        class_multipliers: &[f32],
        mut delta: DisjointSlice<f32>,
        d: &[u32],
        p: &[f32],
    ) {
        let g = thread::index_1d().get();
        let spatial = (d[0] * d[1]) as usize;
        let anchors = d[2] as usize;
        let slots = anchors * spatial;
        if g >= d[7] as usize * slots {
            return;
        }
        let batch = g / slots;
        let slot = g % slots;
        let coords = d[4] as usize;
        let truths = d[5] as usize * d[6] as usize;
        let del = delta.as_mut_ptr();
        let obj_index = nas_entry(d, batch, slot, coords);
        let obj_norm = p[2];
        let cls_norm = p[1];

        if slot_truth[g] < 0 {
            // C++ unassigned branch: negative VFL classes first, then the ignore band.
            if p[8] > 0.0 {
                nas_negative_class(output, del, d, batch, slot, cls_norm, p[8]);
            }
            let pred = nas_decode(output, d, batch, slot);
            let mut best = 0.0f32;
            let mut t = 0;
            while t < d[5] as usize {
                let base = batch * truths + t * d[6] as usize;
                if truth[base] == 0.0 {
                    break;
                }
                let tb = (truth[base], truth[base + 1], truth[base + 2], truth[base + 3]);
                best = best.max(nas_clamp01(nas_iou(pred, tb)));
                t += 1;
            }
            if best > p[5] {
                if p[9] > 0.5 {
                    nas_add_objectness(output, del, obj_index, best, obj_norm);
                }
                return;
            }
            nas_add_objectness(output, del, obj_index, 0.0, obj_norm);
            return;
        }

        // Assigned branch.
        let class_id = slot_class[g] as usize;
        let class_mult = if class_multipliers.len() > 0 { class_multipliers[class_id] } else { 1.0 };
        let quality = slot_quality[g].max(0.0).min(1.0);
        let bx = (
            slot_box[4 * g],
            slot_box[4 * g + 1],
            slot_box[4 * g + 2],
            slot_box[4 * g + 3],
        );

        nas_add_objectness(output, del, obj_index, quality, obj_norm);
        nas_class_multi(
            output, del, d, batch, slot, class_quality, g, class_multipliers, class_id,
            quality, cls_norm, p[7], p[8],
        );
        nas_add_dfl_box(output, del, d, batch, slot, bx, p[0] * class_mult * p[4]);
        nas_add_dfl_giou(output, del, d, batch, slot, bx, p[0] * class_mult * p[3] * quality, p[6]);
    }
}

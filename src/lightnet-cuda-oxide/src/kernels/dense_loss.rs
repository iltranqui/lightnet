//! Dense-head loss and decode kernels: CenterNet, modern YOLO, segmentation, JevCe, LayaLoss and region.

use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

// ---- Classic YOLO GPU loss (port of raptor-fp4 `yolo_layer.cu`) ----
//
// `d` packs the integer layer shape: [width, height, anchors, total_anchors, classes,
// max_boxes, truth_size, truths, outputs, net_width, net_height, new_coords,
// objectness_smooth]. `p` packs the float options: [ignore_thresh, iou_normalizer,
// obj_normalizer, max_delta, iou_thresh, truth_thresh]. Per-image metrics are
// [total_iou, total_iou_loss, delta_squared, classification_delta_squared, count,
// total_bbox, rewritten_bbox].

type YBox = (f32, f32, f32, f32);

#[device]
fn yolo_entry(d: &[u32], batch: usize, anchor: usize, location: usize, entry: usize) -> usize {
    let spatial = (d[0] * d[1]) as usize;
    batch * d[8] as usize + anchor * spatial * (d[4] as usize + 5) + entry * spatial + location
}

#[device]
fn yolo_truth_box(truth: &[f32], base: usize) -> YBox {
    (truth[base], truth[base + 1], truth[base + 2], truth[base + 3])
}

#[device]
fn yolo_decode(output: *const f32, biases: &[f32], bias_anchor: usize, box_index: usize, column: usize, row: usize, d: &[u32]) -> YBox {
    let spatial = (d[0] * d[1]) as usize;
    let (w, h) = (d[0] as f32, d[1] as f32);
    unsafe {
        let x = (column as f32 + *output.add(box_index)) / w;
        let y = (row as f32 + *output.add(box_index + spatial)) / h;
        let tw = *output.add(box_index + 2 * spatial);
        let th = *output.add(box_index + 3 * spatial);
        if d[11] != 0 {
            (x, y, tw * tw * 4.0 * biases[2 * bias_anchor] / d[9] as f32, th * th * 4.0 * biases[2 * bias_anchor + 1] / d[10] as f32)
        } else {
            (x, y, tw.exp() * biases[2 * bias_anchor] / d[9] as f32, th.exp() * biases[2 * bias_anchor + 1] / d[10] as f32)
        }
    }
}

#[device]
fn yolo_iou(a: YBox, b: YBox) -> f32 {
    let width = (a.0 + a.2 * 0.5).min(b.0 + b.2 * 0.5) - (a.0 - a.2 * 0.5).max(b.0 - b.2 * 0.5);
    let height = (a.1 + a.3 * 0.5).min(b.1 + b.3 * 0.5) - (a.1 - a.3 * 0.5).max(b.1 - b.3 * 0.5);
    if width <= 0.0 || height <= 0.0 {
        return 0.0;
    }
    let overlap = width * height;
    overlap / (a.2 * a.3 + b.2 * b.3 - overlap)
}

#[device]
fn yolo_finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[device]
fn yolo_sanitize(value: f32, max_delta: f32) -> f32 {
    if value.is_finite() { value.max(-max_delta).min(max_delta) } else { 0.0 }
}

/// CIoU derivative of the prediction box (same operation order as `dx_box_iou(..., CIOU)`).
#[device]
fn yolo_ciou_derivative(pred: YBox, truth: YBox) -> YBox {
    let pi = core::f32::consts::PI;
    let (pl0, pr0, pt0, pb0) = (pred.0 - pred.2 * 0.5, pred.0 + pred.2 * 0.5, pred.1 - pred.3 * 0.5, pred.1 + pred.3 * 0.5);
    let pred_t = pt0.min(pb0);
    let pred_b = pt0.max(pb0);
    let pred_l = pl0.min(pr0);
    let pred_r = pl0.max(pr0);
    let (tl, tr, tt, tb) = (truth.0 - truth.2 * 0.5, truth.0 + truth.2 * 0.5, truth.1 - truth.3 * 0.5, truth.1 + truth.3 * 0.5);

    let pred_area = (pred_b - pred_t) * (pred_r - pred_l);
    let truth_area = (tb - tt) * (tr - tl);
    let intersection_h = pred_b.min(tb) - pred_t.max(tt);
    let intersection_w = pred_r.min(tr) - pred_l.max(tl);
    let intersection_area = intersection_w * intersection_h;
    let union_area = pred_area + truth_area - intersection_area;
    let center_distance = (pred.0 - truth.0) * (pred.0 - truth.0) + (pred.1 - truth.1) * (pred.1 - truth.1);

    let d_int_t = if pred_t > tt { -intersection_w } else { 0.0 };
    let d_int_b = if pred_b < tb { intersection_w } else { 0.0 };
    let d_int_l = if pred_l > tl { -intersection_h } else { 0.0 };
    let d_int_r = if pred_r < tr { intersection_h } else { 0.0 };
    let d_union_t = -(pred_r - pred_l) - d_int_t;
    let d_union_b = (pred_r - pred_l) - d_int_b;
    let d_union_l = -(pred_b - pred_t) - d_int_l;
    let d_union_r = (pred_b - pred_t) - d_int_r;

    let (mut p_t, mut p_b, mut p_l, mut p_r) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    if union_area > 0.0 {
        let union_squared = union_area * union_area;
        p_t = (union_area * d_int_t - intersection_area * d_union_t) / union_squared;
        p_b = (union_area * d_int_b - intersection_area * d_union_b) / union_squared;
        p_l = (union_area * d_int_l - intersection_area * d_union_l) / union_squared;
        p_r = (union_area * d_int_r - intersection_area * d_union_r) / union_squared;
    }
    // Mirrors the original sequential assignments.
    let (p_t, p_b) = if pt0 < pb0 { (p_t, p_b) } else { (p_b, p_b) };
    let (p_l, p_r) = if pl0 < pr0 { (p_l, p_r) } else { (p_r, p_r) };

    let enclosing_w = (pred.0 + pred.2 * 0.5).max(truth.0 + truth.2 * 0.5) - (pred.0 - pred.2 * 0.5).min(truth.0 - truth.2 * 0.5);
    let enclosing_h = (pred.1 + pred.3 * 0.5).max(truth.1 + truth.3 * 0.5) - (pred.1 - pred.3 * 0.5).min(truth.1 - truth.3 * 0.5);
    let diagonal_squared = enclosing_w * enclosing_w + enclosing_h * enclosing_h;
    let d_ew_x = (if pred_r > tr { 1.0 } else { 0.0 }) - (if pred_l < tl { 1.0 } else { 0.0 });
    let d_ew_w = (if pred_r > tr { 0.5 } else { 0.0 }) - (if pred_l < tl { -0.5 } else { 0.0 });
    let d_eh_y = (if pred_b > tb { 1.0 } else { 0.0 }) - (if pred_t < tt { 1.0 } else { 0.0 });
    let d_eh_h = (if pred_b > tb { 0.5 } else { 0.0 }) - (if pred_t < tt { -0.5 } else { 0.0 });

    let aspect_angle = truth.2.atan2(truth.3) - pred.2.atan2(pred.3);
    let aspect_loss = 4.0 / (pi * pi) * aspect_angle * aspect_angle;
    let alpha = aspect_loss / (1.0 - intersection_area / union_area + aspect_loss + 0.000001);

    let no_overlap = intersection_w <= 0.0 || intersection_h <= 0.0;
    let mut d: YBox = if no_overlap { (0.0, 0.0, 0.0, 0.0) } else { (p_l + p_r, p_t + p_b, p_r - p_l, p_b - p_t) };
    if diagonal_squared > 0.0 || no_overlap {
        let diagonal_fourth = diagonal_squared * diagonal_squared;
        d.0 += (2.0 * (truth.0 - pred.0) * diagonal_squared - 2.0 * enclosing_w * d_ew_x * center_distance) / diagonal_fourth;
        d.1 += (2.0 * (truth.1 - pred.1) * diagonal_squared - 2.0 * enclosing_h * d_eh_y * center_distance) / diagonal_fourth;
        d.2 += (2.0 * enclosing_w * d_ew_w * center_distance) / diagonal_fourth + alpha * (8.0 / (pi * pi) * aspect_angle * pred.3);
        d.3 += (2.0 * enclosing_h * d_eh_h * center_distance) / diagonal_fourth + alpha * (-8.0 / (pi * pi) * aspect_angle * pred.2);
    }
    d
}

/// Box delta of the cell/anchor matched to a truth; `m` accumulates
/// [total_iou, total_iou_loss, total_bbox, rewritten_bbox].
#[device]
fn yolo_assign_box(
    truth: YBox, output: *const f32, delta: *mut f32, biases: &[f32], bias_anchor: usize,
    box_index: usize, column: usize, row: usize, d: &[u32], p: &[f32], m: &mut [f32; 4],
) {
    let spatial = (d[0] * d[1]) as usize;
    unsafe {
        if *delta.add(box_index) != 0.0 || *delta.add(box_index + spatial) != 0.0
            || *delta.add(box_index + 2 * spatial) != 0.0 || *delta.add(box_index + 3 * spatial) != 0.0 {
            m[3] += 1.0;
        }
        let mut pred = yolo_decode(output, biases, bias_anchor, box_index, column, row, d);
        let iou = yolo_iou(pred, truth);
        m[0] += iou;
        m[1] += 1.0 - iou;
        if pred.2 == 0.0 { pred.2 = 1.0; }
        if pred.3 == 0.0 { pred.3 = 1.0; }
        let g = yolo_ciou_derivative(pred, truth);
        let (norm, max_delta) = (p[1], p[3]);
        let (sw, sh) = if d[11] != 0 {
            (1.0, 1.0)
        } else {
            ((*output.add(box_index + 2 * spatial)).exp(), (*output.add(box_index + 3 * spatial)).exp())
        };
        *delta.add(box_index) += yolo_sanitize(g.0 * norm, max_delta);
        *delta.add(box_index + spatial) += yolo_sanitize(g.1 * norm, max_delta);
        *delta.add(box_index + 2 * spatial) += yolo_sanitize(g.2 * sw * norm, max_delta);
        *delta.add(box_index + 3 * spatial) += yolo_sanitize(g.3 * sh * norm, max_delta);
        m[2] += 1.0;
    }
}

/// Objectness delta of a matched cell; with objectness_smooth it only fills an empty delta.
#[device]
fn yolo_assign_objectness(output: *const f32, delta: *mut f32, obj_index: usize, d: &[u32], p: &[f32]) {
    unsafe {
        let value = p[2] * (1.0 - yolo_finite(*output.add(obj_index)));
        if d[12] == 0 || *delta.add(obj_index) == 0.0 {
            *delta.add(obj_index) = value;
        }
    }
}

#[device]
fn yolo_assign_class(output: *const f32, delta: *mut f32, class_index: usize, class_id: usize, d: &[u32]) {
    let spatial = (d[0] * d[1]) as usize;
    let classes = d[4] as usize;
    unsafe {
        let target = class_index + class_id * spatial;
        if *delta.add(target) != 0.0 {
            let value = 1.0 - *output.add(target);
            if value.is_finite() { *delta.add(target) = value; }
            return;
        }
        let mut c = 0;
        while c < classes {
            let index = class_index + c * spatial;
            let value = (if c == class_id { 1.0 } else { 0.0 }) - *output.add(index);
            if value.is_finite() { *delta.add(index) = value; }
            c += 1;
        }
    }
}

/// Number of leading truth boxes of an image (x == 0 or a bad class id ends the list).
#[device]
fn yolo_truth_count(truth: &[f32], batch: usize, d: &[u32]) -> usize {
    let (max_boxes, size, base) = (d[5] as usize, d[6] as usize, batch * d[7] as usize);
    let mut t = 0;
    while t < max_boxes {
        if truth[base + t * size] == 0.0 { break; }
        let class_id = truth[base + t * size + 4] as i32;
        if class_id < 0 || class_id >= d[4] as i32 { break; }
        t += 1;
    }
    t
}

// ---- Modern anchor-free heads: [yolox] / [ppyoloe] / [yolonas] training loss ----
//
// Device port of `layers/modern_yolo/cpu.rs` (`assign_yolox`, `assign_tal_like`, `yolox_train_batch`).
// `d` packs [width, height, classes, coords, kind, max_boxes, truth_size, batches] (kind 0 YOLOX,
// 1 PP-YOLOE, 2 YOLO-NAS). `p` packs [cls_normalizer, obj_normalizer, iou_normalizer, ignore_thresh,
// max_delta, label_smooth_eps, objectness_smooth, min_object_cells, vfl_gamma]. The head output is
// the activated tensor and the delta covers the whole batch, both `[batch, coords + classes + 1, H, W]`.
//
// Per image the scratch holds `MS_SLOT` floats per cell
//   [truth_index or -1, class_id, quality, iou, tx, ty, tw, th, conflict_score, conflict_area, rank_used, _]
// then `classes` class-quality floats per cell, then `MS_CAND` candidate floats per cell
//   [loc, iou, score, cost, used].

const MS_SLOT: usize = 12;
const MS_CAND: usize = 5;
const MS_EPS: f32 = 1e-8;
const MS_BIG: f32 = 3.0e38;

#[device]
fn ms_per_image(d: &[u32]) -> usize {
    let s = (d[0] * d[1]) as usize;
    s * MS_SLOT + s * d[2] as usize + s * MS_CAND
}

#[device]
fn ms_entry(d: &[u32], b: usize, entry: usize, loc: usize) -> usize {
    let s = (d[0] * d[1]) as usize;
    b * (d[3] as usize + d[2] as usize + 1) * s + entry * s + loc
}

#[device]
fn ms_ln(v: f32) -> f32 {
    v.max(MS_EPS).ln()
}

#[device]
fn ms_dfl_expect(out: *const f32, d: &[u32], b: usize, loc: usize, side: usize) -> f32 {
    let bins = ((d[3] / 4) as usize).max(2);
    let s = (d[0] * d[1]) as usize;
    let start = ms_entry(d, b, side * bins, loc);
    let mut m = f32::NEG_INFINITY;
    let mut k = 0;
    while k < bins {
        m = m.max(yolo_finite(unsafe { *out.add(start + k * s) }));
        k += 1;
    }
    let (mut denom, mut expected) = (0.0f32, 0.0f32);
    k = 0;
    while k < bins {
        let e = (yolo_finite(unsafe { *out.add(start + k * s) }) - m).max(-60.0).min(60.0).exp();
        denom += e;
        expected += k as f32 * e;
        k += 1;
    }
    expected / denom.max(MS_EPS)
}

#[device]
fn ms_sanitize(bx: YBox) -> YBox {
    (yolo_finite(bx.0), yolo_finite(bx.1), yolo_finite(bx.2).max(1e-6), yolo_finite(bx.3).max(1e-6))
}

#[device]
fn ms_box(out: *const f32, d: &[u32], b: usize, loc: usize) -> YBox {
    let (w, h) = (d[0] as usize, d[1] as usize);
    let s = w * h;
    let (col, row) = ((loc % w) as f32, (loc / w) as f32);
    if d[4] == 0 {
        let idx = ms_entry(d, b, 0, loc);
        unsafe {
            return ms_sanitize((
                (col + yolo_finite(*out.add(idx))) / w as f32,
                (row + yolo_finite(*out.add(idx + s))) / h as f32,
                yolo_finite(*out.add(idx + 2 * s)).max(-10.0).min(10.0).exp() / w as f32,
                yolo_finite(*out.add(idx + 3 * s)).max(-10.0).min(10.0).exp() / h as f32,
            ));
        }
    }
    let (cx, cy) = (col + 0.5, row + 0.5);
    let left = ms_dfl_expect(out, d, b, loc, 0);
    let top = ms_dfl_expect(out, d, b, loc, 1);
    let right = ms_dfl_expect(out, d, b, loc, 2);
    let bottom = ms_dfl_expect(out, d, b, loc, 3);
    let (mut l, mut t) = (yolo_finite((cx - left) / w as f32), yolo_finite((cy - top) / h as f32));
    let (mut r, mut bo) = (yolo_finite((cx + right) / w as f32), yolo_finite((cy + bottom) / h as f32));
    if r < l { let tmp = l; l = r; r = tmp; }
    if bo < t { let tmp = t; t = bo; bo = tmp; }
    ms_sanitize((0.5 * (l + r), 0.5 * (t + bo), (r - l).max(1e-6), (bo - t).max(1e-6)))
}

/// C++ `box_iou` as used by the modern heads (centre-size boxes).
#[device]
fn ms_iou(a: YBox, b: YBox) -> f32 {
    let iw = ((a.0 + 0.5 * a.2).min(b.0 + 0.5 * b.2) - (a.0 - 0.5 * a.2).max(b.0 - 0.5 * b.2)).max(0.0);
    let ih = ((a.1 + 0.5 * a.3).min(b.1 + 0.5 * b.3) - (a.1 - 0.5 * a.3).max(b.1 - 0.5 * b.3)).max(0.0);
    let inter = iw * ih;
    if inter == 0.0 {
        return 0.0;
    }
    let union = a.2.max(0.0) * a.3.max(0.0) + b.2.max(0.0) * b.3.max(0.0) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[device]
fn ms_in_center(w: usize, h: usize, loc: usize, t: YBox) -> bool {
    let gx = (loc % w) as f32 + 0.5;
    let gy = (loc / w) as f32 + 0.5;
    (gx - t.0 * w as f32).abs() <= 2.5 && (gy - t.1 * h as f32).abs() <= 2.5
}

#[device]
fn ms_in_box(w: usize, h: usize, loc: usize, t: YBox) -> bool {
    let cx = ((loc % w) as f32 + 0.5) / w as f32;
    let cy = ((loc / w) as f32 + 0.5) / h as f32;
    cx >= t.0 - 0.5 * t.2 && cx <= t.0 + 0.5 * t.2 && cy >= t.1 - 0.5 * t.3 && cy <= t.1 + 0.5 * t.3
}

#[device]
fn ms_clamped(out: *const f32, index: usize) -> f32 {
    yolo_finite(unsafe { *out.add(index) }).max(0.0).min(1.0)
}

// ---- CenterNet training loss ----
//
// Device port of `layers/centernet/cpu.rs::centernet_train_batch`. `d` packs [width, height, classes,
// max_boxes, truth_size, batches, net_width, net_height, center_min_radius, anisotropic]; `p` packs
// [small_ref_size, small_boost, scale_min_px, scale_max_px, gaussian_iou, focal_alpha, focal_beta,
// hm_normalizer, wh_normalizer, off_normalizer, max_delta].

#[device]
fn ct_clamp_delta(v: f32, max_delta: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    if max_delta >= 3.4028235e38 {
        return v;
    }
    v.max(-max_delta).min(max_delta)
}

#[device]
fn ct_safe_log(v: f32) -> f32 {
    v.max(1e-9).ln()
}

#[device]
fn ct_smooth_l1_loss(e: f32) -> f32 {
    let a = e.abs();
    if a < 1.0 { 0.5 * e * e } else { a - 0.5 }
}

#[device]
fn ct_smooth_l1_grad(e: f32) -> f32 {
    if e > 1.0 { 1.0 } else if e < -1.0 { -1.0 } else { e }
}

#[device]
fn ct_gaussian_radius(width: f32, height: f32, min_overlap: f32) -> i32 {
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return 0;
    }
    let (w, h) = (width.max(1e-6), height.max(1e-6));
    let m = min_overlap.max(0.01).min(0.99);
    let (b1, c1) = (h + w, w * h * (1.0 - m) / (1.0 + m));
    let d1 = (b1 * b1 - 4.0 * c1).max(0.0);
    let r1 = (b1 + d1.sqrt()) / 2.0;
    let (b2, c2) = (2.0 * (h + w), (1.0 - m) * w * h);
    let d2 = (b2 * b2 - 16.0 * c2).max(0.0);
    let r2 = (b2 + d2.sqrt()) / 2.0;
    let (a3, b3, c3) = (4.0 * m, -2.0 * m * (h + w), (m - 1.0) * w * h);
    let d3 = (b3 * b3 - 4.0 * a3 * c3).max(0.0);
    let r3 = (b3 + d3.sqrt()) / (2.0 * a3);
    (r1.min(r2.min(r3)).floor() as i32).max(0)
}

#[device]
fn ct_small_weight(p: &[f32], min_side_px: f32) -> f32 {
    let max_weight = p[1].max(1.0);
    let ref_px = p[0].max(1.0);
    if !min_side_px.is_finite() || min_side_px <= 0.0 {
        return max_weight;
    }
    (ref_px / min_side_px.max(1e-3)).max(1.0).min(max_weight)
}

#[device]
fn ct_index(d: &[u32], b: usize, entry: usize, loc: usize) -> usize {
    let s = (d[0] * d[1]) as usize;
    b * (d[2] as usize + 4) * s + entry * s + loc
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Modern heads, phase 1: one thread per image runs the SimOTA / task-aligned assignment into `scratch`.
    #[kernel]
    pub fn modern_yolo_assign(
        output: &[f32], truth: &[f32], mut scratch: DisjointSlice<f32>, mut status: DisjointSlice<f32>,
        d: &[u32], p: &[f32], batches: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batches { return; }
        let (w, h) = (d[0] as usize, d[1] as usize);
        let s = w * h;
        let (classes, kind, max_boxes, ts) = (d[2] as usize, d[4], d[5] as usize, d[6] as usize);
        let out = output.as_ptr();
        let slots = unsafe { scratch.as_mut_ptr().add(b * ms_per_image(d)) };
        let cq = unsafe { slots.add(s * MS_SLOT) };
        let cand = unsafe { cq.add(s * classes) };
        let mut i = 0;
        while i < s {
            unsafe {
                let q = slots.add(i * MS_SLOT);
                *q = -1.0;
                *q.add(1) = -1.0;
                *q.add(8) = -1.0;
                *q.add(9) = MS_BIG;
            }
            i += 1;
        }
        let tb = b * max_boxes * ts;
        let mut err = 0.0f32;
        let mut t = 0;
        while t < max_boxes {
            let o = tb + t * ts;
            let tx = truth[o];
            if tx == 0.0 { break; }
            let (ty, tw, th) = (truth[o + 1], truth[o + 2], truth[o + 3]);
            if !(tx.is_finite() && ty.is_finite() && tw.is_finite() && th.is_finite()
                && tx > 0.0 && tx <= 1.0 && ty >= 0.0 && ty <= 1.0 && tw > 0.0 && th > 0.0)
            {
                err = 1.0;
                break;
            }
            let class_id = truth[o + 4] as usize;
            if class_id >= classes { err = 1.0; break; }
            if kind != 0 && p[7] > 0.0 && (tw * (w as f32) < p[7] || th * (h as f32) < p[7]) {
                t += 1;
                continue;
            }
            let tbox: YBox = (tx, ty, tw, th);
            let mut n = 0usize;
            let mut loc = 0;
            while loc < s {
                let center = ms_in_center(w, h, loc, tbox);
                let ok = if kind == 0 { center } else { ms_in_box(w, h, loc, tbox) || center };
                if ok {
                    let pred = ms_box(out, d, b, loc);
                    let iou = yolo_finite(ms_iou(pred, tbox)).max(0.0).min(1.0);
                    let obj = ms_clamped(out, ms_entry(d, b, d[3] as usize, loc));
                    let cls = ms_clamped(out, ms_entry(d, b, d[3] as usize + 1 + class_id, loc));
                    let (score, cost) = if kind == 0 {
                        let cost = -ms_ln((cls * obj).max(MS_EPS).sqrt()) + 3.0 * -ms_ln(iou + MS_EPS);
                        (1.0 / (1.0 + cost), cost)
                    } else {
                        let score = (cls * obj).max(MS_EPS) * iou.max(MS_EPS).powf(6.0);
                        (score, -ms_ln(score + MS_EPS))
                    };
                    if score.is_finite() && cost.is_finite() {
                        unsafe {
                            let c = cand.add(n * MS_CAND);
                            *c = loc as f32;
                            *c.add(1) = iou;
                            *c.add(2) = score;
                            *c.add(3) = cost;
                            *c.add(4) = 0.0;
                        }
                        n += 1;
                    }
                }
                loc += 1;
            }
            if n == 0 { t += 1; continue; }
            let truth_area = tw * th;
            let keep: usize;
            if kind == 0 {
                // SimOTA: k = max(1, int(sum of the top-10 IoUs)), then the k cheapest candidates.
                let top = n.min(10);
                let mut sum_iou = 0.0f32;
                let mut r = 0;
                while r < top {
                    let (mut best, mut at) = (-1.0f32, n);
                    i = 0;
                    while i < n {
                        let c = unsafe { cand.add(i * MS_CAND) };
                        if unsafe { *c.add(4) } == 0.0 && unsafe { *c.add(1) } > best {
                            best = unsafe { *c.add(1) };
                            at = i;
                        }
                        i += 1;
                    }
                    unsafe { *cand.add(at * MS_CAND + 4) = 1.0; }
                    sum_iou += best;
                    r += 1;
                }
                i = 0;
                while i < n { unsafe { *cand.add(i * MS_CAND + 4) = 0.0; } i += 1; }
                keep = (sum_iou as usize).max(1).min(n);
            } else {
                keep = n.min(13);
            }
            let mut r = 0;
            while r < keep {
                let (mut best, mut at) = (if kind == 0 { MS_BIG } else { -1.0f32 }, n);
                i = 0;
                while i < n {
                    let c = unsafe { cand.add(i * MS_CAND) };
                    let (used, cscore, ccost) = unsafe { (*c.add(4), *c.add(2), *c.add(3)) };
                    // YOLOX keeps the lowest cost, the task-aligned heads the highest score.
                    let key = if kind == 0 { ccost } else { cscore };
                    let better = if kind == 0 { key < best } else { key > best };
                    if used == 0.0 && better {
                        best = key;
                        at = i;
                    }
                    i += 1;
                }
                if at == n { break; }
                let c = unsafe { cand.add(at * MS_CAND) };
                unsafe { *c.add(4) = 1.0; }
                let (cl, ciou, cscore, ccost) = unsafe { (*c as usize, *c.add(1), *c.add(2), *c.add(3)) };
                let q = unsafe { slots.add(cl * MS_SLOT) };
                if kind == 0 {
                    if unsafe { *q } < 0.0 || ccost < -unsafe { *q.add(2) } {
                        unsafe {
                            *q = t as f32;
                            *q.add(1) = class_id as f32;
                            *q.add(2) = -ccost;
                            *q.add(3) = ciou;
                            *q.add(4) = tx;
                            *q.add(5) = ty;
                            *q.add(6) = tw;
                            *q.add(7) = th;
                        }
                    }
                } else {
                    let quality = ciou.max(0.05).min(1.0);
                    unsafe {
                        let slot_cq = cq.add(cl * classes + class_id);
                        if quality > *slot_cq { *slot_cq = quality; }
                    }
                    let smaller = truth_area < unsafe { *q.add(9) } - MS_EPS;
                    let tied = (truth_area - unsafe { *q.add(9) }).abs() <= MS_EPS && cscore > unsafe { *q.add(8) };
                    if smaller || tied {
                        unsafe {
                            *q.add(9) = truth_area;
                            *q.add(8) = cscore;
                            *q = t as f32;
                            *q.add(1) = class_id as f32;
                            *q.add(2) = quality;
                            *q.add(3) = ciou;
                            *q.add(4) = tx;
                            *q.add(5) = ty;
                            *q.add(6) = tw;
                            *q.add(7) = th;
                        }
                    }
                }
                r += 1;
            }
            t += 1;
        }
        if kind == 0 {
            // YOLOX trains with hard class labels.
            i = 0;
            while i < s {
                let q = unsafe { slots.add(i * MS_SLOT) };
                if unsafe { *q } >= 0.0 { unsafe { *q.add(2) = 1.0; } }
                i += 1;
            }
        } else if kind == 2 {
            // YOLO-NAS: down-weight duplicates of one truth by IoU rank.
            let mut tr = 0;
            while tr < max_boxes {
                let mut rank = 0usize;
                loop {
                    let (mut best, mut at) = (-1.0f32, s);
                    i = 0;
                    while i < s {
                        let q = unsafe { slots.add(i * MS_SLOT) };
                        if unsafe { *q } == tr as f32 && unsafe { *q.add(10) } == 0.0 && unsafe { *q.add(3) } > best {
                            best = unsafe { *q.add(3) };
                            at = i;
                        }
                        i += 1;
                    }
                    if at == s { break; }
                    let q = unsafe { slots.add(at * MS_SLOT) };
                    unsafe {
                        *q.add(10) = 1.0;
                        *q.add(2) = (best * (-0.35f32 * rank as f32).exp()).max(0.02).min(1.0);
                    }
                    rank += 1;
                }
                tr += 1;
            }
        }
        // SAFETY: one thread owns each image's status slot.
        unsafe { *status.as_mut_ptr().add(b) = err; }
    }

    /// Modern heads, phase 2: one thread per cell turns the assignment into deltas and per-cell stats.
    #[kernel]
    pub fn modern_yolo_delta(
        output: &[f32], truth: &[f32], scratch: &[f32], mut delta: DisjointSlice<f32>,
        mut stats: DisjointSlice<f32>, d: &[u32], p: &[f32], batches: usize,
    ) {
        let g = thread::index_1d().get();
        let (w, h) = (d[0] as usize, d[1] as usize);
        let s = w * h;
        if g >= batches * s { return; }
        let (b, loc) = (g / s, g % s);
        let (classes, kind, max_boxes, ts) = (d[2] as usize, d[4], d[5] as usize, d[6] as usize);
        let coords = d[3] as usize;
        let out = output.as_ptr();
        let del = delta.as_mut_ptr();
        let base = b * ms_per_image(d);
        let q = base + loc * MS_SLOT;
        let obj_idx = ms_entry(d, b, coords, loc);
        let obj_out = yolo_finite(unsafe { *out.add(obj_idx) });
        let truth_index = scratch[q];
        if truth_index < 0.0 {
            let pred = ms_box(out, d, b, loc);
            let tb = b * max_boxes * ts;
            let mut best = 0.0f32;
            let mut t = 0;
            while t < max_boxes {
                let o = tb + t * ts;
                if truth[o] == 0.0 { break; }
                let iou = yolo_finite(ms_iou(pred, (truth[o], truth[o + 1], truth[o + 2], truth[o + 3]))).max(0.0).min(1.0);
                best = best.max(iou);
                t += 1;
            }
            let target = if best > p[3] {
                if p[6] != 0.0 { best } else { return; }
            } else {
                0.0
            };
            let dv = p[1] * (target.max(0.0).min(1.0) - obj_out);
            if dv.is_finite() { unsafe { *del.add(obj_idx) += dv; } }
            return;
        }
        let class_id = scratch[q + 1] as usize;
        let quality = scratch[q + 2].max(0.0).min(1.0);
        let iou = scratch[q + 3];
        let tbox: YBox = (scratch[q + 4], scratch[q + 5], scratch[q + 6], scratch[q + 7]);
        let obj_target = if kind == 0 { 1.0 } else { quality };
        let dv = p[1] * (obj_target.max(0.0).min(1.0) - obj_out);
        if dv.is_finite() { unsafe { *del.add(obj_idx) += dv; } }
        let use_vfl = p[8] > 0.0 && kind != 0;
        let cq_base = base + s * MS_SLOT + loc * classes;
        let mut c = 0;
        while c < classes {
            let raw = if kind == 0 {
                if c == class_id { quality } else { 0.0 }
            } else {
                scratch[cq_base + c].max(0.0).min(1.0)
            };
            let mut target = raw;
            if p[5] != 0.0 && !use_vfl { target = target * (1.0 - p[5]) + 0.5 * p[5]; }
            let idx = ms_entry(d, b, coords + 1 + c, loc);
            let pr = yolo_finite(unsafe { *out.add(idx) }).max(0.0).min(1.0);
            let mut dc = p[0] * (target - pr);
            if use_vfl {
                dc *= if target > 0.0 { target } else { 0.75 * pr.powf(p[8]) };
            }
            if dc.is_finite() { unsafe { *del.add(idx) += dc; } }
            c += 1;
        }
        let normalizer = p[2];
        if kind != 0 {
            let bins = (coords / 4).max(2);
            let max_bin = bins - 1;
            let (col, row) = ((loc % w) as f32, (loc / w) as f32);
            let (cx, cy) = (col + 0.5, row + 0.5);
            let mut side = 0;
            while side < 4 {
                let left = (tbox.0 - 0.5 * tbox.2) * w as f32;
                let top = (tbox.1 - 0.5 * tbox.3) * h as f32;
                let right = (tbox.0 + 0.5 * tbox.2) * w as f32;
                let bottom = (tbox.1 + 0.5 * tbox.3) * h as f32;
                let dist = if side == 0 { cx - left } else if side == 1 { cy - top } else if side == 2 { right - cx } else { bottom - cy };
                let dist = yolo_finite(dist).max(0.0).min(max_bin as f32 - 1e-4);
                let left_bin = (dist.floor() as usize).min(max_bin);
                let right_bin = (left_bin + 1).min(max_bin);
                let right_weight = dist - left_bin as f32;
                let left_weight = 1.0 - right_weight;
                let start = ms_entry(d, b, side * bins, loc);
                let mut m = f32::NEG_INFINITY;
                let mut k = 0;
                while k < bins {
                    m = m.max(yolo_finite(unsafe { *out.add(start + k * s) }));
                    k += 1;
                }
                let mut denom = 0.0f32;
                k = 0;
                while k < bins {
                    denom += (yolo_finite(unsafe { *out.add(start + k * s) }) - m).max(-60.0).min(60.0).exp();
                    k += 1;
                }
                let denom = denom.max(MS_EPS);
                k = 0;
                while k < bins {
                    let prob = (yolo_finite(unsafe { *out.add(start + k * s) }) - m).max(-60.0).min(60.0).exp() / denom;
                    let mut target = 0.0f32;
                    if k == left_bin { target += left_weight; }
                    if k == right_bin { target += right_weight; }
                    let dk = normalizer * (target - prob);
                    if dk.is_finite() { unsafe { *del.add(start + k * s) += dk; } }
                    k += 1;
                }
                side += 1;
            }
        } else {
            let idx = ms_entry(d, b, 0, loc);
            let (col, row) = ((loc % w) as f32, (loc / w) as f32);
            let targets = [
                tbox.0 * w as f32 - col,
                tbox.1 * h as f32 - row,
                ms_ln(tbox.2 * w as f32),
                ms_ln(tbox.3 * h as f32),
            ];
            let mut e = 0;
            while e < 4 {
                let mut dv = normalizer * (targets[e] - yolo_finite(unsafe { *out.add(idx + e * s) }));
                if p[4] < 3.4028235e38 { dv = dv.max(-p[4]).min(p[4]); }
                if dv.is_finite() { unsafe { *del.add(idx + e * s) += dv; } }
                e += 1;
            }
        }
        // SAFETY: one thread owns each cell's stats slot.
        unsafe {
            let st = stats.as_mut_ptr().add(g * 3);
            *st = iou;
            *st.add(1) = 1.0 - iou;
            *st.add(2) = 1.0;
        }
    }

    /// Modern heads, phase 3: per-image sums of the cell stats and of the squared delta.
    #[kernel]
    pub fn modern_yolo_reduce(
        delta: &[f32], stats: &[f32], status: &[f32], mut metrics: DisjointSlice<f32>, d: &[u32], batches: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batches { return; }
        let s = (d[0] * d[1]) as usize;
        let outputs = (d[3] as usize + d[2] as usize + 1) * s;
        let (mut iou, mut iou_loss, mut positives, mut squared) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut i = 0;
        while i < s {
            let at = (b * s + i) * 3;
            iou += stats[at];
            iou_loss += stats[at + 1];
            positives += stats[at + 2];
            i += 1;
        }
        i = 0;
        while i < outputs {
            let v = delta[b * outputs + i];
            squared += v * v;
            i += 1;
        }
        // SAFETY: one thread owns each image's metrics record.
        unsafe {
            let record = metrics.as_mut_ptr().add(b * 5);
            *record = iou;
            *record.add(1) = iou_loss;
            *record.add(2) = positives;
            *record.add(3) = squared;
            *record.add(4) = status[b];
        }
    }

    /// Inference decode of the distribution heads (`[ppyoloe]`, `[yolonas]`): one thread per cell writes
    /// `[batch, 5 + classes, H, W]` with the final centre-size box in channels 0..4 and the activated
    /// objectness and class planes copied behind it. `d` is the layout of `modern_yolo_assign`.
    #[kernel]
    pub fn modern_yolo_predecode(
        output: &[f32], mut decoded: DisjointSlice<f32>, d: &[u32], batches: usize,
    ) {
        let g = thread::index_1d().get();
        let s = (d[0] * d[1]) as usize;
        if g >= batches * s { return; }
        let (b, loc) = (g / s, g % s);
        let (classes, coords) = (d[2] as usize, d[3] as usize);
        let out = output.as_ptr();
        let dst = decoded.as_mut_ptr();
        let pred = ms_box(out, d, b, loc);
        let base = b * (5 + classes) * s + loc;
        // SAFETY: each cell owns its column of the decoded tensor.
        unsafe {
            *dst.add(base) = pred.0;
            *dst.add(base + s) = pred.1;
            *dst.add(base + 2 * s) = pred.2;
            *dst.add(base + 3 * s) = pred.3;
            let mut e = 0;
            while e < classes + 1 {
                *dst.add(base + (4 + e) * s) = *out.add(ms_entry(d, b, coords + e, loc));
                e += 1;
            }
        }
    }

    /// CenterNet, phase 1: one thread per image draws the max-blended Gaussian heatmap targets and weights.
    /// `targets` holds `[batch * classes * H * W]` targets then the same count of weights.
    #[kernel]
    pub fn centernet_targets(
        truth: &[f32], mut targets: DisjointSlice<f32>, mut status: DisjointSlice<f32>,
        d: &[u32], p: &[f32], batches: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batches { return; }
        let (w, h) = (d[0] as usize, d[1] as usize);
        let s = w * h;
        let (classes, max_boxes, ts) = (d[2] as usize, d[3] as usize, d[4] as usize);
        let image = classes * s;
        let total = batches * image;
        let tgt = unsafe { targets.as_mut_ptr().add(b * image) };
        let wgt = unsafe { targets.as_mut_ptr().add(total + b * image) };
        let mut i = 0;
        while i < image { unsafe { *wgt.add(i) = 1.0; } i += 1; }
        let (net_w, net_h) = (d[6] as f32, d[7] as f32);
        let center_min = d[8] as i32;
        let tb = b * max_boxes * ts;
        let mut err = 0.0f32;
        let mut t = 0;
        while t < max_boxes {
            let o = tb + t * ts;
            let x = truth[o];
            if x == 0.0 { break; }
            let (y, tw, th) = (truth[o + 1], truth[o + 2], truth[o + 3]);
            let class_id = truth[o + 4] as usize;
            if class_id >= classes { err = 1.0; break; }
            if !(x.is_finite() && y.is_finite() && tw.is_finite() && th.is_finite())
                || x <= 0.0 || x > 1.0 || y <= 0.0 || y > 1.0 || tw <= 0.0 || th <= 0.0
            {
                err = 1.0;
                break;
            }
            let max_side_px = (tw * net_w).max(th * net_h);
            if max_side_px < p[2] || max_side_px >= p[3] { t += 1; continue; }
            let gx = x * w as f32;
            let gy = y * h as f32;
            let ix = (gx.floor() as i32).max(0).min(w as i32 - 1);
            let iy = (gy.floor() as i32).max(0).min(h as i32 - 1);
            let w_cells = (tw * w as f32).max(1e-6);
            let h_cells = (th * h as f32).max(1e-6);
            let small = ct_small_weight(p, (tw * net_w).min(th * net_h));
            let base_radius = center_min.max(ct_gaussian_radius(w_cells, h_cells, p[4]));
            let (mut rx, mut ry) = (base_radius, base_radius);
            if d[9] != 0 {
                let aspect_x = (w_cells / h_cells).sqrt();
                let aspect_y = 1.0 / aspect_x;
                rx = ((base_radius as f32 * aspect_x).round() as i32).max(center_min);
                ry = ((base_radius as f32 * aspect_y).round() as i32).max(center_min);
                // ceil(v) as -floor(-v), the safety cap on elongated labels.
                rx = rx.min(center_min.max((-(-(w_cells * 2.0 + 1.0)).floor()) as i32));
                ry = ry.min(center_min.max((-(-(h_cells * 2.0 + 1.0)).floor()) as i32));
            }
            let (rx, ry) = (rx.max(0), ry.max(0));
            let sigma_x = ((2.0 * rx as f32 + 1.0) / 6.0).max(1e-6);
            let sigma_y = ((2.0 * ry as f32 + 1.0) / 6.0).max(1e-6);
            let inv_x = 1.0 / (2.0 * sigma_x * sigma_x);
            let inv_y = 1.0 / (2.0 * sigma_y * sigma_y);
            let cls_base = class_id * s;
            let mut dy = -ry;
            while dy <= ry {
                let yy = iy + dy;
                if yy >= 0 && yy < h as i32 {
                    let mut dx = -rx;
                    while dx <= rx {
                        let xx = ix + dx;
                        if xx >= 0 && xx < w as i32 {
                            let value = (-((dx * dx) as f32 * inv_x + (dy * dy) as f32 * inv_y)).exp();
                            let at = cls_base + (yy as usize) * w + xx as usize;
                            unsafe {
                                if value > *tgt.add(at) {
                                    *tgt.add(at) = value;
                                    if small > *wgt.add(at) { *wgt.add(at) = small; }
                                }
                            }
                        }
                        dx += 1;
                    }
                }
                dy += 1;
            }
            t += 1;
        }
        // SAFETY: one thread owns each image's status slot.
        unsafe { *status.as_mut_ptr().add(b) = err; }
    }

    /// CenterNet, phase 2: one thread per heatmap element writes the focal-loss delta and loss.
    #[kernel]
    pub fn centernet_focal(
        output: &[f32], targets: &[f32], mut delta: DisjointSlice<f32>, mut losses: DisjointSlice<f32>,
        d: &[u32], p: &[f32], batches: usize,
    ) {
        let g = thread::index_1d().get();
        let s = (d[0] * d[1]) as usize;
        let classes = d[2] as usize;
        let image = classes * s;
        if g >= batches * image { return; }
        let (b, rem) = (g / image, g % image);
        let (c, loc) = (rem / s, rem % s);
        let outi = ct_index(d, b, c, loc);
        let y = targets[g].max(0.0).min(1.0);
        let weight = targets[batches * image + g].max(0.0);
        let pr = output[outi].max(1e-6).min(1.0 - 1e-6);
        let (grad, loss) = if y >= 0.999 {
            let focal = (1.0 - pr).powf(p[5]);
            (weight * focal * (1.0 - pr), p[7] * weight * focal * -ct_safe_log(pr))
        } else {
            let neg = (1.0 - y).powf(p[6]);
            let focal = pr.powf(p[5]);
            (-weight * neg * focal * pr, p[7] * weight * neg * focal * -ct_safe_log(1.0 - pr))
        };
        // SAFETY: each element owns its delta entry and loss slot.
        unsafe {
            *delta.as_mut_ptr().add(outi) = ct_clamp_delta(p[7] * grad, p[10]);
            *losses.as_mut_ptr().add(g) = loss;
        }
    }

    /// CenterNet, phase 3: one thread per image regresses size and offset at the true centres and
    /// writes the per-image metrics `[hm, wh, off, iou, positives, invalid]`.
    #[kernel]
    pub fn centernet_regress(
        output: &[f32], truth: &[f32], losses: &[f32], status: &[f32], mut delta: DisjointSlice<f32>,
        mut metrics: DisjointSlice<f32>, d: &[u32], p: &[f32], batches: usize,
    ) {
        let b = thread::index_1d().get();
        if b >= batches { return; }
        let (w, h) = (d[0] as usize, d[1] as usize);
        let s = w * h;
        let (classes, max_boxes, ts) = (d[2] as usize, d[3] as usize, d[4] as usize);
        let (net_w, net_h) = (d[6] as f32, d[7] as f32);
        let del = delta.as_mut_ptr();
        let (mut wh_loss, mut off_loss, mut total_iou, mut positives) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let tb = b * max_boxes * ts;
        let mut t = 0;
        while t < max_boxes && status[b] == 0.0 {
            let o = tb + t * ts;
            let x = truth[o];
            if x == 0.0 { break; }
            let (y, tw, th) = (truth[o + 1], truth[o + 2], truth[o + 3]);
            let class_id = truth[o + 4] as usize;
            let max_side_px = (tw * net_w).max(th * net_h);
            if class_id >= classes || max_side_px < p[2] || max_side_px >= p[3] { t += 1; continue; }
            let gx = x * w as f32;
            let gy = y * h as f32;
            let ix = (gx.floor() as i32).max(0).min(w as i32 - 1);
            let iy = (gy.floor() as i32).max(0).min(h as i32 - 1);
            let loc = (iy as usize) * w + ix as usize;
            let weight = ct_small_weight(p, (tw * net_w).min(th * net_h));
            let target_w = ct_safe_log((tw * w as f32).max(1e-6));
            let target_h = ct_safe_log((th * h as f32).max(1e-6));
            let target_ox = gx - ix as f32 - 0.5;
            let target_oy = gy - iy as f32 - 0.5;
            let raw_w = output[ct_index(d, b, classes, loc)];
            let raw_h = output[ct_index(d, b, classes + 1, loc)];
            let raw_ox = output[ct_index(d, b, classes + 2, loc)];
            let raw_oy = output[ct_index(d, b, classes + 3, loc)];
            let ew = yolo_finite(raw_w) - target_w;
            let eh = yolo_finite(raw_h) - target_h;
            let eox = yolo_finite(raw_ox) - target_ox;
            let eoy = yolo_finite(raw_oy) - target_oy;
            unsafe {
                *del.add(ct_index(d, b, classes, loc)) += ct_clamp_delta(-p[8] * weight * ct_smooth_l1_grad(ew), p[10]);
                *del.add(ct_index(d, b, classes + 1, loc)) += ct_clamp_delta(-p[8] * weight * ct_smooth_l1_grad(eh), p[10]);
                *del.add(ct_index(d, b, classes + 2, loc)) += ct_clamp_delta(-p[9] * weight * ct_smooth_l1_grad(eox), p[10]);
                *del.add(ct_index(d, b, classes + 3, loc)) += ct_clamp_delta(-p[9] * weight * ct_smooth_l1_grad(eoy), p[10]);
            }
            wh_loss += p[8] * weight * (ct_smooth_l1_loss(ew) + ct_smooth_l1_loss(eh));
            off_loss += p[9] * weight * (ct_smooth_l1_loss(eox) + ct_smooth_l1_loss(eoy));
            // Decoded box at the centre cell, as `decode_centernet_box`.
            let ox = yolo_finite(raw_ox).max(-0.75).min(0.75);
            let oy = yolo_finite(raw_oy).max(-0.75).min(0.75);
            let px = yolo_finite(((loc % w) as f32 + 0.5 + ox) / w as f32);
            let py = yolo_finite(((loc / w) as f32 + 0.5 + oy) / h as f32);
            let pw = yolo_finite(yolo_finite(raw_w).max(-20.0).min(20.0).exp() / w as f32).max(1e-6);
            let ph = yolo_finite(yolo_finite(raw_h).max(-20.0).min(20.0).exp() / h as f32).max(1e-6);
            let inter_w = (px + 0.5 * pw).min(x + 0.5 * tw) - (px - 0.5 * pw).max(x - 0.5 * tw);
            let inter_h = (py + 0.5 * ph).min(y + 0.5 * th) - (py - 0.5 * ph).max(y - 0.5 * th);
            let inter = inter_w.max(0.0) * inter_h.max(0.0);
            let union = pw.max(0.0) * ph.max(0.0) + tw.max(0.0) * th.max(0.0) - inter;
            if union > 0.0 { total_iou += inter / union; }
            positives += 1.0;
            t += 1;
        }
        let image = classes * s;
        let mut hm = 0.0f32;
        let mut i = 0;
        while i < image { hm += losses[b * image + i]; i += 1; }
        // SAFETY: one thread owns each image's metrics record.
        unsafe {
            let record = metrics.as_mut_ptr().add(b * 6);
            *record = hm;
            *record.add(1) = wh_loss;
            *record.add(2) = off_loss;
            *record.add(3) = total_iou;
            *record.add(4) = positives;
            *record.add(5) = status[b];
        }
    }

    /// Classic YOLO loss, phase 1: one thread per prediction writes the background
    /// objectness delta, and the `truth_thresh` positives (flagged in `flags`).
    #[kernel]
    pub fn yolo_loss_background(
        output: &[f32], truth: &[f32], mut delta: DisjointSlice<f32>, mut flags: DisjointSlice<f32>,
        d: &[u32], p: &[f32], mask: &[u32], biases: &[f32], batches: usize,
    ) {
        let g = thread::index_1d().get();
        let (width, spatial) = (d[0] as usize, (d[0] * d[1]) as usize);
        let (anchors, classes) = (d[2] as usize, d[4] as usize);
        let per_image = anchors * spatial;
        if g >= batches * per_image { return; }
        let (batch, prediction) = (g / per_image, g % per_image);
        let (anchor, location) = (prediction / spatial, prediction % spatial);
        let (column, row) = (location % width, location / width);
        let out = output.as_ptr();
        let del = delta.as_mut_ptr();
        let box_index = yolo_entry(d, batch, anchor, location, 0);
        let obj_index = yolo_entry(d, batch, anchor, location, 4);
        let class_index = yolo_entry(d, batch, anchor, location, 5);
        let truth_count = yolo_truth_count(truth, batch, d);
        let base = batch * d[7] as usize;
        let size = d[6] as usize;
        let pred = yolo_decode(out, biases, mask[anchor] as usize, box_index, column, row, d);
        let mut class_match = false;
        let mut c = 0;
        while c < classes {
            if unsafe { *out.add(class_index + c * spatial) } > 0.25 { class_match = true; }
            c += 1;
        }
        let mut best_match_iou = 0.0f32;
        let mut t = 0;
        while class_match && t < truth_count {
            best_match_iou = best_match_iou.max(yolo_iou(pred, yolo_truth_box(truth, base + t * size)));
            t += 1;
        }
        unsafe {
            // The CPU path zeroes a non-finite objectness while scanning truths; read it as 0 here.
            let obj = yolo_finite(*out.add(obj_index));
            let background = -p[2] * obj;
            if best_match_iou > p[0] {
                let delta_obj = p[2] * (best_match_iou - obj);
                *del.add(obj_index) = if d[12] != 0 { delta_obj.max(background) } else { 0.0 };
            } else {
                *del.add(obj_index) = background;
            }
        }
        // truth_thresh < 1: the truth with the highest IoU (first on ties) makes this prediction a positive.
        if p[5] < 1.0 && truth_count > 0 {
            let mut best_iou = 0.0f32;
            let mut best_t = 0;
            t = 0;
            while t < truth_count {
                let iou = yolo_iou(pred, yolo_truth_box(truth, base + t * size));
                if iou > best_iou { best_iou = iou; best_t = t; }
                t += 1;
            }
            if best_iou > p[5] {
                let target = yolo_truth_box(truth, base + best_t * size);
                let class_id = truth[base + best_t * size + 4] as usize;
                let multiplier = best_iou * best_iou;
                let obj = yolo_finite(unsafe { *out.add(obj_index) });
                unsafe {
                    *del.add(obj_index) = if d[12] != 0 { p[2] * (multiplier - obj) } else { p[2] * (1.0 - obj) };
                }
                yolo_assign_class(out, del, class_index, class_id, d);
                if d[12] != 0 {
                    unsafe {
                        let idx = class_index + class_id * spatial;
                        *del.add(idx) = multiplier - *out.add(idx);
                    }
                }
                let mut scratch = [0.0f32; 4];
                yolo_assign_box(target, out, del, biases, mask[anchor] as usize, box_index, column, row, d, p, &mut scratch);
                // SAFETY: one thread owns each prediction's flag.
                unsafe { *flags.as_mut_ptr().add(g) = 1.0; }
            }
        }
    }

    /// Classic YOLO loss, phase 2: one thread per image assigns the truths sequentially
    /// (later truths may rewrite earlier cells, as in the CPU loop).
    #[kernel]
    pub fn yolo_loss_positives(
        output: &[f32], truth: &[f32], mut delta: DisjointSlice<f32>, mut metrics: DisjointSlice<f32>,
        d: &[u32], p: &[f32], mask: &[u32], biases: &[f32], batches: usize,
    ) {
        let batch = thread::index_1d().get();
        if batch >= batches { return; }
        let (width, height) = (d[0] as i32, d[1] as i32);
        let (anchors, total_anchors) = (d[2] as usize, d[3] as usize);
        let out = output.as_ptr();
        let del = delta.as_mut_ptr();
        let truth_count = yolo_truth_count(truth, batch, d);
        let base = batch * d[7] as usize;
        let size = d[6] as usize;
        let mut m = [0.0f32; 4];
        let mut count = 0.0f32;
        let mut t = 0;
        while t < truth_count {
            let target = yolo_truth_box(truth, base + t * size);
            let class_id = truth[base + t * size + 4] as usize;
            let mut best_iou = 0.0f32;
            let mut best_anchor = 0usize;
            let shape = (0.0f32, 0.0f32, target.2, target.3);
            let mut a = 0;
            while a < total_anchors {
                let candidate = (0.0f32, 0.0f32, biases[2 * a] / d[9] as f32, biases[2 * a + 1] / d[10] as f32);
                let iou = yolo_iou(candidate, shape);
                if iou > best_iou { best_iou = iou; best_anchor = a; }
                a += 1;
            }
            let column = ((target.0 * width as f32) as i32).max(0).min(width - 1) as usize;
            let row = ((target.1 * height as f32) as i32).max(0).min(height - 1) as usize;
            let location = row * width as usize + column;
            // iou_thresh < 1: every other masked anchor whose shape fits the truth is assigned too.
            if p[4] < 1.0 {
                a = 0;
                while a < total_anchors {
                    let mut extra = 0;
                    while extra < anchors && mask[extra] as usize != a { extra += 1; }
                    if extra != anchors && a != best_anchor {
                        let candidate = (0.0f32, 0.0f32, biases[2 * a] / d[9] as f32, biases[2 * a + 1] / d[10] as f32);
                        if yolo_iou(candidate, shape) > p[4] {
                            yolo_assign_box(target, out, del, biases, a, yolo_entry(d, batch, extra, location, 0), column, row, d, p, &mut m);
                            yolo_assign_objectness(out, del, yolo_entry(d, batch, extra, location, 4), d, p);
                            yolo_assign_class(out, del, yolo_entry(d, batch, extra, location, 5), class_id, d);
                            count += 1.0;
                        }
                    }
                    a += 1;
                }
            }
            let mut masked = 0;
            while masked < anchors && mask[masked] as usize != best_anchor { masked += 1; }
            if masked != anchors {
                yolo_assign_box(target, out, del, biases, best_anchor, yolo_entry(d, batch, masked, location, 0), column, row, d, p, &mut m);
                yolo_assign_objectness(out, del, yolo_entry(d, batch, masked, location, 4), d, p);
                yolo_assign_class(out, del, yolo_entry(d, batch, masked, location, 5), class_id, d);
                count += 1.0;
            }
            t += 1;
        }
        // SAFETY: one thread owns each image's metrics record.
        unsafe {
            let record = metrics.as_mut_ptr().add(batch * 7);
            *record = m[0];
            *record.add(1) = m[1];
            *record.add(4) = count;
            *record.add(5) = m[2];
            *record.add(6) = m[3];
        }
    }

    /// Classic YOLO loss, phase 3 (iou_thresh < 1): average each cell's box deltas over its classes.
    #[kernel]
    pub fn yolo_loss_average(mut delta: DisjointSlice<f32>, d: &[u32], batches: usize) {
        let g = thread::index_1d().get();
        let spatial = (d[0] * d[1]) as usize;
        let (anchors, classes) = (d[2] as usize, d[4] as usize);
        let per_image = anchors * spatial;
        if g >= batches * per_image { return; }
        let (batch, prediction) = (g / per_image, g % per_image);
        let (anchor, location) = (prediction / spatial, prediction % spatial);
        let del = delta.as_mut_ptr();
        // SAFETY: each prediction thread owns its four box slots and only reads its own class/objectness slots.
        unsafe {
            if *del.add(yolo_entry(d, batch, anchor, location, 4)) == 0.0 { return; }
            let mut positive = 0.0f32;
            let mut c = 0;
            while c < classes {
                if *del.add(yolo_entry(d, batch, anchor, location, 5 + c)) > 0.0 { positive += 1.0; }
                c += 1;
            }
            if positive > 0.0 {
                let mut e = 0;
                while e < 4 {
                    *del.add(yolo_entry(d, batch, anchor, location, e)) /= positive;
                    e += 1;
                }
            }
        }
    }

    /// Classic YOLO loss, phase 4: per-image squared-delta sums and the threshold-positive count.
    #[kernel]
    pub fn yolo_loss_reduce(delta: &[f32], flags: &[f32], mut metrics: DisjointSlice<f32>, d: &[u32], batches: usize) {
        let batch = thread::index_1d().get();
        if batch >= batches { return; }
        let spatial = (d[0] * d[1]) as usize;
        let (outputs, classes) = (d[8] as usize, d[4] as usize);
        let per_image = d[2] as usize * spatial;
        let (mut total, mut classification) = (0.0f32, 0.0f32);
        let mut i = 0;
        while i < outputs {
            let v = delta[batch * outputs + i];
            total += v * v;
            if (i / spatial) % (classes + 5) >= 4 { classification += v * v; }
            i += 1;
        }
        let mut flagged = 0.0f32;
        i = 0;
        while i < per_image { flagged += flags[batch * per_image + i]; i += 1; }
        // SAFETY: one thread owns each image's metrics record.
        unsafe {
            let record = metrics.as_mut_ptr().add(batch * 7);
            *record.add(2) = total;
            *record.add(3) = classification;
            *record.add(5) += flagged;
        }
    }


    /// Classic YOLO activation: channel entries are grouped by anchor.
    #[kernel]
    pub fn yolo_activate(input: &[f32], mut output: DisjointSlice<f32>, spatial: usize, entries: usize, scale_xy: f32, new_coords: bool) {
        let i = thread::index_1d().get();
        if i >= output.len() { return; }
        let entry = (i / spatial) % entries;
        let mut value = input[i];
        if !new_coords && entry != 2 && entry != 3 { value = 1.0 / (1.0 + (-value).exp()); }
        if entry < 2 { value = value * scale_xy - 0.5 * (scale_xy - 1.0); }
        // SAFETY: one thread owns each output element.
        unsafe { *output.as_mut_ptr().add(i) = value; }
    }

    /// CenterNet forward. NCHW class planes receive sigmoid; the four
    /// regression planes remain linear. Non-finite post-activation values are
    /// sanitized, with +inf class logits naturally saturating to one.
    #[kernel]
    pub fn centernet_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        classes: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= output.len() {
            return;
        }
        let channel = (index / spatial) % channels;
        let mut value = input[index];
        if channel < classes {
            value = 1.0 / (1.0 + (-value).exp());
        }
        if !value.is_finite() {
            value = 0.0;
        }
        // SAFETY: one thread owns each output element.
        unsafe {
            *output.as_mut_ptr().add(index) = value;
        }
    }

    /// Shared modern-YOLO activation in NCHW layout. Each element is sanitized;
    /// only the trailing objectness/class channels receive the bounded sigmoid.
    #[kernel]
    pub fn modern_yolo_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        classes: usize,
    ) {
        let index = thread::index_1d().get();
        if index >= output.len() {
            return;
        }
        let channel = (index / spatial) % channels;
        let class_begin = channels - classes - 1;
        let mut value = input[index];
        if !value.is_finite() {
            value = 0.0;
        }
        if channel >= class_begin {
            value = 1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp());
        }
        // SAFETY: one thread owns each output element.
        unsafe {
            *output.as_mut_ptr().add(index) = value;
        }
    }

    /// Per-pixel binary sigmoid or channel-strided softmax for segmentation.
    /// One thread owns a complete pixel's class row and writes disjoint NCHW
    /// slots across all classes.
    #[kernel]
    pub fn segmentation_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        channels: usize,
        spatial: usize,
        classes: usize,
        temperature: f32,
    ) {
        let pixel = thread::index_1d().get();
        let pixels = output.len() / channels;
        if pixel >= pixels {
            return;
        }
        let batch = pixel / spatial;
        let location = pixel % spatial;
        let base = batch * channels * spatial;
        if classes == 1 {
            let mut value = input[base + location];
            if !value.is_finite() {
                value = 0.0;
            }
            value = 1.0 / (1.0 + (-value).exp());
            // SAFETY: this pixel thread owns this binary output slot.
            unsafe {
                *output.as_mut_ptr().add(base + location) = value;
            }
            return;
        }
        let temp = if temperature > 1.0e-6 {
            temperature
        } else {
            1.0e-6
        };
        let mut maximum = f32::NEG_INFINITY;
        let mut class = 0usize;
        while class < classes {
            let value = input[base + class * spatial + location] / temp;
            if value > maximum {
                maximum = value;
            }
            class += 1;
        }
        let mut sum = 0.0f32;
        class = 0;
        while class < classes {
            let index = base + class * spatial + location;
            let value = (input[index] / temp - maximum).exp();
            unsafe {
                *output.as_mut_ptr().add(index) = value;
            }
            sum += value;
            class += 1;
        }
        class = 0;
        while class < classes {
            let index = base + class * spatial + location;
            unsafe {
                *output.as_mut_ptr().add(index) /= sum;
            }
            class += 1;
        }
    }

    /// One thread owns one segmentation sample: computes the per-element
    /// cross-entropy and `truth - prediction` delta, applies `lambda_ce` and
    /// the correct-pixel gate weight, and writes per-sample loss and suppressed
    /// pixel counts for the host reduction.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn segmentation_loss(
        output: &[f32],
        truth: &[f32],
        mut delta: DisjointSlice<f32>,
        mut partial_loss: DisjointSlice<f32>,
        mut suppressed: DisjointSlice<f32>,
        batch: usize,
        classes: usize,
        spatial: usize,
        softmax_loss: u32,
        scale: f32,
        gate_strength: f32,
        t_correct: f32,
    ) {
        let sample = thread::index_1d().get();
        if sample >= batch {
            return;
        }
        let base = sample * classes * spatial;
        let mut sample_loss = 0.0f32;
        let mut suppressed_count = 0usize;
        let mut location = 0usize;
        while location < spatial {
            let (is_match, p_y) = if classes == 1 {
                let p = output[base + location];
                let y = truth[base + location];
                ((p > 0.5) == (y > 0.5), if y > 0.5 { p } else { 1.0 - p })
            } else {
                let mut y_class = 0usize;
                let mut pred_class = 0usize;
                let mut max_truth = truth[base + location];
                let mut max_pred = output[base + location];
                let mut class = 1usize;
                while class < classes {
                    let index = base + class * spatial + location;
                    if truth[index] > max_truth {
                        max_truth = truth[index];
                        y_class = class;
                    }
                    if output[index] > max_pred {
                        max_pred = output[index];
                        pred_class = class;
                    }
                    class += 1;
                }
                (pred_class == y_class, output[base + y_class * spatial + location])
            };
            let gated = gate_strength > 0.0 && is_match && p_y >= t_correct;
            let weight = if gated { 1.0 - gate_strength } else { 1.0 };
            if gated {
                suppressed_count += 1;
            }
            let mut class = 0usize;
            while class < classes {
                let index = base + class * spatial + location;
                let t = truth[index];
                let p = output[index];
                let (error, d) = if softmax_loss != 0 {
                    let pc = if p < 1.0e-12 {
                        1.0e-12
                    } else if p > 1.0 {
                        1.0
                    } else {
                        p
                    };
                    (if t != 0.0 { -pc.ln() } else { 0.0 }, t - p)
                } else {
                    let pc = if p < 1.0e-12 {
                        1.0e-12
                    } else if p > 1.0 - 1.0e-12 {
                        1.0 - 1.0e-12
                    } else {
                        p
                    };
                    (
                        -t * pc.ln() - (1.0 - t) * (1.0 - pc).ln(),
                        t - p,
                    )
                };
                let mut l = error * scale * weight;
                if !l.is_finite() {
                    l = 0.0;
                }
                let d = d * scale * weight;
                unsafe {
                    *delta.as_mut_ptr().add(index) = d;
                }
                sample_loss += l;
                class += 1;
            }
            location += 1;
        }
        if let Some(slot) = partial_loss.get_mut(thread::index_1d()) {
            *slot = sample_loss;
        }
        if let Some(slot) = suppressed.get_mut(thread::index_1d()) {
            *slot = suppressed_count as f32;
        }
    }

    /// Applies the gate compensation and `max_delta` clip to a segmentation
    /// delta buffer in place.
    #[kernel]
    pub fn segmentation_scale_delta(
        mut delta: DisjointSlice<f32>,
        compensation: f32,
        max_delta: f32,
        clipping: u32,
    ) {
        let idx = thread::index_1d();
        if let Some(slot) = delta.get_mut(idx) {
            let mut value = *slot * compensation;
            if !value.is_finite() {
                value = 0.0;
            }
            if clipping != 0 {
                if value > max_delta {
                    value = max_delta;
                } else if value < -max_delta {
                    value = -max_delta;
                }
            }
            *slot = value;
        }
    }

    /// One thread owns a complete `[jev_ce]` group and writes its softmax row.
    #[kernel]
    pub fn jev_ce_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        groups: usize,
        candidates: usize,
        temperature: f32,
        channel_major: u32,
    ) {
        let group_index = thread::index_1d().get();
        if group_index >= output.len() / candidates {
            return;
        }
        let batch = group_index / groups;
        let group = group_index % groups;
        let base = batch * groups * candidates;
        let temp = if temperature > 1.0e-6 {
            temperature
        } else {
            1.0e-6
        };
        let index = |candidate: usize| {
            if channel_major != 0 {
                base + candidate * groups + group
            } else {
                base + group * candidates + candidate
            }
        };
        let mut maximum = f32::NEG_INFINITY;
        let mut candidate = 0usize;
        while candidate < candidates {
            let value = input[index(candidate)];
            if value > maximum {
                maximum = value;
            }
            candidate += 1;
        }
        let mut sum = 0.0f32;
        candidate = 0;
        while candidate < candidates {
            let value = ((input[index(candidate)] - maximum) / temp).exp();
            unsafe {
                *output.as_mut_ptr().add(index(candidate)) = value;
            }
            sum += value;
            candidate += 1;
        }
        candidate = 0;
        while candidate < candidates {
            unsafe {
                *output.as_mut_ptr().add(index(candidate)) /= sum;
            }
            candidate += 1;
        }
    }

    /// One thread owns one `[jev_ce]` group: computes its softmax,
    /// soft-target cross-entropy, and descent-direction delta.  `losses` and
    /// `active` have one slot per sample/group; masked empty groups leave the
    /// delta zero and mark `active` zero.
    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub fn jev_ce_loss(
        input: &[f32],
        truth: &[f32],
        mut delta: DisjointSlice<f32>,
        mut losses: DisjointSlice<f32>,
        mut active: DisjointSlice<f32>,
        batch: usize,
        groups: usize,
        candidates: usize,
        temperature: f32,
        normalize: u32,
        ignore_empty: u32,
        label_smoothing: f32,
        scale: f32,
        channel_major: u32,
    ) {
        let group_index = thread::index_1d().get();
        if group_index >= batch * groups {
            return;
        }
        let sample = group_index / groups;
        let group = group_index % groups;
        let base = sample * groups * candidates;
        let temp = if temperature > 1.0e-6 {
            temperature
        } else {
            1.0e-6
        };
        let index = |candidate: usize| {
            if channel_major != 0 {
                base + candidate * groups + group
            } else {
                base + group * candidates + candidate
            }
        };
        let mut target_sum = 0.0f32;
        let mut candidate = 0usize;
        while candidate < candidates {
            target_sum += truth[index(candidate)];
            candidate += 1;
        }
        let empty = target_sum <= 1.0e-8;
        if empty && ignore_empty != 0 {
            candidate = 0;
            while candidate < candidates {
                unsafe {
                    *delta.as_mut_ptr().add(index(candidate)) = 0.0;
                }
                candidate += 1;
            }
            if let Some(slot) = losses.get_mut(thread::index_1d()) {
                *slot = 0.0;
            }
            if let Some(slot) = active.get_mut(thread::index_1d()) {
                *slot = 0.0;
            }
            return;
        }

        let mut maximum = f32::NEG_INFINITY;
        candidate = 0;
        while candidate < candidates {
            let value = input[index(candidate)];
            if value > maximum {
                maximum = value;
            }
            candidate += 1;
        }
        let mut sum = 0.0f32;
        candidate = 0;
        while candidate < candidates {
            sum += ((input[index(candidate)] - maximum) / temp).exp();
            candidate += 1;
        }
        let eps = if label_smoothing > 0.0 {
            label_smoothing.min(0.999)
        } else {
            0.0
        };
        let uniform = 1.0 / candidates as f32;
        let mut loss = 0.0f32;
        candidate = 0;
        while candidate < candidates {
            let prob = ((input[index(candidate)] - maximum) / temp).exp() / sum;
            let mut target = if empty {
                uniform
            } else {
                truth[index(candidate)] / target_sum
            };
            if eps > 0.0 {
                target = (1.0 - eps) * target + eps * uniform;
            }
            loss += -target * prob.max(1.0e-12).ln();
            unsafe {
                *delta.as_mut_ptr().add(index(candidate)) = (target - prob) / temp;
            }
            candidate += 1;
        }
        if normalize != 0 {
            let mut sample_active = 0usize;
            let mut other_group = 0usize;
            while other_group < groups {
                let mut other_sum = 0.0f32;
                let mut other_candidate = 0usize;
                while other_candidate < candidates {
                    let other_index = if channel_major != 0 {
                        base + other_candidate * groups + other_group
                    } else {
                        base + other_group * candidates + other_candidate
                    };
                    other_sum += truth[other_index];
                    other_candidate += 1;
                }
                if other_sum > 1.0e-8 || ignore_empty == 0 {
                    sample_active += 1;
                }
                other_group += 1;
            }
            let inv = 1.0 / sample_active.max(1) as f32;
            candidate = 0;
            while candidate < candidates {
                unsafe {
                    *delta.as_mut_ptr().add(index(candidate)) *= inv;
                }
                candidate += 1;
            }
        }
        candidate = 0;
        while candidate < candidates {
            unsafe {
                *delta.as_mut_ptr().add(index(candidate)) *= scale;
            }
            candidate += 1;
        }
        if let Some(slot) = losses.get_mut(thread::index_1d()) {
            *slot = loss;
        }
        if let Some(slot) = active.get_mut(thread::index_1d()) {
            *slot = 1.0;
        }
    }

    /// One thread owns one sample row and writes its stable K-way softmax.
    #[kernel]
    pub fn laya_loss_forward(input: &[f32], mut output: DisjointSlice<f32>, max_options: usize) {
        let batch = thread::index_1d().get();
        let batches = output.len() / max_options;
        if batch >= batches {
            return;
        }
        let base = batch * max_options;
        let mut maximum = -1.0e300f64;
        let mut option = 0usize;
        while option < max_options {
            maximum = maximum.max(input[base + option] as f64);
            option += 1;
        }
        let mut sum = 0.0f64;
        option = 0;
        while option < max_options {
            sum += (input[base + option] as f64 - maximum).exp();
            option += 1;
        }
        option = 0;
        while option < max_options {
            let index = base + option;
            let probability = (input[index] as f64 - maximum).exp() / sum;
            unsafe {
                *output.as_mut_ptr().add(index) = probability as f32;
            }
            option += 1;
        }
    }

    /// Region forward. One thread owns a batch/location/anchor group and
    /// writes its interleaved entries after objectness sigmoid and optional
    /// class softmax.
    #[kernel]
    pub fn region_forward(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        spatial: usize,
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: u32,
    ) {
        let group = thread::index_1d().get();
        let groups_per_batch = spatial * anchors;
        if group >= output.len() / entries {
            return;
        }
        let batch = group / groups_per_batch;
        let rem = group % groups_per_batch;
        let location = rem / anchors;
        let anchor = rem % anchors;
        let channels = anchors * entries;
        let source = |entry: usize| {
            input[(batch * channels + anchor * entries + entry) * spatial + location]
        };
        let destination = (batch * spatial + location) * channels + anchor * entries;
        let mut entry = 0usize;
        while entry < entries {
            // SAFETY: this group exclusively owns its `entries`-wide output row.
            unsafe {
                *output.as_mut_ptr().add(destination + entry) = source(entry);
            }
            entry += 1;
        }
        let objectness = source(4);
        let sigmoid = 1.0 / (1.0 + (-objectness).exp());
        unsafe {
            *output.as_mut_ptr().add(destination + 4) = sigmoid;
        }
        if softmax_classes != 0 {
            let mut max_value = f32::NEG_INFINITY;
            entry = 0;
            while entry < classes {
                let value = source(5 + entry);
                if value > max_value {
                    max_value = value;
                }
                entry += 1;
            }
            let mut sum = 0.0f32;
            entry = 0;
            while entry < classes {
                let value = (source(5 + entry) - max_value).exp();
                unsafe {
                    *output.as_mut_ptr().add(destination + 5 + entry) = value;
                }
                sum += value;
                entry += 1;
            }
            if sum > 0.0 {
                entry = 0;
                while entry < classes {
                    unsafe {
                        *output.as_mut_ptr().add(destination + 5 + entry) /= sum;
                    }
                    entry += 1;
                }
            }
        }
    }

    /// Exact adjoint of `region_forward`, writing channel-major NCHW deltas.
    #[kernel]
    pub fn region_backward(
        output: &[f32],
        output_grad: &[f32],
        mut input_grad: DisjointSlice<f32>,
        spatial: usize,
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: u32,
    ) {
        let group = thread::index_1d().get();
        let groups_per_batch = spatial * anchors;
        if group >= input_grad.len() / entries {
            return;
        }
        let batch = group / groups_per_batch;
        let rem = group % groups_per_batch;
        let location = rem / anchors;
        let anchor = rem % anchors;
        let channels = anchors * entries;
        let base = (batch * spatial + location) * channels + anchor * entries;
        let mut dot = 0.0f32;
        if softmax_classes != 0 {
            let mut c = 0usize;
            while c < classes {
                dot += output[base + 5 + c] * output_grad[base + 5 + c];
                c += 1;
            }
        }
        let mut entry = 0usize;
        while entry < entries {
            let value = if entry == 4 {
                let p = output[base + 4];
                output_grad[base + 4] * p * (1.0 - p)
            } else if softmax_classes != 0 && entry >= 5 && entry < 5 + classes {
                let p = output[base + entry];
                p * (output_grad[base + entry] - dot)
            } else {
                output_grad[base + entry]
            };
            let target = (batch * channels + anchor * entries + entry) * spatial + location;
            // SAFETY: each batch/location/anchor/entry maps to one unique input slot.
            unsafe {
                *input_grad.as_mut_ptr().add(target) = value;
            }
            entry += 1;
        }
    }
}

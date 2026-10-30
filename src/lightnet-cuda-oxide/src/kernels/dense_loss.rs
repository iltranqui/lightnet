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

#[cuda_module]
pub mod kernels {
    use super::*;

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

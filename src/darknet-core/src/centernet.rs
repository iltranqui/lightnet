//! `[centernet]`: the CenterNet-style anchor-free detection head (Phase 2A of
//! `tasks/rust/layers.md`), ported from `Jev_laya:src-lib/centernet_layer.cpp`.
//!
//! No weights and no kernels, like the other modern heads.
//!
//! Output layout of the preceding `[convolutional]`:
//!
//! ```text
//! filters  = classes + 4
//! channels = [class_0 .. class_{C-1}, log_w, log_h, off_x, off_y]
//! ```
//!
//! channel-major with an `H * W` stride.  The forward copies the input, applies a
//! **sigmoid to the `classes` heatmap entries only** (the four regression entries
//! stay raw), then sanitises non-finite values.
//!
//! Decode (normalised):
//!
//! ```text
//! cx = (col + 0.5 + clamp(off_x, -0.75, 0.75)) / W
//! cy = (row + 0.5 + clamp(off_y, -0.75, 0.75)) / H
//! w  = exp(clamp(log_w, -20, 20)) / W
//! h  = exp(clamp(log_h, -20, 20)) / H
//! ```
//!
//! Training has three parts, and the header's contract is worth restating because
//! two of them are *not* what upstream CenterNet does:
//!
//! 1. **Heatmap focal loss** against an adaptive *anisotropic* Gaussian target
//!    (an axis-aligned ellipse stretched by the box aspect ratio) rather than the
//!    original isotropic disk, so a thin 2x12 px object does not get the same
//!    circular blob as a 6x6 px one.  Exact centres (`target >= 0.999`) get a
//!    positive gradient; the halo only suppresses nearby negatives.
//! 2. **Continuous tiny-object weighting**: `weight = clamp(small_ref_size /
//!    min(w_px, h_px), 1, small_boost)` replaces the old hard threshold.
//! 3. **Sparse regression** (smooth-L1 on `log_w`, `log_h`, `off_x`, `off_y`)
//!    at the true centre cells only, scaled by `wh_normalizer` /
//!    `off_normalizer` and the same tiny weight.
//!
//! `cost = (hm_loss + wh_loss + off_loss) / batch`.

use crate::model::CenterNetSpec;

/// C++ `ct_index`: the flat index of heatmap/regression `entry` at cell `loc`.
#[inline]
pub fn ct_index(l: &CenterNetLayout, batch: usize, entry: usize, loc: usize) -> usize {
    batch * l.outputs() + entry * l.spatial() + loc
}

/// The strides of one `[centernet]` feature map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CenterNetLayout {
    pub batch: usize,
    pub w: usize,
    pub h: usize,
    pub classes: usize,
}

impl CenterNetLayout {
    pub fn spatial(&self) -> usize {
        self.w * self.h
    }

    /// Entries per cell: `classes + 4`.
    pub fn entries(&self) -> usize {
        self.classes + 4
    }

    pub fn outputs(&self) -> usize {
        self.spatial() * self.entries()
    }

    pub fn len(&self) -> usize {
        self.batch * self.outputs()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn for_spec(batch: usize, w: usize, h: usize, spec: &CenterNetSpec) -> Self {
        Self { batch, w, h, classes: spec.classes }
    }
}

/// Rejected configuration or truth, matching C++'s `darknet_fatal_error`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CenterNetError(pub String);

impl std::fmt::Display for CenterNetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn clamp_delta(value: f32, max_delta: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    if max_delta >= f32::MAX {
        return value;
    }
    value.clamp(-max_delta, max_delta)
}

/// C++ `bounded_exp`.
fn bounded_exp(value: f32) -> f32 {
    finite_or_zero(value).clamp(-20.0, 20.0).exp()
}

/// C++ `safe_log`, with its 1e-9 floor (note: *not* the 1e-8 the modern heads use).
fn safe_log(value: f32) -> f32 {
    value.max(1e-9).ln()
}

/// C++ `smooth_l1_loss`.
fn smooth_l1_loss(error: f32) -> f32 {
    let abs_error = error.abs();
    if abs_error < 1.0 { 0.5 * error * error } else { abs_error - 0.5 }
}

/// C++ `smooth_l1_grad`.
fn smooth_l1_grad(error: f32) -> f32 {
    if error > 1.0 {
        1.0
    } else if error < -1.0 {
        -1.0
    } else {
        error
    }
}

/// C++ `gaussian_radius`: the CenterNet radius from the smallest of the three
/// overlap solutions, floored at 0.
pub fn gaussian_radius(width: f32, height: f32, min_overlap: f32) -> i64 {
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return 0;
    }
    let (w, h) = (width.max(1e-6), height.max(1e-6));
    let m = min_overlap.clamp(0.01, 0.99);

    let (a1, b1, c1) = (1.0f32, h + w, w * h * (1.0 - m) / (1.0 + m));
    let d1 = (b1 * b1 - 4.0 * a1 * c1).max(0.0);
    let r1 = (b1 + d1.sqrt()) / 2.0;

    let (a2, b2, c2) = (4.0f32, 2.0 * (h + w), (1.0 - m) * w * h);
    let d2 = (b2 * b2 - 4.0 * a2 * c2).max(0.0);
    let r2 = (b2 + d2.sqrt()) / 2.0;

    let (a3, b3, c3) = (4.0 * m, -2.0 * m * (h + w), (m - 1.0) * w * h);
    let d3 = (b3 * b3 - 4.0 * a3 * c3).max(0.0);
    let r3 = (b3 + d3.sqrt()) / (2.0 * a3);

    (r1.min(r2.min(r3)).floor() as i64).max(0)
}

/// C++ `continuous_small_weight`: the tiny-object multiplier.
pub fn continuous_small_weight(spec: &CenterNetSpec, min_side_px: f32) -> f32 {
    let max_weight = spec.small_boost.max(1.0);
    let ref_px = spec.small_ref_size.max(1.0);
    if !min_side_px.is_finite() || min_side_px <= 0.0 {
        return max_weight;
    }
    (ref_px / min_side_px.max(1e-3)).clamp(1.0, max_weight)
}

/// C++ `adaptive_gaussian_radii`: the ellipse axes, with the safety cap.
pub fn adaptive_gaussian_radii(
    spec: &CenterNetSpec,
    w_cells: f32,
    h_cells: f32,
) -> (i64, i64) {
    let base_radius = spec.center_min_radius.max(gaussian_radius(w_cells, h_cells, spec.gaussian_iou));
    if spec.anisotropic_gaussian == 0 {
        return (base_radius, base_radius);
    }
    let (safe_w, safe_h) = (w_cells.max(1e-6), h_cells.max(1e-6));
    let aspect_x = (safe_w / safe_h).sqrt();
    let aspect_y = 1.0 / aspect_x;
    let mut radius_x = (base_radius as f32 * aspect_x).round() as i64;
    let mut radius_y = (base_radius as f32 * aspect_y).round() as i64;
    radius_x = radius_x.max(spec.center_min_radius);
    radius_y = radius_y.max(spec.center_min_radius);
    // Safety cap so an extremely elongated label cannot write a huge blob.
    radius_x = radius_x.min(spec.center_min_radius.max((safe_w * 2.0 + 1.0).ceil() as i64));
    radius_y = radius_y.min(spec.center_min_radius.max((safe_h * 2.0 + 1.0).ceil() as i64));
    (radius_x, radius_y)
}

/// C++ `draw_center_gaussian`: a max-blended Gaussian blob plus a max-blended
/// per-cell weight.
pub fn draw_center_gaussian(
    l: &CenterNetLayout,
    target: &mut [f32],
    weight: &mut [f32],
    batch: usize,
    class_id: usize,
    cx: i64,
    cy: i64,
    radius_x: i64,
    radius_y: i64,
    small_weight: f32,
) {
    let base = (batch * l.classes + class_id) * l.spatial();
    let (rx, ry) = (radius_x.max(0), radius_y.max(0));
    let sigma_x = ((2.0 * rx as f32 + 1.0) / 6.0).max(1e-6);
    let sigma_y = ((2.0 * ry as f32 + 1.0) / 6.0).max(1e-6);
    let inv_2sx2 = 1.0 / (2.0 * sigma_x * sigma_x);
    let inv_2sy2 = 1.0 / (2.0 * sigma_y * sigma_y);
    let (w, h) = (l.w as i64, l.h as i64);
    for dy in -ry..=ry {
        let y = cy + dy;
        if y < 0 || y >= h {
            continue;
        }
        for dx in -rx..=rx {
            let x = cx + dx;
            if x < 0 || x >= w {
                continue;
            }
            let value = (-((dx * dx) as f32 * inv_2sx2 + (dy * dy) as f32 * inv_2sy2)).exp();
            let index = base + (y * l.w as i64 + x) as usize;
            if value > target[index] {
                target[index] = value;
                weight[index] = weight[index].max(small_weight);
            }
        }
    }
}

/// The forward pass: copy the input, sigmoid the heatmap entries, sanitise.
pub fn centernet_forward(input: &[f32], l: &CenterNetLayout) -> Vec<f32> {
    let mut output = input[..l.len().min(input.len())].to_vec();
    output.resize(l.len(), 0.0);
    for b in 0..l.batch {
        for c in 0..l.classes {
            let base = ct_index(l, b, c, 0);
            for v in output[base..base + l.spatial()].iter_mut() {
                *v = 1.0 / (1.0 + (-(*v)).exp());
            }
        }
    }
    for v in output.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    output
}

/// C++ `decode_centernet_box`.
pub fn decode_centernet_box(
    output: &[f32],
    l: &CenterNetLayout,
    batch: usize,
    loc: usize,
) -> (f32, f32, f32, f32) {
    let row = loc / l.w;
    let col = loc % l.w;
    let raw_w = output[ct_index(l, batch, l.classes, loc)];
    let raw_h = output[ct_index(l, batch, l.classes + 1, loc)];
    let raw_ox = output[ct_index(l, batch, l.classes + 2, loc)];
    let raw_oy = output[ct_index(l, batch, l.classes + 3, loc)];
    let ox = finite_or_zero(raw_ox).clamp(-0.75, 0.75);
    let oy = finite_or_zero(raw_oy).clamp(-0.75, 0.75);
    let x = finite_or_zero((col as f32 + 0.5 + ox) / l.w as f32);
    let y = finite_or_zero((row as f32 + 0.5 + oy) / l.h as f32);
    let w = bounded_exp(raw_w) / l.w as f32;
    let h = bounded_exp(raw_h) / l.h as f32;
    (x, y, finite_or_zero(w).max(1e-6), finite_or_zero(h).max(1e-6))
}

/// C++ `is_local_peak`: a 3x3 strict-maximum test on the heatmap
/// (`peak_nms = 0` disables it and accepts every cell).
pub fn is_local_peak(output: &[f32], l: &CenterNetLayout, batch: usize, class_id: usize, loc: usize) -> bool {
    let row = loc / l.w;
    let col = loc % l.w;
    let center = output[ct_index(l, batch, class_id, loc)];
    for dy in -1i64..=1 {
        let y = row as i64 + dy;
        if y < 0 || y >= l.h as i64 {
            continue;
        }
        for dx in -1i64..=1 {
            let x = col as i64 + dx;
            if x < 0 || x >= l.w as i64 || (dx == 0 && dy == 0) {
                continue;
            }
            let neighbor = output[ct_index(l, batch, class_id, (y * l.w as i64 + x) as usize)];
            if neighbor > center {
                return false;
            }
        }
    }
    true
}

/// Net-level knobs (`state.net` in C++, plus the head's own default when the net
/// does not carry them).
#[derive(Clone, Debug, PartialEq)]
pub struct CenterNetSettings {
    pub class_multipliers: Option<Vec<f32>>,
    pub class_map: Option<Vec<usize>>,
    pub truth_size: usize,
    /// `[net] width`, used for the pixel-space tiny-object and scale gates.
    pub net_width: usize,
    /// `[net] height`.
    pub net_height: usize,
}

impl Default for CenterNetSettings {
    fn default() -> Self {
        Self {
            class_multipliers: None,
            class_map: None,
            truth_size: 6,
            net_width: 1,
            net_height: 1,
        }
    }
}

/// What `[centernet]` computes during its forward.
#[derive(Clone, Debug, PartialEq)]
pub struct CenterNetBatch {
    pub delta: Vec<f32>,
    pub hm_loss: f32,
    pub wh_loss: f32,
    pub off_loss: f32,
    pub total_iou: f32,
    pub positives: usize,
    /// The class id claimed by each cell, for the decode path.
    pub class_ids: Vec<i32>,
    /// The tracking id (truth column 5) of each claimed cell.
    pub labels: Vec<i32>,
}

/// C++ `forward_centernet_layer`'s training half: build the heatmap targets,
/// apply the focal loss everywhere, then regress sparsely at the true centres.
pub fn centernet_train_batch(
    output: &[f32],
    truth: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    settings: &CenterNetSettings,
    b: usize,
) -> Result<CenterNetBatch, CenterNetError> {
    let spatial = l.spatial();
    let truths = spec.max_boxes * settings.truth_size;
    if truth.len() < truths * l.batch {
        return Err(CenterNetError(format!(
            "[centernet] truth needs {} entries for {} images x {} boxes x {} fields, got {}",
            truths * l.batch,
            l.batch,
            spec.max_boxes,
            settings.truth_size,
            truth.len()
        )));
    }
    let mut delta = vec![0.0f32; l.outputs()];
    let mut hm_target = vec![0.0f32; l.batch * l.classes * spatial];
    let mut hm_weight = vec![1.0f32; l.batch * l.classes * spatial];
    let (mut hm_loss, mut wh_loss, mut off_loss) = (0.0f32, 0.0f32, 0.0f32);
    let mut total_iou = 0.0f32;
    let mut positives = 0usize;
    let mut class_ids = vec![-1i32; l.batch * spatial];
    let mut labels = vec![-1i32; l.batch * spatial];
    let net_w = settings.net_width as f32;
    let net_h = settings.net_height as f32;

    let mut read_truth = |t: usize| -> (f32, f32, f32, f32, usize, usize) {
        let base = b * truths + t * settings.truth_size;
        (
            truth[base],
            truth[base + 1],
            truth[base + 2],
            truth[base + 3],
            truth[base + 4] as usize,
            truth[base + 5] as usize,
        )
    };

    // ---- 1. heatmap targets (max-blended, so overlaps keep the strongest) ----
    for t in 0..spec.max_boxes {
        let (x, y, w, h, class_id, _track) = read_truth(t);
        if x == 0.0 {
            break;
        }
        if class_id >= l.classes {
            return Err(CenterNetError(format!(
                "[centernet] invalid class id {class_id} outside [0,{})",
                l.classes
            )));
        }
        let class_id = match &settings.class_map {
            Some(map) => map[class_id],
            None => class_id,
        };
        if !(x.is_finite() && y.is_finite() && w.is_finite() && h.is_finite())
            || x <= 0.0
            || x > 1.0
            || y <= 0.0
            || y > 1.0
            || w <= 0.0
            || h <= 0.0
        {
            return Err(CenterNetError(format!(
                "[centernet] invalid truth box x={x} y={y} w={w} h={h}"
            )));
        }
        // Optional FPN assignment gate on the largest side, in input pixels.
        let max_side_px = (w * net_w).max(h * net_h);
        if max_side_px < spec.scale_min_px || max_side_px >= spec.scale_max_px {
            continue;
        }
        let gx = x * l.w as f32;
        let gy = y * l.h as f32;
        let ix = (gx.floor() as i64).clamp(0, l.w as i64 - 1);
        let iy = (gy.floor() as i64).clamp(0, l.h as i64 - 1);
        let w_cells = (w * l.w as f32).max(1e-6);
        let h_cells = (h * l.h as f32).max(1e-6);
        let min_side_px = (w * net_w).min(h * net_h);
        let small_weight = continuous_small_weight(spec, min_side_px);
        let (radius_x, radius_y) = adaptive_gaussian_radii(spec, w_cells, h_cells);
        draw_center_gaussian(
            l, &mut hm_target, &mut hm_weight, b, class_id, ix, iy, radius_x, radius_y,
            small_weight,
        );
    }

    // ---- 2. heatmap focal loss over every cell ----
    let focal_alpha = spec.focal_alpha.max(0.0);
    let focal_beta = spec.focal_beta.max(0.0);
    for bb in 0..l.batch {
        for c in 0..l.classes {
            for loc in 0..spatial {
                let hti = (bb * l.classes + c) * spatial + loc;
                let outi = ct_index(l, bb, c, loc);
                let y = hm_target[hti].clamp(0.0, 1.0);
                let p = output[outi].clamp(1e-6, 1.0 - 1e-6);
                let w = hm_weight[hti].max(0.0);
                if y >= 0.999 {
                    let focal = (1.0 - p).powf(focal_alpha);
                    let grad = w * focal * (1.0 - p);
                    delta[outi] += clamp_delta(spec.hm_normalizer * grad, spec.max_delta);
                    hm_loss += spec.hm_normalizer * w * focal * -safe_log(p);
                } else {
                    let neg_weight = (1.0 - y).powf(focal_beta);
                    let focal = p.powf(focal_alpha);
                    let grad = -w * neg_weight * focal * p;
                    delta[outi] += clamp_delta(spec.hm_normalizer * grad, spec.max_delta);
                    hm_loss += spec.hm_normalizer * w * neg_weight * focal * -safe_log(1.0 - p);
                }
            }
        }
    }

    // ---- 3. sparse regression at the true centres ----
    for t in 0..spec.max_boxes {
        let (x, y, w, h, class_id, track_id) = read_truth(t);
        if x == 0.0 {
            break;
        }
        if class_id >= l.classes {
            continue; // C++ `continue`s here, unlike the target loop
        }
        let class_id = match &settings.class_map {
            Some(map) => map[class_id],
            None => class_id,
        };
        let max_side_px = (w * net_w).max(h * net_h);
        if max_side_px < spec.scale_min_px || max_side_px >= spec.scale_max_px {
            continue;
        }
        let gx = x * l.w as f32;
        let gy = y * l.h as f32;
        let ix = (gx.floor() as i64).clamp(0, l.w as i64 - 1);
        let iy = (gy.floor() as i64).clamp(0, l.h as i64 - 1);
        let loc = (iy * l.w as i64 + ix) as usize;
        let min_side_px = (w * net_w).min(h * net_h);
        let small_weight = continuous_small_weight(spec, min_side_px);
        let class_multiplier = settings
            .class_multipliers
            .as_ref()
            .map(|m| m[class_id])
            .unwrap_or(1.0);
        let weight = small_weight * class_multiplier;
        let target_log_w = safe_log((w * l.w as f32).max(1e-6));
        let target_log_h = safe_log((h * l.h as f32).max(1e-6));
        let target_off_x = gx - ix as f32 - 0.5;
        let target_off_y = gy - iy as f32 - 0.5;
        let ew = finite_or_zero(output[ct_index(l, b, l.classes, loc)]) - target_log_w;
        let eh = finite_or_zero(output[ct_index(l, b, l.classes + 1, loc)]) - target_log_h;
        let eox = finite_or_zero(output[ct_index(l, b, l.classes + 2, loc)]) - target_off_x;
        let eoy = finite_or_zero(output[ct_index(l, b, l.classes + 3, loc)]) - target_off_y;
        delta[ct_index(l, b, l.classes, loc)] +=
            clamp_delta(-spec.wh_normalizer * weight * smooth_l1_grad(ew), spec.max_delta);
        delta[ct_index(l, b, l.classes + 1, loc)] +=
            clamp_delta(-spec.wh_normalizer * weight * smooth_l1_grad(eh), spec.max_delta);
        delta[ct_index(l, b, l.classes + 2, loc)] +=
            clamp_delta(-spec.off_normalizer * weight * smooth_l1_grad(eox), spec.max_delta);
        delta[ct_index(l, b, l.classes + 3, loc)] +=
            clamp_delta(-spec.off_normalizer * weight * smooth_l1_grad(eoy), spec.max_delta);
        wh_loss += spec.wh_normalizer * weight * (smooth_l1_loss(ew) + smooth_l1_loss(eh));
        off_loss += spec.off_normalizer * weight * (smooth_l1_loss(eox) + smooth_l1_loss(eoy));
        class_ids[b * spatial + loc] = class_id as i32;
        labels[b * spatial + loc] = track_id as i32;
        let (px, py, pw, ph) = decode_centernet_box(output, l, b, loc);
        let inter_w = (px + 0.5 * pw).min(x + 0.5 * w) - (px - 0.5 * pw).max(x - 0.5 * w);
        let inter_h = (py + 0.5 * ph).min(y + 0.5 * h) - (py - 0.5 * ph).max(y - 0.5 * h);
        let intersection = inter_w.max(0.0) * inter_h.max(0.0);
        let union = pw.max(0.0) * ph.max(0.0) + w.max(0.0) * h.max(0.0) - intersection;
        total_iou += if union > 0.0 { intersection / union } else { 0.0 };
        positives += 1;
    }
    for d in delta.iter_mut() {
        if !d.is_finite() {
            *d = 0.0;
        }
    }
    Ok(CenterNetBatch { delta, hm_loss, wh_loss, off_loss, total_iou, positives, class_ids, labels })
}

/// The whole training contribution.  `output` must be the activated forward
/// (see [`centernet_forward`]); the returned `cost` is
/// `(hm_loss + wh_loss + off_loss) / batch`, as C++ reports it.
pub fn centernet_training_delta(
    output: &[f32],
    truth: &[f32],
    l: &CenterNetLayout,
    spec: &CenterNetSpec,
    settings: &CenterNetSettings,
) -> Result<(Vec<f32>, f32, usize), CenterNetError> {
    let mut delta = vec![0.0f32; l.len()];
    let mut hm_loss = 0.0f32;
    let mut wh_loss = 0.0f32;
    let mut off_loss = 0.0f32;
    let mut positives = 0usize;
    for b in 0..l.batch {
        let batch = centernet_train_batch(output, truth, l, spec, settings, b)?;
        let out = l.outputs();
        for (dst, src) in delta[b * out..(b + 1) * out].iter_mut().zip(&batch.delta) {
            *dst += src;
        }
        hm_loss += batch.hm_loss;
        wh_loss += batch.wh_loss;
        off_loss += batch.off_loss;
        positives += batch.positives;
    }
    let cost = (hm_loss + wh_loss + off_loss) / l.batch.max(1) as f32;
    Ok((delta, cost, positives))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(classes: usize) -> CenterNetSpec {
        CenterNetSpec {
            classes,
            max_boxes: 4,
            jitter: 0.2,
            resize: 1.0,
            center_min_radius: 1,
            peak_nms: 1,
            anisotropic_gaussian: 1,
            small_threshold: 8.0,
            small_ref_size: 32.0,
            small_boost: 8.0,
            scale_min_px: 0.0,
            scale_max_px: f32::MAX,
            gaussian_iou: 0.7,
            focal_alpha: 2.0,
            focal_beta: 4.0,
            hm_normalizer: 1.0,
            wh_normalizer: 0.1,
            off_normalizer: 1.0,
            max_delta: 10.0,
            delta_normalizer: 1.0,
        }
    }

    fn layout(classes: usize, w: usize, h: usize) -> CenterNetLayout {
        CenterNetLayout { batch: 1, w, h, classes }
    }

    #[test]
    fn the_channel_layout_is_classes_plus_four() {
        let l = layout(3, 2, 2);
        assert_eq!(l.entries(), 7);
        assert_eq!(l.outputs(), 4 * 7);
        // `ct_index` is `batch * outputs + entry * spatial + loc`.
        assert_eq!(ct_index(&l, 0, 0, 2), 2);
        assert_eq!(ct_index(&l, 0, 3, 2), 3 * 4 + 2);
        assert_eq!(ct_index(&l, 1, 6, 1), l.outputs() + 6 * 4 + 1);
    }

    #[test]
    fn the_forward_sigmoids_only_the_heatmap_entries() {
        let l = layout(2, 2, 2);
        let mut input = vec![0.0f32; l.len()];
        for v in input.iter_mut() {
            *v = 1.0;
        }
        let out = centernet_forward(&input, &l);
        for c in 0..l.classes {
            for loc in 0..l.spatial() {
                let v = out[ct_index(&l, 0, c, loc)];
                assert!((v - 0.731_058_6).abs() < 1e-6, "sigmoid(1) = {v}");
            }
        }
        // The four regression entries stay raw.
        for entry in l.classes..l.entries() {
            for loc in 0..l.spatial() {
                assert_eq!(out[ct_index(&l, 0, entry, loc)], 1.0, "entry {entry} must not be squashed");
            }
        }
        // Non-finite inputs are sanitised after the activation.
        let mut dirty = vec![0.0f32; l.len()];
        dirty[ct_index(&l, 0, 0, 0)] = f32::NAN;
        dirty[ct_index(&l, 0, l.classes, 1)] = f32::INFINITY;
        let out = centernet_forward(&dirty, &l);
        // C++ applies the sigmoid *before* the non-finite sweep, so a NaN
        // becomes sigmoid(NaN) = NaN and is then zeroed (not sigmoid(0)).
        assert_eq!(out[ct_index(&l, 0, 0, 0)], 0.0, "NaN is zeroed after the sigmoid");
        assert_eq!(out[ct_index(&l, 0, l.classes, 1)], 0.0, "Inf -> 0 after sanitise");
    }

    #[test]
    fn the_decode_matches_the_hand_computation() {
        let l = layout(1, 4, 4);
        let spatial = l.spatial();
        let mut out = vec![0.0f32; l.len()];
        // Cell 5 is (row 1, col 1). off = (0.25, -0.25), log_w = log_h = ln(8).
        out[ct_index(&l, 0, 1, 5)] = 8f32.ln();
        out[ct_index(&l, 0, 2, 5)] = 8f32.ln();
        out[ct_index(&l, 0, 3, 5)] = 0.25;
        out[ct_index(&l, 0, 4, 5)] = -0.25;
        let (x, y, w, h) = decode_centernet_box(&out, &l, 0, 5);
        assert!((x - (1.0 + 0.5 + 0.25) / 4.0).abs() < 1e-6, "cx {x}");
        assert!((y - (1.0 + 0.5 - 0.25) / 4.0).abs() < 1e-6, "cy {y}");
        assert!((w - 8.0 / 4.0).abs() < 1e-6, "w {w}");
        assert!((h - 8.0 / 4.0).abs() < 1e-6, "h {h}");
        // The offsets are clamped to +-0.75 and the sizes to exp(+-20).
        let mut extreme = vec![0.0f32; l.len()];
        extreme[ct_index(&l, 0, 3, 5)] = 50.0;
        extreme[ct_index(&l, 0, 4, 5)] = -50.0;
        extreme[ct_index(&l, 0, 1, 5)] = 500.0; // log_w far past the exp clamp
        let (x, y, w, _) = decode_centernet_box(&extreme, &l, 0, 5);
        assert!((x - (1.0 + 0.5 + 0.75) / 4.0).abs() < 1e-6, "offset clamp: {x}");
        assert!((y - (1.0 + 0.5 - 0.75) / 4.0).abs() < 1e-6, "offset clamp: {y}");
        assert!(w <= 20f32.exp() / 4.0, "exp is clamped to [-20, 20], got {w}");
        assert!(spatial == 16);
    }

    #[test]
    fn gaussian_radius_shrinks_as_the_overlap_requirement_grows() {
        let r_easy = gaussian_radius(4.0, 4.0, 0.1);
        let r_hard = gaussian_radius(4.0, 4.0, 0.9);
        assert!(r_hard < r_easy, "{r_hard} vs {r_easy}");
        // A degenerate box has no radius at all.
        assert_eq!(gaussian_radius(0.0, 4.0, 0.7), 0);
        assert_eq!(gaussian_radius(f32::NAN, 4.0, 0.7), 0);
        // At a fixed overlap a bigger box needs a bigger blob.
        let r_small = gaussian_radius(4.0, 4.0, 0.7);
        let r_big = gaussian_radius(20.0, 20.0, 0.7);
        assert!(r_big > r_small, "{r_big} vs {r_small}");
        // The radius is the *smallest* of the three overlap solutions, so a
        // demanding `min_overlap` can round it to 0 on a small box.
        assert_eq!(gaussian_radius(2.0, 2.0, 0.9), 0);
    }

    #[test]
    fn the_anisotropic_target_stretches_with_the_aspect_ratio() {
        let mut s = spec(1);
        let (rx, ry) = adaptive_gaussian_radii(&s, 12.0, 2.0);
        assert!(rx > ry, "a wide box gets a wide blob: {rx} vs {ry}");
        // The isotropic mode collapses both axes onto the base radius.
        s.anisotropic_gaussian = 0;
        let (rx, ry) = adaptive_gaussian_radii(&s, 12.0, 2.0);
        assert_eq!(rx, ry);
        // The minimum radius is always respected.
        s.anisotropic_gaussian = 1;
        s.center_min_radius = 5;
        let (rx, ry) = adaptive_gaussian_radii(&s, 1.0, 1.0);
        assert!(rx >= 5 && ry >= 5, "{rx} {ry}");
    }

    #[test]
    fn the_tiny_object_weight_is_continuous_and_capped() {
        let s = spec(1);
        // small_ref_size = 32, small_boost = 8, per the header's example.
        assert_eq!(continuous_small_weight(&s, 32.0), 1.0);
        assert!((continuous_small_weight(&s, 16.0) - 2.0).abs() < 1e-6);
        assert!((continuous_small_weight(&s, 8.0) - 4.0).abs() < 1e-6);
        assert!((continuous_small_weight(&s, 4.0) - 8.0).abs() < 1e-6);
        assert_eq!(continuous_small_weight(&s, 2.0), 8.0, "capped at small_boost");
        assert_eq!(continuous_small_weight(&s, 0.0), 8.0);
    }

    #[test]
    fn the_gaussian_peak_is_one_and_the_halo_decays() {
        let l = layout(1, 7, 7);
        let mut target = vec![0.0f32; l.classes * l.spatial()];
        let mut weight = vec![1.0f32; l.classes * l.spatial()];
        draw_center_gaussian(&l, &mut target, &mut weight, 0, 0, 3, 3, 2, 2, 8.0);
        let at = |x: usize, y: usize| target[y * l.w + x];
        assert!((at(3, 3) - 1.0).abs() < 1e-6, "the centre is exactly 1");
        assert!(at(4, 3) < at(3, 3) && at(4, 3) > 0.0, "a positive halo");
        assert!(at(5, 3) < at(4, 3), "the halo decays outward");
        assert_eq!(weight[3 * l.w + 3], 8.0, "the tiny weight lands on the blob");
        // A blob that reaches outside the map is clipped, not wrapped.
        draw_center_gaussian(&l, &mut target, &mut weight, 0, 0, 0, 0, 4, 4, 1.0);
        assert!(target.iter().all(|v| *v <= 1.0));
    }

    #[test]
    fn peak_nms_keeps_only_strict_local_maxima() {
        let l = layout(1, 3, 1);
        let mut out = vec![0.0f32; l.len()];
        for v in out.iter_mut() {
            *v = 0.0;
        }
        out[ct_index(&l, 0, 0, 1)] = 1.0;
        assert!(!is_local_peak(&out, &l, 0, 0, 0), "a cell below its neighbour is not a peak");
        assert!(is_local_peak(&out, &l, 0, 0, 1), "the maximum is a peak");
        // Equal neighbours do not disqualify (C++ uses a strict `>`).
        out[ct_index(&l, 0, 0, 1)] = 1.0;
        out[ct_index(&l, 0, 0, 0)] = 1.0;
        assert!(is_local_peak(&out, &l, 0, 0, 0));
    }

    #[test]
    fn a_true_centre_cell_gets_a_positive_heatmap_gradient() {
        let s = spec(1);
        let l = CenterNetLayout { batch: 1, w: 4, h: 4, classes: 1 };
        let mut out = vec![0.0f32; l.len()];
        // The heatmap is sigmoid(0) = 0.5 everywhere: exact centres still push up.
        let out = centernet_forward(&out, &l);
        let settings = CenterNetSettings { net_width: 4, net_height: 4, ..CenterNetSettings::default() };
        // One box centred on cell (1, 1).
        let mut truth = vec![0.0f32; s.max_boxes * 6];
        truth[0] = 0.4; // cx -> gx = 1.6 -> ix = 1
        truth[1] = 0.4;
        truth[2] = 0.5;
        truth[3] = 0.5;
        truth[4] = 0.0;
        let (delta, cost, positives) = centernet_training_delta(&out, &truth, &l, &s, &settings).unwrap();
        assert_eq!(positives, 1);
        assert!(cost > 0.0);
        let ix = 1usize;
        let iy = 1usize;
        let centre = ct_index(&l, 0, 0, iy * l.w + ix);
        assert!(delta[centre] > 0.0, "the exact centre is pulled up: {}", delta[centre]);
        // A far-away cell is a negative and is pushed down.
        let far = ct_index(&l, 0, 0, 0);
        assert!(delta[far] < 0.0, "the background is pushed down: {}", delta[far]);
        // The regression entries at the centre moved towards the truth.
        assert!(delta[ct_index(&l, 0, 1, iy * l.w + ix)].abs() > 0.0);
    }

    #[test]
    fn invalid_truth_and_classes_are_errors() {
        let s = spec(1);
        let l = CenterNetLayout { batch: 1, w: 4, h: 4, classes: 1 };
        let out = centernet_forward(&vec![0.0f32; l.len()], &l);
        let settings = CenterNetSettings::default();
        // A short truth buffer.
        let err = centernet_training_delta(&out, &[0.0; 3], &l, &s, &settings).unwrap_err();
        assert!(format!("{err}").contains("truth needs"), "{err}");
        // A class id past the head.
        let mut truth = vec![0.0f32; s.max_boxes * 6];
        truth[0] = 0.5;
        truth[1] = 0.5;
        truth[2] = 0.2;
        truth[3] = 0.2;
        truth[4] = 5.0;
        let err = centernet_training_delta(&out, &truth, &l, &s, &settings).unwrap_err();
        assert!(format!("{err}").contains("invalid class id"), "{err}");
        // A truth box that escapes [0, 1].
        let mut truth = vec![0.0f32; s.max_boxes * 6];
        truth[0] = 1.5;
        truth[1] = 0.5;
        truth[2] = 0.2;
        truth[3] = 0.2;
        truth[4] = 0.0;
        let err = centernet_training_delta(&out, &truth, &l, &s, &settings).unwrap_err();
        assert!(format!("{err}").contains("invalid truth box"), "{err}");
    }

    #[test]
    fn the_scale_gate_skips_out_of_range_objects() {
        let mut s = spec(1);
        s.scale_min_px = 10.0;
        s.scale_max_px = 1000.0;
        let l = CenterNetLayout { batch: 1, w: 4, h: 4, classes: 1 };
        let out = centernet_forward(&vec![0.0f32; l.len()], &l);
        let settings = CenterNetSettings { net_width: 100, net_height: 100, ..CenterNetSettings::default() };
        let mut truth = vec![0.0f32; s.max_boxes * 6];
        truth[0] = 0.4;
        truth[1] = 0.4;
        truth[2] = 0.02; // 2 px on a 100 px input: below scale_min_px
        truth[3] = 0.02;
        truth[4] = 0.0;
        let (_delta, _cost, positives) = centernet_training_delta(&out, &truth, &l, &s, &settings).unwrap();
        assert_eq!(positives, 0, "the scale gate dropped the object");
    }

    #[test]
    fn smooth_l1_matches_the_cpp_shape() {
        assert_eq!(smooth_l1_loss(0.5), 0.125);
        assert_eq!(smooth_l1_loss(2.0), 1.5);
        assert_eq!(smooth_l1_grad(0.5), 0.5);
        assert_eq!(smooth_l1_grad(2.0), 1.0);
        assert_eq!(smooth_l1_grad(-2.0), -1.0);
    }
}

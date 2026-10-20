//! Host reference for the `[deform_conv]` forward stages (`sanitize`,
//! `centered_lhtan`, `stable_sigmoid`, per-batch offset statistics, the offset
//! envelope, mask probabilities and the deformable accumulation).
//!
//! This is the default-body implementation for backends without device kernels;
//! the device twins live in `darknet-cuda-oxide/src/kernels/deform_sample.rs`
//! and must produce the same numbers. The source of truth is
//! `darknet_core::deform_conv` (hand-derived from the C++ layer); no compiled
//! C++ golden exists.

use super::bilinear_gather::sample_at;

/// Source `VALUE_LIMIT`.
pub const VALUE_LIMIT: f32 = 10_000.0;
/// Source `SIGMOID_LIMIT`.
pub const SIGMOID_LIMIT: f32 = 80.0;
/// Source `MIN_OFFSET_STDDEV`.
pub const MIN_OFFSET_STDDEV: f64 = 1.0e-12;

/// Finite -> clamp to +/-10000; non-finite -> 0.
pub fn sanitize(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-VALUE_LIMIT, VALUE_LIMIT)
    } else {
        0.0
    }
}

/// Source `centered_lhtan` value. `max_offset` must be finite and positive.
pub fn centered_lhtan(value: f32, max_offset: f32) -> Result<f32, String> {
    if !max_offset.is_finite() || max_offset <= 0.0 {
        return Err("centered_lhtan requires a finite positive max_offset".to_owned());
    }
    let scaled = value / (2.0 * max_offset) + 0.5;
    let mapped = if scaled < 0.0 {
        0.001 * scaled
    } else if scaled > 1.0 {
        1.0 + 0.001 * (scaled - 1.0)
    } else {
        scaled
    };
    Ok(sanitize((2.0 * mapped - 1.0) * max_offset))
}

/// Source `stable_sigmoid` probability.
pub fn stable_sigmoid(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.5;
    }
    let clipped = value.clamp(-SIGMOID_LIMIT, SIGMOID_LIMIT);
    if clipped >= 0.0 {
        1.0 / (1.0 + (-clipped).exp())
    } else {
        let z = clipped.exp();
        z / (1.0 + z)
    }
}

/// Per-batch offset statistics over the sanitized values.
#[derive(Clone, Debug, PartialEq)]
pub struct OffsetStats {
    pub mean: Vec<f32>,
    pub stddev: Vec<f32>,
    pub inv_std: Vec<f32>,
    /// 1.0 when the batch normalizes, 0.0 otherwise.
    pub normalize: Vec<f32>,
}

pub fn offset_stats(raw: &[f32], per_batch: usize, max_offset: f32) -> Result<OffsetStats, String> {
    if per_batch == 0 || raw.is_empty() || raw.len() % per_batch != 0 {
        return Err("offset_stats requires a positive per_batch that divides the input".to_owned());
    }
    if !max_offset.is_finite() || max_offset <= 0.0 {
        return Err("offset_stats requires a finite positive max_offset".to_owned());
    }
    let batch = raw.len() / per_batch;
    let max_offset = f64::from(max_offset);
    let mut stats = OffsetStats {
        mean: Vec::with_capacity(batch),
        stddev: Vec::with_capacity(batch),
        inv_std: Vec::with_capacity(batch),
        normalize: Vec::with_capacity(batch),
    };
    for b in 0..batch {
        let slice = &raw[b * per_batch..(b + 1) * per_batch];
        let n = per_batch as f64;
        let sum: f64 = slice.iter().map(|&v| f64::from(sanitize(v))).sum();
        let sum_sq: f64 = slice
            .iter()
            .map(|&v| {
                let v = f64::from(sanitize(v));
                v * v
            })
            .sum();
        let mean = sum / n;
        let variance = (sum_sq / n - mean * mean).max(0.0);
        let stddev = variance.sqrt();
        // The source requires spread before either gate, so a constant batch
        // never normalizes even when |mean| > max_offset.
        let normalize =
            stddev > MIN_OFFSET_STDDEV && (stddev > max_offset || mean.abs() > max_offset);
        stats.mean.push(mean as f32);
        stats.stddev.push(stddev as f32);
        stats.inv_std.push(if normalize {
            (1.0 / stddev) as f32
        } else {
            1.0
        });
        stats.normalize.push(if normalize { 1.0 } else { 0.0 });
    }
    Ok(stats)
}

/// `sanitize -> optional per-batch normalize -> sanitize -> centered lhtan`.
#[allow(clippy::too_many_arguments)]
pub fn offset_envelope(
    raw: &[f32],
    mean: &[f32],
    inv_std: &[f32],
    normalize: &[f32],
    planes: usize,
    spatial: usize,
    max_offset: f32,
) -> Result<Vec<f32>, String> {
    let per_batch = planes
        .checked_mul(spatial)
        .filter(|&v| v > 0)
        .ok_or_else(|| "offset_envelope requires positive planes and spatial".to_owned())?;
    if raw.is_empty() || raw.len() % per_batch != 0 {
        return Err("offset_envelope: raw is not a multiple of planes*spatial".to_owned());
    }
    let batch = raw.len() / per_batch;
    if mean.len() != batch || inv_std.len() != batch || normalize.len() != batch {
        return Err("offset_envelope: per-batch operand length mismatch".to_owned());
    }
    let mut out = Vec::with_capacity(raw.len());
    for (i, &value) in raw.iter().enumerate() {
        let b = i / per_batch;
        let mut value = sanitize(value);
        if normalize[b] != 0.0 {
            value = sanitize((value - mean[b]) * inv_std[b]);
        }
        out.push(centered_lhtan(value, max_offset)?);
    }
    Ok(out)
}

/// `stable_sigmoid(sanitize(logit))` per element.
pub fn mask_sigmoid(raw: &[f32]) -> Vec<f32> {
    raw.iter().map(|&v| stable_sigmoid(sanitize(v))).collect()
}

/// Slope of `stable_sigmoid(sanitize(logit))` with respect to `logit`:
/// `p*(1-p)` strictly inside the clamp band, zero at/outside it.
pub fn mask_sigmoid_slope(logit: f32) -> f32 {
    let sanitized = sanitize(logit);
    if !sanitized.is_finite() || sanitized.abs() >= SIGMOID_LIMIT {
        return 0.0;
    }
    let p = stable_sigmoid(sanitized);
    p * (1.0 - p) * sanitize_slope(logit)
}

/// Elementwise adjoint of [`mask_sigmoid`].
pub fn mask_sigmoid_backward(logits: &[f32], output_grad: &[f32]) -> Result<Vec<f32>, String> {
    if logits.len() != output_grad.len() {
        return Err("mask_sigmoid_backward: length mismatch".to_owned());
    }
    Ok(logits
        .iter()
        .zip(output_grad)
        .map(|(&logit, &d)| d * mask_sigmoid_slope(logit))
        .collect())
}

/// Geometry of one `[deform_conv]` accumulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeformGeometry {
    pub batch: usize,
    pub channels: usize,
    pub filters: usize,
    pub channels_per_group: usize,
    pub filters_per_group: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub padding: usize,
    pub dilation: usize,
}

impl DeformGeometry {
    pub fn kernels(&self) -> usize {
        self.size * self.size
    }
    pub fn plane(&self) -> usize {
        self.output_height * self.output_width
    }
}

/// `sum_c sum_k weight[f,c,k] * sample(base + offset) * mask`, then `sanitize`.
/// `offsets` is `[B, 2*K, OH*OW]` (plane `2k` = dy, `2k+1` = dx); `masks` is
/// `[B, K, OH*OW]` and is ignored when `None`.
pub fn accumulate(
    input: &[f32],
    offsets: &[f32],
    masks: Option<&[f32]>,
    weights: &[f32],
    g: &DeformGeometry,
) -> Result<Vec<f32>, String> {
    let kernels = g.kernels();
    let plane = g.plane();
    if g.channels % g.channels_per_group != 0 || g.filters % g.filters_per_group != 0 {
        return Err("deform accumulate: groups must divide channels and filters".to_owned());
    }
    let input_len = g
        .batch
        .checked_mul(g.channels)
        .and_then(|v| v.checked_mul(g.input_height * g.input_width))
        .ok_or_else(|| "deform accumulate: input count overflows".to_owned())?;
    let offset_len = g.batch * 2 * kernels * plane;
    let mask_len = g.batch * kernels * plane;
    let weight_len = g.filters * g.channels_per_group * kernels;
    if input.len() != input_len
        || offsets.len() != offset_len
        || weights.len() != weight_len
        || masks.map_or(false, |m| m.len() != mask_len)
    {
        return Err("deform accumulate: operand length mismatch".to_owned());
    }
    let mut out = vec![0.0f32; g.batch * g.filters * plane];
    for b in 0..g.batch {
        for f in 0..g.filters {
            let group = f / g.filters_per_group;
            for pos in 0..plane {
                let (oy, ox) = (pos / g.output_width, pos % g.output_width);
                let mut sum = 0.0f32;
                for lc in 0..g.channels_per_group {
                    let c = group * g.channels_per_group + lc;
                    let base = (b * g.channels + c) * g.input_height * g.input_width;
                    for k in 0..kernels {
                        let (ky, kx) = (k / g.size, k % g.size);
                        let dy = offsets[b * 2 * kernels * plane + 2 * k * plane + pos];
                        let dx = offsets[b * 2 * kernels * plane + (2 * k + 1) * plane + pos];
                        let y = (oy * g.stride_y) as f32 - (g.padding * g.dilation) as f32
                            + (ky * g.dilation) as f32
                            + dy;
                        let x = (ox * g.stride_x) as f32 - (g.padding * g.dilation) as f32
                            + (kx * g.dilation) as f32
                            + dx;
                        let sample = sample_at(input, base, g.input_height, g.input_width, y, x);
                        let mask = masks.map_or(1.0, |m| m[b * kernels * plane + k * plane + pos]);
                        let weight = weights[(f * g.channels_per_group + lc) * kernels + k];
                        sum += weight * sample * mask;
                    }
                }
                out[(b * g.filters + f) * plane + pos] = sanitize(sum);
            }
        }
    }
    Ok(out)
}

/// Gradients of [`accumulate`] with respect to every operand.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformGradients {
    pub d_input: Vec<f32>,
    pub d_offsets: Vec<f32>,
    /// Empty when the forward ran without masks.
    pub d_masks: Vec<f32>,
    pub d_weights: Vec<f32>,
}

/// Slope of the source `sanitize` at `value`: 1 inside the open band, else 0.
fn sanitize_slope(value: f32) -> f32 {
    if value.is_finite() && value.abs() < VALUE_LIMIT {
        1.0
    } else {
        0.0
    }
}

/// Exact adjoint of [`accumulate`], including the `sanitize` saturation on the
/// summed output and on the sanitized input.
///
/// The source evaluates the output slope at the *unsanitized* sum, then walks
/// `f, group, local_channel, k` exactly like the forward.
pub fn accumulate_backward(
    input: &[f32],
    offsets: &[f32],
    masks: Option<&[f32]>,
    weights: &[f32],
    output_grad: &[f32],
    g: &DeformGeometry,
) -> Result<DeformGradients, String> {
    let kernels = g.kernels();
    let plane = g.plane();
    let input_len = g.batch * g.channels * g.input_height * g.input_width;
    let offset_len = g.batch * 2 * kernels * plane;
    let mask_len = g.batch * kernels * plane;
    let weight_len = g.filters * g.channels_per_group * kernels;
    if input.len() != input_len
        || offsets.len() != offset_len
        || weights.len() != weight_len
        || output_grad.len() != g.batch * g.filters * plane
        || masks.map_or(false, |m| m.len() != mask_len)
    {
        return Err("deform accumulate_backward: operand length mismatch".to_owned());
    }
    let mut grad = DeformGradients {
        d_input: vec![0.0; input_len],
        d_offsets: vec![0.0; offset_len],
        d_masks: if masks.is_some() {
            vec![0.0; mask_len]
        } else {
            Vec::new()
        },
        d_weights: vec![0.0; weight_len],
    };
    for b in 0..g.batch {
        for f in 0..g.filters {
            let group = f / g.filters_per_group;
            for pos in 0..plane {
                let (oy, ox) = (pos / g.output_width, pos % g.output_width);
                let out_index = (b * g.filters + f) * plane + pos;
                let mut raw_sum = 0.0f32;
                for lc in 0..g.channels_per_group {
                    let c = group * g.channels_per_group + lc;
                    let base = (b * g.channels + c) * g.input_height * g.input_width;
                    for k in 0..kernels {
                        let (ky, kx) = (k / g.size, k % g.size);
                        let offset_base = b * 2 * kernels * plane + 2 * k * plane + pos;
                        let y = (oy * g.stride_y) as f32 - (g.padding * g.dilation) as f32
                            + (ky * g.dilation) as f32
                            + offsets[offset_base];
                        let x = (ox * g.stride_x) as f32 - (g.padding * g.dilation) as f32
                            + (kx * g.dilation) as f32
                            + offsets[offset_base + plane];
                        let sample = sample_at(input, base, g.input_height, g.input_width, y, x);
                        let mask = masks.map_or(1.0, |m| m[b * kernels * plane + k * plane + pos]);
                        raw_sum +=
                            weights[(f * g.channels_per_group + lc) * kernels + k] * sample * mask;
                    }
                }
                let g_out = output_grad[out_index] * sanitize_slope(raw_sum);
                if g_out == 0.0 {
                    continue;
                }
                for lc in 0..g.channels_per_group {
                    let c = group * g.channels_per_group + lc;
                    let base = (b * g.channels + c) * g.input_height * g.input_width;
                    for k in 0..kernels {
                        let (ky, kx) = (k / g.size, k % g.size);
                        let offset_base = b * 2 * kernels * plane + 2 * k * plane + pos;
                        let y = (oy * g.stride_y) as f32 - (g.padding * g.dilation) as f32
                            + (ky * g.dilation) as f32
                            + offsets[offset_base];
                        let x = (ox * g.stride_x) as f32 - (g.padding * g.dilation) as f32
                            + (kx * g.dilation) as f32
                            + offsets[offset_base + plane];
                        let mask_index = b * kernels * plane + k * plane + pos;
                        let mask = masks.map_or(1.0, |m| m[mask_index]);
                        let w_index = (f * g.channels_per_group + lc) * kernels + k;
                        let weight = weights[w_index];
                        let sample = sample_at(input, base, g.input_height, g.input_width, y, x);
                        grad.d_weights[w_index] += g_out * sample * mask;
                        if masks.is_some() {
                            grad.d_masks[mask_index] += g_out * weight * sample;
                        }
                        if y < -1.0
                            || y > g.input_height as f32
                            || x < -1.0
                            || x > g.input_width as f32
                        {
                            continue;
                        }
                        let y0f = y.floor();
                        let x0f = x.floor();
                        let (ly, lx) = (y - y0f, x - x0f);
                        let (hy, hx) = (1.0 - ly, 1.0 - lx);
                        let corners = [
                            (y0f as isize, x0f as isize, hy * hx),
                            (y0f as isize, x0f as isize + 1, hy * lx),
                            (y0f as isize + 1, x0f as isize, ly * hx),
                            (y0f as isize + 1, x0f as isize + 1, ly * lx),
                        ];
                        let scale = g_out * weight * mask;
                        let mut values = [0.0f32; 4];
                        for (slot, (cy, cx, corner)) in corners.iter().enumerate() {
                            if *cy >= 0
                                && *cy < g.input_height as isize
                                && *cx >= 0
                                && *cx < g.input_width as isize
                            {
                                let index = base + *cy as usize * g.input_width + *cx as usize;
                                values[slot] = input[index];
                                grad.d_input[index] += scale * corner;
                            }
                        }
                        let dy_slope =
                            -hx * values[0] - lx * values[1] + hx * values[2] + lx * values[3];
                        let dx_slope =
                            -hy * values[0] + hy * values[1] - ly * values[2] + ly * values[3];
                        grad.d_offsets[offset_base] += scale * dy_slope;
                        grad.d_offsets[offset_base + plane] += scale * dx_slope;
                    }
                }
            }
        }
    }
    for (value, input_value) in grad.d_input.iter_mut().zip(input) {
        *value *= sanitize_slope(*input_value);
    }
    Ok(grad)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> DeformGeometry {
        DeformGeometry {
            batch: 1,
            channels: 1,
            filters: 1,
            channels_per_group: 1,
            filters_per_group: 1,
            input_height: 2,
            input_width: 3,
            output_height: 2,
            output_width: 3,
            size: 3,
            stride_x: 1,
            stride_y: 1,
            padding: 1,
            dilation: 1,
        }
    }

    #[test]
    fn sanitize_and_lhtan_follow_the_source_piecewise_map() {
        assert_eq!(sanitize(f32::NAN), 0.0);
        assert_eq!(sanitize(20_000.0), 10_000.0);
        assert_eq!(sanitize(-20_000.0), -10_000.0);
        assert!((centered_lhtan(0.0, 4.0).unwrap() - 0.0).abs() < 1e-6);
        assert!((centered_lhtan(2.0, 4.0).unwrap() - 2.0).abs() < 1e-6);
        // Beyond the band the map is linear with slope 0.001, so it is *not*
        // hard-clamped at max_offset: scaled 1.5 -> 4.004.
        assert!((centered_lhtan(8.0, 4.0).unwrap() - 4.004).abs() < 1e-5);
        assert!(centered_lhtan(0.0, 0.0).is_err());
    }

    #[test]
    fn sigmoid_matches_the_stable_form_and_clamps() {
        assert!((stable_sigmoid(0.0) - 0.5).abs() < 1e-7);
        assert!((stable_sigmoid(f32::NAN) - 0.5).abs() < 1e-7);
        assert_eq!(
            stable_sigmoid(1e9).to_bits(),
            stable_sigmoid(80.0).to_bits()
        );
        assert_eq!(
            stable_sigmoid(-1e9).to_bits(),
            stable_sigmoid(-80.0).to_bits()
        );
        assert!((stable_sigmoid(80.0) - (1.0 - stable_sigmoid(-80.0))).abs() < 1e-6);
    }

    #[test]
    fn offset_stats_require_spread_before_the_mean_gate() {
        let stats = offset_stats(&[7.0; 6], 6, 3.0).unwrap();
        assert_eq!(stats.normalize, vec![0.0]);
        assert_eq!(stats.inv_std, vec![1.0]);
        assert!((stats.mean[0] - 7.0).abs() < 1e-6);
        let stats = offset_stats(&[7.0, 7.01, 6.99, 7.02, 6.98, 7.0], 6, 3.0).unwrap();
        assert_eq!(stats.normalize, vec![1.0]);
        assert!(stats.inv_std[0] > 1.0);
    }

    #[test]
    fn envelope_applies_normalization_then_lhtan() {
        let raw = [0.5f32, -0.5, 100.0, -100.0];
        let off = offset_envelope(&raw, &[0.0], &[1.0], &[0.0], 1, 4, 2.0).unwrap();
        for (i, &v) in raw.iter().enumerate() {
            assert!((off[i] - centered_lhtan(sanitize(v), 2.0).unwrap()).abs() < 1e-6);
        }
        let on = offset_envelope(&raw, &[0.25], &[2.0], &[1.0], 1, 4, 2.0).unwrap();
        let want = centered_lhtan(sanitize((sanitize(raw[0]) - 0.25) * 2.0), 2.0).unwrap();
        assert!((on[0] - want).abs() < 1e-6);
        assert!(offset_envelope(&raw, &[0.0, 1.0], &[1.0], &[0.0], 1, 4, 2.0).is_err());
    }

    #[test]
    fn accumulate_zero_offsets_is_a_plain_convolution() {
        let g = geometry();
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let offsets = vec![0.0f32; 2 * 9 * 6];
        let weights: Vec<f32> = (0..9).map(|i| 0.1 * (i as f32 + 1.0)).collect();
        let out = accumulate(&input, &offsets, None, &weights, &g).unwrap();
        let mut want = 0.0f32;
        for ky in 0..2usize {
            for kx in 0..3usize {
                let k = ky * 3 + kx;
                want += weights[k] * input[ky * 3 + kx];
            }
        }
        assert!((out[4] - want).abs() < 1e-5, "got {} want {want}", out[4]);
    }

    #[test]
    fn accumulate_honours_masks_and_rejects_bad_operands() {
        let g = geometry();
        let input = vec![1.0f32; 6];
        let offsets = vec![0.0f32; 2 * 9 * 6];
        let weights = vec![1.0f32; 9];
        let masks = vec![0.0f32; 9 * 6];
        let out = accumulate(&input, &offsets, Some(&masks), &weights, &g).unwrap();
        assert!(out.iter().all(|&v| v == 0.0));
        assert!(accumulate(&input, &offsets[1..], None, &weights, &g).is_err());
        assert!(accumulate(&input, &offsets, Some(&masks[1..]), &weights, &g).is_err());
        assert!(accumulate(&input, &offsets, None, &weights[1..], &g).is_err());
    }

    // ---- accumulation adjoint ----

    fn backward_geometry() -> DeformGeometry {
        DeformGeometry {
            batch: 1,
            channels: 2,
            filters: 2,
            channels_per_group: 2,
            filters_per_group: 2,
            input_height: 2,
            input_width: 3,
            output_height: 2,
            output_width: 3,
            size: 3,
            stride_x: 1,
            stride_y: 1,
            padding: 1,
            dilation: 1,
        }
    }

    fn operands(g: &DeformGeometry) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let input: Vec<f32> = (0..g.batch * g.channels * g.input_height * g.input_width)
            .map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5)
            .collect();
        let offsets: Vec<f32> = (0..g.batch * 2 * g.kernels() * g.plane())
            .map(|i| (((i * 5) % 7) as f32) * 0.15 - 0.45)
            .collect();
        let masks: Vec<f32> = (0..g.batch * g.kernels() * g.plane())
            .map(|i| 0.2 + ((i % 4) as f32) * 0.2)
            .collect();
        let weights: Vec<f32> = (0..g.filters * g.channels_per_group * g.kernels())
            .map(|i| 0.5 - ((i % 3) as f32) * 0.25)
            .collect();
        let seed: Vec<f32> = (0..g.batch * g.filters * g.plane())
            .map(|i| 0.3 + ((i % 5) as f32) * 0.1)
            .collect();
        (input, offsets, masks, weights, seed)
    }

    fn objective(
        input: &[f32],
        offsets: &[f32],
        masks: Option<&[f32]>,
        weights: &[f32],
        seed: &[f32],
        g: &DeformGeometry,
    ) -> f32 {
        accumulate(input, offsets, masks, weights, g)
            .unwrap()
            .iter()
            .zip(seed)
            .map(|(v, s)| v * s)
            .sum()
    }

    fn check_fd(analytic: f32, base: f32, what: &str, index: usize, f: impl Fn(f32) -> f32) {
        let eps = 1e-2f32;
        let want = (f(base + eps) - f(base - eps)) / (2.0 * eps);
        assert!(
            (analytic - want).abs() < 5e-3 + 5e-3 * want.abs(),
            "{what}[{index}]: analytic {analytic} vs fd {want}"
        );
    }

    #[test]
    fn backward_matches_finite_differences_for_all_operands() {
        let g = backward_geometry();
        let (input, offsets, masks, weights, seed) = operands(&g);
        let mask_ref = Some(masks.as_slice());
        let grad = accumulate_backward(&input, &offsets, mask_ref, &weights, &seed, &g).unwrap();

        for i in (0..input.len()).step_by(3) {
            check_fd(grad.d_input[i], input[i], "d_input", i, |v| {
                let mut x = input.clone();
                x[i] = v;
                objective(&x, &offsets, mask_ref, &weights, &seed, &g)
            });
        }
        for i in (0..offsets.len()).step_by(5) {
            check_fd(grad.d_offsets[i], offsets[i], "d_offsets", i, |v| {
                let mut o = offsets.clone();
                o[i] = v;
                objective(&input, &o, mask_ref, &weights, &seed, &g)
            });
        }
        for i in 0..masks.len() {
            check_fd(grad.d_masks[i], masks[i], "d_masks", i, |v| {
                let mut m = masks.clone();
                m[i] = v;
                objective(&input, &offsets, Some(m.as_slice()), &weights, &seed, &g)
            });
        }
        for i in 0..weights.len() {
            check_fd(grad.d_weights[i], weights[i], "d_weights", i, |v| {
                let mut w = weights.clone();
                w[i] = v;
                objective(&input, &offsets, mask_ref, &w, &seed, &g)
            });
        }
    }

    #[test]
    fn backward_without_masks_leaves_d_masks_empty_and_rejects_bad_lengths() {
        let g = backward_geometry();
        let (input, offsets, _masks, weights, seed) = operands(&g);
        let grad = accumulate_backward(&input, &offsets, None, &weights, &seed, &g).unwrap();
        assert!(grad.d_masks.is_empty());
        assert_eq!(grad.d_input.len(), input.len());
        assert!(accumulate_backward(&input, &offsets[1..], None, &weights, &seed, &g).is_err());
        assert!(accumulate_backward(&input, &offsets, None, &weights[1..], &seed, &g).is_err());
        assert!(accumulate_backward(&input, &offsets, None, &weights, &seed[1..], &g).is_err());
    }
}

/// Slope of [`centered_lhtan`] at `value` (product of the piecewise map slope
/// and the outer `sanitize` slope). Matches the source: slope 1 on the closed
/// band `[0, 1]` of the scaled value (including the exact endpoints, which the
/// source flags as boundaries), 0.001 outside it.
pub fn centered_lhtan_slope(value: f32, max_offset: f32) -> Result<f32, String> {
    if !max_offset.is_finite() || max_offset <= 0.0 {
        return Err("centered_lhtan_slope requires a finite positive max_offset".to_owned());
    }
    let scaled = value / (2.0 * max_offset) + 0.5;
    let mapped = if scaled < 0.0 {
        0.001 * scaled
    } else if scaled > 1.0 {
        1.0 + 0.001 * (scaled - 1.0)
    } else {
        scaled
    };
    let slope_scaled = if scaled < 0.0 || scaled > 1.0 {
        0.001f32
    } else {
        1.0f32
    };
    let unclamped = (2.0 * mapped - 1.0) * max_offset;
    Ok(slope_scaled * sanitize_slope(unclamped))
}

/// Adjoint of [`offset_envelope`] with respect to `raw`, treating the per-batch
/// statistics as constants. The statistics' own dependence on `raw` is the
/// separate (harder) stage handled by the source's normalization section.
pub fn offset_envelope_backward(
    raw: &[f32],
    mean: &[f32],
    inv_std: &[f32],
    normalize: &[f32],
    output_grad: &[f32],
    planes: usize,
    spatial: usize,
    max_offset: f32,
) -> Result<Vec<f32>, String> {
    let per_batch = planes
        .checked_mul(spatial)
        .filter(|&v| v > 0)
        .ok_or_else(|| {
            "offset_envelope_backward requires positive planes and spatial".to_owned()
        })?;
    if raw.is_empty()
        || raw.len() % per_batch != 0
        || output_grad.len() != raw.len()
        || mean.len() != raw.len() / per_batch
        || inv_std.len() != mean.len()
        || normalize.len() != mean.len()
    {
        return Err("offset_envelope_backward: operand length mismatch".to_owned());
    }
    let mut out = Vec::with_capacity(raw.len());
    for (i, (&value, &grad)) in raw.iter().zip(output_grad).enumerate() {
        let b = i / per_batch;
        let v1 = sanitize(value);
        let (v2, slope_norm) = if normalize[b] != 0.0 {
            ((v1 - mean[b]) * inv_std[b], inv_std[b])
        } else {
            (v1, 1.0)
        };
        let v2 = if normalize[b] != 0.0 {
            sanitize(v2)
        } else {
            v2
        };
        let d_v2 = grad * centered_lhtan_slope(v2, max_offset)?;
        let slope_sanitize = if normalize[b] != 0.0 {
            sanitize_slope((v1 - mean[b]) * inv_std[b])
        } else {
            1.0
        };
        out.push(d_v2 * slope_norm * slope_sanitize * sanitize_slope(value));
    }
    Ok(out)
}

#[cfg(test)]
mod envelope_backward_tests {
    use super::*;

    fn check_fd(analytic: f32, base: f32, what: &str, index: usize, f: impl Fn(f32) -> f32) {
        let eps = 1e-2f32;
        let want = (f(base + eps) - f(base - eps)) / (2.0 * eps);
        assert!(
            (analytic - want).abs() < 5e-3 + 5e-3 * want.abs(),
            "{what}[{index}]: analytic {analytic} vs fd {want}"
        );
    }

    #[test]
    fn envelope_backward_matches_finite_differences_both_modes() {
        let (planes, spatial, max_offset) = (2usize, 4usize, 3.0f32);
        let per_batch = planes * spatial;
        let batch = 2;
        let raw: Vec<f32> = (0..batch * per_batch)
            // Offset the grid so no value lands exactly on a piecewise
            // boundary (scaled == 0 or 1), where the source's slope is a
            // one-sided choice that a central difference cannot confirm.
            .map(|i| ((i * 9 % 17) as f32) * 0.5 - 4.0 + 0.17)
            .collect();
        let mean = vec![-0.5f32, 1.25];
        let inv_std = vec![0.75f32, 1.5];
        let seed: Vec<f32> = (0..raw.len())
            .map(|i| 0.2 + ((i % 5) as f32) * 0.15)
            .collect();

        for normalize in [[0.0f32, 0.0], [1.0, 1.0]] {
            let grad = offset_envelope_backward(
                &raw, &mean, &inv_std, &normalize, &seed, planes, spatial, max_offset,
            )
            .unwrap();
            let objective = |values: &[f32]| -> f32 {
                offset_envelope(
                    values, &mean, &inv_std, &normalize, planes, spatial, max_offset,
                )
                .unwrap()
                .iter()
                .zip(&seed)
                .map(|(v, s)| v * s)
                .sum()
            };
            for i in 0..raw.len() {
                check_fd(grad[i], raw[i], "d_raw", i, |v| {
                    let mut x = raw.clone();
                    x[i] = v;
                    objective(&x)
                });
            }
        }
    }

    #[test]
    fn envelope_backward_rejects_bad_operands() {
        let raw = [0.0f32; 8];
        let one = [0.0f32; 1];
        let two = [0.0f32; 2];
        assert!(offset_envelope_backward(&raw, &one, &one, &one, &raw, 2, 4, 1.0).is_ok());
        assert!(offset_envelope_backward(&raw, &one, &one, &one, &raw[1..], 2, 4, 1.0).is_err());
        assert!(offset_envelope_backward(&raw, &one, &one, &one, &raw, 2, 4, 0.0).is_err());
        assert!(offset_envelope_backward(&raw, &two, &one, &one, &raw, 2, 4, 1.0).is_err());
    }
}

/// Adjoint of the per-batch conditional normalization, i.e. the dependence of
/// `(x - mean)/stddev` on every `x` in the batch (mean and stddev are functions
/// of the same batch). Mirrors the source's normalization section:
///
/// ```text
/// d_out[i] = (d[i] - sum(d)/n) / std - (x[i] - mean) * dot(d, x - mean) / (n * std^3)
/// ```
///
/// `normalized[b] == 0` passes `d` through unchanged. Accumulation is f64 like
/// the source; `mean`/`stddev` are the f32 values the forward exposes.
pub fn offset_normalization_backward(
    stage1: &[f32],
    mean: &[f32],
    stddev: &[f32],
    normalized: &[f32],
    gradient: &[f32],
    per_batch: usize,
) -> Result<Vec<f32>, String> {
    if per_batch == 0
        || stage1.is_empty()
        || stage1.len() % per_batch != 0
        || gradient.len() != stage1.len()
    {
        return Err("offset_normalization_backward: length mismatch".to_owned());
    }
    let batch = stage1.len() / per_batch;
    if mean.len() != batch || stddev.len() != batch || normalized.len() != batch {
        return Err("offset_normalization_backward: per-batch operand length mismatch".to_owned());
    }
    let mut out = Vec::with_capacity(stage1.len());
    for b in 0..batch {
        let start = b * per_batch;
        let slice = &stage1[start..start + per_batch];
        let g = &gradient[start..start + per_batch];
        if normalized[b] == 0.0 {
            out.extend_from_slice(g);
            continue;
        }
        let count = per_batch as f64;
        let m = f64::from(mean[b]);
        let std = f64::from(stddev[b]);
        if std <= 0.0 || !std.is_finite() {
            return Err(format!(
                "offset_normalization_backward: batch {b} has a non-positive stddev {std}"
            ));
        }
        let sum_gradient: f64 = g.iter().map(|&v| f64::from(v)).sum();
        let centered_dot: f64 = g
            .iter()
            .zip(slice)
            .map(|(&gi, &x)| f64::from(gi) * (f64::from(x) - m))
            .sum();
        for (&gi, &x) in g.iter().zip(slice) {
            let centered = f64::from(x) - m;
            let value = ((f64::from(gi) - sum_gradient / count) / std)
                - centered * centered_dot / (count * std * std * std);
            out.push(value as f32);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod normalization_backward_tests {
    use super::*;

    fn check_fd(analytic: f32, base: f32, what: &str, index: usize, f: impl Fn(f32) -> f32) {
        let eps = 1e-2f32;
        let want = (f(base + eps) - f(base - eps)) / (2.0 * eps);
        assert!(
            (analytic - want).abs() < 1e-2 + 1e-2 * want.abs(),
            "{what}[{index}]: analytic {analytic} vs fd {want}"
        );
    }

    /// The gate threshold used by the fixture: batch 0 has std ~30 (normalizes),
    /// batch 1 has std ~0.07 and mean ~0 (stays raw).
    const MAX_OFFSET: f32 = 10.0;

    /// Forward of the normalization stage only, recomputing the batch stats.
    fn normalize(stage1: &[f32], per_batch: usize) -> (Vec<f32>, OffsetStats) {
        let stats = offset_stats(stage1, per_batch, MAX_OFFSET).unwrap();
        let batch = stage1.len() / per_batch;
        let mut out = Vec::with_capacity(stage1.len());
        for b in 0..batch {
            for i in 0..per_batch {
                let x = sanitize(stage1[b * per_batch + i]);
                out.push(if stats.normalize[b] != 0.0 {
                    (x - stats.mean[b]) * stats.inv_std[b]
                } else {
                    x
                });
            }
        }
        (out, stats)
    }

    #[test]
    fn normalization_backward_matches_finite_differences() {
        let per_batch = 6usize;
        let batch = 2usize;
        // Wide batch 0 (normalizes) and tight batch 1 (does not).
        let stage1: Vec<f32> = vec![
            40.0, -40.0, 25.0, -25.0, 10.0, -10.0, // normalizes: std > max_offset
            0.1, -0.1, 0.05, 0.0, -0.05, 0.02, // stays raw
        ];
        assert_eq!(stage1.len(), per_batch * batch);
        let seed: Vec<f32> = (0..stage1.len())
            .map(|i| 0.3 + ((i % 5) as f32) * 0.2)
            .collect();
        let (_, stats) = normalize(&stage1, per_batch);
        assert_eq!(
            stats.normalize,
            vec![1.0, 0.0],
            "fixture must span both modes"
        );

        let grad = offset_normalization_backward(
            &stage1,
            &stats.mean,
            &stats.stddev,
            &stats.normalize,
            &seed,
            per_batch,
        )
        .unwrap();

        let objective = |values: &[f32]| -> f32 {
            let (out, _) = normalize(values, per_batch);
            out.iter().zip(&seed).map(|(v, s)| v * s).sum()
        };
        for i in 0..stage1.len() {
            check_fd(grad[i], stage1[i], "d_stage1", i, |v| {
                let mut x = stage1.clone();
                x[i] = v;
                objective(&x)
            });
        }
    }

    #[test]
    fn normalization_backward_passes_through_and_rejects_bad_input() {
        let stage1 = [1.0f32, -2.0, 3.0, -4.0];
        let mean = [0.0f32];
        let std = [2.0f32];
        let off = [0.0f32];
        let grad = [0.5f32, -0.5, 1.0, 2.0];
        let out = offset_normalization_backward(&stage1, &mean, &std, &off, &grad, 4).unwrap();
        assert_eq!(out, grad.to_vec(), "unnormalized batches pass through");
        assert!(
            offset_normalization_backward(&stage1, &mean, &std, &[1.0], &grad[1..], 4).is_err()
        );
        assert!(offset_normalization_backward(&stage1, &mean, &[0.0], &[1.0], &grad, 4).is_err());
        assert!(offset_normalization_backward(&stage1, &[], &std, &off, &grad, 4).is_err());
    }
}

/// Plain (unshifted) im2col columns for the `[deform_conv]` predictors:
/// `columns[b, (c*size+ky)*size+kx, pos]` with zero fill outside the image.
/// Matches `darknet_core::deform_conv::predictor_columns`, which is why the
/// predictors are ordinary convolutions.
#[allow(clippy::too_many_arguments)]
pub fn predictor_columns(
    input: &[f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>, String> {
    predictor_columns_xy(
        input, batch, channels, in_h, in_w, size, stride, stride, padding, dilation, out_h, out_w,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn predictor_columns_xy(
    input: &[f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
    dilation: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>, String> {
    let kernels = size * size;
    let columns = channels * kernels;
    let plane = out_h * out_w;
    if input.len() != batch * channels * in_h * in_w {
        return Err("predictor_columns: input length mismatch".to_owned());
    }
    let mut out = vec![0.0f32; batch * columns * plane];
    for b in 0..batch {
        for c in 0..channels {
            for ky in 0..size {
                for kx in 0..size {
                    let column = (c * size + ky) * size + kx;
                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let y = (oy * stride_y) as isize - (padding * dilation) as isize
                                + (ky * dilation) as isize;
                            let x = (ox * stride_x) as isize - (padding * dilation) as isize
                                + (kx * dilation) as isize;
                            if y >= 0 && y < in_h as isize && x >= 0 && x < in_w as isize {
                                out[(b * columns + column) * plane + oy * out_w + ox] =
                                    input[(b * channels + c) * in_h * in_w
                                        + y as usize * in_w
                                        + x as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Adjoint of the dense predictor `out[f,pos] = sum_{c,k} W[f,c,k] * col[c,k,pos]
/// + bias[f]` with respect to all three operands.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictorGradients {
    pub d_input: Vec<f32>,
    pub d_weights: Vec<f32>,
    pub d_biases: Vec<f32>,
    pub d_columns: Vec<f32>,
}

/// Backward through the predictor **and** the column gather, i.e. all the way to
/// the layer input. `input_slope` is applied because the source samples the
/// sanitized input.
#[allow(clippy::too_many_arguments)]
pub fn predictor_backward(
    input: &[f32],
    columns: &[f32],
    weights: &[f32],
    output_grad: &[f32],
    batch: usize,
    channels: usize,
    filters: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<PredictorGradients, String> {
    predictor_backward_xy(
        input,
        columns,
        weights,
        output_grad,
        batch,
        channels,
        filters,
        in_h,
        in_w,
        out_h,
        out_w,
        size,
        stride,
        stride,
        padding,
        dilation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn predictor_backward_xy(
    input: &[f32],
    columns: &[f32],
    weights: &[f32],
    output_grad: &[f32],
    batch: usize,
    channels: usize,
    filters: usize,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    size: usize,
    stride_x: usize,
    stride_y: usize,
    padding: usize,
    dilation: usize,
) -> Result<PredictorGradients, String> {
    let kernels = size * size;
    let column_count = channels * kernels;
    let plane = out_h * out_w;
    let weight_len = filters * column_count;
    if input.len() != batch * channels * in_h * in_w
        || columns.len() != batch * column_count * plane
        || weights.len() != weight_len
        || output_grad.len() != batch * filters * plane
    {
        return Err("predictor_backward: operand length mismatch".to_owned());
    }
    let mut d_columns = vec![0.0f32; columns.len()];
    let mut d_weights = vec![0.0f32; weight_len];
    let mut d_biases = vec![0.0f32; filters];
    for b in 0..batch {
        for f in 0..filters {
            for pos in 0..plane {
                let g = output_grad[(b * filters + f) * plane + pos];
                if g == 0.0 {
                    continue;
                }
                d_biases[f] += g;
                for column in 0..column_count {
                    let w_index = f * column_count + column;
                    let col = columns[(b * column_count + column) * plane + pos];
                    d_weights[w_index] += g * col;
                    d_columns[(b * column_count + column) * plane + pos] += g * weights[w_index];
                }
            }
        }
    }
    // Column-gather adjoint: scatter each column back to its source pixel.
    let mut d_input = vec![0.0f32; input.len()];
    for b in 0..batch {
        for c in 0..channels {
            for ky in 0..size {
                for kx in 0..size {
                    let column = (c * size + ky) * size + kx;
                    for oy in 0..out_h {
                        for ox in 0..out_w {
                            let y = (oy * stride_y) as isize - (padding * dilation) as isize
                                + (ky * dilation) as isize;
                            let x = (ox * stride_x) as isize - (padding * dilation) as isize
                                + (kx * dilation) as isize;
                            if y >= 0 && y < in_h as isize && x >= 0 && x < in_w as isize {
                                d_input[(b * channels + c) * in_h * in_w
                                    + y as usize * in_w
                                    + x as usize] += d_columns
                                    [(b * column_count + column) * plane + oy * out_w + ox];
                            }
                        }
                    }
                }
            }
        }
    }
    for (value, &x) in d_input.iter_mut().zip(input) {
        *value *= sanitize_slope(x);
    }
    Ok(PredictorGradients {
        d_input,
        d_weights,
        d_biases,
        d_columns,
    })
}

#[cfg(test)]
mod predictor_backward_tests {
    use super::*;

    fn check_fd(analytic: f32, base: f32, what: &str, index: usize, f: impl Fn(f32) -> f32) {
        let eps = 1e-2f32;
        let want = (f(base + eps) - f(base - eps)) / (2.0 * eps);
        assert!(
            (analytic - want).abs() < 5e-3 + 5e-3 * want.abs(),
            "{what}[{index}]: analytic {analytic} vs fd {want}"
        );
    }

    #[test]
    fn predictor_backward_matches_finite_differences() {
        let (batch, channels, filters) = (2usize, 3usize, 4usize);
        let (in_h, in_w) = (3usize, 4usize);
        let (size, stride, padding, dilation) = (3usize, 1usize, 1usize, 1usize);
        let (out_h, out_w) = (3usize, 4usize);
        let kernels = size * size;
        let column_count = channels * kernels;
        let plane = out_h * out_w;
        let weights: Vec<f32> = (0..filters * column_count)
            .map(|i| ((i * 5 % 11) as f32) * 0.125 - 0.6)
            .collect();
        let biases: Vec<f32> = (0..filters).map(|i| 0.05 * i as f32 - 0.1).collect();
        let input: Vec<f32> = (0..batch * channels * in_h * in_w)
            .map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5)
            .collect();
        let seed: Vec<f32> = (0..batch * filters * plane)
            .map(|i| 0.2 + ((i % 5) as f32) * 0.1)
            .collect();

        let columns = predictor_columns(
            &input, batch, channels, in_h, in_w, size, stride, padding, dilation, out_h, out_w,
        )
        .unwrap();
        let grad = predictor_backward(
            &input, &columns, &weights, &seed, batch, channels, filters, in_h, in_w, out_h, out_w,
            size, stride, padding, dilation,
        )
        .unwrap();

        // Objective over (input, weights, biases) recomputing the columns.
        let objective = |input: &[f32], weights: &[f32], biases: &[f32]| -> f32 {
            let cols = predictor_columns(
                input, batch, channels, in_h, in_w, size, stride, padding, dilation, out_h, out_w,
            )
            .unwrap();
            let mut sum = 0.0f32;
            for b in 0..batch {
                for f in 0..filters {
                    for pos in 0..plane {
                        let mut v = biases[f];
                        for column in 0..column_count {
                            v += weights[f * column_count + column]
                                * cols[(b * column_count + column) * plane + pos];
                        }
                        sum += v * seed[(b * filters + f) * plane + pos];
                    }
                }
            }
            sum
        };

        for i in (0..input.len()).step_by(3) {
            check_fd(grad.d_input[i], input[i], "d_input", i, |v| {
                let mut x = input.clone();
                x[i] = v;
                objective(&x, &weights, &biases)
            });
        }
        for i in 0..weights.len() {
            check_fd(grad.d_weights[i], weights[i], "d_weights", i, |v| {
                let mut w = weights.clone();
                w[i] = v;
                objective(&input, &w, &biases)
            });
        }
        for i in 0..biases.len() {
            check_fd(grad.d_biases[i], biases[i], "d_biases", i, |v| {
                let mut b = biases.clone();
                b[i] = v;
                objective(&input, &weights, &b)
            });
        }
    }

    #[test]
    fn predictor_columns_and_adjoint_support_distinct_axis_strides() {
        let input: Vec<f32> = (1..=12).map(|value| value as f32).collect();
        let columns = predictor_columns_xy(&input, 1, 1, 3, 4, 1, 1, 2, 0, 1, 2, 4).unwrap();
        assert_eq!(columns, vec![1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);

        let grad = predictor_backward_xy(
            &input,
            &columns,
            &[2.0],
            &[1.0; 8],
            1,
            1,
            1,
            3,
            4,
            2,
            4,
            1,
            1,
            2,
            0,
            1,
        )
        .unwrap();
        assert_eq!(
            grad.d_input,
            vec![2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(grad.d_weights, vec![52.0]);
        assert_eq!(grad.d_biases, vec![8.0]);
    }

    #[test]
    fn predictor_backward_rejects_bad_lengths() {
        let columns = vec![0.0f32; 1 * 9 * 4];
        let input = vec![0.0f32; 4];
        let weights = vec![0.0f32; 9];
        let grad = vec![0.0f32; 4];
        assert!(
            predictor_backward(
                &input, &columns, &weights, &grad, 1, 1, 1, 2, 2, 2, 2, 3, 1, 0, 1
            )
            .is_ok()
        );
        assert!(
            predictor_backward(
                &input,
                &columns[1..],
                &weights,
                &grad,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                3,
                1,
                0,
                1
            )
            .is_err()
        );
    }
}

/// Adjoint of the **lhtan + sanitize** stage only, i.e. the derivative with
/// respect to the value that enters it (already normalized for normalizing
/// batches). The normalization's own Jacobian is
/// [`offset_normalization_backward`] and must be chained separately; folding it
/// in here would double-count the `inv_std` factor.
pub fn offset_envelope_pre_backward(
    raw: &[f32],
    mean: &[f32],
    inv_std: &[f32],
    normalize: &[f32],
    output_grad: &[f32],
    planes: usize,
    spatial: usize,
    max_offset: f32,
) -> Result<Vec<f32>, String> {
    let per_batch = planes
        .checked_mul(spatial)
        .filter(|&v| v > 0)
        .ok_or_else(|| {
            "offset_envelope_pre_backward requires positive planes and spatial".to_owned()
        })?;
    if raw.is_empty()
        || raw.len() % per_batch != 0
        || output_grad.len() != raw.len()
        || mean.len() != raw.len() / per_batch
        || inv_std.len() != mean.len()
        || normalize.len() != mean.len()
    {
        return Err("offset_envelope_pre_backward: operand length mismatch".to_owned());
    }
    let mut out = Vec::with_capacity(raw.len());
    for (i, (&value, &grad)) in raw.iter().zip(output_grad).enumerate() {
        let b = i / per_batch;
        let v1 = sanitize(value);
        let pre = if normalize[b] != 0.0 {
            (v1 - mean[b]) * inv_std[b]
        } else {
            v1
        };
        let v2 = if normalize[b] != 0.0 {
            sanitize(pre)
        } else {
            pre
        };
        let slope = if normalize[b] != 0.0 {
            sanitize_slope(pre)
        } else {
            1.0
        };
        out.push(grad * centered_lhtan_slope(v2, max_offset)? * slope);
    }
    Ok(out)
}

/// Full `[deform_conv]` forward at reference level, in the same order as the
/// device composition: sanitize -> offset predictor -> stats -> envelope ->
/// (mask predictor -> sigmoid) -> accumulate. Returns the raw layer output
/// (before the layer's bias/BN/activation) plus the intermediates the backward
/// needs.
pub struct DeformForward {
    pub input_sanitized: Vec<f32>,
    pub columns: Vec<f32>,
    pub raw_offsets: Vec<f32>,
    pub stage1: Vec<f32>,
    pub stats: OffsetStats,
    pub offsets: Vec<f32>,
    pub mask_logits: Option<Vec<f32>>,
    pub masks: Option<Vec<f32>>,
    pub output: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn forward(
    input: &[f32],
    main_weights: &[f32],
    offset_weights: &[f32],
    offset_biases: &[f32],
    mask_weights: Option<&[f32]>,
    mask_biases: Option<&[f32]>,
    g: &DeformGeometry,
    max_offset: f32,
) -> Result<DeformForward, String> {
    let kernels = g.kernels();
    let plane = g.plane();
    let input_sanitized: Vec<f32> = input.iter().copied().map(sanitize).collect();
    let columns = predictor_columns_xy(
        &input_sanitized,
        g.batch,
        g.channels,
        g.input_height,
        g.input_width,
        g.size,
        g.stride_x,
        g.stride_y,
        g.padding,
        g.dilation,
        g.output_height,
        g.output_width,
    )?;
    let column_count = g.channels * kernels;
    let dense = |weights: &[f32], biases: &[f32], filters: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; g.batch * filters * plane];
        for b in 0..g.batch {
            for f in 0..filters {
                for pos in 0..plane {
                    let mut v = biases[f];
                    for column in 0..column_count {
                        v += weights[f * column_count + column]
                            * columns[(b * column_count + column) * plane + pos];
                    }
                    out[(b * filters + f) * plane + pos] = v;
                }
            }
        }
        out
    };
    let raw_offsets = dense(offset_weights, offset_biases, 2 * kernels);
    let stage1: Vec<f32> = raw_offsets.iter().copied().map(sanitize).collect();
    let stats = offset_stats(&raw_offsets, 2 * kernels * plane, max_offset)?;
    let offsets = offset_envelope(
        &raw_offsets,
        &stats.mean,
        &stats.inv_std,
        &stats.normalize,
        2 * kernels,
        plane,
        max_offset,
    )?;
    let (mask_logits, masks) = match (mask_weights, mask_biases) {
        (Some(w), Some(b)) => {
            let logits = dense(w, b, kernels);
            let masks = mask_sigmoid(&logits);
            (Some(logits), Some(masks))
        }
        _ => (None, None),
    };
    let output = accumulate(
        &input_sanitized,
        &offsets,
        masks.as_deref(),
        main_weights,
        g,
    )?;
    Ok(DeformForward {
        input_sanitized,
        columns,
        raw_offsets,
        stage1,
        stats,
        offsets,
        mask_logits,
        masks,
        output,
    })
}

/// Gradients of the full composed forward.
pub struct DeformForwardGradients {
    pub d_input: Vec<f32>,
    pub d_main_weights: Vec<f32>,
    pub d_offset_weights: Vec<f32>,
    pub d_offset_biases: Vec<f32>,
    pub d_mask_weights: Vec<f32>,
    pub d_mask_biases: Vec<f32>,
}

/// Backward of [`forward`], applying the sanitize slope exactly once per path.
#[allow(clippy::too_many_arguments)]
pub fn backward(
    input: &[f32],
    fwd: &DeformForward,
    main_weights: &[f32],
    offset_weights: &[f32],
    mask_weights: Option<&[f32]>,
    output_grad: &[f32],
    g: &DeformGeometry,
    max_offset: f32,
) -> Result<DeformForwardGradients, String> {
    let kernels = g.kernels();
    let plane = g.plane();
    let per_batch = 2 * kernels * plane;
    let accum = accumulate_backward(
        &fwd.input_sanitized,
        &fwd.offsets,
        fwd.masks.as_deref(),
        main_weights,
        output_grad,
        g,
    )?;
    // Offsets path: lhtan/sanitize -> normalization Jacobian -> stage1 slope.
    let d_pre = offset_envelope_pre_backward(
        &fwd.raw_offsets,
        &fwd.stats.mean,
        &fwd.stats.inv_std,
        &fwd.stats.normalize,
        &accum.d_offsets,
        2 * kernels,
        plane,
        max_offset,
    )?;
    let d_stage1_norm = offset_normalization_backward(
        &fwd.stage1,
        &fwd.stats.mean,
        &fwd.stats.stddev,
        &fwd.stats.normalize,
        &d_pre,
        per_batch,
    )?;
    let d_offset_predictor: Vec<f32> = d_stage1_norm
        .iter()
        .zip(&fwd.raw_offsets)
        .map(|(&d, &raw)| d * sanitize_slope(raw))
        .collect();
    let offset_grad = predictor_backward_xy(
        &fwd.input_sanitized,
        &fwd.columns,
        offset_weights,
        &d_offset_predictor,
        g.batch,
        g.channels,
        2 * kernels,
        g.input_height,
        g.input_width,
        g.output_height,
        g.output_width,
        g.size,
        g.stride_x,
        g.stride_y,
        g.padding,
        g.dilation,
    )?;
    // Mask path: sigmoid slope of the sanitized logit, then the same predictor.
    let (d_mask_weights, d_mask_biases, d_input_mask) =
        match (mask_weights, fwd.mask_logits.as_ref(), fwd.masks.as_ref()) {
            (Some(weights), Some(logits), Some(_)) => {
                let d_logits: Vec<f32> = accum
                    .d_masks
                    .iter()
                    .zip(logits)
                    .map(|(&d, &logit)| d * mask_sigmoid_slope(logit))
                    .collect();
                let grads = predictor_backward_xy(
                    &fwd.input_sanitized,
                    &fwd.columns,
                    weights,
                    &d_logits,
                    g.batch,
                    g.channels,
                    kernels,
                    g.input_height,
                    g.input_width,
                    g.output_height,
                    g.output_width,
                    g.size,
                    g.stride_x,
                    g.stride_y,
                    g.padding,
                    g.dilation,
                )?;
                (grads.d_weights, grads.d_biases, grads.d_input)
            }
            _ => (Vec::new(), Vec::new(), vec![0.0; fwd.input_sanitized.len()]),
        };
    // The stage functions already applied the *sanitized* input's slope (which
    // is 1); the raw input's slope belongs here, exactly once for the sum.
    let mut d_input = vec![0.0f32; input.len()];
    for i in 0..input.len() {
        d_input[i] = (accum.d_input[i] + offset_grad.d_input[i] + d_input_mask[i])
            * sanitize_slope(input[i]);
    }
    Ok(DeformForwardGradients {
        d_input,
        d_main_weights: accum.d_weights,
        d_offset_weights: offset_grad.d_weights,
        d_offset_biases: offset_grad.d_biases,
        d_mask_weights,
        d_mask_biases,
    })
}

#[cfg(test)]
mod composed_backward_tests {
    use super::*;

    fn check_fd(analytic: f32, base: f32, what: &str, index: usize, f: impl Fn(f32) -> f32) {
        let eps = 1e-2f32;
        let want = (f(base + eps) - f(base - eps)) / (2.0 * eps);
        assert!(
            (analytic - want).abs() < 2e-2 + 2e-2 * want.abs(),
            "{what}[{index}]: analytic {analytic} vs fd {want}"
        );
    }

    struct Fixture {
        input: Vec<f32>,
        main_w: Vec<f32>,
        off_w: Vec<f32>,
        off_b: Vec<f32>,
        mask_w: Vec<f32>,
        mask_b: Vec<f32>,
        seed: Vec<f32>,
        g: DeformGeometry,
    }

    fn fixture() -> Fixture {
        let g = DeformGeometry {
            batch: 2,
            channels: 2,
            filters: 2,
            channels_per_group: 2,
            filters_per_group: 2,
            input_height: 2,
            input_width: 3,
            output_height: 2,
            output_width: 3,
            size: 3,
            stride_x: 1,
            stride_y: 1,
            padding: 1,
            dilation: 1,
        };
        let kernels = g.kernels();
        let plane = g.plane();
        let input: Vec<f32> = (0..g.batch * g.channels * g.input_height * g.input_width)
            .map(|i| ((i * 7 % 13) as f32) * 0.25 - 1.5)
            .collect();
        let main_w: Vec<f32> = (0..g.filters * g.channels_per_group * kernels)
            .map(|i| 0.5 - ((i % 3) as f32) * 0.25)
            .collect();
        // Keep the predictor output (and therefore every sample offset) in a
        // narrow band around 0.45: the sampling base is integral, so an offset
        // near 0 or 1 puts the bilinear cell on a seam where the analytic slope
        // is one-sided and a central difference would disagree.
        let off_w: Vec<f32> = (0..2 * kernels * g.channels * kernels)
            .map(|i| ((i * 5 % 11) as f32) * 0.0002 - 0.001)
            .collect();
        // The sampling base is integral, so offsets near 0 or an integer put a
        // sample on the floor-cell seam where the analytic slope is one-sided.
        // A +0.37 shift keeps every fractional part near 0.37.
        let off_b: Vec<f32> = (0..2 * kernels).map(|i| 0.45 + 0.0005 * i as f32).collect();
        let mask_w: Vec<f32> = (0..kernels * g.channels * kernels)
            .map(|i| ((i * 3 % 7) as f32) * 0.05 - 0.15)
            .collect();
        let mask_b: Vec<f32> = (0..kernels).map(|i| 0.02 * i as f32).collect();
        let seed: Vec<f32> = (0..g.batch * g.filters * plane)
            .map(|i| 0.3 + ((i % 5) as f32) * 0.1)
            .collect();
        Fixture {
            input,
            main_w,
            off_w,
            off_b,
            mask_w,
            mask_b,
            seed,
            g,
        }
    }

    // A large `max_offset` keeps the per-batch gate off and the lhtan band wide,
    // so the finite differences never straddle a piecewise boundary.
    const MAX_OFFSET: f32 = 100.0;

    fn objective(
        f: &Fixture,
        input: &[f32],
        main_w: &[f32],
        off_w: &[f32],
        off_b: &[f32],
        mask_w: &[f32],
        mask_b: &[f32],
    ) -> f32 {
        let fwd = forward(
            input,
            main_w,
            off_w,
            off_b,
            Some(mask_w),
            Some(mask_b),
            &f.g,
            MAX_OFFSET,
        )
        .unwrap();
        fwd.output.iter().zip(&f.seed).map(|(v, s)| v * s).sum()
    }

    #[test]
    fn composed_backward_matches_finite_differences() {
        let f = fixture();
        let fwd = forward(
            &f.input,
            &f.main_w,
            &f.off_w,
            &f.off_b,
            Some(&f.mask_w),
            Some(&f.mask_b),
            &f.g,
            MAX_OFFSET,
        )
        .unwrap();
        let grad = backward(
            &f.input,
            &fwd,
            &f.main_w,
            &f.off_w,
            Some(&f.mask_w),
            &f.seed,
            &f.g,
            MAX_OFFSET,
        )
        .unwrap();

        for i in (0..f.input.len()).step_by(3) {
            check_fd(grad.d_input[i], f.input[i], "d_input", i, |v| {
                let mut x = f.input.clone();
                x[i] = v;
                objective(&f, &x, &f.main_w, &f.off_w, &f.off_b, &f.mask_w, &f.mask_b)
            });
        }
        for i in 0..f.main_w.len() {
            check_fd(grad.d_main_weights[i], f.main_w[i], "d_main_w", i, |v| {
                let mut w = f.main_w.clone();
                w[i] = v;
                objective(&f, &f.input, &w, &f.off_w, &f.off_b, &f.mask_w, &f.mask_b)
            });
        }
        for i in (0..f.off_w.len()).step_by(7) {
            check_fd(grad.d_offset_weights[i], f.off_w[i], "d_off_w", i, |v| {
                let mut w = f.off_w.clone();
                w[i] = v;
                objective(&f, &f.input, &f.main_w, &w, &f.off_b, &f.mask_w, &f.mask_b)
            });
        }
        for i in 0..f.off_b.len() {
            check_fd(grad.d_offset_biases[i], f.off_b[i], "d_off_b", i, |v| {
                let mut b = f.off_b.clone();
                b[i] = v;
                objective(&f, &f.input, &f.main_w, &f.off_w, &b, &f.mask_w, &f.mask_b)
            });
        }
        for i in (0..f.mask_w.len()).step_by(5) {
            check_fd(grad.d_mask_weights[i], f.mask_w[i], "d_mask_w", i, |v| {
                let mut w = f.mask_w.clone();
                w[i] = v;
                objective(&f, &f.input, &f.main_w, &f.off_w, &f.off_b, &w, &f.mask_b)
            });
        }
        for i in 0..f.mask_b.len() {
            check_fd(grad.d_mask_biases[i], f.mask_b[i], "d_mask_b", i, |v| {
                let mut b = f.mask_b.clone();
                b[i] = v;
                objective(&f, &f.input, &f.main_w, &f.off_w, &f.off_b, &f.mask_w, &b)
            });
        }
    }
}

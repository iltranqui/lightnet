//! `[segmentation]`: per-pixel semantic segmentation, CPU reference (Phase 1b of
//! `tasks/rust/layers.md`).
//!
//! Ported from `v6-dev-seg:src-seg/segmentation_layer.cpp`,
//! `seg_gate.cpp` and the `logistic_x_ent_cpu` / `softmax_x_ent_cpu` kernels in
//! `src-lib/blas.cpp`.  The C++ notes:
//!
//! * `classes == 1` is *binary* segmentation: `sigmoid` per pixel plus logistic
//!   cross-entropy.  `classes > 1` is a per-pixel softmax over the class
//!   channels (stride `H * W`, i.e. NCHW) plus softmax cross-entropy against a
//!   one-hot mask.  Both deltas are the plain Darknet descent direction
//!   `truth - prediction`, so the layer is a pass-through for the backward.
//! * The per-element loss is scaled by `lambda_ce * ce_normalizer *
//!   cls_normalizer`, then optionally gated (below), then sanitised
//!   (non-finite -> 0) and clipped to `max_delta`.  The reported cost is
//!   `sum(loss) / (batch * H * W)`, i.e. per pixel, not per element.
//! * The "correct-pixel gate" damps pixels the network already gets right with
//!   confidence: a pixel whose predicted class matches the truth and whose
//!   truth probability is at least `gate_t_correct` has its delta and loss
//!   multiplied by `1 - gate_strength`, and the surviving deltas are then scaled
//!   up by `min(total / active, gate_max_compensation)` so the mean step size
//!   does not shrink with the gate.  `gate_strength` is a raised cosine over
//!   `gate_ramp_iters` after `gate_warmup_iters`.
//! * `temperature` is set to 1 by `make_segmentation_layer` and
//!   `parse_segmentation_section` never reads a cfg key for it, so it is always
//!   1 in C++; it is kept here so the softmax call matches the source exactly.
//!
//! The gate needs the training iteration, which the loss API did not carry; see
//! `crate::dense_loss::compute_dense_training_loss_at`.

use crate::model::{SegmentationLoss, SegmentationSpec};

/// Largest finite `f32`, C++ `FLT_MAX`: the default for `max_delta`.
pub const FLT_MAX: f32 = f32::MAX;

/// Gate bookkeeping (C++ `Darknet::GateStats`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GateStats {
    pub total_pixels: usize,
    pub suppressed_count: usize,
    pub wrong_count: usize,
    pub low_conf_count: usize,
    pub mean_p_y: f32,
}

/// The layer output, and (with truth) the loss and the backward seed.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentationOutput {
    pub output: Vec<f32>,
    pub delta: Option<Vec<f32>>,
    /// Per-pixel cross-entropy: `sum(loss) / (batch * spatial)`.
    pub loss: f32,
    pub gate: Option<GateStats>,
}

/// Gate strength at `iteration`: a raised cosine over `gate_ramp_iters`, starting
/// after `gate_warmup_iters`.  Returns 0 when the gate is disabled.
pub fn gate_strength(spec: &SegmentationSpec, iteration: usize) -> f32 {
    if !spec.gate_enabled {
        return 0.0;
    }
    // C++ computes `(float)(cur_iter - warmup) / max(1, ramp)` in integers first,
    // so iterations before the warmup give a negative numerator, and the clamp
    // folds that back to 0.
    let raw = iteration as i64 - spec.gate_warmup_iters as i64;
    let t = (raw as f32 / spec.gate_ramp_iters.max(1) as f32).clamp(0.0, 1.0);
    0.5 * (1.0 - (std::f32::consts::PI * t).cos())
}

/// The `ce_scale` the C++ applies to both the delta and the per-element loss.
fn ce_scale(spec: &SegmentationSpec) -> f32 {
    spec.lambda_ce * spec.ce_normalizer * spec.cls_normalizer
}

/// Layer forward.  `batch` and `spatial` describe the `[batch, classes, H*W]`
/// (NCHW) layout of `input`, which is also the shape of the truth tensor.
pub fn segmentation_forward(
    input: &[f32],
    spec: &SegmentationSpec,
    batch: usize,
    spatial: usize,
) -> SegmentationOutput {
    let total = batch * spec.classes * spatial;
    debug_assert_eq!(input.len(), total, "segmentation input length");
    let mut output = vec![0.0f32; total];
    if spec.classes == 1 {
        // Binary: C++ copies the input, replaces non-finite values with 0, then
        // applies LOGISTIC.
        for i in 0..total {
            let v = input[i];
            let v = if v.is_finite() { v } else { 0.0 };
            output[i] = 1.0 / (1.0 + (-v).exp()); // C++ LOGISTIC
        }
    } else {
        let temp = spec.temperature.max(1e-6);
        for b in 0..batch {
            let base = b * spec.classes * spatial;
            for p in 0..spatial {
                // C++ `softmax(input + p, classes, temperature, out + p, spatial)`:
                // a numerically stabilised softmax with channel stride `spatial`.
                let mut max = f32::NEG_INFINITY;
                for c in 0..spec.classes {
                    let v = input[base + c * spatial + p] / temp;
                    if v > max {
                        max = v;
                    }
                }
                let mut sum = 0.0f32;
                for c in 0..spec.classes {
                    let e = (input[base + c * spatial + p] / temp - max).exp();
                    output[base + c * spatial + p] = e;
                    sum += e;
                }
                for c in 0..spec.classes {
                    output[base + c * spatial + p] /= sum;
                }
            }
        }
    }
    SegmentationOutput {
        output,
        delta: None,
        loss: 0.0,
        gate: None,
    }
}

/// Cross-entropy on the activated output: the per-element loss, the
/// `truth - prediction` seed, and the C++ post-processing chain (scale, gate,
/// compensation, sanitise, clip).  `gate_strength` is
/// [`gate_strength`] evaluated at the caller's iteration.
pub fn segmentation_loss(
    output: &[f32],
    truth: &[f32],
    spec: &SegmentationSpec,
    batch: usize,
    spatial: usize,
    gate_strength: f32,
) -> SegmentationOutput {
    let total = batch * spec.classes * spatial;
    debug_assert_eq!(output.len(), total);
    debug_assert_eq!(truth.len(), total);
    let mut delta = vec![0.0f32; total];
    let mut loss = vec![0.0f32; total];
    let scale = ce_scale(spec);
    for i in 0..total {
        let t = truth[i];
        let p = output[i];
        let (error, d) = match spec.loss {
            // `logistic_x_ent_cpu`
            SegmentationLoss::Bce => {
                let pc = p.clamp(1e-12, 1.0 - 1e-12);
                (-t * pc.ln() - (1.0 - t) * (1.0 - pc).ln(), t - p)
            }
            // `softmax_x_ent_cpu`: only the truth entries contribute.
            SegmentationLoss::SoftmaxCe => (
                if t != 0.0 {
                    -p.clamp(1e-12, 1.0).ln()
                } else {
                    0.0
                },
                t - p,
            ),
        };
        loss[i] = error;
        delta[i] = d;
    }
    // C++ `scale_component_cpu` scales the loss buffer and the delta together.
    if scale != 1.0 {
        for i in 0..total {
            loss[i] *= scale;
            delta[i] *= scale;
        }
    }
    let mut gate = None;
    if gate_strength > 0.0 {
        let stats = suppress_correct_pixels(
            &mut delta,
            &mut loss,
            output,
            truth,
            batch,
            spec.classes,
            spatial,
            spec.gate_t_correct,
            gate_strength,
        );
        // `apply_gate_compensation_cpu`
        let active = stats.total_pixels.saturating_sub(stats.suppressed_count);
        if active > 0 && active < stats.total_pixels {
            let ratio = stats.total_pixels as f32 / active as f32;
            let compensation = ratio.min(spec.gate_max_compensation);
            if compensation > 1.0 {
                for d in delta.iter_mut() {
                    *d *= compensation;
                }
            }
        }
        gate = Some(stats);
    }
    // `apply_component_delta_cpu`: sanitise, then clip.
    let clipping = spec.max_delta < FLT_MAX;
    for i in 0..total {
        if !loss[i].is_finite() {
            loss[i] = 0.0;
        }
        if !delta[i].is_finite() {
            delta[i] = 0.0;
        }
        if clipping {
            delta[i] = delta[i].clamp(-spec.max_delta, spec.max_delta);
        }
    }
    // C++ `loss_ce = sum_array(l.loss, total) / (batch * spatial)`, summed
    // *after* the scale, the gate and the sanitise - so a gated pixel really
    // does reduce the reported cost.
    let total_loss: f32 = loss.iter().sum();
    SegmentationOutput {
        output: output.to_vec(),
        delta: Some(delta),
        loss: total_loss / (batch * spatial) as f32,
        gate,
    }
}

/// C++ `suppress_correct_pixels_cpu`: scale the delta and loss of every channel
/// of a pixel by `1 - gate_strength` when the pixel is already correct and
/// confident.  Works in place and returns the statistics.
pub fn suppress_correct_pixels(
    delta: &mut [f32],
    loss: &mut [f32],
    output: &[f32],
    truth: &[f32],
    batch: usize,
    classes: usize,
    spatial: usize,
    t_correct: f32,
    gate_strength: f32,
) -> GateStats {
    let mut stats = GateStats::default();
    let mut sum_p_y = 0.0f64;
    for b in 0..batch {
        for s in 0..spatial {
            let (is_match, p_y) = if classes == 1 {
                let p = output[b * spatial + s];
                let y = truth[b * spatial + s];
                ((p > 0.5) == (y > 0.5), if y > 0.5 { p } else { 1.0 - p })
            } else {
                let base = b * classes * spatial;
                let mut y_class = 0usize;
                let mut pred_class = 0usize;
                let mut max_truth = truth[base + s];
                let mut max_pred = output[base + s];
                for c in 1..classes {
                    let idx = base + c * spatial + s;
                    if truth[idx] > max_truth {
                        max_truth = truth[idx];
                        y_class = c;
                    }
                    if output[idx] > max_pred {
                        max_pred = output[idx];
                        pred_class = c;
                    }
                }
                (pred_class == y_class, output[base + y_class * spatial + s])
            };
            stats.total_pixels += 1;
            sum_p_y += p_y as f64;
            let weight = if is_match && p_y >= t_correct {
                stats.suppressed_count += 1;
                1.0 - gate_strength
            } else {
                if is_match {
                    stats.low_conf_count += 1;
                } else {
                    stats.wrong_count += 1;
                }
                1.0
            };
            for c in 0..classes {
                let idx = b * classes * spatial + c * spatial + s;
                delta[idx] *= weight;
                loss[idx] *= weight;
            }
        }
    }
    stats.mean_p_y = if stats.total_pixels > 0 {
        (sum_p_y / stats.total_pixels as f64) as f32
    } else {
        0.0
    };
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(classes: usize) -> SegmentationSpec {
        SegmentationSpec {
            classes,
            temperature: 1.0,
            max_delta: FLT_MAX,
            delta_normalizer: 1.0,
            cls_normalizer: 1.0,
            loss: if classes == 1 {
                SegmentationLoss::Bce
            } else {
                SegmentationLoss::SoftmaxCe
            },
            lambda_ce: 1.0,
            ce_normalizer: 1.0,
            gate_enabled: false,
            gate_t_correct: 0.9,
            gate_warmup_iters: 1000,
            gate_ramp_iters: 2000,
            gate_max_compensation: 10.0,
        }
    }

    #[test]
    fn binary_forward_is_the_logistic_of_a_sanitised_input() {
        let s = spec(1);
        // 1 -> logistic(1) = 0.73105858, -1 -> 0.26894142, NaN -> logistic(0) = 0.5.
        let input = vec![1.0f32, -1.0, f32::NAN];
        let out = segmentation_forward(&input, &s, 3, 1);
        assert!(
            (out.output[0] - 0.731_058_6).abs() < 1e-6,
            "{}",
            out.output[0]
        );
        assert!(
            (out.output[1] - 0.268_941_4).abs() < 1e-6,
            "{}",
            out.output[1]
        );
        assert!((out.output[2] - 0.5).abs() < 1e-7, "{}", out.output[2]);
    }

    #[test]
    fn multiclass_forward_is_a_per_pixel_softmax_over_nchw_channels() {
        let s = spec(3);
        // NCHW: element (class c, pixel p) is at `c * spatial + p`, so the
        // channel stride is `spatial` and the *pixel* stride is 1 - exactly the
        // C++ `softmax(input + p, classes, temperature, out + p, spatial)` call.
        // batch 1, spatial 2, classes 3, both pixels with equal logits.
        let input = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = segmentation_forward(&input, &s, 1, 2);
        for c in 0..3 {
            for p in 0..2 {
                assert!(
                    (out.output[c * 2 + p] - 1.0 / 3.0).abs() < 1e-6,
                    "pixel {p} channel {c}: {}",
                    out.output[c * 2 + p]
                );
            }
        }
        // One pixel, logits (0, ln 2, large): softmax(0, ln 2) has ratio 2.
        let input = vec![0.0f32, std::f32::consts::LN_2, 5.0];
        let out = segmentation_forward(&input, &s, 1, 1);
        let sum = out.output[0] + out.output[1] + out.output[2];
        assert!((sum - 1.0).abs() < 1e-6, "softmax must sum to 1, got {sum}");
        assert!((out.output[1] / out.output[0] - 2.0).abs() < 1e-5);
        assert!(
            out.output[2] > out.output[1] && out.output[1] > out.output[0],
            "the larger logit wins"
        );
        // Hand-computed: exp(0-5), exp(ln2-5), exp(0) over the sum.
        let e0 = (-5.0f32).exp();
        let e1 = (std::f32::consts::LN_2 - 5.0).exp();
        let want1 = e1 / (e0 + e1 + 1.0);
        assert!(
            (out.output[1] - want1).abs() < 1e-6,
            "{} vs {want1}",
            out.output[1]
        );
    }

    #[test]
    fn binary_loss_is_logistic_cross_entropy_per_pixel() {
        let s = spec(1);
        // batch 2, spatial 1, one channel each.
        let output = vec![0.8f32, 0.25];
        let truth = vec![1.0f32, 1.0];
        let out = segmentation_loss(&output, &truth, &s, 2, 1, 0.0);
        let delta = out.delta.unwrap();
        // delta = t - p
        assert!((delta[0] - 0.2).abs() < 1e-6, "{}", delta[0]);
        assert!((delta[1] - 0.75).abs() < 1e-6, "{}", delta[1]);
        // loss = mean over pixels of -t*ln p - (1-t)*ln(1-p), with t = 1 for
        // both pixels: -ln(0.8) and -ln(0.25).
        let want = (-0.8f32.ln() + -0.25f32.ln()) / 2.0;
        assert!(
            (out.loss - want).abs() < 1e-5,
            "got {} want {want}",
            out.loss
        );
    }

    #[test]
    fn multiclass_loss_only_charges_the_truth_channel() {
        let s = spec(3);
        let output = vec![0.2f32, 0.5, 0.3];
        let truth = vec![0.0f32, 1.0, 0.0];
        let out = segmentation_loss(&output, &truth, &s, 1, 1, 0.0);
        let delta = out.delta.unwrap();
        assert!((delta[0] - (-0.2)).abs() < 1e-6);
        assert!((delta[1] - 0.5).abs() < 1e-6);
        assert!((delta[2] - (-0.3)).abs() < 1e-6);
        assert!((out.loss - (-0.5f32.ln())).abs() < 1e-5, "got {}", out.loss);
    }

    #[test]
    fn ce_scale_multiplies_the_loss_and_the_seed() {
        let mut s = spec(1);
        s.lambda_ce = 2.0;
        s.ce_normalizer = 3.0;
        s.cls_normalizer = 0.5;
        let plain = spec(1);
        let output = vec![0.7f32];
        let truth = vec![1.0f32];
        let scaled = segmentation_loss(&output, &truth, &s, 1, 1, 0.0);
        let base = segmentation_loss(&output, &truth, &plain, 1, 1, 0.0);
        assert!(
            (scaled.loss - base.loss * 3.0).abs() < 1e-5,
            "{} vs {}",
            scaled.loss,
            base.loss * 3.0
        );
        let d = scaled.delta.unwrap();
        assert!((d[0] - base.delta.unwrap()[0] * 3.0).abs() < 1e-6);
    }

    #[test]
    fn max_delta_clips_the_seed_after_the_loss() {
        let mut s = spec(1);
        s.max_delta = 0.1;
        // A confident wrong prediction gives delta = t - p = -1.
        let output = vec![1.0f32];
        let truth = vec![0.0f32];
        let out = segmentation_loss(&output, &truth, &s, 1, 1, 0.0);
        let d = out.delta.unwrap();
        assert!(
            (d[0] - (-0.1)).abs() < 1e-6,
            "clipped to -0.1, got {}",
            d[0]
        );
    }

    #[test]
    fn gate_strength_is_a_raised_cosine_after_the_warmup() {
        let mut s = spec(2);
        s.gate_enabled = true;
        assert_eq!(gate_strength(&s, 0), 0.0, "before the warmup");
        assert_eq!(gate_strength(&s, 1000), 0.0, "at the warmup boundary");
        // Halfway through the ramp: 0.5 * (1 - cos(pi/2)) = 0.5.
        assert!(
            (gate_strength(&s, 2000) - 0.5).abs() < 1e-6,
            "{}",
            gate_strength(&s, 2000)
        );
        assert!(
            (gate_strength(&s, 3000) - 1.0).abs() < 1e-6,
            "{}",
            gate_strength(&s, 3000)
        );
        assert!(
            (gate_strength(&s, 9999) - 1.0).abs() < 1e-6,
            "saturates at 1"
        );
        s.gate_enabled = false;
        assert_eq!(gate_strength(&s, 3000), 0.0, "disabled gate is always 0");
    }

    #[test]
    fn gate_suppresses_correct_confident_pixels_and_compensates() {
        let s = spec(1);
        // 4 pixels: 2 already right and confident (p = 0.95 with y = 1), 2 wrong.
        let output = vec![0.95f32, 0.95, 0.05, 0.05];
        let truth = vec![1.0f32, 1.0, 1.0, 1.0];
        let strength = 0.5;
        let base = segmentation_loss(&output, &truth, &s, 4, 1, 0.0);
        let gated = segmentation_loss(&output, &truth, &s, 4, 1, strength);
        let stats = gated.gate.unwrap();
        assert_eq!(stats.total_pixels, 4);
        assert_eq!(stats.suppressed_count, 2);
        assert_eq!(stats.wrong_count, 2);
        assert_eq!(stats.low_conf_count, 0);
        let d = gated.delta.unwrap();
        let b = base.delta.unwrap();
        // Pixels 0,1: gated by (1 - 0.5), then the whole buffer is compensated
        // by total/active = 4/2 = 2, so the net factor is 1.
        assert!(
            (d[0] - b[0]).abs() < 1e-6,
            "suppressed then compensated: {} vs {}",
            d[0],
            b[0]
        );
        // Pixels 2,3 are wrong, so the gate leaves them alone - but the C++
        // compensation scales the *whole* delta buffer, so they get exactly the
        // 2x too. That is what keeps the mean step size constant.
        assert!(
            (d[2] - b[2] * 2.0).abs() < 1e-6,
            "wrong pixel gets only the compensation: {} vs {}",
            d[2],
            b[2] * 2.0
        );
        assert!((d[3] - b[3] * 2.0).abs() < 1e-6);
        // With the compensation disabled the gated pixels really are damped.
        let mut s2 = s.clone();
        s2.gate_max_compensation = 1.0;
        let ungated_total = segmentation_loss(&output, &truth, &s2, 4, 1, 0.0);
        let gated_total = segmentation_loss(&output, &truth, &s2, 4, 1, strength);
        let dt = gated_total.delta.unwrap();
        assert!(
            (dt[0] - ungated_total.delta.unwrap()[0] * 0.5).abs() < 1e-6,
            "damped by 1 - strength"
        );
        // The loss is gated the same way, so the mean per-pixel loss drops.
        assert!(gated.loss < base.loss, "{} vs {}", gated.loss, base.loss);
    }

    #[test]
    fn gate_leaves_everything_alone_when_nothing_matches() {
        let s = spec(1);
        let output = vec![0.1f32, 0.2];
        let truth = vec![1.0f32, 1.0];
        let base = segmentation_loss(&output, &truth, &s, 2, 1, 0.0);
        let gated = segmentation_loss(&output, &truth, &s, 2, 1, 0.75);
        assert_eq!(gated.delta, base.delta);
        let stats = gated.gate.unwrap();
        assert_eq!(stats.suppressed_count, 0);
        assert_eq!(stats.wrong_count, 2);
    }

    #[test]
    fn multiclass_gate_uses_the_argmax_class() {
        let s = spec(3);
        // NCHW with spatial 2: class c of pixel p is at `c * 2 + p`.
        // Pixel 0: truth class 2, prediction class 2 with p = 0.95 -> suppressed.
        // Pixel 1: truth class 1, prediction class 0 -> wrong.
        let output = vec![0.02f32, 0.6, 0.03, 0.3, 0.95, 0.1];
        let truth = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];
        let gated = segmentation_loss(&output, &truth, &s, 1, 2, 0.5);
        let stats = gated.gate.unwrap();
        assert_eq!(stats.suppressed_count, 1, "pixel 0 is right and confident");
        assert_eq!(stats.wrong_count, 1, "pixel 1 predicts the wrong class");
        assert_eq!(stats.low_conf_count, 0);
        assert!(
            (stats.mean_p_y - (0.95 + 0.3) / 2.0).abs() < 1e-6,
            "mean truth probability {} (p_y is the predicted probability of the truth class)",
            stats.mean_p_y
        );
    }

    #[test]
    fn seeds_are_the_exact_derivatives_of_the_loss() {
        // Central finite differences of `L = sum(seed * output)` must match the
        // seed, for both loss types and with the scales active.
        for classes in [1usize, 3] {
            let mut s = spec(classes);
            s.cls_normalizer = 1.7;
            s.lambda_ce = 0.6;
            // No clipping: the clip is a deliberate kink, checked separately by
            // `max_delta_clips_the_seed_after_the_loss`.
            s.max_delta = FLT_MAX;
            let batch = 2;
            let spatial = 2;
            let input: Vec<f32> = (0..batch * classes * spatial)
                .map(|i| ((i * 7 % 11) as f32 - 5.0) / 4.0)
                .collect();
            // One-hot (or binary) truth, one entry per pixel, in the layer's
            // `[batch, classes, H*W]` layout - the batch offset matters, two
            // images' pixels otherwise collide on the same slots.
            let mut truth = vec![0.0f32; batch * classes * spatial];
            for b in 0..batch {
                let base = b * classes * spatial;
                for p in 0..spatial {
                    let pixel = b * spatial + p;
                    if classes == 1 {
                        truth[base + p] = if pixel % 2 == 0 { 1.0 } else { 0.0 };
                    } else {
                        truth[base + (pixel % classes) * spatial + p] = 1.0;
                    }
                }
            }
            let out = segmentation_forward(&input, &s, batch, spatial);
            let result = segmentation_loss(&out.output, &truth, &s, batch, spatial, 0.0);
            let seed = result.delta.unwrap();
            let eps = 1e-3f32;
            for i in 0..input.len() {
                let mut hi = input.clone();
                hi[i] += eps;
                let mut lo = input.clone();
                lo[i] -= eps;
                let o_hi = segmentation_forward(&hi, &s, batch, spatial);
                let o_lo = segmentation_forward(&lo, &s, batch, spatial);
                let l_hi = segmentation_loss(&o_hi.output, &truth, &s, batch, spatial, 0.0);
                let l_lo = segmentation_loss(&o_lo.output, &truth, &s, batch, spatial, 0.0);
                // `loss` is a per-pixel mean and the seed is Darknet's *descent
                // direction* `target - prediction`, so d(loss)/d(input) is the
                // negative of the seed (up to the 1/(batch*spatial) the mean
                // divides by).
                let numeric = (l_hi.loss - l_lo.loss) / (2.0 * eps);
                let analytic = seed[i] / (batch * spatial) as f32;
                let slack = 3e-2 * analytic.abs().max(1e-2);
                assert!(
                    (numeric + analytic).abs() <= slack,
                    "classes={classes} i={i}: numeric {numeric} vs -seed {analytic}"
                );
            }
        }
    }
}

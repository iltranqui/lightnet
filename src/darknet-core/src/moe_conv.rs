//! `[moe_convolutional]` router maths (Phase 2B of `tasks/rust/layers.md`),
//! ported from `Jev_laya:src-lib/moe_convolutional_layer.cpp`.
//!
//! The layer owns `experts` ordinary Darknet convolutional layers plus a tiny
//! router.  This module holds only the router-side maths, which depends on
//! nothing but numbers; `runtime.rs` does the expert convolution and blending so
//! it can reuse the existing convolution forward/backward.
//!
//! ```text
//! per image b:
//!   pooled[c]        = mean over H*W of input[b, c, :, :]        (global average pool)
//!   logits[e]        = (router_bias[e] + sum_c router_w[e,c]*pooled[c]) / temperature
//!   gates[b,e]       = softmax_topk(logits, top_k)
//!   output[b]        = sum_e gates[b,e] * expert_e(x)[b]
//! ```
//!
//! `load_balance` adds `alpha * max(0, E * sum_e(mean_gate[e]^2) - 1)`, which is
//! zero for a uniform router (`sum_e (1/E)^2 = 1/E`) and grows as traffic
//! collapses onto one expert.

#[cfg(test)]
use crate::model::Activation;
use crate::model::{ConvolutionalSpec, MoeSpec};

/// Largest finite `f32`, C++ `FLT_MAX`: the sentinel a masked logit gets.
const NEG_INF: f32 = f32::MAX;

/// C++ `softmax_topk`: temperature-scaled softmax over the router logits with an
/// optional top-k mask.
///
/// `top_k` of `0` means soft (every expert), `>= experts` means no mask.  The
/// mask keeps every logit `>= ` the k-th largest, which on ties keeps *more* than
/// `k` experts - C++'s `nth_element` + `v < kth` test, reproduced here because
/// silently dropping a tied expert would change the gate normalisation.
pub fn softmax_topk(logits: &[f32], top_k: usize) -> Vec<f32> {
    let experts = logits.len();
    assert!(experts > 0, "softmax_topk: no experts");
    let mut masked = logits.to_vec();
    if top_k > 0 && top_k < experts {
        let mut keep = logits.to_vec();
        keep.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let kth = keep[top_k - 1];
        for v in masked.iter_mut() {
            if *v < kth {
                *v = -NEG_INF;
            }
        }
    }
    let max_logit = masked.iter().copied().fold(-NEG_INF, f32::max);
    let mut out = vec![0.0f32; experts];
    let mut sum = 0.0f32;
    for (e, out) in out.iter_mut().enumerate() {
        let value = if masked[e] <= -NEG_INF * 0.5 { 0.0 } else { (masked[e] - max_logit).exp() };
        *out = value;
        sum += value;
    }
    if sum <= 0.0 || !sum.is_finite() {
        // C++'s defensive fallback: a uniform mixture keeps the layer usable
        // instead of propagating NaNs through the expert blend.
        return vec![1.0 / experts as f32; experts];
    }
    for v in out.iter_mut() {
        *v /= sum;
    }
    out
}

/// C++ `router_temperature`.
pub fn router_temperature(spec: &MoeSpec) -> f32 {
    spec.temperature.max(1e-6)
}

/// C++ `compute_cpu_router`: global average pool, linear router, softmax+top-k.
/// Returns the `[batch, experts]` gate row-major.
pub fn compute_router(
    input: &[f32],
    batch: usize,
    channels: usize,
    spatial: usize,
    router_weights: &[f32],
    router_biases: &[f32],
    spec: &MoeSpec,
) -> Vec<f32> {
    let experts = spec.experts;
    assert_eq!(router_weights.len(), experts * channels, "router weights must be experts x channels");
    assert_eq!(router_biases.len(), experts, "router biases must have one entry per expert");
    assert_eq!(input.len(), batch * channels * spatial, "router input length");
    let inv_spatial = if spatial > 0 { 1.0 / spatial as f32 } else { 0.0 };
    let temperature = router_temperature(spec);
    let mut gates = vec![0.0f32; batch * experts];
    let mut logits = vec![0.0f32; experts];
    for b in 0..batch {
        let mut pooled = vec![0.0f32; channels];
        for (c, pooled_c) in pooled.iter_mut().enumerate() {
            let base = b * channels * spatial + c * spatial;
            let sum: f32 = input[base..base + spatial].iter().sum();
            *pooled_c = sum * inv_spatial;
        }
        for e in 0..experts {
            let mut v = router_biases[e];
            let w = &router_weights[e * channels..(e + 1) * channels];
            for c in 0..channels {
                v += w[c] * pooled[c];
            }
            logits[e] = v / temperature;
        }
        gates[b * experts..(b + 1) * experts]
            .copy_from_slice(&softmax_topk(&logits, spec.top_k));
    }
    gates
}

/// C++ `compute_balance_loss_and_raw_grad`: the auxiliary load-balance loss and
/// its gradient contribution to `raw_gate_grad`.
///
/// Returns `(loss, raw_gate_grad)` with `raw_gate_grad` `[batch, experts]`.  The
/// reported loss is the *clamped* value, while the gradient below is that of the
/// unclamped expression (C++'s comment says so explicitly): the clamp only stops
/// a balanced router from being penalised, it must not zero a gradient that
/// would otherwise pull the router back toward balance.
pub fn balance_loss_and_raw_grad(
    gates: &[f32],
    batch: usize,
    spec: &MoeSpec,
) -> (f32, Vec<f32>) {
    let experts = spec.experts;
    let mut raw = vec![0.0f32; batch * experts];
    if spec.load_balance <= 0.0 || batch == 0 {
        return (0.0, raw);
    }
    let mut mean_gate = vec![0.0f32; experts];
    for b in 0..batch {
        for e in 0..experts {
            mean_gate[e] += gates[b * experts + e];
        }
    }
    for m in mean_gate.iter_mut() {
        *m /= batch as f32;
    }
    let sum_sq: f32 = mean_gate.iter().map(|m| m * m).sum();
    let balance = spec.load_balance * (experts as f32 * sum_sq - 1.0).max(0.0);
    for b in 0..batch {
        for e in 0..experts {
            raw[b * experts + e] +=
                spec.load_balance * 2.0 * experts as f32 * mean_gate[e] / batch as f32;
        }
    }
    (balance, raw)
}

/// C++ `softmax_topk`'s backward, fused with the temperature:
/// `dlogit[b,e] = gate[b,e] * (raw[b,e] - sum_j raw[b,j] * gate[b,j]) / temperature`.
///
/// Masked experts have gate 0 and therefore a zero gradient, which is what makes
/// top-k sparse rather than soft.
pub fn router_gate_gradient(gates: &[f32], raw: &[f32], spec: &MoeSpec) -> Vec<f32> {
    let experts = spec.experts;
    let temperature = router_temperature(spec);
    let mut out = vec![0.0f32; gates.len()];
    for b in 0..gates.len() / experts.max(1) {
        let row = &gates[b * experts..(b + 1) * experts];
        let raw_row = &raw[b * experts..(b + 1) * experts];
        let weighted: f32 = row.iter().zip(raw_row).map(|(g, r)| g * r).sum();
        for e in 0..experts {
            out[b * experts + e] = row[e] * (raw_row[e] - weighted) / temperature;
        }
    }
    out
}

/// C++ `compute_cpu_router`'s pooled descriptor, reused by the backward: the
/// router's input gradient needs the same pooled vector the forward used.
pub fn pool_for_router(input: &[f32], batch: usize, channels: usize, spatial: usize) -> Vec<f32> {
    let inv_spatial = if spatial > 0 { 1.0 / spatial as f32 } else { 0.0 };
    let mut pooled = vec![0.0f32; batch * channels];
    for b in 0..batch {
        for c in 0..channels {
            let base = b * channels * spatial + c * spatial;
            pooled[b * channels + c] = input[base..base + spatial].iter().sum::<f32>() * inv_spatial;
        }
    }
    pooled
}

/// The `.weights` size of one expert, matching C++'s
/// `save_convolutional_weights` block for `expert`.
pub fn expert_weights_len(expert: &ConvolutionalSpec) -> usize {
    let k = expert.size * expert.size * (expert.filters / expert.groups.max(1));
    expert.filters + k
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(experts: usize, top_k: usize) -> MoeSpec {
        MoeSpec {
            experts,
            top_k,
            load_balance: 0.01,
            temperature: 1.0,
            expert_dropout: 0.0,
            debug_interval: 0,
            expert: ConvolutionalSpec {
                filters: 8,
                size: 1,
                stride_x: 1,
                stride_y: 1,
                dilation: 1,
                padding: 0,
                groups: 1,
                share_index: None,
                batch_normalize: false,
                activation: Activation::Linear,
            },
        }
    }

    #[test]
    fn softmax_matches_the_hand_computation() {
        let gates = softmax_topk(&[0.0, 0.0, 0.0, 0.0], 0);
        assert_eq!(gates, vec![0.25; 4]);
        let gates = softmax_topk(&[1.0, 0.0, 0.0, 0.0], 0);
        // Reference: shift by the max, so e0 gets exp(0) = 1 and the rest
        // exp(-1).
        let want = 1.0 / (1.0 + 3.0 * (-1.0f32).exp());
        assert!((gates[0] - want).abs() < 1e-6, "{} vs {want}", gates[0]);
        // Subtracting the max is what keeps it stable.
        let gates = softmax_topk(&[1000.0, 999.0, 998.0, 997.0], 0);
        assert!(gates.iter().all(|g| g.is_finite()), "{gates:?}");
        assert!(gates[0] > gates[3]);
    }

    #[test]
    fn top_k_masks_the_weak_experts_exactly() {
        // 0 < top_k < experts: only the two largest survive.
        let gates = softmax_topk(&[4.0, 3.0, 2.0, 1.0], 2);
        assert_eq!(gates[2], 0.0, "masked experts get exactly zero gate");
        assert_eq!(gates[3], 0.0);
        assert_eq!(gates[0] + gates[1], 1.0, "the kept gates renormalise to 1");
        assert!(gates[0] > gates[1]);
        // top_k >= experts and top_k == 0 both mean "no mask".
        assert_eq!(softmax_topk(&[4.0, 3.0, 2.0, 1.0], 4), softmax_topk(&[4.0, 3.0, 2.0, 1.0], 0));
        // top_k == 1 keeps a single expert with gate 1.
        let gates = softmax_topk(&[4.0, 3.0, 2.0, 1.0], 1);
        assert_eq!(gates, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn ties_survive_the_top_k_mask_like_cxx() {
        // C++ masks with `v < kth`, so tied experts are *kept*: three equal
        // maxima with top_k = 1 keep all three and split the mass three ways.
        // Reproducing that is deliberate - dropping a tie would change the
        // normalisation.
        let gates = softmax_topk(&[5.0, 5.0, 5.0, 1.0], 1);
        assert!((gates[0] - 1.0 / 3.0).abs() < 1e-6, "{gates:?}");
        assert!((gates[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((gates[2] - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(gates[3], 0.0);
    }

    #[test]
    fn a_pathological_row_falls_back_to_uniform() {
        // Every logit masked away would make exp() sum to zero; C++ returns a
        // uniform mixture instead of NaNs.
        let gates = softmax_topk(&[-f32::MAX, -f32::MAX, -f32::MAX, -f32::MAX], 1);
        assert_eq!(gates, vec![0.25; 4], "{gates:?}");
    }

    #[test]
    fn the_router_pools_and_applies_the_linear_map() {
        let batch = 2;
        let channels = 2;
        let spatial = 4;
        // Image 0: channel 0 all 1, channel 1 all 0. Image 1: channel 0 all 0,
        // channel 1 all 3. Layout is [b][c][hw].
        let mut input = vec![0.0f32; batch * channels * spatial];
        for i in 0..spatial {
            input[0 * channels * spatial + 0 * spatial + i] = 1.0;
            input[0 * channels * spatial + 1 * spatial + i] = 0.0;
            input[1 * channels * spatial + 0 * spatial + i] = 0.0;
            input[1 * channels * spatial + 1 * spatial + i] = 3.0;
        }
        let rw = vec![1.0, 0.0, 0.0, 1.0]; // [experts=2, channels=2]
        let rb = vec![0.0, 0.0];
        let gates = compute_router(&input, batch, channels, spatial, &rw, &rb, &spec(2, 0));
        // Image 0: logits (1, 0) -> softmax. Image 1: logits (0, 3) -> softmax.
        let e = 1.0f32.exp();
        let want0 = [e / (e + 1.0), 1.0 / (e + 1.0)];
        assert!((gates[0] - want0[0]).abs() < 1e-6, "{gates:?}");
        // Image 1's logit gap is 3, not 1.
        let e3 = 3.0f32.exp();
        let want1 = [1.0 / (1.0 + e3), e3 / (1.0 + e3)];
        assert!((gates[2] - want1[0]).abs() < 1e-6, "{gates:?}");
        assert!((gates[3] - want1[1]).abs() < 1e-6);
        for b in 0..batch {
            let row: f32 = gates[b * 2..(b + 1) * 2].iter().sum();
            assert!((row - 1.0).abs() < 1e-6, "gates must be a distribution");
        }
    }

    #[test]
    fn the_temperature_flattens_the_gates() {
        let input = vec![1.0f32; 1 * 1 * 2];
        let mut hot = spec(2, 0);
        hot.temperature = 1.0;
        let mut cold = spec(2, 0);
        cold.temperature = 8.0;
        // With one expert the gate is always 1, so use two experts to see the
        // effect: expert 0 always wins at T=1 and is much softer at T=8.
        let rw2 = vec![2.0, 0.0];
        let rb2 = vec![0.0, 0.0];
        let g_hot = compute_router(&input, 1, 1, 2, &rw2, &rb2, &hot);
        let g_cold = compute_router(&input, 1, 1, 2, &rw2, &rb2, &cold);
        assert!(g_hot[0] > g_cold[0], "a higher temperature must flatten: {} vs {}", g_hot[0], g_cold[0]);
    }

    #[test]
    fn a_uniform_router_has_no_load_balance_loss() {
        let s = spec(4, 0);
        let gates = vec![0.25f32; 8 * 4];
        let (loss, raw) = balance_loss_and_raw_grad(&gates, 8, &s);
        // sum_e (1/E)^2 = 1/E, so E * that = 1 and the penalty is zero.
        assert!(loss.abs() < 1e-6, "{loss}");
        // The unclamped gradient is still non-zero, as C++ notes.
        assert!(raw.iter().any(|v| v.abs() > 0.0), "{raw:?}");
    }

    #[test]
    fn collapsed_routing_is_penalised() {
        let s = spec(4, 0);
        // All traffic on expert 0: sum_e mean_gate^2 = 1, so E*1 - 1 = 3.
        let mut gates = vec![0.0f32; 4 * 4];
        for b in 0..4 {
            gates[b * 4] = 1.0;
        }
        let (loss, raw) = balance_loss_and_raw_grad(&gates, 4, &s);
        assert!((loss - 0.01 * 3.0).abs() < 1e-6, "{loss}");
        // raw grad = alpha * 2 * E * mean_gate / B, so only expert 0 is pushed.
        for (b, row) in raw.chunks(4).enumerate() {
            assert!((row[0] - 0.01 * 2.0 * 4.0 * 1.0 / 4.0).abs() < 1e-6, "b={b}: {row:?}");
            assert!(
                row[1..].iter().all(|v| *v == 0.0),
                "b={b}: unused experts must not move: {row:?}"
            );
        }
        // Disabling load balancing gives no loss and no gradient.
        let mut off = s.clone();
        off.load_balance = 0.0;
        let (loss, raw) = balance_loss_and_raw_grad(&gates, 4, &off);
        assert_eq!(loss, 0.0);
        assert!(raw.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn the_load_balance_gradient_is_the_exact_derivative() {
        // Central finite differences of `balance_loss` with respect to one gate,
        // through the mean, compared with `raw_gate_grad` for that gate.
        let experts = 3usize;
        let batch = 5usize;
        let mut s = spec(experts, 0);
        s.load_balance = 0.7;
        let mut rng = 0u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 33) as f32 / (1u64 << 31) as f32).clamp(0.01, 0.9)
        };
        let mut gates = vec![0.0f32; batch * experts];
        for g in gates.iter_mut() {
            *g = next();
        }
        let (loss, raw) = balance_loss_and_raw_grad(&gates, batch, &s);
        let eps = 1e-3f32;
        for i in 0..gates.len() {
            let mut hi = gates.clone();
            hi[i] += eps;
            let mut lo = gates.clone();
            lo[i] -= eps;
            let (l_hi, _) = balance_loss_and_raw_grad(&hi, batch, &s);
            let (l_lo, _) = balance_loss_and_raw_grad(&lo, batch, &s);
            let numeric = (l_hi - l_lo) / (2.0 * eps);
            assert!(
                (numeric - raw[i]).abs() <= 1e-3 * raw[i].abs().max(1e-3),
                "gate {i}: numeric {numeric} analytic {} (loss {loss})",
                raw[i]
            );
        }
    }

    #[test]
    fn the_gate_gradient_is_the_softmax_jacobian_with_the_temperature() {
        let experts = 3usize;
        let batch = 2usize;
        let mut s = spec(experts, 0);
        s.temperature = 2.0;
        let gates = vec![0.2f32, 0.3, 0.5, 0.1, 0.6, 0.3];
        let raw = vec![1.0f32, -0.5, 0.25, 2.0, 0.0, -1.0];
        let dlogit = router_gate_gradient(&gates, &raw, &s);
        for b in 0..batch {
            let row = &gates[b * experts..(b + 1) * experts];
            let raw_row = &raw[b * experts..(b + 1) * experts];
            let weighted: f32 = row.iter().zip(raw_row).map(|(g, r)| g * r).sum();
            for e in 0..experts {
                let want = row[e] * (raw_row[e] - weighted) / 2.0;
                assert!(
                    (dlogit[b * experts + e] - want).abs() < 1e-6,
                    "b={b} e={e}: {} vs {want}",
                    dlogit[b * experts + e]
                );
            }
        }
        // And it must agree with the finite-difference derivative of the softmax
        // (with the temperature folded in), for an unmasked row.
        let logits = [0.7f32, -0.3, 1.1];
        let raw_fixed = [0.4f32, -1.2, 0.8];
        let gates = softmax_topk(&logits, 0);
        let d = router_gate_gradient(&gates, &raw_fixed, &spec(experts, 0));
        // d(sum_e raw_e * gate_e(logits/temp)) / d(logit_e / temp) is the softmax
        // Jacobian, so scale by 1/temp against the logits.
        for e in 0..experts {
            let mut hi = logits;
            hi[e] += eps_per_temp();
            let mut lo = logits;
            lo[e] -= eps_per_temp();
            let g_hi = softmax_topk(&hi, 0);
            let g_lo = softmax_topk(&lo, 0);
            let numeric = (0..experts)
                .map(|j| raw_fixed[j] * (g_hi[j] - g_lo[j]))
                .sum::<f32>()
                / (2.0 * eps_per_temp());
            assert!(
                (numeric - d[e]).abs() < 2e-2 * numeric.abs().max(1e-3),
                "e={e}: numeric {numeric} analytic {}",
                d[e]
            );
        }
    }

    fn eps_per_temp() -> f32 {
        1e-2
    }
}

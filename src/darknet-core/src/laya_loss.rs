//! `[laya_loss]`: the LAYA masked soft-target loss over the scorer's K option
//! logits (Phase 1b of `tasks/rust/layers.md`), ported from
//! `Jev_laya:src-lib/laya_loss_layer.cpp`.
//!
//! Two modes:
//!  * `ce`   — `L = -sum y log softmax(l)` over the valid options (masked softmax).
//!  * `rlcd` — the upstream RLCD fine-tuning objective: Gaussian-noised copies of
//!             the logits scored by a strictly-proper reward, a group-mean
//!             baseline normalised by the batch std, and a REINFORCE term plus a
//!             weighted cross-entropy.
//!
//! Truth is `[target K | valid K | qtype | ep_group | ep_step]` (stride `2K+3`).
//! Samples with no valid option are episode padding: no loss, no delta, and left
//! out of every mean.
//!
//! **Gradient convention.** `delta` is Darknet's descent direction
//! (`-dL/dlogit`): for `ce` it is `y - p` on valid slots and 0 elsewhere.  The
//! `rlcd` term is a REINFORCE surrogate (upstream differentiates only
//! `-adv * logp`, treating the group baseline as a constant), so its gradient is
//! the intended estimator rather than the exact derivative of the reported loss
//! — ported as-is, matching C++.

use crate::model::LayaLossSpec;

/// The three outputs of [`laya_loss_compute`].
#[derive(Clone, Debug, PartialEq)]
pub struct LayaLossOutput {
    /// Mean loss over the samples with at least one valid option.
    pub loss: f32,
    /// `[N, K]` masked softmax probabilities (the layer's forward output).
    pub probs: Vec<f32>,
    /// `[N, K]` descent-direction delta (`-dL/dlogit`).
    pub delta: Vec<f32>,
}

/// C++ `masked_softmax`: softmax over the valid slots only (invalid -> 0).
fn masked_softmax(z: &[f32], valid: &[f32], k: usize, p: &mut [f64]) {
    let mut mx = -1.0e300f64;
    for j in 0..k {
        if valid[j] > 0.5 {
            mx = mx.max(z[j] as f64);
        }
    }
    let mut sum = 0.0f64;
    for j in 0..k {
        p[j] = if valid[j] > 0.5 {
            (z[j] as f64 - mx).exp()
        } else {
            0.0
        };
        sum += p[j];
    }
    for value in p.iter_mut().take(k) {
        *value = if sum > 0.0 { *value / sum } else { 0.0 };
    }
}

/// C++ `proper_reward` (`rl_common.py`), for one reported distribution `q`.
#[allow(clippy::too_many_arguments)]
fn proper_reward(
    q: &[f64],
    y: &[f32],
    valid: &[f32],
    k: usize,
    qtype: i32,
    w_sph: f32,
    w_rps: f32,
    log_floor: f32,
) -> f64 {
    let (mut log_score, mut dot, mut qn) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..k {
        let qk = if valid[j] > 0.5 { q[j] } else { 0.0 };
        let lq = (qk.max(1.0e-12).ln()).max(log_floor as f64);
        log_score += y[j] as f64 * lq;
        dot += y[j] as f64 * qk;
        qn += qk * qk;
    }
    let mut r = log_score + w_sph as f64 * dot / qn.sqrt().max(1.0e-9);
    if qtype == 1 {
        let mut k_valid = 0i32;
        let (mut cq, mut ct, mut rps) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..k {
            let m = if valid[j] > 0.5 { 1.0 } else { 0.0 };
            if m > 0.0 {
                k_valid += 1;
            }
            cq += q[j] * m;
            ct += y[j] as f64;
            rps += (cq - ct) * (cq - ct) * m;
        }
        r -= w_rps as f64 * rps / 1.0f64.max((k_valid.max(2) - 1) as f64);
    }
    r
}

/// C++ `laya_loss_compute`.
///
/// `logits` is `[N, K]`, `truth` is `[N, 2K+1]` (compact) and `znorm` is the
/// caller's `[G, N, K]` standard-normal draws (required for `rlcd`, ignored for
/// `ce`).  `sigma` is the current RLCD noise scale.
pub fn laya_loss_compute(
    logits: &[f32],
    truth: &[f32],
    n: usize,
    k: usize,
    spec: &LayaLossSpec,
    sigma: f32,
    znorm: Option<&[f32]>,
) -> Result<LayaLossOutput, String> {
    if k == 0 || n == 0 {
        return Err("laya_loss: batch and max_options must be > 0".to_owned());
    }
    let stride = 2 * k + 1;
    if truth.len() != n * stride {
        return Err(format!(
            "laya_loss: truth has {} values, expected N*(2K+1)={}",
            truth.len(),
            n * stride
        ));
    }
    if logits.len() != n * k {
        return Err(format!(
            "laya_loss: logits have {} values, expected N*K={}",
            logits.len(),
            n * k
        ));
    }
    let mode = if spec.rlcd { 1 } else { 0 };
    let g = spec.group.max(1);
    if mode == 1 {
        let needed = g * n * k;
        match znorm {
            Some(z) if z.len() == needed => {}
            Some(z) => {
                return Err(format!(
                    "laya_loss: znorm has {} values, expected G*N*K={needed}",
                    z.len()
                ));
            }
            None => return Err("laya_loss: rlcd mode requires znorm".to_owned()),
        }
    }

    let mut probs = vec![0.0f32; n * k];
    let mut delta = vec![0.0f32; n * k];
    let mut p = vec![0.0f64; k];
    let mut ce_sum = 0.0f64;

    // Samples with no valid option are batch padding.
    let mut real = vec![false; n];
    let mut n_real = 0usize;
    for sample in 0..n {
        let valid = &truth[sample * stride + k..sample * stride + 2 * k];
        real[sample] = valid.iter().any(|v| *v > 0.5);
        if real[sample] {
            n_real += 1;
        }
    }

    for sample in 0..n {
        let y = &truth[sample * stride..sample * stride + stride];
        let valid = &y[k..2 * k];
        masked_softmax(&logits[sample * k..(sample + 1) * k], valid, k, &mut p);
        for j in 0..k {
            probs[sample * k + j] = p[j] as f32;
            let v = valid[j] > 0.5;
            delta[sample * k + j] = if v {
                let scale = if mode == 1 { spec.w_ce } else { 1.0 };
                (scale * (y[j] - p[j] as f32)) as f32
            } else {
                0.0
            };
            if v && y[j] > 0.0 {
                ce_sum -= y[j] as f64 * p[j].max(1.0e-30).ln();
            }
        }
    }
    let mut loss = ce_sum / 1.0f64.max(n_real as f64) as f64;
    if mode != 1 {
        return Ok(LayaLossOutput {
            loss: loss as f32,
            probs,
            delta,
        });
    }

    let znorm = znorm.expect("validated above");
    // RLCD: eps = znorm * sigma on valid slots, re-centred to zero mean.
    let mut eps = vec![0.0f64; g * n * k];
    let mut r = vec![0.0f64; g * n];
    let mut q = vec![0.0f64; k];
    for gg in 0..g {
        for sample in 0..n {
            let y = &truth[sample * stride..sample * stride + stride];
            let valid = &y[k..2 * k];
            let e = &mut eps[(gg * n + sample) * k..(gg * n + sample) * k + k];
            let mut mean = 0.0f64;
            let mut kv = 0i32;
            for j in 0..k {
                if valid[j] > 0.5 {
                    e[j] = znorm[(gg * n + sample) * k + j] as f64 * sigma as f64;
                    mean += e[j];
                    kv += 1;
                }
            }
            mean /= 1.0f64.max(kv as f64);
            let mut z = vec![0.0f32; k];
            for j in 0..k {
                if valid[j] > 0.5 {
                    e[j] -= mean;
                }
                z[j] = logits[sample * k + j] + e[j] as f32;
            }
            masked_softmax(&z, valid, k, &mut q);
            let qtype = y[2 * k].round() as i32;
            r[gg * n + sample] = proper_reward(
                &q,
                y,
                valid,
                k,
                qtype,
                spec.w_sph,
                spec.w_rps,
                spec.log_floor,
            );
        }
    }

    // Advantage: minus the per-item group mean, then divided by the unbiased
    // std over all G*N (torch `.std()`).
    let mut adv = vec![0.0f64; r.len()];
    for sample in 0..n {
        if !real[sample] {
            continue;
        }
        let mut m = 0.0f64;
        for gg in 0..g {
            m += r[gg * n + sample];
        }
        m /= g as f64;
        for gg in 0..g {
            adv[gg * n + sample] = r[gg * n + sample] - m;
        }
    }
    let n_adv = (g * n_real) as f64;
    let mut am = 0.0f64;
    for sample in 0..n {
        if real[sample] {
            for gg in 0..g {
                am += adv[gg * n + sample];
            }
        }
    }
    am /= 1.0f64.max(n_adv);
    let mut var = 0.0f64;
    for sample in 0..n {
        if real[sample] {
            for gg in 0..g {
                let d = adv[gg * n + sample] - am;
                var += d * d;
            }
        }
    }
    let sd = (var / 1.0f64.max(n_adv - 1.0)).sqrt();
    for a in adv.iter_mut() {
        *a /= sd + 1.0e-6;
    }

    // loss_rl = -mean(adv * logp), logp = -sum eps^2 / (2 sigma^2);
    // d logp / dl = eps / sigma^2.
    let mut rl = 0.0f64;
    let inv_s2 = 1.0 / (sigma as f64 * sigma as f64);
    for gg in 0..g {
        for sample in 0..n {
            let a = adv[gg * n + sample];
            let e = &eps[(gg * n + sample) * k..(gg * n + sample) * k + k];
            let mut e2 = 0.0f64;
            for j in 0..k {
                e2 += e[j] * e[j];
            }
            rl += a * e2 * 0.5 * inv_s2;
            for j in 0..k {
                delta[sample * k + j] += (a * e[j] * inv_s2 / g as f64) as f32;
            }
        }
    }
    loss += rl / (g as f64 * 1.0f64.max(n_real as f64));
    Ok(LayaLossOutput {
        loss: loss as f32,
        probs,
        delta,
    })
}

/// The layer's forward output: masked softmax over all options, i.e. a plain
/// softmax (C++ eval mode fills the truth with "all valid" before computing).
pub fn laya_probs(logits: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut probs = vec![0.0f32; n * k];
    let mut p = vec![0.0f64; k];
    let valid = vec![1.0f32; k];
    for sample in 0..n {
        masked_softmax(&logits[sample * k..(sample + 1) * k], &valid, k, &mut p);
        for j in 0..k {
            probs[sample * k + j] = p[j] as f32;
        }
    }
    probs
}

/// C++ `laya_td_lambda_targets`: `truth` has stride `2K+3`
/// (`[target K | valid K | qtype | ep_group | ep_step]`); the result has stride
/// `2K+1`.  With `lam >= 0`, the targets of episode samples (`ep_group >= 0`)
/// become TD(lambda) returns `G_t = (1-lam) p_true(t+1) + lam G_{t+1}`, with
/// `p_true` the detached softmax of the current logits.
pub fn td_lambda_targets(
    logits: &[f32],
    truth: &[f32],
    n: usize,
    k: usize,
    lam: f32,
) -> Result<Vec<f32>, String> {
    if k == 0 || n == 0 {
        return Err("laya_loss: batch and max_options must be > 0".to_owned());
    }
    let in_stride = 2 * k + 3;
    let out_stride = 2 * k + 1;
    if truth.len() != n * in_stride {
        return Err(format!(
            "laya_loss: truth has {} values, expected N*(2K+3)={}",
            truth.len(),
            n * in_stride
        ));
    }
    let mut out = vec![0.0f32; n * out_stride];
    let mut p_true = vec![0.0f64; n];
    // ep_group -> (ep_step, sample)
    let mut groups: std::collections::BTreeMap<i64, Vec<(i64, usize)>> =
        std::collections::BTreeMap::new();
    let mut p = vec![0.0f64; k];
    for sample in 0..n {
        let y = &truth[sample * in_stride..sample * in_stride + in_stride];
        out[sample * out_stride..sample * out_stride + out_stride]
            .copy_from_slice(&y[..out_stride]);
        let group = y[2 * k + 1].round() as i64;
        if group >= 0 && lam >= 0.0 {
            masked_softmax(
                &logits[sample * k..(sample + 1) * k],
                &y[k..2 * k],
                k,
                &mut p,
            );
            p_true[sample] = p[1.min(k - 1)];
            groups
                .entry(group)
                .or_default()
                .push((y[2 * k + 2].round() as i64, sample));
        }
    }
    for idx in groups.values_mut() {
        idx.sort();
        let mut g = truth[idx[idx.len() - 1].1 * in_stride + 1] as f64;
        for j in (0..idx.len()).rev() {
            if j + 1 < idx.len() {
                g = (1.0 - lam as f64) * p_true[idx[j + 1].1] + lam as f64 * g;
            }
            let o = &mut out[idx[j].1 * out_stride..idx[j].1 * out_stride + out_stride];
            o[0] = (1.0 - g) as f32;
            o[1] = g as f32;
        }
    }
    Ok(out)
}

/// Deterministic standard-normal draws for RLCD.  C++ uses
/// `std::mt19937_64` + `std::normal_distribution` (libstdc++ specific), which is
/// not reproduced bit-for-bit; this is a documented deterministic
/// SplitMix64 + Box-Muller substitute so a run is reproducible in Rust.
pub fn gaussian_normals(seed: u64, iteration: usize, count: usize) -> Vec<f32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(iteration as u64 + 1);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let u1 = ((next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = ((next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < count {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(k: usize, rlcd: bool) -> LayaLossSpec {
        LayaLossSpec {
            max_options: k,
            rlcd,
            group: 4,
            sigma_start: 0.4,
            sigma_end: 0.1,
            w_sph: 0.75,
            w_rps: 1.0,
            w_ce: 1.0,
            log_floor: -9.21,
            seed: 1234,
            td_lambda: -1.0,
        }
    }

    #[test]
    fn ce_matches_the_hand_computation() {
        let out = laya_loss_compute(
            &[0.0, 0.0],
            &[1.0, 0.0, 1.0, 1.0, 0.0],
            1,
            2,
            &spec(2, false),
            0.0,
            None,
        )
        .unwrap();
        let ln2 = 2.0f32.ln();
        assert!((out.loss - ln2).abs() < 1e-6, "{}", out.loss);
        assert!((out.probs[0] - 0.5).abs() < 1e-6);
        assert!((out.delta[0] - 0.5).abs() < 1e-6, "{:?}", out.delta);
        assert!((out.delta[1] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn invalid_slots_are_masked() {
        // Only option 0 valid, target 0 -> loss 0, delta 0.
        let out = laya_loss_compute(
            &[0.0, 5.0],
            &[1.0, 0.0, 1.0, 0.0, 0.0],
            1,
            2,
            &spec(2, false),
            0.0,
            None,
        )
        .unwrap();
        assert_eq!(out.probs, vec![1.0, 0.0]);
        assert_eq!(out.delta, vec![0.0, 0.0]);
        assert!(out.loss.abs() < 1e-9);
    }

    #[test]
    fn padding_samples_are_left_out_of_the_mean() {
        // Two samples: the second has no valid option, so only sample 0 counts.
        let truth = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let out = laya_loss_compute(
            &[0.0, 0.0, 3.0, 3.0],
            &truth,
            2,
            2,
            &spec(2, false),
            0.0,
            None,
        )
        .unwrap();
        let ln2 = 2.0f32.ln();
        assert!((out.loss - ln2).abs() < 1e-6, "{}", out.loss);
        assert_eq!(out.delta[2], 0.0);
        assert_eq!(out.delta[3], 0.0);
    }

    #[test]
    fn ce_delta_is_the_negative_loss_gradient() {
        // Central difference of the reported CE loss vs the delta.
        let logits = [0.3f32, -0.7, 1.1];
        let truth = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let base = laya_loss_compute(&logits, &truth, 1, 3, &spec(3, false), 0.0, None).unwrap();
        let eps = 1e-3f32;
        for i in 0..3 {
            let mut hi = logits;
            hi[i] += eps;
            let mut lo = logits;
            lo[i] -= eps;
            let l_hi = laya_loss_compute(&hi, &truth, 1, 3, &spec(3, false), 0.0, None)
                .unwrap()
                .loss;
            let l_lo = laya_loss_compute(&lo, &truth, 1, 3, &spec(3, false), 0.0, None)
                .unwrap()
                .loss;
            let numeric = -((l_hi - l_lo) / (2.0 * eps));
            assert!(
                (base.delta[i] - numeric).abs() < 2e-3,
                "i={i}: delta {} numeric {numeric}",
                base.delta[i]
            );
        }
    }

    #[test]
    fn td_lambda_targets_follow_the_prefix_recursion() {
        // Two samples in episode group 0, steps 0 and 1; the last keeps the
        // outcome y[1]=1, the first becomes (1-lam)*p_true(next) + lam*outcome.
        let in_stride = 2 * 2 + 3;
        let mut truth = vec![0.0f32; 2 * in_stride];
        // sample 0: target [0,1], valid [1,1], qtype 0, group 0, step 0
        truth[0] = 0.0;
        truth[1] = 1.0;
        truth[2] = 1.0;
        truth[3] = 1.0;
        truth[4] = 0.0;
        truth[5] = 0.0;
        truth[6] = 0.0;
        // sample 1: outcome target [0,1], group 0, step 1
        truth[in_stride + 1] = 1.0;
        truth[in_stride + 2] = 1.0;
        truth[in_stride + 3] = 1.0;
        truth[in_stride + 5] = 0.0;
        truth[in_stride + 6] = 1.0;
        let out = td_lambda_targets(&[0.0, 0.0, 0.0, 0.0], &truth, 2, 2, 0.5).unwrap();
        let out_stride = 2 * 2 + 1;
        // sample 1 keeps the outcome G = truth[1] = 1 -> [0, 1].
        assert!((out[out_stride] - 0.0).abs() < 1e-6);
        assert!((out[out_stride + 1] - 1.0).abs() < 1e-6);
        // sample 0: p_true = softmax([0,0])[1] = 0.5; G = 0.5*0.5 + 0.5*1 = 0.75.
        assert!((out[0] - 0.25).abs() < 1e-6, "{}", out[0]);
        assert!((out[1] - 0.75).abs() < 1e-6, "{}", out[1]);
    }

    #[test]
    fn rlcd_matches_the_hand_formula_with_fixed_noise() {
        let k = 2usize;
        let n = 1usize;
        let g = 2usize;
        let mut s = spec(k, true);
        s.group = g;
        s.w_sph = 0.0; // simplify the reward to the log score
        s.w_rps = 0.0;
        let logits = [0.0f32, 0.0];
        // target [1,0], valid [1,1], qtype 0.
        let truth = [1.0, 0.0, 1.0, 1.0, 0.0];
        let znorm = [1.0f32, -1.0, -1.0, 1.0]; // [g, n, k]
        let sigma = 0.5f32;
        let out = laya_loss_compute(&logits, &truth, n, k, &s, sigma, Some(&znorm)).unwrap();

        // Recompute eps/gradient by hand for group 0.
        let eps0 = [0.5f64, -0.5]; // 1*0.5, -1*0.5, mean 0
        // q for z = logits+eps0 = [0.5,-0.5] -> softmax; reward = log q[target=0].
        let p0 = (0.5f64.exp()) / (0.5f64.exp() + (-0.5f64).exp());
        let reward0 = p0.ln();
        let eps1 = [-0.5f64, 0.5];
        let p1 = ((-0.5f64).exp()) / ((-0.5f64).exp() + (0.5f64).exp());
        let reward1 = p1.ln();
        // Advantages: group mean removed; both are +/- half the gap.
        let gap = reward0 - reward1;
        let a0 = gap / 2.0;
        let a1 = -gap / 2.0;
        let adv = [a0, a1];
        let am = (a0 + a1) / 2.0;
        let var = (a0 - am).powi(2) + (a1 - am).powi(2);
        let sd = (var / 1.0f64).sqrt();
        let adv = [a0 / (sd + 1e-6), a1 / (sd + 1e-6)];
        let inv_s2 = 1.0 / (sigma as f64 * sigma as f64);
        // Group 0 gradient: a*eps*inv_s2/G; group 1 mirrors with eps1.
        let want_delta0 =
            (adv[0] * eps0[0] * inv_s2 / g as f64 + adv[1] * eps1[0] * inv_s2 / g as f64) as f32;
        let want_delta1 =
            (adv[0] * eps0[1] * inv_s2 / g as f64 + adv[1] * eps1[1] * inv_s2 / g as f64) as f32;
        // The plus the CE part (w_ce=1): y - p with p = [0.5, 0.5].
        assert!(
            (out.delta[0] - (0.5 + want_delta0)).abs() < 2e-3,
            "{} vs {}",
            out.delta[0],
            0.5 + want_delta0
        );
        assert!(
            (out.delta[1] - (-0.5 + want_delta1)).abs() < 2e-3,
            "{} vs {}",
            out.delta[1],
            -0.5 + want_delta1
        );
    }

    #[test]
    fn gaussian_normals_are_deterministic_and_finite() {
        let a = gaussian_normals(1234, 7, 64);
        let b = gaussian_normals(1234, 7, 64);
        let c = gaussian_normals(1234, 8, 64);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 64);
        assert!(a.iter().all(|v| v.is_finite()));
        let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.5, "mean {mean}");
    }

    #[test]
    fn shape_errors_are_reported() {
        assert!(laya_loss_compute(&[0.0], &[0.0], 1, 1, &spec(1, false), 0.0, None).is_err());
        assert!(laya_loss_compute(&[0.0, 0.0], &[0.0], 1, 2, &spec(2, false), 0.0, None).is_err());
        // rlcd needs znorm.
        assert!(
            laya_loss_compute(
                &[0.0, 0.0],
                &[1.0, 0.0, 1.0, 1.0, 0.0],
                1,
                2,
                &spec(2, true),
                0.5,
                None
            )
            .is_err()
        );
    }
}

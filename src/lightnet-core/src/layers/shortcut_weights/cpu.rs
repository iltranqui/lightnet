//! Weighted `[shortcut]` (tasks/rust/layers.md Task 6).

use crate::model::ShortcutWeightsNorm;

/// C++ `eps` in the weight normalisation.
const EPS: f32 = 0.0001;

/// Per-slot normalised weights. `raw` is laid out `[input i][slot]` with
/// `layer_step` slots per input (1 for per_feature, C for per_channel).
pub(crate) fn effective_weights(
    raw: &[f32],
    norm: ShortcutWeightsNorm,
    layer_step: usize,
) -> Vec<f32> {
    let n1 = raw.len() / layer_step;
    let mut eff = raw.to_vec();
    for slot in 0..layer_step {
        let at = |i: usize| slot + i * layer_step;
        match norm {
            ShortcutWeightsNorm::None => {}
            ShortcutWeightsNorm::Relu => {
                let sum = EPS + (0..n1).map(|i| raw[at(i)].max(0.0)).sum::<f32>();
                for i in 0..n1 {
                    eff[at(i)] = raw[at(i)].max(0.0) / sum;
                }
            }
            ShortcutWeightsNorm::Softmax => {
                let max = (0..n1)
                    .map(|i| raw[at(i)])
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum = EPS + (0..n1).map(|i| (raw[at(i)] - max).exp()).sum::<f32>();
                for i in 0..n1 {
                    eff[at(i)] = (raw[at(i)] - max).exp() / sum;
                }
            }
        }
    }
    eff
}

/// `out[b,e] = eff[e/step]*input[b,e] + sum_i [e < len_i] eff[e/step + (i+1)*layer_step]*sources[i][b*len_i+e]`
/// (`len_i` = per-sample length of source `i`; a shorter source only covers the flat prefix,
/// like the C++ `src_i < add_outputs` test).
pub(crate) fn weighted_forward(
    input: &[f32],
    sources: &[&[f32]],
    batch: usize,
    per_sample: usize,
    raw: &[f32],
    norm: ShortcutWeightsNorm,
    layer_step: usize,
) -> Vec<f32> {
    let eff = effective_weights(raw, norm, layer_step);
    let step = per_sample / layer_step;
    let mut out = vec![0.0f32; batch * per_sample];
    for b in 0..batch {
        for e in 0..per_sample {
            let slot = e / step;
            let mut value = eff[slot] * input[b * per_sample + e];
            for (i, source) in sources.iter().enumerate() {
                let len = source.len() / batch;
                if e < len {
                    value += eff[slot + (i + 1) * layer_step] * source[b * len + e];
                }
            }
            out[b * per_sample + e] = value;
        }
    }
    out
}

pub(crate) struct WeightedGrad {
    pub input: Vec<f32>,
    pub sources: Vec<Vec<f32>>,
    pub weights: Vec<f32>,
}

/// Exact backward. `z[k]` = dL/d(effective weight k) = sum over the slot's elements of
/// `delta * x_i`. Chain through the normalisation:
/// - none:    dL/dw_k = z_k
/// - relu:    eff_i = r_i/S, r = relu(w), S = eps + sum r  ->  dL/dw_k = [w_k>0] (z_k/S - sum_i z_i r_i / S^2)
/// - softmax: eff_i = e_i/S                                 ->  dL/dw_k = eff_k (z_k - sum_i z_i eff_i)
/// `cpp_grad` reproduces C++ (`grad = 1`, i.e. dL/dw_k = z_k for every normalisation).
pub(crate) fn weighted_backward(
    delta: &[f32],
    input: &[f32],
    sources: &[&[f32]],
    batch: usize,
    per_sample: usize,
    raw: &[f32],
    norm: ShortcutWeightsNorm,
    layer_step: usize,
    cpp_grad: bool,
) -> WeightedGrad {
    let eff = effective_weights(raw, norm, layer_step);
    let step = per_sample / layer_step;
    let mut grad = WeightedGrad {
        input: vec![0.0; delta.len()],
        sources: sources.iter().map(|s| vec![0.0; s.len()]).collect(),
        weights: vec![0.0; raw.len()],
    };
    let mut z = vec![0.0f32; raw.len()];
    for b in 0..batch {
        for e in 0..per_sample {
            let slot = e / step;
            let d = delta[b * per_sample + e];
            grad.input[b * per_sample + e] = d * eff[slot];
            z[slot] += d * input[b * per_sample + e];
            for (i, source) in sources.iter().enumerate() {
                let len = source.len() / batch;
                if e < len {
                    let idx = slot + (i + 1) * layer_step;
                    grad.sources[i][b * len + e] = d * eff[idx];
                    z[idx] += d * source[b * len + e];
                }
            }
        }
    }
    grad.weights = normalized_weight_gradient(raw, norm, layer_step, &z, cpp_grad);
    grad
}

/// Apply the derivative of weight normalization to per-input/per-slot dot
/// products `z`. This is shared by CPU backward and the GPU reduction path.
pub(crate) fn normalized_weight_gradient(
    raw: &[f32],
    norm: ShortcutWeightsNorm,
    layer_step: usize,
    z: &[f32],
    cpp_grad: bool,
) -> Vec<f32> {
    assert!(layer_step > 0 && raw.len().is_multiple_of(layer_step));
    assert_eq!(raw.len(), z.len());
    let normalization = match norm {
        ShortcutWeightsNorm::None => lightnet_cuda::F32ShortcutWeightNormalization::None,
        ShortcutWeightsNorm::Relu => lightnet_cuda::F32ShortcutWeightNormalization::Relu,
        ShortcutWeightsNorm::Softmax => lightnet_cuda::F32ShortcutWeightNormalization::Softmax,
    };
    lightnet_cuda::host_ops::shortcut_weight_normalized_gradient(
        raw,
        z,
        layer_step,
        normalization,
        cpp_grad,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ShortcutWeightsNorm::{None as NoNorm, Relu, Softmax};

    #[test]
    fn per_feature_relu_matches_hand_computation() {
        // raw [2, 1], eps 1e-4: eff = [2/3.0001, 1/3.0001]; x = 3, y = 6 -> 12/3.0001
        let out = weighted_forward(&[3.0], &[&[6.0]], 1, 1, &[2.0, 1.0], Relu, 1);
        assert!((out[0] - 12.0 / 3.0001).abs() < 1e-5, "{out:?}");
        // softmax of equal weights is 1/(2+eps) each: out = (3 + 6) / 2.0001
        let out = weighted_forward(&[3.0], &[&[6.0]], 1, 1, &[0.0, 0.0], Softmax, 1);
        assert!((out[0] - 9.0 / 2.0001).abs() < 1e-5, "{out:?}");
        // no normalisation: plain weighted sum
        let out = weighted_forward(&[3.0], &[&[6.0]], 1, 1, &[2.0, 0.5], NoNorm, 1);
        assert_eq!(out, vec![9.0]);
    }

    #[test]
    fn mismatched_source_adds_only_its_prefix() {
        // per_sample 4, source has 2 elements: elements 2,3 only see the layer input
        let out = weighted_forward(
            &[1.0, 1.0, 1.0, 1.0],
            &[&[10.0, 10.0]],
            1,
            4,
            &[1.0, 1.0],
            NoNorm,
            1,
        );
        assert_eq!(out, vec![11.0, 11.0, 1.0, 1.0]);
    }

    #[test]
    fn shortcut_axis_layout_uses_nchw_channel_slots_per_batch() {
        let input = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ];
        let source = [
            101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0,
        ];
        // [input, source] x [channel 0, channel 1], with 3 spatial cells per channel.
        let raw = [2.0, 3.0, 5.0, 7.0];
        let output = weighted_forward(&input, &[&source], 2, 6, &raw, NoNorm, 2);
        assert_eq!(
            output,
            vec![
                507.0, 514.0, 521.0, 740.0, 750.0, 760.0, 570.0, 640.0, 710.0, 1100.0, 1200.0,
                1300.0,
            ]
        );
    }

    fn loss(
        input: &[f32],
        sources: &[&[f32]],
        raw: &[f32],
        norm: ShortcutWeightsNorm,
        seed: &[f32],
    ) -> f32 {
        // batch 2, per_sample 6, per_channel with 2 channels (layer_step 2, step 3)
        weighted_forward(input, sources, 2, 6, raw, norm, 2)
            .iter()
            .zip(seed)
            .map(|(o, s)| o * s)
            .sum()
    }

    #[test]
    fn exact_gradients_match_central_differences_for_every_normalisation() {
        let input: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).sin()).collect();
        let s0: Vec<f32> = (0..12).map(|i| (i as f32 * 0.71).cos()).collect(); // len 6/sample
        let s1: Vec<f32> = (0..8).map(|i| (i as f32 * 0.53).sin() + 0.3).collect(); // len 4/sample (prefix)
        let seed: Vec<f32> = (0..12).map(|i| (i as f32 * 0.91).cos()).collect();
        let raw = vec![0.5f32, -0.3, 0.8, 1.2, 0.4, 0.9]; // (n+1)=3 inputs x 2 slots
        let h = 1e-2f32;
        for norm in [NoNorm, Relu, Softmax] {
            let g = weighted_backward(&seed, &input, &[&s0, &s1], 2, 6, &raw, norm, 2, false);
            let check = |name: &str, analytic: f32, plus: f32, minus: f32| {
                let fd = (plus - minus) / (2.0 * h);
                assert!(
                    (fd - analytic).abs() <= 5e-3 + 5e-3 * fd.abs(),
                    "{norm:?} {name}: fd {fd}, analytic {analytic}"
                );
            };
            for k in 0..raw.len() {
                let (mut p, mut m) = (raw.clone(), raw.clone());
                p[k] += h;
                m[k] -= h;
                check(
                    &format!("w[{k}]"),
                    g.weights[k],
                    loss(&input, &[&s0, &s1], &p, norm, &seed),
                    loss(&input, &[&s0, &s1], &m, norm, &seed),
                );
            }
            for k in 0..input.len() {
                let (mut p, mut m) = (input.clone(), input.clone());
                p[k] += h;
                m[k] -= h;
                check(
                    &format!("x[{k}]"),
                    g.input[k],
                    loss(&p, &[&s0, &s1], &raw, norm, &seed),
                    loss(&m, &[&s0, &s1], &raw, norm, &seed),
                );
            }
            for k in 0..s1.len() {
                let (mut p, mut m) = (s1.clone(), s1.clone());
                p[k] += h;
                m[k] -= h;
                check(
                    &format!("s1[{k}]"),
                    g.sources[1][k],
                    loss(&input, &[&s0, &p], &raw, norm, &seed),
                    loss(&input, &[&s0, &m], &raw, norm, &seed),
                );
            }
        }
    }

    #[test]
    fn cpp_grad_mode_ignores_the_normalisation_derivative() {
        let raw = [2.0f32, 1.0];
        let g = weighted_backward(&[1.0], &[3.0], &[&[6.0]], 1, 1, &raw, Relu, 1, true);
        assert_eq!(g.weights, vec![3.0, 6.0]); // sum(delta * x_i) exactly, like blas.cpp (grad = 1)
    }
}

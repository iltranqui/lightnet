//! `[laya_scorer]`: the LAYA decision-model scorer (Phase 1b of
//! `tasks/rust/layers.md`, unblocked by Open Question 2), ported from
//! `Jev_laya:src-lib/laya_scorer_layer.cpp`.
//!
//! A frozen 3-stage MLP applied **independently to each option slot** (no
//! cross-slot mixing), turning each gathered hidden-state column into one
//! scalar logit:
//!
//! ```text
//! for each row (one of batch * max_options slots):
//!     n   = LayerNorm(x, ln_w, ln_b)                (WITH bias, eps = ln_eps)
//!     h   = gelu_erf(n @ fc1_w^T + fc1_b)            (Linear hidden -> hidden, WITH bias)
//!     out = h @ fc2_w^T + fc2_b                      (Linear hidden -> 1, WITH bias)
//! ```
//!
//! Two details that are easy to get wrong and are called out in the C++ header:
//!
//! * **GELU is the exact erf variant** (`torch.nn.GELU()`'s default, what
//!   `[laya_encoder]`'s GeGLU also uses), *not* darknet's built-in
//!   tanh-approximate `gelu`.  `activation_math::gelu` is the tanh one, so this
//!   module has its own [`gelu_erf`].
//! * The output is **raw, unmasked** logits: the `-1e4` invalid-option masking
//!   and the softmax/temperature stay CLI-side post-processing.
//!
//! Layout.  Input is channel-major `[batch, hidden, 1, max_options]` (the
//! `[laya_marker_gather]` output); the layer transposes it into `rows =
//! batch * max_options` independent vectors of length `hidden` and writes
//! `[batch, 1, 1, max_options]` (one channel, so the packed `[rows, 1]` result
//! is bit-identical to it).

use crate::model::LayaScorerSpec;

/// `LAYA_LAYERNORM_EPS`, the default for the LayerNorm epsilon.
pub const LAYA_LAYERNORM_EPS: f32 = 1e-5;

/// The exact-erf GELU (`torch.nn.GELU()`): `0.5 x (1 + erf(x / sqrt(2)))`.
pub fn gelu_erf(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * std::f32::consts::FRAC_1_SQRT_2).erf())
}

/// C++ `sc_layernorm`: row-wise LayerNorm **with bias**, population variance.
///
/// `rows` vectors of length `cols`; `out` may not alias `x`.
pub fn layer_norm_with_bias(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    eps: f32,
) {
    let inv_c = 1.0 / cols as f32;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut mean = 0.0f32;
        for v in row {
            mean += *v;
        }
        mean *= inv_c;
        let mut var = 0.0f32;
        for v in row {
            let d = *v - mean;
            var += d * d;
        }
        var *= inv_c;
        let inv_std = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter().enumerate() {
            out[r * cols + i] = gamma[i] * ((*v - mean) * inv_std) + beta[i];
        }
    }
}

/// Which of the six tensors take weight decay, in C++ blob order
/// (`kScLnW .. kScFc2B`).  The LayerNorm affine terms and the two biases do not.
pub const PARAM_DECAYS: [bool; 6] = [false, false, true, false, true, false];

/// C++ `laya_encoder_blob_float_count(hidden, n_layers, intermediate, vocab)`:
/// the number of floats `[laya_encoder]` occupies at the front of the shared
/// `.weights` file.
pub fn encoder_blob_float_count(
    hidden: usize,
    n_layers: usize,
    intermediate: usize,
    vocab: usize,
) -> usize {
    let (h, i, nl) = (hidden, intermediate, n_layers);
    let mut total = h * vocab; // tok_embeddings
    total += h; // embeddings.norm
    for layer in 0..nl {
        if layer > 0 {
            total += h; // attn_norm (absent for layer 0)
        }
        total += 3 * h * h; // Wqkv
        total += h * h; // Wo (attn)
        total += h; // mlp_norm
        total += 2 * i * h; // Wi
        total += h * i; // Wo (mlp)
    }
    total += h; // final_norm
    total
}

/// C++ `laya_head_blob_float_count(hidden, n_layers, ffn_dim)`.  The folded-in
/// `type_emb.weight` (`3 * hidden`) is already counted here, so the scorer's
/// offset must **not** add it again.
pub fn head_blob_float_count(hidden: usize, n_layers: usize, ffn_dim: usize) -> usize {
    let (h, f) = (hidden, ffn_dim);
    let per_layer = 2 * h // norm1 w,b
        + 3 * h * h + 3 * h // in_proj w,b
        + h * h + h // out_proj w,b
        + 2 * h // norm2 w,b
        + f * h + f // linear1 w,b
        + h * f + h; // linear2 w,b
    n_layers * per_layer + 3 * h
}

/// Where `[laya_scorer]`'s blob starts inside the shared `.weights` file:
/// `20` header bytes (3 ints + one u64) plus the encoder and head blobs.
/// C++ computes the same value as `sc_blob_offset`.
pub fn scorer_blob_offset_bytes(
    enc_hidden: usize,
    enc_layers: usize,
    enc_intermediate: usize,
    enc_vocab: usize,
    head_hidden: usize,
    head_layers: usize,
    head_ffn_dim: usize,
    f16: bool,
) -> usize {
    let floats = encoder_blob_float_count(enc_hidden, enc_layers, enc_intermediate, enc_vocab)
        + head_blob_float_count(head_hidden, head_layers, head_ffn_dim);
    20 + floats * if f16 { 2 } else { 4 }
}

/// The six frozen tensors, in the C++ blob order (`kScLnW .. kScFc2B`).
#[derive(Clone, Debug, PartialEq)]
pub struct LayaScorerWeights {
    /// `scorer.0.weight` — LayerNorm gamma.
    pub ln_w: Vec<f32>,
    /// `scorer.0.bias` — LayerNorm beta.
    pub ln_b: Vec<f32>,
    /// `scorer.1.weight` — `hidden x hidden`, row-major.
    pub fc1_w: Vec<f32>,
    /// `scorer.1.bias`.
    pub fc1_b: Vec<f32>,
    /// `scorer.3.weight` — `1 x hidden`, row-major.
    pub fc2_w: Vec<f32>,
    /// `scorer.3.bias`.
    pub fc2_b: f32,
}

/// C++ `laya_scorer_blob_float_count`: `2H + H*H + 2H + 1`.
pub fn blob_float_count(hidden: usize) -> usize {
    2 * hidden + hidden * hidden + 2 * hidden + 1
}

impl LayaScorerWeights {
    /// All-zero weights of the right sizes (C++ `make_laya_scorer_layer`
    /// zero-initialises and expects `load_laya_scorer_weights` to fill them).
    pub fn zeroed(hidden: usize) -> Self {
        Self {
            ln_w: vec![0.0; hidden],
            ln_b: vec![0.0; hidden],
            fc1_w: vec![0.0; hidden * hidden],
            fc1_b: vec![0.0; hidden],
            fc2_w: vec![0.0; hidden],
            fc2_b: 0.0,
        }
    }

    /// Per-tensor sizes in blob order.
    pub fn sizes(&self) -> [usize; 6] {
        [
            self.ln_w.len(),
            self.ln_b.len(),
            self.fc1_w.len(),
            self.fc1_b.len(),
            self.fc2_w.len(),
            1,
        ]
    }

    /// The blob flattened in C++ read order, for `.weights` round trips.
    pub fn to_blob(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.sizes().iter().sum());
        out.extend_from_slice(&self.ln_w);
        out.extend_from_slice(&self.ln_b);
        out.extend_from_slice(&self.fc1_w);
        out.extend_from_slice(&self.fc1_b);
        out.extend_from_slice(&self.fc2_w);
        out.push(self.fc2_b);
        out
    }

    /// Inverse of [`Self::to_blob`]; the blob must be exactly
    /// [`blob_float_count`] long.
    pub fn from_blob(blob: &[f32]) -> Result<Self, String> {
        // The smallest blob is hidden=1: 1 + 1 + 1 + 1 + 1 + 1 = 6.
        if blob.len() < 6 {
            return Err(format!("laya_scorer blob is too short: {}", blob.len()));
        }
        // The blob is `H^2 + 4H + 1`, so `H = (sqrt(4 * len + 12) - 4) / 2`.
        let len = blob.len() as f64;
        let hidden = (((4.0 * len + 12.0).sqrt() - 4.0) / 2.0).round() as usize;
        if blob_float_count(hidden) != blob.len() {
            return Err(format!(
                "laya_scorer blob of {} floats does not match any hidden size (closest {} for hidden={hidden})",
                blob.len(),
                blob_float_count(hidden)
            ));
        }
        let mut at = 0usize;
        let mut take = |n: usize| {
            let slice = blob[at..at + n].to_vec();
            at += n;
            slice
        };
        let ln_w = take(hidden);
        let ln_b = take(hidden);
        let fc1_w = take(hidden * hidden);
        let fc1_b = take(hidden);
        let fc2_w = take(hidden);
        let fc2_b = blob[at];
        Ok(Self {
            ln_w,
            ln_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        })
    }
}

/// The layer's forward.  `input` is channel-major `[batch, hidden, 1, k]`,
/// the result is `[batch, 1, 1, k]` (which is `[batch * k]` floats).
pub fn scorer_forward(
    input: &[f32],
    weights: &LayaScorerWeights,
    spec: &LayaScorerSpec,
    batch: usize,
) -> Result<Vec<f32>, String> {
    let (h, k) = (spec.hidden, spec.max_options);
    if input.len() != batch * h * k {
        return Err(format!(
            "laya_scorer input has {} entries, expected batch*hidden*max_options = {}",
            input.len(),
            batch * h * k
        ));
    }
    if weights.ln_w.len() != h
        || weights.ln_b.len() != h
        || weights.fc1_w.len() != h * h
        || weights.fc1_b.len() != h
        || weights.fc2_w.len() != h
    {
        return Err(format!("laya_scorer weights do not match hidden={h}"));
    }
    let rows = batch * k;
    let eps = if spec.ln_eps > 0.0 {
        spec.ln_eps
    } else {
        LAYA_LAYERNORM_EPS
    };
    let mut output = vec![0.0f32; rows];

    // 1. unpack channel-major [B,H,1,K] -> row-major [B*K, H]
    let mut tokens = vec![0.0f32; rows * h];
    for b in 0..batch {
        for j in 0..k {
            let dst = (b * k + j) * h;
            for c in 0..h {
                tokens[dst + c] = input[(b * h + c) * k + j];
            }
        }
    }
    // 2. LayerNorm with bias
    let mut xhat = vec![0.0f32; rows * h];
    layer_norm_with_bias(
        &tokens,
        &weights.ln_w,
        &weights.ln_b,
        &mut xhat,
        rows,
        h,
        eps,
    );

    // 3. Linear(H -> H) then the exact-erf GELU
    let mut hidden_buf = vec![0.0f32; rows * h];
    for r in 0..rows {
        let x = &xhat[r * h..(r + 1) * h];
        for m in 0..h {
            let mut acc = 0.0f32;
            for i in 0..h {
                acc += x[i] * weights.fc1_w[m * h + i];
            }
            hidden_buf[r * h + m] = gelu_erf(acc + weights.fc1_b[m]);
        }
    }
    // 4. Linear(H -> 1) plus the scalar bias
    for r in 0..rows {
        let mut acc = 0.0f32;
        for i in 0..h {
            acc += hidden_buf[r * h + i] * weights.fc2_w[i];
        }
        output[r] = acc + weights.fc2_b;
    }
    Ok(output)
}

/// The exact derivative of [`scorer_forward`] with respect to the input, given
/// the seed `dL/doutput` (`batch * max_options` entries).
///
/// The forward intermediates are recomputed from `input` rather than carried
/// (C++ keeps `sc_tokens` / `sc_xhat` / `sc_hidden_buf` alive for the same
/// purpose).  The chain is
///
/// ```text
/// out   = hb @ fc2_w^T + b2                -> dL/dhb   = seed * fc2_w
/// hb    = gelu_erf(z)                       -> dL/dz    = dL/dhb * gelu_erf'(z)
/// z     = xhat @ fc1_w^T + b1              -> dL/dxhat = dL/dz @ fc1_w
/// xhat  = gamma * xhat_raw + beta           -> LayerNorm backward
/// ```
///
/// with the standard LayerNorm reduction
/// `dx = g - mean(g) - mean(xhat_raw) * g`, where
/// `g = dL/dxhat * gamma * inv_std`.  The result is scattered back into the
/// channel-major `[batch, hidden, 1, max_options]` input layout.
pub fn scorer_backward_input(
    input: &[f32],
    seed: &[f32],
    weights: &LayaScorerWeights,
    spec: &LayaScorerSpec,
    batch: usize,
) -> Result<Vec<f32>, String> {
    let (h, k) = (spec.hidden, spec.max_options);
    let rows = batch * k;
    if input.len() != rows * h {
        return Err(format!(
            "laya_scorer input has {} entries, expected {}",
            input.len(),
            rows * h
        ));
    }
    if seed.len() != rows {
        return Err(format!(
            "laya_scorer seed has {} entries, expected {rows}",
            seed.len()
        ));
    }
    let eps = if spec.ln_eps > 0.0 {
        spec.ln_eps
    } else {
        LAYA_LAYERNORM_EPS
    };
    let inv_h = 1.0 / h as f32;

    // Forward intermediates.
    let mut tokens = vec![0.0f32; rows * h];
    for b in 0..batch {
        for j in 0..k {
            let dst = (b * k + j) * h;
            for c in 0..h {
                tokens[dst + c] = input[(b * h + c) * k + j];
            }
        }
    }
    let (mut mean, mut inv_std, mut xhat_raw) = (
        vec![0.0f32; rows],
        vec![0.0f32; rows],
        vec![0.0f32; rows * h],
    );
    let mut xhat = vec![0.0f32; rows * h];
    for r in 0..rows {
        let row = &tokens[r * h..(r + 1) * h];
        let m = row.iter().sum::<f32>() * inv_h;
        let var = row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() * inv_h;
        mean[r] = m;
        inv_std[r] = 1.0 / (var + eps).sqrt();
        for (i, v) in row.iter().enumerate() {
            let raw = (*v - m) * inv_std[r];
            xhat_raw[r * h + i] = raw;
            xhat[r * h + i] = weights.ln_w[i] * raw + weights.ln_b[i];
        }
    }
    let mut z = vec![0.0f32; rows * h];
    for r in 0..rows {
        for m in 0..h {
            let mut acc = 0.0f32;
            for i in 0..h {
                acc += xhat[r * h + i] * weights.fc1_w[m * h + i];
            }
            z[r * h + m] = acc + weights.fc1_b[m];
        }
    }

    // Backward.
    let mut grad_hidden = vec![0.0f32; rows * h];
    for r in 0..rows {
        for i in 0..h {
            grad_hidden[r * h + i] = seed[r] * weights.fc2_w[i];
        }
    }
    let mut grad_z = vec![0.0f32; rows * h];
    for r in 0..rows {
        for m in 0..h {
            grad_z[r * h + m] = grad_hidden[r * h + m] * gelu_erf_grad(z[r * h + m]);
        }
    }
    let mut grad_xhat = vec![0.0f32; rows * h];
    for r in 0..rows {
        for i in 0..h {
            let mut acc = 0.0f32;
            for m in 0..h {
                acc += grad_z[r * h + m] * weights.fc1_w[m * h + i];
            }
            grad_xhat[r * h + i] = acc;
        }
    }
    // LayerNorm backward, one row at a time.
    let mut grad_input = vec![0.0f32; batch * h * k];
    for r in 0..rows {
        let mut g = vec![0.0f32; h];
        let mut mean_g = 0.0f32;
        let mut mean_gx = 0.0f32;
        for i in 0..h {
            g[i] = grad_xhat[r * h + i] * weights.ln_w[i] * inv_std[r];
            mean_g += g[i];
            // The second LayerNorm reduction is over `g * xhat`, not over
            // `xhat` alone.
            mean_gx += g[i] * xhat_raw[r * h + i];
        }
        mean_g *= inv_h;
        mean_gx *= inv_h;
        let b = r / k;
        let j = r % k;
        for i in 0..h {
            let dx = g[i] - mean_g - xhat_raw[r * h + i] * mean_gx;
            grad_input[(b * h + i) * k + j] += dx;
        }
    }
    Ok(grad_input)
}

/// The elementwise seed of [`gelu_erf`].
pub fn gelu_erf_grad(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + (x * std::f32::consts::FRAC_1_SQRT_2).erf());
    let pdf = (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt();
    cdf + x * pdf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(hidden: usize, max_options: usize) -> LayaScorerSpec {
        LayaScorerSpec {
            hidden,
            max_options,
            ln_eps: LAYA_LAYERNORM_EPS,
        }
    }

    #[test]
    fn gelu_erf_matches_the_closed_form_at_known_points() {
        assert!(gelu_erf(0.0).abs() < 1e-7, "gelu(0) = 0");
        // erf-GELU: 0.5 x (1 + erf(x / sqrt(2))). At x = -1 and x = 1 the erf
        // factor is 0.6826894921, so the values are -0.15865525 / 0.84134475.
        // (The tanh approximation gives -0.15880801 / 0.84119199, so these two
        // constants also pin that the two GELU variants are not confused.)
        assert!(
            (gelu_erf(-1.0) + 0.158_655_25).abs() < 1e-6,
            "{}",
            gelu_erf(-1.0)
        );
        assert!(
            (gelu_erf(1.0) - 0.841_344_75).abs() < 1e-6,
            "{}",
            gelu_erf(1.0)
        );
        assert!(
            (gelu_erf(2.0) - 1.954_499_7).abs() < 1e-5,
            "{}",
            gelu_erf(2.0)
        );
        // The tanh approximation darknet's own gelu uses differs measurably.
        let tanh = crate::activation_math::gelu(1.0);
        assert!(
            (tanh - gelu_erf(1.0)).abs() > 1e-4,
            "erf and tanh GELU must differ: {tanh} vs {}",
            gelu_erf(1.0)
        );
    }

    #[test]
    fn layer_norm_zero_means_unit_scale() {
        // A constant row normalises to (0 - mean) / sqrt(0 + eps) = 0, so the
        // output is just the bias.
        let x = [2.0f32, 2.0, 2.0, 2.0];
        let gamma = [1.0f32; 4];
        let beta = [0.5f32, -0.5, 0.25, 0.0];
        let mut out = [0.0f32; 4];
        layer_norm_with_bias(&x, &gamma, &beta, &mut out, 1, 4, 1e-5);
        assert_eq!(out, beta);
        // Two rows are normalised independently.
        let x = [1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let gamma = [1.0f32; 4];
        let beta = [0.0f32; 4];
        let mut out = [0.0f32; 8];
        layer_norm_with_bias(&x, &gamma, &beta, &mut out, 2, 4, 1e-5);
        let expect = |v: f32, mean: f32, var: f32| (v - mean) / (var + 1e-5).sqrt();
        assert!((out[0] - expect(1.0, 2.5, 1.25)).abs() < 1e-6, "{}", out[0]);
        assert!((out[3] - expect(4.0, 2.5, 1.25)).abs() < 1e-6, "{}", out[3]);
        assert!(
            (out[4] - expect(10.0, 25.0, 125.0)).abs() < 1e-6,
            "{}",
            out[4]
        );
        assert!(
            (out[7] - expect(40.0, 25.0, 125.0)).abs() < 1e-6,
            "{}",
            out[7]
        );
    }

    #[test]
    fn forward_is_per_slot_and_matches_a_hand_computation() {
        let h = 3;
        let k = 2;
        let s = spec(h, k);
        let w = LayaScorerWeights {
            ln_w: vec![1.0; h],
            ln_b: vec![0.0; h],
            // fc1: identity, so `hidden_buf = gelu_erf(xhat)`
            fc1_w: {
                let mut m = vec![0.0f32; h * h];
                for i in 0..h {
                    m[i * h + i] = 1.0;
                }
                m
            },
            fc1_b: vec![0.0; h],
            // fc2: sums the three features
            fc2_w: vec![1.0, 1.0, 1.0],
            fc2_b: 0.5,
        };
        // batch 1: two slots, each a constant row (so LayerNorm outputs 0).
        let input = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
        let got = scorer_forward(&input, &w, &s, 1).unwrap();
        // Every slot is constant -> xhat = 0 -> gelu(0) = 0 -> out = fc2_b.
        assert_eq!(got, vec![0.5, 0.5]);

        // Now a non-degenerate slot: slot 0 varies, slot 1 stays constant, which
        // shows there is no cross-slot mixing.
        let mut input = vec![0.0f32; 6];
        // channel-major [1, 3, 1, 2]: slot 0 is input[0], input[2], input[4].
        input[0] = 1.0;
        input[2] = 0.0;
        input[4] = -1.0;
        let got = scorer_forward(&input, &w, &s, 1).unwrap();
        // slot 0: mean 0, var 2/3, xhat = [a, 0, -a] with a = 1/sqrt(2/3).
        let a = 1.0f32 / (2.0f32 / 3.0 + 1e-5).sqrt();
        let expect0 = gelu_erf(a) + gelu_erf(0.0) + gelu_erf(-a) + 0.5;
        assert!((got[0] - expect0).abs() < 1e-5, "{} vs {expect0}", got[0]);
        assert!(
            (got[1] - 0.5).abs() < 1e-6,
            "slot 1 is constant, got {}",
            got[1]
        );
    }

    #[test]
    fn forward_rejects_mismatched_inputs_and_weights() {
        let s = spec(3, 2);
        let w = LayaScorerWeights::zeroed(3);
        let err = scorer_forward(&[0.0; 5], &w, &s, 1).unwrap_err();
        assert!(err.contains("expected batch*hidden*max_options"), "{err}");
        let bad = LayaScorerWeights::zeroed(4);
        let err = scorer_forward(&[0.0; 6], &bad, &s, 1).unwrap_err();
        assert!(err.contains("do not match hidden"), "{err}");
    }

    #[test]
    fn the_backward_input_is_the_exact_derivative() {
        // Central finite differences of `L = sum(seed * scorer_forward(x))` must
        // reproduce `scorer_backward_input`, for several seeds and shapes.
        for (hidden, k, batch) in [(3usize, 2usize, 2usize), (5, 3, 1), (4, 1, 3)] {
            let s = spec(hidden, k);
            let mut w = LayaScorerWeights::zeroed(hidden);
            // Non-trivial weights so every path is exercised: gamma != 1,
            // beta != 0, dense fc1/fc2 and non-zero biases.
            for (i, v) in w.ln_w.iter_mut().enumerate() {
                *v = 0.7 + 0.1 * (i % 5) as f32;
            }
            for (i, v) in w.ln_b.iter_mut().enumerate() {
                *v = -0.3 + 0.05 * (i % 4) as f32;
            }
            for (i, v) in w.fc1_w.iter_mut().enumerate() {
                *v = ((i * 37 % 23) as f32 - 11.0) / 17.0;
            }
            for (i, v) in w.fc1_b.iter_mut().enumerate() {
                *v = (i % 3) as f32 * 0.25 - 0.2;
            }
            for (i, v) in w.fc2_w.iter_mut().enumerate() {
                *v = ((i * 13 % 11) as f32 - 5.0) / 9.0;
            }
            w.fc2_b = 0.45;
            let rows = batch * k;
            let input: Vec<f32> = (0..rows * hidden)
                .map(|i| ((i * 19 % 17) as f32 - 8.0) / 5.0)
                .collect();
            let seed: Vec<f32> = (0..rows).map(|i| (i as f32 * 0.37 - 0.5).sin()).collect();
            let analytic = scorer_backward_input(&input, &seed, &w, &s, batch).unwrap();
            let eps = 1e-3f32;
            for i in 0..input.len() {
                let mut hi = input.clone();
                hi[i] += eps;
                let mut lo = input.clone();
                lo[i] -= eps;
                let l_hi = scorer_forward(&hi, &w, &s, batch).unwrap();
                let l_lo = scorer_forward(&lo, &w, &s, batch).unwrap();
                let numeric: f32 = (0..rows)
                    .map(|r| seed[r] * (l_hi[r] - l_lo[r]))
                    .sum::<f32>()
                    / (2.0 * eps);
                let want = analytic[i];
                let slack = 2e-2 * want.abs().max(1e-2);
                assert!(
                    (numeric - want).abs() <= slack,
                    "hidden={hidden} k={k} batch={batch} i={i}: numeric {numeric} analytic {want}"
                );
            }
        }
    }

    #[test]
    fn the_backward_rejects_mismatched_inputs_and_seeds() {
        let s = spec(3, 2);
        let w = LayaScorerWeights::zeroed(3);
        assert!(
            scorer_backward_input(&[0.0; 5], &[0.0; 2], &w, &s, 1)
                .unwrap_err()
                .contains("expected 6")
        );
        assert!(
            scorer_backward_input(&[0.0; 6], &[0.0; 1], &w, &s, 1)
                .unwrap_err()
                .contains("seed has 1 entries")
        );
    }

    #[test]
    fn the_blob_round_trips_and_recovers_the_hidden_size() {
        for hidden in [1usize, 4, 17, 64] {
            let w = LayaScorerWeights {
                ln_w: (0..hidden).map(|i| i as f32).collect(),
                ln_b: (0..hidden).map(|i| -(i as f32)).collect(),
                fc1_w: (0..hidden * hidden).map(|i| i as f32 * 0.5).collect(),
                fc1_b: (0..hidden).map(|i| i as f32 + 1.0).collect(),
                fc2_w: (0..hidden).map(|i| i as f32 * 0.25).collect(),
                fc2_b: 7.5,
            };
            let blob = w.to_blob();
            assert_eq!(blob.len(), blob_float_count(hidden), "hidden={hidden}");
            assert_eq!(
                LayaScorerWeights::from_blob(&blob).unwrap(),
                w,
                "hidden={hidden}"
            );
        }
        // A blob of the wrong length is rejected rather than mis-sliced.
        let err = LayaScorerWeights::from_blob(&[0.0; 7]).unwrap_err();
        assert!(err.contains("does not match any hidden size"), "{err}");
    }

    #[test]
    fn blob_float_count_matches_the_cpp_arithmetic() {
        for hidden in [1usize, 8, 1024] {
            assert_eq!(
                blob_float_count(hidden),
                2 * hidden + hidden * hidden + 2 * hidden + 1
            );
        }
    }
}

#[cfg(test)]
mod real_weights {
    //! Loads the scorer blob out of the *real* `test_data/laya/laya_english.weights`
    //! using C++'s own offset arithmetic.  This is the check that the blob order
    //! and the offset are right: a wrong order or a missing `type_emb` would
    //! produce LayerNorm gamma values that are not near 1.

    use super::*;
    use std::path::Path;

    fn weights_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/laya/laya_english.weights")
    }

    /// Little-endian f32 read at a byte offset, mirroring `laya_read_floats`.
    fn read_f32s(bytes: &[u8], offset: usize, count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                let at = offset + 4 * i;
                f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
            })
            .collect()
    }

    #[test]
    fn the_default_encoder_and_head_keys_place_the_scorer_blob() {
        let offset = scorer_blob_offset_bytes(
            1024,  // encoder_hidden
            28,    // encoder_layers
            2624,  // encoder_intermediate
            50368, // encoder_vocab
            1024,  // head_hidden
            2,     // head_layers
            4096,  // head_ffn_dim
            false,
        );
        let floats = (offset - 20) / 4;
        assert_eq!(
            floats,
            encoder_blob_float_count(1024, 28, 2624, 50368) + head_blob_float_count(1024, 2, 4096)
        );
        println!("scorer blob offset: {offset} bytes ({floats} floats past the 20-byte header)");
    }

    #[test]
    fn the_real_weights_file_yields_a_plausible_scorer_blob() {
        let path = weights_path();
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: {} is not present", path.display());
            return;
        };
        let offset = scorer_blob_offset_bytes(1024, 28, 2624, 50368, 1024, 2, 4096, false);
        let hidden = 1024usize;
        let count = blob_float_count(hidden);
        assert!(
            offset + 4 * count <= bytes.len(),
            "blob needs {} bytes at offset {offset}, file has {}",
            4 * count,
            bytes.len()
        );
        let blob = read_f32s(&bytes, offset, count);
        let w = LayaScorerWeights::from_blob(&blob).expect("blob shape");

        // LayerNorm gamma of a trained model sits near 1 and beta near 0.  If the
        // offset or the blob order were wrong these would be arbitrary.
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        println!(
            "ln_w mean {:.4} rms {:.4} | ln_b mean {:.4} rms {:.4} | fc1_w rms {:.4} | fc2_b {:.4}",
            mean(&w.ln_w),
            rms(&w.ln_w),
            mean(&w.ln_b),
            rms(&w.ln_b),
            rms(&w.fc1_w),
            w.fc2_b
        );
        assert!(
            (mean(&w.ln_w) - 1.0).abs() < 0.25 && rms(&w.ln_w) < 1.6,
            "ln_w should look like a LayerNorm gamma (mean ~1), got mean {}",
            mean(&w.ln_w)
        );
        assert!(
            rms(&w.ln_b) < 0.6,
            "ln_b should look like a LayerNorm beta (near 0), rms {}",
            rms(&w.ln_b)
        );
        assert!(rms(&w.fc1_w) > 1e-3, "fc1_w must not be all zeros");
        assert!(
            w.fc2_b.abs() < 10.0,
            "fc2_b is a single logit bias, got {}",
            w.fc2_b
        );

        // And the forward must run on the real weights over a real-shaped input.
        let spec = LayaScorerSpec {
            hidden,
            max_options: 4,
            ln_eps: LAYA_LAYERNORM_EPS,
        };
        let batch = 2;
        // Deliberately asymmetric between the two images: a bug that ignored
        // the image index would silently produce identical logits for both.
        let input: Vec<f32> = (0..batch * hidden * 4)
            .map(|i| {
                let (image, rest) = (i / (hidden * 4), i % (hidden * 4));
                if image == 0 {
                    (rest % 17) as f32 * 0.01 - 0.08
                } else {
                    (rest % 5) as f32 * 0.2 + 0.5
                }
            })
            .collect();
        let out = scorer_forward(&input, &w, &spec, batch).expect("forward");
        assert_eq!(out.len(), batch * 4);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "logits must be finite: {out:?}"
        );
        assert!(
            (out[0] - out[4]).abs() > 1e-4,
            "the two images must not score identically: {out:?}"
        );
        println!("scorer logits (first 8): {:?}", &out[..8]);
    }
}

#[cfg(test)]
mod wiring {
    //! End-to-end checks that `[laya_scorer]` parses, builds parameters, runs
    //! on the CPU and the GPU, and loads its real blob.

    use super::*;
    use crate::{ConfigFile, ExecutableNetwork, LayerParams, Network, Tensor, TensorShape};

    fn cfg(hidden: usize, max_options: usize) -> String {
        format!(
            "[net]\nwidth={max_options}\nheight=1\nchannels={hidden}\nbatch=2\n\
             [convolutional]\nfilters={hidden}\nsize=1\nactivation=linear\n\
             [laya_scorer]\nhidden={hidden}\nmax_options={max_options}\n"
        )
    }

    #[test]
    fn the_layer_parses_builds_parameters_and_runs() {
        let network =
            Network::from_config(ConfigFile::parse_str("laya.cfg", &cfg(4, 3)).unwrap()).unwrap();
        // The shape contract: [batch, hidden, 1, max_options] -> [batch, 1, 1, max_options].
        assert_eq!(network.layers[1].input_shape.channels, 4);
        assert_eq!(network.layers[1].input_shape.width, 3);
        let out = network.layers[1].output_shape;
        assert_eq!(
            (out.channels, out.height, out.width, out.batch),
            (1, 1, 3, 2)
        );
        assert!(
            network.layers[1].weights().is_none(),
            "the blob is not the sequential stream"
        );

        let mut exec = ExecutableNetwork::new(network);
        let LayerParams::LayaScorer(_) = &exec.params[1] else {
            panic!("expected scorer params")
        };
        let input_shape = exec.network.net.input_shape();
        let data: Vec<f32> = (0..input_shape.batch * input_shape.element_count())
            .map(|i| (i % 9) as f32 * 0.1 - 0.4)
            .collect();
        let input = Tensor::from_data(input_shape, data.clone()).unwrap();
        let forward = exec.forward_training(&input).unwrap();
        assert_eq!(forward.layer_outputs[1].data.len(), 2 * 3);
        assert!(forward.layer_outputs[1].data.iter().all(|v| v.is_finite()));
        // With all-zero weights LayerNorm gives 0, GELU(0) = 0, so the logit is
        // just fc2_b (0.0).
        assert!(
            forward.layer_outputs[1].data.iter().all(|v| v.abs() < 1e-6),
            "{:?}",
            forward.layer_outputs[1].data
        );
    }

    #[test]
    fn the_backward_reaches_the_previous_layer() {
        let network =
            Network::from_config(ConfigFile::parse_str("laya.cfg", &cfg(4, 3)).unwrap()).unwrap();
        let mut exec = ExecutableNetwork::new(network);
        {
            // The convolution is zero-initialised, which would swallow the
            // gradient before it reaches the network input, so give it content.
            let LayerParams::Convolutional(conv) = &mut exec.params[0] else {
                panic!()
            };
            for (i, v) in conv.weights.iter_mut().enumerate() {
                *v = (i % 5) as f32 * 0.1 - 0.2;
            }
            conv.biases.iter_mut().for_each(|b| *b = 0.0);
            let LayerParams::LayaScorer(weights) = &mut exec.params[1] else {
                panic!()
            };
            // C++ zero-initialises the scorer and expects the loader to fill
            // it, so the *unloaded* LayerNorm gamma is 0 and the whole chain
            // collapses.  A trained-like gamma (near 1) and a dense fc1 are what
            // make the seed reach the input.
            weights.ln_w.fill(1.0);
            for (i, v) in weights.fc1_w.iter_mut().enumerate() {
                *v = ((i * 7 % 9) as f32 - 4.0) / 6.0;
            }
            for (i, v) in weights.fc2_w.iter_mut().enumerate() {
                *v = (i as f32 % 3.0) * 0.5 - 0.25;
            }
            weights.fc2_b = 0.1;
        }
        let input_shape = exec.network.net.input_shape();
        // A *varying* input matters here: LayerNorm's Jacobian annihilates the
        // all-ones direction, so a constant row would legitimately produce a
        // zero input gradient and the test would prove nothing.
        let data: Vec<f32> = (0..input_shape.batch * input_shape.element_count())
            .map(|i| (i % 7) as f32 * 0.3 - 0.9)
            .collect();
        let input = Tensor::from_data(input_shape, data).unwrap();
        let forward = exec.forward_training(&input).unwrap();
        let mut seeds = vec![None; exec.network.layers.len()];
        seeds[1] =
            Some(Tensor::from_data(exec.network.layers[1].output_shape, vec![1.0f32; 6]).unwrap());
        let result = exec
            .backward_accumulate_seeded(&input, &forward, &seeds)
            .unwrap();
        assert_eq!(
            result.input_gradient.data.len(),
            input_shape.batch * input_shape.element_count()
        );
        assert!(
            result.input_gradient.data.iter().any(|v| v.abs() > 1e-6),
            "the seed must reach the convolution: {:?}",
            result.input_gradient.data
        );
    }

    #[test]
    fn the_shape_contract_is_enforced_at_parse_time() {
        let bad_channel = format!(
            "[net]\nwidth=3\nheight=1\nchannels=5\n[convolutional]\nfilters=5\nsize=1\n\
             [laya_scorer]\nhidden=4\nmax_options=3\n"
        );
        let err = Network::from_config(ConfigFile::parse_str("laya.cfg", &bad_channel).unwrap())
            .expect_err("channels must equal hidden");
        assert!(format!("{err}").contains("channels=hidden=4"), "{err}");
        let bad_width = format!(
            "[net]\nwidth=5\nheight=1\nchannels=4\n[convolutional]\nfilters=4\nsize=1\n\
             [laya_scorer]\nhidden=4\nmax_options=3\n"
        );
        let err = Network::from_config(ConfigFile::parse_str("laya.cfg", &bad_width).unwrap())
            .expect_err("width must equal max_options");
        assert!(format!("{err}").contains("width=max_options=3"), "{err}");
    }

    #[test]
    fn the_real_blob_loads_into_the_layer_parameters() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/laya/laya_english.weights");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: {} is not present", path.display());
            return;
        };
        // The scorer must follow a layer that produces `[batch, hidden, 1,
        // max_options]`, which is exactly what `[laya_marker_gather]` would; a
        // 1x1 convolution stands in here so the test does not need that layer
        // ported first.
        let network =
            Network::from_config(ConfigFile::parse_str("laya.cfg", &cfg(1024, 4)).unwrap())
                .unwrap();
        let mut exec = ExecutableNetwork::new(network);
        let offset = scorer_blob_offset_bytes(1024, 28, 2624, 50368, 1024, 2, 4096, false);
        {
            let LayerParams::LayaScorer(weights) = &mut exec.params[1] else {
                panic!()
            };
            let blob = &bytes[offset..offset + 4 * blob_float_count(1024)];
            *weights = LayaScorerWeights::from_blob(
                &blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>(),
            )
            .unwrap();
        }
        let input_shape = exec.network.net.input_shape();
        let input = Tensor::zeros(input_shape);
        let forward = exec.forward_training(&input).unwrap();
        assert!(
            forward.layer_outputs[1].data.iter().all(|v| v.is_finite()),
            "the real scorer must produce finite logits"
        );
    }
}

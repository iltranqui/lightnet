//! `[laya_encoder]`: the frozen LAYA ModernBERT-large encoder (Phase 4 of
//! `tasks/rust/layers.md`), ported from
//! `Jev_laya:src-lib/laya_encoder_layer.cpp`.
//!
//! Frozen, CPU forward only (C++ `freeze=1`, backward is a no-op).  A
//! ModernBERT-large stack: token-embedding gather -> embed LayerNorm ->
//! `n_layers` pre-norm blocks (fused-QKV attention with alternating full /
//! sliding-window RoPE, GeGLU MLP) -> final LayerNorm -> channel-major output.
//!
//! Input is channel-major `[batch, 2, 1, seq_len]` (channel 0 = token id as a
//! float, channel 1 = 0/1 attention mask).  Output is channel-major
//! `[batch, hidden, 1, seq_len]`.
//!
//! Two details that are easy to get wrong (called out in the C++ header):
//! * Every LayerNorm is **bias-free** (gamma only), eps = 1e-5.
//! * GeGLU applies GELU (exact erf) to the **first** chunk of `Wi(x)` and
//!   multiplies by the raw second chunk (`input, gate = chunk(2,-1)`).
//! * The RoPE theta alternates: block `i % 3 == 0` is full-attention (theta
//!   `rope_theta_full`), all others use the sliding local window (theta
//!   `rope_theta_local`, radius `local_window / 2`).

/// The frozen encoder tensors, in C++ read order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaEncoderWeights {
    /// `[vocab, hidden]`, row-major.
    pub tok_embed: Vec<f32>,
    /// `[hidden]`.
    pub embed_norm: Vec<f32>,
    /// `[(n_layers - 1), hidden]` (absent for block 0).
    pub attn_norm: Vec<f32>,
    /// `[n_layers, 3*hidden, hidden]`.
    pub wqkv: Vec<f32>,
    /// `[n_layers, hidden, hidden]`.
    pub wo: Vec<f32>,
    /// `[n_layers, hidden]`.
    pub mlp_norm: Vec<f32>,
    /// `[n_layers, 2*intermediate, hidden]`.
    pub wi: Vec<f32>,
    /// `[n_layers, hidden, intermediate]`.
    pub wo_mlp: Vec<f32>,
    /// `[hidden]`.
    pub final_norm: Vec<f32>,
}

/// C++ `laya_encoder_blob_float_count(hidden, n_layers, intermediate, vocab)`.
pub fn blob_float_count(hidden: usize, n_layers: usize, intermediate: usize, vocab: usize) -> usize {
    let mut total = hidden * vocab;
    total += hidden;
    for i in 0..n_layers {
        if i > 0 {
            total += hidden;
        }
        total += 3 * hidden * hidden;
        total += hidden * hidden;
        total += hidden;
        total += 2 * intermediate * hidden;
        total += hidden * intermediate;
    }
    total += hidden;
    total
}

/// The geometry the forward needs (mirrors the `[laya_encoder]` cfg keys).
#[derive(Clone, Debug, PartialEq)]
pub struct LayaEncoderGeometry {
    pub batch: usize,
    pub seq_len: usize,
    pub hidden: usize,
    pub heads: usize,
    pub n_layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub local_window: usize,
    pub rope_theta_full: f32,
    pub rope_theta_local: f32,
    pub ln_eps: f32,
}

impl LayaEncoderWeights {
    pub fn zeroed(hidden: usize, n_layers: usize, intermediate: usize, vocab: usize) -> Self {
        // C++ `make_laya_encoder_layer` defaults every LayerNorm gamma to 1 so
        // an un-loaded encoder is a well-defined identity-scale LayerNorm.
        Self {
            tok_embed: vec![0.0; vocab * hidden],
            embed_norm: vec![1.0; hidden],
            attn_norm: vec![1.0; n_layers.saturating_sub(1) * hidden],
            wqkv: vec![0.0; n_layers * 3 * hidden * hidden],
            wo: vec![0.0; n_layers * hidden * hidden],
            mlp_norm: vec![1.0; n_layers * hidden],
            wi: vec![0.0; n_layers * 2 * intermediate * hidden],
            wo_mlp: vec![0.0; n_layers * hidden * intermediate],
            final_norm: vec![1.0; hidden],
        }
    }

    /// Per-tensor sizes in blob order.
    pub fn sizes(&self) -> [usize; 9] {
        [
            self.tok_embed.len(),
            self.embed_norm.len(),
            self.attn_norm.len(),
            self.wqkv.len(),
            self.wo.len(),
            self.mlp_norm.len(),
            self.wi.len(),
            self.wo_mlp.len(),
            self.final_norm.len(),
        ]
    }

    /// Length validation for a given geometry.
    pub fn matches(&self, g: &LayaEncoderGeometry) -> bool {
        let h = g.hidden;
        let i = g.intermediate;
        self.tok_embed.len() == g.vocab * h
            && self.embed_norm.len() == h
            && self.attn_norm.len() == g.n_layers.saturating_sub(1) * h
            && self.wqkv.len() == g.n_layers * 3 * h * h
            && self.wo.len() == g.n_layers * h * h
            && self.mlp_norm.len() == g.n_layers * h
            && self.wi.len() == g.n_layers * 2 * i * h
            && self.wo_mlp.len() == g.n_layers * h * i
            && self.final_norm.len() == h
    }

    /// The blob in C++ read order (layer-interleaved, matching
    /// `load_laya_encoder_weights`).
    pub fn to_blob(
        &self,
        hidden: usize,
        n_layers: usize,
        intermediate: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let (h, i) = (hidden, intermediate);
        let mut out = Vec::with_capacity(blob_float_count(hidden, n_layers, intermediate, vocab));
        out.extend_from_slice(&self.tok_embed);
        out.extend_from_slice(&self.embed_norm);
        for l in 0..n_layers {
            if l > 0 {
                out.extend_from_slice(&self.attn_norm[(l - 1) * h..l * h]);
            }
            out.extend_from_slice(&self.wqkv[l * 3 * h * h..(l + 1) * 3 * h * h]);
            out.extend_from_slice(&self.wo[l * h * h..(l + 1) * h * h]);
            out.extend_from_slice(&self.mlp_norm[l * h..(l + 1) * h]);
            out.extend_from_slice(&self.wi[l * 2 * i * h..(l + 1) * 2 * i * h]);
            out.extend_from_slice(&self.wo_mlp[l * h * i..(l + 1) * h * i]);
        }
        out.extend_from_slice(&self.final_norm);
        out
    }

    /// Inverse of [`Self::to_blob`]; the blob must be exactly
    /// [`blob_float_count`] long and in C++ layer-interleaved order.
    pub fn from_blob(
        blob: &[f32],
        hidden: usize,
        n_layers: usize,
        intermediate: usize,
        vocab: usize,
    ) -> Result<Self, String> {
        let expected = blob_float_count(hidden, n_layers, intermediate, vocab);
        if blob.len() != expected {
            return Err(format!(
                "laya_encoder blob has {} floats, expected {expected}",
                blob.len()
            ));
        }
        let (h, i, v) = (hidden, intermediate, vocab);
        let mut at = 0usize;
        let mut take = |len: usize| {
            let s = &blob[at..at + len];
            at += len;
            s
        };
        let tok_embed = take(v * h).to_vec();
        let embed_norm = take(h).to_vec();
        let mut attn_norm = Vec::with_capacity(n_layers.saturating_sub(1) * h);
        let mut wqkv = Vec::with_capacity(n_layers * 3 * h * h);
        let mut wo = Vec::with_capacity(n_layers * h * h);
        let mut mlp_norm = Vec::with_capacity(n_layers * h);
        let mut wi = Vec::with_capacity(n_layers * 2 * i * h);
        let mut wo_mlp = Vec::with_capacity(n_layers * h * i);
        for l in 0..n_layers {
            if l > 0 {
                attn_norm.extend_from_slice(take(h));
            }
            wqkv.extend_from_slice(take(3 * h * h));
            wo.extend_from_slice(take(h * h));
            mlp_norm.extend_from_slice(take(h));
            wi.extend_from_slice(take(2 * i * h));
            wo_mlp.extend_from_slice(take(h * i));
        }
        let final_norm = take(h).to_vec();
        Ok(Self {
            tok_embed,
            embed_norm,
            attn_norm,
            wqkv,
            wo,
            mlp_norm,
            wi,
            wo_mlp,
            final_norm,
        })
    }
}

/// Exact erf GELU (`torch.nn.GELU()`): `0.5 x (1 + erf(x / sqrt(2)))`.
pub fn gelu_erf(x: f32) -> f32 {
    crate::laya_scorer::gelu_erf(x)
}

/// Bias-free row-wise LayerNorm (gamma only).  `out` may alias `x`: each row's
/// mean/variance are computed before any element of the row is overwritten.
pub fn layernorm_bias_free(x: &[f32], gamma: &[f32], out: &mut [f32], rows: usize, c: usize, eps: f32) {
    let inv_c = 1.0 / c as f32;
    for r in 0..rows {
        let row = &x[r * c..(r + 1) * c];
        let mean = row.iter().sum::<f32>() * inv_c;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * inv_c;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..c {
            out[r * c + i] = gamma[i] * ((row[i] - mean) * inv_std);
        }
    }
}

/// RoPE cos/sin tables, `[seq_len * (head_dim/2)]`, entry `(p,j) = p * theta^(-j/half)`.
pub fn build_rope_table(seq_len: usize, head_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos_tab = vec![0.0f32; seq_len * half];
    let mut sin_tab = vec![0.0f32; seq_len * half];
    for p in 0..seq_len {
        for j in 0..half {
            let freq = theta.powf(-(j as f32) / (half as f32));
            let angle = p as f32 * freq;
            cos_tab[p * half + j] = angle.cos();
            sin_tab[p * half + j] = angle.sin();
        }
    }
    (cos_tab, sin_tab)
}

/// Apply RoPE (rotate-half / LLaMA-ModernBERT) to one `head_dim` vector in place.
pub fn apply_rope(v: &mut [f32], head_dim: usize, cos_row: &[f32], sin_row: &[f32]) {
    let half = head_dim / 2;
    for j in 0..half {
        let a = v[j];
        let b = v[j + half];
        let c = cos_row[j];
        let s = sin_row[j];
        v[j] = a * c - b * s;
        v[j + half] = b * c + a * s;
    }
}

/// `out[rows, n] = x[rows, k] @ w[n, k]^T` (C++ `gemm_cpu` transA=0, transB=1).
fn linear_nt(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize, out: &mut [f32]) {
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for j in 0..n {
            let wr = &w[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for d in 0..k {
                acc += xr[d] * wr[d];
            }
            out[r * n + j] = acc;
        }
    }
}

/// C++ `la_forward_impl`.  `input` is channel-major `[batch, 2, 1, seq_len]`;
/// the result is channel-major `[batch, hidden, 1, seq_len]`.
pub fn encoder_forward(
    input: &[f32],
    weights: &LayaEncoderWeights,
    g: &LayaEncoderGeometry,
) -> Result<Vec<f32>, String> {
    let (b, t, h, nh, nl, i, v) = (
        g.batch,
        g.seq_len,
        g.hidden,
        g.heads,
        g.n_layers,
        g.intermediate,
        g.vocab,
    );
    if b == 0 || t == 0 || h == 0 || nh == 0 || nl == 0 || i == 0 || v == 0 {
        return Err("laya_encoder: batch/seq/hidden/heads/layers/intermediate/vocab must be > 0".to_owned());
    }
    if h % nh != 0 {
        return Err(format!("laya_encoder: hidden={h} must be divisible by heads={nh}"));
    }
    let hd = h / nh;
    if hd % 2 != 0 {
        return Err(format!("laya_encoder: head_dim={hd} must be even (RoPE pairs)"));
    }
    if input.len() != b * 2 * t {
        return Err(format!(
            "laya_encoder: input has {} values, expected batch*2*seq_len={}",
            input.len(),
            b * 2 * t
        ));
    }
    if !weights.matches(g) {
        return Err("laya_encoder: weight tensors do not match the geometry".to_owned());
    }

    let half = hd / 2;
    let window_radius = g.local_window / 2;
    let (cos_full, sin_full) = build_rope_table(t, hd, g.rope_theta_full);
    let (cos_local, sin_local) = build_rope_table(t, hd, g.rope_theta_local);

    // 1. embedding gather + mask extraction.
    let mut mask = vec![0.0f32; b * t];
    let mut tokens = vec![0.0f32; b * t * h];
    for bi in 0..b {
        for ti in 0..t {
            let tok_f = input[(bi * 2) * t + ti];
            let mask_f = input[(bi * 2 + 1) * t + ti];
            let mut tid = tok_f.round() as i64;
            if tid < 0 {
                tid = 0;
            }
            if tid >= v as i64 {
                tid = v as i64 - 1;
            }
            mask[bi * t + ti] = mask_f;
            tokens[(bi * t + ti) * h..(bi * t + ti) * h + h]
                .copy_from_slice(&weights.tok_embed[tid as usize * h..tid as usize * h + h]);
        }
    }

    // 2. embedding LayerNorm (in place).
    let mut embed_ln = vec![0.0f32; b * t * h];
    layernorm_bias_free(&tokens, &weights.embed_norm, &mut embed_ln, b * t, h, g.ln_eps);
    tokens.copy_from_slice(&embed_ln);

    let scale = 1.0 / (hd as f32).sqrt();
    let mut xhat = vec![0.0f32; b * t * h];
    let mut qkv = vec![0.0f32; b * t * 3 * h];
    let mut attn_out = vec![0.0f32; b * t * h];
    let mut ctx = vec![0.0f32; b * t * h];
    let mut wi_out = vec![0.0f32; b * t * 2 * i];
    let mut ffn_out = vec![0.0f32; b * t * i];
    let mut scores = vec![0.0f32; b * nh * t * t];

    for blk in 0..nl {
        let sliding = blk % 3 != 0;
        let (cos_tab, sin_tab) = if sliding {
            (&cos_local, &sin_local)
        } else {
            (&cos_full, &sin_full)
        };
        let wqkv = &weights.wqkv[blk * 3 * h * h..];
        let wo = &weights.wo[blk * h * h..];
        let mlp_norm = &weights.mlp_norm[blk * h..];
        let wi = &weights.wi[blk * 2 * i * h..];
        let wo_mlp = &weights.wo_mlp[blk * h * i..];

        // attention half: pre-norm (block 0 has no attn_norm).
        let n: &[f32] = if blk == 0 {
            &tokens
        } else {
            let attn_norm = &weights.attn_norm[(blk - 1) * h..blk * h];
            layernorm_bias_free(&tokens, attn_norm, &mut xhat, b * t, h, g.ln_eps);
            &xhat
        };
        let mut qkv_view = vec![0.0f32; b * t * 3 * h];
        linear_nt(n, wqkv, b * t, h, 3 * h, &mut qkv_view);
        qkv.copy_from_slice(&qkv_view);

        // RoPE on q,k per (batch, token, head).
        for bi in 0..b {
            for ti in 0..t {
                let cos_row = &cos_tab[ti * half..ti * half + half];
                let sin_row = &sin_tab[ti * half..ti * half + half];
                let row = &mut qkv[(bi * t + ti) * 3 * h..(bi * t + ti) * 3 * h + 3 * h];
                for head in 0..nh {
                    apply_rope(&mut row[head * hd..head * hd + hd], hd, cos_row, sin_row);
                    apply_rope(
                        &mut row[h + head * hd..h + head * hd + hd],
                        hd,
                        cos_row,
                        sin_row,
                    );
                }
            }
        }

        // scaled dot-product attention with padding + optional local mask.
        for bi in 0..b {
            for head in 0..nh {
                let qkv_b = &qkv[bi * t * 3 * h..(bi + 1) * t * 3 * h];
                let mask_row = &mask[bi * t..(bi + 1) * t];
                let srow = &mut scores[(bi * nh + head) * t * t..(bi * nh + head + 1) * t * t];
                let aout = &mut attn_out[bi * t * h..(bi + 1) * t * h];
                for qi in 0..t {
                    let q = &qkv_b[qi * 3 * h + head * hd..qi * 3 * h + head * hd + hd];
                    let sr = &mut srow[qi * t..(qi + 1) * t];
                    let mut row_max = f32::NEG_INFINITY;
                    let mut any_valid = false;
                    for kj in 0..t {
                        let mut valid = mask_row[kj] > 0.5;
                        if sliding && valid {
                            valid = (qi as i64 - kj as i64).abs() <= window_radius as i64;
                        }
                        if !valid {
                            sr[kj] = f32::NEG_INFINITY;
                            continue;
                        }
                        any_valid = true;
                        let k = &qkv_b[kj * 3 * h + h + head * hd..kj * 3 * h + h + head * hd + hd];
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q[d] * k[d];
                        }
                        dot *= scale;
                        sr[kj] = dot;
                        if dot > row_max {
                            row_max = dot;
                        }
                    }
                    let ar = &mut aout[qi * h + head * hd..qi * h + head * hd + hd];
                    if !any_valid {
                        for d in ar.iter_mut() {
                            *d = 0.0;
                        }
                        continue;
                    }
                    let mut sum = 0.0f32;
                    for kj in 0..t {
                        if sr[kj] == f32::NEG_INFINITY {
                            sr[kj] = 0.0;
                            continue;
                        }
                        let e = (sr[kj] - row_max).exp();
                        sr[kj] = e;
                        sum += e;
                    }
                    let inv = 1.0 / sum;
                    for d in ar.iter_mut() {
                        *d = 0.0;
                    }
                    for kj in 0..t {
                        let p = sr[kj] * inv;
                        if p == 0.0 {
                            continue;
                        }
                        let vv =
                            &qkv_b[kj * 3 * h + 2 * h + head * hd..kj * 3 * h + 2 * h + head * hd + hd];
                        for d in 0..hd {
                            ar[d] += p * vv[d];
                        }
                    }
                }
            }
        }

        // Wo + residual (on the pre-norm stream).
        linear_nt(&attn_out, wo, b * t, h, h, &mut ctx);
        for idx in 0..b * t * h {
            tokens[idx] += ctx[idx];
        }

        // MLP half (GeGLU).
        layernorm_bias_free(&tokens, mlp_norm, &mut xhat, b * t, h, g.ln_eps);
        linear_nt(&xhat, wi, b * t, h, 2 * i, &mut wi_out);
        for r in 0..b * t {
            let wrow = &wi_out[r * 2 * i..(r + 1) * 2 * i];
            let orow = &mut ffn_out[r * i..(r + 1) * i];
            for j in 0..i {
                orow[j] = gelu_erf(wrow[j]) * wrow[i + j];
            }
        }
        linear_nt(&ffn_out, wo_mlp, b * t, i, h, &mut ctx);
        for idx in 0..b * t * h {
            tokens[idx] += ctx[idx];
        }
    }

    // 4. final LayerNorm (in place).
    let final_ln = {
        let mut out = vec![0.0f32; b * t * h];
        layernorm_bias_free(&tokens, &weights.final_norm, &mut out, b * t, h, g.ln_eps);
        out
    };
    tokens.copy_from_slice(&final_ln);

    // 5. transpose token-major -> channel-major.
    let mut output = vec![0.0f32; b * h * t];
    for bi in 0..b {
        for ti in 0..t {
            let src = &tokens[(bi * t + ti) * h..(bi * t + ti) * h + h];
            let dst = &mut output[bi * h * t..(bi + 1) * h * t];
            for e in 0..h {
                dst[e * t + ti] = src[e];
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(batch: usize, seq_len: usize, hidden: usize, heads: usize, n_layers: usize, intermediate: usize, vocab: usize) -> LayaEncoderGeometry {
        LayaEncoderGeometry {
            batch,
            seq_len,
            hidden,
            heads,
            n_layers,
            intermediate,
            vocab,
            local_window: 128,
            rope_theta_full: 160000.0,
            rope_theta_local: 10000.0,
            ln_eps: 1e-5,
        }
    }

    fn weights_with_unit_norms(hidden: usize, n_layers: usize, intermediate: usize, vocab: usize) -> LayaEncoderWeights {
        let mut w = LayaEncoderWeights::zeroed(hidden, n_layers, intermediate, vocab);
        w.embed_norm.fill(1.0);
        w.final_norm.fill(1.0);
        w.mlp_norm.fill(1.0);
        w.attn_norm.fill(1.0);
        w
    }

    #[test]
    fn bias_free_layernorm_matches_the_hand_computation() {
        let x = [1.0f32, 3.0];
        let gamma = [2.0f32, 0.5];
        let mut out = [0.0f32; 2];
        layernorm_bias_free(&x, &gamma, &mut out, 1, 2, 1e-5);
        // mean 2, var 1, inv_std ~ 1/sqrt(1+1e-5).
        let inv = 1.0 / (1.0f32 + 1e-5).sqrt();
        assert!((out[0] - 2.0 * (-1.0 * inv)).abs() < 1e-6, "{out:?}");
        assert!((out[1] - 0.5 * (1.0 * inv)).abs() < 1e-6);
    }

    #[test]
    fn rope_is_identity_at_position_zero_and_rotates_after() {
        let (cos, sin) = build_rope_table(2, 4, 10000.0);
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        apply_rope(&mut v, 4, &cos[0..2], &sin[0..2]);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
        // Atpos 1 the rotation preserves the vector norm.
        let mut v = [1.0f32, 2.0, 3.0, 4.0];
        let before = v.iter().map(|x| x * x).sum::<f32>();
        apply_rope(&mut v, 4, &cos[2..4], &sin[2..4]);
        let after = v.iter().map(|x| x * x).sum::<f32>();
        assert!((before - after).abs() < 1e-4, "{before} vs {after}");
    }

    #[test]
    fn zero_weights_with_unit_norms_gives_zero_output() {
        let g = geometry(1, 3, 4, 2, 2, 8, 10);
        let w = weights_with_unit_norms(4, 2, 8, 10);
        // All-zero token embeddings -> zero hidden states throughout.
        let input = vec![0.0f32, 1.0, 2.0, /*mask*/ 1.0, 1.0, 1.0];
        let out = encoder_forward(&input, &w, &g).unwrap();
        assert_eq!(out.len(), 1 * 4 * 3);
        assert!(out.iter().all(|v| v.abs() < 1e-6), "{out:?}");
    }

    #[test]
    fn a_zero_mean_unit_var_token_passes_through_with_zero_projections() {
        let g = geometry(1, 1, 2, 1, 1, 2, 3);
        let mut w = weights_with_unit_norms(2, 1, 2, 3);
        // Token 1 embedding = [1, -1] (mean 0, var 1).
        w.tok_embed[1 * 2] = 1.0;
        w.tok_embed[1 * 2 + 1] = -1.0;
        let input = vec![1.0f32, /*token*/ 1.0 /*mask*/];
        let out = encoder_forward(&input, &w, &g).unwrap();
        let inv = 1.0 / (1.0f32 + 1e-5).sqrt();
        assert!((out[0] - inv).abs() < 1e-5, "{out:?}");
        assert!((out[1] + inv).abs() < 1e-5);
    }

    #[test]
    fn blob_round_trip_and_count_match_the_scorer_helper() {
        let (hidden, n_layers, intermediate, vocab) = (4usize, 3usize, 6usize, 12usize);
        let w = weights_with_unit_norms(hidden, n_layers, intermediate, vocab);
        assert_eq!(
            w.to_blob(hidden, n_layers, intermediate, vocab).len(),
            crate::laya_scorer::encoder_blob_float_count(hidden, n_layers, intermediate, vocab)
        );
        assert_eq!(
            w.to_blob(hidden, n_layers, intermediate, vocab).len(),
            blob_float_count(hidden, n_layers, intermediate, vocab)
        );
        let back =
            LayaEncoderWeights::from_blob(&w.to_blob(hidden, n_layers, intermediate, vocab), hidden, n_layers, intermediate, vocab)
                .unwrap();
        assert_eq!(w, back);
        assert!(LayaEncoderWeights::from_blob(
            &w.to_blob(hidden, n_layers, intermediate, vocab)[..5],
            hidden,
            n_layers,
            intermediate,
            vocab
        )
        .is_err());
    }

    #[test]
    fn attention_respects_the_padding_mask() {
        // One head, hidden=2, seq=2, one layer, no projections except Wo = I and
        // Wqkv mapping so that v = token embedding.  With mask [1,0], token 1's
        // output attention context must be v(token 0).
        let g = geometry(1, 2, 2, 1, 1, 2, 3);
        let mut w = weights_with_unit_norms(2, 1, 2, 3);
        // Wqkv: q rows 0,1 = I; k rows 2,3 = I; v rows 4,5 = I (all [3H, H]).
        w.wqkv[0] = 1.0;
        w.wqkv[3] = 1.0; // row1 -> [0,1]
        w.wqkv[2 * 2 + 0] = 1.0;
        w.wqkv[2 * 2 + 3] = 1.0;
        w.wqkv[4 * 2 + 0] = 1.0;
        w.wqkv[4 * 2 + 3] = 1.0;
        // Wo = I.
        w.wo[0] = 1.0;
        w.wo[3] = 1.0;
        // distinct token embeddings
        w.tok_embed[0 * 2] = 10.0;
        w.tok_embed[0 * 2 + 1] = -10.0;
        w.tok_embed[1 * 2] = 1.0;
        w.tok_embed[1 * 2 + 1] = -1.0;
        // input: token 0, token 1, mask [1, 0]
        let input = vec![0.0f32, 1.0, 1.0, 0.0];
        let out = encoder_forward(&input, &w, &g).unwrap();
        assert_eq!(out.len(), 1 * 2 * 2);
        // Both positions attend only to position 0, so the two token columns
        // are equal: channel e, position t is out[e*T + t].
        assert!((out[0] - out[1]).abs() < 1e-4, "{out:?}");
        assert!((out[2] - out[3]).abs() < 1e-4);
    }

    #[test]
    fn shape_errors_are_reported() {
        let g = geometry(1, 2, 2, 1, 1, 2, 3);
        let w = weights_with_unit_norms(2, 1, 2, 3);
        // input too short
        assert!(encoder_forward(&[0.0, 1.0], &w, &g).is_err());
        // hidden not divisible by heads
        let mut bad = g.clone();
        bad.heads = 3;
        assert!(encoder_forward(&[0.0, 1.0, 1.0, 1.0], &w, &bad).is_err());
    }
}

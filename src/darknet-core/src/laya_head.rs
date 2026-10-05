//! `[laya_head]`: the frozen LAYA decision head (Phase 4 of
//! `tasks/rust/layers.md`), ported from `Jev_laya:src-lib/laya_head_layer.cpp`.
//!
//! Two plain pre-norm `nn.TransformerEncoderLayer`s (norm_first=True) over the
//! encoder's hidden states, after adding a per-question-type embedding:
//!
//! ```text
//! x = encoder_output + type_emb[qtype]              (broadcast over tokens)
//! for i in 0,1:
//!     n    = LayerNorm(x, norm1_w[i], norm1_b[i])       (WITH bias)
//!     qkv  = n @ in_proj_w[i]^T + in_proj_b[i]          (fused QKV, WITH bias)
//!     x    = x + (MHA(qkv, key_padding_mask) @ out_proj_w[i]^T + out_proj_b[i])
//!     n2   = LayerNorm(x, norm2_w[i], norm2_b[i])
//!     h1   = relu(n2 @ linear1_w[i]^T + linear1_b[i])   (ReLU, not GELU)
//!     x    = x + (h1 @ linear2_w[i]^T + linear2_b[i])
//! ```
//!
//! Input is channel-major `[batch, hidden+1(+1), 1, seq_len]`: channels
//! `[0,hidden)` are the encoder hidden states, channel `hidden` is the 0/1
//! attention mask, and with `qtype=-1` channel `hidden+1` carries a per-sample
//! qtype (rounded, clamped to `[0,2]`, read at `t=0`).  Output is channel-major
//! `[batch, hidden, 1, seq_len]`.
//!
//! `type_emb.weight[3,hidden]` is folded into this layer even though it sits in
//! the next ("decision head") blob: the blob order is the 24-tensor head blob
//! followed immediately by `type_emb`, matching C++ `load_laya_head_weights`.

/// The frozen head tensors, in C++ read order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaHeadWeights {
    pub norm1_w: Vec<f32>,
    pub norm1_b: Vec<f32>,
    /// `[n_layers, 3*hidden, hidden]`.
    pub in_proj_w: Vec<f32>,
    /// `[n_layers, 3*hidden]`.
    pub in_proj_b: Vec<f32>,
    pub out_proj_w: Vec<f32>,
    pub out_proj_b: Vec<f32>,
    pub norm2_w: Vec<f32>,
    pub norm2_b: Vec<f32>,
    pub linear1_w: Vec<f32>,
    pub linear1_b: Vec<f32>,
    pub linear2_w: Vec<f32>,
    pub linear2_b: Vec<f32>,
    /// `[3, hidden]`.
    pub type_emb: Vec<f32>,
}

/// C++ `laya_head_blob_float_count(hidden, n_layers, ffn_dim)`: the 24-tensor
/// head blob plus the folded-in `type_emb.weight (3*hidden)`.  Matches
/// `laya_scorer::head_blob_float_count`.
pub fn blob_float_count(hidden: usize, n_layers: usize, ffn_dim: usize) -> usize {
    let per_layer = 2 * hidden
        + 3 * hidden * hidden
        + 3 * hidden
        + hidden * hidden
        + hidden
        + 2 * hidden
        + ffn_dim * hidden
        + ffn_dim
        + hidden * ffn_dim
        + hidden;
    n_layers * per_layer + 3 * hidden
}

/// Geometry for the forward (mirrors the `[laya_head]` cfg keys).
#[derive(Clone, Debug, PartialEq)]
pub struct LayaHeadGeometry {
    pub batch: usize,
    pub seq_len: usize,
    pub hidden: usize,
    pub heads: usize,
    pub n_layers: usize,
    pub ffn_dim: usize,
    pub ln_eps: f32,
    /// `0..=2`, or `-1` to read a per-sample qtype from input channel `hidden+1`.
    pub qtype: i32,
}

impl LayaHeadWeights {
    pub fn zeroed(hidden: usize, n_layers: usize, ffn_dim: usize) -> Self {
        Self {
            norm1_w: vec![0.0; n_layers * hidden],
            norm1_b: vec![0.0; n_layers * hidden],
            in_proj_w: vec![0.0; n_layers * 3 * hidden * hidden],
            in_proj_b: vec![0.0; n_layers * 3 * hidden],
            out_proj_w: vec![0.0; n_layers * hidden * hidden],
            out_proj_b: vec![0.0; n_layers * hidden],
            norm2_w: vec![0.0; n_layers * hidden],
            norm2_b: vec![0.0; n_layers * hidden],
            linear1_w: vec![0.0; n_layers * ffn_dim * hidden],
            linear1_b: vec![0.0; n_layers * ffn_dim],
            linear2_w: vec![0.0; n_layers * hidden * ffn_dim],
            linear2_b: vec![0.0; n_layers * hidden],
            type_emb: vec![0.0; 3 * hidden],
        }
    }

    /// Per-tensor sizes in blob order.
    pub fn sizes(&self) -> [usize; 13] {
        [
            self.norm1_w.len(),
            self.norm1_b.len(),
            self.in_proj_w.len(),
            self.in_proj_b.len(),
            self.out_proj_w.len(),
            self.out_proj_b.len(),
            self.norm2_w.len(),
            self.norm2_b.len(),
            self.linear1_w.len(),
            self.linear1_b.len(),
            self.linear2_w.len(),
            self.linear2_b.len(),
            self.type_emb.len(),
        ]
    }

    pub fn matches(&self, g: &LayaHeadGeometry) -> bool {
        let (h, f, nl) = (g.hidden, g.ffn_dim, g.n_layers);
        self.norm1_w.len() == nl * h
            && self.norm1_b.len() == nl * h
            && self.in_proj_w.len() == nl * 3 * h * h
            && self.in_proj_b.len() == nl * 3 * h
            && self.out_proj_w.len() == nl * h * h
            && self.out_proj_b.len() == nl * h
            && self.norm2_w.len() == nl * h
            && self.norm2_b.len() == nl * h
            && self.linear1_w.len() == nl * f * h
            && self.linear1_b.len() == nl * f
            && self.linear2_w.len() == nl * h * f
            && self.linear2_b.len() == nl * h
            && self.type_emb.len() == 3 * h
    }

    /// The blob in C++ read order (layer-interleaved, matching
    /// `load_laya_head_weights`), with `type_emb` folded in at the end.
    pub fn to_blob(&self, hidden: usize, n_layers: usize, ffn_dim: usize) -> Vec<f32> {
        let (h, f) = (hidden, ffn_dim);
        let mut out = Vec::with_capacity(blob_float_count(hidden, n_layers, ffn_dim));
        for l in 0..n_layers {
            out.extend_from_slice(&self.norm1_w[l * h..(l + 1) * h]);
            out.extend_from_slice(&self.norm1_b[l * h..(l + 1) * h]);
            out.extend_from_slice(&self.in_proj_w[l * 3 * h * h..(l + 1) * 3 * h * h]);
            out.extend_from_slice(&self.in_proj_b[l * 3 * h..(l + 1) * 3 * h]);
            out.extend_from_slice(&self.out_proj_w[l * h * h..(l + 1) * h * h]);
            out.extend_from_slice(&self.out_proj_b[l * h..(l + 1) * h]);
            out.extend_from_slice(&self.norm2_w[l * h..(l + 1) * h]);
            out.extend_from_slice(&self.norm2_b[l * h..(l + 1) * h]);
            out.extend_from_slice(&self.linear1_w[l * f * h..(l + 1) * f * h]);
            out.extend_from_slice(&self.linear1_b[l * f..(l + 1) * f]);
            out.extend_from_slice(&self.linear2_w[l * h * f..(l + 1) * h * f]);
            out.extend_from_slice(&self.linear2_b[l * h..(l + 1) * h]);
        }
        out.extend_from_slice(&self.type_emb);
        out
    }

    /// Inverse of [`Self::to_blob`]; per-layer tensors are flattened across
    /// layers in the order the blob stores them.
    pub fn from_blob(
        blob: &[f32],
        hidden: usize,
        n_layers: usize,
        ffn_dim: usize,
    ) -> Result<Self, String> {
        let expected = blob_float_count(hidden, n_layers, ffn_dim);
        if blob.len() != expected {
            return Err(format!(
                "laya_head blob has {} floats, expected {expected}",
                blob.len()
            ));
        }
        // The blob interleaves all 12 per-layer tensors per layer, then
        // type_emb; rebuild the per-tensor flat arrays.
        let mut w = Self::zeroed(hidden, n_layers, ffn_dim);
        let (h, f) = (hidden, ffn_dim);
        let mut at = 0usize;
        let mut take = |len: usize| {
            let s = &blob[at..at + len];
            at += len;
            s
        };
        for layer in 0..n_layers {
            w.norm1_w[layer * h..(layer + 1) * h].copy_from_slice(take(h));
            w.norm1_b[layer * h..(layer + 1) * h].copy_from_slice(take(h));
            let n = 3 * h * h;
            w.in_proj_w[layer * n..(layer + 1) * n].copy_from_slice(take(n));
            w.in_proj_b[layer * 3 * h..(layer + 1) * 3 * h].copy_from_slice(take(3 * h));
            w.out_proj_w[layer * h * h..(layer + 1) * h * h].copy_from_slice(take(h * h));
            w.out_proj_b[layer * h..(layer + 1) * h].copy_from_slice(take(h));
            w.norm2_w[layer * h..(layer + 1) * h].copy_from_slice(take(h));
            w.norm2_b[layer * h..(layer + 1) * h].copy_from_slice(take(h));
            w.linear1_w[layer * f * h..(layer + 1) * f * h].copy_from_slice(take(f * h));
            w.linear1_b[layer * f..(layer + 1) * f].copy_from_slice(take(f));
            w.linear2_w[layer * h * f..(layer + 1) * h * f].copy_from_slice(take(h * f));
            w.linear2_b[layer * h..(layer + 1) * h].copy_from_slice(take(h));
        }
        w.type_emb.copy_from_slice(take(3 * h));
        Ok(w)
    }
}

/// Row-wise LayerNorm **with bias**.  `out` may alias `x` (each row's mean/variance
/// are computed before any element of the row is overwritten).
pub fn layernorm(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32], rows: usize, c: usize, eps: f32) {
    let inv_c = 1.0 / c as f32;
    for r in 0..rows {
        let row = &x[r * c..(r + 1) * c];
        let mean = row.iter().sum::<f32>() * inv_c;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() * inv_c;
        let inv_std = 1.0 / (var + eps).sqrt();
        for i in 0..c {
            out[r * c + i] = gamma[i] * ((row[i] - mean) * inv_std) + beta[i];
        }
    }
}

/// `out[rows, n] = x[rows, k] @ w[n, k]^T`, optionally adding a per-row bias.
fn linear_nt(x: &[f32], w: &[f32], bias: Option<&[f32]>, rows: usize, k: usize, n: usize, out: &mut [f32]) {
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for j in 0..n {
            let wr = &w[j * k..(j + 1) * k];
            let mut acc = match bias {
                Some(b) => b[j],
                None => 0.0,
            };
            for d in 0..k {
                acc += xr[d] * wr[d];
            }
            out[r * n + j] = acc;
        }
    }
}

/// The per-sample qtype (C++ `laya_head_sample_qtype`).
pub fn sample_qtype(cfg_qtype: i32, input: &[f32], b: usize, c: usize, hidden: usize, t: usize) -> i32 {
    if cfg_qtype >= 0 {
        return cfg_qtype;
    }
    let q = input[(b * c + hidden + 1) * t].round() as i32;
    q.clamp(0, 2)
}

/// Number of input channels for `qtype`.
pub fn input_channels(hidden: usize, qtype: i32) -> usize {
    hidden + if qtype < 0 { 2 } else { 1 }
}

/// C++ `lh_forward_impl`.  `input` is channel-major `[batch, C, 1, seq_len]`.
pub fn head_forward(
    input: &[f32],
    weights: &LayaHeadWeights,
    g: &LayaHeadGeometry,
) -> Result<Vec<f32>, String> {
    let (b, t, h, nh, nl, f) = (
        g.batch,
        g.seq_len,
        g.hidden,
        g.heads,
        g.n_layers,
        g.ffn_dim,
    );
    if b == 0 || t == 0 || h == 0 || nh == 0 || nl == 0 || f == 0 {
        return Err("laya_head: batch/seq/hidden/heads/layers/ffn_dim must be > 0".to_owned());
    }
    if h % nh != 0 {
        return Err(format!("laya_head: hidden={h} must be divisible by heads={nh}"));
    }
    if !(-1..=2).contains(&g.qtype) {
        return Err(format!("laya_head: qtype={} must be in [0,2], or -1", g.qtype));
    }
    let c = input_channels(h, g.qtype);
    if input.len() != b * c * t {
        return Err(format!(
            "laya_head: input has {} values, expected batch*C*seq_len={}",
            input.len(),
            b * c * t
        ));
    }
    if !weights.matches(g) {
        return Err("laya_head: weight tensors do not match the geometry".to_owned());
    }
    let hd = h / nh;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut mask = vec![0.0f32; b * t];
    let mut tokens = vec![0.0f32; b * t * h];
    for bi in 0..b {
        let qt = sample_qtype(g.qtype, input, bi, c, h, t) as usize;
        let type_row = &weights.type_emb[qt * h..(qt + 1) * h];
        for ti in 0..t {
            let dst = &mut tokens[(bi * t + ti) * h..(bi * t + ti) * h + h];
            for e in 0..h {
                dst[e] = input[(bi * c + e) * t + ti] + type_row[e];
            }
            mask[bi * t + ti] = input[(bi * c + h) * t + ti];
        }
    }

    let mut xhat = vec![0.0f32; b * t * h];
    let mut qkv = vec![0.0f32; b * t * 3 * h];
    let mut attn_out = vec![0.0f32; b * t * h];
    let mut ctx = vec![0.0f32; b * t * h];
    let mut ffn = vec![0.0f32; b * t * f];
    let mut scores = vec![0.0f32; b * nh * t * t];

    for blk in 0..nl {
        let norm1_w = &weights.norm1_w[blk * h..(blk + 1) * h];
        let norm1_b = &weights.norm1_b[blk * h..(blk + 1) * h];
        let in_proj_w = &weights.in_proj_w[blk * 3 * h * h..];
        let in_proj_b = &weights.in_proj_b[blk * 3 * h..(blk + 1) * 3 * h];
        let out_proj_w = &weights.out_proj_w[blk * h * h..];
        let out_proj_b = &weights.out_proj_b[blk * h..(blk + 1) * h];
        let norm2_w = &weights.norm2_w[blk * h..(blk + 1) * h];
        let norm2_b = &weights.norm2_b[blk * h..(blk + 1) * h];
        let linear1_w = &weights.linear1_w[blk * f * h..];
        let linear1_b = &weights.linear1_b[blk * f..(blk + 1) * f];
        let linear2_w = &weights.linear2_w[blk * h * f..];
        let linear2_b = &weights.linear2_b[blk * h..(blk + 1) * h];

        // self-attention half (pre-norm).
        layernorm(&tokens, norm1_w, norm1_b, &mut xhat, b * t, h, g.ln_eps);
        linear_nt(&xhat, in_proj_w, Some(in_proj_b), b * t, h, 3 * h, &mut qkv);

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
                        if mask_row[kj] <= 0.5 {
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

        linear_nt(&attn_out, out_proj_w, None, b * t, h, h, &mut ctx);
        for r in 0..b * t {
            let tok = &mut tokens[r * h..(r + 1) * h];
            let cctx = &ctx[r * h..(r + 1) * h];
            for e in 0..h {
                tok[e] += cctx[e] + out_proj_b[e];
            }
        }

        // FFN half (ReLU).
        layernorm(&tokens, norm2_w, norm2_b, &mut xhat, b * t, h, g.ln_eps);
        linear_nt(&xhat, linear1_w, Some(linear1_b), b * t, h, f, &mut ffn);
        for value in ffn.iter_mut() {
            *value = value.max(0.0);
        }
        linear_nt(&ffn, linear2_w, None, b * t, f, h, &mut ctx);
        for r in 0..b * t {
            let tok = &mut tokens[r * h..(r + 1) * h];
            let cctx = &ctx[r * h..(r + 1) * h];
            for e in 0..h {
                tok[e] += cctx[e] + linear2_b[e];
            }
        }
    }

    // token-major -> channel-major.
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

    fn geometry(batch: usize, seq_len: usize, hidden: usize, heads: usize, n_layers: usize, ffn_dim: usize, qtype: i32) -> LayaHeadGeometry {
        LayaHeadGeometry {
            batch,
            seq_len,
            hidden,
            heads,
            n_layers,
            ffn_dim,
            ln_eps: 1e-5,
            qtype,
        }
    }

    fn unit_norm_weights(hidden: usize, n_layers: usize, ffn_dim: usize) -> LayaHeadWeights {
        let mut w = LayaHeadWeights::zeroed(hidden, n_layers, ffn_dim);
        w.norm1_w.fill(1.0);
        w.norm2_w.fill(1.0);
        w
    }

    #[test]
    fn layernorm_with_bias_matches_the_hand_computation() {
        let x = [1.0f32, 3.0];
        let gamma = [2.0f32, 0.5];
        let beta = [0.25f32, -0.5];
        let mut out = [0.0f32; 2];
        layernorm(&x, &gamma, &beta, &mut out, 1, 2, 1e-5);
        let inv = 1.0 / (1.0f32 + 1e-5).sqrt();
        assert!((out[0] - (2.0 * (-1.0 * inv) + 0.25)).abs() < 1e-6, "{out:?}");
        assert!((out[1] - (0.5 * (1.0 * inv) - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn zero_projections_leave_a_double_normalised_signal() {
        // One layer, hidden=2, zero projections, unit norms -> output = LN(LN(x)).
        let g = geometry(1, 1, 2, 1, 1, 2, 0);
        let w = unit_norm_weights(2, 1, 2);
        let input = vec![1.0f32, -1.0, 1.0]; // hidden [1,-1], mask 1
        let out = head_forward(&input, &w, &g).unwrap();
        let inv = 1.0 / (1.0f32 + 1e-5).sqrt();
        assert!((out[0] - inv).abs() < 1e-5, "{out:?}");
        assert!((out[1] + inv).abs() < 1e-5);
    }

    #[test]
    fn qtype_embedding_is_added_before_the_blocks() {
        let g = geometry(1, 1, 2, 1, 1, 2, 1);
        let mut w = unit_norm_weights(2, 1, 2);
        // type_emb[1] = [1, 1]; input hidden = [0, 0] -> x = [1, 1] -> LN = 0.
        w.type_emb[1 * 2] = 1.0;
        w.type_emb[1 * 2 + 1] = 1.0;
        let input = vec![0.0f32, 0.0, 1.0];
        let out = head_forward(&input, &w, &g).unwrap();
        // Pre-norm residual: with zero projections the type embedding survives.
        assert!((out[0] - 1.0).abs() < 1e-5, "{out:?}");
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn qtype_minus_one_reads_the_per_sample_channel() {
        // Two samples with input qtypes 0 and 2; type_emb[2] shifts sample 1.
        let g = geometry(2, 1, 2, 1, 1, 2, -1);
        let mut w = unit_norm_weights(2, 1, 2);
        w.type_emb[2 * 2] = 5.0;
        w.type_emb[2 * 2 + 1] = 5.0;
        // [b][c][h][w]: C = hidden + 2 = 4. Sample 0 qtype channel = 0,
        // sample 1 = 2. hidden = [0,0] both.
        let mut input = vec![0.0f32; 2 * 4 * 1];
        input[(1 * 4 + 3) * 1] = 2.0; // sample 1 qtype channel
        let out = head_forward(&input, &w, &g).unwrap();
        assert!(out[0].abs() < 1e-6 && out[1].abs() < 1e-6, "sample 0 unshifted {out:?}");
        // sample 1 has x = [5,5] -> LN = 0 too, so both are zero; use a
        // non-uniform shift instead to prove the qtype row was selected.
        w.type_emb[2 * 2] = 1.0;
        let out = head_forward(&input, &w, &g).unwrap();
        // sample 1 x = [1,0] -> LN = [1,-1]/sqrt(0.5+eps)-ish, nonzero.
        assert!(out[2].abs() > 1e-3 || out[3].abs() > 1e-3, "{out:?}");
    }

    #[test]
    fn blob_round_trip_and_count_match_the_scorer_helper() {
        let (hidden, n_layers, ffn_dim) = (4usize, 2usize, 6usize);
        let mut w = unit_norm_weights(hidden, n_layers, ffn_dim);
        for (i, v) in w.linear1_w.iter_mut().enumerate() {
            *v = i as f32 * 0.01;
        }
        for (i, v) in w.type_emb.iter_mut().enumerate() {
            *v = i as f32 * 0.1;
        }
        assert_eq!(
            w.to_blob(hidden, n_layers, ffn_dim).len(),
            crate::laya_scorer::head_blob_float_count(hidden, n_layers, ffn_dim)
        );
        assert_eq!(
            w.to_blob(hidden, n_layers, ffn_dim).len(),
            blob_float_count(hidden, n_layers, ffn_dim)
        );
        let back =
            LayaHeadWeights::from_blob(&w.to_blob(hidden, n_layers, ffn_dim), hidden, n_layers, ffn_dim)
                .unwrap();
        assert_eq!(w, back);
        assert!(LayaHeadWeights::from_blob(
            &w.to_blob(hidden, n_layers, ffn_dim)[..3],
            hidden,
            n_layers,
            ffn_dim
        )
        .is_err());
    }

    #[test]
    fn shape_errors_are_reported() {
        let g = geometry(1, 2, 2, 1, 1, 2, 0);
        let w = unit_norm_weights(2, 1, 2);
        // input too short (needs C=3 * T=2 = 6).
        assert!(head_forward(&[0.0, 1.0], &w, &g).is_err());
        // hidden not divisible by heads.
        let mut bad = g.clone();
        bad.heads = 3;
        assert!(head_forward(&[0.0; 6], &w, &bad).is_err());
    }
}

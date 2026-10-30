//! Flash attention forward and backward (dQ, dK/dV), shared by the transformer layers.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// ViT MSA scores in `[batch, heads, query, key]` order. The activity
    /// flag records the score-clamp derivative before selecting blocks.
    #[kernel]
    pub fn vit_msa_scores(qkv: &[f32], mut scores: DisjointSlice<f32>,
        mut active: DisjointSlice<f32>, tokens: usize, channels: usize, heads: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= scores.len() { return; }
        let key = i % tokens;
        let query = (i / tokens) % tokens;
        let head = (i / (tokens * tokens)) % heads;
        let batch = i / (tokens * tokens * heads);
        let dim = channels / heads;
        let qbase = (batch * tokens + query) * 3 * channels + head * dim;
        let kbase = (batch * tokens + key) * 3 * channels + channels + head * dim;
        let mut dot = 0.0f32;
        for d in 0..dim {
            let q = qkv[qbase + d];
            let k = qkv[kbase + d];
            let q = if q.is_finite() { q.max(-128.0).min(128.0) } else { 0.0 };
            let k = if k.is_finite() { k.max(-128.0).min(128.0) } else { 0.0 };
            dot += q * k;
        }
        let raw = dot / (dim as f32).sqrt();
        let score = if raw.is_nan() { 0.0 } else { raw.max(-10.0).min(10.0) };
        if let Some(dst) = scores.get_mut(idx) { *dst = score; }
        if let Some(dst) = active.get_mut(thread::index_1d()) {
            *dst = if raw.is_finite() && raw.abs() < 10.0 { 1.0 } else { 0.0 };
        }
    }

    /// Each thread owns one output channel and gathers the selected values.
    #[kernel]
    pub fn vit_msa_context(qkv: &[f32], probabilities: &[f32],
        mut context: DisjointSlice<f32>, tokens: usize, channels: usize, heads: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= context.len() { return; }
        let channel = i % channels;
        let query = (i / channels) % tokens;
        let batch = i / (channels * tokens);
        let head = channel / (channels / heads);
        let row = ((batch * heads + head) * tokens + query) * tokens;
        let mut value = 0.0f32;
        for key in 0..tokens {
            value += probabilities[row + key] * qkv[(batch * tokens + key) * 3 * channels + 2 * channels + channel];
        }
        if let Some(dst) = context.get_mut(idx) { *dst = value; }
    }

    /// Gather dQ, dK and dV in the original fused per-token QKV order.
    #[kernel]
    pub fn vit_msa_backward(qkv: &[f32], probabilities: &[f32], active: &[f32],
        context: &[f32], grad: &[f32], mut dqkv: DisjointSlice<f32>,
        tokens: usize, channels: usize, heads: usize) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= dqkv.len() { return; }
        let part = (i % (3 * channels)) / channels;
        let channel = i % channels;
        let token = (i / (3 * channels)) % tokens;
        let batch = i / (3 * channels * tokens);
        let dim = channels / heads;
        let head = channel / dim;
        let head_start = head * dim;
        let mut result = 0.0f32;
        if part == 2 {
            for query in 0..tokens {
                let row = ((batch * heads + head) * tokens + query) * tokens;
                result += probabilities[row + token] * grad[(batch * tokens + query) * channels + channel];
            }
        } else {
            let raw = qkv[i];
            if raw.is_finite() && raw.abs() < 128.0 {
                for other in 0..tokens {
                    let (query, key) = if part == 0 { (token, other) } else { (other, token) };
                    let row = ((batch * heads + head) * tokens + query) * tokens;
                    if active[row + key] == 0.0 { continue; }
                    let mut dp = 0.0f32;
                    let mut dot = 0.0f32;
                    for d in 0..dim {
                        let ch = head_start + d;
                        let dy = grad[(batch * tokens + query) * channels + ch];
                        let v = qkv[(batch * tokens + key) * 3 * channels + 2 * channels + ch];
                        dp += dy * v;
                        dot += dy * context[(batch * tokens + query) * channels + ch];
                    }
                    let ds = probabilities[row + key] * (dp - dot);
                    if ds == 0.0 { continue; }
                    let offset = if part == 0 { channels } else { 0 };
                    let other_token = if part == 0 { key } else { query };
                    let other_raw = qkv[(batch * tokens + other_token) * 3 * channels + offset + channel];
                    let other_value = if other_raw.is_finite() {
                        other_raw.max(-128.0).min(128.0)
                    } else { 0.0 };
                    result += ds * other_value / (dim as f32).sqrt();
                }
            }
        }
        if let Some(dst) = dqkv.get_mut(idx) { *dst = result; }
    }

    /// One query row per thread. Online max/sum and the value accumulator keep
    /// the attention workspace linear in sequence length; the host rejects
    /// head dimensions above the fixed local accumulator size.
    #[kernel]
    pub fn flash_attention_forward(
        qkv: &[f32],
        mut out: DisjointSlice<f32>,
        mut lse: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let query = thread::index_1d().get();
        let query_count = batch * heads * tokens;
        if query >= query_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = query / tokens;
        let tq = query - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let query_row = b * tokens + tq;
        let head_start = h * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut acc = [0.0f32; 128];
        let q_base = query_row * 3 * channels + head_start;
        let mut d = 0usize;
        while d < head_dim {
            let raw = qkv[q_base + d];
            q[d] = if raw.is_finite() {
                raw.max(-128.0).min(128.0)
            } else {
                0.0
            };
            d += 1;
        }

        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        let mut tk = 0usize;
        while tk < tokens {
            let key_row = b * tokens + tk;
            let k_base = key_row * 3 * channels + channels + head_start;
            let mut dot = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw = qkv[k_base + d];
                let k = if raw.is_finite() {
                    raw.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q[d] * k;
                d += 1;
            }
            let raw_score = dot * scale;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let new_max = running_max.max(score);
            let correction = (running_max - new_max).exp();
            let p = (score - new_max).exp();
            running_sum = running_sum * correction + p;
            let v_base = key_row * 3 * channels + 2 * channels + head_start;
            d = 0;
            while d < head_dim {
                acc[d] = acc[d] * correction + p * qkv[v_base + d];
                d += 1;
            }
            running_max = new_max;
            tk += 1;
        }

        let output_base = query_row * channels + head_start;
        d = 0;
        while d < head_dim {
            // SAFETY: each (batch, head, query) thread owns this head's disjoint
            // output channel interval for one query row.
            unsafe {
                *out.as_mut_ptr().add(output_base + d) = acc[d] / running_sum;
            }
            d += 1;
        }
        // LSE is flattened as `[batch, heads, tokens]`, matching `query`.
        unsafe {
            *lse.as_mut_ptr().add(query) = running_max + running_sum.ln();
        }
    }

    /// One query row per thread computes dQ and D_i = dot(context_i, dO_i).
    /// The following key-row kernel consumes D_i to form dK and dV without atomics.
    #[kernel]
    pub fn flash_attention_backward_dq(
        qkv: &[f32],
        context: &[f32],
        output_grad: &[f32],
        lse: &[f32],
        mut dqkv: DisjointSlice<f32>,
        mut row_dot: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let query = thread::index_1d().get();
        let query_count = batch * heads * tokens;
        if query >= query_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = query / tokens;
        let tq = query - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let query_row = b * tokens + tq;
        let head_start = h * head_dim;
        let q_base = query_row * 3 * channels + head_start;
        let row_base = query_row * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut q = [0.0f32; 128];
        let mut dq = [0.0f32; 128];
        let mut row_product = 0.0f32;
        let mut d = 0usize;
        while d < head_dim {
            let raw_q = qkv[q_base + d];
            q[d] = if raw_q.is_finite() {
                raw_q.max(-128.0).min(128.0)
            } else {
                0.0
            };
            row_product += output_grad[row_base + d] * context[row_base + d];
            dq[d] = 0.0;
            d += 1;
        }

        let lse_i = lse[query];
        let mut tk = 0usize;
        while tk < tokens {
            let key_row = b * tokens + tk;
            let k_base = key_row * 3 * channels + channels + head_start;
            let v_base = key_row * 3 * channels + 2 * channels + head_start;
            let mut dot = 0.0f32;
            let mut dprob = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw_k = qkv[k_base + d];
                let k = if raw_k.is_finite() {
                    raw_k.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q[d] * k;
                dprob += output_grad[row_base + d] * qkv[v_base + d];
                d += 1;
            }
            let raw_score = dot * scale;
            let active = raw_score.is_finite() && raw_score.abs() < 10.0;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let p = (score - lse_i).exp();
            if active {
                let dscore = p * (dprob - row_product);
                d = 0;
                while d < head_dim {
                    let raw_k = qkv[k_base + d];
                    let k = if raw_k.is_finite() {
                        raw_k.max(-128.0).min(128.0)
                    } else {
                        0.0
                    };
                    dq[d] += dscore * k * scale;
                    d += 1;
                }
            }
            tk += 1;
        }

        d = 0;
        while d < head_dim {
            let raw_q = qkv[q_base + d];
            let grad = if raw_q.is_finite() && raw_q.abs() < 128.0 {
                dq[d]
            } else {
                0.0
            };
            // SAFETY: each query/head thread owns this Q channel interval.
            unsafe {
                *dqkv
                    .as_mut_ptr()
                    .add(query_row * 3 * channels + head_start + d) = grad;
            }
            d += 1;
        }
        // SAFETY: one thread owns each flattened (batch, head, query) scalar.
        unsafe {
            *row_dot.as_mut_ptr().add(query) = row_product;
        }
    }

    /// One key row per thread accumulates dK/dV across all query rows; every output
    /// row is owned by one thread, so the recompute path requires no atomics.
    #[kernel]
    pub fn flash_attention_backward_dkdv(
        qkv: &[f32],
        output_grad: &[f32],
        lse: &[f32],
        row_dot: &[f32],
        mut dqkv: DisjointSlice<f32>,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
    ) {
        let key = thread::index_1d().get();
        let key_count = batch * heads * tokens;
        if key >= key_count {
            return;
        }
        let head_dim = channels / heads;
        let bh = key / tokens;
        let tk = key - bh * tokens;
        let b = bh / heads;
        let h = bh - b * heads;
        let key_row = b * tokens + tk;
        let head_start = h * head_dim;
        let k_base = key_row * 3 * channels + channels + head_start;
        let v_base = key_row * 3 * channels + 2 * channels + head_start;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut k = [0.0f32; 128];
        let mut dk = [0.0f32; 128];
        let mut dv = [0.0f32; 128];
        let mut d = 0usize;
        while d < head_dim {
            let raw_k = qkv[k_base + d];
            k[d] = if raw_k.is_finite() {
                raw_k.max(-128.0).min(128.0)
            } else {
                0.0
            };
            dk[d] = 0.0;
            dv[d] = 0.0;
            d += 1;
        }

        let mut tq = 0usize;
        while tq < tokens {
            let query_row = b * tokens + tq;
            let query = (b * heads + h) * tokens + tq;
            let q_base = query_row * 3 * channels + head_start;
            let dy_base = query_row * channels + head_start;
            let mut dot = 0.0f32;
            let mut dprob = 0.0f32;
            d = 0;
            while d < head_dim {
                let raw_q = qkv[q_base + d];
                let q = if raw_q.is_finite() {
                    raw_q.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dot += q * k[d];
                dprob += output_grad[dy_base + d] * qkv[v_base + d];
                d += 1;
            }
            let raw_score = dot * scale;
            let active = raw_score.is_finite() && raw_score.abs() < 10.0;
            let score = if raw_score.is_nan() {
                0.0
            } else if raw_score > 10.0 {
                10.0
            } else if raw_score < -10.0 {
                -10.0
            } else {
                raw_score
            };
            let p = (score - lse[query]).exp();
            let dscore = p * (dprob - row_dot[query]);
            d = 0;
            while d < head_dim {
                let raw_q = qkv[q_base + d];
                let q = if raw_q.is_finite() {
                    raw_q.max(-128.0).min(128.0)
                } else {
                    0.0
                };
                dv[d] += p * output_grad[dy_base + d];
                if active {
                    dk[d] += dscore * q * scale;
                }
                d += 1;
            }
            tq += 1;
        }

        d = 0;
        while d < head_dim {
            let raw_k = qkv[k_base + d];
            let dkey = if raw_k.is_finite() && raw_k.abs() < 128.0 {
                dk[d]
            } else {
                0.0
            };
            // SAFETY: each key/head thread owns one K and one V channel interval.
            unsafe {
                *dqkv
                    .as_mut_ptr()
                    .add(key_row * 3 * channels + channels + head_start + d) = dkey;
                *dqkv
                    .as_mut_ptr()
                    .add(key_row * 3 * channels + 2 * channels + head_start + d) = dv[d];
            }
            d += 1;
        }
    }
}

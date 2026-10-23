//! Causal grouped-query attention and its exact adjoint.
//!
//! `q` is `[rows, q_heads, head_dim]`, `k`/`v` are `[rows, kv_heads,
//! head_dim]`.  A query may attend to a key only when both rows are valid, the
//! segments match, and the key's time index is not later than the query's.
//! Query heads are grouped contiguously: head `h` reads KV head
//! `h / (q_heads / kv_heads)`.

use crate::{DecoderConfig, Result, TokenBatch};

/// Masked causal GQA forward.  RoPE must already have been applied by the
/// caller; this function only uses segment/valid/time.
pub fn causal_gqa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: &TokenBatch,
    cfg: &DecoderConfig,
) -> Result<Vec<f32>> {
    let dims = GqaDims::new(q, k, v, batch, cfg)?;
    let mut out = vec![0.0; q.len()];
    for row in 0..dims.rows {
        if !batch.valid[row] {
            continue;
        }
        let time_index = row % batch.time;
        for query_head in 0..dims.q_heads {
            let key_head = query_head / dims.group;
            let (keys, probabilities) =
                masked_softmax(q, k, batch, row, time_index, query_head, key_head, &dims);
            if keys.is_empty() {
                continue;
            }
            let query_base = (row * dims.q_heads + query_head) * dims.head_dim;
            for (index, &key_row) in keys.iter().enumerate() {
                let probability = probabilities[index];
                let key_base = (key_row * dims.kv_heads + key_head) * dims.head_dim;
                for channel in 0..dims.head_dim {
                    out[query_base + channel] += probability * v[key_base + channel];
                }
            }
        }
    }
    Ok(out)
}

/// Exact adjoint of [`causal_gqa`]; returns `(dq, dk, dv)`.
pub fn causal_gqa_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dy: &[f32],
    batch: &TokenBatch,
    cfg: &DecoderConfig,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let dims = GqaDims::new(q, k, v, batch, cfg)?;
    if dy.len() != q.len() {
        return Err(crate::Error::InvalidConfig(format!(
            "attention dy has length {}, expected {}",
            dy.len(),
            q.len()
        )));
    }

    let mut dq = vec![0.0; q.len()];
    let mut dk = vec![0.0; k.len()];
    let mut dv = vec![0.0; v.len()];

    for row in 0..dims.rows {
        if !batch.valid[row] {
            continue;
        }
        let time_index = row % batch.time;
        for query_head in 0..dims.q_heads {
            let key_head = query_head / dims.group;
            let (keys, probabilities) =
                masked_softmax(q, k, batch, row, time_index, query_head, key_head, &dims);
            if keys.is_empty() {
                continue;
            }
            let query_base = (row * dims.q_heads + query_head) * dims.head_dim;
            let dy_row = &dy[query_base..query_base + dims.head_dim];

            let mut out = vec![0.0; dims.head_dim];
            for (index, &key_row) in keys.iter().enumerate() {
                let key_base = (key_row * dims.kv_heads + key_head) * dims.head_dim;
                for channel in 0..dims.head_dim {
                    out[channel] += probabilities[index] * v[key_base + channel];
                }
            }
            let dy_dot_out: f32 = dy_row
                .iter()
                .zip(&out)
                .map(|(&left, &right)| left * right)
                .sum();

            for (index, &key_row) in keys.iter().enumerate() {
                let key_base = (key_row * dims.kv_heads + key_head) * dims.head_dim;
                let dy_dot_value: f32 = dy_row
                    .iter()
                    .zip(&v[key_base..key_base + dims.head_dim])
                    .map(|(&left, &right)| left * right)
                    .sum();
                let score_grad = probabilities[index] * (dy_dot_value - dy_dot_out);
                let query = &q[query_base..query_base + dims.head_dim];
                let key = &k[key_base..key_base + dims.head_dim];
                for channel in 0..dims.head_dim {
                    dq[query_base + channel] += dims.scale * score_grad * key[channel];
                    dk[key_base + channel] += dims.scale * score_grad * query[channel];
                    dv[key_base + channel] += probabilities[index] * dy_row[channel];
                }
            }
        }
    }

    Ok((dq, dk, dv))
}

struct GqaDims {
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    group: usize,
    head_dim: usize,
    scale: f32,
}

impl GqaDims {
    fn new(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch: &TokenBatch,
        cfg: &DecoderConfig,
    ) -> Result<Self> {
        cfg.validate()?;
        batch.validate(cfg)?;
        let rows = batch.rows();
        let head_dim = cfg.head_dim();
        let q_len = rows
            .checked_mul(cfg.q_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| {
                crate::Error::InvalidConfig("attention Q size overflows usize".to_owned())
            })?;
        let kv_len = rows
            .checked_mul(cfg.kv_heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| {
                crate::Error::InvalidConfig("attention KV size overflows usize".to_owned())
            })?;
        if q.len() != q_len {
            return Err(crate::Error::InvalidConfig(format!(
                "attention Q has length {}, expected {q_len}",
                q.len()
            )));
        }
        if k.len() != kv_len || v.len() != kv_len {
            return Err(crate::Error::InvalidConfig(format!(
                "attention K/V lengths ({}, {}) do not match expected {kv_len}",
                k.len(),
                v.len()
            )));
        }
        Ok(Self {
            rows,
            q_heads: cfg.q_heads,
            kv_heads: cfg.kv_heads,
            group: cfg.q_heads / cfg.kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn masked_softmax(
    q: &[f32],
    k: &[f32],
    batch: &TokenBatch,
    row: usize,
    time_index: usize,
    query_head: usize,
    key_head: usize,
    dims: &GqaDims,
) -> (Vec<usize>, Vec<f32>) {
    let query_base = (row * dims.q_heads + query_head) * dims.head_dim;
    let query = &q[query_base..query_base + dims.head_dim];
    let batch_start = row - time_index;
    let mut keys = Vec::new();
    let mut scores = Vec::new();
    for key_row in batch_start..=row {
        if !batch.valid[key_row] || batch.segments[key_row] != batch.segments[row] {
            continue;
        }
        let key_base = (key_row * dims.kv_heads + key_head) * dims.head_dim;
        let score: f32 = query
            .iter()
            .zip(&k[key_base..key_base + dims.head_dim])
            .map(|(&left, &right)| left * right)
            .sum();
        keys.push(key_row);
        scores.push(score * dims.scale);
    }
    if scores.is_empty() {
        return (keys, scores);
    }
    let max = scores
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &score| acc.max(score));
    let mut total = 0.0f32;
    for score in &mut scores {
        *score = (*score - max).exp();
        total += *score;
    }
    if total > 0.0 && total.is_finite() {
        for score in &mut scores {
            *score /= total;
        }
    }
    (keys, scores)
}

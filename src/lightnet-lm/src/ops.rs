//! Differentiable CPU primitives for the decoder.
//!
//! All operations are plain f32 slices in row-major layout.  A row is
//! `b * time + t`; within a row the hidden axis is contiguous.  Every backward
//! function is the exact adjoint of the forward in this module, with no extra
//! sign flip at the optimizer boundary: cross-entropy returns the descent
//! direction `one_hot(target) - softmax(logits)`.

use crate::{Error, LossSummary, Result};

/// Root-mean-square layer norm without bias.
///
/// `x` is `rows * gamma.len()`; `gamma` is the per-channel scale.
pub fn rms_norm(x: &[f32], gamma: &[f32], rows: usize, eps: f32) -> Result<Vec<f32>> {
    let (hidden, inv) = rms_inverse(x, rows, eps)?;
    if gamma.len() != hidden {
        return Err(Error::InvalidConfig(format!(
            "rms_norm gamma has length {}, expected {hidden}",
            gamma.len()
        )));
    }
    let mut out = vec![0.0; x.len()];
    for (row, &inv_row) in inv.iter().enumerate() {
        let base = row * hidden;
        for channel in 0..hidden {
            out[base + channel] = x[base + channel] * inv_row * gamma[channel];
        }
    }
    Ok(out)
}

/// Exact adjoint of [`rms_norm`]; returns `(dx, dgamma)`.
pub fn rms_norm_backward(
    x: &[f32],
    gamma: &[f32],
    dy: &[f32],
    rows: usize,
    eps: f32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if dy.len() != x.len() {
        return Err(Error::InvalidConfig(format!(
            "rms_norm dy has length {}, expected {}",
            dy.len(),
            x.len()
        )));
    }
    let (hidden, inv) = rms_inverse(x, rows, eps)?;
    if gamma.len() != hidden {
        return Err(Error::InvalidConfig(format!(
            "rms_norm gamma has length {}, expected {hidden}",
            gamma.len()
        )));
    }

    let mut dx = vec![0.0; x.len()];
    let mut dgamma = vec![0.0; hidden];
    for (row, &inv_row) in inv.iter().enumerate() {
        let base = row * hidden;
        let inv_cubed = inv_row * inv_row * inv_row;
        // c = sum_j gamma_j * x_j * dy_j
        let mut c = 0.0f32;
        for channel in 0..hidden {
            c += gamma[channel] * x[base + channel] * dy[base + channel];
        }
        let scale = inv_cubed / hidden as f32;
        for channel in 0..hidden {
            let x_value = x[base + channel];
            let dy_value = dy[base + channel];
            dx[base + channel] = gamma[channel] * inv_row * dy_value - x_value * scale * c;
            dgamma[channel] += x_value * inv_row * dy_value;
        }
    }
    Ok((dx, dgamma))
}

fn rms_inverse(x: &[f32], rows: usize, eps: f32) -> Result<(usize, Vec<f32>)> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::InvalidConfig(
            "rms_norm epsilon must be finite and positive".to_owned(),
        ));
    }
    if rows == 0 {
        return Err(Error::InvalidConfig(
            "rms_norm needs at least one row".to_owned(),
        ));
    }
    if !x.len().is_multiple_of(rows) {
        return Err(Error::InvalidConfig(format!(
            "rms_norm input length {} is not divisible by {rows} rows",
            x.len()
        )));
    }
    let hidden = x.len() / rows;
    if hidden == 0 {
        return Err(Error::InvalidConfig(
            "rms_norm hidden size must be positive".to_owned(),
        ));
    }
    let mut inv = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * hidden;
        let mut sum_squares = 0.0f64;
        for channel in 0..hidden {
            let value = x[base + channel];
            sum_squares += f64::from(value) * f64::from(value);
        }
        let mean_square = sum_squares / hidden as f64;
        inv.push((1.0 / (mean_square + f64::from(eps)).sqrt()) as f32);
    }
    Ok((hidden, inv))
}

/// Gather rows from a `[vocab, hidden]` embedding table.
pub fn embedding(table: &[f32], ids: &[u32], hidden: usize) -> Result<Vec<f32>> {
    let vocab = embedding_vocab(table, hidden)?;
    let mut out = vec![0.0; checked_product(ids.len(), hidden, "embedding output")?];
    for (row, &id) in ids.iter().enumerate() {
        if id as usize >= vocab {
            return Err(Error::InvalidConfig(format!(
                "embedding id {id} is outside vocabulary ({vocab})"
            )));
        }
        let source = id as usize * hidden;
        let target = row * hidden;
        out[target..target + hidden].copy_from_slice(&table[source..source + hidden]);
    }
    Ok(out)
}

/// Scatter-add the embedding adjoint.  Repeated IDs accumulate every row.
pub fn embedding_backward(
    ids: &[u32],
    dy: &[f32],
    vocab: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    if vocab == 0 || hidden == 0 {
        return Err(Error::InvalidConfig(
            "embedding vocabulary and hidden size must be positive".to_owned(),
        ));
    }
    let expected = checked_product(ids.len(), hidden, "embedding dy")?;
    if dy.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "embedding dy has length {}, expected {expected}",
            dy.len()
        )));
    }
    let mut dtable = vec![0.0; checked_product(vocab, hidden, "embedding table")?];
    for (row, &id) in ids.iter().enumerate() {
        if id as usize >= vocab {
            return Err(Error::InvalidConfig(format!(
                "embedding id {id} is outside vocabulary ({vocab})"
            )));
        }
        let target = id as usize * hidden;
        let source = row * hidden;
        for channel in 0..hidden {
            dtable[target + channel] += dy[source + channel];
        }
    }
    Ok(dtable)
}

fn embedding_vocab(table: &[f32], hidden: usize) -> Result<usize> {
    if hidden == 0 {
        return Err(Error::InvalidConfig(
            "embedding hidden size must be positive".to_owned(),
        ));
    }
    if !table.len().is_multiple_of(hidden) {
        return Err(Error::InvalidConfig(format!(
            "embedding table length {} is not divisible by hidden {hidden}",
            table.len()
        )));
    }
    let vocab = table.len() / hidden;
    if vocab == 0 {
        return Err(Error::InvalidConfig(
            "embedding vocabulary must be positive".to_owned(),
        ));
    }
    Ok(vocab)
}

/// Split-half rotary position embedding, in place.
///
/// `x` is `rows * heads * head_dim`; `positions` has one absolute position per
/// row.  Pair `i` rotates channel `i` against channel `i + head_dim / 2` by
/// `position * theta^(-2i/head_dim)`.  `inverse` applies the transpose rotation.
pub fn rope_in_place(
    x: &mut [f32],
    positions: &[u32],
    heads: usize,
    head_dim: usize,
    theta: f32,
    inverse: bool,
) -> Result<()> {
    if heads == 0 || head_dim == 0 {
        return Err(Error::InvalidConfig(
            "rope heads and head dimension must be positive".to_owned(),
        ));
    }
    if !head_dim.is_multiple_of(2) {
        return Err(Error::InvalidConfig(format!(
            "rope head dimension {head_dim} must be even"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::InvalidConfig(
            "rope theta must be finite and positive".to_owned(),
        ));
    }
    let hidden = checked_product(heads, head_dim, "rope hidden size")?;
    let expected = checked_product(positions.len(), hidden, "rope input")?;
    if x.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "rope input has length {}, expected {expected}",
            x.len()
        )));
    }

    let half = head_dim / 2;
    for (row, &position) in positions.iter().enumerate() {
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for pair in 0..half {
                let exponent = -2.0 * pair as f64 / head_dim as f64;
                let angle = f64::from(position) * f64::from(theta).powf(exponent);
                let sin = angle.sin() as f32;
                let cos = angle.cos() as f32;
                let first = x[base + pair];
                let second = x[base + pair + half];
                let (rotated_first, rotated_second) = if inverse {
                    (first * cos + second * sin, -first * sin + second * cos)
                } else {
                    (first * cos - second * sin, first * sin + second * cos)
                };
                x[base + pair] = rotated_first;
                x[base + pair + half] = rotated_second;
            }
        }
    }
    Ok(())
}

/// SwiGLU activation: `silu(gate) * up`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
    if gate.len() != up.len() {
        return Err(Error::InvalidConfig(format!(
            "swiglu gate length {} does not match up length {}",
            gate.len(),
            up.len()
        )));
    }
    Ok(gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| silu(gate) * up)
        .collect())
}

/// Exact adjoint of [`swiglu`]; returns `(dgate, dup)`.
pub fn swiglu_backward(gate: &[f32], up: &[f32], dy: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    if gate.len() != up.len() || gate.len() != dy.len() {
        return Err(Error::InvalidConfig(format!(
            "swiglu lengths differ: gate {}, up {}, dy {}",
            gate.len(),
            up.len(),
            dy.len()
        )));
    }
    let mut dgate = vec![0.0; gate.len()];
    let mut dup = vec![0.0; gate.len()];
    for index in 0..gate.len() {
        let gate_value = gate[index];
        let sigmoid = sigmoid(gate_value);
        let silu = gate_value * sigmoid;
        let d_silu = sigmoid * (1.0 + gate_value * (1.0 - sigmoid));
        dgate[index] = dy[index] * up[index] * d_silu;
        dup[index] = dy[index] * silu;
    }
    Ok((dgate, dup))
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

/// Sparse next-token cross-entropy over `[rows, vocab]` logits.
///
/// `targets[row] == -100` ignores the row.  Returns the summed loss and the
/// unnormalized descent seeds `one_hot(target) - softmax(logits)`.  The stable
/// log-sum-exp keeps wide logits finite; active rows with non-finite logits are
/// errors and ignored rows contribute nothing.
pub fn sparse_ce(logits: &[f32], targets: &[i32], vocab: usize) -> Result<(LossSummary, Vec<f32>)> {
    if vocab == 0 {
        return Err(Error::InvalidConfig(
            "cross-entropy vocabulary must be positive".to_owned(),
        ));
    }
    if !logits.len().is_multiple_of(vocab) {
        return Err(Error::InvalidConfig(format!(
            "logit length {} is not divisible by vocabulary {vocab}",
            logits.len()
        )));
    }
    let rows = logits.len() / vocab;
    if targets.len() != rows {
        return Err(Error::InvalidConfig(format!(
            "target count {} does not match {rows} logit rows",
            targets.len()
        )));
    }

    let mut seeds = vec![0.0; logits.len()];
    let mut sum = 0.0f64;
    let mut active_tokens = 0u64;
    for (row, &target) in targets.iter().enumerate() {
        if target == -100 {
            continue;
        }
        if target < 0 || target as usize >= vocab {
            return Err(Error::InvalidConfig(format!(
                "target {target} at row {row} is outside vocabulary ({vocab})"
            )));
        }
        if active_tokens == u64::MAX {
            return Err(Error::InvalidConfig(
                "cross-entropy active-token count overflows u64".to_owned(),
            ));
        }
        let base = row * vocab;
        let row_logits = &logits[base..base + vocab];
        if row_logits.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidConfig(format!(
                "row {row} has non-finite logits"
            )));
        }
        let max = row_logits
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &value| acc.max(value));
        let mut exp_sum = 0.0f64;
        for &value in row_logits {
            exp_sum += f64::from((value - max).exp());
        }
        let log_sum_exp = f64::from(max) + exp_sum.ln();
        let target_logit = f64::from(row_logits[target as usize]);
        sum += log_sum_exp - target_logit;
        active_tokens += 1;

        let inverse_sum = 1.0 / exp_sum;
        for (class, &value) in row_logits.iter().enumerate() {
            let probability = f64::from((value - max).exp()) * inverse_sum;
            let one_hot = if class == target as usize { 1.0 } else { 0.0 };
            seeds[base + class] = (one_hot - probability) as f32;
        }
    }

    Ok((LossSummary { sum, active_tokens }, seeds))
}

fn checked_product(left: usize, right: usize, what: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::InvalidConfig(format!("{what} size overflows usize")))
}

//! Host-reference operations for the WMHF wavelet, scan, split and gate primitives.
//!
//! They back the default bodies of the `wmhf_*` ops in
//! [`crate::F32TensorOpsBackend`] and follow the layer's CPU reference
//! (`lightnet-core` `layers/wmhf/cpu.rs`): NCHW tensors, Haar sub-bands of
//! `ceil(H/2) x ceil(W/2)` with odd edges replicated, and the four-direction
//! EMA scan with `a = 0.98 * sigmoid(raw_a)` over `[4, C]` weights.

fn checked_len(dims: &[usize], what: &str) -> Result<usize, String> {
    if dims.contains(&0) {
        return Err(format!("wmhf {what} dimensions must be positive"));
    }
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .ok_or_else(|| format!("wmhf {what} element count overflows"))
    })
}

fn expect_len(values: &[f32], expected: usize, what: &str) -> Result<(), String> {
    if values.len() != expected {
        return Err(format!(
            "wmhf {what} length mismatch: expected {expected}, got {}",
            values.len()
        ));
    }
    Ok(())
}

/// The four input pixels `(y0, y1, x0, x1)` of Haar tile `(y, x)`; odd edges
/// repeat the last row or column.
fn tile(y: usize, x: usize, height: usize, width: usize) -> (usize, usize, usize, usize) {
    let y0 = (2 * y).min(height - 1);
    let y1 = (y0 + 1).min(height - 1);
    let x0 = (2 * x).min(width - 1);
    let x1 = (x0 + 1).min(width - 1);
    (y0, y1, x0, x1)
}

/// Haar analysis: `(LL, LH, HL, HH)`, each `[batch, channels, ceil(H/2), ceil(W/2)]`.
pub fn haar_dwt(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<[Vec<f32>; 4], String> {
    expect_len(
        input,
        checked_len(&[batch, channels, height, width], "DWT input")?,
        "DWT input",
    )?;
    let (height2, width2) = (height.div_ceil(2), width.div_ceil(2));
    let len = batch * channels * height2 * width2;
    let mut bands = [vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]];
    for plane in 0..batch * channels {
        let base = plane * height * width;
        for y in 0..height2 {
            for x in 0..width2 {
                let (y0, y1, x0, x1) = tile(y, x, height, width);
                let v00 = input[base + y0 * width + x0];
                let v01 = input[base + y0 * width + x1];
                let v10 = input[base + y1 * width + x0];
                let v11 = input[base + y1 * width + x1];
                let o = plane * height2 * width2 + y * width2 + x;
                bands[0][o] = 0.5 * (v00 + v01 + v10 + v11);
                bands[1][o] = 0.5 * (v00 - v01 + v10 - v11);
                bands[2][o] = 0.5 * (v00 + v01 - v10 - v11);
                bands[3][o] = 0.5 * (v00 - v01 - v10 + v11);
            }
        }
    }
    Ok(bands)
}

/// Exact adjoint of [`haar_dwt`]; it is also the Haar synthesis (IWT), which
/// accumulates the replicated odd-edge corners.
#[allow(clippy::too_many_arguments)]
pub fn haar_dwt_backward(
    ll: &[f32],
    lh: &[f32],
    hl: &[f32],
    hh: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let output_len = checked_len(&[batch, channels, height, width], "IWT output")?;
    let (height2, width2) = (height.div_ceil(2), width.div_ceil(2));
    let coeff_len = batch * channels * height2 * width2;
    expect_len(ll, coeff_len, "LL coefficients")?;
    expect_len(lh, coeff_len, "LH coefficients")?;
    expect_len(hl, coeff_len, "HL coefficients")?;
    expect_len(hh, coeff_len, "HH coefficients")?;
    let mut output = vec![0.0; output_len];
    for plane in 0..batch * channels {
        let base = plane * height * width;
        for y in 0..height2 {
            for x in 0..width2 {
                let (y0, y1, x0, x1) = tile(y, x, height, width);
                let o = plane * height2 * width2 + y * width2 + x;
                let (a, b, c, d) = (0.5 * ll[o], 0.5 * lh[o], 0.5 * hl[o], 0.5 * hh[o]);
                output[base + y0 * width + x0] += a + b + c + d;
                output[base + y0 * width + x1] += a - b + c - d;
                output[base + y1 * width + x0] += a + b - c - d;
                output[base + y1 * width + x1] += a - b - c + d;
            }
        }
    }
    Ok(output)
}

fn stable_sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

/// Spatial index sequences of one channel plane in scan order: each row left
/// to right then right to left, then each column top to bottom then bottom
/// to top.
fn scan_sequences(height: usize, width: usize) -> Vec<Vec<usize>> {
    let mut sequences = Vec::with_capacity(2 * (height + width));
    for y in 0..height {
        sequences.push((0..width).map(|x| y * width + x).collect());
        sequences.push((0..width).rev().map(|x| y * width + x).collect());
    }
    for x in 0..width {
        sequences.push((0..height).map(|y| y * width + x).collect());
        sequences.push((0..height).rev().map(|y| y * width + x).collect());
    }
    sequences
}

fn validate_scan(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<usize, String> {
    let len = checked_len(&[batch, channels, height, width], "scan input")?;
    expect_len(input, len, "scan input")?;
    expect_len(weights, 4 * channels, "scan weights")?;
    Ok(len)
}

/// Four-direction EMA scan: every sequence adds `0.25 * (c * state + d * x)`.
pub fn scan4_forward(
    input: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let len = validate_scan(input, weights, batch, channels, height, width)?;
    let sequences = scan_sequences(height, width);
    let mut output = vec![0.0; len];
    for plane in 0..batch * channels {
        let channel = plane % channels;
        let a = 0.98 * stable_sigmoid(weights[channel]);
        let b = weights[channels + channel];
        let c = weights[2 * channels + channel];
        let d = weights[3 * channels + channel];
        let base = plane * height * width;
        for sequence in &sequences {
            let mut state = 0.0;
            for &spatial in sequence {
                let value = input[base + spatial];
                state = a * state + b * value;
                output[base + spatial] += 0.25 * (c * state + d * value);
            }
        }
    }
    Ok(output)
}

/// Exact `(input, [4, C] weight)` adjoint of [`scan4_forward`].
pub fn scan4_backward(
    input: &[f32],
    output_grad: &[f32],
    weights: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let len = validate_scan(input, weights, batch, channels, height, width)?;
    expect_len(output_grad, len, "scan output gradient")?;
    let sequences = scan_sequences(height, width);
    let mut input_grad = vec![0.0; len];
    let mut weight_grad = vec![0.0; 4 * channels];
    let mut states = Vec::with_capacity(height.max(width));
    for plane in 0..batch * channels {
        let channel = plane % channels;
        let sigmoid = stable_sigmoid(weights[channel]);
        let a = 0.98 * sigmoid;
        let b = weights[channels + channel];
        let c = weights[2 * channels + channel];
        let d = weights[3 * channels + channel];
        let base = plane * height * width;
        for sequence in &sequences {
            states.clear();
            let mut state = 0.0;
            for &spatial in sequence {
                state = a * state + b * input[base + spatial];
                states.push(state);
            }
            let (mut da, mut db, mut dc, mut dd, mut ds_next) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for t in (0..sequence.len()).rev() {
                let index = base + sequence[t];
                let dy = 0.25 * output_grad[index];
                let value = input[index];
                let state_prev = if t == 0 { 0.0 } else { states[t - 1] };
                dc += dy * states[t];
                dd += dy * value;
                let ds = ds_next + dy * c;
                da += ds * state_prev;
                db += ds * value;
                input_grad[index] += dy * d + ds * b;
                ds_next = ds * a;
            }
            weight_grad[channel] += da * 0.98 * sigmoid * (1.0 - sigmoid);
            weight_grad[channels + channel] += db;
            weight_grad[2 * channels + channel] += dc;
            weight_grad[3 * channels + channel] += dd;
        }
    }
    Ok((input_grad, weight_grad))
}

/// `|LH| + |HL| + |HH|` nearest-upsampled to `[batch, channels, H, W]`.
pub fn high_frequency_forward(
    lh: &[f32],
    hl: &[f32],
    hh: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let output_len = checked_len(&[batch, channels, height, width], "high-frequency energy")?;
    let (height2, width2) = (height.div_ceil(2), width.div_ceil(2));
    let coeff_len = batch * channels * height2 * width2;
    expect_len(lh, coeff_len, "LH coefficients")?;
    expect_len(hl, coeff_len, "HL coefficients")?;
    expect_len(hh, coeff_len, "HH coefficients")?;
    let mut output = vec![0.0; output_len];
    for plane in 0..batch * channels {
        for y in 0..height {
            for x in 0..width {
                let coeff = plane * height2 * width2 + (y / 2) * width2 + x / 2;
                output[(plane * height + y) * width + x] =
                    lh[coeff].abs() + hl[coeff].abs() + hh[coeff].abs();
            }
        }
    }
    Ok(output)
}

fn sign_no_zero(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// `(dLH, dHL, dHH)` of [`high_frequency_forward`]; `sign(0) = 0`.
#[allow(clippy::too_many_arguments)]
pub fn high_frequency_backward(
    output_grad: &[f32],
    lh: &[f32],
    hl: &[f32],
    hh: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<[Vec<f32>; 3], String> {
    let output_len = checked_len(&[batch, channels, height, width], "high-frequency gradient")?;
    expect_len(output_grad, output_len, "high-frequency gradient")?;
    let (height2, width2) = (height.div_ceil(2), width.div_ceil(2));
    let coeff_len = batch * channels * height2 * width2;
    expect_len(lh, coeff_len, "LH coefficients")?;
    expect_len(hl, coeff_len, "HL coefficients")?;
    expect_len(hh, coeff_len, "HH coefficients")?;
    let mut grads = [vec![0.0; coeff_len], vec![0.0; coeff_len], vec![0.0; coeff_len]];
    for plane in 0..batch * channels {
        for y in 0..height {
            for x in 0..width {
                let coeff = plane * height2 * width2 + (y / 2) * width2 + x / 2;
                let upstream = output_grad[(plane * height + y) * width + x];
                grads[0][coeff] += upstream * sign_no_zero(lh[coeff]);
                grads[1][coeff] += upstream * sign_no_zero(hl[coeff]);
                grads[2][coeff] += upstream * sign_no_zero(hh[coeff]);
            }
        }
    }
    Ok(grads)
}

/// Channels `[begin, begin + count)` of an NCHW tensor.
pub fn split_channels(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    begin: usize,
    count: usize,
) -> Result<Vec<f32>, String> {
    expect_len(
        input,
        checked_len(&[batch, channels, height, width], "channel split input")?,
        "channel split input",
    )?;
    let output_len = checked_len(&[batch, count, height, width], "channel split output")?;
    if begin.checked_add(count).is_none_or(|end| end > channels) {
        return Err("wmhf channel split range exceeds the input channels".to_owned());
    }
    let area = height * width;
    let mut output = Vec::with_capacity(output_len);
    for b in 0..batch {
        let start = (b * channels + begin) * area;
        output.extend_from_slice(&input[start..start + count * area]);
    }
    Ok(output)
}

/// Channel-major `[3x | 5x | 7x]` concatenation of the local depthwise branches.
pub fn concat_local(
    branch3: &[f32],
    branch5: &[f32],
    branch7: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    let branch_len = checked_len(&[batch, channels, height, width], "local branch")?;
    expect_len(branch3, branch_len, "3x local branch")?;
    expect_len(branch5, branch_len, "5x local branch")?;
    expect_len(branch7, branch_len, "7x local branch")?;
    let per_batch = channels * height * width;
    let mut output = Vec::with_capacity(3 * branch_len);
    for b in 0..batch {
        let range = b * per_batch..(b + 1) * per_batch;
        output.extend_from_slice(&branch3[range.clone()]);
        output.extend_from_slice(&branch5[range.clone()]);
        output.extend_from_slice(&branch7[range]);
    }
    Ok(output)
}

/// `fuse + freq_scale * gate * projected (+ shortcut)`.
pub fn apply_gate(
    fuse: &[f32],
    gate: &[f32],
    projected: &[f32],
    shortcut: Option<&[f32]>,
    freq_scale: f32,
) -> Result<Vec<f32>, String> {
    expect_len(gate, fuse.len(), "gate values")?;
    expect_len(projected, fuse.len(), "projected values")?;
    if let Some(shortcut) = shortcut {
        expect_len(shortcut, fuse.len(), "shortcut values")?;
    }
    Ok((0..fuse.len())
        .map(|i| fuse[i] + freq_scale * gate[i] * projected[i] + shortcut.map_or(0.0, |s| s[i]))
        .collect())
}

/// `(d_fuse, d_gate, d_projected, d_shortcut)` of [`apply_gate`]; the
/// shortcut gradient is zero when the shortcut is unused.
pub fn apply_gate_backward(
    output_grad: &[f32],
    gate: &[f32],
    projected: &[f32],
    freq_scale: f32,
    use_shortcut: bool,
) -> Result<[Vec<f32>; 4], String> {
    expect_len(gate, output_grad.len(), "gate values")?;
    expect_len(projected, output_grad.len(), "projected values")?;
    let gate_grad = (0..output_grad.len())
        .map(|i| freq_scale * projected[i] * output_grad[i])
        .collect();
    let projected_grad = (0..output_grad.len())
        .map(|i| freq_scale * gate[i] * output_grad[i])
        .collect();
    let shortcut_grad = if use_shortcut {
        output_grad.to_vec()
    } else {
        vec![0.0; output_grad.len()]
    };
    Ok([output_grad.to_vec(), gate_grad, projected_grad, shortcut_grad])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    fn pattern(len: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i * 7 % 19) as f32 - 8.0) * scale + offset)
            .collect()
    }

    #[test]
    fn wmhf_host_haar_pair_is_adjoint_on_odd_shapes() {
        let (batch, channels, height, width) = (2, 2, 3, 5);
        let input = pattern(batch * channels * height * width, 0.17, -0.31);
        let bands = haar_dwt(&input, batch, channels, height, width).unwrap();
        let seeds = [
            pattern(bands[0].len(), 0.11, 0.2),
            pattern(bands[0].len(), -0.13, -0.1),
            pattern(bands[0].len(), 0.07, 0.4),
            pattern(bands[0].len(), 0.19, -0.2),
        ];
        let adjoint = haar_dwt_backward(
            &seeds[0], &seeds[1], &seeds[2], &seeds[3], batch, channels, height, width,
        )
        .unwrap();
        let lhs: f32 = bands.iter().zip(&seeds).map(|(b, s)| dot(b, s)).sum();
        assert!((lhs - dot(&input, &adjoint)).abs() < 2.0e-5);
    }

    #[test]
    fn wmhf_host_scan_matches_hand_computation_and_finite_differences() {
        let output = scan4_forward(&[1.0, 2.0], &[0.0, 2.0, 3.0, 0.25], 1, 1, 1, 2).unwrap();
        assert!((output[0] - 7.72).abs() < 2.0e-6 && (output[1] - 13.235).abs() < 2.0e-6);

        let (batch, channels, height, width) = (1, 2, 2, 3);
        let input = pattern(batch * channels * height * width, 0.13, -0.22);
        let weights = pattern(4 * channels, 0.09, 0.31);
        let upstream = pattern(input.len(), -0.07, 0.1);
        let (input_grad, weight_grad) =
            scan4_backward(&input, &upstream, &weights, batch, channels, height, width).unwrap();
        let loss = |x: &[f32], w: &[f32]| {
            dot(&scan4_forward(x, w, batch, channels, height, width).unwrap(), &upstream)
        };
        let eps = 1.0e-3;
        for i in 0..input.len() {
            let (mut plus, mut minus) = (input.clone(), input.clone());
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&plus, &weights) - loss(&minus, &weights)) / (2.0 * eps);
            assert!((input_grad[i] - fd).abs() < 2.0e-3, "input {i}");
        }
        for i in 0..weights.len() {
            let (mut plus, mut minus) = (weights.clone(), weights.clone());
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&input, &plus) - loss(&input, &minus)) / (2.0 * eps);
            assert!((weight_grad[i] - fd).abs() < 2.0e-3, "weight {i}");
        }
    }
}

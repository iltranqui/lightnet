//! Host reference primitives for the Swin-style windowed transformer.
//!
//! Geometry, mask, attention, and two-stream mHC helpers are kept separate
//! from cfg/runtime integration so their indexing and derivatives can be
//! verified independently. The source-derived CPU contract is in
//! `tasks/rust/swin_plan.md`.

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WindowAttentionCache {
    tokens: usize,
    channels: usize,
    heads: usize,
    head_dim: usize,
    bias_table_len: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_active: Vec<bool>,
    k_active: Vec<bool>,
    probabilities: Vec<f32>,
    score_active: Vec<bool>,
    relative_position_indices: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WindowAttentionGradients {
    /// Gradient in fused `[T,3C]` QKV order.
    pub qkv: Vec<f32>,
    /// Accumulated gradients in `[heads,(2*window-1)^2]` order.
    pub relative_position_bias: Vec<f32>,
}

fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| format!("Swin {what} size overflow"))
    })
}

fn validate_spatial(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Result<(), String> {
    if batch == 0 || channels == 0 || height == 0 || width == 0 {
        return Err(
            "Swin spatial tensors require positive batch, channels, height, and width".into(),
        );
    }
    Ok(())
}

fn padded_extent(extent: usize, window: usize) -> Result<usize, String> {
    if extent == 0 || window == 0 {
        return Err("Swin padding requires positive extents and window size".into());
    }
    let rounded = extent
        .checked_add(window - 1)
        .ok_or("Swin padded extent overflow")?
        / window;
    rounded
        .checked_mul(window)
        .ok_or_else(|| "Swin padded extent overflow".into())
}

pub(crate) fn padded_shape(
    height: usize,
    width: usize,
    window: usize,
) -> Result<(usize, usize), String> {
    Ok((
        padded_extent(height, window)?,
        padded_extent(width, window)?,
    ))
}

/// Pad an NCHW tensor with zeros on the bottom and right, as the C++ layer does.
pub(crate) fn pad_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    padded_height: usize,
    padded_width: usize,
) -> Result<Vec<f32>, String> {
    validate_spatial(batch, channels, height, width)?;
    if padded_height < height || padded_width < width {
        return Err("Swin padded dimensions must cover the input".into());
    }
    let input_len = checked_product(&[batch, channels, height, width], "input")?;
    if input.len() != input_len {
        return Err(format!(
            "Swin input has {} values, expected {input_len}",
            input.len()
        ));
    }
    let output_len = checked_product(
        &[batch, channels, padded_height, padded_width],
        "padded input",
    )?;
    let mut output = vec![0.0; output_len];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                let src = ((b * channels + c) * height + y) * width;
                let dst = ((b * channels + c) * padded_height + y) * padded_width;
                output[dst..dst + width].copy_from_slice(&input[src..src + width]);
            }
        }
    }
    Ok(output)
}

/// Crop padded NCHW storage back to the original bottom/right extent.
pub(crate) fn crop_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    padded_height: usize,
    padded_width: usize,
    height: usize,
    width: usize,
) -> Result<Vec<f32>, String> {
    validate_spatial(batch, channels, height, width)?;
    if padded_height < height || padded_width < width {
        return Err("Swin crop dimensions must fit inside the padded tensor".into());
    }
    let input_len = checked_product(
        &[batch, channels, padded_height, padded_width],
        "crop input",
    )?;
    if input.len() != input_len {
        return Err(format!(
            "Swin crop input has {} values, expected {input_len}",
            input.len()
        ));
    }
    let output_len = checked_product(&[batch, channels, height, width], "crop output")?;
    let mut output = vec![0.0; output_len];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                let src = ((b * channels + c) * padded_height + y) * padded_width;
                let dst = ((b * channels + c) * height + y) * width;
                output[dst..dst + width].copy_from_slice(&input[src..src + width]);
            }
        }
    }
    Ok(output)
}

/// Apply the source cyclic permutation: output[y,x] = input[y-dy,x-dx] with wrap.
pub(crate) fn cyclic_shift_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    dy: isize,
    dx: isize,
) -> Result<Vec<f32>, String> {
    validate_spatial(batch, channels, height, width)?;
    let expected = checked_product(&[batch, channels, height, width], "cyclic shift")?;
    if input.len() != expected {
        return Err(format!(
            "Swin cyclic-shift input has {} values, expected {expected}",
            input.len()
        ));
    }
    let h = isize::try_from(height).map_err(|_| "Swin height exceeds cyclic-shift range")?;
    let w = isize::try_from(width).map_err(|_| "Swin width exceeds cyclic-shift range")?;
    let mut output = vec![0.0; expected];
    for b in 0..batch {
        for c in 0..channels {
            for y in 0..height {
                let source_y = (y as isize - dy).rem_euclid(h) as usize;
                for x in 0..width {
                    let source_x = (x as isize - dx).rem_euclid(w) as usize;
                    let source = ((b * channels + c) * height + source_y) * width + source_x;
                    let destination = ((b * channels + c) * height + y) * width + x;
                    output[destination] = input[source];
                }
            }
        }
    }
    Ok(output)
}

/// Convert padded NCHW spatial input into raster-ordered `[B*nW,T,C]` windows.
pub(crate) fn window_partition_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    padded_height: usize,
    padded_width: usize,
    window: usize,
) -> Result<Vec<f32>, String> {
    validate_spatial(batch, channels, padded_height, padded_width)?;
    if window == 0 || padded_height % window != 0 || padded_width % window != 0 {
        return Err("Swin partition requires positive window-aligned padded dimensions".into());
    }
    let input_len = checked_product(
        &[batch, channels, padded_height, padded_width],
        "partition input",
    )?;
    if input.len() != input_len {
        return Err(format!(
            "Swin partition input has {} values, expected {input_len}",
            input.len()
        ));
    }
    let windows_h = padded_height / window;
    let windows_w = padded_width / window;
    let tokens = checked_product(&[window, window], "window token")?;
    let total_windows = checked_product(&[batch, windows_h, windows_w], "window count")?;
    let output_len = checked_product(&[total_windows, tokens, channels], "partition output")?;
    let mut output = vec![0.0; output_len];
    for b in 0..batch {
        for wh in 0..windows_h {
            for ww in 0..windows_w {
                let window_index = b * windows_h * windows_w + wh * windows_w + ww;
                for iy in 0..window {
                    for ix in 0..window {
                        let token = iy * window + ix;
                        let y = wh * window + iy;
                        let x = ww * window + ix;
                        for c in 0..channels {
                            let source =
                                ((b * channels + c) * padded_height + y) * padded_width + x;
                            let destination = (window_index * tokens + token) * channels + c;
                            output[destination] = input[source];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Inverse of [`window_partition_nchw`], returning padded NCHW storage.
pub(crate) fn window_unpartition_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    padded_height: usize,
    padded_width: usize,
    window: usize,
) -> Result<Vec<f32>, String> {
    validate_spatial(batch, channels, padded_height, padded_width)?;
    if window == 0 || padded_height % window != 0 || padded_width % window != 0 {
        return Err("Swin unpartition requires positive window-aligned padded dimensions".into());
    }
    let windows_h = padded_height / window;
    let windows_w = padded_width / window;
    let tokens = checked_product(&[window, window], "window token")?;
    let total_windows = checked_product(&[batch, windows_h, windows_w], "window count")?;
    let expected = checked_product(&[total_windows, tokens, channels], "unpartition input")?;
    if input.len() != expected {
        return Err(format!(
            "Swin unpartition input has {} values, expected {expected}",
            input.len()
        ));
    }
    let output_len = checked_product(
        &[batch, channels, padded_height, padded_width],
        "unpartition output",
    )?;
    let mut output = vec![0.0; output_len];
    for b in 0..batch {
        for wh in 0..windows_h {
            for ww in 0..windows_w {
                let window_index = b * windows_h * windows_w + wh * windows_w + ww;
                for iy in 0..window {
                    for ix in 0..window {
                        let token = iy * window + ix;
                        let y = wh * window + iy;
                        let x = ww * window + ix;
                        for c in 0..channels {
                            let source = (window_index * tokens + token) * channels + c;
                            let destination =
                                ((b * channels + c) * padded_height + y) * padded_width + x;
                            output[destination] = input[source];
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Build the source mask shared by every batch item. It assigns shifted-window
/// region IDs in the padded grid, then marks bottom/right padding by its grid
/// coordinates, matching `build_attention_mask` in `jepa:swin_layer.cpp`.
pub(crate) fn build_attention_mask(
    height: usize,
    width: usize,
    padded_height: usize,
    padded_width: usize,
    window: usize,
    shifted: bool,
) -> Result<Vec<f32>, String> {
    if height == 0 || width == 0 || window == 0 || padded_height < height || padded_width < width {
        return Err(
            "Swin mask requires positive input/window dimensions and covering padding".into(),
        );
    }
    if padded_height % window != 0 || padded_width % window != 0 {
        return Err("Swin mask requires window-aligned padded dimensions".into());
    }
    let tokens = checked_product(&[window, window], "mask token")?;
    let windows_h = padded_height / window;
    let windows_w = padded_width / window;
    let window_count = checked_product(&[windows_h, windows_w], "mask window count")?;
    let matrix_len = checked_product(&[tokens, tokens], "mask matrix")?;
    let mask_len = checked_product(&[window_count, matrix_len], "attention mask")?;
    let mut mask = vec![0.0; mask_len];
    let shift_size = if shifted { window / 2 } else { 0 };
    let has_padding = padded_height != height || padded_width != width;
    if shift_size == 0 && !has_padding {
        return Ok(mask);
    }

    let grid_len = checked_product(&[padded_height, padded_width], "mask grid")?;
    let mut region_map = vec![0isize; grid_len];
    if shift_size > 0 {
        let h_slices = [0, padded_height - shift_size, padded_height];
        let w_slices = [0, padded_width - shift_size, padded_width];
        let mut region = 0isize;
        for hi in 0..2 {
            for wi in 0..2 {
                for y in h_slices[hi]..h_slices[hi + 1] {
                    for x in w_slices[wi]..w_slices[wi + 1] {
                        region_map[y * padded_width + x] = region;
                    }
                }
                region += 1;
            }
        }
    }
    if has_padding {
        for y in 0..padded_height {
            for x in 0..padded_width {
                if y >= height || x >= width {
                    region_map[y * padded_width + x] = -1;
                }
            }
        }
    }

    for wh in 0..windows_h {
        for ww in 0..windows_w {
            let window_index = wh * windows_w + ww;
            for query in 0..tokens {
                let query_y = wh * window + query / window;
                let query_x = ww * window + query % window;
                let query_region = region_map[query_y * padded_width + query_x];
                for key in 0..tokens {
                    let key_y = wh * window + key / window;
                    let key_x = ww * window + key % window;
                    let key_region = region_map[key_y * padded_width + key_x];
                    if query_region == -1 || key_region == -1 || query_region != key_region {
                        mask[window_index * matrix_len + query * tokens + key] = -100.0;
                    }
                }
            }
        }
    }
    Ok(mask)
}

/// Relative-position lookup indices for an `window × window` token grid.
pub(crate) fn relative_position_indices(window: usize) -> Result<Vec<usize>, String> {
    if window == 0 {
        return Err("Swin relative-position index requires a positive window".into());
    }
    let tokens = checked_product(&[window, window], "relative-position token")?;
    let table_width = window
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or("Swin relative-position table width overflow")?;
    let pair_count = checked_product(&[tokens, tokens], "relative-position index")?;
    let mut indices = vec![0; pair_count];
    for query in 0..tokens {
        let query_y = query / window;
        let query_x = query % window;
        for key in 0..tokens {
            let key_y = key / window;
            let key_x = key % window;
            let dy = query_y as isize - key_y as isize + window as isize - 1;
            let dx = query_x as isize - key_x as isize + window as isize - 1;
            indices[query * tokens + key] = dy as usize * table_width + dx as usize;
        }
    }
    Ok(indices)
}

fn mhc_sigmoid(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.5;
    }
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn mhc_parameters(params: &[f32], site: usize) -> Result<(f32, f32, f32, f32, f32), String> {
    if params.len() != 12 || site >= 2 {
        return Err(format!(
            "Swin mHC expects 12 parameters and site 0 or 1 (got {}, {site})",
            params.len()
        ));
    }
    let offset = site * 6;
    let z = 0.5 * (params[offset] + params[offset + 3] - params[offset + 1] - params[offset + 2]);
    let p = mhc_sigmoid(z);
    let post_skip = 2.0 * mhc_sigmoid(params[offset + 4]);
    let post_branch = 2.0 * mhc_sigmoid(params[offset + 5]);
    let skip = post_skip * p + post_branch * (1.0 - p);
    let branch = post_skip * (1.0 - p) + post_branch * p;
    Ok((skip, branch, p, post_skip, post_branch))
}

/// Apply the source two-stream mHC residual mix without an extra feature clamp.
pub(crate) fn mhc_residual_forward(
    skip: &[f32],
    branch: &[f32],
    params: &[f32],
    site: usize,
) -> Result<Vec<f32>, String> {
    if skip.len() != branch.len() {
        return Err("Swin mHC inputs must have equal lengths".into());
    }
    let (skip_coefficient, branch_coefficient, _, _, _) = mhc_parameters(params, site)?;
    Ok(skip
        .iter()
        .zip(branch)
        .map(|(&skip, &branch)| skip_coefficient * skip + branch_coefficient * branch)
        .collect())
}

/// Exact derivative of [`mhc_residual_forward`]. The returned parameter
/// gradient has 12 entries, with only the selected site's six entries set.
pub(crate) fn mhc_residual_backward(
    skip: &[f32],
    branch: &[f32],
    output_gradient: &[f32],
    params: &[f32],
    site: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if skip.len() != branch.len() || skip.len() != output_gradient.len() {
        return Err("Swin mHC inputs and output gradient must have equal lengths".into());
    }
    let (skip_coefficient, branch_coefficient, p, post_skip, post_branch) =
        mhc_parameters(params, site)?;
    let mut skip_gradient = vec![0.0; skip.len()];
    let mut branch_gradient = vec![0.0; skip.len()];
    let mut parameter_gradient = vec![0.0; 12];
    let mut grad_post_skip = 0.0;
    let mut grad_post_branch = 0.0;
    let mut grad_p = 0.0;
    for i in 0..skip.len() {
        let gradient = output_gradient[i];
        skip_gradient[i] = skip_coefficient * gradient;
        branch_gradient[i] = branch_coefficient * gradient;
        grad_post_skip += gradient * (p * skip[i] + (1.0 - p) * branch[i]);
        grad_post_branch += gradient * ((1.0 - p) * skip[i] + p * branch[i]);
        grad_p += gradient * (post_skip - post_branch) * (skip[i] - branch[i]);
    }
    let offset = site * 6;
    let logits = &params[offset..offset + 4];
    let z = 0.5 * (logits[0] + logits[3] - logits[1] - logits[2]);
    if z.is_finite() {
        let dz = 0.5 * grad_p * p * (1.0 - p);
        parameter_gradient[offset] = dz;
        parameter_gradient[offset + 1] = -dz;
        parameter_gradient[offset + 2] = -dz;
        parameter_gradient[offset + 3] = dz;
    }
    if params[offset + 4].is_finite() {
        parameter_gradient[offset + 4] = grad_post_skip * post_skip * (1.0 - 0.5 * post_skip);
    }
    if params[offset + 5].is_finite() {
        parameter_gradient[offset + 5] = grad_post_branch * post_branch * (1.0 - 0.5 * post_branch);
    }
    Ok((skip_gradient, branch_gradient, parameter_gradient))
}

fn source_constrain(value: f32, limit: f32) -> f32 {
    (-limit).max(value).min(limit)
}

fn source_softmax_row(values: &mut [f32]) {
    let clip = 80.0;
    for value in values.iter_mut() {
        if value.is_nan() {
            *value = 0.0;
        } else if value.is_infinite() {
            *value = if value.is_sign_positive() {
                clip
            } else {
                -clip
            };
        } else {
            *value = value.clamp(-clip, clip);
        }
    }
    let max_value = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max_value).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum <= 0.0 {
        let uniform = 1.0 / values.len() as f32;
        values.fill(uniform);
        return;
    }
    let inverse_sum = 1.0 / (sum + 1.0e-9);
    for value in values {
        *value *= inverse_sum;
    }
}

/// One-window, multi-head attention using the source C++ score clamps, relative
/// bias, additive mask, and stabilized row softmax. `qkv` is `[T,3C]`.
pub(crate) fn window_attention_forward(
    qkv: &[f32],
    window: usize,
    channels: usize,
    heads: usize,
    relative_position_bias: &[f32],
    mask: &[f32],
) -> Result<(Vec<f32>, WindowAttentionCache), String> {
    if window == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
        return Err(
            "Swin attention requires positive dimensions and channels divisible by heads".into(),
        );
    }
    let tokens = checked_product(&[window, window], "attention token")?;
    let expected_qkv = checked_product(&[tokens, channels, 3], "attention QKV")?;
    if qkv.len() != expected_qkv {
        return Err(format!(
            "Swin attention QKV has {} values, expected {expected_qkv}",
            qkv.len()
        ));
    }
    let matrix_len = checked_product(&[tokens, tokens], "attention matrix")?;
    if mask.len() != matrix_len {
        return Err(format!(
            "Swin attention mask has {} values, expected {matrix_len}",
            mask.len()
        ));
    }
    let table_width = window
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or("Swin relative-position table width overflow")?;
    let bias_table_len = checked_product(&[table_width, table_width], "relative-position bias")?;
    let expected_bias = checked_product(&[heads, bias_table_len], "relative-position bias")?;
    if relative_position_bias.len() != expected_bias {
        return Err(format!(
            "Swin relative-position bias has {} values, expected {expected_bias}",
            relative_position_bias.len()
        ));
    }
    let relative_position_indices = relative_position_indices(window)?;
    let head_dim = channels / heads;
    let head_elements = checked_product(&[tokens, channels], "attention head")?;
    let all_head_elements = checked_product(&[heads, head_elements], "attention heads")?;
    let mut q = vec![0.0; all_head_elements];
    let mut k = vec![0.0; all_head_elements];
    let mut v = vec![0.0; all_head_elements];
    let mut q_active = vec![false; all_head_elements];
    let mut k_active = vec![false; all_head_elements];
    let mut probabilities =
        vec![0.0; checked_product(&[heads, matrix_len], "attention probabilities")?];
    let mut score_active = vec![false; probabilities.len()];
    let mut context = vec![0.0; head_elements];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for head in 0..heads {
        let channel_start = head * head_dim;
        let head_offset = head * head_elements;
        let mut scores = vec![0.0; matrix_len];
        for token in 0..tokens {
            let source = token * 3 * channels;
            for d in 0..head_dim {
                let index = head_offset + token * head_dim + d;
                let raw_q = qkv[source + channel_start + d];
                let raw_k = qkv[source + channels + channel_start + d];
                q[index] = source_constrain(raw_q, 256.0);
                k[index] = source_constrain(raw_k, 256.0);
                v[index] = qkv[source + 2 * channels + channel_start + d];
                q_active[index] = raw_q.is_finite() && raw_q > -256.0 && raw_q < 256.0;
                k_active[index] = raw_k.is_finite() && raw_k > -256.0 && raw_k < 256.0;
            }
        }
        for query in 0..tokens {
            for key in 0..tokens {
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot +=
                        q[head_offset + query * head_dim + d] * k[head_offset + key * head_dim + d];
                }
                let index = query * tokens + key;
                let bias_index = head * bias_table_len + relative_position_indices[index];
                let raw_score = (dot * scale + relative_position_bias[bias_index]) + mask[index];
                scores[index] = source_constrain(raw_score, 20.0);
                score_active[head * matrix_len + index] =
                    raw_score.is_finite() && raw_score > -20.0 && raw_score < 20.0;
            }
            source_softmax_row(&mut scores[query * tokens..(query + 1) * tokens]);
        }
        probabilities[head * matrix_len..(head + 1) * matrix_len].copy_from_slice(&scores);
        for query in 0..tokens {
            for key in 0..tokens {
                let probability = scores[query * tokens + key];
                for d in 0..head_dim {
                    context[query * channels + channel_start + d] +=
                        probability * v[head_offset + key * head_dim + d];
                }
            }
        }
    }

    Ok((
        context,
        WindowAttentionCache {
            tokens,
            channels,
            heads,
            head_dim,
            bias_table_len,
            q,
            k,
            v,
            q_active,
            k_active,
            probabilities,
            score_active,
            relative_position_indices,
        },
    ))
}

/// Exact piecewise derivative of [`window_attention_forward`]. Score and Q/K
/// clamp derivatives are zero outside their open intervals.
pub(crate) fn window_attention_backward(
    context_gradient: &[f32],
    cache: &WindowAttentionCache,
) -> Result<WindowAttentionGradients, String> {
    let expected_context = checked_product(&[cache.tokens, cache.channels], "attention gradient")?;
    if context_gradient.len() != expected_context {
        return Err(format!(
            "Swin context gradient has {} values, expected {expected_context}",
            context_gradient.len()
        ));
    }
    let matrix_len = checked_product(&[cache.tokens, cache.tokens], "attention matrix")?;
    let head_elements = checked_product(&[cache.tokens, cache.head_dim], "attention head")?;
    let scale = 1.0 / (cache.head_dim as f32).sqrt();
    let mut dqkv =
        vec![0.0; checked_product(&[cache.tokens, cache.channels, 3], "attention QKV gradient")?];
    let mut relative_bias_gradient = vec![0.0; cache.heads * cache.bias_table_len];
    let mut dv = vec![0.0; cache.heads * head_elements];
    let mut dscore = vec![0.0; cache.heads * matrix_len];

    for head in 0..cache.heads {
        let channel_start = head * cache.head_dim;
        let head_offset = head * head_elements;
        let matrix_offset = head * matrix_len;
        for query in 0..cache.tokens {
            for key in 0..cache.tokens {
                let probability = cache.probabilities[matrix_offset + query * cache.tokens + key];
                let mut probability_gradient = 0.0;
                for d in 0..cache.head_dim {
                    let gradient = context_gradient[query * cache.channels + channel_start + d];
                    let value_index = head_offset + key * cache.head_dim + d;
                    probability_gradient += gradient * cache.v[value_index];
                    dv[value_index] += probability * gradient;
                }
                dscore[matrix_offset + query * cache.tokens + key] = probability_gradient;
            }
        }
        for query in 0..cache.tokens {
            let row_start = matrix_offset + query * cache.tokens;
            let mut dot = 0.0;
            for key in 0..cache.tokens {
                dot += cache.probabilities[row_start + key] * dscore[row_start + key];
            }
            for key in 0..cache.tokens {
                let index = row_start + key;
                dscore[index] = if cache.score_active[index] {
                    cache.probabilities[index] * (dscore[index] - dot)
                } else {
                    0.0
                };
                let bias_index = head * cache.bias_table_len
                    + cache.relative_position_indices[query * cache.tokens + key];
                relative_bias_gradient[bias_index] += dscore[index];
            }
        }

        for query in 0..cache.tokens {
            for d in 0..cache.head_dim {
                let mut gradient = 0.0;
                for key in 0..cache.tokens {
                    gradient += dscore[matrix_offset + query * cache.tokens + key]
                        * cache.k[head_offset + key * cache.head_dim + d];
                }
                let index = head_offset + query * cache.head_dim + d;
                if cache.q_active[index] {
                    dqkv[query * 3 * cache.channels + channel_start + d] = gradient * scale;
                }
            }
        }
        for key in 0..cache.tokens {
            for d in 0..cache.head_dim {
                let mut gradient = 0.0;
                for query in 0..cache.tokens {
                    gradient += dscore[matrix_offset + query * cache.tokens + key]
                        * cache.q[head_offset + query * cache.head_dim + d];
                }
                let index = head_offset + key * cache.head_dim + d;
                if cache.k_active[index] {
                    dqkv[key * 3 * cache.channels + cache.channels + channel_start + d] =
                        gradient * scale;
                }
                dqkv[key * 3 * cache.channels + 2 * cache.channels + channel_start + d] = dv[index];
            }
        }
    }
    Ok(WindowAttentionGradients {
        qkv: dqkv,
        relative_position_bias: relative_bias_gradient,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        build_attention_mask, crop_nchw, cyclic_shift_nchw, mhc_residual_backward,
        mhc_residual_forward, pad_nchw, padded_shape, relative_position_indices,
        window_attention_backward, window_attention_forward, window_partition_nchw,
        window_unpartition_nchw,
    };

    #[test]
    fn padding_partition_and_unpartition_preserve_nchw_order() {
        let input: Vec<f32> = (0..60).map(|value| value as f32).collect();
        let (padded_height, padded_width) = padded_shape(3, 5, 2).unwrap();
        assert_eq!((padded_height, padded_width), (4, 6));
        let padded = pad_nchw(&input, 2, 2, 3, 5, padded_height, padded_width).unwrap();
        assert_eq!(padded.len(), 2 * 2 * 4 * 6);
        for batch in 0..2 {
            for channel in 0..2 {
                for y in 0..4 {
                    for x in 0..6 {
                        let index = ((batch * 2 + channel) * 4 + y) * 6 + x;
                        if y >= 3 || x >= 5 {
                            assert_eq!(padded[index], 0.0);
                        }
                    }
                }
            }
        }
        assert_eq!(crop_nchw(&padded, 2, 2, 4, 6, 3, 5).unwrap(), input);
        let windows = window_partition_nchw(&padded, 2, 2, 4, 6, 2).unwrap();
        let restored = window_unpartition_nchw(&windows, 2, 2, 4, 6, 2).unwrap();
        assert_eq!(restored, padded);
    }

    #[test]
    fn padding_crop_shift_and_partition_obey_adjoint_identities() {
        let x: Vec<f32> = (0..30).map(|value| value as f32 * 0.03 - 0.2).collect();
        let padded_gradient: Vec<f32> = (0..48).map(|value| value as f32 * -0.02 + 0.4).collect();
        let padded = pad_nchw(&x, 1, 2, 3, 5, 4, 6).unwrap();
        let cropped_gradient = crop_nchw(&padded_gradient, 1, 2, 4, 6, 3, 5).unwrap();
        let lhs: f32 = padded
            .iter()
            .zip(&padded_gradient)
            .map(|(left, right)| left * right)
            .sum();
        let rhs: f32 = x
            .iter()
            .zip(&cropped_gradient)
            .map(|(left, right)| left * right)
            .sum();
        assert!((lhs - rhs).abs() < 1.0e-5);

        let spatial: Vec<f32> = (0..48).map(|value| value as f32 * 0.01 - 0.1).collect();
        let spatial_gradient: Vec<f32> = (0..48).map(|value| value as f32 * -0.015 + 0.2).collect();
        let shifted = cyclic_shift_nchw(&spatial, 1, 2, 4, 6, -1, 2).unwrap();
        let inverse_shifted_gradient =
            cyclic_shift_nchw(&spatial_gradient, 1, 2, 4, 6, 1, -2).unwrap();
        let lhs: f32 = shifted
            .iter()
            .zip(&spatial_gradient)
            .map(|(left, right)| left * right)
            .sum();
        let rhs: f32 = spatial
            .iter()
            .zip(&inverse_shifted_gradient)
            .map(|(left, right)| left * right)
            .sum();
        assert!((lhs - rhs).abs() < 1.0e-5);

        let windows = window_partition_nchw(&spatial, 1, 2, 4, 6, 2).unwrap();
        let window_gradient: Vec<f32> = (0..windows.len())
            .map(|value| value as f32 * 0.007 - 0.1)
            .collect();
        let unpartitioned_gradient =
            window_unpartition_nchw(&window_gradient, 1, 2, 4, 6, 2).unwrap();
        let lhs: f32 = windows
            .iter()
            .zip(&window_gradient)
            .map(|(left, right)| left * right)
            .sum();
        let rhs: f32 = spatial
            .iter()
            .zip(&unpartitioned_gradient)
            .map(|(left, right)| left * right)
            .sum();
        assert!((lhs - rhs).abs() < 1.0e-5);
    }

    #[test]
    fn cyclic_shift_is_a_permutation_with_an_exact_inverse() {
        let input: Vec<f32> = (0..30).map(|value| value as f32).collect();
        let shifted = cyclic_shift_nchw(&input, 1, 2, 3, 5, -1, 2).unwrap();
        let restored = cyclic_shift_nchw(&shifted, 1, 2, 3, 5, 1, -2).unwrap();
        assert_eq!(restored, input);
    }

    #[test]
    fn source_masks_cover_shifted_regions_and_padding() {
        let no_shift_padding = build_attention_mask(2, 3, 4, 4, 2, false).unwrap();
        let matrix_len = 16;
        // First window contains only valid pixels.
        assert!(
            no_shift_padding[..matrix_len]
                .iter()
                .all(|&value| value == 0.0)
        );
        // In the top-right window, tokens 1 and 3 are padded columns.
        let top_right = matrix_len;
        assert_eq!(no_shift_padding[top_right + 0 * 4 + 1], -100.0);
        assert_eq!(no_shift_padding[top_right + 1 * 4 + 0], -100.0);
        assert!(
            no_shift_padding[top_right + 1 * 4..top_right + 2 * 4]
                .iter()
                .all(|&value| value == -100.0)
        );

        let shifted = build_attention_mask(4, 4, 4, 4, 2, true).unwrap();
        // Window 1 straddles the source region boundary at x=Wp-shift=3.
        assert_eq!(shifted[matrix_len + 0 * 4 + 1], -100.0);
        assert_eq!(shifted[matrix_len + 1 * 4 + 0], -100.0);
        // The top-left window is entirely in one shift region.
        assert!(shifted[..matrix_len].iter().all(|&value| value == 0.0));

        let combined = build_attention_mask(3, 3, 4, 4, 2, true).unwrap();
        // In the top-right window, token 1 is padded: it is masked as a key,
        // and its complete query row is masked as in the C++ reference.
        assert_eq!(combined[matrix_len], 0.0);
        assert_eq!(combined[matrix_len + 1], -100.0);
        assert!(
            combined[matrix_len + 4..matrix_len + 8]
                .iter()
                .all(|&value| value == -100.0)
        );
    }

    #[test]
    fn relative_position_indices_match_source_coordinate_formula() {
        let indices = relative_position_indices(2).unwrap();
        assert_eq!(indices.len(), 16);
        assert_eq!(indices[0], 4);
        assert_eq!(indices[1], 3);
        assert_eq!(indices[4], 5);
        assert_eq!(indices[5], 4);
        assert!(relative_position_indices(0).is_err());
    }

    #[test]
    fn mhc_residual_backward_matches_finite_differences() {
        let skip = [0.3, -0.7, 0.2];
        let branch = [-0.1, 0.5, 0.8];
        let output_gradient = [0.4, -0.2, 0.7];
        let mut params = crate::vit::MHC_DEFAULTS.to_vec();
        params[0] = 0.6;
        params[1] = -0.2;
        params[2] = -0.5;
        params[3] = 0.9;
        params[4] = 0.2;
        params[5] = -0.3;
        let (skip_gradient, branch_gradient, parameter_gradient) =
            mhc_residual_backward(&skip, &branch, &output_gradient, &params, 0).unwrap();
        let objective = |out: &[f32]| {
            out.iter()
                .zip(output_gradient)
                .map(|(&value, grad)| value * grad)
                .sum::<f32>()
        };
        let epsilon = 1.0e-3;
        for index in 0..params.len() {
            let mut plus = params.clone();
            plus[index] += epsilon;
            let mut minus = params.clone();
            minus[index] -= epsilon;
            let numerical = (objective(&mhc_residual_forward(&skip, &branch, &plus, 0).unwrap())
                - objective(&mhc_residual_forward(&skip, &branch, &minus, 0).unwrap()))
                / (2.0 * epsilon);
            assert!(
                (parameter_gradient[index] - numerical).abs() < 2.0e-4,
                "mHC param {index}: analytic={} numeric={numerical}",
                parameter_gradient[index]
            );
        }
        for index in 0..skip.len() {
            let mut plus = skip;
            plus[index] += epsilon;
            let mut minus = skip;
            minus[index] -= epsilon;
            let numerical = (objective(&mhc_residual_forward(&plus, &branch, &params, 0).unwrap())
                - objective(&mhc_residual_forward(&minus, &branch, &params, 0).unwrap()))
                / (2.0 * epsilon);
            assert!((skip_gradient[index] - numerical).abs() < 2.0e-4);

            let mut plus = branch;
            plus[index] += epsilon;
            let mut minus = branch;
            minus[index] -= epsilon;
            let numerical = (objective(&mhc_residual_forward(&skip, &plus, &params, 0).unwrap())
                - objective(&mhc_residual_forward(&skip, &minus, &params, 0).unwrap()))
                / (2.0 * epsilon);
            assert!((branch_gradient[index] - numerical).abs() < 2.0e-4);
        }
        assert!(mhc_residual_forward(&skip, &branch, &params, 2).is_err());
    }

    #[test]
    fn mhc_nonfinite_sigmoid_inputs_have_zero_sanitizer_derivatives() {
        let skip = [0.4, -0.2];
        let branch = [-0.1, 0.8];
        let gradient = [0.7, -0.3];
        let mut params = crate::vit::MHC_DEFAULTS.to_vec();
        params[0] = f32::INFINITY;
        params[4] = f32::NAN;
        let output = mhc_residual_forward(&skip, &branch, &params, 0).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        let (_, _, parameter_gradient) =
            mhc_residual_backward(&skip, &branch, &gradient, &params, 0).unwrap();
        assert!(parameter_gradient[..4].iter().all(|&value| value == 0.0));
        assert_eq!(parameter_gradient[4], 0.0);
        assert!(parameter_gradient[5].is_finite());
    }

    #[test]
    fn window_attention_matches_uniform_hand_computation() {
        let window = 2;
        let channels = 1;
        let qkv: Vec<f32> = [2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0, 7.0]
            .into_iter()
            .collect();
        let bias = vec![0.0; 9];
        let mask = vec![0.0; 16];
        let (context, _) =
            window_attention_forward(&qkv, window, channels, 1, &bias, &mask).unwrap();
        assert_eq!(context, vec![4.0; 4]);
    }

    #[test]
    fn additive_negative_one_hundred_mask_retains_source_padded_key_leakage() {
        let mut qkv = vec![0.0; 4 * 3];
        for token in 0..4 {
            qkv[token * 3 + 2] = if token % 2 == 0 { 0.0 } else { 1.0e8 };
        }
        let bias = vec![0.0; 9];
        let mask = [0.0, -100.0, 0.0, -100.0].repeat(4);
        let (context, _) = window_attention_forward(&qkv, 2, 1, 1, &bias, &mask).unwrap();
        // The C++ source clamps additive -100 to score -20 rather than -inf.
        assert!(
            context[0] > 0.1,
            "masked key contribution was {}",
            context[0]
        );
    }

    #[test]
    fn window_attention_qkv_and_relative_bias_gradients_match_finite_differences() {
        let window = 2;
        let channels = 2;
        let tokens = window * window;
        let qkv: Vec<f32> = (0..tokens * channels * 3)
            .map(|index| (index as f32 - 10.0) * 0.013)
            .collect();
        let bias: Vec<f32> = (0..9).map(|index| (index as f32 - 4.0) * 0.017).collect();
        let mask = vec![0.0; tokens * tokens];
        let output_gradient: Vec<f32> = (0..tokens * channels)
            .map(|index| (index as f32 - 3.0) * 0.11)
            .collect();
        let (output, cache) =
            window_attention_forward(&qkv, window, channels, 1, &bias, &mask).unwrap();
        let gradients = window_attention_backward(&output_gradient, &cache).unwrap();
        let objective = |qkv: &[f32], bias: &[f32]| {
            let (output, _) =
                window_attention_forward(qkv, window, channels, 1, bias, &mask).unwrap();
            output
                .iter()
                .zip(&output_gradient)
                .map(|(&value, &grad)| value * grad)
                .sum::<f32>()
        };
        let epsilon = 1.0e-3;
        for index in [0usize, 5, 8, 14, 20] {
            let mut plus = qkv.clone();
            plus[index] += epsilon;
            let mut minus = qkv.clone();
            minus[index] -= epsilon;
            let numerical = (objective(&plus, &bias) - objective(&minus, &bias)) / (2.0 * epsilon);
            assert!(
                (gradients.qkv[index] - numerical).abs() < 2.0e-3,
                "QKV {index}: analytic={} numeric={numerical}",
                gradients.qkv[index]
            );
        }
        for index in [0usize, 3, 5, 8] {
            let mut plus = bias.clone();
            plus[index] += epsilon;
            let mut minus = bias.clone();
            minus[index] -= epsilon;
            let numerical = (objective(&qkv, &plus) - objective(&qkv, &minus)) / (2.0 * epsilon);
            assert!(
                (gradients.relative_position_bias[index] - numerical).abs() < 2.0e-3,
                "relative bias {index}: analytic={} numeric={numerical}",
                gradients.relative_position_bias[index]
            );
        }
        assert_eq!(output.len(), tokens * channels);
    }
}

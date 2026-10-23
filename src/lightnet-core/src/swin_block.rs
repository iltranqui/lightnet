//! Full host-reference forward/backward for one Rust Swin block.
//!
//! The audited C++ branch has no Swin weights reader/writer. This module uses
//! the Rust-owned [`crate::SwinWeights`] order and differentiates this Rust
//! forward, not the source backward's clipped-gradient heuristics.

use std::sync::Arc;

use crate::model::Activation;
use crate::swin::{
    SwinWeights, WindowAttentionCache, crop_nchw, cyclic_shift_nchw, mhc_residual_backward,
    mhc_residual_forward, pad_nchw, padded_shape, relative_position_indices,
    window_attention_backward, window_attention_forward_with_indices, window_partition_nchw,
    window_unpartition_nchw,
};
use crate::vit::{
    LayerNormCache, layer_norm_backward, layer_norm_forward, linear_backward, linear_forward,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SwinGeometry {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub filters: usize,
    pub heads: usize,
    pub window: usize,
    pub shift: bool,
    pub ffn_ratio: usize,
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SwinForwardCache {
    geometry: SwinGeometry,
    padded_height: usize,
    padded_width: usize,
    windowed_input: Vec<f32>,
    ln1_output: Vec<f32>,
    ln1: LayerNormCache,
    qkv: Vec<f32>,
    attention: Vec<WindowAttentionCache>,
    attention_output: Vec<f32>,
    first_skip: Vec<f32>,
    projection: Vec<f32>,
    res1: Vec<f32>,
    ln2_output: Vec<f32>,
    ln2: LayerNormCache,
    ffn1_raw: Vec<f32>,
    ffn1_activated: Vec<f32>,
    ffn_output: Vec<f32>,
    output_clamp_active: Vec<bool>,
}

fn checked_product(factors: &[usize], what: &str) -> Result<usize, String> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| format!("Swin {what} size overflow"))
    })
}

fn validate_geometry(input: &[f32], weights: &SwinWeights, g: &SwinGeometry) -> Result<(), String> {
    if g.batch == 0
        || g.channels == 0
        || g.height == 0
        || g.width == 0
        || g.filters == 0
        || g.heads == 0
        || g.window == 0
        || g.ffn_ratio == 0
        || g.channels % g.heads != 0
    {
        return Err(
            "Swin geometry requires positive dimensions and channels divisible by heads".into(),
        );
    }
    let expected_input = checked_product(&[g.batch, g.channels, g.height, g.width], "input")?;
    if input.len() != expected_input {
        return Err(format!(
            "Swin input has {} values, expected {expected_input}",
            input.len()
        ));
    }
    if !weights.matches_geometry(g.channels, g.filters, g.heads, g.window, g.ffn_ratio) {
        return Err("Swin parameter geometry does not match the layer".into());
    }
    weights.to_blob()?;
    Ok(())
}

fn activation_forward(activation: &Activation, x: f32) -> Result<f32, String> {
    use Activation::*;
    Ok(match activation {
        Linear => x,
        Logistic => 1.0 / (1.0 + (-x).exp()),
        Loggy => 2.0 / (1.0 + (-x).exp()) - 1.0,
        Relu => x.max(0.0),
        Leaky | Revleaky => {
            if x > 0.0 {
                x
            } else {
                0.1 * x
            }
        }
        Mish => {
            let softplus = if x > 20.0 {
                x
            } else if x < -20.0 {
                x.exp()
            } else {
                (1.0 + x.exp()).ln()
            };
            x * softplus.tanh()
        }
        Swish => x / (1.0 + (-x).exp()),
        Tanh => x.tanh(),
        Relu6 => x.clamp(0.0, 6.0),
        Ramp => x * (x > 0.0) as i32 as f32 + 0.1 * x,
        Gelu => crate::activation_math::gelu(x),
        Elu => crate::activation_math::elu(x),
        Selu => crate::activation_math::selu(x),
        Relie => crate::activation_math::relie(x),
        Plse => crate::activation_math::plse(x),
        Hardtan => crate::activation_math::hardtan(x),
        Lhtan => crate::activation_math::lhtan(x),
        Stair => crate::activation_math::stair(x),
        HardMish => crate::activation_math::hard_mish(x),
        NormalizeChannels | NormalizeChannelsSoftmax | NormalizeChannelsSoftmaxMaxval => {
            return Err("channel-normalizing activations are unsupported by [swin]".into());
        }
        Other(name) => return Err(format!("unsupported [swin] activation `{name}`")),
    })
}

fn activation_backward(activation: &Activation, x: f32, y: f32) -> Result<f32, String> {
    use Activation::*;
    Ok(match activation {
        Linear => 1.0,
        Logistic => y * (1.0 - y),
        Loggy => 0.5 * (1.0 - y * y),
        Relu => (x > 0.0) as i32 as f32,
        Leaky | Revleaky => {
            if x > 0.0 {
                1.0
            } else {
                0.1
            }
        }
        Mish => {
            let softplus = if x > 20.0 {
                x
            } else if x < -20.0 {
                x.exp()
            } else {
                (1.0 + x.exp()).ln()
            };
            let grad_sp = 1.0 - (-softplus).exp();
            let tanh_sp = softplus.tanh();
            x * (1.0 - tanh_sp * tanh_sp) * grad_sp + tanh_sp
        }
        Swish => {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            sigmoid + x * sigmoid * (1.0 - sigmoid)
        }
        Tanh => 1.0 - y * y,
        Relu6 => (x > 0.0 && x < 6.0) as i32 as f32,
        Ramp => (x > 0.0) as i32 as f32 + 0.1,
        Gelu => crate::activation_math::gelu_grad(x),
        Elu => crate::activation_math::elu_grad(x),
        Selu => crate::activation_math::selu_grad(x),
        Relie => {
            if x > 0.0 {
                1.0
            } else {
                0.01
            }
        }
        Plse => crate::activation_math::plse_grad(x),
        Hardtan => crate::activation_math::hardtan_grad(x),
        Lhtan => crate::activation_math::lhtan_grad(x),
        Stair => crate::activation_math::stair_grad(x),
        HardMish => crate::activation_math::hard_mish_grad(x),
        NormalizeChannels | NormalizeChannelsSoftmax | NormalizeChannelsSoftmaxMaxval => {
            return Err(
                "channel-normalizing activation gradients are unsupported by [swin]".into(),
            );
        }
        Other(name) => return Err(format!("unsupported [swin] activation gradient `{name}`")),
    })
}

fn source_constrain(value: f32, limit: f32) -> f32 {
    if value.is_nan() {
        -limit
    } else {
        value.clamp(-limit, limit)
    }
}

fn add_assign(destination: &mut [f32], source: &[f32]) -> Result<(), String> {
    if destination.len() != source.len() {
        return Err(format!(
            "Swin gradient lengths differ: {} and {}",
            destination.len(),
            source.len()
        ));
    }
    for (destination, source) in destination.iter_mut().zip(source) {
        *destination += source;
    }
    Ok(())
}

/// One Rust Swin block forward. Output uses NCHW and preserves H/W while
/// changing channels from `channels` to `filters`.
pub(crate) fn swin_forward(
    input: &[f32],
    weights: &SwinWeights,
    g: &SwinGeometry,
) -> Result<(Vec<f32>, SwinForwardCache), String> {
    validate_geometry(input, weights, g)?;
    let (padded_height, padded_width) = padded_shape(g.height, g.width, g.window)?;
    let tokens = checked_product(&[g.window, g.window], "window tokens")?;
    let spatial_windows = checked_product(
        &[padded_height / g.window, padded_width / g.window],
        "spatial windows",
    )?;
    let total_windows = checked_product(&[g.batch, spatial_windows], "batch windows")?;
    let rows = checked_product(&[total_windows, tokens], "window rows")?;
    let hidden = checked_product(&[g.filters, g.ffn_ratio], "FFN hidden")?;
    let input_len = checked_product(&[g.batch, g.channels, g.height, g.width], "input")?;
    let output_len = checked_product(&[g.batch, g.filters, g.height, g.width], "output")?;
    let padded_input = pad_nchw(
        input,
        g.batch,
        g.channels,
        g.height,
        g.width,
        padded_height,
        padded_width,
    )?;
    let shifted_input = if g.shift {
        cyclic_shift_nchw(
            &padded_input,
            g.batch,
            g.channels,
            padded_height,
            padded_width,
            -(g.window as isize / 2),
            -(g.window as isize / 2),
        )?
    } else {
        padded_input
    };
    let windowed_input = window_partition_nchw(
        &shifted_input,
        g.batch,
        g.channels,
        padded_height,
        padded_width,
        g.window,
    )?;
    let (ln1_output, ln1) = layer_norm_forward(
        &windowed_input,
        &weights.ln1_gamma,
        &weights.ln1_beta,
        rows,
        g.channels,
        1.0e-5,
    )?;
    let qkv = linear_forward(
        &ln1_output,
        &weights.qkv_weight,
        &weights.qkv_bias,
        rows,
        g.channels,
        3 * g.channels,
    )?;
    let masks = crate::swin::build_attention_mask(
        g.height,
        g.width,
        padded_height,
        padded_width,
        g.window,
        g.shift,
    )?;
    let mask_matrix = checked_product(&[tokens, tokens], "attention mask matrix")?;
    let bias_width = g.window * 2 - 1;
    let bias_table_len = checked_product(&[bias_width, bias_width], "relative bias table")?;
    let expected_bias_len = checked_product(&[g.heads, bias_table_len], "relative bias")?;
    if weights.relative_position_bias.len() != expected_bias_len {
        return Err(format!(
            "Swin relative-position bias has {} values, expected {expected_bias_len}",
            weights.relative_position_bias.len()
        ));
    }
    let relative_indices: Arc<[usize]> = relative_position_indices(g.window)?.into();
    let mut attention = Vec::with_capacity(total_windows);
    let mut attention_output = vec![0.0; checked_product(&[rows, g.channels], "attention output")?];
    for window_index in 0..total_windows {
        let row_start = window_index * tokens;
        let qkv_start = row_start * 3 * g.channels;
        let qkv_end = qkv_start + tokens * 3 * g.channels;
        let spatial_index = window_index % spatial_windows;
        let mask_start = spatial_index * mask_matrix;
        let mask_end = mask_start + mask_matrix;
        let (context, cache) = window_attention_forward_with_indices(
            &qkv[qkv_start..qkv_end],
            g.window,
            g.channels,
            g.heads,
            &weights.relative_position_bias,
            &masks[mask_start..mask_end],
            Arc::clone(&relative_indices),
        )?;
        let output_start = row_start * g.channels;
        attention_output[output_start..output_start + context.len()].copy_from_slice(&context);
        attention.push(cache);
    }
    let projection = linear_forward(
        &attention_output,
        &weights.wo,
        &weights.wo_bias,
        rows,
        g.channels,
        g.filters,
    )?;
    let first_skip = if let Some(res_proj) = &weights.res_proj {
        linear_forward(
            &windowed_input,
            res_proj,
            &vec![0.0; g.filters],
            rows,
            g.channels,
            g.filters,
        )?
    } else {
        windowed_input.clone()
    };
    let res1 = mhc_residual_forward(&first_skip, &projection, &weights.mhc, 0)?;
    let (ln2_output, ln2) = layer_norm_forward(
        &res1,
        &weights.ln2_gamma,
        &weights.ln2_beta,
        rows,
        g.filters,
        1.0e-5,
    )?;
    let ffn1_raw = linear_forward(
        &ln2_output,
        &weights.ffn_w1,
        &weights.ffn_b1,
        rows,
        g.filters,
        hidden,
    )?;
    let ffn1_activated = ffn1_raw
        .iter()
        .map(|&value| activation_forward(&g.activation, value))
        .collect::<Result<Vec<_>, _>>()?;
    let ffn_output = linear_forward(
        &ffn1_activated,
        &weights.ffn_w2,
        &weights.ffn_b2,
        rows,
        hidden,
        g.filters,
    )?;
    let windowed_output = mhc_residual_forward(&res1, &ffn_output, &weights.mhc, 1)?;
    let padded_output = window_unpartition_nchw(
        &windowed_output,
        g.batch,
        g.filters,
        padded_height,
        padded_width,
        g.window,
    )?;
    let unshifted_output = if g.shift {
        cyclic_shift_nchw(
            &padded_output,
            g.batch,
            g.filters,
            padded_height,
            padded_width,
            g.window as isize / 2,
            g.window as isize / 2,
        )?
    } else {
        padded_output
    };
    let cropped_output = crop_nchw(
        &unshifted_output,
        g.batch,
        g.filters,
        padded_height,
        padded_width,
        g.height,
        g.width,
    )?;
    debug_assert_eq!(cropped_output.len(), output_len);
    debug_assert_eq!(input.len(), input_len);
    let mut output_clamp_active = vec![false; output_len];
    let output = cropped_output
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            output_clamp_active[index] = value.is_finite() && value > -100.0 && value < 100.0;
            source_constrain(value, 100.0)
        })
        .collect();
    Ok((
        output,
        SwinForwardCache {
            geometry: g.clone(),
            padded_height,
            padded_width,
            windowed_input,
            ln1_output,
            ln1,
            qkv,
            attention,
            attention_output,
            first_skip,
            projection,
            res1,
            ln2_output,
            ln2,
            ffn1_raw,
            ffn1_activated,
            ffn_output,
            output_clamp_active,
        },
    ))
}

/// Exact derivative of [`swin_forward`], returning parameter gradients and the
/// input gradient. Clamp derivatives are zero outside open forward intervals.
pub(crate) fn swin_backward(
    output_gradient: &[f32],
    weights: &SwinWeights,
    cache: &SwinForwardCache,
) -> Result<(SwinWeights, Vec<f32>), String> {
    let g = &cache.geometry;
    if !weights.matches_geometry(g.channels, g.filters, g.heads, g.window, g.ffn_ratio) {
        return Err("Swin backward parameter geometry does not match its forward cache".into());
    }
    weights.to_blob()?;
    let output_len = checked_product(&[g.batch, g.filters, g.height, g.width], "output gradient")?;
    if output_gradient.len() != output_len || cache.output_clamp_active.len() != output_len {
        return Err(format!(
            "Swin output gradient has {} values, expected {output_len}",
            output_gradient.len()
        ));
    }
    let tokens = checked_product(&[g.window, g.window], "window tokens")?;
    let spatial_windows = checked_product(
        &[
            cache.padded_height / g.window,
            cache.padded_width / g.window,
        ],
        "spatial windows",
    )?;
    let total_windows = checked_product(&[g.batch, spatial_windows], "batch windows")?;
    let rows = checked_product(&[total_windows, tokens], "window rows")?;
    let hidden = checked_product(&[g.filters, g.ffn_ratio], "FFN hidden")?;
    let mut gradients =
        SwinWeights::zeroed_gradients(g.channels, g.filters, g.heads, g.window, g.ffn_ratio)?;

    // Reverse final clamp/crop/shift/unpartition.
    let clamped_gradient: Vec<f32> = output_gradient
        .iter()
        .zip(&cache.output_clamp_active)
        .map(|(&gradient, &active)| if active { gradient } else { 0.0 })
        .collect();
    let padded_gradient = pad_nchw(
        &clamped_gradient,
        g.batch,
        g.filters,
        g.height,
        g.width,
        cache.padded_height,
        cache.padded_width,
    )?;
    let unshifted_gradient = if g.shift {
        cyclic_shift_nchw(
            &padded_gradient,
            g.batch,
            g.filters,
            cache.padded_height,
            cache.padded_width,
            -(g.window as isize / 2),
            -(g.window as isize / 2),
        )?
    } else {
        padded_gradient
    };
    let d_windowed_output = window_partition_nchw(
        &unshifted_gradient,
        g.batch,
        g.filters,
        cache.padded_height,
        cache.padded_width,
        g.window,
    )?;

    // Second mHC residual and FFN.
    let (d_res1_from_skip, d_ffn_output, d_mhc2) = mhc_residual_backward(
        &cache.res1,
        &cache.ffn_output,
        &d_windowed_output,
        &weights.mhc,
        1,
    )?;
    add_assign(&mut gradients.mhc, &d_mhc2)?;
    let (d_ffn1_activated, d_ffn_w2, d_ffn_b2) = linear_backward(
        &cache.ffn1_activated,
        &weights.ffn_w2,
        &d_ffn_output,
        rows,
        hidden,
        g.filters,
    )?;
    gradients.ffn_w2 = d_ffn_w2;
    gradients.ffn_b2 = d_ffn_b2;
    let mut d_ffn1_raw = vec![0.0; d_ffn1_activated.len()];
    for index in 0..d_ffn1_raw.len() {
        d_ffn1_raw[index] = d_ffn1_activated[index]
            * activation_backward(
                &g.activation,
                cache.ffn1_raw[index],
                cache.ffn1_activated[index],
            )?;
    }
    let (d_ln2_output, d_ffn_w1, d_ffn_b1) = linear_backward(
        &cache.ln2_output,
        &weights.ffn_w1,
        &d_ffn1_raw,
        rows,
        g.filters,
        hidden,
    )?;
    gradients.ffn_w1 = d_ffn_w1;
    gradients.ffn_b1 = d_ffn_b1;
    let (d_res1_from_ln2, d_ln2_gamma, d_ln2_beta) =
        layer_norm_backward(&d_ln2_output, &weights.ln2_gamma, &cache.ln2)?;
    gradients.ln2_gamma = d_ln2_gamma;
    gradients.ln2_beta = d_ln2_beta;
    let mut d_res1 = d_res1_from_skip;
    add_assign(&mut d_res1, &d_res1_from_ln2)?;

    // First mHC residual, optional C->N skip projection, and attention output projection.
    let (d_first_skip, d_projection, d_mhc1) = mhc_residual_backward(
        &cache.first_skip,
        &cache.projection,
        &d_res1,
        &weights.mhc,
        0,
    )?;
    add_assign(&mut gradients.mhc, &d_mhc1)?;
    let mut d_windowed_input = if let Some(res_proj) = &weights.res_proj {
        let (d_input, d_res_proj, _) = linear_backward(
            &cache.windowed_input,
            res_proj,
            &d_first_skip,
            rows,
            g.channels,
            g.filters,
        )?;
        gradients.res_proj = Some(d_res_proj);
        d_input
    } else {
        d_first_skip
    };
    let (d_attention_output, d_wo, d_wo_bias) = linear_backward(
        &cache.attention_output,
        &weights.wo,
        &d_projection,
        rows,
        g.channels,
        g.filters,
    )?;
    gradients.wo = d_wo;
    gradients.wo_bias = d_wo_bias;

    // Per-window attention, then fused QKV projection.
    if cache.attention.len() != total_windows {
        return Err(format!(
            "Swin attention cache has {} windows, expected {total_windows}",
            cache.attention.len()
        ));
    }
    let mut d_qkv = vec![0.0; rows * 3 * g.channels];
    for window_index in 0..total_windows {
        let row_start = window_index * tokens;
        let gradient_start = row_start * g.channels;
        let gradient_end = gradient_start + tokens * g.channels;
        let attention_gradient = window_attention_backward(
            &d_attention_output[gradient_start..gradient_end],
            &cache.attention[window_index],
        )?;
        let qkv_start = row_start * 3 * g.channels;
        let qkv_end = qkv_start + tokens * 3 * g.channels;
        d_qkv[qkv_start..qkv_end].copy_from_slice(&attention_gradient.qkv);
        add_assign(
            &mut gradients.relative_position_bias,
            &attention_gradient.relative_position_bias,
        )?;
    }
    let (d_ln1_output, d_qkv_weight, d_qkv_bias) = linear_backward(
        &cache.ln1_output,
        &weights.qkv_weight,
        &d_qkv,
        rows,
        g.channels,
        3 * g.channels,
    )?;
    gradients.qkv_weight = d_qkv_weight;
    gradients.qkv_bias = d_qkv_bias;
    let (d_windowed_from_ln1, d_ln1_gamma, d_ln1_beta) =
        layer_norm_backward(&d_ln1_output, &weights.ln1_gamma, &cache.ln1)?;
    gradients.ln1_gamma = d_ln1_gamma;
    gradients.ln1_beta = d_ln1_beta;
    add_assign(&mut d_windowed_input, &d_windowed_from_ln1)?;

    // Reverse partition, the input cyclic shift, and bottom/right padding.
    let shifted_gradient = window_unpartition_nchw(
        &d_windowed_input,
        g.batch,
        g.channels,
        cache.padded_height,
        cache.padded_width,
        g.window,
    )?;
    let padded_input_gradient = if g.shift {
        cyclic_shift_nchw(
            &shifted_gradient,
            g.batch,
            g.channels,
            cache.padded_height,
            cache.padded_width,
            g.window as isize / 2,
            g.window as isize / 2,
        )?
    } else {
        shifted_gradient
    };
    let input_gradient = crop_nchw(
        &padded_input_gradient,
        g.batch,
        g.channels,
        cache.padded_height,
        cache.padded_width,
        g.height,
        g.width,
    )?;
    Ok((gradients, input_gradient))
}

#[cfg(test)]
mod tests {
    use super::{SwinGeometry, swin_backward, swin_forward};
    use crate::model::Activation;
    use crate::swin::SwinWeights;

    fn fixture() -> (Vec<f32>, SwinWeights, SwinGeometry) {
        let geometry = SwinGeometry {
            batch: 1,
            channels: 2,
            height: 2,
            width: 3,
            filters: 3,
            heads: 1,
            window: 2,
            shift: true,
            ffn_ratio: 2,
            activation: Activation::Gelu,
        };
        let mut weights = SwinWeights::zeroed(2, 3, 1, 2, 2).unwrap();
        let fill = |values: &mut [f32], scale: f32, bias: f32| {
            for (index, value) in values.iter_mut().enumerate() {
                *value = ((index as f32 * 0.37 + bias).sin()) * scale;
            }
        };
        fill(&mut weights.qkv_weight, 0.08, 0.2);
        fill(&mut weights.qkv_bias, 0.09, 0.6);
        fill(&mut weights.wo, 0.07, 0.8);
        fill(&mut weights.wo_bias, 0.08, 1.0);
        fill(&mut weights.ln1_gamma, 0.04, 1.3);
        for value in &mut weights.ln1_gamma {
            *value += 1.0;
        }
        fill(&mut weights.ln1_beta, 0.03, 1.5);
        fill(&mut weights.ln2_gamma, 0.04, 1.7);
        for value in &mut weights.ln2_gamma {
            *value += 1.0;
        }
        fill(&mut weights.ln2_beta, 0.03, 1.9);
        fill(&mut weights.ffn_w1, 0.07, 2.1);
        fill(&mut weights.ffn_b1, 0.06, 2.3);
        fill(&mut weights.ffn_w2, 0.07, 2.5);
        fill(&mut weights.ffn_b2, 0.05, 2.7);
        fill(weights.res_proj.as_mut().unwrap(), 0.06, 2.9);
        fill(&mut weights.relative_position_bias, 0.01, 3.1);
        weights.mhc.copy_from_slice(&[
            0.3, -0.2, 0.1, 0.4, 0.2, -0.1, -0.3, 0.1, 0.25, 0.2, -0.15, 0.3,
        ]);
        let input: Vec<f32> = (0..12)
            .map(|index| ((index as f32 * 0.29 + 0.1).sin()) * 0.7)
            .collect();
        (input, weights, geometry)
    }

    fn loss(input: &[f32], weights: &SwinWeights, geometry: &SwinGeometry, seed: &[f32]) -> f32 {
        let (output, _) = swin_forward(input, weights, geometry).unwrap();
        output.iter().zip(seed).map(|(a, b)| a * b).sum()
    }

    fn near(name: &str, analytic: f32, numeric: f32) {
        let tolerance = 3.0e-3 + 3.0e-2 * analytic.abs().max(numeric.abs());
        assert!(
            (analytic - numeric).abs() <= tolerance,
            "{name}: analytic {analytic}, numeric {numeric}, tolerance {tolerance}"
        );
    }

    #[test]
    fn full_swin_forward_backward_gradients_match_finite_differences() {
        let (input, weights, geometry) = fixture();
        let (output, cache) = swin_forward(&input, &weights, &geometry).unwrap();
        assert_eq!(
            output.len(),
            geometry.batch * geometry.filters * geometry.height * geometry.width
        );
        assert!(output.iter().all(|value| value.is_finite()));
        let seed: Vec<f32> = (0..output.len())
            .map(|index| ((index as f32 * 0.41 + 0.2).cos()) * 0.3)
            .collect();
        let (gradients, input_gradient) = swin_backward(&seed, &weights, &cache).unwrap();
        let epsilon = 1.0e-3;
        let input_index = 4;
        let mut plus_input = input.clone();
        let mut minus_input = input.clone();
        plus_input[input_index] += epsilon;
        minus_input[input_index] -= epsilon;
        near(
            "input",
            input_gradient[input_index],
            (loss(&plus_input, &weights, &geometry, &seed)
                - loss(&minus_input, &weights, &geometry, &seed))
                / (2.0 * epsilon),
        );
        let check_weight = |name: &str,
                            get: fn(&SwinWeights) -> f32,
                            set: fn(&mut SwinWeights, f32),
                            analytic: f32| {
            let original = get(&weights);
            let mut plus = weights.clone();
            let mut minus = weights.clone();
            set(&mut plus, original + epsilon);
            set(&mut minus, original - epsilon);
            near(
                name,
                analytic,
                (loss(&input, &plus, &geometry, &seed) - loss(&input, &minus, &geometry, &seed))
                    / (2.0 * epsilon),
            );
        };
        check_weight(
            "qkv weight",
            |w| w.qkv_weight[0],
            |w, value| w.qkv_weight[0] = value,
            gradients.qkv_weight[0],
        );
        check_weight(
            "qkv bias",
            |w| w.qkv_bias[1],
            |w, value| w.qkv_bias[1] = value,
            gradients.qkv_bias[1],
        );
        check_weight(
            "value QKV bias",
            |w| w.qkv_bias[4],
            |w, value| w.qkv_bias[4] = value,
            gradients.qkv_bias[4],
        );
        check_weight(
            "output projection",
            |w| w.wo[0],
            |w, value| w.wo[0] = value,
            gradients.wo[0],
        );
        check_weight(
            "output projection bias",
            |w| w.wo_bias[0],
            |w, value| w.wo_bias[0] = value,
            gradients.wo_bias[0],
        );
        check_weight(
            "LayerNorm gamma",
            |w| w.ln1_gamma[0],
            |w, value| w.ln1_gamma[0] = value,
            gradients.ln1_gamma[0],
        );
        check_weight(
            "LayerNorm beta",
            |w| w.ln1_beta[0],
            |w, value| w.ln1_beta[0] = value,
            gradients.ln1_beta[0],
        );
        check_weight(
            "LayerNorm 2 gamma",
            |w| w.ln2_gamma[0],
            |w, value| w.ln2_gamma[0] = value,
            gradients.ln2_gamma[0],
        );
        check_weight(
            "LayerNorm 2 beta",
            |w| w.ln2_beta[0],
            |w, value| w.ln2_beta[0] = value,
            gradients.ln2_beta[0],
        );
        check_weight(
            "FFN W1",
            |w| w.ffn_w1[0],
            |w, value| w.ffn_w1[0] = value,
            gradients.ffn_w1[0],
        );
        check_weight(
            "FFN b1",
            |w| w.ffn_b1[0],
            |w, value| w.ffn_b1[0] = value,
            gradients.ffn_b1[0],
        );
        check_weight(
            "FFN W2",
            |w| w.ffn_w2[0],
            |w, value| w.ffn_w2[0] = value,
            gradients.ffn_w2[0],
        );
        check_weight(
            "FFN b2",
            |w| w.ffn_b2[0],
            |w, value| w.ffn_b2[0] = value,
            gradients.ffn_b2[0],
        );
        check_weight(
            "relative position bias",
            |w| w.relative_position_bias[0],
            |w, value| w.relative_position_bias[0] = value,
            gradients.relative_position_bias[0],
        );
        check_weight(
            "mHC scale",
            |w| w.mhc[0],
            |w, value| w.mhc[0] = value,
            gradients.mhc[0],
        );
        check_weight(
            "second mHC site",
            |w| w.mhc[6],
            |w, value| w.mhc[6] = value,
            gradients.mhc[6],
        );
        check_weight(
            "residual projection",
            |w| w.res_proj.as_ref().unwrap()[0],
            |w, value| w.res_proj.as_mut().unwrap()[0] = value,
            gradients.res_proj.as_ref().unwrap()[0],
        );
    }

    #[test]
    fn equal_channels_use_the_identity_skip_with_exact_backward() {
        let geometry = SwinGeometry {
            batch: 1,
            channels: 2,
            height: 2,
            width: 2,
            filters: 2,
            heads: 1,
            window: 2,
            shift: false,
            ffn_ratio: 2,
            activation: Activation::Tanh,
        };
        let mut weights = SwinWeights::zeroed(2, 2, 1, 2, 2).unwrap();
        weights.qkv_weight.fill(0.07);
        weights.wo.fill(0.08);
        weights.wo_bias.fill(0.02);
        weights.ffn_w1.fill(0.06);
        weights.ffn_w2.fill(0.05);
        weights.ffn_b1.fill(0.03);
        weights.ffn_b2.fill(0.02);
        weights.relative_position_bias.fill(0.01);
        weights.mhc.copy_from_slice(&[
            0.2, -0.1, 0.3, 0.1, 0.2, -0.2, -0.1, 0.3, 0.15, -0.2, 0.25, 0.1,
        ]);
        assert!(weights.res_proj.is_none());
        let input = [-0.7, 0.2, 0.4, -0.1, 0.8, 0.3, -0.2, -0.5];
        let (output, cache) = swin_forward(&input, &weights, &geometry).unwrap();
        let seed = [0.2; 8];
        let (gradients, input_gradient) = swin_backward(&seed, &weights, &cache).unwrap();
        let index = 3;
        let epsilon = 1.0e-3;
        let mut plus = input;
        let mut minus = input;
        plus[index] += epsilon;
        minus[index] -= epsilon;
        let numeric = (loss(&plus, &weights, &geometry, &seed)
            - loss(&minus, &weights, &geometry, &seed))
            / (2.0 * epsilon);
        near(
            "equal-channel identity skip input",
            input_gradient[index],
            numeric,
        );
        let mut plus_weights = weights.clone();
        let mut minus_weights = weights.clone();
        plus_weights.ln2_beta[0] += epsilon;
        minus_weights.ln2_beta[0] -= epsilon;
        let numeric = (loss(&input, &plus_weights, &geometry, &seed)
            - loss(&input, &minus_weights, &geometry, &seed))
            / (2.0 * epsilon);
        near("equal-channel LN2 beta", gradients.ln2_beta[0], numeric);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn mish_activation_backward_matches_finite_difference() {
        let (input, weights, mut geometry) = fixture();
        geometry.activation = Activation::Mish;
        let (output, cache) = swin_forward(&input, &weights, &geometry).unwrap();
        let seed = vec![0.17; output.len()];
        let (gradients, _) = swin_backward(&seed, &weights, &cache).unwrap();
        let index = 0;
        let epsilon = 1.0e-3;
        let mut plus = weights.clone();
        let mut minus = weights.clone();
        plus.ffn_b1[index] += epsilon;
        minus.ffn_b1[index] -= epsilon;
        let numeric = (loss(&input, &plus, &geometry, &seed)
            - loss(&input, &minus, &geometry, &seed))
            / (2.0 * epsilon);
        near("Mish activation", gradients.ffn_b1[index], numeric);
    }

    #[test]
    fn final_output_clamp_has_zero_derivative_outside_its_open_interval() {
        let (input, mut weights, geometry) = fixture();
        weights.wo_bias.fill(500.0);
        let (output, cache) = swin_forward(&input, &weights, &geometry).unwrap();
        assert!(output.iter().all(|value| *value == 100.0));
        let seed = vec![1.0; output.len()];
        let (gradients, input_gradient) = swin_backward(&seed, &weights, &cache).unwrap();
        assert!(input_gradient.iter().all(|value| *value == 0.0));
        assert!(gradients.wo_bias.iter().all(|value| *value == 0.0));
    }
}

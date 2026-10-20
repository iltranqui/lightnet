use super::*;

pub(super) fn accumulate_negative_vec(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(Error::InvalidConfig(format!(
            "gradient accumulation length mismatch: expected {}, got {}",
            dst.len(),
            src.len()
        )));
    }
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst -= *src;
    }
    Ok(())
}

pub(super) fn accumulate_vit_updates(
    dst: &mut crate::vit::VitWeights,
    src: &crate::vit::VitWeights,
) -> Result<()> {
    accumulate_negative_vec(&mut dst.patch_embed, &src.patch_embed)?;
    accumulate_negative_vec(&mut dst.patch_bias, &src.patch_bias)?;
    accumulate_negative_vec(&mut dst.qkv_weight, &src.qkv_weight)?;
    accumulate_negative_vec(&mut dst.qkv_bias, &src.qkv_bias)?;
    accumulate_negative_vec(&mut dst.ln1_gamma, &src.ln1_gamma)?;
    accumulate_negative_vec(&mut dst.ln1_beta, &src.ln1_beta)?;
    accumulate_negative_vec(&mut dst.wo, &src.wo)?;
    accumulate_negative_vec(&mut dst.wo_bias, &src.wo_bias)?;
    accumulate_negative_vec(&mut dst.ln2_gamma, &src.ln2_gamma)?;
    accumulate_negative_vec(&mut dst.ln2_beta, &src.ln2_beta)?;
    accumulate_negative_vec(&mut dst.ffn_w1, &src.ffn_w1)?;
    accumulate_negative_vec(&mut dst.ffn_b1, &src.ffn_b1)?;
    accumulate_negative_vec(&mut dst.ffn_w2, &src.ffn_w2)?;
    accumulate_negative_vec(&mut dst.ffn_b2, &src.ffn_b2)?;
    accumulate_negative_vec(&mut dst.pos_embed, &src.pos_embed)?;
    accumulate_negative_vec(&mut dst.mhc, &src.mhc)
}

pub(super) fn accumulate_tucker_attention_updates(
    dst: &mut TuckerAttentionTrainBuffers,
    src: &crate::tucker_attention::TuckerAttentionParams,
) -> Result<()> {
    accumulate_negative_vec(&mut dst.bias_updates, &src.biases)?;
    accumulate_negative_vec(&mut dst.weight_updates, &src.weights)
}

pub(super) fn accumulate_swin_updates(
    dst: &mut crate::swin::SwinWeights,
    src: &crate::swin::SwinWeights,
) -> Result<()> {
    accumulate_negative_vec(&mut dst.qkv_weight, &src.qkv_weight)?;
    accumulate_negative_vec(&mut dst.qkv_bias, &src.qkv_bias)?;
    accumulate_negative_vec(&mut dst.wo, &src.wo)?;
    accumulate_negative_vec(&mut dst.wo_bias, &src.wo_bias)?;
    accumulate_negative_vec(&mut dst.ln1_gamma, &src.ln1_gamma)?;
    accumulate_negative_vec(&mut dst.ln1_beta, &src.ln1_beta)?;
    accumulate_negative_vec(&mut dst.ln2_gamma, &src.ln2_gamma)?;
    accumulate_negative_vec(&mut dst.ln2_beta, &src.ln2_beta)?;
    accumulate_negative_vec(&mut dst.ffn_w1, &src.ffn_w1)?;
    accumulate_negative_vec(&mut dst.ffn_b1, &src.ffn_b1)?;
    accumulate_negative_vec(&mut dst.ffn_w2, &src.ffn_w2)?;
    accumulate_negative_vec(&mut dst.ffn_b2, &src.ffn_b2)?;
    match (&mut dst.res_proj, &src.res_proj) {
        (Some(dst), Some(src)) => accumulate_negative_vec(dst, src)?,
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "Swin residual projection gradient mismatch".into(),
            ));
        }
    }
    accumulate_negative_vec(&mut dst.relative_position_bias, &src.relative_position_bias)?;
    accumulate_negative_vec(&mut dst.mhc, &src.mhc)
}

pub(super) fn scale_swin_updates(updates: &mut crate::swin::SwinWeights, scale: f32) {
    for values in [
        &mut updates.qkv_weight,
        &mut updates.qkv_bias,
        &mut updates.wo,
        &mut updates.wo_bias,
        &mut updates.ln1_gamma,
        &mut updates.ln1_beta,
        &mut updates.ln2_gamma,
        &mut updates.ln2_beta,
        &mut updates.ffn_w1,
        &mut updates.ffn_b1,
        &mut updates.ffn_w2,
        &mut updates.ffn_b2,
        &mut updates.relative_position_bias,
        &mut updates.mhc,
    ] {
        for value in values.iter_mut() {
            *value *= scale;
        }
    }
    if let Some(res_proj) = &mut updates.res_proj {
        for value in res_proj {
            *value *= scale;
        }
    }
}

pub(super) fn sanitize_swin_updates(updates: &mut crate::swin::SwinWeights) {
    for values in [
        &mut updates.qkv_weight,
        &mut updates.qkv_bias,
        &mut updates.wo,
        &mut updates.wo_bias,
        &mut updates.ln1_gamma,
        &mut updates.ln1_beta,
        &mut updates.ln2_gamma,
        &mut updates.ln2_beta,
        &mut updates.ffn_w1,
        &mut updates.ffn_b1,
        &mut updates.ffn_w2,
        &mut updates.ffn_b2,
        &mut updates.relative_position_bias,
        &mut updates.mhc,
    ] {
        for value in values.iter_mut() {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
    }
    if let Some(res_proj) = &mut updates.res_proj {
        for value in res_proj {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
    }
}

pub(super) fn swin_update_clip_scale(updates: &crate::swin::SwinWeights, max_norm: f64) -> f32 {
    let mut norm_sq = 0.0f64;
    let mut arrays: Vec<&[f32]> = vec![
        &updates.qkv_weight,
        &updates.qkv_bias,
        &updates.wo,
        &updates.wo_bias,
        &updates.ln1_gamma,
        &updates.ln1_beta,
        &updates.ln2_gamma,
        &updates.ln2_beta,
        &updates.ffn_w1,
        &updates.ffn_b1,
        &updates.ffn_w2,
        &updates.ffn_b2,
        &updates.relative_position_bias,
        &updates.mhc,
    ];
    if let Some(res_proj) = &updates.res_proj {
        arrays.push(res_proj);
    }
    for values in arrays {
        for &value in values {
            norm_sq += (value as f64) * (value as f64);
        }
    }
    let norm = norm_sq.sqrt();
    if !norm.is_finite() {
        0.0
    } else if norm > max_norm {
        (max_norm / norm) as f32
    } else {
        1.0
    }
}

pub(super) fn update_tucker_attention_params(
    params: &mut crate::tucker_attention::TuckerAttentionParams,
    train: &mut TuckerAttentionTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    sgd_update(
        &mut params.weights,
        &mut train.weight_updates,
        lr,
        momentum,
        decay,
        batch,
    );
    sgd_update(
        &mut params.biases,
        &mut train.bias_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
}

pub(super) fn update_swin_params(
    params: &mut crate::swin::SwinWeights,
    train: &mut SwinTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    sanitize_swin_updates(&mut train.updates);
    let clip_scale = swin_update_clip_scale(&train.updates, 5.0);
    scale_swin_updates(&mut train.updates, clip_scale);
    for (weights, updates) in [
        (&mut params.qkv_weight, &mut train.updates.qkv_weight),
        (&mut params.wo, &mut train.updates.wo),
        (&mut params.ffn_w1, &mut train.updates.ffn_w1),
        (&mut params.ffn_w2, &mut train.updates.ffn_w2),
    ] {
        sgd_update(weights, updates, lr, momentum, decay, batch);
    }
    if let (Some(weights), Some(updates)) = (&mut params.res_proj, &mut train.updates.res_proj) {
        sgd_update(weights, updates, lr, momentum, decay, batch);
    }
    for (weights, updates) in [
        (&mut params.qkv_bias, &mut train.updates.qkv_bias),
        (&mut params.wo_bias, &mut train.updates.wo_bias),
        (&mut params.ln1_gamma, &mut train.updates.ln1_gamma),
        (&mut params.ln1_beta, &mut train.updates.ln1_beta),
        (&mut params.ln2_gamma, &mut train.updates.ln2_gamma),
        (&mut params.ln2_beta, &mut train.updates.ln2_beta),
        (&mut params.ffn_b1, &mut train.updates.ffn_b1),
        (&mut params.ffn_b2, &mut train.updates.ffn_b2),
        (
            &mut params.relative_position_bias,
            &mut train.updates.relative_position_bias,
        ),
        (&mut params.mhc, &mut train.updates.mhc),
    ] {
        sgd_update(weights, updates, lr, momentum, 0.0, batch);
    }
    for value in &mut params.mhc {
        *value = if value.is_finite() {
            value.clamp(-8.0, 8.0)
        } else {
            0.0
        };
    }
}

pub(super) fn scale_vit_updates(updates: &mut crate::vit::VitWeights, scale: f32) {
    for values in [
        &mut updates.patch_embed,
        &mut updates.patch_bias,
        &mut updates.qkv_weight,
        &mut updates.qkv_bias,
        &mut updates.ln1_gamma,
        &mut updates.ln1_beta,
        &mut updates.wo,
        &mut updates.wo_bias,
        &mut updates.ln2_gamma,
        &mut updates.ln2_beta,
        &mut updates.ffn_w1,
        &mut updates.ffn_b1,
        &mut updates.ffn_w2,
        &mut updates.ffn_b2,
        &mut updates.pos_embed,
        &mut updates.mhc,
    ] {
        for value in values.iter_mut() {
            *value *= scale;
        }
    }
}

pub(super) fn sanitize_vit_updates(updates: &mut crate::vit::VitWeights, learned_pos_embed: bool) {
    let mut arrays = vec![
        &mut updates.patch_embed,
        &mut updates.patch_bias,
        &mut updates.qkv_weight,
        &mut updates.qkv_bias,
        &mut updates.ln1_gamma,
        &mut updates.ln1_beta,
        &mut updates.wo,
        &mut updates.wo_bias,
        &mut updates.ln2_gamma,
        &mut updates.ln2_beta,
        &mut updates.ffn_w1,
        &mut updates.ffn_b1,
        &mut updates.ffn_w2,
        &mut updates.ffn_b2,
        &mut updates.mhc,
    ];
    if learned_pos_embed {
        arrays.push(&mut updates.pos_embed);
    }
    for values in arrays {
        for value in values.iter_mut() {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
    }
}

pub(super) fn vit_update_clip_scale(
    updates: &crate::vit::VitWeights,
    limit: f32,
    learned_pos_embed: bool,
) -> f32 {
    let mut arrays: Vec<&[f32]> = vec![
        &updates.patch_embed,
        &updates.patch_bias,
        &updates.qkv_weight,
        &updates.qkv_bias,
        &updates.ln1_gamma,
        &updates.ln1_beta,
        &updates.wo,
        &updates.wo_bias,
        &updates.ln2_gamma,
        &updates.ln2_beta,
        &updates.ffn_w1,
        &updates.ffn_b1,
        &updates.ffn_w2,
        &updates.ffn_b2,
        &updates.mhc,
    ];
    if learned_pos_embed {
        arrays.push(&updates.pos_embed);
    }
    let norm_sq = arrays
        .into_iter()
        .flatten()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>();
    let norm = norm_sq.sqrt() as f32;
    if norm > limit && norm.is_finite() {
        limit / norm
    } else if !norm.is_finite() {
        0.0
    } else {
        1.0
    }
}

pub(super) fn accumulate_vec(dst: &mut [f32], src: &[f32]) -> Result<()> {
    if dst.len() != src.len() {
        return Err(Error::InvalidConfig(format!(
            "gradient accumulation length mismatch: expected {}, got {}",
            dst.len(),
            src.len()
        )));
    }
    for (dst, src) in dst.iter_mut().zip(src) {
        *dst += *src;
    }
    Ok(())
}

pub(super) fn sgd_update(
    weights: &mut [f32],
    updates: &mut [f32],
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    // This is Darknet's update rule, not the convention used by optimizers
    // that receive a positive dL/dw tensor.  `updates` is accumulated from
    // Darknet deltas, so it already points downhill and is added to weights.
    let batch = batch.max(1) as f32;
    let step = lr / batch;
    for (weight, update) in weights.iter_mut().zip(updates.iter_mut()) {
        *update += -decay * batch * *weight;
        *weight += step * *update;
        *update *= momentum;
    }
}

pub(super) fn update_dpt_head_sgd(
    spec: &DptHeadSpec,
    input_shape: TensorShape,
    params: &mut crate::dpt_head::DptHeadWeights,
    train: &mut DptHeadTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    let Ok(layout) = crate::dpt_head::dpt_head_weight_layout(spec.geometry(input_shape)) else {
        return;
    };
    if params.values.len() != layout.total || train.weight_updates.len() != layout.total {
        return;
    }
    for &index in &crate::dpt_head::DPT_MUON_WEIGHT_SLOTS {
        let range = layout.get(index);
        let end = range.start + range.len;
        sgd_update(
            &mut params.values[range.start..end],
            &mut train.weight_updates[range.start..end],
            lr,
            momentum,
            decay,
            batch,
        );
    }
    for &index in &crate::dpt_head::DPT_BIAS_SLOTS {
        let range = layout.get(index);
        let end = range.start + range.len;
        sgd_update(
            &mut params.values[range.start..end],
            &mut train.weight_updates[range.start..end],
            lr,
            momentum,
            0.0,
            batch,
        );
    }
}

pub(super) fn update_vit_params(
    params: &mut crate::vit::VitWeights,
    train: &mut VitTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
    learned_pos_embed: bool,
) {
    sanitize_vit_updates(&mut train.updates, learned_pos_embed);
    let scale = vit_update_clip_scale(&train.updates, 5.0, learned_pos_embed);
    scale_vit_updates(&mut train.updates, scale);
    for (weights, updates) in [
        (&mut params.patch_embed, &mut train.updates.patch_embed),
        (&mut params.qkv_weight, &mut train.updates.qkv_weight),
        (&mut params.wo, &mut train.updates.wo),
        (&mut params.ffn_w1, &mut train.updates.ffn_w1),
        (&mut params.ffn_w2, &mut train.updates.ffn_w2),
    ] {
        sgd_update(weights, updates, lr, momentum, decay, batch);
    }
    for (weights, updates) in [
        (&mut params.patch_bias, &mut train.updates.patch_bias),
        (&mut params.qkv_bias, &mut train.updates.qkv_bias),
        (&mut params.ln1_gamma, &mut train.updates.ln1_gamma),
        (&mut params.ln1_beta, &mut train.updates.ln1_beta),
        (&mut params.wo_bias, &mut train.updates.wo_bias),
        (&mut params.ln2_gamma, &mut train.updates.ln2_gamma),
        (&mut params.ln2_beta, &mut train.updates.ln2_beta),
        (&mut params.ffn_b1, &mut train.updates.ffn_b1),
        (&mut params.ffn_b2, &mut train.updates.ffn_b2),
    ] {
        sgd_update(weights, updates, lr, momentum, 0.0, batch);
    }
    if learned_pos_embed {
        sgd_update(
            &mut params.pos_embed,
            &mut train.updates.pos_embed,
            lr,
            momentum,
            decay,
            batch,
        );
    }
    sgd_update(
        &mut params.mhc,
        &mut train.updates.mhc,
        lr * 0.1,
        momentum,
        0.0,
        batch,
    );
    for value in &mut params.mhc {
        *value = if value.is_finite() {
            value.clamp(-8.0, 8.0)
        } else {
            0.0
        };
    }
}

pub(super) fn update_batchnorm_params(
    params: &mut BatchNormParams,
    train: &mut StandaloneBatchNormTrainBuffers,
    lr: f32,
    momentum: f32,
    batch: usize,
) {
    sgd_update(
        &mut params.biases,
        &mut train.bias_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    sgd_update(
        &mut params.scales,
        &mut train.scale_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
}

/// One Adam step on a host-resident parameter tensor, zeroing the accumulated
/// update like the device kernel (`adam_update_gpu` clears `weight_updates`).
pub(super) fn apply_host_adam(
    weights: &mut [f32],
    updates: &mut [f32],
    moments: &mut Option<(Vec<f32>, Vec<f32>)>,
    params: &AdamParams,
) -> Result<()> {
    if weights.len() != updates.len() {
        return Err(Error::InvalidConfig(
            "host Adam weight/update length mismatch".to_owned(),
        ));
    }
    let (m, v) = moments
        .take()
        .unwrap_or_else(|| (vec![0.0; weights.len()], vec![0.0; weights.len()]));
    if m.len() != weights.len() || v.len() != weights.len() {
        return Err(Error::InvalidConfig(
            "host Adam moment length mismatch".to_owned(),
        ));
    }
    let (updated, m2, v2) = darknet_cuda::host_ops::adam_update(weights, updates, &m, &v, params);
    weights.copy_from_slice(&updated);
    for value in updates.iter_mut() {
        *value = 0.0;
    }
    *moments = Some((m2, v2));
    Ok(())
}

pub(super) fn apply_host_adam_non_weight(
    weights: &mut [f32],
    updates: &mut [f32],
    moments: &mut Option<(Vec<f32>, Vec<f32>)>,
    adam: &AdamParams,
) -> Result<()> {
    let mut params = *adam;
    if params.decoupled_decay {
        params.decay = 0.0;
    }
    apply_host_adam(weights, updates, moments, &params)
}

pub(super) fn host_adam_child(state: &mut HostAdamLayerState, index: usize) -> &mut HostAdamLayerState {
    if state.children.len() <= index {
        state
            .children
            .resize_with(index + 1, HostAdamLayerState::default);
    }
    &mut state.children[index]
}

pub(super) fn apply_host_adam_connected(
    params: &mut ConnectedParams,
    train: &mut ConnectedTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
) -> Result<()> {
    apply_host_adam(
        &mut params.weights,
        &mut train.weight_updates,
        &mut state.weights,
        adam,
    )?;
    apply_host_adam_non_weight(
        &mut params.biases,
        &mut train.bias_updates,
        &mut state.biases,
        adam,
    )?;
    match (&mut params.batch_norm, &mut train.batch_norm) {
        (Some(params_bn), Some(train_bn)) => apply_host_adam_non_weight(
            &mut params_bn.scales,
            &mut train_bn.scale_updates,
            &mut state.scales,
            adam,
        )?,
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "connected Adam batchnorm state mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn apply_host_adam_convolutional(
    params: &mut ConvolutionalParams,
    train: &mut ConvolutionalTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
) -> Result<()> {
    apply_host_adam(
        &mut params.weights,
        &mut train.weight_updates,
        &mut state.weights,
        adam,
    )?;
    apply_host_adam_non_weight(
        &mut params.biases,
        &mut train.bias_updates,
        &mut state.biases,
        adam,
    )?;
    match (&mut params.batch_norm, &mut train.batch_norm) {
        (Some(params_bn), Some(train_bn)) => apply_host_adam_non_weight(
            &mut params_bn.scales,
            &mut train_bn.scale_updates,
            &mut state.scales,
            adam,
        )?,
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "convolutional Adam batchnorm state mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn apply_host_adam_vit(
    params: &mut crate::vit::VitWeights,
    train: &mut VitTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
    learned_pos_embed: bool,
) -> Result<()> {
    sanitize_vit_updates(&mut train.updates, learned_pos_embed);
    let clip_scale = vit_update_clip_scale(&train.updates, 5.0, learned_pos_embed);
    scale_vit_updates(&mut train.updates, clip_scale);
    let tensors: [(usize, &mut [f32], &mut [f32], bool); 16] = [
        (
            0,
            &mut params.patch_embed,
            &mut train.updates.patch_embed,
            true,
        ),
        (
            1,
            &mut params.patch_bias,
            &mut train.updates.patch_bias,
            false,
        ),
        (2, &mut params.qkv_bias, &mut train.updates.qkv_bias, false),
        (
            3,
            &mut params.ln1_gamma,
            &mut train.updates.ln1_gamma,
            false,
        ),
        (4, &mut params.ln1_beta, &mut train.updates.ln1_beta, false),
        (
            5,
            &mut params.qkv_weight,
            &mut train.updates.qkv_weight,
            true,
        ),
        (6, &mut params.wo, &mut train.updates.wo, true),
        (7, &mut params.wo_bias, &mut train.updates.wo_bias, false),
        (
            8,
            &mut params.ln2_gamma,
            &mut train.updates.ln2_gamma,
            false,
        ),
        (9, &mut params.ln2_beta, &mut train.updates.ln2_beta, false),
        (10, &mut params.ffn_w1, &mut train.updates.ffn_w1, true),
        (11, &mut params.ffn_b1, &mut train.updates.ffn_b1, false),
        (12, &mut params.ffn_w2, &mut train.updates.ffn_w2, true),
        (13, &mut params.ffn_b2, &mut train.updates.ffn_b2, false),
        (
            14,
            &mut params.pos_embed,
            &mut train.updates.pos_embed,
            true,
        ),
        (15, &mut params.mhc, &mut train.updates.mhc, false),
    ];
    for (index, weights, updates, decay) in tensors {
        if index == 14 && !learned_pos_embed {
            continue;
        }
        let moments = host_adam_child(state, index);
        if index == 15 {
            let mhc_adam = AdamParams {
                lr: adam.lr * 0.1,
                decay: 0.0,
                ..*adam
            };
            apply_host_adam(weights, updates, &mut moments.weights, &mhc_adam)?;
        } else if decay {
            apply_host_adam(weights, updates, &mut moments.weights, adam)?;
        } else {
            apply_host_adam_non_weight(weights, updates, &mut moments.weights, adam)?;
        }
    }
    for value in &mut params.mhc {
        *value = if value.is_finite() {
            value.clamp(-8.0, 8.0)
        } else {
            0.0
        };
    }
    Ok(())
}

pub(super) fn apply_host_adam_swin_tensor(
    weights: &mut [f32],
    updates: &mut [f32],
    state: &mut HostAdamLayerState,
    slot: usize,
    decay: bool,
    adam: &AdamParams,
) -> Result<()> {
    let moments = &mut host_adam_child(state, slot).weights;
    if decay {
        apply_host_adam(weights, updates, moments, adam)
    } else {
        apply_host_adam_non_weight(weights, updates, moments, adam)
    }
}

pub(super) fn apply_host_adam_swin(
    params: &mut crate::swin::SwinWeights,
    train: &mut SwinTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
) -> Result<()> {
    sanitize_swin_updates(&mut train.updates);
    let clip_scale = swin_update_clip_scale(&train.updates, 5.0);
    scale_swin_updates(&mut train.updates, clip_scale);
    apply_host_adam_swin_tensor(
        &mut params.qkv_weight,
        &mut train.updates.qkv_weight,
        state,
        0,
        true,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.qkv_bias,
        &mut train.updates.qkv_bias,
        state,
        1,
        false,
        adam,
    )?;
    apply_host_adam_swin_tensor(&mut params.wo, &mut train.updates.wo, state, 2, true, adam)?;
    apply_host_adam_swin_tensor(
        &mut params.wo_bias,
        &mut train.updates.wo_bias,
        state,
        3,
        false,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ln1_gamma,
        &mut train.updates.ln1_gamma,
        state,
        4,
        false,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ln1_beta,
        &mut train.updates.ln1_beta,
        state,
        5,
        false,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ln2_gamma,
        &mut train.updates.ln2_gamma,
        state,
        6,
        false,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ln2_beta,
        &mut train.updates.ln2_beta,
        state,
        7,
        false,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ffn_w1,
        &mut train.updates.ffn_w1,
        state,
        8,
        true,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ffn_b1,
        &mut train.updates.ffn_b1,
        state,
        9,
        false,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ffn_w2,
        &mut train.updates.ffn_w2,
        state,
        10,
        true,
        adam,
    )?;
    apply_host_adam_swin_tensor(
        &mut params.ffn_b2,
        &mut train.updates.ffn_b2,
        state,
        11,
        false,
        adam,
    )?;
    let mut slot = 12;
    if let (Some(weights), Some(updates)) = (&mut params.res_proj, &mut train.updates.res_proj) {
        apply_host_adam_swin_tensor(weights, updates, state, slot, true, adam)?;
        slot += 1;
    }
    apply_host_adam_swin_tensor(
        &mut params.relative_position_bias,
        &mut train.updates.relative_position_bias,
        state,
        slot,
        false,
        adam,
    )?;
    slot += 1;
    apply_host_adam_swin_tensor(
        &mut params.mhc,
        &mut train.updates.mhc,
        state,
        slot,
        false,
        adam,
    )?;
    for value in &mut params.mhc {
        *value = if value.is_finite() {
            value.clamp(-8.0, 8.0)
        } else {
            0.0
        };
    }
    Ok(())
}

pub(super) fn prepare_muon_layer_gradients(
    kind: &LayerKind,
    options: crate::model::LayerTrainOptions,
    params: &LayerParams,
    train: &mut LayerTrainBuffers,
) -> Result<()> {
    if options.grad_centr
        && let (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train)) =
            (params, &mut *train)
        && params.batch_norm.is_some()
    {
        let filter_size = params.weights.len() / params.biases.len().max(1);
        if filter_size > 0 {
            train.weight_updates =
                darknet_cuda::host_ops::gradient_centralization(&train.weight_updates, filter_size);
        }
    }
    match (kind, train) {
        (LayerKind::Vit(spec), LayerTrainBuffers::Vit(train)) => {
            sanitize_vit_updates(&mut train.updates, spec.learned_pos_embed);
            let scale = vit_update_clip_scale(&train.updates, 5.0, spec.learned_pos_embed);
            scale_vit_updates(&mut train.updates, scale);
        }
        (LayerKind::Swin(_), LayerTrainBuffers::Swin(train)) => {
            sanitize_swin_updates(&mut train.updates);
            let scale = swin_update_clip_scale(&train.updates, 5.0);
            scale_swin_updates(&mut train.updates, scale);
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn apply_host_adam_mambavision(
    params: &mut crate::mambavision::MambaVisionParams,
    train: &mut MambaVisionTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
) -> Result<()> {
    let blocks =
        crate::mambavision::mambavision_param_blocks(params).map_err(Error::InvalidConfig)?;
    for (slot, block) in blocks.into_iter().enumerate() {
        let block_adam = AdamParams {
            decay: if block.decay { adam.decay } else { 0.0 },
            ..*adam
        };
        let child = host_adam_child(state, slot);
        apply_host_adam(
            &mut params.values[block.range.clone()],
            &mut train.updates[block.range],
            &mut child.weights,
            &block_adam,
        )?;
    }
    Ok(())
}

pub(super) fn apply_host_adam_dpt(
    spec: &DptHeadSpec,
    input_shape: TensorShape,
    params: &mut crate::dpt_head::DptHeadWeights,
    train: &mut DptHeadTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
    include_matrices: bool,
) -> Result<()> {
    let layout = crate::dpt_head::dpt_head_weight_layout(spec.geometry(input_shape))?;
    if params.values.len() != layout.total || train.weight_updates.len() != layout.total {
        return Err(Error::InvalidConfig(format!(
            "[dpt_head] Adam parameter/update lengths are {}/{}, expected {}",
            params.values.len(),
            train.weight_updates.len(),
            layout.total
        )));
    }
    if include_matrices {
        for &index in &crate::dpt_head::DPT_MUON_WEIGHT_SLOTS {
            let range = layout.get(index);
            let end = range.start + range.len;
            let child = host_adam_child(state, index);
            apply_host_adam(
                &mut params.values[range.start..end],
                &mut train.weight_updates[range.start..end],
                &mut child.weights,
                adam,
            )?;
        }
    } else if adam.decoupled_decay {
        // Muon replaces the Adam direction for matrices, not AdamW's
        // decoupled weight decay. Apply the same multiplicative term as
        // `adam_update` without creating or advancing matrix Adam moments.
        let factor = 1.0 - adam.lr * adam.decay;
        for &index in &crate::dpt_head::DPT_MUON_WEIGHT_SLOTS {
            let range = layout.get(index);
            let end = range.start + range.len;
            for weight in &mut params.values[range.start..end] {
                *weight *= factor;
            }
        }
    }
    for &index in &crate::dpt_head::DPT_BIAS_SLOTS {
        let range = layout.get(index);
        let end = range.start + range.len;
        let child = host_adam_child(state, index);
        apply_host_adam_non_weight(
            &mut params.values[range.start..end],
            &mut train.weight_updates[range.start..end],
            &mut child.biases,
            adam,
        )?;
    }
    Ok(())
}

pub(super) fn apply_host_adam_layer(
    kind: &LayerKind,
    input_shape: TensorShape,
    params: &mut LayerParams,
    train: &mut LayerTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
) -> Result<()> {
    match (params, train) {
        (
            LayerParams::HyperbolicFoveation(params),
            LayerTrainBuffers::HyperbolicFoveation(train),
        ) => {
            let LayerKind::HyperbolicFoveation(spec) = kind else {
                return Err(Error::InvalidConfig(
                    "hyperbolic-foveation parameters do not match layer kind during Adam update"
                        .to_owned(),
                ));
            };
            validate_hyperbolic_foveation_param_layout(spec, params)?;
            if spec.pred_hidden > 0 {
                let first = host_adam_child(state, 0);
                apply_host_adam(
                    &mut params.pred_w1,
                    &mut train.pred_w1_updates,
                    &mut first.weights,
                    adam,
                )?;
                apply_host_adam_non_weight(
                    &mut params.pred_b1,
                    &mut train.pred_b1_updates,
                    &mut first.biases,
                    adam,
                )?;
                let second = host_adam_child(state, 1);
                apply_host_adam(
                    &mut params.pred_w2,
                    &mut train.pred_w2_updates,
                    &mut second.weights,
                    adam,
                )?;
                apply_host_adam_non_weight(
                    &mut params.pred_b2,
                    &mut train.pred_b2_updates,
                    &mut second.biases,
                    adam,
                )?;
            } else {
                apply_host_adam_non_weight(
                    &mut params.hf_params,
                    &mut train.hf_param_updates,
                    &mut state.biases,
                    adam,
                )?;
            }
            clamp_hyperbolic_foveation_params(spec, params);
        }
        (LayerParams::FoveationFilm(params), LayerTrainBuffers::FoveationFilm(train)) => {
            let LayerKind::FoveationFilm(spec) = kind else {
                return Err(Error::InvalidConfig(
                    "[foveation_film] parameters do not match layer kind during Adam update"
                        .to_owned(),
                ));
            };
            validate_foveation_film_param_layout(spec, params.b_gamma.len(), params)?;
            apply_host_adam(
                &mut params.w_embed,
                &mut train.w_embed_updates,
                &mut state.weights,
                adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.b_embed,
                &mut train.b_embed_updates,
                &mut state.biases,
                adam,
            )?;
            let gamma = host_adam_child(state, 0);
            apply_host_adam(
                &mut params.w_gamma,
                &mut train.w_gamma_updates,
                &mut gamma.weights,
                adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.b_gamma,
                &mut train.b_gamma_updates,
                &mut gamma.biases,
                adam,
            )?;
            let beta = host_adam_child(state, 1);
            apply_host_adam(
                &mut params.w_beta,
                &mut train.w_beta_updates,
                &mut beta.weights,
                adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.b_beta,
                &mut train.b_beta_updates,
                &mut beta.biases,
                adam,
            )?;
        }
        (LayerParams::RecursiveBlock(params), LayerTrainBuffers::RecursiveBlock(train)) => {
            let LayerKind::RecursiveBlock(spec) = kind else {
                return Err(Error::InvalidConfig(
                    "recursive-block parameters do not match layer kind during Adam update"
                        .to_owned(),
                ));
            };
            if spec.body.len() != params.body.len() || spec.body.len() != train.body.len() {
                return Err(Error::InvalidConfig(
                    "recursive-block Adam body parameter/update count mismatch".to_owned(),
                ));
            }
            for (index, ((layer, params), train)) in spec
                .body
                .iter()
                .zip(&mut params.body)
                .zip(&mut train.body)
                .enumerate()
            {
                apply_host_adam_layer(
                    &layer.kind,
                    layer.input_shape,
                    params,
                    train,
                    host_adam_child(state, index),
                    adam,
                )?;
            }
            if spec.ouroboros == 1 {
                apply_host_adam(
                    &mut params.controller_weights,
                    &mut train.controller_weight_updates,
                    &mut state.weights,
                    adam,
                )?;
                apply_host_adam_non_weight(
                    &mut params.controller_biases,
                    &mut train.controller_bias_updates,
                    &mut state.biases,
                    adam,
                )?;
            }
        }
        (LayerParams::BatchNorm(params), LayerTrainBuffers::BatchNorm(train)) => {
            apply_host_adam_non_weight(
                &mut params.scales,
                &mut train.scale_updates,
                &mut state.scales,
                adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.biases,
                &mut train.bias_updates,
                &mut state.biases,
                adam,
            )?;
        }
        (LayerParams::Convolutional(params), LayerTrainBuffers::Convolutional(train)) => {
            apply_host_adam_convolutional(params, train, state, adam)?;
        }
        (LayerParams::Connected(params), LayerTrainBuffers::Connected(train)) => {
            apply_host_adam_connected(params, train, state, adam)?;
        }
        (LayerParams::Local(params), LayerTrainBuffers::Local(train)) => {
            apply_host_adam(
                &mut params.weights,
                &mut train.weight_updates,
                &mut state.weights,
                adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.biases,
                &mut train.bias_updates,
                &mut state.biases,
                adam,
            )?;
        }
        (LayerParams::Shortcut(params), LayerTrainBuffers::Shortcut(train)) => {
            apply_host_adam(
                &mut params.weights,
                &mut train.weight_updates,
                &mut state.weights,
                adam,
            )?;
        }
        (LayerParams::Gru(params), LayerTrainBuffers::Gru(train)) => {
            for (slot, params, train) in [
                (0, &mut params.input_z, &mut train.input_z),
                (1, &mut params.state_z, &mut train.state_z),
                (2, &mut params.input_r, &mut train.input_r),
                (3, &mut params.state_r, &mut train.state_r),
                (4, &mut params.input_h, &mut train.input_h),
                (5, &mut params.state_h, &mut train.state_h),
            ] {
                apply_host_adam_connected(params, train, host_adam_child(state, slot), adam)?;
            }
        }
        (LayerParams::Rnn(params), LayerTrainBuffers::Rnn(train)) => {
            for (slot, params, train) in [
                (0, &mut params.input, &mut train.input),
                (1, &mut params.self_params, &mut train.self_params),
                (2, &mut params.output, &mut train.output),
            ] {
                apply_host_adam_connected(params, train, host_adam_child(state, slot), adam)?;
            }
        }
        (LayerParams::Lstm(params), LayerTrainBuffers::Lstm(train)) => {
            for (slot, params, train) in [
                (0, &mut params.wf, &mut train.wf),
                (1, &mut params.wi, &mut train.wi),
                (2, &mut params.wg, &mut train.wg),
                (3, &mut params.wo, &mut train.wo),
                (4, &mut params.uf, &mut train.uf),
                (5, &mut params.ui, &mut train.ui),
                (6, &mut params.ug, &mut train.ug),
                (7, &mut params.uo, &mut train.uo),
            ] {
                apply_host_adam_connected(params, train, host_adam_child(state, slot), adam)?;
            }
        }
        (LayerParams::Crnn(params), LayerTrainBuffers::Crnn(train)) => {
            for (slot, params, train) in [
                (0, &mut params.input, &mut train.input),
                (1, &mut params.self_params, &mut train.self_params),
                (2, &mut params.output, &mut train.output),
            ] {
                apply_host_adam_convolutional(params, train, host_adam_child(state, slot), adam)?;
            }
        }
        (LayerParams::Vit(params), LayerTrainBuffers::Vit(train)) => {
            let learned_pos_embed = matches!(kind, LayerKind::Vit(spec) if spec.learned_pos_embed);
            apply_host_adam_vit(params, train, state, adam, learned_pos_embed)?;
        }
        (LayerParams::Swin(params), LayerTrainBuffers::Swin(train)) => {
            apply_host_adam_swin(params, train, state, adam)?;
        }
        (LayerParams::TuckerAttention(params), LayerTrainBuffers::TuckerAttention(train)) => {
            apply_host_adam(
                &mut params.weights,
                &mut train.weight_updates,
                &mut state.weights,
                adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.biases,
                &mut train.bias_updates,
                &mut state.biases,
                adam,
            )?;
        }
        (LayerParams::MambaVision(params), LayerTrainBuffers::MambaVision(train)) => {
            apply_host_adam_mambavision(params, train, state, adam)?;
        }
        (LayerParams::KimiKda(params), LayerTrainBuffers::KimiKda(train)) => {
            apply_host_adam(
                &mut params.weights,
                &mut train.weight_updates,
                &mut state.weights,
                adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.biases,
                &mut train.bias_updates,
                &mut state.biases,
                adam,
            )?;
        }
        (LayerParams::GraphConv(params), LayerTrainBuffers::GraphConv(train)) => {
            apply_host_adam_convolutional(
                &mut params.projection,
                &mut train.projection,
                state,
                adam,
            )?;
            match (&mut params.self_weights, &mut train.self_weight_updates) {
                (Some(weights), Some(updates)) => apply_host_adam(
                    weights,
                    updates,
                    &mut host_adam_child(state, 0).weights,
                    adam,
                )?,
                (None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "graph_conv Adam self-weight mismatch".to_owned(),
                    ));
                }
            }
            match (
                &mut params.edge_kernel,
                &mut train.edge_kernel_updates,
                &mut params.edge_biases,
                &mut train.edge_bias_updates,
            ) {
                (Some(kernel), Some(kernel_updates), Some(biases), Some(bias_updates)) => {
                    let edge_state = host_adam_child(state, 1);
                    apply_host_adam(kernel, kernel_updates, &mut edge_state.weights, adam)?;
                    apply_host_adam_non_weight(biases, bias_updates, &mut edge_state.biases, adam)?;
                }
                (None, None, None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "graph_conv Adam edge-parameter mismatch".to_owned(),
                    ));
                }
            }
        }
        (LayerParams::DeformConv(params), LayerTrainBuffers::DeformConv(train)) => {
            apply_host_adam_convolutional(&mut params.main, &mut train.main, state, adam)?;
            let predictor_adam = AdamParams {
                lr: adam.lr * 0.01,
                ..*adam
            };
            let offset_state = host_adam_child(state, 0);
            apply_host_adam(
                &mut params.offset_weights,
                &mut train.offset_weight_updates,
                &mut offset_state.weights,
                &predictor_adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.offset_biases,
                &mut train.offset_bias_updates,
                &mut offset_state.biases,
                &predictor_adam,
            )?;
            match (&mut params.mask_weights, &mut train.mask_weight_updates) {
                (Some(weights), Some(updates)) => apply_host_adam(
                    weights,
                    updates,
                    &mut host_adam_child(state, 1).weights,
                    &predictor_adam,
                )?,
                (None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "deform_conv Adam mask weight mismatch".to_owned(),
                    ));
                }
            }
            match (&mut params.mask_biases, &mut train.mask_bias_updates) {
                (Some(biases), Some(updates)) => apply_host_adam_non_weight(
                    biases,
                    updates,
                    &mut host_adam_child(state, 1).biases,
                    &predictor_adam,
                )?,
                (None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "deform_conv Adam mask bias mismatch".to_owned(),
                    ));
                }
            }
        }
        (LayerParams::DsConv(params), LayerTrainBuffers::DsConv(train)) => {
            apply_host_adam_convolutional(&mut params.collapse, &mut train.collapse, state, adam)?;
            let offset_adam = AdamParams {
                lr: adam.lr * 0.01,
                ..*adam
            };
            let offset_state = host_adam_child(state, 0);
            apply_host_adam(
                &mut params.offset_weights,
                &mut train.offset_weight_updates,
                &mut offset_state.weights,
                &offset_adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.offset_biases,
                &mut train.offset_bias_updates,
                &mut offset_state.biases,
                &offset_adam,
            )?;
        }
        (LayerParams::Clifford(params), LayerTrainBuffers::Clifford(train)) => {
            let LayerKind::Clifford(spec) = kind else {
                return Err(Error::InvalidConfig(
                    "Clifford parameters do not match layer kind during Adam update".to_owned(),
                ));
            };
            let mut next_slot = 0;
            if crate::clifford::has_local(spec) {
                apply_host_adam(
                    &mut params.determinant_weights,
                    &mut train.determinant_weight_updates,
                    &mut state.weights,
                    adam,
                )?;
                apply_host_adam_non_weight(
                    &mut params.determinant_biases,
                    &mut train.determinant_bias_updates,
                    &mut state.biases,
                    adam,
                )?;
                if params.depthwise.len() != train.depthwise.len() {
                    return Err(Error::InvalidConfig(
                        "[clifford] Adam depthwise parameter/update mismatch".to_owned(),
                    ));
                }
                for (conv, updates) in params.depthwise.iter_mut().zip(&mut train.depthwise) {
                    apply_host_adam_convolutional(
                        conv,
                        updates,
                        host_adam_child(state, next_slot),
                        adam,
                    )?;
                    next_slot += 1;
                }
                apply_host_adam_clifford_branch(
                    &mut params.local,
                    &mut train.local,
                    host_adam_child(state, next_slot),
                    adam,
                )?;
                next_slot += 1;
            }
            match (
                crate::clifford::has_global(spec),
                &mut params.global,
                &mut train.global,
            ) {
                (true, Some(params), Some(train)) => {
                    apply_host_adam_clifford_branch(
                        params,
                        train,
                        host_adam_child(state, next_slot),
                        adam,
                    )?;
                    next_slot += 1;
                }
                (false, None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "[clifford] Adam global parameter mismatch".to_owned(),
                    ));
                }
            }
            {
                let norm_state = host_adam_child(state, next_slot);
                apply_host_adam_non_weight(
                    &mut params.layer_norm_gamma,
                    &mut train.layer_norm_gamma_updates,
                    &mut norm_state.scales,
                    adam,
                )?;
                apply_host_adam_non_weight(
                    &mut params.layer_norm_beta,
                    &mut train.layer_norm_beta_updates,
                    &mut norm_state.biases,
                    adam,
                )?;
            }
            next_slot += 1;
            apply_host_adam_non_weight(
                &mut params.layer_scale,
                &mut train.layer_scale_updates,
                &mut host_adam_child(state, next_slot).biases,
                adam,
            )?;
            next_slot += 1;
            match (
                crate::clifford::has_fourier(spec),
                &mut params.fourier,
                &mut train.fourier,
            ) {
                (true, Some(params), Some(train)) => {
                    apply_host_adam(
                        &mut params.real,
                        &mut train.real_updates,
                        &mut host_adam_child(state, next_slot).weights,
                        adam,
                    )?;
                    apply_host_adam(
                        &mut params.imaginary,
                        &mut train.imaginary_updates,
                        &mut host_adam_child(state, next_slot + 1).weights,
                        adam,
                    )?;
                }
                (false, None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "[clifford] Adam Fourier parameter mismatch".to_owned(),
                    ));
                }
            }
        }
        (LayerParams::Wmhf(params), LayerTrainBuffers::Wmhf(train)) => {
            apply_host_adam_convolutional(&mut params.pre, &mut train.pre, state, adam)?;
            for (slot, params, train) in [
                (0, &mut params.local_dw3, &mut train.local_dw3),
                (1, &mut params.local_dw5, &mut train.local_dw5),
                (2, &mut params.local_dw7, &mut train.local_dw7),
                (3, &mut params.local_mix, &mut train.local_mix),
                (4, &mut params.hf_gate, &mut train.hf_gate),
                (5, &mut params.fuse, &mut train.fuse),
            ] {
                apply_host_adam_convolutional(params, train, host_adam_child(state, slot), adam)?;
            }
            apply_host_adam(
                &mut params.scan_weights,
                &mut train.scan_weight_updates,
                &mut host_adam_child(state, 6).weights,
                adam,
            )?;
        }
        (LayerParams::Dcnv4(params), LayerTrainBuffers::Dcnv4(train)) => {
            let offset_adam = AdamParams {
                lr: adam.lr * 0.01,
                ..*adam
            };
            apply_host_adam(
                &mut params.offset_weights,
                &mut train.offset_weight_updates,
                &mut state.weights,
                &offset_adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.offset_biases,
                &mut train.offset_bias_updates,
                &mut host_adam_child(state, 0).biases,
                &offset_adam,
            )?;
            if params.batch_norm.is_none() {
                apply_host_adam_non_weight(
                    &mut params.biases,
                    &mut train.bias_updates,
                    &mut state.biases,
                    adam,
                )?;
            }
            match (&mut params.batch_norm, &mut train.batch_norm) {
                (Some(params_bn), Some(train_bn)) => apply_host_adam_non_weight(
                    &mut params_bn.scales,
                    &mut train_bn.scale_updates,
                    &mut state.scales,
                    adam,
                )?,
                (None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "DCNv4 Adam batchnorm mismatch".to_owned(),
                    ));
                }
            }
        }
        (LayerParams::DptHead(params), LayerTrainBuffers::DptHead(train)) => {
            let LayerKind::DptHead(spec) = kind else {
                return Err(Error::InvalidConfig(
                    "DPT parameters do not match layer kind during Adam update".to_owned(),
                ));
            };
            apply_host_adam_dpt(spec, input_shape, params, train, state, adam, true)?;
        }
        (LayerParams::Moe(params), LayerTrainBuffers::Moe(train)) => {
            if params.experts.len() != train.experts.len() {
                return Err(Error::InvalidConfig(
                    "MoE Adam expert state mismatch".to_owned(),
                ));
            }
            let router_adam = AdamParams {
                decay: 0.0,
                ..*adam
            };
            apply_host_adam(
                &mut params.router_weights,
                &mut train.router_weight_updates,
                &mut state.weights,
                &router_adam,
            )?;
            apply_host_adam_non_weight(
                &mut params.router_biases,
                &mut train.router_bias_updates,
                &mut state.biases,
                &router_adam,
            )?;
            for (slot, (expert, updates)) in params
                .experts
                .iter_mut()
                .zip(&mut train.experts)
                .enumerate()
            {
                apply_host_adam_convolutional(expert, updates, host_adam_child(state, slot), adam)?;
            }
        }
        (LayerParams::EmlConvolutional(params), LayerTrainBuffers::EmlConvolutional(train)) => {
            apply_host_adam_convolutional(
                &mut params.input,
                &mut train.input,
                host_adam_child(state, 0),
                adam,
            )?;
            apply_host_adam_convolutional(
                &mut params.self_params,
                &mut train.self_params,
                host_adam_child(state, 1),
                adam,
            )?;
        }
        (LayerParams::None, LayerTrainBuffers::None)
        | (LayerParams::LayaScorer(_), LayerTrainBuffers::None)
        | (LayerParams::LayaActHead(_), LayerTrainBuffers::None)
        | (LayerParams::LayaEncoder(_), LayerTrainBuffers::None)
        | (LayerParams::LayaHead(_), LayerTrainBuffers::None)
        | (LayerParams::DepthAnything(_), LayerTrainBuffers::None) => {}
        _ => {
            return Err(Error::InvalidConfig(format!(
                "CPU Adam parameter/train-buffer mismatch for {} layer",
                kind.name()
            )));
        }
    }
    Ok(())
}

pub(super) fn apply_host_adam_clifford_branch(
    params: &mut CliffordBranchParams,
    train: &mut CliffordBranchTrainBuffers,
    state: &mut HostAdamLayerState,
    adam: &AdamParams,
) -> Result<()> {
    apply_host_adam(
        &mut params.projection_weights,
        &mut train.projection_weight_updates,
        &mut state.weights,
        adam,
    )?;
    apply_host_adam_non_weight(
        &mut params.projection_biases,
        &mut train.projection_bias_updates,
        &mut state.biases,
        adam,
    )?;
    let gate_state = host_adam_child(state, 0);
    apply_host_adam(
        &mut params.gate_weights,
        &mut train.gate_weight_updates,
        &mut gate_state.weights,
        adam,
    )?;
    apply_host_adam_non_weight(
        &mut params.gate_biases,
        &mut train.gate_bias_updates,
        &mut gate_state.biases,
        adam,
    )
}

pub(super) fn update_connected_params(
    params: &mut ConnectedParams,
    train: &mut ConnectedTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    sgd_update(
        &mut params.weights,
        &mut train.weight_updates,
        lr,
        momentum,
        decay,
        batch,
    );
    sgd_update(
        &mut params.biases,
        &mut train.bias_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    if let (Some(params_bn), Some(train_bn)) = (&mut params.batch_norm, &mut train.batch_norm) {
        sgd_update(
            &mut params_bn.scales,
            &mut train_bn.scale_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
    }
}

pub(super) fn update_clifford_branch(
    params: &mut CliffordBranchParams,
    train: &mut CliffordBranchTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    sgd_update(
        &mut params.projection_weights,
        &mut train.projection_weight_updates,
        lr,
        momentum,
        decay,
        batch,
    );
    sgd_update(
        &mut params.projection_biases,
        &mut train.projection_bias_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    sgd_update(
        &mut params.gate_weights,
        &mut train.gate_weight_updates,
        lr,
        momentum,
        decay,
        batch,
    );
    sgd_update(
        &mut params.gate_biases,
        &mut train.gate_bias_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
}

pub(super) fn update_clifford_params(
    spec: &crate::model::CliffordSpec,
    params: &mut CliffordParams,
    train: &mut CliffordTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    if crate::clifford::has_local(spec) {
        sgd_update(
            &mut params.determinant_weights,
            &mut train.determinant_weight_updates,
            lr,
            momentum,
            decay,
            batch,
        );
        sgd_update(
            &mut params.determinant_biases,
            &mut train.determinant_bias_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
        update_clifford_branch(
            &mut params.local,
            &mut train.local,
            lr,
            momentum,
            decay,
            batch,
        );
        for (params, train) in params.depthwise.iter_mut().zip(&mut train.depthwise) {
            update_convolutional_params(params, train, lr, momentum, decay, batch);
        }
    }
    sgd_update(
        &mut params.layer_norm_gamma,
        &mut train.layer_norm_gamma_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    sgd_update(
        &mut params.layer_norm_beta,
        &mut train.layer_norm_beta_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    sgd_update(
        &mut params.layer_scale,
        &mut train.layer_scale_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    if crate::clifford::has_global(spec) {
        if let (Some(params), Some(train)) = (&mut params.global, &mut train.global) {
            update_clifford_branch(params, train, lr, momentum, decay, batch);
        }
    }
    // FNO coefficients are trainable elementwise spectral multipliers. The
    // guarded C++ backward accumulates them, although its update callback omits
    // them; the Rust CPU path applies the ordinary decayed weight update.
    if crate::clifford::has_fourier(spec)
        && let (Some(params), Some(train)) = (&mut params.fourier, &mut train.fourier)
    {
        sgd_update(
            &mut params.real,
            &mut train.real_updates,
            lr,
            momentum,
            decay,
            batch,
        );
        sgd_update(
            &mut params.imaginary,
            &mut train.imaginary_updates,
            lr,
            momentum,
            decay,
            batch,
        );
    }
}

pub(super) fn update_foveation_film_sgd(
    params: &mut FoveationFilmParams,
    train: &mut FoveationFilmTrainBuffers,
    lr: f32,
    momentum: f32,
    batch: usize,
) {
    for (weights, updates) in [
        (&mut params.w_embed, &mut train.w_embed_updates),
        (&mut params.w_gamma, &mut train.w_gamma_updates),
        (&mut params.w_beta, &mut train.w_beta_updates),
        (&mut params.b_embed, &mut train.b_embed_updates),
        (&mut params.b_gamma, &mut train.b_gamma_updates),
        (&mut params.b_beta, &mut train.b_beta_updates),
    ] {
        // C++'s decay argument is unused for this layer; preserve the Rust
        // runtime's downhill-update sign convention and do not decay biases.
        sgd_update(weights, updates, lr, momentum, 0.0, batch);
    }
}

pub(super) fn update_hyperbolic_foveation_sgd(
    spec: &crate::model::HyperbolicFoveationSpec,
    params: &mut HyperbolicFoveationParams,
    train: &mut HyperbolicFoveationTrainBuffers,
    lr: f32,
    momentum: f32,
    batch: usize,
) {
    let lr = lr * spec.hf_lr_mult;
    if spec.pred_hidden > 0 {
        train.hf_param_updates.fill(0.0);
        sgd_update(
            &mut params.pred_w1,
            &mut train.pred_w1_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
        sgd_update(
            &mut params.pred_b1,
            &mut train.pred_b1_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
        sgd_update(
            &mut params.pred_w2,
            &mut train.pred_w2_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
        sgd_update(
            &mut params.pred_b2,
            &mut train.pred_b2_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
    } else {
        sgd_update(
            &mut params.hf_params,
            &mut train.hf_param_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
    }
    clamp_hyperbolic_foveation_params(spec, params);
}

pub(super) fn update_recursive_block_sgd(
    spec: &crate::model::RecursiveBlockSpec,
    params: &mut RecursiveBlockParams,
    train: &mut RecursiveBlockTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    for ((layer, params), train) in spec.body.iter().zip(&mut params.body).zip(&mut train.body) {
        match (&layer.kind, params, train) {
            (
                LayerKind::Convolutional(_),
                LayerParams::Convolutional(params),
                LayerTrainBuffers::Convolutional(train),
            ) => update_convolutional_params(params, train, lr, momentum, decay, batch),
            (
                LayerKind::Connected(_),
                LayerParams::Connected(params),
                LayerTrainBuffers::Connected(train),
            ) => update_connected_params(params, train, lr, momentum, decay, batch),
            (
                LayerKind::MaxPool(_)
                | LayerKind::LocalAvgPool(_)
                | LayerKind::AvgPool
                | LayerKind::Upsample(_)
                | LayerKind::ChannelShuffle(_)
                | LayerKind::Dropout { .. },
                LayerParams::None,
                LayerTrainBuffers::None,
            ) => {}
            _ => {}
        }
    }
    if spec.ouroboros == 1 {
        sgd_update(
            &mut params.controller_weights,
            &mut train.controller_weight_updates,
            lr,
            momentum,
            decay,
            batch,
        );
        sgd_update(
            &mut params.controller_biases,
            &mut train.controller_bias_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
    }
}

pub(super) fn update_convolutional_params(
    params: &mut ConvolutionalParams,
    train: &mut ConvolutionalTrainBuffers,
    lr: f32,
    momentum: f32,
    decay: f32,
    batch: usize,
) {
    sgd_update(
        &mut params.weights,
        &mut train.weight_updates,
        lr,
        momentum,
        decay,
        batch,
    );
    sgd_update(
        &mut params.biases,
        &mut train.bias_updates,
        lr,
        momentum,
        0.0,
        batch,
    );
    if let (Some(params_bn), Some(train_bn)) = (&mut params.batch_norm, &mut train.batch_norm) {
        sgd_update(
            &mut params_bn.scales,
            &mut train_bn.scale_updates,
            lr,
            momentum,
            0.0,
            batch,
        );
    }
}

pub(super) fn accumulate_clifford_branch(
    train: &mut CliffordBranchTrainBuffers,
    gradient: &CliffordBranchParams,
) -> Result<()> {
    accumulate_vec(
        &mut train.projection_weight_updates,
        &gradient.projection_weights,
    )?;
    accumulate_vec(
        &mut train.projection_bias_updates,
        &gradient.projection_biases,
    )?;
    accumulate_vec(&mut train.gate_weight_updates, &gradient.gate_weights)?;
    accumulate_vec(&mut train.gate_bias_updates, &gradient.gate_biases)
}

pub(super) fn accumulate_convolutional_param_gradients(
    dst: &mut LayerParams,
    src: &ConvolutionalParams,
) -> Result<()> {
    let LayerParams::Convolutional(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected convolutional gradient destination".to_owned(),
        ));
    };
    accumulate_convolutional_params(dst, src)
}

/// The same accumulation for a convolution that is nested inside another
/// layer's parameters - `[moe_convolutional]`'s experts, whose gradient slot is
/// a plain `ConvolutionalParams` rather than a `LayerParams`.
pub(super) fn accumulate_convolutional_params(
    dst: &mut ConvolutionalParams,
    src: &ConvolutionalParams,
) -> Result<()> {
    accumulate_vec(&mut dst.biases, &src.biases)?;
    accumulate_vec(&mut dst.weights, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scales, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.rolling_mean, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.rolling_variance, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "convolutional batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn accumulate_connected_param_gradients(
    dst: &mut LayerParams,
    src: &ConnectedParams,
) -> Result<()> {
    let LayerParams::Connected(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected connected gradient destination".to_owned(),
        ));
    };
    accumulate_vec(&mut dst.biases, &src.biases)?;
    accumulate_vec(&mut dst.weights, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scales, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.rolling_mean, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.rolling_variance, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "connected batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn accumulate_rnn_param_gradients(dst: &mut LayerParams, src: &RnnParams) -> Result<()> {
    let LayerParams::Rnn(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected RNN gradient destination".to_owned(),
        ));
    };
    accumulate_connected_gradient(&mut dst.input, &src.input)?;
    accumulate_connected_gradient(&mut dst.self_params, &src.self_params)?;
    accumulate_connected_gradient(&mut dst.output, &src.output)
}

pub(super) fn accumulate_lstm_param_gradients(dst: &mut LayerParams, src: &LstmParams) -> Result<()> {
    let LayerParams::Lstm(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected LSTM gradient destination".to_owned(),
        ));
    };
    accumulate_connected_gradient(&mut dst.wf, &src.wf)?;
    accumulate_connected_gradient(&mut dst.wi, &src.wi)?;
    accumulate_connected_gradient(&mut dst.wg, &src.wg)?;
    accumulate_connected_gradient(&mut dst.wo, &src.wo)?;
    accumulate_connected_gradient(&mut dst.uf, &src.uf)?;
    accumulate_connected_gradient(&mut dst.ui, &src.ui)?;
    accumulate_connected_gradient(&mut dst.ug, &src.ug)?;
    accumulate_connected_gradient(&mut dst.uo, &src.uo)
}

pub(super) fn accumulate_crnn_param_gradients(dst: &mut LayerParams, src: &CrnnParams) -> Result<()> {
    let LayerParams::Crnn(dst) = dst else {
        return Err(Error::InvalidConfig(
            "expected CRNN gradient destination".to_owned(),
        ));
    };
    accumulate_convolutional_gradient(&mut dst.input, &src.input)?;
    accumulate_convolutional_gradient(&mut dst.self_params, &src.self_params)?;
    accumulate_convolutional_gradient(&mut dst.output, &src.output)
}

pub(super) fn accumulate_connected_train_buffer(
    dst: &mut ConnectedTrainBuffers,
    src: &ConnectedParams,
) -> Result<()> {
    accumulate_vec(&mut dst.bias_updates, &src.biases)?;
    accumulate_vec(&mut dst.weight_updates, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scale_updates, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.mean_delta, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.variance_delta, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "connected recurrent batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn accumulate_convolutional_train_buffer(
    dst: &mut ConvolutionalTrainBuffers,
    src: &ConvolutionalParams,
) -> Result<()> {
    accumulate_vec(&mut dst.bias_updates, &src.biases)?;
    accumulate_vec(&mut dst.weight_updates, &src.weights)?;
    match (&mut dst.batch_norm, &src.batch_norm) {
        (Some(dst_bn), Some(src_bn)) => {
            accumulate_vec(&mut dst_bn.scale_updates, &src_bn.scales)?;
            accumulate_vec(&mut dst_bn.mean_delta, &src_bn.rolling_mean)?;
            accumulate_vec(&mut dst_bn.variance_delta, &src_bn.rolling_variance)?;
        }
        (None, None) => {}
        _ => {
            return Err(Error::InvalidConfig(
                "convolutional recurrent batch norm gradient mismatch".to_owned(),
            ));
        }
    }
    Ok(())
}

pub(super) fn accumulate_recursive_block_train_buffers(
    spec: &crate::model::RecursiveBlockSpec,
    train: &mut RecursiveBlockTrainBuffers,
    gradient: &RecursiveBlockParams,
) -> Result<()> {
    if spec.body.len() != train.body.len() || spec.body.len() != gradient.body.len() {
        return Err(Error::InvalidConfig(
            "recursive-block body gradient/update count mismatch".to_owned(),
        ));
    }
    for ((layer, train), gradient) in spec.body.iter().zip(&mut train.body).zip(&gradient.body) {
        match (&layer.kind, train, gradient) {
            (
                LayerKind::Convolutional(_),
                LayerTrainBuffers::Convolutional(train),
                LayerParams::Convolutional(gradient),
            ) => accumulate_convolutional_train_buffer(train, gradient)?,
            (
                LayerKind::Connected(_),
                LayerTrainBuffers::Connected(train),
                LayerParams::Connected(gradient),
            ) => accumulate_connected_train_buffer(train, gradient)?,
            (
                LayerKind::MaxPool(_)
                | LayerKind::LocalAvgPool(_)
                | LayerKind::AvgPool
                | LayerKind::Upsample(_)
                | LayerKind::ChannelShuffle(_)
                | LayerKind::Dropout { .. },
                LayerTrainBuffers::None,
                LayerParams::None,
            ) => {}
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "unsupported recursive-block gradient layout for body layer #{} ({})",
                    layer.index,
                    layer.kind.name()
                )));
            }
        }
    }
    accumulate_vec(
        &mut train.controller_weight_updates,
        &gradient.controller_weights,
    )?;
    accumulate_vec(
        &mut train.controller_bias_updates,
        &gradient.controller_biases,
    )
}

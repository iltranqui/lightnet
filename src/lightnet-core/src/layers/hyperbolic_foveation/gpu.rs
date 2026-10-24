use crate::error::{Error, Result};
use crate::model::{LayerKind, Network};
use crate::runtime::*;

pub(crate) fn hyperbolic_foveation_predictor_initial_biases(
    spec: &crate::model::HyperbolicFoveationSpec,
) -> Vec<f32> {
    if spec.pred_hidden == 0 {
        return Vec::new();
    }
    let param_count = spec.param_count();
    let mut biases = vec![0.0; param_count];
    let center_logit = |center: f32| (center.clamp(-0.899, 0.899) / 0.9).atanh();
    biases[0] = center_logit(spec.init_cx);
    biases[1] = center_logit(spec.init_cy);
    let inverse_softplus = |value: f32| {
        let safe = value.max(1.0e-6);
        if safe > 20.0 {
            safe
        } else {
            safe.exp_m1().ln()
        }
    };
    let radius_x = spec.init_radius_x.clamp(0.1, 2.0);
    biases[2] = inverse_softplus((radius_x - 0.1) * 0.5);
    if spec.radius_mode == crate::model::HfRadiusMode::Anisotropic {
        let radius_y = spec.init_radius_y.clamp(0.1, 2.0);
        biases[3] = inverse_softplus((radius_y - 0.1) * 0.5);
    }
    let strength = spec.init_strength.clamp(1.0e-4, 1.0 - 1.0e-4);
    biases[param_count - 1] = (strength / (1.0 - strength)).ln();
    biases
}

pub(crate) fn clamp_hyperbolic_foveation_params(
    spec: &crate::model::HyperbolicFoveationSpec,
    params: &mut HyperbolicFoveationParams,
) {
    params.hf_params[0] = params.hf_params[0].clamp(-0.9, 0.9);
    params.hf_params[1] = params.hf_params[1].clamp(-0.9, 0.9);
    match spec.radius_mode {
        crate::model::HfRadiusMode::Isotropic => {
            params.hf_params[2] = params.hf_params[2].clamp(0.1, 2.0);
        }
        crate::model::HfRadiusMode::Anisotropic => {
            params.hf_params[2] = params.hf_params[2].clamp(0.1, 2.0);
            params.hf_params[3] = params.hf_params[3].clamp(0.1, 2.0);
        }
    }
    let strength = params.hf_params.len() - 1;
    params.hf_params[strength] = params.hf_params[strength].clamp(0.0, 1.0);
}

pub(crate) fn validate_foveation_film_param_layout(
    spec: &crate::model::FoveationFilmSpec,
    channels: usize,
    params: &FoveationFilmParams,
) -> Result<()> {
    let embed_weights = spec
        .embed_dim
        .checked_mul(spec.param_count)
        .ok_or_else(|| {
            Error::InvalidConfig("[foveation_film] embedding size overflows".to_owned())
        })?;
    let channel_weights = channels.checked_mul(spec.embed_dim).ok_or_else(|| {
        Error::InvalidConfig("[foveation_film] channel size overflows".to_owned())
    })?;
    let actual = (
        params.w_embed.len(),
        params.b_embed.len(),
        params.w_gamma.len(),
        params.b_gamma.len(),
        params.w_beta.len(),
        params.b_beta.len(),
    );
    let expected = (
        embed_weights,
        spec.embed_dim,
        channel_weights,
        channels,
        channel_weights,
        channels,
    );
    if spec.param_count == 0 || spec.embed_dim == 0 || channels == 0 || actual != expected {
        return Err(Error::InvalidConfig(format!(
            "[foveation_film] parameter layout mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_hyperbolic_foveation_param_layout(
    spec: &crate::model::HyperbolicFoveationSpec,
    params: &HyperbolicFoveationParams,
) -> Result<()> {
    let input_dim = spec.predictor_input_dim().ok_or_else(|| {
        Error::InvalidConfig(
            "[hyperbolic_foveation] predictor input dimension overflows".to_owned(),
        )
    })?;
    let expected = [
        spec.param_count(),
        spec.pred_hidden.saturating_mul(input_dim),
        spec.pred_hidden,
        spec.param_count().saturating_mul(spec.pred_hidden),
        if spec.pred_hidden > 0 {
            spec.param_count()
        } else {
            0
        },
    ];
    let actual = [
        params.hf_params.len(),
        params.pred_w1.len(),
        params.pred_b1.len(),
        params.pred_w2.len(),
        params.pred_b2.len(),
    ];
    if actual != expected {
        return Err(Error::InvalidConfig(format!(
            "[hyperbolic_foveation] parameter layout mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

pub(crate) fn hyperbolic_foveation_params_from_values(
    values: &[f32],
    spec: &crate::model::HyperbolicFoveationSpec,
) -> Result<crate::hyperbolic_foveation::HfParams> {
    if values.len() != spec.param_count() || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidConfig(format!(
            "[hyperbolic_foveation] expected {} finite parameters, got {}",
            spec.param_count(),
            values.len()
        )));
    }
    let (radius_x, radius_y, strength_index) = match spec.radius_mode {
        crate::model::HfRadiusMode::Isotropic => (values[2], values[2], 3),
        crate::model::HfRadiusMode::Anisotropic => (values[2], values[3], 4),
    };
    Ok(crate::hyperbolic_foveation::HfParams {
        center_x: values[0].clamp(-0.9, 0.9),
        center_y: values[1].clamp(-0.9, 0.9),
        radius_x: radius_x.clamp(0.1, 2.0),
        radius_y: radius_y.clamp(0.1, 2.0),
        strength: values[strength_index].clamp(0.0, 1.0),
        alpha: spec.alpha,
        p_exp: spec.p_exp,
        radius_mode: spec.radius_mode,
    })
}

pub(crate) fn hyperbolic_foveation_auxiliary_param_gradients(
    network: &Network,
    forward: &ForwardResult,
    labels: &[Vec<crate::train::YoloLabel>],
) -> Result<Vec<Option<Vec<f32>>>> {
    let batch = network.net.batch.max(1);
    if labels.len() != batch {
        return Err(Error::InvalidConfig(format!(
            "[hyperbolic_foveation] auxiliary label batch mismatch: expected {batch}, got {}",
            labels.len()
        )));
    }
    let has_truth_layer = network.layers.iter().any(|layer| {
        matches!(
            layer.kind,
            LayerKind::Yolo(_) | LayerKind::GaussianYolo(_) | LayerKind::ObbYolo(_)
        )
    });
    let mut gradients = vec![None; network.layers.len()];
    for layer in &network.layers {
        let LayerKind::HyperbolicFoveation(spec) = &layer.kind else {
            continue;
        };
        if spec.tiny_center_loss == 0.0
            && spec.tiny_radius_loss == 0.0
            && spec.tiny_strength_loss == 0.0
            && spec.strength_zero_loss == 0.0
        {
            continue;
        }
        let count = spec.param_count();
        let cache = forward
            .hyperbolic_foveation_caches
            .get(layer.index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing [hyperbolic_foveation] auxiliary cache for layer #{}",
                    layer.index
                ))
            })?;
        if cache.params_per_batch.len() != batch * count {
            return Err(Error::InvalidConfig(format!(
                "[hyperbolic_foveation] layer #{} auxiliary parameter count mismatch",
                layer.index
            )));
        }

        let mut targets = vec![None; batch];
        if has_truth_layer && spec.small_box_px > 0.0 && spec.small_box_strength > 0.0 {
            for (batch_index, batch_labels) in labels.iter().enumerate() {
                if batch_labels.iter().any(|label| {
                    !label.x.is_finite()
                        || !label.y.is_finite()
                        || !label.w.is_finite()
                        || !label.h.is_finite()
                }) {
                    return Err(Error::InvalidConfig(format!(
                        "[hyperbolic_foveation] layer #{} received non-finite labels",
                        layer.index
                    )));
                }
                let tiny = batch_labels
                    .iter()
                    .filter(|label| {
                        label.w * layer.output_shape.width as f32 > 0.0
                            && label.h * layer.output_shape.height as f32 > 0.0
                            && label.w * (layer.output_shape.width as f32) < spec.small_box_px
                            && label.h * (layer.output_shape.height as f32) < spec.small_box_px
                    })
                    .collect::<Vec<_>>();
                if tiny.is_empty() {
                    continue;
                }
                let mut total_weight = 0.0f32;
                let mut target_x = 0.0f32;
                let mut target_y = 0.0f32;
                for label in &tiny {
                    let width = label.w * layer.output_shape.width as f32;
                    let height = label.h * layer.output_shape.height as f32;
                    let weight = 1.0 / (width * height).max(1.0);
                    target_x += weight * (2.0 * label.x - 1.0);
                    target_y += weight * (2.0 * label.y - 1.0);
                    total_weight += weight;
                }
                if !total_weight.is_finite() || total_weight <= 0.0 {
                    return Err(Error::InvalidConfig(format!(
                        "[hyperbolic_foveation] layer #{} tiny-box target has invalid total weight",
                        layer.index
                    )));
                }
                target_x /= total_weight;
                target_y /= total_weight;
                let mut target_radius = 0.25f32;
                for label in &tiny {
                    let dx = 2.0 * label.x - 1.0 - target_x;
                    let dy = 2.0 * label.y - 1.0 - target_y;
                    target_radius = target_radius
                        .max((dx * dx + dy * dy).sqrt() + 2.0 * label.w.max(label.h) + 0.15);
                }
                let mut target = vec![0.0; count];
                target[0] = target_x.clamp(-0.9, 0.9);
                target[1] = target_y.clamp(-0.9, 0.9);
                target[2] = target_radius.clamp(0.1, 2.0);
                if spec.radius_mode == crate::model::HfRadiusMode::Anisotropic {
                    target[3] = target[2];
                }
                target[count - 1] = spec.small_box_strength.clamp(0.0, 1.0);
                targets[batch_index] = Some(target);
            }
        }
        let tiny_image_count = targets.iter().filter(|target| target.is_some()).count();
        let mut auxiliary = vec![0.0; batch * count];
        for batch_index in 0..batch {
            let start = batch_index * count;
            let current = &cache.params_per_batch[start..start + count];
            if let Some(target) = &targets[batch_index] {
                let norm = 1.0 / tiny_image_count as f32;
                auxiliary[start] += norm * spec.tiny_center_loss * (current[0] - target[0]);
                auxiliary[start + 1] += norm * spec.tiny_center_loss * (current[1] - target[1]);
                auxiliary[start + 2] += norm * spec.tiny_radius_loss * (current[2] - target[2]);
                if spec.radius_mode == crate::model::HfRadiusMode::Anisotropic {
                    auxiliary[start + 3] += norm * spec.tiny_radius_loss * (current[3] - target[3]);
                }
                auxiliary[start + count - 1] +=
                    norm * spec.tiny_strength_loss * (current[count - 1] - target[count - 1]);
            }
            if spec.strength_zero_loss > 0.0 && current[count - 1] > 0.0 {
                auxiliary[start + count - 1] += spec.strength_zero_loss;
            }
        }
        if auxiliary.iter().any(|value| *value != 0.0) {
            gradients[layer.index] = Some(auxiliary);
        }
    }
    Ok(gradients)
}

pub(crate) fn forward_hyperbolic_foveation(
    network: &Network,
    layer: &crate::model::Layer,
    spec: &crate::model::HyperbolicFoveationSpec,
    params: &HyperbolicFoveationParams,
    network_input: &Tensor,
    current_input: &Tensor,
    layer_outputs: &[Tensor],
) -> Result<(Tensor, HyperbolicFoveationForwardCache)> {
    validate_hyperbolic_foveation_param_layout(spec, params)?;
    let warp_input = match spec.warp_source {
        crate::model::HfWarpSource::Previous => current_input,
        crate::model::HfWarpSource::NetworkInput => network_input,
    };
    if warp_input.shape != layer.output_shape {
        return Err(Error::InvalidConfig(format!(
            "[hyperbolic_foveation] warp source shape mismatch: expected {}, got {}",
            layer.output_shape, warp_input.shape
        )));
    }
    let predictor_source = match spec.pred_source_layer {
        Some(index) => layer_outputs.get(index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[hyperbolic_foveation] predictor source layer #{index} is unavailable"
            ))
        })?,
        None => current_input,
    };
    if predictor_source.shape != spec.pred_input_shape {
        return Err(Error::InvalidConfig(format!(
            "[hyperbolic_foveation] predictor source shape mismatch: expected {}, got {}",
            spec.pred_input_shape, predictor_source.shape
        )));
    }

    let predictor = if spec.pred_hidden > 0 {
        Some(crate::hyperbolic_foveation::predictor_forward_nchw(
            &predictor_source.data,
            predictor_source.shape.batch,
            predictor_source.shape.channels,
            predictor_source.shape.height,
            predictor_source.shape.width,
            spec.downsample,
            spec.pred_hidden,
            spec.param_count(),
            &params.pred_w1,
            &params.pred_b1,
            &params.pred_w2,
            &params.pred_b2,
        )?)
    } else {
        None
    };
    let mut params_per_batch = if let Some(predictor) = &predictor {
        predictor.constrained_params.clone()
    } else {
        let mut values = Vec::with_capacity(warp_input.shape.batch * spec.param_count());
        for _ in 0..warp_input.shape.batch {
            values.extend_from_slice(&params.hf_params);
        }
        values
    };
    if params_per_batch.len() != warp_input.shape.batch * spec.param_count() {
        return Err(Error::InvalidConfig(
            "[hyperbolic_foveation] per-batch parameter count mismatch".to_owned(),
        ));
    }
    let params_for_grid = params_per_batch
        .chunks_exact(spec.param_count())
        .map(|values| hyperbolic_foveation_params_from_values(values, spec))
        .collect::<Result<Vec<_>>>()?;
    let grid = crate::hyperbolic_foveation::sampling_grid_nchw(
        &params_for_grid,
        warp_input.shape.batch,
        warp_input.shape.height,
        warp_input.shape.width,
        spec.inverse_method,
        spec.nr_iters,
        spec.nr_eta,
    )?;
    let output_data = crate::hyperbolic_foveation::warp_nchw(
        &warp_input.data,
        warp_input.shape.batch,
        warp_input.shape.channels,
        warp_input.shape.height,
        warp_input.shape.width,
        &grid,
    )?;
    let output = Tensor::from_data(layer.output_shape, output_data)?;
    if network.layers.get(layer.index).is_none() {
        return Err(Error::InvalidConfig(format!(
            "[hyperbolic_foveation] layer #{} is not part of its network",
            layer.index
        )));
    }
    Ok((
        output,
        HyperbolicFoveationForwardCache {
            grid,
            params_per_batch: std::mem::take(&mut params_per_batch),
            predictor,
        },
    ))
}

pub(crate) fn backward_hyperbolic_foveation(
    spec: &crate::model::HyperbolicFoveationSpec,
    params: &HyperbolicFoveationParams,
    warp_input: &Tensor,
    output_gradient: &Tensor,
    cache: &HyperbolicFoveationForwardCache,
    auxiliary_param_gradients: Option<&[f32]>,
    precomputed_grid_gradient: Option<&[f32]>,
    precomputed_semantic_gradients: Option<&[f32]>,
    defer_predictor_backward: bool,
) -> Result<(
    Tensor,
    Option<(Option<usize>, Tensor)>,
    HyperbolicFoveationParams,
    Vec<f32>,
)> {
    validate_hyperbolic_foveation_param_layout(spec, params)?;
    if output_gradient.shape != warp_input.shape {
        return Err(Error::InvalidConfig(format!(
            "[hyperbolic_foveation] output gradient shape {} does not match warp source {}",
            output_gradient.shape, warp_input.shape
        )));
    }
    let per_batch = spec.param_count();
    if cache.params_per_batch.len() != warp_input.shape.batch * per_batch {
        return Err(Error::InvalidConfig(
            "[hyperbolic_foveation] cached per-batch parameter layout mismatch".to_owned(),
        ));
    }
    if auxiliary_param_gradients
        .is_some_and(|gradients| gradients.len() != warp_input.shape.batch * per_batch)
    {
        return Err(Error::InvalidConfig(
            "[hyperbolic_foveation] auxiliary parameter-gradient layout mismatch".to_owned(),
        ));
    }
    let params_for_grid = cache
        .params_per_batch
        .chunks_exact(per_batch)
        .map(|values| hyperbolic_foveation_params_from_values(values, spec))
        .collect::<Result<Vec<_>>>()?;
    let (warp_input_gradient_data, semantic_gradients) = if let Some(semantic) =
        precomputed_semantic_gradients
    {
        if semantic.len() != warp_input.shape.batch * 5 {
            return Err(Error::InvalidConfig(
                "[hyperbolic_foveation] precomputed semantic gradient layout mismatch".to_owned(),
            ));
        }
        let semantic = semantic
            .chunks_exact(5)
            .map(|values| [values[0], values[1], values[2], values[3], values[4]])
            .collect();
        (vec![0.0; warp_input.data.len()], semantic)
    } else if let Some(grid_gradient) = precomputed_grid_gradient {
        let semantic = crate::hyperbolic_foveation::warp_parameter_gradients_from_grid_gradient(
            grid_gradient,
            warp_input.shape.batch,
            warp_input.shape.height,
            warp_input.shape.width,
            &params_for_grid,
            spec.inverse_method,
            spec.nr_iters,
            spec.nr_eta,
        )?;
        (vec![0.0; warp_input.data.len()], semantic)
    } else {
        let (warp, semantic) = crate::hyperbolic_foveation::warp_backward_with_param_gradients(
            &warp_input.data,
            &output_gradient.data,
            warp_input.shape.batch,
            warp_input.shape.channels,
            warp_input.shape.height,
            warp_input.shape.width,
            &cache.grid,
            &params_for_grid,
            spec.inverse_method,
            spec.nr_iters,
            spec.nr_eta,
        )?;
        (warp.input, semantic)
    };
    let warp_input_gradient = Tensor::from_data(warp_input.shape, warp_input_gradient_data)?;
    let mut gradient = HyperbolicFoveationParams {
        hf_params: vec![0.0; per_batch],
        pred_w1: vec![0.0; params.pred_w1.len()],
        pred_b1: vec![0.0; params.pred_b1.len()],
        pred_w2: vec![0.0; params.pred_w2.len()],
        pred_b2: vec![0.0; params.pred_b2.len()],
    };
    let mut constrained_seed = vec![0.0; warp_input.shape.batch * per_batch];
    if spec.pred_hidden > 0 {
        for (batch_index, semantic) in semantic_gradients.iter().enumerate() {
            let start = batch_index * per_batch;
            constrained_seed[start] = semantic[0];
            constrained_seed[start + 1] = semantic[1];
            constrained_seed[start + 2] = semantic[2];
            match spec.radius_mode {
                crate::model::HfRadiusMode::Isotropic => constrained_seed[start + 3] = semantic[4],
                crate::model::HfRadiusMode::Anisotropic => {
                    constrained_seed[start + 3] = semantic[3];
                    constrained_seed[start + 4] = semantic[4];
                }
            }
            if let Some(auxiliary) = auxiliary_param_gradients {
                let extra = &auxiliary[start..start + per_batch];
                constrained_seed[start] += extra[0];
                constrained_seed[start + 1] += extra[1];
                constrained_seed[start + 2] += extra[2];
                if spec.radius_mode == crate::model::HfRadiusMode::Anisotropic {
                    constrained_seed[start + 3] += extra[3];
                }
                constrained_seed[start + per_batch - 1] += extra[per_batch - 1];
            }
        }
    }
    let predictor_gradient = if spec.pred_hidden > 0 {
        let predictor = cache.predictor.as_ref().ok_or_else(|| {
            Error::InvalidConfig(
                "[hyperbolic_foveation] missing predictor forward cache".to_owned(),
            )
        })?;
        if defer_predictor_backward {
            Some((spec.pred_source_layer, Tensor::zeros(spec.pred_input_shape)))
        } else {
            let predictor_source_gradient = crate::hyperbolic_foveation::predictor_backward_nchw(
                (
                    spec.pred_input_shape.batch,
                    spec.pred_input_shape.channels,
                    spec.pred_input_shape.height,
                    spec.pred_input_shape.width,
                ),
                spec.downsample,
                spec.pred_hidden,
                per_batch,
                &params.pred_w1,
                &params.pred_w2,
                predictor,
                &constrained_seed,
            )?;
            gradient.pred_w1 = predictor_source_gradient.w1;
            gradient.pred_b1 = predictor_source_gradient.b1;
            gradient.pred_w2 = predictor_source_gradient.w2;
            gradient.pred_b2 = predictor_source_gradient.b2;
            Some((
                spec.pred_source_layer,
                Tensor::from_data(spec.pred_input_shape, predictor_source_gradient.source)?,
            ))
        }
    } else {
        if cache.predictor.is_some() {
            return Err(Error::InvalidConfig(
                "[hyperbolic_foveation] unexpected predictor forward cache".to_owned(),
            ));
        }
        for semantic in &semantic_gradients {
            if params.hf_params[0] > -0.9 && params.hf_params[0] < 0.9 {
                gradient.hf_params[0] += semantic[0];
            }
            if params.hf_params[1] > -0.9 && params.hf_params[1] < 0.9 {
                gradient.hf_params[1] += semantic[1];
            }
            gradient.hf_params[2] += semantic[2];
            match spec.radius_mode {
                crate::model::HfRadiusMode::Isotropic => {
                    if params.hf_params[3] > 0.0 && params.hf_params[3] < 1.0 {
                        gradient.hf_params[3] += semantic[4];
                    }
                }
                crate::model::HfRadiusMode::Anisotropic => {
                    gradient.hf_params[3] += semantic[3];
                    if params.hf_params[4] > 0.0 && params.hf_params[4] < 1.0 {
                        gradient.hf_params[4] += semantic[4];
                    }
                }
            }
        }
        if let Some(auxiliary) = auxiliary_param_gradients {
            for values in auxiliary.chunks_exact(per_batch) {
                for (gradient, extra) in gradient.hf_params.iter_mut().zip(values) {
                    *gradient += extra;
                }
            }
        }
        None
    };
    Ok((
        warp_input_gradient,
        predictor_gradient,
        gradient,
        constrained_seed,
    ))
}

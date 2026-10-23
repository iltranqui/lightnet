use super::*;

impl ExecutableNetwork {
    pub fn backward(
        &self,
        input: &Tensor,
        forward: &ForwardResult,
        output_gradient: &Tensor,
    ) -> Result<BackwardResult> {
        let mut layer_output_gradient_seeds = vec![None; self.network.layers.len()];
        let last_index = self
            .network
            .layers
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidConfig("network has no layers".to_owned()))?;
        layer_output_gradient_seeds[last_index] = Some(output_gradient.clone());
        self.backward_seeded(input, forward, &layer_output_gradient_seeds)
    }

    pub fn backward_seeded(
        &self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<BackwardResult> {
        self.backward_seeded_with_grad_mode(
            input,
            forward,
            layer_output_gradient_seeds,
            GradMode::from_env(),
        )
    }

    pub(super) fn backward_seeded_with_grad_mode(
        &self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
        grad_mode: GradMode,
    ) -> Result<BackwardResult> {
        // Allocate a local workspace for this standalone (non-accumulate) backward call.
        // backward_accumulate_seeded reuses the pre-allocated self.workspace instead.
        let mut local_workspace = vec![0.0f32; compute_cpu_workspace_size(&self.network)];
        let mut attention_backward =
            |d_context: &[f32], _: &[f32], attention: &crate::vit::AttentionCache| {
                crate::vit::attention_backward(d_context, attention)
            };
        self.backward_seeded_core(
            &mut local_workspace,
            input,
            forward,
            layer_output_gradient_seeds,
            None,
            grad_mode,
            &mut attention_backward,
        )
    }

    pub(super) fn backward_seeded_core<F>(
        &self,
        workspace: &mut Vec<f32>,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
        hf_auxiliary_param_gradients: Option<&[Option<Vec<f32>>]>,
        grad_mode: GradMode,
        attention_backward: &mut F,
    ) -> Result<BackwardResult>
    where
        F: FnMut(
            &[f32],
            &[f32],
            &crate::vit::AttentionCache,
        ) -> std::result::Result<Vec<f32>, String>,
    {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        if self
            .network
            .layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::ObbRefine(_)))
        {
            return Err(Error::InvalidConfig(
                "backward for networks containing [obb_refine] is unsupported until CARPN Stage-2 gradients and updates are implemented".to_owned(),
            ));
        }
        if forward.layer_outputs.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward result layer count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.layer_outputs.len()
            )));
        }
        if forward.layer_activation_inputs.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward activation cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.layer_activation_inputs.len()
            )));
        }
        if forward.dropout_masks.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward dropout mask count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.dropout_masks.len()
            )));
        }
        if forward.crop_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward crop cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.crop_caches.len()
            )));
        }
        if forward.depth_anything_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward [depth_anything] cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.depth_anything_caches.len()
            )));
        }
        if forward.dpt_head_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward [dpt_head] cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.dpt_head_caches.len()
            )));
        }
        if forward.swin_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward Swin cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.swin_caches.len()
            )));
        }
        if forward.tucker_attention_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward Tucker attention cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.tucker_attention_caches.len()
            )));
        }
        if forward.mambavision_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward MambaVision cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.mambavision_caches.len()
            )));
        }
        if forward.kimi_kda_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward Kimi KDA cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.kimi_kda_caches.len()
            )));
        }
        if forward.deform_conv_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward deform_conv cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.deform_conv_caches.len()
            )));
        }
        if forward.wmhf_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward [wmhf] cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.wmhf_caches.len()
            )));
        }
        if forward.clifford_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward [clifford] cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.clifford_caches.len()
            )));
        }
        if forward.recursive_block_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward [recursive_block] cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.recursive_block_caches.len()
            )));
        }
        if forward.foveation_film_caches.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "forward [foveation_film] cache count mismatch: expected {}, got {}",
                self.network.layers.len(),
                forward.foveation_film_caches.len()
            )));
        }
        if layer_output_gradient_seeds.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "backward seed count mismatch: expected {}, got {}",
                self.network.layers.len(),
                layer_output_gradient_seeds.len()
            )));
        }

        let mut input_gradient = Tensor::zeros(input.shape);
        let mut mambavision_state_gradients = vec![None; self.network.layers.len()];
        let mut layer_output_gradients = self
            .network
            .layers
            .iter()
            .map(|layer| Tensor::zeros(layer.output_shape))
            .collect::<Vec<_>>();
        let mut param_gradients = self
            .network
            .layers
            .iter()
            .map(default_layer_params)
            .collect::<Vec<_>>();
        let mut foveation_hf_auxiliary_gradients: Vec<Option<Vec<f32>>> =
            vec![None; self.network.layers.len()];
        for gradient in &mut param_gradients {
            match gradient {
                LayerParams::HyperbolicFoveation(gradient) => gradient.hf_params.fill(0.0),
                LayerParams::DepthAnything(gradient) => gradient.as_mut_slice().fill(0.0),
                LayerParams::DptHead(gradient) => gradient.values.fill(0.0),
                _ => {}
            }
        }
        for (layer, seed) in self.network.layers.iter().zip(layer_output_gradient_seeds) {
            if let Some(seed) = seed {
                if seed.shape != layer.output_shape {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} backward seed shape mismatch: expected {}, got {}",
                        layer.index, layer.output_shape, seed.shape
                    )));
                }
                layer_output_gradients[layer.index] = seed.clone();
            }
        }

        for (layer, params) in self.network.layers.iter().zip(&self.params).rev() {
            // Accumulate, do not replace, gradients here.  Several later
            // layers can feed the same earlier layer through route/shortcut
            // fan-in, matching C++'s additive `layer.delta` behavior.
            let grad = layer_output_gradients[layer.index].clone();
            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing saved forward output for layer #{} input",
                        layer.index
                    ))
                })?
            };

            match (&layer.kind, params) {
                (
                    LayerKind::HyperbolicFoveation(spec),
                    LayerParams::HyperbolicFoveation(params),
                ) => {
                    let cache = forward
                        .hyperbolic_foveation_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [hyperbolic_foveation] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let warp_input = match spec.warp_source {
                        crate::model::HfWarpSource::Previous => current_input,
                        crate::model::HfWarpSource::NetworkInput => input,
                    };
                    let mut auxiliary_gradient = hf_auxiliary_param_gradients
                        .and_then(|gradients| gradients.get(layer.index))
                        .and_then(Option::as_ref)
                        .cloned();
                    if let Some(film_gradient) = foveation_hf_auxiliary_gradients
                        .get(layer.index)
                        .and_then(Option::as_ref)
                    {
                        let combined = auxiliary_gradient
                            .get_or_insert_with(|| vec![0.0; film_gradient.len()]);
                        if combined.len() != film_gradient.len() {
                            return Err(Error::InvalidConfig(format!(
                                "[foveation_film] HF auxiliary gradient layout mismatch at layer #{}",
                                layer.index
                            )));
                        }
                        for (combined, film) in combined.iter_mut().zip(film_gradient) {
                            *combined += *film;
                        }
                    }
                    let (warp_gradient, predictor_gradient, param_gradient, _) =
                        backward_hyperbolic_foveation(
                            spec,
                            params,
                            warp_input,
                            &grad,
                            cache,
                            auxiliary_gradient.as_deref(),
                            None,
                            None,
                            false,
                        )?;
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [hyperbolic_foveation] parameter-gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::HyperbolicFoveation(param_gradient);
                    match spec.warp_source {
                        crate::model::HfWarpSource::Previous => accumulate_previous_gradient(
                            layer.index,
                            &warp_gradient,
                            &mut input_gradient,
                            &mut layer_output_gradients,
                        )?,
                        crate::model::HfWarpSource::NetworkInput => {
                            accumulate_tensor(&mut input_gradient, &warp_gradient)?
                        }
                    }
                    if let Some((source_layer, source_gradient)) = predictor_gradient {
                        if let Some(source_layer) = source_layer {
                            let Some(destination) = layer_output_gradients.get_mut(source_layer)
                            else {
                                return Err(Error::InvalidConfig(format!(
                                    "[hyperbolic_foveation] predictor source layer #{source_layer} is unavailable during backward"
                                )));
                            };
                            accumulate_tensor(destination, &source_gradient)?;
                        } else {
                            accumulate_previous_gradient(
                                layer.index,
                                &source_gradient,
                                &mut input_gradient,
                                &mut layer_output_gradients,
                            )?;
                        }
                    }
                }
                (LayerKind::FoveationFilm(spec), LayerParams::FoveationFilm(params)) => {
                    let film_cache = forward
                        .foveation_film_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [foveation_film] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let hf_cache = forward
                        .hyperbolic_foveation_caches
                        .get(spec.hf_layer)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "[foveation_film] selected HF layer #{} has no forward cache",
                                spec.hf_layer
                            ))
                        })?;
                    let gradients = crate::foveation_film::backward_nchw(
                        &current_input.data,
                        &grad.data,
                        &hf_cache.params_per_batch,
                        layer.input_shape.batch,
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        spec.param_count,
                        spec.embed_dim,
                        params,
                        film_cache,
                    )?;
                    if spec.backprop_to_hf {
                        let Some(slot) = foveation_hf_auxiliary_gradients.get_mut(spec.hf_layer)
                        else {
                            return Err(Error::InvalidConfig(format!(
                                "[foveation_film] selected HF layer #{} is unavailable during backward",
                                spec.hf_layer
                            )));
                        };
                        if let Some(existing) = slot {
                            if existing.len() != gradients.hf_params.len() {
                                return Err(Error::InvalidConfig(format!(
                                    "[foveation_film] selected HF layer #{} gradient layout mismatch",
                                    spec.hf_layer
                                )));
                            }
                            for (sum, value) in existing.iter_mut().zip(&gradients.hf_params) {
                                *sum += *value;
                            }
                        } else {
                            *slot = Some(gradients.hf_params.clone());
                        }
                    }
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [foveation_film] parameter-gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::FoveationFilm(gradients.params);
                    let layer_input_gradient =
                        Tensor::from_data(layer.input_shape, gradients.input)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &layer_input_gradient,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(params)) => {
                    let cache = forward
                        .recursive_block_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [recursive_block] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let (layer_input_gradient, layer_param_gradient) = backward_recursive_block(
                        &self.network,
                        self.train_iteration,
                        layer,
                        spec,
                        params,
                        current_input,
                        &grad,
                        cache,
                    )?;
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [recursive_block] parameter-gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = layer_param_gradient;
                    accumulate_previous_gradient(
                        layer.index,
                        &layer_input_gradient,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let try_fix_nan = self.network.net.try_fix_nan;
                    // C++ `try_fix_nan`: clamp the output delta to [-1, 1]
                    // before the activation gradient, reset the input delta after.
                    let grad = if try_fix_nan {
                        Tensor {
                            shape: grad.shape,
                            data: lightnet_cuda::host_ops::constrain(&grad.data, 1.0),
                        }
                    } else {
                        grad
                    };
                    let (mut input_grad, layer_param_gradients) = backward_convolutional(
                        spec,
                        conv,
                        current_input,
                        layer.output_shape,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                        workspace,
                    )?;
                    if try_fix_nan {
                        input_grad.data = lightnet_cuda::host_ops::sanitize_nan_inf(
                            &input_grad.data,
                            NanInfMode::Reset,
                        );
                    }
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    accumulate_convolutional_param_gradients(
                        param_gradients.get_mut(owner).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing convolutional gradient slot for owner layer #{owner}"
                            ))
                        })?,
                        &layer_param_gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::GraphConv(spec), LayerParams::GraphConv(params)) => {
                    let cache = forward
                        .graph_conv_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [graph_conv] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let preactivation = forward.layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [graph_conv] activation cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let output = &forward.layer_outputs[layer.index];
                    let activation_gradient =
                        if let Some(normalization) = spec.activation.to_channel_normalization() {
                            Tensor {
                                shape: layer.output_shape,
                                data: lightnet_cuda::host_ops::normalize_channels_backward(
                                    &preactivation.data,
                                    &output.data,
                                    &grad.data,
                                    layer.output_shape.batch,
                                    layer.output_shape.channels,
                                    layer.output_shape.width * layer.output_shape.height,
                                    normalization,
                                    grad_mode == GradMode::Cpp,
                                ),
                            }
                        } else {
                            apply_output_activation_gradient(
                                output,
                                &grad,
                                &spec.activation,
                                "graph_conv",
                                Some(preactivation),
                            )?
                        };
                    let (raw_gradient, bias_gradient, batch_norm_gradient) =
                        if let Some(batch_norm) = &params.projection.batch_norm {
                            let mut effective_batch_norm = batch_norm.clone();
                            effective_batch_norm
                                .biases
                                .clone_from(&params.projection.biases);
                            let (raw_gradient, mut batch_norm_gradient) = backward_batchnorm(
                                &effective_batch_norm,
                                &cache.math.raw_output,
                                &activation_gradient,
                                cache.training,
                            )?;
                            let bias_gradient = std::mem::take(&mut batch_norm_gradient.biases);
                            batch_norm_gradient.biases.fill(0.0);
                            (raw_gradient, bias_gradient, Some(batch_norm_gradient))
                        } else {
                            let mut bias_gradient = vec![0.0; spec.filters];
                            for batch in 0..layer.output_shape.batch {
                                for filter in 0..spec.filters {
                                    for y in 0..layer.output_shape.height {
                                        for x in 0..layer.output_shape.width {
                                            bias_gradient[filter] +=
                                                activation_gradient.get(batch, filter, y, x);
                                        }
                                    }
                                }
                            }
                            (activation_gradient, bias_gradient, None)
                        };
                    let math_gradient = crate::graph_conv::graph_conv_backward(
                        spec,
                        current_input,
                        &params.projection.weights,
                        params.self_weights.as_deref(),
                        params.edge_kernel.as_deref(),
                        &raw_gradient,
                        &cache.math,
                    )?;
                    let gradient = GraphConvParams {
                        projection: ConvolutionalParams {
                            biases: bias_gradient,
                            batch_norm: batch_norm_gradient,
                            weights: math_gradient.weights,
                        },
                        self_weights: math_gradient.self_weights,
                        edge_kernel: math_gradient.edge_kernel,
                        edge_biases: math_gradient.edge_biases,
                    };
                    let Some(LayerParams::GraphConv(dst)) = param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [graph_conv] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *dst = gradient;
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current_input.shape, math_gradient.input)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params)) => {
                    let cache = forward
                        .dcnv4_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [dcnv4] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let preactivation = forward.layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [dcnv4] activation cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let output = &forward.layer_outputs[layer.index];
                    let delta =
                        if let Some(normalization) = spec.activation.to_channel_normalization() {
                            Tensor {
                                shape: layer.output_shape,
                                data: lightnet_cuda::host_ops::normalize_channels_backward(
                                    &preactivation.data,
                                    &output.data,
                                    &grad.data,
                                    layer.output_shape.batch,
                                    layer.output_shape.channels,
                                    layer.output_shape.width * layer.output_shape.height,
                                    normalization,
                                    grad_mode == GradMode::Cpp,
                                ),
                            }
                        } else {
                            apply_output_activation_gradient(
                                output,
                                &grad,
                                &spec.activation,
                                "dcnv4",
                                Some(preactivation),
                            )?
                        };
                    let mut aggregation_gradient = delta.data.clone();
                    let mut biases_gradient = vec![0.0; params.biases.len()];
                    let mut batch_norm_gradient =
                        params.batch_norm.as_ref().map(|bn| BatchNormParams {
                            biases: vec![0.0; bn.biases.len()],
                            scales: vec![0.0; bn.scales.len()],
                            rolling_mean: vec![0.0; bn.rolling_mean.len()],
                            rolling_variance: vec![0.0; bn.rolling_variance.len()],
                        });
                    let spatial = layer.output_shape.width * layer.output_shape.height;
                    for b in 0..layer.output_shape.batch {
                        for channel in 0..layer.output_shape.channels {
                            for q in 0..spatial {
                                let index =
                                    (b * layer.output_shape.channels + channel) * spatial + q;
                                if let Some(bn) = params.batch_norm.as_ref() {
                                    let inv_std = 1.0
                                        / (bn.rolling_variance[channel] + BATCH_NORM_EPSILON)
                                            .sqrt();
                                    let normalized = (cache.raw_output[index]
                                        - bn.rolling_mean[channel])
                                        * inv_std;
                                    batch_norm_gradient
                                        .as_mut()
                                        .expect("batch-norm gradient exists")
                                        .scales[channel] += delta.data[index] * normalized;
                                    aggregation_gradient[index] *= bn.scales[channel] * inv_std;
                                } else {
                                    biases_gradient[channel] += delta.data[index];
                                }
                            }
                        }
                    }
                    let math_gradient = crate::dcnv4::dcnv4_backward(
                        &current_input.data,
                        current_input.shape,
                        spec,
                        &params.offset_weights,
                        cache,
                        &aggregation_gradient,
                    )?;
                    let gradients = Dcnv4Params {
                        biases: biases_gradient,
                        batch_norm: batch_norm_gradient,
                        dummy_weights: vec![0.0; params.dummy_weights.len()],
                        offset_weights: math_gradient.offset_weights,
                        offset_biases: math_gradient.offset_biases,
                    };
                    let Some(LayerParams::Dcnv4(dst)) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [dcnv4] parameter-gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *dst = gradients;
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current_input.shape, math_gradient.input)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DeformConv(spec), LayerParams::DeformConv(params)) => {
                    let cache = forward
                        .deform_conv_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [deform_conv] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let preactivation = forward.layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [deform_conv] activation cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    if spec.activation.to_channel_normalization().is_none()
                        && grad
                            .data
                            .iter()
                            .zip(&cache.output_nondiff_boundary)
                            .any(|(gradient, boundary)| *gradient != 0.0 && *boundary)
                    {
                        return Err(Error::InvalidConfig(
                            "[deform_conv] output derivative is undefined at a nonfinite or finite-clamp boundary".to_owned(),
                        ));
                    }
                    let delta =
                        if let Some(normalization) = spec.activation.to_channel_normalization() {
                            Tensor {
                                shape: layer.output_shape,
                                data: lightnet_cuda::host_ops::normalize_channels_backward(
                                    &preactivation.data,
                                    &forward.layer_outputs[layer.index].data,
                                    &grad.data,
                                    layer.output_shape.batch,
                                    layer.output_shape.channels,
                                    layer.output_shape.width * layer.output_shape.height,
                                    normalization,
                                    grad_mode == GradMode::Cpp,
                                ),
                            }
                        } else {
                            let mut clamped_grad = grad.clone();
                            for (value, slope) in
                                clamped_grad.data.iter_mut().zip(&cache.output_clamp_slope)
                            {
                                *value *= *slope;
                            }
                            apply_output_activation_gradient(
                                &forward.layer_outputs[layer.index],
                                &clamped_grad,
                                &spec.activation,
                                "deform_conv",
                                Some(preactivation),
                            )?
                        };
                    let spatial = layer.output_shape.width * layer.output_shape.height;
                    let mut raw_gradient = delta.clone();
                    let mut main_bias_gradient = vec![0.0; params.main.biases.len()];
                    let mut batch_norm_gradient =
                        params.main.batch_norm.as_ref().map(|bn| BatchNormParams {
                            biases: vec![0.0; bn.biases.len()],
                            scales: vec![0.0; bn.scales.len()],
                            rolling_mean: vec![0.0; bn.rolling_mean.len()],
                            rolling_variance: vec![0.0; bn.rolling_variance.len()],
                        });
                    for b in 0..layer.output_shape.batch {
                        for channel in 0..layer.output_shape.channels {
                            let inv_std = params.main.batch_norm.as_ref().map(|bn| {
                                1.0 / (bn.rolling_variance[channel] + BATCH_NORM_EPSILON).sqrt()
                            });
                            for q in 0..spatial {
                                let index =
                                    (b * layer.output_shape.channels + channel) * spatial + q;
                                let dy = delta.data[index];
                                main_bias_gradient[channel] += dy;
                                if let (Some(bn), Some(inv_std)) =
                                    (params.main.batch_norm.as_ref(), inv_std)
                                {
                                    let normalized = (cache.math.raw_output[index]
                                        - bn.rolling_mean[channel])
                                        * inv_std;
                                    batch_norm_gradient
                                        .as_mut()
                                        .expect("BN gradient allocated")
                                        .scales[channel] += dy * normalized;
                                    raw_gradient.data[index] *= bn.scales[channel] * inv_std;
                                }
                            }
                        }
                    }
                    let math_gradient = crate::deform_conv::deform_conv_backward(
                        current_input.shape,
                        spec,
                        layer.output_shape,
                        &cache.math,
                        &params.main.weights,
                        &params.offset_weights,
                        params.mask_weights.as_deref(),
                        &raw_gradient,
                    )?;
                    let gradient = DeformConvParams {
                        main: ConvolutionalParams {
                            biases: main_bias_gradient,
                            batch_norm: batch_norm_gradient,
                            weights: math_gradient.weights,
                        },
                        offset_weights: math_gradient.offset_weights,
                        offset_biases: math_gradient.offset_biases,
                        mask_weights: math_gradient.mask_weights,
                        mask_biases: math_gradient.mask_biases,
                    };
                    let Some(LayerParams::DeformConv(dst)) = param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [deform_conv] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *dst = gradient;
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current_input.shape, math_gradient.input)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DsConv(spec), LayerParams::DsConv(params)) => {
                    let cache = forward
                        .dsconv_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [dsconv] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let preactivation = forward.layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [dsconv] activation cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let delta =
                        if let Some(normalization) = spec.activation.to_channel_normalization() {
                            Tensor {
                                shape: layer.output_shape,
                                data: lightnet_cuda::host_ops::normalize_channels_backward(
                                    &preactivation.data,
                                    &forward.layer_outputs[layer.index].data,
                                    &grad.data,
                                    layer.output_shape.batch,
                                    layer.output_shape.channels,
                                    layer.output_shape.width * layer.output_shape.height,
                                    normalization,
                                    grad_mode == GradMode::Cpp,
                                ),
                            }
                        } else {
                            apply_output_activation_gradient(
                                &forward.layer_outputs[layer.index],
                                &grad,
                                &spec.activation,
                                "dsconv",
                                Some(preactivation),
                            )?
                        };
                    let math_gradient = crate::dsconv::dsconv_backward(
                        current_input,
                        spec,
                        &params.collapse.weights,
                        &params.offset_weights,
                        &cache.math,
                        &delta.data,
                    )?;
                    let gradient = DsConvParams {
                        collapse: ConvolutionalParams {
                            biases: math_gradient.biases,
                            batch_norm: None,
                            weights: math_gradient.weights,
                        },
                        offset_weights: math_gradient.offset_weights,
                        offset_biases: math_gradient.offset_biases,
                    };
                    let Some(LayerParams::DsConv(dst)) = param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [dsconv] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *dst = gradient;
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current_input.shape, math_gradient.input)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Wmhf(spec), LayerParams::Wmhf(params)) => {
                    let cache = forward
                        .wmhf_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [wmhf] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let preactivation = forward.layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [wmhf] activation cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let delta = apply_output_activation_gradient(
                        &forward.layer_outputs[layer.index],
                        &grad,
                        &spec.activation,
                        "wmhf",
                        Some(preactivation),
                    )?;
                    let use_shortcut = spec.shortcut
                        && current_input.shape.channels == spec.filters
                        && current_input.shape == layer.output_shape;
                    let gate_gradient = crate::wmhf::apply_gate_backward(
                        &delta.data,
                        &cache.hf_gate.output.data,
                        &cache.pre.output.data,
                        use_shortcut,
                        spec.freq_scale,
                    )?;
                    let spatial = layer.output_shape.width * layer.output_shape.height;
                    let mut pre_gradient = Tensor::zeros(layer.output_shape);
                    crate::wmhf::insert_channels(
                        &gate_gradient.projected,
                        &mut pre_gradient.data,
                        layer.output_shape.batch,
                        spec.filters,
                        spec.filters,
                        spatial,
                        0,
                        1.0,
                    )?;

                    let fuse_spec = wmhf_internal_convolution_spec(
                        spec.filters,
                        1,
                        1,
                        spec.batch_normalize,
                        Activation::Linear,
                    );
                    let (fuse_input_gradient, fuse_param_gradient) = backward_wmhf_convolution(
                        &fuse_spec,
                        &params.fuse,
                        &cache.fuse,
                        &Tensor::from_data(layer.output_shape, gate_gradient.fuse)?,
                        cache.training,
                    )?;
                    let fuse_parts = crate::wmhf::split_fuse_gradient(
                        &fuse_input_gradient.data,
                        layer.output_shape.batch,
                        spec.filters,
                        spec.identity_channels,
                        spec.local_channels,
                        spatial,
                    )?;
                    crate::wmhf::insert_channels(
                        &fuse_parts.projected,
                        &mut pre_gradient.data,
                        layer.output_shape.batch,
                        spec.filters,
                        spec.filters,
                        spatial,
                        0,
                        1.0,
                    )?;

                    let gate_spec = wmhf_internal_convolution_spec(
                        spec.filters,
                        1,
                        1,
                        false,
                        Activation::Logistic,
                    );
                    let (energy_gradient, gate_param_gradient) = backward_wmhf_convolution(
                        &gate_spec,
                        &params.hf_gate,
                        &cache.hf_gate,
                        &Tensor::from_data(layer.output_shape, gate_gradient.gate)?,
                        cache.training,
                    )?;

                    let local_shape = cache.local_mix.output.shape;
                    let local_mix_spec = wmhf_internal_convolution_spec(
                        spec.local_channels,
                        1,
                        1,
                        spec.batch_normalize,
                        Activation::Linear,
                    );
                    let (local_cat_gradient, local_mix_param_gradient) = backward_wmhf_convolution(
                        &local_mix_spec,
                        &params.local_mix,
                        &cache.local_mix,
                        &Tensor::from_data(local_shape, fuse_parts.local)?,
                        cache.training,
                    )?;
                    let (dw3_gradient, dw5_gradient, dw7_gradient) =
                        crate::wmhf::split_local_gradient(
                            &local_cat_gradient.data,
                            local_shape.batch,
                            spec.local_channels,
                            spatial,
                        )?;
                    let dw3_spec = wmhf_internal_convolution_spec(
                        spec.local_channels,
                        3,
                        spec.local_channels,
                        false,
                        Activation::Linear,
                    );
                    let dw5_spec = wmhf_internal_convolution_spec(
                        spec.local_channels,
                        5,
                        spec.local_channels,
                        false,
                        Activation::Linear,
                    );
                    let dw7_spec = wmhf_internal_convolution_spec(
                        spec.local_channels,
                        7,
                        spec.local_channels,
                        false,
                        Activation::Linear,
                    );
                    let (mut local_input_gradient, dw3_param_gradient) = backward_wmhf_convolution(
                        &dw3_spec,
                        &params.local_dw3,
                        &cache.local_dw3,
                        &Tensor::from_data(local_shape, dw3_gradient)?,
                        cache.training,
                    )?;
                    let (dw5_input_gradient, dw5_param_gradient) = backward_wmhf_convolution(
                        &dw5_spec,
                        &params.local_dw5,
                        &cache.local_dw5,
                        &Tensor::from_data(local_shape, dw5_gradient)?,
                        cache.training,
                    )?;
                    let (dw7_input_gradient, dw7_param_gradient) = backward_wmhf_convolution(
                        &dw7_spec,
                        &params.local_dw7,
                        &cache.local_dw7,
                        &Tensor::from_data(local_shape, dw7_gradient)?,
                        cache.training,
                    )?;
                    for (dst, src) in local_input_gradient
                        .data
                        .iter_mut()
                        .zip(dw5_input_gradient.data.iter().zip(&dw7_input_gradient.data))
                    {
                        *dst += src.0 + src.1;
                    }
                    crate::wmhf::insert_channels(
                        &local_input_gradient.data,
                        &mut pre_gradient.data,
                        layer.output_shape.batch,
                        spec.filters,
                        spec.local_channels,
                        spatial,
                        spec.identity_channels,
                        1.0,
                    )?;

                    let wavelet_output_gradient = crate::wmhf::haar_iwt_backward(
                        &fuse_parts.global,
                        layer.output_shape.batch,
                        spec.global_channels,
                        layer.output_shape.height,
                        layer.output_shape.width,
                    )?;
                    let energy_coeff_gradient = crate::wmhf::high_frequency_energy_backward(
                        &energy_gradient.data,
                        &cache.dwt,
                    )?;
                    let mut dwt_output_gradient = wavelet_output_gradient;
                    for (dst, src) in dwt_output_gradient
                        .lh
                        .iter_mut()
                        .zip(&energy_coeff_gradient.lh)
                    {
                        *dst += *src;
                    }
                    for (dst, src) in dwt_output_gradient
                        .hl
                        .iter_mut()
                        .zip(&energy_coeff_gradient.hl)
                    {
                        *dst += *src;
                    }
                    for (dst, src) in dwt_output_gradient
                        .hh
                        .iter_mut()
                        .zip(&energy_coeff_gradient.hh)
                    {
                        *dst += *src;
                    }
                    let (scan_input_gradient, scan_weight_gradient) = crate::wmhf::scan4_backward(
                        &cache.dwt.ll,
                        &dwt_output_gradient.ll,
                        &params.scan_weights,
                        layer.output_shape.batch,
                        spec.global_channels,
                        cache.dwt.height2,
                        cache.dwt.width2,
                    )?;
                    dwt_output_gradient.ll = scan_input_gradient;
                    let global_input_gradient =
                        crate::wmhf::haar_dwt_backward(&dwt_output_gradient)?;
                    crate::wmhf::insert_channels(
                        &global_input_gradient,
                        &mut pre_gradient.data,
                        layer.output_shape.batch,
                        spec.filters,
                        spec.global_channels,
                        spatial,
                        spec.identity_channels + spec.local_channels,
                        1.0,
                    )?;

                    let pre_spec = wmhf_internal_convolution_spec(
                        spec.filters,
                        1,
                        1,
                        spec.batch_normalize,
                        Activation::Linear,
                    );
                    let (mut pre_input_gradient, pre_param_gradient) = backward_wmhf_convolution(
                        &pre_spec,
                        &params.pre,
                        &cache.pre,
                        &pre_gradient,
                        cache.training,
                    )?;
                    if use_shortcut {
                        for (dst, src) in pre_input_gradient.data.iter_mut().zip(&delta.data) {
                            *dst += *src;
                        }
                    }
                    let gradient = WmhfParams {
                        pre: pre_param_gradient,
                        local_dw3: dw3_param_gradient,
                        local_dw5: dw5_param_gradient,
                        local_dw7: dw7_param_gradient,
                        local_mix: local_mix_param_gradient,
                        hf_gate: gate_param_gradient,
                        fuse: fuse_param_gradient,
                        scan_weights: scan_weight_gradient,
                    };
                    let Some(LayerParams::Wmhf(dst)) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [wmhf] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *dst = gradient;
                    accumulate_previous_gradient(
                        layer.index,
                        &pre_input_gradient,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Clifford(spec), LayerParams::Clifford(params)) => {
                    let cache = forward
                        .clifford_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [clifford] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let (input_grad, parameter_grad) =
                        crate::clifford::backward(spec, params, cache, &grad)?;
                    let Some(LayerParams::Clifford(destination)) =
                        param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [clifford] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *destination = parameter_grad;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                // `[moe_convolutional]`, ported from C++
                // `backward_moe_convolutional_layer`.  Three gradients meet here:
                //   1. dL/dgate[b,e] = dot(delta[b], expert_e(x)[b]), seeded with
                //      the load-balance term;
                //   2. the softmax/temperature Jacobian turns that into dlogit,
                //      which feeds both the router parameters and the pooled
                //      descriptor the router consumed;
                //   3. each expert sees `gate[b,e] * delta[b]`, because the
                //      layer output is a gate-weighted sum of the experts.
                (LayerKind::Moe(spec), LayerParams::Moe(params)) => {
                    let cache = forward
                        .moe_caches
                        .get(layer.index)
                        .and_then(|slot| slot.as_ref())
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [moe_convolutional] forward cache for layer #{}; backward needs its gates and expert outputs",
                                layer.index
                            ))
                        })?;
                    if grad.shape != layer.output_shape {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} ([moe_convolutional]) gradient shape mismatch: expected {}, got {}",
                            layer.index, layer.output_shape, grad.shape
                        )));
                    }
                    let batch = current_input.shape.batch;
                    let channels = current_input.shape.channels;
                    let experts = spec.experts;
                    let per_image = layer.output_shape.element_count();
                    if experts == 0 || cache.gates.len() != batch * experts {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} ([moe_convolutional]) cached {} gates for batch {batch} x {experts} experts",
                            layer.index,
                            cache.gates.len()
                        )));
                    }

                    // 1. Load-balance gradient, then the mixture gradient.
                    let mut raw =
                        crate::moe_conv::balance_loss_and_raw_grad(&cache.gates, batch, spec).1;
                    for (index, value) in raw.iter_mut().enumerate() {
                        let b = index / experts;
                        let expert = index % experts;
                        let expert_out = cache.expert_outputs.get(expert).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "layer #{} ([moe_convolutional]) is missing cached expert {expert}",
                                layer.index
                            ))
                        })?;
                        let base = b * per_image;
                        let delta = &grad.data[base..base + per_image];
                        let out = &expert_out.data[base..base + per_image];
                        *value += delta.iter().zip(out).map(|(d, o)| d * o).sum::<f32>();
                    }

                    // 2. Softmax + temperature backward.
                    let dlogit = crate::moe_conv::router_gate_gradient(&cache.gates, &raw, spec);

                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [moe_convolutional] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    // As everywhere else in this file the gradient slot reuses
                    // `MoeParams`, so `router_biases`/`router_weights` hold
                    // gradient deltas here rather than parameter values.
                    let gradients = match slot {
                        LayerParams::Moe(gradients) => gradients,
                        _ => {
                            return Err(Error::InvalidConfig(format!(
                                "layer #{} ([moe_convolutional]) gradient slot does not hold MoE parameters",
                                layer.index
                            )));
                        }
                    };
                    if gradients.router_biases.len() != experts
                        || gradients.router_weights.len() != experts * channels
                    {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} ([moe_convolutional]) router gradient is sized {} x {}, not {experts} x {channels}",
                            layer.index,
                            gradients.router_biases.len(),
                            gradients.router_weights.len() / channels.max(1)
                        )));
                    }
                    for e in 0..experts {
                        let mut sum = 0.0f32;
                        for b in 0..batch {
                            sum += dlogit[b * experts + e];
                        }
                        gradients.router_biases[e] += sum;
                        for c in 0..channels {
                            let mut sum = 0.0f32;
                            for b in 0..batch {
                                sum += dlogit[b * experts + e] * cache.pooled[b * channels + c];
                            }
                            gradients.router_weights[e * channels + c] += sum;
                        }
                    }

                    // The router reads a global average pool, so its input
                    // gradient spreads each channel's pooled gradient evenly
                    // over that channel's spatial positions.
                    let mut pooled_grad = vec![0.0f32; batch * channels];
                    for b in 0..batch {
                        for e in 0..experts {
                            let value = dlogit[b * experts + e];
                            for c in 0..channels {
                                pooled_grad[b * channels + c] +=
                                    value * params.router_weights[e * channels + c];
                            }
                        }
                    }
                    let spatial = current_input.shape.width * current_input.shape.height;
                    if spatial > 0 {
                        let inv_spatial = 1.0 / spatial as f32;
                        let mut input_grad = vec![0.0f32; current_input.data.len()];
                        for b in 0..batch {
                            for c in 0..channels {
                                let share = pooled_grad[b * channels + c] * inv_spatial;
                                let base = b * channels * spatial + c * spatial;
                                for dst in input_grad[base..base + spatial].iter_mut() {
                                    *dst += share;
                                }
                            }
                        }
                        accumulate_previous_gradient(
                            layer.index,
                            &Tensor::from_data(current_input.shape, input_grad)?,
                            &mut input_gradient,
                            &mut layer_output_gradients,
                        )?;
                    }

                    // 3. Scaled expert gradients.
                    for e in 0..experts {
                        let expert = params.experts.get(e).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "layer #{} ([moe_convolutional]) is missing expert {e}",
                                layer.index
                            ))
                        })?;
                        let mut expert_grad = vec![0.0f32; grad.data.len()];
                        for b in 0..batch {
                            let gate = cache.gates[b * experts + e];
                            let base = b * per_image;
                            let slice = base..base + per_image;
                            for (dst, src) in expert_grad[slice.clone()]
                                .iter_mut()
                                .zip(grad.data[slice].iter())
                            {
                                *dst = gate * src;
                            }
                        }
                        let (expert_input_grad, expert_param_gradients) = backward_convolutional(
                            &spec.expert,
                            expert,
                            current_input,
                            layer.output_shape,
                            &Tensor::from_data(layer.output_shape, expert_grad)?,
                            cache
                                .expert_activation_inputs
                                .get(e)
                                .and_then(|t| t.as_ref()),
                            workspace,
                        )?;
                        let LayerParams::Moe(gradients) = slot else {
                            unreachable!("the slot was matched as MoE above");
                        };
                        let Some(expert_gradients) = gradients.experts.get_mut(e) else {
                            return Err(Error::InvalidConfig(format!(
                                "layer #{} ([moe_convolutional]) is missing expert {e} gradients",
                                layer.index
                            )));
                        };
                        accumulate_convolutional_params(expert_gradients, &expert_param_gradients)?;
                        accumulate_previous_gradient(
                            layer.index,
                            &expert_input_grad,
                            &mut input_gradient,
                            &mut layer_output_gradients,
                        )?;
                    }
                }

                (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(params)) => {
                    let cache = forward
                        .eml_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [eml_convolutional] branch cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let math = crate::eml_conv_math::EmlConvMathParams::from_cpp_values(
                        spec.eml_clamp,
                        spec.eml_eps,
                        spec.eml_scale,
                    );
                    let use_residual = spec.residual
                        && layer.input_shape.element_count() == layer.output_shape.element_count();
                    let branch_grad = crate::eml_conv_math::backward(
                        &cache.input_branch_output.data,
                        &cache.self_branch_output.data,
                        &grad.data,
                        use_residual.then_some(current_input.data.as_slice()),
                        math,
                    )
                    .map_err(|error| Error::InvalidConfig(error.to_owned()))?;
                    let x_delta = Tensor::from_data(layer.output_shape, branch_grad.x_branch)?;
                    let y_delta = Tensor::from_data(layer.output_shape, branch_grad.self_branch)?;
                    let (mut input_grad, input_param_grad) = backward_convolutional(
                        &spec.branch,
                        &params.input,
                        current_input,
                        layer.output_shape,
                        &x_delta,
                        None,
                        workspace,
                    )?;
                    let (self_input_grad, self_param_grad) = backward_convolutional(
                        &spec.branch,
                        &params.self_params,
                        current_input,
                        layer.output_shape,
                        &y_delta,
                        None,
                        workspace,
                    )?;
                    for (dst, src) in input_grad.data.iter_mut().zip(&self_input_grad.data) {
                        *dst += src;
                    }
                    if let Some(residual_grad) = branch_grad.residual {
                        for (dst, src) in input_grad.data.iter_mut().zip(residual_grad) {
                            *dst += src;
                        }
                    }
                    let Some(LayerParams::EmlConvolutional(gradients)) =
                        param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing EML parameter-gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_convolutional_params(&mut gradients.input, &input_param_grad)?;
                    accumulate_convolutional_params(&mut gradients.self_params, &self_param_grad)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let delta = apply_output_activation_gradient(
                        &forward.layer_outputs[layer.index],
                        &grad,
                        &spec.activation,
                        "deconvolutional",
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    // C++ also scales the weight gradient by 1/batch (and the optimizer divides again).
                    let weight_scale = if grad_mode == GradMode::Cpp {
                        1.0 / current_input.shape.batch as f32
                    } else {
                        1.0
                    };
                    let g = crate::deconvolutional::deconv_backward(
                        current_input,
                        &p.weights,
                        &delta,
                        spec.size,
                        spec.stride,
                        weight_scale,
                    );
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing deconvolutional gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::Convolutional(ConvolutionalParams {
                        biases: g.biases,
                        batch_norm: None,
                        weights: g.weights,
                    });
                    accumulate_previous_gradient(
                        layer.index,
                        &g.input,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MaxPool(spec), LayerParams::None) if spec.depth => {
                    let input_grad = backward_maxpool_depth(current_input, &grad)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    let input_grad = backward_maxpool(
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.padding,
                        current_input,
                        &grad,
                        false,
                        spec.zero_nonmax,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    let input_grad = backward_maxpool(
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.padding,
                        current_input,
                        &grad,
                        true,
                        false,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::AvgPool, LayerParams::None) => {
                    let input_grad = backward_avgpool(&grad, current_input.shape)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Route(spec), LayerParams::None) => {
                    backward_route(
                        spec,
                        layer.index,
                        &grad,
                        &mut layer_output_gradients,
                        &self.network.layers,
                    )?;
                }
                (LayerKind::LayaMarkerGather(spec), LayerParams::None) => {
                    backward_laya_marker_gather(
                        spec,
                        layer.index,
                        &forward.layer_outputs,
                        &grad,
                        &mut layer_output_gradients,
                        &self.network.layers,
                    )?;
                }
                (LayerKind::LayaActHead(spec), LayerParams::LayaActHead(weights)) => {
                    backward_laya_act_head(
                        spec,
                        weights,
                        layer.index,
                        &forward.layer_outputs,
                        &grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => {
                    let output = &forward.layer_outputs[layer.index];
                    let delta = apply_output_activation_gradient(
                        output,
                        &grad,
                        &spec.activation,
                        "shortcut",
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    let sources = shortcut_sources(
                        spec,
                        layer.index,
                        &forward.layer_outputs,
                        current_input.shape.batch,
                    )?;
                    let slices: Vec<&[f32]> = sources.iter().map(|t| t.data.as_slice()).collect();
                    let g = crate::shortcut_weights::weighted_backward(
                        &delta.data,
                        &current_input.data,
                        &slices,
                        current_input.shape.batch,
                        current_input.shape.element_count(),
                        &params.weights,
                        spec.weights_normalization,
                        spec.weight_slots(current_input.shape.channels),
                        grad_mode == GradMode::Cpp,
                    );
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing shortcut gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::Shortcut(ShortcutParams { weights: g.weights });
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current_input.shape, g.input)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    for (raw_index, source_grad) in spec.from.iter().zip(g.sources) {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let shape = forward.layer_outputs[source_index].shape;
                        accumulate_tensor(
                            &mut layer_output_gradients[source_index],
                            &Tensor::from_data(shape, source_grad)?,
                        )?;
                    }
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let accum_grad = backward_shortcut(
                        spec,
                        layer.index,
                        &forward.layer_outputs,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &accum_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source_shape = self
                            .network
                            .layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            })?
                            .output_shape;
                        let source_gradient = shortcut_source_gradient(&accum_grad, source_shape)?;
                        accumulate_tensor(
                            &mut layer_output_gradients[source_index],
                            &source_gradient,
                        )?;
                    }
                }
                (LayerKind::Upsample(spec), LayerParams::None) => {
                    let input_grad = backward_upsample(spec, &grad, current_input.shape)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing connected output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (input_grad, layer_param_gradients) = backward_connected(
                        spec,
                        connected,
                        current_input,
                        layer.output_shape,
                        &grad,
                        output,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                        workspace,
                    )?;
                    accumulate_connected_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing connected gradient slot for layer #{}",
                                layer.index
                            ))
                        })?,
                        &layer_param_gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Local(spec), LayerParams::Local(local)) => {
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing local output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (input_grad, layer_param_gradients) = backward_local(
                        spec,
                        local,
                        current_input,
                        layer.output_shape,
                        &grad,
                        output,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing local gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::Local(layer_param_gradients);
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(batch_norm)) => {
                    let (input_grad, layer_param_gradients) = backward_batchnorm(
                        batch_norm,
                        current_input,
                        &grad,
                        forward.layer_activation_inputs[layer.index].is_some(),
                    )?;
                    let Some(slot) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing batch-norm gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = LayerParams::BatchNorm(layer_param_gradients);
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crop(_), LayerParams::None) => {
                    let cache = forward.crop_caches[layer.index].as_ref().ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing crop forward cache for layer #{}",
                            layer.index
                        ))
                    })?;
                    let input_grad = backward_crop(cache, current_input.shape, &grad)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Reorg(spec), LayerParams::None) => {
                    let input_grad = backward_reorg(spec, &grad, current_input.shape)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_shape = self
                        .network
                        .layers
                        .get(source_index)
                        .ok_or(Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.from,
                        })?
                        .output_shape;
                    let source_grad = crate::channel_slice::slice_channels_backward(
                        &grad,
                        source_shape,
                        spec.start,
                    )?;
                    accumulate_tensor(&mut layer_output_gradients[source_index], &source_grad)?;
                }
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => {
                    let input_grad =
                        crate::channel_shuffle::shuffle_channels(&grad, spec.groups, true)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Softmax { groups }, LayerParams::None) => {
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing softmax output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let input_grad = if grad_mode == GradMode::Cpp {
                        // C++ passes dy straight through (cost layer folds the Jacobian in).
                        grad.clone()
                    } else {
                        backward_softmax(output, &grad, *groups)?
                    };
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dropout { .. }, LayerParams::None) => {
                    let input_grad =
                        backward_dropout(&grad, forward.dropout_masks[layer.index].as_ref())?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source =
                        forward
                            .layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing scale_channels output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (mut input_grad, source_grad) = backward_scale_channels(
                        spec,
                        current_input,
                        source,
                        output,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    if spec.scale_wh && grad_mode == GradMode::Cpp {
                        let channel_size = (output.shape.height * output.shape.width) as f32;
                        for v in &mut input_grad.data {
                            *v /= channel_size;
                        }
                    }
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    accumulate_tensor(&mut layer_output_gradients[source_index], &source_grad)?;
                }
                (LayerKind::Sam(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source =
                        forward
                            .layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing sam output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let (input_grad, source_grad) = backward_sam(
                        spec,
                        current_input,
                        source,
                        output,
                        &grad,
                        forward.layer_activation_inputs[layer.index].as_ref(),
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                    accumulate_tensor(&mut layer_output_gradients[source_index], &source_grad)?;
                }
                (LayerKind::Region(spec), LayerParams::None) => {
                    let output = forward.layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing [region] output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let input_grad = backward_region_layer(spec, output, &grad)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::ObbYolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Logistic, LayerParams::None)
                | (LayerKind::Segmentation(_), LayerParams::None)
                | (LayerKind::OmdLoss(_), LayerParams::None)
                | (LayerKind::Contrastive(_), LayerParams::None)
                | (LayerKind::Yolox(_), LayerParams::None)
                | (LayerKind::Ppyoloe(_), LayerParams::None)
                | (LayerKind::Yolonas(_), LayerParams::None)
                | (LayerKind::CenterNet(_), LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None)
                | (LayerKind::JevCe(_), LayerParams::None)
                | (LayerKind::LayaLoss(_), LayerParams::None)
                | (LayerKind::Yolo3d(_), LayerParams::None) => {
                    accumulate_previous_gradient(
                        layer.index,
                        &grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Vit(_), LayerParams::Vit(weights)) => {
                    let cache = forward
                        .vit_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [vit] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let gradients = crate::vit::vit_backward_with_attention(
                        &grad.data,
                        weights,
                        cache,
                        |d_context, context, attention| {
                            attention_backward(d_context, context, attention)
                        },
                    )
                    .map_err(Error::InvalidConfig)?;
                    let Some(LayerParams::Vit(slot)) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [vit] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = crate::vit::VitWeights {
                        patch_embed: gradients.patch_embed,
                        patch_bias: gradients.patch_bias,
                        qkv_weight: gradients.qkv_weight,
                        qkv_bias: gradients.qkv_bias,
                        ln1_gamma: gradients.ln1_gamma,
                        ln1_beta: gradients.ln1_beta,
                        wo: gradients.wo,
                        wo_bias: gradients.wo_bias,
                        ln2_gamma: gradients.ln2_gamma,
                        ln2_beta: gradients.ln2_beta,
                        ffn_w1: gradients.ffn_w1,
                        ffn_b1: gradients.ffn_b1,
                        ffn_w2: gradients.ffn_w2,
                        ffn_b2: gradients.ffn_b2,
                        pos_embed: gradients.pos_embed,
                        mhc: gradients.mhc,
                    };
                    let input_grad = Tensor::from_data(current_input.shape, gradients.input)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Swin(_), LayerParams::Swin(weights)) => {
                    let cache = forward
                        .swin_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [swin] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let (gradients, input_grad) =
                        crate::swin_block::swin_backward(&grad.data, weights, cache)
                            .map_err(Error::InvalidConfig)?;
                    let Some(LayerParams::Swin(slot)) = param_gradients.get_mut(layer.index) else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [swin] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    *slot = gradients;
                    let input_grad = Tensor::from_data(current_input.shape, input_grad)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::TuckerAttention(_), LayerParams::TuckerAttention(weights)) => {
                    let cache = forward
                        .tucker_attention_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [tucker_attention] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let gradients = crate::tucker_attention::tucker_attention_backward(
                        &grad.data, weights, cache,
                    )
                    .map_err(Error::InvalidConfig)?;
                    let Some(LayerParams::TuckerAttention(slot)) =
                        param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [tucker_attention] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    let mut gradient_params = weights.clone();
                    gradient_params.biases = gradients.biases;
                    gradient_params.weights = gradients.weights;
                    *slot = gradient_params;
                    let input_grad = Tensor::from_data(current_input.shape, gradients.input)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MambaVision(_), LayerParams::MambaVision(weights)) => {
                    let cache = forward
                        .mambavision_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [mambavision] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let gradients =
                        crate::mambavision::mambavision_backward(&grad.data, weights, cache, None)
                            .map_err(Error::InvalidConfig)?;
                    let Some(LayerParams::MambaVision(slot)) = param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [mambavision] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    slot.values = gradients.params;
                    mambavision_state_gradients[layer.index] = Some(gradients.state_in);
                    let input_grad = Tensor::from_data(current_input.shape, gradients.input)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::KimiKda(_), LayerParams::KimiKda(weights)) => {
                    let cache = forward
                        .kimi_kda_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [kimi_kda] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    let gradients = crate::kimi_kda::kimi_kda_backward(&grad.data, weights, cache)
                        .map_err(Error::InvalidConfig)?;
                    let Some(LayerParams::KimiKda(slot)) = param_gradients.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [kimi_kda] gradient slot for layer #{}",
                            layer.index
                        )));
                    };
                    slot.biases = gradients.biases;
                    slot.weights = gradients.weights;
                    let input_grad = Tensor::from_data(current_input.shape, gradients.input)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DepthAnything(spec), LayerParams::DepthAnything(weights)) => {
                    let cache = forward
                        .depth_anything_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing [depth_anything] forward cache for layer #{}",
                                layer.index
                            ))
                        })?;
                    if let Some(da_input_gradient) = depth_anything_input_gradient(
                        grad_mode,
                        spec,
                        current_input.shape,
                        &grad.data,
                        weights,
                        cache,
                    )? {
                        accumulate_previous_gradient(
                            layer.index,
                            &da_input_gradient,
                            &mut input_gradient,
                            &mut layer_output_gradients,
                        )?;
                    }
                }
                (LayerKind::DptHead(spec), LayerParams::DptHead(weights)) => {
                    if grad_mode != GradMode::Cpp {
                        let cache = forward
                            .dpt_head_caches
                            .get(layer.index)
                            .and_then(Option::as_ref)
                            .ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing [dpt_head] forward cache for layer #{}",
                                    layer.index
                                ))
                            })?;
                        let gradients = crate::dpt_head::backward(
                            &grad.data,
                            weights,
                            spec.geometry(current_input.shape),
                            cache,
                        )?;
                        let Some(LayerParams::DptHead(slot)) = param_gradients.get_mut(layer.index)
                        else {
                            return Err(Error::InvalidConfig(format!(
                                "missing [dpt_head] gradient slot for layer #{}",
                                layer.index
                            )));
                        };
                        slot.values = gradients.weight_updates;
                        let input_grad = Tensor::from_data(current_input.shape, gradients.input)?;
                        accumulate_previous_gradient(
                            layer.index,
                            &input_grad,
                            &mut input_gradient,
                            &mut layer_output_gradients,
                        )?;
                    }
                }
                (LayerKind::LayaEncoder(_), LayerParams::LayaEncoder(_))
                | (LayerKind::LayaHead(_), LayerParams::LayaHead(_)) => {
                    // Frozen (`freeze=1`): no gradient flows to the input.
                }
                (LayerKind::LayaScorer(spec), LayerParams::LayaScorer(weights)) => {
                    let current = forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                        Error::InvalidConfig(format!("missing input for layer #{}", layer.index))
                    })?;
                    let input_grad = crate::laya_scorer::scorer_backward_input(
                        &current.data,
                        &grad.data,
                        weights,
                        spec,
                        current.shape.batch,
                    )
                    .map_err(Error::InvalidConfig)?;
                    accumulate_previous_gradient(
                        layer.index,
                        &Tensor::from_data(current.shape, input_grad)?,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Rnn(spec), LayerParams::Rnn(params)) => {
                    let Some(RecurrentForwardCache::Rnn(cache)) = forward
                        .recurrent_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing RNN forward cache for layer #{}",
                            layer.index
                        )));
                    };
                    let (input_grad, gradients) =
                        backward_rnn_layer(spec, params, cache, current_input, &grad, workspace)?;
                    accumulate_rnn_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing RNN gradient slot for layer #{}",
                                layer.index
                            ))
                        })?,
                        &gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Lstm(spec), LayerParams::Lstm(params)) => {
                    let Some(RecurrentForwardCache::Lstm(cache)) = forward
                        .recurrent_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing LSTM forward cache for layer #{}",
                            layer.index
                        )));
                    };
                    let (input_grad, gradients) =
                        backward_lstm_layer(spec, params, cache, current_input, &grad, workspace)?;
                    accumulate_lstm_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing LSTM gradient slot for layer #{}",
                                layer.index
                            ))
                        })?,
                        &gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crnn(spec), LayerParams::Crnn(params)) => {
                    let Some(RecurrentForwardCache::Crnn(cache)) = forward
                        .recurrent_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing CRNN forward cache for layer #{}",
                            layer.index
                        )));
                    };
                    let (input_grad, gradients) =
                        backward_crnn_layer(spec, params, cache, current_input, &grad, workspace)?;
                    accumulate_crnn_param_gradients(
                        param_gradients.get_mut(layer.index).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing CRNN gradient slot for layer #{}",
                                layer.index
                            ))
                        })?,
                        &gradients,
                    )?;
                    accumulate_previous_gradient(
                        layer.index,
                        &input_grad,
                        &mut input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Unsupported(_), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{}_backward", layer.kind.name()),
                    });
                }
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ({}) and parameter set are inconsistent",
                        layer.index,
                        layer.kind.name()
                    )));
                }
            }
        }

        Ok(BackwardResult {
            input_gradient,
            layer_output_gradients,
            param_gradients,
            mambavision_state_gradients,
        })
    }

}

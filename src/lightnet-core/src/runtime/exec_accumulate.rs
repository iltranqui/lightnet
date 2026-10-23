use super::*;

impl ExecutableNetwork {
    pub fn backward_accumulate(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
        output_gradient: &Tensor,
    ) -> Result<BackwardResult> {
        // Build seed vec and delegate to backward_accumulate_seeded so both paths
        // share the pre-allocated self.workspace (C++ net->workspace parity).
        let mut seeds = vec![None; self.network.layers.len()];
        let last = self
            .network
            .layers
            .len()
            .checked_sub(1)
            .ok_or_else(|| Error::InvalidConfig("network has no layers".to_owned()))?;
        seeds[last] = Some(output_gradient.clone());
        self.backward_accumulate_seeded(input, forward, &seeds)
    }

    pub fn backward_accumulate_seeded(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<BackwardResult> {
        self.backward_accumulate_seeded_with_attention(
            input,
            forward,
            layer_output_gradient_seeds,
            |d_context, _, attention| crate::vit::attention_backward(d_context, attention),
        )
    }

    /// Accumulate ordinary seeded backward plus source-derived hyperbolic-
    /// foveation small-box and strength regularizers from the original YOLO labels.
    pub fn backward_accumulate_seeded_with_hf_auxiliary_gradients(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
        labels: &[Vec<crate::train::YoloLabel>],
    ) -> Result<BackwardResult> {
        let auxiliary_gradients =
            hyperbolic_foveation_auxiliary_param_gradients(&self.network, forward, labels)?;
        self.backward_accumulate_seeded_with_attention_and_hf_auxiliary(
            input,
            forward,
            layer_output_gradient_seeds,
            Some(&auxiliary_gradients),
            |d_context, _, attention| crate::vit::attention_backward(d_context, attention),
        )
    }

    pub(crate) fn backward_accumulate_seeded_with_attention<F>(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
        attention_backward: F,
    ) -> Result<BackwardResult>
    where
        F: FnMut(
            &[f32],
            &[f32],
            &crate::vit::AttentionCache,
        ) -> std::result::Result<Vec<f32>, String>,
    {
        self.backward_accumulate_seeded_with_attention_and_hf_auxiliary(
            input,
            forward,
            layer_output_gradient_seeds,
            None,
            attention_backward,
        )
    }

    pub(super) fn backward_accumulate_seeded_with_attention_and_hf_auxiliary<F>(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
        layer_output_gradient_seeds: &[Option<Tensor>],
        hf_auxiliary_param_gradients: Option<&[Option<Vec<f32>>]>,
        mut attention_backward: F,
    ) -> Result<BackwardResult>
    where
        F: FnMut(
            &[f32],
            &[f32],
            &crate::vit::AttentionCache,
        ) -> std::result::Result<Vec<f32>, String>,
    {
        // Pull out the pre-allocated workspace to avoid the borrow-split problem:
        // mem::take replaces self.workspace with an empty Vec, then we restore it after.
        let mut ws = std::mem::take(&mut self.workspace);
        let result = self.backward_seeded_core(
            &mut ws,
            input,
            forward,
            layer_output_gradient_seeds,
            hf_auxiliary_param_gradients,
            GradMode::from_env(),
            &mut attention_backward,
        );
        self.workspace = ws;
        let result = result?;
        self.update_convolution_batch_norm_running_stats(input, forward)?;
        self.accumulate_train_buffers(&result.param_gradients)?;
        Ok(result)
    }

    pub(super) fn update_convolution_batch_norm_running_stats(
        &mut self,
        input: &Tensor,
        forward: &ForwardResult,
    ) -> Result<()> {
        for layer in &self.network.layers {
            let LayerKind::Convolutional(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize || spec.share_index.is_some() {
                continue;
            }

            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous CPU layer output for batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
            };
            let (mean, variance) = {
                let params =
                    resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                let raw_output =
                    raw_convolution_output(spec, params, current_input, layer.output_shape);
                channel_mean_variance(&raw_output)
            };

            let Some(LayerParams::Convolutional(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing convolutional params for layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing batch-norm params for convolutional layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(
                batch_norm,
                &mean,
                &variance,
                BATCH_NORM_ROLLING_KEEP_CPU,
            )?;
        }

        for layer in &self.network.layers {
            let LayerKind::Dcnv4(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }
            let cache = forward
                .dcnv4_caches
                .get(layer.index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing [dcnv4] forward cache for BN stats at layer #{}",
                        layer.index
                    ))
                })?;
            let raw = Tensor::from_data(layer.output_shape, cache.raw_output.clone())?;
            let (mean, variance) = channel_mean_variance(&raw);
            let Some(LayerParams::Dcnv4(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [dcnv4] params for BN stats at layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing [dcnv4] batch-norm params at layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(
                batch_norm,
                &mean,
                &variance,
                BATCH_NORM_ROLLING_KEEP_CPU,
            )?;
        }

        for layer in &self.network.layers {
            let LayerKind::DeformConv(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }
            let cache = forward
                .deform_conv_caches
                .get(layer.index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing [deform_conv] forward cache for BN stats at layer #{}",
                        layer.index
                    ))
                })?;
            let raw = Tensor::from_data(layer.output_shape, cache.math.raw_output.clone())?;
            let (mean, variance) = channel_mean_variance(&raw);
            let Some(LayerParams::DeformConv(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [deform_conv] params for BN stats at layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.main.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing [deform_conv] batch-norm params at layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(
                batch_norm,
                &mean,
                &variance,
                BATCH_NORM_ROLLING_KEEP_CPU,
            )?;
        }

        for layer in &self.network.layers {
            let LayerKind::GraphConv(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }
            let cache = forward
                .graph_conv_caches
                .get(layer.index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing [graph_conv] forward cache for BN stats at layer #{}",
                        layer.index
                    ))
                })?;
            if !cache.training {
                continue;
            }
            let (mean, variance) = batchnorm_channel_stats(&cache.math.raw_output);
            let Some(LayerParams::GraphConv(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [graph_conv] params for BN stats at layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.projection.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing [graph_conv] batch-norm params at layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(
                batch_norm,
                &mean,
                &variance,
                BATCH_NORM_ROLLING_KEEP_CPU,
            )?;
        }

        for layer in &self.network.layers {
            let LayerKind::Wmhf(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }
            let cache = forward
                .wmhf_caches
                .get(layer.index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing [wmhf] forward cache for BN stats at layer #{}",
                        layer.index
                    ))
                })?;
            if !cache.training {
                continue;
            }
            let Some(LayerParams::Wmhf(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [wmhf] params for BN stats at layer #{}",
                    layer.index
                )));
            };
            for (conv, raw) in [
                (&mut params.pre, &cache.pre.raw_output),
                (&mut params.local_mix, &cache.local_mix.raw_output),
                (&mut params.fuse, &cache.fuse.raw_output),
            ] {
                let Some(batch_norm) = conv.batch_norm.as_mut() else {
                    return Err(Error::InvalidConfig(format!(
                        "missing [wmhf] BN parameters for layer #{}",
                        layer.index
                    )));
                };
                let (mean, variance) = batchnorm_channel_stats(raw);
                update_batch_norm_running_stats(
                    batch_norm,
                    &mean,
                    &variance,
                    BATCH_NORM_ROLLING_KEEP_CPU,
                )?;
            }
        }

        for layer in &self.network.layers {
            let LayerKind::Clifford(_) = &layer.kind else {
                continue;
            };
            let cache = forward
                .clifford_caches
                .get(layer.index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing [clifford] forward cache for BN stats at layer #{}",
                        layer.index
                    ))
                })?;
            if !cache.training {
                continue;
            }
            let Some(LayerParams::Clifford(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [clifford] params for BN stats at layer #{}",
                    layer.index
                )));
            };
            if let Some(local) = &cache.local {
                if params.depthwise.len() != local.depthwise.len() {
                    return Err(Error::InvalidConfig(format!(
                        "[clifford] depthwise cache count mismatch at layer #{}",
                        layer.index
                    )));
                }
                for (conv, subcache) in params.depthwise.iter_mut().zip(&local.depthwise) {
                    let Some(batch_norm) = conv.batch_norm.as_mut() else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [clifford] depthwise BN at layer #{}",
                            layer.index
                        )));
                    };
                    let (mean, variance) = batchnorm_channel_stats(&subcache.raw_output);
                    update_batch_norm_running_stats(
                        batch_norm,
                        &mean,
                        &variance,
                        BATCH_NORM_ROLLING_KEEP_CPU,
                    )?;
                }
            }
        }

        for layer in &self.network.layers {
            let LayerKind::Connected(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }

            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous CPU layer output for connected batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
            };
            let Some(LayerParams::Connected(params)) = self.params.get(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing connected params for layer #{}",
                    layer.index
                )));
            };
            let raw_output = raw_connected_output(params, current_input, layer.output_shape);
            let (mean, variance) = channel_mean_variance(&raw_output);
            let Some(LayerParams::Connected(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing mutable connected params for layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing batch-norm params for connected layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(
                batch_norm,
                &mean,
                &variance,
                BATCH_NORM_ROLLING_KEEP_CPU,
            )?;
        }

        for layer in &self.network.layers {
            if !matches!(layer.kind, LayerKind::BatchNorm(_)) {
                continue;
            }
            if forward.layer_activation_inputs[layer.index].is_none() {
                continue;
            }
            let current_input = if layer.index == 0 {
                input
            } else {
                forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous CPU layer output for standalone batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
            };
            let (mean, variance) = batchnorm_channel_stats(current_input);
            let Some(LayerParams::BatchNorm(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing standalone batch-norm params for layer #{}",
                    layer.index
                )));
            };
            update_batch_norm_running_stats(params, &mean, &variance, BATCH_NORM_ROLLING_KEEP_CPU)?;
        }
        for layer in &self.network.layers {
            let LayerKind::RecursiveBlock(spec) = &layer.kind else {
                continue;
            };
            let cache = forward
                .recursive_block_caches
                .get(layer.index)
                .and_then(Option::as_ref)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing [recursive_block] forward cache for BN stats at layer #{}",
                        layer.index
                    ))
                })?;
            let Some(LayerParams::RecursiveBlock(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing [recursive_block] parameters for BN stats at layer #{}",
                    layer.index
                )));
            };
            update_recursive_body_batch_norm_running_stats(layer, spec, params, cache)?;
        }
        update_recurrent_batch_norm_running_stats(&self.network, &mut self.params, input, forward)?;
        Ok(())
    }

    pub(super) fn accumulate_train_buffers(&mut self, param_gradients: &[LayerParams]) -> Result<()> {
        if param_gradients.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "parameter gradient count mismatch: expected {}, got {}",
                self.network.layers.len(),
                param_gradients.len()
            )));
        }

        for (layer, gradient) in self.network.layers.iter().zip(param_gradients) {
            match (&layer.kind, gradient) {
                (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(gradient)) => {
                    let Some(LayerTrainBuffers::RecursiveBlock(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing recursive-block train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_recursive_block_train_buffers(spec, train, gradient)?;
                }
                (LayerKind::HyperbolicFoveation(_), LayerParams::HyperbolicFoveation(gradient)) => {
                    let Some(LayerTrainBuffers::HyperbolicFoveation(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing hyperbolic-foveation train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.hf_param_updates, &gradient.hf_params)?;
                    accumulate_vec(&mut train.pred_w1_updates, &gradient.pred_w1)?;
                    accumulate_vec(&mut train.pred_b1_updates, &gradient.pred_b1)?;
                    accumulate_vec(&mut train.pred_w2_updates, &gradient.pred_w2)?;
                    accumulate_vec(&mut train.pred_b2_updates, &gradient.pred_b2)?;
                }
                (LayerKind::FoveationFilm(_), LayerParams::FoveationFilm(gradient)) => {
                    let Some(LayerTrainBuffers::FoveationFilm(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [foveation_film] train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.w_embed_updates, &gradient.w_embed)?;
                    accumulate_vec(&mut train.b_embed_updates, &gradient.b_embed)?;
                    accumulate_vec(&mut train.w_gamma_updates, &gradient.w_gamma)?;
                    accumulate_vec(&mut train.b_gamma_updates, &gradient.b_gamma)?;
                    accumulate_vec(&mut train.w_beta_updates, &gradient.w_beta)?;
                    accumulate_vec(&mut train.b_beta_updates, &gradient.b_beta)?;
                }
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(gradient)) => {
                    let Some(LayerTrainBuffers::BatchNorm(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing standalone batch-norm train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &gradient.biases)?;
                    accumulate_vec(&mut train.scale_updates, &gradient.scales)?;
                }
                (LayerKind::Convolutional(spec), LayerParams::Convolutional(conv_grad)) => {
                    let owner = spec.share_index.unwrap_or(layer.index);
                    let Some(LayerTrainBuffers::Convolutional(train)) =
                        self.train_buffers.get_mut(owner)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing convolutional train buffers for layer #{} owner #{}",
                            layer.index, owner
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &conv_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &conv_grad.weights)?;
                    match (&mut train.batch_norm, &conv_grad.batch_norm) {
                        (Some(train_bn), Some(grad_bn)) => {
                            accumulate_vec(&mut train_bn.scale_updates, &grad_bn.scales)?;
                            accumulate_vec(&mut train_bn.mean_delta, &grad_bn.rolling_mean)?;
                            accumulate_vec(
                                &mut train_bn.variance_delta,
                                &grad_bn.rolling_variance,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(format!(
                                "batch norm train buffer mismatch while accumulating layer #{}",
                                layer.index
                            )));
                        }
                    }
                }
                (LayerKind::Deconvolutional(_), LayerParams::Convolutional(conv_grad)) => {
                    let Some(LayerTrainBuffers::Convolutional(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing deconvolutional train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &conv_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &conv_grad.weights)?;
                }
                (LayerKind::Connected(_), LayerParams::Connected(connected_grad)) => {
                    let Some(LayerTrainBuffers::Connected(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing connected train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &connected_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &connected_grad.weights)?;
                    match (&mut train.batch_norm, &connected_grad.batch_norm) {
                        (Some(train_bn), Some(grad_bn)) => {
                            accumulate_vec(&mut train_bn.scale_updates, &grad_bn.scales)?;
                            accumulate_vec(&mut train_bn.mean_delta, &grad_bn.rolling_mean)?;
                            accumulate_vec(
                                &mut train_bn.variance_delta,
                                &grad_bn.rolling_variance,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(format!(
                                "batch norm train buffer mismatch while accumulating connected layer #{}",
                                layer.index
                            )));
                        }
                    }
                }
                (LayerKind::Local(_), LayerParams::Local(local_grad)) => {
                    let Some(LayerTrainBuffers::Local(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing local train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &local_grad.biases)?;
                    accumulate_vec(&mut train.weight_updates, &local_grad.weights)?;
                }
                (LayerKind::Shortcut(_), LayerParams::Shortcut(gradient)) => {
                    let Some(LayerTrainBuffers::Shortcut(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing shortcut train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.weight_updates, &gradient.weights)?;
                }
                (LayerKind::Rnn(_), LayerParams::Rnn(gradient)) => {
                    let Some(LayerTrainBuffers::Rnn(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing RNN train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_connected_train_buffer(&mut train.input, &gradient.input)?;
                    accumulate_connected_train_buffer(
                        &mut train.self_params,
                        &gradient.self_params,
                    )?;
                    accumulate_connected_train_buffer(&mut train.output, &gradient.output)?;
                }
                (LayerKind::Lstm(_), LayerParams::Lstm(gradient)) => {
                    let Some(LayerTrainBuffers::Lstm(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing LSTM train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_connected_train_buffer(&mut train.wf, &gradient.wf)?;
                    accumulate_connected_train_buffer(&mut train.wi, &gradient.wi)?;
                    accumulate_connected_train_buffer(&mut train.wg, &gradient.wg)?;
                    accumulate_connected_train_buffer(&mut train.wo, &gradient.wo)?;
                    accumulate_connected_train_buffer(&mut train.uf, &gradient.uf)?;
                    accumulate_connected_train_buffer(&mut train.ui, &gradient.ui)?;
                    accumulate_connected_train_buffer(&mut train.ug, &gradient.ug)?;
                    accumulate_connected_train_buffer(&mut train.uo, &gradient.uo)?;
                }
                (LayerKind::Crnn(_), LayerParams::Crnn(gradient)) => {
                    let Some(LayerTrainBuffers::Crnn(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing CRNN train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_convolutional_train_buffer(&mut train.input, &gradient.input)?;
                    accumulate_convolutional_train_buffer(
                        &mut train.self_params,
                        &gradient.self_params,
                    )?;
                    accumulate_convolutional_train_buffer(&mut train.output, &gradient.output)?;
                }
                (LayerKind::Moe(_), LayerParams::Moe(gradient)) => {
                    let Some(LayerTrainBuffers::Moe(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing MoE train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.router_weight_updates, &gradient.router_weights)?;
                    accumulate_vec(&mut train.router_bias_updates, &gradient.router_biases)?;
                    if train.experts.len() != gradient.experts.len() {
                        return Err(Error::InvalidConfig(format!(
                            "MoE expert train-buffer count mismatch at layer #{}",
                            layer.index
                        )));
                    }
                    for (train_expert, grad_expert) in
                        train.experts.iter_mut().zip(&gradient.experts)
                    {
                        accumulate_convolutional_train_buffer(train_expert, grad_expert)?;
                    }
                }
                (LayerKind::EmlConvolutional(_), LayerParams::EmlConvolutional(gradient)) => {
                    let Some(LayerTrainBuffers::EmlConvolutional(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing EML train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_convolutional_train_buffer(&mut train.input, &gradient.input)?;
                    accumulate_convolutional_train_buffer(
                        &mut train.self_params,
                        &gradient.self_params,
                    )?;
                }
                (LayerKind::DeformConv(_), LayerParams::DeformConv(gradient)) => {
                    let Some(LayerTrainBuffers::DeformConv(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing deform_conv train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_convolutional_train_buffer(&mut train.main, &gradient.main)?;
                    accumulate_vec(&mut train.offset_weight_updates, &gradient.offset_weights)?;
                    accumulate_vec(&mut train.offset_bias_updates, &gradient.offset_biases)?;
                    match (&mut train.mask_weight_updates, &gradient.mask_weights) {
                        (Some(train), Some(gradient)) => accumulate_vec(train, gradient)?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "deform_conv mask weight gradient mismatch".to_owned(),
                            ));
                        }
                    }
                    match (&mut train.mask_bias_updates, &gradient.mask_biases) {
                        (Some(train), Some(gradient)) => accumulate_vec(train, gradient)?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "deform_conv mask bias gradient mismatch".to_owned(),
                            ));
                        }
                    }
                }
                (LayerKind::DsConv(_), LayerParams::DsConv(gradient)) => {
                    let Some(LayerTrainBuffers::DsConv(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing dsconv train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_convolutional_train_buffer(&mut train.collapse, &gradient.collapse)?;
                    accumulate_vec(&mut train.offset_weight_updates, &gradient.offset_weights)?;
                    accumulate_vec(&mut train.offset_bias_updates, &gradient.offset_biases)?;
                }
                (LayerKind::Wmhf(_), LayerParams::Wmhf(gradient)) => {
                    let Some(LayerTrainBuffers::Wmhf(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [wmhf] train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    for (train, gradient) in [
                        (&mut train.pre, &gradient.pre),
                        (&mut train.local_dw3, &gradient.local_dw3),
                        (&mut train.local_dw5, &gradient.local_dw5),
                        (&mut train.local_dw7, &gradient.local_dw7),
                        (&mut train.local_mix, &gradient.local_mix),
                        (&mut train.hf_gate, &gradient.hf_gate),
                        (&mut train.fuse, &gradient.fuse),
                    ] {
                        accumulate_convolutional_train_buffer(train, gradient)?;
                    }
                    accumulate_vec(&mut train.scan_weight_updates, &gradient.scan_weights)?;
                }
                (LayerKind::Clifford(_), LayerParams::Clifford(gradient)) => {
                    let Some(LayerTrainBuffers::Clifford(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [clifford] train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(
                        &mut train.layer_norm_gamma_updates,
                        &gradient.layer_norm_gamma,
                    )?;
                    accumulate_vec(
                        &mut train.layer_norm_beta_updates,
                        &gradient.layer_norm_beta,
                    )?;
                    accumulate_vec(
                        &mut train.determinant_weight_updates,
                        &gradient.determinant_weights,
                    )?;
                    accumulate_vec(
                        &mut train.determinant_bias_updates,
                        &gradient.determinant_biases,
                    )?;
                    accumulate_vec(&mut train.layer_scale_updates, &gradient.layer_scale)?;
                    if train.depthwise.len() != gradient.depthwise.len() {
                        return Err(Error::InvalidConfig(format!(
                            "[clifford] depthwise gradient count mismatch at layer #{}",
                            layer.index
                        )));
                    }
                    for (updates, gradient) in train.depthwise.iter_mut().zip(&gradient.depthwise) {
                        accumulate_convolutional_train_buffer(updates, gradient)?;
                    }
                    accumulate_clifford_branch(&mut train.local, &gradient.local)?;
                    match (&mut train.global, &gradient.global) {
                        (Some(train), Some(gradient)) => {
                            accumulate_clifford_branch(train, gradient)?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "[clifford] global gradient mismatch".to_owned(),
                            ));
                        }
                    }
                    match (&mut train.fourier, &gradient.fourier) {
                        (Some(train), Some(gradient)) => {
                            accumulate_vec(&mut train.real_updates, &gradient.real)?;
                            accumulate_vec(&mut train.imaginary_updates, &gradient.imaginary)?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "[clifford] Fourier gradient mismatch".to_owned(),
                            ));
                        }
                    }
                }
                (LayerKind::GraphConv(_), LayerParams::GraphConv(gradient)) => {
                    let Some(LayerTrainBuffers::GraphConv(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing graph_conv train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_convolutional_train_buffer(
                        &mut train.projection,
                        &gradient.projection,
                    )?;
                    match (&mut train.self_weight_updates, &gradient.self_weights) {
                        (Some(updates), Some(gradient)) => accumulate_vec(updates, gradient)?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "graph_conv self-weight train-buffer mismatch".to_owned(),
                            ));
                        }
                    }
                    match (&mut train.edge_kernel_updates, &gradient.edge_kernel) {
                        (Some(updates), Some(gradient)) => accumulate_vec(updates, gradient)?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "graph_conv edge-kernel train-buffer mismatch".to_owned(),
                            ));
                        }
                    }
                    match (&mut train.edge_bias_updates, &gradient.edge_biases) {
                        (Some(updates), Some(gradient)) => accumulate_vec(updates, gradient)?,
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "graph_conv edge-bias train-buffer mismatch".to_owned(),
                            ));
                        }
                    }
                }
                (LayerKind::Dcnv4(_), LayerParams::Dcnv4(gradient)) => {
                    let Some(LayerTrainBuffers::Dcnv4(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing DCNv4 train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.bias_updates, &gradient.biases)?;
                    accumulate_vec(&mut train.offset_weight_updates, &gradient.offset_weights)?;
                    accumulate_vec(&mut train.offset_bias_updates, &gradient.offset_biases)?;
                    match (&mut train.batch_norm, &gradient.batch_norm) {
                        (Some(train_bn), Some(gradient_bn)) => {
                            accumulate_vec(&mut train_bn.scale_updates, &gradient_bn.scales)?;
                            accumulate_vec(&mut train_bn.mean_delta, &gradient_bn.rolling_mean)?;
                            accumulate_vec(
                                &mut train_bn.variance_delta,
                                &gradient_bn.rolling_variance,
                            )?;
                        }
                        (None, None) => {}
                        _ => {
                            return Err(Error::InvalidConfig(
                                "DCNv4 train-buffer batchnorm mismatch".to_owned(),
                            ));
                        }
                    }
                }
                (LayerKind::Vit(_), LayerParams::Vit(gradient)) => {
                    let Some(LayerTrainBuffers::Vit(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing ViT train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vit_updates(&mut train.updates, gradient)?;
                }
                (LayerKind::Swin(_), LayerParams::Swin(gradient)) => {
                    let Some(LayerTrainBuffers::Swin(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing Swin train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_swin_updates(&mut train.updates, gradient)?;
                }
                (LayerKind::TuckerAttention(_), LayerParams::TuckerAttention(gradient)) => {
                    let Some(LayerTrainBuffers::TuckerAttention(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing Tucker attention train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_tucker_attention_updates(train, gradient)?;
                }
                (LayerKind::MambaVision(_), LayerParams::MambaVision(gradient)) => {
                    let Some(LayerTrainBuffers::MambaVision(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing MambaVision train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.updates, &gradient.values)?;
                }
                (LayerKind::KimiKda(_), LayerParams::KimiKda(gradient)) => {
                    let Some(LayerTrainBuffers::KimiKda(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing Kimi KDA train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_negative_vec(&mut train.bias_updates, &gradient.biases)?;
                    accumulate_negative_vec(&mut train.weight_updates, &gradient.weights)?;
                }
                (_, LayerParams::None) => {}
                // `[laya_scorer]` is frozen by default, so it has no train
                // buffers and contributes no parameter gradient.
                (_, LayerParams::LayaScorer(_)) => {}
                (_, LayerParams::LayaActHead(_)) => {}
                (LayerKind::DepthAnything(_), LayerParams::DepthAnything(_)) => {}
                (LayerKind::DptHead(_), LayerParams::DptHead(gradient)) => {
                    let Some(LayerTrainBuffers::DptHead(train)) =
                        self.train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing [dpt_head] train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    if train.weight_updates.len() != gradient.values.len() {
                        return Err(Error::InvalidConfig(format!(
                            "[dpt_head] gradient/update length mismatch: {} != {}",
                            gradient.values.len(),
                            train.weight_updates.len()
                        )));
                    }
                    train.has_gradient |= gradient.values.iter().any(|&value| value != 0.0);
                    accumulate_negative_vec(&mut train.weight_updates, &gradient.values)?;
                }
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} produced gradients incompatible with train buffers",
                        layer.index
                    )));
                }
            }
        }

        Ok(())
    }
}

pub(super) fn recursive_body_executable(
    network: &Network,
    layer: &crate::model::Layer,
    spec: &crate::model::RecursiveBlockSpec,
    params: &RecursiveBlockParams,
    train_iteration: usize,
) -> Result<ExecutableNetwork> {
    if spec.body.len() != params.body.len() {
        return Err(Error::InvalidConfig(format!(
            "[recursive_block] layer #{} body parameter count mismatch: expected {}, got {}",
            layer.index,
            spec.body.len(),
            params.body.len()
        )));
    }
    let mut body_network = Network {
        source: network.source.clone(),
        net: network.net.clone(),
        layers: spec.body.clone(),
    };
    body_network.net.width = layer.input_shape.width;
    body_network.net.height = layer.input_shape.height;
    body_network.net.channels = layer.input_shape.channels;
    body_network.net.batch = layer.input_shape.batch;
    let mut executable = ExecutableNetwork::new(body_network);
    executable.params.clone_from(&params.body);
    executable.train_iteration = train_iteration;
    Ok(executable)
}

pub(super) fn recursive_body_rng_seed(train_iteration: usize, layer_index: usize, step: usize) -> u64 {
    let mut value = 0x9e37_79b9_7f4a_7c15_u64
        ^ (train_iteration as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ (layer_index as u64).wrapping_mul(0x94d0_49bb_1331_11eb)
        ^ (step as u64).wrapping_mul(0x369d_ea0f_31a5_3f85);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    if value == 0 { 0x9e37_79b9_u64 } else { value }
}

pub(super) fn recursive_controller_values(
    hidden: &Tensor,
    spec: &crate::model::RecursiveBlockSpec,
    params: &RecursiveBlockParams,
    step: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if spec.ouroboros == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let channels = hidden.shape.channels;
    let inputs = channels
        .checked_add(1)
        .ok_or_else(|| Error::InvalidConfig("recursive controller input overflows".to_owned()))?;
    let outputs = channels
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidConfig("recursive controller output overflows".to_owned()))?;
    let weight_count = inputs
        .checked_mul(outputs)
        .ok_or_else(|| Error::InvalidConfig("recursive controller weights overflow".to_owned()))?;
    if params.controller_weights.len() != weight_count || params.controller_biases.len() != outputs
    {
        return Err(Error::InvalidConfig(
            "[recursive_block] controller parameter layout does not match channel count".to_owned(),
        ));
    }
    let batch = hidden.shape.batch;
    let area = hidden.shape.width * hidden.shape.height;
    let step_value = if spec.loops > 1 {
        step as f32 / (spec.loops - 1) as f32
    } else {
        0.0
    };
    let mut controller_inputs = vec![0.0; batch * inputs];
    let mut controller_outputs = vec![0.0; batch * outputs];
    for b in 0..batch {
        let in_offset = b * inputs;
        let batch_offset = b * channels * area;
        for c in 0..channels {
            let channel_offset = batch_offset + c * area;
            let sum = hidden.data[channel_offset..channel_offset + area]
                .iter()
                .sum::<f32>();
            controller_inputs[in_offset + c] = sum / area.max(1) as f32;
        }
        controller_inputs[in_offset + channels] = step_value;
        let out_offset = b * outputs;
        for o in 0..outputs {
            let weight_offset = o * inputs;
            let mut value = params.controller_biases[o];
            for i in 0..inputs {
                value +=
                    params.controller_weights[weight_offset + i] * controller_inputs[in_offset + i];
            }
            controller_outputs[out_offset + o] = value;
        }
    }
    Ok((controller_inputs, controller_outputs))
}

pub(super) fn recursive_clip(value: f32, limit: f32) -> f32 {
    if limit > 0.0 {
        value.clamp(-limit, limit)
    } else {
        value
    }
}

pub(super) fn recursive_sigmoid(value: f32) -> f32 {
    let clipped = value.clamp(-30.0, 30.0);
    1.0 / (1.0 + (-clipped).exp())
}

pub(super) fn recursive_alive_probability(value: f32, temperature: f32) -> (f32, f32) {
    if !value.is_finite() {
        return (0.0, 0.0);
    }
    if (0.0..=1.0).contains(&value) {
        return (value, 1.0);
    }
    let scaled = value / temperature.max(1e-6);
    let clipped = scaled.clamp(-30.0, 30.0);
    let alive = 1.0 / (1.0 + (-clipped).exp());
    let derivative = if scaled > -30.0 && scaled < 30.0 {
        alive * (1.0 - alive) / temperature.max(1e-6)
    } else {
        0.0
    };
    (alive, derivative)
}

pub(super) fn recursive_lattice_project(
    previous: &Tensor,
    raw_state: &Tensor,
    spec: &crate::model::RecursiveBlockSpec,
) -> Result<Tensor> {
    if previous.shape != raw_state.shape {
        return Err(Error::InvalidConfig(
            "[recursive_block] lattice input shape mismatch".to_owned(),
        ));
    }
    let blend = spec.lattice_mix;
    if blend == 0.0 {
        return Ok(raw_state.clone());
    }
    let channels = raw_state.shape.channels;
    let area = raw_state.shape.width * raw_state.shape.height;
    let batch = raw_state.shape.batch;
    let group = if spec.lattice_group > 1 {
        spec.lattice_group
    } else {
        1
    };
    let groups_per_cell = channels / group;
    let mut output = raw_state.clone();
    for b in 0..batch {
        for spatial in 0..area {
            for group_index in 0..groups_per_cell {
                let start_channel = group_index * group;
                let mut projected = vec![0.0; group];
                let mut best_alive = -1.0f32;
                let mut best_index = 0usize;
                let mut any_alive = false;
                for (offset, projected_value) in projected.iter_mut().enumerate() {
                    let channel = start_channel + offset;
                    let index = b * channels * area + channel * area + spatial;
                    let (previous_alive, _) =
                        recursive_alive_probability(previous.data[index], spec.lattice_temperature);
                    let (step_alive, _) = recursive_alive_probability(
                        raw_state.data[index],
                        spec.lattice_temperature,
                    );
                    let alive = previous_alive.min(step_alive);
                    if alive > best_alive {
                        best_alive = alive;
                        best_index = offset;
                    }
                    if alive < spec.lattice_threshold {
                        *projected_value = 0.0;
                    } else {
                        any_alive = true;
                        *projected_value = alive;
                    }
                }
                if group > 1 && !any_alive && best_alive > 0.0 {
                    projected[best_index] = best_alive;
                }
                for (offset, projected_value) in projected.iter().enumerate() {
                    let channel = start_channel + offset;
                    let index = b * channels * area + channel * area + spatial;
                    output.data[index] =
                        (1.0 - blend) * raw_state.data[index] + blend * projected_value;
                }
            }
        }
    }
    Ok(output)
}

pub(super) fn recursive_lattice_backward(
    previous: &Tensor,
    raw_state: &Tensor,
    output_gradient: &Tensor,
    spec: &crate::model::RecursiveBlockSpec,
) -> Result<(Tensor, Tensor)> {
    if previous.shape != raw_state.shape || output_gradient.shape != raw_state.shape {
        return Err(Error::InvalidConfig(
            "[recursive_block] lattice backward shape mismatch".to_owned(),
        ));
    }
    let mut previous_gradient = Tensor::zeros(previous.shape);
    let mut raw_gradient = Tensor::zeros(raw_state.shape);
    let blend = spec.lattice_mix;
    if blend == 0.0 {
        raw_gradient.data.copy_from_slice(&output_gradient.data);
        return Ok((previous_gradient, raw_gradient));
    }
    let channels = raw_state.shape.channels;
    let area = raw_state.shape.width * raw_state.shape.height;
    let batch = raw_state.shape.batch;
    let group = if spec.lattice_group > 1 {
        spec.lattice_group
    } else {
        1
    };
    let groups_per_cell = channels / group;
    for b in 0..batch {
        for spatial in 0..area {
            for group_index in 0..groups_per_cell {
                let start_channel = group_index * group;
                let mut previous_alive = vec![0.0; group];
                let mut previous_slope = vec![0.0; group];
                let mut step_alive = vec![0.0; group];
                let mut step_slope = vec![0.0; group];
                let mut met = vec![0.0; group];
                let mut survives = vec![false; group];
                let mut best_alive = -1.0f32;
                let mut best_index = 0usize;
                let mut any_alive = false;
                for offset in 0..group {
                    let channel = start_channel + offset;
                    let index = b * channels * area + channel * area + spatial;
                    (previous_alive[offset], previous_slope[offset]) =
                        recursive_alive_probability(previous.data[index], spec.lattice_temperature);
                    (step_alive[offset], step_slope[offset]) = recursive_alive_probability(
                        raw_state.data[index],
                        spec.lattice_temperature,
                    );
                    met[offset] = previous_alive[offset].min(step_alive[offset]);
                    if met[offset] > best_alive {
                        best_alive = met[offset];
                        best_index = offset;
                    }
                    survives[offset] = met[offset] >= spec.lattice_threshold;
                    any_alive |= survives[offset];
                }
                let rescue = group > 1 && !any_alive && best_alive > 0.0;
                for offset in 0..group {
                    let channel = start_channel + offset;
                    let index = b * channels * area + channel * area + spatial;
                    let dy = output_gradient.data[index];
                    raw_gradient.data[index] += dy * (1.0 - blend);
                    let projected_gradient = dy * blend;
                    let active = if rescue {
                        offset == best_index
                    } else {
                        survives[offset]
                    };
                    if active {
                        if previous_alive[offset] <= step_alive[offset] {
                            previous_gradient.data[index] +=
                                projected_gradient * previous_slope[offset];
                        } else {
                            raw_gradient.data[index] += projected_gradient * step_slope[offset];
                        }
                    }
                }
            }
        }
    }
    Ok((previous_gradient, raw_gradient))
}

pub(super) fn accumulate_recursive_body_param_gradients(
    spec: &crate::model::RecursiveBlockSpec,
    destination: &mut [LayerParams],
    source: &[LayerParams],
) -> Result<()> {
    if spec.body.len() != destination.len() || spec.body.len() != source.len() {
        return Err(Error::InvalidConfig(
            "recursive-block body parameter-gradient count mismatch".to_owned(),
        ));
    }
    for ((layer, destination), source) in spec.body.iter().zip(destination).zip(source) {
        match (&layer.kind, destination, source) {
            (LayerKind::Convolutional(_), destination, LayerParams::Convolutional(source)) => {
                accumulate_convolutional_param_gradients(destination, source)?;
            }
            (LayerKind::Connected(_), destination, LayerParams::Connected(source)) => {
                accumulate_connected_param_gradients(destination, source)?;
            }
            (
                LayerKind::MaxPool(_)
                | LayerKind::LocalAvgPool(_)
                | LayerKind::AvgPool
                | LayerKind::Upsample(_)
                | LayerKind::ChannelShuffle(_)
                | LayerKind::Dropout { .. },
                LayerParams::None,
                LayerParams::None,
            ) => {}
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "unsupported recursive-block parameter-gradient layout for body layer #{} ({})",
                    layer.index,
                    layer.kind.name()
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn backward_recursive_block(
    network: &Network,
    train_iteration: usize,
    layer: &crate::model::Layer,
    spec: &crate::model::RecursiveBlockSpec,
    params: &RecursiveBlockParams,
    input: &Tensor,
    output_gradient: &Tensor,
    cache: &RecursiveBlockForwardCache,
) -> Result<(Tensor, LayerParams)> {
    if output_gradient.shape != layer.output_shape
        || cache.body_passes.len() != spec.loops
        || cache.hidden_inputs.len() != spec.loops
        || cache.controller_inputs.len() != spec.loops
        || cache.controller_outputs.len() != spec.loops
        || cache.lattice_inputs.len() != spec.loops
    {
        return Err(Error::InvalidConfig(format!(
            "[recursive_block] layer #{} forward/backward tape layout mismatch",
            layer.index
        )));
    }
    let body = recursive_body_executable(network, layer, spec, params, train_iteration)?;
    let mut input_gradient = Tensor::zeros(input.shape);
    let mut hidden_gradient = output_gradient.clone();
    let mut parameter_gradient = default_layer_params(layer);
    let LayerParams::RecursiveBlock(parameter_gradient) = &mut parameter_gradient else {
        unreachable!("recursive-block layer requires recursive-block gradients");
    };
    let channels = input.shape.channels;
    let area = input.shape.width * input.shape.height;
    let batch = input.shape.batch;

    for step in (0..spec.loops).rev() {
        let hidden = &cache.hidden_inputs[step];
        let mut hidden_previous_gradient = Tensor::zeros(hidden.shape);
        let (lattice_previous_gradient, mut recurrence_gradient) = if step + 1 < spec.loops {
            if let Some(raw_state) = &cache.lattice_inputs[step] {
                recursive_lattice_backward(hidden, raw_state, &hidden_gradient, spec)?
            } else {
                (Tensor::zeros(hidden.shape), hidden_gradient.clone())
            }
        } else {
            (Tensor::zeros(hidden.shape), hidden_gradient.clone())
        };
        hidden_previous_gradient
            .data
            .copy_from_slice(&lattice_previous_gradient.data);

        if spec.injection {
            for (dst, dy) in input_gradient
                .data
                .iter_mut()
                .zip(&recurrence_gradient.data)
            {
                *dst += spec.injection_scale * dy;
            }
        }

        let body_output = cache.body_passes[step].final_output().ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[recursive_block] layer #{} body tape has no output",
                layer.index
            ))
        })?;
        let mut body_gradient = Tensor::zeros(body_output.shape);
        if spec.ouroboros == 0 {
            for index in 0..recurrence_gradient.data.len() {
                body_gradient.data[index] = spec.body_scale * recurrence_gradient.data[index];
                hidden_previous_gradient.data[index] +=
                    spec.residual_scale * recurrence_gradient.data[index];
            }
        } else {
            let outputs = channels * 2;
            let controller_inputs = channels + 1;
            let input_values = &cache.controller_inputs[step];
            let output_values = &cache.controller_outputs[step];
            if input_values.len() != batch * controller_inputs
                || output_values.len() != batch * outputs
            {
                return Err(Error::InvalidConfig(
                    "[recursive_block] controller forward tape size mismatch".to_owned(),
                ));
            }
            let mut controller_gradient = vec![0.0; batch * outputs];
            for b in 0..batch {
                let controller_offset = b * outputs;
                for channel in 0..channels {
                    let gamma_raw = output_values[controller_offset + channel];
                    let gate_raw = output_values[controller_offset + channels + channel];
                    let gamma = recursive_clip(gamma_raw, spec.gamma_clip);
                    let clipped_gate = recursive_clip(gate_raw, spec.gate_clip);
                    let gate_input = clipped_gate.clamp(-30.0, 30.0);
                    let gate = recursive_sigmoid(clipped_gate);
                    let gamma_derivative =
                        spec.gamma_clip <= 0.0 || gamma_raw.abs() < spec.gamma_clip;
                    let gate_clip_derivative =
                        spec.gate_clip <= 0.0 || gate_raw.abs() < spec.gate_clip;
                    let sigmoid_derivative = gate_input.abs() < 30.0;
                    let mut gamma_gradient = 0.0;
                    let mut gate_gradient = 0.0;
                    for spatial in 0..area {
                        let index = b * channels * area + channel * area + spatial;
                        let dy = recurrence_gradient.data[index];
                        let candidate = spec.body_scale * body_output.data[index];
                        body_gradient.data[index] += dy * spec.body_scale * gate * (1.0 + gamma);
                        hidden_previous_gradient.data[index] += dy * (1.0 - gate);
                        gamma_gradient += dy * gate * candidate;
                        gate_gradient += dy * (candidate * (1.0 + gamma) - hidden.data[index]);
                    }
                    if gamma_derivative {
                        controller_gradient[controller_offset + channel] = gamma_gradient;
                    }
                    if gate_clip_derivative && sigmoid_derivative {
                        controller_gradient[controller_offset + channels + channel] =
                            gate_gradient * gate * (1.0 - gate);
                    }
                }
            }
            let controller_weight_count = outputs * controller_inputs;
            if parameter_gradient.controller_weights.len() != controller_weight_count
                || parameter_gradient.controller_biases.len() != outputs
            {
                return Err(Error::InvalidConfig(
                    "[recursive_block] controller gradient layout mismatch".to_owned(),
                ));
            }
            for b in 0..batch {
                let input_offset = b * controller_inputs;
                let output_offset = b * outputs;
                let mut controller_input_gradient = vec![0.0; controller_inputs];
                for o in 0..outputs {
                    let d = controller_gradient[output_offset + o];
                    parameter_gradient.controller_biases[o] += d;
                    let weight_offset = o * controller_inputs;
                    for i in 0..controller_inputs {
                        parameter_gradient.controller_weights[weight_offset + i] +=
                            d * input_values[input_offset + i];
                        controller_input_gradient[i] +=
                            d * params.controller_weights[weight_offset + i];
                    }
                }
                let hidden_offset = b * channels * area;
                for channel in 0..channels {
                    let spatial_gradient = controller_input_gradient[channel] / area.max(1) as f32;
                    let channel_offset = hidden_offset + channel * area;
                    for spatial in 0..area {
                        hidden_previous_gradient.data[channel_offset + spatial] += spatial_gradient;
                    }
                }
            }
        }

        let body_backward = body.backward(hidden, &cache.body_passes[step], &body_gradient)?;
        for (dst, src) in hidden_previous_gradient
            .data
            .iter_mut()
            .zip(body_backward.input_gradient.data)
        {
            *dst += src;
        }
        accumulate_recursive_body_param_gradients(
            spec,
            &mut parameter_gradient.body,
            &body_backward.param_gradients,
        )?;
        hidden_gradient = hidden_previous_gradient;
        recurrence_gradient.data.clear();
    }
    for (dst, recurrent) in input_gradient.data.iter_mut().zip(hidden_gradient.data) {
        *dst += recurrent;
    }
    Ok((
        input_gradient,
        LayerParams::RecursiveBlock(parameter_gradient.clone()),
    ))
}

pub(super) fn forward_recursive_block(
    network: &Network,
    train_iteration: usize,
    layer: &crate::model::Layer,
    spec: &crate::model::RecursiveBlockSpec,
    params: &RecursiveBlockParams,
    input: &Tensor,
    training: bool,
) -> Result<(Tensor, RecursiveBlockForwardCache)> {
    if input.shape != layer.input_shape || layer.output_shape != layer.input_shape {
        return Err(Error::InvalidConfig(format!(
            "[recursive_block] layer #{} requires a size-preserving input, got {} -> {}",
            layer.index, layer.input_shape, layer.output_shape
        )));
    }
    let body = recursive_body_executable(network, layer, spec, params, train_iteration)?;
    let mut hidden = input.clone();
    let mut body_passes = Vec::with_capacity(spec.loops);
    let mut hidden_inputs = Vec::with_capacity(spec.loops);
    let mut controller_inputs = Vec::with_capacity(spec.loops);
    let mut controller_outputs = Vec::with_capacity(spec.loops);
    let mut lattice_inputs = Vec::with_capacity(spec.loops);

    for step in 0..spec.loops {
        hidden_inputs.push(hidden.clone());
        let (controller_input, controller_output) =
            recursive_controller_values(&hidden, spec, params, step)?;
        let body_forward = if training {
            body.forward_training_with_rng_seed(
                &hidden,
                recursive_body_rng_seed(train_iteration, layer.index, step),
            )?
        } else {
            body.forward(&hidden)?
        };
        let body_output = body_forward.final_output().ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[recursive_block] layer #{} body produced no output",
                layer.index
            ))
        })?;
        if body_output.shape != layer.output_shape {
            return Err(Error::InvalidConfig(format!(
                "[recursive_block] layer #{} body output shape mismatch: expected {}, got {}",
                layer.index, layer.output_shape, body_output.shape
            )));
        }
        let channels = input.shape.channels;
        let area = input.shape.width * input.shape.height;
        let controller_channels = channels * 2;
        let mut next = Tensor::zeros(layer.output_shape);
        for b in 0..input.shape.batch {
            for channel in 0..channels {
                let controller_offset = b * controller_channels;
                let gamma = if spec.ouroboros == 1 {
                    recursive_clip(
                        controller_output[controller_offset + channel],
                        spec.gamma_clip,
                    )
                } else {
                    0.0
                };
                let gate = if spec.ouroboros == 1 {
                    recursive_sigmoid(recursive_clip(
                        controller_output[controller_offset + channels + channel],
                        spec.gate_clip,
                    ))
                } else {
                    0.0
                };
                for spatial in 0..area {
                    let index = b * channels * area + channel * area + spatial;
                    let candidate = spec.body_scale * body_output.data[index];
                    let value = if spec.ouroboros == 1 {
                        gate * candidate * (1.0 + gamma) + (1.0 - gate) * hidden.data[index]
                    } else {
                        candidate + spec.residual_scale * hidden.data[index]
                    };
                    next.data[index] = value
                        + if spec.injection {
                            spec.injection_scale * input.data[index]
                        } else {
                            0.0
                        };
                }
            }
        }
        body_passes.push(body_forward);
        controller_inputs.push(controller_input);
        controller_outputs.push(controller_output);
        if step + 1 < spec.loops && spec.lattice_mix > 0.0 {
            let projected = recursive_lattice_project(&hidden, &next, spec)?;
            lattice_inputs.push(Some(next));
            hidden = projected;
        } else {
            lattice_inputs.push(None);
            hidden = next;
        }
    }
    Ok((
        hidden,
        RecursiveBlockForwardCache {
            body_passes,
            hidden_inputs,
            controller_inputs,
            controller_outputs,
            lattice_inputs,
        },
    ))
}

pub fn compute_learning_rate(net: &NetConfig, batch_num: usize) -> f32 {
    let mut lr = net.learning_rate;

    if net.burn_in > 0 && batch_num < net.burn_in {
        let progress = batch_num as f32 / net.burn_in as f32;
        return net.learning_rate * progress.powf(net.power);
    }

    match net.policy {
        LrPolicy::Constant => lr,
        LrPolicy::Steps => {
            for (step, scale) in net.steps.iter().zip(&net.scales) {
                if batch_num >= *step {
                    lr *= *scale;
                }
            }
            lr
        }
        LrPolicy::Step => {
            net.learning_rate
                * net
                    .scale
                    .powi((batch_num / net.step.max(1)).try_into().unwrap_or(i32::MAX))
        }
        LrPolicy::Exp => net.gamma.powi(batch_num.try_into().unwrap_or(i32::MAX)) * lr,
        LrPolicy::Sig => {
            net.learning_rate / (1.0 + (net.gamma * (batch_num as f32 - net.step as f32)).exp())
        }
        LrPolicy::Sgdr => {
            let mut cycle_start = 0usize;
            let mut cycle_size = net.sgdr_cycle.max(1);
            while cycle_start.saturating_add(cycle_size) < batch_num {
                cycle_start = cycle_start.saturating_add(cycle_size);
                cycle_size = cycle_size.saturating_mul(net.sgdr_mult.max(1)).max(1);
            }
            let phase = (batch_num.saturating_sub(cycle_start)) as f32 / cycle_size as f32;
            net.learning_rate_min
                + 0.5
                    * (net.learning_rate - net.learning_rate_min)
                    * (1.0 + (std::f32::consts::PI * phase).cos())
        }
        LrPolicy::Poly => {
            if net.max_batches == 0 || batch_num >= net.max_batches {
                0.0
            } else {
                let progress = 1.0 - batch_num as f32 / net.max_batches as f32;
                net.learning_rate * progress.powf(net.power)
            }
        }
    }
}

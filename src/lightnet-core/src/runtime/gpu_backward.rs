use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Accumulate seeded GPU backward and return the network-input gradient.
    /// This opt-in diagnostic path may download the input gradient; ordinary
    /// training calls do not retain or transfer it.
    pub fn backward_accumulate_seeded_gpu_with_input_gradient(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<Option<Tensor>> {
        *self.captured_input_gradient.borrow_mut() = None;
        self.capture_input_gradient.set(true);
        let result = self.backward_accumulate_seeded_gpu(forward, layer_output_gradient_seeds);
        self.capture_input_gradient.set(false);
        result?;
        Ok(self.captured_input_gradient.borrow_mut().take())
    }

    /// Accumulates the layer gradients into the device train buffers.  With
    /// `loss_scale != 1` every seed is multiplied by it first (the matching
    /// division happens in `update_weights_gpu`).
    pub fn backward_accumulate_seeded_gpu(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        let loss_scale = self.network.net.loss_scale;
        if loss_scale == 1.0 || !(loss_scale > 0.0 && loss_scale.is_finite()) {
            return self.backward_accumulate_gpu_unscaled(forward, layer_output_gradient_seeds);
        }
        let scaled: Vec<Option<Tensor>> = layer_output_gradient_seeds
            .iter()
            .map(|seed| {
                seed.as_ref().map(|seed| Tensor {
                    shape: seed.shape,
                    data: seed.data.iter().map(|g| g * loss_scale).collect(),
                })
            })
            .collect();
        self.backward_accumulate_gpu_unscaled(forward, &scaled)
    }

    pub(super) fn backward_accumulate_gpu_unscaled(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
        layer_output_gradient_seeds: &[Option<Tensor>],
    ) -> Result<()> {
        reject_gpu_stage1_cpu_layers(&self.network, "backward")?;
        if layer_output_gradient_seeds.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "GPU backward seed count mismatch: expected {}, got {}",
                self.network.layers.len(),
                layer_output_gradient_seeds.len()
            )));
        }
        if self.requires_cpu_backward_fallback() || has_host_only_deform(&self.network) {
            return self
                .backward_accumulate_seeded_gpu_cpu_fallback(forward, layer_output_gradient_seeds);
        }
        let train_buffers_uninitialized = {
            let conv = self.device_train_buffers.borrow();
            let moe = self.device_moe_train_buffers.borrow();
            let eml = self.device_eml_train_buffers.borrow();
            let dcnv4 = self.device_dcnv4_train_buffers.borrow();
            conv.len() != self.network.layers.len()
                || moe.len() != self.network.layers.len()
                || eml.len() != self.network.layers.len()
                || dcnv4.len() != self.network.layers.len()
                || (conv.iter().all(Option::is_none)
                    && moe.iter().all(Option::is_none)
                    && eml.iter().all(Option::is_none)
                    && dcnv4.iter().all(Option::is_none))
        };
        if train_buffers_uninitialized {
            self.init_train_buffers_gpu()?;
        }

        let config = ElementwiseLaunchConfig::default();
        let mut layer_output_gradients: Vec<Option<B::Tensor>> =
            vec![None; self.network.layers.len()];
        for (layer_index, seed) in layer_output_gradient_seeds.iter().enumerate() {
            if let Some(seed) = seed {
                layer_output_gradients[layer_index] = Some(
                    self.backend
                        .upload_f32(&tensor_backend_shape(seed.shape), &seed.data)
                        .map_err(|err| backend_error("upload GPU backward seed", err))?,
                );
            }
        }
        let mut foveation_hf_auxiliary_gradients: Vec<Option<Vec<f32>>> =
            vec![None; self.network.layers.len()];

        for layer in self.network.layers.iter().rev() {
            let Some(grad) = layer_output_gradients[layer.index].take() else {
                continue;
            };

            match (&layer.kind, self.params.get(layer.index)) {
                (LayerKind::FoveationFilm(spec), Some(LayerParams::FoveationFilm(params))) => {
                    let film_cache = forward
                        .cpu_forward
                        .foveation_film_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Error::InvalidConfig(format!(
                            "missing [foveation_film] forward cache for layer #{}",
                            layer.index
                        )))?;
                    let hf_cache = forward
                        .cpu_forward
                        .hyperbolic_foveation_caches
                        .get(spec.hf_layer)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Error::InvalidConfig(format!(
                            "[foveation_film] selected HF layer #{} has no forward cache",
                            spec.hf_layer
                        )))?;
                    let current_input = if layer.index == 0 {
                        &forward.input_gpu
                    } else {
                        forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous GPU output for FoveationFilm layer #{}",
                                layer.index
                            ))
                        })?
                    };
                    let batch = layer.input_shape.batch;
                    let channels = layer.input_shape.channels;
                    let hf_params = self.backend
                        .upload_f32(&[batch, spec.param_count], &hf_cache.params_per_batch)
                        .map_err(|err| backend_error("upload FoveationFilm backward HF cache", err))?;
                    let embedding = self.backend
                        .upload_f32(&[batch, spec.embed_dim], &film_cache.embedding)
                        .map_err(|err| backend_error("upload FoveationFilm embedding cache", err))?;
                    let gamma = self.backend
                        .upload_f32(&[batch, channels], &film_cache.gamma)
                        .map_err(|err| backend_error("upload FoveationFilm gamma cache", err))?;
                    let w_embed = self.backend
                        .upload_f32(&[spec.embed_dim, spec.param_count], &params.w_embed)
                        .map_err(|err| backend_error("upload FoveationFilm embedding weights", err))?;
                    let w_gamma = self.backend
                        .upload_f32(&[channels, spec.embed_dim], &params.w_gamma)
                        .map_err(|err| backend_error("upload FoveationFilm gamma weights", err))?;
                    let w_beta = self.backend
                        .upload_f32(&[channels, spec.embed_dim], &params.w_beta)
                        .map_err(|err| backend_error("upload FoveationFilm beta weights", err))?;
                    let gradients = self.backend
                        .foveation_film_backward_f32(
                            current_input,
                            &grad,
                            &hf_params,
                            &embedding,
                            &gamma,
                            &w_embed,
                            &w_gamma,
                            &w_beta,
                            &tensor_backend_shape(layer.input_shape),
                            spec.param_count,
                            spec.embed_dim,
                            config,
                        )
                        .map_err(|err| backend_error("FoveationFilm native backward", err))?;
                    let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
                    let Some(LayerTrainBuffers::FoveationFilm(train)) =
                        buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing FoveationFilm train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    accumulate_vec(&mut train.w_embed_updates, &self.backend.download_f32(&gradients.w_embed)
                        .map_err(|err| backend_error("download FoveationFilm embedding gradient", err))?)?;
                    accumulate_vec(&mut train.b_embed_updates, &self.backend.download_f32(&gradients.b_embed)
                        .map_err(|err| backend_error("download FoveationFilm embedding bias gradient", err))?)?;
                    accumulate_vec(&mut train.w_gamma_updates, &self.backend.download_f32(&gradients.w_gamma)
                        .map_err(|err| backend_error("download FoveationFilm gamma gradient", err))?)?;
                    accumulate_vec(&mut train.b_gamma_updates, &self.backend.download_f32(&gradients.b_gamma)
                        .map_err(|err| backend_error("download FoveationFilm gamma bias gradient", err))?)?;
                    accumulate_vec(&mut train.w_beta_updates, &self.backend.download_f32(&gradients.w_beta)
                        .map_err(|err| backend_error("download FoveationFilm beta gradient", err))?)?;
                    accumulate_vec(&mut train.b_beta_updates, &self.backend.download_f32(&gradients.b_beta)
                        .map_err(|err| backend_error("download FoveationFilm beta bias gradient", err))?)?;
                    drop(buffers);
                    if spec.backprop_to_hf {
                        let hf_gradient = self.backend.download_f32(&gradients.hf_params)
                            .map_err(|err| backend_error("download FoveationFilm HF gradient", err))?;
                        let slot = foveation_hf_auxiliary_gradients
                            .get_mut(spec.hf_layer)
                            .ok_or_else(|| Error::InvalidConfig(format!(
                                "FoveationFilm selected HF layer #{} is unavailable during backward",
                                spec.hf_layer
                            )))?;
                        if let Some(existing) = slot {
                            if existing.len() != hf_gradient.len() {
                                return Err(Error::InvalidConfig(format!(
                                    "FoveationFilm HF gradient layout mismatch at layer #{}",
                                    spec.hf_layer
                                )));
                            }
                            for (sum, value) in existing.iter_mut().zip(hf_gradient) {
                                *sum += value;
                            }
                        } else {
                            *slot = Some(hf_gradient);
                        }
                    }
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        gradients.input,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::HyperbolicFoveation(spec),
                    Some(LayerParams::HyperbolicFoveation(params)),
                ) => {
                    let cache = forward
                        .cpu_forward
                        .hyperbolic_foveation_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| Error::InvalidConfig(format!(
                            "missing [hyperbolic_foveation] forward cache for layer #{}",
                            layer.index
                        )))?;
                    let network_input = self.download_tensor(
                        &forward.input_gpu,
                        self.network.net.input_shape(),
                    ).map_err(|err| backend_error("download HF network input", err))?;
                    let previous_input = if layer.index == 0 {
                        &network_input
                    } else {
                        forward.cpu_forward.layer_outputs.get(layer.index - 1).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing host input for hyperbolic_foveation layer #{}",
                                layer.index
                            ))
                        })?
                    };
                    let warp_input = match spec.warp_source {
                        crate::model::HfWarpSource::Previous => previous_input,
                        crate::model::HfWarpSource::NetworkInput => &network_input,
                    };
                    let output_gradient = self.download_tensor(&grad, layer.output_shape)
                        .map_err(|err| backend_error("download HF output gradient", err))?;
                    let warp_input_gpu = match spec.warp_source {
                        crate::model::HfWarpSource::Previous if layer.index > 0 => {
                            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing GPU warp source for hyperbolic_foveation layer #{}",
                                    layer.index
                                ))
                            })?
                        }
                        _ => &forward.input_gpu,
                    };
                    let grid_gpu = self.backend.upload_f32(
                        &[warp_input.shape.batch, warp_input.shape.height, warp_input.shape.width, 2],
                        &cache.grid,
                    ).map_err(|err| backend_error("upload HF backward grid", err))?;
                    let (native_warp_gradient, native_grid_gradient) = self.backend
                        .hyperbolic_foveation_warp_backward_f32(
                            warp_input_gpu,
                            &grid_gpu,
                            &grad,
                            &tensor_backend_shape(warp_input.shape),
                            config,
                        )
                        .map_err(|err| backend_error("native HF warp backward", err))?;
                    let batch = warp_input.shape.batch;
                    let param_count = spec.param_count();
                    let params_gpu = self.backend.upload_f32(
                        &[batch, param_count], &cache.params_per_batch,
                    ).map_err(|err| backend_error("upload HF inverse parameters", err))?;
                    let shape = tensor_backend_shape(warp_input.shape);
                    let native_semantic_gradient = self.backend
                        .hyperbolic_foveation_inverse_param_grad_f32(
                            &params_gpu, &native_grid_gradient, &shape, param_count,
                            spec.alpha, spec.p_exp, spec.nr_iters, spec.nr_eta,
                            spec.inverse_method == crate::model::HfInverseMethod::Newton, config,
                        )
                        .map_err(|err| backend_error("native HF inverse parameter adjoint", err))?
                        .map(|semantic| self.backend.download_f32(&semantic)
                            .map_err(|err| backend_error("download HF semantic parameter gradients", err)))
                        .transpose()?;
                    let native_grid_gradient = self.backend.download_f32(&native_grid_gradient)
                        .map_err(|err| backend_error("download native HF grid gradient", err))?;
                    let auxiliary = foveation_hf_auxiliary_gradients
                        .get(layer.index)
                        .and_then(Option::as_deref);
                    let (_host_warp_gradient, mut predictor_gradient, mut param_gradient, constrained_seed) =
                        backward_hyperbolic_foveation(
                            spec, params, warp_input, &output_gradient, cache, auxiliary,
                            Some(&native_grid_gradient), native_semantic_gradient.as_deref(),
                            true,
                        )?;
                    if spec.pred_hidden > 0 {
                        let predictor = cache.predictor.as_ref().ok_or_else(|| Error::InvalidConfig(
                            "missing HF predictor cache for native backward".to_owned(),
                        ))?;
                        let batch = warp_input.shape.batch;
                        let input_dim = predictor.pooled.len() / batch;
                        let pooled = self.backend.upload_f32(&[batch, input_dim], &predictor.pooled)
                            .map_err(|err| backend_error("upload HF predictor pooled cache", err))?;
                        let preactivation = self.backend.upload_f32(&[batch, spec.pred_hidden], &predictor.hidden_preactivation)
                            .map_err(|err| backend_error("upload HF predictor preactivation cache", err))?;
                        let hidden = self.backend.upload_f32(&[batch, spec.pred_hidden], &predictor.hidden)
                            .map_err(|err| backend_error("upload HF predictor hidden cache", err))?;
                        let raw = self.backend.upload_f32(&[batch, spec.param_count()], &predictor.raw_output)
                            .map_err(|err| backend_error("upload HF predictor raw cache", err))?;
                        let seed = self.backend.upload_f32(&[batch, spec.param_count()], &constrained_seed)
                            .map_err(|err| backend_error("upload HF predictor adjoint seed", err))?;
                        let w1 = self.backend.upload_f32(&[spec.pred_hidden, input_dim], &params.pred_w1)
                            .map_err(|err| backend_error("upload HF predictor w1", err))?;
                        let w2 = self.backend.upload_f32(&[spec.param_count(), spec.pred_hidden], &params.pred_w2)
                            .map_err(|err| backend_error("upload HF predictor w2", err))?;
                        let (source_gradient, w1_and_b1, w2_and_b2) = self.backend
                            .hyperbolic_foveation_predictor_backward_f32(
                                &pooled, &preactivation, &hidden, &raw, &seed, &w1, &w2,
                                &tensor_backend_shape(spec.pred_input_shape), spec.downsample,
                                spec.pred_hidden, spec.param_count(), config,
                            )
                            .map_err(|err| backend_error("native HF predictor backward", err))?;
                        let w1_and_b1 = self.backend.download_f32(&w1_and_b1)
                            .map_err(|err| backend_error("download HF predictor w1/b1 gradients", err))?;
                        let w2_and_b2 = self.backend.download_f32(&w2_and_b2)
                            .map_err(|err| backend_error("download HF predictor w2/b2 gradients", err))?;
                        let w1_count = param_gradient.pred_w1.len();
                        let w2_count = param_gradient.pred_w2.len();
                        if w1_and_b1.len() != w1_count + param_gradient.pred_b1.len()
                            || w2_and_b2.len() != w2_count + param_gradient.pred_b2.len() {
                            return Err(Error::InvalidConfig("native HF predictor gradient lengths mismatch".to_owned()));
                        }
                        param_gradient.pred_w1.copy_from_slice(&w1_and_b1[..w1_count]);
                        param_gradient.pred_b1.copy_from_slice(&w1_and_b1[w1_count..]);
                        param_gradient.pred_w2.copy_from_slice(&w2_and_b2[..w2_count]);
                        param_gradient.pred_b2.copy_from_slice(&w2_and_b2[w2_count..]);
                        let source_gradient = self.backend.download_f32(&source_gradient)
                            .map_err(|err| backend_error("download HF predictor source gradient", err))?;
                        predictor_gradient = Some((
                            spec.pred_source_layer,
                            Tensor::from_data(spec.pred_input_shape, source_gradient)?,
                        ));
                    }
                    {
                        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
                        let Some(LayerTrainBuffers::HyperbolicFoveation(train)) =
                            buffers.get_mut(layer.index)
                        else {
                            return Err(Error::InvalidConfig(format!(
                                "missing hyperbolic_foveation train buffers for layer #{}",
                                layer.index
                            )));
                        };
                        accumulate_vec(&mut train.hf_param_updates, &param_gradient.hf_params)?;
                        accumulate_vec(&mut train.pred_w1_updates, &param_gradient.pred_w1)?;
                        accumulate_vec(&mut train.pred_b1_updates, &param_gradient.pred_b1)?;
                        accumulate_vec(&mut train.pred_w2_updates, &param_gradient.pred_w2)?;
                        accumulate_vec(&mut train.pred_b2_updates, &param_gradient.pred_b2)?;
                    }
                    if spec.warp_source == crate::model::HfWarpSource::Previous {
                        self.accumulate_previous_gpu_gradient(
                            layer.index,
                            native_warp_gradient,
                            &mut layer_output_gradients,
                        )?;
                    }
                    if let Some((source_layer, source_gradient)) = predictor_gradient {
                        let source_gradient = self.backend.upload_f32(
                            &tensor_backend_shape(source_gradient.shape),
                            &source_gradient.data,
                        ).map_err(|err| backend_error("upload HF predictor source gradient", err))?;
                        if let Some(source_layer) = source_layer {
                            let Some(destination) = layer_output_gradients.get_mut(source_layer) else {
                                return Err(Error::InvalidConfig(format!(
                                    "HF predictor source layer #{source_layer} is unavailable during backward"
                                )));
                            };
                            self.accumulate_gpu_gradient(destination, source_gradient)?;
                        } else {
                            self.accumulate_previous_gpu_gradient(
                                layer.index,
                                source_gradient,
                                &mut layer_output_gradients,
                            )?;
                        }
                    }
                }
                (
                    LayerKind::Convolutional(spec),
                    Some(LayerParams::Convolutional(_) | LayerParams::None),
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    // The batch-statistics BN backward only needs the raw tensors.
                    let cached = if forward.gpu_batch_norm_caches[layer.index].is_some() {
                        self.cached_convolution_params_raw(
                            owner,
                            layer.input_shape.channels,
                            spec,
                            conv,
                        )?
                    } else {
                        self.cached_convolution_params(
                            owner,
                            layer.input_shape.channels,
                            spec,
                            conv,
                        )?
                    };
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for convolutional layer #{}",
                                layer.index
                            ))
                        })?;
                    let try_fix_nan = self.network.net.try_fix_nan;
                    // C++ `try_fix_nan`: clamp the output delta to [-1, 1]
                    // before the activation gradient.
                    let grad = if try_fix_nan {
                        self.backend
                            .constrain_f32(&grad, &tensor_backend_shape(layer.output_shape), 1.0, config)
                            .map_err(|err| backend_error("GPU try_fix_nan delta clamp", err))?
                    } else {
                        grad
                    };
                    let delta = if let Some(normalization) =
                        spec.activation.to_channel_normalization()
                    {
                        let output = &forward.gpu_layer_outputs[layer.index];
                        self.backend
                            .normalize_channels_backward_f32(
                                activation_input,
                                output,
                                &grad,
                                &tensor_backend_shape(layer.output_shape),
                                layer.output_shape.batch,
                                layer.output_shape.channels,
                                layer.output_shape.width * layer.output_shape.height,
                                normalization,
                                GradMode::from_env() == GradMode::Cpp,
                                config,
                            )
                            .map_err(|err| {
                                backend_error("GPU convolution channel-normalization backward", err)
                            })?
                    } else {
                        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                        self.backend
                            .activation_backward_f32(activation_input, &grad, activation, config)
                            .map_err(|err| backend_error("GPU convolution activation backward", err))?
                    };
                    let current_input =
                        if layer.index == 0 {
                            &forward.input_gpu
                        } else {
                            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous GPU layer output for convolutional layer #{}",
                                layer.index
                            ))
                        })?
                        };
                    let (bias_grad, scale_grad, grad_for_conv) = if let Some(cache) =
                        forward.gpu_batch_norm_caches[layer.index].as_ref()
                    {
                        // Batch-statistics batch norm: exact dx, dscale, dbias.
                        let (dx, dscale, dbias) = self
                            .backend
                            .batchnorm_train_backward_f32(
                                &delta,
                                &cache.x_hat,
                                &cache.variance,
                                &cache.scales,
                                &tensor_backend_shape(layer.output_shape),
                                config,
                            )
                            .map_err(|err| backend_error("GPU batch-norm backward", err))?;
                        (dbias, Some(dscale), dx)
                    } else {
                        let bias_grad = self
                            .backend
                            .bias_backward_f32(
                                &delta,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                config,
                            )
                            .map_err(|err| backend_error("GPU convolution bias backward", err))?;
                        let scale_grad = if let Some(batch_norm) = conv.batch_norm.as_ref() {
                            let raw_output = self
                                .backend
                                .conv2d_f32(
                                    current_input,
                                    &cached.raw_weights,
                                    &tensor_backend_shape(layer.output_shape),
                                    layer.input_shape.channels,
                                    layer.input_shape.height,
                                    layer.input_shape.width,
                                    spec.filters,
                                    spec.size,
                                    spec.stride_x,
                                    spec.stride_y,
                                    spec.dilation,
                                    spec.padding,
                                    spec.groups.max(1),
                                    config,
                                )
                                .map_err(|err| backend_error("GPU batch-norm raw convolution", err))?;
                            let scale_grad = self.batch_norm_scale_gradient_gpu(
                                &raw_output,
                                &delta,
                                batch_norm,
                                layer.output_shape,
                            )?;
                            Some(scale_grad)
                        } else {
                            None
                        };
                        let grad_for_conv = if let Some(scale) = cached.batch_norm_scales.as_ref() {
                            self.backend
                                .channel_scale_f32(
                                    &delta,
                                    scale,
                                    spec.filters,
                                    layer.output_shape.width * layer.output_shape.height,
                                    config,
                                )
                                .map_err(|err| backend_error("GPU batch-norm delta scale", err))?
                        } else {
                            delta.clone()
                        };
                        (bias_grad, scale_grad, grad_for_conv)
                    };
                    let grouped_channels = layer.input_shape.channels / spec.groups.max(1);
                    let weight_grad = self
                        .backend
                        .conv2d_backward_weights_f32(
                            current_input,
                            &grad_for_conv,
                            &[spec.filters, grouped_channels, spec.size, spec.size],
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.filters,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.dilation,
                            spec.padding,
                            spec.groups.max(1),
                            config,
                        )
                        .map_err(|err| backend_error("GPU convolution weight backward", err))?;
                    let input_grad = self
                        .backend
                        .conv2d_backward_input_f32(
                            &grad_for_conv,
                            &cached.raw_weights,
                            &tensor_backend_shape(layer.input_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.filters,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.dilation,
                            spec.padding,
                            spec.groups.max(1),
                            config,
                        )
                        .map_err(|err| backend_error("GPU convolution input backward", err))?;
                    let input_grad = if try_fix_nan {
                        self.backend
                            .sanitize_nan_inf_f32(
                                &input_grad,
                                &tensor_backend_shape(layer.input_shape),
                                NanInfMode::Reset,
                                config,
                            )
                            .map_err(|err| backend_error("GPU try_fix_nan input delta reset", err))?
                    } else {
                        input_grad
                    };
                    self.accumulate_device_train_buffer(owner, weight_grad, bias_grad, scale_grad)?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::BatchNorm(_), Some(LayerParams::BatchNorm(_))) => {
                    self.backward_batch_norm_gpu(
                        layer,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::GraphConv(spec), Some(LayerParams::GraphConv(params))) => {
                    if graph_conv_spec_needs_host(spec) {
                        return Err(Error::InvalidConfig(format!(
                            "native GraphConv backward reached unsupported layer #{}",
                            layer.index
                        )));
                    }
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for GraphConv layer #{}",
                                layer.index
                            ))
                        })?;
                    let activation_delta = if let Some(normalization) = spec.activation.to_channel_normalization()
                    {
                        self.backend
                            .normalize_channels_backward_f32(
                                activation_input,
                                &forward.gpu_layer_outputs[layer.index],
                                &grad,
                                &tensor_backend_shape(layer.output_shape),
                                layer.output_shape.batch,
                                layer.output_shape.channels,
                                layer.output_shape.width * layer.output_shape.height,
                                normalization,
                                GradMode::from_env() == GradMode::Cpp,
                                config,
                            )
                            .map_err(|err| {
                                backend_error("GraphConv channel-normalization backward", err)
                            })?
                    } else {
                        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                        self.backend
                            .activation_backward_f32(activation_input, &grad, activation, config)
                            .map_err(|err| backend_error("GraphConv activation backward", err))?
                    };
                    let (delta, bn_bias_gradient, scale_gradient) =
                        if let Some(cache) = forward.gpu_batch_norm_caches[layer.index].as_ref() {
                            let (delta, scale_gradient, bias_gradient) = self
                                .backend
                                .batchnorm_train_backward_f32(
                                    &activation_delta,
                                    &cache.x_hat,
                                    &cache.variance,
                                    &cache.scales,
                                    &tensor_backend_shape(layer.output_shape),
                                    config,
                                )
                                .map_err(|err| {
                                    backend_error("GraphConv GPU batch-norm backward", err)
                                })?;
                            (delta, Some(bias_gradient), Some(scale_gradient))
                        } else {
                            (activation_delta, None, None)
                        };
                    let groups = spec.groups.max(1);
                    let current_input = if layer.index == 0 {
                        &forward.input_gpu
                    } else {
                        &forward.gpu_layer_outputs[layer.index - 1]
                    };
                    let geometry = graph_conv_aggregation_geometry(
                        spec,
                        layer.input_shape,
                        layer.output_shape,
                    );
                    let (aggregate, edge_kernel, edge_biases) = if spec.edge_mode == 0 {
                        (
                            self.backend
                                .graph_conv_aggregate_f32(current_input, geometry, config)
                                .map_err(|err| backend_error("GraphConv backward aggregation", err))?,
                            None,
                            None,
                        )
                    } else {
                        let edge_kernel_values = params.edge_kernel.as_deref().ok_or_else(|| {
                            Error::InvalidConfig("GraphConv learned edge kernel is missing".to_owned())
                        })?;
                        let edge_bias_values = params.edge_biases.as_deref().ok_or_else(|| {
                            Error::InvalidConfig("GraphConv learned edge biases are missing".to_owned())
                        })?;
                        let edge_kernel = self.backend
                            .upload_f32(&[edge_kernel_values.len()], edge_kernel_values)
                            .map_err(|err| backend_error("upload GraphConv backward edge kernel", err))?;
                        let edge_biases = self.backend
                            .upload_f32(&[edge_bias_values.len()], edge_bias_values)
                            .map_err(|err| backend_error("upload GraphConv backward edge biases", err))?;
                        let aggregate = self.backend
                            .graph_conv_learned_aggregate_f32(
                                current_input,
                                &edge_kernel,
                                &edge_biases,
                                geometry,
                                groups,
                                config,
                            )
                            .map_err(|err| backend_error("GraphConv learned backward aggregation", err))?;
                        (aggregate, Some(edge_kernel), Some(edge_biases))
                    };
                    let weights = self
                        .backend
                        .upload_f32(
                            &[
                                spec.filters,
                                layer.input_shape.channels / groups,
                                1,
                                1,
                            ],
                            &params.projection.weights,
                        )
                        .map_err(|err| backend_error("upload GraphConv backward weights", err))?;
                    let aggregate_shape = [
                        layer.input_shape.batch,
                        layer.input_shape.channels,
                        layer.output_shape.height,
                        layer.output_shape.width,
                    ];
                    let bias_gradient = match bn_bias_gradient {
                        Some(gradient) => gradient,
                        None => self
                            .backend
                            .bias_backward_f32(
                                &delta,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                config,
                            )
                            .map_err(|err| backend_error("GraphConv bias backward", err))?,
                    };
                    let weight_gradient = self
                        .backend
                        .conv2d_backward_weights_f32(
                            &aggregate,
                            &delta,
                            &[
                                spec.filters,
                                layer.input_shape.channels / groups,
                                1,
                                1,
                            ],
                            layer.input_shape.channels,
                            layer.output_shape.height,
                            layer.output_shape.width,
                            spec.filters,
                            1,
                            1,
                            1,
                            1,
                            0,
                            groups,
                            config,
                        )
                        .map_err(|err| backend_error("GraphConv weight backward", err))?;
                    let aggregate_gradient = self
                        .backend
                        .conv2d_backward_input_f32(
                            &delta,
                            &weights,
                            &aggregate_shape,
                            layer.input_shape.channels,
                            layer.output_shape.height,
                            layer.output_shape.width,
                            spec.filters,
                            1,
                            1,
                            1,
                            1,
                            0,
                            groups,
                            config,
                        )
                        .map_err(|err| backend_error("GraphConv projection input backward", err))?;
                    let (input_gradient, edge_kernel_values, edge_bias_values) =
                        if let (Some(edge_kernel), Some(edge_biases)) =
                            (edge_kernel.as_ref(), edge_biases.as_ref())
                        {
                            let (input_gradient, edge_kernel_gradient, edge_bias_gradient) = self
                                .backend
                                .graph_conv_learned_aggregate_backward_f32(
                                    current_input,
                                    edge_kernel,
                                    edge_biases,
                                    &aggregate_gradient,
                                    geometry,
                                    groups,
                                    config,
                                )
                                .map_err(|err| {
                                    backend_error("GraphConv learned aggregation backward", err)
                                })?;
                            let edge_kernel_values = self.backend
                                .download_f32(&edge_kernel_gradient)
                                .map_err(|err| backend_error("download GraphConv edge-kernel gradient", err))?;
                            let edge_bias_values = self.backend
                                .download_f32(&edge_bias_gradient)
                                .map_err(|err| backend_error("download GraphConv edge-bias gradient", err))?;
                            (input_gradient, Some(edge_kernel_values), Some(edge_bias_values))
                        } else {
                            let input_gradient = self
                                .backend
                                .graph_conv_aggregate_backward_f32(&aggregate_gradient, geometry, config)
                                .map_err(|err| backend_error("GraphConv aggregation backward", err))?;
                            (input_gradient, None, None)
                        };
                    let (input_gradient, self_weight_values) = if spec.use_self {
                        let reference = self.backend
                            .graph_conv_reference_f32(current_input, geometry, config)
                            .map_err(|err| backend_error("GraphConv backward center-reference gather", err))?;
                        let self_values = params.self_weights.as_deref().ok_or_else(|| {
                            Error::InvalidConfig("GraphConv self weights are missing".to_owned())
                        })?;
                        let self_weights = self.backend
                            .upload_f32(
                                &[spec.filters, layer.input_shape.channels / groups, 1, 1],
                                self_values,
                            )
                            .map_err(|err| backend_error("upload GraphConv backward self weights", err))?;
                        let self_weight_gradient = self.backend
                            .conv2d_backward_weights_f32(
                                &reference,
                                &delta,
                                &[spec.filters, layer.input_shape.channels / groups, 1, 1],
                                layer.input_shape.channels,
                                layer.output_shape.height,
                                layer.output_shape.width,
                                spec.filters,
                                1,
                                1,
                                1,
                                1,
                                0,
                                groups,
                                config,
                            )
                            .map_err(|err| backend_error("GraphConv self-weight backward", err))?;
                        let reference_gradient = self.backend
                            .conv2d_backward_input_f32(
                                &delta,
                                &self_weights,
                                &aggregate_shape,
                                layer.input_shape.channels,
                                layer.output_shape.height,
                                layer.output_shape.width,
                                spec.filters,
                                1,
                                1,
                                1,
                                1,
                                0,
                                groups,
                                config,
                            )
                            .map_err(|err| backend_error("GraphConv self-projection input backward", err))?;
                        let self_input_gradient = self.backend
                            .graph_conv_reference_backward_f32(&reference_gradient, geometry, config)
                            .map_err(|err| backend_error("GraphConv center-reference backward", err))?;
                        let input_gradient = self.backend
                            .shortcut_add_f32(&input_gradient, &self_input_gradient, config)
                            .map_err(|err| backend_error("GraphConv self input-gradient sum", err))?;
                        let values = self.backend
                            .download_f32(&self_weight_gradient)
                            .map_err(|err| backend_error("download GraphConv self-weight gradient", err))?;
                        (input_gradient, Some(values))
                    } else {
                        (input_gradient, None)
                    };
                    let weight_values = self
                        .backend
                        .download_f32(&weight_gradient)
                        .map_err(|err| backend_error("download GraphConv weight gradient", err))?;
                    let bias_values = self
                        .backend
                        .download_f32(&bias_gradient)
                        .map_err(|err| backend_error("download GraphConv bias gradient", err))?;
                    let scale_values = scale_gradient
                        .as_ref()
                        .map(|gradient| self.backend.download_f32(gradient))
                        .transpose()
                        .map_err(|err| backend_error("download GraphConv BN scale gradient", err))?;
                    {
                        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
                        let Some(LayerTrainBuffers::GraphConv(train)) =
                            buffers.get_mut(layer.index)
                        else {
                            return Err(Error::InvalidConfig(format!(
                                "missing GraphConv host gradient buffers for layer #{}",
                                layer.index
                            )));
                        };
                        for (dst, src) in train
                            .projection
                            .weight_updates
                            .iter_mut()
                            .zip(weight_values)
                        {
                            *dst += src;
                        }
                        for (dst, src) in train
                            .projection
                            .bias_updates
                            .iter_mut()
                            .zip(bias_values)
                        {
                            *dst += src;
                        }
                        if let Some(values) = scale_values {
                            let Some(batch_norm) = train.projection.batch_norm.as_mut() else {
                                return Err(Error::InvalidConfig(
                                    "missing GraphConv BN scale gradient buffer".to_owned(),
                                ));
                            };
                            for (dst, src) in batch_norm.scale_updates.iter_mut().zip(values) {
                                *dst += src;
                            }
                        }
                        if let Some(values) = self_weight_values {
                            let Some(updates) = train.self_weight_updates.as_mut() else {
                                return Err(Error::InvalidConfig(
                                    "missing GraphConv self-weight gradient buffer".to_owned(),
                                ));
                            };
                            for (dst, src) in updates.iter_mut().zip(values) {
                                *dst += src;
                            }
                        }
                        if let Some(values) = edge_kernel_values {
                            let Some(updates) = train.edge_kernel_updates.as_mut() else {
                                return Err(Error::InvalidConfig(
                                    "missing GraphConv edge-kernel gradient buffer".to_owned(),
                                ));
                            };
                            for (dst, src) in updates.iter_mut().zip(values) {
                                *dst += src;
                            }
                        }
                        if let Some(values) = edge_bias_values {
                            let Some(updates) = train.edge_bias_updates.as_mut() else {
                                return Err(Error::InvalidConfig(
                                    "missing GraphConv edge-bias gradient buffer".to_owned(),
                                ));
                            };
                            for (dst, src) in updates.iter_mut().zip(values) {
                                *dst += src;
                            }
                        }
                    }
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::MaxPool(spec), Some(LayerParams::None)) => {
                    self.backward_maxpool_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LocalAvgPool(spec), Some(LayerParams::None)) => {
                    self.backward_local_avgpool_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::AvgPool, Some(LayerParams::None)) => {
                    self.backward_avgpool_gpu(
                        layer,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crop(_), Some(LayerParams::None)) => {
                    self.backward_crop_gpu(
                        layer,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Route(spec), Some(LayerParams::None)) => {
                    self.backward_route_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LayaMarkerGather(spec), Some(LayerParams::None)) => {
                    self.backward_laya_marker_gather_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::LayaActHead(_), Some(LayerParams::LayaActHead(_))) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: "laya_act_head GPU backward".to_owned(),
                    });
                }
                (LayerKind::Shortcut(spec), Some(LayerParams::Shortcut(params))) => {
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for weighted shortcut layer #{}",
                                layer.index
                            ))
                        })?;
                    let delta = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, activation, config)
                        .map_err(|err| {
                            backend_error("GPU weighted shortcut activation backward", err)
                        })?;
                    if layer.input_shape != layer.output_shape {
                        return Err(Error::InvalidConfig(format!(
                            "weighted shortcut layer #{} input/output shapes differ: {} vs {}",
                            layer.index, layer.input_shape, layer.output_shape
                        )));
                    }
                    let per_sample = layer.output_shape.element_count();
                    let layer_step = spec.weight_slots(layer.output_shape.channels);
                    if layer_step == 0
                        || per_sample % layer_step != 0
                        || params.weights.len() != spec.weight_count(layer.output_shape.channels)
                    {
                        return Err(Error::InvalidConfig(format!(
                            "weighted shortcut layer #{} has inconsistent weight geometry",
                            layer.index
                        )));
                    }
                    let slots_per_feature = per_sample / layer_step;
                    let effective = crate::shortcut_weights::effective_weights(
                        &params.weights,
                        spec.weights_normalization,
                        layer_step,
                    );
                    let scale_pattern = |source_slot: usize| {
                        (0..layer.output_shape.batch * per_sample)
                            .map(|index| {
                                let feature = index % per_sample;
                                effective[feature / slots_per_feature + source_slot * layer_step]
                            })
                            .collect::<Vec<_>>()
                    };
                    let backend_shape = tensor_backend_shape(layer.output_shape);
                    let input_scales = self
                        .backend
                        .upload_f32(&backend_shape, &scale_pattern(0))
                        .map_err(|err| {
                            backend_error("upload weighted shortcut input scales", err)
                        })?;
                    let input_gradient = self
                        .backend
                        .vector_mul_f32(&delta, &input_scales, &backend_shape, config)
                        .map_err(|err| backend_error("weighted shortcut input gradient", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_gradient,
                        &mut layer_output_gradients,
                    )?;

                    let num_inputs = spec.from.len() + 1;
                    let mut raw_weight_products = self
                        .backend
                        .zeros_f32(&[num_inputs, layer_step])
                        .map_err(|err| backend_error("allocate weighted shortcut products", err))?;
                    let input_gpu = if layer.index == 0 {
                        &forward.input_gpu
                    } else {
                        &forward.gpu_layer_outputs[layer.index - 1]
                    };
                    raw_weight_products = self
                        .backend
                        .shortcut_weight_gradient_accumulate_f32(
                            &raw_weight_products,
                            &delta,
                            input_gpu,
                            layer.output_shape.batch,
                            per_sample,
                            layer.input_shape.element_count(),
                            num_inputs,
                            layer_step,
                            0,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("weighted shortcut input weight reduction", err)
                        })?;
                    let mut source_shapes = Vec::with_capacity(spec.from.len());
                    for (source_slot, raw_index) in spec.from.iter().enumerate() {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source_layer = self.network.layers.get(source_index).ok_or(
                            Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            },
                        )?;
                        let source_shape = source_layer.output_shape;
                        if source_shape.batch != layer.output_shape.batch {
                            return Err(Error::InvalidConfig(format!(
                                "weighted shortcut layer #{} mixes incompatible batch shapes",
                                layer.index
                            )));
                        }
                        let source_gpu = forward.gpu_layer_outputs.get(source_index).ok_or(
                            Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: *raw_index,
                            },
                        )?;
                        raw_weight_products = self
                            .backend
                            .shortcut_weight_gradient_accumulate_f32(
                                &raw_weight_products,
                                &delta,
                                source_gpu,
                                layer.output_shape.batch,
                                per_sample,
                                source_shape.element_count(),
                                num_inputs,
                                layer_step,
                                source_slot + 1,
                                config,
                            )
                            .map_err(|err| {
                                backend_error("weighted shortcut source weight reduction", err)
                            })?;
                        source_shapes.push(source_shape);
                    }
                    let raw_weights = self
                        .backend
                        .upload_f32(&[num_inputs, layer_step], &params.weights)
                        .map_err(|err| backend_error("upload weighted shortcut raw weights", err))?;
                    let normalization = match spec.weights_normalization {
                        crate::model::ShortcutWeightsNorm::None => {
                            lightnet_cuda::F32ShortcutWeightNormalization::None
                        }
                        crate::model::ShortcutWeightsNorm::Relu => {
                            lightnet_cuda::F32ShortcutWeightNormalization::Relu
                        }
                        crate::model::ShortcutWeightsNorm::Softmax => {
                            lightnet_cuda::F32ShortcutWeightNormalization::Softmax
                        }
                    };
                    let weight_gradient = self
                        .backend
                        .shortcut_weight_normalized_gradient_f32(
                            &raw_weights,
                            &raw_weight_products,
                            num_inputs,
                            layer_step,
                            normalization,
                            GradMode::from_env() == GradMode::Cpp,
                            config,
                        )
                        .map_err(|err| {
                            backend_error("weighted shortcut normalization derivative", err)
                        })?;
                    let weight_gradient = self
                        .backend
                        .download_f32(&weight_gradient)
                        .map_err(|err| {
                            backend_error("download weighted shortcut weight gradient", err)
                        })?;
                    let mut train_buffers = self.cpu_fallback_train_buffers.borrow_mut();
                    let Some(LayerTrainBuffers::Shortcut(train)) =
                        train_buffers.get_mut(layer.index)
                    else {
                        return Err(Error::InvalidConfig(format!(
                            "missing weighted shortcut train buffers for layer #{}",
                            layer.index
                        )));
                    };
                    if train.weight_updates.len() != weight_gradient.len() {
                        return Err(Error::InvalidConfig(format!(
                            "weighted shortcut train buffer mismatch for layer #{}",
                            layer.index
                        )));
                    }
                    for (update, gradient) in
                        train.weight_updates.iter_mut().zip(weight_gradient)
                    {
                        *update += gradient;
                    }
                    drop(train_buffers);

                    for (source_slot, (raw_index, source_shape)) in
                        spec.from.iter().zip(source_shapes).enumerate()
                    {
                        let source_scales = self
                            .backend
                            .upload_f32(&backend_shape, &scale_pattern(source_slot + 1))
                            .map_err(|err| {
                                backend_error("upload weighted shortcut source scales", err)
                            })?;
                        let output_gradient = self
                            .backend
                            .vector_mul_f32(&delta, &source_scales, &backend_shape, config)
                            .map_err(|err| {
                                backend_error("weighted shortcut source gradient", err)
                            })?;
                        let source_gradient = if source_shape != layer.output_shape {
                            self.backend
                                .shortcut_fit_f32(
                                    &output_gradient,
                                    source_shape.batch,
                                    &tensor_backend_shape(source_shape),
                                    config,
                                )
                                .map_err(|err| {
                                    backend_error("weighted shortcut source gradient fit", err)
                                })?
                        } else {
                            output_gradient
                        };
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        self.accumulate_gpu_gradient(
                            &mut layer_output_gradients[source_index],
                            source_gradient,
                        )?;
                    }
                }
                (LayerKind::Shortcut(spec), Some(LayerParams::None)) => {
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing GPU activation cache for shortcut layer #{}",
                                layer.index
                            ))
                        })?;
                    let accum_grad = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, activation, config)
                        .map_err(|err| backend_error("GPU shortcut activation backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        accum_grad.clone(),
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
                        // Mismatched source: crop / zero-pad the gradient flat
                        // per sample, like the forward add.
                        let source_grad = if source_shape != layer.output_shape {
                            self.backend
                                .shortcut_fit_f32(
                                    &accum_grad,
                                    source_shape.batch,
                                    &tensor_backend_shape(source_shape),
                                    config,
                                )
                                .map_err(|err| backend_error("GPU shortcut source gradient", err))?
                        } else {
                            accum_grad.clone()
                        };
                        self.accumulate_gpu_gradient(
                            &mut layer_output_gradients[source_index],
                            source_grad,
                        )?;
                    }
                }
                (LayerKind::Upsample(spec), Some(LayerParams::None)) => {
                    self.backward_upsample_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Region(spec), Some(LayerParams::None)) => {
                    let output = forward.gpu_layer_outputs.get(layer.index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing GPU Region output for layer #{}",
                            layer.index
                        ))
                    })?;
                    let input_grad = self
                        .backend
                        .region_backward_f32(
                            output,
                            &grad,
                            &tensor_backend_shape(layer.input_shape),
                            spec.num,
                            spec.coords + spec.classes + 1,
                            spec.classes,
                            spec.softmax,
                            config,
                        )
                        .map_err(|err| backend_error("GPU Region backward", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Softmax { groups }, Some(LayerParams::None)) => {
                    self.backward_softmax_gpu(
                        layer,
                        groups,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ChannelSlice(spec), Some(LayerParams::None)) => {
                    self.backward_channel_slice_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ChannelShuffle(spec), Some(LayerParams::None)) => {
                    self.backward_channel_shuffle_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Reorg(spec), Some(LayerParams::None)) => {
                    self.backward_reorg_gpu(
                        layer,
                        spec,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dropout { .. }, Some(LayerParams::None)) => {
                    let input_grad = match forward.cpu_forward.dropout_masks[layer.index].as_ref() {
                        Some(mask) => {
                            let mask_gpu = self
                                .backend
                                .upload_f32(&tensor_backend_shape(mask.shape), &mask.data)
                                .map_err(|err| backend_error("upload dropout mask", err))?;
                            self.backend
                                .dropout_backward_f32(
                                    &grad,
                                    &mask_gpu,
                                    &tensor_backend_shape(mask.shape),
                                    config,
                                )
                                .map_err(|err| backend_error("GPU dropout backward", err))?
                        }
                        None => grad,
                    };
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Sam(spec), Some(LayerParams::None)) => {
                    self.backward_sam_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::ScaleChannels(spec), Some(LayerParams::None)) => {
                    self.backward_scale_channels_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Local(spec),
                    Some(LayerParams::Local(params)),
                ) if spec.activation.to_cuda_f32_kind().is_some() => {
                    self.backward_local_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Deconvolutional(spec),
                    Some(LayerParams::Convolutional(params)),
                ) if deconvolutional_spec_is_native(spec) => {
                    self.backward_deconvolutional_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::Connected(spec),
                    Some(LayerParams::Connected(connected)),
                ) if spec.activation.to_cuda_f32_kind().is_some() => {
                    self.backward_connected_gpu(
                        layer,
                        spec,
                        connected,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::EmlConvolutional(spec),
                    Some(LayerParams::EmlConvolutional(params)),
                ) => {
                    self.backward_eml_convolutional_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Moe(spec), Some(LayerParams::Moe(params))) => {
                    let cache = forward
                        .gpu_moe_caches
                        .get(layer.index)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing MoE device cache for GPU layer #{}",
                                layer.index
                            ))
                        })?;
                    let batch = layer.input_shape.batch;
                    let channels = layer.input_shape.channels;
                    let experts = spec.experts;
                    let output_spatial = layer.output_shape.width * layer.output_shape.height;
                    let per_image = layer.output_shape.element_count();
                    if experts == 0
                        || cache.expert_activation_inputs.len() != experts
                        || params.experts.len() != experts
                        || params
                            .experts
                            .iter()
                            .any(|expert| expert.batch_norm.is_some() != spec.expert.batch_normalize)
                        || params.router_weights.len() != experts * channels
                        || params.router_biases.len() != experts
                    {
                        return Err(Error::InvalidConfig(format!(
                            "MoE GPU backward cache or parameter shape mismatch at layer #{}",
                            layer.index
                        )));
                    }
                    let current_input = if layer.index == 0 {
                        &forward.input_gpu
                    } else {
                        forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous GPU output for MoE layer #{}",
                                layer.index
                            ))
                        })?
                    };
                    let dlogit = self
                        .backend
                        .moe_router_backward_f32(
                            &cache.gates,
                            &grad,
                            &cache.expert_outputs,
                            batch,
                            experts,
                            per_image,
                            spec.load_balance,
                            crate::moe_conv::router_temperature(spec),
                            config,
                        )
                        .map_err(|err| backend_error("MoE router backward", err))?;
                    let pooled = cache.pooled.clone();
                    let router_weights = self
                        .backend
                        .upload_f32(&[experts, channels], &params.router_weights)
                        .map_err(|err| backend_error("upload MoE router weights", err))?;
                    let (pooled_gradient, router_weight_gradient, router_bias_gradient) = self
                        .backend
                        .connected_backward_f32(
                            &pooled,
                            &router_weights,
                            &dlogit,
                            batch,
                            channels,
                            experts,
                            config,
                        )
                        .map_err(|err| backend_error("MoE router connected backward", err))?;
                    let router_input_gradient = self
                        .backend
                        .moe_router_pool_backward_f32(
                            &pooled_gradient,
                            &tensor_backend_shape(layer.input_shape),
                            config,
                        )
                        .map_err(|err| backend_error("MoE router pool backward", err))?;

                    let activation = gpu_activation_kind(layer.index, &spec.expert.activation)?;
                    let mut expert_input_gradient = None;
                    let mut expert_weight_gradients = Vec::with_capacity(experts);
                    let mut expert_bias_gradients = Vec::with_capacity(experts);
                    let mut expert_scale_gradients = Vec::with_capacity(experts);
                    for expert_index in 0..experts {
                        let expert = &params.experts[expert_index];
                        let cached = self.cached_convolution_params_at(
                            layer.index,
                            expert_index,
                            channels,
                            &spec.expert,
                            expert,
                        )?;
                        let activation_input =
                            cache.expert_activation_inputs[expert_index].clone();
                        let sample_gates = self
                            .backend
                            .moe_gate_column_f32(
                                &cache.gates,
                                batch,
                                experts,
                                expert_index,
                                config,
                            )
                            .map_err(|err| backend_error("select MoE backward expert gates", err))?;
                        let gated_gradient = self
                            .backend
                            .channel_scale_f32(&grad, &sample_gates, batch, per_image, config)
                            .map_err(|err| backend_error("scale MoE expert output gradient", err))?;
                        let activation_delta = self
                            .backend
                            .activation_backward_f32(
                                &activation_input,
                                &gated_gradient,
                                activation,
                                config,
                            )
                            .map_err(|err| backend_error("MoE expert activation backward", err))?;
                        let (delta, bias_gradient, scale_gradient) =
                            if let Some(batch_norm) = expert.batch_norm.as_ref() {
                                validate_batchnorm_params(batch_norm, spec.expert.filters)?;
                                let raw_output = self
                                    .backend
                                    .conv2d_f32(
                                        current_input,
                                        &cached.raw_weights,
                                        &tensor_backend_shape(layer.output_shape),
                                        channels,
                                        layer.input_shape.height,
                                        layer.input_shape.width,
                                        spec.expert.filters,
                                        spec.expert.size,
                                        spec.expert.stride_x,
                                        spec.expert.stride_y,
                                        spec.expert.dilation,
                                        spec.expert.padding,
                                        spec.expert.groups.max(1),
                                        config,
                                    )
                                    .map_err(|err| backend_error("MoE raw expert convolution", err))?;
                                let mean = batch_norm
                                    .rolling_mean
                                    .iter()
                                    .map(|value| -*value)
                                    .collect::<Vec<_>>();
                                let inv_std = batch_norm
                                    .rolling_variance
                                    .iter()
                                    .map(|variance| {
                                        1.0 / (variance + BATCH_NORM_EPSILON).sqrt()
                                    })
                                    .collect::<Vec<_>>();
                                let centered = self
                                    .backend
                                    .upload_f32(&[spec.expert.filters], &mean)
                                    .and_then(|mean| {
                                        self.backend.bias_add_f32(
                                            &raw_output,
                                            &mean,
                                            spec.expert.filters,
                                            output_spatial,
                                            config,
                                        )
                                    })
                                    .map_err(|err| {
                                        backend_error("center MoE raw expert convolution", err)
                                    })?;
                                let inv_std_gpu = self
                                    .backend
                                    .upload_f32(&[spec.expert.filters], &inv_std)
                                    .map_err(|err| {
                                        backend_error("upload MoE expert BN inverse std", err)
                                    })?;
                                let x_hat = self
                                    .backend
                                    .channel_scale_f32(
                                        &centered,
                                        &inv_std_gpu,
                                        spec.expert.filters,
                                        output_spatial,
                                        config,
                                    )
                                    .map_err(|err| {
                                        backend_error("normalize MoE expert BN input", err)
                                    })?;
                                let scaled_delta = self
                                    .backend
                                    .vector_mul_f32(
                                        &activation_delta,
                                        &x_hat,
                                        &tensor_backend_shape(layer.output_shape),
                                        config,
                                    )
                                    .map_err(|err| {
                                        backend_error("multiply MoE expert BN scale gradient", err)
                                    })?;
                                let scale_gradient = self
                                    .backend
                                    .bias_backward_f32(
                                        &scaled_delta,
                                        spec.expert.filters,
                                        output_spatial,
                                        config,
                                    )
                                    .map_err(|err| {
                                        backend_error("MoE expert BN scale backward", err)
                                    })?;
                                let bias_gradient = self
                                    .backend
                                    .bias_backward_f32(
                                        &activation_delta,
                                        spec.expert.filters,
                                        output_spatial,
                                        config,
                                    )
                                    .map_err(|err| {
                                        backend_error("MoE expert BN bias backward", err)
                                    })?;
                                let affine = batch_norm
                                    .scales
                                    .iter()
                                    .zip(&inv_std)
                                    .map(|(scale, inv_std)| scale * inv_std)
                                    .collect::<Vec<_>>();
                                let affine_gpu = self
                                    .backend
                                    .upload_f32(&[spec.expert.filters], &affine)
                                    .map_err(|err| {
                                        backend_error("upload MoE expert BN affine gradient", err)
                                    })?;
                                let delta = self
                                    .backend
                                    .channel_scale_f32(
                                        &activation_delta,
                                        &affine_gpu,
                                        spec.expert.filters,
                                        output_spatial,
                                        config,
                                    )
                                    .map_err(|err| {
                                        backend_error("scale MoE expert BN gradient", err)
                                    })?;
                                (delta, bias_gradient, Some(scale_gradient))
                            } else {
                                let bias_gradient = self
                                    .backend
                                    .bias_backward_f32(
                                        &activation_delta,
                                        spec.expert.filters,
                                        output_spatial,
                                        config,
                                    )
                                    .map_err(|err| {
                                        backend_error("MoE expert bias backward", err)
                                    })?;
                                (activation_delta, bias_gradient, None)
                            };
                        let weight_gradient = self
                            .backend
                            .conv2d_backward_weights_f32(
                                current_input,
                                &delta,
                                &[
                                    spec.expert.filters,
                                    channels / spec.expert.groups.max(1),
                                    spec.expert.size,
                                    spec.expert.size,
                                ],
                                channels,
                                layer.input_shape.height,
                                layer.input_shape.width,
                                spec.expert.filters,
                                spec.expert.size,
                                spec.expert.stride_x,
                                spec.expert.stride_y,
                                spec.expert.dilation,
                                spec.expert.padding,
                                spec.expert.groups.max(1),
                                config,
                            )
                            .map_err(|err| backend_error("MoE expert weight backward", err))?;
                        let input_gradient = self
                            .backend
                            .conv2d_backward_input_f32(
                                &delta,
                                &cached.raw_weights,
                                &tensor_backend_shape(layer.input_shape),
                                channels,
                                layer.input_shape.height,
                                layer.input_shape.width,
                                spec.expert.filters,
                                spec.expert.size,
                                spec.expert.stride_x,
                                spec.expert.stride_y,
                                spec.expert.dilation,
                                spec.expert.padding,
                                spec.expert.groups.max(1),
                                config,
                            )
                            .map_err(|err| backend_error("MoE expert input backward", err))?;
                        self.accumulate_gpu_gradient(
                            &mut expert_input_gradient,
                            input_gradient,
                        )?;
                        expert_weight_gradients.push(weight_gradient);
                        expert_bias_gradients.push(bias_gradient);
                        expert_scale_gradients.push(scale_gradient);
                    }
                    self.accumulate_device_moe_train_buffer(
                        layer.index,
                        &router_weight_gradient,
                        &router_bias_gradient,
                        &expert_weight_gradients,
                        &expert_bias_gradients,
                        &expert_scale_gradients,
                    )?;
                    let expert_input_gradient = expert_input_gradient.ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "MoE layer #{} has no expert input gradients",
                            layer.index
                        ))
                    })?;
                    let input_gradient = self
                        .backend
                        .vector_add_f32(
                            &expert_input_gradient,
                            &router_input_gradient,
                            config,
                        )
                        .map_err(|err| backend_error("combine MoE input gradients", err))?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Rnn(spec), Some(LayerParams::Rnn(params))) => {
                    self.backward_rnn_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Lstm(spec), Some(LayerParams::Lstm(params))) => {
                    self.backward_lstm_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Crnn(spec), Some(LayerParams::Crnn(params))) => {
                    self.backward_crnn_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (
                    LayerKind::LayaScorer(spec),
                    Some(LayerParams::LayaScorer(weights)),
                ) => {
                    let current = forward.gpu_layer_outputs.get(layer.index - 1).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: -1,
                        },
                    )?;
                    let input_gradient = laya_scorer_gpu::backward_input(
                        &self.backend,
                        spec,
                        weights,
                        current,
                        &grad,
                        layer.input_shape,
                        layer.output_shape,
                        config,
                    )?;
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        input_gradient,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Yolo(_), Some(LayerParams::None))
                | (LayerKind::ObbYolo(_), Some(LayerParams::None))
                | (LayerKind::GaussianYolo(_), Some(LayerParams::None))
                | (LayerKind::Detection(_), Some(LayerParams::None))
                | (LayerKind::Cost { .. }, Some(LayerParams::None))
                | (LayerKind::Logistic, Some(LayerParams::None))
                | (LayerKind::Segmentation(_), Some(LayerParams::None))
                | (LayerKind::OmdLoss(_), Some(LayerParams::None))
                | (LayerKind::Contrastive(_), Some(LayerParams::None))
                | (LayerKind::Regression(_), Some(LayerParams::None))
                | (LayerKind::JevCe(_), Some(LayerParams::None))
                | (LayerKind::LayaLoss(_), Some(LayerParams::None))
                | (LayerKind::LayaEncoder(_), Some(LayerParams::LayaEncoder(_)))
                | (LayerKind::LayaHead(_), Some(LayerParams::LayaHead(_)))
                // Modern YOLO head loss/backward is host-owned; its forward
                // activation now runs natively on the device.
                | (LayerKind::Yolox(_), Some(LayerParams::None))
                | (LayerKind::Ppyoloe(_), Some(LayerParams::None))
                | (LayerKind::Yolonas(_), Some(LayerParams::None))
                | (LayerKind::CenterNet(_), Some(LayerParams::None)) => {
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Dcnv4(spec), Some(LayerParams::Dcnv4(params))) => {
                    self.backward_dcnv4_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DsConv(spec), Some(LayerParams::DsConv(params))) => {
                    self.backward_dsconv_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::DeformConv(spec), Some(LayerParams::DeformConv(params))) => {
                    // Native path. Device forward/backward; the only host hop is
                    // the batch-coupled offset-normalization Jacobian.
                    let (main_weights, offset_weights) = (
                        params.main.weights.clone(),
                        params.offset_weights.clone(),
                    );
                    let (mask_weights, mask_biases) =
                        (params.mask_weights.clone(), params.mask_biases.clone());
                    // Clone the device intermediates out of the RefCell so the
                    // following `&mut self` calls can borrow freely.
                    let (input_sanitized, raw_offsets, mean, inv_std, stddev, normalize, offsets, masks, mask_logits) = {
                        let caches = self.deform_device_caches.borrow();
                        let c = caches[layer.index].as_ref().ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing device [deform_conv] cache for layer #{}",
                                layer.index
                            ))
                        })?;
                        (
                            c.input_sanitized.clone(),
                            c.raw_offsets.clone(),
                            c.mean.clone(),
                            c.inv_std.clone(),
                            c.stddev.clone(),
                            c.normalize.clone(),
                            c.offsets.clone(),
                            c.masks.clone(),
                            c.mask_logits.clone(),
                        )
                    };
                    let plane = layer.output_shape.width * layer.output_shape.height;
                    let kernels = spec.size * spec.size;
                    let max_offset = spec.size as f32 * spec.dilation as f32 * 2.0;
                    // Offset/mask tensors are `[B, 2K, OH, OW]` and `[B, K, OH, OW]`;
                    // they are NOT the layer output shape.
                    let offsets_shape = tensor_backend_shape(TensorShape::new(
                        layer.output_shape.width,
                        layer.output_shape.height,
                        2 * kernels,
                        layer.input_shape.batch,
                    ));
                    let masks_shape = tensor_backend_shape(TensorShape::new(
                        layer.output_shape.width,
                        layer.output_shape.height,
                        kernels,
                        layer.input_shape.batch,
                    ));
                    let geometry = lightnet_cuda::host_ops::deform_conv::DeformGeometry {
                        batch: layer.input_shape.batch,
                        channels: layer.input_shape.channels,
                        filters: spec.filters,
                        channels_per_group: layer.input_shape.channels / spec.groups.max(1),
                        filters_per_group: spec.filters / spec.groups.max(1),
                        input_height: layer.input_shape.height,
                        input_width: layer.input_shape.width,
                        output_height: layer.output_shape.height,
                        output_width: layer.output_shape.width,
                        size: spec.size,
                        stride_x: spec.stride_x,
                        stride_y: spec.stride_y,
                        padding: spec.padding,
                        dilation: spec.dilation,
                    };
                    let activation_input = forward.gpu_layer_activation_inputs[layer.index]
                        .as_ref()
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing activation cache for [deform_conv] layer #{}",
                                layer.index
                            ))
                        })?;
                    let kind = gpu_activation_kind(layer.index, &spec.activation)?;
                    let delta = self
                        .backend
                        .activation_backward_f32(activation_input, &grad, kind, config)
                        .map_err(|err| backend_error("deform activation backward", err))?;
                    let d_main_bias = self
                        .backend
                        .bias_backward_f32(&delta, spec.filters, plane, config)
                        .map_err(|err| backend_error("deform bias backward", err))?;
                    let empty = self
                        .backend
                        .zeros_f32(&[1])
                        .map_err(|err| backend_error("deform empty masks", err))?;
                    let main_w_device = self
                        .backend
                        .upload_f32(&[main_weights.len()], &main_weights)
                        .map_err(|err| backend_error("upload deform main weights", err))?;
                    let (d_input_a, d_offsets, d_masks, d_main_w) = self
                        .backend
                        .deform_conv_accumulate_backward_f32(
                            &input_sanitized,
                            &offsets,
                            masks.as_ref().unwrap_or(&empty),
                            &main_w_device,
                            &delta,
                            geometry,
                            masks.is_some(),
                            config,
                        )
                        .map_err(|err| backend_error("deform accumulate backward", err))?;
                    let d_pre = self
                        .backend
                        .deform_envelope_pre_backward_f32(
                            &raw_offsets,
                            &mean,
                            &inv_std,
                            &normalize,
                            &d_offsets,
                            2 * kernels,
                            plane,
                            max_offset,
                            &offsets_shape,
                            config,
                        )
                        .map_err(|err| backend_error("deform envelope backward", err))?;
                    let d_stage1 = {
                        let d_pre_host = self
                            .backend
                            .download_f32(&d_pre)
                            .map_err(|err| backend_error("download deform d_pre", err))?;
                        let mean_host = self
                            .backend
                            .download_f32(&mean)
                            .map_err(|err| backend_error("download deform mean", err))?;
                        let std_host = self
                            .backend
                            .download_f32(&stddev)
                            .map_err(|err| backend_error("download deform stddev", err))?;
                        let normalize_host = self
                            .backend
                            .download_f32(&normalize)
                            .map_err(|err| backend_error("download deform normalize", err))?;
                        let raw_host = self
                            .backend
                            .download_f32(&raw_offsets)
                            .map_err(|err| backend_error("download deform raw offsets", err))?;
                        let stage1: Vec<f32> = raw_host
                            .iter()
                            .copied()
                            .map(lightnet_cuda::host_ops::deform_conv::sanitize)
                            .collect();
                        let grads =
                            lightnet_cuda::host_ops::deform_conv::offset_normalization_backward(
                                &stage1,
                                &mean_host,
                                &std_host,
                                &normalize_host,
                                &d_pre_host,
                                2 * kernels * plane,
                            )
                            .map_err(Error::InvalidConfig)?;
                        let scaled: Vec<f32> = grads
                            .iter()
                            .zip(&raw_host)
                            .map(|(&d, &raw)| {
                                let slope = if raw.is_finite() && raw.abs() < 10_000.0 {
                                    1.0
                                } else {
                                    0.0
                                };
                                d * slope
                            })
                            .collect();
                        self.backend
                            .upload_f32(&offsets_shape, &scaled)
                            .map_err(|err| backend_error("upload deform d_stage1", err))?
                    };
                    let off_w_device = self
                        .backend
                        .upload_f32(&[offset_weights.len()], &offset_weights)
                        .map_err(|err| backend_error("upload deform offset weights", err))?;
                    let (d_input_o, d_off_w, d_off_b) =
                        self.backend
                            .deform_predictor_backward_f32(
                                &input_sanitized,
                                &off_w_device,
                                &d_stage1,
                                layer.input_shape.batch,
                                layer.input_shape.channels,
                                2 * kernels,
                                layer.input_shape.height,
                                layer.input_shape.width,
                                layer.output_shape.height,
                                layer.output_shape.width,
                                spec.size,
                                spec.stride_x,
                                spec.stride_y,
                                spec.padding,
                                spec.dilation,
                                config,
                            )
                            .map_err(|err| {
                                backend_error("deform offset predictor backward", err)
                            })?;
                    let (d_input_m, d_mask_w, d_mask_b) = match (
                        mask_logits.as_ref(),
                        mask_weights.as_ref(),
                        mask_biases.as_ref(),
                    ) {
                        (Some(logits), Some(weights), Some(_)) => {
                            let d_logits = self
                                .backend
                                .deform_mask_sigmoid_backward_f32(
                                    logits,
                                    &d_masks,
                                    &masks_shape,
                                    config,
                                )
                                .map_err(|err| {
                                    backend_error("deform mask sigmoid backward", err)
                                })?;
                            let w_device = self
                                .backend
                                .upload_f32(&[weights.len()], weights)
                                .map_err(|err| {
                                    backend_error("upload deform mask weights", err)
                                })?;
                            let (d_in, d_w, d_b) = self
                                .backend
                                .deform_predictor_backward_f32(
                                    &input_sanitized,
                                    &w_device,
                                    &d_logits,
                                    layer.input_shape.batch,
                                    layer.input_shape.channels,
                                    kernels,
                                    layer.input_shape.height,
                                    layer.input_shape.width,
                                    layer.output_shape.height,
                                    layer.output_shape.width,
                                    spec.size,
                                    spec.stride_x,
                                    spec.stride_y,
                                    spec.padding,
                                    spec.dilation,
                                    config,
                                )
                                .map_err(|err| {
                                    backend_error("deform mask predictor backward", err)
                                })?;
                            (Some(d_in), Some(d_w), Some(d_b))
                        }
                        _ => (None, None, None),
                    };
                    let mut sum = self
                        .backend
                        .vector_add_f32(&d_input_a, &d_input_o, config)
                        .map_err(|err| backend_error("deform input grad sum", err))?;
                    if let Some(d_in) = d_input_m.as_ref() {
                        sum = self
                            .backend
                            .vector_add_f32(&sum, d_in, config)
                            .map_err(|err| backend_error("deform input grad sum", err))?;
                    }
                    // The raw (unsanitized) layer input drives the final slope.
                    let raw_input = if layer.index == 0 {
                        forward.input_gpu.clone()
                    } else {
                        forward.gpu_layer_outputs[layer.index - 1].clone()
                    };
                    let d_input = self
                        .backend
                        .deform_scale_input_gradient_f32(
                            &raw_input,
                            &sum,
                            &tensor_backend_shape(layer.input_shape),
                            config,
                        )
                        .map_err(|err| backend_error("deform input slope", err))?;
                    {
                        let d_main = self
                            .backend
                            .download_f32(&d_main_w)
                            .map_err(|err| backend_error("download deform main w grad", err))?;
                        let d_bias = self
                            .backend
                            .download_f32(&d_main_bias)
                            .map_err(|err| backend_error("download deform main b grad", err))?;
                        let d_ow = self
                            .backend
                            .download_f32(&d_off_w)
                            .map_err(|err| backend_error("download deform off w grad", err))?;
                        let d_ob = self
                            .backend
                            .download_f32(&d_off_b)
                            .map_err(|err| backend_error("download deform off b grad", err))?;
                        let d_mw = match d_mask_w.as_ref() {
                            Some(t) => Some(
                                self.backend
                                    .download_f32(t)
                                    .map_err(|err| backend_error("download deform mask w", err))?,
                            ),
                            None => None,
                        };
                        let d_mb = match d_mask_b.as_ref() {
                            Some(t) => Some(
                                self.backend
                                    .download_f32(t)
                                    .map_err(|err| backend_error("download deform mask b", err))?,
                            ),
                            None => None,
                        };
                        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
                        let LayerTrainBuffers::DeformConv(train) = &mut buffers[layer.index] else {
                            return Err(Error::InvalidConfig(format!(
                                "[deform_conv] layer #{} has no train buffers",
                                layer.index
                            )));
                        };
                        for (dst, src) in train.main.weight_updates.iter_mut().zip(d_main) {
                            *dst += src;
                        }
                        for (dst, src) in train.main.bias_updates.iter_mut().zip(d_bias) {
                            *dst += src;
                        }
                        for (dst, src) in train.offset_weight_updates.iter_mut().zip(d_ow) {
                            *dst += src;
                        }
                        for (dst, src) in train.offset_bias_updates.iter_mut().zip(d_ob) {
                            *dst += src;
                        }
                        if let (Some(dw), Some(db)) = (d_mw, d_mb) {
                            if let (Some(dst_w), Some(dst_b)) = (
                                train.mask_weight_updates.as_mut(),
                                train.mask_bias_updates.as_mut(),
                            ) {
                                for (dst, src) in dst_w.iter_mut().zip(dw) {
                                    *dst += src;
                                }
                                for (dst, src) in dst_b.iter_mut().zip(db) {
                                    *dst += src;
                                }
                            }
                        }
                    }
                    self.accumulate_previous_gpu_gradient(
                        layer.index,
                        d_input,
                        &mut layer_output_gradients,
                    )?;
                }
                _ => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{}_gpu_backward", layer.kind.name()),
                    });
                }
            }
        }

        self.update_gpu_batch_norm_running_stats_from_forward(forward)?;
        self.update_dcnv4_rolling_stats_gpu(forward)?;
        if self.network.layers.iter().any(|layer| {
            matches!(&layer.kind, LayerKind::Rnn(spec) if spec.batch_normalize)
                || matches!(&layer.kind, LayerKind::Lstm(spec) if spec.batch_normalize)
                || matches!(&layer.kind, LayerKind::Crnn(spec) if spec.batch_normalize)
        }) {
            let input = self
                .download_tensor(&forward.input_gpu, self.network.net.input_shape())
                .map_err(|err| {
                    backend_error("download BN recurrent input for rolling stats", err)
                })?;
            update_recurrent_batch_norm_running_stats(
                &self.network,
                &mut self.params,
                &input,
                &forward.cpu_forward,
            )?;
        }
        Ok(())
    }

}

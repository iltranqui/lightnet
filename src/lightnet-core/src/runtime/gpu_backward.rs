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
                    self.backward_convolutional_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
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
                    self.backward_graph_conv_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
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
                    self.backward_weighted_shortcut_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
                }
                (LayerKind::Shortcut(spec), Some(LayerParams::None)) => {
                    self.backward_shortcut_gpu(
                        layer,
                        spec,
                        forward,
                        grad,
                        config,
                        &mut layer_output_gradients,
                    )?;
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
                    self.backward_moe_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
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
                    self.backward_deform_conv_gpu(
                        layer,
                        spec,
                        params,
                        forward,
                        grad,
                        config,
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

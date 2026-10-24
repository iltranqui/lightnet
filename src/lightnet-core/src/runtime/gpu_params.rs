use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(super) fn batch_norm_scale_gradient_gpu(
        &self,
        raw_output: &B::Tensor,
        delta: &B::Tensor,
        batch_norm: &BatchNormParams,
        output_shape: TensorShape,
    ) -> Result<B::Tensor> {
        let raw_output = self
            .backend
            .download_f32(raw_output)
            .map_err(|err| backend_error("download GPU batch-norm raw output", err))?;
        let delta = self
            .backend
            .download_f32(delta)
            .map_err(|err| backend_error("download GPU batch-norm delta", err))?;
        if raw_output.len() != delta.len() {
            return Err(Error::InvalidConfig(format!(
                "GPU batch-norm scale gradient shape mismatch: raw output has {} values, delta has {}",
                raw_output.len(),
                delta.len()
            )));
        }
        if batch_norm.rolling_mean.len() != output_shape.channels
            || batch_norm.rolling_variance.len() != output_shape.channels
        {
            return Err(Error::InvalidConfig(format!(
                "GPU batch-norm rolling stat mismatch: expected {} channels",
                output_shape.channels
            )));
        }

        let channel_stride = output_shape.width * output_shape.height;
        let mut scale_grad = vec![0.0; output_shape.channels];
        for b in 0..output_shape.batch {
            for (channel, grad) in scale_grad.iter_mut().enumerate() {
                let inv_std =
                    1.0 / (batch_norm.rolling_variance[channel] + BATCH_NORM_EPSILON).sqrt();
                let base = (b * output_shape.channels + channel) * channel_stride;
                for idx in 0..channel_stride {
                    let offset = base + idx;
                    let normalized =
                        (raw_output[offset] - batch_norm.rolling_mean[channel]) * inv_std;
                    *grad += delta[offset] * normalized;
                }
            }
        }

        let scale_grad = self
            .backend
            .upload_f32(&[output_shape.channels], &scale_grad)
            .map_err(|err| backend_error("upload GPU batch-norm scale gradient", err))?;
        Ok(scale_grad)
    }

    pub(super) fn update_gpu_convolution_batch_norm_running_stats(
        &mut self,
        owner: usize,
        mean: &[f32],
        variance: &[f32],
    ) -> Result<()> {
        let Some(LayerParams::Convolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU convolution params for layer #{owner}"
            )));
        };
        let Some(batch_norm) = params.batch_norm.as_mut() else {
            return Err(Error::InvalidConfig(format!(
                "missing batch-norm params for convolutional layer #{owner}"
            )));
        };
        update_batch_norm_running_stats(batch_norm, mean, variance, BATCH_NORM_ROLLING_KEEP)?;
        self.refresh_gpu_batch_norm_cache(owner)
    }

    /// Stores the device-computed rolling statistics of the batch-statistics
    /// forward.  The host copy is refreshed now unless host syncs are deferred.
    pub(super) fn set_device_batch_norm_rolling_stats(
        &mut self,
        owner: usize,
        mean: B::Tensor,
        variance: B::Tensor,
    ) -> Result<()> {
        let mut cached = self.device_conv_params_at(owner, 0).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing cached GPU convolution params for layer #{owner}"
            ))
        })?;
        let Some(bn) = cached.bn.as_mut() else {
            return Err(Error::InvalidConfig(format!(
                "missing device batch-norm state for layer #{owner}"
            )));
        };
        bn.rolling_mean = mean;
        bn.rolling_variance = variance;
        cached.folded_stale = true;
        self.set_device_conv_params(owner, 0, cached.clone())?;
        if self.defer_host_sync {
            self.host_weights_stale = true;
        } else {
            self.sync_convolution_params_from_device(owner, &cached)?;
        }
        Ok(())
    }

    /// Marks the folded device params stale after the host BN stats changed and
    /// re-uploads the host BN state to the device.
    pub(super) fn refresh_gpu_batch_norm_cache(&mut self, owner: usize) -> Result<()> {
        let cached = self.device_conv_params_at(owner, 0).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing cached GPU convolution params for layer #{owner}"
            ))
        })?;
        let mut cached = cached;
        if let Some(LayerParams::Convolutional(ConvolutionalParams {
            batch_norm: Some(batch_norm),
            ..
        })) = self.params.get(owner)
        {
            cached.bn = Some(self.upload_batch_norm_state(batch_norm)?);
        }
        cached.folded_stale = true;
        self.set_device_conv_params(owner, 0, cached)?;
        Ok(())
    }

    pub(super) fn set_device_eml_batch_norm_rolling_stats(
        &mut self,
        owner: usize,
        slot: usize,
        mean: B::Tensor,
        variance: B::Tensor,
    ) -> Result<()> {
        let host_mean = self
            .backend
            .download_f32(&mean)
            .map_err(|err| backend_error("download EML rolling mean", err))?;
        let host_variance = self
            .backend
            .download_f32(&variance)
            .map_err(|err| backend_error("download EML rolling variance", err))?;
        let mut cached = self.device_conv_params_at(owner, slot).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing cached EML convolution params for layer #{owner}, slot {slot}"
            ))
        })?;
        let Some(bn) = cached.bn.as_mut() else {
            return Err(Error::InvalidConfig(format!(
                "missing device EML batch-norm state for layer #{owner}, slot {slot}"
            )));
        };
        bn.rolling_mean = mean;
        bn.rolling_variance = variance;
        cached.folded_stale = true;
        self.set_device_conv_params(owner, slot, cached)?;

        let Some(LayerParams::EmlConvolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU EML params for layer #{owner}"
            )));
        };
        let branch = match slot {
            0 => &mut params.input,
            1 => &mut params.self_params,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "invalid EML BN slot {slot} at layer #{owner}"
                )));
            }
        };
        let Some(batch_norm) = branch.batch_norm.as_mut() else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU EML batch-norm params at layer #{owner}, slot {slot}"
            )));
        };
        batch_norm.rolling_mean = host_mean;
        batch_norm.rolling_variance = host_variance;
        Ok(())
    }

    pub(super) fn update_gpu_batch_norm_running_stats_from_forward(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
    ) -> Result<()> {
        let config = ElementwiseLaunchConfig::default();
        let mut batch_norm_stats = Vec::new();
        let mut batch_norm_rolling = Vec::new();
        let mut graph_conv_batch_norm_rolling = Vec::new();
        let mut standalone_batch_norm_rolling = Vec::new();
        let mut eml_batch_norm_rolling = Vec::new();
        for layer in &self.network.layers {
            let LayerKind::Convolutional(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize || spec.share_index.is_some() {
                continue;
            }
            if let Some(cache) = forward.gpu_batch_norm_caches[layer.index].as_ref() {
                // Batch-statistics forward: the device already did the EMA.
                if let Some((mean, variance)) = cache.rolling.as_ref() {
                    batch_norm_rolling.push((layer.index, mean.clone(), variance.clone()));
                }
                continue;
            }

            let current_input =
                if layer.index == 0 {
                    &forward.input_gpu
                } else {
                    forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous GPU layer output for batch-norm stats at layer #{}",
                        layer.index
                    ))
                })?
                };
            let cached = self.device_conv_params_at(layer.index, 0).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing cached GPU convolution params for layer #{}",
                    layer.index
                ))
            })?;
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
                .map_err(|err| backend_error("GPU batch-norm raw convolution stats", err))?;
            let (mean, variance) =
                self.batch_norm_channel_stats_gpu(&raw_output, layer.output_shape)?;
            batch_norm_stats.push((layer.index, mean, variance));
        }
        for layer in &self.network.layers {
            let LayerKind::GraphConv(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }
            if let Some(cache) = forward.gpu_batch_norm_caches[layer.index].as_ref()
                && let Some((mean, _)) = cache.rolling.as_ref()
            {
                graph_conv_batch_norm_rolling.push((
                    layer.index,
                    mean.clone(),
                    cache.variance.clone(),
                ));
            }
        }
        for layer in &self.network.layers {
            if !matches!(layer.kind, LayerKind::BatchNorm(_)) {
                continue;
            }
            if let Some(cache) = forward.gpu_batch_norm_caches[layer.index].as_ref()
                && let Some((mean, _)) = cache.rolling.as_ref()
            {
                standalone_batch_norm_rolling.push((
                    layer.index,
                    mean.clone(),
                    cache.variance.clone(),
                ));
            }
        }
        for layer in &self.network.layers {
            let LayerKind::EmlConvolutional(spec) = &layer.kind else {
                continue;
            };
            if !spec.branch.batch_normalize {
                continue;
            }
            let Some(cache) = forward
                .gpu_eml_caches
                .get(layer.index)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            for (slot, branch_cache) in [
                cache.input_batch_norm.as_ref(),
                cache.self_batch_norm.as_ref(),
            ]
            .into_iter()
            .enumerate()
            {
                if let Some((mean, variance)) =
                    branch_cache.and_then(|cache| cache.rolling.as_ref())
                {
                    eml_batch_norm_rolling.push((
                        layer.index,
                        slot,
                        mean.clone(),
                        variance.clone(),
                    ));
                }
            }
        }

        for (owner, mean, variance) in batch_norm_stats {
            self.update_gpu_convolution_batch_norm_running_stats(owner, &mean, &variance)?;
        }
        for (owner, mean, variance) in batch_norm_rolling {
            self.set_device_batch_norm_rolling_stats(owner, mean, variance)?;
        }
        for (layer_index, mean, batch_variance) in standalone_batch_norm_rolling {
            let mean = self
                .backend
                .download_f32(&mean)
                .map_err(|err| backend_error("download standalone BN rolling mean", err))?;
            let batch_variance = self
                .backend
                .download_f32(&batch_variance)
                .map_err(|err| backend_error("download standalone BN batch variance", err))?;
            let Some(LayerParams::BatchNorm(params)) = self.params.get_mut(layer_index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing standalone BN params for rolling stats at layer #{layer_index}"
                )));
            };
            params.rolling_mean = mean;
            let update = 1.0 - BATCH_NORM_ROLLING_KEEP_CPU;
            for (dst, src) in params.rolling_variance.iter_mut().zip(batch_variance) {
                *dst = BATCH_NORM_ROLLING_KEEP_CPU * *dst + update * src;
            }
        }
        for (layer_index, mean, batch_variance) in graph_conv_batch_norm_rolling {
            let mean = self
                .backend
                .download_f32(&mean)
                .map_err(|err| backend_error("download GraphConv BN rolling mean", err))?;
            let batch_variance = self
                .backend
                .download_f32(&batch_variance)
                .map_err(|err| backend_error("download GraphConv BN batch variance", err))?;
            let Some(LayerParams::GraphConv(params)) = self.params.get_mut(layer_index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing GraphConv params for BN rolling stats at layer #{layer_index}"
                )));
            };
            let Some(batch_norm) = params.projection.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing GraphConv BN params for rolling stats at layer #{layer_index}"
                )));
            };
            batch_norm.rolling_mean = mean;
            let update = 1.0 - BATCH_NORM_ROLLING_KEEP_CPU;
            for (dst, src) in batch_norm.rolling_variance.iter_mut().zip(batch_variance) {
                *dst = BATCH_NORM_ROLLING_KEEP_CPU * *dst + update * src;
            }
        }
        for layer in &self.network.layers {
            let LayerKind::Connected(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }
            let Some(LayerParams::Connected(params)) = self.params.get(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing connected params for BN statistics at layer #{}",
                    layer.index
                )));
            };
            let current_input =
                if layer.index == 0 {
                    &forward.input_gpu
                } else {
                    forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous GPU output for connected BN statistics at layer #{}",
                        layer.index
                    ))
                })?
                };
            let inputs = layer.input_shape.element_count();
            let weights = self
                .backend
                .upload_f32(&[spec.output, inputs], &params.weights)
                .map_err(|err| backend_error("upload connected BN statistic weights", err))?;
            let zero_bias = self
                .backend
                .zeros_f32(&[spec.output])
                .map_err(|err| backend_error("allocate connected BN statistic bias", err))?;
            let raw_output = self
                .backend
                .connected_f32(
                    current_input,
                    &weights,
                    &zero_bias,
                    layer.input_shape.batch,
                    inputs,
                    spec.output,
                    config,
                )
                .map_err(|err| backend_error("connected BN raw output for rolling stats", err))?;
            let (mean, variance) =
                self.batch_norm_channel_stats_gpu(&raw_output, layer.output_shape)?;
            let Some(LayerParams::Connected(params)) = self.params.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing mutable connected params for BN statistics at layer #{}",
                    layer.index
                )));
            };
            let Some(batch_norm) = params.batch_norm.as_mut() else {
                return Err(Error::InvalidConfig(format!(
                    "missing connected batch-norm params for layer #{}",
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

        for (owner, slot, mean, variance) in eml_batch_norm_rolling {
            self.set_device_eml_batch_norm_rolling_stats(owner, slot, mean, variance)?;
        }
        Ok(())
    }

    pub(super) fn batch_norm_channel_stats_gpu(
        &self,
        raw_output: &B::Tensor,
        output_shape: TensorShape,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let raw_output = self
            .backend
            .download_f32(raw_output)
            .map_err(|err| backend_error("download GPU batch-norm raw output stats", err))?;
        Ok(channel_mean_variance(&Tensor {
            shape: output_shape,
            data: raw_output,
        }))
    }

    pub(super) fn device_conv_params_at(
        &self,
        owner: usize,
        slot: usize,
    ) -> Option<GpuConvolutionLayerParams<B::Tensor>> {
        self.device_conv_params
            .borrow()
            .get(owner)
            .and_then(|slots| slots.get(slot))
            .and_then(Clone::clone)
    }

    pub(super) fn set_device_conv_params(
        &self,
        owner: usize,
        slot: usize,
        params: GpuConvolutionLayerParams<B::Tensor>,
    ) -> Result<()> {
        let mut cache = self.device_conv_params.borrow_mut();
        let slots = cache.get_mut(owner).ok_or_else(|| {
            Error::InvalidConfig(format!("missing device convolution cache owner #{owner}"))
        })?;
        if slots.len() <= slot {
            slots.resize_with(slot + 1, || None);
        }
        slots[slot] = Some(params);
        Ok(())
    }

    pub(super) fn clear_device_conv_params(&mut self) {
        // Callers replace the host params right before; nothing is stale then.
        self.host_weights_stale = false;
        for slots in self.device_conv_params.get_mut().iter_mut() {
            slots.clear();
        }
        for entry in self.f16_conv_weights.get_mut().iter_mut() {
            *entry = None;
        }
    }

    pub(super) fn clear_device_train_buffers(&mut self) {
        for entry in self.device_train_buffers.get_mut().iter_mut() {
            *entry = None;
        }
        for entry in self.device_moe_train_buffers.get_mut().iter_mut() {
            *entry = None;
        }
        for entry in self.device_eml_train_buffers.get_mut().iter_mut() {
            *entry = None;
        }
        for entry in self.device_dcnv4_train_buffers.get_mut().iter_mut() {
            *entry = None;
        }
    }

    pub(super) fn warm_device_conv_params(&self) -> Result<()> {
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    let _ = self.cached_convolution_params(
                        owner,
                        layer.input_shape.channels,
                        spec,
                        conv,
                    )?;
                }
                (LayerKind::Moe(spec), LayerParams::Moe(moe)) => {
                    for (slot, expert) in moe.experts.iter().enumerate() {
                        let _ = self.cached_convolution_params_at(
                            layer.index,
                            slot,
                            layer.input_shape.channels,
                            &spec.expert,
                            expert,
                        )?;
                    }
                }
                (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(eml)) => {
                    for (slot, params) in [&eml.input, &eml.self_params].into_iter().enumerate() {
                        let _ = self.cached_convolution_params_at(
                            layer.index,
                            slot,
                            layer.input_shape.channels,
                            &spec.branch,
                            params,
                        )?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Cached device params with the BN-folded copies up to date.
    pub(super) fn cached_convolution_params(
        &self,
        layer_index: usize,
        input_channels: usize,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        self.cached_convolution_params_at(layer_index, 0, input_channels, spec, params)
    }

    pub(crate) fn cached_convolution_params_at(
        &self,
        layer_index: usize,
        slot: usize,
        input_channels: usize,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        let cached =
            self.cached_convolution_params_raw_at(layer_index, slot, input_channels, spec, params)?;
        if !cached.folded_stale {
            return Ok(cached);
        }
        let refolded = self.cached_convolution_upload(
            spec,
            input_channels,
            params,
            Some((cached.raw_weights, cached.raw_biases, cached.bn)),
        )?;
        self.set_device_conv_params(layer_index, slot, refolded.clone())?;
        Ok(refolded)
    }

    /// Cached device params whose raw tensors are current but whose folded
    /// copies may be stale (training BN forward/backward only read the raw ones).
    pub(super) fn cached_convolution_params_raw(
        &self,
        layer_index: usize,
        input_channels: usize,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        self.cached_convolution_params_raw_at(layer_index, 0, input_channels, spec, params)
    }

    pub(crate) fn cached_convolution_params_raw_at(
        &self,
        layer_index: usize,
        slot: usize,
        input_channels: usize,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        if let Some(cached) = self.device_conv_params_at(layer_index, slot) {
            return Ok(cached);
        }

        let (folded_biases, folded_weights) =
            fold_convolutional_params_for_inference(spec, params, input_channels)?;
        let raw_weights = self
            .backend
            .upload_f32(
                &[
                    spec.filters,
                    input_channels / spec.groups.max(1),
                    spec.size,
                    spec.size,
                ],
                &params.weights,
            )
            .map_err(|err| backend_error("upload raw convolution weights", err))?;
        let raw_biases = self
            .backend
            .upload_f32(&[spec.filters], &params.biases)
            .map_err(|err| backend_error("upload raw convolution biases", err))?;
        let uploaded = GpuConvolutionLayerParams {
            raw_weights: raw_weights.clone(),
            raw_biases: raw_biases.clone(),
            weights: if spec.batch_normalize {
                self.backend
                    .upload_f32(
                        &[
                            spec.filters,
                            input_channels / spec.groups.max(1),
                            spec.size,
                            spec.size,
                        ],
                        &folded_weights,
                    )
                    .map_err(|err| backend_error("upload folded convolution weights", err))?
            } else {
                raw_weights
            },
            biases: if spec.batch_normalize {
                self.backend
                    .upload_f32(&[spec.filters], &folded_biases)
                    .map_err(|err| backend_error("upload folded convolution biases", err))?
            } else {
                raw_biases
            },
            folded_stale: false,
            batch_norm_scales: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| {
                    let scales = batch_norm
                        .scales
                        .iter()
                        .zip(&batch_norm.rolling_variance)
                        .map(|(scale, variance)| scale / (variance + BATCH_NORM_EPSILON).sqrt())
                        .collect::<Vec<_>>();
                    self.backend
                        .upload_f32(&[spec.filters], &scales)
                        .map_err(|err| backend_error("upload batch-norm scales", err))
                })
                .transpose()?,
            bn: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| self.upload_batch_norm_state(batch_norm))
                .transpose()?,
        };

        self.set_device_conv_params(layer_index, slot, uploaded.clone())?;
        Ok(self
            .device_conv_params_at(layer_index, slot)
            .expect("device params were just initialized"))
    }

    /// Stores the updated raw tensors; the BN-folded copies are re-folded lazily.
    /// Nothing is copied to the host (see `sync_convolution_params_from_device`).
    pub(super) fn refresh_device_convolution_params(
        &self,
        owner: usize,
        raw_weights: B::Tensor,
        raw_biases: B::Tensor,
        scales: Option<B::Tensor>,
        weight_updates: B::Tensor,
        bias_updates: B::Tensor,
        scale_updates: Option<B::Tensor>,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        let layer =
            self.network.layers.get(owner).ok_or_else(|| {
                Error::InvalidConfig(format!("missing convolutional layer #{owner}"))
            })?;
        let LayerKind::Convolutional(spec) = &layer.kind else {
            return Err(Error::InvalidConfig(format!(
                "layer #{owner} is not convolutional"
            )));
        };
        let params = resolve_convolutional_params(&self.network, &self.params, owner)?;
        let folds = spec.batch_normalize && params.batch_norm.is_some();
        let mut refreshed = self.device_conv_params_at(owner, 0).ok_or_else(|| {
            Error::InvalidConfig(format!("missing cached device params for layer #{owner}"))
        })?;
        refreshed.raw_weights = raw_weights.clone();
        refreshed.raw_biases = raw_biases.clone();
        if let (Some(scales), Some(bn)) = (scales, refreshed.bn.as_mut()) {
            bn.scales = scales;
        }
        if folds {
            refreshed.folded_stale = true;
        } else {
            refreshed.weights = raw_weights;
            refreshed.biases = raw_biases;
        }
        self.set_device_conv_params(owner, 0, refreshed.clone())?;
        self.device_train_buffers.borrow_mut()[owner] = Some(GpuConvolutionTrainBuffers {
            weight_updates,
            bias_updates,
            scale_updates,
            pending: None,
        });
        Ok(refreshed)
    }

    pub(super) fn upload_batch_norm_state(
        &self,
        batch_norm: &BatchNormParams,
    ) -> Result<GpuBatchNormState<B::Tensor>> {
        let upload = |name: &str, host: &[f32]| {
            self.backend
                .upload_f32(&[host.len()], host)
                .map_err(|err| backend_error(name, err))
        };
        Ok(GpuBatchNormState {
            scales: upload("upload batch-norm scales", &batch_norm.scales)?,
            rolling_mean: upload("upload batch-norm rolling mean", &batch_norm.rolling_mean)?,
            rolling_variance: upload(
                "upload batch-norm rolling variance",
                &batch_norm.rolling_variance,
            )?,
        })
    }

    pub(super) fn cached_convolution_upload(
        &self,
        spec: &ConvolutionalSpec,
        input_channels: usize,
        params: &ConvolutionalParams,
        raw: Option<(B::Tensor, B::Tensor, Option<GpuBatchNormState<B::Tensor>>)>,
    ) -> Result<GpuConvolutionLayerParams<B::Tensor>> {
        let weight_shape = [
            spec.filters,
            input_channels / spec.groups.max(1),
            spec.size,
            spec.size,
        ];
        // With device-resident raw weights and BN state the fold runs on the
        // device (`scale = gamma / sqrt(var + eps)`, `w * scale` per filter,
        // `bias - mean * scale`, the same arithmetic as
        // `fold_convolutional_params_for_inference`), so the host copies may be
        // stale.
        if let Some((raw_weights, raw_biases, bn)) = raw {
            let mut weights = raw_weights.clone();
            let mut biases = raw_biases.clone();
            let mut batch_norm_scales = None;
            if let Some(bn) = bn.as_ref() {
                let (scales, folded_biases) = self
                    .backend
                    .batchnorm_fold_f32(
                        &bn.scales,
                        &bn.rolling_mean,
                        &bn.rolling_variance,
                        &raw_biases,
                        BATCH_NORM_EPSILON,
                        ElementwiseLaunchConfig::default(),
                    )
                    .map_err(|err| backend_error("fold batch-norm parameters", err))?;
                if spec.batch_normalize {
                    weights = self
                        .backend
                        .channel_scale_f32(
                            &raw_weights,
                            &scales,
                            spec.filters,
                            weight_shape[1] * spec.size * spec.size,
                            ElementwiseLaunchConfig::default(),
                        )
                        .map_err(|err| backend_error("fold convolution weights", err))?;
                    biases = folded_biases;
                }
                batch_norm_scales = Some(scales);
            }
            return Ok(GpuConvolutionLayerParams {
                raw_weights,
                raw_biases,
                weights,
                biases,
                folded_stale: false,
                batch_norm_scales,
                bn,
            });
        }
        let (folded_biases, folded_weights) =
            fold_convolutional_params_for_inference(spec, params, input_channels)?;
        let raw_weights = self
            .backend
            .upload_f32(&weight_shape, &params.weights)
            .map_err(|err| backend_error("upload raw convolution weights", err))?;
        let raw_biases = self
            .backend
            .upload_f32(&[spec.filters], &params.biases)
            .map_err(|err| backend_error("upload raw convolution biases", err))?;
        let (weights, biases) = if spec.batch_normalize {
            (
                self.backend
                    .upload_f32(&weight_shape, &folded_weights)
                    .map_err(|err| backend_error("upload folded convolution weights", err))?,
                self.backend
                    .upload_f32(&[spec.filters], &folded_biases)
                    .map_err(|err| backend_error("upload folded convolution biases", err))?,
            )
        } else {
            (raw_weights.clone(), raw_biases.clone())
        };
        Ok(GpuConvolutionLayerParams {
            raw_weights,
            raw_biases,
            weights,
            biases,
            folded_stale: false,
            batch_norm_scales: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| {
                    let scales = batch_norm
                        .scales
                        .iter()
                        .zip(&batch_norm.rolling_variance)
                        .map(|(scale, variance)| scale / (variance + BATCH_NORM_EPSILON).sqrt())
                        .collect::<Vec<_>>();
                    self.backend
                        .upload_f32(&[spec.filters], &scales)
                        .map_err(|err| backend_error("upload batch-norm scales", err))
                })
                .transpose()?,
            bn: params
                .batch_norm
                .as_ref()
                .map(|batch_norm| self.upload_batch_norm_state(batch_norm))
                .transpose()?,
        })
    }

    /// Copies the device raw weights, biases and BN gamma / rolling stats of one
    /// owner convolution to the host params.
    pub(super) fn sync_convolution_params_from_device(
        &mut self,
        owner: usize,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
    ) -> Result<()> {
        let download = |name: &str, tensor: &B::Tensor| {
            self.backend
                .download_f32(tensor)
                .map_err(|err| backend_error(name, err))
        };
        let raw_weights = download(
            "download synced raw convolution weights",
            &cached.raw_weights,
        )?;
        let raw_biases = download("download synced raw convolution biases", &cached.raw_biases)?;
        let bn = match cached.bn.as_ref() {
            Some(bn) => Some((
                download("download synced batch-norm scales", &bn.scales)?,
                download("download synced batch-norm rolling mean", &bn.rolling_mean)?,
                download(
                    "download synced batch-norm rolling variance",
                    &bn.rolling_variance,
                )?,
            )),
            None => None,
        };
        let Some(LayerParams::Convolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU convolution params for layer #{owner}"
            )));
        };
        params.weights = raw_weights;
        params.biases = raw_biases;
        if let (Some((scales, mean, variance)), Some(batch_norm)) = (bn, params.batch_norm.as_mut())
        {
            batch_norm.scales = scales;
            batch_norm.rolling_mean = mean;
            batch_norm.rolling_variance = variance;
        }
        Ok(())
    }

    pub(super) fn sync_eml_branch_params_from_device(
        &mut self,
        owner: usize,
        branch_slot: usize,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
    ) -> Result<()> {
        let raw_weights = self
            .backend
            .download_f32(&cached.raw_weights)
            .map_err(|err| backend_error("download synced EML branch weights", err))?;
        let raw_biases = self
            .backend
            .download_f32(&cached.raw_biases)
            .map_err(|err| backend_error("download synced EML branch biases", err))?;
        let batch_norm = cached
            .bn
            .as_ref()
            .map(|bn| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
                Ok((
                    self.backend
                        .download_f32(&bn.scales)
                        .map_err(|err| backend_error("download synced EML BN scales", err))?,
                    self.backend
                        .download_f32(&bn.rolling_mean)
                        .map_err(|err| backend_error("download synced EML BN rolling mean", err))?,
                    self.backend
                        .download_f32(&bn.rolling_variance)
                        .map_err(|err| {
                            backend_error("download synced EML BN rolling variance", err)
                        })?,
                ))
            })
            .transpose()?;
        let Some(LayerParams::EmlConvolutional(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU EML params for layer #{owner}"
            )));
        };
        let branch = match branch_slot {
            0 => &mut params.input,
            1 => &mut params.self_params,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "invalid EML branch slot {branch_slot} at layer #{owner}"
                )));
            }
        };
        if branch.batch_norm.is_some() != batch_norm.is_some() {
            return Err(Error::InvalidConfig(format!(
                "CPU/device BN mismatch for EML branch at layer #{owner}, slot {branch_slot}"
            )));
        }
        branch.weights = raw_weights;
        branch.biases = raw_biases;
        if let (Some((scales, mean, variance)), Some(batch_norm)) =
            (batch_norm, branch.batch_norm.as_mut())
        {
            batch_norm.scales = scales;
            batch_norm.rolling_mean = mean;
            batch_norm.rolling_variance = variance;
        }
        Ok(())
    }

    pub(super) fn sync_moe_expert_params_from_device(
        &mut self,
        owner: usize,
        expert_index: usize,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
    ) -> Result<()> {
        let raw_weights = self
            .backend
            .download_f32(&cached.raw_weights)
            .map_err(|err| backend_error("download synced MoE expert weights", err))?;
        let raw_biases = self
            .backend
            .download_f32(&cached.raw_biases)
            .map_err(|err| backend_error("download synced MoE expert biases", err))?;
        let bn = cached
            .bn
            .as_ref()
            .map(|bn| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
                Ok((
                    self.backend
                        .download_f32(&bn.scales)
                        .map_err(|err| backend_error("download synced MoE BN scales", err))?,
                    self.backend
                        .download_f32(&bn.rolling_mean)
                        .map_err(|err| backend_error("download synced MoE BN rolling mean", err))?,
                    self.backend
                        .download_f32(&bn.rolling_variance)
                        .map_err(|err| {
                            backend_error("download synced MoE BN rolling variance", err)
                        })?,
                ))
            })
            .transpose()?;
        let Some(LayerParams::Moe(params)) = self.params.get_mut(owner) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU MoE params for layer #{owner}"
            )));
        };
        let Some(expert) = params.experts.get_mut(expert_index) else {
            return Err(Error::InvalidConfig(format!(
                "missing CPU MoE expert #{expert_index} for layer #{owner}"
            )));
        };
        if expert.batch_norm.is_some() != bn.is_some() {
            return Err(Error::InvalidConfig(format!(
                "CPU/device BN mismatch for MoE expert #{expert_index} at layer #{owner}"
            )));
        }
        expert.weights = raw_weights;
        expert.biases = raw_biases;
        if let (Some((scales, mean, variance)), Some(batch_norm)) = (bn, expert.batch_norm.as_mut())
        {
            batch_norm.scales = scales;
            batch_norm.rolling_mean = mean;
            batch_norm.rolling_variance = variance;
        }
        Ok(())
    }
}

use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(super) fn forward_eml_branch_gpu(
        &self,
        owner: usize,
        slot: usize,
        input: &B::Tensor,
        input_shape: TensorShape,
        output_shape: TensorShape,
        spec: &ConvolutionalSpec,
        params: &ConvolutionalParams,
        batch_norm_training: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<GpuBatchNormCache<B::Tensor>>)> {
        if spec.batch_normalize != params.batch_norm.is_some() {
            return Err(Error::InvalidConfig(format!(
                "EML branch BN parameter mismatch at layer #{owner}, slot {slot}"
            )));
        }
        let use_batch_stats = batch_norm_training && params.batch_norm.is_some();
        let cached = if use_batch_stats {
            self.cached_convolution_params_raw_at(owner, slot, input_shape.channels, spec, params)?
        } else {
            self.cached_convolution_params_at(owner, slot, input_shape.channels, spec, params)?
        };
        let weights = if use_batch_stats {
            &cached.raw_weights
        } else {
            &cached.weights
        };
        let raw_output = self
            .backend
            .conv2d_f32(
                input,
                weights,
                &tensor_backend_shape(output_shape),
                input_shape.channels,
                input_shape.height,
                input_shape.width,
                spec.filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.groups.max(1),
                config,
            )
            .map_err(|err| backend_error("EML branch convolution", err))?;
        if let Some(batch_norm) = params.batch_norm.as_ref()
            && batch_norm_training
        {
            let (output, cache) = self.forward_batchnorm_training_gpu(
                &raw_output,
                &cached,
                batch_norm,
                output_shape,
                true,
                config,
            )?;
            return Ok((output, Some(cache)));
        }
        let output = self
            .backend
            .bias_add_f32(
                &raw_output,
                &cached.biases,
                spec.filters,
                output_shape.width * output_shape.height,
                config,
            )
            .map_err(|err| backend_error("EML branch bias", err))?;
        Ok((output, None))
    }

    /// Batch-statistics batch norm of a raw (bias-free) convolution output.
    /// The shift is the layer's raw biases (the BN biases); the rolling EMA is
    /// computed on the device and applied to the host params after backward.
    pub(super) fn forward_batchnorm_training_gpu(
        &self,
        raw_output: &B::Tensor,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
        batch_norm: &BatchNormParams,
        shape: TensorShape,
        update_rolling: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, GpuBatchNormCache<B::Tensor>)> {
        validate_batchnorm_params(batch_norm, shape.channels)?;
        let state = cached.bn.as_ref().ok_or_else(|| {
            Error::InvalidConfig("missing device batch-norm state for a BN layer".to_owned())
        })?;
        let (scales, rolling_mean, rolling_variance) = (
            state.scales.clone(),
            state.rolling_mean.clone(),
            state.rolling_variance.clone(),
        );
        let result = self
            .backend
            .batchnorm_train_forward_f32(
                raw_output,
                &scales,
                &cached.raw_biases,
                &rolling_mean,
                &rolling_variance,
                &tensor_backend_shape(shape),
                BATCH_NORM_ROLLING_KEEP,
                config,
            )
            .map_err(|err| backend_error("GPU batch-norm training forward", err))?;
        let cache = GpuBatchNormCache {
            scales,
            x_hat: result.x_hat,
            variance: result.variance,
            rolling: update_rolling.then_some((result.rolling_mean, result.rolling_variance)),
        };
        Ok((result.output, cache))
    }

    pub(super) fn forward_batchnorm_training_gpu_from_host_params(
        &self,
        raw_output: &B::Tensor,
        raw_biases: &[f32],
        batch_norm: &BatchNormParams,
        shape: TensorShape,
        config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, GpuBatchNormCache<B::Tensor>)> {
        validate_batchnorm_params(batch_norm, shape.channels)?;
        let scales = self
            .backend
            .upload_f32(&[shape.channels], &batch_norm.scales)
            .map_err(|err| backend_error("upload GraphConv BN scales", err))?;
        let biases = self
            .backend
            .upload_f32(&[shape.channels], raw_biases)
            .map_err(|err| backend_error("upload GraphConv BN biases", err))?;
        let rolling_mean = self
            .backend
            .upload_f32(&[shape.channels], &batch_norm.rolling_mean)
            .map_err(|err| backend_error("upload GraphConv BN rolling mean", err))?;
        let rolling_variance = self
            .backend
            .upload_f32(&[shape.channels], &batch_norm.rolling_variance)
            .map_err(|err| backend_error("upload GraphConv BN rolling variance", err))?;
        // Match GraphConv's CPU reference rolling-stat update coefficient.
        let result = self
            .backend
            .batchnorm_train_forward_f32(
                raw_output,
                &scales,
                &biases,
                &rolling_mean,
                &rolling_variance,
                &tensor_backend_shape(shape),
                BATCH_NORM_ROLLING_KEEP_CPU,
                config,
            )
            .map_err(|err| backend_error("GraphConv GPU batch-norm training forward", err))?;
        Ok((
            result.output,
            GpuBatchNormCache {
                scales,
                x_hat: result.x_hat,
                variance: result.variance,
                rolling: Some((result.rolling_mean, result.rolling_variance)),
            },
        ))
    }

    pub(super) fn forward_batchnorm_inference_gpu(
        &self,
        input: &B::Tensor,
        shape: TensorShape,
        params: &BatchNormParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let (scales, biases) = folded_batchnorm_inference_params(params, shape.channels)?;
        let scales = self
            .backend
            .upload_f32(&[shape.channels], &scales)
            .map_err(|err| backend_error("upload batchnorm inference scales", err))?;
        let biases = self
            .backend
            .upload_f32(&[shape.channels], &biases)
            .map_err(|err| backend_error("upload batchnorm inference biases", err))?;
        let scaled = self
            .backend
            .channel_scale_f32(
                input,
                &scales,
                shape.channels,
                shape.width * shape.height,
                config,
            )
            .map_err(|err| backend_error("batchnorm inference scale", err))?;
        self.backend
            .bias_add_f32(
                &scaled,
                &biases,
                shape.channels,
                shape.width * shape.height,
                config,
            )
            .map_err(|err| backend_error("batchnorm inference bias", err))
    }

    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
        Ok(self.forward_with_state(input)?.cpu_forward)
    }

    pub fn forward_with_state(&self, input: &Tensor) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_mode(input, false)
    }

    pub fn forward_training_with_state(
        &self,
        input: &Tensor,
    ) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_mode(input, true)
    }

    /// Training forward for the GPU backward path: only the yolo head outputs
    /// (what the host loss reads) are downloaded; every other
    /// `cpu_forward.layer_outputs` entry is a shape-only tensor with empty data
    /// and `layer_activation_inputs` is `None`.  Networks that need the CPU
    /// backward fallback still get the full host copies.
    pub fn forward_training_device(&self, input: &Tensor) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_impl(input, true, false)
    }

    pub(super) fn forward_with_state_mode(
        &self,
        input: &Tensor,
        training: bool,
    ) -> Result<GpuForwardState<B::Tensor>> {
        self.forward_with_state_impl(input, training, true)
    }

    pub(super) fn forward_with_state_impl(
        &self,
        input: &Tensor,
        training: bool,
        host_copies: bool,
    ) -> Result<GpuForwardState<B::Tensor>> {
        reject_gpu_stage1_cpu_layers(&self.network, "execution")?;
        let has_batch_normalized_recurrent = self.network.layers.iter().any(|layer| {
            matches!(&layer.kind, LayerKind::Rnn(spec) if spec.batch_normalize)
                || matches!(&layer.kind, LayerKind::Lstm(spec) if spec.batch_normalize)
                || matches!(&layer.kind, LayerKind::Crnn(spec) if spec.batch_normalize)
        });
        // Device `[deform_conv]` covers dilated and non-dilated non-BN paths;
        // batch-normalized variants keep the host path. This global decision
        // makes forward and backward choose the same cache producer.
        let host_only_deform = has_host_only_deform(&self.network);
        let has_hyperbolic_foveation = self
            .network
            .layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::HyperbolicFoveation(_)));
        let host_copies = host_copies
            || self.requires_cpu_backward_fallback()
            || has_batch_normalized_recurrent
            || host_only_deform
            || has_host_only_graph_conv(&self.network)
            || has_hyperbolic_foveation;
        // Path selection for both deformable families must match the backward's
        // gate exactly; independent of whether host copies are retained.
        // DCNv4's own unsupported variants are in requires_cpu_backward_fallback.
        let deform_device = !self.requires_cpu_backward_fallback() && !host_only_deform;
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        if let Some(layer) = self
            .network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::Wmhf(_)))
        {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[wmhf] GPU execution is not implemented in the Stage-1 CPU port; use the CPU forward path".to_owned(),
            });
        }
        if let Some(layer) = self
            .network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::RecursiveBlock(_)))
        {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[recursive_block] GPU execution is not implemented in the Stage-1 CPU port; use the CPU forward path"
                    .to_owned(),
            });
        }
        if let Some(layer) = self
            .network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::Clifford(_)))
        {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[clifford] GPU execution is not implemented in the Stage-1 CPU port; use the CPU forward path".to_owned(),
            });
        }
        if let Some(layer) = self
            .network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::ObbRefine(_)))
        {
            let message = if training {
                "GPU CARPN training is unsupported until Stage-2 assignment, backward, and updates are implemented"
            } else {
                "GPU CARPN inference is not implemented; use the CPU forward path"
            };
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: format!("[obb_refine]: {message}"),
            });
        }

        let launch_config = ElementwiseLaunchConfig::default();
        let input_gpu = self
            .backend
            .upload_f32(&tensor_backend_shape(input.shape), &input.data)
            .map_err(|err| backend_error("upload input tensor", err))?;

        let mut gpu_layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut gpu_layer_activation_inputs = Vec::with_capacity(self.network.layers.len());
        let mut layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut layer_activation_inputs = Vec::with_capacity(self.network.layers.len());
        let mut recurrent_caches = Vec::with_capacity(self.network.layers.len());
        let mut gpu_rnn_caches = Vec::with_capacity(self.network.layers.len());
        let mut gpu_crnn_caches = Vec::with_capacity(self.network.layers.len());
        let mut gpu_lstm_caches = Vec::with_capacity(self.network.layers.len());
        let mut gpu_moe_caches = Vec::with_capacity(self.network.layers.len());
        let mut gpu_eml_caches = Vec::with_capacity(self.network.layers.len());
        let mut graph_conv_caches = Vec::with_capacity(self.network.layers.len());
        let mut dropout_masks = Vec::with_capacity(self.network.layers.len());
        let mut crop_caches = Vec::with_capacity(self.network.layers.len());
        let mut moe_caches: Vec<Option<MoeForwardCache>> = Vec::new();
        let mut eml_caches: Vec<Option<EmlForwardCache>> =
            Vec::with_capacity(self.network.layers.len());
        let mut dcnv4_caches = Vec::with_capacity(self.network.layers.len());
        let mut gpu_dcnv4_caches = Vec::with_capacity(self.network.layers.len());
        let mut deform_conv_caches = Vec::with_capacity(self.network.layers.len());
        let mut deform_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut vit_caches: Vec<Option<crate::vit::VitForwardCache>> =
            Vec::with_capacity(self.network.layers.len());
        let mut swin_caches: Vec<Option<crate::swin_block::SwinForwardCache>> =
            Vec::with_capacity(self.network.layers.len());
        let mut tucker_attention_caches: Vec<
            Option<crate::tucker_attention::TuckerAttentionForwardCache>,
        > = Vec::with_capacity(self.network.layers.len());
        let mut mambavision_caches: Vec<Option<crate::mambavision::MambaVisionForward>> =
            Vec::with_capacity(self.network.layers.len());
        let mut kimi_kda_caches: Vec<Option<crate::kimi_kda::KimiKdaForwardCache>> =
            Vec::with_capacity(self.network.layers.len());
        let mut gpu_batch_norm_caches = Vec::with_capacity(self.network.layers.len());
        let mut hyperbolic_foveation_caches: Vec<Option<HyperbolicFoveationForwardCache>> =
            Vec::with_capacity(self.network.layers.len());
        let mut foveation_film_caches: Vec<
            Option<crate::foveation_film::FoveationFilmForwardCache>,
        > = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();
        let mut modern_heads = Vec::new();
        let mut rng_state = 0x9e37_79b9_u64;
        // Keep legacy rolling-stat behavior for networks with BN recurrent
        // layers while their recurrent forward uses the host reference path.
        let batch_norm_training =
            training && !self.requires_cpu_backward_fallback() && !has_batch_normalized_recurrent;

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                &input_gpu
            } else {
                gpu_layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let mut activation_input_gpu = None;
            let mut dropout_mask = None;
            let mut crop_cache = None;
            let mut batch_norm_cache = None;
            let mut vit_cache = None;
            let mut swin_cache = None;
            let mut tucker_attention_cache = None;
            let mut mambavision_cache = None;
            let mut kimi_kda_cache = None;
            let mut dcnv4_cache = None;
            let mut gpu_dcnv4_cache = None;
            let mut deform_conv_cache = None;
            let mut deform_device_cache: Option<DeformDeviceCache<B::Tensor>> = None;
            let mut moe_cache = None;
            let mut eml_cache = None;
            let mut gpu_rnn_cache = None;
            let mut gpu_crnn_cache = None;
            let mut gpu_lstm_cache = None;
            let mut gpu_moe_cache = None;
            let mut gpu_eml_cache = None;
            let mut graph_conv_cache = None;
            let mut hyperbolic_foveation_cache = None;
            let mut foveation_film_cache = None;
            let output_gpu = match (&layer.kind, params) {
                (
                    LayerKind::HyperbolicFoveation(spec),
                    LayerParams::HyperbolicFoveation(params),
                ) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download hyperbolic_foveation input", err))?;
                    let (_host_output, mut cache) = forward_hyperbolic_foveation(
                        &self.network,
                        layer,
                        spec,
                        params,
                        input,
                        &host_input,
                        &layer_outputs,
                    )?;
                    let native_grid_supported = matches!(spec.param_count(), 4 | 5);
                    if native_grid_supported && spec.pred_hidden > 0 {
                        let predictor_source = if let Some(source_layer) = spec.pred_source_layer {
                            gpu_layer_outputs.get(source_layer).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "[hyperbolic_foveation] predictor source layer #{source_layer} is unavailable on device"
                                ))
                            })?
                        } else {
                            current_input
                        };
                        let predicted = self
                            .backend
                            .hyperbolic_foveation_predictor_f32(
                                predictor_source,
                                &tensor_backend_shape(spec.pred_input_shape),
                                spec.downsample,
                                spec.pred_hidden,
                                spec.param_count(),
                                &params.pred_w1,
                                &params.pred_b1,
                                &params.pred_w2,
                                &params.pred_b2,
                                launch_config,
                            )
                            .map_err(|err| backend_error("native HF predictor forward", err))?;
                        let predicted = self.backend.download_f32(&predicted).map_err(|err| {
                            backend_error("download native HF predictor parameters", err)
                        })?;
                        if predicted.len() != cache.params_per_batch.len() {
                            return Err(Error::InvalidConfig(format!(
                                "native HF predictor produced {} parameters, expected {}",
                                predicted.len(),
                                cache.params_per_batch.len()
                            )));
                        }
                        cache.params_per_batch.clone_from(&predicted);
                        if let Some(predictor_cache) = cache.predictor.as_mut() {
                            predictor_cache.constrained_params = predicted;
                        }
                    }
                    let grid = if native_grid_supported {
                        let params_per_batch = self
                            .backend
                            .upload_f32(
                                &[layer.output_shape.batch, spec.param_count()],
                                &cache.params_per_batch,
                            )
                            .map_err(|err| backend_error("upload HF isotropic parameters", err))?;
                        match spec.inverse_method {
                            crate::model::HfInverseMethod::FixedPoint => self
                                .backend
                                .hyperbolic_fixed_point_grid_f32(
                                    &params_per_batch,
                                    layer.output_shape.batch,
                                    layer.output_shape.height,
                                    layer.output_shape.width,
                                    spec.param_count(),
                                    spec.alpha,
                                    spec.p_exp,
                                    spec.nr_iters,
                                    spec.nr_eta,
                                    launch_config,
                                )
                                .map_err(|err| backend_error("native HF fixed-point grid", err))?,
                            crate::model::HfInverseMethod::Newton => self
                                .backend
                                .hyperbolic_newton_grid_f32(
                                    &params_per_batch,
                                    layer.output_shape.batch,
                                    layer.output_shape.height,
                                    layer.output_shape.width,
                                    spec.param_count(),
                                    spec.alpha,
                                    spec.p_exp,
                                    spec.nr_iters,
                                    spec.nr_eta,
                                    launch_config,
                                )
                                .map_err(|err| backend_error("native HF Newton grid", err))?,
                        }
                    } else {
                        self.backend
                            .upload_f32(
                                &[
                                    layer.output_shape.batch,
                                    layer.output_shape.height,
                                    layer.output_shape.width,
                                    2,
                                ],
                                &cache.grid,
                            )
                            .map_err(|err| backend_error("upload hyperbolic_foveation grid", err))?
                    };
                    if native_grid_supported {
                        cache.grid = self.backend.download_f32(&grid).map_err(|err| {
                            backend_error("download native HF grid for backward cache", err)
                        })?;
                    }
                    let warp_input = match spec.warp_source {
                        crate::model::HfWarpSource::Previous => current_input,
                        crate::model::HfWarpSource::NetworkInput => &input_gpu,
                    };
                    let output = self
                        .backend
                        .hyperbolic_foveation_warp_f32(
                            warp_input,
                            &grid,
                            &tensor_backend_shape(layer.output_shape),
                            launch_config,
                        )
                        .map_err(|err| backend_error("native hyperbolic_foveation warp", err))?;
                    hyperbolic_foveation_cache = Some(cache);
                    output
                }
                (LayerKind::FoveationFilm(spec), LayerParams::FoveationFilm(params)) => {
                    let hf_cache = hyperbolic_foveation_caches
                        .get(spec.hf_layer)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "[foveation_film] selected HF layer #{} has no forward cache",
                                spec.hf_layer
                            ))
                        })?;
                    let batch = layer.input_shape.batch;
                    let channels = layer.input_shape.channels;
                    let hf_params = self
                        .backend
                        .upload_f32(&[batch, spec.param_count], &hf_cache.params_per_batch)
                        .map_err(|err| backend_error("upload FoveationFilm HF parameters", err))?;
                    let w_embed = self
                        .backend
                        .upload_f32(&[spec.embed_dim, spec.param_count], &params.w_embed)
                        .map_err(|err| {
                            backend_error("upload FoveationFilm embedding weights", err)
                        })?;
                    let b_embed = self
                        .backend
                        .upload_f32(&[spec.embed_dim], &params.b_embed)
                        .map_err(|err| backend_error("upload FoveationFilm embedding bias", err))?;
                    let w_gamma = self
                        .backend
                        .upload_f32(&[channels, spec.embed_dim], &params.w_gamma)
                        .map_err(|err| backend_error("upload FoveationFilm gamma weights", err))?;
                    let b_gamma = self
                        .backend
                        .upload_f32(&[channels], &params.b_gamma)
                        .map_err(|err| backend_error("upload FoveationFilm gamma bias", err))?;
                    let w_beta = self
                        .backend
                        .upload_f32(&[channels, spec.embed_dim], &params.w_beta)
                        .map_err(|err| backend_error("upload FoveationFilm beta weights", err))?;
                    let b_beta = self
                        .backend
                        .upload_f32(&[channels], &params.b_beta)
                        .map_err(|err| backend_error("upload FoveationFilm beta bias", err))?;
                    let (output, embedding, gamma, beta) = self
                        .backend
                        .foveation_film_forward_f32(
                            current_input,
                            &hf_params,
                            &w_embed,
                            &b_embed,
                            &w_gamma,
                            &b_gamma,
                            &w_beta,
                            &b_beta,
                            &tensor_backend_shape(layer.input_shape),
                            spec.param_count,
                            spec.embed_dim,
                            launch_config,
                        )
                        .map_err(|err| backend_error("FoveationFilm native forward", err))?;
                    foveation_film_cache = Some(crate::foveation_film::FoveationFilmForwardCache {
                        embedding: self.backend.download_f32(&embedding).map_err(|err| {
                            backend_error("download FoveationFilm embedding cache", err)
                        })?,
                        gamma: self.backend.download_f32(&gamma).map_err(|err| {
                            backend_error("download FoveationFilm gamma cache", err)
                        })?,
                        beta: self.backend.download_f32(&beta).map_err(|err| {
                            backend_error("download FoveationFilm beta cache", err)
                        })?,
                    });
                    output
                }
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    let cached = if batch_norm_training && conv.batch_norm.is_some() {
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
                    // Batch-norm training convolves with the raw weights; the
                    // folded (rolling-stat) weights serve inference.
                    let weights = if batch_norm_training && conv.batch_norm.is_some() {
                        &cached.raw_weights
                    } else {
                        &cached.weights
                    };
                    let conv_output = self
                        .backend
                        .conv2d_f32(
                            current_input,
                            weights,
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
                            launch_config,
                        )
                        .map_err(|err| backend_error("convolution", err))?;
                    let biased = match conv.batch_norm.as_ref() {
                        Some(batch_norm) if batch_norm_training => {
                            let (normalized, cache) = self.forward_batchnorm_training_gpu(
                                &conv_output,
                                &cached,
                                batch_norm,
                                layer.output_shape,
                                spec.share_index.is_none(),
                                launch_config,
                            )?;
                            batch_norm_cache = Some(cache);
                            normalized
                        }
                        _ => self
                            .backend
                            .bias_add_f32(
                                &conv_output,
                                &cached.biases,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                launch_config,
                            )
                            .map_err(|err| backend_error("convolution bias add", err))?,
                    };
                    activation_input_gpu = Some(biased.clone());
                    if let Some(normalization) = spec.activation.to_channel_normalization() {
                        self.backend
                            .normalize_channels_f32(
                                &biased,
                                &tensor_backend_shape(layer.output_shape),
                                layer.output_shape.batch,
                                layer.output_shape.channels,
                                layer.output_shape.width * layer.output_shape.height,
                                normalization,
                                launch_config,
                            )
                            .map_err(|err| {
                                backend_error("convolution channel normalization", err)
                            })?
                    } else {
                        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                        self.backend
                            .activation_f32(&biased, activation, launch_config)
                            .map_err(|err| backend_error("convolution activation", err))?
                    }
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => if spec.depth {
                    self.backend.maxpool_depth_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        launch_config,
                    )
                } else {
                    self.backend.maxpool_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.padding,
                        launch_config,
                    )
                }
                .map_err(|err| backend_error("maxpool", err))?,
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "local_avgpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .local_avgpool_f32(
                            current_input,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            spec.size,
                            spec.stride_x,
                            spec.stride_y,
                            spec.padding,
                            launch_config,
                        )
                        .map_err(|err| backend_error("local_avgpool", err))?
                }
                (LayerKind::AvgPool, LayerParams::None) => self
                    .backend
                    .avgpool_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        launch_config,
                    )
                    .map_err(|err| backend_error("avgpool", err))?,
                (LayerKind::Crop(spec), LayerParams::None) => {
                    let (cache, adjustments) = plan_crop_forward(
                        spec,
                        layer.input_shape,
                        layer.output_shape,
                        training,
                        &mut rng_state,
                    )?;
                    let source_x = self
                        .backend
                        .upload_f32(&[cache.source_x.len()], &cache.source_x)
                        .map_err(|err| backend_error("upload crop source x", err))?;
                    let source_y = self
                        .backend
                        .upload_f32(&[cache.source_y.len()], &cache.source_y)
                        .map_err(|err| backend_error("upload crop source y", err))?;
                    let input_scales = self
                        .backend
                        .upload_f32(&[cache.input_scales.len()], &cache.input_scales)
                        .map_err(|err| backend_error("upload crop input scales", err))?;
                    let input_offsets = self
                        .backend
                        .upload_f32(&[cache.input_offsets.len()], &cache.input_offsets)
                        .map_err(|err| backend_error("upload crop input offsets", err))?;
                    let input_saturations: Vec<f32> = adjustments
                        .iter()
                        .map(|adjustment| adjustment.saturation)
                        .collect();
                    let input_exposures: Vec<f32> = adjustments
                        .iter()
                        .map(|adjustment| adjustment.exposure)
                        .collect();
                    let input_saturations = self
                        .backend
                        .upload_f32(&[input_saturations.len()], &input_saturations)
                        .map_err(|err| backend_error("upload crop saturations", err))?;
                    let input_exposures = self
                        .backend
                        .upload_f32(&[input_exposures.len()], &input_exposures)
                        .map_err(|err| backend_error("upload crop exposures", err))?;
                    let output = self
                        .backend
                        .crop_bilinear_f32(
                            current_input,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            layer.input_shape.height,
                            layer.input_shape.width,
                            &source_x,
                            &source_y,
                            &input_scales,
                            &input_offsets,
                            &input_saturations,
                            &input_exposures,
                            training,
                            launch_config,
                        )
                        .map_err(|err| backend_error("GPU crop", err))?;
                    crop_cache = Some(cache);
                    output
                }
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => {
                    if training {
                        let (output, cache) = self
                            .forward_batchnorm_training_gpu_from_host_params(
                                current_input,
                                &params.biases,
                                params,
                                layer.output_shape,
                                launch_config,
                            )?;
                        batch_norm_cache = Some(cache);
                        output
                    } else {
                        self.forward_batchnorm_inference_gpu(
                            current_input,
                            layer.input_shape,
                            params,
                            launch_config,
                        )?
                    }
                }
                (LayerKind::Route(spec), LayerParams::None) => {
                    let mut route_inputs = Vec::with_capacity(spec.layers.len());
                    let mut route_channel_offsets = Vec::with_capacity(spec.layers.len());
                    let mut route_channels = Vec::with_capacity(spec.layers.len());
                    let mut route_total_channels = Vec::with_capacity(spec.layers.len());
                    for raw_index in &spec.layers {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source =
                            gpu_layer_outputs
                                .get(source_index)
                                .ok_or(Error::InvalidRoute {
                                    layer_index: layer.index,
                                    reference: *raw_index,
                                })?;
                        let source_layer =
                            self.network.layers.get(source_index).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing source layer #{source_index} for route layer #{}",
                                    layer.index
                                ))
                            })?;
                        let chunk_channels =
                            source_layer.output_shape.channels / spec.groups.max(1);
                        route_inputs.push(source);
                        route_channel_offsets.push(chunk_channels * spec.group_id);
                        route_channels.push(chunk_channels);
                        route_total_channels.push(source_layer.output_shape.channels);
                    }

                    self.backend
                        .route_concat_f32(
                            &route_inputs,
                            &route_channel_offsets,
                            &route_channels,
                            &route_total_channels,
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("route concat", err))?
                }
                (LayerKind::LayaMarkerGather(spec), LayerParams::None) => {
                    let a_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let b_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let a_layer = self.network.layers.get(a_index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing source layer #{a_index} for [laya_marker_gather] layer #{}",
                            layer.index
                        ))
                    })?;
                    let b_layer = self.network.layers.get(b_index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing source layer #{b_index} for [laya_marker_gather] layer #{}",
                            layer.index
                        ))
                    })?;
                    let a_source = gpu_layer_outputs.get(a_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.layers[0],
                    })?;
                    let b_source = gpu_layer_outputs.get(b_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.layers[1],
                    })?;
                    self.backend
                        .marker_gather_f32(
                            a_source,
                            b_source,
                            &tensor_backend_shape(a_layer.output_shape),
                            &tensor_backend_shape(b_layer.output_shape),
                            spec.max_options,
                            launch_config,
                        )
                        .map_err(|err| backend_error("GPU laya_marker_gather", err))?
                }
                (LayerKind::LayaActHead(spec), LayerParams::LayaActHead(weights)) => {
                    let head_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let scorer_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let marker_index = resolve_relative_layer(layer.index, spec.layers[2])?;
                    let head_layer = self.network.layers.get(head_index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing source layer #{head_index} for [laya_act_head] layer #{}",
                            layer.index
                        ))
                    })?;
                    let scorer_layer = self.network.layers.get(scorer_index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing source layer #{scorer_index} for [laya_act_head] layer #{}",
                            layer.index
                        ))
                    })?;
                    let marker_layer = self.network.layers.get(marker_index).ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "missing source layer #{marker_index} for [laya_act_head] layer #{}",
                            layer.index
                        ))
                    })?;
                    let head_source =
                        gpu_layer_outputs
                            .get(head_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.layers[0],
                            })?;
                    let scorer_source =
                        gpu_layer_outputs
                            .get(scorer_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.layers[1],
                            })?;
                    let marker_source =
                        gpu_layer_outputs
                            .get(marker_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.layers[2],
                            })?;
                    laya_act_gpu::forward(
                        &self.backend,
                        spec,
                        weights,
                        [head_source, scorer_source, marker_source],
                        [
                            head_layer.output_shape,
                            scorer_layer.output_shape,
                            marker_layer.output_shape,
                        ],
                        launch_config,
                    )?
                }
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => {
                    let mut inputs = Vec::with_capacity(spec.from.len() + 1);
                    inputs.push(current_input.clone());
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source =
                            gpu_layer_outputs
                                .get(source_index)
                                .ok_or(Error::InvalidRoute {
                                    layer_index: layer.index,
                                    reference: *raw_index,
                                })?;
                        let source_layer =
                            self.network.layers.get(source_index).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing source layer #{source_index} for shortcut layer #{}",
                                    layer.index
                                ))
                            })?;
                        if source_layer.output_shape.batch != layer.output_shape.batch {
                            return Err(Error::InvalidConfig(format!(
                                "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                                layer.index, layer.output_shape, source_layer.output_shape
                            )));
                        }
                        if source_layer.output_shape == layer.output_shape {
                            inputs.push(source.clone());
                        } else {
                            let fitted = self
                                .backend
                                .shortcut_fit_f32(
                                    source,
                                    layer.output_shape.batch,
                                    &tensor_backend_shape(layer.output_shape),
                                    launch_config,
                                )
                                .map_err(|err| {
                                    backend_error("weighted shortcut resize source", err)
                                })?;
                            inputs.push(fitted);
                        }
                    }
                    let layer_step = spec.weight_slots(layer.output_shape.channels);
                    let per_sample = layer.output_shape.element_count();
                    let effective = crate::shortcut_weights::effective_weights(
                        &params.weights,
                        spec.weights_normalization,
                        layer_step,
                    );
                    let refs: Vec<&B::Tensor> = inputs.iter().collect();
                    let accumulated = self
                        .backend
                        .shortcut_weighted_sum_f32(
                            &refs,
                            &effective,
                            layer_step,
                            per_sample / layer_step,
                            launch_config,
                        )
                        .map_err(|err| backend_error("weighted shortcut sum", err))?;
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    activation_input_gpu = Some(accumulated.clone());
                    self.backend
                        .activation_f32(&accumulated, activation, launch_config)
                        .map_err(|err| backend_error("weighted shortcut activation", err))?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let mut accumulated = current_input.clone();
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source =
                            gpu_layer_outputs
                                .get(source_index)
                                .ok_or(Error::InvalidRoute {
                                    layer_index: layer.index,
                                    reference: *raw_index,
                                })?;
                        let source_layer =
                            self.network.layers.get(source_index).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing source layer #{source_index} for shortcut layer #{}",
                                    layer.index
                                ))
                            })?;
                        if source_layer.output_shape.batch != layer.output_shape.batch {
                            return Err(Error::InvalidConfig(format!(
                                "shortcut layer #{} mixes incompatible batch shapes {} and {}",
                                layer.index, layer.output_shape, source_layer.output_shape
                            )));
                        }
                        // Darknet resizes a mismatched source flat per sample
                        // (crop / zero-pad), then adds it.
                        let fitted;
                        let source = if source_layer.output_shape != layer.output_shape {
                            fitted = self
                                .backend
                                .shortcut_fit_f32(
                                    source,
                                    layer.output_shape.batch,
                                    &tensor_backend_shape(layer.output_shape),
                                    launch_config,
                                )
                                .map_err(|err| backend_error("shortcut resize source", err))?;
                            &fitted
                        } else {
                            source
                        };
                        accumulated = self
                            .backend
                            .shortcut_add_f32(&accumulated, source, launch_config)
                            .map_err(|err| backend_error("shortcut add", err))?;
                    }

                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    activation_input_gpu = Some(accumulated.clone());
                    self.backend
                        .activation_f32(&accumulated, activation, launch_config)
                        .map_err(|err| backend_error("shortcut activation", err))?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => self
                    .backend
                    .upsample_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        spec.stride.unsigned_abs(),
                        spec.scale,
                        spec.reverse,
                        launch_config,
                    )
                    .map_err(|err| backend_error("upsample", err))?,
                (LayerKind::JevCe(spec), LayerParams::None) => self
                    .backend
                    .jev_ce_forward_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        spec.groups,
                        spec.candidates,
                        spec.temperature,
                        spec.channel_major,
                        launch_config,
                    )
                    .map_err(|err| backend_error("JevCe forward", err))?,
                (LayerKind::LayaLoss(spec), LayerParams::None) => self
                    .backend
                    .laya_loss_forward_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        spec.max_options,
                        launch_config,
                    )
                    .map_err(|err| backend_error("LayaLoss forward", err))?,
                (LayerKind::GraphConv(spec), LayerParams::GraphConv(params)) => {
                    let graph_conv_host_fallback = graph_conv_spec_needs_compact_host(spec)
                        || (training
                            && (graph_conv_spec_needs_host(spec)
                                || self.requires_cpu_backward_fallback()
                                || has_host_only_deform(&self.network)));
                    if graph_conv_host_fallback {
                        let current_host = if layer.index == 0 {
                            input
                        } else {
                            layer_outputs.get(layer.index - 1).ok_or_else(|| {
                                Error::InvalidConfig(format!(
                                    "missing host input for GraphConv fallback layer #{}",
                                    layer.index
                                ))
                            })?
                        };
                        let (host_output, host_activation, cache) = forward_graph_conv_with_cache(
                            spec,
                            params,
                            current_host,
                            layer.output_shape,
                            training,
                        )?;
                        graph_conv_cache = Some(cache);
                        activation_input_gpu = Some(
                            self.backend
                                .upload_f32(
                                    &tensor_backend_shape(layer.output_shape),
                                    &host_activation.data,
                                )
                                .map_err(|err| {
                                    backend_error("upload GraphConv fallback activation", err)
                                })?,
                        );
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload GraphConv fallback output", err))?
                    } else {
                        let geometry = graph_conv_aggregation_geometry(
                            spec,
                            layer.input_shape,
                            layer.output_shape,
                        );
                        let groups = spec.groups.max(1);
                        let aggregate = if spec.edge_mode == 0 {
                            self.backend
                                .graph_conv_aggregate_f32(current_input, geometry, launch_config)
                                .map_err(|err| {
                                    backend_error("GraphConv uniform aggregation", err)
                                })?
                        } else {
                            let edge_kernel_values =
                                params.edge_kernel.as_deref().ok_or_else(|| {
                                    Error::InvalidConfig(
                                        "GraphConv learned edge kernel is missing".to_owned(),
                                    )
                                })?;
                            let edge_bias_values =
                                params.edge_biases.as_deref().ok_or_else(|| {
                                    Error::InvalidConfig(
                                        "GraphConv learned edge biases are missing".to_owned(),
                                    )
                                })?;
                            let edge_kernel = self
                                .backend
                                .upload_f32(&[edge_kernel_values.len()], edge_kernel_values)
                                .map_err(|err| {
                                    backend_error("upload GraphConv edge kernel", err)
                                })?;
                            let edge_biases = self
                                .backend
                                .upload_f32(&[edge_bias_values.len()], edge_bias_values)
                                .map_err(|err| {
                                    backend_error("upload GraphConv edge biases", err)
                                })?;
                            self.backend
                                .graph_conv_learned_aggregate_f32(
                                    current_input,
                                    &edge_kernel,
                                    &edge_biases,
                                    geometry,
                                    groups,
                                    launch_config,
                                )
                                .map_err(|err| {
                                    backend_error("GraphConv learned aggregation", err)
                                })?
                        };
                        let weight_shape =
                            [spec.filters, layer.input_shape.channels / groups, 1, 1];
                        let weights = self
                            .backend
                            .upload_f32(&weight_shape, &params.projection.weights)
                            .map_err(|err| {
                                backend_error("upload GraphConv projection weights", err)
                            })?;
                        let raw = self
                            .backend
                            .conv2d_f32(
                                &aggregate,
                                &weights,
                                &tensor_backend_shape(layer.output_shape),
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
                                launch_config,
                            )
                            .map_err(|err| backend_error("GraphConv pointwise projection", err))?;
                        let raw = if spec.use_self {
                            let reference = self
                                .backend
                                .graph_conv_reference_f32(current_input, geometry, launch_config)
                                .map_err(|err| {
                                    backend_error("GraphConv center-reference gather", err)
                                })?;
                            let self_values = params.self_weights.as_deref().ok_or_else(|| {
                                Error::InvalidConfig(
                                    "GraphConv self weights are missing".to_owned(),
                                )
                            })?;
                            let self_weights = self
                                .backend
                                .upload_f32(&weight_shape, self_values)
                                .map_err(|err| {
                                    backend_error("upload GraphConv self weights", err)
                                })?;
                            let self_raw = self
                                .backend
                                .conv2d_f32(
                                    &reference,
                                    &self_weights,
                                    &tensor_backend_shape(layer.output_shape),
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
                                    launch_config,
                                )
                                .map_err(|err| backend_error("GraphConv self projection", err))?;
                            self.backend
                                .shortcut_add_f32(&raw, &self_raw, launch_config)
                                .map_err(|err| backend_error("GraphConv projection sum", err))?
                        } else {
                            raw
                        };
                        let biased = if spec.batch_normalize && training {
                            let batch_norm =
                                params.projection.batch_norm.as_ref().ok_or_else(|| {
                                    Error::InvalidConfig(
                                        "GraphConv batch-norm parameters are missing".to_owned(),
                                    )
                                })?;
                            let (normalized, cache) = self
                                .forward_batchnorm_training_gpu_from_host_params(
                                    &raw,
                                    &params.projection.biases,
                                    batch_norm,
                                    layer.output_shape,
                                    launch_config,
                                )?;
                            batch_norm_cache = Some(cache);
                            normalized
                        } else if spec.batch_normalize {
                            let mut batch_norm =
                                params.projection.batch_norm.clone().ok_or_else(|| {
                                    Error::InvalidConfig(
                                        "GraphConv batch-norm parameters are missing".to_owned(),
                                    )
                                })?;
                            batch_norm.biases.clone_from(&params.projection.biases);
                            self.forward_batchnorm_inference_gpu(
                                &raw,
                                layer.output_shape,
                                &batch_norm,
                                launch_config,
                            )?
                        } else {
                            let biases = self
                                .backend
                                .upload_f32(&[spec.filters], &params.projection.biases)
                                .map_err(|err| {
                                    backend_error("upload GraphConv projection biases", err)
                                })?;
                            self.backend
                                .bias_add_f32(
                                    &raw,
                                    &biases,
                                    spec.filters,
                                    layer.output_shape.width * layer.output_shape.height,
                                    launch_config,
                                )
                                .map_err(|err| backend_error("GraphConv projection bias", err))?
                        };
                        activation_input_gpu = Some(biased.clone());
                        if let Some(normalization) = spec.activation.to_channel_normalization() {
                            self.backend
                                .normalize_channels_f32(
                                    &biased,
                                    &tensor_backend_shape(layer.output_shape),
                                    layer.output_shape.batch,
                                    layer.output_shape.channels,
                                    layer.output_shape.width * layer.output_shape.height,
                                    normalization,
                                    launch_config,
                                )
                                .map_err(|err| {
                                    backend_error("GraphConv channel normalization", err)
                                })?
                        } else {
                            let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                            self.backend
                                .activation_f32(&biased, activation, launch_config)
                                .map_err(|err| backend_error("GraphConv activation", err))?
                        }
                    }
                }
                (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params)) => {
                    if deform_device {
                        let device_params = {
                            let mut caches = self.dcnv4_device_params.borrow_mut();
                            dcnv4_gpu::device_parameters(
                                &self.backend,
                                &mut caches[layer.index],
                                layer,
                                spec,
                                params,
                            )?
                        };
                        let (output, pre, cache) = dcnv4_gpu::forward(
                            &self.backend,
                            current_input,
                            layer,
                            spec,
                            params,
                            &device_params,
                        )?;
                        activation_input_gpu = Some(pre);
                        gpu_dcnv4_cache = Some(cache);
                        output
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download DCNv4 input", err))?;
                        let (host_output, host_activation, cache) = forward_dcnv4_with_cache(
                            spec,
                            params,
                            &host_input,
                            layer.output_shape,
                        )?;
                        dcnv4_cache = Some(cache);
                        activation_input_gpu = Some(
                            self.backend
                                .upload_f32(
                                    &tensor_backend_shape(layer.output_shape),
                                    &host_activation.data,
                                )
                                .map_err(|err| {
                                    backend_error("upload DCNv4 activation cache", err)
                                })?,
                        );
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload DCNv4 output", err))?
                    }
                }
                (LayerKind::DsConv(spec), LayerParams::DsConv(params)) => {
                    let activation = spec.activation.to_cuda_f32_kind().ok_or_else(|| {
                        Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: format!(
                                "[dsconv] unsupported GPU activation {:?}",
                                spec.activation
                            ),
                        }
                    })?;
                    let geometry = darknet_cuda::host_ops::DsConvGeometry {
                        batch: layer.input_shape.batch,
                        channels: layer.input_shape.channels,
                        filters: spec.filters,
                        input_height: layer.input_shape.height,
                        input_width: layer.input_shape.width,
                        kernel_size: spec.size,
                        extend_scope: spec.extend_scope,
                        morph: spec.morph,
                    };
                    let offset_channels = 2 * spec.size;
                    let offset_weights = self
                        .backend
                        .upload_f32(
                            &[offset_channels, layer.input_shape.channels, 3, 3],
                            &params.offset_weights,
                        )
                        .map_err(|err| backend_error("upload DSConv offset weights", err))?;
                    let offset_biases = self
                        .backend
                        .upload_f32(&[offset_channels], &params.offset_biases)
                        .map_err(|err| backend_error("upload DSConv offset biases", err))?;
                    let weights = self
                        .backend
                        .upload_f32(
                            &[spec.filters, layer.input_shape.channels, spec.size],
                            &params.collapse.weights,
                        )
                        .map_err(|err| backend_error("upload DSConv collapse weights", err))?;
                    let biases = self
                        .backend
                        .upload_f32(&[spec.filters], &params.collapse.biases)
                        .map_err(|err| backend_error("upload DSConv collapse biases", err))?;
                    self.backend
                        .dsconv_forward_native_f32(
                            current_input,
                            &offset_weights,
                            &offset_biases,
                            &weights,
                            &biases,
                            geometry,
                            activation,
                            launch_config,
                        )
                        .ok_or_else(|| Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "backend does not provide native f32 [dsconv]".to_owned(),
                        })?
                        .map_err(|err| backend_error("native DSConv forward", err))?
                }
                (LayerKind::DeformConv(spec), LayerParams::DeformConv(params)) => {
                    if deform_device {
                        let geometry = darknet_cuda::host_ops::deform_conv::DeformGeometry {
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
                        let kernels = spec.size * spec.size;
                        let plane = geometry.output_width * geometry.output_height;
                        let in_shape = tensor_backend_shape(layer.input_shape);
                        let out_shape = tensor_backend_shape(layer.output_shape);
                        let sanitized = self
                            .backend
                            .deform_sanitize_f32(current_input, &in_shape, launch_config)
                            .map_err(|err| backend_error("deform_conv sanitize", err))?;
                        let offset_w = self
                            .backend
                            .upload_f32(&[params.offset_weights.len()], &params.offset_weights)
                            .map_err(|err| backend_error("upload deform offset weights", err))?;
                        let offset_b = self
                            .backend
                            .upload_f32(&[params.offset_biases.len()], &params.offset_biases)
                            .map_err(|err| backend_error("upload deform offset biases", err))?;
                        let predictor_shape = [
                            geometry.batch,
                            2 * kernels,
                            geometry.output_height,
                            geometry.output_width,
                        ];
                        let raw_offsets = self
                            .backend
                            .deform_predictor_f32(
                                &sanitized,
                                &offset_w,
                                &offset_b,
                                &predictor_shape,
                                geometry.channels,
                                geometry.input_height,
                                geometry.input_width,
                                2 * kernels,
                                geometry.size,
                                geometry.stride_x,
                                geometry.stride_y,
                                geometry.padding,
                                geometry.dilation,
                                geometry.output_height,
                                geometry.output_width,
                                launch_config,
                            )
                            .map_err(|err| backend_error("deform offset predictor", err))?;
                        let max_offset = spec.size as f32 * spec.dilation as f32 * 2.0;
                        let (mean, inv_std, stddev, normalize) = {
                            let (m, i, n) = self
                                .backend
                                .deform_offset_stats(
                                    &raw_offsets,
                                    2 * kernels * plane,
                                    max_offset,
                                    launch_config,
                                )
                                .map_err(|err| backend_error("deform offset stats", err))?;
                            // `stddev` is not exposed by the trait signature;
                            // recover it on the host from the small [batch] tensor.
                            let inv = self
                                .backend
                                .download_f32(&i)
                                .map_err(|err| backend_error("download deform inv_std", err))?;
                            let std_host: Vec<f32> = inv
                                .iter()
                                .map(|&v| if v > 0.0 { 1.0 / v } else { 0.0 })
                                .collect();
                            let stddev = self
                                .backend
                                .upload_f32(&[std_host.len()], &std_host)
                                .map_err(|err| backend_error("upload deform stddev", err))?;
                            (m, i, stddev, n)
                        };
                        let offsets = self
                            .backend
                            .deform_offset_envelope(
                                &raw_offsets,
                                &mean,
                                &inv_std,
                                &normalize,
                                2 * kernels,
                                plane,
                                max_offset,
                                &predictor_shape,
                                launch_config,
                            )
                            .map_err(|err| backend_error("deform offset envelope", err))?;
                        let (masks, mask_logits) = if spec.use_mask {
                            let weights = params.mask_weights.as_ref().ok_or_else(|| {
                                Error::InvalidConfig(
                                    "[deform_conv]: missing mask weights".to_owned(),
                                )
                            })?;
                            let biases = params.mask_biases.as_ref().ok_or_else(|| {
                                Error::InvalidConfig(
                                    "[deform_conv]: missing mask biases".to_owned(),
                                )
                            })?;
                            let w = self
                                .backend
                                .upload_f32(&[weights.len()], weights)
                                .map_err(|err| backend_error("upload deform mask weights", err))?;
                            let b = self
                                .backend
                                .upload_f32(&[biases.len()], biases)
                                .map_err(|err| backend_error("upload deform mask biases", err))?;
                            let logits_shape = [
                                geometry.batch,
                                kernels,
                                geometry.output_height,
                                geometry.output_width,
                            ];
                            let logits = self
                                .backend
                                .deform_predictor_f32(
                                    &sanitized,
                                    &w,
                                    &b,
                                    &logits_shape,
                                    geometry.channels,
                                    geometry.input_height,
                                    geometry.input_width,
                                    kernels,
                                    geometry.size,
                                    geometry.stride_x,
                                    geometry.stride_y,
                                    geometry.padding,
                                    geometry.dilation,
                                    geometry.output_height,
                                    geometry.output_width,
                                    launch_config,
                                )
                                .map_err(|err| backend_error("deform mask predictor", err))?;
                            let masks = self
                                .backend
                                .deform_mask_sigmoid(&logits, &logits_shape, launch_config)
                                .map_err(|err| backend_error("deform mask sigmoid", err))?;
                            (Some(masks), Some(logits))
                        } else {
                            (None, None)
                        };
                        let main_w = self
                            .backend
                            .upload_f32(&[params.main.weights.len()], &params.main.weights)
                            .map_err(|err| backend_error("upload deform main weights", err))?;
                        let empty_masks = self
                            .backend
                            .zeros_f32(&[1])
                            .map_err(|err| backend_error("deform empty masks", err))?;
                        let raw = self
                            .backend
                            .deform_conv_accumulate_f32(
                                &sanitized,
                                &offsets,
                                masks.as_ref().unwrap_or(&empty_masks),
                                &main_w,
                                geometry,
                                masks.is_some(),
                                &out_shape,
                                launch_config,
                            )
                            .map_err(|err| backend_error("deform accumulate", err))?;
                        let main_bias = self
                            .backend
                            .upload_f32(&[params.main.biases.len()], &params.main.biases)
                            .map_err(|err| backend_error("upload deform main biases", err))?;
                        let biased = self
                            .backend
                            .bias_add_f32(&raw, &main_bias, spec.filters, plane, launch_config)
                            .map_err(|err| backend_error("deform bias add", err))?;
                        activation_input_gpu = Some(biased.clone());
                        deform_device_cache = Some(DeformDeviceCache {
                            input_sanitized: sanitized,
                            raw_offsets,
                            mean,
                            inv_std,
                            stddev,
                            normalize,
                            offsets,
                            masks,
                            mask_logits,
                        });
                        self.backend
                            .activation_f32(
                                &biased,
                                gpu_activation_kind(layer.index, &spec.activation)?,
                                launch_config,
                            )
                            .map_err(|err| backend_error("deform activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download deform_conv input", err))?;
                        let (host_output, host_activation, cache) = forward_deform_conv_with_cache(
                            spec,
                            params,
                            &host_input,
                            layer.output_shape,
                        )?;
                        deform_conv_cache = Some(cache);
                        activation_input_gpu = Some(
                            self.backend
                                .upload_f32(
                                    &tensor_backend_shape(layer.output_shape),
                                    &host_activation.data,
                                )
                                .map_err(|err| {
                                    backend_error("upload deform_conv activation cache", err)
                                })?,
                        );
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload deform_conv output", err))?
                    }
                }
                (LayerKind::Vit(spec), LayerParams::Vit(weights)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download vit input", err))?;
                    let geometry = vit_geometry(layer, spec);
                    let (data, cache) = crate::vit::vit_forward_with_flash_attention(
                        &host_input.data,
                        weights,
                        &geometry,
                        |qkv, batch, tokens, channels, heads| {
                            if channels / heads > 128 {
                                return crate::vit::attention_forward_flash(
                                    qkv, batch, tokens, channels, heads,
                                );
                            }
                            let qkv_tensor = self
                                .backend
                                .upload_f32(&[batch, tokens, 3 * channels], qkv)
                                .map_err(|err| format!("upload ViT flash QKV: {err}"))?;
                            let (context_tensor, lse_tensor) = self
                                .backend
                                .flash_attention_forward_f32(
                                    &qkv_tensor,
                                    &[batch, tokens, channels],
                                    &[batch, heads, tokens],
                                    batch,
                                    tokens,
                                    channels,
                                    heads,
                                )
                                .map_err(|err| format!("ViT flash forward: {err}"))?;
                            let context = self
                                .backend
                                .download_f32(&context_tensor)
                                .map_err(|err| format!("download ViT flash context: {err}"))?;
                            let lse = self
                                .backend
                                .download_f32(&lse_tensor)
                                .map_err(|err| format!("download ViT flash LSE: {err}"))?;
                            let attention = crate::vit::attention_flash_cache_from_lse(
                                qkv, batch, tokens, channels, heads, &lse,
                            )?;
                            Ok((context, attention))
                        },
                    )
                    .map_err(Error::InvalidConfig)?;
                    vit_cache = Some(cache);
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
                        .map_err(|err| backend_error("upload vit output", err))?
                }
                (LayerKind::Swin(spec), LayerParams::Swin(weights)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download swin input", err))?;
                    let geometry = crate::swin_block::SwinGeometry {
                        batch: layer.input_shape.batch,
                        channels: layer.input_shape.channels,
                        height: layer.input_shape.height,
                        width: layer.input_shape.width,
                        filters: spec.filters,
                        heads: spec.heads,
                        window: spec.size,
                        shift: spec.shift,
                        ffn_ratio: spec.ffn_ratio,
                        activation: spec.activation.clone(),
                    };
                    let (data, cache) =
                        crate::swin_block::swin_forward(&host_input.data, weights, &geometry)
                            .map_err(Error::InvalidConfig)?;
                    swin_cache = Some(cache);
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
                        .map_err(|err| backend_error("upload swin output", err))?
                }
                (LayerKind::TuckerAttention(spec), LayerParams::TuckerAttention(weights)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download Tucker attention input", err))?;
                    let geometry = crate::tucker_attention::TuckerGeometry {
                        batch: layer.input_shape.batch,
                        channels: layer.input_shape.channels,
                        height: layer.input_shape.height,
                        width: layer.input_shape.width,
                        size: spec.size,
                        heads: spec.heads,
                        rank_q: spec.rank_q,
                        rank_k: spec.rank_k,
                        rank_v: spec.rank_v,
                        rank_o: spec.rank_o,
                        activation: spec.activation.clone(),
                        msa_enabled: spec.msa_enabled,
                        msa_block_size: spec.msa_block_size,
                        msa_topk: spec.msa_topk,
                    };
                    let (data, cache) = crate::tucker_attention::tucker_attention_forward(
                        &host_input.data,
                        weights,
                        &geometry,
                    )
                    .map_err(Error::InvalidConfig)?;
                    tucker_attention_cache = Some(cache);
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
                        .map_err(|err| backend_error("upload Tucker attention output", err))?
                }
                (LayerKind::MambaVision(spec), LayerParams::MambaVision(weights)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download MambaVision input", err))?;
                    let forward = crate::mambavision::mambavision_forward(
                        &host_input.data,
                        host_input.shape,
                        spec,
                        weights,
                        None,
                    )
                    .map_err(Error::InvalidConfig)?;
                    mambavision_cache = Some(forward.clone());
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &forward.output)
                        .map_err(|err| backend_error("upload MambaVision output", err))?
                }
                (LayerKind::KimiKda(spec), LayerParams::KimiKda(weights)) => {
                    let host_input = self
                        .download_tensor(current_input, layer.input_shape)
                        .map_err(|err| backend_error("download Kimi KDA input", err))?;
                    let (data, cache) = crate::kimi_kda::kimi_kda_forward(
                        &host_input.data,
                        host_input.shape,
                        spec,
                        weights,
                    )
                    .map_err(Error::InvalidConfig)?;
                    kimi_kda_cache = Some(cache);
                    self.backend
                        .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
                        .map_err(|err| backend_error("upload Kimi KDA output", err))?
                }
                (LayerKind::LayaEncoder(spec), LayerParams::LayaEncoder(weights)) => {
                    laya_encoder_gpu::forward(
                        &self.backend,
                        spec,
                        weights,
                        current_input,
                        layer.input_shape,
                        layer.output_shape,
                        launch_config,
                    )?
                }
                (LayerKind::LayaHead(spec), LayerParams::LayaHead(weights)) => {
                    laya_head_gpu::forward(
                        &self.backend,
                        spec,
                        weights,
                        current_input,
                        layer.input_shape,
                        layer.output_shape,
                        launch_config,
                    )?
                }
                (LayerKind::Logistic, LayerParams::None) => self
                    .backend
                    .activation_f32(
                        current_input,
                        darknet_cuda::F32ActivationKind::Logistic,
                        launch_config,
                    )
                    .map_err(|err| backend_error("logistic", err))?,
                (LayerKind::Segmentation(spec), LayerParams::None) => self
                    .backend
                    .segmentation_forward_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        spec.classes,
                        spec.temperature,
                        launch_config,
                    )
                    .map_err(|err| backend_error("segmentation forward", err))?,
                (LayerKind::LayaScorer(spec), LayerParams::LayaScorer(weights)) => {
                    laya_scorer_gpu::forward(
                        &self.backend,
                        spec,
                        weights,
                        current_input,
                        layer.input_shape,
                        layer.output_shape,
                        launch_config,
                    )?
                }
                // Native strided NCHW sigmoid for heatmap channels; regression
                // channels are copied and all output values are sanitized.
                (LayerKind::CenterNet(spec), LayerParams::None) => self
                    .backend
                    .centernet_forward_f32(
                        current_input,
                        &tensor_backend_shape(layer.input_shape),
                        spec.classes,
                        launch_config,
                    )
                    .map_err(|err| backend_error("CenterNet forward", err))?,
                (LayerKind::Yolox(spec), LayerParams::None)
                | (LayerKind::Ppyoloe(spec), LayerParams::None)
                | (LayerKind::Yolonas(spec), LayerParams::None) => self
                    .backend
                    .modern_yolo_forward_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        spec.classes,
                        launch_config,
                    )
                    .map_err(|err| backend_error("modern-YOLO forward", err))?,
                (LayerKind::Region(spec), LayerParams::None) => self
                    .backend
                    .region_forward_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        spec.num,
                        spec.coords + spec.classes + 1,
                        spec.classes,
                        spec.softmax,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU Region forward", err))?,
                // `[omd_loss]` is a pure pass-through on the forward, exactly as
                // in C++ (`memcpy(l.output, state.input, ...)`).
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::ObbYolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None)
                | (LayerKind::OmdLoss(_), LayerParams::None)
                | (LayerKind::Contrastive(_), LayerParams::None) => current_input.clone(),
                (
                    LayerKind::Dropout {
                        probability,
                        dropblock,
                    },
                    LayerParams::None,
                ) => {
                    let probability = probability.unwrap_or(0.2);
                    check_dropout_probability(probability)?;
                    let block_plan = dropblock.as_ref().map(|spec| {
                        dropblock_plan(
                            spec,
                            probability,
                            layer.output_shape,
                            self.train_iteration,
                            self.network.net.max_batches,
                        )
                    });
                    if !training
                        || probability == 0.0
                        || matches!(block_plan, Some((p, _)) if p <= 0.0)
                    {
                        current_input.clone()
                    } else {
                        let seed = advance_rng(&mut rng_state);
                        let (output, mask) = match block_plan {
                            Some((block_prob, block_size)) => self
                                .backend
                                .dropblock_f32(
                                    current_input,
                                    &tensor_backend_shape(layer.output_shape),
                                    block_prob,
                                    block_size,
                                    dropblock_seed(seed, self.train_iteration),
                                    launch_config,
                                )
                                .map_err(|err| backend_error("GPU dropblock", err))?,
                            None => self
                                .backend
                                .dropout_f32(
                                    current_input,
                                    &tensor_backend_shape(layer.output_shape),
                                    probability,
                                    seed,
                                    launch_config,
                                )
                                .map_err(|err| backend_error("GPU dropout", err))?,
                        };
                        dropout_mask = Some(
                            self.download_tensor(&mask, layer.output_shape)
                                .map_err(|err| backend_error("download dropout mask", err))?,
                        );
                        output
                    }
                }
                (LayerKind::Local(spec), LayerParams::Local(local)) => {
                    if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                        let locations = layer.output_shape.width * layer.output_shape.height;
                        let weights = self
                            .backend
                            .upload_f32(
                                &[
                                    locations,
                                    spec.filters,
                                    layer.input_shape.channels,
                                    spec.size,
                                    spec.size,
                                ],
                                &local.weights,
                            )
                            .map_err(|err| backend_error("upload local weights", err))?;
                        let biases = self
                            .backend
                            .upload_f32(&[locations, spec.filters], &local.biases)
                            .map_err(|err| backend_error("upload local biases", err))?;
                        let pre_activation = self
                            .backend
                            .local_f32(
                                current_input,
                                &weights,
                                &biases,
                                &tensor_backend_shape(layer.input_shape),
                                &tensor_backend_shape(layer.output_shape),
                                spec.filters,
                                spec.size,
                                spec.stride_x,
                                spec.stride_y,
                                spec.padding,
                                launch_config,
                            )
                            .map_err(|err| backend_error("local forward", err))?;
                        activation_input_gpu = Some(pre_activation.clone());
                        self.backend
                            .activation_f32(&pre_activation, activation, launch_config)
                            .map_err(|err| backend_error("local activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download local input", err))?;
                        let (host_output, host_activation_input) =
                            forward_local_with_cache(spec, local, &host_input, layer.output_shape)?;
                        if let Some(pre_activation) = host_activation_input {
                            activation_input_gpu = Some(
                                self.backend
                                    .upload_f32(
                                        &tensor_backend_shape(layer.output_shape),
                                        &pre_activation.data,
                                    )
                                    .map_err(|err| {
                                        backend_error("upload local activation cache", err)
                                    })?,
                            );
                        }
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload local output", err))?
                    }
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    let inputs = layer.input_shape.element_count();
                    if connected.weights.len() != spec.output * inputs
                        || connected.biases.len() != spec.output
                    {
                        return Err(Error::InvalidConfig(format!(
                            "connected layer #{} expects {} weights and {} biases, got {} and {}",
                            layer.index,
                            spec.output * inputs,
                            spec.output,
                            connected.weights.len(),
                            connected.biases.len()
                        )));
                    }
                    if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                        // Inference batch norm is a per-output affine map, so
                        // fold it into the uploaded weights and biases.
                        let (weights, biases) = match connected.batch_norm.as_ref() {
                            Some(bn) => {
                                let mut weights = connected.weights.clone();
                                let mut biases = connected.biases.clone();
                                for (o, row) in weights.chunks_mut(inputs).enumerate() {
                                    let k = bn.scales[o]
                                        / (bn.rolling_variance[o] + BATCH_NORM_EPSILON).sqrt();
                                    row.iter_mut().for_each(|w| *w *= k);
                                    biases[o] -= bn.rolling_mean[o] * k;
                                }
                                (weights, biases)
                            }
                            None => (connected.weights.clone(), connected.biases.clone()),
                        };
                        let weights_gpu = self
                            .backend
                            .upload_f32(&[spec.output, inputs], &weights)
                            .map_err(|err| backend_error("upload connected weights", err))?;
                        let biases_gpu = self
                            .backend
                            .upload_f32(&[spec.output], &biases)
                            .map_err(|err| backend_error("upload connected biases", err))?;
                        let pre_activation = self
                            .backend
                            .connected_f32(
                                current_input,
                                &weights_gpu,
                                &biases_gpu,
                                layer.input_shape.batch,
                                inputs,
                                spec.output,
                                launch_config,
                            )
                            .map_err(|err| backend_error("connected", err))?;
                        activation_input_gpu = Some(pre_activation.clone());
                        self.backend
                            .activation_f32(&pre_activation, activation, launch_config)
                            .map_err(|err| backend_error("connected activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download connected input", err))?;
                        let (host_output, host_activation_input) = forward_connected_with_cache(
                            spec,
                            connected,
                            &host_input,
                            layer.output_shape,
                        );
                        if let Some(activation_input) = host_activation_input {
                            activation_input_gpu = Some(
                                self.backend
                                    .upload_f32(
                                        &tensor_backend_shape(layer.output_shape),
                                        &activation_input.data,
                                    )
                                    .map_err(|err| {
                                        backend_error("upload connected activation cache", err)
                                    })?,
                            );
                        }
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload connected output", err))?
                    }
                }
                (LayerKind::Reorg(spec), LayerParams::None) => {
                    if spec.stride == 0 {
                        return Err(Error::InvalidConfig(
                            "reorg stride cannot be zero".to_owned(),
                        ));
                    }
                    self.backend
                        .reorg_f32(
                            current_input,
                            &tensor_backend_shape(layer.input_shape),
                            &tensor_backend_shape(layer.output_shape),
                            spec.stride,
                            spec.reverse,
                            launch_config,
                        )
                        .map_err(|err| backend_error("reorg", err))?
                }
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source =
                        gpu_layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let source_channels = self.network.layers[source_index].output_shape.channels;
                    self.backend
                        .route_concat_f32(
                            &[source],
                            &[spec.start],
                            &[spec.end - spec.start],
                            &[source_channels],
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("channel_slice", err))?
                }
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => self
                    .backend
                    .channel_shuffle_f32(
                        current_input,
                        &tensor_backend_shape(layer.input_shape),
                        spec.groups,
                        false,
                        launch_config,
                    )
                    .map_err(|err| backend_error("channel_shuffle", err))?,
                (LayerKind::Softmax { groups }, LayerParams::None) => self
                    .backend
                    .softmax_f32(
                        current_input,
                        &tensor_backend_shape(layer.output_shape),
                        softmax_group_size(layer.input_shape, *groups)?,
                        launch_config,
                    )
                    .map_err(|err| backend_error("softmax", err))?,
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_layer =
                        self.network
                            .layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let source_gpu =
                        gpu_layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                        if source_layer.output_shape != layer.output_shape {
                            return Err(Error::InvalidConfig(format!(
                                "scale_channels source/output mismatch: {} vs {}",
                                source_layer.output_shape, layer.output_shape
                            )));
                        }
                        let pre_activation = self
                            .backend
                            .scale_channels_f32(
                                current_input,
                                source_gpu,
                                &tensor_backend_shape(layer.output_shape),
                                spec.scale_wh,
                                launch_config,
                            )
                            .map_err(|err| backend_error("scale_channels", err))?;
                        activation_input_gpu = Some(pre_activation.clone());
                        self.backend
                            .activation_f32(&pre_activation, activation, launch_config)
                            .map_err(|err| backend_error("scale_channels activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download scale_channels input", err))?;
                        let host_source = self
                            .download_tensor(source_gpu, source_layer.output_shape)
                            .map_err(|err| backend_error("download scale_channels source", err))?;
                        let (host_output, host_activation_input) = forward_scale_channels(
                            spec,
                            &host_input,
                            &host_source,
                            layer.output_shape,
                        )?;
                        if let Some(activation_input) = host_activation_input {
                            activation_input_gpu = Some(
                                self.backend
                                    .upload_f32(
                                        &tensor_backend_shape(layer.output_shape),
                                        &activation_input.data,
                                    )
                                    .map_err(|err| {
                                        backend_error("upload scale_channels activation cache", err)
                                    })?,
                            );
                        }
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload scale_channels output", err))?
                    }
                }
                (LayerKind::Sam(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source_layer =
                        self.network
                            .layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    let source_gpu =
                        gpu_layer_outputs
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.from,
                            })?;
                    if layer.input_shape != layer.output_shape
                        || source_layer.output_shape != layer.output_shape
                    {
                        return Err(Error::InvalidConfig(format!(
                            "sam shape mismatch: input {}, source {}, output {}",
                            layer.input_shape, source_layer.output_shape, layer.output_shape
                        )));
                    }
                    if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                        let pre_activation = self
                            .backend
                            .vector_mul_f32(
                                current_input,
                                source_gpu,
                                &tensor_backend_shape(layer.output_shape),
                                launch_config,
                            )
                            .map_err(|err| backend_error("sam multiply", err))?;
                        activation_input_gpu = Some(pre_activation.clone());
                        self.backend
                            .activation_f32(&pre_activation, activation, launch_config)
                            .map_err(|err| backend_error("sam activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download sam input", err))?;
                        let host_source = self
                            .download_tensor(source_gpu, source_layer.output_shape)
                            .map_err(|err| backend_error("download sam source", err))?;
                        let (host_output, host_activation_input) =
                            forward_sam(spec, &host_input, &host_source, layer.output_shape)?;
                        if let Some(activation_input) = host_activation_input {
                            activation_input_gpu = Some(
                                self.backend
                                    .upload_f32(
                                        &tensor_backend_shape(layer.output_shape),
                                        &activation_input.data,
                                    )
                                    .map_err(|err| {
                                        backend_error("upload sam activation cache", err)
                                    })?,
                            );
                        }
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload sam output", err))?
                    }
                }
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(params)) => {
                    if spec.activation.to_cuda_f32_kind().is_some() {
                        let weights = self
                            .backend
                            .upload_f32(
                                &[
                                    layer.input_shape.channels,
                                    spec.filters,
                                    spec.size,
                                    spec.size,
                                ],
                                &params.weights,
                            )
                            .map_err(|err| backend_error("upload deconvolutional weights", err))?;
                        let raw = self
                            .backend
                            .deconvolutional_f32(
                                current_input,
                                &weights,
                                &tensor_backend_shape(layer.output_shape),
                                layer.input_shape.channels,
                                layer.input_shape.height,
                                layer.input_shape.width,
                                spec.filters,
                                spec.size,
                                spec.stride,
                                launch_config,
                            )
                            .map_err(|err| backend_error("GPU deconvolution", err))?;
                        let bias = self
                            .backend
                            .upload_f32(&[spec.filters], &params.biases)
                            .map_err(|err| backend_error("upload deconvolutional biases", err))?;
                        let biased = self
                            .backend
                            .bias_add_f32(
                                &raw,
                                &bias,
                                spec.filters,
                                layer.output_shape.height * layer.output_shape.width,
                                launch_config,
                            )
                            .map_err(|err| backend_error("deconvolution bias add", err))?;
                        activation_input_gpu = Some(biased.clone());
                        self.backend
                            .activation_f32(
                                &biased,
                                gpu_activation_kind(layer.index, &spec.activation)?,
                                launch_config,
                            )
                            .map_err(|err| backend_error("deconvolution activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download deconvolutional input", err))?;
                        let pre = crate::deconvolutional::deconv_forward(
                            &host_input,
                            &params.weights,
                            &params.biases,
                            spec.size,
                            spec.stride,
                            layer.output_shape,
                        );
                        activation_input_gpu = Some(
                            self.backend
                                .upload_f32(&tensor_backend_shape(layer.output_shape), &pre.data)
                                .map_err(|err| {
                                    backend_error("upload deconvolutional activation cache", err)
                                })?,
                        );
                        let activated: Vec<f32> = pre
                            .data
                            .iter()
                            .map(|value| activate(spec.activation.clone(), *value))
                            .collect();
                        self.backend
                            .upload_f32(&tensor_backend_shape(layer.output_shape), &activated)
                            .map_err(|err| backend_error("upload deconvolutional output", err))?
                    }
                }
                (LayerKind::Rnn(spec), LayerParams::Rnn(recurrent)) => {
                    if spec.activation.to_cuda_f32_kind().is_some()
                        && (spec.logistic != 2 || Activation::Loggy.to_cuda_f32_kind().is_some())
                    {
                        let (output, cache, device_cache) = forward_rnn_layer_device(
                            &self.backend,
                            current_input,
                            layer.input_shape,
                            spec,
                            recurrent,
                            layer.output_shape,
                            layer.index,
                            host_copies,
                            launch_config,
                        )?;
                        recurrent_caches.push(cache.map(RecurrentForwardCache::Rnn));
                        gpu_rnn_cache = Some(device_cache);
                        output
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download RNN input", err))?;
                        let (host_output, cache) =
                            forward_rnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
                        recurrent_caches.push(Some(RecurrentForwardCache::Rnn(cache)));
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload RNN output", err))?
                    }
                }
                (LayerKind::Lstm(spec), LayerParams::Lstm(recurrent)) => {
                    let (output, cache, device_cache) = forward_lstm_layer_device(
                        &self.backend,
                        spec,
                        recurrent,
                        current_input,
                        layer.input_shape,
                        layer.output_shape,
                        layer.index,
                        host_copies,
                        launch_config,
                    )?;
                    recurrent_caches.push(cache.map(RecurrentForwardCache::Lstm));
                    gpu_lstm_cache = Some(device_cache);
                    output
                }
                (LayerKind::Crnn(spec), LayerParams::Crnn(recurrent)) => {
                    if spec.activation.to_cuda_f32_kind().is_some() {
                        let (output, cache, device_cache) = forward_crnn_layer_device(
                            &self.backend,
                            spec,
                            recurrent,
                            current_input,
                            layer.input_shape,
                            layer.output_shape,
                            layer.index,
                            host_copies,
                            launch_config,
                        )?;
                        recurrent_caches.push(cache.map(RecurrentForwardCache::Crnn));
                        gpu_crnn_cache = Some(device_cache);
                        output
                    } else {
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download CRNN input", err))?;
                        let (host_output, cache) =
                            forward_crnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
                        recurrent_caches.push(Some(RecurrentForwardCache::Crnn(cache)));
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload CRNN output", err))?
                    }
                }
                (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(eml)) => {
                    let (input_output, input_batch_norm) = self.forward_eml_branch_gpu(
                        layer.index,
                        0,
                        current_input,
                        layer.input_shape,
                        layer.output_shape,
                        &spec.branch,
                        &eml.input,
                        batch_norm_training,
                        launch_config,
                    )?;
                    let (self_output, self_batch_norm) = self.forward_eml_branch_gpu(
                        layer.index,
                        1,
                        current_input,
                        layer.input_shape,
                        layer.output_shape,
                        &spec.branch,
                        &eml.self_params,
                        batch_norm_training,
                        launch_config,
                    )?;

                    let use_residual = spec.residual
                        && layer.input_shape.element_count() == layer.output_shape.element_count();
                    let residual = use_residual.then_some(current_input);
                    gpu_eml_cache = Some(GpuEmlBackwardCache {
                        input_branch_output: input_output.clone(),
                        self_branch_output: self_output.clone(),
                        input_batch_norm,
                        self_batch_norm,
                    });
                    let output = self
                        .backend
                        .eml_convolutional_f32(
                            &input_output,
                            &self_output,
                            residual,
                            &tensor_backend_shape(layer.output_shape),
                            spec.eml_clamp,
                            spec.eml_eps,
                            spec.eml_scale,
                            launch_config,
                        )
                        .map_err(|err| backend_error("EML branch combination", err))?;

                    if host_copies {
                        // The CPU backward fallback consumes the branch outputs,
                        // but the public forward still runs both branches and
                        // their combine on the selected tensor backend.
                        let host_input = self
                            .download_tensor(current_input, layer.input_shape)
                            .map_err(|err| backend_error("download EML input for fallback", err))?;
                        let (input_branch_output, _) = forward_convolutional_with_cache(
                            &spec.branch,
                            &eml.input,
                            &host_input,
                            layer.output_shape,
                        );
                        let (self_branch_output, _) = forward_convolutional_with_cache(
                            &spec.branch,
                            &eml.self_params,
                            &host_input,
                            layer.output_shape,
                        );
                        eml_cache = Some(EmlForwardCache {
                            input_branch_output,
                            self_branch_output,
                        });
                    }
                    output
                }
                (LayerKind::Moe(spec), LayerParams::Moe(moe)) => {
                    let host_input = if host_copies {
                        Some(
                            self.download_tensor(current_input, layer.input_shape)
                                .map_err(|err| backend_error("download MoE router input", err))?,
                        )
                    } else {
                        None
                    };
                    let batch = layer.input_shape.batch;
                    let router_weights = self
                        .backend
                        .upload_f32(
                            &[spec.experts, layer.input_shape.channels],
                            &moe.router_weights,
                        )
                        .map_err(|err| backend_error("upload MoE router weights", err))?;
                    let router_biases = self
                        .backend
                        .upload_f32(&[spec.experts], &moe.router_biases)
                        .map_err(|err| backend_error("upload MoE router biases", err))?;
                    let router_gates = self
                        .backend
                        .moe_router_gates_f32(
                            current_input,
                            &router_weights,
                            &router_biases,
                            batch,
                            layer.input_shape.channels,
                            layer.input_shape.width * layer.input_shape.height,
                            spec.experts,
                            spec.top_k,
                            crate::moe_conv::router_temperature(spec),
                            launch_config,
                        )
                        .map_err(|err| backend_error("compute MoE router gates", err))?;
                    let host_experts = if host_copies {
                        Some(
                            self.backend
                                .download_f32(&router_gates)
                                .map_err(|err| backend_error("download MoE router gates", err))?,
                        )
                    } else {
                        None
                    };
                    let pooled_gpu = self
                        .backend
                        .moe_router_pool_f32(
                            current_input,
                            batch,
                            layer.input_shape.channels,
                            layer.input_shape.width * layer.input_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("pool MoE router input", err))?;
                    let per_image = layer.output_shape.element_count();
                    let mut blended_gpu = None;
                    let mut host_cache = host_copies.then(|| MoeForwardCache {
                        gates: host_experts
                            .as_ref()
                            .expect("host cache requires host gates")
                            .clone(),
                        expert_outputs: Vec::with_capacity(spec.experts),
                        expert_activation_inputs: Vec::with_capacity(spec.experts),
                        pooled: crate::moe_conv::pool_for_router(
                            &host_input
                                .as_ref()
                                .expect("host cache requires host input")
                                .data,
                            batch,
                            layer.input_shape.channels,
                            layer.input_shape.width * layer.input_shape.height,
                        ),
                    });
                    let mut expert_outputs_gpu = Vec::with_capacity(spec.experts);
                    let mut expert_activation_inputs_gpu = Vec::with_capacity(spec.experts);
                    for (slot, expert) in moe.experts.iter().enumerate() {
                        let cached = self.cached_convolution_params_at(
                            layer.index,
                            slot,
                            layer.input_shape.channels,
                            &spec.expert,
                            expert,
                        )?;
                        let raw_output = self
                            .backend
                            .conv2d_f32(
                                current_input,
                                &cached.weights,
                                &tensor_backend_shape(layer.output_shape),
                                layer.input_shape.channels,
                                layer.input_shape.height,
                                layer.input_shape.width,
                                spec.expert.filters,
                                spec.expert.size,
                                spec.expert.stride_x,
                                spec.expert.stride_y,
                                spec.expert.dilation,
                                spec.expert.padding,
                                spec.expert.groups.max(1),
                                launch_config,
                            )
                            .map_err(|err| backend_error("MoE expert convolution", err))?;
                        let biased = self
                            .backend
                            .bias_add_f32(
                                &raw_output,
                                &cached.biases,
                                spec.expert.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                launch_config,
                            )
                            .map_err(|err| backend_error("MoE expert bias", err))?;
                        let activation = gpu_activation_kind(layer.index, &spec.expert.activation)?;
                        let output_gpu = self
                            .backend
                            .activation_f32(&biased, activation, launch_config)
                            .map_err(|err| backend_error("MoE expert activation", err))?;
                        let sample_gates = self
                            .backend
                            .moe_gate_column_f32(
                                &router_gates,
                                batch,
                                spec.experts,
                                slot,
                                launch_config,
                            )
                            .map_err(|err| backend_error("select MoE expert gates", err))?;
                        // Treat each sample as one channel with `per_image`
                        // elements, applying its scalar gate to the full expert
                        // output without moving that output back to the host.
                        let weighted = self
                            .backend
                            .channel_scale_f32(
                                &output_gpu,
                                &sample_gates,
                                batch,
                                per_image,
                                launch_config,
                            )
                            .map_err(|err| backend_error("scale MoE expert output", err))?;
                        blended_gpu = Some(match blended_gpu {
                            Some(accumulated) => self
                                .backend
                                .saxpy_f32(1.0, &weighted, &accumulated, launch_config)
                                .map_err(|err| backend_error("blend MoE expert outputs", err))?,
                            None => weighted,
                        });
                        expert_outputs_gpu.push(output_gpu.clone());
                        expert_activation_inputs_gpu.push(biased.clone());
                        if let Some(cache) = &mut host_cache {
                            // The CPU fallback needs host expert outputs and
                            // activation caches. Device backward uses the
                            // tensors retained above instead.
                            let (cpu_output, activation_input) = forward_convolutional_with_cache(
                                &spec.expert,
                                expert,
                                host_input.as_ref().expect("host cache requires host input"),
                                layer.output_shape,
                            );
                            cache.expert_outputs.push(cpu_output);
                            cache.expert_activation_inputs.push(activation_input);
                        }
                    }
                    moe_cache = host_cache;
                    gpu_moe_cache = Some(GpuMoeBackwardCache {
                        gates: router_gates,
                        pooled: pooled_gpu,
                        expert_outputs: expert_outputs_gpu,
                        expert_activation_inputs: expert_activation_inputs_gpu,
                    });
                    blended_gpu.ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "layer #{} ([moe_convolutional]) has no expert outputs",
                            layer.index
                        ))
                    })?
                }
                (LayerKind::Moe(_), _) => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ([moe_convolutional]) and parameter set are inconsistent",
                        layer.index
                    )));
                }
                (LayerKind::Gru(spec), LayerParams::Gru(params)) if !training => {
                    forward_gru_layer_device(
                        &self.backend,
                        current_input,
                        layer.input_shape,
                        spec,
                        params,
                        layer.output_shape,
                        layer.index,
                        launch_config,
                    )?
                }
                (LayerKind::Gru(_), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: layer.kind.name().to_owned(),
                    });
                }
                (LayerKind::Unsupported(_), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: layer.kind.name().to_owned(),
                    });
                }
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ({}) and parameter set are inconsistent",
                        layer.index,
                        layer.kind.name()
                    )));
                }
            };

            // C++ `try_fix_nan`: convolution outputs are fixed for NaN/Inf.
            let output_gpu = if self.network.net.try_fix_nan
                && matches!(layer.kind, LayerKind::Convolutional(_))
            {
                self.backend
                    .sanitize_nan_inf_f32(
                        &output_gpu,
                        &tensor_backend_shape(layer.output_shape),
                        NanInfMode::Fix,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU try_fix_nan output fix", err))?
            } else {
                output_gpu
            };
            let is_yolo = matches!(
                layer.kind,
                LayerKind::Yolo(_) | LayerKind::ObbYolo(_) | LayerKind::GaussianYolo(_)
            );
            let (output, activation_input) = if host_copies || is_yolo {
                let output = self
                    .download_tensor(&output_gpu, layer.output_shape)
                    .map_err(|err| backend_error("download layer output", err))?;
                let activation_input = match activation_input_gpu.as_ref() {
                    Some(tensor) if host_copies => Some(
                        self.download_tensor(tensor, layer.output_shape)
                            .map_err(|err| backend_error("download activation cache", err))?,
                    ),
                    _ => None,
                };
                (output, activation_input)
            } else {
                (
                    Tensor {
                        shape: layer.output_shape,
                        data: Vec::new(),
                    },
                    None,
                )
            };

            if is_yolo {
                yolo_heads.push(YoloHeadOutput {
                    layer_index: layer.index,
                    output: output.clone(),
                });
            }
            if is_modern_head(&layer.kind) {
                modern_heads.push(ModernHeadOutput {
                    layer_index: layer.index,
                    output: output.clone(),
                });
            }

            gpu_layer_outputs.push(output_gpu);
            gpu_layer_activation_inputs.push(activation_input_gpu);
            layer_outputs.push(output);
            layer_activation_inputs.push(activation_input);
            dropout_masks.push(dropout_mask);
            crop_caches.push(crop_cache);
            gpu_batch_norm_caches.push(batch_norm_cache);
            gpu_rnn_caches.push(gpu_rnn_cache);
            gpu_crnn_caches.push(gpu_crnn_cache);
            gpu_lstm_caches.push(gpu_lstm_cache);
            gpu_moe_caches.push(gpu_moe_cache);
            gpu_eml_caches.push(gpu_eml_cache);
            graph_conv_caches.push(graph_conv_cache);
            if recurrent_caches.len() < layer_outputs.len() {
                recurrent_caches.push(None);
            }
            moe_caches.push(moe_cache);
            eml_caches.push(eml_cache);
            dcnv4_caches.push(dcnv4_cache);
            gpu_dcnv4_caches.push(gpu_dcnv4_cache);
            deform_conv_caches.push(deform_conv_cache);
            deform_device_caches.push(deform_device_cache);
            vit_caches.push(vit_cache);
            swin_caches.push(swin_cache);
            tucker_attention_caches.push(tucker_attention_cache);
            mambavision_caches.push(mambavision_cache);
            kimi_kda_caches.push(kimi_kda_cache);
            hyperbolic_foveation_caches.push(hyperbolic_foveation_cache);
            foveation_film_caches.push(foveation_film_cache);
        }

        *self.deform_device_caches.borrow_mut() = deform_device_caches;
        Ok(GpuForwardState {
            input_gpu,
            gpu_layer_outputs,
            gpu_layer_activation_inputs,
            gpu_batch_norm_caches,
            gpu_rnn_caches,
            gpu_crnn_caches,
            gpu_lstm_caches,
            gpu_moe_caches,
            gpu_eml_caches,
            gpu_dcnv4_caches,
            cpu_forward: ForwardResult {
                layer_outputs,
                layer_activation_inputs,
                dropout_masks,
                crop_caches,
                recurrent_caches,
                moe_caches,
                eml_caches,
                dcnv4_caches,
                deform_conv_caches,
                dsconv_caches: vec![None; self.network.layers.len()],
                graph_conv_caches,
                wmhf_caches: vec![None; self.network.layers.len()],
                clifford_caches: vec![None; self.network.layers.len()],
                recursive_block_caches: vec![None; self.network.layers.len()],
                hyperbolic_foveation_caches,
                foveation_film_caches,
                vit_caches,
                depth_anything_caches: vec![None; self.network.layers.len()],
                dpt_head_caches: vec![None; self.network.layers.len()],
                swin_caches,
                tucker_attention_caches,
                mambavision_caches,
                kimi_kda_caches,
                obb_refine_outputs: Vec::new(),
                yolo_heads,
                yolo_3d_heads: Vec::new(),
                modern_heads,
            },
        })
    }

}

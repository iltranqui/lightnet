use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
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
        // `[vit]` keeps its device backward cache only when that same native
        // backward will run; otherwise training uses the host cache.
        let vit_device_training = training && deform_device;
        // `[swin]` uses the same rule for its device backward cache.
        let swin_device_training = training && deform_device;
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
        // `[wmhf]` slots are written by index in its arm: the host cache feeds
        // the CPU backward fallback, the device cache the native backward.
        let mut wmhf_caches: Vec<Option<WmhfRuntimeCache>> = vec![None; self.network.layers.len()];
        let mut wmhf_device_caches: Vec<
            Option<crate::layers::wmhf::gpu::DeviceCache<B::Tensor>>,
        > = vec![None; self.network.layers.len()];
        let mut clifford_caches = vec![None; self.network.layers.len()];
        let mut clifford_device_caches = vec![None; self.network.layers.len()];
        let mut vit_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut swin_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut tucker_attention_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut mambavision_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut kimi_kda_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut dpt_head_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut dpt_head_caches = Vec::with_capacity(self.network.layers.len());
        let mut depth_anything_caches = Vec::with_capacity(self.network.layers.len());
        let mut depth_anything_device_caches = Vec::with_capacity(self.network.layers.len());
        let mut recursive_block_caches = Vec::with_capacity(self.network.layers.len());
        let mut recursive_block_device_caches = Vec::with_capacity(self.network.layers.len());
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
            let mut vit_device_cache = None;
            let mut swin_device_cache = None;
            let mut swin_cache = None;
            let mut tucker_attention_cache = None;
            let mut tucker_attention_device_cache = None;
            let mut mambavision_cache = None;
            let mut mambavision_device_cache = None;
            let mut kimi_kda_cache = None;
            let mut kimi_kda_device_cache = None;
            let mut dpt_head_device_cache = None;
            let mut dpt_head_cache = None;
            let mut depth_anything_cache = None;
            let mut depth_anything_device_cache = None;
            let mut recursive_block_cache = None;
            let mut recursive_block_device_cache = None;
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
                (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(params)) => {
                    if (!training || deform_device) && crate::layers::recursive_block::gpu::supported(spec) {
                        let (output, cache) = self.forward_recursive_block_gpu(layer, spec, params,
                            current_input, launch_config, training)?;
                        recursive_block_device_cache = cache;
                        output
                    } else {
                        let input = self.download_tensor(current_input, layer.input_shape)
                            .map_err(|e| backend_error("download recursive fallback input", e))?;
                        let (output, cache) = forward_recursive_block(&self.network, self.train_iteration,
                            layer, spec, params, &input, training)?;
                        recursive_block_cache = Some(cache);
                        self.backend.upload_f32(&tensor_backend_shape(output.shape), &output.data)
                            .map_err(|e| backend_error("upload recursive fallback output", e))?
                    }
                }
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
                    let (output, cache) = self.forward_foveation_film_gpu(
                        layer,
                        spec,
                        params,
                        &hyperbolic_foveation_caches,
                        current_input,
                        launch_config,
                    )?;
                    foveation_film_cache = Some(cache);
                    output
                }
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let (output, activation_input, cache) = self.forward_convolutional_gpu(
                        layer,
                        spec,
                        current_input,
                        batch_norm_training,
                        launch_config,
                    )?;
                    activation_input_gpu = Some(activation_input);
                    batch_norm_cache = cache;
                    output
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    self.forward_maxpool_gpu(layer, spec, current_input, launch_config)?
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    self.forward_local_avgpool_gpu(layer, spec, current_input, launch_config)?
                }
                (LayerKind::AvgPool, LayerParams::None) => {
                    self.forward_avgpool_gpu(layer, current_input, launch_config)?
                }
                (LayerKind::Crop(spec), LayerParams::None) => {
                    let (output, cache) = self.forward_crop_gpu(
                        layer,
                        spec,
                        current_input,
                        training,
                        &mut rng_state,
                        launch_config,
                    )?;
                    crop_cache = Some(cache);
                    output
                }
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => {
                    let (output, cache) = self.forward_batch_norm_gpu(
                        layer,
                        params,
                        current_input,
                        training,
                        launch_config,
                    )?;
                    batch_norm_cache = cache;
                    output
                }
                (LayerKind::Route(spec), LayerParams::None) => {
                    self.forward_route_gpu(layer, spec, &gpu_layer_outputs, launch_config)?
                }
                (LayerKind::LayaMarkerGather(spec), LayerParams::None) => self
                    .forward_laya_marker_gather_gpu(
                        layer,
                        spec,
                        &gpu_layer_outputs,
                        launch_config,
                    )?,
                (LayerKind::LayaActHead(spec), LayerParams::LayaActHead(weights)) => self
                    .forward_laya_act_head_gpu(
                        layer,
                        spec,
                        weights,
                        &gpu_layer_outputs,
                        launch_config,
                    )?,
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => {
                    let (output, accumulated) = self.forward_weighted_shortcut_gpu(
                        layer,
                        spec,
                        params,
                        current_input,
                        &gpu_layer_outputs,
                        launch_config,
                    )?;
                    activation_input_gpu = Some(accumulated);
                    output
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let (output, accumulated) = self.forward_shortcut_gpu(
                        layer,
                        spec,
                        current_input,
                        &gpu_layer_outputs,
                        launch_config,
                    )?;
                    activation_input_gpu = Some(accumulated);
                    output
                }
                (LayerKind::Upsample(spec), LayerParams::None) => {
                    self.forward_upsample_gpu(layer, spec, current_input, launch_config)?
                }
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
                    let forward = if graph_conv_host_fallback {
                        self.forward_graph_conv_host_gpu(
                            layer,
                            spec,
                            params,
                            input,
                            &layer_outputs,
                            training,
                        )?
                    } else {
                        self.forward_graph_conv_device_gpu(
                            layer,
                            spec,
                            params,
                            current_input,
                            training,
                            launch_config,
                        )?
                    };
                    graph_conv_cache = forward.host_cache;
                    batch_norm_cache = forward.batch_norm_cache;
                    activation_input_gpu = Some(forward.activation_input);
                    forward.output
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
                    self.forward_dsconv_gpu(layer, spec, params, current_input, launch_config)?
                }
                (LayerKind::DeformConv(spec), LayerParams::DeformConv(params)) => {
                    let forward = if deform_device {
                        self.forward_deform_conv_device_gpu(
                            layer,
                            spec,
                            params,
                            current_input,
                            launch_config,
                        )?
                    } else {
                        self.forward_deform_conv_host_gpu(layer, spec, params, current_input)?
                    };
                    activation_input_gpu = Some(forward.activation_input);
                    deform_device_cache = forward.device_cache;
                    deform_conv_cache = forward.host_cache;
                    forward.output
                }
                // Native when the network takes the native backward and the
                // activation has a device kernel; otherwise the host reference.
                (LayerKind::Wmhf(spec), LayerParams::Wmhf(params)) => {
                    let forward = self.forward_wmhf_gpu(
                        layer,
                        spec,
                        params,
                        current_input,
                        training,
                        deform_device,
                        launch_config,
                    )?;
                    activation_input_gpu = Some(forward.activation_input);
                    wmhf_caches[layer.index] = forward.host_cache;
                    wmhf_device_caches[layer.index] = forward.device_cache;
                    forward.output
                }
                (LayerKind::Clifford(spec), LayerParams::Clifford(params)) => {
                    let mask = if training && spec.drop_path > 0.0 {
                        let keep = 1.0 - spec.drop_path;
                        (0..layer.input_shape.batch)
                            .map(|_| {
                                if next_random_unit(&mut rng_state) < keep {
                                    1.0 / keep
                                } else {
                                    0.0
                                }
                            })
                            .collect::<Vec<_>>()
                    } else {
                        vec![1.0; layer.input_shape.batch]
                    };
                    let (output, device_cache, host_cache) = self.forward_clifford_gpu(
                        layer,
                        spec,
                        params,
                        current_input,
                        training,
                        deform_device,
                        &mask,
                        launch_config,
                    )?;
                    clifford_device_caches[layer.index] = device_cache;
                    clifford_caches[layer.index] = host_cache;
                    output
                }
                (LayerKind::Vit(spec), LayerParams::Vit(weights))
                    if (!training || vit_device_training)
                        && vit_gpu::supported(&vit_geometry(layer, spec)) =>
                {
                    let geometry = vit_geometry(layer, spec);
                    if vit_device_training {
                        // Training: the native backward reads this device cache.
                        let (output, cache) = vit_gpu::forward_training(
                            &self.backend,
                            &geometry,
                            weights,
                            current_input,
                            layer.output_shape,
                            launch_config,
                        )?;
                        vit_device_cache = Some(cache);
                        output
                    } else {
                        vit_gpu::forward(
                            &self.backend,
                            &geometry,
                            weights,
                            current_input,
                            layer.output_shape,
                            launch_config,
                        )?
                    }
                }
                (LayerKind::Vit(spec), LayerParams::Vit(weights)) => {
                    let (output, cache) = self.forward_vit_host(
                        layer,
                        &vit_geometry(layer, spec),
                        weights,
                        current_input,
                    )?;
                    vit_cache = Some(cache);
                    output
                }
                (LayerKind::Swin(spec), LayerParams::Swin(weights))
                    if (!training || swin_device_training)
                        && crate::layers::swin::gpu::supported(
                            &crate::layers::swin::gpu::geometry(layer, spec),
                        ) =>
                {
                    let geometry = crate::layers::swin::gpu::geometry(layer, spec);
                    if swin_device_training {
                        // Training: the native backward reads this device cache.
                        let (output, cache) = crate::layers::swin::gpu::forward_training(
                            &self.backend,
                            &geometry,
                            weights,
                            current_input,
                            layer.output_shape,
                            launch_config,
                        )?;
                        swin_device_cache = Some(cache);
                        output
                    } else {
                        crate::layers::swin::gpu::forward(
                            &self.backend,
                            &geometry,
                            weights,
                            current_input,
                            layer.output_shape,
                            launch_config,
                        )?
                    }
                }
                (LayerKind::Swin(spec), LayerParams::Swin(weights)) => {
                    let (output, cache) =
                        self.forward_swin_host(layer, spec, weights, current_input)?;
                    swin_cache = Some(cache);
                    output
                }
                (LayerKind::TuckerAttention(spec), LayerParams::TuckerAttention(weights))
                    if (!training || deform_device)
                        && crate::layers::tucker_attention::gpu::supported(
                            &crate::layers::tucker_attention::gpu::geometry(layer, spec)) =>
                {
                    let (output, cache) = self.forward_tucker_attention_gpu(
                        layer, spec, weights, current_input, launch_config, training)?;
                    tucker_attention_device_cache = cache;
                    output
                }
                (LayerKind::TuckerAttention(spec), LayerParams::TuckerAttention(weights)) => {
                    let (output, cache) =
                        self.forward_tucker_attention_host(layer, spec, weights, current_input)?;
                    tucker_attention_cache = Some(cache);
                    output
                }
                (LayerKind::MambaVision(spec), LayerParams::MambaVision(weights))
                    if (!training || deform_device)
                        && crate::layers::mambavision::gpu::supported(layer, spec) =>
                {
                    let (output, cache) = self.forward_mambavision_gpu(
                        layer, spec, weights, current_input, launch_config, training)?;
                    mambavision_device_cache = cache;
                    output
                }
                (LayerKind::MambaVision(spec), LayerParams::MambaVision(weights)) => {
                    let (output, cache) =
                        self.forward_mambavision_host(layer, spec, weights, current_input)?;
                    mambavision_cache = Some(cache);
                    output
                }
                (LayerKind::KimiKda(spec), LayerParams::KimiKda(weights))
                    if (!training || deform_device)
                        && crate::layers::kimi_kda::gpu::supported(layer, spec) =>
                {
                    let (output, cache) = self.forward_kimi_kda_gpu(
                        layer, spec, weights, current_input, launch_config, training)?;
                    kimi_kda_device_cache = cache;
                    output
                }
                (LayerKind::KimiKda(spec), LayerParams::KimiKda(weights)) => {
                    let (output, cache) =
                        self.forward_kimi_kda_host(layer, spec, weights, current_input)?;
                    kimi_kda_cache = Some(cache);
                    output
                }
                (LayerKind::DepthAnything(spec), LayerParams::DepthAnything(weights)) => {
                    if (!training || deform_device) && crate::layers::depth_anything::gpu::supported(layer, spec) {
                        let (output, cache) = self.forward_depth_anything_gpu(layer, spec, weights,
                            current_input, launch_config, training)?;
                        depth_anything_device_cache = cache;
                        output
                    } else {
                        let input = self.download_tensor(current_input, layer.input_shape)
                            .map_err(|e| backend_error("download DepthAnything fallback input", e))?;
                        let (output, cache) = crate::depth_anything::forward(&input.data, weights,
                            &spec.geometry(layer.input_shape)).map_err(Error::InvalidConfig)?;
                        depth_anything_cache = Some(cache);
                        self.backend.upload_f32(&tensor_backend_shape(layer.output_shape), &output)
                            .map_err(|e| backend_error("upload DepthAnything fallback output", e))?
                    }
                }
                (LayerKind::DptHead(spec), LayerParams::DptHead(weights)) => {
                    if !training || deform_device {
                        let (output, cache) = self.forward_dpt_head_gpu(
                            layer, spec, weights, current_input, launch_config, training)?;
                        dpt_head_device_cache = cache;
                        output
                    } else {
                        let input = self.download_tensor(current_input, layer.input_shape)
                            .map_err(|e| backend_error("download DPT fallback input", e))?;
                        let (output, cache) = crate::dpt_head::forward(&input.data, weights,
                            spec.geometry(layer.input_shape))?;
                        dpt_head_cache = Some(cache);
                        self.backend.upload_f32(&tensor_backend_shape(layer.output_shape), &output)
                            .map_err(|e| backend_error("upload DPT fallback output", e))?
                    }
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
                        lightnet_cuda::F32ActivationKind::Logistic,
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
                    let (output, activation_input) =
                        self.forward_local_gpu(layer, spec, local, current_input, launch_config)?;
                    activation_input_gpu = activation_input;
                    output
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    let (output, activation_input) = self.forward_connected_gpu(
                        layer,
                        spec,
                        connected,
                        current_input,
                        launch_config,
                    )?;
                    activation_input_gpu = activation_input;
                    output
                }
                (LayerKind::Reorg(spec), LayerParams::None) => {
                    self.forward_reorg_gpu(layer, spec, current_input, launch_config)?
                }
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    self.forward_channel_slice_gpu(layer, spec, &gpu_layer_outputs, launch_config)?
                }
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => {
                    self.forward_channel_shuffle_gpu(layer, spec, current_input, launch_config)?
                }
                (LayerKind::Softmax { groups }, LayerParams::None) => {
                    self.forward_softmax_gpu(layer, *groups, current_input, launch_config)?
                }
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let (output, activation_input) = self.forward_scale_channels_gpu(
                        layer,
                        spec,
                        current_input,
                        &gpu_layer_outputs,
                        launch_config,
                    )?;
                    if activation_input.is_some() {
                        activation_input_gpu = activation_input;
                    }
                    output
                }
                (LayerKind::Sam(spec), LayerParams::None) => {
                    let (output, activation_input) = self.forward_sam_gpu(
                        layer,
                        spec,
                        current_input,
                        &gpu_layer_outputs,
                        launch_config,
                    )?;
                    if activation_input.is_some() {
                        activation_input_gpu = activation_input;
                    }
                    output
                }
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(params)) => {
                    let (output, activation_input) = self.forward_deconvolutional_gpu(
                        layer,
                        spec,
                        params,
                        current_input,
                        launch_config,
                    )?;
                    activation_input_gpu = Some(activation_input);
                    output
                }
                (LayerKind::Rnn(spec), LayerParams::Rnn(recurrent)) => {
                    let (output, cache, device_cache) = self.forward_rnn_gpu(
                        layer,
                        spec,
                        recurrent,
                        current_input,
                        host_copies,
                        launch_config,
                    )?;
                    recurrent_caches.push(cache);
                    gpu_rnn_cache = device_cache;
                    output
                }
                (LayerKind::Lstm(spec), LayerParams::Lstm(recurrent)) => {
                    let (output, cache, device_cache) = self.forward_lstm_gpu(
                        layer,
                        spec,
                        recurrent,
                        current_input,
                        host_copies,
                        launch_config,
                    )?;
                    recurrent_caches.push(cache);
                    gpu_lstm_cache = Some(device_cache);
                    output
                }
                (LayerKind::Crnn(spec), LayerParams::Crnn(recurrent)) => {
                    let (output, cache, device_cache) = self.forward_crnn_gpu(
                        layer,
                        spec,
                        recurrent,
                        current_input,
                        host_copies,
                        launch_config,
                    )?;
                    recurrent_caches.push(cache);
                    gpu_crnn_cache = device_cache;
                    output
                }
                (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(eml)) => {
                    let (output, gpu_cache, host_cache) = self.forward_eml_convolutional_gpu(
                        layer,
                        spec,
                        eml,
                        current_input,
                        batch_norm_training,
                        host_copies,
                        launch_config,
                    )?;
                    gpu_eml_cache = Some(gpu_cache);
                    eml_cache = host_cache;
                    output
                }
                (LayerKind::Moe(spec), LayerParams::Moe(moe)) => {
                    let (output, gpu_cache, host_cache) = self.forward_moe_gpu(
                        layer,
                        spec,
                        moe,
                        current_input,
                        host_copies,
                        launch_config,
                    )?;
                    gpu_moe_cache = Some(gpu_cache);
                    moe_cache = host_cache;
                    output
                }
                (LayerKind::Moe(_), _) => {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ([moe_convolutional]) and parameter set are inconsistent",
                        layer.index
                    )));
                }
                (LayerKind::Gru(spec), LayerParams::Gru(params)) if !training => {
                    self.forward_gru_gpu(layer, spec, params, current_input, launch_config)?
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
            vit_device_caches.push(vit_device_cache);
            swin_device_caches.push(swin_device_cache);
            swin_caches.push(swin_cache);
            tucker_attention_caches.push(tucker_attention_cache);
            tucker_attention_device_caches.push(tucker_attention_device_cache);
            mambavision_caches.push(mambavision_cache);
            mambavision_device_caches.push(mambavision_device_cache);
            kimi_kda_caches.push(kimi_kda_cache);
            kimi_kda_device_caches.push(kimi_kda_device_cache);
            dpt_head_device_caches.push(dpt_head_device_cache);
            dpt_head_caches.push(dpt_head_cache);
            depth_anything_caches.push(depth_anything_cache);
            depth_anything_device_caches.push(depth_anything_device_cache);
            recursive_block_caches.push(recursive_block_cache);
            recursive_block_device_caches.push(recursive_block_device_cache);
            hyperbolic_foveation_caches.push(hyperbolic_foveation_cache);
            foveation_film_caches.push(foveation_film_cache);
        }

        *self.deform_device_caches.borrow_mut() = deform_device_caches;
        *self.clifford_device_caches.borrow_mut() = clifford_device_caches;
        *self.vit_device_caches.borrow_mut() = vit_device_caches;
        *self.wmhf_device_caches.borrow_mut() = wmhf_device_caches;
        *self.swin_device_caches.borrow_mut() = swin_device_caches;
        *self.tucker_attention_device_caches.borrow_mut() = tucker_attention_device_caches;
        *self.mambavision_device_caches.borrow_mut() = mambavision_device_caches;
        *self.kimi_kda_device_caches.borrow_mut() = kimi_kda_device_caches;
        *self.dpt_head_device_caches.borrow_mut() = dpt_head_device_caches;
        *self.depth_anything_device_caches.borrow_mut() = depth_anything_device_caches;
        *self.recursive_block_device_caches.borrow_mut() = recursive_block_device_caches;
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
                wmhf_caches,
                clifford_caches,
                recursive_block_caches,
                hyperbolic_foveation_caches,
                foveation_film_caches,
                vit_caches,
                depth_anything_caches,
                dpt_head_caches,
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

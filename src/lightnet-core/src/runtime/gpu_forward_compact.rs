use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub fn forward_compact(&self, input: &Tensor) -> Result<GpuForwardSummary> {
        // Compact f32 has its own dispatch coverage. The stricter shared
        // half-storage guard remains in forward_compact_f16.
        reject_gpu_stage1_cpu_layers(&self.network, "compact inference")?;
        if !self.resident_parameter_path_enabled()
            && self.resident_parameter_banks.borrow().iter().flatten().any(|s|s.dirty) {
            return Err(Error::InvalidConfig("synchronize deferred resident parameters before switching to CPU/hybrid execution".into()));
        }
        if self.network.layers.iter().any(|layer| matches!(layer.kind,
            LayerKind::HyperbolicFoveation(_) | LayerKind::FoveationFilm(_))) {
            reject_gpu_hyperbolic_foveation(&self.network, "compact inference")?;
        }
        if let Some(layer) = self
            .network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::ObbRefine(_)))
        {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[obb_refine] GPU inference is not implemented; use the CPU forward path"
                    .to_owned(),
            });
        }
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
            .find(|layer| matches!(layer.kind, LayerKind::DeformConv(_)))
        {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[deform_conv] compact GPU execution is unsupported; use f32 forward"
                    .to_owned(),
            });
        }
        if let Some(layer) = self.network.layers.iter().find(|layer| {
            matches!(
                &layer.kind,
                LayerKind::GraphConv(spec)
                    if graph_conv_spec_needs_compact_host(spec)
            )
        }) {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[graph_conv] compact GPU execution is unsupported; use f32 GPU forward"
                    .to_owned(),
            });
        }
        if let Some(layer) = self
            .network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::DsConv(_)))
        {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[dsconv] compact f16 GPU execution is unsupported; use f32 GPU execution"
                    .to_owned(),
            });
        }
        if self.network.layers.iter().any(|layer| {
            matches!(
                layer.kind,
                LayerKind::EmlConvolutional(_)
                    | LayerKind::Dcnv4(_)
                    | LayerKind::Contrastive(_)
                    | LayerKind::LayaEncoder(_)
                    | LayerKind::Rnn(_)
                    | LayerKind::Lstm(_)
                    | LayerKind::Crnn(_)
                    | LayerKind::Shortcut(crate::model::ShortcutSpec {
                        weights_type: crate::model::ShortcutWeightsType::PerFeature
                            | crate::model::ShortcutWeightsType::PerChannel,
                        ..
                    })
            )
        }) || self.network.layers.iter().any(|layer| {
            matches!(
                &layer.kind,
                LayerKind::ScaleChannels(spec)
                    if spec.activation.to_cuda_f32_kind().is_none()
            ) || matches!(
                &layer.kind,
                LayerKind::Connected(spec)
                    if spec.activation.to_cuda_f32_kind().is_none()
            )
        }) || self.network.layers.iter().any(|layer| match &layer.kind {
            LayerKind::RecursiveBlock(spec) => !crate::layers::recursive_block::gpu::supported(spec),
            LayerKind::Vit(spec) => !vit_gpu::supported(&vit_geometry(layer, spec)),
            LayerKind::Swin(spec) => !crate::layers::swin::gpu::supported(
                &crate::layers::swin::gpu::geometry(layer, spec)),
            LayerKind::TuckerAttention(spec) => !crate::layers::tucker_attention::gpu::supported(
                &crate::layers::tucker_attention::gpu::geometry(layer, spec)),
            LayerKind::MambaVision(spec) => !crate::layers::mambavision::gpu::supported(layer, spec),
            LayerKind::KimiKda(spec) => !crate::layers::kimi_kda::gpu::supported(layer, spec),
            LayerKind::DepthAnything(spec) => !crate::layers::depth_anything::gpu::supported(layer, spec),
            LayerKind::Wmhf(spec) => crate::layers::wmhf::gpu::needs_host(spec),
            LayerKind::Clifford(spec) => crate::layers::clifford::gpu::needs_host(spec),
            _ => false,
        }) {
            let state = self.forward_with_state(input)?;
            return Ok(GpuForwardSummary {
                final_output: state.cpu_forward.final_output().cloned(),
                yolo_heads: state.cpu_forward.yolo_heads,
                modern_heads: state.cpu_forward.modern_heads,
            });
        }

        let launch_config = ElementwiseLaunchConfig::default();
        let input_gpu = self
            .backend
            .upload_f32(&tensor_backend_shape(input.shape), &input.data)
            .map_err(|err| backend_error("upload input tensor", err))?;
        let mut gpu_layer_outputs = vec![None; self.network.layers.len()];
        let mut remaining_uses = gpu_layer_output_use_counts(&self.network)?;
        let mut final_output = None;
        let mut yolo_heads = Vec::new();
        let mut modern_heads = Vec::new();

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                Some(&input_gpu)
            } else if layer_uses_previous_input(&layer.kind) {
                Some(
                    gpu_layer_outputs
                        .get(layer.index - 1)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous GPU layer output for layer #{}",
                                layer.index
                            ))
                        })?,
                )
            } else {
                None
            };

            let output_gpu = match (&layer.kind, params) {
                (LayerKind::Gru(spec), LayerParams::Gru(params)) => self.forward_gru_gpu(
                    layer,spec,params,current_input.expect("GRU requires current input"),launch_config)?,
                (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(weights)) => self.forward_recursive_block_gpu(
                    layer, spec, weights, current_input.expect("recursive requires current input"), launch_config, false)?.0,
                (LayerKind::Vit(spec), LayerParams::Vit(weights)) => self.forward_vit_native_gpu(
                    layer, &vit_geometry(layer, spec), weights,
                    current_input.expect("ViT requires current input"), launch_config, false)?.0,
                (LayerKind::Swin(spec), LayerParams::Swin(weights)) => self.forward_swin_native_gpu(
                    layer, &crate::layers::swin::gpu::geometry(layer, spec), weights,
                    current_input.expect("Swin requires current input"), launch_config, false)?.0,
                (LayerKind::TuckerAttention(spec), LayerParams::TuckerAttention(weights)) => self.forward_tucker_attention_gpu(
                    layer, spec, weights, current_input.expect("Tucker attention requires current input"), launch_config, false)?.0,
                (LayerKind::MambaVision(spec), LayerParams::MambaVision(weights)) => self.forward_mambavision_gpu(
                    layer, spec, weights, current_input.expect("MambaVision requires current input"), launch_config, false)?.0,
                (LayerKind::KimiKda(spec), LayerParams::KimiKda(weights)) => self.forward_kimi_kda_gpu(
                    layer, spec, weights, current_input.expect("Kimi KDA requires current input"), launch_config, false)?.0,
                (LayerKind::DepthAnything(spec), LayerParams::DepthAnything(weights)) => self.forward_depth_anything_gpu(
                    layer, spec, weights, current_input.expect("DepthAnything requires current input"), launch_config, false)?.0,
                (LayerKind::DptHead(spec), LayerParams::DptHead(weights)) => self.forward_dpt_head_gpu(
                    layer, spec, weights, current_input.expect("DPT requires current input"), launch_config, false)?.0,
                (LayerKind::Wmhf(spec), LayerParams::Wmhf(weights)) => self.forward_wmhf_gpu(
                    layer, spec, weights, current_input.expect("WMHF requires current input"), false, true, launch_config)?.output,
                (LayerKind::Clifford(spec), LayerParams::Clifford(weights)) => self.forward_clifford_gpu(
                    layer, spec, weights, current_input.expect("Clifford requires current input"), false, true,
                    &vec![1.0;layer.input_shape.batch], launch_config)?.0,
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => self.forward_convolutional_compact_gpu(
                    layer,
                    spec,
                    current_input.expect("convolution requires current input"),
                    launch_config,
                )?,
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(params)) => self
                    .forward_deconvolutional_compact_gpu(
                        layer,
                        spec,
                        params,
                        current_input.expect("deconvolution requires current input"),
                        launch_config,
                    )?,
                (LayerKind::GraphConv(spec), LayerParams::GraphConv(params)) => self
                    .forward_graph_conv_compact_gpu(
                        layer,
                        spec,
                        params,
                        current_input.expect("GraphConv requires current input"),
                        launch_config,
                    )?,
                (LayerKind::MaxPool(spec), LayerParams::None) => self.forward_maxpool_gpu(
                    layer,
                    spec,
                    current_input.expect("maxpool requires current input"),
                    launch_config,
                )?,
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => self
                    .forward_local_avgpool_gpu(
                        layer,
                        spec,
                        current_input.expect("local_avgpool requires current input"),
                        launch_config,
                    )?,
                (LayerKind::AvgPool, LayerParams::None) => self.forward_avgpool_gpu(
                    layer,
                    current_input.expect("avgpool requires current input"),
                    launch_config,
                )?,
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => self
                    .forward_batchnorm_inference_gpu(
                        current_input.expect("batchnorm requires current input"),
                        layer.input_shape,
                        params,
                        launch_config,
                    )?,
                (LayerKind::Route(spec), LayerParams::None) => {
                    self.forward_route_compact_gpu(layer, spec, &gpu_layer_outputs, launch_config)?
                }
                (LayerKind::LayaMarkerGather(spec), LayerParams::None) => self
                    .forward_laya_marker_gather_compact_gpu(
                        layer,
                        spec,
                        &gpu_layer_outputs,
                        launch_config,
                    )?,
                (LayerKind::LayaHead(spec), LayerParams::LayaHead(weights)) => {
                    laya_head_gpu::forward(
                        &self.backend,
                        spec,
                        weights,
                        current_input.expect("LayaHead requires current input"),
                        layer.input_shape,
                        layer.output_shape,
                        launch_config,
                    )?
                }
                (LayerKind::LayaEncoder(spec), LayerParams::LayaEncoder(weights)) => {
                    laya_encoder_gpu::forward(
                        &self.backend,
                        spec,
                        weights,
                        current_input.expect("LayaEncoder requires current input"),
                        layer.input_shape,
                        layer.output_shape,
                        launch_config,
                    )?
                }
                (LayerKind::LayaScorer(spec), LayerParams::LayaScorer(weights)) => {
                    laya_scorer_gpu::forward(
                        &self.backend,
                        spec,
                        weights,
                        current_input.expect("LayaScorer requires current input"),
                        layer.input_shape,
                        layer.output_shape,
                        launch_config,
                    )?
                }
                (LayerKind::LayaActHead(spec), LayerParams::LayaActHead(weights)) => self
                    .forward_laya_act_head_compact_gpu(
                        layer,
                        spec,
                        weights,
                        &gpu_layer_outputs,
                        launch_config,
                    )?,
                (LayerKind::Shortcut(spec), LayerParams::None) => self
                    .forward_shortcut_compact_gpu(
                        layer,
                        spec,
                        &gpu_layer_outputs,
                        current_input.expect("shortcut requires current input"),
                        launch_config,
                    )?,
                (LayerKind::Upsample(spec), LayerParams::None) => self.forward_upsample_gpu(
                    layer,
                    spec,
                    current_input.expect("upsample requires current input"),
                    launch_config,
                )?,
                (LayerKind::Sam(spec), LayerParams::None) => self.forward_sam_compact_gpu(
                    layer,
                    spec,
                    &gpu_layer_outputs,
                    current_input.expect("sam requires current input"),
                    launch_config,
                )?,
                (LayerKind::LayaLoss(spec), LayerParams::None) => self
                    .backend
                    .laya_loss_forward_f32(
                        current_input.expect("LayaLoss requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        spec.max_options,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU LayaLoss compact forward", err))?,
                (LayerKind::JevCe(spec), LayerParams::None) => self
                    .backend
                    .jev_ce_forward_f32(
                        current_input.expect("JevCe requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        spec.groups,
                        spec.candidates,
                        spec.temperature,
                        spec.channel_major,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU JevCe compact forward", err))?,
                (LayerKind::Segmentation(spec), LayerParams::None) => self
                    .backend
                    .segmentation_forward_f32(
                        current_input.expect("segmentation requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        spec.classes,
                        spec.temperature,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU segmentation compact forward", err))?,
                (LayerKind::Region(spec), LayerParams::None) => self
                    .backend
                    .region_forward_f32(
                        current_input.expect("region requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        spec.num,
                        spec.coords + spec.classes + 1,
                        spec.classes,
                        spec.softmax,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU Region compact forward", err))?,
                (LayerKind::CenterNet(spec), LayerParams::None) => self
                    .backend
                    .centernet_forward_f32(
                        current_input.expect("CenterNet requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        spec.classes,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU CenterNet compact forward", err))?,
                (LayerKind::Yolox(spec), LayerParams::None)
                | (LayerKind::Ppyoloe(spec), LayerParams::None)
                | (LayerKind::Yolonas(spec), LayerParams::None) => self
                    .backend
                    .modern_yolo_forward_f32(
                        current_input.expect("modern-YOLO requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        spec.classes,
                        launch_config,
                    )
                    .map_err(|err| backend_error("modern-YOLO compact forward", err))?,
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => self
                    .forward_connected_compact_gpu(
                        layer,
                        spec,
                        connected,
                        current_input.expect("Connected requires current input"),
                        launch_config,
                    )?,
                (LayerKind::ScaleChannels(spec), LayerParams::None) => self
                    .forward_scale_channels_compact_gpu(
                        layer,
                        spec,
                        &gpu_layer_outputs,
                        current_input.expect("ScaleChannels requires current input"),
                        launch_config,
                    )?,
                (LayerKind::Reorg(spec), LayerParams::None) => self.forward_reorg_compact_gpu(
                    layer,
                    spec,
                    current_input.expect("Reorg requires current input"),
                    launch_config,
                )?,
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => self
                    .backend
                    .channel_shuffle_f32(
                        current_input.expect("ChannelShuffle requires current input"),
                        &tensor_backend_shape(layer.input_shape),
                        spec.groups,
                        false,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU ChannelShuffle compact forward", err))?,
                (LayerKind::ChannelSlice(spec), LayerParams::None) => self
                    .forward_channel_slice_compact_gpu(layer, spec, &gpu_layer_outputs, launch_config)?,
                (LayerKind::Logistic, LayerParams::None) => self
                    .backend
                    .activation_f32(
                        current_input.expect("Logistic requires current input"),
                        lightnet_cuda::F32ActivationKind::Logistic,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU Logistic compact forward", err))?,
                (LayerKind::Softmax { groups, spatial: false }, LayerParams::None) => self
                    .backend
                    .softmax_f32(
                        current_input.expect("Softmax requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        softmax_group_size(layer.input_shape, *groups)?,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU Softmax compact forward", err))?,
                (LayerKind::Softmax { spatial: true, .. }, LayerParams::None) => {
                    self.backend.spatial_softmax_f32(current_input.expect("softmax input"), &tensor_backend_shape(layer.output_shape), launch_config)
                        .map_err(|err| backend_error("GPU spatial softmax compact forward", err))?
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::ObbYolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Dropout { .. }, LayerParams::None)
                | (LayerKind::OmdLoss(_), LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None)
                | (LayerKind::Contrastive(_), LayerParams::None) => current_input
                    .expect("staging layer requires current input")
                    .clone(),
                (LayerKind::Connected(_), _)
                | (LayerKind::Reorg(_), _)
                | (LayerKind::Rnn(_), _)
                | (LayerKind::Lstm(_), _)
                | (LayerKind::Crnn(_), _)
                | (LayerKind::Unsupported(_), _) => {
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

            let should_download_final = layer.index + 1 == self.network.layers.len();
            let should_collect_yolo = matches!(
                layer.kind,
                LayerKind::Yolo(_) | LayerKind::ObbYolo(_) | LayerKind::GaussianYolo(_)
            );
            // The modern anchor-free heads need the same host round trip so
            // `detect.rs` can decode them.
            let should_collect_modern = is_modern_head(&layer.kind);
            let downloaded =
                if should_download_final || should_collect_yolo || should_collect_modern {
                    Some(
                        self.download_tensor(&output_gpu, layer.output_shape)
                            .map_err(|err| backend_error("download layer output", err))?,
                    )
                } else {
                    None
                };

            if should_collect_yolo {
                yolo_heads.push(YoloHeadOutput {
                    layer_index: layer.index,
                    output: downloaded
                        .as_ref()
                        .expect("yolo output should have been downloaded")
                        .clone(),
                });
            }
            if should_collect_modern {
                modern_heads.push(ModernHeadOutput {
                    layer_index: layer.index,
                    output: downloaded
                        .as_ref()
                        .expect("modern head output should have been downloaded")
                        .clone(),
                });
            }
            if should_download_final {
                final_output = downloaded;
            }

            gpu_layer_outputs[layer.index] = Some(output_gpu);
            release_gpu_sources_for_layer(
                layer.index,
                &layer.kind,
                &mut remaining_uses,
                &mut gpu_layer_outputs,
            )?;

            if remaining_uses[layer.index] == 0 {
                gpu_layer_outputs[layer.index] = None;
            }
        }

        Ok(GpuForwardSummary {
            final_output,
            yolo_heads,
            modern_heads,
        })
    }

}

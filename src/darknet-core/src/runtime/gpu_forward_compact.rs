use super::*;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub fn forward_compact(&self, input: &Tensor) -> Result<GpuForwardSummary> {
        reject_gpu_hyperbolic_foveation(&self.network, "compact inference")?;
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
        if let Some(layer) = self
            .network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::Wmhf(_)))
        {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "[wmhf] compact GPU execution is not implemented in the Stage-1 CPU port; use the CPU forward path".to_owned(),
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
                name: "[recursive_block] compact GPU execution is not implemented in the Stage-1 CPU port; use the CPU forward path"
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
                name: "[clifford] compact GPU execution is not implemented in the Stage-1 CPU port; use the CPU forward path".to_owned(),
            });
        }
        if self.network.layers.iter().any(|layer| {
            matches!(
                layer.kind,
                LayerKind::Deconvolutional(_)
                    | LayerKind::EmlConvolutional(_)
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
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    let owner = resolve_convolutional_owner(&self.network, layer.index)?;
                    let cached = self.cached_convolution_params(
                        owner,
                        layer.input_shape.channels,
                        spec,
                        conv,
                    )?;
                    let conv_output = self
                        .backend
                        .conv2d_f32(
                            current_input.expect("convolution requires current input"),
                            &cached.weights,
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
                    if let Some(normalization) = spec.activation.to_channel_normalization() {
                        let biased = self
                            .backend
                            .bias_add_f32(
                                &conv_output,
                                &cached.biases,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                launch_config,
                            )
                            .map_err(|err| backend_error("convolution bias add", err))?;
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
                            .bias_activation_f32(
                                &conv_output,
                                &cached.biases,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                activation,
                                launch_config,
                            )
                            .map_err(|err| backend_error("convolution bias + activation", err))?
                    }
                }
                (LayerKind::GraphConv(spec), LayerParams::GraphConv(params)) => {
                    let input = current_input.expect("GraphConv requires current input");
                    let groups = spec.groups.max(1);
                    let geometry = graph_conv_aggregation_geometry(
                        spec,
                        layer.input_shape,
                        layer.output_shape,
                    );
                    let aggregate = if spec.edge_mode == 0 {
                        self.backend
                            .graph_conv_aggregate_f32(input, geometry, launch_config)
                            .map_err(|err| backend_error("compact GraphConv aggregation", err))?
                    } else {
                        let edge_kernel_values =
                            params.edge_kernel.as_deref().ok_or_else(|| {
                                Error::InvalidConfig(
                                    "GraphConv learned edge kernel is missing".to_owned(),
                                )
                            })?;
                        let edge_bias_values = params.edge_biases.as_deref().ok_or_else(|| {
                            Error::InvalidConfig(
                                "GraphConv learned edge biases are missing".to_owned(),
                            )
                        })?;
                        let edge_kernel = self
                            .backend
                            .upload_f32(&[edge_kernel_values.len()], edge_kernel_values)
                            .map_err(|err| backend_error("compact GraphConv edge kernel", err))?;
                        let edge_biases = self
                            .backend
                            .upload_f32(&[edge_bias_values.len()], edge_bias_values)
                            .map_err(|err| backend_error("compact GraphConv edge biases", err))?;
                        self.backend
                            .graph_conv_learned_aggregate_f32(
                                input,
                                &edge_kernel,
                                &edge_biases,
                                geometry,
                                groups,
                                launch_config,
                            )
                            .map_err(|err| {
                                backend_error("compact GraphConv learned aggregation", err)
                            })?
                    };
                    let weight_shape = [spec.filters, layer.input_shape.channels / groups, 1, 1];
                    let weights = self
                        .backend
                        .upload_f32(&weight_shape, &params.projection.weights)
                        .map_err(|err| {
                            backend_error("compact GraphConv projection weights", err)
                        })?;
                    let mut raw = self
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
                        .map_err(|err| backend_error("compact GraphConv projection", err))?;
                    if spec.use_self {
                        let reference = self
                            .backend
                            .graph_conv_reference_f32(input, geometry, launch_config)
                            .map_err(|err| {
                                backend_error("compact GraphConv reference gather", err)
                            })?;
                        let self_values = params.self_weights.as_deref().ok_or_else(|| {
                            Error::InvalidConfig("GraphConv self weights are missing".to_owned())
                        })?;
                        let self_weights = self
                            .backend
                            .upload_f32(&weight_shape, self_values)
                            .map_err(|err| backend_error("compact GraphConv self weights", err))?;
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
                            .map_err(|err| {
                                backend_error("compact GraphConv self projection", err)
                            })?;
                        raw = self
                            .backend
                            .shortcut_add_f32(&raw, &self_raw, launch_config)
                            .map_err(|err| {
                                backend_error("compact GraphConv projection sum", err)
                            })?;
                    }
                    let biased = if spec.batch_normalize {
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
                            .map_err(|err| backend_error("compact GraphConv biases", err))?;
                        self.backend
                            .bias_add_f32(
                                &raw,
                                &biases,
                                spec.filters,
                                layer.output_shape.width * layer.output_shape.height,
                                launch_config,
                            )
                            .map_err(|err| backend_error("compact GraphConv bias", err))?
                    };
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
                                backend_error("compact GraphConv channel normalization", err)
                            })?
                    } else {
                        self.backend
                            .activation_f32(
                                &biased,
                                gpu_activation_kind(layer.index, &spec.activation)?,
                                launch_config,
                            )
                            .map_err(|err| backend_error("compact GraphConv activation", err))?
                    }
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    let maxpool_input = current_input.expect("maxpool requires current input");
                    if spec.depth {
                        self.backend.maxpool_depth_f32(
                            maxpool_input,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            launch_config,
                        )
                    } else {
                        self.backend.maxpool_f32(
                            maxpool_input,
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
                    .map_err(|err| backend_error("maxpool", err))?
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "local_avgpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .local_avgpool_f32(
                            current_input.expect("local_avgpool requires current input"),
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
                        current_input.expect("avgpool requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        launch_config,
                    )
                    .map_err(|err| backend_error("avgpool", err))?,
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => self
                    .forward_batchnorm_inference_gpu(
                        current_input.expect("batchnorm requires current input"),
                        layer.input_shape,
                        params,
                        launch_config,
                    )?,
                (LayerKind::Route(spec), LayerParams::None) => {
                    let mut route_inputs = Vec::with_capacity(spec.layers.len());
                    let mut route_channel_offsets = Vec::with_capacity(spec.layers.len());
                    let mut route_channels = Vec::with_capacity(spec.layers.len());
                    let mut route_total_channels = Vec::with_capacity(spec.layers.len());
                    for raw_index in &spec.layers {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source = gpu_layer_outputs
                            .get(source_index)
                            .and_then(Option::as_ref)
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
                    let a_source = gpu_layer_outputs
                        .get(a_index)
                        .and_then(Option::as_ref)
                        .ok_or(Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[0],
                        })?;
                    let b_source = gpu_layer_outputs
                        .get(b_index)
                        .and_then(Option::as_ref)
                        .ok_or(Error::InvalidRoute {
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
                    let head_source = gpu_layer_outputs
                        .get(head_index)
                        .and_then(Option::as_ref)
                        .ok_or(Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[0],
                        })?;
                    let scorer_source = gpu_layer_outputs
                        .get(scorer_index)
                        .and_then(Option::as_ref)
                        .ok_or(Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[1],
                        })?;
                    let marker_source = gpu_layer_outputs
                        .get(marker_index)
                        .and_then(Option::as_ref)
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
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let mut accumulated = current_input
                        .expect("shortcut requires current input")
                        .clone();
                    for raw_index in &spec.from {
                        let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                        let source = gpu_layer_outputs
                            .get(source_index)
                            .and_then(Option::as_ref)
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
                    self.backend
                        .activation_f32(&accumulated, activation, launch_config)
                        .map_err(|err| backend_error("shortcut activation", err))?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => self
                    .backend
                    .upsample_f32(
                        current_input.expect("upsample requires current input"),
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
                    let source_gpu = gpu_layer_outputs
                        .get(source_index)
                        .and_then(Option::as_ref)
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
                    let input_gpu = current_input.expect("sam requires current input");
                    if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                        let pre_activation = self
                            .backend
                            .vector_mul_f32(
                                input_gpu,
                                source_gpu,
                                &tensor_backend_shape(layer.output_shape),
                                launch_config,
                            )
                            .map_err(|err| backend_error("sam multiply", err))?;
                        self.backend
                            .activation_f32(&pre_activation, activation, launch_config)
                            .map_err(|err| backend_error("sam activation", err))?
                    } else {
                        let host_input = self
                            .download_tensor(input_gpu, layer.input_shape)
                            .map_err(|err| backend_error("download sam input", err))?;
                        let host_source = self
                            .download_tensor(source_gpu, source_layer.output_shape)
                            .map_err(|err| backend_error("download sam source", err))?;
                        let (host_output, _) =
                            forward_sam(spec, &host_input, &host_source, layer.output_shape)?;
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &host_output.data,
                            )
                            .map_err(|err| backend_error("upload sam output", err))?
                    }
                }
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
                        .map_err(|err| backend_error("compact connected weights", err))?;
                    let biases_gpu = self
                        .backend
                        .upload_f32(&[spec.output], &biases)
                        .map_err(|err| backend_error("compact connected biases", err))?;
                    let pre_activation = self
                        .backend
                        .connected_f32(
                            current_input.expect("Connected requires current input"),
                            &weights_gpu,
                            &biases_gpu,
                            layer.input_shape.batch,
                            inputs,
                            spec.output,
                            launch_config,
                        )
                        .map_err(|err| backend_error("compact connected", err))?;
                    self.backend
                        .activation_f32(
                            &pre_activation,
                            gpu_activation_kind(layer.index, &spec.activation)?,
                            launch_config,
                        )
                        .map_err(|err| backend_error("compact connected activation", err))?
                }
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = gpu_layer_outputs
                        .get(source_index)
                        .and_then(Option::as_ref)
                        .ok_or(Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.from,
                        })?;
                    let pre_activation = self
                        .backend
                        .scale_channels_f32(
                            current_input.expect("ScaleChannels requires current input"),
                            source,
                            &tensor_backend_shape(layer.output_shape),
                            spec.scale_wh,
                            launch_config,
                        )
                        .map_err(|err| backend_error("GPU ScaleChannels compact", err))?;
                    self.backend
                        .activation_f32(
                            &pre_activation,
                            gpu_activation_kind(layer.index, &spec.activation)?,
                            launch_config,
                        )
                        .map_err(|err| backend_error("GPU ScaleChannels compact activation", err))?
                }
                (LayerKind::Reorg(spec), LayerParams::None) => {
                    if spec.stride == 0 {
                        return Err(Error::InvalidConfig(
                            "reorg stride cannot be zero".to_owned(),
                        ));
                    }
                    self.backend
                        .reorg_f32(
                            current_input.expect("Reorg requires current input"),
                            &tensor_backend_shape(layer.input_shape),
                            &tensor_backend_shape(layer.output_shape),
                            spec.stride,
                            spec.reverse,
                            launch_config,
                        )
                        .map_err(|err| backend_error("GPU Reorg compact forward", err))?
                }
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
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = gpu_layer_outputs
                        .get(source_index)
                        .and_then(Option::as_ref)
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
                        .map_err(|err| backend_error("GPU ChannelSlice compact forward", err))?
                }
                (LayerKind::Logistic, LayerParams::None) => self
                    .backend
                    .activation_f32(
                        current_input.expect("Logistic requires current input"),
                        darknet_cuda::F32ActivationKind::Logistic,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU Logistic compact forward", err))?,
                (LayerKind::Softmax { groups }, LayerParams::None) => self
                    .backend
                    .softmax_f32(
                        current_input.expect("Softmax requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        softmax_group_size(layer.input_shape, *groups)?,
                        launch_config,
                    )
                    .map_err(|err| backend_error("GPU Softmax compact forward", err))?,
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

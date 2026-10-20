use super::*;

/// f16-storage inference (P6).  Training and every non-inference op stay on
/// the f32 path.
impl<B> GpuExecutableNetwork<B>
where
    B: F16TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// f16 twin of [`GpuExecutableNetwork::forward_compact`]: weights and
    /// activations are stored as binary16 on the device (f32 accumulation,
    /// bias and activation), the yolo heads are widened to f32 on download.
    /// Supports convolutional, maxpool, (local) avgpool, standalone batchnorm,
    /// route, LAYA marker gather, SAM, shortcut (mismatched sources are cropped /
    /// zero-padded like Darknet), upsample (also `reverse`), Logistic sigmoid and
    /// OMD loss inference pass-through, group-major JevCe softmax, and yolo-style
    /// pass-through layers;
    /// any other layer returns `UnsupportedLayer` (use `forward_compact`).
    pub fn forward_compact_f16(&self, input: &Tensor) -> Result<GpuForwardSummary> {
        reject_gpu_hyperbolic_foveation(&self.network, "f16 inference")?;
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        for layer in &self.network.layers {
            match &layer.kind {
                LayerKind::Convolutional(_) => {}
                LayerKind::Connected(spec) => {
                    if spec.activation.to_cuda_f32_kind().is_none() {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: format!("{} (f16 inference)", layer.kind.name()),
                        });
                    }
                }
                LayerKind::MaxPool(_)
                | LayerKind::LocalAvgPool(_)
                | LayerKind::AvgPool
                | LayerKind::BatchNorm(_)
                | LayerKind::Route(_)
                | LayerKind::LayaMarkerGather(_)
                | LayerKind::Sam(_)
                | LayerKind::Region(_)
                | LayerKind::ChannelSlice(_)
                | LayerKind::ChannelShuffle(_)
                | LayerKind::Reorg(_)
                | LayerKind::Softmax { .. }
                | LayerKind::ScaleChannels(_)
                | LayerKind::Dropout { .. }
                | LayerKind::Cost { .. }
                | LayerKind::Logistic
                | LayerKind::OmdLoss(_)
                | LayerKind::JevCe(_)
                | LayerKind::LayaActHead(_)
                | LayerKind::Upsample(_)
                | LayerKind::Yolo(_)
                | LayerKind::ObbYolo(_)
                | LayerKind::GaussianYolo(_)
                | LayerKind::Detection(_)
                | LayerKind::Regression(_) => {}
                LayerKind::Shortcut(spec) => {
                    if spec.from.is_empty() {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "shortcut without sources (f16 inference)".to_owned(),
                        });
                    }
                }
                other => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: format!("{} (f16 inference)", other.name()),
                    });
                }
            }
        }

        let launch_config = ElementwiseLaunchConfig::default();
        let input_gpu = self
            .backend
            .upload_f16(&tensor_backend_shape(input.shape), &input.data)
            .map_err(|err| backend_error("upload f16 input tensor", err))?;
        let mut gpu_layer_outputs: Vec<Option<B::TensorF16>> =
            vec![None; self.network.layers.len()];
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
                    let groups = spec.groups.max(1);
                    let cached_w16 = self.f16_conv_weights.borrow()[owner]
                        .as_ref()
                        .and_then(|shared| shared.downcast_ref::<B::TensorF16>())
                        .cloned();
                    let weights16 = match cached_w16 {
                        Some(w) => w,
                        None => {
                            let per_filter =
                                layer.input_shape.channels / groups * spec.size * spec.size;
                            let w = self
                                .backend
                                .conv_weights_f16(&cached.weights, spec.filters, per_filter)
                                .map_err(|err| backend_error("convert conv weights to f16", err))?;
                            self.f16_conv_weights.borrow_mut()[owner] =
                                Some(std::rc::Rc::new(w.clone()));
                            w
                        }
                    };
                    let normalization = spec.activation.to_channel_normalization();
                    let activation = if normalization.is_some() {
                        F32ActivationKind::Linear
                    } else {
                        gpu_activation_kind(layer.index, &spec.activation)?
                    };
                    let convolution = self
                        .backend
                        .conv2d_bias_act_f16(
                            current_input.expect("convolution requires current input"),
                            &weights16,
                            &cached.biases,
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
                            groups,
                            activation,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 convolution", err))?;
                    if let Some(normalization) = normalization {
                        self.backend
                            .normalize_channels_f16(
                                &convolution,
                                &tensor_backend_shape(layer.output_shape),
                                normalization,
                                launch_config,
                            )
                            .map_err(|err| {
                                backend_error("f16 convolution channel normalization", err)
                            })?
                    } else {
                        convolution
                    }
                }
                (LayerKind::Connected(spec), LayerParams::Connected(params)) => {
                    let inputs = layer.input_shape.element_count();
                    if params.weights.len() != spec.output * inputs
                        || params.biases.len() != spec.output
                    {
                        return Err(Error::InvalidConfig(format!(
                            "connected layer #{} expects {} weights and {} biases, got {} and {}",
                            layer.index,
                            spec.output * inputs,
                            spec.output,
                            params.weights.len(),
                            params.biases.len()
                        )));
                    }
                    let (weights, biases) = if let Some(bn) = params.batch_norm.as_ref() {
                        validate_batchnorm_params(bn, spec.output)?;
                        let mut weights = params.weights.clone();
                        let mut biases = params.biases.clone();
                        for (output, row) in weights.chunks_mut(inputs).enumerate() {
                            let scale = bn.scales[output]
                                / (bn.rolling_variance[output] + BATCH_NORM_EPSILON).sqrt();
                            row.iter_mut().for_each(|weight| *weight *= scale);
                            biases[output] -= bn.rolling_mean[output] * scale;
                        }
                        (weights, biases)
                    } else {
                        (params.weights.clone(), params.biases.clone())
                    };
                    self.backend
                        .connected_f16(
                            current_input.expect("connected requires current input"),
                            &weights,
                            &biases,
                            &tensor_backend_shape(layer.input_shape),
                            &tensor_backend_shape(layer.output_shape),
                            spec.output,
                            gpu_activation_kind(layer.index, &spec.activation)?,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 connected", err))?
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => {
                    let maxpool_input = current_input.expect("maxpool requires current input");
                    if spec.depth {
                        self.backend.maxpool_depth_f16(
                            maxpool_input,
                            &tensor_backend_shape(layer.output_shape),
                            layer.input_shape.channels,
                            launch_config,
                        )
                    } else {
                        self.backend.maxpool_f16(
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
                    .map_err(|err| backend_error("f16 maxpool", err))?
                }
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => {
                    if layer.output_shape.channels != layer.input_shape.channels {
                        return Err(Error::UnsupportedLayer {
                            layer_index: layer.index,
                            name: "local_avgpool_depth".to_owned(),
                        });
                    }
                    self.backend
                        .local_avgpool_f16(
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
                        .map_err(|err| backend_error("f16 local_avgpool", err))?
                }
                (LayerKind::AvgPool, LayerParams::None) => self
                    .backend
                    .avgpool_f16(
                        current_input.expect("avgpool requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        launch_config,
                    )
                    .map_err(|err| backend_error("f16 avgpool", err))?,
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(params)) => {
                    let shape = layer.input_shape;
                    let (scales, biases) =
                        folded_batchnorm_inference_params(params, shape.channels)?;
                    let scales = self
                        .backend
                        .upload_f32(&[shape.channels], &scales)
                        .map_err(|err| backend_error("upload batchnorm inference scales", err))?;
                    let biases = self
                        .backend
                        .upload_f32(&[shape.channels], &biases)
                        .map_err(|err| backend_error("upload batchnorm inference biases", err))?;
                    self.backend
                        .channel_affine_act_f16(
                            current_input.expect("batchnorm requires current input"),
                            Some(&scales),
                            &biases,
                            shape.channels,
                            shape.width * shape.height,
                            F32ActivationKind::Linear,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 batchnorm inference", err))?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    let mut accumulated = current_input
                        .expect("shortcut requires current input")
                        .clone();
                    let last = spec.from.len().saturating_sub(1);
                    for (n, raw_index) in spec.from.iter().enumerate() {
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
                        // Darknet resizes a mismatched source flat per sample.
                        let fitted;
                        let source = if source_layer.output_shape != layer.output_shape {
                            fitted = self
                                .backend
                                .shortcut_fit_f16(
                                    source,
                                    layer.output_shape.batch,
                                    &tensor_backend_shape(layer.output_shape),
                                    launch_config,
                                )
                                .map_err(|err| backend_error("f16 shortcut resize source", err))?;
                            &fitted
                        } else {
                            source
                        };
                        // The activation rides on the last add (one rounding).
                        let act = if n == last {
                            activation
                        } else {
                            F32ActivationKind::Linear
                        };
                        accumulated = self
                            .backend
                            .shortcut_add_act_f16(&accumulated, source, act, launch_config)
                            .map_err(|err| backend_error("f16 shortcut add", err))?;
                    }
                    accumulated
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
                    let source = gpu_layer_outputs
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
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    self.backend
                        .sam_f16(
                            current_input.expect("sam requires current input"),
                            source,
                            &tensor_backend_shape(layer.output_shape),
                            activation,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 sam", err))?
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
                    let activation = gpu_activation_kind(layer.index, &spec.activation)?;
                    self.backend
                        .scale_channels_f16(
                            current_input.expect("scale_channels requires current input"),
                            source,
                            &tensor_backend_shape(layer.input_shape),
                            &tensor_backend_shape(layer.output_shape),
                            spec.scale_wh,
                            activation,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 scale_channels", err))?
                }
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
                        .route_concat_f16(
                            &[source],
                            &[spec.start],
                            &[spec.end - spec.start],
                            &[source_channels],
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 channel_slice", err))?
                }
                (LayerKind::Softmax { groups }, LayerParams::None) => self
                    .backend
                    .softmax_f16(
                        current_input.expect("softmax requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        *groups,
                        launch_config,
                    )
                    .map_err(|err| backend_error("f16 softmax", err))?,
                (LayerKind::JevCe(spec), LayerParams::None) => {
                    let per_sample =
                        layer.input_shape.element_count() / layer.input_shape.batch.max(1);
                    let scale = self
                        .backend
                        .upload_f32(&[1], &[1.0 / spec.temperature.max(1.0e-6)])
                        .map_err(|err| backend_error("upload JevCe f16 temperature", err))?;
                    let bias = self
                        .backend
                        .upload_f32(&[1], &[0.0])
                        .map_err(|err| backend_error("upload JevCe f16 bias", err))?;
                    let scaled = self
                        .backend
                        .channel_affine_act_f16(
                            current_input.expect("jev_ce requires current input"),
                            Some(&scale),
                            &bias,
                            1,
                            per_sample,
                            F32ActivationKind::Linear,
                            launch_config,
                        )
                        .map_err(|err| backend_error("scale JevCe f16 logits", err))?;
                    if spec.channel_major {
                        self.backend
                            .softmax_channel_major_f16(
                                &scaled,
                                &tensor_backend_shape(layer.output_shape),
                                spec.candidates,
                                spec.groups,
                                launch_config,
                            )
                            .map_err(|err| backend_error("f16 JevCe channel-major softmax", err))?
                    } else {
                        self.backend
                            .softmax_f16(
                                &scaled,
                                &tensor_backend_shape(layer.output_shape),
                                spec.groups,
                                launch_config,
                            )
                            .map_err(|err| backend_error("f16 JevCe softmax", err))?
                    }
                }
                (LayerKind::Region(spec), LayerParams::None) => self
                    .backend
                    .region_forward_f16(
                        current_input.expect("region requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        spec.num,
                        spec.coords + spec.classes + 1,
                        spec.classes,
                        spec.softmax,
                        launch_config,
                    )
                    .map_err(|err| backend_error("f16 Region forward", err))?,
                (LayerKind::Logistic, LayerParams::None) => {
                    let bias = self
                        .backend
                        .upload_f32(&[1], &[0.0])
                        .map_err(|err| backend_error("upload logistic f16 bias", err))?;
                    self.backend
                        .channel_affine_act_f16(
                            current_input.expect("logistic requires current input"),
                            None,
                            &bias,
                            1,
                            layer.input_shape.element_count() / layer.input_shape.batch.max(1),
                            F32ActivationKind::Logistic,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 logistic", err))?
                }
                (LayerKind::Dropout { .. }, LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::OmdLoss(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None) => current_input
                    .expect("identity layer requires current input")
                    .clone(),
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => self
                    .backend
                    .channel_shuffle_f16(
                        current_input.expect("channel_shuffle requires current input"),
                        &tensor_backend_shape(layer.output_shape),
                        layer.input_shape.channels,
                        spec.groups,
                        launch_config,
                    )
                    .map_err(|err| backend_error("f16 channel_shuffle", err))?,
                (LayerKind::Reorg(spec), LayerParams::None) => self
                    .backend
                    .reorg_f16(
                        current_input.expect("reorg requires current input"),
                        &tensor_backend_shape(layer.input_shape),
                        &tensor_backend_shape(layer.output_shape),
                        spec.stride,
                        spec.reverse,
                        launch_config,
                    )
                    .map_err(|err| backend_error("f16 reorg", err))?,
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
                        .route_concat_f16(
                            &route_inputs,
                            &route_channel_offsets,
                            &route_channels,
                            &route_total_channels,
                            &tensor_backend_shape(layer.output_shape),
                            layer.output_shape.width * layer.output_shape.height,
                            launch_config,
                        )
                        .map_err(|err| backend_error("f16 route concat", err))?
                }
                (LayerKind::LayaMarkerGather(spec), LayerParams::None) => {
                    let a_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let b_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let a_layer = self
                        .network
                        .layers
                        .get(a_index)
                        .ok_or(Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[0],
                        })?;
                    let b_layer = self
                        .network
                        .layers
                        .get(b_index)
                        .ok_or(Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[1],
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
                        .marker_gather_f16(
                            a_source,
                            b_source,
                            &tensor_backend_shape(a_layer.output_shape),
                            &tensor_backend_shape(b_layer.output_shape),
                            spec.max_options,
                            launch_config,
                        )
                        .map_err(|err| backend_error("GPU f16 laya_marker_gather", err))?
                }
                (LayerKind::LayaActHead(spec), LayerParams::LayaActHead(weights)) => {
                    let head_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let scorer_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let marker_index = resolve_relative_layer(layer.index, spec.layers[2])?;
                    let head_layer =
                        self.network
                            .layers
                            .get(head_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.layers[0],
                            })?;
                    let scorer_layer =
                        self.network
                            .layers
                            .get(scorer_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.layers[1],
                            })?;
                    let marker_layer =
                        self.network
                            .layers
                            .get(marker_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index: layer.index,
                                reference: spec.layers[2],
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
                    laya_act_gpu::forward_f16(
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
                (LayerKind::Upsample(spec), LayerParams::None) => self
                    .backend
                    .upsample_f16(
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
                    .map_err(|err| backend_error("f16 upsample", err))?,
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::ObbYolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None) => current_input
                    .expect("staging layer requires current input")
                    .clone(),
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
                    Some(Tensor {
                        shape: layer.output_shape,
                        data: self
                            .backend
                            .download_f16(&output_gpu)
                            .map_err(|err| backend_error("download f16 layer output", err))?,
                    })
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

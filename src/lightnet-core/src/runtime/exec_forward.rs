use super::*;

impl ExecutableNetwork {
    pub fn forward(&self, input: &Tensor) -> Result<ForwardResult> {
        self.forward_with_mode(input, false)
    }

    /// Run inference with optional per-layer MambaVision recurrent states.
    /// Pass an empty slice to initialize every MambaVision layer with zeros;
    /// otherwise the slice must have one entry per network layer.
    pub fn forward_with_mambavision_states(
        &self,
        input: &Tensor,
        states: &[Option<crate::mambavision::MambaVisionState>],
    ) -> Result<ForwardResult> {
        self.forward_with_mode_and_states(input, false, states, 0x9e37_79b9_u64)
    }

    /// Training counterpart to `forward_with_mambavision_states`.
    pub fn forward_training_with_mambavision_states(
        &self,
        input: &Tensor,
        states: &[Option<crate::mambavision::MambaVisionState>],
    ) -> Result<ForwardResult> {
        self.forward_with_mode_and_states(input, true, states, 0x9e37_79b9_u64)
    }

    /// Execute an inference-only graph while retaining tensors only until their
    /// final route/shortcut consumer.  This is intentionally restricted to
    /// layers whose inference state is fully represented by their output; all
    /// other networks retain the complete forward graph for compatibility.
    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor> {
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        if self.network.layers.iter().any(|layer| {
            !matches!(
                layer.kind,
                LayerKind::Convolutional(_)
                    | LayerKind::MaxPool(crate::model::MaxPoolSpec { depth: false, .. })
                    | LayerKind::AvgPool
                    | LayerKind::Route(_)
                    | LayerKind::Shortcut(crate::model::ShortcutSpec {
                        weights_type: crate::model::ShortcutWeightsType::None,
                        ..
                    })
                    | LayerKind::Upsample(_)
                    | LayerKind::Yolo(_)
                    | LayerKind::ObbYolo(_)
                    | LayerKind::GaussianYolo(_)
                    | LayerKind::Region(_)
                    | LayerKind::Detection(_)
                    | LayerKind::Softmax { .. }
                    | LayerKind::Cost { .. }
                    | LayerKind::Contrastive(_)
            )
        }) {
            return self
                .forward(input)?
                .final_output()
                .cloned()
                .ok_or_else(|| Error::InvalidConfig("network has no layers".to_owned()));
        }

        let mut outputs = vec![None; self.network.layers.len()];
        let mut remaining_uses = gpu_layer_output_use_counts(&self.network)?;
        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            let current_input = if layer.index == 0 {
                Some(input)
            } else if layer_uses_previous_input(&layer.kind) {
                Some(
                    outputs
                        .get(layer.index - 1)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "missing previous inference output for layer #{}",
                                layer.index
                            ))
                        })?,
                )
            } else {
                None
            };

            let output = match (&layer.kind, params) {
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    forward_convolutional(
                        spec,
                        conv,
                        current_input.expect("conv input"),
                        layer.output_shape,
                    )
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input.expect("maxpool input"),
                    layer.output_shape,
                    false,
                )?,
                (LayerKind::AvgPool, LayerParams::None) => {
                    forward_avgpool(current_input.expect("avgpool input"))
                }
                (LayerKind::Route(spec), LayerParams::None) => {
                    forward_route_compact(spec, layer.index, &outputs, layer.output_shape)?
                }
                (LayerKind::LayaMarkerGather(spec), LayerParams::None) => {
                    let a_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let b_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let a = outputs.get(a_index).and_then(Option::as_ref).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[0],
                        },
                    )?;
                    let b = outputs.get(b_index).and_then(Option::as_ref).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[1],
                        },
                    )?;
                    laya_marker_gather_forward(spec, a, b, layer.output_shape)?
                }
                (LayerKind::LayaActHead(spec), LayerParams::LayaActHead(weights)) => {
                    let head_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let scorer_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let marker_index = resolve_relative_layer(layer.index, spec.layers[2])?;
                    let head = outputs.get(head_index).and_then(Option::as_ref).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[0],
                        },
                    )?;
                    let scorer = outputs.get(scorer_index).and_then(Option::as_ref).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[1],
                        },
                    )?;
                    let marker = outputs.get(marker_index).and_then(Option::as_ref).ok_or(
                        Error::InvalidRoute {
                            layer_index: layer.index,
                            reference: spec.layers[2],
                        },
                    )?;
                    laya_act_head_forward(spec, weights, head, scorer, marker, layer.output_shape)?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => forward_shortcut_compact(
                    spec,
                    current_input.expect("shortcut input"),
                    layer.index,
                    &outputs,
                )?,
                (LayerKind::Upsample(spec), LayerParams::None) => forward_upsample(
                    spec,
                    current_input.expect("upsample input"),
                    layer.output_shape,
                )?,
                (LayerKind::Softmax { groups }, LayerParams::None) => {
                    forward_softmax(current_input.expect("softmax input"), *groups)?
                }
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::ObbYolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Region(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Contrastive(_), LayerParams::None) => {
                    current_input.expect("pass-through input").clone()
                }
                _ => unreachable!("layer-family preflight must cover compact inference"),
            };

            outputs[layer.index] = Some(output);
            release_gpu_sources_for_layer(
                layer.index,
                &layer.kind,
                &mut remaining_uses,
                &mut outputs,
            )?;
            if layer.index + 1 == self.network.layers.len() {
                return outputs[layer.index].take().ok_or_else(|| {
                    Error::InvalidConfig("missing final inference output".to_owned())
                });
            }
            if remaining_uses[layer.index] == 0 {
                outputs[layer.index] = None;
            }
        }
        Err(Error::InvalidConfig("network has no layers".to_owned()))
    }

    /// Run a training forward pass.  Darknet's dropout layer is active only
    /// during training; keeping this explicit prevents inference commands from
    /// becoming stochastic while allowing the detector trainer to use the
    /// correct inverted-dropout semantics.
    pub fn forward_training(&self, input: &Tensor) -> Result<ForwardResult> {
        self.forward_with_mode(input, true)
    }

    pub(super) fn forward_with_mode(&self, input: &Tensor, training: bool) -> Result<ForwardResult> {
        self.forward_with_mode_and_states(input, training, &[], 0x9e37_79b9_u64)
    }

    pub(super) fn forward_training_with_rng_seed(
        &self,
        input: &Tensor,
        rng_seed: u64,
    ) -> Result<ForwardResult> {
        self.forward_with_mode_and_states(input, true, &[], rng_seed)
    }

    pub(super) fn forward_with_mode_and_states(
        &self,
        input: &Tensor,
        training: bool,
        mambavision_states: &[Option<crate::mambavision::MambaVisionState>],
        rng_seed: u64,
    ) -> Result<ForwardResult> {
        if !mambavision_states.is_empty() && mambavision_states.len() != self.network.layers.len() {
            return Err(Error::InvalidConfig(format!(
                "MambaVision state list has {} entries, expected {} network layers",
                mambavision_states.len(),
                self.network.layers.len()
            )));
        }
        let expected = self.network.net.input_shape();
        if input.shape != expected {
            return Err(Error::InvalidConfig(format!(
                "input shape mismatch: expected {expected}, got {}",
                input.shape
            )));
        }
        let has_obb_refine = self
            .network
            .layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::ObbRefine(_)));
        if has_obb_refine && training {
            return Err(Error::InvalidConfig(
                "Rust training for networks containing [obb_refine] is unsupported until CARPN Stage-2 assignment, loss, backward, and updates are implemented".to_owned(),
            ));
        }
        if has_obb_refine && input.shape.batch != 1 {
            return Err(Error::InvalidConfig(format!(
                "CPU CARPN inference currently requires batch size one, got {}",
                input.shape.batch
            )));
        }

        let mut layer_outputs = Vec::with_capacity(self.network.layers.len());
        let mut layer_activation_inputs = Vec::with_capacity(self.network.layers.len());
        let mut dropout_masks = Vec::with_capacity(self.network.layers.len());
        let mut crop_caches = Vec::with_capacity(self.network.layers.len());
        let mut recurrent_caches = Vec::with_capacity(self.network.layers.len());
        let mut yolo_heads = Vec::new();
        let mut yolo_3d_heads = Vec::new();
        let mut modern_heads = Vec::new();
        let mut moe_caches = Vec::with_capacity(self.network.layers.len());
        let mut eml_caches = Vec::with_capacity(self.network.layers.len());
        let mut dcnv4_caches = Vec::with_capacity(self.network.layers.len());
        let mut deform_conv_caches = Vec::with_capacity(self.network.layers.len());
        let mut dsconv_caches = Vec::with_capacity(self.network.layers.len());
        let mut graph_conv_caches = Vec::with_capacity(self.network.layers.len());
        let mut wmhf_caches = Vec::with_capacity(self.network.layers.len());
        let mut clifford_caches = Vec::with_capacity(self.network.layers.len());
        let mut recursive_block_caches = Vec::with_capacity(self.network.layers.len());
        let mut hyperbolic_foveation_caches: Vec<Option<HyperbolicFoveationForwardCache>> =
            Vec::with_capacity(self.network.layers.len());
        let mut foveation_film_caches = Vec::with_capacity(self.network.layers.len());
        let mut vit_caches = Vec::with_capacity(self.network.layers.len());
        let mut depth_anything_caches = Vec::with_capacity(self.network.layers.len());
        let mut dpt_head_caches = Vec::with_capacity(self.network.layers.len());
        let mut swin_caches = Vec::with_capacity(self.network.layers.len());
        let mut tucker_attention_caches = Vec::with_capacity(self.network.layers.len());
        let mut mambavision_caches = Vec::with_capacity(self.network.layers.len());
        let mut kimi_kda_caches = Vec::with_capacity(self.network.layers.len());
        let mut obb_refine_outputs = Vec::new();
        let mut rng_state = rng_seed;

        for (layer, params) in self.network.layers.iter().zip(&self.params) {
            // This match is the Rust equivalent of dispatching a C++ layer's
            // `forward` callback.  Previous outputs stay alive because later
            // route/shortcut/SAM layers and the backward pass need them.
            let current_input = if layer.index == 0 {
                input
            } else {
                layer_outputs.last().ok_or_else(|| {
                    Error::InvalidConfig("missing previous layer output".to_owned())
                })?
            };

            let mut dropout_mask = None;
            let mut crop_cache = None;
            let mut recurrent_cache = None;
            let mut moe_cache = None;
            let mut eml_cache = None;
            let mut vit_cache = None;
            let mut depth_anything_cache = None;
            let mut dpt_head_cache = None;
            let mut swin_cache = None;
            let mut tucker_attention_cache = None;
            let mut mambavision_cache = None;
            let mut kimi_kda_cache = None;
            let mut dcnv4_cache = None;
            let mut deform_conv_cache = None;
            let mut dsconv_cache = None;
            let mut graph_conv_cache = None;
            let mut wmhf_cache = None;
            let mut clifford_cache = None;
            let mut recursive_block_cache = None;
            let mut hyperbolic_foveation_cache = None;
            let mut foveation_film_cache = None;
            let (output, activation_input) = match (&layer.kind, params) {
                (
                    LayerKind::HyperbolicFoveation(spec),
                    LayerParams::HyperbolicFoveation(params),
                ) => {
                    let (output, cache) = forward_hyperbolic_foveation(
                        &self.network,
                        layer,
                        spec,
                        params,
                        input,
                        current_input,
                        &layer_outputs,
                    )?;
                    hyperbolic_foveation_cache = Some(cache);
                    (output, None)
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
                    let (output, cache) = crate::foveation_film::forward_nchw(
                        &current_input.data,
                        &hf_cache.params_per_batch,
                        layer.input_shape.batch,
                        layer.input_shape.channels,
                        layer.input_shape.height,
                        layer.input_shape.width,
                        spec.param_count,
                        spec.embed_dim,
                        params,
                    )?;
                    foveation_film_cache = Some(cache);
                    (Tensor::from_data(layer.output_shape, output)?, None)
                }
                (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(params)) => {
                    let (output, cache) = forward_recursive_block(
                        &self.network,
                        self.train_iteration,
                        layer,
                        spec,
                        params,
                        current_input,
                        training,
                    )?;
                    recursive_block_cache = Some(cache);
                    (output, None)
                }
                (
                    LayerKind::Convolutional(spec),
                    LayerParams::Convolutional(_) | LayerParams::None,
                ) => {
                    let conv =
                        resolve_convolutional_params(&self.network, &self.params, layer.index)?;
                    forward_convolutional_with_cache(spec, conv, current_input, layer.output_shape)
                }
                (LayerKind::GraphConv(spec), LayerParams::GraphConv(params)) => {
                    let (output, activation_input, cache) = forward_graph_conv_with_cache(
                        spec,
                        params,
                        current_input,
                        layer.output_shape,
                        training,
                    )?;
                    graph_conv_cache = Some(cache);
                    (output, Some(activation_input))
                }
                (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params)) => {
                    let (output, activation_input, cache) =
                        forward_dcnv4_with_cache(spec, params, current_input, layer.output_shape)?;
                    dcnv4_cache = Some(cache);
                    (output, Some(activation_input))
                }
                (LayerKind::DsConv(spec), LayerParams::DsConv(params)) => {
                    let (output, activation_input, cache) =
                        forward_dsconv_with_cache(spec, params, current_input, layer.output_shape)?;
                    dsconv_cache = Some(cache);
                    (output, Some(activation_input))
                }
                (LayerKind::Wmhf(spec), LayerParams::Wmhf(params)) => {
                    let (output, activation_input, cache) = forward_wmhf_with_cache(
                        spec,
                        params,
                        current_input,
                        layer.output_shape,
                        training,
                    )?;
                    wmhf_cache = Some(cache);
                    (output, Some(activation_input))
                }
                (LayerKind::Clifford(spec), LayerParams::Clifford(params)) => {
                    let drop_mask = if training && spec.drop_path > 0.0 {
                        let keep_probability = 1.0 - spec.drop_path;
                        Some(
                            (0..current_input.shape.batch)
                                .map(|_| {
                                    if next_random_unit(&mut rng_state) < keep_probability {
                                        1.0 / keep_probability
                                    } else {
                                        0.0
                                    }
                                })
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    };
                    let (output, cache) = crate::clifford::forward(
                        spec,
                        params,
                        current_input,
                        training,
                        drop_mask.as_deref(),
                    )?;
                    clifford_cache = Some(cache);
                    (output, None)
                }
                (LayerKind::DeformConv(spec), LayerParams::DeformConv(params)) => {
                    let (output, activation_input, cache) = forward_deform_conv_with_cache(
                        spec,
                        params,
                        current_input,
                        layer.output_shape,
                    )?;
                    deform_conv_cache = Some(cache);
                    (output, Some(activation_input))
                }
                (LayerKind::Deconvolutional(spec), LayerParams::Convolutional(p)) => {
                    let pre = crate::deconvolutional::deconv_forward(
                        current_input,
                        &p.weights,
                        &p.biases,
                        spec.size,
                        spec.stride,
                        layer.output_shape,
                    );
                    let data = pre
                        .data
                        .iter()
                        .map(|v| activate(spec.activation.clone(), *v))
                        .collect();
                    (Tensor::from_data(pre.shape, data)?, Some(pre))
                }
                (LayerKind::Connected(spec), LayerParams::Connected(connected)) => {
                    forward_connected_with_cache(spec, connected, current_input, layer.output_shape)
                }
                (LayerKind::ObbRefine(spec), LayerParams::Connected(connected)) => {
                    let (output, refine_output) = crate::carpn::forward_refine_layer(
                        &self.network,
                        layer.index,
                        spec,
                        layer.output_shape,
                        &layer_outputs,
                        connected,
                    )?;
                    obb_refine_outputs.push(refine_output);
                    (output, None)
                }
                (LayerKind::Local(spec), LayerParams::Local(local)) => {
                    forward_local_with_cache(spec, local, current_input, layer.output_shape)?
                }
                (LayerKind::Gru(spec), LayerParams::Gru(gru)) => (
                    forward_gru_layer(spec, gru, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::BatchNorm(_), LayerParams::BatchNorm(batch_norm)) => {
                    forward_batchnorm_with_cache(batch_norm, current_input, training)?
                }
                (LayerKind::Crop(spec), LayerParams::None) => {
                    let (output, cache) = forward_crop(
                        spec,
                        current_input,
                        layer.output_shape,
                        training,
                        &mut rng_state,
                    )?;
                    crop_cache = Some(cache);
                    (output, None)
                }
                (LayerKind::MaxPool(spec), LayerParams::None) if spec.depth => {
                    forward_maxpool_depth(current_input, layer.output_shape)
                        .map(|output| (output, None))?
                }
                (LayerKind::MaxPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input,
                    layer.output_shape,
                    false,
                )
                .map(|output| (output, None))?,
                (LayerKind::LocalAvgPool(spec), LayerParams::None) => forward_maxpool(
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.padding,
                    current_input,
                    layer.output_shape,
                    true,
                )
                .map(|output| (output, None))?,
                (LayerKind::AvgPool, LayerParams::None) => (forward_avgpool(current_input), None),
                (LayerKind::Route(spec), LayerParams::None) => (
                    forward_route(spec, layer.index, &layer_outputs, layer.output_shape)?,
                    None,
                ),
                (LayerKind::LayaMarkerGather(spec), LayerParams::None) => {
                    let a_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let b_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let a = layer_outputs.get(a_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.layers[0],
                    })?;
                    let b = layer_outputs.get(b_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.layers[1],
                    })?;
                    (
                        laya_marker_gather_forward(spec, a, b, layer.output_shape)?,
                        None,
                    )
                }
                (LayerKind::LayaActHead(spec), LayerParams::LayaActHead(weights)) => {
                    let head_index = resolve_relative_layer(layer.index, spec.layers[0])?;
                    let scorer_index = resolve_relative_layer(layer.index, spec.layers[1])?;
                    let marker_index = resolve_relative_layer(layer.index, spec.layers[2])?;
                    let head = layer_outputs.get(head_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.layers[0],
                    })?;
                    let scorer = layer_outputs.get(scorer_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.layers[1],
                    })?;
                    let marker = layer_outputs.get(marker_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.layers[2],
                    })?;
                    (
                        laya_act_head_forward(
                            spec,
                            weights,
                            head,
                            scorer,
                            marker,
                            layer.output_shape,
                        )?,
                        None,
                    )
                }
                (LayerKind::Shortcut(spec), LayerParams::Shortcut(params)) => {
                    forward_shortcut_weighted(
                        spec,
                        params,
                        current_input,
                        layer.index,
                        &layer_outputs,
                    )?
                }
                (LayerKind::Shortcut(spec), LayerParams::None) => {
                    forward_shortcut_with_cache(spec, current_input, layer.index, &layer_outputs)?
                }
                (LayerKind::Upsample(spec), LayerParams::None) => (
                    forward_upsample(spec, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::JevCe(spec), LayerParams::None) => (
                    Tensor::from_data(
                        current_input.shape,
                        crate::jev_ce::jev_ce_probs(
                            &current_input.data,
                            spec,
                            current_input.shape.batch,
                        ),
                    )?,
                    None,
                ),
                (LayerKind::LayaLoss(spec), LayerParams::None) => (
                    Tensor::from_data(
                        current_input.shape,
                        crate::laya_loss::laya_probs(
                            &current_input.data,
                            current_input.shape.batch,
                            spec.max_options,
                        ),
                    )?,
                    None,
                ),
                (LayerKind::Vit(spec), LayerParams::Vit(weights)) => {
                    let geometry = vit_geometry(layer, spec);
                    let (data, cache) =
                        crate::vit::vit_forward(&current_input.data, weights, &geometry)
                            .map_err(Error::InvalidConfig)?;
                    vit_cache = Some(cache);
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                (LayerKind::Swin(spec), LayerParams::Swin(weights)) => {
                    let geometry = crate::swin_block::SwinGeometry {
                        batch: current_input.shape.batch,
                        channels: current_input.shape.channels,
                        height: current_input.shape.height,
                        width: current_input.shape.width,
                        filters: spec.filters,
                        heads: spec.heads,
                        window: spec.size,
                        shift: spec.shift,
                        ffn_ratio: spec.ffn_ratio,
                        activation: spec.activation.clone(),
                    };
                    let (data, cache) =
                        crate::swin_block::swin_forward(&current_input.data, weights, &geometry)
                            .map_err(Error::InvalidConfig)?;
                    swin_cache = Some(cache);
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                (LayerKind::TuckerAttention(spec), LayerParams::TuckerAttention(weights)) => {
                    let geometry = crate::tucker_attention::TuckerGeometry {
                        batch: current_input.shape.batch,
                        channels: current_input.shape.channels,
                        height: current_input.shape.height,
                        width: current_input.shape.width,
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
                        &current_input.data,
                        weights,
                        &geometry,
                    )
                    .map_err(Error::InvalidConfig)?;
                    tucker_attention_cache = Some(cache);
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                (LayerKind::MambaVision(spec), LayerParams::MambaVision(weights)) => {
                    let forward = crate::mambavision::mambavision_forward(
                        &current_input.data,
                        current_input.shape,
                        spec,
                        weights,
                        mambavision_states.get(layer.index).and_then(Option::as_ref),
                    )
                    .map_err(Error::InvalidConfig)?;
                    let output = Tensor::from_data(layer.output_shape, forward.output.clone())?;
                    mambavision_cache = Some(forward);
                    (output, None)
                }
                (LayerKind::KimiKda(spec), LayerParams::KimiKda(weights)) => {
                    let (data, cache) = crate::kimi_kda::kimi_kda_forward(
                        &current_input.data,
                        current_input.shape,
                        spec,
                        weights,
                    )
                    .map_err(Error::InvalidConfig)?;
                    kimi_kda_cache = Some(cache);
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                (LayerKind::DepthAnything(spec), LayerParams::DepthAnything(weights)) => {
                    let geometry = spec.geometry(current_input.shape);
                    let (data, cache) =
                        crate::depth_anything::forward(&current_input.data, weights, &geometry)
                            .map_err(Error::InvalidConfig)?;
                    depth_anything_cache = Some(cache);
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                (LayerKind::DptHead(spec), LayerParams::DptHead(weights)) => {
                    let geometry = spec.geometry(current_input.shape);
                    let (data, cache) =
                        crate::dpt_head::forward(&current_input.data, weights, geometry)?;
                    dpt_head_cache = Some(cache);
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                (LayerKind::LayaEncoder(spec), LayerParams::LayaEncoder(weights)) => (
                    laya_encoder_forward(spec, weights, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::LayaHead(spec), LayerParams::LayaHead(weights)) => (
                    laya_head_forward(spec, weights, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::Logistic, LayerParams::None) => {
                    let data = current_input
                        .data
                        .iter()
                        .map(|v| activate(Activation::Logistic, *v))
                        .collect();
                    (Tensor::from_data(current_input.shape, data)?, None)
                }
                (LayerKind::Segmentation(spec), LayerParams::None) => {
                    let out = crate::segmentation::segmentation_forward(
                        &current_input.data,
                        spec,
                        current_input.shape.batch,
                        current_input.shape.width * current_input.shape.height,
                    );
                    (Tensor::from_data(current_input.shape, out.output)?, None)
                }
                // C++ `memcpy(l.output, state.input, ...)` then
                // `activate_modern_output`, so the head *does* transform its
                // input: the last `1 + classes` entries of every cell go
                // through a sigmoid while the four box entries stay raw.
                (LayerKind::Yolox(spec), LayerParams::None)
                | (LayerKind::Ppyoloe(spec), LayerParams::None)
                | (LayerKind::Yolonas(spec), LayerParams::None) => {
                    let layout = crate::modern_yolo::ModernLayout::for_spec(
                        current_input.shape.batch,
                        current_input.shape.width,
                        current_input.shape.height,
                        spec,
                    );
                    let mut data = current_input.data.clone();
                    if data.len() == layout.len() {
                        crate::modern_yolo::activate_modern_output(&mut data, &layout);
                    }
                    (Tensor::from_data(current_input.shape, data)?, None)
                }
                // `[centernet]` sigmoids the `classes` heatmap entries of
                // every cell and leaves the four regression entries raw.
                (LayerKind::CenterNet(spec), LayerParams::None) => {
                    let layout = crate::centernet::CenterNetLayout::for_spec(
                        current_input.shape.batch,
                        current_input.shape.width,
                        current_input.shape.height,
                        spec,
                    );
                    let data = crate::centernet::centernet_forward(&current_input.data, &layout);
                    (Tensor::from_data(current_input.shape, data)?, None)
                }
                // `[laya_scorer]`: LayerNorm -> Linear -> GELU(erf) -> Linear,
                // applied independently to each of the `max_options` slots.
                (LayerKind::LayaScorer(spec), LayerParams::LayaScorer(weights)) => {
                    let batch = current_input.shape.batch;
                    let data = crate::laya_scorer::scorer_forward(
                        &current_input.data,
                        weights,
                        spec,
                        batch,
                    )
                    .map_err(Error::InvalidConfig)?;
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                (LayerKind::EmlConvolutional(spec), LayerParams::EmlConvolutional(params)) => {
                    let (input_branch_output, _) = forward_convolutional_with_cache(
                        &spec.branch,
                        &params.input,
                        current_input,
                        layer.output_shape,
                    );
                    let (self_branch_output, _) = forward_convolutional_with_cache(
                        &spec.branch,
                        &params.self_params,
                        current_input,
                        layer.output_shape,
                    );
                    let use_residual = spec.residual
                        && layer.input_shape.element_count() == layer.output_shape.element_count();
                    let residual = use_residual.then_some(current_input.data.as_slice());
                    let math = crate::eml_conv_math::EmlConvMathParams::from_cpp_values(
                        spec.eml_clamp,
                        spec.eml_eps,
                        spec.eml_scale,
                    );
                    let data = crate::eml_conv_math::forward(
                        &input_branch_output.data,
                        &self_branch_output.data,
                        residual,
                        math,
                    )
                    .map_err(|error| Error::InvalidConfig(error.to_owned()))?;
                    eml_cache = Some(EmlForwardCache {
                        input_branch_output,
                        self_branch_output,
                    });
                    (Tensor::from_data(layer.output_shape, data)?, None)
                }
                // `[moe_convolutional]`: route each image to a soft gate per expert, run
                // every expert convolution, and blend with those gates.  This
                // mirrors C++ `forward_moe_convolutional_layer`, which computes
                // the router first, zeroes `l.output`, then axpys each expert's
                // output scaled by `l.rand[b * E + e]`.
                (LayerKind::Moe(spec), LayerParams::Moe(params)) => {
                    let batch = current_input.shape.batch;
                    let channels = current_input.shape.channels;
                    let spatial = current_input.shape.width * current_input.shape.height;
                    let gates = crate::moe_conv::compute_router(
                        &current_input.data,
                        batch,
                        channels,
                        spatial,
                        &params.router_weights,
                        &params.router_biases,
                        spec,
                    );
                    let mut blended = vec![
                        0.0f32;
                        current_input
                            .data
                            .len()
                            .max(layer.output_shape.element_count() * batch,)
                    ];
                    let mut cache_entry = MoeForwardCache {
                        gates,
                        expert_outputs: Vec::with_capacity(spec.experts),
                        expert_activation_inputs: Vec::with_capacity(spec.experts),
                        pooled: crate::moe_conv::pool_for_router(
                            &current_input.data,
                            batch,
                            channels,
                            spatial,
                        ),
                    };
                    for e in 0..spec.experts {
                        let expert = params.experts.get(e).ok_or_else(|| {
                            Error::InvalidConfig(format!(
                                "layer #{} ([moe_convolutional]) is missing expert {e} of {}",
                                layer.index, spec.experts
                            ))
                        })?;
                        let (output, activation_input) = forward_convolutional_with_cache(
                            &spec.expert,
                            expert,
                            &current_input,
                            layer.output_shape,
                        );
                        // `TensorShape::element_count()` is per image, which is
                        // exactly C++'s `O` (l.outputs) and the stride both the
                        // blend and the gate lookup need.
                        let per_image = layer.output_shape.element_count();
                        for b in 0..batch {
                            let gate = cache_entry.gates[b * spec.experts + e];
                            let slice = (b * per_image)..((b + 1) * per_image);
                            for (dst, src) in blended[slice.clone()]
                                .iter_mut()
                                .zip(output.data[slice].iter())
                            {
                                *dst += gate * src;
                            }
                        }
                        cache_entry.expert_outputs.push(output);
                        cache_entry.expert_activation_inputs.push(activation_input);
                    }
                    moe_cache = Some(cache_entry);
                    (Tensor::from_data(layer.output_shape, blended)?, None)
                }
                // Rust YOLO-family convention: keep raw logits; loss/decode
                // apply activations. Backward is exactly this identity's derivative.
                (LayerKind::Yolo3d(_), LayerParams::None) => (current_input.clone(), None),
                // C++ `memcpy(l.output, state.input, ...)` before any loss work.
                (LayerKind::OmdLoss(_), LayerParams::None) => (current_input.clone(), None),
                // `[contrastive]` never touches its output either: all of the
                // work happens in the delta it derives from the input.
                (LayerKind::Contrastive(_), LayerParams::None) => (current_input.clone(), None),
                (LayerKind::Region(spec), LayerParams::None) => (
                    forward_region_layer(spec, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::Yolo(_), LayerParams::None)
                | (LayerKind::ObbYolo(_), LayerParams::None)
                | (LayerKind::GaussianYolo(_), LayerParams::None)
                | (LayerKind::Detection(_), LayerParams::None)
                | (LayerKind::Cost { .. }, LayerParams::None)
                | (LayerKind::Regression(_), LayerParams::None) => (current_input.clone(), None),
                (LayerKind::Softmax { groups }, LayerParams::None) => {
                    (forward_softmax(current_input, *groups)?, None)
                }
                (
                    LayerKind::Dropout {
                        probability,
                        dropblock,
                    },
                    LayerParams::None,
                ) => {
                    let (output, mask) = match dropblock {
                        Some(spec) => forward_dropblock(
                            current_input,
                            spec,
                            probability.unwrap_or(0.2),
                            training,
                            &mut rng_state,
                            self.train_iteration,
                            self.network.net.max_batches,
                        )?,
                        None => forward_dropout(
                            current_input,
                            probability.unwrap_or(0.2),
                            training,
                            &mut rng_state,
                        )?,
                    };
                    dropout_mask = mask;
                    (output, None)
                }
                (LayerKind::ScaleChannels(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    forward_scale_channels(spec, current_input, source, layer.output_shape)?
                }
                (LayerKind::Sam(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    forward_sam(spec, current_input, source, layer.output_shape)?
                }
                (LayerKind::Reorg(spec), LayerParams::None) => (
                    forward_reorg(spec.stride, spec.reverse, current_input, layer.output_shape)?,
                    None,
                ),
                (LayerKind::ChannelSlice(spec), LayerParams::None) => {
                    let source_index = resolve_relative_layer(layer.index, spec.from)?;
                    let source = layer_outputs.get(source_index).ok_or(Error::InvalidRoute {
                        layer_index: layer.index,
                        reference: spec.from,
                    })?;
                    (
                        crate::channel_slice::slice_channels(source, spec.start, spec.end)?,
                        None,
                    )
                }
                (LayerKind::ChannelShuffle(spec), LayerParams::None) => (
                    crate::channel_shuffle::shuffle_channels(current_input, spec.groups, false)?,
                    None,
                ),
                (LayerKind::Rnn(spec), LayerParams::Rnn(recurrent)) => {
                    let (output, cache) =
                        forward_rnn_layer(spec, recurrent, current_input, layer.output_shape)?;
                    recurrent_cache = Some(RecurrentForwardCache::Rnn(cache));
                    (output, None)
                }
                (LayerKind::Lstm(spec), LayerParams::Lstm(recurrent)) => {
                    let (output, cache) =
                        forward_lstm_layer(spec, recurrent, current_input, layer.output_shape)?;
                    recurrent_cache = Some(RecurrentForwardCache::Lstm(cache));
                    (output, None)
                }
                (LayerKind::Crnn(spec), LayerParams::Crnn(recurrent)) => {
                    let (output, cache) =
                        forward_crnn_layer(spec, recurrent, current_input, layer.output_shape)?;
                    recurrent_cache = Some(RecurrentForwardCache::Crnn(cache));
                    (output, None)
                }
                (LayerKind::Unsupported(name), _) => {
                    return Err(Error::UnsupportedLayer {
                        layer_index: layer.index,
                        name: name.clone(),
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
            let output = if self.network.net.try_fix_nan
                && matches!(layer.kind, LayerKind::Convolutional(_))
            {
                Tensor {
                    shape: output.shape,
                    data: lightnet_cuda::host_ops::sanitize_nan_inf(&output.data, NanInfMode::Fix),
                }
            } else {
                output
            };

            if matches!(layer.kind, LayerKind::Yolo3d(_)) {
                yolo_3d_heads.push(Yolo3dHeadOutput {
                    layer_index: layer.index,
                    output: output.clone(),
                });
            }
            if matches!(
                layer.kind,
                LayerKind::Yolo(_) | LayerKind::ObbYolo(_) | LayerKind::GaussianYolo(_)
            ) {
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

            layer_outputs.push(output);
            layer_activation_inputs.push(activation_input);
            dropout_masks.push(dropout_mask);
            crop_caches.push(crop_cache);
            recurrent_caches.push(recurrent_cache);
            moe_caches.push(moe_cache);
            eml_caches.push(eml_cache);
            dcnv4_caches.push(dcnv4_cache);
            deform_conv_caches.push(deform_conv_cache);
            dsconv_caches.push(dsconv_cache);
            graph_conv_caches.push(graph_conv_cache);
            wmhf_caches.push(wmhf_cache);
            clifford_caches.push(clifford_cache);
            recursive_block_caches.push(recursive_block_cache);
            hyperbolic_foveation_caches.push(hyperbolic_foveation_cache);
            foveation_film_caches.push(foveation_film_cache);
            vit_caches.push(vit_cache);
            depth_anything_caches.push(depth_anything_cache);
            dpt_head_caches.push(dpt_head_cache);
            swin_caches.push(swin_cache);
            tucker_attention_caches.push(tucker_attention_cache);
            mambavision_caches.push(mambavision_cache);
            kimi_kda_caches.push(kimi_kda_cache);
        }

        Ok(ForwardResult {
            layer_outputs,
            layer_activation_inputs,
            dropout_masks,
            crop_caches,
            recurrent_caches,
            moe_caches,
            eml_caches,
            dcnv4_caches,
            deform_conv_caches,
            dsconv_caches,
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
            obb_refine_outputs,
            yolo_heads,
            yolo_3d_heads,
            modern_heads,
        })
    }

}

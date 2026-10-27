use super::*;

pub(crate) fn resolve_convolutional_owner(network: &Network, layer_index: usize) -> Result<usize> {
    let Some(layer) = network.layers.get(layer_index) else {
        return Err(Error::InvalidConfig(format!(
            "missing convolutional layer #{layer_index}"
        )));
    };
    let LayerKind::Convolutional(spec) = &layer.kind else {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} is not convolutional"
        )));
    };
    Ok(spec.share_index.unwrap_or(layer_index))
}

pub(crate) fn resolve_convolutional_params<'a>(
    network: &Network,
    params: &'a [LayerParams],
    layer_index: usize,
) -> Result<&'a ConvolutionalParams> {
    let owner = resolve_convolutional_owner(network, layer_index)?;
    match params.get(owner) {
        Some(LayerParams::Convolutional(conv)) => Ok(conv),
        _ => Err(Error::InvalidConfig(format!(
            "missing convolutional parameters for layer #{} owner #{}",
            layer_index, owner
        ))),
    }
}

pub(crate) fn fold_convolutional_params_for_inference(
    spec: &ConvolutionalSpec,
    params: &ConvolutionalParams,
    input_channels: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let groups = spec.groups.max(1);
    if !spec.filters.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "convolution filters {} are not divisible by groups {}",
            spec.filters, groups
        )));
    }
    if !input_channels.is_multiple_of(groups) {
        return Err(Error::InvalidConfig(format!(
            "input channels {input_channels} are not divisible by groups {groups}"
        )));
    }
    if params.biases.len() != spec.filters {
        return Err(Error::InvalidConfig(format!(
            "convolution bias length mismatch: expected {}, got {}",
            spec.filters,
            params.biases.len()
        )));
    }

    let channels_per_group = input_channels / groups;
    let weights_per_filter = channels_per_group * spec.size * spec.size;
    let expected_weight_count = spec.filters * weights_per_filter;
    if params.weights.len() != expected_weight_count {
        return Err(Error::InvalidConfig(format!(
            "convolution weight length mismatch: expected {expected_weight_count}, got {}",
            params.weights.len()
        )));
    }

    let mut folded_biases = params.biases.clone();
    let mut folded_weights = params.weights.clone();
    if let Some(batch_norm) = params.batch_norm.as_ref() {
        if batch_norm.scales.len() != spec.filters
            || batch_norm.rolling_mean.len() != spec.filters
            || batch_norm.rolling_variance.len() != spec.filters
        {
            return Err(Error::InvalidConfig(format!(
                "batch norm parameter length mismatch for {} filters",
                spec.filters
            )));
        }

        for (filter, folded_bias) in folded_biases.iter_mut().enumerate() {
            let scale = batch_norm.scales[filter]
                / (batch_norm.rolling_variance[filter] + BATCH_NORM_EPSILON).sqrt();
            *folded_bias -= batch_norm.rolling_mean[filter] * scale;
            let weight_base = filter * weights_per_filter;
            for weight in &mut folded_weights[weight_base..weight_base + weights_per_filter] {
                *weight *= scale;
            }
        }
    }

    Ok((folded_biases, folded_weights))
}

pub(super) fn load_weights_into_params(
    network: &Network,
    params: &mut [LayerParams],
    path: impl AsRef<Path>,
) -> Result<WeightsHeader> {
    let mut file = File::open(path)?;
    let header = read_header(&mut file)?;
    let transpose_connected = header.major > 1000 || header.minor > 1000;

    for (layer, params) in network.layers.iter().zip(params.iter_mut()) {
        match params {
            LayerParams::None => {}
            LayerParams::HyperbolicFoveation(hf) => {
                let LayerKind::HyperbolicFoveation(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} hyperbolic-foveation parameters do not match layer kind",
                        layer.index
                    )));
                };
                validate_hyperbolic_foveation_param_layout(spec, hf)?;
                read_f32_into(&mut file, &mut hf.hf_params)?;
                if spec.pred_hidden > 0 {
                    read_f32_into(&mut file, &mut hf.pred_w1)?;
                    read_f32_into(&mut file, &mut hf.pred_b1)?;
                    read_f32_into(&mut file, &mut hf.pred_w2)?;
                    read_f32_into(&mut file, &mut hf.pred_b2)?;
                }
            }
            LayerParams::FoveationFilm(film) => {
                let LayerKind::FoveationFilm(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} foveation-film parameters do not match layer kind",
                        layer.index
                    )));
                };
                validate_foveation_film_param_layout(spec, layer.input_shape.channels, film)?;
                read_f32_into(&mut file, &mut film.w_embed)?;
                read_f32_into(&mut file, &mut film.b_embed)?;
                read_f32_into(&mut file, &mut film.w_gamma)?;
                read_f32_into(&mut file, &mut film.b_gamma)?;
                read_f32_into(&mut file, &mut film.w_beta)?;
                read_f32_into(&mut file, &mut film.b_beta)?;
            }
            LayerParams::RecursiveBlock(recursive) => {
                let LayerKind::RecursiveBlock(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} recursive-block parameters do not match layer kind",
                        layer.index
                    )));
                };
                read_recursive_body_weights(&mut file, &spec.body, &mut recursive.body)?;
            }
            LayerParams::BatchNorm(batch_norm) => {
                read_f32_into(&mut file, &mut batch_norm.biases)?;
                read_f32_into(&mut file, &mut batch_norm.scales)?;
                read_f32_into(&mut file, &mut batch_norm.rolling_mean)?;
                read_f32_into(&mut file, &mut batch_norm.rolling_variance)?;
            }
            LayerParams::DepthAnything(weights) => {
                let LayerKind::DepthAnything(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} depth_anything parameters do not match layer kind",
                        layer.index
                    )));
                };
                if spec.weights_path.is_none() {
                    let geometry = spec.geometry(layer.input_shape);
                    if !weights.matches(&geometry) {
                        return Err(Error::Weights(format!(
                            "layer #{} ([depth_anything]) parameter layout does not match config",
                            layer.index
                        )));
                    }
                    read_f32_into(&mut file, weights.as_mut_slice())?;
                } else if spec.external_weights.is_none() {
                    return Err(Error::Weights(format!(
                        "layer #{} ([depth_anything]) has a weights path but no loaded external blob",
                        layer.index
                    )));
                }
            }
            LayerParams::DptHead(weights) => {
                let LayerKind::DptHead(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} [dpt_head] parameters do not match layer kind",
                        layer.index
                    )));
                };
                let expected =
                    crate::dpt_head::dpt_head_weight_layout(spec.geometry(layer.input_shape))?
                        .total;
                if weights.values.len() != expected {
                    return Err(Error::Weights(format!(
                        "layer #{} ([dpt_head]) parameter layout does not match config",
                        layer.index
                    )));
                }
                read_f32_into(&mut file, &mut weights.values)?;
            }
            // The scorer blob does **not** live in the sequential stream: C++
            // seeks past the encoder and head blobs with `laya_open_weights`
            // (`sc_blob_offset = 20 + skip_floats * 4`) and then calls
            // `load_laya_scorer_weights`.  Reaching it here means a plain
            // sequential reader would read the wrong bytes, so it is an error
            // rather than a silent misread; use `load_laya_scorer_blob`.
            LayerParams::LayaScorer(_) => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([laya_scorer]) must be loaded with `load_laya_scorer_blob`, \
                     which seeks to its offset in the shared .weights file",
                    layer.index
                )));
            }
            LayerParams::LayaActHead(_) => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([laya_act_head]) must be loaded with `load_laya_act_head_blob`, \
                     which seeks to its offset in the shared .weights file",
                    layer.index
                )));
            }
            LayerParams::LayaEncoder(_) => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([laya_encoder]) must be loaded with `load_laya_encoder_blob`, \
                     which seeks to its offset in the shared .weights file",
                    layer.index
                )));
            }
            LayerParams::LayaHead(_) => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ([laya_head]) must be loaded with `load_laya_head_blob`, \
                     which seeks to its offset in the shared .weights file",
                    layer.index
                )));
            }
            LayerParams::Convolutional(conv) => {
                read_f32_into(&mut file, &mut conv.biases)?;
                if let Some(batch_norm) = conv.batch_norm.as_mut() {
                    read_f32_into(&mut file, &mut batch_norm.scales)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_mean)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_variance)?;
                }
                read_f32_into(&mut file, &mut conv.weights)?;
            }
            LayerParams::GraphConv(graph) => {
                let LayerKind::GraphConv(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} graph_conv parameters do not match layer kind",
                        layer.index
                    )));
                };
                read_convolutional_params(
                    &mut file,
                    &mut graph.projection,
                    layer.input_shape.channels,
                    spec.filters,
                    1,
                    spec.groups,
                )?;
                match (spec.use_self, &mut graph.self_weights) {
                    (true, Some(weights)) => read_f32_into(&mut file, weights)?,
                    (false, None) => {}
                    _ => {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} graph_conv self-weight layout does not match config",
                            layer.index
                        )));
                    }
                }
                match (
                    spec.edge_mode,
                    &mut graph.edge_kernel,
                    &mut graph.edge_biases,
                ) {
                    (0, None, None) => {}
                    (1, Some(kernel), Some(biases)) => {
                        read_f32_into(&mut file, kernel)?;
                        read_f32_into(&mut file, biases)?;
                    }
                    _ => {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} graph_conv edge-parameter layout does not match config",
                            layer.index
                        )));
                    }
                }
            }
            LayerParams::Connected(connected) => {
                if matches!(&layer.kind, LayerKind::ObbRefine(_)) {
                    if connected.batch_norm.is_some() {
                        return Err(Error::Weights(format!(
                            "layer #{} ([obb_refine]) cannot have batch normalization parameters",
                            layer.index
                        )));
                    }
                    read_f32_into(&mut file, &mut connected.biases)?;
                    read_f32_into(&mut file, &mut connected.weights)?;
                } else {
                    read_connected_params(
                        &mut file,
                        connected,
                        layer.input_shape.element_count(),
                        connected.biases.len(),
                        transpose_connected,
                    )?;
                }
            }
            LayerParams::Local(local) => {
                read_f32_into(&mut file, &mut local.biases)?;
                read_f32_into(&mut file, &mut local.weights)?;
            }
            // C++ `load_moe_convolutional_weights` reads the experts in order
            // and finishes with the router biases and weights, matching
            // `Layer::weights()`.
            LayerParams::Moe(moe) => {
                for expert in moe.experts.iter_mut() {
                    read_f32_into(&mut file, &mut expert.biases)?;
                    if let Some(batch_norm) = expert.batch_norm.as_mut() {
                        read_f32_into(&mut file, &mut batch_norm.scales)?;
                        read_f32_into(&mut file, &mut batch_norm.rolling_mean)?;
                        read_f32_into(&mut file, &mut batch_norm.rolling_variance)?;
                    }
                    read_f32_into(&mut file, &mut expert.weights)?;
                }
                read_f32_into(&mut file, &mut moe.router_biases)?;
                read_f32_into(&mut file, &mut moe.router_weights)?;
            }
            LayerParams::EmlConvolutional(eml) => {
                let LayerKind::EmlConvolutional(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} EML parameters do not match layer kind",
                        layer.index
                    )));
                };
                let branch = &spec.branch;
                for params in [&mut eml.input, &mut eml.self_params] {
                    read_convolutional_params(
                        &mut file,
                        params,
                        layer.input_shape.channels,
                        branch.filters,
                        branch.size,
                        branch.groups,
                    )?;
                }
            }
            LayerParams::Dcnv4(dcnv4) => {
                read_f32_into(&mut file, &mut dcnv4.biases)?;
                if let Some(batch_norm) = dcnv4.batch_norm.as_mut() {
                    read_f32_into(&mut file, &mut batch_norm.scales)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_mean)?;
                    read_f32_into(&mut file, &mut batch_norm.rolling_variance)?;
                }
                read_f32_into(&mut file, &mut dcnv4.dummy_weights)?;
                read_f32_into(&mut file, &mut dcnv4.offset_weights)?;
                read_f32_into(&mut file, &mut dcnv4.offset_biases)?;
            }
            LayerParams::DsConv(dsconv) => {
                let LayerKind::DsConv(_) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} dsconv parameters do not match layer kind",
                        layer.index
                    )));
                };
                read_f32_into(&mut file, &mut dsconv.collapse.biases)?;
                read_f32_into(&mut file, &mut dsconv.collapse.weights)?;
                read_f32_into(&mut file, &mut dsconv.offset_weights)?;
                read_f32_into(&mut file, &mut dsconv.offset_biases)?;
                for block in &mut dsconv.extra_offset_weights { read_f32_into(&mut file, block)?; }
            }
            LayerParams::Wmhf(wmhf) => {
                let LayerKind::Wmhf(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} [wmhf] parameters do not match layer kind",
                        layer.index
                    )));
                };
                for (conv, input_channels, filters, size, groups) in [
                    (
                        &mut wmhf.pre,
                        layer.input_shape.channels,
                        spec.filters,
                        1,
                        1,
                    ),
                    (
                        &mut wmhf.local_dw3,
                        spec.local_channels,
                        spec.local_channels,
                        3,
                        spec.local_channels,
                    ),
                    (
                        &mut wmhf.local_dw5,
                        spec.local_channels,
                        spec.local_channels,
                        5,
                        spec.local_channels,
                    ),
                    (
                        &mut wmhf.local_dw7,
                        spec.local_channels,
                        spec.local_channels,
                        7,
                        spec.local_channels,
                    ),
                    (
                        &mut wmhf.local_mix,
                        3 * spec.local_channels,
                        spec.local_channels,
                        1,
                        1,
                    ),
                    (&mut wmhf.hf_gate, spec.global_channels, spec.filters, 1, 1),
                    (&mut wmhf.fuse, spec.filters, spec.filters, 1, 1),
                ] {
                    read_convolutional_params(
                        &mut file,
                        conv,
                        input_channels,
                        filters,
                        size,
                        groups,
                    )?;
                }
                read_f32_into(&mut file, &mut wmhf.scan_weights)?;
            }
            LayerParams::Clifford(clifford) => {
                let LayerKind::Clifford(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} [clifford] parameters do not match layer kind",
                        layer.index
                    )));
                };
                read_f32_into(&mut file, &mut clifford.layer_norm_gamma)?;
                read_f32_into(&mut file, &mut clifford.layer_norm_beta)?;
                read_f32_into(&mut file, &mut clifford.determinant_weights)?;
                read_f32_into(&mut file, &mut clifford.determinant_biases)?;
                for conv in &mut clifford.depthwise {
                    read_convolutional_params(
                        &mut file,
                        conv,
                        layer.input_shape.channels,
                        layer.input_shape.channels,
                        spec.dwconv_size,
                        layer.input_shape.channels,
                    )?;
                }
                read_clifford_branch(&mut file, &mut clifford.local)?;
                read_f32_into(&mut file, &mut clifford.layer_scale)?;
                match (
                    spec.gffn_mode != crate::model::CliffordGffnMode::Local,
                    &mut clifford.global,
                ) {
                    (true, Some(global)) => read_clifford_branch(&mut file, global)?,
                    (false, None) => {}
                    _ => {
                        return Err(Error::InvalidConfig(format!(
                            "layer #{} [clifford] global parameter layout does not match config",
                            layer.index
                        )));
                    }
                }
                if spec.gffn_mode != crate::model::CliffordGffnMode::Local
                    && spec.global_op == crate::model::CliffordGlobalOp::Fourier
                {
                    let fourier = clifford.fourier.as_mut().ok_or_else(|| {
                        Error::InvalidConfig(format!(
                            "layer #{} [clifford] Fourier parameters are missing",
                            layer.index
                        ))
                    })?;
                    read_f32_into(&mut file, &mut fourier.real)?;
                    read_f32_into(&mut file, &mut fourier.imaginary)?;
                } else if clifford.fourier.is_some() {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} [clifford] has unexpected Fourier parameters",
                        layer.index
                    )));
                }
            }
            LayerParams::DeformConv(deform) => {
                let LayerKind::DeformConv(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} deform_conv parameters do not match layer kind",
                        layer.index
                    )));
                };
                read_convolutional_params(
                    &mut file,
                    &mut deform.main,
                    layer.input_shape.channels,
                    spec.filters,
                    spec.size,
                    spec.groups,
                )?;
                read_f32_into(&mut file, &mut deform.offset_weights)?;
                read_f32_into(&mut file, &mut deform.offset_biases)?;
                match (&mut deform.mask_weights, &mut deform.mask_biases) {
                    (Some(weights), Some(biases)) => {
                        read_f32_into(&mut file, weights)?;
                        read_f32_into(&mut file, biases)?;
                    }
                    (None, None) => {}
                    _ => {
                        return Err(Error::InvalidConfig(
                            "deform_conv mask parameter mismatch while loading weights".to_owned(),
                        ));
                    }
                }
            }
            LayerParams::Shortcut(shortcut) => {
                read_f32_into(&mut file, &mut shortcut.weights)?;
            }
            LayerParams::Gru(gru) => {
                let inputs = layer.input_shape.element_count();
                let output = gru.input_z.biases.len();
                for (params, params_inputs) in [
                    (&mut gru.input_z, inputs),
                    (&mut gru.state_z, output),
                    (&mut gru.input_r, inputs),
                    (&mut gru.state_r, output),
                    (&mut gru.input_h, inputs),
                    (&mut gru.state_h, output),
                ] {
                    read_connected_params(
                        &mut file,
                        params,
                        params_inputs,
                        output,
                        transpose_connected,
                    )?;
                }
            }
            LayerParams::Rnn(recurrent) => {
                let inputs = layer.input_shape.element_count();
                let hidden = recurrent.input.biases.len();
                let output = recurrent.output.biases.len();
                read_connected_params(
                    &mut file,
                    &mut recurrent.input,
                    inputs,
                    hidden,
                    transpose_connected,
                )?;
                read_connected_params(
                    &mut file,
                    &mut recurrent.self_params,
                    hidden,
                    hidden,
                    transpose_connected,
                )?;
                read_connected_params(
                    &mut file,
                    &mut recurrent.output,
                    hidden,
                    output,
                    transpose_connected,
                )?;
            }
            LayerParams::Lstm(recurrent) => {
                let inputs = layer.input_shape.element_count();
                let output = recurrent.wf.biases.len();
                for (params, params_inputs) in [
                    (&mut recurrent.wf, output),
                    (&mut recurrent.wi, output),
                    (&mut recurrent.wg, output),
                    (&mut recurrent.wo, output),
                    (&mut recurrent.uf, inputs),
                    (&mut recurrent.ui, inputs),
                    (&mut recurrent.ug, inputs),
                    (&mut recurrent.uo, inputs),
                ] {
                    read_connected_params(
                        &mut file,
                        params,
                        params_inputs,
                        output,
                        transpose_connected,
                    )?;
                }
            }
            LayerParams::Crnn(recurrent) => {
                let input_channels = layer.input_shape.channels;
                let hidden = recurrent.self_params.biases.len();
                let output = recurrent.output.biases.len();
                let size = match &layer.kind {
                    LayerKind::Crnn(spec) => spec.size,
                    _ => 1,
                };
                let groups = match &layer.kind {
                    LayerKind::Crnn(spec) => spec.groups,
                    _ => 1,
                };
                read_convolutional_params(
                    &mut file,
                    &mut recurrent.input,
                    input_channels,
                    hidden,
                    size,
                    groups,
                )?;
                read_convolutional_params(
                    &mut file,
                    &mut recurrent.self_params,
                    hidden,
                    hidden,
                    size,
                    groups,
                )?;
                read_convolutional_params(
                    &mut file,
                    &mut recurrent.output,
                    hidden,
                    output,
                    size,
                    groups,
                )?;
            }
            LayerParams::Vit(weights) => {
                let LayerKind::Vit(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} ViT weights do not match layer kind",
                        layer.index
                    )));
                };
                let geometry = vit_geometry(layer, spec);
                let has_mhc = (header.major, header.minor, header.revision)
                    >= (
                        crate::DARKNET_WEIGHTS_VERSION_MAJOR,
                        crate::DARKNET_WEIGHTS_VERSION_MINOR,
                        crate::DARKNET_WEIGHTS_VERSION_PATCH,
                    );
                let count = crate::vit::vit_blob_float_count(
                    &geometry,
                    layer.output_shape.height,
                    layer.output_shape.width,
                    has_mhc,
                )
                .map_err(Error::Weights)?;
                let mut blob = vec![0.0; count];
                read_f32_into(&mut file, &mut blob)?;
                *weights = crate::vit::VitWeights::from_blob(
                    &blob,
                    &geometry,
                    layer.output_shape.height,
                    layer.output_shape.width,
                    has_mhc,
                )
                .map_err(Error::Weights)?;
            }
            LayerParams::Swin(weights) => {
                let LayerKind::Swin(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} Swin weights do not match layer kind",
                        layer.index
                    )));
                };
                let modern_header = (header.major, header.minor, header.revision)
                    >= (
                        crate::DARKNET_WEIGHTS_VERSION_MAJOR,
                        crate::DARKNET_WEIGHTS_VERSION_MINOR,
                        crate::DARKNET_WEIGHTS_VERSION_PATCH,
                    );
                if !modern_header {
                    return Err(Error::Weights(format!(
                        "layer #{} ([swin]) requires the Rust-defined modern weights layout",
                        layer.index
                    )));
                }
                let count = layer
                    .weights()
                    .ok_or_else(|| Error::Weights("Swin layer has no weight metadata".into()))?
                    .parameter_count();
                let mut blob = vec![0.0; count];
                read_f32_into(&mut file, &mut blob)?;
                *weights = crate::swin::SwinWeights::from_blob(
                    &blob,
                    layer.input_shape.channels,
                    spec.filters,
                    spec.heads,
                    spec.size,
                    spec.ffn_ratio,
                )
                .map_err(Error::Weights)?;
            }
            LayerParams::TuckerAttention(weights) => {
                let LayerKind::TuckerAttention(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} Tucker attention weights do not match layer kind",
                        layer.index
                    )));
                };
                let count = layer
                    .weights()
                    .ok_or_else(|| {
                        Error::Weights("Tucker attention layer has no weight metadata".into())
                    })?
                    .parameter_count();
                let mut blob = vec![0.0; count];
                read_f32_into(&mut file, &mut blob)?;
                *weights = crate::tucker_attention::TuckerAttentionParams::from_blob(
                    &blob,
                    layer.input_shape.channels,
                    spec.heads,
                    spec.rank_q,
                    spec.rank_k,
                    spec.rank_v,
                    spec.rank_o,
                )
                .map_err(Error::Weights)?;
            }
            LayerParams::MambaVision(weights) => {
                let LayerKind::MambaVision(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} MambaVision weights do not match layer kind",
                        layer.index
                    )));
                };
                let count = layer
                    .weights()
                    .ok_or_else(|| {
                        Error::Weights("MambaVision layer has no weight metadata".into())
                    })?
                    .parameter_count();
                let mut blob = vec![0.0; count];
                read_f32_into(&mut file, &mut blob)?;
                *weights = crate::mambavision::MambaVisionParams::from_blob(
                    &blob,
                    layer.input_shape.channels,
                    spec.filters,
                    spec.state,
                    spec.conv_size,
                    spec.dt_rank,
                    spec.ffn_ratio,
                )
                .map_err(Error::Weights)?;
            }
            LayerParams::KimiKda(weights) => {
                let LayerKind::KimiKda(spec) = &layer.kind else {
                    return Err(Error::InvalidConfig(format!(
                        "layer #{} Kimi KDA weights do not match layer kind",
                        layer.index
                    )));
                };
                let count = layer
                    .weights()
                    .ok_or_else(|| Error::Weights("Kimi KDA layer has no weight metadata".into()))?
                    .parameter_count();
                let mut blob = vec![0.0; count];
                read_f32_into(&mut file, &mut blob)?;
                *weights = crate::kimi_kda::KimiKdaParams::from_blob(
                    &blob,
                    layer.input_shape.channels,
                    spec.rank,
                    spec.experts,
                    spec.hidden,
                )
                .map_err(Error::Weights)?;
            }
        }
    }

    Ok(header)
}

pub(crate) fn tensor_backend_shape(shape: TensorShape) -> [usize; 4] {
    [shape.batch, shape.channels, shape.height, shape.width]
}

pub(super) fn layer_uses_previous_input(kind: &LayerKind) -> bool {
    !matches!(
        kind,
        LayerKind::Route(_)
            | LayerKind::ChannelSlice(_)
            | LayerKind::LayaMarkerGather(_)
            | LayerKind::LayaActHead(_)
    )
}

pub(super) fn gpu_layer_output_use_counts(network: &Network) -> Result<Vec<usize>> {
    let mut use_counts = vec![0usize; network.layers.len()];
    for layer in &network.layers {
        if layer.index > 0 && layer_uses_previous_input(&layer.kind) {
            use_counts[layer.index - 1] += 1;
        }

        match &layer.kind {
            LayerKind::Route(spec) => {
                for raw_index in &spec.layers {
                    let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                    use_counts[source_index] += 1;
                }
            }
            LayerKind::Shortcut(spec) => {
                for raw_index in &spec.from {
                    let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                    use_counts[source_index] += 1;
                }
            }
            LayerKind::ChannelSlice(spec) => {
                let source_index = resolve_relative_layer(layer.index, spec.from)?;
                use_counts[source_index] += 1;
            }
            LayerKind::ScaleChannels(spec) => {
                let source_index = resolve_relative_layer(layer.index, spec.from)?;
                use_counts[source_index] += 1;
            }
            LayerKind::Sam(spec) => {
                let source_index = resolve_relative_layer(layer.index, spec.from)?;
                use_counts[source_index] += 1;
            }
            LayerKind::LayaMarkerGather(spec) => {
                for raw_index in &spec.layers {
                    let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                    use_counts[source_index] += 1;
                }
            }
            LayerKind::LayaActHead(spec) => {
                for raw_index in &spec.layers {
                    let source_index = resolve_relative_layer(layer.index, *raw_index)?;
                    use_counts[source_index] += 1;
                }
            }
            _ => {}
        }
    }

    Ok(use_counts)
}

pub(super) fn release_gpu_source<T: Clone>(
    source_index: usize,
    remaining_uses: &mut [usize],
    gpu_layer_outputs: &mut [Option<T>],
) -> Result<()> {
    let remaining = remaining_uses.get_mut(source_index).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "missing GPU use-count entry for source layer #{source_index}"
        ))
    })?;
    if *remaining == 0 {
        return Err(Error::InvalidConfig(format!(
            "GPU source layer #{source_index} was released too early"
        )));
    }
    *remaining -= 1;
    if *remaining == 0 {
        gpu_layer_outputs[source_index] = None;
    }
    Ok(())
}

pub(super) fn release_gpu_sources_for_layer<T: Clone>(
    layer_index: usize,
    kind: &LayerKind,
    remaining_uses: &mut [usize],
    gpu_layer_outputs: &mut [Option<T>],
) -> Result<()> {
    if layer_index > 0 && layer_uses_previous_input(kind) {
        release_gpu_source(layer_index - 1, remaining_uses, gpu_layer_outputs)?;
    }

    match kind {
        LayerKind::Route(spec) => {
            for raw_index in &spec.layers {
                let source_index = resolve_relative_layer(layer_index, *raw_index)?;
                release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
            }
        }
        LayerKind::Shortcut(spec) => {
            for raw_index in &spec.from {
                let source_index = resolve_relative_layer(layer_index, *raw_index)?;
                release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
            }
        }
        LayerKind::ChannelSlice(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
        }
        LayerKind::ScaleChannels(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
        }
        LayerKind::Sam(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
        }
        LayerKind::LayaMarkerGather(spec) => {
            for raw_index in &spec.layers {
                let source_index = resolve_relative_layer(layer_index, *raw_index)?;
                release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
            }
        }
        LayerKind::LayaActHead(spec) => {
            for raw_index in &spec.layers {
                let source_index = resolve_relative_layer(layer_index, *raw_index)?;
                release_gpu_source(source_index, remaining_uses, gpu_layer_outputs)?;
            }
        }
        _ => {}
    }

    Ok(())
}

pub(crate) fn gpu_activation_kind(layer_index: usize, activation: &Activation) -> Result<F32ActivationKind> {
    activation.to_cuda_f32_kind().ok_or_else(|| {
        Error::InvalidConfig(format!(
            "layer #{layer_index} uses activation {:?}, which the GPU f32 backend does not support",
            activation
        ))
    })
}

pub(crate) fn backend_error(err_context: &str, err: impl std::fmt::Display) -> Error {
    Error::Backend(format!("{err_context}: {err}"))
}

pub(super) fn read_header(file: &mut File) -> Result<WeightsHeader> {
    let major = read_i32(file)?;
    let minor = read_i32(file)?;
    let revision = read_i32(file)?;
    let (seen, header_bytes) = if (major * 10 + minor) >= 2 {
        (read_u64(file)?, 3 * 4 + 8)
    } else {
        (u64::from(read_u32(file)?), 3 * 4 + 4)
    };

    Ok(WeightsHeader {
        major,
        minor,
        revision,
        seen,
        header_bytes,
    })
}

pub(super) fn read_f32_into(file: &mut File, dst: &mut [f32]) -> Result<()> {
    let mut bytes = vec![0u8; dst.len() * 4];
    file.read_exact(&mut bytes)?;
    for (chunk, value) in bytes.chunks_exact(4).zip(dst.iter_mut()) {
        *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

pub(super) fn read_i32(file: &mut File) -> Result<i32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

pub(super) fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

pub(super) fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

pub(super) fn read_recursive_body_weights(
    file: &mut File,
    body: &[crate::model::Layer],
    params: &mut [LayerParams],
) -> Result<()> {
    if body.len() != params.len() {
        return Err(Error::InvalidConfig(
            "recursive-block body weight parameter count mismatch".to_owned(),
        ));
    }
    for (layer, params) in body.iter().zip(params) {
        match (&layer.kind, params) {
            (LayerKind::Convolutional(_), LayerParams::Convolutional(conv)) => {
                read_f32_into(file, &mut conv.biases)?;
                if let Some(batch_norm) = conv.batch_norm.as_mut() {
                    read_f32_into(file, &mut batch_norm.scales)?;
                    read_f32_into(file, &mut batch_norm.rolling_mean)?;
                    read_f32_into(file, &mut batch_norm.rolling_variance)?;
                }
                read_f32_into(file, &mut conv.weights)?;
            }
            (LayerKind::Connected(_), LayerParams::Connected(connected)) => {
                read_connected_params(
                    file,
                    connected,
                    layer.input_shape.element_count(),
                    connected.biases.len(),
                    false,
                )?;
            }
            (
                LayerKind::MaxPool(_)
                | LayerKind::LocalAvgPool(_)
                | LayerKind::AvgPool
                | LayerKind::Upsample(_)
                | LayerKind::ChannelShuffle(_)
                | LayerKind::Dropout { .. },
                LayerParams::None,
            ) => {}
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "unsupported recursive-block weight layout for body layer #{} ({})",
                    layer.index,
                    layer.kind.name()
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn read_connected_params(
    file: &mut File,
    params: &mut ConnectedParams,
    inputs: usize,
    outputs: usize,
    transpose: bool,
) -> Result<()> {
    if params.biases.len() != outputs || params.weights.len() != inputs * outputs {
        return Err(Error::InvalidConfig(
            "connected recurrent weight shape mismatch".to_owned(),
        ));
    }
    read_f32_into(file, &mut params.biases)?;
    read_f32_into(file, &mut params.weights)?;
    if transpose {
        let mut transposed = vec![0.0; params.weights.len()];
        for input in 0..inputs {
            for output in 0..outputs {
                transposed[output * inputs + input] = params.weights[input * outputs + output];
            }
        }
        params.weights = transposed;
    }
    if let Some(batch_norm) = params.batch_norm.as_mut() {
        read_f32_into(file, &mut batch_norm.scales)?;
        read_f32_into(file, &mut batch_norm.rolling_mean)?;
        read_f32_into(file, &mut batch_norm.rolling_variance)?;
    }
    Ok(())
}

pub(super) fn read_convolutional_params(
    file: &mut File,
    params: &mut ConvolutionalParams,
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
) -> Result<()> {
    let expected = filters * (input_channels / groups.max(1)) * size * size;
    if params.biases.len() != filters || params.weights.len() != expected {
        return Err(Error::InvalidConfig(
            "convolutional recurrent weight shape mismatch".to_owned(),
        ));
    }
    read_f32_into(file, &mut params.biases)?;
    if let Some(batch_norm) = params.batch_norm.as_mut() {
        read_f32_into(file, &mut batch_norm.scales)?;
        read_f32_into(file, &mut batch_norm.rolling_mean)?;
        read_f32_into(file, &mut batch_norm.rolling_variance)?;
    }
    read_f32_into(file, &mut params.weights)?;
    Ok(())
}

pub(super) fn read_clifford_branch(file: &mut File, branch: &mut CliffordBranchParams) -> Result<()> {
    read_f32_into(file, &mut branch.projection_weights)?;
    read_f32_into(file, &mut branch.projection_biases)?;
    read_f32_into(file, &mut branch.gate_weights)?;
    read_f32_into(file, &mut branch.gate_biases)
}

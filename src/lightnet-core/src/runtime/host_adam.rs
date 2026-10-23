use super::*;

/// Adam moments of the host-resident parameters of one layer (connected).
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct HostAdamLayerState {
    pub(super) weights: Option<(Vec<f32>, Vec<f32>)>,
    pub(super) biases: Option<(Vec<f32>, Vec<f32>)>,
    pub(super) scales: Option<(Vec<f32>, Vec<f32>)>,
    pub(super) children: Vec<HostAdamLayerState>,
}

pub(super) fn connected_parameter_count(params: &ConnectedParams) -> usize {
    params.weights.len()
        + params.biases.len()
        + params
            .batch_norm
            .as_ref()
            .map_or(0, |bn| bn.scales.len() + bn.biases.len())
}

pub(super) fn convolutional_parameter_count(params: &ConvolutionalParams) -> usize {
    params.weights.len()
        + params.biases.len()
        + params
            .batch_norm
            .as_ref()
            .map_or(0, |bn| bn.scales.len() + bn.biases.len())
}

pub(super) fn host_adam_scalar_capacity(params: &LayerParams) -> usize {
    match params {
        LayerParams::RecursiveBlock(params) => {
            params.controller_weights.len()
                + params.controller_biases.len()
                + params.body.len()
                + params
                    .body
                    .iter()
                    .map(host_adam_scalar_capacity)
                    .sum::<usize>()
        }
        LayerParams::FoveationFilm(params) => {
            params.w_embed.len()
                + params.b_embed.len()
                + params.w_gamma.len()
                + params.b_gamma.len()
                + params.w_beta.len()
                + params.b_beta.len()
        }
        LayerParams::HyperbolicFoveation(params) => {
            if params.pred_b1.is_empty() {
                params.hf_params.len()
            } else {
                params.pred_w1.len()
                    + params.pred_b1.len()
                    + params.pred_w2.len()
                    + params.pred_b2.len()
            }
        }
        LayerParams::BatchNorm(params) => params.scales.len() + params.biases.len(),
        LayerParams::Convolutional(params) => convolutional_parameter_count(params),
        LayerParams::Connected(params) => connected_parameter_count(params),
        LayerParams::Local(params) => params.weights.len() + params.biases.len(),
        LayerParams::Shortcut(params) => params.weights.len(),
        LayerParams::Gru(params) => [
            &params.input_z,
            &params.state_z,
            &params.input_r,
            &params.state_r,
            &params.input_h,
            &params.state_h,
        ]
        .into_iter()
        .map(connected_parameter_count)
        .sum(),
        LayerParams::Rnn(params) => [&params.input, &params.self_params, &params.output]
            .into_iter()
            .map(connected_parameter_count)
            .sum(),
        LayerParams::Lstm(params) => [
            &params.wf, &params.wi, &params.wg, &params.wo, &params.uf, &params.ui, &params.ug,
            &params.uo,
        ]
        .into_iter()
        .map(connected_parameter_count)
        .sum(),
        LayerParams::Crnn(params) => [&params.input, &params.self_params, &params.output]
            .into_iter()
            .map(convolutional_parameter_count)
            .sum(),
        LayerParams::Vit(params) => [
            params.patch_embed.len(),
            params.patch_bias.len(),
            params.qkv_bias.len(),
            params.ln1_gamma.len(),
            params.ln1_beta.len(),
            params.qkv_weight.len(),
            params.wo.len(),
            params.wo_bias.len(),
            params.ln2_gamma.len(),
            params.ln2_beta.len(),
            params.ffn_w1.len(),
            params.ffn_b1.len(),
            params.ffn_w2.len(),
            params.ffn_b2.len(),
            params.pos_embed.len(),
            params.mhc.len(),
        ]
        .into_iter()
        .sum(),
        LayerParams::Swin(params) => params.parameter_count(),
        LayerParams::TuckerAttention(params) => params.parameter_count(),
        LayerParams::MambaVision(params) => params.parameter_count(),
        LayerParams::KimiKda(params) => params.parameter_count(),
        LayerParams::DptHead(params) => params.values.len(),
        LayerParams::Moe(params) => {
            params.router_weights.len()
                + params.router_biases.len()
                + params
                    .experts
                    .iter()
                    .map(convolutional_parameter_count)
                    .sum::<usize>()
        }
        LayerParams::EmlConvolutional(params) => {
            convolutional_parameter_count(&params.input)
                + convolutional_parameter_count(&params.self_params)
        }
        LayerParams::GraphConv(params) => {
            convolutional_parameter_count(&params.projection)
                + params.self_weights.as_ref().map_or(0, Vec::len)
                + params.edge_kernel.as_ref().map_or(0, Vec::len)
                + params.edge_biases.as_ref().map_or(0, Vec::len)
        }
        LayerParams::Dcnv4(params) => {
            params.biases.len()
                + params.offset_weights.len()
                + params.offset_biases.len()
                + params
                    .batch_norm
                    .as_ref()
                    .map_or(0, |bn| bn.scales.len() + bn.biases.len())
        }
        LayerParams::DeformConv(params) => {
            convolutional_parameter_count(&params.main)
                + params.offset_weights.len()
                + params.offset_biases.len()
                + params.mask_weights.as_ref().map_or(0, Vec::len)
                + params.mask_biases.as_ref().map_or(0, Vec::len)
        }
        LayerParams::DsConv(params) => {
            convolutional_parameter_count(&params.collapse)
                + params.offset_weights.len()
                + params.offset_biases.len()
        }
        LayerParams::Wmhf(params) => {
            convolutional_parameter_count(&params.pre)
                + convolutional_parameter_count(&params.local_dw3)
                + convolutional_parameter_count(&params.local_dw5)
                + convolutional_parameter_count(&params.local_dw7)
                + convolutional_parameter_count(&params.local_mix)
                + convolutional_parameter_count(&params.hf_gate)
                + convolutional_parameter_count(&params.fuse)
                + params.scan_weights.len()
        }
        LayerParams::Clifford(params) => {
            params.layer_norm_gamma.len()
                + params.layer_norm_beta.len()
                + params.determinant_weights.len()
                + params.determinant_biases.len()
                + params
                    .depthwise
                    .iter()
                    .map(convolutional_parameter_count)
                    .sum::<usize>()
                + params.local.projection_weights.len()
                + params.local.projection_biases.len()
                + params.local.gate_weights.len()
                + params.local.gate_biases.len()
                + params.layer_scale.len()
                + params.global.as_ref().map_or(0, |global| {
                    global.projection_weights.len()
                        + global.projection_biases.len()
                        + global.gate_weights.len()
                        + global.gate_biases.len()
                })
                + params
                    .fourier
                    .as_ref()
                    .map_or(0, |fno| fno.real.len() + fno.imaginary.len())
        }
        LayerParams::None
        | LayerParams::LayaScorer(_)
        | LayerParams::LayaActHead(_)
        | LayerParams::LayaEncoder(_)
        | LayerParams::LayaHead(_)
        | LayerParams::DepthAnything(_) => 0,
    }
}

pub(super) fn validate_host_adam_slot(
    name: &str,
    moments: &Option<(Vec<f32>, Vec<f32>)>,
    expected_len: usize,
) -> Result<()> {
    match moments {
        Some((first, second)) if first.len() == expected_len && second.len() == expected_len => {
            Ok(())
        }
        Some((first, second)) => Err(Error::Weights(format!(
            "recursive-block Adam {name} moments have lengths {}/{}, expected {expected_len}",
            first.len(),
            second.len()
        ))),
        None if expected_len == 0 => Ok(()),
        None => Err(Error::Weights(format!(
            "recursive-block Adam {name} moments are missing (expected {expected_len})"
        ))),
    }
}

pub(super) fn validate_recursive_body_adam_state(
    layer: &crate::model::Layer,
    params: &LayerParams,
    state: &HostAdamLayerState,
) -> Result<()> {
    if !state.children.is_empty() {
        return Err(Error::Weights(format!(
            "recursive-block Adam leaf layer #{} has unexpected child states",
            layer.index
        )));
    }
    match (&layer.kind, params) {
        (LayerKind::Convolutional(_), LayerParams::Convolutional(params)) => {
            validate_host_adam_slot(
                "body convolution weight",
                &state.weights,
                params.weights.len(),
            )?;
            validate_host_adam_slot("body convolution bias", &state.biases, params.biases.len())?;
            validate_host_adam_slot(
                "body convolution scale",
                &state.scales,
                params.batch_norm.as_ref().map_or(0, |bn| bn.scales.len()),
            )
        }
        (LayerKind::Connected(_), LayerParams::Connected(params)) => {
            validate_host_adam_slot(
                "body connected weight",
                &state.weights,
                params.weights.len(),
            )?;
            validate_host_adam_slot("body connected bias", &state.biases, params.biases.len())?;
            validate_host_adam_slot(
                "body connected scale",
                &state.scales,
                params.batch_norm.as_ref().map_or(0, |bn| bn.scales.len()),
            )
        }
        (
            LayerKind::MaxPool(_)
            | LayerKind::LocalAvgPool(_)
            | LayerKind::AvgPool
            | LayerKind::Upsample(_)
            | LayerKind::ChannelShuffle(_)
            | LayerKind::Dropout { .. },
            LayerParams::None,
        ) => {
            validate_host_adam_slot("body non-parameter weight", &state.weights, 0)?;
            validate_host_adam_slot("body non-parameter bias", &state.biases, 0)?;
            validate_host_adam_slot("body non-parameter scale", &state.scales, 0)
        }
        _ => Err(Error::Weights(format!(
            "recursive-block Adam state does not match body layer #{} ({})",
            layer.index,
            layer.kind.name()
        ))),
    }
}

pub(super) fn validate_hyperbolic_foveation_adam_slot(
    name: &str,
    moments: &Option<(Vec<f32>, Vec<f32>)>,
    expected_len: usize,
) -> Result<()> {
    match moments {
        Some((first, second)) if first.len() == expected_len && second.len() == expected_len => {
            Ok(())
        }
        Some((first, second)) => Err(Error::Weights(format!(
            "hyperbolic-foveation Adam {name} moments have lengths {}/{}, expected {expected_len}",
            first.len(),
            second.len()
        ))),
        None if expected_len == 0 => Ok(()),
        None => Err(Error::Weights(format!(
            "hyperbolic-foveation Adam {name} moments are missing (expected {expected_len})"
        ))),
    }
}

pub(super) fn validate_foveation_film_adam_state(
    layer: &crate::model::Layer,
    params: &LayerParams,
    state: &HostAdamLayerState,
) -> Result<()> {
    let (LayerKind::FoveationFilm(spec), LayerParams::FoveationFilm(params)) =
        (&layer.kind, params)
    else {
        return Err(Error::Weights(
            "[foveation_film] Adam state does not match layer parameters".to_owned(),
        ));
    };
    validate_foveation_film_param_layout(spec, layer.input_shape.channels, params)
        .map_err(|error| Error::Weights(error.to_string()))?;
    validate_host_adam_slot("embedding weight", &state.weights, params.w_embed.len())?;
    validate_host_adam_slot("embedding bias", &state.biases, params.b_embed.len())?;
    validate_host_adam_slot("projection scale", &state.scales, 0)?;
    if state.children.len() != 2 {
        return Err(Error::Weights(format!(
            "[foveation_film] Adam projection state has {} children, expected 2",
            state.children.len()
        )));
    }
    for (index, child) in state.children.iter().enumerate() {
        if !child.children.is_empty() || child.scales.is_some() {
            return Err(Error::Weights(format!(
                "[foveation_film] Adam projection child {index} has unexpected nested state"
            )));
        }
        let (weights, biases) = if index == 0 {
            (params.w_gamma.len(), params.b_gamma.len())
        } else {
            (params.w_beta.len(), params.b_beta.len())
        };
        validate_host_adam_slot("projection weight", &child.weights, weights)?;
        validate_host_adam_slot("projection bias", &child.biases, biases)?;
    }
    Ok(())
}

pub(super) fn validate_hyperbolic_foveation_adam_state(
    layer: &crate::model::Layer,
    params: &LayerParams,
    state: &HostAdamLayerState,
) -> Result<()> {
    let (LayerKind::HyperbolicFoveation(spec), LayerParams::HyperbolicFoveation(params)) =
        (&layer.kind, params)
    else {
        return Err(Error::Weights(
            "hyperbolic-foveation Adam state does not match layer parameters".to_owned(),
        ));
    };
    validate_hyperbolic_foveation_param_layout(spec, params)
        .map_err(|error| Error::Weights(error.to_string()))?;
    if spec.pred_hidden == 0 {
        validate_hyperbolic_foveation_adam_slot("weight", &state.weights, 0)?;
        validate_hyperbolic_foveation_adam_slot(
            "global parameter",
            &state.biases,
            params.hf_params.len(),
        )?;
        validate_hyperbolic_foveation_adam_slot("scale", &state.scales, 0)?;
        if !state.children.is_empty() {
            return Err(Error::Weights(
                "hyperbolic-foveation Adam state has unexpected predictor children".to_owned(),
            ));
        }
    } else {
        validate_hyperbolic_foveation_adam_slot("weight", &state.weights, 0)?;
        validate_hyperbolic_foveation_adam_slot("bias", &state.biases, 0)?;
        validate_hyperbolic_foveation_adam_slot("scale", &state.scales, 0)?;
        if state.children.len() != 2 {
            return Err(Error::Weights(format!(
                "hyperbolic-foveation Adam predictor has {} states, expected 2",
                state.children.len()
            )));
        }
        for (index, child) in state.children.iter().enumerate() {
            if !child.children.is_empty() || child.scales.is_some() {
                return Err(Error::Weights(format!(
                    "hyperbolic-foveation Adam predictor child {index} has unexpected nested state"
                )));
            }
            validate_hyperbolic_foveation_adam_slot(
                "predictor matrix",
                &child.weights,
                if index == 0 {
                    params.pred_w1.len()
                } else {
                    params.pred_w2.len()
                },
            )?;
            validate_hyperbolic_foveation_adam_slot(
                "predictor bias",
                &child.biases,
                if index == 0 {
                    params.pred_b1.len()
                } else {
                    params.pred_b2.len()
                },
            )?;
        }
    }
    Ok(())
}

pub(super) fn validate_dpt_head_adam_state(
    layer: &crate::model::Layer,
    params: &LayerParams,
    state: &HostAdamLayerState,
    muon_mode: bool,
) -> Result<()> {
    let (LayerKind::DptHead(spec), LayerParams::DptHead(params)) = (&layer.kind, params) else {
        return Err(Error::Weights(
            "DPT Adam state does not match layer parameters".to_owned(),
        ));
    };
    let layout = crate::dpt_head::dpt_head_weight_layout(spec.geometry(layer.input_shape))
        .map_err(|error| Error::Weights(error.to_string()))?;
    for (name, moments) in [
        ("top-level weights", &state.weights),
        ("top-level biases", &state.biases),
        ("top-level scales", &state.scales),
    ] {
        validate_host_adam_slot(name, moments, 0)?;
    }
    if state.children.len() > crate::dpt_head::DPT_WEIGHT_SLOTS {
        return Err(Error::Weights(format!(
            "DPT Adam state has {} child slots, expected at most {}",
            state.children.len(),
            crate::dpt_head::DPT_WEIGHT_SLOTS
        )));
    }
    let is_matrix = |slot: usize| crate::dpt_head::DPT_MUON_WEIGHT_SLOTS.contains(&slot);
    let is_bias = |slot: usize| crate::dpt_head::DPT_BIAS_SLOTS.contains(&slot);
    for slot in 0..crate::dpt_head::DPT_WEIGHT_SLOTS {
        let child = state.children.get(slot);
        let weight_len = if !muon_mode && is_matrix(slot) {
            layout.get(slot).len
        } else {
            0
        };
        let bias_len = if is_bias(slot) {
            layout.get(slot).len
        } else {
            0
        };
        validate_host_adam_slot(
            &format!("DPT slot {slot} weight"),
            &child.and_then(|child| child.weights.clone()),
            weight_len,
        )?;
        validate_host_adam_slot(
            &format!("DPT slot {slot} bias"),
            &child.and_then(|child| child.biases.clone()),
            bias_len,
        )?;
        validate_host_adam_slot(
            &format!("DPT slot {slot} scale"),
            &child.and_then(|child| child.scales.clone()),
            0,
        )?;
        if child.is_some_and(|child| !child.children.is_empty()) {
            return Err(Error::Weights(format!(
                "DPT Adam state slot {slot} unexpectedly contains nested children"
            )));
        }
    }
    if params.values.len() != layout.total {
        return Err(Error::Weights(
            "DPT Adam parameter layout does not match layer geometry".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn validate_recursive_block_adam_state(
    layer: &crate::model::Layer,
    params: &LayerParams,
    state: &HostAdamLayerState,
) -> Result<()> {
    let (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(params)) =
        (&layer.kind, params)
    else {
        return Err(Error::Weights(
            "recursive-block Adam state does not match layer parameters".to_owned(),
        ));
    };
    validate_host_adam_slot(
        "controller weight",
        &state.weights,
        if spec.ouroboros == 1 {
            params.controller_weights.len()
        } else {
            0
        },
    )?;
    validate_host_adam_slot(
        "controller bias",
        &state.biases,
        if spec.ouroboros == 1 {
            params.controller_biases.len()
        } else {
            0
        },
    )?;
    validate_host_adam_slot("controller scale", &state.scales, 0)?;
    if state.children.len() != spec.body.len() || params.body.len() != spec.body.len() {
        return Err(Error::Weights(format!(
            "recursive-block Adam body has {} states and {} parameters, expected {}",
            state.children.len(),
            params.body.len(),
            spec.body.len()
        )));
    }
    for ((body_layer, body_params), body_state) in
        spec.body.iter().zip(&params.body).zip(&state.children)
    {
        validate_recursive_body_adam_state(body_layer, body_params, body_state)?;
    }
    Ok(())
}

pub(super) fn write_host_adam_moments(
    file: &mut impl Write,
    moments: &Option<(Vec<f32>, Vec<f32>)>,
) -> Result<()> {
    let Some((first, second)) = moments else {
        file.write_all(&[0])?;
        return Ok(());
    };
    if first.len() != second.len() {
        return Err(Error::Weights("host Adam moment lengths differ".to_owned()));
    }
    file.write_all(&[1])?;
    file.write_all(&(first.len() as u64).to_le_bytes())?;
    crate::weights::write_f32_slice(file, first)?;
    crate::weights::write_f32_slice(file, second)?;
    Ok(())
}

pub(super) fn write_muon_layer_state(file: &mut impl Write, state: &MuonLayerState) -> Result<()> {
    file.write_all(&(state.matrices.len() as u64).to_le_bytes())?;
    for matrix in &state.matrices {
        if matrix.rows == 0
            || matrix.cols == 0
            || matrix.rows.checked_mul(matrix.cols) != Some(matrix.momentum.len())
            || matrix.momentum.iter().any(|value| !value.is_finite())
        {
            return Err(Error::Weights(
                "cannot save malformed or non-finite Muon momentum".to_owned(),
            ));
        }
        file.write_all(&(matrix.rows as u64).to_le_bytes())?;
        file.write_all(&(matrix.cols as u64).to_le_bytes())?;
        file.write_all(&(matrix.momentum.len() as u64).to_le_bytes())?;
        for value in &matrix.momentum {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

pub(super) fn read_muon_layer_state(file: &mut File, max_tensor_len: usize) -> Result<MuonLayerState> {
    let mut bytes8 = [0u8; 8];
    file.read_exact(&mut bytes8)?;
    let count = usize::try_from(u64::from_le_bytes(bytes8))
        .map_err(|_| Error::Weights("Muon matrix count is too large".to_owned()))?;
    if count == 0 || count > max_tensor_len {
        return Err(Error::Weights(
            "Muon matrix count is empty or exceeds layer parameter capacity".to_owned(),
        ));
    }
    let mut matrices = Vec::with_capacity(count);
    for _ in 0..count {
        file.read_exact(&mut bytes8)?;
        let rows = usize::try_from(u64::from_le_bytes(bytes8))
            .map_err(|_| Error::Weights("Muon matrix row count is too large".to_owned()))?;
        file.read_exact(&mut bytes8)?;
        let cols = usize::try_from(u64::from_le_bytes(bytes8))
            .map_err(|_| Error::Weights("Muon matrix column count is too large".to_owned()))?;
        file.read_exact(&mut bytes8)?;
        let len = usize::try_from(u64::from_le_bytes(bytes8))
            .map_err(|_| Error::Weights("Muon momentum length is too large".to_owned()))?;
        if rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(len) || len > max_tensor_len {
            return Err(Error::Weights(
                "Muon momentum shape is malformed or exceeds layer parameter capacity".to_owned(),
            ));
        }
        let mut momentum = Vec::with_capacity(len);
        for _ in 0..len {
            let mut bytes4 = [0u8; 4];
            file.read_exact(&mut bytes4)?;
            let value = f32::from_le_bytes(bytes4);
            if !value.is_finite() {
                return Err(Error::Weights(
                    "Muon momentum contains a non-finite value".to_owned(),
                ));
            }
            momentum.push(value);
        }
        matrices.push(MuonMatrixState {
            rows,
            cols,
            momentum,
        });
    }
    Ok(MuonLayerState { matrices })
}

pub(super) fn write_host_adam_state(file: &mut impl Write, state: &HostAdamLayerState) -> Result<()> {
    write_host_adam_moments(file, &state.weights)?;
    write_host_adam_moments(file, &state.biases)?;
    write_host_adam_moments(file, &state.scales)?;
    file.write_all(&(state.children.len() as u64).to_le_bytes())?;
    for child in &state.children {
        write_host_adam_state(file, child)?;
    }
    Ok(())
}

pub(super) fn read_host_adam_moments(
    file: &mut File,
    max_tensor_len: usize,
    remaining_scalars: &mut usize,
) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
    let mut flag = [0; 1];
    file.read_exact(&mut flag)?;
    match flag[0] {
        0 => Ok(None),
        1 => {
            let mut len_bytes = [0; 8];
            file.read_exact(&mut len_bytes)?;
            let len = usize::try_from(u64::from_le_bytes(len_bytes))
                .map_err(|_| Error::Weights("host Adam moment is too large".to_owned()))?;
            let pair_len = len
                .checked_mul(2)
                .ok_or_else(|| Error::Weights("host Adam moment is too large".to_owned()))?;
            if len > max_tensor_len || pair_len > *remaining_scalars {
                return Err(Error::Weights(
                    "host Adam moment length exceeds the layer parameter size".to_owned(),
                ));
            }
            *remaining_scalars -= pair_len;
            let byte_len = len
                .checked_mul(4)
                .ok_or_else(|| Error::Weights("host Adam moment is too large".to_owned()))?;
            let mut read_values = || -> Result<Vec<f32>> {
                let mut bytes = vec![0; byte_len];
                file.read_exact(&mut bytes)?;
                Ok(bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect())
            };
            Ok(Some((read_values()?, read_values()?)))
        }
        _ => Err(Error::Weights("invalid host Adam moment flag".to_owned())),
    }
}

pub(super) fn read_host_adam_state(file: &mut File, max_tensor_len: usize) -> Result<HostAdamLayerState> {
    fn read_node(
        file: &mut File,
        max_tensor_len: usize,
        remaining_scalars: &mut usize,
        depth: usize,
    ) -> Result<HostAdamLayerState> {
        if depth > 4 {
            return Err(Error::Weights(
                "host Adam state is too deeply nested".to_owned(),
            ));
        }
        let weights = read_host_adam_moments(file, max_tensor_len, remaining_scalars)?;
        let biases = read_host_adam_moments(file, max_tensor_len, remaining_scalars)?;
        let scales = read_host_adam_moments(file, max_tensor_len, remaining_scalars)?;
        let mut count_bytes = [0; 8];
        file.read_exact(&mut count_bytes)?;
        let child_count = usize::try_from(u64::from_le_bytes(count_bytes))
            .map_err(|_| Error::Weights("host Adam child count is too large".to_owned()))?;
        if child_count > max_tensor_len {
            return Err(Error::Weights(
                "host Adam child count exceeds layer capacity".to_owned(),
            ));
        }
        let mut children = Vec::new();
        for _ in 0..child_count {
            children.push(read_node(
                file,
                max_tensor_len,
                remaining_scalars,
                depth + 1,
            )?);
        }
        Ok(HostAdamLayerState {
            weights,
            biases,
            scales,
            children,
        })
    }

    let mut remaining_scalars = max_tensor_len
        .checked_mul(2)
        .ok_or_else(|| Error::Weights("host Adam layer is too large".to_owned()))?;
    read_node(file, max_tensor_len, &mut remaining_scalars, 0)
}

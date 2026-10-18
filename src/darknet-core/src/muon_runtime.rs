//! Layer-to-matrix mapping and staged hybrid Muon/AdamW CPU updates.
//!
//! Each callback invocation is one independent logical matrix. Its visit order
//! is stable and is persisted as the matrix order in DNOPT6 optimizer sidecars.

use crate::error::{Error, Result};
use crate::model::{LayerKind, TensorShape};
use crate::muon::{MuonMatrixState, MuonMatrixStep, compute_muon_matrix_step};
use crate::runtime::{
    ConnectedParams, ConnectedTrainBuffers, ConvolutionalParams, ConvolutionalTrainBuffers,
    LayerParams, LayerTrainBuffers,
};

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct MuonLayerState {
    pub matrices: Vec<MuonMatrixState>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StagedMuonMatrix {
    pub rows: usize,
    pub cols: usize,
    pub learning_rate: f32,
    pub step: MuonMatrixStep,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct StagedMuonLayer {
    pub matrices: Vec<StagedMuonMatrix>,
}

pub(crate) type MatrixVisitor<'a> =
    dyn FnMut(&mut [f32], &mut [f32], usize, usize, f32) -> Result<()> + 'a;

fn checked_product(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or_else(|| Error::InvalidConfig(format!("Muon {name} size overflow")))
    })
}

fn visit_matrix(
    weights: &mut [f32],
    updates: &mut [f32],
    rows: usize,
    cols: usize,
    learning_rate_scale: f32,
    visitor: &mut MatrixVisitor<'_>,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(Error::InvalidConfig(
            "Muon matrices require nonzero rows and columns".to_owned(),
        ));
    }
    let len = checked_product(&[rows, cols], "matrix")?;
    if weights.len() != len || updates.len() != len {
        return Err(Error::InvalidConfig(format!(
            "Muon matrix weights/updates have lengths {}/{}, expected {len} for {rows}x{cols}",
            weights.len(),
            updates.len()
        )));
    }
    if !learning_rate_scale.is_finite() || learning_rate_scale < 0.0 {
        return Err(Error::InvalidConfig(
            "Muon matrix learning-rate scale must be finite and non-negative".to_owned(),
        ));
    }
    visitor(weights, updates, rows, cols, learning_rate_scale)
}

fn visit_grouped_convolution(
    params: &mut ConvolutionalParams,
    train: &mut ConvolutionalTrainBuffers,
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
    learning_rate_scale: f32,
    visitor: &mut MatrixVisitor<'_>,
) -> Result<()> {
    let groups = groups.max(1);
    if filters == 0 || input_channels == 0 || filters % groups != 0 || input_channels % groups != 0
    {
        return Err(Error::InvalidConfig(
            "Muon convolution has invalid grouped-channel geometry".to_owned(),
        ));
    }
    let rows = filters / groups;
    let cols = checked_product(&[input_channels / groups, size, size], "convolution matrix")?;
    let group_len = checked_product(&[rows, cols], "convolution group")?;
    let total_len = checked_product(&[groups, group_len], "grouped convolution")?;
    if params.weights.len() != total_len || train.weight_updates.len() != total_len {
        return Err(Error::InvalidConfig(format!(
            "Muon convolution weights/updates have lengths {}/{}, expected {}",
            params.weights.len(),
            train.weight_updates.len(),
            total_len
        )));
    }
    for group in 0..groups {
        let start = group * group_len;
        let end = start + group_len;
        visit_matrix(
            &mut params.weights[start..end],
            &mut train.weight_updates[start..end],
            rows,
            cols,
            learning_rate_scale,
            visitor,
        )?;
    }
    Ok(())
}

fn visit_dpt_head(
    spec: &crate::model::DptHeadSpec,
    input_shape: TensorShape,
    params: &mut crate::dpt_head::DptHeadWeights,
    train: &mut crate::runtime::DptHeadTrainBuffers,
    visitor: &mut MatrixVisitor<'_>,
) -> Result<()> {
    let geometry = spec.geometry(input_shape);
    let layout = crate::dpt_head::dpt_head_weight_layout(geometry)?;
    if params.values.len() != layout.total || train.weight_updates.len() != layout.total {
        return Err(Error::InvalidConfig(format!(
            "Muon DPT weights/updates have lengths {}/{}, expected {}",
            params.values.len(),
            train.weight_updates.len(),
            layout.total
        )));
    }
    for &index in &crate::dpt_head::DPT_MUON_WEIGHT_SLOTS {
        let tap = match index {
            0 | 2 | 4 | 6 => index / 2,
            8 | 10 | 14 => (index - 8) / 2,
            16..=19 => index - 16,
            _ => 0,
        };
        let (rows, cols) = match index {
            0 | 2 | 4 | 6 => (geometry.neck[tap], geometry.embed_dim),
            8 | 10 | 14 => (
                geometry.neck[tap],
                checked_product(
                    &[geometry.neck[tap], [4usize, 2, 0, 3][tap].pow(2)],
                    "DPT resize matrix",
                )?,
            ),
            16..=19 => (
                geometry.fusion,
                checked_product(&[geometry.neck[tap], 9], "DPT neck matrix")?,
            ),
            20 | 22 | 24 | 26 => (geometry.fusion, geometry.fusion),
            30 | 32 | 34 | 38 | 40 | 42 | 44 | 46 | 48 | 50 | 52 | 54 | 56 | 58 => (
                geometry.fusion,
                checked_product(&[geometry.fusion, 9], "DPT residual matrix")?,
            ),
            60 => (
                geometry.fusion / 2,
                checked_product(&[geometry.fusion, 9], "DPT head conv1")?,
            ),
            62 => (
                geometry.head_hidden,
                checked_product(&[geometry.fusion / 2, 9], "DPT head conv2")?,
            ),
            64 => (1, geometry.head_hidden),
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "Muon DPT matrix slot {index} has no declared view"
                )));
            }
        };
        let range = layout.get(index);
        let end = range
            .start
            .checked_add(range.len)
            .ok_or_else(|| Error::InvalidConfig("Muon DPT weight range overflows".to_owned()))?;
        visit_matrix(
            &mut params.values[range.start..end],
            &mut train.weight_updates[range.start..end],
            rows,
            cols,
            1.0,
            visitor,
        )?;
    }
    Ok(())
}

fn visit_connected(
    params: &mut ConnectedParams,
    train: &mut ConnectedTrainBuffers,
    visitor: &mut MatrixVisitor<'_>,
) -> Result<()> {
    let rows = params.biases.len();
    if rows == 0 || params.weights.len() % rows != 0 {
        return Err(Error::InvalidConfig(
            "Muon connected weight geometry is invalid".to_owned(),
        ));
    }
    let cols = params.weights.len() / rows;
    visit_matrix(
        &mut params.weights,
        &mut train.weight_updates,
        rows,
        cols,
        1.0,
        visitor,
    )
}

fn visit_conv_range(
    weights: &mut [f32],
    updates: &mut [f32],
    start: usize,
    rows: usize,
    cols: usize,
    count: usize,
    learning_rate_scale: f32,
    visitor: &mut MatrixVisitor<'_>,
) -> Result<()> {
    let matrix_len = checked_product(&[rows, cols], "factor matrix")?;
    let end = start
        .checked_add(checked_product(&[matrix_len, count], "factor block")?)
        .ok_or_else(|| Error::InvalidConfig("Muon factor range overflow".to_owned()))?;
    if end > weights.len() || end > updates.len() {
        return Err(Error::InvalidConfig(
            "Muon factor block exceeds its parameter/update tensor".to_owned(),
        ));
    }
    for matrix_index in 0..count {
        let from = start + matrix_index * matrix_len;
        let to = from + matrix_len;
        visit_matrix(
            &mut weights[from..to],
            &mut updates[from..to],
            rows,
            cols,
            learning_rate_scale,
            visitor,
        )?;
    }
    Ok(())
}

fn visit_clifford_branch(
    params: &mut crate::clifford::CliffordBranchParams,
    train: &mut crate::clifford::CliffordBranchTrainBuffers,
    channels: usize,
    raw_channels: usize,
    visitor: &mut MatrixVisitor<'_>,
) -> Result<()> {
    visit_matrix(
        &mut params.projection_weights,
        &mut train.projection_weight_updates,
        channels,
        raw_channels,
        1.0,
        visitor,
    )?;
    visit_matrix(
        &mut params.gate_weights,
        &mut train.gate_weight_updates,
        channels,
        2 * channels,
        1.0,
        visitor,
    )
}

/// Visit the eligible trainable matrices of one layer in deterministic order.
pub(crate) fn visit_muon_matrices(
    kind: &LayerKind,
    input_shape: TensorShape,
    output_shape: TensorShape,
    params: &mut LayerParams,
    train: &mut LayerTrainBuffers,
    visitor: &mut MatrixVisitor<'_>,
) -> Result<()> {
    // Shared convolution weights are visited only at their owning layer.
    if matches!(kind, LayerKind::Convolutional(spec) if spec.share_index.is_some()) {
        return Ok(());
    }
    match (kind, params, train) {
        (
            LayerKind::FoveationFilm(spec),
            LayerParams::FoveationFilm(params),
            LayerTrainBuffers::FoveationFilm(train),
        ) => {
            crate::runtime::validate_foveation_film_param_layout(
                spec,
                input_shape.channels,
                params,
            )?;
            visit_matrix(
                &mut params.w_embed,
                &mut train.w_embed_updates,
                spec.embed_dim,
                spec.param_count,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.w_gamma,
                &mut train.w_gamma_updates,
                input_shape.channels,
                spec.embed_dim,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.w_beta,
                &mut train.w_beta_updates,
                input_shape.channels,
                spec.embed_dim,
                1.0,
                visitor,
            )
        }
        (
            LayerKind::HyperbolicFoveation(spec),
            LayerParams::HyperbolicFoveation(params),
            LayerTrainBuffers::HyperbolicFoveation(train),
        ) => {
            crate::runtime::validate_hyperbolic_foveation_param_layout(spec, params)?;
            if spec.pred_hidden == 0 {
                return Ok(());
            }
            let input_dim = spec.predictor_input_dim().ok_or_else(|| {
                Error::InvalidConfig(
                    "Muon hyperbolic-foveation predictor input dimension overflows".to_owned(),
                )
            })?;
            visit_matrix(
                &mut params.pred_w1,
                &mut train.pred_w1_updates,
                spec.pred_hidden,
                input_dim,
                spec.hf_lr_mult,
                visitor,
            )?;
            visit_matrix(
                &mut params.pred_w2,
                &mut train.pred_w2_updates,
                spec.param_count(),
                spec.pred_hidden,
                spec.hf_lr_mult,
                visitor,
            )
        }
        (
            LayerKind::RecursiveBlock(spec),
            LayerParams::RecursiveBlock(params),
            LayerTrainBuffers::RecursiveBlock(train),
        ) => {
            if spec.body.len() != params.body.len() || spec.body.len() != train.body.len() {
                return Err(Error::InvalidConfig(
                    "Muon recursive-block body parameter/update count mismatch".to_owned(),
                ));
            }
            if spec.ouroboros == 1 {
                let channels = input_shape.channels;
                visit_matrix(
                    &mut params.controller_weights,
                    &mut train.controller_weight_updates,
                    channels * 2,
                    channels + 1,
                    1.0,
                    visitor,
                )?;
            }
            for ((layer, params), train) in
                spec.body.iter().zip(&mut params.body).zip(&mut train.body)
            {
                visit_muon_matrices(
                    &layer.kind,
                    layer.input_shape,
                    layer.output_shape,
                    params,
                    train,
                    visitor,
                )?;
            }
            Ok(())
        }
        (
            LayerKind::Convolutional(spec),
            LayerParams::Convolutional(params),
            LayerTrainBuffers::Convolutional(train),
        ) => visit_grouped_convolution(
            params,
            train,
            input_shape.channels,
            spec.filters,
            spec.size,
            spec.groups,
            1.0,
            visitor,
        ),
        (
            LayerKind::Deconvolutional(spec),
            LayerParams::Convolutional(params),
            LayerTrainBuffers::Convolutional(train),
        ) => visit_matrix(
            &mut params.weights,
            &mut train.weight_updates,
            input_shape.channels,
            checked_product(
                &[spec.filters, spec.size, spec.size],
                "deconvolution matrix",
            )?,
            1.0,
            visitor,
        ),
        (
            LayerKind::DptHead(spec),
            LayerParams::DptHead(params),
            LayerTrainBuffers::DptHead(train),
        ) => visit_dpt_head(spec, input_shape, params, train, visitor),
        (
            LayerKind::Connected(_),
            LayerParams::Connected(params),
            LayerTrainBuffers::Connected(train),
        ) => visit_connected(params, train, visitor),
        (LayerKind::Local(spec), LayerParams::Local(params), LayerTrainBuffers::Local(train)) => {
            let locations = checked_product(
                &[output_shape.width, output_shape.height],
                "local output locations",
            )?;
            let cols = checked_product(
                &[input_shape.channels, spec.size, spec.size],
                "local matrix",
            )?;
            let matrix_len = checked_product(&[spec.filters, cols], "local matrix")?;
            if locations.checked_mul(matrix_len) != Some(params.weights.len())
                || params.weights.len() != train.weight_updates.len()
            {
                return Err(Error::InvalidConfig(
                    "Muon local weight/update geometry is invalid".to_owned(),
                ));
            }
            for location in 0..locations {
                let start = location * matrix_len;
                let end = start + matrix_len;
                visit_matrix(
                    &mut params.weights[start..end],
                    &mut train.weight_updates[start..end],
                    spec.filters,
                    cols,
                    1.0,
                    visitor,
                )?;
            }
            Ok(())
        }
        (LayerKind::Gru(_), LayerParams::Gru(params), LayerTrainBuffers::Gru(train)) => {
            for (params, train) in [
                (&mut params.input_z, &mut train.input_z),
                (&mut params.state_z, &mut train.state_z),
                (&mut params.input_r, &mut train.input_r),
                (&mut params.state_r, &mut train.state_r),
                (&mut params.input_h, &mut train.input_h),
                (&mut params.state_h, &mut train.state_h),
            ] {
                visit_connected(params, train, visitor)?;
            }
            Ok(())
        }
        (LayerKind::Rnn(_), LayerParams::Rnn(params), LayerTrainBuffers::Rnn(train)) => {
            for (params, train) in [
                (&mut params.input, &mut train.input),
                (&mut params.self_params, &mut train.self_params),
                (&mut params.output, &mut train.output),
            ] {
                visit_connected(params, train, visitor)?;
            }
            Ok(())
        }
        (LayerKind::Lstm(_), LayerParams::Lstm(params), LayerTrainBuffers::Lstm(train)) => {
            for (params, train) in [
                (&mut params.wf, &mut train.wf),
                (&mut params.wi, &mut train.wi),
                (&mut params.wg, &mut train.wg),
                (&mut params.wo, &mut train.wo),
                (&mut params.uf, &mut train.uf),
                (&mut params.ui, &mut train.ui),
                (&mut params.ug, &mut train.ug),
                (&mut params.uo, &mut train.uo),
            ] {
                visit_connected(params, train, visitor)?;
            }
            Ok(())
        }
        (LayerKind::Crnn(spec), LayerParams::Crnn(params), LayerTrainBuffers::Crnn(train)) => {
            for (params, train, input_channels, filters) in [
                (
                    &mut params.input,
                    &mut train.input,
                    input_shape.channels,
                    spec.hidden,
                ),
                (
                    &mut params.self_params,
                    &mut train.self_params,
                    spec.hidden,
                    spec.hidden,
                ),
                (
                    &mut params.output,
                    &mut train.output,
                    spec.hidden,
                    spec.output,
                ),
            ] {
                visit_grouped_convolution(
                    params,
                    train,
                    input_channels,
                    filters,
                    spec.size,
                    spec.groups,
                    1.0,
                    visitor,
                )?;
            }
            Ok(())
        }
        (LayerKind::Vit(spec), LayerParams::Vit(params), LayerTrainBuffers::Vit(train)) => {
            let patch_cols = checked_product(
                &[input_shape.channels, spec.patch_size, spec.patch_size],
                "ViT patch projection",
            )?;
            visit_matrix(
                &mut params.patch_embed,
                &mut train.updates.patch_embed,
                spec.dim,
                patch_cols,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.qkv_weight,
                &mut train.updates.qkv_weight,
                checked_product(&[3, spec.dim], "ViT QKV rows")?,
                spec.dim,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.wo,
                &mut train.updates.wo,
                spec.dim,
                spec.dim,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.ffn_w1,
                &mut train.updates.ffn_w1,
                spec.mlp_dim,
                spec.dim,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.ffn_w2,
                &mut train.updates.ffn_w2,
                spec.dim,
                spec.mlp_dim,
                1.0,
                visitor,
            )
        }
        (LayerKind::Swin(spec), LayerParams::Swin(params), LayerTrainBuffers::Swin(train)) => {
            let hidden = checked_product(&[spec.filters, spec.ffn_ratio], "Swin FFN")?;
            visit_matrix(
                &mut params.qkv_weight,
                &mut train.updates.qkv_weight,
                checked_product(&[3, input_shape.channels], "Swin QKV rows")?,
                input_shape.channels,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.wo,
                &mut train.updates.wo,
                spec.filters,
                input_shape.channels,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.ffn_w1,
                &mut train.updates.ffn_w1,
                hidden,
                spec.filters,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.ffn_w2,
                &mut train.updates.ffn_w2,
                spec.filters,
                hidden,
                1.0,
                visitor,
            )?;
            match (&mut params.res_proj, &mut train.updates.res_proj) {
                (Some(weights), Some(updates)) => visit_matrix(
                    weights,
                    updates,
                    spec.filters,
                    input_shape.channels,
                    1.0,
                    visitor,
                ),
                (None, None) => Ok(()),
                _ => Err(Error::InvalidConfig(
                    "Muon Swin residual-projection mismatch".to_owned(),
                )),
            }
        }
        (
            LayerKind::TuckerAttention(spec),
            LayerParams::TuckerAttention(params),
            LayerTrainBuffers::TuckerAttention(train),
        ) => {
            let channels = spec.filters;
            let heads = spec.heads;
            if heads == 0 || channels % heads != 0 {
                return Err(Error::InvalidConfig(
                    "Muon Tucker channels must be divisible by heads".to_owned(),
                ));
            }
            let dim = channels / heads;
            let mut cursor = 0usize;
            for (rows, cols) in [
                (channels, spec.rank_q),
                (channels, spec.rank_k),
                (channels, spec.rank_v),
            ] {
                visit_conv_range(
                    &mut params.weights,
                    &mut train.weight_updates,
                    cursor,
                    rows,
                    cols,
                    1,
                    1.0,
                    visitor,
                )?;
                cursor += rows * cols;
            }
            for (rank, heads_count) in [
                (spec.rank_q, heads),
                (spec.rank_k, heads),
                (spec.rank_v, heads),
            ] {
                for _ in 0..heads_count {
                    visit_conv_range(
                        &mut params.weights,
                        &mut train.weight_updates,
                        cursor,
                        rank,
                        dim,
                        1,
                        1.0,
                        visitor,
                    )?;
                    cursor += rank * dim;
                }
            }
            for _ in 0..heads {
                visit_conv_range(
                    &mut params.weights,
                    &mut train.weight_updates,
                    cursor,
                    dim,
                    spec.rank_o,
                    1,
                    1.0,
                    visitor,
                )?;
                cursor += dim * spec.rank_o;
            }
            visit_conv_range(
                &mut params.weights,
                &mut train.weight_updates,
                cursor,
                spec.rank_o,
                channels,
                1,
                1.0,
                visitor,
            )?;
            cursor += spec.rank_o * channels;
            if cursor != params.weights.len() || cursor != train.weight_updates.len() {
                return Err(Error::InvalidConfig(
                    "Muon Tucker factor layout does not cover the weights".to_owned(),
                ));
            }
            Ok(())
        }
        (
            LayerKind::MambaVision(spec),
            LayerParams::MambaVision(params),
            LayerTrainBuffers::MambaVision(train),
        ) => {
            let blocks = crate::mambavision::mambavision_param_blocks(params)
                .map_err(Error::InvalidConfig)?;
            let half = spec.filters / 2;
            let projection_width = checked_product(&[spec.state, 2], "Mamba state width")?
                .checked_add(spec.dt_rank)
                .ok_or_else(|| Error::InvalidConfig("Muon Mamba projection overflow".to_owned()))?;
            let hidden = checked_product(&[spec.filters, spec.ffn_ratio], "Mamba FFN")?;
            for block in blocks {
                let (rows, cols, groups) = match block.name {
                    "mambavision_in_proj_weights" => (spec.filters, input_shape.channels, 1),
                    "mambavision_conv_x" | "mambavision_conv_z" => (half, spec.conv_size, half),
                    "mambavision_x_proj" => (projection_width, half, 1),
                    "mambavision_dt_proj" => (half, spec.dt_rank, 1),
                    "mambavision_out_proj" => (spec.filters, spec.filters, 1),
                    "mambavision_res_proj" => (spec.filters, input_shape.channels, 1),
                    "mambavision_ffn_w1" => (hidden, spec.filters, 1),
                    "mambavision_ffn_w2" => (spec.filters, hidden, 1),
                    _ => continue,
                };
                if groups > 1 {
                    let kernel_len = checked_product(&[rows, cols], "Mamba depthwise block")?;
                    if params.values[block.range.clone()].len() != kernel_len
                        || train.updates[block.range.clone()].len() != kernel_len
                    {
                        return Err(Error::InvalidConfig(
                            "Muon Mamba depthwise block shape mismatch".to_owned(),
                        ));
                    }
                    for channel in 0..rows {
                        let start = block.range.start + channel * cols;
                        let end = start + cols;
                        visit_matrix(
                            &mut params.values[start..end],
                            &mut train.updates[start..end],
                            1,
                            cols,
                            1.0,
                            visitor,
                        )?;
                    }
                } else {
                    visit_matrix(
                        &mut params.values[block.range.clone()],
                        &mut train.updates[block.range],
                        rows,
                        cols,
                        1.0,
                        visitor,
                    )?;
                }
            }
            Ok(())
        }
        (
            LayerKind::KimiKda(spec),
            LayerParams::KimiKda(params),
            LayerTrainBuffers::KimiKda(train),
        ) => {
            let channels = spec.filters;
            let rank = spec.rank;
            let experts = spec.experts;
            let hidden = spec.hidden;
            let mut cursor = 0usize;
            for (rows, cols) in [
                (rank, channels),
                (rank, channels),
                (channels, channels),
                (channels, channels),
                (rank, channels),
                (rank, channels),
                (experts, channels),
            ] {
                let len = checked_product(&[rows, cols], "Kimi projection")?;
                visit_conv_range(
                    &mut params.weights,
                    &mut train.weight_updates,
                    cursor,
                    rows,
                    cols,
                    1,
                    1.0,
                    visitor,
                )?;
                cursor += len;
            }
            for _ in 0..=experts {
                visit_conv_range(
                    &mut params.weights,
                    &mut train.weight_updates,
                    cursor,
                    hidden,
                    channels,
                    1,
                    1.0,
                    visitor,
                )?;
                cursor += hidden * channels;
            }
            for _ in 0..=experts {
                visit_conv_range(
                    &mut params.weights,
                    &mut train.weight_updates,
                    cursor,
                    channels,
                    hidden,
                    1,
                    1.0,
                    visitor,
                )?;
                cursor += channels * hidden;
            }
            if cursor != params.weights.len() || cursor != train.weight_updates.len() {
                return Err(Error::InvalidConfig(
                    "Muon Kimi KDA matrix layout does not cover the weights".to_owned(),
                ));
            }
            Ok(())
        }
        (LayerKind::Moe(spec), LayerParams::Moe(params), LayerTrainBuffers::Moe(train)) => {
            visit_matrix(
                &mut params.router_weights,
                &mut train.router_weight_updates,
                spec.experts,
                input_shape.channels,
                1.0,
                visitor,
            )?;
            if params.experts.len() != train.experts.len() || params.experts.len() != spec.experts {
                return Err(Error::InvalidConfig(
                    "Muon MoE expert count mismatch".to_owned(),
                ));
            }
            for (expert, updates) in params.experts.iter_mut().zip(&mut train.experts) {
                visit_grouped_convolution(
                    expert,
                    updates,
                    input_shape.channels,
                    spec.expert.filters,
                    spec.expert.size,
                    spec.expert.groups,
                    1.0,
                    visitor,
                )?;
            }
            Ok(())
        }
        (
            LayerKind::EmlConvolutional(spec),
            LayerParams::EmlConvolutional(params),
            LayerTrainBuffers::EmlConvolutional(train),
        ) => {
            for (params, train) in [
                (&mut params.input, &mut train.input),
                (&mut params.self_params, &mut train.self_params),
            ] {
                visit_grouped_convolution(
                    params,
                    train,
                    input_shape.channels,
                    spec.branch.filters,
                    spec.branch.size,
                    spec.branch.groups,
                    1.0,
                    visitor,
                )?;
            }
            Ok(())
        }
        (LayerKind::Dcnv4(spec), LayerParams::Dcnv4(params), LayerTrainBuffers::Dcnv4(train)) => {
            let (_, _, offset_dim) =
                crate::model::dcnv4_parameter_layout(input_shape.channels, spec).ok_or_else(
                    || Error::InvalidConfig("Muon DCNv4 geometry overflow".to_owned()),
                )?;
            let cols = checked_product(
                &[input_shape.channels, spec.size, spec.size],
                "DCNv4 offset projection",
            )?;
            visit_matrix(
                &mut params.offset_weights,
                &mut train.offset_weight_updates,
                offset_dim,
                cols,
                0.01,
                visitor,
            )
        }
        (
            LayerKind::GraphConv(spec),
            LayerParams::GraphConv(params),
            LayerTrainBuffers::GraphConv(train),
        ) => {
            visit_grouped_convolution(
                &mut params.projection,
                &mut train.projection,
                input_shape.channels,
                spec.filters,
                1,
                spec.groups,
                1.0,
                visitor,
            )?;
            let groups = spec.groups.max(1);
            let rows = spec.filters / groups;
            let cols = input_shape.channels / groups;
            match (&mut params.self_weights, &mut train.self_weight_updates) {
                (Some(weights), Some(updates)) => {
                    let group_len = rows * cols;
                    if weights.len() != groups * group_len || updates.len() != groups * group_len {
                        return Err(Error::InvalidConfig(
                            "Muon graph_conv self-weight geometry is invalid".to_owned(),
                        ));
                    }
                    for group in 0..groups {
                        let start = group * group_len;
                        visit_matrix(
                            &mut weights[start..start + group_len],
                            &mut updates[start..start + group_len],
                            rows,
                            cols,
                            1.0,
                            visitor,
                        )?;
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "Muon graph_conv self-weight mismatch".to_owned(),
                    ));
                }
            }
            match (&mut params.edge_kernel, &mut train.edge_kernel_updates) {
                (Some(weights), Some(updates)) => {
                    let area = checked_product(&[spec.size, spec.size], "graph edge kernel")?;
                    visit_conv_range(
                        weights,
                        updates,
                        0,
                        area,
                        checked_product(&[2, cols], "graph edge kernel")?,
                        groups,
                        1.0,
                        visitor,
                    )?;
                }
                (None, None) => {}
                _ => {
                    return Err(Error::InvalidConfig(
                        "Muon graph_conv edge-kernel mismatch".to_owned(),
                    ));
                }
            }
            Ok(())
        }
        (
            LayerKind::DeformConv(spec),
            LayerParams::DeformConv(params),
            LayerTrainBuffers::DeformConv(train),
        ) => {
            visit_grouped_convolution(
                &mut params.main,
                &mut train.main,
                input_shape.channels,
                spec.filters,
                spec.size,
                spec.groups,
                1.0,
                visitor,
            )?;
            let area = checked_product(&[spec.size, spec.size], "deformable kernel")?;
            let columns = checked_product(&[input_shape.channels, area], "deform predictor")?;
            visit_matrix(
                &mut params.offset_weights,
                &mut train.offset_weight_updates,
                checked_product(&[2, area], "deform offset rows")?,
                columns,
                0.01,
                visitor,
            )?;
            match (&mut params.mask_weights, &mut train.mask_weight_updates) {
                (Some(weights), Some(updates)) => {
                    visit_matrix(weights, updates, area, columns, 0.01, visitor)
                }
                (None, None) => Ok(()),
                _ => Err(Error::InvalidConfig(
                    "Muon deform_conv mask-weight mismatch".to_owned(),
                )),
            }
        }
        (
            LayerKind::DsConv(spec),
            LayerParams::DsConv(params),
            LayerTrainBuffers::DsConv(train),
        ) => {
            visit_matrix(
                &mut params.collapse.weights,
                &mut train.collapse.weight_updates,
                spec.filters,
                checked_product(&[input_shape.channels, spec.size], "DSConv collapse")?,
                1.0,
                visitor,
            )?;
            visit_matrix(
                &mut params.offset_weights,
                &mut train.offset_weight_updates,
                checked_product(&[2, spec.size], "DSConv offset rows")?,
                checked_product(&[input_shape.channels, 9], "DSConv offset columns")?,
                0.01,
                visitor,
            )
        }
        (
            LayerKind::Clifford(spec),
            LayerParams::Clifford(params),
            LayerTrainBuffers::Clifford(train),
        ) => {
            let channels = input_shape.channels;
            let raw_channels = crate::clifford::raw_channels(spec, channels);
            if crate::clifford::has_local(spec) {
                visit_matrix(
                    &mut params.determinant_weights,
                    &mut train.determinant_weight_updates,
                    channels,
                    channels,
                    1.0,
                    visitor,
                )?;
                if params.depthwise.len() != train.depthwise.len() {
                    return Err(Error::InvalidConfig(
                        "Muon Clifford depthwise parameter/update count mismatch".to_owned(),
                    ));
                }
                for (params, train) in params.depthwise.iter_mut().zip(&mut train.depthwise) {
                    visit_grouped_convolution(
                        params,
                        train,
                        channels,
                        channels,
                        spec.dwconv_size,
                        channels,
                        1.0,
                        visitor,
                    )?;
                }
                visit_clifford_branch(
                    &mut params.local,
                    &mut train.local,
                    channels,
                    raw_channels,
                    visitor,
                )?;
            }
            match (
                crate::clifford::has_global(spec),
                &mut params.global,
                &mut train.global,
            ) {
                (true, Some(params), Some(train)) => {
                    visit_clifford_branch(params, train, channels, raw_channels, visitor)
                }
                (false, None, None) => Ok(()),
                _ => Err(Error::InvalidConfig(
                    "Muon Clifford global parameter/update mismatch".to_owned(),
                )),
            }
        }
        (LayerKind::Wmhf(spec), LayerParams::Wmhf(params), LayerTrainBuffers::Wmhf(train)) => {
            let local = spec.local_channels;
            for (params, train, channels, filters, size, groups) in [
                (
                    &mut params.pre,
                    &mut train.pre,
                    input_shape.channels,
                    spec.filters,
                    1,
                    1,
                ),
                (
                    &mut params.local_dw3,
                    &mut train.local_dw3,
                    local,
                    local,
                    3,
                    local,
                ),
                (
                    &mut params.local_dw5,
                    &mut train.local_dw5,
                    local,
                    local,
                    5,
                    local,
                ),
                (
                    &mut params.local_dw7,
                    &mut train.local_dw7,
                    local,
                    local,
                    7,
                    local,
                ),
                (
                    &mut params.local_mix,
                    &mut train.local_mix,
                    3 * local,
                    local,
                    1,
                    1,
                ),
                (
                    &mut params.hf_gate,
                    &mut train.hf_gate,
                    spec.global_channels,
                    spec.filters,
                    1,
                    1,
                ),
                (
                    &mut params.fuse,
                    &mut train.fuse,
                    spec.filters,
                    spec.filters,
                    1,
                    1,
                ),
            ] {
                visit_grouped_convolution(
                    params, train, channels, filters, size, groups, 1.0, visitor,
                )?;
            }
            Ok(())
        }
        // Batch-norm values, biases, norm/embedding tables, scan vectors,
        // structural shortcut coefficients, and frozen layers intentionally
        // stay in the AdamW fallback rather than being coerced into matrices.
        _ => Ok(()),
    }
}

pub(crate) fn matrix_shapes(
    kind: &LayerKind,
    input_shape: TensorShape,
    output_shape: TensorShape,
    params: &mut LayerParams,
    train: &mut LayerTrainBuffers,
) -> Result<Vec<(usize, usize)>> {
    let mut shapes = Vec::new();
    visit_muon_matrices(
        kind,
        input_shape,
        output_shape,
        params,
        train,
        &mut |_, _, rows, cols, _| {
            shapes.push((rows, cols));
            Ok(())
        },
    )?;
    Ok(shapes)
}

pub(crate) fn stage_layer(
    kind: &LayerKind,
    input_shape: TensorShape,
    output_shape: TensorShape,
    params: &mut LayerParams,
    train: &mut LayerTrainBuffers,
    state: &MuonLayerState,
    learning_rate: f32,
    batch: usize,
    loss_scale: f32,
    momentum: f32,
) -> Result<StagedMuonLayer> {
    if !learning_rate.is_finite() || learning_rate < 0.0 {
        return Err(Error::InvalidConfig(
            "Muon learning rate must be finite and non-negative".to_owned(),
        ));
    }
    let mut staged = StagedMuonLayer::default();
    visit_muon_matrices(
        kind,
        input_shape,
        output_shape,
        params,
        train,
        &mut |_, updates, rows, cols, learning_rate_scale| {
            let index = staged.matrices.len();
            let previous = match state.matrices.get(index) {
                Some(previous) => previous.clone(),
                None if state.matrices.is_empty() => MuonMatrixState::zeroed(rows, cols)?,
                None => {
                    return Err(Error::InvalidConfig(format!(
                        "Muon state has {} matrices but layer layout has at least {}",
                        state.matrices.len(),
                        index + 1
                    )));
                }
            };
            staged.matrices.push(StagedMuonMatrix {
                rows,
                cols,
                learning_rate: learning_rate * learning_rate_scale,
                step: compute_muon_matrix_step(
                    updates, rows, cols, batch, loss_scale, momentum, &previous,
                )?,
            });
            Ok(())
        },
    )?;
    if staged.matrices.len() != state.matrices.len() && !state.matrices.is_empty() {
        return Err(Error::InvalidConfig(format!(
            "Muon state has {} matrices but layer layout has {}",
            state.matrices.len(),
            staged.matrices.len()
        )));
    }
    // Keep staged computations transactional: zero updates only once every
    // matrix direction has passed shape and finite-value validation.
    let mut index = 0usize;
    visit_muon_matrices(
        kind,
        input_shape,
        output_shape,
        params,
        train,
        &mut |_, updates, _, _, _| {
            if index >= staged.matrices.len() {
                return Err(Error::InvalidConfig(
                    "Muon matrix layout changed while staging an update".to_owned(),
                ));
            }
            updates.fill(0.0);
            index += 1;
            Ok(())
        },
    )?;
    if index != staged.matrices.len() {
        return Err(Error::InvalidConfig(
            "Muon matrix layout changed while staging an update".to_owned(),
        ));
    }
    Ok(staged)
}

pub(crate) fn apply_staged_layer(
    kind: &LayerKind,
    input_shape: TensorShape,
    output_shape: TensorShape,
    params: &mut LayerParams,
    train: &mut LayerTrainBuffers,
    state: &mut MuonLayerState,
    staged: StagedMuonLayer,
) -> Result<()> {
    let mut new_states = Vec::with_capacity(staged.matrices.len());
    let mut index = 0usize;
    visit_muon_matrices(
        kind,
        input_shape,
        output_shape,
        params,
        train,
        &mut |weights, _, rows, cols, _| {
            let matrix = staged.matrices.get(index).ok_or_else(|| {
                Error::InvalidConfig("Muon staged matrix count is too small".to_owned())
            })?;
            if matrix.rows != rows || matrix.cols != cols {
                return Err(Error::InvalidConfig(format!(
                    "Muon staged matrix {} has shape {}x{}, expected {rows}x{cols}",
                    index, matrix.rows, matrix.cols
                )));
            }
            matrix.step.add_direction(weights, matrix.learning_rate)?;
            new_states.push(MuonMatrixState {
                rows,
                cols,
                momentum: matrix.step.next_momentum.clone(),
            });
            index += 1;
            Ok(())
        },
    )?;
    if index != staged.matrices.len() {
        return Err(Error::InvalidConfig(format!(
            "Muon staged matrix count is {}, but layer layout has {index}",
            staged.matrices.len()
        )));
    }
    state.matrices = new_states;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::ExecutableNetwork;

    fn matrix_shapes_for(cfg: &str) -> Vec<(usize, usize)> {
        let network =
            Network::from_config(ConfigFile::parse_str("muon-layout.cfg", cfg).expect("parse cfg"))
                .expect("build network");
        let mut executable = ExecutableNetwork::new(network);
        assert_eq!(executable.network.layers.len(), 1);
        let layer = executable.network.layers[0].clone();
        matrix_shapes(
            &layer.kind,
            layer.input_shape,
            layer.output_shape,
            &mut executable.params[0],
            &mut executable.train_buffers[0],
        )
        .expect("Muon matrix layout")
    }

    #[test]
    fn muon_grouped_convolution_and_local_layer_keep_independent_matrices() {
        let grouped = matrix_shapes_for(
            r#"[net]
width=3
height=3
channels=4
[convolutional]
filters=4
size=3
groups=2
"#,
        );
        assert_eq!(grouped, vec![(2, 18), (2, 18)]);

        let local = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=2
[local]
filters=2
size=1
stride=1
"#,
        );
        assert_eq!(local, vec![(2, 2); 4]);
    }

    #[test]
    fn muon_recurrent_containers_map_each_projection_separately() {
        let rnn = matrix_shapes_for(
            r#"[net]
width=1
height=1
channels=2
[rnn]
hidden=3
output=2
activation=linear
"#,
        );
        assert_eq!(rnn, vec![(3, 2), (3, 3), (2, 3)]);

        let gru = matrix_shapes_for(
            r#"[net]
width=1
height=1
channels=2
[gru]
output=3
activation=linear
"#,
        );
        assert_eq!(gru, vec![(3, 2), (3, 3), (3, 2), (3, 3), (3, 2), (3, 3)]);

        let lstm = matrix_shapes_for(
            r#"[net]
width=1
height=1
channels=2
[lstm]
output=3
"#,
        );
        assert_eq!(
            lstm,
            vec![(3, 3); 4]
                .into_iter()
                .chain(vec![(3, 2); 4])
                .collect::<Vec<_>>()
        );

        let crnn = matrix_shapes_for(
            r#"[net]
width=1
height=1
channels=2
[crnn]
hidden=3
output=2
size=1
activation=linear
"#,
        );
        assert_eq!(crnn, vec![(3, 2), (3, 3), (2, 3)]);
    }

    #[test]
    fn muon_convolutional_family_layouts_cover_the_trainable_projections() {
        let deconvolution = matrix_shapes_for(
            r#"[net]
width=1
height=1
channels=2
[deconvolutional]
filters=3
size=2
stride=1
"#,
        );
        assert_eq!(deconvolution, vec![(2, 12)]);

        let eml = matrix_shapes_for(
            r#"[net]
width=1
height=1
channels=2
[eml_convolutional]
filters=2
size=1
"#,
        );
        assert_eq!(eml, vec![(2, 2), (2, 2)]);

        let moe = matrix_shapes_for(
            r#"[net]
width=1
height=1
channels=2
[moe_convolutional]
experts=2
filters=4
size=1
"#,
        );
        assert_eq!(moe, vec![(2, 2), (4, 2), (4, 2)]);

        let graph = matrix_shapes_for(
            r#"[net]
width=3
height=3
channels=4
[graph_conv]
filters=4
groups=2
size=3
edge_mode=1
use_self=1
"#,
        );
        assert_eq!(graph, vec![(2, 2), (2, 2), (2, 2), (2, 2), (9, 4), (9, 4)]);

        let deform = matrix_shapes_for(
            r#"[net]
width=3
height=3
channels=4
[deform_conv]
filters=4
groups=2
size=3
mask=1
"#,
        );
        assert_eq!(deform, vec![(2, 18), (2, 18), (18, 36), (9, 36)]);

        let dcnv4 = matrix_shapes_for(
            r#"[net]
width=3
height=3
channels=4
[dcnv4]
filters=4
groups=2
size=3
"#,
        );
        assert_eq!(dcnv4, vec![(56, 36)]);

        let dsconv = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=2
[dsconv]
filters=3
size=3
morph=0
templates=1
"#,
        );
        assert_eq!(dsconv, vec![(3, 6), (6, 18)]);

        let wmhf = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=4
[wmhf]
filters=4
"#,
        );
        assert_eq!(
            wmhf,
            vec![
                (4, 4),
                (1, 9),
                (1, 9),
                (1, 25),
                (1, 25),
                (1, 49),
                (1, 49),
                (2, 6),
                (4, 1),
                (4, 4),
            ]
        );
    }

    #[test]
    fn muon_attention_factor_and_projection_layouts_match_runtime_blobs() {
        let vit = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=2
[vit]
dim=4
heads=2
patch_size=1
mlp_dim=8
"#,
        );
        assert_eq!(vit, vec![(4, 2), (12, 4), (4, 4), (8, 4), (4, 8)]);

        let swin = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=2
[swin]
filters=4
size=2
heads=2
ffn_ratio=2
"#,
        );
        assert_eq!(swin, vec![(6, 2), (4, 2), (8, 4), (4, 8), (4, 2)]);

        let tucker = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=4
[tucker_attention]
filters=4
size=2
heads=2
rank_q=2
rank_k=2
rank_v=2
rank_o=2
"#,
        );
        assert_eq!(tucker.len(), 12);
        assert_eq!(&tucker[..3], &[(4, 2), (4, 2), (4, 2)]);
        assert_eq!(&tucker[3..9], &[(2, 2); 6]);
        assert_eq!(&tucker[9..11], &[(2, 2); 2]);
        assert_eq!(tucker[11], (2, 4));

        let mamba = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=2
[mambavision]
filters=4
state=2
conv_size=3
dt_rank=2
ffn_ratio=2
"#,
        );
        assert_eq!(
            mamba,
            vec![
                (4, 2),
                (1, 3),
                (1, 3),
                (1, 3),
                (1, 3),
                (6, 2),
                (2, 2),
                (4, 4),
                (4, 2),
                (8, 4),
                (4, 8),
            ]
        );

        let kimi = matrix_shapes_for(
            r#"[net]
width=2
height=2
channels=4
[kimi_kda]
filters=4
rank=2
experts=2
hidden=3
"#,
        );
        assert_eq!(kimi.len(), 13);
        assert_eq!(
            &kimi[..7],
            &[(2, 4), (2, 4), (4, 4), (4, 4), (2, 4), (2, 4), (2, 4)]
        );
        assert_eq!(&kimi[7..10], &[(3, 4); 3]);
        assert_eq!(&kimi[10..], &[(4, 3); 3]);
    }
}

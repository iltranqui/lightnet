//! Darknet binary `.weights` compatibility.
//!
//! Darknet serializes arrays in a fixed order instead of storing a
//! self-describing tensor graph.  This module centralizes that order for
//! inspection, loading, and saving so recurrent nested layers and batch-norm
//! arrays cannot silently drift from the C++ format.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::DARKNET_WEIGHTS_VERSION_MAJOR;
use crate::DARKNET_WEIGHTS_VERSION_MINOR;
use crate::DARKNET_WEIGHTS_VERSION_PATCH;
use crate::error::{Error, Result};
use crate::model::{Layer, LayerKind, Network};
use crate::runtime::{ConnectedParams, ConvolutionalParams, LayerParams};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightsHeader {
    pub major: i32,
    pub minor: i32,
    pub revision: i32,
    pub seen: u64,
    pub header_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightsArrayInspection {
    pub name: String,
    pub len: usize,
    pub byte_offset: u64,
    pub byte_len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerWeightsInspection {
    pub layer_index: usize,
    pub layer_name: String,
    pub arrays: Vec<WeightsArrayInspection>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightsInspection {
    pub header: WeightsHeader,
    pub layers: Vec<LayerWeightsInspection>,
    pub accounted_bytes: u64,
    pub file_size: u64,
    pub trailing_bytes: u64,
}

pub fn inspect_weights_file(
    network: &Network,
    path: impl AsRef<Path>,
) -> Result<WeightsInspection> {
    let path = path.as_ref();
    let file_size = std::fs::metadata(path)?.len();
    let mut file = File::open(path)?;

    let header = read_header(&mut file)?;
    let mut cursor = header.header_bytes;
    let mut layers = Vec::new();

    for layer in &network.layers {
        let Some(weights) = layer.weights() else {
            continue;
        };

        let arrays = consume_layer_weights(
            &mut file,
            &mut cursor,
            layer,
            weights.parameter_count(),
            weights,
        )?;
        layers.push(LayerWeightsInspection {
            layer_index: layer.index,
            layer_name: layer.kind.name().to_owned(),
            arrays,
        });
    }

    let trailing_bytes = file_size.saturating_sub(cursor);
    Ok(WeightsInspection {
        header,
        layers,
        accounted_bytes: cursor,
        file_size,
        trailing_bytes,
    })
}

pub fn save_weights(
    network: &Network,
    params: &[LayerParams],
    path: impl AsRef<Path>,
    seen: u64,
) -> Result<()> {
    if params.len() != network.layers.len() {
        return Err(Error::Weights(format!(
            "weights parameter count mismatch: expected {}, got {}",
            network.layers.len(),
            params.len()
        )));
    }

    let mut file = BufWriter::with_capacity(1 << 20, File::create(path)?);
    file.write_all(&DARKNET_WEIGHTS_VERSION_MAJOR.to_le_bytes())?;
    file.write_all(&DARKNET_WEIGHTS_VERSION_MINOR.to_le_bytes())?;
    file.write_all(&DARKNET_WEIGHTS_VERSION_PATCH.to_le_bytes())?;
    file.write_all(&seen.to_le_bytes())?;

    for (layer, params) in network.layers.iter().zip(params) {
        match (&layer.kind, params) {
            (LayerKind::HyperbolicFoveation(spec), LayerParams::HyperbolicFoveation(hf)) => {
                crate::runtime::validate_hyperbolic_foveation_param_layout(spec, hf)?;
                write_f32_slice(&mut file, &hf.hf_params)?;
                if spec.pred_hidden > 0 {
                    write_f32_slice(&mut file, &hf.pred_w1)?;
                    write_f32_slice(&mut file, &hf.pred_b1)?;
                    write_f32_slice(&mut file, &hf.pred_w2)?;
                    write_f32_slice(&mut file, &hf.pred_b2)?;
                }
            }
            (LayerKind::FoveationFilm(spec), LayerParams::FoveationFilm(film)) => {
                crate::runtime::validate_foveation_film_param_layout(
                    spec,
                    layer.input_shape.channels,
                    film,
                )?;
                write_f32_slice(&mut file, &film.w_embed)?;
                write_f32_slice(&mut file, &film.b_embed)?;
                write_f32_slice(&mut file, &film.w_gamma)?;
                write_f32_slice(&mut file, &film.b_gamma)?;
                write_f32_slice(&mut file, &film.w_beta)?;
                write_f32_slice(&mut file, &film.b_beta)?;
            }
            (LayerKind::RecursiveBlock(spec), LayerParams::RecursiveBlock(recursive)) => {
                write_recursive_body_weights(&mut file, &spec.body, &recursive.body)?
            }
            (LayerKind::Convolutional(_), LayerParams::Convolutional(conv)) => {
                write_f32_slice(&mut file, &conv.biases)?;
                if let Some(batch_norm) = conv.batch_norm.as_ref() {
                    write_f32_slice(&mut file, &batch_norm.scales)?;
                    write_f32_slice(&mut file, &batch_norm.rolling_mean)?;
                    write_f32_slice(&mut file, &batch_norm.rolling_variance)?;
                }
                write_f32_slice(&mut file, &conv.weights)?;
            }
            (LayerKind::Connected(_), LayerParams::Connected(connected)) => {
                write_connected_params(&mut file, connected)?;
            }
            (LayerKind::ObbRefine(_), LayerParams::Connected(connected)) => {
                if connected.batch_norm.is_some() {
                    return Err(Error::Weights(format!(
                        "layer #{} ([obb_refine]) cannot have batch normalization parameters",
                        layer.index
                    )));
                }
                write_f32_slice(&mut file, &connected.biases)?;
                write_f32_slice(&mut file, &connected.weights)?;
            }
            (LayerKind::Deconvolutional(_), LayerParams::Convolutional(conv)) => {
                write_f32_slice(&mut file, &conv.biases)?;
                write_f32_slice(&mut file, &conv.weights)?;
            }
            (LayerKind::Local(_), LayerParams::Local(local)) => {
                write_f32_slice(&mut file, &local.biases)?;
                write_f32_slice(&mut file, &local.weights)?;
            }
            (LayerKind::Shortcut(_), LayerParams::Shortcut(shortcut)) => {
                write_f32_slice(&mut file, &shortcut.weights)?;
            }
            // Rust-only: C++ never writes a standalone [batchnorm]; order mirrors
            // `load_weights_into_params` and the C++ `save_batchnorm_weights` helper.
            (LayerKind::BatchNorm(_), LayerParams::BatchNorm(bn)) => {
                write_f32_slice(&mut file, &bn.biases)?;
                write_f32_slice(&mut file, &bn.scales)?;
                write_f32_slice(&mut file, &bn.rolling_mean)?;
                write_f32_slice(&mut file, &bn.rolling_variance)?;
            }
            (LayerKind::Gru(_), LayerParams::Gru(gru)) => {
                for params in [
                    &gru.input_z,
                    &gru.state_z,
                    &gru.input_r,
                    &gru.state_r,
                    &gru.input_h,
                    &gru.state_h,
                ] {
                    write_connected_params(&mut file, params)?;
                }
            }
            (LayerKind::Rnn(_), LayerParams::Rnn(recurrent)) => {
                write_connected_params(&mut file, &recurrent.input)?;
                write_connected_params(&mut file, &recurrent.self_params)?;
                write_connected_params(&mut file, &recurrent.output)?;
            }
            (LayerKind::Lstm(_), LayerParams::Lstm(recurrent)) => {
                write_connected_params(&mut file, &recurrent.wf)?;
                write_connected_params(&mut file, &recurrent.wi)?;
                write_connected_params(&mut file, &recurrent.wg)?;
                write_connected_params(&mut file, &recurrent.wo)?;
                write_connected_params(&mut file, &recurrent.uf)?;
                write_connected_params(&mut file, &recurrent.ui)?;
                write_connected_params(&mut file, &recurrent.ug)?;
                write_connected_params(&mut file, &recurrent.uo)?;
            }
            (LayerKind::Crnn(_), LayerParams::Crnn(recurrent)) => {
                write_convolutional_params(&mut file, &recurrent.input)?;
                write_convolutional_params(&mut file, &recurrent.self_params)?;
                write_convolutional_params(&mut file, &recurrent.output)?;
            }
            (LayerKind::Vit(_), LayerParams::Vit(vit)) => {
                write_f32_slice(&mut file, &vit.to_blob())?;
            }
            (LayerKind::Swin(_), LayerParams::Swin(swin)) => {
                let blob = swin.to_blob().map_err(Error::Weights)?;
                write_f32_slice(&mut file, &blob)?;
            }
            (LayerKind::TuckerAttention(_), LayerParams::TuckerAttention(tucker)) => {
                let blob = tucker.to_blob().map_err(Error::Weights)?;
                write_f32_slice(&mut file, &blob)?;
            }
            (LayerKind::MambaVision(_), LayerParams::MambaVision(mamba)) => {
                let blob = mamba.to_blob().map_err(Error::Weights)?;
                write_f32_slice(&mut file, &blob)?;
            }
            (LayerKind::KimiKda(_), LayerParams::KimiKda(kimi)) => {
                let blob = kimi.to_blob().map_err(Error::Weights)?;
                write_f32_slice(&mut file, &blob)?;
            }
            // C++ `save_moe_convolutional_weights`: every expert's complete
            // convolution block first, then the router biases and weights.
            (LayerKind::Moe(_), LayerParams::Moe(moe)) => {
                for expert in &moe.experts {
                    write_convolutional_params(&mut file, expert)?;
                }
                write_f32_slice(&mut file, &moe.router_biases)?;
                write_f32_slice(&mut file, &moe.router_weights)?;
            }
            (LayerKind::EmlConvolutional(_), LayerParams::EmlConvolutional(eml)) => {
                write_convolutional_params(&mut file, &eml.input)?;
                write_convolutional_params(&mut file, &eml.self_params)?;
            }
            (LayerKind::GraphConv(spec), LayerParams::GraphConv(graph)) => {
                write_convolutional_params(&mut file, &graph.projection)?;
                match (spec.use_self, &graph.self_weights) {
                    (true, Some(weights)) => write_f32_slice(&mut file, weights)?,
                    (false, None) => {}
                    _ => {
                        return Err(Error::Weights(format!(
                            "layer #{} ([graph_conv]) self-weight layout does not match config",
                            layer.index
                        )));
                    }
                }
                match (spec.edge_mode, &graph.edge_kernel, &graph.edge_biases) {
                    (0, None, None) => {}
                    (1, Some(kernel), Some(biases)) => {
                        write_f32_slice(&mut file, kernel)?;
                        write_f32_slice(&mut file, biases)?;
                    }
                    _ => {
                        return Err(Error::Weights(format!(
                            "layer #{} ([graph_conv]) edge-parameter layout does not match config",
                            layer.index
                        )));
                    }
                }
            }
            (LayerKind::Dcnv4(_), LayerParams::Dcnv4(dcnv4)) => {
                write_f32_slice(&mut file, &dcnv4.biases)?;
                if let Some(batch_norm) = &dcnv4.batch_norm {
                    write_f32_slice(&mut file, &batch_norm.scales)?;
                    write_f32_slice(&mut file, &batch_norm.rolling_mean)?;
                    write_f32_slice(&mut file, &batch_norm.rolling_variance)?;
                }
                write_f32_slice(&mut file, &dcnv4.dummy_weights)?;
                write_f32_slice(&mut file, &dcnv4.offset_weights)?;
                write_f32_slice(&mut file, &dcnv4.offset_biases)?;
            }
            (LayerKind::DeformConv(_), LayerParams::DeformConv(deform)) => {
                write_convolutional_params(&mut file, &deform.main)?;
                write_f32_slice(&mut file, &deform.offset_weights)?;
                write_f32_slice(&mut file, &deform.offset_biases)?;
                if let (Some(weights), Some(biases)) = (&deform.mask_weights, &deform.mask_biases) {
                    write_f32_slice(&mut file, weights)?;
                    write_f32_slice(&mut file, biases)?;
                }
            }
            (LayerKind::DsConv(_), LayerParams::DsConv(dsconv)) => {
                write_f32_slice(&mut file, &dsconv.collapse.biases)?;
                write_f32_slice(&mut file, &dsconv.collapse.weights)?;
                write_f32_slice(&mut file, &dsconv.offset_weights)?;
                write_f32_slice(&mut file, &dsconv.offset_biases)?;
            }
            (LayerKind::Wmhf(_), LayerParams::Wmhf(wmhf)) => {
                for conv in [
                    &wmhf.pre,
                    &wmhf.local_dw3,
                    &wmhf.local_dw5,
                    &wmhf.local_dw7,
                    &wmhf.local_mix,
                    &wmhf.hf_gate,
                    &wmhf.fuse,
                ] {
                    write_convolutional_params(&mut file, conv)?;
                }
                write_f32_slice(&mut file, &wmhf.scan_weights)?;
            }
            (LayerKind::Clifford(spec), LayerParams::Clifford(clifford)) => {
                write_f32_slice(&mut file, &clifford.layer_norm_gamma)?;
                write_f32_slice(&mut file, &clifford.layer_norm_beta)?;
                write_f32_slice(&mut file, &clifford.determinant_weights)?;
                write_f32_slice(&mut file, &clifford.determinant_biases)?;
                for conv in &clifford.depthwise {
                    write_convolutional_params(&mut file, conv)?;
                }
                write_clifford_branch(&mut file, &clifford.local)?;
                write_f32_slice(&mut file, &clifford.layer_scale)?;
                match (
                    spec.gffn_mode != crate::model::CliffordGffnMode::Local,
                    &clifford.global,
                ) {
                    (true, Some(global)) => write_clifford_branch(&mut file, global)?,
                    (false, None) => {}
                    _ => {
                        return Err(Error::Weights(format!(
                            "layer #{} ([clifford]) global parameter layout does not match config",
                            layer.index
                        )));
                    }
                }
                if spec.gffn_mode != crate::model::CliffordGffnMode::Local
                    && spec.global_op == crate::model::CliffordGlobalOp::Fourier
                {
                    let fourier = clifford.fourier.as_ref().ok_or_else(|| {
                        Error::Weights(format!(
                            "layer #{} ([clifford]) is missing Fourier weights",
                            layer.index
                        ))
                    })?;
                    write_f32_slice(&mut file, &fourier.real)?;
                    write_f32_slice(&mut file, &fourier.imaginary)?;
                } else if clifford.fourier.is_some() {
                    return Err(Error::Weights(format!(
                        "layer #{} ([clifford]) has unexpected Fourier weights",
                        layer.index
                    )));
                }
            }
            // Cfg-local pretrained encoder weights are deliberately excluded
            // from the trainable network checkpoint (`dontsave`). Without an
            // external blob, preserve C++'s sequential source-order format.
            (LayerKind::DepthAnything(spec), LayerParams::DepthAnything(weights)) => {
                if spec.weights_path.is_none() {
                    write_f32_slice(&mut file, weights.as_slice())?;
                }
            }
            (LayerKind::DptHead(spec), LayerParams::DptHead(weights)) => {
                let expected =
                    crate::dpt_head::dpt_head_weight_layout(spec.geometry(layer.input_shape))?
                        .total;
                if weights.values.len() != expected {
                    return Err(Error::Weights(format!(
                        "layer #{} ([dpt_head]) has {} parameters, expected {expected}",
                        layer.index,
                        weights.values.len()
                    )));
                }
                write_f32_slice(&mut file, &weights.values)?;
            }
            // C++ `dontsave`: the frozen LAYA layers load from an external
            // shared blob, so nothing is written for them in this stream.
            (LayerKind::LayaEncoder(_), LayerParams::LayaEncoder(_)) => {}
            (LayerKind::LayaHead(_), LayerParams::LayaHead(_)) => {}
            (_, LayerParams::None) => {}
            _ => {
                return Err(Error::Weights(format!(
                    "layer #{} ({}) and parameter set are inconsistent while saving weights",
                    layer.index,
                    layer.kind.name()
                )));
            }
        }
    }

    file.flush()?;
    Ok(())
}

fn consume_layer_weights(
    file: &mut File,
    cursor: &mut u64,
    layer: &Layer,
    _parameter_count: usize,
    weights: crate::model::LayerWeights,
) -> Result<Vec<WeightsArrayInspection>> {
    let mut arrays = Vec::with_capacity(weights.entries.len());

    for entry in weights.entries {
        let byte_len = entry.len as u64 * 4;
        let start = *cursor;
        file.seek(SeekFrom::Current(byte_len as i64))?;
        *cursor += byte_len;
        arrays.push(WeightsArrayInspection {
            name: entry.array.name().to_owned(),
            len: entry.len,
            byte_offset: start,
            byte_len,
        });
    }

    if arrays.is_empty() {
        return Err(Error::Weights(format!(
            "layer #{} ({}) reported weights but produced no arrays",
            layer.index,
            layer.kind.name()
        )));
    }

    Ok(arrays)
}

fn read_header(file: &mut File) -> Result<WeightsHeader> {
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

fn read_i32(file: &mut File) -> Result<i32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u32(file: &mut File) -> Result<u32> {
    let mut bytes = [0u8; 4];
    file.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(file: &mut File) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Write `values` as little-endian f32, encoding in chunks so large tensors
/// cost a few writes instead of one per element.
pub(crate) fn write_f32_slice<W: Write>(file: &mut W, values: &[f32]) -> Result<()> {
    const CHUNK_VALUES: usize = 64 * 1024;
    let mut bytes = Vec::with_capacity(values.len().min(CHUNK_VALUES) * 4);
    for chunk in values.chunks(CHUNK_VALUES) {
        bytes.clear();
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        file.write_all(&bytes)?;
    }
    Ok(())
}

fn write_recursive_body_weights<W: Write>(
    file: &mut W,
    body: &[Layer],
    params: &[LayerParams],
) -> Result<()> {
    if body.len() != params.len() {
        return Err(Error::Weights(
            "recursive-block body weight parameter count mismatch".to_owned(),
        ));
    }
    for (layer, params) in body.iter().zip(params) {
        match (&layer.kind, params) {
            (LayerKind::Convolutional(_), LayerParams::Convolutional(conv)) => {
                write_convolutional_params(file, conv)?;
            }
            (LayerKind::Connected(_), LayerParams::Connected(connected)) => {
                write_connected_params(file, connected)?;
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
                return Err(Error::Weights(format!(
                    "unsupported recursive-block weight layout for body layer #{} ({})",
                    layer.index,
                    layer.kind.name()
                )));
            }
        }
    }
    Ok(())
}

fn write_connected_params<W: Write>(file: &mut W, params: &ConnectedParams) -> Result<()> {
    write_f32_slice(file, &params.biases)?;
    write_f32_slice(file, &params.weights)?;
    if let Some(batch_norm) = params.batch_norm.as_ref() {
        write_f32_slice(file, &batch_norm.scales)?;
        write_f32_slice(file, &batch_norm.rolling_mean)?;
        write_f32_slice(file, &batch_norm.rolling_variance)?;
    }
    Ok(())
}

fn write_convolutional_params<W: Write>(file: &mut W, params: &ConvolutionalParams) -> Result<()> {
    write_f32_slice(file, &params.biases)?;
    if let Some(batch_norm) = params.batch_norm.as_ref() {
        write_f32_slice(file, &batch_norm.scales)?;
        write_f32_slice(file, &batch_norm.rolling_mean)?;
        write_f32_slice(file, &batch_norm.rolling_variance)?;
    }
    write_f32_slice(file, &params.weights)?;
    Ok(())
}

fn write_clifford_branch<W: Write>(
    file: &mut W,
    branch: &crate::clifford::CliffordBranchParams,
) -> Result<()> {
    write_f32_slice(file, &branch.projection_weights)?;
    write_f32_slice(file, &branch.projection_biases)?;
    write_f32_slice(file, &branch.gate_weights)?;
    write_f32_slice(file, &branch.gate_biases)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{ExecutableNetwork, LayerParams};

    use super::{inspect_weights_file, save_weights, write_f32_slice};

    #[test]
    fn chunked_f32_writer_matches_per_element_encoding() {
        for len in [0usize, 1, 65_535, 65_536, 65_537, 200_000] {
            let values: Vec<f32> = (0..len).map(|i| i as f32 * 0.37 - 5.0).collect();
            let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut got = Vec::new();
            write_f32_slice(&mut got, &values).unwrap();
            assert_eq!(got, expected, "len {len}");
        }
    }

    #[test]
    fn clifford_weights_round_trip_and_match_source_byte_order() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "clifford-weights.cfg",
                r#"[net]
width=2
height=1
channels=2
[clifford]
filters=2
shifts=1
gffn_mode=hybrid
global_op=fourier
fno_modes_h=1
fno_modes_w=2
dwconv_size=1
num_dwconv=2
"#,
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Clifford(params) = &mut executable.params[0] else {
            panic!("Clifford parameters");
        };
        let mut next = 1.0f32;
        let fill = |values: &mut [f32], next: &mut f32| {
            for value in values {
                *value = *next;
                *next += 1.0;
            }
        };
        fill(&mut params.layer_norm_gamma, &mut next);
        fill(&mut params.layer_norm_beta, &mut next);
        fill(&mut params.determinant_weights, &mut next);
        fill(&mut params.determinant_biases, &mut next);
        for conv in &mut params.depthwise {
            fill(&mut conv.biases, &mut next);
            let bn = conv.batch_norm.as_mut().unwrap();
            fill(&mut bn.scales, &mut next);
            fill(&mut bn.rolling_mean, &mut next);
            fill(&mut bn.rolling_variance, &mut next);
            fill(&mut conv.weights, &mut next);
        }
        let fill_branch = |branch: &mut crate::clifford::CliffordBranchParams, next: &mut f32| {
            fill(&mut branch.projection_weights, next);
            fill(&mut branch.projection_biases, next);
            fill(&mut branch.gate_weights, next);
            fill(&mut branch.gate_biases, next);
        };
        fill_branch(&mut params.local, &mut next);
        fill(&mut params.layer_scale, &mut next);
        fill_branch(params.global.as_mut().unwrap(), &mut next);
        let fourier = params.fourier.as_mut().unwrap();
        fill(&mut fourier.real, &mut next);
        fill(&mut fourier.imaginary, &mut next);

        let params_before = executable.params.clone();
        let LayerParams::Clifford(params) = &params_before[0] else {
            unreachable!();
        };
        fn append(values: &mut Vec<f32>, slice: &[f32]) {
            values.extend_from_slice(slice);
        }
        fn append_branch(values: &mut Vec<f32>, branch: &crate::clifford::CliffordBranchParams) {
            append(values, &branch.projection_weights);
            append(values, &branch.projection_biases);
            append(values, &branch.gate_weights);
            append(values, &branch.gate_biases);
        }
        let mut expected_values = Vec::new();
        append(&mut expected_values, &params.layer_norm_gamma);
        append(&mut expected_values, &params.layer_norm_beta);
        append(&mut expected_values, &params.determinant_weights);
        append(&mut expected_values, &params.determinant_biases);
        for conv in &params.depthwise {
            append(&mut expected_values, &conv.biases);
            let bn = conv.batch_norm.as_ref().unwrap();
            append(&mut expected_values, &bn.scales);
            append(&mut expected_values, &bn.rolling_mean);
            append(&mut expected_values, &bn.rolling_variance);
            append(&mut expected_values, &conv.weights);
        }
        append_branch(&mut expected_values, &params.local);
        append(&mut expected_values, &params.layer_scale);
        append_branch(&mut expected_values, params.global.as_ref().unwrap());
        let fourier = params.fourier.as_ref().unwrap();
        append(&mut expected_values, &fourier.real);
        append(&mut expected_values, &fourier.imaginary);
        let expected_bytes: Vec<u8> = expected_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        let path = std::env::temp_dir().join(format!(
            "darknet-clifford-weights-{}.weights",
            std::process::id()
        ));
        save_weights(&network, &executable.params, &path, 17).unwrap();
        let bytes = fs::read(&path).unwrap();
        let inspection = inspect_weights_file(&network, &path).unwrap();
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(
            &bytes[inspection.header.header_bytes as usize..],
            expected_bytes.as_slice()
        );
        let mut reloaded = ExecutableNetwork::new(network);
        reloaded.load_weights_file(&path).unwrap();
        assert_eq!(reloaded.params, params_before);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn carpn_refine_weights_round_trip_in_source_order() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "carpn-refine.cfg",
                r#"
                    [net]
                    width=4
                    height=4
                    channels=1
                    batch=1
                    obb=1

                    [convolutional]
                    filters=8
                    size=1
                    batch_normalize=0
                    activation=linear

                    [convolutional]
                    filters=8
                    size=1
                    batch_normalize=0
                    activation=linear

                    [obb_yolo]
                    classes=1
                    num=1
                    anchors=4,2
                    max=2

                    [obb_refine]
                    proposal_topk_train=3
                    proposal_topk_infer=4
                    proposal_jitter=0
                    sample_grid=2
                "#,
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let stage2_index = network.layers.len() - 1;
        let LayerParams::Connected(params) = &mut executable.params[stage2_index] else {
            panic!("CARPN refine parameters use the connected affine storage");
        };
        for (i, value) in params.biases.iter_mut().enumerate() {
            *value = i as f32 + 0.25;
        }
        for (i, value) in params.weights.iter_mut().enumerate() {
            *value = i as f32 * 0.125 - 3.0;
        }

        let modern_path = std::env::temp_dir().join("darknet-carpn-refine-modern.weights");
        save_weights(&network, &executable.params, &modern_path, 19).unwrap();
        assert_eq!(
            fs::metadata(&modern_path).unwrap().len(),
            20 + network.total_parameter_count() as u64 * 4
        );
        let inspection = inspect_weights_file(&network, &modern_path).unwrap();
        assert_eq!(inspection.trailing_bytes, 0);
        let stage2 = inspection
            .layers
            .iter()
            .find(|layer| layer.layer_index == stage2_index)
            .expect("refine weights are inspected");
        assert_eq!(
            stage2
                .arrays
                .iter()
                .map(|array| array.name.as_str())
                .collect::<Vec<_>>(),
            ["biases", "weights"]
        );
        assert_eq!(
            stage2
                .arrays
                .iter()
                .map(|array| array.len)
                .collect::<Vec<_>>(),
            [5, 5 * 2 * 2 * 8]
        );

        let mut modern = ExecutableNetwork::new(network.clone());
        modern.load_weights_file(&modern_path).unwrap();
        assert_eq!(modern.params, executable.params);

        let legacy_path = std::env::temp_dir().join("darknet-carpn-refine-legacy.weights");
        let modern_bytes = fs::read(&modern_path).unwrap();
        let mut legacy_bytes = Vec::with_capacity(modern_bytes.len() - 4);
        legacy_bytes.extend_from_slice(&0i32.to_le_bytes());
        legacy_bytes.extend_from_slice(&1i32.to_le_bytes());
        legacy_bytes.extend_from_slice(&0i32.to_le_bytes());
        legacy_bytes.extend_from_slice(&19u32.to_le_bytes());
        legacy_bytes.extend_from_slice(&modern_bytes[20..]);
        fs::write(&legacy_path, legacy_bytes).unwrap();
        let mut legacy = ExecutableNetwork::new(network.clone());
        legacy.load_weights_file(&legacy_path).unwrap();
        assert_eq!(legacy.params, executable.params);
        assert_eq!(
            inspect_weights_file(&network, &legacy_path)
                .unwrap()
                .trailing_bytes,
            0
        );
        let _ = fs::remove_file(modern_path);
        let _ = fs::remove_file(legacy_path);
    }

    #[test]
    fn graph_conv_weights_round_trip_in_source_order() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "graph-conv-weights.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=4\n[graph_conv]\nfilters=4\ngroups=2\nsize=3\npad=1\nbatch_normalize=1\nedge_mode=1\ngraph_use_self=1\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let expected_blob_len = network.layers[0].weights().unwrap().byte_len();
        {
            let LayerParams::GraphConv(graph) = &mut executable.params[0] else {
                panic!("graph_conv params");
            };
            graph
                .projection
                .biases
                .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            let batch_norm = graph.projection.batch_norm.as_mut().unwrap();
            batch_norm.scales.copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);
            batch_norm
                .rolling_mean
                .copy_from_slice(&[9.0, 10.0, 11.0, 12.0]);
            batch_norm
                .rolling_variance
                .copy_from_slice(&[13.0, 14.0, 15.0, 16.0]);
            graph.projection.weights.copy_from_slice(&[17.0; 8]);
            graph
                .self_weights
                .as_mut()
                .unwrap()
                .copy_from_slice(&[18.0; 8]);
            graph
                .edge_kernel
                .as_mut()
                .unwrap()
                .copy_from_slice(&[19.0; 72]);
            graph
                .edge_biases
                .as_mut()
                .unwrap()
                .copy_from_slice(&[20.0; 18]);
        }
        let path = std::env::temp_dir().join(format!(
            "darknet-graph-conv-weights-{}.weights",
            std::process::id()
        ));
        save_weights(&network, &executable.params, &path, 23).unwrap();
        let inspection = inspect_weights_file(&network, &path).unwrap();
        assert_eq!(inspection.header.seen, 23);
        assert_eq!(inspection.file_size, 20 + expected_blob_len);
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(
            inspection.layers[0]
                .arrays
                .iter()
                .map(|array| (array.name.as_str(), array.len))
                .collect::<Vec<_>>(),
            vec![
                ("biases", 4),
                ("scales", 4),
                ("rolling_mean", 4),
                ("rolling_variance", 4),
                ("weights", 8),
                ("graph_self_weights", 8),
                ("graph_edge_kernel", 72),
                ("graph_edge_biases", 18),
            ]
        );
        let mut restored = ExecutableNetwork::new(network.clone());
        restored.load_weights_file(&path).unwrap();
        assert_eq!(restored.params, executable.params);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn graph_conv_optional_weight_blocks_round_trip_for_all_layouts() {
        let path = std::env::temp_dir().join(format!(
            "darknet-graph-conv-layout-{}.weights",
            std::process::id()
        ));
        for batch_normalize in [false, true] {
            for edge_mode in [0usize, 1] {
                for use_self in [false, true] {
                    let config = format!(
                        "[net]\nwidth=2\nheight=2\nchannels=4\n[graph_conv]\nfilters=4\ngroups=2\nsize=3\npad=1\nbatch_normalize={}\ngraph_edge_mode={edge_mode}\ngraph_use_self={}\n",
                        usize::from(batch_normalize),
                        usize::from(use_self),
                    );
                    let network = Network::from_config(
                        ConfigFile::parse_str("graph-conv-layout.cfg", &config).unwrap(),
                    )
                    .unwrap();
                    let executable = ExecutableNetwork::new(network.clone());
                    save_weights(&network, &executable.params, &path, 0).unwrap();
                    let inspection = inspect_weights_file(&network, &path).unwrap();
                    let mut arrays = vec![("biases", 4)];
                    if batch_normalize {
                        arrays.extend([
                            ("scales", 4),
                            ("rolling_mean", 4),
                            ("rolling_variance", 4),
                        ]);
                    }
                    arrays.push(("weights", 8));
                    if use_self {
                        arrays.push(("graph_self_weights", 8));
                    }
                    if edge_mode == 1 {
                        arrays.extend([("graph_edge_kernel", 72), ("graph_edge_biases", 18)]);
                    }
                    let parameter_count: usize = arrays.iter().map(|(_, len)| len).sum();
                    assert_eq!(inspection.file_size, 20 + 4 * parameter_count as u64);
                    assert_eq!(inspection.trailing_bytes, 0);
                    assert_eq!(
                        inspection.layers[0]
                            .arrays
                            .iter()
                            .map(|array| (array.name.as_str(), array.len))
                            .collect::<Vec<_>>(),
                        arrays,
                        "BN={batch_normalize}, edge_mode={edge_mode}, self={use_self}"
                    );
                    let mut restored = ExecutableNetwork::new(network.clone());
                    restored.load_weights_file(&path).unwrap();
                    assert_eq!(restored.params, executable.params);
                }
            }
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn swin_weights_round_trip_rust_blob_order_and_reject_old_headers() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "swin-roundtrip.cfg",
                "[net]\nwidth=4\nheight=3\nchannels=4\n\
                 [swin]\nfilters=6\nsize=2\nheads=2\nshift=1\nffn_ratio=3\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let expected_blob_len;
        {
            let LayerParams::Swin(weights) = &mut executable.params[0] else {
                panic!("Swin params")
            };
            let fill = |values: &mut [f32], start: usize| {
                for (index, value) in values.iter_mut().enumerate() {
                    *value = start as f32 + index as f32 * 0.01;
                }
            };
            fill(&mut weights.qkv_weight, 1);
            fill(&mut weights.qkv_bias, 2);
            fill(&mut weights.wo, 3);
            fill(&mut weights.wo_bias, 4);
            fill(&mut weights.ln1_gamma, 5);
            fill(&mut weights.ln1_beta, 6);
            fill(&mut weights.ln2_gamma, 7);
            fill(&mut weights.ln2_beta, 8);
            fill(&mut weights.ffn_w1, 9);
            fill(&mut weights.ffn_b1, 10);
            fill(&mut weights.ffn_w2, 11);
            fill(&mut weights.ffn_b2, 12);
            fill(weights.res_proj.as_mut().unwrap(), 13);
            fill(&mut weights.relative_position_bias, 14);
            fill(&mut weights.mhc, 15);
            expected_blob_len = weights.to_blob().unwrap().len();
        }
        let path = std::env::temp_dir().join("darknet-swin-rust-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 23).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            20 + expected_blob_len as u64 * 4
        );
        let inspection = inspect_weights_file(&network, &path).unwrap();
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(
            inspection
                .layers
                .iter()
                .flat_map(|layer| layer.arrays.iter())
                .map(|array| array.name.as_str())
                .collect::<Vec<_>>(),
            [
                "swin_qkv_weights",
                "swin_qkv_bias",
                "swin_wo",
                "swin_wo_bias",
                "swin_ln1_gamma",
                "swin_ln1_beta",
                "swin_ln2_gamma",
                "swin_ln2_beta",
                "swin_ffn_w1",
                "swin_ffn_b1",
                "swin_ffn_w2",
                "swin_ffn_b2",
                "swin_res_proj",
                "swin_rel_pos_bias",
                "swin_mhc_scales",
            ]
        );
        let mut loaded = ExecutableNetwork::new(network.clone());
        loaded.load_weights_file(&path).unwrap();
        assert_eq!(loaded.params, executable.params);

        let old_path = std::env::temp_dir().join("darknet-swin-old-header.weights");
        let mut bytes = fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&4i32.to_le_bytes());
        fs::write(&old_path, bytes).unwrap();
        let mut old = ExecutableNetwork::new(network);
        assert!(old.load_weights_file(&old_path).is_err());
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(old_path);
    }

    #[test]
    fn swin_weights_round_trip_without_residual_projection() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "swin-equal-channels.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\n\
                 [swin]\nfilters=2\nsize=1\nheads=1\nffn_ratio=2\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Swin(weights) = &mut executable.params[0] else {
            panic!("Swin params")
        };
        assert!(weights.res_proj.is_none());
        weights.qkv_weight.fill(0.125);
        weights.relative_position_bias.fill(-0.25);
        let path = std::env::temp_dir().join("darknet-swin-equal-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 29).unwrap();
        let mut loaded = ExecutableNetwork::new(network.clone());
        loaded.load_weights_file(&path).unwrap();
        assert_eq!(loaded.params, executable.params);
        let inspection = inspect_weights_file(&network, &path).unwrap();
        assert_eq!(inspection.trailing_bytes, 0);
        assert!(
            !inspection.layers[0]
                .arrays
                .iter()
                .any(|array| array.name == "swin_res_proj")
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn vit_weights_round_trip_modern_and_legacy_headers() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "vit.cfg",
                "[net]\nwidth=4\nheight=4\nchannels=1\n\
                 [vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=8\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let blob_len;
        {
            let LayerParams::Vit(weights) = &mut executable.params[0] else {
                panic!("Vit params")
            };
            let fill = |v: &mut [f32], start: usize| {
                for (i, x) in v.iter_mut().enumerate() {
                    *x = start as f32 + i as f32 * 0.01;
                }
            };
            fill(&mut weights.patch_embed, 1);
            fill(&mut weights.patch_bias, 2);
            fill(&mut weights.qkv_weight, 3);
            fill(&mut weights.qkv_bias, 4);
            fill(&mut weights.ln1_gamma, 5);
            fill(&mut weights.ln1_beta, 6);
            fill(&mut weights.wo, 7);
            fill(&mut weights.wo_bias, 8);
            fill(&mut weights.ln2_gamma, 9);
            fill(&mut weights.ln2_beta, 10);
            fill(&mut weights.ffn_w1, 11);
            fill(&mut weights.ffn_b1, 12);
            fill(&mut weights.ffn_w2, 13);
            fill(&mut weights.ffn_b2, 14);
            fill(&mut weights.pos_embed, 15);
            fill(&mut weights.mhc, 16);
            blob_len = weights.to_blob().len();
        }
        let modern_path = std::env::temp_dir().join("darknet-vit-modern-roundtrip.weights");
        save_weights(&network, &executable.params, &modern_path, 17).unwrap();
        let expected_len = 20 + blob_len as u64 * 4;
        assert_eq!(fs::metadata(&modern_path).unwrap().len(), expected_len);
        let mut modern = ExecutableNetwork::new(network.clone());
        modern.load_weights_file(&modern_path).unwrap();
        assert_eq!(modern.params, executable.params);
        assert_eq!(
            inspect_weights_file(&network, &modern_path)
                .unwrap()
                .trailing_bytes,
            0
        );

        let legacy_path = std::env::temp_dir().join("darknet-vit-legacy-roundtrip.weights");
        let mut bytes = fs::read(&modern_path).unwrap();
        bytes[8..12].copy_from_slice(&4i32.to_le_bytes()); // same header width, before the mHC format revision
        bytes.truncate(bytes.len() - 12 * 4);
        fs::write(&legacy_path, bytes).unwrap();
        assert_eq!(
            fs::metadata(&legacy_path).unwrap().len(),
            20 + (blob_len as u64 - 12) * 4
        );
        let mut legacy = ExecutableNetwork::new(network);
        legacy.load_weights_file(&legacy_path).unwrap();
        let LayerParams::Vit(legacy_params) = &legacy.params[0] else {
            panic!("Vit params")
        };
        let LayerParams::Vit(modern_params) = &executable.params[0] else {
            panic!("Vit params")
        };
        let mut expected = modern_params.clone();
        expected.mhc = crate::vit::MHC_DEFAULTS.to_vec();
        assert_eq!(legacy_params, &expected);
        let _ = fs::remove_file(modern_path);
        let _ = fs::remove_file(legacy_path);
    }

    #[test]
    fn tucker_attention_weights_round_trip_bias_then_flat_factors_with_legacy_header() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "tucker-roundtrip.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=2\n\
                 [tucker_attention]\nsize=2\nheads=1\nrank=1\nrank_k=2\nrank_o=2\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::TuckerAttention(params) = &mut executable.params[0] else {
            panic!("Tucker attention params")
        };
        params.biases.copy_from_slice(&[0.25, -0.5]);
        for (index, weight) in params.weights.iter_mut().enumerate() {
            *weight = index as f32 * 0.125 - 0.25;
        }
        let weight_count = params.weights.len();

        let modern_path = std::env::temp_dir().join("darknet-tucker-roundtrip.weights");
        save_weights(&network, &executable.params, &modern_path, 71).unwrap();
        let inspection = inspect_weights_file(&network, &modern_path).unwrap();
        assert_eq!(inspection.trailing_bytes, 0);
        let arrays = &inspection.layers[0].arrays;
        assert_eq!(
            arrays
                .iter()
                .map(|array| array.name.as_str())
                .collect::<Vec<_>>(),
            ["biases", "weights"]
        );
        assert_eq!(
            arrays.iter().map(|array| array.len).collect::<Vec<_>>(),
            [2, weight_count]
        );
        assert_eq!(arrays[0].byte_offset, 20);
        assert_eq!(arrays[1].byte_offset, 28);
        let mut reloaded = ExecutableNetwork::new(network.clone());
        reloaded.load_weights_file(&modern_path).unwrap();
        assert_eq!(reloaded.params, executable.params);

        let legacy_path = std::env::temp_dir().join("darknet-tucker-legacy.weights");
        let modern_bytes = fs::read(&modern_path).unwrap();
        let mut legacy_bytes = Vec::with_capacity(modern_bytes.len() - 4);
        legacy_bytes.extend_from_slice(&0i32.to_le_bytes());
        legacy_bytes.extend_from_slice(&1i32.to_le_bytes());
        legacy_bytes.extend_from_slice(&0i32.to_le_bytes());
        legacy_bytes.extend_from_slice(&19u32.to_le_bytes());
        legacy_bytes.extend_from_slice(&modern_bytes[20..]);
        fs::write(&legacy_path, legacy_bytes).unwrap();
        let mut legacy = ExecutableNetwork::new(network);
        assert_eq!(legacy.load_weights_file(&legacy_path).unwrap().seen, 19);
        assert_eq!(legacy.params, executable.params);
        assert_eq!(
            inspect_weights_file(&legacy.network, &legacy_path)
                .unwrap()
                .trailing_bytes,
            0
        );
        let _ = fs::remove_file(modern_path);
        let _ = fs::remove_file(legacy_path);
    }

    #[test]
    fn mambavision_weights_round_trip_in_source_order_with_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "mambavision-roundtrip.cfg",
                "[net]\nwidth=2\nheight=1\nchannels=2\n\
                 [mambavision]\nfilters=4\nstate=1\nconv_size=1\ndt_rank=1\nffn_ratio=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::MambaVision(params) = &mut executable.params[0] else {
            panic!("MambaVision params")
        };
        for (index, value) in params.values.iter_mut().enumerate() {
            *value = index as f32 * 0.03125 - 0.25;
        }
        let count = params.parameter_count();
        let path = std::env::temp_dir().join("darknet-mambavision-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 31).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + count as u64 * 4);
        let inspection = inspect_weights_file(&network, &path).unwrap();
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(
            inspection.layers[0]
                .arrays
                .iter()
                .map(|array| array.name.as_str())
                .collect::<Vec<_>>(),
            [
                "mambavision_in_proj_weights",
                "mambavision_in_proj_bias",
                "mambavision_conv_x",
                "mambavision_conv_x_bias",
                "mambavision_conv_z",
                "mambavision_conv_z_bias",
                "mambavision_x_proj",
                "mambavision_dt_proj",
                "mambavision_dt_bias",
                "mambavision_A_log",
                "mambavision_D",
                "mambavision_out_proj",
                "mambavision_out_bias",
                "mambavision_res_proj",
                "mambavision_ln1_gamma",
                "mambavision_ln1_beta",
                "mambavision_ln2_gamma",
                "mambavision_ln2_beta",
                "mambavision_ffn_w1",
                "mambavision_ffn_b1",
                "mambavision_ffn_w2",
                "mambavision_ffn_b2",
            ]
        );
        let mut loaded = ExecutableNetwork::new(network);
        loaded.load_weights_file(&path).unwrap();
        assert_eq!(loaded.params, executable.params);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn standalone_batchnorm_weights_round_trip_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "bn.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=3\n[batchnorm]\n",
            )
            .expect("cfg"),
        )
        .expect("network");
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::BatchNorm(bn) = &mut executable.params[0] else {
            panic!("expected batchnorm params");
        };
        bn.biases = vec![0.1, 0.2, 0.3];
        bn.scales = vec![1.0, 2.0, 3.0];
        bn.rolling_mean = vec![4.0, 5.0, 6.0];
        bn.rolling_variance = vec![7.0, 8.0, 9.0];

        let path = std::env::temp_dir().join("darknet-batchnorm-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 3).expect("save must not fail");
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + 4 * 12);
        let mut reloaded = ExecutableNetwork::new(network);
        reloaded.load_weights_file(&path).expect("load");
        assert_eq!(reloaded.params, executable.params);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn shortcut_weights_round_trip_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "sc.cfg",
                "[net]\nwidth=2\nheight=2\nchannels=2\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
                 [shortcut]\nfrom=-2\nweights_type=per_channel\nweights_normalization=softmax\nactivation=linear\n",
            )
            .expect("cfg"),
        )
        .expect("network");
        // (n+1)*c = 2*2 = 4 shortcut weights after 2 convs of (2 bias + 4 weights)
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Shortcut(sc) = &mut executable.params[2] else {
            panic!("shortcut params")
        };
        sc.weights = vec![0.5, 1.5, 2.5, 3.5];
        let path = std::env::temp_dir().join("darknet-shortcut-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 1).expect("save");
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            20 + 4 * (2 * (2 + 4) + 4)
        );
        let mut reloaded = ExecutableNetwork::new(network.clone());
        reloaded.load_weights_file(&path).expect("load");
        assert_eq!(reloaded.params, executable.params);
        let inspection = inspect_weights_file(&network, &path).expect("inspect");
        assert_eq!(inspection.trailing_bytes, 0);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn deconvolutional_weights_round_trip_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str("d.cfg", "[net]\nwidth=2\nheight=2\nchannels=2\n[deconvolutional]\nfilters=3\nsize=2\nstride=2\nactivation=linear\n").expect("cfg"),
        )
        .expect("network");
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Convolutional(p) = &mut executable.params[0] else {
            panic!("conv-shaped params")
        };
        p.biases = vec![0.5, 1.5, 2.5];
        p.weights = (0..2 * 3 * 4).map(|i| i as f32).collect();
        let path = std::env::temp_dir().join("darknet-deconv-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 1).expect("save");
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + 4 * (3 + 24));
        let mut reloaded = ExecutableNetwork::new(network.clone());
        reloaded.load_weights_file(&path).expect("load");
        assert_eq!(reloaded.params, executable.params);
        assert_eq!(
            inspect_weights_file(&network, &path)
                .unwrap()
                .trailing_bytes,
            0
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn inspects_synthetic_weights_file() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=8
                height=8
                channels=3

                [convolutional]
                filters=4
                size=3
                stride=1
                pad=1
                batch_normalize=1
                activation=leaky
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");

        let path = std::env::temp_dir().join("darknet-core-test.weights");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0i32.to_le_bytes());
        bytes.extend_from_slice(&2i32.to_le_bytes());
        bytes.extend_from_slice(&5i32.to_le_bytes());
        bytes.extend_from_slice(&123u64.to_le_bytes());

        let params = network.total_parameter_count();
        bytes.resize(bytes.len() + params * 4, 0);
        fs::write(&path, bytes).expect("weights file should write");

        let inspection = inspect_weights_file(&network, &path).expect("inspection should succeed");
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(inspection.header.seen, 123);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn save_load_weights_round_trip() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=1
                height=1
                channels=1

                [convolutional]
                filters=2
                size=1
                stride=1
                batch_normalize=1
                activation=linear
            "#,
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("network builds");
        let mut executable = ExecutableNetwork::new(network.clone());
        match &mut executable.params[0] {
            LayerParams::Convolutional(conv) => {
                conv.biases = vec![0.25, 0.5];
                conv.weights = vec![1.5, 2.5];
                let bn = conv.batch_norm.as_mut().expect("bn exists");
                bn.scales = vec![3.0, 4.0];
                bn.rolling_mean = vec![5.0, 6.0];
                bn.rolling_variance = vec![7.0, 8.0];
            }
            other => panic!("unexpected params: {other:?}"),
        }

        let path = std::env::temp_dir().join("darknet-core-save-load.weights");
        save_weights(&network, &executable.params, &path, 42).expect("weights save");

        let mut reloaded = ExecutableNetwork::new(network);
        let header = reloaded.load_weights_file(&path).expect("weights load");
        assert_eq!(header.seen, 42);
        assert_eq!(reloaded.params, executable.params);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn recurrent_weight_blocks_round_trip_in_darknet_order() {
        for (name, layer) in [
            ("rnn", "[rnn]\nhidden=3\noutput=2\nactivation=linear\n"),
            ("lstm", "[lstm]\noutput=2\n"),
            (
                "crnn",
                "[crnn]\nsize=1\nhidden=2\noutput=2\nactivation=linear\n",
            ),
            // `[moe_convolutional]`'s block is experts-then-router, which is
            // exactly what this round trip pins.
            (
                "moe_convolutional",
                "[moe_convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=0\n\
                 activation=linear\nexperts=3\n",
            ),
            (
                "eml_convolutional",
                "[eml_convolutional]\nfilters=2\nsize=1\nstride=1\nbatch_normalize=1\nresidual=0\n",
            ),
            (
                "kimi_kda",
                "[kimi_kda]\nrank=1\nexperts=2\ntop_k=1\nhidden=3\n",
            ),
            (
                "deform_conv",
                "[deform_conv]\nfilters=2\nsize=1\nbatch_normalize=1\nmask=1\nactivation=linear\n",
            ),
            ("dsconv", "[dsconv]\nfilters=2\nsize=3\nactivation=linear\n"),
        ] {
            let cfg_text = if name == "crnn" {
                format!("[net]\nwidth=2\nheight=2\nchannels=1\nbatch=2\ntime_steps=2\n{layer}")
            } else {
                format!("[net]\ninputs=2\nbatch=2\ntime_steps=2\n{layer}")
            };
            let network = Network::from_config(
                ConfigFile::parse_str("recurrent.cfg", &cfg_text).expect("cfg parses"),
            )
            .expect("network builds");
            let mut executable = ExecutableNetwork::new(network.clone());
            for params in &mut executable.params {
                match params {
                    // The scorer's blob lives at an offset inside a shared file,
                    // so it is never part of this sequential stream.
                    LayerParams::LayaScorer(_) => {}
                    // Same for the act head.
                    LayerParams::LayaActHead(_) => {}
                    // And the encoder.
                    LayerParams::LayaEncoder(_) => {}
                    // And the head.
                    LayerParams::LayaHead(_) => {}
                    LayerParams::Moe(moe) => {
                        for expert in &mut moe.experts {
                            for value in &mut expert.weights {
                                *value = 0.25;
                            }
                        }
                        for value in &mut moe.router_weights {
                            *value = 0.5;
                        }
                    }
                    LayerParams::EmlConvolutional(eml) => {
                        for conv in [&mut eml.input, &mut eml.self_params] {
                            for value in &mut conv.weights {
                                *value = 0.25;
                            }
                        }
                    }
                    LayerParams::Dcnv4(dcnv4) => {
                        dcnv4.biases.fill(0.1);
                        dcnv4.dummy_weights.fill(0.2);
                        dcnv4.offset_weights.fill(0.3);
                        dcnv4.offset_biases.fill(0.4);
                    }
                    LayerParams::DsConv(dsconv) => {
                        dsconv.collapse.biases.fill(0.1);
                        dsconv.collapse.weights.fill(0.25);
                        dsconv.offset_weights.fill(0.6);
                        dsconv.offset_biases.fill(0.7);
                    }
                    LayerParams::DeformConv(deform) => {
                        deform.main.biases.fill(0.1);
                        deform.main.weights.fill(0.25);
                        if let Some(batch_norm) = &mut deform.main.batch_norm {
                            batch_norm.scales.fill(0.3);
                            batch_norm.rolling_mean.fill(0.4);
                            batch_norm.rolling_variance.fill(0.5);
                        }
                        deform.offset_weights.fill(0.6);
                        deform.offset_biases.fill(0.7);
                        if let Some(weights) = &mut deform.mask_weights {
                            weights.fill(0.8);
                        }
                        if let Some(biases) = &mut deform.mask_biases {
                            biases.fill(0.9);
                        }
                    }
                    LayerParams::Rnn(recurrent) => {
                        for nested in [
                            &mut recurrent.input,
                            &mut recurrent.self_params,
                            &mut recurrent.output,
                        ] {
                            for value in &mut nested.weights {
                                *value = 0.25;
                            }
                        }
                    }
                    LayerParams::Lstm(recurrent) => {
                        for nested in [
                            &mut recurrent.wf,
                            &mut recurrent.wi,
                            &mut recurrent.wg,
                            &mut recurrent.wo,
                            &mut recurrent.uf,
                            &mut recurrent.ui,
                            &mut recurrent.ug,
                            &mut recurrent.uo,
                        ] {
                            for value in &mut nested.weights {
                                *value = 0.25;
                            }
                        }
                    }
                    LayerParams::Crnn(recurrent) => {
                        for nested in [
                            &mut recurrent.input,
                            &mut recurrent.self_params,
                            &mut recurrent.output,
                        ] {
                            for value in &mut nested.weights {
                                *value = 0.25;
                            }
                        }
                    }
                    LayerParams::KimiKda(kimi) => {
                        kimi.biases.fill(0.1);
                        kimi.weights.fill(0.25);
                    }
                    LayerParams::None
                    | LayerParams::BatchNorm(_)
                    | LayerParams::Convolutional(_)
                    | LayerParams::Connected(_)
                    | LayerParams::Local(_)
                    | LayerParams::Shortcut(_)
                    | LayerParams::Gru(_)
                    | LayerParams::Vit(_)
                    | LayerParams::Swin(_)
                    | LayerParams::TuckerAttention(_)
                    | LayerParams::MambaVision(_)
                    | LayerParams::GraphConv(_)
                    | LayerParams::Wmhf(_)
                    | LayerParams::Clifford(_)
                    | LayerParams::RecursiveBlock(_)
                    | LayerParams::HyperbolicFoveation(_)
                    | LayerParams::FoveationFilm(_)
                    | LayerParams::DepthAnything(_)
                    | LayerParams::DptHead(_) => {}
                }
            }
            let path = std::env::temp_dir().join(format!("darknet-{name}-roundtrip.weights"));
            save_weights(&network, &executable.params, &path, 7).expect("weights save");
            let mut reloaded = ExecutableNetwork::new(network);
            reloaded
                .load_weights_file(&path)
                .expect("recurrent weights load");
            assert_eq!(reloaded.params, executable.params, "{name} weights differ");
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn wmhf_weights_round_trip_in_source_order_with_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "wmhf-weights.cfg",
                "[net]\nwidth=3\nheight=3\nchannels=2\n[wmhf]\nfilters=4\nbatch_normalize=1\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::Wmhf(params) = &mut executable.params[0] else {
            panic!("wmhf parameters")
        };
        for (block, conv) in [
            &mut params.pre,
            &mut params.local_dw3,
            &mut params.local_dw5,
            &mut params.local_dw7,
            &mut params.local_mix,
            &mut params.hf_gate,
            &mut params.fuse,
        ]
        .into_iter()
        .enumerate()
        {
            conv.biases.fill(0.1 + block as f32);
            conv.weights.fill(0.2 + block as f32);
            if let Some(batch_norm) = &mut conv.batch_norm {
                batch_norm.scales.fill(0.3 + block as f32);
                batch_norm.rolling_mean.fill(0.4 + block as f32);
                batch_norm.rolling_variance.fill(0.5 + block as f32);
            }
        }
        params.scan_weights.fill(0.875);
        let parameter_count = network.layers[0].weights().unwrap().parameter_count();
        let path = std::env::temp_dir().join(format!(
            "darknet-wmhf-roundtrip-{}.weights",
            std::process::id()
        ));
        save_weights(&network, &executable.params, &path, 17).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            20 + parameter_count as u64 * 4
        );
        let inspection = inspect_weights_file(&network, &path).unwrap();
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(
            inspection.layers[0]
                .arrays
                .iter()
                .map(|a| a.len)
                .sum::<usize>(),
            parameter_count
        );
        assert_eq!(
            inspection.layers[0]
                .arrays
                .iter()
                .map(|array| (array.name.as_str(), array.len))
                .collect::<Vec<_>>(),
            [
                ("biases", 4),
                ("scales", 4),
                ("rolling_mean", 4),
                ("rolling_variance", 4),
                ("weights", 8),
                ("biases", 2),
                ("weights", 18),
                ("biases", 2),
                ("weights", 50),
                ("biases", 2),
                ("weights", 98),
                ("biases", 2),
                ("scales", 2),
                ("rolling_mean", 2),
                ("rolling_variance", 2),
                ("weights", 12),
                ("biases", 4),
                ("weights", 4),
                ("biases", 4),
                ("scales", 4),
                ("rolling_mean", 4),
                ("rolling_variance", 4),
                ("weights", 16),
                ("scan_weights", 4),
            ]
        );
        let mut reloaded = ExecutableNetwork::new(network);
        reloaded.load_weights_file(&path).unwrap();
        assert_eq!(reloaded.params, executable.params);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn deform_conv_weights_round_trip_with_source_order_and_exact_file_size() {
        for batch_normalize in [false, true] {
            for use_mask in [false, true] {
                let cfg = ConfigFile::parse_str(
                    "deform-conv-weights.cfg",
                    &format!(
                        "[net]\nwidth=3\nheight=3\nchannels=4\nbatch=1\n[deform_conv]\nfilters=4\ngroups=2\nsize=3\npadding=1\nbatch_normalize={}\nuse_mask={}\nactivation=linear\n",
                        usize::from(batch_normalize),
                        usize::from(use_mask),
                    ),
                )
                .expect("cfg parses");
                let network = Network::from_config(cfg).expect("network builds");
                let mut executable = ExecutableNetwork::new(network.clone());
                let LayerParams::DeformConv(params) = &mut executable.params[0] else {
                    panic!("deform_conv params")
                };
                for (index, value) in params.main.biases.iter_mut().enumerate() {
                    *value = 0.11 + index as f32;
                }
                for (index, value) in params.main.weights.iter_mut().enumerate() {
                    *value = 0.22 + index as f32;
                }
                let mut expected = params.main.biases.clone();
                if let Some(batch_norm) = &mut params.main.batch_norm {
                    for (index, value) in batch_norm.scales.iter_mut().enumerate() {
                        *value = 0.33 + index as f32;
                    }
                    for (index, value) in batch_norm.rolling_mean.iter_mut().enumerate() {
                        *value = 0.44 + index as f32;
                    }
                    for (index, value) in batch_norm.rolling_variance.iter_mut().enumerate() {
                        *value = 0.55 + index as f32;
                    }
                    expected.extend_from_slice(&batch_norm.scales);
                    expected.extend_from_slice(&batch_norm.rolling_mean);
                    expected.extend_from_slice(&batch_norm.rolling_variance);
                }
                expected.extend_from_slice(&params.main.weights);
                for (index, value) in params.offset_weights.iter_mut().enumerate() {
                    *value = 0.66 + index as f32;
                }
                for (index, value) in params.offset_biases.iter_mut().enumerate() {
                    *value = 0.77 + index as f32;
                }
                expected.extend_from_slice(&params.offset_weights);
                expected.extend_from_slice(&params.offset_biases);
                if let Some(weights) = &mut params.mask_weights {
                    for (index, value) in weights.iter_mut().enumerate() {
                        *value = 0.88 + index as f32;
                    }
                    expected.extend_from_slice(weights);
                }
                if let Some(biases) = &mut params.mask_biases {
                    for (index, value) in biases.iter_mut().enumerate() {
                        *value = 0.99 + index as f32;
                    }
                    expected.extend_from_slice(biases);
                }

                let path = std::env::temp_dir().join(format!(
                    "darknet-deform-conv-weights-{}-{batch_normalize}-{use_mask}.weights",
                    std::process::id()
                ));
                save_weights(&network, &executable.params, &path, 19).expect("weights save");
                let inspection = inspect_weights_file(&network, &path).expect("weights inspect");
                let parameter_count = network.layers[0].weights().unwrap().parameter_count();
                assert_eq!(inspection.file_size, 20 + 4 * parameter_count as u64);
                assert_eq!(inspection.trailing_bytes, 0);
                let expected_names = match (batch_normalize, use_mask) {
                    (false, false) => vec!["biases", "weights", "offset_weights", "offset_biases"],
                    (false, true) => vec![
                        "biases",
                        "weights",
                        "offset_weights",
                        "offset_biases",
                        "mask_weights",
                        "mask_biases",
                    ],
                    (true, false) => vec![
                        "biases",
                        "scales",
                        "rolling_mean",
                        "rolling_variance",
                        "weights",
                        "offset_weights",
                        "offset_biases",
                    ],
                    (true, true) => vec![
                        "biases",
                        "scales",
                        "rolling_mean",
                        "rolling_variance",
                        "weights",
                        "offset_weights",
                        "offset_biases",
                        "mask_weights",
                        "mask_biases",
                    ],
                };
                assert_eq!(
                    inspection.layers[0]
                        .arrays
                        .iter()
                        .map(|array| array.name.as_str())
                        .collect::<Vec<_>>(),
                    expected_names
                );
                let bytes = fs::read(&path).expect("read weights file");
                let stream = bytes[20..]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect::<Vec<_>>();
                assert_eq!(stream, expected, "source-order stream mismatch");
                let mut reloaded = ExecutableNetwork::new(network);
                reloaded.load_weights_file(&path).expect("weights load");
                assert_eq!(reloaded.params, executable.params);
                let _ = fs::remove_file(path);
            }
        }
    }

    #[test]
    fn dsconv_weights_round_trip_with_exact_source_order_and_file_size() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "dsconv-weights.cfg",
                "[net]\nwidth=3\nheight=2\nchannels=2\n[dynamic_snake_conv]\nfilters=3\nsize=3\nmorph=2\nactivation=linear\n",
            )
            .unwrap(),
        )
        .unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let LayerParams::DsConv(params) = &mut executable.params[0] else {
            panic!("dsconv params")
        };
        for (index, value) in params.collapse.biases.iter_mut().enumerate() {
            *value = 0.1 + index as f32;
        }
        for (index, value) in params.collapse.weights.iter_mut().enumerate() {
            *value = 1.0 + index as f32 * 0.01;
        }
        for (index, value) in params.offset_weights.iter_mut().enumerate() {
            *value = -0.5 + index as f32 * 0.001;
        }
        for (index, value) in params.offset_biases.iter_mut().enumerate() {
            *value = 0.25 + index as f32 * 0.1;
        }
        let mut expected = params.collapse.biases.clone();
        expected.extend_from_slice(&params.collapse.weights);
        expected.extend_from_slice(&params.offset_weights);
        expected.extend_from_slice(&params.offset_biases);

        let path = std::env::temp_dir().join(format!(
            "darknet-dsconv-weights-{}.weights",
            std::process::id()
        ));
        save_weights(&network, &executable.params, &path, 23).expect("weights save");
        let inspection = inspect_weights_file(&network, &path).expect("weights inspect");
        let parameter_count = network.layers[0].weights().unwrap().parameter_count();
        assert_eq!(inspection.file_size, 20 + 4 * parameter_count as u64);
        assert_eq!(inspection.trailing_bytes, 0);
        assert_eq!(
            inspection.layers[0]
                .arrays
                .iter()
                .map(|array| array.name.as_str())
                .collect::<Vec<_>>(),
            vec!["biases", "weights", "offset_weights", "offset_biases"]
        );
        let bytes = fs::read(&path).expect("read weights");
        let stream = bytes[20..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(stream, expected);
        let mut reloaded = ExecutableNetwork::new(network.clone());
        reloaded.load_weights_file(&path).expect("weights load");
        assert_eq!(reloaded.params, executable.params);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn dcnv4_weights_round_trip_with_source_order_and_exact_file_size() {
        for batch_normalize in [false, true] {
            let cfg = ConfigFile::parse_str(
                "dcnv4-weights.cfg",
                &format!(
                    "[net]\nwidth=3\nheight=3\nchannels=2\nbatch=1\n\
                     [dcnv4]\nsize=3\npadding=1\ngroups=2\nremove_center=1\
                     \nd_stride=2\nsoftmax=1\nbatch_normalize={}\nactivation=linear\n",
                    usize::from(batch_normalize)
                ),
            )
            .expect("cfg parses");
            let network = Network::from_config(cfg).expect("network builds");
            let mut executable = ExecutableNetwork::new(network.clone());
            let LayerParams::Dcnv4(params) = &mut executable.params[0] else {
                panic!("DCNv4 params")
            };
            params.biases.fill(0.11);
            params.dummy_weights.fill(0.22);
            params.offset_weights.fill(0.33);
            params.offset_biases.fill(0.44);
            if let Some(batch_norm) = &mut params.batch_norm {
                batch_norm.scales.fill(0.55);
                batch_norm.rolling_mean.fill(0.66);
                batch_norm.rolling_variance.fill(0.77);
            }

            let path = std::env::temp_dir().join(format!(
                "darknet-dcnv4-weights-{}-{batch_normalize}.weights",
                std::process::id()
            ));
            save_weights(&network, &executable.params, &path, 19).expect("weights save");
            let inspection = inspect_weights_file(&network, &path).expect("weights inspect");
            let parameter_count = network.layers[0].weights().unwrap().parameter_count();
            assert_eq!(inspection.file_size, 20 + 4 * parameter_count as u64);
            assert_eq!(inspection.trailing_bytes, 0);
            let arrays = &inspection.layers[0].arrays;
            let expected_names = if batch_normalize {
                vec![
                    "biases",
                    "scales",
                    "rolling_mean",
                    "rolling_variance",
                    "weights",
                    "offset_weights",
                    "offset_biases",
                ]
            } else {
                vec!["biases", "weights", "offset_weights", "offset_biases"]
            };
            assert_eq!(
                arrays
                    .iter()
                    .map(|array| array.name.as_str())
                    .collect::<Vec<_>>(),
                expected_names
            );

            let mut reloaded = ExecutableNetwork::new(network);
            reloaded.load_weights_file(&path).expect("weights load");
            assert_eq!(reloaded.params, executable.params);
            let _ = fs::remove_file(path);
        }
    }
}

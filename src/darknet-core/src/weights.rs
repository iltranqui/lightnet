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
            // C++ `save_moe_convolutional_weights`: every expert's complete
            // convolution block first, then the router biases and weights.
            (LayerKind::Moe(_), LayerParams::Moe(moe)) => {
                for expert in &moe.experts {
                    write_convolutional_params(&mut file, expert)?;
                }
                write_f32_slice(&mut file, &moe.router_biases)?;
                write_f32_slice(&mut file, &moe.router_weights)?;
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
    fn vit_weights_round_trip_modern_and_legacy_headers() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "vit.cfg",
                "[net]\nwidth=4\nheight=4\nchannels=1\n\
                 [vit]\ndim=4\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=8\n",
            ).unwrap(),
        ).unwrap();
        let mut executable = ExecutableNetwork::new(network.clone());
        let blob_len;
        {
        let LayerParams::Vit(weights) = &mut executable.params[0] else { panic!("Vit params") };
        let fill = |v: &mut [f32], start: usize| {
            for (i, x) in v.iter_mut().enumerate() { *x = start as f32 + i as f32 * 0.01; }
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
        assert_eq!(inspect_weights_file(&network, &modern_path).unwrap().trailing_bytes, 0);

        let legacy_path = std::env::temp_dir().join("darknet-vit-legacy-roundtrip.weights");
        let mut bytes = fs::read(&modern_path).unwrap();
        bytes[8..12].copy_from_slice(&4i32.to_le_bytes()); // same header width, before the mHC format revision
        bytes.truncate(bytes.len() - 12 * 4);
        fs::write(&legacy_path, bytes).unwrap();
        assert_eq!(fs::metadata(&legacy_path).unwrap().len(), 20 + (blob_len as u64 - 12) * 4);
        let mut legacy = ExecutableNetwork::new(network);
        legacy.load_weights_file(&legacy_path).unwrap();
        let LayerParams::Vit(legacy_params) = &legacy.params[0] else { panic!("Vit params") };
        let LayerParams::Vit(modern_params) = &executable.params[0] else { panic!("Vit params") };
        let mut expected = modern_params.clone();
        expected.mhc = crate::vit::MHC_DEFAULTS.to_vec();
        assert_eq!(legacy_params, &expected);
        let _ = fs::remove_file(modern_path);
        let _ = fs::remove_file(legacy_path);
    }

    #[test]
    fn standalone_batchnorm_weights_round_trip_exact_length() {
        let network = Network::from_config(
            ConfigFile::parse_str("bn.cfg", "[net]\nwidth=2\nheight=2\nchannels=3\n[batchnorm]\n")
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
        let LayerParams::Shortcut(sc) = &mut executable.params[2] else { panic!("shortcut params") };
        sc.weights = vec![0.5, 1.5, 2.5, 3.5];
        let path = std::env::temp_dir().join("darknet-shortcut-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 1).expect("save");
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + 4 * (2 * (2 + 4) + 4));
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
        let LayerParams::Convolutional(p) = &mut executable.params[0] else { panic!("conv-shaped params") };
        p.biases = vec![0.5, 1.5, 2.5];
        p.weights = (0..2 * 3 * 4).map(|i| i as f32).collect();
        let path = std::env::temp_dir().join("darknet-deconv-roundtrip.weights");
        save_weights(&network, &executable.params, &path, 1).expect("save");
        assert_eq!(fs::metadata(&path).unwrap().len(), 20 + 4 * (3 + 24));
        let mut reloaded = ExecutableNetwork::new(network.clone());
        reloaded.load_weights_file(&path).expect("load");
        assert_eq!(reloaded.params, executable.params);
        assert_eq!(inspect_weights_file(&network, &path).unwrap().trailing_bytes, 0);
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
                    LayerParams::None
                    | LayerParams::BatchNorm(_)
                    | LayerParams::Convolutional(_)
                    | LayerParams::Connected(_)
                    | LayerParams::Local(_)
                    | LayerParams::Shortcut(_)
                    | LayerParams::Gru(_)
                    | LayerParams::Vit(_) => {}
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
}

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Error, Result};
use crate::model::{Layer, Network};

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

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::cfg::ConfigFile;
    use crate::model::Network;

    use super::inspect_weights_file;

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

        let path = PathBuf::from(std::env::temp_dir()).join("darknet-core-test.weights");
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
}

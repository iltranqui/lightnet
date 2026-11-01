//! Checkpoints: Darknet-compatible `.weights` and our own resume file.
//!
//! `.weights` is written **unfolded** (bias, BN scales, rolling mean/var, weights) exactly like
//! Darknet's training checkpoints, so the same file loads in DarkSuite and through
//! [`crate::weights::load`] (`_shared/darknet-semantics.md#training-state-files`).
//! The `.state` file has no compatibility claim: Darknet's `optimizer.state` layout is unpublished.

use crate::error::{Error, Result};
use crate::model::LayerKind;
use crate::train::net::TrainNet;
use std::io::{BufWriter, Write};
use std::path::Path;

fn io(e: std::io::Error, p: &Path) -> Error {
    Error::Io { path: p.display().to_string(), source: e }
}

/// Write `path` in Darknet's binary format; `seen` goes in the header.
pub fn save_weights(path: impl AsRef<Path>, net: &TrainNet, seen: u64) -> Result<()> {
    let p = path.as_ref();
    let mut w = BufWriter::new(std::fs::File::create(p).map_err(|e| io(e, p))?);
    let werr = |e: std::io::Error| io(e, p);
    for v in [0i32, 2, 0] {
        w.write_all(&v.to_le_bytes()).map_err(werr)?;
    }
    w.write_all(&seen.to_le_bytes()).map_err(werr)?;
    let mut floats = |values: &[f32]| -> Result<()> {
        for v in values {
            w.write_all(&v.to_le_bytes()).map_err(werr)?;
        }
        Ok(())
    };
    for (i, layer) in net.model.layers.iter().enumerate() {
        let LayerKind::Conv(c) = &layer.kind else { continue };
        let conv = net.convs[i].as_ref().expect("conv state");
        floats(&conv.bias)?;
        if c.cfg.batch_normalize {
            floats(&conv.scales)?;
            floats(&conv.rolling_mean)?;
            floats(&conv.rolling_var)?;
        }
        floats(&conv.weights)?;
    }
    w.flush().map_err(werr)
}

/// Load a Darknet `.weights` file into a training net; returns `seen`.
pub fn load_weights(path: impl AsRef<Path>, net: &mut TrainNet) -> Result<u64> {
    let p = path.as_ref();
    let bytes = std::fs::read(p).map_err(|e| io(e, p))?;
    let mut pos = 0usize;
    let mut take = |n: usize, what: &str| -> Result<&[u8]> {
        let end = pos + n;
        if end > bytes.len() {
            return Err(Error::Weights(format!("{}: truncated reading {what}", p.display())));
        }
        let s = &bytes[pos..end];
        pos = end;
        Ok(s)
    };
    let i32_at = |b: &[u8]| i32::from_le_bytes(b.try_into().unwrap());
    let major = i32_at(take(4, "major")?);
    let minor = i32_at(take(4, "minor")?);
    let _rev = i32_at(take(4, "revision")?);
    let seen = if major * 10 + minor >= 2 {
        u64::from_le_bytes(take(8, "seen")?.try_into().unwrap())
    } else {
        u32::from_le_bytes(take(4, "seen")?.try_into().unwrap()) as u64
    };
    let mut floats = |n: usize, what: &str| -> Result<Vec<f32>> {
        let b = take(n * 4, what)?;
        Ok(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    };
    for (i, layer) in net.model.layers.iter().enumerate() {
        let LayerKind::Conv(c) = &layer.kind else { continue };
        let n = c.cfg.filters;
        let conv = net.convs[i].as_mut().expect("conv state");
        conv.bias = floats(n, &format!("layer {i} biases"))?;
        if c.cfg.batch_normalize {
            conv.scales = floats(n, &format!("layer {i} scales"))?;
            conv.rolling_mean = floats(n, &format!("layer {i} rolling mean"))?;
            conv.rolling_var = floats(n, &format!("layer {i} rolling variance"))?;
            conv.mean.copy_from_slice(&conv.rolling_mean);
            conv.var.copy_from_slice(&conv.rolling_var);
        }
        conv.weights = floats(c.n_weights(), &format!("layer {i} weights"))?;
    }
    if pos != bytes.len() {
        return Err(Error::Weights(format!(
            "{}: {} trailing bytes after the last layer",
            p.display(),
            bytes.len() - pos
        )));
    }
    Ok(seen)
}

const STATE_MAGIC: &[u8] = b"HIPDARKST\x01\x00\x00\x00";

/// Save weights + momentum buffers + `seen` so a run can resume exactly.
pub fn save_state(path: impl AsRef<Path>, net: &TrainNet, seen: u64) -> Result<()> {
    let p = path.as_ref();
    let mut w = BufWriter::new(std::fs::File::create(p).map_err(|e| io(e, p))?);
    let werr = |e: std::io::Error| io(e, p);
    w.write_all(STATE_MAGIC).map_err(werr)?;
    w.write_all(&seen.to_le_bytes()).map_err(werr)?;
    let mut floats = |values: &[f32]| -> Result<()> {
        for v in values {
            w.write_all(&v.to_le_bytes()).map_err(werr)?;
        }
        Ok(())
    };
    for conv in net.convs.iter().flatten() {
        floats(&conv.weights)?;
        floats(&conv.bias)?;
        floats(&conv.scales)?;
        floats(&conv.rolling_mean)?;
        floats(&conv.rolling_var)?;
        floats(&conv.weight_updates)?;
        floats(&conv.bias_updates)?;
        floats(&conv.scale_updates)?;
    }
    w.flush().map_err(werr)
}

/// Restore what [`save_state`] wrote; returns `seen`.
pub fn load_state(path: impl AsRef<Path>, net: &mut TrainNet) -> Result<u64> {
    let p = path.as_ref();
    let bytes = std::fs::read(p).map_err(|e| io(e, p))?;
    if bytes.len() < STATE_MAGIC.len() || &bytes[..STATE_MAGIC.len()] != STATE_MAGIC {
        return Err(Error::Weights(format!("{}: not a hip_darknet training state file", p.display())));
    }
    let mut pos = STATE_MAGIC.len();
    let seen = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
    pos += 8;
    let mut floats = |n: usize| -> Result<Vec<f32>> {
        let end = pos + n * 4;
        if end > bytes.len() {
            return Err(Error::Weights(format!("{}: truncated state file", p.display())));
        }
        let v = bytes[pos..end].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        pos = end;
        Ok(v)
    };
    for conv in net.convs.iter_mut().flatten() {
        let nw = conv.weights.len();
        let nb = conv.bias.len();
        conv.weights = floats(nw)?;
        conv.bias = floats(nb)?;
        conv.scales = floats(nb)?;
        conv.rolling_mean = floats(nb)?;
        conv.rolling_var = floats(nb)?;
        conv.weight_updates = floats(nw)?;
        conv.bias_updates = floats(nb)?;
        conv.scale_updates = floats(nb)?;
    }
    Ok(seen)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::weights;

    const CFG: &str = "[net]\nwidth=8\nheight=8\nchannels=3\nbatch=1\n\
        [convolutional]\nbatch_normalize=1\nfilters=2\nsize=3\npad=1\nactivation=leaky\n\
        [convolutional]\nfilters=3\nsize=1\nactivation=linear\n";

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("hipd_{name}_{}", std::process::id()))
    }

    #[test]
    fn weights_round_trip_and_inference_fold() {
        let cfg = Cfg::parse(CFG).unwrap();
        let mut net = TrainNet::new(&cfg, 1, 3).unwrap();
        // Distinct values so a mis-ordered field would show.
        {
            let c = net.convs[0].as_mut().unwrap();
            for (i, v) in c.weights.iter_mut().enumerate() {
                *v = i as f32 * 0.01;
            }
            for i in 0..2 {
                c.scales[i] = 1.5 + i as f32;
                c.rolling_mean[i] = 0.25 * i as f32;
                c.rolling_var[i] = 2.0 + i as f32;
                c.bias[i] = 0.125 * i as f32;
            }
        }
        let p = tmp("weights");
        save_weights(&p, &net, 48000).unwrap();

        // Round trip into a fresh net.
        let mut other = TrainNet::new(&cfg, 1, 99).unwrap();
        let seen = load_weights(&p, &mut other).unwrap();
        assert_eq!(seen, 48000);
        for (a, b) in net.convs.iter().zip(&other.convs) {
            let (a, b) = (a.as_ref().unwrap(), b.as_ref().unwrap());
            assert_eq!(a.weights, b.weights);
            assert_eq!(a.bias, b.bias);
            assert_eq!(a.scales, b.scales);
            assert_eq!(a.rolling_mean, b.rolling_mean);
            assert_eq!(a.rolling_var, b.rolling_var);
        }

        // The same file must load through the inference loader (BN folded) and to EOF.
        let model = crate::Model::from_parts(CFG, &std::fs::read(&p).unwrap()).unwrap();
        assert_eq!(model.seen, 48000);
        assert_eq!(weights::expected_floats(&model) * 4 + 20, std::fs::metadata(&p).unwrap().len() as usize);
        let convs: Vec<_> = model.convs().collect();
        let k = 1.5f64 / (2.0f64 + 1e-5).sqrt();
        assert!((convs[0].1.bias[0] as f64).abs() < 1e-6);
        assert!((convs[0].1.weights[4] as f64 - 0.04 * k).abs() < 1e-6, "{}", convs[0].1.weights[4]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn state_round_trip() {
        let cfg = Cfg::parse(CFG).unwrap();
        let mut net = TrainNet::new(&cfg, 2, 7).unwrap();
        net.convs[1].as_mut().unwrap().weight_updates[0] = -3.5;
        net.convs[0].as_mut().unwrap().scale_updates[1] = 0.75;
        let p = tmp("state");
        save_state(&p, &net, 123).unwrap();
        let mut other = TrainNet::new(&cfg, 2, 8).unwrap();
        assert_eq!(load_state(&p, &mut other).unwrap(), 123);
        assert_eq!(other.convs[1].as_ref().unwrap().weight_updates[0], -3.5);
        assert_eq!(other.convs[0].as_ref().unwrap().scale_updates[1], 0.75);
        assert_eq!(other.convs[0].as_ref().unwrap().weights, net.convs[0].as_ref().unwrap().weights);
        std::fs::remove_file(&p).ok();
    }
}

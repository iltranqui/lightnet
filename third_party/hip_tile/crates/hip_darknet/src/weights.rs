//! `.weights` reader: header, per-conv `bias, [bn scales, mean, var], weights`, BN folding.

use crate::error::{Error, Result};
use crate::model::{Bn, LayerKind, Model};

/// Darknet folds BN with `sqrt(var + 1e-5)` (`fuse_conv_batchnorm`, darknet_network.cpp:2905).
pub const BN_EPSILON: f64 = 0.00001;

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn bytes(&mut self, n: usize, what: &str) -> Result<&[u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len());
        let end = end.ok_or_else(|| {
            Error::Weights(format!("file too short reading {what} (need {n} bytes at offset {})", self.pos))
        })?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn i32(&mut self, what: &str) -> Result<i32> {
        Ok(i32::from_le_bytes(self.bytes(4, what)?.try_into().unwrap()))
    }
    fn floats(&mut self, n: usize, what: &str) -> Result<Vec<f32>> {
        Ok(self.bytes(n * 4, what)?.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }
}

/// Fold BN into conv: `w' = w * s / sqrt(var+eps)`, `b' = b - s * mean / sqrt(var+eps)`.
/// Arithmetic in f64 then rounded, like Darknet.
pub(crate) fn fold_bn(weights: &mut [f32], bias: &mut [f32], bn: &Bn, per_filter: usize) {
    for f in 0..bias.len() {
        let inv = (bn.var[f] as f64 + BN_EPSILON).sqrt();
        bias[f] = (bias[f] as f64 - bn.scales[f] as f64 * bn.mean[f] as f64 / inv) as f32;
        let pre = bn.scales[f] as f64 / inv;
        for w in &mut weights[f * per_filter..(f + 1) * per_filter] {
            *w = (*w as f64 * pre) as f32;
        }
    }
}

/// Number of f32 values the conv layers of `model` consume (excluding the header).
pub fn expected_floats(model: &Model) -> usize {
    model
        .convs()
        .map(|(_, c)| c.cfg.filters * if c.cfg.batch_normalize { 4 } else { 1 } + c.n_weights())
        .sum()
}

/// Read `bytes` into `model`, folding BN. The file must be consumed exactly.
pub fn load(model: &mut Model, bytes: &[u8]) -> Result<()> {
    let mut r = Reader { buf: bytes, pos: 0 };
    let major = r.i32("major")?;
    let minor = r.i32("minor")?;
    let _revision = r.i32("revision")?;
    if major > 1000 || minor > 1000 {
        return Err(Error::Weights("transposed weights (major/minor > 1000) not supported".into()));
    }
    model.seen = if major * 10 + minor >= 2 {
        u64::from_le_bytes(r.bytes(8, "seen")?.try_into().unwrap())
    } else {
        u32::from_le_bytes(r.bytes(4, "seen")?.try_into().unwrap()) as u64
    };
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let LayerKind::Conv(c) = &mut layer.kind else { continue };
        let n = c.cfg.filters;
        let what = |s: &str| format!("layer {i} {s}");
        c.bias = r.floats(n, &what("biases"))?;
        let bn = if c.cfg.batch_normalize {
            Some(Bn {
                scales: r.floats(n, &what("bn scales"))?,
                mean: r.floats(n, &what("bn mean"))?,
                var: r.floats(n, &what("bn variance"))?,
            })
        } else {
            None
        };
        c.weights = r.floats(c.n_weights(), &what("weights"))?;
        if let Some(bn) = bn {
            let k = c.k();
            fold_bn(&mut c.weights, &mut c.bias, &bn, k);
        }
    }
    if r.pos != bytes.len() {
        return Err(Error::Weights(format!(
            "{} trailing bytes after the last layer (read {} of {})",
            bytes.len() - r.pos,
            r.pos,
            bytes.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;

    fn f(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn header(seen: u64) -> Vec<u8> {
        let mut b = vec![];
        for v in [0i32, 2, 5] {
            b.extend(v.to_le_bytes());
        }
        b.extend(seen.to_le_bytes());
        b
    }

    const CFG: &str = "[net]\nwidth=4\nheight=4\nchannels=1\n\
        [convolutional]\nbatch_normalize=1\nfilters=2\nsize=3\npad=1\nactivation=leaky\n\
        [convolutional]\nfilters=3\nsize=1\nactivation=linear\n";

    fn blob() -> Vec<u8> {
        let mut b = header(48000);
        b.extend(f(&[0.5, -1.0])); // biases
        b.extend(f(&[2.0, 1.0])); // scales
        b.extend(f(&[1.0, 0.0])); // mean
        b.extend(f(&[3.00_f32 - 0.00001, 1.0])); // var
        b.extend(f(&(0..18).map(|v| v as f32).collect::<Vec<_>>())); // 2*1*3*3
        b.extend(f(&[0.1, 0.2, 0.3])); // biases of conv 2
        b.extend(f(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])); // 3*2
        b
    }

    #[test]
    fn float_count_matches_and_folds() {
        let b = blob();
        let mut m = Model::from_cfg(&Cfg::parse(CFG).unwrap()).unwrap();
        assert_eq!(expected_floats(&m) * 4 + 20, b.len());
        load(&mut m, &b).unwrap();
        assert_eq!(m.seen, 48000);
        let convs: Vec<_> = m.convs().collect();
        let c0 = convs[0].1;
        // filter 0: var+eps = 3 -> inv=sqrt(3); scale/inv = 2/sqrt(3); bias = 0.5 - 2*1/sqrt(3).
        let k = 2.0f64 / 3f64.sqrt();
        assert!((c0.bias[0] as f64 - (0.5 - k)).abs() < 1e-6);
        assert!((c0.weights[4] as f64 - 4.0 * k).abs() < 1e-5);
        // filter 1: var+eps = 1+1e-5, scale 1, mean 0.
        let k1 = 1.0 / (1.0f64 + 1e-5).sqrt();
        assert!((c0.bias[1] as f64 - -1.0).abs() < 1e-6);
        assert!((c0.weights[9 + 2] as f64 - 11.0 * k1).abs() < 1e-5);
        // Layout preserved for the unnormalized conv.
        assert_eq!(convs[1].1.weights, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(convs[1].1.bias, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn eof_must_match_exactly() {
        let mut m = Model::from_cfg(&Cfg::parse(CFG).unwrap()).unwrap();
        let mut b = blob();
        b.extend([0u8; 4]);
        assert!(matches!(load(&mut m, &b), Err(Error::Weights(e)) if e.contains("trailing")));
        let mut m = Model::from_cfg(&Cfg::parse(CFG).unwrap()).unwrap();
        let b = blob();
        assert!(matches!(load(&mut m, &b[..b.len() - 4]), Err(Error::Weights(e)) if e.contains("too short")));
    }
}

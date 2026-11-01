//! Layer graph with inferred shapes and (after loading) BN-folded conv weights.

use crate::cfg::{Activation, Cfg, ConvCfg, LayerCfg, MaxpoolCfg, NetCfg, YoloCfg};
use crate::error::{Error, Result};
use std::path::Path;

/// Activation tensor shape, laid out CHW (NCHW with batch 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl Shape {
    pub fn len(&self) -> usize {
        self.c * self.h * self.w
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Convolution with batch-norm already folded into `weights` / `bias`.
#[derive(Debug, Clone)]
pub struct Conv {
    pub cfg: ConvCfg,
    pub c_in: usize,
    /// Contiguous `[out][in * kh * kw]` (Darknet order: in, then ky, then kx). Empty until weights are loaded.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

/// Raw batch-norm parameters, only alive while loading.
pub(crate) struct Bn {
    pub scales: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

impl Conv {
    /// Number of weights per output channel (`in * kh * kw`).
    pub fn k(&self) -> usize {
        self.c_in * self.cfg.size * self.cfg.size
    }
    pub fn n_weights(&self) -> usize {
        self.cfg.filters * self.k()
    }
}

#[derive(Debug, Clone)]
pub enum LayerKind {
    Conv(Conv),
    Maxpool(MaxpoolCfg),
    /// Concatenation of channel group `group_id` of each source layer's output.
    Route { sources: Vec<usize>, groups: usize, group_id: usize },
    Upsample { stride: usize },
    /// Output is the (raw, pre-activation) input; see `yolo::decode`.
    Yolo(YoloCfg),
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub kind: LayerKind,
    /// Shape of the primary input (previous layer; for routes, the first source).
    pub in_shape: Shape,
    pub out_shape: Shape,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub net: NetCfg,
    pub layers: Vec<Layer>,
    /// Header `seen` count, informational.
    pub seen: u64,
}

impl Model {
    /// Infer shapes. Weights are zero-sized until `load_weights`.
    pub fn from_cfg(cfg: &Cfg) -> Result<Model> {
        let net = cfg.net.clone();
        let input = Shape { c: net.channels, h: net.height, w: net.width };
        let mut layers: Vec<Layer> = Vec::with_capacity(cfg.layers.len());
        for (i, lc) in cfg.layers.iter().enumerate() {
            let prev = layers.last().map_or(input, |l| l.out_shape);
            let bad = |m: String| Error::Model(format!("layer {i}: {m}"));
            let layer = match lc {
                LayerCfg::Conv(c) => {
                    let p = c.padding;
                    if prev.h + 2 * p < c.size || prev.w + 2 * p < c.size {
                        return Err(bad("input smaller than kernel".into()));
                    }
                    let out = Shape {
                        c: c.filters,
                        h: (prev.h + 2 * p - c.size) / c.stride + 1,
                        w: (prev.w + 2 * p - c.size) / c.stride + 1,
                    };
                    let conv = Conv { cfg: c.clone(), c_in: prev.c, weights: vec![], bias: vec![] };
                    Layer { kind: LayerKind::Conv(conv), in_shape: prev, out_shape: out }
                }
                LayerCfg::Maxpool(m) => {
                    if prev.h + m.padding < m.size || prev.w + m.padding < m.size {
                        return Err(bad("input smaller than pool window".into()));
                    }
                    let out = Shape {
                        c: prev.c,
                        h: (prev.h + m.padding - m.size) / m.stride + 1,
                        w: (prev.w + m.padding - m.size) / m.stride + 1,
                    };
                    Layer { kind: LayerKind::Maxpool(m.clone()), in_shape: prev, out_shape: out }
                }
                LayerCfg::Route(r) => {
                    let mut sources = vec![];
                    for &v in &r.layers {
                        let idx = if v < 0 { i as i64 + v as i64 } else { v as i64 };
                        if idx < 0 || idx >= i as i64 {
                            return Err(bad(format!("route source {v} out of range")));
                        }
                        sources.push(idx as usize);
                    }
                    let first = layers[sources[0]].out_shape;
                    let mut c = 0;
                    for &s in &sources {
                        let sh = layers[s].out_shape;
                        if (sh.h, sh.w) != (first.h, first.w) {
                            return Err(bad("route sources differ in H/W".into()));
                        }
                        if !sh.c.is_multiple_of(r.groups) {
                            return Err(bad("route source channels not divisible by groups".into()));
                        }
                        c += sh.c / r.groups;
                    }
                    Layer {
                        kind: LayerKind::Route { sources, groups: r.groups, group_id: r.group_id },
                        in_shape: first,
                        out_shape: Shape { c, ..first },
                    }
                }
                LayerCfg::Upsample { stride } => Layer {
                    kind: LayerKind::Upsample { stride: *stride },
                    in_shape: prev,
                    out_shape: Shape { c: prev.c, h: prev.h * stride, w: prev.w * stride },
                },
                LayerCfg::Yolo(y) => {
                    let want = y.mask.len() * (y.classes + 5);
                    if prev.c != want {
                        return Err(bad(format!(
                            "yolo expects {want} input channels (mask*(classes+5)), got {}",
                            prev.c
                        )));
                    }
                    Layer { kind: LayerKind::Yolo(y.clone()), in_shape: prev, out_shape: prev }
                }
            };
            layers.push(layer);
        }
        Ok(Model { net, layers, seen: 0 })
    }

    /// Parse cfg text and read weights bytes (BN folded, EOF checked).
    pub fn from_parts(cfg_text: &str, weights: &[u8]) -> Result<Model> {
        let mut m = Model::from_cfg(&Cfg::parse(cfg_text)?)?;
        crate::weights::load(&mut m, weights)?;
        Ok(m)
    }

    pub fn load(cfg_path: impl AsRef<Path>, weights_path: impl AsRef<Path>) -> Result<Model> {
        let rd = |p: &Path| std::fs::read(p).map_err(|source| Error::Io { path: p.display().to_string(), source });
        let cfg = rd(cfg_path.as_ref())?;
        let w = rd(weights_path.as_ref())?;
        Model::from_parts(&String::from_utf8_lossy(&cfg), &w)
    }

    pub fn input_shape(&self) -> Shape {
        Shape { c: self.net.channels, h: self.net.height, w: self.net.width }
    }

    /// Indices of the `[yolo]` layers, in cfg order.
    pub fn yolo_layers(&self) -> Vec<usize> {
        (0..self.layers.len()).filter(|&i| matches!(self.layers[i].kind, LayerKind::Yolo(_))).collect()
    }

    /// The conv layers (index, conv) in cfg order.
    pub fn convs(&self) -> impl Iterator<Item = (usize, &Conv)> {
        self.layers.iter().enumerate().filter_map(|(i, l)| match &l.kind {
            LayerKind::Conv(c) => Some((i, c)),
            _ => None,
        })
    }
}

impl Activation {
    /// Darknet `leaky_activate` uses slope 0.1; logistic is `1 / (1 + e^-x)`.
    pub fn apply(self, x: f32) -> f32 {
        match self {
            Activation::Linear => x,
            Activation::Leaky => {
                if x > 0.0 {
                    x
                } else {
                    0.1 * x
                }
            }
            Activation::Logistic => 1.0 / (1.0 + (-x).exp()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "[net]\nwidth=14\nheight=10\nchannels=3\n\
        [convolutional]\nfilters=8\nsize=3\nstride=2\npad=1\n\
        [route]\nlayers=-1\ngroups=2\ngroup_id=1\n\
        [maxpool]\nsize=2\nstride=2\n\
        [upsample]\nstride=2\n";

    fn shapes(m: &Model) -> Vec<(usize, usize, usize)> {
        m.layers.iter().map(|l| (l.out_shape.c, l.out_shape.h, l.out_shape.w)).collect()
    }

    #[test]
    fn shapes_route_groups_upsample_maxpool() {
        let m = Model::from_cfg(&Cfg::parse(HEAD).unwrap()).unwrap();
        // conv: (10+2-3)/2+1=5, (14+2-3)/2+1=7; group route halves channels;
        // maxpool pad=size-1: (5+1-2)/2+1=3, (7+1-2)/2+1=4; upsample x2.
        assert_eq!(shapes(&m), vec![(8, 5, 7), (4, 5, 7), (4, 3, 4), (4, 6, 8)]);
    }

    #[test]
    fn route_absolute_and_relative_concat() {
        // layer 4: -1 (upsample 4x6x8) is incompatible with 0 (8x5x7): error.
        let bad = format!("{HEAD}[route]\nlayers=-1, 0\n");
        assert!(Model::from_cfg(&Cfg::parse(&bad).unwrap()).is_err());
        // layers 1 (4x5x7) and 0 (8x5x7) concat -> 12 channels; same via -4, -5 -> idx 0 and -1.
        let ok = "[net]\nwidth=14\nheight=10\nchannels=3\n[convolutional]\nfilters=8\nsize=3\npad=1\n\
            [route]\nlayers=-1\ngroups=2\ngroup_id=1\n[route]\nlayers=-1, 0\n[route]\nlayers=-1, 1\n";
        let m = Model::from_cfg(&Cfg::parse(ok).unwrap()).unwrap();
        assert_eq!(m.layers[2].out_shape.c, 12);
        assert_eq!(m.layers[3].out_shape.c, 16);
        let LayerKind::Route { sources, .. } = &m.layers[3].kind else { panic!() };
        assert_eq!(sources, &vec![2, 1]);
        assert!(Model::from_cfg(&Cfg::parse(&ok.replace("-1, 1", "-1, 3")).unwrap()).is_err());
    }

    #[test]
    fn yolo_channel_check() {
        let t = "[net]\nwidth=8\nheight=8\nchannels=3\n[convolutional]\nfilters=30\nsize=1\n\
            [yolo]\nmask=0,1,2\nanchors=1,1,2,2,3,3\nnum=3\nclasses=5\n";
        assert!(Model::from_cfg(&Cfg::parse(t).unwrap()).is_ok());
        assert!(Model::from_cfg(&Cfg::parse(&t.replace("classes=5", "classes=4")).unwrap()).is_err());
    }
}

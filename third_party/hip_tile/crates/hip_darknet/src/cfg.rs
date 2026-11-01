//! Darknet `.cfg` parser. Only inference-relevant keys are read; training-only keys are ignored,
//! unsupported layer types or inference-changing values are errors.

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Leaky,
    Logistic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmsKind {
    /// Plain IoU (`default`, `cornersnms`).
    Default,
    /// DIoU with Darknet's hard-coded exponent 0.6 (`greedynms`).
    Greedy,
    /// DIoU with `beta_nms` as exponent (`diounms`).
    Diou,
    /// CIoU (DIoU plus the aspect-ratio term). Not a Darknet cfg value: only set through
    /// `DetectOptions::nms_kind`.
    Ciou,
}

/// `iou_loss` of a `[yolo]` head. Defaults to `Mse` like Darknet (`darknet_cfg.cpp:2452`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IouLoss {
    Mse,
    Iou,
    Giou,
    Diou,
    Ciou,
}

/// `policy` of the learning-rate schedule.
#[derive(Debug, Clone, PartialEq)]
pub enum LrPolicy {
    Constant,
    /// Rate is multiplied by `scales[i]` once `steps[i]` iterations have passed (`darknet_network.cpp:718-735`).
    Steps { steps: Vec<usize>, scales: Vec<f32> },
}

/// Training-only `[net]` keys. Parsed always, used only by `train`. Semantics:
/// `_shared/darknet-semantics.md` (Training sections).
#[derive(Debug, Clone, PartialEq)]
pub struct TrainCfg {
    pub batch: usize,
    pub subdivisions: usize,
    pub momentum: f32,
    pub decay: f32,
    pub learning_rate: f32,
    pub burn_in: usize,
    pub max_batches: usize,
    /// `power` default 4 (`darknet_cfg.cpp:1678`).
    pub power: f32,
    pub policy: LrPolicy,
    pub angle: f32,
    pub flip: bool,
    pub saturation: f32,
    pub exposure: f32,
    pub hue: f32,
}

impl Default for TrainCfg {
    fn default() -> Self {
        TrainCfg {
            batch: 1,
            subdivisions: 1,
            momentum: 0.9,
            decay: 0.0005,
            learning_rate: 0.001,
            burn_in: 0,
            max_batches: 0,
            power: 4.0,
            policy: LrPolicy::Constant,
            angle: 0.0,
            flip: false,
            saturation: 1.0,
            exposure: 1.0,
            hue: 0.0,
        }
    }
}

impl TrainCfg {
    /// Iterations of `batch * subdivisions` images in one update (`darknet_network.cpp:589`).
    pub fn effective_batch(&self) -> usize {
        self.batch * self.subdivisions
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NetCfg {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvCfg {
    pub filters: usize,
    pub size: usize,
    pub stride: usize,
    /// Zero padding per side (`size / 2` when `pad=1`, else `padding`).
    pub padding: usize,
    pub batch_normalize: bool,
    pub activation: Activation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaxpoolCfg {
    pub size: usize,
    pub stride: usize,
    /// Total padding (Darknet default `size - 1`); the window offset is `-padding / 2`.
    pub padding: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouteCfg {
    /// As written in the cfg: negative = relative to this layer, else absolute.
    pub layers: Vec<i32>,
    pub groups: usize,
    pub group_id: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct YoloCfg {
    pub classes: usize,
    pub mask: Vec<usize>,
    /// Flat `w0, h0, w1, h1, ...` in network pixels.
    pub anchors: Vec<f32>,
    pub scale_x_y: f32,
    pub nms_kind: NmsKind,
    pub beta_nms: f32,
    /// Training-only keys; ignored by the inference path.
    pub train: YoloTrainCfg,
}

/// Training-only `[yolo]` keys (`darknet_cfg.cpp:2437-2468`).
#[derive(Debug, Clone, PartialEq)]
pub struct YoloTrainCfg {
    pub jitter: f32,
    pub resize: f32,
    pub iou_loss: IouLoss,
    pub iou_normalizer: f32,
    pub obj_normalizer: f32,
    pub cls_normalizer: f32,
    pub ignore_thresh: f32,
    pub truth_thresh: f32,
    pub max_delta: f32,
}

impl Default for YoloTrainCfg {
    fn default() -> Self {
        YoloTrainCfg {
            jitter: 0.0,
            resize: 1.0,
            iou_loss: IouLoss::Mse,
            iou_normalizer: 0.75,
            obj_normalizer: 1.0,
            cls_normalizer: 1.0,
            ignore_thresh: 0.5,
            truth_thresh: 1.0,
            max_delta: f32::MAX,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayerCfg {
    Conv(ConvCfg),
    Maxpool(MaxpoolCfg),
    Route(RouteCfg),
    Upsample { stride: usize },
    Yolo(YoloCfg),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cfg {
    pub net: NetCfg,
    /// Training-only `[net]` keys ([`TrainCfg::default`] when absent).
    pub train: TrainCfg,
    pub layers: Vec<LayerCfg>,
}

struct Section {
    name: String,
    line: usize,
    kv: Vec<(String, String, usize)>,
}

impl Section {
    fn err(&self, line: usize, msg: impl Into<String>) -> Error {
        Error::Cfg { line, msg: msg.into() }
    }

    fn raw(&self, key: &str) -> Option<&(String, String, usize)> {
        self.kv.iter().find(|(k, _, _)| k == key)
    }

    fn int(&self, key: &str, default: i64) -> Result<i64> {
        match self.raw(key) {
            None => Ok(default),
            Some((_, v, l)) => v
                .parse::<i64>()
                .map_err(|_| self.err(*l, format!("`{key}` is not an integer: `{v}`"))),
        }
    }

    fn uint(&self, key: &str, default: usize) -> Result<usize> {
        let v = self.int(key, default as i64)?;
        usize::try_from(v).map_err(|_| self.err(self.line, format!("`{key}` must be >= 0")))
    }

    fn positive(&self, key: &str, default: usize) -> Result<usize> {
        let v = self.uint(key, default)?;
        if v == 0 {
            return Err(self.err(self.line, format!("`{key}` must be > 0")));
        }
        Ok(v)
    }

    fn float(&self, key: &str, default: f32) -> Result<f32> {
        match self.raw(key) {
            None => Ok(default),
            Some((_, v, l)) => v
                .parse::<f32>()
                .map_err(|_| self.err(*l, format!("`{key}` is not a number: `{v}`"))),
        }
    }

    fn str(&self, key: &str, default: &str) -> String {
        self.raw(key).map_or(default.to_string(), |(_, v, _)| v.clone())
    }

    fn int_list(&self, key: &str) -> Result<Vec<i64>> {
        match self.raw(key) {
            None => Ok(vec![]),
            Some((_, v, l)) => v
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| {
                    s.trim()
                        .parse::<i64>()
                        .map_err(|_| self.err(*l, format!("`{key}` has bad entry `{s}`")))
                })
                .collect(),
        }
    }

    fn float_list(&self, key: &str) -> Result<Vec<f32>> {
        match self.raw(key) {
            None => Ok(vec![]),
            Some((_, v, l)) => v
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| {
                    s.trim()
                        .parse::<f32>()
                        .map_err(|_| self.err(*l, format!("`{key}` has bad entry `{s}`")))
                })
                .collect(),
        }
    }

    /// Error if any of `keys` is present with a value other than `ok`.
    fn forbid(&self, keys: &[&str], ok: i64) -> Result<()> {
        for k in keys {
            if self.int(k, ok)? != ok {
                return Err(Error::Unsupported(format!(
                    "[{}] line {}: `{k}` != {ok} changes inference and is not supported",
                    self.name, self.line
                )));
            }
        }
        Ok(())
    }
}

fn sections(text: &str) -> Result<Vec<Section>> {
    let mut out: Vec<Section> = vec![];
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let t = raw.trim();
        if t.is_empty() || t.starts_with('#') || t.starts_with(';') {
            continue;
        }
        if let Some(rest) = t.strip_prefix('[') {
            let name = rest
                .strip_suffix(']')
                .ok_or(Error::Cfg { line, msg: "unterminated section header".into() })?;
            out.push(Section { name: name.trim().to_string(), line, kv: vec![] });
        } else {
            let sec = out
                .last_mut()
                .ok_or(Error::Cfg { line, msg: "key before first section".into() })?;
            let (k, v) = t
                .split_once('=')
                .ok_or(Error::Cfg { line, msg: format!("expected `key=value`, got `{t}`") })?;
            sec.kv.push((k.trim().to_string(), v.trim().to_string(), line));
        }
    }
    Ok(out)
}

fn parse_train(s: &Section) -> Result<TrainCfg> {
    // Augmentations we do not implement must fail loudly, never silently no-op.
    s.forbid(&["mosaic", "mixup", "cutmix"], 0)?;
    for k in ["adam", "burn_in_adam"] {
        if s.raw(k).is_some() {
            return Err(Error::Unsupported(format!("[net] line {}: `{k}` (optimizer must be SGD)", s.line)));
        }
    }
    let angle = s.float("angle", 0.0)?;
    if angle != 0.0 {
        return Err(Error::Unsupported(format!("[net] line {}: `angle` rotation is not implemented", s.line)));
    }
    let policy = match s.str("policy", "constant").as_str() {
        "constant" => LrPolicy::Constant,
        "steps" => {
            let steps: Vec<usize> = s.int_list("steps")?.into_iter().map(|v| v.max(0) as usize).collect();
            let scales = s.float_list("scales")?;
            if steps.is_empty() || steps.len() != scales.len() {
                return Err(s.err(s.line, "`policy=steps` needs matching `steps` and `scales` lists"));
            }
            if steps.windows(2).any(|w| w[0] >= w[1]) {
                return Err(s.err(s.line, "`steps` must be strictly increasing"));
            }
            LrPolicy::Steps { steps, scales }
        }
        other => return Err(Error::Unsupported(format!("[net] line {}: policy `{other}`", s.line))),
    };
    Ok(TrainCfg {
        batch: s.positive("batch", 1)?,
        subdivisions: s.positive("subdivisions", 1)?,
        momentum: s.float("momentum", 0.9)?,
        decay: s.float("decay", 0.0005)?,
        learning_rate: s.float("learning_rate", 0.001)?,
        burn_in: s.uint("burn_in", 0)?,
        max_batches: s.uint("max_batches", 0)?,
        power: s.float("power", 4.0)?,
        policy,
        angle,
        flip: s.int("flip", 0)? != 0,
        saturation: s.float("saturation", 1.0)?,
        exposure: s.float("exposure", 1.0)?,
        hue: s.float("hue", 0.0)?,
    })
}

fn parse_conv(s: &Section) -> Result<ConvCfg> {
    s.forbid(
        &[
            "binary", "xnor", "bin_output", "sway", "rotate", "stretch", "stretch_sway",
            "antialiasing", "cbn", "flipped", "assisted_excitation",
        ],
        0,
    )?;
    s.forbid(&["groups", "dilation"], 1)?;
    if s.raw("share_index").is_some() {
        return Err(Error::Unsupported(format!("[convolutional] line {}: share_index", s.line)));
    }
    let size = s.positive("size", 1)?;
    let stride = s.positive("stride", 1)?;
    for k in ["stride_x", "stride_y"] {
        if s.int(k, stride as i64)? != stride as i64 {
            return Err(Error::Unsupported(format!("[convolutional] line {}: {k} != stride", s.line)));
        }
    }
    let padding = if s.int("pad", 0)? != 0 { size / 2 } else { s.uint("padding", 0)? };
    let activation = match s.str("activation", "logistic").as_str() {
        "leaky" => Activation::Leaky,
        "linear" => Activation::Linear,
        "logistic" => Activation::Logistic,
        other => return Err(Error::Unsupported(format!("activation `{other}` (line {})", s.line))),
    };
    Ok(ConvCfg {
        filters: s.positive("filters", 1)?,
        size,
        stride,
        padding,
        batch_normalize: s.int("batch_normalize", 0)? != 0,
        activation,
    })
}

fn parse_yolo(s: &Section) -> Result<YoloCfg> {
    s.forbid(&["new_coords"], 0)?;
    s.forbid(&["coords"], 4)?;
    s.forbid(&["objectness_smooth", "focal_loss"], 0)?;
    if s.raw("classes_multipliers").is_some() {
        return Err(Error::Unsupported(format!("[yolo] line {}: classes_multipliers", s.line)));
    }
    let label_smooth_eps = s.float("label_smooth_eps", 0.0)?;
    if label_smooth_eps != 0.0 {
        return Err(Error::Unsupported(format!("[yolo] line {}: label_smooth_eps", s.line)));
    }
    let iou_loss = match s.str("iou_loss", "mse").as_str() {
        "mse" => IouLoss::Mse,
        "iou" => IouLoss::Iou,
        "giou" => IouLoss::Giou,
        "diou" => IouLoss::Diou,
        "ciou" => IouLoss::Ciou,
        other => return Err(Error::Unsupported(format!("iou_loss `{other}` (line {})", s.line))),
    };
    let classes = s.int("classes", 20)?;
    let num = s.positive("num", 1)?;
    let mask: Vec<usize> = if s.raw("mask").is_some() {
        s.int_list("mask")?.into_iter().map(|v| v as usize).collect()
    } else {
        (0..num).collect()
    };
    let anchors = s.float_list("anchors")?;
    if anchors.len() != 2 * num {
        return Err(s.err(s.line, format!("expected {} anchor values (num={num}), got {}", 2 * num, anchors.len())));
    }
    if mask.iter().any(|&m| m >= num) {
        return Err(s.err(s.line, "mask index >= num"));
    }
    let nms_kind = match s.str("nms_kind", "default").as_str() {
        "default" | "cornersnms" => NmsKind::Default,
        "greedynms" => NmsKind::Greedy,
        "diounms" => NmsKind::Diou,
        other => return Err(Error::Unsupported(format!("nms_kind `{other}` (line {})", s.line))),
    };
    Ok(YoloCfg {
        classes: usize::try_from(classes).map_err(|_| s.err(s.line, "classes < 0"))?,
        mask,
        anchors,
        scale_x_y: s.float("scale_x_y", 1.0)?,
        nms_kind,
        beta_nms: s.float("beta_nms", 0.6)?,
        train: YoloTrainCfg {
            jitter: s.float("jitter", 0.2)?,
            resize: s.float("resize", 1.0)?,
            iou_loss,
            iou_normalizer: s.float("iou_normalizer", 0.75)?,
            obj_normalizer: s.float("obj_normalizer", 1.0)?,
            cls_normalizer: s.float("cls_normalizer", 1.0)?,
            ignore_thresh: s.float("ignore_thresh", 0.5)?,
            truth_thresh: s.float("truth_thresh", 1.0)?,
            max_delta: s.float("max_delta", f32::MAX)?,
        },
    })
}

impl Cfg {
    pub fn parse(text: &str) -> Result<Cfg> {
        let secs = sections(text)?;
        let first = secs.first().ok_or(Error::Model("empty cfg".into()))?;
        if first.name != "net" && first.name != "network" {
            return Err(Error::Cfg { line: first.line, msg: "first section must be [net]".into() });
        }
        let net = NetCfg {
            width: first.positive("width", 0)?,
            height: first.positive("height", 0)?,
            channels: first.positive("channels", 3)?,
        };
        let train = parse_train(first)?;
        let mut layers = vec![];
        for s in &secs[1..] {
            layers.push(match s.name.as_str() {
                "convolutional" => LayerCfg::Conv(parse_conv(s)?),
                "maxpool" => {
                    s.forbid(&["maxpool_depth", "antialiasing", "maxpool_zero_nonmax"], 0)?;
                    let stride = s.positive("stride", 1)?;
                    for k in ["stride_x", "stride_y"] {
                        if s.int(k, stride as i64)? != stride as i64 {
                            return Err(Error::Unsupported(format!("[maxpool] line {}: {k} != stride", s.line)));
                        }
                    }
                    let size = s.positive("size", stride)?;
                    LayerCfg::Maxpool(MaxpoolCfg { size, stride, padding: s.uint("padding", size - 1)? })
                }
                "route" => {
                    let layers: Vec<i32> = s.int_list("layers")?.into_iter().map(|v| v as i32).collect();
                    if layers.is_empty() {
                        return Err(s.err(s.line, "route needs `layers`"));
                    }
                    let groups = s.positive("groups", 1)?;
                    let group_id = s.uint("group_id", 0)?;
                    if group_id >= groups {
                        return Err(s.err(s.line, "group_id >= groups"));
                    }
                    LayerCfg::Route(RouteCfg { layers, groups, group_id })
                }
                "upsample" => {
                    let stride = s.int("stride", 2)?;
                    if stride < 1 {
                        return Err(Error::Unsupported(format!("[upsample] line {}: stride < 1", s.line)));
                    }
                    LayerCfg::Upsample { stride: stride as usize }
                }
                "yolo" => LayerCfg::Yolo(parse_yolo(s)?),
                other => return Err(Error::Unsupported(format!("layer type [{other}] (line {})", s.line))),
            });
        }
        Ok(Cfg { net, train, layers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NET: &str = "[net]\nwidth=32\nheight=16\nchannels=3\nlearning_rate=0.1\n";

    #[test]
    fn defaults() {
        let c = Cfg::parse(&format!(
            "{NET}[convolutional]\nfilters=4\nsize=3\npad=1\nactivation=leaky\n[maxpool]\nsize=2\nstride=2\n\
             [convolutional]\nfilters=2\nsize=1\n[upsample]\n"
        ))
        .unwrap();
        let LayerCfg::Conv(a) = &c.layers[0] else { panic!() };
        assert_eq!((a.stride, a.padding, a.activation, a.batch_normalize), (1, 1, Activation::Leaky, false));
        assert_eq!(c.layers[1], LayerCfg::Maxpool(MaxpoolCfg { size: 2, stride: 2, padding: 1 }));
        let LayerCfg::Conv(b) = &c.layers[2] else { panic!() };
        // pad absent -> 0 padding; activation defaults to logistic.
        assert_eq!((b.padding, b.activation), (0, Activation::Logistic));
        assert_eq!(c.layers[3], LayerCfg::Upsample { stride: 2 });
    }

    #[test]
    fn pad_uses_size_half() {
        let c = Cfg::parse(&format!("{NET}[convolutional]\nfilters=1\nsize=5\npad=1\nstride=2\n")).unwrap();
        let LayerCfg::Conv(a) = &c.layers[0] else { panic!() };
        assert_eq!((a.padding, a.stride), (2, 2));
    }

    #[test]
    fn rejects_unsupported() {
        assert!(Cfg::parse(&format!("{NET}[shortcut]\nfrom=-3\n")).is_err());
        assert!(Cfg::parse(&format!("{NET}[convolutional]\nfilters=2\ngroups=2\n")).is_err());
        assert!(Cfg::parse(&format!("{NET}[convolutional]\nfilters=2\nactivation=mish\n")).is_err());
    }

    #[test]
    fn route_and_yolo() {
        let c = Cfg::parse(&format!(
            "{NET}[route]\nlayers = -1, 23\ngroups=2\ngroup_id=1\n[yolo]\nmask = 3,4,5\n\
             anchors=9, 9, 14, 12, 41, 41, 48, 47, 68, 65, 77, 74\nclasses=5\nnum=6\nscale_x_y = 1.05\nnms_kind=greedynms\n"
        ))
        .unwrap();
        assert_eq!(c.layers[0], LayerCfg::Route(RouteCfg { layers: vec![-1, 23], groups: 2, group_id: 1 }));
        let LayerCfg::Yolo(y) = &c.layers[1] else { panic!() };
        assert_eq!((y.mask.clone(), y.classes, y.nms_kind), (vec![3, 4, 5], 5, NmsKind::Greedy));
        assert_eq!(y.anchors.len(), 12);
        assert!((y.scale_x_y - 1.05).abs() < 1e-6);
    }
}

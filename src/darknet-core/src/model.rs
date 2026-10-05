//! Parsed model description and layer-shape inference.
//!
//! This is the Rust equivalent of Darknet's parser plus the metadata stored in
//! its `network` and `layer` structs.  It owns no runtime buffers or mutable
//! weights, allowing CPU and GPU executors to share one validated graph.

use std::fmt::{Display, Formatter};
use std::fs;
use std::path::{Path, PathBuf};

use crate::cfg::{ConfigFile, RawSection};
use crate::error::{Error, Result};
use darknet_cuda::F32ActivationKind;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TensorShape {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub batch: usize,
}

impl TensorShape {
    pub const fn new(width: usize, height: usize, channels: usize, batch: usize) -> Self {
        Self {
            width,
            height,
            channels,
            batch,
        }
    }

    pub fn element_count(self) -> usize {
        self.width * self.height * self.channels
    }
}

impl Display for TensorShape {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.width, self.height, self.channels)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NetConfig {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub batch: usize,
    pub time_steps: usize,
    pub subdivisions: usize,
    pub learning_rate: f32,
    pub learning_rate_min: f32,
    pub momentum: f32,
    pub decay: f32,
    pub max_batches: usize,
    pub burn_in: usize,
    pub power: f32,
    pub step: usize,
    pub scale: f32,
    pub gamma: f32,
    pub sgdr_cycle: usize,
    pub sgdr_mult: usize,
    pub policy: LrPolicy,
    pub steps: Vec<usize>,
    pub scales: Vec<f32>,
    pub equidistant_point: usize,
    pub badlabels_rejection_percentage: f32,
    pub num_sigmas_reject_badlabels: f32,
    /// `adam=1` (C++ `net.adam`); SGD with momentum otherwise.
    pub adam: bool,
    /// `B1`/`B2`/`eps` (only read when `adam=1`, like C++).
    pub adam_b1: f32,
    pub adam_b2: f32,
    pub adam_eps: f32,
    /// Rust extension `decoupled_decay=1`: AdamW instead of C++'s L2-in-the-moments.
    pub decoupled_decay: bool,
    /// `loss_scale` (C++ default 1): gradients are scaled by this and the
    /// update divides it back out.
    pub loss_scale: f32,
    /// `try_fix_nan=1` (C++ `net.try_fix_nan`): convolution outputs are fixed
    /// for NaN/Inf, output deltas are clamped to [-1, 1] and input deltas are
    /// reset for NaN/Inf.
    pub try_fix_nan: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LrPolicy {
    Constant,
    Steps,
    Poly,
    Step,
    Exp,
    Sig,
    Sgdr,
}

impl NetConfig {
    fn from_section(section: &RawSection) -> Result<Self> {
        // Parse graph dimensions and training policy together.  The effective
        // batch and recurrent time-step values must be fixed before layer
        // shapes are inferred.
        if section.name != "net" && section.name != "network" {
            return Err(Error::InvalidConfig(format!(
                "first section must be [net], found [{}] at line {}",
                section.name, section.line
            )));
        }

        let inputs = section.get_usize("inputs", 0)?;
        let width = section.get_usize("width", if inputs > 0 { 1 } else { 0 })?;
        let height = section.get_usize("height", if inputs > 0 { 1 } else { 0 })?;
        let channels = section.get_usize("channels", inputs)?;

        if width == 0 || height == 0 || channels == 0 {
            return Err(Error::InvalidConfig(format!(
                "[net] must define width, height, and channels (line {})",
                section.line
            )));
        }

        let policy = match section
            .get_str("policy", "constant")
            .to_ascii_lowercase()
            .as_str()
        {
            "constant" => LrPolicy::Constant,
            "steps" => LrPolicy::Steps,
            "step" => LrPolicy::Step,
            "poly" => LrPolicy::Poly,
            "exp" => LrPolicy::Exp,
            "sig" | "sigmoid" => LrPolicy::Sig,
            "sgdr" => LrPolicy::Sgdr,
            other => {
                return Err(Error::InvalidConfig(format!(
                    "unsupported learning rate policy {other} in [net]"
                )));
            }
        };
        let steps: Vec<usize> = section
            .get("steps")
            .map(|_| {
                section.get_isize_list("steps").map(|values| {
                    values
                        .into_iter()
                        .map(|value| value.max(0) as usize)
                        .collect()
                })
            })
            .transpose()?
            .unwrap_or_default();
        let scales = section
            .get("scales")
            .map(|_| section.get_f32_list("scales"))
            .transpose()?
            .unwrap_or_default();
        if matches!(policy, LrPolicy::Steps) && !steps.is_empty() && steps.len() != scales.len() {
            return Err(Error::InvalidConfig(format!(
                "[net] steps/scales length mismatch: {} vs {}",
                steps.len(),
                scales.len()
            )));
        }

        let subdivisions = section.get_usize("subdivisions", 1)?.max(1);
        let time_steps = section.get_usize("time_steps", 1)?.max(1);
        let batch = section.get_usize("batch", 1)? / subdivisions * time_steps;
        let max_batches = section.get_usize("max_batches", 0)?;
        let adam = section.get_usize("adam", 0)? != 0;

        Ok(Self {
            width,
            height,
            channels,
            batch: batch.max(time_steps),
            time_steps,
            subdivisions,
            learning_rate: section.get_f32("learning_rate", 0.001)?,
            learning_rate_min: section.get_f32("learning_rate_min", 0.00001)?,
            momentum: section.get_f32("momentum", 0.9)?,
            decay: section.get_f32("decay", 0.0005)?,
            max_batches,
            burn_in: section.get_usize("burn_in", 0)?,
            power: section.get_f32("power", 4.0)?,
            step: section.get_usize("step", 1)?.max(1),
            scale: section.get_f32("scale", 1.0)?,
            gamma: section.get_f32("gamma", 1.0)?,
            sgdr_cycle: section.get_usize("sgdr_cycle", max_batches)?.max(1),
            sgdr_mult: section.get_usize("sgdr_mult", 2)?.max(1),
            policy,
            steps,
            scales,
            equidistant_point: section.get_usize("equidistant_point", 0)?,
            badlabels_rejection_percentage: section
                .get_f32("badlabels_rejection_percentage", 0.0)?,
            num_sigmas_reject_badlabels: section.get_f32("num_sigmas_reject_badlabels", 0.0)?,
            adam,
            adam_b1: if adam { section.get_f32("b1", 0.9)? } else { 0.9 },
            adam_b2: if adam { section.get_f32("b2", 0.999)? } else { 0.999 },
            adam_eps: if adam { section.get_f32("eps", 0.000001)? } else { 0.000001 },
            decoupled_decay: section.get_usize("decoupled_decay", 0)? != 0,
            loss_scale: section.get_f32("loss_scale", 1.0)?,
            try_fix_nan: section.get_usize("try_fix_nan", 0)? != 0,
        })
    }

    pub fn input_shape(&self) -> TensorShape {
        TensorShape::new(self.width, self.height, self.channels, self.batch)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    Linear,
    Logistic,
    Loggy,
    Relu,
    Leaky,
    Mish,
    Swish,
    Tanh,
    Relu6,
    Ramp,
    Gelu,
    Elu,
    Selu,
    Relie,
    Plse,
    Hardtan,
    Lhtan,
    Stair,
    HardMish,
    Revleaky,
    Other(String),
}

impl Activation {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "linear" => Self::Linear,
            "logistic" => Self::Logistic,
            "loggy" => Self::Loggy,
            "relu" => Self::Relu,
            "leaky" => Self::Leaky,
            "mish" => Self::Mish,
            "swish" => Self::Swish,
            "tanh" => Self::Tanh,
            "relu6" => Self::Relu6,
            "ramp" => Self::Ramp,
            "gelu" => Self::Gelu,
            "elu" => Self::Elu,
            "selu" => Self::Selu,
            "relie" => Self::Relie,
            "plse" => Self::Plse,
            "hardtan" => Self::Hardtan,
            "lhtan" => Self::Lhtan,
            "stair" => Self::Stair,
            "hard_mish" => Self::HardMish,
            "revleaky" => Self::Revleaky,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn to_cuda_f32_kind(&self) -> Option<F32ActivationKind> {
        match self {
            Self::Linear => Some(F32ActivationKind::Linear),
            Self::Logistic => Some(F32ActivationKind::Logistic),
            Self::Loggy | Self::Elu | Self::Selu | Self::Relie | Self::Plse | Self::Hardtan | Self::Lhtan | Self::Stair | Self::HardMish | Self::Revleaky => None,
            Self::Relu => Some(F32ActivationKind::Relu),
            Self::Leaky => Some(F32ActivationKind::Leaky),
            Self::Mish => Some(F32ActivationKind::Mish),
            Self::Swish => Some(F32ActivationKind::Swish),
            Self::Tanh => Some(F32ActivationKind::Tanh),
            Self::Relu6 => Some(F32ActivationKind::Relu6),
            Self::Ramp => Some(F32ActivationKind::Ramp),
            Self::Gelu => Some(F32ActivationKind::Gelu),
            Self::Other(_) => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvolutionalSpec {
    pub filters: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub dilation: usize,
    pub padding: usize,
    pub groups: usize,
    pub share_index: Option<usize>,
    pub batch_normalize: bool,
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConnectedSpec {
    pub output: usize,
    pub batch_normalize: bool,
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeconvolutionalSpec {
    pub filters: usize,
    pub size: usize,
    pub stride: usize,
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegressionSpec {
    pub scale: f32,
}

/// `[segmentation]` loss: binary logistic cross-entropy (`classes == 1`) or
/// per-pixel softmax cross-entropy (`classes > 1`).  C++ derives this from the
/// class count and only lets `loss=bce|logistic` / `loss=ce|softmax` agree with
/// it, so the enum has no independent choice (`src/darknet-core/src/segmentation.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentationLoss {
    /// `classes == 1`: sigmoid output, `logistic_x_ent_cpu`.
    Bce,
    /// `classes > 1`: per-pixel softmax output, `softmax_x_ent_cpu`.
    SoftmaxCe,
}

impl SegmentationLoss {
    /// The cfg spellings C++ accepts for this loss (`darknet_enums.cpp`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "bce" | "logistic" => Some(Self::Bce),
            "ce" | "softmax" => Some(Self::SoftmaxCe),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Bce => "bce",
            Self::SoftmaxCe => "ce",
        }
    }

    /// C++ `parse_segmentation_section`: the requested loss must be the one the
    /// class count implies.
    pub fn for_classes(classes: usize) -> Self {
        if classes == 1 { Self::Bce } else { Self::SoftmaxCe }
    }
}

/// `[segmentation]` (Phase 1b of `tasks/rust/layers.md`), ported from
/// `v6-dev-seg:src-lib/darknet_cfg.cpp::parse_segmentation_section`.  The layer
/// has no weights and passes its shape through unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentationSpec {
    pub classes: usize,
    /// Always 1 in C++: `make_segmentation_layer` sets it and the parser reads
    /// no cfg key.  Kept so the softmax call matches the source.
    pub temperature: f32,
    /// Gradient clip; the C++ default is `FLT_MAX`, i.e. no clipping.
    pub max_delta: f32,
    pub delta_normalizer: f32,
    pub cls_normalizer: f32,
    pub loss: SegmentationLoss,
    pub lambda_ce: f32,
    pub ce_normalizer: f32,
    pub gate_enabled: bool,
    pub gate_t_correct: f32,
    pub gate_warmup_iters: usize,
    pub gate_ramp_iters: usize,
    pub gate_max_compensation: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JevCeSpec {
    pub groups: usize,
    pub candidates: usize,
    pub temperature: f32,
    pub normalize: bool,
    pub ignore_empty: bool,
    pub label_smoothing: f32,
    pub scale: f32,
    pub channel_major: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxPoolSpec {
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub padding: usize,
    pub out_channels: usize,
    /// C++ `zero_nonmax`: during backward, zero the input delta at every
    /// position that was not an argmax of the pooling window.
    pub zero_nonmax: bool,
    /// C++ `maxpool_depth=1`: pool over strided channel groups; w x h unchanged,
    /// channels = `out_channels`.
    pub depth: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RouteSpec {
    pub layers: Vec<isize>,
    pub groups: usize,
    pub group_id: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShortcutSpec {
    pub from: Vec<isize>,
    pub activation: Activation,
    pub weights_type: ShortcutWeightsType,
    pub weights_normalization: ShortcutWeightsNorm,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShortcutWeightsType {
    #[default]
    None,
    PerFeature,
    PerChannel,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShortcutWeightsNorm {
    #[default]
    None,
    Relu,
    Softmax,
}

impl ShortcutWeightsType {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "per_feature" => Ok(Self::PerFeature),
            "per_channel" => Ok(Self::PerChannel),
            other => Err(Error::InvalidConfig(format!("weights type \"{other}\" is not supported"))),
        }
    }
}

impl ShortcutWeightsNorm {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "relu" | "avg_relu" => Ok(Self::Relu),
            "softmax" => Ok(Self::Softmax),
            other => Err(Error::InvalidConfig(format!("weights normalization \"{other}\" is not supported"))),
        }
    }
}

impl ShortcutSpec {
    /// C++ `nweights`: `n+1` (per_feature) or `(n+1)*c` (per_channel), 0 without weights.
    pub fn weight_count(&self, channels: usize) -> usize {
        match self.weights_type {
            ShortcutWeightsType::None => 0,
            ShortcutWeightsType::PerFeature => self.from.len() + 1,
            ShortcutWeightsType::PerChannel => (self.from.len() + 1) * channels,
        }
    }

    /// Weight slots per input (C++ `layer_step`).
    pub fn weight_slots(&self, channels: usize) -> usize {
        match self.weights_type {
            ShortcutWeightsType::None => 0,
            ShortcutWeightsType::PerFeature => 1,
            ShortcutWeightsType::PerChannel => channels,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct UpsampleSpec {
    pub stride: isize,
    pub scale: f32,
    pub reverse: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct YoloSpec {
    pub classes: usize,
    pub num: usize,
    pub mask: Vec<usize>,
    pub anchors: Vec<f32>,
    pub scale_x_y: f32,
    pub new_coords: bool,
    pub cls_normalizer: f32,
    pub obj_normalizer: f32,
    pub iou_normalizer: f32,
    pub ignore_thresh: f32,
    pub truth_thresh: f32,
    pub iou_thresh: f32,
    pub iou_loss: YoloIouLoss,
    pub iou_thresh_kind: YoloIouLoss,
    pub focal_loss: bool,
    pub label_smooth_eps: f32,
    pub objectness_smooth: bool,
    pub max_delta: f32,
    /// C++ `delta_normalizer`: scales the head delta when it is added to the previous layer's
    /// delta in backward (GPU path, `backward_yolo_layer_gpu`); cost/loss are unaffected.
    pub delta_normalizer: f32,
    pub class_multipliers: Option<Vec<f32>>,
    pub class_map: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YoloIouLoss {
    Mse,
    Iou,
    Giou,
    Diou,
    Ciou,
}

impl YoloIouLoss {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "mse" => Ok(Self::Mse),
            "iou" => Ok(Self::Iou),
            "giou" => Ok(Self::Giou),
            "diou" => Ok(Self::Diou),
            "ciou" => Ok(Self::Ciou),
            other => Err(Error::InvalidConfig(format!(
                "unsupported YOLO iou_loss {other}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReorgSpec {
    pub stride: usize,
    pub reverse: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSliceSpec {
    /// Layer whose output is sliced (relative negative or absolute, like route).
    pub from: isize,
    pub start: usize,
    /// Exclusive.
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelShuffleSpec {
    pub groups: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScaleChannelsSpec {
    pub from: isize,
    pub scale_wh: bool,
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamSpec {
    pub from: isize,
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RnnSpec {
    pub hidden: usize,
    pub output: usize,
    pub steps: usize,
    pub batch_normalize: bool,
    pub activation: Activation,
    pub logistic: i64,
    pub shortcut: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LstmSpec {
    pub output: usize,
    pub steps: usize,
    pub batch_normalize: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrnnSpec {
    pub hidden: usize,
    pub output: usize,
    pub size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    pub groups: usize,
    pub steps: usize,
    pub batch_normalize: bool,
    pub activation: Activation,
    pub shortcut: bool,
}

/// Spatial data augmentation used by legacy classifier cfgs.
#[derive(Clone, Debug, PartialEq)]
pub struct CropSpec {
    pub width: usize,
    pub height: usize,
    pub flip: bool,
    pub angle: f32,
    pub saturation: f32,
    pub exposure: f32,
    pub shift: f32,
    pub noadjust: bool,
}

/// A standalone channel-wise batch-normalization layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchNormSpec;

/// A locally-connected convolution: each output spatial position owns weights.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalSpec {
    pub filters: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub padding: usize,
    pub activation: Activation,
}

/// Legacy YOLOv1 detection-loss configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionSpec {
    pub classes: usize,
    pub coords: usize,
    pub side: usize,
    pub num: usize,
    pub rescore: bool,
    pub softmax: bool,
    pub sqrt: bool,
    pub jitter: f32,
    pub object_scale: f32,
    pub noobject_scale: f32,
    pub class_scale: f32,
    pub coord_scale: f32,
}

/// Gated recurrent unit configuration.  Parameters are owned by the runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GruSpec {
    pub output: usize,
    pub steps: usize,
    pub batch_normalize: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionSpec {
    pub classes: usize,
    pub num: usize,
    pub coords: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
    Convolutional(ConvolutionalSpec),
    Deconvolutional(DeconvolutionalSpec),
    Connected(ConnectedSpec),
    MaxPool(MaxPoolSpec),
    LocalAvgPool(MaxPoolSpec),
    AvgPool,
    Route(RouteSpec),
    Shortcut(ShortcutSpec),
    Upsample(UpsampleSpec),
    Yolo(YoloSpec),
    GaussianYolo(YoloSpec),
    Region(RegionSpec),
    Reorg(ReorgSpec),
    ChannelSlice(ChannelSliceSpec),
    ChannelShuffle(ChannelShuffleSpec),
    Softmax { groups: usize },
    Cost { kind: String, scale: f32 },
    Logistic,
    Regression(RegressionSpec),
    JevCe(JevCeSpec),
    Segmentation(SegmentationSpec),
    Dropout {
        probability: Option<f32>,
        /// `dropblock=1` selects DropBlock instead of element-wise dropout.
        dropblock: Option<DropblockSpec>,
    },
    ScaleChannels(ScaleChannelsSpec),
    Sam(SamSpec),
    Contrastive { classes: usize, yolo_layer: isize },
    Rnn(RnnSpec),
    Gru(GruSpec),
    Lstm(LstmSpec),
    Crnn(CrnnSpec),
    Crop(CropSpec),
    BatchNorm(BatchNormSpec),
    Local(LocalSpec),
    Detection(DetectionSpec),
    Unsupported(String),
}

impl LayerKind {
    pub fn name(&self) -> &str {
        match self {
            Self::Convolutional(_) => "convolutional",
            Self::Deconvolutional(_) => "deconvolutional",
            Self::Connected(_) => "connected",
            Self::MaxPool(_) => "maxpool",
            Self::LocalAvgPool(_) => "local_avgpool",
            Self::AvgPool => "avgpool",
            Self::Route(_) => "route",
            Self::Shortcut(_) => "shortcut",
            Self::Upsample(_) => "upsample",
            Self::Yolo(_) => "yolo",
            Self::GaussianYolo(_) => "gaussian_yolo",
            Self::Region(_) => "region",
            Self::Reorg(_) => "reorg",
            Self::ChannelSlice(_) => "channel_slice",
            Self::ChannelShuffle(_) => "channel_shuffle",
            Self::Softmax { .. } => "softmax",
            Self::Cost { .. } => "cost",
            Self::Logistic => "logistic",
            Self::Regression(_) => "regression",
            Self::JevCe(_) => "jev_ce",
            Self::Segmentation(_) => "segmentation",
            Self::Dropout { .. } => "dropout",
            Self::ScaleChannels(_) => "scale_channels",
            Self::Sam(_) => "sam",
            Self::Contrastive { .. } => "contrastive",
            Self::Rnn(_) => "rnn",
            Self::Gru(_) => "gru",
            Self::Lstm(_) => "lstm",
            Self::Crnn(_) => "crnn",
            Self::Crop(_) => "crop",
            Self::BatchNorm(_) => "batchnorm",
            Self::Local(_) => "local",
            Self::Detection(_) => "detection",
            Self::Unsupported(name) => name.as_str(),
        }
    }

    /// The elementwise activation this layer applies, if it has one.
    pub fn activation(&self) -> Option<&Activation> {
        match self {
            Self::Convolutional(spec) => Some(&spec.activation),
            Self::Deconvolutional(spec) => Some(&spec.activation),
            Self::Connected(spec) => Some(&spec.activation),
            Self::Shortcut(spec) => Some(&spec.activation),
            Self::ScaleChannels(spec) => Some(&spec.activation),
            Self::Sam(spec) => Some(&spec.activation),
            Self::Local(spec) => Some(&spec.activation),
            Self::Rnn(spec) => Some(&spec.activation),
            Self::Crnn(spec) => Some(&spec.activation),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightsArray {
    Biases,
    Scales,
    RollingMean,
    RollingVariance,
    Weights,
}

impl WeightsArray {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Biases => "biases",
            Self::Scales => "scales",
            Self::RollingMean => "rolling_mean",
            Self::RollingVariance => "rolling_variance",
            Self::Weights => "weights",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightsEntry {
    pub array: WeightsArray,
    pub len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerWeights {
    pub entries: Vec<WeightsEntry>,
}

impl LayerWeights {
    pub fn parameter_count(&self) -> usize {
        self.entries.iter().map(|entry| entry.len).sum()
    }

    pub fn byte_len(&self) -> u64 {
        self.parameter_count() as u64 * 4
    }
}

/// DropBlock block size, parsed like C++ `[dropout] dropblock=1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DropblockSpec {
    /// `dropblock_size_rel`: block side as a fraction of the layer width/height
    /// (0 when the absolute size is used).
    pub size_rel: f32,
    /// `dropblock_size_abs`: block side in cells (default 7 when neither key is set).
    pub size_abs: usize,
}

/// Per-layer training keys that C++ reads for every layer section
/// (`darknet_cfg.cpp`) and `grad_centr` from the convolutional section.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerTrainOptions {
    /// `learning_rate`: per-layer multiplier of the global rate (convolutions).
    pub learning_rate_scale: f32,
    /// `dont_update=N`: the layer is only updated once `N < iteration`.
    pub dont_update: i64,
    /// `burnin_update=N`: no update while `N * burn_in > iteration`.
    pub burnin_update: i64,
    /// `train_only_bn=1`: the update is skipped for this layer and, like C++
    /// `set_train_only_bn`, for every layer before it.
    pub train_only_bn: bool,
    /// `clip=C`: after the update the convolution weights are clamped to [-C, C].
    pub clip: f32,
    /// `grad_centr=1`: with batch norm, subtract the per-filter mean from the
    /// weight updates before the optimizer step.
    pub grad_centr: bool,
}

impl Default for LayerTrainOptions {
    fn default() -> Self {
        Self {
            learning_rate_scale: 1.0,
            dont_update: 0,
            burnin_update: 0,
            train_only_bn: false,
            clip: 0.0,
            grad_centr: false,
        }
    }
}

impl LayerTrainOptions {
    fn from_section(section: &RawSection) -> Result<Self> {
        Ok(Self {
            learning_rate_scale: section.get_f32("learning_rate", 1.0)?,
            dont_update: section.get_i64("dont_update", 0)?,
            burnin_update: section.get_i64("burnin_update", 0)?,
            train_only_bn: section.get_usize("train_only_bn", 0)? != 0,
            clip: section.get_f32("clip", 0.0)?,
            grad_centr: section.get_usize("grad_centr", 0)? != 0,
        })
    }

    /// C++ `update_network_gpu` gating: whether a layer is stepped at
    /// `iteration_num` (completed iterations, `seen / (batch * subdivisions)`).
    pub fn updates_at(&self, iteration_num: i64, burn_in: usize) -> bool {
        if self.burnin_update != 0 && self.burnin_update * burn_in as i64 > iteration_num {
            return false;
        }
        !self.train_only_bn && self.dont_update < iteration_num
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub index: usize,
    pub line: usize,
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    pub kind: LayerKind,
    pub train: LayerTrainOptions,
}

impl Layer {
    pub fn weights(&self) -> Option<LayerWeights> {
        match &self.kind {
            LayerKind::Convolutional(spec) => {
                if spec.share_index.is_some() {
                    return None;
                }
                let filters = spec.filters;
                let kernel_area = spec.size * spec.size;
                let grouped_channels = self.input_shape.channels / spec.groups.max(1);
                let weights = filters * grouped_channels * kernel_area;

                let mut entries = vec![WeightsEntry {
                    array: WeightsArray::Biases,
                    len: filters,
                }];

                if spec.batch_normalize {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Scales,
                            len: filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingMean,
                            len: filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingVariance,
                            len: filters,
                        },
                    ]);
                }

                entries.push(WeightsEntry {
                    array: WeightsArray::Weights,
                    len: weights,
                });

                Some(LayerWeights { entries })
            }
            LayerKind::Connected(spec) => {
                let outputs = spec.output;
                let inputs = self.input_shape.element_count();
                let mut entries = vec![
                    WeightsEntry {
                        array: WeightsArray::Biases,
                        len: outputs,
                    },
                    WeightsEntry {
                        array: WeightsArray::Weights,
                        len: outputs * inputs,
                    },
                ];

                if spec.batch_normalize {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Scales,
                            len: outputs,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingMean,
                            len: outputs,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingVariance,
                            len: outputs,
                        },
                    ]);
                }

                Some(LayerWeights { entries })
            }
            LayerKind::Shortcut(spec) => {
                let len = spec.weight_count(self.input_shape.channels);
                (len > 0).then(|| LayerWeights {
                    entries: vec![WeightsEntry {
                        array: WeightsArray::Weights,
                        len,
                    }],
                })
            }
            LayerKind::Deconvolutional(spec) => Some(LayerWeights {
                entries: vec![
                    WeightsEntry {
                        array: WeightsArray::Biases,
                        len: spec.filters,
                    },
                    WeightsEntry {
                        array: WeightsArray::Weights,
                        len: self.input_shape.channels * spec.filters * spec.size * spec.size,
                    },
                ],
            }),
            LayerKind::BatchNorm(_) => Some(LayerWeights {
                entries: vec![
                    WeightsEntry {
                        array: WeightsArray::Biases,
                        len: self.input_shape.channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Scales,
                        len: self.input_shape.channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::RollingMean,
                        len: self.input_shape.channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::RollingVariance,
                        len: self.input_shape.channels,
                    },
                ],
            }),
            LayerKind::Gru(spec) => {
                let inputs = self.input_shape.element_count();
                let mut entries = Vec::new();
                for layer_inputs in [
                    inputs,
                    spec.output,
                    inputs,
                    spec.output,
                    inputs,
                    spec.output,
                ] {
                    append_connected_weight_entries(
                        &mut entries,
                        layer_inputs,
                        spec.output,
                        spec.batch_normalize,
                    );
                }
                Some(LayerWeights { entries })
            }
            LayerKind::Local(spec) => {
                let locations = self.output_shape.width * self.output_shape.height;
                Some(LayerWeights {
                    entries: vec![
                        WeightsEntry {
                            array: WeightsArray::Biases,
                            len: locations * spec.filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::Weights,
                            len: locations
                                * spec.filters
                                * self.input_shape.channels
                                * spec.size
                                * spec.size,
                        },
                    ],
                })
            }
            LayerKind::Rnn(spec) => {
                let inputs = self.input_shape.element_count();
                let mut entries = Vec::new();
                append_connected_weight_entries(
                    &mut entries,
                    inputs,
                    spec.hidden,
                    spec.batch_normalize,
                );
                append_connected_weight_entries(
                    &mut entries,
                    spec.hidden,
                    spec.hidden,
                    spec.batch_normalize,
                );
                append_connected_weight_entries(
                    &mut entries,
                    spec.hidden,
                    spec.output,
                    spec.batch_normalize,
                );
                Some(LayerWeights { entries })
            }
            LayerKind::Lstm(spec) => {
                let inputs = self.input_shape.element_count();
                let mut entries = Vec::new();
                for (layer_inputs, layer_outputs) in [
                    (spec.output, spec.output),
                    (spec.output, spec.output),
                    (spec.output, spec.output),
                    (spec.output, spec.output),
                    (inputs, spec.output),
                    (inputs, spec.output),
                    (inputs, spec.output),
                    (inputs, spec.output),
                ] {
                    append_connected_weight_entries(
                        &mut entries,
                        layer_inputs,
                        layer_outputs,
                        spec.batch_normalize,
                    );
                }
                Some(LayerWeights { entries })
            }
            LayerKind::Crnn(spec) => {
                let mut entries = Vec::new();
                append_convolutional_weight_entries(
                    &mut entries,
                    self.input_shape.channels,
                    spec.hidden,
                    spec.size,
                    spec.groups,
                    spec.batch_normalize,
                );
                append_convolutional_weight_entries(
                    &mut entries,
                    spec.hidden,
                    spec.hidden,
                    spec.size,
                    spec.groups,
                    spec.batch_normalize,
                );
                append_convolutional_weight_entries(
                    &mut entries,
                    spec.hidden,
                    spec.output,
                    spec.size,
                    spec.groups,
                    spec.batch_normalize,
                );
                Some(LayerWeights { entries })
            }
            _ => None,
        }
    }
}

fn append_connected_weight_entries(
    entries: &mut Vec<WeightsEntry>,
    inputs: usize,
    outputs: usize,
    batch_normalize: bool,
) {
    entries.push(WeightsEntry {
        array: WeightsArray::Biases,
        len: outputs,
    });
    entries.push(WeightsEntry {
        array: WeightsArray::Weights,
        len: outputs * inputs,
    });
    if batch_normalize {
        entries.extend([
            WeightsEntry {
                array: WeightsArray::Scales,
                len: outputs,
            },
            WeightsEntry {
                array: WeightsArray::RollingMean,
                len: outputs,
            },
            WeightsEntry {
                array: WeightsArray::RollingVariance,
                len: outputs,
            },
        ]);
    }
}

fn append_convolutional_weight_entries(
    entries: &mut Vec<WeightsEntry>,
    input_channels: usize,
    filters: usize,
    size: usize,
    groups: usize,
    batch_normalize: bool,
) {
    entries.push(WeightsEntry {
        array: WeightsArray::Biases,
        len: filters,
    });
    if batch_normalize {
        entries.extend([
            WeightsEntry {
                array: WeightsArray::Scales,
                len: filters,
            },
            WeightsEntry {
                array: WeightsArray::RollingMean,
                len: filters,
            },
            WeightsEntry {
                array: WeightsArray::RollingVariance,
                len: filters,
            },
        ]);
    }
    entries.push(WeightsEntry {
        array: WeightsArray::Weights,
        len: filters * (input_channels / groups.max(1)) * size * size,
    });
}

#[derive(Clone, Debug, PartialEq)]
pub struct Network {
    pub source: std::path::PathBuf,
    pub net: NetConfig,
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let cfg = ConfigFile::parse_file(path)?;
        Self::from_config(cfg)
    }

    pub fn from_config(cfg: ConfigFile) -> Result<Self> {
        // Shape inference runs in source order.  Route and shortcut layers
        // therefore use the same preceding-layer index space as C++ Darknet.
        let Some(net_section) = cfg.sections.first() else {
            return Err(Error::InvalidConfig(format!(
                "empty config file {}",
                cfg.path.display()
            )));
        };

        let net = NetConfig::from_section(net_section)?;
        let mut current = net.input_shape();
        let mut layers = Vec::new();
        let cfg_parent = cfg
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        for (index, section) in cfg.sections.iter().enumerate().skip(1) {
            let layer_index = index - 1;
            let input_shape = current;
            let kind = parse_layer_kind(
                section,
                layer_index,
                current,
                &layers,
                &cfg_parent,
                net.time_steps,
            )?;
            if let Some(Activation::Other(name)) = kind.activation() {
                let hint = if name.starts_with("normalize_channels") {
                    " (channel-normalising activations need the layer geometry and are not implemented)"
                } else if name == "eml" {
                    " (eml is only used by [eml_convolutional], which is not implemented)"
                } else {
                    ""
                };
                return Err(Error::InvalidConfig(format!(
                    "[{}] at line {}: unsupported activation \"{name}\"{hint}; C++ would silently use relu",
                    section.name, section.line
                )));
            }
            let output_shape = infer_output_shape(&kind, layer_index, input_shape, &layers)?;

            layers.push(Layer {
                index: layer_index,
                line: section.line,
                input_shape,
                output_shape,
                kind,
                train: LayerTrainOptions::from_section(section)?,
            });

            current = output_shape;
        }

        // C++ `set_train_only_bn`: the flag of a layer also applies to every
        // layer before it.
        let mut train_only_bn = false;
        for layer in layers.iter_mut().rev() {
            train_only_bn |= layer.train.train_only_bn;
            layer.train.train_only_bn = train_only_bn;
        }

        Ok(Self {
            source: cfg.path,
            net,
            layers,
        })
    }

    pub fn weighted_layers(&self) -> impl Iterator<Item = &Layer> {
        self.layers.iter().filter(|layer| layer.weights().is_some())
    }

    pub fn total_parameter_count(&self) -> usize {
        self.weighted_layers()
            .filter_map(Layer::weights)
            .map(|weights| weights.parameter_count())
            .sum()
    }

    pub fn with_batch(&self, batch: usize) -> Self {
        let mut network = self.clone();
        network.net.batch = batch;
        for layer in &mut network.layers {
            layer.input_shape.batch = batch;
            layer.output_shape.batch = batch;
        }
        network
    }

    /// Rebuild the graph for a new input resolution (multi-scale `random`).
    /// Layer shapes are re-inferred in order so route/shortcut references stay
    /// consistent; layer specs and weights are unchanged.  Connected layers
    /// whose weight layout depends on the input size are rejected.
    pub fn with_input_size(&self, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidConfig(
                "network input size must be non-zero".to_owned(),
            ));
        }
        if self
            .layers
            .iter()
            .any(|layer| matches!(layer.kind, LayerKind::Connected(_)))
        {
            return Err(Error::InvalidConfig(
                "multi-scale resizing is not supported for networks with connected layers"
                    .to_owned(),
            ));
        }
        let mut network = self.clone();
        network.net.width = width;
        network.net.height = height;
        for index in 0..network.layers.len() {
            let current = if index == 0 {
                network.net.input_shape()
            } else {
                network.layers[index - 1].output_shape
            };
            let kind = network.layers[index].kind.clone();
            let output = infer_output_shape(&kind, index, current, &network.layers[..index])?;
            network.layers[index].input_shape = current;
            network.layers[index].output_shape = output;
        }
        Ok(network)
    }
}

fn parse_layer_kind(
    section: &RawSection,
    layer_index: usize,
    current: TensorShape,
    built_layers: &[Layer],
    cfg_parent: &Path,
    time_steps: usize,
) -> Result<LayerKind> {
    let name = section.name.as_str();
    Ok(match name {
        "conv" | "convolutional" => {
            let stride = section.get_usize("stride", 1)?;
            let size = section.get_usize("size", 1)?;
            let pad = section.get_bool_int("pad", false)?;
            let padding = if pad {
                size / 2
            } else {
                section.get_usize("padding", 0)?
            };
            let filters = section.get_usize("filters", 1)?;
            let groups = section.get_usize("groups", 1)?;
            let share_index = section
                .get("share_index")
                .map(|value| {
                    let raw = value.parse::<isize>().map_err(|_| {
                        Error::InvalidConfig(format!(
                            "invalid share_index for convolutional layer #{layer_index}"
                        ))
                    })?;
                    let source_index = resolve_relative_layer(layer_index, raw)?;
                    let source_layer =
                        built_layers
                            .get(source_index)
                            .ok_or(Error::InvalidRoute {
                                layer_index,
                                reference: raw,
                            })?;
                    let LayerKind::Convolutional(source_spec) = &source_layer.kind else {
                        return Err(Error::InvalidConfig(format!(
                            "convolutional layer #{layer_index} share_index points to non-convolutional layer #{source_index}"
                        )));
                    };
                    if source_spec.filters != filters
                        || source_spec.size != size
                        || source_spec.groups != groups
                        || source_layer.input_shape.channels != current.channels
                    {
                        return Err(Error::InvalidConfig(format!(
                            "convolutional layer #{layer_index} share_index #{source_index} is incompatible with current weights layout"
                        )));
                    }
                    Ok(source_index)
                })
                .transpose()?;

            LayerKind::Convolutional(ConvolutionalSpec {
                filters,
                size,
                stride_x: section.get_usize("stride_x", stride)?,
                stride_y: section.get_usize("stride_y", stride)?,
                dilation: section.get_usize("dilation", 1)?,
                padding,
                groups,
                share_index,
                batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
                activation: Activation::parse(&section.get_str("activation", "logistic")),
            })
        }
        "conn" | "connected" => LayerKind::Connected(ConnectedSpec {
            output: section.get_usize("output", 1)?,
            batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
            activation: Activation::parse(&section.get_str("activation", "logistic")),
        }),
        "deconvolutional" => {
            if section.get_usize("pad", 0)? != 0 || section.get_usize("padding", 0)? != 0 {
                return Err(Error::InvalidConfig(format!(
                    "[deconvolutional] at line {} does not support pad/padding",
                    section.line
                )));
            }
            LayerKind::Deconvolutional(DeconvolutionalSpec {
                filters: section.get_usize("filters", 1)?,
                size: section.get_usize("size", 1)?,
                stride: section.get_usize("stride", 1)?,
                activation: Activation::parse(&section.get_str("activation", "logistic")),
            })
        }
        "max" | "maxpool" => {
            let stride = section.get_usize("stride", 1)?;
            let size = section.get_usize("size", stride)?;
            LayerKind::MaxPool(MaxPoolSpec {
                size,
                stride_x: section.get_usize("stride_x", stride)?,
                stride_y: section.get_usize("stride_y", stride)?,
                padding: section.get_usize("padding", size.saturating_sub(1))?,
                // C++ default is 1 and the value only matters with maxpool_depth=1.
                out_channels: section.get_usize("out_channels", 1)?,
                // C++ key; the pre-port Rust key `zero_nonmax` is still accepted.
                zero_nonmax: section.get_bool_int(
                    "maxpool_zero_nonmax",
                    section.get_bool_int("zero_nonmax", false)?,
                )?,
                depth: section.get_bool_int("maxpool_depth", false)?,
            })
        }
        "local_avg" | "local_avgpool" => {
            let stride = section.get_usize("stride", 1)?;
            let size = section.get_usize("size", stride)?;
            LayerKind::LocalAvgPool(MaxPoolSpec {
                size,
                stride_x: section.get_usize("stride_x", stride)?,
                stride_y: section.get_usize("stride_y", stride)?,
                padding: section.get_usize("padding", size.saturating_sub(1))?,
                out_channels: current.channels,
                zero_nonmax: false,
                depth: false,
            })
        }
        "avg" | "avgpool" => LayerKind::AvgPool,
        "crop" => LayerKind::Crop(CropSpec {
            width: section.get_usize("crop_width", current.width)?,
            height: section.get_usize("crop_height", current.height)?,
            flip: section.get_bool_int("flip", false)?,
            angle: section.get_f32("angle", 0.0)?,
            saturation: section.get_f32("saturation", 1.0)?,
            exposure: section.get_f32("exposure", 1.0)?,
            shift: section.get_f32("shift", 0.0)?,
            noadjust: section.get_bool_int("noadjust", false)?,
        }),
        "batchnorm" => LayerKind::BatchNorm(BatchNormSpec),
        "local" => {
            let stride = section.get_usize("stride", 1)?;
            let size = section.get_usize("size", 1)?;
            let padding = if section.get_bool_int("pad", false)? {
                size / 2
            } else {
                section.get_usize("padding", 0)?
            };
            LayerKind::Local(LocalSpec {
                filters: section.get_usize("filters", 1)?,
                size,
                stride_x: section.get_usize("stride_x", stride)?,
                stride_y: section.get_usize("stride_y", stride)?,
                padding,
                activation: Activation::parse(&section.get_str("activation", "logistic")),
            })
        }
        "detection" => LayerKind::Detection(DetectionSpec {
            classes: section.get_usize("classes", 20)?,
            coords: section.get_usize("coords", 4)?,
            side: section.get_usize("side", 7)?,
            num: section.get_usize("num", 2)?,
            rescore: section.get_bool_int("rescore", false)?,
            softmax: section.get_bool_int("softmax", false)?,
            sqrt: section.get_bool_int("sqrt", false)?,
            jitter: section.get_f32("jitter", 0.2)?,
            object_scale: section.get_f32("object_scale", 1.0)?,
            noobject_scale: section.get_f32("noobject_scale", 1.0)?,
            class_scale: section.get_f32("class_scale", 1.0)?,
            coord_scale: section.get_f32("coord_scale", 1.0)?,
        }),
        "route" => LayerKind::Route(RouteSpec {
            layers: section.get_isize_list("layers")?,
            groups: section.get_usize("groups", 1)?,
            group_id: section.get_usize("group_id", 0)?,
        }),
        "shortcut" => LayerKind::Shortcut(ShortcutSpec {
            from: section.get_isize_list("from")?,
            activation: Activation::parse(&section.get_str("activation", "linear")),
            weights_type: ShortcutWeightsType::parse(&section.get_str("weights_type", "none"))?,
            weights_normalization: ShortcutWeightsNorm::parse(
                &section.get_str("weights_normalization", "none"),
            )?,
        }),
        "channel_slice" => {
            let from = section
                .get("from")
                .and_then(|value| value.trim().parse::<isize>().ok())
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[channel_slice] at line {} requires an integer from=",
                        section.line
                    ))
                })?;
            if section.get_usize("axis", 1)? != 1 {
                return Err(Error::InvalidConfig(format!(
                    "[channel_slice] at line {}: only axis=1 is supported",
                    section.line
                )));
            }
            LayerKind::ChannelSlice(ChannelSliceSpec {
                from,
                start: require_usize(section, "start")?,
                end: require_usize(section, "end")?,
            })
        }
        "channel_shuffle" => LayerKind::ChannelShuffle(ChannelShuffleSpec {
            groups: section.get_usize("groups", 2)?,
        }),
        "upsample" => {
            let stride = section.get_isize("stride", 2)?;
            LayerKind::Upsample(UpsampleSpec {
                stride,
                scale: section.get_f32("scale", 1.0)?,
                reverse: stride < 0,
            })
        }
        "yolo" => {
            let mask = section.get_usize_list("mask")?;
            let classes = section.get_usize("classes", 20)?;
            let max_delta = section.get_f32("max_delta", f32::MAX)?;
            LayerKind::Yolo(YoloSpec {
                classes,
                num: mask.len().max(section.get_usize("num", 1)?),
                mask,
                anchors: section.get_f32_list("anchors")?,
                scale_x_y: section.get_f32("scale_x_y", 1.0)?,
                new_coords: section.get_i64("new_coords", 0)? != 0,
                cls_normalizer: section.get_f32("cls_normalizer", 1.0)?,
                obj_normalizer: section.get_f32("obj_normalizer", 1.0)?,
                iou_normalizer: section.get_f32("iou_normalizer", 0.75)?,
                ignore_thresh: section.get_f32("ignore_thresh", 0.5)?,
                truth_thresh: section.get_f32("truth_thresh", 1.0)?,
                iou_thresh: section.get_f32("iou_thresh", 1.0)?,
                iou_loss: YoloIouLoss::parse(&section.get_str("iou_loss", "mse"))?,
                iou_thresh_kind: YoloIouLoss::parse(&section.get_str("iou_thresh_kind", "iou"))?,
                focal_loss: section.get_bool_int("focal_loss", false)?,
                label_smooth_eps: section.get_f32("label_smooth_eps", 0.0)?,
                objectness_smooth: section.get_bool_int("objectness_smooth", false)?,
                max_delta,
                delta_normalizer: section.get_f32("delta_normalizer", 1.0)?,
                class_multipliers: parse_class_multipliers(section, classes, max_delta)?,
                class_map: parse_class_map(section, cfg_parent)?,
            })
        }
        "gaussian_yolo" => {
            let mask = section.get_usize_list("mask")?;
            let classes = section.get_usize("classes", 20)?;
            let max_delta = section.get_f32("max_delta", f32::MAX)?;
            LayerKind::GaussianYolo(YoloSpec {
                classes,
                num: mask.len().max(section.get_usize("num", 1)?),
                mask,
                anchors: section.get_f32_list("anchors")?,
                scale_x_y: section.get_f32("scale_x_y", 1.0)?,
                new_coords: section.get_i64("new_coords", 0)? != 0,
                cls_normalizer: section.get_f32("cls_normalizer", 1.0)?,
                obj_normalizer: section.get_f32("obj_normalizer", 1.0)?,
                iou_normalizer: section.get_f32("iou_normalizer", 0.75)?,
                ignore_thresh: section.get_f32("ignore_thresh", 0.5)?,
                truth_thresh: section.get_f32("truth_thresh", 1.0)?,
                iou_thresh: section.get_f32("iou_thresh", 1.0)?,
                iou_loss: YoloIouLoss::parse(&section.get_str("iou_loss", "mse"))?,
                iou_thresh_kind: YoloIouLoss::parse(&section.get_str("iou_thresh_kind", "iou"))?,
                focal_loss: section.get_bool_int("focal_loss", false)?,
                label_smooth_eps: section.get_f32("label_smooth_eps", 0.0)?,
                objectness_smooth: section.get_bool_int("objectness_smooth", false)?,
                max_delta,
                delta_normalizer: section.get_f32("delta_normalizer", 1.0)?,
                class_multipliers: parse_class_multipliers(section, classes, max_delta)?,
                class_map: parse_class_map(section, cfg_parent)?,
            })
        }
        "region" => LayerKind::Region(RegionSpec {
            classes: section.get_usize("classes", 20)?,
            num: section.get_usize("num", 1)?,
            coords: section.get_usize("coords", 4)?,
        }),
        "reorg" | "reorg3d" => LayerKind::Reorg(ReorgSpec {
            stride: section.get_usize("stride", 1)?,
            reverse: section.get_i64("reverse", 0)? != 0,
        }),
        "soft" | "softmax" => LayerKind::Softmax {
            groups: section.get_usize("groups", 1)?,
        },
        "cost" => LayerKind::Cost {
            kind: section.get_str("type", "sse"),
            scale: section.get_f32("scale", 1.0)?,
        },
        "logistic" => LayerKind::Logistic,
        "regression" => LayerKind::Regression(RegressionSpec {
            scale: section.get_f32("scale", 1.0)?,
        }),
        "jev_ce" => {
            let inputs = current.element_count();
            let groups = section.get_usize("groups", 81)?;
            let line = section.line;
            if groups == 0 {
                return Err(Error::InvalidConfig(format!("[jev_ce] at line {line}: groups must be > 0")));
            }
            let mut candidates = section.get_usize("candidates", 0)?;
            if candidates == 0 {
                if inputs % groups != 0 {
                    return Err(Error::InvalidConfig(format!(
                        "[jev_ce] at line {line}: cannot infer candidates, {inputs} outputs are not divisible by groups={groups}"
                    )));
                }
                candidates = inputs / groups;
            }
            if groups * candidates != inputs {
                return Err(Error::InvalidConfig(format!(
                    "[jev_ce] at line {line}: inputs ({inputs}) must equal groups*candidates ({groups}*{candidates})"
                )));
            }
            let label_smoothing =
                section.get_f32("label_smoothing", section.get_f32("smoothing", 0.0)?)?;
            if !(0.0..1.0).contains(&label_smoothing) {
                return Err(Error::InvalidConfig(format!("[jev_ce] at line {line}: label_smoothing must be in [0,1)")));
            }
            let layout = section.get_str("layout", "group");
            let channel_major = match layout.as_str() {
                "channel" | "channel_major" | "cm" | "chw" => true,
                "group" | "group_major" | "gm" => section.get_usize("channel_major", 0)? != 0,
                other => {
                    return Err(Error::InvalidConfig(format!("[jev_ce] at line {line}: unknown layout '{other}' (expected group or channel)")));
                }
            };
            LayerKind::JevCe(JevCeSpec {
                groups,
                candidates,
                temperature: section.get_f32("temperature", section.get_f32("temp", 1.0)?)?,
                normalize: section.get_usize("normalize", 1)? != 0,
                ignore_empty: section.get_usize("ignore_empty", 1)? != 0,
                label_smoothing,
                scale: section.get_f32("scale", 1.0)?,
                channel_major,
            })
        }
        // C++ accepts both spellings (`darknet_enums.cpp`: "segmenter" and
        // "segmentation" both map to ELayerType::SEGMENTATION).
        "segmentation" | "segmenter" => {
            let line = section.line;
            let classes = section.get_usize("classes", 2)?;
            if classes < 1 {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: classes={classes} must be >= 1"
                )));
            }
            // C++ only warns when the input channels do not match `classes`;
            // keep that (a config it accepts must not become an error here).
            let max_delta = section.get_f32("max_delta", f32::MAX)?;
            if !(max_delta > 0.0) {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: max_delta={max_delta} must be > 0"
                )));
            }
            let delta_normalizer = section.get_f32("delta_normalizer", 1.0)?;
            if !(delta_normalizer > 0.0) {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: delta_normalizer={delta_normalizer} must be > 0"
                )));
            }
            let cls_normalizer = section.get_f32("cls_normalizer", 1.0)?;
            if !(cls_normalizer > 0.0) {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: cls_normalizer={cls_normalizer} must be > 0"
                )));
            }
            let expected = SegmentationLoss::for_classes(classes);
            let loss_name = section.get_str("loss", expected.name()).trim().to_ascii_lowercase();
            let requested = SegmentationLoss::from_name(&loss_name).ok_or_else(|| {
                Error::InvalidConfig(format!("[segmentation] at line {line}: loss \"{loss_name}\" is not supported"))
            })?;
            if requested != expected {
                let wanted = if classes == 1 { "bce/logistic" } else { "ce/softmax" };
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: classes={classes} uses {wanted}, not loss={loss_name}"
                )));
            }
            let lambda_ce = section.get_f32("lambda_ce", 1.0)?;
            if !(lambda_ce > 0.0) {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: lambda_ce={lambda_ce} must be > 0"
                )));
            }
            let ce_normalizer =
                section.get_f32("ce_normalizer", section.get_f32("loss_normalizer_ce", 1.0)?)?;
            if !(ce_normalizer > 0.0) {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: ce_normalizer={ce_normalizer} must be > 0"
                )));
            }
            let gate_enabled = section.get_usize("gate_enabled", 0)? != 0;
            let gate_t_correct = section.get_f32("gate_t_correct", 0.9)?;
            if !(0.0..=1.0).contains(&gate_t_correct) {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: gate_t_correct={gate_t_correct} must be in [0, 1]"
                )));
            }
            let gate_warmup_iters = section.get_usize("gate_warmup_iters", 1000)?;
            let gate_ramp_iters = section.get_usize("gate_ramp_iters", 2000)?;
            let gate_max_compensation = section.get_f32("gate_max_compensation", 10.0)?;
            if gate_max_compensation < 1.0 {
                return Err(Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: gate_max_compensation={gate_max_compensation} must be >= 1"
                )));
            }
            LayerKind::Segmentation(SegmentationSpec {
                classes,
                temperature: 1.0,
                max_delta,
                delta_normalizer,
                cls_normalizer,
                loss: requested,
                lambda_ce,
                ce_normalizer,
                gate_enabled,
                gate_t_correct,
                gate_warmup_iters,
                gate_ramp_iters,
                gate_max_compensation,
            })
        }
        "dropout" => LayerKind::Dropout {
            probability: section
                .get("probability")
                .map(|value| value.parse().unwrap_or(0.5)),
            dropblock: if section.get_usize("dropblock", 0)? != 0 {
                // C++ make_dropout_layer / parse_dropout: the absolute size
                // is capped by the layer size, defaults to 7 when neither key
                // is set, and wins when both are.
                let mut size_rel = section.get_f32("dropblock_size_rel", 0.0)?;
                let mut size_abs = section.get_f32("dropblock_size_abs", 0.0)? as usize;
                size_abs = size_abs.min(current.width.min(current.height).max(1));
                if size_rel == 0.0 && size_abs == 0 {
                    size_abs = 7;
                }
                if size_rel != 0.0 && size_abs != 0 {
                    size_rel = 0.0;
                }
                Some(DropblockSpec { size_rel, size_abs })
            } else {
                None
            },
        },
        "scale_channels" => LayerKind::ScaleChannels(ScaleChannelsSpec {
            from: section.get_isize("from", 0)?,
            scale_wh: section.get_bool_int("scale_wh", false)?,
            activation: Activation::parse(&section.get_str("activation", "linear")),
        }),
        "sam" => LayerKind::Sam(SamSpec {
            from: section.get_isize("from", 0)?,
            activation: Activation::parse(&section.get_str("activation", "linear")),
        }),
        "contrastive" => LayerKind::Contrastive {
            classes: section.get_usize("classes", 1000)?,
            yolo_layer: section.get_isize("yolo_layer", 0)?,
        },
        "rnn" => LayerKind::Rnn(RnnSpec {
            hidden: section.get_usize("hidden", current.element_count())?,
            output: section.get_usize("output", current.element_count())?,
            steps: time_steps,
            batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
            activation: Activation::parse(&section.get_str("activation", "logistic")),
            logistic: section.get_i64("logistic", 0)?,
            shortcut: section.get_i64("shortcut", 0)? != 0,
        }),
        "gru" => LayerKind::Gru(GruSpec {
            output: section.get_usize("output", current.element_count())?,
            steps: time_steps,
            batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
        }),
        "lstm" => LayerKind::Lstm(LstmSpec {
            output: section.get_usize("output", current.element_count())?,
            steps: time_steps,
            batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
        }),
        "crnn" => {
            let size = section.get_usize("size", 3)?;
            let stride = section.get_usize("stride", 1)?;
            let padding = if section.get_i64("pad", 0)? != 0 {
                size / 2
            } else {
                section.get_usize("padding", 0)?
            };
            LayerKind::Crnn(CrnnSpec {
                hidden: section.get_usize("hidden", current.channels)?,
                output: section.get_usize("output", current.channels)?,
                size,
                stride,
                dilation: section.get_usize("dilation", 1)?,
                padding,
                groups: section.get_usize("groups", 1)?.max(1),
                steps: time_steps,
                batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
                activation: Activation::parse(&section.get_str("activation", "logistic")),
                shortcut: section.get_i64("shortcut", 0)? != 0,
            })
        }
        other => {
            if built_layers.is_empty() && layer_index == 0 && other == "net" {
                return Err(Error::InvalidConfig("duplicate [net] section".to_owned()));
            }
            LayerKind::Unsupported(other.to_owned())
        }
    })
}

fn infer_output_shape(
    kind: &LayerKind,
    layer_index: usize,
    current: TensorShape,
    built_layers: &[Layer],
) -> Result<TensorShape> {
    match kind {
        LayerKind::Convolutional(spec) => Ok(TensorShape::new(
            conv_dim(current.width, spec.padding, spec.size, spec.stride_x),
            conv_dim(current.height, spec.padding, spec.size, spec.stride_y),
            spec.filters,
            current.batch,
        )),
        LayerKind::Deconvolutional(spec) => {
            if spec.filters == 0 || spec.size == 0 || spec.stride == 0 || current.width == 0 || current.height == 0 {
                return Err(Error::InvalidConfig(format!(
                    "deconvolutional layer #{layer_index} needs filters/size/stride > 0 and a non-empty input"
                )));
            }
            Ok(TensorShape::new(
                spec.stride * (current.width - 1) + spec.size,
                spec.stride * (current.height - 1) + spec.size,
                spec.filters,
                current.batch,
            ))
        }
        LayerKind::Connected(spec) => Ok(TensorShape::new(1, 1, spec.output, current.batch)),
        LayerKind::Gru(spec) => Ok(TensorShape::new(1, 1, spec.output, current.batch)),
        LayerKind::Crop(spec) => Ok(TensorShape::new(
            spec.width,
            spec.height,
            current.channels,
            current.batch,
        )),
        LayerKind::BatchNorm(_) | LayerKind::Detection(_) => Ok(current),
        LayerKind::Local(spec) => Ok(TensorShape::new(
            conv_dim(current.width, spec.padding, spec.size, spec.stride_x),
            conv_dim(current.height, spec.padding, spec.size, spec.stride_y),
            spec.filters,
            current.batch,
        )),
        LayerKind::MaxPool(spec) | LayerKind::LocalAvgPool(spec) => {
            if spec.depth {
                if spec.out_channels == 0 || spec.out_channels > current.channels {
                    return Err(Error::InvalidConfig(format!(
                        "maxpool_depth layer #{layer_index}: out_channels={} must be in 1..={}",
                        spec.out_channels, current.channels
                    )));
                }
                return Ok(TensorShape::new(
                    current.width,
                    current.height,
                    spec.out_channels,
                    current.batch,
                ));
            }
            Ok(TensorShape::new(
                pool_dim(current.width, spec.padding, spec.size, spec.stride_x),
                pool_dim(current.height, spec.padding, spec.size, spec.stride_y),
                current.channels,
                current.batch,
            ))
        }
        LayerKind::AvgPool => Ok(TensorShape::new(1, 1, current.channels, current.batch)),
        LayerKind::Route(spec) => infer_route_shape(spec, layer_index, built_layers, current.batch),
        LayerKind::Shortcut(spec) => {
            for raw_index in &spec.from {
                let source = resolve_relative_layer(layer_index, *raw_index)?;
                let from_shape = built_layers
                    .get(source)
                    .ok_or(Error::InvalidRoute {
                        layer_index,
                        reference: *raw_index,
                    })?
                    .output_shape;
                if from_shape.batch != current.batch {
                    return Err(Error::InvalidConfig(format!(
                        "shortcut layer #{layer_index} mixes incompatible batch shapes {} and {}",
                        current, from_shape
                    )));
                }
            }
            Ok(current)
        }
        LayerKind::Upsample(spec) => {
            let stride = spec.stride.unsigned_abs();
            if stride == 0 {
                return Err(Error::InvalidConfig(format!(
                    "upsample layer #{layer_index} cannot use stride=0"
                )));
            }
            if spec.reverse {
                Ok(TensorShape::new(
                    current.width / stride,
                    current.height / stride,
                    current.channels,
                    current.batch,
                ))
            } else {
                Ok(TensorShape::new(
                    current.width * stride,
                    current.height * stride,
                    current.channels,
                    current.batch,
                ))
            }
        }
        LayerKind::Yolo(_) | LayerKind::GaussianYolo(_) | LayerKind::Region(_) => Ok(current),
        LayerKind::Reorg(spec) => {
            if spec.reverse {
                Ok(TensorShape::new(
                    current.width * spec.stride,
                    current.height * spec.stride,
                    current.channels / (spec.stride * spec.stride).max(1),
                    current.batch,
                ))
            } else {
                Ok(TensorShape::new(
                    current.width / spec.stride.max(1),
                    current.height / spec.stride.max(1),
                    current.channels * spec.stride * spec.stride,
                    current.batch,
                ))
            }
        }
        LayerKind::Softmax { .. }
        | LayerKind::Cost { .. }
        | LayerKind::Logistic
        | LayerKind::Regression(_)
        | LayerKind::JevCe(_)
        | LayerKind::Segmentation(_)
        | LayerKind::Dropout { .. }
        | LayerKind::Contrastive { .. } => Ok(current),
        LayerKind::ChannelSlice(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            let source = built_layers
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: spec.from,
                })?
                .output_shape;
            if spec.end <= spec.start || spec.end > source.channels {
                return Err(Error::InvalidConfig(format!(
                    "channel_slice layer #{layer_index}: range [{}, {}) is invalid for {} channels",
                    spec.start, spec.end, source.channels
                )));
            }
            Ok(TensorShape::new(
                source.width,
                source.height,
                spec.end - spec.start,
                source.batch,
            ))
        }
        LayerKind::ChannelShuffle(spec) => {
            if spec.groups == 0 || current.channels % spec.groups != 0 {
                return Err(Error::InvalidConfig(format!(
                    "channel_shuffle layer #{layer_index} requires channels={} divisible by groups={}",
                    current.channels, spec.groups
                )));
            }
            Ok(current)
        }
        LayerKind::ScaleChannels(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            let source = built_layers
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: spec.from,
                })?
                .output_shape;
            if spec.scale_wh {
                if current.channels != 1
                    || source.width != current.width
                    || source.height != current.height
                {
                    return Err(Error::InvalidConfig(format!(
                        "scale_channels layer #{layer_index} with scale_wh=1 requires current shape C=1 and source spatial shape {}, got current {} and source {}",
                        current, current, source
                    )));
                }
            } else if current.width != 1
                || current.height != 1
                || source.channels != current.channels
            {
                return Err(Error::InvalidConfig(format!(
                    "scale_channels layer #{layer_index} requires a 1x1 channel scale with matching channels, got current {} and source {}",
                    current, source
                )));
            }
            Ok(source)
        }
        LayerKind::Sam(spec) => {
            let source_index = resolve_relative_layer(layer_index, spec.from)?;
            let source = built_layers
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: spec.from,
                })?
                .output_shape;
            if source.width != current.width
                || source.height != current.height
                || source.channels != current.channels
            {
                return Err(Error::InvalidConfig(format!(
                    "sam layer #{layer_index} requires matching current/source shapes, got {} and {}",
                    current, source
                )));
            }
            Ok(current)
        }
        LayerKind::Rnn(RnnSpec { output, .. }) | LayerKind::Lstm(LstmSpec { output, .. }) => {
            Ok(TensorShape::new(1, 1, *output, current.batch))
        }
        LayerKind::Crnn(spec) => Ok(TensorShape::new(
            conv_dim(current.width, spec.padding, spec.size, spec.stride),
            conv_dim(current.height, spec.padding, spec.size, spec.stride),
            spec.output,
            current.batch,
        )),
        LayerKind::Unsupported(name) => Err(Error::UnsupportedLayer {
            layer_index,
            name: name.clone(),
        }),
    }
}

fn infer_route_shape(
    spec: &RouteSpec,
    layer_index: usize,
    built_layers: &[Layer],
    batch: usize,
) -> Result<TensorShape> {
    if spec.layers.is_empty() {
        return Err(Error::InvalidConfig(format!(
            "route layer #{layer_index} must list at least one source layer"
        )));
    }

    let first_index = resolve_relative_layer(layer_index, spec.layers[0])?;
    let first = built_layers
        .get(first_index)
        .ok_or(Error::InvalidRoute {
            layer_index,
            reference: spec.layers[0],
        })?
        .output_shape;

    let mut channels = 0usize;
    for raw_index in &spec.layers {
        let source = resolve_relative_layer(layer_index, *raw_index)?;
        let shape = built_layers
            .get(source)
            .ok_or(Error::InvalidRoute {
                layer_index,
                reference: *raw_index,
            })?
            .output_shape;

        if shape.width != first.width || shape.height != first.height {
            return Err(Error::InvalidConfig(format!(
                "route layer #{layer_index} mixes incompatible spatial shapes {} and {}",
                first, shape
            )));
        }

        channels += shape.channels;
    }

    let grouped_channels = channels / spec.groups.max(1);
    Ok(TensorShape::new(
        first.width,
        first.height,
        grouped_channels,
        batch,
    ))
}

fn require_usize(section: &RawSection, key: &str) -> Result<usize> {
    section
        .get(key)
        .ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[{}] at line {} requires {key}=",
                section.name, section.line
            ))
        })?
        .trim()
        .parse::<usize>()
        .map_err(|_| {
            Error::InvalidConfig(format!("[{}] at line {}: invalid {key}", section.name, section.line))
        })
}

fn resolve_relative_layer(layer_index: usize, raw_index: isize) -> Result<usize> {    let absolute = if raw_index < 0 {
        layer_index
            .checked_add_signed(raw_index)
            .ok_or(Error::InvalidRoute {
                layer_index,
                reference: raw_index,
            })?
    } else {
        usize::try_from(raw_index).map_err(|_| Error::InvalidRoute {
            layer_index,
            reference: raw_index,
        })?
    };

    if absolute >= layer_index {
        return Err(Error::InvalidRoute {
            layer_index,
            reference: raw_index,
        });
    }

    Ok(absolute)
}

fn parse_class_multipliers(
    section: &RawSection,
    classes: usize,
    max_delta: f32,
) -> Result<Option<Vec<f32>>> {
    let mut counters = section.get_usize_list("counters_per_class")?;
    if counters.is_empty() {
        return Ok(None);
    }
    if counters.len() != classes {
        return Err(Error::InvalidConfig(format!(
            "[{}] counters_per_class length {} does not match classes={classes}",
            section.name,
            counters.len()
        )));
    }

    let mut max_counter = 0usize;
    for counter in &mut counters {
        if *counter < 1 {
            *counter = 1;
        }
        max_counter = max_counter.max(*counter);
    }

    let mut multipliers = Vec::with_capacity(counters.len());
    for counter in counters {
        multipliers.push((max_counter as f32 / counter as f32).min(max_delta));
    }
    Ok(Some(multipliers))
}

fn parse_class_map(section: &RawSection, cfg_parent: &Path) -> Result<Option<Vec<usize>>> {
    let Some(raw_path) = section.get("map") else {
        return Ok(None);
    };
    let path = resolve_cfg_path(cfg_parent, raw_path);
    let text = fs::read_to_string(&path)?;
    let mut mapping = Vec::new();

    for (index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        let mapped = line.parse::<usize>().map_err(|_| {
            Error::Parse(format!(
                "failed to parse {} line {} as class map entry",
                path.display(),
                index + 1
            ))
        })?;
        mapping.push(mapped);
    }

    if mapping.is_empty() {
        return Err(Error::InvalidConfig(format!(
            "map file {} did not contain any class ids",
            path.display()
        )));
    }

    Ok(Some(mapping))
}

fn resolve_cfg_path(parent: &Path, raw: &str) -> PathBuf {
    let path = Path::new(raw);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        parent.join(path)
    }
}

fn conv_dim(input: usize, padding: usize, size: usize, stride: usize) -> usize {
    ((input + 2 * padding).saturating_sub(size) / stride.max(1)) + 1
}

fn pool_dim(input: usize, padding: usize, size: usize, stride: usize) -> usize {
    ((input + padding).saturating_sub(size) / stride.max(1)) + 1
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::cfg::ConfigFile;

    use super::{LayerKind, LrPolicy, Network, SegmentationLoss};

    fn build(text: &str) -> crate::error::Result<Network> {
        Network::from_config(ConfigFile::parse_str("t.cfg", text)?)
    }

    #[test]
    fn unknown_activation_is_a_config_error_not_identity() {
        let cfg = |act: &str| {
            format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={act}\n")
        };
        for ok in [
            "linear", "leaky", "mish", "swish", "gelu", "elu", "selu", "relie", "plse",
            "hardtan", "lhtan", "stair", "hard_mish", "revleaky",
        ] {
            assert!(build(&cfg(ok)).is_ok(), "{ok} must parse");
        }
        for bad in ["banana", "normalize_channels", "normalize_channels_softmax", "eml"] {
            let err = build(&cfg(bad)).expect_err(bad).to_string();
            assert!(err.contains(bad) && err.contains("line"), "{err}");
        }
        // the same check covers shortcut / connected / scale_channels / sam / rnn / crnn / local
        let shortcut = "[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\n[shortcut]\nfrom=-1\nactivation=banana\n";
        assert!(build(shortcut).is_err());
    }

    #[test]
    fn maxpool_depth_and_zero_nonmax_keys_follow_cpp() {
        let net = build(
            "[net]\nwidth=4\nheight=6\nchannels=4\n\
             [maxpool]\nsize=2\nstride=2\nmaxpool_depth=1\nout_channels=2\n\
             [maxpool]\nsize=2\nstride=2\nmaxpool_zero_nonmax=1\n",
        )
        .expect("builds");
        let LayerKind::MaxPool(depth) = &net.layers[0].kind else { panic!() };
        assert!(depth.depth);
        // depth pooling keeps w x h and sets the channels to out_channels
        let s = net.layers[0].output_shape;
        assert_eq!((s.width, s.height, s.channels), (4, 6, 2));
        let LayerKind::MaxPool(plain) = &net.layers[1].kind else { panic!() };
        assert!(plain.zero_nonmax && !plain.depth);
        let s = net.layers[1].output_shape;
        assert_eq!((s.width, s.height, s.channels), (2, 3, 2)); // out_channels ignored without depth
        // out_channels larger than the input channels is rejected
        assert!(build("[net]\nwidth=4\nheight=4\nchannels=2\n[maxpool]\nmaxpool_depth=1\nout_channels=3\n").is_err());
    }

    #[test]
    fn shortcut_weight_keys_follow_cpp_names_and_counts() {
        let mk = |extra: &str| {
            build(&format!(
                "[net]\nwidth=2\nheight=2\nchannels=3\n[convolutional]\nfilters=3\nsize=1\n[convolutional]\nfilters=3\nsize=1\n[shortcut]\nfrom=-2\n{extra}\n"
            ))
        };
        let n = mk("weights_type=per_feature\nweights_normalization=avg_relu").unwrap();
        assert_eq!(n.layers[2].weights().unwrap().parameter_count(), 2); // n+1
        let n = mk("weights_type=per_channel").unwrap();
        assert_eq!(n.layers[2].weights().unwrap().parameter_count(), 6); // (n+1)*c
        assert!(mk("").unwrap().layers[2].weights().is_none());
        assert!(mk("weights_type=banana").is_err());
        assert!(mk("weights_normalization=banana").is_err());
    }

    #[test]
    fn channel_slice_shape_comes_from_source_layer_and_feeds_route() {
        let net = build(
            "[net]\nwidth=4\nheight=3\nchannels=2\nbatch=2\n\
             [convolutional]\nfilters=6\nsize=1\n\
             [convolutional]\nfilters=5\nsize=1\n\
             [channel_slice]\nfrom=-2\nstart=1\nend=4\n\
             [route]\nlayers=-1,-2\n",
        )
        .unwrap();
        let s = net.layers[2].output_shape; // 3 channels of layer 0 (6 ch), not of the previous layer (5 ch)
        assert_eq!((s.width, s.height, s.channels, s.batch), (4, 3, 3, 2));
        assert_eq!(net.layers[3].output_shape.channels, 3 + 5);
        for bad in [
            "from=-2\nstart=2\nend=2",         // C++ default start=end=2: empty slice
            "from=-2\nstart=0\nend=7",         // beyond the source channels
            "from=-2\nstart=1",                // end missing
            "start=1\nend=2",                  // from missing
            "from=-2\nstart=0\nend=2\naxis=0", // only axis 1
            "from=5\nstart=0\nend=1",          // not a previous layer
        ] {
            let cfg = format!("[net]\nwidth=4\nheight=3\nchannels=2\n[convolutional]\nfilters=6\nsize=1\n[convolutional]\nfilters=5\nsize=1\n[channel_slice]\n{bad}\n");
            assert!(build(&cfg).is_err(), "{bad}");
        }
    }

    #[test]
    fn channel_shuffle_requires_divisible_channels() {
        let ok = build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=3\n").unwrap();
        assert_eq!(ok.layers[0].output_shape.channels, 6);
        assert!(build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=4\n").is_err());
        assert!(build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=0\n").is_err());
        // C++ default groups=2
        assert!(build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\n").is_ok());
    }

    #[test]
    fn deconvolutional_output_is_stride_times_h_minus_one_plus_size() {
        let net = build(
            "[net]\nwidth=3\nheight=2\nchannels=2\n\
             [deconvolutional]\nfilters=4\nsize=3\nstride=2\nactivation=leaky\n",
        )
        .unwrap();
        let s = net.layers[0].output_shape;
        assert_eq!((s.width, s.height, s.channels), (2 * 2 + 3, 2 * 1 + 3, 4));
        // weights(): biases[n] then weights[c*n*k*k] (C++ save_deconvolutional_weights order)
        let w = net.layers[0].weights().unwrap();
        assert_eq!(w.entries[0].len, 4);
        assert_eq!(w.entries[1].len, 2 * 4 * 9);
        // C++ default activation is logistic; pad/padding are rejected
        let net = build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\n").unwrap();
        assert!(matches!(&net.layers[0].kind, LayerKind::Deconvolutional(s) if s.activation == super::Activation::Logistic));
        assert!(build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\npad=1\n").is_err());
        assert!(build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\npadding=1\n").is_err());
    }

    #[test]
    fn loss_head_sections_parse_with_scale() {
        let net = build(
            "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\n\
             [cost]\ntype=masked\nscale=0.25\n[logistic]\n[regression]\nscale=3\n",
        )
        .unwrap();
        assert!(matches!(&net.layers[1].kind, LayerKind::Cost { kind, scale } if kind == "masked" && *scale == 0.25));
        assert!(matches!(net.layers[2].kind, LayerKind::Logistic));
        assert!(matches!(&net.layers[3].kind, LayerKind::Regression(r) if r.scale == 3.0));
        assert_eq!(net.layers[3].output_shape, net.layers[2].output_shape);
    }

    #[test]
    fn jev_ce_keys_follow_cpp_defaults_and_validation() {
        let base = "[net]\nwidth=1\nheight=1\nchannels=12\n[connected]\noutput=12\n";
        let net = build(&format!("{base}[jev_ce]\ngroups=4\ntemp=2\nsmoothing=0.1\nlayout=cm\n")).unwrap();
        let LayerKind::JevCe(s) = &net.layers[1].kind else { panic!() };
        assert_eq!((s.groups, s.candidates, s.temperature, s.label_smoothing, s.channel_major), (4, 3, 2.0, 0.1, true));
        assert!(s.normalize && s.ignore_empty && s.scale == 1.0);
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);
        assert!(build(&format!("{base}[jev_ce]\n")).is_err()); // default groups=81 does not divide 12
        assert!(build(&format!("{base}[jev_ce]\ngroups=5\n")).is_err());
        assert!(build(&format!("{base}[jev_ce]\ngroups=4\ncandidates=4\n")).is_err()); // 4*4 != 12
        assert!(build(&format!("{base}[jev_ce]\ngroups=4\nlabel_smoothing=1.0\n")).is_err());
        assert!(build(&format!("{base}[jev_ce]\ngroups=4\nlayout=zigzag\n")).is_err());
    }

    #[test]
    fn segmentation_keys_follow_cpp_defaults_and_validation() {
        // C++ defaults (parse_segmentation_section): classes=2, max_delta=FLT_MAX,
        // delta_normalizer=1, cls_normalizer=1, lambda_ce=1, ce_normalizer=1,
        // gate_enabled=0, gate_t_correct=0.9, gate_warmup_iters=1000,
        // gate_ramp_iters=2000, gate_max_compensation=10.
        let base = "[net]\nwidth=2\nheight=3\nchannels=5\n[convolutional]\nfilters=5\nsize=1\n";
        let net = build(&format!("{base}[segmentation]\n")).unwrap();
        let LayerKind::Segmentation(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.classes, 2);
        assert_eq!(s.loss, SegmentationLoss::SoftmaxCe);
        assert_eq!(s.max_delta, f32::MAX, "FLT_MAX means no clipping");
        assert_eq!((s.delta_normalizer, s.cls_normalizer, s.lambda_ce, s.ce_normalizer), (1.0, 1.0, 1.0, 1.0));
        assert!(!s.gate_enabled);
        assert_eq!((s.gate_t_correct, s.gate_warmup_iters, s.gate_ramp_iters, s.gate_max_compensation), (0.9, 1000, 2000, 10.0));
        assert_eq!(s.temperature, 1.0, "C++ never reads a temperature key");
        // The layer has no weights and passes its shape through unchanged.
        assert!(net.layers[1].weights().is_none());
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        // Every C++-accepted key round-trips.
        let net = build(&format!(
            "{base}[segmentation]\nclasses=3\nmax_delta=2.5\ndelta_normalizer=0.5\ncls_normalizer=2\n\
             lambda_ce=0.25\nce_normalizer=4\ngate_enabled=1\ngate_t_correct=0.5\ngate_warmup_iters=5\n\
             gate_ramp_iters=10\ngate_max_compensation=3\nloss=softmax\n"
        ))
        .unwrap();
        let LayerKind::Segmentation(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.classes, 3);
        assert_eq!((s.max_delta, s.delta_normalizer, s.cls_normalizer), (2.5, 0.5, 2.0));
        assert_eq!((s.lambda_ce, s.ce_normalizer), (0.25, 4.0));
        assert!(s.gate_enabled);
        assert_eq!((s.gate_t_correct, s.gate_warmup_iters, s.gate_ramp_iters, s.gate_max_compensation), (0.5, 5, 10, 3.0));
        // `loss_normalizer_ce` is the C++ fallback spelling of `ce_normalizer`.
        let net = build(&format!("{base}[segmentation]\nloss_normalizer_ce=7\n")).unwrap();
        let LayerKind::Segmentation(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.ce_normalizer, 7.0);
        // `segmenter` is the C++ alias of `segmentation`.
        let net = build(&format!("{base}[segmenter]\n")).unwrap();
        assert_eq!(net.layers[1].kind.name(), "segmentation");

        // classes=1 selects logistic BCE and rejects the softmax spelling.
        let net = build(&format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\n[segmentation]\nclasses=1\n")).unwrap();
        let LayerKind::Segmentation(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.loss, SegmentationLoss::Bce);
        let one = "[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\n[segmentation]\nclasses=1\n";
        assert!(build(&format!("{one}loss=ce\n")).is_err(), "classes=1 must not take loss=ce");
        assert!(build(&format!("{one}loss=softmax\n")).is_err());
        assert!(build(&format!("{one}loss=logistic\n")).is_ok());
        // classes=2 must not take the logistic spelling.
        let two = &base[..base.len()];
        assert!(build(&format!("{two}[segmentation]\nclasses=2\nloss=bce\n")).is_err());
        assert!(build(&format!("{two}[segmentation]\nclasses=2\nloss=hinge\n")).is_err());

        // C++ validation: every one of these is a fatal error.
        for bad in [
            "classes=0",
            "max_delta=0",
            "max_delta=-1",
            "delta_normalizer=0",
            "cls_normalizer=-0.5",
            "lambda_ce=0",
            "ce_normalizer=0",
            "loss_normalizer_ce=0",
            "gate_t_correct=-0.1",
            "gate_t_correct=1.5",
            "gate_max_compensation=0.5",
        ] {
            assert!(build(&format!("{base}[segmentation]\n{bad}\n")).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn segmentation_classes_one_is_a_single_channel_binary_head() {
        let net = build(
            "[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\n\
             [segmentation]\nclasses=1\n",
        )
        .unwrap();
        // Binary masks are `[batch, 1, H*W]`, so the pass-through shape keeps
        // exactly one channel: nothing to reshape.
        assert_eq!(net.layers[1].output_shape.channels, 1);
        assert_eq!(net.layers[1].output_shape.width * net.layers[1].output_shape.height, 4);
    }

    fn repo_cfg(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../cfg")
            .join(name)
    }

    #[test]
    fn parses_yolov3_tiny() {
        let network = Network::from_path(repo_cfg("yolov3-tiny.cfg")).expect("cfg should parse");
        assert!(!network.layers.is_empty());
        assert_eq!(network.net.width, 416);
        assert!(matches!(
            network.layers.last().map(|layer| &layer.kind),
            Some(LayerKind::Yolo(_))
        ));
    }

    #[test]
    fn with_input_size_reinfers_all_layer_shapes() {
        let network = Network::from_path(repo_cfg("yolov3-tiny.cfg")).expect("cfg should parse");
        let resized = network.with_input_size(320, 256).expect("resize");
        assert_eq!(resized.net.width, 320);
        assert_eq!(resized.net.height, 256);
        assert_eq!(resized.layers.len(), network.layers.len());
        let restored = resized
            .with_input_size(network.net.width, network.net.height)
            .expect("restore");
        for (a, b) in restored.layers.iter().zip(&network.layers) {
            assert_eq!(a.output_shape, b.output_shape);
        }
    }

    #[test]
    fn route_layers_resolve_expected_shape() {
        let network = Network::from_path(repo_cfg("yolov4-tiny.cfg")).expect("cfg should parse");
        let route = network
            .layers
            .iter()
            .find(|layer| matches!(layer.kind, LayerKind::Route(_)))
            .expect("expected at least one route layer");
        assert!(route.output_shape.channels > 0);
    }

    #[test]
    fn parses_net_training_hyperparameters() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "inline.cfg",
                r#"
                    [net]
                    width=32
                    height=32
                    channels=3
                    batch=8
                    subdivisions=2
                    learning_rate=0.01
                    momentum=0.8
                    decay=0.0001
                    burn_in=100
                    max_batches=900
                    policy=steps
                    steps=400,800
                    scales=0.1,0.1

                    [yolo]
                    mask=0
                    anchors=10,13
                    classes=1
                    num=1
                "#,
            )
            .expect("cfg parses"),
        )
        .expect("network builds");

        assert_eq!(network.net.batch, 4);
        assert_eq!(network.net.subdivisions, 2);
        assert_eq!(network.net.learning_rate, 0.01);
        assert_eq!(network.net.momentum, 0.8);
        assert_eq!(network.net.decay, 0.0001);
        assert_eq!(network.net.burn_in, 100);
        assert_eq!(network.net.max_batches, 900);
        assert_eq!(network.net.policy, LrPolicy::Steps);
        assert_eq!(network.net.steps, vec![400, 800]);
        assert_eq!(network.net.scales, vec![0.1, 0.1]);
        assert!(!network.net.adam && !network.net.decoupled_decay);
        assert_eq!(network.net.loss_scale, 1.0);
        assert_eq!(network.net.equidistant_point, 0);
        assert_eq!(network.net.badlabels_rejection_percentage, 0.0);
        assert_eq!(network.net.num_sigmas_reject_badlabels, 0.0);
    }

    #[test]
    fn parses_legacy_cfg_layer_families_with_their_output_shapes() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "legacy.cfg",
                r#"
                    [net]
                    width=8
                    height=6
                    channels=3
                    batch=2

                    [crop]
                    crop_width=4
                    crop_height=2
                    flip=1

                    [batchnorm]

                    [local]
                    filters=5
                    size=3
                    stride=1
                    pad=1

                    [detection]
                    classes=2
                    coords=4
                    side=7
                    num=2

                    [gru]
                    output=11
                "#,
            )
            .expect("cfg parses"),
        )
        .expect("legacy cfg constructs");

        assert!(matches!(network.layers[0].kind, LayerKind::Crop(_)));
        assert_eq!(network.layers[0].output_shape.width, 4);
        assert_eq!(network.layers[0].output_shape.height, 2);
        assert!(matches!(network.layers[1].kind, LayerKind::BatchNorm(_)));
        assert!(matches!(network.layers[2].kind, LayerKind::Local(_)));
        assert_eq!(network.layers[2].output_shape.channels, 5);
        assert!(matches!(network.layers[3].kind, LayerKind::Detection(_)));
        assert!(matches!(network.layers[4].kind, LayerKind::Gru(_)));
        assert_eq!(network.layers[4].output_shape.channels, 11);
        assert_eq!(network.layers[4].output_shape.width, 1);
        assert_eq!(network.layers[4].output_shape.height, 1);
    }

    #[test]
    fn parses_extended_yolo_training_options() {
        let dir = std::env::temp_dir().join("darknet-model-yolo-options");
        let _ = fs::create_dir_all(&dir);
        let map_path = dir.join("labels.map");
        let cfg_path = dir.join("sample.cfg");
        fs::write(&map_path, "1\n0\n").expect("map writes");
        fs::write(
            &cfg_path,
            r#"
                [net]
                width=32
                height=32
                channels=6
                max_batches=100
                equidistant_point=75
                badlabels_rejection_percentage=0.2
                num_sigmas_reject_badlabels=3

                [yolo]
                mask=0,1
                anchors=10,12,12,14
                classes=2
                num=2
                scale_x_y=1.05
                new_coords=1
                cls_normalizer=0.8
                obj_normalizer=1.1
                iou_normalizer=0.9
                ignore_thresh=0.4
                truth_thresh=0.7
                iou_thresh=0.5
                iou_loss=ciou
                iou_thresh_kind=giou
                focal_loss=1
                label_smooth_eps=0.05
                objectness_smooth=1
                max_delta=3.0
                counters_per_class=10,5
                map=labels.map
            "#,
        )
        .expect("cfg writes");

        let network = Network::from_path(&cfg_path).expect("cfg should parse");
        assert_eq!(network.net.equidistant_point, 75);
        assert_eq!(network.net.badlabels_rejection_percentage, 0.2);
        assert_eq!(network.net.num_sigmas_reject_badlabels, 3.0);
        let LayerKind::Yolo(spec) = &network.layers[0].kind else {
            panic!("expected yolo layer");
        };

        assert_eq!(spec.iou_normalizer, 0.9);
        assert_eq!(spec.iou_thresh, 0.5);
        assert_eq!(spec.iou_loss, super::YoloIouLoss::Ciou);
        assert_eq!(spec.iou_thresh_kind, super::YoloIouLoss::Giou);
        assert!(spec.focal_loss);
        assert!(spec.objectness_smooth);
        assert_eq!(spec.label_smooth_eps, 0.05);
        assert_eq!(spec.max_delta, 3.0);
        assert_eq!(spec.class_multipliers, Some(vec![1.0, 2.0]));
        assert_eq!(spec.class_map, Some(vec![1, 0]));
    }
}

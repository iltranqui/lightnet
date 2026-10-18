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
use darknet_cuda::{F32ActivationKind, F32ChannelNormalization};

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
    /// AdamW fallback `B1`/`B2`/`eps` (read for `adam=1` or `muon=1`).
    pub adam_b1: f32,
    pub adam_b2: f32,
    pub adam_eps: f32,
    /// Rust extension `decoupled_decay=1`: AdamW instead of C++'s L2-in-the-moments.
    pub decoupled_decay: bool,
    /// Rust CPU hybrid Muon/AdamW optimizer (`muon=1`).
    pub muon: bool,
    /// Additive Muon momentum coefficient (`muon_momentum`, default 0.95).
    pub muon_momentum: f32,
    /// `loss_scale` (C++ default 1): gradients are scaled by this and the
    /// update divides it back out.
    pub loss_scale: f32,
    /// `try_fix_nan=1` (C++ `net.try_fix_nan`): convolution outputs are fixed
    /// for NaN/Inf, output deltas are clamped to [-1, 1] and input deltas are
    /// reset for NaN/Inf.
    pub try_fix_nan: bool,
    /// C++ `parms.label_smooth_eps`: `[contrastive]` treats a class probability
    /// above this as a hard label.
    pub label_smooth_eps: f32,
    /// C++ `parms.adversarial`: `[contrastive]` then labels images `b % 2`
    /// instead of `b / 2`.
    pub adversarial: bool,
    /// C++ `[net] obb=1`: opts the graph into oriented-box detection heads.
    pub obb: bool,
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
        let muon_value = section.get_usize("muon", 0)?;
        if muon_value > 1 {
            return Err(Error::InvalidConfig(format!(
                "invalid muon={muon_value} in [net] on line {}; expected 0 or 1",
                section.line
            )));
        }
        let muon = muon_value != 0;
        if muon && adam {
            return Err(Error::InvalidConfig(
                "[net] muon=1 conflicts with adam=1; Muon uses its AdamW fallback automatically"
                    .to_owned(),
            ));
        }
        let decoupled_decay_value = section.get_usize("decoupled_decay", 0)?;
        if muon && decoupled_decay_value > 1 {
            return Err(Error::InvalidConfig(format!(
                "invalid decoupled_decay={decoupled_decay_value} with muon=1 in [net]; expected 1"
            )));
        }
        if muon && section.get("decoupled_decay").is_some() && decoupled_decay_value == 0 {
            return Err(Error::InvalidConfig(
                "[net] muon=1 requires AdamW; explicit decoupled_decay=0 conflicts".to_owned(),
            ));
        }
        let muon_momentum = match section.get("muon_momentum") {
            Some(_) if !muon => {
                return Err(Error::InvalidConfig(
                    "[net] muon_momentum requires muon=1".to_owned(),
                ));
            }
            Some(_) => section.get_f32("muon_momentum", 0.95)?,
            None => 0.95,
        };
        if muon && (!muon_momentum.is_finite() || !(0.0..1.0).contains(&muon_momentum)) {
            return Err(Error::InvalidConfig(
                "[net] muon_momentum must be finite and in [0, 1)".to_owned(),
            ));
        }
        let adam_b1 = if adam || muon {
            section.get_f32("b1", 0.9)?
        } else {
            0.9
        };
        let adam_b2 = if adam || muon {
            section.get_f32("b2", 0.999)?
        } else {
            0.999
        };
        let adam_eps = if adam || muon {
            section.get_f32("eps", 0.000001)?
        } else {
            0.000001
        };
        let decay = section.get_f32("decay", 0.0005)?;
        if muon
            && (!adam_b1.is_finite()
                || !(0.0..1.0).contains(&adam_b1)
                || !adam_b2.is_finite()
                || !(0.0..1.0).contains(&adam_b2)
                || !adam_eps.is_finite()
                || adam_eps <= 0.0
                || !decay.is_finite()
                || decay < 0.0)
        {
            return Err(Error::InvalidConfig(
                "[net] Muon requires finite B1/B2 in [0, 1), positive eps, and non-negative finite decay"
                    .to_owned(),
            ));
        }
        let obb = section.get_i64("obb", 0)?;
        if !(0..=1).contains(&obb) {
            return Err(Error::InvalidConfig(format!(
                "invalid obb={obb} in [net] on line {}; expected 0 or 1",
                section.line
            )));
        }

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
            decay,
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
            adam_b1,
            adam_b2,
            adam_eps,
            decoupled_decay: muon || decoupled_decay_value != 0,
            muon,
            muon_momentum,
            loss_scale: section.get_f32("loss_scale", 1.0)?,
            try_fix_nan: section.get_usize("try_fix_nan", 0)? != 0,
            label_smooth_eps: section.get_f32("label_smooth_eps", 0.0)?,
            adversarial: section.get_usize("adversarial", 0)? != 0,
            obb: obb != 0,
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
    NormalizeChannels,
    NormalizeChannelsSoftmax,
    NormalizeChannelsSoftmaxMaxval,
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
            "normalize_channels" => Self::NormalizeChannels,
            "normalize_channels_softmax" => Self::NormalizeChannelsSoftmax,
            "normalize_channels_softmax_maxval" => Self::NormalizeChannelsSoftmaxMaxval,
            "revleaky" => Self::Revleaky,
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn to_cuda_f32_kind(&self) -> Option<F32ActivationKind> {
        match self {
            Self::Linear => Some(F32ActivationKind::Linear),
            Self::Loggy => Some(F32ActivationKind::Loggy),
            Self::Hardtan => Some(F32ActivationKind::Hardtan),
            Self::Relie => Some(F32ActivationKind::Relie),
            Self::Lhtan => Some(F32ActivationKind::Lhtan),
            Self::Plse => Some(F32ActivationKind::Plse),
            Self::Stair => Some(F32ActivationKind::Stair),
            Self::Logistic => Some(F32ActivationKind::Logistic),
            Self::Revleaky => None,
            Self::Relu => Some(F32ActivationKind::Relu),
            Self::Leaky => Some(F32ActivationKind::Leaky),
            Self::Mish => Some(F32ActivationKind::Mish),
            Self::Swish => Some(F32ActivationKind::Swish),
            Self::Tanh => Some(F32ActivationKind::Tanh),
            Self::Relu6 => Some(F32ActivationKind::Relu6),
            Self::Ramp => Some(F32ActivationKind::Ramp),
            Self::Gelu => Some(F32ActivationKind::Gelu),
            Self::Elu => Some(F32ActivationKind::Elu),
            Self::Selu => Some(F32ActivationKind::Selu),
            Self::HardMish => Some(F32ActivationKind::HardMish),
            Self::NormalizeChannels
            | Self::NormalizeChannelsSoftmax
            | Self::NormalizeChannelsSoftmaxMaxval
            | Self::Other(_) => None,
        }
    }

    pub fn to_channel_normalization(&self) -> Option<F32ChannelNormalization> {
        match self {
            Self::NormalizeChannels => Some(F32ChannelNormalization::Relu),
            Self::NormalizeChannelsSoftmax => Some(F32ChannelNormalization::Softmax),
            Self::NormalizeChannelsSoftmaxMaxval => Some(F32ChannelNormalization::SoftmaxMaxval),
            _ => None,
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

/// `[moe_convolutional]` (Phase 2B of `tasks/rust/layers.md`), ported from
/// `Jev_laya:src-lib/moe_convolutional_layer.cpp`.
///
/// The layer owns `experts` ordinary convolutional layers - `expert` carries each
/// one's geometry - plus a tiny router that assigns each image a soft gate per
/// expert.  `top_k == 0` is the soft mixture, `top_k > 0` masks every expert but
/// the k strongest.  `load_balance` weights the auxiliary penalty
/// `max(0, experts * sum_e mean_gate[e]^2 - 1)`; `temperature` divides the router
/// logits before softmax, so larger values give flatter gates.
///
/// `expert_dropout` and `debug_interval` are parsed for compatibility with the
/// cfg keys of the same name; C++ reserves the former and prints gate telemetry
/// for the latter, and neither changes the output.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeSpec {
    pub experts: usize,
    pub top_k: usize,
    pub load_balance: f32,
    pub temperature: f32,
    pub expert_dropout: f32,
    pub debug_interval: usize,
    /// The shared convolution geometry of every expert.
    pub expert: ConvolutionalSpec,
}

/// `[eml_convolutional]`: two parallel linear convolutions combined by the
/// source-derived EML branch math. Both branches share this convolution spec.
#[derive(Clone, Debug, PartialEq)]
pub struct EmlConvolutionalSpec {
    pub branch: ConvolutionalSpec,
    pub eml_clamp: f32,
    pub eml_eps: f32,
    pub eml_scale: f32,
    pub residual: bool,
}

/// `[graph_conv]` grouped graph aggregation/projection, sourced from `Jev_laya`.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphConvSpec {
    pub filters: usize,
    pub groups: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub dilation: usize,
    pub padding: usize,
    /// 0 = uniform valid-neighbor mean, 1 = learned-edge softmax.
    pub edge_mode: usize,
    pub use_self: bool,
    pub batch_normalize: bool,
    pub activation: Activation,
}

/// `[dcnv4]` deformable spatial aggregation, sourced from `Jev_laya`.
#[derive(Clone, Debug, PartialEq)]
pub struct Dcnv4Spec {
    pub filters: usize,
    pub groups: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub dilation: usize,
    pub padding: usize,
    pub batch_normalize: bool,
    pub offset_scale: f32,
    pub remove_center: bool,
    pub d_stride: usize,
    pub block_thread: usize,
    pub softmax: bool,
    pub activation: Activation,
}

/// `[deform_conv]` / `[deformable_convolutional]` (DCNv1 and optional DCNv2 mask).
#[derive(Clone, Debug, PartialEq)]
pub struct DeformConvSpec {
    pub filters: usize,
    pub groups: usize,
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub dilation: usize,
    pub padding: usize,
    pub batch_normalize: bool,
    pub use_mask: bool,
    pub activation: Activation,
}

/// `[dsconv]` / `[dynamic_snake_conv]`, the single-template Dynamic Snake Conv.
#[derive(Clone, Debug, PartialEq)]
pub struct DsConvSpec {
    pub filters: usize,
    pub size: usize,
    pub padding: usize,
    pub extend_scope: f32,
    /// 0 = x-snake, 1 = y-snake, 2 = concatenate both axes.
    pub morph: usize,
    pub templates: usize,
    pub drop_prob: f32,
    pub activation: Activation,
}

/// `[wmhf]`: identity/local/global channel partition for the wavelet-Mamba
/// high-frequency fusion layer.
#[derive(Clone, Debug, PartialEq)]
pub struct WmhfSpec {
    pub filters: usize,
    pub identity_channels: usize,
    pub local_channels: usize,
    pub global_channels: usize,
    pub freq_scale: f32,
    pub shortcut: bool,
    pub batch_normalize: bool,
    pub activation: Activation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliffordContextMode {
    Abs,
    Diff,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliffordInteractionMode {
    Inner,
    Wedge,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliffordGffnMode {
    Local,
    Global,
    Hybrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliffordHigherMode {
    Off,
    VectorBivector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CliffordGlobalOp {
    AvgPool,
    Fourier,
}

/// Source-derived `[clifford]` geometric feature mixer.
#[derive(Clone, Debug, PartialEq)]
pub struct CliffordSpec {
    pub filters: usize,
    /// Normalized modulo `filters`; order and duplicates are significant.
    pub wedge_shifts: Vec<usize>,
    /// Normalized modulo `filters`; equal schedules share the wedge schedule.
    pub inner_shifts: Vec<usize>,
    pub context_mode: CliffordContextMode,
    pub interaction_mode: CliffordInteractionMode,
    pub gffn_mode: CliffordGffnMode,
    pub higher_mode: CliffordHigherMode,
    pub global_op: CliffordGlobalOp,
    pub fno_modes_h: usize,
    pub fno_modes_w: usize,
    pub dwconv_size: usize,
    pub num_dwconv: usize,
    pub drop_path: f32,
    pub layerscale_init: f32,
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
        if classes == 1 {
            Self::Bce
        } else {
            Self::SoftmaxCe
        }
    }
}

/// `[laya_scorer]`: the LAYA decision-model scorer (Phase 1b, unblocked by
/// Open Question 2), ported from
/// `Jev_laya:src-lib/darknet_cfg.cpp::parse_laya_scorer_section`.  The layer
/// needs `hidden * max_options` frozen weights; see
/// `src/darknet-core/src/laya_scorer.rs` for the maths and the blob order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaScorerSpec {
    /// Width of each option-slot vector; must equal the input channels.
    pub hidden: usize,
    /// Number of option slots; must equal the input width.
    pub max_options: usize,
    /// LayerNorm epsilon; `<= 0` falls back to `LAYA_LAYERNORM_EPS` (1e-5).
    pub ln_eps: f32,
}

/// `[laya_marker_gather]`: the LAYA per-option marker gather (Phase 1b),
/// ported from
/// `Jev_laya:src-lib/darknet_cfg.cpp::parse_laya_marker_gather_section`.  Two
/// route-style inputs (`layers=idxA,idxB`), no weights; see
/// `src/darknet-core/src/laya_marker_gather.rs` for the maths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayaMarkerGatherSpec {
    /// Exactly two source layers: `[0]` hidden-state (input A), `[1]` marker
    /// info (input B).  Raw cfg indices; negative values resolve relative to
    /// this layer at shape-inference time.
    pub layers: [isize; 2],
    /// Must equal input A's channel count.
    pub hidden: usize,
    /// Number of option slots; input B needs `>= 2 * max_options` channels.
    pub max_options: usize,
}

/// `[laya_act_head]`: the LAYA act/escalate head (Phase 1b), ported from
/// `Jev_laya:src-lib/darknet_cfg.cpp::parse_laya_act_head_section`.  Three
/// route-style inputs (`layers=head,scorer,marker`); a small frozen MLP over
/// masked-softmax summary features.  See
/// `src/darknet-core/src/laya_act_head.rs` for the maths and blob order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayaActHeadSpec {
    /// Exactly three source layers: `[head, scorer, marker]`.
    pub layers: [isize; 3],
    /// Must equal the head source's channel count.
    pub hidden: usize,
    /// Number of option slots (`>= 2`); must equal the scorer's width.
    pub max_options: usize,
    /// Hidden width of the act head's single hidden layer.
    pub fc_dim: usize,
    /// Number of act logits produced.
    pub n_act: usize,
}

/// `[laya_loss]`: the LAYA masked soft-target loss over the scorer's K option
/// logits (Phase 1b), ported from
/// `Jev_laya:src-lib/darknet_cfg.cpp::parse_laya_loss_section`.  See
/// `src/darknet-core/src/laya_loss.rs` for the maths.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaLossSpec {
    /// Number of option logits (must equal the input's per-sample size).
    pub max_options: usize,
    /// `mode=rlcd` (true) vs `mode=ce` (false).
    pub rlcd: bool,
    /// RLCD sample group size `G`.
    pub group: usize,
    pub sigma_start: f32,
    pub sigma_end: f32,
    pub w_sph: f32,
    pub w_rps: f32,
    pub w_ce: f32,
    pub log_floor: f32,
    pub seed: u64,
    /// `>= 0` enables TD(lambda) episode targets.
    pub td_lambda: f32,
}

/// `[laya_encoder]`: the frozen LAYA ModernBERT-large encoder (Phase 4),
/// ported from `Jev_laya:src-lib/darknet_cfg.cpp::parse_laya_encoder_section`.
/// Input is `[batch, 2, 1, seq_len]` (token id + attention mask); see
/// `src/darknet-core/src/laya_encoder.rs` for the maths and blob order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaEncoderSpec {
    pub hidden: usize,
    pub heads: usize,
    pub n_layers: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub ln_eps: f32,
    pub local_window: usize,
    pub rope_theta_full: f32,
    pub rope_theta_local: f32,
}

/// `[laya_head]`: the frozen LAYA decision head (Phase 4), two plain pre-norm
/// TransformerEncoderLayers (bias, ReLU FFN) over the encoder hidden states
/// plus a per-question-type embedding.  See
/// `src/darknet-core/src/laya_head.rs`.
#[derive(Clone, Debug, PartialEq)]
pub struct LayaHeadSpec {
    pub hidden: usize,
    pub heads: usize,
    pub n_layers: usize,
    pub ffn_dim: usize,
    pub ln_eps: f32,
    /// `0..=2`, or `-1` for a per-sample qtype input channel.
    pub qtype: i32,
}

/// Frozen DINOv2 ViT-S/14 feature encoder used by `[depth_anything]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAnythingSpec {
    pub patch_size: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp_dim: usize,
    pub pos_grid_h: usize,
    pub pos_grid_w: usize,
    pub ln_eps: f32,
    pub apply_imagenet_norm: bool,
    pub imagenet_mean: [f32; 3],
    pub imagenet_std: [f32; 3],
    pub taps: Vec<usize>,
    pub output_clamp: f32,
    /// Resolved path and layer-only tensors from cfg `weights=`. Such weights
    /// are excluded from the network's sequential checkpoint stream.
    pub weights_path: Option<PathBuf>,
    pub external_weights: Option<crate::depth_anything::DepthAnythingWeights>,
}

impl DepthAnythingSpec {
    pub fn geometry(&self, input: TensorShape) -> crate::depth_anything::DepthAnythingGeometry {
        crate::depth_anything::DepthAnythingGeometry {
            batch: input.batch,
            height: input.height,
            width: input.width,
            patch_size: self.patch_size,
            dim: self.dim,
            depth: self.depth,
            heads: self.heads,
            mlp_dim: self.mlp_dim,
            pos_grid_h: self.pos_grid_h,
            pos_grid_w: self.pos_grid_w,
            ln_eps: self.ln_eps,
            apply_imagenet_norm: self.apply_imagenet_norm,
            imagenet_mean: self.imagenet_mean,
            imagenet_std: self.imagenet_std,
            taps: self.taps.clone(),
            output_clamp: self.output_clamp,
        }
    }
}

/// Token ordering used by the historical VC attention path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VitVcPermutation {
    Sequence,
    Spatial8,
}

/// `[vit]`: a global vision-transformer block from `Jev_laya`.
#[derive(Clone, Debug, PartialEq)]
pub struct VitSpec {
    pub dim: usize,
    pub patch_size: usize,
    pub patch_stride: usize,
    pub patch_pad: usize,
    pub heads: usize,
    pub ffn_ratio: usize,
    pub mlp_dim: usize,
    pub learned_pos_embed: bool,
    pub pos_init_zero: bool,
    pub activation: Activation,
    pub msa_enabled: bool,
    pub msa_block_size: usize,
    pub msa_topk: usize,
    /// Selects the online-softmax reference implementation without a T×T cache.
    pub flash_attention: bool,
    /// Selects the quantized VC reference path, which requires flash attention.
    pub vc_attention: bool,
    /// ExpCast mode 0, 1, or 2.
    pub vc_expcast: u8,
    pub vc_perm: VitVcPermutation,
}

/// `[swin]`/`[transformer]`: source-shaped windowed attention block.
#[derive(Clone, Debug, PartialEq)]
pub struct SwinSpec {
    pub filters: usize,
    pub size: usize,
    pub heads: usize,
    pub shift: bool,
    pub ffn_ratio: usize,
    pub activation: Activation,
}

/// `[tucker]` / `[tucker_attention]` / `[tucker_attn]`: factorized local attention.
#[derive(Clone, Debug, PartialEq)]
pub struct TuckerAttentionSpec {
    pub filters: usize,
    pub size: usize,
    pub heads: usize,
    pub rank_q: usize,
    pub rank_k: usize,
    pub rank_v: usize,
    pub rank_o: usize,
    pub activation: Activation,
    pub msa_enabled: bool,
    pub msa_block_size: usize,
    pub msa_topk: usize,
}

/// `[mambavision]` / `[mambavision_block]`: selective-scan vision mixer.
#[derive(Clone, Debug, PartialEq)]
pub struct MambaVisionSpec {
    pub filters: usize,
    pub state: usize,
    pub conv_size: usize,
    pub dt_rank: usize,
    pub ffn_ratio: usize,
    pub activation: Activation,
    pub msa_enabled: bool,
    pub msa_block_size: usize,
    pub msa_topk: usize,
    pub msa_heads: usize,
    pub causal: bool,
}

/// `[kimi_kda]` / `[kimi_delta_attention]`: low-rank causal delta-rule mixer
/// followed by one shared MLP and top-k routed experts.
#[derive(Clone, Debug, PartialEq)]
pub struct KimiKdaSpec {
    pub filters: usize,
    pub rank: usize,
    pub experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub temperature: f32,
    pub load_balance: f32,
    pub activation: Activation,
}

/// `[yolox]`: the anchor-free detection head (Phase 2A of `tasks/rust/layers.md`),
/// ported from `Jev_laya:src-lib/darknet_cfg.cpp::parse_yolox_section`.  The
/// layer has no weights and no kernels; see
/// `src/darknet-core/src/modern_yolo.rs` for the decode, SimOTA assignment and
/// deltas.
#[derive(Clone, Debug, PartialEq)]
pub struct ModernYoloSpec {
    pub kind: ModernYoloKind,
    pub classes: usize,
    /// Maximum truth rows per image.
    pub max_boxes: usize,
    /// Distribution width for the ltrb heads; C++ clamps it to `[1, 64]` and
    /// ignores it for YOLOX.  Default 16.
    pub reg_max: usize,
    /// Ground-truth jitter used by the data loader.
    pub jitter: f32,
    /// Ground-truth resize factor used by the data loader.
    pub resize: f32,
    /// Truths smaller than this in grid cells are skipped by the Task-Aligned
    /// assignment.  `0` disables the gate (the C++ default).  It exists because
    /// on a coarse head an object narrower than one cell would otherwise compete
    /// for cells that belong to properly-scaled objects.
    pub min_object_cells: f32,
    /// Varifocal-loss exponent for the per-class targets.  `<= 0` (the default)
    /// disables the VFL weighting; the nested-object fix's per-class targets are
    /// applied either way.
    pub vfl_gamma: f32,
}

/// `[contrastive]`: the supervised-contrastive head, classifier branch
/// (deferred item (c) of `tasks/rust/layers.md` Phase 1b), ported from
/// `Jev_laya:src-lib/darknet_cfg.cpp::parse_contrastive_section`.  `yolo_layer`
/// selects the detection branch, which this port does not implement.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastiveSpec {
    /// Class count; the truth is `batch * classes` class probabilities.
    pub classes: usize,
    /// C++ `yolo_layer`: a relative index of the `[yolo]` layer to take the
    /// per-box class ids from, `0` for the classifier branch.
    pub yolo_layer: isize,
    pub temperature: f32,
    pub cls_normalizer: f32,
    /// Delta clip; the C++ default is `FLT_MAX`.
    pub max_delta: f32,
    /// Detection branch only: how many negative partners per positive.
    pub contrastive_neg_max: usize,
}

/// `[centernet]` (Phase 2A of `tasks/rust/layers.md`), ported from
/// `Jev_laya:src-lib/darknet_cfg.cpp::parse_centernet_section`.  No weights and no
/// kernels; see `src/darknet-core/src/centernet.rs` for the maths.
#[derive(Clone, Debug, PartialEq)]
pub struct CenterNetSpec {
    pub classes: usize,
    /// Maximum truth rows per image.
    pub max_boxes: usize,
    pub jitter: f32,
    pub resize: f32,
    /// Minimum output-heatmap radius, in grid cells.
    pub center_min_radius: i64,
    /// 3x3 strict-maximum peak filter at decode time.
    pub peak_nms: usize,
    /// 1 = ellipse target, 0 = the original circular one.
    pub anisotropic_gaussian: usize,
    /// Legacy diagnostic cutoff in input pixels (no longer a hard boost).
    pub small_threshold: f32,
    /// Reference size for the continuous tiny-object multiplier.
    pub small_ref_size: f32,
    /// Cap on that multiplier.
    pub small_boost: f32,
    /// Optional FPN assignment gate on the largest object side, in pixels.
    pub scale_min_px: f32,
    pub scale_max_px: f32,
    /// CenterNet radius overlap target.
    pub gaussian_iou: f32,
    /// Centre-heatmap focal-loss exponents.
    pub focal_alpha: f32,
    pub focal_beta: f32,
    pub hm_normalizer: f32,
    pub wh_normalizer: f32,
    pub off_normalizer: f32,
    pub max_delta: f32,
    pub delta_normalizer: f32,
}

/// Which of the three anchor-free heads a `[yolox]` / `[ppyoloe]` /
/// `[yolonas]` section selects (C++ `ModernYoloHeadKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModernYoloKind {
    /// cxcywh + SimOTA dynamic-k assignment, hard class targets.
    Yolox,
    /// ltrb distribution regression + Task-Aligned assignment, soft quality targets.
    Ppyoloe,
    /// As PP-YOLOE plus the NMS-aware duplicate down-weighting.
    Yolonas,
}

impl ModernYoloSpec {
    /// C++ `make_modern_yolo_layer`: YOLOX regresses four raw box values, the
    /// distribution heads regress `4 * (reg_max + 1)` logits.
    pub fn coords(&self) -> usize {
        match self.kind {
            ModernYoloKind::Yolox => 4,
            _ => 4 * (self.reg_max.clamp(1, 64) + 1),
        }
    }
}

/// `[omd_loss]` (Phase 1b of `tasks/rust/layers.md`), ported from
/// `Jev_laya:src-lib/darknet_cfg.cpp::parse_omd_loss_section`.  The layer is a
/// pass-through with no weights; every key below is the C++ one, including its
/// alias spellings.
#[derive(Clone, Debug, PartialEq)]
pub struct OmdLossSpec {
    /// Class count; `0` disables the class filter on box truth and selects the
    /// classification sampler when `max_boxes == 0`.
    pub classes: usize,
    /// Box truth rows per image; `0` switches to the classification sampler.
    pub max_boxes: usize,
    /// Floats per box row; `truth[0] == 0` ends an image's rows.
    pub truth_size: usize,
    /// The loss weight (C++ spells it `lambda`, `normalizer` or `cls_normalizer`).
    pub cls_normalizer: f32,
    /// Cosine-distance margin in `[0, 2]`.
    pub margin: f32,
    pub positive_weight: f32,
    pub negative_weight: f32,
    /// Also the per-sample weight of a background sample.
    pub background_weight: f32,
    /// Smallest-object side, in pixels of the network input, that gets boosted.
    pub small_threshold: f32,
    pub small_boost: f32,
    /// Sample budget; must be `>= 2`.
    pub max_samples: usize,
    /// Extra background samples per image (0 = none).
    pub background_samples: usize,
    /// `pool=1` averages each channel over the whole image for the
    /// classification sampler.
    pub pool: usize,
    /// Floor for the squared norm during normalisation.
    pub eps: f32,
    /// Delta clip; the C++ default is 10, not `FLT_MAX`.
    pub max_delta: f32,
    /// Log the pair statistics every iteration.
    pub show_details: bool,
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
            other => Err(Error::InvalidConfig(format!(
                "weights type \"{other}\" is not supported"
            ))),
        }
    }
}

impl ShortcutWeightsNorm {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "relu" | "avg_relu" => Ok(Self::Relu),
            "softmax" => Ok(Self::Softmax),
            other => Err(Error::InvalidConfig(format!(
                "weights normalization \"{other}\" is not supported"
            ))),
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

/// NMS variants implemented for the legacy OBB decoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObbNmsKind {
    /// Source default NMS: suppress using the decoded axis-aligned box IoU.
    Default,
    /// Source corners NMS also suppresses by decoded axis-aligned box IoU.
    Corners,
    /// Source greedy NMS: AABB IoU minus the fixed-power center-distance term.
    Greedy,
    /// Source DIoU NMS: AABB IoU minus the configurable center-distance term.
    Diou,
    /// Orientation-aware greedy NMS over OBB detections.
    Obb,
    /// DIoU-penalized OBB suppression with confidence×IoU weighted geometry merge.
    Weighted,
}

impl ObbNmsKind {
    fn parse(value: &str, line: usize) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" => Ok(Self::Default),
            "cornersnms" => Ok(Self::Corners),
            "greedynms" => Ok(Self::Greedy),
            "diounms" => Ok(Self::Diou),
            "obb" | "obb_nms" | "obbnms" => Ok(Self::Obb),
            "obb_w" | "obb_wnms" => Ok(Self::Weighted),
            other => Err(Error::InvalidConfig(format!(
                "[obb_yolo] at line {line}: unsupported nms_kind {other:?}; implemented kinds are `default`, `cornersnms`, `greedynms`, `diounms`, `obb`, and `obb_w`"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObbYoloSpec {
    pub yolo: YoloSpec,
    /// Source maximum labels/detections for this proposal head (`max`, default 200).
    pub max_boxes: usize,
    /// Optional source `obb_anchors` (u, v) directional prior per global anchor.
    pub front_anchor_offsets: Option<Vec<[f32; 2]>>,
    /// Source `front_loss_normalizer`, including its `front_point_normalizer` alias.
    pub front_loss_normalizer: f32,
    /// Source normalizer selecting covariance-W2 match/regression scoring.
    pub covariance_normalizer: f32,
    /// Source normalizer selecting COBB match scoring.
    pub cobb_normalizer: f32,
    /// Covariance-W2 front-length ratio factor.
    pub obb_length_ratio_k: f32,
    /// QBBV vIoU blend weight, clamped to `[0, 1]` like source cfg parsing.
    pub obb_viou_weight: f32,
    pub cobb_box_weight: f32,
    pub cobb_r_weight: f32,
    pub cobb_s_weight: f32,
    pub cobb_iou_lambda: f32,
    /// Whether `iou_loss` selects source CIoU-OBB match scoring.
    pub ciou_obb_loss: bool,
    /// Whether the selected CIoU-OBB variant uses the multiplicative loss.
    pub ciou_obb_multiplicative_loss: bool,
    pub nms_kind: ObbNmsKind,
    /// Source DIoU-NMS exponent (`beta_nms`, default 0.6).
    pub beta_nms: f32,
}

impl std::ops::Deref for ObbYoloSpec {
    type Target = YoloSpec;

    fn deref(&self) -> &Self::Target {
        &self.yolo
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObbRefineLoss {
    Carpn,
    CiouObb,
}

/// Source `[obb_refine]` configuration for the inference-only CARPN stage-2 head.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbRefineSpec {
    /// Absolute index of the sampled spatial feature layer.
    pub feature_layer: usize,
    /// Absolute index of the OBB-YOLO proposal source layer.
    pub proposal_layer: usize,
    pub classes: usize,
    pub max_boxes: usize,
    pub proposal_topk_train: usize,
    pub proposal_topk_infer: usize,
    pub proposal_jitter: usize,
    pub sample_grid: usize,
    pub proposal_capacity: usize,
    /// Flattened sample count times the feature-layer channel count.
    pub inputs: usize,
    pub front_range: f32,
    pub front_loss_normalizer: f32,
    pub quality_normalizer: f32,
    pub proposal_thresh: f32,
    pub csda_topk: usize,
    pub anchor_scales: Vec<f32>,
    pub loss: ObbRefineLoss,
    pub nms_kind: ObbNmsKind,
    pub beta_nms: f32,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfRadiusMode {
    Isotropic,
    Anisotropic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfInverseMethod {
    Newton,
    FixedPoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HfWarpSource {
    Previous,
    NetworkInput,
}

/// `[foveation_film]` / `[hf_film]`: per-channel affine conditioning from an earlier HF layer.
#[derive(Clone, Debug, PartialEq)]
pub struct FoveationFilmSpec {
    /// Resolved absolute index of the selected preceding HF layer.
    pub hf_layer: usize,
    pub param_count: usize,
    pub embed_dim: usize,
    pub backprop_to_hf: bool,
    pub film_init_scale: f32,
}

/// `[hyperbolic_foveation]` / `[hf]`: learnable radial image warp.
#[derive(Clone, Debug, PartialEq)]
pub struct HyperbolicFoveationSpec {
    pub init_cx: f32,
    pub init_cy: f32,
    pub init_radius: f32,
    pub init_radius_x: f32,
    pub init_radius_y: f32,
    pub init_strength: f32,
    pub small_box_px: f32,
    pub small_box_strength: f32,
    pub strength_loss: f32,
    pub tiny_center_loss: f32,
    pub tiny_radius_loss: f32,
    pub tiny_strength_loss: f32,
    pub strength_zero_loss: f32,
    pub hf_lr_mult: f32,
    pub alpha: f32,
    pub p_exp: f32,
    pub downsample: usize,
    pub pred_hidden: usize,
    pub nr_iters: usize,
    pub nr_eta: f32,
    pub inverse_method: HfInverseMethod,
    /// `None` means the previous-layer input; `Some(i)` selects layer i's output.
    pub pred_source_layer: Option<usize>,
    pub pred_input_shape: TensorShape,
    pub warp_source: HfWarpSource,
    pub radius_mode: HfRadiusMode,
    pub debug: bool,
    pub debug_interval: usize,
}

impl HyperbolicFoveationSpec {
    pub const fn param_count(&self) -> usize {
        match self.radius_mode {
            HfRadiusMode::Isotropic => 4,
            HfRadiusMode::Anisotropic => 5,
        }
    }

    pub fn predictor_input_dim(&self) -> Option<usize> {
        let height = (self.pred_input_shape.height / self.downsample).max(1);
        let width = (self.pred_input_shape.width / self.downsample).max(1);
        self.pred_input_shape
            .channels
            .checked_mul(height)?
            .checked_mul(width)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecursiveBlockSpec {
    pub loops: usize,
    pub injection: bool,
    pub body_scale: f32,
    pub residual_scale: f32,
    pub injection_scale: f32,
    /// 0: static blend; 1: controller-gated Ouroboros recurrence.
    pub ouroboros: usize,
    pub gate_bias: f32,
    pub gamma_clip: f32,
    pub gate_clip: f32,
    pub lattice_group: usize,
    pub lattice_threshold: f32,
    pub lattice_temperature: f32,
    pub lattice_mix: f32,
    /// Consumed nested body layers, in cfg source order.
    pub body: Vec<Layer>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerKind {
    RecursiveBlock(RecursiveBlockSpec),
    FoveationFilm(FoveationFilmSpec),
    HyperbolicFoveation(HyperbolicFoveationSpec),
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
    /// Legacy six-coordinate `[obb_yolo]`; inference is decoded separately
    /// from axis-aligned YOLO and training remains unsupported.
    ObbYolo(ObbYoloSpec),
    /// Inference-only CARPN stage-2 head, sourced by `[obb_yolo]` proposals.
    ObbRefine(ObbRefineSpec),
    GaussianYolo(YoloSpec),
    Region(RegionSpec),
    Reorg(ReorgSpec),
    ChannelSlice(ChannelSliceSpec),
    ChannelShuffle(ChannelShuffleSpec),
    Softmax {
        groups: usize,
    },
    Cost {
        kind: String,
        scale: f32,
    },
    Logistic,
    Regression(RegressionSpec),
    JevCe(JevCeSpec),
    Segmentation(SegmentationSpec),
    OmdLoss(OmdLossSpec),
    Dropout {
        probability: Option<f32>,
        /// `dropblock=1` selects DropBlock instead of element-wise dropout.
        dropblock: Option<DropblockSpec>,
    },
    ScaleChannels(ScaleChannelsSpec),
    Sam(SamSpec),
    Contrastive(ContrastiveSpec),
    /// `[yolox]`, `[ppyoloe]` and `[yolonas]`; all three share
    /// [`ModernYoloSpec`] and its `kind`.
    Yolox(ModernYoloSpec),
    Ppyoloe(ModernYoloSpec),
    Yolonas(ModernYoloSpec),
    /// `[centernet]`.
    CenterNet(CenterNetSpec),
    /// `[laya_scorer]`, whose weights live in its own blob type.
    LayaScorer(LayaScorerSpec),
    /// `[laya_marker_gather]`: two-input, weightless per-option gather.
    LayaMarkerGather(LayaMarkerGatherSpec),
    /// `[laya_act_head]`: three-input act/escalate head over masked-softmax
    /// summary features.
    LayaActHead(LayaActHeadSpec),
    /// `[laya_loss]`: masked soft-target loss (CE or RLCD) over option logits.
    LayaLoss(LayaLossSpec),
    /// `[laya_encoder]`: frozen LAYA ModernBERT-large encoder.
    LayaEncoder(LayaEncoderSpec),
    /// `[laya_head]`: frozen LAYA decision head.
    LayaHead(LayaHeadSpec),
    /// `[depth_anything]`: frozen DINOv2 ViT-S/14 feature encoder.
    DepthAnything(DepthAnythingSpec),
    /// `[vit]`: global vision transformer with learned patch projection.
    Vit(VitSpec),
    /// `[swin]`, `[swin_block]`, `[transformer]`, `[transformer_block]`.
    Swin(SwinSpec),
    /// `[tucker]`, `[tucker_attention]`, `[tucker_attn]`.
    TuckerAttention(TuckerAttentionSpec),
    /// `[mambavision]` / `[mambavision_block]` selective-scan vision mixer.
    MambaVision(MambaVisionSpec),
    /// `[kimi_kda]` / `[kimi_delta_attention]` causal delta-rule/MoE mixer.
    KimiKda(KimiKdaSpec),
    /// `[moe_convolutional]`: a learned mixture of `MoeSpec::experts`
    /// convolutional experts gated per image by a small router.
    Moe(MoeSpec),
    /// `[eml_convolutional]`: two parallel convolutions and an EML combine.
    EmlConvolutional(EmlConvolutionalSpec),
    /// `[graph_conv]` / `[graph_convolutional]`: learned valid-neighbor aggregation.
    GraphConv(GraphConvSpec),
    /// `[dcnv4]` / `[dcnv4_convolutional]`: channel-preserving deformable aggregation.
    Dcnv4(Dcnv4Spec),
    /// `[deform_conv]` / `[deformable_convolutional]`: DCNv1/2 sampling.
    DeformConv(DeformConvSpec),
    /// `[dsconv]` / `[dynamic_snake_conv]`: Dynamic Snake Conv.
    DsConv(DsConvSpec),
    /// `[wmhf]` / wavelet-Mamba High-frequency Fusion.
    Wmhf(WmhfSpec),
    /// `[clifford]` geometric feature mixer.
    Clifford(CliffordSpec),
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
            Self::RecursiveBlock(_) => "recursive_block",
            Self::FoveationFilm(_) => "foveation_film",
            Self::HyperbolicFoveation(_) => "hyperbolic_foveation",
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
            Self::ObbYolo(_) => "obb_yolo",
            Self::ObbRefine(_) => "obb_refine",
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
            Self::OmdLoss(_) => "omd_loss",
            Self::Dropout { .. } => "dropout",
            Self::ScaleChannels(_) => "scale_channels",
            Self::Sam(_) => "sam",
            Self::Contrastive(_) => "contrastive",
            Self::Yolox(_) => "yolox",
            Self::Ppyoloe(_) => "ppyoloe",
            Self::Yolonas(_) => "yolonas",
            Self::CenterNet(_) => "centernet",
            Self::LayaScorer(_) => "laya_scorer",
            Self::LayaMarkerGather(_) => "laya_marker_gather",
            Self::LayaActHead(_) => "laya_act_head",
            Self::LayaLoss(_) => "laya_loss",
            Self::LayaEncoder(_) => "laya_encoder",
            Self::LayaHead(_) => "laya_head",
            Self::DepthAnything(_) => "depth_anything",
            Self::Vit(_) => "vit",
            Self::Swin(_) => "swin",
            Self::TuckerAttention(_) => "tucker_attention",
            Self::MambaVision(_) => "mambavision",
            Self::KimiKda(_) => "kimi_kda",
            Self::Moe(_) => "moe_convolutional",
            Self::EmlConvolutional(_) => "eml_convolutional",
            Self::GraphConv(_) => "graph_conv",
            Self::Dcnv4(_) => "dcnv4",
            Self::DeformConv(_) => "deform_conv",
            Self::DsConv(_) => "dsconv",
            Self::Wmhf(_) => "wmhf",
            Self::Clifford(_) => "clifford",
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
            Self::RecursiveBlock(_) => None,
            Self::Convolutional(spec) => Some(&spec.activation),
            Self::GraphConv(spec) => Some(&spec.activation),
            Self::Dcnv4(spec) => Some(&spec.activation),
            Self::DeformConv(spec) => Some(&spec.activation),
            Self::DsConv(spec) => Some(&spec.activation),
            Self::Wmhf(spec) => Some(&spec.activation),
            Self::Deconvolutional(spec) => Some(&spec.activation),
            Self::Connected(spec) => Some(&spec.activation),
            Self::Shortcut(spec) => Some(&spec.activation),
            Self::ScaleChannels(spec) => Some(&spec.activation),
            Self::Sam(spec) => Some(&spec.activation),
            Self::Local(spec) => Some(&spec.activation),
            Self::Rnn(spec) => Some(&spec.activation),
            Self::Crnn(spec) => Some(&spec.activation),
            Self::Vit(spec) => Some(&spec.activation),
            Self::Swin(spec) => Some(&spec.activation),
            Self::TuckerAttention(spec) => Some(&spec.activation),
            Self::MambaVision(spec) => Some(&spec.activation),
            Self::KimiKda(spec) => Some(&spec.activation),
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
    Named(String),
}

impl WeightsArray {
    pub fn name(&self) -> &str {
        match self {
            Self::Biases => "biases",
            Self::Scales => "scales",
            Self::RollingMean => "rolling_mean",
            Self::RollingVariance => "rolling_variance",
            Self::Weights => "weights",
            Self::Named(name) => name,
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
            LayerKind::RecursiveBlock(spec) => {
                let entries = spec
                    .body
                    .iter()
                    .filter_map(Layer::weights)
                    .flat_map(|weights| weights.entries)
                    .collect::<Vec<_>>();
                (!entries.is_empty()).then_some(LayerWeights { entries })
            }
            LayerKind::DepthAnything(spec) => {
                if spec.weights_path.is_some() {
                    return None;
                }
                let geometry = spec.geometry(self.input_shape);
                let entries = crate::depth_anything::weight_array_sizes(&geometry)
                    .expect("validated depth_anything weight geometry")
                    .into_iter()
                    .map(|(name, len)| WeightsEntry {
                        array: WeightsArray::Named(name.to_owned()),
                        len,
                    })
                    .collect();
                Some(LayerWeights { entries })
            }
            LayerKind::HyperbolicFoveation(spec) => {
                let mut entries = vec![WeightsEntry {
                    array: WeightsArray::Named("hf_params".to_owned()),
                    len: spec.param_count(),
                }];
                if spec.pred_hidden > 0 {
                    let input_dim = spec
                        .predictor_input_dim()
                        .expect("validated hyperbolic-foveation predictor dimensions");
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Named("hf_pred_w1".to_owned()),
                            len: spec.pred_hidden * input_dim,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("hf_pred_b1".to_owned()),
                            len: spec.pred_hidden,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("hf_pred_w2".to_owned()),
                            len: spec.param_count() * spec.pred_hidden,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("hf_pred_b2".to_owned()),
                            len: spec.param_count(),
                        },
                    ]);
                }
                Some(LayerWeights { entries })
            }
            LayerKind::FoveationFilm(spec) => Some(LayerWeights {
                entries: vec![
                    WeightsEntry {
                        array: WeightsArray::Named("w_embed".to_owned()),
                        len: spec.embed_dim * spec.param_count,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("b_embed".to_owned()),
                        len: spec.embed_dim,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("w_gamma".to_owned()),
                        len: self.input_shape.channels * spec.embed_dim,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("b_gamma".to_owned()),
                        len: self.input_shape.channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("w_beta".to_owned()),
                        len: self.input_shape.channels * spec.embed_dim,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("b_beta".to_owned()),
                        len: self.input_shape.channels,
                    },
                ],
            }),
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
            LayerKind::ObbRefine(spec) => Some(LayerWeights {
                entries: vec![
                    WeightsEntry {
                        array: WeightsArray::Biases,
                        len: 5,
                    },
                    WeightsEntry {
                        array: WeightsArray::Weights,
                        len: 5 * spec.inputs,
                    },
                ],
            }),
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
            LayerKind::Swin(spec) => {
                let input_channels = self.input_shape.channels;
                let output_channels = self.output_shape.channels;
                let hidden = output_channels * spec.ffn_ratio;
                let bias_table_len = (2 * spec.size - 1) * (2 * spec.size - 1);
                let mut entries = Vec::with_capacity(16);
                for (name, len) in [
                    ("swin_qkv_weights", 3 * input_channels * input_channels),
                    ("swin_qkv_bias", 3 * input_channels),
                    ("swin_wo", output_channels * input_channels),
                    ("swin_wo_bias", output_channels),
                    ("swin_ln1_gamma", input_channels),
                    ("swin_ln1_beta", input_channels),
                    ("swin_ln2_gamma", output_channels),
                    ("swin_ln2_beta", output_channels),
                    ("swin_ffn_w1", hidden * output_channels),
                    ("swin_ffn_b1", hidden),
                    ("swin_ffn_w2", output_channels * hidden),
                    ("swin_ffn_b2", output_channels),
                ] {
                    entries.push(WeightsEntry {
                        array: WeightsArray::Named(name.to_owned()),
                        len,
                    });
                }
                if input_channels != output_channels {
                    entries.push(WeightsEntry {
                        array: WeightsArray::Named("swin_res_proj".to_owned()),
                        len: output_channels * input_channels,
                    });
                }
                entries.extend([
                    WeightsEntry {
                        array: WeightsArray::Named("swin_rel_pos_bias".to_owned()),
                        len: spec.heads * bias_table_len,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("swin_mhc_scales".to_owned()),
                        len: 12,
                    },
                ]);
                Some(LayerWeights { entries })
            }
            LayerKind::TuckerAttention(spec) => {
                let channels = self.input_shape.channels;
                let weights = crate::tucker_attention::tucker_weight_count(
                    channels,
                    spec.heads,
                    spec.rank_q,
                    spec.rank_k,
                    spec.rank_v,
                    spec.rank_o,
                )
                .expect("validated Tucker attention geometry");
                Some(LayerWeights {
                    entries: vec![
                        WeightsEntry {
                            array: WeightsArray::Biases,
                            len: channels,
                        },
                        WeightsEntry {
                            array: WeightsArray::Weights,
                            len: weights,
                        },
                    ],
                })
            }
            LayerKind::MambaVision(spec) => Some(LayerWeights {
                entries: crate::mambavision::mambavision_weight_blocks(
                    self.input_shape.channels,
                    spec,
                )
                .expect("validated MambaVision geometry")
                .into_iter()
                .map(|block| WeightsEntry {
                    array: WeightsArray::Named(block.name.to_owned()),
                    len: block.range.len(),
                })
                .collect(),
            }),
            LayerKind::KimiKda(spec) => {
                let (biases, weights) = crate::kimi_kda::kimi_kda_parameter_counts(
                    self.input_shape.channels,
                    spec.rank,
                    spec.experts,
                    spec.hidden,
                )
                .expect("validated Kimi KDA geometry");
                Some(LayerWeights {
                    entries: vec![
                        WeightsEntry {
                            array: WeightsArray::Biases,
                            len: biases,
                        },
                        WeightsEntry {
                            array: WeightsArray::Weights,
                            len: weights,
                        },
                    ],
                })
            }
            LayerKind::Vit(spec) => {
                let dim = spec.dim;
                let patch_dim = spec.patch_size * spec.patch_size * self.input_shape.channels;
                let tokens = self.output_shape.width * self.output_shape.height;
                let mut entries = Vec::with_capacity(16);
                for (name, len) in [
                    ("vit_patch_embed", dim * patch_dim),
                    ("vit_patch_bias", dim),
                    ("biases", 3 * dim),
                    ("vit_ln1_gamma", dim),
                    ("vit_ln1_beta", dim),
                    ("weights", 3 * dim * dim),
                    ("vit_wo", dim * dim),
                    ("vit_wo_bias", dim),
                    ("vit_ln2_gamma", dim),
                    ("vit_ln2_beta", dim),
                    ("vit_ffn_w1", spec.mlp_dim * dim),
                    ("vit_ffn_b1", spec.mlp_dim),
                    ("vit_ffn_w2", dim * spec.mlp_dim),
                    ("vit_ffn_b2", dim),
                    ("vit_pos_embed", tokens * dim),
                    ("scales", 12),
                ] {
                    entries.push(WeightsEntry {
                        array: WeightsArray::Named(name.to_owned()),
                        len,
                    });
                }
                Some(LayerWeights { entries })
            }
            // `[moe_convolutional]`'s blob order is C++
            // `save_moe_convolutional_weights`: one complete
            // `save_convolutional_weights` block per expert, then the router
            // biases and router weights.
            LayerKind::Moe(spec) => {
                let mut entries = Vec::new();
                for _ in 0..spec.experts {
                    append_convolutional_weight_entries(
                        &mut entries,
                        self.input_shape.channels,
                        spec.expert.filters,
                        spec.expert.size,
                        spec.expert.groups,
                        spec.expert.batch_normalize,
                    );
                }
                entries.push(WeightsEntry {
                    array: WeightsArray::Biases,
                    len: spec.experts,
                });
                entries.push(WeightsEntry {
                    array: WeightsArray::Weights,
                    len: spec.experts * self.input_shape.channels,
                });
                Some(LayerWeights { entries })
            }
            LayerKind::EmlConvolutional(spec) => {
                let mut entries = Vec::new();
                for _ in 0..2 {
                    append_convolutional_weight_entries(
                        &mut entries,
                        self.input_shape.channels,
                        spec.branch.filters,
                        spec.branch.size,
                        spec.branch.groups,
                        spec.branch.batch_normalize,
                    );
                }
                Some(LayerWeights { entries })
            }
            LayerKind::GraphConv(spec) => {
                let channels_per_group = self.input_shape.channels / spec.groups;
                let kernel_area = spec.size * spec.size;
                let mut entries = vec![WeightsEntry {
                    array: WeightsArray::Biases,
                    len: spec.filters,
                }];
                if spec.batch_normalize {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Scales,
                            len: spec.filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingMean,
                            len: spec.filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingVariance,
                            len: spec.filters,
                        },
                    ]);
                }
                entries.push(WeightsEntry {
                    array: WeightsArray::Weights,
                    len: spec.filters * channels_per_group,
                });
                if spec.use_self {
                    entries.push(WeightsEntry {
                        array: WeightsArray::Named("graph_self_weights".to_owned()),
                        len: spec.filters * channels_per_group,
                    });
                }
                if spec.edge_mode == 1 {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Named("graph_edge_kernel".to_owned()),
                            len: spec.groups * kernel_area * 2 * channels_per_group,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("graph_edge_biases".to_owned()),
                            len: spec.groups * kernel_area,
                        },
                    ]);
                }
                Some(LayerWeights { entries })
            }
            LayerKind::Dcnv4(spec) => {
                let channels = self.input_shape.channels;
                let (dummy_weights, offset_weights, offset_dim) =
                    dcnv4_parameter_layout(channels, spec)
                        .expect("validated DCNv4 parameter geometry");
                let mut entries = vec![WeightsEntry {
                    array: WeightsArray::Biases,
                    len: channels,
                }];
                if spec.batch_normalize {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Scales,
                            len: channels,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingMean,
                            len: channels,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingVariance,
                            len: channels,
                        },
                    ]);
                }
                entries.extend([
                    WeightsEntry {
                        array: WeightsArray::Weights,
                        len: dummy_weights,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("offset_weights".to_owned()),
                        len: offset_weights,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("offset_biases".to_owned()),
                        len: offset_dim,
                    },
                ]);
                Some(LayerWeights { entries })
            }
            LayerKind::DsConv(spec) => {
                let channels = self.input_shape.channels;
                let main_weights = spec
                    .filters
                    .checked_mul(channels)
                    .and_then(|count| count.checked_mul(spec.size))
                    .expect("validated dsconv collapse weight count");
                let offset_weights = 2usize
                    .checked_mul(spec.size)
                    .and_then(|count| count.checked_mul(channels))
                    .and_then(|count| count.checked_mul(9))
                    .expect("validated dsconv offset weight count");
                Some(LayerWeights {
                    entries: vec![
                        WeightsEntry {
                            array: WeightsArray::Biases,
                            len: spec.filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::Weights,
                            len: main_weights,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("offset_weights".to_owned()),
                            len: offset_weights,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("offset_biases".to_owned()),
                            len: 2 * spec.size,
                        },
                    ],
                })
            }
            LayerKind::Wmhf(spec) => {
                let channels = self.input_shape.channels;
                let local_mix_channels = spec
                    .local_channels
                    .checked_mul(3)
                    .expect("validated WMHF local mixer channels");
                let scan_parameter_count = spec
                    .global_channels
                    .checked_mul(4)
                    .expect("validated WMHF scan parameter count");
                let mut entries = Vec::new();
                append_convolutional_weight_entries(
                    &mut entries,
                    channels,
                    spec.filters,
                    1,
                    1,
                    spec.batch_normalize,
                );
                for size in [3, 5, 7] {
                    append_convolutional_weight_entries(
                        &mut entries,
                        spec.local_channels,
                        spec.local_channels,
                        size,
                        spec.local_channels,
                        false,
                    );
                }
                append_convolutional_weight_entries(
                    &mut entries,
                    local_mix_channels,
                    spec.local_channels,
                    1,
                    1,
                    spec.batch_normalize,
                );
                append_convolutional_weight_entries(
                    &mut entries,
                    spec.global_channels,
                    spec.filters,
                    1,
                    1,
                    false,
                );
                append_convolutional_weight_entries(
                    &mut entries,
                    spec.filters,
                    spec.filters,
                    1,
                    1,
                    spec.batch_normalize,
                );
                entries.push(WeightsEntry {
                    array: WeightsArray::Named("scan_weights".to_owned()),
                    len: scan_parameter_count,
                });
                Some(LayerWeights { entries })
            }
            LayerKind::Clifford(spec) => {
                let channels = self.input_shape.channels;
                let raw_channels = crate::clifford::raw_channels(spec, channels);
                let mut entries = vec![
                    WeightsEntry {
                        array: WeightsArray::Named("ln_gamma".to_owned()),
                        len: channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("ln_beta".to_owned()),
                        len: channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("w_det".to_owned()),
                        len: channels * channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("b_det".to_owned()),
                        len: channels,
                    },
                ];
                for _ in 0..spec.num_dwconv {
                    append_convolutional_weight_entries(
                        &mut entries,
                        channels,
                        channels,
                        spec.dwconv_size,
                        channels,
                        true,
                    );
                }
                entries.extend([
                    WeightsEntry {
                        array: WeightsArray::Named("w_proj".to_owned()),
                        len: channels * raw_channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("b_proj".to_owned()),
                        len: channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("w_gate".to_owned()),
                        len: channels * 2 * channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("b_gate".to_owned()),
                        len: channels,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("layer_scale".to_owned()),
                        len: channels,
                    },
                ]);
                if spec.gffn_mode != CliffordGffnMode::Local {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Named("w_proj_g".to_owned()),
                            len: channels * raw_channels,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("b_proj_g".to_owned()),
                            len: channels,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("w_gate_g".to_owned()),
                            len: channels * 2 * channels,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("b_gate_g".to_owned()),
                            len: channels,
                        },
                    ]);
                    if spec.global_op == CliffordGlobalOp::Fourier {
                        let fourier_weights = channels * spec.fno_modes_h * spec.fno_modes_w;
                        entries.extend([
                            WeightsEntry {
                                array: WeightsArray::Named("fno_w_re".to_owned()),
                                len: fourier_weights,
                            },
                            WeightsEntry {
                                array: WeightsArray::Named("fno_w_im".to_owned()),
                                len: fourier_weights,
                            },
                        ]);
                    }
                }
                Some(LayerWeights { entries })
            }
            LayerKind::DeformConv(spec) => {
                let channels = self.input_shape.channels;
                let kernel_area = spec
                    .size
                    .checked_mul(spec.size)
                    .expect("validated deformable-convolution kernel size");
                let offset_filters = kernel_area
                    .checked_mul(2)
                    .expect("validated deformable-convolution offset count");
                let main_weights = spec
                    .filters
                    .checked_mul(channels / spec.groups)
                    .and_then(|count| count.checked_mul(kernel_area))
                    .expect("validated deformable-convolution weight count");
                let offset_weights = offset_filters
                    .checked_mul(channels)
                    .and_then(|count| count.checked_mul(kernel_area))
                    .expect("validated deformable-convolution offset weight count");
                let mut entries = vec![WeightsEntry {
                    array: WeightsArray::Biases,
                    len: spec.filters,
                }];
                if spec.batch_normalize {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Scales,
                            len: spec.filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingMean,
                            len: spec.filters,
                        },
                        WeightsEntry {
                            array: WeightsArray::RollingVariance,
                            len: spec.filters,
                        },
                    ]);
                }
                entries.extend([
                    WeightsEntry {
                        array: WeightsArray::Weights,
                        len: main_weights,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("offset_weights".to_owned()),
                        len: offset_weights,
                    },
                    WeightsEntry {
                        array: WeightsArray::Named("offset_biases".to_owned()),
                        len: offset_filters,
                    },
                ]);
                if spec.use_mask {
                    entries.extend([
                        WeightsEntry {
                            array: WeightsArray::Named("mask_weights".to_owned()),
                            len: kernel_area * channels * kernel_area,
                        },
                        WeightsEntry {
                            array: WeightsArray::Named("mask_biases".to_owned()),
                            len: kernel_area,
                        },
                    ]);
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

        let mut net = NetConfig::from_section(net_section)?;
        if !net_section.values.contains_key("try_fix_nan") {
            net.try_fix_nan = cfg.sections.iter().skip(1).any(|section| {
                matches!(
                    section.name.as_str(),
                    "deform_conv"
                        | "deformable_convolutional"
                        | "dcnv4"
                        | "dcnv4_convolutional"
                        | "dsconv"
                        | "dynamic_snake_conv"
                )
            });
        }
        let mut current = net.input_shape();
        let mut layers = Vec::new();
        let cfg_parent = cfg
            .path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();

        let mut section_index = 1;
        while let Some(section) = cfg.sections.get(section_index) {
            let layer_index = layers.len();
            let input_shape = current;
            let (kind, consumed_body_sections) =
                if matches!(section.name.as_str(), "recursive_block" | "recurrent_block") {
                    let spec = parse_recursive_block_spec(
                        section,
                        &cfg.sections,
                        section_index,
                        current,
                        &cfg_parent,
                        net.time_steps,
                        net.obb,
                    )?;
                    let consumed = spec.body.len();
                    (LayerKind::RecursiveBlock(spec), consumed)
                } else {
                    (
                        parse_layer_kind(
                            section,
                            layer_index,
                            current,
                            &layers,
                            &cfg_parent,
                            net.time_steps,
                            net.obb,
                        )?,
                        0,
                    )
                };
            if let Some(activation) = kind.activation()
                && let Activation::Other(name) = activation
            {
                let hint = if name == "eml" {
                    " (eml is only used by [eml_convolutional], which is not implemented)"
                } else {
                    ""
                };
                return Err(Error::InvalidConfig(format!(
                    "[{}] at line {}: unsupported activation \"{name}\"{hint}; C++ would silently use relu",
                    section.name, section.line
                )));
            }
            let channel_normalization = match &kind {
                LayerKind::Convolutional(spec) => {
                    spec.activation.to_channel_normalization().is_some()
                }
                LayerKind::Moe(spec) => spec.expert.activation.to_channel_normalization().is_some(),
                _ => kind
                    .activation()
                    .is_some_and(|activation| activation.to_channel_normalization().is_some()),
            };
            if channel_normalization
                && !matches!(&kind, LayerKind::Convolutional(_) | LayerKind::Dcnv4(_))
            {
                return Err(Error::InvalidConfig(format!(
                    "[{}] at line {}: channel-normalizing activations are only supported on [convolutional]",
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
            section_index += consumed_body_sections + 1;
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
            set_layer_batch(layer, batch);
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
        if self.layers.iter().any(layer_contains_connected) {
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
            let mut kind = network.layers[index].kind.clone();
            if let LayerKind::RecursiveBlock(spec) = &mut kind {
                reshape_recursive_body(spec, current, index)?;
            }
            let output = infer_output_shape(&kind, index, current, &network.layers[..index])?;
            network.layers[index].kind = kind;
            network.layers[index].input_shape = current;
            network.layers[index].output_shape = output;
        }
        Ok(network)
    }
}

fn set_layer_batch(layer: &mut Layer, batch: usize) {
    layer.input_shape.batch = batch;
    layer.output_shape.batch = batch;
    if let LayerKind::RecursiveBlock(spec) = &mut layer.kind {
        for body_layer in &mut spec.body {
            set_layer_batch(body_layer, batch);
        }
    }
}

fn layer_contains_connected(layer: &Layer) -> bool {
    match &layer.kind {
        LayerKind::Connected(_) => true,
        LayerKind::RecursiveBlock(spec) => spec.body.iter().any(layer_contains_connected),
        _ => false,
    }
}

fn reshape_recursive_body(
    spec: &mut RecursiveBlockSpec,
    input_shape: TensorShape,
    parent_index: usize,
) -> Result<()> {
    let mut current = input_shape;
    let mut built = Vec::with_capacity(spec.body.len());
    for index in 0..spec.body.len() {
        let mut kind = spec.body[index].kind.clone();
        if let LayerKind::RecursiveBlock(nested) = &mut kind {
            reshape_recursive_body(nested, current, index)?;
        }
        let output = infer_output_shape(&kind, index, current, &built)?;
        spec.body[index].kind = kind;
        spec.body[index].index = index;
        spec.body[index].input_shape = current;
        spec.body[index].output_shape = output;
        built.push(spec.body[index].clone());
        current = output;
    }
    if current.width != input_shape.width
        || current.height != input_shape.height
        || current.channels != input_shape.channels
    {
        return Err(Error::InvalidConfig(format!(
            "[recursive_block] layer #{parent_index}: resized body output {} must match input {}",
            current, input_shape
        )));
    }
    Ok(())
}

fn parse_recursive_block_spec(
    section: &RawSection,
    sections: &[RawSection],
    section_index: usize,
    input_shape: TensorShape,
    cfg_parent: &Path,
    time_steps: usize,
    net_obb: bool,
) -> Result<RecursiveBlockSpec> {
    const PARENT_KEYS: &[&str] = &[
        "loops",
        "injection",
        "body",
        "body_scale",
        "residual_scale",
        "injection_scale",
        "ouroboros",
        "gate_bias",
        "gamma_clip",
        "gate_clip",
        "lora_rank",
        "lora_alpha",
        "lora_freeze_base",
        "lora_diag_clip",
        "lora_diag_init",
        "lattice_group",
        "lattice_threshold",
        "lattice_temperature",
        "lattice_mix",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    reject_section_keys(section, PARENT_KEYS)?;
    let clip = section.get_f32("clip", 0.0)?;
    let grad_centr = section.get_usize("grad_centr", 0)?;
    if !clip.is_finite() || clip < 0.0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: clip must be finite and non-negative",
            section.name, section.line
        )));
    }
    if clip > 0.0 || grad_centr != 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: recursive-block clip/grad_centr updates are unsupported",
            section.name, section.line
        )));
    }

    if let Some(key) = section.values.keys().find(|key| key.starts_with("lora_")) {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: `{key}` is a Stage-2 Conv-LoRA option and is unsupported in the Stage-1 CPU port",
            section.name, section.line
        )));
    }

    let line = section.line;
    let loops = section.get_usize("loops", 4)?;
    if loops == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: loops must be >= 1",
            section.name
        )));
    }
    let injection_value = section.get_i64("injection", 1)?;
    if !matches!(injection_value, 0 | 1) {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: injection must be 0 or 1, got {injection_value}",
            section.name
        )));
    }
    let injection = injection_value != 0;
    let body_count = section.get_usize("body", 1)?;
    if body_count == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: body must be >= 1",
            section.name
        )));
    }
    let end = section_index
        .checked_add(body_count)
        .ok_or_else(|| Error::InvalidConfig("recursive_block body range overflows".to_owned()))?;
    if end >= sections.len() {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: body={body_count} but not enough sections follow",
            section.name
        )));
    }

    let body_scale = section.get_f32("body_scale", if injection { 1.0 / 3.0 } else { 0.5 })?;
    let residual_scale =
        section.get_f32("residual_scale", if injection { 1.0 / 3.0 } else { 0.5 })?;
    let injection_scale =
        section.get_f32("injection_scale", if injection { 1.0 / 3.0 } else { 0.0 })?;
    for (name, value) in [
        ("body_scale", body_scale),
        ("residual_scale", residual_scale),
        ("injection_scale", injection_scale),
    ] {
        if !value.is_finite() {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {line}: {name} must be finite",
                section.name
            )));
        }
    }

    let ouroboros_value = section.get_i64("ouroboros", 0)?;
    if !matches!(ouroboros_value, 0 | 1) {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: ouroboros must be 0 or 1 in the Stage-1 CPU port; mode 2 (Conv-LoRA) is Stage 2",
            section.name
        )));
    }
    let ouroboros = ouroboros_value as usize;
    if ouroboros == 1 {
        let channels = input_shape.channels;
        let outputs = channels.checked_mul(2).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: controller output size overflows",
                section.name
            ))
        })?;
        let inputs = channels.checked_add(1).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: controller input size overflows",
                section.name
            ))
        })?;
        outputs.checked_mul(inputs).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: controller weight count overflows",
                section.name
            ))
        })?;
    }
    let has_controller_option = ["gate_bias", "gamma_clip", "gate_clip"]
        .iter()
        .any(|key| section.get(key).is_some());
    if ouroboros == 0 && has_controller_option {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: gate_bias/gamma_clip/gate_clip require ouroboros=1",
            section.name
        )));
    }
    let gate_bias = section.get_f32("gate_bias", -2.0)?;
    let gamma_clip = section.get_f32("gamma_clip", 4.0)?;
    let gate_clip = section.get_f32("gate_clip", 30.0)?;
    if ouroboros == 1
        && [gate_bias, gamma_clip, gate_clip]
            .into_iter()
            .any(|value| !value.is_finite())
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: controller bias and clip values must be finite",
            section.name
        )));
    }

    let lattice_group = section.get_usize("lattice_group", 0)?;
    let lattice_threshold = section.get_f32("lattice_threshold", 0.10)?;
    let lattice_temperature = section.get_f32("lattice_temperature", 1.0)?;
    let lattice_mix = section.get_f32("lattice_mix", 0.0)?;
    if !lattice_threshold.is_finite()
        || !(0.0..=1.0).contains(&lattice_threshold)
        || !lattice_mix.is_finite()
        || !(0.0..=1.0).contains(&lattice_mix)
        || !lattice_temperature.is_finite()
        || lattice_temperature <= 0.0
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: lattice threshold/mix must be in [0,1] and temperature must be finite and positive",
            section.name
        )));
    }
    if lattice_group > 1 && input_shape.channels % lattice_group != 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: lattice_group={lattice_group} must divide input channels={}",
            section.name, input_shape.channels
        )));
    }

    let mut body = Vec::with_capacity(body_count);
    let mut body_input = input_shape;
    for body_index in 0..body_count {
        let body_section = &sections[section_index + body_index + 1];
        validate_recursive_body_keys(body_section)?;
        if body_section.get("share_index").is_some() {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {}: shared convolution parameters are unsupported inside [recursive_block]",
                body_section.name, body_section.line
            )));
        }
        let kind = parse_layer_kind(
            body_section,
            body_index,
            body_input,
            &body,
            cfg_parent,
            time_steps,
            net_obb,
        )?;
        if let Some(Activation::Other(name)) = kind.activation() {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {}: unsupported activation `{name}` in recursive body",
                body_section.name, body_section.line
            )));
        }
        let output_shape = infer_output_shape(&kind, body_index, body_input, &body)?;
        body.push(Layer {
            index: body_index,
            line: body_section.line,
            input_shape: body_input,
            output_shape,
            kind,
            train: LayerTrainOptions::default(),
        });
        body_input = output_shape;
    }
    if body_input.width != input_shape.width
        || body_input.height != input_shape.height
        || body_input.channels != input_shape.channels
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: body changes dimensions from {} to {}; body output must match input",
            section.name, input_shape, body_input
        )));
    }

    Ok(RecursiveBlockSpec {
        loops,
        injection,
        body_scale,
        residual_scale,
        injection_scale,
        ouroboros,
        gate_bias,
        gamma_clip,
        gate_clip,
        lattice_group,
        lattice_threshold,
        lattice_temperature,
        lattice_mix,
        body,
    })
}

fn reject_section_keys(section: &RawSection, allowed: &[&str]) -> Result<()> {
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown or unsupported option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn validate_recursive_body_keys(section: &RawSection) -> Result<()> {
    let keys: &[&str] = match section.name.as_str() {
        "conv" | "convolutional" => &[
            "filters",
            "groups",
            "size",
            "stride",
            "stride_x",
            "stride_y",
            "dilation",
            "pad",
            "padding",
            "activation",
            "batch_normalize",
            "share_index",
        ],
        "max" | "maxpool" => &[
            "stride",
            "stride_x",
            "stride_y",
            "size",
            "padding",
            "out_channels",
            "maxpool_zero_nonmax",
            "zero_nonmax",
            "maxpool_depth",
        ],
        "avg" | "avgpool" => &[],
        "local_avg" | "local_avgpool" => &["stride", "stride_x", "stride_y", "size", "padding"],
        "upsample" => &["stride", "scale"],
        "channel_shuffle" => &["groups"],
        "conn" | "connected" => &["output", "batch_normalize", "activation"],
        "dropout" => &[
            "probability",
            "dropblock",
            "dropblock_size_rel",
            "dropblock_size_abs",
        ],
        _ => {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {}: unsupported layer kind in [recursive_block] body",
                section.name, section.line
            )));
        }
    };
    reject_section_keys(section, keys)?;
    if section.name == "dropout" {
        let probability = section.get_f32("probability", 0.2)?;
        if !(0.0..1.0).contains(&probability) {
            return Err(Error::InvalidConfig(format!(
                "[dropout] at line {}: probability must be in [0,1)",
                section.line
            )));
        }
    }
    Ok(())
}

fn parse_depth_anything_spec(
    section: &RawSection,
    current: TensorShape,
    cfg_parent: &Path,
) -> Result<DepthAnythingSpec> {
    let line = section.line;
    const ALLOWED_KEYS: &[&str] = &[
        "variant",
        "patch_size",
        "dim",
        "filters",
        "depth",
        "heads",
        "mlp_dim",
        "ffn_dim",
        "pos_grid",
        "pos_grid_h",
        "pos_grid_w",
        "ln_eps",
        "apply_imagenet_norm",
        "imagenet_norm",
        "taps",
        "mean",
        "std",
        "clamp",
        "weights",
        "freeze",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED_KEYS.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[depth_anything] at line {line}: unsupported key {key:?}"
        )));
    }
    let freeze = section.get_i64("freeze", 1)?;
    if !matches!(freeze, 0 | 1) {
        return Err(Error::InvalidConfig(format!(
            "[depth_anything] at line {line}: freeze must be 0 or 1, got {freeze}"
        )));
    }
    for key in ["stream", "wait_stream"] {
        if section.get_i64(key, -1)? != -1 {
            return Err(Error::InvalidConfig(format!(
                "[depth_anything] at line {line}: {key} is unsupported by Stage-1 CPU execution"
            )));
        }
    }
    if current.width == 0 || current.height == 0 || current.channels != 3 {
        return Err(Error::InvalidConfig(format!(
            "[depth_anything] at line {line} requires a non-empty NCHW RGB input, got {current}"
        )));
    }
    let variant = section.get_str("variant", "small");
    if !matches!(
        variant.as_str(),
        "small" | "vits" | "depth_anything_v2_small"
    ) {
        return Err(Error::InvalidConfig(format!(
            "[depth_anything] at line {line}: variant={variant} is unsupported; expected small, vits, or depth_anything_v2_small"
        )));
    }
    let dim = section.get_usize("dim", section.get_usize("filters", 384)?)?;
    let default_mlp_dim = dim.checked_mul(4).ok_or_else(|| {
        Error::InvalidConfig(format!("[depth_anything] at line {line}: dim*4 overflows"))
    })?;
    let mlp_dim = section.get_usize("mlp_dim", section.get_usize("ffn_dim", default_mlp_dim)?)?;
    let pos_grid = section.get_usize("pos_grid", 37)?;
    let mean = match section.get_f32_list("mean")? {
        values if values.is_empty() => [0.485, 0.456, 0.406],
        values if values.len() == 3 => [values[0], values[1], values[2]],
        values => {
            return Err(Error::InvalidConfig(format!(
                "[depth_anything] at line {line}: mean must contain exactly 3 values, got {}",
                values.len()
            )));
        }
    };
    let std = match section.get_f32_list("std")? {
        values if values.is_empty() => [0.229, 0.224, 0.225],
        values if values.len() == 3 => [values[0], values[1], values[2]],
        values => {
            return Err(Error::InvalidConfig(format!(
                "[depth_anything] at line {line}: std must contain exactly 3 values, got {}",
                values.len()
            )));
        }
    };
    let imagenet_norm = if section.get("apply_imagenet_norm").is_some() {
        section.get_bool_int("apply_imagenet_norm", true)?
    } else {
        section.get_bool_int("imagenet_norm", true)?
    };
    let taps = match section.get_usize_list("taps")? {
        values if values.is_empty() => vec![2, 5, 8, 11],
        values => values,
    };
    let mut spec = DepthAnythingSpec {
        patch_size: section.get_usize("patch_size", 14)?,
        dim,
        depth: section.get_usize("depth", 12)?,
        heads: section.get_usize("heads", 6)?,
        mlp_dim,
        pos_grid_h: section.get_usize("pos_grid_h", pos_grid)?,
        pos_grid_w: section.get_usize("pos_grid_w", pos_grid)?,
        ln_eps: section.get_f32("ln_eps", 1.0e-6)?,
        apply_imagenet_norm: imagenet_norm,
        imagenet_mean: mean,
        imagenet_std: std,
        taps,
        output_clamp: section.get_f32("clamp", 0.0)?,
        weights_path: None,
        external_weights: None,
    };
    let geometry = spec.geometry(current);
    geometry.validate().map_err(|error| {
        Error::InvalidConfig(format!("[depth_anything] at line {line}: {error}"))
    })?;

    let raw_weights_path = section.get_str("weights", "");
    if !raw_weights_path.is_empty() {
        let weights_path = resolve_depth_anything_weights_path(&raw_weights_path, cfg_parent)
            .map_err(|error| {
                Error::InvalidConfig(format!("[depth_anything] at line {line}: {error}"))
            })?;
        let bytes = fs::read(&weights_path).map_err(|error| {
            Error::InvalidConfig(format!(
                "[depth_anything] at line {line}: could not read weights={} ({error})",
                weights_path.display()
            ))
        })?;
        const HEADER_BYTES: usize = 20;
        let payload_bytes = geometry
            .weight_count()
            .and_then(|count| {
                count
                    .checked_mul(4)
                    .ok_or_else(|| "weight blob byte size overflows usize".to_owned())
            })
            .map_err(|error| {
                Error::InvalidConfig(format!("[depth_anything] at line {line}: {error}"))
            })?;
        let expected_bytes = HEADER_BYTES.checked_add(payload_bytes).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[depth_anything] at line {line}: external weight file size overflows"
            ))
        })?;
        if bytes.len() != expected_bytes {
            return Err(Error::InvalidConfig(format!(
                "[depth_anything] at line {line}: weights={} has {} bytes, expected exactly {expected_bytes} (20-byte header + {payload_bytes}-byte type-major blob)",
                weights_path.display(),
                bytes.len()
            )));
        }
        let blob = bytes[HEADER_BYTES..]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect::<Vec<_>>();
        spec.external_weights = Some(
            crate::depth_anything::DepthAnythingWeights::from_blob(&blob, &geometry).map_err(
                |error| Error::InvalidConfig(format!("[depth_anything] at line {line}: {error}")),
            )?,
        );
        spec.weights_path = Some(weights_path);
    }
    Ok(spec)
}

fn resolve_depth_anything_weights_path(raw: &str, cfg_parent: &Path) -> Result<PathBuf> {
    let requested = PathBuf::from(raw);
    if requested.exists() {
        return Ok(requested);
    }
    for base in [Some(cfg_parent), cfg_parent.parent()] {
        let Some(base) = base else {
            continue;
        };
        let candidate = base.join(&requested);
        if candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(Error::InvalidConfig(format!(
        "could not find weights={raw:?} in the working directory, cfg directory, or cfg parent"
    )))
}

fn parse_layer_kind(
    section: &RawSection,
    layer_index: usize,
    current: TensorShape,
    built_layers: &[Layer],
    cfg_parent: &Path,
    time_steps: usize,
    net_obb: bool,
) -> Result<LayerKind> {
    let name = section.name.as_str();
    Ok(match name {
        "depth_anything" | "depth_anything_v2" | "dinov2" | "dino" => {
            LayerKind::DepthAnything(parse_depth_anything_spec(section, current, cfg_parent)?)
        }
        "clifford" => LayerKind::Clifford(parse_clifford_spec(section, current)?),
        "foveation_film" | "hf_film" => LayerKind::FoveationFilm(parse_foveation_film_spec(
            section,
            layer_index,
            current,
            built_layers,
        )?),
        "hyperbolic_foveation" | "hf" => LayerKind::HyperbolicFoveation(
            parse_hyperbolic_foveation_spec(section, layer_index, current, built_layers)?,
        ),
        "graph_conv" | "graph_convolutional" => {
            LayerKind::GraphConv(parse_graph_conv_spec(section, current)?)
        }
        "dcnv4" | "dcnv4_convolutional" => LayerKind::Dcnv4(parse_dcnv4_spec(section, current)?),
        "deform_conv" | "deformable_convolutional" => {
            LayerKind::DeformConv(parse_deform_conv_spec(section, current)?)
        }
        "dsconv" | "dynamic_snake_conv" => LayerKind::DsConv(parse_dsconv_spec(section, current)?),
        "wmhf" | "wavelet_mamba_hf" | "wavelet_mamba_high_frequency" => {
            LayerKind::Wmhf(parse_wmhf_spec(section, current)?)
        }
        "eml_convolutional" => {
            let line = section.line;
            if current.width == 0 || current.height == 0 || current.channels == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[eml_convolutional] at line {line}: preceding layer must output a non-empty image"
                )));
            }
            let filters = section.get_usize("filters", 1)?;
            let size = section.get_usize("size", 1)?;
            let groups = section.get_i64("groups", 1)?.max(1) as usize;
            let stride = section.get_i64("stride", 1)?;
            let stride_x = section.get_i64("stride_x", -1)?;
            let stride_y = section.get_i64("stride_y", -1)?;
            let stride_x = if stride_x < 1 { stride } else { stride_x };
            let stride_y = if stride_y < 1 { stride } else { stride_y };
            let stride_x = usize::try_from(stride_x).map_err(|_| {
                Error::InvalidConfig(format!(
                    "[eml_convolutional] at line {line}: stride_x must be > 0"
                ))
            })?;
            let stride_y = usize::try_from(stride_y).map_err(|_| {
                Error::InvalidConfig(format!(
                    "[eml_convolutional] at line {line}: stride_y must be > 0"
                ))
            })?;
            let padding = if section.get_bool_int("pad", false)? {
                size / 2
            } else {
                section.get_usize("padding", 0)?
            };
            let branch = ConvolutionalSpec {
                filters,
                size,
                stride_x,
                stride_y,
                dilation: if size == 1 {
                    1
                } else {
                    section.get_usize("dilation", 1)?
                },
                padding,
                groups,
                share_index: None,
                batch_normalize: section.get_bool_int("batch_normalize", true)?,
                activation: Activation::Linear,
            };
            LayerKind::EmlConvolutional(EmlConvolutionalSpec {
                branch,
                eml_clamp: section.get_f32("eml_clamp", 4.0)?,
                eml_eps: section.get_f32("eml_eps", 1e-6)?,
                eml_scale: section.get_f32("eml_scale", 0.1)?,
                residual: section.get_bool_int("residual", true)?,
            })
        }
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
            if net_obb && section.get_i64("allow_obb_truth", 0)? == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[yolo] at line {} is incompatible with [net] obb=1; use [obb_yolo]",
                    section.line
                )));
            }
            if section.get_i64("obb", 0)? != 0 {
                return Err(Error::InvalidConfig(format!(
                    "[yolo] at line {} is axis-aligned and requires obb=0; use [obb_yolo]",
                    section.line
                )));
            }
            if section.get_usize("coords", 4)? != 4 {
                return Err(Error::InvalidConfig(format!(
                    "[yolo] at line {} requires coords=4; use [obb_yolo] for oriented boxes",
                    section.line
                )));
            }
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
        "obb_yolo" => {
            if !net_obb {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {} requires obb=1 in [net]",
                    section.line
                )));
            }
            if section.get_i64("obb", 1)? != 1 {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {} requires obb=1 or an omitted obb key",
                    section.line
                )));
            }
            if section.get_usize("coords", 6)? != 6 {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {} requires coords=6",
                    section.line
                )));
            }
            let line = section.line;
            let classes = section.get_usize("classes", 20)?;
            let max_delta = section.get_f32("max_delta", f32::MAX)?;
            let total = section.get_usize("num", 1)?;
            let max_boxes = section.get_usize("max", 200)?;
            let mut mask = section.get_usize_list("mask")?;
            if total == 0 || classes == 0 || max_boxes == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: num, classes, and max must be > 0"
                )));
            }
            if mask.is_empty() {
                mask.extend(0..total);
            }
            if mask.iter().any(|&index| index >= total) {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: mask indices must be less than num={total}"
                )));
            }
            let required_anchors = total.checked_mul(2).ok_or_else(|| {
                Error::InvalidConfig(format!("[obb_yolo] at line {line}: anchor count overflows"))
            })?;
            let required_directional_anchors = total.checked_mul(4).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: directional anchor count overflows"
                ))
            })?;
            let (anchors, front_anchor_offsets) = if section.get("obb_anchors").is_some() {
                if section.get("anchors").is_some() {
                    return Err(Error::InvalidConfig(format!(
                        "[obb_yolo] at line {line}: use either anchors or obb_anchors, not both"
                    )));
                }
                let configured = section.get_f32_list("obb_anchors")?;
                if configured.len() == required_anchors {
                    (configured, None)
                } else if configured.len() == required_directional_anchors {
                    let mut anchors = Vec::with_capacity(required_anchors);
                    let mut offsets = Vec::with_capacity(total);
                    for anchor in configured.chunks_exact(4) {
                        anchors.extend_from_slice(&anchor[..2]);
                        if !anchor[2].is_finite() || !anchor[3].is_finite() {
                            return Err(Error::InvalidConfig(format!(
                                "[obb_yolo] at line {line}: directional anchor offsets must be finite"
                            )));
                        }
                        offsets.push([anchor[2], anchor[3]]);
                    }
                    (anchors, Some(offsets))
                } else {
                    return Err(Error::InvalidConfig(format!(
                        "[obb_yolo] at line {line}: obb_anchors must contain {total} width/height pairs or quadruples"
                    )));
                }
            } else {
                (section.get_f32_list("anchors")?, None)
            };
            if anchors.len() < required_anchors
                || anchors[..required_anchors]
                    .iter()
                    .any(|&anchor| !anchor.is_finite() || anchor <= 0.0)
            {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: anchors must contain {total} positive finite width/height pairs"
                )));
            }
            let entries = classes.checked_add(7).ok_or_else(|| {
                Error::InvalidConfig(format!("[obb_yolo] at line {line}: entry count overflows"))
            })?;
            let channels = mask.len().checked_mul(entries).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: output channels overflow"
                ))
            })?;
            if current.channels != channels {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: preceding layer has {} channels; expected {channels} for {} anchors and {classes} classes",
                    current.channels,
                    mask.len()
                )));
            }
            let scale_x_y = section.get_f32("scale_x_y", 1.0)?;
            let front_point_normalizer = section.get_f32("front_point_normalizer", 0.35)?;
            let front_loss_normalizer =
                section.get_f32("front_loss_normalizer", front_point_normalizer)?;
            let (iou_loss, ciou_obb_loss, ciou_obb_multiplicative_loss) =
                parse_obb_iou_loss(section)?;
            let iou_thresh_kind = YoloIouLoss::parse(&section.get_str("iou_thresh_kind", "iou"))?;
            let nms_kind = ObbNmsKind::parse(&section.get_str("nms_kind", "obb"), line)?;
            let beta_nms = section.get_f32("beta_nms", 0.6)?;
            if !beta_nms.is_finite() {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: beta_nms must be finite"
                )));
            }
            if !scale_x_y.is_finite() {
                return Err(Error::InvalidConfig(format!(
                    "[obb_yolo] at line {line}: scale_x_y must be finite"
                )));
            }
            LayerKind::ObbYolo(ObbYoloSpec {
                max_boxes,
                front_anchor_offsets,
                front_loss_normalizer,
                covariance_normalizer: section.get_f32("covariance_normalizer", 0.0)?,
                cobb_normalizer: section.get_f32("cobb_normalizer", 0.0)?,
                obb_length_ratio_k: section.get_f32("obb_length_ratio_k", 1.0)?,
                obb_viou_weight: section.get_f32("obb_viou_weight", 0.0)?.clamp(0.0, 1.0),
                cobb_box_weight: section.get_f32("cobb_box_weight", 1.0)?,
                cobb_r_weight: section.get_f32("cobb_r_weight", 1.0)?,
                cobb_s_weight: section.get_f32("cobb_s_weight", 1.0)?,
                cobb_iou_lambda: section.get_f32("cobb_iou_lambda", 2.0)?,
                ciou_obb_loss,
                ciou_obb_multiplicative_loss,
                beta_nms,
                yolo: YoloSpec {
                    classes,
                    num: total,
                    mask,
                    anchors,
                    scale_x_y,
                    new_coords: section.get_i64("new_coords", 0)? != 0,
                    cls_normalizer: section.get_f32("cls_normalizer", 1.0)?,
                    obj_normalizer: section.get_f32("obj_normalizer", 1.0)?,
                    iou_normalizer: section.get_f32("iou_normalizer", 0.75)?,
                    ignore_thresh: section.get_f32("ignore_thresh", 0.5)?,
                    truth_thresh: section.get_f32("truth_thresh", 1.0)?,
                    iou_thresh: section.get_f32("iou_thresh", 1.0)?,
                    iou_loss,
                    iou_thresh_kind,
                    focal_loss: section.get_i64("focal_loss", 0)? != 0,
                    label_smooth_eps: section.get_f32("label_smooth_eps", 0.0)?,
                    objectness_smooth: section.get_i64("objectness_smooth", 0)? != 0,
                    max_delta: section.get_f32("max_delta", f32::MAX)?,
                    delta_normalizer: section.get_f32("delta_normalizer", 1.0)?,
                    class_multipliers: parse_class_multipliers(section, classes, max_delta)?,
                    class_map: parse_class_map(section, cfg_parent)?,
                },
                nms_kind,
            })
        }
        "obb_refine" => LayerKind::ObbRefine(parse_obb_refine_spec(
            section,
            layer_index,
            current,
            built_layers,
            net_obb,
        )?),
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
                return Err(Error::InvalidConfig(format!(
                    "[jev_ce] at line {line}: groups must be > 0"
                )));
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
                return Err(Error::InvalidConfig(format!(
                    "[jev_ce] at line {line}: label_smoothing must be in [0,1)"
                )));
            }
            let layout = section.get_str("layout", "group");
            let channel_major = match layout.as_str() {
                "channel" | "channel_major" | "cm" | "chw" => true,
                "group" | "group_major" | "gm" => section.get_usize("channel_major", 0)? != 0,
                other => {
                    return Err(Error::InvalidConfig(format!(
                        "[jev_ce] at line {line}: unknown layout '{other}' (expected group or channel)"
                    )));
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
            let loss_name = section
                .get_str("loss", expected.name())
                .trim()
                .to_ascii_lowercase();
            let requested = SegmentationLoss::from_name(&loss_name).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[segmentation] at line {line}: loss \"{loss_name}\" is not supported"
                ))
            })?;
            if requested != expected {
                let wanted = if classes == 1 {
                    "bce/logistic"
                } else {
                    "ce/softmax"
                };
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
        // `[omd_loss]`: C++ reads several keys under two spellings each
        // (primary first, alias otherwise), and the loss weight under three.
        "omd_loss" | "omd" => {
            let line = section.line;
            let alias_int = |primary: &str, alias: &str, fallback: usize| -> Result<usize> {
                if section.get(primary).is_some() {
                    section.get_usize(primary, fallback)
                } else {
                    section.get_usize(alias, fallback)
                }
            };
            let alias_float = |primary: &str, alias: &str, fallback: f32| -> Result<f32> {
                if section.get(primary).is_some() {
                    section.get_f32(primary, fallback)
                } else {
                    section.get_f32(alias, fallback)
                }
            };
            let cls_normalizer = if section.get("lambda").is_some() {
                section.get_f32("lambda", 0.03)?
            } else if section.get("normalizer").is_some() {
                section.get_f32("normalizer", 0.03)?
            } else {
                section.get_f32("cls_normalizer", 0.03)?
            };
            let max_samples = section.get_usize("max_samples", 128)?;
            if max_samples < 2 {
                return Err(Error::InvalidConfig(format!(
                    "[omd_loss] at line {line}: max_samples must be >= 2"
                )));
            }
            let margin = section.get_f32("margin", 0.5)?;
            if !(0.0..=2.0).contains(&margin) {
                return Err(Error::InvalidConfig(format!(
                    "[omd_loss] at line {line}: margin must be in [0,2]"
                )));
            }
            LayerKind::OmdLoss(OmdLossSpec {
                classes: section.get_usize("classes", 0)?,
                max_boxes: alias_int("max_boxes", "max", 200)?,
                truth_size: section.get_usize("truth_size", 6)?,
                cls_normalizer,
                margin,
                positive_weight: alias_float("positive_weight", "pos_weight", 1.0)?,
                negative_weight: alias_float("negative_weight", "neg_weight", 1.0)?,
                background_weight: alias_float("background_weight", "bg_weight", 1.0)?,
                small_threshold: alias_float("small_threshold", "small_threshold_px", 0.0)?,
                small_boost: section.get_f32("small_boost", 1.0)?,
                max_samples,
                background_samples: alias_int("background_samples", "bg_samples", 0)?,
                pool: section.get_usize("pool", 0)?,
                eps: section.get_f32("eps", 1e-6)?,
                max_delta: section.get_f32("max_delta", 10.0)?,
                show_details: section.get_usize("show_details", 0)? != 0,
            })
        }
        "dropout" => LayerKind::Dropout {
            probability: if section.get("probability").is_some() {
                Some(section.get_f32("probability", 0.2)?)
            } else {
                None
            },
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
        "centernet" => {
            let line = section.line;
            let classes = section.get_usize("classes", 20)?;
            if classes == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[centernet] at line {line}: classes must be > 0"
                )));
            }
            let max_boxes = section.get_usize("max", 200)?;
            if max_boxes == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[centernet] at line {line}: max must be > 0"
                )));
            }
            LayerKind::CenterNet(CenterNetSpec {
                classes,
                max_boxes,
                jitter: section.get_f32("jitter", 0.2)?,
                resize: section.get_f32("resize", 1.0)?,
                center_min_radius: section.get_isize("center_min_radius", 1)?.max(0) as i64,
                peak_nms: section.get_usize("peak_nms", 1)?,
                anisotropic_gaussian: section.get_usize("anisotropic_gaussian", 1)?,
                small_threshold: section.get_f32("small_threshold", 8.0)?,
                small_ref_size: section.get_f32("small_ref_size", 32.0)?,
                small_boost: section.get_f32("small_boost", 8.0)?,
                scale_min_px: section.get_f32("scale_min_px", 0.0)?,
                scale_max_px: section.get_f32("scale_max_px", f32::MAX)?,
                gaussian_iou: section.get_f32("gaussian_iou", 0.7)?,
                focal_alpha: section.get_f32("focal_alpha", 2.0)?,
                focal_beta: section.get_f32("focal_beta", 4.0)?,
                hm_normalizer: section.get_f32("hm_normalizer", 1.0)?,
                wh_normalizer: section.get_f32("wh_normalizer", 0.1)?,
                off_normalizer: section.get_f32("off_normalizer", 1.0)?,
                max_delta: section.get_f32("max_delta", 10.0)?,
                delta_normalizer: section.get_f32("delta_normalizer", 1.0)?,
            })
        }
        "yolox" | "ppyoloe" | "yolonas" => {
            let spec = parse_modern_yolo_spec(&section)?;
            match spec.kind {
                ModernYoloKind::Yolox => LayerKind::Yolox(spec),
                ModernYoloKind::Ppyoloe => LayerKind::Ppyoloe(spec),
                ModernYoloKind::Yolonas => LayerKind::Yolonas(spec),
            }
        }
        "swin" | "swin_block" | "transformer" | "transformer_block" => {
            LayerKind::Swin(parse_swin_spec(&section, current)?)
        }
        "tucker" | "tucker_attention" | "tucker_attn" => {
            LayerKind::TuckerAttention(parse_tucker_attention_spec(&section, current)?)
        }
        "mambavision" | "mambavision_block" => {
            LayerKind::MambaVision(parse_mambavision_spec(&section, current)?)
        }
        "kimi_kda" | "kimi_delta_attention" => {
            LayerKind::KimiKda(parse_kimi_kda_spec(&section, current)?)
        }
        "vit" => {
            let line = section.line;
            let has_filters = section.get("filters").is_some();
            let has_dim = section.get("dim").is_some();
            let filters = section.get_usize("filters", VIT_DIM_DEFAULT)?;
            let dim = section.get_usize("dim", filters)?;
            if has_filters && has_dim && filters != dim {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line} has conflicting filters={filters} and dim={dim}; use one embedding width"
                )));
            }
            let dim = if has_dim { dim } else { filters };
            let patch_size = section.get_usize("patch_size", 1)?;
            let patch_stride = section.get_usize("patch_stride", patch_size)?;
            let patch_pad = section.get_usize("patch_pad", 0)?;
            let heads = section.get_usize("heads", VIT_HEADS_DEFAULT)?;
            let has_ratio = section.get("ffn_ratio").is_some();
            let has_mlp = section.get("mlp_dim").is_some();
            let ffn_ratio = section.get_usize("ffn_ratio", 4)?;
            let ratio_mlp = dim.checked_mul(ffn_ratio).ok_or_else(|| {
                Error::InvalidConfig(format!("[vit] at line {line}: dim * ffn_ratio overflows"))
            })?;
            let mlp_dim = section.get_usize("mlp_dim", ratio_mlp)?;
            if has_ratio && has_mlp && mlp_dim != ratio_mlp {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line} has conflicting ffn_ratio={ffn_ratio} and mlp_dim={mlp_dim} for dim={dim}"
                )));
            }
            if dim == 0
                || heads == 0
                || dim % heads != 0
                || patch_size == 0
                || patch_stride == 0
                || mlp_dim == 0
            {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: dim={dim}, heads={heads}, patch_size={patch_size}, patch_stride={patch_stride}, mlp_dim={mlp_dim} invalid"
                )));
            }
            let twice_pad = patch_pad.checked_mul(2).ok_or_else(|| {
                Error::InvalidConfig(format!("[vit] at line {line}: patch padding overflows"))
            })?;
            let effective_h = current
                .height
                .checked_add(twice_pad)
                .and_then(|v| v.checked_sub(patch_size))
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[vit] at line {line}: input height is too small for the patch"
                    ))
                })?;
            let effective_w = current
                .width
                .checked_add(twice_pad)
                .and_then(|v| v.checked_sub(patch_size))
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[vit] at line {line}: input width is too small for the patch"
                    ))
                })?;
            if effective_h % patch_stride != 0 || effective_w % patch_stride != 0 {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: effective patch extents {effective_w}x{effective_h} must be divisible by patch_stride={patch_stride}"
                )));
            }
            let pos_embed = section.get_str("pos_embed", "learned").to_ascii_lowercase();
            let learned_pos_embed = match pos_embed.as_str() {
                "learned" | "absolute" => true,
                "sinusoidal" | "2d_sinusoidal" | "simple" => false,
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "[vit] at line {line}: pos_embed={pos_embed} must be learned or sinusoidal"
                    )));
                }
            };
            let pos_init = section.get_str("pos_init", "random").to_ascii_lowercase();
            let pos_init_zero = match pos_init.as_str() {
                "random" | "small_random" => false,
                "zero" | "zeros" => true,
                _ => {
                    return Err(Error::InvalidConfig(format!(
                        "[vit] at line {line}: pos_init={pos_init} must be random or zero"
                    )));
                }
            };
            let activation = Activation::parse(&section.get_str("activation", "gelu"));
            if let Activation::Other(ref other) = activation {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: unsupported activation={other}"
                )));
            }
            let msa_enabled = if section.get("msa").is_some() {
                section.get_bool_int("msa", false)?
            } else {
                section.get_bool_int("minimax_sparse_attention", false)?
            };
            let msa_block_size = if section.get("msa_block").is_some() {
                section.get_usize("msa_block", 16)?
            } else {
                section.get_usize("msa_block_size", 16)?
            }
            .max(1);
            let msa_topk = if section.get("msa_topk").is_some() {
                section.get_usize("msa_topk", 4)?
            } else {
                section.get_usize("msa_top_blocks", 4)?
            }
            .max(1);
            // Match the source alias precedence: flash_attention overrides flash_attn.
            let flash_attention = if section.get("flash_attention").is_some() {
                section.get_bool_int("flash_attention", false)?
            } else {
                section.get_bool_int("flash_attn", false)?
            };
            if flash_attention && msa_enabled {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: flash_attention and msa are mutually exclusive"
                )));
            }
            let vc_attention = if section.get("vc_attention").is_some() {
                section.get_bool_int("vc_attention", false)?
            } else {
                section.get_bool_int("vc", false)?
            };
            let vc_expcast_value = if section.get("vc_expcast").is_some() {
                section.get_usize("vc_expcast", 1)?
            } else {
                section.get_usize("expcast", 1)?
            };
            let vc_expcast = u8::try_from(vc_expcast_value)
                .ok()
                .filter(|mode| *mode <= 2)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[vit] at line {line}: vc_expcast must be 0, 1, or 2"
                    ))
                })?;
            let vc_perm = match section
                .get_str("vc_perm", "sequence")
                .to_ascii_lowercase()
                .as_str()
            {
                "sequence" => VitVcPermutation::Sequence,
                "spatial" | "window" | "spatial8" => VitVcPermutation::Spatial8,
                value => {
                    return Err(Error::InvalidConfig(format!(
                        "[vit] at line {line}: vc_perm={value} must be sequence, spatial, window, or spatial8"
                    )));
                }
            };
            if vc_attention && !flash_attention {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: vc_attention requires flash_attention"
                )));
            }
            LayerKind::Vit(VitSpec {
                dim,
                patch_size,
                patch_stride,
                patch_pad,
                heads,
                ffn_ratio,
                mlp_dim,
                learned_pos_embed,
                pos_init_zero,
                activation,
                msa_enabled,
                msa_block_size,
                msa_topk,
                flash_attention,
                vc_attention,
                vc_expcast,
                vc_perm,
            })
        }
        "laya_scorer" => {
            // C++ `parse_laya_scorer_section`: the layer is constructed for
            // `max_options x hidden` and checks the incoming shape against it,
            // so both come from the section (not from `[net]`).
            let line = section.line;
            let hidden = section.get_usize("hidden", LAYA_SCORER_HIDDEN)?;
            if hidden == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_scorer] at line {line}: hidden must be > 0"
                )));
            }
            let max_options = section.get_usize("max_options", current.width)?;
            if max_options == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_scorer] at line {line}: max_options must be > 0"
                )));
            }
            if current.channels != hidden {
                return Err(Error::InvalidConfig(format!(
                    "[laya_scorer] at line {line}: requires channels=hidden={hidden}, got {}",
                    current.channels
                )));
            }
            if current.width != max_options {
                return Err(Error::InvalidConfig(format!(
                    "[laya_scorer] at line {line}: requires width=max_options={max_options}, got {}",
                    current.width
                )));
            }
            LayerKind::LayaScorer(LayaScorerSpec {
                hidden,
                max_options,
                ln_eps: section.get_f32("ln_eps", crate::laya_scorer::LAYA_LAYERNORM_EPS)?,
            })
        }
        "laya_marker_gather" => {
            // C++ `parse_laya_marker_gather_section`: exactly two sources via
            // `layers=idxA,idxB` (route's negative-index convention), `hidden`
            // (default 1024) and `max_options` (default 4).  The shape checks
            // against input A/B live in `infer_shape`, where the sources are
            // known.
            let line = section.line;
            let layers = section.get_isize_list("layers")?;
            let layers: [isize; 2] = layers.try_into().map_err(|_| {
                Error::InvalidConfig(format!(
                    "[laya_marker_gather] at line {line} requires exactly 2 input layers via layers=idxA,idxB (hidden-state, marker-info)"
                ))
            })?;
            let hidden = section.get_usize("hidden", LAYA_MARKER_GATHER_HIDDEN)?;
            if hidden == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_marker_gather] at line {line}: hidden must be > 0"
                )));
            }
            let max_options = section.get_usize("max_options", 4)?;
            if max_options == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_marker_gather] at line {line}: max_options must be > 0"
                )));
            }
            LayerKind::LayaMarkerGather(LayaMarkerGatherSpec {
                layers,
                hidden,
                max_options,
            })
        }
        "laya_act_head" => {
            // C++ `parse_laya_act_head_section`: three route-style sources
            // `layers=head,scorer,marker`, plus hidden/max_options/fc_dim/n_act.
            // The cross-input shape checks live in `infer_shape`.
            let line = section.line;
            let layers = section.get_isize_list("layers")?;
            let layers: [isize; 3] = layers.try_into().map_err(|_| {
                Error::InvalidConfig(format!(
                    "[laya_act_head] at line {line} requires exactly 3 input layers via layers=head,scorer,marker"
                ))
            })?;
            let hidden = section.get_usize("hidden", LAYA_HEAD_HIDDEN)?;
            let max_options = section.get_usize("max_options", 4)?;
            let fc_dim = section.get_usize("fc_dim", crate::laya_act_head::LAYA_ACT_HEAD_FC_DIM)?;
            let n_act = section.get_usize("n_act", crate::laya_act_head::LAYA_ACT_HEAD_N_ACT)?;
            if hidden == 0 || max_options < 2 || fc_dim == 0 || n_act == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_act_head] at line {line}: hidden={hidden} max_options={max_options} (>=2) fc_dim={fc_dim} n_act={n_act} invalid"
                )));
            }
            LayerKind::LayaActHead(LayaActHeadSpec {
                layers,
                hidden,
                max_options,
                fc_dim,
                n_act,
            })
        }
        "laya_loss" => {
            // C++ `parse_laya_loss_section`: `max_options` defaults to the input
            // count, `mode` is `ce` (default) or `rlcd`.
            let line = section.line;
            let per_sample = current.width * current.height * current.channels;
            let max_options = section.get_usize("max_options", per_sample)?;
            let mode = section.get_str("mode", "ce");
            let rlcd = match mode.trim().to_ascii_lowercase().as_str() {
                "ce" => false,
                "rlcd" => true,
                other => {
                    return Err(Error::InvalidConfig(format!(
                        "[laya_loss] at line {line}: mode={other} must be ce or rlcd"
                    )));
                }
            };
            if per_sample != max_options {
                return Err(Error::InvalidConfig(format!(
                    "[laya_loss] at line {line}: input has {per_sample} values, expected max_options={max_options}"
                )));
            }
            LayerKind::LayaLoss(LayaLossSpec {
                max_options,
                rlcd,
                group: section.get_usize("group", 4)?,
                sigma_start: section.get_f32("sigma_start", 0.4)?,
                sigma_end: section.get_f32("sigma_end", 0.1)?,
                w_sph: section.get_f32("w_sph", 0.75)?,
                w_rps: section.get_f32("w_rps", 1.0)?,
                w_ce: section.get_f32("w_ce", 1.0)?,
                log_floor: section.get_f32("log_floor", -9.21)?,
                seed: section.get_usize("seed", 1234)? as u64,
                td_lambda: section.get_f32("td_lambda", -1.0)?,
            })
        }
        "laya_encoder" => {
            // C++ `parse_laya_encoder_section`: requires channels=2, height=1.
            let line = section.line;
            if current.channels != 2 || current.height != 1 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_encoder] at line {line} requires channels=2 and height=1 (token id + attention mask), got c={} h={}",
                    current.channels, current.height
                )));
            }
            let hidden = section.get_usize("hidden", LAYA_ENCODER_HIDDEN)?;
            let heads = section.get_usize("heads", LAYA_ENCODER_HEADS)?;
            let n_layers = section.get_usize("layers", LAYA_ENCODER_LAYERS)?;
            let intermediate = section.get_usize("intermediate", LAYA_ENCODER_INTERMEDIATE)?;
            let vocab = section.get_usize("vocab", LAYA_ENCODER_VOCAB)?;
            if hidden == 0 || heads == 0 || hidden % heads != 0 || (hidden / heads) % 2 != 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_encoder] at line {line}: hidden={hidden} must be divisible by heads={heads} with an even head_dim"
                )));
            }
            if n_layers == 0 || intermediate == 0 || vocab == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_encoder] at line {line}: layers={n_layers} intermediate={intermediate} vocab={vocab} must all be >= 1"
                )));
            }
            LayerKind::LayaEncoder(LayaEncoderSpec {
                hidden,
                heads,
                n_layers,
                intermediate,
                vocab,
                ln_eps: section.get_f32("ln_eps", LAYA_LAYERNORM_EPS)?,
                local_window: section.get_usize("local_window", LAYA_ENCODER_LOCAL_WINDOW)?,
                rope_theta_full: section
                    .get_f32("rope_theta_full", LAYA_ENCODER_ROPE_THETA_FULL)?,
                rope_theta_local: section
                    .get_f32("rope_theta_local", LAYA_ENCODER_ROPE_THETA_LOCAL)?,
            })
        }
        "laya_head" => {
            // C++ `parse_laya_head_section`: channels = hidden + mask (+ qtype
            // channel when qtype=-1); height=1.
            let line = section.line;
            if current.height != 1 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_head] at line {line} requires height=1, got h={}",
                    current.height
                )));
            }
            let hidden = section.get_usize("hidden", LAYA_HEAD_HIDDEN)?;
            let heads = section.get_usize("heads", LAYA_HEAD_HEADS)?;
            let n_layers = section.get_usize("layers", LAYA_HEAD_LAYERS)?;
            let ffn_dim = section.get_usize("ffn_dim", LAYA_HEAD_FFN_DIM)?;
            let qtype = section.get_isize("qtype", 0)?;
            if !(-1..=2).contains(&qtype) {
                return Err(Error::InvalidConfig(format!(
                    "[laya_head] at line {line}: qtype={qtype} must be in [0,2], or -1"
                )));
            }
            if hidden == 0 || heads == 0 || hidden % heads != 0 || n_layers == 0 || ffn_dim == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_head] at line {line}: hidden={hidden} heads={heads} layers={n_layers} ffn_dim={ffn_dim} invalid"
                )));
            }
            let expected_c = hidden + if qtype < 0 { 2 } else { 1 };
            if current.channels != expected_c {
                return Err(Error::InvalidConfig(format!(
                    "[laya_head] at line {line} requires channels={expected_c} (hidden + mask{}), got c={}",
                    if qtype < 0 { " + qtype" } else { "" },
                    current.channels
                )));
            }
            LayerKind::LayaHead(LayaHeadSpec {
                hidden,
                heads,
                n_layers,
                ffn_dim,
                ln_eps: section.get_f32("ln_eps", LAYA_LAYERNORM_EPS)?,
                qtype: qtype as i32,
            })
        }
        "contrastive" => LayerKind::Contrastive(ContrastiveSpec {
            classes: section.get_usize("classes", 1000)?,
            yolo_layer: section.get_isize("yolo_layer", 0)?,
            temperature: section.get_f32("temperature", 1.0)?,
            cls_normalizer: section.get_f32("cls_normalizer", 1.0)?,
            max_delta: section.get_f32("max_delta", f32::MAX)?,
            contrastive_neg_max: section.get_usize("contrastive_neg_max", 3)?,
        }),
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
        "moe_convolutional" => {
            // C++ `parse_moe_convolutional_section`.  The two hard errors it
            // raises are both about the router, and both must stay errors here:
            // a silently clamped `experts` or `top_k` would change the
            // parameter count the `.weights` file is read with.
            let line = section.line;
            if current.element_count() == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[moe_convolutional] at line {line}: the preceding layer must output an image"
                )));
            }
            let experts = section.get_usize("experts", 4)?;
            if experts < 1 {
                return Err(Error::InvalidConfig(format!(
                    "[moe_convolutional] at line {line}: experts must be >= 1, got {experts}"
                )));
            }
            let top_k = section.get_usize("top_k", 0)?;
            if top_k > experts {
                return Err(Error::InvalidConfig(format!(
                    "[moe_convolutional] at line {line}: top_k={top_k} is invalid for experts={experts}"
                )));
            }
            let size = section.get_usize("size", 1)?;
            let mut stride_x = section.get_usize("stride_x", 0)?;
            let mut stride_y = section.get_usize("stride_y", 0)?;
            if stride_x == 0 || stride_y == 0 {
                let stride = section.get_usize("stride", 1)?;
                if stride_x == 0 {
                    stride_x = stride;
                }
                if stride_y == 0 {
                    stride_y = stride;
                }
            }
            let padding = if section.get_i64("pad", 0)? != 0 {
                size / 2
            } else {
                section.get_usize("padding", 0)?
            };
            // `temperature` is the modern key; `gate_temperature` is the name
            // used by the shipped cfg and is what C++ falls back to.
            let temperature =
                section.get_f32("temperature", section.get_f32("gate_temperature", 1.0)?)?;
            LayerKind::Moe(MoeSpec {
                experts,
                top_k,
                load_balance: section.get_f32("load_balance", 0.01)?,
                temperature,
                expert_dropout: section.get_f32("expert_dropout", 0.0)?,
                debug_interval: section.get_usize("moe_debug", section.get_usize("debug", 0)?)?,
                expert: ConvolutionalSpec {
                    filters: section.get_usize("filters", 1)?,
                    size,
                    stride_x,
                    stride_y,
                    dilation: if size == 1 {
                        1
                    } else {
                        section.get_usize("dilation", 1)?
                    },
                    padding,
                    groups: section.get_usize("groups", 1)?.max(1),
                    share_index: None,
                    batch_normalize: section.get_i64("batch_normalize", 1)? != 0,
                    activation: Activation::parse(&section.get_str("activation", "leaky")),
                },
            })
        }
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
        LayerKind::RecursiveBlock(spec) => {
            let Some(last) = spec.body.last() else {
                return Err(Error::InvalidConfig(format!(
                    "[recursive_block] layer #{layer_index} requires a non-empty body"
                )));
            };
            let output = last.output_shape;
            if output.width != current.width
                || output.height != current.height
                || output.channels != current.channels
            {
                return Err(Error::InvalidConfig(format!(
                    "[recursive_block] layer #{layer_index}: body output {} must match input {}",
                    output, current
                )));
            }
            Ok(current)
        }
        LayerKind::FoveationFilm(_) => Ok(current),
        LayerKind::DepthAnything(spec) => {
            let geometry = spec.geometry(current);
            geometry.validate().map_err(|error| {
                Error::InvalidConfig(format!("[depth_anything] layer #{layer_index}: {error}"))
            })?;
            Ok(TensorShape::new(
                current.width / spec.patch_size,
                current.height / spec.patch_size,
                geometry.output_channels().map_err(Error::InvalidConfig)?,
                current.batch,
            ))
        }
        LayerKind::HyperbolicFoveation(spec) => match spec.warp_source {
            HfWarpSource::Previous => Ok(current),
            HfWarpSource::NetworkInput => Ok(built_layers
                .first()
                .map_or(current, |first| first.input_shape)),
        },
        LayerKind::Convolutional(spec) => Ok(TensorShape::new(
            conv_dim(current.width, spec.padding, spec.size, spec.stride_x),
            conv_dim(current.height, spec.padding, spec.size, spec.stride_y),
            spec.filters,
            current.batch,
        )),
        LayerKind::GraphConv(spec) => {
            let width = dcnv4_output_dim(
                current.width,
                spec.padding,
                spec.size,
                spec.stride_x,
                spec.dilation,
            )
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[graph_conv] layer #{layer_index}: invalid or empty output width"
                ))
            })?;
            let height = dcnv4_output_dim(
                current.height,
                spec.padding,
                spec.size,
                spec.stride_y,
                spec.dilation,
            )
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[graph_conv] layer #{layer_index}: invalid or empty output height"
                ))
            })?;
            Ok(TensorShape::new(width, height, spec.filters, current.batch))
        }
        LayerKind::Dcnv4(spec) => {
            if current.channels == 0 || spec.groups == 0 || current.channels % spec.groups != 0 {
                return Err(Error::InvalidConfig(format!(
                    "[dcnv4] layer #{layer_index}: groups must be positive and divide input channels"
                )));
            }
            let width = dcnv4_output_dim(
                current.width,
                spec.padding,
                spec.size,
                spec.stride_x,
                spec.dilation,
            )
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[dcnv4] layer #{layer_index}: invalid or empty output width"
                ))
            })?;
            let height = dcnv4_output_dim(
                current.height,
                spec.padding,
                spec.size,
                spec.stride_y,
                spec.dilation,
            )
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[dcnv4] layer #{layer_index}: invalid or empty output height"
                ))
            })?;
            Ok(TensorShape::new(
                width,
                height,
                current.channels,
                current.batch,
            ))
        }
        LayerKind::DsConv(spec) => {
            if current.width == 0 || current.height == 0 || current.channels == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[dsconv] layer #{layer_index}: preceding layer must output a non-empty image"
                )));
            }
            if spec.filters == 0 || spec.size == 0 || spec.padding != spec.size / 2 {
                return Err(Error::InvalidConfig(format!(
                    "[dsconv] layer #{layer_index}: filters/size must be positive with same padding"
                )));
            }
            Ok(TensorShape::new(
                current.width,
                current.height,
                spec.filters,
                current.batch,
            ))
        }
        LayerKind::Clifford(spec) => {
            if current.batch == 0
                || current.width == 0
                || current.height == 0
                || current.channels == 0
                || spec.filters != current.channels
            {
                return Err(Error::InvalidConfig(format!(
                    "[clifford] layer #{layer_index}: input must be a non-empty NCHW image and filters must equal input channels"
                )));
            }
            Ok(current)
        }
        LayerKind::Wmhf(spec) => {
            if current.width == 0
                || current.height == 0
                || current.channels == 0
                || current.batch == 0
            {
                return Err(Error::InvalidConfig(format!(
                    "[wmhf] layer #{layer_index}: preceding layer must output a non-empty image"
                )));
            }
            let split_sum = spec
                .identity_channels
                .checked_add(spec.local_channels)
                .and_then(|count| count.checked_add(spec.global_channels));
            if spec.filters < 4
                || spec.local_channels == 0
                || spec.global_channels == 0
                || split_sum != Some(spec.filters)
                || !spec.freq_scale.is_finite()
            {
                return Err(Error::InvalidConfig(format!(
                    "[wmhf] layer #{layer_index}: invalid channel split or frequency scale"
                )));
            }
            current
                .channels
                .checked_mul(spec.filters)
                .and_then(|_| current.width.checked_mul(current.height))
                .and_then(|area| area.checked_mul(current.batch))
                .and_then(|count| count.checked_mul(spec.filters))
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[wmhf] layer #{layer_index}: output element count overflows"
                    ))
                })?;
            Ok(TensorShape::new(
                current.width,
                current.height,
                spec.filters,
                current.batch,
            ))
        }
        LayerKind::DeformConv(spec) => {
            if current.width == 0 || current.height == 0 || current.channels == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[deform_conv] layer #{layer_index}: preceding layer must output a non-empty image"
                )));
            }
            if spec.filters == 0
                || spec.groups == 0
                || current.channels % spec.groups != 0
                || spec.filters % spec.groups != 0
            {
                return Err(Error::InvalidConfig(format!(
                    "[deform_conv] layer #{layer_index}: groups must be positive and divide input and output channels"
                )));
            }
            let width = deform_conv_output_dim(
                current.width,
                spec.padding,
                spec.size,
                spec.stride_x,
                spec.dilation,
            )
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[deform_conv] layer #{layer_index}: invalid or empty output width"
                ))
            })?;
            let height = deform_conv_output_dim(
                current.height,
                spec.padding,
                spec.size,
                spec.stride_y,
                spec.dilation,
            )
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[deform_conv] layer #{layer_index}: invalid or empty output height"
                ))
            })?;
            Ok(TensorShape::new(width, height, spec.filters, current.batch))
        }
        LayerKind::EmlConvolutional(spec) => {
            let conv = &spec.branch;
            if conv.filters == 0
                || conv.size == 0
                || conv.stride_x == 0
                || conv.stride_y == 0
                || conv.dilation == 0
                || conv.groups == 0
                || current.channels % conv.groups != 0
                || conv.filters % conv.groups != 0
            {
                return Err(Error::InvalidConfig(format!(
                    "[eml_convolutional] layer #{layer_index}: filters, size, strides, dilation, groups must be valid and groups must divide input channels and filters"
                )));
            }
            let output = TensorShape::new(
                conv_dim(current.width, conv.padding, conv.size, conv.stride_x),
                conv_dim(current.height, conv.padding, conv.size, conv.stride_y),
                conv.filters,
                current.batch,
            );
            if output.width == 0 || output.height == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[eml_convolutional] layer #{layer_index}: convolution produces an empty spatial shape"
                )));
            }
            Ok(output)
        }
        LayerKind::Deconvolutional(spec) => {
            if spec.filters == 0
                || spec.size == 0
                || spec.stride == 0
                || current.width == 0
                || current.height == 0
            {
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
        LayerKind::Yolo(_)
        | LayerKind::ObbYolo(_)
        | LayerKind::GaussianYolo(_)
        | LayerKind::Region(_) => Ok(current),
        LayerKind::ObbRefine(spec) => Ok(TensorShape::new(
            spec.proposal_capacity,
            1,
            5,
            current.batch,
        )),
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
        | LayerKind::OmdLoss(_)
        | LayerKind::Dropout { .. }
        | LayerKind::Contrastive(_)
        | LayerKind::LayaLoss(_) => Ok(current),
        // `[laya_encoder]` turns `[batch, 2, 1, T]` into `[batch, hidden, 1, T]`.
        LayerKind::LayaEncoder(spec) => {
            if current.channels != 2 || current.height != 1 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_encoder] layer #{layer_index} requires channels=2 and height=1, got c={} h={}",
                    current.channels, current.height
                )));
            }
            if current.width == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_encoder] layer #{layer_index} requires seq_len > 0"
                )));
            }
            Ok(TensorShape::new(
                current.width,
                1,
                spec.hidden,
                current.batch,
            ))
        }
        // `[laya_head]` is a pre-norm transformer over `[batch, hidden+mask, 1, T]`.
        LayerKind::LayaHead(spec) => {
            let expected_c = spec.hidden + if spec.qtype < 0 { 2 } else { 1 };
            if current.channels != expected_c || current.height != 1 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_head] layer #{layer_index} requires channels={expected_c} and height=1, got c={} h={}",
                    current.channels, current.height
                )));
            }
            if current.width == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_head] layer #{layer_index} requires seq_len > 0"
                )));
            }
            Ok(TensorShape::new(
                current.width,
                1,
                spec.hidden,
                current.batch,
            ))
        }
        LayerKind::Swin(spec) => {
            if current.width == 0 || current.height == 0 || current.channels == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[swin] layer #{layer_index} requires non-empty image-shaped input"
                )));
            }
            Ok(TensorShape::new(
                current.width,
                current.height,
                spec.filters,
                current.batch,
            ))
        }
        LayerKind::TuckerAttention(spec) => {
            if current.width == 0 || current.height == 0 || current.channels == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[tucker_attention] layer #{layer_index} requires non-empty image-shaped input"
                )));
            }
            if spec.filters != current.channels {
                return Err(Error::InvalidConfig(format!(
                    "[tucker_attention] layer #{layer_index} requires filters={} to preserve residual channels, got {}",
                    current.channels, spec.filters
                )));
            }
            Ok(current)
        }
        LayerKind::MambaVision(spec) => {
            if current.width == 0 || current.height == 0 || current.channels == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[mambavision] layer #{layer_index} requires non-empty image-shaped input"
                )));
            }
            Ok(TensorShape::new(
                current.width,
                current.height,
                spec.filters,
                current.batch,
            ))
        }
        LayerKind::KimiKda(spec) => {
            if current.width == 0 || current.height == 0 || current.channels == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[kimi_kda] layer #{layer_index} requires non-empty image-shaped input"
                )));
            }
            if spec.filters != current.channels {
                return Err(Error::InvalidConfig(format!(
                    "[kimi_kda] layer #{layer_index} requires filters={} to preserve residual channels, got {}",
                    current.channels, spec.filters
                )));
            }
            crate::kimi_kda::validate_geometry(current, spec).map_err(|error| {
                Error::InvalidConfig(format!("[kimi_kda] layer #{layer_index}: {error}"))
            })?;
            Ok(current)
        }
        LayerKind::Vit(spec) => {
            let line = layer_index;
            let twice_pad = spec.patch_pad.checked_mul(2).ok_or_else(|| {
                Error::InvalidConfig(format!("[vit] layer #{line}: patch padding overflows"))
            })?;
            let eff_h = current
                .height
                .checked_add(twice_pad)
                .and_then(|v| v.checked_sub(spec.patch_size))
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[vit] layer #{line}: invalid patch height geometry"
                    ))
                })?;
            let eff_w = current
                .width
                .checked_add(twice_pad)
                .and_then(|v| v.checked_sub(spec.patch_size))
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "[vit] layer #{line}: invalid patch width geometry"
                    ))
                })?;
            if spec.patch_stride == 0
                || eff_h % spec.patch_stride != 0
                || eff_w % spec.patch_stride != 0
            {
                return Err(Error::InvalidConfig(format!(
                    "[vit] layer #{line}: patch geometry is not divisible by patch_stride={}",
                    spec.patch_stride
                )));
            }
            Ok(TensorShape::new(
                eff_w / spec.patch_stride + 1,
                eff_h / spec.patch_stride + 1,
                spec.dim,
                current.batch,
            ))
        }
        LayerKind::Yolox(_) | LayerKind::Ppyoloe(_) | LayerKind::Yolonas(_) => Ok(current),
        LayerKind::CenterNet(_) => Ok(current),
        // `[laya_scorer]` turns `[batch, hidden, 1, max_options]` into
        // `[batch, 1, 1, max_options]`.
        LayerKind::LayaScorer(spec) => Ok(TensorShape::new(spec.max_options, 1, 1, current.batch)),
        // `[laya_marker_gather]` gathers one hidden-state column per option, so
        // it needs both source shapes (route-style).
        LayerKind::LayaMarkerGather(spec) => {
            let a_index = resolve_relative_layer(layer_index, spec.layers[0])?;
            let b_index = resolve_relative_layer(layer_index, spec.layers[1])?;
            let a = built_layers
                .get(a_index)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: spec.layers[0],
                })?
                .output_shape;
            let b = built_layers
                .get(b_index)
                .ok_or(Error::InvalidRoute {
                    layer_index,
                    reference: spec.layers[1],
                })?
                .output_shape;
            if a.channels != spec.hidden {
                return Err(Error::InvalidConfig(format!(
                    "[laya_marker_gather] layer #{layer_index}: input A (layer #{a_index}) has {} channels, expected hidden={}",
                    a.channels, spec.hidden
                )));
            }
            if b.channels < 2 * spec.max_options {
                return Err(Error::InvalidConfig(format!(
                    "[laya_marker_gather] layer #{layer_index}: input B (layer #{b_index}) has {} channels, needs >= 2*max_options={}",
                    b.channels,
                    2 * spec.max_options
                )));
            }
            if a.width != b.width {
                return Err(Error::InvalidConfig(format!(
                    "[laya_marker_gather] layer #{layer_index}: input A/B sequence lengths differ ({} vs {})",
                    a.width, b.width
                )));
            }
            if a.width == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_marker_gather] layer #{layer_index}: input A sequence length must be > 0"
                )));
            }
            Ok(TensorShape::new(
                spec.max_options,
                1,
                spec.hidden,
                current.batch,
            ))
        }
        // `[laya_act_head]` reads three source shapes and emits `n_act` logits.
        LayerKind::LayaActHead(spec) => {
            let head_index = resolve_relative_layer(layer_index, spec.layers[0])?;
            let scorer_index = resolve_relative_layer(layer_index, spec.layers[1])?;
            let marker_index = resolve_relative_layer(layer_index, spec.layers[2])?;
            let lookup = |index: usize, raw: isize| -> Result<TensorShape> {
                Ok(built_layers
                    .get(index)
                    .ok_or(Error::InvalidRoute {
                        layer_index,
                        reference: raw,
                    })?
                    .output_shape)
            };
            let head = lookup(head_index, spec.layers[0])?;
            let scorer = lookup(scorer_index, spec.layers[1])?;
            let marker = lookup(marker_index, spec.layers[2])?;
            if head.channels != spec.hidden {
                return Err(Error::InvalidConfig(format!(
                    "[laya_act_head] layer #{layer_index}: head input (layer #{head_index}) has {} channels, expected hidden={}",
                    head.channels, spec.hidden
                )));
            }
            if scorer.channels != 1 || scorer.width != spec.max_options {
                return Err(Error::InvalidConfig(format!(
                    "[laya_act_head] layer #{layer_index}: scorer input (layer #{scorer_index}) must be 1 x {} logits, got c={} w={}",
                    spec.max_options, scorer.channels, scorer.width
                )));
            }
            if marker.channels != 2 * spec.max_options {
                return Err(Error::InvalidConfig(format!(
                    "[laya_act_head] layer #{layer_index}: marker input (layer #{marker_index}) has {} channels, expected 2*max_options={}",
                    marker.channels,
                    2 * spec.max_options
                )));
            }
            if head.width == 0 || marker.width == 0 {
                return Err(Error::InvalidConfig(format!(
                    "[laya_act_head] layer #{layer_index}: head/marker sequence lengths must be > 0"
                )));
            }
            Ok(TensorShape::new(1, 1, spec.n_act, current.batch))
        }
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
        // `[moe_convolutional]`'s output shape is exactly its experts': the
        // blend happens per image, so the spatial size is unchanged.
        LayerKind::Moe(spec) => Ok(TensorShape::new(
            conv_dim(
                current.width,
                spec.expert.padding,
                spec.expert.size,
                spec.expert.stride_x,
            ),
            conv_dim(
                current.height,
                spec.expert.padding,
                spec.expert.size,
                spec.expert.stride_y,
            ),
            spec.expert.filters,
            current.batch,
        )),
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
            Error::InvalidConfig(format!(
                "[{}] at line {}: invalid {key}",
                section.name, section.line
            ))
        })
}

fn resolve_relative_layer(layer_index: usize, raw_index: isize) -> Result<usize> {
    let absolute = if raw_index < 0 {
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

fn parse_obb_refine_spec(
    section: &RawSection,
    layer_index: usize,
    current: TensorShape,
    built_layers: &[Layer],
    net_obb: bool,
) -> Result<ObbRefineSpec> {
    let line = section.line;
    let invalid =
        |message: &str| Error::InvalidConfig(format!("[obb_refine] at line {line}: {message}"));
    if !net_obb {
        return Err(invalid("requires obb=1 in [net]"));
    }

    let feature_ref = section.get_isize("from", -2)?;
    let feature_layer = resolve_relative_layer(layer_index, feature_ref)
        .map_err(|_| invalid(&format!("invalid from={feature_ref}")))?;
    let feature = built_layers
        .get(feature_layer)
        .ok_or_else(|| invalid("feature source does not exist"))?;
    let feature_shape = feature.output_shape;
    if feature_shape.width == 0 || feature_shape.height == 0 || feature_shape.channels == 0 {
        return Err(invalid(
            "from= must reference a non-empty spatial feature layer",
        ));
    }

    let proposal_ref = section.get_isize("proposal_from", -1)?;
    let proposal_layer = if proposal_ref == -1 {
        built_layers
            .iter()
            .rposition(|layer| matches!(layer.kind, LayerKind::ObbYolo(_)))
            .ok_or_else(|| invalid("requires a preceding [obb_yolo] proposal layer"))?
    } else {
        resolve_relative_layer(layer_index, proposal_ref)
            .map_err(|_| invalid(&format!("invalid proposal_from={proposal_ref}")))?
    };
    let proposal = built_layers
        .get(proposal_layer)
        .ok_or_else(|| invalid("proposal source does not exist"))?;
    let LayerKind::ObbYolo(proposal_spec) = &proposal.kind else {
        return Err(invalid(
            "proposal_from= must reference a preceding [obb_yolo] layer",
        ));
    };

    let classes = section.get_usize("classes", proposal_spec.classes)?;
    let max_boxes = section.get_usize("max", proposal_spec.max_boxes)?;
    if classes == 0 || max_boxes == 0 {
        return Err(invalid("classes and max must be > 0"));
    }
    if classes != proposal_spec.classes {
        return Err(invalid(&format!(
            "classes={classes} does not match proposal classes={}",
            proposal_spec.classes
        )));
    }

    let proposal_topk_train = section.get_isize("proposal_topk_train", 128)?.max(0) as usize;
    let proposal_topk_infer = section.get_isize("proposal_topk_infer", 300)?.max(0) as usize;
    let proposal_jitter = section.get_isize("proposal_jitter", 1)?.max(0) as usize;
    let sample_grid = section.get_usize("sample_grid", 3)?.max(1);
    let train_copies = proposal_jitter
        .checked_add(1)
        .ok_or_else(|| invalid("proposal_jitter overflows capacity"))?;
    let minimum_train_capacity = current
        .batch
        .checked_mul(max_boxes)
        .and_then(|capacity| capacity.checked_mul(train_copies))
        .ok_or_else(|| invalid("default proposal_capacity overflows"))?;
    let default_capacity = proposal_topk_train
        .max(proposal_topk_infer)
        .max(minimum_train_capacity);
    let proposal_capacity = section.get_usize("proposal_capacity", default_capacity)?;
    if proposal_capacity == 0 {
        return Err(invalid("proposal_capacity must be > 0"));
    }
    proposal_capacity
        .checked_mul(5)
        .ok_or_else(|| invalid("proposal output size overflows"))?;
    let inputs = sample_grid
        .checked_mul(sample_grid)
        .and_then(|samples| samples.checked_mul(feature_shape.channels))
        .ok_or_else(|| invalid("sample_grid feature count overflows"))?;
    let _parameter_count = inputs
        .checked_mul(5)
        .and_then(|weights| weights.checked_add(5))
        .ok_or_else(|| invalid("refine parameter count overflows"))?;

    let front_range = section.get_f32("front_range", 1.0)?;
    if !front_range.is_finite() || front_range <= 0.0 {
        return Err(invalid("front_range must be finite and > 0"));
    }
    let proposal_thresh = section.get_f32("proposal_thresh", 0.01)?;
    let front_loss_normalizer = section.get_f32("front_loss_normalizer", 1.0)?;
    let quality_normalizer = section.get_f32("quality_normalizer", 0.0)?;
    if !proposal_thresh.is_finite()
        || !front_loss_normalizer.is_finite()
        || !quality_normalizer.is_finite()
    {
        return Err(invalid("proposal and normalization values must be finite"));
    }

    let representation = section.get_str("representation", "carpn");
    if !representation.is_empty() && representation != "carpn" {
        return Err(invalid("only representation=carpn is supported"));
    }
    let loss = match section
        .get_str("loss", "ciou_obb")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "carpn" => ObbRefineLoss::Carpn,
        "ciou_obb" | "" => ObbRefineLoss::CiouObb,
        other => {
            return Err(invalid(&format!(
                "unsupported loss={other}; expected carpn or ciou_obb"
            )));
        }
    };
    let csda_topk = section.get_usize("csda_topk", 9)?.max(1);
    let configured_scales = section.get_f32_list("anchor_scales")?;
    let anchor_scales = if configured_scales.is_empty() {
        vec![1.0, 2.0, 4.0]
    } else {
        configured_scales
            .into_iter()
            .take(3)
            .map(|scale| if scale > 0.0 { scale } else { 1.0 })
            .collect()
    };

    Ok(ObbRefineSpec {
        feature_layer,
        proposal_layer,
        classes,
        max_boxes,
        proposal_topk_train,
        proposal_topk_infer,
        proposal_jitter,
        sample_grid,
        proposal_capacity,
        inputs,
        front_range,
        front_loss_normalizer,
        quality_normalizer,
        proposal_thresh,
        csda_topk,
        anchor_scales,
        loss,
        nms_kind: proposal_spec.nms_kind,
        beta_nms: proposal_spec.beta_nms,
    })
}

fn parse_obb_iou_loss(section: &RawSection) -> Result<(YoloIouLoss, bool, bool)> {
    let loss = section.get_str("iou_loss", "ciou_obb");
    match loss.trim().to_ascii_lowercase().as_str() {
        "ciou_obb" => Ok((YoloIouLoss::Ciou, true, false)),
        "ciou_obb_mul" => Ok((YoloIouLoss::Ciou, true, true)),
        _ => Ok((YoloIouLoss::parse(&loss)?, false, false)),
    }
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

/// C++ `LAYA_SCORER_HIDDEN`, the default `hidden` of `[laya_scorer]`.
pub const LAYA_SCORER_HIDDEN: usize = 1024;

/// C++ `LAYA_MARKER_GATHER_HIDDEN`, the default `hidden` of
/// `[laya_marker_gather]`.
pub const LAYA_MARKER_GATHER_HIDDEN: usize = 1024;

/// C++ `LAYA_HEAD_HIDDEN`, the default `hidden` of `[laya_act_head]` (and the
/// LAYA head's own hidden width).
pub const LAYA_HEAD_HIDDEN: usize = 1024;

/// C++ `LAYA_ENCODER_*` defaults.
pub const LAYA_ENCODER_HIDDEN: usize = 1024;
pub const LAYA_ENCODER_HEADS: usize = 16;
pub const LAYA_ENCODER_LAYERS: usize = 28;
pub const LAYA_ENCODER_INTERMEDIATE: usize = 2624;
pub const LAYA_ENCODER_VOCAB: usize = 50368;
pub const LAYA_ENCODER_LOCAL_WINDOW: usize = 128;
pub const LAYA_ENCODER_ROPE_THETA_FULL: f32 = 160000.0;
pub const LAYA_ENCODER_ROPE_THETA_LOCAL: f32 = 10000.0;
pub const LAYA_HEAD_HEADS: usize = 16;
pub const LAYA_HEAD_LAYERS: usize = 2;
pub const LAYA_HEAD_FFN_DIM: usize = 4096;
pub const VIT_DIM_DEFAULT: usize = 128;
pub const VIT_HEADS_DEFAULT: usize = 4;
pub const LAYA_LAYERNORM_EPS: f32 = 1.0e-5;

fn reject_unknown_dcnv4_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "filters",
        "groups",
        "size",
        "stride",
        "stride_x",
        "stride_y",
        "dilation",
        "pad",
        "padding",
        "activation",
        "batch_normalize",
        "offset_scale",
        "remove_center",
        "d_stride",
        "block_thread",
        "softmax",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

pub(crate) fn dcnv4_parameter_layout(
    channels: usize,
    spec: &Dcnv4Spec,
) -> Option<(usize, usize, usize)> {
    let area = spec.size.checked_mul(spec.size)?;
    let points = area.checked_sub(usize::from(spec.remove_center))?;
    let raw_offset_dim = spec.groups.checked_mul(points)?.checked_mul(3)?;
    let offset_dim = raw_offset_dim
        .checked_add(7)?
        .checked_div(8)?
        .checked_mul(8)?;
    let dummy_weights = channels
        .checked_div(spec.groups)?
        .checked_mul(spec.filters)?
        .checked_mul(area)?;
    let offset_weights = offset_dim.checked_mul(channels)?.checked_mul(area)?;
    Some((dummy_weights, offset_weights, offset_dim))
}

fn dcnv4_output_dim(
    input: usize,
    padding: usize,
    size: usize,
    stride: usize,
    dilation: usize,
) -> Option<usize> {
    if stride == 0 || size == 0 || dilation == 0 {
        return None;
    }
    let padded = input.checked_add(padding.checked_mul(2)?)?;
    let kernel_extent = dilation.checked_mul(size.checked_sub(1)?)?;
    padded
        .checked_sub(kernel_extent)?
        .checked_sub(1)?
        .checked_div(stride)?
        .checked_add(1)
}

fn deform_conv_output_dim(
    input: usize,
    padding: usize,
    size: usize,
    stride: usize,
    dilation: usize,
) -> Option<usize> {
    dcnv4_output_dim(input, padding, size, stride, dilation)
}

fn parse_foveation_film_spec(
    section: &RawSection,
    layer_index: usize,
    input: TensorShape,
    built_layers: &[Layer],
) -> Result<FoveationFilmSpec> {
    let line = section.line;
    let invalid = |message: &str| {
        Error::InvalidConfig(format!("[{}] at line {line}: {message}", section.name))
    };
    reject_section_keys(
        section,
        &[
            "hf_layer",
            "embed_dim",
            "backprop_to_hf",
            "film_init_scale",
            "learning_rate",
            "dont_update",
            "burnin_update",
            "train_only_bn",
            "clip",
            "grad_centr",
            "batch_normalize",
            "cbn",
            "stream",
            "wait_stream",
        ],
    )?;
    if input.batch == 0 || input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid(
            "preceding layer must output a non-empty NCHW tensor",
        ));
    }

    let hf_layer_raw = section.get_isize("hf_layer", -1)?;
    let hf_layer = if hf_layer_raw == -1 {
        built_layers
            .iter()
            .rposition(|layer| matches!(layer.kind, LayerKind::HyperbolicFoveation(_)))
            .ok_or_else(|| invalid("requires a preceding [hyperbolic_foveation] layer"))?
    } else {
        let resolved = if hf_layer_raw < -1 {
            layer_index.checked_add_signed(hf_layer_raw)
        } else {
            usize::try_from(hf_layer_raw).ok()
        }
        .ok_or_else(|| invalid(&format!("invalid hf_layer={hf_layer_raw}")))?;
        if resolved >= layer_index || resolved >= built_layers.len() {
            return Err(invalid("hf_layer must reference an earlier layer"));
        }
        resolved
    };
    let selected_hf = built_layers
        .get(hf_layer)
        .ok_or_else(|| invalid("selected hf_layer does not exist"))?;
    let param_count = match &selected_hf.kind {
        LayerKind::HyperbolicFoveation(spec) => spec.param_count(),
        _ => return Err(invalid("hf_layer must reference [hyperbolic_foveation]")),
    };

    let embed_dim_raw = section.get_i64("embed_dim", 256)?;
    let embed_dim =
        usize::try_from(embed_dim_raw.max(1)).map_err(|_| invalid("embed_dim is out of range"))?;
    let backprop_to_hf = section.get_i64("backprop_to_hf", 1)? != 0;
    let film_init_scale = section.get_f32("film_init_scale", 0.0)?;
    if !film_init_scale.is_finite() {
        return Err(invalid("film_init_scale must be finite"));
    }
    let learning_rate = section.get_f32("learning_rate", 1.0)?;
    if !learning_rate.is_finite() || learning_rate < 0.0 {
        return Err(invalid("learning_rate must be finite and non-negative"));
    }
    if section.get_i64("batch_normalize", 0)? != 0 || section.get_i64("cbn", 0)? != 0 {
        return Err(invalid("batch normalization is unsupported"));
    }
    if section.get_f32("clip", 0.0)? != 0.0 || section.get_i64("grad_centr", 0)? != 0 {
        return Err(invalid("clip and grad_centr are unsupported"));
    }
    if section.get_i64("stream", -1)? != -1 || section.get_i64("wait_stream", -1)? != -1 {
        return Err(invalid(
            "GPU stream options are unsupported in Stage-1 CPU support",
        ));
    }

    embed_dim
        .checked_mul(param_count)
        .and_then(|value| value.checked_add(embed_dim))
        .and_then(|value| {
            input
                .channels
                .checked_mul(embed_dim)
                .and_then(|c| c.checked_mul(2))
                .and_then(|c| value.checked_add(c))
        })
        .and_then(|value| {
            input
                .channels
                .checked_mul(2)
                .and_then(|c| value.checked_add(c))
        })
        .ok_or_else(|| invalid("parameter count overflows"))?;

    Ok(FoveationFilmSpec {
        hf_layer,
        param_count,
        embed_dim,
        backprop_to_hf,
        film_init_scale,
    })
}

fn reject_unknown_hyperbolic_foveation_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "init_cx",
        "init_cy",
        "init_radius",
        "init_radius_x",
        "init_radius_y",
        "init_strength",
        "small_box_px",
        "small_box_strength",
        "strength_loss",
        "tiny_center_loss",
        "tiny_radius_loss",
        "tiny_strength_loss",
        "strength_zero_loss",
        "hf_lr_mult",
        "alpha",
        "p_exp",
        "downsample",
        "params",
        "pred_hidden",
        "nr_iters",
        "nr_eta",
        "inverse_method",
        "warp_source",
        "radius_mode",
        "pred_source_layer",
        "debug",
        "debug_interval",
        "batch_normalize",
        "cbn",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn parse_hyperbolic_foveation_spec(
    section: &RawSection,
    layer_index: usize,
    input: TensorShape,
    built_layers: &[Layer],
) -> Result<HyperbolicFoveationSpec> {
    let line = section.line;
    let invalid = |message: &str| {
        Error::InvalidConfig(format!("[{}] at line {line}: {message}", section.name))
    };
    reject_unknown_hyperbolic_foveation_keys(section)?;
    if input.batch == 0 || input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid(
            "preceding layer must output a non-empty NCHW tensor",
        ));
    }

    let init_cx = section.get_f32("init_cx", 0.0)?;
    let init_cy = section.get_f32("init_cy", 0.0)?;
    let init_radius = section.get_f32("init_radius", 0.75)?;
    let init_radius_x = section.get_f32("init_radius_x", init_radius)?;
    let init_radius_y = section.get_f32("init_radius_y", init_radius)?;
    let init_strength = section.get_f32("init_strength", 0.05)?;
    let small_box_px = section.get_f32("small_box_px", 14.0)?;
    let small_box_strength = section.get_f32("small_box_strength", 0.85)?;
    let strength_loss = section.get_f32("strength_loss", 1.0)?;
    let tiny_center_loss = section.get_f32("tiny_center_loss", strength_loss)?;
    let tiny_radius_loss = section.get_f32("tiny_radius_loss", 0.25 * strength_loss)?;
    let tiny_strength_loss = section.get_f32("tiny_strength_loss", strength_loss)?;
    let strength_zero_loss = section.get_f32("strength_zero_loss", 0.0)?;
    let hf_lr_mult = section.get_f32("hf_lr_mult", 1.0)?;
    let alpha = section.get_f32("alpha", 2.0)?;
    let p_exp = section.get_f32("p_exp", 2.0)?;

    let params_mode = section.get_str("params", "");
    let mut pred_hidden_raw = section.get_i64("pred_hidden", 0)?;
    match params_mode.as_str() {
        "learned" => {
            if pred_hidden_raw <= 0 {
                pred_hidden_raw = 64;
            }
        }
        "fixed" => pred_hidden_raw = 0,
        "" | "trainable" | "global" => {}
        other => return Err(invalid(&format!("unsupported params mode `{other}`"))),
    }
    if pred_hidden_raw < 0 {
        return Err(invalid("pred_hidden must be non-negative"));
    }
    let pred_hidden =
        usize::try_from(pred_hidden_raw).map_err(|_| invalid("pred_hidden is out of range"))?;

    let downsample_raw = section.get_i64("downsample", 8)?;
    if downsample_raw <= 0 {
        return Err(invalid("downsample must be positive"));
    }
    let downsample =
        usize::try_from(downsample_raw).map_err(|_| invalid("downsample is out of range"))?;
    let nr_iters_raw = section.get_i64("nr_iters", 10)?;
    if nr_iters_raw <= 0 {
        return Err(invalid("nr_iters must be positive"));
    }
    let nr_iters =
        usize::try_from(nr_iters_raw).map_err(|_| invalid("nr_iters is out of range"))?;
    let nr_eta = section.get_f32("nr_eta", 0.5)?;
    if !nr_eta.is_finite() || !(0.05..=1.0).contains(&nr_eta) {
        return Err(invalid("nr_eta must be finite and in [0.05, 1]"));
    }

    let inverse_method = match section.get_str("inverse_method", "newton").as_str() {
        "newton" => HfInverseMethod::Newton,
        "fixed_point" | "fixed-point" => HfInverseMethod::FixedPoint,
        other => return Err(invalid(&format!("unsupported inverse_method `{other}`"))),
    };
    let warp_source = match section.get_str("warp_source", "previous").as_str() {
        "previous" => HfWarpSource::Previous,
        "network_input" => HfWarpSource::NetworkInput,
        other => return Err(invalid(&format!("unsupported warp_source `{other}`"))),
    };
    let radius_mode = match section.get_str("radius_mode", "isotropic").as_str() {
        "isotropic" => HfRadiusMode::Isotropic,
        "anisotropic" => HfRadiusMode::Anisotropic,
        other => return Err(invalid(&format!("unsupported radius_mode `{other}`"))),
    };

    let pred_source_raw = section.get_isize("pred_source_layer", -1)?;
    let pred_source_layer = if pred_source_raw == -1 {
        None
    } else {
        let source_index = if pred_source_raw < -1 {
            layer_index.checked_add_signed(pred_source_raw)
        } else {
            usize::try_from(pred_source_raw).ok()
        }
        .ok_or_else(|| invalid(&format!("invalid pred_source_layer={pred_source_raw}")))?;
        if source_index >= layer_index {
            return Err(invalid("pred_source_layer must reference an earlier layer"));
        }
        if source_index >= built_layers.len() {
            return Err(invalid("pred_source_layer does not exist"));
        }
        Some(source_index)
    };
    let pred_input_shape = pred_source_layer
        .map(|source_index| built_layers[source_index].output_shape)
        .unwrap_or(input);
    if pred_input_shape.batch != input.batch
        || pred_input_shape.width == 0
        || pred_input_shape.height == 0
        || pred_input_shape.channels == 0
    {
        return Err(invalid(
            "predictor source must output a non-empty compatible NCHW tensor",
        ));
    }

    for (name, value) in [
        ("init_cx", init_cx),
        ("init_cy", init_cy),
        ("init_radius", init_radius),
        ("init_radius_x", init_radius_x),
        ("init_radius_y", init_radius_y),
        ("init_strength", init_strength),
        ("small_box_px", small_box_px),
        ("small_box_strength", small_box_strength),
        ("strength_loss", strength_loss),
        ("tiny_center_loss", tiny_center_loss),
        ("tiny_radius_loss", tiny_radius_loss),
        ("tiny_strength_loss", tiny_strength_loss),
        ("strength_zero_loss", strength_zero_loss),
        ("hf_lr_mult", hf_lr_mult),
        ("alpha", alpha),
        ("p_exp", p_exp),
    ] {
        if !value.is_finite() {
            return Err(invalid(&format!("{name} must be finite")));
        }
    }
    if !(-1.0..=1.0).contains(&init_cx) || !(-1.0..=1.0).contains(&init_cy) {
        return Err(invalid("initial center coordinates must be in [-1, 1]"));
    }
    if init_radius <= 0.0 || init_radius_x <= 0.0 || init_radius_y <= 0.0 {
        return Err(invalid("initial radii must be positive"));
    }
    if !(0.0..=1.0).contains(&init_strength)
        || small_box_px < 0.0
        || !(0.0..=1.0).contains(&small_box_strength)
        || strength_loss < 0.0
        || tiny_center_loss < 0.0
        || tiny_radius_loss < 0.0
        || tiny_strength_loss < 0.0
        || strength_zero_loss < 0.0
        || hf_lr_mult < 0.0
        || alpha <= 0.0
        || p_exp <= 0.0
    {
        return Err(invalid(
            "strengths, losses, lr multiplier, alpha, or p_exp are outside their supported ranges",
        ));
    }
    if section.get_i64("batch_normalize", 0)? != 0 || section.get_i64("cbn", 0)? != 0 {
        return Err(invalid("batch normalization is unsupported"));
    }
    if section.get_i64("grad_centr", 0)? != 0 || section.get_f32("clip", 0.0)? != 0.0 {
        return Err(invalid("grad_centr and clip are unsupported"));
    }
    let learning_rate = section.get_f32("learning_rate", 1.0)?;
    if !learning_rate.is_finite() || learning_rate < 0.0 {
        return Err(invalid("learning_rate must be finite and non-negative"));
    }
    if section.get_i64("stream", -1)? != -1 || section.get_i64("wait_stream", -1)? != -1 {
        return Err(invalid(
            "GPU stream options are unsupported in the Stage-1 CPU port",
        ));
    }
    let debug = section.get_i64("debug", 0)? != 0;
    let debug_interval_raw = section.get_i64("debug_interval", 100)?;
    if debug {
        return Err(invalid(
            "debug summaries are unsupported in the Stage-1 CPU port",
        ));
    }
    if debug_interval_raw <= 0 {
        return Err(invalid("debug_interval must be positive"));
    }
    if section.get("debug_interval").is_some() && debug_interval_raw != 100 {
        return Err(invalid(
            "custom debug_interval is unsupported without debug summaries",
        ));
    }
    let debug_interval = usize::try_from(debug_interval_raw)
        .map_err(|_| invalid("debug_interval is out of range"))?;

    let input_dim = pred_input_shape
        .channels
        .checked_mul((pred_input_shape.height / downsample).max(1))
        .and_then(|value| value.checked_mul((pred_input_shape.width / downsample).max(1)))
        .ok_or_else(|| invalid("predictor input dimension overflows"))?;
    if pred_hidden > 0 {
        let param_count = match radius_mode {
            HfRadiusMode::Isotropic => 4usize,
            HfRadiusMode::Anisotropic => 5usize,
        };
        pred_hidden
            .checked_mul(input_dim)
            .and_then(|value| value.checked_add(pred_hidden))
            .and_then(|value| {
                param_count
                    .checked_mul(pred_hidden)
                    .and_then(|x| x.checked_add(value))
            })
            .and_then(|value| value.checked_add(param_count))
            .and_then(|value| value.checked_add(param_count))
            .ok_or_else(|| invalid("predictor parameter count overflows"))?;
    }

    Ok(HyperbolicFoveationSpec {
        init_cx,
        init_cy,
        init_radius,
        init_radius_x,
        init_radius_y,
        init_strength,
        small_box_px,
        small_box_strength,
        strength_loss,
        tiny_center_loss,
        tiny_radius_loss,
        tiny_strength_loss,
        strength_zero_loss,
        hf_lr_mult,
        alpha,
        p_exp,
        downsample,
        pred_hidden,
        nr_iters,
        nr_eta,
        inverse_method,
        pred_source_layer,
        pred_input_shape,
        warp_source,
        radius_mode,
        debug,
        debug_interval,
    })
}

fn reject_unknown_graph_conv_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "filters",
        "groups",
        "size",
        "stride",
        "stride_x",
        "stride_y",
        "dilation",
        "pad",
        "padding",
        "activation",
        "batch_normalize",
        "cbn",
        "binary",
        "xnor",
        "bin_output",
        "antialiasing",
        "assisted_excitation",
        "share_index",
        "graph_edge_mode",
        "edge_mode",
        "graph_use_self",
        "use_self",
        "graph_valid_mask_zero",
        "valid_mask_zero",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn parse_graph_conv_spec(section: &RawSection, input: TensorShape) -> Result<GraphConvSpec> {
    let line = section.line;
    reject_unknown_graph_conv_keys(section)?;
    let invalid = |message: &str| {
        Error::InvalidConfig(format!("[{}] at line {line}: {message}", section.name))
    };
    if input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid("preceding layer must output a non-empty image"));
    }

    let filters = section.get_i64("filters", 1)?;
    let groups = section.get_i64("groups", 1)?.max(1);
    let size = section.get_i64("size", 1)?;
    let stride = section.get_i64("stride", 1)?;
    let mut stride_x = section.get_i64("stride_x", -1)?;
    let mut stride_y = section.get_i64("stride_y", -1)?;
    if stride_x < 1 {
        stride_x = stride;
    }
    if stride_y < 1 {
        stride_y = stride;
    }
    let mut dilation = section.get_i64("dilation", 1)?;
    if size == 1 {
        dilation = 1;
    }
    let pad = section.get_i64("pad", 0)? != 0;
    let raw_padding = if pad {
        size / 2
    } else {
        section.get_i64("padding", 0)?
    };
    let edge_mode = if section.get("graph_edge_mode").is_some() {
        section.get_i64("graph_edge_mode", 1)?
    } else {
        section.get_i64("edge_mode", 1)?
    };
    let use_self = if section.get("graph_use_self").is_some() {
        section.get_i64("graph_use_self", 1)?
    } else {
        section.get_i64("use_self", 1)?
    } != 0;
    let valid_mask_zero = if section.get("graph_valid_mask_zero").is_some() {
        section.get_i64("graph_valid_mask_zero", 1)?
    } else {
        section.get_i64("valid_mask_zero", 1)?
    };
    let batch_normalize =
        section.get_i64("batch_normalize", 0)? != 0 || section.get_i64("cbn", 0)? != 0;
    let activation = Activation::parse(&section.get_str("activation", "logistic"));

    if matches!(activation, Activation::Other(_)) {
        return Err(invalid("activation name is unsupported"));
    }
    if filters <= 0 || size <= 0 || size % 2 == 0 {
        return Err(invalid(
            "filters must be positive and size must be positive and odd",
        ));
    }
    if stride_x <= 0 || stride_y <= 0 || dilation <= 0 {
        return Err(invalid("stride_x, stride_y, and dilation must be positive"));
    }
    if raw_padding < 0 {
        return Err(invalid("padding must be non-negative"));
    }
    let filters = usize::try_from(filters).map_err(|_| invalid("filters is out of range"))?;
    let groups = usize::try_from(groups).map_err(|_| invalid("groups is out of range"))?;
    let size = usize::try_from(size).map_err(|_| invalid("size is out of range"))?;
    let stride_x = usize::try_from(stride_x).map_err(|_| invalid("stride_x is out of range"))?;
    let stride_y = usize::try_from(stride_y).map_err(|_| invalid("stride_y is out of range"))?;
    let dilation = usize::try_from(dilation).map_err(|_| invalid("dilation is out of range"))?;
    let padding = usize::try_from(raw_padding).map_err(|_| invalid("padding is out of range"))?;
    if input.channels % groups != 0 || filters % groups != 0 {
        return Err(invalid(
            "groups must divide both input channels and filters",
        ));
    }
    if edge_mode != 0 && edge_mode != 1 {
        return Err(invalid("graph_edge_mode must be 0 or 1"));
    }
    if valid_mask_zero != 1 {
        return Err(invalid("graph_valid_mask_zero must be 1"));
    }
    if section.get_i64("binary", 0)? != 0
        || section.get_i64("xnor", 0)? != 0
        || section.get_i64("bin_output", 0)? != 0
    {
        return Err(invalid(
            "binary, xnor, and bin_output modes are unsupported",
        ));
    }
    if section.get_i64("antialiasing", 0)? != 0 {
        return Err(invalid("antialiasing is unsupported"));
    }
    if section.get_f32("assisted_excitation", 0.0)? != 0.0 {
        return Err(invalid("assisted_excitation is unsupported"));
    }
    if section.get_i64("share_index", -1_000_000_000)? != -1_000_000_000 {
        return Err(invalid("shared graph-convolution layers are unsupported"));
    }
    if section.get_f32("grad_centr", 0.0)? != 0.0 {
        return Err(invalid("grad_centr is unsupported for graph_conv"));
    }
    if section.get_f32("clip", 0.0)? != 0.0 {
        return Err(invalid("clip is unsupported for graph_conv"));
    }

    let kernel_area = size
        .checked_mul(size)
        .ok_or_else(|| invalid("kernel area overflows"))?;
    let channels_per_group = input.channels / groups;
    let projection_count = filters
        .checked_mul(channels_per_group)
        .ok_or_else(|| invalid("projection parameter count overflows"))?;
    let self_count = if use_self { projection_count } else { 0 };
    let edge_count = if edge_mode == 1 {
        groups
            .checked_mul(kernel_area)
            .and_then(|count| count.checked_mul(2))
            .and_then(|count| count.checked_mul(channels_per_group))
            .and_then(|count| count.checked_add(groups.checked_mul(kernel_area)?))
            .ok_or_else(|| invalid("edge parameter count overflows"))?
    } else {
        0
    };
    projection_count
        .checked_add(self_count)
        .and_then(|count| count.checked_add(edge_count))
        .and_then(|count| count.checked_add(filters))
        .ok_or_else(|| invalid("parameter count overflows"))?;
    for (dimension, stride) in [(input.width, stride_x), (input.height, stride_y)] {
        dcnv4_output_dim(dimension, padding, size, stride, dilation)
            .ok_or_else(|| invalid("kernel geometry has an invalid or empty output"))?;
    }

    Ok(GraphConvSpec {
        filters,
        groups,
        size,
        stride_x,
        stride_y,
        dilation,
        padding,
        edge_mode: edge_mode as usize,
        use_self,
        batch_normalize,
        activation,
    })
}

fn reject_unknown_clifford_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "filters",
        "shifts",
        "shifts_wedge",
        "shifts_inner",
        "ctx_mode",
        "cli_mode",
        "gffn_mode",
        "cli_higher_mode",
        "global_op",
        "fno_modes_h",
        "fno_modes_w",
        "dwconv_size",
        "num_dwconv",
        "activation",
        "drop_path",
        "layerscale",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
        "stream",
        "wait_stream",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn parse_clifford_spec(section: &RawSection, input: TensorShape) -> Result<CliffordSpec> {
    let line = section.line;
    reject_unknown_clifford_keys(section)?;
    let invalid =
        |message: &str| Error::InvalidConfig(format!("[clifford] at line {line}: {message}"));
    if input.batch == 0 || input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid(
            "preceding layer must output a non-empty NCHW image",
        ));
    }
    let default_filters = i64::try_from(input.channels)
        .map_err(|_| invalid("input channel count exceeds the supported range"))?;
    let filters = section.get_i64("filters", default_filters)?;
    if filters <= 0 || usize::try_from(filters).ok() != Some(input.channels) {
        return Err(invalid("filters must be positive and equal input channels"));
    }

    let mut shifts = section.get_isize_list("shifts")?;
    let mut wedge_shifts = section.get_isize_list("shifts_wedge")?;
    let mut inner_shifts = section.get_isize_list("shifts_inner")?;
    if shifts.is_empty() && wedge_shifts.is_empty() && inner_shifts.is_empty() {
        shifts = vec![1, 2];
    }
    if wedge_shifts.is_empty() {
        wedge_shifts = if shifts.is_empty() {
            inner_shifts.clone()
        } else {
            shifts.clone()
        };
    }
    if inner_shifts.is_empty() {
        if shifts.is_empty() {
            inner_shifts = wedge_shifts.clone();
        } else if wedge_shifts != shifts {
            inner_shifts = shifts.clone();
        }
    }
    if wedge_shifts.is_empty() {
        return Err(invalid("at least one shift is required"));
    }
    let channels = isize::try_from(input.channels)
        .map_err(|_| invalid("input channel count exceeds the supported shift range"))?;
    let normalize = |value: isize| value.rem_euclid(channels) as usize;
    let wedge_shifts = wedge_shifts.into_iter().map(normalize).collect::<Vec<_>>();
    let inner_shifts = if inner_shifts.is_empty() {
        wedge_shifts.clone()
    } else {
        inner_shifts.into_iter().map(normalize).collect::<Vec<_>>()
    };

    let context_mode = match section.get_str("ctx_mode", "diff").as_str() {
        "abs" => CliffordContextMode::Abs,
        "diff" => CliffordContextMode::Diff,
        other => return Err(invalid(&format!("unsupported ctx_mode `{other}`"))),
    };
    let interaction_mode = match section.get_str("cli_mode", "full").as_str() {
        "inner" => CliffordInteractionMode::Inner,
        "wedge" => CliffordInteractionMode::Wedge,
        "full" => CliffordInteractionMode::Full,
        other => return Err(invalid(&format!("unsupported cli_mode `{other}`"))),
    };
    let gffn_mode = match section.get_str("gffn_mode", "local").as_str() {
        "local" => CliffordGffnMode::Local,
        "global" => CliffordGffnMode::Global,
        "hybrid" => CliffordGffnMode::Hybrid,
        other => return Err(invalid(&format!("unsupported gffn_mode `{other}`"))),
    };
    let higher_mode = match section.get_str("cli_higher_mode", "off").as_str() {
        "off" => CliffordHigherMode::Off,
        "vb" => CliffordHigherMode::VectorBivector,
        other => return Err(invalid(&format!("unsupported cli_higher_mode `{other}`"))),
    };
    let global_op = match section.get_str("global_op", "avgpool").as_str() {
        "avgpool" => CliffordGlobalOp::AvgPool,
        "fourier" => CliffordGlobalOp::Fourier,
        other => return Err(invalid(&format!("unsupported global_op `{other}`"))),
    };
    let fno_modes_h_raw = section.get_i64("fno_modes_h", 4)?;
    let fno_modes_w_raw = section.get_i64("fno_modes_w", 4)?;
    let (fno_modes_h, fno_modes_w) = if global_op == CliffordGlobalOp::Fourier {
        if fno_modes_h_raw < 1 || fno_modes_w_raw < 1 {
            return Err(invalid(
                "Fourier global_op requires positive fno_modes_h and fno_modes_w",
            ));
        }
        (
            usize::try_from(fno_modes_h_raw).map_err(|_| invalid("fno_modes_h is out of range"))?,
            usize::try_from(fno_modes_w_raw).map_err(|_| invalid("fno_modes_w is out of range"))?,
        )
    } else {
        (0, 0)
    };
    let dwconv_size = section.get_i64("dwconv_size", 3)?;
    if dwconv_size < 1 || dwconv_size % 2 == 0 {
        return Err(invalid("dwconv_size must be a positive odd integer"));
    }
    let dwconv_size =
        usize::try_from(dwconv_size).map_err(|_| invalid("dwconv_size is out of range"))?;
    let num_dwconv = usize::try_from(section.get_i64("num_dwconv", 1)?.max(1))
        .map_err(|_| invalid("num_dwconv is out of range"))?;
    let activation = Activation::parse(&section.get_str("activation", "swish"));
    if activation != Activation::Swish {
        return Err(invalid("only activation=swish is supported"));
    }
    let drop_path = section.get_f32("drop_path", 0.0)?;
    if !drop_path.is_finite() || !(0.0..1.0).contains(&drop_path) {
        return Err(invalid("drop_path must be finite and in [0, 1)"));
    }
    let layerscale_init = section.get_f32("layerscale", 0.0001)?;
    if !layerscale_init.is_finite() {
        return Err(invalid("layerscale must be finite"));
    }
    let learning_rate = section.get_f32("learning_rate", 1.0)?;
    if !learning_rate.is_finite() {
        return Err(invalid("learning_rate must be finite"));
    }
    if section.get_f32("clip", 0.0)? != 0.0 {
        return Err(invalid("clip is not supported for clifford"));
    }
    if section.get_i64("grad_centr", 0)? != 0 {
        return Err(invalid("grad_centr is not supported for clifford"));
    }
    if section.get_i64("stream", -1)? != -1 || section.get_i64("wait_stream", -1)? != -1 {
        return Err(invalid(
            "non-default stream scheduling is GPU-only and unsupported by the CPU port",
        ));
    }

    let raw_shift_count = match interaction_mode {
        CliffordInteractionMode::Inner => inner_shifts.len(),
        CliffordInteractionMode::Wedge => wedge_shifts.len(),
        CliffordInteractionMode::Full => wedge_shifts
            .len()
            .checked_add(inner_shifts.len())
            .ok_or_else(|| invalid("interaction channel count overflows"))?,
    };
    let raw_channels = raw_shift_count
        .checked_mul(input.channels)
        .ok_or_else(|| invalid("interaction channel count overflows"))?;
    let channels = input.channels;
    let channels_squared = channels
        .checked_mul(channels)
        .ok_or_else(|| invalid("parameter count overflows"))?;
    let gate_weights = channels_squared
        .checked_mul(2)
        .ok_or_else(|| invalid("gate parameter count overflows"))?;
    let kernel_area = dwconv_size
        .checked_mul(dwconv_size)
        .ok_or_else(|| invalid("depthwise-convolution kernel size overflows"))?;
    let dwconv_parameters = channels
        .checked_mul(
            4usize
                .checked_add(kernel_area)
                .ok_or_else(|| invalid("depthwise-convolution parameter count overflows"))?,
        )
        .and_then(|count| count.checked_mul(num_dwconv))
        .ok_or_else(|| invalid("depthwise-convolution parameter count overflows"))?;
    let projection_weights = channels
        .checked_mul(raw_channels)
        .ok_or_else(|| invalid("projection parameter count overflows"))?;
    let mut parameter_count = 0usize;
    for count in [
        channels
            .checked_mul(2)
            .ok_or_else(|| invalid("parameter count overflows"))?,
        channels_squared,
        channels,
        dwconv_parameters,
        projection_weights,
        channels,
        gate_weights,
        channels,
        channels,
    ] {
        parameter_count = parameter_count
            .checked_add(count)
            .ok_or_else(|| invalid("parameter count overflows"))?;
    }
    if gffn_mode != CliffordGffnMode::Local {
        for count in [projection_weights, channels, gate_weights, channels] {
            parameter_count = parameter_count
                .checked_add(count)
                .ok_or_else(|| invalid("global parameter count overflows"))?;
        }
        if global_op == CliffordGlobalOp::Fourier {
            let fourier_count = channels
                .checked_mul(fno_modes_h)
                .and_then(|count| count.checked_mul(fno_modes_w))
                .and_then(|count| count.checked_mul(2))
                .ok_or_else(|| invalid("Fourier parameter count overflows"))?;
            parameter_count = parameter_count
                .checked_add(fourier_count)
                .ok_or_else(|| invalid("parameter count overflows"))?;
        }
    }
    let _ = parameter_count;
    input
        .width
        .checked_mul(input.height)
        .and_then(|area| area.checked_mul(input.channels))
        .and_then(|count| count.checked_mul(input.batch))
        .ok_or_else(|| invalid("tensor element count overflows"))?;

    Ok(CliffordSpec {
        filters: input.channels,
        wedge_shifts,
        inner_shifts,
        context_mode,
        interaction_mode,
        gffn_mode,
        higher_mode,
        global_op,
        fno_modes_h,
        fno_modes_w,
        dwconv_size,
        num_dwconv,
        drop_path,
        layerscale_init,
    })
}

fn reject_unknown_wmhf_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "filters",
        "identity_ratio",
        "local_ratio",
        "freq_scale",
        "shortcut",
        "batch_normalize",
        "activation",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn parse_wmhf_spec(section: &RawSection, input: TensorShape) -> Result<WmhfSpec> {
    let line = section.line;
    reject_unknown_wmhf_keys(section)?;
    let invalid = |message: &str| {
        Error::InvalidConfig(format!("[{}] at line {line}: {message}", section.name))
    };
    if input.batch == 0 || input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid("preceding layer must output a non-empty image"));
    }
    let default_filters = i64::try_from(input.channels)
        .map_err(|_| invalid("input channels exceed the supported range"))?;
    let filters = section.get_i64("filters", default_filters)?;
    let identity_ratio = section.get_f32("identity_ratio", 0.25)?;
    let local_ratio = section.get_f32("local_ratio", 0.375)?;
    let freq_scale = section.get_f32("freq_scale", 0.25)?;
    let shortcut = section.get_i64("shortcut", 1)? != 0;
    let batch_normalize = section.get_i64("batch_normalize", 1)? != 0;
    let activation = Activation::parse(&section.get_str("activation", "leaky"));
    if !identity_ratio.is_finite() || !local_ratio.is_finite() {
        return Err(invalid("identity_ratio and local_ratio must be finite"));
    }
    if !freq_scale.is_finite() {
        return Err(invalid("freq_scale must be finite"));
    }
    if matches!(activation, Activation::Other(_)) {
        return Err(invalid("activation name is unsupported"));
    }
    if filters < 4 {
        return Err(invalid("filters must be at least 4"));
    }
    if section.get_f32("clip", 0.0)? != 0.0 {
        return Err(invalid("clip is unsupported for wmhf"));
    }
    if section.get_i64("grad_centr", 0)? != 0 {
        return Err(invalid("grad_centr is unsupported for wmhf"));
    }

    let round_ratio = |ratio: f32, name: &str| -> Result<i64> {
        let scaled = filters as f32 * ratio;
        let rounded = scaled.round();
        if !scaled.is_finite()
            || !rounded.is_finite()
            || rounded < i64::MIN as f32
            || rounded >= i64::MAX as f32
        {
            return Err(invalid(&format!(
                "{name} produces an out-of-range channel split"
            )));
        }
        Ok(rounded as i64)
    };
    let mut identity = round_ratio(identity_ratio, "identity_ratio")?.max(0);
    let mut local = round_ratio(local_ratio, "local_ratio")?.max(1);
    if identity.saturating_add(local) > filters - 1 {
        local = filters
            .checked_sub(identity)
            .and_then(|remaining| remaining.checked_sub(1))
            .unwrap_or(i64::MIN)
            .max(1);
    }
    let mut global = filters
        .checked_sub(identity)
        .and_then(|remaining| remaining.checked_sub(local))
        .unwrap_or(-1);
    if global < 1 {
        global = 1;
        if local > 1 {
            local -= 1;
        } else if identity > 0 {
            identity -= 1;
        }
    }
    if identity < 0
        || local < 1
        || global < 1
        || identity
            .checked_add(local)
            .and_then(|sum| sum.checked_add(global))
            != Some(filters)
    {
        return Err(invalid(
            "channel split cannot produce positive local and global groups",
        ));
    }
    let filters = usize::try_from(filters).map_err(|_| invalid("filters is out of range"))?;
    let identity_channels =
        usize::try_from(identity).map_err(|_| invalid("identity channel count is out of range"))?;
    let local_channels =
        usize::try_from(local).map_err(|_| invalid("local channel count is out of range"))?;
    let global_channels =
        usize::try_from(global).map_err(|_| invalid("global channel count is out of range"))?;

    let mut parameter_count = 0usize;
    let mut add_conv = |input_channels: usize,
                        output_channels: usize,
                        kernel: usize,
                        groups: usize,
                        batch_norm: bool|
     -> Result<()> {
        let weights = output_channels
            .checked_mul(input_channels / groups)
            .and_then(|count| count.checked_mul(kernel))
            .and_then(|count| count.checked_mul(kernel))
            .ok_or_else(|| invalid("convolution parameter count overflows"))?;
        let arrays = output_channels
            .checked_mul(if batch_norm { 4 } else { 1 })
            .ok_or_else(|| invalid("convolution parameter count overflows"))?;
        parameter_count = parameter_count
            .checked_add(weights)
            .and_then(|count| count.checked_add(arrays))
            .ok_or_else(|| invalid("parameter count overflows"))?;
        Ok(())
    };
    let local_mix_channels = local_channels
        .checked_mul(3)
        .ok_or_else(|| invalid("local mixer channel count overflows"))?;
    let scan_parameter_count = global_channels
        .checked_mul(4)
        .ok_or_else(|| invalid("scan parameter count overflows"))?;
    add_conv(input.channels, filters, 1, 1, batch_normalize)?;
    for kernel in [3, 5, 7] {
        add_conv(
            local_channels,
            local_channels,
            kernel,
            local_channels,
            false,
        )?;
    }
    add_conv(local_mix_channels, local_channels, 1, 1, batch_normalize)?;
    add_conv(global_channels, filters, 1, 1, false)?;
    add_conv(filters, filters, 1, 1, batch_normalize)?;
    parameter_count
        .checked_add(scan_parameter_count)
        .ok_or_else(|| invalid("parameter count overflows"))?;
    input
        .width
        .checked_mul(input.height)
        .and_then(|area| area.checked_mul(input.batch))
        .and_then(|count| count.checked_mul(filters))
        .ok_or_else(|| invalid("output element count overflows"))?;

    Ok(WmhfSpec {
        filters,
        identity_channels,
        local_channels,
        global_channels,
        freq_scale,
        shortcut,
        batch_normalize,
        activation,
    })
}

fn reject_unknown_dsconv_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "filters",
        "size",
        "stride",
        "stride_x",
        "stride_y",
        "pad",
        "padding",
        "activation",
        "extend_scope",
        "morph",
        "templates",
        "drop_prob",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn parse_dsconv_spec(section: &RawSection, input: TensorShape) -> Result<DsConvSpec> {
    let line = section.line;
    reject_unknown_dsconv_keys(section)?;
    let invalid = |message: &str| {
        Error::InvalidConfig(format!("[{}] at line {line}: {message}", section.name))
    };
    if input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid("preceding layer must output a non-empty image"));
    }

    let filters = section.get_i64("filters", input.channels as i64)?;
    let raw_size = section.get_i64("size", 9)?;
    let mut stride_x = section.get_i64("stride_x", -1)?;
    let mut stride_y = section.get_i64("stride_y", -1)?;
    if stride_x < 1 || stride_y < 1 {
        let stride = section.get_i64("stride", 1)?;
        if stride_x < 1 {
            stride_x = stride;
        }
        if stride_y < 1 {
            stride_y = stride;
        }
    }
    let mut padding = section.get_i64("padding", -1)?;
    if section.get_i64("pad", 1)? != 0 {
        padding = raw_size / 2;
    }
    if padding < 0 {
        padding = 0;
    }
    if raw_size % 2 == 0 {
        return Err(invalid("size must be odd"));
    }
    if stride_x != 1 || stride_y != 1 || padding != raw_size / 2 {
        return Err(invalid(
            "only stride=1 and same padding (padding=size/2) are supported",
        ));
    }
    if filters <= 0 {
        return Err(invalid("filters must be positive"));
    }
    let size = if raw_size < 1 {
        1
    } else {
        usize::try_from(raw_size).map_err(|_| invalid("size is out of range"))?
    };
    let padding = usize::try_from(padding).map_err(|_| invalid("padding must be non-negative"))?;
    let filters = usize::try_from(filters).map_err(|_| invalid("filters must be positive"))?;

    let extend_scope = section.get_f32("extend_scope", 1.0)?;
    let mut morph = section.get_i64("morph", 0)?;
    let mut templates = section.get_i64("templates", 1)?;
    let mut drop_prob = section.get_f32("drop_prob", 0.0)?;
    if !extend_scope.is_finite() || !drop_prob.is_finite() {
        return Err(invalid("extend_scope and drop_prob must be finite"));
    }
    if morph < 0 || morph > 2 {
        morph = 0;
    }
    if templates < 1 {
        templates = 1;
    }
    if morph == 2 && filters < 2 {
        return Err(invalid("morph=2 requires at least two filters"));
    }
    if morph == 2 && templates != 1 {
        templates = 1;
        drop_prob = 0.0;
    }
    drop_prob = drop_prob.clamp(0.0, 1.0);
    if templates > 1 {
        return Err(invalid(
            "templates>1 multi-view execution is not yet supported by the Rust CPU port",
        ));
    }

    filters
        .checked_mul(input.channels)
        .and_then(|count| count.checked_mul(size))
        .ok_or_else(|| invalid("collapse weight count overflows"))?;
    size.checked_mul(2)
        .and_then(|count| count.checked_mul(input.channels))
        .and_then(|count| count.checked_mul(9))
        .ok_or_else(|| invalid("offset weight count overflows"))?;
    Ok(DsConvSpec {
        filters,
        size,
        padding,
        extend_scope,
        morph: morph as usize,
        templates: templates as usize,
        drop_prob,
        activation: Activation::parse(&section.get_str("activation", "relu")),
    })
}

fn reject_unknown_deform_conv_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "filters",
        "groups",
        "size",
        "stride",
        "stride_x",
        "stride_y",
        "dilation",
        "antialiasing",
        "pad",
        "padding",
        "activation",
        "assisted_excitation",
        "share_index",
        "batch_normalize",
        "cbn",
        "binary",
        "xnor",
        "bin_output",
        "use_mask",
        "mask",
        "flipped",
        "dot",
        "angle",
        "grad_centr",
        "reverse",
        "coordconv",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn parse_deform_conv_spec(section: &RawSection, input: TensorShape) -> Result<DeformConvSpec> {
    let line = section.line;
    reject_unknown_deform_conv_keys(section)?;
    let invalid = |message: &str| {
        Error::InvalidConfig(format!("[{}] at line {line}: {message}", section.name))
    };
    let filters = section.get_i64("filters", 1)?;
    let groups = section.get_i64("groups", 1)?;
    let size = section.get_i64("size", 1)?;
    let stride = section.get_i64("stride", 1)?;
    let stride_x = section.get_i64("stride_x", -1)?;
    let stride_y = section.get_i64("stride_y", -1)?;
    let dilation = section.get_i64("dilation", 1)?;
    let pad = section.get_i64("pad", 0)? != 0;
    let padding = section.get_i64("padding", 0)?;
    let padding = if pad {
        usize::try_from(size / 2).map_err(|_| invalid("size must be positive"))?
    } else {
        usize::try_from(padding).map_err(|_| invalid("padding must be non-negative"))?
    };
    let batch_normalize = section.get_i64("batch_normalize", 0)?;
    let cbn = section.get_i64("cbn", 0)?;
    let activation = Activation::parse(&section.get_str("activation", "logistic"));
    let use_mask = if section.get("use_mask").is_some() {
        section.get_i64("use_mask", 1)? != 0
    } else {
        section.get_i64("mask", 1)? != 0
    };

    if input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid("preceding layer must output a non-empty image"));
    }
    if filters <= 0 || groups <= 0 || size <= 0 || stride <= 0 {
        return Err(invalid(
            "filters, groups, size, and stride must be positive",
        ));
    }
    if batch_normalize != 0 && batch_normalize != 1 {
        return Err(invalid(
            "batch_normalize must be 0 or 1; cross-batch BN is unsupported",
        ));
    }
    if cbn != 0 {
        return Err(invalid("cbn/cross-batch normalization is unsupported"));
    }
    if section.get_i64("binary", 0)? != 0
        || section.get_i64("xnor", 0)? != 0
        || section.get_i64("bin_output", 0)? != 0
    {
        return Err(invalid(
            "binary, xnor, and bin_output modes are unsupported",
        ));
    }
    if section.get_i64("antialiasing", 0)? != 0 {
        return Err(invalid("antialiasing is unsupported"));
    }
    if section.get_f32("assisted_excitation", 0.0)? != 0.0 {
        return Err(invalid("assisted_excitation is unsupported"));
    }
    if section.get_i64("share_index", -1_000_000_000)? != -1_000_000_000 {
        return Err(invalid(
            "shared deformable-convolution layers are unsupported",
        ));
    }
    if section.get_i64("flipped", 0)? != 0
        || section.get_f32("dot", 0.0)? != 0.0
        || section.get_f32("angle", 15.0)? != 15.0
        || section.get_f32("reverse", 0.0)? != 0.0
        || section.get_i64("coordconv", 0)? != 0
    {
        return Err(invalid(
            "specialized convolution transforms are unsupported",
        ));
    }
    if section.get_i64("stream", -1)? != -1 || section.get_i64("wait_stream", -1)? != -1 {
        return Err(invalid("non-default stream execution is unsupported"));
    }
    if section.get_i64("grad_centr", 0)? != 0 {
        return Err(invalid(
            "grad_centr is unsupported for deformable convolution",
        ));
    }

    let size = size as usize;
    let dilation = if size == 1 {
        1
    } else {
        usize::try_from(dilation).map_err(|_| invalid("dilation must be positive"))?
    };
    let stride_x = if stride_x < 1 { stride } else { stride_x };
    let stride_y = if stride_y < 1 { stride } else { stride_y };
    let stride_x = usize::try_from(stride_x).map_err(|_| invalid("stride_x must be positive"))?;
    let stride_y = usize::try_from(stride_y).map_err(|_| invalid("stride_y must be positive"))?;
    let filters = filters as usize;
    let groups = groups as usize;
    if input.channels % groups != 0 || filters % groups != 0 {
        return Err(invalid(
            "groups must divide both input channels and filters",
        ));
    }
    let kernel_area = size
        .checked_mul(size)
        .ok_or_else(|| invalid("kernel area overflows"))?;
    let offset_filters = kernel_area
        .checked_mul(2)
        .ok_or_else(|| invalid("offset count overflows"))?;
    let _offset_weights = offset_filters
        .checked_mul(input.channels)
        .and_then(|count| count.checked_mul(kernel_area))
        .ok_or_else(|| invalid("offset weight count overflows"))?;
    let _main_weights = filters
        .checked_mul(input.channels / groups)
        .and_then(|count| count.checked_mul(kernel_area))
        .ok_or_else(|| invalid("main weight count overflows"))?;
    if use_mask {
        kernel_area
            .checked_mul(input.channels)
            .and_then(|count| count.checked_mul(kernel_area))
            .ok_or_else(|| invalid("mask weight count overflows"))?;
    }
    if deform_conv_output_dim(input.width, padding, size, stride_x, dilation).is_none()
        || deform_conv_output_dim(input.height, padding, size, stride_y, dilation).is_none()
    {
        return Err(invalid("invalid or empty output geometry"));
    }
    Ok(DeformConvSpec {
        filters,
        groups,
        size,
        stride_x,
        stride_y,
        dilation,
        padding,
        batch_normalize: batch_normalize != 0,
        use_mask,
        activation,
    })
}

fn parse_dcnv4_spec(section: &RawSection, input: TensorShape) -> Result<Dcnv4Spec> {
    let line = section.line;
    reject_unknown_dcnv4_keys(section)?;
    let invalid = |message: &str| {
        Error::InvalidConfig(format!("[{}] at line {line}: {message}", section.name))
    };

    // C++ forces filters to the input channel count, but still parses the cfg key.
    let _configured_filters = section.get_i64("filters", 1)?;
    let groups = section.get_i64("groups", 1)?.max(1) as usize;
    let size = section.get_i64("size", 1)?.max(1) as usize;
    let stride = section.get_i64("stride", 1)?.max(1) as usize;
    let stride_x = section.get_i64("stride_x", -1)?;
    let stride_y = section.get_i64("stride_y", -1)?;
    let stride_x = if stride_x < 1 {
        stride
    } else {
        stride_x as usize
    };
    let stride_y = if stride_y < 1 {
        stride
    } else {
        stride_y as usize
    };
    let dilation = section.get_i64("dilation", 1)?.max(1) as usize;
    let padding = if section.get_i64("pad", 0)? != 0 {
        size / 2
    } else {
        section.get_usize("padding", 0)?
    };
    let batch_normalize = section.get_bool_int("batch_normalize", false)?;
    let offset_scale = section.get_f32("offset_scale", 1.0)?;
    let remove_center = section.get_bool_int("remove_center", false)?;
    let d_stride = section.get_i64("d_stride", 8)?.max(1) as usize;
    let block_thread = section.get_i64("block_thread", 256)?.max(1) as usize;
    let block_thread = if block_thread >= 1024 {
        1024
    } else {
        block_thread.next_power_of_two()
    };
    let softmax = section.get_bool_int("softmax", false)?;
    let activation = Activation::parse(&section.get_str("activation", "logistic"));

    if input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(invalid("preceding layer must output a non-empty image"));
    }
    if groups == 0 || input.channels % groups != 0 {
        return Err(invalid(&format!(
            "groups={groups} must evenly divide input channels={}",
            input.channels
        )));
    }
    let area = size
        .checked_mul(size)
        .ok_or_else(|| invalid("kernel area overflows"))?;
    if remove_center && area <= 1 {
        return Err(invalid("remove_center=1 leaves no active kernel points"));
    }
    if !offset_scale.is_finite() {
        return Err(invalid("offset_scale must be finite"));
    }
    let spec = Dcnv4Spec {
        filters: input.channels,
        groups,
        size,
        stride_x,
        stride_y,
        dilation,
        padding,
        batch_normalize,
        offset_scale,
        remove_center,
        d_stride,
        block_thread,
        softmax,
        activation,
    };
    if dcnv4_parameter_layout(input.channels, &spec).is_none() {
        return Err(invalid("parameter dimensions overflow"));
    }
    if dcnv4_output_dim(input.width, padding, size, stride_x, dilation).is_none()
        || dcnv4_output_dim(input.height, padding, size, stride_y, dilation).is_none()
    {
        return Err(invalid("invalid or empty output geometry"));
    }
    Ok(spec)
}

fn reject_unknown_swin_keys(section: &RawSection) -> Result<()> {
    const ALLOWED: &[&str] = &[
        "filters",
        "size",
        "heads",
        "shift",
        "ffn_ratio",
        "activation",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {}: unknown option `{key}`",
            section.name, section.line
        )));
    }
    Ok(())
}

fn parse_swin_spec(section: &RawSection, input: TensorShape) -> Result<SwinSpec> {
    let line = section.line;
    reject_unknown_swin_keys(section)?;
    if input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: preceding layer must output a non-empty image",
            section.name
        )));
    }
    let filters = section.get_usize("filters", 128)?;
    let size = section.get_usize("size", 7)?;
    let heads = section.get_usize("heads", 4)?;
    let shift = section.get_i64("shift", 0)?;
    let ffn_ratio = section.get_usize("ffn_ratio", 4)?;
    if filters == 0 || size == 0 || heads == 0 || ffn_ratio == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: filters, size, heads, and ffn_ratio must be > 0",
            section.name
        )));
    }
    if input.channels % heads != 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: input channels={} must be divisible by heads={heads}",
            section.name, input.channels
        )));
    }
    let shift = match shift {
        0 => false,
        1 => true,
        value => {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {line}: shift must be 0 or 1, got {value}",
                section.name
            )));
        }
    };
    for key in ["stream", "wait_stream"] {
        let stream = section.get_i64(key, -1)?;
        if stream != -1 {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {line}: {key}={stream} is unsupported by the Rust runtime",
                section.name
            )));
        }
    }
    let (padded_height, padded_width) = crate::swin::padded_shape(input.height, input.width, size)
        .map_err(|error| {
            Error::InvalidConfig(format!("[{}] at line {line}: {error}", section.name))
        })?;
    let hidden = filters.checked_mul(ffn_ratio).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: filters * ffn_ratio overflows",
            section.name
        ))
    })?;
    let table_width = size
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: relative-bias width overflows",
                section.name
            ))
        })?;
    let table_len = table_width.checked_mul(table_width).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: relative-bias table overflows",
            section.name
        ))
    })?;
    let tokens = size.checked_mul(size).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: window token count overflows",
            section.name
        ))
    })?;
    let _ = tokens.checked_mul(tokens).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: attention matrix size overflows",
            section.name
        ))
    })?;
    let windows = [input.batch, padded_height / size, padded_width / size]
        .into_iter()
        .try_fold(1usize, |product, factor| {
            product.checked_mul(factor).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[{}] at line {line}: padded window count overflows",
                    section.name
                ))
            })
        })?;
    let mut total_parameters = 0usize;
    for (what, factors) in [
        ("QKV weights", vec![3, input.channels, input.channels]),
        ("QKV bias", vec![3, input.channels]),
        ("output projection", vec![filters, input.channels]),
        ("output bias", vec![filters]),
        ("LayerNorm 1 parameters", vec![2, input.channels]),
        ("LayerNorm 2 parameters", vec![2, filters]),
        ("FFN weights", vec![hidden, filters]),
        ("FFN hidden bias", vec![hidden]),
        ("FFN output bias", vec![filters]),
        (
            "residual projection",
            if input.channels == filters {
                vec![0]
            } else {
                vec![filters, input.channels]
            },
        ),
        ("relative-position bias", vec![heads, table_len]),
        ("mHC parameters", vec![12]),
    ] {
        let block_size = factors.into_iter().try_fold(1usize, |product, factor| {
            product.checked_mul(factor).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[{}] at line {line}: {what} size overflows",
                    section.name
                ))
            })
        })?;
        total_parameters = total_parameters.checked_add(block_size).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: total parameter count overflows",
                section.name
            ))
        })?;
    }
    for (what, factors) in [
        ("windowed input", vec![windows, tokens, input.channels]),
        ("windowed output", vec![windows, tokens, filters]),
        ("attention scores", vec![windows, heads, tokens, tokens]),
        ("FFN workspace", vec![windows, tokens, hidden]),
    ] {
        factors.into_iter().try_fold(1usize, |product, factor| {
            product.checked_mul(factor).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[{}] at line {line}: {what} size overflows",
                    section.name
                ))
            })
        })?;
    }
    debug_assert!(total_parameters > 0);
    let activation = Activation::parse(&section.get_str("activation", "gelu"));
    if let Activation::Other(ref other) = activation {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: unsupported activation={other}",
            section.name
        )));
    }
    if matches!(
        activation,
        Activation::NormalizeChannels
            | Activation::NormalizeChannelsSoftmax
            | Activation::NormalizeChannelsSoftmaxMaxval
    ) {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: channel-normalizing activations are unsupported",
            section.name
        )));
    }
    Ok(SwinSpec {
        filters,
        size,
        heads,
        shift,
        ffn_ratio,
        activation,
    })
}

fn parse_tucker_attention_spec(
    section: &RawSection,
    input: TensorShape,
) -> Result<TuckerAttentionSpec> {
    const ALLOWED: &[&str] = &[
        "filters",
        "size",
        "heads",
        "rank",
        "rank_q",
        "rank_k",
        "rank_v",
        "rank_o",
        "activation",
        "msa",
        "minimax_sparse_attention",
        "msa_block",
        "msa_block_size",
        "msa_topk",
        "msa_top_blocks",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    let line = section.line;
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: unknown option `{key}`",
            section.name
        )));
    }
    if input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: preceding layer must output a non-empty image",
            section.name
        )));
    }
    let filters = section.get_usize("filters", input.channels)?;
    let size = section.get_usize("size", 7)?;
    let heads = section.get_usize("heads", 4)?;
    if filters == 0 || size == 0 || heads == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: filters, size, and heads must be > 0",
            section.name
        )));
    }
    if filters != input.channels {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: filters={filters} must equal input channels={} for the identity residual",
            section.name, input.channels
        )));
    }
    if input.channels % heads != 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: input channels={} must be divisible by heads={heads}",
            section.name, input.channels
        )));
    }
    let common_rank = section.get_i64("rank", 0)?;
    let common_rank = if common_rank > 0 {
        Some(usize::try_from(common_rank).map_err(|_| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: rank is too large for this platform",
                section.name
            ))
        })?)
    } else {
        None
    };
    let default_rank = (input.channels / 8).max(1);
    let rank_q = section.get_usize("rank_q", common_rank.unwrap_or(default_rank))?;
    let rank_k = section.get_usize("rank_k", common_rank.unwrap_or(rank_q))?;
    let rank_v = section.get_usize("rank_v", common_rank.unwrap_or(rank_q))?;
    let rank_o = section.get_usize("rank_o", common_rank.unwrap_or(rank_q))?;
    if rank_q == 0 || rank_k == 0 || rank_v == 0 || rank_o == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: rank_q, rank_k, rank_v, and rank_o must be > 0",
            section.name
        )));
    }
    for key in ["stream", "wait_stream"] {
        let stream = section.get_i64(key, -1)?;
        if stream != -1 {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {line}: {key}={stream} is unsupported by the Rust runtime",
                section.name
            )));
        }
    }
    let msa_enabled = if section.get("msa").is_some() {
        section.get_bool_int("msa", false)?
    } else {
        section.get_bool_int("minimax_sparse_attention", false)?
    };
    let msa_block_size = if section.get("msa_block").is_some() {
        section.get_i64("msa_block", 1)?.max(1) as usize
    } else if section.get("msa_block_size").is_some() {
        section.get_i64("msa_block_size", 1)?.max(1) as usize
    } else {
        size
    };
    let msa_topk = if section.get("msa_topk").is_some() {
        section.get_i64("msa_topk", 2)?.max(1) as usize
    } else if section.get("msa_top_blocks").is_some() {
        section.get_i64("msa_top_blocks", 2)?.max(1) as usize
    } else {
        2
    };
    let activation = Activation::parse(&section.get_str("activation", "linear"));
    if let Activation::Other(ref other) = activation {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: unsupported activation={other}",
            section.name
        )));
    }
    if matches!(
        activation,
        Activation::NormalizeChannels
            | Activation::NormalizeChannelsSoftmax
            | Activation::NormalizeChannelsSoftmaxMaxval
    ) {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: channel-normalizing activations are unsupported",
            section.name
        )));
    }
    let weight_count = crate::tucker_attention::tucker_weight_count(
        input.channels,
        heads,
        rank_q,
        rank_k,
        rank_v,
        rank_o,
    )
    .map_err(|error| Error::InvalidConfig(format!("[{}] at line {line}: {error}", section.name)))?;
    input.channels.checked_add(weight_count).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: total parameter count overflows",
            section.name
        ))
    })?;
    crate::tucker_attention::validate_geometry(&crate::tucker_attention::TuckerGeometry {
        batch: input.batch,
        channels: input.channels,
        height: input.height,
        width: input.width,
        size,
        heads,
        rank_q,
        rank_k,
        rank_v,
        rank_o,
        activation: activation.clone(),
        msa_enabled,
        msa_block_size,
        msa_topk,
    })
    .map_err(|error| Error::InvalidConfig(format!("[{}] at line {line}: {error}", section.name)))?;
    Ok(TuckerAttentionSpec {
        filters,
        size,
        heads,
        rank_q,
        rank_k,
        rank_v,
        rank_o,
        activation,
        msa_enabled,
        msa_block_size,
        msa_topk,
    })
}

/// C++ `parse_yolox_section`: `classes`(20), `max`(200), `jitter`(0.2) and
/// `resize`(1.0).  The PP-YOLOE / YOLO-NAS siblings share the first two keys and
/// add `reg_max`(16); they are not ported yet.
fn parse_mambavision_spec(section: &RawSection, input: TensorShape) -> Result<MambaVisionSpec> {
    const ALLOWED: &[&str] = &[
        "filters",
        "state",
        "d_state",
        "conv_size",
        "size",
        "dt_rank",
        "ffn_ratio",
        "activation",
        "msa",
        "minimax_sparse_attention",
        "msa_block",
        "msa_block_size",
        "msa_topk",
        "msa_top_blocks",
        "msa_heads",
        "causal",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    let line = section.line;
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: unknown option `{key}`",
            section.name
        )));
    }
    if input.width == 0 || input.height == 0 || input.channels == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: preceding layer must output a non-empty image",
            section.name
        )));
    }
    let filters = section.get_usize("filters", input.channels)?;
    let state = if section.get("state").is_some() {
        section.get_usize("state", 8)?
    } else {
        section.get_usize("d_state", 8)?
    };
    let conv_size = if section.get("conv_size").is_some() {
        section.get_usize("conv_size", 3)?
    } else {
        section.get_usize("size", 3)?
    };
    let dt_rank_raw = section.get_i64("dt_rank", 0)?;
    let dt_rank = if dt_rank_raw > 0 {
        usize::try_from(dt_rank_raw).map_err(|_| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: dt_rank is too large",
                section.name
            ))
        })?
    } else {
        filters.checked_add(15).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: default dt_rank overflows",
                section.name
            ))
        })? / 16
    };
    let ffn_ratio = section.get_usize("ffn_ratio", 4)?;
    if filters < 2 || filters % 2 != 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: filters must be even and >= 2, got {filters}",
            section.name
        )));
    }
    if state == 0 || conv_size == 0 || conv_size % 2 == 0 || ffn_ratio == 0 || dt_rank == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: state, dt_rank, ffn_ratio, and odd conv_size must be positive",
            section.name
        )));
    }
    let msa_enabled = if section.get("msa").is_some() {
        section.get_i64("msa", 0)? != 0
    } else {
        section.get_i64("minimax_sparse_attention", 0)? != 0
    };
    let msa_block_value = if section.get("msa_block").is_some() {
        section.get_i64("msa_block", 16)?
    } else if section.get("msa_block_size").is_some() {
        section.get_i64("msa_block_size", 16)?
    } else {
        16
    };
    let msa_topk_value = if section.get("msa_topk").is_some() {
        section.get_i64("msa_topk", 4)?
    } else if section.get("msa_top_blocks").is_some() {
        section.get_i64("msa_top_blocks", 4)?
    } else {
        4
    };
    let msa_heads_value = section.get_i64("msa_heads", 1)?.max(1);
    let mut msa_heads = usize::try_from(msa_heads_value).map_err(|_| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: msa_heads is too large",
            section.name
        ))
    })?;
    if filters % msa_heads != 0 {
        msa_heads = 1;
    }
    for key in ["stream", "wait_stream"] {
        let stream = section.get_i64(key, -1)?;
        if stream != -1 {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {line}: {key}={stream} is unsupported by the Rust runtime",
                section.name
            )));
        }
    }
    let activation = Activation::parse(&section.get_str("activation", "gelu"));
    if let Activation::Other(ref other) = activation {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: unsupported activation={other}",
            section.name
        )));
    }
    if matches!(
        activation,
        Activation::NormalizeChannels
            | Activation::NormalizeChannelsSoftmax
            | Activation::NormalizeChannelsSoftmaxMaxval
    ) {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: channel-normalizing activations are unsupported",
            section.name
        )));
    }
    let spec = MambaVisionSpec {
        filters,
        state,
        conv_size,
        dt_rank,
        ffn_ratio,
        activation,
        msa_enabled,
        msa_block_size: usize::try_from(msa_block_value.max(1)).map_err(|_| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: msa_block is too large",
                section.name
            ))
        })?,
        msa_topk: usize::try_from(msa_topk_value.max(1)).map_err(|_| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: msa_topk is too large",
                section.name
            ))
        })?,
        msa_heads,
        causal: section.get_i64("causal", 0)? != 0,
    };
    crate::mambavision::validate_geometry(input, &spec).map_err(|error| {
        Error::InvalidConfig(format!("[{}] at line {line}: {error}", section.name))
    })?;
    Ok(spec)
}

fn parse_kimi_kda_spec(section: &RawSection, input: TensorShape) -> Result<KimiKdaSpec> {
    const ALLOWED: &[&str] = &[
        "filters",
        "rank",
        "experts",
        "top_k",
        "hidden",
        "ffn_ratio",
        "temperature",
        "load_balance",
        "activation",
        "kimi_debug",
        "stream",
        "wait_stream",
        "learning_rate",
        "dont_update",
        "burnin_update",
        "train_only_bn",
        "clip",
        "grad_centr",
    ];
    let line = section.line;
    if let Some(key) = section
        .values
        .keys()
        .find(|key| !ALLOWED.contains(&key.as_str()))
    {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: unknown option `{key}`",
            section.name
        )));
    }
    if input.width == 0 || input.height == 0 || input.channels == 0 || input.batch == 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: preceding layer must output a non-empty image",
            section.name
        )));
    }
    let filters = section.get_usize("filters", input.channels)?;
    if filters != input.channels {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: filters={filters} must equal input channels={} in v1",
            section.name, input.channels
        )));
    }
    let requested_rank = section.get_i64("rank", 0)?;
    let rank = if requested_rank > 0 {
        usize::try_from(requested_rank).map_err(|_| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: rank is too large",
                section.name
            ))
        })?
    } else {
        64.min((input.channels / 4).max(1))
    }
    .min(input.channels);
    if rank == 0 || rank > 64 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: effective rank must be in 1..=min(channels,64), got {rank}",
            section.name
        )));
    }
    let experts_raw = section.get_i64("experts", 4)?.max(1);
    let experts = usize::try_from(experts_raw).map_err(|_| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: experts is too large",
            section.name
        ))
    })?;
    if experts > 64 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: experts must be <= 64, got {experts}",
            section.name
        )));
    }
    let top_k_raw = section.get_i64("top_k", 2)?;
    if top_k_raw < 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: top_k must be in 0..={experts}, got {top_k_raw}",
            section.name
        )));
    }
    let top_k = usize::try_from(top_k_raw).map_err(|_| {
        Error::InvalidConfig(format!(
            "[{}] at line {line}: top_k is too large",
            section.name
        ))
    })?;
    if top_k > experts {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: top_k must be in 0..={experts}, got {top_k}",
            section.name
        )));
    }
    let ffn_ratio_raw = section.get_i64("ffn_ratio", 2)?;
    let ffn_ratio = if ffn_ratio_raw < 1 {
        2
    } else {
        usize::try_from(ffn_ratio_raw).map_err(|_| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: ffn_ratio is too large",
                section.name
            ))
        })?
    };
    let hidden_raw = section.get_i64("hidden", 0)?;
    let hidden = if hidden_raw < 1 {
        input
            .channels
            .checked_mul(ffn_ratio)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "[{}] at line {line}: channels * ffn_ratio overflows",
                    section.name
                ))
            })?
            .max(1)
    } else {
        usize::try_from(hidden_raw).map_err(|_| {
            Error::InvalidConfig(format!(
                "[{}] at line {line}: hidden is too large",
                section.name
            ))
        })?
    };
    let temperature = section.get_f32("temperature", 1.0)?;
    if !temperature.is_finite() {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: temperature must be finite",
            section.name
        )));
    }
    let temperature = if temperature <= 0.0 { 1.0 } else { temperature };
    let load_balance = section.get_f32("load_balance", 0.01)?;
    if !load_balance.is_finite() {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: load_balance must be finite",
            section.name
        )));
    }
    let load_balance = load_balance.max(0.0);
    let activation = Activation::parse(&section.get_str("activation", "gelu"));
    if let Activation::Other(ref other) = activation {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: unsupported activation={other}",
            section.name
        )));
    }
    if matches!(
        activation,
        Activation::NormalizeChannels
            | Activation::NormalizeChannelsSoftmax
            | Activation::NormalizeChannelsSoftmaxMaxval
    ) {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: channel-normalizing activations are unsupported",
            section.name
        )));
    }
    let debug_interval = section.get_i64("kimi_debug", 0)?;
    if debug_interval != 0 {
        return Err(Error::InvalidConfig(format!(
            "[{}] at line {line}: kimi_debug={debug_interval} is unsupported by the Rust runtime",
            section.name
        )));
    }
    for key in ["stream", "wait_stream"] {
        let stream = section.get_i64(key, -1)?;
        if stream != -1 {
            return Err(Error::InvalidConfig(format!(
                "[{}] at line {line}: {key}={stream} is unsupported by the Rust runtime",
                section.name
            )));
        }
    }
    let spec = KimiKdaSpec {
        filters,
        rank,
        experts,
        top_k,
        hidden,
        temperature,
        load_balance,
        activation,
    };
    crate::kimi_kda::validate_geometry(input, &spec).map_err(|error| {
        Error::InvalidConfig(format!("[{}] at line {line}: {error}", section.name))
    })?;
    Ok(spec)
}

fn parse_modern_yolo_spec(section: &RawSection) -> Result<ModernYoloSpec> {
    let line = section.line;
    let kind = match section.name.as_str() {
        "yolox" => ModernYoloKind::Yolox,
        "ppyoloe" => ModernYoloKind::Ppyoloe,
        "yolonas" => ModernYoloKind::Yolonas,
        other => {
            return Err(Error::InvalidConfig(format!(
                "[{other}] at line {line} is not an anchor-free modern head"
            )));
        }
    };
    let classes = section.get_usize("classes", 20)?;
    if classes == 0 {
        return Err(Error::InvalidConfig(format!(
            "[yolox] at line {line}: classes must be > 0"
        )));
    }
    let max_boxes = section.get_usize("max", 200)?;
    if max_boxes == 0 {
        return Err(Error::InvalidConfig(format!(
            "[yolox] at line {line}: max must be > 0"
        )));
    }
    Ok(ModernYoloSpec {
        kind,
        classes,
        max_boxes,
        reg_max: section.get_usize("reg_max", 16)?,
        jitter: section.get_f32("jitter", 0.2)?,
        resize: section.get_f32("resize", 1.0)?,
        min_object_cells: section.get_f32("min_object_cells", 0.0)?,
        vfl_gamma: section.get_f32("vfl_gamma", 0.0)?,
    })
}

fn pool_dim(input: usize, padding: usize, size: usize, stride: usize) -> usize {
    ((input + padding).saturating_sub(size) / stride.max(1)) + 1
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use crate::cfg::ConfigFile;

    use super::{
        CliffordContextMode, CliffordGffnMode, CliffordGlobalOp, CliffordHigherMode,
        CliffordInteractionMode, LayerKind, LrPolicy, ModernYoloKind, Network, ObbNmsKind,
        SegmentationLoss, TensorShape,
    };

    fn build(text: &str) -> crate::error::Result<Network> {
        Network::from_config(ConfigFile::parse_str("t.cfg", text)?)
    }

    #[test]
    fn deform_conv_aliases_defaults_and_source_order_weights() {
        for alias in ["deform_conv", "deformable_convolutional"] {
            let network = build(&format!(
                "[net]\nwidth=5\nheight=4\nchannels=4\n[{alias}]\nfilters=6\ngroups=2\n"
            ))
            .unwrap_or_else(|error| panic!("[{alias}] should parse: {error}"));
            let layer = &network.layers[0];
            let LayerKind::DeformConv(spec) = &layer.kind else {
                panic!("expected DCNv1/2 for [{alias}]")
            };
            assert_eq!(layer.kind.name(), "deform_conv");
            assert_eq!(layer.output_shape, super::TensorShape::new(5, 4, 6, 1));
            assert_eq!(
                (
                    spec.filters,
                    spec.groups,
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.dilation,
                    spec.padding,
                    spec.batch_normalize,
                    spec.use_mask,
                    &spec.activation,
                ),
                (
                    6,
                    2,
                    1,
                    1,
                    1,
                    1,
                    0,
                    false,
                    true,
                    &super::Activation::Logistic
                )
            );
            let weights = layer
                .weights()
                .expect("deform_conv has serialized parameters");
            assert_eq!(
                weights
                    .entries
                    .iter()
                    .map(|entry| (entry.array.name(), entry.len))
                    .collect::<Vec<_>>(),
                vec![
                    ("biases", 6),
                    ("weights", 12),
                    ("offset_weights", 8),
                    ("offset_biases", 2),
                    ("mask_weights", 4),
                    ("mask_biases", 1),
                ]
            );
            assert_eq!(weights.parameter_count(), 33);
        }
    }

    #[test]
    fn deform_conv_enables_default_try_fix_nan_for_both_aliases() {
        for alias in ["deform_conv", "deformable_convolutional"] {
            let default = build(&format!(
                "[net]\nwidth=2\nheight=2\nchannels=1\n[{alias}]\n"
            ))
            .unwrap();
            assert!(
                default.net.try_fix_nan,
                "[{alias}] should enable try_fix_nan"
            );
            let disabled = build(&format!(
                "[net]\nwidth=2\nheight=2\nchannels=1\ntry_fix_nan=0\n[{alias}]\n"
            ))
            .unwrap();
            assert!(!disabled.net.try_fix_nan);
            let enabled = build(&format!(
                "[net]\nwidth=2\nheight=2\nchannels=1\ntry_fix_nan=1\n[{alias}]\n"
            ))
            .unwrap();
            assert!(enabled.net.try_fix_nan);
        }
    }

    #[test]
    fn deform_conv_parses_stride_padding_mask_and_dilation_precedence() {
        let network = build(
            "[net]\nwidth=8\nheight=7\nchannels=4\n\
             [deform_conv]\nfilters=6\ngroups=2\nsize=3\nstride=2\n\
             stride_x=1\nstride_y=3\ndilation=2\npadding=2\npad=1\n\
             use_mask=0\nmask=1\nactivation=linear\n",
        )
        .expect("valid deform_conv geometry should parse");
        let layer = &network.layers[0];
        let LayerKind::DeformConv(spec) = &layer.kind else {
            panic!("expected deform_conv")
        };
        assert_eq!(
            (
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.use_mask
            ),
            (1, 3, 2, 1, false)
        );
        assert_eq!(layer.output_shape, super::TensorShape::new(6, 2, 6, 1));
        assert_eq!(
            layer
                .weights()
                .unwrap()
                .entries
                .iter()
                .map(|entry| (entry.array.name(), entry.len))
                .collect::<Vec<_>>(),
            vec![
                ("biases", 6),
                ("weights", 108),
                ("offset_weights", 648),
                ("offset_biases", 18),
            ]
        );

        let padded = build(
            "[net]\nwidth=8\nheight=7\nchannels=4\n\
             [deformable_convolutional]\nfilters=4\nsize=3\nstride=2\n\
             stride_x=1\nstride_y=3\npadding=2\nmask=0\n",
        )
        .unwrap();
        assert_eq!(
            padded.layers[0].output_shape,
            super::TensorShape::new(10, 3, 4, 1)
        );
        let LayerKind::DeformConv(spec) = &padded.layers[0].kind else {
            unreachable!()
        };
        assert_eq!(spec.padding, 2);
        assert!(!spec.use_mask);

        let size_one = build(
            "[net]\nwidth=4\nheight=4\nchannels=1\n\
             [deform_conv]\nsize=1\ndilation=7\nuse_mask=1\nmask=0\n",
        )
        .unwrap();
        let LayerKind::DeformConv(spec) = &size_one.layers[0].kind else {
            unreachable!()
        };
        assert_eq!(spec.dilation, 1);
        assert!(spec.use_mask);
    }

    #[test]
    fn deform_conv_batchnorm_weights_match_source_order() {
        let network = build(
            "[net]\nwidth=5\nheight=4\nchannels=4\n\
             [deform_conv]\nfilters=6\ngroups=2\nsize=3\nbatch_normalize=1\nmask=1\n",
        )
        .unwrap();
        let weights = network.layers[0].weights().unwrap();
        assert_eq!(
            weights
                .entries
                .iter()
                .map(|entry| (entry.array.name(), entry.len))
                .collect::<Vec<_>>(),
            vec![
                ("biases", 6),
                ("scales", 6),
                ("rolling_mean", 6),
                ("rolling_variance", 6),
                ("weights", 108),
                ("offset_weights", 648),
                ("offset_biases", 18),
                ("mask_weights", 324),
                ("mask_biases", 9),
            ]
        );
        assert_eq!(weights.parameter_count(), 1131);
    }

    #[test]
    fn deform_conv_rejects_unknown_options_unsupported_modes_and_bad_geometry() {
        for (options, expected) in [
            ("mystery=1", "unknown option `mystery`"),
            ("cbn=1", "cross-batch normalization is unsupported"),
            (
                "xnor=1",
                "binary, xnor, and bin_output modes are unsupported",
            ),
            ("antialiasing=1", "antialiasing is unsupported"),
            (
                "assisted_excitation=1",
                "assisted_excitation is unsupported",
            ),
            (
                "share_index=-1",
                "shared deformable-convolution layers are unsupported",
            ),
            (
                "coordconv=1",
                "specialized convolution transforms are unsupported",
            ),
            ("stream=2", "non-default stream execution is unsupported"),
            ("grad_centr=1", "grad_centr is unsupported"),
            (
                "groups=3",
                "groups must divide both input channels and filters",
            ),
            (
                "stride=0",
                "filters, groups, size, and stride must be positive",
            ),
            ("padding=-1", "padding must be non-negative"),
            ("size=9", "invalid or empty output geometry"),
        ] {
            let cfg = format!("[net]\nwidth=4\nheight=4\nchannels=4\n[deform_conv]\n{options}\n");
            let error = build(&cfg).expect_err("invalid deform_conv config must reject");
            assert!(error.to_string().contains(expected), "{options}: {error}");
        }
    }

    #[test]
    fn dcnv4_aliases_defaults_and_weight_metadata() {
        for alias in ["dcnv4", "dcnv4_convolutional"] {
            let network = build(&format!(
                "[net]\nwidth=5\nheight=4\nchannels=8\n[{alias}]\n"
            ))
            .unwrap_or_else(|error| panic!("[{alias}] should parse: {error}"));
            let layer = &network.layers[0];
            let LayerKind::Dcnv4(spec) = &layer.kind else {
                panic!("expected DCNv4 for [{alias}]")
            };
            assert_eq!(layer.kind.name(), "dcnv4");
            assert_eq!(layer.output_shape, super::TensorShape::new(5, 4, 8, 1));
            assert_eq!(
                (
                    spec.filters,
                    spec.groups,
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.dilation,
                    spec.padding,
                    spec.batch_normalize,
                    spec.offset_scale,
                ),
                (8, 1, 1, 1, 1, 1, 0, false, 1.0)
            );
            assert_eq!(
                (
                    spec.remove_center,
                    spec.d_stride,
                    spec.block_thread,
                    spec.softmax,
                    &spec.activation,
                ),
                (false, 8, 256, false, &super::Activation::Logistic)
            );
            let weights = layer.weights().expect("DCNv4 has serialized parameters");
            assert_eq!(
                weights
                    .entries
                    .iter()
                    .map(|entry| (entry.array.name(), entry.len))
                    .collect::<Vec<_>>(),
                vec![
                    ("biases", 8),
                    ("weights", 64),
                    ("offset_weights", 64),
                    ("offset_biases", 8),
                ]
            );
            assert_eq!(weights.parameter_count(), 144);
        }
    }

    #[test]
    fn dcnv4_preserves_channels_and_matches_padding_stride_semantics() {
        let network = build(
            "[net]\nwidth=8\nheight=7\nchannels=4\n\
             [dcnv4]\nfilters=9\ngroups=2\nsize=3\nstride=1\n\
             stride_x=2\nstride_y=3\ndilation=1\npadding=2\npad=1\nblock_thread=3\n",
        )
        .expect("DCNv4 geometry should parse");
        let LayerKind::Dcnv4(spec) = &network.layers[0].kind else {
            panic!("expected DCNv4")
        };
        assert_eq!(
            network.layers[0].output_shape,
            super::TensorShape::new(4, 3, 4, 1)
        );
        assert_eq!(
            (spec.filters, spec.groups, spec.padding, spec.block_thread),
            (4, 2, 1, 4)
        );
    }

    #[test]
    fn dcnv4_batchnorm_weight_blocks_match_darknet_order() {
        let network = build(
            "[net]\nwidth=5\nheight=4\nchannels=4\n\
             [dcnv4]\nfilters=4\ngroups=2\nsize=3\nremove_center=1\n\
             batch_normalize=1\nsoftmax=1\n",
        )
        .expect("valid DCNv4 BN config should parse");
        let weights = network.layers[0].weights().unwrap();
        assert_eq!(
            weights
                .entries
                .iter()
                .map(|entry| (entry.array.name(), entry.len))
                .collect::<Vec<_>>(),
            vec![
                ("biases", 4),
                ("scales", 4),
                ("rolling_mean", 4),
                ("rolling_variance", 4),
                ("weights", 72),
                ("offset_weights", 1728),
                ("offset_biases", 48),
            ]
        );
        assert_eq!(weights.parameter_count(), 1864);
    }

    #[test]
    fn dcnv4_enables_default_try_fix_nan_but_respects_explicit_net_value() {
        let config = |net_option: &str| {
            ConfigFile::parse_str(
                "dcnv4-try-fix-nan.cfg",
                &format!(
                    "[net]\nwidth=2\nheight=2\nchannels=1\n{net_option}\n\
                     [dcnv4]\nsize=1\nactivation=linear\n"
                ),
            )
            .unwrap()
        };
        assert!(Network::from_config(config("")).unwrap().net.try_fix_nan);
        assert!(
            !Network::from_config(config("try_fix_nan=0"))
                .unwrap()
                .net
                .try_fix_nan
        );
        assert!(
            Network::from_config(config("try_fix_nan=1"))
                .unwrap()
                .net
                .try_fix_nan
        );
    }

    #[test]
    fn dcnv4_rejects_unknown_options_and_invalid_geometry() {
        for (cfg, expected) in [
            (
                "[net]\nwidth=4\nheight=4\nchannels=4\n[dcnv4]\nmystery=1\n",
                "unknown option `mystery`",
            ),
            (
                "[net]\nwidth=4\nheight=4\nchannels=4\n[dcnv4]\ngroups=3\n",
                "groups=3",
            ),
            (
                "[net]\nwidth=4\nheight=4\nchannels=4\n[dcnv4]\nsize=1\nremove_center=1\n",
                "remove_center",
            ),
            (
                "[net]\nwidth=4\nheight=4\nchannels=4\n[dcnv4]\nactivation=unknown\n",
                "unsupported activation",
            ),
            (
                "[net]\nwidth=4\nheight=4\nchannels=4\n[dcnv4]\noffset_scale=nan\n",
                "offset_scale must be finite",
            ),
            (
                "[net]\nwidth=2\nheight=2\nchannels=4\n[dcnv4]\nsize=5\n",
                "output",
            ),
        ] {
            let error = build(cfg).expect_err("invalid DCNv4 config must fail");
            assert!(
                error.to_string().contains(expected),
                "expected error containing {expected:?}, got {error}"
            );
        }
    }

    #[test]
    fn kimi_kda_aliases_defaults_identity_shape_and_weight_metadata() {
        for alias in ["kimi_kda", "kimi_delta_attention"] {
            let network = build(&format!(
                "[net]\nwidth=3\nheight=2\nchannels=8\n[{alias}]\n"
            ))
            .unwrap();
            let layer = &network.layers[0];
            let LayerKind::KimiKda(spec) = &layer.kind else {
                panic!("expected Kimi KDA for [{alias}]")
            };
            assert_eq!(
                (
                    spec.filters,
                    spec.rank,
                    spec.experts,
                    spec.top_k,
                    spec.hidden,
                    spec.temperature,
                    spec.load_balance,
                ),
                (8, 2, 4, 2, 16, 1.0, 0.01)
            );
            assert_eq!(spec.activation, super::Activation::Gelu);
            assert_eq!(layer.kind.name(), "kimi_kda");
            assert_eq!(layer.output_shape, super::TensorShape::new(3, 2, 8, 1));
            let weights = layer.weights().unwrap();
            assert_eq!(weights.entries.len(), 2);
            assert_eq!(weights.entries[0].array.name(), "biases");
            assert_eq!(weights.entries[1].array.name(), "weights");
            let params = crate::KimiKdaParams::zeroed(8, 2, 4, 16).unwrap();
            assert_eq!(weights.parameter_count(), params.parameter_count());
        }
    }

    #[test]
    fn kimi_kda_matches_source_fallback_defaults_and_rejects_unsupported_options() {
        let network = build(
            "[net]\nwidth=2\nheight=2\nchannels=6\n[kimi_kda]\n\
             rank=-1\nexperts=0\ntop_k=0\nffn_ratio=0\nhidden=0\n\
             temperature=0\nload_balance=-0.5\nkimi_debug=0\nstream=-1\nwait_stream=-1\n",
        )
        .unwrap();
        let LayerKind::KimiKda(spec) = &network.layers[0].kind else {
            panic!("expected Kimi KDA")
        };
        assert_eq!(
            (
                spec.filters,
                spec.rank,
                spec.experts,
                spec.top_k,
                spec.hidden,
                spec.temperature,
                spec.load_balance,
            ),
            (6, 1, 1, 0, 12, 1.0, 0.0)
        );
        for options in [
            "mystery=1",
            "filters=7",
            "rank=65",
            "experts=65",
            "top_k=-1",
            "top_k=5",
            "activation=unknown",
            "activation=normalize_channels",
            "kimi_debug=1",
            "stream=0",
            "temperature=nan",
            "load_balance=nan",
        ] {
            let channels = if options == "rank=65" { 128 } else { 8 };
            let cfg =
                format!("[net]\nwidth=2\nheight=2\nchannels={channels}\n[kimi_kda]\n{options}\n");
            assert!(
                build(&cfg).is_err(),
                "accepted invalid Kimi KDA option: {options}"
            );
        }
    }

    #[test]
    fn mambavision_aliases_defaults_shape_and_weight_metadata() {
        for alias in ["mambavision", "mambavision_block"] {
            let network = build(&format!(
                "[net]\nwidth=3\nheight=2\nchannels=8\n[{alias}]\n"
            ))
            .unwrap();
            let layer = &network.layers[0];
            let LayerKind::MambaVision(spec) = &layer.kind else {
                panic!("expected MambaVision for [{alias}]")
            };
            assert_eq!(
                (
                    spec.filters,
                    spec.state,
                    spec.conv_size,
                    spec.dt_rank,
                    spec.ffn_ratio,
                    spec.msa_enabled,
                    spec.msa_block_size,
                    spec.msa_topk,
                    spec.msa_heads,
                    spec.causal,
                ),
                (8, 8, 3, 1, 4, false, 16, 4, 1, false)
            );
            assert_eq!(spec.activation, super::Activation::Gelu);
            assert_eq!(layer.kind.name(), "mambavision");
            assert_eq!(layer.output_shape, super::TensorShape::new(3, 2, 8, 1));
            let weights = layer.weights().unwrap();
            assert_eq!(
                weights.entries.first().unwrap().array.name(),
                "mambavision_in_proj_weights"
            );
            assert_eq!(
                weights.entries.last().unwrap().array.name(),
                "mambavision_ffn_b2"
            );
            assert_eq!(
                weights.parameter_count(),
                crate::mambavision::MambaVisionParams::zeroed(8, 8, 8, 3, 1, 4)
                    .unwrap()
                    .parameter_count()
            );
        }

        let network = build(
            "[net]\nwidth=4\nheight=2\nchannels=8\n\
             [mambavision]\nfilters=10\nstate=4\nd_state=5\nconv_size=5\nsize=7\n\
             dt_rank=2\nffn_ratio=2\nactivation=linear\nmsa=0\nminimax_sparse_attention=1\n\
             msa_block=6\nmsa_block_size=3\nmsa_topk=5\nmsa_top_blocks=2\nmsa_heads=3\ncausal=1\n",
        )
        .unwrap();
        let LayerKind::MambaVision(spec) = &network.layers[0].kind else {
            panic!("expected MambaVision")
        };
        assert_eq!(
            (
                spec.filters,
                spec.state,
                spec.conv_size,
                spec.dt_rank,
                spec.ffn_ratio
            ),
            (10, 4, 5, 2, 2)
        );
        assert_eq!(
            (
                spec.msa_enabled,
                spec.msa_block_size,
                spec.msa_topk,
                spec.msa_heads,
                spec.causal
            ),
            (false, 6, 5, 1, true)
        );
        assert_eq!(spec.activation, super::Activation::Linear);
    }

    #[test]
    fn mambavision_rejects_unknown_options_and_invalid_geometry() {
        for options in [
            "mystery=1",
            "filters=3",
            "state=0",
            "conv_size=2",
            "ffn_ratio=0",
            "dt_rank=bad",
            "activation=unknown",
            "stream=0",
        ] {
            let cfg = format!("[net]\nwidth=2\nheight=2\nchannels=4\n[mambavision]\n{options}\n");
            assert!(
                build(&cfg).is_err(),
                "accepted invalid MambaVision option: {options}"
            );
        }
    }

    #[test]
    fn tucker_attention_aliases_defaults_weights_and_rank_precedence() {
        for alias in ["tucker", "tucker_attention", "tucker_attn"] {
            let network = build(&format!(
                "[net]\nwidth=3\nheight=2\nchannels=8\n[{alias}]\n"
            ))
            .unwrap();
            let layer = &network.layers[0];
            let LayerKind::TuckerAttention(spec) = &layer.kind else {
                panic!("expected Tucker attention for [{alias}]")
            };
            assert_eq!(
                (
                    spec.filters,
                    spec.size,
                    spec.heads,
                    spec.rank_q,
                    spec.rank_k,
                    spec.rank_v,
                    spec.rank_o,
                    spec.msa_enabled,
                    spec.msa_block_size,
                    spec.msa_topk,
                ),
                (8, 7, 4, 1, 1, 1, 1, false, 7, 2)
            );
            assert_eq!(spec.activation, super::Activation::Linear);
            assert_eq!(layer.kind.name(), "tucker_attention");
            assert_eq!(layer.output_shape, layer.input_shape);
            let entries = &layer.weights().unwrap().entries;
            assert_eq!(entries.len(), 2);
            assert_eq!(entries[0].array.name(), "biases");
            assert_eq!(entries[0].len, 8);
            assert_eq!(entries[1].array.name(), "weights");
            assert_eq!(entries[1].len, 64);
        }

        let network = build(
            "[net]\nwidth=3\nheight=2\nchannels=8\n\
             [tucker_attention]\nrank=2\nrank_q=3\nrank_k=4\n\
             msa=1\nminimax_sparse_attention=0\n\
             msa_block=4\nmsa_block_size=2\nmsa_topk=5\nmsa_top_blocks=1\n",
        )
        .unwrap();
        let LayerKind::TuckerAttention(spec) = &network.layers[0].kind else {
            panic!("expected Tucker attention")
        };
        assert_eq!(
            (spec.rank_q, spec.rank_k, spec.rank_v, spec.rank_o),
            (3, 4, 2, 2)
        );
        assert!(spec.msa_enabled);
        assert_eq!((spec.msa_block_size, spec.msa_topk), (4, 5));
    }

    #[test]
    fn tucker_attention_rejects_unknown_and_invalid_geometry() {
        for options in [
            "filters=4",
            "size=0",
            "heads=0",
            "heads=3",
            "rank_q=0",
            "rank=-1\nrank_q=0",
            "unknown_option=1",
            "activation=normalize_channels",
            "stream=0",
        ] {
            let text = format!("[net]\nwidth=3\nheight=2\nchannels=8\n[tucker]\n{options}\n");
            assert!(build(&text).is_err(), "accepted invalid options: {options}");
        }
    }

    #[test]
    fn swin_aliases_parse_defaults_and_preserve_spatial_shape() {
        for alias in ["swin", "swin_block", "transformer", "transformer_block"] {
            let network = build(&format!(
                "[net]\nwidth=5\nheight=3\nchannels=8\n[{alias}]\n"
            ))
            .unwrap();
            let layer = &network.layers[0];
            let LayerKind::Swin(spec) = &layer.kind else {
                panic!("expected Swin for [{alias}]")
            };
            assert_eq!(
                (
                    spec.filters,
                    spec.size,
                    spec.heads,
                    spec.shift,
                    spec.ffn_ratio
                ),
                (128, 7, 4, false, 4)
            );
            assert_eq!(spec.activation, super::Activation::Gelu);
            assert_eq!(layer.kind.name(), "swin");
            assert_eq!(
                (
                    layer.output_shape.width,
                    layer.output_shape.height,
                    layer.output_shape.channels
                ),
                (5, 3, 128)
            );
        }
    }

    #[test]
    fn swin_config_and_rust_weight_order_are_strict_and_shape_aware() {
        let network = build(
            "[net]\nwidth=5\nheight=3\nchannels=4\n\
             [swin]\nfilters=6\nsize=2\nheads=2\nshift=1\nffn_ratio=3\nactivation=relu\n",
        )
        .unwrap();
        let layer = &network.layers[0];
        let LayerKind::Swin(spec) = &layer.kind else {
            panic!("expected Swin")
        };
        assert_eq!(
            (
                spec.filters,
                spec.size,
                spec.heads,
                spec.shift,
                spec.ffn_ratio
            ),
            (6, 2, 2, true, 3)
        );
        assert_eq!(spec.activation, super::Activation::Relu);
        let weights = layer.weights().expect("Swin has trainable parameters");
        let names: Vec<&str> = weights
            .entries
            .iter()
            .map(|entry| entry.array.name())
            .collect();
        assert_eq!(
            names,
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
        assert_eq!(
            weights
                .entries
                .iter()
                .map(|entry| entry.len)
                .collect::<Vec<_>>(),
            [48, 12, 24, 6, 4, 4, 6, 6, 108, 18, 108, 6, 24, 18, 12]
        );

        let equal_channels = build(
            "[net]\nwidth=4\nheight=4\nchannels=4\n\
             [transformer]\nfilters=4\nheads=2\nsize=2\n",
        )
        .unwrap();
        let equal_weights = equal_channels.layers[0].weights().unwrap();
        assert!(
            !equal_weights
                .entries
                .iter()
                .any(|entry| entry.array.name() == "swin_res_proj")
        );

        for (options, expected) in [
            (
                "filters=0",
                "filters, size, heads, and ffn_ratio must be > 0",
            ),
            ("size=0", "filters, size, heads, and ffn_ratio must be > 0"),
            ("heads=0", "filters, size, heads, and ffn_ratio must be > 0"),
            (
                "ffn_ratio=0",
                "filters, size, heads, and ffn_ratio must be > 0",
            ),
            ("shift=2", "shift must be 0 or 1"),
            ("shift=-1", "shift must be 0 or 1"),
            ("heads=3", "input channels=4 must be divisible by heads=3"),
            ("activation=unknown", "unsupported activation=unknown"),
            (
                "activation=normalize_channels",
                "channel-normalizing activations are unsupported",
            ),
            ("mystery=1", "unknown option `mystery`"),
            ("stream=0", "stream=0 is unsupported"),
        ] {
            let error = build(&format!(
                "[net]\nwidth=4\nheight=3\nchannels=4\n[swin]\n{options}\n"
            ))
            .unwrap_err()
            .to_string();
            assert!(error.contains(expected), "{options}: {error}");
        }
    }

    #[test]
    fn vit_parses_defaults_aliases_and_patch_grid_shape() {
        let default = build(
            "[net]\nwidth=8\nheight=6\nchannels=3\n\
             [vit]\ndim=8\nheads=2\npatch_size=2\npatch_stride=2\n",
        )
        .unwrap();
        let LayerKind::Vit(spec) = &default.layers[0].kind else {
            panic!("expected Vit")
        };
        assert_eq!((spec.dim, spec.mlp_dim, spec.ffn_ratio), (8, 32, 4));
        assert!(spec.learned_pos_embed && !spec.pos_init_zero);
        assert_eq!(spec.activation, super::Activation::Gelu);
        assert!(!spec.flash_attention);
        let shape = default.layers[0].output_shape;
        assert_eq!((shape.width, shape.height, shape.channels), (4, 3, 8));

        let aliases = build(
            "[net]\nwidth=8\nheight=6\nchannels=3\n\
             [vit]\nfilters=8\ndim=8\nheads=2\npatch_size=2\npatch_stride=2\n\
             ffn_ratio=3\nmlp_dim=24\npos_embed=simple\npos_init=zeros\nactivation=relu\n\
             minimax_sparse_attention=1\nmsa_block_size=2\nmsa_top_blocks=1\n",
        )
        .unwrap();
        let LayerKind::Vit(spec) = &aliases.layers[0].kind else {
            panic!("expected Vit")
        };
        assert_eq!((spec.dim, spec.mlp_dim, spec.ffn_ratio), (8, 24, 3));
        assert!(!spec.learned_pos_embed && spec.pos_init_zero);
        assert!(spec.msa_enabled);
        assert_eq!((spec.msa_block_size, spec.msa_topk), (2, 1));
        assert_eq!(
            (
                aliases.layers[0].output_shape.width,
                aliases.layers[0].output_shape.height
            ),
            (4, 3)
        );
    }

    #[test]
    fn vit_flash_attention_aliases_select_online_attention_and_reject_msa() {
        for option in ["flash_attention=1", "flash_attn=1"] {
            let net = build(&format!(
                "[net]\nwidth=4\nheight=1\nchannels=1\n[vit]\ndim=2\nheads=1\n{option}\n"
            ))
            .unwrap();
            let LayerKind::Vit(spec) = &net.layers[0].kind else {
                panic!("expected Vit")
            };
            assert!(spec.flash_attention, "{option}");
        }

        let error = build(
            "[net]\nwidth=4\nheight=1\nchannels=1\n[vit]\ndim=2\nheads=1\nflash_attention=1\nmsa=1\n",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("mutually exclusive"), "{error}");
    }

    #[test]
    fn vit_layer_weights_list_matches_the_cpp_writer_order() {
        let net = build(
            "[net]\nwidth=8\nheight=6\nchannels=3\n\
             [vit]\ndim=8\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=32\n",
        )
        .unwrap();
        let layer_weights = net.layers[0].weights().expect("vit has weights");
        let names: Vec<&str> = layer_weights
            .entries
            .iter()
            .map(|entry| entry.array.name())
            .collect();
        assert_eq!(
            names,
            [
                "vit_patch_embed",
                "vit_patch_bias",
                "biases",
                "vit_ln1_gamma",
                "vit_ln1_beta",
                "weights",
                "vit_wo",
                "vit_wo_bias",
                "vit_ln2_gamma",
                "vit_ln2_beta",
                "vit_ffn_w1",
                "vit_ffn_b1",
                "vit_ffn_w2",
                "vit_ffn_b2",
                "vit_pos_embed",
                "scales",
            ]
        );
        assert_eq!(
            layer_weights
                .entries
                .iter()
                .map(|entry| entry.len)
                .collect::<Vec<_>>(),
            [96, 8, 24, 8, 8, 192, 64, 8, 8, 8, 256, 32, 256, 8, 96, 12,]
        );
        assert_eq!(
            layer_weights.parameter_count(),
            crate::vit::vit_blob_float_count(
                &crate::vit::VitGeometry {
                    batch: 1,
                    channels: 3,
                    height: 6,
                    width: 8,
                    patch_size: 2,
                    patch_stride: 2,
                    patch_pad: 0,
                    hidden: 8,
                    heads: 2,
                    mlp_dim: 32,
                    ln_eps: 1e-5,
                    activation: super::Activation::Gelu,
                    learned_pos_embed: true,
                    pos_init_zero: false,
                    msa: None,
                    flash_attention: false,
                    vc_attention: false,
                    vc_expcast: 1,
                    vc_perm: super::VitVcPermutation::Sequence,
                },
                3,
                4,
                true
            )
            .unwrap()
        );
    }

    #[test]
    fn vit_rejects_conflicts_invalid_geometry_and_unknown_options() {
        let cfg = |layer: &str| format!("[net]\nwidth=8\nheight=6\nchannels=3\n[vit]\n{layer}\n");
        for invalid in [
            "filters=8\ndim=16",
            "dim=8\nheads=3",
            "dim=8\nheads=2\nffn_ratio=2\nmlp_dim=12",
            "dim=8\nheads=2\npatch_size=0",
            "dim=8\nheads=2\npatch_size=3\npatch_stride=2",
            "dim=8\nheads=2\npos_embed=rotary",
            "dim=8\nheads=2\npos_init=ones",
            "dim=8\nheads=2\nactivation=unknown_vit_act",
            "dim=8\nheads=2\nvc_attention=1",
            "dim=8\nheads=2\nvc=1",
            "dim=8\nheads=2\nvc_expcast=3",
            "dim=8\nheads=2\nexpcast=3",
            "dim=8\nheads=2\nvc_perm=spatial_typo",
        ] {
            let err = build(&cfg(invalid)).expect_err(invalid).to_string();
            assert!(
                err.contains("vit") || err.contains("activation"),
                "{invalid}: {err}"
            );
        }
    }

    #[test]
    fn vit_vc_attention_aliases_modes_and_permutations_parse_strictly() {
        let base = "[net]\nwidth=8\nheight=6\nchannels=3\n[vit]\ndim=8\nheads=2\n";
        let config = |options: &str| format!("{base}{options}\n");
        let parse = |options: &str| build(&config(options)).unwrap();
        let LayerKind::Vit(defaults) = &parse("flash_attention=1").layers[0].kind else {
            panic!("expected Vit")
        };
        assert!(!defaults.vc_attention);
        assert_eq!(defaults.vc_expcast, 1);
        assert_eq!(defaults.vc_perm, super::VitVcPermutation::Sequence);

        for (options, enabled, mode, permutation) in [
            (
                "flash_attention=1\nvc_attention=1\nvc_expcast=0\nvc_perm=sequence",
                true,
                0u8,
                super::VitVcPermutation::Sequence,
            ),
            (
                "flash_attn=1\nvc=1\nexpcast=2\nvc_perm=spatial",
                true,
                2u8,
                super::VitVcPermutation::Spatial8,
            ),
            (
                "flash_attention=1\nvc_attention=0\nvc=1\nvc_expcast=2\nexpcast=0\nvc_perm=window",
                false,
                2u8,
                super::VitVcPermutation::Spatial8,
            ),
        ] {
            let network = parse(options);
            let LayerKind::Vit(spec) = &network.layers[0].kind else {
                panic!("expected Vit")
            };
            assert_eq!(spec.vc_attention, enabled);
            assert_eq!(spec.vc_expcast, mode);
            assert_eq!(spec.vc_perm, permutation);
        }
        for invalid in [
            "flash_attention=1\nvc_expcast=3",
            "flash_attention=1\nvc_perm=spatial_typo",
        ] {
            assert!(build(&config(invalid)).is_err(), "{invalid}");
        }
    }

    #[test]
    fn unknown_activation_is_a_config_error_not_identity() {
        let cfg = |act: &str| {
            format!(
                "[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\nactivation={act}\n"
            )
        };
        for ok in [
            "linear",
            "loggy",
            "leaky",
            "mish",
            "swish",
            "gelu",
            "elu",
            "selu",
            "relie",
            "plse",
            "hardtan",
            "lhtan",
            "stair",
            "hard_mish",
            "revleaky",
        ] {
            assert!(build(&cfg(ok)).is_ok(), "{ok} must parse");
        }
        for bad in ["banana", "eml"] {
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
        let LayerKind::MaxPool(depth) = &net.layers[0].kind else {
            panic!()
        };
        assert!(depth.depth);
        // depth pooling keeps w x h and sets the channels to out_channels
        let s = net.layers[0].output_shape;
        assert_eq!((s.width, s.height, s.channels), (4, 6, 2));
        let LayerKind::MaxPool(plain) = &net.layers[1].kind else {
            panic!()
        };
        assert!(plain.zero_nonmax && !plain.depth);
        let s = net.layers[1].output_shape;
        assert_eq!((s.width, s.height, s.channels), (2, 3, 2)); // out_channels ignored without depth
        // out_channels larger than the input channels is rejected
        assert!(
            build(
                "[net]\nwidth=4\nheight=4\nchannels=2\n[maxpool]\nmaxpool_depth=1\nout_channels=3\n"
            )
            .is_err()
        );
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
            let cfg = format!(
                "[net]\nwidth=4\nheight=3\nchannels=2\n[convolutional]\nfilters=6\nsize=1\n[convolutional]\nfilters=5\nsize=1\n[channel_slice]\n{bad}\n"
            );
            assert!(build(&cfg).is_err(), "{bad}");
        }
    }

    #[test]
    fn channel_shuffle_requires_divisible_channels() {
        let ok =
            build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=3\n").unwrap();
        assert_eq!(ok.layers[0].output_shape.channels, 6);
        assert!(
            build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=4\n").is_err()
        );
        assert!(
            build("[net]\nwidth=2\nheight=2\nchannels=6\n[channel_shuffle]\ngroups=0\n").is_err()
        );
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
        let net =
            build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\n").unwrap();
        assert!(
            matches!(&net.layers[0].kind, LayerKind::Deconvolutional(s) if s.activation == super::Activation::Logistic)
        );
        assert!(
            build("[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\npad=1\n")
                .is_err()
        );
        assert!(
            build(
                "[net]\nwidth=3\nheight=2\nchannels=2\n[deconvolutional]\nfilters=1\npadding=1\n"
            )
            .is_err()
        );
    }

    #[test]
    fn loss_head_sections_parse_with_scale() {
        let net = build(
            "[net]\nwidth=1\nheight=1\nchannels=2\n[connected]\noutput=2\n\
             [cost]\ntype=masked\nscale=0.25\n[logistic]\n[regression]\nscale=3\n",
        )
        .unwrap();
        assert!(
            matches!(&net.layers[1].kind, LayerKind::Cost { kind, scale } if kind == "masked" && *scale == 0.25)
        );
        assert!(matches!(net.layers[2].kind, LayerKind::Logistic));
        assert!(matches!(&net.layers[3].kind, LayerKind::Regression(r) if r.scale == 3.0));
        assert_eq!(net.layers[3].output_shape, net.layers[2].output_shape);
    }

    #[test]
    fn jev_ce_keys_follow_cpp_defaults_and_validation() {
        let base = "[net]\nwidth=1\nheight=1\nchannels=12\n[connected]\noutput=12\n";
        let net = build(&format!(
            "{base}[jev_ce]\ngroups=4\ntemp=2\nsmoothing=0.1\nlayout=cm\n"
        ))
        .unwrap();
        let LayerKind::JevCe(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(
            (
                s.groups,
                s.candidates,
                s.temperature,
                s.label_smoothing,
                s.channel_major
            ),
            (4, 3, 2.0, 0.1, true)
        );
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
        let LayerKind::Segmentation(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.classes, 2);
        assert_eq!(s.loss, SegmentationLoss::SoftmaxCe);
        assert_eq!(s.max_delta, f32::MAX, "FLT_MAX means no clipping");
        assert_eq!(
            (
                s.delta_normalizer,
                s.cls_normalizer,
                s.lambda_ce,
                s.ce_normalizer
            ),
            (1.0, 1.0, 1.0, 1.0)
        );
        assert!(!s.gate_enabled);
        assert_eq!(
            (
                s.gate_t_correct,
                s.gate_warmup_iters,
                s.gate_ramp_iters,
                s.gate_max_compensation
            ),
            (0.9, 1000, 2000, 10.0)
        );
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
        let LayerKind::Segmentation(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.classes, 3);
        assert_eq!(
            (s.max_delta, s.delta_normalizer, s.cls_normalizer),
            (2.5, 0.5, 2.0)
        );
        assert_eq!((s.lambda_ce, s.ce_normalizer), (0.25, 4.0));
        assert!(s.gate_enabled);
        assert_eq!(
            (
                s.gate_t_correct,
                s.gate_warmup_iters,
                s.gate_ramp_iters,
                s.gate_max_compensation
            ),
            (0.5, 5, 10, 3.0)
        );
        // `loss_normalizer_ce` is the C++ fallback spelling of `ce_normalizer`.
        let net = build(&format!("{base}[segmentation]\nloss_normalizer_ce=7\n")).unwrap();
        let LayerKind::Segmentation(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.ce_normalizer, 7.0);
        // `segmenter` is the C++ alias of `segmentation`.
        let net = build(&format!("{base}[segmenter]\n")).unwrap();
        assert_eq!(net.layers[1].kind.name(), "segmentation");

        // classes=1 selects logistic BCE and rejects the softmax spelling.
        let net = build(&format!("[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\n[segmentation]\nclasses=1\n")).unwrap();
        let LayerKind::Segmentation(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.loss, SegmentationLoss::Bce);
        let one = "[net]\nwidth=2\nheight=2\nchannels=1\n[convolutional]\nfilters=1\nsize=1\n[segmentation]\nclasses=1\n";
        assert!(
            build(&format!("{one}loss=ce\n")).is_err(),
            "classes=1 must not take loss=ce"
        );
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
            assert!(
                build(&format!("{base}[segmentation]\n{bad}\n")).is_err(),
                "accepted {bad}"
            );
        }
    }

    #[test]
    fn yolox_keys_follow_cpp_defaults_and_validation() {
        // C++ `parse_yolox_section`: classes=20, max=200, jitter=0.2, resize=1.0.
        let base = "[net]\nwidth=4\nheight=4\nchannels=7\n[convolutional]\nfilters=7\nsize=1\n";
        let net = build(&format!("{base}[yolox]\n")).unwrap();
        let LayerKind::Yolox(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(
            (s.classes, s.max_boxes, s.jitter, s.resize),
            (20, 200, 0.2, 1.0)
        );
        assert!(net.layers[1].weights().is_none(), "the head has no weights");
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        let net = build(&format!(
            "{base}[yolox]\nclasses=3\nmax=64\njitter=0.1\nresize=1.5\n"
        ))
        .unwrap();
        let LayerKind::Yolox(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!((s.classes, s.max_boxes), (3, 64));
        assert!((s.jitter - 0.1).abs() < 1e-6 && (s.resize - 1.5).abs() < 1e-6);
        assert!(build(&format!("{base}[yolox]\nclasses=0\n")).is_err());
        assert!(build(&format!("{base}[yolox]\nmax=0\n")).is_err());
    }

    #[test]
    fn centernet_keys_follow_cpp_defaults_and_validation() {
        // C++ defaults: classes=20, max=200, center_min_radius=1, peak_nms=1,
        // anisotropic_gaussian=1, small_threshold=8, small_ref_size=32,
        // small_boost=8, scale_min_px=0, scale_max_px=FLT_MAX, gaussian_iou=0.7,
        // focal_alpha=2, focal_beta=4, hm/wh/off_normalizer=1/0.1/1,
        // max_delta=10, delta_normalizer=1.
        let base = "[net]\nwidth=4\nheight=4\nchannels=3\n[convolutional]\nfilters=3\nsize=1\n";
        let net = build(&format!("{base}[centernet]\n")).unwrap();
        let LayerKind::CenterNet(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!((s.classes, s.max_boxes), (20, 200));
        assert_eq!(
            (s.center_min_radius, s.peak_nms, s.anisotropic_gaussian),
            (1, 1, 1)
        );
        assert_eq!(
            (s.small_threshold, s.small_ref_size, s.small_boost),
            (8.0, 32.0, 8.0)
        );
        assert_eq!((s.scale_min_px, s.scale_max_px), (0.0, f32::MAX));
        assert_eq!(
            (s.gaussian_iou, s.focal_alpha, s.focal_beta),
            (0.7, 2.0, 4.0)
        );
        assert_eq!(
            (s.hm_normalizer, s.wh_normalizer, s.off_normalizer),
            (1.0, 0.1, 1.0)
        );
        assert_eq!((s.max_delta, s.delta_normalizer), (10.0, 1.0));
        assert!(net.layers[1].weights().is_none(), "the head has no weights");
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        let net = build(&format!(
            "{base}[centernet]\nclasses=3\nmax=64\ncenter_min_radius=2\npeak_nms=0\n\
             anisotropic_gaussian=0\nsmall_ref_size=16\nsmall_boost=4\nscale_min_px=4\n\
             scale_max_px=64\ngaussian_iou=0.5\nfocal_alpha=1.5\nfocal_beta=3\n\
             hm_normalizer=0.5\nwh_normalizer=0.25\noff_normalizer=2\nmax_delta=5\n\
             delta_normalizer=0.75\n"
        ))
        .unwrap();
        let LayerKind::CenterNet(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(
            (
                s.classes,
                s.max_boxes,
                s.center_min_radius,
                s.peak_nms,
                s.anisotropic_gaussian
            ),
            (3, 64, 2, 0, 0)
        );
        assert_eq!(
            (
                s.small_ref_size,
                s.small_boost,
                s.scale_min_px,
                s.scale_max_px
            ),
            (16.0, 4.0, 4.0, 64.0)
        );
        assert_eq!(
            (s.gaussian_iou, s.focal_alpha, s.focal_beta),
            (0.5, 1.5, 3.0)
        );
        assert_eq!(
            (s.hm_normalizer, s.wh_normalizer, s.off_normalizer),
            (0.5, 0.25, 2.0)
        );
        assert_eq!((s.max_delta, s.delta_normalizer), (5.0, 0.75));

        assert!(build(&format!("{base}[centernet]\nclasses=0\n")).is_err());
        assert!(build(&format!("{base}[centernet]\nmax=0\n")).is_err());
    }

    #[test]
    fn the_modern_heads_take_reg_max_and_the_nested_object_keys() {
        let base = "[net]\nwidth=2\nheight=2\nchannels=70\n[convolutional]\nfilters=70\nsize=1\n";
        let net = build(&format!("{base}[ppyoloe]\nclasses=1\nreg_max=16\n")).unwrap();
        let LayerKind::Ppyoloe(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.kind, ModernYoloKind::Ppyoloe);
        assert_eq!(s.coords(), 4 * 17);
        assert_eq!(
            (s.min_object_cells, s.vfl_gamma),
            (0.0, 0.0),
            "both default off"
        );
        let net = build(&format!(
            "{base}[yolonas]\nclasses=1\nreg_max=8\nmin_object_cells=2\nvfl_gamma=1.5\n"
        ))
        .unwrap();
        let LayerKind::Yolonas(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.kind, ModernYoloKind::Yolonas);
        assert_eq!(s.coords(), 4 * 9);
        assert_eq!((s.min_object_cells, s.vfl_gamma), (2.0, 1.5));
        // YOLOX ignores `reg_max` and uses four raw box entries.
        let net = build(
            "[net]\nwidth=2\nheight=2\nchannels=7\n[convolutional]\nfilters=7\nsize=1\n[yolox]\nclasses=2\nreg_max=16\n",
        )
        .unwrap();
        let LayerKind::Yolox(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.coords(), 4);
        // Each section reports its own name.
        assert_eq!(net.layers[1].kind.name(), "yolox");
        assert!(build(&format!("{base}[ppyoloe]\nclasses=0\n")).is_err());
        assert!(build(&format!("{base}[ppyoloe]\nmax=0\n")).is_err());
    }

    #[test]
    fn contrastive_keys_follow_cpp_defaults_and_validation() {
        // C++ defaults: classes=1000, yolo_layer=0, temperature=1.0,
        // cls_normalizer=1.0, max_delta=FLT_MAX, contrastive_neg_max=3.
        let base = "[net]\nwidth=1\nheight=1\nchannels=4\n[convolutional]\nfilters=4\nsize=1\n";
        let net = build(&format!("{base}[contrastive]\n")).unwrap();
        let LayerKind::Contrastive(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.classes, 1000);
        assert_eq!(s.yolo_layer, 0, "0 selects the classifier branch");
        assert_eq!(
            (s.temperature, s.cls_normalizer, s.max_delta),
            (1.0, 1.0, f32::MAX)
        );
        assert_eq!(s.contrastive_neg_max, 3);
        assert!(net.layers[1].weights().is_none());
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        let net = build(&format!(
            "{base}[contrastive]\nclasses=3\nyolo_layer=-1\ntemperature=0.5\ncls_normalizer=2\n\
             max_delta=4\ncontrastive_neg_max=7\n"
        ))
        .unwrap();
        let LayerKind::Contrastive(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.classes, 3);
        assert_eq!(s.yolo_layer, -1, "a negative index is relative, as in C++");
        assert_eq!(
            (
                s.temperature,
                s.cls_normalizer,
                s.max_delta,
                s.contrastive_neg_max
            ),
            (0.5, 2.0, 4.0, 7)
        );

        // `[net] adversarial` switches the fallback label from `b / 2` to `b % 2`.
        let net = build(&format!(
            "[net]\nwidth=1\nheight=1\nchannels=2\nlabel_smooth_eps=0.05\nadversarial=1\n\
             [convolutional]\nfilters=2\nsize=1\n[contrastive]\n"
        ))
        .unwrap();
        assert!(net.net.adversarial);
        assert!((net.net.label_smooth_eps - 0.05).abs() < 1e-6);
        let net = build(&format!("{base}[contrastive]\n")).unwrap();
        assert!(!net.net.adversarial);
        assert_eq!(net.net.label_smooth_eps, 0.0);
    }

    #[test]
    fn omd_loss_keys_follow_cpp_defaults_aliases_and_validation() {
        // C++ defaults: classes=0, max_boxes=200, truth_size=6,
        // cls_normalizer/lambda=0.03, margin=0.5, the three weights 1.0,
        // small_threshold=0, small_boost=1, max_samples=128,
        // background_samples=0, pool=0, eps=1e-6, max_delta=10.
        let base = "[net]\nwidth=2\nheight=2\nchannels=3\n[convolutional]\nfilters=3\nsize=1\n";
        let net = build(&format!("{base}[omd_loss]\n")).unwrap();
        let LayerKind::OmdLoss(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!((s.classes, s.max_boxes, s.truth_size), (0, 200, 6));
        assert_eq!((s.cls_normalizer, s.margin, s.max_delta), (0.03, 0.5, 10.0));
        assert_eq!(
            (s.positive_weight, s.negative_weight, s.background_weight),
            (1.0, 1.0, 1.0)
        );
        assert_eq!(
            (s.small_threshold, s.small_boost, s.max_samples),
            (0.0, 1.0, 128)
        );
        assert_eq!((s.background_samples, s.pool, s.eps), (0, 0, 1e-6));
        assert!(!s.show_details);
        assert!(
            net.layers[1].weights().is_none(),
            "the layer has no weights"
        );
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        // Primary key wins over its alias; the loss weight has three spellings.
        for (keys, want) in [("max_boxes=7\nmax=9", 7usize), ("max=9", 9)] {
            let net = build(&format!("{base}[omd_loss]\n{keys}\n")).unwrap();
            let LayerKind::OmdLoss(s) = &net.layers[1].kind else {
                panic!()
            };
            assert_eq!(s.max_boxes, want, "{keys}");
        }
        for (key, want) in [
            ("lambda", 0.03f32),
            ("normalizer", 0.03),
            ("cls_normalizer", 0.03),
        ] {
            let net = build(&format!("{base}[omd_loss]\n{key}=0.5\n")).unwrap();
            let LayerKind::OmdLoss(s) = &net.layers[1].kind else {
                panic!()
            };
            assert!((s.cls_normalizer - 0.5).abs() < 1e-6, "{key}");
            let _ = want;
        }
        let net = build(&format!(
            "{base}[omd_loss]\npos_weight=2\nbg_samples=4\nsmall_threshold_px=8\n"
        ))
        .unwrap();
        let LayerKind::OmdLoss(s) = &net.layers[1].kind else {
            panic!()
        };
        assert_eq!(s.positive_weight, 2.0);
        assert_eq!(s.background_samples, 4);
        assert_eq!(s.small_threshold, 8.0);

        // C++ validation.
        assert!(build(&format!("{base}[omd_loss]\nmax_samples=1\n")).is_err());
        assert!(build(&format!("{base}[omd_loss]\nmargin=-0.1\n")).is_err());
        assert!(build(&format!("{base}[omd_loss]\nmargin=2.5\n")).is_err());
        assert!(
            build(&format!("{base}[omd_loss]\nmargin=2\n")).is_ok(),
            "2 is inclusive"
        );
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
        assert_eq!(
            net.layers[1].output_shape.width * net.layers[1].output_shape.height,
            4
        );
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
    fn muon_config_enables_adamw_and_rejects_conflicting_or_invalid_values() {
        let network = build(
            r#"[net]
width=1
height=1
channels=2
muon=1
[connected]
output=2
"#,
        )
        .expect("Muon config should parse");
        assert!(network.net.muon);
        assert!(!network.net.adam);
        assert!(network.net.decoupled_decay);
        assert_eq!(network.net.muon_momentum, 0.95);

        for cfg in [
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=2\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=1\nadam=1\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=1\ndecoupled_decay=0\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=1\nmuon_momentum=1\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon_momentum=0.9\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=1\nb1=1\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=1\neps=0\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=1\ndecay=-0.1\n",
        ] {
            assert!(build(cfg).is_err(), "invalid config should reject: {cfg}");
        }

        let explicit = build(
            "[net]\nwidth=1\nheight=1\nchannels=1\nmuon=1\nmuon_momentum=0.8\nb1=0.8\nb2=0.95\neps=0.001\ndecoupled_decay=1\n",
        )
        .expect("explicit Muon config should parse");
        assert_eq!(explicit.net.muon_momentum, 0.8);
        assert_eq!(
            (
                explicit.net.adam_b1,
                explicit.net.adam_b2,
                explicit.net.adam_eps
            ),
            (0.8, 0.95, 0.001)
        );
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
    fn eml_convolutional_parses_source_defaults_and_validates_geometry() {
        let default = build(
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\n\
             [eml_convolutional]\nfilters=2\nsize=1\n",
        )
        .expect("default EML config should parse");
        let LayerKind::EmlConvolutional(spec) = &default.layers[0].kind else {
            panic!("expected EML convolution");
        };
        assert!(spec.branch.batch_normalize);
        assert_eq!(spec.branch.activation, super::Activation::Linear);
        assert_eq!(
            (spec.eml_clamp, spec.eml_eps, spec.eml_scale),
            (4.0, 1e-6, 0.1)
        );
        assert!(spec.residual);
        assert_eq!(
            default.layers[0].output_shape,
            default.layers[0].input_shape
        );

        let explicit = build(
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\n\
             [eml_convolutional]\nfilters=1\nsize=1\nbatch_normalize=0\nresidual=0\n\
             eml_clamp=2.5\neml_eps=0.001\neml_scale=0.3\n",
        )
        .expect("explicit EML config should parse");
        let LayerKind::EmlConvolutional(spec) = &explicit.layers[0].kind else {
            panic!("expected EML convolution");
        };
        assert!(!spec.branch.batch_normalize && !spec.residual);
        assert_eq!(
            (spec.eml_clamp, spec.eml_eps, spec.eml_scale),
            (2.5, 0.001, 0.3)
        );

        let residual_mismatch = build(
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\n\
             [eml_convolutional]\nfilters=1\nsize=1\n",
        )
        .expect("C++ skips residual addition when input/output sizes differ");
        assert!(matches!(
            residual_mismatch.layers[0].kind,
            LayerKind::EmlConvolutional(_)
        ));
        let invalid_groups = build(
            "[net]\nwidth=2\nheight=2\nchannels=2\nbatch=1\n\
             [eml_convolutional]\nfilters=3\nsize=1\ngroups=2\nresidual=0\n",
        );
        assert!(invalid_groups.is_err(), "groups must divide filters");
    }

    #[test]
    fn parses_obb_yolo_for_inference_with_source_shape_checks() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "obb.cfg",
                "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
            )
            .expect("cfg parses"),
        )
        .expect("valid OBB cfg builds");
        assert!(network.net.obb);
        assert_eq!(network.layers[0].kind.name(), "obb_yolo");
        let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        assert_eq!(spec.classes, 1);
        assert_eq!(spec.num, 1);
        assert_eq!(spec.mask, vec![0]);
        assert_eq!(spec.anchors, vec![4.0, 2.0]);
        assert_eq!(spec.nms_kind, ObbNmsKind::Obb);
        assert_eq!(network.layers[0].output_shape.channels, 8);

        let explicit = Network::from_config(
            ConfigFile::parse_str(
                "obb-explicit-nms.cfg",
                "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\nnms_kind=OBB\n",
            )
            .expect("explicit OBB NMS cfg parses"),
        )
        .expect("case-insensitive obb NMS kind is accepted");
        assert!(matches!(
            &explicit.layers[0].kind,
            LayerKind::ObbYolo(spec) if spec.nms_kind == ObbNmsKind::Obb
        ));
        for (kind, expected) in [
            ("default", ObbNmsKind::Default),
            ("cornersnms", ObbNmsKind::Corners),
            ("greedynms", ObbNmsKind::Greedy),
            ("diounms", ObbNmsKind::Diou),
            ("obb_nms", ObbNmsKind::Obb),
            ("obbnms", ObbNmsKind::Obb),
            ("obb_w", ObbNmsKind::Weighted),
            ("obb_wnms", ObbNmsKind::Weighted),
        ] {
            let cfg = format!(
                "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\nnms_kind={kind}\n"
            );
            let network = Network::from_config(
                ConfigFile::parse_str("obb-nms-kind.cfg", &cfg).expect("NMS cfg parses"),
            )
            .expect("supported NMS kind builds");
            assert!(matches!(
                &network.layers[0].kind,
                LayerKind::ObbYolo(spec) if spec.nms_kind == expected
            ));
        }
        let beta_cfg = Network::from_config(
            ConfigFile::parse_str(
                "obb-beta-nms.cfg",
                "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\nnms_kind=diounms\nbeta_nms=1.25\n",
            )
            .expect("DIoU beta cfg parses"),
        )
        .expect("DIoU beta cfg builds");
        assert!(matches!(
            &beta_cfg.layers[0].kind,
            LayerKind::ObbYolo(spec) if spec.beta_nms == 1.25
        ));

        for cfg in [
            "[net]\nwidth=4\nheight=2\nchannels=8\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
            "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\ncoords=5\nclasses=1\nnum=1\nanchors=4,2\n",
            "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nmask=1\nanchors=4,2\n",
            "[net]\nwidth=4\nheight=2\nchannels=7\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
            "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\nbeta_nms=NaN\n",
        ] {
            assert!(
                Network::from_config(
                    ConfigFile::parse_str("bad-obb.cfg", cfg).expect("cfg parses")
                )
                .is_err(),
                "invalid OBB cfg should fail: {cfg}"
            );
        }
    }

    #[test]
    fn carpn_refine_cfg_resolves_sources_and_derives_capacity_and_weights() {
        let network = build(
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
                nms_kind=diounms
                beta_nms=1.25

                [obb_refine]
                proposal_topk_train=3
                proposal_topk_infer=4
                proposal_jitter=0
                sample_grid=2
            "#,
        )
        .expect("valid CARPN refine cfg builds");

        assert_eq!(network.layers[3].kind.name(), "obb_refine");
        let LayerKind::ObbRefine(spec) = &network.layers[3].kind else {
            panic!("expected CARPN refine layer");
        };
        assert_eq!(spec.feature_layer, 1); // from=-2, independently of proposal source
        assert_eq!(spec.proposal_layer, 2); // -1 resolves to nearest prior OBB head
        assert_eq!(spec.classes, 1);
        assert_eq!(spec.max_boxes, 2);
        assert_eq!(spec.proposal_topk_train, 3);
        assert_eq!(spec.proposal_topk_infer, 4);
        assert_eq!(spec.proposal_jitter, 0);
        assert_eq!(spec.sample_grid, 2);
        assert_eq!(spec.proposal_capacity, 4);
        assert_eq!(spec.inputs, 2 * 2 * 8);
        assert_eq!(spec.front_range, 1.0);
        assert_eq!(spec.proposal_thresh, 0.01);
        assert_eq!(spec.quality_normalizer, 0.0);
        assert_eq!(spec.anchor_scales, [1.0, 2.0, 4.0]);
        assert_eq!(spec.nms_kind, ObbNmsKind::Diou);
        assert_eq!(spec.beta_nms, 1.25);
        assert_eq!(
            network.layers[3].output_shape,
            super::TensorShape::new(4, 1, 5, 1)
        );
        let weights = network.layers[3]
            .weights()
            .expect("refine layer has weights");
        assert_eq!(
            weights
                .entries
                .iter()
                .map(|entry| (entry.array.name(), entry.len))
                .collect::<Vec<_>>(),
            [("biases", 5), ("weights", 5 * 2 * 2 * 8)]
        );
        assert_eq!(weights.parameter_count(), 5 + 5 * 2 * 2 * 8);

        let absolute = build(
            r#"
                [net]
                width=1
                height=1
                channels=1
                obb=1
                [convolutional]
                filters=8
                size=1
                batch_normalize=0
                [convolutional]
                filters=8
                size=1
                batch_normalize=0
                [obb_yolo]
                classes=1
                num=1
                anchors=1,1
                [obb_refine]
                from=0
                proposal_from=2
                sample_grid=1
                proposal_capacity=1
                anchor_scales=0.5,2,4,8
            "#,
        )
        .expect("absolute source indexes build");
        let LayerKind::ObbRefine(absolute_spec) = &absolute.layers[3].kind else {
            panic!("expected CARPN refine layer");
        };
        assert_eq!(absolute_spec.feature_layer, 0);
        assert_eq!(absolute_spec.proposal_layer, 2);
        assert_eq!(absolute_spec.anchor_scales, [0.5, 2.0, 4.0]);
    }

    #[test]
    fn carpn_refine_cfg_rejects_invalid_sources_and_options() {
        let valid_prefix = r#"
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
        "#;
        for options in [
            "from=-5",
            "proposal_from=0",
            "proposal_from=-2",
            "classes=2",
            "representation=quad",
            "loss=unsupported",
            "front_range=0",
            "front_range=NaN",
            "proposal_capacity=0",
        ] {
            let cfg = format!("{valid_prefix}\n[obb_refine]\n{options}\n");
            assert!(
                build(&cfg).is_err(),
                "invalid CARPN option accepted: {options}"
            );
        }

        let no_proposal = r#"
            [net]
            width=4
            height=4
            channels=1
            obb=1
            [convolutional]
            filters=8
            size=1
            [convolutional]
            filters=8
            size=1
            [obb_refine]
        "#;
        assert!(
            build(no_proposal).is_err(),
            "a prior OBB proposal head is required"
        );

        let no_obb_flag = r#"
            [net]
            width=4
            height=4
            channels=1
            [convolutional]
            filters=8
            size=1
            [obb_refine]
        "#;
        assert!(build(no_obb_flag).is_err(), "[net] obb=1 is required");

        let overflow = format!(
            "{valid_prefix}\n[obb_yolo]\nmax={}\n[obb_refine]\nproposal_jitter=1\n",
            usize::MAX
        );
        assert!(
            build(&overflow).is_err(),
            "default proposal capacity must not overflow"
        );
    }

    #[test]
    fn obb_yolo_parses_class_multipliers_from_per_class_counters() {
        let network = build(
            "[net]\nwidth=1\nheight=1\nchannels=9\nobb=1\n[obb_yolo]\nclasses=2\nnum=1\nanchors=1,1\ncounters_per_class=1,4\nmax_delta=1.5\n",
        )
        .expect("OBB class counters parse");
        let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        assert_eq!(spec.class_multipliers, Some(vec![1.5, 1.0]));
        let training =
            crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
                .expect("parsed class multipliers route into training assignment");
        assert_eq!(
            training.assignment_config.class_multipliers,
            spec.class_multipliers
        );

        let invalid = build(
            "[net]\nwidth=1\nheight=1\nchannels=9\nobb=1\n[obb_yolo]\nclasses=2\nnum=1\nanchors=1,1\ncounters_per_class=3\n",
        );
        assert!(invalid.is_err(), "counter cardinality must match classes");
    }

    #[test]
    fn obb_yolo_parses_and_routes_class_map_files() {
        let dir = std::env::temp_dir().join("darknet-model-obb-class-map");
        let _ = fs::create_dir_all(&dir);
        let map_path = dir.join("labels.map");
        let cfg_path = dir.join("obb.cfg");
        fs::write(&map_path, "1\n0\n").expect("map writes");
        fs::write(
            &cfg_path,
            "[net]\nwidth=1\nheight=1\nchannels=9\nobb=1\n[obb_yolo]\nclasses=2\nnum=1\nanchors=1,1\nmap=labels.map\n",
        )
        .expect("cfg writes");

        let network = Network::from_path(&cfg_path).expect("OBB map path resolves");
        let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        assert_eq!(spec.class_map, Some(vec![1, 0]));
        let training =
            crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
                .expect("parsed class map routes into training assignment");
        assert_eq!(training.assignment_config.class_map, spec.class_map);
    }

    #[test]
    fn obb_yolo_routes_covariance_cobb_and_viou_options() {
        let network = build(
            "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=giou\niou_thresh_kind=diou\nfront_loss_normalizer=0.7\ncovariance_normalizer=1.3\ncobb_normalizer=0.4\nobb_length_ratio_k=1.4\nobb_viou_weight=1.5\ncobb_box_weight=1.2\ncobb_r_weight=0.8\ncobb_s_weight=1.4\ncobb_iou_lambda=2.2\n",
        )
        .expect("OBB score options parse");
        let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        let training =
            crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
                .expect("OBB score options route into training configs");
        assert_eq!(
            training.match_config.score_config,
            training.box_delta_config.score
        );
        assert_eq!(
            training.match_config.score_config.covariance_normalizer,
            1.3
        );
        assert!(!training.match_config.score_config.ciou_obb);
        assert_eq!(spec.iou_thresh_kind, super::YoloIouLoss::Diou);
        assert_eq!(
            training.front_point_delta_config.iou_loss,
            super::YoloIouLoss::Giou
        );
        assert_eq!(
            training
                .match_config
                .score_config
                .covariance_front_length_weight,
            0.7
        );
        assert_eq!(training.match_config.score_config.cobb_normalizer, 0.4);
        assert_eq!(
            training
                .match_config
                .score_config
                .covariance_front_length_ratio_k,
            1.4
        );
        assert_eq!(training.match_config.score_config.viou_weight, 1.0);
        assert_eq!(
            training.match_config.score_config.cobb_loss,
            crate::obb_geometry::ObbCobbLossConfig {
                box_weight: 1.2,
                ratio_weight: 0.8,
                score_weight: 1.4,
                iou_lambda: 2.2,
            }
        );
    }

    #[test]
    fn obb_yolo_mse_uses_ciou_obb_box_delta_branch_with_weighted_matching() {
        let network = build(
            "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=mse\ncovariance_normalizer=1.0\n",
        )
        .expect("weighted MSE OBB config parses");
        let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        let training =
            crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
                .expect("weighted MSE OBB config builds");

        assert!(!training.match_config.score_config.ciou_obb);
        assert!(training.box_delta_config.score.ciou_obb);
        assert_eq!(
            training.match_config.score_config.covariance_normalizer,
            training.box_delta_config.score.covariance_normalizer
        );
    }

    #[test]
    fn obb_yolo_defaults_to_ciou_obb_match_scoring() {
        let network = build(
            "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\n",
        )
        .expect("default OBB loss parses");
        let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        assert!(spec.ciou_obb_loss);
        assert_eq!(spec.iou_loss, super::YoloIouLoss::Ciou);
        assert_eq!(spec.iou_thresh_kind, super::YoloIouLoss::Iou);
        let training =
            crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
                .expect("default OBB score config builds");
        assert!(training.match_config.score_config.ciou_obb);

        let multiplicative = build(
            "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=ciou_obb_mul\n",
        )
        .expect("multiplicative CIoU-OBB parses");
        let LayerKind::ObbYolo(spec) = &multiplicative.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        assert!(spec.ciou_obb_loss);
        assert!(spec.ciou_obb_multiplicative_loss);
        assert_eq!(spec.iou_loss, super::YoloIouLoss::Ciou);
        let training =
            crate::obb_training::build_obb_yolo_training_head_config(spec, 0, 1, 1, 1, 1)
                .expect("multiplicative CIoU-OBB training config builds");
        assert!(training.box_delta_config.ciou_obb_multiplicative_loss);
        assert!(
            training
                .front_point_delta_config
                .ciou_obb_multiplicative_loss
        );

        let invalid = build(
            "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\niou_loss=unsupported\n",
        );
        assert!(invalid.is_err(), "unsupported OBB iou_loss must fail");
    }

    #[test]
    fn parses_obb_anchor_pair_and_directional_quadruple_forms() {
        let build = |anchor_key: &str, anchor_values: &str| {
            let cfg = format!(
                "[net]\nwidth=4\nheight=2\nchannels=16\nobb=1\n[obb_yolo]\nclasses=1\nnum=2\nmask=0,1\n{anchor_key}={anchor_values}\n"
            );
            Network::from_config(ConfigFile::parse_str("obb-anchor-forms.cfg", &cfg).unwrap())
        };

        let directional = build("obb_anchors", "4,2,0.25,-0.5,8,6,-0.4,0.3")
            .expect("four-value source anchors parse");
        let LayerKind::ObbYolo(spec) = &directional.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        assert_eq!(spec.anchors, vec![4.0, 2.0, 8.0, 6.0]);
        assert_eq!(
            spec.front_anchor_offsets,
            Some(vec![[0.25, -0.5], [-0.4, 0.3]])
        );
        let training =
            crate::obb_training::build_obb_yolo_training_head_config(spec, 7, 2, 3, 640, 480)
                .expect("parsed spec builds matching and assignment configs");
        assert_eq!(training.match_config.head_index, 7);
        assert_eq!(
            (
                training.match_config.grid_width,
                training.match_config.grid_height
            ),
            (2, 3)
        );
        assert_eq!(
            training.match_config.front_anchor_offsets,
            spec.front_anchor_offsets
        );
        assert_eq!(
            training.assignment_config.front_anchor_offsets,
            spec.front_anchor_offsets
        );
        assert_eq!(training.front_point_delta_config.normalizer, 0.35);
        assert_eq!(
            training.box_delta_config.iou_normalizer,
            spec.iou_normalizer
        );

        let direction_free =
            build("obb_anchors", "4,2,8,6").expect("two-value source anchors parse");
        let LayerKind::ObbYolo(spec) = &direction_free.layers[0].kind else {
            panic!("expected OBB YOLO layer");
        };
        assert_eq!(spec.anchors, vec![4.0, 2.0, 8.0, 6.0]);
        assert_eq!(spec.front_anchor_offsets, None);

        for values in [
            "4,2,0.25",                  // wrong cardinality
            "4,2,NaN,-0.5,8,6,-0.4,0.3", // non-finite directional offset
        ] {
            assert!(build("obb_anchors", values).is_err(), "accepted {values}");
        }
        let both_keys = Network::from_config(
            ConfigFile::parse_str(
                "obb-anchor-forms.cfg",
                "[net]\nwidth=4\nheight=2\nchannels=16\nobb=1\n[obb_yolo]\nclasses=1\nnum=2\nmask=0,1\nanchors=4,2,8,6\nobb_anchors=4,2,8,6\n",
            )
            .unwrap(),
        );
        assert!(both_keys.is_err(), "ambiguous anchor keys must fail");

        let base = "[net]\nwidth=4\nheight=2\nchannels=16\nobb=1\n[obb_yolo]\nclasses=1\nnum=2\nmask=0,1\nobb_anchors=4,2,8,6\n";
        let parse_front_normalizer = |options: &str| {
            let cfg = format!("{base}{options}");
            Network::from_config(ConfigFile::parse_str("obb-front-loss.cfg", &cfg).unwrap())
                .unwrap()
        };
        let read_front_normalizer = |network: &Network| {
            let LayerKind::ObbYolo(spec) = &network.layers[0].kind else {
                panic!("expected OBB YOLO layer");
            };
            spec.front_loss_normalizer
        };
        assert_eq!(read_front_normalizer(&parse_front_normalizer("")), 0.35);
        assert_eq!(
            read_front_normalizer(&parse_front_normalizer("front_point_normalizer=0.8\n")),
            0.8
        );
        assert_eq!(
            read_front_normalizer(&parse_front_normalizer("front_loss_normalizer=0.6\n")),
            0.6
        );
        assert_eq!(
            read_front_normalizer(&parse_front_normalizer(
                "front_point_normalizer=0.8\nfront_loss_normalizer=0.4\n"
            )),
            0.4,
            "front_loss_normalizer takes precedence over its alias"
        );
    }

    #[test]
    fn net_obb_flag_is_binary_and_rejects_axis_aligned_yolo() {
        for cfg in [
            "[net]\nwidth=1\nheight=1\nchannels=1\nobb=2\n",
            "[net]\nwidth=1\nheight=1\nchannels=1\nobb=1\n[yolo]\nclasses=1\n",
        ] {
            assert!(
                Network::from_config(
                    ConfigFile::parse_str("bad-net-obb.cfg", cfg).expect("cfg parses")
                )
                .is_err(),
                "invalid OBB opt-in should fail: {cfg}"
            );
        }
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

    #[test]
    fn wmhf_aliases_defaults_split_shape_and_source_weight_layout() {
        for alias in ["wmhf", "wavelet_mamba_hf", "wavelet_mamba_high_frequency"] {
            let network = build(&format!(
                "[net]\nwidth=5\nheight=3\nchannels=4\n[{alias}]\n"
            ))
            .unwrap_or_else(|error| panic!("[{alias}] should parse: {error}"));
            let layer = &network.layers[0];
            let LayerKind::Wmhf(spec) = &layer.kind else {
                panic!("expected WMHF for [{alias}]");
            };
            assert_eq!(layer.kind.name(), "wmhf");
            assert_eq!(layer.output_shape, super::TensorShape::new(5, 3, 4, 1));
            assert_eq!(
                (
                    spec.filters,
                    spec.identity_channels,
                    spec.local_channels,
                    spec.global_channels,
                    spec.freq_scale,
                    spec.shortcut,
                    spec.batch_normalize,
                    spec.activation.clone(),
                ),
                (4, 1, 2, 1, 0.25, true, true, super::Activation::Leaky)
            );
            let entries = layer.weights().expect("WMHF weights").entries;
            assert_eq!(
                entries.iter().map(|entry| entry.len).collect::<Vec<_>>(),
                vec![
                    4, 4, 4, 4, 16, // pre: bias, BN arrays, weights
                    2, 18, // depthwise 3x3
                    2, 50, // depthwise 5x5
                    2, 98, // depthwise 7x7
                    2, 2, 2, 2, 12, // local mix: bias, BN arrays, weights
                    4, 4, // high-frequency gate
                    4, 4, 4, 4, 16, // final fuse
                    4,  // four scan vectors
                ]
            );
            assert_eq!(
                entries
                    .iter()
                    .map(|entry| entry.array.name())
                    .collect::<Vec<_>>(),
                vec![
                    "biases",
                    "scales",
                    "rolling_mean",
                    "rolling_variance",
                    "weights",
                    "biases",
                    "weights",
                    "biases",
                    "weights",
                    "biases",
                    "weights",
                    "biases",
                    "scales",
                    "rolling_mean",
                    "rolling_variance",
                    "weights",
                    "biases",
                    "weights",
                    "biases",
                    "scales",
                    "rolling_mean",
                    "rolling_variance",
                    "weights",
                    "scan_weights",
                ]
            );
        }
    }

    #[test]
    fn wmhf_split_normalization_and_strict_config_are_pinned() {
        let make = |options: &str| {
            build(&format!(
                "[net]\nwidth=5\nheight=3\nchannels=4\n[wmhf]\n{options}\n"
            ))
        };
        let normalized =
            make("filters=5\nidentity_ratio=0.4\nlocal_ratio=0\nshortcut=0\nbatch_normalize=0")
                .expect("finite source-normalized split");
        let LayerKind::Wmhf(spec) = &normalized.layers[0].kind else {
            panic!("WMHF layer");
        };
        assert_eq!(
            (
                spec.identity_channels,
                spec.local_channels,
                spec.global_channels
            ),
            (2, 1, 2)
        );
        assert!(!spec.shortcut && !spec.batch_normalize);
        assert_eq!(
            normalized.layers[0]
                .weights()
                .unwrap()
                .entries
                .iter()
                .map(|entry| entry.array.name())
                .filter(|name| *name == "scales"
                    || *name == "rolling_mean"
                    || *name == "rolling_variance")
                .count(),
            0
        );

        for (options, expected) in [
            ("filters=3", "filters must be at least 4"),
            ("freq_scale=NaN", "freq_scale must be finite"),
            (
                "identity_ratio=inf",
                "identity_ratio and local_ratio must be finite",
            ),
            ("identity_ratio=2\nlocal_ratio=2", "channel split"),
            ("unknown_option=1", "unknown option"),
            (
                "activation=not_an_activation",
                "activation name is unsupported",
            ),
        ] {
            let error = make(options).expect_err("invalid WMHF config should fail");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn clifford_defaults_shape_and_weight_order_match_source_contract() {
        let network = build("[net]\nwidth=5\nheight=3\nchannels=4\nbatch=2\n[clifford]\n")
            .expect("default Clifford config");
        let layer = &network.layers[0];
        let LayerKind::Clifford(spec) = &layer.kind else {
            panic!("expected Clifford layer");
        };
        assert_eq!(layer.output_shape, TensorShape::new(5, 3, 4, 2));
        assert_eq!(
            (spec.filters, &spec.wedge_shifts, &spec.inner_shifts),
            (4, &vec![1, 2], &vec![1, 2])
        );
        assert_eq!(spec.context_mode, CliffordContextMode::Diff);
        assert_eq!(spec.interaction_mode, CliffordInteractionMode::Full);
        assert_eq!(spec.gffn_mode, CliffordGffnMode::Local);
        assert_eq!(spec.higher_mode, CliffordHigherMode::Off);
        assert_eq!(spec.global_op, CliffordGlobalOp::AvgPool);
        assert_eq!((spec.dwconv_size, spec.num_dwconv), (3, 1));
        assert_eq!(spec.drop_path, 0.0);
        assert_eq!(spec.layerscale_init, 0.0001);
        let entries = layer.weights().expect("Clifford weights").entries;
        assert_eq!(
            entries.iter().map(|entry| entry.len).collect::<Vec<_>>(),
            vec![4, 4, 16, 4, 4, 4, 4, 4, 36, 64, 4, 32, 4, 4]
        );
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.array.name())
                .collect::<Vec<_>>(),
            vec![
                "ln_gamma",
                "ln_beta",
                "w_det",
                "b_det",
                "biases",
                "scales",
                "rolling_mean",
                "rolling_variance",
                "weights",
                "w_proj",
                "b_proj",
                "w_gate",
                "b_gate",
                "layer_scale",
            ]
        );
    }

    #[test]
    fn clifford_modes_shift_normalization_and_global_weight_layout_are_pinned() {
        let network = build(
            "[net]\nwidth=7\nheight=3\nchannels=4\n[clifford]\nfilters=4\nshifts=1,2\nshifts_wedge=-1,2\nshifts_inner=2,1\nctx_mode=abs\ncli_mode=full\ngffn_mode=hybrid\ncli_higher_mode=vb\nglobal_op=fourier\nfno_modes_h=2\nfno_modes_w=3\ndwconv_size=1\nnum_dwconv=2\nactivation=swish\ndrop_path=0.25\nlayerscale=0.01\n",
        )
        .expect("full-mode Clifford config");
        let layer = &network.layers[0];
        let LayerKind::Clifford(spec) = &layer.kind else {
            panic!("expected Clifford layer");
        };
        assert_eq!(spec.wedge_shifts, vec![3, 2]);
        assert_eq!(spec.inner_shifts, vec![2, 1]);
        assert_eq!(spec.context_mode, CliffordContextMode::Abs);
        assert_eq!(spec.interaction_mode, CliffordInteractionMode::Full);
        assert_eq!(spec.gffn_mode, CliffordGffnMode::Hybrid);
        assert_eq!(spec.higher_mode, CliffordHigherMode::VectorBivector);
        assert_eq!(spec.global_op, CliffordGlobalOp::Fourier);
        assert_eq!((spec.fno_modes_h, spec.fno_modes_w), (2, 3));
        assert_eq!((spec.dwconv_size, spec.num_dwconv), (1, 2));
        assert_eq!(spec.drop_path, 0.25);
        assert_eq!(spec.layerscale_init, 0.01);
        let entries = layer.weights().expect("Clifford weights").entries;
        assert_eq!(entries[0].array.name(), "ln_gamma");
        assert_eq!(entries[4].array.name(), "biases");
        assert_eq!(entries[4].len, 4);
        assert_eq!(entries[9].array.name(), "biases");
        assert_eq!(entries[9].len, 4);
        assert_eq!(entries[14].array.name(), "w_proj");
        assert_eq!(entries[14].len, 4 * 16);
        assert_eq!(entries[18].array.name(), "layer_scale");
        assert_eq!(entries[19].array.name(), "w_proj_g");
        assert_eq!(entries[23].array.name(), "fno_w_re");
        assert_eq!(entries[23].len, 4 * 2 * 3);
        assert_eq!(entries[24].array.name(), "fno_w_im");
        assert_eq!(layer.output_shape, TensorShape::new(7, 3, 4, 1));
    }

    #[test]
    fn clifford_rejects_unknown_or_unsupported_config_and_invalid_shapes() {
        let make = |options: &str| {
            build(&format!(
                "[net]\nwidth=5\nheight=3\nchannels=4\n[clifford]\n{options}\n"
            ))
        };
        for (options, expected) in [
            (
                "filters=3",
                "filters must be positive and equal input channels",
            ),
            ("activation=relu", "only activation=swish is supported"),
            ("ctx_mode=typo", "unsupported ctx_mode"),
            ("cli_mode=typo", "unsupported cli_mode"),
            ("gffn_mode=typo", "unsupported gffn_mode"),
            ("cli_higher_mode=typo", "unsupported cli_higher_mode"),
            ("global_op=typo", "unsupported global_op"),
            (
                "global_op=fourier\nfno_modes_h=0",
                "requires positive fno_modes_h",
            ),
            ("dwconv_size=2", "positive odd integer"),
            ("drop_path=1", "drop_path must be finite and in [0, 1)"),
            ("drop_path=NaN", "drop_path must be finite and in [0, 1)"),
            ("layerscale=inf", "layerscale must be finite"),
            ("clip=0.1", "clip is not supported"),
            ("grad_centr=1", "grad_centr is not supported"),
            ("stream=0", "non-default stream scheduling"),
            ("unknown_option=1", "unknown option"),
        ] {
            let error = make(options).expect_err("invalid Clifford config should fail");
            assert!(error.to_string().contains(expected), "{options}: {error}");
        }
    }

    #[test]
    fn dsconv_aliases_defaults_shape_and_weights_match_source_contract() {
        for alias in ["dsconv", "dynamic_snake_conv"] {
            let network = build(&format!(
                "[net]\nwidth=5\nheight=4\nchannels=3\n[{alias}]\n"
            ))
            .unwrap_or_else(|error| panic!("[{alias}] should parse: {error}"));
            let layer = &network.layers[0];
            let LayerKind::DsConv(spec) = &layer.kind else {
                panic!("expected DSConv for [{alias}]")
            };
            assert_eq!(layer.kind.name(), "dsconv");
            assert_eq!(layer.output_shape, super::TensorShape::new(5, 4, 3, 1));
            assert_eq!(
                (
                    spec.filters,
                    spec.size,
                    spec.padding,
                    spec.extend_scope,
                    spec.morph,
                    spec.templates,
                    spec.drop_prob,
                ),
                (3, 9, 4, 1.0, 0, 1, 0.0)
            );
            let weights = layer.weights().expect("DSConv weights");
            assert_eq!(
                weights
                    .entries
                    .iter()
                    .map(|entry| (entry.array.name(), entry.len))
                    .collect::<Vec<_>>(),
                vec![
                    ("biases", 3),
                    ("weights", 3 * 3 * 9),
                    ("offset_weights", 2 * 9 * 3 * 9),
                    ("offset_biases", 2 * 9),
                ]
            );
        }
    }

    #[test]
    fn dsconv_parser_pins_axis_modes_normalization_and_unsupported_options() {
        let make = |options: &str| {
            build(&format!(
                "[net]\nwidth=5\nheight=4\nchannels=2\n[dsconv]\n{options}\n"
            ))
        };
        for (morph, expected) in [("0", 0), ("1", 1), ("2", 2), ("99", 0), ("-1", 0)] {
            let network = make(&format!("filters=3\nsize=3\nmorph={morph}"))
                .unwrap_or_else(|error| panic!("morph={morph}: {error}"));
            let LayerKind::DsConv(spec) = &network.layers[0].kind else {
                panic!("DSConv layer")
            };
            assert_eq!(spec.morph, expected);
        }
        let normalized = make("filters=3\nsize=-1\ntemplates=0\ndrop_prob=2")
            .expect("source-normalized factory options parse");
        let LayerKind::DsConv(spec) = &normalized.layers[0].kind else {
            panic!("DSConv layer")
        };
        assert_eq!(
            (spec.size, spec.padding, spec.templates, spec.drop_prob),
            (1, 0, 1, 1.0)
        );

        let morph2 = make("filters=3\nsize=3\nmorph=2\ntemplates=4\ndrop_prob=0.4")
            .expect("morph=2 forces one template");
        let LayerKind::DsConv(spec) = &morph2.layers[0].kind else {
            panic!("DSConv layer")
        };
        assert_eq!((spec.templates, spec.drop_prob), (1, 0.0));

        for (options, expected_message) in [
            ("size=2", "size must be odd"),
            ("stride=2", "only stride=1"),
            ("filters=1\nsize=3\nmorph=2", "morph=2 requires"),
            ("templates=2", "templates>1 multi-view"),
            ("extend_scope=NaN", "must be finite"),
            ("unknown_option=1", "unknown option"),
        ] {
            let error = make(options).expect_err("invalid DSConv cfg should fail");
            assert!(error.to_string().contains(expected_message), "{error}");
        }
    }

    #[test]
    fn dsconv_try_fix_nan_default_covers_both_aliases_and_respects_override() {
        for alias in ["dsconv", "dynamic_snake_conv"] {
            let base = format!("[net]\nwidth=2\nheight=2\nchannels=1\n[{alias}]\nsize=1\n");
            assert!(build(&base).unwrap().net.try_fix_nan);
            assert!(
                !build(&base.replace("[net]", "[net]\ntry_fix_nan=0"))
                    .unwrap()
                    .net
                    .try_fix_nan
            );
            assert!(
                build(&base.replace("[net]", "[net]\ntry_fix_nan=1"))
                    .unwrap()
                    .net
                    .try_fix_nan
            );
        }
    }

    #[test]
    fn foveation_film_alias_defaults_shape_and_source_order_weights() {
        for alias in ["foveation_film", "hf_film"] {
            let network = build(&format!(
                "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\n[hf]\nradius_mode=anisotropic\n[{alias}]\n"
            ))
            .unwrap_or_else(|error| panic!("[{alias}] should parse: {error}"));
            let layer = &network.layers[1];
            let super::LayerKind::FoveationFilm(spec) = &layer.kind else {
                panic!("expected foveation film for [{alias}]");
            };
            assert_eq!(layer.kind.name(), "foveation_film");
            assert_eq!(spec.hf_layer, 0);
            assert_eq!(spec.param_count, 5);
            assert_eq!(spec.embed_dim, 256);
            assert!(spec.backprop_to_hf);
            assert_eq!(spec.film_init_scale, 0.0);
            assert_eq!(layer.output_shape, layer.input_shape);
            assert_eq!(layer.output_shape, super::TensorShape::new(5, 3, 2, 2));
            assert_eq!(
                layer
                    .weights()
                    .unwrap()
                    .entries
                    .iter()
                    .map(|entry| (entry.array.name(), entry.len))
                    .collect::<Vec<_>>(),
                vec![
                    ("w_embed", 256 * 5),
                    ("b_embed", 256),
                    ("w_gamma", 2 * 256),
                    ("b_gamma", 2),
                    ("w_beta", 2 * 256),
                    ("b_beta", 2),
                ]
            );
        }
    }

    #[test]
    fn foveation_film_resolves_nearest_absolute_and_relative_hf_layers() {
        let config = |hf_layer: &str| {
            build(&format!(
                "[net]\nwidth=4\nheight=3\nchannels=1\n[hf]\nradius_mode=isotropic\n[convolutional]\nfilters=1\nsize=1\nactivation=linear\n[hf]\nradius_mode=anisotropic\n[convolutional]\nfilters=1\nsize=1\nactivation=linear\n[hf_film]\n{hf_layer}\n"
            ))
        };
        for (option, expected_index, expected_params) in
            [("", 2, 5), ("hf_layer=0", 0, 4), ("hf_layer=-4", 0, 4)]
        {
            let network = config(option).unwrap();
            let super::LayerKind::FoveationFilm(spec) = &network.layers[4].kind else {
                panic!("expected foveation film");
            };
            assert_eq!(
                (spec.hf_layer, spec.param_count),
                (expected_index, expected_params)
            );
        }
    }

    #[test]
    fn foveation_film_rejects_unknown_options_and_invalid_hf_sources() {
        let make = |extra: &str, layers: &str| {
            build(&format!(
                "[net]\nwidth=4\nheight=3\nchannels=1\n{layers}\n[foveation_film]\n{extra}\n"
            ))
        };
        for (extra, layers, expected) in [
            ("unknown=1", "[hf]", "unknown or unsupported option"),
            (
                "hf_layer=0",
                "[convolutional]\nfilters=1\nsize=1",
                "must reference [hyperbolic_foveation]",
            ),
            ("hf_layer=4", "[hf]", "must reference an earlier layer"),
            ("hf_layer=-9", "[hf]", "invalid hf_layer"),
            ("", "", "requires a preceding [hyperbolic_foveation] layer"),
            (
                "film_init_scale=NaN",
                "[hf]",
                "film_init_scale must be finite",
            ),
            (
                "learning_rate=-1",
                "[hf]",
                "learning_rate must be finite and non-negative",
            ),
            ("stream=0", "[hf]", "GPU stream options are unsupported"),
        ] {
            let error = make(extra, layers).expect_err("invalid FiLM config must fail");
            assert!(
                error.to_string().contains(expected),
                "{extra} {layers}: {error}"
            );
        }
    }

    #[test]
    fn hyperbolic_foveation_aliases_defaults_shape_and_weights() {
        for alias in ["hyperbolic_foveation", "hf"] {
            let network = build(&format!(
                "[net]\nwidth=5\nheight=3\nchannels=2\nbatch=2\n[{alias}]\n"
            ))
            .unwrap_or_else(|error| panic!("[{alias}] should parse: {error}"));
            let layer = &network.layers[0];
            let super::LayerKind::HyperbolicFoveation(spec) = &layer.kind else {
                panic!("expected hyperbolic foveation for [{alias}]");
            };
            assert_eq!(layer.kind.name(), "hyperbolic_foveation");
            assert_eq!(layer.output_shape, super::TensorShape::new(5, 3, 2, 2));
            assert_eq!(
                (
                    spec.init_cx,
                    spec.init_cy,
                    spec.init_radius,
                    spec.init_radius_x,
                    spec.init_radius_y,
                    spec.init_strength,
                ),
                (0.0, 0.0, 0.75, 0.75, 0.75, 0.05)
            );
            assert_eq!(
                (
                    spec.alpha,
                    spec.p_exp,
                    spec.downsample,
                    spec.pred_hidden,
                    spec.nr_iters,
                    spec.nr_eta,
                ),
                (2.0, 2.0, 8, 0, 10, 0.5)
            );
            assert_eq!(spec.radius_mode, super::HfRadiusMode::Isotropic);
            assert_eq!(spec.inverse_method, super::HfInverseMethod::Newton);
            assert_eq!(spec.warp_source, super::HfWarpSource::Previous);
            assert_eq!(spec.pred_source_layer, None);
            assert_eq!(spec.pred_input_shape, super::TensorShape::new(5, 3, 2, 2));
            let weights = layer.weights().expect("hyperbolic foveation weights");
            assert_eq!(
                weights
                    .entries
                    .iter()
                    .map(|entry| (entry.array.name(), entry.len))
                    .collect::<Vec<_>>(),
                vec![("hf_params", 4)]
            );
        }
    }

    #[test]
    fn hyperbolic_foveation_resolves_predictor_edges_and_network_input_shape() {
        for source in ["-2", "0"] {
            let network = build(&format!(
                "[net]\nwidth=8\nheight=6\nchannels=2\nbatch=2\n\
                 [convolutional]\nfilters=4\nsize=1\nactivation=linear\n\
                 [convolutional]\nfilters=5\nsize=1\nactivation=linear\n\
                 [hf]\nparams=learned\npred_hidden=3\ndownsample=2\n\
                 pred_source_layer={source}\nwarp_source=network_input\nradius_mode=anisotropic\n"
            ))
            .unwrap_or_else(|error| panic!("pred_source_layer={source}: {error}"));
            let layer = &network.layers[2];
            let super::LayerKind::HyperbolicFoveation(spec) = &layer.kind else {
                panic!("expected hyperbolic foveation");
            };
            assert_eq!(spec.pred_source_layer, Some(0));
            assert_eq!(spec.pred_input_shape, super::TensorShape::new(8, 6, 4, 2));
            assert_eq!(spec.predictor_input_dim(), Some(4 * 3 * 4));
            assert_eq!(spec.param_count(), 5);
            assert_eq!(
                layer.output_shape,
                super::TensorShape::new(8, 6, 2, 2),
                "network_input warp uses the original input shape"
            );
            assert_eq!(
                layer
                    .weights()
                    .unwrap()
                    .entries
                    .iter()
                    .map(|entry| (entry.array.name(), entry.len))
                    .collect::<Vec<_>>(),
                vec![
                    ("hf_params", 5),
                    ("hf_pred_w1", 3 * 48),
                    ("hf_pred_b1", 3),
                    ("hf_pred_w2", 5 * 3),
                    ("hf_pred_b2", 5),
                ]
            );
        }

        let previous = build(
            "[net]\nwidth=4\nheight=3\nchannels=1\n\
             [convolutional]\nfilters=2\nsize=1\nactivation=linear\n\
             [hf]\n",
        )
        .expect("previous-source output shape");
        assert_eq!(
            previous.layers[1].output_shape,
            previous.layers[1].input_shape
        );
    }

    #[test]
    fn hyperbolic_foveation_accepts_params_and_inverse_aliases() {
        for (mode, explicit_hidden, expected_hidden) in [
            ("fixed", "pred_hidden=3\n", 0),
            ("trainable", "pred_hidden=3\n", 3),
            ("global", "pred_hidden=3\n", 3),
            ("learned", "pred_hidden=0\n", 64),
        ] {
            let network = build(&format!(
                "[net]\nwidth=4\nheight=3\nchannels=1\n[hf]\nparams={mode}\n{explicit_hidden}"
            ))
            .expect("supported hyperbolic-foveation params mode");
            let super::LayerKind::HyperbolicFoveation(spec) = &network.layers[0].kind else {
                panic!("expected hyperbolic foveation");
            };
            assert_eq!(spec.pred_hidden, expected_hidden, "params={mode}");
        }
        for inverse in ["fixed_point", "fixed-point"] {
            let network = build(&format!(
                "[net]\nwidth=4\nheight=3\nchannels=1\n[hf]\ninverse_method={inverse}\nwarp_source=network_input\nradius_mode=anisotropic\n"
            ))
            .expect("supported fixed-point inverse alias");
            let super::LayerKind::HyperbolicFoveation(spec) = &network.layers[0].kind else {
                panic!("expected hyperbolic foveation");
            };
            assert_eq!(spec.inverse_method, super::HfInverseMethod::FixedPoint);
            assert_eq!(spec.warp_source, super::HfWarpSource::NetworkInput);
            assert_eq!(spec.radius_mode, super::HfRadiusMode::Anisotropic);
        }
    }

    #[test]
    fn hyperbolic_foveation_rejects_unknown_or_unsupported_options() {
        let make = |options: &str| {
            build(&format!(
                "[net]\nwidth=4\nheight=3\nchannels=1\n[hf]\n{options}\n"
            ))
        };
        for (options, expected) in [
            ("unknown_option=1", "unknown option"),
            ("params=adaptive", "unsupported params mode"),
            ("inverse_method=bad", "unsupported inverse_method"),
            ("warp_source=bad", "unsupported warp_source"),
            ("radius_mode=bad", "unsupported radius_mode"),
            ("nr_eta=2", "nr_eta must be finite"),
            ("downsample=0", "downsample must be positive"),
            ("debug=1", "debug summaries are unsupported"),
            ("debug_interval=200", "custom debug_interval is unsupported"),
            ("batch_normalize=1", "batch normalization is unsupported"),
            ("stream=0", "GPU stream options are unsupported"),
            ("pred_hidden=-1", "pred_hidden must be non-negative"),
            (
                "init_cx=1.1",
                "initial center coordinates must be in [-1, 1]",
            ),
            (
                "learning_rate=-1",
                "learning_rate must be finite and non-negative",
            ),
            ("init_strength=NaN", "init_strength must be finite"),
        ] {
            let error = make(options).expect_err("unsupported hyperbolic-foveation cfg must fail");
            assert!(error.to_string().contains(expected), "{options}: {error}");
        }

        for source in ["1", "-3"] {
            let error = build(&format!(
                "[net]\nwidth=4\nheight=3\nchannels=1\n\
                 [convolutional]\nfilters=2\nsize=1\n\
                 [hf]\npred_source_layer={source}\n"
            ))
            .expect_err("predictor source must be earlier and valid");
            assert!(error.to_string().contains("pred_source_layer"), "{error}");
        }
    }

    #[test]
    fn graph_conv_aliases_defaults_geometry_and_source_weight_order() {
        for alias in ["graph_conv", "graph_convolutional"] {
            let network = build(&format!(
                "[net]\nwidth=7\nheight=5\nchannels=4\n[{alias}]\nfilters=6\ngroups=2\nsize=3\nstride_x=2\nstride_y=1\ndilation=2\npadding=1\nedge_mode=0\nuse_self=0\nactivation=linear\n"
            ))
            .unwrap_or_else(|error| panic!("[{alias}] should parse: {error}"));
            let layer = &network.layers[0];
            let LayerKind::GraphConv(spec) = &layer.kind else {
                panic!("expected graph-conv layer for [{alias}]");
            };
            assert_eq!(layer.kind.name(), "graph_conv");
            assert_eq!(layer.output_shape, super::TensorShape::new(3, 3, 6, 1));
            assert_eq!(
                (
                    spec.filters,
                    spec.groups,
                    spec.size,
                    spec.stride_x,
                    spec.stride_y,
                    spec.dilation,
                    spec.padding,
                    spec.edge_mode,
                    spec.use_self,
                    spec.batch_normalize,
                    &spec.activation,
                ),
                (
                    6,
                    2,
                    3,
                    2,
                    1,
                    2,
                    1,
                    0,
                    false,
                    false,
                    &super::Activation::Linear
                )
            );
            let weights = layer.weights().expect("graph-conv weights");
            assert_eq!(
                weights
                    .entries
                    .iter()
                    .map(|entry| (entry.array.name(), entry.len))
                    .collect::<Vec<_>>(),
                vec![("biases", 6), ("weights", 12)]
            );
        }

        let defaults = build("[net]\nwidth=2\nheight=3\nchannels=4\n[graph_conv]\n")
            .expect("default graph-conv config");
        let layer = &defaults.layers[0];
        let LayerKind::GraphConv(spec) = &layer.kind else {
            panic!("graph-conv layer");
        };
        assert_eq!(layer.output_shape, super::TensorShape::new(2, 3, 1, 1));
        assert_eq!((spec.edge_mode, spec.use_self, spec.size), (1, true, 1));
        assert!(!defaults.net.try_fix_nan);
        assert_eq!(
            layer
                .weights()
                .unwrap()
                .entries
                .iter()
                .map(|entry| (entry.array.name(), entry.len))
                .collect::<Vec<_>>(),
            vec![
                ("biases", 1),
                ("weights", 4),
                ("graph_self_weights", 4),
                ("graph_edge_kernel", 8),
                ("graph_edge_biases", 1),
            ]
        );

        let cbn = build(
            "[net]\nwidth=1\nheight=1\nchannels=2\n[graph_convolutional]\nfilters=2\ncbn=1\nsize=1\ndilation=0\n",
        )
        .expect("CPU CBN config remains ordinary batch norm");
        let LayerKind::GraphConv(spec) = &cbn.layers[0].kind else {
            panic!("graph-conv layer");
        };
        assert!(spec.batch_normalize);
        assert_eq!(spec.dilation, 1, "size=1 forces dilation one");
        assert_eq!(
            cbn.layers[0]
                .weights()
                .unwrap()
                .entries
                .iter()
                .map(|entry| entry.array.name())
                .collect::<Vec<_>>(),
            vec![
                "biases",
                "scales",
                "rolling_mean",
                "rolling_variance",
                "weights",
                "graph_self_weights",
                "graph_edge_kernel",
                "graph_edge_biases"
            ]
        );
    }

    #[test]
    fn graph_conv_parser_rejects_unsupported_and_unknown_configuration() {
        let make = |options: &str| {
            build(&format!(
                "[net]\nwidth=3\nheight=2\nchannels=4\n[graph_conv]\n{options}\n"
            ))
        };
        for (options, expected) in [
            ("filters=0", "filters must be positive"),
            ("size=2", "positive and odd"),
            ("stride=0", "must be positive"),
            ("groups=3", "groups must divide"),
            ("graph_edge_mode=2", "must be 0 or 1"),
            ("valid_mask_zero=0", "must be 1"),
            ("binary=1", "binary, xnor"),
            ("bin_output=1", "binary, xnor"),
            ("antialiasing=1", "antialiasing"),
            ("assisted_excitation=0.1", "assisted_excitation"),
            ("share_index=0", "shared graph-convolution"),
            ("grad_centr=1", "grad_centr"),
            ("clip=1", "clip is unsupported"),
            ("activation=unsupported", "activation name is unsupported"),
            ("unknown_option=1", "unknown option"),
        ] {
            let error = make(options).expect_err("unsupported graph-conv config must fail");
            assert!(error.to_string().contains(expected), "{options}: {error}");
        }
    }
}

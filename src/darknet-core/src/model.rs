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
    /// C++ `parms.label_smooth_eps`: `[contrastive]` treats a class probability
    /// above this as a hard label.
    pub label_smooth_eps: f32,
    /// C++ `parms.adversarial`: `[contrastive]` then labels images `b % 2`
    /// instead of `b / 2`.
    pub adversarial: bool,
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
            label_smooth_eps: section.get_f32("label_smooth_eps", 0.0)?,
            adversarial: section.get_usize("adversarial", 0)? != 0,
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
            Self::Loggy | Self::Selu | Self::Relie | Self::Plse | Self::Hardtan | Self::Lhtan | Self::Stair | Self::HardMish | Self::Revleaky => None,
            Self::Relu => Some(F32ActivationKind::Relu),
            Self::Leaky => Some(F32ActivationKind::Leaky),
            Self::Mish => Some(F32ActivationKind::Mish),
            Self::Swish => Some(F32ActivationKind::Swish),
            Self::Tanh => Some(F32ActivationKind::Tanh),
            Self::Relu6 => Some(F32ActivationKind::Relu6),
            Self::Ramp => Some(F32ActivationKind::Ramp),
            Self::Gelu => Some(F32ActivationKind::Gelu),
            Self::Elu => Some(F32ActivationKind::Elu),
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
    /// `[vit]`: global vision transformer with learned patch projection.
    Vit(VitSpec),
    /// `[moe_convolutional]`: a learned mixture of `MoeSpec::experts`
    /// convolutional experts gated per image by a small router.
    Moe(MoeSpec),
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
            Self::Vit(_) => "vit",
            Self::Moe(_) => "moe_convolutional",
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
            Self::Vit(spec) => Some(&spec.activation),
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
        // `[omd_loss]`: C++ reads several keys under two spellings each
        // (primary first, alias otherwise), and the loss weight under three.
        "omd_loss" | "omd" => {
            let line = section.line;
            let alias_int = |primary: &str, alias: &str, fallback: usize| -> Result<usize> {
                if section.get(primary).is_some() { section.get_usize(primary, fallback) } else { section.get_usize(alias, fallback) }
            };
            let alias_float = |primary: &str, alias: &str, fallback: f32| -> Result<f32> {
                if section.get(primary).is_some() { section.get_f32(primary, fallback) } else { section.get_f32(alias, fallback) }
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
                return Err(Error::InvalidConfig(format!("[omd_loss] at line {line}: max_samples must be >= 2")));
            }
            let margin = section.get_f32("margin", 0.5)?;
            if !(0.0..=2.0).contains(&margin) {
                return Err(Error::InvalidConfig(format!("[omd_loss] at line {line}: margin must be in [0,2]")));
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
        "centernet" => {
            let line = section.line;
            let classes = section.get_usize("classes", 20)?;
            if classes == 0 {
                return Err(Error::InvalidConfig(format!("[centernet] at line {line}: classes must be > 0")));
            }
            let max_boxes = section.get_usize("max", 200)?;
            if max_boxes == 0 {
                return Err(Error::InvalidConfig(format!("[centernet] at line {line}: max must be > 0")));
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
        "vit" => {
            let line = section.line;
            for unsupported in [
                "flash_attention", "flash_attn", "vc_attention", "vc",
                "vc_expcast", "expcast", "vc_perm",
            ] {
                if section.get(unsupported).is_some() {
                    return Err(Error::InvalidConfig(format!(
                        "[vit] at line {line}: {unsupported} is not supported by the Rust reference path yet"
                    )));
                }
            }
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
            let ratio_mlp = dim
                .checked_mul(ffn_ratio)
                .ok_or_else(|| Error::InvalidConfig(format!("[vit] at line {line}: dim * ffn_ratio overflows")))?;
            let mlp_dim = section.get_usize("mlp_dim", ratio_mlp)?;
            if has_ratio && has_mlp && mlp_dim != ratio_mlp {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line} has conflicting ffn_ratio={ffn_ratio} and mlp_dim={mlp_dim} for dim={dim}"
                )));
            }
            if dim == 0 || heads == 0 || dim % heads != 0 || patch_size == 0 || patch_stride == 0 || mlp_dim == 0 {
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
                .ok_or_else(|| Error::InvalidConfig(format!("[vit] at line {line}: input height is too small for the patch")))?;
            let effective_w = current
                .width
                .checked_add(twice_pad)
                .and_then(|v| v.checked_sub(patch_size))
                .ok_or_else(|| Error::InvalidConfig(format!("[vit] at line {line}: input width is too small for the patch")))?;
            if effective_h % patch_stride != 0 || effective_w % patch_stride != 0 {
                return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: effective patch extents {effective_w}x{effective_h} must be divisible by patch_stride={patch_stride}"
                )));
            }
            let pos_embed = section.get_str("pos_embed", "learned").to_ascii_lowercase();
            let learned_pos_embed = match pos_embed.as_str() {
                "learned" | "absolute" => true,
                "sinusoidal" | "2d_sinusoidal" | "simple" => false,
                _ => return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: pos_embed={pos_embed} must be learned or sinusoidal"
                ))),
            };
            let pos_init = section.get_str("pos_init", "random").to_ascii_lowercase();
            let pos_init_zero = match pos_init.as_str() {
                "random" | "small_random" => false,
                "zero" | "zeros" => true,
                _ => return Err(Error::InvalidConfig(format!(
                    "[vit] at line {line}: pos_init={pos_init} must be random or zero"
                ))),
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
            }.max(1);
            let msa_topk = if section.get("msa_topk").is_some() {
                section.get_usize("msa_topk", 4)?
            } else {
                section.get_usize("msa_top_blocks", 4)?
            }.max(1);
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
            })
        }
        "laya_scorer" => {
            // C++ `parse_laya_scorer_section`: the layer is constructed for
            // `max_options x hidden` and checks the incoming shape against it,
            // so both come from the section (not from `[net]`).
            let line = section.line;
            let hidden = section.get_usize("hidden", LAYA_SCORER_HIDDEN)?;
            if hidden == 0 {
                return Err(Error::InvalidConfig(format!("[laya_scorer] at line {line}: hidden must be > 0")));
            }
            let max_options = section.get_usize("max_options", current.width)?;
            if max_options == 0 {
                return Err(Error::InvalidConfig(format!("[laya_scorer] at line {line}: max_options must be > 0")));
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
                    )))
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
            let temperature = section
                .get_f32("temperature", section.get_f32("gate_temperature", 1.0)?)?;
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
                    dilation: if size == 1 { 1 } else { section.get_usize("dilation", 1)? },
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
            Ok(TensorShape::new(current.width, 1, spec.hidden, current.batch))
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
            Ok(TensorShape::new(current.width, 1, spec.hidden, current.batch))
        }
        LayerKind::Vit(spec) => {
            let line = layer_index;
            let twice_pad = spec.patch_pad.checked_mul(2).ok_or_else(|| {
                Error::InvalidConfig(format!("[vit] layer #{line}: patch padding overflows"))
            })?;
            let eff_h = current.height.checked_add(twice_pad)
                .and_then(|v| v.checked_sub(spec.patch_size))
                .ok_or_else(|| Error::InvalidConfig(format!("[vit] layer #{line}: invalid patch height geometry")))?;
            let eff_w = current.width.checked_add(twice_pad)
                .and_then(|v| v.checked_sub(spec.patch_size))
                .ok_or_else(|| Error::InvalidConfig(format!("[vit] layer #{line}: invalid patch width geometry")))?;
            if spec.patch_stride == 0 || eff_h % spec.patch_stride != 0 || eff_w % spec.patch_stride != 0 {
                return Err(Error::InvalidConfig(format!("[vit] layer #{line}: patch geometry is not divisible by patch_stride={}", spec.patch_stride)));
            }
            Ok(TensorShape::new(
                eff_w / spec.patch_stride + 1,
                eff_h / spec.patch_stride + 1,
                spec.dim,
                current.batch,
            ))
        }
        | LayerKind::Yolox(_)
        | LayerKind::Ppyoloe(_)
        | LayerKind::Yolonas(_) => Ok(current),
        | LayerKind::CenterNet(_) => Ok(current),
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
            Ok(TensorShape::new(spec.max_options, 1, spec.hidden, current.batch))
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
            conv_dim(current.width, spec.expert.padding, spec.expert.size, spec.expert.stride_x),
            conv_dim(current.height, spec.expert.padding, spec.expert.size, spec.expert.stride_y),
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

/// C++ `parse_yolox_section`: `classes`(20), `max`(200), `jitter`(0.2) and
/// `resize`(1.0).  The PP-YOLOE / YOLO-NAS siblings share the first two keys and
/// add `reg_max`(16); they are not ported yet.
fn parse_modern_yolo_spec(section: &RawSection) -> Result<ModernYoloSpec> {
    let line = section.line;
    let kind = match section.name.as_str() {
        "yolox" => ModernYoloKind::Yolox,
        "ppyoloe" => ModernYoloKind::Ppyoloe,
        "yolonas" => ModernYoloKind::Yolonas,
        other => {
            return Err(Error::InvalidConfig(format!(
                "[{other}] at line {line} is not an anchor-free modern head"
            )))
        }
    };
    let classes = section.get_usize("classes", 20)?;
    if classes == 0 {
        return Err(Error::InvalidConfig(format!("[yolox] at line {line}: classes must be > 0")));
    }
    let max_boxes = section.get_usize("max", 200)?;
    if max_boxes == 0 {
        return Err(Error::InvalidConfig(format!("[yolox] at line {line}: max must be > 0")));
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

    use super::{LayerKind, LrPolicy, ModernYoloKind, Network, SegmentationLoss};

    fn build(text: &str) -> crate::error::Result<Network> {
        Network::from_config(ConfigFile::parse_str("t.cfg", text)?)
    }

    #[test]
    fn vit_parses_defaults_aliases_and_patch_grid_shape() {
        let default = build(
            "[net]\nwidth=8\nheight=6\nchannels=3\n\
             [vit]\ndim=8\nheads=2\npatch_size=2\npatch_stride=2\n",
        ).unwrap();
        let LayerKind::Vit(spec) = &default.layers[0].kind else { panic!("expected Vit") };
        assert_eq!((spec.dim, spec.mlp_dim, spec.ffn_ratio), (8, 32, 4));
        assert!(spec.learned_pos_embed && !spec.pos_init_zero);
        assert_eq!(spec.activation, super::Activation::Gelu);
        let shape = default.layers[0].output_shape;
        assert_eq!((shape.width, shape.height, shape.channels), (4, 3, 8));

        let aliases = build(
            "[net]\nwidth=8\nheight=6\nchannels=3\n\
             [vit]\nfilters=8\ndim=8\nheads=2\npatch_size=2\npatch_stride=2\n\
             ffn_ratio=3\nmlp_dim=24\npos_embed=simple\npos_init=zeros\nactivation=relu\n\
             minimax_sparse_attention=1\nmsa_block_size=2\nmsa_top_blocks=1\n",
        ).unwrap();
        let LayerKind::Vit(spec) = &aliases.layers[0].kind else { panic!("expected Vit") };
        assert_eq!((spec.dim, spec.mlp_dim, spec.ffn_ratio), (8, 24, 3));
        assert!(!spec.learned_pos_embed && spec.pos_init_zero);
        assert!(spec.msa_enabled);
        assert_eq!((spec.msa_block_size, spec.msa_topk), (2, 1));
        assert_eq!((aliases.layers[0].output_shape.width, aliases.layers[0].output_shape.height), (4, 3));
    }

    #[test]
    fn vit_layer_weights_list_matches_the_cpp_writer_order() {
        let net = build(
            "[net]\nwidth=8\nheight=6\nchannels=3\n\
             [vit]\ndim=8\nheads=2\npatch_size=2\npatch_stride=2\nmlp_dim=32\n",
        ).unwrap();
        let layer_weights = net.layers[0].weights().expect("vit has weights");
        let names: Vec<&str> = layer_weights.entries.iter().map(|entry| entry.array.name()).collect();
        assert_eq!(names, [
            "vit_patch_embed", "vit_patch_bias", "biases", "vit_ln1_gamma", "vit_ln1_beta",
            "weights", "vit_wo", "vit_wo_bias", "vit_ln2_gamma", "vit_ln2_beta",
            "vit_ffn_w1", "vit_ffn_b1", "vit_ffn_w2", "vit_ffn_b2", "vit_pos_embed", "scales",
        ]);
        assert_eq!(layer_weights.entries.iter().map(|entry| entry.len).collect::<Vec<_>>(), [
            96, 8, 24, 8, 8, 192, 64, 8, 8, 8, 256, 32, 256, 8, 96, 12,
        ]);
        assert_eq!(layer_weights.parameter_count(), crate::vit::vit_blob_float_count(
            &crate::vit::VitGeometry {
                batch: 1, channels: 3, height: 6, width: 8, patch_size: 2, patch_stride: 2,
                patch_pad: 0, hidden: 8, heads: 2, mlp_dim: 32, ln_eps: 1e-5,
                activation: super::Activation::Gelu, learned_pos_embed: true, pos_init_zero: false, msa: None,
            }, 3, 4, true
        ).unwrap());
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
            "dim=8\nheads=2\nflash_attention=1",
            "dim=8\nheads=2\nflash_attn=1",
            "dim=8\nheads=2\nvc_attention=1",
            "dim=8\nheads=2\nvc=1",
            "dim=8\nheads=2\nvc_expcast=2",
            "dim=8\nheads=2\nexpcast=2",
            "dim=8\nheads=2\nvc_perm=spatial",
        ] {
            let err = build(&cfg(invalid)).expect_err(invalid).to_string();
            assert!(err.contains("vit") || err.contains("activation"), "{invalid}: {err}");
        }
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
    fn yolox_keys_follow_cpp_defaults_and_validation() {
        // C++ `parse_yolox_section`: classes=20, max=200, jitter=0.2, resize=1.0.
        let base = "[net]\nwidth=4\nheight=4\nchannels=7\n[convolutional]\nfilters=7\nsize=1\n";
        let net = build(&format!("{base}[yolox]\n")).unwrap();
        let LayerKind::Yolox(s) = &net.layers[1].kind else { panic!() };
        assert_eq!((s.classes, s.max_boxes, s.jitter, s.resize), (20, 200, 0.2, 1.0));
        assert!(net.layers[1].weights().is_none(), "the head has no weights");
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        let net = build(&format!("{base}[yolox]\nclasses=3\nmax=64\njitter=0.1\nresize=1.5\n")).unwrap();
        let LayerKind::Yolox(s) = &net.layers[1].kind else { panic!() };
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
        let LayerKind::CenterNet(s) = &net.layers[1].kind else { panic!() };
        assert_eq!((s.classes, s.max_boxes), (20, 200));
        assert_eq!((s.center_min_radius, s.peak_nms, s.anisotropic_gaussian), (1, 1, 1));
        assert_eq!((s.small_threshold, s.small_ref_size, s.small_boost), (8.0, 32.0, 8.0));
        assert_eq!((s.scale_min_px, s.scale_max_px), (0.0, f32::MAX));
        assert_eq!((s.gaussian_iou, s.focal_alpha, s.focal_beta), (0.7, 2.0, 4.0));
        assert_eq!((s.hm_normalizer, s.wh_normalizer, s.off_normalizer), (1.0, 0.1, 1.0));
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
        let LayerKind::CenterNet(s) = &net.layers[1].kind else { panic!() };
        assert_eq!((s.classes, s.max_boxes, s.center_min_radius, s.peak_nms, s.anisotropic_gaussian), (3, 64, 2, 0, 0));
        assert_eq!((s.small_ref_size, s.small_boost, s.scale_min_px, s.scale_max_px), (16.0, 4.0, 4.0, 64.0));
        assert_eq!((s.gaussian_iou, s.focal_alpha, s.focal_beta), (0.5, 1.5, 3.0));
        assert_eq!((s.hm_normalizer, s.wh_normalizer, s.off_normalizer), (0.5, 0.25, 2.0));
        assert_eq!((s.max_delta, s.delta_normalizer), (5.0, 0.75));

        assert!(build(&format!("{base}[centernet]\nclasses=0\n")).is_err());
        assert!(build(&format!("{base}[centernet]\nmax=0\n")).is_err());
    }

    #[test]
    fn the_modern_heads_take_reg_max_and_the_nested_object_keys() {
        let base = "[net]\nwidth=2\nheight=2\nchannels=70\n[convolutional]\nfilters=70\nsize=1\n";
        let net = build(&format!("{base}[ppyoloe]\nclasses=1\nreg_max=16\n")).unwrap();
        let LayerKind::Ppyoloe(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.kind, ModernYoloKind::Ppyoloe);
        assert_eq!(s.coords(), 4 * 17);
        assert_eq!((s.min_object_cells, s.vfl_gamma), (0.0, 0.0), "both default off");
        let net = build(&format!("{base}[yolonas]\nclasses=1\nreg_max=8\nmin_object_cells=2\nvfl_gamma=1.5\n")).unwrap();
        let LayerKind::Yolonas(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.kind, ModernYoloKind::Yolonas);
        assert_eq!(s.coords(), 4 * 9);
        assert_eq!((s.min_object_cells, s.vfl_gamma), (2.0, 1.5));
        // YOLOX ignores `reg_max` and uses four raw box entries.
        let net = build(
            "[net]\nwidth=2\nheight=2\nchannels=7\n[convolutional]\nfilters=7\nsize=1\n[yolox]\nclasses=2\nreg_max=16\n",
        )
        .unwrap();
        let LayerKind::Yolox(s) = &net.layers[1].kind else { panic!() };
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
        let LayerKind::Contrastive(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.classes, 1000);
        assert_eq!(s.yolo_layer, 0, "0 selects the classifier branch");
        assert_eq!((s.temperature, s.cls_normalizer, s.max_delta), (1.0, 1.0, f32::MAX));
        assert_eq!(s.contrastive_neg_max, 3);
        assert!(net.layers[1].weights().is_none());
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        let net = build(&format!(
            "{base}[contrastive]\nclasses=3\nyolo_layer=-1\ntemperature=0.5\ncls_normalizer=2\n\
             max_delta=4\ncontrastive_neg_max=7\n"
        ))
        .unwrap();
        let LayerKind::Contrastive(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.classes, 3);
        assert_eq!(s.yolo_layer, -1, "a negative index is relative, as in C++");
        assert_eq!((s.temperature, s.cls_normalizer, s.max_delta, s.contrastive_neg_max), (0.5, 2.0, 4.0, 7));

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
        let LayerKind::OmdLoss(s) = &net.layers[1].kind else { panic!() };
        assert_eq!((s.classes, s.max_boxes, s.truth_size), (0, 200, 6));
        assert_eq!((s.cls_normalizer, s.margin, s.max_delta), (0.03, 0.5, 10.0));
        assert_eq!((s.positive_weight, s.negative_weight, s.background_weight), (1.0, 1.0, 1.0));
        assert_eq!((s.small_threshold, s.small_boost, s.max_samples), (0.0, 1.0, 128));
        assert_eq!((s.background_samples, s.pool, s.eps), (0, 0, 1e-6));
        assert!(!s.show_details);
        assert!(net.layers[1].weights().is_none(), "the layer has no weights");
        assert_eq!(net.layers[1].output_shape, net.layers[0].output_shape);

        // Primary key wins over its alias; the loss weight has three spellings.
        for (keys, want) in [
            ("max_boxes=7\nmax=9", 7usize),
            ("max=9", 9),
        ] {
            let net = build(&format!("{base}[omd_loss]\n{keys}\n")).unwrap();
            let LayerKind::OmdLoss(s) = &net.layers[1].kind else { panic!() };
            assert_eq!(s.max_boxes, want, "{keys}");
        }
        for (key, want) in [("lambda", 0.03f32), ("normalizer", 0.03), ("cls_normalizer", 0.03)] {
            let net = build(&format!("{base}[omd_loss]\n{key}=0.5\n")).unwrap();
            let LayerKind::OmdLoss(s) = &net.layers[1].kind else { panic!() };
            assert!((s.cls_normalizer - 0.5).abs() < 1e-6, "{key}");
            let _ = want;
        }
        let net = build(&format!("{base}[omd_loss]\npos_weight=2\nbg_samples=4\nsmall_threshold_px=8\n")).unwrap();
        let LayerKind::OmdLoss(s) = &net.layers[1].kind else { panic!() };
        assert_eq!(s.positive_weight, 2.0);
        assert_eq!(s.background_samples, 4);
        assert_eq!(s.small_threshold, 8.0);

        // C++ validation.
        assert!(build(&format!("{base}[omd_loss]\nmax_samples=1\n")).is_err());
        assert!(build(&format!("{base}[omd_loss]\nmargin=-0.1\n")).is_err());
        assert!(build(&format!("{base}[omd_loss]\nmargin=2.5\n")).is_err());
        assert!(build(&format!("{base}[omd_loss]\nmargin=2\n")).is_ok(), "2 is inclusive");
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

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
            other => Self::Other(other.to_owned()),
        }
    }

    pub fn to_cuda_f32_kind(&self) -> Option<F32ActivationKind> {
        match self {
            Self::Linear => Some(F32ActivationKind::Linear),
            Self::Logistic => Some(F32ActivationKind::Logistic),
            Self::Loggy => None,
            Self::Relu => Some(F32ActivationKind::Relu),
            Self::Leaky => Some(F32ActivationKind::Leaky),
            Self::Mish => Some(F32ActivationKind::Mish),
            Self::Swish => Some(F32ActivationKind::Swish),
            Self::Tanh => Some(F32ActivationKind::Tanh),
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaxPoolSpec {
    pub size: usize,
    pub stride_x: usize,
    pub stride_y: usize,
    pub padding: usize,
    pub out_channels: usize,
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
    Softmax { groups: usize },
    Cost { kind: String },
    Dropout { probability: Option<f32> },
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
            Self::Softmax { .. } => "softmax",
            Self::Cost { .. } => "cost",
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

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub index: usize,
    pub line: usize,
    pub input_shape: TensorShape,
    pub output_shape: TensorShape,
    pub kind: LayerKind,
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
            let output_shape = infer_output_shape(&kind, layer_index, input_shape, &layers)?;

            layers.push(Layer {
                index: layer_index,
                line: section.line,
                input_shape,
                output_shape,
                kind,
            });

            current = output_shape;
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
        "max" | "maxpool" => {
            let stride = section.get_usize("stride", 1)?;
            let size = section.get_usize("size", stride)?;
            LayerKind::MaxPool(MaxPoolSpec {
                size,
                stride_x: section.get_usize("stride_x", stride)?,
                stride_y: section.get_usize("stride_y", stride)?,
                padding: section.get_usize("padding", size.saturating_sub(1))?,
                out_channels: section.get_usize("out_channels", current.channels)?,
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
        },
        "dropout" => LayerKind::Dropout {
            probability: section
                .get("probability")
                .map(|value| value.parse().unwrap_or(0.5)),
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
            let channels = if spec.out_channels == 1 {
                current.channels
            } else {
                spec.out_channels
            };
            Ok(TensorShape::new(
                pool_dim(current.width, spec.padding, spec.size, spec.stride_x),
                pool_dim(current.height, spec.padding, spec.size, spec.stride_y),
                channels,
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
        | LayerKind::Dropout { .. }
        | LayerKind::Contrastive { .. } => Ok(current),
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

    use super::{LayerKind, LrPolicy, Network};

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

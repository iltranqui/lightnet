use std::fmt::{Display, Formatter};
use std::path::Path;

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
    pub subdivisions: usize,
    pub learning_rate: f32,
    pub momentum: f32,
    pub decay: f32,
    pub max_batches: usize,
    pub burn_in: usize,
    pub policy: LrPolicy,
    pub steps: Vec<usize>,
    pub scales: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LrPolicy {
    Constant,
    Steps,
    Poly,
}

impl NetConfig {
    fn from_section(section: &RawSection) -> Result<Self> {
        if section.name != "net" && section.name != "network" {
            return Err(Error::InvalidConfig(format!(
                "first section must be [net], found [{}] at line {}",
                section.name, section.line
            )));
        }

        let width = section.get_usize("width", 0)?;
        let height = section.get_usize("height", 0)?;
        let channels = section.get_usize("channels", 0)?;

        if width == 0 || height == 0 || channels == 0 {
            return Err(Error::InvalidConfig(format!(
                "[net] must define width, height, and channels (line {})",
                section.line
            )));
        }

        let policy = match section.get_str("policy", "constant").to_ascii_lowercase().as_str() {
            "constant" => LrPolicy::Constant,
            "steps" => LrPolicy::Steps,
            "poly" => LrPolicy::Poly,
            other => {
                return Err(Error::InvalidConfig(format!(
                    "unsupported learning rate policy {other} in [net]"
                )));
            }
        };
        let steps = section
            .get("steps")
            .map(|_| section.get_usize_list("steps"))
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

        Ok(Self {
            width,
            height,
            channels,
            batch: section.get_usize("batch", 1)?,
            subdivisions: section.get_usize("subdivisions", 1)?,
            learning_rate: section.get_f32("learning_rate", 0.001)?,
            momentum: section.get_f32("momentum", 0.9)?,
            decay: section.get_f32("decay", 0.0005)?,
            max_batches: section.get_usize("max_batches", 0)?,
            burn_in: section.get_usize("burn_in", 0)?,
            policy,
            steps,
            scales,
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
    pub iou_loss: YoloIouLoss,
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
    pub activation: Activation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SamSpec {
    pub from: isize,
    pub activation: Activation,
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
    Softmax {
        groups: usize,
    },
    Cost {
        kind: String,
    },
    Dropout {
        probability: Option<f32>,
    },
    ScaleChannels(ScaleChannelsSpec),
    Sam(SamSpec),
    Contrastive {
        classes: usize,
        yolo_layer: isize,
    },
    Rnn {
        hidden: usize,
        output: usize,
    },
    Lstm {
        output: usize,
        batch_normalize: bool,
    },
    Crnn {
        hidden: usize,
        output: usize,
        size: usize,
    },
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
            Self::Rnn { .. } => "rnn",
            Self::Lstm { .. } => "lstm",
            Self::Crnn { .. } => "crnn",
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
            _ => None,
        }
    }
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
        let Some(net_section) = cfg.sections.first() else {
            return Err(Error::InvalidConfig(format!(
                "empty config file {}",
                cfg.path.display()
            )));
        };

        let net = NetConfig::from_section(net_section)?;
        let mut current = net.input_shape();
        let mut layers = Vec::new();

        for (index, section) in cfg.sections.iter().enumerate().skip(1) {
            let layer_index = index - 1;
            let input_shape = current;
            let kind = parse_layer_kind(section, layer_index, current, &layers)?;
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
) -> Result<LayerKind> {
    let name = section.name.as_str();
    Ok(match name {
        "convolutional" => {
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
        "connected" => LayerKind::Connected(ConnectedSpec {
            output: section.get_usize("output", 1)?,
            batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
            activation: Activation::parse(&section.get_str("activation", "logistic")),
        }),
        "maxpool" => {
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
        "local_avgpool" => {
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
        "avgpool" => LayerKind::AvgPool,
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
            LayerKind::Yolo(YoloSpec {
                classes: section.get_usize("classes", 20)?,
                num: mask.len().max(section.get_usize("num", 1)?),
                mask,
                anchors: section.get_f32_list("anchors")?,
                scale_x_y: section.get_f32("scale_x_y", 1.0)?,
                new_coords: section.get_i64("new_coords", 0)? != 0,
                cls_normalizer: section.get_f32("cls_normalizer", 1.0)?,
                obj_normalizer: section.get_f32("obj_normalizer", 1.0)?,
                iou_normalizer: section.get_f32("iou_normalizer", 0.07)?,
                ignore_thresh: section.get_f32("ignore_thresh", 0.5)?,
                truth_thresh: section.get_f32("truth_thresh", 1.0)?,
                iou_loss: YoloIouLoss::parse(&section.get_str("iou_loss", "mse"))?,
            })
        }
        "gaussian_yolo" => {
            let mask = section.get_usize_list("mask")?;
            LayerKind::GaussianYolo(YoloSpec {
                classes: section.get_usize("classes", 20)?,
                num: mask.len().max(section.get_usize("num", 1)?),
                mask,
                anchors: section.get_f32_list("anchors")?,
                scale_x_y: section.get_f32("scale_x_y", 1.0)?,
                new_coords: section.get_i64("new_coords", 0)? != 0,
                cls_normalizer: section.get_f32("cls_normalizer", 1.0)?,
                obj_normalizer: section.get_f32("obj_normalizer", 1.0)?,
                iou_normalizer: section.get_f32("iou_normalizer", 0.07)?,
                ignore_thresh: section.get_f32("ignore_thresh", 0.5)?,
                truth_thresh: section.get_f32("truth_thresh", 1.0)?,
                iou_loss: YoloIouLoss::parse(&section.get_str("iou_loss", "mse"))?,
            })
        }
        "region" => LayerKind::Region(RegionSpec {
            classes: section.get_usize("classes", 20)?,
            num: section.get_usize("num", 1)?,
            coords: section.get_usize("coords", 4)?,
        }),
        "reorg" => LayerKind::Reorg(ReorgSpec {
            stride: section.get_usize("stride", 1)?,
            reverse: section.get_i64("reverse", 0)? != 0,
        }),
        "softmax" => LayerKind::Softmax {
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
        "rnn" => LayerKind::Rnn {
            hidden: section.get_usize("hidden", current.element_count())?,
            output: section.get_usize("output", current.element_count())?,
        },
        "lstm" => LayerKind::Lstm {
            output: section.get_usize("output", current.element_count())?,
            batch_normalize: section.get_i64("batch_normalize", 0)? != 0,
        },
        "crnn" => LayerKind::Crnn {
            hidden: section.get_usize("hidden", current.channels)?,
            output: section.get_usize("output", current.channels)?,
            size: section.get_usize("size", 3)?,
        },
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
                if from_shape.width != current.width
                    || from_shape.height != current.height
                    || from_shape.channels != current.channels
                {
                    return Err(Error::InvalidConfig(format!(
                        "shortcut layer #{layer_index} mixes incompatible shapes {} and {}",
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
        | LayerKind::ScaleChannels(_)
        | LayerKind::Sam(_)
        | LayerKind::Contrastive { .. } => Ok(current),
        LayerKind::Rnn { output, .. } | LayerKind::Lstm { output, .. } => {
            Ok(TensorShape::new(1, 1, *output, current.batch))
        }
        LayerKind::Crnn { output, .. } => Ok(TensorShape::new(
            current.width,
            current.height,
            *output,
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

fn conv_dim(input: usize, padding: usize, size: usize, stride: usize) -> usize {
    ((input + 2 * padding).saturating_sub(size) / stride.max(1)) + 1
}

fn pool_dim(input: usize, padding: usize, size: usize, stride: usize) -> usize {
    ((input + padding).saturating_sub(size) / stride.max(1)) + 1
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::cfg::ConfigFile;

    use super::{LayerKind, LrPolicy, Network};

    fn repo_cfg(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../cfg")
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

        assert_eq!(network.net.batch, 8);
        assert_eq!(network.net.subdivisions, 2);
        assert_eq!(network.net.learning_rate, 0.01);
        assert_eq!(network.net.momentum, 0.8);
        assert_eq!(network.net.decay, 0.0001);
        assert_eq!(network.net.burn_in, 100);
        assert_eq!(network.net.max_batches, 900);
        assert_eq!(network.net.policy, LrPolicy::Steps);
        assert_eq!(network.net.steps, vec![400, 800]);
        assert_eq!(network.net.scales, vec![0.1, 0.1]);
    }
}

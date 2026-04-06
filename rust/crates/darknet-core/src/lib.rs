pub mod cfg;
pub mod detect;
pub mod error;
pub mod image;
pub mod model;
pub mod names;
pub mod runtime;
pub mod train;
pub mod weights;

pub use cfg::{ConfigFile, RawSection};
pub use detect::{Detection, DetectionBox, DetectionOptions, decode_yolo_detections};
pub use error::{Error, Result};
pub use image::PlanarRgbImage;
pub use model::{
    Activation, Layer, LayerKind, LayerWeights, LrPolicy, NetConfig, Network, TensorShape,
    WeightsArray,
};
pub use names::load_names_file;
pub use runtime::{
    BackwardResult, BatchNormTrainBuffers, ConnectedTrainBuffers, ConvolutionalTrainBuffers,
    ExecutableNetwork, ForwardResult, GpuExecutableNetwork, GpuForwardState, GpuForwardSummary,
    LayerParams, LayerTrainBuffers, Tensor, YoloHeadOutput, compute_learning_rate,
};
pub use train::{
    DataConfig, TrainingBatch, YoloLabel, load_paths_file, load_training_batch,
    load_training_batch_from_paths, load_yolo_labels, replace_image_to_label,
};
pub use weights::{WeightsHeader, WeightsInspection, inspect_weights_file, save_weights};

pub const DARKNET_WEIGHTS_VERSION_MAJOR: i32 = 0;
pub const DARKNET_WEIGHTS_VERSION_MINOR: i32 = 2;
pub const DARKNET_WEIGHTS_VERSION_PATCH: i32 = 5;

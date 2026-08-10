//! CPU-first Rust implementation of the Darknet/YOLO runtime.
//!
//! The port keeps Darknet's observable contracts—`.cfg` parsing, planar
//! `B x C x H x W` tensors, layer ordering, `.weights` byte ordering,
//! detector target assignment, and SGD update semantics—while replacing
//! pointer-owned C++ objects with explicit Rust values.  Parsing, model
//! description, execution, training, and post-processing are separate modules
//! so the CPU executor can remain the correctness reference for GPU backends.
//!
//! The public re-exports below form the application-facing API.  The modules
//! remain public for advanced users extending a specific part of the port.

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
    Activation, CrnnSpec, Layer, LayerKind, LayerWeights, LrPolicy, LstmSpec, NetConfig, Network,
    RnnSpec, TensorShape, WeightsArray, YoloIouLoss,
};
pub use names::load_names_file;
pub use runtime::{
    BackwardResult, BatchNormTrainBuffers, ConnectedTrainBuffers, ConvolutionalTrainBuffers,
    CrnnParams, CrnnTrainBuffers, ExecutableNetwork, ForwardResult, GpuExecutableNetwork,
    GpuForwardState, GpuForwardSummary, LayerParams, LayerTrainBuffers, LstmParams,
    LstmTrainBuffers, RecurrentForwardCache, RnnParams, RnnTrainBuffers, Tensor, YoloHeadOutput,
    compute_learning_rate,
};
pub use train::{
    DataConfig, TrainingBatch, YoloLabel, YoloTrainingLoss, YoloTrainingState,
    compute_yolo_training_loss, compute_yolo_training_loss_with_state, load_paths_file,
    load_training_batch, load_training_batch_from_paths, load_yolo_labels, replace_image_to_label,
};
pub use weights::{WeightsHeader, WeightsInspection, inspect_weights_file, save_weights};

pub const DARKNET_WEIGHTS_VERSION_MAJOR: i32 = 0;
pub const DARKNET_WEIGHTS_VERSION_MINOR: i32 = 2;
pub const DARKNET_WEIGHTS_VERSION_PATCH: i32 = 5;

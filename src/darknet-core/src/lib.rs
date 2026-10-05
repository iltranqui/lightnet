//! `float_erf` is needed by `[laya_scorer]`, whose activation is the *exact
//! erf* GELU (torch `nn.GELU()`'s default) rather than darknet's tanh
//! approximation; C++ uses `std::erf` there, so a hand-rolled approximation
//! would be a real numerical divergence. The toolchain is already pinned to
//! nightly (see `rust-toolchain.toml`).
#![feature(float_erf)]

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

pub(crate) mod activation_math;
pub mod augmentation;
pub use augmentation::DetectorAugmentation;
pub mod cfg;
pub(crate) mod channel_shuffle;
pub(crate) mod channel_slice;
pub mod centernet;
pub mod chart;
pub(crate) mod contrastive;
pub use chart::{ChartPoint, write_training_chart};
pub mod dense_loss;
pub use dense_loss::{DenseTrainingLoss, compute_dense_training_loss, compute_dense_training_loss_at};
pub mod detect;
pub(crate) mod deconvolutional;
pub mod error;
pub mod evaluate;
pub use evaluate::{MapReport, REPORT_CONFIDENCE, ValidationDataset, detector_map};
pub mod image;
pub mod laya_scorer;
pub mod moe_conv;
pub(crate) mod jev_ce;
pub mod initialize;
pub mod loader_profile;
pub use initialize::initialize_training_params;
pub mod model;
pub(crate) mod maxpool_depth;
pub mod modern_yolo;
pub mod names;
pub(crate) mod omd_loss;
pub mod runtime;
pub(crate) mod segmentation;
pub(crate) mod shortcut_weights;
pub mod train;
pub mod weights;

pub use cfg::{ConfigFile, RawSection};
pub use detect::{
    Detection, DetectionBox, DetectionOptions, decode_modern_detections, decode_yolo_detections,
};
pub use error::{Error, Result};
pub use image::PlanarRgbImage;
pub use model::{
    Activation, CrnnSpec, Layer, LayerKind, LayerWeights, LrPolicy, LstmSpec, NetConfig, Network,
    RnnSpec, ShortcutWeightsNorm, ShortcutWeightsType, TensorShape, WeightsArray, YoloIouLoss,
};
pub use names::load_names_file;
pub use runtime::{
    BackwardResult, BatchNormTrainBuffers, ConnectedTrainBuffers, ConvolutionalTrainBuffers,
    CrnnParams, CrnnTrainBuffers, ExecutableNetwork, ForwardResult, GpuExecutableNetwork,
    GpuForwardState, GpuForwardSummary, LayerParams, LayerTrainBuffers, LstmParams,
    LstmTrainBuffers, RecurrentForwardCache, RnnParams, RnnTrainBuffers, ShortcutParams,
    ShortcutTrainBuffers, Tensor, YoloHeadOutput, compute_learning_rate,
};
pub use train::{
    DataConfig, TrainingBatch, YoloLabel, YoloTrainingLoss, YoloTrainingState,
    compute_yolo_training_loss, compute_yolo_training_loss_with_state,
    load_augmented_training_batch_from_paths, load_paths_file, load_training_batch,
    load_training_batch_from_paths, load_yolo_labels, replace_image_to_label,
};
pub use weights::{WeightsHeader, WeightsInspection, inspect_weights_file, save_weights};

pub const DARKNET_WEIGHTS_VERSION_MAJOR: i32 = 0;
pub const DARKNET_WEIGHTS_VERSION_MINOR: i32 = 2;
pub const DARKNET_WEIGHTS_VERSION_PATCH: i32 = 5;

//! Executable networks for the Darknet/YOLO port: the CPU reference executor,
//! the backend-generic GPU executor, the per-layer GPU glue (`layers/*/gpu.rs`),
//! `.weights` I/O, parameter initialisation, the Muon optimiser state and the
//! ONNX exporter.
//!
//! This crate sits between `lightnet-layers` (CPU reference math) and
//! `lightnet-train` (losses, augmentation, detection decode). It never names
//! `train` or `detect`. Module names that used to be crate-private in
//! `lightnet-core` are `pub` and `#[doc(hidden)]` here so the upper crates can
//! reach them; `lightnet-core` re-exports every old public path.

#[doc(hidden)]
pub mod layers;

#[doc(hidden)]
pub use lightnet_model::activation_math;
#[doc(hidden)]
pub use lightnet_model::axis_fixture;
#[doc(hidden)]
pub use lightnet_model::detection_geometry;
pub use lightnet_model::{cfg, error, model, names, training_types};

pub use layers::carpn;
pub use layers::centernet;
#[doc(hidden)]
pub use layers::channel_shuffle;
#[doc(hidden)]
pub use layers::channel_slice;
pub use layers::clifford;
#[doc(hidden)]
pub use layers::contrastive;
#[doc(hidden)]
pub use layers::dcnv4;
#[doc(hidden)]
pub use layers::deconvolutional;
#[doc(hidden)]
pub use layers::deform_conv;
pub use layers::depth_anything;
#[doc(hidden)]
pub use layers::dpt_head;
#[doc(hidden)]
pub use layers::dsconv;
#[doc(hidden)]
pub use layers::eml_conv_math;
#[doc(hidden)]
pub use layers::foveation_film;
#[doc(hidden)]
pub use layers::graph_conv;
#[doc(hidden)]
pub use layers::hyperbolic_foveation;
#[doc(hidden)]
pub use layers::jev_ce;
pub use layers::kimi_kda;
pub use layers::laya_act_head;
pub use layers::laya_encoder;
pub use layers::laya_head;
pub use layers::laya_loss;
pub use layers::laya_marker_gather;
pub use layers::laya_scorer;
pub use layers::mambavision;
#[doc(hidden)]
pub use layers::maxpool_depth;
pub use layers::modern_yolo;
pub use layers::moe_conv;
#[doc(hidden)]
pub use layers::omd_loss;
#[doc(hidden)]
pub use layers::segmentation;
#[doc(hidden)]
pub use layers::shortcut_weights;
#[doc(hidden)]
pub use layers::swin;
#[doc(hidden)]
pub use layers::swin::block as swin_block;
#[doc(hidden)]
pub use layers::tucker_attention;
#[doc(hidden)]
pub use layers::vc_attention;
#[doc(hidden)]
pub use layers::vit;
#[doc(hidden)]
pub use layers::wmhf;

pub mod image;
pub mod initialize;
pub mod loader_profile;
#[doc(hidden)]
pub mod muon;
#[doc(hidden)]
pub mod muon_runtime;
pub mod onnx;
pub mod runtime;
pub mod weights;

pub use carpn::{ObbRefineOutput, ObbRefineProposal};
pub use cfg::{ConfigFile, RawSection};
pub use clifford::{CliffordBranchTrainBuffers, CliffordParams, CliffordTrainBuffers};
pub use dpt_head::DptHeadWeights;
pub use error::{Error, Result};
pub use image::PlanarRgbImage;
pub use initialize::initialize_training_params;
pub use kimi_kda::{KimiKdaForwardCache, KimiKdaParams};
pub use mambavision::{
    MambaVisionForward, MambaVisionGradients, MambaVisionParams, MambaVisionState,
    mambavision_backward, mambavision_forward,
};
pub use model::{
    Activation, CliffordContextMode, CliffordGffnMode, CliffordGlobalOp, CliffordHigherMode,
    CliffordInteractionMode, CliffordSpec, CrnnSpec, DeformConvSpec, DsConvSpec, HfInverseMethod,
    HfRadiusMode, HfWarpSource, HyperbolicFoveationSpec, KimiKdaSpec, Layer, LayerKind,
    LayerWeights, LrPolicy, LstmSpec, MambaVisionSpec, NetConfig, Network, ObbRefineLoss,
    ObbRefineSpec, RecursiveBlockSpec, RnnSpec, ShortcutWeightsNorm, ShortcutWeightsType, SwinSpec,
    TensorShape, TuckerAttentionSpec, WeightsArray, WmhfSpec, YoloIouLoss,
};
pub use names::load_names_file;
pub use onnx::{OnnxOptions, export_onnx};
pub use runtime::{
    BackwardResult, BatchNormTrainBuffers, ConnectedTrainBuffers, ConvolutionalTrainBuffers,
    CrnnParams, CrnnTrainBuffers, DeformConvParams, DeformConvTrainBuffers, DsConvParams,
    DsConvTrainBuffers, ExecutableNetwork, ForwardResult, GpuExecutableNetwork, GpuForwardState, GpuYoloLoss,
    GpuForwardSummary, HyperbolicFoveationParams, HyperbolicFoveationTrainBuffers,
    KimiKdaTrainBuffers, LayerParams, LayerTrainBuffers, LossScaler, LstmParams, LstmTrainBuffers,
    MambaVisionTrainBuffers, RecurrentForwardCache, ScaleStep, RecursiveBlockForwardCache,
    RecursiveBlockParams, RecursiveBlockTrainBuffers, RnnParams, RnnTrainBuffers, ShortcutParams,
    ShortcutTrainBuffers, Tensor, TuckerAttentionTrainBuffers, WmhfParams, WmhfTrainBuffers,
    Yolo3dHeadOutput, YoloHeadOutput, compute_learning_rate,
};
pub use swin::SwinWeights;
pub use swin_block::SwinForwardCache;
pub use tucker_attention::{TuckerAttentionForwardCache, TuckerAttentionParams};
pub use weights::{WeightsHeader, WeightsInspection, inspect_weights_file, save_weights};

pub const DARKNET_WEIGHTS_VERSION_MAJOR: i32 = 0;
pub const DARKNET_WEIGHTS_VERSION_MINOR: i32 = 2;
pub const DARKNET_WEIGHTS_VERSION_PATCH: i32 = 5;

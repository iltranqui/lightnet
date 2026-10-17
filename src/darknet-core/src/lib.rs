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
pub mod carpn;
pub mod centernet;
pub mod cfg;
pub(crate) mod channel_shuffle;
pub(crate) mod channel_slice;
pub mod chart;
pub(crate) mod contrastive;
pub(crate) mod dcnv4;
pub(crate) mod deform_conv;
pub(crate) mod dsconv;
pub(crate) mod graph_conv;
pub use chart::{ChartPoint, write_training_chart};
pub mod dense_loss;
pub use dense_loss::{
    DenseTrainingLoss, compute_dense_training_loss, compute_dense_training_loss_at,
};
pub(crate) mod deconvolutional;
pub mod detect;
pub(crate) mod eml_conv_math;
pub mod error;
pub mod evaluate;
pub use evaluate::{
    MapReport, ObbValidationDataset, REPORT_CONFIDENCE, ValidationDataset, carpn_detector_map,
    detector_map, obb_detector_map,
};
pub mod image;
pub mod initialize;
pub(crate) mod jev_ce;
pub mod kimi_kda;
pub mod laya_act_head;
pub mod laya_encoder;
pub mod laya_head;
pub mod laya_loss;
pub mod laya_marker_gather;
pub mod laya_scorer;
pub mod loader_profile;
pub mod moe_conv;
pub use initialize::initialize_training_params;
pub mod mambavision;
pub(crate) mod maxpool_depth;
pub mod model;
pub mod modern_yolo;
pub mod names;
pub mod onnx;
pub use onnx::{OnnxOptions, export_onnx};
pub mod obb_geometry;
pub mod obb_labels;
pub use obb_labels::{ObbLabel, flip_obb_labels, load_obb_labels, rotate_obb_labels};
pub mod obb_training;
pub use obb_training::{
    OBB_YOLO_TRUTH_SIZE, ObbYoloAnchorAssignment, ObbYoloBatchCellMatch,
    ObbYoloBatchCellPrediction, ObbYoloBatchPositiveAssignment, ObbYoloBatchTrainingDeltaStats,
    ObbYoloBoxDeltaConfig, ObbYoloCellTruthMatch, ObbYoloClassDeltaResult, ObbYoloClassDeltaStats,
    ObbYoloFrontPointDeltaConfig, ObbYoloFrontPointDeltas, ObbYoloFrontPointMetrics,
    ObbYoloHeadAssignmentSpec, ObbYoloHeadMatchConfig, ObbYoloHeadTrainingDeltaStats,
    ObbYoloMatchScore, ObbYoloMatchScoreBranch, ObbYoloMatchScoreConfig, ObbYoloPositiveAssignment,
    ObbYoloPositiveKind, ObbYoloPositiveObjectnessConfig, ObbYoloPredictionMatchContext,
    ObbYoloTrainingHeadConfig, ObbYoloTruthBatch, apply_obb_yolo_batch_head_objectness_deltas,
    apply_obb_yolo_batch_head_positive_box_deltas, apply_obb_yolo_batch_head_positive_class_deltas,
    apply_obb_yolo_batch_head_positive_front_point_deltas,
    apply_obb_yolo_batch_head_positive_objectness_deltas,
    apply_obb_yolo_batch_head_training_deltas,
    apply_obb_yolo_batch_head_training_deltas_with_front_point,
    apply_obb_yolo_batch_head_truth_threshold_deltas,
    apply_obb_yolo_batch_head_truth_threshold_front_point_deltas,
    apply_obb_yolo_batch_multi_head_training_deltas,
    apply_obb_yolo_batch_multi_head_training_deltas_with_front_point,
    apply_obb_yolo_batch_multi_head_training_deltas_with_front_point_configs,
    assign_obb_yolo_anchor, build_obb_yolo_training_head_config, decode_obb_yolo_front_point,
    encode_obb_yolo_truth_batch, enumerate_obb_yolo_batch_positives,
    enumerate_obb_yolo_head_positives, match_obb_yolo_batch_cell_predictions,
    match_obb_yolo_cell_prediction, match_obb_yolo_cell_prediction_with_score_config,
    obb_yolo_anchor_match_score, obb_yolo_angular_prediction_match_score_with_viou,
    obb_yolo_box_regression_deltas, obb_yolo_cell_objectness_delta,
    obb_yolo_ciou_prediction_match_score, obb_yolo_class_deltas,
    obb_yolo_cobb_prediction_match_score_with_viou, obb_yolo_covariance_prediction_match_score,
    obb_yolo_covariance_prediction_match_score_with_viou, obb_yolo_front_point_regression_deltas,
    obb_yolo_match_score, select_obb_yolo_cell_truth_match, visit_obb_yolo_batch_head_matches,
};
pub(crate) mod omd_loss;
pub mod runtime;
pub(crate) mod segmentation;
pub(crate) mod shortcut_weights;
pub(crate) mod swin;
pub(crate) mod swin_block;
pub mod train;
pub(crate) mod tucker_attention;
#[allow(dead_code)] // VC reference helpers are integrated incrementally.
pub(crate) mod vc_attention;
pub(crate) mod vit;
pub mod weights;
pub(crate) mod wmhf;

pub use carpn::{ObbRefineOutput, ObbRefineProposal};
pub use cfg::{ConfigFile, RawSection};
pub use detect::{
    Detection, DetectionBox, DetectionOptions, ObbDetection, ObbDetectionGeometry, apply_obb_nms,
    decode_modern_detections, decode_obb_network_detections, decode_obb_yolo_detections,
    decode_obb_yolo_head, decode_yolo_detections,
};
pub use error::{Error, Result};
pub use image::PlanarRgbImage;
pub use kimi_kda::{KimiKdaForwardCache, KimiKdaParams};
pub use mambavision::{
    MambaVisionForward, MambaVisionGradients, MambaVisionParams, MambaVisionState,
    mambavision_backward, mambavision_forward,
};
pub use model::{
    Activation, CrnnSpec, DeformConvSpec, DsConvSpec, KimiKdaSpec, Layer, LayerKind, LayerWeights,
    LrPolicy, LstmSpec, MambaVisionSpec, NetConfig, Network, ObbRefineLoss, ObbRefineSpec, RnnSpec,
    ShortcutWeightsNorm, ShortcutWeightsType, SwinSpec, TensorShape, TuckerAttentionSpec,
    WeightsArray, WmhfSpec, YoloIouLoss,
};
pub use names::load_names_file;
pub use runtime::{
    BackwardResult, BatchNormTrainBuffers, ConnectedTrainBuffers, ConvolutionalTrainBuffers,
    CrnnParams, CrnnTrainBuffers, DeformConvParams, DeformConvTrainBuffers, DsConvParams,
    DsConvTrainBuffers, ExecutableNetwork, ForwardResult, GpuExecutableNetwork, GpuForwardState,
    GpuForwardSummary, KimiKdaTrainBuffers, LayerParams, LayerTrainBuffers, LstmParams,
    LstmTrainBuffers, MambaVisionTrainBuffers, RecurrentForwardCache, RnnParams, RnnTrainBuffers,
    ShortcutParams, ShortcutTrainBuffers, Tensor, TuckerAttentionTrainBuffers, WmhfParams,
    WmhfTrainBuffers, YoloHeadOutput, compute_learning_rate,
};
pub use swin::SwinWeights;
pub use swin_block::SwinForwardCache;
pub use train::{
    DataConfig, ObbTrainingBatch, ObbYoloTrainingBackward, ObbYoloTrainingDeltaBatch,
    TrainingBatch, YoloLabel, YoloTrainingLoss, YoloTrainingState, backward_obb_yolo_training,
    compute_obb_yolo_training_deltas, compute_yolo_training_loss,
    compute_yolo_training_loss_with_state, load_augmented_training_batch_from_paths,
    load_obb_training_batch, load_obb_training_batch_from_paths,
    load_obb_training_batch_with_flips_from_paths, load_paths_file, load_training_batch,
    load_training_batch_from_paths, load_yolo_labels, replace_image_to_label,
    train_obb_yolo_batch_step, train_obb_yolo_subdivisions,
};
pub use tucker_attention::{TuckerAttentionForwardCache, TuckerAttentionParams};
pub use weights::{WeightsHeader, WeightsInspection, inspect_weights_file, save_weights};

pub const DARKNET_WEIGHTS_VERSION_MAJOR: i32 = 0;
pub const DARKNET_WEIGHTS_VERSION_MINOR: i32 = 2;
pub const DARKNET_WEIGHTS_VERSION_PATCH: i32 = 5;

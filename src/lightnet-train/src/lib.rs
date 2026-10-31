//! Training, detection decode and evaluation for the Darknet/YOLO port: the
//! YOLO and dense-head losses, data augmentation and batch loading, OBB and
//! 3D-cuboid label handling, mean-average-precision evaluation and the
//! training chart.
//!
//! This crate sits above `lightnet-runtime` and re-exports everything below it
//! (`pub use lightnet_runtime::*`), so `crate::runtime::...`, `crate::model::...`
//! and the other module paths keep working inside the moved sources.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub use lightnet_runtime::*;

pub mod augmentation;
pub mod chart;
pub mod cuboid_labels;
pub mod dense_loss;
pub mod detect;
pub mod evaluate;
pub mod obb_geometry;
pub mod obb_labels;
pub mod obb_training;
pub mod segmentation_data;
pub mod train;
pub mod yolo_3d;

pub use augmentation::DetectorAugmentation;
pub use chart::{ChartPoint, write_training_chart};
pub use cuboid_labels::{
    CuboidLabel, MixedLabel, Yolo3dCrop, Yolo3dImageMeta, Yolo3dTruthBatch,
    encode_mixed_yolo_3d_truths, encode_yolo_3d_truths, load_cuboid_labels,
    load_pure_cuboid_labels, parse_cuboid_labels, parse_mixed_cuboid_labels,
};
pub use dense_loss::{
    DenseTrainingLoss, compute_dense_training_loss, compute_dense_training_loss_at,
};
pub use detect::{
    Detection, DetectionBox, DetectionOptions, ObbDetection, ObbDetectionGeometry, apply_obb_nms,
    decode_modern_detections, decode_network_detections, decode_obb_network_detections,
    decode_obb_yolo_detections, decode_obb_yolo_head, decode_yolo_detections,
};
pub use evaluate::{
    MapReport, ObbValidationDataset, REPORT_CONFIDENCE, ValidationDataset, carpn_detector_map,
    detector_map, obb_detector_map,
};
pub use obb_labels::{ObbLabel, flip_obb_labels, load_obb_labels, rotate_obb_labels};
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

//! Source-derived training primitives for legacy front-point OBB YOLO heads.
//!
//! This module currently covers direction-free anchor matching, per-truth and
//! batch/multi-head positive enumeration, per-cell and batch-cell OBB score
//! dispatch/match selection, objectness/class/front-point channel deltas, and
//! source-layout truth-row packing. Front-point deltas remain standalone and
//! are not yet routed through positive-anchor writeback or a network trainer.
//! It does not implement complete truth assignment, YOLO loss, backward
//! propagation, or optimizer integration.

use crate::detect::DetectionBox;
use crate::error::{Error, Result};
use crate::model::YoloIouLoss;
use crate::obb_geometry::{
    BoxObb, ObbCobbLossConfig, blend_obb_score_with_qbbv_viou, box_ciou_angular_quality_score,
    box_ciou_obb_quality_score, box_cobb_obb_quality_score_with_viou,
    box_covariance_w2_obb_loss_and_gradient, box_covariance_w2_obb_quality_score,
};
use crate::obb_labels::ObbLabel;

const OBB_YOLO_CLASS_CONFIDENCE_THRESHOLD: f32 = 0.25;

/// A truth's selected global anchor and its location in one OBB YOLO head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloAnchorAssignment {
    /// Index in the complete `[w0, h0, w1, h1, ...]` anchor list.
    pub anchor_index: usize,
    /// Index of `anchor_index` in the head's `mask`.
    pub head_anchor_index: usize,
    /// Grid location computed from the normalized truth center.
    pub cell_x: usize,
    pub cell_y: usize,
    /// Size IoU used to select the global anchor.
    pub match_score: f32,
}

/// Why a truth-to-head positive match was emitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObbYoloPositiveKind {
    /// The truth's highest-scoring anchor across all heads.
    BestAnchor,
    /// A non-best anchor in this head whose score strictly exceeds `iou_thresh`.
    AdditionalAnchor,
}

/// One primary or additional positive anchor match for a truth on one head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloPositiveAssignment {
    /// Index into the input truth slice.
    pub truth_index: usize,
    /// Ground-truth class before any optional class mapping.
    pub class_id: usize,
    /// Primary-best or threshold-selected additional positive.
    pub kind: ObbYoloPositiveKind,
    /// Anchor slot and grid cell selected on this head.
    pub assignment: ObbYoloAnchorAssignment,
}

/// Per-head cfg data needed to enumerate source OBB anchor positives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObbYoloHeadAssignmentSpec {
    pub mask: Vec<usize>,
    pub grid_width: usize,
    pub grid_height: usize,
}

/// Source OBB YOLO metadata for traversing one batch-major head output tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbYoloHeadMatchConfig {
    pub head_index: usize,
    pub anchors: Vec<f32>,
    pub mask: Vec<usize>,
    pub grid_width: usize,
    pub grid_height: usize,
    pub classes: usize,
    pub net_width: usize,
    pub net_height: usize,
    pub new_coords: bool,
    /// Optional source `obb_anchor_offsets` values, indexed by global anchor.
    pub front_anchor_offsets: Option<Vec<[f32; 2]>>,
    pub score_config: ObbYoloMatchScoreConfig,
    pub ignore_thresh: f32,
    pub truth_thresh: f32,
    pub objectness_normalizer: f32,
    pub objectness_smooth: bool,
}

/// Source anchor/class settings for the positive-assignment objectness pass.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbYoloPositiveObjectnessConfig {
    pub head: ObbYoloHeadAssignmentSpec,
    pub anchors: Vec<f32>,
    pub classes: usize,
    pub net_width: usize,
    pub net_height: usize,
    /// Optional source `obb_anchor_offsets` values, indexed by global anchor.
    pub front_anchor_offsets: Option<Vec<[f32; 2]>>,
    pub iou_thresh: f32,
    pub objectness_normalizer: f32,
    pub objectness_smooth: bool,
    pub class_map: Option<Vec<usize>>,
    pub class_multipliers: Option<Vec<f32>>,
}

/// One positive assignment tagged with its batch slot and OBB YOLO head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloBatchPositiveAssignment {
    pub batch_index: usize,
    pub head_index: usize,
    /// Truth index remains local to this batch image's label slice.
    pub positive: ObbYoloPositiveAssignment,
}

/// One caller-selected OBB YOLO cell/anchor prediction in a batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbYoloBatchCellPrediction {
    pub batch_index: usize,
    pub head_index: usize,
    pub anchor_index: usize,
    pub head_anchor_index: usize,
    pub prediction: [f32; 6],
    pub class_outputs: Vec<f32>,
    pub context: ObbYoloPredictionMatchContext,
    pub score_config: ObbYoloMatchScoreConfig,
    pub ignore_thresh: f32,
    pub truth_thresh: f32,
    pub predicted_objectness: f32,
    pub objectness_normalizer: f32,
    pub objectness_smooth: bool,
}

/// Prediction-conditioned match gates tagged with their batch/head/cell slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloBatchCellMatch {
    pub batch_index: usize,
    pub head_index: usize,
    pub anchor_index: usize,
    pub head_anchor_index: usize,
    pub cell_x: usize,
    pub cell_y: usize,
    pub truth_match: ObbYoloCellTruthMatch,
    pub objectness_delta: f32,
}

/// Prediction-match gates for one cell/anchor against its truth candidates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloCellTruthMatch {
    /// First truth with the maximum match score, or `None` when no truths exist.
    pub best_truth_index: Option<usize>,
    /// Highest source-style prediction match score (zero for no truths).
    pub best_score: f32,
    /// Best truth when `best_score > truth_thresh`; strict comparison.
    pub positive_truth_index: Option<usize>,
    /// Best score for ignore matching, or zero unless a class output exceeds 0.25.
    pub ignore_match_score: f32,
    /// Whether `ignore_match_score > ignore_thresh`; this may coexist with a
    /// positive match, whose source branch later overrides the ignore delta.
    pub ignore_match: bool,
}

/// Class-channel deltas for one OBB YOLO positive.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbYoloClassDeltaResult {
    /// Updated class deltas, preserving incoming values where the source leaves them untouched.
    pub deltas: Vec<f32>,
    /// Selected class output, which the caller may add to the source `avg_cat` statistic.
    pub class_output_for_average: f32,
}

/// `avg_cat` contributions from the best/additional positive-anchor pass.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObbYoloClassDeltaStats {
    pub class_output_sum: f32,
    pub positive_count: usize,
}

impl ObbYoloClassDeltaStats {
    pub fn mean_class_output(self) -> Option<f32> {
        (self.positive_count > 0).then(|| self.class_output_sum / self.positive_count as f32)
    }
}

/// Counts and class metrics produced by one head's composed OBB delta passes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObbYoloHeadTrainingDeltaStats {
    /// Truth-threshold cell positives (not included in source `avg_cat`).
    pub truth_threshold_positive_cells: usize,
    /// `avg_cat` sum/count from best/additional anchor class assignments.
    pub anchor_class_stats: ObbYoloClassDeltaStats,
    /// Best/additional positive assignments that received box deltas.
    pub anchor_box_positive_count: usize,
    /// Source `net.total_bbox`: threshold cells plus best/additional assignments.
    pub total_bbox_count: usize,
}

/// Per-head detail and source-style totals from a multi-head delta dispatch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ObbYoloBatchTrainingDeltaStats {
    /// Results in the same order as the head tensors passed to the dispatcher.
    pub per_head: Vec<ObbYoloHeadTrainingDeltaStats>,
    /// `avg_cat` sum/count aggregated across best/additional assignments only.
    pub anchor_class_stats: ObbYoloClassDeltaStats,
    /// Truth-threshold cell positives across heads (not included in `avg_cat`).
    pub truth_threshold_positive_cells: usize,
    /// Best/additional positive box assignments across heads.
    pub anchor_box_positive_count: usize,
    /// Source `net.total_bbox` aggregated across all heads.
    pub total_bbox_count: usize,
}

/// Geometry shared by the per-cell prediction-match score calculation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloPredictionMatchContext {
    /// Selected anchor dimensions in network pixels.
    pub anchor_width: f32,
    pub anchor_height: f32,
    /// Cell coordinates in the head grid.
    pub cell_x: usize,
    pub cell_y: usize,
    /// Head grid dimensions.
    pub grid_width: usize,
    pub grid_height: usize,
    /// Network input dimensions.
    pub net_width: usize,
    pub net_height: usize,
    /// Whether box width/height use `[yolo] new_coords` decoding.
    pub new_coords: bool,
}

/// The active score branch selected by source OBB loss-normalizer priority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObbYoloMatchScoreBranch {
    Covariance,
    Cobb,
    LegacyAngular,
}

/// Branch settings for source one-cell OBB matching-score dispatch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloMatchScoreConfig {
    pub covariance_normalizer: f32,
    pub cobb_normalizer: f32,
    pub covariance_front_length_ratio_k: f32,
    pub covariance_front_length_weight: f32,
    pub cobb_loss: ObbCobbLossConfig,
    pub viou_weight: f32,
}

impl Default for ObbYoloMatchScoreConfig {
    fn default() -> Self {
        Self {
            covariance_normalizer: 0.0,
            cobb_normalizer: 0.0,
            covariance_front_length_ratio_k: 1.0,
            covariance_front_length_weight: 0.35,
            cobb_loss: ObbCobbLossConfig::default(),
            viou_weight: 0.0,
        }
    }
}

/// Source configuration for finite-difference OBB box-channel deltas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloBoxDeltaConfig {
    pub score: ObbYoloMatchScoreConfig,
    pub iou_normalizer: f32,
    /// `f32::MAX` selects the source default absolute delta clip of 1.0.
    pub max_delta: f32,
    pub accumulate: bool,
}

impl Default for ObbYoloBoxDeltaConfig {
    fn default() -> Self {
        Self {
            score: ObbYoloMatchScoreConfig::default(),
            iou_normalizer: 1.0,
            max_delta: f32::MAX,
            accumulate: true,
        }
    }
}

/// Source front-point channel scaling and clipping configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloFrontPointDeltaConfig {
    /// `front_loss_normalizer`, already scaled by the source class multiplier.
    pub normalizer: f32,
    /// `f32::MAX` selects the source default absolute delta clip of 1.0.
    pub max_delta: f32,
    /// Source `MSE` front-box derivative falls back to CIoU.
    pub iou_loss: YoloIouLoss,
}

impl Default for ObbYoloFrontPointDeltaConfig {
    fn default() -> Self {
        Self {
            normalizer: 1.0,
            max_delta: f32::MAX,
            iou_loss: YoloIouLoss::Mse,
        }
    }
}

/// Raw front-channel deltas and optional center-chain contributions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObbYoloFrontPointDeltas {
    pub front_x: f32,
    pub front_y: f32,
    pub center_x: f32,
    pub center_y: f32,
    /// `max(0, 1 - front-box CIoU)`, before normalizer scaling.
    pub loss: f32,
}

/// Result of source single-prediction OBB score-branch selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbYoloMatchScore {
    pub branch: ObbYoloMatchScoreBranch,
    pub score: f32,
}

/// Number of float values in one source OBB YOLO truth row.
pub const OBB_YOLO_TRUTH_SIZE: usize = 7;

/// Source-layout OBB YOLO truth rows, stored as `[batch][box-slot][7]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbYoloTruthBatch {
    pub batch_size: usize,
    pub max_boxes: usize,
    /// Each row is `[cx, cy, w, h, class, front_x, front_y]`.
    pub values: Vec<f32>,
}

/// Encode per-image OBB labels into the seven-float rows consumed by
/// `obb_yolo_truth_from_state`. Labels are kept in caller order, absent fronts
/// use the center placeholder, and unused box slots remain all-zero. Overflow
/// is an error rather than a silent truncation.
pub fn encode_obb_yolo_truth_batch(
    labels: &[Vec<ObbLabel>],
    max_boxes: usize,
    classes: usize,
) -> Result<ObbYoloTruthBatch> {
    if labels.is_empty() || max_boxes == 0 || classes == 0 {
        return Err(Error::InvalidConfig(
            "OBB truth encoding requires a nonempty batch, max_boxes, and classes".to_owned(),
        ));
    }
    for (batch_index, image_labels) in labels.iter().enumerate() {
        if image_labels.len() > max_boxes {
            return Err(Error::InvalidConfig(format!(
                "OBB image {batch_index} has {} labels, exceeding max_boxes={max_boxes}",
                image_labels.len()
            )));
        }
    }
    let value_count = labels
        .len()
        .checked_mul(max_boxes)
        .and_then(|rows| rows.checked_mul(OBB_YOLO_TRUTH_SIZE))
        .ok_or_else(|| Error::InvalidConfig("OBB truth tensor size overflows".to_owned()))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(value_count)
        .map_err(|_| Error::InvalidConfig("OBB truth tensor allocation failed".to_owned()))?;
    values.resize(value_count, 0.0);

    for (batch_index, image_labels) in labels.iter().enumerate() {
        for (box_slot, label) in image_labels.iter().enumerate() {
            validate_truth(label)?;
            if label.class_id >= classes {
                return Err(Error::InvalidConfig(format!(
                    "OBB class id {} exceeds classes={classes} in image {batch_index}",
                    label.class_id
                )));
            }
            let class_value = label.class_id as f32;
            if label.class_id > (1usize << 24) || class_value as usize != label.class_id {
                return Err(Error::InvalidConfig(format!(
                    "OBB class id {} cannot be represented exactly in the truth tensor",
                    label.class_id
                )));
            }
            let front = if label.has_front {
                [label.front_x, label.front_y]
            } else {
                [label.x, label.y]
            };
            let row = (batch_index * max_boxes + box_slot) * OBB_YOLO_TRUTH_SIZE;
            values[row..row + OBB_YOLO_TRUTH_SIZE].copy_from_slice(&[
                label.x,
                label.y,
                label.w,
                label.h,
                class_value,
                front[0],
                front[1],
            ]);
        }
    }

    Ok(ObbYoloTruthBatch {
        batch_size: labels.len(),
        max_boxes,
        values,
    })
}

/// Compute the source's direction-free OBB YOLO anchor-match score.
///
/// This is the IoU between the normalized anchor width/height and the truth
/// width/height, both centered at the origin. The truth's front point does not
/// affect direction-free matching; the parser's center placeholder for a
/// missing front point is valid, as in the source implementation.
pub fn obb_yolo_anchor_match_score(
    truth: &ObbLabel,
    anchor_width: f32,
    anchor_height: f32,
    net_width: usize,
    net_height: usize,
) -> Result<f32> {
    validate_truth(truth)?;
    if net_width == 0 || net_height == 0 {
        return Err(Error::InvalidConfig(
            "OBB anchor matching requires nonzero network dimensions".to_owned(),
        ));
    }
    if !anchor_width.is_finite()
        || !anchor_height.is_finite()
        || anchor_width <= 0.0
        || anchor_height <= 0.0
    {
        return Err(Error::InvalidConfig(
            "OBB anchors must have positive finite width and height".to_owned(),
        ));
    }

    let anchor_width = anchor_width / net_width as f32;
    let anchor_height = anchor_height / net_height as f32;
    let intersection = anchor_width.min(truth.w) * anchor_height.min(truth.h);
    let union = anchor_width * anchor_height + truth.w * truth.h - intersection;
    if !intersection.is_finite() || !union.is_finite() || union <= 0.0 {
        return Ok(0.0);
    }
    let score = intersection / union;
    Ok(if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    })
}

/// Select the best all-head anchor, then report it only if it is present in
/// this head's mask, following the source OBB YOLO assignment loop.
///
/// `anchors` is the flat cfg representation `[w0, h0, w1, h1, ...]`; widths
/// and heights are in network pixels. Strict score comparison preserves the
/// first anchor on ties. A globally best anchor not present in `mask` returns
/// `Ok(None)` rather than falling back to the best masked anchor.
pub fn assign_obb_yolo_anchor(
    truth: &ObbLabel,
    anchors: &[f32],
    mask: &[usize],
    net_width: usize,
    net_height: usize,
    grid_width: usize,
    grid_height: usize,
) -> Result<Option<ObbYoloAnchorAssignment>> {
    validate_truth(truth)?;
    validate_anchor_assignment_inputs(
        anchors,
        mask,
        net_width,
        net_height,
        grid_width,
        grid_height,
    )?;
    let (best_anchor, best_score) = best_obb_yolo_anchor(truth, anchors, net_width, net_height)?;
    let Some(head_anchor_index) = mask.iter().position(|&index| index == best_anchor) else {
        return Ok(None);
    };
    let (cell_x, cell_y) = obb_yolo_truth_cell(truth, grid_width, grid_height);
    Ok(Some(ObbYoloAnchorAssignment {
        anchor_index: best_anchor,
        head_anchor_index,
        cell_x,
        cell_y,
        match_score: best_score,
    }))
}

/// Enumerate this head's primary and threshold-selected OBB YOLO positive matches.
///
/// For each truth, the highest-scoring anchor is selected across the complete
/// anchor list (first anchor wins ties), and emitted as `BestAnchor` only when
/// present in `mask`. Other masked anchors are emitted as `AdditionalAnchor`
/// when `iou_thresh < 1` and their size-IoU is strictly greater than the
/// threshold. Truth order is preserved; within each truth, the primary match
/// precedes additional matches, which follow global anchor order. This is only
/// the source's anchor-positive enumeration, not prediction-conditioned ignore
/// matching, loss/delta generation, or complete training assignment.
///
/// `truths` contains one image's labels; call separately for each image/head.
pub fn enumerate_obb_yolo_head_positives(
    truths: &[ObbLabel],
    classes: usize,
    anchors: &[f32],
    mask: &[usize],
    net_width: usize,
    net_height: usize,
    grid_width: usize,
    grid_height: usize,
    iou_thresh: f32,
) -> Result<Vec<ObbYoloPositiveAssignment>> {
    if classes == 0 || !iou_thresh.is_finite() {
        return Err(Error::InvalidConfig(
            "OBB positive assignment requires classes > 0 and a finite iou_thresh".to_owned(),
        ));
    }
    if mask.is_empty() {
        return Err(Error::InvalidConfig(
            "OBB positive assignment requires a nonempty head mask".to_owned(),
        ));
    }
    validate_anchor_assignment_inputs(
        anchors,
        mask,
        net_width,
        net_height,
        grid_width,
        grid_height,
    )?;
    let max_matches = truths
        .len()
        .checked_mul(mask.len())
        .ok_or_else(|| Error::InvalidConfig("OBB positive assignment size overflows".to_owned()))?;
    let mut assignments = Vec::new();
    assignments.try_reserve(max_matches).map_err(|_| {
        Error::InvalidConfig("OBB positive assignment allocation failed".to_owned())
    })?;

    for (truth_index, truth) in truths.iter().enumerate() {
        validate_truth(truth)?;
        if truth.class_id >= classes {
            return Err(Error::InvalidConfig(format!(
                "OBB class id {} exceeds classes={classes} in truth {truth_index}",
                truth.class_id
            )));
        }
        let (best_anchor, best_score) =
            best_obb_yolo_anchor(truth, anchors, net_width, net_height)?;
        let (cell_x, cell_y) = obb_yolo_truth_cell(truth, grid_width, grid_height);
        if let Some(head_anchor_index) = mask.iter().position(|&index| index == best_anchor) {
            assignments.push(ObbYoloPositiveAssignment {
                truth_index,
                class_id: truth.class_id,
                kind: ObbYoloPositiveKind::BestAnchor,
                assignment: ObbYoloAnchorAssignment {
                    anchor_index: best_anchor,
                    head_anchor_index,
                    cell_x,
                    cell_y,
                    match_score: best_score,
                },
            });
        }

        if iou_thresh < 1.0 {
            for (anchor_index, pair) in anchors.chunks_exact(2).enumerate() {
                if anchor_index == best_anchor {
                    continue;
                }
                let Some(head_anchor_index) = mask.iter().position(|&index| index == anchor_index)
                else {
                    continue;
                };
                let score =
                    obb_yolo_anchor_match_score(truth, pair[0], pair[1], net_width, net_height)?;
                if score > iou_thresh {
                    assignments.push(ObbYoloPositiveAssignment {
                        truth_index,
                        class_id: truth.class_id,
                        kind: ObbYoloPositiveKind::AdditionalAnchor,
                        assignment: ObbYoloAnchorAssignment {
                            anchor_index,
                            head_anchor_index,
                            cell_x,
                            cell_y,
                            match_score: score,
                        },
                    });
                }
            }
        }
    }
    Ok(assignments)
}

/// Enumerate primary/additional positives in source head-then-batch order.
///
/// Each per-head enumerator selects the global best anchor before applying that
/// head's mask. Results retain the image-local truth index and are tagged with
/// the batch and head indices. This ports assignment enumeration only; it does
/// not read prediction tensors or generate loss deltas.
pub fn enumerate_obb_yolo_batch_positives(
    batch_truths: &[Vec<ObbLabel>],
    classes: usize,
    anchors: &[f32],
    heads: &[ObbYoloHeadAssignmentSpec],
    net_width: usize,
    net_height: usize,
    iou_thresh: f32,
) -> Result<Vec<ObbYoloBatchPositiveAssignment>> {
    if batch_truths.is_empty() || heads.is_empty() {
        return Err(Error::InvalidConfig(
            "OBB batch positive assignment requires a nonempty batch and head list".to_owned(),
        ));
    }
    let mut assignments = Vec::new();
    for (head_index, head) in heads.iter().enumerate() {
        for (batch_index, truths) in batch_truths.iter().enumerate() {
            let positives = enumerate_obb_yolo_head_positives(
                truths,
                classes,
                anchors,
                &head.mask,
                net_width,
                net_height,
                head.grid_width,
                head.grid_height,
                iou_thresh,
            )?;
            assignments.try_reserve(positives.len()).map_err(|_| {
                Error::InvalidConfig("OBB batch positive allocation failed".to_owned())
            })?;
            assignments.extend(positives.into_iter().map(|positive| {
                ObbYoloBatchPositiveAssignment {
                    batch_index,
                    head_index,
                    positive,
                }
            }));
        }
    }
    Ok(assignments)
}

/// Select the source OBB YOLO best-truth, positive, and ignore gates for one cell.
///
/// `prediction_match_scores` contains one already-computed, source-clamped
/// score in `[0,1]` per truth. `class_outputs` contains this cell/anchor's
/// finite, source-sanitized class-channel outputs;
/// the source ignore gate is enabled when *any* output is strictly above 0.25,
/// including in `new_coords` mode where these channels may be logits. The best
/// truth is chosen independently of class confidence, and strict score
/// comparisons preserve the first truth on ties. This function selects gates
/// only; it does not calculate prediction scores or generate training deltas.
pub fn select_obb_yolo_cell_truth_match(
    prediction_match_scores: &[f32],
    class_outputs: &[f32],
    ignore_thresh: f32,
    truth_thresh: f32,
) -> Result<ObbYoloCellTruthMatch> {
    if class_outputs.is_empty()
        || !ignore_thresh.is_finite()
        || !truth_thresh.is_finite()
        || class_outputs.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidConfig(
            "OBB cell matching requires finite thresholds and class outputs".to_owned(),
        ));
    }
    if prediction_match_scores
        .iter()
        .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
    {
        return Err(Error::InvalidConfig(
            "OBB prediction match scores must be finite and within [0, 1]".to_owned(),
        ));
    }

    let mut best_truth_index = None;
    let mut best_score = 0.0;
    for (truth_index, &score) in prediction_match_scores.iter().enumerate() {
        if best_truth_index.is_none() || score > best_score {
            best_truth_index = Some(truth_index);
            best_score = score;
        }
    }
    let positive_truth_index = best_truth_index.filter(|_| best_score > truth_thresh);
    let class_confident = class_outputs
        .iter()
        .any(|&value| value > OBB_YOLO_CLASS_CONFIDENCE_THRESHOLD);
    let ignore_match_score = if class_confident { best_score } else { 0.0 };

    Ok(ObbYoloCellTruthMatch {
        best_truth_index,
        best_score,
        positive_truth_index,
        ignore_match_score,
        ignore_match: ignore_match_score > ignore_thresh,
    })
}

/// Calculate the objectness delta for the source OBB YOLO per-cell matching pass.
///
/// This consumes the gates from [`select_obb_yolo_cell_truth_match`], not raw
/// predictions. Ignore matching first applies the negative-objectness delta;
/// with objectness smoothing it keeps the larger ignore-target delta, otherwise
/// it zeros the delta. A positive best-truth match then overrides that result,
/// using `best_score²` as the smoothed target or `1` without smoothing. The
/// caller remains responsible for per-truth anchor assignments, class/box
/// deltas, losses, and trainer integration. Non-finite objectness and final
/// deltas are zeroed as in the C++ training forward.
pub fn obb_yolo_cell_objectness_delta(
    cell_match: ObbYoloCellTruthMatch,
    predicted_objectness: f32,
    obj_normalizer: f32,
    objectness_smooth: bool,
) -> Result<f32> {
    if !obj_normalizer.is_finite()
        || !cell_match.best_score.is_finite()
        || !(0.0..=1.0).contains(&cell_match.best_score)
        || !cell_match.ignore_match_score.is_finite()
        || !(0.0..=1.0).contains(&cell_match.ignore_match_score)
        || (cell_match.positive_truth_index.is_some() && cell_match.best_truth_index.is_none())
    {
        return Err(Error::InvalidConfig(
            "OBB objectness delta requires valid match scores and a finite normalizer".to_owned(),
        ));
    }

    let predicted_objectness = if predicted_objectness.is_finite() {
        predicted_objectness
    } else {
        0.0
    };
    let mut delta = obj_normalizer * (0.0 - predicted_objectness);
    if cell_match.ignore_match {
        if objectness_smooth {
            let ignore_delta =
                obj_normalizer * (cell_match.ignore_match_score - predicted_objectness);
            if ignore_delta > delta {
                delta = ignore_delta;
            }
        } else {
            delta = 0.0;
        }
    }

    if cell_match.positive_truth_index.is_some() {
        let target = if objectness_smooth {
            cell_match.best_score * cell_match.best_score
        } else {
            1.0
        };
        delta = obj_normalizer * (target - predicted_objectness);
    }

    Ok(if delta.is_finite() { delta } else { 0.0 })
}

/// Covariance-W2 raw-channel matching score without QBBV-vIoU blending.
///
/// This is the `viou_weight = 0` form of
/// [`obb_yolo_covariance_prediction_match_score_with_viou`].
pub fn obb_yolo_covariance_prediction_match_score(
    prediction: [f32; 6],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    covariance_normalizer: f32,
    front_length_ratio_k: f32,
    front_length_weight: f32,
) -> Result<f32> {
    obb_yolo_covariance_prediction_match_score_with_viou(
        prediction,
        truth,
        context,
        covariance_normalizer,
        front_length_ratio_k,
        front_length_weight,
        0.0,
    )
}

/// Decode one source YOLO output row and score it with covariance-W2 and the
/// optional source QBBV-vIoU blend.
///
/// `prediction` contains the six post-YOLO box/front channels: x/y already
/// include source activation and `scale_x_y`, while front channels remain raw.
/// `covariance_normalizer` must be finite and positive to select this branch;
/// it gates the branch but does not scale its returned score. Call this only
/// when the source iou-loss selector chooses the covariance branch. COBB score
/// computation and multi-branch aggregation are separate. Unlike the CIoU
/// helper, a direction-free truth retains its finite center placeholder because
/// the source covariance branch evaluates it rather than short-circuiting.
#[allow(clippy::too_many_arguments)]
pub fn obb_yolo_covariance_prediction_match_score_with_viou(
    prediction: [f32; 6],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    covariance_normalizer: f32,
    front_length_ratio_k: f32,
    front_length_weight: f32,
    viou_weight: f32,
) -> Result<f32> {
    validate_truth(truth)?;
    if !covariance_normalizer.is_finite()
        || covariance_normalizer <= 0.0
        || !front_length_ratio_k.is_finite()
        || !front_length_weight.is_finite()
    {
        return Err(Error::InvalidConfig(
            "OBB covariance matching requires a positive finite branch and finite score weights"
                .to_owned(),
        ));
    }
    let prediction = decode_obb_yolo_prediction(prediction, context)?;
    let truth = BoxObb {
        bbox: DetectionBox {
            x: truth.x,
            y: truth.y,
            w: truth.w,
            h: truth.h,
        },
        front_x: truth.front_x,
        front_y: truth.front_y,
    };
    let score = box_covariance_w2_obb_quality_score(
        prediction,
        truth,
        front_length_ratio_k,
        front_length_weight,
    );
    Ok(blend_obb_score_with_qbbv_viou(
        score,
        prediction,
        truth,
        viou_weight,
    ))
}

/// Decode one source YOLO output row and score it using the source legacy
/// angular fallback plus its optional QBBV-vIoU blend.
///
/// This is used when both covariance and COBB matching branches are disabled;
/// it is distinct from `ciou_obb` scoring and uses the direct front-channel
/// decode (not the optional center-front-anchor transform). A direction-free
/// truth has zero angular score, but a nonzero QBBV weight can still contribute
/// its source direction-placeholder vIoU.
pub fn obb_yolo_angular_prediction_match_score_with_viou(
    prediction: [f32; 6],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    viou_weight: f32,
) -> Result<f32> {
    validate_truth(truth)?;
    let prediction = decode_obb_yolo_prediction(prediction, context)?;
    let truth = BoxObb {
        bbox: DetectionBox {
            x: truth.x,
            y: truth.y,
            w: truth.w,
            h: truth.h,
        },
        front_x: truth.front_x,
        front_y: truth.front_y,
    };
    let score = box_ciou_angular_quality_score(prediction, truth);
    Ok(blend_obb_score_with_qbbv_viou(
        score,
        prediction,
        truth,
        viou_weight,
    ))
}

/// Decode one source YOLO output row and score it with the gated COBB branch.
///
/// The covariance branch takes priority; a positive finite covariance
/// normalizer, or an inactive COBB normalizer, is rejected. As in the source,
/// the COBB normalizer gates branch selection but does not scale this score.
/// The decoded geometry uses the direct front-channel transform; center-front
/// anchor adjustment and full batch/training integration are not handled here.
#[allow(clippy::too_many_arguments)]
pub fn obb_yolo_cobb_prediction_match_score_with_viou(
    prediction: [f32; 6],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    covariance_normalizer: f32,
    cobb_normalizer: f32,
    cobb_config: ObbCobbLossConfig,
    viou_weight: f32,
) -> Result<f32> {
    validate_truth(truth)?;
    if (covariance_normalizer.is_finite() && covariance_normalizer > 0.0)
        || !cobb_normalizer.is_finite()
        || cobb_normalizer <= 0.0
    {
        return Err(Error::InvalidConfig(
            "OBB matching requires an active COBB branch without covariance priority".to_owned(),
        ));
    }
    let prediction = decode_obb_yolo_prediction(prediction, context)?;
    let truth = BoxObb {
        bbox: DetectionBox {
            x: truth.x,
            y: truth.y,
            w: truth.w,
            h: truth.h,
        },
        front_x: truth.front_x,
        front_y: truth.front_y,
    };
    Ok(box_cobb_obb_quality_score_with_viou(
        prediction,
        truth,
        cobb_config,
        viou_weight,
    ))
}

/// Select and compute the source covariance/COBB/legacy-angular score for one
/// decoded OBB YOLO cell.
///
/// Covariance has priority when its normalizer is finite and positive. COBB is
/// next when its normalizer is finite and positive; otherwise the legacy angular
/// fallback is selected. This composes per-cell score branches only; it does
/// not perform multi-truth assignment, batch aggregation, or `ciou_obb` scoring.
pub fn obb_yolo_match_score(
    prediction: [f32; 6],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    config: ObbYoloMatchScoreConfig,
) -> Result<ObbYoloMatchScore> {
    let covariance_active =
        config.covariance_normalizer.is_finite() && config.covariance_normalizer > 0.0;
    if covariance_active {
        return Ok(ObbYoloMatchScore {
            branch: ObbYoloMatchScoreBranch::Covariance,
            score: obb_yolo_covariance_prediction_match_score_with_viou(
                prediction,
                truth,
                context,
                config.covariance_normalizer,
                config.covariance_front_length_ratio_k.max(0.05),
                if config.covariance_front_length_weight.is_finite() {
                    config.covariance_front_length_weight.max(0.0)
                } else {
                    0.0
                },
                config.viou_weight,
            )?,
        });
    }
    let cobb_active = config.cobb_normalizer.is_finite() && config.cobb_normalizer > 0.0;
    if cobb_active {
        return Ok(ObbYoloMatchScore {
            branch: ObbYoloMatchScoreBranch::Cobb,
            score: obb_yolo_cobb_prediction_match_score_with_viou(
                prediction,
                truth,
                context,
                config.covariance_normalizer,
                config.cobb_normalizer,
                config.cobb_loss,
                config.viou_weight,
            )?,
        });
    }
    Ok(ObbYoloMatchScore {
        branch: ObbYoloMatchScoreBranch::LegacyAngular,
        score: obb_yolo_angular_prediction_match_score_with_viou(
            prediction,
            truth,
            context,
            config.viou_weight,
        )?,
    })
}

fn obb_yolo_match_score_with_front_anchor_offset(
    prediction: [f32; 6],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    config: ObbYoloMatchScoreConfig,
    front_anchor_offset: Option<[f32; 2]>,
) -> Result<ObbYoloMatchScore> {
    let Some(front_anchor_offset) = front_anchor_offset else {
        return obb_yolo_match_score(prediction, truth, context, config);
    };
    let front = decode_obb_yolo_front_point(prediction, context, Some(front_anchor_offset))?;
    let mut adjusted_prediction = prediction;
    adjusted_prediction[4] = front[0] * context.grid_width as f32 - context.cell_x as f32;
    adjusted_prediction[5] = front[1] * context.grid_height as f32 - context.cell_y as f32;
    if !adjusted_prediction[4].is_finite() || !adjusted_prediction[5].is_finite() {
        return Err(Error::InvalidConfig(
            "OBB adjusted front channels are non-finite".to_owned(),
        ));
    }
    obb_yolo_match_score(adjusted_prediction, truth, context, config)
}

/// Decode one source YOLO output row and score it against a truth with the
/// source CIoU-OBB matching metric.
///
/// `prediction` is the six box/front channels from the OBB YOLO layer output:
/// x/y have already received the source `scale_x_y` transform and, unless
/// `new_coords`, the logistic activation; front channels remain raw linear
/// values. Non-finite channels are zeroed as in the source forward. This covers
/// the separate `ciou_obb` quality path, not the legacy angular fallback,
/// covariance/COBB weighted score paths, class confidence gate, or batch loop.
/// The decoded geometry and truth are normalized to network dimensions.
#[allow(clippy::too_many_arguments)]
pub fn obb_yolo_ciou_prediction_match_score(
    prediction: [f32; 6],
    truth: &ObbLabel,
    anchor_width: f32,
    anchor_height: f32,
    cell_x: usize,
    cell_y: usize,
    grid_width: usize,
    grid_height: usize,
    net_width: usize,
    net_height: usize,
    new_coords: bool,
) -> Result<f32> {
    validate_truth(truth)?;
    let prediction = decode_obb_yolo_prediction(
        prediction,
        ObbYoloPredictionMatchContext {
            anchor_width,
            anchor_height,
            cell_x,
            cell_y,
            grid_width,
            grid_height,
            net_width,
            net_height,
            new_coords,
        },
    )?;
    if !truth.has_front {
        return Ok(0.0);
    }
    let truth = BoxObb {
        bbox: DetectionBox {
            x: truth.x,
            y: truth.y,
            w: truth.w,
            h: truth.h,
        },
        front_x: truth.front_x,
        front_y: truth.front_y,
    };
    Ok(box_ciou_obb_quality_score(prediction, truth))
}

/// Compute source finite-difference box-channel deltas for one OBB positive.
///
/// This covers the source covariance analytic-gradient branch and the COBB or
/// legacy-angular finite-difference branch, including normalizer scaling,
/// sigmoid-chain factors, accumulation, and clipping. Front-channel gradients
/// are intentionally separate. Direct front-channel decoding is used, without
/// source `center_front_anchor` correction. Inputs use the same post-YOLO
/// channel representation as [`obb_yolo_match_score`].
pub fn obb_yolo_box_regression_deltas(
    prediction: [f32; 6],
    existing_deltas: [f32; 4],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    config: ObbYoloBoxDeltaConfig,
) -> Result<[f32; 4]> {
    obb_yolo_box_regression_deltas_with_front_anchor_offset(
        prediction,
        existing_deltas,
        truth,
        context,
        config,
        None,
    )
}

fn obb_yolo_box_regression_deltas_with_front_anchor_offset(
    prediction: [f32; 6],
    existing_deltas: [f32; 4],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    config: ObbYoloBoxDeltaConfig,
    front_anchor_offset: Option<[f32; 2]>,
) -> Result<[f32; 4]> {
    validate_truth(truth)?;
    let covariance_active =
        config.score.covariance_normalizer.is_finite() && config.score.covariance_normalizer > 0.0;
    if !config.iou_normalizer.is_finite()
        || !config.max_delta.is_finite()
        || config.max_delta < 0.0
        || existing_deltas.iter().any(|delta| !delta.is_finite())
    {
        return Err(Error::InvalidConfig(
            "OBB box deltas require finite normalizers, clipping, and existing deltas".to_owned(),
        ));
    }

    let mut deltas = if config.accumulate {
        existing_deltas
    } else {
        [0.0; 4]
    };
    let clip = if config.max_delta == f32::MAX {
        1.0
    } else {
        config.max_delta
    };
    if covariance_active {
        let decoded = decode_obb_yolo_prediction_with_front_anchor_offset(
            prediction,
            context,
            front_anchor_offset,
        )?;
        let cov_w = (decoded.bbox.w - 1e-6).max(1e-6);
        let cov_h = (decoded.bbox.h - 1e-6).max(1e-6);
        let inverse_softplus = |value: f32| {
            if value > 20.0 {
                value
            } else {
                value.exp_m1().ln()
            }
        };
        let cov = box_covariance_w2_obb_loss_and_gradient(
            decoded.bbox.x,
            decoded.bbox.y,
            inverse_softplus(cov_w),
            inverse_softplus(cov_h),
            decoded.front_x,
            decoded.front_y,
            BoxObb {
                bbox: DetectionBox {
                    x: truth.x,
                    y: truth.y,
                    w: truth.w,
                    h: truth.h,
                },
                front_x: truth.front_x,
                front_y: truth.front_y,
            },
            config.score.covariance_front_length_ratio_k.max(0.05),
            if config.score.covariance_front_length_weight.is_finite() {
                config.score.covariance_front_length_weight.max(0.0)
            } else {
                0.0
            },
        );
        let softplus_grad = |value: f32| (-(-value).exp_m1()).max(1e-6);
        let d_loss_dw = cov.gradient.d_raw_w / softplus_grad(cov_w);
        let d_loss_dh = cov.gradient.d_raw_h / softplus_grad(cov_h);
        let d_pred_w_d_tw = if context.new_coords {
            8.0 * prediction[2] * context.anchor_width / context.net_width as f32
        } else {
            decoded.bbox.w
        };
        let d_pred_h_d_th = if context.new_coords {
            8.0 * prediction[3] * context.anchor_height / context.net_height as f32
        } else {
            decoded.bbox.h
        };
        let d_loss_dtx = (cov.gradient.dx
            + if front_anchor_offset.is_some() {
                cov.gradient.dfx
            } else {
                0.0
            })
            / context.grid_width as f32;
        let d_loss_dty = (cov.gradient.dy
            + if front_anchor_offset.is_some() {
                cov.gradient.dfy
            } else {
                0.0
            })
            / context.grid_height as f32;
        let sigmoid_deriv_x = if context.new_coords {
            1.0
        } else {
            prediction[0] * (1.0 - prediction[0])
        };
        let sigmoid_deriv_y = if context.new_coords {
            1.0
        } else {
            prediction[1] * (1.0 - prediction[1])
        };
        let scale = config.iou_normalizer * config.score.covariance_normalizer.max(0.0);
        let mut updates = [
            -d_loss_dtx * sigmoid_deriv_x * scale,
            -d_loss_dty * sigmoid_deriv_y * scale,
            -d_loss_dw * d_pred_w_d_tw * scale,
            -d_loss_dh * d_pred_h_d_th * scale,
        ];
        for (index, update) in updates.iter_mut().enumerate() {
            if !update.is_finite() {
                *update = 0.0;
            }
            deltas[index] += update.clamp(-clip, clip);
        }
        return Ok(deltas);
    }

    let evaluate = |channels: [f32; 6]| -> Result<f32> {
        Ok(obb_yolo_match_score_with_front_anchor_offset(
            channels,
            truth,
            context,
            config.score,
            front_anchor_offset,
        )?
        .score)
    };
    let scale = config.iou_normalizer
        * if config.score.cobb_normalizer.is_finite() && config.score.cobb_normalizer > 0.0 {
            config.score.cobb_normalizer.max(0.0)
        } else {
            1.0
        };
    for channel in 0..4 {
        let base = prediction[channel];
        if !base.is_finite() {
            continue;
        }
        let epsilon = 1e-3 * (base.abs() + 1.0);
        let mut plus = prediction;
        let mut minus = prediction;
        plus[channel] = base + epsilon;
        minus[channel] = base - epsilon;
        let plus_score = evaluate(plus)?;
        let minus_score = evaluate(minus)?;
        let gradient = (plus_score - minus_score) / (2.0 * epsilon);
        // The source finite-difference branch applies sigmoid derivatives to
        // x/y unconditionally, including `new_coords` mode.
        let chain = if channel < 2 {
            base * (1.0 - base)
        } else {
            1.0
        };
        let mut delta = gradient * chain * scale;
        if !delta.is_finite() {
            delta = 0.0;
        }
        deltas[channel] += delta.clamp(-clip, clip);
    }
    Ok(deltas)
}

/// Compute source front-point deltas for one positive prediction.
///
/// The source front-point decoder is cell-relative even when directional
/// anchor offsets exist; those offsets only signal that the same gradient also
/// contributes to the box-center channels. Absent-front Rust labels do not
/// supervise this directional term.
pub fn obb_yolo_front_point_regression_deltas(
    prediction: [f32; 6],
    truth: &ObbLabel,
    context: ObbYoloPredictionMatchContext,
    config: ObbYoloFrontPointDeltaConfig,
    center_front_anchor: bool,
) -> Result<ObbYoloFrontPointDeltas> {
    validate_truth(truth)?;
    if !config.normalizer.is_finite() || !config.max_delta.is_finite() || config.max_delta < 0.0 {
        return Err(Error::InvalidConfig(
            "OBB front-point deltas require finite normalization and clipping".to_owned(),
        ));
    }
    if !truth.has_front || config.normalizer <= 0.0 {
        return Ok(ObbYoloFrontPointDeltas::default());
    }

    let decoded = decode_obb_yolo_prediction(prediction, context)?;
    let front_width = (truth.w * 0.25).max(1e-5);
    let front_height = (truth.h * 0.25).max(1e-5);
    let (ciou, gradient_x, gradient_y) = crate::train::obb_front_point_ciou_gradient(
        [decoded.front_x, decoded.front_y, front_width, front_height],
        [truth.front_x, truth.front_y, front_width, front_height],
        config.iou_loss,
    );
    let clip = if config.max_delta == f32::MAX {
        1.0
    } else {
        config.max_delta
    };
    let delta_x = finite_or_zero(gradient_x / context.grid_width as f32 * config.normalizer)
        .clamp(-clip, clip);
    let delta_y = finite_or_zero(gradient_y / context.grid_height as f32 * config.normalizer)
        .clamp(-clip, clip);
    Ok(ObbYoloFrontPointDeltas {
        front_x: delta_x,
        front_y: delta_y,
        center_x: if center_front_anchor { delta_x } else { 0.0 },
        center_y: if center_front_anchor { delta_y } else { 0.0 },
        loss: finite_or_zero((1.0 - ciou).max(0.0)),
    })
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn decode_obb_yolo_prediction(
    prediction: [f32; 6],
    context: ObbYoloPredictionMatchContext,
) -> Result<BoxObb> {
    let ObbYoloPredictionMatchContext {
        anchor_width,
        anchor_height,
        cell_x,
        cell_y,
        grid_width,
        grid_height,
        net_width,
        net_height,
        new_coords,
    } = context;
    if grid_width == 0
        || grid_height == 0
        || net_width == 0
        || net_height == 0
        || cell_x >= grid_width
        || cell_y >= grid_height
    {
        return Err(Error::InvalidConfig(
            "OBB prediction matching requires a valid cell, grid, and network shape".to_owned(),
        ));
    }
    if !anchor_width.is_finite()
        || !anchor_height.is_finite()
        || anchor_width <= 0.0
        || anchor_height <= 0.0
    {
        return Err(Error::InvalidConfig(
            "OBB prediction matching requires positive finite anchors".to_owned(),
        ));
    }

    let prediction = prediction.map(finite_or_zero);
    let x = finite_or_zero((cell_x as f32 + prediction[0]) / grid_width as f32);
    let y = finite_or_zero((cell_y as f32 + prediction[1]) / grid_height as f32);
    let (w, h) = if new_coords {
        (
            finite_or_zero(prediction[2] * prediction[2] * 4.0 * anchor_width / net_width as f32)
                .max(1e-6),
            finite_or_zero(prediction[3] * prediction[3] * 4.0 * anchor_height / net_height as f32)
                .max(1e-6),
        )
    } else {
        let tw = finite_or_zero(prediction[2]).clamp(-10.0, 10.0);
        let th = finite_or_zero(prediction[3]).clamp(-10.0, 10.0);
        (
            finite_or_zero(tw.exp() * anchor_width / net_width as f32).max(1e-6),
            finite_or_zero(th.exp() * anchor_height / net_height as f32).max(1e-6),
        )
    };
    Ok(BoxObb {
        bbox: DetectionBox { x, y, w, h },
        front_x: (cell_x as f32 + prediction[4]) / grid_width as f32,
        front_y: (cell_y as f32 + prediction[5]) / grid_height as f32,
    })
}

/// Decode the source front-point channels, optionally using directional anchor priors.
///
/// With `Some([u, v])`, this applies the source center-front-anchor formula:
/// `bbox_center + [u * anchor_width / net_width, v * anchor_height / net_height]`
/// plus the raw front channels divided by grid dimensions. `None` preserves the
/// legacy cell-relative decoding. Existing score/training paths do not implicitly
/// select offsets; callers must pass this decoded point into their chosen path.
pub fn decode_obb_yolo_front_point(
    prediction: [f32; 6],
    context: ObbYoloPredictionMatchContext,
    front_anchor_offset: Option<[f32; 2]>,
) -> Result<[f32; 2]> {
    if front_anchor_offset.is_some_and(|offset| !offset.into_iter().all(f32::is_finite)) {
        return Err(Error::InvalidConfig(
            "OBB front-anchor offsets must be finite".to_owned(),
        ));
    }
    let decoded = decode_obb_yolo_prediction(prediction, context)?;
    let front = if let Some([offset_u, offset_v]) = front_anchor_offset {
        [
            decoded.bbox.x
                + offset_u * context.anchor_width / context.net_width as f32
                + finite_or_zero(prediction[4]) / context.grid_width as f32,
            decoded.bbox.y
                + offset_v * context.anchor_height / context.net_height as f32
                + finite_or_zero(prediction[5]) / context.grid_height as f32,
        ]
    } else {
        [decoded.front_x, decoded.front_y]
    };
    if front.iter().all(|coordinate| coordinate.is_finite()) {
        Ok(front)
    } else {
        Err(Error::InvalidConfig(
            "OBB decoded front point is non-finite".to_owned(),
        ))
    }
}

fn decode_obb_yolo_prediction_with_front_anchor_offset(
    prediction: [f32; 6],
    context: ObbYoloPredictionMatchContext,
    front_anchor_offset: Option<[f32; 2]>,
) -> Result<BoxObb> {
    let mut decoded = decode_obb_yolo_prediction(prediction, context)?;
    if front_anchor_offset.is_some() {
        let [front_x, front_y] =
            decode_obb_yolo_front_point(prediction, context, front_anchor_offset)?;
        decoded.front_x = front_x;
        decoded.front_y = front_y;
    }
    Ok(decoded)
}

/// Calculate one positive cell's source OBB YOLO class-channel deltas.
///
/// This ports `delta_obb_yolo_class` over channel slices. If the selected
/// class already has a nonzero delta, the source updates only that class and
/// applies its multiplier without `cls_normalizer`; otherwise focal loss
/// ignores smoothing/multipliers, while the non-focal path applies smoothing
/// and applies `class_multiplier * cls_normalizer` only to the selected class
/// when multipliers are configured. Inputs are post-YOLO class outputs.
pub fn obb_yolo_class_deltas(
    class_outputs: &[f32],
    existing_deltas: &[f32],
    class_id: usize,
    focal_loss: bool,
    label_smooth_eps: f32,
    class_multipliers: Option<&[f32]>,
    cls_normalizer: f32,
) -> Result<ObbYoloClassDeltaResult> {
    let classes = class_outputs.len();
    if classes == 0
        || existing_deltas.len() != classes
        || class_id >= classes
        || !label_smooth_eps.is_finite()
        || !cls_normalizer.is_finite()
        || class_outputs.iter().any(|value| !value.is_finite())
        || existing_deltas.iter().any(|value| !value.is_finite())
        || class_multipliers.is_some_and(|multipliers| {
            multipliers.len() != classes || multipliers.iter().any(|value| !value.is_finite())
        })
    {
        return Err(Error::InvalidConfig(
            "OBB class deltas require matching finite channels and a valid class id".to_owned(),
        ));
    }

    let mut deltas = Vec::new();
    deltas
        .try_reserve_exact(classes)
        .map_err(|_| Error::InvalidConfig("OBB class-delta allocation failed".to_owned()))?;
    deltas.extend_from_slice(existing_deltas);

    if existing_deltas[class_id] != 0.0 {
        let target = 1.0 * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps;
        let result_delta = target - class_outputs[class_id];
        if result_delta.is_finite() {
            deltas[class_id] = result_delta;
        }
        if let Some(multipliers) = class_multipliers {
            deltas[class_id] *= multipliers[class_id];
        }
        return Ok(ObbYoloClassDeltaResult {
            deltas,
            class_output_for_average: class_outputs[class_id],
        });
    }

    if focal_loss {
        let pt = class_outputs[class_id] + 1.0e-15;
        let grad = -(1.0 - pt) * (2.0 * pt * pt.ln() + pt - 1.0);
        for (index, (&output, delta)) in class_outputs.iter().zip(&mut deltas).enumerate() {
            let target = if index == class_id { 1.0 } else { 0.0 };
            let result_delta = (target - output) * 0.5 * grad;
            if result_delta.is_finite() {
                *delta = result_delta;
            }
        }
    } else {
        for (index, (&output, delta)) in class_outputs.iter().zip(&mut deltas).enumerate() {
            let target = if index == class_id { 1.0 } else { 0.0 };
            let target = target * (1.0 - label_smooth_eps) + 0.5 * label_smooth_eps;
            let result_delta = target - output;
            if result_delta.is_finite() {
                *delta = result_delta;
            }
            if index == class_id {
                if let Some(multipliers) = class_multipliers {
                    *delta *= multipliers[class_id] * cls_normalizer;
                }
            }
        }
    }

    Ok(ObbYoloClassDeltaResult {
        deltas,
        class_output_for_average: class_outputs[class_id],
    })
}

/// Compute one cell's source CIoU scores against truths, then apply the source
/// best-truth, class-gated ignore, and positive-threshold selection.
///
/// `prediction` is the six post-YOLO box/front channels for this cell/anchor;
/// `class_outputs` contains its class channels. Truth rows are visited in order
/// and matching stops at the source `truth.x == 0` empty-slot sentinel. Scores
/// are computed independently of class confidence; the selector's strict
/// thresholds and first-on-tie rule then match the source cell loop. This is a
/// single-cell CIoU/default-score path, not the batch trainer or weighted
/// covariance/COBB alternatives.
pub fn match_obb_yolo_cell_prediction(
    prediction: [f32; 6],
    truths: &[ObbLabel],
    class_outputs: &[f32],
    context: ObbYoloPredictionMatchContext,
    ignore_thresh: f32,
    truth_thresh: f32,
) -> Result<ObbYoloCellTruthMatch> {
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(truths.len())
        .map_err(|_| Error::InvalidConfig("OBB cell truth-score allocation failed".to_owned()))?;
    for truth in truths {
        if truth.x == 0.0 {
            break;
        }
        scores.push(obb_yolo_ciou_prediction_match_score(
            prediction,
            truth,
            context.anchor_width,
            context.anchor_height,
            context.cell_x,
            context.cell_y,
            context.grid_width,
            context.grid_height,
            context.net_width,
            context.net_height,
            context.new_coords,
        )?);
    }
    select_obb_yolo_cell_truth_match(&scores, class_outputs, ignore_thresh, truth_thresh)
}

/// Compute source-priority OBB scores against truth rows, then select one
/// cell's best-truth, ignore, and positive gates.
///
/// The covariance/COBB/angular branch dispatcher is run independently for each
/// truth. Truth order, strict thresholds, class gating, and the `truth.x == 0`
/// empty-slot sentinel match the source cell loop. Batch/trainer integration and
/// `ciou_obb` scoring remain separate.
pub fn match_obb_yolo_cell_prediction_with_score_config(
    prediction: [f32; 6],
    truths: &[ObbLabel],
    class_outputs: &[f32],
    context: ObbYoloPredictionMatchContext,
    score_config: ObbYoloMatchScoreConfig,
    ignore_thresh: f32,
    truth_thresh: f32,
) -> Result<ObbYoloCellTruthMatch> {
    match_obb_yolo_cell_prediction_with_front_anchor_offset(
        prediction,
        truths,
        class_outputs,
        context,
        score_config,
        ignore_thresh,
        truth_thresh,
        None,
    )
}

fn match_obb_yolo_cell_prediction_with_front_anchor_offset(
    prediction: [f32; 6],
    truths: &[ObbLabel],
    class_outputs: &[f32],
    context: ObbYoloPredictionMatchContext,
    score_config: ObbYoloMatchScoreConfig,
    ignore_thresh: f32,
    truth_thresh: f32,
    front_anchor_offset: Option<[f32; 2]>,
) -> Result<ObbYoloCellTruthMatch> {
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(truths.len())
        .map_err(|_| Error::InvalidConfig("OBB cell truth-score allocation failed".to_owned()))?;
    for truth in truths {
        if truth.x == 0.0 {
            break;
        }
        scores.push(
            obb_yolo_match_score_with_front_anchor_offset(
                prediction,
                truth,
                context,
                score_config,
                front_anchor_offset,
            )?
            .score,
        );
    }
    select_obb_yolo_cell_truth_match(&scores, class_outputs, ignore_thresh, truth_thresh)
}

/// Match caller-selected cells against the truth labels for their batch slots.
///
/// Input order is preserved, and every cell uses its own head score settings
/// and prediction context. This composes batch lookup with the source one-cell
/// matcher; tensor unpacking, positive-anchor loss deltas, and the trainer loop
/// remain the caller's responsibility.
pub fn match_obb_yolo_batch_cell_predictions(
    batch_truths: &[Vec<ObbLabel>],
    cell_predictions: &[ObbYoloBatchCellPrediction],
) -> Result<Vec<ObbYoloBatchCellMatch>> {
    let mut matches = Vec::new();
    matches
        .try_reserve_exact(cell_predictions.len())
        .map_err(|_| Error::InvalidConfig("OBB batch match allocation failed".to_owned()))?;
    for cell in cell_predictions {
        let truths = batch_truths.get(cell.batch_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "OBB cell batch index {} exceeds batch size {}",
                cell.batch_index,
                batch_truths.len()
            ))
        })?;
        let truth_match = match_obb_yolo_cell_prediction_with_score_config(
            cell.prediction,
            truths,
            &cell.class_outputs,
            cell.context,
            cell.score_config,
            cell.ignore_thresh,
            cell.truth_thresh,
        )?;
        let objectness_delta = obb_yolo_cell_objectness_delta(
            truth_match,
            cell.predicted_objectness,
            cell.objectness_normalizer,
            cell.objectness_smooth,
        )?;
        matches.push(ObbYoloBatchCellMatch {
            batch_index: cell.batch_index,
            head_index: cell.head_index,
            anchor_index: cell.anchor_index,
            head_anchor_index: cell.head_anchor_index,
            cell_x: cell.context.cell_x,
            cell_y: cell.context.cell_y,
            truth_match,
            objectness_delta,
        });
    }
    Ok(matches)
}

/// Visit all batch/cell/anchor matches in one source OBB YOLO head output.
///
/// `head_output` uses source layer layout `[batch][entry][grid-cell]`, where
/// each masked anchor has seven box/objectness entries followed by class
/// entries. Its first two coordinates, objectness, and (unless `new_coords`)
/// classes are the forward-activated/scaled values consumed by the source loss.
/// The callback keeps the traversal memory-bounded; callback errors stop the
/// visit and are returned unchanged.
pub fn visit_obb_yolo_batch_head_matches(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    config: &ObbYoloHeadMatchConfig,
    mut visit: impl FnMut(ObbYoloBatchCellMatch) -> Result<()>,
) -> Result<()> {
    if batch_truths.is_empty()
        || config.classes == 0
        || !config.ignore_thresh.is_finite()
        || !config.truth_thresh.is_finite()
    {
        return Err(Error::InvalidConfig(
            "OBB head matching requires truths, classes, and finite thresholds".to_owned(),
        ));
    }
    validate_anchor_assignment_inputs(
        &config.anchors,
        &config.mask,
        config.net_width,
        config.net_height,
        config.grid_width,
        config.grid_height,
    )?;
    validate_anchor_front_offsets(
        config.front_anchor_offsets.as_deref(),
        config.anchors.len() / 2,
    )?;
    let stride = config
        .grid_width
        .checked_mul(config.grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB head grid size overflows".to_owned()))?;
    let entries = config
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB head entry count overflows".to_owned()))?;
    let per_batch = config
        .mask
        .len()
        .checked_mul(entries)
        .and_then(|count| count.checked_mul(stride))
        .ok_or_else(|| Error::InvalidConfig("OBB head tensor size overflows".to_owned()))?;
    let expected = batch_truths
        .len()
        .checked_mul(per_batch)
        .ok_or_else(|| Error::InvalidConfig("OBB batch tensor size overflows".to_owned()))?;
    if head_output.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB batch head tensor length mismatch: expected {expected}, got {}",
            head_output.len()
        )));
    }

    let mut class_outputs = Vec::new();
    class_outputs
        .try_reserve_exact(config.classes)
        .map_err(|_| Error::InvalidConfig("OBB class-output allocation failed".to_owned()))?;
    for (batch_index, truths) in batch_truths.iter().enumerate() {
        let batch_base = batch_index * per_batch;
        for cell_y in 0..config.grid_height {
            for cell_x in 0..config.grid_width {
                let cell_index = cell_y * config.grid_width + cell_x;
                for (head_anchor_index, &anchor_index) in config.mask.iter().enumerate() {
                    let anchor_base = batch_base + head_anchor_index * entries * stride;
                    let channel_value = |entry: usize| {
                        finite_or_zero(head_output[anchor_base + entry * stride + cell_index])
                    };
                    let prediction = std::array::from_fn(|channel| channel_value(channel));
                    class_outputs.clear();
                    class_outputs
                        .extend((0..config.classes).map(|class_id| channel_value(7 + class_id)));
                    let context = ObbYoloPredictionMatchContext {
                        anchor_width: config.anchors[anchor_index * 2],
                        anchor_height: config.anchors[anchor_index * 2 + 1],
                        cell_x,
                        cell_y,
                        grid_width: config.grid_width,
                        grid_height: config.grid_height,
                        net_width: config.net_width,
                        net_height: config.net_height,
                        new_coords: config.new_coords,
                    };
                    let truth_match = match_obb_yolo_cell_prediction_with_front_anchor_offset(
                        prediction,
                        truths,
                        &class_outputs,
                        context,
                        config.score_config,
                        config.ignore_thresh,
                        config.truth_thresh,
                        config
                            .front_anchor_offsets
                            .as_ref()
                            .map(|offsets| offsets[anchor_index]),
                    )?;
                    let objectness_delta = obb_yolo_cell_objectness_delta(
                        truth_match,
                        channel_value(6),
                        config.objectness_normalizer,
                        config.objectness_smooth,
                    )?;
                    visit(ObbYoloBatchCellMatch {
                        batch_index,
                        head_index: config.head_index,
                        anchor_index,
                        head_anchor_index,
                        cell_x,
                        cell_y,
                        truth_match,
                        objectness_delta,
                    })?;
                }
            }
        }
    }
    Ok(())
}

/// Write source OBB objectness deltas into one head's `[batch][entry][cell]` delta tensor.
///
/// Only entry 6 for each cell/anchor is touched; coordinate, front, and class
/// deltas are preserved for subsequent training passes. Like the source layer,
/// objectness deltas are assigned rather than accumulated.
pub fn apply_obb_yolo_batch_head_objectness_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    head_delta: &mut [f32],
    config: &ObbYoloHeadMatchConfig,
) -> Result<()> {
    if head_delta.len() != head_output.len() {
        return Err(Error::InvalidConfig(format!(
            "OBB head delta length mismatch: expected {}, got {}",
            head_output.len(),
            head_delta.len()
        )));
    }
    let stride = config
        .grid_width
        .checked_mul(config.grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB head grid size overflows".to_owned()))?;
    let entries = config
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB head entry count overflows".to_owned()))?;
    let per_batch = config
        .mask
        .len()
        .checked_mul(entries)
        .and_then(|count| count.checked_mul(stride))
        .ok_or_else(|| Error::InvalidConfig("OBB head tensor size overflows".to_owned()))?;
    let expected = batch_truths
        .len()
        .checked_mul(per_batch)
        .ok_or_else(|| Error::InvalidConfig("OBB batch tensor size overflows".to_owned()))?;
    if head_delta.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB head delta length mismatch: expected {expected}, got {}",
            head_delta.len()
        )));
    }

    visit_obb_yolo_batch_head_matches(batch_truths, head_output, config, |matched| {
        let cell_index = matched.cell_y * config.grid_width + matched.cell_x;
        let channel = matched.head_anchor_index * entries + 6;
        let index = matched.batch_index * per_batch + channel * stride + cell_index;
        head_delta[index] = matched.objectness_delta;
        Ok(())
    })
}

/// Apply the source `truth_thresh` positive class and box deltas for every cell.
///
/// The complete-head objectness pass must run first; this function leaves entry
/// 6 untouched, matching the source per-cell branch. It applies class smoothing
/// and the smoothed `best_score²` class target override, then dispatches the
/// source covariance/COBB/angular box delta with the mapped class multiplier.
/// These cells do not contribute to `avg_cat` in the source; the return value is
/// only their positive-cell count. Best/additional anchor passes remain separate.
pub fn apply_obb_yolo_batch_head_truth_threshold_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    head_delta: &mut [f32],
    match_config: &ObbYoloHeadMatchConfig,
    positive_config: &ObbYoloPositiveObjectnessConfig,
    focal_loss: bool,
    label_smooth_eps: f32,
    cls_normalizer: f32,
    box_delta_config: ObbYoloBoxDeltaConfig,
) -> Result<usize> {
    if match_config.anchors != positive_config.anchors
        || match_config.mask != positive_config.head.mask
        || match_config.grid_width != positive_config.head.grid_width
        || match_config.grid_height != positive_config.head.grid_height
        || match_config.classes != positive_config.classes
        || match_config.net_width != positive_config.net_width
        || match_config.net_height != positive_config.net_height
        || match_config.objectness_normalizer != positive_config.objectness_normalizer
        || match_config.objectness_smooth != positive_config.objectness_smooth
        || match_config.front_anchor_offsets != positive_config.front_anchor_offsets
        || match_config.score_config != box_delta_config.score
        || !label_smooth_eps.is_finite()
        || !cls_normalizer.is_finite()
        || !box_delta_config.iou_normalizer.is_finite()
        || !box_delta_config.max_delta.is_finite()
        || box_delta_config.max_delta < 0.0
        || head_delta.iter().any(|delta| !delta.is_finite())
        || positive_config.class_map.as_ref().is_some_and(|mapping| {
            mapping.len() != positive_config.classes
                || mapping
                    .iter()
                    .any(|&class_id| class_id >= positive_config.classes)
        })
        || positive_config
            .class_multipliers
            .as_ref()
            .is_some_and(|multipliers| {
                multipliers.len() != positive_config.classes
                    || multipliers.iter().any(|multiplier| {
                        !multiplier.is_finite()
                            || !(box_delta_config.iou_normalizer * multiplier).is_finite()
                    })
            })
    {
        return Err(Error::InvalidConfig(
            "OBB truth-threshold training configs or tensor values are invalid".to_owned(),
        ));
    }
    let stride = match_config
        .grid_width
        .checked_mul(match_config.grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB head grid size overflows".to_owned()))?;
    let entries = match_config
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB head entry count overflows".to_owned()))?;
    let per_batch = match_config
        .mask
        .len()
        .checked_mul(entries)
        .and_then(|count| count.checked_mul(stride))
        .ok_or_else(|| Error::InvalidConfig("OBB head tensor size overflows".to_owned()))?;
    let expected = batch_truths
        .len()
        .checked_mul(per_batch)
        .ok_or_else(|| Error::InvalidConfig("OBB batch tensor size overflows".to_owned()))?;
    if head_output.len() != expected || head_delta.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB truth-threshold tensor length mismatch: expected {expected}, output {}, delta {}",
            head_output.len(),
            head_delta.len()
        )));
    }
    let mut class_outputs = Vec::new();
    class_outputs
        .try_reserve_exact(match_config.classes)
        .map_err(|_| Error::InvalidConfig("OBB class-output allocation failed".to_owned()))?;
    let mut existing_deltas = Vec::new();
    existing_deltas
        .try_reserve_exact(match_config.classes)
        .map_err(|_| Error::InvalidConfig("OBB class-delta allocation failed".to_owned()))?;
    let mut positive_count = 0;

    visit_obb_yolo_batch_head_matches(batch_truths, head_output, match_config, |matched| {
        let Some(truth_index) = matched.truth_match.positive_truth_index else {
            return Ok(());
        };
        let truth = &batch_truths[matched.batch_index][truth_index];
        let class_id = positive_config
            .class_map
            .as_ref()
            .map_or(truth.class_id, |mapping| mapping[truth.class_id]);
        let class_multiplier = positive_config
            .class_multipliers
            .as_ref()
            .map_or(1.0, |multipliers| multipliers[class_id]);
        let cell = matched.cell_y * match_config.grid_width + matched.cell_x;
        let base =
            matched.batch_index * per_batch + matched.head_anchor_index * entries * stride + cell;
        class_outputs.clear();
        existing_deltas.clear();
        for class_index in 0..match_config.classes {
            let index = base + (7 + class_index) * stride;
            class_outputs.push(finite_or_zero(head_output[index]));
            existing_deltas.push(head_delta[index]);
        }
        let mut class_result = obb_yolo_class_deltas(
            &class_outputs,
            &existing_deltas,
            class_id,
            focal_loss,
            label_smooth_eps,
            positive_config.class_multipliers.as_deref(),
            cls_normalizer,
        )?;
        let iou_multiplier = matched.truth_match.best_score * matched.truth_match.best_score;
        if match_config.objectness_smooth {
            class_result.deltas[class_id] =
                class_multiplier * (iou_multiplier - class_outputs[class_id]);
        }
        for (class_index, &delta) in class_result.deltas.iter().enumerate() {
            let index = base + (7 + class_index) * stride;
            head_delta[index] = delta;
        }

        let prediction =
            std::array::from_fn(|channel| finite_or_zero(head_output[base + channel * stride]));
        let context = ObbYoloPredictionMatchContext {
            anchor_width: match_config.anchors[matched.anchor_index * 2],
            anchor_height: match_config.anchors[matched.anchor_index * 2 + 1],
            cell_x: matched.cell_x,
            cell_y: matched.cell_y,
            grid_width: match_config.grid_width,
            grid_height: match_config.grid_height,
            net_width: match_config.net_width,
            net_height: match_config.net_height,
            new_coords: match_config.new_coords,
        };
        let mut truth_box_delta_config = box_delta_config;
        truth_box_delta_config.iou_normalizer *= class_multiplier;
        let box_deltas = obb_yolo_box_regression_deltas_with_front_anchor_offset(
            prediction,
            std::array::from_fn(|channel| head_delta[base + channel * stride]),
            truth,
            context,
            truth_box_delta_config,
            match_config
                .front_anchor_offsets
                .as_ref()
                .map(|offsets| offsets[matched.anchor_index]),
        )?;
        for (channel, delta) in box_deltas.into_iter().enumerate() {
            head_delta[base + channel * stride] = delta;
        }
        positive_count += 1;
        Ok(())
    })?;
    Ok(positive_count)
}

/// Apply best-anchor and threshold-selected OBB positive objectness overrides.
///
/// Positives follow source global-anchor selection and mask membership. With
/// objectness smoothing, an existing nonzero delta is preserved; otherwise the
/// positive target overwrites the earlier negative/ignore objectness delta.
/// Only objectness entries are changed; class and box training remain separate.
pub fn apply_obb_yolo_batch_head_positive_objectness_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    head_delta: &mut [f32],
    config: &ObbYoloPositiveObjectnessConfig,
) -> Result<()> {
    if config.classes == 0
        || !config.objectness_normalizer.is_finite()
        || config.class_map.as_ref().is_some_and(|mapping| {
            mapping.len() != config.classes
                || mapping.iter().any(|&class_id| class_id >= config.classes)
        })
        || config
            .class_multipliers
            .as_ref()
            .is_some_and(|multipliers| {
                multipliers.len() != config.classes
                    || multipliers.iter().any(|value| !value.is_finite())
            })
    {
        return Err(Error::InvalidConfig(
            "OBB positive objectness assignment has invalid class or normalizer settings"
                .to_owned(),
        ));
    }
    let stride = config
        .head
        .grid_width
        .checked_mul(config.head.grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB head grid size overflows".to_owned()))?;
    let entries = config
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB head entry count overflows".to_owned()))?;
    let per_batch = config
        .head
        .mask
        .len()
        .checked_mul(entries)
        .and_then(|count| count.checked_mul(stride))
        .ok_or_else(|| Error::InvalidConfig("OBB head tensor size overflows".to_owned()))?;
    let expected = batch_truths
        .len()
        .checked_mul(per_batch)
        .ok_or_else(|| Error::InvalidConfig("OBB batch tensor size overflows".to_owned()))?;
    if head_output.len() != expected || head_delta.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB positive-assignment tensor length mismatch: expected {expected}, output {}, delta {}",
            head_output.len(),
            head_delta.len()
        )));
    }

    let head_specs = [config.head.clone()];
    let positives = enumerate_obb_yolo_batch_positives(
        batch_truths,
        config.classes,
        &config.anchors,
        &head_specs,
        config.net_width,
        config.net_height,
        config.iou_thresh,
    )?;
    for tagged in positives {
        let positive = tagged.positive;
        let truth = &batch_truths[tagged.batch_index][positive.truth_index];
        let class_id = config
            .class_map
            .as_ref()
            .map_or(truth.class_id, |mapping| mapping[truth.class_id]);
        let class_multiplier = config
            .class_multipliers
            .as_ref()
            .map_or(1.0, |multipliers| multipliers[class_id]);
        let cell_index =
            positive.assignment.cell_y * config.head.grid_width + positive.assignment.cell_x;
        let channel = positive.assignment.head_anchor_index * entries + 6;
        let index = tagged.batch_index * per_batch + channel * stride + cell_index;
        let predicted_objectness = finite_or_zero(head_output[index]);
        let positive_delta =
            class_multiplier * config.objectness_normalizer * (1.0 - predicted_objectness);
        if !positive_delta.is_finite() {
            return Err(Error::InvalidConfig(
                "OBB positive objectness delta is non-finite".to_owned(),
            ));
        }
        if !config.objectness_smooth || head_delta[index] == 0.0 {
            head_delta[index] = positive_delta;
        }
    }
    Ok(())
}

/// Apply source OBB class-channel deltas for each best/additional positive anchor.
///
/// The head output and delta tensors use `[batch][entry][grid-cell]` storage.
/// Class mapping and multipliers are applied in source order; incoming target
/// class deltas retain the helper's special positive-delta branch semantics.
/// Returns `avg_cat` sum/count for these anchor assignments; objectness,
/// geometry deltas, truth-threshold positives, and broader metrics remain separate.
#[allow(clippy::too_many_arguments)]
pub fn apply_obb_yolo_batch_head_positive_class_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    head_delta: &mut [f32],
    config: &ObbYoloPositiveObjectnessConfig,
    focal_loss: bool,
    label_smooth_eps: f32,
    cls_normalizer: f32,
) -> Result<ObbYoloClassDeltaStats> {
    if config.classes == 0
        || !label_smooth_eps.is_finite()
        || !cls_normalizer.is_finite()
        || head_delta.iter().any(|delta| !delta.is_finite())
        || config.class_map.as_ref().is_some_and(|mapping| {
            mapping.len() != config.classes
                || mapping.iter().any(|&class_id| class_id >= config.classes)
        })
        || config
            .class_multipliers
            .as_ref()
            .is_some_and(|multipliers| {
                multipliers.len() != config.classes
                    || multipliers.iter().any(|value| !value.is_finite())
            })
    {
        return Err(Error::InvalidConfig(
            "OBB positive class deltas have invalid class or normalizer settings".to_owned(),
        ));
    }
    let stride = config
        .head
        .grid_width
        .checked_mul(config.head.grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB head grid size overflows".to_owned()))?;
    let entries = config
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB head entry count overflows".to_owned()))?;
    let per_batch = config
        .head
        .mask
        .len()
        .checked_mul(entries)
        .and_then(|count| count.checked_mul(stride))
        .ok_or_else(|| Error::InvalidConfig("OBB head tensor size overflows".to_owned()))?;
    let expected = batch_truths
        .len()
        .checked_mul(per_batch)
        .ok_or_else(|| Error::InvalidConfig("OBB batch tensor size overflows".to_owned()))?;
    if head_output.len() != expected || head_delta.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB positive-class tensor length mismatch: expected {expected}, output {}, delta {}",
            head_output.len(),
            head_delta.len()
        )));
    }

    let head_specs = [config.head.clone()];
    let positives = enumerate_obb_yolo_batch_positives(
        batch_truths,
        config.classes,
        &config.anchors,
        &head_specs,
        config.net_width,
        config.net_height,
        config.iou_thresh,
    )?;
    let mut class_outputs = Vec::new();
    class_outputs
        .try_reserve_exact(config.classes)
        .map_err(|_| Error::InvalidConfig("OBB class-output allocation failed".to_owned()))?;
    let mut existing_deltas = Vec::new();
    existing_deltas
        .try_reserve_exact(config.classes)
        .map_err(|_| Error::InvalidConfig("OBB class-delta allocation failed".to_owned()))?;

    let mut stats = ObbYoloClassDeltaStats::default();
    for tagged in positives {
        let truth = &batch_truths[tagged.batch_index][tagged.positive.truth_index];
        let class_id = config
            .class_map
            .as_ref()
            .map_or(truth.class_id, |mapping| mapping[truth.class_id]);
        let entry_base = tagged.batch_index * per_batch
            + tagged.positive.assignment.head_anchor_index * entries * stride;
        let cell_index = tagged.positive.assignment.cell_y * config.head.grid_width
            + tagged.positive.assignment.cell_x;
        class_outputs.clear();
        existing_deltas.clear();
        for class_index in 0..config.classes {
            let index = entry_base + (7 + class_index) * stride + cell_index;
            class_outputs.push(finite_or_zero(head_output[index]));
            existing_deltas.push(head_delta[index]);
        }
        let result = obb_yolo_class_deltas(
            &class_outputs,
            &existing_deltas,
            class_id,
            focal_loss,
            label_smooth_eps,
            config.class_multipliers.as_deref(),
            cls_normalizer,
        )?;
        stats.class_output_sum += result.class_output_for_average;
        stats.positive_count += 1;
        for (class_index, &delta) in result.deltas.iter().enumerate() {
            let index = entry_base + (7 + class_index) * stride + cell_index;
            head_delta[index] = delta;
        }
    }
    Ok(stats)
}

/// Apply source finite-difference box regression deltas to one batch/head tensor.
///
/// Best-anchor and threshold-selected additional positives are processed in
/// truth order. Only coordinate entries 0..4 are changed; front/objectness/class
/// deltas are preserved. This routes covariance analytic gradients or COBB/
/// legacy angular finite differences, but does not implement the source
/// `center_front_anchor` decode correction.
pub fn apply_obb_yolo_batch_head_positive_box_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    head_delta: &mut [f32],
    assignment: &ObbYoloPositiveObjectnessConfig,
    new_coords: bool,
    delta_config: ObbYoloBoxDeltaConfig,
) -> Result<usize> {
    if !delta_config.iou_normalizer.is_finite()
        || !delta_config.max_delta.is_finite()
        || delta_config.max_delta < 0.0
        || head_delta.iter().any(|value| !value.is_finite())
        || assignment.class_map.as_ref().is_some_and(|mapping| {
            mapping.len() != assignment.classes
                || mapping
                    .iter()
                    .any(|&class_id| class_id >= assignment.classes)
        })
        || assignment
            .class_multipliers
            .as_ref()
            .is_some_and(|multipliers| {
                multipliers.len() != assignment.classes
                    || multipliers.iter().any(|value| {
                        !value.is_finite() || !(delta_config.iou_normalizer * value).is_finite()
                    })
            })
    {
        return Err(Error::InvalidConfig(
            "OBB batch box deltas require finite normalizers, clipping, and existing deltas"
                .to_owned(),
        ));
    }
    validate_anchor_front_offsets(
        assignment.front_anchor_offsets.as_deref(),
        assignment.anchors.len() / 2,
    )?;
    let stride = assignment
        .head
        .grid_width
        .checked_mul(assignment.head.grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB head grid size overflows".to_owned()))?;
    let entries = assignment
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB head entry count overflows".to_owned()))?;
    let per_batch = assignment
        .head
        .mask
        .len()
        .checked_mul(entries)
        .and_then(|count| count.checked_mul(stride))
        .ok_or_else(|| Error::InvalidConfig("OBB head tensor size overflows".to_owned()))?;
    let expected = batch_truths
        .len()
        .checked_mul(per_batch)
        .ok_or_else(|| Error::InvalidConfig("OBB batch tensor size overflows".to_owned()))?;
    if head_output.len() != expected || head_delta.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB positive-box tensor length mismatch: expected {expected}, output {}, delta {}",
            head_output.len(),
            head_delta.len()
        )));
    }
    let head_specs = [assignment.head.clone()];
    let positives = enumerate_obb_yolo_batch_positives(
        batch_truths,
        assignment.classes,
        &assignment.anchors,
        &head_specs,
        assignment.net_width,
        assignment.net_height,
        assignment.iou_thresh,
    )?;
    for tagged in &positives {
        let anchor_index = tagged.positive.assignment.anchor_index;
        let anchor_offset = anchor_index
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidConfig("OBB anchor index overflows".to_owned()))?;
        if anchor_offset + 1 >= assignment.anchors.len() {
            return Err(Error::InvalidConfig(
                "OBB positive assignment references an invalid anchor".to_owned(),
            ));
        }
        let positive = tagged.positive.assignment;
        let cell = positive.cell_y * assignment.head.grid_width + positive.cell_x;
        let base =
            tagged.batch_index * per_batch + positive.head_anchor_index * entries * stride + cell;
        let prediction =
            std::array::from_fn(|channel| finite_or_zero(head_output[base + channel * stride]));
        let existing = std::array::from_fn(|channel| head_delta[base + channel * stride]);
        let context = ObbYoloPredictionMatchContext {
            anchor_width: assignment.anchors[anchor_offset],
            anchor_height: assignment.anchors[anchor_offset + 1],
            cell_x: positive.cell_x,
            cell_y: positive.cell_y,
            grid_width: assignment.head.grid_width,
            grid_height: assignment.head.grid_height,
            net_width: assignment.net_width,
            net_height: assignment.net_height,
            new_coords,
        };
        let truth = &batch_truths[tagged.batch_index][tagged.positive.truth_index];
        let class_id = assignment
            .class_map
            .as_ref()
            .map_or(truth.class_id, |mapping| mapping[truth.class_id]);
        let class_multiplier = assignment
            .class_multipliers
            .as_ref()
            .map_or(1.0, |multipliers| multipliers[class_id]);
        let mut truth_delta_config = delta_config;
        truth_delta_config.iou_normalizer *= class_multiplier;
        let deltas = obb_yolo_box_regression_deltas_with_front_anchor_offset(
            prediction,
            existing,
            truth,
            context,
            truth_delta_config,
            assignment
                .front_anchor_offsets
                .as_ref()
                .map(|offsets| offsets[anchor_index]),
        )?;
        for (channel, delta) in deltas.into_iter().enumerate() {
            head_delta[base + channel * stride] = delta;
        }
    }
    Ok(positives.len())
}

/// Add standalone front-point deltas to best/additional positive anchor slots.
///
/// This writes front channels and, when directional offsets are configured,
/// their source center-chain contribution. It does not cover truth-threshold
/// cells or compose the complete head training pass.
pub fn apply_obb_yolo_batch_head_positive_front_point_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    head_delta: &mut [f32],
    assignment: &ObbYoloPositiveObjectnessConfig,
    new_coords: bool,
    delta_config: ObbYoloFrontPointDeltaConfig,
) -> Result<usize> {
    if assignment.classes == 0
        || !delta_config.normalizer.is_finite()
        || !delta_config.max_delta.is_finite()
        || delta_config.max_delta < 0.0
        || head_delta.iter().any(|value| !value.is_finite())
        || assignment.class_map.as_ref().is_some_and(|mapping| {
            mapping.len() != assignment.classes
                || mapping
                    .iter()
                    .any(|&class_id| class_id >= assignment.classes)
        })
        || assignment
            .class_multipliers
            .as_ref()
            .is_some_and(|multipliers| {
                multipliers.len() != assignment.classes
                    || multipliers.iter().any(|value| {
                        !value.is_finite() || !(delta_config.normalizer * value).is_finite()
                    })
            })
    {
        return Err(Error::InvalidConfig(
            "OBB front-point writeback requires valid classes, scaling, and deltas".to_owned(),
        ));
    }
    validate_anchor_front_offsets(
        assignment.front_anchor_offsets.as_deref(),
        assignment.anchors.len() / 2,
    )?;
    let stride = assignment
        .head
        .grid_width
        .checked_mul(assignment.head.grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB head grid size overflows".to_owned()))?;
    let entries = assignment
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB head entry count overflows".to_owned()))?;
    let per_batch = assignment
        .head
        .mask
        .len()
        .checked_mul(entries)
        .and_then(|count| count.checked_mul(stride))
        .ok_or_else(|| Error::InvalidConfig("OBB head tensor size overflows".to_owned()))?;
    let expected = batch_truths
        .len()
        .checked_mul(per_batch)
        .ok_or_else(|| Error::InvalidConfig("OBB batch tensor size overflows".to_owned()))?;
    if head_output.len() != expected || head_delta.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB front-point tensor length mismatch: expected {expected}, output {}, delta {}",
            head_output.len(),
            head_delta.len()
        )));
    }

    let head_specs = [assignment.head.clone()];
    let positives = enumerate_obb_yolo_batch_positives(
        batch_truths,
        assignment.classes,
        &assignment.anchors,
        &head_specs,
        assignment.net_width,
        assignment.net_height,
        assignment.iou_thresh,
    )?;
    let center_front_anchor = assignment.front_anchor_offsets.is_some();
    for tagged in &positives {
        let positive = tagged.positive.assignment;
        let anchor_offset = positive
            .anchor_index
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidConfig("OBB anchor index overflows".to_owned()))?;
        if anchor_offset + 1 >= assignment.anchors.len() {
            return Err(Error::InvalidConfig(
                "OBB positive assignment references an invalid anchor".to_owned(),
            ));
        }
        let cell = positive.cell_y * assignment.head.grid_width + positive.cell_x;
        let base =
            tagged.batch_index * per_batch + positive.head_anchor_index * entries * stride + cell;
        let prediction =
            std::array::from_fn(|channel| finite_or_zero(head_output[base + channel * stride]));
        let context = ObbYoloPredictionMatchContext {
            anchor_width: assignment.anchors[anchor_offset],
            anchor_height: assignment.anchors[anchor_offset + 1],
            cell_x: positive.cell_x,
            cell_y: positive.cell_y,
            grid_width: assignment.head.grid_width,
            grid_height: assignment.head.grid_height,
            net_width: assignment.net_width,
            net_height: assignment.net_height,
            new_coords,
        };
        let truth = &batch_truths[tagged.batch_index][tagged.positive.truth_index];
        let class_id = assignment
            .class_map
            .as_ref()
            .map_or(truth.class_id, |mapping| mapping[truth.class_id]);
        let class_multiplier = assignment
            .class_multipliers
            .as_ref()
            .map_or(1.0, |multipliers| multipliers[class_id]);
        let mut truth_delta_config = delta_config;
        truth_delta_config.normalizer *= class_multiplier;
        if !truth_delta_config.normalizer.is_finite() {
            return Err(Error::InvalidConfig(
                "OBB front-point class-scaled normalizer is non-finite".to_owned(),
            ));
        }
        let front_deltas = obb_yolo_front_point_regression_deltas(
            prediction,
            truth,
            context,
            truth_delta_config,
            center_front_anchor,
        )?;
        let updates = [
            (base + 4 * stride, front_deltas.front_x),
            (base + 5 * stride, front_deltas.front_y),
            (base, front_deltas.center_x),
            (base + stride, front_deltas.center_y),
        ];
        let mut results = [0.0; 4];
        for (index, (delta_index, update)) in updates.into_iter().enumerate() {
            results[index] = head_delta[delta_index] + update;
            if !results[index].is_finite() {
                return Err(Error::InvalidConfig(
                    "OBB front-point writeback produced a non-finite delta".to_owned(),
                ));
            }
        }
        for ((delta_index, _), value) in updates.into_iter().zip(results) {
            head_delta[delta_index] = value;
        }
    }
    Ok(positives.len())
}

/// Dispatch source-ordered training deltas for one complete OBB YOLO head.
///
/// The pass order is all-cell objectness, truth-threshold class/box deltas,
/// best/additional positive objectness, then their class and box deltas. This
/// composes the existing per-head primitives; it does not run a multi-head
/// trainer, front-point loss, full metric aggregation, or network backprop.
#[allow(clippy::too_many_arguments)]
pub fn apply_obb_yolo_batch_head_training_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_output: &[f32],
    head_delta: &mut [f32],
    match_config: &ObbYoloHeadMatchConfig,
    assignment_config: &ObbYoloPositiveObjectnessConfig,
    focal_loss: bool,
    label_smooth_eps: f32,
    cls_normalizer: f32,
    box_delta_config: ObbYoloBoxDeltaConfig,
) -> Result<ObbYoloHeadTrainingDeltaStats> {
    if match_config.anchors != assignment_config.anchors
        || match_config.mask != assignment_config.head.mask
        || match_config.grid_width != assignment_config.head.grid_width
        || match_config.grid_height != assignment_config.head.grid_height
        || match_config.classes != assignment_config.classes
        || match_config.net_width != assignment_config.net_width
        || match_config.net_height != assignment_config.net_height
        || match_config.objectness_normalizer != assignment_config.objectness_normalizer
        || match_config.objectness_smooth != assignment_config.objectness_smooth
        || match_config.front_anchor_offsets != assignment_config.front_anchor_offsets
    {
        return Err(Error::InvalidConfig(
            "OBB head training passes require matching head and assignment configs".to_owned(),
        ));
    }

    apply_obb_yolo_batch_head_objectness_deltas(
        batch_truths,
        head_output,
        head_delta,
        match_config,
    )?;
    let truth_threshold_positive_cells = apply_obb_yolo_batch_head_truth_threshold_deltas(
        batch_truths,
        head_output,
        head_delta,
        match_config,
        assignment_config,
        focal_loss,
        label_smooth_eps,
        cls_normalizer,
        box_delta_config,
    )?;
    apply_obb_yolo_batch_head_positive_objectness_deltas(
        batch_truths,
        head_output,
        head_delta,
        assignment_config,
    )?;
    let anchor_class_stats = apply_obb_yolo_batch_head_positive_class_deltas(
        batch_truths,
        head_output,
        head_delta,
        assignment_config,
        focal_loss,
        label_smooth_eps,
        cls_normalizer,
    )?;
    let anchor_box_positive_count = apply_obb_yolo_batch_head_positive_box_deltas(
        batch_truths,
        head_output,
        head_delta,
        assignment_config,
        match_config.new_coords,
        box_delta_config,
    )?;
    let total_bbox_count = truth_threshold_positive_cells
        .checked_add(anchor_box_positive_count)
        .ok_or_else(|| Error::InvalidConfig("OBB bbox-positive count overflows".to_owned()))?;

    Ok(ObbYoloHeadTrainingDeltaStats {
        truth_threshold_positive_cells,
        anchor_class_stats,
        anchor_box_positive_count,
        total_bbox_count,
    })
}

/// Dispatch one-head OBB delta passes over separate head tensors in caller order.
///
/// Head outputs/deltas and both config slices must use the same ordering and
/// common global anchor/class/network definitions. Per-head grids, masks, score
/// branches, and coordinate modes may differ. This is multi-head delta dispatch,
/// not a network training loop or backward/optimizer integration.
#[allow(clippy::too_many_arguments)]
pub fn apply_obb_yolo_batch_multi_head_training_deltas(
    batch_truths: &[Vec<ObbLabel>],
    head_outputs: &[Vec<f32>],
    head_deltas: &mut [Vec<f32>],
    match_configs: &[ObbYoloHeadMatchConfig],
    assignment_configs: &[ObbYoloPositiveObjectnessConfig],
    focal_loss: bool,
    label_smooth_eps: f32,
    cls_normalizer: f32,
    box_delta_configs: &[ObbYoloBoxDeltaConfig],
) -> Result<ObbYoloBatchTrainingDeltaStats> {
    let head_count = head_outputs.len();
    if head_count == 0
        || head_deltas.len() != head_count
        || match_configs.len() != head_count
        || assignment_configs.len() != head_count
        || box_delta_configs.len() != head_count
    {
        return Err(Error::InvalidConfig(
            "OBB multi-head training requires matching nonempty head slices".to_owned(),
        ));
    }
    let first_assignment = &assignment_configs[0];
    for head_index in 0..head_count {
        let assignment = &assignment_configs[head_index];
        let matching = &match_configs[head_index];
        if assignment.anchors != first_assignment.anchors
            || assignment.classes != first_assignment.classes
            || assignment.net_width != first_assignment.net_width
            || assignment.net_height != first_assignment.net_height
            || assignment.front_anchor_offsets != first_assignment.front_anchor_offsets
            || match_configs[..head_index]
                .iter()
                .any(|previous| previous.head_index == matching.head_index)
        {
            return Err(Error::InvalidConfig(
                "OBB multi-head training requires shared anchors/classes/network size and unique head ids"
                    .to_owned(),
            ));
        }
    }

    let mut totals = ObbYoloBatchTrainingDeltaStats {
        per_head: Vec::new(),
        ..ObbYoloBatchTrainingDeltaStats::default()
    };
    totals
        .per_head
        .try_reserve_exact(head_count)
        .map_err(|_| Error::InvalidConfig("OBB head-stat allocation failed".to_owned()))?;
    for head_index in 0..head_count {
        let head_stats = apply_obb_yolo_batch_head_training_deltas(
            batch_truths,
            &head_outputs[head_index],
            &mut head_deltas[head_index],
            &match_configs[head_index],
            &assignment_configs[head_index],
            focal_loss,
            label_smooth_eps,
            cls_normalizer,
            box_delta_configs[head_index],
        )?;
        totals.truth_threshold_positive_cells = totals
            .truth_threshold_positive_cells
            .checked_add(head_stats.truth_threshold_positive_cells)
            .ok_or_else(|| Error::InvalidConfig("OBB truth-positive count overflows".to_owned()))?;
        totals.anchor_class_stats.class_output_sum +=
            head_stats.anchor_class_stats.class_output_sum;
        if !totals.anchor_class_stats.class_output_sum.is_finite() {
            return Err(Error::InvalidConfig(
                "OBB aggregate class-output sum is non-finite".to_owned(),
            ));
        }
        totals.anchor_class_stats.positive_count = totals
            .anchor_class_stats
            .positive_count
            .checked_add(head_stats.anchor_class_stats.positive_count)
            .ok_or_else(|| Error::InvalidConfig("OBB class-positive count overflows".to_owned()))?;
        totals.anchor_box_positive_count = totals
            .anchor_box_positive_count
            .checked_add(head_stats.anchor_box_positive_count)
            .ok_or_else(|| Error::InvalidConfig("OBB box-positive count overflows".to_owned()))?;
        totals.total_bbox_count = totals
            .total_bbox_count
            .checked_add(head_stats.total_bbox_count)
            .ok_or_else(|| Error::InvalidConfig("OBB bbox-positive count overflows".to_owned()))?;
        totals.per_head.push(head_stats);
    }
    Ok(totals)
}

fn validate_anchor_front_offsets(offsets: Option<&[[f32; 2]]>, anchor_count: usize) -> Result<()> {
    if offsets.is_some_and(|offsets| {
        offsets.len() != anchor_count || offsets.iter().flatten().any(|value| !value.is_finite())
    }) {
        return Err(Error::InvalidConfig(
            "OBB front-anchor offsets must be finite and match the anchor list".to_owned(),
        ));
    }
    Ok(())
}

fn validate_anchor_assignment_inputs(
    anchors: &[f32],
    mask: &[usize],
    net_width: usize,
    net_height: usize,
    grid_width: usize,
    grid_height: usize,
) -> Result<()> {
    if anchors.is_empty() || anchors.len() % 2 != 0 {
        return Err(Error::InvalidConfig(
            "OBB anchor list must contain width/height pairs".to_owned(),
        ));
    }
    if mask.iter().any(|&index| index >= anchors.len() / 2) {
        return Err(Error::InvalidConfig(
            "OBB anchor mask index exceeds the anchor list".to_owned(),
        ));
    }
    if net_width == 0 || net_height == 0 {
        return Err(Error::InvalidConfig(
            "OBB anchor assignment requires nonzero network dimensions".to_owned(),
        ));
    }
    if grid_width == 0 || grid_height == 0 {
        return Err(Error::InvalidConfig(
            "OBB anchor assignment requires a nonzero grid".to_owned(),
        ));
    }
    if anchors.chunks_exact(2).any(|pair| {
        !pair[0].is_finite() || !pair[1].is_finite() || pair[0] <= 0.0 || pair[1] <= 0.0
    }) {
        return Err(Error::InvalidConfig(
            "OBB anchors must have positive finite width and height".to_owned(),
        ));
    }
    Ok(())
}

fn best_obb_yolo_anchor(
    truth: &ObbLabel,
    anchors: &[f32],
    net_width: usize,
    net_height: usize,
) -> Result<(usize, f32)> {
    let mut best_anchor = 0;
    let mut best_score = 0.0;
    for (anchor_index, pair) in anchors.chunks_exact(2).enumerate() {
        let score = obb_yolo_anchor_match_score(truth, pair[0], pair[1], net_width, net_height)?;
        if score > best_score {
            best_score = score;
            best_anchor = anchor_index;
        }
    }
    Ok((best_anchor, best_score))
}

fn obb_yolo_truth_cell(truth: &ObbLabel, grid_width: usize, grid_height: usize) -> (usize, usize) {
    (
        ((truth.x * grid_width as f32) as usize).min(grid_width - 1),
        ((truth.y * grid_height as f32) as usize).min(grid_height - 1),
    )
}

fn validate_truth(truth: &ObbLabel) -> Result<()> {
    if !truth.x.is_finite()
        || !truth.y.is_finite()
        || !truth.w.is_finite()
        || !truth.h.is_finite()
        || !truth.front_x.is_finite()
        || !truth.front_y.is_finite()
        || !(0.0..=1.0).contains(&truth.x)
        || !(0.0..=1.0).contains(&truth.y)
        || truth.w <= 0.0
        || truth.h <= 0.0
    {
        return Err(Error::InvalidConfig(
            "OBB truth requires finite coordinates, a normalized center, and positive dimensions"
                .to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        OBB_YOLO_TRUTH_SIZE, ObbYoloAnchorAssignment, ObbYoloBatchCellPrediction,
        ObbYoloBoxDeltaConfig, ObbYoloCellTruthMatch, ObbYoloFrontPointDeltaConfig,
        ObbYoloHeadAssignmentSpec, ObbYoloHeadMatchConfig, ObbYoloMatchScoreBranch,
        ObbYoloMatchScoreConfig, ObbYoloPositiveKind, ObbYoloPositiveObjectnessConfig,
        ObbYoloPredictionMatchContext, apply_obb_yolo_batch_head_objectness_deltas,
        apply_obb_yolo_batch_head_positive_box_deltas,
        apply_obb_yolo_batch_head_positive_class_deltas,
        apply_obb_yolo_batch_head_positive_front_point_deltas,
        apply_obb_yolo_batch_head_positive_objectness_deltas,
        apply_obb_yolo_batch_head_training_deltas,
        apply_obb_yolo_batch_head_truth_threshold_deltas,
        apply_obb_yolo_batch_multi_head_training_deltas, assign_obb_yolo_anchor,
        decode_obb_yolo_front_point, decode_obb_yolo_prediction, encode_obb_yolo_truth_batch,
        enumerate_obb_yolo_batch_positives, enumerate_obb_yolo_head_positives,
        match_obb_yolo_batch_cell_predictions, match_obb_yolo_cell_prediction,
        match_obb_yolo_cell_prediction_with_score_config, obb_yolo_anchor_match_score,
        obb_yolo_angular_prediction_match_score_with_viou, obb_yolo_box_regression_deltas,
        obb_yolo_cell_objectness_delta, obb_yolo_ciou_prediction_match_score,
        obb_yolo_class_deltas, obb_yolo_cobb_prediction_match_score_with_viou,
        obb_yolo_covariance_prediction_match_score,
        obb_yolo_covariance_prediction_match_score_with_viou,
        obb_yolo_front_point_regression_deltas, obb_yolo_match_score,
        select_obb_yolo_cell_truth_match, visit_obb_yolo_batch_head_matches,
    };
    use crate::obb_labels::ObbLabel;

    fn truth(x: f32, y: f32, w: f32, h: f32) -> ObbLabel {
        ObbLabel {
            class_id: 0,
            x,
            y,
            w,
            h,
            left: (x - w / 2.0).clamp(0.0, 1.0),
            right: (x + w / 2.0).clamp(0.0, 1.0),
            top: (y - h / 2.0).clamp(0.0, 1.0),
            bottom: (y + h / 2.0).clamp(0.0, 1.0),
            front_x: x,
            front_y: y,
            has_front: false,
        }
    }

    #[test]
    fn truth_encoder_preserves_batch_slots_and_front_placeholders() {
        let absent = truth(0.25, 0.5, 0.2, 0.4);
        let mut explicit = truth(0.75, 0.25, 0.1, 0.2);
        explicit.class_id = 2;
        explicit.front_x = 0.8;
        explicit.front_y = 0.125;
        explicit.has_front = true;
        let mut second_image = truth(0.5, 0.75, 0.3, 0.1);
        second_image.class_id = 1;

        let encoded =
            encode_obb_yolo_truth_batch(&[vec![absent, explicit], vec![second_image]], 2, 3)
                .expect("source-layout truth batch");
        assert_eq!(encoded.batch_size, 2);
        assert_eq!(encoded.max_boxes, 2);
        assert_eq!(OBB_YOLO_TRUTH_SIZE, 7);
        assert_eq!(
            encoded.values,
            vec![
                absent.x,
                absent.y,
                absent.w,
                absent.h,
                0.0,
                absent.x,
                absent.y,
                explicit.x,
                explicit.y,
                explicit.w,
                explicit.h,
                2.0,
                explicit.front_x,
                explicit.front_y,
                second_image.x,
                second_image.y,
                second_image.w,
                second_image.h,
                1.0,
                second_image.x,
                second_image.y,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        );
    }

    #[test]
    fn truth_encoder_rejects_invalid_dimensions_classes_and_capacity() {
        let label = truth(0.5, 0.5, 0.25, 0.25);
        assert!(encode_obb_yolo_truth_batch(&[], 1, 1).is_err());
        assert!(encode_obb_yolo_truth_batch(&[vec![label]], 0, 1).is_err());
        assert!(encode_obb_yolo_truth_batch(&[vec![label]], 1, 0).is_err());
        assert!(encode_obb_yolo_truth_batch(&[vec![label, label]], 1, 1).is_err());

        let mut invalid_class = label;
        invalid_class.class_id = 1;
        assert!(encode_obb_yolo_truth_batch(&[vec![invalid_class]], 1, 1).is_err());
        let mut imprecise_class = label;
        imprecise_class.class_id = (1usize << 24) + 1;
        assert!(
            encode_obb_yolo_truth_batch(&[vec![imprecise_class]], 1, imprecise_class.class_id + 1)
                .is_err()
        );
        let mut invalid_size = label;
        invalid_size.w = 0.0;
        assert!(encode_obb_yolo_truth_batch(&[vec![invalid_size]], 1, 1).is_err());
        let mut invalid_center = label;
        invalid_center.x = f32::NAN;
        assert!(encode_obb_yolo_truth_batch(&[vec![invalid_center]], 1, 1).is_err());
    }

    #[test]
    fn positive_enumerator_emits_primary_then_strictly_qualified_masked_anchors() {
        let mut label = truth(0.51, 0.39, 0.4, 0.4);
        label.class_id = 1;
        let assignments = enumerate_obb_yolo_head_positives(
            &[label],
            2,
            &[40.0, 40.0, 30.0, 40.0, 10.0, 10.0],
            &[2, 0, 1],
            100,
            100,
            5,
            4,
            0.7,
        )
        .expect("positive matches");
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].truth_index, 0);
        assert_eq!(assignments[0].class_id, 1);
        assert_eq!(assignments[0].kind, ObbYoloPositiveKind::BestAnchor);
        assert_eq!(
            assignments[0].assignment,
            ObbYoloAnchorAssignment {
                anchor_index: 0,
                head_anchor_index: 1,
                cell_x: 2,
                cell_y: 1,
                match_score: 1.0,
            }
        );
        assert_eq!(assignments[1].truth_index, 0);
        assert_eq!(assignments[1].class_id, 1);
        assert_eq!(assignments[1].kind, ObbYoloPositiveKind::AdditionalAnchor);
        assert_eq!(assignments[1].assignment.anchor_index, 1);
        assert_eq!(assignments[1].assignment.head_anchor_index, 2);
        assert_eq!(assignments[1].assignment.cell_x, 2);
        assert_eq!(assignments[1].assignment.cell_y, 1);
        assert!((assignments[1].assignment.match_score - 0.75).abs() < 1e-6);
    }

    #[test]
    fn batch_positive_enumerator_keeps_head_batch_truth_order_and_masks() {
        let mut first = truth(0.15, 0.15, 0.1, 0.1);
        first.class_id = 0;
        let mut second = truth(0.55, 0.55, 0.2, 0.2);
        second.class_id = 1;
        let heads = [
            ObbYoloHeadAssignmentSpec {
                mask: vec![0],
                grid_width: 10,
                grid_height: 10,
            },
            ObbYoloHeadAssignmentSpec {
                mask: vec![1],
                grid_width: 10,
                grid_height: 10,
            },
        ];
        let assigned = enumerate_obb_yolo_batch_positives(
            &[vec![first], vec![second]],
            2,
            &[10.0, 10.0, 20.0, 20.0],
            &heads,
            100,
            100,
            0.2,
        )
        .unwrap();
        assert_eq!(assigned.len(), 4);
        let expected = [
            (0, 0, 0, ObbYoloPositiveKind::BestAnchor, 0, 1),
            (0, 1, 1, ObbYoloPositiveKind::AdditionalAnchor, 0, 5),
            (1, 0, 0, ObbYoloPositiveKind::AdditionalAnchor, 1, 1),
            (1, 1, 1, ObbYoloPositiveKind::BestAnchor, 1, 5),
        ];
        for (actual, (head, batch, class_id, kind, anchor, cell)) in assigned.iter().zip(expected) {
            assert_eq!(actual.head_index, head);
            assert_eq!(actual.batch_index, batch);
            assert_eq!(actual.positive.truth_index, 0);
            assert_eq!(actual.positive.class_id, class_id);
            assert_eq!(actual.positive.kind, kind);
            assert_eq!(actual.positive.assignment.anchor_index, anchor);
            assert_eq!(actual.positive.assignment.head_anchor_index, 0);
            assert_eq!(actual.positive.assignment.cell_x, cell);
            assert_eq!(actual.positive.assignment.cell_y, cell);
        }
    }

    #[test]
    fn batch_positive_enumerator_rejects_empty_batch_and_head_lists() {
        let head = ObbYoloHeadAssignmentSpec {
            mask: vec![0],
            grid_width: 1,
            grid_height: 1,
        };
        let anchors = [10.0, 10.0];
        assert!(
            enumerate_obb_yolo_batch_positives(
                &[],
                1,
                &anchors,
                std::slice::from_ref(&head),
                10,
                10,
                1.0,
            )
            .is_err()
        );
        assert!(
            enumerate_obb_yolo_batch_positives(&[Vec::new()], 1, &anchors, &[], 10, 10, 1.0,)
                .is_err()
        );
    }

    #[test]
    fn front_point_decode_applies_source_directional_anchor_prior() {
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 24.0,
            anchor_height: 12.0,
            cell_x: 2,
            cell_y: 1,
            grid_width: 8,
            grid_height: 4,
            net_width: 64,
            net_height: 32,
            new_coords: false,
        };
        let prediction = [0.5, 0.25, 0.0, 0.0, 0.4, 0.75];
        let direct = decode_obb_yolo_front_point(prediction, context, None).unwrap();
        assert!((direct[0] - 0.3).abs() < 1e-6);
        assert!((direct[1] - 0.4375).abs() < 1e-6);
        let anchored =
            decode_obb_yolo_front_point(prediction, context, Some([0.25, -0.5])).unwrap();
        assert!((anchored[0] - 0.45625).abs() < 1e-6);
        assert!((anchored[1] - 0.3125).abs() < 1e-6);
        assert!(decode_obb_yolo_front_point(prediction, context, Some([f32::NAN, 0.0])).is_err());
    }

    #[test]
    fn batch_cell_matching_uses_each_image_truth_and_preserves_prediction_order() {
        let first_truth = truth(0.35, 0.35, 0.1, 0.1);
        let second_truth = truth(1.0, 1.0, 0.1, 0.1);
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 10.0,
            anchor_height: 10.0,
            cell_x: 3,
            cell_y: 3,
            grid_width: 10,
            grid_height: 10,
            net_width: 100,
            net_height: 100,
            new_coords: false,
        };
        let score_config = ObbYoloMatchScoreConfig {
            covariance_normalizer: 1.0,
            ..ObbYoloMatchScoreConfig::default()
        };
        let prediction = [0.5, 0.5, 0.0, 0.0, 0.5, 0.5];
        let matches = match_obb_yolo_batch_cell_predictions(
            &[vec![first_truth], vec![second_truth]],
            &[
                ObbYoloBatchCellPrediction {
                    batch_index: 1,
                    head_index: 2,
                    anchor_index: 4,
                    head_anchor_index: 0,
                    prediction,
                    class_outputs: vec![0.9],
                    context,
                    score_config,
                    ignore_thresh: 0.5,
                    truth_thresh: 0.5,
                    predicted_objectness: 0.25,
                    objectness_normalizer: 1.0,
                    objectness_smooth: false,
                },
                ObbYoloBatchCellPrediction {
                    batch_index: 0,
                    head_index: 1,
                    anchor_index: 3,
                    head_anchor_index: 0,
                    prediction,
                    class_outputs: vec![0.9],
                    context,
                    score_config,
                    ignore_thresh: 0.5,
                    truth_thresh: 0.5,
                    predicted_objectness: 0.25,
                    objectness_normalizer: 1.0,
                    objectness_smooth: false,
                },
            ],
        )
        .unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!((matches[0].batch_index, matches[0].head_index), (1, 2));
        assert_eq!(matches[0].anchor_index, 4);
        assert_eq!(matches[0].head_anchor_index, 0);
        assert!(matches[0].truth_match.best_score < 0.5);
        assert_eq!(matches[0].truth_match.positive_truth_index, None);
        assert!((matches[0].objectness_delta + 0.25).abs() < 1e-6);
        assert_eq!((matches[1].batch_index, matches[1].head_index), (0, 1));
        assert_eq!(matches[1].anchor_index, 3);
        assert_eq!(matches[1].head_anchor_index, 0);
        assert_eq!(matches[1].truth_match.positive_truth_index, Some(0));
        assert!(matches[1].truth_match.best_score > 0.99);
        assert!((matches[1].objectness_delta - 0.75).abs() < 1e-6);
    }

    #[test]
    fn batch_cell_matching_rejects_invalid_batch_indices() {
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 10.0,
            anchor_height: 10.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 10,
            net_height: 10,
            new_coords: false,
        };
        let cell = ObbYoloBatchCellPrediction {
            batch_index: 1,
            head_index: 0,
            anchor_index: 0,
            head_anchor_index: 0,
            prediction: [0.0; 6],
            class_outputs: vec![0.0],
            context,
            score_config: ObbYoloMatchScoreConfig::default(),
            ignore_thresh: 0.5,
            truth_thresh: 0.5,
            predicted_objectness: 0.0,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        assert!(match_obb_yolo_batch_cell_predictions(&[Vec::new()], &[cell]).is_err());
    }

    #[test]
    fn batch_head_objectness_delta_writeback_preserves_other_entries() {
        let truths = [vec![truth(0.5, 0.5, 0.2, 0.2)], Vec::new()];
        let config = ObbYoloHeadMatchConfig {
            head_index: 0,
            anchors: vec![10.0, 10.0, 20.0, 20.0],
            mask: vec![1],
            grid_width: 1,
            grid_height: 1,
            classes: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
            front_anchor_offsets: None,
            score_config: ObbYoloMatchScoreConfig {
                covariance_normalizer: 1.0,
                ..ObbYoloMatchScoreConfig::default()
            },
            ignore_thresh: 0.5,
            truth_thresh: 0.5,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        let head_output = [
            0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.25, 0.9,
        ];
        let mut head_delta = vec![7.0; head_output.len()];
        apply_obb_yolo_batch_head_objectness_deltas(
            &truths,
            &head_output,
            &mut head_delta,
            &config,
        )
        .unwrap();
        assert!((head_delta[6] - 0.1).abs() < 1e-6);
        assert!((head_delta[14] + 0.25).abs() < 1e-6);
        for (index, &value) in head_delta.iter().enumerate() {
            if index != 6 && index != 14 {
                assert_eq!(value, 7.0, "delta entry {index} changed");
            }
        }
    }

    #[test]
    fn batch_head_objectness_delta_writeback_rejects_mismatched_delta_shape() {
        let config = ObbYoloHeadMatchConfig {
            head_index: 0,
            anchors: vec![10.0, 10.0],
            mask: vec![0],
            grid_width: 1,
            grid_height: 1,
            classes: 1,
            net_width: 10,
            net_height: 10,
            new_coords: false,
            front_anchor_offsets: None,
            score_config: ObbYoloMatchScoreConfig::default(),
            ignore_thresh: 0.5,
            truth_thresh: 0.5,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        let mut head_delta = [0.0; 7];
        assert!(
            apply_obb_yolo_batch_head_objectness_deltas(
                &[Vec::new()],
                &[0.0; 8],
                &mut head_delta,
                &config,
            )
            .is_err()
        );
    }

    #[test]
    fn positive_anchor_objectness_overrides_use_mask_slots_mapping_and_smoothing() {
        let truth = truth(0.5, 0.5, 0.2, 0.2);
        let config = ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0, 1],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: vec![10.0, 10.0, 20.0, 20.0],
            classes: 2,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: None,
            iou_thresh: 0.2,
            objectness_normalizer: 2.0,
            objectness_smooth: false,
            class_map: Some(vec![1, 0]),
            class_multipliers: Some(vec![2.0, 3.0]),
        };
        let mut head_output = vec![0.0; 18];
        head_output[6] = 0.2;
        head_output[15] = 0.3;
        let mut head_delta = vec![-5.0; 18];
        apply_obb_yolo_batch_head_positive_objectness_deltas(
            &[vec![truth]],
            &head_output,
            &mut head_delta,
            &config,
        )
        .unwrap();
        assert!((head_delta[6] - 4.8).abs() < 1e-6);
        assert!((head_delta[15] - 4.2).abs() < 1e-6);
        for (index, &value) in head_delta.iter().enumerate() {
            if index != 6 && index != 15 {
                assert_eq!(value, -5.0, "delta entry {index} changed");
            }
        }

        let mut smooth_config = config;
        smooth_config.objectness_smooth = true;
        let mut smooth_delta = vec![0.0; 18];
        smooth_delta[15] = 0.4;
        apply_obb_yolo_batch_head_positive_objectness_deltas(
            &[vec![truth]],
            &head_output,
            &mut smooth_delta,
            &smooth_config,
        )
        .unwrap();
        assert!((smooth_delta[6] - 4.8).abs() < 1e-6);
        assert_eq!(smooth_delta[15], 0.4);
    }

    #[test]
    fn front_point_deltas_match_source_chain_and_center_prior_gate() {
        let mut label = truth(0.5, 0.5, 0.2, 0.1);
        label.front_x = 0.66;
        label.front_y = 0.57;
        label.has_front = true;
        let prediction = [0.5, 0.5, 0.0, 0.0, 1.2, 0.7];
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 20.0,
            anchor_height: 10.0,
            cell_x: 1,
            cell_y: 1,
            grid_width: 4,
            grid_height: 3,
            net_width: 100,
            net_height: 60,
            new_coords: false,
        };
        let config = ObbYoloFrontPointDeltaConfig {
            normalizer: 1.4,
            max_delta: 100.0,
            ..ObbYoloFrontPointDeltaConfig::default()
        };
        let actual =
            obb_yolo_front_point_regression_deltas(prediction, &label, context, config, true)
                .unwrap();
        let decoded = decode_obb_yolo_prediction(prediction, context).unwrap();
        let front_width = (label.w * 0.25).max(1e-5);
        let front_height = (label.h * 0.25).max(1e-5);
        let (ciou, gradient_x, gradient_y) = crate::train::obb_front_point_ciou_gradient(
            [decoded.front_x, decoded.front_y, front_width, front_height],
            [label.front_x, label.front_y, front_width, front_height],
            config.iou_loss,
        );
        let expected_x = (gradient_x / context.grid_width as f32 * config.normalizer)
            .clamp(-config.max_delta, config.max_delta);
        let expected_y = (gradient_y / context.grid_height as f32 * config.normalizer)
            .clamp(-config.max_delta, config.max_delta);
        assert!((actual.front_x - expected_x).abs() < 1e-6);
        assert!((actual.front_y - expected_y).abs() < 1e-6);
        assert!((actual.loss - (1.0 - ciou).max(0.0)).abs() < 1e-6);
        assert!((actual.center_x - actual.front_x).abs() < 1e-7);
        assert!((actual.center_y - actual.front_y).abs() < 1e-7);
        assert!(actual.loss.is_finite() && actual.loss >= 0.0);

        let direct =
            obb_yolo_front_point_regression_deltas(prediction, &label, context, config, false)
                .unwrap();
        assert_eq!(direct.center_x, 0.0);
        assert_eq!(direct.center_y, 0.0);
        let absent = obb_yolo_front_point_regression_deltas(
            prediction,
            &truth(0.5, 0.5, 0.2, 0.1),
            context,
            config,
            true,
        )
        .unwrap();
        assert_eq!(absent, Default::default());
    }

    #[test]
    fn finite_difference_box_deltas_match_source_branch_scale_and_accumulation() {
        let truth = truth(0.5, 0.5, 0.2, 0.1);
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 20.0,
            anchor_height: 10.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
        };
        let prediction = [0.35, 0.6, 0.15, -0.1, 0.7, 0.4];
        let existing = [0.1, -0.2, 0.3, -0.4];
        let score_config = ObbYoloMatchScoreConfig {
            cobb_normalizer: 2.5,
            viou_weight: 0.2,
            ..ObbYoloMatchScoreConfig::default()
        };
        let config = ObbYoloBoxDeltaConfig {
            score: score_config,
            iou_normalizer: 0.3,
            max_delta: f32::MAX,
            accumulate: true,
        };
        let actual =
            obb_yolo_box_regression_deltas(prediction, existing, &truth, context, config).unwrap();
        let mut expected = existing;
        for channel in 0..4 {
            let base = prediction[channel];
            let epsilon = 1e-3 * (base.abs() + 1.0);
            let mut plus = prediction;
            let mut minus = prediction;
            plus[channel] += epsilon;
            minus[channel] -= epsilon;
            let plus_score = obb_yolo_match_score(plus, &truth, context, score_config)
                .unwrap()
                .score;
            let minus_score = obb_yolo_match_score(minus, &truth, context, score_config)
                .unwrap()
                .score;
            let sigmoid_chain = if channel < 2 {
                base * (1.0 - base)
            } else {
                1.0
            };
            let gradient = (plus_score - minus_score) / (2.0 * epsilon);
            expected[channel] += (gradient * sigmoid_chain * 0.3 * 2.5).clamp(-1.0, 1.0);
        }
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }

        let reset_and_clipped = obb_yolo_box_regression_deltas(
            prediction,
            existing,
            &truth,
            context,
            ObbYoloBoxDeltaConfig {
                max_delta: 0.01,
                accumulate: false,
                ..config
            },
        )
        .unwrap();
        assert!(reset_and_clipped.iter().all(|delta| delta.abs() <= 0.01));
        assert!(reset_and_clipped.iter().any(|delta| *delta != 0.0));

        let new_coords_context = ObbYoloPredictionMatchContext {
            new_coords: true,
            ..context
        };
        let new_coords_actual = obb_yolo_box_regression_deltas(
            prediction,
            existing,
            &truth,
            new_coords_context,
            config,
        )
        .unwrap();
        let epsilon = 1e-3 * (prediction[0].abs() + 1.0);
        let mut plus = prediction;
        let mut minus = prediction;
        plus[0] += epsilon;
        minus[0] -= epsilon;
        let plus_score = obb_yolo_match_score(plus, &truth, new_coords_context, score_config)
            .unwrap()
            .score;
        let minus_score = obb_yolo_match_score(minus, &truth, new_coords_context, score_config)
            .unwrap()
            .score;
        let expected_x = existing[0]
            + (((plus_score - minus_score) / (2.0 * epsilon))
                * prediction[0]
                * (1.0 - prediction[0])
                * 0.3
                * 2.5)
                .clamp(-1.0, 1.0);
        assert!((new_coords_actual[0] - expected_x).abs() < 1e-6);
    }

    #[test]
    fn truth_threshold_cell_deltas_apply_class_override_and_weighted_box_gradients() {
        let mut label = truth(0.5, 0.5, 0.2, 0.1);
        label.front_x = 0.68;
        label.front_y = 0.55;
        label.has_front = true;
        let labels = [vec![label]];
        let anchors = vec![20.0, 10.0];
        let head_match = ObbYoloHeadMatchConfig {
            head_index: 0,
            anchors: anchors.clone(),
            mask: vec![0],
            grid_width: 1,
            grid_height: 1,
            classes: 2,
            net_width: 100,
            net_height: 100,
            new_coords: false,
            front_anchor_offsets: None,
            score_config: ObbYoloMatchScoreConfig::default(),
            ignore_thresh: 0.5,
            truth_thresh: 0.1,
            objectness_normalizer: 2.0,
            objectness_smooth: true,
        };
        let assignment = ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0],
                grid_width: 1,
                grid_height: 1,
            },
            anchors,
            classes: 2,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: None,
            iou_thresh: 1.0,
            objectness_normalizer: 2.0,
            objectness_smooth: true,
            class_map: Some(vec![1, 0]),
            class_multipliers: Some(vec![2.0, 3.0]),
        };
        let output = [0.52, 0.47, 0.1, -0.1, 0.65, 0.55, 0.3, 0.2, 0.4];
        let mut delta = [0.0; 9];
        apply_obb_yolo_batch_head_objectness_deltas(&labels, &output, &mut delta, &head_match)
            .unwrap();
        let objectness_delta = delta[6];
        let box_delta_config = ObbYoloBoxDeltaConfig {
            score: head_match.score_config,
            iou_normalizer: 0.5,
            ..ObbYoloBoxDeltaConfig::default()
        };
        assert_eq!(
            apply_obb_yolo_batch_head_truth_threshold_deltas(
                &labels,
                &output,
                &mut delta,
                &head_match,
                &assignment,
                false,
                0.1,
                0.5,
                box_delta_config,
            )
            .unwrap(),
            1
        );
        let prediction = [0.52, 0.47, 0.1, -0.1, 0.65, 0.55];
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 20.0,
            anchor_height: 10.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
        };
        let match_result = match_obb_yolo_cell_prediction_with_score_config(
            prediction,
            &labels[0],
            &[0.2, 0.4],
            context,
            head_match.score_config,
            head_match.ignore_thresh,
            head_match.truth_thresh,
        )
        .unwrap();
        assert!(match_result.positive_truth_index.is_some());
        let iou_multiplier = match_result.best_score * match_result.best_score;
        assert!((delta[6] - objectness_delta).abs() < 1e-6);
        assert!((delta[7] + 0.15).abs() < 1e-6);
        assert!((delta[8] - 3.0 * (iou_multiplier - 0.4)).abs() < 1e-6);
        let expected_box = obb_yolo_box_regression_deltas(
            prediction,
            [0.0; 4],
            &labels[0][0],
            context,
            ObbYoloBoxDeltaConfig {
                iou_normalizer: 1.5,
                ..box_delta_config
            },
        )
        .unwrap();
        for channel in 0..4 {
            assert!((delta[channel] - expected_box[channel]).abs() < 1e-6);
        }
        assert_eq!(delta[4], 0.0);
        assert_eq!(delta[5], 0.0);

        let mut sequenced = [0.0; 9];
        apply_obb_yolo_batch_head_objectness_deltas(&labels, &output, &mut sequenced, &head_match)
            .unwrap();
        let truth_threshold_positive_cells = apply_obb_yolo_batch_head_truth_threshold_deltas(
            &labels,
            &output,
            &mut sequenced,
            &head_match,
            &assignment,
            false,
            0.1,
            0.5,
            box_delta_config,
        )
        .unwrap();
        apply_obb_yolo_batch_head_positive_objectness_deltas(
            &labels,
            &output,
            &mut sequenced,
            &assignment,
        )
        .unwrap();
        let anchor_class_stats = apply_obb_yolo_batch_head_positive_class_deltas(
            &labels,
            &output,
            &mut sequenced,
            &assignment,
            false,
            0.1,
            0.5,
        )
        .unwrap();
        let anchor_box_positive_count = apply_obb_yolo_batch_head_positive_box_deltas(
            &labels,
            &output,
            &mut sequenced,
            &assignment,
            head_match.new_coords,
            box_delta_config,
        )
        .unwrap();
        let mut dispatched = [0.0; 9];
        let stats = apply_obb_yolo_batch_head_training_deltas(
            &labels,
            &output,
            &mut dispatched,
            &head_match,
            &assignment,
            false,
            0.1,
            0.5,
            box_delta_config,
        )
        .unwrap();
        assert_eq!(dispatched, sequenced);
        assert_eq!(
            stats.truth_threshold_positive_cells,
            truth_threshold_positive_cells
        );
        assert_eq!(stats.anchor_class_stats, anchor_class_stats);
        assert_eq!(stats.anchor_box_positive_count, anchor_box_positive_count);
        assert_eq!(
            stats.total_bbox_count,
            truth_threshold_positive_cells + anchor_box_positive_count
        );
    }

    #[test]
    fn head_training_routes_directional_anchor_priors_through_matching_and_box_deltas() {
        let mut label = truth(0.5, 0.5, 0.2, 0.1);
        label.front_x = 0.68;
        label.front_y = 0.55;
        label.has_front = true;
        let labels = [vec![label]];
        let anchors = vec![20.0, 10.0];
        let matching = |offsets| ObbYoloHeadMatchConfig {
            head_index: 0,
            anchors: anchors.clone(),
            mask: vec![0],
            grid_width: 1,
            grid_height: 1,
            classes: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
            front_anchor_offsets: offsets,
            score_config: ObbYoloMatchScoreConfig {
                covariance_normalizer: 1.0,
                ..ObbYoloMatchScoreConfig::default()
            },
            ignore_thresh: 0.5,
            truth_thresh: 0.0,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        let assignment = |offsets| ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: anchors.clone(),
            classes: 1,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: offsets,
            iou_thresh: 1.0,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
            class_map: None,
            class_multipliers: None,
        };
        let output = [0.55, 0.45, 0.0, 0.0, 0.08, 0.05, 0.4, 0.3];
        let score = ObbYoloMatchScoreConfig {
            covariance_normalizer: 1.0,
            ..ObbYoloMatchScoreConfig::default()
        };
        let box_config = ObbYoloBoxDeltaConfig {
            score,
            iou_normalizer: 0.5,
            ..ObbYoloBoxDeltaConfig::default()
        };
        let offsets = Some(vec![[0.25, 0.5]]);
        let mut anchored_delta = [0.0; 8];
        let stats = apply_obb_yolo_batch_head_training_deltas(
            &labels,
            &output,
            &mut anchored_delta,
            &matching(offsets.clone()),
            &assignment(offsets),
            false,
            0.0,
            1.0,
            box_config,
        )
        .unwrap();
        assert_eq!(stats.truth_threshold_positive_cells, 1);
        assert_eq!(stats.anchor_class_stats.positive_count, 1);
        assert_eq!(stats.anchor_box_positive_count, 1);
        assert!(
            visit_obb_yolo_batch_head_matches(
                &labels,
                &output,
                &matching(Some(vec![])),
                |_| Ok(()),
            )
            .is_err()
        );
        assert!(
            visit_obb_yolo_batch_head_matches(
                &labels,
                &output,
                &matching(Some(vec![[f32::NAN, 0.0]])),
                |_| Ok(()),
            )
            .is_err()
        );

        let mut direct_delta = [0.0; 8];
        apply_obb_yolo_batch_head_training_deltas(
            &labels,
            &output,
            &mut direct_delta,
            &matching(None),
            &assignment(None),
            false,
            0.0,
            1.0,
            box_config,
        )
        .unwrap();
        assert!(
            anchored_delta[..2]
                .iter()
                .zip(&direct_delta[..2])
                .any(|(anchored, direct)| (anchored - direct).abs() > 1e-5)
        );
    }

    #[test]
    fn multi_head_training_dispatches_each_head_and_returns_tagged_metrics() {
        let mut label = truth(0.5, 0.5, 0.2, 0.1);
        label.front_x = 0.68;
        label.front_y = 0.55;
        label.has_front = true;
        let labels = [vec![label]];
        let anchors = vec![20.0, 10.0, 30.0, 15.0];
        let match_config = |head_index, mask| ObbYoloHeadMatchConfig {
            head_index,
            anchors: anchors.clone(),
            mask: vec![mask],
            grid_width: 1,
            grid_height: 1,
            classes: 2,
            net_width: 100,
            net_height: 100,
            new_coords: false,
            front_anchor_offsets: None,
            score_config: ObbYoloMatchScoreConfig::default(),
            ignore_thresh: 0.5,
            truth_thresh: 0.1,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        let assignment_config = |mask| ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![mask],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: anchors.clone(),
            classes: 2,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: None,
            iou_thresh: 0.1,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
            class_map: None,
            class_multipliers: None,
        };
        let outputs = vec![
            vec![0.5, 0.5, 0.0, 0.0, 0.68, 0.55, 0.4, 0.3, 0.2],
            vec![
                0.5,
                0.5,
                (20.0f32 / 30.0).ln(),
                (10.0f32 / 15.0).ln(),
                0.68,
                0.55,
                0.4,
                0.3,
                0.2,
            ],
        ];
        let matching = [match_config(5, 0), match_config(9, 1)];
        let assignments = [assignment_config(0), assignment_config(1)];
        let box_configs = [
            ObbYoloBoxDeltaConfig::default(),
            ObbYoloBoxDeltaConfig::default(),
        ];
        let mut deltas = vec![vec![0.0; 9], vec![0.0; 9]];
        let stats = apply_obb_yolo_batch_multi_head_training_deltas(
            &labels,
            &outputs,
            &mut deltas,
            &matching,
            &assignments,
            false,
            0.0,
            1.0,
            &box_configs,
        )
        .unwrap();
        assert_eq!(stats.per_head.len(), 2);
        assert_eq!(stats.truth_threshold_positive_cells, 2);
        assert_eq!(stats.anchor_class_stats.positive_count, 2);
        assert!((stats.anchor_class_stats.class_output_sum - 0.6).abs() < 1e-6);
        assert_eq!(stats.anchor_box_positive_count, 2);
        assert_eq!(stats.total_bbox_count, 4);
        for head_stats in &stats.per_head {
            assert_eq!(head_stats.truth_threshold_positive_cells, 1);
            assert_eq!(head_stats.anchor_class_stats.positive_count, 1);
            assert_eq!(head_stats.anchor_class_stats.class_output_sum, 0.3);
            assert_eq!(head_stats.anchor_box_positive_count, 1);
            assert_eq!(head_stats.total_bbox_count, 2);
        }
        for head_delta in deltas {
            assert!((head_delta[6] - 0.6).abs() < 1e-6);
            assert!((head_delta[7] - 0.7).abs() < 1e-6);
            assert!((head_delta[8] + 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn positive_anchor_box_delta_writeback_touches_only_box_channels() {
        let assignment = ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0, 1],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: vec![10.0, 10.0, 20.0, 20.0],
            classes: 1,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: None,
            iou_thresh: 0.2,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
            class_map: None,
            class_multipliers: Some(vec![2.0]),
        };
        let mut head_output = vec![0.0; 16];
        head_output[..6].copy_from_slice(&[0.4, 0.6, 0.1, -0.1, 0.7, 0.5]);
        head_output[8..14].copy_from_slice(&[0.55, 0.45, -0.1, 0.2, 0.6, 0.4]);
        let mut head_delta = vec![0.77; 16];
        head_delta[0..4].copy_from_slice(&[0.1, -0.2, 0.3, -0.4]);
        head_delta[8..12].copy_from_slice(&[-0.1, 0.2, -0.3, 0.4]);
        let delta_config = ObbYoloBoxDeltaConfig {
            score: ObbYoloMatchScoreConfig {
                cobb_normalizer: 1.0,
                ..ObbYoloMatchScoreConfig::default()
            },
            iou_normalizer: 0.5,
            max_delta: 0.03,
            accumulate: true,
        };
        let labels = [vec![truth(0.5, 0.5, 0.2, 0.2)]];
        let applied = apply_obb_yolo_batch_head_positive_box_deltas(
            &labels,
            &head_output,
            &mut head_delta,
            &assignment,
            false,
            delta_config,
        )
        .unwrap();
        assert_eq!(applied, 2);
        for (slot, (anchor_width, anchor_height)) in
            [(10.0, 10.0), (20.0, 20.0)].into_iter().enumerate()
        {
            let base = slot * 8;
            let prediction = std::array::from_fn(|channel| head_output[base + channel]);
            let mut expected_config = delta_config;
            expected_config.iou_normalizer *= 2.0;
            let expected = obb_yolo_box_regression_deltas(
                prediction,
                if slot == 0 {
                    [0.1, -0.2, 0.3, -0.4]
                } else {
                    [-0.1, 0.2, -0.3, 0.4]
                },
                &labels[0][0],
                ObbYoloPredictionMatchContext {
                    anchor_width,
                    anchor_height,
                    cell_x: 0,
                    cell_y: 0,
                    grid_width: 1,
                    grid_height: 1,
                    net_width: 100,
                    net_height: 100,
                    new_coords: false,
                },
                expected_config,
            )
            .unwrap();
            for channel in 0..4 {
                assert!((head_delta[base + channel] - expected[channel]).abs() < 1e-6);
            }
            for channel in 4..8 {
                assert_eq!(head_delta[base + channel], 0.77);
            }
        }
    }

    #[test]
    fn positive_anchor_front_point_delta_writeback_preserves_other_channels() {
        let mut label = truth(0.5, 0.5, 0.2, 0.1);
        label.front_x = 0.66;
        label.front_y = 0.57;
        label.has_front = true;
        let labels = [vec![label]];
        let mut assignment = ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: vec![20.0, 10.0],
            classes: 1,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: Some(vec![[0.4, -0.2]]),
            iou_thresh: 0.1,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
            class_map: None,
            class_multipliers: Some(vec![1.5]),
        };
        let output = [0.5, 0.5, 0.0, 0.0, 0.68, 0.55, 0.4, 0.3];
        let initial = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut deltas = initial;
        let front_config = ObbYoloFrontPointDeltaConfig {
            normalizer: 1.2,
            max_delta: 0.5,
            ..ObbYoloFrontPointDeltaConfig::default()
        };
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 20.0,
            anchor_height: 10.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
        };
        let expected = obb_yolo_front_point_regression_deltas(
            output[..6].try_into().unwrap(),
            &labels[0][0],
            context,
            ObbYoloFrontPointDeltaConfig {
                normalizer: front_config.normalizer * 1.5,
                ..front_config
            },
            true,
        )
        .unwrap();
        assert_eq!(
            apply_obb_yolo_batch_head_positive_front_point_deltas(
                &labels,
                &output,
                &mut deltas,
                &assignment,
                false,
                front_config,
            )
            .unwrap(),
            1
        );
        assert!((deltas[0] - (initial[0] + expected.center_x)).abs() < 1e-6);
        assert!((deltas[1] - (initial[1] + expected.center_y)).abs() < 1e-6);
        assert_eq!(deltas[2], initial[2]);
        assert_eq!(deltas[3], initial[3]);
        assert!((deltas[4] - (initial[4] + expected.front_x)).abs() < 1e-6);
        assert!((deltas[5] - (initial[5] + expected.front_y)).abs() < 1e-6);
        assert_eq!(deltas[6], initial[6]);
        assert_eq!(deltas[7], initial[7]);

        assignment.front_anchor_offsets = None;
        let mut direct_deltas = initial;
        apply_obb_yolo_batch_head_positive_front_point_deltas(
            &labels,
            &output,
            &mut direct_deltas,
            &assignment,
            false,
            front_config,
        )
        .unwrap();
        assert_eq!(direct_deltas[0], initial[0]);
        assert_eq!(direct_deltas[1], initial[1]);
        assert_ne!(direct_deltas[4], initial[4]);
        assert_ne!(direct_deltas[5], initial[5]);
    }

    #[test]
    fn positive_anchor_covariance_box_delta_writeback_preserves_other_channels() {
        let assignment = ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: vec![20.0, 10.0],
            classes: 1,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: None,
            iou_thresh: 1.0,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
            class_map: None,
            class_multipliers: None,
        };
        let head_output = [0.45, 0.55, 0.2, -0.1, 0.7, 0.5, 0.4, 0.9];
        let mut head_delta = [0.1, -0.1, 0.05, -0.05, 0.66, 0.67, 0.68, 0.69];
        let initial = head_delta;
        let delta_config = ObbYoloBoxDeltaConfig {
            score: ObbYoloMatchScoreConfig {
                covariance_normalizer: 1.2,
                ..ObbYoloMatchScoreConfig::default()
            },
            accumulate: false,
            ..ObbYoloBoxDeltaConfig::default()
        };
        let labels = [vec![truth(0.5, 0.5, 0.2, 0.1)]];
        assert_eq!(
            apply_obb_yolo_batch_head_positive_box_deltas(
                &labels,
                &head_output,
                &mut head_delta,
                &assignment,
                false,
                delta_config,
            )
            .unwrap(),
            1
        );
        assert_ne!(&head_delta[..4], &initial[..4]);
        assert_eq!(&head_delta[4..], &initial[4..]);
    }

    #[test]
    fn covariance_box_deltas_match_source_analytic_chain_and_scale() {
        let mut label = truth(0.5, 0.5, 0.2, 0.1);
        label.front_x = 0.68;
        label.front_y = 0.54;
        label.has_front = true;
        let prediction = [0.45, 0.57, 0.1, -0.2, 0.68, 0.54];
        let existing = [0.1, -0.1, 0.05, -0.05];
        let delta_config = ObbYoloBoxDeltaConfig {
            score: ObbYoloMatchScoreConfig {
                covariance_normalizer: 1.4,
                covariance_front_length_ratio_k: 0.7,
                covariance_front_length_weight: 0.4,
                ..ObbYoloMatchScoreConfig::default()
            },
            iou_normalizer: 0.3,
            ..ObbYoloBoxDeltaConfig::default()
        };
        for new_coords in [false, true] {
            let context = ObbYoloPredictionMatchContext {
                anchor_width: 20.0,
                anchor_height: 10.0,
                cell_x: 0,
                cell_y: 0,
                grid_width: 1,
                grid_height: 1,
                net_width: 100,
                net_height: 100,
                new_coords,
            };
            let actual =
                obb_yolo_box_regression_deltas(prediction, existing, &label, context, delta_config)
                    .unwrap();
            let decoded = decode_obb_yolo_prediction(prediction, context).unwrap();
            let cov_w = (decoded.bbox.w - 1e-6).max(1e-6);
            let cov_h = (decoded.bbox.h - 1e-6).max(1e-6);
            let inverse_softplus = |value: f32| {
                if value > 20.0 {
                    value
                } else {
                    value.exp_m1().ln()
                }
            };
            let gradient = crate::obb_geometry::box_covariance_w2_obb_loss_and_gradient(
                decoded.bbox.x,
                decoded.bbox.y,
                inverse_softplus(cov_w),
                inverse_softplus(cov_h),
                decoded.front_x,
                decoded.front_y,
                crate::obb_geometry::BoxObb {
                    bbox: crate::detect::DetectionBox {
                        x: label.x,
                        y: label.y,
                        w: label.w,
                        h: label.h,
                    },
                    front_x: label.front_x,
                    front_y: label.front_y,
                },
                0.7,
                0.4,
            )
            .gradient;
            let sp_w = (-(-cov_w).exp_m1()).max(1e-6);
            let sp_h = (-(-cov_h).exp_m1()).max(1e-6);
            let width_chain = if new_coords {
                8.0 * prediction[2] * context.anchor_width / context.net_width as f32
            } else {
                decoded.bbox.w
            };
            let height_chain = if new_coords {
                8.0 * prediction[3] * context.anchor_height / context.net_height as f32
            } else {
                decoded.bbox.h
            };
            let x_chain = if new_coords {
                1.0
            } else {
                prediction[0] * (1.0 - prediction[0])
            };
            let y_chain = if new_coords {
                1.0
            } else {
                prediction[1] * (1.0 - prediction[1])
            };
            let scale = delta_config.iou_normalizer * 1.4;
            let updates = [
                -gradient.dx * x_chain * scale,
                -gradient.dy * y_chain * scale,
                -(gradient.d_raw_w / sp_w) * width_chain * scale,
                -(gradient.d_raw_h / sp_h) * height_chain * scale,
            ];
            let expected: [f32; 4] =
                std::array::from_fn(|index| existing[index] + updates[index].clamp(-1.0, 1.0));
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn positive_anchor_class_delta_writeback_matches_source_and_preserves_other_entries() {
        let config = ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0, 1],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: vec![10.0, 10.0, 20.0, 20.0],
            classes: 2,
            net_width: 100,
            net_height: 100,
            front_anchor_offsets: None,
            iou_thresh: 0.2,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
            class_map: Some(vec![1, 0]),
            class_multipliers: Some(vec![2.0, 3.0]),
        };
        let mut head_output = vec![0.0; 18];
        head_output[7] = 0.2;
        head_output[8] = 0.3;
        head_output[16] = 0.4;
        head_output[17] = 0.6;
        let mut head_delta = vec![0.0; 18];
        head_delta[16] = 0.4;
        head_delta[17] = 0.1;
        let stats = apply_obb_yolo_batch_head_positive_class_deltas(
            &[vec![truth(0.5, 0.5, 0.2, 0.2)]],
            &head_output,
            &mut head_delta,
            &config,
            false,
            0.1,
            0.5,
        )
        .unwrap();
        assert_eq!(stats.positive_count, 2);
        assert!((stats.class_output_sum - 0.9).abs() < 1e-6);
        assert!((stats.mean_class_output().unwrap() - 0.45).abs() < 1e-6);
        assert!((head_delta[7] + 0.15).abs() < 1e-6);
        assert!((head_delta[8] - 0.975).abs() < 1e-6);
        assert_eq!(head_delta[16], 0.4);
        assert!((head_delta[17] - 1.05).abs() < 1e-6);
        for index in 0..18 {
            if ![7, 8, 16, 17].contains(&index) {
                assert_eq!(head_delta[index], 0.0, "delta entry {index} changed");
            }
        }
    }

    #[test]
    fn positive_anchor_objectness_rejects_bad_mapping() {
        let config = ObbYoloPositiveObjectnessConfig {
            head: ObbYoloHeadAssignmentSpec {
                mask: vec![0],
                grid_width: 1,
                grid_height: 1,
            },
            anchors: vec![10.0, 10.0],
            classes: 1,
            net_width: 10,
            net_height: 10,
            front_anchor_offsets: None,
            iou_thresh: 1.0,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
            class_map: Some(vec![1]),
            class_multipliers: None,
        };
        let head_output = [0.0; 8];
        let mut head_delta = [0.0; 8];
        assert!(
            apply_obb_yolo_batch_head_positive_objectness_deltas(
                &[vec![truth(0.5, 0.5, 0.1, 0.1)]],
                &head_output,
                &mut head_delta,
                &config,
            )
            .is_err()
        );
        assert_eq!(head_delta, [0.0; 8]);
    }

    #[test]
    fn batch_head_tensor_traversal_extracts_source_channels_and_tags_matches() {
        let truths = [
            vec![truth(0.5, 0.5, 0.2, 0.2)],
            vec![truth(1.0, 1.0, 0.2, 0.2)],
        ];
        let config = ObbYoloHeadMatchConfig {
            head_index: 3,
            anchors: vec![10.0, 10.0, 20.0, 20.0],
            mask: vec![1],
            grid_width: 1,
            grid_height: 1,
            classes: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
            front_anchor_offsets: None,
            score_config: ObbYoloMatchScoreConfig {
                covariance_normalizer: 1.0,
                ..ObbYoloMatchScoreConfig::default()
            },
            ignore_thresh: 0.5,
            truth_thresh: 0.99,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        // `[batch][entry][cell]`: box/front, objectness, and one class.
        let head_output = [
            0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.9, 0.9, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.9, 0.9,
        ];
        let mut matches = Vec::new();
        visit_obb_yolo_batch_head_matches(&truths, &head_output, &config, |matched| {
            matches.push(matched);
            Ok(())
        })
        .unwrap();
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].batch_index, 0);
        assert_eq!(matches[0].head_index, 3);
        assert_eq!(matches[0].anchor_index, 1);
        assert_eq!((matches[0].cell_x, matches[0].cell_y), (0, 0));
        assert_eq!(matches[0].truth_match.positive_truth_index, Some(0));
        assert!(matches[0].truth_match.best_score > 0.99);
        assert!((matches[0].objectness_delta - 0.1).abs() < 1e-6);
        assert_eq!(matches[1].batch_index, 1);
        assert_eq!(matches[1].truth_match.positive_truth_index, None);
        assert!(matches[1].truth_match.best_score < 0.99);
    }

    #[test]
    fn batch_head_tensor_traversal_visits_all_cells_and_mask_slots_in_source_order() {
        let config = ObbYoloHeadMatchConfig {
            head_index: 4,
            anchors: vec![10.0, 10.0, 20.0, 20.0],
            mask: vec![1, 0],
            grid_width: 2,
            grid_height: 1,
            classes: 1,
            net_width: 20,
            net_height: 10,
            new_coords: false,
            front_anchor_offsets: None,
            score_config: ObbYoloMatchScoreConfig::default(),
            ignore_thresh: 0.5,
            truth_thresh: 0.5,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        let head_output = vec![0.0; 32];
        let mut matches = Vec::new();
        visit_obb_yolo_batch_head_matches(&[Vec::new()], &head_output, &config, |matched| {
            matches.push(matched);
            Ok(())
        })
        .unwrap();
        let observed: Vec<_> = matches
            .iter()
            .map(|matched| {
                (
                    matched.anchor_index,
                    matched.head_anchor_index,
                    matched.cell_x,
                    matched.cell_y,
                )
            })
            .collect();
        assert_eq!(
            observed,
            [(1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 1, 0)]
        );
        assert!(
            matches
                .iter()
                .all(|matched| matched.truth_match.best_truth_index.is_none())
        );
    }

    #[test]
    fn batch_head_tensor_traversal_rejects_wrong_output_length() {
        let config = ObbYoloHeadMatchConfig {
            head_index: 0,
            anchors: vec![10.0, 10.0],
            mask: vec![0],
            grid_width: 1,
            grid_height: 1,
            classes: 1,
            net_width: 10,
            net_height: 10,
            new_coords: false,
            front_anchor_offsets: None,
            score_config: ObbYoloMatchScoreConfig::default(),
            ignore_thresh: 0.5,
            truth_thresh: 0.5,
            objectness_normalizer: 1.0,
            objectness_smooth: false,
        };
        assert!(
            visit_obb_yolo_batch_head_matches(&[Vec::new()], &[0.0; 7], &config, |_| Ok(()))
                .is_err()
        );
    }

    #[test]
    fn positive_enumerator_preserves_truth_order_across_batch_slots() {
        let truths = [truth(0.25, 0.25, 0.4, 0.4), truth(0.75, 0.75, 0.1, 0.1)];
        let assignments = enumerate_obb_yolo_head_positives(
            &truths,
            1,
            &[40.0, 40.0, 30.0, 40.0, 10.0, 10.0],
            &[2, 0, 1],
            100,
            100,
            4,
            4,
            1.0,
        )
        .expect("truth order");
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].truth_index, 0);
        assert_eq!(assignments[0].assignment.anchor_index, 0);
        assert_eq!(assignments[0].assignment.cell_x, 1);
        assert_eq!(assignments[0].assignment.cell_y, 1);
        assert_eq!(assignments[1].truth_index, 1);
        assert_eq!(assignments[1].assignment.anchor_index, 2);
        assert_eq!(assignments[1].assignment.head_anchor_index, 0);
        assert_eq!(assignments[1].assignment.cell_x, 3);
        assert_eq!(assignments[1].assignment.cell_y, 3);
    }

    #[test]
    fn positive_enumerator_keeps_masked_extras_when_best_anchor_is_elsewhere() {
        let label = truth(0.5, 0.5, 0.4, 0.4);
        let assignments = enumerate_obb_yolo_head_positives(
            &[label],
            1,
            &[40.0, 40.0, 30.0, 40.0, 10.0, 10.0],
            &[1, 2],
            100,
            100,
            5,
            5,
            0.7,
        )
        .expect("additional match without the global best");
        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].kind, ObbYoloPositiveKind::AdditionalAnchor);
        assert_eq!(assignments[0].assignment.anchor_index, 1);
        assert_eq!(assignments[0].assignment.head_anchor_index, 0);
    }

    #[test]
    fn positive_enumerator_uses_strict_threshold_and_default_disables_extras() {
        let label = truth(0.5, 0.5, 0.4, 0.4);
        let args = (
            &[label][..],
            1,
            &[40.0, 40.0, 30.0, 40.0, 10.0, 10.0][..],
            &[0, 1, 2][..],
            100,
            100,
            5,
            5,
        );
        let boundary = obb_yolo_anchor_match_score(&label, 30.0, 40.0, 100, 100)
            .expect("matching anchor score");
        let at_boundary = enumerate_obb_yolo_head_positives(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, boundary,
        )
        .expect("strict threshold");
        assert_eq!(at_boundary.len(), 1);
        assert_eq!(at_boundary[0].kind, ObbYoloPositiveKind::BestAnchor);
        let default_threshold = enumerate_obb_yolo_head_positives(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, 1.0,
        )
        .expect("source default threshold");
        assert_eq!(default_threshold, at_boundary);
    }

    #[test]
    fn positive_enumerator_validates_empty_input_configuration_and_truth_classes() {
        let anchors = [40.0, 40.0];
        assert!(
            enumerate_obb_yolo_head_positives(&[], 1, &anchors, &[0], 100, 100, 5, 5, 1.0)
                .unwrap()
                .is_empty()
        );
        assert!(
            enumerate_obb_yolo_head_positives(&[], 0, &anchors, &[0], 100, 100, 5, 5, 1.0).is_err()
        );
        assert!(
            enumerate_obb_yolo_head_positives(&[], 1, &anchors, &[0], 100, 100, 5, 5, f32::NAN)
                .is_err()
        );
        assert!(
            enumerate_obb_yolo_head_positives(&[], 1, &[0.0, 40.0], &[0], 100, 100, 5, 5, 1.0)
                .is_err()
        );
        assert!(
            enumerate_obb_yolo_head_positives(&[], 1, &anchors, &[], 100, 100, 5, 5, 1.0).is_err()
        );

        let mut invalid_class = truth(0.5, 0.5, 0.2, 0.2);
        invalid_class.class_id = 1;
        assert!(
            enumerate_obb_yolo_head_positives(
                &[invalid_class],
                1,
                &anchors,
                &[0],
                100,
                100,
                5,
                5,
                1.0
            )
            .is_err()
        );
    }

    #[test]
    fn cell_truth_match_uses_first_best_truth_and_any_confident_class_for_ignore() {
        let scores = [0.2, 0.8, 0.8];
        let at_class_boundary = select_obb_yolo_cell_truth_match(&scores, &[0.2, 0.25], 0.5, 0.7)
            .expect("class confidence boundary");
        assert_eq!(
            at_class_boundary,
            ObbYoloCellTruthMatch {
                best_truth_index: Some(1),
                best_score: 0.8,
                positive_truth_index: Some(1),
                ignore_match_score: 0.0,
                ignore_match: false,
            }
        );

        let confident = select_obb_yolo_cell_truth_match(&scores, &[0.2501], 0.5, 0.7)
            .expect("confident class output");
        assert_eq!(confident.best_truth_index, Some(1));
        assert_eq!(confident.positive_truth_index, Some(1));
        assert_eq!(confident.ignore_match_score, 0.8);
        assert!(confident.ignore_match);
    }

    #[test]
    fn cell_truth_match_uses_strict_ignore_and_positive_thresholds() {
        let selected =
            select_obb_yolo_cell_truth_match(&[0.5], &[0.3], 0.5, 0.5).expect("threshold equality");
        assert_eq!(selected.best_truth_index, Some(0));
        assert_eq!(selected.best_score, 0.5);
        assert_eq!(selected.positive_truth_index, None);
        assert_eq!(selected.ignore_match_score, 0.5);
        assert!(!selected.ignore_match);
    }

    #[test]
    fn covariance_prediction_match_decodes_channels_and_scores_absent_front_truths() {
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 200.0,
            anchor_height: 400.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
        };
        let mut exact = truth(0.5, 0.5, 2.0, 4.0);
        exact.front_x = 0.5;
        exact.front_y = 4.5;
        exact.has_front = true;
        let prediction = [0.5, 0.5, 0.0, 0.0, 0.5, 4.5];
        assert_eq!(
            obb_yolo_covariance_prediction_match_score(prediction, &exact, context, 1.0, 1.0, 0.35)
                .unwrap(),
            1.0
        );

        let mut rotated = exact;
        rotated.front_x = 4.5;
        rotated.front_y = 0.5;
        let rotated_score = obb_yolo_covariance_prediction_match_score(
            prediction, &rotated, context, 1.0, 1.0, 0.0,
        )
        .unwrap();
        assert!(rotated_score < 1.0);
        let blended_score = obb_yolo_covariance_prediction_match_score_with_viou(
            prediction, &rotated, context, 1.0, 1.0, 0.0, 0.35,
        )
        .unwrap();
        let prediction_geometry = crate::obb_geometry::BoxObb {
            bbox: crate::detect::DetectionBox {
                x: 0.5,
                y: 0.5,
                w: 2.0,
                h: 4.0,
            },
            front_x: 0.5,
            front_y: 4.5,
        };
        let rotated_geometry = crate::obb_geometry::BoxObb {
            bbox: prediction_geometry.bbox,
            front_x: 4.5,
            front_y: 0.5,
        };
        assert_eq!(
            blended_score,
            crate::obb_geometry::blend_obb_score_with_qbbv_viou(
                rotated_score,
                prediction_geometry,
                rotated_geometry,
                0.35,
            )
        );

        let direction_free = truth(0.5, 0.5, 2.0, 4.0);
        let direction_free_score = obb_yolo_covariance_prediction_match_score(
            prediction,
            &direction_free,
            context,
            1.0,
            1.0,
            0.35,
        )
        .unwrap();
        assert!(direction_free_score.is_finite());
        assert!((0.0..=1.0).contains(&direction_free_score));
        assert!(obb_yolo_covariance_prediction_match_score(
            prediction,
            &exact,
            context,
            0.0,
            1.0,
            0.35,
        )
        .is_err());
    }

    #[test]
    fn configured_cell_matching_scores_all_truths_and_honors_empty_sentinel() {
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 2.0,
            anchor_height: 4.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 1,
            net_height: 1,
            new_coords: false,
        };
        let mut matched = truth(0.5, 0.5, 2.0, 4.0);
        matched.front_x = 0.5;
        matched.front_y = 4.5;
        matched.has_front = true;
        let mut later = matched;
        later.front_x = 4.5;
        later.front_y = 0.5;
        let empty = truth(0.0, 0.5, 2.0, 4.0);
        let config = ObbYoloMatchScoreConfig {
            covariance_normalizer: 1.0,
            viou_weight: 0.0,
            ..ObbYoloMatchScoreConfig::default()
        };
        let matched_cell = match_obb_yolo_cell_prediction_with_score_config(
            [0.5, 0.5, 0.0, 0.0, 0.5, 4.5],
            &[matched, later],
            &[0.9],
            context,
            config,
            0.1,
            0.5,
        )
        .unwrap();
        assert_eq!(matched_cell.positive_truth_index, Some(0));
        assert_eq!(matched_cell.best_score, 1.0);

        let after_empty = match_obb_yolo_cell_prediction_with_score_config(
            [0.5, 0.5, 0.0, 0.0, 0.5, 4.5],
            &[empty, matched],
            &[0.9],
            context,
            config,
            0.1,
            0.5,
        )
        .unwrap();
        assert_eq!(after_empty.best_truth_index, None);
        assert_eq!(after_empty.best_score, 0.0);
    }

    #[test]
    fn match_score_dispatch_obeys_covariance_cobb_angular_priority() {
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 2.0,
            anchor_height: 4.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 1,
            net_height: 1,
            new_coords: false,
        };
        let mut label = truth(0.5, 0.5, 2.0, 4.0);
        label.front_x = 0.5;
        label.front_y = 4.5;
        label.has_front = true;
        let prediction = [0.5, 0.5, 0.0, 0.0, 4.5, 0.5];

        let covariance_config = ObbYoloMatchScoreConfig {
            covariance_normalizer: 1.0,
            cobb_normalizer: 1.0,
            viou_weight: 0.2,
            ..ObbYoloMatchScoreConfig::default()
        };
        let covariance =
            obb_yolo_match_score(prediction, &label, context, covariance_config).unwrap();
        assert_eq!(covariance.branch, ObbYoloMatchScoreBranch::Covariance);
        assert_eq!(
            covariance.score,
            obb_yolo_covariance_prediction_match_score_with_viou(
                prediction, &label, context, 1.0, 1.0, 0.35, 0.2,
            )
            .unwrap()
        );

        let cobb_config = ObbYoloMatchScoreConfig {
            covariance_normalizer: 0.0,
            cobb_normalizer: 1.0,
            viou_weight: 0.2,
            ..ObbYoloMatchScoreConfig::default()
        };
        let cobb = obb_yolo_match_score(prediction, &label, context, cobb_config).unwrap();
        assert_eq!(cobb.branch, ObbYoloMatchScoreBranch::Cobb);
        assert_eq!(
            cobb.score,
            obb_yolo_cobb_prediction_match_score_with_viou(
                prediction,
                &label,
                context,
                0.0,
                1.0,
                crate::obb_geometry::ObbCobbLossConfig::default(),
                0.2,
            )
            .unwrap()
        );

        let angular_config = ObbYoloMatchScoreConfig {
            viou_weight: 0.2,
            ..ObbYoloMatchScoreConfig::default()
        };
        let angular = obb_yolo_match_score(prediction, &label, context, angular_config).unwrap();
        assert_eq!(angular.branch, ObbYoloMatchScoreBranch::LegacyAngular);
        assert_eq!(
            angular.score,
            obb_yolo_angular_prediction_match_score_with_viou(prediction, &label, context, 0.2,)
                .unwrap()
        );

        let nonfinite_covariance = ObbYoloMatchScoreConfig {
            covariance_normalizer: f32::NAN,
            cobb_normalizer: 1.0,
            ..ObbYoloMatchScoreConfig::default()
        };
        assert_eq!(
            obb_yolo_match_score(prediction, &label, context, nonfinite_covariance)
                .unwrap()
                .branch,
            ObbYoloMatchScoreBranch::Cobb
        );
    }

    #[test]
    fn angular_prediction_match_decodes_and_blends_with_qbbv_viou() {
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 2.0,
            anchor_height: 4.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 1,
            net_height: 1,
            new_coords: false,
        };
        let mut label = truth(0.5, 0.5, 2.0, 4.0);
        label.front_x = 0.5;
        label.front_y = 4.5;
        label.has_front = true;
        let prediction = [0.5, 0.5, 0.0, 0.0, 4.5, 0.5];
        let score =
            obb_yolo_angular_prediction_match_score_with_viou(prediction, &label, context, 0.35)
                .unwrap();

        let prediction_geometry = crate::obb_geometry::BoxObb {
            bbox: crate::detect::DetectionBox {
                x: 0.5,
                y: 0.5,
                w: 2.0,
                h: 4.0,
            },
            front_x: 4.5,
            front_y: 0.5,
        };
        let truth_geometry = crate::obb_geometry::BoxObb {
            bbox: prediction_geometry.bbox,
            front_x: 0.5,
            front_y: 4.5,
        };
        let base_score = crate::obb_geometry::box_ciou_angular_quality_score(
            prediction_geometry,
            truth_geometry,
        );
        assert!(base_score < 1.0);
        assert_eq!(
            score,
            crate::obb_geometry::blend_obb_score_with_qbbv_viou(
                base_score,
                prediction_geometry,
                truth_geometry,
                0.35,
            )
        );

        let direction_free = truth(0.5, 0.5, 2.0, 4.0);
        assert_eq!(
            obb_yolo_angular_prediction_match_score_with_viou(
                [0.5, 0.5, 0.0, 0.0, 4.5, 0.5],
                &direction_free,
                context,
                0.0,
            )
            .unwrap(),
            0.0
        );
        assert_eq!(
            obb_yolo_angular_prediction_match_score_with_viou(
                [0.5, 0.5, 0.0, 0.0, 4.5, 0.5],
                &direction_free,
                context,
                1.0,
            )
            .unwrap(),
            1.0
        );
    }

    #[test]
    fn cobb_prediction_match_decodes_branch_gates_and_qbbv_blend() {
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 2.0,
            anchor_height: 4.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 1,
            net_height: 1,
            new_coords: false,
        };
        let mut label = truth(0.5, 0.5, 2.0, 4.0);
        label.front_x = 0.5;
        label.front_y = 4.5;
        label.has_front = true;
        let config = crate::obb_geometry::ObbCobbLossConfig::default();
        let exact = [0.5, 0.5, 0.0, 0.0, 0.5, 4.5];
        assert_eq!(
            obb_yolo_cobb_prediction_match_score_with_viou(
                exact, &label, context, 0.0, 1.0, config, 0.2,
            )
            .unwrap(),
            1.0
        );

        let rotated = [0.5, 0.5, 0.0, 0.0, 4.5, 0.5];
        let base = obb_yolo_cobb_prediction_match_score_with_viou(
            rotated, &label, context, 0.0, 1.0, config, 0.0,
        )
        .unwrap();
        assert!(base < 1.0);
        let blended = obb_yolo_cobb_prediction_match_score_with_viou(
            rotated, &label, context, 0.0, 1.0, config, 0.35,
        )
        .unwrap();
        let prediction_geometry = crate::obb_geometry::BoxObb {
            bbox: crate::detect::DetectionBox {
                x: 0.5,
                y: 0.5,
                w: 2.0,
                h: 4.0,
            },
            front_x: 4.5,
            front_y: 0.5,
        };
        let truth_geometry = crate::obb_geometry::BoxObb {
            bbox: prediction_geometry.bbox,
            front_x: 0.5,
            front_y: 4.5,
        };
        assert_eq!(
            blended,
            crate::obb_geometry::box_cobb_obb_quality_score_with_viou(
                prediction_geometry,
                truth_geometry,
                config,
                0.35,
            )
        );
        assert!(
            obb_yolo_cobb_prediction_match_score_with_viou(
                exact, &label, context, 1.0, 1.0, config, 0.2,
            )
            .is_err()
        );
        assert!(
            obb_yolo_cobb_prediction_match_score_with_viou(
                exact, &label, context, 0.0, 0.0, config, 0.2,
            )
            .is_err()
        );
    }

    #[test]
    fn ciou_prediction_match_score_decodes_legacy_and_new_coordinate_rows() {
        let mut label = truth(0.5, 0.5, 0.25, 0.25);
        label.front_x = 0.5;
        label.front_y = 0.75;
        label.has_front = true;
        assert_eq!(
            obb_yolo_ciou_prediction_match_score(
                [0.5, 0.5, 0.0, 0.0, 0.5, 0.75],
                &label,
                25.0,
                25.0,
                0,
                0,
                1,
                1,
                100,
                100,
                false,
            )
            .unwrap(),
            1.0
        );
        assert_eq!(
            obb_yolo_ciou_prediction_match_score(
                [0.5, 0.5, 0.5, 0.5, 0.5, 0.75],
                &label,
                25.0,
                25.0,
                0,
                0,
                1,
                1,
                100,
                100,
                true,
            )
            .unwrap(),
            1.0
        );
    }

    #[test]
    fn ciou_prediction_match_score_preserves_cell_offsets_and_absent_front_behavior() {
        let mut label = truth(0.5, 0.5, 0.25, 0.25);
        label.front_x = 0.5;
        label.front_y = 0.75;
        label.has_front = true;
        assert_eq!(
            obb_yolo_ciou_prediction_match_score(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                &label,
                25.0,
                25.0,
                1,
                1,
                2,
                2,
                100,
                100,
                false,
            )
            .unwrap(),
            1.0
        );
        assert_eq!(
            obb_yolo_ciou_prediction_match_score(
                [0.0, 0.0, 0.0, 0.0, f32::NAN, 0.5],
                &label,
                25.0,
                25.0,
                1,
                1,
                2,
                2,
                100,
                100,
                false,
            )
            .unwrap(),
            1.0
        );
        label.has_front = false;
        assert_eq!(
            obb_yolo_ciou_prediction_match_score(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                &label,
                25.0,
                25.0,
                1,
                1,
                2,
                2,
                100,
                100,
                false,
            )
            .unwrap(),
            0.0
        );
    }

    #[test]
    fn ciou_prediction_match_score_rejects_invalid_shape_and_anchors() {
        let label = truth(0.5, 0.5, 0.25, 0.25);
        assert!(
            obb_yolo_ciou_prediction_match_score(
                [0.5; 6], &label, 25.0, 25.0, 0, 0, 0, 1, 100, 100, false,
            )
            .is_err()
        );
        assert!(
            obb_yolo_ciou_prediction_match_score(
                [0.5; 6],
                &label,
                f32::NAN,
                25.0,
                0,
                0,
                1,
                1,
                100,
                100,
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn cell_prediction_match_scores_truths_before_applying_class_ignore_gate() {
        let mut exact = truth(0.5, 0.5, 0.25, 0.25);
        exact.front_x = 0.5;
        exact.front_y = 0.75;
        exact.has_front = true;
        let mut distant = truth(0.1, 0.1, 0.25, 0.25);
        distant.front_x = 0.1;
        distant.front_y = 0.35;
        distant.has_front = true;
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 25.0,
            anchor_height: 25.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
        };

        let not_confident = match_obb_yolo_cell_prediction(
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.75],
            &[distant, exact],
            &[0.25],
            context,
            0.5,
            0.9,
        )
        .unwrap();
        assert_eq!(not_confident.best_truth_index, Some(1));
        assert_eq!(not_confident.best_score, 1.0);
        assert_eq!(not_confident.positive_truth_index, Some(1));
        assert_eq!(not_confident.ignore_match_score, 0.0);
        assert!(!not_confident.ignore_match);

        let confident = match_obb_yolo_cell_prediction(
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.75],
            &[distant, exact],
            &[0.2501],
            context,
            0.5,
            0.9,
        )
        .unwrap();
        assert_eq!(confident.best_truth_index, Some(1));
        assert_eq!(confident.ignore_match_score, 1.0);
        assert!(confident.ignore_match);
    }

    #[test]
    fn cell_prediction_match_keeps_first_ties_and_stops_at_empty_truth_sentinel() {
        let mut exact = truth(0.5, 0.5, 0.25, 0.25);
        exact.front_x = 0.5;
        exact.front_y = 0.75;
        exact.has_front = true;
        let empty = truth(0.0, 0.5, 0.25, 0.25);
        let context = ObbYoloPredictionMatchContext {
            anchor_width: 25.0,
            anchor_height: 25.0,
            cell_x: 0,
            cell_y: 0,
            grid_width: 1,
            grid_height: 1,
            net_width: 100,
            net_height: 100,
            new_coords: false,
        };

        let tied = match_obb_yolo_cell_prediction(
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.75],
            &[exact, exact],
            &[0.3],
            context,
            0.5,
            1.0,
        )
        .unwrap();
        assert_eq!(tied.best_truth_index, Some(0));
        assert_eq!(tied.positive_truth_index, None);

        let after_sentinel = match_obb_yolo_cell_prediction(
            [0.5, 0.5, 0.0, 0.0, 0.5, 0.75],
            &[empty, exact],
            &[0.3],
            context,
            0.5,
            0.0,
        )
        .unwrap();
        assert_eq!(after_sentinel.best_truth_index, None);
        assert_eq!(after_sentinel.positive_truth_index, None);
    }

    #[test]
    fn class_delta_existing_positive_branch_updates_only_target_without_cls_normalizer() {
        let result = obb_yolo_class_deltas(
            &[0.2, 0.7, 0.1],
            &[0.11, -0.5, -0.4],
            1,
            true,
            0.2,
            Some(&[10.0, 2.0, 10.0]),
            3.0,
        )
        .unwrap();
        assert_eq!(result.deltas[0], 0.11);
        assert!((result.deltas[1] - 0.4).abs() < 1e-6);
        assert_eq!(result.deltas[2], -0.4);
        assert_eq!(result.class_output_for_average, 0.7);
    }

    #[test]
    fn class_delta_non_focal_branch_smooths_and_scales_only_target() {
        let result = obb_yolo_class_deltas(
            &[0.2, 0.7, 0.4],
            &[0.0; 3],
            1,
            false,
            0.2,
            Some(&[1.0, 2.0, 1.0]),
            1.5,
        )
        .unwrap();
        assert!((result.deltas[0] - (-0.1)).abs() < 1e-6);
        assert!((result.deltas[1] - 0.6).abs() < 1e-6);
        assert!((result.deltas[2] - (-0.3)).abs() < 1e-6);

        let unweighted =
            obb_yolo_class_deltas(&[0.2, 0.7], &[0.0; 2], 1, false, 0.0, None, 9.0).unwrap();
        assert!((unweighted.deltas[0] - (-0.2)).abs() < 1e-6);
        assert!((unweighted.deltas[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn class_delta_focal_branch_ignores_smoothing_and_multipliers() {
        let outputs = [0.2, 0.7];
        let result =
            obb_yolo_class_deltas(&outputs, &[0.0; 2], 1, true, 0.8, Some(&[11.0, 13.0]), 99.0)
                .unwrap();
        let pt = outputs[1] + 1.0e-15;
        let grad = -(1.0 - pt) * (2.0 * pt * pt.ln() + pt - 1.0);
        assert!((result.deltas[0] - (-outputs[0] * 0.5 * grad)).abs() < 1e-6);
        assert!((result.deltas[1] - ((1.0 - outputs[1]) * 0.5 * grad)).abs() < 1e-6);
        assert_eq!(result.class_output_for_average, outputs[1]);

        let invalid_probability =
            obb_yolo_class_deltas(&[0.2, -1.0], &[0.3, 0.0], 1, true, 0.0, None, 1.0).unwrap();
        assert_eq!(invalid_probability.deltas, [0.3, 0.0]);
    }

    #[test]
    fn class_delta_rejects_bad_shapes_classes_and_nonfinite_config() {
        assert!(obb_yolo_class_deltas(&[], &[], 0, false, 0.0, None, 1.0).is_err());
        assert!(obb_yolo_class_deltas(&[0.2], &[0.0], 1, false, 0.0, None, 1.0).is_err());
        assert!(obb_yolo_class_deltas(&[0.2], &[0.0], 0, false, 0.0, Some(&[]), 1.0).is_err());
        assert!(obb_yolo_class_deltas(&[0.2], &[0.0], 0, false, f32::NAN, None, 1.0).is_err());
    }

    #[test]
    fn cell_objectness_delta_matches_negative_ignore_and_positive_branches() {
        let negative = select_obb_yolo_cell_truth_match(&[], &[0.1], 0.5, 0.7).unwrap();
        assert_eq!(
            obb_yolo_cell_objectness_delta(negative, 0.2, 3.0, false).unwrap(),
            -0.6
        );

        let ignored = select_obb_yolo_cell_truth_match(&[0.6], &[0.3], 0.5, 0.9).unwrap();
        assert!(ignored.ignore_match);
        assert_eq!(
            obb_yolo_cell_objectness_delta(ignored, 0.7, 2.0, false).unwrap(),
            0.0
        );
        assert!(
            (obb_yolo_cell_objectness_delta(ignored, 0.7, 2.0, true).unwrap() + 0.2).abs() < 1e-6
        );
        assert!(
            (obb_yolo_cell_objectness_delta(ignored, 0.3, 2.0, true).unwrap() - 0.6).abs() < 1e-6
        );

        let positive = select_obb_yolo_cell_truth_match(&[0.8], &[0.3], 0.5, 0.7).unwrap();
        assert!(positive.ignore_match);
        assert!(
            (obb_yolo_cell_objectness_delta(positive, 0.4, 2.0, true).unwrap() - 0.48).abs() < 1e-6
        );
        assert!(
            (obb_yolo_cell_objectness_delta(positive, 0.4, 2.0, false).unwrap() - 1.2).abs() < 1e-6
        );
    }

    #[test]
    fn cell_objectness_delta_sanitizes_nonfinite_output_and_rejects_invalid_scores() {
        let negative = select_obb_yolo_cell_truth_match(&[], &[0.1], 0.5, 0.7).unwrap();
        assert_eq!(
            obb_yolo_cell_objectness_delta(negative, f32::NAN, 2.0, false).unwrap(),
            0.0
        );
        assert_eq!(
            obb_yolo_cell_objectness_delta(negative, -f32::MAX, f32::MAX, false).unwrap(),
            0.0
        );
        assert!(obb_yolo_cell_objectness_delta(negative, 0.1, f32::INFINITY, false).is_err());

        let invalid_match = ObbYoloCellTruthMatch {
            best_truth_index: None,
            best_score: 1.1,
            positive_truth_index: None,
            ignore_match_score: 0.0,
            ignore_match: false,
        };
        assert!(obb_yolo_cell_objectness_delta(invalid_match, 0.1, 1.0, false).is_err());
    }

    #[test]
    fn cell_truth_match_handles_empty_truths_without_fabricating_a_positive() {
        assert_eq!(
            select_obb_yolo_cell_truth_match(&[], &[0.9], 0.5, -1.0)
                .expect("empty truth candidates"),
            ObbYoloCellTruthMatch {
                best_truth_index: None,
                best_score: 0.0,
                positive_truth_index: None,
                ignore_match_score: 0.0,
                ignore_match: false,
            }
        );
    }

    #[test]
    fn cell_truth_match_rejects_non_finite_or_out_of_range_inputs() {
        assert!(select_obb_yolo_cell_truth_match(&[f32::NAN], &[0.5], 0.5, 0.5).is_err());
        assert!(select_obb_yolo_cell_truth_match(&[1.1], &[0.5], 0.5, 0.5).is_err());
        assert!(select_obb_yolo_cell_truth_match(&[0.5], &[], 0.5, 0.5).is_err());
        assert!(select_obb_yolo_cell_truth_match(&[0.5], &[f32::INFINITY], 0.5, 0.5).is_err());
        assert!(select_obb_yolo_cell_truth_match(&[0.5], &[0.5], f32::NAN, 0.5).is_err());
    }

    #[test]
    fn direction_free_match_score_is_size_iou_for_absent_and_explicit_fronts() {
        let absent = truth(0.5, 0.5, 0.25, 0.25);
        assert_eq!(
            obb_yolo_anchor_match_score(&absent, 160.0, 80.0, 640, 320).unwrap(),
            1.0
        );
        assert_eq!(
            obb_yolo_anchor_match_score(&absent, 40.0, 80.0, 640, 320).unwrap(),
            0.25
        );

        let mut explicit = absent;
        explicit.front_x = 0.8;
        explicit.front_y = 0.2;
        explicit.has_front = true;
        assert_eq!(
            obb_yolo_anchor_match_score(&explicit, 160.0, 80.0, 640, 320).unwrap(),
            1.0
        );
    }

    #[test]
    fn assignment_selects_global_best_before_checking_head_mask_and_clamps_cells() {
        let label = truth(1.0, 0.0, 0.25, 0.25);
        let anchors = [40.0, 80.0, 160.0, 80.0, 160.0, 160.0];
        assert_eq!(
            assign_obb_yolo_anchor(&label, &anchors, &[0, 2], 640, 320, 13, 7).unwrap(),
            None,
            "the head must not fall back to its best masked anchor"
        );
        assert_eq!(
            assign_obb_yolo_anchor(&label, &anchors, &[2, 1], 640, 320, 13, 7).unwrap(),
            Some(ObbYoloAnchorAssignment {
                anchor_index: 1,
                head_anchor_index: 1,
                cell_x: 12,
                cell_y: 0,
                match_score: 1.0,
            })
        );
    }

    #[test]
    fn assignment_keeps_first_global_anchor_on_ties_and_reports_mask_slot() {
        let label = truth(0.5, 0.5, 0.25, 0.25);
        let assignment = assign_obb_yolo_anchor(
            &label,
            &[160.0, 80.0, 160.0, 80.0],
            &[1, 0],
            640,
            320,
            13,
            7,
        )
        .unwrap()
        .unwrap();
        assert_eq!(assignment.anchor_index, 0);
        assert_eq!(assignment.head_anchor_index, 1);
        assert_eq!((assignment.cell_x, assignment.cell_y), (6, 3));
    }

    #[test]
    fn invalid_truth_anchors_network_and_grid_dimensions_are_rejected() {
        let label = truth(0.5, 0.5, 0.25, 0.25);
        assert!(obb_yolo_anchor_match_score(&label, 1.0, 1.0, 0, 1).is_err());
        assert!(obb_yolo_anchor_match_score(&label, f32::NAN, 1.0, 1, 1).is_err());
        assert!(assign_obb_yolo_anchor(&label, &[1.0], &[0], 1, 1, 1, 1).is_err());
        assert!(assign_obb_yolo_anchor(&label, &[1.0, 1.0], &[1], 1, 1, 1, 1).is_err());
        assert!(assign_obb_yolo_anchor(&label, &[1.0, 1.0], &[0], 0, 1, 1, 1).is_err());
        assert!(assign_obb_yolo_anchor(&label, &[1.0, 1.0], &[0], 1, 1, 0, 1).is_err());

        let mut invalid_truth = label;
        invalid_truth.x = f32::INFINITY;
        assert!(obb_yolo_anchor_match_score(&invalid_truth, 1.0, 1.0, 1, 1).is_err());
    }
}

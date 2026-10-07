//! Source-derived training primitives for legacy front-point OBB YOLO heads.
//!
//! This module currently covers direction-free anchor matching, per-truth
//! head-positive enumeration, per-cell OBB score-branch dispatch/match selection,
//! objectness and class-channel deltas, and source-layout truth-row packing.
//! It does not implement complete truth assignment, YOLO loss, backward
//! propagation, or optimizer integration.

use crate::detect::DetectionBox;
use crate::error::{Error, Result};
use crate::obb_geometry::{
    BoxObb, ObbCobbLossConfig, blend_obb_score_with_qbbv_viou, box_ciou_angular_quality_score,
    box_ciou_obb_quality_score, box_cobb_obb_quality_score_with_viou,
    box_covariance_w2_obb_quality_score,
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
    let mut scores = Vec::new();
    scores
        .try_reserve_exact(truths.len())
        .map_err(|_| Error::InvalidConfig("OBB cell truth-score allocation failed".to_owned()))?;
    for truth in truths {
        if truth.x == 0.0 {
            break;
        }
        scores.push(obb_yolo_match_score(prediction, truth, context, score_config)?.score);
    }
    select_obb_yolo_cell_truth_match(&scores, class_outputs, ignore_thresh, truth_thresh)
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
        OBB_YOLO_TRUTH_SIZE, ObbYoloAnchorAssignment, ObbYoloCellTruthMatch,
        ObbYoloMatchScoreBranch, ObbYoloMatchScoreConfig, ObbYoloPositiveKind,
        ObbYoloPredictionMatchContext, assign_obb_yolo_anchor, encode_obb_yolo_truth_batch,
        enumerate_obb_yolo_head_positives, match_obb_yolo_cell_prediction,
        match_obb_yolo_cell_prediction_with_score_config, obb_yolo_anchor_match_score,
        obb_yolo_angular_prediction_match_score_with_viou, obb_yolo_cell_objectness_delta,
        obb_yolo_ciou_prediction_match_score, obb_yolo_class_deltas,
        obb_yolo_cobb_prediction_match_score_with_viou, obb_yolo_covariance_prediction_match_score,
        obb_yolo_covariance_prediction_match_score_with_viou, obb_yolo_match_score,
        select_obb_yolo_cell_truth_match,
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

//! Source-derived training primitives for legacy front-point OBB YOLO heads.
//!
//! This module currently covers direction-free anchor matching, per-truth
//! head-positive enumeration, per-cell match selection and its objectness
//! delta, cell selection, and source-layout truth-row packing only. It does not
//! implement complete truth assignment, YOLO loss, backward propagation, or
//! optimizer integration.

use crate::error::{Error, Result};
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
        OBB_YOLO_TRUTH_SIZE, ObbYoloAnchorAssignment, ObbYoloCellTruthMatch, ObbYoloPositiveKind,
        assign_obb_yolo_anchor, encode_obb_yolo_truth_batch, enumerate_obb_yolo_head_positives,
        obb_yolo_anchor_match_score, obb_yolo_cell_objectness_delta,
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

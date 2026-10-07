//! Source-derived training primitives for legacy front-point OBB YOLO heads.
//!
//! This module currently covers direction-free anchor matching, cell selection,
//! and source-layout truth-row packing only. It does not implement complete
//! truth assignment, YOLO loss, backward propagation, or optimizer integration.

use crate::error::{Error, Result};
use crate::obb_labels::ObbLabel;

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
    if grid_width == 0 || grid_height == 0 {
        return Err(Error::InvalidConfig(
            "OBB anchor assignment requires a nonzero grid".to_owned(),
        ));
    }

    let mut best_anchor = 0;
    let mut best_score = 0.0;
    for (anchor_index, pair) in anchors.chunks_exact(2).enumerate() {
        let score = obb_yolo_anchor_match_score(truth, pair[0], pair[1], net_width, net_height)?;
        if score > best_score {
            best_score = score;
            best_anchor = anchor_index;
        }
    }

    let Some(head_anchor_index) = mask.iter().position(|&index| index == best_anchor) else {
        return Ok(None);
    };
    let cell_x = ((truth.x * grid_width as f32) as usize).min(grid_width - 1);
    let cell_y = ((truth.y * grid_height as f32) as usize).min(grid_height - 1);
    Ok(Some(ObbYoloAnchorAssignment {
        anchor_index: best_anchor,
        head_anchor_index,
        cell_x,
        cell_y,
        match_score: best_score,
    }))
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
        OBB_YOLO_TRUTH_SIZE, ObbYoloAnchorAssignment, assign_obb_yolo_anchor,
        encode_obb_yolo_truth_batch, obb_yolo_anchor_match_score,
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

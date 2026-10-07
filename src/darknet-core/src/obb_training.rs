//! Source-derived training primitives for legacy front-point OBB YOLO heads.
//!
//! This module currently covers direction-free anchor matching and cell
//! selection only. It does not implement complete truth assignment, YOLO loss,
//! backward propagation, or optimizer integration.

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
            "OBB anchor matching requires finite truth coordinates, a normalized center, and positive dimensions".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ObbYoloAnchorAssignment, assign_obb_yolo_anchor, obb_yolo_anchor_match_score};
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

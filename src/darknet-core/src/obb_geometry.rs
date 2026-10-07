//! Source-derived oriented bounding-box geometry for the legacy front-point OBB contract.
//!
//! This module provides geometry for OBB decoding/evaluation and the source
//! CIoU-OBB match-quality score. It does not implement a detector layer,
//! rotated-label loading, training loss, NMS dispatch, or CARPN's explicit
//! four-vertex polygon contract.

use crate::detect::DetectionBox;

const FRONT_MIN_LENGTH: f32 = 1e-4;

type Point = [f32; 2];
pub type BoxObbVertices = [[f32; 2]; 4];

/// A rectangle represented by its center/dimensions and a point on its height axis.
///
/// This follows `raptor`'s `Darknet::BoxOBB`: the bbox dimensions define the
/// rectangle size, while the center-to-front vector defines its orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoxObb {
    pub bbox: DetectionBox,
    pub front_x: f32,
    pub front_y: f32,
}

/// Construct corners in the same order as `raptor`'s `box_obb_points`.
///
/// Non-finite or degenerate boxes return `None`; callers must not silently
/// reinterpret them as axis-aligned boxes.
pub fn box_obb_vertices(boxobb: BoxObb) -> Option<BoxObbVertices> {
    let bbox = boxobb.bbox;
    if !bbox.x.is_finite()
        || !bbox.y.is_finite()
        || !bbox.w.is_finite()
        || !bbox.h.is_finite()
        || !boxobb.front_x.is_finite()
        || !boxobb.front_y.is_finite()
        || bbox.w <= 0.0
        || bbox.h <= 0.0
    {
        return None;
    }

    let dx = boxobb.front_x - bbox.x;
    let dy = boxobb.front_y - bbox.y;
    let length_squared = dx * dx + dy * dy;
    if !length_squared.is_finite() || length_squared < FRONT_MIN_LENGTH * FRONT_MIN_LENGTH {
        return None;
    }

    let length = length_squared.sqrt();
    let axis_h = [dx / length, dy / length];
    let axis_w = [-axis_h[1], axis_h[0]];
    let half_h = [axis_h[0] * bbox.h * 0.5, axis_h[1] * bbox.h * 0.5];
    let half_w = [axis_w[0] * bbox.w * 0.5, axis_w[1] * bbox.w * 0.5];
    Some([
        [
            bbox.x + half_w[0] + half_h[0],
            bbox.y + half_w[1] + half_h[1],
        ],
        [
            bbox.x - half_w[0] + half_h[0],
            bbox.y - half_w[1] + half_h[1],
        ],
        [
            bbox.x - half_w[0] - half_h[0],
            bbox.y - half_w[1] - half_h[1],
        ],
        [
            bbox.x + half_w[0] - half_h[0],
            bbox.y + half_w[1] - half_h[1],
        ],
    ])
}

/// Intersection-over-union of two legacy front-point oriented rectangles.
///
/// Invalid geometry, non-finite intersection/union, and zero-area overlap all
/// return `0.0`, matching the source's explicit no-overlap behavior.
pub fn box_obb_iou(a: BoxObb, b: BoxObb) -> f32 {
    let Some(a_vertices) = box_obb_vertices(a) else {
        return 0.0;
    };
    let Some(b_vertices) = box_obb_vertices(b) else {
        return 0.0;
    };
    if !a_vertices
        .iter()
        .chain(&b_vertices)
        .flatten()
        .all(|coordinate| coordinate.is_finite())
    {
        return 0.0;
    }

    let intersection = convex_intersection(&a_vertices, &b_vertices);
    let intersection_area = polygon_area(&intersection);
    if !intersection_area.is_finite() || intersection_area <= 0.0 {
        return 0.0;
    }

    let union_area = a.bbox.w * a.bbox.h + b.bbox.w * b.bbox.h - intersection_area;
    if !union_area.is_finite() || union_area <= 0.0 {
        return 0.0;
    }
    (intersection_area / union_area).clamp(0.0, 1.0)
}

/// Source CIoU-OBB matching quality for two decoded front-point rectangles.
///
/// This is the score returned by `box_ciou_obb_loss_terms`: rotated IoU minus
/// the center-distance and aspect-ratio CIoU penalties, multiplied by the
/// cosine half-angle orientation score, then clamped to `[0, 1]`. It is a
/// matching quality only, not the source's training loss or raw-channel decode.
/// Invalid/degenerate geometry returns zero.
pub fn box_ciou_obb_quality_score(pred: BoxObb, truth: BoxObb) -> f32 {
    if box_obb_vertices(pred).is_none() || box_obb_vertices(truth).is_none() {
        return 0.0;
    }

    let rotated_iou = box_obb_iou(pred, truth);
    let pred_box = pred.bbox;
    let truth_box = truth.bbox;
    let left = (pred_box.x - pred_box.w * 0.5).min(truth_box.x - truth_box.w * 0.5);
    let right = (pred_box.x + pred_box.w * 0.5).max(truth_box.x + truth_box.w * 0.5);
    let top = (pred_box.y - pred_box.h * 0.5).min(truth_box.y - truth_box.h * 0.5);
    let bottom = (pred_box.y + pred_box.h * 0.5).max(truth_box.y + truth_box.h * 0.5);
    let enclosing_width = right - left;
    let enclosing_height = bottom - top;
    let enclosing_diagonal_squared =
        enclosing_width * enclosing_width + enclosing_height * enclosing_height;
    let ciou = if enclosing_diagonal_squared == 0.0 {
        rotated_iou
    } else {
        let center_dx = pred_box.x - truth_box.x;
        let center_dy = pred_box.y - truth_box.y;
        let center_distance =
            (center_dx * center_dx + center_dy * center_dy) / enclosing_diagonal_squared;
        let pred_aspect = pred_box.w / pred_box.h;
        let truth_aspect = truth_box.w / truth_box.h;
        let aspect_delta = truth_aspect.atan() - pred_aspect.atan();
        let aspect_loss =
            4.0 / (std::f32::consts::PI * std::f32::consts::PI) * aspect_delta * aspect_delta;
        let alpha = aspect_loss / (1.0 - rotated_iou + aspect_loss + 1e-6);
        rotated_iou - center_distance - alpha * aspect_loss
    };

    let pred_dx = pred.front_x - pred_box.x;
    let pred_dy = pred.front_y - pred_box.y;
    let truth_dx = truth.front_x - truth_box.x;
    let truth_dy = truth.front_y - truth_box.y;
    let pred_length_squared = pred_dx * pred_dx + pred_dy * pred_dy;
    let truth_length_squared = truth_dx * truth_dx + truth_dy * truth_dy;
    if !pred_length_squared.is_finite()
        || !truth_length_squared.is_finite()
        || pred_length_squared < FRONT_MIN_LENGTH * FRONT_MIN_LENGTH
        || truth_length_squared < FRONT_MIN_LENGTH * FRONT_MIN_LENGTH
    {
        return 0.0;
    }
    let cosine = ((pred_dx * truth_dx + pred_dy * truth_dy)
        / (pred_length_squared.sqrt() * truth_length_squared.sqrt()))
    .clamp(-1.0, 1.0);
    let cosine_half_angle = (0.5 * (cosine + 1.0)).max(0.0).sqrt();
    let score = ciou * cosine_half_angle;
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// IoU for two CARPN polygons represented as four interleaved `(x, y)` pairs.
///
/// Both inputs must be convex quadrilaterals, matching OpenCV's
/// `intersectConvexConvex` contract used by the source implementation. Invalid,
/// degenerate, or non-finite geometry returns `0.0` rather than an AABB fallback.
pub fn carpn_polygon_iou(pred_vertices: [f32; 8], gt_vertices: [f32; 8]) -> f32 {
    let pred: [Point; 4] =
        std::array::from_fn(|index| [pred_vertices[2 * index], pred_vertices[2 * index + 1]]);
    let gt: [Point; 4] =
        std::array::from_fn(|index| [gt_vertices[2 * index], gt_vertices[2 * index + 1]]);
    if !pred
        .iter()
        .chain(&gt)
        .flatten()
        .all(|value| value.is_finite())
    {
        return 0.0;
    }

    let pred_area = polygon_area(&pred);
    let gt_area = polygon_area(&gt);
    if !pred_area.is_finite() || pred_area <= 0.0 || !gt_area.is_finite() || gt_area <= 0.0 {
        return 0.0;
    }

    let intersection = convex_intersection(&pred, &gt);
    let intersection_area = polygon_area(&intersection);
    if !intersection_area.is_finite() || intersection_area <= 0.0 {
        return 0.0;
    }

    let union_area = pred_area + gt_area - intersection_area;
    if !union_area.is_finite() || union_area <= 0.0 {
        return 0.0;
    }
    (intersection_area / union_area).clamp(0.0, 1.0)
}

fn cross(a: Point, b: Point) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}

fn subtract(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}

fn polygon_signed_area(points: &[Point]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut twice_area = 0.0;
    for index in 0..points.len() {
        let next = (index + 1) % points.len();
        twice_area += cross(points[index], points[next]);
    }
    twice_area * 0.5
}

fn polygon_area(points: &[Point]) -> f32 {
    polygon_signed_area(points).abs()
}

fn line_intersection(
    start: Point,
    end: Point,
    clip_start: Point,
    clip_end: Point,
) -> Option<Point> {
    let segment = subtract(end, start);
    let clip_edge = subtract(clip_end, clip_start);
    let denominator = cross(segment, clip_edge);
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }
    let t = cross(subtract(clip_start, start), clip_edge) / denominator;
    if !t.is_finite() {
        return None;
    }
    Some([start[0] + t * segment[0], start[1] + t * segment[1]])
}

/// Sutherland-Hodgman clipping; both inputs are convex and the clip polygon's
/// signed area determines which side of each edge is interior.
fn convex_intersection(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    let clip_area = polygon_signed_area(clip);
    if !clip_area.is_finite() || clip_area == 0.0 {
        return Vec::new();
    }
    let orientation = clip_area.signum();
    let mut output = subject.to_vec();

    for edge_index in 0..clip.len() {
        if output.is_empty() {
            return output;
        }
        let clip_start = clip[edge_index];
        let clip_end = clip[(edge_index + 1) % clip.len()];
        let input = std::mem::take(&mut output);
        let mut previous = *input.last().expect("nonempty polygon");
        let mut previous_inside = cross(
            subtract(clip_end, clip_start),
            subtract(previous, clip_start),
        ) * orientation
            >= 0.0;

        for current in input {
            let current_inside = cross(
                subtract(clip_end, clip_start),
                subtract(current, clip_start),
            ) * orientation
                >= 0.0;
            if current_inside != previous_inside {
                if let Some(intersection) =
                    line_intersection(previous, current, clip_start, clip_end)
                {
                    output.push(intersection);
                }
            }
            if current_inside {
                output.push(current);
            }
            previous = current;
            previous_inside = current_inside;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(center: [f32; 2], width: f32, height: f32, front: [f32; 2]) -> BoxObb {
        BoxObb {
            bbox: DetectionBox {
                x: center[0],
                y: center[1],
                w: width,
                h: height,
            },
            front_x: front[0],
            front_y: front[1],
        }
    }

    fn close(actual: f32, expected: f32, tolerance: f32) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "got {actual}, expected {expected} within {tolerance}"
        );
    }

    #[test]
    fn vertices_follow_front_point_axis_and_source_corner_order() {
        let vertices = box_obb_vertices(rect([0.0, 0.0], 2.0, 4.0, [0.0, 1.0]))
            .expect("valid oriented rectangle");
        assert_eq!(
            vertices,
            [[-1.0, 2.0], [1.0, 2.0], [1.0, -2.0], [-1.0, -2.0]]
        );

        let rotated = box_obb_vertices(rect([0.0, 0.0], 4.0, 2.0, [1.0, 0.0]))
            .expect("valid rotated rectangle");
        assert_eq!(
            rotated,
            [[1.0, 2.0], [1.0, -2.0], [-1.0, -2.0], [-1.0, 2.0]]
        );
    }

    #[test]
    fn iou_handles_identical_disjoint_partial_and_contained_boxes() {
        let square = rect([0.0, 0.0], 2.0, 2.0, [0.0, 1.0]);
        assert_eq!(box_obb_iou(square, square), 1.0);
        assert_eq!(
            box_obb_iou(square, rect([5.0, 0.0], 2.0, 2.0, [5.0, 1.0])),
            0.0
        );
        close(
            box_obb_iou(square, rect([1.0, 0.0], 2.0, 2.0, [1.0, 1.0])),
            1.0 / 3.0,
            1e-6,
        );
        close(
            box_obb_iou(
                rect([0.0, 0.0], 4.0, 2.0, [0.0, 1.0]),
                rect([0.0, 0.0], 4.0, 2.0, [1.0, 0.0]),
            ),
            1.0 / 3.0,
            1e-6,
        );
        let angled_a = rect([0.0, 0.0], 2.0, 4.0, [0.707_106_77, 0.707_106_77]);
        let angled_b = rect([-0.707_106_77, 0.707_106_77], 2.0, 4.0, [0.0, 1.414_213_5]);
        close(box_obb_iou(angled_a, angled_b), 1.0 / 3.0, 2e-5);
        close(
            box_obb_iou(angled_a, angled_b),
            box_obb_iou(angled_b, angled_a),
            1e-6,
        );
        close(
            box_obb_iou(
                rect([0.0, 0.0], 4.0, 4.0, [0.0, 1.0]),
                rect([0.0, 0.0], 2.0, 2.0, [0.0, 1.0]),
            ),
            0.25,
            1e-6,
        );
    }

    #[test]
    fn ciou_obb_quality_score_combines_rotated_iou_ciou_penalty_and_orientation() {
        let truth = rect([0.0, 0.0], 2.0, 2.0, [0.0, 1.0]);
        assert_eq!(box_ciou_obb_quality_score(truth, truth), 1.0);

        let opposite_front = rect([0.0, 0.0], 2.0, 2.0, [0.0, -1.0]);
        assert_eq!(box_ciou_obb_quality_score(opposite_front, truth), 0.0);

        // Rotated IoU is 1/3; the equal aspect ratios remove the aspect penalty.
        // Center distance is 1 and the enclosing AABB diagonal squared is 13.
        let shifted = rect([1.0, 0.0], 2.0, 2.0, [1.0, 1.0]);
        close(
            box_ciou_obb_quality_score(shifted, truth),
            1.0 / 3.0 - 1.0 / 13.0,
            2e-6,
        );
    }

    #[test]
    fn ciou_obb_quality_score_returns_zero_for_invalid_or_directionless_geometry() {
        let valid = rect([0.0, 0.0], 2.0, 2.0, [0.0, 1.0]);
        assert_eq!(
            box_ciou_obb_quality_score(rect([0.0, 0.0], 2.0, 2.0, [0.0, 0.0]), valid),
            0.0
        );
        assert_eq!(
            box_ciou_obb_quality_score(rect([0.0, 0.0], 0.0, 2.0, [0.0, 1.0]), valid),
            0.0
        );
    }

    #[test]
    fn carpn_polygon_iou_handles_convex_quads_and_both_windings() {
        let square = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        assert_eq!(carpn_polygon_iou(square, square), 1.0);
        assert_eq!(
            carpn_polygon_iou(square, [0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
            1.0
        );

        let partial = [1.0, 0.0, 3.0, 0.0, 3.0, 2.0, 1.0, 2.0];
        close(carpn_polygon_iou(square, partial), 1.0 / 3.0, 1e-6);
        close(carpn_polygon_iou(partial, square), 1.0 / 3.0, 1e-6);
        let contained = [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5];
        close(carpn_polygon_iou(square, contained), 0.25, 1e-6);
        let disjoint = [3.0, 0.0, 5.0, 0.0, 5.0, 2.0, 3.0, 2.0];
        assert_eq!(carpn_polygon_iou(square, disjoint), 0.0);

        // This rotated diamond is a convex polygon, not a front-point rectangle.
        let diamond = [1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0];
        close(carpn_polygon_iou(square, diamond), 0.5, 1e-6);
        let reverse_diamond = [0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0];
        assert_eq!(carpn_polygon_iou(diamond, reverse_diamond), 1.0);
    }

    #[test]
    fn carpn_polygon_iou_rejects_degenerate_and_non_finite_vertices() {
        let square = [0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0];
        let collinear = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        assert_eq!(carpn_polygon_iou(square, collinear), 0.0);
        let non_finite = [0.0, 0.0, 2.0, 0.0, f32::NAN, 2.0, 0.0, 2.0];
        assert_eq!(carpn_polygon_iou(square, non_finite), 0.0);
        let infinite = [0.0, 0.0, 2.0, 0.0, f32::INFINITY, 2.0, 0.0, 2.0];
        assert_eq!(carpn_polygon_iou(infinite, square), 0.0);
    }

    #[test]
    fn invalid_geometry_has_no_vertices_and_zero_iou() {
        let valid = rect([0.0, 0.0], 2.0, 2.0, [0.0, 1.0]);
        assert!(box_obb_vertices(rect([0.0, 0.0], 1.0, 1.0, [FRONT_MIN_LENGTH, 0.0],)).is_some());
        for invalid in [
            rect([0.0, 0.0], 0.0, 2.0, [0.0, 1.0]),
            rect([0.0, 0.0], 2.0, -1.0, [0.0, 1.0]),
            rect([0.0, 0.0], 2.0, 2.0, [0.0, 0.0]),
            rect([0.0, 0.0], 2.0, 2.0, [FRONT_MIN_LENGTH * 0.5, 0.0]),
            rect([0.0, 0.0], 2.0, 2.0, [f32::NAN, 1.0]),
            rect([f32::INFINITY, 0.0], 2.0, 2.0, [0.0, 1.0]),
        ] {
            assert!(box_obb_vertices(invalid).is_none());
            assert_eq!(box_obb_iou(invalid, valid), 0.0);
        }
    }
}

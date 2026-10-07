//! Source-derived oriented bounding-box geometry for the legacy front-point OBB contract.
//!
//! This module provides geometry for OBB decoding/evaluation, source angular,
//! CIoU, covariance-W2 and QBBV matching scores, and CARPN's explicit polygon
//! contract. It does not implement a detector layer, rotated-label loading,
//! complete training losses, or NMS dispatch.

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

#[derive(Clone, Copy)]
struct QbbvFrame {
    cx: f32,
    cy: f32,
    half_w: f32,
    half_h: f32,
    ux: f32,
    uy: f32,
    vx: f32,
    vy: f32,
}

fn qbbv_frame(boxobb: BoxObb) -> QbbvFrame {
    let mut frame = QbbvFrame {
        cx: boxobb.bbox.x,
        cy: boxobb.bbox.y,
        half_w: (0.5 * boxobb.bbox.w).max(1e-6),
        half_h: (0.5 * boxobb.bbox.h).max(1e-6),
        ux: 1.0,
        uy: 0.0,
        vx: 0.0,
        vy: 1.0,
    };
    let dir_x = boxobb.front_x - boxobb.bbox.x;
    let dir_y = boxobb.front_y - boxobb.bbox.y;
    let dir_len = (dir_x * dir_x + dir_y * dir_y).sqrt();
    if dir_len.is_finite() && dir_len >= 1e-6 {
        frame.ux = dir_x / dir_len;
        frame.uy = dir_y / dir_len;
    }
    frame.vx = -frame.uy;
    frame.vy = frame.ux;

    frame.cx = finite_or(frame.cx, 0.0);
    frame.cy = finite_or(frame.cy, 0.0);
    frame.half_w = if frame.half_w.is_finite() {
        frame.half_w.max(1e-6)
    } else {
        1e-6
    };
    frame.half_h = if frame.half_h.is_finite() {
        frame.half_h.max(1e-6)
    } else {
        1e-6
    };
    frame.ux = finite_or(frame.ux, 1.0);
    frame.uy = finite_or(frame.uy, 0.0);
    frame.vx = finite_or(frame.vx, 0.0);
    frame.vy = finite_or(frame.vy, 1.0);
    frame
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn qbbv_heatmap_from_frame(frame: QbbvFrame, x: f32, y: f32) -> f32 {
    let dx = x - frame.cx;
    let dy = y - frame.cy;
    let gx = (dx * frame.ux + dy * frame.uy) / frame.half_w.max(1e-6);
    let gy = (dx * frame.vx + dy * frame.vy) / frame.half_h.max(1e-6);
    let hx = (1.0 - gx.abs()).max(0.0);
    let hy = (1.0 - gy.abs()).max(0.0);
    finite_or(hx * hy, 0.0)
}

/// Evaluate the source QBBV tent heatmap for one point and front-point OBB.
///
/// A degenerate front vector falls back to the source axis-aligned local frame.
pub fn obb_qbbv_heatmap_value(boxobb: BoxObb, x: f32, y: f32) -> f32 {
    qbbv_heatmap_from_frame(qbbv_frame(boxobb), x, y)
}

/// Default sample count used by source QBBV-vIoU matching.
pub const QBBV_DEFAULT_SAMPLES_PER_AXIS: usize = 24;

/// Approximate source QBBV-vIoU with center-sampled heatmap intersections.
///
/// The sample count is clamped to `[4, 128]`. A zero-span or invalid union AABB
/// returns zero; identical valid boxes return one up to floating-point error.
pub fn box_obb_qbbv_viou(pred: BoxObb, truth: BoxObb, samples_per_axis: usize) -> f32 {
    let pred_frame = qbbv_frame(pred);
    let truth_frame = qbbv_frame(truth);
    let aabb = |frame: QbbvFrame| {
        let ext_x = frame.half_w * frame.ux.abs() + frame.half_h * frame.vx.abs();
        let ext_y = frame.half_w * frame.uy.abs() + frame.half_h * frame.vy.abs();
        [
            frame.cx - ext_x,
            frame.cy - ext_y,
            frame.cx + ext_x,
            frame.cy + ext_y,
        ]
    };
    let pred_aabb = aabb(pred_frame);
    let truth_aabb = aabb(truth_frame);
    let min_x = pred_aabb[0].min(truth_aabb[0]);
    let min_y = pred_aabb[1].min(truth_aabb[1]);
    let max_x = pred_aabb[2].max(truth_aabb[2]);
    let max_y = pred_aabb[3].max(truth_aabb[3]);
    let samples = samples_per_axis.clamp(4, 128);
    let span_x = max_x - min_x;
    let span_y = max_y - min_y;
    if !span_x.is_finite() || !span_y.is_finite() || span_x <= 0.0 || span_y <= 0.0 {
        return 0.0;
    }
    let step_x = span_x / samples as f32;
    let step_y = span_y / samples as f32;
    if !step_x.is_finite() || !step_y.is_finite() || step_x <= 0.0 || step_y <= 0.0 {
        return 0.0;
    }

    let mut numerator = 0.0f64;
    let mut denominator = 0.0f64;
    for y_index in 0..samples {
        let y = min_y + (y_index as f32 + 0.5) * step_y;
        for x_index in 0..samples {
            let x = min_x + (x_index as f32 + 0.5) * step_x;
            let pred_value = qbbv_heatmap_from_frame(pred_frame, x, y);
            let truth_value = qbbv_heatmap_from_frame(truth_frame, x, y);
            let pred_squared = pred_value * pred_value;
            let truth_squared = truth_value * truth_value;
            numerator += pred_squared.min(truth_squared) as f64;
            denominator += pred_squared.max(truth_squared) as f64;
        }
    }
    if denominator <= 1e-12 {
        return 0.0;
    }
    finite_or((numerator / denominator) as f32, 0.0).clamp(0.0, 1.0)
}

/// Blend a base OBB score with source QBBV-vIoU using the configured weight.
///
/// The weight is clamped to `[0, 1]`; NaN weights and non-finite scores follow
/// the source sanitization. The source uses 24 samples per axis here.
pub fn blend_obb_score_with_qbbv_viou(
    base_score: f32,
    pred: BoxObb,
    truth: BoxObb,
    viou_weight: f32,
) -> f32 {
    let score = finite_or(base_score, 0.0);
    if viou_weight.is_nan() {
        return 0.0;
    }
    let weight = viou_weight.clamp(0.0, 1.0);
    if weight <= 0.0 {
        return score.clamp(0.0, 1.0);
    }
    let viou = box_obb_qbbv_viou(pred, truth, QBBV_DEFAULT_SAMPLES_PER_AXIS);
    let score = (1.0 - weight) * score + weight * viou;
    finite_or(score, 0.0).clamp(0.0, 1.0)
}

/// Source covariance Wasserstein-2 loss components for decoded OBB geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObbCovarianceW2LossTerms {
    /// Center + covariance + weighted front-length loss.
    pub total: f32,
    /// Squared center displacement.
    pub center: f32,
    /// Squared Wasserstein covariance distance.
    pub covariance: f32,
    /// Unweighted squared front-length residual.
    pub front_length: f32,
}

fn valid_covariance_w2_inputs(
    pred: BoxObb,
    truth: BoxObb,
    front_length_ratio_k: f32,
    front_length_weight: f32,
) -> bool {
    [
        pred.bbox.x,
        pred.bbox.y,
        pred.bbox.w,
        pred.bbox.h,
        pred.front_x,
        pred.front_y,
        truth.bbox.x,
        truth.bbox.y,
        truth.bbox.w,
        truth.bbox.h,
        truth.front_x,
        truth.front_y,
        front_length_ratio_k,
        front_length_weight,
    ]
    .iter()
    .all(|value| value.is_finite())
        && pred.bbox.w > 0.0
        && pred.bbox.h > 0.0
        && truth.bbox.w >= 0.0
        && truth.bbox.h >= 0.0
}

/// Compute the source covariance-W2 loss terms for already-decoded OBB geometry.
///
/// The default source epsilons are used. `front_length_ratio_k` is clamped to at
/// least 0.05 and `front_length_weight` to at least zero, matching the OBB loss
/// score configuration. This is not the raw-channel decoder, optional QBBV-vIoU
/// blend, or training gradient.
pub fn box_covariance_w2_obb_loss_terms(
    pred: BoxObb,
    truth: BoxObb,
    front_length_ratio_k: f32,
    front_length_weight: f32,
) -> ObbCovarianceW2LossTerms {
    const EPS_NORM: f32 = 1e-6;
    const EPS_SPD: f32 = 1e-6;
    const EPS_SIZE: f32 = 1e-6;
    const EPS_SMOOTH_MAX: f32 = 1e-6;

    if !valid_covariance_w2_inputs(pred, truth, front_length_ratio_k, front_length_weight) {
        return ObbCovarianceW2LossTerms::default();
    }

    let front_length_ratio_k = front_length_ratio_k.max(0.05);
    let front_length_weight = front_length_weight.max(0.0);
    let decoded_size = |size: f32| {
        let positive = (size - EPS_SIZE).max(EPS_SIZE);
        let raw = if positive > 20.0 {
            positive
        } else {
            positive.exp_m1().ln()
        };
        (-raw.abs()).exp().ln_1p() + raw.max(0.0) + EPS_SIZE
    };
    let width = decoded_size(pred.bbox.w);
    let height = decoded_size(pred.bbox.h);
    let truth_width = truth.bbox.w.abs() + EPS_SIZE;
    let truth_height = truth.bbox.h.abs() + EPS_SIZE;

    let front_dx = pred.front_x - pred.bbox.x;
    let front_dy = pred.front_y - pred.bbox.y;
    let front_norm = (front_dx * front_dx + front_dy * front_dy + EPS_NORM).sqrt();
    let ux = front_dx / front_norm;
    let uy = front_dy / front_norm;
    let truth_front_dx = truth.front_x - truth.bbox.x;
    let truth_front_dy = truth.front_y - truth.bbox.y;
    let truth_front_norm =
        (truth_front_dx * truth_front_dx + truth_front_dy * truth_front_dy + EPS_NORM).sqrt();
    let truth_ux = truth_front_dx / truth_front_norm;
    let truth_uy = truth_front_dy / truth_front_norm;

    let covariance = |w: f32, h: f32, ux: f32, uy: f32| {
        let a = 0.25 * w * w;
        let b = 0.25 * h * h;
        [
            a * ux * ux + b * uy * uy,
            (a - b) * ux * uy,
            a * uy * uy + b * ux * ux,
        ]
    };
    let pred_cov = covariance(width, height, ux, uy);
    let truth_cov = covariance(truth_width, truth_height, truth_ux, truth_uy);
    let determinant = |cov: [f32; 3]| cov[0] * cov[2] - cov[1] * cov[1];
    let trace_pred = pred_cov[0] + pred_cov[2];
    let trace_truth = truth_cov[0] + truth_cov[2];
    let trace_product =
        pred_cov[0] * truth_cov[0] + 2.0 * pred_cov[1] * truth_cov[1] + pred_cov[2] * truth_cov[2];
    let q = (determinant(pred_cov) * determinant(truth_cov) + EPS_SPD).sqrt();
    let s = (trace_product + 2.0 * q + EPS_SPD).sqrt();
    let covariance_loss = trace_pred + trace_truth - 2.0 * s;

    let center_dx = pred.bbox.x - truth.bbox.x;
    let center_dy = pred.bbox.y - truth.bbox.y;
    let center = center_dx * center_dx + center_dy * center_dy;
    let diff_wh = width - height;
    let smooth_wh = (diff_wh * diff_wh + EPS_SMOOTH_MAX).sqrt();
    let max_wh = 0.5 * (width + height + smooth_wh);
    let length_residual = front_norm - front_length_ratio_k * max_wh;
    let front_length = length_residual * length_residual;
    let total = center + covariance_loss + front_length_weight * front_length;

    let sanitize = |value: f32| if value.is_finite() { value } else { 0.0 };
    ObbCovarianceW2LossTerms {
        total: sanitize(total),
        center: sanitize(center),
        covariance: sanitize(covariance_loss),
        front_length: sanitize(front_length),
    }
}

/// Source covariance-W2 matching quality for decoded OBB geometry.
///
/// The matching score is `exp(-total_loss)`, clamped to `[0, 1]`. Optional
/// QBBV-vIoU blending and raw prediction decoding are separate source paths.
pub fn box_covariance_w2_obb_quality_score(
    pred: BoxObb,
    truth: BoxObb,
    front_length_ratio_k: f32,
    front_length_weight: f32,
) -> f32 {
    if !valid_covariance_w2_inputs(pred, truth, front_length_ratio_k, front_length_weight) {
        return 0.0;
    }
    let terms =
        box_covariance_w2_obb_loss_terms(pred, truth, front_length_ratio_k, front_length_weight);
    let score = (-terms.total).exp();
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Source legacy angular-OBB matching score for decoded geometry.
///
/// This is axis-aligned `box_iou` times the cosine half-angle of the front
/// directions, unlike the separate CIoU-OBB quality above. An absent/invalid
/// front direction gives zero angular score.
pub fn box_ciou_angular_quality_score(pred: BoxObb, truth: BoxObb) -> f32 {
    let pred_dx = pred.front_x - pred.bbox.x;
    let pred_dy = pred.front_y - pred.bbox.y;
    let truth_dx = truth.front_x - truth.bbox.x;
    let truth_dy = truth.front_y - truth.bbox.y;
    let pred_length_squared = pred_dx * pred_dx + pred_dy * pred_dy;
    let truth_length_squared = truth_dx * truth_dx + truth_dy * truth_dy;
    if !pred_length_squared.is_finite()
        || !truth_length_squared.is_finite()
        || pred_length_squared < FRONT_MIN_LENGTH * FRONT_MIN_LENGTH
        || truth_length_squared < FRONT_MIN_LENGTH * FRONT_MIN_LENGTH
    {
        return 0.0;
    }

    let pred_bbox = pred.bbox;
    let truth_bbox = truth.bbox;
    if ![
        pred_bbox.x,
        pred_bbox.y,
        pred_bbox.w,
        pred_bbox.h,
        truth_bbox.x,
        truth_bbox.y,
        truth_bbox.w,
        truth_bbox.h,
    ]
    .into_iter()
    .all(f32::is_finite)
        || pred_bbox.w <= 0.0
        || pred_bbox.h <= 0.0
        || truth_bbox.w <= 0.0
        || truth_bbox.h <= 0.0
    {
        return 0.0;
    }

    let left = (pred_bbox.x - pred_bbox.w * 0.5).max(truth_bbox.x - truth_bbox.w * 0.5);
    let right = (pred_bbox.x + pred_bbox.w * 0.5).min(truth_bbox.x + truth_bbox.w * 0.5);
    let top = (pred_bbox.y - pred_bbox.h * 0.5).max(truth_bbox.y - truth_bbox.h * 0.5);
    let bottom = (pred_bbox.y + pred_bbox.h * 0.5).min(truth_bbox.y + truth_bbox.h * 0.5);
    let intersection_width = right - left;
    let intersection_height = bottom - top;
    if intersection_width <= 0.0 || intersection_height <= 0.0 {
        return 0.0;
    }
    let intersection = intersection_width * intersection_height;
    let union = pred_bbox.w * pred_bbox.h + truth_bbox.w * truth_bbox.h - intersection;
    if !intersection.is_finite() || !union.is_finite() || union <= 0.0 {
        return 0.0;
    }

    let pred_length = pred_length_squared.sqrt();
    let truth_length = truth_length_squared.sqrt();
    let cosine =
        ((pred_dx * truth_dx + pred_dy * truth_dy) / (pred_length * truth_length)).clamp(-1.0, 1.0);
    let cosine_half_angle = (0.5 * (cosine + 1.0)).max(0.0).sqrt();
    let score = (intersection / union) * cosine_half_angle;
    finite_or(score, 0.0).clamp(0.0, 1.0)
}

/// Weights and target exponent for the source COBB assignment/loss score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbCobbLossConfig {
    pub box_weight: f32,
    pub ratio_weight: f32,
    pub score_weight: f32,
    pub iou_lambda: f32,
}

impl Default for ObbCobbLossConfig {
    fn default() -> Self {
        Self {
            box_weight: 1.0,
            ratio_weight: 1.0,
            score_weight: 1.0,
            iou_lambda: 2.0,
        }
    }
}

/// Component losses and target/prediction score rows from the source COBB loss.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObbCobbLossTerms {
    pub total: f32,
    pub box_term: f32,
    pub ratio_term: f32,
    pub score_term: f32,
    pub truth_type: Option<usize>,
    pub truth_scores: [f32; 4],
    pub prediction_scores: [f32; 4],
}

#[derive(Clone, Copy)]
struct CobbOuterParams {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    rs: f32,
}

#[derive(Clone, Copy)]
struct CobbCandidate {
    obb: BoxObb,
}

#[derive(Clone, Copy)]
struct CobbTargets {
    outer: CobbOuterParams,
    ratio_signal: f32,
    type_index: usize,
    scores: [f32; 4],
}

fn cobb_outer_from_obb(boxobb: BoxObb) -> Option<CobbOuterParams> {
    let vertices = box_obb_vertices(boxobb)?;
    let mut min_x = vertices[0][0];
    let mut max_x = vertices[0][0];
    let mut min_y = vertices[0][1];
    let mut max_y = vertices[0][1];
    let mut xs = vertices.map(|point| point[0]);
    let mut ys = vertices.map(|point| point[1]);
    for point in &vertices[1..] {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    xs.sort_by(f32::total_cmp);
    ys.sort_by(f32::total_cmp);

    let width = (max_x - min_x).max(1e-6);
    let height = (max_y - min_y).max(1e-6);
    let ratio = if width < height {
        (xs[1] - xs[0]) / width
    } else {
        (ys[1] - ys[0]) / height
    };
    Some(CobbOuterParams {
        x: boxobb.bbox.x,
        y: boxobb.bbox.y,
        w: width,
        h: height,
        rs: finite_or(ratio, 0.0).clamp(0.0, 0.5),
    })
}

fn cobb_candidate_from_vertices(vertices: [Point; 4], outer: CobbOuterParams) -> CobbCandidate {
    let edge01 = [
        vertices[1][0] - vertices[0][0],
        vertices[1][1] - vertices[0][1],
    ];
    let edge12 = [
        vertices[2][0] - vertices[1][0],
        vertices[2][1] - vertices[1][1],
    ];
    CobbCandidate {
        obb: BoxObb {
            bbox: DetectionBox {
                x: outer.x,
                y: outer.y,
                w: (edge01[0] * edge01[0] + edge01[1] * edge01[1])
                    .sqrt()
                    .max(1e-6),
                h: (edge12[0] * edge12[0] + edge12[1] * edge12[1])
                    .sqrt()
                    .max(1e-6),
            },
            front_x: 0.5 * (vertices[0][0] + vertices[1][0]),
            front_y: 0.5 * (vertices[0][1] + vertices[1][1]),
        },
    }
}

fn cobb_build_candidates(outer: CobbOuterParams) -> [CobbCandidate; 4] {
    let width = outer.w.max(1e-6);
    let height = outer.h.max(1e-6);
    let ratio = outer.rs.clamp(0.0, 0.5);
    let ratio_term = ratio * (1.0 - ratio);
    let (x_shift, y_shift) = if width >= height {
        let inside = (1.0 - 4.0 * height * height / (width * width) * ratio_term).max(0.0);
        (
            0.5 * inside.sqrt() * width,
            0.5 * (1.0 - 2.0 * ratio) * height,
        )
    } else {
        let inside = (1.0 - 4.0 * width * width / (height * height) * ratio_term).max(0.0);
        (
            0.5 * (1.0 - 2.0 * ratio) * width,
            0.5 * inside.sqrt() * height,
        )
    };
    let center_x = outer.x;
    let center_y = outer.y;
    let half_width = 0.5 * width;
    let half_height = 0.5 * height;
    let vertices = [
        [
            [center_x - x_shift, center_y - half_height],
            [center_x + half_width, center_y + y_shift],
            [center_x + x_shift, center_y + half_height],
            [center_x - half_width, center_y - y_shift],
        ],
        [
            [center_x + x_shift, center_y - half_height],
            [center_x + half_width, center_y + y_shift],
            [center_x - x_shift, center_y + half_height],
            [center_x - half_width, center_y - y_shift],
        ],
        [
            [center_x - x_shift, center_y - half_height],
            [center_x + half_width, center_y - y_shift],
            [center_x + x_shift, center_y + half_height],
            [center_x - half_width, center_y + y_shift],
        ],
        [
            [center_x + x_shift, center_y - half_height],
            [center_x + half_width, center_y - y_shift],
            [center_x - x_shift, center_y + half_height],
            [center_x - half_width, center_y + y_shift],
        ],
    ];
    vertices.map(|candidate| cobb_candidate_from_vertices(candidate, outer))
}

fn cobb_safe_div(numerator: f32, denominator: f32) -> f32 {
    if !numerator.is_finite() || !denominator.is_finite() || denominator.abs() < 1e-12 {
        0.0
    } else {
        numerator / denominator
    }
}

fn cobb_iou_row(width: f32, height: f32, ratio: f32, type_index: usize) -> [f32; 4] {
    let width = width.max(1e-6);
    let height = height.max(1e-6);
    let ratio = ratio.clamp(0.0, 0.5);
    let ratio_term = ratio * (1.0 - ratio);
    let (mut ratio_x, mut ratio_y) = (ratio, ratio);
    if width >= height {
        let inside = (1.0 - 4.0 * height * height / (width * width) * ratio_term).max(0.0);
        ratio_x = 0.5 * (1.0 - inside.sqrt());
    } else {
        let inside = (1.0 - 4.0 * width * width / (height * height) * ratio_term).max(0.0);
        ratio_y = 0.5 * (1.0 - inside.sqrt());
    }
    ratio_x = ratio_x.clamp(0.0, 0.5);
    ratio_y = ratio_y.clamp(0.0, 0.5);

    let l1 = ((ratio_x * width).powi(2) + (ratio_y * height).powi(2)).sqrt();
    let l2 = (((1.0 - ratio_x) * width).powi(2) + ((1.0 - ratio_y) * height).powi(2)).sqrt();
    let l3 = ((ratio_x * width).powi(2) + ((1.0 - ratio_y) * height).powi(2)).sqrt();
    let l4 = (((1.0 - ratio_x) * width).powi(2) + (ratio_y * height).powi(2)).sqrt();

    let i01 =
        (1.0 - cobb_safe_div(
            (1.0 - 2.0 * ratio_x) * ratio_x * width * width,
            (1.0 - ratio_y) * height * height,
        )) * l1
            * l2;
    let iou01 = cobb_safe_div(i01, l1 * l2 + l3 * l4 - i01).clamp(0.0, 1.0);
    let i02 =
        (1.0 - cobb_safe_div(
            (1.0 - 2.0 * ratio_y) * ratio_y * height * height,
            (1.0 - ratio_x) * width * width,
        )) * l1
            * l2;
    let iou02 = cobb_safe_div(i02, l1 * l2 + l3 * l4 - i02).clamp(0.0, 1.0);
    let i03_numerator = ratio_x + ratio_y - 2.0 * ratio_x * ratio_y;
    let i03 = cobb_safe_div(
        i03_numerator * i03_numerator,
        (1.0 - ratio_x) * (1.0 - ratio_y),
    ) * width
        * height
        * 0.5;
    let iou03 = if i03.abs() > 1e-12 {
        cobb_safe_div(i03, 2.0 * l1 * l2 - i03).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let h1 = 0.5 * width - cobb_safe_div(0.5 - ratio_y, 1.0 - ratio_y) * ratio_x * width;
    let h2 = 0.5 * height - cobb_safe_div(0.5 - ratio_x, 1.0 - ratio_x) * ratio_y * height;
    let tan_a = cobb_safe_div(
        cobb_safe_div(0.5 - ratio_x, 1.0 - ratio_x) * l4,
        cobb_safe_div(0.5, 1.0 - ratio_y) * l3,
    );
    let tan_b = cobb_safe_div(
        cobb_safe_div(0.5 - ratio_y, 1.0 - ratio_y) * l3,
        cobb_safe_div(0.5, 1.0 - ratio_x) * l4,
    );
    let tan_product = tan_a * tan_b;
    let mut i12 = 2.0 * h1 * h2;
    let mut denominator12 = 2.0 * l3 * l4 - 2.0 * h1 * h2;
    if tan_product.abs() > 1e-12 {
        i12 = 2.0 * cobb_safe_div(tan_product, tan_a + tan_b) * (h1 * h1 + h2 * h2) + 2.0 * h1 * h2;
        denominator12 = 2.0 * l3 * l4 - i12;
    }
    let iou12 = cobb_safe_div(i12, denominator12).clamp(0.0, 1.0);

    let rows = [
        [1.0, iou01, iou02, iou03],
        [iou01, 1.0, iou12, iou02],
        [iou02, iou12, 1.0, iou01],
        [iou03, iou02, iou01, 1.0],
    ];
    rows[type_index.min(3)]
}

fn cobb_build_targets(truth: BoxObb, config: ObbCobbLossConfig) -> Option<CobbTargets> {
    let outer = cobb_outer_from_obb(truth)?;
    let candidates = cobb_build_candidates(outer);
    let mut best_type = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (index, candidate) in candidates.into_iter().enumerate() {
        let score = box_obb_qbbv_viou(candidate.obb, truth, 12);
        let score = finite_or(score, 0.0);
        if score > best_score {
            best_score = score;
            best_type = index;
        }
    }
    let lambda = finite_or(config.iou_lambda, 0.0).max(0.0);
    let scores = cobb_iou_row(outer.w, outer.h, outer.rs, best_type)
        .map(|score| finite_or(score.clamp(0.0, 1.0).powf(lambda), 0.0));
    Some(CobbTargets {
        outer,
        ratio_signal: 2.0 * outer.rs,
        type_index: best_type,
        scores,
    })
}

fn cobb_smooth_l1(abs_diff: f32) -> f32 {
    if abs_diff < 1.0 {
        0.5 * abs_diff * abs_diff
    } else {
        abs_diff - 0.5
    }
}

fn invalid_cobb_terms() -> ObbCobbLossTerms {
    ObbCobbLossTerms {
        total: 10.0,
        ..ObbCobbLossTerms::default()
    }
}

/// Source COBB loss components for an already-decoded prediction/truth pair.
///
/// This ports the target candidate/type selection, analytic four-type overlap
/// row, box/ratio/score Smooth-L1 terms, and invalid-geometry fallback. It does
/// not decode YOLO channels or apply QBBV score blending.
pub fn box_cobb_obb_loss_terms(
    pred: BoxObb,
    truth: BoxObb,
    config: ObbCobbLossConfig,
) -> ObbCobbLossTerms {
    let Some(targets) = cobb_build_targets(truth, config) else {
        return invalid_cobb_terms();
    };
    let Some(pred_outer) = cobb_outer_from_obb(pred) else {
        return invalid_cobb_terms();
    };
    let prediction_scores = cobb_iou_row(
        pred_outer.w,
        pred_outer.h,
        pred_outer.rs,
        targets.type_index,
    );
    let box_weight = finite_or(config.box_weight, 0.0).max(0.0);
    let ratio_weight = finite_or(config.ratio_weight, 0.0).max(0.0);
    let score_weight = finite_or(config.score_weight, 0.0).max(0.0);
    let box_term = cobb_smooth_l1((pred_outer.x - targets.outer.x).abs())
        + cobb_smooth_l1((pred_outer.y - targets.outer.y).abs())
        + cobb_smooth_l1((pred_outer.w - targets.outer.w).abs())
        + cobb_smooth_l1((pred_outer.h - targets.outer.h).abs());
    let ratio_term = cobb_smooth_l1((2.0 * pred_outer.rs - targets.ratio_signal).abs());
    let score_term = prediction_scores
        .iter()
        .zip(targets.scores)
        .map(|(prediction, target)| cobb_smooth_l1((prediction - target).abs()))
        .sum::<f32>();
    let total = box_weight * box_term + ratio_weight * ratio_term + score_weight * score_term;
    ObbCobbLossTerms {
        total: finite_or(total, 0.0).max(0.0),
        box_term: finite_or(box_term, 0.0),
        ratio_term: finite_or(ratio_term, 0.0),
        score_term: finite_or(score_term, 0.0),
        truth_type: Some(targets.type_index),
        truth_scores: targets.scores,
        prediction_scores,
    }
}

/// Source COBB matching quality `exp(-loss)` for decoded OBB geometry.
pub fn box_cobb_obb_quality_score(pred: BoxObb, truth: BoxObb, config: ObbCobbLossConfig) -> f32 {
    (-box_cobb_obb_loss_terms(pred, truth, config).total)
        .exp()
        .clamp(0.0, 1.0)
}

/// Source COBB matching quality with the common QBBV-vIoU blend applied.
pub fn box_cobb_obb_quality_score_with_viou(
    pred: BoxObb,
    truth: BoxObb,
    config: ObbCobbLossConfig,
    viou_weight: f32,
) -> f32 {
    blend_obb_score_with_qbbv_viou(
        box_cobb_obb_quality_score(pred, truth, config),
        pred,
        truth,
        viou_weight,
    )
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
    fn cobb_score_row_and_loss_match_source_shape_terms() {
        let row0 = cobb_iou_row(1.0, 1.0, 0.0, 0);
        let row1 = cobb_iou_row(1.0, 1.0, 0.0, 1);
        assert_eq!(row0, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(row1[0], row0[1]);
        assert_eq!(row1[2], 1.0);
        assert_eq!(row1[3], row0[2]);

        let truth = rect([0.5, 0.5], 0.2, 0.4, [0.5, 0.7]);
        let identical = box_cobb_obb_loss_terms(truth, truth, ObbCobbLossConfig::default());
        assert_eq!(identical.total, 0.0);
        assert!(identical.truth_type.is_some());
        assert_eq!(identical.truth_scores, identical.prediction_scores);
        assert_eq!(
            box_cobb_obb_quality_score(truth, truth, ObbCobbLossConfig::default()),
            1.0
        );
        assert_eq!(
            box_cobb_obb_quality_score_with_viou(truth, truth, ObbCobbLossConfig::default(), 0.4,),
            1.0
        );

        let shifted = rect([0.7, 0.5], 0.2, 0.4, [0.7, 0.7]);
        let shifted_terms = box_cobb_obb_loss_terms(shifted, truth, ObbCobbLossConfig::default());
        assert!(shifted_terms.box_term > 0.0);
        assert!(shifted_terms.total > 0.0);
        let shifted_quality =
            box_cobb_obb_quality_score(shifted, truth, ObbCobbLossConfig::default());
        assert!(shifted_quality < 1.0);
        assert_eq!(
            box_cobb_obb_quality_score_with_viou(
                shifted,
                truth,
                ObbCobbLossConfig::default(),
                0.35,
            ),
            blend_obb_score_with_qbbv_viou(shifted_quality, shifted, truth, 0.35)
        );

        let no_loss = ObbCobbLossConfig {
            box_weight: 0.0,
            ratio_weight: 0.0,
            score_weight: 0.0,
            ..ObbCobbLossConfig::default()
        };
        assert_eq!(box_cobb_obb_loss_terms(shifted, truth, no_loss).total, 0.0);
    }

    #[test]
    fn cobb_loss_uses_source_invalid_geometry_fallback() {
        let valid = rect([0.5, 0.5], 0.2, 0.4, [0.5, 0.7]);
        let direction_free = rect([0.5, 0.5], 0.2, 0.4, [0.5, 0.5]);
        let terms = box_cobb_obb_loss_terms(valid, direction_free, ObbCobbLossConfig::default());
        assert_eq!(terms.total, 10.0);
        assert_eq!(terms.truth_type, None);
        assert_eq!(
            box_cobb_obb_quality_score(valid, direction_free, ObbCobbLossConfig::default()),
            (-10.0f32).exp()
        );
    }

    #[test]
    fn ciou_angular_quality_score_uses_axis_iou_and_front_half_angle() {
        let vertical = rect([0.5, 0.5], 0.2, 0.4, [0.5, 0.7]);
        assert_eq!(box_ciou_angular_quality_score(vertical, vertical), 1.0);

        let horizontal = rect([0.5, 0.5], 0.2, 0.4, [0.7, 0.5]);
        close(
            box_ciou_angular_quality_score(horizontal, vertical),
            std::f32::consts::FRAC_1_SQRT_2,
            1e-6,
        );
        let opposite = rect([0.5, 0.5], 0.2, 0.4, [0.5, 0.3]);
        assert_eq!(box_ciou_angular_quality_score(opposite, vertical), 0.0);

        let disjoint = rect([0.9, 0.9], 0.2, 0.4, [0.9, 1.1]);
        assert_eq!(box_ciou_angular_quality_score(disjoint, vertical), 0.0);
        let direction_free = rect([0.5, 0.5], 0.2, 0.4, [0.5, 0.5]);
        assert_eq!(
            box_ciou_angular_quality_score(direction_free, vertical),
            0.0
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
    fn qbbv_heatmap_has_source_center_edge_and_midpoint_values() {
        let boxobb = rect([0.5, 0.5], 0.4, 0.2, [0.6, 0.5]);
        assert_eq!(obb_qbbv_heatmap_value(boxobb, 0.5, 0.5), 1.0);
        close(obb_qbbv_heatmap_value(boxobb, 0.7, 0.5), 0.0, 1e-6);
        close(obb_qbbv_heatmap_value(boxobb, 0.6, 0.5), 0.5, 1e-6);

        let direction_free = rect([0.5, 0.5], 0.4, 0.2, [0.5, 0.5]);
        assert_eq!(obb_qbbv_heatmap_value(direction_free, 0.5, 0.5), 1.0);
    }

    #[test]
    fn qbbv_viou_handles_identical_disjoint_partial_and_sample_clamping() {
        let identical = rect([0.5, 0.5], 0.32, 0.16, [0.58, 0.5]);
        assert_eq!(box_obb_qbbv_viou(identical, identical, 48), 1.0);
        let disjoint_a = rect([0.2, 0.2], 0.2, 0.1, [0.3, 0.2]);
        let disjoint_b = rect([0.8, 0.8], 0.2, 0.1, [0.9, 0.8]);
        assert_eq!(box_obb_qbbv_viou(disjoint_a, disjoint_b, 48), 0.0);

        let partial_a = rect([0.45, 0.5], 0.3, 0.16, [0.53, 0.5]);
        let partial_b = rect([0.55, 0.5], 0.3, 0.16, [0.63, 0.5]);
        let partial_score = box_obb_qbbv_viou(partial_a, partial_b, 48);
        assert!(partial_score > 0.0 && partial_score < 1.0);
        close(
            partial_score,
            box_obb_qbbv_viou(partial_b, partial_a, 48),
            1e-6,
        );
        assert_eq!(
            box_obb_qbbv_viou(partial_a, partial_b, 0),
            box_obb_qbbv_viou(partial_a, partial_b, 4)
        );
        assert_eq!(
            box_obb_qbbv_viou(partial_a, partial_b, usize::MAX),
            box_obb_qbbv_viou(partial_a, partial_b, 128)
        );
    }

    #[test]
    fn qbbv_blending_clamps_weights_and_sanitizes_nonfinite_inputs() {
        let boxobb = rect([0.5, 0.5], 0.4, 0.2, [0.6, 0.5]);
        close(
            blend_obb_score_with_qbbv_viou(0.4, boxobb, boxobb, 0.25),
            0.55,
            1e-6,
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(0.0, boxobb, boxobb, 0.0),
            0.0
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(0.0, boxobb, boxobb, 1.0),
            1.0
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(0.4, boxobb, boxobb, -1.0),
            0.4
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(0.4, boxobb, boxobb, 2.0),
            1.0
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(f32::NAN, boxobb, boxobb, 1.0),
            1.0
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(0.4, boxobb, boxobb, f32::INFINITY),
            1.0
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(0.4, boxobb, boxobb, f32::NEG_INFINITY),
            0.4
        );
        assert_eq!(
            blend_obb_score_with_qbbv_viou(0.4, boxobb, boxobb, f32::NAN),
            0.0
        );

        let disjoint = rect([0.8, 0.8], 0.4, 0.2, [0.9, 0.8]);
        close(
            blend_obb_score_with_qbbv_viou(0.8, boxobb, disjoint, 0.5),
            0.4,
            1e-6,
        );
    }

    #[test]
    fn covariance_w2_score_tracks_center_shape_and_front_length_terms() {
        let identical = rect([0.5, 0.5], 2.0, 4.0, [0.5, 4.5]);
        let identical_terms = box_covariance_w2_obb_loss_terms(identical, identical, 1.0, 0.35);
        assert_eq!(identical_terms.center, 0.0);
        assert!(identical_terms.covariance.abs() < 1e-5);
        assert!(identical_terms.front_length < 1e-8);
        assert_eq!(
            box_covariance_w2_obb_quality_score(identical, identical, 1.0, 0.35),
            1.0
        );

        let shifted = rect([0.6, 0.5], 2.0, 4.0, [0.6, 4.5]);
        let shifted_terms = box_covariance_w2_obb_loss_terms(identical, shifted, 1.0, 0.0);
        close(shifted_terms.center, 0.01, 1e-6);
        close(
            box_covariance_w2_obb_quality_score(identical, shifted, 1.0, 0.0),
            (-0.01f32).exp(),
            2e-5,
        );

        let rotated = rect([0.5, 0.5], 2.0, 4.0, [4.5, 0.5]);
        let orientation_terms = box_covariance_w2_obb_loss_terms(identical, rotated, 1.0, 0.0);
        assert!(orientation_terms.covariance > 0.0);
        assert!(box_covariance_w2_obb_quality_score(identical, rotated, 1.0, 0.0) < 1.0);

        let short_front = rect([0.5, 0.5], 2.0, 4.0, [0.5, 0.6]);
        let front_terms = box_covariance_w2_obb_loss_terms(short_front, identical, 0.05, 0.5);
        close(front_terms.front_length, 0.01, 2e-5);
        close(front_terms.total, front_terms.covariance + 0.005, 2e-5);
    }

    #[test]
    fn covariance_w2_score_returns_zero_for_invalid_geometry_or_config() {
        let valid = rect([0.0, 0.0], 2.0, 2.0, [0.0, 1.0]);
        let invalid = rect([0.0, 0.0], 0.0, 2.0, [0.0, 1.0]);
        assert_eq!(
            box_covariance_w2_obb_quality_score(invalid, valid, 1.0, 0.35),
            0.0
        );
        assert_eq!(
            box_covariance_w2_obb_quality_score(valid, valid, f32::NAN, 0.35),
            0.0
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

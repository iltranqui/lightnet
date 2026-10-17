//! YOLO output decoding and CPU non-maximum suppression, with separate OBB
//! post-processing primitives.
//!
//! The runtime leaves YOLO heads in Darknet's flat channel-major layout.  This
//! module converts those tensors to image-space detections and keeps
//! post-processing separate from neural-network execution, like the C++
//! `get_yolo_detections()` and NMS helpers.

use crate::carpn::{ObbRefineOutput, ObbRefineProposal};
use crate::error::{Error, Result};
use crate::model::{
    HfRadiusMode, HyperbolicFoveationSpec, LayerKind, Network, ObbNmsKind, ObbRefineSpec,
    TensorShape, YoloSpec,
};
use crate::obb_geometry::{BoxObb, box_obb_iou, carpn_polygon_iou};
use crate::runtime::{ForwardResult, ModernHeadOutput, Tensor, YoloHeadOutput};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl DetectionBox {
    pub fn left(self) -> f32 {
        self.x - self.w * 0.5
    }

    pub fn top(self) -> f32 {
        self.y - self.h * 0.5
    }

    pub fn right(self) -> f32 {
        self.x + self.w * 0.5
    }

    pub fn bottom(self) -> f32 {
        self.y + self.h * 0.5
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub class_id: usize,
    pub confidence: f32,
    pub objectness: f32,
    pub bbox: DetectionBox,
}

/// Geometry carried by a detection from an OBB head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ObbDetectionGeometry {
    /// Legacy Darknet OBB representation: one point on the height axis.
    FrontPoint { x: f32, y: f32 },
    /// CARPN representation: four interleaved vertex `(x, y)` pairs.
    CarpnQuad([f32; 8]),
}

/// A class-specific detection with OBB geometry, separate from the existing
/// axis-aligned [`Detection`] API.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbDetection {
    pub detection: Detection,
    pub geometry: ObbDetectionGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectionOptions {
    pub confidence_threshold: f32,
    pub nms_threshold: f32,
    pub letterbox: bool,
}

impl Default for DetectionOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.25,
            nms_threshold: 0.45,
            letterbox: true,
        }
    }
}

pub fn decode_yolo_detections(
    network: &Network,
    yolo_heads: &[YoloHeadOutput],
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();

    for head in yolo_heads {
        let layer = network.layers.get(head.layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!("missing layer #{} for YOLO head", head.layer_index))
        })?;
        let spec = match &layer.kind {
            LayerKind::Yolo(spec) | LayerKind::GaussianYolo(spec) => spec,
            LayerKind::ObbYolo(_) => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} is an OBB head; use decode_obb_yolo_detections",
                    head.layer_index
                )));
            }
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} is not a YOLO head",
                    head.layer_index
                )));
            }
        };
        decode_yolo_head(
            &head.output.data,
            head.output.shape.width,
            head.output.shape.height,
            spec,
            network.net.width,
            network.net.height,
            image_width,
            image_height,
            options,
            &mut detections,
        )?;
    }

    apply_nms(&mut detections, options.nms_threshold);
    detections.sort_by(|lhs, rhs| rhs.confidence.total_cmp(&lhs.confidence));
    Ok(detections)
}

/// Decode all axis-aligned heads for one batch item and invert the last
/// hyperbolic-foveation layer using that item's cached parameters.
pub fn decode_network_detections(
    network: &Network,
    forward: &ForwardResult,
    batch_index: usize,
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<Detection>> {
    if image_width == 0 || image_height == 0 {
        return Err(Error::InvalidConfig(
            "image dimensions must be positive for network detection decoding".to_owned(),
        ));
    }
    let mut head_options = options;
    head_options.nms_threshold = 1.0;
    let mut yolo_heads = Vec::with_capacity(forward.yolo_heads.len());
    for head in &forward.yolo_heads {
        let layer = network.layers.get(head.layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!("missing layer #{} for YOLO head", head.layer_index))
        })?;
        let spec = match &layer.kind {
            LayerKind::Yolo(spec) | LayerKind::GaussianYolo(spec) => spec,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} is not a YOLO head",
                    head.layer_index
                )));
            }
        };
        if batch_index >= head.output.shape.batch {
            return Err(Error::InvalidConfig(format!(
                "batch index {batch_index} is outside YOLO head batch {}",
                head.output.shape.batch
            )));
        }
        let sample_len = head
            .output
            .shape
            .width
            .checked_mul(head.output.shape.height)
            .and_then(|value| value.checked_mul(spec.mask.len()))
            .and_then(|value| value.checked_mul(spec.classes + 5))
            .ok_or_else(|| Error::InvalidConfig("YOLO head size overflow".to_owned()))?;
        let expected_len = sample_len
            .checked_mul(head.output.shape.batch)
            .ok_or_else(|| Error::InvalidConfig("YOLO head batch size overflow".to_owned()))?;
        if head.output.data.len() != expected_len {
            return Err(Error::InvalidConfig(format!(
                "YOLO head tensor length mismatch: expected {expected_len}, got {}",
                head.output.data.len()
            )));
        }
        let start = batch_index
            .checked_mul(sample_len)
            .ok_or_else(|| Error::InvalidConfig("YOLO head batch offset overflow".to_owned()))?;
        let output = Tensor::from_data(
            TensorShape::new(
                head.output.shape.width,
                head.output.shape.height,
                head.output.shape.channels,
                1,
            ),
            head.output.data[start..start + sample_len].to_vec(),
        )?;
        yolo_heads.push(YoloHeadOutput {
            layer_index: head.layer_index,
            output,
        });
    }
    let mut modern_heads = Vec::with_capacity(forward.modern_heads.len());
    for head in &forward.modern_heads {
        if batch_index >= head.output.shape.batch {
            return Err(Error::InvalidConfig(format!(
                "batch index {batch_index} is outside modern head batch {}",
                head.output.shape.batch
            )));
        }
        let sample_len = head.output.shape.element_count();
        let expected_len = sample_len
            .checked_mul(head.output.shape.batch)
            .ok_or_else(|| Error::InvalidConfig("modern head batch size overflow".to_owned()))?;
        if head.output.data.len() != expected_len {
            return Err(Error::InvalidConfig(format!(
                "modern head tensor length mismatch: expected {expected_len}, got {}",
                head.output.data.len()
            )));
        }
        let start = batch_index
            .checked_mul(sample_len)
            .ok_or_else(|| Error::InvalidConfig("modern head batch offset overflow".to_owned()))?;
        let output = Tensor::from_data(
            TensorShape::new(
                head.output.shape.width,
                head.output.shape.height,
                head.output.shape.channels,
                1,
            ),
            head.output.data[start..start + sample_len].to_vec(),
        )?;
        modern_heads.push(ModernHeadOutput {
            layer_index: head.layer_index,
            output,
        });
    }

    let mut detections = decode_yolo_detections(
        network,
        &yolo_heads,
        image_width,
        image_height,
        head_options,
    )?;
    detections.extend(decode_modern_detections(
        network,
        &modern_heads,
        image_width,
        image_height,
        head_options,
    )?);
    inverse_hyperbolic_foveation_detections(
        network,
        forward,
        batch_index,
        image_width,
        image_height,
        &mut detections,
    )?;
    apply_nms(&mut detections, options.nms_threshold);
    detections.sort_by(|lhs, rhs| rhs.confidence.total_cmp(&lhs.confidence));
    Ok(detections)
}

fn inverse_hyperbolic_foveation_detections(
    network: &Network,
    forward: &ForwardResult,
    batch_index: usize,
    image_width: usize,
    image_height: usize,
    detections: &mut [Detection],
) -> Result<()> {
    let Some((spec, params)) =
        hyperbolic_foveation_detection_params(network, forward, batch_index)?
    else {
        return Ok(());
    };
    for detection in detections {
        let normalized = [
            detection.bbox.x / image_width as f32,
            detection.bbox.y / image_height as f32,
            detection.bbox.w / image_width as f32,
            detection.bbox.h / image_height as f32,
        ];
        let transformed = crate::hyperbolic_foveation::transform_box_with_jacobian(
            normalized,
            params,
            true,
            spec.inverse_method,
            spec.nr_iters,
            spec.nr_eta,
        );
        detection.bbox = DetectionBox {
            x: transformed[0] * image_width as f32,
            y: transformed[1] * image_height as f32,
            w: transformed[2] * image_width as f32,
            h: transformed[3] * image_height as f32,
        };
    }
    Ok(())
}

fn hyperbolic_foveation_detection_params(
    network: &Network,
    forward: &ForwardResult,
    batch_index: usize,
) -> Result<
    Option<(
        HyperbolicFoveationSpec,
        crate::hyperbolic_foveation::HfParams,
    )>,
> {
    let Some(layer) = network
        .layers
        .iter()
        .rev()
        .find(|layer| matches!(layer.kind, LayerKind::HyperbolicFoveation(_)))
    else {
        return Ok(None);
    };
    let LayerKind::HyperbolicFoveation(spec) = &layer.kind else {
        unreachable!("filtered hyperbolic-foveation layer")
    };
    let cache = forward
        .hyperbolic_foveation_caches
        .get(layer.index)
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing [hyperbolic_foveation] detection cache for layer #{}",
                layer.index
            ))
        })?;
    let param_count = spec.param_count();
    let start = batch_index.checked_mul(param_count).ok_or_else(|| {
        Error::InvalidConfig("hyperbolic-foveation detection batch offset overflow".to_owned())
    })?;
    let end = start.checked_add(param_count).ok_or_else(|| {
        Error::InvalidConfig("hyperbolic-foveation detection range overflow".to_owned())
    })?;
    let values = cache.params_per_batch.get(start..end).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "hyperbolic-foveation detection cache has no parameters for batch {batch_index}"
        ))
    })?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidConfig(
            "hyperbolic-foveation detection parameters are non-finite".to_owned(),
        ));
    }
    let (radius_x, radius_y, strength_index) = match spec.radius_mode {
        HfRadiusMode::Isotropic => (values[2], values[2], 3),
        HfRadiusMode::Anisotropic => (values[2], values[3], 4),
    };
    let params = crate::hyperbolic_foveation::HfParams {
        center_x: values[0].clamp(-0.9, 0.9),
        center_y: values[1].clamp(-0.9, 0.9),
        radius_x: radius_x.clamp(0.1, 2.0),
        radius_y: radius_y.clamp(0.1, 2.0),
        strength: values[strength_index].clamp(0.0, 1.0),
        alpha: spec.alpha,
        p_exp: spec.p_exp,
        radius_mode: spec.radius_mode,
    };
    Ok(Some((spec.clone(), params)))
}

fn inverse_hyperbolic_foveation_point(
    network: &Network,
    forward: &ForwardResult,
    batch_index: usize,
    image_width: usize,
    image_height: usize,
    point: [f32; 2],
) -> Result<[f32; 2]> {
    let Some((spec, params)) =
        hyperbolic_foveation_detection_params(network, forward, batch_index)?
    else {
        return Ok(point);
    };
    let normalized = [
        point[0] / image_width as f32,
        point[1] / image_height as f32,
    ];
    let poincare = [2.0 * normalized[0] - 1.0, 2.0 * normalized[1] - 1.0];
    let restored = crate::hyperbolic_foveation::inverse_point(
        poincare,
        params,
        spec.inverse_method,
        spec.nr_iters,
        spec.nr_eta,
    )
    .point;
    Ok([
        (0.5 * (restored[0] + 1.0)) * image_width as f32,
        (0.5 * (restored[1] + 1.0)) * image_height as f32,
    ])
}

/// Decode and merge the network's legacy `[obb_yolo]` heads, then run
/// orientation-aware NMS once across all heads. This API currently supports
/// batch size one. OBB training and network-level mAP remain unsupported;
/// [`crate::evaluate::obb_detector_map`] only scores caller-provided samples.
pub fn decode_obb_yolo_detections(
    network: &Network,
    obb_heads: &[YoloHeadOutput],
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<ObbDetection>> {
    decode_obb_yolo_detections_internal(
        network,
        obb_heads,
        image_width,
        image_height,
        options,
        None,
    )
}

fn decode_obb_yolo_detections_internal(
    network: &Network,
    obb_heads: &[YoloHeadOutput],
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
    foveation_forward: Option<&ForwardResult>,
) -> Result<Vec<ObbDetection>> {
    let mut detections = Vec::new();
    let mut nms_kind = None;
    let mut beta_nms = None;
    for head in obb_heads {
        let layer = network.layers.get(head.layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing layer #{} for OBB YOLO head",
                head.layer_index
            ))
        })?;
        let spec = match &layer.kind {
            LayerKind::ObbYolo(spec) => spec,
            _ => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} is not an OBB YOLO head",
                    head.layer_index
                )));
            }
        };
        if head.output.shape.batch != 1 {
            return Err(Error::InvalidConfig(format!(
                "OBB YOLO decode currently requires batch size one, got {}",
                head.output.shape.batch
            )));
        }
        if nms_kind.is_some_and(|kind| kind != spec.nms_kind) {
            return Err(Error::InvalidConfig(
                "OBB YOLO heads use different NMS kinds".to_owned(),
            ));
        }
        if matches!(spec.nms_kind, ObbNmsKind::Diou | ObbNmsKind::Weighted) {
            if beta_nms.is_some_and(|beta| beta != spec.beta_nms) {
                return Err(Error::InvalidConfig(
                    "OBB YOLO heads use different beta_nms values".to_owned(),
                ));
            }
            beta_nms = Some(spec.beta_nms);
        }
        nms_kind = Some(spec.nms_kind);
        detections.extend(decode_obb_yolo_head(
            &head.output.data,
            head.output.shape.width,
            head.output.shape.height,
            spec,
            network.net.width,
            network.net.height,
            image_width,
            image_height,
            options,
        )?);
    }

    if let Some(forward) = foveation_forward {
        inverse_hyperbolic_foveation_obb_detections(
            network,
            forward,
            image_width,
            image_height,
            &mut detections,
        )?;
    }
    apply_obb_nms_by_kind(
        &mut detections,
        options.nms_threshold,
        nms_kind.unwrap_or(ObbNmsKind::Obb),
        beta_nms.unwrap_or(0.6),
    );
    detections.sort_by(|lhs, rhs| {
        rhs.detection
            .confidence
            .total_cmp(&lhs.detection.confidence)
    });
    Ok(detections)
}

/// Decode OBB detections from a complete network forward pass.
///
/// Networks with `[obb_refine]` use only typed Stage-2 outputs and return CARPN
/// quadrilaterals; legacy networks use the shared OBB YOLO head decoder.
/// This CPU decoder currently supports batch size one.
pub fn decode_obb_network_detections(
    network: &Network,
    forward: &ForwardResult,
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<ObbDetection>> {
    if image_width == 0 || image_height == 0 || network.net.width == 0 || network.net.height == 0 {
        return Err(Error::InvalidConfig(
            "invalid image or network geometry for OBB decoding".to_owned(),
        ));
    }
    let refine_layers = network
        .layers
        .iter()
        .filter(|layer| matches!(layer.kind, LayerKind::ObbRefine(_)))
        .collect::<Vec<_>>();
    if refine_layers.is_empty() {
        return decode_obb_yolo_detections_internal(
            network,
            &forward.yolo_heads,
            image_width,
            image_height,
            options,
            Some(forward),
        );
    }
    if forward.obb_refine_outputs.len() != refine_layers.len() {
        return Err(Error::InvalidConfig(format!(
            "CARPN forward result has {} refine outputs, expected {}",
            forward.obb_refine_outputs.len(),
            refine_layers.len()
        )));
    }

    let correction = ObbImageCorrection::new(
        network.net.width,
        network.net.height,
        image_width,
        image_height,
        options.letterbox,
    )?;
    let mut detections = Vec::new();
    let mut nms_kind = None;
    let mut beta_nms = None;
    for (layer, refine_output) in refine_layers.into_iter().zip(&forward.obb_refine_outputs) {
        let LayerKind::ObbRefine(spec) = &layer.kind else {
            unreachable!("filtered refine layers")
        };
        validate_refine_output(layer.index, spec, refine_output, forward)?;
        if nms_kind.is_some_and(|kind| kind != spec.nms_kind) {
            return Err(Error::InvalidConfig(
                "CARPN refine layers use different NMS kinds".to_owned(),
            ));
        }
        if matches!(spec.nms_kind, ObbNmsKind::Diou | ObbNmsKind::Weighted) {
            if beta_nms.is_some_and(|beta| beta != spec.beta_nms) {
                return Err(Error::InvalidConfig(
                    "CARPN refine layers use different beta_nms values".to_owned(),
                ));
            }
            beta_nms = Some(spec.beta_nms);
        }
        nms_kind = Some(spec.nms_kind);

        let output_tensor = &forward.layer_outputs[layer.index];
        for proposal in &refine_output.proposals {
            if proposal.class_id >= spec.classes {
                return Err(Error::InvalidConfig(format!(
                    "CARPN proposal class {} is outside configured class count {}",
                    proposal.class_id, spec.classes
                )));
            }
            let capacity = spec.proposal_capacity;
            let raw = std::array::from_fn(|channel| {
                output_tensor.data[channel * capacity + proposal.proposal_index]
            });
            if raw != proposal.raw_regression {
                return Err(Error::InvalidConfig(format!(
                    "CARPN layer #{} typed proposal {} disagrees with its raw output tensor",
                    layer.index, proposal.proposal_index
                )));
            }
            if let Some(detection) = decode_carpn_proposal(
                spec,
                proposal,
                raw,
                image_width,
                image_height,
                correction,
                options.confidence_threshold,
            )? {
                detections.push(detection);
            }
        }
    }

    inverse_hyperbolic_foveation_obb_detections(
        network,
        forward,
        image_width,
        image_height,
        &mut detections,
    )?;
    apply_obb_nms_by_kind(
        &mut detections,
        options.nms_threshold,
        nms_kind.unwrap_or(ObbNmsKind::Obb),
        beta_nms.unwrap_or(0.6),
    );
    detections.sort_by(|lhs, rhs| {
        rhs.detection
            .confidence
            .total_cmp(&lhs.detection.confidence)
    });
    Ok(detections)
}

fn inverse_hyperbolic_foveation_obb_detections(
    network: &Network,
    forward: &ForwardResult,
    image_width: usize,
    image_height: usize,
    detections: &mut [ObbDetection],
) -> Result<()> {
    let mut axis_aligned = detections
        .iter()
        .map(|detection| detection.detection.clone())
        .collect::<Vec<_>>();
    inverse_hyperbolic_foveation_detections(
        network,
        forward,
        0,
        image_width,
        image_height,
        &mut axis_aligned,
    )?;
    for (detection, transformed) in detections.iter_mut().zip(axis_aligned) {
        detection.detection.bbox = transformed.bbox;
        match &mut detection.geometry {
            ObbDetectionGeometry::FrontPoint { x, y } => {
                let point = inverse_hyperbolic_foveation_point(
                    network,
                    forward,
                    0,
                    image_width,
                    image_height,
                    [*x, *y],
                )?;
                *x = point[0];
                *y = point[1];
            }
            ObbDetectionGeometry::CarpnQuad(vertices) => {
                for vertex in 0..4 {
                    let point = inverse_hyperbolic_foveation_point(
                        network,
                        forward,
                        0,
                        image_width,
                        image_height,
                        [vertices[2 * vertex], vertices[2 * vertex + 1]],
                    )?;
                    vertices[2 * vertex] = point[0];
                    vertices[2 * vertex + 1] = point[1];
                }
            }
        }
    }
    Ok(())
}

fn validate_refine_output(
    layer_index: usize,
    spec: &ObbRefineSpec,
    output: &ObbRefineOutput,
    forward: &ForwardResult,
) -> Result<()> {
    if output.layer_index != layer_index
        || output.proposal_count != output.proposals.len()
        || output.proposal_count > spec.proposal_capacity
    {
        return Err(Error::InvalidConfig(format!(
            "invalid CARPN metadata for layer #{layer_index}"
        )));
    }
    let tensor = forward.layer_outputs.get(layer_index).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "CARPN forward result is missing layer #{layer_index} output"
        ))
    })?;
    if tensor.shape.width != spec.proposal_capacity
        || tensor.shape.height != 1
        || tensor.shape.channels != 5
        || tensor.shape.batch != 1
        || tensor.data.len() != spec.proposal_capacity * 5
    {
        return Err(Error::InvalidConfig(format!(
            "CARPN layer #{layer_index} has invalid raw output tensor shape {}",
            tensor.shape
        )));
    }
    for (index, proposal) in output.proposals.iter().enumerate() {
        if proposal.proposal_index != index || proposal.source_layer_index != spec.proposal_layer {
            return Err(Error::InvalidConfig(format!(
                "CARPN layer #{layer_index} has invalid proposal/source index metadata"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_carpn_proposal(
    spec: &ObbRefineSpec,
    proposal: &ObbRefineProposal,
    raw: [f32; 5],
    image_width: usize,
    image_height: usize,
    correction: ObbImageCorrection,
    confidence_threshold: f32,
) -> Result<Option<ObbDetection>> {
    if !proposal.score.is_finite() || proposal.score <= confidence_threshold {
        return Ok(None);
    }
    if !spec.front_range.is_finite() || spec.front_range <= 0.0 {
        return Err(Error::InvalidConfig(
            "CARPN front_range must be finite and > 0".to_owned(),
        ));
    }
    let anchor_radius = proposal.anchor_radius.max(1e-6);
    let dr = finite_or_zero(raw[2]).clamp(-5.0, 5.0);
    let dx = finite_or_zero(raw[0]).clamp(-spec.front_range, spec.front_range);
    let dy = finite_or_zero(raw[1]).clamp(-spec.front_range, spec.front_range);
    let dalpha = finite_or_zero(raw[3]).clamp(-spec.front_range, spec.front_range);
    let dbeta = finite_or_zero(raw[4]).clamp(-spec.front_range, spec.front_range);
    let center_x = proposal.anchor_x + dx * anchor_radius;
    let center_y = proposal.anchor_y + dy * anchor_radius;
    let radius = anchor_radius * dr.exp();
    if !center_x.is_finite() || !center_y.is_finite() || !radius.is_finite() || radius <= 0.0 {
        return Err(Error::InvalidConfig(
            "CARPN proposal has invalid decoded center or radius".to_owned(),
        ));
    }
    let dalpha = (dalpha * radius).clamp(-radius, radius);
    let dbeta = (dbeta * radius).clamp(-radius, radius);
    let perp_alpha = (radius * radius - dalpha * dalpha).max(0.0).sqrt();
    let perp_beta = (radius * radius - dbeta * dbeta).max(0.0).sqrt();
    let normalized_vertices = [
        center_x + dalpha,
        center_y - perp_alpha,
        center_x + perp_beta,
        center_y + dbeta,
        center_x - dalpha,
        center_y + perp_alpha,
        center_x - perp_beta,
        center_y - dbeta,
    ];
    let mut vertices = [0.0f32; 8];
    for vertex in 0..4 {
        let corrected = correction.correct_front_point(
            [
                normalized_vertices[vertex * 2],
                normalized_vertices[vertex * 2 + 1],
            ],
            image_width as f32,
            image_height as f32,
        );
        vertices[vertex * 2] = corrected[0];
        vertices[vertex * 2 + 1] = corrected[1];
    }
    if vertices.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(Error::InvalidConfig(
            "CARPN proposal decodes to non-finite vertices".to_owned(),
        ));
    }

    let mut min_x = vertices[0];
    let mut max_x = vertices[0];
    let mut min_y = vertices[1];
    let mut max_y = vertices[1];
    for vertex in 1..4 {
        let x = vertices[vertex * 2];
        let y = vertices[vertex * 2 + 1];
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let bbox = DetectionBox {
        x: 0.5 * (min_x + max_x),
        y: 0.5 * (min_y + max_y),
        w: (max_x - min_x).max(1e-6),
        h: (max_y - min_y).max(1e-6),
    };
    Ok(Some(ObbDetection {
        detection: Detection {
            class_id: proposal.class_id,
            confidence: proposal.score,
            objectness: proposal.objectness,
            bbox,
        },
        geometry: ObbDetectionGeometry::CarpnQuad(vertices),
    }))
}

/// Decode one six-coordinate legacy OBB YOLO head into per-class detections.

/// `data` is one image's output in the existing Darknet channel-major layout
/// with six box channels, one objectness channel, and then `classes` channels
/// per anchor. The returned front point is letterbox-corrected and expressed in
/// image pixels, like the
/// bbox. This per-head helper does not run NMS; merge head outputs first, then
/// call [`apply_obb_nms`].
#[allow(clippy::too_many_arguments)]
pub fn decode_obb_yolo_head(
    data: &[f32],
    grid_width: usize,
    grid_height: usize,
    spec: &YoloSpec,
    net_width: usize,
    net_height: usize,
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<ObbDetection>> {
    if grid_width == 0
        || grid_height == 0
        || net_width == 0
        || net_height == 0
        || image_width == 0
        || image_height == 0
        || spec.classes == 0
        || spec.mask.is_empty()
        || !spec.scale_x_y.is_finite()
    {
        return Err(Error::InvalidConfig(
            "invalid geometry or spec for OBB YOLO decoding".to_owned(),
        ));
    }

    let stride = grid_width
        .checked_mul(grid_height)
        .ok_or_else(|| Error::InvalidConfig("OBB YOLO grid size overflow".to_owned()))?;
    let entries_per_anchor = spec
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("OBB YOLO entry count overflow".to_owned()))?;
    let expected = stride
        .checked_mul(spec.mask.len())
        .and_then(|size| size.checked_mul(entries_per_anchor))
        .ok_or_else(|| Error::InvalidConfig("OBB YOLO head size overflow".to_owned()))?;
    if data.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "OBB YOLO head tensor length mismatch: expected {expected}, got {}",
            data.len()
        )));
    }

    let correction = ObbImageCorrection::new(
        net_width,
        net_height,
        image_width,
        image_height,
        options.letterbox,
    )?;
    let mut detections = Vec::new();
    for index in 0..stride {
        let row = index / grid_width;
        let col = index % grid_width;
        for (anchor_offset, &mask_idx) in spec.mask.iter().enumerate() {
            let anchor = anchor_pair(spec, mask_idx)?;
            if !anchor.0.is_finite() || !anchor.1.is_finite() || anchor.0 <= 0.0 || anchor.1 <= 0.0
            {
                return Err(Error::InvalidConfig(format!(
                    "invalid OBB YOLO anchor for mask index {mask_idx}"
                )));
            }

            let anchor_base = anchor_offset * entries_per_anchor;
            let objectness = finite_or_zero(head_activation(
                data[(anchor_base + 6) * stride + index],
                spec.new_coords,
            ));
            if objectness <= options.confidence_threshold {
                continue;
            }

            let bbox = decode_obb_bbox(
                data[(anchor_base + 0) * stride + index],
                data[(anchor_base + 1) * stride + index],
                data[(anchor_base + 2) * stride + index],
                data[(anchor_base + 3) * stride + index],
                col,
                row,
                grid_width,
                grid_height,
                net_width,
                net_height,
                anchor,
                spec.scale_x_y,
                spec.new_coords,
            );
            let bbox = correction.correct_bbox(bbox, image_width as f32, image_height as f32);

            // Source channels 4/5 are raw, linear, cell-relative front offsets.
            let front_x = finite_or_zero(data[(anchor_base + 4) * stride + index]);
            let front_y = finite_or_zero(data[(anchor_base + 5) * stride + index]);
            let front_x = (col as f32 + front_x) / grid_width as f32;
            let front_y = (row as f32 + front_y) / grid_height as f32;
            let [front_x, front_y] = correction.correct_front_point(
                [front_x, front_y],
                image_width as f32,
                image_height as f32,
            );

            for class_id in 0..spec.classes {
                let class_probability = finite_or_zero(head_activation(
                    data[(anchor_base + 7 + class_id) * stride + index],
                    spec.new_coords,
                ));
                let confidence = objectness * class_probability;
                if confidence > options.confidence_threshold {
                    detections.push(ObbDetection {
                        detection: Detection {
                            class_id,
                            confidence,
                            objectness,
                            bbox,
                        },
                        geometry: ObbDetectionGeometry::FrontPoint {
                            x: front_x,
                            y: front_y,
                        },
                    });
                }
            }
        }
    }
    Ok(detections)
}

#[allow(clippy::too_many_arguments)]
fn decode_yolo_head(
    data: &[f32],
    grid_width: usize,
    grid_height: usize,
    spec: &YoloSpec,
    net_width: usize,
    net_height: usize,
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
    out: &mut Vec<Detection>,
) -> Result<()> {
    let anchors_per_cell = spec.mask.len();
    let entries_per_anchor = spec.classes + 5;
    let expected = grid_width
        .checked_mul(grid_height)
        .and_then(|v| v.checked_mul(anchors_per_cell))
        .and_then(|v| v.checked_mul(entries_per_anchor))
        .ok_or_else(|| Error::InvalidConfig("YOLO head size overflow".to_owned()))?;
    if data.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "YOLO head tensor length mismatch: expected {expected}, got {}",
            data.len()
        )));
    }

    let stride = grid_width * grid_height;
    for (anchor_offset, &mask_idx) in spec.mask.iter().enumerate() {
        let anchor = anchor_pair(spec, mask_idx)?;
        let anchor_base = anchor_offset * entries_per_anchor;
        for index in 0..stride {
            let row = index / grid_width;
            let col = index % grid_width;
            let tx = data[anchor_base * stride + index];
            let ty = data[(anchor_base + 1) * stride + index];
            let tw = data[(anchor_base + 2) * stride + index];
            let th = data[(anchor_base + 3) * stride + index];
            let objectness =
                head_activation(data[(anchor_base + 4) * stride + index], spec.new_coords);

            if objectness <= options.confidence_threshold {
                continue;
            }

            let bbox = decode_bbox(
                tx,
                ty,
                tw,
                th,
                col,
                row,
                grid_width,
                grid_height,
                net_width,
                net_height,
                anchor,
                spec.scale_x_y,
                spec.new_coords,
            );
            let bbox = correct_bbox(
                bbox,
                image_width as f32,
                image_height as f32,
                net_width as f32,
                net_height as f32,
                options.letterbox,
            );

            for class_id in 0..spec.classes {
                let class_prob = head_activation(
                    data[(anchor_base + 5 + class_id) * stride + index],
                    spec.new_coords,
                );
                let confidence = objectness * class_prob;
                if confidence > options.confidence_threshold {
                    out.push(Detection {
                        class_id,
                        confidence,
                        objectness,
                        bbox,
                    });
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Decode the modern anchor-free heads (`[yolox]`, `[ppyoloe]`, `[yolonas]`,
/// `[centernet]`) into the same [`Detection`] shape as the yolo head, so
/// `detect-image` and `evaluate.rs` work unchanged on a modern cfg.
///
/// * YOLOX scores `obj * cls` and regresses cxcywh directly.
/// * PP-YOLOE / YOLO-NAS score `obj * cls` on the DFL-decoded box.
/// * CenterNet scores the sigmoid heatmap against its peak (`peak_nms`) and uses
///   the clamped sub-cell offset.
///
/// Letterbox correction, NMS and sorting run once over the merged list, exactly
/// as for the yolo head.
pub fn decode_modern_detections(
    network: &Network,
    modern_heads: &[ModernHeadOutput],
    image_width: usize,
    image_height: usize,
    options: DetectionOptions,
) -> Result<Vec<Detection>> {
    let mut detections = Vec::new();
    for head in modern_heads {
        let layer = network.layers.get(head.layer_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing layer #{} for a modern head",
                head.layer_index
            ))
        })?;
        let (w, h, classes) = (
            head.output.shape.width,
            head.output.shape.height,
            head.output.shape.channels,
        );
        let grid_w = w as f32;
        let grid_h = h as f32;
        match &layer.kind {
            LayerKind::Yolox(spec) => {
                let layout =
                    crate::modern_yolo::ModernLayout::for_spec(head.output.shape.batch, w, h, spec);
                if head.output.data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "yolox head tensor length mismatch: expected {}, got {}",
                        layout.len(),
                        head.output.data.len()
                    )));
                }
                let stride = layout.spatial();
                for n in 0..layout.n {
                    for loc in 0..stride {
                        let obj = crate::modern_yolo::modern_objectness(
                            &head.output.data,
                            &layout,
                            0,
                            n,
                            loc,
                        );
                        for c in 0..layout.classes {
                            let cls = crate::modern_yolo::modern_class_score(
                                &head.output.data,
                                &layout,
                                0,
                                n,
                                loc,
                                c,
                            );
                            let score = obj * cls;
                            if score <= options.confidence_threshold {
                                continue;
                            }
                            let box_ = crate::modern_yolo::decode_modern_box(
                                &head.output.data,
                                &layout,
                                0,
                                n,
                                loc,
                            );
                            detections.push(Detection {
                                class_id: c,
                                confidence: score,
                                objectness: obj,
                                bbox: correct_bbox(
                                    DetectionBox {
                                        x: box_.x,
                                        y: box_.y,
                                        w: box_.w,
                                        h: box_.h,
                                    },
                                    image_width as f32,
                                    image_height as f32,
                                    network.net.width as f32,
                                    network.net.height as f32,
                                    options.letterbox,
                                ),
                            });
                        }
                    }
                }
            }
            LayerKind::Ppyoloe(spec) | LayerKind::Yolonas(spec) => {
                let layout =
                    crate::modern_yolo::ModernLayout::for_spec(head.output.shape.batch, w, h, spec);
                if head.output.data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "distribution head tensor length mismatch: expected {}, got {}",
                        layout.len(),
                        head.output.data.len()
                    )));
                }
                let stride = layout.spatial();
                for n in 0..layout.n {
                    for loc in 0..stride {
                        let obj = crate::modern_yolo::modern_objectness(
                            &head.output.data,
                            &layout,
                            0,
                            n,
                            loc,
                        );
                        for c in 0..layout.classes {
                            let cls = crate::modern_yolo::modern_class_score(
                                &head.output.data,
                                &layout,
                                0,
                                n,
                                loc,
                                c,
                            );
                            // The distribution heads gate on the class score, not
                            // on the product, because the objectness logit is
                            // trained towards a soft quality target that is 0 for
                            // most of the map.
                            let score = cls.max(obj * cls);
                            if score <= options.confidence_threshold {
                                continue;
                            }
                            let box_ = crate::modern_yolo::decode_modern_box(
                                &head.output.data,
                                &layout,
                                0,
                                n,
                                loc,
                            );
                            detections.push(Detection {
                                class_id: c,
                                confidence: score,
                                objectness: obj,
                                bbox: correct_bbox(
                                    DetectionBox {
                                        x: box_.x,
                                        y: box_.y,
                                        w: box_.w,
                                        h: box_.h,
                                    },
                                    image_width as f32,
                                    image_height as f32,
                                    network.net.width as f32,
                                    network.net.height as f32,
                                    options.letterbox,
                                ),
                            });
                        }
                    }
                }
            }
            LayerKind::CenterNet(spec) => {
                let layout = crate::centernet::CenterNetLayout::for_spec(
                    head.output.shape.batch,
                    w,
                    h,
                    spec,
                );
                if head.output.data.len() != layout.len() {
                    return Err(Error::InvalidConfig(format!(
                        "centernet tensor length mismatch: expected {}, got {}",
                        layout.len(),
                        head.output.data.len()
                    )));
                }
                let stride = layout.spatial();
                for n in 0..layout.batch {
                    for c in 0..layout.classes {
                        for loc in 0..stride {
                            if spec.peak_nms != 0
                                && !crate::centernet::is_local_peak(
                                    &head.output.data,
                                    &layout,
                                    n,
                                    c,
                                    loc,
                                )
                            {
                                continue;
                            }
                            let score = crate::centernet::heatmap_score(
                                &head.output.data,
                                &layout,
                                n,
                                c,
                                loc,
                            );
                            if score <= options.confidence_threshold {
                                continue;
                            }
                            let (bx, by, bw, bh) = crate::centernet::decode_centernet_box(
                                &head.output.data,
                                &layout,
                                n,
                                loc,
                            );
                            detections.push(Detection {
                                class_id: c,
                                confidence: score,
                                objectness: score,
                                bbox: correct_bbox(
                                    DetectionBox {
                                        x: bx,
                                        y: by,
                                        w: bw,
                                        h: bh,
                                    },
                                    image_width as f32,
                                    image_height as f32,
                                    network.net.width as f32,
                                    network.net.height as f32,
                                    options.letterbox,
                                ),
                            });
                        }
                    }
                }
            }
            other => {
                return Err(Error::InvalidConfig(format!(
                    "layer #{} ({}) is not a modern anchor-free head",
                    head.layer_index,
                    other.name()
                )));
            }
        }
        let _ = (grid_w, grid_h, classes);
    }
    apply_nms(&mut detections, options.nms_threshold);
    detections.sort_by(|lhs, rhs| rhs.confidence.total_cmp(&lhs.confidence));
    Ok(detections)
}

fn decode_bbox(
    tx: f32,
    ty: f32,
    tw: f32,
    th: f32,
    col: usize,
    row: usize,
    grid_width: usize,
    grid_height: usize,
    net_width: usize,
    net_height: usize,
    anchor: (f32, f32),
    scale_x_y: f32,
    new_coords: bool,
) -> DetectionBox {
    let bx = (col as f32 + scale_x_y * head_activation(tx, new_coords) - 0.5 * (scale_x_y - 1.0))
        / grid_width as f32;
    let by = (row as f32 + scale_x_y * head_activation(ty, new_coords) - 0.5 * (scale_x_y - 1.0))
        / grid_height as f32;
    let (bw, bh) = if new_coords {
        (
            tw * tw * 4.0 * anchor.0 / net_width as f32,
            th * th * 4.0 * anchor.1 / net_height as f32,
        )
    } else {
        (
            tw.exp() * anchor.0 / net_width as f32,
            th.exp() * anchor.1 / net_height as f32,
        )
    };

    DetectionBox {
        x: bx,
        y: by,
        w: bw,
        h: bh,
    }
}

pub(crate) fn decode_obb_bbox(
    tx: f32,
    ty: f32,
    tw: f32,
    th: f32,
    col: usize,
    row: usize,
    grid_width: usize,
    grid_height: usize,
    net_width: usize,
    net_height: usize,
    anchor: (f32, f32),
    scale_x_y: f32,
    new_coords: bool,
) -> DetectionBox {
    let scale_offset = 0.5 * (scale_x_y - 1.0);
    let tx = finite_or_zero(scale_x_y * head_activation(tx, new_coords) - scale_offset);
    let ty = finite_or_zero(scale_x_y * head_activation(ty, new_coords) - scale_offset);
    let tw = finite_or_zero(tw);
    let th = finite_or_zero(th);
    let (width, height) = if new_coords {
        (
            tw * tw * 4.0 * anchor.0 / net_width as f32,
            th * th * 4.0 * anchor.1 / net_height as f32,
        )
    } else {
        (
            tw.clamp(-10.0, 10.0).exp() * anchor.0 / net_width as f32,
            th.clamp(-10.0, 10.0).exp() * anchor.1 / net_height as f32,
        )
    };
    DetectionBox {
        x: finite_or_zero((col as f32 + tx) / grid_width as f32),
        y: finite_or_zero((row as f32 + ty) / grid_height as f32),
        w: finite_or_zero(width).max(1e-6),
        h: finite_or_zero(height).max(1e-6),
    }
}

pub(crate) fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

#[derive(Clone, Copy, Debug)]
struct ObbImageCorrection {
    delta_width: f32,
    delta_height: f32,
    ratio_width: f32,
    ratio_height: f32,
    net_width: f32,
    net_height: f32,
}

impl ObbImageCorrection {
    fn new(
        net_width: usize,
        net_height: usize,
        image_width: usize,
        image_height: usize,
        letterbox: bool,
    ) -> Result<Self> {
        let (new_width, new_height) = if letterbox {
            if net_width as f32 / (image_width as f32) < net_height as f32 / (image_height as f32) {
                (
                    net_width,
                    image_height.checked_mul(net_width).ok_or_else(|| {
                        Error::InvalidConfig("OBB letterbox height overflow".to_owned())
                    })? / image_width,
                )
            } else {
                (
                    image_width.checked_mul(net_height).ok_or_else(|| {
                        Error::InvalidConfig("OBB letterbox width overflow".to_owned())
                    })? / image_height,
                    net_height,
                )
            }
        } else {
            (net_width, net_height)
        };
        if new_width == 0 || new_height == 0 {
            return Err(Error::InvalidConfig(
                "OBB letterbox dimensions collapse to zero".to_owned(),
            ));
        }

        let (net_width, net_height) = (net_width as f32, net_height as f32);
        let (new_width, new_height) = (new_width as f32, new_height as f32);
        Ok(Self {
            delta_width: net_width - new_width,
            delta_height: net_height - new_height,
            ratio_width: new_width / net_width,
            ratio_height: new_height / net_height,
            net_width,
            net_height,
        })
    }

    fn correct_bbox(
        self,
        mut bbox: DetectionBox,
        image_width: f32,
        image_height: f32,
    ) -> DetectionBox {
        bbox.x = (bbox.x - self.delta_width / 2.0 / self.net_width) / self.ratio_width;
        bbox.y = (bbox.y - self.delta_height / 2.0 / self.net_height) / self.ratio_height;
        bbox.w /= self.ratio_width;
        bbox.h /= self.ratio_height;
        bbox.x *= image_width;
        bbox.y *= image_height;
        bbox.w *= image_width;
        bbox.h *= image_height;
        bbox
    }

    fn correct_front_point(
        self,
        mut front: [f32; 2],
        image_width: f32,
        image_height: f32,
    ) -> [f32; 2] {
        front[0] = (front[0] - self.delta_width / 2.0 / self.net_width) / self.ratio_width;
        front[1] = (front[1] - self.delta_height / 2.0 / self.net_height) / self.ratio_height;
        front[0] *= image_width;
        front[1] *= image_height;
        front
    }
}

fn correct_bbox(
    mut bbox: DetectionBox,
    image_width: f32,
    image_height: f32,
    net_width: f32,
    net_height: f32,
    letterbox: bool,
) -> DetectionBox {
    let (new_w, new_h) = if letterbox {
        if net_width / image_width < net_height / image_height {
            (net_width, image_height * net_width / image_width)
        } else {
            (image_width * net_height / image_height, net_height)
        }
    } else {
        (net_width, net_height)
    };

    let delta_w = net_width - new_w;
    let delta_h = net_height - new_h;
    let ratio_w = new_w / net_width;
    let ratio_h = new_h / net_height;

    bbox.x = (bbox.x - delta_w / 2.0 / net_width) / ratio_w;
    bbox.y = (bbox.y - delta_h / 2.0 / net_height) / ratio_h;
    bbox.w /= ratio_w;
    bbox.h /= ratio_h;

    bbox.x *= image_width;
    bbox.y *= image_height;
    bbox.w *= image_width;
    bbox.h *= image_height;
    bbox
}

fn apply_nms(detections: &mut Vec<Detection>, threshold: f32) {
    if threshold <= 0.0 {
        return;
    }

    let mut suppressed = vec![false; detections.len()];
    for i in 0..detections.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if suppressed[j] || detections[i].class_id != detections[j].class_id {
                continue;
            }
            let (keep_idx, drop_idx) = if detections[i].confidence >= detections[j].confidence {
                (i, j)
            } else {
                (j, i)
            };
            if bbox_iou(detections[keep_idx].bbox, detections[drop_idx].bbox) > threshold {
                suppressed[drop_idx] = true;
            }
        }
    }

    let mut keep = 0usize;
    for (idx, &is_suppressed) in suppressed.iter().enumerate() {
        if !is_suppressed {
            if keep != idx {
                detections.swap(keep, idx);
            }
            keep += 1;
        }
    }
    detections.truncate(keep);
}

fn bbox_iou(lhs: DetectionBox, rhs: DetectionBox) -> f32 {
    let inter_left = lhs.left().max(rhs.left());
    let inter_top = lhs.top().max(rhs.top());
    let inter_right = lhs.right().min(rhs.right());
    let inter_bottom = lhs.bottom().min(rhs.bottom());

    let inter_w = (inter_right - inter_left).max(0.0);
    let inter_h = (inter_bottom - inter_top).max(0.0);
    let inter_area = inter_w * inter_h;
    if inter_area <= 0.0 {
        return 0.0;
    }

    let lhs_area = lhs.w.max(0.0) * lhs.h.max(0.0);
    let rhs_area = rhs.w.max(0.0) * rhs.h.max(0.0);
    inter_area / (lhs_area + rhs_area - inter_area)
}

/// Source `box_diou`/`box_diounms`: AABB IoU minus `(center_distance² / enclosing_diagonal²)^beta`.
fn bbox_diou(lhs: DetectionBox, rhs: DetectionBox, beta: f32) -> f32 {
    let enclosing_width = lhs.right().max(rhs.right()) - lhs.left().min(rhs.left());
    let enclosing_height = lhs.bottom().max(rhs.bottom()) - lhs.top().min(rhs.top());
    let enclosing_diagonal_squared =
        enclosing_width * enclosing_width + enclosing_height * enclosing_height;
    let iou = bbox_iou(lhs, rhs);
    if enclosing_diagonal_squared == 0.0 {
        return iou;
    }
    let center_distance_squared = (lhs.x - rhs.x).powi(2) + (lhs.y - rhs.y).powi(2);
    iou - (center_distance_squared / enclosing_diagonal_squared).powf(beta)
}

/// Source `box_diou_penalty`, used with rotated polygon IoU in weighted OBB NMS.
fn bbox_diou_penalty(lhs: DetectionBox, rhs: DetectionBox, beta: f32) -> f32 {
    let enclosing_width = lhs.right().max(rhs.right()) - lhs.left().min(rhs.left());
    let enclosing_height = lhs.bottom().max(rhs.bottom()) - lhs.top().min(rhs.top());
    let enclosing_diagonal_squared =
        enclosing_width * enclosing_width + enclosing_height * enclosing_height;
    if enclosing_diagonal_squared <= 0.0 || !enclosing_diagonal_squared.is_finite() {
        return 0.0;
    }

    let center_distance_squared = (lhs.x - rhs.x).powi(2) + (lhs.y - rhs.y).powi(2);
    let penalty = (center_distance_squared / enclosing_diagonal_squared).powf(beta);
    if penalty.is_finite() { penalty } else { 0.0 }
}

/// IoU dispatch used by OBB post-processing.
///
/// Two CARPN quads use explicit polygon IoU. All other OBB pairs use the first
/// point pair as the legacy front point, matching `detection_obb_iou` when one
/// detection is CARPN and the other is not. This helper accepts only OBB records
/// and never silently falls back to AABB IoU.
pub fn detection_obb_iou(lhs: &ObbDetection, rhs: &ObbDetection) -> f32 {
    if let (
        ObbDetectionGeometry::CarpnQuad(lhs_vertices),
        ObbDetectionGeometry::CarpnQuad(rhs_vertices),
    ) = (lhs.geometry, rhs.geometry)
    {
        return carpn_polygon_iou(lhs_vertices, rhs_vertices);
    }

    let front_point = |geometry| match geometry {
        ObbDetectionGeometry::FrontPoint { x, y } => [x, y],
        ObbDetectionGeometry::CarpnQuad(vertices) => [vertices[0], vertices[1]],
    };
    let [lhs_front_x, lhs_front_y] = front_point(lhs.geometry);
    let [rhs_front_x, rhs_front_y] = front_point(rhs.geometry);
    box_obb_iou(
        BoxObb {
            bbox: lhs.detection.bbox,
            front_x: lhs_front_x,
            front_y: lhs_front_y,
        },
        BoxObb {
            bbox: rhs.detection.bbox,
            front_x: rhs_front_x,
            front_y: rhs_front_y,
        },
    )
}

/// Apply classwise greedy OBB NMS, sorting candidates by descending confidence.
///
/// `confidence` is the per-class probability for this class-specific record.
/// Exactly-zero objectness or confidence records remain in the output but do
/// not participate in suppression, matching `diounms_sort`. Suppression uses a
/// strict `IoU > threshold` comparison. Equal-score order is stable and defined
/// here because the C++ sorter does not specify tie order.
pub fn apply_obb_nms(detections: &mut Vec<ObbDetection>, threshold: f32) {
    apply_obb_nms_by_overlap(detections, threshold, detection_obb_iou);
}

fn apply_obb_nms_by_kind(
    detections: &mut Vec<ObbDetection>,
    threshold: f32,
    kind: ObbNmsKind,
    beta_nms: f32,
) {
    match kind {
        ObbNmsKind::Default | ObbNmsKind::Corners => {
            apply_obb_nms_by_overlap(detections, threshold, |lhs, rhs| {
                bbox_iou(lhs.detection.bbox, rhs.detection.bbox)
            })
        }
        ObbNmsKind::Greedy => apply_obb_nms_by_overlap(detections, threshold, |lhs, rhs| {
            bbox_diou(lhs.detection.bbox, rhs.detection.bbox, 0.6)
        }),
        ObbNmsKind::Diou => apply_obb_nms_by_overlap(detections, threshold, |lhs, rhs| {
            bbox_diou(lhs.detection.bbox, rhs.detection.bbox, beta_nms)
        }),
        ObbNmsKind::Obb => apply_obb_nms(detections, threshold),
        ObbNmsKind::Weighted => apply_obb_weighted_nms(detections, threshold, beta_nms),
    }
}

fn apply_obb_nms_by_overlap(
    detections: &mut Vec<ObbDetection>,
    threshold: f32,
    overlap: impl Fn(&ObbDetection, &ObbDetection) -> f32,
) {
    detections.sort_by(|lhs, rhs| {
        let lhs_inactive = lhs.detection.objectness == 0.0;
        let rhs_inactive = rhs.detection.objectness == 0.0;
        lhs_inactive.cmp(&rhs_inactive).then_with(|| {
            rhs.detection
                .confidence
                .total_cmp(&lhs.detection.confidence)
        })
    });

    let mut suppressed = vec![false; detections.len()];
    for i in 0..detections.len() {
        let keeper = &detections[i];
        if suppressed[i] || keeper.detection.objectness == 0.0 || keeper.detection.confidence == 0.0
        {
            continue;
        }

        for j in (i + 1)..detections.len() {
            let candidate = &detections[j];
            if suppressed[j]
                || keeper.detection.class_id != candidate.detection.class_id
                || candidate.detection.objectness == 0.0
                || candidate.detection.confidence == 0.0
            {
                continue;
            }
            if overlap(keeper, candidate) > threshold {
                suppressed[j] = true;
            }
        }
    }

    let mut kept = 0usize;
    for (index, &is_suppressed) in suppressed.iter().enumerate() {
        if !is_suppressed {
            if kept != index {
                detections.swap(kept, index);
            }
            kept += 1;
        }
    }
    detections.truncate(kept);
}

fn weighted_nms_active(detection: &ObbDetection) -> bool {
    detection.detection.objectness.is_finite()
        && detection.detection.objectness != 0.0
        && detection.detection.confidence.is_finite()
        && detection.detection.confidence != 0.0
}

fn obb_front_point(geometry: ObbDetectionGeometry) -> Option<(f32, f32)> {
    match geometry {
        ObbDetectionGeometry::FrontPoint { x, y } => Some((x, y)),
        ObbDetectionGeometry::CarpnQuad(vertices) => Some((vertices[0], vertices[1])),
    }
}

fn set_obb_front_point(geometry: &mut ObbDetectionGeometry, x: f32, y: f32) {
    match geometry {
        ObbDetectionGeometry::FrontPoint {
            x: front_x,
            y: front_y,
        } => {
            *front_x = x;
            *front_y = y;
        }
        ObbDetectionGeometry::CarpnQuad(vertices) => {
            vertices[0] = x;
            vertices[1] = y;
        }
    }
}

/// Source `OBB_W_NMS`: DIoU-penalized polygon suppression and confidence×IoU
/// weighted averaging of box coordinates and the keeper's front point.
fn apply_obb_weighted_nms(detections: &mut Vec<ObbDetection>, threshold: f32, beta_nms: f32) {
    detections.sort_by(|lhs, rhs| {
        let lhs_inactive = !weighted_nms_active(lhs);
        let rhs_inactive = !weighted_nms_active(rhs);
        lhs_inactive.cmp(&rhs_inactive).then_with(|| {
            rhs.detection
                .confidence
                .total_cmp(&lhs.detection.confidence)
        })
    });

    let mut suppressed = vec![false; detections.len()];
    for i in 0..detections.len() {
        if suppressed[i] || !weighted_nms_active(&detections[i]) {
            continue;
        }

        // The source accumulates from the unmodified keeper and writes the merge
        // only after scanning all lower-ranked detections.
        let keeper = detections[i].clone();
        let self_weight = keeper.detection.confidence;
        let mut coordinate_weight_sum = self_weight;
        let mut x_sum = keeper.detection.bbox.x * self_weight;
        let mut y_sum = keeper.detection.bbox.y * self_weight;
        let mut w_sum = keeper.detection.bbox.w * self_weight;
        let mut h_sum = keeper.detection.bbox.h * self_weight;
        let keeper_front = obb_front_point(keeper.geometry);
        let mut front_weight_sum = if keeper_front.is_some() {
            self_weight
        } else {
            0.0
        };
        let mut front_x_sum = keeper_front.map_or(0.0, |(x, _)| x * self_weight);
        let mut front_y_sum = keeper_front.map_or(0.0, |(_, y)| y * self_weight);

        for j in (i + 1)..detections.len() {
            if suppressed[j]
                || detections[j].detection.class_id != keeper.detection.class_id
                || !weighted_nms_active(&detections[j])
            {
                continue;
            }

            let obb_iou = detection_obb_iou(&keeper, &detections[j]);
            if !obb_iou.is_finite() {
                continue;
            }
            let overlap = obb_iou
                - bbox_diou_penalty(
                    keeper.detection.bbox,
                    detections[j].detection.bbox,
                    beta_nms,
                );
            if overlap > threshold {
                let candidate_weight = detections[j].detection.confidence * obb_iou;
                if candidate_weight.is_finite() {
                    let candidate_box = detections[j].detection.bbox;
                    coordinate_weight_sum += candidate_weight;
                    x_sum += candidate_box.x * candidate_weight;
                    y_sum += candidate_box.y * candidate_weight;
                    w_sum += candidate_box.w * candidate_weight;
                    h_sum += candidate_box.h * candidate_weight;
                    if keeper_front.is_some()
                        && let Some((front_x, front_y)) = obb_front_point(detections[j].geometry)
                    {
                        front_weight_sum += candidate_weight;
                        front_x_sum += front_x * candidate_weight;
                        front_y_sum += front_y * candidate_weight;
                    }
                }
                suppressed[j] = true;
            }
        }

        if coordinate_weight_sum.is_finite() && coordinate_weight_sum > 0.0 {
            let merged = [
                x_sum / coordinate_weight_sum,
                y_sum / coordinate_weight_sum,
                w_sum / coordinate_weight_sum,
                h_sum / coordinate_weight_sum,
            ];
            if merged.iter().all(|value| value.is_finite()) {
                let bbox = &mut detections[i].detection.bbox;
                bbox.x = merged[0];
                bbox.y = merged[1];
                bbox.w = merged[2];
                bbox.h = merged[3];
            }
        }
        if front_weight_sum.is_finite() && front_weight_sum > 0.0 {
            let merged_front = [
                front_x_sum / front_weight_sum,
                front_y_sum / front_weight_sum,
            ];
            if merged_front.iter().all(|value| value.is_finite()) {
                set_obb_front_point(
                    &mut detections[i].geometry,
                    merged_front[0],
                    merged_front[1],
                );
            }
        }
    }

    let mut kept = 0usize;
    for (index, &is_suppressed) in suppressed.iter().enumerate() {
        if !is_suppressed {
            if kept != index {
                detections.swap(kept, index);
            }
            kept += 1;
        }
    }
    detections.truncate(kept);
}

pub(crate) fn anchor_pair(spec: &YoloSpec, index: usize) -> Result<(f32, f32)> {
    let anchor_index = index
        .checked_mul(2)
        .ok_or_else(|| Error::InvalidConfig("anchor index overflow".to_owned()))?;
    let width = *spec.anchors.get(anchor_index).ok_or_else(|| {
        Error::InvalidConfig(format!("missing anchor width for YOLO mask index {index}"))
    })?;
    let height = *spec.anchors.get(anchor_index + 1).ok_or_else(|| {
        Error::InvalidConfig(format!("missing anchor height for YOLO mask index {index}"))
    })?;
    Ok((width, height))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

/// The runtime leaves the head tensor as the previous layer wrote it. Without `new_coords`
/// the C++ yolo layer applies the logistic to x, y, objectness and classes in forward;
/// with `new_coords` it does not (the preceding layer already activated them), so
/// `get_yolo_detections` reads the values as they are.
pub(crate) fn head_activation(raw: f32, new_coords: bool) -> f32 {
    if new_coords { raw } else { sigmoid(raw) }
}

#[cfg(test)]
mod tests {
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{Tensor, YoloHeadOutput};

    use super::{DetectionOptions, decode_yolo_detections};

    /// C++ `get_yolo_detections` + `get_yolo_box` with `new_coords=1`: head values are used as
    /// stored (already activated), x,y get `scale_x_y*v - 0.5*(scale_x_y-1)`, w = 4*tw^2*anchor/net.
    /// Grid 1x1, net 32, anchor 10x12, scale_x_y 1.2, tx=.5 ty=.25 tw=.5 th=.25, obj .9, cls .8:
    /// bx = 1.2*.5 - .1 = .5 -> 16 px; by = 1.2*.25 - .1 = .2 -> 6.4 px;
    /// w = 4*.25*10 = 10 px; h = 4*.0625*12 = 3 px; confidence = .9*.8 = .72.
    #[test]
    fn new_coords_head_is_decoded_without_sigmoid() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=32
                height=32
                channels=6

                [maxpool]
                size=32
                stride=32

                [yolo]
                mask=0
                anchors=10,12
                classes=1
                num=1
                scale_x_y=1.2
                new_coords=1
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let head = YoloHeadOutput {
            layer_index: 1,
            output: Tensor::from_data(
                network.layers[1].output_shape,
                vec![0.5, 0.25, 0.5, 0.25, 0.9, 0.8],
            )
            .expect("tensor"),
        };
        let detections = decode_yolo_detections(
            &network,
            &[head],
            32,
            32,
            DetectionOptions {
                confidence_threshold: 0.25,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .expect("decode should succeed");
        assert_eq!(detections.len(), 1);
        let d = &detections[0];
        assert!((d.confidence - 0.72).abs() < 1.0e-5, "{}", d.confidence);
        assert!((d.objectness - 0.9).abs() < 1.0e-6);
        assert!((d.bbox.x - 16.0).abs() < 1.0e-4, "{}", d.bbox.x);
        assert!((d.bbox.y - 6.4).abs() < 1.0e-4, "{}", d.bbox.y);
        assert!((d.bbox.w - 10.0).abs() < 1.0e-4, "{}", d.bbox.w);
        assert!((d.bbox.h - 3.0).abs() < 1.0e-4, "{}", d.bbox.h);
    }

    #[test]
    fn decodes_single_yolo_detection() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=32
                height=32
                channels=6

                [maxpool]
                size=32
                stride=32

                [yolo]
                mask=0
                anchors=10,12
                classes=1
                num=1
                scale_x_y=1.0
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let head = YoloHeadOutput {
            layer_index: 1,
            output: Tensor::from_data(
                network.layers[1].output_shape,
                vec![0.0, 0.0, 0.0, 0.0, 8.0, 8.0],
            )
            .expect("tensor"),
        };

        let detections = decode_yolo_detections(
            &network,
            &[head],
            32,
            32,
            DetectionOptions {
                confidence_threshold: 0.25,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .expect("decode should succeed");

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_id, 0);
        assert!(detections[0].confidence > 0.99);
        assert!((detections[0].bbox.x - 16.0).abs() < 0.01);
        assert!((detections[0].bbox.w - 10.0).abs() < 0.01);
    }

    #[test]
    fn suppresses_overlapping_boxes_with_nms() {
        let cfg = ConfigFile::parse_str(
            "inline.cfg",
            r#"
                [net]
                width=32
                height=32
                channels=12

                [maxpool]
                size=32
                stride=32

                [yolo]
                mask=0,1
                anchors=10,12,10,12
                classes=1
                num=2
            "#,
        )
        .expect("cfg should parse");
        let network = Network::from_config(cfg).expect("network should build");
        let mut data = vec![0.0; 12];
        data[4] = 8.0;
        data[5] = 8.0;
        data[10] = 7.0;
        data[11] = 7.0;
        let head = YoloHeadOutput {
            layer_index: 1,
            output: Tensor::from_data(network.layers[1].output_shape, data).expect("tensor"),
        };

        let detections =
            decode_yolo_detections(&network, &[head], 32, 32, DetectionOptions::default())
                .expect("decode should succeed");

        assert_eq!(detections.len(), 1);
    }
}

#[cfg(test)]
mod obb_decode_tests {
    use crate::model::{YoloIouLoss, YoloSpec};

    use super::{DetectionOptions, ObbDetectionGeometry, decode_obb_yolo_head};

    fn spec(new_coords: bool, scale_x_y: f32) -> YoloSpec {
        YoloSpec {
            classes: 2,
            num: 1,
            mask: vec![0],
            anchors: vec![10.0, 12.0],
            scale_x_y,
            new_coords,
            cls_normalizer: 1.0,
            obj_normalizer: 1.0,
            iou_normalizer: 1.0,
            ignore_thresh: 0.5,
            truth_thresh: 1.0,
            iou_thresh: 1.0,
            iou_loss: YoloIouLoss::Mse,
            iou_thresh_kind: YoloIouLoss::Iou,
            focal_loss: false,
            label_smooth_eps: 0.0,
            objectness_smooth: false,
            max_delta: 5.0,
            delta_normalizer: 1.0,
            class_multipliers: None,
            class_map: None,
        }
    }

    fn options(confidence_threshold: f32, letterbox: bool) -> DetectionOptions {
        DetectionOptions {
            confidence_threshold,
            nms_threshold: 0.45,
            letterbox,
        }
    }

    #[test]
    fn logistic_obb_head_decodes_box_front_and_strict_thresholds() {
        let outputs = decode_obb_yolo_head(
            &[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 2.0, -2.0],
            1,
            1,
            &spec(false, 1.2),
            32,
            32,
            32,
            32,
            options(0.1, false),
        )
        .expect("valid OBB head");
        assert_eq!(outputs.len(), 1, "only class 0 exceeds the score threshold");
        let detection = &outputs[0].detection;
        assert_eq!(detection.class_id, 0);
        assert!((detection.objectness - 0.5).abs() < 1e-6);
        assert!((detection.confidence - 0.5 * (1.0 / (1.0 + (-2.0f32).exp()))).abs() < 1e-6);
        assert!((detection.bbox.x - 16.0).abs() < 1e-6);
        assert!((detection.bbox.y - 16.0).abs() < 1e-6);
        assert!((detection.bbox.w - 10.0).abs() < 1e-6);
        assert!((detection.bbox.h - 12.0).abs() < 1e-6);
        let ObbDetectionGeometry::FrontPoint { x, y } = outputs[0].geometry else {
            panic!("legacy front-point output expected");
        };
        assert!((x - 8.0).abs() < 1e-6);
        assert!((y - 16.0).abs() < 1e-6);

        let at_objectness_threshold = decode_obb_yolo_head(
            &[0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.0, 2.0, -2.0],
            1,
            1,
            &spec(false, 1.2),
            32,
            32,
            32,
            32,
            options(0.5, false),
        )
        .expect("valid OBB head");
        assert!(
            at_objectness_threshold.is_empty(),
            "objectness threshold is strict"
        );
    }

    #[test]
    fn new_coords_obb_head_corrects_front_point_for_letterbox() {
        let outputs = decode_obb_yolo_head(
            &[0.5, 0.5, 0.5, 0.5, 0.75, 0.25, 0.9, 0.8, 0.7],
            1,
            1,
            &spec(true, 1.2),
            32,
            32,
            64,
            32,
            options(0.7, true),
        )
        .expect("valid new-coordinates OBB head");
        assert_eq!(
            outputs.len(),
            1,
            "NMS is deferred and only class 0 meets threshold"
        );
        let detection = &outputs[0].detection;
        assert_eq!(detection.class_id, 0);
        assert!((detection.confidence - 0.72).abs() < 1e-6);
        assert!((detection.bbox.x - 32.0).abs() < 1e-6);
        assert!((detection.bbox.y - 16.0).abs() < 1e-6);
        assert!((detection.bbox.w - 20.0).abs() < 1e-6);
        assert!((detection.bbox.h - 24.0).abs() < 1e-6);
        let ObbDetectionGeometry::FrontPoint { x, y } = outputs[0].geometry else {
            panic!("legacy front-point output expected");
        };
        assert!((x - 48.0).abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    #[test]
    fn obb_letterbox_uses_integer_resized_dimensions() {
        let outputs = decode_obb_yolo_head(
            &[0.5, 0.5, 0.5, 0.5, 0.75, 0.25, 0.9, 0.8, 0.7],
            1,
            1,
            &spec(true, 1.0),
            32,
            32,
            5,
            3,
            options(0.7, true),
        )
        .expect("valid OBB head");
        let ObbDetectionGeometry::FrontPoint { x, y } = outputs[0].geometry else {
            panic!("legacy front-point output expected");
        };
        assert!((x - 3.75).abs() < 1e-6);
        // C++ computes new_h=(3*32)/5 using integer division, so 19/32 is
        // used instead of the mathematically exact 19.2/32 scale.
        assert!((y - 0.236_842_1).abs() < 1e-6);
    }

    #[test]
    fn obb_yolo_decoder_rejects_bad_tensor_lengths_and_zero_grids() {
        let spec = spec(false, 1.0);
        let error =
            decode_obb_yolo_head(&[0.0; 8], 1, 1, &spec, 32, 32, 32, 32, options(0.1, false))
                .expect_err("the six-coordinate OBB layout needs nine entries");
        assert!(format!("{error}").contains("length mismatch"));

        let error = decode_obb_yolo_head(&[], 0, 1, &spec, 32, 32, 32, 32, options(0.1, false))
            .expect_err("zero grid dimensions are invalid");
        assert!(format!("{error}").contains("invalid geometry"));
    }
}

#[cfg(test)]
mod obb_postprocess_tests {
    use crate::model::ObbNmsKind;

    use super::{
        Detection, DetectionBox, ObbDetection, ObbDetectionGeometry, apply_obb_nms,
        apply_obb_nms_by_kind, detection_obb_iou,
    };

    fn rect_detection(
        class_id: usize,
        confidence: f32,
        objectness: f32,
        front: [f32; 2],
    ) -> ObbDetection {
        ObbDetection {
            detection: Detection {
                class_id,
                confidence,
                objectness,
                bbox: DetectionBox {
                    x: 0.0,
                    y: 0.0,
                    w: 4.0,
                    h: 2.0,
                },
            },
            geometry: ObbDetectionGeometry::FrontPoint {
                x: front[0],
                y: front[1],
            },
        }
    }

    #[test]
    fn detection_iou_dispatches_carpon_and_legacy_front_points() {
        let horizontal = rect_detection(0, 0.9, 0.9, [0.0, 1.0]);
        let vertical = rect_detection(0, 0.8, 0.8, [1.0, 0.0]);
        assert!((detection_obb_iou(&horizontal, &vertical) - 1.0 / 3.0).abs() < 1e-6);

        let square = ObbDetection {
            detection: horizontal.detection.clone(),
            geometry: ObbDetectionGeometry::CarpnQuad([0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0]),
        };
        let diamond = ObbDetection {
            detection: vertical.detection.clone(),
            geometry: ObbDetectionGeometry::CarpnQuad([1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0]),
        };
        assert!((detection_obb_iou(&square, &diamond) - 0.5).abs() < 1e-6);

        // C++ falls through to its legacy path for mixed OBB representations,
        // using the CARPN mask's first vertex as the front point.
        let mixed = ObbDetection {
            detection: vertical.detection.clone(),
            geometry: ObbDetectionGeometry::CarpnQuad([1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 0.0, 1.0]),
        };
        assert!((detection_obb_iou(&horizontal, &mixed) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn obb_nms_is_classwise_orientation_aware_and_uses_strict_thresholds() {
        let low = rect_detection(0, 0.8, 0.9, [1.0, 0.0]);
        let high = rect_detection(0, 0.9, 0.9, [0.0, 1.0]);
        let other_class = rect_detection(1, 0.7, 0.9, [1.0, 0.0]);
        let overlap = detection_obb_iou(&low, &high);
        assert!((overlap - 1.0 / 3.0).abs() < 1e-6);

        let mut orientation_safe = vec![low.clone(), high.clone()];
        apply_obb_nms(&mut orientation_safe, 0.5);
        assert_eq!(orientation_safe.len(), 2, "OBB IoU is below AABB IoU");

        let mut below_threshold = vec![low.clone(), other_class.clone(), high.clone()];
        apply_obb_nms(&mut below_threshold, overlap - 1e-4);
        assert_eq!(below_threshold.len(), 2);
        assert_eq!(
            below_threshold[0].detection.confidence,
            high.detection.confidence
        );
        assert!(below_threshold.iter().any(|d| d.detection.class_id == 1));

        let mut at_threshold = vec![low, high, other_class];
        apply_obb_nms(&mut at_threshold, overlap);
        assert_eq!(at_threshold.len(), 3, "IoU equal to threshold is retained");
    }

    #[test]
    fn source_default_and_corners_nms_use_axis_aligned_box_iou() {
        let horizontal = rect_detection(0, 0.9, 0.9, [0.0, 1.0]);
        let vertical = rect_detection(0, 0.8, 0.8, [1.0, 0.0]);
        assert!((detection_obb_iou(&horizontal, &vertical) - 1.0 / 3.0).abs() < 1e-6);

        for kind in [ObbNmsKind::Default, ObbNmsKind::Corners] {
            let mut detections = vec![horizontal.clone(), vertical.clone()];
            apply_obb_nms_by_kind(&mut detections, 0.5, kind, 0.6);
            assert_eq!(detections.len(), 1, "{kind:?} uses AABB IoU = 1");
        }

        let mut oriented = vec![horizontal, vertical];
        apply_obb_nms_by_kind(&mut oriented, 0.5, ObbNmsKind::Obb, 0.6);
        assert_eq!(oriented.len(), 2, "OBB NMS uses orientation-aware IoU");
    }

    #[test]
    fn greedy_and_diou_nms_use_source_distance_penalties() {
        let first = rect_detection(0, 0.9, 0.9, [0.0, 1.0]);
        let mut second = rect_detection(0, 0.8, 0.8, [0.0, 1.0]);
        second.detection.bbox.x = 2.5;
        let iou = super::bbox_iou(first.detection.bbox, second.detection.bbox);
        assert!((iou - 3.0 / 13.0).abs() < 1e-6);

        let mut greedy = vec![first.clone(), second.clone()];
        apply_obb_nms_by_kind(&mut greedy, 0.1, ObbNmsKind::Greedy, 0.6);
        assert_eq!(greedy.len(), 2, "fixed-power DIoU score is below threshold");

        let mut diou = vec![first, second];
        apply_obb_nms_by_kind(&mut diou, 0.1, ObbNmsKind::Diou, 2.0);
        assert_eq!(diou.len(), 1, "beta_nms=2 keeps positive penalized overlap");
    }

    #[test]
    fn weighted_merge_two_overlapping_boxes() {
        let keeper = rect_detection(0, 0.8, 0.9, [0.0, 1.0]);
        let mut candidate = rect_detection(0, 0.8, 0.9, [1.0, 1.0]);
        candidate.detection.bbox.x = 1.0;

        // OBB IoU = 0.6; the enclosing AABB has c² = 5² + 2² = 29,
        // so the DIoU-penalized overlap is 0.6 - (1/29) > 0.55.
        // Candidate weight = 0.8 × 0.6 = 0.48, giving x = (0×0.8 + 1×0.48)/1.28.
        let mut detections = vec![keeper, candidate];
        apply_obb_nms_by_kind(&mut detections, 0.55, ObbNmsKind::Weighted, 1.0);

        assert_eq!(detections.len(), 1);
        assert!((detections[0].detection.bbox.x - 0.375).abs() < 1e-6);
        assert!((detections[0].detection.bbox.y).abs() < 1e-6);
        assert!((detections[0].detection.bbox.w - 4.0).abs() < 1e-6);
        assert!((detections[0].detection.bbox.h - 2.0).abs() < 1e-6);
        assert!(matches!(
            detections[0].geometry,
            ObbDetectionGeometry::FrontPoint { x, y }
                if (x - 0.375).abs() < 1e-6 && (y - 1.0).abs() < 1e-6
        ));
    }

    #[test]
    fn weighted_merge_single_box_unchanged() {
        let original = rect_detection(0, 0.8, 0.9, [0.0, 1.0]);
        let mut detections = vec![original.clone()];
        apply_obb_nms_by_kind(&mut detections, 0.5, ObbNmsKind::Weighted, 0.6);
        assert_eq!(detections, vec![original]);
    }

    #[test]
    fn weighted_merge_disjoint_boxes_kept() {
        let first = rect_detection(0, 0.9, 0.9, [0.0, 1.0]);
        let mut second = rect_detection(0, 0.8, 0.9, [10.0, 1.0]);
        second.detection.bbox.x = 10.0;
        let mut detections = vec![first, second];
        apply_obb_nms_by_kind(&mut detections, 0.1, ObbNmsKind::Weighted, 0.6);
        assert_eq!(detections.len(), 2);
    }

    #[test]
    fn weighted_merge_nan_score_does_not_panic() {
        let nan_scored = rect_detection(0, f32::NAN, 0.9, [0.0, 1.0]);
        let valid = rect_detection(0, 0.8, 0.9, [0.0, 1.0]);
        let mut detections = vec![nan_scored.clone(), valid.clone()];
        apply_obb_nms_by_kind(&mut detections, 0.0, ObbNmsKind::Weighted, 0.6);
        assert_eq!(detections.len(), 2);
        assert!(detections.iter().any(|d| d.detection.confidence.is_nan()));
        assert!(detections.iter().any(|d| d == &valid));
    }

    #[test]
    fn zero_objectness_candidates_never_suppress_or_get_suppressed() {
        let inactive = rect_detection(0, 0.99, 0.0, [0.0, 1.0]);
        let active = rect_detection(0, 0.8, 0.9, [0.0, 1.0]);
        let mut detections = vec![active.clone(), inactive.clone()];
        apply_obb_nms(&mut detections, 0.0);
        assert_eq!(detections.len(), 2);
        assert!(detections.iter().any(|d| d == &inactive));
        assert!(detections.iter().any(|d| d == &active));

        let zero_probability = rect_detection(0, 0.0, 0.9, [0.0, 1.0]);
        let active = rect_detection(0, 0.8, 0.9, [0.0, 1.0]);
        let mut zero_score = vec![active.clone(), zero_probability.clone()];
        apply_obb_nms(&mut zero_score, 0.0);
        assert_eq!(zero_score.len(), 2);
        assert!(zero_score.iter().any(|d| d == &zero_probability));

        let duplicate = rect_detection(0, 0.7, 0.9, [0.0, 1.0]);
        let mut positive = vec![active, duplicate];
        apply_obb_nms(&mut positive, 0.0);
        assert_eq!(positive.len(), 1, "positive overlap exceeds zero threshold");
    }
}

#[cfg(test)]
mod obb_network_decode_tests {
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{ExecutableNetwork, Tensor, YoloHeadOutput};

    use super::{
        DetectionOptions, ObbDetectionGeometry, decode_obb_yolo_detections, decode_yolo_detections,
    };

    #[test]
    fn network_heads_merge_before_orientation_aware_nms() {
        let cfg = ConfigFile::parse_str(
            "obb-detect.cfg",
            "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\n",
        )
        .expect("cfg parses");
        let network = Network::from_config(cfg).expect("OBB network builds");
        let executable = ExecutableNetwork::new(network.clone());
        let stride = network.net.width * network.net.height;
        let mut data = vec![0.0; network.net.channels * stride];
        data[4 * stride] = 0.25; // Raw linear front-point x offset.
        data[6 * stride] = 2.0; // Objectness logit.
        data[7 * stride] = 2.0; // Class logit.
        let input = Tensor::from_data(network.net.input_shape(), data).expect("input tensor");
        let forward = executable.forward(&input).expect("forward pass");
        assert_eq!(forward.yolo_heads.len(), 2);

        let detections = decode_obb_yolo_detections(
            &network,
            &forward.yolo_heads,
            4,
            2,
            DetectionOptions::default(),
        )
        .expect("OBB heads decode");
        assert_eq!(detections.len(), 1, "duplicate heads should be suppressed");
        assert!((detections[0].detection.bbox.x - 0.5).abs() < 1e-6);
        assert!((detections[0].detection.bbox.y - 0.5).abs() < 1e-6);
        assert!((detections[0].detection.bbox.w - 4.0).abs() < 1e-6);
        assert!((detections[0].detection.bbox.h - 2.0).abs() < 1e-6);
        assert!(matches!(
            detections[0].geometry,
            ObbDetectionGeometry::FrontPoint { x, y }
                if (x - 0.25).abs() < 1e-6 && y.abs() < 1e-6
        ));
        assert!(
            decode_yolo_detections(
                &network,
                &forward.yolo_heads,
                4,
                2,
                DetectionOptions::default(),
            )
            .is_err(),
            "OBB heads must not enter axis-aligned decode"
        );
    }

    #[test]
    fn network_decode_rejects_mismatched_diou_betas_across_heads() {
        for nms_kind in ["diounms", "obb_w"] {
            let cfg = ConfigFile::parse_str(
                "obb-beta-mismatch.cfg",
                &format!(
                    "[net]\nwidth=4\nheight=2\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\nnms_kind={nms_kind}\nbeta_nms=0.6\n[obb_yolo]\nclasses=1\nnum=1\nanchors=4,2\nnms_kind={nms_kind}\nbeta_nms=1.2\n"
                ),
            )
            .expect("cfg parses");
            let network = Network::from_config(cfg).expect("OBB network builds");
            let heads: Vec<_> = network
                .layers
                .iter()
                .map(|layer| YoloHeadOutput {
                    layer_index: layer.index,
                    output: Tensor::zeros(layer.output_shape),
                })
                .collect();
            let error =
                decode_obb_yolo_detections(&network, &heads, 4, 2, DetectionOptions::default())
                    .expect_err("DIoU and weighted OBB heads need one network-wide beta");
            assert!(error.to_string().contains("different beta_nms values"));
        }
    }
}

#[cfg(test)]
mod modern_decode_tests {
    use super::*;
    use crate::model::{CenterNetSpec, ModernYoloKind, ModernYoloSpec};
    use crate::runtime::{ModernHeadOutput, Tensor};
    use crate::{ConfigFile, Network, TensorShape};

    fn net(text: &str) -> Network {
        Network::from_config(ConfigFile::parse_str("t.cfg", text).expect("cfg parses"))
            .expect("builds")
    }

    fn yolox_cfg() -> String {
        // 4x4 grid, 2 classes -> filters = classes + 5 = 7.
        String::from(
            "[net]\nwidth=16\nheight=16\nchannels=7\n[convolutional]\nfilters=7\nsize=4\nstride=4\nactivation=linear\n\
             [yolox]\nclasses=2\nmax=4\n",
        )
    }

    #[test]
    fn yolox_decodes_a_confident_box() {
        let network = net(&yolox_cfg());
        let head = &network.layers[1];
        let classes = 2usize;
        let spatial = 16usize;
        let entries = classes + 5;
        let mut data = vec![0.0f32; spatial * entries];
        // Cell (row 2, col 1): obj = 1, class 0 = 1 -> a confident detection.
        let loc = 2 * 4 + 1;
        data[0 * spatial + loc] = 1.0f32; // tx: centre at col + 1 = 2 of 4
        data[1 * spatial + loc] = 2.0f32; // ty: row + 2 = 4 of 4
        data[2 * spatial + loc] = 4f32.ln(); // w = 4 / 4 = 1
        data[3 * spatial + loc] = 4f32.ln(); // h = 4 / 4 = 1
        data[4 * spatial + loc] = 10.0; // objectness
        data[5 * spatial + loc] = 10.0; // class 0
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            &[out],
            16,
            16,
            DetectionOptions {
                confidence_threshold: 0.3,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert_eq!(dets[0].class_id, 0);
        assert!(dets[0].confidence > 0.99, "{}", dets[0].confidence);
        // cx = 2/4, cy = 4/4, w = h = 1 in a 16x16 input.
        assert!((dets[0].bbox.x - 8.0).abs() < 1e-3, "x {}", dets[0].bbox.x);
        assert!((dets[0].bbox.y - 16.0).abs() < 1e-3, "y {}", dets[0].bbox.y);
        assert!((dets[0].bbox.w - 16.0).abs() < 1e-3, "w {}", dets[0].bbox.w);
        assert!((dets[0].bbox.h - 16.0).abs() < 1e-3, "h {}", dets[0].bbox.h);
    }

    #[test]
    fn the_confidence_threshold_filters_and_nms_deduplicates() {
        let network = net(&yolox_cfg());
        let head = &network.layers[1];
        let spatial = 16usize;
        let mut data = vec![0.0f32; spatial * 7];
        // Two adjacent cells both fire on class 1: NMS must merge them.  The
        // objectness and class entries are already sigmoid-activated by the
        // head's forward, so the decode reads probabilities, not logits.
        for loc in [0usize, 1] {
            data[0 * spatial + loc] = 0.0;
            data[1 * spatial + loc] = 0.0;
            data[2 * spatial + loc] = 4f32.ln();
            data[3 * spatial + loc] = 4f32.ln();
            data[4 * spatial + loc] = 0.99; // objectness
            data[6 * spatial + loc] = 0.98; // class 1
        }
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            std::slice::from_ref(&out),
            16,
            16,
            DetectionOptions {
                confidence_threshold: 0.9,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "NMS merged the overlapping pair: {dets:?}");
        // A high threshold keeps nothing.
        let none = decode_modern_detections(
            &network,
            &[out.clone()],
            16,
            16,
            DetectionOptions {
                confidence_threshold: 0.995,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .unwrap();
        assert!(none.is_empty(), "{none:?}");
    }

    #[test]
    fn centernet_decodes_only_peaks() {
        // 4x4 grid, 1 class -> filters = 1 + 4 = 5.
        let cfg = String::from(
            "[net]\nwidth=16\nheight=16\nchannels=5\n[convolutional]\nfilters=5\nsize=4\nstride=4\nactivation=linear\n\
             [centernet]\nclasses=1\nmax=4\n",
        );
        let network = net(&cfg);
        let head = &network.layers[1];
        let spatial = 16usize;
        let mut data = vec![0.0f32; spatial * 5];
        // A ridge: cell 5 is the peak, its neighbours are lower.
        for (loc, v) in [(5usize, 10.0f32), (4, 6.0), (6, 6.0), (9, 6.0)] {
            data[loc] = v;
        }
        data[1 * spatial + 5] = 4f32.ln(); // log_w = 1 cell
        data[2 * spatial + 5] = 4f32.ln();
        data[3 * spatial + 5] = 0.0;
        data[4 * spatial + 5] = 0.0;
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            &[out],
            16,
            16,
            DetectionOptions {
                confidence_threshold: 0.5,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "only the strict peak survives: {dets:?}");
        assert_eq!(dets[0].class_id, 0);
        // cx = cy = (col + 0.5) / 4 = 0.375 -> 6 px of 16.
        assert!((dets[0].bbox.x - 6.0).abs() < 1e-3, "x {}", dets[0].bbox.x);
        assert!((dets[0].bbox.y - 6.0).abs() < 1e-3, "y {}", dets[0].bbox.y);
        // exp(ln 4) / 4 cells = the full 16 px width.
        assert!((dets[0].bbox.w - 16.0).abs() < 1e-3, "w {}", dets[0].bbox.w);
    }

    #[test]
    fn the_distribution_heads_decode_through_the_dfl_expectation() {
        // 4x4 grid, 1 class, reg_max = 3 -> coords = 16, filters = 16 + 1 + 1 = 18.
        let cfg = String::from(
            "[net]\nwidth=16\nheight=16\nchannels=18\n[convolutional]\nfilters=18\nsize=4\nstride=4\nactivation=linear\n\
             [ppyoloe]\nclasses=1\nmax=4\nreg_max=3\n",
        );
        let network = net(&cfg);
        let head = &network.layers[1];
        let bins = 4usize;
        let spatial = 16usize;
        let mut data = vec![0.0f32; spatial * 18];
        let loc = 9usize;
        // Make every side collapse onto bin 2 (i.e. a 2-cell distance) and the
        // class score confident.
        for side in 0..4 {
            data[(side * bins + 2) * spatial + loc] = 40.0;
        }
        data[(4 * bins) * spatial + loc] = 40.0; // quality (objectness)
        data[(4 * bins + 1) * spatial + loc] = 40.0; // class 0
        let out = ModernHeadOutput {
            layer_index: head.index,
            output: Tensor::from_data(head.output_shape, data).unwrap(),
        };
        let dets = decode_modern_detections(
            &network,
            &[out],
            16,
            16,
            DetectionOptions {
                confidence_threshold: 0.3,
                nms_threshold: 0.45,
                letterbox: false,
            },
        )
        .unwrap();
        assert_eq!(dets.len(), 1, "{dets:?}");
        // All four distances are 2 cells around cell (row 2, col 1), so the box
        // centre is the cell centre (1.5 / 4 -> 6 px) and both sides span 4
        // cells (16 px) in total.
        assert!((dets[0].bbox.x - 6.0).abs() < 1e-2, "x {}", dets[0].bbox.x);
        // row 2, col 1 -> cy = 2.5 / 4 = 0.625 -> 10 px.
        assert!((dets[0].bbox.y - 10.0).abs() < 1e-2, "y {}", dets[0].bbox.y);
        assert!((dets[0].bbox.w - 16.0).abs() < 1e-2, "w {}", dets[0].bbox.w);
    }

    #[test]
    fn a_non_modern_layer_is_rejected() {
        let network = net(&yolox_cfg());
        let out = ModernHeadOutput {
            layer_index: 0, // the convolution, not a head
            output: Tensor::from_data(
                network.layers[0].output_shape,
                vec![
                    0.0;
                    network.layers[0].output_shape.batch
                        * network.layers[0].output_shape.element_count()
                ],
            )
            .unwrap(),
        };
        let err = decode_modern_detections(&network, &[out], 16, 16, DetectionOptions::default())
            .expect_err("a convolution is not a modern head");
        assert!(
            format!("{err}").contains("not a modern anchor-free head"),
            "{err}"
        );
    }
}

#[cfg(test)]
mod carpn_decode_tests {
    use crate::TensorShape;
    use crate::carpn::ObbRefineProposal;
    use crate::cfg::ConfigFile;
    use crate::model::{LayerKind, Network};
    use crate::runtime::{ExecutableNetwork, ForwardResult, LayerParams, Tensor, YoloHeadOutput};

    use super::{
        DetectionOptions, ObbDetectionGeometry, ObbImageCorrection, decode_carpn_proposal,
        decode_obb_network_detections, decode_obb_yolo_detections,
    };

    fn tiny_carpn_forward(objectness: f32, two_refines: bool) -> (Network, ForwardResult) {
        let refine = "[obb_refine]\nfrom=0\nproposal_from=2\nproposal_topk_train=2\nproposal_topk_infer=2\nproposal_capacity=2\nsample_grid=1\nfront_range=1\n";
        let cfg = format!(
            "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=1\nobb=1\n\
             [convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nactivation=linear\n\
             [convolutional]\nfilters=8\nsize=1\nbatch_normalize=0\nactivation=linear\n\
             [obb_yolo]\nclasses=1\nnum=1\nmask=0\nanchors=1,1\nnew_coords=1\nmax=1\n{refine}{}",
            if two_refines { refine } else { "" },
        );
        let network =
            Network::from_config(ConfigFile::parse_str("carpn-decode.cfg", &cfg).expect("cfg"))
                .expect("network");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Convolutional(feature) = &mut executable.params[0] else {
            panic!("feature convolution");
        };
        feature.weights.copy_from_slice(&[2.0, 3.0]);
        feature.biases.fill(0.0);
        let LayerParams::Convolutional(head) = &mut executable.params[1] else {
            panic!("proposal convolution");
        };
        head.weights.fill(0.0);
        head.biases
            .copy_from_slice(&[0.9, 0.4, 0.5, 0.5, 0.0, 0.0, objectness, 0.8]);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let forward = executable.forward(&input).expect("forward");
        (executable.network, forward)
    }

    fn tiny_carpn_forward_with_hyperbolic_foveation(objectness: f32) -> (Network, ForwardResult) {
        let cfg = "[net]\nwidth=2\nheight=1\nchannels=1\nbatch=1\nobb=1\n\
            [convolutional]\nfilters=2\nsize=1\nbatch_normalize=0\nactivation=linear\n\
            [hyperbolic_foveation]\nparams=fixed\ninit_cx=0.15\ninit_cy=-0.1\ninit_radius=0.8\ninit_strength=0.6\n\
            [convolutional]\nfilters=8\nsize=1\nbatch_normalize=0\nactivation=linear\n\
            [obb_yolo]\nclasses=1\nnum=1\nmask=0\nanchors=1,1\nnew_coords=1\nmax=1\n\
            [obb_refine]\nfrom=0\nproposal_from=3\nproposal_topk_train=1\nproposal_topk_infer=1\nproposal_capacity=1\nsample_grid=1\nfront_range=1\n";
        let network =
            Network::from_config(ConfigFile::parse_str("carpn-hf-decode.cfg", cfg).expect("cfg"))
                .expect("network");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Convolutional(feature) = &mut executable.params[0] else {
            panic!("feature convolution");
        };
        feature.weights.copy_from_slice(&[2.0, 3.0]);
        let LayerParams::Convolutional(head) = &mut executable.params[2] else {
            panic!("proposal convolution");
        };
        head.weights.fill(0.0);
        head.biases
            .copy_from_slice(&[0.9, 0.4, 0.5, 0.5, 0.0, 0.0, objectness, 0.8]);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let forward = executable.forward(&input).expect("forward");
        (executable.network, forward)
    }

    fn options(confidence_threshold: f32, nms_threshold: f32, letterbox: bool) -> DetectionOptions {
        DetectionOptions {
            confidence_threshold,
            nms_threshold,
            letterbox,
        }
    }

    #[test]
    fn carpn_hyperbolic_foveation_reprojects_aabb_and_every_quad_vertex() {
        let (network, forward) = tiny_carpn_forward_with_hyperbolic_foveation(0.9);
        let detections =
            decode_obb_network_detections(&network, &forward, 2, 1, options(0.1, 0.0, false))
                .expect("CARPN foveated detections");
        assert_eq!(detections.len(), 1);
        let LayerKind::HyperbolicFoveation(hf_spec) = &network.layers[1].kind else {
            panic!("hyperbolic-foveation layer");
        };
        let params = crate::hyperbolic_foveation::HfParams {
            center_x: 0.15,
            center_y: -0.1,
            radius_x: 0.8,
            radius_y: 0.8,
            strength: 0.6,
            alpha: hf_spec.alpha,
            p_exp: hf_spec.p_exp,
            radius_mode: hf_spec.radius_mode,
        };
        let LayerKind::ObbRefine(refine_spec) = &network.layers[4].kind else {
            panic!("CARPN refine layer");
        };
        let proposal = &forward.obb_refine_outputs[0].proposals[0];
        let correction = ObbImageCorrection::new(2, 1, 2, 1, false).expect("correction");
        let unwarped = decode_carpn_proposal(
            refine_spec,
            proposal,
            proposal.raw_regression,
            2,
            1,
            correction,
            0.1,
        )
        .expect("base decode")
        .expect("base proposal");
        let expected_box = crate::hyperbolic_foveation::transform_box_with_jacobian(
            [
                unwarped.detection.bbox.x / 2.0,
                unwarped.detection.bbox.y,
                unwarped.detection.bbox.w / 2.0,
                unwarped.detection.bbox.h,
            ],
            params,
            true,
            hf_spec.inverse_method,
            hf_spec.nr_iters,
            hf_spec.nr_eta,
        );
        let actual_box = detections[0].detection.bbox;
        for (actual, expected) in [
            (actual_box.x / 2.0, expected_box[0]),
            (actual_box.y, expected_box[1]),
            (actual_box.w / 2.0, expected_box[2]),
            (actual_box.h, expected_box[3]),
        ] {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        let (
            ObbDetectionGeometry::CarpnQuad(actual_vertices),
            ObbDetectionGeometry::CarpnQuad(unwarped_vertices),
        ) = (&detections[0].geometry, &unwarped.geometry)
        else {
            panic!("CARPN detection must carry a quad");
        };
        for vertex in 0..4 {
            let source = [
                2.0 * (unwarped_vertices[2 * vertex] / 2.0) - 1.0,
                2.0 * unwarped_vertices[2 * vertex + 1] - 1.0,
            ];
            let expected = crate::hyperbolic_foveation::inverse_point(
                source,
                params,
                hf_spec.inverse_method,
                hf_spec.nr_iters,
                hf_spec.nr_eta,
            )
            .point;
            assert!((actual_vertices[2 * vertex] / 2.0 - 0.5 * (expected[0] + 1.0)).abs() < 1e-5);
            assert!((actual_vertices[2 * vertex + 1] - 0.5 * (expected[1] + 1.0)).abs() < 1e-5);
        }
    }

    #[test]
    fn carpn_refine_decode_uses_stage_two_rows_and_static_anchor_geometry() {
        let (network, forward) = tiny_carpn_forward(0.9, false);
        assert_eq!(forward.obb_refine_outputs.len(), 1);
        assert_eq!(forward.obb_refine_outputs[0].proposal_count, 2);
        let detections =
            decode_obb_network_detections(&network, &forward, 2, 1, options(0.1, 1.1, false))
                .expect("CARPN detections");
        assert_eq!(detections.len(), 2, "{detections:?}");
        for (index, detection) in detections.iter().enumerate() {
            assert_eq!(detection.detection.class_id, 0);
            assert!((detection.detection.confidence - 0.72).abs() < 1e-6);
            assert_eq!(detection.detection.objectness, 0.9);
            let center_x = if index == 0 { 0.5 } else { 1.5 };
            assert!((detection.detection.bbox.x - center_x).abs() < 1e-5);
            let radius = 0.5 * 1.25f32.sqrt();
            assert!((detection.detection.bbox.w - 4.0 * radius).abs() < 1e-5);
            assert!((detection.detection.bbox.h - 2.0 * radius).abs() < 1e-5);
            let ObbDetectionGeometry::CarpnQuad(vertices) = detection.geometry else {
                panic!("CARPN graph must not expose Stage-1 front points");
            };
            assert!((vertices[1] - (0.5 - radius)).abs() < 1e-5);
            assert!((vertices[5] - (0.5 + radius)).abs() < 1e-5);
        }
    }

    #[test]
    fn carpn_refine_decode_corrects_every_vertex_for_letterbox_and_scales_the_aabb() {
        let (network, _) = tiny_carpn_forward(0.9, false);
        let LayerKind::ObbRefine(spec) = &network.layers[3].kind else {
            panic!("CARPN layer");
        };
        let proposal = ObbRefineProposal {
            proposal_index: 0,
            source_layer_index: spec.proposal_layer,
            anchor_index: 0,
            cell_x: 0,
            cell_y: 0,
            class_id: 0,
            objectness: 0.8,
            class_probability: 0.9,
            score: 0.72,
            bbox: [0.5, 0.5, 1.0, 1.0],
            anchor_x: 0.5,
            anchor_y: 0.5,
            anchor_radius: 0.5,
            raw_regression: [0.0; 5],
        };
        let correction = ObbImageCorrection::new(4, 2, 2, 2, true).expect("correction");
        let detection = decode_carpn_proposal(spec, &proposal, [0.0; 5], 2, 2, correction, 0.1)
            .expect("decode")
            .expect("detection");
        let ObbDetectionGeometry::CarpnQuad(vertices) = detection.geometry else {
            panic!("CARPN quad");
        };
        let expected = [1.0, 0.0, 3.0, 1.0, 1.0, 2.0, -1.0, 1.0];
        for (actual, expected) in vertices.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(detection.detection.bbox.x, 1.0);
        assert_eq!(detection.detection.bbox.y, 1.0);
        assert_eq!(detection.detection.bbox.w, 4.0);
        assert_eq!(detection.detection.bbox.h, 2.0);
    }

    #[test]
    fn carpn_refine_decode_stays_explicit_when_no_proposals_survive() {
        let (network, forward) = tiny_carpn_forward(0.005, false);
        assert!(!forward.yolo_heads.is_empty());
        let detections =
            decode_obb_network_detections(&network, &forward, 2, 1, options(0.001, 0.0, false))
                .expect("empty CARPN detections");
        assert!(detections.is_empty());

        let mut incomplete = forward.clone();
        incomplete.obb_refine_outputs.clear();
        assert!(
            decode_obb_network_detections(&network, &incomplete, 2, 1, options(0.001, 0.0, false),)
                .is_err()
        );
    }

    #[test]
    fn carpn_refine_decode_merges_multiple_layers_before_one_nms_pass() {
        let (network, forward) = tiny_carpn_forward(0.9, true);
        assert_eq!(forward.obb_refine_outputs.len(), 2);
        let detections =
            decode_obb_network_detections(&network, &forward, 2, 1, options(0.1, 0.99, false))
                .expect("merged CARPN detections");
        assert_eq!(
            detections.len(),
            2,
            "duplicate layers should suppress once: {detections:?}"
        );
    }

    #[test]
    fn carpn_refine_decode_preserves_legacy_front_point_compatibility() {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "legacy-obb.cfg",
                "[net]\nwidth=1\nheight=1\nchannels=8\nobb=1\n[obb_yolo]\nclasses=1\nnum=1\nanchors=1,1\nnew_coords=1\n",
            )
            .expect("cfg"),
        )
        .expect("legacy network");
        let mut data = vec![0.0; 8];
        data[0] = 0.5;
        data[1] = 0.5;
        data[2] = 0.5;
        data[3] = 0.5;
        data[4] = 0.1;
        data[5] = 0.2;
        data[6] = 0.9;
        data[7] = 0.8;
        let head = YoloHeadOutput {
            layer_index: 0,
            output: Tensor::from_data(TensorShape::new(1, 1, 8, 1), data).expect("head"),
        };
        let legacy = decode_obb_yolo_detections(
            &network,
            std::slice::from_ref(&head),
            1,
            1,
            options(0.1, 0.0, false),
        )
        .expect("legacy decoder");
        let forward = ForwardResult::from_heads(vec![head], Vec::new());
        let network_decode =
            decode_obb_network_detections(&network, &forward, 1, 1, options(0.1, 0.0, false))
                .expect("compatibility decoder");
        assert_eq!(network_decode, legacy);
        assert!(matches!(
            network_decode[0].geometry,
            ObbDetectionGeometry::FrontPoint { .. }
        ));
    }
}

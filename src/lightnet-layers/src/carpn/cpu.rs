//! CPU inference primitives for CARPN's OBB-YOLO proposal-refinement head.
//!
//! Proposal decoding, sampling, and affine projection are hand-derived from
//! `raptor:src-lib/obb_stage2_proposal.cpp` and
//! `raptor:src-lib/obb_stage2_refine_layer.cpp`; no compiled-C++ oracle is
//! available.

use crate::detection_geometry::{anchor_pair, decode_obb_bbox, finite_or_zero, head_activation};
use crate::error::{Error, Result};
use crate::model::{LayerKind, Network, ObbRefineSpec, ObbYoloSpec, TensorShape};
use crate::params::ConnectedParams;
use crate::model::Tensor;

/// Stage-1 proposal metadata retained for one valid Stage-2 regression row.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbRefineProposal {
    /// Ordinal after deterministic score ranking and top-K truncation.
    pub proposal_index: usize,
    pub source_layer_index: usize,
    /// Global anchor index from the proposal head's `mask`.
    pub anchor_index: usize,
    pub cell_x: usize,
    pub cell_y: usize,
    pub class_id: usize,
    pub objectness: f32,
    pub class_probability: f32,
    pub score: f32,
    /// Stage-1 decoded `[x, y, width, height]`, normalized to network space.
    pub bbox: [f32; 4],
    /// Static grid-cell center used by the CARPN regression anchor.
    pub anchor_x: f32,
    pub anchor_y: f32,
    /// Prior-anchor half diagonal, normalized by the network dimensions.
    pub anchor_radius: f32,
    /// Five raw affine outputs `[dx, dy, dr, dalpha, dbeta]`.
    pub raw_regression: [f32; 5],
}

/// Per-layer valid proposals from one `[obb_refine]` layer.
///
/// The matching `ForwardResult::layer_outputs[layer_index]` is a fixed-capacity
/// `[proposal_capacity, 1, 5, 1]` tensor. `proposal_count` and `proposals`
/// distinguish real rows from its zero-padded tail.
#[derive(Clone, Debug, PartialEq)]
pub struct ObbRefineOutput {
    pub layer_index: usize,
    pub proposal_count: usize,
    pub proposals: Vec<ObbRefineProposal>,
}

/// Execute one CPU `[obb_refine]` layer using its explicitly routed feature and
/// Stage-1 proposal layers.
#[doc(hidden)]
pub fn forward_refine_layer(
    network: &Network,
    layer_index: usize,
    spec: &ObbRefineSpec,
    output_shape: TensorShape,
    layer_outputs: &[Tensor],
    params: &ConnectedParams,
) -> Result<(Tensor, ObbRefineOutput)> {
    if output_shape.width != spec.proposal_capacity
        || output_shape.height != 1
        || output_shape.channels != 5
        || output_shape.batch != 1
    {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) requires output shape [{}, 1, 5, 1], got {output_shape}",
            spec.proposal_capacity
        )));
    }
    if params.batch_norm.is_some() {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) cannot have batch normalization parameters"
        )));
    }
    let expected_weights = spec
        .inputs
        .checked_mul(5)
        .ok_or_else(|| Error::InvalidConfig("CARPN parameter count overflow".to_owned()))?;
    if params.biases.len() != 5 || params.weights.len() != expected_weights {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) parameter shape mismatch: expected 5 biases and {expected_weights} weights, got {} and {}",
            params.biases.len(),
            params.weights.len()
        )));
    }

    let feature = layer_outputs.get(spec.feature_layer).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) is missing feature layer #{} output",
            spec.feature_layer
        ))
    })?;
    if feature.shape.width == 0 || feature.shape.height == 0 || feature.shape.channels == 0 {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) feature layer #{} has an empty spatial shape",
            spec.feature_layer
        )));
    }
    if feature.shape.batch != 1 {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) currently requires batch size one, got {}",
            feature.shape.batch
        )));
    }
    let expected_inputs = spec
        .sample_grid
        .checked_mul(spec.sample_grid)
        .and_then(|samples| samples.checked_mul(feature.shape.channels))
        .ok_or_else(|| Error::InvalidConfig("CARPN sample size overflow".to_owned()))?;
    if expected_inputs != spec.inputs {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) feature size mismatch: expected {} inputs, configured {}",
            expected_inputs, spec.inputs
        )));
    }

    let proposal_layer = network.layers.get(spec.proposal_layer).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) is missing proposal layer #{}",
            spec.proposal_layer
        ))
    })?;
    let LayerKind::ObbYolo(proposal_spec) = &proposal_layer.kind else {
        return Err(Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) proposal layer #{} is not [obb_yolo]",
            spec.proposal_layer
        )));
    };
    let raw_head = layer_outputs.get(spec.proposal_layer).ok_or_else(|| {
        Error::InvalidConfig(format!(
            "layer #{layer_index} ([obb_refine]) is missing proposal layer #{} output",
            spec.proposal_layer
        ))
    })?;
    let mut proposals =
        collect_proposals(network, spec, proposal_layer.index, proposal_spec, raw_head)?;

    let mut output = Tensor::zeros(output_shape);
    for (proposal_index, proposal) in proposals.iter_mut().enumerate() {
        let sampled = sample_proposal_features(feature, proposal, spec.sample_grid)?;
        let mut raw = [0.0f32; 5];
        for out_channel in 0..5 {
            let mut value = params.biases[out_channel];
            let weight_offset = out_channel * spec.inputs;
            for (input, feature_value) in sampled.iter().copied().enumerate() {
                value += params.weights[weight_offset + input] * feature_value;
            }
            raw[out_channel] = finite_or_zero(value);
            // Tensor storage is NCHW: one channel plane contains all proposal rows.
            output.data[out_channel * spec.proposal_capacity + proposal_index] = raw[out_channel];
        }
        proposal.proposal_index = proposal_index;
        proposal.raw_regression = raw;
    }

    let proposal_count = proposals.len();
    Ok((
        output,
        ObbRefineOutput {
            layer_index,
            proposal_count,
            proposals,
        },
    ))
}

#[doc(hidden)]
pub fn collect_proposals(
    network: &Network,
    refine: &ObbRefineSpec,
    source_layer_index: usize,
    source: &ObbYoloSpec,
    head: &Tensor,
) -> Result<Vec<ObbRefineProposal>> {
    if head.shape.batch != 1 {
        return Err(Error::InvalidConfig(format!(
            "CARPN proposal head #{source_layer_index} requires batch size one, got {}",
            head.shape.batch
        )));
    }
    if head.shape.width == 0 || head.shape.height == 0 || source.classes != refine.classes {
        return Err(Error::InvalidConfig(format!(
            "CARPN proposal head #{source_layer_index} has invalid grid geometry or class count"
        )));
    }
    let stride = head
        .shape
        .width
        .checked_mul(head.shape.height)
        .ok_or_else(|| Error::InvalidConfig("CARPN proposal grid size overflow".to_owned()))?;
    let entries_per_anchor = source
        .classes
        .checked_add(7)
        .ok_or_else(|| Error::InvalidConfig("CARPN proposal entry count overflow".to_owned()))?;
    let expected = stride
        .checked_mul(source.mask.len())
        .and_then(|size| size.checked_mul(entries_per_anchor))
        .ok_or_else(|| Error::InvalidConfig("CARPN proposal head size overflow".to_owned()))?;
    if head.data.len() != expected {
        return Err(Error::InvalidConfig(format!(
            "CARPN proposal head #{source_layer_index} tensor length mismatch: expected {expected}, got {}",
            head.data.len()
        )));
    }
    let topk = refine.proposal_topk_infer.min(refine.proposal_capacity);
    if topk == 0 {
        return Ok(Vec::new());
    }
    let threshold = refine.proposal_thresh.max(0.001);
    let mut candidates = Vec::<(usize, ObbRefineProposal)>::new();
    let mut source_order = 0usize;

    // Match source enumeration: anchor, cell, then class. Equal scores use this
    // explicit source-order tie-break because C++ std::sort leaves ties undefined.
    for (anchor_slot, &anchor_index) in source.mask.iter().enumerate() {
        let anchor = anchor_pair(&source.yolo, anchor_index)?;
        if !anchor.0.is_finite() || !anchor.1.is_finite() || anchor.0 <= 0.0 || anchor.1 <= 0.0 {
            return Err(Error::InvalidConfig(format!(
                "CARPN proposal head #{source_layer_index} has invalid anchor {anchor_index}"
            )));
        }
        for cell in 0..stride {
            let col = cell % head.shape.width;
            let row = cell / head.shape.width;
            let entry_offset = anchor_slot * entries_per_anchor * stride + cell;
            let objectness = finite_or_zero(head_activation(
                head.data[entry_offset + 6 * stride],
                source.new_coords,
            ));
            if objectness < threshold {
                source_order += source.classes;
                continue;
            }
            let bbox = decode_obb_bbox(
                head.data[entry_offset],
                head.data[entry_offset + stride],
                head.data[entry_offset + 2 * stride],
                head.data[entry_offset + 3 * stride],
                col,
                row,
                head.shape.width,
                head.shape.height,
                network.net.width,
                network.net.height,
                anchor,
                source.scale_x_y,
                source.new_coords,
            );
            let prior_width = anchor.0 / network.net.width as f32;
            let prior_height = anchor.1 / network.net.height as f32;
            let anchor_radius =
                0.5 * (prior_width * prior_width + prior_height * prior_height).sqrt();
            if !anchor_radius.is_finite() || anchor_radius <= 0.0 {
                return Err(Error::InvalidConfig(format!(
                    "CARPN proposal head #{source_layer_index} has invalid prior radius for anchor {anchor_index}"
                )));
            }
            let anchor_x = (col as f32 + 0.5) / head.shape.width as f32;
            let anchor_y = (row as f32 + 0.5) / head.shape.height as f32;
            for class_id in 0..source.classes {
                let order = source_order;
                source_order += 1;
                let class_probability = finite_or_zero(head_activation(
                    head.data[entry_offset + (7 + class_id) * stride],
                    source.new_coords,
                ));
                let score = finite_or_zero(objectness * class_probability);
                if score < threshold {
                    continue;
                }
                candidates.push((
                    order,
                    ObbRefineProposal {
                        proposal_index: 0,
                        source_layer_index,
                        anchor_index,
                        cell_x: col,
                        cell_y: row,
                        class_id,
                        objectness,
                        class_probability,
                        score,
                        bbox: [bbox.x, bbox.y, bbox.w, bbox.h],
                        anchor_x,
                        anchor_y,
                        anchor_radius,
                        raw_regression: [0.0; 5],
                    },
                ));
            }
        }
    }

    candidates.sort_by(|(order_a, a), (order_b, b)| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| order_a.cmp(order_b))
    });
    candidates.truncate(topk);
    Ok(candidates
        .into_iter()
        .map(|(_, proposal)| proposal)
        .collect())
}

#[doc(hidden)]
pub fn sample_proposal_features(
    feature: &Tensor,
    proposal: &ObbRefineProposal,
    sample_grid: usize,
) -> Result<Vec<f32>> {
    let count = sample_grid
        .checked_mul(sample_grid)
        .and_then(|samples| samples.checked_mul(feature.shape.channels))
        .ok_or_else(|| Error::InvalidConfig("CARPN sample feature size overflow".to_owned()))?;
    if sample_grid == 0
        || feature.shape.width == 0
        || feature.shape.height == 0
        || feature.shape.channels == 0
        || feature.data.len() != feature.shape.element_count()
    {
        return Err(Error::InvalidConfig(
            "invalid feature tensor for CARPN sampling".to_owned(),
        ));
    }
    let center_x = proposal.bbox[0] * feature.shape.width.saturating_sub(1) as f32;
    let center_y = proposal.bbox[1] * feature.shape.height.saturating_sub(1) as f32;
    let radius_x = (proposal.bbox[2] * feature.shape.width as f32 * 0.5).max(0.5);
    let radius_y = (proposal.bbox[3] * feature.shape.height as f32 * 0.5).max(0.5);
    let half = (sample_grid - 1) as f32 * 0.5;
    let mut samples = Vec::with_capacity(count);
    for gy in 0..sample_grid {
        let oy = if sample_grid == 1 {
            0.0
        } else {
            (gy as f32 - half) / half
        };
        for gx in 0..sample_grid {
            let ox = if sample_grid == 1 {
                0.0
            } else {
                (gx as f32 - half) / half
            };
            let x = center_x + ox * radius_x;
            let y = center_y + oy * radius_y;
            for channel in 0..feature.shape.channels {
                samples.push(sample_feature_bilinear(feature, 0, channel, x, y)?);
            }
        }
    }
    Ok(samples)
}

#[doc(hidden)]
pub fn sample_feature_bilinear(
    feature: &Tensor,
    batch: usize,
    channel: usize,
    x: f32,
    y: f32,
) -> Result<f32> {
    let shape = feature.shape;
    if shape.width == 0
        || shape.height == 0
        || channel >= shape.channels
        || batch >= shape.batch
        || feature.data.len() != shape.batch * shape.element_count()
    {
        return Err(Error::InvalidConfig(
            "invalid feature tensor for CARPN bilinear sampling".to_owned(),
        ));
    }
    if !x.is_finite() || !y.is_finite() {
        return Ok(0.0);
    }
    let x = x.clamp(0.0, (shape.width - 1) as f32);
    let y = y.clamp(0.0, (shape.height - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(shape.width - 1);
    let y1 = (y0 + 1).min(shape.height - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;
    let plane = shape.width * shape.height;
    let base = batch * shape.element_count() + channel * plane;
    let v00 = feature.data[base + y0 * shape.width + x0];
    let v01 = feature.data[base + y0 * shape.width + x1];
    let v10 = feature.data[base + y1 * shape.width + x0];
    let v11 = feature.data[base + y1 * shape.width + x1];
    let top = v00 * (1.0 - wx) + v01 * wx;
    let bottom = v10 * (1.0 - wx) + v11 * wx;
    Ok(finite_or_zero(top * (1.0 - wy) + bottom * wy))
}


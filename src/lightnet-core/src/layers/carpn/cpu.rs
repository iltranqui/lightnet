//! CPU inference primitives for CARPN's OBB-YOLO proposal-refinement head.
//!
//! Proposal decoding, sampling, and affine projection are hand-derived from
//! `raptor:src-lib/obb_stage2_proposal.cpp` and
//! `raptor:src-lib/obb_stage2_refine_layer.cpp`; no compiled-C++ oracle is
//! available.

use crate::detect::{anchor_pair, decode_obb_bbox, finite_or_zero, head_activation};
use crate::error::{Error, Result};
use crate::model::{LayerKind, Network, ObbRefineSpec, ObbYoloSpec, TensorShape};
use crate::runtime::{ConnectedParams, Tensor};

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
pub(crate) fn forward_refine_layer(
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

fn collect_proposals(
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

fn sample_proposal_features(
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

fn sample_feature_bilinear(
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

#[cfg(test)]
mod tests {
    use crate::TensorShape;
    use crate::cfg::ConfigFile;
    use crate::model::Network;
    use crate::runtime::{ExecutableNetwork, LayerParams, Tensor};

    use super::{
        ObbRefineProposal, collect_proposals, sample_feature_bilinear, sample_proposal_features,
    };

    fn tiny_exec(objectness: f32) -> ExecutableNetwork {
        let network = Network::from_config(
            ConfigFile::parse_str(
                "carpn-forward.cfg",
                r#"
                    [net]
                    width=2
                    height=1
                    channels=1
                    batch=1
                    obb=1

                    [convolutional]
                    filters=2
                    size=1
                    batch_normalize=0
                    activation=linear

                    [convolutional]
                    filters=8
                    size=1
                    batch_normalize=0
                    activation=linear

                    [obb_yolo]
                    classes=1
                    num=1
                    mask=0
                    anchors=1,1
                    new_coords=1
                    max=1

                    [obb_refine]
                    from=0
                    proposal_from=2
                    proposal_topk_train=2
                    proposal_topk_infer=2
                    proposal_capacity=2
                    sample_grid=1
                    front_range=1
                "#,
            )
            .expect("CARPN cfg"),
        )
        .expect("CARPN network");
        let mut executable = ExecutableNetwork::new(network);
        let LayerParams::Convolutional(feature) = &mut executable.params[0] else {
            panic!("expected feature convolution parameters");
        };
        feature.weights.copy_from_slice(&[2.0, 3.0]);
        feature.biases.fill(0.0);
        let LayerParams::Convolutional(head) = &mut executable.params[1] else {
            panic!("expected proposal convolution parameters");
        };
        head.weights.fill(0.0);
        head.biases
            .copy_from_slice(&[0.9, 0.4, 0.5, 0.5, 0.0, 0.0, objectness, 0.8]);
        let LayerParams::Connected(refine) = &mut executable.params[3] else {
            panic!("expected CARPN parameters");
        };
        refine.biases.fill(0.0);
        refine.weights.fill(0.0);
        refine.weights[0] = 0.1;
        refine.weights[3] = 0.1;
        refine.weights[4] = 0.05;
        refine.weights[7] = 0.2;
        refine.weights[8] = -0.1;
        executable
    }

    #[test]
    fn carpn_refine_forward_executes_routed_feature_and_proposal_sources() {
        let executable = tiny_exec(0.9);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let forward = executable.forward(&input).expect("CARPN forward");
        assert_eq!(forward.yolo_heads.len(), 1);
        assert_eq!(forward.yolo_heads[0].layer_index, 2);
        assert_eq!(forward.obb_refine_outputs.len(), 1);

        let refine = &forward.obb_refine_outputs[0];
        assert_eq!(refine.layer_index, 3);
        assert_eq!(refine.proposal_count, 2);
        assert_eq!(refine.proposals.len(), 2);
        let proposal = &refine.proposals[0];
        assert_eq!(proposal.source_layer_index, 2);
        assert_eq!(proposal.anchor_index, 0);
        assert_eq!(
            (proposal.cell_x, proposal.cell_y, proposal.class_id),
            (0, 0, 0)
        );
        assert_eq!(proposal.bbox, [0.45, 0.4, 0.5, 1.0]);
        assert_eq!((proposal.anchor_x, proposal.anchor_y), (0.25, 0.5));
        assert!((proposal.anchor_radius - 0.5 * 1.25f32.sqrt()).abs() < 1e-6);
        assert_eq!(proposal.objectness, 0.9);
        assert_eq!(proposal.class_probability, 0.8);
        assert!((proposal.score - 0.72).abs() < 1e-6);
        let expected_first = [0.29, 0.435, 0.145, 0.87, -0.29];
        for (actual, expected) in proposal.raw_regression.iter().zip(expected_first) {
            assert!((actual - expected).abs() < 1e-5);
        }
        let second = &refine.proposals[1];
        assert_eq!((second.cell_x, second.cell_y), (1, 0));
        assert_eq!(second.bbox, [0.95, 0.4, 0.5, 1.0]);
        assert_eq!((second.anchor_x, second.anchor_y), (0.75, 0.5));
        let expected_second = [0.39, 0.585, 0.195, 1.17, -0.39];
        for (actual, expected) in second.raw_regression.iter().zip(expected_second) {
            assert!((actual - expected).abs() < 1e-5);
        }

        let raw = &forward.layer_outputs[3];
        assert_eq!(raw.shape, TensorShape::new(2, 1, 5, 1));
        let expected_channel_major = [
            0.29, 0.39, 0.435, 0.585, 0.145, 0.195, 0.87, 1.17, -0.29, -0.39,
        ];
        for (actual, expected) in raw.data.iter().zip(expected_channel_major) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn carpn_refine_forward_keeps_zero_proposals_as_padded_output() {
        let executable = tiny_exec(0.005);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let forward = executable.forward(&input).expect("CARPN forward");
        assert_eq!(forward.obb_refine_outputs[0].proposal_count, 0);
        assert!(forward.obb_refine_outputs[0].proposals.is_empty());
        assert_eq!(forward.layer_outputs[3].shape, TensorShape::new(2, 1, 5, 1));
        assert_eq!(forward.layer_outputs[3].data, [0.0; 10]);
    }

    #[test]
    fn carpn_proposals_apply_topk_source_order_ties_and_nonfinite_filtering() {
        use crate::model::LayerKind;

        let cfg = r#"
            [net]
            width=2
            height=1
            channels=1
            batch=1
            obb=1
            [convolutional]
            filters=2
            size=1
            batch_normalize=0
            activation=linear
            [convolutional]
            filters=9
            size=1
            batch_normalize=0
            activation=linear
            [obb_yolo]
            classes=2
            num=1
            mask=0
            anchors=1,1
            new_coords=1
            [obb_refine]
            from=0
            proposal_from=2
            proposal_topk_train=1
            proposal_topk_infer=1
            proposal_capacity=1
            sample_grid=1
        "#;
        let network =
            Network::from_config(ConfigFile::parse_str("carpn-topk.cfg", cfg).expect("cfg"))
                .expect("network");
        let LayerKind::ObbRefine(refine) = &network.layers[3].kind else {
            panic!("expected CARPN refine layer");
        };
        let LayerKind::ObbYolo(source) = &network.layers[2].kind else {
            panic!("expected OBB proposal head");
        };

        let mut data = vec![0.0; 9 * 2];
        for cell in 0..2 {
            data[6 * 2 + cell] = 0.9;
            data[7 * 2 + cell] = 0.8;
            data[8 * 2 + cell] = 0.8;
        }
        data[6 * 2 + 1] = 0.0;
        let head =
            Tensor::from_data(TensorShape::new(2, 1, 9, 1), data.clone()).expect("head tensor");
        let tied = collect_proposals(&network, refine, 2, source, &head).expect("tied proposals");
        assert_eq!(tied.len(), 1);
        assert_eq!((tied[0].cell_x, tied[0].class_id), (0, 0));
        assert_eq!(tied[0].source_layer_index, 2);

        data[7 * 2] = 0.7;
        data[8 * 2] = 0.6;
        data[6 * 2 + 1] = 0.9;
        data[7 * 2 + 1] = 0.5;
        data[8 * 2 + 1] = 0.9;
        let ranked = Tensor::from_data(TensorShape::new(2, 1, 9, 1), data.clone())
            .expect("ranked head tensor");
        let top =
            collect_proposals(&network, refine, 2, source, &ranked).expect("ranked proposals");
        assert_eq!(top.len(), 1);
        assert_eq!((top[0].cell_x, top[0].class_id), (1, 1));

        data[6 * 2] = f32::NAN;
        data[6 * 2 + 1] = 0.0;
        let non_finite =
            Tensor::from_data(TensorShape::new(2, 1, 9, 1), data).expect("non-finite head tensor");
        assert!(
            collect_proposals(&network, refine, 2, source, &non_finite)
                .expect("non-finite proposals")
                .is_empty()
        );
    }

    #[test]
    fn carpn_feature_sampling_uses_grid_order_bilinear_values_and_clamping() {
        let feature = Tensor::from_data(TensorShape::new(2, 2, 1, 1), vec![1.0, 3.0, 5.0, 7.0])
            .expect("feature");
        assert_eq!(
            sample_feature_bilinear(&feature, 0, 0, 0.5, 0.5).unwrap(),
            4.0
        );
        assert_eq!(
            sample_feature_bilinear(&feature, 0, 0, -1.0, 2.0).unwrap(),
            5.0
        );
        let proposal = ObbRefineProposal {
            proposal_index: 0,
            source_layer_index: 0,
            anchor_index: 0,
            cell_x: 0,
            cell_y: 0,
            class_id: 0,
            objectness: 1.0,
            class_probability: 1.0,
            score: 1.0,
            bbox: [0.5, 0.5, 1.0, 1.0],
            anchor_x: 0.5,
            anchor_y: 0.5,
            anchor_radius: 1.0,
            raw_regression: [0.0; 5],
        };
        assert_eq!(
            sample_proposal_features(&feature, &proposal, 2).unwrap(),
            [1.0, 3.0, 5.0, 7.0]
        );
        let non_finite = Tensor::from_data(TensorShape::new(1, 1, 1, 1), vec![f32::NAN])
            .expect("non-finite feature");
        assert_eq!(
            sample_feature_bilinear(&non_finite, 0, 0, 0.0, 0.0).unwrap(),
            0.0
        );
    }

    #[test]
    fn carpn_refine_forward_rejects_training_until_stage_two_updates_exist() {
        let executable = tiny_exec(0.9);
        let input =
            Tensor::from_data(executable.network.net.input_shape(), vec![1.0, 2.0]).expect("input");
        let error = executable
            .forward_training(&input)
            .expect_err("CARPN training is unsupported");
        assert!(error.to_string().contains("[obb_refine]"));
    }
}

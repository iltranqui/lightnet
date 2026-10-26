//! Native delta-rule scan, deterministic routed/shared experts and exact adjoint.
//! Serialized weights and update directions keep the CPU source order.
use std::ops::Range;
use super::cpu::{KimiKdaGeometry, KimiOffsets};
use crate::error::{Error, Result};
use crate::layers::vit::gpu::device_activation;
use crate::model::{KimiKdaSpec, Layer};
use crate::runtime::{GpuExecutableNetwork, LayerTrainBuffers, ResidentMomentSlot,
    ResidentParameterRange, backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda::kimi_kda::{KdaForward, KdaGeometry};

pub(crate) fn supported(layer: &Layer, spec: &KimiKdaSpec) -> bool {
    KimiKdaGeometry::from_layer(layer.input_shape, spec).is_ok()
        && device_activation(&spec.activation).is_some()
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    geometry: KimiKdaGeometry,
    weights: Vec<T>,
    resident_generation: Option<u64>,
    gamma1: T,
    gamma2: T,
    tokens: T,
    ln1: T,
    projections: Vec<T>,
    scan: KdaForward<T>,
    pre_expert: T,
    ln2: T,
    gates: T,
    expert_pre: Vec<T>,
    expert_act: Vec<T>,
    expert_out: Vec<T>,
}

// Weight block index order: Q, K, V, beta, decay, O, router, then
// alternating W1/W2 for the shared expert and each routed expert.
fn blocks(g: &KimiKdaGeometry, o: &KimiOffsets) -> Vec<(Range<usize>, Range<usize>, usize, usize)> {
    let (c, r, e, h) = (g.channels, g.rank, g.experts, g.hidden);
    let mut out = vec![
        (o.wq..o.wq + r*c, o.bq..o.bq + r, c, r),
        (o.wk..o.wk + r*c, o.bk..o.bk + r, c, r),
        (o.wv..o.wv + c*c, o.bv..o.bv + c, c, c),
        (o.wbeta..o.wbeta + r*c, o.bbeta..o.bbeta + r, c, r),
        (o.wdecay..o.wdecay + r*c, o.bdecay..o.bdecay + r, c, r),
        (o.wo..o.wo + c*c, o.bo..o.bo + c, c, c),
        (o.wrouter..o.wrouter + e*c, o.brouter..o.brouter + e, c, e),
    ];
    for expert in 0..=e {
        out.push((o.w1 + expert*h*c..o.w1 + (expert+1)*h*c,
            o.b1 + expert*h..o.b1 + (expert+1)*h, c, h));
        out.push((o.w2 + expert*c*h..o.w2 + (expert+1)*c*h,
            o.b2 + expert*c..o.b2 + (expert+1)*c, h, c));
    }
    out
}

fn scan_geometry(g: &KimiKdaGeometry) -> KdaGeometry {
    KdaGeometry { batch: g.batch, tokens: g.tokens(), channels: g.channels, rank: g.rank }
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_kimi_kda_gpu(
        &self, layer: &Layer, spec: &KimiKdaSpec, params: &super::KimiKdaParams,
        input: &B::Tensor, config: ElementwiseLaunchConfig, training: bool,
    ) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)> {
        let g = KimiKdaGeometry::from_layer(layer.input_shape, spec).map_err(Error::InvalidConfig)?;
        params.validate().map_err(Error::InvalidConfig)?;
        if !params.matches(&g) || layer.input_shape != layer.output_shape {
            return Err(Error::InvalidConfig("Kimi KDA GPU parameter/output geometry mismatch".into()));
        }
        let activation = device_activation(&g.activation).ok_or_else(|| Error::UnsupportedLayer {
            layer_index: layer.index, name: "Kimi KDA GPU activation".into(),
        })?;
        let be = &self.backend;
        let err = |e| backend_error("GPU Kimi KDA forward", e);
        let (rows, c) = (g.rows(), g.channels);
        let o = KimiOffsets::new(c, g.rank, g.experts, g.hidden).map_err(Error::InvalidConfig)?;
        let descriptions = blocks(&g, &o);
        let mut weights = Vec::with_capacity(descriptions.len());
        let mut biases = Vec::with_capacity(descriptions.len());
        let (gamma1, beta1, gamma2, beta2, resident_generation) = if self.resident_parameter_path_enabled() {
            let total = params.parameter_count();
            let shift = o.total_biases;
            let layout = vec![
                ResidentParameterRange { range:0..shift, decay:false, coupled_decay:true,
                    moment_slot:Some(ResidentMomentSlot::Biases), active:true },
                ResidentParameterRange { range:shift..total, decay:true, coupled_decay:true,
                    moment_slot:Some(ResidentMomentSlot::Weights), active:true },
            ];
            let mut slices = Vec::with_capacity(2 * descriptions.len() + 4);
            for (w, b, _, _) in &descriptions {
                slices.push(w.start + shift..w.end + shift);
                slices.push(b.clone());
            }
            for start in [o.ln1_gamma, o.ln1_beta, o.ln2_gamma, o.ln2_beta] {
                slices.push(start..start + c);
            }
            let host = params.to_blob().map_err(Error::InvalidConfig)?;
            let (tensors, generation) = self.resident_parameter_slices(layer.index, &host, layout, &slices, config)?;
            let mut tensors = tensors.into_iter();
            for _ in &descriptions {
                weights.push(tensors.next().expect("Kimi weight slice"));
                biases.push(tensors.next().expect("Kimi bias slice"));
            }
            (tensors.next().expect("Kimi LN1 gamma"), tensors.next().expect("Kimi LN1 beta"),
                tensors.next().expect("Kimi LN2 gamma"), tensors.next().expect("Kimi LN2 beta"), Some(generation))
        } else {
            for (w, b, ins, outs) in &descriptions {
                weights.push(be.upload_f32(&[*outs, *ins], &params.weights[w.clone()]).map_err(err)?);
                biases.push(be.upload_f32(&[*outs], &params.biases[b.clone()]).map_err(err)?);
            }
            (be.upload_f32(&[c], &params.biases[o.ln1_gamma..o.ln1_gamma+c]).map_err(err)?,
                be.upload_f32(&[c], &params.biases[o.ln1_beta..o.ln1_beta+c]).map_err(err)?,
                be.upload_f32(&[c], &params.biases[o.ln2_gamma..o.ln2_gamma+c]).map_err(err)?,
                be.upload_f32(&[c], &params.biases[o.ln2_beta..o.ln2_beta+c]).map_err(err)?, None)
        };
        let lin = |x: &B::Tensor, i: usize| {
            let (_, _, ins, outs) = &descriptions[i];
            be.connected_f32(x, &weights[i], &biases[i], rows, *ins, *outs, config).map_err(err)
        };
        let tokens = be.vit_nchw_to_tokens_f32(input, g.batch, g.height, g.width, c, config).map_err(err)?;
        let ln1 = be.layernorm_forward_f32(&tokens, Some(&gamma1), Some(&beta1),
            &[rows, c], rows, c, 1e-5, config).map_err(err)?;
        let projections = (0..5).map(|i| lin(&ln1, i)).collect::<Result<Vec<_>>>()?;
        let scan = be.kimi_scan_f32(&projections[0], &projections[1], &projections[2],
            &projections[3], &projections[4], scan_geometry(&g), config).map_err(err)?;
        let attention = lin(&scan.output, 5)?;
        let pre_expert = be.residual_bias_f32(&tokens, &attention, None,
            &[rows, c], rows, c, config).map_err(err)?;
        let ln2 = be.layernorm_forward_f32(&pre_expert, Some(&gamma2), Some(&beta2),
            &[rows, c], rows, c, 1e-5, config).map_err(err)?;
        let logits = lin(&ln2, 6)?;
        let gates = be.kimi_router_gates_f32(&logits, rows, g.experts, g.top_k, g.temperature, config).map_err(err)?;
        let mut expert_pre = Vec::with_capacity(g.experts + 1);
        let mut expert_act = Vec::with_capacity(g.experts + 1);
        let mut expert_out = Vec::with_capacity(g.experts + 1);
        let mut output = pre_expert.clone();
        for expert in 0..=g.experts {
            let pre = lin(&ln2, 7 + 2 * expert)?;
            let act = be.activation_f32(&pre, activation, config).map_err(err)?;
            let out = lin(&act, 8 + 2 * expert)?;
            let weighted = if expert == 0 { out.clone() } else {
                let gate = be.moe_gate_column_f32(&gates, rows, g.experts, expert - 1, config).map_err(err)?;
                be.scale_channels_f32(&gate, &out, &[rows, 1, 1, c], false, config).map_err(err)?
            };
            output = be.residual_bias_f32(&output, &weighted, None, &[rows, c], rows, c, config).map_err(err)?;
            expert_pre.push(pre); expert_act.push(act); expert_out.push(out);
        }
        let output = be.vit_tokens_to_nchw_f32(&output, g.batch, g.height, g.width, c, config).map_err(err)?;
        let cache = training.then(|| DeviceCache { geometry: g, weights, resident_generation, gamma1, gamma2,
            tokens, ln1, projections, scan, pre_expert, ln2, gates, expert_pre, expert_act, expert_out });
        Ok((output, cache))
    }

    pub(crate) fn backward_kimi_kda_gpu(
        &self, layer: &Layer, spec: &KimiKdaSpec,
        forward: &crate::runtime::GpuForwardState<B::Tensor>, grad: B::Tensor,
        config: ElementwiseLaunchConfig, layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let g = KimiKdaGeometry::from_layer(layer.input_shape, spec).map_err(Error::InvalidConfig)?;
        let cache = forward.kimi_kda_device_caches.get(layer.index).and_then(Option::as_ref).ok_or_else(||
            Error::InvalidConfig(format!("missing Kimi KDA device cache for layer #{}", layer.index)))?;
        if cache.geometry != g { return Err(Error::InvalidConfig("Kimi KDA cache geometry mismatch".into())); }
        let activation = device_activation(&g.activation).ok_or_else(|| Error::UnsupportedLayer {
            layer_index: layer.index, name: "Kimi KDA GPU backward activation".into(),
        })?;
        let be = &self.backend;
        let err = |e| backend_error("GPU Kimi KDA backward", e);
        let (rows, c) = (g.rows(), g.channels);
        let o = KimiOffsets::new(c, g.rank, g.experts, g.hidden).map_err(Error::InvalidConfig)?;
        let descriptions = blocks(&g, &o);
        let mut gradients = Vec::with_capacity(2 * descriptions.len() + 4);
        let add = |x: &B::Tensor, y: &B::Tensor| {
            be.residual_bias_f32(x, y, None, &[rows, c], rows, c, config).map_err(err)
        };
        let mut lin = |x: &B::Tensor, dy: &B::Tensor, i: usize| -> Result<B::Tensor> {
            let (wr, br, ins, outs) = &descriptions[i];
            let (dx, dw, db) = be.connected_backward_f32(x, &cache.weights[i], dy,
                rows, *ins, *outs, config).map_err(err)?;
            gradients.push((wr.start + o.total_biases..wr.end + o.total_biases, dw));
            gradients.push((br.clone(), db));
            Ok(dx)
        };
        let norm = |x: &B::Tensor, dy: &B::Tensor, gamma: &B::Tensor| -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
            let dx = be.layernorm_backward_f32(dy, x, Some(gamma), &[rows, c], rows, c, 1e-5, config).map_err(err)?;
            let xhat = be.layernorm_forward_f32(x, None, None, &[rows, c], rows, c, 1e-5, config).map_err(err)?;
            let product = be.vector_mul_f32(dy, &xhat, &[rows, c], config).map_err(err)?;
            let dg = be.bias_backward_f32(&product, c, 1, config).map_err(err)?;
            let db = be.bias_backward_f32(dy, c, 1, config).map_err(err)?;
            Ok((dx, dg, db))
        };
        let dy = be.vit_nchw_to_tokens_f32(&grad, g.batch, g.height, g.width, c, config).map_err(err)?;
        let mut dln2 = be.zeros_f32(&[rows, c]).map_err(err)?;
        for expert in 0..=g.experts {
            let expert_dy = if expert == 0 { dy.clone() } else {
                let gate = be.moe_gate_column_f32(&cache.gates, rows, g.experts, expert - 1, config).map_err(err)?;
                be.scale_channels_f32(&gate, &dy, &[rows, 1, 1, c], false, config).map_err(err)?
            };
            let dact = lin(&cache.expert_act[expert], &expert_dy, 8 + 2 * expert)?;
            let dpre = be.activation_backward_f32(&cache.expert_pre[expert], &dact, activation, config).map_err(err)?;
            dln2 = add(&dln2, &lin(&cache.ln2, &dpre, 7 + 2 * expert)?)?;
        }
        // Use temperature=1 in the shared softmax adjoint and scale afterwards:
        // Kimi accepts all positive temperatures; MoE clamps tiny temperatures.
        let dlogits = be.moe_router_backward_f32(&cache.gates, &dy, &cache.expert_out[1..],
            rows, g.experts, c, g.load_balance, 1.0, config).map_err(err)?;
        let inverse_temperature = be.upload_f32(&[1], &[1.0 / g.temperature]).map_err(err)?;
        let dlogits = be.channel_scale_f32(&dlogits, &inverse_temperature, 1, 1, config).map_err(err)?;
        dln2 = add(&dln2, &lin(&cache.ln2, &dlogits, 6)?)?;
        let (dpre, dg2, db2) = norm(&cache.pre_expert, &dln2, &cache.gamma2)?;
        let dpre = add(&dy, &dpre)?;
        let dattn = lin(&cache.scan.output, &dpre, 5)?;
        let scan_grad = be.kimi_scan_backward_f32(&cache.projections[0], &cache.projections[2],
            &cache.projections[3], &cache.projections[4], &cache.scan, &dattn,
            scan_geometry(&g), config).map_err(err)?;
        let mut dln1 = be.zeros_f32(&[rows, c]).map_err(err)?;
        for (i, grad) in [scan_grad.q, scan_grad.k, scan_grad.v, scan_grad.beta, scan_grad.decay].iter().enumerate() {
            dln1 = add(&dln1, &lin(&cache.ln1, grad, i)?)?;
        }
        drop(lin);
        let (dtokens, dg1, db1) = norm(&cache.tokens, &dln1, &cache.gamma1)?;
        for (start, tensor) in [(o.ln1_gamma, dg1), (o.ln1_beta, db1), (o.ln2_gamma, dg2), (o.ln2_beta, db2)] {
            gradients.push((start..start+c, tensor));
        }
        // The input residual receives dpre (dy plus LN2 adjoint), not only dy.
        let dtokens = add(&dpre, &dtokens)?;
        let dx = be.vit_tokens_to_nchw_f32(&dtokens, g.batch, g.height, g.width, c, config).map_err(err)?;
        let mut nonfinite = be.any_non_finite_f32(&dx).map_err(err)?;
        for (_, gradient) in &gradients {
            nonfinite |= be.any_non_finite_f32(gradient).map_err(err)?;
        }
        if nonfinite {
            return Err(Error::InvalidConfig("Kimi KDA produced a nonfinite gradient".into()));
        }
        if let Some(generation) = cache.resident_generation {
            self.accumulate_resident_parameter_gradients(layer.index, generation, gradients, config)?;
        } else {
            // Explicit hybrid/Muon path: stage gradients in source array order.
            let mut dweights = vec![0.0; o.total_weights];
            let mut dbiases = vec![0.0; o.total_biases];
            for (range, tensor) in gradients {
                let values = be.download_f32(&tensor).map_err(err)?;
                if values.len() != range.len() {
                    return Err(Error::InvalidConfig("Kimi KDA gradient length mismatch".into()));
                }
                if range.end <= o.total_biases { dbiases[range].copy_from_slice(&values); }
                else { dweights[range.start-o.total_biases..range.end-o.total_biases].copy_from_slice(&values); }
            }
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let Some(LayerTrainBuffers::KimiKda(train)) = buffers.get_mut(layer.index) else {
                return Err(Error::InvalidConfig("missing Kimi KDA train buffers".into()));
            };
            if train.weight_updates.len() != dweights.len() || train.bias_updates.len() != dbiases.len() {
                return Err(Error::InvalidConfig("Kimi KDA train buffer length mismatch".into()));
            }
            for (dst, src) in train.weight_updates.iter_mut().zip(dweights) { *dst -= src; }
            for (dst, src) in train.bias_updates.iter_mut().zip(dbiases) { *dst -= src; }
        }
        self.accumulate_previous_gpu_gradient(layer.index, dx, layer_output_gradients)
    }

    /// Download the input, run the host Kimi KDA forward, upload the output.
    pub(crate) fn forward_kimi_kda_host(
        &self,
        layer: &Layer,
        spec: &KimiKdaSpec,
        weights: &super::KimiKdaParams,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::KimiKdaForwardCache)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download Kimi KDA input", err))?;
        let (data, cache) =
            super::kimi_kda_forward(&host_input.data, host_input.shape, spec, weights)
                .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
            .map_err(|err| backend_error("upload Kimi KDA output", err))?;
        Ok((output, cache))
    }
}

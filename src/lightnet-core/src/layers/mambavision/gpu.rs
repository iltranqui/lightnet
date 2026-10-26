//! Native MambaVision composition and exact input/parameter adjoints.
//! Parameters and update directions retain the CPU serialized block order.
//! Native SGD/Adam/Muon retain parameters, gradients and moments on device.
//! Boundary flags are read at backward entry.
use std::ops::Range;
use super::cpu::{MambaOffsets, MambaVisionGeometry};
use crate::layers::vit::gpu::device_activation;
use crate::error::{Error, Result};
use crate::model::{Layer, MambaVisionSpec};
use crate::runtime::{GpuExecutableNetwork, LayerTrainBuffers, ResidentMomentSlot, ResidentParameterRange,
    backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};
use lightnet_cuda::mambavision::{MambaScanForward, MambaScanGeometry};

pub(crate) fn supported(layer: &Layer, spec: &MambaVisionSpec) -> bool {
    MambaVisionGeometry::from_layer(layer.input_shape, spec).is_ok()
        && device_activation(&spec.activation).is_some()
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    geometry: MambaVisionGeometry,
    parameters: Vec<T>,
    resident_generation: Option<u64>,
    tokens: T,
    ln1: T,
    projected: T,
    x_pre: T,
    x: T,
    z_pre: T,
    selective: T,
    scan: MambaScanForward<T>,
    initial_state: T,
    msa: Option<(T, T, T, T)>,
    mixer: T,
    residual: T,
    ln2: T,
    ffn_pre: T,
    ffn_act: T,
}

// A stable index for every serialized block. Keeping parameters separate
// allows connected/depthwise/scan ops to use them without slicing on the host.
const IN_W: usize = 0;
const IN_B: usize = 1;
const X_W: usize = 2;
const X_B: usize = 3;
const Z_W: usize = 4;
const Z_B: usize = 5;
const SELECT_W: usize = 6;
const DT_W: usize = 7;
const DT_B: usize = 8;
const A_LOG: usize = 9;
const SKIP: usize = 10;
const OUT_W: usize = 11;
const OUT_B: usize = 12;
const RES_W: usize = 13;
const LN1_G: usize = 14;
const LN1_B: usize = 15;
const LN2_G: usize = 16;
const LN2_B: usize = 17;
const FF1_W: usize = 18;
const FF1_B: usize = 19;
const FF2_W: usize = 20;
const FF2_B: usize = 21;

fn ranges(o: &MambaOffsets) -> [Range<usize>; 22] {
    [o.in_w.clone(), o.in_b.clone(), o.conv_x_w.clone(), o.conv_x_b.clone(),
        o.conv_z_w.clone(), o.conv_z_b.clone(), o.x_proj_w.clone(),
        o.dt_proj_w.clone(), o.dt_proj_b.clone(), o.a_log.clone(), o.d_skip.clone(),
        o.out_w.clone(), o.out_b.clone(), o.res_w.clone().unwrap_or(0..0),
        o.ln1_gamma.clone(), o.ln1_beta.clone(), o.ln2_gamma.clone(), o.ln2_beta.clone(),
        o.ffn1_w.clone(), o.ffn1_b.clone(), o.ffn2_w.clone(), o.ffn2_b.clone()]
}

fn offsets(g: &MambaVisionGeometry) -> Result<MambaOffsets> {
    MambaOffsets::new(g.channels, g.filters, g.state_size, g.conv_size,
        g.dt_rank, g.ffn_ratio).map_err(Error::InvalidConfig)
}

fn scan_geometry(g: &MambaVisionGeometry) -> MambaScanGeometry {
    MambaScanGeometry { batch: g.batch, tokens: g.tokens(), channels: g.half(),
        state: g.state_size, rank: g.dt_rank }
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_mambavision_gpu(
        &self, layer: &Layer, spec: &MambaVisionSpec, params: &super::MambaVisionParams,
        input: &B::Tensor, config: ElementwiseLaunchConfig, training: bool,
    ) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)> {
        let g = MambaVisionGeometry::from_layer(layer.input_shape, spec).map_err(Error::InvalidConfig)?;
        params.validate().map_err(Error::InvalidConfig)?;
        if !params.matches(&g) || layer.output_shape.batch != g.batch
            || layer.output_shape.channels != g.filters || layer.output_shape.height != g.height
            || layer.output_shape.width != g.width {
            return Err(Error::InvalidConfig("MambaVision GPU parameter/output geometry mismatch".into()));
        }
        let activation = device_activation(&g.activation).ok_or_else(|| Error::UnsupportedLayer {
            layer_index: layer.index, name: "MambaVision GPU activation".into(),
        })?;
        let (b, t, c, n, d, rows, hidden) =
            (g.batch, g.tokens(), g.channels, g.filters, g.half(), g.rows(), g.hidden());
        let be = &self.backend;
        let err = |e| backend_error("GPU MambaVision forward", e);
        let o = offsets(&g)?;
        let (p,resident_generation) = if self.resident_parameter_path_enabled() {
            let layout = super::mambavision_param_blocks(params).map_err(Error::InvalidConfig)?
                .into_iter().enumerate().map(|(i,b)|ResidentParameterRange {
                    range:b.range, decay:b.decay, coupled_decay:b.decay,
                    moment_slot:Some(ResidentMomentSlot::ChildWeights(i)), active:true,
                }).collect();
            let (p,generation) = self.resident_parameter_slices(layer.index,&params.values,
                layout,&ranges(&o),config)?;
            (p,Some(generation))
        } else {
            let mut p = Vec::with_capacity(22);
            for range in ranges(&o) {
                p.push(if range.is_empty() { be.zeros_f32(&[1]).map_err(err)? }
                    else { be.upload_f32(&[range.len()], &params.values[range]).map_err(err)? });
            }
            (p,None)
        };
        let lin = |x: &B::Tensor, w: usize, bias: Option<usize>, ins: usize, outs: usize| {
            let zero;
            let bias = if let Some(i) = bias { &p[i] }
                else { zero = be.zeros_f32(&[outs]).map_err(err)?; &zero };
            be.connected_f32(x, &p[w], bias, rows, ins, outs, config).map_err(err)
        };
        let norm = |x: &B::Tensor, gamma: usize, beta: usize, cols: usize| {
            be.layernorm_forward_f32(x, Some(&p[gamma]), Some(&p[beta]),
                &[rows, cols], rows, cols, 1e-5, config).map_err(err)
        };
        let tokens = be.vit_nchw_to_tokens_f32(input, b, g.height, g.width, c, config).map_err(err)?;
        let ln1 = norm(&tokens, LN1_G, LN1_B, c)?;
        let projected = lin(&ln1, IN_W, Some(IN_B), c, n)?;
        let (x_pre, x) = be.mambavision_depthwise_f32(&projected, &p[X_W], &p[X_B],
            b, t, d, 0, g.conv_size, g.causal, config).map_err(err)?;
        let (z_pre, z) = be.mambavision_depthwise_f32(&projected, &p[Z_W], &p[Z_B],
            b, t, d, 1, g.conv_size, g.causal, config).map_err(err)?;
        // Deliberately consume flat BDT as rows, exactly as the CPU reference.
        let selective = lin(&x, SELECT_W, None, d, g.projection_width())?;
        let initial_state = be.zeros_f32(&[b, d, g.state_size]).map_err(err)?;
        let scan = be.mambavision_scan_f32(&x, &selective, &p[DT_W], &p[DT_B],
            &p[A_LOG], &p[SKIP], &initial_state, scan_geometry(&g), config).map_err(err)?;
        let cat = be.route_concat_f32(&[&scan.output, &z], &[0, 0], &[d, d], &[d, d],
            &[b, n, 1, t], t, config).map_err(err)?;
        let cat = be.vit_nchw_to_tokens_f32(&cat, b, 1, t, n, config).map_err(err)?;
        let (mixer, msa) = if g.msa_enabled {
            let (mixed, probabilities, active, boundary) = be.mambavision_msa_f32(&cat,
                b, t, n, g.msa_heads, g.msa_block_size, g.msa_topk, config).map_err(err)?;
            (mixed, Some((cat, probabilities, active, boundary)))
        } else { (cat, None) };
        let projected_out = lin(&mixer, OUT_W, Some(OUT_B), n, n)?;
        let skip = if o.res_w.is_some() { lin(&tokens, RES_W, None, c, n)? }
            else { tokens.clone() };
        let residual = be.residual_bias_f32(&projected_out, &skip, None,
            &[rows, n], rows, n, config).map_err(err)?;
        let ln2 = norm(&residual, LN2_G, LN2_B, n)?;
        let ffn_pre = lin(&ln2, FF1_W, Some(FF1_B), n, hidden)?;
        let ffn_act = be.activation_f32(&ffn_pre, activation, config).map_err(err)?;
        let ffn_out = lin(&ffn_act, FF2_W, Some(FF2_B), hidden, n)?;
        let output = be.residual_bias_f32(&residual, &ffn_out, None,
            &[rows, n], rows, n, config).map_err(err)?;
        let output = be.vit_tokens_to_nchw_f32(&output, b, g.height, g.width, n, config).map_err(err)?;
        let cache = training.then(|| DeviceCache { geometry: g, parameters: p, resident_generation, tokens,
            ln1, projected, x_pre, x, z_pre, selective, scan, initial_state,
            msa, mixer, residual, ln2, ffn_pre, ffn_act });
        Ok((output, cache))
    }

    pub(crate) fn backward_mambavision_gpu(
        &self, layer: &Layer, spec: &MambaVisionSpec,
        forward: &crate::runtime::GpuForwardState<B::Tensor>, grad: B::Tensor,
        config: ElementwiseLaunchConfig, layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let g = MambaVisionGeometry::from_layer(layer.input_shape, spec).map_err(Error::InvalidConfig)?;
        let cache = forward.mambavision_device_caches.get(layer.index).and_then(Option::as_ref).ok_or_else(||
            Error::InvalidConfig(format!("missing MambaVision device cache for layer #{}", layer.index)))?;
        if cache.geometry != g {
            return Err(Error::InvalidConfig("MambaVision device cache geometry mismatch".into()));
        }
        let activation = device_activation(&g.activation).ok_or_else(|| Error::UnsupportedLayer {
            layer_index: layer.index, name: "MambaVision GPU backward activation".into(),
        })?;
        let be = &self.backend;
        let err = |e| backend_error("GPU MambaVision backward", e);
        // Match the CPU's rejection of undefined piecewise derivatives.
        let boundary = be.download_f32(&cache.scan.boundary).map_err(err)?;
        if boundary.iter().any(|&x| x != 0.0) {
            return Err(Error::InvalidConfig("MambaVision scan derivative is undefined at a clamp boundary".into()));
        }
        if let Some((_, _, _, boundary)) = &cache.msa {
            if be.download_f32(boundary).map_err(err)?.iter().any(|&x| x != 0.0) {
                return Err(Error::InvalidConfig("MambaVision MSA score derivative is undefined for nonfinite or boundary scores".into()));
            }
        }
        let (b, t, c, n, d, rows, hidden) =
            (g.batch, g.tokens(), g.channels, g.filters, g.half(), g.rows(), g.hidden());
        let p = &cache.parameters;
        let o = offsets(&g)?;
        let descriptions = ranges(&o);
        let mut parameter_grads: Vec<(usize, B::Tensor)> = Vec::with_capacity(22);
        let lin = |x: &B::Tensor, dy: &B::Tensor, w: usize, ins: usize, outs: usize| {
            be.connected_backward_f32(x, &p[w], dy, rows, ins, outs, config).map_err(err)
        };
        let add = |x: &B::Tensor, y: &B::Tensor, cols: usize| {
            be.residual_bias_f32(x, y, None, &[rows, cols], rows, cols, config).map_err(err)
        };
        let norm = |x: &B::Tensor, dy: &B::Tensor, gamma: usize, cols: usize| {
            let dx = be.layernorm_backward_f32(dy, x, Some(&p[gamma]), &[rows, cols],
                rows, cols, 1e-5, config).map_err(err)?;
            let xhat = be.layernorm_forward_f32(x, None, None, &[rows, cols],
                rows, cols, 1e-5, config).map_err(err)?;
            let product = be.vector_mul_f32(dy, &xhat, &[rows, cols], config).map_err(err)?;
            Ok::<_, Error>((dx, be.bias_backward_f32(&product, cols, 1, config).map_err(err)?,
                be.bias_backward_f32(dy, cols, 1, config).map_err(err)?))
        };
        let dy = be.vit_nchw_to_tokens_f32(&grad, b, g.height, g.width, n, config).map_err(err)?;
        let (d_act, dw, db) = lin(&cache.ffn_act, &dy, FF2_W, hidden, n)?;
        parameter_grads.extend([(FF2_W, dw), (FF2_B, db)]);
        let d_pre = be.activation_backward_f32(&cache.ffn_pre, &d_act, activation, config).map_err(err)?;
        let (d_ln2, dw, db) = lin(&cache.ln2, &d_pre, FF1_W, n, hidden)?;
        parameter_grads.extend([(FF1_W, dw), (FF1_B, db)]);
        let (d_res, dg, db) = norm(&cache.residual, &d_ln2, LN2_G, n)?;
        parameter_grads.extend([(LN2_G, dg), (LN2_B, db)]);
        let d_res = add(&dy, &d_res, n)?;
        let mut d_tokens = if o.res_w.is_some() {
            let (dx, dw, _) = lin(&cache.tokens, &d_res, RES_W, c, n)?;
            parameter_grads.push((RES_W, dw));
            dx
        } else { d_res.clone() };
        let (mut d_mixer, dw, db) = lin(&cache.mixer, &d_res, OUT_W, n, n)?;
        parameter_grads.extend([(OUT_W, dw), (OUT_B, db)]);
        if let Some((input, probabilities, active, _)) = &cache.msa {
            d_mixer = be.mambavision_msa_backward_f32(input, probabilities, active, &d_mixer,
                b, t, n, g.msa_heads, config).map_err(err)?;
        }
        let d_cat = be.vit_tokens_to_nchw_f32(&d_mixer, b, 1, t, n, config).map_err(err)?;
        let split = be.route_backward_f32(&d_cat, &[vec![b, d, 1, t], vec![b, d, 1, t]],
            &[0, 0], &[d, d], &[d, d], t, config).map_err(err)?;
        if split.len() != 2 { return Err(Error::InvalidConfig("MambaVision mixer adjoint arity".into())); }
        let zero_state = be.zeros_f32(&[b, d, g.state_size]).map_err(err)?;
        let scan_grad = be.mambavision_scan_backward_f32(&cache.x, &cache.selective,
            &p[DT_W], &p[A_LOG], &p[SKIP], &cache.initial_state,
            &cache.scan.dt_pre, &cache.scan.states, &split[0], &zero_state,
            scan_geometry(&g), config).map_err(err)?;
        parameter_grads.extend([(DT_W, scan_grad.dt_weights), (DT_B, scan_grad.dt_bias),
            (A_LOG, scan_grad.a_log), (SKIP, scan_grad.d_skip)]);
        let (dx_select, dw, _) = lin(&cache.x, &scan_grad.projection, SELECT_W, d, g.projection_width())?;
        parameter_grads.push((SELECT_W, dw));
        let dx = add(&scan_grad.input, &dx_select, d)?;
        let (dx_projected, dw, db) = be.mambavision_depthwise_backward_f32(&cache.projected,
            &p[X_W], &cache.x_pre, &dx, b, t, d, 0, g.conv_size, g.causal, config).map_err(err)?;
        parameter_grads.extend([(X_W, dw), (X_B, db)]);
        let (dz_projected, dw, db) = be.mambavision_depthwise_backward_f32(&cache.projected,
            &p[Z_W], &cache.z_pre, &split[1], b, t, d, 1, g.conv_size, g.causal, config).map_err(err)?;
        parameter_grads.extend([(Z_W, dw), (Z_B, db)]);
        let d_projected = add(&dx_projected, &dz_projected, n)?;
        let (d_ln1, dw, db) = lin(&cache.ln1, &d_projected, IN_W, c, n)?;
        parameter_grads.extend([(IN_W, dw), (IN_B, db)]);
        let (dx, dg, db) = norm(&cache.tokens, &d_ln1, LN1_G, c)?;
        parameter_grads.extend([(LN1_G, dg), (LN1_B, db)]);
        d_tokens = add(&d_tokens, &dx, c)?;
        let dx = be.vit_tokens_to_nchw_f32(&d_tokens, b, g.height, g.width, c, config).map_err(err)?;
        if let Some(generation) = cache.resident_generation {
            let gradients = parameter_grads.into_iter()
                .map(|(block,tensor)|(descriptions[block].clone(),tensor)).collect();
            self.accumulate_resident_parameter_gradients(layer.index,generation,gradients,config)?;
            return self.accumulate_previous_gpu_gradient(layer.index,dx,layer_output_gradients);
        }
        // Stage complete gradients before mutating accumulation buffers.
        let mut updates = vec![0.0; o.total];
        for (block, tensor) in parameter_grads {
            let values = be.download_f32(&tensor).map_err(err)?;
            let range = descriptions[block].clone();
            if values.len() != range.len() {
                return Err(Error::InvalidConfig("MambaVision parameter gradient length mismatch".into()));
            }
            updates[range].copy_from_slice(&values);
        }
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::MambaVision(train)) = buffers.get_mut(layer.index) else {
            return Err(Error::InvalidConfig("missing MambaVision train buffers".into()));
        };
        if train.updates.len() != updates.len() {
            return Err(Error::InvalidConfig("MambaVision train buffer length mismatch".into()));
        }
        for (dst, gradient) in train.updates.iter_mut().zip(updates) { *dst -= gradient; }
        drop(buffers);
        self.accumulate_previous_gpu_gradient(layer.index, dx, layer_output_gradients)
    }

    /// Download the input, run the host MambaVision forward, upload the output.
    pub(crate) fn forward_mambavision_host(
        &self,
        layer: &Layer,
        spec: &MambaVisionSpec,
        weights: &super::MambaVisionParams,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::MambaVisionForward)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download MambaVision input", err))?;
        let forward = super::mambavision_forward(
            &host_input.data,
            host_input.shape,
            spec,
            weights,
            None,
        )
        .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &forward.output)
            .map_err(|err| backend_error("upload MambaVision output", err))?;
        Ok((output, forward))
    }
}

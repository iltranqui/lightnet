//! Native f32 Clifford runtime. Activations and their adjoints stay on the
//! device; parameter gradients and BN statistics are staged for host updates.
//! Fourier branches retain the reported Rust host fallback. No C++ golden.
use super::cpu::*;
use crate::error::{Error, Result};
use crate::model::{
    CliffordContextMode, CliffordHigherMode, CliffordInteractionMode, CliffordSpec, Layer, TensorShape,
};
use crate::runtime::{
    BATCH_NORM_EPSILON, BATCH_NORM_ROLLING_KEEP_CPU, ConvolutionalParams,
    GpuExecutableNetwork, LayerParams, LayerTrainBuffers, backend_error, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

pub(crate) fn needs_host(spec: &CliffordSpec) -> bool {
    has_fourier(spec)
}

fn device_error(error: impl std::fmt::Display) -> Error {
    backend_error("GPU clifford", error)
}

#[derive(Clone, Debug)]
struct LinearCache<T> {
    rows: T,
    weights: T,
    output: T,
    inputs: usize,
    outputs: usize,
}

#[derive(Clone, Debug)]
struct BranchCache<T> {
    projection: LinearCache<T>,
    gate_projection: LinearCache<T>,
    gate: T,
}

#[derive(Clone, Debug)]
struct DepthwiseCache<T> {
    input: T,
    weights: T,
    xhat: T,
    scales: T,
    mean: T,
    variance: T,
    // Only inference uses this rolling-stat affine slope.
    slope: Option<T>,
}

#[derive(Clone, Debug)]
struct LocalCache<T> {
    determinant: LinearCache<T>,
    depthwise: Vec<DepthwiseCache<T>>,
    context_pre: T,
    context: T,
    branch: BranchCache<T>,
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    shape: TensorShape,
    spec: CliffordSpec,
    training: bool,
    input_rows: T,
    gamma: T,
    normalized: T,
    local: Option<LocalCache<T>>,
    global: Option<(T, BranchCache<T>)>,
    mixed: T,
    scale: T,
    mask: T,
}

struct Ops<'a, B: F32TensorOpsBackend> {
    be: &'a B,
    shape: TensorShape,
    config: ElementwiseLaunchConfig,
}

impl<B: F32TensorOpsBackend> Ops<'_, B>
where B::Error: std::fmt::Display {
    fn spatial(&self) -> usize { self.shape.height * self.shape.width }
    fn dims(&self, channels: usize) -> [usize; 4] {
        [self.shape.batch, channels, self.shape.height, self.shape.width]
    }
    fn upload(&self, data: &[f32]) -> Result<B::Tensor> {
        self.be.upload_f32(&[data.len()], data).map_err(device_error)
    }
    fn zeros(&self, channels: usize) -> Result<B::Tensor> {
        self.be.zeros_f32(&self.dims(channels)).map_err(device_error)
    }
    fn add(&self, x: &B::Tensor, y: &B::Tensor) -> Result<B::Tensor> {
        self.be.saxpy_f32(1.0, x, y, self.config).map_err(device_error)
    }
    fn scaled_add(&self, alpha: f32, x: &B::Tensor, y: &B::Tensor) -> Result<B::Tensor> {
        self.be.saxpy_f32(alpha, x, y, self.config).map_err(device_error)
    }
    fn mul(&self, x: &B::Tensor, y: &B::Tensor) -> Result<B::Tensor> {
        self.be.vector_mul_f32(x, y, &self.dims(self.shape.channels), self.config).map_err(device_error)
    }
    fn activate(&self, x: &B::Tensor, kind: F32ActivationKind) -> Result<B::Tensor> {
        self.be.activation_f32(x, kind, self.config).map_err(device_error)
    }
    fn slope(&self, x: &B::Tensor, grad: &B::Tensor, kind: F32ActivationKind) -> Result<B::Tensor> {
        self.be.activation_backward_f32(x, grad, kind, self.config).map_err(device_error)
    }
    fn rows(&self, x: &B::Tensor, channels: usize) -> Result<B::Tensor> {
        self.be.vit_nchw_to_tokens_f32(x, self.shape.batch, self.shape.height,
            self.shape.width, channels, self.config).map_err(device_error)
    }
    fn nchw(&self, x: &B::Tensor, channels: usize) -> Result<B::Tensor> {
        self.be.vit_tokens_to_nchw_f32(x, self.shape.batch, self.shape.height,
            self.shape.width, channels, self.config).map_err(device_error)
    }
    fn concat(&self, inputs: &[&B::Tensor], channels: &[usize]) -> Result<B::Tensor> {
        self.be.route_concat_f32(inputs, &vec![0; inputs.len()], channels, channels,
            &self.dims(channels.iter().sum()), self.spatial(), self.config).map_err(device_error)
    }
    fn split(&self, grad: &B::Tensor, channels: &[usize]) -> Result<Vec<B::Tensor>> {
        self.be.route_backward_f32(grad,
            &channels.iter().map(|&c| self.dims(c).to_vec()).collect::<Vec<_>>(),
            &vec![0; channels.len()], channels, channels, self.spatial(), self.config).map_err(device_error)
    }
    fn linear(&self, x: &B::Tensor, w: &[f32], bias: &[f32], inputs: usize) -> Result<LinearCache<B::Tensor>> {
        let outputs = bias.len();
        let rows = self.rows(x, inputs)?;
        let weights = self.be.upload_f32(&[outputs, inputs], w).map_err(device_error)?;
        let bias = self.upload(bias)?;
        let out = self.be.connected_f32(&rows, &weights, &bias,
            self.shape.batch * self.spatial(), inputs, outputs, self.config).map_err(device_error)?;
        Ok(LinearCache { rows, weights, output: self.nchw(&out, outputs)?, inputs, outputs })
    }
    fn linear_backward(&self, cache: &LinearCache<B::Tensor>, grad: &B::Tensor)
        -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
        let dy = self.rows(grad, cache.outputs)?;
        let (dx, dw, db) = self.be.connected_backward_f32(&cache.rows, &cache.weights, &dy,
            self.shape.batch * self.spatial(), cache.inputs, cache.outputs, self.config).map_err(device_error)?;
        Ok((self.nchw(&dx, cache.inputs)?, dw, db))
    }
    fn interaction(&self, x: &B::Tensor, ctx: &B::Tensor, shift: usize, inner: bool, pooled: bool) -> Result<B::Tensor> {
        self.be.clifford_interaction_f32(x, ctx, self.shape.batch, self.shape.channels,
            self.spatial(), shift, inner, pooled, self.config).map_err(device_error)
    }
    fn interaction_backward(&self, x: &B::Tensor, ctx: &B::Tensor, grad: &B::Tensor,
        shift: usize, inner: bool, pooled: bool) -> Result<(B::Tensor, B::Tensor)> {
        self.be.clifford_interaction_backward_f32(x, ctx, grad, self.shape.batch, self.shape.channels,
            self.spatial(), shift, inner, pooled, self.config).map_err(device_error)
    }
    fn branch(&self, spec: &CliffordSpec, params: &CliffordBranchParams,
        normalized: &B::Tensor, state: &B::Tensor, context: &B::Tensor, pooled: bool) -> Result<BranchCache<B::Tensor>> {
        let schedule = schedule(spec);
        let blocks = schedule.iter().map(|&(shift, inner)| self.interaction(state, context, shift, inner, pooled))
            .collect::<Result<Vec<_>>>()?;
        let c = self.shape.channels;
        let raw = self.concat(&blocks.iter().collect::<Vec<_>>(), &vec![c; blocks.len()])?;
        let projection = self.linear(&raw, &params.projection_weights, &params.projection_biases, c * blocks.len())?;
        let gate_input = self.concat(&[normalized, &projection.output], &[c, c])?;
        let gate_projection = self.linear(&gate_input, &params.gate_weights, &params.gate_biases, 2 * c)?;
        let gate = self.activate(&gate_projection.output, F32ActivationKind::Logistic)?;
        Ok(BranchCache { projection, gate_projection, gate })
    }
    fn depthwise(&self, params: &ConvolutionalParams, input: &B::Tensor, size: usize, training: bool)
        -> Result<(B::Tensor, DepthwiseCache<B::Tensor>)> {
        let c = self.shape.channels;
        let dims = self.dims(c);
        let weights = self.be.upload_f32(&[c, 1, size, size], &params.weights).map_err(device_error)?;
        let raw = self.be.conv2d_f32(input, &weights, &dims, c, self.shape.height, self.shape.width,
            c, size, 1, 1, 1, size / 2, c, self.config).map_err(device_error)?;
        let bn = params.batch_norm.as_ref().ok_or_else(|| Error::InvalidConfig("[clifford] missing depthwise BN".to_owned()))?;
        let scales = self.upload(&bn.scales)?;
        // As the CPU sub-convolution, BN uses conv.biases as its shift.
        let bias = self.upload(&params.biases)?;
        let mean = self.upload(&bn.rolling_mean)?;
        let variance = self.upload(&bn.rolling_variance)?;
        if training {
            let result = self.be.batchnorm_train_forward_f32(&raw, &scales, &bias, &mean, &variance,
                &dims, BATCH_NORM_ROLLING_KEEP_CPU, self.config).map_err(device_error)?;
            Ok((result.output, DepthwiseCache { input: input.clone(), weights, xhat: result.x_hat,
                scales, mean: result.mean, variance: result.variance, slope: None }))
        } else {
            let inv = bn.rolling_variance.iter().map(|v| 1.0 / (v + BATCH_NORM_EPSILON).sqrt()).collect::<Vec<_>>();
            let offset = self.upload(&bn.rolling_mean.iter().zip(&inv).map(|(m, s)| -m * s).collect::<Vec<_>>())?;
            let slope = self.upload(&bn.scales.iter().zip(&inv).map(|(g, s)| g * s).collect::<Vec<_>>())?;
            let inv = self.upload(&inv)?;
            let centered = self.be.channel_scale_f32(&raw, &inv, c, self.spatial(), self.config).map_err(device_error)?;
            let xhat = self.be.bias_add_f32(&centered, &offset, c, self.spatial(), self.config).map_err(device_error)?;
            let scaled = self.be.channel_scale_f32(&xhat, &scales, c, self.spatial(), self.config).map_err(device_error)?;
            let output = self.be.bias_add_f32(&scaled, &bias, c, self.spatial(), self.config).map_err(device_error)?;
            Ok((output, DepthwiseCache { input: input.clone(), weights, xhat, scales, mean, variance, slope: Some(slope) }))
        }
    }

    fn host(&self, tensor: &B::Tensor) -> Result<Vec<f32>> {
        self.be.download_f32(tensor).map_err(device_error)
    }

    fn branch_backward(&self, spec: &CliffordSpec, cache: &BranchCache<B::Tensor>,
        state: &B::Tensor, context: &B::Tensor, pooled: bool, grad: &B::Tensor,
        extra_features: Option<&B::Tensor>)
        -> Result<(B::Tensor, B::Tensor, B::Tensor, CliffordBranchParams)> {
        let c = self.shape.channels;
        let feature = &cache.projection.output;
        let d_gate = self.mul(grad, feature)?;
        let d_logits = self.slope(&cache.gate_projection.output, &d_gate, F32ActivationKind::Logistic)?;
        let (d_gate_input, dgw, dgb) = self.linear_backward(&cache.gate_projection, &d_logits)?;
        let parts = self.split(&d_gate_input, &[c, c])?;
        let mut d_feature = self.add(&self.mul(grad, &cache.gate)?, &parts[1])?;
        if let Some(extra) = extra_features { d_feature = self.add(&d_feature, extra)?; }
        let (d_raw, dpw, dpb) = self.linear_backward(&cache.projection, &d_feature)?;
        let schedule = schedule(spec);
        let raw_parts = self.split(&d_raw, &vec![c; schedule.len()])?;
        let mut ds = self.zeros(c)?;
        let mut dc = if pooled { self.be.zeros_f32(&[self.shape.batch, c, 1, 1]).map_err(device_error)? }
            else { self.zeros(c)? };
        for ((shift, inner), block) in schedule.into_iter().zip(&raw_parts) {
            let (dx, dy) = self.interaction_backward(state, context, block, shift, inner, pooled)?;
            ds = self.add(&ds, &dx)?;
            dc = self.add(&dc, &dy)?;
        }
        // Only parameter gradients cross the host boundary.
        let params = CliffordBranchParams { projection_weights: self.host(&dpw)?,
            projection_biases: self.host(&dpb)?, gate_weights: self.host(&dgw)?, gate_biases: self.host(&dgb)? };
        Ok((parts[0].clone(), ds, dc, params))
    }

    fn depthwise_backward(&self, cache: &DepthwiseCache<B::Tensor>, grad: &B::Tensor, size: usize)
        -> Result<(B::Tensor, B::Tensor, B::Tensor, B::Tensor)> {
        let c = self.shape.channels;
        let dims = self.dims(c);
        let (raw_grad, dg, db) = if let Some(slope) = &cache.slope {
            let db = self.be.bias_backward_f32(grad, c, self.spatial(), self.config).map_err(device_error)?;
            let product = self.mul(grad, &cache.xhat)?;
            let dg = self.be.bias_backward_f32(&product, c, self.spatial(), self.config).map_err(device_error)?;
            let dx = self.be.channel_scale_f32(grad, slope, c, self.spatial(), self.config).map_err(device_error)?;
            (dx, dg, db)
        } else {
            self.be.batchnorm_train_backward_f32(grad, &cache.xhat, &cache.variance, &cache.scales,
                &dims, self.config).map_err(device_error)?
        };
        let dx = self.be.conv2d_backward_input_f32(&raw_grad, &cache.weights, &dims, c,
            self.shape.height, self.shape.width, c, size, 1, 1, 1, size / 2, c, self.config).map_err(device_error)?;
        let dw = self.be.conv2d_backward_weights_f32(&cache.input, &raw_grad, &[c, 1, size, size], c,
            self.shape.height, self.shape.width, c, size, 1, 1, 1, size / 2, c, self.config).map_err(device_error)?;
        Ok((dx, dw, db, dg))
    }
}

// Full mode interleaves matching wedge/inner slots, then appends the longer
// schedule's tail. Shift order and duplicates are part of the CPU contract.
fn schedule(spec: &CliffordSpec) -> Vec<(usize, bool)> {
    let wedge = spec.interaction_mode != CliffordInteractionMode::Inner;
    let inner = spec.interaction_mode != CliffordInteractionMode::Wedge;
    let mut slots = Vec::new();
    for i in 0..spec.wedge_shifts.len().max(spec.inner_shifts.len()) {
        if wedge && let Some(&shift) = spec.wedge_shifts.get(i) { slots.push((shift, false)); }
        if inner && let Some(&shift) = spec.inner_shifts.get(i) { slots.push((shift, true)); }
    }
    slots
}

fn higher(spec: &CliffordSpec) -> bool {
    has_local(spec) && spec.higher_mode == CliffordHigherMode::VectorBivector
        && spec.interaction_mode != CliffordInteractionMode::Inner
}

impl<B: F32TensorOpsBackend> GpuExecutableNetwork<B>
where B::Error: std::fmt::Display {
    pub(crate) fn forward_clifford_gpu(
        &self, layer: &Layer, spec: &CliffordSpec, params: &CliffordParams,
        input: &B::Tensor, training: bool, native: bool, mask: &[f32], config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>, Option<CliffordRuntimeCache>)> {
        validate_params(spec, params, layer.input_shape)?;
        if layer.input_shape != layer.output_shape || spec.filters != layer.input_shape.channels
            || !spec.drop_path.is_finite() || !(0.0..1.0).contains(&spec.drop_path)
            || spec.dwconv_size == 0 || spec.dwconv_size % 2 == 0 || schedule(spec).is_empty() {
            return Err(Error::InvalidConfig("[clifford] invalid GPU geometry or configuration".to_owned()));
        }
        let keep = 1.0 / (1.0 - spec.drop_path);
        if mask.len() != layer.input_shape.batch || mask.iter().any(|&v|
            !v.is_finite() || if training && spec.drop_path > 0.0 { v != 0.0 && (v - keep).abs() > 1e-6 } else { v != 1.0 }) {
            return Err(Error::InvalidConfig("[clifford] invalid DropPath sample mask".to_owned()));
        }
        if needs_host(spec) || (training && !native) {
            eprintln!("[clifford] layer #{}: host fallback ({})", layer.index,
                if needs_host(spec) { "Fourier global operator" } else { "network requires CPU backward" });
            let host = self.download_tensor(input, layer.input_shape).map_err(device_error)?;
            let (out, cache) = forward(spec, params, &host, training, Some(mask))?;
            let output = self.backend.upload_f32(&tensor_backend_shape(out.shape), &out.data).map_err(device_error)?;
            return Ok((output, None, Some(cache)));
        }
        let op = Ops { be: &self.backend, shape: layer.input_shape, config };
        let c = layer.input_shape.channels;
        let rows = layer.input_shape.batch * op.spatial();
        let input_rows = op.rows(input, c)?;
        let gamma = op.upload(&params.layer_norm_gamma)?;
        let beta = op.upload(&params.layer_norm_beta)?;
        let norm_rows = self.backend.layernorm_forward_f32(&input_rows, Some(&gamma), Some(&beta),
            &[rows, c], rows, c, 1e-5, config).map_err(device_error)?;
        let normalized = op.nchw(&norm_rows, c)?;
        let mut mixed = op.activate(&normalized, F32ActivationKind::Swish)?;
        let local = if has_local(spec) {
            let determinant = op.linear(&normalized, &params.determinant_weights, &params.determinant_biases, c)?;
            let mut current = normalized.clone();
            let mut depthwise = Vec::new();
            for conv in &params.depthwise {
                let (output, cache) = op.depthwise(conv, &current, spec.dwconv_size, training)?;
                current = output;
                depthwise.push(cache);
            }
            let context_pre = current;
            let mut context = op.activate(&context_pre, F32ActivationKind::Swish)?;
            if spec.context_mode == CliffordContextMode::Diff {
                context = op.scaled_add(-1.0, &determinant.output, &context)?;
            }
            let branch = op.branch(spec, &params.local, &normalized, &determinant.output, &context, false)?;
            mixed = op.add(&mixed, &op.mul(&branch.gate, &branch.projection.output)?)?;
            if higher(spec) && !spec.wedge_shifts.is_empty() {
                let scale = 1.0 / spec.wedge_shifts.len() as f32;
                for &shift in &spec.wedge_shifts {
                    let term = op.interaction(&determinant.output, &branch.projection.output, shift, false, false)?;
                    mixed = op.scaled_add(scale, &term, &mixed)?;
                }
            }
            Some(LocalCache { determinant, depthwise, context_pre, context, branch })
        } else { None };
        let global = if has_global(spec) {
            let pooled = self.backend.avgpool_f32(&normalized, &[layer.input_shape.batch, c, 1, 1],
                c, layer.input_shape.height, layer.input_shape.width, config).map_err(device_error)?;
            let params = params.global.as_ref().ok_or_else(|| Error::InvalidConfig("[clifford] missing global weights".to_owned()))?;
            let branch = op.branch(spec, params, &normalized, &normalized, &pooled, true)?;
            mixed = op.add(&mixed, &op.mul(&branch.gate, &branch.projection.output)?)?;
            Some((pooled, branch))
        } else { None };
        let scale = op.upload(&params.layer_scale)?;
        let mask = op.upload(mask)?;
        let output = self.backend.clifford_residual_f32(input, &mixed, &scale, &mask,
            layer.input_shape.batch, c, op.spatial(), config).map_err(device_error)?;
        let cache = training.then_some(DeviceCache { shape: layer.input_shape, spec: spec.clone(), training,
            input_rows, gamma, normalized, local, global, mixed, scale, mask });
        Ok((output, cache, None))
    }

    pub(crate) fn backward_clifford_gpu(
        &self, layer: &Layer, spec: &CliffordSpec, params: &CliffordParams,
        forward: &crate::runtime::GpuForwardState<B::Tensor>,
        grad: B::Tensor, config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        validate_params(spec, params, layer.input_shape)?;
        let caches = &forward.clifford_device_caches;
        let cache = caches.get(layer.index).and_then(Option::as_ref).ok_or_else(||
            Error::InvalidConfig(format!("missing [clifford] device training cache at layer #{}", layer.index)))?;
        if cache.shape != layer.input_shape || cache.spec != *spec || !cache.training {
            return Err(Error::InvalidConfig("[clifford] device cache geometry/training mismatch".to_owned()));
        }
        let op = Ops { be: &self.backend, shape: cache.shape, config };
        let c = cache.shape.channels;
        let rows = cache.shape.batch * op.spatial();
        let mut gradients = params.zero_gradients_like();
        let (d_mixed, d_scale) = self.backend.clifford_residual_backward_f32(&cache.mixed, &grad,
            &cache.scale, &cache.mask, cache.shape.batch, c, op.spatial(), config).map_err(device_error)?;
        gradients.layer_scale = op.host(&d_scale)?;
        let mut d_norm = op.slope(&cache.normalized, &d_mixed, F32ActivationKind::Swish)?;
        if let Some(local) = &cache.local {
            let mut d_det = op.zeros(c)?;
            let mut d_feature = op.zeros(c)?;
            if higher(spec) && !spec.wedge_shifts.is_empty() {
                let scale = 1.0 / spec.wedge_shifts.len() as f32;
                for &shift in &spec.wedge_shifts {
                    let (dd, df) = op.interaction_backward(&local.determinant.output,
                        &local.branch.projection.output, &d_mixed, shift, false, false)?;
                    d_det = op.scaled_add(scale, &dd, &d_det)?;
                    d_feature = op.scaled_add(scale, &df, &d_feature)?;
                }
            }
            let (dn, dd, mut dc, branch_gradient) = op.branch_backward(spec, &local.branch,
                &local.determinant.output, &local.context, false, &d_mixed, Some(&d_feature))?;
            gradients.local = branch_gradient;
            d_norm = op.add(&d_norm, &dn)?;
            d_det = op.add(&d_det, &dd)?;
            if spec.context_mode == CliffordContextMode::Diff {
                d_det = op.scaled_add(-1.0, &dc, &d_det)?;
            }
            dc = op.slope(&local.context_pre, &dc, F32ActivationKind::Swish)?;
            for (i, sub) in local.depthwise.iter().enumerate().rev() {
                let (dx, dw, db, dg) = op.depthwise_backward(sub, &dc, spec.dwconv_size)?;
                gradients.depthwise[i].weights = op.host(&dw)?;
                gradients.depthwise[i].biases = op.host(&db)?;
                if let Some(bn) = &mut gradients.depthwise[i].batch_norm { bn.scales = op.host(&dg)?; }
                dc = dx;
            }
            d_norm = op.add(&d_norm, &dc)?;
            let (dn, dw, db) = op.linear_backward(&local.determinant, &d_det)?;
            gradients.determinant_weights = op.host(&dw)?;
            gradients.determinant_biases = op.host(&db)?;
            d_norm = op.add(&d_norm, &dn)?;
        }
        if let Some((pooled, global)) = &cache.global {
            let (dn, ds, dc, branch_gradient) = op.branch_backward(spec, global, &cache.normalized,
                pooled, true, &d_mixed, None)?;
            gradients.global = Some(branch_gradient);
            let d_pooled = self.backend.avgpool_backward_f32(&dc, &op.dims(c), c,
                cache.shape.height, cache.shape.width, config).map_err(device_error)?;
            d_norm = op.add(&op.add(&op.add(&d_norm, &dn)?, &ds)?, &d_pooled)?;
        }
        let d_rows = op.rows(&d_norm, c)?;
        let dx_rows = self.backend.layernorm_backward_f32(&d_rows, &cache.input_rows, Some(&cache.gamma),
            &[rows, c], rows, c, 1e-5, config).map_err(device_error)?;
        let xhat = self.backend.layernorm_forward_f32(&cache.input_rows, None, None,
            &[rows, c], rows, c, 1e-5, config).map_err(device_error)?;
        let product = self.backend.vector_mul_f32(&d_rows, &xhat, &[rows, c], config).map_err(device_error)?;
        gradients.layer_norm_gamma = op.host(&self.backend.bias_backward_f32(&product, c, 1, config).map_err(device_error)?)?;
        gradients.layer_norm_beta = op.host(&self.backend.bias_backward_f32(&d_rows, c, 1, config).map_err(device_error)?)?;
        let dx = op.add(&grad, &op.nchw(&dx_rows, c)?)?;
        let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
        let Some(LayerTrainBuffers::Clifford(train)) = buffers.get_mut(layer.index) else {
            return Err(Error::InvalidConfig("[clifford] missing host train buffers".to_owned()));
        };
        stage(&mut train.layer_norm_gamma_updates, &gradients.layer_norm_gamma)?;
        stage(&mut train.layer_norm_beta_updates, &gradients.layer_norm_beta)?;
        stage(&mut train.determinant_weight_updates, &gradients.determinant_weights)?;
        stage(&mut train.determinant_bias_updates, &gradients.determinant_biases)?;
        stage(&mut train.layer_scale_updates, &gradients.layer_scale)?;
        stage_branch(&mut train.local, &gradients.local)?;
        match (&mut train.global, &gradients.global) {
            (Some(train), Some(gradient)) => stage_branch(train, gradient)?,
            (None, None) => {}
            _ => return Err(Error::InvalidConfig("[clifford] global train buffer mismatch".to_owned())),
        }
        if train.depthwise.len() != gradients.depthwise.len() {
            return Err(Error::InvalidConfig("[clifford] depthwise train buffer count mismatch".to_owned()));
        }
        for (train, gradient) in train.depthwise.iter_mut().zip(&gradients.depthwise) {
            stage(&mut train.weight_updates, &gradient.weights)?;
            stage(&mut train.bias_updates, &gradient.biases)?;
            match (&mut train.batch_norm, &gradient.batch_norm) {
                (Some(train), Some(gradient)) => stage(&mut train.scale_updates, &gradient.scales)?,
                (None, None) => {}
                _ => return Err(Error::InvalidConfig("[clifford] BN train buffer mismatch".to_owned())),
            }
        }
        drop(buffers);
        self.accumulate_previous_gpu_gradient(layer.index, dx, layer_output_gradients)
    }

    /// CPU-compatible biased batch variance and EMA (0.9 keep), applied at
    /// the same boundary as CPU backward accumulation, never during forward.
    pub(crate) fn update_clifford_rolling_stats_gpu(&mut self,
        forward: &crate::runtime::GpuForwardState<B::Tensor>) -> Result<()> {
        let caches = &forward.clifford_device_caches;
        for (index, cache) in caches.iter().enumerate() {
            let Some(cache) = cache.as_ref().filter(|c| c.training) else { continue; };
            let Some(local) = &cache.local else { continue; };
            let Some(LayerParams::Clifford(params)) = self.params.get_mut(index) else {
                return Err(Error::InvalidConfig("[clifford] missing parameters for BN statistics".to_owned()));
            };
            if params.depthwise.len() != local.depthwise.len() {
                return Err(Error::InvalidConfig("[clifford] BN cache count mismatch".to_owned()));
            }
            for (conv, sub) in params.depthwise.iter_mut().zip(&local.depthwise) {
                let bn = conv.batch_norm.as_mut().ok_or_else(|| Error::InvalidConfig("[clifford] missing BN statistics".to_owned()))?;
                let mean = self.backend.download_f32(&sub.mean).map_err(device_error)?;
                let variance = self.backend.download_f32(&sub.variance).map_err(device_error)?;
                if mean.len() != bn.rolling_mean.len() || variance.len() != bn.rolling_variance.len() {
                    return Err(Error::InvalidConfig("[clifford] BN statistic length mismatch".to_owned()));
                }
                for (dst, src) in bn.rolling_mean.iter_mut().zip(mean) {
                    *dst = BATCH_NORM_ROLLING_KEEP_CPU * *dst + (1.0 - BATCH_NORM_ROLLING_KEEP_CPU) * src;
                }
                for (dst, src) in bn.rolling_variance.iter_mut().zip(variance) {
                    *dst = BATCH_NORM_ROLLING_KEEP_CPU * *dst + (1.0 - BATCH_NORM_ROLLING_KEEP_CPU) * src;
                }
            }
        }
        Ok(())
    }
}

// Clifford CPU buffers accumulate positive derivatives (unlike ViT's sign).
fn stage(dst: &mut [f32], gradient: &[f32]) -> Result<()> {
    if dst.len() != gradient.len() {
        return Err(Error::InvalidConfig("[clifford] parameter gradient length mismatch".to_owned()));
    }
    for (dst, src) in dst.iter_mut().zip(gradient) { *dst += src; }
    Ok(())
}

fn stage_branch(train: &mut CliffordBranchTrainBuffers, gradient: &CliffordBranchParams) -> Result<()> {
    stage(&mut train.projection_weight_updates, &gradient.projection_weights)?;
    stage(&mut train.projection_bias_updates, &gradient.projection_biases)?;
    stage(&mut train.gate_weight_updates, &gradient.gate_weights)?;
    stage(&mut train.gate_bias_updates, &gradient.gate_biases)
}

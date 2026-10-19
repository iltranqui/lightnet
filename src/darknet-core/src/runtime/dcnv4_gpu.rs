//! Bounded DCNv4 GPU math: <=169 active points, elementwise/channel activation.
//! BN normalizes with rolling stats; its 0.9 EMA runs only after backward.
//! Parameter gradients/SGD/Adam stay host-resident; forward/backward tensor math
//! is native on Oxide. Backward recomputes offsets, never uses last-forward state.

use super::*;
use crate::model::Layer;
use darknet_cuda::host_ops::dcnv4::{Dcnv4Composition, Dcnv4Geometry};

pub(super) fn needs_host(spec: &Dcnv4Spec) -> bool {
    spec.size
        .checked_mul(spec.size)
        .and_then(|v| v.checked_sub(usize::from(spec.remove_center)))
        .is_none_or(|points| points > 169)
        || (spec.activation.to_cuda_f32_kind().is_none()
            && spec.activation.to_channel_normalization().is_none())
}

fn descriptor(layer: &Layer, spec: &Dcnv4Spec, params: &Dcnv4Params) -> Result<Dcnv4Composition> {
    let i = layer.input_shape;
    let o = layer.output_shape;
    let desc = Dcnv4Composition {
        geometry: Dcnv4Geometry {
            batch: i.batch,
            channels: i.channels,
            groups: spec.groups,
            input_height: i.height,
            input_width: i.width,
            output_height: o.height,
            output_width: o.width,
            size: spec.size,
            stride_x: spec.stride_x,
            stride_y: spec.stride_y,
            padding: spec.padding,
            dilation: spec.dilation,
            softmax: spec.softmax,
            remove_center: spec.remove_center,
        },
        d_stride: spec.d_stride,
        offset_scale: spec.offset_scale,
    };
    desc.checked().map_err(Error::InvalidConfig)?;
    if params.offset_weights.len() != desc.weight_count()
        || params.offset_biases.len() != desc.geometry.offset_dim()
        || params.biases.len() != spec.filters
        || params.batch_norm.is_some() != spec.batch_normalize
        || params.batch_norm.as_ref().is_some_and(|bn| {
            [
                bn.scales.len(),
                bn.biases.len(),
                bn.rolling_mean.len(),
                bn.rolling_variance.len(),
            ]
            .iter()
            .any(|&len| len != spec.filters)
        })
    {
        return Err(Error::InvalidConfig(
            "DCNv4 parameter storage mismatch".to_owned(),
        ));
    }
    Ok(desc)
}

pub(super) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    input: &B::Tensor,
    layer: &Layer,
    spec: &Dcnv4Spec,
    params: &Dcnv4Params,
) -> Result<(B::Tensor, B::Tensor)>
where
    B::Error: std::fmt::Display,
{
    if needs_host(spec) {
        return Err(Error::InvalidConfig(
            "DCNv4 device path selected for a host-only variant".to_owned(),
        ));
    }
    let cfg = ElementwiseLaunchConfig::default();
    let desc = descriptor(layer, spec, params)?;
    let weights = backend
        .upload_f32(&[params.offset_weights.len()], &params.offset_weights)
        .map_err(|err| backend_error("upload DCNv4 weights", err))?;
    let biases = backend
        .upload_f32(&[params.offset_biases.len()], &params.offset_biases)
        .map_err(|err| backend_error("upload DCNv4 offset biases", err))?;
    let (raw, _) = backend
        .dcnv4_forward_f32(input, &weights, &biases, desc, cfg)
        .map_err(|err| backend_error("DCNv4 native raw forward", err))?;
    let pre = if let Some(bn) = &params.batch_norm {
        let normalized = normalized_raw(backend, &raw, bn, desc.geometry.spatial())?;
        let gamma = backend
            .upload_f32(&[bn.scales.len()], &bn.scales)
            .map_err(|err| backend_error("upload DCNv4 gamma", err))?;
        backend
            .channel_scale_f32(
                &normalized,
                &gamma,
                spec.filters,
                desc.geometry.spatial(),
                cfg,
            )
            .map_err(|err| backend_error("DCNv4 rolling BN scale", err))?
    } else {
        let bias = backend
            .upload_f32(&[params.biases.len()], &params.biases)
            .map_err(|err| backend_error("upload DCNv4 output bias", err))?;
        backend
            .bias_add_f32(&raw, &bias, spec.filters, desc.geometry.spatial(), cfg)
            .map_err(|err| backend_error("DCNv4 bias add", err))?
    };
    let out = activation_output(backend, &pre, layer, spec)?;
    Ok((out, pre))
}

pub(super) fn activation_output<B: F32TensorOpsBackend>(
    backend: &B, pre: &B::Tensor, layer: &Layer, spec: &Dcnv4Spec,
) -> Result<B::Tensor> where B::Error: std::fmt::Display {
    let cfg=ElementwiseLaunchConfig::default();
    if let Some(norm)=spec.activation.to_channel_normalization() {
        let s=layer.output_shape;
        backend.normalize_channels_f32(pre,&tensor_backend_shape(s),s.batch,s.channels,
            s.height*s.width,norm,cfg)
            .map_err(|err| backend_error("DCNv4 channel normalization",err))
    } else {
        backend.activation_f32(pre,gpu_activation_kind(layer.index,&spec.activation)?,cfg)
            .map_err(|err| backend_error("DCNv4 activation",err))
    }
}

/// Explicit mode supports testing without mutating process-wide environment.
/// Channel output recomputes on device from this forward's preactivation.
pub(super) fn activation_delta<B: F32TensorOpsBackend>(
    backend: &B, pre: &B::Tensor, seed: &B::Tensor, layer: &Layer, spec: &Dcnv4Spec,
    mode: GradMode,
) -> Result<B::Tensor> where B::Error: std::fmt::Display {
    let cfg=ElementwiseLaunchConfig::default();
    if let Some(norm)=spec.activation.to_channel_normalization() {
        let s=layer.output_shape;
        let output=activation_output(backend,pre,layer,spec)?;
        backend.normalize_channels_backward_f32(pre,&output,seed,&tensor_backend_shape(s),
            s.batch,s.channels,s.height*s.width,norm,mode==GradMode::Cpp,cfg)
            .map_err(|err| backend_error("DCNv4 channel-normalization adjoint",err))
    } else {
        backend.activation_backward_f32(pre,seed,gpu_activation_kind(layer.index,&spec.activation)?,cfg)
            .map_err(|err| backend_error("DCNv4 activation backward",err))
    }
}

pub(super) struct Gradients<T> {
    pub input: T,
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
    pub scales: Option<Vec<f32>>,
}

pub(super) fn backward<B: F32TensorOpsBackend>(
    backend: &B,
    input: &B::Tensor,
    pre: &B::Tensor,
    seed: &B::Tensor,
    layer: &Layer,
    spec: &Dcnv4Spec,
    params: &Dcnv4Params,
) -> Result<Gradients<B::Tensor>>
where
    B::Error: std::fmt::Display,
{
    if needs_host(spec) {
        return Err(Error::InvalidConfig(
            "DCNv4 device backward selected for a host-only variant".to_owned(),
        ));
    }
    let cfg = ElementwiseLaunchConfig::default();
    let desc = descriptor(layer, spec, params)?;
    let weights = backend
        .upload_f32(&[params.offset_weights.len()], &params.offset_weights)
        .map_err(|err| backend_error("upload DCNv4 backward weights", err))?;
    let biases = backend
        .upload_f32(&[params.offset_biases.len()], &params.offset_biases)
        .map_err(|err| backend_error("upload DCNv4 backward offset biases", err))?;
    // Recompute on device from this forward state's input. No host-cache hop,
    // but redundant projection/aggregation work; no residency/speedup claim.
    let (raw, offsets) = backend
        .dcnv4_forward_f32(input, &weights, &biases, desc, cfg)
        .map_err(|err| backend_error("DCNv4 native offset recomputation", err))?;
    let delta = activation_delta(backend, pre, seed, layer, spec, GradMode::from_env())?;
    let (aggregation_grad, biases, scales) = if let Some(bn) = &params.batch_norm {
        let normalized = normalized_raw(backend, &raw, bn, desc.geometry.spatial())?;
        let product = backend
            .vector_mul_f32(
                &delta,
                &normalized,
                &tensor_backend_shape(layer.output_shape),
                cfg,
            )
            .map_err(|err| backend_error("DCNv4 gamma product", err))?;
        let dscale = backend
            .bias_backward_f32(&product, spec.filters, desc.geometry.spatial(), cfg)
            .map_err(|err| backend_error("DCNv4 gamma reduction", err))?;
        let gains: Vec<f32> = bn
            .scales
            .iter()
            .zip(&bn.rolling_variance)
            .map(|(s, v)| s * (1.0 / (v + BATCH_NORM_EPSILON).sqrt()))
            .collect();
        let gains = backend
            .upload_f32(&[gains.len()], &gains)
            .map_err(|err| backend_error("upload DCNv4 BN gains", err))?;
        let d_raw = backend
            .channel_scale_f32(&delta, &gains, spec.filters, desc.geometry.spatial(), cfg)
            .map_err(|err| backend_error("DCNv4 rolling BN adjoint", err))?;
        let ds = backend
            .download_f32(&dscale)
            .map_err(|err| backend_error("download DCNv4 gamma gradient", err))?;
        (d_raw, vec![0.0; params.biases.len()], Some(ds))
    } else {
        let bias_grad = backend
            .bias_backward_f32(&delta, spec.filters, desc.geometry.spatial(), cfg)
            .map_err(|err| backend_error("DCNv4 bias backward", err))?;
        let db = backend
            .download_f32(&bias_grad)
            .map_err(|err| backend_error("download DCNv4 bias gradient", err))?;
        (delta, db, None)
    };
    let (input, dw, db) = backend
        .dcnv4_backward_f32(input, &weights, &offsets, &aggregation_grad, desc, cfg)
        .map_err(|err| backend_error("DCNv4 native backward", err))?;
    Ok(Gradients {
        input,
        biases,
        scales,
        weights: backend
            .download_f32(&dw)
            .map_err(|err| backend_error("download DCNv4 weight gradient", err))?,
        offset_biases: backend
            .download_f32(&db)
            .map_err(|err| backend_error("download DCNv4 offset bias gradient", err))?,
    })
}

pub(super) fn accumulate<T>(train: &mut Dcnv4TrainBuffers, grad: &Gradients<T>) -> Result<()> {
    // Check every block before changing any buffer. Dummy weights never receive
    // a derivative or update; optimizer learning-rate scaling remains unchanged.
    if train.bias_updates.len() != grad.biases.len()
        || train.offset_weight_updates.len() != grad.weights.len()
        || train.offset_bias_updates.len() != grad.offset_biases.len()
        || match (&train.batch_norm, &grad.scales) {
            (Some(bn), Some(ds)) => bn.scale_updates.len() != ds.len(),
            (None, None) => false,
            _ => true,
        }
    {
        return Err(Error::InvalidConfig(
            "DCNv4 gradient storage mismatch".to_owned(),
        ));
    }
    if let (Some(bn), Some(ds)) = (&mut train.batch_norm, &grad.scales) {
        for (dst, src) in bn.scale_updates.iter_mut().zip(ds) {
            *dst += src;
        }
    }
    for (dst, src) in train.bias_updates.iter_mut().zip(&grad.biases) {
        *dst += src;
    }
    for (dst, src) in train.offset_weight_updates.iter_mut().zip(&grad.weights) {
        *dst += src;
    }
    for (dst, src) in train
        .offset_bias_updates
        .iter_mut()
        .zip(&grad.offset_biases)
    {
        *dst += src;
    }
    Ok(())
}

fn normalized_raw<B: F32TensorOpsBackend>(
    backend: &B,
    raw: &B::Tensor,
    bn: &BatchNormParams,
    spatial: usize,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let cfg = ElementwiseLaunchConfig::default();
    let minus_mean: Vec<f32> = bn.rolling_mean.iter().map(|v| -v).collect();
    let inv: Vec<f32> = bn
        .rolling_variance
        .iter()
        .map(|v| 1.0 / (v + BATCH_NORM_EPSILON).sqrt())
        .collect();
    let mean = backend
        .upload_f32(&[minus_mean.len()], &minus_mean)
        .map_err(|err| backend_error("upload DCNv4 negative mean", err))?;
    let inv = backend
        .upload_f32(&[inv.len()], &inv)
        .map_err(|err| backend_error("upload DCNv4 inv std", err))?;
    let centered = backend
        .bias_add_f32(raw, &mean, bn.scales.len(), spatial, cfg)
        .map_err(|err| backend_error("DCNv4 BN centering", err))?;
    backend
        .channel_scale_f32(&centered, &inv, bn.scales.len(), spatial, cfg)
        .map_err(|err| backend_error("DCNv4 BN normalization", err))
}

impl<B: F32TensorOpsBackend> GpuExecutableNetwork<B>
where
    B::Error: std::fmt::Display,
{
    /// Match CPU DCNv4's post-backward 0.9 EMA, including unseeded layers.
    /// On Oxide raw statistics reduce on device; only channel EMA values return
    /// to the canonical host parameters. No batch-stat normalization is used
    /// for the actual layer output or its derivatives.
    pub(super) fn update_dcnv4_rolling_stats_gpu(
        &mut self,
        forward: &GpuForwardState<B::Tensor>,
    ) -> Result<()> {
        let mut updates = Vec::new();
        let cfg = ElementwiseLaunchConfig::default();
        for layer in &self.network.layers {
            let LayerKind::Dcnv4(spec) = &layer.kind else {
                continue;
            };
            if !spec.batch_normalize {
                continue;
            }
            let LayerParams::Dcnv4(params) = &self.params[layer.index] else {
                return Err(Error::InvalidConfig(
                    "missing DCNv4 BN parameters".to_owned(),
                ));
            };
            let desc = descriptor(layer, spec, params)?;
            let bn = params.batch_norm.as_ref().expect("validated BN option");
            let input = if layer.index == 0 {
                &forward.input_gpu
            } else {
                &forward.gpu_layer_outputs[layer.index - 1]
            };
            let w = self
                .backend
                .upload_f32(&[params.offset_weights.len()], &params.offset_weights)
                .map_err(|err| backend_error("upload DCNv4 stats weights", err))?;
            let b = self
                .backend
                .upload_f32(&[params.offset_biases.len()], &params.offset_biases)
                .map_err(|err| backend_error("upload DCNv4 stats biases", err))?;
            let (raw, _) = self
                .backend
                .dcnv4_forward_f32(input, &w, &b, desc, cfg)
                .map_err(|err| backend_error("DCNv4 raw stats recompute", err))?;
            let mean = self
                .backend
                .upload_f32(&[spec.filters], &bn.rolling_mean)
                .map_err(|err| backend_error("upload DCNv4 rolling mean", err))?;
            let variance = self
                .backend
                .upload_f32(&[spec.filters], &bn.rolling_variance)
                .map_err(|err| backend_error("upload DCNv4 rolling variance", err))?;
            let zero = self
                .backend
                .zeros_f32(&[spec.filters])
                .map_err(|err| backend_error("DCNv4 stats scratch", err))?;
            // Reuse only the stats/EMA reduction; discard this primitive's
            // batch-normalized output and its other backward caches.
            let stats = self
                .backend
                .batchnorm_train_forward_f32(
                    &raw,
                    &zero,
                    &zero,
                    &mean,
                    &variance,
                    &tensor_backend_shape(layer.output_shape),
                    BATCH_NORM_ROLLING_KEEP_CPU,
                    cfg,
                )
                .map_err(|err| backend_error("DCNv4 native stats EMA", err))?;
            let mean = self
                .backend
                .download_f32(&stats.rolling_mean)
                .map_err(|err| backend_error("download DCNv4 rolling mean", err))?;
            let variance = self
                .backend
                .download_f32(&stats.rolling_variance)
                .map_err(|err| backend_error("download DCNv4 rolling variance", err))?;
            updates.push((layer.index, mean, variance));
        }
        for (index, mean, variance) in updates {
            let LayerParams::Dcnv4(params) = &mut self.params[index] else {
                unreachable!()
            };
            let bn = params.batch_norm.as_mut().expect("validated BN option");
            bn.rolling_mean = mean;
            bn.rolling_variance = variance;
        }
        Ok(())
    }
}

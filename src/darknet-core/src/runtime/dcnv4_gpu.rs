//! Bounded DCNv4 GPU math: <=625 active points, elementwise/channel activation.
//! BN normalizes with rolling stats; its 0.9 EMA runs only after backward.
//! Parameter gradients and SGD/Adam math use device tensors; DCNv4 Adam moments
//! persist on-device between updates and bridge through the host sidecar only
//! on save/load. Canonical parameters remain host-side. Each GPU forward state
//! owns raw, offset, and projection-weight tensors for backward, so backward
//! uses the exact forward weights without another parameter upload.

use super::*;
use crate::model::Layer;
use darknet_cuda::host_ops::dcnv4::{Dcnv4Composition, Dcnv4Geometry};

pub(super) fn needs_host(spec: &Dcnv4Spec) -> bool {
    spec.size
        .checked_mul(spec.size)
        .and_then(|v| v.checked_sub(usize::from(spec.remove_center)))
        .is_none_or(|points| points > 625)
        || (spec.activation.to_cuda_f32_kind().is_none()
            && spec.activation.to_channel_normalization().is_none())
}

pub(super) fn descriptor(
    layer: &Layer,
    spec: &Dcnv4Spec,
    params: &Dcnv4Params,
) -> Result<Dcnv4Composition> {
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

#[derive(Clone, Debug)]
struct ParameterSnapshot {
    offset_weights: Vec<f32>,
    offset_biases: Vec<f32>,
    biases: Vec<f32>,
    scales: Option<Vec<f32>>,
}

impl ParameterSnapshot {
    fn matches(&self, other: &Self) -> bool {
        fn same_values(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        }
        same_values(&self.offset_weights, &other.offset_weights)
            && same_values(&self.offset_biases, &other.offset_biases)
            && same_values(&self.biases, &other.biases)
            && match (&self.scales, &other.scales) {
                (Some(a), Some(b)) => same_values(a, b),
                (None, None) => true,
                _ => false,
            }
    }
}

#[derive(Clone, Debug)]
pub(super) struct DeviceParameters<T> {
    snapshot: ParameterSnapshot,
    pub(super) offset_weights: T,
    pub(super) offset_biases: T,
    pub(super) biases: T,
    pub(super) scales: Option<T>,
}

pub(super) fn device_parameters<B: F32TensorOpsBackend>(
    backend: &B,
    cached: &mut Option<DeviceParameters<B::Tensor>>,
    layer: &Layer,
    spec: &Dcnv4Spec,
    params: &Dcnv4Params,
) -> Result<DeviceParameters<B::Tensor>>
where
    B::Error: std::fmt::Display,
{
    let desc = descriptor(layer, spec, params)?;
    let snapshot = ParameterSnapshot {
        offset_weights: params.offset_weights.clone(),
        offset_biases: params.offset_biases.clone(),
        biases: params.biases.clone(),
        scales: params.batch_norm.as_ref().map(|bn| bn.scales.clone()),
    };
    if let Some(current) = cached
        .as_ref()
        .filter(|current| current.snapshot.matches(&snapshot))
    {
        return Ok(current.clone());
    }
    let shape = [spec.filters];
    let value = DeviceParameters {
        snapshot,
        offset_weights: backend
            .upload_f32(
                &[
                    desc.geometry.offset_dim(),
                    desc.geometry.channels,
                    spec.size,
                    spec.size,
                ],
                &params.offset_weights,
            )
            .map_err(|err| backend_error("upload DCNv4 weights", err))?,
        offset_biases: backend
            .upload_f32(&[desc.geometry.offset_dim()], &params.offset_biases)
            .map_err(|err| backend_error("upload DCNv4 offset biases", err))?,
        biases: backend
            .upload_f32(&shape, &params.biases)
            .map_err(|err| backend_error("upload DCNv4 output biases", err))?,
        scales: params
            .batch_norm
            .as_ref()
            .map(|bn| {
                backend
                    .upload_f32(&shape, &bn.scales)
                    .map_err(|err| backend_error("upload DCNv4 gamma", err))
            })
            .transpose()?,
    };
    *cached = Some(value.clone());
    Ok(value)
}

pub(super) fn store_device_parameters<T>(
    cached: &mut Option<DeviceParameters<T>>,
    params: &Dcnv4Params,
    offset_weights: T,
    offset_biases: T,
    biases: T,
    scales: Option<T>,
) {
    *cached = Some(DeviceParameters {
        snapshot: ParameterSnapshot {
            offset_weights: params.offset_weights.clone(),
            offset_biases: params.offset_biases.clone(),
            biases: params.biases.clone(),
            scales: params.batch_norm.as_ref().map(|bn| bn.scales.clone()),
        },
        offset_weights,
        offset_biases,
        biases,
        scales,
    });
}

#[derive(Clone, Debug)]
pub(crate) struct ForwardCache<T> {
    raw: T,
    offsets: T,
    pub(super) weights: T,
}

pub(super) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    input: &B::Tensor,
    layer: &Layer,
    spec: &Dcnv4Spec,
    params: &Dcnv4Params,
    device_params: &DeviceParameters<B::Tensor>,
) -> Result<(B::Tensor, B::Tensor, ForwardCache<B::Tensor>)>
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
    let (raw, offsets) = backend
        .dcnv4_forward_f32(
            input,
            &device_params.offset_weights,
            &device_params.offset_biases,
            desc,
            cfg,
        )
        .map_err(|err| backend_error("DCNv4 native raw forward", err))?;
    let pre = if let Some(bn) = &params.batch_norm {
        let normalized = normalized_raw(backend, &raw, bn, desc.geometry.spatial())?;
        let gamma = device_params.scales.as_ref().ok_or_else(|| {
            Error::InvalidConfig("DCNv4 BN gamma is missing from device parameters".to_owned())
        })?;
        backend
            .channel_scale_f32(
                &normalized,
                gamma,
                spec.filters,
                desc.geometry.spatial(),
                cfg,
            )
            .map_err(|err| backend_error("DCNv4 rolling BN scale", err))?
    } else {
        backend
            .bias_add_f32(
                &raw,
                &device_params.biases,
                spec.filters,
                desc.geometry.spatial(),
                cfg,
            )
            .map_err(|err| backend_error("DCNv4 bias add", err))?
    };
    let out = activation_output(backend, &pre, layer, spec)?;
    Ok((
        out,
        pre,
        ForwardCache {
            raw,
            offsets,
            weights: device_params.offset_weights.clone(),
        },
    ))
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
    pub biases: T,
    pub weights: T,
    pub offset_biases: T,
    pub scales: Option<T>,
}

pub(super) fn backward<B: F32TensorOpsBackend>(
    backend: &B,
    input: &B::Tensor,
    pre: &B::Tensor,
    seed: &B::Tensor,
    cache: &ForwardCache<B::Tensor>,
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
    let weights = &cache.weights;
    let raw = &cache.raw;
    let offsets = &cache.offsets;
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
        let dbias = backend
            .zeros_f32(&[params.biases.len()])
            .map_err(|err| backend_error("allocate zero DCNv4 bias gradient", err))?;
        (d_raw, dbias, Some(dscale))
    } else {
        let bias_grad = backend
            .bias_backward_f32(&delta, spec.filters, desc.geometry.spatial(), cfg)
            .map_err(|err| backend_error("DCNv4 bias backward", err))?;
        (delta, bias_grad, None)
    };
    let (input, dw, db) = backend
        .dcnv4_backward_f32(input, weights, offsets, &aggregation_grad, desc, cfg)
        .map_err(|err| backend_error("DCNv4 native backward", err))?;
    Ok(Gradients {
        input,
        biases,
        scales,
        weights: dw,
        offset_biases: db,
    })
}

pub(super) fn accumulate<B: F32TensorOpsBackend>(
    backend: &B,
    train: &mut GpuDcnv4TrainBuffers<B::Tensor>,
    grad: &Gradients<B::Tensor>,
) -> Result<()>
where
    B::Error: std::fmt::Display,
{
    if train.scale_updates.is_some() != grad.scales.is_some() {
        return Err(Error::InvalidConfig(
            "DCNv4 gradient storage mismatch".to_owned(),
        ));
    }
    let cfg = ElementwiseLaunchConfig::default();
    let add = |name: &'static str, alpha: f32, x: &B::Tensor, y: &B::Tensor| {
        backend
            .saxpy_f32(alpha, x, y, cfg)
            .map_err(|err| backend_error(name, err))
    };
    if !train.dirty {
        // The backend may retain multidimensional gradient layouts. Seed the
        // accumulator with an exact-layout device copy instead of the flat
        // zero allocation used for clean buffers.
        let weights = add("DCNv4 offset-weight gradient initialization", 0.0, &grad.weights, &grad.weights)?;
        let offset_biases = add("DCNv4 offset-bias gradient initialization", 0.0, &grad.offset_biases, &grad.offset_biases)?;
        let biases = add("DCNv4 bias gradient initialization", 0.0, &grad.biases, &grad.biases)?;
        let scales = grad
            .scales
            .as_ref()
            .map(|scales| add("DCNv4 scale gradient initialization", 0.0, scales, scales))
            .transpose()?;
        train.offset_weight_updates = weights;
        train.offset_bias_updates = offset_biases;
        train.bias_updates = biases;
        train.scale_updates = scales;
        train.dirty = true;
        return Ok(());
    }
    // Stage results first so a backend failure cannot partially mutate the
    // accumulated gradients. The dummy parameter block remains excluded.
    let weights = add("DCNv4 offset-weight gradient accumulation", 1.0, &grad.weights, &train.offset_weight_updates)?;
    let offset_biases = add("DCNv4 offset-bias gradient accumulation", 1.0, &grad.offset_biases, &train.offset_bias_updates)?;
    let biases = add("DCNv4 bias gradient accumulation", 1.0, &grad.biases, &train.bias_updates)?;
    let scales = match (&grad.scales, &train.scale_updates) {
        (Some(gradient), Some(updates)) => Some(add("DCNv4 scale gradient accumulation", 1.0, gradient, updates)?),
        (None, None) => None,
        _ => unreachable!("validated DCNv4 scale-gradient metadata"),
    };
    train.offset_weight_updates = weights;
    train.offset_bias_updates = offset_biases;
    train.bias_updates = biases;
    train.scale_updates = scales;
    train.dirty = true;
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
            let device_params = {
                let mut caches = self.dcnv4_device_params.borrow_mut();
                device_parameters(
                    &self.backend,
                    &mut caches[layer.index],
                    layer,
                    spec,
                    params,
                )?
            };
            let (raw, _) = self
                .backend
                .dcnv4_forward_f32(
                    input,
                    &device_params.offset_weights,
                    &device_params.offset_biases,
                    desc,
                    cfg,
                )
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

//! Bounded DCNv4 GPU math: no BN, <=25 active points, elementwise activation.
//! Parameter gradients/SGD/Adam stay host-resident; forward/backward tensor math
//! is native on Oxide. Backward recomputes offsets, never uses last-forward state.

use super::*;
use crate::model::Layer;
use darknet_cuda::host_ops::dcnv4::{Dcnv4Composition, Dcnv4Geometry};

pub(super) fn needs_host(spec: &Dcnv4Spec) -> bool {
    spec.batch_normalize
        || spec
            .size
            .checked_mul(spec.size)
            .and_then(|v| v.checked_sub(usize::from(spec.remove_center)))
            .is_none_or(|points| points > 25)
        || spec.activation.to_cuda_f32_kind().is_none()
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
    if needs_host(spec) || params.batch_norm.is_some() {
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
    let bias = backend
        .upload_f32(&[params.biases.len()], &params.biases)
        .map_err(|err| backend_error("upload DCNv4 output bias", err))?;
    let pre = backend
        .bias_add_f32(&raw, &bias, spec.filters, desc.geometry.spatial(), cfg)
        .map_err(|err| backend_error("DCNv4 bias add", err))?;
    let out = backend
        .activation_f32(
            &pre,
            gpu_activation_kind(layer.index, &spec.activation)?,
            cfg,
        )
        .map_err(|err| backend_error("DCNv4 activation", err))?;
    Ok((out, pre))
}

pub(super) struct Gradients<T> {
    pub input: T,
    pub biases: Vec<f32>,
    pub weights: Vec<f32>,
    pub offset_biases: Vec<f32>,
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
    if needs_host(spec) || params.batch_norm.is_some() {
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
    let (_, offsets) = backend
        .dcnv4_forward_f32(input, &weights, &biases, desc, cfg)
        .map_err(|err| backend_error("DCNv4 native offset recomputation", err))?;
    let delta = backend
        .activation_backward_f32(
            pre,
            seed,
            gpu_activation_kind(layer.index, &spec.activation)?,
            cfg,
        )
        .map_err(|err| backend_error("DCNv4 activation backward", err))?;
    let bias_grad = backend
        .bias_backward_f32(&delta, spec.filters, desc.geometry.spatial(), cfg)
        .map_err(|err| backend_error("DCNv4 bias backward", err))?;
    let (input, dw, db) = backend
        .dcnv4_backward_f32(input, &weights, &offsets, &delta, desc, cfg)
        .map_err(|err| backend_error("DCNv4 native backward", err))?;
    Ok(Gradients {
        input,
        biases: backend
            .download_f32(&bias_grad)
            .map_err(|err| backend_error("download DCNv4 bias gradient", err))?,
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
    {
        return Err(Error::InvalidConfig(
            "DCNv4 gradient storage mismatch".to_owned(),
        ));
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

//! Native f32 forward composition only. Backward/update semantics stay unchanged.
use crate::runtime::{backend_error, tensor_backend_shape};
use crate::error::{Error, Result};
use crate::laya_act_head::LayaActHeadWeights;
use crate::model::{LayaActHeadSpec, TensorShape};
use lightnet_cuda::{
    ElementwiseLaunchConfig, F16TensorOpsBackend, F32ActivationKind, F32TensorOpsBackend,
};

fn count(name: &str, a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| {
            Error::InvalidConfig(format!(
                "[laya_act_head] act-head parameter {name}: zero/overflow dimensions"
            ))
        })
}

pub(crate) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &LayaActHeadSpec,
    weights: &LayaActHeadWeights,
    sources: [&B::Tensor; 3],
    shapes: [TensorShape; 3],
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let hs = tensor_backend_shape(shapes[0]);
    let ss = tensor_backend_shape(shapes[1]);
    let ms = tensor_backend_shape(shapes[2]);
    let d = lightnet_cuda::host_ops::laya_act::Layout::new(&hs, &ss, &ms)
        .map_err(Error::InvalidConfig)?;
    if d.hidden != spec.hidden || d.options != spec.max_options {
        return Err(Error::InvalidConfig(
            "[laya_act_head] act-head spec/source dimensions disagree".into(),
        ));
    }
    let inputs = spec
        .hidden
        .checked_add(4)
        .ok_or_else(|| Error::InvalidConfig("act-head input overflow".into()))?;
    let w1 = count("fc1_w", spec.fc_dim, inputs)?;
    let w2 = count("fc2_w", spec.n_act, spec.fc_dim)?;
    count("hidden output", d.batch, spec.fc_dim)?;
    count("final output", d.batch, spec.n_act)?;
    for (name, actual, expected) in [
        ("fc1_w", weights.fc1_w.len(), w1),
        ("fc1_b", weights.fc1_b.len(), spec.fc_dim),
        ("fc2_w", weights.fc2_w.len(), w2),
        ("fc2_b", weights.fc2_b.len(), spec.n_act),
    ] {
        if actual != expected {
            return Err(Error::InvalidConfig(format!(
                "[laya_act_head] act-head parameter {name} has {actual} values, expected {expected}"
            )));
        }
    }
    // Deliberately uncached: reloads and public host-param edits are immediately
    // visible. No residency/performance claim; no stale device weight reuse.
    let w1 = backend
        .upload_f32(&[spec.fc_dim, inputs], &weights.fc1_w)
        .map_err(|e| backend_error("upload act fc1 weights", e))?;
    let b1 = backend
        .upload_f32(&[spec.fc_dim], &weights.fc1_b)
        .map_err(|e| backend_error("upload act fc1 bias", e))?;
    let w2 = backend
        .upload_f32(&[spec.n_act, spec.fc_dim], &weights.fc2_w)
        .map_err(|e| backend_error("upload act fc2 weights", e))?;
    let b2 = backend
        .upload_f32(&[spec.n_act], &weights.fc2_b)
        .map_err(|e| backend_error("upload act fc2 bias", e))?;
    let features = backend
        .laya_act_features_f32(sources[0], sources[1], sources[2], &hs, &ss, &ms, config)
        .map_err(|e| backend_error("GPU act features", e))?;
    let first = backend
        .connected_f32(&features, &w1, &b1, d.batch, inputs, spec.fc_dim, config)
        .map_err(|e| backend_error("GPU act fc1", e))?;
    let activated = backend
        .gelu_erf_f32(&first, &[d.batch, spec.fc_dim], config)
        .map_err(|e| backend_error("GPU act exact GELU", e))?;
    backend
        .connected_f32(
            &activated,
            &w2,
            &b2,
            d.batch,
            spec.fc_dim,
            spec.n_act,
            config,
        )
        .map_err(|e| backend_error("GPU act fc2", e))
}

pub(crate) fn forward_f16<B: F16TensorOpsBackend>(
    backend: &B,
    spec: &LayaActHeadSpec,
    weights: &LayaActHeadWeights,
    sources: [&B::TensorF16; 3],
    shapes: [TensorShape; 3],
    config: ElementwiseLaunchConfig,
) -> Result<B::TensorF16>
where
    B::Error: std::fmt::Display,
{
    let source_shapes = shapes.map(tensor_backend_shape);
    let d = lightnet_cuda::host_ops::laya_act::Layout::new(
        &source_shapes[0],
        &source_shapes[1],
        &source_shapes[2],
    )
    .map_err(Error::InvalidConfig)?;
    if d.hidden != spec.hidden || d.options != spec.max_options {
        return Err(Error::InvalidConfig(
            "[laya_act_head] act-head spec/source dimensions disagree".into(),
        ));
    }
    let inputs = spec
        .hidden
        .checked_add(4)
        .ok_or_else(|| Error::InvalidConfig("act-head input overflow".into()))?;
    let w1_len = count("fc1_w", spec.fc_dim, inputs)?;
    let w2_len = count("fc2_w", spec.n_act, spec.fc_dim)?;
    for (name, actual, expected) in [
        ("fc1_w", weights.fc1_w.len(), w1_len),
        ("fc1_b", weights.fc1_b.len(), spec.fc_dim),
        ("fc2_w", weights.fc2_w.len(), w2_len),
        ("fc2_b", weights.fc2_b.len(), spec.n_act),
    ] {
        if actual != expected {
            return Err(Error::InvalidConfig(format!(
                "[laya_act_head] act-head parameter {name} has {actual} values, expected {expected}"
            )));
        }
    }
    let features = backend
        .laya_act_features_f16(
            sources[0],
            sources[1],
            sources[2],
            &source_shapes[0],
            &source_shapes[1],
            &source_shapes[2],
            config,
        )
        .map_err(|e| backend_error("GPU f16 act features", e))?;
    let first_shape = [d.batch, inputs, 1, 1];
    let hidden_shape = [d.batch, spec.fc_dim, 1, 1];
    let logits_shape = [d.batch, spec.n_act, 1, 1];
    let first = backend
        .connected_f16(
            &features,
            &weights.fc1_w,
            &weights.fc1_b,
            &first_shape,
            &hidden_shape,
            spec.fc_dim,
            F32ActivationKind::Linear,
            config,
        )
        .map_err(|e| backend_error("GPU f16 act fc1", e))?;
    let activated = backend
        .gelu_erf_f16(&first, &hidden_shape, config)
        .map_err(|e| backend_error("GPU f16 act exact GELU", e))?;
    backend
        .connected_f16(
            &activated,
            &weights.fc2_w,
            &weights.fc2_b,
            &hidden_shape,
            &logits_shape,
            spec.n_act,
            F32ActivationKind::Linear,
            config,
        )
        .map_err(|e| backend_error("GPU f16 act fc2", e))
}

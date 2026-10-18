//! Standalone act-head gradient composition. No runtime state or updates.
use crate::host_ops::laya_act::MlpLayout;
use crate::{ElementwiseLaunchConfig, F32TensorOpsBackend};

/// Resident operands for the exact gradient of the act-head MLP.
/// Fc2 bias values are unnecessary for its Jacobian; d(fc2_bias) is still
/// returned. Seeds are [B,A] or the equivalent [B,A,1,1] view. Source
/// storage is interpreted using MlpLayout's declared NCHW views, as in the
/// feature primitive; parameters use canonical matrix/vector metadata.
pub struct LayaActBackwardInputs<'a, T> {
    pub head: &'a T,
    pub scorer: &'a T,
    pub markers: &'a T,
    pub fc1_w: &'a T,
    pub fc1_b: &'a T,
    pub fc2_w: &'a T,
    pub seeds: &'a T,
}

/// Positive, batch-summed derivatives; not optimizer descent directions.
/// No scorer/marker gradients: the four summary features are detached.
#[derive(Clone, Debug)]
pub struct LayaActHeadGrad<T> {
    pub fc1_w: T,
    pub fc1_b: T,
    pub fc2_w: T,
    pub fc2_b: T,
    /// Only CLS adjoints, [B,H]; not the full sequence scatter/routing.
    pub dcls: T,
}

/// Implementation shared by the trait default and checked Oxide override.
/// Oxide's entry point validates all operands before delegating; generic
/// backends enforce their primitive preconditions. This helper
/// composes backend operations only: no tensor downloads, optimizer mutation,
/// implicit batch averaging or persistent activation cache.
#[doc(hidden)]
pub fn backward_composed<B: F32TensorOpsBackend + ?Sized>(
    backend: &B,
    args: &LayaActBackwardInputs<'_, B::Tensor>,
    layout: &MlpLayout,
    config: ElementwiseLaunchConfig,
) -> Result<LayaActHeadGrad<B::Tensor>, B::Error> {
    let d = layout.features();
    let f = layout.fc_dim();
    let a = layout.n_act();
    let i = layout.inputs();
    let x = backend.laya_act_features_f32(
        args.head,
        args.scorer,
        args.markers,
        &layout.head_shape(),
        &layout.scorer_shape(),
        &layout.marker_shape(),
        config,
    )?;
    let pre = backend.connected_f32(&x, args.fc1_w, args.fc1_b, d.batch, i, f, config)?;
    let hidden = backend.gelu_erf_f32(&pre, &[d.batch, f], config)?;
    let (dh, fc2_w, fc2_b) =
        backend.connected_backward_f32(&hidden, args.fc2_w, args.seeds, d.batch, f, a, config)?;
    let du = backend.gelu_erf_backward_f32(&pre, &dh, &[d.batch, f], config)?;
    let (dx, fc1_w, fc1_b) =
        backend.connected_backward_f32(&x, args.fc1_w, &du, d.batch, i, f, config)?;
    // Matrix dx is [B,H+4]. Copy only its CLS channels; summary derivatives
    // must not escape into scorer or marker inputs.
    let dcls = backend.route_concat_f32(
        &[&dx],
        &[0],
        &[d.hidden],
        &[i],
        &[d.batch, d.hidden],
        1,
        config,
    )?;
    Ok(LayaActHeadGrad {
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        dcls,
    })
}

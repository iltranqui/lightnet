//! Native f32 forward composition for frozen `[laya_scorer]` layers.
use crate::runtime::{backend_error, tensor_backend_shape};
use crate::error::{Error, Result};
use crate::laya_scorer::LayaScorerWeights;
use crate::model::{LayaScorerSpec, TensorShape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

fn count(name: &str, a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| {
            Error::InvalidConfig(format!("[laya_scorer] {name}: zero/overflow dimensions"))
        })
}

pub(crate) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &LayaScorerSpec,
    weights: &LayaScorerWeights,
    input: &B::Tensor,
    input_shape: TensorShape,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let batch = input_shape.batch;
    let hidden = spec.hidden;
    let options = spec.max_options;
    if batch == 0
        || hidden == 0
        || options == 0
        || input_shape.channels != hidden
        || input_shape.height != 1
        || input_shape.width != options
        || output_shape.batch != batch
        || output_shape.channels != 1
        || output_shape.height != 1
        || output_shape.width != options
    {
        return Err(Error::InvalidConfig(format!(
            "[laya_scorer] source/output dimensions disagree with hidden={hidden}, options={options}: {input_shape} -> {output_shape}"
        )));
    }
    let rows = count("rows", batch, options)?;
    let input_len = count("input", rows, hidden)?;
    let fc1_len = count("fc1 weights", hidden, hidden)?;
    for (name, actual, expected) in [
        ("ln_w", weights.ln_w.len(), hidden),
        ("ln_b", weights.ln_b.len(), hidden),
        ("fc1_w", weights.fc1_w.len(), fc1_len),
        ("fc1_b", weights.fc1_b.len(), hidden),
        ("fc2_w", weights.fc2_w.len(), hidden),
    ] {
        if actual != expected {
            return Err(Error::InvalidConfig(format!(
                "[laya_scorer] weight {name} has {actual} values, expected {expected}"
            )));
        }
    }
    if !weights.fc2_b.is_finite()
        || weights
            .ln_w
            .iter()
            .chain(&weights.ln_b)
            .chain(&weights.fc1_w)
            .chain(&weights.fc1_b)
            .chain(&weights.fc2_w)
            .any(|v| !v.is_finite())
    {
        return Err(Error::InvalidConfig(
            "[laya_scorer] frozen weights contain non-finite values".into(),
        ));
    }

    // Upload every call so edits to public host-side frozen weights are visible.
    let ln_w = backend
        .upload_f32(&[hidden], &weights.ln_w)
        .map_err(|e| backend_error("upload LayaScorer LayerNorm weight", e))?;
    let ln_b = backend
        .upload_f32(&[hidden], &weights.ln_b)
        .map_err(|e| backend_error("upload LayaScorer LayerNorm bias", e))?;
    let fc1_w = backend
        .upload_f32(&[hidden, hidden], &weights.fc1_w)
        .map_err(|e| backend_error("upload LayaScorer FC1 weights", e))?;
    let fc1_b = backend
        .upload_f32(&[hidden], &weights.fc1_b)
        .map_err(|e| backend_error("upload LayaScorer FC1 bias", e))?;
    let fc2_w = backend
        .upload_f32(&[1, hidden], &weights.fc2_w)
        .map_err(|e| backend_error("upload LayaScorer FC2 weights", e))?;
    let fc2_b = backend
        .upload_f32(&[1], &[weights.fc2_b])
        .map_err(|e| backend_error("upload LayaScorer FC2 bias", e))?;

    let rows_tensor = backend
        .laya_scorer_unpack_f32(
            input,
            &tensor_backend_shape(input_shape),
            hidden,
            options,
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer unpack", e))?;
    debug_assert_eq!(input_len, rows * hidden);
    let normalized = backend
        .layernorm_forward_f32(
            &rows_tensor,
            Some(&ln_w),
            Some(&ln_b),
            &[rows, hidden],
            rows,
            hidden,
            if spec.ln_eps > 0.0 { spec.ln_eps } else { 1e-5 },
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer LayerNorm", e))?;
    let projected = backend
        .connected_f32(&normalized, &fc1_w, &fc1_b, rows, hidden, hidden, config)
        .map_err(|e| backend_error("GPU LayaScorer FC1", e))?;
    let activated = backend
        .gelu_erf_f32(&projected, &[rows, hidden], config)
        .map_err(|e| backend_error("GPU LayaScorer exact GELU", e))?;
    let logits = backend
        .connected_f32(&activated, &fc2_w, &fc2_b, rows, hidden, 1, config)
        .map_err(|e| backend_error("GPU LayaScorer FC2", e))?;
    backend
        .laya_scorer_pack_f32(
            &logits,
            batch,
            options,
            &tensor_backend_shape(output_shape),
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer pack", e))
}

pub(crate) fn backward_input<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &LayaScorerSpec,
    weights: &LayaScorerWeights,
    input: &B::Tensor,
    seed: &B::Tensor,
    input_shape: TensorShape,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let batch = input_shape.batch;
    let hidden = spec.hidden;
    let options = spec.max_options;
    if batch == 0
        || hidden == 0
        || options == 0
        || input_shape.channels != hidden
        || input_shape.height != 1
        || input_shape.width != options
        || output_shape.batch != batch
        || output_shape.channels != 1
        || output_shape.height != 1
        || output_shape.width != options
    {
        return Err(Error::InvalidConfig(format!(
            "[laya_scorer] source/output dimensions disagree with hidden={hidden}, options={options}"
        )));
    }
    let rows = count("rows", batch, options)?;
    count("input", rows, hidden)?;
    let fc1_len = count("fc1 weights", hidden, hidden)?;
    for (name, actual, expected) in [
        ("ln_w", weights.ln_w.len(), hidden),
        ("ln_b", weights.ln_b.len(), hidden),
        ("fc1_w", weights.fc1_w.len(), fc1_len),
        ("fc1_b", weights.fc1_b.len(), hidden),
        ("fc2_w", weights.fc2_w.len(), hidden),
    ] {
        if actual != expected {
            return Err(Error::InvalidConfig(format!(
                "[laya_scorer] weight {name} has {actual} values, expected {expected}"
            )));
        }
    }
    if !weights.fc2_b.is_finite()
        || weights
            .ln_w
            .iter()
            .chain(&weights.ln_b)
            .chain(&weights.fc1_w)
            .chain(&weights.fc1_b)
            .chain(&weights.fc2_w)
            .any(|v| !v.is_finite())
    {
        return Err(Error::InvalidConfig(
            "[laya_scorer] frozen weights contain non-finite values".into(),
        ));
    }
    let ln_w = backend
        .upload_f32(&[hidden], &weights.ln_w)
        .map_err(|e| backend_error("upload LayaScorer LayerNorm weight", e))?;
    let ln_b = backend
        .upload_f32(&[hidden], &weights.ln_b)
        .map_err(|e| backend_error("upload LayaScorer LayerNorm bias", e))?;
    let fc1_w = backend
        .upload_f32(&[hidden, hidden], &weights.fc1_w)
        .map_err(|e| backend_error("upload LayaScorer FC1 weights", e))?;
    let fc1_b = backend
        .upload_f32(&[hidden], &weights.fc1_b)
        .map_err(|e| backend_error("upload LayaScorer FC1 bias", e))?;
    let fc2_w = backend
        .upload_f32(&[1, hidden], &weights.fc2_w)
        .map_err(|e| backend_error("upload LayaScorer FC2 weights", e))?;
    let eps = if spec.ln_eps > 0.0 { spec.ln_eps } else { 1e-5 };
    let rows_tensor = backend
        .laya_scorer_unpack_f32(
            input,
            &tensor_backend_shape(input_shape),
            hidden,
            options,
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer backward unpack", e))?;
    let normalized = backend
        .layernorm_forward_f32(
            &rows_tensor,
            Some(&ln_w),
            Some(&ln_b),
            &[rows, hidden],
            rows,
            hidden,
            eps,
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer backward LayerNorm", e))?;
    let preactivation = backend
        .connected_f32(&normalized, &fc1_w, &fc1_b, rows, hidden, hidden, config)
        .map_err(|e| backend_error("GPU LayaScorer backward FC1", e))?;
    let activated = backend
        .gelu_erf_f32(&preactivation, &[rows, hidden], config)
        .map_err(|e| backend_error("GPU LayaScorer backward GELU", e))?;
    let seed_rows = backend
        .laya_scorer_seed_rows_f32(seed, batch, options, config)
        .map_err(|e| backend_error("GPU LayaScorer reshape seed", e))?;
    let (d_activated, _, _) = backend
        .connected_backward_f32(&activated, &fc2_w, &seed_rows, rows, hidden, 1, config)
        .map_err(|e| backend_error("GPU LayaScorer FC2 backward", e))?;
    let d_preactivation = backend
        .gelu_erf_backward_f32(&preactivation, &d_activated, &[rows, hidden], config)
        .map_err(|e| backend_error("GPU LayaScorer GELU backward", e))?;
    let (d_normalized, _, _) = backend
        .connected_backward_f32(
            &normalized,
            &fc1_w,
            &d_preactivation,
            rows,
            hidden,
            hidden,
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer FC1 input backward", e))?;
    let d_rows = backend
        .layernorm_backward_f32(
            &d_normalized,
            &rows_tensor,
            Some(&ln_w),
            &[rows, hidden],
            rows,
            hidden,
            eps,
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer LayerNorm backward", e))?;
    backend
        .laya_scorer_pack_input_grad_f32(
            &d_rows,
            batch,
            hidden,
            options,
            &tensor_backend_shape(input_shape),
            config,
        )
        .map_err(|e| backend_error("GPU LayaScorer pack input gradient", e))
}

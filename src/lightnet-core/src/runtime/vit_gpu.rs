//! Native f32 inference composition for `[vit]` (no cache, no MSA, no VC attention).
use super::backend_error;
use crate::error::{Error, Result};
use crate::model::{Activation, TensorShape};
use crate::vit::{VitGeometry, VitWeights};
use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

const FEATURE_CLIP: f32 = 10.0;

/// Device activation matching `vit::tensor_activation`; `None` keeps the host path.
pub(super) fn device_activation(activation: &Activation) -> Option<F32ActivationKind> {
    use Activation::*;
    Some(match activation {
        Linear => F32ActivationKind::Linear,
        Logistic => F32ActivationKind::Logistic,
        Loggy => F32ActivationKind::Loggy,
        Relu => F32ActivationKind::Relu,
        Leaky => F32ActivationKind::Leaky,
        Mish => F32ActivationKind::Mish,
        Swish => F32ActivationKind::Swish,
        Tanh => F32ActivationKind::Tanh,
        Relu6 => F32ActivationKind::Relu6,
        Ramp => F32ActivationKind::Ramp,
        Gelu => F32ActivationKind::Gelu,
        Elu => F32ActivationKind::Elu,
        Selu => F32ActivationKind::Selu,
        Relie => F32ActivationKind::Relie,
        Plse => F32ActivationKind::Plse,
        Hardtan => F32ActivationKind::Hardtan,
        Lhtan => F32ActivationKind::Lhtan,
        Stair => F32ActivationKind::Stair,
        HardMish => F32ActivationKind::HardMish,
        _ => return None,
    })
}

/// Whether the native inference path covers this geometry.
pub(super) fn supported(g: &VitGeometry) -> bool {
    g.msa.is_none()
        && !g.vc_attention
        && g.heads > 0
        && g.hidden % g.heads == 0
        && g.hidden / g.heads <= 128
        && device_activation(&g.activation).is_some()
}

pub(super) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    g: &VitGeometry,
    weights: &VitWeights,
    input: &B::Tensor,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let activation = device_activation(&g.activation)
        .ok_or_else(|| Error::InvalidConfig("[vit] activation has no device path".into()))?;
    if !g.ln_eps.is_finite() || g.ln_eps <= 0.0 {
        return Err(Error::InvalidConfig("[vit] ln_eps must be positive and finite".into()));
    }
    let up = |what: &str, shape: &[usize], data: &[f32]| {
        backend
            .upload_f32(shape, data)
            .map_err(|e| backend_error(&format!("upload vit {what}"), e))
    };
    let (hidden, mlp) = (g.hidden, g.mlp_dim);
    let patch_dim = g.patch_size * g.patch_size * g.channels;
    let (out_h, out_w, patches) = backend
        .vit_patchify_f32(
            input,
            g.batch,
            g.channels,
            g.height,
            g.width,
            g.patch_size,
            g.patch_stride,
            g.patch_pad,
            config,
        )
        .map_err(|e| backend_error("GPU vit patchify", e))?;
    let tokens_per_sample = out_h * out_w;
    let rows = g.batch * tokens_per_sample;
    let expected = |name: &str, got: usize, want: usize| -> Result<()> {
        if got == want {
            Ok(())
        } else {
            Err(Error::InvalidConfig(format!(
                "[vit] {name} has {got} values, expected {want}"
            )))
        }
    };
    expected("patch_embed", weights.patch_embed.len(), hidden * patch_dim)?;
    expected("patch_bias", weights.patch_bias.len(), hidden)?;
    expected("pos_embed", weights.pos_embed.len(), tokens_per_sample * hidden)?;
    expected("qkv_weight", weights.qkv_weight.len(), 3 * hidden * hidden)?;
    expected("qkv_bias", weights.qkv_bias.len(), 3 * hidden)?;
    expected("wo", weights.wo.len(), hidden * hidden)?;
    expected("wo_bias", weights.wo_bias.len(), hidden)?;
    expected("ffn_w1", weights.ffn_w1.len(), mlp * hidden)?;
    expected("ffn_b1", weights.ffn_b1.len(), mlp)?;
    expected("ffn_w2", weights.ffn_w2.len(), hidden * mlp)?;
    expected("ffn_b2", weights.ffn_b2.len(), hidden)?;
    for (name, v) in [
        ("ln1_gamma", &weights.ln1_gamma),
        ("ln1_beta", &weights.ln1_beta),
        ("ln2_gamma", &weights.ln2_gamma),
        ("ln2_beta", &weights.ln2_beta),
    ] {
        expected(name, v.len(), hidden)?;
    }
    let (skip0, branch0, ..) =
        crate::vit::mhc_coefficients(&weights.mhc, 0).map_err(Error::InvalidConfig)?;
    let (skip1, branch1, ..) =
        crate::vit::mhc_coefficients(&weights.mhc, 1).map_err(Error::InvalidConfig)?;

    let lin = |what: &str,
               x: &B::Tensor,
               w: &[f32],
               b: &[f32],
               inf: usize,
               outf: usize|
     -> Result<B::Tensor> {
        let wt = up(&format!("{what} weights"), &[outf, inf], w)?;
        let bt = up(&format!("{what} bias"), &[outf], b)?;
        backend
            .linear_f32(x, &wt, Some(&bt), &[rows, outf], rows, inf, outf, config)
            .map_err(|e| backend_error(&format!("GPU vit {what}"), e))
    };
    let norm = |what: &str, x: &B::Tensor, gamma: &[f32], beta: &[f32]| -> Result<B::Tensor> {
        let gt = up(&format!("{what} gamma"), &[hidden], gamma)?;
        let bt = up(&format!("{what} beta"), &[hidden], beta)?;
        backend
            .layernorm_forward_f32(
                x,
                Some(&gt),
                Some(&bt),
                &[rows, hidden],
                rows,
                hidden,
                g.ln_eps,
                config,
            )
            .map_err(|e| backend_error(&format!("GPU vit {what}"), e))
    };

    let embedded = lin("patch embed", &patches, &weights.patch_embed, &weights.patch_bias, patch_dim, hidden)?;
    let pos = up("positions", &[tokens_per_sample, hidden], &weights.pos_embed)?;
    let tokens = backend
        .vit_add_pos_embed_f32(&embedded, &pos, config)
        .map_err(|e| backend_error("GPU vit positions", e))?;
    let ln1 = norm("ln1", &tokens, &weights.ln1_gamma, &weights.ln1_beta)?;
    let qkv = lin("qkv", &ln1, &weights.qkv_weight, &weights.qkv_bias, hidden, 3 * hidden)?;
    let (context, _lse) = backend
        .flash_attention_forward_f32(
            &qkv,
            &[g.batch, tokens_per_sample, hidden],
            &[g.batch, g.heads, tokens_per_sample],
            g.batch,
            tokens_per_sample,
            hidden,
            g.heads,
        )
        .map_err(|e| backend_error("GPU vit attention", e))?;
    let projection = lin("wo", &context, &weights.wo, &weights.wo_bias, hidden, hidden)?;
    let res1 = backend
        .vit_mhc_residual_f32(&tokens, &projection, skip0, branch0, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit attention residual", e))?;
    let ln2 = norm("ln2", &res1, &weights.ln2_gamma, &weights.ln2_beta)?;
    let ffn1 = lin("ffn1", &ln2, &weights.ffn_w1, &weights.ffn_b1, hidden, mlp)?;
    let clamped = backend
        .vit_clamp_feature_f32(&ffn1, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit ffn pre-clamp", e))?;
    let activated = backend
        .activation_f32(&clamped, activation, config)
        .map_err(|e| backend_error("GPU vit activation", e))?;
    let activated = backend
        .vit_clamp_feature_f32(&activated, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit ffn post-clamp", e))?;
    let ffn2 = lin("ffn2", &activated, &weights.ffn_w2, &weights.ffn_b2, mlp, hidden)?;
    let mixed = backend
        .vit_mhc_residual_f32(&res1, &ffn2, skip1, branch1, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit ffn residual", e))?;
    if output_shape.batch != g.batch
        || output_shape.channels != hidden
        || output_shape.height != out_h
        || output_shape.width != out_w
    {
        return Err(Error::InvalidConfig(format!(
            "[vit] output shape {output_shape} does not match {out_h}x{out_w} tokens"
        )));
    }
    let out = backend
        .vit_tokens_to_nchw_f32(&mixed, g.batch, out_h, out_w, hidden, config)
        .map_err(|e| backend_error("GPU vit output", e))?;
    Ok(out)
}

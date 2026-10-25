//! Native f32 composition for `[vit]` (no MSA, no VC attention): inference
//! forward, a training forward that keeps its intermediates on the device, and
//! the native backward that consumes them. Parameter gradients are staged into
//! the host train buffers, where the existing SGD/Adam/Muon arms update them.
use crate::runtime::{
    GpuExecutableNetwork, LayerTrainBuffers, backend_error, tensor_backend_shape,
};
use crate::error::{Error, Result};
use crate::model::{Activation, Layer, TensorShape};
use crate::vit::{MHC_PARAM_COUNT, VitForwardCache, VitGeometry, VitWeights};
use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

const FEATURE_CLIP: f32 = 10.0;

/// Device activation matching `vit::tensor_activation`; `None` keeps the host path.
pub(crate) fn device_activation(activation: &Activation) -> Option<F32ActivationKind> {
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

/// Whether the native path covers this geometry. The same gate selects the
/// native training forward and backward, so the two always agree.
pub(crate) fn supported(g: &VitGeometry) -> bool {
    g.msa.is_none()
        && !g.vc_attention
        && g.heads > 0
        && g.hidden % g.heads == 0
        && g.hidden / g.heads <= 128
        && device_activation(&g.activation).is_some()
}

/// Device intermediates of one native training forward: the operands of the
/// native backward, plus the weight uploads it reuses. Attention keeps the
/// raw fused QKV, the context and the row log-sum-exp, so the backward
/// recomputes probabilities instead of storing a `tokens x tokens` matrix.
#[derive(Clone, Debug)]
pub(crate) struct DeviceCache<T> {
    geometry: VitGeometry,
    out_h: usize,
    out_w: usize,
    patches: T,
    tokens: T,
    ln1: T,
    qkv: T,
    context: T,
    lse: T,
    projection: T,
    res1: T,
    ln2: T,
    ffn1_raw: T,
    ffn1_clamped: T,
    /// Activation output before the post-activation clamp.
    ffn1_act: T,
    ffn1_activated: T,
    ffn2: T,
    patch_embed: T,
    qkv_weight: T,
    wo: T,
    ffn_w1: T,
    ffn_w2: T,
    ln1_gamma: T,
    ln2_gamma: T,
}

/// Native inference forward; no intermediates are kept.
pub(crate) fn forward<B: F32TensorOpsBackend>(
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
    Ok(run(backend, g, weights, input, output_shape, config, false)?.0)
}

/// Native training forward: the same composition as [`forward`], returning the
/// device cache that [`GpuExecutableNetwork::backward_vit_gpu`] consumes.
pub(crate) fn forward_training<B: F32TensorOpsBackend>(
    backend: &B,
    g: &VitGeometry,
    weights: &VitWeights,
    input: &B::Tensor,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<(B::Tensor, DeviceCache<B::Tensor>)>
where
    B::Error: std::fmt::Display,
{
    let (output, cache) = run(backend, g, weights, input, output_shape, config, true)?;
    let cache = cache.ok_or_else(|| {
        Error::InvalidConfig("[vit] native training forward produced no cache".into())
    })?;
    Ok((output, cache))
}

fn run<B: F32TensorOpsBackend>(
    backend: &B,
    g: &VitGeometry,
    weights: &VitWeights,
    input: &B::Tensor,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
    keep: bool,
) -> Result<(B::Tensor, Option<DeviceCache<B::Tensor>>)>
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
    if output_shape.batch != g.batch
        || output_shape.channels != hidden
        || output_shape.height != out_h
        || output_shape.width != out_w
    {
        return Err(Error::InvalidConfig(format!(
            "[vit] output shape {output_shape} does not match {out_h}x{out_w} tokens"
        )));
    }
    let (skip0, branch0, ..) =
        crate::vit::mhc_coefficients(&weights.mhc, 0).map_err(Error::InvalidConfig)?;
    let (skip1, branch1, ..) =
        crate::vit::mhc_coefficients(&weights.mhc, 1).map_err(Error::InvalidConfig)?;

    let patch_embed = up("patch_embed", &[hidden, patch_dim], &weights.patch_embed)?;
    let patch_bias = up("patch_bias", &[hidden], &weights.patch_bias)?;
    let pos = up("positions", &[tokens_per_sample, hidden], &weights.pos_embed)?;
    let ln1_gamma = up("ln1 gamma", &[hidden], &weights.ln1_gamma)?;
    let ln1_beta = up("ln1 beta", &[hidden], &weights.ln1_beta)?;
    let qkv_weight = up("qkv weights", &[3 * hidden, hidden], &weights.qkv_weight)?;
    let qkv_bias = up("qkv bias", &[3 * hidden], &weights.qkv_bias)?;
    let wo = up("wo weights", &[hidden, hidden], &weights.wo)?;
    let wo_bias = up("wo bias", &[hidden], &weights.wo_bias)?;
    let ln2_gamma = up("ln2 gamma", &[hidden], &weights.ln2_gamma)?;
    let ln2_beta = up("ln2 beta", &[hidden], &weights.ln2_beta)?;
    let ffn_w1 = up("ffn1 weights", &[mlp, hidden], &weights.ffn_w1)?;
    let ffn_b1 = up("ffn1 bias", &[mlp], &weights.ffn_b1)?;
    let ffn_w2 = up("ffn2 weights", &[hidden, mlp], &weights.ffn_w2)?;
    let ffn_b2 = up("ffn2 bias", &[hidden], &weights.ffn_b2)?;

    // Torch `Linear` (`[out, in]` weights) is the connected-layer GEMM with a
    // device bias epilogue.
    let lin = |what: &str,
               x: &B::Tensor,
               w: &B::Tensor,
               b: &B::Tensor,
               inf: usize,
               outf: usize|
     -> Result<B::Tensor> {
        backend
            .connected_f32(x, w, b, rows, inf, outf, config)
            .map_err(|e| backend_error(&format!("GPU vit {what}"), e))
    };
    let norm = |what: &str, x: &B::Tensor, gamma: &B::Tensor, beta: &B::Tensor| {
        backend
            .layernorm_forward_f32(
                x,
                Some(gamma),
                Some(beta),
                &[rows, hidden],
                rows,
                hidden,
                g.ln_eps,
                config,
            )
            .map_err(|e| backend_error(&format!("GPU vit {what}"), e))
    };

    let embedded = lin("patch embed", &patches, &patch_embed, &patch_bias, patch_dim, hidden)?;
    let tokens = backend
        .vit_add_pos_embed_f32(&embedded, &pos, config)
        .map_err(|e| backend_error("GPU vit positions", e))?;
    let ln1 = norm("ln1", &tokens, &ln1_gamma, &ln1_beta)?;
    let qkv = lin("qkv", &ln1, &qkv_weight, &qkv_bias, hidden, 3 * hidden)?;
    let (context, lse) = backend
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
    let projection = lin("wo", &context, &wo, &wo_bias, hidden, hidden)?;
    let res1 = backend
        .vit_mhc_residual_f32(&tokens, &projection, skip0, branch0, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit attention residual", e))?;
    let ln2 = norm("ln2", &res1, &ln2_gamma, &ln2_beta)?;
    let ffn1_raw = lin("ffn1", &ln2, &ffn_w1, &ffn_b1, hidden, mlp)?;
    let ffn1_clamped = backend
        .vit_clamp_feature_f32(&ffn1_raw, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit ffn pre-clamp", e))?;
    let ffn1_act = backend
        .activation_f32(&ffn1_clamped, activation, config)
        .map_err(|e| backend_error("GPU vit activation", e))?;
    let ffn1_activated = backend
        .vit_clamp_feature_f32(&ffn1_act, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit ffn post-clamp", e))?;
    let ffn2 = lin("ffn2", &ffn1_activated, &ffn_w2, &ffn_b2, mlp, hidden)?;
    let mixed = backend
        .vit_mhc_residual_f32(&res1, &ffn2, skip1, branch1, FEATURE_CLIP, config)
        .map_err(|e| backend_error("GPU vit ffn residual", e))?;
    let out = backend
        .vit_tokens_to_nchw_f32(&mixed, g.batch, out_h, out_w, hidden, config)
        .map_err(|e| backend_error("GPU vit output", e))?;
    let cache = keep.then(|| DeviceCache {
        geometry: g.clone(),
        out_h,
        out_w,
        patches,
        tokens,
        ln1,
        qkv,
        context,
        lse,
        projection,
        res1,
        ln2,
        ffn1_raw,
        ffn1_clamped,
        ffn1_act,
        ffn1_activated,
        ffn2,
        patch_embed,
        qkv_weight,
        wo,
        ffn_w1,
        ffn_w2,
        ln1_gamma,
        ln2_gamma,
    });
    Ok((out, cache))
}

/// Host part of the mHC parameter adjoint for one site: the six logits'
/// gradients from the device reductions `[grad_post_skip, grad_post_branch,
/// grad_p]`, exactly as `vit::mhc_residual_backward` folds them.
fn mhc_site_gradient(params: &[f32], site: usize, sums: &[f32], out: &mut [f32]) -> Result<()> {
    let (_, _, p, post_skip, post_branch) =
        crate::vit::mhc_coefficients(params, site).map_err(Error::InvalidConfig)?;
    let &[grad_post_skip, grad_post_branch, grad_p] = sums else {
        return Err(Error::InvalidConfig(format!(
            "[vit] mHC site {site} reduction has {} values, expected 3",
            sums.len()
        )));
    };
    let dz = 0.5 * grad_p * p * (1.0 - p);
    let o = site * 6;
    out[o] = dz;
    out[o + 1] = -dz;
    out[o + 2] = -dz;
    out[o + 3] = dz;
    out[o + 4] = grad_post_skip * post_skip * (1.0 - 0.5 * post_skip);
    out[o + 5] = grad_post_branch * post_branch * (1.0 - 0.5 * post_branch);
    Ok(())
}

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Host ViT forward with a device flash-attention hook; fills the host cache
    /// that the host backward reads (MSA, VC, unmapped activations, and training
    /// networks that need the CPU backward fallback for another layer).
    pub(crate) fn forward_vit_host(
        &self,
        layer: &Layer,
        geometry: &VitGeometry,
        weights: &VitWeights,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, VitForwardCache)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download vit input", err))?;
        let (data, cache) = crate::vit::vit_forward_with_flash_attention(
            &host_input.data,
            weights,
            geometry,
            |qkv, batch, tokens, channels, heads| {
                if channels / heads > 128 {
                    return crate::vit::attention_forward_flash(qkv, batch, tokens, channels, heads);
                }
                let qkv_tensor = self
                    .backend
                    .upload_f32(&[batch, tokens, 3 * channels], qkv)
                    .map_err(|err| format!("upload ViT flash QKV: {err}"))?;
                let (context_tensor, lse_tensor) = self
                    .backend
                    .flash_attention_forward_f32(
                        &qkv_tensor,
                        &[batch, tokens, channels],
                        &[batch, heads, tokens],
                        batch,
                        tokens,
                        channels,
                        heads,
                    )
                    .map_err(|err| format!("ViT flash forward: {err}"))?;
                let context = self
                    .backend
                    .download_f32(&context_tensor)
                    .map_err(|err| format!("download ViT flash context: {err}"))?;
                let lse = self
                    .backend
                    .download_f32(&lse_tensor)
                    .map_err(|err| format!("download ViT flash LSE: {err}"))?;
                let attention = crate::vit::attention_flash_cache_from_lse(
                    qkv, batch, tokens, channels, heads, &lse,
                )?;
                Ok((context, attention))
            },
        )
        .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
            .map_err(|err| backend_error("upload vit output", err))?;
        Ok((output, cache))
    }

    /// Native `[vit]` backward from the device cache of the training forward:
    /// mHC, MLP, LayerNorm, attention (flash dQ/dK/dV), patch-embedding and
    /// positional adjoints all run on the device and the input gradient stays
    /// there. Parameter gradients are then downloaded once and staged into the
    /// host train buffers (`updates -= grad`, as `accumulate_vit_updates`).
    ///
    /// Dense (`flash_attention=0`) networks use the same online-softmax device
    /// attention as the native forward, so their probabilities differ from the
    /// CPU dense softmax only by its `1e-9` denominator epsilon.
    pub(crate) fn backward_vit_gpu(
        &self,
        layer: &Layer,
        geometry: &VitGeometry,
        weights: &VitWeights,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let activation = device_activation(&geometry.activation).ok_or_else(|| {
            Error::UnsupportedLayer {
                layer_index: layer.index,
                name: format!("[vit] unsupported GPU activation {:?}", geometry.activation),
            }
        })?;
        let caches = self.vit_device_caches.borrow();
        let cache = caches
            .get(layer.index)
            .and_then(Option::as_ref)
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing [vit] device forward cache for layer #{}; run a native training forward first",
                    layer.index
                ))
            })?;
        if cache.geometry != *geometry {
            return Err(Error::InvalidConfig(format!(
                "[vit] device cache of layer #{} was built for another geometry",
                layer.index
            )));
        }
        let g = geometry;
        let (hidden, mlp) = (g.hidden, g.mlp_dim);
        let patch_dim = g.patch_size * g.patch_size * g.channels;
        let tokens_per_sample = cache.out_h * cache.out_w;
        let rows = g.batch * tokens_per_sample;
        let be = &self.backend;
        let err = |what: &str| {
            let what = what.to_owned();
            move |e: B::Error| backend_error(&format!("GPU vit backward {what}"), e)
        };
        let (skip0, branch0, p0, post_skip0, post_branch0) =
            crate::vit::mhc_coefficients(&weights.mhc, 0).map_err(Error::InvalidConfig)?;
        let (skip1, branch1, p1, post_skip1, post_branch1) =
            crate::vit::mhc_coefficients(&weights.mhc, 1).map_err(Error::InvalidConfig)?;

        // LayerNorm adjoint: input gradient from the existing kernel; gamma
        // and beta reduce `dy * xhat` and `dy` over rows (xhat is the
        // unscaled normalization of the cached input).
        let layer_norm = |what: &str,
                          dy: &B::Tensor,
                          x: &B::Tensor,
                          gamma: &B::Tensor|
         -> Result<(B::Tensor, B::Tensor, B::Tensor)> {
            let dx = be
                .layernorm_backward_f32(dy, x, Some(gamma), &[rows, hidden], rows, hidden, g.ln_eps, config)
                .map_err(err(what))?;
            let xhat = be
                .layernorm_forward_f32(x, None, None, &[rows, hidden], rows, hidden, g.ln_eps, config)
                .map_err(err(what))?;
            let dy_xhat = be
                .vector_mul_f32(dy, &xhat, &[rows, hidden], config)
                .map_err(err(what))?;
            let dgamma = be.bias_backward_f32(&dy_xhat, hidden, 1, config).map_err(err(what))?;
            let dbeta = be.bias_backward_f32(dy, hidden, 1, config).map_err(err(what))?;
            Ok((dx, dgamma, dbeta))
        };

        let d_mixed = be
            .vit_nchw_to_tokens_f32(&grad, g.batch, cache.out_h, cache.out_w, hidden, config)
            .map_err(err("output layout"))?;
        let (d_res1_skip, d_ffn2, mhc1_sums) = be
            .vit_mhc_residual_backward_f32(
                &cache.res1,
                &cache.ffn2,
                &d_mixed,
                skip1,
                branch1,
                p1,
                post_skip1,
                post_branch1,
                FEATURE_CLIP,
                config,
            )
            .map_err(err("ffn residual"))?;
        let (d_ffn1_activated, d_ffn_w2, d_ffn_b2) = be
            .connected_backward_f32(&cache.ffn1_activated, &cache.ffn_w2, &d_ffn2, rows, mlp, hidden, config)
            .map_err(err("ffn2"))?;
        // Post-clamp mask, activation slope at the clamped input, pre-clamp mask.
        let d_ffn1_act = be
            .vit_clamp_backward_f32(&cache.ffn1_act, &d_ffn1_activated, FEATURE_CLIP, config)
            .map_err(err("ffn post-clamp"))?;
        let d_ffn1_clamped = be
            .activation_backward_f32(&cache.ffn1_clamped, &d_ffn1_act, activation, config)
            .map_err(err("activation"))?;
        let d_ffn1_raw = be
            .vit_clamp_backward_f32(&cache.ffn1_raw, &d_ffn1_clamped, FEATURE_CLIP, config)
            .map_err(err("ffn pre-clamp"))?;
        let (d_ln2, d_ffn_w1, d_ffn_b1) = be
            .connected_backward_f32(&cache.ln2, &cache.ffn_w1, &d_ffn1_raw, rows, hidden, mlp, config)
            .map_err(err("ffn1"))?;
        let (d_res1_norm, d_ln2_gamma, d_ln2_beta) =
            layer_norm("ln2", &d_ln2, &cache.res1, &cache.ln2_gamma)?;
        // Explicit-shape add (no bias): the two addends carry different
        // nominal shapes but the same `[rows, hidden]` layout.
        let add = |what: &str, x: &B::Tensor, y: &B::Tensor| {
            be.residual_bias_f32(x, y, None, &[rows, hidden], rows, hidden, config)
                .map_err(err(what))
        };
        let d_res1 = add("ln2 residual", &d_res1_norm, &d_res1_skip)?;
        let (d_tokens_skip, d_projection, mhc0_sums) = be
            .vit_mhc_residual_backward_f32(
                &cache.tokens,
                &cache.projection,
                &d_res1,
                skip0,
                branch0,
                p0,
                post_skip0,
                post_branch0,
                FEATURE_CLIP,
                config,
            )
            .map_err(err("attention residual"))?;
        let (d_context, d_wo, d_wo_bias) = be
            .connected_backward_f32(&cache.context, &cache.wo, &d_projection, rows, hidden, hidden, config)
            .map_err(err("wo"))?;
        let d_qkv = be
            .flash_attention_backward_f32(
                &cache.qkv,
                &cache.context,
                &d_context,
                &cache.lse,
                &[g.batch, tokens_per_sample, 3 * hidden],
                g.batch,
                tokens_per_sample,
                hidden,
                g.heads,
            )
            .map_err(err("attention"))?;
        let (d_ln1, d_qkv_weight, d_qkv_bias) = be
            .connected_backward_f32(&cache.ln1, &cache.qkv_weight, &d_qkv, rows, hidden, 3 * hidden, config)
            .map_err(err("qkv"))?;
        let (d_tokens_norm, d_ln1_gamma, d_ln1_beta) =
            layer_norm("ln1", &d_ln1, &cache.tokens, &cache.ln1_gamma)?;
        let d_tokens = add("ln1 residual", &d_tokens_norm, &d_tokens_skip)?;
        let (d_patches, d_patch_embed, d_patch_bias) = be
            .connected_backward_f32(&cache.patches, &cache.patch_embed, &d_tokens, rows, patch_dim, hidden, config)
            .map_err(err("patch embed"))?;
        let d_input = be
            .vit_patch_scatter_f32(
                &d_patches,
                g.batch,
                g.channels,
                g.height,
                g.width,
                g.patch_size,
                g.patch_stride,
                g.patch_pad,
                cache.out_h,
                cache.out_w,
                config,
            )
            .map_err(err("patch scatter"))?;
        let d_pos = if g.learned_pos_embed {
            Some(
                be.vit_pos_embed_backward_f32(&d_tokens, g.batch, tokens_per_sample * hidden, config)
                    .map_err(err("positions"))?,
            )
        } else {
            None
        };

        // Parameter staging: one download per tensor at the optimizer boundary.
        let host = |what: &str, t: &B::Tensor| {
            be.download_f32(t)
                .map_err(|e| backend_error(&format!("download vit {what} gradient"), e))
        };
        let mut d_mhc = vec![0.0; MHC_PARAM_COUNT];
        mhc_site_gradient(&weights.mhc, 0, &host("mHC site 0", &mhc0_sums)?, &mut d_mhc)?;
        mhc_site_gradient(&weights.mhc, 1, &host("mHC site 1", &mhc1_sums)?, &mut d_mhc)?;
        let d_pos = d_pos.map(|t| host("pos_embed", &t)).transpose()?;
        {
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let Some(LayerTrainBuffers::Vit(train)) = buffers.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing ViT train buffers for layer #{}",
                    layer.index
                )));
            };
            let u = &mut train.updates;
            let mut staged = vec![
                ("patch_embed", &mut u.patch_embed, host("patch_embed", &d_patch_embed)?),
                ("patch_bias", &mut u.patch_bias, host("patch_bias", &d_patch_bias)?),
                ("qkv_weight", &mut u.qkv_weight, host("qkv_weight", &d_qkv_weight)?),
                ("qkv_bias", &mut u.qkv_bias, host("qkv_bias", &d_qkv_bias)?),
                ("ln1_gamma", &mut u.ln1_gamma, host("ln1_gamma", &d_ln1_gamma)?),
                ("ln1_beta", &mut u.ln1_beta, host("ln1_beta", &d_ln1_beta)?),
                ("wo", &mut u.wo, host("wo", &d_wo)?),
                ("wo_bias", &mut u.wo_bias, host("wo_bias", &d_wo_bias)?),
                ("ln2_gamma", &mut u.ln2_gamma, host("ln2_gamma", &d_ln2_gamma)?),
                ("ln2_beta", &mut u.ln2_beta, host("ln2_beta", &d_ln2_beta)?),
                ("ffn_w1", &mut u.ffn_w1, host("ffn_w1", &d_ffn_w1)?),
                ("ffn_b1", &mut u.ffn_b1, host("ffn_b1", &d_ffn_b1)?),
                ("ffn_w2", &mut u.ffn_w2, host("ffn_w2", &d_ffn_w2)?),
                ("ffn_b2", &mut u.ffn_b2, host("ffn_b2", &d_ffn_b2)?),
                ("mhc", &mut u.mhc, d_mhc),
            ];
            // Sinusoidal tables are not trained; their CPU gradient is zero.
            if let Some(d_pos) = d_pos {
                staged.push(("pos_embed", &mut u.pos_embed, d_pos));
            }
            for (name, dst, gradient) in staged {
                if dst.len() != gradient.len() {
                    return Err(Error::InvalidConfig(format!(
                        "[vit] {name} gradient has {} values, expected {}",
                        gradient.len(),
                        dst.len()
                    )));
                }
                for (dst, src) in dst.iter_mut().zip(gradient) {
                    *dst -= src;
                }
            }
        }
        drop(caches);
        self.accumulate_previous_gpu_gradient(layer.index, d_input, layer_output_gradients)
    }
}

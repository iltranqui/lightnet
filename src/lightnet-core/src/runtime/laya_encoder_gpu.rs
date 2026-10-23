//! Native f32 inference composition for the frozen LayaEncoder.
use super::{backend_error, tensor_backend_shape};
use crate::error::{Error, Result};
use crate::laya_encoder::{LayaEncoderGeometry, LayaEncoderWeights, build_rope_table};
use crate::model::{LayaEncoderSpec, TensorShape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

fn count(name: &str, values: &[usize]) -> Result<usize> {
    let n = values.iter().try_fold(1usize, |acc, &v| acc.checked_mul(v));
    n.filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| {
            Error::InvalidConfig(format!("[laya_encoder] {name}: invalid/overflow extent"))
        })
}

pub(super) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &LayaEncoderSpec,
    weights: &LayaEncoderWeights,
    input: &B::Tensor,
    input_shape: TensorShape,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let (batch, sequence, hidden, heads, layers, intermediate, vocab) = (
        input_shape.batch,
        input_shape.width,
        spec.hidden,
        spec.heads,
        spec.n_layers,
        spec.intermediate,
        spec.vocab,
    );
    if batch == 0
        || sequence == 0
        || hidden == 0
        || heads == 0
        || layers == 0
        || intermediate == 0
        || vocab == 0
        || input_shape.channels != 2
        || input_shape.height != 1
        || output_shape.batch != batch
        || output_shape.channels != hidden
        || output_shape.height != 1
        || output_shape.width != sequence
        || hidden % heads != 0
        || (hidden / heads) % 2 != 0
        || (hidden / heads) > 128
    {
        return Err(Error::InvalidConfig(format!(
            "[laya_encoder] invalid input/output geometry {input_shape} -> {output_shape}"
        )));
    }
    let geometry = LayaEncoderGeometry {
        batch,
        seq_len: sequence,
        hidden,
        heads,
        n_layers: layers,
        intermediate,
        vocab,
        local_window: spec.local_window,
        rope_theta_full: spec.rope_theta_full,
        rope_theta_local: spec.rope_theta_local,
        ln_eps: spec.ln_eps,
    };
    let rows = count("rows", &[batch, sequence])?;
    count("input", &[batch, 2, sequence])?;
    count("hidden rows", &[rows, hidden])?;
    count("embedding table", &[vocab, hidden])?;
    count("attention norms", &[layers - 1, hidden])?;
    count("all QKV weights", &[layers, 3, hidden, hidden])?;
    count("all attention output weights", &[layers, hidden, hidden])?;
    count("all MLP norms", &[layers, hidden])?;
    count("all MLP input weights", &[layers, 2, intermediate, hidden])?;
    count("all MLP output weights", &[layers, hidden, intermediate])?;
    count("RoPE table", &[sequence, hidden / heads / 2])?;
    if !weights.matches(&geometry) {
        return Err(Error::InvalidConfig(
            "[laya_encoder] weight tensors do not match the geometry".into(),
        ));
    }
    let half_dim = hidden / heads / 2;
    let (cos_full_host, sin_full_host) =
        build_rope_table(sequence, hidden / heads, spec.rope_theta_full);
    let (cos_local_host, sin_local_host) =
        build_rope_table(sequence, hidden / heads, spec.rope_theta_local);

    // The large frozen table is uploaded once per call; per-layer matrices are
    // uploaded just before use to bound live device weight storage.
    let embedding = backend
        .upload_f32(&[vocab, hidden], &weights.tok_embed)
        .map_err(|e| backend_error("upload LayaEncoder token embeddings", e))?;
    let (cos_full, sin_full, cos_local, sin_local) = (
        backend
            .upload_f32(&[sequence, half_dim], &cos_full_host)
            .map_err(|e| backend_error("upload LayaEncoder full RoPE cos", e))?,
        backend
            .upload_f32(&[sequence, half_dim], &sin_full_host)
            .map_err(|e| backend_error("upload LayaEncoder full RoPE sin", e))?,
        backend
            .upload_f32(&[sequence, half_dim], &cos_local_host)
            .map_err(|e| backend_error("upload LayaEncoder local RoPE cos", e))?,
        backend
            .upload_f32(&[sequence, half_dim], &sin_local_host)
            .map_err(|e| backend_error("upload LayaEncoder local RoPE sin", e))?,
    );
    let (mut tokens, mask) = backend
        .laya_encoder_embed_f32(
            input,
            &embedding,
            &tensor_backend_shape(input_shape),
            hidden,
            vocab,
            config,
        )
        .map_err(|e| backend_error("GPU LayaEncoder embedding gather", e))?;
    let embed_norm = backend
        .upload_f32(&[hidden], &weights.embed_norm)
        .map_err(|e| backend_error("upload LayaEncoder embedding norm", e))?;
    tokens = backend
        .layernorm_forward_f32(
            &tokens,
            Some(&embed_norm),
            None,
            &[rows, hidden],
            rows,
            hidden,
            spec.ln_eps,
            config,
        )
        .map_err(|e| backend_error("GPU LayaEncoder embedding LayerNorm", e))?;

    let zero_qkv_bias = vec![0.0; 3 * hidden];
    let zero_hidden_bias = vec![0.0; hidden];
    let zero_mlp_bias = vec![0.0; 2 * intermediate];
    let zero_hidden_output_bias = vec![0.0; hidden];
    let window_radius = spec.local_window / 2;
    for block in 0..layers {
        let attention_input = if block == 0 {
            tokens.clone()
        } else {
            let start = (block - 1) * hidden;
            let norm = backend
                .upload_f32(&[hidden], &weights.attn_norm[start..start + hidden])
                .map_err(|e| backend_error("upload LayaEncoder attention norm", e))?;
            backend
                .layernorm_forward_f32(
                    &tokens,
                    Some(&norm),
                    None,
                    &[rows, hidden],
                    rows,
                    hidden,
                    spec.ln_eps,
                    config,
                )
                .map_err(|e| backend_error("GPU LayaEncoder attention LayerNorm", e))?
        };
        let qkv_start = block * 3 * hidden * hidden;
        let qkv_weights = backend
            .upload_f32(
                &[3 * hidden, hidden],
                &weights.wqkv[qkv_start..qkv_start + 3 * hidden * hidden],
            )
            .map_err(|e| backend_error("upload LayaEncoder QKV weights", e))?;
        let qkv_bias = backend
            .upload_f32(&[3 * hidden], &zero_qkv_bias)
            .map_err(|e| backend_error("upload LayaEncoder QKV bias", e))?;
        let qkv = backend
            .connected_f32(
                &attention_input,
                &qkv_weights,
                &qkv_bias,
                rows,
                hidden,
                3 * hidden,
                config,
            )
            .map_err(|e| backend_error("GPU LayaEncoder QKV projection", e))?;
        let sliding = block % 3 != 0;
        let (cos, sin) = if sliding {
            (&cos_local, &sin_local)
        } else {
            (&cos_full, &sin_full)
        };
        let context = backend
            .laya_encoder_attention_f32(
                &qkv,
                &mask,
                cos,
                sin,
                batch,
                sequence,
                hidden,
                heads,
                sliding,
                window_radius,
                &[rows, hidden],
                config,
            )
            .map_err(|e| backend_error("GPU LayaEncoder RoPE attention", e))?;
        let wo_start = block * hidden * hidden;
        let wo = backend
            .upload_f32(
                &[hidden, hidden],
                &weights.wo[wo_start..wo_start + hidden * hidden],
            )
            .map_err(|e| backend_error("upload LayaEncoder attention output weights", e))?;
        let wo_bias = backend
            .upload_f32(&[hidden], &zero_hidden_bias)
            .map_err(|e| backend_error("upload LayaEncoder attention output bias", e))?;
        let projection = backend
            .connected_f32(&context, &wo, &wo_bias, rows, hidden, hidden, config)
            .map_err(|e| backend_error("GPU LayaEncoder attention output projection", e))?;
        tokens = backend
            .vector_add_f32(&tokens, &projection, config)
            .map_err(|e| backend_error("GPU LayaEncoder attention residual", e))?;

        let mlp_norm_start = block * hidden;
        let mlp_norm = backend
            .upload_f32(
                &[hidden],
                &weights.mlp_norm[mlp_norm_start..mlp_norm_start + hidden],
            )
            .map_err(|e| backend_error("upload LayaEncoder MLP norm", e))?;
        let normalized = backend
            .layernorm_forward_f32(
                &tokens,
                Some(&mlp_norm),
                None,
                &[rows, hidden],
                rows,
                hidden,
                spec.ln_eps,
                config,
            )
            .map_err(|e| backend_error("GPU LayaEncoder MLP LayerNorm", e))?;
        let wi_start = block * 2 * intermediate * hidden;
        let wi = backend
            .upload_f32(
                &[2 * intermediate, hidden],
                &weights.wi[wi_start..wi_start + 2 * intermediate * hidden],
            )
            .map_err(|e| backend_error("upload LayaEncoder MLP input weights", e))?;
        let wi_bias = backend
            .upload_f32(&[2 * intermediate], &zero_mlp_bias)
            .map_err(|e| backend_error("upload LayaEncoder MLP input bias", e))?;
        let wi_out = backend
            .connected_f32(
                &normalized,
                &wi,
                &wi_bias,
                rows,
                hidden,
                2 * intermediate,
                config,
            )
            .map_err(|e| backend_error("GPU LayaEncoder MLP input projection", e))?;
        let geglu = backend
            .laya_encoder_geglu_f32(&wi_out, rows, intermediate, &[rows, intermediate], config)
            .map_err(|e| backend_error("GPU LayaEncoder GeGLU", e))?;
        let wo_mlp_start = block * hidden * intermediate;
        let wo_mlp = backend
            .upload_f32(
                &[hidden, intermediate],
                &weights.wo_mlp[wo_mlp_start..wo_mlp_start + hidden * intermediate],
            )
            .map_err(|e| backend_error("upload LayaEncoder MLP output weights", e))?;
        let wo_mlp_bias = backend
            .upload_f32(&[hidden], &zero_hidden_output_bias)
            .map_err(|e| backend_error("upload LayaEncoder MLP output bias", e))?;
        let mlp_projection = backend
            .connected_f32(
                &geglu,
                &wo_mlp,
                &wo_mlp_bias,
                rows,
                intermediate,
                hidden,
                config,
            )
            .map_err(|e| backend_error("GPU LayaEncoder MLP output projection", e))?;
        tokens = backend
            .vector_add_f32(&tokens, &mlp_projection, config)
            .map_err(|e| backend_error("GPU LayaEncoder MLP residual", e))?;
    }

    let final_norm = backend
        .upload_f32(&[hidden], &weights.final_norm)
        .map_err(|e| backend_error("upload LayaEncoder final norm", e))?;
    let tokens = backend
        .layernorm_forward_f32(
            &tokens,
            Some(&final_norm),
            None,
            &[rows, hidden],
            rows,
            hidden,
            spec.ln_eps,
            config,
        )
        .map_err(|e| backend_error("GPU LayaEncoder final LayerNorm", e))?;
    backend
        .laya_encoder_pack_f32(
            &tokens,
            batch,
            sequence,
            hidden,
            &tensor_backend_shape(output_shape),
            config,
        )
        .map_err(|e| backend_error("GPU LayaEncoder output pack", e))
}

//! Native f32 frozen LayaHead forward composition.
use super::{backend_error, tensor_backend_shape};
use crate::error::{Error, Result};
use crate::laya_head::{LayaHeadGeometry, LayaHeadWeights};
use crate::model::{LayaHeadSpec, TensorShape};
use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

fn extent(name: &str, a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .filter(|&n| n > 0 && n <= i32::MAX as usize)
        .ok_or_else(|| {
            Error::InvalidConfig(format!("[laya_head] {name} has zero/overflow dimensions"))
        })
}

pub(super) fn forward<B: F32TensorOpsBackend>(
    backend: &B,
    spec: &LayaHeadSpec,
    weights: &LayaHeadWeights,
    input: &B::Tensor,
    input_shape: TensorShape,
    output_shape: TensorShape,
    config: ElementwiseLaunchConfig,
) -> Result<B::Tensor>
where
    B::Error: std::fmt::Display,
{
    let input_dims = tensor_backend_shape(input_shape);
    let batch = input_shape.batch;
    let sequence = input_shape.width;
    let rows = extent("batch*sequence", batch, sequence)?;
    let hidden = spec.hidden;
    let heads = spec.heads;
    let layers = spec.n_layers;
    let ffn_dim = spec.ffn_dim;
    if input_dims[2] != 1 || hidden == 0 || heads == 0 || hidden % heads != 0 {
        return Err(Error::InvalidConfig(format!(
            "[laya_head] incompatible input geometry {input_dims:?} for hidden={hidden}, heads={heads}"
        )));
    }
    if !(-1..=2).contains(&spec.qtype) {
        return Err(Error::InvalidConfig(format!(
            "[laya_head] invalid qtype {}",
            spec.qtype
        )));
    }
    let input_channels = hidden + if spec.qtype < 0 { 2 } else { 1 };
    if input_dims[1] != input_channels
        || tensor_backend_shape(output_shape) != [batch, hidden, 1, sequence]
    {
        return Err(Error::InvalidConfig(format!(
            "[laya_head] expected input channels={input_channels}, output=[{batch},{hidden},1,{sequence}], got {input_dims:?} -> {:?}",
            tensor_backend_shape(output_shape)
        )));
    }
    let geometry = LayaHeadGeometry {
        batch,
        seq_len: sequence,
        hidden,
        heads,
        n_layers: layers,
        ffn_dim,
        ln_eps: if spec.ln_eps > 0.0 {
            spec.ln_eps
        } else {
            crate::model::LAYA_LAYERNORM_EPS
        },
        qtype: spec.qtype,
    };
    if !weights.matches(&geometry) {
        return Err(Error::InvalidConfig(
            "[laya_head] weight tensors do not match the geometry".into(),
        ));
    }
    let token_len = extent("batch*sequence*hidden", rows, hidden)?;
    let qkv_width = extent("3*hidden", hidden, 3)?;
    let qkv_len = extent("batch*sequence*3*hidden", rows, qkv_width)?;
    let ffn_len = extent("batch*sequence*ffn_dim", rows, ffn_dim)?;
    let _ = (token_len, qkv_len, ffn_len);
    let (mut tokens, mask) = backend
        .laya_head_prepare_f32(
            input,
            &input_dims,
            &weights.type_emb,
            hidden,
            spec.qtype,
            config,
        )
        .map_err(|e| backend_error("GPU LayaHead type embedding / unpack", e))?;

    for block in 0..layers {
        let h = block * hidden;
        let qkv_w = block * 3 * hidden * hidden;
        let qkv_b = block * 3 * hidden;
        let out_w = block * hidden * hidden;
        let out_b = block * hidden;
        let ffn1_w = block * ffn_dim * hidden;
        let ffn1_b = block * ffn_dim;
        let ffn2_w = block * hidden * ffn_dim;
        let ffn2_b = block * hidden;
        let norm1_w = backend
            .upload_f32(&[hidden], &weights.norm1_w[h..h + hidden])
            .map_err(|e| backend_error("upload LayaHead norm1 gamma", e))?;
        let norm1_b = backend
            .upload_f32(&[hidden], &weights.norm1_b[h..h + hidden])
            .map_err(|e| backend_error("upload LayaHead norm1 beta", e))?;
        let norm2_w = backend
            .upload_f32(&[hidden], &weights.norm2_w[h..h + hidden])
            .map_err(|e| backend_error("upload LayaHead norm2 gamma", e))?;
        let norm2_b = backend
            .upload_f32(&[hidden], &weights.norm2_b[h..h + hidden])
            .map_err(|e| backend_error("upload LayaHead norm2 beta", e))?;
        let in_w = backend
            .upload_f32(
                &[3 * hidden, hidden],
                &weights.in_proj_w[qkv_w..qkv_w + 3 * hidden * hidden],
            )
            .map_err(|e| backend_error("upload LayaHead QKV weights", e))?;
        let in_b = backend
            .upload_f32(&[3 * hidden], &weights.in_proj_b[qkv_b..qkv_b + 3 * hidden])
            .map_err(|e| backend_error("upload LayaHead QKV bias", e))?;
        let out_proj_w = backend
            .upload_f32(
                &[hidden, hidden],
                &weights.out_proj_w[out_w..out_w + hidden * hidden],
            )
            .map_err(|e| backend_error("upload LayaHead output projection weights", e))?;
        let out_proj_b = backend
            .upload_f32(&[hidden], &weights.out_proj_b[out_b..out_b + hidden])
            .map_err(|e| backend_error("upload LayaHead output projection bias", e))?;
        let linear1_w = backend
            .upload_f32(
                &[ffn_dim, hidden],
                &weights.linear1_w[ffn1_w..ffn1_w + ffn_dim * hidden],
            )
            .map_err(|e| backend_error("upload LayaHead FFN1 weights", e))?;
        let linear1_b = backend
            .upload_f32(&[ffn_dim], &weights.linear1_b[ffn1_b..ffn1_b + ffn_dim])
            .map_err(|e| backend_error("upload LayaHead FFN1 bias", e))?;
        let linear2_w = backend
            .upload_f32(
                &[hidden, ffn_dim],
                &weights.linear2_w[ffn2_w..ffn2_w + hidden * ffn_dim],
            )
            .map_err(|e| backend_error("upload LayaHead FFN2 weights", e))?;
        let linear2_b = backend
            .upload_f32(&[hidden], &weights.linear2_b[ffn2_b..ffn2_b + hidden])
            .map_err(|e| backend_error("upload LayaHead FFN2 bias", e))?;

        let normalized = backend
            .layernorm_forward_f32(
                &tokens,
                Some(&norm1_w),
                Some(&norm1_b),
                &[rows, hidden],
                rows,
                hidden,
                geometry.ln_eps,
                config,
            )
            .map_err(|e| backend_error("GPU LayaHead norm1", e))?;
        let qkv = backend
            .connected_f32(&normalized, &in_w, &in_b, rows, hidden, 3 * hidden, config)
            .map_err(|e| backend_error("GPU LayaHead QKV projection", e))?;
        let context = backend
            .laya_head_attention_f32(
                &qkv,
                &mask,
                &[rows, hidden],
                batch,
                sequence,
                hidden,
                heads,
                config,
            )
            .map_err(|e| backend_error("GPU LayaHead masked attention", e))?;
        let projected = backend
            .connected_f32(
                &context,
                &out_proj_w,
                &out_proj_b,
                rows,
                hidden,
                hidden,
                config,
            )
            .map_err(|e| backend_error("GPU LayaHead output projection", e))?;
        tokens = backend
            .vector_add_f32(&tokens, &projected, config)
            .map_err(|e| backend_error("GPU LayaHead attention residual", e))?;

        let normalized = backend
            .layernorm_forward_f32(
                &tokens,
                Some(&norm2_w),
                Some(&norm2_b),
                &[rows, hidden],
                rows,
                hidden,
                geometry.ln_eps,
                config,
            )
            .map_err(|e| backend_error("GPU LayaHead norm2", e))?;
        let ffn = backend
            .connected_f32(
                &normalized,
                &linear1_w,
                &linear1_b,
                rows,
                hidden,
                ffn_dim,
                config,
            )
            .map_err(|e| backend_error("GPU LayaHead FFN1", e))?;
        let ffn = backend
            .activation_f32(&ffn, F32ActivationKind::Relu, config)
            .map_err(|e| backend_error("GPU LayaHead FFN ReLU", e))?;
        let ffn = backend
            .connected_f32(&ffn, &linear2_w, &linear2_b, rows, ffn_dim, hidden, config)
            .map_err(|e| backend_error("GPU LayaHead FFN2", e))?;
        tokens = backend
            .vector_add_f32(&tokens, &ffn, config)
            .map_err(|e| backend_error("GPU LayaHead FFN residual", e))?;
    }
    backend
        .laya_head_pack_f32(
            &tokens,
            batch,
            sequence,
            hidden,
            &tensor_backend_shape(output_shape),
            config,
        )
        .map_err(|e| backend_error("GPU LayaHead output layout", e))
}

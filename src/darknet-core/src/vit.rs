//! Pure host helpers for the `[vit]` layer.
//!
//! Layouts follow `Jev_laya:src-lib/vit_layer.cpp`: input is NCHW, patches and
//! token features are batch-major NTC, and patch flattening is `(patch_y,
//! patch_x, channel)` inside each token.

use crate::model::Activation;

/// Patchify NCHW input into `[batch, out_h*out_w, patch_size*patch_size*channels]`.
/// Samples outside the image are zero, matching `vit_patchify_cpu`.
pub fn patchify_nchw(
    input: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    patch_stride: usize,
    pad: usize,
) -> Result<(usize, usize, Vec<f32>), String> {
    if batch == 0 || channels == 0 || height == 0 || width == 0 {
        return Err("vit patchify requires nonzero batch, channels, height, and width".into());
    }
    if patch_size == 0 || patch_stride == 0 {
        return Err("vit patch_size and patch_stride must be > 0".into());
    }
    let twice_pad = pad.checked_mul(2).ok_or("vit patch padding overflow")?;
    let eff_h = height
        .checked_add(twice_pad)
        .ok_or("vit patch height overflow")?;
    let eff_w = width
        .checked_add(twice_pad)
        .ok_or("vit patch width overflow")?;
    let out_h = eff_h
        .checked_sub(patch_size)
        .ok_or("vit patch is larger than padded input height")?;
    let out_w = eff_w
        .checked_sub(patch_size)
        .ok_or("vit patch is larger than padded input width")?;
    if out_h % patch_stride != 0 || out_w % patch_stride != 0 {
        return Err("vit effective patch extents must be divisible by patch_stride".into());
    }
    let out_h = out_h / patch_stride + 1;
    let out_w = out_w / patch_stride + 1;
    let input_len = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(height))
        .and_then(|v| v.checked_mul(width))
        .ok_or("vit input size overflow")?;
    if input.len() != input_len {
        return Err(format!(
            "vit input has {} values, expected {input_len}",
            input.len()
        ));
    }
    let tokens = out_h.checked_mul(out_w).ok_or("vit token count overflow")?;
    let patch_dim = patch_size
        .checked_mul(patch_size)
        .and_then(|v| v.checked_mul(channels))
        .ok_or("vit patch dimension overflow")?;
    let output_len = batch
        .checked_mul(tokens)
        .and_then(|v| v.checked_mul(patch_dim))
        .ok_or("vit patch output size overflow")?;
    let mut patches = vec![0.0; output_len];
    for b in 0..batch {
        for py in 0..out_h {
            for px in 0..out_w {
                let t = py * out_w + px;
                for dy in 0..patch_size {
                    let y = py * patch_stride + dy;
                    if y < pad || y - pad >= height {
                        continue;
                    }
                    let iy = y - pad;
                    for dx in 0..patch_size {
                        let x = px * patch_stride + dx;
                        if x < pad || x - pad >= width {
                            continue;
                        }
                        let ix = x - pad;
                        let spatial = iy * width + ix;
                        let patch_offset = (dy * patch_size + dx) * channels;
                        for c in 0..channels {
                            patches[(b * tokens + t) * patch_dim + patch_offset + c] =
                                input[(b * channels + c) * height * width + spatial];
                        }
                    }
                }
            }
        }
    }
    Ok((out_h, out_w, patches))
}

/// Scatter `[batch, tokens, patch_dim]` patch gradients into an NCHW input
/// gradient. Overlapping patches accumulate into shared input pixels.
pub fn scatter_patch_gradients_nchw(
    patch_gradients: &[f32],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    patch_stride: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>, String> {
    if batch == 0
        || channels == 0
        || height == 0
        || width == 0
        || patch_size == 0
        || patch_stride == 0
    {
        return Err("vit patch gradient scatter requires positive dimensions".into());
    }
    let tokens = out_h.checked_mul(out_w).ok_or("vit token count overflow")?;
    let patch_dim = patch_size
        .checked_mul(patch_size)
        .and_then(|v| v.checked_mul(channels))
        .ok_or("vit patch dimension overflow")?;
    let expected = batch
        .checked_mul(tokens)
        .and_then(|v| v.checked_mul(patch_dim))
        .ok_or("vit patch gradient size overflow")?;
    if patch_gradients.len() != expected {
        return Err(format!(
            "vit patch gradients have {} values, expected {expected}",
            patch_gradients.len()
        ));
    }
    let input_len = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(height))
        .and_then(|v| v.checked_mul(width))
        .ok_or("vit input gradient size overflow")?;
    let mut dx = vec![0.0; input_len];
    for b in 0..batch {
        for py in 0..out_h {
            for px in 0..out_w {
                let t = py * out_w + px;
                for dy in 0..patch_size {
                    let y = py * patch_stride + dy;
                    if y < pad || y - pad >= height {
                        continue;
                    }
                    let iy = y - pad;
                    for dx_patch in 0..patch_size {
                        let x = px * patch_stride + dx_patch;
                        if x < pad || x - pad >= width {
                            continue;
                        }
                        let ix = x - pad;
                        let spatial = iy * width + ix;
                        let patch_offset = (dy * patch_size + dx_patch) * channels;
                        for c in 0..channels {
                            dx[(b * channels + c) * height * width + spatial] +=
                                patch_gradients[(b * tokens + t) * patch_dim + patch_offset + c];
                        }
                    }
                }
            }
        }
    }
    Ok(dx)
}

/// C++ `vit_fill_2d_sinusoidal_pos_embed`, row-major `[height*width, channels]`.
pub fn sinusoidal_pos_embed(height: usize, width: usize, channels: usize) -> Vec<f32> {
    let quarter = (channels / 4).max(1);
    let mut out = vec![0.0; height * width * channels];
    for y in 0..height {
        for x in 0..width {
            let row = (y * width + x) * channels;
            for c in 0..channels {
                let band = c / 4;
                let div = (-10000.0f32.ln() * band as f32 / quarter as f32).exp();
                out[row + c] = match c % 4 {
                    0 => (y as f32 * div).sin(),
                    1 => (y as f32 * div).cos(),
                    2 => (x as f32 * div).sin(),
                    _ => (x as f32 * div).cos(),
                };
            }
        }
    }
    out
}

/// Cache for the affine LayerNorm backward.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNormCache {
    rows: usize,
    cols: usize,
    xhat: Vec<f32>,
    inv_std: Vec<f32>,
}

/// Row-wise population LayerNorm with learned gamma/beta, matching the ViT
/// LayerNorm math (`eps=1e-5`). Returns `(output, cache)`.
pub fn layer_norm_forward(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(Vec<f32>, LayerNormCache), String> {
    if rows == 0 || cols == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err("vit LayerNorm requires positive rows/cols and positive finite eps".into());
    }
    let len = rows
        .checked_mul(cols)
        .ok_or("vit LayerNorm size overflow")?;
    if x.len() != len || gamma.len() != cols || beta.len() != cols {
        return Err(format!(
            "vit LayerNorm got x/gamma/beta lengths {}/{}/{}, expected {len}/{cols}/{cols}",
            x.len(),
            gamma.len(),
            beta.len()
        ));
    }
    let mut out = vec![0.0; len];
    let mut xhat = vec![0.0; len];
    let mut inv_std = vec![0.0; rows];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let variance = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (variance + eps).sqrt();
        inv_std[r] = inv;
        for c in 0..cols {
            let normalized = (row[c] - mean) * inv;
            xhat[r * cols + c] = normalized;
            out[r * cols + c] = normalized * gamma[c] + beta[c];
        }
    }
    Ok((
        out,
        LayerNormCache {
            rows,
            cols,
            xhat,
            inv_std,
        },
    ))
}

/// Exact LayerNorm backward including shared gamma and beta gradients.
pub fn layer_norm_backward(
    output_grad: &[f32],
    gamma: &[f32],
    cache: &LayerNormCache,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let LayerNormCache {
        rows,
        cols,
        xhat,
        inv_std,
    } = cache;
    let len = rows * cols;
    if output_grad.len() != len || gamma.len() != *cols {
        return Err(format!(
            "vit LayerNorm backward got dy/gamma lengths {}/{}, expected {len}/{}",
            output_grad.len(),
            gamma.len(),
            cols
        ));
    }
    let mut dx = vec![0.0; len];
    let mut dgamma = vec![0.0; *cols];
    let mut dbeta = vec![0.0; *cols];
    for r in 0..*rows {
        let mut sum_dxh = 0.0;
        let mut sum_dxh_xh = 0.0;
        for c in 0..*cols {
            let dy = output_grad[r * cols + c];
            let dxh = dy * gamma[c];
            let xh = xhat[r * cols + c];
            sum_dxh += dxh;
            sum_dxh_xh += dxh * xh;
            dgamma[c] += dy * xh;
            dbeta[c] += dy;
        }
        for c in 0..*cols {
            let dxh = output_grad[r * cols + c] * gamma[c];
            dx[r * cols + c] =
                inv_std[r] * (dxh - (sum_dxh + xhat[r * cols + c] * sum_dxh_xh) / *cols as f32);
        }
    }
    Ok((dx, dgamma, dbeta))
}

/// Row-major torch `Linear`: `y[r,o] = sum_i x[r,i]*weights[o,i] + bias[o]`.
pub fn linear_forward(
    x: &[f32],
    weights: &[f32],
    bias: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || in_features == 0 || out_features == 0 {
        return Err("vit linear requires positive dimensions".into());
    }
    let x_len = rows
        .checked_mul(in_features)
        .ok_or("vit linear input size overflow")?;
    let w_len = out_features
        .checked_mul(in_features)
        .ok_or("vit linear weight size overflow")?;
    let y_len = rows
        .checked_mul(out_features)
        .ok_or("vit linear output size overflow")?;
    if x.len() != x_len || weights.len() != w_len || bias.len() != out_features {
        return Err(format!(
            "vit linear got x/w/b lengths {}/{}/{}, expected {x_len}/{w_len}/{out_features}",
            x.len(),
            weights.len(),
            bias.len()
        ));
    }
    let mut y = vec![0.0; y_len];
    for r in 0..rows {
        for o in 0..out_features {
            let mut sum = bias[o];
            for i in 0..in_features {
                sum += x[r * in_features + i] * weights[o * in_features + i];
            }
            y[r * out_features + o] = sum;
        }
    }
    Ok(y)
}

/// Exact linear backward; weight layout matches [`linear_forward`].
pub fn linear_backward(
    x: &[f32],
    weights: &[f32],
    output_grad: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if rows == 0 || in_features == 0 || out_features == 0 {
        return Err("vit linear backward dimensions must be positive".into());
    }
    let x_len = rows
        .checked_mul(in_features)
        .ok_or("vit linear input size overflow")?;
    let w_len = out_features
        .checked_mul(in_features)
        .ok_or("vit linear weight size overflow")?;
    let y_len = rows
        .checked_mul(out_features)
        .ok_or("vit linear output size overflow")?;
    if x.len() != x_len || weights.len() != w_len || output_grad.len() != y_len {
        return Err("vit linear backward shape mismatch".into());
    }
    let mut dx = vec![0.0; x_len];
    let mut dw = vec![0.0; w_len];
    let mut db = vec![0.0; out_features];
    for r in 0..rows {
        for o in 0..out_features {
            let dy = output_grad[r * out_features + o];
            db[o] += dy;
            for i in 0..in_features {
                dx[r * in_features + i] += dy * weights[o * in_features + i];
                dw[o * in_features + i] += dy * x[r * in_features + i];
            }
        }
    }
    Ok((dx, dw, db))
}

/// Attention intermediates required by the analytical backward pass.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionCache {
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
    head_dim: usize,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    q_active: Vec<bool>,
    k_active: Vec<bool>,
    probabilities: Vec<f32>,
    score_active: Vec<bool>,
    flash_lse: Option<Vec<f32>>,
    vc_heads: Option<Vec<crate::vc_attention::VcHeadCache>>,
}

pub(crate) struct FlashAttentionBackwardInputs {
    pub qkv: Vec<f32>,
    pub lse: Vec<f32>,
    pub batch: usize,
    pub tokens: usize,
    pub channels: usize,
    pub heads: usize,
}

/// Build backend inputs from the forward cache without retaining raw QKV. The
/// synthetic NaN placeholders preserve the forward's non-finite-to-zero value
/// and zero clamp derivative; saturated finite values are represented by ±128.
pub(crate) fn attention_flash_backward_inputs(
    cache: &AttentionCache,
) -> Result<Option<FlashAttentionBackwardInputs>, String> {
    let Some(lse) = &cache.flash_lse else {
        return Ok(None);
    };
    let elements = cache
        .batch
        .checked_mul(cache.tokens)
        .and_then(|n| n.checked_mul(cache.channels))
        .ok_or_else(|| "vit flash backward cache size overflow".to_owned())?;
    if cache.q.len() != elements
        || cache.k.len() != elements
        || cache.v.len() != elements
        || cache.q_active.len() != elements
        || cache.k_active.len() != elements
        || lse.len() != cache.batch * cache.heads * cache.tokens
    {
        return Err("vit flash backward cache tensors have inconsistent lengths".into());
    }
    let mut qkv = vec![0.0; elements * 3];
    for i in 0..elements {
        let raw_q = if cache.q_active[i] {
            cache.q[i]
        } else if cache.q[i].abs() < 128.0 {
            f32::NAN
        } else {
            cache.q[i]
        };
        let raw_k = if cache.k_active[i] {
            cache.k[i]
        } else if cache.k[i].abs() < 128.0 {
            f32::NAN
        } else {
            cache.k[i]
        };
        let row = i / cache.channels;
        let channel = i % cache.channels;
        qkv[row * 3 * cache.channels + channel] = raw_q;
        qkv[row * 3 * cache.channels + cache.channels + channel] = raw_k;
        qkv[row * 3 * cache.channels + 2 * cache.channels + channel] = cache.v[i];
    }
    Ok(Some(FlashAttentionBackwardInputs {
        qkv,
        lse: lse.clone(),
        batch: cache.batch,
        tokens: cache.tokens,
        channels: cache.channels,
        heads: cache.heads,
    }))
}

fn attention_score(
    q: &[f32],
    k: &[f32],
    query_row: usize,
    key_row: usize,
    channels: usize,
    head_start: usize,
    head_dim: usize,
    scale: f32,
) -> (f32, bool) {
    let mut dot = 0.0;
    for d in 0..head_dim {
        let c = head_start + d;
        dot += q[query_row * channels + c] * k[key_row * channels + c];
    }
    let raw = dot * scale;
    let active = raw.is_finite() && raw.abs() < 10.0;
    let score = if raw.is_nan() {
        0.0
    } else if raw.is_infinite() {
        if raw.is_sign_positive() { 10.0 } else { -10.0 }
    } else {
        raw.clamp(-10.0, 10.0)
    };
    (score, active)
}

/// Global multi-head attention for fused per-token QKV `[B,T,3N]`. When
/// `sparse_blocks` is `Some((block_size, topk_blocks))`, applies the C++ MSA
/// row selection; otherwise each row uses dense softmax. Q/K clamp to 128,
/// score clamp to 10, as in `forward_vit_layer`.
pub fn attention_forward(
    qkv: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
    sparse_blocks: Option<(usize, usize)>,
) -> Result<(Vec<f32>, AttentionCache), String> {
    attention_forward_impl(qkv, batch, tokens, channels, heads, sparse_blocks, false)
}

/// Online-softmax attention using O(B*T*channels) cached intermediates rather
/// than materializing the O(B*heads*T*T) probability matrix. Backward recomputes
/// each probability from the saved row log-sum-exp.
pub fn attention_forward_flash(
    qkv: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
) -> Result<(Vec<f32>, AttentionCache), String> {
    attention_forward_impl(qkv, batch, tokens, channels, heads, None, true)
}

/// Source-shaped VC attention over all batch-heads. Each head owns an O(T*C)
/// reference cache; no B*H*T*T probability tensor is materialized.
pub(crate) fn attention_forward_vc(
    qkv: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
    mode: crate::vc_attention::ExpCastMode,
    permutation: Option<&[usize]>,
) -> Result<(Vec<f32>, AttentionCache), String> {
    if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
        return Err(
            "vit VC attention requires positive dimensions and channels divisible by heads".into(),
        );
    }
    let rows = batch
        .checked_mul(tokens)
        .ok_or("vit VC row count overflow")?;
    let qkv_width = channels.checked_mul(3).ok_or("vit VC QKV width overflow")?;
    let expected = rows
        .checked_mul(qkv_width)
        .ok_or("vit VC QKV size overflow")?;
    if qkv.len() != expected {
        return Err(format!(
            "vit VC QKV has {} values, expected {expected}",
            qkv.len()
        ));
    }
    let head_dim = channels / heads;
    let head_elements = tokens
        .checked_mul(head_dim)
        .ok_or("vit VC head size overflow")?;
    let output_len = rows
        .checked_mul(channels)
        .ok_or("vit VC output size overflow")?;
    let mut context = vec![0.0; output_len];
    let head_count = batch
        .checked_mul(heads)
        .ok_or("vit VC batch-head count overflow")?;
    let mut vc_heads = Vec::with_capacity(head_count);
    let scale = 1.0 / (head_dim as f32).sqrt();
    for b in 0..batch {
        for h in 0..heads {
            let channel_offset = h * head_dim;
            let mut q = vec![0.0; head_elements];
            let mut k = vec![0.0; head_elements];
            let mut v = vec![0.0; head_elements];
            for token in 0..tokens {
                let source = (b * tokens + token) * qkv_width;
                for d in 0..head_dim {
                    let c = channel_offset + d;
                    let target = token * head_dim + d;
                    q[target] = qkv[source + c];
                    k[target] = qkv[source + channels + c];
                    v[target] = qkv[source + 2 * channels + c];
                }
            }
            let (head_context, _head_lse, head_cache) = crate::vc_attention::vc_head_forward(
                &q,
                &k,
                &v,
                tokens,
                head_dim,
                scale,
                10.0,
                mode,
                permutation,
            )?;
            for token in 0..tokens {
                for d in 0..head_dim {
                    let c = channel_offset + d;
                    context[(b * tokens + token) * channels + c] =
                        head_context[token * head_dim + d];
                }
            }
            vc_heads.push(head_cache);
        }
    }
    Ok((
        context,
        AttentionCache {
            batch,
            tokens,
            channels,
            heads,
            head_dim,
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            q_active: Vec::new(),
            k_active: Vec::new(),
            probabilities: Vec::new(),
            score_active: Vec::new(),
            flash_lse: None,
            vc_heads: Some(vc_heads),
        },
    ))
}

/// Construct the analytical-backward cache for a context/LSE pair produced by
/// a native online-attention forward. Q/K clamp masks are rebuilt from the
/// original fused QKV; score probabilities remain implicit in row LSE.
pub fn attention_flash_cache_from_lse(
    qkv: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
    lse: &[f32],
) -> Result<AttentionCache, String> {
    if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
        return Err(
            "vit flash cache requires positive dimensions and channels divisible by heads".into(),
        );
    }
    let rows = batch
        .checked_mul(tokens)
        .ok_or("vit flash-cache row count overflow")?;
    let qkv_width = channels.checked_mul(3).ok_or("vit QKV width overflow")?;
    let expected_qkv = rows.checked_mul(qkv_width).ok_or("vit QKV size overflow")?;
    if qkv.len() != expected_qkv {
        return Err(format!(
            "vit QKV has {} values, expected {expected_qkv}",
            qkv.len()
        ));
    }
    let lse_len = batch
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(tokens))
        .ok_or("vit flash-cache LSE size overflow")?;
    if lse.len() != lse_len {
        return Err(format!(
            "vit flash LSE has {} values, expected {lse_len}",
            lse.len()
        ));
    }
    let q_len = rows
        .checked_mul(channels)
        .ok_or("vit QKV component size overflow")?;
    let mut q = vec![0.0; q_len];
    let mut k = vec![0.0; q_len];
    let mut v = vec![0.0; q_len];
    let mut q_active = vec![false; q_len];
    let mut k_active = vec![false; q_len];
    for row in 0..rows {
        for c in 0..channels {
            let raw_q = qkv[row * 3 * channels + c];
            let raw_k = qkv[row * 3 * channels + channels + c];
            let i = row * channels + c;
            q_active[i] = raw_q.is_finite() && raw_q.abs() < 128.0;
            k_active[i] = raw_k.is_finite() && raw_k.abs() < 128.0;
            q[i] = clamp_feature(raw_q, 128.0);
            k[i] = clamp_feature(raw_k, 128.0);
            v[i] = qkv[row * 3 * channels + 2 * channels + c];
        }
    }
    Ok(AttentionCache {
        batch,
        tokens,
        channels,
        heads,
        head_dim: channels / heads,
        q,
        k,
        v,
        q_active,
        k_active,
        probabilities: Vec::new(),
        score_active: Vec::new(),
        flash_lse: Some(lse.to_vec()),
        vc_heads: None,
    })
}

fn attention_forward_impl(
    qkv: &[f32],
    batch: usize,
    tokens: usize,
    channels: usize,
    heads: usize,
    sparse_blocks: Option<(usize, usize)>,
    flash: bool,
) -> Result<(Vec<f32>, AttentionCache), String> {
    if batch == 0 || tokens == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
        return Err(
            "vit attention requires positive dimensions and channels divisible by heads".into(),
        );
    }
    if flash && sparse_blocks.is_some() {
        return Err("vit flash attention and MSA are mutually exclusive".into());
    }
    let rows = batch
        .checked_mul(tokens)
        .ok_or("vit attention row count overflow")?;
    let qkv_width = channels.checked_mul(3).ok_or("vit QKV width overflow")?;
    let expected = rows.checked_mul(qkv_width).ok_or("vit QKV size overflow")?;
    if qkv.len() != expected {
        return Err(format!(
            "vit QKV has {} values, expected {expected}",
            qkv.len()
        ));
    }
    let head_dim = channels / heads;
    let mut q = vec![0.0; rows * channels];
    let mut k = vec![0.0; rows * channels];
    let mut v = vec![0.0; rows * channels];
    let mut q_active = vec![false; rows * channels];
    let mut k_active = vec![false; rows * channels];
    for row in 0..rows {
        for c in 0..channels {
            let q_raw = qkv[row * 3 * channels + c];
            let k_raw = qkv[row * 3 * channels + channels + c];
            let qi = row * channels + c;
            q_active[qi] = q_raw.is_finite() && q_raw.abs() < 128.0;
            k_active[qi] = k_raw.is_finite() && k_raw.abs() < 128.0;
            q[qi] = clamp_feature(q_raw, 128.0);
            k[qi] = clamp_feature(k_raw, 128.0);
            v[qi] = qkv[row * 3 * channels + 2 * channels + c];
        }
    }
    let rows_per_head = batch
        .checked_mul(heads)
        .and_then(|v| v.checked_mul(tokens))
        .ok_or("vit attention head-row count overflow")?;
    let score_count = if flash {
        0
    } else {
        rows_per_head
            .checked_mul(tokens)
            .ok_or("vit attention score size overflow")?
    };
    let output_len = rows
        .checked_mul(channels)
        .ok_or("vit attention output size overflow")?;
    let mut probabilities = vec![0.0; score_count];
    let mut score_active = vec![false; score_count];
    let mut flash_lse = if flash {
        vec![0.0; rows_per_head]
    } else {
        Vec::new()
    };
    let mut output = vec![0.0; output_len];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for b in 0..batch {
        for h in 0..heads {
            let head_row = b * heads + h;
            let dense_head_offset = if flash { 0 } else { head_row * tokens * tokens };
            for tq in 0..tokens {
                let dense_row_offset = if flash {
                    0
                } else {
                    dense_head_offset + tq * tokens
                };
                let query_row = b * tokens + tq;
                if flash {
                    let mut running_max = f32::NEG_INFINITY;
                    let mut running_sum = 0.0;
                    let mut acc = vec![0.0; head_dim];
                    for tk in 0..tokens {
                        let key_row = b * tokens + tk;
                        let (score, _) = attention_score(
                            &q,
                            &k,
                            query_row,
                            key_row,
                            channels,
                            h * head_dim,
                            head_dim,
                            scale,
                        );
                        let new_max = running_max.max(score);
                        let correction = (running_max - new_max).exp();
                        let p = (score - new_max).exp();
                        running_sum = running_sum * correction + p;
                        for d in 0..head_dim {
                            let c = h * head_dim + d;
                            acc[d] = acc[d] * correction + p * v[key_row * channels + c];
                        }
                        running_max = new_max;
                    }
                    let lse = running_max + running_sum.ln();
                    flash_lse[head_row * tokens + tq] = lse;
                    for d in 0..head_dim {
                        let c = h * head_dim + d;
                        output[query_row * channels + c] = acc[d] / running_sum;
                    }
                } else {
                    let mut scores = vec![0.0; tokens];
                    for tk in 0..tokens {
                        let (score, active) = attention_score(
                            &q,
                            &k,
                            query_row,
                            b * tokens + tk,
                            channels,
                            h * head_dim,
                            head_dim,
                            scale,
                        );
                        score_active[dense_row_offset + tk] = active;
                        scores[tk] = score;
                    }
                    let probs = if let Some((block, topk)) = sparse_blocks {
                        sparse_softmax_row(&scores, tq, block, topk, 10.0)?
                    } else {
                        vit_dense_softmax(&scores)
                    };
                    probabilities[dense_row_offset..dense_row_offset + tokens]
                        .copy_from_slice(&probs);
                    for d in 0..head_dim {
                        let c = h * head_dim + d;
                        let mut context = 0.0;
                        for tk in 0..tokens {
                            context += probs[tk] * v[(b * tokens + tk) * channels + c];
                        }
                        output[query_row * channels + c] = context;
                    }
                }
            }
        }
    }
    let cache = AttentionCache {
        batch,
        tokens,
        channels,
        heads,
        head_dim,
        q,
        k,
        v,
        q_active,
        k_active,
        probabilities,
        score_active,
        flash_lse: if flash { Some(flash_lse) } else { None },
        vc_heads: None,
    };
    Ok((output, cache))
}

/// Exact analytical input derivative of dense or online attention, returned in
/// fused `[B,T,3N]` QKV order. The sparse block-selection decision is fixed for
/// the local derivative; unselected probabilities have zero gradient.
pub fn attention_backward(output_grad: &[f32], cache: &AttentionCache) -> Result<Vec<f32>, String> {
    if let Some(vc_heads) = &cache.vc_heads {
        let expected = cache
            .batch
            .checked_mul(cache.tokens)
            .and_then(|rows| rows.checked_mul(cache.channels))
            .ok_or("vit VC output-gradient size overflow")?;
        if output_grad.len() != expected {
            return Err(format!(
                "vit VC output gradient has {} values, expected {expected}",
                output_grad.len()
            ));
        }
        let head_count = cache
            .batch
            .checked_mul(cache.heads)
            .ok_or("vit VC batch-head count overflow")?;
        if vc_heads.len() != head_count {
            return Err("vit VC head-cache count mismatch".into());
        }
        let fused_len = expected
            .checked_mul(3)
            .ok_or("vit VC QKV gradient size overflow")?;
        let mut fused_grad = vec![0.0; fused_len];
        for b in 0..cache.batch {
            for h in 0..cache.heads {
                let head_index = b * cache.heads + h;
                let mut head_output_grad = vec![0.0; cache.tokens * cache.head_dim];
                for token in 0..cache.tokens {
                    for d in 0..cache.head_dim {
                        let channel = h * cache.head_dim + d;
                        head_output_grad[token * cache.head_dim + d] =
                            output_grad[(b * cache.tokens + token) * cache.channels + channel];
                    }
                }
                let (dq, dk, dv) = crate::vc_attention::vc_head_backward(
                    &head_output_grad,
                    &vc_heads[head_index],
                )?;
                for token in 0..cache.tokens {
                    for d in 0..cache.head_dim {
                        let channel = h * cache.head_dim + d;
                        let destination = (b * cache.tokens + token) * 3 * cache.channels;
                        let source = token * cache.head_dim + d;
                        fused_grad[destination + channel] = dq[source];
                        fused_grad[destination + cache.channels + channel] = dk[source];
                        fused_grad[destination + 2 * cache.channels + channel] = dv[source];
                    }
                }
            }
        }
        return Ok(fused_grad);
    }
    let AttentionCache {
        batch,
        tokens,
        channels,
        heads,
        head_dim,
        q,
        k,
        v,
        q_active,
        k_active,
        probabilities,
        score_active,
        flash_lse,
        vc_heads: _,
    } = cache;
    let expected = batch * tokens * channels;
    if output_grad.len() != expected {
        return Err(format!(
            "vit attention output gradient has {} values, expected {expected}",
            output_grad.len()
        ));
    }
    let mut dq = vec![0.0; expected];
    let mut dk = vec![0.0; expected];
    let mut dv = vec![0.0; expected];
    let scale = 1.0 / (*head_dim as f32).sqrt();
    for b in 0..*batch {
        for h in 0..*heads {
            let head_row = b * *heads + h;
            let dense_head_offset = if flash_lse.is_some() {
                0
            } else {
                head_row * *tokens * *tokens
            };
            for tq in 0..*tokens {
                let dense_row_offset = if flash_lse.is_some() {
                    0
                } else {
                    dense_head_offset + tq * *tokens
                };
                let query_row = b * *tokens + tq;
                let mut dprob = vec![0.0; *tokens];
                let mut row_probabilities = vec![0.0; *tokens];
                let mut row_score_active = vec![false; *tokens];
                for tk in 0..*tokens {
                    let (p, active) = if let Some(lse) = flash_lse {
                        let (score, active) = attention_score(
                            q,
                            k,
                            query_row,
                            b * *tokens + tk,
                            *channels,
                            h * *head_dim,
                            *head_dim,
                            scale,
                        );
                        ((score - lse[head_row * *tokens + tq]).exp(), active)
                    } else {
                        (
                            probabilities[dense_row_offset + tk],
                            score_active[dense_row_offset + tk],
                        )
                    };
                    row_probabilities[tk] = p;
                    row_score_active[tk] = active;
                    for d in 0..*head_dim {
                        let c = h * *head_dim + d;
                        let dy = output_grad[query_row * *channels + c];
                        let vi = (b * *tokens + tk) * *channels + c;
                        dprob[tk] += dy * v[vi];
                        dv[vi] += p * dy;
                    }
                }
                let dot: f32 = dprob
                    .iter()
                    .zip(&row_probabilities)
                    .map(|(g, p)| g * p)
                    .sum();
                for tk in 0..*tokens {
                    let dscore = row_probabilities[tk] * (dprob[tk] - dot);
                    if !row_score_active[tk] || dscore == 0.0 {
                        continue;
                    }
                    for d in 0..*head_dim {
                        let c = h * *head_dim + d;
                        let qi = query_row * *channels + c;
                        let ki = (b * *tokens + tk) * *channels + c;
                        dq[qi] += dscore * k[ki] * scale;
                        dk[ki] += dscore * q[qi] * scale;
                    }
                }
            }
        }
    }
    let mut dqkv = vec![0.0; expected * 3];
    for i in 0..expected {
        dqkv[(i / *channels) * (3 * *channels) + i % *channels] =
            if q_active[i] { dq[i] } else { 0.0 };
        dqkv[(i / *channels) * (3 * *channels) + *channels + i % *channels] =
            if k_active[i] { dk[i] } else { 0.0 };
        dqkv[(i / *channels) * (3 * *channels) + 2 * *channels + i % *channels] = dv[i];
    }
    Ok(dqkv)
}

/// Default mHC residual parameters, two six-scalar sites; matches
/// `init_mhc_residual_params` in the C++ constructor.
pub const MHC_PARAM_COUNT: usize = 12;
pub const MHC_DEFAULTS: [f32; MHC_PARAM_COUNT] = [
    4.0, -4.0, -4.0, 4.0, 0.0, -8.0, 4.0, -4.0, -4.0, 4.0, 0.0, -8.0,
];

fn mhc_coefficients(params: &[f32], site: usize) -> Result<(f32, f32, f32, f32, f32), String> {
    if params.len() != MHC_PARAM_COUNT || site >= 2 {
        return Err(format!(
            "vit mHC expects 12 parameters and site 0 or 1 (got {}, {site})",
            params.len()
        ));
    }
    let o = site * 6;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let p = sigmoid(0.5 * (params[o] + params[o + 3] - params[o + 1] - params[o + 2]));
    let post_skip = 2.0 * sigmoid(params[o + 4]);
    let post_branch = 2.0 * sigmoid(params[o + 5]);
    let skip = post_skip * p + post_branch * (1.0 - p);
    let branch = post_skip * (1.0 - p) + post_branch * p;
    Ok((skip, branch, p, post_skip, post_branch))
}

fn clamp_feature(x: f32, clip: f32) -> f32 {
    if x.is_finite() {
        x.clamp(-clip, clip)
    } else {
        0.0
    }
}

/// The two-stream, Birkhoff-constrained residual mix used at both ViT residual
/// sites. `out[i] = clamp(a*clamp(skip[i]) + b*clamp(branch[i]))`.
pub fn mhc_residual_forward(
    skip: &[f32],
    branch: &[f32],
    params: &[f32],
    site: usize,
    feature_clip: f32,
) -> Result<Vec<f32>, String> {
    if skip.len() != branch.len() || !feature_clip.is_finite() || feature_clip <= 0.0 {
        return Err("vit mHC residual requires equal vectors and positive finite clip".into());
    }
    let (a, b, _, _, _) = mhc_coefficients(params, site)?;
    Ok(skip
        .iter()
        .zip(branch)
        .map(|(&s, &br)| {
            clamp_feature(
                a * clamp_feature(s, feature_clip) + b * clamp_feature(br, feature_clip),
                feature_clip,
            )
        })
        .collect())
}

/// Exact analytical derivative of [`mhc_residual_forward`], returning
/// `(d_skip, d_branch, d_params[12])`. Derivatives of all three finite clamps
/// are included, unlike the C++ heuristic gradient path.
pub fn mhc_residual_backward(
    skip: &[f32],
    branch: &[f32],
    output_grad: &[f32],
    params: &[f32],
    site: usize,
    feature_clip: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if skip.len() != branch.len()
        || skip.len() != output_grad.len()
        || !feature_clip.is_finite()
        || feature_clip <= 0.0
    {
        return Err("vit mHC backward requires equal vectors and positive finite clip".into());
    }
    let (a, b, p, post_skip, post_branch) = mhc_coefficients(params, site)?;
    let mut ds = vec![0.0; skip.len()];
    let mut db = vec![0.0; skip.len()];
    let mut dp = vec![0.0; MHC_PARAM_COUNT];
    let mut grad_post_skip = 0.0;
    let mut grad_post_branch = 0.0;
    let mut grad_p = 0.0;
    let o = site * 6;
    for i in 0..skip.len() {
        let raw_s = skip[i];
        let raw_b = branch[i];
        let s = clamp_feature(raw_s, feature_clip);
        let br = clamp_feature(raw_b, feature_clip);
        let mixed = a * s + b * br;
        if !mixed.is_finite() || mixed.abs() >= feature_clip {
            continue;
        }
        let g = output_grad[i];
        if !g.is_finite() {
            continue;
        }
        let gs = g * a;
        let gb = g * b;
        ds[i] = if raw_s.is_finite() && raw_s.abs() < feature_clip {
            gs
        } else {
            0.0
        };
        db[i] = if raw_b.is_finite() && raw_b.abs() < feature_clip {
            gb
        } else {
            0.0
        };
        grad_post_skip += g * (p * s + (1.0 - p) * br);
        grad_post_branch += g * ((1.0 - p) * s + p * br);
        grad_p += g * (post_skip - post_branch) * (s - br);
    }
    let dz = 0.5 * grad_p * p * (1.0 - p);
    dp[o] = dz;
    dp[o + 1] = -dz;
    dp[o + 2] = -dz;
    dp[o + 3] = dz;
    dp[o + 4] = grad_post_skip * post_skip * (1.0 - 0.5 * post_skip);
    dp[o + 5] = grad_post_branch * post_branch * (1.0 - 0.5 * post_branch);
    Ok((ds, db, dp))
}

/// C++ `minimax_sparse_softmax_row_cpu`: sanitize/clamp scores, retain the
/// query's local block plus the `topk_blocks-1` blocks with largest maxima,
/// and return a probability row. Ties select the lower block index.
pub fn sparse_softmax_row(
    scores: &[f32],
    query_token: usize,
    block_size: usize,
    topk_blocks: usize,
    score_clip: f32,
) -> Result<Vec<f32>, String> {
    let tokens = scores.len();
    if tokens == 0 || block_size == 0 || topk_blocks == 0 {
        return Err("vit sparse softmax requires nonempty scores and positive block/topk".into());
    }
    if query_token >= tokens {
        return Err(format!(
            "vit query token {query_token} is outside {tokens} tokens"
        ));
    }
    let clip = if score_clip > 0.0 { score_clip } else { 80.0 };
    let values: Vec<f32> = scores
        .iter()
        .map(|&s| {
            if s.is_nan() {
                0.0
            } else if s.is_infinite() {
                if s.is_sign_positive() { clip } else { -clip }
            } else if score_clip > 0.0 {
                s.clamp(-score_clip, score_clip)
            } else {
                s
            }
        })
        .collect();
    let blocks = tokens.div_ceil(block_size);
    let topk = topk_blocks.min(blocks);
    if topk == blocks {
        return Ok(softmax(&values));
    }
    let mut maxima = vec![f32::NEG_INFINITY; blocks];
    for (i, &v) in values.iter().enumerate() {
        maxima[i / block_size] = maxima[i / block_size].max(v);
    }
    let local = (query_token / block_size).min(blocks - 1);
    let mut selected = vec![false; blocks];
    selected[local] = true;
    for _ in 1..topk {
        let mut best: Option<usize> = None;
        for block in 0..blocks {
            if !selected[block] && best.is_none_or(|current| maxima[block] > maxima[current]) {
                best = Some(block);
            }
        }
        if let Some(block) = best {
            selected[block] = true;
        }
    }
    let selected_max = values
        .iter()
        .enumerate()
        .filter(|(i, _)| selected[i / block_size])
        .map(|(_, &v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0; tokens];
    let mut sum = 0.0;
    for i in 0..tokens {
        if selected[i / block_size] {
            probs[i] = (values[i] - selected_max).exp();
            sum += probs[i];
        }
    }
    if !sum.is_finite() || sum <= 0.0 {
        let start = local * block_size;
        let end = (start + block_size).min(tokens);
        probs.fill(0.0);
        let uniform = 1.0 / (end - start) as f32;
        probs[start..end].fill(uniform);
    } else {
        for p in &mut probs {
            *p /= sum;
        }
    }
    Ok(probs)
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        out.fill(1.0 / x.len() as f32);
    } else {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

/// The dense ViT helper uses a small denominator epsilon after max-subtracted
/// exponentiation; the MSA helper intentionally does not.
fn vit_dense_softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = out.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        out.fill(1.0 / x.len() as f32);
    } else {
        let inv = 1.0 / (sum + 1.0e-9);
        out.iter_mut().for_each(|v| *v *= inv);
    }
    out
}

/// Trainable ViT parameters in the exact `save_vit_weights` block order.
#[derive(Clone, Debug, PartialEq)]
pub struct VitWeights {
    pub patch_embed: Vec<f32>,
    pub patch_bias: Vec<f32>,
    pub qkv_weight: Vec<f32>,
    pub qkv_bias: Vec<f32>,
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub wo: Vec<f32>,
    pub wo_bias: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    pub ffn_w1: Vec<f32>,
    pub ffn_b1: Vec<f32>,
    pub ffn_w2: Vec<f32>,
    pub ffn_b2: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub mhc: Vec<f32>,
}

impl VitWeights {
    /// Construct shape-correct defaults. Projection tensors are zero-filled;
    /// mHC starts from the C++ identity-like logits, and sinusoidal positions
    /// are generated rather than serialized state when selected.
    pub fn zeroed(g: &VitGeometry, out_h: usize, out_w: usize) -> Result<Self, String> {
        let sizes = vit_weight_sizes(g, out_h, out_w)?;
        let mut parts = sizes[..15].iter().map(|&n| vec![0.0; n]);
        let patch_embed = parts.next().unwrap();
        let patch_bias = parts.next().unwrap();
        let qkv_bias = parts.next().unwrap();
        let mut ln1_gamma = parts.next().unwrap();
        ln1_gamma.fill(1.0);
        let ln1_beta = parts.next().unwrap();
        let qkv_weight = parts.next().unwrap();
        let wo = parts.next().unwrap();
        let wo_bias = parts.next().unwrap();
        let mut ln2_gamma = parts.next().unwrap();
        ln2_gamma.fill(1.0);
        let ln2_beta = parts.next().unwrap();
        let ffn_w1 = parts.next().unwrap();
        let ffn_b1 = parts.next().unwrap();
        let ffn_w2 = parts.next().unwrap();
        let ffn_b2 = parts.next().unwrap();
        let mut pos_embed = parts.next().unwrap();
        if !g.learned_pos_embed {
            pos_embed = sinusoidal_pos_embed(out_h, out_w, g.hidden);
        }
        Ok(Self {
            patch_embed,
            patch_bias,
            qkv_weight,
            qkv_bias,
            ln1_gamma,
            ln1_beta,
            wo,
            wo_bias,
            ln2_gamma,
            ln2_beta,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
            pos_embed,
            mhc: MHC_DEFAULTS.to_vec(),
        })
    }

    /// Serialize in the exact C++ `save_vit_weights` order. Modern files always
    /// include the 12-float mHC suffix.
    pub fn to_blob(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for part in [
            &self.patch_embed,
            &self.patch_bias,
            &self.qkv_bias,
            &self.ln1_gamma,
            &self.ln1_beta,
            &self.qkv_weight,
            &self.wo,
            &self.wo_bias,
            &self.ln2_gamma,
            &self.ln2_beta,
            &self.ffn_w1,
            &self.ffn_b1,
            &self.ffn_w2,
            &self.ffn_b2,
            &self.pos_embed,
            &self.mhc,
        ] {
            out.extend_from_slice(part);
        }
        out
    }

    /// Parse C++ blob order. `has_mhc_scales` must come from the file header
    /// version comparison used by C++ (old headers omit exactly 12 floats).
    pub fn from_blob(
        blob: &[f32],
        g: &VitGeometry,
        out_h: usize,
        out_w: usize,
        has_mhc_scales: bool,
    ) -> Result<Self, String> {
        let sizes = vit_weight_sizes(g, out_h, out_w)?;
        let expected: usize = sizes[..15].iter().try_fold(0usize, |n, &v| {
            n.checked_add(v).ok_or("vit blob size overflow")
        })?;
        let expected = if has_mhc_scales {
            expected
                .checked_add(MHC_PARAM_COUNT)
                .ok_or("vit blob size overflow")?
        } else {
            expected
        };
        if blob.len() != expected {
            return Err(format!(
                "vit blob has {} floats, expected {expected}",
                blob.len()
            ));
        }
        let mut at = 0;
        let mut take = |n: usize| {
            let part = blob[at..at + n].to_vec();
            at += n;
            part
        };
        let patch_embed = take(sizes[0]);
        let patch_bias = take(sizes[1]);
        let qkv_bias = take(sizes[2]);
        let ln1_gamma = take(sizes[3]);
        let ln1_beta = take(sizes[4]);
        let qkv_weight = take(sizes[5]);
        let wo = take(sizes[6]);
        let wo_bias = take(sizes[7]);
        let ln2_gamma = take(sizes[8]);
        let ln2_beta = take(sizes[9]);
        let ffn_w1 = take(sizes[10]);
        let ffn_b1 = take(sizes[11]);
        let ffn_w2 = take(sizes[12]);
        let ffn_b2 = take(sizes[13]);
        let saved_pos_embed = take(sizes[14]);
        let pos_embed = if g.learned_pos_embed {
            saved_pos_embed
        } else {
            sinusoidal_pos_embed(out_h, out_w, g.hidden)
        };
        let mhc = if has_mhc_scales {
            take(MHC_PARAM_COUNT)
        } else {
            MHC_DEFAULTS.to_vec()
        };
        Ok(Self {
            patch_embed,
            patch_bias,
            qkv_weight,
            qkv_bias,
            ln1_gamma,
            ln1_beta,
            wo,
            wo_bias,
            ln2_gamma,
            ln2_beta,
            ffn_w1,
            ffn_b1,
            ffn_w2,
            ffn_b2,
            pos_embed,
            mhc,
        })
    }
}

/// Parameter lengths ordered as `save_vit_weights` (last item is mHC scales).
fn vit_weight_sizes(g: &VitGeometry, out_h: usize, out_w: usize) -> Result<[usize; 16], String> {
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or("vit parameter count overflow");
    let patch_dim = mul(mul(g.patch_size, g.patch_size)?, g.channels)?;
    let t = mul(out_h, out_w)?;
    Ok([
        mul(g.hidden, patch_dim)?,
        g.hidden,
        mul(3, g.hidden)?,
        g.hidden,
        g.hidden,
        mul(3, mul(g.hidden, g.hidden)?)?,
        mul(g.hidden, g.hidden)?,
        g.hidden,
        g.hidden,
        g.hidden,
        mul(g.mlp_dim, g.hidden)?,
        g.mlp_dim,
        mul(g.hidden, g.mlp_dim)?,
        g.hidden,
        mul(t, g.hidden)?,
        MHC_PARAM_COUNT,
    ])
}

pub fn vit_blob_float_count(
    g: &VitGeometry,
    out_h: usize,
    out_w: usize,
    has_mhc_scales: bool,
) -> Result<usize, String> {
    let sizes = vit_weight_sizes(g, out_h, out_w)?;
    let base = sizes[..15].iter().try_fold(0usize, |n, &v| {
        n.checked_add(v).ok_or("vit blob size overflow")
    })?;
    if has_mhc_scales {
        base.checked_add(MHC_PARAM_COUNT)
            .ok_or_else(|| "vit blob size overflow".into())
    } else {
        Ok(base)
    }
}

/// Geometry and execution choices for one global ViT block.
#[derive(Clone, Debug, PartialEq)]
pub struct VitGeometry {
    pub batch: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub patch_size: usize,
    pub patch_stride: usize,
    pub patch_pad: usize,
    pub hidden: usize,
    pub heads: usize,
    pub mlp_dim: usize,
    pub ln_eps: f32,
    pub activation: Activation,
    pub learned_pos_embed: bool,
    pub pos_init_zero: bool,
    pub msa: Option<(usize, usize)>,
    pub flash_attention: bool,
    pub vc_attention: bool,
    pub vc_expcast: u8,
    pub vc_perm: crate::model::VitVcPermutation,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VitForwardCache {
    geometry: VitGeometry,
    patches: Vec<f32>,
    tokens: Vec<f32>,
    ln1: LayerNormCache,
    ln1_output: Vec<f32>,
    attention: AttentionCache,
    context: Vec<f32>,
    projection: Vec<f32>,
    ln2: LayerNormCache,
    ln2_output: Vec<f32>,
    ffn1_raw: Vec<f32>,
    ffn1_clamped: Vec<f32>,
    ffn1_activated: Vec<f32>,
    ffn2: Vec<f32>,
    res1: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VitGradients {
    pub input: Vec<f32>,
    pub patch_embed: Vec<f32>,
    pub patch_bias: Vec<f32>,
    pub qkv_weight: Vec<f32>,
    pub qkv_bias: Vec<f32>,
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub wo: Vec<f32>,
    pub wo_bias: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    pub ffn_w1: Vec<f32>,
    pub ffn_b1: Vec<f32>,
    pub ffn_w2: Vec<f32>,
    pub ffn_b2: Vec<f32>,
    pub pos_embed: Vec<f32>,
    pub mhc: Vec<f32>,
}

pub(crate) fn tensor_activation(activation: &Activation, x: f32) -> Result<f32, String> {
    use Activation::*;
    Ok(match activation {
        Linear => x,
        Logistic => 1.0 / (1.0 + (-x).exp()),
        Loggy => 2.0 / (1.0 + (-x).exp()) - 1.0,
        Relu => x.max(0.0),
        Leaky | Revleaky => {
            if x > 0.0 {
                x
            } else {
                0.1 * x
            }
        }
        Mish => x * (1.0 + x.exp()).ln().tanh(),
        Swish => x / (1.0 + (-x).exp()),
        Tanh => x.tanh(),
        Relu6 => x.clamp(0.0, 6.0),
        Ramp => x * (x > 0.0) as i32 as f32 + 0.1 * x,
        Gelu => crate::activation_math::gelu(x),
        Elu => crate::activation_math::elu(x),
        Selu => crate::activation_math::selu(x),
        Relie => crate::activation_math::relie(x),
        Plse => crate::activation_math::plse(x),
        Hardtan => crate::activation_math::hardtan(x),
        Lhtan => crate::activation_math::lhtan(x),
        Stair => crate::activation_math::stair(x),
        HardMish => crate::activation_math::hard_mish(x),
        NormalizeChannels | NormalizeChannelsSoftmax | NormalizeChannelsSoftmaxMaxval => {
            return Err(
                "channel-normalizing activations are only supported on convolutional layers"
                    .to_owned(),
            );
        }
        Other(name) => return Err(format!("unsupported [vit] activation `{name}`")),
    })
}

pub(crate) fn tensor_activation_grad(
    activation: &Activation,
    x: f32,
    y: f32,
) -> Result<f32, String> {
    use Activation::*;
    Ok(match activation {
        Linear => 1.0,
        Logistic => y * (1.0 - y),
        Loggy => 0.5 * (1.0 - y * y),
        Relu => (x > 0.0) as i32 as f32,
        Leaky | Revleaky => {
            if x > 0.0 {
                1.0
            } else {
                0.1
            }
        }
        Mish => {
            let softplus = if x > 20.0 {
                x
            } else if x < -20.0 {
                x.exp()
            } else {
                (1.0 + x.exp()).ln()
            };
            let grad_sp = 1.0 - (-softplus).exp();
            let tsp = softplus.tanh();
            x * (1.0 - tsp * tsp) * grad_sp + tsp
        }
        Swish => {
            let sig = 1.0 / (1.0 + (-x).exp());
            sig + x * sig * (1.0 - sig)
        }
        Tanh => 1.0 - y * y,
        Relu6 => (x > 0.0 && x < 6.0) as i32 as f32,
        Ramp => (x > 0.0) as i32 as f32 + 0.1,
        Gelu => crate::activation_math::gelu_grad(x),
        Elu => crate::activation_math::elu_grad(x),
        Selu => crate::activation_math::selu_grad(x),
        Relie => {
            if x > 0.0 {
                1.0
            } else {
                0.01
            }
        }
        Plse => crate::activation_math::plse_grad(x),
        Hardtan => crate::activation_math::hardtan_grad(x),
        Lhtan => crate::activation_math::lhtan_grad(x),
        Stair => crate::activation_math::stair_grad(x),
        HardMish => crate::activation_math::hard_mish_grad(x),
        NormalizeChannels | NormalizeChannelsSoftmax | NormalizeChannelsSoftmaxMaxval => {
            return Err(
                "channel-normalizing activations are only supported on convolutional layers"
                    .to_owned(),
            );
        }
        Other(name) => return Err(format!("unsupported [vit] activation gradient `{name}`")),
    })
}

/// Validate parameter lengths against the cfg geometry.
fn validate_vit_weights(
    w: &VitWeights,
    g: &VitGeometry,
    out_h: usize,
    out_w: usize,
) -> Result<(), String> {
    let sizes = vit_weight_sizes(g, out_h, out_w)?;
    let lengths = [
        ("patch_embed", w.patch_embed.len(), sizes[0]),
        ("patch_bias", w.patch_bias.len(), sizes[1]),
        ("qkv_bias", w.qkv_bias.len(), sizes[2]),
        ("ln1_gamma", w.ln1_gamma.len(), sizes[3]),
        ("ln1_beta", w.ln1_beta.len(), sizes[4]),
        ("qkv_weight", w.qkv_weight.len(), sizes[5]),
        ("wo", w.wo.len(), sizes[6]),
        ("wo_bias", w.wo_bias.len(), sizes[7]),
        ("ln2_gamma", w.ln2_gamma.len(), sizes[8]),
        ("ln2_beta", w.ln2_beta.len(), sizes[9]),
        ("ffn_w1", w.ffn_w1.len(), sizes[10]),
        ("ffn_b1", w.ffn_b1.len(), sizes[11]),
        ("ffn_w2", w.ffn_w2.len(), sizes[12]),
        ("ffn_b2", w.ffn_b2.len(), sizes[13]),
        ("pos_embed", w.pos_embed.len(), sizes[14]),
        ("mhc", w.mhc.len(), sizes[15]),
    ];
    for (name, got, expected) in lengths {
        if got != expected {
            return Err(format!("vit {name} has {got} values, expected {expected}"));
        }
    }
    Ok(())
}

/// Pure ViT forward: patch projection, positions, pre-norm MHSA, mHC residuals,
/// GELU/selected activation FFN, and NCHW output.
pub fn vit_forward(
    input: &[f32],
    weights: &VitWeights,
    g: &VitGeometry,
) -> Result<(Vec<f32>, VitForwardCache), String> {
    vit_forward_with_flash_attention(input, weights, g, attention_forward_flash)
}

/// ViT forward with an injectable standard-flash implementation. The callback
/// is used only when `g.flash_attention` is true; normal CPU callers use
/// [`attention_forward_flash`], while runtime backends may provide native
/// context/LSE and build the same exact analytical-backward cache.
pub fn vit_forward_with_flash_attention<F>(
    input: &[f32],
    weights: &VitWeights,
    g: &VitGeometry,
    mut flash_attention: F,
) -> Result<(Vec<f32>, VitForwardCache), String>
where
    F: FnMut(&[f32], usize, usize, usize, usize) -> Result<(Vec<f32>, AttentionCache), String>,
{
    let (out_h, out_w, patches) = patchify_nchw(
        input,
        g.batch,
        g.channels,
        g.height,
        g.width,
        g.patch_size,
        g.patch_stride,
        g.patch_pad,
    )?;
    validate_vit_weights(weights, g, out_h, out_w)?;
    if g.hidden == 0 || g.mlp_dim == 0 || g.heads == 0 || g.hidden % g.heads != 0 {
        return Err("vit requires positive hidden/mlp/heads and hidden divisible by heads".into());
    }
    if !g.ln_eps.is_finite() || g.ln_eps <= 0.0 {
        return Err("vit ln_eps must be positive and finite".into());
    }
    if g.flash_attention && g.msa.is_some() {
        return Err("vit flash attention and MSA are mutually exclusive".into());
    }
    if g.vc_attention && !g.flash_attention {
        return Err("vit VC attention requires flash attention".into());
    }
    let tokens_per_sample = out_h * out_w;
    let rows = g.batch * tokens_per_sample;
    let mut tokens = linear_forward(
        &patches,
        &weights.patch_embed,
        &weights.patch_bias,
        rows,
        g.patch_size * g.patch_size * g.channels,
        g.hidden,
    )?;
    for b in 0..g.batch {
        for t in 0..tokens_per_sample {
            for c in 0..g.hidden {
                tokens[(b * tokens_per_sample + t) * g.hidden + c] +=
                    weights.pos_embed[t * g.hidden + c];
            }
        }
    }
    let (ln1_output, ln1) = layer_norm_forward(
        &tokens,
        &weights.ln1_gamma,
        &weights.ln1_beta,
        rows,
        g.hidden,
        g.ln_eps,
    )?;
    let qkv = linear_forward(
        &ln1_output,
        &weights.qkv_weight,
        &weights.qkv_bias,
        rows,
        g.hidden,
        3 * g.hidden,
    )?;
    let (context, attention) = if g.flash_attention && g.vc_attention {
        let mode = crate::vc_attention::ExpCastMode::try_from(g.vc_expcast)?;
        let permutation = match g.vc_perm {
            crate::model::VitVcPermutation::Sequence => None,
            crate::model::VitVcPermutation::Spatial8 => {
                Some(crate::vc_attention::spatial8_permutation(out_h, out_w)?)
            }
        };
        attention_forward_vc(
            &qkv,
            g.batch,
            tokens_per_sample,
            g.hidden,
            g.heads,
            mode,
            permutation.as_deref(),
        )?
    } else if g.flash_attention {
        flash_attention(&qkv, g.batch, tokens_per_sample, g.hidden, g.heads)?
    } else {
        attention_forward(&qkv, g.batch, tokens_per_sample, g.hidden, g.heads, g.msa)?
    };
    let projection = linear_forward(
        &context,
        &weights.wo,
        &weights.wo_bias,
        rows,
        g.hidden,
        g.hidden,
    )?;
    let res1 = mhc_residual_forward(&tokens, &projection, &weights.mhc, 0, 10.0)?;
    let (ln2_output, ln2) = layer_norm_forward(
        &res1,
        &weights.ln2_gamma,
        &weights.ln2_beta,
        rows,
        g.hidden,
        g.ln_eps,
    )?;
    let ffn1_raw = linear_forward(
        &ln2_output,
        &weights.ffn_w1,
        &weights.ffn_b1,
        rows,
        g.hidden,
        g.mlp_dim,
    )?;
    let ffn1_clamped: Vec<f32> = ffn1_raw.iter().map(|&v| clamp_feature(v, 10.0)).collect();
    let mut ffn1_activated = Vec::with_capacity(ffn1_clamped.len());
    for &v in &ffn1_clamped {
        ffn1_activated.push(clamp_feature(tensor_activation(&g.activation, v)?, 10.0));
    }
    let ffn2 = linear_forward(
        &ffn1_activated,
        &weights.ffn_w2,
        &weights.ffn_b2,
        rows,
        g.mlp_dim,
        g.hidden,
    )?;
    let mixed = mhc_residual_forward(&res1, &ffn2, &weights.mhc, 1, 10.0)?;
    let mut output = vec![0.0; g.batch * g.hidden * tokens_per_sample];
    for b in 0..g.batch {
        for t in 0..tokens_per_sample {
            for c in 0..g.hidden {
                output[(b * g.hidden + c) * tokens_per_sample + t] =
                    mixed[(b * tokens_per_sample + t) * g.hidden + c];
            }
        }
    }
    let cache = VitForwardCache {
        geometry: g.clone(),
        patches,
        tokens,
        ln1,
        ln1_output,
        attention,
        context,
        projection,
        ln2,
        ln2_output,
        ffn1_raw,
        ffn1_clamped,
        ffn1_activated,
        ffn2,
        res1,
    };
    Ok((output, cache))
}

/// Pure exact backward for [`vit_forward`]. The seed uses the same NCHW layout
/// as the output; returned gradients include input and every trainable tensor.
pub fn vit_backward(
    output_grad: &[f32],
    weights: &VitWeights,
    cache: &VitForwardCache,
) -> Result<VitGradients, String> {
    vit_backward_with_attention(output_grad, weights, cache, |d_context, _, attention| {
        attention_backward(d_context, attention)
    })
}

/// ViT backward with an injectable attention derivative. This permits a GPU
/// backend to compute dQ/dK/dV while the remaining transformer derivative stays
/// on the exact Rust host path.
pub fn vit_backward_with_attention<F>(
    output_grad: &[f32],
    weights: &VitWeights,
    cache: &VitForwardCache,
    mut attention_backward_fn: F,
) -> Result<VitGradients, String>
where
    F: FnMut(&[f32], &[f32], &AttentionCache) -> Result<Vec<f32>, String>,
{
    let g = &cache.geometry;
    let t = cache.attention.tokens;
    let rows = g.batch * t;
    let output_len = g.batch * g.hidden * t;
    if output_grad.len() != output_len {
        return Err(format!(
            "vit output gradient has {}, expected {output_len}",
            output_grad.len()
        ));
    }
    let mut d_ntc = vec![0.0; output_len];
    for b in 0..g.batch {
        for ti in 0..t {
            for c in 0..g.hidden {
                d_ntc[(b * t + ti) * g.hidden + c] = output_grad[(b * g.hidden + c) * t + ti];
            }
        }
    }
    let (d_res1, d_ffn2, d_mhc_1) =
        mhc_residual_backward(&cache.res1, &cache.ffn2, &d_ntc, &weights.mhc, 1, 10.0)?;
    let (d_ffn1_act, ffn_w2, ffn_b2) = linear_backward(
        &cache.ffn1_activated,
        &weights.ffn_w2,
        &d_ffn2,
        rows,
        g.mlp_dim,
        g.hidden,
    )?;
    let mut d_ffn1_clamped = vec![0.0; d_ffn1_act.len()];
    for i in 0..d_ffn1_act.len() {
        let act_pre = cache.ffn1_clamped[i];
        let act = tensor_activation(&g.activation, act_pre)?;
        let unclamped = tensor_activation(&g.activation, act_pre)?;
        let d_post_clamp = if unclamped.is_finite() && unclamped.abs() < 10.0 {
            d_ffn1_act[i]
        } else {
            0.0
        };
        let d_activation = tensor_activation_grad(&g.activation, act_pre, act)?;
        let d_pre_clamp = if cache.ffn1_raw[i].is_finite() && cache.ffn1_raw[i].abs() < 10.0 {
            1.0
        } else {
            0.0
        };
        d_ffn1_clamped[i] = d_post_clamp * d_activation * d_pre_clamp;
    }
    let (d_ln2_output, ffn_w1, ffn_b1) = linear_backward(
        &cache.ln2_output,
        &weights.ffn_w1,
        &d_ffn1_clamped,
        rows,
        g.hidden,
        g.mlp_dim,
    )?;
    let (d_ln2_input, ln2_gamma, ln2_beta) =
        layer_norm_backward(&d_ln2_output, &weights.ln2_gamma, &cache.ln2)?;
    let d_res1_total: Vec<f32> = d_ln2_input.iter().zip(d_res1).map(|(a, b)| a + b).collect();
    let (d_tokens_skip, d_projection, mut d_mhc_0) = mhc_residual_backward(
        &cache.tokens,
        &cache.projection,
        &d_res1_total,
        &weights.mhc,
        0,
        10.0,
    )?;
    let (d_context, wo, wo_bias) = linear_backward(
        &cache.context,
        &weights.wo,
        &d_projection,
        rows,
        g.hidden,
        g.hidden,
    )?;
    let d_qkv = attention_backward_fn(&d_context, &cache.context, &cache.attention)?;
    let (d_ln1_output, qkv_weight, qkv_bias) = linear_backward(
        &cache.ln1_output,
        &weights.qkv_weight,
        &d_qkv,
        rows,
        g.hidden,
        3 * g.hidden,
    )?;
    let (d_tokens_norm, ln1_gamma, ln1_beta) =
        layer_norm_backward(&d_ln1_output, &weights.ln1_gamma, &cache.ln1)?;
    let d_tokens: Vec<f32> = d_tokens_norm
        .iter()
        .zip(d_tokens_skip)
        .map(|(a, b)| a + b)
        .collect();
    let (d_patches, patch_embed, patch_bias) = linear_backward(
        &cache.patches,
        &weights.patch_embed,
        &d_tokens,
        rows,
        g.patch_size * g.patch_size * g.channels,
        g.hidden,
    )?;
    let input = scatter_patch_gradients_nchw(
        &d_patches,
        g.batch,
        g.channels,
        g.height,
        g.width,
        g.patch_size,
        g.patch_stride,
        g.patch_pad,
        (g.height + 2 * g.patch_pad - g.patch_size) / g.patch_stride + 1,
        (g.width + 2 * g.patch_pad - g.patch_size) / g.patch_stride + 1,
    )?;
    let mut pos_embed = vec![0.0; t * g.hidden];
    if g.learned_pos_embed {
        for b in 0..g.batch {
            for ti in 0..t {
                for c in 0..g.hidden {
                    pos_embed[ti * g.hidden + c] += d_tokens[(b * t + ti) * g.hidden + c];
                }
            }
        }
    }
    d_mhc_0.iter_mut().zip(d_mhc_1).for_each(|(a, b)| *a += b);
    Ok(VitGradients {
        input,
        patch_embed,
        patch_bias,
        qkv_weight,
        qkv_bias,
        ln1_gamma,
        ln1_beta,
        wo,
        wo_bias,
        ln2_gamma,
        ln2_beta,
        ffn_w1,
        ffn_b1,
        ffn_w2,
        ffn_b2,
        pos_embed,
        mhc: d_mhc_0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vit_fixture() -> (VitWeights, VitGeometry, Vec<f32>, Vec<f32>) {
        let geometry = VitGeometry {
            batch: 1,
            channels: 1,
            height: 1,
            width: 2,
            patch_size: 1,
            patch_stride: 1,
            patch_pad: 0,
            hidden: 2,
            heads: 1,
            mlp_dim: 2,
            ln_eps: 1e-5,
            activation: Activation::Gelu,
            learned_pos_embed: true,
            pos_init_zero: false,
            msa: None,
            flash_attention: false,
            vc_attention: false,
            vc_expcast: 1,
            vc_perm: crate::model::VitVcPermutation::Sequence,
        };
        let weights = VitWeights {
            patch_embed: vec![0.7, -0.4],
            patch_bias: vec![0.1, -0.2],
            qkv_weight: (0..12).map(|i| (i as f32 - 5.0) * 0.035).collect(),
            qkv_bias: vec![0.03, -0.04, 0.02, 0.01, -0.03, 0.05],
            ln1_gamma: vec![1.1, 0.9],
            ln1_beta: vec![0.02, -0.01],
            wo: vec![0.3, -0.2, 0.15, 0.25],
            wo_bias: vec![0.01, -0.02],
            ln2_gamma: vec![0.95, 1.05],
            ln2_beta: vec![-0.03, 0.04],
            ffn_w1: vec![0.2, -0.1, 0.15, 0.3],
            ffn_b1: vec![0.02, -0.01],
            ffn_w2: vec![-0.25, 0.12, 0.18, 0.22],
            ffn_b2: vec![0.01, 0.03],
            pos_embed: vec![0.03, -0.02, 0.01, 0.04],
            mhc: vec![
                0.2, -0.1, 0.1, 0.3, -0.2, 0.1, -0.1, 0.15, -0.05, 0.2, -0.1, 0.05,
            ],
        };
        (
            weights,
            geometry,
            vec![0.2, -0.4],
            vec![0.7, -0.3, -0.2, 0.9],
        )
    }

    #[test]
    fn vit_blob_round_trips_modern_and_legacy_header_orders() {
        let (weights, geometry, _, _) = tiny_vit_fixture();
        let (out_h, out_w, _) = patchify_nchw(&[0.2, -0.4], 1, 1, 1, 2, 1, 1, 0).unwrap();
        let modern = weights.to_blob();
        assert_eq!(
            modern.len(),
            vit_blob_float_count(&geometry, out_h, out_w, true).unwrap()
        );
        assert_eq!(
            VitWeights::from_blob(&modern, &geometry, out_h, out_w, true).unwrap(),
            weights
        );
        let legacy_len = vit_blob_float_count(&geometry, out_h, out_w, false).unwrap();
        assert_eq!(legacy_len + MHC_PARAM_COUNT, modern.len());
        let legacy =
            VitWeights::from_blob(&modern[..legacy_len], &geometry, out_h, out_w, false).unwrap();
        assert_eq!(legacy.mhc, MHC_DEFAULTS);
        assert_eq!(legacy.to_blob()[..legacy_len], modern[..legacy_len]);
        assert!(
            VitWeights::from_blob(&modern[..modern.len() - 1], &geometry, out_h, out_w, true)
                .is_err()
        );
    }

    #[test]
    fn full_vit_flash_path_matches_input_finite_differences() {
        let (weights, mut geometry, input, seed) = tiny_vit_fixture();
        geometry.flash_attention = true;
        let (_, cache) = vit_forward(&input, &weights, &geometry).unwrap();
        assert!(cache.attention.probabilities.is_empty());
        assert!(cache.attention.score_active.is_empty());
        let grad = vit_backward(&seed, &weights, &cache).unwrap();
        let loss = |x: &[f32]| {
            vit_forward(x, &weights, &geometry)
                .unwrap()
                .0
                .iter()
                .zip(&seed)
                .map(|(y, dy)| y * dy)
                .sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..input.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            assert!(
                (grad.input[i] - fd).abs() < 2e-3,
                "flash input[{i}] {} != {fd}",
                grad.input[i]
            );
        }
    }

    #[test]
    fn full_vit_forward_backward_matches_finite_differences() {
        let (weights, geometry, input, seed) = tiny_vit_fixture();
        let (out, cache) = vit_forward(&input, &weights, &geometry).unwrap();
        assert_eq!(out.len(), 4);
        let grad = vit_backward(&seed, &weights, &cache).unwrap();
        let loss = |x: &[f32], w: &VitWeights| {
            vit_forward(x, w, &geometry)
                .unwrap()
                .0
                .iter()
                .zip(&seed)
                .map(|(y, d)| y * d)
                .sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..input.len() {
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&plus, &weights) - loss(&minus, &weights)) / (2.0 * eps);
            assert!(
                (grad.input[i] - fd).abs() < 2e-3,
                "input[{i}] {} != {fd}",
                grad.input[i]
            );
        }
        let checks: [(&str, &[f32], &[f32], fn(&mut VitWeights) -> &mut Vec<f32>); 8] = [
            (
                "patch_embed",
                &grad.patch_embed,
                &weights.patch_embed,
                |w| &mut w.patch_embed,
            ),
            ("qkv_weight", &grad.qkv_weight, &weights.qkv_weight, |w| {
                &mut w.qkv_weight
            }),
            ("ln1_gamma", &grad.ln1_gamma, &weights.ln1_gamma, |w| {
                &mut w.ln1_gamma
            }),
            ("wo", &grad.wo, &weights.wo, |w| &mut w.wo),
            ("ffn_w1", &grad.ffn_w1, &weights.ffn_w1, |w| &mut w.ffn_w1),
            ("ffn_w2", &grad.ffn_w2, &weights.ffn_w2, |w| &mut w.ffn_w2),
            ("pos_embed", &grad.pos_embed, &weights.pos_embed, |w| {
                &mut w.pos_embed
            }),
            ("mhc", &grad.mhc, &weights.mhc, |w| &mut w.mhc),
        ];
        for (name, analytic, base, select) in checks {
            let indices: Vec<usize> = if name == "qkv_weight" {
                vec![0, 7, 11]
            } else if name == "mhc" {
                vec![0, 4, 6, 10]
            } else {
                (0..base.len()).collect()
            };
            for i in indices {
                let mut plus = weights.clone();
                let mut minus = weights.clone();
                select(&mut plus)[i] += eps;
                select(&mut minus)[i] -= eps;
                let fd = (loss(&input, &plus) - loss(&input, &minus)) / (2.0 * eps);
                assert!(
                    (analytic[i] - fd).abs() < 2e-3,
                    "{name}[{i}] {} != {fd}",
                    analytic[i]
                );
            }
        }
    }

    #[test]
    fn patchify_matches_nchw_to_token_hand_computation_and_scatter_adjoint() {
        // One 1-channel 2x3 image, 2x2 overlapping patch windows.
        let (oh, ow, patches) =
            patchify_nchw(&[1., 2., 3., 4., 5., 6.], 1, 1, 2, 3, 2, 1, 0).unwrap();
        assert_eq!((oh, ow), (1, 2));
        assert_eq!(patches, vec![1., 2., 4., 5., 2., 3., 5., 6.]);
        let scattered =
            scatter_patch_gradients_nchw(&vec![1.; 8], 1, 1, 2, 3, 2, 1, 0, oh, ow).unwrap();
        assert_eq!(scattered, vec![1., 2., 1., 1., 2., 1.]);
        // Adjoint identity: <patchify(x), g> = <x, scatter(g)>.
        let lhs: f32 = patches.iter().sum();
        let rhs: f32 = [1., 2., 3., 4., 5., 6.]
            .iter()
            .zip(scattered)
            .map(|(x, g)| x * g)
            .sum();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn patchify_axis_layout_keeps_channel_fastest_order_across_batches() {
        // NCHW, B=2, C=2, H=2, W=3, with distinct batch/channel sentinels.
        let input = vec![
            1., 2., 3., 4., 5., 6., 101., 102., 103., 104., 105., 106., 201., 202., 203., 204.,
            205., 206., 301., 302., 303., 304., 305., 306.,
        ];
        let (oh, ow, patches) = patchify_nchw(&input, 2, 2, 2, 3, 2, 1, 0).unwrap();
        assert_eq!((oh, ow), (1, 2));
        // Each patch is [patch_y, patch_x, channel], tokens are row-major,
        // and the two batch rows remain independent.
        assert_eq!(
            patches,
            vec![
                1., 101., 2., 102., 4., 104., 5., 105., 2., 102., 3., 103., 5., 105., 6., 106.,
                201., 301., 202., 302., 204., 304., 205., 305., 202., 302., 203., 303., 205., 305.,
                206., 306.,
            ]
        );
    }

    #[test]
    fn patchify_zero_pads_and_rejects_bad_stride() {
        let (oh, ow, patches) = patchify_nchw(&[2.], 1, 1, 1, 1, 3, 1, 1).unwrap();
        assert_eq!((oh, ow), (1, 1));
        assert_eq!(patches, vec![0., 0., 0., 0., 2., 0., 0., 0., 0.]);
        assert!(patchify_nchw(&[0.; 4], 1, 1, 2, 2, 1, 2, 0).is_err());
    }

    #[test]
    fn sinusoidal_embedding_matches_the_four_axis_channels() {
        let pos = sinusoidal_pos_embed(2, 2, 4);
        assert_eq!(&pos[..4], &[0.0, 1.0, 0.0, 1.0]);
        let row = &pos[3 * 4..4 * 4];
        assert!((row[0] - 1.0f32.sin()).abs() < 1e-7);
        assert!((row[1] - 1.0f32.cos()).abs() < 1e-7);
        assert!((row[2] - 1.0f32.sin()).abs() < 1e-7);
        assert!((row[3] - 1.0f32.cos()).abs() < 1e-7);
    }

    #[test]
    fn layer_norm_matches_hand_computation_and_finite_differences() {
        let x = [1.0f32, 2.0, 3.0];
        let gamma = [2.0f32, 1.0, 0.5];
        let beta = [0.1f32, -0.2, 0.3];
        let (y, cache) = layer_norm_forward(&x, &gamma, &beta, 1, 3, 1e-5).unwrap();
        let mean = 2.0f32;
        let inv = (2.0f32 / 3.0 + 1e-5).sqrt().recip();
        let want = [-inv * 2.0 + 0.1, -0.2, inv * 0.5 + 0.3];
        for i in 0..3 {
            assert!((y[i] - want[i]).abs() < 1e-5);
        }
        assert_eq!(mean, x.iter().sum::<f32>() / 3.0);
        let upstream = [0.7f32, -0.3, 1.1];
        let (dx, dgamma, dbeta) = layer_norm_backward(&upstream, &gamma, &cache).unwrap();
        let loss = |input: &[f32], g: &[f32], b: &[f32]| {
            layer_norm_forward(input, g, b, 1, 3, 1e-5)
                .unwrap()
                .0
                .iter()
                .zip(upstream)
                .map(|(v, d)| v * d)
                .sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..3 {
            let mut plus = x;
            let mut minus = x;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&plus, &gamma, &beta) - loss(&minus, &gamma, &beta)) / (2.0 * eps);
            assert!((dx[i] - fd).abs() < 5e-4, "dx[{i}] {} != {fd}", dx[i]);
            let mut gp = gamma;
            let mut gm = gamma;
            gp[i] += eps;
            gm[i] -= eps;
            let fd = (loss(&x, &gp, &beta) - loss(&x, &gm, &beta)) / (2.0 * eps);
            assert!(
                (dgamma[i] - fd).abs() < 5e-4,
                "dgamma[{i}] {} != {fd}",
                dgamma[i]
            );
            let mut bp = beta;
            let mut bm = beta;
            bp[i] += eps;
            bm[i] -= eps;
            let fd = (loss(&x, &gamma, &bp) - loss(&x, &gamma, &bm)) / (2.0 * eps);
            assert!(
                (dbeta[i] - fd).abs() < 5e-4,
                "dbeta[{i}] {} != {fd}",
                dbeta[i]
            );
        }
    }

    #[test]
    fn linear_matches_hand_computation_and_finite_differences() {
        let x = [1.0f32, 2.0, -1.0, 3.0];
        let w = [2.0f32, -1.0, 0.5, 3.0, -2.0, 1.0];
        let b = [0.25f32, -0.5, 1.0];
        let y = linear_forward(&x, &w, &b, 2, 2, 3).unwrap();
        assert_eq!(y, vec![0.25, 6.0, 1.0, -4.75, 8.0, 6.0]);
        let dy = [0.7f32, -0.3, 1.1, -0.2, 0.4, 0.9];
        let (dx, dw, db) = linear_backward(&x, &w, &dy, 2, 2, 3).unwrap();
        let objective = |xx: &[f32], ww: &[f32], bb: &[f32]| {
            linear_forward(xx, ww, bb, 2, 2, 3)
                .unwrap()
                .iter()
                .zip(dy)
                .map(|(v, d)| v * d)
                .sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..x.len() {
            let mut plus = x;
            let mut minus = x;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (objective(&plus, &w, &b) - objective(&minus, &w, &b)) / (2.0 * eps);
            assert!((dx[i] - fd).abs() < 5e-4, "dx[{i}] {} != {fd}", dx[i]);
        }
        for i in 0..w.len() {
            let mut plus = w;
            let mut minus = w;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (objective(&x, &plus, &b) - objective(&x, &minus, &b)) / (2.0 * eps);
            assert!((dw[i] - fd).abs() < 5e-4, "dw[{i}] {} != {fd}", dw[i]);
        }
        for i in 0..b.len() {
            let mut plus = b;
            let mut minus = b;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (objective(&x, &w, &plus) - objective(&x, &w, &minus)) / (2.0 * eps);
            assert!((db[i] - fd).abs() < 5e-4, "db[{i}] {} != {fd}", db[i]);
        }
    }

    #[test]
    fn vit_vc_mode_uses_the_quantized_reference_instead_of_standard_flash_callback() {
        let (weights, mut geometry, input, _) = tiny_vit_fixture();
        geometry.flash_attention = true;
        geometry.vc_attention = true;
        geometry.vc_expcast = 0;
        geometry.vc_perm = crate::model::VitVcPermutation::Spatial8;
        let (expected, expected_cache) =
            vit_forward(&input, &weights, &geometry).expect("VC reference forward");
        assert!(expected_cache.attention.vc_heads.is_some());

        let (actual, cache) =
            vit_forward_with_flash_attention(&input, &weights, &geometry, |_, _, _, _, _| {
                Err("standard flash callback must be bypassed for VC".into())
            })
            .unwrap();
        assert_eq!(actual, expected);
        assert!(cache.attention.vc_heads.is_some());

        geometry.flash_attention = false;
        assert!(vit_forward(&input, &weights, &geometry).is_err());
    }

    #[test]
    fn attention_matches_a_hand_computed_two_token_case() {
        // N=H=1, Q=K=0 gives uniform attention over V=[2,4].
        let qkv = [0., 0., 2., 0., 0., 4.];
        let (out, cache) = attention_forward(&qkv, 1, 2, 1, 1, None).unwrap();
        assert_eq!(out, vec![3., 3.]);
        assert_eq!(cache.probabilities, vec![0.5; 4]);
        assert_eq!(
            attention_backward(&[1., 2.], &cache).unwrap(),
            vec![0., 0., 1.5, 0., 0., 1.5]
        );
    }

    #[test]
    fn attention_input_gradient_matches_finite_differences() {
        let qkv = [0.2f32, 0.3, 0.7, -0.4, 0.1, -0.2];
        let upstream = [0.6f32, -0.8];
        let (out, cache) = attention_forward(&qkv, 1, 2, 1, 1, None).unwrap();
        let analytic = attention_backward(&upstream, &cache).unwrap();
        let loss = |x: &[f32]| {
            attention_forward(x, 1, 2, 1, 1, None)
                .unwrap()
                .0
                .iter()
                .zip(upstream)
                .map(|(y, g)| y * g)
                .sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..qkv.len() {
            let mut plus = qkv;
            let mut minus = qkv;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            assert!(
                (analytic[i] - fd).abs() < 2e-4,
                "qkv[{i}] {} != {fd}; out={out:?}",
                analytic[i]
            );
        }
    }

    #[test]
    fn online_attention_matches_dense_and_recomputes_exact_gradients() {
        let qkv: Vec<f32> = (0..36).map(|i| ((i as f32) * 0.17).sin() * 0.6).collect();
        let upstream: Vec<f32> = (0..12).map(|i| ((i as f32) * 0.31).cos() * 0.8).collect();
        let (dense_out, dense_cache) = attention_forward(&qkv, 1, 3, 4, 2, None).unwrap();
        let (flash_out, flash_cache) = attention_forward_flash(&qkv, 1, 3, 4, 2).unwrap();
        for (i, (dense, flash)) in dense_out.iter().zip(&flash_out).enumerate() {
            assert!((dense - flash).abs() < 2e-6, "out[{i}]: {dense} != {flash}");
        }
        assert!(flash_cache.probabilities.is_empty());
        assert!(flash_cache.score_active.is_empty());
        assert_eq!(flash_cache.flash_lse.as_ref().unwrap().len(), 6);

        let dense_grad = attention_backward(&upstream, &dense_cache).unwrap();
        let flash_grad = attention_backward(&upstream, &flash_cache).unwrap();
        for (i, (dense, flash)) in dense_grad.iter().zip(&flash_grad).enumerate() {
            assert!(
                (dense - flash).abs() < 3e-6,
                "dqkv[{i}]: {dense} != {flash}"
            );
        }

        let loss = |values: &[f32]| {
            attention_forward_flash(values, 1, 3, 4, 2)
                .unwrap()
                .0
                .iter()
                .zip(&upstream)
                .map(|(y, dy)| y * dy)
                .sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..qkv.len() {
            let mut plus = qkv.clone();
            let mut minus = qkv.clone();
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            assert!(
                (flash_grad[i] - fd).abs() < 2e-4,
                "flash qkv[{i}] {} != {fd}",
                flash_grad[i]
            );
        }
    }

    #[test]
    fn vc_attention_dispatches_batch_heads_and_packs_qkv_gradients() {
        let (batch, tokens, channels, heads) = (2, 4, 8, 2);
        let head_dim = channels / heads;
        let qkv: Vec<f32> = (0..batch * tokens * 3 * channels)
            .map(|index| (index as f32 * 0.173 + 0.21).sin() * 0.65)
            .collect();
        let output_grad: Vec<f32> = (0..batch * tokens * channels)
            .map(|index| (index as f32 * 0.37 + 0.1).cos() * 0.8)
            .collect();
        let permutation: Vec<usize> = (0..tokens).rev().collect();
        let mode = crate::vc_attention::ExpCastMode::Order0;
        let (context, cache) = attention_forward_vc(
            &qkv,
            batch,
            tokens,
            channels,
            heads,
            mode,
            Some(&permutation),
        )
        .unwrap();
        assert_eq!(context.len(), batch * tokens * channels);
        assert_eq!(cache.vc_heads.as_ref().unwrap().len(), batch * heads);
        assert!(cache.q.is_empty() && cache.k.is_empty() && cache.v.is_empty());
        assert!(cache.probabilities.is_empty() && cache.score_active.is_empty());

        let actual_gradient = attention_backward(&output_grad, &cache).unwrap();
        assert_eq!(actual_gradient.len(), batch * tokens * 3 * channels);
        assert!(actual_gradient.iter().all(|gradient| gradient.is_finite()));

        let mut expected_context = vec![0.0; context.len()];
        let mut expected_gradient = vec![0.0; actual_gradient.len()];
        for b in 0..batch {
            for h in 0..heads {
                let mut q = vec![0.0; tokens * head_dim];
                let mut k = vec![0.0; tokens * head_dim];
                let mut v = vec![0.0; tokens * head_dim];
                let mut head_output_grad = vec![0.0; tokens * head_dim];
                for token in 0..tokens {
                    for d in 0..head_dim {
                        let channel = h * head_dim + d;
                        let row = (b * tokens + token) * 3 * channels;
                        let target = token * head_dim + d;
                        q[target] = qkv[row + channel];
                        k[target] = qkv[row + channels + channel];
                        v[target] = qkv[row + 2 * channels + channel];
                        head_output_grad[target] =
                            output_grad[(b * tokens + token) * channels + channel];
                    }
                }
                let (head_context, _, head_cache) = crate::vc_attention::vc_head_forward(
                    &q,
                    &k,
                    &v,
                    tokens,
                    head_dim,
                    1.0 / (head_dim as f32).sqrt(),
                    10.0,
                    mode,
                    Some(&permutation),
                )
                .unwrap();
                let (dq, dk, dv) =
                    crate::vc_attention::vc_head_backward(&head_output_grad, &head_cache).unwrap();
                for token in 0..tokens {
                    for d in 0..head_dim {
                        let channel = h * head_dim + d;
                        let row = (b * tokens + token) * channels;
                        let fused_row = (b * tokens + token) * 3 * channels;
                        let source = token * head_dim + d;
                        expected_context[row + channel] = head_context[source];
                        expected_gradient[fused_row + channel] = dq[source];
                        expected_gradient[fused_row + channels + channel] = dk[source];
                        expected_gradient[fused_row + 2 * channels + channel] = dv[source];
                    }
                }
            }
        }
        assert_eq!(context, expected_context);
        assert_eq!(actual_gradient, expected_gradient);
    }

    #[test]
    fn flash_backend_inputs_preserve_nonfinite_and_saturated_qk_semantics() {
        let qkv = [
            200.0,
            f32::NAN,
            -200.0,
            f32::INFINITY,
            1.0,
            2.0,
            0.0,
            128.0,
            f32::NEG_INFINITY,
            0.5,
            3.0,
            4.0,
        ];
        let (_, original_cache) = attention_forward_flash(&qkv, 1, 2, 2, 1).unwrap();
        let backend_inputs = attention_flash_backward_inputs(&original_cache)
            .unwrap()
            .expect("flash cache");
        let reconstructed_cache = attention_flash_cache_from_lse(
            &backend_inputs.qkv,
            backend_inputs.batch,
            backend_inputs.tokens,
            backend_inputs.channels,
            backend_inputs.heads,
            &backend_inputs.lse,
        )
        .unwrap();
        assert!(backend_inputs.qkv[1].is_nan());
        assert!(backend_inputs.qkv[3].is_nan());
        assert_eq!(backend_inputs.qkv[0], 128.0);
        assert_eq!(backend_inputs.qkv[2], -128.0);
        assert_eq!(backend_inputs.qkv[7], 128.0);
        assert_eq!(backend_inputs.qkv[4], 1.0);
        assert_eq!(backend_inputs.qkv[5], 2.0);
        assert_eq!(backend_inputs.qkv[10], 3.0);
        assert_eq!(backend_inputs.qkv[11], 4.0);
        let upstream = [0.2, -0.3, 0.4, 0.1];
        let original_grad = attention_backward(&upstream, &original_cache).unwrap();
        let reconstructed_grad = attention_backward(&upstream, &reconstructed_cache).unwrap();
        assert_eq!(original_grad, reconstructed_grad);
    }

    #[test]
    fn attention_sparse_mask_has_zero_unselected_probability_and_gradient() {
        let qkv = [0., 0., 0., 0., 0., 1., 0., 0., 2., 0., 0., 3.];
        let (out, cache) = attention_forward(&qkv, 1, 4, 1, 1, Some((1, 2))).unwrap();
        assert_eq!(cache.probabilities[2], 0.0); // block 2 is not selected for query token 0
        assert!(out.iter().all(|v| v.is_finite()));
        let grad = attention_backward(&[1.; 4], &cache).unwrap();
        assert!(grad.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn mhc_default_parameters_approximate_skip_identity() {
        let out = mhc_residual_forward(&[0.4, -0.6], &[0.7, 0.2], &MHC_DEFAULTS, 0, 10.0).unwrap();
        let (a, b, _, _, _) = mhc_coefficients(&MHC_DEFAULTS, 0).unwrap();
        assert!(a > 0.99 && b < 0.01);
        assert!((out[0] - (a * 0.4 + b * 0.7)).abs() < 1e-6);
        assert!((out[1] - (a * -0.6 + b * 0.2)).abs() < 1e-6);
    }

    #[test]
    fn mhc_backward_matches_finite_differences_for_inputs_and_site_parameters() {
        let skip = [0.3f32, -0.7];
        let branch = [-0.2f32, 0.5];
        let upstream = [0.8f32, -1.1];
        let mut params = [0.2f32; MHC_PARAM_COUNT];
        params[0..6].copy_from_slice(&[0.3, -0.2, 0.1, 0.4, -0.1, 0.2]);
        let (ds, db, dparams) =
            mhc_residual_backward(&skip, &branch, &upstream, &params, 0, 10.0).unwrap();
        let loss = |s: &[f32], b: &[f32], p: &[f32]| {
            mhc_residual_forward(s, b, p, 0, 10.0)
                .unwrap()
                .iter()
                .zip(upstream)
                .map(|(y, g)| y * g)
                .sum::<f32>()
        };
        let eps = 1e-3;
        for i in 0..skip.len() {
            let mut plus = skip;
            let mut minus = skip;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&plus, &branch, &params) - loss(&minus, &branch, &params)) / (2.0 * eps);
            assert!((ds[i] - fd).abs() < 2e-4, "skip {i}: {} != {fd}", ds[i]);
            let mut plus = branch;
            let mut minus = branch;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&skip, &plus, &params) - loss(&skip, &minus, &params)) / (2.0 * eps);
            assert!((db[i] - fd).abs() < 2e-4, "branch {i}: {} != {fd}", db[i]);
        }
        for i in 0..6 {
            let mut plus = params;
            let mut minus = params;
            plus[i] += eps;
            minus[i] -= eps;
            let fd = (loss(&skip, &branch, &plus) - loss(&skip, &branch, &minus)) / (2.0 * eps);
            assert!(
                (dparams[i] - fd).abs() < 2e-4,
                "param {i}: {} != {fd}",
                dparams[i]
            );
        }
    }

    #[test]
    fn mhc_backward_includes_clamp_derivatives() {
        let skip = [11.0, 0.2];
        let branch = [0.4, 0.1];
        let grad = [1.0, 1.0];
        let (ds, db, _) =
            mhc_residual_backward(&skip, &branch, &grad, &MHC_DEFAULTS, 0, 10.0).unwrap();
        assert_eq!(ds[0], 0.0); // saturated skip input
        assert!(db[0] != 0.0);
    }

    #[test]
    fn sparse_softmax_selects_local_and_best_blocks() {
        let p = sparse_softmax_row(&[0., 1., 2., 3., 4., 5.], 0, 2, 2, 10.).unwrap();
        assert!(p[0] > 0.0 && p[1] > 0.0 && p[4] > 0.0 && p[5] > 0.0);
        assert_eq!(p[2], 0.0);
        assert_eq!(p[3], 0.0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sparse_softmax_is_dense_when_topk_covers_all_blocks_and_ties_are_stable() {
        let dense = sparse_softmax_row(&[0., 1., 2.], 0, 1, 3, 10.).unwrap();
        let expected = softmax(&[0., 1., 2.]);
        assert_eq!(dense, expected);
        // Query's local block is block 0; equal maxima choose block 1 before block 2.
        let tied = sparse_softmax_row(&[0., 2., 2.], 0, 1, 2, 10.).unwrap();
        assert!(tied[0] > 0.0);
        assert!(tied[1] > 0.0);
        assert_eq!(tied[2], 0.0);
    }
}

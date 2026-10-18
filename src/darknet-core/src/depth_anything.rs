//! CPU math for the frozen `[depth_anything]` DINOv2 encoder.
//!
//! Tensor ordering and weight order follow `Jev_laya:src-lib/depth_anything_layer.cpp`
//! and `weights.cpp`. This is a hand-derived Rust implementation; it does not
//! claim a compiled-C++ golden comparison. The pretrained parameters are frozen,
//! but [`backward`] computes the complete input derivative of this Rust forward.

use crate::laya_scorer::{gelu_erf, gelu_erf_grad};
use crate::vit::{LayerNormCache, layer_norm_backward, layer_norm_forward, linear_forward};

const RGB_CHANNELS: usize = 3;

/// Geometry for one `[depth_anything]` CPU invocation.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAnythingGeometry {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub patch_size: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp_dim: usize,
    pub pos_grid_h: usize,
    pub pos_grid_w: usize,
    pub ln_eps: f32,
    pub apply_imagenet_norm: bool,
    pub imagenet_mean: [f32; RGB_CHANNELS],
    pub imagenet_std: [f32; RGB_CHANNELS],
    /// Transformer block indices, in output-channel order. Repeated taps are
    /// accepted because the source layer permits selecting the same block more
    /// than once.
    pub taps: Vec<usize>,
    /// Positive values clamp each final-normalized tap; zero disables clamp.
    pub output_clamp: f32,
}

impl DepthAnythingGeometry {
    pub fn validate(&self) -> Result<(), String> {
        if self.batch == 0
            || self.height == 0
            || self.width == 0
            || self.patch_size == 0
            || self.dim == 0
            || self.depth == 0
            || self.heads == 0
            || self.mlp_dim == 0
            || self.pos_grid_h == 0
            || self.pos_grid_w == 0
        {
            return Err(
                "depth_anything dimensions, depth, heads, and taps must be positive".into(),
            );
        }
        if self.dim % self.heads != 0 {
            return Err(format!(
                "depth_anything dim={} must be divisible by heads={}",
                self.dim, self.heads
            ));
        }
        if self.height % self.patch_size != 0 || self.width % self.patch_size != 0 {
            return Err(format!(
                "depth_anything input {}x{} must be divisible by patch_size={}",
                self.width, self.height, self.patch_size
            ));
        }
        if self.taps.is_empty() || self.taps.len() > 8 {
            return Err(format!(
                "depth_anything requires 1..=8 taps, got {}",
                self.taps.len()
            ));
        }
        if let Some(tap) = self.taps.iter().find(|tap| **tap >= self.depth) {
            return Err(format!(
                "depth_anything tap {tap} is outside depth={}",
                self.depth
            ));
        }
        if !self.ln_eps.is_finite() || self.ln_eps <= 0.0 {
            return Err("depth_anything ln_eps must be finite and positive".into());
        }
        if !self.output_clamp.is_finite() || self.output_clamp < 0.0 {
            return Err("depth_anything clamp must be finite and nonnegative".into());
        }
        if self.imagenet_mean.iter().any(|value| !value.is_finite()) {
            return Err("depth_anything imagenet_mean values must be finite".into());
        }
        if self
            .imagenet_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("depth_anything imagenet_std values must be finite and positive".into());
        }
        // Force all public dimensions through checked arithmetic here, before
        // any tensor allocation or index calculation.
        let tokens = self.token_count()?;
        let _ = self.input_len()?;
        let _ = checked_product(&[self.batch, tokens, self.dim], "token tensor size")?;
        let _ = checked_product(&[self.batch, tokens, 3, self.dim], "QKV tensor size")?;
        let _ = checked_product(
            &[self.batch, self.heads, tokens, tokens],
            "attention probability size",
        )?;
        let _ = self.output_len()?;
        let _ = self.weight_count()?;
        Ok(())
    }

    pub fn grid_h(&self) -> usize {
        self.height / self.patch_size
    }

    pub fn grid_w(&self) -> usize {
        self.width / self.patch_size
    }

    pub fn patch_count(&self) -> Result<usize, String> {
        checked_mul(self.grid_h(), self.grid_w(), "patch count")
    }

    pub fn token_count(&self) -> Result<usize, String> {
        checked_add(self.patch_count()?, 1, "token count")
    }

    pub fn input_len(&self) -> Result<usize, String> {
        checked_product(
            &[self.batch, RGB_CHANNELS, self.height, self.width],
            "input size",
        )
    }

    pub fn output_channels(&self) -> Result<usize, String> {
        checked_mul(self.taps.len(), self.dim, "output channel count")
    }

    pub fn output_len(&self) -> Result<usize, String> {
        checked_product(
            &[
                self.batch,
                self.output_channels()?,
                self.grid_h(),
                self.grid_w(),
            ],
            "output size",
        )
    }

    /// Number of f32 values in the C++ type-major encoder blob.
    pub fn weight_count(&self) -> Result<usize, String> {
        Ok(weight_offsets(self)?.total)
    }
}

/// Flat frozen parameters in the exact type-major `load_depth_anything_weights`
/// order. Keeping this blob flat makes the checkpoint layout explicit and
/// avoids silently transposing Q/K/V or patch kernels.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAnythingWeights {
    values: Vec<f32>,
}

impl DepthAnythingWeights {
    pub fn zeroed(geometry: &DepthAnythingGeometry) -> Result<Self, String> {
        geometry.validate()?;
        let offsets = weight_offsets(geometry)?;
        let mut values = vec![0.0; offsets.total];
        values[offsets.norm_gamma.range()].fill(1.0);
        for field in [
            offsets.ln1_gamma,
            offsets.ls1,
            offsets.ln2_gamma,
            offsets.ls2,
        ] {
            values[field.range()].fill(1.0);
        }
        Ok(Self { values })
    }

    pub fn from_blob(blob: &[f32], geometry: &DepthAnythingGeometry) -> Result<Self, String> {
        geometry.validate()?;
        let expected = weight_offsets(geometry)?.total;
        if blob.len() != expected {
            return Err(format!(
                "depth_anything weight blob has {} floats, expected {expected}",
                blob.len()
            ));
        }
        Ok(Self {
            values: blob.to_vec(),
        })
    }

    pub fn to_blob(&self) -> Vec<f32> {
        self.values.clone()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    pub fn matches(&self, geometry: &DepthAnythingGeometry) -> bool {
        geometry
            .weight_count()
            .is_ok_and(|expected| self.values.len() == expected)
    }
}

/// Per-array lengths in the source `load_depth_anything_weights` order.
pub fn weight_array_sizes(
    geometry: &DepthAnythingGeometry,
) -> Result<Vec<(&'static str, usize)>, String> {
    geometry.validate()?;
    let offsets = weight_offsets(geometry)?;
    Ok(vec![
        ("patch_w", offsets.patch_w.len),
        ("patch_b", offsets.patch_b.len),
        ("cls_token", offsets.cls_token.len),
        ("pos_embed", offsets.pos_embed.len),
        ("norm_gamma", offsets.norm_gamma.len),
        ("norm_beta", offsets.norm_beta.len),
        ("ln1_gamma", offsets.ln1_gamma.len),
        ("ln1_beta", offsets.ln1_beta.len),
        ("qkv_w", offsets.qkv_w.len),
        ("qkv_b", offsets.qkv_b.len),
        ("proj_w", offsets.proj_w.len),
        ("proj_b", offsets.proj_b.len),
        ("ls1", offsets.ls1.len),
        ("ln2_gamma", offsets.ln2_gamma.len),
        ("ln2_beta", offsets.ln2_beta.len),
        ("fc1_w", offsets.fc1_w.len),
        ("fc1_b", offsets.fc1_b.len),
        ("fc2_w", offsets.fc2_w.len),
        ("fc2_b", offsets.fc2_b.len),
        ("ls2", offsets.ls2.len),
    ])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WeightField {
    start: usize,
    len: usize,
    stride: usize,
}

impl WeightField {
    fn range(self) -> std::ops::Range<usize> {
        self.start..self.start + self.len
    }

    fn block(self, index: usize) -> std::ops::Range<usize> {
        let start = self.start + index * self.stride;
        start..start + self.stride
    }
}

#[derive(Clone, Copy, Debug)]
struct WeightOffsets {
    patch_w: WeightField,
    patch_b: WeightField,
    cls_token: WeightField,
    pos_embed: WeightField,
    norm_gamma: WeightField,
    norm_beta: WeightField,
    ln1_gamma: WeightField,
    ln1_beta: WeightField,
    qkv_w: WeightField,
    qkv_b: WeightField,
    proj_w: WeightField,
    proj_b: WeightField,
    ls1: WeightField,
    ln2_gamma: WeightField,
    ln2_beta: WeightField,
    fc1_w: WeightField,
    fc1_b: WeightField,
    fc2_w: WeightField,
    fc2_b: WeightField,
    ls2: WeightField,
    total: usize,
}

fn weight_offsets(g: &DepthAnythingGeometry) -> Result<WeightOffsets, String> {
    let e = g.dim;
    let m = g.mlp_dim;
    let p2 = checked_mul(g.patch_size, g.patch_size, "patch area")?;
    let mut at = 0usize;
    let mut append = |len: usize, stride: usize| -> Result<WeightField, String> {
        let start = at;
        at = checked_add(at, len, "weight blob size")?;
        Ok(WeightField { start, len, stride })
    };
    let block = |per_block: usize| -> Result<usize, String> {
        checked_mul(g.depth, per_block, "block weight size")
    };

    // Source `load_depth_anything_weights`: six shared tensors followed by
    // each tensor type across all blocks (not block-interleaved).
    let patch_w = append(checked_product(&[e, RGB_CHANNELS, p2], "patch weights")?, 0)?;
    let patch_b = append(e, 0)?;
    let cls_token = append(e, 0)?;
    let pos_embed = append(
        checked_product(
            &[
                checked_add(
                    checked_product(&[g.pos_grid_h, g.pos_grid_w], "position grid")?,
                    1,
                    "position token count",
                )?,
                e,
            ],
            "position embeddings",
        )?,
        0,
    )?;
    let norm_gamma = append(e, 0)?;
    let norm_beta = append(e, 0)?;
    let ln1_gamma = append(block(e)?, e)?;
    let ln1_beta = append(block(e)?, e)?;
    let qkv_stride = checked_product(&[3, e, e], "QKV weights")?;
    let qkv_w = append(block(qkv_stride)?, qkv_stride)?;
    let qkv_bias_stride = checked_mul(3, e, "QKV biases")?;
    let qkv_b = append(block(qkv_bias_stride)?, qkv_bias_stride)?;
    let proj_stride = checked_mul(e, e, "projection weights")?;
    let proj_w = append(block(proj_stride)?, proj_stride)?;
    let proj_b = append(block(e)?, e)?;
    let ls1 = append(block(e)?, e)?;
    let ln2_gamma = append(block(e)?, e)?;
    let ln2_beta = append(block(e)?, e)?;
    let fc1_stride = checked_mul(m, e, "fc1 weights")?;
    let fc1_w = append(block(fc1_stride)?, fc1_stride)?;
    let fc1_b = append(block(m)?, m)?;
    let fc2_stride = checked_mul(e, m, "fc2 weights")?;
    let fc2_w = append(block(fc2_stride)?, fc2_stride)?;
    let fc2_b = append(block(e)?, e)?;
    let ls2 = append(block(e)?, e)?;
    Ok(WeightOffsets {
        patch_w,
        patch_b,
        cls_token,
        pos_embed,
        norm_gamma,
        norm_beta,
        ln1_gamma,
        ln1_beta,
        qkv_w,
        qkv_b,
        proj_w,
        proj_b,
        ls1,
        ln2_gamma,
        ln2_beta,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        ls2,
        total: at,
    })
}

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("depth_anything {what} overflows usize"))
}

fn checked_add(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("depth_anything {what} overflows usize"))
}

fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |n, value| checked_mul(n, *value, what))
}

/// Flatten NCHW RGB patches in the C++ `im2col_cpu` order
/// `(channel, patch_y, patch_x)`. The returned rows are batch-major, then
/// row-major patch-grid positions.
pub fn patchify_nchw_channel_major(
    input: &[f32],
    geometry: &DepthAnythingGeometry,
) -> Result<Vec<f32>, String> {
    geometry.validate()?;
    let expected = geometry.input_len()?;
    if input.len() != expected {
        return Err(format!(
            "depth_anything patchify input has {} values, expected {expected}",
            input.len()
        ));
    }
    let patch_area = checked_mul(geometry.patch_size, geometry.patch_size, "patch area")?;
    let patch_dim = checked_mul(RGB_CHANNELS, patch_area, "patch feature count")?;
    let patches = geometry.patch_count()?;
    let rows = checked_mul(geometry.batch, patches, "patch row count")?;
    let mut output = vec![0.0; checked_mul(rows, patch_dim, "patch matrix size")?];
    for b in 0..geometry.batch {
        for py in 0..geometry.grid_h() {
            for px in 0..geometry.grid_w() {
                let row = (b * geometry.grid_h() + py) * geometry.grid_w() + px;
                let mut col = 0;
                for c in 0..RGB_CHANNELS {
                    for dy in 0..geometry.patch_size {
                        for dx in 0..geometry.patch_size {
                            let y = py * geometry.patch_size + dy;
                            let x = px * geometry.patch_size + dx;
                            let src =
                                ((b * RGB_CHANNELS + c) * geometry.height + y) * geometry.width + x;
                            output[row * patch_dim + col] = input[src];
                            col += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Bicubic resize of token-major `[source_h, source_w, channels]` values,
/// matching the source's `align_corners=false`, border-clamped Keys kernel
/// (`A=-0.75`, no antialias and no coordinate offset).
pub fn bicubic_pos_embed(
    source: &[f32],
    source_h: usize,
    source_w: usize,
    channels: usize,
    target_h: usize,
    target_w: usize,
) -> Result<Vec<f32>, String> {
    if source_h == 0 || source_w == 0 || channels == 0 || target_h == 0 || target_w == 0 {
        return Err("depth_anything bicubic dimensions must be positive".into());
    }
    let source_len = checked_product(&[source_h, source_w, channels], "bicubic source size")?;
    if source.len() != source_len {
        return Err(format!(
            "depth_anything bicubic source has {} values, expected {source_len}",
            source.len()
        ));
    }
    let target_len = checked_product(&[target_h, target_w, channels], "bicubic target size")?;
    if source_h == target_h && source_w == target_w {
        return Ok(source.to_vec());
    }
    let mut output = vec![0.0; target_len];
    let rh = source_h as f32 / target_h as f32;
    let rw = source_w as f32 / target_w as f32;
    for dh in 0..target_h {
        let h_real = rh * (dh as f32 + 0.5) - 0.5;
        let h_base = h_real.floor() as isize;
        let h_weights = cubic_coeffs(h_real - h_base as f32);
        let h_indices = [
            clamp_index(h_base - 1, source_h),
            clamp_index(h_base, source_h),
            clamp_index(h_base + 1, source_h),
            clamp_index(h_base + 2, source_h),
        ];
        for dw in 0..target_w {
            let w_real = rw * (dw as f32 + 0.5) - 0.5;
            let w_base = w_real.floor() as isize;
            let w_weights = cubic_coeffs(w_real - w_base as f32);
            let w_indices = [
                clamp_index(w_base - 1, source_w),
                clamp_index(w_base, source_w),
                clamp_index(w_base + 1, source_w),
                clamp_index(w_base + 2, source_w),
            ];
            let dst = (dh * target_w + dw) * channels;
            for channel in 0..channels {
                let mut acc = 0.0f32;
                for a in 0..4 {
                    for b in 0..4 {
                        let src = (h_indices[a] * source_w + w_indices[b]) * channels + channel;
                        acc += h_weights[a] * w_weights[b] * source[src];
                    }
                }
                output[dst + channel] = acc;
            }
        }
    }
    Ok(output)
}

fn cubic_coeffs(t: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let conv1 = |x: f32| ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let conv2 = |x: f32| ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
    [conv2(t + 1.0), conv1(t), conv1(1.0 - t), conv2(2.0 - t)]
}

fn clamp_index(index: isize, extent: usize) -> usize {
    index.clamp(0, extent as isize - 1) as usize
}

/// Cache retained for the Rust-mode input derivative. Frozen parameters do not
/// receive gradients; each block input is enough to recompute its local cache.
#[derive(Clone, Debug, PartialEq)]
pub struct DepthAnythingForwardCache {
    batch: usize,
    tokens: usize,
    dim: usize,
    block_inputs: Vec<Vec<f32>>,
}

/// Run the source-derived encoder and return tap-major NCHW output plus the
/// forward-time cache required by [`backward`].
pub fn forward(
    input: &[f32],
    weights: &DepthAnythingWeights,
    geometry: &DepthAnythingGeometry,
) -> Result<(Vec<f32>, DepthAnythingForwardCache), String> {
    geometry.validate()?;
    let offsets = weight_offsets(geometry)?;
    if weights.values.len() != offsets.total {
        return Err(format!(
            "depth_anything has {} weight values, expected {}",
            weights.values.len(),
            offsets.total
        ));
    }
    if input.len() != geometry.input_len()? {
        return Err(format!(
            "depth_anything input has {} values, expected {}",
            input.len(),
            geometry.input_len()?
        ));
    }
    let g = geometry;
    let batch = g.batch;
    let dim = g.dim;
    let grid_h = g.grid_h();
    let grid_w = g.grid_w();
    let patches = g.patch_count()?;
    let tokens = g.token_count()?;
    let patch_area = g.patch_size * g.patch_size;
    let patch_dim = RGB_CHANNELS * patch_area;

    let mut normalized = input.to_vec();
    if g.apply_imagenet_norm {
        for b in 0..batch {
            for c in 0..RGB_CHANNELS {
                let start = (b * RGB_CHANNELS + c) * g.height * g.width;
                let end = start + g.height * g.width;
                for value in &mut normalized[start..end] {
                    *value = (*value - g.imagenet_mean[c]) / g.imagenet_std[c];
                }
            }
        }
    }
    let patch_matrix = patchify_nchw_channel_major(&normalized, g)?;
    let patch_features = linear_forward(
        &patch_matrix,
        weight_slice(&weights.values, offsets.patch_w),
        weight_slice(&weights.values, offsets.patch_b),
        batch * patches,
        patch_dim,
        dim,
    )
    .map_err(|error| format!("depth_anything patch projection: {error}"))?;

    let pos_grid_len = g.pos_grid_h * g.pos_grid_w;
    let pos_embed = weight_slice(&weights.values, offsets.pos_embed);
    let interpolated = bicubic_pos_embed(
        &pos_embed[dim..(1 + pos_grid_len) * dim],
        g.pos_grid_h,
        g.pos_grid_w,
        dim,
        grid_h,
        grid_w,
    )?;
    let mut pos = vec![0.0; tokens * dim];
    pos[..dim].copy_from_slice(&pos_embed[..dim]);
    pos[dim..].copy_from_slice(&interpolated);

    let mut hidden = vec![0.0; batch * tokens * dim];
    let cls = weight_slice(&weights.values, offsets.cls_token);
    for b in 0..batch {
        hidden[b * tokens * dim..b * tokens * dim + dim].copy_from_slice(cls);
        for patch in 0..patches {
            let dst = (b * tokens + patch + 1) * dim;
            let src = (b * patches + patch) * dim;
            for e in 0..dim {
                hidden[dst + e] = patch_features[src + e];
            }
        }
        for t in 0..tokens {
            let row = (b * tokens + t) * dim;
            for e in 0..dim {
                hidden[row + e] += pos[t * dim + e];
            }
        }
    }

    let mut output = vec![0.0; g.output_len()?];
    let mut block_inputs = Vec::with_capacity(g.depth);
    for block in 0..g.depth {
        block_inputs.push(hidden.clone());
        let block_w = block_weights(&weights.values, &offsets, block);
        hidden = block_forward(&hidden, block_w, g)?;
        for (tap_index, &tap) in g.taps.iter().enumerate() {
            if tap == block {
                let (normalized_tap, _) = layer_norm_forward(
                    &hidden,
                    weight_slice(&weights.values, offsets.norm_gamma),
                    weight_slice(&weights.values, offsets.norm_beta),
                    batch * tokens,
                    dim,
                    g.ln_eps,
                )
                .map_err(|error| format!("depth_anything final LayerNorm: {error}"))?;
                write_tap(&normalized_tap, &mut output, g, tap_index, grid_h, grid_w);
            }
        }
    }
    Ok((
        output,
        DepthAnythingForwardCache {
            batch,
            tokens,
            dim,
            block_inputs,
        },
    ))
}

/// Exact input derivative of [`forward`]. Gradients from repeated taps and
/// multiple taps are accumulated at their source transformer blocks.
pub fn backward(
    output_gradient: &[f32],
    weights: &DepthAnythingWeights,
    geometry: &DepthAnythingGeometry,
    cache: &DepthAnythingForwardCache,
) -> Result<Vec<f32>, String> {
    geometry.validate()?;
    let offsets = weight_offsets(geometry)?;
    if weights.values.len() != offsets.total {
        return Err(format!(
            "depth_anything has {} weight values, expected {}",
            weights.values.len(),
            offsets.total
        ));
    }
    let expected_output = geometry.output_len()?;
    if output_gradient.len() != expected_output {
        return Err(format!(
            "depth_anything output gradient has {} values, expected {expected_output}",
            output_gradient.len()
        ));
    }
    let tokens = geometry.token_count()?;
    let token_values = checked_product(
        &[geometry.batch, tokens, geometry.dim],
        "token gradient size",
    )?;
    if cache.batch != geometry.batch
        || cache.tokens != tokens
        || cache.dim != geometry.dim
        || cache.block_inputs.len() != geometry.depth
        || cache
            .block_inputs
            .iter()
            .any(|block| block.len() != token_values)
    {
        return Err("depth_anything forward cache does not match geometry".into());
    }

    let mut grad_tokens = vec![0.0; token_values];
    for block in (0..geometry.depth).rev() {
        let block_w = block_weights(&weights.values, &offsets, block);
        let (block_output, block_cache) =
            block_forward_cached(&cache.block_inputs[block], block_w, geometry)?;
        for (tap_index, &tap) in geometry.taps.iter().enumerate() {
            if tap != block {
                continue;
            }
            let (normalized_tap, final_norm_cache) = layer_norm_forward(
                &block_output,
                weight_slice(&weights.values, offsets.norm_gamma),
                weight_slice(&weights.values, offsets.norm_beta),
                geometry.batch * tokens,
                geometry.dim,
                geometry.ln_eps,
            )
            .map_err(|error| format!("depth_anything final LayerNorm: {error}"))?;
            let mut grad_normalized = vec![0.0; token_values];
            let output_channels = geometry.output_channels()?;
            for b in 0..geometry.batch {
                for y in 0..geometry.grid_h() {
                    for x in 0..geometry.grid_w() {
                        let patch = y * geometry.grid_w() + x;
                        let token_row = (b * tokens + patch + 1) * geometry.dim;
                        for e in 0..geometry.dim {
                            let channel = tap_index * geometry.dim + e;
                            let out_index = ((b * output_channels + channel) * geometry.grid_h()
                                + y)
                                * geometry.grid_w()
                                + x;
                            let value = normalized_tap[token_row + e];
                            let saturated = geometry.output_clamp > 0.0
                                && (value < -geometry.output_clamp
                                    || value > geometry.output_clamp);
                            if !saturated {
                                grad_normalized[token_row + e] += output_gradient[out_index];
                            }
                        }
                    }
                }
            }
            let (grad_block_output, _, _) = layer_norm_backward(
                &grad_normalized,
                weight_slice(&weights.values, offsets.norm_gamma),
                &final_norm_cache,
            )
            .map_err(|error| format!("depth_anything final LayerNorm backward: {error}"))?;
            for (dst, src) in grad_tokens.iter_mut().zip(grad_block_output) {
                *dst += src;
            }
        }
        grad_tokens = block_backward(&grad_tokens, block_w, &block_cache, geometry)?;
    }

    // The class token, positional embeddings and frozen weights are constants
    // with respect to the input. Patch projection is the only route to RGB.
    let patch_dim = checked_product(
        &[RGB_CHANNELS, geometry.patch_size, geometry.patch_size],
        "patch feature count",
    )?;
    let patches = geometry.patch_count()?;
    let mut patch_token_gradient = vec![0.0; geometry.batch * patches * geometry.dim];
    for b in 0..geometry.batch {
        for patch in 0..patches {
            let src = (b * tokens + patch + 1) * geometry.dim;
            let dst = (b * patches + patch) * geometry.dim;
            patch_token_gradient[dst..dst + geometry.dim]
                .copy_from_slice(&grad_tokens[src..src + geometry.dim]);
        }
    }
    let patch_grad = linear_input_backward(
        &patch_token_gradient,
        weight_slice(&weights.values, offsets.patch_w),
        geometry.batch * patches,
        patch_dim,
        geometry.dim,
    );
    let mut input_gradient = vec![0.0; geometry.input_len()?];
    for b in 0..geometry.batch {
        for py in 0..geometry.grid_h() {
            for px in 0..geometry.grid_w() {
                let patch_row = (b * geometry.grid_h() + py) * geometry.grid_w() + px;
                let mut col = 0;
                for c in 0..RGB_CHANNELS {
                    let input_scale = if geometry.apply_imagenet_norm {
                        1.0 / geometry.imagenet_std[c]
                    } else {
                        1.0
                    };
                    for dy in 0..geometry.patch_size {
                        for dx in 0..geometry.patch_size {
                            let y = py * geometry.patch_size + dy;
                            let x = px * geometry.patch_size + dx;
                            let dst =
                                ((b * RGB_CHANNELS + c) * geometry.height + y) * geometry.width + x;
                            input_gradient[dst] =
                                patch_grad[patch_row * patch_dim + col] * input_scale;
                            col += 1;
                        }
                    }
                }
            }
        }
    }
    Ok(input_gradient)
}

#[derive(Clone, Copy)]
struct BlockWeightRefs<'a> {
    ln1_gamma: &'a [f32],
    ln1_beta: &'a [f32],
    qkv_w: &'a [f32],
    qkv_b: &'a [f32],
    proj_w: &'a [f32],
    proj_b: &'a [f32],
    ls1: &'a [f32],
    ln2_gamma: &'a [f32],
    ln2_beta: &'a [f32],
    fc1_w: &'a [f32],
    fc1_b: &'a [f32],
    fc2_w: &'a [f32],
    fc2_b: &'a [f32],
    ls2: &'a [f32],
}

fn weight_slice(values: &[f32], field: WeightField) -> &[f32] {
    &values[field.range()]
}

fn block_weights<'a>(
    values: &'a [f32],
    offsets: &WeightOffsets,
    block: usize,
) -> BlockWeightRefs<'a> {
    let slice = |field: WeightField| &values[field.block(block)];
    BlockWeightRefs {
        ln1_gamma: slice(offsets.ln1_gamma),
        ln1_beta: slice(offsets.ln1_beta),
        qkv_w: slice(offsets.qkv_w),
        qkv_b: slice(offsets.qkv_b),
        proj_w: slice(offsets.proj_w),
        proj_b: slice(offsets.proj_b),
        ls1: slice(offsets.ls1),
        ln2_gamma: slice(offsets.ln2_gamma),
        ln2_beta: slice(offsets.ln2_beta),
        fc1_w: slice(offsets.fc1_w),
        fc1_b: slice(offsets.fc1_b),
        fc2_w: slice(offsets.fc2_w),
        fc2_b: slice(offsets.fc2_b),
        ls2: slice(offsets.ls2),
    }
}

struct BlockCache {
    ln1: LayerNormCache,
    qkv: Vec<f32>,
    probabilities: Vec<f32>,
    ln2: LayerNormCache,
    fc1_pre: Vec<f32>,
}

fn block_forward(
    input: &[f32],
    weights: BlockWeightRefs<'_>,
    g: &DepthAnythingGeometry,
) -> Result<Vec<f32>, String> {
    Ok(block_forward_cached(input, weights, g)?.0)
}

fn block_forward_cached(
    input: &[f32],
    weights: BlockWeightRefs<'_>,
    g: &DepthAnythingGeometry,
) -> Result<(Vec<f32>, BlockCache), String> {
    let tokens = g.token_count()?;
    let rows = g.batch * tokens;
    let e = g.dim;
    let (norm1, ln1) = layer_norm_forward(
        input,
        weights.ln1_gamma,
        weights.ln1_beta,
        rows,
        e,
        g.ln_eps,
    )
    .map_err(|error| format!("depth_anything block LayerNorm 1: {error}"))?;
    let qkv = linear_forward(&norm1, weights.qkv_w, weights.qkv_b, rows, e, 3 * e)
        .map_err(|error| format!("depth_anything QKV projection: {error}"))?;
    let (context, probabilities) = attention_forward(&qkv, g)?;
    let projected = linear_forward(&context, weights.proj_w, weights.proj_b, rows, e, e)
        .map_err(|error| format!("depth_anything attention projection: {error}"))?;
    let mut after_attention = input.to_vec();
    for row in 0..rows {
        for channel in 0..e {
            after_attention[row * e + channel] +=
                weights.ls1[channel] * projected[row * e + channel];
        }
    }
    let (norm2, ln2) = layer_norm_forward(
        &after_attention,
        weights.ln2_gamma,
        weights.ln2_beta,
        rows,
        e,
        g.ln_eps,
    )
    .map_err(|error| format!("depth_anything block LayerNorm 2: {error}"))?;
    let fc1_pre = linear_forward(&norm2, weights.fc1_w, weights.fc1_b, rows, e, g.mlp_dim)
        .map_err(|error| format!("depth_anything fc1: {error}"))?;
    let gelu_hidden = fc1_pre
        .iter()
        .map(|value| gelu_erf(*value))
        .collect::<Vec<_>>();
    let mlp = linear_forward(
        &gelu_hidden,
        weights.fc2_w,
        weights.fc2_b,
        rows,
        g.mlp_dim,
        e,
    )
    .map_err(|error| format!("depth_anything fc2: {error}"))?;
    let mut output = after_attention;
    for row in 0..rows {
        for channel in 0..e {
            output[row * e + channel] += weights.ls2[channel] * mlp[row * e + channel];
        }
    }
    let cache = BlockCache {
        ln1,
        qkv,
        probabilities,
        ln2,
        fc1_pre,
    };
    Ok((output, cache))
}

fn attention_forward(
    qkv: &[f32],
    g: &DepthAnythingGeometry,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let tokens = g.token_count()?;
    let e = g.dim;
    let head_dim = e / g.heads;
    let rows = g.batch * tokens;
    let qkv_len = checked_product(&[rows, 3, e], "QKV tensor")?;
    if qkv.len() != qkv_len {
        return Err(format!(
            "depth_anything QKV tensor has {} values, expected {qkv_len}",
            qkv.len()
        ));
    }
    let probability_len = checked_product(
        &[g.batch, g.heads, tokens, tokens],
        "attention probabilities",
    )?;
    let mut probabilities = vec![0.0; probability_len];
    let mut context = vec![0.0; rows * e];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut scores = vec![0.0; tokens];
    for b in 0..g.batch {
        for head in 0..g.heads {
            for query in 0..tokens {
                let qbase = (b * tokens + query) * 3 * e + head * head_dim;
                let mut max_score = f32::NEG_INFINITY;
                for key in 0..tokens {
                    let kbase = (b * tokens + key) * 3 * e + e + head * head_dim;
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        score += qkv[qbase + d] * qkv[kbase + d];
                    }
                    score *= scale;
                    scores[key] = score;
                    if score > max_score {
                        max_score = score;
                    }
                }
                let prob_base = ((b * g.heads + head) * tokens + query) * tokens;
                let mut sum = 0.0;
                for key in 0..tokens {
                    let value = (scores[key] - max_score).exp();
                    probabilities[prob_base + key] = value;
                    sum += value;
                }
                for key in 0..tokens {
                    probabilities[prob_base + key] /= sum;
                }
                let out_base = (b * tokens + query) * e + head * head_dim;
                for key in 0..tokens {
                    let p = probabilities[prob_base + key];
                    let vbase = (b * tokens + key) * 3 * e + 2 * e + head * head_dim;
                    for d in 0..head_dim {
                        context[out_base + d] += p * qkv[vbase + d];
                    }
                }
            }
        }
    }
    Ok((context, probabilities))
}

fn attention_backward(
    context_gradient: &[f32],
    qkv: &[f32],
    probabilities: &[f32],
    g: &DepthAnythingGeometry,
) -> Result<Vec<f32>, String> {
    let tokens = g.token_count()?;
    let e = g.dim;
    let head_dim = e / g.heads;
    let rows = g.batch * tokens;
    if context_gradient.len() != rows * e
        || qkv.len() != rows * 3 * e
        || probabilities.len() != g.batch * g.heads * tokens * tokens
    {
        return Err("depth_anything attention backward cache/shape mismatch".into());
    }
    let mut gradient = vec![0.0; rows * 3 * e];
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut grad_prob = vec![0.0; tokens];
    for b in 0..g.batch {
        for head in 0..g.heads {
            for query in 0..tokens {
                let prob_base = ((b * g.heads + head) * tokens + query) * tokens;
                let qbase = (b * tokens + query) * 3 * e + head * head_dim;
                let grad_out = (b * tokens + query) * e + head * head_dim;
                let mut softmax_dot = 0.0;
                for key in 0..tokens {
                    let vbase = (b * tokens + key) * 3 * e + 2 * e + head * head_dim;
                    let mut dp = 0.0;
                    for d in 0..head_dim {
                        dp += context_gradient[grad_out + d] * qkv[vbase + d];
                    }
                    grad_prob[key] = dp;
                    softmax_dot += dp * probabilities[prob_base + key];
                }
                for key in 0..tokens {
                    let p = probabilities[prob_base + key];
                    let ds = p * (grad_prob[key] - softmax_dot) * scale;
                    let kbase = (b * tokens + key) * 3 * e + e + head * head_dim;
                    let vbase = (b * tokens + key) * 3 * e + 2 * e + head * head_dim;
                    for d in 0..head_dim {
                        gradient[qbase + d] += ds * qkv[kbase + d];
                        gradient[kbase + d] += ds * qkv[qbase + d];
                        gradient[vbase + d] += p * context_gradient[grad_out + d];
                    }
                }
            }
        }
    }
    Ok(gradient)
}

fn block_backward(
    output_gradient: &[f32],
    weights: BlockWeightRefs<'_>,
    cache: &BlockCache,
    g: &DepthAnythingGeometry,
) -> Result<Vec<f32>, String> {
    let tokens = g.token_count()?;
    let rows = g.batch * tokens;
    let e = g.dim;
    let m = g.mlp_dim;
    if output_gradient.len() != rows * e {
        return Err("depth_anything block output gradient shape mismatch".into());
    }

    // Second residual and MLP branch.
    let mut grad_after_attention = output_gradient.to_vec();
    let mut grad_mlp = output_gradient.to_vec();
    for row in 0..rows {
        for channel in 0..e {
            grad_mlp[row * e + channel] *= weights.ls2[channel];
        }
    }
    let grad_fc1_gelu = linear_input_backward(&grad_mlp, weights.fc2_w, rows, m, e);
    let mut grad_fc1_pre = grad_fc1_gelu;
    for (gradient, pre_gelu) in grad_fc1_pre.iter_mut().zip(&cache.fc1_pre) {
        *gradient *= gelu_erf_grad(*pre_gelu);
    }
    let grad_ln2 = linear_input_backward(&grad_fc1_pre, weights.fc1_w, rows, e, m);
    let (grad_ln2_input, _, _) = layer_norm_backward(&grad_ln2, weights.ln2_gamma, &cache.ln2)
        .map_err(|error| format!("depth_anything block LayerNorm 2 backward: {error}"))?;
    for (dst, src) in grad_after_attention.iter_mut().zip(grad_ln2_input) {
        *dst += src;
    }

    // First residual and attention branch.
    let mut grad_projected = grad_after_attention.clone();
    for row in 0..rows {
        for channel in 0..e {
            grad_projected[row * e + channel] *= weights.ls1[channel];
        }
    }
    let grad_context = linear_input_backward(&grad_projected, weights.proj_w, rows, e, e);
    let grad_qkv = attention_backward(&grad_context, &cache.qkv, &cache.probabilities, g)?;
    let grad_ln1 = linear_input_backward(&grad_qkv, weights.qkv_w, rows, e, 3 * e);
    let (grad_ln1_input, _, _) = layer_norm_backward(&grad_ln1, weights.ln1_gamma, &cache.ln1)
        .map_err(|error| format!("depth_anything block LayerNorm 1 backward: {error}"))?;
    for (dst, src) in grad_after_attention.iter_mut().zip(grad_ln1_input) {
        *dst += src;
    }
    Ok(grad_after_attention)
}

/// Compute `W^T * dy` for row-major linear weights `[out_features,in_features]`.
fn linear_input_backward(
    output_gradient: &[f32],
    weights: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut input_gradient = vec![0.0; rows * in_features];
    for row in 0..rows {
        let grad_row = row * out_features;
        for output in 0..out_features {
            let dy = output_gradient[grad_row + output];
            let weight_row = output * in_features;
            for input in 0..in_features {
                input_gradient[row * in_features + input] += dy * weights[weight_row + input];
            }
        }
    }
    input_gradient
}

fn write_tap(
    normalized: &[f32],
    output: &mut [f32],
    g: &DepthAnythingGeometry,
    tap_index: usize,
    grid_h: usize,
    grid_w: usize,
) {
    let tokens = grid_h * grid_w + 1;
    let out_channels = g.taps.len() * g.dim;
    for b in 0..g.batch {
        for y in 0..grid_h {
            for x in 0..grid_w {
                let patch = y * grid_w + x;
                let token = (b * tokens + patch + 1) * g.dim;
                for e in 0..g.dim {
                    let channel = tap_index * g.dim + e;
                    let dst = ((b * out_channels + channel) * grid_h + y) * grid_w + x;
                    let value = normalized[token + e];
                    output[dst] = if g.output_clamp > 0.0 {
                        value.clamp(-g.output_clamp, g.output_clamp)
                    } else {
                        value
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(
        batch: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        dim: usize,
        depth: usize,
        heads: usize,
        mlp_dim: usize,
        pos_grid_h: usize,
        pos_grid_w: usize,
        taps: Vec<usize>,
    ) -> DepthAnythingGeometry {
        DepthAnythingGeometry {
            batch,
            height,
            width,
            patch_size,
            dim,
            depth,
            heads,
            mlp_dim,
            pos_grid_h,
            pos_grid_w,
            ln_eps: 1.0e-6,
            apply_imagenet_norm: false,
            imagenet_mean: [0.485, 0.456, 0.406],
            imagenet_std: [0.229, 0.224, 0.225],
            taps,
            output_clamp: 0.0,
        }
    }

    #[test]
    fn depth_anything_bicubic_pos_embed_matches_pytorch_reference() {
        let source = [1.0, 2.0, 3.0, 4.0];
        let output = bicubic_pos_embed(&source, 2, 2, 1, 4, 4).unwrap();
        let expected = [
            0.68359375, 1.015625, 1.5625, 1.89453125, 1.34765625, 1.6796875, 2.2265625, 2.55859375,
            2.44140625, 2.7734375, 3.3203125, 3.65234375, 3.10546875, 3.4375, 3.984375, 4.31640625,
        ];
        for (index, (actual, expected)) in output.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-5,
                "index {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn depth_anything_patchify_uses_nchw_channel_patch_y_patch_x_order() {
        let g = geometry(2, 2, 4, 2, 2, 1, 1, 2, 1, 2, vec![0]);
        g.validate().unwrap();
        let mut input = vec![0.0; g.input_len().unwrap()];
        for b in 0..2 {
            for c in 0..3 {
                for y in 0..2 {
                    for x in 0..4 {
                        input[((b * 3 + c) * 2 + y) * 4 + x] =
                            (b * 100 + c * 20 + y * 4 + x) as f32;
                    }
                }
            }
        }
        let patches = patchify_nchw_channel_major(&input, &g).unwrap();
        assert_eq!(
            &patches[..12],
            &[
                0.0, 1.0, 4.0, 5.0, 20.0, 21.0, 24.0, 25.0, 40.0, 41.0, 44.0, 45.0
            ]
        );
        assert_eq!(
            &patches[24..36],
            &[
                100.0, 101.0, 104.0, 105.0, 120.0, 121.0, 124.0, 125.0, 140.0, 141.0, 144.0, 145.0
            ]
        );
    }

    #[test]
    fn depth_anything_type_major_weight_arrays_match_source_order_and_lengths() {
        let g = geometry(1, 1, 1, 1, 2, 1, 1, 3, 1, 1, vec![0]);
        assert_eq!(
            weight_array_sizes(&g).unwrap(),
            [
                ("patch_w", 6),
                ("patch_b", 2),
                ("cls_token", 2),
                ("pos_embed", 4),
                ("norm_gamma", 2),
                ("norm_beta", 2),
                ("ln1_gamma", 2),
                ("ln1_beta", 2),
                ("qkv_w", 12),
                ("qkv_b", 6),
                ("proj_w", 4),
                ("proj_b", 2),
                ("ls1", 2),
                ("ln2_gamma", 2),
                ("ln2_beta", 2),
                ("fc1_w", 6),
                ("fc1_b", 3),
                ("fc2_w", 6),
                ("fc2_b", 2),
                ("ls2", 2),
            ]
        );
    }

    #[test]
    fn depth_anything_weights_round_trip_exact_source_blob_length() {
        let g = geometry(1, 4, 6, 2, 4, 2, 2, 7, 2, 3, vec![0, 1]);
        let count = g.weight_count().unwrap();
        let blob = (0..count).map(|i| i as f32 - 17.0).collect::<Vec<_>>();
        let weights = DepthAnythingWeights::from_blob(&blob, &g).unwrap();
        assert_eq!(weights.to_blob(), blob);
        assert!(DepthAnythingWeights::from_blob(&weights.as_slice()[..count - 1], &g).is_err());

        let zeroed = DepthAnythingWeights::zeroed(&g).unwrap();
        let offsets = weight_offsets(&g).unwrap();
        assert!(
            zeroed.as_slice()[offsets.ln1_gamma.range()]
                .iter()
                .all(|v| *v == 1.0)
        );
        assert!(
            zeroed.as_slice()[offsets.ln2_gamma.range()]
                .iter()
                .all(|v| *v == 1.0)
        );
        assert!(
            zeroed.as_slice()[offsets.ls1.range()]
                .iter()
                .all(|v| *v == 1.0)
        );
        assert!(
            zeroed.as_slice()[offsets.ls2.range()]
                .iter()
                .all(|v| *v == 1.0)
        );
    }

    #[test]
    fn depth_anything_tiny_transformer_matches_hand_computed_forward() {
        let g = geometry(1, 1, 1, 1, 2, 1, 1, 3, 1, 1, vec![0]);
        let mut weights = DepthAnythingWeights::zeroed(&g).unwrap();
        let offsets = weight_offsets(&g).unwrap();
        let values = weights.as_mut_slice();
        values[offsets.patch_w.range()].copy_from_slice(&[0.1, -0.2, 0.3, -0.4, 0.5, 0.2]);
        values[offsets.patch_b.range()].copy_from_slice(&[0.05, -0.03]);
        values[offsets.cls_token.range()].copy_from_slice(&[0.1, -0.2]);
        values[offsets.pos_embed.range()].copy_from_slice(&[0.02, 0.01, -0.03, 0.04]);
        values[offsets.norm_gamma.range()].copy_from_slice(&[0.9, 1.2]);
        values[offsets.norm_beta.range()].copy_from_slice(&[0.03, -0.06]);
        values[offsets.ln1_gamma.block(0)].copy_from_slice(&[0.8, 1.2]);
        values[offsets.ln1_beta.block(0)].copy_from_slice(&[-0.1, 0.07]);
        values[offsets.qkv_w.block(0)].copy_from_slice(&[
            0.3, -0.2, 0.1, 0.4, -0.1, 0.5, 0.2, 0.3, 0.7, -0.2, -0.4, 0.6,
        ]);
        values[offsets.qkv_b.block(0)].copy_from_slice(&[0.01, -0.02, 0.03, 0.04, -0.05, 0.02]);
        values[offsets.proj_w.block(0)].copy_from_slice(&[0.6, -0.3, 0.2, 0.5]);
        values[offsets.proj_b.block(0)].copy_from_slice(&[0.02, -0.01]);
        values[offsets.ls1.block(0)].copy_from_slice(&[0.4, 0.6]);
        values[offsets.ln2_gamma.block(0)].copy_from_slice(&[1.1, 0.7]);
        values[offsets.ln2_beta.block(0)].copy_from_slice(&[-0.05, 0.1]);
        values[offsets.fc1_w.block(0)].copy_from_slice(&[0.5, -0.2, -0.3, 0.4, 0.2, 0.3]);
        values[offsets.fc1_b.block(0)].copy_from_slice(&[0.01, -0.02, 0.03]);
        values[offsets.fc2_w.block(0)].copy_from_slice(&[0.4, -0.1, 0.2, -0.3, 0.5, 0.1]);
        values[offsets.fc2_b.block(0)].copy_from_slice(&[0.02, -0.04]);
        values[offsets.ls2.block(0)].copy_from_slice(&[0.2, 0.3]);

        let output = forward(&[0.2, -0.4, 0.7], &weights, &g).unwrap().0;
        // Independent two-token attention, population LayerNorm, and erf-GELU calculation.
        let expected = [0.92999844, -1.25999792];
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn depth_anything_taps_are_batch_major_and_tap_major_nchw() {
        let g = geometry(2, 2, 3, 1, 3, 2, 1, 4, 2, 3, vec![0, 1]);
        let mut weights = DepthAnythingWeights::zeroed(&g).unwrap();
        let offsets = weight_offsets(&g).unwrap();
        let values = weights.as_mut_slice();
        // Patch projection is RGB identity. Block zero is an identity residual;
        // block one's channel-specific FC2 bias distinguishes tap positions.
        for e in 0..3 {
            values[offsets.patch_w.start + e * 3 + e] = 1.0;
        }
        values[offsets.fc2_b.block(1)].copy_from_slice(&[0.2, -0.1, 0.35]);
        values[offsets.norm_gamma.range()].fill(1.0);
        values[offsets.norm_beta.range()].fill(0.0);
        let mut input = vec![0.0; g.input_len().unwrap()];
        for b in 0..g.batch {
            for y in 0..g.height {
                for x in 0..g.width {
                    let p = y * g.width + x;
                    input[(b * 3) * 6 + p] = (b * 20 + p + 1) as f32;
                    input[(b * 3 + 1) * 6 + p] = (b * 7 + 2 * p + 3) as f32;
                    input[(b * 3 + 2) * 6 + p] = (5 * b + p * p + 2) as f32;
                }
            }
        }
        let (out, _) = forward(&input, &weights, &g).unwrap();
        assert_eq!(out.len(), 2 * 2 * 3 * 2 * 3);
        for b in 0..g.batch {
            for y in 0..2 {
                for x in 0..3 {
                    let p = y * 3 + x;
                    let pixel = [
                        input[(b * 3) * 6 + p],
                        input[(b * 3 + 1) * 6 + p],
                        input[(b * 3 + 2) * 6 + p],
                    ];
                    for (tap, bias) in [[0.0, 0.0, 0.0], [0.2, -0.1, 0.35]].into_iter().enumerate()
                    {
                        let shifted = [pixel[0] + bias[0], pixel[1] + bias[1], pixel[2] + bias[2]];
                        let mean = shifted.iter().sum::<f32>() / 3.0;
                        let var = shifted
                            .iter()
                            .map(|value| (value - mean).powi(2))
                            .sum::<f32>()
                            / 3.0;
                        let inv = 1.0 / (var + g.ln_eps).sqrt();
                        for c in 0..3 {
                            let channel = tap * 3 + c;
                            let index = ((b * 6 + channel) * 2 + y) * 3 + x;
                            let expected = (shifted[c] - mean) * inv;
                            assert!((out[index] - expected).abs() < 2.0e-5, "index {index}");
                        }
                    }
                }
            }
        }
        let duplicate_taps = geometry(1, 2, 2, 1, 2, 1, 1, 2, 1, 1, vec![0, 0]);
        assert!(duplicate_taps.validate().is_ok());
    }

    #[test]
    fn depth_anything_output_clamp_saturates_taps_and_zeros_the_derivative() {
        let mut g = geometry(1, 1, 1, 1, 2, 1, 1, 2, 1, 1, vec![0]);
        g.output_clamp = 0.5;
        let mut weights = DepthAnythingWeights::zeroed(&g).unwrap();
        let offsets = weight_offsets(&g).unwrap();
        weights.as_mut_slice()[offsets.norm_beta.range()].copy_from_slice(&[3.0, -3.0]);
        let (output, cache) = forward(&[0.1, -0.3, 0.7], &weights, &g).unwrap();
        assert_eq!(output, [0.5, -0.5]);
        let input_gradient = backward(&[1.0, 1.0], &weights, &g, &cache).unwrap();
        assert_eq!(input_gradient, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn depth_anything_backward_matches_input_central_finite_differences() {
        let mut g = geometry(1, 2, 3, 1, 4, 2, 2, 5, 2, 3, vec![0, 1]);
        g.apply_imagenet_norm = true;
        g.output_clamp = 0.0;
        let mut blob = DepthAnythingWeights::zeroed(&g).unwrap().to_blob();
        for (i, value) in blob.iter_mut().enumerate() {
            let phase = (i % 29) as f32;
            *value = (phase * 0.37).sin() * 0.08;
        }
        let weights = DepthAnythingWeights::from_blob(&blob, &g).unwrap();
        let input = (0..g.input_len().unwrap())
            .map(|i| 0.03 + i as f32 * 0.017)
            .collect::<Vec<_>>();
        let (output, cache) = forward(&input, &weights, &g).unwrap();
        let seed = (0..output.len())
            .map(|i| ((i * 7 % 19) as f32 - 8.0) * 0.03)
            .collect::<Vec<_>>();
        let analytic = backward(&seed, &weights, &g, &cache).unwrap();
        for input_index in [0, 3, 7, 11, 15, 17] {
            let epsilon = 1.0e-3;
            let mut plus = input.clone();
            let mut minus = input.clone();
            plus[input_index] += epsilon;
            minus[input_index] -= epsilon;
            let plus_out = forward(&plus, &weights, &g).unwrap().0;
            let minus_out = forward(&minus, &weights, &g).unwrap().0;
            let plus_loss = plus_out.iter().zip(&seed).map(|(a, b)| a * b).sum::<f32>();
            let minus_loss = minus_out.iter().zip(&seed).map(|(a, b)| a * b).sum::<f32>();
            let numeric = (plus_loss - minus_loss) / (2.0 * epsilon);
            let error = (analytic[input_index] - numeric).abs();
            assert!(
                error < 2.5e-3 + numeric.abs() * 2.0e-2,
                "input {input_index}: analytic={} numeric={numeric} abs_error={error}",
                analytic[input_index]
            );
        }
    }

    #[test]
    fn depth_anything_geometry_rejects_invalid_taps_and_nondivisible_input() {
        let mut g = geometry(1, 3, 4, 2, 4, 2, 2, 8, 1, 1, vec![0]);
        assert!(g.validate().is_err());
        g.height = 4;
        g.taps = vec![2];
        assert!(g.validate().is_err());
    }
}

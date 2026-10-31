//! Geometry and weight layout for depth_anything.

#[doc(hidden)]
pub const RGB_CHANNELS: usize = 3;

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
    #[doc(hidden)]
    pub values: Vec<f32>,
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
#[doc(hidden)]
pub struct WeightField {
    #[doc(hidden)]
    pub start: usize,
    #[doc(hidden)]
    pub len: usize,
    #[doc(hidden)]
    pub stride: usize,
}

impl WeightField {
    #[doc(hidden)]
    pub fn range(self) -> std::ops::Range<usize> {
        self.start..self.start + self.len
    }

    #[doc(hidden)]
    pub fn block(self, index: usize) -> std::ops::Range<usize> {
        let start = self.start + index * self.stride;
        start..start + self.stride
    }
}

#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct WeightOffsets {
    #[doc(hidden)]
    pub patch_w: WeightField,
    #[doc(hidden)]
    pub patch_b: WeightField,
    #[doc(hidden)]
    pub cls_token: WeightField,
    #[doc(hidden)]
    pub pos_embed: WeightField,
    #[doc(hidden)]
    pub norm_gamma: WeightField,
    #[doc(hidden)]
    pub norm_beta: WeightField,
    #[doc(hidden)]
    pub ln1_gamma: WeightField,
    #[doc(hidden)]
    pub ln1_beta: WeightField,
    #[doc(hidden)]
    pub qkv_w: WeightField,
    #[doc(hidden)]
    pub qkv_b: WeightField,
    #[doc(hidden)]
    pub proj_w: WeightField,
    #[doc(hidden)]
    pub proj_b: WeightField,
    #[doc(hidden)]
    pub ls1: WeightField,
    #[doc(hidden)]
    pub ln2_gamma: WeightField,
    #[doc(hidden)]
    pub ln2_beta: WeightField,
    #[doc(hidden)]
    pub fc1_w: WeightField,
    #[doc(hidden)]
    pub fc1_b: WeightField,
    #[doc(hidden)]
    pub fc2_w: WeightField,
    #[doc(hidden)]
    pub fc2_b: WeightField,
    #[doc(hidden)]
    pub ls2: WeightField,
    #[doc(hidden)]
    pub total: usize,
}

#[doc(hidden)]
pub fn weight_offsets(g: &DepthAnythingGeometry) -> Result<WeightOffsets, String> {
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

#[doc(hidden)]
pub fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("depth_anything {what} overflows usize"))
}

fn checked_add(a: usize, b: usize, what: &str) -> Result<usize, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("depth_anything {what} overflows usize"))
}

#[doc(hidden)]
pub fn checked_product(values: &[usize], what: &str) -> Result<usize, String> {
    values
        .iter()
        .try_fold(1usize, |n, value| checked_mul(n, *value, what))
}


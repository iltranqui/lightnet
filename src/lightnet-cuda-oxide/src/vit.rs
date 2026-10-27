//! Native ViT tensor layout and residual primitives.

use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::{OxideBackend, OxideError, OxideTensorF32, launch, launch_err, shape_err};

impl OxideBackend {
    /// ViT patch gather to `[batch * tokens, patch * patch * channels]`.
    pub fn vit_patchify_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(usize, usize, OxideTensorF32), OxideError> {
        let op = "vit_patchify_f32";
        if batch == 0 || channels == 0 || height == 0 || width == 0 || patch_size == 0 || patch_stride == 0 {
            return Err(shape_err(op, "dimensions, patch size and stride must be positive".to_owned()));
        }
        let padded_h = height + 2 * pad;
        let padded_w = width + 2 * pad;
        if padded_h < patch_size
            || padded_w < patch_size
            || (padded_h - patch_size) % patch_stride != 0
            || (padded_w - patch_size) % patch_stride != 0
        {
            return Err(shape_err(op, "patch extent does not tile the padded input".to_owned()));
        }
        let out_h = (padded_h - patch_size) / patch_stride + 1;
        let out_w = (padded_w - patch_size) / patch_stride + 1;
        let patch_dim = patch_size * patch_size * channels;
        let rows = batch * out_h * out_w;
        let output_len = rows
            .checked_mul(patch_dim)
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let input = self.f32v(input)?;
        if input.shape() != [batch, channels, height, width] {
            return Err(shape_err(op, "input shape mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: every output element is written once from a bounds-checked source.
        unsafe {
            self.vit.vit_patchify(
                &self.stream,
                launch::elementwise(output_len, config),
                &input.buf,
                &mut output,
                channels,
                height,
                width,
                patch_size,
                patch_stride,
                pad,
                out_h,
                out_w,
            )
        }
        .map_err(launch_err)?;
        Ok((out_h, out_w, OxideTensorF32::new(output, &[rows, patch_dim])))
    }

    /// Add a `[tokens, hidden]` table to each sample of `[batch * tokens, hidden]`.
    pub fn vit_add_pos_embed_f32(
        &self,
        tokens: &OxideTensorF32,
        pos: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_add_pos_embed_f32";
        let tokens = self.f32v(tokens)?;
        let pos = self.f32v(pos)?;
        if pos.len() == 0 || tokens.len() % pos.len() != 0 {
            return Err(shape_err(op, "token length is not a multiple of the table".to_owned()));
        }
        let len = tokens.len();
        let shape = tokens.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_add_pos_embed(
                &self.stream,
                launch::elementwise(len, config),
                &tokens.buf,
                &pos.buf,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// Finite clamp used by the ViT feature guards (non-finite becomes zero).
    pub fn vit_clamp_feature_f32(
        &self,
        input: &OxideTensorF32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_clamp_feature_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let input = self.f32v(input)?;
        let len = input.len();
        let shape = input.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_clamp_feature(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                clip,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// ViT mHC residual mix with host-computed scalar coefficients.
    pub fn vit_mhc_residual_f32(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_mhc_residual_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let skip = self.f32v(skip)?;
        let branch = self.f32v(branch)?;
        if skip.len() != branch.len() {
            return Err(shape_err(op, "skip and branch lengths differ".to_owned()));
        }
        let len = skip.len();
        let shape = skip.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_mhc_residual(
                &self.stream,
                launch::elementwise(len, config),
                &skip.buf,
                &branch.buf,
                &mut output,
                skip_scale,
                branch_scale,
                clip,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// `[batch * tokens, hidden]` token rows to NCHW `[batch, hidden, out_h, out_w]`.
    pub fn vit_tokens_to_nchw_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_tokens_to_nchw_f32";
        let tokens = out_h * out_w;
        if batch == 0 || tokens == 0 || hidden == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let len = batch * tokens * hidden;
        let input = self.f32v(input)?;
        if input.len() != len {
            return Err(shape_err(op, "input length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_tokens_to_nchw(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                tokens,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, hidden, out_h, out_w]))
    }

    /// NCHW `[batch, hidden, out_h, out_w]` to `[batch * tokens, hidden]` rows.
    pub fn vit_nchw_to_tokens_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        out_h: usize,
        out_w: usize,
        hidden: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_nchw_to_tokens_f32";
        let tokens = out_h * out_w;
        if batch == 0 || tokens == 0 || hidden == 0 {
            return Err(shape_err(op, "dimensions must be positive".to_owned()));
        }
        let len = batch * tokens * hidden;
        let input = self.f32v(input)?;
        if input.len() != len {
            return Err(shape_err(op, "input length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_nchw_to_tokens(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                tokens,
                hidden,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch * tokens, hidden]))
    }

    /// Adjoint of [`Self::vit_patchify_f32`]: `[batch * tokens, patch_dim]`
    /// patch gradients accumulated into an NCHW input gradient.
    pub fn vit_patch_scatter_f32(
        &self,
        patch_grad: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        patch_size: usize,
        patch_stride: usize,
        pad: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_patch_scatter_f32";
        if batch == 0
            || channels == 0
            || height == 0
            || width == 0
            || patch_size == 0
            || patch_stride == 0
            || out_h == 0
            || out_w == 0
        {
            return Err(shape_err(op, "dimensions, patch size and stride must be positive".to_owned()));
        }
        let patch_dim = patch_size * patch_size * channels;
        let patch_len = batch
            .checked_mul(out_h * out_w)
            .and_then(|n| n.checked_mul(patch_dim))
            .ok_or_else(|| shape_err(op, "patch gradient size overflows".to_owned()))?;
        let output_len = batch
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(height))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| shape_err(op, "output element count overflows".to_owned()))?;
        let patch_grad = self.f32v(patch_grad)?;
        if patch_grad.len() != patch_len {
            return Err(shape_err(op, "patch gradient length mismatch".to_owned()));
        }
        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: one thread owns each output pixel and reads only in-range taps.
        unsafe {
            self.vit.vit_patch_scatter(
                &self.stream,
                launch::elementwise(output_len, config),
                &patch_grad.buf,
                &mut output,
                channels,
                height,
                width,
                patch_size,
                patch_stride,
                pad,
                out_h,
                out_w,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, channels, height, width]))
    }

    /// Adjoint of [`Self::vit_clamp_feature_f32`] evaluated at its raw input.
    pub fn vit_clamp_backward_f32(
        &self,
        raw: &OxideTensorF32,
        grad: &OxideTensorF32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_clamp_backward_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let raw = self.f32v(raw)?;
        let grad = self.f32v(grad)?;
        if raw.len() != grad.len() {
            return Err(shape_err(op, "raw and gradient lengths differ".to_owned()));
        }
        let len = grad.len();
        let shape = grad.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.vit.vit_clamp_backward(
                &self.stream,
                launch::elementwise(len, config),
                &raw.buf,
                &grad.buf,
                &mut output,
                clip,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    /// Adjoint of [`Self::vit_mhc_residual_f32`]: `(d_skip, d_branch, sums)`,
    /// where `sums` holds the `[grad_post_skip, grad_post_branch, grad_mix]`
    /// reductions the host turns into the six mHC parameter gradients.
    pub fn vit_mhc_residual_backward_f32(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        grad: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "vit_mhc_residual_backward_f32";
        if !clip.is_finite() || clip <= 0.0 {
            return Err(shape_err(op, "clip must be positive and finite".to_owned()));
        }
        let skip = self.f32v(skip)?;
        let branch = self.f32v(branch)?;
        let grad = self.f32v(grad)?;
        if skip.len() != branch.len() || skip.len() != grad.len() || grad.len() == 0 {
            return Err(shape_err(op, "skip, branch and gradient lengths differ".to_owned()));
        }
        let len = grad.len();
        let shape = grad.shape().to_vec();
        let mut d_skip = self.alloc_uninit(len)?;
        let mut d_branch = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each element of both outputs.
        unsafe {
            self.vit.vit_mhc_residual_backward(
                &self.stream,
                launch::elementwise(len, config),
                &skip.buf,
                &branch.buf,
                &grad.buf,
                &mut d_skip,
                &mut d_branch,
                skip_scale,
                branch_scale,
                clip,
            )
        }
        .map_err(launch_err)?;
        let mut terms = self.alloc_uninit(3 * len)?;
        // SAFETY: one thread owns each of the `3 * len` planar term elements.
        unsafe {
            self.vit.vit_mhc_param_terms(
                &self.stream,
                launch::elementwise(3 * len, config),
                &skip.buf,
                &branch.buf,
                &grad.buf,
                &mut terms,
                skip_scale,
                branch_scale,
                mix,
                post_skip,
                post_branch,
                clip,
            )
        }
        .map_err(launch_err)?;
        // Planar `[3, len]` terms: three channels of stride `len`, one row group.
        let terms = OxideTensorF32::new(terms, &[3, len]);
        let sums = <Self as F32TensorOpsBackend>::bias_backward_f32(self, &terms, 3, len, config)?;
        Ok((
            OxideTensorF32::new(d_skip, &shape),
            OxideTensorF32::new(d_branch, &shape),
            sums,
        ))
    }

    /// Sum a `[batch, table]` token gradient over the batch (learned positions).
    pub fn vit_pos_embed_backward_f32(
        &self,
        grad: &OxideTensorF32,
        batch: usize,
        table: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "vit_pos_embed_backward_f32";
        if batch == 0 || table == 0 {
            return Err(shape_err(op, "batch and table must be positive".to_owned()));
        }
        let grad = self.f32v(grad)?;
        if Some(grad.len()) != batch.checked_mul(table) {
            return Err(shape_err(op, "gradient length is not batch * table".to_owned()));
        }
        let mut output = self.alloc_uninit(table)?;
        // SAFETY: one thread owns each table entry.
        unsafe {
            self.vit.vit_pos_embed_backward(
                &self.stream,
                launch::elementwise(table, config),
                &grad.buf,
                &mut output,
                batch,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[table]))
    }

}

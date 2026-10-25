//! Native `[swin]` backend ops: window partition/unpartition with the cyclic
//! shift and padding, the plain mHC mix and its adjoint, and windowed
//! attention with relative-position bias plus its adjoint. The trait methods
//! in `lib.rs` forward here; kernels live in `kernels/swin.rs`.

use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::launch;
use crate::{OxideBackend, OxideError, OxideTensorF32, launch_err};

fn shape_err(op: &str, msg: &str) -> OxideError {
    OxideError::Shape(format!("{op}: {msg}"))
}

fn product(op: &str, factors: &[usize]) -> Result<usize, OxideError> {
    factors
        .iter()
        .try_fold(1usize, |acc, &f| acc.checked_mul(f))
        .ok_or_else(|| shape_err(op, "element count overflows"))
}

/// Validate the spatial window geometry shared by partition and unpartition;
/// returns `(spatial_windows, tokens)`.
#[allow(clippy::too_many_arguments)]
fn window_geometry(
    op: &str,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    padded_height: usize,
    padded_width: usize,
    window: usize,
    shift: usize,
) -> Result<(usize, usize), OxideError> {
    if batch == 0 || channels == 0 || height == 0 || width == 0 || window == 0 {
        return Err(shape_err(op, "dimensions and window must be positive"));
    }
    if padded_height < height
        || padded_width < width
        || padded_height % window != 0
        || padded_width % window != 0
    {
        return Err(shape_err(op, "padded extents must cover the input and be window-aligned"));
    }
    if shift >= window {
        return Err(shape_err(op, "shift must be smaller than the window"));
    }
    let windows = (padded_height / window) * (padded_width / window);
    Ok((windows, window * window))
}

/// Validate the attention geometry; returns `(tokens, rows, matrix_len, table_len)`.
fn attention_geometry(
    op: &str,
    total_windows: usize,
    window: usize,
    channels: usize,
    heads: usize,
) -> Result<(usize, usize, usize, usize), OxideError> {
    if total_windows == 0 || window == 0 || channels == 0 || heads == 0 || channels % heads != 0 {
        return Err(shape_err(
            op,
            "windows, window, channels and heads must be positive with channels divisible by heads",
        ));
    }
    let tokens = window * window;
    let rows = product(op, &[total_windows, tokens])?;
    let matrix_len = product(op, &[total_windows, heads, tokens, tokens])?;
    let table_width = 2 * window - 1;
    let table_len = product(op, &[heads, table_width, table_width])?;
    Ok((tokens, rows, matrix_len, table_len))
}

impl OxideBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn swin_window_partition_native(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "swin_window_partition_f32";
        let (windows, tokens) = window_geometry(
            OP,
            batch,
            channels,
            height,
            width,
            padded_height,
            padded_width,
            window,
            shift,
        )?;
        let input = self.f32v(input)?;
        if input.len() != product(OP, &[batch, channels, height, width])? {
            return Err(shape_err(OP, "input length mismatch"));
        }
        let rows = product(OP, &[batch, windows, tokens])?;
        let len = product(OP, &[rows, channels])?;
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element; reads stay in the
        // validated input extent.
        unsafe {
            self.swin.swin_window_partition(
                &self.stream,
                launch::elementwise(len, config),
                &input.buf,
                &mut output,
                channels,
                height,
                width,
                padded_height,
                padded_width,
                window,
                shift,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[rows, channels]))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn swin_window_unpartition_native(
        &self,
        windows: &OxideTensorF32,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        padded_height: usize,
        padded_width: usize,
        window: usize,
        shift: usize,
        clip: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "swin_window_unpartition_f32";
        let (spatial_windows, tokens) = window_geometry(
            OP,
            batch,
            channels,
            height,
            width,
            padded_height,
            padded_width,
            window,
            shift,
        )?;
        if !clip.is_finite() || clip < 0.0 {
            return Err(shape_err(OP, "clip must be finite and non-negative"));
        }
        let windows = self.f32v(windows)?;
        if windows.len() != product(OP, &[batch, spatial_windows, tokens, channels])? {
            return Err(shape_err(OP, "window tensor length mismatch"));
        }
        let len = product(OP, &[batch, channels, height, width])?;
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output pixel and reads one in-range
        // window element.
        unsafe {
            self.swin.swin_window_unpartition(
                &self.stream,
                launch::elementwise(len, config),
                &windows.buf,
                &mut output,
                channels,
                height,
                width,
                padded_height,
                padded_width,
                window,
                shift,
                clip,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &[batch, channels, height, width]))
    }

    pub(crate) fn swin_mhc_residual_native(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "swin_mhc_residual_f32";
        let skip = self.f32v(skip)?;
        let branch = self.f32v(branch)?;
        if skip.len() != branch.len() || skip.is_empty() {
            return Err(shape_err(OP, "skip and branch lengths differ or are empty"));
        }
        let len = skip.len();
        let shape = skip.shape().to_vec();
        let mut output = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each output element.
        unsafe {
            self.swin.swin_mhc_residual(
                &self.stream,
                launch::elementwise(len, config),
                &skip.buf,
                &branch.buf,
                &mut output,
                skip_scale,
                branch_scale,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, &shape))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn swin_mhc_residual_backward_native(
        &self,
        skip: &OxideTensorF32,
        branch: &OxideTensorF32,
        grad: &OxideTensorF32,
        skip_scale: f32,
        branch_scale: f32,
        mix: f32,
        post_skip: f32,
        post_branch: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        const OP: &str = "swin_mhc_residual_backward_f32";
        let skip = self.f32v(skip)?;
        let branch = self.f32v(branch)?;
        let grad = self.f32v(grad)?;
        if skip.len() != branch.len() || skip.len() != grad.len() || grad.is_empty() {
            return Err(shape_err(OP, "skip, branch and gradient lengths differ"));
        }
        let len = grad.len();
        let shape = grad.shape().to_vec();
        let mut d_skip = self.alloc_uninit(len)?;
        let mut d_branch = self.alloc_uninit(len)?;
        // SAFETY: one thread owns each element of both outputs.
        unsafe {
            self.swin.swin_mhc_residual_backward(
                &self.stream,
                launch::elementwise(len, config),
                &grad.buf,
                &mut d_skip,
                &mut d_branch,
                skip_scale,
                branch_scale,
            )
        }
        .map_err(launch_err)?;
        let terms_len = product(OP, &[3, len])?;
        let mut terms = self.alloc_uninit(terms_len)?;
        // SAFETY: one thread owns each of the `3 * len` planar term elements.
        unsafe {
            self.swin.swin_mhc_param_terms(
                &self.stream,
                launch::elementwise(terms_len, config),
                &skip.buf,
                &branch.buf,
                &grad.buf,
                &mut terms,
                mix,
                post_skip,
                post_branch,
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn swin_window_attention_native(
        &self,
        qkv: &OxideTensorF32,
        relative_position_bias: &OxideTensorF32,
        mask: &OxideTensorF32,
        total_windows: usize,
        spatial_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        const OP: &str = "swin_window_attention_f32";
        let (tokens, rows, matrix_len, table_len) =
            attention_geometry(OP, total_windows, window, channels, heads)?;
        if spatial_windows == 0 || total_windows % spatial_windows != 0 {
            return Err(shape_err(OP, "total windows must be a multiple of the spatial windows"));
        }
        let qkv = self.f32v(qkv)?;
        let bias = self.f32v(relative_position_bias)?;
        let mask = self.f32v(mask)?;
        if qkv.len() != product(OP, &[rows, 3, channels])? {
            return Err(shape_err(OP, "QKV length mismatch"));
        }
        if bias.len() != table_len {
            return Err(shape_err(OP, "relative-position bias length mismatch"));
        }
        if mask.len() != product(OP, &[spatial_windows, tokens, tokens])? {
            return Err(shape_err(OP, "mask length mismatch"));
        }
        let threads = product(OP, &[total_windows, heads, tokens])?;
        let mut context = self.alloc_uninit(product(OP, &[rows, channels])?)?;
        let mut probabilities = self.alloc_uninit(matrix_len)?;
        let mut active = self.alloc_uninit(matrix_len)?;
        // SAFETY: each (window, head, query) thread writes its own head
        // channel interval of one context row and its own probability and
        // activity rows; reads stay in the validated extents.
        unsafe {
            self.swin.swin_window_attention_forward(
                &self.stream,
                launch::elementwise(threads, config),
                &qkv.buf,
                &bias.buf,
                &mask.buf,
                &mut context,
                &mut probabilities,
                &mut active,
                total_windows,
                spatial_windows,
                window,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        let matrix_shape = [total_windows, heads, tokens, tokens];
        Ok((
            OxideTensorF32::new(context, &[rows, channels]),
            OxideTensorF32::new(probabilities, &matrix_shape),
            OxideTensorF32::new(active, &matrix_shape),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn swin_window_attention_backward_native(
        &self,
        qkv: &OxideTensorF32,
        probabilities: &OxideTensorF32,
        active: &OxideTensorF32,
        context_grad: &OxideTensorF32,
        total_windows: usize,
        window: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        const OP: &str = "swin_window_attention_backward_f32";
        let (tokens, rows, matrix_len, table_len) =
            attention_geometry(OP, total_windows, window, channels, heads)?;
        let qkv = self.f32v(qkv)?;
        let probabilities = self.f32v(probabilities)?;
        let active = self.f32v(active)?;
        let context_grad = self.f32v(context_grad)?;
        if qkv.len() != product(OP, &[rows, 3, channels])? {
            return Err(shape_err(OP, "QKV length mismatch"));
        }
        if probabilities.len() != matrix_len || active.len() != matrix_len {
            return Err(shape_err(OP, "probability or activity length mismatch"));
        }
        let context_len = product(OP, &[rows, channels])?;
        if context_grad.len() != context_len {
            return Err(shape_err(OP, "context gradient length mismatch"));
        }
        let threads = product(OP, &[total_windows, heads, tokens])?;
        let mut score_grad = self.alloc_uninit(matrix_len)?;
        let mut query_grad = self.alloc_uninit(context_len)?;
        // SAFETY: each (window, head, query) thread owns one score-gradient
        // row and one head channel interval of one query-gradient row.
        unsafe {
            self.swin.swin_window_attention_backward_rows(
                &self.stream,
                launch::elementwise(threads, config),
                &qkv.buf,
                &probabilities.buf,
                &active.buf,
                &context_grad.buf,
                &mut score_grad,
                &mut query_grad,
                total_windows,
                window,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        let mut qkv_grad = self.alloc_uninit(product(OP, &[rows, 3, channels])?)?;
        // SAFETY: each (window, head, key) thread owns that head's Q, K and V
        // channel intervals of one fused gradient row.
        unsafe {
            self.swin.swin_window_attention_backward_cols(
                &self.stream,
                launch::elementwise(threads, config),
                &qkv.buf,
                &probabilities.buf,
                &score_grad,
                &context_grad.buf,
                &query_grad,
                &mut qkv_grad,
                total_windows,
                window,
                channels,
                heads,
            )
        }
        .map_err(launch_err)?;
        let mut bias_grad = self.alloc_uninit(table_len)?;
        // SAFETY: one thread owns each bias-table entry.
        unsafe {
            self.swin.swin_relative_bias_backward(
                &self.stream,
                launch::elementwise(table_len, config),
                &score_grad,
                &mut bias_grad,
                total_windows,
                window,
                heads,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(qkv_grad, &[rows, 3 * channels]),
            OxideTensorF32::new(bias_grad, &[table_len]),
        ))
    }
}

//! Deformable convolution backend operations.

use lightnet_cuda::{BilinearGatherGeometry, ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::{deform_gather_shape, launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    /// Device `[deform_conv]` **accumulation stage** over prepared offsets and
    /// masks. Per `(b, filter, oy, ox)` it sums
    /// `weight[f, c, k] * bilinear_sample(input, base + offset) * mask` over the
    /// group's channels and kernel points, then applies the source `sanitize`
    /// clamp.
    ///
    /// Deliberately not wired into the runtime yet: the predictor columns,
    /// per-batch conditional offset normalization and the lhtan/sigmoid
    /// envelopes are the caller's responsibility, and the CPU forward/backward
    /// remains the reference. `layer-support.md` keeps `[deform_conv]` at
    /// `host` / `cpu` until the full forward is assembled and tested.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_accumulate(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        masks: &OxideTensorF32,
        weights: &OxideTensorF32,
        channels: usize,
        filters: usize,
        channels_per_group: usize,
        filters_per_group: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        use_mask: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_conv_accumulate";
        if channels == 0
            || filters == 0
            || in_h == 0
            || in_w == 0
            || out_h == 0
            || out_w == 0
            || size == 0
            || channels_per_group == 0
            || filters_per_group == 0
        {
            return Err(shape_err(op, "all dimensions must be positive".to_owned()));
        }
        if channels % channels_per_group != 0 || filters % filters_per_group != 0 {
            return Err(shape_err(
                op,
                "groups must divide both input channels and filters".to_owned(),
            ));
        }
        let kernels = size
            .checked_mul(size)
            .ok_or_else(|| shape_err(op, "kernel area overflows".to_owned()))?;
        let plane = out_h * out_w;
        let in_plane = in_h * in_w;
        let per_batch = channels
            .checked_mul(in_plane)
            .ok_or_else(|| shape_err(op, "input channel count overflows".to_owned()))?;
        if input.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, not a multiple of channels*H*W",
                    input.len()
                ),
            ));
        }
        let batch = input.len() / per_batch;
        let offset_len = batch * 2 * kernels * plane;
        if offsets.len() != offset_len {
            return Err(shape_err(
                op,
                format!(
                    "offsets have {} elements, expected {offset_len}",
                    offsets.len()
                ),
            ));
        }
        if use_mask && masks.len() != batch * kernels * plane {
            return Err(shape_err(
                op,
                format!(
                    "masks have {} elements, expected {}",
                    masks.len(),
                    batch * kernels * plane
                ),
            ));
        }
        let weight_len = filters
            .checked_mul(channels_per_group)
            .and_then(|v| v.checked_mul(kernels))
            .ok_or_else(|| shape_err(op, "weight count overflows".to_owned()))?;
        if weights.len() != weight_len {
            return Err(shape_err(
                op,
                format!(
                    "weights have {} elements, expected {weight_len}",
                    weights.len()
                ),
            ));
        }
        let out_len = batch
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(plane))
            .ok_or_else(|| shape_err(op, "output count overflows".to_owned()))?;
        let mut out = self.alloc_uninit(out_len)?;
        // SAFETY: one thread per output element; every input/offset/mask/weight
        // index is inside a buffer validated above and inside the kernel's
        // own in-image checks.
        unsafe {
            self.deform_sample.deform_conv_accumulate(
                &self.stream,
                launch::elementwise(out_len, config),
                &input.buf,
                &offsets.buf,
                &masks.buf,
                &weights.buf,
                &mut out,
                channels,
                filters,
                channels_per_group,
                filters_per_group,
                in_h,
                in_w,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding,
                dilation,
                u32::from(use_mask),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[batch, filters, out_h, out_w]))
    }

    /// `[deform_conv]` offset/mask predictor: a plain convolution of the input
    /// with the predictor weights plus its bias — i.e. `W * im2col(x) + b`, the
    /// same quantity `deform_conv::predictor_columns` produces. `groups` is not
    /// a parameter: the source predictor is always dense over all input
    /// channels (`offset_filters = 2*size*size`, `mask_filters = size*size`).
    ///
    /// Dilation greater than one uses an explicitly-shaped native predictor
    /// kernel because DeformConv's output extent differs from `[convolutional]`.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_offset_predict(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        input_channels: usize,
        in_h: usize,
        in_w: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_offset_predict";
        if input_channels == 0
            || in_h == 0
            || in_w == 0
            || filters == 0
            || size == 0
            || out_h == 0
            || out_w == 0
            || stride_x == 0
            || stride_y == 0
            || dilation == 0
        {
            return Err(shape_err(
                op,
                "all dimensions, strides and dilation must be positive".to_owned(),
            ));
        }
        let per_batch = input_channels
            .checked_mul(in_h)
            .and_then(|v| v.checked_mul(in_w))
            .ok_or_else(|| shape_err(op, "input element count overflows".to_owned()))?;
        if input.len() == 0 || input.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, not a multiple of C*H*W = {per_batch}",
                    input.len()
                ),
            ));
        }
        let batch = input.len() / per_batch;
        if input.shape() != [batch, input_channels, in_h, in_w] {
            return Err(shape_err(
                op,
                format!(
                    "input tensor shape {:?} does not match predictor geometry",
                    input.shape()
                ),
            ));
        }
        let weight_len = filters
            .checked_mul(input_channels)
            .and_then(|v| v.checked_mul(size))
            .and_then(|v| v.checked_mul(size))
            .ok_or_else(|| shape_err(op, "weight count overflows".to_owned()))?;
        if weights.len() != weight_len {
            return Err(shape_err(
                op,
                format!(
                    "weights have {} elements, expected {weight_len}",
                    weights.len()
                ),
            ));
        }
        if biases.len() != filters {
            return Err(shape_err(
                op,
                format!("biases have {} elements, expected {filters}", biases.len()),
            ));
        }
        let shape = [batch, filters, out_h, out_w];
        let convolved = if dilation == 1 {
            self.conv2d_f32(
                input,
                weights,
                &shape,
                input_channels,
                in_h,
                in_w,
                filters,
                size,
                stride_x,
                stride_y,
                dilation,
                padding,
                1,
                config,
            )?
        } else {
            let output_len = batch
                .checked_mul(filters)
                .and_then(|n| n.checked_mul(out_h))
                .and_then(|n| n.checked_mul(out_w))
                .ok_or_else(|| shape_err(op, "output length overflows".to_owned()))?;
            let input = self.f32v(input)?;
            let weights = self.f32v(weights)?;
            let mut output = self.alloc_uninit(output_len)?;
            // SAFETY: checked operand lengths and explicit output shape bound all taps.
            unsafe {
                self.deform_backward.deform_predictor_forward_dilated(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &input.buf,
                    &weights.buf,
                    &mut output,
                    batch,
                    input_channels,
                    filters,
                    in_h,
                    in_w,
                    out_h,
                    out_w,
                    size,
                    stride_x,
                    stride_y,
                    padding,
                    dilation,
                )
            }
            .map_err(launch_err)?;
            OxideTensorF32::new(output, &shape)
        };
        self.bias_add_f32(&convolved, biases, filters, out_h * out_w, config)
    }

    /// Elementwise source `sanitize`.
    pub fn deform_sanitize_f32(
        &self,
        raw: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if raw.len() == 0 {
            return Err(shape_err(
                "deform_sanitize_f32",
                "input must not be empty".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element, same length as the validated input.
        unsafe {
            self.deform_sample.deform_sanitize_tensor(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// Full native `[deform_conv]` **forward** (raw output, before the layer's
    /// own bias/BN/activation epilogue), composed from the tested stages:
    /// sanitize input -> offset predictor -> per-batch stats -> envelope ->
    /// (optional mask predictor + sigmoid) -> deformable accumulation.
    ///
    /// Dilated predictors use an explicit-extent direct CUDA kernel; the same
    /// stages are dispatched by the generic runtime for non-BN variants.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_forward_native(
        &self,
        input: &OxideTensorF32,
        main_weights: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        offset_biases: &OxideTensorF32,
        masks: Option<(&OxideTensorF32, &OxideTensorF32)>,
        channels: usize,
        filters: usize,
        groups: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_conv_forward_native";
        if groups == 0 || channels % groups != 0 || filters % groups != 0 {
            return Err(shape_err(
                op,
                "groups must divide both input channels and filters".to_owned(),
            ));
        }
        let kernel_area = size
            .checked_mul(size)
            .ok_or_else(|| shape_err(op, "kernel area overflows".to_owned()))?;
        let plane = out_h * out_w;
        let max_offset = size as f32 * dilation as f32 * 2.0;
        let sanitized = self.deform_sanitize_f32(input, config)?;
        let raw_offsets = self.deform_offset_predict(
            &sanitized,
            offset_weights,
            offset_biases,
            channels,
            in_h,
            in_w,
            2 * kernel_area,
            size,
            stride_x,
            stride_y,
            padding,
            dilation,
            out_h,
            out_w,
            config,
        )?;
        let (mean, inv_std, normalize) =
            self.deform_offset_stats(&raw_offsets, 2 * kernel_area * plane, max_offset, config)?;
        let offsets = self.deform_offset_envelope(
            &raw_offsets,
            &mean,
            &inv_std,
            &normalize,
            2 * kernel_area,
            plane,
            max_offset,
            config,
        )?;
        let mask_probabilities = match masks {
            Some((weights, biases)) => Some(self.deform_mask_sigmoid(
                &self.deform_offset_predict(
                    &sanitized,
                    weights,
                    biases,
                    channels,
                    in_h,
                    in_w,
                    kernel_area,
                    size,
                    stride_x,
                    stride_y,
                    padding,
                    dilation,
                    out_h,
                    out_w,
                    config,
                )?,
                config,
            )?),
            None => None,
        };
        // The accumulation kernel ignores `masks` when `use_mask == 0`; a
        // one-element buffer keeps the operand list uniform without an empty
        // allocation.
        let empty = self.upload_f32(&[1], &[0.0])?;
        self.deform_conv_accumulate(
            &sanitized,
            &offsets,
            mask_probabilities.as_ref().unwrap_or(&empty),
            main_weights,
            channels,
            filters,
            channels / groups,
            filters / groups,
            in_h,
            in_w,
            out_h,
            out_w,
            size,
            stride_x,
            stride_y,
            padding,
            dilation,
            mask_probabilities.is_some(),
            config,
        )
    }

    /// Adjoint of [`Self::deform_conv_accumulate`]: returns
    /// `(d_input, d_offsets, d_masks, d_weights)`. Atomic scatter targets, so
    /// results agree with the host reference to accumulation-order tolerance.
    /// The input `sanitize` slope is **not** applied here; the caller does it
    /// once on the summed input gradient.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_accumulate_backward(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        masks: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::deform_conv::DeformGeometry,
        use_mask: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
        ),
        OxideError,
    > {
        let op = "deform_conv_accumulate_backward";
        let kernels = geometry.kernels();
        let plane = geometry.plane();
        let input_len =
            geometry.batch * geometry.channels * geometry.input_height * geometry.input_width;
        let offset_len = geometry.batch * 2 * kernels * plane;
        let mask_len = geometry.batch * kernels * plane;
        let weight_len = geometry.filters * geometry.channels_per_group * kernels;
        if input.len() != input_len
            || offsets.len() != offset_len
            || weights.len() != weight_len
            || output_grad.len() != geometry.batch * geometry.filters * plane
            || (use_mask && masks.len() != mask_len)
        {
            return Err(shape_err(
                op,
                "operand length does not match the geometry".to_owned(),
            ));
        }
        let d_input = self.alloc(input_len)?;
        let d_offsets = self.alloc(offset_len)?;
        let d_masks = self.alloc(if use_mask { mask_len } else { 1 })?;
        let d_weights = self.alloc(weight_len)?;
        // SAFETY: one thread per output element; all four targets are zeroed
        // buffers of the validated extents and are written only by atomic adds.
        unsafe {
            self.deform_backward.deform_conv_accumulate_backward(
                &self.stream,
                launch::elementwise(output_grad.len(), config),
                &input.buf,
                &offsets.buf,
                &masks.buf,
                &weights.buf,
                &output_grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                d_offsets.cu_deviceptr() as *mut f32,
                d_masks.cu_deviceptr() as *mut f32,
                d_weights.cu_deviceptr() as *mut f32,
                geometry.channels,
                geometry.filters,
                geometry.channels_per_group,
                geometry.filters_per_group,
                geometry.input_height,
                geometry.input_width,
                geometry.output_height,
                geometry.output_width,
                geometry.size,
                geometry.stride_x,
                geometry.stride_y,
                geometry.padding,
                geometry.dilation,
                u32::from(use_mask),
            )
        }
        .map_err(launch_err)?;
        let mask_shape = [if use_mask { mask_len } else { 0 }];
        Ok((
            OxideTensorF32::new(
                d_input,
                &[
                    geometry.batch,
                    geometry.channels,
                    geometry.input_height,
                    geometry.input_width,
                ],
            ),
            OxideTensorF32::new(
                d_offsets,
                &[
                    geometry.batch,
                    2 * kernels,
                    geometry.output_height,
                    geometry.output_width,
                ],
            ),
            OxideTensorF32::new(d_masks, &mask_shape),
            OxideTensorF32::new(d_weights, &[weight_len]),
        ))
    }

    /// Adjoint of [`Self::deform_offset_predict`]: returns
    /// `(d_input, d_weights, d_biases)`. Atomic scatter; the input `sanitize`
    /// slope is the caller's job.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_predictor_backward(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        channels: usize,
        filters: usize,
        in_h: usize,
        in_w: usize,
        out_h: usize,
        out_w: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "deform_predictor_backward";
        let kernel_area = size
            .checked_mul(size)
            .ok_or_else(|| shape_err(op, "kernel area overflows".to_owned()))?;
        let column_count = channels * kernel_area;
        let plane = out_h * out_w;
        let weight_len = filters * column_count;
        if input.len() != batch * channels * in_h * in_w
            || weights.len() != weight_len
            || output_grad.len() != batch * filters * plane
        {
            return Err(shape_err(
                op,
                "operand length does not match the geometry".to_owned(),
            ));
        }
        let input_len = batch * channels * in_h * in_w;
        let d_input = self.alloc(input_len)?;
        let d_weights = self.alloc(weight_len)?;
        let d_biases = self.alloc(filters)?;
        // SAFETY: one thread per column element; all three targets are zeroed
        // buffers of the validated extents and are written only by atomic adds.
        unsafe {
            self.deform_backward.deform_predictor_backward(
                &self.stream,
                launch::elementwise(batch * column_count * plane, config),
                &input.buf,
                &weights.buf,
                &output_grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                d_weights.cu_deviceptr() as *mut f32,
                d_biases.cu_deviceptr() as *mut f32,
                channels,
                filters,
                in_h,
                in_w,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding,
                dilation,
                batch,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(d_input, &[batch, channels, in_h, in_w]),
            OxideTensorF32::new(d_weights, &[weight_len]),
            OxideTensorF32::new(d_biases, &[filters]),
        ))
    }

    /// Pre-normalization envelope adjoint (lhtan + sanitize slopes).
    #[allow(clippy::too_many_arguments)]
    pub fn deform_envelope_pre_backward(
        &self,
        raw: &OxideTensorF32,
        mean: &OxideTensorF32,
        inv_std: &OxideTensorF32,
        normalize: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_envelope_pre_backward";
        let per_batch = planes
            .checked_mul(spatial)
            .filter(|&v| v > 0)
            .ok_or_else(|| shape_err(op, "planes and spatial must be positive".to_owned()))?;
        if raw.len() == 0 || raw.len() % per_batch != 0 || output_grad.len() != raw.len() {
            return Err(shape_err(op, "raw/gradient length mismatch".to_owned()));
        }
        let batch = raw.len() / per_batch;
        for (name, tensor) in [
            ("mean", mean),
            ("inv_std", inv_std),
            ("normalize", normalize),
        ] {
            if tensor.len() != batch {
                return Err(shape_err(
                    op,
                    format!("{name} has {} elements, expected {batch}", tensor.len()),
                ));
            }
        }
        if !max_offset.is_finite() || max_offset <= 0.0 {
            return Err(shape_err(
                op,
                "max_offset must be finite and positive".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element; every operand is length-checked above.
        unsafe {
            self.deform_backward.deform_envelope_pre_backward(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mean.buf,
                &inv_std.buf,
                &normalize.buf,
                &output_grad.buf,
                &mut out,
                per_batch,
                max_offset,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// Mask sigmoid adjoint.
    pub fn deform_mask_sigmoid_backward(
        &self,
        logits: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_mask_sigmoid_backward";
        if logits.len() == 0 || logits.len() != output_grad.len() {
            return Err(shape_err(op, "logit/gradient length mismatch".to_owned()));
        }
        let mut out = self.alloc_uninit(logits.len())?;
        // SAFETY: one thread per element, same length as both inputs.
        unsafe {
            self.deform_backward.deform_mask_sigmoid_backward(
                &self.stream,
                launch::elementwise(logits.len(), config),
                &logits.buf,
                &output_grad.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, logits.shape()))
    }

    /// Scale a summed input gradient by the raw input's `sanitize` slope.
    pub fn deform_scale_input_gradient(
        &self,
        raw: &OxideTensorF32,
        grad: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_scale_input_gradient";
        if raw.len() == 0 || raw.len() != grad.len() {
            return Err(shape_err(op, "raw/gradient length mismatch".to_owned()));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element, same length as both inputs.
        unsafe {
            self.deform_backward.deform_scale_input_gradient(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &grad.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// Composed native `[deform_conv]` backward. Every per-element stage runs on
    /// the device; the **only** host hop is the batch-serial normalization
    /// Jacobian (mean/std coupling), which also applies the stage-1 slope.
    ///
    /// Returns `(d_input, d_main_weights, d_offset_weights, d_offset_biases,
    /// d_mask_weights, d_mask_biases)`.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_conv_backward_native(
        &self,
        input_sanitized: &OxideTensorF32,
        raw_input: &OxideTensorF32,
        main_weights: &OxideTensorF32,
        offset_weights: &OxideTensorF32,
        mask_weights: Option<&OxideTensorF32>,
        raw_offsets: &OxideTensorF32,
        mean: &OxideTensorF32,
        inv_std: &OxideTensorF32,
        stddev: &OxideTensorF32,
        normalize: &OxideTensorF32,
        offsets: &OxideTensorF32,
        masks: Option<&OxideTensorF32>,
        mask_logits: Option<&OxideTensorF32>,
        output_grad: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::deform_conv::DeformGeometry,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<
        (
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            OxideTensorF32,
            Option<OxideTensorF32>,
            Option<OxideTensorF32>,
        ),
        OxideError,
    > {
        let op = "deform_conv_backward_native";
        let kernels = geometry.kernels();
        let plane = geometry.plane();
        let per_batch = 2 * kernels * plane;
        let empty = self.upload_f32(&[1], &[0.0])?;
        let masks_operand = masks.unwrap_or(&empty);
        // 1. accumulation adjoint.
        let (d_input_accum, d_offsets, d_masks, d_main_weights) = self
            .deform_conv_accumulate_backward(
                input_sanitized,
                offsets,
                masks_operand,
                main_weights,
                output_grad,
                geometry,
                masks.is_some(),
                config,
            )?;
        // 2. lhtan/sanitize slopes (device).
        let d_pre = self.deform_envelope_pre_backward(
            raw_offsets,
            mean,
            inv_std,
            normalize,
            &d_offsets,
            2 * kernels,
            plane,
            max_offset,
            config,
        )?;
        // 3. the one host hop: batch-coupled normalization Jacobian + stage-1 slope.
        let stage1_host = self.download_f32(raw_offsets)?;
        let stage1_host: Vec<f32> = stage1_host
            .iter()
            .map(|&v| lightnet_cuda::host_ops::deform_conv::sanitize(v))
            .collect();
        let d_pre_host = self.download_f32(&d_pre)?;
        let mean_host = self.download_f32(mean)?;
        let std_host = self.download_f32(stddev)?;
        let normalize_host = self.download_f32(normalize)?;
        let d_stage1 = lightnet_cuda::host_ops::deform_conv::offset_normalization_backward(
            &stage1_host,
            &mean_host,
            &std_host,
            &normalize_host,
            &d_pre_host,
            per_batch,
        )
        .map_err(|msg| shape_err(op, msg))?;
        let raw_host = self.download_f32(raw_offsets)?;
        let d_offsets_predictor: Vec<f32> = d_stage1
            .iter()
            .zip(&raw_host)
            .map(|(&d, &raw)| {
                let slope = if raw.is_finite() && raw.abs() < 10_000.0 {
                    1.0
                } else {
                    0.0
                };
                d * slope
            })
            .collect();
        let d_offsets_predictor = self.upload_f32(raw_offsets.shape(), &d_offsets_predictor)?;
        // 4. predictor + column adjoint (device).
        let (d_input_off, d_offset_weights, d_offset_biases) = self.deform_predictor_backward(
            input_sanitized,
            offset_weights,
            &d_offsets_predictor,
            geometry.channels,
            2 * kernels,
            geometry.input_height,
            geometry.input_width,
            geometry.output_height,
            geometry.output_width,
            geometry.size,
            geometry.stride_x,
            geometry.stride_y,
            geometry.padding,
            geometry.dilation,
            geometry.batch,
            config,
        )?;
        // 5. mask path (device), when present.
        let (d_input_mask, d_mask_weights, d_mask_biases) =
            match (mask_weights, mask_logits, masks.is_some()) {
                (Some(weights), Some(logits), true) => {
                    let d_logits = self.deform_mask_sigmoid_backward(logits, &d_masks, config)?;
                    let (d_in, d_w, d_b) = self.deform_predictor_backward(
                        input_sanitized,
                        weights,
                        &d_logits,
                        geometry.channels,
                        kernels,
                        geometry.input_height,
                        geometry.input_width,
                        geometry.output_height,
                        geometry.output_width,
                        geometry.size,
                        geometry.stride_x,
                        geometry.stride_y,
                        geometry.padding,
                        geometry.dilation,
                        geometry.batch,
                        config,
                    )?;
                    (Some(d_in), Some(d_w), Some(d_b))
                }
                _ => (None, None, None),
            };
        // 6. sum and apply the input slope exactly once.
        let mut sum = self.vector_add_f32(&d_input_accum, &d_input_off, config)?;
        if let Some(d_in) = d_input_mask.as_ref() {
            sum = self.vector_add_f32(&sum, d_in, config)?;
        }
        let d_input = self.deform_scale_input_gradient(raw_input, &sum, config)?;
        Ok((
            d_input,
            d_main_weights,
            d_offset_weights,
            d_offset_biases,
            d_mask_weights,
            d_mask_biases,
        ))
    }


    /// Per-batch offset statistics over the sanitized stage-1 offsets. Mirrors
    /// the source f64 accumulation and returns `(mean, inv_std, normalize)`
    /// (the last as 0.0/1.0), each of length `batch`.
    pub fn deform_offset_stats(
        &self,
        raw: &OxideTensorF32,
        per_batch: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "deform_offset_stats";
        if per_batch == 0 || raw.len() == 0 || raw.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "raw has {} elements, not a positive multiple of per_batch {per_batch}",
                    raw.len()
                ),
            ));
        }
        if !max_offset.is_finite() || max_offset <= 0.0 {
            return Err(shape_err(
                op,
                "max_offset must be finite and positive".to_owned(),
            ));
        }
        let batch = raw.len() / per_batch;
        let mut mean = self.alloc_uninit(batch)?;
        let mut inv_std = self.alloc_uninit(batch)?;
        let mut normalize = self.alloc_uninit(batch)?;
        // SAFETY: one thread per batch; each thread owns its own output slot and
        // reads only its own contiguous `per_batch` slice.
        unsafe {
            self.deform_sample.deform_offset_stats(
                &self.stream,
                launch::elementwise(batch, config),
                &raw.buf,
                &mut mean,
                &mut inv_std,
                &mut normalize,
                per_batch,
                f64::from(max_offset),
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(mean, &[batch]),
            OxideTensorF32::new(inv_std, &[batch]),
            OxideTensorF32::new(normalize, &[batch]),
        ))
    }

    /// `[deform_conv]` offset envelope: `sanitize -> optional per-batch
    /// normalization -> sanitize -> centered lhtan`. `mean`, `inv_std` and
    /// `normalize` (0.0/1.0) are per-batch tensors produced by the caller; the
    /// stats themselves are not computed here.
    #[allow(clippy::too_many_arguments)]
    pub fn deform_offset_envelope(
        &self,
        raw: &OxideTensorF32,
        mean: &OxideTensorF32,
        inv_std: &OxideTensorF32,
        normalize: &OxideTensorF32,
        planes: usize,
        spatial: usize,
        max_offset: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "deform_offset_envelope";
        let per_batch = planes
            .checked_mul(spatial)
            .filter(|&v| v > 0)
            .ok_or_else(|| shape_err(op, "planes and spatial must be positive".to_owned()))?;
        if raw.len() == 0 || raw.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "raw has {} elements, not a multiple of planes*spatial = {per_batch}",
                    raw.len()
                ),
            ));
        }
        let batch = raw.len() / per_batch;
        for (name, tensor) in [
            ("mean", mean),
            ("inv_std", inv_std),
            ("normalize", normalize),
        ] {
            if tensor.len() != batch {
                return Err(shape_err(
                    op,
                    format!("{name} has {} elements, expected {batch}", tensor.len()),
                ));
            }
        }
        if !max_offset.is_finite() || max_offset <= 0.0 {
            return Err(shape_err(
                op,
                "max_offset must be finite and positive".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per offset element; all four inputs are validated
        // above and indexed by decode-verified positions.
        unsafe {
            self.deform_sample.deform_offset_envelope(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mean.buf,
                &inv_std.buf,
                &normalize.buf,
                &mut out,
                planes,
                spatial,
                max_offset,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }

    /// `[deform_conv]` mask probabilities: `stable_sigmoid(sanitize(logit))`.
    pub fn deform_mask_sigmoid(
        &self,
        raw: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if raw.len() == 0 {
            return Err(shape_err(
                "deform_mask_sigmoid",
                "logits must not be empty".to_owned(),
            ));
        }
        let mut out = self.alloc_uninit(raw.len())?;
        // SAFETY: one thread per element, same length as the validated input.
        unsafe {
            self.deform_sample.deform_mask_sigmoid(
                &self.stream,
                launch::elementwise(raw.len(), config),
                &raw.buf,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, raw.shape()))
    }
}

impl OxideBackend {
    pub(crate) fn deform_bilinear_gather_backend_f32(
        &self,
        input: &OxideTensorF32,
        sample_y: &OxideTensorF32,
        sample_x: &OxideTensorF32,
        geometry: BilinearGatherGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let y_f = self.f32v(sample_y)?;
        let y = &*y_f;
        let x_f = self.f32v(sample_x)?;
        let x = &*x_f;
        let op = "deform_bilinear_gather_f32";
        deform_gather_shape(op, input, y, x, None, geometry)?;
        let mut out = self.alloc_uninit(geometry.output_len())?;
        // SAFETY: one thread per output element; every tap is bounds-checked
        // in the kernel and only reads `input`.
        unsafe {
            self.deform_sample.deform_gather(
                &self.stream,
                launch::elementwise(geometry.output_len(), config),
                &input.buf,
                &y.buf,
                &x.buf,
                &mut out,
                geometry.channels,
                geometry.input_height,
                geometry.input_width,
                geometry.samples,
                geometry.output_height,
                geometry.output_width,
            )
        }
        .map_err(launch_err)?;
        let shape = [
            geometry.batch,
            geometry.samples,
            geometry.channels,
            geometry.output_height,
            geometry.output_width,
        ];
        Ok(OxideTensorF32::new(out, &shape))
    }

    pub(crate) fn deform_bilinear_gather_backward_backend_f32(
        &self,
        input: &OxideTensorF32,
        sample_y: &OxideTensorF32,
        sample_x: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        geometry: BilinearGatherGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let input_f = self.f32v(input)?;
        let input = &*input_f;
        let y_f = self.f32v(sample_y)?;
        let y = &*y_f;
        let x_f = self.f32v(sample_x)?;
        let x = &*x_f;
        let grad_f = self.f32v(output_grad)?;
        let grad = &*grad_f;
        let op = "deform_bilinear_gather_backward_f32";
        deform_gather_shape(op, input, y, x, Some(grad), geometry)?;
        // `alloc` is zero-filled: the kernel only touches `d_input` through
        // in-image atomic adds, so the zeros are the scatter identity.
        let d_input = self.alloc(geometry.input_len())?;
        let mut d_y = self.alloc_uninit(geometry.sample_len())?;
        let mut d_x = self.alloc_uninit(geometry.sample_len())?;
        // SAFETY: one thread per sample cell owns `d_y`/`d_x`; `d_input` is a
        // live zeroed buffer written only by atomic adds on validated taps.
        unsafe {
            self.deform_sample.deform_gather_backward(
                &self.stream,
                launch::elementwise(geometry.sample_len(), config),
                &input.buf,
                &y.buf,
                &x.buf,
                &grad.buf,
                d_input.cu_deviceptr() as *mut f32,
                &mut d_y,
                &mut d_x,
                geometry.channels,
                geometry.input_height,
                geometry.input_width,
                geometry.samples,
                geometry.output_height,
                geometry.output_width,
            )
        }
        .map_err(launch_err)?;
        let input_shape = [
            geometry.batch,
            geometry.channels,
            geometry.input_height,
            geometry.input_width,
        ];
        let sample_shape = [
            geometry.batch,
            geometry.samples,
            geometry.output_height,
            geometry.output_width,
        ];
        Ok((
            OxideTensorF32::new(d_input, &input_shape),
            OxideTensorF32::new(d_y, &sample_shape),
            OxideTensorF32::new(d_x, &sample_shape),
        ))
    }

    pub(crate) fn deform_predictor_backend_f32(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        in_h: usize,
        in_w: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        dilation: usize,
        out_h: usize,
        out_w: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let input = self.f32v(input)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let per_batch = channels
            .checked_mul(in_h)
            .and_then(|n| n.checked_mul(in_w))
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                shape_err("deform_predictor_f32", "input extent overflows".to_owned())
            })?;
        if input.len() % per_batch != 0 {
            return Err(shape_err(
                "deform_predictor_f32",
                "input length is not a whole batch".to_owned(),
            ));
        }
        let batch = input.len() / per_batch;
        if output_shape != [batch, filters, out_h, out_w] {
            return Err(shape_err(
                "deform_predictor_f32",
                "output shape does not match predictor geometry".to_owned(),
            ));
        }
        OxideBackend::deform_offset_predict(
            self, &input, &weights, &biases, channels, in_h, in_w, filters, size, stride_x,
            stride_y, padding, dilation, out_h, out_w, config,
        )
    }

    pub(crate) fn deform_conv_accumulate_backend_f32(
        &self,
        input: &OxideTensorF32,
        offsets: &OxideTensorF32,
        masks: &OxideTensorF32,
        weights: &OxideTensorF32,
        geometry: lightnet_cuda::host_ops::deform_conv::DeformGeometry,
        use_mask: bool,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let out = OxideBackend::deform_conv_accumulate(
            self,
            input,
            offsets,
            masks,
            weights,
            geometry.channels,
            geometry.filters,
            geometry.channels_per_group,
            geometry.filters_per_group,
            geometry.input_height,
            geometry.input_width,
            geometry.output_height,
            geometry.output_width,
            geometry.size,
            geometry.stride_x,
            geometry.stride_y,
            geometry.padding,
            geometry.dilation,
            use_mask,
            config,
        )?;
        let _ = shape;
        let expected = [
            geometry.batch,
            geometry.filters,
            geometry.output_height,
            geometry.output_width,
        ];
        if out.shape() != expected {
            return Err(shape_err(
                "deform_conv_accumulate_f32",
                format!("output shape {:?} != {expected:?}", out.shape()),
            ));
        }
        Ok(out)
    }
}

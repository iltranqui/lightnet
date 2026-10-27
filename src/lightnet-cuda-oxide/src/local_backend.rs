//! Native local-layer backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{shape_err, launch, launch_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn local_backend_f32(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        biases: &OxideTensorF32,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if input_shape.len() != 4 || output_shape.len() != 4 {
            return Err(shape_err(
                "local_f32",
                "input and output shapes must be NCHW".into(),
            ));
        }
        let [batch, channels, height, width] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];
        let [out_batch, out_filters, out_h, out_w] = [
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        ];
        if [
            batch,
            channels,
            height,
            width,
            filters,
            out_h,
            out_w,
            kernel_size,
            stride_x,
            stride_y,
        ]
        .contains(&0)
            || out_batch != batch
            || out_filters != filters
        {
            return Err(shape_err("local_f32", "invalid local geometry".into()));
        }
        let locations = out_h
            .checked_mul(out_w)
            .ok_or_else(|| shape_err("local_f32", "output location count overflow".into()))?;
        let kernel = channels
            .checked_mul(kernel_size)
            .and_then(|v| v.checked_mul(kernel_size))
            .ok_or_else(|| shape_err("local_f32", "kernel size overflow".into()))?;
        let expected_x = batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(height))
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| shape_err("local_f32", "input size overflow".into()))?;
        let expected_w = locations
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(kernel))
            .ok_or_else(|| shape_err("local_f32", "weight size overflow".into()))?;
        let expected_out = batch
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(locations))
            .ok_or_else(|| shape_err("local_f32", "output size overflow".into()))?;
        if x.len() != expected_x
            || weights.len() != expected_w
            || output_shape.iter().product::<usize>() != expected_out
        {
            return Err(shape_err(
                "local_f32",
                format!(
                    "operand lengths input/weights/output-shape mismatch; expected {expected_x}/{expected_w}/{expected_out}"
                ),
            ));
        }
        let expected_biases = locations
            .checked_mul(filters)
            .ok_or_else(|| shape_err("local_f32", "bias size overflow".into()))?;
        if biases.len() != expected_biases {
            return Err(shape_err(
                "local_f32",
                format!("expected {expected_biases} biases, got {}", biases.len()),
            ));
        }
        let x = self.f32v(x)?;
        let weights = self.f32v(weights)?;
        let biases = self.f32v(biases)?;
        let mut output = self.alloc_uninit(expected_out)?;
        // SAFETY: launch geometry and all indexed extents were checked above;
        // each output element has exactly one owning thread.
        unsafe {
            self.local.local_fwd(
                &self.stream,
                launch::elementwise(expected_out, config),
                &x.buf,
                &weights.buf,
                &biases.buf,
                &mut output,
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
                padding,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn local_backend_backward_f32(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        if input_shape.len() != 4 || output_shape.len() != 4 {
            return Err(shape_err(
                "local_backward_f32",
                "input and output shapes must be NCHW".into(),
            ));
        }
        let (batch, channels, height, width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (out_h, out_w) = (output_shape[2], output_shape[3]);
        let locations = out_h.checked_mul(out_w).ok_or_else(|| {
            shape_err(
                "local_backward_f32",
                "output location count overflow".into(),
            )
        })?;
        let kernel = channels
            .checked_mul(kernel_size)
            .and_then(|v| v.checked_mul(kernel_size))
            .ok_or_else(|| shape_err("local_backward_f32", "kernel size overflow".into()))?;
        let x_len = batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(height))
            .and_then(|v| v.checked_mul(width))
            .ok_or_else(|| shape_err("local_backward_f32", "input size overflow".into()))?;
        let output_len = batch
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(locations))
            .ok_or_else(|| shape_err("local_backward_f32", "output size overflow".into()))?;
        let weights_len = locations
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(kernel))
            .ok_or_else(|| shape_err("local_backward_f32", "weight size overflow".into()))?;
        if x.len() != x_len
            || weights.len() != weights_len
            || output_grad.len() != output_len
            || output_shape[0] != batch
            || output_shape[1] != filters
            || [
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
            ]
            .contains(&0)
        {
            return Err(shape_err(
                "local_backward_f32",
                "invalid operand lengths or geometry".into(),
            ));
        }
        let x = self.f32v(x)?;
        let weights = self.f32v(weights)?;
        let delta = self.f32v(output_grad)?;
        let mut dx = self.alloc_uninit(x_len)?;
        let mut dw = self.alloc_uninit(weights_len)?;
        let mut db = self.alloc_uninit(locations * filters)?;
        // SAFETY: dgrad gathers overlapping patches; wgrad and bgrad each
        // reduce batch in a single owning thread, so no atomics are needed.
        unsafe {
            self.local.local_dgrad(
                &self.stream,
                launch::elementwise(x_len, config),
                &delta.buf,
                &weights.buf,
                &mut dx,
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
                padding,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.local.local_wgrad(
                &self.stream,
                launch::elementwise(weights_len, config),
                &x.buf,
                &delta.buf,
                &mut dw,
                batch,
                channels,
                height,
                width,
                filters,
                out_h,
                out_w,
                kernel_size,
                stride_x,
                stride_y,
                padding,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.local.local_bgrad(
                &self.stream,
                launch::elementwise(locations * filters, config),
                &delta.buf,
                &mut db,
                batch,
                filters,
                locations,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(dx, input_shape),
            OxideTensorF32::new(
                dw,
                &[locations, filters, channels, kernel_size, kernel_size],
            ),
            OxideTensorF32::new(db, &[locations, filters]),
        ))
    }
}

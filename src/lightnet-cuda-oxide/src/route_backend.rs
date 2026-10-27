//! Native route backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF16, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn route_backend_concat_f32(
        &self,
        inputs: &[&OxideTensorF32],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if inputs.iter().any(|t| t.is_half()) {
            let halves = inputs
                .iter()
                .map(|t| self.as_half(t))
                .collect::<Result<Vec<_>, _>>()?;
            let refs: Vec<&OxideTensorF16> = halves.iter().collect();
            let out = self.route_concat_f16(
                &refs,
                input_channel_offsets,
                input_channels,
                input_total_channels,
                output_shape,
                channel_stride,
                config,
            )?;
            return Ok(self.wrap_half(out));
        }
        let op = "route_concat_f32";
        let n = inputs.len();
        if n == 0
            || input_channel_offsets.len() != n
            || input_channels.len() != n
            || input_total_channels.len() != n
        {
            return Err(shape_err(
                op,
                "needs >= 1 input and equal-length metadata".to_owned(),
            ));
        }
        if channel_stride == 0 {
            return Err(shape_err(op, "channel_stride must be > 0".to_owned()));
        }
        let out_total: usize = input_channels.iter().sum();
        let out_len: usize = output_shape.iter().product();
        let mut batch = None;
        for k in 0..n {
            if input_channels[k] == 0
                || input_channel_offsets[k] + input_channels[k] > input_total_channels[k]
            {
                return Err(shape_err(
                    op,
                    format!(
                        "input {k}: slice [{}..{}) outside {} channels",
                        input_channel_offsets[k],
                        input_channel_offsets[k] + input_channels[k],
                        input_total_channels[k]
                    ),
                ));
            }
            let per_batch = input_total_channels[k] * channel_stride;
            if inputs[k].len() == 0 || inputs[k].len() % per_batch != 0 {
                return Err(shape_err(
                    op,
                    format!(
                        "input {k} length {} not a multiple of {per_batch}",
                        inputs[k].len()
                    ),
                ));
            }
            let b = inputs[k].len() / per_batch;
            if *batch.get_or_insert(b) != b {
                return Err(shape_err(
                    op,
                    format!("input {k} batch {b} differs from {batch:?}"),
                ));
            }
        }
        let batch = batch.unwrap_or(0);
        if out_len != batch * out_total * channel_stride {
            return Err(shape_err(
                op,
                format!(
                    "output shape {output_shape:?} needs {} elements, inputs give {}",
                    out_len,
                    batch * out_total * channel_stride
                ),
            ));
        }
        let mut out = self.alloc_uninit(out_len)?;
        let mut dst_offset = 0usize;
        for k in 0..n {
            let count = batch * input_channels[k] * channel_stride;
            // SAFETY: one thread per copied element, all with distinct
            // destinations inside `out` (validated above).
            unsafe {
                self.route.route_copy(
                    &self.stream,
                    launch::elementwise(count, config),
                    &inputs[k].buf,
                    &mut out,
                    batch,
                    input_channels[k],
                    input_total_channels[k],
                    input_channel_offsets[k],
                    out_total,
                    dst_offset,
                    channel_stride,
                )
            }
            .map_err(launch_err)?;
            dst_offset += input_channels[k];
        }
        Ok(OxideTensorF32::new(out, output_shape))
    }

    pub(crate) fn route_backend_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<OxideTensorF32>, OxideError> {
        if output_grad.is_half() {
            return self.route_backward_amp(
                output_grad,
                input_shapes,
                input_channel_offsets,
                input_channels,
                input_total_channels,
                channel_stride,
                config,
            );
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "route_backward_f32";
        let n = input_shapes.len();
        if n == 0
            || input_channel_offsets.len() != n
            || input_channels.len() != n
            || input_total_channels.len() != n
        {
            return Err(shape_err(
                op,
                "needs >= 1 input and equal-length metadata".to_owned(),
            ));
        }
        if channel_stride == 0 {
            return Err(shape_err(op, "channel_stride must be > 0".to_owned()));
        }
        let out_total: usize = input_channels.iter().sum();
        let per_batch = out_total * channel_stride;
        if per_batch == 0 || output_grad.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "gradient length {} not a multiple of {per_batch}",
                    output_grad.len()
                ),
            ));
        }
        let batch = output_grad.len() / per_batch;
        let mut grads = Vec::with_capacity(n);
        let mut src_offset = 0usize;
        for k in 0..n {
            let len: usize = input_shapes[k].iter().product();
            if input_channels[k] == 0
                || input_channel_offsets[k] + input_channels[k] > input_total_channels[k]
                || len != batch * input_total_channels[k] * channel_stride
            {
                return Err(shape_err(
                    op,
                    format!(
                        "input {k}: shape {:?} / slice [{}..{}) inconsistent with {} channels, batch {batch}",
                        input_shapes[k],
                        input_channel_offsets[k],
                        input_channel_offsets[k] + input_channels[k],
                        input_total_channels[k]
                    ),
                ));
            }
            // A slice that covers every channel is fully overwritten; otherwise the
            // untouched channels must read zero.
            let mut grad = if input_channels[k] == input_total_channels[k] {
                self.alloc_uninit(len)?
            } else {
                self.alloc(len)?
            };
            let count = batch * input_channels[k] * channel_stride;
            // SAFETY: distinct destination per thread inside `grad`, which
            // covers `batch * input_total_channels[k] * stride` (validated).
            unsafe {
                self.route.route_copy(
                    &self.stream,
                    launch::elementwise(count, config),
                    &output_grad.buf,
                    &mut grad,
                    batch,
                    input_channels[k],
                    out_total,
                    src_offset,
                    input_total_channels[k],
                    input_channel_offsets[k],
                    channel_stride,
                )
            }
            .map_err(launch_err)?;
            grads.push(OxideTensorF32::new(grad, &input_shapes[k]));
            src_offset += input_channels[k];
        }
        Ok(grads)
    }
}

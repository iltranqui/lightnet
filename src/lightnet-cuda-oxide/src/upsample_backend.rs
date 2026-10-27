//! Native upsample backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{launch, launch_err, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn upsample_backend_f32(
        &self,
        x: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if x.is_half() && !reverse {
            let xh = self.as_half(x)?;
            let out = self.upsample_f16(
                &xh,
                output_shape,
                channels,
                input_height,
                input_width,
                stride,
                scale,
                reverse,
                config,
            )?;
            return Ok(self.wrap_half(out));
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "upsample_f32";
        if stride == 0 || channels == 0 || input_height == 0 || input_width == 0 {
            return Err(shape_err(
                op,
                "channels, spatial dims and stride must be > 0".to_owned(),
            ));
        }
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} is not [batch, {channels}, h, w]"),
            ));
        }
        let (batch, out_h, out_w) = (output_shape[0], output_shape[2], output_shape[3]);
        if x.len() != batch * channels * input_height * input_width {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {batch}*{channels}*{input_height}*{input_width}",
                    x.len()
                ),
            ));
        }
        if reverse {
            // floor (C++) and ceil (cuTile) sized outputs are both accepted.
            if out_h == 0
                || out_w == 0
                || out_h > input_height.div_ceil(stride)
                || out_w > input_width.div_ceil(stride)
                || out_h < input_height / stride
                || out_w < input_width / stride
            {
                return Err(shape_err(
                    op,
                    format!(
                        "reverse output {out_h}x{out_w} incompatible with input {input_height}x{input_width} / stride {stride}"
                    ),
                ));
            }
        } else if out_h != input_height * stride || out_w != input_width * stride {
            return Err(shape_err(
                op,
                format!(
                    "output {out_h}x{out_w} != input {input_height}x{input_width} * stride {stride}"
                ),
            ));
        }
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
        // SAFETY: one thread per output element; input reads are in range by
        // the shape checks above (reverse additionally bounds-checks taps).
        unsafe {
            if reverse {
                self.resample.upsample_reverse(
                    &self.stream,
                    launch::elementwise(out.len(), config),
                    &x.buf,
                    &mut out,
                    input_height,
                    input_width,
                    out_h,
                    out_w,
                    stride,
                    scale,
                )
            } else {
                self.resample.upsample_nearest(
                    &self.stream,
                    launch::elementwise(out.len(), config),
                    &x.buf,
                    &mut out,
                    input_height,
                    input_width,
                    out_h,
                    out_w,
                    stride,
                    scale,
                )
            }
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    /// The trait carries no `reverse` flag: the layer direction is inferred from
    /// the shapes.  `output_height == input_height * stride` is a normal
    /// (expanding) layer; anything smaller is a `reverse` layer (identical
    /// results when `stride == 1`).
    pub(crate) fn upsample_backend_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if output_grad.is_half()
            && input_shape.len() == 4
            && output_height == input_shape[2] * stride
            && output_width == input_shape[3] * stride
        {
            return self.upsample_backward_amp(
                output_grad,
                input_shape,
                channels,
                output_height,
                output_width,
                stride,
                scale,
                config,
            );
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "upsample_backward_f32";
        if stride == 0 || input_shape.len() != 4 || input_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("bad stride {stride} or input shape {input_shape:?}"),
            ));
        }
        let (batch, in_h, in_w) = (input_shape[0], input_shape[2], input_shape[3]);
        if in_h == 0 || in_w == 0 || output_height == 0 || output_width == 0 {
            return Err(shape_err(op, "spatial dims must be > 0".to_owned()));
        }
        if output_grad.len() != batch * channels * output_height * output_width {
            return Err(shape_err(
                op,
                format!(
                    "gradient has {} elements, expected {batch}*{channels}*{output_height}*{output_width}",
                    output_grad.len()
                ),
            ));
        }
        let reverse = if output_height == in_h * stride && output_width == in_w * stride {
            false
        } else if output_height >= in_h / stride
            && output_height <= in_h.div_ceil(stride)
            && output_width >= in_w / stride
            && output_width <= in_w.div_ceil(stride)
        {
            true
        } else {
            return Err(shape_err(
                op,
                format!(
                    "output {output_height}x{output_width} fits neither expand nor reverse of input {in_h}x{in_w}, stride {stride}"
                ),
            ));
        };
        let mut dx = self.alloc_uninit(input_shape.iter().product())?;
        // SAFETY: one thread per input element (gather); dy reads are bounds-checked.
        unsafe {
            self.resample.upsample_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mut dx,
                in_h,
                in_w,
                output_height,
                output_width,
                stride,
                scale,
                reverse as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }

}

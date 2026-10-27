//! Native pooling backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{
    launch, launch_err, nchw_batch, pool_layout, shape_err, OxideBackend, OxideError,
    OxideTensorF32,
};

impl OxideBackend {
    pub(crate) fn pool_backend_maxpool_f32(
        &self,
        x: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if x.is_half() {
            let xh = self.as_half(x)?;
            let out = self.maxpool_f16(
                &xh,
                output_shape,
                channels,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                padding,
                config,
            )?;
            return Ok(self.wrap_half(out));
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let (_, out_h, out_w) = pool_layout(
            "maxpool_f32",
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
        // SAFETY: one thread per output element; all input reads are
        // bounds-checked against the image and `x` covers the validated extent.
        unsafe {
            self.pool.maxpool_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    pub(crate) fn pool_backend_maxpool_backward_f32(
        &self,
        x: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if x.is_half() {
            return self.maxpool_backward_amp(
                x,
                output_grad,
                channels,
                input_height,
                input_width,
                size,
                stride_x,
                stride_y,
                padding,
                config,
            );
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let (out_h, out_w) = Self::grad_hw(
            "maxpool_backward_f32",
            output_grad,
            x.len(),
            channels,
            input_height,
            input_width,
        )?;
        pool_layout(
            "maxpool_backward_f32",
            x.len(),
            output_grad.shape(),
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let mut dx = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per input element (gather, no atomics); reads are
        // bounds-checked and `x` / `output_grad` cover the validated extents.
        unsafe {
            self.pool.maxpool_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &x.buf,
                &output_grad.buf,
                &mut dx,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, x.shape()))
    }

    pub(crate) fn pool_backend_local_avgpool_f32(
        &self,
        x: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let (_, out_h, out_w) = pool_layout(
            "local_avgpool_f32",
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let mut out = self.alloc_uninit(output_shape.iter().product())?;
        // SAFETY: one thread per output element; reads are bounds-checked.
        unsafe {
            self.pool.local_avgpool_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    pub(crate) fn pool_backend_local_avgpool_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "local_avgpool_backward_f32";
        let batch = nchw_batch(
            op,
            "input",
            input_shape,
            channels,
            input_height,
            input_width,
        )?;
        let (out_h, out_w) = Self::grad_hw(
            op,
            output_grad,
            batch * channels * input_height * input_width,
            channels,
            input_height,
            input_width,
        )?;
        pool_layout(
            op,
            input_shape.iter().product(),
            output_grad.shape(),
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let mut dx = self.alloc_uninit(input_shape.iter().product())?;
        // SAFETY: one thread per input element (gather); reads bounds-checked.
        unsafe {
            self.pool.local_avgpool_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mut dx,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }

    pub(crate) fn pool_backend_avgpool_f32(
        &self,
        x: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let op = "avgpool_f32";
        let batch = nchw_batch(op, "output", output_shape, channels, 1, 1)?;
        let area = input_height * input_width;
        if area == 0 || x.len() != batch * channels * area {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {batch}*{channels}*{area}",
                    x.len()
                ),
            ));
        }
        let mut out = self.alloc_uninit(batch * channels)?;
        // SAFETY: one thread per (batch, channel) plane, reading `area` elements.
        unsafe {
            self.pool.avgpool_forward(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                area,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, output_shape))
    }

    pub(crate) fn pool_backend_avgpool_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let op = "avgpool_backward_f32";
        let batch = nchw_batch(
            op,
            "input",
            input_shape,
            channels,
            input_height,
            input_width,
        )?;
        let area = input_height * input_width;
        if area == 0 || output_grad.len() != batch * channels {
            return Err(shape_err(
                op,
                format!(
                    "gradient has {} elements, expected {batch}*{channels}",
                    output_grad.len()
                ),
            ));
        }
        let mut dx = self.alloc_uninit(input_shape.iter().product())?;
        // SAFETY: one thread per input element reading `dy[i / area]`.
        unsafe {
            self.pool.avgpool_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &mut dx,
                area,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(dx, input_shape))
    }

}

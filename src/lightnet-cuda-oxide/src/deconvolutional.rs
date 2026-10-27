//! Native direct transposed convolution.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::conv::{expect_len, expect_shape};
use crate::launch;
use crate::{OxideBackend, OxideError, OxideTensorF32, launch_err};

impl OxideBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deconvolutional_f32(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let weights_f = self.f32v(weights)?;
        self.deconvolutional_forward_native(
            &x_f,
            &weights_f,
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deconvolutional_backward_f32(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let x_f = self.f32v(x)?;
        let weights_f = self.f32v(weights)?;
        let output_grad_f = self.f32v(output_grad)?;
        self.deconvolutional_backward_native(
            &x_f,
            &weights_f,
            &output_grad_f,
            input_shape,
            output_shape,
            filters,
            kernel_size,
            stride,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deconvolutional_forward_native(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "deconvolutional_f32";
        if output_shape.len() != 4 {
            return Err(OxideError::Shape(format!(
                "{OP}: output shape must be [batch, filters, height, width], got {output_shape:?}"
            )));
        }
        let batch = output_shape[0];
        if batch == 0
            || channels == 0
            || filters == 0
            || input_height == 0
            || input_width == 0
            || kernel_size == 0
            || stride == 0
        {
            return Err(OxideError::Shape(format!(
                "{OP}: dimensions must be nonzero"
            )));
        }
        let (output_height, output_width) = (output_shape[2], output_shape[3]);
        let expected_height = stride
            .checked_mul(input_height - 1)
            .and_then(|v| v.checked_add(kernel_size))
            .ok_or_else(|| OxideError::Shape(format!("{OP}: output height overflow")))?;
        let expected_width = stride
            .checked_mul(input_width - 1)
            .and_then(|v| v.checked_add(kernel_size))
            .ok_or_else(|| OxideError::Shape(format!("{OP}: output width overflow")))?;
        expect_shape(
            OP,
            "output",
            output_shape,
            &[batch, filters, expected_height, expected_width],
        )?;
        let x_len = batch
            .checked_mul(channels)
            .and_then(|v| v.checked_mul(input_height))
            .and_then(|v| v.checked_mul(input_width))
            .ok_or_else(|| OxideError::Shape(format!("{OP}: input size overflow")))?;
        let weight_len = channels
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(kernel_size))
            .and_then(|v| v.checked_mul(kernel_size))
            .ok_or_else(|| OxideError::Shape(format!("{OP}: weight size overflow")))?;
        let output_len = batch
            .checked_mul(filters)
            .and_then(|v| v.checked_mul(output_height))
            .and_then(|v| v.checked_mul(output_width))
            .ok_or_else(|| OxideError::Shape(format!("{OP}: output size overflow")))?;
        expect_len(OP, "input", x.len(), x_len)?;
        expect_len(OP, "weights", weights.len(), weight_len)?;

        let mut output = self.alloc_uninit(output_len)?;
        // SAFETY: validation proves all input/output extents and the kernel
        // assigns exactly one output element to each in-range thread.
        unsafe {
            self.deconvolutional.deconv_fwd(
                &self.stream,
                launch::elementwise(output_len, config),
                &x.buf,
                &weights.buf,
                &mut output,
                batch,
                channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride,
                output_height,
                output_width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deconvolutional_backward_native(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        output_shape: &[usize],
        filters: usize,
        kernel_size: usize,
        stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        const OP: &str = "deconvolutional_backward_f32";
        if input_shape.len() != 4 || output_shape.len() != 4 {
            return Err(OxideError::Shape(format!(
                "{OP}: input/output shapes must be NCHW, got {input_shape:?}/{output_shape:?}"
            )));
        }
        let (batch, channels, input_height, input_width) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (output_batch, output_filters, output_height, output_width) = (
            output_shape[0],
            output_shape[1],
            output_shape[2],
            output_shape[3],
        );
        if batch == 0
            || channels == 0
            || filters == 0
            || input_height == 0
            || input_width == 0
            || kernel_size == 0
            || stride == 0
        {
            return Err(OxideError::Shape(format!(
                "{OP}: dimensions must be nonzero"
            )));
        }
        let expected_height = stride
            .checked_mul(input_height - 1)
            .and_then(|v| v.checked_add(kernel_size))
            .ok_or_else(|| OxideError::Shape(format!("{OP}: output height overflow")))?;
        let expected_width = stride
            .checked_mul(input_width - 1)
            .and_then(|v| v.checked_add(kernel_size))
            .ok_or_else(|| OxideError::Shape(format!("{OP}: output width overflow")))?;
        expect_shape(
            OP,
            "output",
            output_shape,
            &[batch, filters, expected_height, expected_width],
        )?;
        if output_batch != batch || output_filters != filters {
            return Err(OxideError::Shape(format!(
                "{OP}: expected output batch/filter {batch}/{filters}, got {output_batch}/{output_filters}"
            )));
        }
        let x_len = batch * channels * input_height * input_width;
        let weight_len = channels * filters * kernel_size * kernel_size;
        let dy_len = batch * filters * output_height * output_width;
        expect_len(OP, "input", x.len(), x_len)?;
        expect_len(OP, "weights", weights.len(), weight_len)?;
        expect_len(OP, "output gradient", output_grad.len(), dy_len)?;

        let mut dx = self.alloc_uninit(x_len)?;
        let mut dw = self.alloc_uninit(weight_len)?;
        let mut db = self.alloc_uninit(filters)?;
        // SAFETY: validated shapes, and each kernel owns a unique destination
        // element. Weight/bias sums use deterministic serial reductions.
        unsafe {
            self.deconvolutional.deconv_dgrad(
                &self.stream,
                launch::elementwise(x_len, config),
                &output_grad.buf,
                &weights.buf,
                &mut dx,
                batch,
                channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride,
                output_height,
                output_width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.deconvolutional.deconv_wgrad(
                &self.stream,
                launch::elementwise(weight_len, config),
                &x.buf,
                &output_grad.buf,
                &mut dw,
                batch,
                channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride,
                output_height,
                output_width,
            )
        }
        .map_err(launch_err)?;
        unsafe {
            self.deconvolutional.deconv_bgrad(
                &self.stream,
                launch::elementwise(filters, config),
                &output_grad.buf,
                &mut db,
                batch,
                filters,
                output_height,
                output_width,
            )
        }
        .map_err(launch_err)?;
        Ok((
            OxideTensorF32::new(dx, input_shape),
            OxideTensorF32::new(dw, &[channels, filters, kernel_size, kernel_size]),
            OxideTensorF32::new(db, &[filters]),
        ))
    }
}

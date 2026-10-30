//! Conv2d backend dispatch.

use std::sync::atomic::Ordering;

use lightnet_cuda::{ConvPrecision, ElementwiseLaunchConfig};

use crate::{OxideBackend, OxideError, OxideTensorF32};

/// Scopes [`ConvPrecision`] to one conv call: the flag is restored on drop, so
/// an error inside the call cannot leave the backend on the f32 path.
pub(crate) struct ConvPrecisionGuard<'a> {
    backend: &'a OxideBackend,
    previous: bool,
}

impl<'a> ConvPrecisionGuard<'a> {
    pub(crate) fn new(backend: &'a OxideBackend, precision: ConvPrecision) -> Self {
        let previous = backend
            .conv_low_precision_off
            .swap(precision == ConvPrecision::NoLowPrecision, Ordering::Relaxed);
        Self { backend, previous }
    }
}

impl Drop for ConvPrecisionGuard<'_> {
    fn drop(&mut self) {
        self.backend
            .conv_low_precision_off
            .store(self.previous, Ordering::Relaxed);
    }
}

impl OxideBackend {
    pub(crate) fn conv2d_f32_backend(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if self.amp {
            return self.conv_forward_amp(
                x,
                weights,
                output_shape,
                input_channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride_x,
                stride_y,
                dilation,
                padding,
                groups,
                config,
            );
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        self.conv_forward(
            x,
            weights,
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config,
        )
    }

    pub(crate) fn conv2d_backward_input_f32_backend(
        &self,
        output_grad: &OxideTensorF32,
        weights: &OxideTensorF32,
        input_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if self.amp {
            return self.conv_backward_input_amp(
                output_grad,
                weights,
                input_shape,
                input_channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride_x,
                stride_y,
                dilation,
                padding,
                groups,
                config,
            );
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let weights_f = self.f32v(weights)?;
        let weights = &*weights_f;
        self.conv_backward_input(
            output_grad,
            weights,
            input_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config,
        )
    }

    pub(crate) fn conv2d_backward_weights_f32_backend(
        &self,
        x: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        weight_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if self.amp {
            return self.conv_backward_weights_amp(
                x,
                output_grad,
                weight_shape,
                input_channels,
                input_height,
                input_width,
                filters,
                kernel_size,
                stride_x,
                stride_y,
                dilation,
                padding,
                groups,
                config,
            );
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        self.conv_backward_weights(
            x,
            output_grad,
            weight_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config,
        )
    }

}

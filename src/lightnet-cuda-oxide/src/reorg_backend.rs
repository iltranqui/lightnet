//! Native reorg backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn reorg_backend_f32(
        &self,
        x: &OxideTensorF32,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        self.reorg_launch(
            "reorg_f32",
            x,
            input_shape,
            output_shape,
            stride,
            reverse,
            false,
            config,
        )
    }

    pub(crate) fn reorg_backend_backward_f32(
        &self,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        // The adjoint swaps the roles: gradient flows output -> input.
        self.reorg_launch(
            "reorg_backward_f32",
            output_grad,
            output_shape,
            input_shape,
            stride,
            reverse,
            true,
            config,
        )
    }
}

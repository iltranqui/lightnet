//! Native Clifford backend operations.

use lightnet_cuda::ElementwiseLaunchConfig;

use crate::{OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn clifford_backend_interaction_f32(
        &self, state: &OxideTensorF32, context: &OxideTensorF32,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.clifford_interaction_native(state, context, batch, channels, spatial, shift, inner, pooled, config)
    }

    pub(crate) fn clifford_backend_interaction_backward_f32(
        &self, state: &OxideTensorF32, context: &OxideTensorF32, grad: &OxideTensorF32,
        batch: usize, channels: usize, spatial: usize, shift: usize,
        inner: bool, pooled: bool, config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        self.clifford_interaction_backward_native(state, context, grad, batch, channels, spatial, shift, inner, pooled, config)
    }

    pub(crate) fn clifford_backend_residual_f32(
        &self, input: &OxideTensorF32, mixed: &OxideTensorF32, scale: &OxideTensorF32,
        mask: &OxideTensorF32, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.clifford_residual_native(input, mixed, scale, mask, batch, channels, spatial, config)
    }

    pub(crate) fn clifford_backend_residual_backward_f32(
        &self, mixed: &OxideTensorF32, grad: &OxideTensorF32, scale: &OxideTensorF32,
        mask: &OxideTensorF32, batch: usize, channels: usize, spatial: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        self.clifford_residual_backward_native(mixed, grad, scale, mask, batch, channels, spatial, config)
    }
}

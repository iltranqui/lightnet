//! Native MambaVision backend operations.

use lightnet_cuda::{mambavision::{MambaScanForward, MambaScanGeometry, MambaScanGrad}, ElementwiseLaunchConfig};

use crate::{OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn mambavision_backend_depthwise_f32(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        bias: &OxideTensorF32,
        batch: usize,
        tokens: usize,
        channels: usize,
        branch: usize,
        kernel: usize,
        causal: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32), OxideError> {
        self.mambavision_depthwise_native(
            input, weights, bias, batch, tokens, channels, branch, kernel, causal, config,
        )
    }

    pub(crate) fn mambavision_backend_depthwise_backward_f32(
        &self,
        input: &OxideTensorF32,
        weights: &OxideTensorF32,
        pre: &OxideTensorF32,
        grad: &OxideTensorF32,
        batch: usize,
        tokens: usize,
        channels: usize,
        branch: usize,
        kernel: usize,
        causal: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        self.mambavision_depthwise_backward_native(
            input, weights, pre, grad, batch, tokens, channels, branch, kernel, causal, config,
        )
    }

    pub(crate) fn mambavision_backend_scan_f32(
        &self,
        x: &OxideTensorF32,
        projection: &OxideTensorF32,
        dt_weights: &OxideTensorF32,
        dt_bias: &OxideTensorF32,
        a_log: &OxideTensorF32,
        d_skip: &OxideTensorF32,
        state_in: &OxideTensorF32,
        geometry: MambaScanGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<MambaScanForward<OxideTensorF32>, OxideError> {
        self.mambavision_scan_native(
            x, projection, dt_weights, dt_bias, a_log, d_skip, state_in, geometry, config,
        )
    }

    pub(crate) fn mambavision_backend_scan_backward_f32(
        &self,
        x: &OxideTensorF32,
        projection: &OxideTensorF32,
        dt_weights: &OxideTensorF32,
        a_log: &OxideTensorF32,
        d_skip: &OxideTensorF32,
        state_in: &OxideTensorF32,
        dt_pre: &OxideTensorF32,
        states: &OxideTensorF32,
        grad: &OxideTensorF32,
        state_grad: &OxideTensorF32,
        geometry: MambaScanGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<MambaScanGrad<OxideTensorF32>, OxideError> {
        self.mambavision_scan_backward_native(
            x, projection, dt_weights, a_log, d_skip, state_in, dt_pre, states, grad,
            state_grad, geometry, config,
        )
    }

    pub(crate) fn mambavision_backend_msa_f32(
        &self,
        input: &OxideTensorF32,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
        block: usize,
        topk: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        self.mambavision_msa_native(input, batch, tokens, channels, heads, block, topk, config)
    }

    pub(crate) fn mambavision_backend_msa_backward_f32(
        &self,
        input: &OxideTensorF32,
        probabilities: &OxideTensorF32,
        active: &OxideTensorF32,
        grad: &OxideTensorF32,
        batch: usize,
        tokens: usize,
        channels: usize,
        heads: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.mambavision_msa_backward_native(
            input, probabilities, active, grad, batch, tokens, channels, heads, config,
        )
    }
}

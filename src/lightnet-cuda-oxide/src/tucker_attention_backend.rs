//! Native Tucker attention backend operations.

use lightnet_cuda::{host_ops::tucker_attention::AttentionGeometry, ElementwiseLaunchConfig};

use crate::{OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn tucker_attention_backend_f32(
        &self,
        q: &OxideTensorF32,
        k: &OxideTensorF32,
        v: &OxideTensorF32,
        geometry: AttentionGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        self.tucker_attention_native(q, k, v, geometry, config)
    }

    pub(crate) fn tucker_attention_backend_backward_f32(
        &self,
        q: &OxideTensorF32,
        k: &OxideTensorF32,
        v: &OxideTensorF32,
        probabilities: &OxideTensorF32,
        active: &OxideTensorF32,
        grad: &OxideTensorF32,
        geometry: AttentionGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        self.tucker_attention_backward_native(q, k, v, probabilities, active, grad, geometry, config)
    }
}

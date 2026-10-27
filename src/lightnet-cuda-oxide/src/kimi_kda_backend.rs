//! Native Kimi KDA backend operations.

use lightnet_cuda::{kimi_kda::{KdaForward, KdaGeometry, KdaGrad}, ElementwiseLaunchConfig};

use crate::{OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn kimi_scan_backend_f32(
        &self,
        q: &OxideTensorF32,
        k: &OxideTensorF32,
        v: &OxideTensorF32,
        beta: &OxideTensorF32,
        decay: &OxideTensorF32,
        geometry: KdaGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<KdaForward<OxideTensorF32>, OxideError> {
        self.kimi_scan_native(q, k, v, beta, decay, geometry, config)
    }

    pub(crate) fn kimi_scan_backend_backward_f32(
        &self,
        q: &OxideTensorF32,
        v: &OxideTensorF32,
        beta: &OxideTensorF32,
        decay: &OxideTensorF32,
        forward: &KdaForward<OxideTensorF32>,
        dy: &OxideTensorF32,
        geometry: KdaGeometry,
        config: ElementwiseLaunchConfig,
    ) -> Result<KdaGrad<OxideTensorF32>, OxideError> {
        self.kimi_scan_backward_native(q, v, beta, decay, forward, dy, geometry, config)
    }

    pub(crate) fn kimi_router_backend_gates_f32(
        &self,
        logits: &OxideTensorF32,
        rows: usize,
        experts: usize,
        top_k: usize,
        temperature: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        self.kimi_router_native(logits, rows, experts, top_k, temperature, config)
    }
}

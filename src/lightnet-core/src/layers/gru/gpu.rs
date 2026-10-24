//! GPU-runtime glue for `[gru]` (inference only).
use crate::error::Result;
use crate::model::{GruSpec, Layer};
use crate::runtime::{GpuExecutableNetwork, GruParams, forward_gru_layer_device};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native inference forward; training is rejected by the caller.
    pub(crate) fn forward_gru_gpu(
        &self,
        layer: &Layer,
        spec: &GruSpec,
        params: &GruParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        forward_gru_layer_device(
            &self.backend,
            current_input,
            layer.input_shape,
            spec,
            params,
            layer.output_shape,
            layer.index,
            launch_config,
        )
    }
}

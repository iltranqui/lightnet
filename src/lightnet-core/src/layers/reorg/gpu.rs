//! GPU-runtime glue for `[reorg]`.
use crate::error::{Error, Result};
use crate::model::{Layer, ReorgSpec};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_reorg_gpu(
        &self,
        layer: &Layer,
        spec: &ReorgSpec,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        if spec.stride == 0 {
            return Err(Error::InvalidConfig(
                "reorg stride cannot be zero".to_owned(),
            ));
        }
        self.backend
            .reorg_f32(
                current_input,
                &tensor_backend_shape(layer.input_shape),
                &tensor_backend_shape(layer.output_shape),
                spec.stride,
                spec.reverse,
                launch_config,
            )
            .map_err(|err| backend_error("reorg", err))
    }
}

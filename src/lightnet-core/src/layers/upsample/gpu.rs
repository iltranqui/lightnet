//! GPU-runtime glue for `[upsample]`.
use crate::error::Result;
use crate::model::{Layer, UpsampleSpec};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_upsample_gpu(
        &self,
        layer: &Layer,
        spec: &UpsampleSpec,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        self.backend
            .upsample_f32(
                current_input,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.stride.unsigned_abs(),
                spec.scale,
                spec.reverse,
                launch_config,
            )
            .map_err(|err| backend_error("upsample", err))
    }
}

//! GPU-runtime glue for `[softmax]`.
use crate::error::Result;
use crate::model::Layer;
use crate::runtime::{
    GpuExecutableNetwork, backend_error, softmax_group_size, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_softmax_gpu(
        &self,
        layer: &Layer,
        groups: usize,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        self.backend
            .softmax_f32(
                current_input,
                &tensor_backend_shape(layer.output_shape),
                softmax_group_size(layer.input_shape, groups)?,
                launch_config,
            )
            .map_err(|err| backend_error("softmax", err))
    }
}

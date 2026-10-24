//! GPU-runtime glue for `[softmax]`.
use crate::error::{Error, Result};
use crate::model::Layer;
use crate::runtime::{
    GradMode,
    GpuForwardState,
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

    /// Softmax backward.
    pub(crate) fn backward_softmax_gpu(
        &self,
        layer: &Layer,
        groups: &usize,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let output = forward.gpu_layer_outputs.get(layer.index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing GPU softmax output for layer #{}",
                layer.index
            ))
        })?;
        let input_grad = if GradMode::from_env() == GradMode::Cpp {
            grad.clone()
        } else {
            self.backend
                .softmax_backward_f32(
                    output,
                    &grad,
                    &tensor_backend_shape(layer.output_shape),
                    softmax_group_size(layer.input_shape, *groups)?,
                    config,
                )
                .map_err(|err| backend_error("GPU softmax backward", err))?
        };
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

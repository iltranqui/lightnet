//! GPU-runtime glue for `[channel_shuffle]`.
use crate::error::Result;
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::model::{ChannelShuffleSpec, Layer};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_channel_shuffle_gpu(
        &self,
        layer: &Layer,
        spec: &ChannelShuffleSpec,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        self.backend
            .channel_shuffle_f32(
                current_input,
                &tensor_backend_shape(layer.input_shape),
                spec.groups,
                false,
                launch_config,
            )
            .map_err(|err| backend_error("channel_shuffle", err))
    }

    /// Channel-shuffle backward.
    pub(crate) fn backward_channel_shuffle_gpu(
        &self,
        layer: &Layer,
        spec: &ChannelShuffleSpec,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let input_grad = self
            .backend
            .channel_shuffle_f32(
                &grad,
                &tensor_backend_shape(layer.output_shape),
                spec.groups,
                true,
                config,
            )
            .map_err(|err| backend_error("GPU channel_shuffle backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

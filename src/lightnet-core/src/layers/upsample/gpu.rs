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

    /// Upsample backward.
    pub(crate) fn backward_upsample_gpu(
        &self,
        layer: &Layer,
        spec: &UpsampleSpec,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let input_grad = self
            .backend
            .upsample_backward_f32(
                &grad,
                &tensor_backend_shape(layer.input_shape),
                layer.input_shape.channels,
                layer.output_shape.height,
                layer.output_shape.width,
                spec.stride.unsigned_abs(),
                spec.scale,
                config,
            )
            .map_err(|err| backend_error("GPU upsample backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

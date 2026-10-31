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

    /// Compact inference forward of `[reorg]`.
    pub(crate) fn forward_reorg_compact_gpu(
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
        let output = self
            .backend
            .reorg_f32(
                current_input,
                &tensor_backend_shape(layer.input_shape),
                &tensor_backend_shape(layer.output_shape),
                spec.stride,
                spec.reverse,
                launch_config,
            )
            .map_err(|err| backend_error("GPU Reorg compact forward", err))?;
        Ok(output)
    }

    /// Reorg backward.
    pub(crate) fn backward_reorg_gpu(
        &self,
        layer: &Layer,
        spec: &ReorgSpec,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let input_grad = self
            .backend
            .reorg_backward_f32(
                &grad,
                &tensor_backend_shape(layer.input_shape),
                &tensor_backend_shape(layer.output_shape),
                spec.stride,
                spec.reverse,
                config,
            )
            .map_err(|err| backend_error("GPU reorg backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

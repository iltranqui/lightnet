//! GPU-runtime glue for `[maxpool]`, `[local_avgpool]` and `[avgpool]`.
use crate::error::{Error, Result};
use crate::model::{Layer, MaxPoolSpec};
use crate::runtime::{
    GpuExecutableNetwork, GpuForwardState, backend_error, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_maxpool_gpu(
        &self,
        layer: &Layer,
        spec: &MaxPoolSpec,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        if spec.depth {
            self.backend.maxpool_depth_f32(
                current_input,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                launch_config,
            )
        } else {
            self.backend.maxpool_f32(
                current_input,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.padding,
                launch_config,
            )
        }
        .map_err(|err| backend_error("maxpool", err))
    }

    pub(crate) fn forward_local_avgpool_gpu(
        &self,
        layer: &Layer,
        spec: &MaxPoolSpec,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        if layer.output_shape.channels != layer.input_shape.channels {
            return Err(Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "local_avgpool_depth".to_owned(),
            });
        }
        self.backend
            .local_avgpool_f32(
                current_input,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.padding,
                launch_config,
            )
            .map_err(|err| backend_error("local_avgpool", err))
    }

    pub(crate) fn forward_avgpool_gpu(
        &self,
        layer: &Layer,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        self.backend
            .avgpool_f32(
                current_input,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                launch_config,
            )
            .map_err(|err| backend_error("avgpool", err))
    }

    /// Max-pool backward.
    pub(crate) fn backward_maxpool_gpu(
        &self,
        layer: &Layer,
        spec: &MaxPoolSpec,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward
                .gpu_layer_outputs
                .get(layer.index - 1)
                .ok_or_else(|| {
                    Error::InvalidConfig(format!(
                        "missing previous GPU layer output for maxpool layer #{}",
                        layer.index
                    ))
                })?
        };
        let input_grad = if spec.depth {
            self.backend.maxpool_depth_backward_f32(
                current_input,
                &grad,
                &tensor_backend_shape(layer.input_shape),
                layer.output_shape.channels,
                config,
            )
        } else {
            self.backend.maxpool_backward_f32(
                current_input,
                &grad,
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.padding,
                config,
            )
        }
        .map_err(|err| backend_error("GPU maxpool backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }

    /// Local average-pool backward.
    pub(crate) fn backward_local_avgpool_gpu(
        &self,
        layer: &Layer,
        spec: &MaxPoolSpec,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let input_grad = self
            .backend
            .local_avgpool_backward_f32(
                &grad,
                &tensor_backend_shape(layer.input_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.padding,
                config,
            )
            .map_err(|err| backend_error("GPU local_avgpool backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }

    /// Average-pool backward.
    pub(crate) fn backward_avgpool_gpu(
        &self,
        layer: &Layer,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let input_grad = self
            .backend
            .avgpool_backward_f32(
                &grad,
                &tensor_backend_shape(layer.input_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                config,
            )
            .map_err(|err| backend_error("GPU avgpool backward", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

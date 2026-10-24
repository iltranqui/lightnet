//! GPU-runtime glue for `[maxpool]`, `[local_avgpool]` and `[avgpool]`.
use crate::error::{Error, Result};
use crate::model::{Layer, MaxPoolSpec};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
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
}

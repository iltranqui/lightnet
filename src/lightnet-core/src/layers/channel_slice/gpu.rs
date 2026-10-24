//! GPU-runtime glue for `[channel_slice]`.
use crate::error::{Error, Result};
use crate::runtime::{GpuExecutableNetwork, backend_error, resolve_relative_layer, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::model::{ChannelSliceSpec, Layer};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_channel_slice_gpu(
        &self,
        layer: &Layer,
        spec: &ChannelSliceSpec,
        gpu_layer_outputs: &[B::Tensor],
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let source_index = resolve_relative_layer(layer.index, spec.from)?;
        let source = gpu_layer_outputs
            .get(source_index)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.from,
            })?;
        let source_channels = self.network.layers[source_index].output_shape.channels;
        self.backend
            .route_concat_f32(
                &[source],
                &[spec.start],
                &[spec.end - spec.start],
                &[source_channels],
                &tensor_backend_shape(layer.output_shape),
                layer.output_shape.width * layer.output_shape.height,
                launch_config,
            )
            .map_err(|err| backend_error("channel_slice", err))
    }
}

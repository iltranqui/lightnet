//! GPU-runtime glue for `[route]`.
use crate::error::{Error, Result};
use crate::runtime::{GpuExecutableNetwork, backend_error, resolve_relative_layer, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::model::{Layer, RouteSpec};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_route_gpu(
        &self,
        layer: &Layer,
        spec: &RouteSpec,
        gpu_layer_outputs: &[B::Tensor],
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let mut route_inputs = Vec::with_capacity(spec.layers.len());
        let mut route_channel_offsets = Vec::with_capacity(spec.layers.len());
        let mut route_channels = Vec::with_capacity(spec.layers.len());
        let mut route_total_channels = Vec::with_capacity(spec.layers.len());
        for raw_index in &spec.layers {
            let source_index = resolve_relative_layer(layer.index, *raw_index)?;
            let source = gpu_layer_outputs
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: *raw_index,
                })?;
            let source_layer = self.network.layers.get(source_index).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing source layer #{source_index} for route layer #{}",
                    layer.index
                ))
            })?;
            let chunk_channels = source_layer.output_shape.channels / spec.groups.max(1);
            route_inputs.push(source);
            route_channel_offsets.push(chunk_channels * spec.group_id);
            route_channels.push(chunk_channels);
            route_total_channels.push(source_layer.output_shape.channels);
        }

        self.backend
            .route_concat_f32(
                &route_inputs,
                &route_channel_offsets,
                &route_channels,
                &route_total_channels,
                &tensor_backend_shape(layer.output_shape),
                layer.output_shape.width * layer.output_shape.height,
                launch_config,
            )
            .map_err(|err| backend_error("route concat", err))
    }
}

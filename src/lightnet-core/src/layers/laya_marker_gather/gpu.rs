//! GPU-runtime glue for `[laya_marker_gather]`.
use crate::error::{Error, Result};
use crate::runtime::{GpuExecutableNetwork, backend_error, resolve_relative_layer, tensor_backend_shape};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

use crate::model::{Layer, LayaMarkerGatherSpec};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    pub(crate) fn forward_laya_marker_gather_gpu(
        &self,
        layer: &Layer,
        spec: &LayaMarkerGatherSpec,
        gpu_layer_outputs: &[B::Tensor],
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let a_index = resolve_relative_layer(layer.index, spec.layers[0])?;
        let b_index = resolve_relative_layer(layer.index, spec.layers[1])?;
        let a_layer = self.network.layers.get(a_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing source layer #{a_index} for [laya_marker_gather] layer #{}",
                layer.index
            ))
        })?;
        let b_layer = self.network.layers.get(b_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing source layer #{b_index} for [laya_marker_gather] layer #{}",
                layer.index
            ))
        })?;
        let a_source = gpu_layer_outputs.get(a_index).ok_or(Error::InvalidRoute {
            layer_index: layer.index,
            reference: spec.layers[0],
        })?;
        let b_source = gpu_layer_outputs.get(b_index).ok_or(Error::InvalidRoute {
            layer_index: layer.index,
            reference: spec.layers[1],
        })?;
        self.backend
            .marker_gather_f32(
                a_source,
                b_source,
                &tensor_backend_shape(a_layer.output_shape),
                &tensor_backend_shape(b_layer.output_shape),
                spec.max_options,
                launch_config,
            )
            .map_err(|err| backend_error("GPU laya_marker_gather", err))
    }

    /// Compact inference forward of `[laya_marker_gather]`.
    pub(crate) fn forward_laya_marker_gather_compact_gpu(
        &self,
        layer: &Layer,
        spec: &LayaMarkerGatherSpec,
        gpu_layer_outputs: &[Option<B::Tensor>],
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let a_index = resolve_relative_layer(layer.index, spec.layers[0])?;
        let b_index = resolve_relative_layer(layer.index, spec.layers[1])?;
        let a_layer = self.network.layers.get(a_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing source layer #{a_index} for [laya_marker_gather] layer #{}",
                layer.index
            ))
        })?;
        let b_layer = self.network.layers.get(b_index).ok_or_else(|| {
            Error::InvalidConfig(format!(
                "missing source layer #{b_index} for [laya_marker_gather] layer #{}",
                layer.index
            ))
        })?;
        let a_source = gpu_layer_outputs
            .get(a_index)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.layers[0],
            })?;
        let b_source = gpu_layer_outputs
            .get(b_index)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.layers[1],
            })?;
        let output = self
            .backend
            .marker_gather_f32(
                a_source,
                b_source,
                &tensor_backend_shape(a_layer.output_shape),
                &tensor_backend_shape(b_layer.output_shape),
                spec.max_options,
                launch_config,
            )
            .map_err(|err| backend_error("GPU laya_marker_gather", err))?;
        Ok(output)
    }
}

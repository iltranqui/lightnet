//! GPU-runtime glue for `[scale_channels]`.
use crate::error::{Error, Result};
use crate::model::{Layer, ScaleChannelsSpec};
use crate::runtime::{
    GpuExecutableNetwork, backend_error, forward_scale_channels, gpu_activation_kind,
    resolve_relative_layer,
    tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Returns the output and, when the layer has an activation, the
    /// pre-activation tensor that backward needs.
    pub(crate) fn forward_scale_channels_gpu(
        &self,
        layer: &Layer,
        spec: &ScaleChannelsSpec,
        current_input: &B::Tensor,
        gpu_layer_outputs: &[B::Tensor],
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<B::Tensor>)> {
        let source_index = resolve_relative_layer(layer.index, spec.from)?;
        let source_layer = self
            .network
            .layers
            .get(source_index)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.from,
            })?;
        let source_gpu = gpu_layer_outputs
            .get(source_index)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.from,
            })?;
        if let Some(activation) = spec.activation.to_cuda_f32_kind() {
            if source_layer.output_shape != layer.output_shape {
                return Err(Error::InvalidConfig(format!(
                    "scale_channels source/output mismatch: {} vs {}",
                    source_layer.output_shape, layer.output_shape
                )));
            }
            let pre_activation = self
                .backend
                .scale_channels_f32(
                    current_input,
                    source_gpu,
                    &tensor_backend_shape(layer.output_shape),
                    spec.scale_wh,
                    launch_config,
                )
                .map_err(|err| backend_error("scale_channels", err))?;
            let output = self
                .backend
                .activation_f32(&pre_activation, activation, launch_config)
                .map_err(|err| backend_error("scale_channels activation", err))?;
            Ok((output, Some(pre_activation)))
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download scale_channels input", err))?;
            let host_source = self
                .download_tensor(source_gpu, source_layer.output_shape)
                .map_err(|err| backend_error("download scale_channels source", err))?;
            let (host_output, host_activation_input) =
                forward_scale_channels(spec, &host_input, &host_source, layer.output_shape)?;
            let activation_cache = match host_activation_input {
                Some(activation_input) => Some(
                    self.backend
                        .upload_f32(
                            &tensor_backend_shape(layer.output_shape),
                            &activation_input.data,
                        )
                        .map_err(|err| {
                            backend_error("upload scale_channels activation cache", err)
                        })?,
                ),
                None => None,
            };
            let output = self
                .backend
                .upload_f32(&tensor_backend_shape(layer.output_shape), &host_output.data)
                .map_err(|err| backend_error("upload scale_channels output", err))?;
            Ok((output, activation_cache))
        }
    }

    /// Compact inference forward of `[scale_channels]`.
    pub(crate) fn forward_scale_channels_compact_gpu(
        &self,
        layer: &Layer,
        spec: &ScaleChannelsSpec,
        gpu_layer_outputs: &[Option<B::Tensor>],
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let source_index = resolve_relative_layer(layer.index, spec.from)?;
        let source = gpu_layer_outputs
            .get(source_index)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.from,
            })?;
        let pre_activation = self
            .backend
            .scale_channels_f32(
                current_input,
                source,
                &tensor_backend_shape(layer.output_shape),
                spec.scale_wh,
                launch_config,
            )
            .map_err(|err| backend_error("GPU ScaleChannels compact", err))?;
        let output = self
            .backend
            .activation_f32(
                &pre_activation,
                gpu_activation_kind(layer.index, &spec.activation)?,
                launch_config,
            )
            .map_err(|err| backend_error("GPU ScaleChannels compact activation", err))?;
        Ok(output)
    }
}

//! GPU-runtime glue for `[sam]`.
use crate::error::{Error, Result};
use crate::model::{Layer, SamSpec};
use crate::runtime::{
    gpu_activation_kind,
    GpuForwardState,
    GpuExecutableNetwork, backend_error, forward_sam, resolve_relative_layer, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Returns the output and, when the layer has an activation, the
    /// pre-activation tensor that backward needs.
    pub(crate) fn forward_sam_gpu(
        &self,
        layer: &Layer,
        spec: &SamSpec,
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
        if layer.input_shape != layer.output_shape || source_layer.output_shape != layer.output_shape
        {
            return Err(Error::InvalidConfig(format!(
                "sam shape mismatch: input {}, source {}, output {}",
                layer.input_shape, source_layer.output_shape, layer.output_shape
            )));
        }
        if let Some(activation) = spec.activation.to_cuda_f32_kind() {
            let pre_activation = self
                .backend
                .vector_mul_f32(
                    current_input,
                    source_gpu,
                    &tensor_backend_shape(layer.output_shape),
                    launch_config,
                )
                .map_err(|err| backend_error("sam multiply", err))?;
            let output = self
                .backend
                .activation_f32(&pre_activation, activation, launch_config)
                .map_err(|err| backend_error("sam activation", err))?;
            Ok((output, Some(pre_activation)))
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download sam input", err))?;
            let host_source = self
                .download_tensor(source_gpu, source_layer.output_shape)
                .map_err(|err| backend_error("download sam source", err))?;
            let (host_output, host_activation_input) =
                forward_sam(spec, &host_input, &host_source, layer.output_shape)?;
            let activation_cache = match host_activation_input {
                Some(activation_input) => Some(
                    self.backend
                        .upload_f32(
                            &tensor_backend_shape(layer.output_shape),
                            &activation_input.data,
                        )
                        .map_err(|err| backend_error("upload sam activation cache", err))?,
                ),
                None => None,
            };
            let output = self
                .backend
                .upload_f32(&tensor_backend_shape(layer.output_shape), &host_output.data)
                .map_err(|err| backend_error("upload sam output", err))?;
            Ok((output, activation_cache))
        }
    }

    /// Compact inference forward of `[sam]`.
    pub(crate) fn forward_sam_compact_gpu(
        &self,
        layer: &Layer,
        spec: &SamSpec,
        gpu_layer_outputs: &[Option<B::Tensor>],
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let source_index = resolve_relative_layer(layer.index, spec.from)?;
        let source_layer =
            self.network
                .layers
                .get(source_index)
                .ok_or(Error::InvalidRoute {
                    layer_index: layer.index,
                    reference: spec.from,
                })?;
        let source_gpu = gpu_layer_outputs
            .get(source_index)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.from,
            })?;
        if layer.input_shape != layer.output_shape
            || source_layer.output_shape != layer.output_shape
        {
            return Err(Error::InvalidConfig(format!(
                "sam shape mismatch: input {}, source {}, output {}",
                layer.input_shape, source_layer.output_shape, layer.output_shape
            )));
        }
        let input_gpu = current_input;
        Ok(if let Some(activation) = spec.activation.to_cuda_f32_kind() {
            let pre_activation = self
                .backend
                .vector_mul_f32(
                    input_gpu,
                    source_gpu,
                    &tensor_backend_shape(layer.output_shape),
                    launch_config,
                )
                .map_err(|err| backend_error("sam multiply", err))?;
            self.backend
                .activation_f32(&pre_activation, activation, launch_config)
                .map_err(|err| backend_error("sam activation", err))?
        } else {
            let host_input = self
                .download_tensor(input_gpu, layer.input_shape)
                .map_err(|err| backend_error("download sam input", err))?;
            let host_source = self
                .download_tensor(source_gpu, source_layer.output_shape)
                .map_err(|err| backend_error("download sam source", err))?;
            let (host_output, _) =
                forward_sam(spec, &host_input, &host_source, layer.output_shape)?;
            self.backend
                .upload_f32(
                    &tensor_backend_shape(layer.output_shape),
                    &host_output.data,
                )
                .map_err(|err| backend_error("upload sam output", err))?
        })
    }

    /// Sam backward: gradients for the input and the source layer.
    pub(crate) fn backward_sam_gpu(
        &self,
        layer: &Layer,
        spec: &SamSpec,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let source_index = resolve_relative_layer(layer.index, spec.from)?;
        let source_layer = self.network.layers.get(source_index).ok_or(
            Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.from,
            },
        )?;
        if layer.input_shape != layer.output_shape
            || source_layer.output_shape != layer.output_shape
        {
            return Err(Error::InvalidConfig(format!(
                "sam backward shape mismatch: input {}, source {}, output {}",
                layer.input_shape, source_layer.output_shape, layer.output_shape
            )));
        }
        let activation_input = forward.gpu_layer_activation_inputs[layer.index]
            .as_ref()
            .ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing GPU activation cache for sam layer #{}",
                    layer.index
                ))
            })?;
        let activation = gpu_activation_kind(layer.index, &spec.activation)?;
        let delta = self
            .backend
            .activation_backward_f32(activation_input, &grad, activation, config)
            .map_err(|err| backend_error("GPU sam activation backward", err))?;
        let source = forward
            .gpu_layer_outputs
            .get(source_index)
            .ok_or(Error::InvalidRoute {
                layer_index: layer.index,
                reference: spec.from,
            })?;
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU layer output for sam layer #{}",
                    layer.index
                ))
            })?
        };
        let shape = tensor_backend_shape(layer.output_shape);
        let input_grad = self
            .backend
            .vector_mul_f32(&delta, source, &shape, config)
            .map_err(|err| backend_error("GPU sam input gradient", err))?;
        let source_grad = self
            .backend
            .vector_mul_f32(&delta, current_input, &shape, config)
            .map_err(|err| backend_error("GPU sam source gradient", err))?;
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_grad,
            layer_output_gradients,
        )?;
        self.accumulate_gpu_gradient(
            &mut layer_output_gradients[source_index],
            source_grad,
        )?;
        Ok(())
    }
}

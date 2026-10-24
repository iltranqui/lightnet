//! GPU-runtime glue for `[local]`.
use crate::error::Result;
use crate::model::{Layer, LocalSpec};
use crate::runtime::{
    GpuExecutableNetwork, LocalParams, backend_error, forward_local_with_cache,
    tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native locally-connected forward for device activations; other activations
    /// round-trip through the host reference. Returns the output and the
    /// pre-activation tensor kept for backward.
    pub(crate) fn forward_local_gpu(
        &self,
        layer: &Layer,
        spec: &LocalSpec,
        local: &LocalParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<B::Tensor>)> {
        let mut activation_input_gpu = None;
        let output = {
            if let Some(activation) = spec.activation.to_cuda_f32_kind() {
                let locations = layer.output_shape.width * layer.output_shape.height;
                let weights = self
                    .backend
                    .upload_f32(
                        &[
                            locations,
                            spec.filters,
                            layer.input_shape.channels,
                            spec.size,
                            spec.size,
                        ],
                        &local.weights,
                    )
                    .map_err(|err| backend_error("upload local weights", err))?;
                let biases = self
                    .backend
                    .upload_f32(&[locations, spec.filters], &local.biases)
                    .map_err(|err| backend_error("upload local biases", err))?;
                let pre_activation = self
                    .backend
                    .local_f32(
                        current_input,
                        &weights,
                        &biases,
                        &tensor_backend_shape(layer.input_shape),
                        &tensor_backend_shape(layer.output_shape),
                        spec.filters,
                        spec.size,
                        spec.stride_x,
                        spec.stride_y,
                        spec.padding,
                        launch_config,
                    )
                    .map_err(|err| backend_error("local forward", err))?;
                activation_input_gpu = Some(pre_activation.clone());
                self.backend
                    .activation_f32(&pre_activation, activation, launch_config)
                    .map_err(|err| backend_error("local activation", err))?
            } else {
                let host_input = self
                    .download_tensor(current_input, layer.input_shape)
                    .map_err(|err| backend_error("download local input", err))?;
                let (host_output, host_activation_input) =
                    forward_local_with_cache(spec, local, &host_input, layer.output_shape)?;
                if let Some(pre_activation) = host_activation_input {
                    activation_input_gpu = Some(
                        self.backend
                            .upload_f32(
                                &tensor_backend_shape(layer.output_shape),
                                &pre_activation.data,
                            )
                            .map_err(|err| {
                                backend_error("upload local activation cache", err)
                            })?,
                    );
                }
                self.backend
                    .upload_f32(
                        &tensor_backend_shape(layer.output_shape),
                        &host_output.data,
                    )
                    .map_err(|err| backend_error("upload local output", err))?
            }
        };
        Ok((output, activation_input_gpu))
    }
}

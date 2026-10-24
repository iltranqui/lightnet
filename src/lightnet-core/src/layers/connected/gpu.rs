//! GPU-runtime glue for `[connected]`.
use crate::error::{Error, Result};
use crate::model::{ConnectedSpec, Layer};
use crate::runtime::{
    BATCH_NORM_EPSILON, ConnectedParams, GpuExecutableNetwork, backend_error,
    forward_connected_with_cache, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native fully-connected forward with inference batch norm folded into the weights;
    /// activations without a device kind round-trip through the host reference. Returns
    /// the output and the pre-activation tensor kept for backward.
    pub(crate) fn forward_connected_gpu(
        &self,
        layer: &Layer,
        spec: &ConnectedSpec,
        connected: &ConnectedParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<B::Tensor>)> {
        let mut activation_input_gpu = None;
        let inputs = layer.input_shape.element_count();
        if connected.weights.len() != spec.output * inputs
            || connected.biases.len() != spec.output
        {
            return Err(Error::InvalidConfig(format!(
                "connected layer #{} expects {} weights and {} biases, got {} and {}",
                layer.index,
                spec.output * inputs,
                spec.output,
                connected.weights.len(),
                connected.biases.len()
            )));
        }
        let output = if let Some(activation) = spec.activation.to_cuda_f32_kind() {
            // Inference batch norm is a per-output affine map, so
            // fold it into the uploaded weights and biases.
            let (weights, biases) = match connected.batch_norm.as_ref() {
                Some(bn) => {
                    let mut weights = connected.weights.clone();
                    let mut biases = connected.biases.clone();
                    for (o, row) in weights.chunks_mut(inputs).enumerate() {
                        let k = bn.scales[o]
                            / (bn.rolling_variance[o] + BATCH_NORM_EPSILON).sqrt();
                        row.iter_mut().for_each(|w| *w *= k);
                        biases[o] -= bn.rolling_mean[o] * k;
                    }
                    (weights, biases)
                }
                None => (connected.weights.clone(), connected.biases.clone()),
            };
            let weights_gpu = self
                .backend
                .upload_f32(&[spec.output, inputs], &weights)
                .map_err(|err| backend_error("upload connected weights", err))?;
            let biases_gpu = self
                .backend
                .upload_f32(&[spec.output], &biases)
                .map_err(|err| backend_error("upload connected biases", err))?;
            let pre_activation = self
                .backend
                .connected_f32(
                    current_input,
                    &weights_gpu,
                    &biases_gpu,
                    layer.input_shape.batch,
                    inputs,
                    spec.output,
                    launch_config,
                )
                .map_err(|err| backend_error("connected", err))?;
            activation_input_gpu = Some(pre_activation.clone());
            self.backend
                .activation_f32(&pre_activation, activation, launch_config)
                .map_err(|err| backend_error("connected activation", err))?
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download connected input", err))?;
            let (host_output, host_activation_input) = forward_connected_with_cache(
                spec,
                connected,
                &host_input,
                layer.output_shape,
            );
            if let Some(activation_input) = host_activation_input {
                activation_input_gpu = Some(
                    self.backend
                        .upload_f32(
                            &tensor_backend_shape(layer.output_shape),
                            &activation_input.data,
                        )
                        .map_err(|err| {
                            backend_error("upload connected activation cache", err)
                        })?,
                );
            }
            self.backend
                .upload_f32(
                    &tensor_backend_shape(layer.output_shape),
                    &host_output.data,
                )
                .map_err(|err| backend_error("upload connected output", err))?
        };
        Ok((output, activation_input_gpu))
    }
}

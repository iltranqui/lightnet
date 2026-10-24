//! GPU-runtime glue for `[deconvolutional]`.
use crate::error::Result;
use crate::model::{DeconvolutionalSpec, Layer};
use crate::runtime::{
    ConvolutionalParams, GpuExecutableNetwork, activate, backend_error,
    deconvolutional_spec_is_native, gpu_activation_kind, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native deconvolution, bias, and activation or channel normalization when the spec
    /// is supported; other specs round-trip through the host reference. Returns the output
    /// and the biased pre-activation tensor kept for backward.
    pub(crate) fn forward_deconvolutional_gpu(
        &self,
        layer: &Layer,
        spec: &DeconvolutionalSpec,
        params: &ConvolutionalParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, B::Tensor)> {
        let activation_input_gpu;
        let output = if deconvolutional_spec_is_native(spec) {
            let weights = self
                .backend
                .upload_f32(
                    &[
                        layer.input_shape.channels,
                        spec.filters,
                        spec.size,
                        spec.size,
                    ],
                    &params.weights,
                )
                .map_err(|err| backend_error("upload deconvolutional weights", err))?;
            let raw = self
                .backend
                .deconvolutional_f32(
                    current_input,
                    &weights,
                    &tensor_backend_shape(layer.output_shape),
                    layer.input_shape.channels,
                    layer.input_shape.height,
                    layer.input_shape.width,
                    spec.filters,
                    spec.size,
                    spec.stride,
                    launch_config,
                )
                .map_err(|err| backend_error("GPU deconvolution", err))?;
            let bias = self
                .backend
                .upload_f32(&[spec.filters], &params.biases)
                .map_err(|err| backend_error("upload deconvolutional biases", err))?;
            let biased = self
                .backend
                .bias_add_f32(
                    &raw,
                    &bias,
                    spec.filters,
                    layer.output_shape.height * layer.output_shape.width,
                    launch_config,
                )
                .map_err(|err| backend_error("deconvolution bias add", err))?;
            activation_input_gpu = biased.clone();
            if let Some(normalization) = spec.activation.to_channel_normalization() {
                self.backend
                    .normalize_channels_f32(
                        &biased,
                        &tensor_backend_shape(layer.output_shape),
                        layer.output_shape.batch,
                        layer.output_shape.channels,
                        layer.output_shape.width * layer.output_shape.height,
                        normalization,
                        launch_config,
                    )
                    .map_err(|err| {
                        backend_error("deconvolution channel normalization", err)
                    })?
            } else {
                self.backend
                    .activation_f32(
                        &biased,
                        gpu_activation_kind(layer.index, &spec.activation)?,
                        launch_config,
                    )
                    .map_err(|err| backend_error("deconvolution activation", err))?
            }
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download deconvolutional input", err))?;
            let pre = crate::deconvolutional::deconv_forward(
                &host_input,
                &params.weights,
                &params.biases,
                spec.size,
                spec.stride,
                layer.output_shape,
            );
            activation_input_gpu = self
                .backend
                .upload_f32(&tensor_backend_shape(layer.output_shape), &pre.data)
                .map_err(|err| backend_error("upload deconvolutional activation cache", err))?;
            let activated: Vec<f32> = pre
                .data
                .iter()
                .map(|value| activate(spec.activation.clone(), *value))
                .collect();
            self.backend
                .upload_f32(&tensor_backend_shape(layer.output_shape), &activated)
                .map_err(|err| backend_error("upload deconvolutional output", err))?
        };
        Ok((output, activation_input_gpu))
    }
}

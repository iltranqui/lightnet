//! GPU-runtime glue for `[convolutional]`.
use crate::error::Result;
use crate::model::{ConvolutionalSpec, Layer};
use crate::runtime::{
    GpuBatchNormCache, GpuExecutableNetwork, backend_error, gpu_activation_kind,
    resolve_convolutional_owner, resolve_convolutional_params, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native convolution, then batch-statistics batch norm (training) or bias, then
    /// activation or channel normalization. Returns the output, the pre-activation tensor kept
    /// for backward, and the batch-norm cache when batch statistics were used.
    pub(crate) fn forward_convolutional_gpu(
        &self,
        layer: &Layer,
        spec: &ConvolutionalSpec,
        current_input: &B::Tensor,
        batch_norm_training: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        B::Tensor,
        Option<GpuBatchNormCache<B::Tensor>>,
    )> {
        let mut batch_norm_cache = None;
        let conv = resolve_convolutional_params(&self.network, &self.params, layer.index)?;
        let owner = resolve_convolutional_owner(&self.network, layer.index)?;
        let cached = if batch_norm_training && conv.batch_norm.is_some() {
            self.cached_convolution_params_raw(
                owner,
                layer.input_shape.channels,
                spec,
                conv,
            )?
        } else {
            self.cached_convolution_params(
                owner,
                layer.input_shape.channels,
                spec,
                conv,
            )?
        };
        // Batch-norm training convolves with the raw weights; the
        // folded (rolling-stat) weights serve inference.
        let weights = if batch_norm_training && conv.batch_norm.is_some() {
            &cached.raw_weights
        } else {
            &cached.weights
        };
        let conv_output = self
            .backend
            .conv2d_f32(
                current_input,
                weights,
                &tensor_backend_shape(layer.output_shape),
                layer.input_shape.channels,
                layer.input_shape.height,
                layer.input_shape.width,
                spec.filters,
                spec.size,
                spec.stride_x,
                spec.stride_y,
                spec.dilation,
                spec.padding,
                spec.groups.max(1),
                launch_config,
            )
            .map_err(|err| backend_error("convolution", err))?;
        let biased = match conv.batch_norm.as_ref() {
            Some(batch_norm) if batch_norm_training => {
                let (normalized, cache) = self.forward_batchnorm_training_gpu(
                    &conv_output,
                    &cached,
                    batch_norm,
                    layer.output_shape,
                    spec.share_index.is_none(),
                    launch_config,
                )?;
                batch_norm_cache = Some(cache);
                normalized
            }
            _ => self
                .backend
                .bias_add_f32(
                    &conv_output,
                    &cached.biases,
                    spec.filters,
                    layer.output_shape.width * layer.output_shape.height,
                    launch_config,
                )
                .map_err(|err| backend_error("convolution bias add", err))?,
        };
        let activation_input = biased.clone();
        let output = if let Some(normalization) = spec.activation.to_channel_normalization() {
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
                    backend_error("convolution channel normalization", err)
                })?
        } else {
            let activation = gpu_activation_kind(layer.index, &spec.activation)?;
            self.backend
                .activation_f32(&biased, activation, launch_config)
                .map_err(|err| backend_error("convolution activation", err))?
        };
        Ok((output, activation_input, batch_norm_cache))
    }
}

//! GPU-runtime glue for `[rnn]`.
use crate::error::{Error, Result};
use crate::model::{Activation, Layer, RnnSpec};
use crate::runtime::{
    accumulate_connected_train_buffer,
    backward_rnn_layer_device,
    upload_rnn_backward_cache,
    LayerTrainBuffers,
    GpuForwardState,
    GpuExecutableNetwork, GpuRnnBackwardCache, RecurrentForwardCache, RnnParams, backend_error,
    forward_rnn_layer, forward_rnn_layer_device, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native sequence forward when the activations have device kinds; otherwise the host
    /// reference. Returns the output, the host cache for CPU backward, and the device cache.
    pub(crate) fn forward_rnn_gpu(
        &self,
        layer: &Layer,
        spec: &RnnSpec,
        recurrent: &RnnParams,
        current_input: &B::Tensor,
        host_copies: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        Option<RecurrentForwardCache>,
        Option<GpuRnnBackwardCache<B::Tensor>>,
    )> {
        if spec.activation.to_cuda_f32_kind().is_some()
            && (spec.logistic != 2 || Activation::Loggy.to_cuda_f32_kind().is_some())
        {
            let (output, cache, device_cache) = forward_rnn_layer_device(
                &self.backend,
                current_input,
                layer.input_shape,
                spec,
                recurrent,
                layer.output_shape,
                layer.index,
                host_copies,
                launch_config,
            )?;
            Ok((
                output,
                cache.map(RecurrentForwardCache::Rnn),
                Some(device_cache),
            ))
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download RNN input", err))?;
            let (host_output, cache) =
                forward_rnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
            let output = self
                .backend
                .upload_f32(
                    &tensor_backend_shape(layer.output_shape),
                    &host_output.data,
                )
                .map_err(|err| backend_error("upload RNN output", err))?;
            Ok((output, Some(RecurrentForwardCache::Rnn(cache)), None))
        }
    }

    /// Native RNN backward: input gradient plus accumulated parameter gradients.
    pub(crate) fn backward_rnn_gpu(
        &self,
        layer: &Layer,
        spec: &RnnSpec,
        params: &RnnParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let has_batch_norm = [
            params.input.batch_norm.is_some(),
            params.self_params.batch_norm.is_some(),
            params.output.batch_norm.is_some(),
        ];
        if has_batch_norm.iter().any(|present| *present != spec.batch_normalize) {
            return Err(Error::InvalidConfig(format!(
                "RNN batch-normalization parameter mismatch at layer #{}",
                layer.index
            )));
        }
        let cache = if let Some(cache) = forward
            .gpu_rnn_caches
            .get(layer.index)
            .and_then(Option::as_ref)
        {
            cache.clone()
        } else {
            let Some(Some(RecurrentForwardCache::Rnn(cache))) =
                forward.cpu_forward.recurrent_caches.get(layer.index)
            else {
                return Err(Error::InvalidConfig(format!(
                    "missing RNN forward cache for GPU layer #{}",
                    layer.index
                )));
            };
            upload_rnn_backward_cache(&self.backend, cache)?
        };
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU output for RNN layer #{}",
                    layer.index
                ))
            })?
        };
        let (input_gradient, gradients) = backward_rnn_layer_device(
            &self.backend,
            spec,
            params,
            &cache,
            current_input,
            layer.input_shape,
            &grad,
            layer.index,
            config,
        )?;
        {
            let mut train_buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let Some(LayerTrainBuffers::Rnn(train)) = train_buffers.get_mut(layer.index) else {
                return Err(Error::InvalidConfig(format!(
                    "missing RNN train buffers for layer #{}",
                    layer.index
                )));
            };
            accumulate_connected_train_buffer(&mut train.input, &gradients.input)?;
            accumulate_connected_train_buffer(&mut train.self_params, &gradients.self_params)?;
            accumulate_connected_train_buffer(&mut train.output, &gradients.output)?;
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_gradient,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

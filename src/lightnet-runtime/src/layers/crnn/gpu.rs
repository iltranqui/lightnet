//! GPU-runtime glue for `[crnn]`.
use crate::error::{Error, Result};
use crate::model::{CrnnSpec, Layer};
use crate::runtime::{
    accumulate_convolutional_train_buffer,
    backward_crnn_layer_device,
    upload_crnn_backward_cache,
    LayerTrainBuffers,
    GpuForwardState,
    CrnnParams, GpuCrnnBackwardCache, GpuExecutableNetwork, RecurrentForwardCache, backend_error,
    forward_crnn_layer, forward_crnn_layer_device, tensor_backend_shape,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native sequence forward when the activation has a device kind; otherwise the host
    /// reference. Returns the output, the host cache for CPU backward, and the device cache.
    pub(crate) fn forward_crnn_gpu(
        &self,
        layer: &Layer,
        spec: &CrnnSpec,
        recurrent: &CrnnParams,
        current_input: &B::Tensor,
        host_copies: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        Option<RecurrentForwardCache>,
        Option<GpuCrnnBackwardCache<B::Tensor>>,
    )> {
        if spec.activation.to_cuda_f32_kind().is_some() {
            let (output, cache, device_cache) = forward_crnn_layer_device(
                &self.backend,
                spec,
                recurrent,
                current_input,
                layer.input_shape,
                layer.output_shape,
                layer.index,
                host_copies,
                launch_config,
            )?;
            Ok((
                output,
                cache.map(RecurrentForwardCache::Crnn),
                Some(device_cache),
            ))
        } else {
            let host_input = self
                .download_tensor(current_input, layer.input_shape)
                .map_err(|err| backend_error("download CRNN input", err))?;
            let (host_output, cache) =
                forward_crnn_layer(spec, recurrent, &host_input, layer.output_shape)?;
            let output = self
                .backend
                .upload_f32(
                    &tensor_backend_shape(layer.output_shape),
                    &host_output.data,
                )
                .map_err(|err| backend_error("upload CRNN output", err))?;
            Ok((output, Some(RecurrentForwardCache::Crnn(cache)), None))
        }
    }

    /// Native CRNN backward: input gradient plus accumulated parameter gradients.
    pub(crate) fn backward_crnn_gpu(
        &self,
        layer: &Layer,
        spec: &CrnnSpec,
        params: &CrnnParams,
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
        if has_batch_norm
            .iter()
            .any(|present| *present != spec.batch_normalize)
        {
            return Err(Error::InvalidConfig(format!(
                "CRNN batch-normalization parameter mismatch at layer #{}",
                layer.index
            )));
        }
        let cache = if let Some(cache) = forward
            .gpu_crnn_caches
            .get(layer.index)
            .and_then(Option::as_ref)
        {
            cache.clone()
        } else {
            let Some(Some(RecurrentForwardCache::Crnn(cache))) =
                forward.cpu_forward.recurrent_caches.get(layer.index)
            else {
                return Err(Error::InvalidConfig(format!(
                    "missing CRNN forward cache for GPU layer #{}",
                    layer.index
                )));
            };
            upload_crnn_backward_cache(&self.backend, cache)?
        };
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU output for CRNN layer #{}",
                    layer.index
                ))
            })?
        };
        let (input_gradient, gradients) = backward_crnn_layer_device(
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
            let Some(LayerTrainBuffers::Crnn(train)) =
                train_buffers.get_mut(layer.index)
            else {
                return Err(Error::InvalidConfig(format!(
                    "missing CRNN train buffers for layer #{}",
                    layer.index
                )));
            };
            accumulate_convolutional_train_buffer(
                &mut train.input,
                &gradients.input,
            )?;
            accumulate_convolutional_train_buffer(
                &mut train.self_params,
                &gradients.self_params,
            )?;
            accumulate_convolutional_train_buffer(
                &mut train.output,
                &gradients.output,
            )?;
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_gradient,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

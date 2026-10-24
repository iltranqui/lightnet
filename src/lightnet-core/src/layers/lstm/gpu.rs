//! GPU-runtime glue for `[lstm]`.
use crate::error::{Error, Result};
use crate::model::{Layer, LstmSpec};
use crate::runtime::{
    accumulate_connected_train_buffer,
    backward_lstm_layer_device,
    upload_lstm_backward_cache,
    LayerTrainBuffers,
    GpuForwardState,
    GpuExecutableNetwork, GpuLstmBackwardCache, LstmParams, RecurrentForwardCache,
    forward_lstm_layer_device,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native sequence forward. Returns the output, the host cache for CPU backward, and
    /// the device cache.
    pub(crate) fn forward_lstm_gpu(
        &self,
        layer: &Layer,
        spec: &LstmSpec,
        recurrent: &LstmParams,
        current_input: &B::Tensor,
        host_copies: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(
        B::Tensor,
        Option<RecurrentForwardCache>,
        GpuLstmBackwardCache<B::Tensor>,
    )> {
        let (output, cache, device_cache) = forward_lstm_layer_device(
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
            cache.map(RecurrentForwardCache::Lstm),
            device_cache,
        ))
    }

    /// Native LSTM backward: input gradient plus accumulated parameter gradients.
    pub(crate) fn backward_lstm_gpu(
        &self,
        layer: &Layer,
        spec: &LstmSpec,
        params: &LstmParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        let has_batch_norm = [
            params.wf.batch_norm.is_some(),
            params.wi.batch_norm.is_some(),
            params.wg.batch_norm.is_some(),
            params.wo.batch_norm.is_some(),
            params.uf.batch_norm.is_some(),
            params.ui.batch_norm.is_some(),
            params.ug.batch_norm.is_some(),
            params.uo.batch_norm.is_some(),
        ];
        if has_batch_norm
            .iter()
            .any(|present| *present != spec.batch_normalize)
        {
            return Err(Error::InvalidConfig(format!(
                "LSTM batch-normalization parameter mismatch at layer #{}",
                layer.index
            )));
        }
        let cache = if let Some(cache) = forward
            .gpu_lstm_caches
            .get(layer.index)
            .and_then(Option::as_ref)
        {
            cache.clone()
        } else {
            let Some(Some(RecurrentForwardCache::Lstm(cache))) =
                forward.cpu_forward.recurrent_caches.get(layer.index)
            else {
                return Err(Error::InvalidConfig(format!(
                    "missing LSTM forward cache for GPU layer #{}",
                    layer.index
                )));
            };
            upload_lstm_backward_cache(&self.backend, cache)?
        };
        let current_input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            forward.gpu_layer_outputs.get(layer.index - 1).ok_or_else(|| {
                Error::InvalidConfig(format!(
                    "missing previous GPU output for LSTM layer #{}",
                    layer.index
                ))
            })?
        };
        let (input_gradient, gradients) = backward_lstm_layer_device(
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
            let Some(LayerTrainBuffers::Lstm(train)) =
                train_buffers.get_mut(layer.index)
            else {
                return Err(Error::InvalidConfig(format!(
                    "missing LSTM train buffers for layer #{}",
                    layer.index
                )));
            };
            accumulate_connected_train_buffer(&mut train.wf, &gradients.wf)?;
            accumulate_connected_train_buffer(&mut train.wi, &gradients.wi)?;
            accumulate_connected_train_buffer(&mut train.wg, &gradients.wg)?;
            accumulate_connected_train_buffer(&mut train.wo, &gradients.wo)?;
            accumulate_connected_train_buffer(&mut train.uf, &gradients.uf)?;
            accumulate_connected_train_buffer(&mut train.ui, &gradients.ui)?;
            accumulate_connected_train_buffer(&mut train.ug, &gradients.ug)?;
            accumulate_connected_train_buffer(&mut train.uo, &gradients.uo)?;
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            input_gradient,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

//! GPU-runtime glue for `[lstm]`.
use crate::error::Result;
use crate::model::{Layer, LstmSpec};
use crate::runtime::{
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
}

//! GPU-runtime glue for `[kimi_kda]`: host forward behind the device tensor boundary.
use crate::error::{Error, Result};
use crate::model::{KimiKdaSpec, Layer};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::F32TensorOpsBackend;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Download the input, run the host Kimi KDA forward, upload the output.
    pub(crate) fn forward_kimi_kda_host(
        &self,
        layer: &Layer,
        spec: &KimiKdaSpec,
        weights: &super::KimiKdaParams,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::KimiKdaForwardCache)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download Kimi KDA input", err))?;
        let (data, cache) =
            super::kimi_kda_forward(&host_input.data, host_input.shape, spec, weights)
                .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
            .map_err(|err| backend_error("upload Kimi KDA output", err))?;
        Ok((output, cache))
    }
}

//! GPU-runtime glue for `[swin]`: host forward behind the device tensor boundary.
use crate::error::{Error, Result};
use crate::model::{Layer, SwinSpec};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::F32TensorOpsBackend;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Download the input, run the host Swin forward, upload the output.
    pub(crate) fn forward_swin_host(
        &self,
        layer: &Layer,
        spec: &SwinSpec,
        weights: &super::SwinWeights,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::block::SwinForwardCache)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download swin input", err))?;
        let geometry = super::block::SwinGeometry {
            batch: layer.input_shape.batch,
            channels: layer.input_shape.channels,
            height: layer.input_shape.height,
            width: layer.input_shape.width,
            filters: spec.filters,
            heads: spec.heads,
            window: spec.size,
            shift: spec.shift,
            ffn_ratio: spec.ffn_ratio,
            activation: spec.activation.clone(),
        };
        let (data, cache) = super::block::swin_forward(&host_input.data, weights, &geometry)
            .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
            .map_err(|err| backend_error("upload swin output", err))?;
        Ok((output, cache))
    }
}

//! GPU-runtime glue for `[tucker_attention]`: host forward behind the device tensor boundary.
use crate::error::{Error, Result};
use crate::model::{Layer, TuckerAttentionSpec};
use crate::runtime::{GpuExecutableNetwork, backend_error, tensor_backend_shape};
use lightnet_cuda::F32TensorOpsBackend;

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Download the input, run the host Tucker attention forward, upload the output.
    pub(crate) fn forward_tucker_attention_host(
        &self,
        layer: &Layer,
        spec: &TuckerAttentionSpec,
        weights: &super::TuckerAttentionParams,
        current_input: &B::Tensor,
    ) -> Result<(B::Tensor, super::TuckerAttentionForwardCache)> {
        let host_input = self
            .download_tensor(current_input, layer.input_shape)
            .map_err(|err| backend_error("download Tucker attention input", err))?;
        let geometry = super::TuckerGeometry {
            batch: layer.input_shape.batch,
            channels: layer.input_shape.channels,
            height: layer.input_shape.height,
            width: layer.input_shape.width,
            size: spec.size,
            heads: spec.heads,
            rank_q: spec.rank_q,
            rank_k: spec.rank_k,
            rank_v: spec.rank_v,
            rank_o: spec.rank_o,
            activation: spec.activation.clone(),
            msa_enabled: spec.msa_enabled,
            msa_block_size: spec.msa_block_size,
            msa_topk: spec.msa_topk,
        };
        let (data, cache) =
            super::tucker_attention_forward(&host_input.data, weights, &geometry)
                .map_err(Error::InvalidConfig)?;
        let output = self
            .backend
            .upload_f32(&tensor_backend_shape(layer.output_shape), &data)
            .map_err(|err| backend_error("upload Tucker attention output", err))?;
        Ok((output, cache))
    }
}

//! GPU-runtime glue for `[dsconv]`.
use crate::error::{Error, Result};
use crate::model::{DsConvSpec, Layer};
use crate::runtime::{DsConvParams, GpuExecutableNetwork, backend_error};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Native offset predictor, deformable gather and collapse for `[dsconv]`.
    pub(crate) fn forward_dsconv_gpu(
        &self,
        layer: &Layer,
        spec: &DsConvSpec,
        params: &DsConvParams,
        current_input: &B::Tensor,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let activation = spec.activation.to_cuda_f32_kind().ok_or_else(|| {
            Error::UnsupportedLayer {
                layer_index: layer.index,
                name: format!(
                    "[dsconv] unsupported GPU activation {:?}",
                    spec.activation
                ),
            }
        })?;
        let geometry = lightnet_cuda::host_ops::DsConvGeometry {
            batch: layer.input_shape.batch,
            channels: layer.input_shape.channels,
            filters: spec.filters,
            input_height: layer.input_shape.height,
            input_width: layer.input_shape.width,
            kernel_size: spec.size,
            extend_scope: spec.extend_scope,
            morph: spec.morph,
        };
        let offset_channels = 2 * spec.size;
        let offset_weights = self
            .backend
            .upload_f32(
                &[offset_channels, layer.input_shape.channels, 3, 3],
                &params.offset_weights,
            )
            .map_err(|err| backend_error("upload DSConv offset weights", err))?;
        let offset_biases = self
            .backend
            .upload_f32(&[offset_channels], &params.offset_biases)
            .map_err(|err| backend_error("upload DSConv offset biases", err))?;
        let weights = self
            .backend
            .upload_f32(
                &[spec.filters, layer.input_shape.channels, spec.size],
                &params.collapse.weights,
            )
            .map_err(|err| backend_error("upload DSConv collapse weights", err))?;
        let biases = self
            .backend
            .upload_f32(&[spec.filters], &params.collapse.biases)
            .map_err(|err| backend_error("upload DSConv collapse biases", err))?;
        let output = self
            .backend
            .dsconv_forward_native_f32(
                current_input,
                &offset_weights,
                &offset_biases,
                &weights,
                &biases,
                geometry,
                activation,
                launch_config,
            )
            .ok_or_else(|| Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "backend does not provide native f32 [dsconv]".to_owned(),
            })?
            .map_err(|err| backend_error("native DSConv forward", err))?;
        Ok(output)
    }
}

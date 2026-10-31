//! GPU-runtime glue for `[dsconv]`.
use crate::error::{Error, Result};
use crate::model::{DsConvSpec, Layer};
use crate::runtime::{
    LayerTrainBuffers,
    GpuForwardState,DsConvParams, GpuExecutableNetwork, backend_error};
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
        if spec.templates > 1 {
            return Err(Error::UnsupportedLayer { layer_index: layer.index, name: "[dsconv] templates>1 GPU execution is not yet supported".to_owned() });
        }
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

    /// Native DSConv backward.
    pub(crate) fn backward_dsconv_gpu(
        &self,
        layer: &Layer,
        spec: &DsConvSpec,
        params: &DsConvParams,
        forward: &GpuForwardState<B::Tensor>,
        grad: B::Tensor,
        config: ElementwiseLaunchConfig,
        layer_output_gradients: &mut [Option<B::Tensor>],
    ) -> Result<()> {
        if spec.templates > 1 {
            return Err(Error::UnsupportedLayer { layer_index: layer.index, name: "[dsconv] templates>1 GPU backward is not yet supported".to_owned() });
        }
        let activation = spec.activation.to_cuda_f32_kind().ok_or_else(|| {
            Error::UnsupportedLayer {
                layer_index: layer.index,
                name: format!("[dsconv] unsupported GPU activation {:?}", spec.activation),
            }
        })?;
        let input = if layer.index == 0 {
            &forward.input_gpu
        } else {
            &forward.gpu_layer_outputs[layer.index - 1]
        };
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
        let (d_input, d_weights, d_biases, d_offset_weights, d_offset_biases) = self
            .backend
            .dsconv_backward_native_f32(
                input,
                &offset_weights,
                &offset_biases,
                &weights,
                &biases,
                &grad,
                geometry,
                activation,
                config,
            )
            .ok_or_else(|| Error::UnsupportedLayer {
                layer_index: layer.index,
                name: "backend does not provide native f32 [dsconv] backward".to_owned(),
            })?
            .map_err(|err| backend_error("native DSConv backward", err))?;
        let d_weights = self
            .backend
            .download_f32(&d_weights)
            .map_err(|err| backend_error("download DSConv collapse gradients", err))?;
        let d_biases = self
            .backend
            .download_f32(&d_biases)
            .map_err(|err| backend_error("download DSConv collapse bias gradients", err))?;
        let d_offset_weights = self
            .backend
            .download_f32(&d_offset_weights)
            .map_err(|err| backend_error("download DSConv offset gradients", err))?;
        let d_offset_biases = self
            .backend
            .download_f32(&d_offset_biases)
            .map_err(|err| backend_error("download DSConv offset bias gradients", err))?;
        {
            let mut buffers = self.cpu_fallback_train_buffers.borrow_mut();
            let LayerTrainBuffers::DsConv(train) = &mut buffers[layer.index] else {
                return Err(Error::InvalidConfig(format!(
                    "[dsconv] layer #{} has no train buffers",
                    layer.index
                )));
            };
            for (dst, src) in train.collapse.weight_updates.iter_mut().zip(d_weights) {
                *dst += src;
            }
            for (dst, src) in train.collapse.bias_updates.iter_mut().zip(d_biases) {
                *dst += src;
            }
            for (dst, src) in train.offset_weight_updates.iter_mut().zip(d_offset_weights) {
                *dst += src;
            }
            for (dst, src) in train.offset_bias_updates.iter_mut().zip(d_offset_biases) {
                *dst += src;
            }
        }
        self.accumulate_previous_gpu_gradient(
            layer.index,
            d_input,
            layer_output_gradients,
        )?;
        Ok(())
    }
}

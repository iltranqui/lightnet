//! GPU-runtime glue for `[batch_normalize]` as a standalone layer, plus the batch-norm
//! forward helpers shared with the convolution, GraphConv and DeformConv arms.
use crate::error::{Error, Result};
use crate::model::{Layer, TensorShape};
use crate::runtime::{
    BATCH_NORM_ROLLING_KEEP, BATCH_NORM_ROLLING_KEEP_CPU, BatchNormParams, GpuBatchNormCache,
    GpuConvolutionLayerParams, GpuExecutableNetwork, backend_error,
    folded_batchnorm_inference_params, tensor_backend_shape, validate_batchnorm_params,
};
use lightnet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend};

impl<B> GpuExecutableNetwork<B>
where
    B: F32TensorOpsBackend,
    B::Error: std::fmt::Display,
{
    /// Standalone `[batchnorm]` arm: batch statistics when training, folded affine otherwise.
    pub(crate) fn forward_batch_norm_gpu(
        &self,
        layer: &Layer,
        params: &BatchNormParams,
        current_input: &B::Tensor,
        training: bool,
        launch_config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, Option<GpuBatchNormCache<B::Tensor>>)> {
        if training {
            let (output, cache) = self.forward_batchnorm_training_gpu_from_host_params(
                current_input,
                &params.biases,
                params,
                layer.output_shape,
                launch_config,
            )?;
            Ok((output, Some(cache)))
        } else {
            let output = self.forward_batchnorm_inference_gpu(
                current_input,
                layer.input_shape,
                params,
                launch_config,
            )?;
            Ok((output, None))
        }
    }

    /// Batch-statistics batch norm of a raw (bias-free) convolution output.
    /// The shift is the layer's raw biases (the BN biases); the rolling EMA is
    /// computed on the device and applied to the host params after backward.
    pub(crate) fn forward_batchnorm_training_gpu(
        &self,
        raw_output: &B::Tensor,
        cached: &GpuConvolutionLayerParams<B::Tensor>,
        batch_norm: &BatchNormParams,
        shape: TensorShape,
        update_rolling: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, GpuBatchNormCache<B::Tensor>)> {
        validate_batchnorm_params(batch_norm, shape.channels)?;
        let state = cached.bn.as_ref().ok_or_else(|| {
            Error::InvalidConfig("missing device batch-norm state for a BN layer".to_owned())
        })?;
        let (scales, rolling_mean, rolling_variance) = (
            state.scales.clone(),
            state.rolling_mean.clone(),
            state.rolling_variance.clone(),
        );
        let result = self
            .backend
            .batchnorm_train_forward_f32(
                raw_output,
                &scales,
                &cached.raw_biases,
                &rolling_mean,
                &rolling_variance,
                &tensor_backend_shape(shape),
                BATCH_NORM_ROLLING_KEEP,
                config,
            )
            .map_err(|err| backend_error("GPU batch-norm training forward", err))?;
        let cache = GpuBatchNormCache {
            scales,
            x_hat: result.x_hat,
            variance: result.variance,
            rolling: update_rolling.then_some((result.rolling_mean, result.rolling_variance)),
        };
        Ok((result.output, cache))
    }

    pub(crate) fn forward_batchnorm_training_gpu_from_host_params(
        &self,
        raw_output: &B::Tensor,
        raw_biases: &[f32],
        batch_norm: &BatchNormParams,
        shape: TensorShape,
        config: ElementwiseLaunchConfig,
    ) -> Result<(B::Tensor, GpuBatchNormCache<B::Tensor>)> {
        validate_batchnorm_params(batch_norm, shape.channels)?;
        let scales = self
            .backend
            .upload_f32(&[shape.channels], &batch_norm.scales)
            .map_err(|err| backend_error("upload GraphConv BN scales", err))?;
        let biases = self
            .backend
            .upload_f32(&[shape.channels], raw_biases)
            .map_err(|err| backend_error("upload GraphConv BN biases", err))?;
        let rolling_mean = self
            .backend
            .upload_f32(&[shape.channels], &batch_norm.rolling_mean)
            .map_err(|err| backend_error("upload GraphConv BN rolling mean", err))?;
        let rolling_variance = self
            .backend
            .upload_f32(&[shape.channels], &batch_norm.rolling_variance)
            .map_err(|err| backend_error("upload GraphConv BN rolling variance", err))?;
        // Match GraphConv's CPU reference rolling-stat update coefficient.
        let result = self
            .backend
            .batchnorm_train_forward_f32(
                raw_output,
                &scales,
                &biases,
                &rolling_mean,
                &rolling_variance,
                &tensor_backend_shape(shape),
                BATCH_NORM_ROLLING_KEEP_CPU,
                config,
            )
            .map_err(|err| backend_error("GraphConv GPU batch-norm training forward", err))?;
        Ok((
            result.output,
            GpuBatchNormCache {
                scales,
                x_hat: result.x_hat,
                variance: result.variance,
                rolling: Some((result.rolling_mean, result.rolling_variance)),
            },
        ))
    }

    pub(crate) fn forward_batchnorm_inference_gpu(
        &self,
        input: &B::Tensor,
        shape: TensorShape,
        params: &BatchNormParams,
        config: ElementwiseLaunchConfig,
    ) -> Result<B::Tensor> {
        let (scales, biases) = folded_batchnorm_inference_params(params, shape.channels)?;
        let scales = self
            .backend
            .upload_f32(&[shape.channels], &scales)
            .map_err(|err| backend_error("upload batchnorm inference scales", err))?;
        let biases = self
            .backend
            .upload_f32(&[shape.channels], &biases)
            .map_err(|err| backend_error("upload batchnorm inference biases", err))?;
        let scaled = self
            .backend
            .channel_scale_f32(
                input,
                &scales,
                shape.channels,
                shape.width * shape.height,
                config,
            )
            .map_err(|err| backend_error("batchnorm inference scale", err))?;
        self.backend
            .bias_add_f32(
                &scaled,
                &biases,
                shape.channels,
                shape.width * shape.height,
                config,
            )
            .map_err(|err| backend_error("batchnorm inference bias", err))
    }
}

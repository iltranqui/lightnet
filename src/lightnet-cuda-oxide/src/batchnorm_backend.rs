//! Batch-normalization training backend operations.

use lightnet_cuda::{BatchNormTrainForward, ElementwiseLaunchConfig};

use crate::{launch, launch_err, same_len, OxideBackend, OxideError, OxideTensorF32};
use crate::kernels::bias_scale::BIAS_BACKWARD_THREADS;

impl OxideBackend {
    pub(crate) fn batchnorm_train_forward_backend_f32(
        &self,
        x: &OxideTensorF32,
        scales: &OxideTensorF32,
        biases: &OxideTensorF32,
        rolling_mean: &OxideTensorF32,
        rolling_variance: &OxideTensorF32,
        shape: &[usize],
        keep: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<BatchNormTrainForward<OxideTensorF32>, OxideError> {
        if x.is_half() {
            return self.batchnorm_train_forward_amp(
                x,
                scales,
                biases,
                rolling_mean,
                rolling_variance,
                shape,
                keep,
                config,
            );
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let scales_f = self.f32v(scales)?;
        let scales = &*scales_f;
        let biases_f = self.f32v(biases)?;
        let biases = &*biases_f;
        let rolling_mean_f = self.f32v(rolling_mean)?;
        let rolling_mean = &*rolling_mean_f;
        let rolling_variance_f = self.f32v(rolling_variance)?;
        let rolling_variance = &*rolling_variance_f;
        let op = "batchnorm_train_forward_f32";
        let (channels, stride, row_groups) = Self::batchnorm_layout(
            op,
            shape,
            x.len(),
            &[scales, biases, rolling_mean, rolling_variance],
        )?;
        let n = (row_groups * stride) as f32;
        let block = BIAS_BACKWARD_THREADS as u32;
        let per_channel = launch::elementwise(channels, config);

        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.batchnorm.bn_sum_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &x.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let mut mean = self.alloc_uninit(channels)?;
        // SAFETY: one thread per channel.
        unsafe {
            self.batchnorm
                .bn_mean_finalize(&self.stream, per_channel, &partial, &mut mean, 1.0 / n)
        }
        .map_err(launch_err)?;

        // SAFETY: as above; every `partial` slot is rewritten.
        unsafe {
            self.batchnorm.bn_sq_dev_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &x.buf,
                &mean,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let sum_sq = self.fold_partials(&partial, channels)?;
        let mut variance = self.alloc_uninit(channels)?;
        let mut new_mean = self.alloc_uninit(channels)?;
        let mut new_var = self.alloc_uninit(channels)?;
        // SAFETY: one thread per channel writes its own slot of each output.
        unsafe {
            self.batchnorm.bn_var_finalize(
                &self.stream,
                per_channel,
                &sum_sq,
                &mean,
                &rolling_mean.buf,
                &rolling_variance.buf,
                &mut variance,
                &mut new_mean,
                &mut new_var,
                1.0 / n,
                1.0 / (n - 1.0).max(1.0),
                keep,
            )
        }
        .map_err(launch_err)?;

        let mut x_hat = self.alloc_uninit(x.len())?;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element writes its own `x_hat` and `out` slot.
        unsafe {
            self.batchnorm.bn_normalize(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mean,
                &variance,
                &scales.buf,
                &biases.buf,
                &mut x_hat,
                &mut out,
                channels,
                stride,
            )
        }
        .map_err(launch_err)?;
        let c = [channels];
        Ok(BatchNormTrainForward {
            output: OxideTensorF32::new(out, shape),
            x_hat: OxideTensorF32::new(x_hat, shape),
            mean: OxideTensorF32::new(mean, &c),
            variance: OxideTensorF32::new(variance, &c),
            rolling_mean: OxideTensorF32::new(new_mean, &c),
            rolling_variance: OxideTensorF32::new(new_var, &c),
        })
    }

    pub(crate) fn batchnorm_train_backward_backend_f32(
        &self,
        output_grad: &OxideTensorF32,
        x_hat: &OxideTensorF32,
        variance: &OxideTensorF32,
        scales: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        if x_hat.is_half() {
            return self.batchnorm_train_backward_amp(
                output_grad,
                x_hat,
                variance,
                scales,
                shape,
                config,
            );
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        let x_hat_f = self.f32v(x_hat)?;
        let x_hat = &*x_hat_f;
        let variance_f = self.f32v(variance)?;
        let variance = &*variance_f;
        let scales_f = self.f32v(scales)?;
        let scales = &*scales_f;
        let op = "batchnorm_train_backward_f32";
        same_len(op, output_grad, x_hat)?;
        let (channels, stride, row_groups) =
            Self::batchnorm_layout(op, shape, output_grad.len(), &[variance, scales])?;
        let n = (row_groups * stride) as f32;
        let block = BIAS_BACKWARD_THREADS as u32;

        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.batchnorm.bn_sum_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &output_grad.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let dbias = self.fold_partials(&partial, channels)?;
        // SAFETY: as above; every `partial` slot is rewritten.
        unsafe {
            self.batchnorm.bn_dot_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &output_grad.buf,
                &x_hat.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        let dscale = self.fold_partials(&partial, channels)?;

        let mut dx = self.alloc_uninit(output_grad.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.batchnorm.bn_backward_dx(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &output_grad.buf,
                &x_hat.buf,
                &variance.buf,
                &scales.buf,
                &dbias,
                &dscale,
                &mut dx,
                channels,
                stride,
                n,
            )
        }
        .map_err(launch_err)?;
        let c = [channels];
        Ok((
            OxideTensorF32::new(dx, shape),
            OxideTensorF32::new(dscale, &c),
            OxideTensorF32::new(dbias, &c),
        ))
    }

}

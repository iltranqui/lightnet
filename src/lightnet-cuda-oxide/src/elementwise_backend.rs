//! Elementwise backend dispatch.

use lightnet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32ChannelNormalization, F32TensorOpsBackend};

use crate::kernels::bias_scale::BIAS_BACKWARD_THREADS;
use crate::{activation_code, channel_normalization_code, channel_normalization_size, check_channel_layout, launch, launch_err, same_len, shape_err, OxideBackend, OxideError, OxideTensorF32};

impl OxideBackend {
    pub(crate) fn bias_add_backend_f32(
        &self,
        x: &OxideTensorF32,
        bias: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if x.is_half() {
            return self.bias_add_amp(x, bias, channels, channel_stride, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let bias_f = self.f32v(bias)?;
        let bias = &*bias_f;
        check_channel_layout(
            "bias_add_f32",
            x.len(),
            Some(bias.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per output element; buffers cover all accesses.
        unsafe {
            self.bias_scale.bias_add(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &bias.buf,
                &mut out,
                channels,
                channel_stride,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    pub(crate) fn bias_activation_backend_f32(
        &self,
        x: &OxideTensorF32,
        bias: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if x.is_half() {
            let biased = self.bias_add_f32(x, bias, channels, channel_stride, config)?;
            return self.activation_f32(&biased, activation, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let bias_f = self.f32v(bias)?;
        let bias = &*bias_f;
        check_channel_layout(
            "bias_activation_f32",
            x.len(),
            Some(bias.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per output element; buffers cover all accesses.
        unsafe {
            self.bias_scale.bias_activation(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &bias.buf,
                &mut out,
                channels,
                channel_stride,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    pub(crate) fn channel_scale_backend_f32(
        &self,
        x: &OxideTensorF32,
        scale: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let scale_f = self.f32v(scale)?;
        let scale = &*scale_f;
        check_channel_layout(
            "channel_scale_f32",
            x.len(),
            Some(scale.len()),
            channels,
            channel_stride,
        )?;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per output element; buffers cover all accesses.
        unsafe {
            self.bias_scale.channel_scale(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &scale.buf,
                &mut out,
                channels,
                channel_stride,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    pub(crate) fn bias_backward_backend_f32(
        &self,
        output_grad: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if output_grad.is_half() {
            return self.bias_backward_amp(output_grad, channels, channel_stride);
        }
        let output_grad_f = self.f32v(output_grad)?;
        let output_grad = &*output_grad_f;
        check_channel_layout(
            "bias_backward_f32",
            output_grad.len(),
            None,
            channels,
            channel_stride,
        )?;
        let row_groups = output_grad.len() / channel_stride / channels;
        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        let mut out = self.alloc_uninit(channels)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.bias_scale.bias_backward_partial(
                &self.stream,
                launch::blocks_of(channels, BIAS_BACKWARD_THREADS as u32),
                &output_grad.buf,
                &mut partial,
                channels,
                channel_stride,
                row_groups,
            )
        }
        .map_err(launch_err)?;
        // SAFETY: one thread per channel; `partial` holds channels * 256 sums.
        unsafe {
            self.bias_scale.bias_backward_reduce(
                &self.stream,
                launch::elementwise(channels, ElementwiseLaunchConfig::default()),
                &partial,
                &mut out,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, &[channels]))
    }

    pub(crate) fn activation_backend_f32(
        &self,
        x: &OxideTensorF32,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if x.is_half() {
            return self.activation_amp(x, activation, config);
        }
        let x_f = self.f32v(x)?;
        let x = &*x_f;
        let mut out = self.alloc_uninit(x.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.activations.activation(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut out,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    pub(crate) fn activation_backward_backend_f32(
        &self,
        activation_input: &OxideTensorF32,
        delta: &OxideTensorF32,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        if activation_input.is_half() {
            return self.activation_backward_amp(activation_input, delta, activation, config);
        }
        let activation_input_f = self.f32v(activation_input)?;
        let activation_input = &*activation_input_f;
        let delta_f = self.f32v(delta)?;
        let delta = &*delta_f;
        same_len("activation_backward_f32", activation_input, delta)?;
        let mut out = self.alloc_uninit(delta.len())?;
        // SAFETY: one thread per element; buffers cover all accesses.
        unsafe {
            self.activations.activation_backward(
                &self.stream,
                launch::elementwise(delta.len(), config),
                &activation_input.buf,
                &delta.buf,
                &mut out,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(out, activation_input.shape()))
    }

    pub(crate) fn normalize_channels_backend_f32(
        &self,
        input: &OxideTensorF32,
        output_shape: &[usize],
        batch: usize,
        channels: usize,
        spatial: usize,
        normalization: F32ChannelNormalization,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "normalize_channels_f32";
        let total = channel_normalization_size(op, output_shape, batch, channels, spatial)?;
        let input = self.f32v(input)?;
        let input = &*input;
        if input.len() != total || input.shape() != output_shape {
            return Err(shape_err(
                op,
                format!(
                    "input shape/length are {:?}/{}, expected {output_shape:?}/{total}",
                    input.shape(),
                    input.len()
                ),
            ));
        }
        let mut output = self.alloc_uninit(total)?;
        // SAFETY: one thread writes each output element; shape and dimensions
        // have been validated and each thread's channel loop stays in bounds.
        unsafe {
            self.normalize_channels.normalize_channels(
                &self.stream,
                launch::elementwise(total, config),
                &input.buf,
                &mut output,
                channels,
                spatial,
                channel_normalization_code(normalization),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(output, output_shape))
    }

    pub(crate) fn normalize_channels_backward_backend_f32(
        &self,
        activation_input: &OxideTensorF32,
        output: &OxideTensorF32,
        delta: &OxideTensorF32,
        output_shape: &[usize],
        batch: usize,
        channels: usize,
        spatial: usize,
        normalization: F32ChannelNormalization,
        cpp_grad: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "normalize_channels_backward_f32";
        let total = channel_normalization_size(op, output_shape, batch, channels, spatial)?;
        let activation_input = self.f32v(activation_input)?;
        let activation_input = &*activation_input;
        let output = self.f32v(output)?;
        let output = &*output;
        let delta = self.f32v(delta)?;
        let delta = &*delta;
        if activation_input.len() != total
            || output.len() != total
            || delta.len() != total
            || activation_input.shape() != output_shape
            || output.shape() != output_shape
            || delta.shape() != output_shape
        {
            return Err(shape_err(
                op,
                format!(
                    "input/output/delta shapes and lengths are {:?}/{}, {:?}/{}, {:?}/{}, expected {output_shape:?}/{total}",
                    activation_input.shape(),
                    activation_input.len(),
                    output.shape(),
                    output.len(),
                    delta.shape(),
                    delta.len()
                ),
            ));
        }
        let mut gradient = self.alloc_uninit(total)?;
        // SAFETY: one thread writes each gradient element; all tensor lengths
        // and nonzero geometry were checked above.
        unsafe {
            self.normalize_channels.normalize_channels_backward(
                &self.stream,
                launch::elementwise(total, config),
                &activation_input.buf,
                &output.buf,
                &delta.buf,
                &mut gradient,
                channels,
                spatial,
                channel_normalization_code(normalization),
                cpp_grad as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF32::new(gradient, output_shape))
    }
}

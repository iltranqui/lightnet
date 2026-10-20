use darknet_cuda::{ElementwiseLaunchConfig, F32ActivationKind, F32TensorOpsBackend};

use crate::backend::{CutileBackend, CutileTensorF32};
use crate::error::CutileBackendError;

impl F32TensorOpsBackend for CutileBackend {
    type Error = CutileBackendError;
    type Tensor = CutileTensorF32;

    fn upload_f32(&self, shape: &[usize], host: &[f32]) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::upload_f32(self, shape, host)
    }

    fn zeros_f32(&self, shape: &[usize]) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::zeros_f32(self, shape)
    }

    fn download_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error> {
        CutileBackend::download_f32(self, tensor)
    }

    fn bias_add_f32(
        &self,
        x: &Self::Tensor,
        bias: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::bias_add_f32(self, x, bias, channels, channel_stride, config.tile_len)
    }

    fn channel_scale_f32(
        &self,
        x: &Self::Tensor,
        scale: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::channel_scale_f32(self, x, scale, channels, channel_stride, config.tile_len)
    }

    fn bias_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::bias_backward_f32(
            self,
            output_grad,
            channels,
            channel_stride,
            config.tile_len,
        )
    }

    fn activation_f32(
        &self,
        x: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::activation_f32(self, x, activation, config.tile_len)
    }

    fn activation_backward_f32(
        &self,
        activation_input: &Self::Tensor,
        delta: &Self::Tensor,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::activation_backward_f32(
            self,
            activation_input,
            delta,
            activation,
            config.tile_len,
        )
    }

    fn shortcut_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::shortcut_add_f32(self, x, y, config.tile_len)
    }

    fn route_concat_f32(
        &self,
        inputs: &[&Self::Tensor],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::route_concat_f32(
            self,
            inputs,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            output_shape,
            channel_stride,
            config.tile_len,
        )
    }

    fn route_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<Self::Tensor>, Self::Error> {
        CutileBackend::route_backward_f32(
            self,
            output_grad,
            input_shapes,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            channel_stride,
            config.tile_len,
        )
    }

    fn conv2d_f32(
        &self,
        x: &Self::Tensor,
        weights: &Self::Tensor,
        output_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::conv2d_f32(
            self,
            x,
            weights,
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config.tile_len,
        )
    }

    fn conv2d_backward_input_f32(
        &self,
        output_grad: &Self::Tensor,
        weights: &Self::Tensor,
        input_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::conv2d_backward_input_f32(
            self,
            output_grad,
            weights,
            input_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config.tile_len,
        )
    }

    fn conv2d_backward_weights_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        weight_shape: &[usize],
        input_channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        kernel_size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::conv2d_backward_weights_f32(
            self,
            x,
            output_grad,
            weight_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            config.tile_len,
        )
    }

    fn upsample_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::upsample_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            stride,
            scale,
            reverse,
            config.tile_len,
        )
    }

    fn upsample_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::upsample_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            output_height,
            output_width,
            stride,
            scale,
            config.tile_len,
        )
    }

    fn maxpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::maxpool_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config.tile_len,
        )
    }

    fn maxpool_backward_f32(
        &self,
        x: &Self::Tensor,
        output_grad: &Self::Tensor,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::maxpool_backward_f32(
            self,
            x,
            output_grad,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config.tile_len,
        )
    }

    fn local_avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::local_avgpool_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
        )
    }

    fn local_avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::local_avgpool_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
        )
    }

    fn avgpool_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::avgpool_f32(self, x, output_shape, channels, input_height, input_width)
    }

    fn avgpool_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        _config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::avgpool_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
        )
    }

    fn crop_bilinear_f32(
        &self,
        x: &Self::Tensor,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &Self::Tensor,
        source_y: &Self::Tensor,
        input_scales: &Self::Tensor,
        input_offsets: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::crop_bilinear_f32(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
            input_offsets,
            config.tile_len,
        )
    }

    fn crop_bilinear_backward_f32(
        &self,
        output_grad: &Self::Tensor,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &Self::Tensor,
        source_y: &Self::Tensor,
        input_scales: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::crop_bilinear_backward_f32(
            self,
            output_grad,
            input_shape,
            channels,
            input_height,
            input_width,
            source_x,
            source_y,
            input_scales,
            config.tile_len,
        )
    }

    fn vector_add_f32(
        &self,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::vector_add_f32(self, x, y, config.tile_len)
    }

    fn saxpy_f32(
        &self,
        alpha: f32,
        x: &Self::Tensor,
        y: &Self::Tensor,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::Tensor, Self::Error> {
        CutileBackend::saxpy_f32(self, alpha, x, y, config.tile_len)
    }

    fn sgd_update_f32(
        &self,
        weights: &Self::Tensor,
        updates: &Self::Tensor,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(Self::Tensor, Self::Tensor), Self::Error> {
        CutileBackend::sgd_update_f32(
            self,
            weights,
            updates,
            lr,
            momentum,
            decay,
            batch,
            config.tile_len,
        )
    }
}

use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

use crate::backend::{CutileBackend, CutileTensorF32, POINTER_TILE_1D, grid_1d, normalize_partition_tile_len};
use crate::error::CutileBackendError;
use crate::kernels::cutile_lightnet_kernels;
use crate::validate::*;

impl CutileBackend {
    pub fn conv2d_f32(
        &self,
        x: &CutileTensorF32,
        weights: &CutileTensorF32,
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
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_conv2d_layout(
            x.len(),
            weights.len(),
            output_shape,
            input_channels,
            input_height,
            input_width,
            filters,
            kernel_size,
            groups,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = self.zeroed_tensor_f32(layout.output_len)?;
        unsafe {
            cutile_lightnet_kernels::conv2d_flat(
                out.device_pointer(),
                x.arc().device_pointer(),
                weights.arc().device_pointer(),
                layout.output_len as i32,
                input_channels as i32,
                input_height as i32,
                input_width as i32,
                filters as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                kernel_size as i32,
                stride_x as i32,
                stride_y as i32,
                dilation as i32,
                padding as i32,
                groups.max(1) as i32,
                layout.channels_per_group as i32,
                layout.filters_per_group as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn conv2d_backward_input_f32(
        &self,
        output_grad: &CutileTensorF32,
        weights: &CutileTensorF32,
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
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_conv2d_backward_input_layout(
            output_grad.len(),
            weights.len(),
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
        )?;
        let _tile_len = tile_len.min(layout.input_len).max(1);
        let input_grad = self.zeroed_tensor_f32(layout.input_len)?;
        unsafe {
            cutile_lightnet_kernels::conv2d_backward_input_flat(
                input_grad.device_pointer(),
                output_grad.arc().device_pointer(),
                weights.arc().device_pointer(),
                layout.input_len as i32,
                input_channels as i32,
                input_height as i32,
                input_width as i32,
                filters as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                kernel_size as i32,
                stride_x as i32,
                stride_y as i32,
                dilation as i32,
                padding as i32,
                groups as i32,
                layout.channels_per_group as i32,
                layout.filters_per_group as i32,
            )
        }
        .grid((grid_1d(layout.input_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(input_grad.reshape(input_shape)?),
        })
    }

    pub fn conv2d_backward_weights_f32(
        &self,
        x: &CutileTensorF32,
        output_grad: &CutileTensorF32,
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
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_conv2d_backward_weights_layout(
            x.len(),
            output_grad.len(),
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
        )?;
        let _tile_len = tile_len.min(layout.weight_len).max(1);
        let weight_grad = self.zeroed_tensor_f32(layout.weight_len)?;
        unsafe {
            cutile_lightnet_kernels::conv2d_backward_weights_flat(
                weight_grad.device_pointer(),
                x.arc().device_pointer(),
                output_grad.arc().device_pointer(),
                layout.weight_len as i32,
                input_channels as i32,
                input_height as i32,
                input_width as i32,
                filters as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                kernel_size as i32,
                stride_x as i32,
                stride_y as i32,
                dilation as i32,
                padding as i32,
                groups as i32,
                layout.batch as i32,
                layout.channels_per_group as i32,
                layout.filters_per_group as i32,
            )
        }
        .grid((grid_1d(layout.weight_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(weight_grad.reshape(weight_shape)?),
        })
    }
}

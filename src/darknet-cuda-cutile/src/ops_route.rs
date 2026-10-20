use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

use crate::backend::{CutileBackend, CutileTensorF32, POINTER_TILE_1D, grid_1d, normalize_partition_tile_len};
use crate::error::CutileBackendError;
use crate::kernels::cutile_darknet_kernels;
use crate::validate::*;

impl CutileBackend {
    pub fn route_concat_f32(
        &self,
        inputs: &[&CutileTensorF32],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let input_lens = inputs.iter().map(|input| input.len()).collect::<Vec<_>>();
        let layout = validate_route_concat_layout(
            &input_lens,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            output_shape,
            channel_stride,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = self.zeroed_tensor_f32(layout.output_len)?;
        let out_ptr = out.device_pointer();
        let mut channel_offset = 0usize;

        for (((input, source_channel_offset), src_channels), source_total_channels) in inputs
            .iter()
            .zip(input_channel_offsets.iter().copied())
            .zip(input_channels.iter().copied())
            .zip(input_total_channels.iter().copied())
        {
            let copy_len = layout.batch * src_channels * channel_stride;
            unsafe {
                cutile_darknet_kernels::route_copy_flat(
                    out_ptr,
                    input.arc().device_pointer(),
                    copy_len as i32,
                    source_total_channels as i32,
                    source_channel_offset as i32,
                    src_channels as i32,
                    layout.total_channels as i32,
                    channel_offset as i32,
                    channel_stride as i32,
                )
            }
            .grid((grid_1d(copy_len, POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;
            channel_offset += src_channels;
        }

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn route_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<Vec<CutileTensorF32>, CutileBackendError> {
        let input_lens = input_shapes
            .iter()
            .map(|shape| shape_num_elements(shape))
            .collect::<Result<Vec<_>, _>>()?;
        let layout = validate_route_concat_layout(
            &input_lens,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            &output_grad.shape_usize(),
            channel_stride,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);

        let mut output_channel_offset = 0usize;
        let mut input_grads = Vec::with_capacity(input_shapes.len());
        for (((input_shape, source_channel_offset), src_channels), source_total_channels) in
            input_shapes
                .iter()
                .zip(input_channel_offsets.iter().copied())
                .zip(input_channels.iter().copied())
                .zip(input_total_channels.iter().copied())
        {
            let input_len = shape_num_elements(input_shape)?;
            let input_grad = self.zeroed_tensor_f32(input_len)?;
            let copy_len = layout.batch * src_channels * channel_stride;
            unsafe {
                cutile_darknet_kernels::route_scatter_flat(
                    input_grad.device_pointer(),
                    output_grad.arc().device_pointer(),
                    copy_len as i32,
                    layout.total_channels as i32,
                    output_channel_offset as i32,
                    src_channels as i32,
                    source_total_channels as i32,
                    source_channel_offset as i32,
                    channel_stride as i32,
                )
            }
            .grid((grid_1d(copy_len, POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;

            input_grads.push(CutileTensorF32 {
                inner: Arc::new(input_grad.reshape(input_shape)?),
            });
            output_channel_offset += src_channels;
        }

        Ok(input_grads)
    }
}

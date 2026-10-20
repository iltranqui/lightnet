use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

use crate::backend::{CutileBackend, CutileTensorF32, POINTER_TILE_1D, grid_1d, normalize_partition_tile_len};
use crate::error::CutileBackendError;
use crate::kernels::cutile_darknet_kernels;
use crate::validate::*;

impl CutileBackend {
    pub fn upsample_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_upsample_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            stride,
            reverse,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = self.zeroed_tensor_f32(layout.output_len)?;

        if reverse {
            unsafe {
                cutile_darknet_kernels::upsample_reverse_flat(
                    out.device_pointer(),
                    x.arc().device_pointer(),
                    x.len() as i32,
                    input_height as i32,
                    input_width as i32,
                    layout.output_height as i32,
                    layout.output_width as i32,
                    stride as i32,
                    scale,
                )
            }
            .grid((grid_1d(x.len(), POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;
        } else {
            unsafe {
                cutile_darknet_kernels::upsample_nearest_flat(
                    out.device_pointer(),
                    x.arc().device_pointer(),
                    layout.output_len as i32,
                    input_height as i32,
                    input_width as i32,
                    layout.output_height as i32,
                    layout.output_width as i32,
                    stride as i32,
                    scale,
                )
            }
            .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
            .sync_on(&self.stream)?;
        }

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn upsample_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        self.upsample_f32(
            output_grad,
            input_shape,
            channels,
            output_height,
            output_width,
            stride,
            scale,
            true,
            tile_len,
        )
    }
}

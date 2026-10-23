use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

use crate::backend::{CutileBackend, CutileTensorF32, POINTER_TILE_1D, grid_1d, normalize_partition_tile_len};
use crate::error::CutileBackendError;
use crate::kernels::cutile_lightnet_kernels;
use crate::validate::*;

impl CutileBackend {
    pub fn crop_bilinear_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &CutileTensorF32,
        source_y: &CutileTensorF32,
        input_scales: &CutileTensorF32,
        input_offsets: &CutileTensorF32,
        _tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_crop_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            source_x.len(),
            source_y.len(),
            input_scales.len(),
            input_offsets.len(),
        )?;
        let out = self.zeroed_tensor_f32(layout.output_len)?;
        unsafe {
            cutile_lightnet_kernels::crop_bilinear_flat(
                out.device_pointer(),
                x.arc().device_pointer(),
                source_x.arc().device_pointer(),
                source_y.arc().device_pointer(),
                input_scales.arc().device_pointer(),
                input_offsets.arc().device_pointer(),
                layout.output_len as i32,
                channels as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn crop_bilinear_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        source_x: &CutileTensorF32,
        source_y: &CutileTensorF32,
        input_scales: &CutileTensorF32,
        _tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let output_shape = output_grad.shape_usize();
        let expected_input = output_shape[0]
            .checked_mul(channels)
            .and_then(|len| len.checked_mul(input_height))
            .and_then(|len| len.checked_mul(input_width))
            .ok_or_else(|| CutileBackendError::Shape("crop input shape overflow".to_owned()))?;
        let layout = validate_crop_layout(
            expected_input,
            &output_shape,
            channels,
            input_height,
            input_width,
            source_x.len(),
            source_y.len(),
            input_scales.len(),
            input_scales.len(),
        )?;
        if output_grad.len() != layout.output_len {
            return Err(CutileBackendError::Shape(format!(
                "crop output-gradient length mismatch: expected {}, got {}",
                layout.output_len,
                output_grad.len()
            )));
        }
        let input_len = shape_num_elements(input_shape)?;
        let expected_input = layout.batch * channels * input_height * input_width;
        if input_len != expected_input {
            return Err(CutileBackendError::Shape(format!(
                "crop_bilinear_backward_f32 input shape {:?} expects {input_len} elements, expected {expected_input}",
                input_shape
            )));
        }
        let out = self.zeroed_tensor_f32(input_len)?;
        unsafe {
            cutile_lightnet_kernels::crop_bilinear_backward_flat(
                out.device_pointer(),
                output_grad.arc().device_pointer(),
                source_x.arc().device_pointer(),
                source_y.arc().device_pointer(),
                input_scales.arc().device_pointer(),
                layout.output_len as i32,
                channels as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(input_shape)?),
        })
    }
}

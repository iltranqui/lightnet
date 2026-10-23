use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

use crate::backend::{CutileBackend, CutileTensorF32, POINTER_TILE_1D, grid_1d, normalize_partition_tile_len};
use crate::error::CutileBackendError;
use crate::kernels::cutile_lightnet_kernels;
use crate::validate::*;

impl CutileBackend {
    pub fn maxpool_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_maxpool_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let out = self.zeroed_tensor_f32(layout.output_len)?;

        unsafe {
            cutile_lightnet_kernels::maxpool_flat(
                out.device_pointer(),
                x.arc().device_pointer(),
                layout.output_len as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                size as i32,
                stride_x as i32,
                stride_y as i32,
                (padding / 2) as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(output_shape)?),
        })
    }

    pub fn maxpool_backward_f32(
        &self,
        x: &CutileTensorF32,
        output_grad: &CutileTensorF32,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_maxpool_layout(
            x.len(),
            &output_grad.shape_usize(),
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let _tile_len = tile_len.min(layout.output_len).max(1);
        let input_grad = self.zeroed_tensor_f32(x.len())?;

        unsafe {
            cutile_lightnet_kernels::maxpool_backward_flat(
                input_grad.device_pointer(),
                x.arc().device_pointer(),
                output_grad.arc().device_pointer(),
                layout.output_len as i32,
                input_height as i32,
                input_width as i32,
                layout.output_height as i32,
                layout.output_width as i32,
                size as i32,
                stride_x as i32,
                stride_y as i32,
                (padding / 2) as i32,
            )
        }
        .grid((grid_1d(layout.output_len, POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok(CutileTensorF32 {
            inner: Arc::new(input_grad.reshape(&x.shape_usize())?),
        })
    }

    pub fn local_avgpool_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let layout = validate_maxpool_layout(
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let input = self.download_f32(x)?;
        let mut out = vec![0.0; layout.output_len];
        let pad_offset = (padding / 2) as isize;
        let batch = output_shape[0];

        for b in 0..batch {
            for c in 0..channels {
                for oy in 0..layout.output_height {
                    for ox in 0..layout.output_width {
                        let mut acc = 0.0;
                        let mut count = 0usize;
                        for ky in 0..size {
                            let in_y = oy as isize * stride_y as isize + ky as isize - pad_offset;
                            if !(0..input_height as isize).contains(&in_y) {
                                continue;
                            }
                            for kx in 0..size {
                                let in_x =
                                    ox as isize * stride_x as isize + kx as isize - pad_offset;
                                if !(0..input_width as isize).contains(&in_x) {
                                    continue;
                                }
                                let src = ((((b * channels) + c) * input_height) + in_y as usize)
                                    * input_width
                                    + in_x as usize;
                                acc += input[src];
                                count += 1;
                            }
                        }
                        if count > 0 {
                            let dst = ((((b * channels) + c) * layout.output_height) + oy)
                                * layout.output_width
                                + ox;
                            out[dst] = acc / count as f32;
                        }
                    }
                }
            }
        }

        self.upload_f32(output_shape, &out)
    }

    pub fn local_avgpool_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if input_shape.len() != 4
            || input_shape[1] != channels
            || input_shape[2] != input_height
            || input_shape[3] != input_width
        {
            return Err(CutileBackendError::Shape(format!(
                "local_avgpool_backward_f32 input shape mismatch: expected [batch, {channels}, {input_height}, {input_width}], got {input_shape:?}"
            )));
        }
        let output_shape = output_grad.shape_usize();
        let layout = validate_maxpool_layout(
            input_shape.iter().product(),
            &output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let output_grad = self.download_f32(output_grad)?;
        let mut input_grad = vec![0.0; input_shape.iter().product()];
        let pad_offset = (padding / 2) as isize;
        let batch = input_shape[0];

        for b in 0..batch {
            for c in 0..channels {
                for oy in 0..layout.output_height {
                    for ox in 0..layout.output_width {
                        let mut valid = Vec::new();
                        for ky in 0..size {
                            let in_y = oy as isize * stride_y as isize + ky as isize - pad_offset;
                            if !(0..input_height as isize).contains(&in_y) {
                                continue;
                            }
                            for kx in 0..size {
                                let in_x =
                                    ox as isize * stride_x as isize + kx as isize - pad_offset;
                                if !(0..input_width as isize).contains(&in_x) {
                                    continue;
                                }
                                valid.push((in_y as usize, in_x as usize));
                            }
                        }
                        if !valid.is_empty() {
                            let src = ((((b * channels) + c) * layout.output_height) + oy)
                                * layout.output_width
                                + ox;
                            let share = output_grad[src] / valid.len() as f32;
                            for (y, x) in valid {
                                let dst =
                                    ((((b * channels) + c) * input_height) + y) * input_width + x;
                                input_grad[dst] += share;
                            }
                        }
                    }
                }
            }
        }

        self.upload_f32(input_shape, &input_grad)
    }

    pub fn avgpool_f32(
        &self,
        x: &CutileTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if output_shape.len() != 4
            || output_shape[1] != channels
            || output_shape[2] != 1
            || output_shape[3] != 1
        {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_f32 output shape must be [batch, {channels}, 1, 1], got {output_shape:?}"
            )));
        }
        let batch_stride = channels
            .checked_mul(input_height)
            .and_then(|value| value.checked_mul(input_width))
            .ok_or_else(|| {
                CutileBackendError::Shape("avgpool_f32 input stride overflow".to_owned())
            })?;
        if batch_stride == 0 || !x.len().is_multiple_of(batch_stride) {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_f32 input length {} is incompatible with channels={channels}, height={input_height}, width={input_width}",
                x.len()
            )));
        }
        let batch = x.len() / batch_stride;
        if output_shape[0] != batch {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_f32 output batch mismatch: expected {batch}, got {}",
                output_shape[0]
            )));
        }

        let input = self.download_f32(x)?;
        let area = (input_height * input_width) as f32;
        let mut out = vec![0.0; batch * channels];
        for b in 0..batch {
            for c in 0..channels {
                let mut acc = 0.0;
                for y in 0..input_height {
                    for x_idx in 0..input_width {
                        let src = ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                        acc += input[src];
                    }
                }
                out[b * channels + c] = acc / area;
            }
        }

        self.upload_f32(output_shape, &out)
    }

    pub fn avgpool_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        input_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if input_shape.len() != 4
            || input_shape[1] != channels
            || input_shape[2] != input_height
            || input_shape[3] != input_width
        {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_backward_f32 input shape mismatch: expected [batch, {channels}, {input_height}, {input_width}], got {input_shape:?}"
            )));
        }
        let batch = input_shape[0];
        if output_grad.len() != batch * channels {
            return Err(CutileBackendError::Shape(format!(
                "avgpool_backward_f32 output gradient length mismatch: expected {}, got {}",
                batch * channels,
                output_grad.len()
            )));
        }

        let output_grad = self.download_f32(output_grad)?;
        let area = (input_height * input_width) as f32;
        let mut input_grad = vec![0.0; input_shape.iter().product()];
        for b in 0..batch {
            for c in 0..channels {
                let grad = output_grad[b * channels + c] / area;
                for y in 0..input_height {
                    for x_idx in 0..input_width {
                        let dst = ((((b * channels) + c) * input_height) + y) * input_width + x_idx;
                        input_grad[dst] = grad;
                    }
                }
            }
        }

        self.upload_f32(input_shape, &input_grad)
    }
}

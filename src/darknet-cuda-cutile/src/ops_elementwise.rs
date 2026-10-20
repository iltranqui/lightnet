use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

use crate::backend::{CutileBackend, CutileTensorF32, POINTER_TILE_1D, grid_1d, normalize_partition_tile_len};
use crate::error::CutileBackendError;
use crate::kernels::cutile_darknet_kernels;
use crate::validate::*;
use darknet_cuda::F32ActivationKind;

impl CutileBackend {
    pub fn bias_add_f32(
        &self,
        x: &CutileTensorF32,
        bias: &CutileTensorF32,
        channels: usize,
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let (rows, cols) = validate_bias_add_layout(x.len(), bias.len(), channels, channel_stride)?;
        let tile_len = normalize_partition_tile_len(cols, tile_len);
        let x_view = x.inner.view(&[rows, cols])?;
        let bias_view = bias.inner.view(&[channels])?;
        let z = self
            .zeroed_tensor_f32(rows * cols)?
            .reshape(&[rows, cols])?
            .partition([1, tile_len]);
        let (z, _, _, _) =
            cutile_darknet_kernels::bias_add_2d(z, &x_view, &bias_view, channels as i32)
                .grid((rows as u32, grid_1d(cols, tile_len), 1))
                .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&x.shape_usize())?),
        })
    }

    pub fn channel_scale_f32(
        &self,
        x: &CutileTensorF32,
        scale: &CutileTensorF32,
        channels: usize,
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let (rows, cols) =
            validate_bias_add_layout(x.len(), scale.len(), channels, channel_stride)?;
        let tile_len = normalize_partition_tile_len(cols, tile_len);
        let x_view = x.inner.view(&[rows, cols])?;
        let scale_view = scale.inner.view(&[channels])?;
        let z = self
            .zeroed_tensor_f32(rows * cols)?
            .reshape(&[rows, cols])?
            .partition([1, tile_len]);
        let (z, _, _, _) =
            cutile_darknet_kernels::channel_scale_2d(z, &x_view, &scale_view, channels as i32)
                .grid((rows as u32, grid_1d(cols, tile_len), 1))
                .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&x.shape_usize())?),
        })
    }

    pub fn bias_backward_f32(
        &self,
        output_grad: &CutileTensorF32,
        channels: usize,
        channel_stride: usize,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let _ = tile_len;
        validate_bias_backward_layout(output_grad.len(), channels, channel_stride)?;
        let out = self.zeroed_tensor_f32(channels)?;
        unsafe {
            cutile_darknet_kernels::bias_backward_flat(
                out.device_pointer(),
                output_grad.arc().device_pointer(),
                output_grad.len() as i32,
                channels as i32,
                channel_stride as i32,
            )
        }
        .grid((grid_1d(output_grad.len(), POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&[channels])?),
        })
    }

    pub fn activation_f32(
        &self,
        x: &CutileTensorF32,
        activation: F32ActivationKind,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        let len = x.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let x_flat = x.inner.view(&[x.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (z, _) = match activation {
            F32ActivationKind::Linear => {
                let (z, _) = cutile_darknet_kernels::activation_linear(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Logistic => {
                let (z, _) = cutile_darknet_kernels::activation_logistic(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Relu => {
                let (z, _) = cutile_darknet_kernels::activation_relu(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Leaky => {
                let (z, _) = cutile_darknet_kernels::activation_leaky(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Mish => {
                let (z, _) = cutile_darknet_kernels::activation_mish(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Swish => {
                let (z, _) = cutile_darknet_kernels::activation_swish(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
            F32ActivationKind::Tanh => {
                let (z, _) = cutile_darknet_kernels::activation_tanh(z, &x_flat)
                    .grid((grid_1d(len, tile_len), 1, 1))
                    .sync_on(&self.stream)?;
                (z, ())
            }
        };
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&x.shape_usize())?),
        })
    }

    pub fn activation_backward_f32(
        &self,
        activation_input: &CutileTensorF32,
        delta: &CutileTensorF32,
        activation: F32ActivationKind,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        ensure_same_shape(
            "activation_backward_f32",
            activation_input.shape(),
            delta.shape(),
        )?;

        let len = activation_input.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let activation_input_flat = activation_input.inner.view(&[activation_input.len()])?;
        let delta_flat = delta.inner.view(&[delta.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (z, _, _) = match activation {
            F32ActivationKind::Linear => cutile_darknet_kernels::activation_backward_linear(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Logistic => cutile_darknet_kernels::activation_backward_logistic(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Relu => cutile_darknet_kernels::activation_backward_relu(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Leaky => cutile_darknet_kernels::activation_backward_leaky(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Mish => cutile_darknet_kernels::activation_backward_mish(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Swish => cutile_darknet_kernels::activation_backward_swish(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
            F32ActivationKind::Tanh => cutile_darknet_kernels::activation_backward_tanh(
                z,
                &activation_input_flat,
                &delta_flat,
            )
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?,
        };
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(out.reshape(&activation_input.shape_usize())?),
        })
    }

    pub fn shortcut_add_f32(
        &self,
        x: &CutileTensorF32,
        y: &CutileTensorF32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        self.vector_add_f32(x, y, tile_len)
    }
}

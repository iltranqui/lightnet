use std::sync::Arc;

use cutile::api;
use cutile::prelude::*;

use crate::backend::{CutileBackend, CutileTensorF32, POINTER_TILE_1D, grid_1d, normalize_partition_tile_len};
use crate::error::CutileBackendError;
use crate::kernels::cutile_lightnet_kernels;
use crate::validate::*;

impl CutileBackend {
    pub fn vector_add_f32(
        &self,
        x: &CutileTensorF32,
        y: &CutileTensorF32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if x.len() != y.len() {
            return Err(CutileBackendError::Shape(format!(
                "vector_add_f32 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        }

        let len = x.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let x_flat = x.inner.view(&[x.len()])?;
        let y_flat = y.inner.view(&[y.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (z, _, _) = cutile_lightnet_kernels::vector_add(z, &x_flat, &y_flat)
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(
                out.reshape(
                    &y.shape()
                        .iter()
                        .map(|dim| *dim as usize)
                        .collect::<Vec<_>>(),
                )?,
            ),
        })
    }

    pub fn saxpy_f32(
        &self,
        alpha: f32,
        x: &CutileTensorF32,
        y: &CutileTensorF32,
        tile_len: usize,
    ) -> Result<CutileTensorF32, CutileBackendError> {
        if x.len() != y.len() {
            return Err(CutileBackendError::Shape(format!(
                "saxpy_f32 shape mismatch: {:?} vs {:?}",
                x.shape(),
                y.shape()
            )));
        }

        let len = x.len();
        let tile_len = normalize_partition_tile_len(len, tile_len);
        let x_flat = x.inner.view(&[x.len()])?;
        let y_flat = y.inner.view(&[y.len()])?;
        let z = self.zeroed_tensor_f32(len)?.partition([tile_len]);
        let (_alpha, z, _, _) = cutile_lightnet_kernels::saxpy(alpha, z, &x_flat, &y_flat)
            .grid((grid_1d(len, tile_len), 1, 1))
            .sync_on(&self.stream)?;
        let out: Tensor<f32> = z.unpartition();
        Ok(CutileTensorF32 {
            inner: Arc::new(
                out.reshape(
                    &y.shape()
                        .iter()
                        .map(|dim| *dim as usize)
                        .collect::<Vec<_>>(),
                )?,
            ),
        })
    }

    pub fn sgd_update_f32(
        &self,
        weights: &CutileTensorF32,
        updates: &CutileTensorF32,
        lr: f32,
        momentum: f32,
        decay: f32,
        batch: usize,
        _tile_len: usize,
    ) -> Result<(CutileTensorF32, CutileTensorF32), CutileBackendError> {
        if weights.len() != updates.len() {
            return Err(CutileBackendError::Shape(format!(
                "sgd_update_f32 shape mismatch: {:?} vs {:?}",
                weights.shape(),
                updates.shape()
            )));
        }

        let batch = batch.max(1) as f32;
        unsafe {
            cutile_lightnet_kernels::sgd_update_flat(
                weights.inner.device_pointer(),
                updates.inner.device_pointer(),
                weights.len() as i32,
                lr / batch,
                momentum,
                decay * batch,
            )
        }
        .grid((grid_1d(weights.len(), POINTER_TILE_1D), 1, 1))
        .sync_on(&self.stream)?;

        Ok((weights.clone(), updates.clone()))
    }
}

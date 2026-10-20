use std::sync::Arc;

use cutile::api;
use cutile::cuda_core::{Device, Stream};
use cutile::prelude::*;

use crate::error::CutileBackendError;
use crate::kernels::cutile_darknet_kernels;
use crate::validate::ensure_num_elements;

pub(crate) const POINTER_TILE_1D: usize = 128;

#[derive(Clone, Debug)]
pub struct CutileTensorF32 {
    pub(crate) inner: Arc<Tensor<f32>>,
}

impl CutileTensorF32 {
    pub fn shape(&self) -> &[i32] {
        self.inner.shape()
    }

    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape().iter().map(|dim| *dim as usize).collect()
    }

    pub fn arc(&self) -> Arc<Tensor<f32>> {
        Arc::clone(&self.inner)
    }
}

#[derive(Clone, Debug)]
pub struct CutileBackend {
    pub(crate) device: usize,
    pub(crate) context: Arc<Device>,
    pub(crate) stream: Arc<Stream>,
}

impl CutileBackend {
    pub fn new(device: usize) -> Result<Self, CutileBackendError> {
        let context = Device::new(device)?;
        let stream = context.new_stream()?;
        Ok(Self {
            device,
            context,
            stream,
        })
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    pub fn context(&self) -> &Arc<Device> {
        &self.context
    }

    pub fn hello_world(&self, grid: (u32, u32, u32)) -> Result<(), CutileBackendError> {
        cutile_darknet_kernels::hello_world_kernel()
            .grid(grid)
            .sync_on(&self.stream)?;
        Ok(())
    }

    pub fn upload_f32(
        &self,
        shape: &[usize],
        host: &[f32],
    ) -> Result<CutileTensorF32, CutileBackendError> {
        ensure_num_elements(shape, host.len())?;

        let host = Arc::new(host.to_vec());
        let tensor = api::copy_host_vec_to_device(&host).sync_on(&self.stream)?;
        let tensor = tensor.reshape(shape)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(tensor),
        })
    }

    pub fn zeros_f32(&self, shape: &[usize]) -> Result<CutileTensorF32, CutileBackendError> {
        let tensor = self.zeroed_tensor_f32(shape.iter().product::<usize>())?;
        let tensor = tensor.reshape(shape)?;
        Ok(CutileTensorF32 {
            inner: Arc::new(tensor),
        })
    }

    pub fn download_f32(&self, tensor: &CutileTensorF32) -> Result<Vec<f32>, CutileBackendError> {
        Ok(api::copy_device_to_host_vec(&tensor.inner).sync_on(&self.stream)?)
    }

    pub(crate) fn zeroed_tensor_f32(&self, len: usize) -> Result<Tensor<f32>, CutileBackendError> {
        // `api::zeros` launches cuTile's generic `creation::full` kernel.  Keep
        // buffer initialization independent of that JIT-only helper so native
        // flat kernels remain usable under the strict AOT policy.
        let host_zeros = Arc::new(vec![0.0f32; len]);
        Ok(api::copy_host_vec_to_device(&host_zeros).sync_on(&self.stream)?)
    }
}

pub(crate) fn normalize_partition_tile_len(limit: usize, requested: usize) -> usize {
    let capped = requested.min(limit).max(1);
    if capped.is_power_of_two() {
        capped
    } else {
        capped.next_power_of_two() / 2
    }
}

pub(crate) fn grid_1d(len: usize, tile_len: usize) -> u32 {
    len.div_ceil(tile_len) as u32
}

//! Device tensor handle: a shared, immutable f32 device buffer plus its shape.

use std::sync::Arc;

use cuda_core::DeviceBuffer;

/// Cloning is cheap (bumps the `Arc`); ops never mutate an existing tensor,
/// they allocate a fresh output buffer.
#[derive(Clone)]
pub struct OxideTensorF32 {
    pub(crate) buf: Arc<DeviceBuffer<f32>>,
    pub(crate) shape: Vec<usize>,
}

impl OxideTensorF32 {
    pub(crate) fn new(buf: DeviceBuffer<f32>, shape: &[usize]) -> Self {
        Self {
            buf: Arc::new(buf),
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.len() == 0
    }
}

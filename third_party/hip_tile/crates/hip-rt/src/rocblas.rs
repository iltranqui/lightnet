//! rocBLAS wrapper (feature `rocblas`), used as a GEMM reference.

use std::ptr;

use hip_sys::rocblas as sys;

use crate::{DeviceBuffer, HipError, Result, Stream};

fn check(status: sys::rocblas_status) -> Result<()> {
    if status == sys::rocblas_status_success {
        Ok(())
    } else {
        Err(HipError { code: status, message: format!("rocBLAS status {status}") })
    }
}

/// A rocBLAS handle; destroyed on drop.
#[derive(Debug)]
pub struct Handle {
    raw: sys::rocblas_handle,
}

impl Handle {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::rocblas_create_handle(&mut raw) })?;
        Ok(Self { raw })
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        check(unsafe { sys::rocblas_set_stream(self.raw, stream.as_raw()) })
    }

    /// Row-major `C (m x n) = A (m x k) * B (k x n)`, tightly packed.
    ///
    /// Row-major C is column-major C^T, so this computes `C^T = B^T * A^T` by swapping
    /// the operands and the `m`/`n` dimensions.
    pub fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if a.len() < m * k || b.len() < k * n || c.len() < m * n {
            return Err(HipError::other("sgemm: buffer too small for given dimensions"));
        }
        let (alpha, beta) = (1.0f32, 0.0f32);
        check(unsafe {
            sys::rocblas_sgemm(
                self.raw,
                sys::rocblas_operation_none,
                sys::rocblas_operation_none,
                n as i32,
                m as i32,
                k as i32,
                &alpha,
                b.as_ptr(),
                n as i32,
                a.as_ptr(),
                k as i32,
                &beta,
                c.as_mut_ptr(),
                n as i32,
            )
        })
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { sys::rocblas_destroy_handle(self.raw) };
    }
}

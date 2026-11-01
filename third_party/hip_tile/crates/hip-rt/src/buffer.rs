use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr;

use hip_sys::{self as sys, hipMemcpyKind};
use hip_tile::{Tensor, TensorMut};

use crate::{HipError, Result, check};

/// Owned device allocation of `len` elements of `T`; freed on drop.
#[derive(Debug)]
pub struct DeviceBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> DeviceBuffer<T> {
    /// Uninitialised buffer of `len` elements.
    pub fn alloc(len: usize) -> Result<Self> {
        let mut raw: *mut c_void = ptr::null_mut();
        // hipMalloc(0) may return null; allocate at least one byte so `ptr` stays valid.
        let bytes = (len * size_of::<T>()).max(1);
        check(unsafe { sys::hipMalloc(&mut raw, bytes) })?;
        Ok(Self { ptr: raw.cast(), len, _marker: PhantomData })
    }

    /// Buffer of `len` zeroed bytes-worth of elements.
    pub fn zeroed(len: usize) -> Result<Self> {
        let buf = Self::alloc(len)?;
        check(unsafe { sys::hipMemset(buf.ptr.cast(), 0, (len * size_of::<T>()).max(1)) })?;
        Ok(buf)
    }

    pub fn from_slice(data: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(data.len())?;
        buf.copy_from_slice(data)?;
        Ok(buf)
    }

    /// Blocking device-to-host copy of the whole buffer.
    pub fn copy_to_vec(&self) -> Result<Vec<T>> {
        let mut out = Vec::<T>::with_capacity(self.len);
        if self.len > 0 {
            check(unsafe {
                sys::hipMemcpy(
                    out.as_mut_ptr().cast(),
                    self.ptr.cast(),
                    self.len * size_of::<T>(),
                    hipMemcpyKind::DeviceToHost,
                )
            })?;
        }
        // SAFETY: `len` elements were just initialised by the copy.
        unsafe { out.set_len(self.len) };
        Ok(out)
    }

    /// Blocking host-to-device copy; `data.len()` must equal `self.len()`.
    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(HipError::other(format!(
                "copy_from_slice: length mismatch (buffer {}, slice {})",
                self.len,
                data.len()
            )));
        }
        if self.len > 0 {
            check(unsafe {
                sys::hipMemcpy(
                    self.ptr.cast(),
                    data.as_ptr().cast(),
                    self.len * size_of::<T>(),
                    hipMemcpyKind::HostToDevice,
                )
            })?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Device pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Device pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Read-only `rows x cols` row-major view (`ld = cols`).
    pub fn tensor(&self, rows: u32, cols: u32) -> Tensor<T> {
        self.check_shape(rows, cols);
        Tensor::new(self.ptr, rows, cols, cols)
    }

    /// Writable `rows x cols` row-major view (`ld = cols`).
    pub fn tensor_mut(&mut self, rows: u32, cols: u32) -> TensorMut<T> {
        self.check_shape(rows, cols);
        TensorMut::new(self.ptr, rows, cols, cols)
    }

    fn check_shape(&self, rows: u32, cols: u32) {
        assert!(
            rows as usize * cols as usize <= self.len,
            "tensor {rows}x{cols} exceeds buffer of {} elements",
            self.len
        );
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        unsafe { sys::hipFree(self.ptr.cast()) };
    }
}

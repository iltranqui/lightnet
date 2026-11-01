//! Types allowed as kernel parameters.

use crate::tensor::{Tensor, TensorMut};

/// A value that can be passed to a `#[kernel]` by value.
///
/// # Safety
///
/// The type must have the same layout on host and device (`#[repr(C)]` or a
/// primitive) and contain no host-only references, so that copying its bytes into
/// the kernel argument segment is valid.
pub unsafe trait KernelArg: Copy {}

macro_rules! kernel_arg {
    ($($t:ty),*) => { $(unsafe impl KernelArg for $t {})* };
}

kernel_arg!(f32, f64, i32, u32, i64, u64);

unsafe impl<T> KernelArg for *const T {}
unsafe impl<T> KernelArg for *mut T {}
unsafe impl<T> KernelArg for Tensor<T> {}
unsafe impl<T> KernelArg for TensorMut<T> {}

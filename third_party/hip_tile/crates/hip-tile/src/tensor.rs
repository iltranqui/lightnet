//! Tensor views passed to kernels by value.
//!
//! These types are shared by host and device, so their layout is the kernel ABI:
//! `#[repr(C)]`, a pointer followed by `u32` dimensions. Storage is row-major with
//! a leading dimension `ld` (elements between consecutive rows), so a view can
//! describe a sub-matrix of a larger buffer.

/// Read-only 2-D view of device memory.
#[repr(C)]
pub struct Tensor<T> {
    ptr: *const T,
    rows: u32,
    cols: u32,
    ld: u32,
}

/// Writable 2-D view of device memory.
#[repr(C)]
pub struct TensorMut<T> {
    ptr: *mut T,
    rows: u32,
    cols: u32,
    ld: u32,
}

impl<T> Tensor<T> {
    /// `rows x cols` row-major matrix with leading dimension `ld` (`ld >= cols`).
    pub const fn new(ptr: *const T, rows: u32, cols: u32, ld: u32) -> Self {
        Self {
            ptr,
            rows,
            cols,
            ld,
        }
    }

    /// 1-D vector of `len` elements, viewed as a single row.
    pub const fn vector(ptr: *const T, len: u32) -> Self {
        Self::new(ptr, 1, len, len)
    }

    #[inline(always)]
    pub const fn ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    #[inline(always)]
    pub const fn cols(&self) -> u32 {
        self.cols
    }

    #[inline(always)]
    pub const fn ld(&self) -> u32 {
        self.ld
    }
}

impl<T> TensorMut<T> {
    /// `rows x cols` row-major matrix with leading dimension `ld` (`ld >= cols`).
    pub const fn new(ptr: *mut T, rows: u32, cols: u32, ld: u32) -> Self {
        Self {
            ptr,
            rows,
            cols,
            ld,
        }
    }

    /// 1-D vector of `len` elements, viewed as a single row.
    pub const fn vector(ptr: *mut T, len: u32) -> Self {
        Self::new(ptr, 1, len, len)
    }

    /// Read-only view of the same memory.
    #[inline(always)]
    pub const fn as_tensor(&self) -> Tensor<T> {
        Tensor::new(self.ptr, self.rows, self.cols, self.ld)
    }

    #[inline(always)]
    pub const fn ptr(&self) -> *mut T {
        self.ptr
    }

    #[inline(always)]
    pub const fn rows(&self) -> u32 {
        self.rows
    }

    #[inline(always)]
    pub const fn cols(&self) -> u32 {
        self.cols
    }

    #[inline(always)]
    pub const fn ld(&self) -> u32 {
        self.ld
    }
}

macro_rules! manual_copy {
    ($($t:ident),*) => {$(
        impl<T> Clone for $t<T> {
            #[inline(always)]
            fn clone(&self) -> Self {
                *self
            }
        }
        impl<T> Copy for $t<T> {}
    )*};
}

manual_copy!(Tensor, TensorMut);

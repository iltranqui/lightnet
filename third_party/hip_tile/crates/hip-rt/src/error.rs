use std::ffi::CStr;
use std::fmt;

use hip_sys as sys;

/// A failed HIP (or rocBLAS) call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipError {
    pub code: i32,
    pub message: String,
}

pub type Result<T> = std::result::Result<T, HipError>;

impl HipError {
    pub(crate) fn from_code(code: sys::hipError_t) -> Self {
        // SAFETY: returns a static NUL-terminated string (or null for unknown codes).
        let ptr = unsafe { sys::hipGetErrorString(code) };
        let message = if ptr.is_null() {
            "unknown error".to_owned()
        } else {
            unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
        };
        Self { code, message }
    }

    /// Error that did not come from a HIP status code.
    pub fn other(message: impl Into<String>) -> Self {
        Self { code: -1, message: message.into() }
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HIP error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for HipError {}

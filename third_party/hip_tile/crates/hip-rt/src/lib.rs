//! Safe RAII host API over the HIP runtime.

mod buffer;
mod device;
mod error;
mod module;
mod stream;

#[cfg(feature = "rocblas")]
pub mod rocblas;

pub use buffer::DeviceBuffer;
pub use device::Device;
pub use error::{HipError, Result};
pub use module::{Dim3, Function, Module};
pub use stream::{Event, Stream};

use hip_sys as sys;

/// Turn a HIP status code into a `Result`.
pub(crate) fn check(code: sys::hipError_t) -> Result<()> {
    if code == sys::hipSuccess {
        Ok(())
    } else {
        Err(HipError::from_code(code))
    }
}

use std::ffi::{CStr, c_char};

use hip_sys as sys;

use crate::{Result, check};

/// A HIP device, selected as the current device on creation.
#[derive(Debug, Clone)]
pub struct Device {
    index: i32,
    name: String,
    arch: String,
}

impl Device {
    /// Initialise HIP and select device `index`.
    pub fn new(index: i32) -> Result<Self> {
        unsafe {
            check(sys::hipInit(0))?;
            check(sys::hipSetDevice(index))?;
            let mut prop = sys::hipDeviceProp_t([0; 2048]);
            check(sys::hipGetDeviceProperties(&mut prop, index))?;
            let field = |off: usize| {
                CStr::from_ptr(prop.0.as_ptr().add(off).cast::<c_char>())
                    .to_string_lossy()
                    .into_owned()
            };
            Ok(Self {
                index,
                name: field(sys::DEVICE_PROP_NAME_OFFSET),
                arch: field(sys::DEVICE_PROP_GCN_ARCH_NAME_OFFSET),
            })
        }
    }

    /// Initialise HIP and select device 0.
    pub fn init() -> Result<Self> {
        Self::new(0)
    }

    pub fn count() -> Result<i32> {
        let mut n = 0;
        unsafe {
            check(sys::hipInit(0))?;
            check(sys::hipGetDeviceCount(&mut n))?;
        }
        Ok(n)
    }

    pub fn index(&self) -> i32 {
        self.index
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Architecture string such as `gfx1031` (feature flags like `:xnack-` stripped).
    pub fn arch(&self) -> &str {
        self.arch.split(':').next().unwrap_or(&self.arch)
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { sys::hipDeviceSynchronize() })
    }
}

use std::ffi::{CString, c_void};
use std::ptr;

use hip_sys as sys;

use crate::{HipError, Result, Stream, check};

/// Grid or block dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl From<u32> for Dim3 {
    fn from(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }
}

impl From<(u32, u32)> for Dim3 {
    fn from((x, y): (u32, u32)) -> Self {
        Self { x, y, z: 1 }
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from((x, y, z): (u32, u32, u32)) -> Self {
        Self { x, y, z }
    }
}

/// A loaded code object; unloaded on drop. Functions must not outlive it.
#[derive(Debug)]
pub struct Module {
    raw: sys::hipModule_t,
}

impl Module {
    /// Load an AMDGPU code object. HIP copies the image during the call.
    pub fn from_elf(elf: &[u8]) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::hipModuleLoadData(&mut raw, elf.as_ptr().cast()) })?;
        Ok(Self { raw })
    }

    pub fn function(&self, name: &str) -> Result<Function<'_>> {
        let cname = CString::new(name).map_err(|_| HipError::other("kernel name contains NUL"))?;
        let mut raw = ptr::null_mut();
        check(unsafe { sys::hipModuleGetFunction(&mut raw, self.raw, cname.as_ptr()) })?;
        Ok(Function { raw, _module: self })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe { sys::hipModuleUnload(self.raw) };
    }
}

/// A kernel entry point inside a [`Module`].
#[derive(Debug, Clone, Copy)]
pub struct Function<'m> {
    raw: sys::hipFunction_t,
    _module: &'m Module,
}

impl Function<'_> {
    /// Asynchronously launch the kernel on `stream` (default stream when `None`).
    ///
    /// # Safety
    ///
    /// `args` must hold one pointer per kernel parameter, each pointing to a live value
    /// of exactly the parameter's type, and device pointers inside must be valid.
    pub unsafe fn launch(
        &self,
        stream: Option<&Stream>,
        grid: Dim3,
        block: Dim3,
        lds_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        check(unsafe {
            sys::hipModuleLaunchKernel(
                self.raw,
                grid.x,
                grid.y,
                grid.z,
                block.x,
                block.y,
                block.z,
                lds_bytes,
                stream.map_or(ptr::null_mut(), Stream::as_raw),
                args.as_mut_ptr(),
                ptr::null_mut(),
            )
        })
    }
}

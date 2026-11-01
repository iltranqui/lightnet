use std::ptr;

use hip_sys as sys;

use crate::{Result, check};

/// A HIP stream; destroyed on drop.
#[derive(Debug)]
pub struct Stream {
    raw: sys::hipStream_t,
}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::hipStreamCreate(&mut raw) })?;
        Ok(Self { raw })
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { sys::hipStreamSynchronize(self.raw) })
    }

    pub fn as_raw(&self) -> sys::hipStream_t {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { sys::hipStreamDestroy(self.raw) };
    }
}

/// A HIP event; destroyed on drop.
#[derive(Debug)]
pub struct Event {
    raw: sys::hipEvent_t,
}

impl Event {
    pub fn new() -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::hipEventCreate(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Record on `stream`, or on the default stream when `None`.
    pub fn record(&self, stream: Option<&Stream>) -> Result<()> {
        check(unsafe { sys::hipEventRecord(self.raw, stream.map_or(ptr::null_mut(), Stream::as_raw)) })
    }

    pub fn synchronize(&self) -> Result<()> {
        check(unsafe { sys::hipEventSynchronize(self.raw) })
    }

    /// Milliseconds from `start` to `self`; waits for `self` to complete.
    pub fn elapsed_ms(&self, start: &Event) -> Result<f32> {
        self.synchronize()?;
        let mut ms = 0.0;
        check(unsafe { sys::hipEventElapsedTime(&mut ms, start.raw, self.raw) })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { sys::hipEventDestroy(self.raw) };
    }
}

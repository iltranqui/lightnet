//! Device tensor handle: a shared, immutable f32 device buffer plus its shape.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use cuda_core::DeviceBuffer;

/// Size-keyed free list of device buffers.  Every op output is a fresh buffer,
/// so without reuse each forward pays a synchronous `cuMemAlloc` / `cuMemFree`
/// per op.  All work runs on one stream, so a recycled buffer is always
/// ordered after the kernels that last used it.
pub(crate) struct BufPool {
    free: Mutex<(HashMap<usize, Vec<DeviceBuffer<f32>>>, usize)>,
    /// Separate free list for 16-bit (f16 / bf16 bit pattern) buffers; shares
    /// the byte cap with the f32 list.
    free16: Mutex<HashMap<usize, Vec<DeviceBuffer<u16>>>>,
    cap_bytes: usize,
}

impl BufPool {
    /// `DARKNET_OXIDE_POOL_MB` caps the cached bytes.  Unset means "auto": half
    /// the free device memory (capped at 4 GiB), which is what batch training
    /// needs to avoid freeing and re-allocating the whole working set every
    /// step.  `0` disables caching.
    pub(crate) fn new() -> Arc<Self> {
        let cap_bytes = match std::env::var("DARKNET_OXIDE_POOL_MB") {
            Ok(v) => v.parse::<usize>().unwrap_or(1024) << 20,
            Err(_) => auto_cap_bytes(),
        };
        Arc::new(Self {
            free: Mutex::new((HashMap::new(), 0)),
            free16: Mutex::new(HashMap::new()),
            cap_bytes,
        })
    }

    pub(crate) fn cached_bytes(&self) -> usize {
        self.free.lock().map(|g| g.1).unwrap_or(0)
    }

    /// Drops every cached buffer.  Called before retrying an allocation that
    /// failed with OOM, since the cache is not in use by any tensor.
    pub(crate) fn flush(&self) {
        if let Ok(mut g) = self.free.lock() {
            g.0.clear();
            g.1 = 0;
        }
        if let Ok(mut g) = self.free16.lock() {
            g.clear();
        }
    }

    pub(crate) fn take(&self, len: usize) -> Option<DeviceBuffer<f32>> {
        let mut g = self.free.lock().ok()?;
        let buf = g.0.get_mut(&len)?.pop()?;
        g.1 -= buf.num_bytes();
        Some(buf)
    }

    pub(crate) fn take16(&self, len: usize) -> Option<DeviceBuffer<u16>> {
        let buf = {
            let mut g = self.free16.lock().ok()?;
            g.get_mut(&len)?.pop()?
        };
        if let Ok(mut f) = self.free.lock() {
            f.1 = f.1.saturating_sub(buf.num_bytes());
        }
        Some(buf)
    }

    fn give16(&self, buf: DeviceBuffer<u16>) {
        if buf.len() == 0 {
            return;
        }
        // The f32 list owns the shared byte counter.
        let Ok(mut f) = self.free.lock() else { return };
        if f.1 + buf.num_bytes() > self.cap_bytes {
            return;
        }
        if let Ok(mut g) = self.free16.lock() {
            f.1 += buf.num_bytes();
            g.entry(buf.len()).or_default().push(buf);
        }
    }

    fn give(&self, buf: DeviceBuffer<f32>) {
        if buf.len() == 0 {
            return;
        }
        if let Ok(mut g) = self.free.lock()
            && g.1 + buf.num_bytes() <= self.cap_bytes
        {
            g.1 += buf.num_bytes();
            g.0.entry(buf.len()).or_default().push(buf);
        }
        // otherwise `buf` drops here and is freed
    }
}

/// Half of the free device memory, capped at 4 GiB; 1 GiB when the query fails
/// (e.g. no current context on this thread).
fn auto_cap_bytes() -> usize {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: valid out-pointers; a context is current when the backend is
    // constructed on this thread.
    let r = unsafe { cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total) };
    if r == cuda_core::sys::cudaError_enum_CUDA_SUCCESS && total > 0 {
        (free / 2).min(4usize << 30)
    } else {
        1024 << 20
    }
}

/// A device buffer that returns to its [`BufPool`] on drop.
pub(crate) struct PooledBuf {
    buf: Option<DeviceBuffer<f32>>,
    pool: Arc<BufPool>,
}

impl PooledBuf {
    pub(crate) fn new(buf: DeviceBuffer<f32>, pool: &Arc<BufPool>) -> Self {
        Self { buf: Some(buf), pool: pool.clone() }
    }


}

impl Deref for PooledBuf {
    type Target = DeviceBuffer<f32>;
    fn deref(&self) -> &DeviceBuffer<f32> {
        self.buf.as_ref().expect("buffer present until drop")
    }
}

impl DerefMut for PooledBuf {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<f32> {
        self.buf.as_mut().expect("buffer present until drop")
    }
}

impl Drop for PooledBuf {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.give(b);
        }
    }
}

/// Cloning is cheap (bumps the `Arc`); ops never mutate an existing tensor,
/// they allocate a fresh output buffer.
///
/// A tensor is either f32 (`half == None`, `buf` holds the data) or, in AMP
/// mode (`OxideBackend::with_amp`), f16 storage (`half == Some(..)` holds the
/// binary16 bit patterns and `buf` is a shared zero-length placeholder).  The
/// f32-only ops widen half tensors on entry (`OxideBackend::f32v`), so the
/// `buf` field is only ever read for f32 tensors.
#[derive(Clone)]
pub struct OxideTensorF32 {
    pub(crate) buf: Arc<PooledBuf>,
    pub(crate) shape: Vec<usize>,
    pub(crate) half: Option<Arc<PooledBuf16>>,
}

impl OxideTensorF32 {
    pub(crate) fn new(buf: PooledBuf, shape: &[usize]) -> Self {
        Self {
            buf: Arc::new(buf),
            shape: shape.to_vec(),
            half: None,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements (of either storage type).
    pub fn len(&self) -> usize {
        match &self.half {
            Some(h) => h.len(),
            None => self.buf.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// True when the data lives in f16 storage (AMP mode).
    pub fn is_half(&self) -> bool {
        self.half.is_some()
    }
}

/// A 16-bit device buffer that returns to its [`BufPool`] on drop.
pub(crate) struct PooledBuf16 {
    buf: Option<DeviceBuffer<u16>>,
    pool: Arc<BufPool>,
}

impl PooledBuf16 {
    pub(crate) fn new(buf: DeviceBuffer<u16>, pool: &Arc<BufPool>) -> Self {
        Self { buf: Some(buf), pool: pool.clone() }
    }
}

impl Deref for PooledBuf16 {
    type Target = DeviceBuffer<u16>;
    fn deref(&self) -> &DeviceBuffer<u16> {
        self.buf.as_ref().expect("buffer present until drop")
    }
}

impl DerefMut for PooledBuf16 {
    fn deref_mut(&mut self) -> &mut DeviceBuffer<u16> {
        self.buf.as_mut().expect("buffer present until drop")
    }
}

impl Drop for PooledBuf16 {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.give16(b);
        }
    }
}

/// Half-precision tensor: IEEE binary16 bit patterns (`u16`) in NCHW (or
/// `[filters, kpad]` for padded conv weights).  Same sharing rules as
/// [`OxideTensorF32`].  The conversion ops also use it to carry bf16 bit
/// patterns; the type itself does not know which 16-bit format it holds.
#[derive(Clone)]
pub struct OxideTensorF16 {
    pub(crate) buf: Arc<PooledBuf16>,
    pub(crate) shape: Vec<usize>,
}

impl OxideTensorF16 {
    pub(crate) fn new(buf: PooledBuf16, shape: &[usize]) -> Self {
        Self { buf: Arc::new(buf), shape: shape.to_vec() }
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

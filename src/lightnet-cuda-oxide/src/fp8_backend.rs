//! FP8 quantization backend.

use super::*;

impl OxideBackend {
    /// Count of quantise kernel launches made by the FP8 path so far.  A cached
    /// weight costs no launch, so tests compare it to pin the weight cache.
    pub fn fp8_quantize_launches(&self) -> u64 {
        self.fp8_quantize_launches.load(Ordering::Relaxed)
    }

    /// Count of stand-alone amax passes made by [`Self::fp8_quantize_half`] so
    /// far (the first pass of current scaling, or of delayed scaling's call 0).
    /// Delayed scaling measures the amax inside the quantise kernel, so after
    /// call 0 a key adds none.
    pub fn fp8_amax_passes(&self) -> u64 {
        self.fp8_amax_passes.load(Ordering::Relaxed)
    }

    pub fn fp8_scaling(&self) -> Fp8Scaling {
        self.fp8_scaling
    }

    /// Test seam: quantise `x` to `fmt` through the activation quantiser
    /// (honouring [`Fp8Scaling`]) and return the bytes and the scale pair
    /// `[factor, 1 / factor]`.  `key` overrides the delayed-history key (the
    /// source buffer address plus its pool incarnation) so a test can simulate
    /// one call site whose buffer changes between calls.
    pub fn fp8_quantize_probe(
        &self,
        x: &OxideTensorF32,
        fmt: Fp8Format,
        key: Option<u64>,
    ) -> Result<(Vec<u8>, [f32; 2]), OxideError> {
        let (bytes, scale) = self.fp8_quantize_keyed(&x.buf, x.len(), fmt, key)?;
        let bytes = if bytes.len() == 0 { Vec::new() } else { bytes.to_host_vec(&self.stream)? };
        let scale = scale.to_host_vec(&self.stream)?;
        Ok((bytes, [scale[0], scale[1]]))
    }

    /// The FP8 formats used by the dgrad / wgrad GEMMs, if enabled.
    pub fn fp8_backward_formats(&self) -> Option<(Fp8Format, Fp8Format)> {
        self.bwd_fp8
    }

    /// Word offset of the next zeroed slot of the amax ring.  Once every
    /// [`FP8_RING_SLOTS`] calls the whole ring is zero-filled again (one
    /// memset), which is safe because the stream orders it after every kernel
    /// of the previous lap.
    pub(crate) fn fp8_ring_guard(&self) -> Result<std::sync::MutexGuard<'_, Fp8Ring>, OxideError> {
        self.fp8_ring
            .lock()
            .map_err(|_| shape_err("fp8_ring", "lock poisoned".into()))
    }

    pub(crate) fn fp8_next_slot(&self) -> Result<u32, OxideError> {
        let mut ring = self.fp8_ring_guard()?;
        if ring.next == FP8_RING_SLOTS {
            ring.buf.zero_async(&self.stream)?;
            ring.next = 0;
        }
        let off = ring.next * FP8_SLOT_WORDS;
        ring.next += 1;
        Ok(off as u32)
    }

    /// Quantise the first `len` elements of a device buffer to `fmt`: one amax
    /// pass, then one quantise pass that derives `factor = max_finite / amax`
    /// from the amax slot and also writes the device scale pair
    /// `[factor, 1 / factor]` the mma epilogue consumes.  The host never reads
    /// either word.
    ///
    /// Note the direction: [`fp8::amax_to_scale`] returns the C++ *divisor*
    /// `amax / max_finite`, and the quantiser multiplies, so the factor is its
    /// reciprocal.  Getting this backwards maps every operand onto the bottom
    /// of the format's range (an e5m2 typical value lands below `2^-16` and is
    /// flushed) and silently destroys precision.
    pub(crate) fn fp8_quantize_operand(
        &self,
        buf: &DeviceBuffer<f32>,
        len: usize,
        fmt: Fp8Format,
        site: u64,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        self.fp8_quantize_keyed(buf, len, fmt, Some(site))
    }

    /// [`Self::fp8_quantize_operand`] with an optional delayed-history key
    /// override; `None` keys on the buffer's device address and pool
    /// incarnation.
    fn fp8_quantize_keyed(
        &self,
        buf: &DeviceBuffer<f32>,
        len: usize,
        fmt: Fp8Format,
        key: Option<u64>,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if self.fp8_scaling == Fp8Scaling::Delayed && len > 0 {
            if let Some(done) = self.fp8_quantize_delayed(Fp8Src::F32(buf), len, fmt, key)? {
                return Ok(done);
            }
        }
        self.fp8_quantize_operand_current(buf, len, fmt)
    }

    /// Delayed scaling for one operand; `None` when the history is full and the
    /// caller should use current scaling.  Call `n` of a key writes slot
    /// `n % 16` and zeroes slot `(n + 2) % 16`; call 0 (empty history) measures
    /// its own amax into its slot, then quantises with it (the two-pass path).
    fn fp8_quantize_delayed(
        &self,
        src: Fp8Src<'_>,
        len: usize,
        fmt: Fp8Format,
        key: Option<u64>,
    ) -> Result<Option<(PooledBuf8, PooledBuf)>, OxideError> {
        let hist_len = gemm_f8::F8_HIST_LEN;
        let mut guard = self
            .fp8_hist
            .lock()
            .map_err(|_| shape_err("fp8_hist", "lock poisoned".into()))?;
        if guard.is_none() {
            *guard = Some(Fp8Hist {
                buf: DeviceBuffer::<f32>::zeroed(&self.stream, FP8_HIST_KEYS * hist_len)?,
                keys: std::collections::HashMap::new(),
            });
        }
        let hist = guard.as_mut().expect("initialised above");
        // 16-bit sources fold their format into the code's upper bits, so a
        // pooled buffer that changes dtype never shares a history.
        let (ptr, code) = match src {
            Fp8Src::F32(b) => (b.cu_deviceptr() as u64, fmt.code()),
            Fp8Src::Half(b, hf) => (b.cu_deviceptr() as u64, fmt.code() | ((hf.code() + 1) << 8)),
        };
        let key = match key {
            Some(k) => (k, 0, len, code),
            None => (ptr, self.bufs.incarnation(ptr as usize), len, code),
        };
        let slots_used = hist.keys.len();
        let entry = match hist.keys.get_mut(&key) {
            Some(e) => e,
            None if slots_used < FP8_HIST_KEYS => hist.keys.entry(key).or_insert((slots_used * hist_len, 0)),
            None => return Ok(None),
        };
        let (base, n) = (entry.0, entry.1);
        entry.1 += 1;
        let w = (n % hist_len as u64) as u32;
        let z = ((n + 2) % hist_len as u64) as u32;
        let mut out = self.alloc8_uninit(len)?;
        let mut scale = self.alloc_uninit(2)?;
        if n == 0 {
            // Empty history: current scaling, with the amax landing in slot `w`.
            // SAFETY: one thread per element; slot `w` of a fresh key is zero and
            // only touched through `fetch_max`.
            let amax_grid = launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32);
            let quant_grid = launch::elementwise(len.max(2), ElementwiseLaunchConfig::default());
            match src {
                Fp8Src::F32(buf) => unsafe {
                    self.gemm_f8
                        .fp8_amax_f32_kernel(&self.stream, amax_grid, buf, &mut hist.buf, base as u32 + w, len as u32)
                        .map_err(launch_err)?;
                    // SAFETY: as `fp8_quantize_operand_current`.
                    self.gemm_f8
                        .fp8_quantize_kernel(
                            &self.stream,
                            quant_grid,
                            buf,
                            &hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32 + w,
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
                Fp8Src::Half(buf, hf) => unsafe {
                    self.fp8_amax_passes.fetch_add(1, Ordering::Relaxed);
                    self.gemm_f8
                        .fp8_amax_half_kernel(&self.stream, amax_grid, buf, &mut hist.buf, base as u32 + w, hf.code(), len as u32)
                        .map_err(launch_err)?;
                    // SAFETY: as `fp8_quantize_half`.
                    self.gemm_f8
                        .fp8_quantize_half_kernel(
                            &self.stream,
                            quant_grid,
                            buf,
                            &hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32 + w,
                            hf.code(),
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
            }
        } else {
            let blocks = len.max(2).div_ceil(gemm_f8::F8_AMAX_BLOCK);
            let grid = launch::blocks_of(blocks, gemm_f8::F8_AMAX_BLOCK as u32);
            // SAFETY: whole 256-thread blocks; the kernel reads only history
            // slots other than `w` / `z`, writes slot `w` atomically and zeroes
            // `z`, writes every byte of `out` and both scale words.
            match src {
                Fp8Src::F32(buf) => unsafe {
                    self.gemm_f8
                        .fp8_quantize_delayed_kernel(
                            &self.stream,
                            grid,
                            buf,
                            &mut hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32,
                            w,
                            z,
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
                Fp8Src::Half(buf, hf) => unsafe {
                    self.gemm_f8
                        .fp8_quantize_delayed_half_kernel(
                            &self.stream,
                            grid,
                            buf,
                            &mut hist.buf,
                            &mut out,
                            &mut scale,
                            fmt.code(),
                            base as u32,
                            w,
                            z,
                            hf.code(),
                            len as u32,
                        )
                        .map_err(launch_err)?;
                },
            }
        }
        self.fp8_quantize_launches.fetch_add(1, Ordering::Relaxed);
        Ok(Some((out, scale)))
    }

    /// Current-scaling quantiser of one operand (two passes: amax, quantise).
    fn fp8_quantize_operand_current(
        &self,
        buf: &DeviceBuffer<f32>,
        len: usize,
        fmt: Fp8Format,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if len == 0 {
            return Ok((self.alloc8_uninit(0)?, self.unit_scale()?));
        }
        // Pooled: no cudaMalloc/free per operand.  The amax slot comes zeroed
        // from the ring; the scale pair and bytes are fully written by the
        // quantise kernel.
        let off = self.fp8_next_slot()?;
        let mut ring = self.fp8_ring_guard()?;
        // SAFETY: one thread per element; the ring slot is zeroed and only
        // touched through `fetch_max` while the kernel runs.
        unsafe {
            self.gemm_f8.fp8_amax_f32_kernel(
                &self.stream,
                launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                buf,
                &mut ring.buf,
                off,
                len as u32,
            )
        }
        .map_err(launch_err)?;
        let mut out = self.alloc8_uninit(len)?;
        let mut scale = self.alloc_uninit(2)?;
        // SAFETY: one thread per element (at least two, so both scale words are
        // written); the kernel writes every byte of `out` and both scale words.
        unsafe {
            self.gemm_f8.fp8_quantize_kernel(
                &self.stream,
                launch::elementwise(len.max(2), ElementwiseLaunchConfig::default()),
                buf,
                &ring.buf,
                &mut out,
                &mut scale,
                fmt.code(),
                off,
                len as u32,
            )
        }
        .map_err(launch_err)?;
        self.fp8_quantize_launches.fetch_add(1, Ordering::Relaxed);
        Ok((out, scale))
    }

    /// [`Self::fp8_quantize_operand_current`] reading 16-bit storage (`hfmt`
    /// says how to widen it): same factor, scale pair and bytes as widening to
    /// f32 first, with no f32 intermediate.  Under [`Fp8Scaling::Delayed`] it
    /// takes the one-pass delayed path of [`Self::fp8_quantize_delayed`] (keyed
    /// on the 16-bit buffer address, its pool incarnation, its length and both
    /// formats); current scaling otherwise.
    pub(crate) fn fp8_quantize_half(
        &self,
        buf: &DeviceBuffer<u16>,
        len: usize,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
        site: u64,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        self.fp8_quantize_half_keyed(buf, len, fmt, hfmt, Some(site))
    }

    /// [`Self::fp8_quantize_half`] with an optional delayed-history key
    /// override; `None` keys on the buffer's device address and pool
    /// incarnation.
    fn fp8_quantize_half_keyed(
        &self,
        buf: &DeviceBuffer<u16>,
        len: usize,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
        key: Option<u64>,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if self.fp8_scaling == Fp8Scaling::Delayed && len > 0 {
            if let Some(done) = self.fp8_quantize_delayed(Fp8Src::Half(buf, hfmt), len, fmt, key)? {
                return Ok(done);
            }
        }
        self.fp8_quantize_half_current(buf, len, fmt, hfmt)
    }

    /// Current-scaling (two-pass) body of [`Self::fp8_quantize_half`].
    fn fp8_quantize_half_current(
        &self,
        buf: &DeviceBuffer<u16>,
        len: usize,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<(PooledBuf8, PooledBuf), OxideError> {
        if len == 0 {
            return Ok((self.alloc8_uninit(0)?, self.unit_scale()?));
        }
        self.fp8_amax_passes.fetch_add(1, Ordering::Relaxed);
        let off = self.fp8_next_slot()?;
        let mut ring = self.fp8_ring_guard()?;
        // SAFETY: one thread per element; the ring slot is zeroed and only
        // touched through `fetch_max` while the kernel runs.
        unsafe {
            self.gemm_f8.fp8_amax_half_kernel(
                &self.stream,
                launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                buf,
                &mut ring.buf,
                off,
                hfmt.code(),
                len as u32,
            )
        }
        .map_err(launch_err)?;
        let mut out = self.alloc8_uninit(len)?;
        let mut scale = self.alloc_uninit(2)?;
        // SAFETY: one thread per element (at least two, so both scale words are
        // written); the kernel writes every byte of `out` and both scale words.
        unsafe {
            self.gemm_f8.fp8_quantize_half_kernel(
                &self.stream,
                launch::elementwise(len.max(2), ElementwiseLaunchConfig::default()),
                buf,
                &ring.buf,
                &mut out,
                &mut scale,
                fmt.code(),
                off,
                hfmt.code(),
                len as u32,
            )
        }
        .map_err(launch_err)?;
        self.fp8_quantize_launches.fetch_add(1, Ordering::Relaxed);
        Ok((out, scale))
    }

    /// Test seam for [`Self::fp8_quantize_half`]: the bytes and the scale pair
    /// `[factor, 1 / factor]`.
    pub fn fp8_quantize_half_probe(
        &self,
        x: &OxideTensorF16,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<(Vec<u8>, [f32; 2]), OxideError> {
        self.fp8_quantize_half_probe_keyed(x, fmt, hfmt, None)
    }

    /// [`Self::fp8_quantize_half_probe`] with a delayed-history key override
    /// (see [`Self::fp8_quantize_probe`]).
    pub fn fp8_quantize_half_probe_keyed(
        &self,
        x: &OxideTensorF16,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
        key: Option<u64>,
    ) -> Result<(Vec<u8>, [f32; 2]), OxideError> {
        let (bytes, scale) = self.fp8_quantize_half_keyed(&x.buf, x.len(), fmt, hfmt, key)?;
        let bytes = if bytes.len() == 0 { Vec::new() } else { bytes.to_host_vec(&self.stream)? };
        let scale = scale.to_host_vec(&self.stream)?;
        Ok((bytes, [scale[0], scale[1]]))
    }

    /// Quantise two operands together: one zero-fill, one amax launch and one
    /// quantise launch instead of two of each.  Falls back to the per-operand
    /// path when either operand is empty.
    #[allow(clippy::type_complexity)]
    pub(crate) fn fp8_quantize_pair(
        &self,
        a: &DeviceBuffer<f32>,
        a_len: usize,
        fmt_a: Fp8Format,
        key_a: u64,
        b: &DeviceBuffer<f32>,
        b_len: usize,
        fmt_b: Fp8Format,
        key_b: u64,
    ) -> Result<((PooledBuf8, PooledBuf), (PooledBuf8, PooledBuf)), OxideError> {
        if a_len == 0 || b_len == 0 || self.fp8_scaling == Fp8Scaling::Delayed {
            return Ok((
                self.fp8_quantize_operand(a, a_len, fmt_a, key_a)?,
                self.fp8_quantize_operand(b, b_len, fmt_b, key_b)?,
            ));
        }
        let off = self.fp8_next_slot()?;
        let mut ring = self.fp8_ring_guard()?;
        // SAFETY: whole 256-thread blocks per operand tile; the ring slot is
        // zeroed and only touched through `fetch_max`.
        let tile = gemm_f8::F8_AMAX_BLOCK;
        let blocks = a_len.div_ceil(tile) + b_len.div_ceil(tile);
        unsafe {
            self.gemm_f8.fp8_amax_pair_kernel(
                &self.stream,
                launch::blocks_of(blocks, tile as u32),
                a,
                b,
                a_len as u32,
                b_len as u32,
                &mut ring.buf,
                off,
            )
        }
        .map_err(launch_err)?;
        let mut out_a = self.alloc8_uninit(a_len)?;
        let mut out_b = self.alloc8_uninit(b_len)?;
        let mut scale_a = self.alloc_uninit(2)?;
        let mut scale_b = self.alloc_uninit(2)?;
        let threads = a_len.max(2) + b_len.max(2);
        // SAFETY: the kernel writes every byte of both outputs and both words of
        // both scale pairs, each from a distinct thread.
        unsafe {
            self.gemm_f8.fp8_quantize_pair_kernel(
                &self.stream,
                launch::elementwise(threads, ElementwiseLaunchConfig::default()),
                a,
                b,
                &ring.buf,
                &mut out_a,
                &mut out_b,
                &mut scale_a,
                &mut scale_b,
                a_len as u32,
                b_len as u32,
                fmt_a.code(),
                fmt_b.code(),
                off,
            )
        }
        .map_err(launch_err)?;
        self.fp8_quantize_launches.fetch_add(2, Ordering::Relaxed);
        Ok(((out_a, scale_a), (out_b, scale_b)))
    }

    /// Absolute maximum of the first `len` elements of `buf` in a ring slot
    /// (returns its word offset; read by the device, never by the host).  Used by the
    /// byte im2col, which quantises while it gathers.
    pub(crate) fn fp8_amax_slot(&self, buf: &DeviceBuffer<f32>, len: usize) -> Result<u32, OxideError> {
        let off = self.fp8_next_slot()?;
        if len > 0 {
            let mut ring = self.fp8_ring_guard()?;
            // SAFETY: one thread per element; the ring slot is zeroed and only
            // touched through `fetch_max`.
            unsafe {
                self.gemm_f8.fp8_amax_f32_kernel(
                    &self.stream,
                    launch::blocks_of(len.div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                    buf,
                    &mut ring.buf,
                    off,
                    len as u32,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(off)
    }

    /// Quantised copy of a weight tensor, cached by buffer identity and format.
    /// Weight tensors are immutable (an optimizer step returns a new tensor), so
    /// the copy refreshes once per update.  Only weight tensors may use this:
    /// pooled activation buffers are reused under the same address, so an
    /// identity key on them would serve stale bytes.
    pub(crate) fn fp8_weights(&self, w: &OxideTensorF32, fmt: Fp8Format) -> Result<mixed::Fp8Weights, OxideError> {
        let ptr = Arc::as_ptr(&w.buf);
        {
            let cache = self
                .fp8_w
                .lock()
                .map_err(|_| shape_err("fp8_weights", "lock poisoned".into()))?;
            if let Some(e) = cache
                .iter()
                .find(|e| e.key.as_ptr() == ptr && e.key.strong_count() > 0 && e.fmt == fmt)
            {
                return Ok(e.value.clone());
            }
        }
        let (bytes, scale) = self.fp8_quantize_operand_current(&w.buf, w.len(), fmt)?;
        let value = mixed::Fp8Weights {
            bytes: Arc::new(bytes),
            scale: Arc::new(scale),
        };
        let mut cache = self
            .fp8_w
            .lock()
            .map_err(|_| shape_err("fp8_weights", "lock poisoned".into()))?;
        cache.retain(|e| e.key.strong_count() > 0);
        if cache.len() >= mixed::W16_CACHE_MAX {
            cache.remove(0);
        }
        cache.push(mixed::Fp8WEntry {
            key: Arc::downgrade(&w.buf),
            fmt,
            value: value.clone(),
        });
        Ok(value)
    }

    /// Quantised copy of a conv output gradient, shared by the dgrad and the
    /// wgrad of one backward: the first caller quantises the whole tensor (one
    /// keyed quantise, so delayed scaling advances its history once), the
    /// second gets the same bytes and scale pair.  Only the most recent
    /// gradient is kept.  Valid because tensors are immutable once built and a
    /// dead tensor's entry is skipped by the `Weak` check.
    pub(crate) fn fp8_dy(&self, dy: &OxideTensorF32, fmt: Fp8Format) -> Result<mixed::Fp8Weights, OxideError> {
        let lock_err = || shape_err("fp8_dy", "lock poisoned".into());
        let ptr = Arc::as_ptr(&dy.buf);
        {
            let cache = self.fp8_dy.lock().map_err(|_| lock_err())?;
            if let Some(e) = cache.as_ref().filter(|e| {
                e.key.as_ptr() == ptr && e.key.strong_count() > 0 && e.len == dy.len() && e.fmt == fmt
            }) {
                return Ok(e.value.clone());
            }
        }
        let (bytes, scale) = self.fp8_quantize_operand(&dy.buf, dy.len(), fmt, FP8_SITE_DY)?;
        let value = mixed::Fp8Weights {
            bytes: Arc::new(bytes),
            scale: Arc::new(scale),
        };
        *self.fp8_dy.lock().map_err(|_| lock_err())? = Some(mixed::Fp8DyEntry {
            key: Arc::downgrade(&dy.buf),
            len: dy.len(),
            fmt,
            value: value.clone(),
        });
        Ok(value)
    }

    /// [`Self::fp8_dy`] for a 16-bit-storage `dy` (AMP): dgrad and wgrad of one
    /// conv backward share one quantise of the whole gradient.  Keyed on the
    /// half buffer identity, its length and the formats.
    pub(crate) fn fp8_dy_half(
        &self,
        dy: &OxideTensorF16,
        fmt: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<mixed::Fp8Weights, OxideError> {
        let lock_err = || shape_err("fp8_dy_half", "lock poisoned".into());
        let ptr = Arc::as_ptr(&dy.buf);
        {
            let cache = self.fp8_dy_half.lock().map_err(|_| lock_err())?;
            if let Some(e) = cache.as_ref().filter(|e| {
                e.key.as_ptr() == ptr && e.key.strong_count() > 0 && e.len == dy.len() && e.fmt == fmt && e.hfmt == hfmt
            }) {
                return Ok(e.value.clone());
            }
        }
        let (bytes, scale) = self.fp8_quantize_half(&dy.buf, dy.len(), fmt, hfmt, FP8_SITE_DY)?;
        let value = mixed::Fp8Weights {
            bytes: Arc::new(bytes),
            scale: Arc::new(scale),
        };
        *self.fp8_dy_half.lock().map_err(|_| lock_err())? = Some(mixed::Fp8DyHalfEntry {
            key: Arc::downgrade(&dy.buf),
            len: dy.len(),
            fmt,
            hfmt,
            value: value.clone(),
        });
        Ok(value)
    }

    /// Largest finite `|x|` over `values`; non-finite entries count as 0.
    /// Same rule as [`fp8::amax_f32`] and the C++ amax kernels.
    pub fn fp8_amax(&self, values: &[f32]) -> Result<f32, OxideError> {
        if values.is_empty() {
            return Ok(0.0);
        }
        let x = DeviceBuffer::<f32>::from_host(&self.stream, values)?;
        let mut slot = DeviceBuffer::<f32>::zeroed(&self.stream, 1)?;
        // SAFETY: one thread per input element; the single atomic slot is
        // zeroed above and only ever read through `fetch_max`.
        unsafe {
            self.gemm_f8.fp8_amax_f32_kernel(
                &self.stream,
                launch::blocks_of(values.len().div_ceil(gemm_f8::F8_AMAX_BLOCK), gemm_f8::F8_AMAX_BLOCK as u32),
                &x,
                &mut slot,
                0,
                values.len() as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(slot.to_host_vec(&self.stream)?[0])
    }

    /// Raw-fragment `mma.m16n8k16` probe: one warp of fragments is 32 lanes x
    /// 8 A bytes and 32 lanes x 4 B bytes; the result is
    /// `out[i * 32 + lane] = D[lane][i]` for a 16x8 f32 tile.
    pub fn fp8_mma_probe16(
        &self,
        a: &[u8],
        b: &[u8],
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<Vec<f32>, OxideError> {
        if a.len() < 256 || b.len() < 128 {
            return Err(shape_err(
                "fp8_mma_probe16",
                format!(
                    "fragments must be 256 A bytes and 128 B bytes, got {} and {}",
                    a.len(),
                    b.len()
                ),
            ));
        }
        let da = DeviceBuffer::<u8>::from_host(&self.stream, &a[..256])?;
        let db = DeviceBuffer::<u8>::from_host(&self.stream, &b[..128])?;
        let mut out = DeviceBuffer::<f32>::from_host(&self.stream, &vec![-999.0f32; 128])?;
        // SAFETY: a single warp; every lane's fragments are present and each
        // of the 128 output slots has exactly one owner.
        unsafe {
            self.gemm_f8.fp8_mma_probe16(
                &self.stream,
                launch::blocks_of(1, 32),
                &da,
                &db,
                &mut out,
                fmt_a.code(),
                fmt_b.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

    /// Raw-fragment `mma.m16n8k32` probe: one warp of fragments is 32 lanes x
    /// 16 A bytes and 32 lanes x 8 B bytes, and the result is
    /// `out[i * 32 + lane] = D[lane][i]` for a 16x8 f32 tile.  Used by
    /// `tests/oxide_gemm_f8.rs` to pin the documented fragment layout.
    pub fn fp8_mma_probe(
        &self,
        a: &[u8],
        b: &[u8],
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<Vec<f32>, OxideError> {
        if a.len() < 512 || b.len() < 256 {
            return Err(shape_err(
                "fp8_mma_probe",
                format!(
                    "fragments must be 512 A bytes and 256 B bytes, got {} and {}",
                    a.len(),
                    b.len()
                ),
            ));
        }
        let da = DeviceBuffer::<u8>::from_host(&self.stream, &a[..512])?;
        let db = DeviceBuffer::<u8>::from_host(&self.stream, &b[..256])?;
        // Filled with a sentinel rather than zeros so a lane that fails to
        // write is distinguishable from a genuine zero result.
        let mut out = DeviceBuffer::<f32>::from_host(&self.stream, &vec![-999.0f32; 128])?;
        // SAFETY: a single warp, `a`/`b` cover every lane's fragments, and
        // each of the 128 output slots has exactly one owner.
        unsafe {
            self.gemm_f8.fp8_mma_probe(
                &self.stream,
                launch::blocks_of(1, 32),
                &da,
                &db,
                &mut out,
                fmt_a.code(),
                fmt_b.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }
}

//! GEMM backend operations.

use super::*;

impl OxideBackend {
    pub fn gemm_mode(&self) -> GemmMode {
        self.gemm_mode
    }

    /// GEMM mode for the convolution call in flight: the configured mode, or
    /// `Fp32` while the runtime runs the convolution immediately before a
    /// detection or segmentation head. Other modes (tf32/f16/bf16) are kept, so
    /// AMP storage still works; only the quantised FP8/FP4 operands are dropped.
    #[inline]
    pub(crate) fn conv_gemm_mode(&self) -> GemmMode {
        match self.gemm_mode {
            GemmMode::Fp8 | GemmMode::Fp4
                if self.conv_low_precision_off.load(Ordering::Relaxed) =>
            {
                GemmMode::Fp32
            }
            mode => mode,
        }
    }

    /// `C = alpha * op(A) * op(B) + beta * C` with `A` and `B` 16-bit buffers
    /// (`hfmt`): each is quantised straight from its 16-bit storage to `fmt_a` /
    /// `fmt_b` and the result accumulates in f32 into `c`.
    pub(crate) fn gemm_fp8_half_operands(
        &self,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<(), OxideError> {
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let a_len = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda).min(a.len());
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb).min(b.len());
        let (aq, sa) = self.fp8_quantize_half(a, a_len, fmt_a, hfmt, FP8_SITE_HALF_A)?;
        let (bq, sb) = self.fp8_quantize_half(b, b_len, fmt_b, hfmt, FP8_SITE_HALF_B)?;
        self.gemm_fp8_args(&aq, &bq, c, g, fmt_a, fmt_b, &sa, &sb)
    }

    /// Tensor-level [`Self::gemm_fp8_half_operands`]: `alpha * op(A) * op(B)`
    /// for dense 16-bit `A` (`m x k`, or `k x m` transposed) and `B`, f32
    /// result `m x n`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_half(
        &self,
        a: &OxideTensorF16,
        b: &OxideTensorF16,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        hfmt: mixed::HalfFormat,
    ) -> Result<OxideTensorF32, OxideError> {
        let mut out = self.alloc_uninit(m * n)?;
        self.gemm_fp8_half_operands(
            &a.buf,
            &b.buf,
            &mut out,
            &GemmArgs {
                m,
                n,
                k,
                lda: if trans_a { m } else { k },
                ldb: if trans_b { k } else { n },
                ldc: n,
                trans_a,
                trans_b,
                alpha,
                beta: 0.0,
                ..GemmArgs::default()
            },
            fmt_a,
            fmt_b,
            hfmt,
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    /// wgrad with the column already FP8 bytes (B) and `dy` (A) quantised once
    /// through [`Self::fp8_dy`].
    pub(crate) fn gemm_fp8_dy_a_bytes(
        &self,
        dy: &OxideTensorF32,
        fmt_a: Fp8Format,
        bq: &DeviceBuffer<u8>,
        fmt_b: Fp8Format,
        sb: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        let dq = self.fp8_dy(dy, fmt_a)?;
        self.gemm_fp8_args(&dq.bytes, bq, c, g, fmt_a, fmt_b, &dq.scale, sb)
    }

    /// `C = alpha * W * op(B) + beta * C` where `W` is a weight tensor and its
    /// quantised copy comes from [`Self::fp8_weights`]; `B` is quantised per call.
    pub(crate) fn gemm_fp8_weights_a(
        &self,
        w: &OxideTensorF32,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<(), OxideError> {
        let wq = self.fp8_weights(w, fmt_a)?;
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        let (bq, sb) = self.fp8_quantize_operand(b, b_len, fmt_b, FP8_SITE_WEIGHTS_B)?;
        self.gemm_fp8_args(&wq.bytes, &bq, c, g, fmt_a, fmt_b, &wq.scale, &sb)
    }

    /// Forward GEMM seam for a weight operand A (conv forward).  In FP8 mode the
    /// weights come from the cache; otherwise this is [`Self::gemm`].
    pub(crate) fn gemm_weights_a(
        &self,
        w: &OxideTensorF32,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        if self.conv_gemm_mode() == GemmMode::Fp8 && !self.amp {
            return self.gemm_fp8_weights_a(w, b, c, g, Fp8Format::E4M3, Fp8Format::E4M3);
        }
        self.gemm(&w.buf, b, c, g)
    }

    /// `C = alpha * A * op(B) + beta * C` with `A` a weight tensor (cached FP8
    /// copy in FP8 mode).  The tensor-level counterpart of the conv seam.
    pub fn gemm_f32_weights_a(
        &self,
        w: &OxideTensorF32,
        b: &OxideTensorF32,
        c: &OxideTensorF32,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<OxideTensorF32, OxideError> {
        let mut out = self.alloc_uninit(c.len())?;
        out.copy_from_device_async(&c.buf, &self.stream)?;
        self.gemm_weights_a(
            w,
            &b.buf,
            &mut out,
            &GemmArgs {
                m,
                n,
                k,
                lda: if trans_a { m } else { k },
                ldb: if trans_b { k } else { n },
                ldc: n,
                trans_a,
                trans_b,
                alpha,
                beta,
                ..GemmArgs::default()
            },
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    /// `C = alpha * op(A) * op(B) + beta * C` with both operands quantised to
    /// `fmt_a` / `fmt_b` per tensor (`amax / max_finite`), run on the Ada FP8
    /// tensor cores.  Accumulates in f32; the error is the operand rounding
    /// (<= 2^-4 relative for e4m3, 2^-3 for e5m2, per operand).
    pub(crate) fn gemm_f32_via_fp8_fmt(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        key_a: u64,
        key_b: u64,
    ) -> Result<(), OxideError> {
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let a_len = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda);
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        let ((aq, sa), (bq, sb)) =
            self.fp8_quantize_pair(a, a_len, fmt_a, key_a, b, b_len, fmt_b, key_b)?;
        // The mma works in the quantised domain: the epilogue multiplies by the
        // device-side reciprocal factors `sa[1] * sb[1]`, so `c` needs no pass.
        self.gemm_fp8_args(&aq, &bq, c, g, fmt_a, fmt_b, &sa, &sb)
    }

    /// e4m3 x e4m3, the forward format pair.
    pub(crate) fn gemm_f32_via_fp8(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        self.gemm_f32_via_fp8_fmt(
            a,
            b,
            c,
            g,
            Fp8Format::E4M3,
            Fp8Format::E4M3,
            FP8_SITE_FORWARD_A,
            FP8_SITE_FORWARD_B,
        )
    }

    /// `C = alpha * op(A) * op(B) + beta * C` with both operands quantised to
    /// packed NVFP4 (E2M1 codes, one E4M3 scale per 16 elements along K, plus
    /// an f32 global scale) and multiplied on the CUDA cores with `__dp4a`.
    ///
    /// This is an *emulation*: native FP4 `mma` needs sm_120a, so it is slower
    /// than fp16 by design and exists for parity with the C++ `raptor-fp4`
    /// reference.  The packed layout is only defined for operands contiguous
    /// along K, so this path transposes on the host (a full round trip) when a
    /// caller hands in an N-contiguous operand - acceptable for an emulation
    /// whose whole purpose is reference parity, not speed.
    pub(crate) fn gemm_f32_via_fp4(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_fp4: {msg}"));
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let a_len = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda);
        let b_len = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        if a_len < g.m * g.k || b_len < g.n * g.k {
            return Err(bad("operand buffer smaller than m*k / n*k"));
        }
        // Instance 0 only: the packed GEMM has no batched form yet.
        if g.count > 1 {
            return Err(bad("the packed NVFP4 path handles a single GEMM instance"));
        }
        let a_raw = a.to_host_vec(&self.stream)?;
        let b_raw = b.to_host_vec(&self.stream)?;
        // Pack A as [m][k] and B as [n][k], transposing whichever operand the
        // caller stored N-contiguous.
        let a_k: Vec<f32> = if g.trans_a {
            let mut v = vec![0.0f32; g.m * g.k];
            for p in 0..g.k {
                for i in 0..g.m {
                    v[i * g.k + p] = a_raw[g.a.0 + p * g.lda + i];
                }
            }
            v
        } else {
            a_raw[g.a.0..g.a.0 + g.m * g.k].to_vec()
        };
        let b_k: Vec<f32> = if g.trans_b {
            b_raw[g.b.0..g.b.0 + g.n * g.k].to_vec()
        } else {
            let mut v = vec![0.0f32; g.n * g.k];
            for p in 0..g.k {
                for j in 0..g.n {
                    v[j * g.k + p] = b_raw[g.b.0 + p * g.ldb + j];
                }
            }
            v
        };
        let pa = fp4::quantize_1x16(&a_k, g.m, g.k);
        let pb = fp4::quantize_1x16(&b_k, g.n, g.k);
        let c_host = if g.beta == 0.0 {
            Vec::new()
        } else {
            c.to_host_vec(&self.stream)?
        };
        let result = self.gemm_fp4(
            &pa,
            &pb,
            if g.beta == 0.0 { None } else { Some(&c_host) },
            g.m,
            g.n,
            g.k,
            g.alpha,
            g.beta,
            false,
        )?;
        c.copy_from_host(&self.stream, &result)?;
        Ok(())
    }

    /// Implicit-GEMM conv forward: same math and instance layout as [`Self::gemm`]
    /// with `B = col` materialised, but the `B` tile is generated from `x` on
    /// the fly (fp32 only; callers fall back to the materialised path for
    /// tf32/fp16 modes).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_conv_implicit(
        &self,
        a: &DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        conv: &ImplicitConv,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_conv_implicit: {msg}"));
        if g.m == 0 || g.n == 0 || g.k == 0 || g.count == 0 {
            return Err(bad("m, n, k and count must be > 0"));
        }
        if g.trans_a || g.trans_b {
            return Err(bad("only the NN layout is supported"));
        }
        if g.a.2 == 0 || g.c.2 == 0 {
            return Err(bad("instance modulus must be > 0"));
        }
        if g.lda < g.k || g.ldc < g.n {
            return Err(bad("leading dimension smaller than row length"));
        }
        if extent(g.a.0, g.a.1, g.a.2, g.count, g.m, g.k, g.lda) > a.len()
            || extent(g.c.0, g.c.1, g.c.2, g.count, g.m, g.n, g.ldc) > c.len()
        {
            return Err(bad("operand buffer too small for the requested view"));
        }
        let (bm, bn) = (conv_implicit::BM, conv_implicit::BN);
        let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn) * g.count;
        let splits = if tiles < SPLITK_TARGET_BLOCKS {
            SPLITK_TARGET_BLOCKS
                .div_ceil(tiles)
                .min(g.k / SPLITK_MIN_K)
                .min(SPLITK_MAX)
                .max(1)
        } else {
            1
        };
        let kchunk = if splits > 1 {
            g.k.div_ceil(splits).next_multiple_of(16)
        } else {
            g.k
        };
        let splits = g.k.div_ceil(kchunk); // drop empty trailing splits
        let mn = g.m * g.n;
        let mut ws = if splits > 1 {
            Some(self.alloc_uninit(splits * g.count * mn)?)
        } else {
            None
        };
        let (alpha, beta, ldc) = if splits > 1 {
            (1.0, 0.0, g.n)
        } else {
            (g.alpha, g.beta, g.ldc)
        };
        let (cb, cs, cm) = if splits > 1 {
            (0, mn, splits * g.count)
        } else {
            g.c
        };
        let target: &mut DeviceBuffer<f32> = match ws.as_mut() {
            Some(w) => w,
            None => c,
        };
        let blocks = tiles * splits;
        // SAFETY: extents validated above; the workspace is sized for every
        // split, and every C element has one owner thread.
        unsafe {
            self.conv_implicit.sgemm_conv_implicit(
                &self.stream,
                launch::blocks_of(blocks, conv_implicit::GEMM_THREADS),
                a,
                x,
                target,
                g.m,
                g.n,
                g.k,
                g.lda,
                ldc,
                alpha,
                beta,
                g.a.0,
                g.a.1,
                g.a.2,
                cb,
                cs,
                cm,
                g.count,
                kchunk,
                conv.x_base,
                conv.channels,
                conv.in_h,
                conv.in_w,
                conv.cpg,
                conv.groups,
                conv.ksize,
                conv.stride_x,
                conv.stride_y,
                conv.dilation,
                conv.padding,
                conv.out_h,
                conv.out_w,
            )
        }
        .map_err(launch_err)?;
        if let Some(w) = ws {
            // SAFETY: one thread per output element; extents validated above.
            unsafe {
                self.gemm.gemm_splitk_reduce(
                    &self.stream,
                    launch::elementwise(g.count * mn, ElementwiseLaunchConfig::default()),
                    &w,
                    c,
                    g.m,
                    g.n,
                    g.ldc,
                    g.count,
                    splits,
                    g.alpha,
                    g.beta,
                    g.c.0,
                    g.c.1,
                    g.c.2,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(())
    }

    /// Single dense GEMM on tensors, for tests and future `connected` layers.
    /// Returns a fresh `m x n` tensor; `c` (read only when `beta != 0`) is
    /// copied via the host, so this is not a hot path.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32(
        &self,
        a: &OxideTensorF32,
        b: &OxideTensorF32,
        c: &OxideTensorF32,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<OxideTensorF32, OxideError> {
        let mut out = self.alloc_uninit(c.len())?;
        out.copy_from_device_async(&c.buf, &self.stream)?;
        self.gemm(
            &a.buf,
            &b.buf,
            &mut out,
            &GemmArgs {
                m,
                n,
                k,
                lda: if trans_a { m } else { k },
                ldb: if trans_b { k } else { n },
                ldc: n,
                trans_a,
                trans_b,
                alpha,
                beta,
                ..GemmArgs::default()
            },
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    /// Validated FP8 GEMM launch (8-bit operands, f32 accumulator) with the
    /// same view, split-K and `beta` semantics as [`Self::gemm`].
    pub(crate) fn gemm_fp8_args(
        &self,
        a: &DeviceBuffer<u8>,
        b: &DeviceBuffer<u8>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
        scale_a: &DeviceBuffer<f32>,
        scale_b: &DeviceBuffer<f32>,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_fp8: {msg}"));
        if g.m == 0 || g.n == 0 || g.k == 0 || g.count == 0 {
            return Err(bad("m, n, k and count must be > 0"));
        }
        if g.a.2 == 0 || g.b.2 == 0 || g.c.2 == 0 {
            return Err(bad("instance modulus must be > 0"));
        }
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        if g.lda < ac || g.ldb < bc || g.ldc < g.n {
            return Err(bad("leading dimension smaller than row length"));
        }
        // Element extents, in bytes for the 8-bit operands.
        let a_bytes = extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda);
        let b_bytes = extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb);
        if a_bytes > a.len() || b_bytes > b.len() {
            return Err(bad("operand buffer too small for the requested view"));
        }
        if extent(g.c.0, g.c.1, g.c.2, g.count, g.m, g.n, g.ldc) > c.len() {
            return Err(bad("output buffer too small for the requested view"));
        }
        let (bm, bn) = (gemm_f8::F8_BM, gemm_f8::F8_BN);
        let tiles = g.m.div_ceil(bm) * g.n.div_ceil(bn) * g.count;
        let splits = if tiles < SPLITK_TARGET_BLOCKS {
            SPLITK_TARGET_BLOCKS
                .div_ceil(tiles)
                .min(g.k / SPLITK_MIN_K)
                .min(SPLITK_MAX)
                .max(1)
        } else {
            1
        };
        let kchunk = if splits > 1 {
            g.k.div_ceil(splits).next_multiple_of(16)
        } else {
            g.k
        };
        let splits = g.k.div_ceil(kchunk);
        let mn = g.m * g.n;
        let mut ws = if splits > 1 {
            Some(self.alloc_uninit(splits * g.count * mn)?)
        } else {
            None
        };
        let (alpha, beta, ldc) = if splits > 1 {
            (1.0, 0.0, g.n)
        } else {
            (g.alpha, g.beta, g.ldc)
        };
        let (cb, cs, cm) = if splits > 1 {
            (0, mn, splits * g.count)
        } else {
            g.c
        };
        let target: &mut DeviceBuffer<f32> = match ws.as_mut() {
            Some(w) => w,
            None => c,
        };
        let blocks = tiles * splits;
        // SAFETY: extents validated above; each C element has one owner
        // thread and instances writing the same `c` slot are not allowed.
        macro_rules! launch_f8 {
            ($kernel:ident) => {
        unsafe {
            self.gemm_f8.$kernel(
                &self.stream,
                launch::blocks_of(blocks, gemm_f8::F8_THREADS),
                a,
                b,
                target,
                g.m,
                g.n,
                g.k,
                g.lda,
                g.ldb,
                ldc,
                g.trans_a as u32,
                g.trans_b as u32,
                alpha,
                beta,
                g.a.0,
                g.a.1,
                g.a.2,
                g.b.0,
                g.b.1,
                g.b.2,
                cb,
                cs,
                cm,
                g.count,
                kchunk,
                fmt_a.code(),
                fmt_b.code(),
                scale_a,
                scale_b,
            )
        }
            };
        }
        // The e4m3 x e4m3 kernel has no runtime format `match` in its mma loop.
        let both_e4m3 = fmt_a == Fp8Format::E4M3
            && fmt_b == Fp8Format::E4M3
            && !self.fp8_generic_kernel.load(Ordering::Relaxed);
        let launched = if both_e4m3 { launch_f8!(sgemm_fp8_mma_e4m3) } else { launch_f8!(sgemm_fp8_mma) };
        launched.map_err(launch_err)?;
        if let Some(w) = ws {
            // SAFETY: one thread per output element; `c` extent validated above.
            unsafe {
                self.gemm.gemm_splitk_reduce(
                    &self.stream,
                    launch::elementwise(g.count * mn, ElementwiseLaunchConfig::default()),
                    &w,
                    c,
                    g.m,
                    g.n,
                    g.ldc,
                    g.count,
                    splits,
                    g.alpha,
                    g.beta,
                    g.c.0,
                    g.c.1,
                    g.c.2,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(())
    }

    /// Dense `C = alpha * op(A) * op(B)` (no `beta`) on FP8 operands, host
    /// round trip; `a` is `m x k` (or `k x m` when `trans_a`), `b` is `k x n`
    /// (or `n x k` when `trans_b`).  Returns a fresh `m x n` f32 tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8(
        &self,
        a: &[u8],
        b: &[u8],
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
        fmt_a: Fp8Format,
        fmt_b: Fp8Format,
    ) -> Result<OxideTensorF32, OxideError> {
        let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
        let (br, bc) = if trans_b { (n, k) } else { (k, n) };
        if a.len() < ar * ac || b.len() < br * bc {
            return Err(shape_err(
                "gemm_fp8",
                format!(
                    "operands too small: {} A bytes for {ar}x{ac}, {} B bytes for {br}x{bc}",
                    a.len(),
                    b.len()
                ),
            ));
        }
        let da = DeviceBuffer::<u8>::from_host(&self.stream, a)?;
        let db = DeviceBuffer::<u8>::from_host(&self.stream, b)?;
        let mut out = self.alloc_uninit(m * n)?;
        let unit = self.unit_scale()?;
        self.gemm_fp8_args(
            &da,
            &db,
            &mut out,
            &GemmArgs {
                m,
                n,
                k,
                lda: ac,
                ldb: bc,
                ldc: n,
                trans_a,
                trans_b,
                alpha,
                beta: 0.0,
                ..GemmArgs::default()
            },
            fmt_a,
            fmt_b,
            &unit,
            &unit,
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    /// Packed NVFP4 GEMM, CUDA cores only (`__dp4a`).
    ///
    /// `C[M, N] = alpha * gA * gB * sum_blocks dot * 0.25 * sA * sB + beta * C`
    /// with both operands "NT" (contiguous along K, which is how
    /// [`Fp4Packed`] stores them).  `reference` picks the one-thread-per-output
    /// oracle kernel instead of the tiled one.  `c` (read only when
    /// `beta != 0`) is supplied by the host.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp4(
        &self,
        a: &Fp4Packed,
        b: &Fp4Packed,
        c: Option<&[f32]>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        reference: bool,
    ) -> Result<Vec<f32>, OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("gemm_fp4: {msg}"));
        if m == 0 || n == 0 || k == 0 {
            return Err(bad("m, n and k must be > 0"));
        }
        if a.rows != m || b.rows != n || a.k != k || b.k != k {
            return Err(bad("packed operand shapes do not match m, n, k"));
        }
        if let Some(c) = c {
            if c.len() < m * n {
                return Err(bad("c is smaller than m * n"));
            }
        }
        let dca = DeviceBuffer::<u8>::from_host(&self.stream, &a.codes)?;
        let dsa = DeviceBuffer::<u8>::from_host(&self.stream, &a.scales)?;
        let dcb = DeviceBuffer::<u8>::from_host(&self.stream, &b.codes)?;
        let dsb = DeviceBuffer::<u8>::from_host(&self.stream, &b.scales)?;
        let ga = DeviceBuffer::<f32>::from_host(&self.stream, &[a.global_scale])?;
        let gb = DeviceBuffer::<f32>::from_host(&self.stream, &[b.global_scale])?;
        let mut out = DeviceBuffer::<f32>::zeroed(&self.stream, m * n)?;
        if beta != 0.0 {
            if let Some(c) = c {
                out.copy_from_host(&self.stream, c)?;
            }
        }
        let (bm, bn) = (gemm_fp4::FP4_BM, gemm_fp4::FP4_BN);
        let blocks = m.div_ceil(bm) * n.div_ceil(bn);
        let grid = if reference {
            launch::elementwise(m * n, ElementwiseLaunchConfig::default())
        } else {
            launch::blocks_of(blocks, gemm_fp4::FP4_THREADS)
        };
        // SAFETY: the packed buffers are sized `rows * ceil(k/16) * 8` and
        // `rows * ceil(k/16)` as the kernels index them; `out` is `m * n` and
        // each element has exactly one owner thread.
        unsafe {
            if reference {
                self.gemm_fp4.fp4_gemm_nt_packed_reference(
                    &self.stream,
                    grid,
                    &dca,
                    &dsa,
                    &dcb,
                    &dsb,
                    &mut out,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    n,
                    &ga,
                    &gb,
                )
            } else {
                self.gemm_fp4.fp4_gemm_nt_packed(
                    &self.stream,
                    grid,
                    &dca,
                    &dsa,
                    &dcb,
                    &dsb,
                    &mut out,
                    m,
                    n,
                    k,
                    alpha,
                    beta,
                    n,
                    &ga,
                    &gb,
                )
            }
        }
        .map_err(launch_err)?;
        Ok(out.to_host_vec(&self.stream)?)
    }

}

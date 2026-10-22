//! f16 storage inference path (P6): conversions, the f16 tensor-core GEMM
//! driver, the fused conv + bias + activation, and f16 maxpool / route /
//! upsample.  Weights and activations are binary16, accumulation, bias and the
//! activation are f32, results are rounded to nearest even on store.  Inference
//! only: there is no backward on f16 tensors (training stays f32).

use cuda_core::{CudaStream, DeviceBuffer};
use darknet_cuda::{
    ElementwiseLaunchConfig, F32ActivationKind, F32ChannelNormalization, F32TensorOpsBackend,
};
use std::sync::Arc;

use crate::kernels::gemm_f16::{HMMA_BM, HMMA_BN, HMMA_THREADS};
use crate::mixed::HalfFormat;
use crate::tensor::PooledBuf16;
use crate::{
    GemmArgs, GemmMode, OxideBackend, OxideError, OxideTensorF16, OxideTensorF32, SPLITK_MAX,
    SPLITK_MIN_K, SPLITK_TARGET_BLOCKS, activation_code, extent, launch, launch_err, pool_layout,
    shape_err,
};

/// Largest finite binary16 value.
pub const F16_MAX: f32 = 65504.0;

/// Host reference: f32 -> binary16 bits, round to nearest even; overflow gives
/// +-Inf, NaN stays NaN (quiet, payload high bits kept).
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if man != 0 {
                0x0200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut h = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// Host reference: binary16 bits -> f32 (exact).
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else if exp == 0 {
        if man == 0 {
            sign
        } else {
            // Subnormal: normalise.
            let mut e = 113u32; // 127 - 15 + 1
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Host reference: f32 -> bfloat16 bits, round to nearest even, NaN kept.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7f80_0000) == 0x7f80_0000 && (b & 0x007f_ffff) != 0 {
        return ((b >> 16) as u16) | 0x0040;
    }
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// Host reference: bfloat16 bits -> f32 (exact).
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// 1-element dummies for the output / bias operands the f16 GEMM kernel does
/// not use in a given epilogue mode.
pub(crate) struct HalfScratch {
    d32: DeviceBuffer<f32>,
    d16: DeviceBuffer<u16>,
}

impl HalfScratch {
    pub(crate) fn new(stream: &Arc<CudaStream>) -> Result<Self, OxideError> {
        Ok(Self {
            d32: DeviceBuffer::<f32>::zeroed(stream, 1)?,
            d16: DeviceBuffer::<u16>::zeroed(stream, 1)?,
        })
    }
}

/// Destination of an f16 GEMM.
pub(crate) enum HOut<'a> {
    /// f32 `C = alpha * A * B + beta * C` (views from `GemmArgs.c`).
    F32(&'a mut DeviceBuffer<f32>),
    /// Fused conv epilogue: `f16(act(alpha * A * B + bias[(inst % bias_mod) * bias_stride + row]))`
    /// stored at the `GemmArgs.c` view (`ldc` = f16 row pitch); only the first
    /// `n_store` columns of each row are stored (N padding is skipped).
    F16 {
        out: &'a mut DeviceBuffer<u16>,
        bias: &'a DeviceBuffer<f32>,
        bias_stride: usize,
        bias_mod: usize,
        act: u32,
        n_store: usize,
    },
}

fn f16_shape_len(op: &str, shape: &[usize], len: usize) -> Result<(), OxideError> {
    let expected: usize = shape.iter().product();
    if expected != len {
        return Err(OxideError::Shape(format!(
            "{op}: shape {shape:?} expects {expected} elements, got {len}"
        )));
    }
    Ok(())
}

impl OxideBackend {
    pub(crate) fn alloc16_uninit(&self, len: usize) -> Result<PooledBuf16, OxideError> {
        match self.bufs.take16(len) {
            Some(b) => Ok(PooledBuf16::new(b, &self.bufs)),
            None => Ok(PooledBuf16::new(
                DeviceBuffer::<u16>::zeroed(&self.stream, len)?,
                &self.bufs,
            )),
        }
    }

    // ---- conversions -----------------------------------------------------

    /// Uploads raw binary16 bit patterns.
    pub fn upload_f16_bits(
        &self,
        shape: &[usize],
        bits: &[u16],
    ) -> Result<OxideTensorF16, OxideError> {
        f16_shape_len("upload_f16_bits", shape, bits.len())?;
        let mut buf = self.alloc16_uninit(bits.len())?;
        if !bits.is_empty() {
            buf.copy_from_host(&self.stream, bits)?;
        }
        Ok(OxideTensorF16::new(buf, shape))
    }

    /// Downloads the raw 16-bit patterns.
    pub fn download_f16_bits(&self, t: &OxideTensorF16) -> Result<Vec<u16>, OxideError> {
        Ok(t.buf.to_host_vec(&self.stream)?)
    }

    /// Uploads f32 host data and rounds it to f16 on the device.
    pub fn upload_f16(&self, shape: &[usize], host: &[f32]) -> Result<OxideTensorF16, OxideError> {
        f16_shape_len("upload_f16", shape, host.len())?;
        let t = OxideTensorF32::new(self.alloc_from_host(host)?, shape);
        self.f32_to_f16(&t, false)
    }

    /// Downloads an f16 tensor widened to f32.
    pub fn download_f16(&self, t: &OxideTensorF16) -> Result<Vec<f32>, OxideError> {
        let wide = self.f16_to_f32(t)?;
        Ok(wide.buf.to_host_vec(&self.stream)?)
    }

    fn alloc_from_host(&self, host: &[f32]) -> Result<crate::tensor::PooledBuf, OxideError> {
        let mut buf = self.alloc_uninit(host.len())?;
        if !host.is_empty() {
            buf.copy_from_host(&self.stream, host)?;
        }
        Ok(buf)
    }

    /// f32 -> f16 on the device (round to nearest even; overflow -> +-Inf, or
    /// +-65504 with `saturate`; NaN stays NaN).
    pub fn f32_to_f16(
        &self,
        x: &OxideTensorF32,
        saturate: bool,
    ) -> Result<OxideTensorF16, OxideError> {
        let mut out = self.alloc16_uninit(x.len())?;
        if x.len() > 0 {
            // SAFETY: one thread per element.
            unsafe {
                self.convert.cvt_f32_to_f16(
                    &self.stream,
                    launch::elementwise(x.len(), ElementwiseLaunchConfig::default()),
                    &x.buf,
                    &mut out,
                    saturate as u32,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF16::new(out, x.shape()))
    }

    /// f16 -> f32 on the device (exact).
    pub fn f16_to_f32(&self, x: &OxideTensorF16) -> Result<OxideTensorF32, OxideError> {
        let mut out = self.alloc_uninit(x.len())?;
        if x.len() > 0 {
            // SAFETY: one thread per element.
            unsafe {
                self.convert.cvt_f16_to_f32(
                    &self.stream,
                    launch::elementwise(x.len(), ElementwiseLaunchConfig::default()),
                    &x.buf,
                    &mut out,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    /// f32 -> bfloat16 bit patterns (carried in an [`OxideTensorF16`]).
    pub fn f32_to_bf16(&self, x: &OxideTensorF32) -> Result<OxideTensorF16, OxideError> {
        let mut out = self.alloc16_uninit(x.len())?;
        if x.len() > 0 {
            // SAFETY: one thread per element.
            unsafe {
                self.convert.cvt_f32_to_bf16(
                    &self.stream,
                    launch::elementwise(x.len(), ElementwiseLaunchConfig::default()),
                    &x.buf,
                    &mut out,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF16::new(out, x.shape()))
    }

    /// bfloat16 bit patterns -> f32 (exact).
    pub fn bf16_to_f32(&self, x: &OxideTensorF16) -> Result<OxideTensorF32, OxideError> {
        let mut out = self.alloc_uninit(x.len())?;
        if x.len() > 0 {
            // SAFETY: one thread per element.
            unsafe {
                self.convert.cvt_bf16_to_f32(
                    &self.stream,
                    launch::elementwise(x.len(), ElementwiseLaunchConfig::default()),
                    &x.buf,
                    &mut out,
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    /// f32 -> 16-bit storage in `fmt` (round to nearest even; f16 overflow
    /// becomes +-Inf, bf16 stays finite up to the f32 range; NaN stays NaN).
    pub fn to_half(&self, x: &OxideTensorF32, fmt: HalfFormat) -> Result<OxideTensorF16, OxideError> {
        let mut out = self.alloc16_uninit(x.len())?;
        if x.len() > 0 {
            // SAFETY: one thread per element.
            unsafe {
                self.convert.cvt_f32_to_half(
                    &self.stream,
                    launch::elementwise(x.len(), ElementwiseLaunchConfig::default()),
                    &x.buf,
                    &mut out,
                    fmt.code(),
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF16::new(out, x.shape()))
    }

    /// 16-bit storage in `fmt` -> f32 (exact).
    pub fn from_half(&self, x: &OxideTensorF16, fmt: HalfFormat) -> Result<OxideTensorF32, OxideError> {
        let mut out = self.alloc_uninit(x.len())?;
        if x.len() > 0 {
            // SAFETY: one thread per element.
            unsafe {
                self.convert.cvt_half_to_f32(
                    &self.stream,
                    launch::elementwise(x.len(), ElementwiseLaunchConfig::default()),
                    &x.buf,
                    &mut out,
                    fmt.code(),
                )
            }
            .map_err(launch_err)?;
        }
        Ok(OxideTensorF32::new(out, x.shape()))
    }

    /// Converts `[filters, per_filter]` f32 conv weights to the f16 GEMM
    /// operand layout `[filters, kpad]`, `kpad = per_filter` rounded up to a
    /// multiple of 8 (zero padded), so the A rows are 16-byte aligned.
    pub fn conv_weights_f16(
        &self,
        w: &OxideTensorF32,
        filters: usize,
        per_filter: usize,
    ) -> Result<OxideTensorF16, OxideError> {
        self.conv_weights_half(w, filters, per_filter, HalfFormat::F16)
    }

    /// [`Self::conv_weights_f16`] in the 16-bit format `fmt`.
    pub fn conv_weights_half(
        &self,
        w: &OxideTensorF32,
        filters: usize,
        per_filter: usize,
        fmt: HalfFormat,
    ) -> Result<OxideTensorF16, OxideError> {
        if filters == 0 || per_filter == 0 || w.len() != filters * per_filter {
            return Err(shape_err(
                "conv_weights_f16",
                format!("{} weights for {filters} filters x {per_filter}", w.len()),
            ));
        }
        let kpad = per_filter.next_multiple_of(8);
        let mut out = self.alloc16_uninit(filters * kpad)?;
        // SAFETY: one thread per output element.
        unsafe {
            self.convert.cvt_f32_to_f16_rows(
                &self.stream,
                launch::elementwise(filters * kpad, ElementwiseLaunchConfig::default()),
                &w.buf,
                &mut out,
                per_filter,
                kpad,
                0,
                fmt.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, &[filters, kpad]))
    }

    /// Inference-only f16 kernels read and write binary16; in bf16 mode they
    /// would reinterpret bf16 bits, so they refuse instead.
    pub(crate) fn require_f16(&self, op: &str) -> Result<(), OxideError> {
        if self.half_fmt == HalfFormat::Bf16 {
            return Err(OxideError::Shape(format!(
                "{op}: f16-only inference op, not available in bf16 mode"
            )));
        }
        Ok(())
    }

    // ---- GEMM ------------------------------------------------------------

    /// Dense GEMM on f16 operands with f32 accumulation and f32 output
    /// (`alpha * op(A) op(B) + beta * C`, `c` copied via the host, not a hot
    /// path).  Row-major, every transpose combination, any `m`, `n`, `k`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16(
        &self,
        a: &OxideTensorF16,
        b: &OxideTensorF16,
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
        out.copy_from_host(&self.stream, &c.buf.to_host_vec(&self.stream)?)?;
        self.hgemm(
            &a.buf,
            &b.buf,
            HOut::F32(&mut out),
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

    /// [`Self::gemm_f16`] for bfloat16 operands (bit patterns from
    /// [`Self::f32_to_bf16`]): `mma.sync m16n8k16 .bf16` with f32 accumulation
    /// and f32 output.  8 significant bits instead of 11, f32 exponent range.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16(
        &self,
        a: &OxideTensorF16,
        b: &OxideTensorF16,
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
        out.copy_from_host(&self.stream, &c.buf.to_host_vec(&self.stream)?)?;
        self.hgemm_ex(
            &a.buf,
            &b.buf,
            HOut::F32(&mut out),
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
            true,
        )?;
        Ok(OxideTensorF32::new(out, &[m, n]))
    }

    /// [`Self::gemm_bf16`] with a bfloat16 *output*: `bf16(alpha * A * B)` from
    /// the fused-epilogue path (zero bias, linear activation, `beta = 0`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16_out16(
        &self,
        a: &OxideTensorF16,
        b: &OxideTensorF16,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        alpha: f32,
    ) -> Result<OxideTensorF16, OxideError> {
        let mut out = self.alloc16_uninit(m * n)?;
        let bias = self.alloc(m)?;
        self.hgemm_ex(
            &a.buf,
            &b.buf,
            HOut::F16 {
                out: &mut out,
                bias: &bias,
                bias_stride: 0,
                bias_mod: 1,
                act: 0,
                n_store: n,
            },
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
                ..GemmArgs::default()
            },
            true,
        )?;
        Ok(OxideTensorF16::new(out, &[m, n]))
    }

    /// `GemmMode::Fp16` / `GemmMode::Bf16` in the f32 API: round both operands
    /// to the 16-bit format on the device (same layout, so leading dims and
    /// views carry over) and run the 16-bit GEMM.  Compute mode only; storage
    /// stays f32.
    pub(crate) fn gemm_f32_operands_as_half(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        g: &GemmArgs,
        bf16: bool,
    ) -> Result<(), OxideError> {
        let cfg = ElementwiseLaunchConfig::default();
        let mut ah = self.alloc16_uninit(a.len())?;
        let mut bh = self.alloc16_uninit(b.len())?;
        let round = |x: &DeviceBuffer<f32>, out: &mut DeviceBuffer<u16>| {
            // SAFETY: one thread per element of the operand.
            unsafe {
                if bf16 {
                    self.convert
                        .cvt_f32_to_bf16(&self.stream, launch::elementwise(x.len(), cfg), x, out)
                } else {
                    self.convert
                        .cvt_f32_to_f16(&self.stream, launch::elementwise(x.len(), cfg), x, out, 0)
                }
            }
            .map_err(launch_err)
        };
        round(a, &mut ah)?;
        round(b, &mut bh)?;
        self.hgemm_ex(&ah, &bh, HOut::F32(c), g, bf16)
    }

    /// Validated f16 GEMM launch (the f16 twin of [`OxideBackend::gemm`],
    /// including deterministic split-K for under-filled grids).
    pub(crate) fn hgemm(
        &self,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        out: HOut,
        g: &GemmArgs,
    ) -> Result<(), OxideError> {
        self.hgemm_ex(a, b, out, g, self.half_fmt == HalfFormat::Bf16)
    }

    /// [`Self::hgemm`] with the operand format selectable: `bf16` reads the
    /// operands as bfloat16 (f32 output only).
    pub(crate) fn hgemm_ex(
        &self,
        a: &DeviceBuffer<u16>,
        b: &DeviceBuffer<u16>,
        out: HOut,
        g: &GemmArgs,
        bf16: bool,
    ) -> Result<(), OxideError> {
        let bad = |msg: &str| OxideError::Shape(format!("hgemm: {msg}"));
        if g.m == 0 || g.n == 0 || g.k == 0 || g.count == 0 {
            return Err(bad("m, n, k and count must be > 0"));
        }
        if g.a.2 == 0 || g.b.2 == 0 || g.c.2 == 0 {
            return Err(bad("instance modulus must be > 0"));
        }
        let (ar, ac) = if g.trans_a { (g.k, g.m) } else { (g.m, g.k) };
        let (br, bc) = if g.trans_b { (g.n, g.k) } else { (g.k, g.n) };
        let n_store = match &out {
            HOut::F32(_) => g.n,
            HOut::F16 { n_store, .. } => *n_store,
        };
        if g.lda < ac || g.ldb < bc || g.ldc < n_store {
            return Err(bad("leading dimension smaller than row length"));
        }
        if n_store == 0 || n_store > g.n {
            return Err(bad("n_store must be in 1..=n"));
        }
        let c_len = match &out {
            HOut::F32(c) => c.len(),
            HOut::F16 { out, .. } => out.len(),
        };
        if extent(g.a.0, g.a.1, g.a.2, g.count, ar, ac, g.lda) > a.len()
            || extent(g.b.0, g.b.1, g.b.2, g.count, br, bc, g.ldb) > b.len()
            || extent(g.c.0, g.c.1, g.c.2, g.count, g.m, n_store, g.ldc) > c_len
        {
            return Err(bad("operand buffer too small for the requested view"));
        }
        if let HOut::F16 {
            bias,
            bias_stride,
            bias_mod,
            ..
        } = &out
        {
            let need = (g.count.min(*bias_mod) - 1) * bias_stride + g.m;
            if bias.len() < need {
                return Err(bad("bias buffer too small"));
            }
            if g.beta != 0.0 {
                return Err(bad("the fused f16 epilogue requires beta == 0"));
            }
        }

        let tiles = g.m.div_ceil(HMMA_BM) * g.n.div_ceil(HMMA_BN) * g.count;
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
            g.k.div_ceil(splits).next_multiple_of(32)
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
        let blocks = tiles * splits;
        let cfg = launch::blocks_of(blocks, HMMA_THREADS);

        let mut guard = self
            .half_scratch
            .lock()
            .map_err(|_| OxideError::Shape("f16 scratch lock poisoned".into()))?;
        let sc = &mut *guard;
        let (alpha, beta, ldc, cv) = if splits > 1 {
            (1.0, 0.0, g.n, (0usize, mn, splits * g.count))
        } else {
            (g.alpha, g.beta, g.ldc, g.c)
        };
        let (m, n, k) = (g.m, g.n, g.k);
        let (ta, tb) = (g.trans_a as u32, g.trans_b as u32);
        let bf = bf16 as u32;
        let (av, bv) = (g.a, g.b);
        // SAFETY (all launches below): extents validated above (the workspace
        // holds every split); each C element has exactly one owner thread;
        // operands unused by the selected epilogue are 1-element dummies that
        // it never touches.
        match out {
            HOut::F32(c) => {
                let r = unsafe {
                    match ws.as_mut() {
                        Some(w) => self.gemm_f16.hgemm_f16_mma(
                            &self.stream,
                            cfg,
                            a,
                            b,
                            &sc.d32,
                            w,
                            &mut sc.d16,
                            m,
                            n,
                            k,
                            n,
                            g.lda,
                            g.ldb,
                            ldc,
                            ta,
                            tb,
                            alpha,
                            beta,
                            av.0,
                            av.1,
                            av.2,
                            bv.0,
                            bv.1,
                            bv.2,
                            cv.0,
                            cv.1,
                            cv.2,
                            g.count,
                            kchunk,
                            0,
                            0,
                            0,
                            1,
                            bf,
                        ),
                        None => self.gemm_f16.hgemm_f16_mma(
                            &self.stream,
                            cfg,
                            a,
                            b,
                            &sc.d32,
                            c,
                            &mut sc.d16,
                            m,
                            n,
                            k,
                            n,
                            g.lda,
                            g.ldb,
                            ldc,
                            ta,
                            tb,
                            alpha,
                            beta,
                            av.0,
                            av.1,
                            av.2,
                            bv.0,
                            bv.1,
                            bv.2,
                            cv.0,
                            cv.1,
                            cv.2,
                            g.count,
                            kchunk,
                            0,
                            0,
                            0,
                            1,
                            bf,
                        ),
                    }
                };
                r.map_err(launch_err)?;
                if let Some(w) = ws {
                    // SAFETY: one thread per output element.
                    unsafe {
                        self.gemm.gemm_splitk_reduce(
                            &self.stream,
                            launch::elementwise(g.count * mn, ElementwiseLaunchConfig::default()),
                            &w,
                            c,
                            m,
                            n,
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
            }
            HOut::F16 {
                out: o,
                bias,
                bias_stride,
                bias_mod,
                act,
                n_store,
            } => {
                let r = unsafe {
                    match ws.as_mut() {
                        Some(w) => self.gemm_f16.hgemm_f16_mma(
                            &self.stream,
                            cfg,
                            a,
                            b,
                            &sc.d32,
                            w,
                            &mut sc.d16,
                            m,
                            n,
                            k,
                            n,
                            g.lda,
                            g.ldb,
                            ldc,
                            ta,
                            tb,
                            alpha,
                            beta,
                            av.0,
                            av.1,
                            av.2,
                            bv.0,
                            bv.1,
                            bv.2,
                            cv.0,
                            cv.1,
                            cv.2,
                            g.count,
                            kchunk,
                            0,
                            0,
                            0,
                            1,
                            bf,
                        ),
                        None => self.gemm_f16.hgemm_f16_mma(
                            &self.stream,
                            cfg,
                            a,
                            b,
                            bias,
                            &mut sc.d32,
                            o,
                            m,
                            n,
                            k,
                            n_store,
                            g.lda,
                            g.ldb,
                            ldc,
                            ta,
                            tb,
                            alpha,
                            beta,
                            av.0,
                            av.1,
                            av.2,
                            bv.0,
                            bv.1,
                            bv.2,
                            cv.0,
                            cv.1,
                            cv.2,
                            g.count,
                            kchunk,
                            1,
                            act,
                            bias_stride,
                            bias_mod,
                            bf,
                        ),
                    }
                };
                r.map_err(launch_err)?;
                if let Some(w) = ws {
                    // SAFETY: one thread per output element.
                    unsafe {
                        self.gemm_f16.hgemm_splitk_reduce_act16(
                            &self.stream,
                            launch::elementwise(g.count * mn, ElementwiseLaunchConfig::default()),
                            &w,
                            bias,
                            o,
                            m,
                            n,
                            n_store,
                            g.ldc,
                            g.count,
                            splits,
                            g.alpha,
                            g.c.0,
                            g.c.1,
                            g.c.2,
                            act,
                            bias_stride,
                            bias_mod,
                            bf,
                        )
                    }
                    .map_err(launch_err)?;
                }
            }
        }
        Ok(())
    }
}

impl OxideBackend {
    /// Fused f16 convolution: `f16(act(conv(x, w) + bias))`, f32 accumulation.
    ///
    /// `x` is NCHW f16, `weights` is `[filters, kpad]` from
    /// [`OxideBackend::conv_weights_f16`] (`kpad = (c / groups) * k * k`
    /// rounded up to 8), `bias` is f32 `[filters]` (folded batch norm bias).
    /// Geometry follows the f32 convs: `out = (in + 2 * pad - k) / stride + 1`.
    /// Like the f32 path, dilation does not shrink the extent; taps outside the
    /// image are zero-filled, matching `im2col_cpu_ext`.
    /// im2col (zero padded to K and N multiples of 8) and the GEMM run once per
    /// chunk of images (instances = image x group, bounded by `COL_CAP_BYTES`), so every operand row is 16-byte aligned
    /// and `cp.async` always takes the vector path; 1x1 stride-1 convs whose
    /// K and H*W are multiples of 8 skip im2col.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_bias_act_f16(
        &self,
        x: &OxideTensorF16,
        weights: &OxideTensorF16,
        bias: &OxideTensorF32,
        output_shape: &[usize],
        c: usize,
        h: usize,
        w: usize,
        f: usize,
        k: usize,
        sx: usize,
        sy: usize,
        dil: usize,
        pad: usize,
        groups: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        const OP: &str = "conv2d_bias_act_f16";
        let bad = |msg: String| OxideError::Shape(format!("{OP}: {msg}"));
        if c == 0
            || h == 0
            || w == 0
            || f == 0
            || k == 0
            || sx == 0
            || sy == 0
            || dil == 0
            || groups == 0
        {
            return Err(bad(
                "non-zero channels, size, filters, kernel, stride, dilation and groups required"
                    .into(),
            ));
        }
        if c % groups != 0 || f % groups != 0 {
            return Err(bad(format!(
                "channels {c} and filters {f} must be divisible by groups {groups}"
            )));
        }
        if h + 2 * pad < k || w + 2 * pad < k {
            return Err(bad(format!("kernel {k} exceeds padded input")));
        }
        let oh = (h + 2 * pad - k) / sy + 1;
        let ow = (w + 2 * pad - k) / sx + 1;
        let plane = c * h * w;
        if x.len() % plane != 0 {
            return Err(bad(format!(
                "input length {} is not a multiple of C*H*W = {plane}",
                x.len()
            )));
        }
        let batch = x.len() / plane;
        if output_shape != [batch, f, oh, ow] {
            return Err(bad(format!(
                "output shape must be {:?}, got {output_shape:?}",
                [batch, f, oh, ow]
            )));
        }
        let (cpg, fpg) = (c / groups, f / groups);
        let kg = cpg * k * k;
        let kpad = kg.next_multiple_of(8);
        if weights.len() != f * kpad {
            return Err(bad(format!(
                "weights hold {} halves, expected [filters, kpad] = {f} x {kpad} (see conv_weights_f16)",
                weights.len()
            )));
        }
        if bias.len() != f {
            return Err(bad(format!("bias length {} != filters {f}", bias.len())));
        }
        let ohw = oh * ow;
        let npad = ohw.next_multiple_of(8);
        let act = activation_code(activation);
        let mut out = self.alloc16_uninit(batch * f * ohw)?;
        let direct = k == 1 && sx == 1 && sy == 1 && pad == 0 && kg % 8 == 0 && ohw % 8 == 0;
        let mut col_guard = self
            .half_col
            .lock()
            .map_err(|_| OxideError::Shape("f16 conv workspace lock poisoned".into()))?;
        // Images per im2col + GEMM round: the column buffer holds `chunk`
        // images (bounded by `COL_CAP_BYTES`), so a whole chunk is one im2col
        // launch and one GEMM launch (instances = image x group).
        let img_col = groups * kpad * npad;
        let chunk = if direct {
            batch
        } else {
            (crate::conv::col_cap_bytes() / 2 / img_col).clamp(1, batch)
        }
        .min(crate::conv::chunk_limit());
        if !direct {
            let need = chunk * img_col;
            if col_guard.as_ref().is_none_or(|b| b.len() < need) {
                *col_guard = Some(DeviceBuffer::<u16>::zeroed(&self.stream, need)?);
            }
        }
        let mut b0 = 0;
        while b0 < batch {
            let nb = chunk.min(batch - b0);
            let (b_buf, b_base, b_stride, ldb, n) = if direct {
                (
                    &*x.buf as &DeviceBuffer<u16>,
                    b0 * plane,
                    kg * ohw,
                    ohw,
                    ohw,
                )
            } else {
                let col = col_guard.as_mut().expect("workspace allocated above");
                // SAFETY: one thread per column element; `col` holds
                // `chunk * groups * kpad * npad` halves and `x` covers images
                // `b0..b0 + nb`.
                unsafe {
                    self.half_ops.im2col_f16(
                        &self.stream,
                        launch::elementwise(nb * img_col, config),
                        &x.buf,
                        &mut *col,
                        b0 * plane,
                        h,
                        w,
                        k,
                        sx,
                        sy,
                        dil,
                        pad,
                        oh,
                        ow,
                        kg,
                        kpad,
                        npad,
                        groups,
                        nb,
                        plane,
                    )
                }
                .map_err(launch_err)?;
                (&*col as &DeviceBuffer<u16>, 0, kpad * npad, npad, npad)
            };
            let inst = nb * groups;
            self.hgemm(
                &weights.buf,
                b_buf,
                HOut::F16 {
                    out: &mut out,
                    bias: &bias.buf,
                    bias_stride: fpg,
                    bias_mod: groups,
                    act,
                    n_store: ohw,
                },
                &GemmArgs {
                    m: fpg,
                    n,
                    k: kpad,
                    lda: kpad,
                    ldb,
                    ldc: ohw,
                    a: (0, fpg * kpad, groups),
                    b: (b_base, b_stride, inst),
                    c: (b0 * f * ohw, fpg * ohw, inst),
                    count: inst,
                    ..GemmArgs::default()
                },
            )?;
            b0 += nb;
        }
        Ok(OxideTensorF16::new(out, &[batch, f, oh, ow]))
    }

    /// f16 depth max pooling over strided channel groups, preserving the first
    /// channel on ties.
    pub fn maxpool_depth_f16(
        &self,
        x: &OxideTensorF16,
        output_shape: &[usize],
        input_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("maxpool_depth_f16")?;
        if x.shape().len() != 4
            || output_shape.len() != 4
            || input_channels == 0
            || output_shape[1] == 0
            || output_shape[2] == 0
            || output_shape[3] == 0
            || output_shape[1] > input_channels
            || x.shape()[1] != input_channels
            || x.shape()[0] != output_shape[0]
            || x.shape()[2] != output_shape[2]
            || x.shape()[3] != output_shape[3]
        {
            return Err(shape_err(
                "maxpool_depth_f16",
                "incompatible NCHW shapes".to_owned(),
            ));
        }
        let spatial = output_shape[2] * output_shape[3];
        let mut output = self.alloc16_uninit(output_shape.iter().product())?;
        // SAFETY: validated nonzero channel/spatial dimensions and matching
        // batch/spatial extents; one thread writes each output exactly once.
        unsafe {
            self.half_ops.maxpool_depth_f16(
                &self.stream,
                launch::elementwise(output.len(), config),
                &x.buf,
                &mut output,
                input_channels,
                output_shape[1],
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, output_shape))
    }

    /// Dense `[batch, inputs] x [outputs, inputs]^T` f16 inference.
    pub fn connected_f16(
        &self,
        input: &OxideTensorF16,
        weights: &[f32],
        biases: &[f32],
        input_shape: &[usize],
        output_shape: &[usize],
        outputs: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("connected_f16")?;
        if input_shape.len() != 4 || output_shape.len() != 4 || input.shape() != input_shape {
            return Err(shape_err(
                "connected_f16",
                format!("expected NCHW input/output, got {input_shape:?} -> {output_shape:?}"),
            ));
        }
        let batch = input_shape[0];
        let inputs = input_shape[1]
            .checked_mul(input_shape[2])
            .and_then(|v| v.checked_mul(input_shape[3]))
            .ok_or_else(|| {
                shape_err("connected_f16", "input feature count overflows".to_owned())
            })?;
        let expected_weights = inputs
            .checked_mul(outputs)
            .ok_or_else(|| shape_err("connected_f16", "weight count overflows".to_owned()))?;
        if outputs == 0
            || batch == 0
            || inputs == 0
            || input.len() != batch * inputs
            || weights.len() != expected_weights
            || biases.len() != outputs
            || output_shape != [batch, outputs, 1, 1]
        {
            return Err(shape_err(
                "connected_f16",
                format!(
                    "incompatible dense dimensions: input={input_shape:?}, output={output_shape:?}, weights={}, biases={}",
                    weights.len(),
                    biases.len()
                ),
            ));
        }
        let mut flat_input = input.clone();
        flat_input.shape = vec![batch, inputs, 1, 1];
        let weights_f32 = self.upload_f32(&[outputs, inputs], weights)?;
        let weights_f16 = self.conv_weights_f16(&weights_f32, outputs, inputs)?;
        let bias = self.upload_f32(&[outputs], biases)?;
        self.conv2d_bias_act_f16(
            &flat_input,
            &weights_f16,
            &bias,
            output_shape,
            inputs,
            1,
            1,
            outputs,
            1,
            1,
            1,
            1,
            0,
            1,
            activation,
            config,
        )
    }

    /// f16 max pool (same window rule as `maxpool_f32`, `off = padding / 2`).
    #[allow(clippy::too_many_arguments)]
    pub fn maxpool_f16(
        &self,
        x: &OxideTensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        let (_, out_h, out_w) = pool_layout(
            "maxpool_f16",
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let mut out = self.alloc16_uninit(output_shape.iter().product())?;
        // SAFETY: one thread per output element; reads are bounds-checked.
        unsafe {
            self.half_ops.maxpool_f16(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
                self.half_fmt.code(),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, output_shape))
    }

    /// f16 nearest upsample; `reverse` sums the `stride x stride` blocks
    /// (floor or ceil sized outputs, like `upsample_f32`).
    #[allow(clippy::too_many_arguments)]
    pub fn upsample_f16(
        &self,
        x: &OxideTensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        let op = "upsample_f16";
        if stride == 0 || channels == 0 || input_height == 0 || input_width == 0 {
            return Err(shape_err(
                op,
                "channels, spatial dims and stride must be > 0".to_owned(),
            ));
        }
        if output_shape.len() != 4 || output_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} is not [batch, {channels}, h, w]"),
            ));
        }
        let (batch, out_h, out_w) = (output_shape[0], output_shape[2], output_shape[3]);
        if x.len() != batch * channels * input_height * input_width {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {batch}*{channels}*{input_height}*{input_width}",
                    x.len()
                ),
            ));
        }
        if reverse {
            if out_h == 0
                || out_w == 0
                || out_h > input_height.div_ceil(stride)
                || out_w > input_width.div_ceil(stride)
                || out_h < input_height / stride
                || out_w < input_width / stride
            {
                return Err(shape_err(
                    op,
                    format!(
                        "reverse output {out_h}x{out_w} incompatible with input {input_height}x{input_width} / stride {stride}"
                    ),
                ));
            }
        } else if out_h != input_height * stride || out_w != input_width * stride {
            return Err(shape_err(
                op,
                format!(
                    "output {out_h}x{out_w} != input {input_height}x{input_width} * stride {stride}"
                ),
            ));
        }
        let mut out = self.alloc16_uninit(output_shape.iter().product())?;
        // SAFETY: one thread per output element; input reads are in range by
        // the shape checks above (reverse additionally bounds-checks taps).
        unsafe {
            if reverse {
                self.require_f16("upsample_f16 reverse")?;
                self.half_ops.upsample_reverse_f16(
                    &self.stream,
                    launch::elementwise(out.len(), config),
                    &x.buf,
                    &mut out,
                    input_height,
                    input_width,
                    out_h,
                    out_w,
                    stride,
                    scale,
                )
            } else {
                self.half_ops.upsample_nearest_f16(
                    &self.stream,
                    launch::elementwise(out.len(), config),
                    &x.buf,
                    &mut out,
                    input_height,
                    input_width,
                    out_h,
                    out_w,
                    stride,
                    scale,
                    self.half_fmt.code(),
                )
            }
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, output_shape))
    }

    /// f16 local average pool (mean over the in-image taps, `off = padding / 2`).
    #[allow(clippy::too_many_arguments)]
    pub fn local_avgpool_f16(
        &self,
        x: &OxideTensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("local_avgpool_f16")?;
        let (_, out_h, out_w) = pool_layout(
            "local_avgpool_f16",
            x.len(),
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let mut out = self.alloc16_uninit(output_shape.iter().product())?;
        // SAFETY: one thread per output element; reads are bounds-checked.
        unsafe {
            self.half_ops.local_avgpool_f16(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, output_shape))
    }

    /// f16 global average pool: `[batch, channels, h, w] -> [batch, channels, 1, 1]`.
    pub fn avgpool_f16(
        &self,
        x: &OxideTensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("avgpool_f16")?;
        let op = "avgpool_f16";
        let area = input_height * input_width;
        if output_shape.len() != 4
            || output_shape[1] != channels
            || output_shape[2] != 1
            || output_shape[3] != 1
        {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} is not [batch, {channels}, 1, 1]"),
            ));
        }
        let batch = output_shape[0];
        if area == 0 || x.len() != batch * channels * area {
            return Err(shape_err(
                op,
                format!(
                    "input has {} elements, expected {batch}*{channels}*{area}",
                    x.len()
                ),
            ));
        }
        let mut out = self.alloc16_uninit(batch * channels)?;
        // SAFETY: one thread per (batch, channel) plane, reading `area` elements.
        unsafe {
            self.half_ops.avgpool_f16(
                &self.stream,
                launch::elementwise(out.len(), config),
                &x.buf,
                &mut out,
                area,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, output_shape))
    }

    /// Darknet's mismatched shortcut source: flat per-sample crop / zero-pad
    /// to `output_shape` (`batch` samples), a bit copy.
    pub fn shortcut_fit_f16(
        &self,
        x: &OxideTensorF16,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("shortcut_fit_f16")?;
        let out_len: usize = output_shape.iter().product();
        if batch == 0
            || x.len() % batch != 0
            || out_len % batch != 0
            || out_len == 0
            || x.len() == 0
        {
            return Err(shape_err(
                "shortcut_fit_f16",
                format!(
                    "batch {batch}, {} source elements and output shape {output_shape:?} are inconsistent",
                    x.len()
                ),
            ));
        }
        let mut out = self.alloc16_uninit(out_len)?;
        // SAFETY: one thread per output element; reads stay inside `x`.
        unsafe {
            self.half_ops.shortcut_fit_f16(
                &self.stream,
                launch::elementwise(out_len, config),
                &x.buf,
                &mut out,
                x.len() / batch,
                out_len / batch,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, output_shape))
    }

    /// `f16(act(a + b))` for same-shaped f16 tensors (shortcut layer).
    pub fn shortcut_add_act_f16(
        &self,
        a: &OxideTensorF16,
        b: &OxideTensorF16,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("shortcut_add_act_f16")?;
        if a.len() != b.len() {
            return Err(shape_err(
                "shortcut_add_act_f16",
                format!("lengths {} and {} differ", a.len(), b.len()),
            ));
        }
        let mut out = self.alloc16_uninit(a.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.half_ops.shortcut_add_act_f16(
                &self.stream,
                launch::elementwise(a.len(), config),
                &a.buf,
                &b.buf,
                &mut out,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, a.shape()))
    }

    /// Native f16-storage channel normalization with f32 reductions.
    pub fn normalize_channels_f16(
        &self,
        input: &OxideTensorF16,
        shape: &[usize],
        normalization: F32ChannelNormalization,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("normalize_channels_f16")?;
        if shape.len() != 4
            || input.shape() != shape
            || input.len() != shape.iter().product::<usize>()
            || shape[1] == 0
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                "normalize_channels_f16",
                format!("expected nonempty NCHW input, got {shape:?}"),
            ));
        }
        let channels = shape[1];
        let spatial = shape[2] * shape[3];
        let code = match normalization {
            F32ChannelNormalization::Relu => 0,
            F32ChannelNormalization::Softmax => 1,
            F32ChannelNormalization::SoftmaxMaxval => 2,
        };
        let mut output = self.alloc16_uninit(input.len())?;
        // SAFETY: the shape validates NCHW strides and each output index is written once.
        unsafe {
            self.half_ops.normalize_channels_f16(
                &self.stream,
                launch::elementwise(output.len(), config),
                &input.buf,
                &mut output,
                channels,
                spatial,
                code,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, shape))
    }

    /// Native f16 scale_channels with per-channel or per-pixel scaling.
    pub fn scale_channels_f16(
        &self,
        input: &OxideTensorF16,
        source: &OxideTensorF16,
        input_shape: &[usize],
        output_shape: &[usize],
        scale_wh: bool,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("scale_channels_f16")?;
        let valid_shapes = input_shape.len() == 4
            && output_shape.len() == 4
            && input.shape() == input_shape
            && source.shape() == output_shape
            && input.len() == input_shape.iter().product::<usize>()
            && source.len() == output_shape.iter().product::<usize>()
            && input_shape[0] == output_shape[0]
            && output_shape[1] > 0
            && output_shape[2] > 0
            && output_shape[3] > 0;
        let valid_scale = if scale_wh {
            input_shape[1] == 1
                && input_shape[2] == output_shape[2]
                && input_shape[3] == output_shape[3]
        } else {
            input_shape[1] == output_shape[1] && input_shape[2] > 0 && input_shape[3] > 0
        };
        if !valid_shapes || !valid_scale {
            return Err(shape_err(
                "scale_channels_f16",
                format!(
                    "incompatible input/output shapes {input_shape:?} -> {output_shape:?}, scale_wh={scale_wh}"
                ),
            ));
        }
        let input_spatial = input_shape[2] * input_shape[3];
        let output_spatial = output_shape[2] * output_shape[3];
        let mut output = self.alloc16_uninit(source.len())?;
        // SAFETY: shape checks ensure the chosen per-channel/per-pixel scale
        // index is valid; one thread writes each output element exactly once.
        unsafe {
            self.half_ops.scale_channels_f16(
                &self.stream,
                launch::elementwise(output.len(), config),
                &input.buf,
                &source.buf,
                &mut output,
                input_shape[1],
                output_shape[1],
                input_spatial,
                output_spatial,
                scale_wh as u32,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, output_shape))
    }

    /// f16 SAM product and activation with exact shape checks.
    pub fn sam_f16(
        &self,
        input: &OxideTensorF16,
        source: &OxideTensorF16,
        output_shape: &[usize],
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("sam_f16")?;
        if input.shape() != output_shape || source.shape() != output_shape {
            return Err(shape_err(
                "sam_f16",
                "input, source and output shapes must match".to_owned(),
            ));
        }
        let mut output = self.alloc16_uninit(input.len())?;
        // SAFETY: all tensors have the same shape and length; one thread writes
        // each element after widening both half inputs for the product.
        unsafe {
            self.half_ops.sam_f16(
                &self.stream,
                launch::elementwise(input.len(), config),
                &input.buf,
                &source.buf,
                &mut output,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, output_shape))
    }

    /// Native half-storage NCHW channel shuffle.
    pub fn channel_shuffle_f16(
        &self,
        x: &OxideTensorF16,
        output_shape: &[usize],
        channels: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("channel_shuffle_f16")?;
        if x.shape() != output_shape
            || output_shape.len() != 4
            || channels == 0
            || groups == 0
            || channels % groups != 0
            || output_shape[1] != channels
            || output_shape[2] == 0
            || output_shape[3] == 0
        {
            return Err(shape_err(
                "channel_shuffle_f16",
                "incompatible NCHW shape or groups".to_owned(),
            ));
        }
        let spatial = output_shape[2] * output_shape[3];
        let mut output = self.alloc16_uninit(x.len())?;
        // SAFETY: matching nonempty NCHW shape, groups divides channels; the
        // kernel maps each output channel/position to an in-bounds input.
        unsafe {
            self.half_ops.channel_shuffle_f16(
                &self.stream,
                launch::elementwise(output.len(), config),
                &x.buf,
                &mut output,
                channels,
                groups,
                spatial,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, output_shape))
    }

    /// Native half-storage LAYA act-head feature extraction.
    pub fn laya_act_features_f16(
        &self,
        head: &OxideTensorF16,
        scorer: &OxideTensorF16,
        marker: &OxideTensorF16,
        head_shape: &[usize],
        scorer_shape: &[usize],
        marker_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("laya_act_features_f16")?;
        let layout =
            darknet_cuda::host_ops::laya_act::Layout::new(head_shape, scorer_shape, marker_shape)
                .map_err(|err| shape_err("laya_act_features_f16", err))?;
        if head.shape() != head_shape
            || scorer.shape() != scorer_shape
            || marker.shape() != marker_shape
            || head.len() != layout.head_len
            || scorer.len() != layout.logits_len
            || marker.len() != layout.marker_len
        {
            return Err(shape_err(
                "laya_act_features_f16",
                "input shapes or lengths disagree with the declared layout".to_owned(),
            ));
        }
        let mut output = self.alloc16_uninit(layout.output_len)?;
        // SAFETY: the checked layout bounds every source index; one thread
        // writes one complete, disjoint sample row.
        unsafe {
            self.layers_extra.laya_act_features_f16(
                &self.stream,
                launch::elementwise(layout.batch, config),
                &head.buf,
                &scorer.buf,
                &marker.buf,
                &mut output,
                layout.hidden,
                layout.options,
                layout.marker_channels,
                layout.head_width,
                layout.marker_width,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, &layout.output_shape()))
    }

    /// Native exact-erf GELU over half-storage values.
    pub fn gelu_erf_f16(
        &self,
        x: &OxideTensorF16,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("gelu_erf_f16")?;
        if x.shape() != shape || x.len() != shape.iter().product::<usize>() {
            return Err(shape_err(
                "gelu_erf_f16",
                format!("tensor shape {:?} does not match {shape:?}", x.shape()),
            ));
        }
        let mut output = self.alloc16_uninit(x.len())?;
        // SAFETY: shape equality validates one input/output element per thread.
        unsafe {
            self.half_ops.gelu_erf_f16(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &mut output,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, shape))
    }

    /// Native f16 softmax with per-group f32 max/sum/normalization.
    pub fn softmax_f16(
        &self,
        x: &OxideTensorF16,
        shape: &[usize],
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("softmax_f16")?;
        if shape.len() != 4
            || x.shape() != shape
            || x.len() != shape.iter().product::<usize>()
            || groups == 0
        {
            return Err(shape_err(
                "softmax_f16",
                format!("incompatible shape {shape:?} or groups={groups}"),
            ));
        }
        let per_batch = shape[1] * shape[2] * shape[3];
        if per_batch == 0 || per_batch % groups != 0 {
            return Err(shape_err(
                "softmax_f16",
                format!("per-sample size {per_batch} is not divisible by groups {groups}"),
            ));
        }
        let group_size = per_batch / groups;
        let n_groups = shape[0] * groups;
        let mut output = self.alloc16_uninit(x.len())?;
        // SAFETY: all group extents are validated and groups own disjoint
        // contiguous output ranges; the kernel writes every element.
        unsafe {
            self.half_ops.softmax_forward_f16(
                &self.stream,
                launch::elementwise(n_groups, config),
                &x.buf,
                &mut output,
                group_size,
                n_groups,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, shape))
    }

    /// Native f16 softmax across channel-major candidates at each spatial group.
    pub fn softmax_channel_major_f16(
        &self,
        x: &OxideTensorF16,
        shape: &[usize],
        candidates: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("softmax_channel_major_f16")?;
        if shape.len() != 4
            || x.shape() != shape
            || x.len() != shape.iter().product::<usize>()
            || candidates == 0
            || groups == 0
            || shape[1] != candidates
            || shape[2] * shape[3] != groups
        {
            return Err(shape_err(
                "softmax_channel_major_f16",
                format!("incompatible shape {shape:?}, candidates={candidates}, groups={groups}"),
            ));
        }
        let n_groups = shape[0] * groups;
        let mut output = self.alloc16_uninit(x.len())?;
        // SAFETY: validated NCHW dimensions give one disjoint candidate stride
        // per (batch, spatial-group) thread, and every input/output slot is covered.
        unsafe {
            self.half_ops.softmax_channel_major_f16(
                &self.stream,
                launch::elementwise(n_groups, config),
                &x.buf,
                &mut output,
                candidates,
                groups,
                n_groups,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, shape))
    }

    /// Native half-storage Region layout and activation forward.
    pub fn region_forward_f16(
        &self,
        x: &OxideTensorF16,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("region_forward_f16")?;
        if shape.len() != 4
            || x.shape() != shape
            || x.len() != shape.iter().product::<usize>()
            || anchors == 0
            || entries < classes.saturating_add(5)
            || classes == 0
            || shape[1] != anchors.saturating_mul(entries)
            || shape[2] == 0
            || shape[3] == 0
        {
            return Err(shape_err(
                "region_forward_f16",
                format!(
                    "incompatible shape/parameters: {shape:?}, anchors={anchors}, entries={entries}, classes={classes}"
                ),
            ));
        }
        let spatial = shape[2] * shape[3];
        let groups = shape[0] * spatial * anchors;
        let mut output = self.alloc16_uninit(x.len())?;
        // SAFETY: shape/layout validation establishes in-bounds channel planes;
        // one thread writes each spatial/anchor entry row exactly once.
        unsafe {
            self.half_ops.region_forward_f16(
                &self.stream,
                launch::elementwise(groups, config),
                &x.buf,
                &mut output,
                spatial,
                anchors,
                entries,
                classes,
                softmax_classes as u32,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, shape))
    }

    /// Native f16 reorg permutation; this is inference forward only.
    pub fn reorg_f16(
        &self,
        x: &OxideTensorF16,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("reorg_f16")?;
        let dims = darknet_cuda::host_ops::reorg_dims(input_shape, output_shape, stride, reverse)
            .map_err(|message| shape_err("reorg_f16", message))?;
        let to_small = !reverse;
        let (expected_input, output_len) = if to_small {
            (dims.big_len(), dims.small_len())
        } else {
            (dims.small_len(), dims.big_len())
        };
        if x.shape() != input_shape || x.len() != expected_input {
            return Err(shape_err(
                "reorg_f16",
                format!(
                    "input has shape {:?} and {} elements, expected {input_shape:?} and {expected_input}",
                    x.shape(),
                    x.len()
                ),
            ));
        }
        let mut output = self.alloc16_uninit(output_len)?;
        // SAFETY: shared reorg_dims validation establishes all bounds; each
        // output element is gathered once and reverse borders are zero-filled.
        unsafe {
            if to_small {
                self.half_ops.reorg_to_small_f16(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &x.buf,
                    &mut output,
                    dims.cb,
                    dims.hb,
                    dims.wb,
                    dims.hs,
                    dims.ws,
                    dims.stride,
                )
            } else {
                self.half_ops.reorg_to_big_f16(
                    &self.stream,
                    launch::elementwise(output_len, config),
                    &x.buf,
                    &mut output,
                    dims.cb,
                    dims.hb,
                    dims.wb,
                    dims.hs,
                    dims.ws,
                    dims.stride,
                )
            }
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, output_shape))
    }

    /// `f16(act(x * scale[c] + bias[c]))` per channel (`c = (i / channel_stride) % channels`);
    /// `scale = None` is bias + activation only.  The folded inference batch norm.
    pub fn channel_affine_act_f16(
        &self,
        x: &OxideTensorF16,
        scale: Option<&OxideTensorF32>,
        bias: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.require_f16("channel_affine_act_f16")?;
        let op = "channel_affine_act_f16";
        if channels == 0 || channel_stride == 0 || x.len() % (channels * channel_stride) != 0 {
            return Err(shape_err(
                op,
                format!(
                    "length {} not divisible by {channels} x {channel_stride}",
                    x.len()
                ),
            ));
        }
        if bias.len() != channels || scale.is_some_and(|s| s.len() != channels) {
            return Err(shape_err(
                op,
                format!("per-channel tensors must hold {channels} elements"),
            ));
        }
        let mut out = self.alloc16_uninit(x.len())?;
        let s = scale.unwrap_or(bias);
        // SAFETY: one thread per element; per-channel indices are in range by the checks above.
        unsafe {
            self.half_ops.channel_affine_act_f16(
                &self.stream,
                launch::elementwise(x.len(), config),
                &x.buf,
                &s.buf,
                &bias.buf,
                &mut out,
                channels,
                channel_stride,
                scale.is_some() as u32,
                activation_code(activation),
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(out, x.shape()))
    }

    /// Standalone f16 bias + activation (`f16(act(x + bias[c]))`).
    pub fn bias_activation_f16(
        &self,
        x: &OxideTensorF16,
        bias: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        self.channel_affine_act_f16(x, None, bias, channels, channel_stride, activation, config)
    }

    /// Native half-storage gather. Feature payloads never pass through f32;
    /// only marker positions are widened on device before rounding/clamping.
    pub fn marker_gather_f16(
        &self,
        features: &OxideTensorF16,
        markers: &OxideTensorF16,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        let op = "marker_gather_f16";
        let d = darknet_cuda::host_ops::marker_gather::Layout::new(
            feature_shape,
            marker_shape,
            options,
        )
        .map_err(|err| shape_err(op, err))?;
        if features.len() != d.input_len || markers.len() != d.markers_len {
            return Err(shape_err(
                op,
                "tensor storage does not match layout".to_owned(),
            ));
        }
        let mut output = self.alloc16_uninit(d.output_len)?;
        // SAFETY: validated NCHW extents/storage bound all reads and each
        // thread writes one distinct half output, including ragged tails.
        unsafe {
            self.layers_extra.marker_gather_f16(
                &self.stream,
                launch::elementwise(d.output_len, config),
                &features.buf,
                &markers.buf,
                &mut output,
                d.hidden,
                d.sequence,
                d.marker_channels,
                d.options,
            )
        }
        .map_err(launch_err)?;
        Ok(OxideTensorF16::new(output, &d.output_shape()))
    }

    /// f16 channel concat with the same slice metadata as `route_concat_f32`.
    #[allow(clippy::too_many_arguments)]
    pub fn route_concat_f16(
        &self,
        inputs: &[&OxideTensorF16],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF16, OxideError> {
        let op = "route_concat_f16";
        let n = inputs.len();
        if n == 0
            || input_channel_offsets.len() != n
            || input_channels.len() != n
            || input_total_channels.len() != n
            || channel_stride == 0
        {
            return Err(shape_err(
                op,
                "needs >= 1 input, equal-length metadata and channel_stride > 0".to_owned(),
            ));
        }
        let out_total: usize = input_channels.iter().sum();
        let out_len: usize = output_shape.iter().product();
        let mut batch = None;
        for k in 0..n {
            if input_channels[k] == 0
                || input_channel_offsets[k] + input_channels[k] > input_total_channels[k]
            {
                return Err(shape_err(
                    op,
                    format!("input {k}: slice outside its channels"),
                ));
            }
            let per_batch = input_total_channels[k] * channel_stride;
            if inputs[k].len() == 0 || inputs[k].len() % per_batch != 0 {
                return Err(shape_err(
                    op,
                    format!(
                        "input {k} length {} not a multiple of {per_batch}",
                        inputs[k].len()
                    ),
                ));
            }
            let b = inputs[k].len() / per_batch;
            if *batch.get_or_insert(b) != b {
                return Err(shape_err(
                    op,
                    format!("input {k} batch {b} differs from {batch:?}"),
                ));
            }
        }
        let batch = batch.unwrap_or(0);
        if out_len != batch * out_total * channel_stride {
            return Err(shape_err(
                op,
                format!("output shape {output_shape:?} does not match the inputs"),
            ));
        }
        let mut out = self.alloc16_uninit(out_len)?;
        let mut dst_offset = 0usize;
        for k in 0..n {
            let count = batch * input_channels[k] * channel_stride;
            // SAFETY: one thread per copied element with distinct destinations.
            unsafe {
                self.half_ops.route_copy_f16(
                    &self.stream,
                    launch::elementwise(count, config),
                    &inputs[k].buf,
                    &mut out,
                    batch,
                    input_channels[k],
                    input_total_channels[k],
                    input_channel_offsets[k],
                    out_total,
                    dst_offset,
                    channel_stride,
                )
            }
            .map_err(launch_err)?;
            dst_offset += input_channels[k];
        }
        Ok(OxideTensorF16::new(out, output_shape))
    }

    /// Benchmark helper: mean device ms of one dense NN f16 GEMM
    /// (`alpha = 1`, `beta = 0`, f32 out) over `iters` launches.  `n` and `k`
    /// are padded up to multiples of 8 exactly like the conv path does (so the
    /// 16-byte `cp.async` path runs; unaligned shapes take the slow element
    /// loop), which makes the timed problem slightly larger than `m x n x k`.
    pub fn bench_gemm_f16(
        &self,
        m: usize,
        n: usize,
        k: usize,
        iters: usize,
    ) -> Result<f64, OxideError> {
        let (n, k) = (n.next_multiple_of(8), k.next_multiple_of(8));
        let a = self.upload_f16_bits(&[m, k], &vec![0x3c00; m * k])?;
        let b = self.upload_f16_bits(&[k, n], &vec![0x3c00; k * n])?;
        let mut c = self.alloc_uninit(m * n)?;
        let g = GemmArgs {
            m,
            n,
            k,
            lda: k,
            ldb: n,
            ldc: n,
            ..GemmArgs::default()
        };
        self.time_gpu_ms(iters, || self.hgemm(&a.buf, &b.buf, HOut::F32(&mut c), &g))
    }

    /// True when the mode asks for f16 storage inference (`DARKNET_OXIDE_GEMM=fp16`).
    pub fn f16_storage_enabled(&self) -> bool {
        self.gemm_mode == GemmMode::Fp16
    }
}

impl darknet_cuda::F16TensorOpsBackend for OxideBackend {
    type TensorF16 = OxideTensorF16;

    fn upload_f16(&self, shape: &[usize], host: &[f32]) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::upload_f16(self, shape, host)
    }

    fn download_f16(&self, tensor: &Self::TensorF16) -> Result<Vec<f32>, Self::Error> {
        OxideBackend::download_f16(self, tensor)
    }

    fn normalize_channels_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        normalization: F32ChannelNormalization,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::normalize_channels_f16(self, x, shape, normalization, config)
    }

    fn conv_weights_f16(
        &self,
        weights: &OxideTensorF32,
        filters: usize,
        per_filter: usize,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::conv_weights_f16(self, weights, filters, per_filter)
    }

    fn conv2d_bias_act_f16(
        &self,
        x: &Self::TensorF16,
        weights: &Self::TensorF16,
        bias: &OxideTensorF32,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        filters: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::conv2d_bias_act_f16(
            self,
            x,
            weights,
            bias,
            output_shape,
            channels,
            input_height,
            input_width,
            filters,
            size,
            stride_x,
            stride_y,
            dilation,
            padding,
            groups,
            activation,
            config,
        )
    }

    fn connected_f16(
        &self,
        x: &Self::TensorF16,
        weights: &[f32],
        biases: &[f32],
        input_shape: &[usize],
        output_shape: &[usize],
        outputs: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::connected_f16(
            self,
            x,
            weights,
            biases,
            input_shape,
            output_shape,
            outputs,
            activation,
            config,
        )
    }

    fn gelu_erf_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::gelu_erf_f16(self, x, shape, config)
    }

    fn maxpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::maxpool_f16(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config,
        )
    }

    fn maxpool_depth_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        input_channels: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::maxpool_depth_f16(self, x, output_shape, input_channels, config)
    }

    fn upsample_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        stride: usize,
        scale: f32,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::upsample_f16(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            stride,
            scale,
            reverse,
            config,
        )
    }

    fn marker_gather_f16(
        &self,
        features: &Self::TensorF16,
        markers: &Self::TensorF16,
        feature_shape: &[usize],
        marker_shape: &[usize],
        options: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::marker_gather_f16(
            self,
            features,
            markers,
            feature_shape,
            marker_shape,
            options,
            config,
        )
    }

    fn laya_act_features_f16(
        &self,
        head: &Self::TensorF16,
        scorer: &Self::TensorF16,
        marker: &Self::TensorF16,
        head_shape: &[usize],
        scorer_shape: &[usize],
        marker_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::laya_act_features_f16(
            self,
            head,
            scorer,
            marker,
            head_shape,
            scorer_shape,
            marker_shape,
            config,
        )
    }

    fn route_concat_f16(
        &self,
        inputs: &[&Self::TensorF16],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        output_shape: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::route_concat_f16(
            self,
            inputs,
            input_channel_offsets,
            input_channels,
            input_total_channels,
            output_shape,
            channel_stride,
            config,
        )
    }

    fn local_avgpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::local_avgpool_f16(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
            padding,
            config,
        )
    }

    fn avgpool_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        input_height: usize,
        input_width: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::avgpool_f16(
            self,
            x,
            output_shape,
            channels,
            input_height,
            input_width,
            config,
        )
    }

    fn shortcut_fit_f16(
        &self,
        x: &Self::TensorF16,
        batch: usize,
        output_shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::shortcut_fit_f16(self, x, batch, output_shape, config)
    }

    fn shortcut_add_act_f16(
        &self,
        a: &Self::TensorF16,
        b: &Self::TensorF16,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::shortcut_add_act_f16(self, a, b, activation, config)
    }

    fn scale_channels_f16(
        &self,
        input: &Self::TensorF16,
        source: &Self::TensorF16,
        input_shape: &[usize],
        output_shape: &[usize],
        scale_wh: bool,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::scale_channels_f16(
            self,
            input,
            source,
            input_shape,
            output_shape,
            scale_wh,
            activation,
            config,
        )
    }

    fn sam_f16(
        &self,
        input: &Self::TensorF16,
        source: &Self::TensorF16,
        output_shape: &[usize],
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::sam_f16(self, input, source, output_shape, activation, config)
    }

    fn channel_shuffle_f16(
        &self,
        x: &Self::TensorF16,
        output_shape: &[usize],
        channels: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::channel_shuffle_f16(self, x, output_shape, channels, groups, config)
    }

    fn reorg_f16(
        &self,
        x: &Self::TensorF16,
        input_shape: &[usize],
        output_shape: &[usize],
        stride: usize,
        reverse: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::reorg_f16(self, x, input_shape, output_shape, stride, reverse, config)
    }

    fn softmax_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::softmax_f16(self, x, shape, groups, config)
    }

    fn softmax_channel_major_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        candidates: usize,
        groups: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::softmax_channel_major_f16(self, x, shape, candidates, groups, config)
    }

    fn region_forward_f16(
        &self,
        x: &Self::TensorF16,
        shape: &[usize],
        anchors: usize,
        entries: usize,
        classes: usize,
        softmax_classes: bool,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::region_forward_f16(
            self,
            x,
            shape,
            anchors,
            entries,
            classes,
            softmax_classes,
            config,
        )
    }

    fn channel_affine_act_f16(
        &self,
        x: &Self::TensorF16,
        scale: Option<&OxideTensorF32>,
        bias: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<Self::TensorF16, Self::Error> {
        OxideBackend::channel_affine_act_f16(
            self,
            x,
            scale,
            bias,
            channels,
            channel_stride,
            activation,
            config,
        )
    }
}

//! Convolution drivers: geometry validation, the direct / im2col+GEMM selector
//! and the launch sequences behind the three `conv2d_*` trait methods.
//!
//! Weights are `[filters, channels / groups, k, k]`, tensors NCHW, a single
//! symmetric `padding`, `out = (in + 2*pad - (dil*(k-1)+1)) / stride + 1`
//! (identical to the CPU reference in `darknet-core` and the cuTile bridge).
//!
//! The im2col path processes one batch element at a time so the cached
//! workspace is only `C * k * k * OH * OW` floats.  Weight-gradient batches
//! are accumulated with sequential GEMM launches (`beta = 1`), which keeps the
//! reduction deterministic.

use cuda_core::DeviceBuffer;
use darknet_cuda::ElementwiseLaunchConfig;

use crate::kernels::conv_direct::WGRAD_THREADS;
use crate::{
    GemmArgs, ImplicitConv, OxideBackend, OxideError, OxideTensorF32, launch, launch_err,
};

/// Which convolution implementation to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvMode {
    Direct,
    Gemm,
    Implicit,
    Auto,
}

impl ConvMode {
    /// `DARKNET_OXIDE_CONV={direct,gemm,implicit,auto}`; unset or unknown means
    /// `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("DARKNET_OXIDE_CONV")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "direct" => Self::Direct,
            "gemm" => Self::Gemm,
            "implicit" => Self::Implicit,
            _ => Self::Auto,
        }
    }
}

/// Default cap on the im2col column buffer used by the batched paths (bytes);
/// larger batches run in chunks of images.  Overridable with
/// `DARKNET_OXIDE_COL_CAP_MB` (A/B the forward/backward chunk size).
pub(crate) const COL_CAP_BYTES: usize = 32 << 20;

/// Cap actually used, see [`COL_CAP_BYTES`].
pub(crate) fn col_cap_bytes() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("DARKNET_OXIDE_COL_CAP_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .map(|v| v << 20)
            .unwrap_or(COL_CAP_BYTES)
    })
}

/// Upper bound on the per-(image, group) weight-gradient partial buffer used by
/// the batched wgrad path (bytes); larger batches run in chunks of images.
pub(crate) const PARTIAL_CAP_BYTES: usize = 64 << 20;

/// Images per batched im2col + GEMM round, at most (`DARKNET_OXIDE_CONV_CHUNK`,
/// default unlimited; `1` restores the old one-launch-per-image loop for A/B).
pub(crate) fn chunk_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("DARKNET_OXIDE_CONV_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(usize::MAX)
    })
}

/// Problems below this many multiply-adds run on the direct kernel in `Auto`.
const TINY_MACS: usize = 1 << 16;

#[derive(Clone, Copy)]
pub(crate) struct Geom {
    pub(crate) batch: usize,
    pub(crate) c: usize,
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) f: usize,
    pub(crate) k: usize,
    pub(crate) sx: usize,
    pub(crate) sy: usize,
    pub(crate) dil: usize,
    pub(crate) pad: usize,
    pub(crate) groups: usize,
    pub(crate) oh: usize,
    pub(crate) ow: usize,
}

impl Geom {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        op: &str,
        batch: usize,
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
    ) -> Result<Self, OxideError> {
        let bad = |msg: String| OxideError::Shape(format!("{op}: {msg}"));
        if batch == 0 || c == 0 || h == 0 || w == 0 || f == 0 || k == 0 {
            return Err(bad("non-zero batch, channels, size, filters and kernel required".into()));
        }
        if sx == 0 || sy == 0 || dil == 0 || groups == 0 {
            return Err(bad("stride, dilation and groups must be > 0".into()));
        }
        if c % groups != 0 || f % groups != 0 {
            return Err(bad(format!(
                "channels {c} and filters {f} must be divisible by groups {groups}"
            )));
        }
        let extent = dil * (k - 1) + 1;
        if h + 2 * pad < extent || w + 2 * pad < extent {
            return Err(bad(format!(
                "kernel extent {extent} exceeds padded input {}x{}",
                h + 2 * pad,
                w + 2 * pad
            )));
        }
        let oh = (h + 2 * pad - extent) / sy + 1;
        let ow = (w + 2 * pad - extent) / sx + 1;
        Ok(Self { batch, c, h, w, f, k, sx, sy, dil, pad, groups, oh, ow })
    }

    pub(crate) fn cpg(&self) -> usize {
        self.c / self.groups
    }
    pub(crate) fn fpg(&self) -> usize {
        self.f / self.groups
    }
    /// Rows of the per-group column matrix.
    pub(crate) fn kg(&self) -> usize {
        self.cpg() * self.k * self.k
    }
    pub(crate) fn ohw(&self) -> usize {
        self.oh * self.ow
    }
    pub(crate) fn x_len(&self) -> usize {
        self.batch * self.c * self.h * self.w
    }
    pub(crate) fn y_len(&self) -> usize {
        self.batch * self.f * self.ohw()
    }
    pub(crate) fn w_len(&self) -> usize {
        self.f * self.kg()
    }
    pub(crate) fn col_len(&self) -> usize {
        self.c * self.k * self.k * self.ohw()
    }
    fn macs(&self) -> usize {
        self.batch * self.f * self.kg() * self.ohw()
    }
}

pub(crate) fn expect_len(op: &str, what: &str, got: usize, want: usize) -> Result<(), OxideError> {
    if got != want {
        return Err(OxideError::Shape(format!(
            "{op}: {what} length mismatch: expected {want}, got {got}"
        )));
    }
    Ok(())
}

pub(crate) fn expect_shape(op: &str, what: &str, got: &[usize], want: &[usize]) -> Result<(), OxideError> {
    if got != want {
        return Err(OxideError::Shape(format!(
            "{op}: {what} shape must be {want:?}, got {got:?}"
        )));
    }
    Ok(())
}

impl OxideBackend {
    fn use_direct(&self, g: &Geom) -> bool {
        match self.conv_mode {
            ConvMode::Direct => true,
            ConvMode::Gemm | ConvMode::Implicit => false,
            ConvMode::Auto => (g.groups == g.c && g.c == g.f) || g.macs() < TINY_MACS,
        }
    }

    /// Whether the forward should use the implicit-GEMM kernel.  Measured on
    /// sm_89 (LegoGears_v2, batch 16): the materialised im2col + GEMM is ~5%
    /// faster, because the materialised `B` tile is read with 16-byte
    /// `cp.async` while the implicit tile can only issue 4-byte scalar copies
    /// (arbitrary input-column alignment).  So it is opt-in via
    /// `DARKNET_OXIDE_CONV=implicit` and never picked by `Auto`.
    fn use_implicit(&self, _g: &Geom) -> bool {
        matches!(self.conv_mode, ConvMode::Implicit)
    }

    /// Makes sure the cached workspace holds `len` floats and runs `body` on it.
    pub(crate) fn with_workspace<R>(
        &self,
        len: usize,
        body: impl FnOnce(&mut DeviceBuffer<f32>) -> Result<R, OxideError>,
    ) -> Result<R, OxideError> {
        let mut guard = self
            .workspace
            .lock()
            .map_err(|_| OxideError::Shape("conv workspace lock poisoned".into()))?;
        if guard.as_ref().is_none_or(|b| b.len() < len) {
            *guard = Some(DeviceBuffer::<f32>::zeroed(&self.stream, len)?);
        }
        body(guard.as_mut().expect("workspace allocated above"))
    }

    /// Benchmark helper: mean device ms per single-image im2col launch for a
    /// `[1, c, h, w]` input (event timed, workspace reused).  Returns
    /// `(ms, column_bytes_written)`.
    #[allow(clippy::too_many_arguments)]
    pub fn bench_im2col_f32(
        &self,
        x: &OxideTensorF32,
        c: usize,
        h: usize,
        w: usize,
        k: usize,
        stride: usize,
        pad: usize,
        iters: usize,
    ) -> Result<(f64, usize), OxideError> {
        let g = Geom::new("bench_im2col", 1, c, h, w, c, k, stride, stride, 1, pad, 1)?;
        expect_len("bench_im2col", "x", x.len(), g.x_len())?;
        let config = ElementwiseLaunchConfig::default();
        let ms = self.with_workspace(g.col_len(), |ws| {
            self.time_gpu_ms(iters, || self.launch_im2col(&g, &x.buf, ws, 0, 1, config))
        })?;
        Ok((ms, g.col_len() * 4))
    }

    fn launch_im2col(
        &self,
        g: &Geom,
        x: &DeviceBuffer<f32>,
        col: &mut DeviceBuffer<f32>,
        batch_index: usize,
        images: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(), OxideError> {
        // SAFETY: one block per column row (`images * C * k * k` rows), each
        // sweeping `images * col_len` positions; `col` holds `images *
        // col_len` floats and `x` covers images `batch_index..+images`
        // (validated by the callers).
        unsafe {
            self.im2col.im2col(
                &self.stream,
                launch::blocks_of(images * g.c * g.k * g.k, launch::block_size(config)),
                x,
                col,
                batch_index * g.c * g.h * g.w,
                g.c,
                g.h,
                g.w,
                g.k,
                g.sx,
                g.sy,
                g.dil,
                g.pad,
                g.oh,
                g.ow,
                images,
            )
        }
        .map_err(launch_err)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_forward(
        &self,
        x: &OxideTensorF32,
        weights: &OxideTensorF32,
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
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "conv2d_f32";
        let plane = c.checked_mul(h).and_then(|v| v.checked_mul(w)).unwrap_or(0);
        if plane == 0 || x.len() % plane != 0 {
            return Err(OxideError::Shape(format!(
                "{OP}: input length {} is not a multiple of C*H*W = {plane}",
                x.len()
            )));
        }
        let g = Geom::new(OP, x.len() / plane, c, h, w, f, k, sx, sy, dil, pad, groups)?;
        expect_len(OP, "weights", weights.len(), g.w_len())?;
        expect_shape(OP, "output", output_shape, &[g.batch, g.f, g.oh, g.ow])?;

        let mut out = self.alloc_uninit(g.y_len())?;
        if self.use_direct(&g) {
            // SAFETY: one thread per output element.
            unsafe {
                self.conv_direct.conv_direct_fwd(
                    &self.stream,
                    launch::elementwise(g.y_len(), config),
                    &x.buf,
                    &weights.buf,
                    &mut out,
                    g.batch,
                    g.c,
                    g.h,
                    g.w,
                    g.f,
                    g.k,
                    g.sx,
                    g.sy,
                    g.dil,
                    g.pad,
                    g.groups,
                    g.oh,
                    g.ow,
                )
            }
            .map_err(launch_err)?;
        } else if self.use_implicit(&g) {
            // One implicit-GEMM launch for the whole batch (no column buffer to
            // bound): instances are `(image, group)`, image-major.
            let (kg, ohw, fpg) = (g.kg(), g.ohw(), g.fpg());
            let inst = g.batch * groups;
            self.gemm_conv_implicit(
                &weights.buf,
                &x.buf,
                &mut out,
                &GemmArgs {
                    m: fpg,
                    n: ohw,
                    k: kg,
                    lda: kg,
                    ldc: ohw,
                    a: (0, fpg * kg, groups),
                    c: (0, fpg * ohw, inst),
                    count: inst,
                    ..GemmArgs::default()
                },
                &ImplicitConv {
                    x_base: 0,
                    channels: g.c,
                    in_h: g.h,
                    in_w: g.w,
                    cpg: g.cpg(),
                    groups,
                    ksize: g.k,
                    stride_x: g.sx,
                    stride_y: g.sy,
                    dilation: g.dil,
                    padding: g.pad,
                    out_h: g.oh,
                    out_w: g.ow,
                },
            )?;
        } else {
            let (kg, ohw, fpg) = (g.kg(), g.ohw(), g.fpg());
            // One im2col + one GEMM launch per chunk of images (instances =
            // image x group); the column buffer is bounded by `col_cap_bytes()`.
            let chunk = (col_cap_bytes() / 4 / g.col_len()).clamp(1, g.batch).min(chunk_limit());
            self.with_workspace(chunk * g.col_len(), |ws| {
                let mut b0 = 0;
                while b0 < g.batch {
                    let nb = chunk.min(g.batch - b0);
                    self.launch_im2col(&g, &x.buf, ws, b0, nb, config)?;
                    let inst = nb * groups;
                    self.gemm(
                        &weights.buf,
                        ws,
                        &mut out,
                        &GemmArgs {
                            m: fpg,
                            n: ohw,
                            k: kg,
                            lda: kg,
                            ldb: ohw,
                            ldc: ohw,
                            a: (0, fpg * kg, groups),
                            b: (0, kg * ohw, inst),
                            c: (b0 * g.f * ohw, fpg * ohw, inst),
                            count: inst,
                            ..GemmArgs::default()
                        },
                    )?;
                    b0 += nb;
                }
                Ok(())
            })?;
        }
        Ok(OxideTensorF32::new(out, &[g.batch, g.f, g.oh, g.ow]))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_backward_input(
        &self,
        dy: &OxideTensorF32,
        weights: &OxideTensorF32,
        input_shape: &[usize],
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
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "conv2d_backward_input_f32";
        if input_shape.len() != 4 {
            return Err(OxideError::Shape(format!(
                "{OP}: input shape must be [batch, channels, height, width], got {input_shape:?}"
            )));
        }
        let g = Geom::new(OP, input_shape[0], c, h, w, f, k, sx, sy, dil, pad, groups)?;
        expect_shape(OP, "input", input_shape, &[g.batch, g.c, g.h, g.w])?;
        expect_len(OP, "weights", weights.len(), g.w_len())?;
        expect_len(OP, "output gradient", dy.len(), g.y_len())?;

        // Both paths write every element (direct: one thread each; col2im is a per-element gather that stores).
        let mut dx = self.alloc_uninit(g.x_len())?;
        if self.use_direct(&g) {
            // SAFETY: one thread per input element.
            unsafe {
                self.conv_direct.conv_direct_dgrad(
                    &self.stream,
                    launch::elementwise(g.x_len(), config),
                    &dy.buf,
                    &weights.buf,
                    &mut dx,
                    g.batch,
                    g.c,
                    g.h,
                    g.w,
                    g.f,
                    g.k,
                    g.sx,
                    g.sy,
                    g.dil,
                    g.pad,
                    g.groups,
                    g.oh,
                    g.ow,
                )
            }
            .map_err(launch_err)?;
        } else {
            let (kg, ohw, fpg) = (g.kg(), g.ohw(), g.fpg());
            // One batched GEMM over a chunk of images (instances = image x
            // group; the instance index is image-major, so `b` and `c` are
            // simple strides), then one col2im per image reading its block of
            // the shared column workspace.
            let chunk = (col_cap_bytes() / 4 / g.col_len()).clamp(1, g.batch).min(chunk_limit());
            self.with_workspace(chunk * g.col_len(), |ws| {
                let mut b0 = 0;
                while b0 < g.batch {
                    let nb = chunk.min(g.batch - b0);
                    let inst = nb * groups;
                    // dcol[b, g] (kg x ohw) = W[g]^T (kg x fpg) * dy[b, g] (fpg x ohw)
                    self.gemm(
                        &weights.buf,
                        &dy.buf,
                        ws,
                        &GemmArgs {
                            m: kg,
                            n: ohw,
                            k: fpg,
                            lda: kg,
                            ldb: ohw,
                            ldc: ohw,
                            trans_a: true,
                            a: (0, fpg * kg, groups),
                            b: (b0 * g.f * ohw, fpg * ohw, inst),
                            c: (0, kg * ohw, inst),
                            count: inst,
                            ..GemmArgs::default()
                        },
                    )?;
                    for bi in 0..nb {
                        let image = b0 + bi;
                        // SAFETY: one thread per input element of batch `image`;
                        // `bi * col_len` selects that image's column block.
                        unsafe {
                            self.im2col.col2im(
                                &self.stream,
                                launch::elementwise(g.c * g.h * g.w, config),
                                &*ws,
                                &mut dx,
                                bi * g.col_len(),
                                image * g.c * g.h * g.w,
                                g.c,
                                g.h,
                                g.w,
                                g.k,
                                g.sx,
                                g.sy,
                                g.dil,
                                g.pad,
                                g.oh,
                                g.ow,
                            )
                        }
                        .map_err(launch_err)?;
                    }
                    b0 += nb;
                }
                Ok(())
            })?;
        }
        Ok(OxideTensorF32::new(dx, &[g.batch, g.c, g.h, g.w]))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_backward_weights(
        &self,
        x: &OxideTensorF32,
        dy: &OxideTensorF32,
        weight_shape: &[usize],
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
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        const OP: &str = "conv2d_backward_weights_f32";
        let plane = c.checked_mul(h).and_then(|v| v.checked_mul(w)).unwrap_or(0);
        if plane == 0 || x.len() % plane != 0 {
            return Err(OxideError::Shape(format!(
                "{OP}: input length {} is not a multiple of C*H*W = {plane}",
                x.len()
            )));
        }
        let g = Geom::new(OP, x.len() / plane, c, h, w, f, k, sx, sy, dil, pad, groups)?;
        expect_shape(OP, "weight", weight_shape, &[g.f, g.cpg(), g.k, g.k])?;
        expect_len(OP, "output gradient", dy.len(), g.y_len())?;

        // Every element is written (direct: one block each; GEMM: `beta = 0` on the first image).
        let mut dw = self.alloc_uninit(g.w_len())?;
        if self.use_direct(&g) {
            // SAFETY: one block per weight element, launched for exactly
            // `w_len` blocks of WGRAD_THREADS threads.
            unsafe {
                self.conv_direct.conv_direct_wgrad(
                    &self.stream,
                    launch::blocks_of(g.w_len(), WGRAD_THREADS),
                    &x.buf,
                    &dy.buf,
                    &mut dw,
                    g.batch,
                    g.c,
                    g.h,
                    g.w,
                    g.f,
                    g.k,
                    g.sx,
                    g.sy,
                    g.dil,
                    g.pad,
                    g.groups,
                    g.oh,
                    g.ow,
                )
            }
            .map_err(launch_err)?;
        } else {
            let (kg, ohw, fpg) = (g.kg(), g.ohw(), g.fpg());
            // Batch im2col across the images of a chunk (same cap as the forward
            // path), then one batched GEMM writing per-(image, group) partial
            // weight gradients into a workspace, reduced over the images in a
            // fixed order (`gemm_splitk_reduce`, deterministic).  The image
            // dimension is folded into `splits`, so `dw` accumulates across
            // chunks with `beta = 1`.
            let col_chunk = (col_cap_bytes() / 4 / g.col_len()).clamp(1, g.batch).min(chunk_limit());
            let part_chunk = (PARTIAL_CAP_BYTES / 4 / (g.f * kg).max(1)).clamp(1, g.batch);
            let chunk = col_chunk.min(part_chunk).min(g.batch);
            self.with_workspace(chunk * g.col_len(), |ws| {
                let mut b0 = 0;
                while b0 < g.batch {
                    let nb = chunk.min(g.batch - b0);
                    self.launch_im2col(&g, &x.buf, ws, b0, nb, config)?;
                    let inst = nb * groups;
                    let mut partial = self.alloc_uninit(inst * fpg * kg)?;
                    // dW_partial[b, g] (fpg x kg) = dy[b, g] (fpg x ohw) * col[b, g]^T (ohw x kg)
                    self.gemm(
                        &dy.buf,
                        ws,
                        &mut partial,
                        &GemmArgs {
                            m: fpg,
                            n: kg,
                            k: ohw,
                            lda: ohw,
                            ldb: ohw,
                            ldc: kg,
                            trans_b: true,
                            beta: 0.0,
                            a: (b0 * g.f * ohw, fpg * ohw, inst),
                            b: (0, kg * ohw, inst),
                            c: (0, fpg * kg, inst),
                            count: inst,
                            ..GemmArgs::default()
                        },
                    )?;
                    // SAFETY: one thread per (group, weight) element; the
                    // partials hold `nb` image blocks for each of `groups`.
                    unsafe {
                        self.gemm.gemm_splitk_reduce(
                            &self.stream,
                            launch::elementwise(groups * fpg * kg, config),
                            &partial,
                            &mut dw,
                            fpg,
                            kg,
                            kg,
                            groups,
                            nb,
                            1.0,
                            if b0 == 0 { 0.0 } else { 1.0 },
                            0,
                            fpg * kg,
                            groups,
                        )
                    }
                    .map_err(launch_err)?;
                    b0 += nb;
                }
                Ok(())
            })?;
        }
        Ok(OxideTensorF32::new(dw, &[g.f, g.cpg(), g.k, g.k]))
    }
}

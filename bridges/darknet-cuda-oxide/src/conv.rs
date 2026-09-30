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
use crate::{GemmArgs, OxideBackend, OxideError, OxideTensorF32, launch, launch_err};

/// Which convolution implementation to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvMode {
    Direct,
    Gemm,
    Auto,
}

impl ConvMode {
    /// `DARKNET_OXIDE_CONV={direct,gemm,auto}`; unset or unknown means `Auto`.
    pub fn from_env() -> Self {
        match std::env::var("DARKNET_OXIDE_CONV")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "direct" => Self::Direct,
            "gemm" => Self::Gemm,
            _ => Self::Auto,
        }
    }
}

/// Problems below this many multiply-adds run on the direct kernel in `Auto`.
const TINY_MACS: usize = 1 << 16;

#[derive(Clone, Copy)]
struct Geom {
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
    oh: usize,
    ow: usize,
}

impl Geom {
    #[allow(clippy::too_many_arguments)]
    fn new(
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

    fn cpg(&self) -> usize {
        self.c / self.groups
    }
    fn fpg(&self) -> usize {
        self.f / self.groups
    }
    /// Rows of the per-group column matrix.
    fn kg(&self) -> usize {
        self.cpg() * self.k * self.k
    }
    fn ohw(&self) -> usize {
        self.oh * self.ow
    }
    fn x_len(&self) -> usize {
        self.batch * self.c * self.h * self.w
    }
    fn y_len(&self) -> usize {
        self.batch * self.f * self.ohw()
    }
    fn w_len(&self) -> usize {
        self.f * self.kg()
    }
    fn col_len(&self) -> usize {
        self.c * self.k * self.k * self.ohw()
    }
    fn macs(&self) -> usize {
        self.batch * self.f * self.kg() * self.ohw()
    }
}

fn expect_len(op: &str, what: &str, got: usize, want: usize) -> Result<(), OxideError> {
    if got != want {
        return Err(OxideError::Shape(format!(
            "{op}: {what} length mismatch: expected {want}, got {got}"
        )));
    }
    Ok(())
}

fn expect_shape(op: &str, what: &str, got: &[usize], want: &[usize]) -> Result<(), OxideError> {
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
            ConvMode::Gemm => false,
            ConvMode::Auto => (g.groups == g.c && g.c == g.f) || g.macs() < TINY_MACS,
        }
    }

    /// Makes sure the cached workspace holds `len` floats and runs `body` on it.
    fn with_workspace<R>(
        &self,
        len: usize,
        body: impl FnOnce(&mut DeviceBuffer<f32>) -> Result<R, OxideError>,
    ) -> Result<R, OxideError> {
        let mut guard = self
            .workspace
            .lock()
            .map_err(|_| OxideError::Shape("conv workspace lock poisoned".into()))?;
        if guard.as_ref().is_none_or(|b| b.len() < len) {
            *guard = Some(self.alloc(len)?);
        }
        body(guard.as_mut().expect("workspace allocated above"))
    }

    fn launch_im2col(
        &self,
        g: &Geom,
        x: &DeviceBuffer<f32>,
        col: &mut DeviceBuffer<f32>,
        batch_index: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<(), OxideError> {
        // SAFETY: one thread per column element; `col` holds `col_len` floats
        // and `x` covers `batch_index` (validated by the callers).
        unsafe {
            self.im2col.im2col(
                &self.stream,
                launch::elementwise(g.col_len(), config),
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

        let mut out = self.alloc(g.y_len())?;
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
        } else {
            let (kg, ohw, fpg) = (g.kg(), g.ohw(), g.fpg());
            self.with_workspace(g.col_len(), |ws| {
                for b in 0..g.batch {
                    self.launch_im2col(&g, &x.buf, ws, b, config)?;
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
                            b: (0, kg * ohw, groups),
                            c: (b * g.f * ohw, fpg * ohw, groups),
                            count: groups,
                            ..GemmArgs::default()
                        },
                    )?;
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

        let mut dx = self.alloc(g.x_len())?;
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
            self.with_workspace(g.col_len(), |ws| {
                for b in 0..g.batch {
                    // dcol[g] (kg x ohw) = W[g]^T (kg x fpg) * dy[b, g] (fpg x ohw)
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
                            b: (b * g.f * ohw, fpg * ohw, groups),
                            c: (0, kg * ohw, groups),
                            count: groups,
                            ..GemmArgs::default()
                        },
                    )?;
                    // SAFETY: one thread per input element of batch `b`.
                    unsafe {
                        self.im2col.col2im(
                            &self.stream,
                            launch::elementwise(g.c * g.h * g.w, config),
                            &*ws,
                            &mut dx,
                            b * g.c * g.h * g.w,
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

        let mut dw = self.alloc(g.w_len())?;
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
            self.with_workspace(g.col_len(), |ws| {
                for b in 0..g.batch {
                    self.launch_im2col(&g, &x.buf, ws, b, config)?;
                    // dW[g] (fpg x kg) += dy[b, g] (fpg x ohw) * col[g]^T (ohw x kg)
                    self.gemm(
                        &dy.buf,
                        ws,
                        &mut dw,
                        &GemmArgs {
                            m: fpg,
                            n: kg,
                            k: ohw,
                            lda: ohw,
                            ldb: ohw,
                            ldc: kg,
                            trans_b: true,
                            beta: if b == 0 { 0.0 } else { 1.0 },
                            a: (b * g.f * ohw, fpg * ohw, groups),
                            b: (0, kg * ohw, groups),
                            c: (0, fpg * kg, groups),
                            count: groups,
                            ..GemmArgs::default()
                        },
                    )?;
                }
                Ok(())
            })?;
        }
        Ok(OxideTensorF32::new(dw, &[g.f, g.cpg(), g.k, g.k]))
    }
}

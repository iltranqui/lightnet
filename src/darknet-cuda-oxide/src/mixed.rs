//! Mixed-precision (AMP) training mode of the f32 trait API.
//!
//! Enabled with [`OxideBackend::with_amp`] / `DARKNET_OXIDE_AMP=1` (which also
//! selects `GemmMode::Fp16`); default is off and then nothing in this module
//! runs.  Scheme (the usual NVIDIA AMP recipe):
//!
//! * f32 master weights, f32 optimizer state, f32 weight / bias / BN-scale
//!   gradients and f32 BN statistics: the runtime keeps owning those through the
//!   unchanged f32 trait ops (`sgd_update_f32`, `adam_update_f32`, ...).
//! * f16 working copies of the conv weights, refreshed automatically after every
//!   update (the update creates a new weight tensor; the copy is cached per
//!   buffer identity, see [`OxideBackend::weights_half`]).
//! * activations and activation gradients are stored in f16
//!   (`OxideTensorF32::is_half`), conv forward / dgrad / wgrad run on the f16
//!   tensor-core GEMM with f32 accumulation, im2col is f16, col2im accumulates in
//!   f32 and rounds once, batch-norm statistics and reductions are f32.
//! * ops without an f16 implementation widen their inputs to f32 on entry
//!   ([`OxideBackend::f32v`]) and return f32 tensors, so correctness never
//!   depends on coverage.
//!
//! Which op takes the f16 path: conv forward / dgrad / wgrad whenever AMP is on;
//! every other f16-capable op follows the storage type of its activation input
//! (a half tensor in, a half tensor out), so f32 tensors that merely pass
//! through (weight-gradient accumulation, host losses) are never rounded.
//!
//! Loss scaling is owned by the runtime (`[net] loss_scale`, seeds are scaled,
//! the update divides it out); overflowing gradients show up as Inf / NaN in
//! the f32 weight gradients and are zeroed by the existing
//! `sanitize_nan_inf_f32(Reset)` before the step (no dynamic scale adjustment).

use std::borrow::Cow;
use std::sync::{Arc, Weak};

use cuda_core::DeviceBuffer;
use darknet_cuda::{BatchNormTrainForward, ElementwiseLaunchConfig, F32ActivationKind};

use crate::conv::{Geom, chunk_limit, col_cap_bytes, expect_len, expect_shape};
use crate::half::HOut;
use crate::tensor::{PooledBuf, PooledBuf16, PooledBuf8};
use crate::{
    BIAS_BACKWARD_THREADS, GemmArgs, OxideBackend, OxideError, OxideTensorF16, OxideTensorF32,
    activation_code, check_channel_layout, launch, launch_err, shape_err,
};

/// The 16-bit storage format of mixed-precision mode.  Kernels that read or
/// write 16-bit data take it as a `fmt: u32` ([`HalfFormat::code`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfFormat {
    F16,
    Bf16,
}

impl HalfFormat {
    /// Kernel argument: `0` = f16, `1` = bf16.
    pub fn code(self) -> u32 {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
        }
    }
}

/// Parses a `DARKNET_OXIDE_AMP` value, case-insensitively: `1`, `true`, `on`,
/// `yes`, `fp16`, `f16` select f16; `bf16`, `bfloat16` select bf16; anything
/// else (including the empty string) means AMP off.
pub fn parse_amp_format(value: &str) -> Option<HalfFormat> {
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "fp16" | "f16" => Some(HalfFormat::F16),
        "bf16" | "bfloat16" => Some(HalfFormat::Bf16),
        _ => None,
    }
}

/// `DARKNET_OXIDE_AMP` through [`parse_amp_format`].
pub(crate) fn amp_format_from_env() -> Option<HalfFormat> {
    parse_amp_format(&std::env::var("DARKNET_OXIDE_AMP").unwrap_or_default())
}

/// True when `DARKNET_OXIDE_AMP` selects either 16-bit format.
pub(crate) fn amp_from_env() -> bool {
    amp_format_from_env().is_some()
}

/// One cached f16 copy of an f32 master-weight tensor.
pub(crate) struct W16Entry {
    key: Weak<PooledBuf>,
    filters: usize,
    per_filter: usize,
    half: OxideTensorF16,
}

/// One cached FP8 copy of an f32 weight tensor: the quantised bytes and the
/// device `[factor, 1 / factor]` pair the GEMM epilogue consumes.
#[derive(Clone)]
pub(crate) struct Fp8Weights {
    pub(crate) bytes: Arc<PooledBuf8>,
    pub(crate) scale: Arc<PooledBuf>,
}

/// One cached FP8 weight copy, keyed on the weight buffer identity and format.
pub(crate) struct Fp8WEntry {
    pub(crate) key: Weak<PooledBuf>,
    pub(crate) fmt: crate::fp8::Fp8Format,
    pub(crate) value: Fp8Weights,
}

/// The most recent quantised output gradient of a conv backward: dgrad and
/// wgrad both read the same `dy` buffer in the same format, so the second one
/// reuses the first one's bytes and scale pair.  Keyed on the buffer identity
/// (a live `Weak` pins the allocation, so the address cannot be reused), the
/// element count and the format.
pub(crate) struct Fp8DyEntry {
    pub(crate) key: Weak<PooledBuf>,
    pub(crate) len: usize,
    pub(crate) fmt: crate::fp8::Fp8Format,
    pub(crate) value: Fp8Weights,
}

pub(crate) const W16_CACHE_MAX: usize = 512;

impl OxideBackend {
    /// Switches the f32 trait API to mixed precision (see the module docs).
    /// Turning it on also selects `GemmMode::Fp16` (the f16 tensor-core GEMM).
    pub fn with_amp(self, on: bool) -> Self {
        self.with_amp_format(on.then_some(HalfFormat::F16))
    }

    /// [`Self::with_amp`] with the 16-bit format selectable; `None` turns AMP
    /// off (the GEMM mode is left as it is, like `with_amp(false)`).  `Some`
    /// selects `GemmMode::Fp16` or `GemmMode::Bf16` to match.
    ///
    pub fn with_amp_format(mut self, format: Option<HalfFormat>) -> Self {
        self.amp = format.is_some();
        if let Some(f) = format {
            self.half_fmt = f;
            self.gemm_mode = match f {
                HalfFormat::F16 => crate::GemmMode::Fp16,
                HalfFormat::Bf16 => crate::GemmMode::Bf16,
            };
        }
        self
    }

    /// The 16-bit storage format in use (`F16` unless AMP selected bf16).
    pub fn half_format(&self) -> HalfFormat {
        self.half_fmt
    }

    pub fn amp_enabled(&self) -> bool {
        self.amp
    }

    /// Rounds an f32 tensor into f16 storage (the form AMP activations take);
    /// an already-half tensor is returned as is.  Used to enter the f16 path
    /// from f32 data (tests, host-provided inputs).
    pub fn to_half_storage(&self, t: &OxideTensorF32) -> Result<OxideTensorF32, OxideError> {
        if t.is_half() {
            return Ok(t.clone());
        }
        Ok(self.wrap_half(self.to_half(t, self.half_fmt)?))
    }

    /// Wraps an f16 tensor as a (half-storage) trait tensor.
    pub(crate) fn wrap_half(&self, h: OxideTensorF16) -> OxideTensorF32 {
        OxideTensorF32 {
            buf: self.empty_f32.clone(),
            shape: h.shape.clone(),
            half: Some(h.buf),
        }
    }

    /// The f16 view of a tensor: shares the buffer of a half tensor, rounds an
    /// f32 tensor (round to nearest even, overflow -> Inf, so an overflowing
    /// scaled gradient stays visible instead of being clamped).
    pub(crate) fn as_half(&self, t: &OxideTensorF32) -> Result<OxideTensorF16, OxideError> {
        match &t.half {
            Some(h) => Ok(OxideTensorF16 {
                buf: h.clone(),
                shape: t.shape.clone(),
            }),
            None => self.to_half(t, self.half_fmt),
        }
    }

    /// The f32 view of a tensor: borrowed when it already is f32, else widened
    /// on the device (exact).
    pub(crate) fn f32v<'a>(
        &self,
        t: &'a OxideTensorF32,
    ) -> Result<Cow<'a, OxideTensorF32>, OxideError> {
        match &t.half {
            None => Ok(Cow::Borrowed(t)),
            Some(h) => {
                let wide = self.from_half(
                    &OxideTensorF16 {
                        buf: h.clone(),
                        shape: t.shape.clone(),
                    },
                    self.half_fmt,
                )?;
                Ok(Cow::Owned(wide))
            }
        }
    }

    pub(crate) fn alloc16_zeroed(&self, len: usize) -> Result<PooledBuf16, OxideError> {
        match self.bufs.take16(len) {
            Some(mut b) => {
                b.zero_async(&self.stream)?;
                Ok(PooledBuf16::new(b, &self.bufs))
            }
            None => Ok(PooledBuf16::new(
                DeviceBuffer::<u16>::zeroed(&self.stream, len)?,
                &self.bufs,
            )),
        }
    }

    /// f16 working copy `[filters, kpad]` of f32 master weights.  Weight
    /// tensors are immutable (an optimizer step returns a new tensor), so the
    /// copy is cached by buffer identity and is thereby refreshed exactly once
    /// per update; entries of dropped weight tensors are pruned.
    pub(crate) fn weights_half(
        &self,
        w: &OxideTensorF32,
        filters: usize,
        per_filter: usize,
    ) -> Result<OxideTensorF16, OxideError> {
        let ptr = Arc::as_ptr(&w.buf);
        {
            let cache = self
                .amp_w16
                .lock()
                .map_err(|_| shape_err("weights_half", "lock poisoned".into()))?;
            if let Some(e) = cache.iter().find(|e| {
                e.key.as_ptr() == ptr
                    && e.key.strong_count() > 0
                    && e.filters == filters
                    && e.per_filter == per_filter
            }) {
                return Ok(e.half.clone());
            }
        }
        let half = self.conv_weights_half(w, filters, per_filter, self.half_fmt)?;
        let mut cache = self
            .amp_w16
            .lock()
            .map_err(|_| shape_err("weights_half", "lock poisoned".into()))?;
        cache.retain(|e| e.key.strong_count() > 0);
        if cache.len() >= W16_CACHE_MAX {
            cache.remove(0);
        }
        cache.push(W16Entry {
            key: Arc::downgrade(&w.buf),
            filters,
            per_filter,
            half: half.clone(),
        });
        Ok(half)
    }

    // ---- convolution -----------------------------------------------------

    /// Conv forward on f16 operands: `f16(conv(x, w))` (no bias, no activation,
    /// those are separate trait ops), f32 accumulation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_forward_amp(
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
        const OP: &str = "conv2d_f32 (amp)";
        if groups == 0 || c % groups != 0 {
            return Err(shape_err(
                OP,
                format!("channels {c} not divisible by groups {groups}"),
            ));
        }
        let per_filter = (c / groups) * k * k;
        expect_len(OP, "weights", weights.len(), f * per_filter)?;
        let xh = self.as_half(x)?;
        let wh = self.weights_half(weights, f, per_filter)?;
        let zero_bias = self.alloc(f)?;
        let bias = OxideTensorF32::new(zero_bias, &[f]);
        let out = self.conv2d_bias_act_f16(
            &xh,
            &wh,
            &bias,
            output_shape,
            c,
            h,
            w,
            f,
            k,
            sx,
            sy,
            dil,
            pad,
            groups,
            F32ActivationKind::Linear,
            config,
        )?;
        Ok(self.wrap_half(out))
    }

    /// dgrad: `dcol = W^T dy` on the f16 GEMM (f32 output), then an f32
    /// accumulating col2im that rounds to f16 once.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_backward_input_amp(
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
        const OP: &str = "conv2d_backward_input_f32 (amp)";
        if input_shape.len() != 4 {
            return Err(shape_err(
                OP,
                format!("input shape must be [batch, c, h, w], got {input_shape:?}"),
            ));
        }
        let g = Geom::new(OP, input_shape[0], c, h, w, f, k, sx, sy, dil, pad, groups)?;
        expect_shape(OP, "input", input_shape, &[g.batch, g.c, g.h, g.w])?;
        expect_len(OP, "weights", weights.len(), g.w_len())?;
        expect_len(OP, "output gradient", dy.len(), g.y_len())?;
        let (kg, ohw, fpg) = (g.kg(), g.ohw(), g.fpg());
        let kpad = kg.next_multiple_of(8);
        let dyh = self.as_half(dy)?;
        let wh = self.weights_half(weights, g.f, kg)?;
        let mut dx = self.alloc16_uninit(g.x_len())?;
        let plane = g.c * g.h * g.w;
        let chunk = (col_cap_bytes() / 4 / g.col_len())
            .clamp(1, g.batch)
            .min(chunk_limit());
        self.with_workspace(chunk * g.col_len(), |ws| {
            let mut b0 = 0;
            while b0 < g.batch {
                let nb = chunk.min(g.batch - b0);
                let inst = nb * groups;
                // dcol[img, g] (kg x ohw) = W[g]^T (kg x fpg) * dy[img, g] (fpg x ohw)
                self.hgemm(
                    &wh.buf,
                    &dyh.buf,
                    HOut::F32(&mut *ws),
                    &GemmArgs {
                        m: kg,
                        n: ohw,
                        k: fpg,
                        lda: kpad,
                        ldb: ohw,
                        ldc: ohw,
                        trans_a: true,
                        a: (0, fpg * kpad, groups),
                        b: (b0 * g.f * ohw, fpg * ohw, inst),
                        c: (0, kg * ohw, inst),
                        count: inst,
                        ..GemmArgs::default()
                    },
                )?;
                // SAFETY: one thread per input element of the `nb` images; `ws`
                // holds `nb * col_len` columns and `dx` covers the images.
                unsafe {
                    self.amp_k.amp_col2im(
                        &self.stream,
                        launch::elementwise(nb * plane, config),
                        &*ws,
                        &mut dx,
                        b0 * plane,
                        nb,
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
                        self.half_format().code(),
                    )
                }
                .map_err(launch_err)?;
                b0 += nb;
            }
            Ok(())
        })?;
        Ok(self.wrap_half(OxideTensorF16::new(dx, &[g.batch, g.c, g.h, g.w])))
    }

    /// wgrad: `dW = dy * col^T` per image, accumulated in f32 across the batch
    /// (sequential `beta = 1` launches, deterministic; the f16 GEMM adds a
    /// deterministic split-K when the output grid under-fills the GPU).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_backward_weights_amp(
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
        const OP: &str = "conv2d_backward_weights_f32 (amp)";
        let plane = c.checked_mul(h).and_then(|v| v.checked_mul(w)).unwrap_or(0);
        if plane == 0 || x.len() % plane != 0 {
            return Err(shape_err(
                OP,
                format!(
                    "input length {} is not a multiple of C*H*W = {plane}",
                    x.len()
                ),
            ));
        }
        let g = Geom::new(OP, x.len() / plane, c, h, w, f, k, sx, sy, dil, pad, groups)?;
        expect_shape(OP, "weight", weight_shape, &[g.f, g.cpg(), g.k, g.k])?;
        expect_len(OP, "output gradient", dy.len(), g.y_len())?;
        let (kg, ohw, fpg) = (g.kg(), g.ohw(), g.fpg());
        let (kpad, npad) = (kg.next_multiple_of(8), ohw.next_multiple_of(8));
        let img_col = groups * kpad * npad;
        let xh = self.as_half(x)?;
        let dyh = self.as_half(dy)?;
        let mut dw = self.alloc_uninit(g.w_len())?;
        let chunk = (col_cap_bytes() / 2 / img_col)
            .clamp(1, g.batch)
            .min(chunk_limit());
        let mut col_guard = self
            .half_col
            .lock()
            .map_err(|_| shape_err(OP, "f16 workspace lock poisoned".into()))?;
        if col_guard.as_ref().is_none_or(|b| b.len() < chunk * img_col) {
            *col_guard = Some(DeviceBuffer::<u16>::zeroed(&self.stream, chunk * img_col)?);
        }
        let col = col_guard.as_mut().expect("workspace allocated above");
        let mut b0 = 0;
        while b0 < g.batch {
            let nb = chunk.min(g.batch - b0);
            // SAFETY: one thread per column element; `col` holds
            // `chunk * img_col` halves and `xh` covers images `b0..b0 + nb`.
            unsafe {
                self.half_ops.im2col_f16(
                    &self.stream,
                    launch::elementwise(nb * img_col, config),
                    &xh.buf,
                    &mut *col,
                    b0 * plane,
                    g.h,
                    g.w,
                    g.k,
                    g.sx,
                    g.sy,
                    g.dil,
                    g.pad,
                    g.oh,
                    g.ow,
                    kg,
                    kpad,
                    npad,
                    groups,
                    nb,
                    plane,
                )
            }
            .map_err(launch_err)?;
            for bi in 0..nb {
                let b = b0 + bi;
                // dW[g] (fpg x kg) += dy[b, g] (fpg x ohw) * col[g]^T (ohw x kg)
                self.hgemm(
                    &dyh.buf,
                    &*col,
                    HOut::F32(&mut dw),
                    &GemmArgs {
                        m: fpg,
                        n: kg,
                        k: ohw,
                        lda: ohw,
                        ldb: npad,
                        ldc: kg,
                        trans_b: true,
                        beta: if b == 0 { 0.0 } else { 1.0 },
                        a: (b * g.f * ohw, fpg * ohw, groups),
                        b: (bi * img_col, kpad * npad, groups),
                        c: (0, fpg * kg, groups),
                        count: groups,
                        ..GemmArgs::default()
                    },
                )?;
            }
            b0 += nb;
        }
        Ok(OxideTensorF32::new(dw, &[g.f, g.cpg(), g.k, g.k]))
    }

    // ---- elementwise -----------------------------------------------------

    pub(crate) fn bias_add_amp(
        &self,
        x: &OxideTensorF32,
        bias: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        check_channel_layout(
            "bias_add_f32 (amp)",
            x.len(),
            Some(bias.len()),
            channels,
            channel_stride,
        )?;
        let xh = self.as_half(x)?;
        let bias = self.f32v(bias)?;
        let mut out = self.alloc16_uninit(x.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.amp_k.amp_bias_add(
                &self.stream,
                launch::elementwise(x.len(), config),
                &xh.buf,
                &bias.buf,
                &mut out,
                channels,
                channel_stride,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        Ok(self.wrap_half(OxideTensorF16::new(out, x.shape())))
    }

    /// Per-channel sum of an f16 gradient into an f32 `[channels]` tensor.
    pub(crate) fn bias_backward_amp(
        &self,
        grad: &OxideTensorF32,
        channels: usize,
        channel_stride: usize,
    ) -> Result<OxideTensorF32, OxideError> {
        check_channel_layout(
            "bias_backward_f32 (amp)",
            grad.len(),
            None,
            channels,
            channel_stride,
        )?;
        let gh = self.as_half(grad)?;
        let row_groups = grad.len() / channel_stride / channels;
        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.amp_k.amp_sum_partial(
                &self.stream,
                launch::blocks_of(channels, BIAS_BACKWARD_THREADS as u32),
                &gh.buf,
                &mut partial,
                channels,
                channel_stride,
                row_groups,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        let out = self.fold_partials(&partial, channels)?;
        Ok(OxideTensorF32::new(out, &[channels]))
    }

    pub(crate) fn activation_amp(
        &self,
        x: &OxideTensorF32,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let xh = self.as_half(x)?;
        let mut out = self.alloc16_uninit(x.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.amp_k.amp_activation(
                &self.stream,
                launch::elementwise(x.len(), config),
                &xh.buf,
                &mut out,
                activation_code(activation),
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        Ok(self.wrap_half(OxideTensorF16::new(out, x.shape())))
    }

    pub(crate) fn activation_backward_amp(
        &self,
        activation_input: &OxideTensorF32,
        delta: &OxideTensorF32,
        activation: F32ActivationKind,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        crate::same_len("activation_backward_f32 (amp)", activation_input, delta)?;
        let xh = self.as_half(activation_input)?;
        let dh = self.as_half(delta)?;
        let mut out = self.alloc16_uninit(delta.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.amp_k.amp_activation_backward(
                &self.stream,
                launch::elementwise(delta.len(), config),
                &xh.buf,
                &dh.buf,
                &mut out,
                activation_code(activation),
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        Ok(self.wrap_half(OxideTensorF16::new(out, activation_input.shape())))
    }

    pub(crate) fn vector_add_amp(
        &self,
        x: &OxideTensorF32,
        y: &OxideTensorF32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        crate::same_len("vector_add_f32 (amp)", x, y)?;
        let (xh, yh) = (self.as_half(x)?, self.as_half(y)?);
        let mut out = self.alloc16_uninit(x.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.amp_k.amp_vector_add(
                &self.stream,
                launch::elementwise(x.len(), config),
                &xh.buf,
                &yh.buf,
                &mut out,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        Ok(self.wrap_half(OxideTensorF16::new(out, y.shape())))
    }

    // ---- pooling / resampling / route -----------------------------------

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn maxpool_backward_amp(
        &self,
        x: &OxideTensorF32,
        output_grad: &OxideTensorF32,
        channels: usize,
        input_height: usize,
        input_width: usize,
        size: usize,
        stride_x: usize,
        stride_y: usize,
        padding: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let (out_h, out_w) = Self::grad_hw(
            "maxpool_backward_f32 (amp)",
            output_grad,
            x.len(),
            channels,
            input_height,
            input_width,
        )?;
        crate::pool_layout(
            "maxpool_backward_f32 (amp)",
            x.len(),
            output_grad.shape(),
            channels,
            input_height,
            input_width,
            size,
            stride_x,
            stride_y,
        )?;
        let (xh, dyh) = (self.as_half(x)?, self.as_half(output_grad)?);
        let mut dx = self.alloc16_uninit(x.len())?;
        // SAFETY: one thread per input element (gather, no atomics).
        unsafe {
            self.amp_k.amp_maxpool_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &xh.buf,
                &dyh.buf,
                &mut dx,
                input_height,
                input_width,
                out_h,
                out_w,
                size,
                stride_x,
                stride_y,
                padding / 2,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        Ok(self.wrap_half(OxideTensorF16::new(dx, x.shape())))
    }

    /// Expanding (non-`reverse`) nearest-upsample backward; the caller falls
    /// back to the f32 op for `reverse` layers.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upsample_backward_amp(
        &self,
        output_grad: &OxideTensorF32,
        input_shape: &[usize],
        channels: usize,
        output_height: usize,
        output_width: usize,
        stride: usize,
        scale: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<OxideTensorF32, OxideError> {
        let op = "upsample_backward_f32 (amp)";
        if stride == 0 || input_shape.len() != 4 || input_shape[1] != channels {
            return Err(shape_err(
                op,
                format!("bad stride {stride} or input shape {input_shape:?}"),
            ));
        }
        let (batch, in_h, in_w) = (input_shape[0], input_shape[2], input_shape[3]);
        if output_grad.len() != batch * channels * output_height * output_width {
            return Err(shape_err(
                op,
                "gradient size does not match the shapes".to_owned(),
            ));
        }
        let dyh = self.as_half(output_grad)?;
        let mut dx = self.alloc16_uninit(input_shape.iter().product())?;
        // SAFETY: one thread per input element (gather).
        unsafe {
            self.amp_k.amp_upsample_backward(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &dyh.buf,
                &mut dx,
                in_h,
                in_w,
                output_height,
                output_width,
                stride,
                scale,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        Ok(self.wrap_half(OxideTensorF16::new(dx, input_shape)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn route_backward_amp(
        &self,
        output_grad: &OxideTensorF32,
        input_shapes: &[Vec<usize>],
        input_channel_offsets: &[usize],
        input_channels: &[usize],
        input_total_channels: &[usize],
        channel_stride: usize,
        config: ElementwiseLaunchConfig,
    ) -> Result<Vec<OxideTensorF32>, OxideError> {
        let op = "route_backward_f32 (amp)";
        let n = input_shapes.len();
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
        let per_batch = out_total * channel_stride;
        if per_batch == 0 || output_grad.len() % per_batch != 0 {
            return Err(shape_err(
                op,
                format!(
                    "gradient length {} not a multiple of {per_batch}",
                    output_grad.len()
                ),
            ));
        }
        let batch = output_grad.len() / per_batch;
        let gh = self.as_half(output_grad)?;
        let mut grads = Vec::with_capacity(n);
        let mut src_offset = 0usize;
        for k in 0..n {
            let len: usize = input_shapes[k].iter().product();
            if input_channels[k] == 0
                || input_channel_offsets[k] + input_channels[k] > input_total_channels[k]
                || len != batch * input_total_channels[k] * channel_stride
            {
                return Err(shape_err(
                    op,
                    format!("input {k}: shape {:?} inconsistent", input_shapes[k]),
                ));
            }
            let mut grad = self.alloc16_zeroed(len)?;
            let count = batch * input_channels[k] * channel_stride;
            // SAFETY: distinct destination per thread inside `grad`.
            unsafe {
                self.half_ops.route_copy_f16(
                    &self.stream,
                    launch::elementwise(count, config),
                    &gh.buf,
                    &mut grad,
                    batch,
                    input_channels[k],
                    out_total,
                    src_offset,
                    input_total_channels[k],
                    input_channel_offsets[k],
                    channel_stride,
                )
            }
            .map_err(launch_err)?;
            grads.push(self.wrap_half(OxideTensorF16::new(grad, &input_shapes[k])));
            src_offset += input_channels[k];
        }
        Ok(grads)
    }

    // ---- batch norm ------------------------------------------------------

    /// Batch-statistics BN forward on an f16 input: statistics in f32, the
    /// normalised input (kept for the backward) and the output stored in f16;
    /// scale / bias / rolling statistics stay f32.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batchnorm_train_forward_amp(
        &self,
        x: &OxideTensorF32,
        scales: &OxideTensorF32,
        biases: &OxideTensorF32,
        rolling_mean: &OxideTensorF32,
        rolling_variance: &OxideTensorF32,
        shape: &[usize],
        keep: f32,
        config: ElementwiseLaunchConfig,
    ) -> Result<BatchNormTrainForward<OxideTensorF32>, OxideError> {
        let op = "batchnorm_train_forward_f32 (amp)";
        let (scales, biases) = (self.f32v(scales)?, self.f32v(biases)?);
        let (rolling_mean, rolling_variance) =
            (self.f32v(rolling_mean)?, self.f32v(rolling_variance)?);
        let (channels, stride, row_groups) = Self::batchnorm_layout(
            op,
            shape,
            x.len(),
            &[&scales, &biases, &rolling_mean, &rolling_variance],
        )?;
        let xh = self.as_half(x)?;
        let n = (row_groups * stride) as f32;
        let block = BIAS_BACKWARD_THREADS as u32;
        let per_channel = launch::elementwise(channels, config);

        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.amp_k.amp_sum_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &xh.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        let mut mean = self.alloc_uninit(channels)?;
        // SAFETY: one thread per channel.
        unsafe {
            self.batchnorm
                .bn_mean_finalize(&self.stream, per_channel, &partial, &mut mean, 1.0 / n)
        }
        .map_err(launch_err)?;
        // SAFETY: as above; every `partial` slot is rewritten.
        unsafe {
            self.amp_k.amp_sq_dev_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &xh.buf,
                &mean,
                &mut partial,
                channels,
                stride,
                row_groups,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        let sum_sq = self.fold_partials(&partial, channels)?;
        let mut variance = self.alloc_uninit(channels)?;
        let mut new_mean = self.alloc_uninit(channels)?;
        let mut new_var = self.alloc_uninit(channels)?;
        // SAFETY: one thread per channel writes its own slot of each output.
        unsafe {
            self.batchnorm.bn_var_finalize(
                &self.stream,
                per_channel,
                &sum_sq,
                &mean,
                &rolling_mean.buf,
                &rolling_variance.buf,
                &mut variance,
                &mut new_mean,
                &mut new_var,
                1.0 / n,
                1.0 / (n - 1.0).max(1.0),
                keep,
            )
        }
        .map_err(launch_err)?;
        let mut x_hat = self.alloc16_uninit(x.len())?;
        let mut out = self.alloc16_uninit(x.len())?;
        // SAFETY: one thread per element writes its own `x_hat` and `out` slot.
        unsafe {
            self.amp_k.amp_bn_normalize(
                &self.stream,
                launch::elementwise(x.len(), config),
                &xh.buf,
                &mean,
                &variance,
                &scales.buf,
                &biases.buf,
                &mut x_hat,
                &mut out,
                channels,
                stride,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        let c = [channels];
        Ok(BatchNormTrainForward {
            output: self.wrap_half(OxideTensorF16::new(out, shape)),
            x_hat: self.wrap_half(OxideTensorF16::new(x_hat, shape)),
            mean: OxideTensorF32::new(mean, &c),
            variance: OxideTensorF32::new(variance, &c),
            rolling_mean: OxideTensorF32::new(new_mean, &c),
            rolling_variance: OxideTensorF32::new(new_var, &c),
        })
    }

    /// Exact BN input gradient from f16 `dy` / `x_hat`; `dscale` / `dbias` are
    /// f32 reductions, `dx` is f16.
    pub(crate) fn batchnorm_train_backward_amp(
        &self,
        output_grad: &OxideTensorF32,
        x_hat: &OxideTensorF32,
        variance: &OxideTensorF32,
        scales: &OxideTensorF32,
        shape: &[usize],
        config: ElementwiseLaunchConfig,
    ) -> Result<(OxideTensorF32, OxideTensorF32, OxideTensorF32), OxideError> {
        let op = "batchnorm_train_backward_f32 (amp)";
        crate::same_len(op, output_grad, x_hat)?;
        let (variance, scales) = (self.f32v(variance)?, self.f32v(scales)?);
        let (channels, stride, row_groups) =
            Self::batchnorm_layout(op, shape, output_grad.len(), &[&variance, &scales])?;
        let (dyh, xhh) = (self.as_half(output_grad)?, self.as_half(x_hat)?);
        let n = (row_groups * stride) as f32;
        let block = BIAS_BACKWARD_THREADS as u32;

        let mut partial = self.alloc_uninit(channels * BIAS_BACKWARD_THREADS)?;
        // SAFETY: `channels` blocks of 256 threads exactly cover `partial`.
        unsafe {
            self.amp_k.amp_sum_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &dyh.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        let dbias = self.fold_partials(&partial, channels)?;
        // SAFETY: as above; every `partial` slot is rewritten.
        unsafe {
            self.amp_k.amp_dot_partial(
                &self.stream,
                launch::blocks_of(channels, block),
                &dyh.buf,
                &xhh.buf,
                &mut partial,
                channels,
                stride,
                row_groups,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        let dscale = self.fold_partials(&partial, channels)?;
        let mut dx = self.alloc16_uninit(output_grad.len())?;
        // SAFETY: one thread per element.
        unsafe {
            self.amp_k.amp_bn_backward_dx(
                &self.stream,
                launch::elementwise(dx.len(), config),
                &dyh.buf,
                &xhh.buf,
                &variance.buf,
                &scales.buf,
                &dbias,
                &dscale,
                &mut dx,
                channels,
                stride,
                n,
                self.half_format().code(),
            )
        }
        .map_err(launch_err)?;
        let c = [channels];
        Ok((
            self.wrap_half(OxideTensorF16::new(dx, shape)),
            OxideTensorF32::new(dscale, &c),
            OxideTensorF32::new(dbias, &c),
        ))
    }
}

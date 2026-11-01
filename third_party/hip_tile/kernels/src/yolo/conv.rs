//! Implicit-GEMM convolution with fused bias + activation.
//!
//! `out[co][oy][ox] = act(bias[co] + sum_{ci,ky,kx} w[co][ci][ky][kx] * in[ci][oy*s+ky-p][ox*s+kx-p])`
//! (zero outside the input). As a GEMM: `A` = weights `[Cout x Cin*k*k]` (row-major, BN
//! already folded by the host), `B` = im2col `[Cin*k*k x OH*OW]` gathered on the fly (no
//! buffer), `C` = `[Cout x OH*OW]`.
//!
//! Operands: `x` = input as `[Cin x H*W]` (any `ld`), `w` = weights tensor `[Cout x Cin*k*k]`,
//! `bias` = `Tensor::vector(ptr, Cout)`, `y` = output `[Cout x OH*OW]` with its own `ld`, so a
//! conv can write into a channel slice of a concat buffer (pass the view at the slice's
//! first channel).
//!
//! One kernel per tile config (`BMxBNxBK`), see [`Config`]. Launch grid is
//! `(ceil(OH*OW / BN), ceil(Cout / BM))`; use [`launch_conv`] on the host.

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const BK: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ConvParams {
    pub cin: u32,
    /// Output channels (`filters`); the weight tensor's row count.
    pub cout: u32,
    pub h: u32,
    pub w: u32,
    /// Kernel size (square).
    pub k: u32,
    pub stride: u32,
    pub pad: u32,
    pub oh: u32,
    pub ow: u32,
    /// 0 = linear, 1 = leaky (0.1).
    pub act: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for ConvParams {}

impl ConvParams {
    /// Output channels; falls back to 0 for the forward-only constructor.
    pub const fn cout(&self) -> u32 {
        self.cout
    }

    pub const LINEAR: u32 = 0;
    pub const LEAKY: u32 = 1;

    /// Output size follows the usual `(in + 2*pad - k) / stride + 1`.
    pub fn new(cin: u32, h: u32, w: u32, k: u32, stride: u32, pad: u32, act: u32) -> Self {
        Self::with_cout(cin, 0, h, w, k, stride, pad, act)
    }

    /// Same, with the output-channel count set (needed by the backward kernels).
    #[allow(clippy::too_many_arguments)]
    pub fn with_cout(cin: u32, cout: u32, h: u32, w: u32, k: u32, stride: u32, pad: u32, act: u32) -> Self {
        Self {
            cin,
            cout,
            h,
            w,
            k,
            stride,
            pad,
            oh: (h + 2 * pad - k) / stride + 1,
            ow: (w + 2 * pad - k) / stride + 1,
            act,
        }
    }
}

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn conv_body<const BM: usize, const BN: usize>(
    x: Tensor<f32>,
    w: Tensor<f32>,
    bias: Tensor<f32>,
    y: TensorMut<f32>,
    p: ConvParams,
) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * BM as u32, bj * BN as u32);
    let k_total = w.cols();
    let kk = p.k * p.k;
    let (xp, ld) = (x.ptr(), x.ld() as usize);
    let n_out = p.oh * p.ow;
    let mut acc = Tile::<f32, BM, BN>::zeros();
    let mut k = 0;
    while k < k_total {
        let ta = w.load_tile::<BM, BK>(r0, k);
        let tb = Tile::<f32, BK, BN>::from_fn(|r, c| {
            let kidx = k + r as u32;
            let pos = c0 + c as u32;
            if kidx >= k_total || pos >= n_out {
                return 0.0;
            }
            let ci = kidx / kk;
            let rem = kidx - ci * kk;
            let ky = rem / p.k;
            let kx = rem - ky * p.k;
            let oy = pos / p.ow;
            let ox = pos - oy * p.ow;
            let iy = (oy * p.stride + ky) as i32 - p.pad as i32;
            let ix = (ox * p.stride + kx) as i32 - p.pad as i32;
            if (iy as u32) < p.h && (ix as u32) < p.w {
                unsafe { *xp.add(ci as usize * ld + (iy as u32 * p.w + ix as u32) as usize) }
            } else {
                0.0
            }
        });
        acc = mma(&ta, &tb, acc);
        k += BK as u32;
    }
    let bp = bias.ptr();
    let cout = y.rows();
    let bt = Tile::<f32, BM, BN>::from_fn(|r, _| {
        let ch = r0 + r as u32;
        if ch < cout {
            unsafe { *bp.add(ch as usize) }
        } else {
            0.0
        }
    });
    let leaky = p.act == ConvParams::LEAKY;
    let out = (acc + bt).map(|v| if leaky && v < 0.0 { 0.1 * v } else { v });
    y.store_tile(r0, c0, &out);
}

macro_rules! conv_kernel {
    ($name:ident, $bm:expr, $bn:expr) => {
        #[kernel(lds = mma_lds_bytes::<f32>($bm, $bn, BK))]
        fn $name(
            x: Tensor<f32>,
            w: Tensor<f32>,
            bias: Tensor<f32>,
            y: TensorMut<f32>,
            p: ConvParams,
        ) {
            conv_body::<$bm, $bn>(x, w, bias, y, p)
        }
    };
}

conv_kernel!(conv_64x64, 64, 64);
conv_kernel!(conv_128x64, 128, 64);
conv_kernel!(conv_32x128, 32, 128);
conv_kernel!(conv_64x32, 64, 32);
conv_kernel!(conv_32x32, 32, 32);

/// Tile configuration (`BM x BN`, `BK = 16`). LDS bytes: see [`Config::lds_bytes`].
#[cfg(not(target_arch = "amdgpu"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Config {
    C64x64,
    C128x64,
    C32x128,
    C64x32,
    C32x32,
}

#[cfg(not(target_arch = "amdgpu"))]
impl Config {
    pub const ALL: [Config; 5] = [
        Self::C64x64,
        Self::C128x64,
        Self::C32x128,
        Self::C64x32,
        Self::C32x32,
    ];

    /// `(BM, BN)`.
    pub const fn tile(self) -> (u32, u32) {
        match self {
            Self::C64x64 => (64, 64),
            Self::C128x64 => (128, 64),
            Self::C32x128 => (32, 128),
            Self::C64x32 => (64, 32),
            Self::C32x32 => (32, 32),
        }
    }

    pub const fn lds_bytes(self) -> u32 {
        match self {
            Self::C64x64 => conv_64x64::LDS_BYTES,
            Self::C128x64 => conv_128x64::LDS_BYTES,
            Self::C32x128 => conv_32x128::LDS_BYTES,
            Self::C64x32 => conv_64x32::LDS_BYTES,
            Self::C32x32 => conv_32x32::LDS_BYTES,
        }
    }

    /// Pick a config from the output shape (see lessons for how it was tuned).
    pub fn choose(cout: u32, n_out: u32) -> Self {
        if n_out >= 2048 {
            if cout <= 32 {
                Self::C32x32
            } else {
                Self::C64x64
            }
        } else if n_out > 64 {
            Self::C64x32
        } else {
            Self::C32x32
        }
    }
}

/// Launch a convolution with the config chosen by [`Config::choose`].
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_conv(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    w: hip_tile::Tensor<f32>,
    bias: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
    p: &ConvParams,
) -> hip_rt::Result<()> {
    let cfg = Config::choose(y.rows(), p.oh * p.ow);
    launch_conv_with(cfg, module, stream, x, w, bias, y, p)
}

/// Launch a convolution with an explicit tile config.
#[cfg(not(target_arch = "amdgpu"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_conv_with(
    cfg: Config,
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    w: hip_tile::Tensor<f32>,
    bias: hip_tile::Tensor<f32>,
    y: hip_tile::TensorMut<f32>,
    p: &ConvParams,
) -> hip_rt::Result<()> {
    assert!(
        x.rows() == p.cin && x.cols() == p.h * p.w,
        "conv: bad input view"
    );
    assert!(
        w.rows() == y.rows() && w.cols() == p.cin * p.k * p.k,
        "conv: bad weights"
    );
    assert!(
        y.cols() == p.oh * p.ow && bias.cols() >= y.rows(),
        "conv: bad output/bias"
    );
    let (bm, bn) = cfg.tile();
    let grid = (y.cols().div_ceil(bn), y.rows().div_ceil(bm));
    let p = *p;
    match cfg {
        Config::C64x64 => conv_64x64::launch(module, stream, grid, x, w, bias, y, p),
        Config::C128x64 => conv_128x64::launch(module, stream, grid, x, w, bias, y, p),
        Config::C32x128 => conv_32x128::launch(module, stream, grid, x, w, bias, y, p),
        Config::C64x32 => conv_64x32::launch(module, stream, grid, x, w, bias, y, p),
        Config::C32x32 => conv_32x32::launch(module, stream, grid, x, w, bias, y, p),
    }
}

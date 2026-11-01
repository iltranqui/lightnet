//! Batch-norm backward for a training step (`batchnorm_layer.cpp:53-66, 134-160`), ported
//! verbatim including Darknet's two quirks: the gradient standardises with `1/(sqrt(var) + 1e-5)`
//! where the forward used `1/sqrt(var + 1e-5)`, and the variance term divides by `n` although the
//! forward variance is unbiased (`/(n-1)`).
//!
//! One block per channel, two passes over the channel: the first reduces `dscale = Σ xhat*dy`,
//! `Σ dy` and `Σ dy*(x − mean)`; the second overwrites `dy` with `dx`.
//! `x` is the pre-normalisation conv sum, `xhat` the normalised value (Darknet's `x_norm`).
//! All tensors are `[batch x spatial]` views of one channel (`ld = C*spatial`).

use hip_tile::KernelArg;
use hip_tile::prelude::*;
use hip_tile_macros::kernel;

pub const TILE_M: usize = 8;
pub const TILE_N: usize = 128;
/// `batchnorm_layer.cpp:446`.
pub const BN_EPS: f32 = 1e-5;

/// `f32::sqrt` is not available on the `no_std` device target; the intrinsic is
/// (`gpu_intrinsics` / `core_intrinsics` are enabled for that target in `hip-tile`).
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
pub(crate) fn sqrtf(x: f32) -> f32 {
    core::intrinsics::sqrt::<f32>(x)
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BnBwdParams {
    pub batch: u32,
    pub spatial: u32,
}

// SAFETY: `repr(C)` struct of `u32`s.
unsafe impl KernelArg for BnBwdParams {}

#[kernel(lds = reduce_lds_bytes::<f32>())]
#[allow(clippy::too_many_arguments)]
fn bn_bwd(
    x: Tensor<f32>,
    xhat: Tensor<f32>,
    dy: TensorMut<f32>,
    mean: Tensor<f32>,
    var: Tensor<f32>,
    scale: Tensor<f32>,
    dscale: TensorMut<f32>,
    p: BnBwdParams,
) {
    let (_, bj) = block_id();
    let ch = bj;
    if ch >= x.rows() {
        return;
    }
    // Channel `ch` of image 0 sits `ch * spatial` elements into a row.
    let coff = ch as usize * p.spatial as usize;
    let (xp, xhp, dyp) = (
        unsafe { x.ptr().add(coff) },
        unsafe { xhat.ptr().add(coff) },
        unsafe { dy.ptr().add(coff) },
    );
    let (xld, dhld, dld) = (x.ld() as usize, xhat.ld() as usize, dy.ld() as usize);
    let m = unsafe { *mean.ptr().add(ch as usize) };
    let v = unsafe { *var.ptr().add(ch as usize) };
    let g = unsafe { *scale.ptr().add(ch as usize) };
    let n = (p.batch * p.spatial) as f32;

    // Pass 1: the three channel sums.
    let mut dscale_sum = 0.0f32;
    let mut sum_dy = 0.0f32;
    let mut sum_dy_x = 0.0f32;
    let mut r = 0u32;
    while r < p.batch {
        let rows = (p.batch - r).min(TILE_M as u32);
        let mut c = 0u32;
        while c < p.spatial {
            let load = |rr: usize, cc: usize| -> (f32, f32, f32) {
                let row = r + rr as u32;
                let col = c + cc as u32;
                if rr as u32 >= rows || row >= p.batch || col >= p.spatial {
                    return (0.0, 0.0, 0.0);
                }
                let i = row as usize;
                let j = col as usize;
                unsafe {
                    (
                        *xhp.add(i * dhld + j),
                        *dyp.add(i * dld + j),
                        *xp.add(i * xld + j) - m,
                    )
                }
            };
            let t_dscale = Tile::<f32, TILE_M, TILE_N>::from_fn(|rr, cc| {
                let (xh, dyv, _) = load(rr, cc);
                xh * dyv
            });
            let t_sum_dy = Tile::<f32, TILE_M, TILE_N>::from_fn(|rr, cc| g * load(rr, cc).1);
            let t_sum_dy_x = Tile::<f32, TILE_M, TILE_N>::from_fn(|rr, cc| {
                let (_, dyv, dx) = load(rr, cc);
                g * dyv * dx
            });
            dscale_sum += t_dscale.sum();
            sum_dy += t_sum_dy.sum();
            sum_dy_x += t_sum_dy_x.sum();
            c += TILE_N as u32;
        }
        r += TILE_M as u32;
    }

    // `(v + eps)^1.5` without `powf` (not available on the device): `(v + eps) * sqrt(v + eps)`.
    let sd = sqrtf(v + BN_EPS);
    let mean_delta = -sum_dy / sd;
    let variance_delta = -0.5 * sum_dy_x / ((v + BN_EPS) * sd);
    let inv_std = 1.0 / (sqrtf(v) + BN_EPS);

    // Pass 2: overwrite `dy` with dx (masked rows/columns are skipped by `store_tile`).
    let mut r = 0u32;
    while r < p.batch {
        let mut c = 0u32;
        while c < p.spatial {
            let t = Tile::<f32, TILE_M, TILE_N>::from_fn(|rr, cc| {
                let row = r + rr as u32;
                let col = c + cc as u32;
                if row >= p.batch || col >= p.spatial {
                    return 0.0;
                }
                let i = row as usize;
                let j = col as usize;
                let dyv = unsafe { *dyp.add(i * dld + j) };
                let xv = unsafe { *xp.add(i * xld + j) };
                g * dyv * inv_std + variance_delta * 2.0 * (xv - m) / n + mean_delta / n
            });
            // A per-channel view, so the store lands in this channel only.
            let dview = TensorMut::new(dyp, p.batch, p.spatial, dy.ld());
            dview.store_tile(r, c, &t);
            c += TILE_N as u32;
        }
        r += TILE_M as u32;
    }

    if thread_id() == 0 {
        unsafe { *dscale.ptr().add(ch as usize) += dscale_sum };
    }
}

/// `x`, `xhat`, `dy`: one channel as `[batch x spatial]` (`ld = C*spatial`); `mean`, `var`,
/// `scale`, `dscale`: `[C]`. `dscale` is accumulated into; `dy` is overwritten with `dx`.
#[cfg(not(target_arch = "amdgpu"))]
#[allow(clippy::too_many_arguments)]
pub fn launch_bn_bwd(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    x: hip_tile::Tensor<f32>,
    xhat: hip_tile::Tensor<f32>,
    dy: hip_tile::TensorMut<f32>,
    mean: hip_tile::Tensor<f32>,
    var: hip_tile::Tensor<f32>,
    scale: hip_tile::Tensor<f32>,
    dscale: hip_tile::TensorMut<f32>,
    p: &BnBwdParams,
) -> hip_rt::Result<()> {
    assert_eq!(x.rows(), dy.rows(), "bn_bwd: bad dy");
    assert_eq!(x.rows(), xhat.rows(), "bn_bwd: bad xhat");
    assert_eq!(x.cols(), p.spatial, "bn_bwd: bad spatial");
    assert_eq!(dscale.cols(), x.rows(), "bn_bwd: bad dscale");
    let grid = (x.rows(), 1);
    bn_bwd::launch(module, stream, grid, x, xhat, dy, mean, var, scale, dscale, *p)
}

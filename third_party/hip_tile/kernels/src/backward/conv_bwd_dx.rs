//! Input gradient of a convolution, written as a **transpose-convolution gather**: no scatter,
//! no atomics, fully deterministic.
//!
//! `dX[b][ci][iy][ix] = Σ_{co,ky,kx} W[co][ci][ky][kx] · δ[b][co][oy][ox]`,
//! with `oy = (iy + p − ky)/s`, `ox = (ix + p − kx)/s`, only when the division is exact and the
//! output position exists.
//!
//! Per image this is `k²` GEMMs against `Co`, sharing one register accumulator for the block's
//! `[Cin × positions]` tile — so all `ky`, `kx` fold into that tile and image `b`'s slice of `dx`
//! (at `+b·Cin·ld`) is accumulated once per image (`store_tile_add`, because several branches may feed the same
//! input buffer).
//!
//! Both operands are gathered with `from_fn`: the weights because a `(ky, kx)` slice of
//! `[co][ci][ky][kx]` is strided in both axes, `δ` because its positions are shifted per tap.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

use super::ConvBwdParams;

pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 16;


#[kernel(lds = mma_lds_bytes::<f32>(BM, BN, BK))]
fn conv_bwd_dx(delta: Tensor<f32>, w: Tensor<f32>, dx: TensorMut<f32>, p: ConvBwdParams) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * BM as u32, bj * BN as u32);
    let conv = p.conv;
    let kk = conv.k * conv.k;
    let (co_tot, ci_tot) = (conv.cout, conv.cin);
    let n_in = conv.h * conv.w;
    let n_out = conv.oh * conv.ow;
    let (wp, dld) = (w.ptr(), delta.ld() as usize);
    for b in 0..p.batch {
        // dX is per image: each image gets its own accumulator and its own output slice.
        let mut acc = Tile::<f32, BM, BN>::zeros();
        let dbase = unsafe { delta.ptr().add(b as usize * co_tot as usize * dld) };
        for ky in 0..conv.k {
            for kx in 0..conv.k {
                let tap = ky * conv.k + kx;
                let mut co = 0u32;
                while co < co_tot {
                    // W as `[ci x co]` for this tap: row stride k*k, column stride Ci*k*k.
                    let ta = Tile::<f32, BM, BK>::from_fn(|r, c| {
                        let ci = r0 + r as u32;
                        let cc = co + c as u32;
                        if ci >= ci_tot || cc >= co_tot {
                            0.0
                        } else {
                            unsafe { *wp.add((cc * ci_tot * kk + ci * kk + tap) as usize) }
                        }
                    });
                    // δ as `[co x shifted positions]`.
                    let tb = Tile::<f32, BK, BN>::from_fn(|r, c| {
                        let cc = co + r as u32;
                        let pos = c0 + c as u32;
                        if cc >= co_tot || pos >= n_in {
                            return 0.0;
                        }
                        let (iy, ix) = (pos / conv.w, pos % conv.w);
                        let ny = iy as i32 + conv.pad as i32 - ky as i32;
                        let nx = ix as i32 + conv.pad as i32 - kx as i32;
                        if ny < 0 || nx < 0 {
                            return 0.0;
                        }
                        let (oy, ox) = (ny / conv.stride as i32, nx / conv.stride as i32);
                        if ny % conv.stride as i32 != 0 || nx % conv.stride as i32 != 0 {
                            return 0.0;
                        }
                        if oy >= conv.oh as i32 || ox >= conv.ow as i32 {
                            return 0.0;
                        }
                        unsafe { *dbase.add(cc as usize * dld + (oy as u32 * conv.ow + ox as u32) as usize) }
                    });
                    acc = mma(&ta, &tb, acc);
                    co += BK as u32;
                }
            }
        }
        let _ = n_out;
        let dxb = TensorMut::new(
            unsafe { dx.ptr().add(b as usize * ci_tot as usize * dx.ld() as usize) },
            dx.rows(),
            dx.cols(),
            dx.ld(),
        );
        dxb.store_tile_add(r0, c0, &acc);
    }
}

/// `delta: [Cout x OH*OW]`, `w: [Cout x Cin*k*k]`, `dx: [Cin x H*W]` (accumulated into).
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_conv_bwd_dx(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    delta: hip_tile::Tensor<f32>,
    w: hip_tile::Tensor<f32>,
    dx: hip_tile::TensorMut<f32>,
    p: &ConvBwdParams,
) -> hip_rt::Result<()> {
    let conv = p.conv;
    assert_eq!(w.rows(), conv.cout, "conv_bwd_dx: bad w rows");
    assert_eq!(w.cols(), conv.cin * conv.k * conv.k, "conv_bwd_dx: bad w cols");
    assert_eq!(delta.rows(), conv.cout, "conv_bwd_dx: bad delta rows");
    assert_eq!(delta.cols(), conv.oh * conv.ow, "conv_bwd_dx: bad delta cols");
    assert_eq!(dx.rows(), conv.cin, "conv_bwd_dx: bad dx rows");
    assert_eq!(dx.cols(), conv.h * conv.w, "conv_bwd_dx: bad dx cols");
    let grid = (dx.cols().div_ceil(BN as u32), dx.rows().div_ceil(BM as u32));
    conv_bwd_dx::launch(module, stream, grid, delta, w, dx, *p)
}

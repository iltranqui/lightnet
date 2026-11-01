//! Weight/bias gradient of an implicit-GEMM convolution.
//!
//! `dW[co][ci][ky][kx] = Σ_b Σ_{oy,ox} δ[b][co][oy][ox] · x[b][ci][oy·s+ky−p][ox·s+kx−p]`
//!
//! As a GEMM per image: `A = δ` `[Cout x OH·OW]`, `B = im2col(x)ᵀ` `[OH·OW x Cin·k²]`, and the
//! result is a `[Cout x Cin·k²]` tile. The block owns one output tile and walks the **whole batch
//! inside its k-loop**, so `dW` is accumulated in registers and never needs atomics or a temporary
//! per-image buffer. The existing contents of `dw` are the starting accumulator (momentum buffer
//! semantics: Darknet's backward adds into `weight_updates`).
//!
//! `x`: `[Cin x H·W]` (image b at `+b·Cin·ld`), `delta`: `[Cout x OH·OW]` (image b at
//! `+b·Cout·ld`), `dw`: `[Cout x Cin·k²]`.

use hip_tile::prelude::*;
use hip_tile_macros::kernel;

use super::ConvBwdParams;

pub const BM: usize = 64;
pub const BN: usize = 64;
pub const BK: usize = 16;


#[kernel(lds = mma_lds_bytes::<f32>(BM, BN, BK))]
fn conv_bwd_dw(delta: Tensor<f32>, x: Tensor<f32>, dw: TensorMut<f32>, p: ConvBwdParams) {
    let (bi, bj) = block_id();
    let (r0, c0) = (bi * BM as u32, bj * BN as u32);
    let conv = p.conv;
    let kk = conv.k * conv.k;
    let k_total = conv.cin * kk;
    let n_out = conv.oh * conv.ow;
    let (cout, xld) = (delta.rows(), x.ld() as usize);
    // Start from what is already there: momentum buffers accumulate.
    let mut acc = dw.load_tile::<BM, BN>(r0, c0);
    for b in 0..p.batch {
        let dbase = unsafe { delta.ptr().add(b as usize * cout as usize * delta.ld() as usize) };
        let xbase = unsafe { x.ptr().add(b as usize * conv.cin as usize * xld) };
        let mut k0 = 0u32;
        while k0 < n_out {
            let ta = Tensor::new(dbase, cout, n_out, delta.ld()).load_tile::<BM, BK>(r0, k0);
            let tb = Tile::<f32, BK, BN>::from_fn(|r, c| {
                let pos = k0 + r as u32;
                let kidx = c0 + c as u32;
                if pos >= n_out || kidx >= k_total {
                    return 0.0;
                }
                let ci = kidx / kk;
                let rem = kidx - ci * kk;
                let ky = rem / conv.k;
                let kx = rem - ky * conv.k;
                let oy = pos / conv.ow;
                let ox = pos - oy * conv.ow;
                let iy = (oy * conv.stride + ky) as i32 - conv.pad as i32;
                let ix = (ox * conv.stride + kx) as i32 - conv.pad as i32;
                if iy >= 0 && ix >= 0 && (iy as u32) < conv.h && (ix as u32) < conv.w {
                    unsafe { *xbase.add(ci as usize * xld + (iy as u32 * conv.w + ix as u32) as usize) }
                } else {
                    0.0
                }
            });
            acc = mma(&ta, &tb, acc);
            k0 += BK as u32;
        }
    }
    // `acc` already holds the buffer's previous contents (loaded above), so this **overwrites**.
    // Using `store_tile_add` here would double the momentum buffer.
    dw.store_tile(r0, c0, &acc);
}

/// `x: [Cin x H*W]`, `delta: [Cout x OH*OW]`, `dw: [Cout x Cin*k*k]` (accumulated into).
#[cfg(not(target_arch = "amdgpu"))]
pub fn launch_conv_bwd_dw(
    module: &hip_rt::Module,
    stream: Option<&hip_rt::Stream>,
    delta: hip_tile::Tensor<f32>,
    x: hip_tile::Tensor<f32>,
    dw: hip_tile::TensorMut<f32>,
    p: &ConvBwdParams,
) -> hip_rt::Result<()> {
    let conv = p.conv;
    assert_eq!(x.rows(), conv.cin, "conv_bwd_dw: bad x rows");
    assert_eq!(x.cols(), conv.h * conv.w, "conv_bwd_dw: bad x cols");
    assert_eq!(delta.rows(), conv.cout(), "conv_bwd_dw: bad delta rows");
    assert_eq!(delta.cols(), conv.oh * conv.ow, "conv_bwd_dw: bad delta cols");
    assert_eq!(dw.rows(), conv.cout(), "conv_bwd_dw: bad dw rows");
    assert_eq!(dw.cols(), conv.cin * conv.k * conv.k, "conv_bwd_dw: bad dw cols");
    let grid = (dw.cols().div_ceil(BN as u32), dw.rows().div_ceil(BM as u32));
    conv_bwd_dw::launch(module, stream, grid, delta, x, dw, *p)
}

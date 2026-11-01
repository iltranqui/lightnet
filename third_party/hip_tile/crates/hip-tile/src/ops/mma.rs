//! Tile matrix multiply-accumulate.
//!
//! RDNA2 has no matrix cores, so `mma` stages both operands in LDS and each thread
//! accumulates its register fragment of `C` with FMAs (a register-blocked outer
//! product; a 64x64 `C` gives each thread a 4x4 block).

use crate::arch::sync_block;
use crate::config::{LDS_PAD, mma_lds_bytes};
use crate::element::Element;
use crate::layout::{FRAG_MAX, thread_cols, thread_coords, thread_rows};
use crate::smem;
use crate::tile::Tile;

/// Write `tile` into LDS at `dst`, row-major with padded rows.
#[inline(always)]
fn stage<T: Element, const M: usize, const N: usize>(dst: *mut T, tile: &Tile<T, M, N>) {
    Tile::<T, M, N>::for_each_owned(|s, r, c| unsafe {
        *dst.add(r * (N + LDS_PAD) + c) = tile.frag[s]
    });
}

/// `acc + a * b`.
#[inline(always)]
pub fn mma<T: Element, const M: usize, const N: usize, const K: usize>(
    a: &Tile<T, M, K>,
    b: &Tile<T, K, N>,
    mut acc: Tile<T, M, N>,
) -> Tile<T, M, N> {
    smem::require(mma_lds_bytes::<T>(M, N, K));
    let sa = smem::base::<T>();
    // SAFETY: `require` checked the region holds both padded operands.
    let sb = unsafe { sa.add(M * (K + LDS_PAD)) };

    sync_block();
    stage(sa, a);
    stage(sb, b);
    sync_block();

    let (ty, tx) = thread_coords::<M, N>();
    let (tr, tc) = (thread_rows(M, N), thread_cols(M, N));
    let (fr, fc) = (Tile::<T, M, N>::FR, Tile::<T, M, N>::FC);
    let mut av = [T::ZERO; FRAG_MAX];
    let mut bv = [T::ZERO; FRAG_MAX];
    for k in 0..K {
        for i in 0..fr {
            av[i] = unsafe { *sa.add((ty + tr * i) * (K + LDS_PAD) + k) };
        }
        for j in 0..fc {
            bv[j] = unsafe { *sb.add(k * (N + LDS_PAD) + tx + tc * j) };
        }
        for i in 0..fr {
            for j in 0..fc {
                let s = i * fc + j;
                acc.frag[s] = av[i].mul_add(bv[j], acc.frag[s]);
            }
        }
    }
    acc
}

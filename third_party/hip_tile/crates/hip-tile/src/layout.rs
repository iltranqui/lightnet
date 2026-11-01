//! How an `M x N` tile is spread over the 256 threads of a block.
//!
//! Threads form a `TR x TC` grid (`TR * TC = 256`). `TC` is the largest power of two
//! that is at most `256 / min(M, 16)` and divides `N`; `TR = 256 / TC` and `M` must be a
//! multiple of `TR`. Thread `(ty, tx)` owns elements `(ty + TR * i, tx + TC * j)`:
//! neighbouring threads own neighbouring columns, so row-major global accesses
//! coalesce. Each thread keeps `M * N / 256` elements in registers.

use crate::arch::thread_id;
use crate::config::{BLOCK_DIM, BLOCK_THREADS};

const THREADS: usize = BLOCK_THREADS as usize;

/// Columns of the thread grid for an `m x n` tile.
pub const fn thread_cols(m: usize, n: usize) -> usize {
    let tr0 = if m >= BLOCK_DIM { BLOCK_DIM } else { m };
    let mut tc = THREADS / tr0;
    while tc > 1 && n % tc != 0 {
        tc /= 2;
    }
    tc
}

/// Rows of the thread grid for an `m x n` tile.
pub const fn thread_rows(m: usize, n: usize) -> usize {
    THREADS / thread_cols(m, n)
}

/// Elements of an `m x n` tile held by each thread.
pub const fn frag_len(m: usize, n: usize) -> usize {
    m * n / THREADS
}

/// Capacity of the register array backing a fragment. Only `frag_len(M, N)` slots are
/// ever touched; loops over them are fully unrolled, so unused slots cost nothing.
pub const FRAG_MAX: usize = 64;

/// Compile-time check that an `m x n` tile maps exactly onto the block.
pub const fn check_shape(m: usize, n: usize) {
    assert!(m > 0 && n > 0, "tile dimensions must be non-zero");
    assert!(
        m * n % THREADS == 0,
        "tile must hold a multiple of 256 elements"
    );
    assert!(
        m * n / THREADS <= FRAG_MAX,
        "tile too large: more than FRAG_MAX elements per thread"
    );
    let (tr, tc) = (thread_rows(m, n), thread_cols(m, n));
    assert!(
        n % tc == 0 && m % tr == 0,
        "tile shape does not map onto the 256-thread grid"
    );
}

/// Position of the calling thread in the thread grid of an `M x N` tile.
#[inline(always)]
pub fn thread_coords<const M: usize, const N: usize>() -> (usize, usize) {
    let t = thread_id() as usize;
    (t / thread_cols(M, N), t % thread_cols(M, N))
}

//! Whole-tile reductions.

use crate::arch::{sync_block, thread_id};
use crate::config::{BLOCK_THREADS, reduce_lds_bytes};
use crate::element::Element;
use crate::layout::frag_len;
use crate::smem;
use crate::tile::Tile;

impl<T: Element, const M: usize, const N: usize> Tile<T, M, N> {
    /// Fold all elements with `f` (associative and commutative); every thread gets the result.
    #[inline(always)]
    pub fn reduce(&self, f: impl Fn(T, T) -> T) -> T {
        let mut acc = self.frag[0];
        for s in 1..frag_len(M, N) {
            acc = f(acc, self.frag[s]);
        }

        // Tree reduction of the 256 per-thread partials in LDS.
        smem::require(reduce_lds_bytes::<T>());
        let lds = smem::base::<T>();
        let t = thread_id() as usize;
        sync_block();
        unsafe { *lds.add(t) = acc };
        let mut half = BLOCK_THREADS as usize / 2;
        while half > 0 {
            sync_block();
            if t < half {
                unsafe { *lds.add(t) = f(*lds.add(t), *lds.add(t + half)) };
            }
            half /= 2;
        }
        sync_block();
        unsafe { *lds }
    }

    #[inline(always)]
    pub fn sum(&self) -> T {
        self.reduce(|a, b| a + b)
    }

    #[inline(always)]
    pub fn max(&self) -> T {
        self.reduce(|a, b| if a > b { a } else { b })
    }
}

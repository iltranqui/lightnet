//! Tile transpose through LDS.

use crate::arch::sync_block;
use crate::config::{LDS_PAD, transpose_lds_bytes};
use crate::element::Element;
use crate::smem;
use crate::tile::Tile;

impl<T: Element, const M: usize, const N: usize> Tile<T, M, N> {
    /// The `N x M` transpose. Threads exchange elements through LDS.
    #[inline(always)]
    pub fn transpose(&self) -> Tile<T, N, M> {
        smem::require(transpose_lds_bytes::<T>(M, N));
        let lds = smem::base::<T>();
        sync_block();
        Self::for_each_owned(|s, r, c| unsafe { *lds.add(c * (M + LDS_PAD) + r) = self.frag[s] });
        sync_block();
        let mut out = Tile::<T, N, M>::zeros();
        Tile::<T, N, M>::for_each_owned(|s, r, c| {
            out.frag[s] = unsafe { *lds.add(r * (M + LDS_PAD) + c) }
        });
        out
    }
}

//! Global memory <-> tile transfers.
//!
//! Coordinates are element offsets of the tile's top-left corner. Elements outside the
//! tensor read as zero and are not written, so ragged edges need no special casing.

use crate::element::Element;
use crate::tensor::{Tensor, TensorMut};
use crate::tile::Tile;

#[inline(always)]
fn fits(r0: u32, c0: u32, m: usize, n: usize, rows: u32, cols: u32) -> bool {
    r0 as usize + m <= rows as usize && c0 as usize + n <= cols as usize
}

impl<T: Element> Tensor<T> {
    /// Load the `M x N` tile whose top-left element is `(r0, c0)`.
    #[inline(always)]
    pub fn load_tile<const M: usize, const N: usize>(&self, r0: u32, c0: u32) -> Tile<T, M, N> {
        let mut t = Tile::<T, M, N>::zeros();
        let (ld, rows, cols) = (
            self.ld() as usize,
            self.rows() as usize,
            self.cols() as usize,
        );
        // SAFETY: the pointer covers `rows x cols` with stride `ld`; masked reads stay inside.
        let base = unsafe { self.ptr().add(r0 as usize * ld + c0 as usize) };
        // `fits` is uniform across the block, so the branch does not diverge.
        if fits(r0, c0, M, N, self.rows(), self.cols()) {
            Tile::<T, M, N>::for_each_owned(|s, r, c| t.frag[s] = unsafe { *base.add(r * ld + c) });
        } else {
            let (rmax, cmax) = (
                rows.saturating_sub(r0 as usize),
                cols.saturating_sub(c0 as usize),
            );
            Tile::<T, M, N>::for_each_owned(|s, r, c| {
                if r < rmax && c < cmax {
                    t.frag[s] = unsafe { *base.add(r * ld + c) };
                }
            });
        }
        t
    }
}

impl<T: Element> TensorMut<T> {
    /// Load the `M x N` tile whose top-left element is `(r0, c0)`.
    #[inline(always)]
    pub fn load_tile<const M: usize, const N: usize>(&self, r0: u32, c0: u32) -> Tile<T, M, N> {
        self.as_tensor().load_tile(r0, c0)
    }

    /// Store `tile` with its top-left element at `(r0, c0)`.
    #[inline(always)]
    pub fn store_tile<const M: usize, const N: usize>(
        &self,
        r0: u32,
        c0: u32,
        tile: &Tile<T, M, N>,
    ) {
        let (ld, rows, cols) = (
            self.ld() as usize,
            self.rows() as usize,
            self.cols() as usize,
        );
        // SAFETY: as in `load_tile`; masked writes stay inside the tensor.
        let base = unsafe { self.ptr().add(r0 as usize * ld + c0 as usize) };
        if fits(r0, c0, M, N, self.rows(), self.cols()) {
            Tile::<T, M, N>::for_each_owned(|s, r, c| unsafe {
                *base.add(r * ld + c) = tile.frag[s]
            });
        } else {
            let (rmax, cmax) = (
                rows.saturating_sub(r0 as usize),
                cols.saturating_sub(c0 as usize),
            );
            Tile::<T, M, N>::for_each_owned(|s, r, c| {
                if r < rmax && c < cmax {
                    unsafe { *base.add(r * ld + c) = tile.frag[s] };
                }
            });
        }
    }

}

/// Accumulating stores need `AddAssign`; kept in their own impl so `load_tile`/`store_tile`
/// stay available for every `Element`.
impl<T: Element + core::ops::AddAssign> TensorMut<T> {
    /// Store `tile` **added** to the destination (its top-left element at `(r0, c0)`).
    ///
    /// The accumulating counterpart of [`Self::store_tile`]; needed wherever several blocks or
    /// several layers contribute to one gradient buffer (route/concat fan-in, `dX` accumulation).
    #[inline(always)]
    pub fn store_tile_add<const M: usize, const N: usize>(
        &self,
        r0: u32,
        c0: u32,
        tile: &Tile<T, M, N>,
    ) {
        let (ld, rows, cols) = (
            self.ld() as usize,
            self.rows() as usize,
            self.cols() as usize,
        );
        // SAFETY: as in `store_tile`; masked writes stay inside the tensor.
        let base = unsafe { self.ptr().add(r0 as usize * ld + c0 as usize) };
        if fits(r0, c0, M, N, self.rows(), self.cols()) {
            Tile::<T, M, N>::for_each_owned(|s, r, c| unsafe {
                *base.add(r * ld + c) += tile.frag[s]
            });
        } else {
            let (rmax, cmax) = (
                rows.saturating_sub(r0 as usize),
                cols.saturating_sub(c0 as usize),
            );
            Tile::<T, M, N>::for_each_owned(|s, r, c| {
                if r < rmax && c < cmax {
                    unsafe { *base.add(r * ld + c) += tile.frag[s] };
                }
            });
        }
    }
}

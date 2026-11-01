//! `Tile<T, M, N>`: an `M x N` block-wide value held in registers.
//!
//! Every thread of the block holds its fragment (see [`layout`](crate::layout)). Tile
//! code must be reached by all 256 threads with the same tile coordinates, like
//! cuTile's block-level programs.

use core::ops::{Add, Mul, Sub};

use crate::element::Element;
use crate::layout::{FRAG_MAX, check_shape, frag_len, thread_cols, thread_coords, thread_rows};

pub struct Tile<T: Element, const M: usize, const N: usize> {
    pub(crate) frag: [T; FRAG_MAX],
}

impl<T: Element, const M: usize, const N: usize> Clone for Tile<T, M, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Element, const M: usize, const N: usize> Copy for Tile<T, M, N> {}

impl<T: Element, const M: usize, const N: usize> Tile<T, M, N> {
    /// Per-thread fragment columns.
    pub(crate) const FC: usize = N / thread_cols(M, N);
    /// Per-thread fragment rows.
    pub(crate) const FR: usize = M / thread_rows(M, N);

    /// Tile with every element equal to `value`.
    #[inline(always)]
    pub fn splat(value: T) -> Self {
        const { check_shape(M, N) };
        Self {
            frag: [value; FRAG_MAX],
        }
    }

    #[inline(always)]
    pub fn zeros() -> Self {
        Self::splat(T::ZERO)
    }

    /// Tile whose element `(row, col)` is `f(row, col)`.
    #[inline(always)]
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut t = Self::zeros();
        Self::for_each_owned(|s, r, c| t.frag[s] = f(r, c));
        t
    }

    /// Apply `f` to every element.
    #[inline(always)]
    pub fn map<U: Element>(self, mut f: impl FnMut(T) -> U) -> Tile<U, M, N> {
        let mut out = Tile::<U, M, N>::zeros();
        for s in 0..frag_len(M, N) {
            out.frag[s] = f(self.frag[s]);
        }
        out
    }

    /// Combine two tiles element by element.
    #[inline(always)]
    pub fn zip<U: Element>(self, other: Self, mut f: impl FnMut(T, T) -> U) -> Tile<U, M, N> {
        let mut out = Tile::<U, M, N>::zeros();
        for s in 0..frag_len(M, N) {
            out.frag[s] = f(self.frag[s], other.frag[s]);
        }
        out
    }

    /// Multiply every element by `s`.
    #[inline(always)]
    pub fn scale(self, s: T) -> Self {
        self.map(|x| x * s)
    }

    /// Visit the calling thread's fragment as `(slot, row, col)` in tile coordinates.
    #[inline(always)]
    pub(crate) fn for_each_owned(mut f: impl FnMut(usize, usize, usize)) {
        let (ty, tx) = thread_coords::<M, N>();
        let (tr, tc) = (thread_rows(M, N), thread_cols(M, N));
        for i in 0..Self::FR {
            for j in 0..Self::FC {
                f(i * Self::FC + j, ty + tr * i, tx + tc * j);
            }
        }
    }
}

macro_rules! elementwise_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: Element, const M: usize, const N: usize> $trait for Tile<T, M, N>
        {
            type Output = Self;

            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                self.zip(rhs, |a, b| a $op b)
            }
        }
    };
}

elementwise_op!(Add, add, +);
elementwise_op!(Sub, sub, -);
elementwise_op!(Mul, mul, *);

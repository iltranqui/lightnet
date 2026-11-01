//! Scalar types a [`Tile`](crate::tile::Tile) can hold.

use core::ops::{Add, Div, Mul, Neg, Sub};

pub trait Element:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + 'static
{
    const ZERO: Self;
    const ONE: Self;

    /// Fused `self * b + c`, a single rounding.
    fn mul_add(self, b: Self, c: Self) -> Self;
}

macro_rules! float_element {
    ($($t:ty),*) => {$(
        impl Element for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;

            #[inline(always)]
            fn mul_add(self, b: Self, c: Self) -> Self {
                core::intrinsics::fma(self, b, c)
            }
        }
    )*};
}

float_element!(f32, f64);

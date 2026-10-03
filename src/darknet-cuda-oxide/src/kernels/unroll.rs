//! Compile-time loop unrolling for device code.
//!
//! The device pipeline does not unroll `while` loops, so per-thread
//! accumulator / fragment arrays indexed by a loop counter are demoted to
//! local memory (one `ld.local`/`st.local` per FMA).  `unroll!(N, i, { .. })`
//! expands the body `N` times with `i` bound to a constant, so every array
//! index is a constant and the arrays stay in registers.

macro_rules! unroll {
    (2, $i:ident, $b:block) => {{
        { let $i = 0usize; $b }
        { let $i = 1usize; $b }
    }};
    (4, $i:ident, $b:block) => {{
        unroll!(2, $i, $b);
        { let $i = 2usize; $b }
        { let $i = 3usize; $b }
    }};
    (8, $i:ident, $b:block) => {{
        unroll!(4, $i, $b);
        { let $i = 4usize; $b }
        { let $i = 5usize; $b }
        { let $i = 6usize; $b }
        { let $i = 7usize; $b }
    }};
    (16, $i:ident, $b:block) => {{
        unroll!(8, $i, $b);
        { let $i = 8usize; $b }
        { let $i = 9usize; $b }
        { let $i = 10usize; $b }
        { let $i = 11usize; $b }
        { let $i = 12usize; $b }
        { let $i = 13usize; $b }
        { let $i = 14usize; $b }
        { let $i = 15usize; $b }
    }};
}

pub(crate) use unroll;

//! Compile-time launch configuration shared by host launchers and device code.

/// Threads per block for every tile kernel: 8 waves of 32 on RDNA.
pub const BLOCK_THREADS: u32 = 256;

/// Threads per block, laid out as a `BLOCK_DIM x BLOCK_DIM` grid by tile ops.
pub const BLOCK_DIM: usize = 16;

/// Extra elements added to each LDS row to avoid bank conflicts.
pub const LDS_PAD: usize = 4;

/// LDS bytes used by one staged `rows x cols` tile of `T` (padded rows).
pub const fn lds_tile_bytes<T>(rows: usize, cols: usize) -> u32 {
    (rows * (cols + LDS_PAD) * core::mem::size_of::<T>()) as u32
}

/// LDS bytes `mma` needs for `M x K` times `K x N` operands of `T`.
pub const fn mma_lds_bytes<T>(m: usize, n: usize, k: usize) -> u32 {
    lds_tile_bytes::<T>(m, k) + lds_tile_bytes::<T>(k, n)
}

/// LDS bytes `Tile::transpose` needs for an `M x N` tile of `T`.
pub const fn transpose_lds_bytes<T>(m: usize, n: usize) -> u32 {
    lds_tile_bytes::<T>(n, m)
}

/// LDS bytes the tile reductions need for `T`.
pub const fn reduce_lds_bytes<T>() -> u32 {
    BLOCK_THREADS * core::mem::size_of::<T>() as u32
}

/// LDS bytes for a kernel using several LDS ops: they share the region, so the largest wins.
pub const fn lds_max(sizes: &[u32]) -> u32 {
    let mut i = 0;
    let mut m = 0;
    while i < sizes.len() {
        if sizes[i] > m {
            m = sizes[i];
        }
        i += 1;
    }
    m
}

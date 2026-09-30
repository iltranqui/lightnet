//! Launch-geometry helpers shared by all kernels.

use cuda_core::simt::LaunchConfig;
use darknet_cuda::{DEFAULT_TILE_1D, ElementwiseLaunchConfig};

/// Thread-block size for an elementwise launch.  `tile_len` is honoured when
/// it is a power of two in `32..=1024`; anything else falls back to the
/// default so odd configs cannot produce an invalid launch.
pub fn block_size(config: ElementwiseLaunchConfig) -> u32 {
    let t = config.tile_len;
    if t.is_power_of_two() && (32..=1024).contains(&t) {
        t as u32
    } else {
        DEFAULT_TILE_1D as u32
    }
}

/// 1-D launch covering `n` elements (`n` need not be a multiple of the block).
pub fn elementwise(n: usize, config: ElementwiseLaunchConfig) -> LaunchConfig {
    let block = block_size(config);
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(block).max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `blocks` blocks of exactly `threads` threads each.
pub fn blocks_of(blocks: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

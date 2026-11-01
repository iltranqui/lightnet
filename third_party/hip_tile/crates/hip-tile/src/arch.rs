//! Thin wrappers over `core::arch::amdgpu` used by the tile ops.

use core::arch::amdgpu;
use core::sync::atomic::{Ordering, fence};

/// Linear thread index within the block (`0..BLOCK_THREADS`).
#[inline(always)]
pub fn thread_id() -> u32 {
    amdgpu::workitem_id_x()
}

/// Block index in the grid, as `(row, col)` = `(y, x)`.
#[inline(always)]
pub fn block_id() -> (u32, u32) {
    (amdgpu::workgroup_id_y(), amdgpu::workgroup_id_x())
}

/// Block-wide barrier that also orders LDS and global memory accesses.
#[inline(always)]
pub fn sync_block() {
    // Pure Rust, no inline asm (project rule). `fence` is system-scope here, so LLVM adds
    // `buffer_gl0_inv`/`buffer_gl1_inv`; about 5% slower gemm than an asm LDS-only barrier.
    fence(Ordering::Release);
    amdgpu::s_barrier();
    fence(Ordering::Acquire);
}

//! Workgroup memory (LDS).
//!
//! Rust cannot declare static LDS, so every kernel uses the launch-sized region and
//! the host passes its size (`#[kernel(lds = ...)]`). Tile ops that stage through LDS
//! all start at offset 0 and follow one rule: begin with [`sync_block`] (so earlier
//! readers are done), end right after their last LDS read.
//!
//! [`sync_block`]: crate::arch::sync_block

use core::intrinsics::gpu::{amdgpu_dispatch_ptr, gpu_launch_sized_workgroup_mem};

/// Byte offset of `group_segment_size` in the HSA kernel dispatch packet.
const GROUP_SEGMENT_SIZE_OFFSET: usize = 28;

/// Start of this block's LDS region.
#[inline(always)]
pub fn base<T>() -> *mut T {
    gpu_launch_sized_workgroup_mem::<T>()
}

/// LDS bytes allocated to this launch.
#[inline(always)]
pub fn size() -> u32 {
    // SAFETY: the dispatch packet is valid for the whole kernel and holds a u32 here.
    unsafe {
        amdgpu_dispatch_ptr()
            .cast::<u8>()
            .add(GROUP_SEGMENT_SIZE_OFFSET)
            .cast::<u32>()
            .read()
    }
}

/// Trap if the launch did not allocate at least `bytes` of LDS.
#[inline(always)]
pub fn require(bytes: u32) {
    if size() < bytes {
        core::process::abort_immediate();
    }
}

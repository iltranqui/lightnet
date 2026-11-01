//! cuTile-style tile programming for AMD GPUs, in pure Rust.
//!
//! Kernels operate on whole tiles (`load_tile`, `mma`, `store_tile`); this crate maps
//! each tile onto the threads, registers and LDS of a 256-thread block at compile time.
//!
//! The crate builds for both targets. On the host it only provides the kernel ABI
//! types ([`Tensor`], [`TensorMut`], [`KernelArg`], [`config`]); on
//! `target_arch = "amdgpu"` it adds the device-side tile operations.

#![no_std]
#![cfg_attr(target_arch = "amdgpu", allow(internal_features))]
#![cfg_attr(
    target_arch = "amdgpu",
    feature(
        stdarch_amdgpu,
        gpu_intrinsics,
        gpu_launch_sized_workgroup_mem,
        core_intrinsics,
        abort_immediate
    )
)]

pub mod arg;
pub mod config;
pub mod tensor;

#[cfg(target_arch = "amdgpu")]
pub mod arch;
#[cfg(target_arch = "amdgpu")]
pub mod element;
#[cfg(target_arch = "amdgpu")]
pub mod layout;
#[cfg(target_arch = "amdgpu")]
pub mod ops;
#[cfg(target_arch = "amdgpu")]
pub mod smem;
#[cfg(target_arch = "amdgpu")]
pub mod tile;

pub use arg::KernelArg;
pub use tensor::{Tensor, TensorMut};

/// Everything a kernel file needs.
pub mod prelude {
    pub use crate::config::{
        BLOCK_THREADS, lds_max, lds_tile_bytes, mma_lds_bytes, reduce_lds_bytes,
        transpose_lds_bytes,
    };
    pub use crate::tensor::{Tensor, TensorMut};

    #[cfg(target_arch = "amdgpu")]
    pub use crate::arch::{block_id, sync_block, thread_id};
    #[cfg(target_arch = "amdgpu")]
    pub use crate::element::Element;
    #[cfg(target_arch = "amdgpu")]
    pub use crate::ops::mma;
    #[cfg(target_arch = "amdgpu")]
    pub use crate::tile::Tile;
}

#[cfg(target_arch = "amdgpu")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    // No unwinding or printing on the GPU: trap so the launch fails visibly.
    core::process::abort_immediate()
}

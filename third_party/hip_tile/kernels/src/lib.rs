//! Device kernels. Built for the host this crate exposes one launcher module per
//! kernel; built for `amdgcn-amd-amdhsa` (by `hip-tile-build`) it is the code object.
//!
//! `no_std` on both targets: the generated launchers only name `::hip_rt` items and
//! `::core`, and a `no_std` crate may depend on a `std` one.

#![no_std]
#![cfg_attr(target_arch = "amdgpu", feature(abi_gpu_kernel, core_intrinsics))]

pub mod backward;
pub mod copy;
pub mod gemm;
pub mod math;
pub mod saxpy;
pub mod shared_ops;
pub mod transpose;
pub mod yolo;

//! Kernels for running YOLOv4-tiny (Darknet) inference: f32, NCHW, batch 1.
//!
//! Every activation is a `[C x H*W]` row-major matrix (`Tensor`/`TensorMut` with an `ld`),
//! so a channel slice of a bigger buffer is just a view at a row offset.
//!
//! - [`conv`]: implicit-GEMM convolution with fused bias + activation.
//! - [`maxpool`]: Darknet max pooling.
//! - [`upsample`]: nearest-neighbour upsample.
//! - [`concat`]: copy into a channel slice (route layers).
//! - [`batched`]: host-side per-image launches of the four above for `batch > 1` training forward.
//!
//! Route layers with `groups=2 group_id=1` need no kernel: a group is a contiguous block
//! of channels, i.e. a `Tensor::new(ptr.add(first_ch * ld), c / groups, hw, ld)` view.

#[cfg(not(target_arch = "amdgpu"))]
pub mod batched;
pub mod concat;
pub mod conv;
pub mod maxpool;
pub mod upsample;

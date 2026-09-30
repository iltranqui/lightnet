//! Device code, one `#[cuda_module]` per source area.  Kernel names must stay
//! unique across the crate because PTX entry symbols are not namespace-qualified.

pub mod activations;
pub mod bias_scale;
pub mod conv_direct;
pub mod elementwise;
pub mod gemm;
pub mod im2col;
pub mod optim;

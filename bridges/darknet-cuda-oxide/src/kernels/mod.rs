//! Device code, one `#[cuda_module]` per source area.  Kernel names must stay
//! unique across the crate because PTX entry symbols are not namespace-qualified.

pub mod activations;
pub mod amp;
pub mod batchnorm;
pub mod bias_scale;
pub mod conv_direct;
pub mod conv_implicit;
pub mod convert;
pub mod elementwise;
pub mod gemm;
pub mod gemm_f16;
pub mod gemm_mma;
pub mod half_ops;
pub mod im2col;
pub mod layers_extra;
pub mod optim;
pub mod pool;
pub mod resample;
pub mod route;
pub(crate) mod unroll;

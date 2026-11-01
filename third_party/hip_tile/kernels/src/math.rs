//! AMD implementation of the `lightnet-kernels-shared` math primitives.
//! The cuda-oxide twin is `OxideMath` in `src/lightnet-cuda-oxide/src/kernels/activations.rs`.

/// `exp` through the LLVM intrinsic; `tanh` is the shared exp-based default.
pub struct AmdMath;

#[cfg(target_arch = "amdgpu")]
impl lightnet_kernels_shared::DeviceMath for AmdMath {
    #[inline(always)]
    fn exp(x: f32) -> f32 {
        core::intrinsics::exp::<f32>(x)
    }
}

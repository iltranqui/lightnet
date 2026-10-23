# `lightnet-cuda-cutile`

This crate is the opt-in `cutile-rs` bridge for the Rust Darknet port.

It provides:

- a concrete `CutileBackend`
- typed `f32` upload and download helpers
- cutile-backed elementwise smoke kernels
- backend ops for bias add, activation, shortcut add, route concatenation, and upsample
- a direct convolution kernel path for the Rust GPU runtime
- global avgpool forward/backward through a host fallback while the device kernel is still pending
- an implementation of `lightnet_cuda::F32TensorOpsBackend`

The bridge is pinned to cutile-rs v0.2.0. Build requirements:

- CUDA 13.3 is recommended upstream; CUDA 13.2+ is suitable for Ada sm_89.
- CUDA_TOOLKIT_PATH set to the CUDA installation root.
- CUDA and cuRAND development headers/libraries, plus libclang for the
  generated bindings (LIBCLANG_PATH can select a non-system installation).

The v0.2.0 default JIT uses the pure-Rust cutile-ir; LLVM 21, MLIR 21, and
Polly are not required by the normal path. CUDA 12.9 can be used for limited
compiler-only checks, but is not the supported target for the current
low-precision features.

Useful commands:

- `./scripts/check-env.sh`
- `cargo check --locked`
- `cargo build --locked`
- `cargo run --locked --example cutile_smoke`

The default library tests validate the bridge without a device. The ignored
device test exercises the actual convolution kernels, compares forward and
analytical backward results with an independent scalar reference, and verifies
a stochastic-gradient update for weights and biases:

```bash
export LIGHTNET_CUTILE_GPU_TEST=1
cargo test --locked --lib -- --ignored --nocapture
```

It is intentionally ignored on CPU-only machines; `cargo test --lib` still
reports it so the hardware-dependent coverage is visible.

The bridge cannot execute kernels without a CUDA-capable device. The main
workspace remains usable without this opt-in crate and retains the documented
CPU-reference fallback for unsupported native GPU operations.

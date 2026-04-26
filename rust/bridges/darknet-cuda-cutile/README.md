# `darknet-cuda-cutile`

This crate is the opt-in `cutile-rs` bridge for the Rust Darknet port.

It provides:

- a concrete `CutileBackend`
- typed `f32` upload and download helpers
- cutile-backed elementwise smoke kernels
- backend ops for bias add, activation, shortcut add, route concatenation, and upsample
- a direct convolution kernel path for the Rust GPU runtime
- global avgpool forward/backward through a host fallback while the device kernel is still pending
- an implementation of `darknet_cuda::F32TensorOpsBackend`

Build requirements:

- CUDA 13.2
- LLVM 21 with `llvm-config` on `PATH`, or `LLVM_CONFIG_PATH` set to the LLVM 21 binary
- MLIR 21 development headers, for example `libmlir-21-dev`
- Polly static libraries for LLVM 21, for example `libpolly-21-dev`
- `CUDA_TOOLKIT_PATH` set to the CUDA 13.2 install root
- `CUDA_TILE_USE_LLVM_INSTALL_DIR` set to the LLVM 21 install root

Useful commands:

- `./scripts/check-env.sh`
- `cargo check --manifest-path rust/bridges/darknet-cuda-cutile/Cargo.toml`
- `cargo build --manifest-path rust/bridges/darknet-cuda-cutile/Cargo.toml`
- `cargo run --manifest-path rust/bridges/darknet-cuda-cutile/Cargo.toml --example cutile_smoke`

Current local status on this machine:

- `/usr/local/cuda-13.2` exists and is a valid candidate for `CUDA_TOOLKIT_PATH`
- LLVM 21, MLIR 21 headers, and Polly static libraries are not currently visible to `./scripts/check-env.sh`
- set `LLVM_CONFIG_PATH` and `CUDA_TILE_USE_LLVM_INSTALL_DIR` after installing LLVM 21, then rerun `./scripts/check-env.sh`

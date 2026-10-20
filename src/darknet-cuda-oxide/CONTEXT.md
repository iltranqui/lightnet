# darknet-cuda-oxide

cuda-oxide GPU backend: Rust SIMT kernels. Not a workspace member.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Native GPU path for layers, written as `#[kernel]` functions compiled by the vendored cuda-oxide codegen backend. Build and test instructions: `README.md` here.

## Responsibilities

- Device kernels and their CPU-parity tests (`tests/`, for example `oxide_gemm_f8.rs`).
- Implements the `darknet-cuda` backend traits for the CLI `oxide` feature.
- Helper scripts under `tools/`.

## Dependencies

- Depends on: `darknet-core`, `darknet-cuda`, `third_party/cuda-oxide` crates `cuda-device` and `cuda-host`, `cuda-core`.
- Depended on by: `darknet-cli` (optional feature `oxide`).

## Constraints

- Build and test only through `./tools/test_oxide.sh`. `cargo oxide` does not work from this directory.
- Tests run with `--test-threads=1` (one CUDA context, large buffers).
- Max 4 build threads, prefer `-j 1` or `-j 2`; codegen is memory heavy.
- FP8 and FP4 work is planned in `tasks/rust/fp8_fp4.md`.

# hip_tile branch todo

## Table of Contents

- [Goal](#goal)
- [Decisions](#decisions)
- [Items](#items)

## Goal

One Rust kernel source that runs on NVIDIA through cuda-oxide (PTX) and on AMD through rustc `amdgcn` (hip_tile runtime). Same code, different code generation per backend.

## Decisions

- hip_tile enters as a plain copy at `third_party/hip_tile` (no `.git`), taken from `/home/kerrigan/src/hip_tile` at `c2c0f7e`. It is excluded from the lightnet workspace and builds as its own workspace.
- First proof: elementwise ops (bias add, activation, shortcut, route, upsample).
- Kernel style: per-thread SIMT in a shared `no_std` crate with a thin adapter per backend.
- AMD test hardware: Radeon 680M iGPU (gfx1031, RDNA2).

## Items

- [x] Copy hip_tile to `third_party/hip_tile`
- [x] Design the adapter surface: pure per-element functions plus a thin entry kernel per backend
- [x] Create the shared `no_std` kernel crate `src/lightnet-kernels-shared` (not a workspace member)
- [ ] Port bias add as the first shared kernel; run on cuda-oxide and on the 680M
- [ ] Port activation, shortcut, route, upsample
- [ ] Parity test: both backends against the CPU reference
- [ ] Update `third_party/CONTEXT.md` and `CLAUDE.md` once the layout is settled

## Design (step 1)

Facts found:
- Two toolchains: lightnet/cuda-oxide pins `nightly-2026-08-28`; hip_tile pins `nightly-2026-10-25`. The shared crate must build on both, so it is plain `#![no_std]`, no dependencies, no nightly features.
- LegoGears detector layers: `[convolutional]` (leaky or linear), `[maxpool]`, `[route]`, `[upsample]`, `[yolo]`. Elementwise subset to share: bias add, leaky/linear activation, route copy, upsample index math.
- Entry attributes differ (`#[kernel]` + `#[cuda_module]` in oxide, `#[kernel]` + `Tensor` in hip_tile), so the entry point cannot be shared. The element math can.

Adapter surface (decision): the shared crate holds pure per-element functions, `fn op_elem(i: usize, inputs: &[f32], params...) -> f32` (or an index map for gather ops). Each backend has a ~5 line entry kernel that computes the global index, calls the shared function, and writes the result. So the adapter is: global thread index (backend), input slice (backend gives a `&[f32]`), output write (backend). No barrier is needed for elementwise ops.

Exceptions (to justify as they appear):
- conv and GEMM: AMD uses hip_tile tile kernels, NVIDIA keeps the tuned oxide kernels.
- Transcendental activations (logistic, mish, swish) need `exp`; `core` has no `f32::exp`. Not needed for the LegoGears backbone (leaky/linear). Decide when a layer needs it.

Layout:
- `src/lightnet-kernels-shared` (new, own `[workspace]`): shared element functions.
- `third_party/hip_tile/kernels/src/shared_ops.rs`: AMD entries.
- `src/lightnet-cuda-oxide/src/kernels/shared_ops.rs`: oxide entries.

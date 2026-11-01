# hip_tile branch todo

## Table of Contents

- [Goal](#goal)
- [Decisions](#decisions)
- [Items](#items)
- [Review](#review)

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

## Progress

- Step 3 detector gate passed (v0.2.211): `hip_darknet --test gpu` on Radeon 680M, 10 valid images, GPU detections equal CPU (class, IoU >= 0.99, score diff <= 1e-3), 38/38 ground truth matched, per-layer max abs 1.91e-5 and max rel 1.69e-4 over 38 layers. Shared source in use: bias add, leaky, upsample index. Exceptions kept per backend: conv and GEMM (tile kernels on AMD, tuned kernels on NVIDIA), maxpool, route (pure layout copy, no math).
- Latency per model: not measured yet.

## Step 4 findings (segmentation)

- Trained weights exist: `datasets/semantic_binary/darknet/backup/semantic_segmentation_best.weights` and `semantic_multi/...`. Binary cfg: 23 conv (22 tanh, 1 linear), 5 maxpool, 5 route, 5 upsample, `[logistic]` head. Multi uses the spatial softmax head.
- hip_tile has no tanh, logistic or softmax. Device `exp` differs per backend, so the shared crate needs a small math trait implemented once per backend.

## Review

Result: LegoGears detector, binary and multi-class segmentation, and their ONNX checks run on the Radeon 680M (gfx1031) through hip_tile and match the existing paths. Versions v0.2.210 to v0.2.214.

### Numbers (all measured on this machine)

| Case | Reference | Result |
|---|---|---|
| Detector, 10 valid images | lightnet-style CPU reference in `hip_darknet` | GPU detections equal CPU (class, IoU >= 0.99, score diff <= 1e-3), 38/38 ground truth matched, 0 class errors; per-layer max abs 1.91e-5, max rel 1.69e-4 over 38 layers |
| Binary segmentation, 5 test images | lightnet CPU forward (`segment-image`, `LIGHTNET_BACKEND=cpu`) | class agreement 1.000000 on all; head (logistic) max abs <= 8.3e-7; last conv max abs <= 3.22e-6; 39 layers compared |
| Multi-class segmentation, 5 test images | same | class agreement 1.000000 on all; head (softmax) max abs <= 1.07e-6; last conv max abs <= 3.34e-6 |
| Segmentation ONNX, 1 image per net, 8 variants each | `onnx_match_seg.sh both --variants --amd` | 16/16 OK; class agreement 100%; fp32 max abs 6.6e-7 (binary), 8.3e-7 (multi); fp16 7.6e-4, 2.2e-3 |
| Detector ONNX, FP32 fused BN | `onnx_match.sh --amd` | yolo_0 and yolo_1 max abs 2.29e-5 (rel to max 1.03e-6), OK at tol 1e-4 |
| Latency, 680M, warm | | detector 4.99 ms forward (CPU 457.65 ms, 91.8x); binary segmentation 37.2 ms; multi 35.5 ms (224x224) |

### Shared source

Written once in `src/lightnet-kernels-shared`: bias add, leaky, tanh, logistic, spatial softmax, upsample index map, plus the `DeviceMath` trait (one `exp`, optional native `tanh`). cuda-oxide production kernels call them (`bias_add`, `bias_activation`, `activate`, `upsample_nearest`); hip_tile calls them from the conv epilogue, upsample and head kernels.

### Exceptions (kept per backend, with the reason)

- conv and GEMM: AMD uses hip_tile tile kernels (no matrix cores on RDNA2, FMA), NVIDIA keeps the tuned oxide kernels.
- maxpool: separate kernels per backend (window logic not moved; not an elementwise op).
- route: a layout copy on both, no math to share.
- tanh: oxide overrides the shared exp-based default with its native `tanh`.

### Not done or not verified

- ONNX checks ran on 1 image per net (segmentation) and 1 image (detector); the 5-image forward comparison above covers more inputs but not the ONNX export.
- Detector ONNX was checked in FP32 only: `onnx_check.py` has no FP16 mode.
- `cargo xtask export_test_onnx_match*` was not run with `--amd`; the scripts were run directly. The xtask arms pass arguments through but that was not tested.
- No AMD training (out of scope). The CPU/GPU training code in hip_darknet got `Tanh` arms only to keep matches exhaustive; the GPU trainer rejects tanh and the new heads.
- NVIDIA side: `oxide_parity` 11, `oxide_pool_parity` 9, `oxide_shared_ops` 2 passed. The full oxide suite was not rerun.
- Clippy exits 0 but reports warnings in `src/lightnet-cuda` (not touched here).
- `bias_add` has no standalone AMD kernel in the detector path (it is fused into the conv epilogue); its AMD entry kernel exists and is tested (`shared_ops`).

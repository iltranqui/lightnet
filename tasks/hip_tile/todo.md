# hip_tile branch todo

## Table of Contents

- [Goal](#goal)
- [Decisions](#decisions)
- [Items](#items)
- [Review](#review)
- [Second pass: shared maxpool, route, tanh and conv gather](#second-pass-shared-maxpool-route-tanh-and-conv-gather)

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

Superseded for maxpool, route, tanh and the conv gather index by the second pass below.

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

## Second pass: shared maxpool, route, tanh and conv gather

Goal: the ops the first pass kept per backend take their bodies from `src/lightnet-kernels-shared`, or have a justified exception. Versions v0.2.215 and up.

### Now in the shared crate

| Op | Shared function | cuda-oxide caller | hip_tile caller |
|---|---|---|---|
| maxpool forward | `maxpool_window_max(load, plane, (h, w), (oy, ox), size, (stride_x, stride_y), off)` | `maxpool_forward` (`pool.rs`) | `maxpool` (`yolo/maxpool.rs`) |
| maxpool backward (argmax) | `maxpool_window_argmax(load, ...)` (same scan, returns the flat index of the first max) | `window_argmax`, `window_argmax_h` (f16) | `maxpool_bwd::selects` |
| route copy | `route_copy_index(i, channels, stride, (src_total, src_off), (dst_total, dst_off))` | `route_copy` (`route.rs`) | `route_copy` entry (`shared_ops.rs`), launched by `hip_darknet` through `launch_route_copy` |
| tanh | `DeviceMath::tanh` default: exp-based, now with an odd series below `|x| < 0.1` | `OxideMath` still overrides it with native `tanh` (kept, see below) | `AmdMath` uses the default |
| conv gather | `conv_tap(kidx, k)` and `conv_input_offset(pos, ow, (ky, kx), strides, dilation, pad, (h, w))` | `sgemm_conv_implicit`, `im2col`, `im2col_fp8` | `conv_body` (`yolo/conv.rs`) |

Memory access goes through a closure (`maxpool_window_max`) or an index, so the shared code has no backend type. The conv index math is `u32`: it runs per element inside the GEMM loops, where 64-bit division is slow on both GPUs.

### Tanh decision

Measured on the NVIDIA GPU over 1,048,576 inputs in [-12, 12] (`tanh_native_vs_exp_based` in `oxide_shared_ops.rs`):

| | max abs vs f64 | max rel vs f64 |
|---|---|---|
| native `tanh` (oxide override) | 9.84e-8 | 1.83e-7 |
| shared exp-based, before the series | 1.47e-7 | 7.76e-2 |
| shared exp-based, with the series | 1.27e-7 | 1.13e-6 |
| native vs shared (with series) | 1.19e-7 | 1.15e-6 |

On the real nets (22 tanh convolutions in each of `semantic_binary` and `semantic_multi`, 5 test images each, every layer's post-activation output compared; oxide uses native tanh, AMD the shared default, CPU libm `tanhf`):

| Pair | worst max abs over 10 runs | worst max rel-to-max |
|---|---|---|
| oxide (native) vs AMD (shared) | 2.00e-6 | 5.71e-6 |
| oxide (native) vs CPU | 2.74e-6 | 5.28e-6 |
| AMD (shared) vs CPU | 2.14e-6 | 4.80e-6 |

The native-vs-shared gap is no larger than either backend's gap to the CPU, so the choice of tanh body is below the accumulated convolution rounding. This compares layer outputs, not the isolated tanh inputs (the dumps hold post-activation values only).

`1 - 2/(exp(2x)+1)` cancels near zero, which gave the 7.8% relative error; the series fixes it for both backends. The override is kept: with it in place the oxide forward matches ONNX Runtime on both segmentation nets, 16/16 variants OK (fp32 max abs 5.96e-7 binary, 8.3e-7 multi; fp16 7.6e-4, 2.2e-3), and the AMD path, which uses the shared default, passes the same gates. The two tanh bodies differ by at most 1.2e-7 absolute.

### Exceptions that stay, and why

- **conv and GEMM tile structure.** hip_tile writes the GEMM as `mma` over `Tile<f32, BM, BN>` values with whole-tile loads and a block-wide LDS layout chosen by the compiler. The oxide kernels are per-thread SIMT with explicit `cp.async` double buffering, 4x4 register blocking and `__syncthreads`. They have no common source form without a tile abstraction on top of SIMT (or the reverse), which the goal rules out. Shared: gather index, leaky, tanh, bias (the epilogue bodies). Not shared: the loop nest, the tiling and the memory staging.
- **maxpool backward.** The argmax is shared (`maxpool_window_argmax`): oxide `window_argmax` and its f16 twin `window_argmax_h`, and hip_tile `maxpool_bwd::selects`, all call it. What stays per backend is the gather loop around it (oxide: one thread per input element; hip_tile: a `Tile` over channels x positions with `store_tile_add`).
- **Native tanh.** See the decision above.
- **Route views on AMD.** The goal asked to keep a zero-copy view path where a route is a view. There was none to keep: before this pass `hip_darknet` gave every route layer its own output buffer and copied each source into it with `launch_concat`. The only "view" was the group slice, a pointer offset `group_id * c * ohw` into the source layer; that is now the `src_offset` argument of `route_copy`, still without a separate copy of the slice. A route with one source and `groups = 1` could alias its input and skip the copy, but that changes buffer ownership in `GpuModel` (a yolo layer already aliases its input via `outs[i] = None`) and the saving is one `ohw`-sized copy per route layer, so it is left out. The training executor (`train/gpu.rs`) still uses `launch_concat_batched`, a batched tile copy; it is out of this goal.
- **Argument order of kernels.** Each backend keeps its own launch signature; only the index and element bodies are shared.

### Gates after the second pass

All rows were rerun in one chain on this machine after the last code change (`final_gates.sh`, scratchpad).

| Gate | Result |
|---|---|
| Shared crate host tests | 19 passed (maxpool value 6 and argmax 2, route 2, conv gather 2, tanh relative accuracy 1, plus the first-pass ones); clippy clean |
| hip_tile workspace tests (release) | exit 0, 33 test binaries ok (includes `maxpool_bwd` and the gradient tests) |
| AMD detector, 10 images | 38/38 matched, 0 class errors, GPU == CPU detections; 38 layers, per-layer max abs 1.91e-5, max rel 1.69e-4 |
| AMD segmentation vs lightnet CPU, 5 images per net | class agreement 1.000000 on all 10; head max abs <= 9.54e-7 |
| AMD ONNX, segmentation (`onnx_match_seg.sh both --variants --amd`, 1 image, 8 variants x 2 nets) | 16/16 OK, agreement 100%; fp32 max abs 7.15e-7 (binary), 9.54e-7 (multi); fp16 7.6e-4, 2.2e-3 |
| AMD ONNX, detector (`onnx_match.sh --amd`, FP32) | yolo_0 and yolo_1 max abs 2.29e-5 (rel to max 1.03e-6), OK |
| NVIDIA forward vs ONNX Runtime (`onnx_match_seg.sh both --variants`, release oxide binary rebuilt with the shared kernels, native tanh kept) | 16/16 OK, agreement 100%; fp32 max abs 5.96e-7 (binary), 8.35e-7 (multi); fp16 7.6e-4, 2.2e-3 |
| Oxide `oxide_parity` / `oxide_pool_parity` / `oxide_shared_ops` / `oxide_amp` | 11 / 9 / 5 / 22 passed (`oxide_amp` covers the f16 argmax) |
| `cargo xtask check`, `clippy`, `tools/check_module_layers.sh` | exit 0 (clippy warnings remain in untouched `lightnet-cli`, `lightnet-cuda`, `lightnet-layers`) |

### Codex review of v0.2.215

`gpt-6-luna`, medium effort, read-only, one bounded task over `git show 6e60816`. It found the maxpool scan, the route index, the grouped and multi-source `hip_darknet` route copy and the tanh series (sign, NaN, continuity at 0.1) equivalent to the removed code. One finding: the callers narrow `usize` conv arguments to `u32` without a range check, so an absurd dilation such as `2^32` would wrap. Fixed as documentation and a test, not a runtime check: `CONV_INDEX_LIMIT` (32,767 for plane sizes, kernel size, strides, dilation, padding) and the contract on `conv_tap` are written down, and `conv_index_math_is_exact_at_the_documented_limit` runs the extremes. The device kernels still do not validate their arguments; the hosts that launch them size layers from Darknet cfgs, orders of magnitude inside the limit.

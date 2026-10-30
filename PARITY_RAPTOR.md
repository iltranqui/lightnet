# Raptor FP4 parity handoff

Last updated: 2026-10-30. This is an **unfinished implementation handoff**, not a parity claim. Nothing in this document has been built, tested, or run on the GPU unless it says so explicitly.

## Table of Contents

- [Objective and reference](#objective-and-reference)
- [Components and responsibilities](#components-and-responsibilities)
- [Current state and changes made](#current-state-and-changes-made)
- [Detection implementation sequence](#detection-implementation-sequence)
- [Verification and commands](#verification-and-commands)

## Objective and reference

Port the detection and semantic segmentation layers and workflows from `/home/kerrigan/src/DarkSuite/darknet` branch `raptor-fp4` into Lightnet, including their training, inference and ONNX export paths. Read the reference with `git show raptor-fp4:<path>`; the Darknet working tree is on another branch and has unrelated user changes. Do not switch or modify it.

The immediate priority is **detection**, especially the native GPU classic YOLO training loss in `src-lib/yolo_layer.cu`. The source-by-source audit is not complete: do not interpret the initial list below as an exhaustive inventory of all relevant layers and options.

Out of scope by user instruction: **OBB and Gaussian YOLO**. Do not implement them. The GPU YOLO loss rejects them explicitly and they stay on the host path.

The user also specified:

- SM 89 native training and inference: FP32 master weights; use FP8 tensor-core convolution for eligible layers. The convolution immediately before each detection or segmentation head remains FP32. FP8 is limited to convolution, maxpool and route; other layers remain higher precision. Pool/route do not use tensor cores; maintain FP8 storage where useful and correct.
- ONNX: FP32 and FP16 exports with NCHW or NHWC external layout; detection offers raw heads or decoded boxes/confidences, with NMS performed by the application. Segmentation defaults to activated probabilities and supports `-activation`/`-noactivation`.
- Acceptance datasets: `/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2`, `semantic_binary`, and `semantic_multi`. Begin with short smoke tests. Compare native FP8 with native FP32, and ONNX FP32/FP16 with native FP32. The user's GPU is an RTX 4060, SM 89.

## Components and responsibilities

```text
Darknet raptor-fp4                          Lightnet
src-lib/yolo_layer.{cpp,cu}                 src/lightnet-core/src/{model.rs,train.rs}
                                            src/lightnet-core/src/runtime/gpu_yolo_loss.rs
src-lib/{region*,gaussian_yolo*,            src/lightnet-core/src/{detect.rs,dense_loss.rs,
  modern_yolo*,yolox*,ppyoloe*,               layers/,runtime/}
  yolonas*,centernet*,detector*}            src/lightnet-cli/src/main.rs
src-seg/                                    src/lightnet-core/src/{segmentation_data.rs,layers/segmentation/}
src-onnx/{darknet_onnx*,                    src/lightnet-core/src/onnx/{export.rs,proto.rs}
  darknet_segmenter_onnx*}                  tools/onnx_check.py
src-test/{test_yolo_layer_cuda,             src/lightnet-cuda/src/lib.rs
  test_modern_yolo_heads,                   src/lightnet-cuda-oxide/src/{dense_loss_backend.rs,kernels/}
  test_segmentation_layer}                  src/lightnet-cuda-oxide/tests/
```

Model parsing and validation own configuration semantics. Head kernels own activation, target assignment, loss, deltas and metrics. Runtime owns device tensors, forward-state lifetimes, loss scaling and optimizer updates. The CLI connects data, training and inference. ONNX export owns graph layout, precision and output semantics. Tests must distinguish hand-derived source comparisons, Rust CPU comparisons, and compiled-C++ reference results.

## Current state and changes made

### Git state

- Work is committed on branch `wip-gpu-parity`, based on `0362beaa` (remote `origin/main`). `4406741` holds the 80 `src/` files. `f1f15d0` and `ed7fcae` follow. `a51d56c` removes the legacy `[detection]` layer. All of it is WIP and has not been built or tested.
- `refs/heads/main` was missing commit `dcbd197a5da7a09db8f3cd887842e6f050ff3fd0`, a zero-byte loose file whose content is unrecoverable. It now points to `0362beaa` (remote `origin/main`), by user choice. HEAD is `wip-gpu-parity`.
- Thirteen zero-byte loose object files were moved, not deleted, to `.git/corrupt-empty-objects/`. Five of the moved objects are blobs that dangling trees reference. None are reachable from HEAD (`git rev-list --objects HEAD` runs without errors).
- The index cache-tree still references missing object `8f5064135548b63d658603b1ba5d8b28e388dae1`. This is a cache only.
- The sixty files outside `src/` (docs, `tasks/`, `CLAUDE.md`, `CONTEXT.md` files) that had been staged are committed in `ed7fcae`. Nothing is staged. This resolves the earlier decision. `Cargo.lock` has an uncommitted modification that is not part of this work.

### Implemented, not yet built or tested

- **Device-seed shape check.** `F32TensorOpsBackend::tensor_shape()` defaults to `None`. Oxide, cuTile and the FakeBackend override it. `backward_accumulate_device_seeded_gpu` in `runtime/gpu_backward.rs` rejects a seed whose shape is not `[batch, c, h, w]` of its layer. The wrong-shape assertion was added to `gpu_device_seeds_match_host_seeds_with_loss_scaling` in `runtime/tests/tests_04.rs`. The earlier FakeBackend pass predates this assertion and must be rerun.
- **Native FP16 spatial softmax.** `amp_spatial_softmax_forward` and `amp_spatial_softmax_backward` in `lightnet-cuda-oxide/src/kernels/amp.rs` keep f16 storage with f32 math. `spatial_softmax_f32` and its backward in `softmax_backend.rs` dispatch to them for half-storage input. The multiclass segmentation forward in `dense_loss_backend.rs` uses them and honours `temperature`. Test: `spatial_softmax_f16_storage_matches_f32_with_temperature` in `tests/oxide_runtime_chain.rs`. The test assumes `to_half_storage` works on a default `OxideBackend`, which may need AMP enabled.
- **Softmax temperature.** Segmentation temperature is applied in the native f32 path, the f16 path, and the ONNX graph (`Div` before `Softmax`). The plain `[softmax]` layer has no temperature in the reference (fixed at 1), so none is added there.
- **Decoded YOLO ONNX output.** `OnnxOptions::decode_detections` (default `false`, so raw heads remain the default) decodes every YOLO head in the graph into `boxes` `[batch, N, 1, 4]` (normalized x1, y1, x2, y2) and `confs` `[batch, N, classes]` (objectness × class). It mirrors the reference `darknet_onnx.cpp` math, including `scale_x_y`, `new_coords`, and skipping sigmoid when the previous layer is logistic. NMS stays with the application. CLI flags: `-decode` (alias `-boxes`) and `-raw` (alias `-noboxes`). Test: `decoded_detection_export_replaces_raw_heads_with_boxes_and_confs` in `onnx/export.rs`. No numerical comparison against the reference has been done.
- **Classic YOLO GPU loss.**
  - Kernels: `yolo_loss_background`, `yolo_loss_positives`, `yolo_loss_average` and `yolo_loss_reduce` in `lightnet-cuda-oxide/src/kernels/dense_loss.rs`. They port `yolo_loss_kernel` from `raptor-fp4` `src-lib/yolo_layer.cu`. The positives kernel runs one thread per image and does not use atomics. Known risk: Oxide codegen may reject raw-pointer device helpers, tuple returns, `&mut [f32; 4]` arguments, or `f32::atan2`.
  - Backend: `YoloGpuLossArgs`, `YoloImageMetrics` and `F32TensorOpsBackend::yolo_loss_f32` in `lightnet-cuda/src/lib.rs`. The default returns `Ok(None)`. Oxide overrides it through `yolo_loss_backend_f32`.
  - Runtime: `compute_yolo_training_loss_gpu` in `runtime/gpu_yolo_loss.rs`. It activates each raw head once with `yolo_activate_f32`, returns device seeds, and applies `delta_normalizer` to the propagated seed only. It returns `Ok(None)` for unsupported configurations: non-CIoU loss, `iou_thresh_kind` other than IoU, focal loss, label smoothing, class multipliers or map, more than 32 anchors, more than 1024 boxes, adversarial, equidistant points, bad-label rejection, OBB, and Gaussian YOLO.
  - CLI: `run_detector_train` uses the GPU loss when `LIGHTNET_YOLO_GPU` is set and not `0`. Otherwise, or when the GPU loss declines, it uses the host loss with `backward_accumulate_seeded_gpu`. The GPU path uses `backward_accumulate_device_seeded_gpu`.
  - Tests: `gpu_yolo_loss_falls_back_when_backend_has_no_native_loss` in `runtime/tests/tests_04.rs`, and `native_yolo_loss_matches_hand_derived_single_truth` in `tests/oxide_runtime_chain.rs`. The Oxide test checks objectness and class deltas (0.8, 0.7) and the metrics for one truth on one cell. It does not check the box delta.
- **Classic YOLO parity fixes (CPU and GPU).**
  - `mask` defaults to `0..num` when absent, as `make_yolo_layer`. `num` is the total anchor count (`model.rs`, `[yolo]` arm).
  - `max` is parsed into `YoloSpec.max_boxes` (default 200). The CPU loss (`train.rs`, `yolo_truths`) and the GPU path (`gpu_yolo_loss.rs`) both stop at that cap. GPU rejects `max > 1024`.
  - CPU truth list ends at the first label with `x == 0`, as `yolo_layer.cpp` does. The GPU kernel already did this.
  - `embedding_layer` and the tracking keys (`track_history_size`, `sim_thresh`, `dets_for_track`, `dets_for_show`, `track_ciou_norm`) on `[yolo]` now fail with `InvalidConfig`. `nms_kind`, `beta_nms`, `resize` and `random` are accepted: shipped cfgs set them (21 to 25 files each), and they do not change the classic loss.
  - `yolo_parity_*` CPU tests in `train.rs`: two truths in different cells, `max` cap, `x == 0` sentinel, extra-anchor (`iou_thresh`). Hand-derived or relational against Rust only.
  - Verified: host `cargo xtask test yolo` 124 passed in `lightnet-core` and 2 in `lightnet-cli`. Oxide `oxide_runtime_chain -- yolo` 3 passed on the RTX 4060.
  - Not verified: no raptor-fp4 golden exists, so none of this is checked against Darknet. The GPU box delta is not checked; the Oxide test covers objectness and class deltas only.
- **Decoded ONNX tests.** `decoded_detection_export_{fp32,fp16}_{nchw,nhwc}` (new) and the existing `decoded_detection_export_replaces_raw_heads_with_boxes_and_confs` (now uses a valid 16x16 `DECODE_CFG`; the old `CFG` gave a head channel mismatch). All 5 pass in the host run. No numerical comparison against Darknet.
- **ONNX export tests from earlier.** The 13 focused exporter tests and the CLI option test passed at their earlier checkpoints. They have not been rerun since the decode option was added.

### Earlier work

- Segmentation: pixel-mask discovery and truth encoding, spatial softmax, semantic and legacy segmentation ONNX graph outputs. Focused CPU and SM 89 spatial-softmax tests passed at their checkpoint.
- Binary and multiclass FP16 NHWC ONNX graphs executed in CPU ONNX Runtime. Binary FP16 vs FP32 ONNX on a seeded synthetic input had max absolute difference 0.0024834 and mean 0.0001714. This is not native/ONNX parity.
- The CPU native binary-model comparison was interrupted after debug inference exceeded a short smoke test. No result exists.
- Existing ONNX work implements FP32/FP16 and NCHW/NHWC boundary options and segmentation activation/raw options.

### Not started

- Region truth loss and decode, modern-head (YOLOX, PP-YOLOE, YOLO-NAS) losses and decode, and CenterNet GPU paths. `tasks/rust/02_state/layer-support.md` reports some of these as incomplete or host-owned. Verify live source before acting on that table.
- Subagent drafts, **uncommitted and unwired**, not built: `layers/modern_yolo/{yolox,ppyoloe,yolonas}_train.rs`, `layers/centernet/train.rs`, and `lightnet-cuda-oxide/src/kernels/{yolox,ppyoloe,yolonas,centernet}_train.rs`. `yolox_train.rs` has a syntax error at line 184. Their assignment logic differs from the existing `modern_yolo/cpu.rs` (`assign_tal_like`, YOLOX centre-only geometry, SIMOTA top-k, L1 and class-term gaps). Undecided: fix `cpu.rs` or replace it with these files.
- Legacy `[detection]` layer: removed in `a51d56c`. It was an identity passthrough, and no shipped cfg uses it. Not built or tested.
- Per-layer FP8 policy and the FP32 head boundary.
- Short LegoGears, `semantic_binary` and `semantic_multi` smoke tests, and the native-vs-native FP8 and ONNX FP32/FP16 comparisons.

## Detection implementation sequence

1. Read `src-lib/yolo_layer.cu`, `yolo_layer.cpp`, `activation_kernels.cu`, and `src-test/test_yolo_layer_cuda.cpp` from `raptor-fp4`; compare each head, loader, decoder, loss option and test with current Rust. Record specific gaps and identify which supported cfg options are currently discarded by parsing. **Status: not done.**
2. Validate device-resident seeded backward: exact seed shapes (**implemented, unverified**), loss scaling, forward-state lifetime, fallback behavior, and parity with the host-seed path on the real Oxide backend. **Status: partial.**
3. Classic YOLO GPU activation, assignment, loss and metrics. **Status: implemented, unverified.** Still to verify: parity with the Rust CPU loss on multi-truth, extra-anchor, objectness-smoothing, `new_coords`, truth-threshold and collision cases, plus the reference-rejected options.
4. Connect head activation and loss to `GpuForwardState` and the device-seed backward path. **Status: implemented as opt-in (`LIGHTNET_YOLO_GPU`), unverified.** Still to verify: no double activation, forward-state lifetime, and mixed-precision loss scaling in the real training loop.
5. Other detection gaps (Region, modern heads, CenterNet). **Status: not started.** Do not imply that the classic YOLO kernel completes all detection heads.
6. Per-layer FP8 policy and FP32 head boundary, then a short LegoGears training and inference smoke test. **Status: not started.** Do not claim FP8 coverage or performance from an FP32-only test. Training runs are limited to the user's authorized short smoke tests; choose a small, explicit iteration count and document it.
7. Decoded detection ONNX output, FP16/FP32 and layout coverage, and comparisons on the three acceptance datasets. **Status: decoded output implemented, unverified; comparisons not started.** Follow user priority if they redirect work.

## Verification and commands

Run builds and checks serially, with at most two jobs, GPU tests with one test thread, and a soft `MemoryHigh=3G`. Never use a hard RAM cap. Use `cargo xtask` where it covers the command. The Oxide backend requires `tools/test_oxide.sh`. Do not treat an interrupted command as a pass. No command in this list has been run against the current working state.

```sh
JOBS=2 MEM=3G cargo xtask check
JOBS=2 MEM=3G cargo xtask gpu-check
JOBS=2 MEM=3G cargo xtask test gpu_device_seeds_match_host_seeds_with_loss_scaling
JOBS=2 MEM=3G cargo xtask test gpu_yolo_loss_falls_back_when_backend_has_no_native_loss
JOBS=2 MEM=3G cargo xtask test decoded_detection_export_replaces_raw_heads_with_boxes_and_confs
cargo xtask oxide --test oxide_runtime_chain -- classic_yolo_activation_matches_reference_layout
cargo xtask oxide --test oxide_runtime_chain -- native_yolo_loss_matches_hand_derived_single_truth
cargo xtask oxide --test oxide_runtime_chain -- spatial_softmax_f16_storage_matches_f32_with_temperature
git diff --check
```

Decided by user (layer interview):

- **Region (`[region]`)**: full truth loss and decode, so training and inference both work. Backend: **GPU directly**, with no CPU reference first. The user accepted that there is no independent CPU oracle for these results. Verify GPU output against the compiled `raptor-fp4` golden or hand-derived values where possible, and say so when a result has only been checked against itself.
- **Legacy Detection (`[detection]`)**: removed in `a51d56c` on user request. The earlier reading of "the [yolo] layer" was only that classic `[yolo]` is the priority.
- **Smoke tests**: deferred. Finish the code paths first; no training runs until the user asks.

Carried over from earlier: after classic YOLO is verified, Region comes next.

Decided by user (second interview):

- **Modern heads (`[yolox]`, `[ppyoloe]`, `[yolonas]`)**: full loss and decode, ported from `raptor-fp4`. Backend: **GPU directly**, same caveat as Region: no independent CPU oracle.
- **CenterNet (`[centernet]`)**: full loss and decode. Backend: **GPU directly**, same caveat.
- **Yolo3d (`[yolo_3d]`)**: leave as-is. No loss, decode or GPU work.
- **Gaussian and OBB YOLO**: out of scope, as stated above.

Decided by user (interview on the convolutional layer, plus the earlier questions):

- **Convolutional, BN on GPU**: enable batch-norm convolution training natively, so BN convolutions no longer get rejected on the Muon path. Scope: the detection acceptance configs.
- **FP8 eligibility**: every convolution except the one immediately before a detection or segmentation head, which stays FP32.
- **Optimizer boundary**: convolution gradients must update natively on device before any smoke test. Host download at the optimizer boundary is not acceptable.
- **Ordering**: FP8 policy comes after the detection heads. Native vs ONNX comparisons come after all code is done.
- **Staged files outside `src/`**: resolved. They are committed in `ed7fcae`.

Next: build and run the commands above, fix Oxide codegen issues in the new YOLO kernels, then add device-seed GPU parity tests against the host seed path and the reference baseline (extra anchors, objectness smoothing, `new_coords`, truth-threshold positives, combined options, collisions, and invalid-option rejection). Test assignment, deltas and metrics against Rust CPU and, where practical, a compiled `raptor-fp4` golden. Run a short real-data smoke test only once the end-to-end path is wired. Do not transfer earlier test results to new changes.

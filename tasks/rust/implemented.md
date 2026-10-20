# Implemented — `tasks/rust`

**Purpose.** Single consolidated inventory of everything delivered so far under
`tasks/rust`. The plan files now keep only *open* work and point here instead
of repeating the completed prose.

**Reading the evidence.** Entries cite the strongest evidence available:
test names/counts, commit hashes, and file paths. `C++ dump parity: NOT RUN`
means no C++ build of the owning branch exists in this environment (see
`STATUS.md` "Known evidence gaps"), so the port is by-construction plus
CPU/GPU/finite-difference checks, not a byte dump against C++.

Facts here are condensed from the former "Done" bodies of `todo.md`,
`STATUS.md`, `layers.md`, and `fp8_fp4.md`. When a detail matters, the source
file is named inline.

---

## Table of Contents

- [1. Workspace, build, tooling](#1-workspace-build-tooling)
- [2. `cuda-oxide` GPU backend (P0–P9)](#2-cuda-oxide-gpu-backend-p0p9)
  - [P0–P1 — scaffold and elementwise](#p0p1--scaffold-and-elementwise)
  - [P2 — Convolution](#p2--convolution)
  - [P2b — Tensor-core GEMM](#p2b--tensor-core-gemm)
  - [P3 — Pooling / resample / route](#p3--pooling--resample--route)
  - [P3b — YOLO head elementwise](#p3b--yolo-head-elementwise)
  - [P4 — CLI + end to end](#p4--cli--end-to-end)
  - [P5 — CPU-fallback gaps](#p5--cpu-fallback-gaps)
  - [P6 — FP16 (inference)](#p6--fp16-inference)
  - [P7 — Benchmarks](#p7--benchmarks)
  - [P8 — GPU training](#p8--gpu-training)
  - [P9 — Training loop + data pipeline](#p9--training-loop--data-pipeline)
- [3. FP8 / FP4 (`fp8_fp4.md`)](#3-fp8--fp4-fp8_fp4md)
- [4. Layer port — Phase 0 (Tier 0 + Tier 1)](#4-layer-port--phase-0-tier-0--tier-1)
- [5. Layer port — Phase 1b](#5-layer-port--phase-1b)
- [6. Layer port — Phase 2A (anchor-free heads) — COMPLETE](#6-layer-port--phase-2a-anchor-free-heads--complete)
- [7. Layer port — Phase 2B (partial: MoE + EML math)](#7-layer-port--phase-2b-partial-moe--eml-math)
- [8. Layer port — Phase 3A shared primitives (prerequisite set)](#8-layer-port--phase-3a-shared-primitives-prerequisite-set)
  - [`[vit]` reference layer path (Phase 3A)](#vit-reference-layer-path-phase-3a)
  - [`[tucker_attention]` reference layer path (Phase 3A)](#tucker_attention-reference-layer-path-phase-3a)
  - [`[mambavision]` reference layer path (Phase 3A)](#mambavision-reference-layer-path-phase-3a)
  - [`[kimi_kda]` reference layer path (Phase 3A)](#kimi_kda-reference-layer-path-phase-3a)
- [9. Inference parity — LegoGears_v2 (the final target)](#9-inference-parity--legogears_v2-the-final-target)
- [10. Data augmentation (`augmentation-audit.md`)](#10-data-augmentation-augmentation-auditmd)
- [11. Housekeeping](#11-housekeeping)
- [12. Gradient-shortcut audit (`DARKNET_GRAD_MODE`)](#12-gradient-shortcut-audit-darknet_grad_mode)
- [13. OBB CPU training gaps (plan-open-items Waves 0–1)](#13-obb-cpu-training-gaps-plan-open-items-waves-01)
- [14. ONNX export](#14-onnx-export)
- [15. CARPN sample-level polygon mAP dispatch](#15-carpn-sample-level-polygon-map-dispatch)
- [16. CARPN CPU network inference and direct mAP](#16-carpn-cpu-network-inference-and-direct-map)
- [17. DCNv4 CPU reference and GPU f32 host fallback](#17-dcnv4-cpu-reference-and-gpu-f32-host-fallback)
- [18. DCNv1 and DCNv2 CPU reference and GPU f32 host fallback](#18-dcnv1-and-dcnv2-cpu-reference-and-gpu-f32-host-fallback)
- [19. DSConv single-template Stage-1 CPU reference](#19-dsconv-single-template-stage-1-cpu-reference)
- [20. Graph-convolution Stage-1 CPU reference](#20-graph-convolution-stage-1-cpu-reference)
- [21. WMHF Stage-1 CPU reference](#21-wmhf-stage-1-cpu-reference)
- [22. CPU Muon hybrid optimizer](#22-cpu-muon-hybrid-optimizer)
- [23. Clifford Stage-1 CPU reference](#23-clifford-stage-1-cpu-reference)
- [24. Recursive-block Stage-1 CPU reference](#24-recursive-block-stage-1-cpu-reference)
- [25. Hyperbolic-foveation Stage-1 CPU reference](#25-hyperbolic-foveation-stage-1-cpu-reference)
- [26. Foveation FiLM Stage-1 CPU reference](#26-foveation-film-stage-1-cpu-reference)
- [27. Depth-Anything Stage-1 CPU reference](#27-depth-anything-stage-1-cpu-reference)
- [28. Stage-1 axis/layout audit (bounded A1 list)](#28-stage-1-axislayout-audit-bounded-a1-list)
- [29. Stage-1 axis/layout audit of already-ported layers](#29-stage-1-axislayout-audit-of-already-ported-layers)
- [30. DPT-head Stage-1 CPU reference](#30-dpt-head-stage-1-cpu-reference)
- [38. YOLO-3D configuration preparation (not a layer port)](#38-yolo-3d-configuration-preparation-not-a-layer-port)
- [39. Cuboid labels and deterministic truth helpers](#39-cuboid-labels-and-deterministic-truth-helpers)
- [40. YOLO-3D raw CPU runtime and fail-closed consumers](#40-yolo-3d-raw-cpu-runtime-and-fail-closed-consumers)
- [41. Native f32 LAYA marker gather](#41-native-f32-laya-marker-gather)
- [42. Native f16 marker-gather inference](#42-native-f16-marker-gather-inference)
- [43. Native detached act-head feature primitive](#43-native-detached-act-head-feature-primitive)
- [44. Native exact-erf GELU primitives](#44-native-exact-erf-gelu-primitives)
- [45. Native act-head f32 forward composition](#45-native-act-head-f32-forward-composition)
- [46. Standalone native act-head gradient math](#46-standalone-native-act-head-gradient-math)
- [47. Native `[deform_conv]` forward and backward](#47-native-deform_conv-forward-and-backward)
- [48. DCNv4 native aggregation primitives (offsets given)](#48-dcnv4-native-aggregation-primitives-offsets-given)
- [49. Native DCNv4 projection and backward composition](#49-native-dcnv4-projection-and-backward-composition)
- [50. Bounded DCNv4 native runtime forward/backward](#50-bounded-dcnv4-native-runtime-forwardbackward)
- [51. Native DCNv4 rolling-stat BN](#51-native-dcnv4-rolling-stat-bn)
- [52. Native DCNv4 channel-normalization activations](#52-native-dcnv4-channel-normalization-activations)
- [53. Extend native DCNv4 kernel bound to 7x7](#53-extend-native-dcnv4-kernel-bound-to-7x7)
- [54. DCNv4 7x7 runtime optimizer parity](#54-dcnv4-7x7-runtime-optimizer-parity)
- [55. Extend native DCNv4 runtime through 9x9](#55-extend-native-dcnv4-runtime-through-9x9)
- [56. Extend native DCNv4 runtime through 11x11](#56-extend-native-dcnv4-runtime-through-11x11)
- [57. Extend native DCNv4 runtime through 13x13](#57-extend-native-dcnv4-runtime-through-13x13)
- [58. Retain native DCNv4 forward cache for backward](#58-retain-native-dcnv4-forward-cache-for-backward)
- [59. Native DCNv4 gradient accumulation](#59-native-dcnv4-gradient-accumulation)
- [60. Native act-head CLS adjoint scatter](#60-native-act-head-cls-adjoint-scatter)
- [61. Native DCNv4 SGD optimizer update](#61-native-dcnv4-sgd-optimizer-update)
- [62. Native DCNv4 Adam optimizer update](#62-native-dcnv4-adam-optimizer-update)
- [63. Native Revleaky activation mapping](#63-native-revleaky-activation-mapping)
- [64. Extend native DCNv4 through 15x15](#64-extend-native-dcnv4-through-15x15)
- [65. Retain native DCNv4 Adam moments on device](#65-retain-native-dcnv4-adam-moments-on-device)
- [66. Reuse DCNv4 projection weights from forward during backward](#66-reuse-dcnv4-projection-weights-from-forward-during-backward)
- [67. Retain native DCNv4 parameters across forwards and updates](#67-retain-native-dcnv4-parameters-across-forwards-and-updates)
- [68. Native GPU Muon orthogonalization primitive](#68-native-gpu-muon-orthogonalization-primitive)
- [69. Connected-layer GPU Muon runtime slice](#69-connected-layer-gpu-muon-runtime-slice)
- [70. Native locally connected layer](#70-native-locally-connected-layer)
- [71. GPU Muon RNN matrices](#71-gpu-muon-rnn-matrices)
- [72. GPU Muon LSTM matrices](#72-gpu-muon-lstm-matrices)
- [73. Native GPU SAM forward and backward](#73-native-gpu-sam-forward-and-backward)
- [74. Hybrid GPU Muon for convolution matrices](#74-hybrid-gpu-muon-for-convolution-matrices)
- [75. Hybrid GPU Muon for non-BN EML branches](#75-hybrid-gpu-muon-for-non-bn-eml-branches)
- [76. Hybrid GPU Muon for non-BN MoE](#76-hybrid-gpu-muon-for-non-bn-moe)
- [77. Hybrid GPU Muon for non-BN DCNv4 offset projection](#77-hybrid-gpu-muon-for-non-bn-dcnv4-offset-projection)
- [78. Hybrid GPU Muon for non-BN deform_conv](#78-hybrid-gpu-muon-for-non-bn-deform_conv)
- [79. Native Oxide GraphConv uniform aggregation primitives](#79-native-oxide-graphconv-uniform-aggregation-primitives)
- [80. GraphConv aggregation coordinate-overflow guards](#80-graphconv-aggregation-coordinate-overflow-guards)
- [81. Native GPU GraphConv uniform runtime path](#81-native-gpu-graphconv-uniform-runtime-path)
- [82. GraphConv learned-edge masked-softmax host contract](#82-graphconv-learned-edge-masked-softmax-host-contract)
- [83. Native GraphConv learned-edge forward primitive](#83-native-graphconv-learned-edge-forward-primitive)
- [84. Native GraphConv learned-edge adjoints](#84-native-graphconv-learned-edge-adjoints)
- [85. GraphConv learned-edge runtime path](#85-graphconv-learned-edge-runtime-path)
- [86. GraphConv center-reference gather primitive](#86-graphconv-center-reference-gather-primitive)
- [87. GraphConv self-projection runtime path](#87-graphconv-self-projection-runtime-path)
- [88. Compact GraphConv inference](#88-compact-graphconv-inference)
- [89. Compact GraphConv batch-normalization inference](#89-compact-graphconv-batch-normalization-inference)
- [90. Full GraphConv batch-normalization inference](#90-full-graphconv-batch-normalization-inference)
- [91. Hybrid GPU Muon for non-BN GraphConv matrices](#91-hybrid-gpu-muon-for-non-bn-graphconv-matrices)
- [92. GraphConv GPU Muon DNOPT6 sidecar resume](#92-graphconv-gpu-muon-dnopt6-sidecar-resume)
- [93. GraphConv GPU Muon update gate](#93-graphconv-gpu-muon-update-gate)
- [94. Grouped GraphConv GPU Muon parity](#94-grouped-graphconv-gpu-muon-parity)
- [95. GraphConv BN GPU Muon through CPU fallback](#95-graphconv-bn-gpu-muon-through-cpu-fallback)
- [96. Native GraphConv BN training](#96-native-graphconv-bn-training)
- [97. Standalone BatchNorm native GPU training](#97-standalone-batchnorm-native-gpu-training)
- [98. Native Crop training geometry](#98-native-crop-training-geometry)
- [99. DCNv4 mixed-network backward validation](#99-dcnv4-mixed-network-backward-validation)
- [100. Explicit GPU rejection for GRU training](#100-explicit-gpu-rejection-for-gru-training)
- [101. Region CPU forward and axis pin](#101-region-cpu-forward-and-axis-pin)
- [102. Native GRU GPU inference](#102-native-gru-gpu-inference)
- [103. Native RNN rolling-stat BN inference](#103-native-rnn-rolling-stat-bn-inference)
- [104. Native LSTM rolling-stat BN inference](#104-native-lstm-rolling-stat-bn-inference)
- [105. Native RNN rolling-stat BN training](#105-native-rnn-rolling-stat-bn-training)
- [106. Native LSTM rolling-stat BN training](#106-native-lstm-rolling-stat-bn-training)
- [107. Native CRNN rolling-stat BN training](#107-native-crnn-rolling-stat-bn-training)
- [108. Hybrid GPU Muon for CRNN matrices](#108-hybrid-gpu-muon-for-crnn-matrices)
- [109. Native GPU maxpool_depth](#109-native-gpu-maxpool_depth)
- [110. MaxPool axis/layout contract pin](#110-maxpool-axislayout-contract-pin)
- [111. AvgPool axis/layout contract pin](#111-avgpool-axislayout-contract-pin)
- [112. LocalAvgPool axis/layout contract pin](#112-localavgpool-axislayout-contract-pin)
- [113. Native f16 maxpool_depth inference](#113-native-f16-maxpool_depth-inference)
- [114. Connected axis/layout contract pin](#114-connected-axislayout-contract-pin)
- [115. Native f16 SAM inference](#115-native-f16-sam-inference)
- [116. Native f16 channel shuffle inference](#116-native-f16-channel-shuffle-inference)
- [117. Native f16 reorg inference](#117-native-f16-reorg-inference)
- [118. Native f32 Region GPU execution](#118-native-f32-region-gpu-execution)
- [119. Native f16 Region inference](#119-native-f16-region-inference)
- [120. Native f16 softmax inference](#120-native-f16-softmax-inference)
- [121. Native f16 channel-slice inference](#121-native-f16-channel-slice-inference)
- [122. Native f16 scale-channels inference](#122-native-f16-scale-channels-inference)
- [123. Native f16 dropout inference](#123-native-f16-dropout-inference)
- [124. Native f16 channel-normalization inference](#124-native-f16-channel-normalization-inference)
- [125. Native f16 connected inference](#125-native-f16-connected-inference)
- [126. Native f16 Cost identity and full ResNet101 coverage](#126-native-f16-cost-identity-and-full-resnet101-coverage)
- [127. Native f16 Detection and Regression identity paths](#127-native-f16-detection-and-regression-identity-paths)
- [128. Native connected-layer BN GPU backward](#128-native-connected-layer-bn-gpu-backward)
- [129. Connected BN GPU backward activation parity](#129-connected-bn-gpu-backward-activation-parity)
- [130. Connected BN GPU backward SGD/Adam parity](#130-connected-bn-gpu-backward-sgdadam-parity)
- [131. Route source/group axis pin](#131-route-sourcegroup-axis-pin)
- [132. Upsample axis pin and ragged reverse extent](#132-upsample-axis-pin-and-ragged-reverse-extent)
- [133. Reorg forward axis pin and reverse-source finding](#133-reorg-forward-axis-pin-and-reverse-source-finding)
- [134. ChannelShuffle axis pin](#134-channelshuffle-axis-pin)
- [135. Softmax grouped-reduction axis pin](#135-softmax-grouped-reduction-axis-pin)
- [136. Dropout element-mask layout evidence and DropBlock finding](#136-dropout-element-mask-layout-evidence-and-dropblock-finding)
- [137. ScaleChannels source and broadcast-axis pin](#137-scalechannels-source-and-broadcast-axis-pin)
- [138. SAM elementwise axis pin](#138-sam-elementwise-axis-pin)
- [31. C++ enum boundary config regression](#31-c-enum-boundary-config-regression)
- [32. Axis-coverage gate and shared NCHW fixtures](#32-axis-coverage-gate-and-shared-nchw-fixtures)
- [33. Convolutional axis pins and the dilation contract](#33-convolutional-axis-pins-and-the-dilation-contract)
- [34. GPU convolution extent parity](#34-gpu-convolution-extent-parity)
- [35. Annotation GUI (`darknet-gui`)](#35-annotation-gui-darknet-gui)
- [36. Annotation GUI plan implementation](#36-annotation-gui-plan-implementation)
- [37. DarkMark hotkey parity and model discovery](#37-darkmark-hotkey-parity-and-model-discovery)
- [Not implemented / open (pointer)](#not-implemented--open-pointer)

## 1. Workspace, build, tooling

- Flattened `rust/` to the repo root, vendored `cuda-oxide`, merged the
  workspace, pinned `nightly-2026-08-28` (`85fb0a91`).
- Dropped the `darknet-cli` cuTile dependency (the `cutile-rs-aot` checkout is
  missing; `src/darknet-cuda-cutile` is excluded and does not build — left
  disabled, re-enable only if the AOT fork returns).
- `cargo oxide` cannot run from the bridge dir; `src/darknet-cuda-oxide/README.md`
  rewritten.
- Build-race SIGBUS fixed via the codegen `output_name = "<crate>_test"` for the
  lib/lib-test `.ll`/`.ptx` race.
- `clippy-preview` component conflict on the pinned toolchain resolved (clippy
  reinstalled as a non-preview component).
- `src/darknet-cuda-oxide` is **not** a workspace member: build/test it with
  `./tools/test_oxide.sh` (`cargo oxide`); `tools/test_host.sh` is the host
  wrapper. Every cargo command uses `-j 4`/`-j 2` and is capped.

## 2. `cuda-oxide` GPU backend (P0–P9)

### P0–P1 — scaffold and elementwise
- P0: `src/darknet-cuda-oxide`, host upload/zeros/download (`e4a42d27`).
- P1: `bias_add`/`channel_scale`/`bias_backward`, 7 activations fwd+bwd,
  `vector_add`/`shortcut`/`saxpy`, SGD (`e4a42d27`); 7 parity tests.
- `bias_backward` block reduction decision (2026-10-01): kept the deterministic
  2-stage fixed-order partials (no atomics); already 205 GB/s = 75% of the
  272 GB/s roofline.

### P2 — Convolution
- `kernels/gemm.rs` (smem-tiled NN/NT/TN), `kernels/im2col.rs`
  (im2col+col2im with stride/dilation/padding/groups), `kernels/conv_direct.rs`
  (fwd/dgrad/wgrad); wired `conv2d_f32` + `conv2d_backward_{input,weights}_f32`.
- `DARKNET_OXIDE_CONV={direct,gemm,auto}` heuristic; cached conv workspace.
- Real-cfg shape coverage: scanned 101 cfgs / 5845 conv layers / 990 unique
  geometries (`tools/cfg_conv_shapes.py`, `tests/data/*.csv`); C++ cross-check
  3822 convs, 0 mismatches. Tests `cfg_conv_direct/gemm/gemm_tf32`, batch-2
  sampling, LegoGears_v2 at real size, dilation Err/parity, mismatched-shape
  shortcuts.
- relu6 + ramp added to the trait/oxide (CPU + GPU, exact output-form
  gradients); before this the CPU runtime silently treated them as identity.

### P2b — Tensor-core GEMM
- `kernels/gemm_mma.rs` mma.sync + ldmatrix, tf32 inputs, f32 accumulate;
  `DARKNET_OXIDE_GEMM={fp32,tf32,auto}`; conv fwd/dgrad/wgrad through it.
- Perf (2026-10-01): root cause of slow GEMMs was un-unrolled `while` loops
  putting accumulator arrays in local memory; new `kernels/unroll.rs` made
  indices constant (tf32 4.1 → 10.8 TFLOP/s at 2048³, 0 stack, 96 regs).
  Added `cp.async` two-stage pipeline and deterministic split-K. 21 LegoGears
  conv GEMMs fwd total fp32 2.211 → 0.872 ms, tf32 3.367 → 0.780 ms.
- `[~]` per-arch wgmma (sm_90) / tcgen05 (sm_100) dispatch does **not** exist;
  the seam is `GemmArgs` + `GemmMode`. Cannot be tested on this sm_89 GPU.

### P3 — Pooling / resample / route
- maxpool fwd/bwd (argmax), avgpool + local_avgpool, upsample (+reverse),
  route_concat / route_backward (grouped), crop_bilinear fwd/bwd; per-op parity.
- Mismatched-shape shortcut: C++ is a flat per-sample crop/pad, not a resample;
  implemented as `shortcut_fit_f32` (host default + oxide kernel), removing the
  old CPU-fallback trigger. `cfg_shortcut_shapes` covers all 18 distinct
  mismatched rows / 30 layers.

### P3b — YOLO head elementwise
- Head output kept RAW on CPU+GPU; logistic + `scale_x_y` applied on the host in
  decode/loss, parity with C++ covered by
  `gpu_yolo_head_matches_cpu_and_cpp_activated_decode`. Delta accumulation uses
  `backward_accumulate_seeded_gpu` incl. `loss_scale`.

### P4 — CLI + end to end
- `oxide` feature + `DARKNET_BACKEND=cpu|oxide`; dead cuTile cfg blocks removed.
- LegoGears_v2 inference: oxide == Rust CPU (score diff ≤5e-7, IoU ≥0.99999) ==
  C++ CUDA (score diff ≤4.5e-4, from C++ `cudnn_half=1`).
- JPEG decode fixed via `mozjpeg` (ISLOW + fancy upsampling): 0 pixel diff vs
  cv2 on 18 files; fixture test `jpeg_decode_matches_opencv_exactly`.
- Preprocessing parity documented: `detector test` = float `resize_image`,
  `detector map` = `cv::resize INTER_LINEAR_EXACT`; Rust `detect-image` modes
  `resize` | `letterbox` | `resize-exact`.

### P5 — CPU-fallback gaps
- Trait host defaults + `kernels/layers_extra.rs` for softmax, reorg, dropout,
  scale_channels, connected (fwd+bwd each); tests `oxide_layers_extra`.
- Runtime dispatch replaced the CPU fallback for those layers. Reorg backward
  bug fixed (was not the adjoint of the forward layout for c_in>1). Dropout mask
  is `dropout_unit(seed, i)`, identical on CPU and GPU.
- `NORM_CHAN*` f32 convolution support: CPU references and native Oxide forward
  and backward are integrated in runtime; exact adjoints are default and
  `DARKNET_GRAD_MODE=cpp` follows the inspected CUDA diagonal shortcuts (C++ CPU
  custom gradients differ). Parser allows only `[convolutional]`; f16 inference
  explicitly rejects these activations. No compiled-C++ golden is available.
- SAM uses native f32 pointwise forward/backward for mapped activations,
  including source-layer gradient fan-out; unsupported activations retain the
  whole-network CPU backward fallback (v0.2.051, §73). No-BN RNN/CRNN/LSTM
  backward uses backend ops; BN RNN/LSTM/CRNN backward also uses backend ops
  with a host forward/cache upload and host optimizer.

### P6 — FP16 (inference)
- Decision: additive opt-in `F16TensorOpsBackend` (f16 storage + f32 accumulate),
  not a f32-trait rewrite. `GemmMode::Fp16` in the f32 API is a compute mode.
- `kernels/convert.rs` exhaustive f32↔f16/bf16 conversion tests;
  `OxideTensorF16`; `kernels/gemm_f16.rs` mma m16n8k16 NN/NT/TN/TT + split-K;
  fused conv epilogue (bias + activation + RNE to f16); `half_ops.rs` for
  maxpool/upsample/route; `forward_compact_f16`.
- LegoGears_v2 fp16: 155/155 dets, max score diff 6.3e-4 vs C++, mAP 99.6685%
  (== fp32 == tf32). Bench: b1 fp16 1.79 ms (557 FPS), b16 9.6 ms; 4096³ GEMM
  25.9 TFLOP/s. Batched f16 im2col gave 19.0 → 9.55 ms b16.
- bf16 GEMM done; tests `oxide_half` (11) + `oxide_runtime_f16`.
- `[ ]` NOT done: f16 dgrad/wgrad, standalone f16 BN/bias/activation, `reverse`
  upsample.

### P7 — Benchmarks
- `tests/oxide_bench.rs` + `tools/oxide_bench.py` + `tools/oxide_bench_cpp.cpp`
  (events, median/min/p95, JSON+CSV, interleaved A/B); micro (GEMM/im2col/
  elementwise/activations/bias/pooling/upsample/route as GFLOP/s or GB/s and %
  roofline), 21 LegoGears conv shapes fwd/dgrad/wgrad, layer+e2e latency/FPS
  b1/b16, peak memory, nsys/ncu passes. Results in `docs/OXIDE_BENCH.md`.

### P8 — GPU training
- BN train fwd/bwd kernels (batch mean/biased var, exact dx/dscale/dbias,
  device rolling EMA, unbiased var); runtime uses them instead of the old
  rolling-stat round trips. C++ keep factors mirrored: GPU 0.99 / CPU 0.9.
- Optimizers: SGD momentum/decay; `adam_update_f32` (C++ semantics + AdamW via
  `decoupled_decay`), wired to `[net] adam=1`, per-owner device moments, Adam
  step state sidecar. Tests `gpu_adam_and_adamw_match_f64_reference_over_steps`.
- Loss scaling + NaN/Inf (`sanitize_nan_inf_f32`, Reset/Fix), subdivision
  accumulation, per-layer cfg keys (`learning_rate`, `dont_update`,
  `burnin_update`, `train_only_bn`, `clip`, `grad_centr`), `try_fix_nan`.
- Dropout/dropblock fwd+bwd (deterministic counter-hash RNG; DropBlock ramp and
  per-(sample,channel) block placement, exact `dy*mask` backward).
- Mixed precision training (AMP): `DARKNET_OXIDE_AMP=1`; f16 working copies, f32
  master weights/optimizer/BN state; `src/darknet-cuda-oxide/src/mixed.rs` +
  `kernels/amp.rs`; tests `oxide_amp` (10). Bench b16 82.7 → 62.9 ms.
  Real-device runtime-chain test with dropout + dropblock verifies masks,
  activations, backward, and SGD against f32; these stochastic layers remain on
  the documented f32 boundary path.
- Yolo loss static diff vs C++ (`tasks/rust/todo.md` table): terms verified line
  by line; fixes landed for box `(2-w*h)` scale, cost definition, `iou_thresh`
  extra anchors, best-anchor search, `new_coords` xy, `delta_normalizer`,
  equidistant/bad-label thresholds. Real step-1 loss now 0.191247 vs C++ 0.191136
  (delta L2 ratio 1.000004, was 70.78 vs 0.191).
- Training-step overhead: fused sanitize+SGD/Adam, device BN fold, deferred host
  sync, dropped zero-fills, multi-tensor update, fused inference bias+activation,
  batched dgrad/wgrad GEMM, streamlined im2col, vectorised f32 GEMM shared
  loads, auto-sized `BufPool`. b16 cumulative 125 → ~49 ms.
- Deferred gradient flush now has real-device `subdivisions=2` coverage over two
  optimizer iterations; `oxide_runtime_chain::subdivisions_deferred_gradient_flush_matches_cpu`
  matches CPU weights/biases after two sub-batch backwards per iteration.

### P9 — Training loop + data pipeline
- Startup/CLI: seeded init (C++ uniform bounds), explicit flag parser (`-clear`,
  `-dont_show`, `-gpus`, `-seed`), `-map`, `-save_every`, Adam resume.
- Data pipeline: jitter + paired resize + letterbox + source-mean padding +
  OpenCV-style u8 bilinear; flip; HSV; box clip/mirror; mixup; multi-scale
  `random`; background prefetch; C++-formula mosaic geometry (Rust RNG differs);
  background loader.
  20 OpenCV fixtures, worst 1 LSB. See `augmentation-audit.md`.
- Loop features: `<name>_best.weights`, periodic/`_last`/`-save_every`,
  `<name>_map.csv`, `<name>_loss.csv`, `chart.png`, `<name>_optimizer.state`,
  C++-field-order console output.
- `[x]` Convergence run (2026-10-02, `docs/OXIDE_BENCH.md` §9): oxide f32, oxide
  AMP and C++ CUDA each 3000 iters from the same init → all mAP@0.50 100.0000%;
  step medians oxide f32 129.1 ms / AMP 113.0 ms (debug).
- Recurrent device backward: no-BN RNN/CRNN/LSTM and BN RNN/LSTM/CRNN paths use
  backend BPTT, projection or convolution operations, with host reference cache
  uploads for BN forward. Rolling-stat BN scale-gradient reductions use backend
  centering/channel-scale/pointwise-multiply/channel-reduction ops; the Oxide
  path no longer transfers raw recurrent projection/convolution outputs and
  deltas to host just for dscale. Forward/cache capture, rolling-stat update,
  and optimizer remain host-side. FakeBackend and real Oxide parity cover
  upstream connected/convolution updates, BN scales, and rolling statistics;
  dynamic loss-scale checks inspect host-buffered connected/recurrent gradients.
- Connected/SAM device backward and host Adam for standalone batchnorm, local,
  weighted shortcut, MoE router/experts, deconvolutional, GRU/RNN/LSTM/CRNN,
  and ViT params with per-block moment state and per-layer rate scaling; ViT
  retains C++ weight order, decay groups, gradient clipping, and reduced mHC
  rate. `maxpool_zero_nonmax`; dynamic loss scale; AMP+Adam trajectory.
- `[ ]` Open: `reverse`/`deform` conv,
  adversarial yolo, implicit-GEMM as default, f32 GEMM epilogue, CUDA graph
  capture, multi-GPU/Blackwell.

## 3. FP8 / FP4 (`fp8_fp4.md`)

- **Phase A**: `kernels/gemm_f8.rs` Ada `mma.m16n8k32` e4m3/e5m2 tiled GEMM
  (64×64, BK=32, NN/NT/TN/TT, batched, split-K), per-tensor quantiser + amax,
  `GemmMode::Fp8` (`DARKNET_OXIDE_GEMM=fp8`). Fragment layout pinned by a wiring
  sweep on the real GPU; `src/fp8.rs` host reference.
- **Phase B**: FP8 dgrad/wgrad (e4m3 weights × e5m2 grads, C++ recipe),
  `DARKNET_OXIDE_GEMM_BWD=fp8`; 3.1%/2.2% worst relative error vs f32.
- **Phase C**: `kernels/quant_fp4.rs` + `kernels/gemm_fp4.rs` packed NVFP4
  (E2M1 + E4M3 block-16 scales + f32 global) via `__dp4a`, **bit-exact** against
  the CPU i32 reference (`src/fp4.rs`); `GemmMode::Fp4`. Emulation only (native
  FP4 mma needs sm_120a).
- **Phase D**: f32/tf32/fp16/fp8/fp4 GEMM bench table (RTX 4060 Laptop, sm_89;
  `gemm_bench_low_precision_kernels` / `gemm_bench_precisions`).

  Kernel only (operands pre-quantised):

  | shape | f32 `sgemm_tiled` | fp8 `mma.m16n8k32` | fp4 `__dp4a` |
  |---|---|---|---|
  | 512³ | 0.091 ms / 2946 GF/s | 0.102 ms / 2632 GF/s | 1.066 ms / 252 GF/s |
  | 1024³ | 0.468 ms / 4593 GF/s | 0.452 ms / 4754 GF/s | 7.35 ms / 292 GF/s |
  | 2048³ | 3.48 ms / 4940 GF/s | 2.45 ms / 7020 GF/s | 54.3 ms / 316 GF/s |

  End-to-end through the `DARKNET_OXIDE_GEMM` selector (fp8/fp4 include their
  quantisation; fp4 also the host transpose):

  | shape | f32 | tf32 | fp16 | fp8 | fp4 |
  |---|---|---|---|---|---|
  | 512³ | 0.822 ms | 0.595 ms | 0.483 ms | 1.356 ms | 54.0 ms |
  | 1024³ | 3.26 ms | 1.62 ms | 0.868 ms | 4.65 ms | 194 ms |
  | 2048³ | 5.50 ms | 1.80 ms | 1.274 ms | 5.90 ms | 785 ms |
  | LegoGears_v2 convs (21 total) | 5.95 ms | 4.82 ms | 3.87 ms | 24.2 ms | 1018 ms |

  Conclusions: the FP8 `mma` (7020 GF/s at 2048³) does **not** beat fp16
  (13485 GF/s) or TF32 (9560 GF/s) on this part — fp8 is a *numerics* feature
  here, not a speed one; per-call quantisation dominates below ~1024³, so a
  network-level fp8 mode must hoist weight quantisation out of the GEMM seam;
  the NVFP4 emulation is ~15x fp16 and only useful for parity with the C++
  `raptor-fp4` reference.
- Findings pinned on hardware (worth reading before enabling fp8 for speed):
  the FP8 `mma` fragment's A row selector is `group` for elements 0-4/8-12 and
  `group+8` for 4-8/12-16, **not** "elements >= 8" (a wrong-but-plausible
  result), pinned by a wiring sweep; B fragments must be staged **transposed**
  in shared memory (`[n][k]`, stride 36) or `ld.shared.b32` misaligns; e4m3
  exponent bias is **7** (smallest subnormal 2^-9), e5m2 has a 5-bit exponent
  (largest finite `0x7b`=57344); `cvt.rn.satfinite` maps every NaN to the
  *unsigned* NaN code (sign dropped); tie-to-even on the *code* with saturation
  applied before the search; tensor-core accumulation order floors comparisons
  at ~5e-5 of the row norm; the quantisation factor is the **reciprocal** of the
  C++ scale (`448/amax` as a multiplier, dequant `1/(fa*fb)`); `__dp4a` couples
  4 bytes in lane order 0..3 (bit-identical to C++).
- `[ ]` NOT done: step 5 (FP8 conv via im2col + LegoGears mAP vs the C++ FP8
  path) — the C++ build is unavailable here.

## 4. Layer port — Phase 0 (Tier 0 + Tier 1)

Commits `8e4b0f24`..`be71eb6e`; verified by `./tools/test_host.sh` (all green)
and `./tools/test_oxide.sh` (all suites `ok`). Tasks 1–12:

- Ten elementwise activations (`gelu elu selu relie plse hardtan lhtan stair
  hard_mish revleaky`) with exact input-form gradients; unknown activation names
  are a config error (no silent identity); `gelu` has an oxide kernel.
- `[maxpool]` reads `maxpool_zero_nonmax` and implements `maxpool_depth` (CPU).
- Standalone `[batchnorm]` can be saved (Rust-only extension).
- Weighted `[shortcut]` (`weights_type`/`weights_normalization`) with exact CPU
  gradients and `.weights` block (`shortcut_weights.rs`). GPU forward composes
  oxide channel-scale + add; backward assembles per-source products and applies
  exact normalization derivatives through backend ops. Only final small weight
  gradients transfer to host train buffers; optimizer/parameter updates remain
  host-side. FakeBackend and real-device runtime parity pass, including short
  source fit and an SGD step
  (`oxide_runtime_chain::weighted_shortcut_runtime_matches_cpu_forward_backward_and_sgd`).
- `[channel_slice]`, `[channel_shuffle]` (+ host op + oxide kernel),
  `[deconvolutional]` (CPU exact backward, GPU host round trip, conv-layout
  weights).
- Dense-truth heads: `[cost]` scale, `[logistic]`, `[regression]`, `[jev_ce]`
  + the `dense_loss` host API (`dense_loss.rs`, `jev_ce.rs`).

C++ dump parity: NOT RUN for any Phase 0 layer (no owning-branch C++ build).

## 5. Layer port — Phase 1b

- **`[segmentation]`** (2026-10-04, `3b42e1f2`): `segmentation.rs` +
  `SegmentationSpec`, binary logistic CE and per-pixel softmax CE, full key set,
  the correct-pixel `seg_gate` (raised-cosine ramp, per-element loss, global
  compensation). 18 tests. Open: label loader, `seg_gate` GPU kernel, cfgs.
- **`[omd_loss]`** (2026-10-04, `e5d8435b`): `omd_loss.rs` + `OmdLossSpec`, all
  three samplers (box truth, background LCG, classification), metric pair loss
  (cos / margin), full key set + aliases. 14 tests. `dense_loss` learns its
  non-output-shaped truth. Open: `show_details` reporting, real detector cfg.
- **`[contrastive]`** classifier branch (2026-10-04, `487b5c3e`):
  `contrastive.rs`, C++ `P[i][j]` definition, positive/negative gradient helpers,
  `label_smooth_eps` + `[net] adversarial`. 11 tests. Detection branch
  self-reports unported (needs the yolo head's per-box class ids).
- **`[laya_scorer]`** (2026-10-04, `e9457b4c`): `laya_scorer.rs` +
  `LayaScorerSpec`, frozen LayerNorm→Linear→GELU(erf)→Linear per option slot,
  exact analytical backward (FD-pinned), full shared-file blob arithmetic, 14
  tests. Validated against the real `test_data/laya/laya_english.weights` blob
  (LayerNorm gamma mean 0.9045, beta rms 0.0425). Needs `#![feature(float_erf)]`.
- **`[laya_marker_gather]`** (2026-10-05, uncommitted):
  `laya_marker_gather.rs` + `LayaMarkerGatherSpec` + `LayerKind::LayaMarkerGather`.
  Two route-style inputs (`layers=idxA,idxB`); no weights. CPU fwd/bwd in all
  three CPU paths, route-style use-counts, GPU-f32 host round trip, f16 error,
  GPU backward via CPU fallback. 10 tests (pure gather + exact adjoint +
  `lround` rounding/clamping + duplicate accumulation + runtime fwd/bwd/parse).
  Evidence: targeted 10 green, `darknet-core --lib` 317/317.
- **`[laya_act_head]`** (2026-10-05, uncommitted): `laya_act_head.rs` +
  `LayaActHeadSpec` + `LayerKind::LayaActHead`. Three route-style inputs
  (`layers=head,scorer,marker`); frozen 2-layer MLP over masked-softmax summary
  features (`top1`, `top1-top2`, entropy/log(k), k/255) concatenated with the
  head's CLS column. Softmax features are detached, so only the head source
  receives the pooled-CLS gradient. Blob order and shared-file offset ported
  (`blob_offset_bytes` reuses the scorer's encoder/head counts). CPU fwd/bwd in
  all three CPU paths, route-style use-counts, GPU-f32 host round trip, f16
  error, GPU backward via CPU fallback. 9 tests (hand-computed features incl.
  all-invalid options, MLP forward, central-difference backward for all four
  tensors, blob round trip + offset, shape errors, runtime fwd/bwd/parse).
  Evidence: targeted 9 green, `darknet-core --lib` 326/326.
- **`[laya_loss]`** (2026-10-05, uncommitted): `laya_loss.rs` + `LayaLossSpec`
  + `LayerKind::LayaLoss` + a `dense_loss` arm. Masked soft-target loss over the
  scorer's option logits, `mode=ce` (masked cross-entropy) and `mode=rlcd`
  (Gaussian-noised logits, strictly-proper reward, group-mean baseline / batch
  std, REINFORCE + weighted CE), plus `td_lambda_targets` for episode prefixes.
  Truth stride `2K+3`; padding samples (no valid option) are excluded from every
  mean. Forward output is the masked softmax; the delta is Darknet's descent
  direction (CE is FD-verified; RLCD is the intended policy-gradient surrogate,
  ported as-is). The RLCD RNG is a documented deterministic SplitMix64 +
  Box-Muller substitute (C++ uses libstdc++ `mt19937_64` + `normal_distribution`,
  not bit-reproducible). 11 tests (CE hand computation + FD, masking, padding,
  TD(lambda), RLCD hand formula, RNG determinism, shape errors, runtime
  forward/seed and RLCD smoke, parse errors). Evidence: targeted 11 green,
  `darknet-core --lib` 337/337.
- **`[laya_encoder]`** (2026-10-05, uncommitted): `laya_encoder.rs` +
  `LayaEncoderSpec` + `LayerKind::LayaEncoder` + a frozen runtime forward (CPU,
  plus a GPU-f32 host round trip). Frozen ModernBERT-large: token-embedding
  gather -> bias-free embed LayerNorm -> `n_layers` pre-norm blocks (fused-QKV
  attention, RoPE theta alternating full (i%3==0) / sliding local window,
  GeGLU with GELU on the first chunk) -> final LayerNorm -> channel-major output.
  Input `[batch,2,1,T]` (token id + mask), output `[batch,hidden,1,T]`. Blob
  order/count ported (`blob_float_count` == the scorer's
  `encoder_blob_float_count`). 9 tests (bias-free LayerNorm hand computation,
  RoPE identity-at-0 + norm preservation, zero-weight and pass-through
  invariants, padding-mask attention, blob round trip, shape errors, runtime
  forward vs module, parse errors). Evidence: targeted 9 green,
  `darknet-core --lib` 346/346.
- **`[laya_head]`** (2026-10-05, uncommitted): `laya_head.rs` + `LayaHeadSpec` +
  `LayerKind::LayaHead` + a frozen runtime forward (CPU + GPU-f32 host round
  trip, GPU/CPU backward no-op). Two plain pre-norm `nn.TransformerEncoderLayer`s
  (bias everywhere, ReLU FFN, no RoPE) over `encoder_output + type_emb[qtype]`,
  `qtype` from cfg `0..=2` or a per-sample input channel when `qtype=-1`.
  Input `[batch, hidden+mask(+qtype), 1, T]`, output `[batch, hidden, 1, T]`.
  Blob order fixed to the C++ layer-interleaved order with `type_emb` folded in.
  8 tests (LayerNorm-with-bias hand computation, zero-projection double-norm
  invariant, type_emb additive, per-sample qtype selection, blob round trip
  matching `head_blob_float_count`, shape errors, runtime forward vs module,
  parse errors). Evidence: targeted 8 green, `darknet-core --lib` 354/354.

**The LAYA family is now complete** (`laya_scorer`, `laya_marker_gather`,
`laya_act_head`, `laya_loss`, `laya_encoder`, `laya_head`).  C++ dump parity is
still NOT RUN (no `Jev_laya` build here; `test_data/laya/` has only the shared
`.weights`, no golden tensors).

## 6. Layer port — Phase 2A (anchor-free heads) — COMPLETE

- **`[yolox]`** (2026-10-04, `2281e019`): `modern_yolo.rs` + `ModernYoloSpec`,
  raw logits with sigmoid on trailing entries, SimOTA assignment, hard class
  targets, box/obj delta. 15 tests.
- **`[ppyoloe]` / `[yolonas]`** (2026-10-04, `eab4720d`): DFL distribution
  regression (softmax expectation, two-bin spread), Task-Aligned assignment,
  soft quality target, YOLO-NAS NMS-aware rank decay. 10 tests.
- **`[centernet]`** (2026-10-04, `aff18c3b`): `centernet.rs` +
  `CenterNetSpec`, sigmoid heatmap, adaptive anisotropic Gaussian, continuous
  tiny-object weighting, focal + smooth-L1 losses, peak NMS. 13 tests.
- **LegoGears nested-object fix** (same commit): per-slot `class_quality`,
  smallest-area-wins conflict resolution, `min_object_cells`, optional VFL.
- **Modern-head decode** (2026-10-04, `ce814b7e`):
  `ForwardResult.modern_heads` + `detect::decode_modern_detections` + CLI merge.
  Per-family scoring (YOLOX `obj*cls`, PP-YOLOE/YOLO-NAS `max(cls, obj*cls)`,
  CenterNet heatmap). 5 decode tests. Cfg end-to-end + `LegoGears_centernet.cfg`
  still open (cfg lives on `Jev_laya`).

## 7. Layer port — Phase 2B (partial: MoE + EML math)

- **`[moe_convolutional]`** CPU path (2026-10-05, `cdf68d00` "WIP Port"):
  `moe_conv.rs` (softmax top-k with tie-preserving mask + uniform fallback,
  global-average-pool router, load-balance loss + unclamped gradient, gate
  Jacobian), `MoeSpec`/`LayerKind::Moe`, C++ parse errors, C++
  `save_moe_convolutional_weights` blob order, CPU fwd/bwd/update, init.
  13 tests (10 `moe_conv` + 3 runtime, hand-computed blend + FD gradients).
  Oxide forward computes routing, cached expert convolutions, and per-sample
  blending on-device. No-BN and BN expert backward—including BN beta/scale
  gradients—use backend ops and accumulate gradients on-device. No-BN SGD/Adam
  updates are device-native; BN SGD updates expert weights, beta, and scales
  on-device with momentum buffers, and BN Adam uses DNOPT4 scale moments. DNOPT1-3
  remain readable, and nested host BN moments migrate on the first BN device
  Adam update. Owner/expert cache slots warm each expert; host-updated experts
  invalidate slots, while device-updated caches support deferred host sync.
  CPU parity: `oxide_runtime_chain::moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu`,
  `oxide_runtime_chain::moe_adam_updates_router_and_experts_on_oxide_match_cpu_gradients`,
  `oxide_runtime_chain::moe_batchnorm_expert_device_backward_matches_cpu_on_oxide`,
  and `oxide_runtime_chain::moe_batchnorm_adam_updates_experts_on_oxide_match_cpu_gradients`.
  Remaining: `eml_convolutional` and `dcnv3` (no reference numerics).
- **`[eml_convolutional]`**: v67 added `eml_conv_math.rs`, a source-derived
  pure forward/backward reference for the clamped exponential / softplus branch
  combination and optional residual, with exact default branch derivatives,
  strict clamp-boundary handling, C++ nonpositive clamp/epsilon fallbacks, and
  finite-difference/hand-reference tests (4/4). v68 added host and Oxide combine
  ops; v69 added cfg/shape parsing, paired convolution parameters, CPU forward /
  backward, `.weights` I/O, SGD/Adam, and host updates. Save order **verified
  2026-10-05**: `weights.cpp:775`/`:2000` write/read `input_layer` then
  `self_layer`. v70 added native f32 Oxide branch convolutions and rolling-stat
  inference. v71 added no-BN native backward. v72 added per-branch batch-stat
  BN caches, native BN and convolution backward, scale/beta/weight gradient
  accumulation, and device-produced rolling EMA propagation; parameter
  optimizers remain host-side. `MEM=3G PKGS=darknet-core
  ./tools/test_host.sh eml_convolutional` passes 3/3, and
  `./tools/test_oxide.sh --test oxide_runtime_chain` passes 19/19. CPU EML
  training uses rolling stats, unlike the native GPU batch-stat training path;
  no CPU/C++ training parity is claimed. Device optimizers and compiled-C++
  golden validation remain open.
- **`dcnv3`**: no reference numerics (stub in all five branches).

## 8. Layer port — Phase 3A shared primitives (prerequisite set)

- GPU LayerNorm fwd/bwd (`1f3c47b1`): host references + oxide kernel
  (thread-per-row, `LAYERNORM_MAX_COLS=1024`), bias-free variant; `oxide_layernorm`.
- Fused residual + per-column bias epilogue (`206accce`).
- Batched-GEMM wrapper `linear_f32` (torch `nn.Linear` layout, bias as GEMM
  `beta=1`) (`593ea30a`).
- Row-softmax backward already existed. The shared set supports `vit`,
  `swin`/`transformer`, Tucker, MambaVision, and Kimi KDA reference paths.
  MiniMax sparse attention is an optional shared utility, not a
  standalone layer in the audited C++ branch; it is wired into ViT, Tucker, and
  MambaVision.

### `[vit]` reference layer path (Phase 3A)
- `model.rs`: cfg aliases/defaults/validation and patch-grid shape inference;
  `Layer::weights()` matches the C++ named block order.
- `vit.rs` + `runtime.rs`: CPU forward and exact analytical backward; seeded
  C++-scale initialization; training gradient accumulation and SGD update
  (global norm clip 5, matrix + learned-position decay, 0.1x mHC LR/clamp).
- `weights.rs`/runtime: modern save and header-versioned legacy loading for the
  12-float mHC suffix; exact-size and round-trip tests cover both.
- GPU f32 uses a host round-trip for non-attention math. Standard flash
  attention dispatches native Oxide forward and recompute-based dQ/dK/dV
  backward for supported head dimensions inside the whole-network CPU fallback;
  unsupported dimensions and other ViT derivatives remain host-side. FakeBackend
  and Oxide runtime-chain parity cover a preceding-convolution chain.
- v0.0.260 (`16e95d9a`): native flash backward and runtime dispatch. Evidence:
  `MEM=3G ./tools/test_host.sh vit` — 28 pass;
  `MEM=3G ./tools/test_oxide.sh --test oxide_layernorm` — 14 pass, including
  host/Oxide backward parity, central finite differences, and clamp masks;
  `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain` — 22 pass, including
  CPU-matched ViT parameter updates after native attention backward dispatch;
  `MEM=3G ./tools/test_host.sh runtime` — 161 runtime tests pass.
  This is Rust backend evidence, not performance or compiled-C++ parity.
- v0.0.261 (`7ca5a58e`): added source-derived host E4M3 saturating encode/decode
  and ExpCast modes 0/1/2 as VC-attention groundwork. Four tests cover every
  finite E4M3 code round trip, special/subnormal/saturation cases, hand-derived
  ExpCast outputs, and invalid inputs. `MEM=3G ./tools/test_host.sh vc_attention`
  passed 4 tests. Helpers are not wired into cfg/runtime; no compiled-C++ parity.
- v0.0.262 (`18ad3547`): added the source-style V-Smooth transformation with
  per-64-token means, global per-channel residual scale, permutation support,
  E4M3 residual reconstruction, and a piecewise-exact derivative. Backward
  explicitly errors at E4M3 rounding boundaries and tied/zero scale maxima,
  where the classical derivative is undefined.
  `MEM=3G ./tools/test_host.sh vc_attention` passed 7 tests, including central
  finite differences away from quantization boundaries. This is an isolated math
  primitive, not an integrated VC attention path or compiled-C++ parity.
- v0.0.263 (`8792f7b9`): added a source-shaped one-query online ExpCast row
  forward/backward. The cache is O(T + T*C); its exact score derivative includes
  continuous inter-tile prefix-max rescaling, while code transitions and prefix
  max ties return an explicit error. `MEM=3G ./tools/test_host.sh vc_attention`
  passed 10 tests, including two-tile finite differences and the single-tile
  zero-score-gradient case. Still isolated math; no cfg/runtime or C++ parity.
- v0.0.264 (`9a085535`): added one-head Q/K quantization and continuous score-scale
  derivative chaining through per-row `sq` and centered-key `sk`, including
  key-mean and Q/K-clamp adjoints; E4M3 code derivatives remain zero and exact
  boundary/tie cases error. `MEM=3G ./tools/test_host.sh vc_attention` passed
  12 tests, including finite differences of Q/K scale paths with a key
  permutation and score-clamp masks. Still not assembled into full attention.
- v0.0.265 (`fabcfba5`): composed those pieces into a single batch-head CPU
  reference forward/cache/backward with O(T*C) retained state; online rows are
  recomputed in backward. LSE is explicitly for the centered, configured-clamp
  score rows, not ordinary uncentered logits. `MEM=3G ./tools/test_host.sh
  vc_attention` passed 13 tests, including Q/K/V finite differences over 128
  keys with a key permutation. This remains reference math, not cfg/runtime or
  compiled-C++ parity.
- v0.0.266 (`fc47742d`): added `[B,H]` packing/unpacking around the single
  batch-head VC reference, storing one O(T*C) cache per head and returning
  gradients in fused QKV order. `MEM=3G ./tools/test_host.sh vc_attention`
  passed 14 tests, including B=2/H=2 mapping and cache-shape coverage. Config,
  runtime, device fallback, and compiled-C++ parity remain open.
- v0.0.267 (`3c6f5c76`): integrated VC aliases (`vc_attention`/`vc`,
  `vc_expcast`/`expcast`) and strict mode/permutation validation into `[vit]`,
  enforcing the flash-attention prerequisite. Added the source 8×8 full-windows-
  first spatial ordering and routed CPU/GPU ViT forwards through the VC reference;
  GPU paths bypass standard flash kernels, and backward consumes the matching VC
  cache via the exact piecewise derivative. Unsupported GPU devices/head sizes
  are handled by the host reference (no native VC kernel). `MEM=3G
  ./tools/test_host.sh vc_attention` passed 16 filtered tests and
  `MEM=3G ./tools/test_host.sh vit` passed 32, including config validation,
  callback bypass, spatial-order, and FakeBackend runtime forward/backward tests.
  No compiled-C++ parity is claimed; remaining Phase 3A layers and native VC
  kernels remain open.
- Earlier focused evidence: full host unit suites pass (`darknet-core` 377,
  `darknet-cli` 18); targeted check/build pass with `-j 4`. The full
  `cfg_corpus` integration run exceeded a 30-minute timeout. C++ dump parity NOT
  RUN (no owning-branch executable).
- v0.0.268 (`0bd26bda`): added source-derived Swin geometry, masks, mHC, and
  one-window attention helpers with analytical derivatives. `MEM=3G
  ./tools/test_host.sh swin` passed 10 focused tests covering hand attention,
  finite differences, source padded-key leakage, and rectangular pad/crop/shift/
  partition adjoints. Hand-derived from `jepa:swin_layer.cpp`; no compiled-C++
  golden.
- v0.0.269 (`4addcb07`): integrated the complete Rust Swin/transformer layer. `model.rs`
  strictly parses all four aliases, infers shape, and records the Rust-owned
  weight order; initialization and modern-header weights save/load are covered.
  `swin_block.rs` implements full CPU forward/backward and exact derivatives of
  that Rust forward; `runtime.rs` adds gradient buffers, accumulation, SGD/Adam,
  optimizer-state round-trip, and GPU f32 host fallback. Masks and relative
  indices are built once per block forward, with indices shared across window
  caches. `MEM=3G ./tools/test_host.sh swin` passed 25 tests, `runtime` passed
  166, and `weights` passed 29. FakeBackend matches CPU forward/backward and
  updates on a preceding-convolution chain. No native Swin kernel, C++ weights
  writer, or compiled-C++ parity is available.

### `[tucker_attention]` reference layer path (Phase 3A)

- v0.0.270 (`ecd4e54e`): added strict `[tucker]` / `[tucker_attention]` / `[tucker_attn]`
  parsing, NCHW shape validation, rank/default and MSA alias precedence, and
  overflow-safe geometry/weight sizing. `Layer::weights()` follows the C++
  two-block stream: `biases[channels]`, then flat Q/K/V/O factors.
- Added source-shaped bottom/right zero-padded non-overlapping local windows,
  Q/K/V low-rank bases and per-head cores, O-core/basis, identity residual,
  activation, optional MiniMax sparse-row selection, and exact backward of the
  Rust forward. Backward finite differences cover input, Tanh, MSA routing, and
  all eight factor blocks; a hand-computed 2×2 window and multi-head derivative
  are pinned. Seeded init uses the source uniform bound; SGD/Adam update only
  flat weights with coupled decay, not biases.
- `weights.rs` writes biases then factors; modern and legacy headers round-trip
  with exact metadata/trailing-byte checks. GPU f32 uses a host round-trip; no
  native Tucker kernel exists. `MEM=3G ./tools/test_host.sh tucker_attention`
  passed 14 tests; `weights` passed 32 and `runtime` 168, including FakeBackend
  CPU forward/backward/update parity and Adam fallback. Source-derived from
  `Jev_laya`/`jepa`; no compiled-C++ golden or parity claim.
- MiniMax sparse attention is a shared option rather than a standalone layer,
  and its source enablement must be audited in each
  future attention integration. VC native-kernel/evidence follow-ups also
  remain; see `todo.md`, `vit_plan.md`, and `swin_plan.md`.

### `[mambavision]` reference layer path (Phase 3A)

- v0.0.271: added strict `[mambavision]` / `[mambavision_block]` parsing,
  shape/geometry checks, source-ordered flat parameters, deterministic seeded
  initialization, and an explicit per-layer recurrent-state API. The source
  recurrence, causal/noncausal depthwise convolution, post-scan identity-QKV MSA,
  residual, LayerNorm, and FFN are implemented in `mambavision.rs`.
- CPU backward differentiates the Rust forward exactly, including scan clamp
  masks, MSA score clipping, and incoming `[batch, filters/2, state]` gradients;
  top-k selection is held fixed and undefined boundary derivatives error.
  Runtime accumulation, per-block SGD decay, host Adam, and optimizer-state
  serialization are integrated. GPU f32 uses the host reference/fallback.
- `.weights` metadata and load/save follow the audited C++ block order, including
  the optional residual projection only when input channels differ from filters.
  Tests cover aliases/defaults/errors, seeded initialization bounds, input and
  every parameter-block finite differences, incoming-state finite differences,
  causal-convolution finite differences, conv-size-1 streaming, MSA clipping,
  exact-length weight round trips, optimizer-state persistence, SGD decay groups,
  and CPU/FakeBackend forward/backward/update parity.
- Evidence: `MEM=3G ./tools/test_host.sh mambavision` — 16 pass;
  `MEM=3G ./tools/test_host.sh runtime` — 171 pass;
  `MEM=3G ./tools/test_host.sh weights` — 33 pass. No native MambaVision
  kernel or owning-branch compiled-C++ golden/parity.

### `[kimi_kda]` reference layer path (Phase 3A)

- v0.0.272 (`3d7202bf`): added strict aliases/config and identity-shape checks,
  source-order biases/weights, deterministic source-bounded initialization, CPU
  low-rank causal scan plus shared/routed experts, and exact backward of the
  Rust forward (including router load-balance cost and piecewise key-normalization
  behavior). SGD/Adam, optimizer-state persistence, `.weights`, and GPU f32 host
  fallback are integrated; no native Kimi kernel exists.
- Tests cover hand-computed forward, finite differences for input and every
  parameter with the auxiliary cost, key-norm floor and top-k edge cases,
  parser errors, initialization bounds, source-order weight round-trip, optimizer
  persistence, and CPU/FakeBackend forward/backward/update parity.
- Evidence: `MEM=3G ./tools/test_host.sh kimi_kda` — 10 pass;
  `MEM=3G ./tools/test_host.sh runtime` — 173 pass;
  `MEM=3G ./tools/test_host.sh weights` — 34 pass. Hand-derived from `Jev_laya`;
  no owning-branch compiled-C++ golden or parity claim.

## 9. Inference parity — LegoGears_v2 (the final target)

`docs/OXIDE_BENCH.md`, `tasks/rust/todo.md` "Final target results":
- 38-layer dump: Rust CPU vs oxide max abs 2.1e-5; C++ CUDA vs oxide up to
  5.7e-3 but the same profile as C++ vs Rust CPU → C++ `cudnn_half=1`, not an
  oxide error.
- 80 val images: 155/155 detections matched at thr 0.25, min IoU 0.99897, max
  score diff 4.5e-4; mAP@0.50 99.6685% == C++ 99.67% (fp32/tf32/fp16).
- Perf after allocator + GEMM + batched-im2col work: b1 ~1.8–2.4 ms (C++ ~5),
  b16 9.6–14.7 ms (C++ ~14–15).
- Known small diffs vs C++ (documented): CPU resize uses `image` Triangle; EXIF
  orientation ignored; dilated conv needs `pad*dilation`; relu/relu6 NaN→0 on
  oxide; mish/swish at -inf → NaN.

## 10. Data augmentation (`augmentation-audit.md`)

- Implemented: `crates/darknet-core/src/augmentation.rs` + training-only loader +
  CPU/oxide CLI wiring; jitter/resize margins, letterbox, source-mean padding,
  OpenCV-style u8 bilinear, flip, HSV, box clip/mirror/subpixel filter, max-box
  cap, mixup, multi-scale `random`, seeded SplitMix64, background prefetch, and
  C++-formula mosaic crop alignment/box clipping (Rust RNG differs).
- Reference generator `tools/augmentation_reference.py`; 20 composite fixtures
  worst 1 LSB (not claimed bit-identical). Unsupported active
  blur/noise/fog/tracking/contrastive modes fail at startup.
- Open: resume image-order replay, C++-bit-identical RNG.

## 11. Housekeeping

- Root `TODO.md` updated with backend status + cuTile-disabled note.
- Commit discipline: minimal per-phase commits on branch `rust`.
- Stale-box audit (2026-10-01) resolved most open checkboxes with inline
  evidence.

## 12. Gradient-shortcut audit (`DARKNET_GRAD_MODE`)

Audited line-by-line against `src-lib` on 2026-10-01; `GradMode` lives in
`src/darknet-core/src/runtime.rs`. Keep exact analytical backward as the default;
`DARKNET_GRAD_MODE=cpp` reproduces C++ shortcuts for softmax, scale_channels
(`scale_wh=1`), and weighted shortcut. Other ops are exact and untouched.

1. **softmax** — `softmax_layer.cpp:97`/`:146` pass `dy` through; exact is
   `dx_i = y_i (dy_i - sum_j dy_j y_j)`. Rust exact by default, `cpp` =
   passthrough (CPU + GPU arms).
2. **scale_channels** (`scale_wh=1`) — `blas_kernels.cu:1683` divides the scale
   gradient by `w*h`; exact is the plain sum over c. `scale_wh=0` (`:1690-1704`)
   exact. Rust exact by default, `cpp` divides by `w*h`.
3. **weighted shortcut** — `blas_kernels.cu:1069-1070`/`:1109-1110` use RELU
   `grad=w` and SOFTMAX `grad=w*(1-w)` (cross terms and `eps=1e-4` dropped);
   unweighted (`:1091`/`:1131`) = exact identity (FD-tested). Rust exact CPU
   backward by default; `cpp` selects raw `sum(delta*x)` for normalized weights.
   GPU fwd composes oxide scale+add; GPU backward assembles each source's raw
   products and applies the exact normalization derivative on the backend. Only
   the final small weight-gradient vector crosses to the host train buffer;
   parameter/optimizer updates remain host-side. Finite differences cover all
   normalizations; FakeBackend and real Oxide tests cover short-source prefixes,
   C++ mode, runtime backward, and SGD parity.
4. **activations** — `activation_kernels.cu:62-85`; gradients take the output
   except mish/swish/hard_mish. Exact from output for
   logistic/tanh/loggy/elu/selu/relu/leaky/relie/ramp/relu6/hardtan/lhtan/plse/
   hard_mish; stair = 0 a.e. **WRONG in C++**: gelu `:78-81` / `:450-454`
   evaluates the derivative at the *output*. Rust oxide implements only
   linear/logistic/relu/leaky/mish/swish/tanh, all exact.
5. **dropout** exact (`delta = mask*scale`); dropblock mask×per-sample scale
   treated as constant. Rust exact (FD with fixed mask).
6. **batch norm** — cuDNN path (`batchnorm_layer.cpp:312-331`) exact; non-cuDNN
   path exact up to epsilon placement (forward 1e-5 vs delta kernels 1e-6);
   `fix_nan_and_inf`/`constrain_ongpu` clamps not implemented in Rust (exact,
   eps 1e-5, FD-tested).
7. **conv** — exact dgrad/wgrad are FD-tested. The GPU update-only `reverse`
   transform and `deform` weight transform remain unimplemented. `grad_centr`
   is implemented in CPU/GPU update paths with the C++ BN gate and warp-size
   skip; `gpu_grad_centr_runtime_matches_cpu_and_respects_batchnorm_gate` and
   `gradient_centralization_matches_reference_and_is_noop_off_warp_sizes`
   verify runtime/reference parity.
   Weight clipping and the `try_fix_nan` sanitization path are also implemented.
8. **yolo/region/gaussian_yolo** deltas are a training policy, not the
   printed-loss derivative (`delta_yolo_box` `yolo_layer.cpp:76-178`); Rust
   mirrors on the host, no kernel, nothing to gate.
9. **maxpool** — exact subgradient, ties to the first tap (`:73-75`); `maxpool_depth`
   exact. SHORTCUT `maxpool_zero_nonmax` (`:135-142`) zeroes non-argmax input
   deltas; `antialiasing` (`:220-232`) runs an extra conv backward. Rust: exact
   maxpool; `zero_nonmax`/`depth`/`antialiasing` noted.
10. **update step** (CORRECTED) — `loss_scale` multiplies softmax and yolo GPU
    deltas and divides lr in every update; SGD decay `axpy(-decay*batch*loss_scale, w)`;
    Adam decays `-decay*batch*w` without `loss_scale`. Out of GRAD_MODE scope.
11. **NORM_CHAN_SOFTMAX / NORM_CHAN** CUDA gradients drop the normalization
    cross terms (`activation_kernels.cu:788-874`). v65 added exact standalone
    backend derivatives and v66 wired them into f32 convolution CPU/Oxide
    runtimes; `DARKNET_GRAD_MODE=cpp` selects the inspected CUDA shortcut. CPU
    C++ custom gradients differ. The parser permits these only on
    `[convolutional]`; f16 inference remains unsupported.
12. **gelu gradient at the output** (see 4).

FD coverage: every audited op with a Rust path is finite-difference tested; the
unweighted shortcut gained
`shortcut_add_backward_is_identity_and_matches_finite_difference`
(`oxide_grad_methods.rs`); weighted shortcut's None/ReLU/softmax derivatives
have central finite-difference coverage in `shortcut_weights::tests`, and the
v65 `NORM_CHAN*` standalone exact derivatives are finite-difference checked in
`oxide_layers_extra`. Not implemented/testable by finite differences yet: gelu &
friends, conv `reverse`/`deform`, and the YOLO kernel. `grad_centr` is an optimizer transform, verified by source-derived
runtime/kernel parity rather than finite differences.
`maxpool_zero_nonmax` uses the existing Oxide maxpool-backward kernel and has
real-device CPU parity coverage.

---

## 13. OBB CPU training gaps (plan-open-items Waves 0–1)

Evidence: `MEM=3G ./tools/test_host.sh obb_` = 135 core + 4 CLI tests pass.
No compiled-C++ parity (hand-derived from `raptor:src-lib/*.cpp`, no golden).

- v0.0.247 (`98003667`): `ForwardResult::from_heads`; fixes the oxide CLI build break.
- v0.0.248 (`2941a2ee`): default CUDA toolkit path in `.cargo/config.toml` (not part of the plan).
- v0.0.249 (`f9664e63`): OBB CPU Adam (`update_weights_adam`, moments round-trip via sidecar).
- v0.0.250 (`5b98fe1c`): OBB photometric augmentation (hue/saturation/exposure).
- v0.0.251 (`968c1beb`): weighted OBB NMS merge (`obb_w`/`obb_wnms`).
- v0.0.252 (`fae49267`): OBB default augmentation settings (zero default jitter).
- v0.0.253 (`31219aba`): `ciou_obb_mul` multiplicative training loss.

---

## 14. ONNX export

- v0.0.254 (`fc50aa57`): added a dependency-free protobuf wire encoder and ONNX message builders; varint, signed-int, model wire structure, and raw float tensor tests pass.
- v0.0.255 (`1b632932`): implemented supported-layer ONNX export for Conv/BN, activations, maxpool, route, shortcut, upsample, and YOLO heads; unsupported constructs fail explicitly. `MEM=3G ./tools/test_host.sh onnx` passed 11 tests, including asymmetric maxpool padding, grouped-route source-channel slicing, separate BN, unsupported dilation/shortcuts, Mish opset 18, and no-YOLO outputs.
- v0.0.256 (`4a0d99e7`): added `export-onnx` CLI parsing, default output path, `--no-fuse`, argument rejection, and usage text. `MEM=3G ./tools/test_host.sh export_onnx_args` and `systemd-run --user --scope -p MemoryHigh=3G cargo build -p darknet-cli -j 2` passed; fused and unfused model exports from a local matching cfg/weights succeeded.
- v0.0.257 (`778242ab`): added `tools/onnx_check.py`; syntax check passed. ONNX Runtime checks against Rust CPU layer dumps yielded worst relative errors of `3.257e-06` (fused), `3.908e-05` (unfused), and `3.831e-06` (Mish-opset-18 variant), all below `1e-4`. Verified with local cfg/weights/image because the canonical dataset paths are absent. This is Rust-runtime parity; no compiled-C++ golden exists.

## 15. CARPN sample-level polygon mAP dispatch

- v0.0.258 (`7681f478`): `obb_detector_map` now dispatches all-CARPN sample sets to the existing polygon evaluator, rejects mixed front-point/CARPN predictions, and preserves the legacy route for empty prediction sets. `MEM=3G ./tools/test_host.sh obb_map` passed 5 core tests, `MEM=3G ./tools/test_host.sh carpn_map` passed 3, and `MEM=3G ./tools/test_host.sh carpn_polygon_iou` passed 2. This is caller-provided sample API support only: Rust CARPN network decoding, CLI/training mAP integration, and compiled-C++ parity remain unsupported/not established.

## 16. CARPN CPU network inference and direct mAP

- v0.0.259 (`29889c5c`): added `[obb_refine]` CPU inference, Stage-2 quadrilateral decoding/NMS, full-forward OBB validation with explicit polygon mAP, `detect-image` integration, and a cfg/weights-backed direct `detector map` route. Generated fixtures verify AP/report counts, a same-AABB/wrong-polygon case (AABB AP 1.0 versus polygon AP 0.0 at IoU 0.6), and required front-point labels even with zero proposals. CARPN `detector train` including `-map` rejects before training because Stage-2 assignment/loss/backward/update support is not implemented. Legacy OBB behavior is preserved (`MEM=3G ./tools/test_host.sh obb_`: 138 core + 4 CLI tests).
- Focused evidence: `carpn_refine_cfg` (2), `carpn_refine_weights` (1), `carpn_refine_forward` (3), `carpn_proposals_apply` (1), `carpn_refine_decode` (5), `carpn_map` (6), `carpn_cli` (1), `detect_image_routes_carpn` (1), and `detector_train_rejects_carpn` (1) all passed through `MEM=3G ./tools/test_host.sh`. `systemd-run --user --scope -p MemoryHigh=3G cargo build -p darknet-cli -j 4`, targeted `rustfmt --check`, and `git diff --check` passed. Workspace-wide `cargo fmt --check` reports pre-existing formatting differences in unrelated ONNX/CUDA files; those were not reformatted. No full cfg-corpus run was required.
- Source behavior is hand-derived from `raptor`; there is no compiled-C++ oracle or canonical CARPN cfg/weights/dataset, so no compiled-C++ parity or canonical-dataset result is claimed. GPU CARPN inference and CARPN training/scheduled `detector train -map` remain deferred.

## 17. DCNv4 CPU reference and GPU f32 host fallback

- v0.0.274 (`e24d7625`): mirrors `Jev_laya:src-lib/darknet_cfg.cpp` by defaulting `[net] try_fix_nan=1` when a deformable-sampling layer is present, unless the net explicitly sets `try_fix_nan`. The `dcnv4` host filter passed 16 tests, including explicit zero/one override coverage. No compiled-C++ parity.
- v0.0.273 (`cb96aa49`): integrated strict `[dcnv4]` / `[dcnv4_convolutional]` config and shape support, C++ source-order `.weights`, deterministic initialization, grouped deformable sampling, coarse `d_stride` offset projection and interpolation, optional softmax masks/center removal, and the exact derivative of the Rust forward. Runtime support includes bias/BN/activation, SGD and Adam, persisted optimizer state, and GPU f32 execution through the host fallback. Compatibility dummy weights serialize but are never differentiated or updated; offsets use the source-derived 0.01 learning-rate scale.
- Verification: `MEM=3G ./tools/test_host.sh dcnv4`, `weights`, and `runtime` passed; the targeted `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4_f32_host_fallback_matches_cpu_chain_and_updates` passed. Output counts were not retained. Tests cover hand-computed sampling/grouping/masks, finite differences for input and offset parameters across standard and coarse-stride cases, BN/optimizer behavior, exact-length BN/non-BN weight round trips, and CPU/FakeBackend/Oxide host-fallback parity. Targeted Rustfmt and `git diff --check` passed. No training was run, no native Oxide kernel was added, and no compiled-C++ golden/parity is claimed.

## 18. DCNv1 and DCNv2 CPU reference and GPU f32 host fallback

- v0.1.001 (`8f5a9a94495cbc376f0b91cbf12f5123fe7ee42c`): integrated both `[deform_conv]` aliases with strict source-derived cfg/shape validation, parameter metadata, deterministic initialization, exact source-order weights, grouped NCHW sampling, offsets and optional sigmoid masks, per-image conditional offset normalization, and exact backward of the Rust forward. Runtime covers bias/BN/activation, rolling statistics, SGD source-rate groups, Adam with persisted moments, update gates, loss scaling, CPU/FakeBackend parity, and GPU f32 host fallback; compact execution rejects explicitly. Unsupported modes, f16/compact paths, and undefined clamp boundaries reject explicitly.
- Verification: `MEM=3G ./tools/test_host.sh deform_conv` passed 23 core tests; `weights` passed 38 core tests; `runtime` passed 184 runtime tests; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- deform_conv_f32_host_fallback_matches_cpu_chain_and_updates` passed 1 Oxide test. Named coverage includes NCHW/group/kernel-axis hand math, zero-padded boundary sampling, all-trainable-block/input finite differences (including conditional normalization and masked grouped sampling), BN scale finite differences and rolling stats, source-order byte layouts, SGD/Adam updates and state round trip, accumulation/gates/loss-scale neutrality, and CPU/FakeBackend/Oxide host fallback. Targeted Rustfmt and `git diff --check` passed.
- Source contract: hand-derived from `Jev_laya`; no compiled-C++ golden/parity. No training run or native Oxide kernel. GPU status is host fallback only. CPU Muon was delivered separately in v0.1.005; this bounded DSConv port still does not complete Phase 3B.

## 19. DSConv single-template Stage-1 CPU reference

- v0.1.002 (`f49cfc56f571329c14f6b43224d635199b6cca1b`): integrated `[dsconv]` / `[dynamic_snake_conv]` with strict source-derived config/shape handling, source-order weights, seeded initialization, NCHW CPU forward for `morph=0/1/2`, exact backward of the Rust forward, activation handling, SGD source-rate groups, and Adam with persisted moments. Multi-template `morph=0/1` explicitly rejects; `morph=2` follows the source normalization to one template.
- Verification: `MEM=3G ./tools/test_host.sh dsconv` passed 12 core tests, `weights` passed 40 core tests, and `runtime` passed 187 runtime tests. Coverage includes named hand-computed NCHW axes, finite differences for input and every parameter block across all morphs, integer/border derivative conventions, exact source-order bytes/file length, seeded initialization, update rates/gates, optimizer-state round trip, and explicit GPU rejection. Targeted `rustfmt --check --edition 2024` and `git diff --check` passed. No training or Oxide tests were run.
- Source contract: hand-derived from `Jev_laya`; no compiled-C++ golden/parity. Stage-1 CPU only: no GPU path or full multi-template/drop-mask gradients. CPU Muon was delivered separately in v0.1.005. Do not claim full DSConv parity or v0.1 completion.

## 20. Graph-convolution Stage-1 CPU reference

- v0.1.003 (`f9a90bd5940f0dfce40e4e2ded0d7c1838f50e11`): integrated `[graph_conv]` / `[graph_convolutional]` with strict source-derived config/shape validation, grouped NCHW projections, uniform valid-neighbor means or learned masked-softmax edges, optional center projection, source-order weights, seeded initialization, BN/activation, exact backward of the Rust forward, SGD, and Adam with persisted moments. GPU execution and updates reject explicitly.
- Verification: `MEM=3G ./tools/test_host.sh graph_conv` passed 14 tests; `weights` passed 42; `runtime` passed 192. Coverage includes non-square NCHW hand math, grouped projection and learned-edge finite differences across input and all trainable blocks, uniform-mode border finite differences and zero upstream, finite-difference bias, preceding-convolution gradient accumulation, BN/rolling statistics, every optional weight-block combination with exact file size/trailing-byte checks, optimizer updates/state round trip, and GPU rejection. Targeted Rustfmt and `git diff --check` passed. No Oxide tests, training, or `cfg_corpus` run.
- Source contract: hand-derived from `Jev_laya`; `LAYER_GRAPH_CONV` is disabled in the source branch and there is no compiled-C++ golden, so no compiled-C++ parity is claimed. Adam is a Rust extension; CBN uses ordinary CPU BN semantics. Stage-1 CPU only; native GPU paths and GPU Muon belong to Stage 2. CPU Muon was delivered separately in v0.1.005.

## 21. WMHF Stage-1 CPU reference

- v0.1.004 (`49ce73ccfc03d76c9a79e53389d283198d5d788e`): integrated the strict `[wmhf]`, `[wavelet_mamba_hf]`, and `[wavelet_mamba_high_frequency]` config paths, seeded initialization, seven source-order convolution blocks plus four scan vectors, Haar DWT/IWT and four-direction EMA scan CPU forward, exact backward, batch-stat BN with rolling-stat updates, SGD, Adam with persistent moments, and exact `.weights` serialization. GPU execution and updates reject explicitly; no GPU fallback or kernels were added.
- Verification: `MEM=3G ./tools/test_host.sh wmhf_` passed 17 tests; `weights` passed 43; `runtime` passed 198. Coverage includes hand-computed full forward, zero-upstream and finite-difference input/parameter gradients across all blocks, preceding-convolution accumulation, odd-edge Haar adjoints, BN/rolling statistics, SGD and GPU rejection, Adam-state persistence, initialization, strict config, and exact source-order file length. Targeted Rustfmt checks on modified core files and `git diff --check` passed. No training, Oxide tests, or `cfg_corpus` run.
- Source contract: hand-derived from `Jev_laya`; `LAYER_WMHF` is disabled and guarded tests are not a numerical oracle, so no compiled-C++ golden/parity is claimed. WMHF is Stage-1 CPU-only. `recursive_block` remains open; do not claim Phase 3B/Stage 1 complete.

## 22. CPU Muon hybrid optimizer

- v0.1.005 (`2674f5741405e33371448bb9a664d482ffa02e8d`): added opt-in `[net] muon=1` hybrid CPU updates. Eligible dense matrix blocks use arXiv:2502.16982's additive momentum/Nesterov lookahead, five Newton–Schulz iterations with `(3.4445,-4.7750,2.0315)`, scale `0.2*sqrt(max(rows,cols))`, and decoupled decay. Explicit per-layer matrix mappings cover convolution/recurrent/attention families; other trainable parameters use AdamW fallback. Conflicting/unsupported config rejects.
- CPU trainer dispatch covers general and OBB CLI training, direct OBB batch helpers, and the benchmark update step. DNOPT6 persists shape-checked Muon momentum and retains DNOPT1–5 reads; loading old state in Muon mode starts fresh Muon/AdamW state. GPU execution/updates and GPU optimizer-state save/load reject explicitly.
- Verification: `MEM=3G ./tools/test_host.sh muon_` passed 13 tests, `weights` passed 43, and `runtime` passed 204, including `train_only_bn` gating. Core-file `rustfmt --check --edition 2024` and `git diff --check` passed. No training or Oxide tests were run.
- Evidence boundary: update is paper-derived plus hand-computed/runtime tests. There is no compiled-C++ Muon implementation or golden, so no compiled-C++ parity is claimed. Stage 1 remains open for `recursive_block`; GPU Muon belongs to Stage 2.

## 23. Clifford Stage-1 CPU reference

- v0.1.006 (`1cc80d92b06a7b4fcbd90bea0f6a38931bb0d0ed`): integrated `[clifford]` as an ordinary source-derived NCHW layer, not a nested config container. CPU support covers local/global/hybrid branches, avgpool/Fourier global context, inner/wedge/full interaction schedules, abs/diff context, higher vector-bivector mode, and cached DropPath. The guarded C++ `LAYER_CLIFFORD` macro is disabled, so this is hand-derived from `Jev_laya`, not compiled-C++ parity.
- Added seeded source-scale initialization; exact source-order `.weights` load/save; training forward/backward with internal depthwise BN and rolling-stat updates; SGD, AdamW, and Muon matrix mappings (with AdamW fallback for non-matrix parameters). DNOPT host-Adam/Muon state round trips validate nested state; GPU full/compact execution and updates reject explicitly in Stage 1.
- Verification: `MEM=3G ./tools/test_host.sh clifford_` passed 13 tests, `weights` passed 44, and `runtime` passed 208. The Clifford tests cover finite differences for input and trainable parameters across avgpool+DropPath, Fourier, inner/abs/local, and wedge/global modes; exact weight bytes/round-trip; seeded initialization; optimizer updates/state; BN stats; and GPU rejection. Targeted Rustfmt and `git diff --check` passed. No training, Oxide tests, or `cfg_corpus` run.
- Evidence boundary: no compiled-C++ golden exists and no compiled-C++ parity is claimed. Clifford is Stage-1 CPU-only; `recursive_block` remains open and native GPU support/GPU Muon belong to Stage 2.

## 24. Recursive-block Stage-1 CPU reference

- v0.1.007 (`94513d426c190ab8b7fb012ee3bf0b5f68ff5e55`): integrated `[recursive_block]` / `[recurrent_block]` with nested body-section consumption, strict option/shape validation, deterministic initialization, per-pass body tapes/dropout masks, and nested BN rolling-stat updates. CPU forward covers static recurrence, controller-gated `ouroboros=1`, and lattice projection; backward is exact for the Rust forward across the full unrolled recurrence, controller, and lattice. SGD, AdamW, Muon traversal/state validation, body-only C++ `.weights` order, and explicit GPU/compact/update rejection are included.
- C++ `.weights` omits controller weights/biases (and LoRA values), so controller learning is not restored by Darknet checkpoints. Stage-1 rejects `ouroboros=2` and every explicit `lora_*` option. Rust full BPTT and lattice derivatives intentionally differ from C++ truncated k=1 backward and stop-gradient lattice behavior.
- Verification: `MEM=3G ./tools/test_host.sh recursive_block` passed 11 core tests, `weights` passed 45 core + 2 CLI tests, `runtime` passed 219 core tests, and `MEM=3G cargo check -p darknet-core -j 4` passed. Tests include all-trainable finite differences for static/controller/lattice branches, dropout mask reproducibility/independence, nested BN running stats, exact body-only weight bytes, optimizer state validation, Muon matrix ordering, and GPU rejection. Targeted rustfmt and `git diff --check` passed. No training, Oxide tests, or feature-specific `cfg_corpus` run; the full host baseline/cfg-corpus status remains non-green/unverified under the memory cap (see `STATUS.md`).
- Evidence boundary: hand-derived from `Jev_laya`; there is no compiled-C++ golden/parity. CPU-only Stage-1 support is complete for this layer; native GPU and GPU Muon remain Stage 2.

## 25. Hyperbolic-foveation Stage-1 CPU reference

- v0.1.008 (`f7b044bc84c6b19ed5b99cc97633fd859724db74`): integrated strict `[hyperbolic_foveation]` / `[hf]` config and shape handling, source-derived NCHW warp and learned predictor, deterministic init, source-order weights, exact finite-iteration inverse derivatives, alternate predictor/network-input fan-in, label-driven tiny-box/strength regularizers, SGD/AdamW/Muon updates and state validation, and axis-aligned/OBB/CARPN detection plus mAP reprojection.
- Full and compact GPU execution, f16 inference, GPU backward and optimizer updates explicitly reject this Stage-1 CPU-only layer. No CUDA/Oxide fallback or kernel was added.
- Verification: `MEM=3G ./tools/test_host.sh hyperbolic_foveation` passed 30 core tests; `runtime` passed 231; `weights` passed 47 core + 2 CLI; `detect` passed 36 core + 10 CLI. Targeted Rustfmt and `git diff --check` passed. No training, Oxide tests, or feature-specific `cfg_corpus` run.
- Evidence boundary: source-derived from `Jev_laya`; its layer macro is disabled and no compiled-C++ golden/parity is available. `foveation_film` Stage-1 CPU support is tracked in §26; native GPU support and the other Phase 4 layers remain open.

## 26. Foveation FiLM Stage-1 CPU reference

- v0.1.009 (`51e381b8a94ae4e23ecc8054f6045a128f87fc8f`): integrated `[foveation_film]` / `[hf_film]` with strict HF-source resolution, source-order six-block parameters, deterministic initialization, NCHW CPU forward, exact backward, additive HF parameter gradients, learned-predictor/source-feature fan-in, SGD/AdamW/Muon updates, and optimizer state validation/persistence. Full/compact/f16 GPU inference, backward, and optimizer updates reject explicitly; no CUDA/Oxide or host fallback was added.
- Verification: `MEM=3G ./tools/test_host.sh foveation_film` passed 12 core tests; `runtime` passed 237; `weights` passed 49 core + 2 CLI. Coverage includes NCHW hand math and finite differences, predictor/source finite differences, detached and additive HF gradients, multiple FiLM sources, train-buffer accumulation, initialization identity/seed reproducibility, exact source-order bytes/file size, optimizer state persistence/gating, and GPU rejection. Rustfmt check passed for new math/runtime-test files; `git diff --check` passed. No training, Oxide tests, or feature-specific `cfg_corpus` run.
- Evidence boundary: hand-derived from `Jev_laya`; `LAYER_FOVEATION_FILM` is disabled, so no compiled-C++ golden/parity is available. C++ SGD has a different sign convention; Rust uses the established downhill-update convention and makes no update-parity claim. Stage-1 CPU only; native GPU support and GPU Muon remain Stage 2. Depth-Anything and DPT are covered in §§27 and 30; YOLO-3D and DETR remain open, so the overall roadmap is not complete.

## 27. Depth-Anything Stage-1 CPU reference

- v0.1.010 (`f0a3f2391655d120044bd13f6ded6ca6c655971e`): integrated the frozen `[depth_anything]` / `[depth_anything_v2]` / `[dinov2]` / `[dino]` DINOv2 ViT-S/14 encoder with strict config/shape validation, source-derived type-major weights, cfg-relative layer-only `weights=` loading, and separate sequential checkpoint behavior when no external blob is configured. CPU forward covers channel-major NCHW patchification, bicubic positional interpolation, LayerNorm, exact-erf GELU, full multi-head attention, LayerScale residuals, tap-major NCHW output, final norm, and clamp. Rust-mode backward differentiates the full forward to input; `GradMode::Cpp` preserves the source no-op. Frozen weights are inert under SGD/AdamW/Muon; full/compact/f16 GPU execution, GPU backward, and eager/deferred GPU updates reject explicitly. No GPU fallback, CUDA/Oxide path, or dependency was added.
- Verification: `MEM=3G ./tools/test_host.sh depth_anything` passed 16 tests; `runtime` passed 244; `weights` passed 52. Coverage includes source-order array lengths, a hand-computed tiny transformer, bicubic reference, asymmetric patch and B>1 non-square/tap-major axes, input finite differences and clamp derivatives, alias/default/invalid-config tests, external and sequential weight round trips, frozen optimizer behavior, upstream convolution fan-in, Cpp-vs-analytical backward, and GPU rejection. Targeted edition-2024 Rustfmt and `git diff --check` passed. No training, Oxide, full `cfg_corpus`, or local HuggingFace dump comparison was run.
- Evidence boundary: hand-derived from `Jev_laya`; no compiled-C++ golden/parity is claimed. Depth-Anything is Stage-1 CPU only; `[dpt_head]` is implemented in §30. `[yolo_3d]`, native GPU support, GPU Muon, and the broader v0.1–v0.3 roadmap remain open. The known full host/cfg-corpus baseline remains non-green/unverified under the memory cap.

## 28. Stage-1 axis/layout audit (bounded A1 list)

- v0.1.011 (`a34aef8f`): added hand-computed, named tests for NCHW batch/channel/spatial ordering in ViT patchification, per-image MoE routing/blending, EML branches, DCNv4 offset/group axes, every `laya_*` tensor family, channel slicing, deconvolution, and weighted shortcut. The source-level contracts and test names are tabulated in `STATUS.md`'s Axis audit section. No forward behavior changed.
- Verification: `MEM=3G ./tools/test_host.sh axis_layout` passed 13 core tests; `MEM=3G ./tools/test_host.sh laya_` passed 66. Targeted edition-2024 Rustfmt checks and `git diff --check` passed.
- Evidence boundary: hand-derived from `Jev_laya` after `git grep`/`git show`; no compiled-C++ golden/parity is claimed. This section records the initial subset; the complete A1 audit of already-ported families is in §29. Per-backlog axis requirements for remaining missing layers, native GPU work, GPU Muon, and v0.1–v0.3 remain open.

## 29. Stage-1 axis/layout audit of already-ported layers

- v0.1.012 (`60db7b444b8db722aa145954919a798169bb45c8`): completed A1 from `AGENTS.md` for all already-ported families: ViT, Swin, Tucker attention, MambaVision, Kimi KDA, MoE, EML, DCNv4, LAYA, channel slice, deconvolution, and shortcut. Added named hand-computed tests for Swin's batch/window/token/channel partition and cyclic-shift axes, Tucker's batch/window/token/channel isolation, MambaVision's NCHW/token/state axes, and Kimi KDA's NCHW raster scan and per-batch reset. No forward behavior changed.
- Verification: `MEM=3G ./tools/test_host.sh axis_layout` passed 18 core tests; focused suites passed `tucker_attention` (15), `mambavision` (17), and `kimi_kda` (11). Edition-2024 Rustfmt checks for the four touched modules and `git diff --check` passed.
- Evidence boundary: hand-derived from `Jev_laya` using `git grep`/`git show`; no compiled-C++ golden/parity is claimed. This closes only the A1 audit for already-ported layers. Every remaining missing-layer port still has its own axis/layout DoD; native GPU work, GPU Muon, and v0.1–v0.3 remain open.

## 30. DPT-head Stage-1 CPU reference

- v0.1.013 (`a995fe95`): added CPU `[dpt]` / `[dpt_head]` / `[depth_anything_head]` / `[neck]` parsing and inference, strict config/shape validation, the source-order 66-slot parameter arena, sequential `.weights` handling, zero initialization, NCHW tap-major forward, and analytical backward for the full Rust forward. `GradMode::Cpp` preserves the source no-op. SGD, AdamW and Muon updates/state are integrated; Muon keeps per-slot matrix views, AdamW bias fallback, and matrix decay. GPU execution/backward/updates and unsupported stream settings reject explicitly.
- Verification: `MEM=3G ./tools/test_host.sh dpt_head` passed 18 tests; `runtime` passed 260; `weights` passed 55 core + 2 CLI; and `axis_layout` passed 19. Tests pin all 66 slot offsets and exact default file length (2,728,513 floats; 10,914,072 bytes including the 20-byte header), batched non-square NCHW/tap axes, independent hand-computed forward, finite differences for input and all nonempty parameter slots, optimizer updates/state round trips, update gates, `GradMode::Cpp`, Depth-Anything chaining, and GPU rejection. Targeted edition-2024 Rustfmt checks and `git diff --check` passed. No training, Oxide tests, or full `cfg_corpus` run.
- Evidence boundary: hand-derived from `Jev_laya`; no compiled-C++ golden/parity is claimed. This is Stage-1 CPU support only; native GPU and GPU Muon remain Stage 2. `[yolo_3d]` is a separate scope-confirmed backlog item and the v0.1–v0.3 roadmap remains open.

## 31. C++ enum boundary config regression

- v0.1.014 (`fe3a554c`): added `model::tests::unused_cpp_layer_enum_names_are_not_rust_noop_layers`. It pins `[network]` as top-level metadata and ensures `[blank]`, `[empty]`, and `[silence]` are rejected instead of becoming silent identity layers. This is a parser-regression test; no runtime behavior changed.
- Verification: `MEM=3G ./tools/test_host.sh unused_cpp_layer_enum_names_are_not_rust_noop_layers` passed 1 test; targeted edition-2024 Rustfmt and `git diff --check` passed.
- Evidence boundary: source-derived from `Jev_laya:src-lib/darknet_enums.cpp` and `darknet_cfg.cpp`; no compiled-C++ golden/parity. The larger v0.1–v0.3 roadmap remains open.

## 32. Axis-coverage gate and shared NCHW fixtures

- v0.1.015 (`e848f4ef`): added `AxisPin` and `AXIS_PINS` in `model.rs` with one decision per `LayerKind` variant (64 variants plus `Unsupported`), and `LayerKind::axis_pin`, whose match is exhaustive so a new variant fails the build until classified. Added `axis_fixture.rs` with `PROBE`/`probe_shape`, `axis_value` (`n*1000+c*100+h*10+w`), `decode_axis_value`, `nchw_index`, and `axis_tensor`, replacing the per-test sentinel loops.
- Verification: `MEM=3G ./tools/test_host.sh every_layer_kind_has_an_axis_pin_decision` passed 1 test; `model::` passed 75; `axis_fixture` passed 3; `axis_layout` passed 19 (unchanged); targeted edition-2024 Rustfmt and `git diff --check` passed.
- Coverage baseline recorded in `tasks/rust/axis-coverage.md` (local-only): 18 variants pinned by 19 tests, 41 C++-backed variants open, 5 Rust-only extensions, 3 variants not ported. No layer math changed, so no behavior claim is made beyond the new checks.
- Evidence boundary: enum audited against `Jev_laya:src-lib/darknet_enums.hpp`; `Logistic` is referenced on `raptor`/`raptor-dev` and `Regression` on `fog`, neither on `Jev_laya`. No compiled-C++ golden/parity.

## 33. Convolutional axis pins and the dilation contract

- v0.1.016 (`051021fe`): added `model::tests::convolutional_axis_layout_pins_dilated_extent_and_groups` (C++ extent for dilated/strided non-square input, `batch=2`, and the `[out][in/groups]` weight cardinality for `filters=6, groups=2, C=4`) and `runtime::tests::convolutional_axis_layout_keeps_nchw_and_group_partitions` (channel-swapping 1x1 weights, then `groups=2` partitioning, both over `batch=2`).
- Verification: `MEM=3G ./tools/test_host.sh convolutional_axis_layout` passed 5; `axis_layout` passed 21 (was 19); `runtime::` passed 261. Full CPU baseline `cargo test -p darknet-core --lib -j 2` passed 897 and `-p darknet-cli` 28; GPU baseline `MEM=3G ./tools/test_oxide.sh --test oxide_parity` passed 11 and `oxide_runtime_chain` 24. Targeted edition-2024 Rustfmt and `git diff --check` passed.
- Finding F1 retracted: `conv_dim` must keep ignoring dilation because `Jev_laya:src-lib/convolutional_layer.cpp:317` does; applying `dilation*(size-1)` shrank the extent and broke `cfg/resnet152_trident.cfg` at route #244 (`16x16` vs `38x38`). `conv_dim` now documents that, and the pin fails if the term returns. The cuDNN/oxide `pad*dilation` difference is recorded as a source-level inconsistency, not a Rust defect.
- Evidence boundary: hand-derived from `Jev_laya`; no compiled-C++ golden and no parity claim.

## 34. GPU convolution extent parity

- v0.1.017 (`263fa448`): fixed `Geom::new` in the cuda-oxide conv driver. It computed the textbook dilated span `(in + 2*pad - (dil*(k-1)+1))/stride + 1`, while `darknet-core`'s `conv_dim` and C++ `convolutional_out_width` compute `(in + 2*pad - k)/stride + 1`: dilation changes the sampling offsets, not the extent. The device map was smaller than the caller requested, so `expect_shape` rejected every dilated convolution. f32 and f16 now use the CPU extent; extent validation uses `k`.
- Verification: `oxide_conv_parity` passed 13 (including `dilation2_pad2`, `fd_dil2`), `oxide_cfg_shapes` 13 (6 dilated geometries, Direct and Gemm, 12 runs, 0 failures), `oxide_gemm_mma` 5, `oxide_half` 13. The `oxide_cfg_shapes` contract was inverted: raw `pad` must reproduce the cfg dimensions and parity, while a `pad*dilation` call with cfg dimensions must be rejected.
- Evidence boundary: hand-derived from `Jev_laya` and `darknet-core`; no compiled-C++ golden/parity. YOLO heads are pass-through on device (the loss/decode runs on the host), so this was the only conv-side parity gap.

## 35. Annotation GUI (`darknet-gui`)

- v0.1.018 (`0ca7613e`): added the opt-in `darknet-gui` crate (egui/eframe 0.32.3, `glow`), the first slice of `gui_plan.md`. `annotation.rs` holds normalized centre/size marks with a 10-decimal YOLO text round trip and click hit testing; `project.rs` scans a flat image folder, reads class names from the first `*.names` or `classes.txt`, and loads/saves sidecars; `main.rs` is the window with folder open, class and image lists, fit-to-window canvas, drag-to-create, click-to-select, rubber band, and per-image undo.
- Semantics: malformed label lines are errors, so a partial read cannot overwrite a file; a degenerate drag creates nothing; an empty annotation deletes the sidecar instead of writing a blank one.
- Verification: `cargo build -p darknet-gui --offline` and `cargo test -p darknet-gui --offline` pass with 7 tests.
- Evidence boundary: builds offline against the locally cached eframe. Not implemented: DarkMark `.json` sidecars, class remap, filters/sort, neighbour copy/interpolation/homography, alternate views, launcher, and the `detect-server` prediction overlay.

## 36. Annotation GUI plan implementation

- v0.1.019 (`fcd23bbd`): split the annotator into `darknet-gui-project` (pure logic), `darknet-gui-engine` (child-process inference) and `darknet-gui` (eframe shell), and implemented the plan's milestones:
  - `annotation.rs`: DarkMark detector JSON (`G3`-`G7`) parsed through `serde_json::Value` so unknown sidecar and mark fields survive; string and integer `annotation_mode`; exactly four points (`D4`); `G4`/`G5` save rules; `.txt` with ten decimals.
  - `project.rs`: recursive `G2` scan with chart/mask/cache exclusions, `<name>.names`/`.data` preference (`D5`), detector-mode validation, mark index counting `completely_empty` as one.
  - `filter.rs`, `nav.rs`: tri-state/regex/class/age filters, stable sort, clamped navigation, unmarked walking that stops on the boundary image.
  - `undo.rs`, `edit.rs`: capped undo/redo and the `G9`-`G16` edit model (smallest-area hit test, corner re-drag with `D8` restore, clamped creation, additive digit classes with refusal, Tab cycling including predictions, accept rules, threshold clamp, nudge/scale/clear/empty/copy).
  - `actions.rs`, `create.rs`, `classes.rs`: interpolation, clipboard JSON, edge snapping, homography; `G1` project creation; dry-run class remap with backup and atomic rename.
  - `darknet-gui-engine`: `DetectWorker` JSON-lines protocol with a 30 s timeout, child-exit and malformed-reply errors.
  - `darknet-cli detect-server`: one request per stdin line, normalized boxes, error objects that keep the server alive, CPU and feature-gated oxide backends.
  - `darknet-gui`: launcher with persisted recents, new-project dialog, image cache, zoom/pan canvas, six view modes, filters, class editor with dry run, prediction overlay.
- Verification: `cargo test -p darknet-gui-project` 36, `-p darknet-gui-engine` 4, `-p darknet-cli` 28; offline build of the three GUI crates is warning-free; `cargo metadata --locked` reports 33 members.
- v0.1.020 (`2640b902`): the overlay's model loader resolves `DARKNET_RS_BIN` before the current executable, so `cargo run -p darknet-gui` can find `darknet-rs`.
- Evidence boundary: DarkMark compatibility is derived from the `G`/`D` tables, not verified against a running DarkMark (`T1.5` still open). Not wired: add/delete classes from the dialog, delete-image/undo-delete, heatmaps, jump, screenshots, train runner.

## 37. DarkMark hotkey parity and model discovery

- v0.1.021 (`8c536b6d`): the key map is transcribed from `DarkMark/src-darkmark/DMContent.cpp::keyPressed()` (`:1013-2050`) rather than the `G16` summary. Corrections: `r` random sort and `R` alphabetical; `,`/`.` apply `scale_all_marks(+-0.01)`, a normalized delta, not a box-relative factor; `o` is mass-delete mode; `P` homography; `` ` `` merge; `l` labels on/off/auto; `b`/`B` bold/shade; `d` snap-all; `D` snapping toggle; `h`/`H` heatmaps; `j` jump; `s`/`S` screenshots and `F12` window screenshot; `t`, `[`/`]`, `(`/`)` tiling; `y`/`Y` copy neighbour; `Shift+U` undo-delete; `Shift+Delete` delete image; `Escape` cancel. `Tab` skips invisible marks and records the recent class/size; `A` accepts only the selected prediction.
- `Project` now discovers the project's model files (`G1`): `<name>.cfg` else the first `*.cfg`, and `<name>_best.weights` / `<name>_final.weights` / `<name>.weights` else the first `*.weights`. The GUI prefills the overlay paths from them.
- Weight parsing remains in `darknet-rs`: `load_weights_file` for inference and `weights-inspect` for a header report. The GUI process never parses `.weights`; it starts `detect-server`, which loads them and reports mismatches as errors.
- Verification: `darknet-gui-project` 38 tests, `darknet-gui-engine` 4, offline builds of the GUI and CLI.
- Evidence boundary: mapped actions for features not yet built (mass-delete, homography, merge, heatmaps, screenshots, tiling, jump, delete-image) report "mapped but not implemented" instead of silently doing nothing. Parity is with the DarkMark source, not a running DarkMark.

## 38. YOLO-3D configuration preparation (not a layer port)

- v0.1.024 (`4506e46e`): added `model/yolo_3d_config.rs`, re-exported through `model.rs`, with `Yolo3dSpec`, `Yolo3dGeometrySpec`, `Yolo3dVertexLoss`, `from_section`, checked anchor-entry arithmetic and identity-shape validation. Defaults/keys were checked directly against `depth_3d` at `9c9e0b4b5e0836330a70a0667fd9e161bd1aac1a`, `src-lib/darknet_cfg.cpp:2556-2697` and `yolo_3d.cpp:1275-1394`.
- Verification: test-first compile failed with E0432/E0425 for the missing types; final `MEM=3G ./tools/test_host.sh yolo_3d_cfg` passed **12 core tests**, and `MEM=3G ./tools/test_host.sh model::` passed **88 core tests**. Both used user systemd `MemoryHigh=3G` scopes and `-j 2`; CLI/integration targets compiled but matched zero tests. Targeted edition-2024 Rustfmt checks and `git diff --check` passed. No training, Oxide, corpus or full-host run.
- Scope audit: `defaults_match_source` and `retains_projected_and_loader_overrides` pin defaults/overrides; `aliases_follow_existing_section_normalization` covers all four aliases and the existing Rust case-folding convention; `projected_channels_and_identity_shape` pins `A*(5+C+3*K)` with B=2 and a 3x5 grid, but is **not** an A2 numeric axis/decode test. Unknown-key, invalid-value, nonfinite, array, overflow and staged-feature tests cover explicit failure boundaries. `truth_intrinsics_and_projected_options_are_not_staged_out` proves focal/winding/OKS config is accepted; it does not prove truth encoding or losses.
- Deliberate strictness: finite/range checks, unique masks, exact `2*num` anchors and three edge weights, and bounded source-int storage reject inputs C++ may ignore or default. Zero loss normalizers still disable terms. `map_file`, counters and loader/NMS options are retained metadata only; class-map loading and class-multiplier conversion await network integration.
- **Still open:** `LayerKind`/network parse and shape dispatch, mixed-head/depth-reference validation, axis-pin registration, labels/truth encoding, forward/backward, loss/decode/evaluator, checkpoint/optimizer proof and CLI/GPU guards. Network construction still rejects this layer (pinned by `preparation_does_not_enable_incomplete_runtime`). This is only the helper portion of plan Step 1; that step and the layer DoD remain incomplete. Depth/metric nondefault config is explicitly rejected pending the second CPU increment.
- Evidence boundary: hand-derived from source, no compiled-C++ golden. GPU status: none. DSConv multi-template, remaining CPU/axis work, all Stage-2 native GPU/GPU Muon and Stage-3 sparse MoE requirements remain open. The known full-host/corpus baseline remains non-green/unverified.

## 39. Cuboid labels and deterministic truth helpers

- v0.1.025 (`8f6178ed`): added `cuboid_labels.rs` and `cuboid_labels/tests.rs`, with public `lib.rs` re-exports. The pure and mixed parsers/loaders preserve source row precedence, clamped coordinates, unknown/absent visibility and source track IDs. Truth helpers encode the 11-float prefix, crop/flip intrinsics, K triplets, padding and mixed ownership, with explicit original/network image sizes. This advances plan Step 2 independently; Step 1 network integration remains open and the existing unsupported-layer regression remains green.
- Source checked directly: `depth_3d` at `9c9e0b4b`, `data.cpp:500-1250,1335-1505` and `utils.cpp:1292-1303` (64-bit Linux DJB2 track hash). Pure files use a zero-based object ordinal; mixed files use one-based ordinals. `MixedLabel::Box2d` includes `track_id` alongside the existing `YoloLabel`, rather than discarding source metadata as the plan sketch's bare payload would. `Yolo3dImageMeta` additionally takes network W/H for the source one-pixel hull filter.
- Verification: initial test-first `MEM=3G ./tools/test_host.sh yolo_3d_label` failed to compile with missing-type/function E0422/E0425/E0433. Final `MEM=3G ./tools/test_host.sh yolo_3d` passed **31 core tests** (19 new label/truth + 12 config); `MEM=3G ./tools/test_host.sh label` passed **43 core tests**. Both used `MemoryHigh=3G` systemd scopes, `-j 2`; CLI/integration targets compiled but matched zero tests. Targeted edition-2024 Rustfmt checks and `git diff --check` passed. No training, Oxide, corpus or full-host run.

Scoped prompt-to-artifact audit (test names below are in `cuboid_labels/tests.rs` and start with `yolo_3d_`):

| Requirement | Concrete evidence |
|---|---|
| Pure/mixed grammar, class conversion and 2D ambiguity | `label_row_parses_source_layout`, `mixed_label_ambiguous_row_precedence_matches_source` |
| Hull visibility, coordinate clamps, finite/range errors | `label_unknown_visibility_is_not_absent`, `label_clamps_coordinates_but_rejects_nonfinite_and_malformed_rows` |
| File I/O/errors and source path/ordinal tracks | `label_file_loaders_preserve_grammar_and_report_io_errors`, `label_source_track_hash_and_pure_mixed_counter_offsets` |
| Prefix, vertex offsets, padding, independent batches | `truth_prefix_offsets_padding_and_independent_batches`, `truth_each_image_uses_its_own_crop_intrinsics_and_flip` |
| Focal/crop math and flip without corner permutation | `truth_intrinsics_hand_computed_and_crop`, `truth_flip_mirrors_px_and_negates_geo_without_reordering_vertices` |
| Clip all corners, transform hull separately | `truth_clipped_absent_corner_zeroes_intrinsics_but_keeps_points`, `truth_cropped_hull_does_not_rebuild_from_clamped_vertices` |
| Mosaic disables projected fields and mixed flags | `truth_mosaic_drops_fields_after_class`, `truth_mixed_mosaic_also_drops_ownership_like_source` |
| Mixed ownership and explicit pure-mode 2D rejection | `truth_mixed_ownership_routes_rows_to_heads` (encoding only, not loss routing) |
| Deterministic cap-before-filter, missing focal, 2D sentinel | `truth_deterministic_cap_before_tiny_filter_and_missing_focal`, `truth_two_dimensional_zero_center_is_source_sentinel` |
| Public-input/storage overflow and bounds errors | `truth_invalid_public_inputs_are_errors_not_panics`, `truth_inconsistent_public_storage_is_an_error` |

- Boundaries: deterministic caller order replaces source random shuffle; cap is applied before tiny-hull filtering without refilling. Nonfinite/malformed tokens, invalid public values and non-UTF-8 file identities error explicitly. Mosaic callers provide already positioned geometry; no tile sampler, clipping-policy implementation or image augmentation loader is added. Crucially, C++ `blend_truth_mosaic` copies only entries 0–4 and drops the trailing mixed flags too, so mixed mosaic records carry flag zero and are not assigned to either head; this is not a claim of useful mixed-mosaic training.
- **Not complete:** no network enum/dispatch, mixed-head build validation, output-axis/decode pin, runtime/backward/loss integration, evaluator, optimizer/checkpoint proof, CLI integration or depth/metric consumption. This suite verifies label/truth math, not end-to-end training or any compiled-C++ comparison. Hand-derived from source, no compiled-C++ golden; GPU status none. DSConv variants, remaining CPU/axis requirements, Stage-2 native GPU/GPU Muon and Stage-3 sparse MoE remain open. No overall parity or green full-host claim.

## 40. YOLO-3D raw CPU runtime and fail-closed consumers

- v0.1.026 (`755e5145`): added `LayerKind::Yolo3d`, alias dispatch, identity-shape re-inference, shared classical-YOLO/cuboid class validation and shared cuboid K validation in `model.rs`. `runtime.rs` captures raw `Yolo3dHeadOutput` separately from 2D heads, passes supplied seeds upstream with the exact identity derivative, and keeps head parameters/train buffers/Adam/Muon state empty. `lib.rs` exports the typed output. CPU compact inference uses the full CPU path; all GPU paths remain rejected by the existing Stage-1 guard.
- `train.rs`, `detect.rs`, `evaluate.rs` and CLI consumers now reject cuboid networks before legacy loss/decode/evaluation can silently discard their heads, including empty caller-supplied head lists. CLI guards cover `detector train`, `detector map`, `detect-image`, and `detect-server`; `detector test` is still an unknown subcommand globally, not a new supported cuboid command. ONNX still rejects through its existing unsupported arm; no ONNX/CUDA source change.
- Source read directly at pinned `depth_3d=9c9e0b4b5e0836330a70a0667fd9e161bd1aac1a`: `darknet_cfg.cpp:1603-1687` for shared class/K checks and `yolo_3d.cpp:1501-1542,1667-1672` for activation storage/backward. Rust deliberately keeps raw logits per plan D3, unlike C++ in-place activation; activations must be applied inside future loss/decode math. This is not C++ forward-output parity.
- Test-first: network dispatch tests failed with missing `LayerKind::Yolo3d` before implementation. The required `MEM=3G ./tools/test_host.sh does_not_exist` compile inventory then passed with zero tests: wildcard arms meant the enum alone did not force further integration. Runtime tests later caught unwanted empty Muon state on the weightless head; CPU Adam/Muon now explicitly skip this kind. Final tests prove both head states remain None after updates and DNOPT round trips.

Scoped prompt-to-artifact audit:

| Contract | Actual evidence |
|---|---|
| All four aliases, weightless identity shape | `model/yolo_3d_config.rs::yolo_3d_cfg_network_aliases_build_weightless_identity_heads`; replaces the old preparatory unsupported-network test |
| Shared classes and K, resize/batch/route geometry, honest open axis decision | `yolo_3d_cfg_mixed_network_requires_shared_classes_and_keypoints`, `yolo_3d_cfg_shape_rebuild_preserves_head_and_route_geometry`; exhaustive axis gate now has 66 rows (65 kinds + Unsupported), with 19 pinned/41 open/6 Rust-only-or-sentinel; `axis-coverage.md` updated |
| Typed raw forward collection, full/training/compact CPU, no head params | `runtime/tests/yolo_3d_runtime_tests.rs::yolo_3d_forward_is_raw_pass_through_and_records_separate_head` (B=2, 3x5 grid, A=2, C=2, K=8) |
| Exact raw identity backward and preceding conv derivative | `yolo_3d_backward_is_identity_and_reaches_upstream_convolution`: central FD on three conv weights of a raw-output dot product, not a cuboid loss |
| Mixed head vectors remain disjoint | `yolo_3d_mixed_forward_keeps_the_two_head_collections_disjoint`; this is not mixed loss routing |
| No checkpoint block or optimizer tensors in head | `yolo_3d_weights_round_trip_exact_length_with_upstream_conv`: 380 bytes (20 + 4*90), trailing_bytes=0; `yolo_3d_seeded_sgd_adam_and_muon_update_only_upstream_weights`: upstream updates, None head buffers/states and DNOPT round trips, using supplied seeds only |
| Existing 2D/OBB/modern decoders, losses and validation datasets reject graph | `yolo_3d_existing_2d_and_obb_apis_reject_even_empty_head_lists` covers nine entry points without relying on the passed head list |
| CPU-only boundary | `yolo_3d_rejects_gpu_execution_backward_and_updates` covers full/compact GPU, immediate/deferred update and backward; f16 is checked via the shared preflight helper, not real device execution. Source inspection confirms the f16 entry calls that helper |
| Constructor and unsupported export | `yolo_3d_from_heads_constructor_has_no_cuboid_outputs`, `yolo_3d_onnx_export_still_rejects`; all ForwardResult constructors compile with the new field |
| CLI rejection before asset loading/training | `darknet-cli::tests::yolo_3d_cli_rejects_train_map_and_detect_server_before_loading_assets`; malformed data and missing weights prove those three paths reject before use. The detect-image guard is source-inspected, not exercised through a CLI process |

Verification commands ran serially under `MemoryHigh=3G` systemd scopes (scripts `-j 2`):

| `MEM=3G ./tools/test_host.sh <filter>` | Core passed | CLI passed |
|---|---|---|
| `yolo_3d` | 42 | 1 |
| `model::` | 90 | 0 |
| `runtime::` | 270 | 0 |
| `yolo` | 106 | 2 |
| `map` | 27 | 8 |
| `weights` | 57 | 2 |
| `axis_layout` | 21 | 0 |
| `dpt_head` | 18 | 0 |
| `onnx` | 12 | 1 |

- `systemd-run --user --scope -p MemoryHigh=3G cargo check -p darknet-cli --features oxide -j 4` passed. Targeted edition-2024 Rustfmt and `git diff --check` passed. No actual training runs, Oxide tests, corpus or unfiltered host suite; existing warnings remain. Temporary weights/sidecars used RAII cleanup; test logs/source snapshots were removed after recording evidence.
- **Remaining layer DoD:** projected activations/math, assignments, losses and exact loss derivatives/Cpp-mode deltas; standalone and mixed loss APIs consuming the label helpers; decode and its A2 anchor/entry/vertex pin; projected evaluator; depth/metric consumption and references; end-to-end loss-to-conv/optimizer tests. Parsed map/counter metadata still needs consumption by these APIs. Supplied-seed optimizer tests are not a substitute for full training-loss integration. The current axis decision stays Open; `axis_layout` remains 21, not a new cuboid pin. No compiled-C++ golden exists. Hand-derived from source, no compiled-C++ golden; GPU status none. Stage 1, all-native GPU/GPU Muon Stage 2, and sparse MoE Stage 3 remain incomplete; no overall parity or green full-host claim.

## 41. Native f32 LAYA marker gather

- v0.2.001 (`f56342ca`): operator explicitly requested GPU priority before CPU completion. Added native Oxide f32 forward/backward for `[laya_marker_gather]`, removed full/compact forward's per-operator downloads/CPU math/uploads, and removed this layer from the whole-network CPU-backward fallback gate. No weights, optimizer tensors or config changes. Other GPU layers/GPU Muon and all remaining CPU requirements are still open; parked YOLO-3D activation work is uncommitted and not part of this release.
- Source inspected directly: `Jev_laya@4af013d2c59d1db54247aeff21f43e1d8d295cda:src-lib/laya_marker_gather_layer.cpp:111-182`. Features are `[B,H,1,T]`; marker channel `2*j`, sequence column zero, chooses each option's position. Masks/extra channels/other sequence columns are ignored. Backward adds duplicate-option gradients in ascending option order to A only, even when A/B alias. Native threads own individual outputs/input gradients, so no atomics/races. Half-away rounding uses f64 before integer conversion; clamp-before-cast handles extreme values safely. NaN -> 0 and infinities -> endpoints preserve the Rust CPU reference's defensive extension, not a claim about C++ `lround` on invalid values.

Scoped requirement-to-artifact audit:

| Requirement | Actual artifacts and verification |
|---|---|
| Checked backend contract/default reference | `darknet-cuda/src/host_ops/marker_gather.rs::Layout`, exposed via `host_ops.rs`; two `F32TensorOpsBackend` methods in `darknet-cuda/src/lib.rs`. The generic default remains a host reference for non-Oxide/FakeBackend, consistent with existing ops; Oxide overrides both |
| Native, device-resident f32 operations | `darknet-cuda-oxide/src/kernels/layers_extra.rs::{marker_position,marker_gather,marker_gather_backward}` and overrides in `src/lib.rs`. Inspection confirms no operator download/upload or host data math; `mixed.rs::f32v` borrows f32 or widens on device. Ordinary full-runtime debug/output downloads remain |
| Per-sample layout, duplicate/clamped/nonfinite indices, ignored metadata, exact adjoint | `oxide_marker_gather.rs::marker_gather_native_forward_backward_and_raw_input_fd`: B=2, H=3, T=7, K=5, CB=12; tile lengths 32/128/256, ragged lengths, actual GPU forward FD at eight input elements with fixed metadata, exact independent reference output/gradient comparisons |
| Deterministic reduction and subtle rounding | `marker_gather_native_duplicate_reduction_has_fixed_option_order`: repeated `[1e20,1,-1e20,1]` sum gives 1; `marker_gather_native_rounds_adjacent_half_values_before_clamping`: adjacent f32 values below/at/above 0.5 and 1.5, extremes and NaN/infinities |
| Errors before unsafe device reads | `marker_gather_native_invalid_layouts_and_storage_error_before_launch`: bad rank/height/batch/T/channel count, zero options/dimensions, checked-product overflow, short feature/marker/gradient storage |
| Correct runtime routing and no forced CPU backward | `runtime.rs` full/compact/backward thin arms and fallback-gate removal; `runtime/tests/marker_gather_gpu_tests.rs::marker_gather_gpu_uses_device_backward_and_accumulates_only_feature_source` asserts gate false, A/B alias handling plus a direct source seed, full/compact agreement, and upstream update agreement on FakeBackend |
| Real-device runtime chain | `oxide_runtime_chain.rs::marker_gather_native_full_compact_backward_and_upstream_sgd_match_cpu`: B=2, two upstream convs, distinct A/B references, full/compact output parity, backward and both conv weight/bias gradient updates through SGD |

Test-first evidence: direct Oxide tests initially failed compilation on missing backend methods. The runtime test then failed specifically because marker gather forced CPU backward; it passed after native dispatch integration. Final serial, `MemoryHigh=3G`, script `-j 2` verification on NVIDIA GeForce RTX 4060 Laptop GPU:

| Command | Result |
|---|---|
| `MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather` | 4 passed |
| `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- marker_gather` | 1 passed, 24 filtered |
| `MEM=3G ./tools/test_oxide.sh --test oxide_layers_extra` | 24 passed |
| `MEM=3G ./tools/test_host.sh marker_gather` | 12 core passed; 0 CLI matched |
| `MEM=3G ./tools/test_host.sh runtime::` | 271 core passed; 0 CLI matched |

Targeted edition-2024 Rustfmt and diff checks passed. Tests ran with parked CPU activation code still in the working tree; it is not selected by these filters and adds no GPU dependency. Temporary GPU logs/source snapshot removed after recording results; no weights/sidecars were created by these new tests. No actual training run, benchmark/speedup measurement, unfiltered host/corpus or full Oxide-suite run. GPU status: **native f32 forward/backward/full/compact; f16 remains explicitly unsupported and was not device-tested**. New optimizer evidence is upstream SGD only; this weightless layer has no Muon/Adam parameter blocks, and GPU Muon remains open. Native GPU vs Rust CPU agreement only. Hand-derived from source, no compiled-C++ golden. No overall Stage-1/2/3 completion claim.

## 42. Native f16 marker-gather inference

- v0.2.002 (`32822894`): extended the source-backed marker-gather routing contract to native half-storage inference. `F16TensorOpsBackend::marker_gather_f16` has a numeric generic-backend reference default; the Oxide trait override delegates to the native method in `darknet-cuda-oxide/src/half.rs`, with checked shared layout and actual buffer lengths before allocation/launch. `kernels/layers_extra.rs::marker_gather_f16` copies feature u16 payloads exactly and widens only marker metadata on device, using the existing rounded/clamped position helper. No feature floating-point conversion or operator host round trip.
- `runtime.rs` permits marker gather in the f16 preflight and replaces its rejection arm with native source dispatch using the existing compact lifetime/use-count mechanism. Existing relative-source resolution and A/B alias handling are preserved. Only `forward_compact_f16` exists for this inference API: no new full-state f16 forward or f16 training/backward API is claimed. F32 training/backward remains from v0.2.001; no parameter, optimizer, checkpoint, config, AMP-global setting or dependency change.

Scoped requirement-to-artifact audit (`oxide_marker_gather_f16.rs`):

| Requirement | Actual test/artifact |
|---|---|
| Half bits copied exactly, independent batches/channels, ignored metadata | `marker_gather_f16_native_preserves_feature_bits_and_rounds_half_metadata`: B=2, H=3, T=9, K=7, CB=16; independent IEEE-half CPU decoder, three tile sizes and ragged lengths. An explicit assertion requires every feature edge-case to be selected: quiet/signaling NaN payloads, negative zero, smallest/largest subnormal, smallest normal and infinities. Poison masks/extra columns must not be read |
| Round after half quantization, not before | `marker_gather_f16_native_uses_quantized_not_original_f32_positions`: f32 immediately below .5/1.5 rounds to half .5/1.5 and therefore selects columns 1/2. Direct bit-input test also covers adjacent half values, duplicate, negative, out-of-range, subnormal and NaN/infinite marker positions |
| Reject invalid dimensions/storage before unsafe launch | `marker_gather_f16_native_rejects_bad_layouts_and_short_storage`: zero/rank/height/batch/T/channel/overflow cases and truncated feature/marker buffers |
| Runtime preflight, compact lifetimes, positive/negative references, aliased sources | `marker_gather_f16_runtime_handles_nonprevious_sources_and_aliased_metadata`: two identity convs, B=2, hidden=16, T=5, K=4; sources `0,1` and `-2,-2`; exact CPU gather reference on quantized input. Nonprevious feature source must survive until gather |
| Native dispatch rather than generic host default | UFCS trait calls in direct tests plus source inspection of the trait override, native allocation/launch and u16 kernel; runtime calls the same trait method |

Tests were written before the new trait/operator existed and failed compilation; the three direct native tests passed before the runtime rejection was replaced. Final serial RTX 4060 Laptop runs, all through 3G systemd scopes with script `-j 2`:

- `MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather_f16`: **4 passed**.
- `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16`: **6 passed**, including existing real-cfg inference and unsupported-mode checks.
- `MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather`: **4 passed** (f32 forward/backward/FD regressions).
- `MEM=3G ./tools/test_host.sh marker_gather`: **12 core passed**, 0 CLI matched.

Targeted edition-2024 Rustfmt/diff checks passed; only the newly inserted half.rs blocks were formatted/checked to avoid rewriting that file's existing style. Temporary logs/formatter snippet removed after recording evidence. Parked CPU activation code remained in the working tree but was not selected or committed. No training run, benchmark, full-host/corpus/full-Oxide run, new optimizer proof or compiled-C++ golden. GPU status for marker gather: native f32 full/compact forward/backward and native f16 compact inference. Other layers, GPU Muon, remaining CPU/axis work and sparse MoE remain open. Half storage is a Rust backend extension of the inspected C++ routing math; hand-derived from source, no compiled-C++ golden. Next bounded GPU work is native LAYA act-head, starting with its exact detached-feature and frozen/update contract audit.

## 43. Native detached act-head feature primitive

- v0.2.003 (`3de7c327`): added the standalone `F32TensorOpsBackend::laya_act_features_f32` operation. `darknet-cuda/src/host_ops/laya_act.rs` owns checked layout and the generic reference, exposed via `host_ops.rs`; Oxide's override in `src/lib.rs` validates actual storage and launches `kernels/layers_extra.rs::laya_act_features`. All input data/reductions stay on device; one thread owns each sample's `[H+4]` row with serial option reductions and no scratch allocation/atomics. `f32v` only borrows or widens on device. This operation is **not yet wired into act-head runtime** and is not the MLP/backward/update port.
- Source audit: pinned `Jev_laya=4af013d2c59d1db54247aeff21f43e1d8d295cda`, retrieved executable `src-lib/laya_act_head_layer.cpp` with git show. `ah_features` (:29–78) uses CLS column zero plus `[top1, top1-top2, entropy/log(max(valid_count,2)), max(valid_count,2)/255]`; masks use odd channel `2*j+1`, column zero, strict `>0.5`. Masked logits become finite `-1e4`, not excluded/-infinity; softmax includes all K slots, with entropy log floor `1e-9`, no extra temperature or entropy/count clamp. Head/marker sequence lengths are independent.
- **Newly confirmed full-port gap:** C++ `ah_backward_sample` (:80–122), `backward_laya_act_head_layer` and `update_laya_act_head_layer` (:309–336) accumulate/update all four parameter blocks. Its GPU backward (:344–383) itself uses host math, which is not the native-GPU target. Rust's pure `act_head_backward` returns four parameter gradients, but `runtime.rs::backward_laya_act_head` forwards only dcls; accumulated ActHead params are ignored and no train buffers are allocated. The module's frozen-weight description is therefore not a full source-training contract. A question about including the training fix was asked but not answered; automatic continuation is not treated as approval to change existing frozen behavior. This release changes no training semantics and advances an independent GPU primitive. Full training parity remains unachieved.
- The existing native `F32ActivationKind::Gelu` is tanh-approximate; CPU act-head uses exact erf-GELU via `laya_scorer::{gelu_erf,gelu_erf_grad}`. A native exact-erf primitive is an independent remaining prerequisite; do not substitute the tanh approximation.

Scoped requirement-to-artifact audit (`oxide_laya_act_features.rs`):

| Requirement | Actual evidence |
|---|---|
| Checked geometry, complete storage, safe device output ownership | Shared Layout requires `[B,H,1,T_head]`, `[B,1,1,K>=2]`, `[B,CB>=2K,1,T_marker]`, positive dimensions, shared B and bounded checked products; Oxide override checks all three buffer lengths before device views/launch |
| Source math, batches/channels/sequence axes, threshold and ignored columns | `laya_act_features_native_matches_source_hand_math_and_cpu_batches`: B=3, H=3, K=4, CB=10, T_head=5, T_marker=3; three tile sizes, poisoned non-CLS/even-mask/unused columns, strict .5 vs .5001 masks, independent CPU helper/reference comparisons |
| Hand math beyond reference-only parity | Same test explicitly checks (.75,.25) top/gap/entropy/count, all-masked uniform K=4 with normalized entropy **2**, and tied masked logits outranking a valid -10001 score |
| Rust defensive nonfinite behavior, uncapped count feature | `laya_act_features_native_preserves_rust_nonfinite_policy_and_large_k`: NaN/+Inf/-Inf logits compared by finite values/NaN classification to Rust CPU, not C++ -Ofast behavior; direct K=257 operation proves count/255 can exceed one. Generic host reference also compared to the CPU helper |
| Invalid input rejection | `laya_act_features_native_rejects_bad_layout_and_truncated_storage`: bad rank/height/scorer channels/K/marker channels/batch, zero size, overflow and three short-buffer cases |
| No premature native-layer claim | Runtime dispatch/fallback, weights and optimizer membership were not modified. No full-network native act-head execution, native MLP, backward, CLS-adjoint scatter or parameter-update proof is claimed |

Test-first: direct tests failed compilation on the missing backend method before implementation. Final serial RTX 4060 Laptop runs, all via 3G scopes and script `-j 2`:

- `MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_features`: **3 passed**.
- `MEM=3G ./tools/test_oxide.sh --test oxide_layers_extra`: **24 passed**.
- `MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather_f16`: **4 passed**.
- `MEM=3G ./tools/test_host.sh laya_act_head`: **9 core passed**, 0 CLI matched.

Targeted edition-2024 Rustfmt/diff checks passed. Temporary source/log snapshots removed after recording evidence; no weights or sidecars created. Parked uncommitted CPU math remained unselected and outside this release. GPU status: feature primitive native f32; **act-head layer remains host fallback**, f16 unsupported. No new derivative/optimizer/checkpoint proof, training run, benchmark, full-host/corpus/full-Oxide run or compiled-C++ golden. Other GPU layers/GPU Muon, CPU/config/axis gaps and sparse MoE remain open. Hand-derived from source, no compiled-C++ golden.

## 44. Native exact-erf GELU primitives

- v0.2.004 (`eec5eab0`): added dedicated `F32TensorOpsBackend::{gelu_erf_f32,gelu_erf_backward_f32}`, without changing `F32ActivationKind::Gelu`, activation IDs, cfg behavior or existing AMP/f16 dispatch. `darknet-cuda/src/host_ops/gelu_erf.rs` supplies checked bounded shape products and exact-erf host reference; `host_ops.rs` exposes it and the CUDA contract crate enables `float_erf`, matching the already pinned-nightly core reference. No Cargo/dependency changes.
- Oxide's overrides in `src/lib.rs` validate positive nonempty declared shapes, actual input/seed metadata and lengths before allocation/launch. `src/kernels/activations.rs` adds dedicated elementwise forward/backward kernels and scalar helpers: y=.5*x*(1+erf(x/sqrt(2))); derivative=normal CDF+x*normal PDF; backward multiplies the supplied seed. Arrays stay on device, with one thread/output and no host math/downloads in the override. Generic trait defaults remain host references for other backends.
- CPU oracle: `laya_scorer::{gelu_erf,gelu_erf_grad}`; source context is the pinned C++ act-head exact-erf function recorded in §43. The custom collector intentionally rejects Rust's std erf shim, so the GPU uses the supported `#[device] extern "C"` libdevice `__nv_erff` route instead. Inspected generated linked LLVM and confirmed an internal **defined** `__nv_erff` body, not just an unresolved declaration; real-device execution passed. No compiler/bridge changes, approximation or host fallback was added. "Exact-erf" identifies the source variant, not bit-exact CPU/libdevice agreement.

Scoped requirement-to-artifact audit (`oxide_gelu_erf.rs`):

| Requirement | Actual evidence |
|---|---|
| Source variant and seeded analytic derivative | `gelu_erf_native_forward_backward_matches_hand_values_and_cpu_not_tanh`: hand y at -1/0/1/2 and derivative at +/-1, full CPU and new host-helper comparisons, nonunit/zero/signed seeds, B=2/C=2/H=3/W=5 and tile lengths 32/128/256 with ragged tails. Existing native tanh GELU stays measurably distinct at x=1 |
| Differentiate actual device forward, not only a CPU surrogate | `gelu_erf_native_backward_matches_actual_gpu_forward_central_differences`: ten selected coordinates spanning both batches, actual GPU forward dot-product scores, epsilon .001 and absolute derivative tolerance .0005 |
| Explicit IEEE/nonfinite contract, no silent sanitation | `gelu_erf_native_extremes_and_nonfinite_follow_literal_cpu_formula`: finite +/-MAX, +/-100/15, tiny/subnormal inputs, NaN/infinite inputs and seeds. Literal -Inf*0 forward and Inf*0 derivatives remain NaN as in the CPU formula; no substituted limiting values or clipping |
| Fail before unsafe device access | `gelu_erf_native_rejects_bad_shape_storage_and_seed_metadata`: empty/zero/overflow shapes, wrong element count, equal-length wrong layout, short input/seed and equal-length seed metadata mismatch |
| Preserve existing APIs and avoid premature layer completion | Existing enum/kind mappings untouched; no act-head runtime, parameter, optimizer or checkpoint modifications. New f32 primitives only; no new f16 or act-head network integration claimed |

Test-first: new tests initially failed compilation on missing methods. Final serial RTX 4060 Laptop runs, via 3G scopes/script `-j 2`:

- `MEM=3G ./tools/test_oxide.sh --test oxide_gelu_erf`: **4 passed**.
- `MEM=3G ./tools/test_oxide.sh --test oxide_grad_methods`: **2 passed, 1 ignored** (timed benchmark not run; cases include leaky/logistic/mish and other existing gradient ops, not an all-activation coverage claim).
- `MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_features`: **3 passed**.
- `MEM=3G ./tools/test_host.sh gelu`: **2 core passed**, 0 CLI matched.

Targeted edition-2024 Rustfmt/diff checks passed; only inserted activation-file blocks were formatted/checked to preserve unrelated existing style. Temporary logs/formatter snippet removed after recording evidence. Generated compiler products remain part of the active incremental build. Parked uncommitted CPU math remained outside this release. No real training, benchmark, full-host/corpus/full-Oxide run or compiled-C++ golden. GPU status: exact-erf f32 forward/backward primitive native; **act-head runtime remains host fallback** and its training-contract question remains unanswered. Next independent work is native act-head f32 forward composition, retaining CPU backward until gradient/update eligibility is resolved and implemented. All broader GPU/GPU Muon, CPU/config/axis and sparse-MoE requirements remain open. Hand-derived from source, no compiled-C++ golden.

## 45. Native act-head f32 forward composition

- v0.2.005 (`79ce20bb`): replaced `[laya_act_head]` full/compact GPU forward's three source downloads, host feature/MLP execution and result upload with shared native composition in `runtime/laya_act_gpu.rs`: native features, connected fc1, dedicated exact-erf GELU, connected fc2. Thin arms in `runtime.rs` retain existing source resolution/use counts. Existing Oxide connected/feature/GELU overrides execute on device; no new backend trait or kernel was needed. Ordinary runtime debug/final output downloads remain.
- Helper checks source/spec agreement, positive/bounded F/A and checked products plus all four parameter lengths before head uploads/math. Matrix orientation follows source row-major `[F,H+4]` / `[A,F]`, with connected output `[B,F]` used for the strict exact-GELU metadata contract. Final `[B,A]` storage is downloaded with the existing layer's NCHW shape, as with other connected operations.
- **Deliberately uncached:** all four parameter blocks upload each call, so direct host edits and blob-based parameter replacement are immediately visible. Native compute does not imply fully resident weights or a speedup. Precision inherits the existing connected/GEMM backend configuration; numerical parity tests explicitly select `GemmMode::Fp32`. Other precision/AMP modes were not newly validated or globally changed. F16 remains rejected.
- CPU-backward fallback and act-head parameter eligibility/update behavior remain unchanged. No native act-head backward, new training buffers, optimizer state or checkpoint/file loader was introduced. In-memory `to_blob/from_blob` replacement is tested, **not shared-file loading**; the existing runtime error's `load_laya_act_head_blob` hint does not identify an implemented loader in core. Do not treat that hint or these forward tests as full `.weights` support. Source-training mismatch/question from §43 remains open.

Scoped requirement-to-artifact audit:

| Requirement | Actual evidence |
|---|---|
| No host act-head math in inference forward | `runtime/tests/laya_act_gpu_tests.rs::laya_act_head_gpu_forward_matches_hand_erf_and_retains_cpu_backward`: test-only thread-local probe in `laya_act_head.rs` asserts no CPU act-head calls from full/compact inference; known fc1 values +/-1 give the closed-form erf outputs. Probe is absent in production builds |
| Keep fallback and existing update behavior | Same FakeBackend test asserts fallback selection remains true, executes backward after GPU training-forward, compares upstream conv SGD updates to CPU and proves a nonzero update; all act-head params remain unchanged. This is preservation evidence on FakeBackend, not native GPU backward proof |
| Checked parameter geometry, all blocks | `laya_act_head_gpu_forward_rejects_all_parameter_blocks_and_bad_dimensions`: short fc1_w/fc1_b/fc2_w/fc2_b in both paths and a post-construction oversized F fail cleanly. This is runtime defense, not a claim that arbitrary oversized cfgs are safe to initialize |
| Real-device full/compact CPU-reference agreement | `oxide_laya_act_forward.rs::laya_act_head_native_full_compact_observe_edits_blob_replacement_and_sources`: B=2, H=6/8, K=T=4, F=5, A=3; valid/all-masked/aliased-marker graphs, absolute and relative references, nonprevious head input surviving compact lifetimes, all four blocks edited after warm calls, then blob-based replacement; outputs must actually change and both GPU paths match CPU |
| Actual native primitive dispatch | Core helper has only validated parameter uploads and backend feature/connected/exact-GELU calls, with no data downloads or host tensor math. Oxide implementations of these operations were directly verified in prior increments; real runtime composition now executes successfully on RTX 4060 |
| Clean errors and unchanged unsupported dtype | `laya_act_head_native_invalid_parameters_error_cleanly`: real GPU full/compact rejects short fc1 weights; f16 still returns its explicit unsupported-layer error |

Test-first: the host probe failed on the old CPU-forward calls and the malformed-parameter assertion failed before the new helper was wired. Final serial 3G scopes/script `-j 2`:

- `MEM=3G ./tools/test_host.sh laya_act_head`: **11 core passed**, 0 CLI matched.
- `MEM=3G ./tools/test_host.sh runtime::`: **273 core passed**, 0 CLI matched.
- `MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_forward`: **2 passed** on RTX 4060 Laptop.
- `MEM=3G ./tools/test_oxide.sh --test oxide_gelu_erf`: **4 passed** on RTX 4060 Laptop.

Targeted edition-2024 Rustfmt/diff checks passed; only new CPU-module probe blocks were formatted to avoid unrelated style churn. Temporary logs/formatter snippet removed after recording evidence; no file weights/sidecars created. Parked CPU math remained unselected/outside this release. No actual training run, benchmark, full-host/corpus/full-Oxide or compiled-C++ golden. GPU status: **native f32 full/compact forward; CPU backward; frozen/no act-head updates preserved; f16 unsupported**. Next independent work can implement standalone native act-head gradient math without changing runtime eligibility or applying updates until the operator question is resolved. All broader CPU/config/axis, GPU/GPU Muon and sparse-MoE requirements remain open. Hand-derived from source, no compiled-C++ golden.

## 46. Standalone native act-head gradient math

- v0.2.006 (`2ffc33e2`): added `F32TensorOpsBackend::laya_act_head_backward_f32`, typed resident operands/results (`LayaActBackwardInputs`, `LayaActHeadGrad`) and shared composition in `darknet-cuda/src/laya_act.rs`. Immutable `MlpLayout` in `host_ops/laya_act.rs` validates source views and positive/bounded MLP products before construction. Oxide's override in `src/lib.rs` checks all seven buffer lengths and canonical parameter/seed metadata before any math; source storage uses the declared NCHW views, like the existing feature primitive. Seeds accept `[B,A]` or `[B,A,1,1]`.
- Composition uses existing feature, connected, exact-erf forward/backward, connected-backward and route-copy operations: recompute x/preactivation/hidden, compute fc2 gradients, apply exact activation chain, compute fc1 gradients, extract the first H input-gradient columns as `[B,H]` dcls. All four parameter gradients are positive, **batch-summed**, not averaged or optimizer updates. The last four feature adjoints are discarded; no scorer/marker gradients escape. Fc2 bias **values** are unnecessary for this Jacobian, so are not an operand; fc2 bias gradients are still returned.
- Existing Oxide overrides for every composed operation use device kernels/private GEMM; no new GPU kernel, host tensor downloads/math, parameter upload, persistent cache, compiler change, runtime dispatch or optimizer mutation was introduced. Generic backends use their own primitive implementations and may execute host references. Verified precision is explicit `GemmMode::Fp32`, not all backend precision/AMP modes.

Scoped requirement-to-artifact audit (`oxide_laya_act_backward.rs`):

| Requirement | Actual evidence |
|---|---|
| All four gradients and dcls match CPU, sign and batch convention | `laya_act_backward_native_all_blocks_and_cls_match_cpu_sums_without_updates`: B=2, H=3, K=4, CB=10, T_head=5, T_marker=3, F=5, A=3; mixed/all-masked inputs, signed/zero/nonunit seeds and three tile lengths. Explicit db2=(.5,.5,-1.5) proves sum rather than mean; all result metadata checked and all seven resident operands remain unchanged |
| Honest oracle domain | CPU `laya_act_head.rs::validate_shapes` requires CB exactly 2*K, whereas native feature views allow extra ignored channels. The first oracle call failed on CB=10; tests now compact only each sample's unused trailing channels for the CPU oracle (CB=8), preserving every read mask value. No CPU validator/schema changed, and no full CPU/config parity for CB>2K is claimed |
| Actual GPU-forward finite differences | `laya_act_backward_native_matches_actual_gpu_forward_parameter_and_cls_fd`: two selected entries per parameter block (8 total) plus 3 CLS/2 non-CLS input coordinates, epsilon .001 and tolerance .001. Device forward composition is scored directly; scorer/marker buffers remain fixed to honor detach semantics |
| Checked descriptors/operand storage/layout | `laya_act_backward_native_checks_layout_all_operands_and_seed_metadata`: zero/overflow F/A, invalid head/marker views, each of seven truncated operands, equal-length wrong fc1/fc2/bias/seed metadata. Constructor state is private/immutable, preventing safe mutation into unchecked dimensions |
| Owned results, not dangling operand views | `laya_act_backward_native_results_outlive_temporary_operand_buffers`: drops all operand buffers, forces allocator-pool reuse with fifteen new uploads and verifies all five gradient outputs afterward |
| No premature native-runtime/update claim | Runtime source, fallback selection, act-head buffers/eligibility, optimizer/checkpoint handling and f16 rejection are unchanged. Returned dcls is not full sequence scatter; the new API is not called by runtime backward |

Test-first: missing imports/methods failed compilation before implementation. Final serial RTX 4060 Laptop runs, 3G scopes/script `-j 2`:

- `MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_backward`: **4 passed**.
- `MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_forward`: **2 passed**.
- `MEM=3G ./tools/test_oxide.sh --test oxide_gelu_erf`: **4 passed**.
- `MEM=3G ./tools/test_host.sh laya_act_head`: **11 core passed**, 0 CLI matched.

Targeted edition-2024 Rustfmt/diff checks passed. Temporary logs removed after recording evidence; no weight/sidecar files created. Parked CPU math remained unselected/outside this release. No real training, benchmark, unfiltered host/corpus/full-Oxide run or compiled-C++ golden. GPU status: standalone f32 gradient math native; act-head runtime still has native f32 forward plus CPU backward, frozen updates and unsupported f16. Source-training question, full-sequence adjoint routing, runtime/update/file-loading integration and broader CPU/axis/GPU Muon/sparse-MoE work remain open. Hand-derived from source, no compiled-C++ golden.

## 47. Native `[deform_conv]` forward and backward

v0.2.007 (`15fc3f8c9cd96cae8c315fbea78821cac867cd40`). The version map was also
missing v0.2.006's row; both are appended (indices 306, 307).

- Shared deformable sampling implemented twice and kept in agreement: host
  contract `darknet_cuda::host_ops::bilinear_gather` (`BilinearGatherGeometry`,
  `forward`, `backward`, `sample_at`) and device kernels
  `src/darknet-cuda-oxide/src/kernels/deform_sample.rs`. Contract taken from
  `darknet_core::deform_conv::bilinear_sample`, including the far-outside
  zero-gradient early-out, the floor-cell one-sided convention at integers, and
  NaN propagation.
- Five forward stages on device: offset/mask predictor (a `groups = 1`
  convolution — `deform_offset_predict`), per-batch statistics
  (`deform_offset_stats`, f64, source gate order), envelope
  (`deform_offset_envelope`: sanitize → optional normalize → sanitize → centered
  lhtan), mask sigmoid, and the deformable accumulation
  (`deform_conv_accumulate`, ±10000 sanitize).
- Four backward stages, each FD-verified in isolation, then composed:
  accumulation adjoint (4 atomic scatter targets), envelope slope adjoint,
  the batch-coupled normalization Jacobian (`offset_normalization_backward`),
  and the predictor + column-gather adjoint.
- Runtime wiring: `DeformDeviceCache` holds the device intermediates; a single
  shared predicate (`has_host_only_deform`: `dilation != 1 || batch_normalize`)
  gates **both** directions; `DeformConv` left
  `requires_cpu_backward_fallback`; the backward arm composes the trait
  primitives, performs exactly one host hop (normalization Jacobian + stage-1
  slope), applies the activation derivative and `bias_backward`, and accumulates
  into the existing `LayerTrainBuffers::DeformConv` fields.

Scoped requirement-to-artifact audit:

| Requirement | Actual evidence |
|---|---|
| Host contract for sampling | `cargo test -p darknet-cuda --lib` (33 tests, 6 for `bilinear_gather`) |
| Device sampling parity incl. adjoint | `MEM=3G ./tools/test_oxide.sh --test oxide_deform_gather` — 5 passed |
| Forward stages + composition | `--test oxide_deform_conv_accumulate` 8 passed, `--test oxide_deform_envelope` 14 passed |
| Forward vs the real CPU runtime | `--test oxide_deform_conv_chain` 3 passed (masked/unmasked + dilation rejection) |
| Backward stages FD-exact | `cargo test -p darknet-cuda --lib` (accumulation 4 operand groups, envelope both modes, normalization Jacobian through recomputed stats, predictor) |
| Backward vs the real CPU backward | `--test oxide_deform_conv_backward_chain` 3 passed (main/offset/mask weights and biases) |
| Device backward composition | included in the same suite's `native_backward_composition_matches_the_cpu_gradients` |
| Runtime device path end to end | `--test oxide_runtime_chain` 26 passed, incl. `deform_conv_device_path_matches_cpu_chain_and_updates` (forward outputs, one SGD step, every parameter block, "parameters actually moved") and the retained BN host-fallback test |
| Host path unaffected | `MEM=3G ./tools/test_host.sh deform_conv` 23 passed |
| Honest scope | `layer-support.md` records `native (dilation 1, no BN) / host otherwise`; the one host hop is documented; no compiled-C++ golden exists |

Deliberately not done: `dilation > 1` and batch-norm variants stay on the host
path (tested); `f16` is not supported for the layer; no benchmark or speedup
claim; `[dcnv4]`, `[dsconv]`, `[graph_conv]` and the remaining hard-reject layers
are untouched.

## 48. DCNv4 native aggregation primitives (offsets given)

**Evidence correction in v0.2.012 (§49):** the v0.2.009–011 stage reference
and kernels shared the wrong point count/row stride (`size²`, unpadded rows).
Their stage parity/FD results did not establish real CPU-layer agreement. Both
host/device layouts are now corrected and those tests rerun; §49 adds the real
CPU runtime oracle. Keep the original counts below as historical evidence only.

- v0.2.011 (`8fe32a09`): native f32 `OxideBackend::dcnv4_aggregate_backward_native`
  returns NCHW input adjoints and packed `[B, OH*OW, groups*3*K]` offset/mask
  adjoints; device kernel `kernels/dcnv4.rs::backward` owns each row and uses
  atomics only for input scatter. Fixed the pending kernel's batch-zero-only
  thread bound (red test: batch 1 input gradient was 0 instead of 0.67130804).
  Shared `Dcnv4Geometry::checked` now rejects dimension-product overflow before
  unsafe launch/allocation arithmetic. No runtime/optimizer behavior changed.
- v0.2.010 (`7874c225`): added native aggregation forward and its checked
  launcher, originally verified by 2 device tests. The fixed mask array permits
  at most 25 kernel points; larger kernels error. This release's 6-test run
  rechecks that forward, including the strided/dilated/edge fixtures below.
- v0.2.009 (`e3cfb27d`): added `host_ops::dcnv4::{aggregate,
  aggregate_backward, Dcnv4Geometry, group_masks}` with host finite differences.
  The host suite was rerun in this increment (35 passed).

Scoped requirement-to-artifact audit (all commands serial, `-j 2`, 3G user
systemd scopes; no training runs):

| Requirement | Actual evidence |
|---|---|
| Native adjoint, all batches/groups and four mask/center modes | `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_aggregate`: **6 passed**; `device_aggregation_backward_matches_the_host_reference` covers B=2, C/G=2/1 and 4/2, signed/nonunit seeds; compares every input and offset element at 2e-5 absolute + relative tolerance |
| Different geometry and edge sampling | `device_adjoint_covers_strided_dilated_and_far_outside_samples`: sizes 1/3/5, groups=2, stride x/y=2/3, dilation=2, non-square input, padding=2, signed masks and large displacements outside all edges, all four modes; forward and both adjoints compared |
| Independent actual-device derivative check | `device_adjoint_matches_device_forward_finite_differences`: every input and packed offset/mask value, B=2/groups=2, all four modes; actual native forward objective with signed seeds; scale=1, coordinate fraction=0.5, epsilon=0.01 (cannot cross floor seams), absolute tolerance 2e-3 |
| Fail closed before unsafe launch | `device_adjoint_rejects_bad_geometry_and_each_short_operand`: each short operand; zero/nondividing groups, zero batch/dilation, >25 points and overflowing dimensions |
| Host oracle regression | `systemd-run --user --scope -p MemoryHigh=3G cargo test -p darknet-cuda --lib -j 2 -- --test-threads=1`: **35 passed**, including host DCNv4 finite differences |
| CPU/runtime behavior preserved | `MEM=3G ./tools/test_host.sh dcnv4`: **17 core passed**, 0 CLI; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4`: **1 passed** (`dcnv4_f32_host_fallback_matches_cpu_chain_and_updates`) |
| Storage ownership | Launcher validates full lengths and allocates distinct zeroed `d_input`/`d_offsets`; `OxideBackend::alloc` zeros recycled buffers; thread bound derives from full seed length; `d_offsets` has a unique writer per row, input scatter is atomic |
| Scope and evidence honesty | Runtime still lists `Dcnv4` in `requires_cpu_backward_fallback`; no trait/runtime wiring, predictor/backward composition, native updates, f16, benchmark or speedup claim. Device vs Rust CPU reference; hand-derived from source, no compiled-C++ golden |

Targeted Rustfmt and `git diff --check` passed. Full host/corpus/Oxide suites
were not run; existing corpus residuals remain open. No temporary weights,
optimizer sidecars, models or test logs were created by this slice. Existing
compiler IR/PTX and older build trees are retained because the backend workflow
and prior evidence refer to them; no unrelated cleanup was attempted.

## 49. Native DCNv4 projection and backward composition

- v0.2.012 (`c51b25f6`): `dcnv4_forward_native` now composes native coarse projection,
  replicated-border lattice interpolation and deformable aggregation, returning
  raw NCHW output and dense NHWC offset cache. `dcnv4_backward_native` returns
  input plus positive batch-summed offset weight/bias gradients through all
  three stages. No tensor downloads/host math, runtime fallback removal,
  epilogue/optimizer behavior change, f16 support or performance claim.
- Corrected shared aggregation layout: `K=size²-remove_center`, row width
  `ceil(groups*3*K/8)*8`, matching `Jev_laya:src-lib/dcnv4_layer.cpp:27–36`
  and real Rust CPU `dcnv4.rs::layout`. The red host layout test observed
  row width **27 vs expected 32**. Size=1/remove_center now fails explicitly.
- Checked `Dcnv4Composition` validates source output extents, positive d_stride,
  finite scale, padded storage/counts, coarse strides and coordinate arithmetic
  before unsafe launches. New kernels live in `kernels/dcnv4_projection.rs`;
  wrappers/types in `dcnv4_native.rs`; aggregation kernels remain in `dcnv4.rs`.
  Input/weight/coarse gradient scatters use atomics: summation order is not
  bitwise deterministic. Direct-loop correctness kernels, not a speedup claim.

Scoped requirement-to-artifact audit (serial `-j 2`, MemoryHigh=3G user scopes):

| Requirement | Actual evidence |
|---|---|
| Real CPU forward and complete math gradients | `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_chain`: **3 passed**. `native_composition_matches_real_cpu_outputs_and_all_math_gradients` constructs a real one-layer `ExecutableNetwork`, uses its `forward`/`backward` outputs and compares every output, input gradient, offset weight and bias at abs+rel 3e-4 tolerance. Eighteen cases: B=2, non-square 7x5 input, sizes=1/3/5, groups=1/2, stride x/y=2/1, d_stride=1/2/3, dilation=1/2, all four softmax/remove_center combinations where valid (size=1/remove_center is explicitly rejected). No synthetic stage oracle |
| Packed axes/padding are real, not shared-proxy agreement | `aggregation_layout_uses_active_points_and_eight_aligned_rows` pins K=9/8 and rows=32/24/48; real CPU parameter cardinalities asserted in chain fixtures. Padding biases are set to 20, but padding weight/bias gradients must remain exactly zero; all active parameter gradients compared |
| Differentiate full native forward, not only aggregation | `native_composition_matches_device_finite_differences`: all four modes, coarse d_stride=2, every bias plus sampled inputs/weights, signed seed objective via actual native forward; epsilon=0.002, abs tolerance 2e-3; derived coordinate margin >0.48 excludes floor-cell seams |
| Cache/lifetime/input immutability | Real CPU comparison drops original forward input/weight/bias uploads before independent backward uploads to force pool reuse; output and cache remain live and unchanged, input/weights unchanged after backward |
| Reject malformed descriptors/storage | `native_composition_rejects_bad_descriptors_and_storage`: zero/overflowing d_stride, nonfinite scale, wrong output extent, overflowing padding, oversized kernel; each short forward operand and each short backward operand rejected |
| Retain aggregation/FD/edge/error coverage after layout fix | `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_aggregate`: **6 passed**, including size=1/remove_center rejection, eight-aligned padding and corrected compact center-removal slots |
| Host math regression and source-layout pin | `systemd-run --user --scope -p MemoryHigh=3G cargo test -p darknet-cuda --lib -j 2 -- --test-threads=1`: **36 passed** |
| CPU/runtime behavior preserved | `MEM=3G ./tools/test_host.sh dcnv4`: **17 core passed**, 0 CLI; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4`: **1 passed** (host fallback plus updates). `runtime.rs` unmodified, `Dcnv4` still forces CPU backward fallback |
| Native execution rather than generic host defaults | Wrappers launch four `dcnv4_projection` device entries plus existing native aggregation/adjoint; wrapper source has no download/CPU math calls. Both new tests exercise an actual Oxide backend |

Targeted Rustfmt and diff checks passed. Initial device compile caught unsupported
DisjointSlice indexing (changed to owned, bounds-checked pointer writes); first
CPU-oracle fixture caught `element_count()` excluding batch (fixed full B=2 data
length). No failed test is left in the slice. No training/data/model downloads,
weights, sidecars or logs were created; existing compiler IR/PTX and unrelated
build trees were inspected and retained. No full-host/corpus/full-Oxide or
compiled-C++ parity claim: **hand-derived from source, no compiled-C++ golden**.

Completion boundary: this completes standalone raw DCNv4 math composition,
not the handbook's all-layer goal. **Next:** add trait defaults/Oxide overrides,
wire both forward and backward/cache selection together, apply existing
bias/activation/BN semantics, then prove two-step optimizer/weight parity before
removing runtime/optimizer rejection. Matrix remains **host fallback**. Stage-1
missing variants/axis pins, other Stage-2 native layers/GPU Muon, and Stage-3
sparse MoE remain open; these selected tests do not cover them.

## 50. Bounded DCNv4 native runtime forward/backward

- v0.2.013 (`3f3c2226`): generic `dcnv4_forward_f32` / `dcnv4_backward_f32` trait entries
  have host defaults (`host_ops/dcnv4_composition.rs`) and direct Oxide native
  overrides. Runtime dispatch is in `runtime/dcnv4_gpu.rs`, with small guarded
  arms in `runtime.rs`. Eligible f32 networks now execute projection/upsample/
  aggregation plus bias/activation and input/parameter gradient math on device.
- Gate: no DCNv4 BN, <=25 active points, supported elementwise activation, no
  whole-network CPU fallback or host-only deform_conv variant. Both forward and
  backward use the same global gate. Excluded/mixed networks still produce and
  consume the original host caches; f16 and compact rejection are unchanged.
- Backward recomputes offsets **on device** from that forward state's own input
  and current unchanged parameters; no mutable last-forward cache or public
  GpuForwardState shape change. Existing contract requires backward before an
  optimizer update. Projection weights/biases upload again and an unnecessary
  aggregation output is recomputed. No residency/performance/speedup claim.
- Parameter gradients download to existing host train buffers; SGD/Adam remain
  host-side for DCNv4, including the 0.01 predictor rate, gates/momentum/decay
  and Adam state layout. Dummy weights remain unchanged. GPU Muon is absent;
  this is bounded native math integration, **not full native-layer completion**.

Requirement-to-artifact checklist (serial builds/tests; user MemoryHigh=3G
scopes, -j 2 except the -j 4 CLI check):

| Requirement | Inspected evidence |
|---|---|
| Backend-generic surface with real native override | `darknet-cuda/src/lib.rs` declares both methods; `darknet-cuda-oxide/src/lib.rs` forwards directly to v0.2.012 native math. Host composition validates geometry/storage and returns positive batch sums |
| Host default derivative/storage validation | `systemd-run --user --scope -p MemoryHigh=3G cargo test -p darknet-cuda --lib -j 2 -- --test-threads=1`: **37 passed**, including every input/weight/bias FD through recomputed coarse offsets (B=2) and short-storage rejection |
| Test-first runtime selection | New `dcnv4_gpu_selects_native_math_without_host_cache` first failed the old unconditional fallback assertion; now passes and verifies the CPU cache is absent |
| No stale offsets or dropped subdivision gradients | `dcnv4_gpu_multiple_forward_states_and_microbatch_accumulation_match_cpu`: forward A then B, backward A then B on a standalone B=2/groups=2 DCNv4; every accumulated output-bias/offset-weight/offset-bias value agrees with CPU |
| Error before backend math | `dcnv4_gpu_rejects_parameter_storage_before_backend_math` truncates each of three parameter blocks; each returns InvalidConfig, not a panic or unsafe launch |
| Preserve whole-network fallback | Host tests `dcnv4_gpu_retains_host_cache_for_bn_and_mixed_fallback` (BN, large kernel, channel normalization, deconvolution neighbor) and `dcnv4_gpu_and_host_only_deform_choose_the_same_whole_network_path` verify cache production and successful backward. Real-device BN fallback and mixed deconvolution cases also pass |
| Mandatory runtime guard regression | `MEM=3G ./tools/test_host.sh runtime::`: **278 passed** (includes muon_runtime matches). Existing ViT fallback, deform_conv device/host gates, optimizer and parked cuboid guard tests remain green |
| Focused host/DCNv4 regression | `MEM=3G ./tools/test_host.sh dcnv4`: **22 core passed**, 0 CLI; includes prior CPU math/weights/optimizer-state tests |
| True device dispatch and two optimizer steps | `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4`: **3 passed**. `dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` uses preceding conv + B=2, groups=2, size=3, dilation=2, d_stride=2, signed/nonunit seeds, softmax/remove_center on+off, swish/leaky. Explicit Fp32 GEMM mode; all layer outputs and both steps' updated values/deltas agree; every conv/DCNv4 trainable block actually moves; dummy weights unchanged |
| Not a hidden CPU backward | In that device test CPU backward on GPU forward artifacts errors specifically for missing DCNv4 cache, while GPU backward and updates succeed; retained BN/mixed cases instead prove CPU backward can consume the GPU-produced host artifacts |
| Composition/adjoint regressions after trait migration | `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_chain`: **3 passed** (18 real CPU cases, device FD, errors/lifetimes); `--test oxide_dcnv4_aggregate`: **6 passed** |
| Integration check | `systemd-run --user --scope -p MemoryHigh=3G cargo check -p darknet-cli --features oxide -j 4`: passed |
| Formatting/scope/hygiene | Targeted Rustfmt/new-module formatting and `git diff --check` passed; only named GPU/host-default/runtime/test/docs files committed. No parked YOLO-3D, Cargo.lock or .gitignore edits included |

**Full runtime-chain suite remains unverified:** attempted serial
`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain` timed out at 600 seconds
while running `vit_flash_attention_runtime_uses_native_forward_and_backward_and_matches_cpu`.
The 25 preceding tests reported `ok` (including the three DCNv4 cases and the
existing deform_conv device/host cases), but there is no final suite result.
No test/build processes survived the timeout. The focused DCNv4 run above then
completed normally. Do not claim a full runtime-chain/Oxide green baseline or
infer the cause of the timeout. Full host/corpus were not run; known corpus
residuals remain unresolved.

Resolved build-only issues: helper needed the existing B::Error Display bound,
test helper needed its parent-qualified path, and Fp32 is GemmMode (not ConvMode).
No persistent model/data/log artifact was created intentionally; compiler IR/PTX
and unrelated build trees are retained because the build workflow/earlier docs
still reference them. Unit step tests only, no dataset training. **Hand-derived
from source, no compiled-C++ golden.**

Completion audit boundary: handbook deliverables are full CPU/source variants
plus axis pins/optimizers, all native GPU layer paths plus GPU Muon, then sparse
MoE with performance evidence, with per-version docs and C++-golden boundaries.
This slice only closes bounded DCNv4 f32 runtime math selection. BN, unsupported
activations/large kernels, compact/f16, native parameter gradients/updates/
residency and GPU Muon remain open, as do all documented CPU/other-GPU/MoE gaps.
Next independent DCNv4 slice: native rolling-stat BN affine math and its exact
adjoint (preserve the CPU DCNv4 rolling-stat semantics, not conv batch stats).

## 51. Native DCNv4 rolling-stat BN

- v0.2.014 (`ceef2037`): DCNv4's bounded f32 native runtime path now accepts BN. Existing
  device bias-add/channel-scale/product/reduction primitives implement
  `xhat=(raw-mean)/sqrt(var+1e-5)`, `pre=xhat*gamma`,
  `d_raw=delta*gamma/sqrt(var+1e-5)` and `d_gamma=sum(delta*xhat)`.
  Gamma may be zero or negative; no division by gamma. Ordinary biases and BN
  beta remain unused and unchanged, matching the existing Rust CPU contract.
- Native post-backward stats reduction preserves the CPU **0.9 EMA with unbiased
  variance**, for every DCNv4 BN layer including unseeded/gated layers. It runs
  after all gradients, never in forward. Reuses only the stats/EMA outputs of
  `batchnorm_train_forward_f32`; that primitive's batch-normalized output is
  discarded, not substituted for the layer's rolling-stat normalization.
  Only per-channel EMA values return to host params (no raw activation download).
- Code is in `runtime/dcnv4_gpu.rs`; `runtime.rs` adds one post-backward hook and
  updates the old BN test expectation. Eligibility still excludes >25 points,
  unsupported activations and whole-network host fallback/host-only deform_conv.
  Existing SGD/Adam buffers, rate scaling, gates, state encoding and ignored
  parameter blocks are unchanged. Parameter gradients/updates remain host-side;
  repeated uploads/raw recomputation and no residency/speedup claim remain.

Scoped prompt-to-artifact checklist (serial, MemoryHigh=3G user scopes, -j 2
except the -j 4 CLI check):

| Requirement | Concrete evidence |
|---|---|
| Test-first BN path relaxation | `dcnv4_bn_gpu_uses_native_path_before_stats_update` first failed the old unconditional BN fallback assertion. Now verifies native selection, no host math cache, no forward EMA update |
| Real CPU output/input/parameter derivatives and gamma=0 | `dcnv4_bn_native_gradients_match_cpu_including_zero_gamma` compares every input, offset weight/bias and gamma gradient with the real CPU backward (B=2/groups=2, non-square map, zero/negative gamma); gamma=0 derivative must be nonzero; rolling values agree after accumulation |
| Historical hand expectation and EMA timing retained | Existing `dcnv4_batchnorm_scale_derivative_and_rolling_stats_use_native_defaults` still pins gamma gradient and rolling mean=0.43, variance=0.7966667; CPU/FakeBackend buffers/params agree, now without CPU backward fallback |
| Unseeded stats, malformed BN metadata | `dcnv4_bn_native_updates_unseeded_stats_and_rejects_bad_bn_storage`: unseeded callback agrees with CPU; truncated beta/gamma/mean/variance or absent BN state return InvalidConfig before backend math |
| Update gates and ignored fields | `dcnv4_bn_native_preserves_optimizer_gates_but_advances_stats`: SGD and Adam, dont_update/train_only_bn/burnin_update; optimizer leaves all params unchanged relative to post-EMA state, while EMA still advances. Existing/real tests pin ordinary-bias/beta/dummy invariance |
| Mandatory full host runtime filter | `MEM=3G ./tools/test_host.sh runtime::`: **282 passed**; includes existing mixed fallback/host-only deform guards and all parked tests. Focused `test_host.sh dcnv4`: **26 core passed**, 0 CLI |
| Actual-device BN derivative check | `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_bn`: **2 passed**. `native_bn_runtime_forward_derivatives_hold_rolling_stats_fixed` differentiates actual GPU runtime forwards: input-adjoint direction via an upstream identity-conv update, sampled predictor weights/biases and both gammas (zero/negative). Supplied-seed lr=1 readout accounts for batch=2 and predictor rate=0.01; FD epsilon=0.002, floor seams excluded; every forward holds EMA fixed |
| Native unseeded EMA and N=1 variance | Same suite's `native_bn_unseeded_and_single_sample_stats_match_cpu` compares per-channel mean/variance against real CPU for B=2/non-square map and B=1/H=W=1; scale values unchanged |
| Native selection, two SGD/Adam steps, all blocks | `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4`: **3 passed**. Expanded two-step test covers BN off/on × SGD/Adam × softmax/remove_center on+off, B=2/groups=2, dilation=2/coarse stride=2, swish/leaky, zero/negative gamma, upstream conv and all active DCNv4 blocks moving; rolling stats agree, ordinary biases/beta/dummy invariant |
| Mixed networks retain original cache/stats path | Same device suite's deconvolution-mixed network now has BN DCNv4 and proves CPU backward can consume GPU-produced host caches. Native BN test proves the opposite (CPU backward fails specifically for missing DCNv4 cache while GPU backward succeeds) |
| Raw math regressions | `--test oxide_dcnv4_chain`: **3 passed** (18 CPU cases, device FD and guards); `--test oxide_dcnv4_aggregate`: **6 passed**; capped `cargo test -p darknet-cuda --lib -j 2 -- --test-threads=1`: **37 passed** |
| Native, not a new host math fallback | Inspected Oxide implementations of vector_mul/channel_scale/bias reductions and batchnorm_train_forward: device launches/reductions. Helper downloads only gradients and per-channel EMA values; CPU counterparts remain defaults for generic/FakeBackend |
| Integration/format/hygiene | Capped `cargo check -p darknet-cli --features oxide -j 4` passed; targeted Rustfmt/diff checks passed; no persistent weights/sidecars/ONNX artifacts found in the work scope. Unrelated user/parked edits excluded |

One compile-only issue was caught and fixed: vector_mul_f32 requires an explicit
output shape. No failed test remains in these focused gates. No new kernels,
optimizer formats, public forward-state fields, global AMP/precision semantics
or dataset training runs. Existing compiler IR/PTX/build trees remain referenced
and were retained. Full host/corpus/Oxide suites were not run; the previous
600-second runtime-chain ViT timeout remains unresolved and is not a pass.

Evidence boundary: **device vs existing Rust CPU contract; hand-derived from
source, no compiled-C++ golden**. C++ DCNv4 calls forward_batchnorm_layer and
its CPU backward is a stub; this is not evidence of C++ training BN parity.
Normalize with the current rolling values for a forward/backward pair; do not
change weights or rolling stats between that pair. Post-backward EMA is state
maintenance, not part of that forward's differentiated function.

Completion audit: BN math/EMA is closed within this increment, not the full
handbook objective. CPU source variants/axis pins, native unsupported/large
DCNv4 paths, compact/f16, device parameter updates/residency, other GPU layers,
GPU Muon and sparse MoE/performance evidence remain open. Next independent slice:
native DCNv4 NormalizeChannels* activations/adjoints using existing backend
primitives, preserving BN ordering and whole-network fallback for other gaps.

## 52. Native DCNv4 channel-normalization activations

- v0.2.015 (`1fc80f27`): DCNv4's bounded f32 native path now accepts `normalize_channels`, `_softmax` and `_softmax_maxval` activations (after optional rolling-stat BN) using the existing `normalize_channels_f32`/backward primitives; channel output is recomputed on device from the forward preactivation. Exact `SoftmaxMaxval` adjoint gained the max-chain term (epsilon sits outside exp(x-max); first max selected at ties; one-channel case is constant, zero translation derivative) in both host_ops and the Oxide kernel; `GradMode::Cpp` shortcut unchanged. Gates (serial, 3G): darknet-cuda lib 40, host dcnv4 28, `runtime::` 284, `oxide_dcnv4_normalization` 3 (3 norms x BN x SGD/Adam two-step CPU/GPU parity, GPU forward FD with fixed stats, mixed-network CPU cache), `oxide_layers_extra` 25. Not compiled-C++ parity; larger variants, other activations, compact/f16, device updates/residency, GPU Muon remain open.

## 53. Extend native DCNv4 kernel bound to 7x7

- v0.2.016 (`a96d6d95`): increased the fixed device mask capacity from 25 to 49 points and expanded the native composition CPU-parity test matrix with 7x7 cases. The runtime eligibility gate admits up to 49 active points; >49 remains host-fallback. Evidence: `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_chain` passed 3 tests (real GPU composition output, input/parameter-gradient CPU parity and error guards, including 7x7); `MEM=3G ./tools/test_host.sh dcnv4` passed 28 core tests including path-selection/fallback regressions. No 7x7 runtime optimizer-parity test, compiled-C++ parity, or performance claim. Other unsupported activations, device updates/residency, compact/f16, GPU Muon and broader parity work remain open.

## 54. DCNv4 7x7 runtime optimizer parity

- v0.2.017 (`8a3ef3b0`): extended `dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` with an actual 7x7 (49 active points) layer. GPU runtime forward/backward takes the native path (the CPU-cache fallback is asserted unavailable) and matches CPU output, parameter deltas and two updates under SGD/Adam, BN on/off. Evidence: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed on the RTX 4060; one test, ~108 seconds. No large-kernel >7x7 support, compiled-C++ parity, or performance claim.

## 55. Extend native DCNv4 runtime through 9x9

- v0.2.018 (`613b243e`): enlarged the fixed native mask buffer and runtime gate from 49 to 81 points. A 9x9 case now passes real-GPU native composition parity against CPU outputs and all math gradients, plus two-step runtime SGD/Adam parity with BN on/off. The over-limit host-fallback fixture now uses 11x11. Evidence: `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_chain -- native_composition_matches_real_cpu_outputs_and_all_math_gradients` passed; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed; `MEM=3G ./tools/test_host.sh dcnv4` passed 28 core tests. >81 points and unsupported activations remain fallback; no performance or compiled-C++ parity claim.

## 56. Extend native DCNv4 runtime through 11x11

- v0.2.019 (`76242245`): enlarged the native fixed mask capacity/runtime gate from 81 to 121 active points. 11x11 real-GPU composition matches CPU output and all math gradients; runtime two-step SGD/Adam parity passes with BN on/off, and the >121 host-fallback fixture uses 13x13. Gates: focused `oxide_dcnv4_chain` composition test passed, `oxide_runtime_chain` DCNv4 optimizer test passed (~186 s), and `MEM=3G ./tools/test_host.sh dcnv4` passed 28 core tests. No compiled-C++ or performance claim; >121/unsupported variants remain host fallback.

## 57. Extend native DCNv4 runtime through 13x13

- v0.2.020 (`c99f83d8`): increased the native fixed mask capacity/runtime gate from 121 to 169 active points. 13x13 real-GPU composition matches CPU outputs and all math gradients; runtime two-step SGD/Adam parity passes with BN on/off, and >169 remains host fallback (exclusion fixture is 15x15). Evidence: focused `oxide_dcnv4_chain` composition and `oxide_runtime_chain` DCNv4 optimizer tests passed; `MEM=3G ./tools/test_host.sh dcnv4` passed 28 core tests. No compiled-C++ or performance claim.

## 58. Retain native DCNv4 forward cache for backward

- v0.2.021 (`7a262a13`): store each native GPU forward's raw aggregation and dense offsets in its `GpuForwardState`; backward consumes that exact cache instead of re-uploading offset biases and recomputing projection/upsampling/aggregation. CPU fallback caches remain separate, and host-resident parameters/gradients/optimizer are unchanged. Per-forward ownership is covered by existing multiple-forward/microbatch tests, with explicit native-cache presence assertions. Gates: `MEM=3G ./tools/test_host.sh dcnv4` passed 28 core tests; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed on the RTX 4060 (1 test, 226 seconds). No speedup measured or claimed; full runtime chain/corpus not run.

## 59. Native DCNv4 gradient accumulation

- v0.2.022 (`27ab3cfb`): native DCNv4 backward now retains bias, offset-weight, offset-bias and optional BN-scale gradients as device tensors and accumulates them on-device across microbatches. No parameter-gradient downloads occur during backward; gradients transfer once at the host optimizer boundary, where existing SGD/Adam logic and parameter residency remain unchanged. The first gradient seeds an exact-layout device accumulator to avoid opaque backend tensor-shape assumptions. Host tests: `MEM=3G ./tools/test_host.sh dcnv4` passed 28 tests, including gradient and multi-forward/microbatch parity. Real GPU: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed on RTX 4060 (1 test, 229 s), covering two-step SGD/Adam CPU parameter parity through device accumulation. `git diff --check` passed. No speedup, compiled-C++ golden or full-suite claim; optimizer updates/parameter residency, unsupported variants, other GPU layers, CPU gaps, GPU Muon and sparse MoE remain open.

## 60. Native act-head CLS adjoint scatter

- v0.2.023 (`2ff4544e`): added a checked backend operation to add the standalone `[B,H]` act-head CLS adjoint into token zero of an existing `[B,H,1,T]` NCHW head gradient. Host reference validates dimensions/storage; Oxide uses a native one-thread-per-output kernel, keeping the full sequence gradient device-resident and leaving scorer/marker summary features detached. Evidence: `systemd-run --user --scope -p MemoryHigh=3G cargo test -p darknet-cuda --lib laya_act -j 2` passed 2; `MEM=3G ./tools/test_host.sh laya_act` passed 11 core tests; `MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_backward` passed 5 on the RTX 4060, including the new actual-device NCHW scatter/parity/error test and the existing MLP gradient parity/FD/guard suite; `git diff --check` passed. This is a primitive, not runtime backward/update integration; f16 and trainable-parameter contract remain unresolved, and no C++ golden or performance claim is made.

## 61. Native DCNv4 SGD optimizer update

- v0.2.024 (`0087d109`): eligible native DCNv4 SGD now combines device-accumulated gradients with device-resident momentum buffers and applies the standard backend SGD update to the offset projection and either ordinary biases or BN scales. The existing 0.01 offset-projection rate, weight decay on offset weights, layer rate scaling, batch handling and update gates are preserved. Parameters remain host-canonical and are uploaded/downloaded per step; Adam still flushes gradients to the existing host optimizer. Whole-network CPU fallback and non-native DCNv4 variants retain the host path. Evidence: `MEM=3G ./tools/test_host.sh dcnv4` passed 28 tests; `MEM=3G ./tools/test_oxide.sh dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed 1 actual-GPU test (RTX 4060, 258.95 s), exercising BN on/off, two-step SGD and Adam parity, multiple kernel sizes and update deltas; `git diff --check` passed. No speedup, parameter residency, compiled-C++ golden or full-suite claim. Native DCNv4 Adam, unsupported/large variants, compact/f16, other GPU layers, CPU gaps, GPU Muon and sparse MoE remain open.

## 62. Native DCNv4 Adam optimizer update

- v0.2.025 (`7c96d45a`): eligible native DCNv4 Adam now applies offset projection and ordinary-bias/BN-scale updates through backend optimizer operations. First/second moments reuse the existing `HostAdamLayerState` layout and optimizer-sidecar serialization, uploaded for the device step and returned to the host state afterward; canonical parameters remain host-side. Layer-rate scaling, the 0.01 offset-projection LR factor, Adam/AdamW decay behavior, update gates and host fallback for mixed/unsupported networks are preserved. Evidence: `MEM=3G ./tools/test_host.sh dcnv4` passed 29 tests, including FakeBackend CPU parameter parity and optimizer-sidecar save/load round-trip; `MEM=3G ./tools/test_oxide.sh dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed on RTX 4060 (1 actual-GPU test, 255.16 s), covering two-step SGD/Adam parity with BN on/off and multiple DCNv4 sizes. No full suite, speedup, persistent parameter residency or compiled-C++ golden is claimed; broader CPU/GPU gaps, GPU Muon and sparse MoE remain open.

## 63. Native Revleaky activation mapping

- v0.2.026 (`fafa95ae`): map the source-derived `Activation::Revleaky` scalar semantics to the existing native Leaky f32 backend operation (same positive/negative slopes and non-positive branch), removing its unnecessary device-path rejection for all f32 GPU consumers. DCNv4 now exercises Revleaky through native runtime forward/backward and SGD/Adam updates in the real-device CPU-parity matrix. Evidence: `MEM=3G ./tools/test_host.sh dcnv4` passed 29 tests, including native selection/cache behavior; `MEM=3G ./tools/test_oxide.sh dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed 1 RTX 4060 test (262.86 s) with the added Revleaky case. No compiled-C++ golden, speedup, or full-suite claim; other GPU layer gaps and overall CPU/axis/Muon/sparse-MoE work remain.

## 64. Extend native DCNv4 through 15x15

- v0.2.027 (`e5217f8a`): raised the native DCNv4 mask capacity and runtime gate from 169 to 225 active points (15x15). Actual GPU composition parity compares outputs and all math gradients against the real CPU layer; the runtime matrix exercises two-step SGD/Adam, BN on/off, and includes the 15x15 case. The >225 fallback guard now uses 17x17. Evidence: `MEM=3G ./tools/test_host.sh dcnv4` passed 29 tests; `MEM=3G ./tools/test_oxide.sh --test oxide_dcnv4_chain` passed 3 tests (15x15 included); `MEM=3G ./tools/test_oxide.sh dcnv4_native_runtime_two_step_sgd_and_adam_match_cpu` passed 1 actual-GPU test (301.50 s). Larger kernels/unsupported activations and mixed networks remain host fallback; parameters remain host-canonical. No speedup, compiled-C++ golden or full-suite claim; CPU/axis/Muon and sparse-MoE requirements remain open.

## 65. Retain native DCNv4 Adam moments on device

- v0.2.028 (`d7b0e5cc`): eligible native DCNv4 Adam now reuses device-resident first/second moments between updates instead of downloading/re-uploading them every step. Parameters remain host-canonical and still cross at each optimizer step. `save_optimizer_state` materializes the device moments into the existing host Adam sidecar structure; loading an existing DNOPT sidecar uploads moments lazily on the next native update, with no format bump. Evidence: `MEM=3G ./tools/test_host.sh dcnv4` passed 29 tests, including device-state creation and byte-identical save/load/save sidecar round-trip. Focused actual-GPU two-step SGD/Adam CPU parity passed on rerun (1 test, 300.16 s); its preceding run had one near-threshold parameter-delta failure. No speedup or compiled-C++ golden claim; only DCNv4 Adam moments are resident, not its parameters. GPU Muon, broader GPU work, CPU/axis gaps and sparse MoE remain open.

## 66. Reuse DCNv4 projection weights from forward during backward

- v0.2.029 (`9b722d35`): each native DCNv4 forward cache now retains the exact uploaded offset-projection weight tensor; backward consumes that tensor rather than uploading the host copy again. This also ensures the adjoint uses the same parameter snapshot as forward if host parameters are changed between the two calls. Focused host DCNv4 tests passed (29), including an assertion that the forward cache holds the expected weights; actual-GPU two-step SGD/Adam CPU parity passed (1 test, 316.19 s). DCNv4 parameters remain host-canonical and are still uploaded for each new forward/optimizer update; this is per-forward lifetime reuse, not persistent network-wide residency or a speedup claim. No compiled-C++ golden; GPU Muon and broad parity gaps remain.

## 67. Retain native DCNv4 parameters across forwards and updates

- v0.2.030 (`6433f57f`): native f32 DCNv4 now caches learned parameter tensors (offset weights/biases, output biases and BN gamma) per layer and reuses them across forwards, backward, rolling-stat recomputation and native SGD/Adam updates. After an optimizer update, the device tensors are replaced in the cache and the existing host-canonical copies are refreshed; the next GPU call needs no parameter upload. Exact host value snapshots detect direct edits to public `params` and trigger a refresh; BN rolling statistics remain separately host-managed. Evidence: `MEM=3G ./tools/test_host.sh dcnv4` passed 29 tests, including measured upload reduction on repeated forward and invalidation after host edits. The actual-GPU two-step SGD/Adam CPU-parity test passed (1, 313.12 s). This retains parameter copies on device but still downloads updates to maintain host canonical state; no speedup or compiled-C++ golden claim. GPU Muon and broader parity requirements remain open.

## 68. Native GPU Muon orthogonalization primitive

- v0.2.031 (`b46ad9c8`): added a native Oxide f32 Newton–Schulz orthogonalization primitive exposed through `F32TensorOpsBackend::muon_orthogonalize_f32`; the generic default uses a host reference, while Oxide uses device Frobenius-norm accumulation, transpose/linear-combination kernels and existing GPU GEMM for five paper-coefficient iterations. Actual-GPU parity passed against the host reference for square, tall and wide matrices (including 17x31 and 31x17); zero and invalid-shape cases passed (`MEM=3G ./tools/test_oxide.sh --test oxide_muon`, 2 tests); host-reference zero/non-finite/transpose-equivariance tests also passed (`MEM=3G PKGS=darknet-cuda ./tools/test_host.sh muon_orthogonalize`, 2 tests). This is a standalone matrix primitive, **not** GPU Muon optimizer integration: `update_weights_gpu` still rejects `[net] muon=1`, and GPU momentum/update, hybrid AdamW fallback, runtime layer eligibility/gates and optimizer-sidecar state remain unimplemented. CPU Muon is still v0.1.005. No compiled-C++ golden or full-suite claim.

## 69. Connected GPU Muon and deconvolution GPU runtime slices

- v0.2.032 (`2ddcaa6b`): added opt-in `[net] muon=1` update dispatch for host-resident connected parameters (plus standalone BatchNorm AdamW-only state); eligible dense matrices call the native Oxide Newton–Schulz primitive, while momentum staging, final direction application, host-canonical parameters and optimizer state remain host-side. DNOPT6 save/load now carries GPU Muon momentum in the same host-only layout as CPU Muon. Unsupported parameter families reject before updates, and deferred host synchronization rejects explicitly. Evidence at the release: `MEM=3G ./tools/test_host.sh gpu_muon_` passed 2 tests and `muon_` passed 23; actual-GPU `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_connected_two_step_update_matches_cpu` passed (1), covering two CPU-parity steps plus sidecar save/resume and a third parity step. This is a limited connected-layer runtime slice, not full GPU Muon: other eligible layer families, device-resident momentum/final parameter updates, and broader gate/optimizer coverage remain open.
- v0.2.033 (`d73a87e7`): added standalone BatchNorm GPU-Muon-path parity for scale/bias AdamW fallback and optimizer state using FakeBackend. Follow-up `MEM=3G ./tools/test_host.sh gpu_muon_` passed 3 tests and `muon_` passed 24. This did not expand native GPU layer coverage.
- v0.2.034 (`f27b370c`): added native Oxide momentum/lookahead and decoupled direction-application ops, then wired connected-layer GPU Muon to keep momentum tensors cached on-device between updates and apply matrix decay/direction on-device. Host momentum mirrors are downloaded per update for DNOPT6 compatibility, and connected weights remain host-canonical (uploaded/downloaded at each step); biases/BN fallback remains host AdamW. Evidence at the release: host `darknet-cuda` Muon 4, `oxide_muon` 3, GPU-Muon host 3, Muon regressions 24, and actual-GPU single-connected two-step/resume parity 1.
- v0.2.035 (`cc0c00c1`): added actual-GPU CPU-parity coverage for two connected layers over two updates and FakeBackend `dont_update=1` gate behavior (parameters/state remain unchanged while the global step advances). Follow-up `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4 and `muon_` passed 25; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_multiple_connected_layers_match_cpu_across_steps` passed 1. Scope remains connected-only, with host-canonical parameters/state mirror; other eligible families and real-device gate coverage remained open at this release.
- v0.2.036 (`86b2fbda`): actual-GPU two-layer test now checks `dont_update=1` against CPU across two updates: the gated first layer stays unchanged initially, the ungated second layer updates, then parity holds when the first layer becomes eligible. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_connected_update_gate_matches_cpu_on_device` passed 1. Actual-device coverage for other gate variants/connected BN and other eligible families remains open.
- v0.2.037 (`281cdc7c`): extended GPU Muon eligibility to deconvolutional matrices, whose forward/backward gradients are host-fallback. The shared hybrid staging path runs Newton–Schulz through Oxide, while momentum, final parameter update, and optimizer state remain host-side. Two-step actual-GPU CPU parity passed with `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_deconvolution_host_gradients_match_cpu_across_steps`; `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4. Convolutional GPU Muon remains explicitly rejected because its gradients are device-resident; other families remain unsupported. This adds GPU Muon eligibility, not a native GPU deconvolution layer path.
- v0.2.038 (`fadfcb95`): added actual-GPU CPU-parity coverage for the connected-layer `burnin_update=1` gate. A first update within burn-in preserves parameters; after setting the iteration beyond burn-in, the Oxide Muon update changes parameters and matches CPU. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_connected_burnin_gate_matches_cpu_on_device` passed 1. Other gate variants/connected BN remain unverified; no compiled-C++ golden or full-suite claim.
- v0.2.039 (`b1fef3f9`): added host-reference and native Oxide f32 transposed-convolution forward, input-gradient, weight-gradient and bias-gradient operations, then integrated the backend operations in GPU runtime forward/backward. Supported activations take the GPU path; unmapped activations retain the legacy host route. `MEM=3G PKGS=darknet-cuda ./tools/test_host.sh deconvolutional` passed; `MEM=3G ./tools/test_host.sh deconvolutional_layer_trains_on_cpu_and_gpu_native_ops_identically` and `gpu_adam_updates_deconvolutional_host_parameters` each passed; native Oxide primitive tests compare outputs/gradients and invalid shapes against the host reference; actual-GPU `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_deconvolution_native_forward_backward_and_muon_match_cpu_across_steps` passed (1), including multi-step parity. Optimizer parameter/gradient staging remains host-side; this does not add f16 or compiled-C++ parity. No full-suite claim.
- v0.2.040 (`798d3389`): added an actual-Oxide runtime-chain test for native deconvolution forward/backward plus two-step CPU parity under both SGD and Adam. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_deconvolution_native_sgd_and_adam_updates_match_cpu_across_steps` passed (1); optimizer updates still stage through host buffers. This is focused GPU evidence, not full-suite or compiled-C++ parity.
- v0.2.041 (`4edd8860`): added actual-Oxide CPU-parity coverage for connected GPU Muon with batch normalization, including output, connected weights/biases, BN scales and rolling stats across two updates. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_connected_batch_norm_matches_cpu_across_steps` passed (1). This exercises the existing CPU-backward/host BN optimizer fallback; connected BN backward is not native. Other Muon families and gate variants remain open.
- v0.2.042 (`56bfbe2f`): added actual-device two-update CPU parity for connected GPU Muon with `train_only_bn=1`; CPU and Oxide both preserve parameters under the propagated gate. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_connected_train_only_bn_gate_matches_cpu_on_device` passed (1). Together with `dont_update` and `burnin_update`, all three layer update gates have connected actual-device coverage; other parameter families remain open.
- v0.2.043 (`506dc9bd`): extended the GPU Muon hybrid path to ViT matrices, using native Oxide Newton–Schulz orthogonalization with the existing host gradients, momentum/state, and final updates. Two-step actual-Oxide CPU parity covers all five eligible ViT matrices and the AdamW fallback parameters; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_vit_host_gradient_matrices_match_cpu_across_steps` and `MEM=3G ./tools/test_host.sh gpu_muon_` passed. This does not add native ViT layer forward/backward or device-resident Muon updates.
- v0.2.044 (`32a111b7`): added Swin matrices to the same hybrid GPU Muon path. Actual-Oxide two-step CPU parity covers QKV/attention/FFN matrices and AdamW-updated parameters; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_swin_host_gradient_matrices_match_cpu_across_steps` passed, and host `gpu_muon_` passed 4. Swin layer forward/backward and optimizer state/final updates remain host-side.
- v0.2.045 (`5bff2104`): added TuckerAttention matrices to hybrid GPU Muon eligibility. Actual-Oxide two-step CPU parity covers its packed factor weights and biases; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_tucker_host_gradient_matrices_match_cpu_across_steps` passed, and host `gpu_muon_` passed 4. TuckerAttention forward/backward and momentum/final updates remain host-side.
- v0.2.046 (`dd40ca39`): added MambaVision matrices to hybrid GPU Muon eligibility. Actual-Oxide two-step CPU parity covers its packed parameter vector; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_mambavision_host_gradient_matrices_match_cpu_across_steps` passed, and host `gpu_muon_` passed 4. MambaVision forward/backward and momentum/final updates remain host-side.
- v0.2.047 (`fb05dad0`): added KimiKda matrices to hybrid GPU Muon eligibility. Actual-Oxide two-step CPU parity covers its packed factor/FFN weights and biases; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_kimi_kda_host_gradient_matrices_match_cpu_across_steps` passed, and host `gpu_muon_` passed 4. KimiKda forward/backward and momentum/final updates remain host-side.

## 70. Native locally connected layer

- v0.2.048: added host-reference and native Oxide f32 locally connected `[local]` forward, input-gradient, weight-gradient and bias-gradient operations, with GPU runtime routing for supported activations. Oxide primitive parity (2 tests) and actual-GPU two-step SGD runtime CPU parity passed; `MEM=3G ./tools/test_host.sh local` passed 11. Fixed the GPU runtime CPU-fallback SGD update gap for local parameters. Extended hybrid GPU Muon to each location-specific local matrix; actual-GPU two-step CPU parity passed under Muon, and `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4. Local optimizer gradients/state/final updates remain host-staged; Muon orthogonalization is native Oxide. Unsupported activations retain the legacy host path. No compiled-C++ golden, full-suite, or performance claim.

## 71. GPU Muon RNN matrices

- v0.2.049: extended hybrid GPU Muon eligibility to RNN's input, recurrent, and output matrices, using the existing native-device RNN forward/backward and Oxide orthogonalization with host-staged gradients/state/final updates. Actual-GPU two-step CPU parity passed for outputs, all three weight groups, and biases (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_rnn_device_matrices_match_cpu_across_steps`); host `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4. No compiled-C++ golden, full-suite, or performance claim.

## 72. GPU Muon LSTM matrices

- v0.2.050: extended hybrid GPU Muon eligibility to LSTM's four input and four recurrent matrices, using native-device forward/backward and Oxide orthogonalization with host-staged gradients/state/final updates. Actual-GPU two-step CPU parity passed for outputs, all eight weight groups and biases (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- gpu_muon_lstm_device_matrices_match_cpu_across_steps`); host `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4. No compiled-C++ golden, full-suite, or performance claim.

## 73. Native GPU SAM forward and backward

- v0.2.051: replaced SAM's host-round-trip forward with native f32 pointwise multiply plus supported activation, and added the exact activation adjoint plus gradients to both the preceding layer and the configured source layer. Unsupported activations retain host forward and whole-network CPU backward fallback. Actual-Oxide two-step SGD parity passed for outputs and both upstream convolution parameter sets; coverage also checks `forward_training_device` and the fan-out source gradient (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- sam_native_gpu_forward_backward_fans_out_source_gradient_matches_cpu_across_steps`). The broad host `sam` filter passed 28 matched tests, though it is not a SAM-specific test. No compiled-C++ golden, full-suite, or performance claim.

## 74. Hybrid GPU Muon for convolution matrices

- v0.2.052: extended hybrid GPU Muon to standard convolutional matrices without batch normalization. The optimizer downloads accumulated device weight/bias gradients at the update boundary, runs Oxide Newton–Schulz orthogonalization, applies host-staged AdamW/Muon updates, refreshes cached device parameters, and clears consumed device gradients. Actual-Oxide grouped-convolution parity passed for two updates with two accumulated seeded backward calls per step, including outputs, weights, and biases (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_convolution_device_gradients_match_cpu_across_steps`); `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4, including the explicit BN-convolution rejection. BN convolutions and other convolutional families remain unsupported; gradients/state/final updates cross the host boundary. No compiled-C++ golden, full-suite, or performance claim.

## 75. Hybrid GPU Muon for non-BN EML branches

- v0.2.053: extended hybrid GPU Muon to both non-batch-normalized EML convolution branches. Accumulated device weight/bias gradients are downloaded at the optimizer boundary, each branch's matrices use Oxide Newton–Schulz, host AdamW/Muon updates apply, and updated parameters are re-uploaded to the appropriate device cache slots while gradients are cleared. Actual-Oxide CPU parity passed across two updates with two accumulated seeded backward calls per step, covering outputs, both branches' weights/biases, and a subsequent forward after cache refresh (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_eml_device_gradients_match_cpu_across_steps`). Host `MEM=3G ./tools/test_host.sh gpu_muon_` passed 4 and rejects BN EML/convolution before mutation. BN EML remains unsupported; gradients/state/final updates cross the host boundary. No compiled-C++ golden, full-suite, or performance claim.

## 76. Hybrid GPU Muon for non-BN MoE

- v0.2.054: extended hybrid GPU Muon to the MoE router matrix and all expert convolution matrices when expert batch normalization is disabled. Accumulated router/expert gradients are downloaded from device at the update boundary, Oxide performs Newton–Schulz orthogonalization, host AdamW/Muon updates apply, expert parameters are re-uploaded to their device cache slots, and device accumulators are cleared. Actual-Oxide CPU parity passed over two updates with two accumulated seeded backward calls per step, covering outputs, router weights/biases and every expert's weights/biases (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_moe_device_gradients_match_cpu_across_steps`). BN MoE remains explicitly rejected by the host regression test. Gradients/state/final updates cross the host boundary. No compiled-C++ golden, full-suite, or performance claim.

## 77. Hybrid GPU Muon for non-BN DCNv4 offset projection

- v0.2.055: extended hybrid GPU Muon to the DCNv4 offset-projection matrix when batch normalization is disabled. The existing per-layer device-gradient flush is now targetable so update-gated layers retain untouched device buffers; eligible accumulated DCNv4 gradients are staged to host, Muon orthogonalizes the offset matrix with the existing 0.01 learning-rate scale, remaining DCNv4 parameters receive the existing AdamW fallback, and updated host parameters are detected/re-uploaded by the DCNv4 snapshot cache on the next forward. Actual-Oxide CPU parity passed across two updates with two accumulated backward calls per step, covering outputs, offset weights/biases and output biases (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_dcnv4_offset_projection_matches_cpu_across_steps`). BN DCNv4 remains explicitly rejected before mutation. This is hybrid: gradients, state and final updates cross the host boundary; no compiled-C++ golden, full-suite or performance claim.

## 78. Hybrid GPU Muon for non-BN deform_conv

- v0.2.056: enabled GPU Muon for supported non-batch-normalized `[deform_conv]` layers. Its native Oxide forward/backward computes parameter gradients; backward currently downloads those gradients into host train buffers, where Muon Newton–Schulz and the existing AdamW fallback/state/update apply. The hybrid path covers the main convolution, offset predictor, and optional mask predictor matrices while biases retain AdamW. Actual-Oxide CPU parity passed across two updates with two accumulated seeded backward calls per step, including a preceding native convolution (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_deform_conv_native_gradients_match_cpu_across_steps`). BN deform_conv remains explicitly rejected before mutation. No compiled-C++ golden, full-suite, or performance claim.

## 79. Native Oxide GraphConv uniform aggregation primitives

- v0.2.057: added checked host-reference and native Oxide f32 forward/adjoint primitives for `[graph_conv]` edge mode 0's uniform valid-neighbor spatial average. NCHW, odd-kernel shape validation covers asymmetric geometry, stride, dilation and padding; backward gathers into unique input-gradient elements without atomics. Actual-device tests pass against the host reference plus finite differences (`MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv`, 2 tests); the existing GraphConv CPU host filter passes 15 tests (`MEM=3G ./tools/test_host.sh graph_conv`). **This does not wire `[graph_conv]` into GPU runtime forward/backward**: learned-edge softmax, self projection, BN, runtime caches, projection epilogue/gradients and optimizer integration remain open. Hand-derived from Rust CPU reference/source, no compiled-C++ golden, no full-suite or performance claim.

## 80. GraphConv aggregation coordinate-overflow guards

- v0.2.058: strengthened the new GraphConv aggregation geometry validator to reject i64 coordinate arithmetic overflow before kernel launch, including large padding/stride combinations whose tensors appear shape-consistent. The actual-Oxide suite now has 3 passing tests, adding explicit overflow rejection to host-reference parity, finite-difference and malformed-output checks (`MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv`). This hardens only the aggregation primitive; GraphConv runtime remains unwired. No compiled-C++ golden, full-suite or performance claim.

## 81. Native GPU GraphConv uniform runtime path

- v0.2.059: wired native Oxide f32 forward/backward for `[graph_conv]` edge mode 0 with no self projection or BN and backend-supported activation. Aggregation and grouped 1x1 projection/bias/activation run on device; backward computes activation, projection parameter, and aggregation-adjoint gradients on device, then stages projection gradients to host train buffers for SGD/Adam updates. Actual-Oxide runtime parity passed for layer outputs and SGD/Adam parameter updates through GraphConv and an upstream convolution (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_uniform_runtime_forward_backward_and_sgd_match_cpu`); host `graph_conv` filter passed 15 tests including host fallback. Unsupported learned-edge/self/BN/activation variants retain CPU fallback; compact GPU inference remains unsupported. Parameters are uploaded for projection and gradients download at backward; no residency/speedup claim. No compiled-C++ golden or full-suite claim. A prior broad runtime-chain run reported 50 passes and one failure because the test passed a GPU host snapshot (which intentionally omits device caches) to CPU backward; the test now pairs each forward state with its matching backward path, and `dcnv4_mixed_network_cpu_and_gpu_backward_succeed` passes.

## 82. GraphConv learned-edge masked-softmax host contract

- v0.2.060: added checked host references for GraphConv learned-edge mode 1 aggregation and its exact input/edge-kernel/edge-bias adjoint. The contract fixes group-local `[groups, size*size, 2*channels_per_group]` layout, row-major edge order, center-reference zeroing when out of bounds, and masking invalid neighbors before stable softmax. Host-reference finite differences cover every input/kernel/bias element; constant-input masking and malformed group/length rejection pass in `MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv` (4 tests total). This is a contract/reference increment only: **no native learned-edge GPU kernel or runtime integration exists yet**; CPU reference source and hand-derived equations only, no compiled-C++ golden. `PKGS=darknet-cuda MEM=3G ./tools/test_host.sh learned_aggregate` could not run because an unrelated existing unit-test compile error in `host_ops/local.rs` prevents the `darknet-cuda` library test target from compiling; the contract is exercised through the Oxide integration test target instead.

## 83. Native GraphConv learned-edge forward primitive

- v0.2.061: added a native Oxide learned-edge mode 1 forward kernel and backend operation for group-local row-major edge parameters. The kernel computes valid-edge logits, masks invalid spatial neighbors before max-stabilized softmax, and aggregates NCHW values; the backend validates geometry, group divisibility, and parameter extents before launch. Actual-Oxide parity against the checked host reference, including boundary sites, plus malformed group/parameter rejection passes in `MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv` (4 tests total); host-reference input/kernel/bias finite differences also remain green. This is **primitive-only**: no native backward or runtime dispatch yet, and no compiled-C++ golden/full suite/performance claim. The trait's default implementation remains host reference for other backends.

## 84. Native GraphConv learned-edge adjoints

- v0.2.062: added actual-Oxide learned-edge backward primitives for input, edge-kernel, and edge-bias gradients. Device scratch stores normalized edge weights and masked-softmax logit adjoints; gather/reduction kernels avoid write races. Oxide outputs for all three gradients match the checked host adjoint, whose every input/kernel/bias element is finite-difference checked. `MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv` passes 4 tests. No GraphConv runtime dispatch yet; this is primitive-only and has no compiled-C++ golden/full-suite/performance claim.

## 85. GraphConv learned-edge runtime path

- v0.2.063: routed learned-edge mode 1 GraphConv through native Oxide aggregation forward/backward for no-self/no-BN supported-activation configurations. Learned edge-kernel/bias gradients are downloaded into existing host optimizer buffers; projection and edge parameters continue to update host-side (SGD/Adam). Actual-Oxide runtime output and updates match CPU across SGD then Adam in `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_learned_runtime_forward_backward_and_updates_match_cpu` (1 test). Unsupported edge modes, self projection, BN, and unsupported activations keep host fallback; compact inference and compiled-C++ golden validation remain open. No full-suite or performance claim.

## 86. GraphConv center-reference gather primitive

- v0.2.064: added checked host and native Oxide center/reference gather and exact input adjoint. Odd-kernel center coordinates use the same stride/padding/dilation mapping as GraphConv; out-of-bounds references are zero, and the adjoint gathers the unique mapped output without atomics. Actual-Oxide forward/backward parity, all-input finite differences, and padded/strided boundary coverage pass in `MEM=3G ./tools/test_oxide.sh --test oxide_graph_conv` (5 tests total). This prepares—but does not yet implement—self-projection runtime support. No compiled-C++ golden/full-suite/performance claim.

## 87. GraphConv self-projection runtime path

- v0.2.065: integrated the native center-reference branch into GraphConv no-BN runtime. Self projection now runs as a grouped pointwise convolution added to the neighbor projection; backward computes its weight gradient, gathers its input adjoint through the reference primitive, and accumulates self-weight updates into existing host optimizer buffers. Actual-Oxide runtime forward/backward and parameter parity against CPU pass across SGD and Adam in `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_self_projection_runtime_forward_backward_and_sgd_adam_match_cpu` (1 test). BN, unsupported activations/edge modes, compact inference, and compiled-C++ golden remain open; optimizer gradients/updates are host-staged. No full-suite/performance claim.

## 88. Compact GraphConv inference

- v0.2.066: enabled the compact f32 GPU inference path for GraphConv configurations already supported by the native runtime (edge modes 0/1, optional self projection, no BN, supported activation). The compact layer arm composes native aggregation/reference gather, grouped pointwise projections, bias, and activation without a backward cache. Actual-Oxide compact output matches CPU for learned edges plus self projection in `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_compact_inference_matches_cpu_with_learned_edges_and_self` (1 test). Unsupported variants retain an explicit preflight rejection; no C++ golden/full-suite/performance claim.

## 89. Compact GraphConv batch-normalization inference

- v0.2.067: extended compact f32 GraphConv inference to apply batch normalization using rolling mean/variance, gamma, and projection beta. The compact-only eligibility check now permits BN while the training/full-runtime predicate still preserves the existing BN fallback. Actual-Oxide CPU parity passes for learned edges plus self projection and BN in `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_compact_inference_matches_cpu` (2 tests, BN and non-BN). Training BN and compiled-C++ golden remain open; no full-suite/performance claim.

## 90. Full GraphConv batch-normalization inference

- v0.2.068: extended the ordinary f32 GPU inference runtime to execute supported GraphConv variants with BN natively, using rolling statistics, gamma and projection beta; compact inference support from v0.2.067 remains. Training BN and unsupported activation/edge variants retain CPU fallback. A FakeBackend test confirms this inference path avoids constructing the CPU GraphConv cache; actual-Oxide full and compact outputs match CPU in `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv_compact_inference_matches_cpu` (2 tests). The focused `MEM=3G ./tools/test_host.sh graph_conv_batchnorm_inference_uses_gpu_backend_without_cpu_cache` passed 1 core test. Hand-derived from the Rust CPU contract; no compiled-C++ golden, full-suite, or performance claim.

## 91. Hybrid GPU Muon for non-BN GraphConv matrices

- v0.2.069 (`d79c3aaf`): enabled GPU Muon for GraphConv projection, self-projection, and learned-edge kernel matrices when BN is disabled. Native Oxide Newton–Schulz orthogonalization is used; GraphConv forward/backward gradients, Muon state, and final parameter updates remain host-staged. Actual-Oxide CPU parity passed over two updates for learned edges plus self projection, comparing projection weights/biases, self weights, edge kernels, and edge biases: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_matrices_match_cpu_across_steps`. A host FakeBackend regression confirms batch-normalized GraphConv rejects before mutation: `MEM=3G ./tools/test_host.sh gpu_muon_rejects_batch_normalized_convolution_eml_moe_dcnv4_and_deform_before_updating` passed 1 core test. BN GraphConv remains unsupported; no compiled-C++ golden, full-suite, or performance claim.

## 92. GraphConv GPU Muon DNOPT6 sidecar resume

- v0.2.070 (`3886b6c2`): extended the GraphConv hybrid GPU Muon parity test through optimizer-state save/load and a resumed third update. CPU and GPU DNOPT6 sidecars are loaded into fresh executables; resumed output and projection/self/edge matrix plus bias updates match CPU. Verified by `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_matrices_match_cpu_across_steps` (1 actual-Oxide test). This verifies the existing sidecar format for the tested learned-edge+self, non-BN variant; other GraphConv variants and full GPU/CPU parity requirements remain open. No compiled-C++ golden, full-suite, or performance claim.

## 93. GraphConv GPU Muon update gate

- v0.2.071 (`9aaa30fb`): added actual-Oxide CPU parity for GraphConv Muon with `dont_update=1`. The test runs forward/backward, confirms CPU and Oxide outputs agree, verifies all GraphConv parameters remain unchanged after the gated optimizer call, and confirms the global optimizer step advances. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_dont_update_gate_preserves_parameters` passed 1 test. This covers only the no-BN learned-edge+self variant; other per-layer gates/configurations remain unverified. No compiled-C++ golden/full-suite/performance claim.

## 94. Grouped GraphConv GPU Muon parity

- v0.2.072 (`543d57e0`): strengthened actual-Oxide GraphConv Muon coverage to `groups=2` for learned-edge+self with batch normalization disabled. Two optimizer updates, DNOPT6 save/load, and a resumed third update match CPU for projection/self/edge matrices and biases. The focused test passes: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_matrices_match_cpu_across_steps` (1 test). This pins tested grouped parameter slicing and sidecar continuation, not all GraphConv variants; no compiled-C++ golden/full-suite/performance claim.

## 95. GraphConv BN GPU Muon through CPU fallback

- v0.2.073 (`007d6eed`): enabled GPU Muon updates for batch-normalized GraphConv by using the existing whole-network CPU forward/backward fallback for BN variants, while Oxide supplies matrix orthogonalization. Actual-device two-step parity covers grouped learned edges+self, projection/self/edge matrices and biases, BN gamma/beta, and rolling mean/variance: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_graph_conv_batch_norm_cpu_fallback_matches_cpu` passed 1 test. This is optimizer eligibility over the CPU training fallback—not native GraphConv BN training. Native BN training, other activation/edge/group variants, and compiled-C++ evidence remain open; no full-suite/performance claim.

## 96. Native GraphConv BN training

- v0.2.074: native f32 GraphConv training now supports batch-statistics BN with the eligible edge modes 0/1 and supported activations. GPU forward caches BN statistics; native backward computes input, gamma, beta, projection, self, and learned-edge gradients. Rolling mean/variance return from the device after backward using the CPU GraphConv reference's 0.9 update coefficient. Oxide actual-device parity passed for grouped learned-edge+self across SGD and Adam, including output, trainable parameters and rolling statistics (`graph_conv_batch_norm_native_training_forward_backward_sgd_adam_matches_cpu`); two-step GPU Muon parity also passes (`gpu_muon_graph_conv_batch_norm_native_training_matches_cpu`). `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain graph_conv` passed 9 tests; focused host BN inference and Muon rejection tests passed. Unsupported activations/edge modes retain fallback; other configurations, compiled-C++ golden, full-suite and performance evidence remain open.

## 97. Standalone BatchNorm native GPU training

- v0.2.075: standalone `[batchnorm]` training now uses native Oxide batch-statistics forward/backward, caches batch statistics, and synchronizes rolling mean plus the CPU-reference biased rolling variance after backward. Scale/bias optimizer application remains the existing host AdamW fallback. Actual-Oxide output and parameter/stat parity against CPU passed in `standalone_batch_norm_native_training_matches_cpu_adam`; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain batch_norm` passed 9 tests, including recurrent/connected/GraphConv BN regressions. This does not add compiled-C++ golden evidence, native standalone-BN optimizer updates, or full-suite/performance claims.

## 98. Native Crop training geometry

- v0.2.076 (`c815a081`): native Crop training supports random crop coordinates, optional horizontal flip, rotation, and the per-in-bounds-tap affine scaling with zero-valued out-of-bounds taps. Training HSV/saturation/exposure/RGB-shift augmentation remains explicitly rejected. Actual-Oxide runtime parity passes for an augmented Conv→Crop chain over SGD and Adam, comparing crop outputs and upstream Conv updates: `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain crop_native_training_augmentation_and_backward_match_cpu` passed. Crop bilinear fwd/bwd primitive parity and FD pass (`oxide_pool_parity crop_bilinear_forward_backward_and_fd`), and host crop filter passed 10 tests. This is Rust CPU/Oxide parity, not compiled-C++ parity; no full-suite/performance claim.

## 99. DCNv4 mixed-network backward validation

- v0.2.077 (`e583f9d2`): corrected a stale mixed-network regression test that passed the GPU forward's host snapshot to CPU backward, although GPU backward caches intentionally live in `GpuForwardState`. The test now pairs the CPU forward with CPU backward and GPU forward with native GPU backward. Its focused test passes; the complete `oxide_runtime_chain` binary passes 62/62 tests (`MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain`, 1273.53 seconds). This is scoped Oxide runtime-chain evidence only—not the entire Rust suite, compiled-C++ golden, or overall objective completion.

## 100. Explicit GPU rejection for GRU training

- v0.2.078 (`57b00348`): added an explicit `UnsupportedLayer` GPU error for `[gru]`, replacing the misleading catch-all parameter/config mismatch. The focused Oxide training-rejection test passes. Native inference was added separately in §102; GPU training/backward remains unsupported.

## 101. Region CPU forward and axis pin

- v0.1.027: hand-ported the CPU portion of `Jev_laya:src-lib/region_layer.cpp:209-235`: C++ `flatten` layout (`blas.cpp:104`), objectness logistic, optional class softmax, and the exact supplied-seed adjoint. `[region] softmax` is parsed and input channels are validated against `num * (coords + classes + 1)`. Named non-square, batch-2 axis test `region_axis_layout_matches_cpp_flatten_objectness_and_classes` checks outputs and adjoints. `MEM=3G ./tools/test_host.sh axis_layout` passed 22 tests; `every_layer_kind_has_an_axis_pin_decision` passed; all 12 cfgs containing `[region]` passed isolated parse/construct/CPU-inference runs. GPU full/compact inference now rejects explicitly; Oxide rejection test passes. Truth-driven Region loss/decode, GPU kernels, and compiled-C++ golden remain open; source-derived, no compiled-C++ parity claim.

## 102. Native GRU GPU inference

- v0.2.079: added a native f32 GPU inference composition for `[gru]` using device connected projections, logistic gates, pointwise reset/update equations, sequence slicing and concatenation. Rolling-stat BN is folded into each projection's weights and bias; training remains explicitly rejected because no native backward/cache/update path exists. The actual Oxide tests `gru_native_inference_matches_cpu_for_multiple_steps` and `gru_native_inference_matches_cpu_with_rolling_batch_norm` compare outputs against Rust CPU, and `gru_gpu_training_path_is_rejected_as_unsupported_not_misconfigured` verifies the boundary. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gru` passed 3 tests; host axis tests passed 23. No C++ golden/parity, training parity, full-suite or performance claim.

## 103. Native RNN rolling-stat BN inference

- v0.2.080: native f32 RNN inference now dispatches through device recurrent projections even when batch normalization is enabled. Rolling-stat BN affine parameters are folded into each projection's weights/biases before upload, matching CPU inference. BN training continues on the existing host fallback because its batch-stat/cache semantics are separate. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain rnn` passed 6 tests, including `rnn_native_inference_matches_cpu_with_rolling_batch_norm` plus RNN/CRNN training regressions. Rust CPU reference parity only; no C++ golden, full-suite, or performance claim.

## 104. Native LSTM rolling-stat BN inference

- v0.2.081: native f32 LSTM inference now dispatches through device recurrent projections with rolling-stat BN folded into the four recurrent and four input projection weight/bias pairs. BN training remains on the host fallback. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain lstm` passed 4 tests, including `lstm_native_inference_matches_cpu_with_rolling_batch_norm` and existing LSTM forward/backward/Muon regressions. Rust CPU-reference parity only; no C++ golden, full-suite or performance claim.

## 105. Native RNN rolling-stat BN training

- v0.2.082: extended the device RNN forward to fold each rolling-stat BN affine into its projection during training as well as inference. The existing GPU recurrent backward path already handles rolling-stat BN affine/scale gradients, so BN RNN no longer needs host forward/backward fallback for supported activations. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain rnn` passed 6 tests, including `batch_normalized_rnn_device_backward_matches_cpu_on_oxide` (forward outputs, backward gradients, and parameter updates vs CPU) and existing non-BN RNN/CRNN regressions. Unsupported activations remain host fallback; no C++ golden/full-suite/performance claim.

## 106. Native LSTM rolling-stat BN training

- v0.2.083: routed all LSTM inference and training forwards through native recurrent projections, folding rolling-stat BN affines into each projection. The existing LSTM device backward computes rolling-stat BN gradients, so BN training no longer needs the whole-layer CPU forward/backward fallback. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain lstm` passed 4 tests, including `batch_normalized_lstm_device_backward_matches_cpu_on_oxide` (output and parameter updates vs CPU) and existing non-BN backward/Muon regressions. No C++ golden, full-suite or performance claim.

## 107. Native CRNN rolling-stat BN training

- v0.2.084: enabled native CRNN recurrent-convolution forward with rolling-stat BN affines folded into the three convolution projections. The existing device CRNN backward path handles BN input/beta/scale adjoints, so the training forward no longer falls back to CPU for supported activations. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain batch_normalized_crnn_device_backward_matches_cpu_on_oxide` passed on Oxide; the FakeBackend `gpu_batch_normalized_crnn_backward_matches_cpu_and_updates_rolling_stats` host test also passed. Coverage includes multi-step output, gradients, parameter updates, rolling stats, upstream convolution and finite-gradient checks. No C++ golden, full-suite or performance claim.

## 108. Hybrid GPU Muon for CRNN matrices

- v0.2.085: enabled CRNN convolution projection matrices in GPU Muon eligibility. Oxide performs the Newton–Schulz orthogonalization; gradients and optimizer state/final updates remain host-managed. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_muon_crnn_device_matrices_match_cpu_across_steps` passed two actual-Oxide updates against CPU Muon with BN enabled, checking forward outputs, all three projection weights/biases, and BN scales. No C++ golden, full-suite or performance claim.

## 109. Native GPU `maxpool_depth`

- v0.2.086: added native Oxide f32 forward and deterministic backward for depth pooling, preserving the Rust/C++-derived strided-channel grouping and first-wins ties. Full GPU runtime dispatch now supports it in training/inference; f16 remains explicitly unsupported. `MEM=3G ./tools/test_host.sh maxpool_depth` passed 2 CPU/config tests, and `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain maxpool_depth` passed 2 actual-Oxide runtime tests (native forward/backward parity plus ordinary maxpool regression). No compiled-C++ golden, full-suite or performance claim.

## 110. MaxPool axis/layout contract pin

- v0.1.028: pinned the MaxPool axis contract to a named non-square, batch-2 hand-computed test covering depth-channel groups, first-wins ties, NCHW batch/channel/spatial order, spatial scan/stride order and `maxpool_zero_nonmax` backward. `MEM=3G ./tools/test_host.sh axis_layout` passed 24 tests; `MEM=3G ./tools/test_host.sh every_layer_kind_has_an_axis_pin_decision` passed. The axis ledger now records 21 pinned variants / 39 open C++ variants / 5 Rust-only extensions. Hand-derived from `Jev_laya:src-lib/maxpool_layer.cpp`; no compiled-C++ golden.

## 111. AvgPool axis/layout contract pin

- v0.1.029: pinned the global AvgPool reduction with a hand-computed non-square B=2/C=2 fixture and supplied-seed backward. The named test verifies `[B,C]` output order, per-sample/per-channel `H*W` means, and uniform per-plane gradients; config shape and the axis registry are checked. `MEM=3G ./tools/test_host.sh axis_layout` passed 25; the standalone test and registry test passed. Hand-derived from `Jev_laya:src-lib/avgpool_layer.cpp:51`; no compiled-C++ golden.

## 112. LocalAvgPool axis/layout contract pin

- v0.1.030: pinned local-average pooling with a hand-computed non-square B=2/C=2 fixture covering odd window size, distinct x/y strides, partial edge windows from padding, NCHW output order, and the exact overlap-summed seeded backward. The named config also checks output geometry. `MEM=3G ./tools/test_host.sh axis_layout` passed 26; `MEM=3G ./tools/test_host.sh every_layer_kind_has_an_axis_pin_decision` passed, and the focused test passed. Hand-derived from `Jev_laya:src-lib/maxpool_layer.cpp:440`; no compiled-C++ golden.

## 113. Native f16 `maxpool_depth` inference

- v0.2.087: extended compact Oxide f16 inference for `[maxpool] maxpool_depth=1` with a native half-storage kernel. The channel-group scan widens for comparisons, retains the first channel on ties, and stores the selected half bits without an unnecessary round trip. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed 7 tests; the new depth case matches compact f32 within the suite's 4e-3 tolerance (measured relative max error 2.488e-4). The f32 depth path and backend shape guards remain intact. No f16 training/backward or C++ golden claim.

## 114. Connected axis/layout contract pin

- v0.1.031: added a named non-square B=2 Connected test with C=2, H=2, W=3 and hand-set `[out][in]` weights. Hand-computed forward sentinels pin flattened NCHW features and batch-major output rows; a supplied seed pins the exact input adjoint per independent sample. Updated the exhaustive axis registry/ledger to 24 pinned and 36 open C++-backed variants. `MEM=3G ./tools/test_host.sh axis_layout` passed 27 tests; `MEM=3G ./tools/test_host.sh every_layer_kind_has_an_axis_pin_decision` passed. Hand-derived from `Jev_laya:src-lib/connected_layer.cpp:186`; no compiled-C++ golden.

## 115. Native f16 SAM inference

- v0.2.088: added a native Oxide f16 SAM product/activation kernel and compact-runtime dispatch for supported activations. Fixed compact GPU source-lifetime accounting to retain/release SAM's explicit source reference as well as its previous-layer input. An actual-Oxide two-convolution + non-adjacent-source SAM case matches f32 compact inference (relative max error 6.223e-4); `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed all 8 tests. Inference only; SAM training remains the f32 path. No compiled-C++ golden.

## 116. Native f16 channel-shuffle inference

- v0.2.089: added a native Oxide half-storage channel permutation and compact f16 runtime dispatch for `[channel_shuffle]`. Actual-Oxide batch-2/channel-6 output matches f32 compact inference with relative max error 2.437e-4; the full `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` suite passed 9 tests. This is inference only; no compiled-C++ golden.

## 117. Native f16 reorg inference

- v0.2.090: added native Oxide f16 forward and reverse reorg permutations, reusing the checked shared reorg geometry and preserving zero-filled uncovered borders on reverse. Compact f16 dispatch now accepts Reorg. Actual-Oxide batch-2 non-square odd-size forward+reverse case matched f32 compact inference with relative max error 4.646e-4; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed 10 tests before the fixture was refined, and the refined focused test passed. Inference only; no compiled-C++ golden.

## 118. Native f32 Region GPU execution

- v0.2.091: replaced Region's full/compact GPU hard rejection with native Oxide f32 layout/activation forward and exact Rust-forward input adjoint. Forward transposes channel-major NCHW values to spatial/anchor/entry rows, applies objectness sigmoid and optional class softmax; backward differentiates those transforms and returns NCHW gradients. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain region_native_gpu_forward_backward_matches_cpu` passed actual-Oxide CPU-reference comparisons for full training forward, compact inference and supplied-seed gradients with batch 2, non-square geometry, softmax both off/on. `MEM=3G ./tools/test_host.sh region` passed 2 tests including the hand-derived axis pin. Region truth-driven loss/decode remain absent. C++ GPU backward passes deltas through, unlike this exact derivative of the Rust forward; no compiled-C++ golden or parity claim.

## 119. Native f16 Region inference

- v0.2.092: added compact Oxide f16 Region inference with a half-storage channel-layout transpose, f32 objectness sigmoid/optional class softmax, and half stores. Both softmax-disabled/enabled batch-2 cases matched compact f32 (relative max errors 2.324e-4 and 2.856e-4); the full `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` suite passed 11 tests. f16 backward/training and Region truth-loss/decode remain unsupported; no compiled-C++ golden.

## 120. Native f16 softmax inference

- v0.2.093: added native Oxide compact f16 softmax, with per-sample equal contiguous groups, f32 max/exp/sum/normalization and one half rounding per result. Actual-Oxide group-count 1 and 2 outputs matched compact f32 (relative max errors 4.106e-4 and 5.549e-4); `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed 12 tests. Inference only; no f16 backward/training or compiled-C++ golden.

## 121. Native f16 channel-slice inference

- v0.2.094: compact f16 dispatch now handles `[channel_slice]` by selecting a source tensor and reusing the native half-storage route-concat copy. Actual-Oxide batch-2 output matched f32 compact with relative max error 2.353e-4; `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed 13 tests. Inference only; no compiled-C++ golden.

## 122. Native f16 scale-channels inference

- v0.2.095: added compact f16 ScaleChannels for both per-channel (global-average source) and per-spatial scale modes, with source activation and source-lifetime accounting. Actual-Oxide batch-2 outputs matched f32 (relative max errors 6.633e-4 and 1.701e-3); `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed 14 tests. Inference only; no compiled-C++ golden.

## 123. Native f16 dropout inference

- v0.2.096: compact f16 inference now accepts Dropout as a device-resident identity, matching inference semantics without RNG or host round-trip. Actual-Oxide output matched compact f32 (relative max error 3.890e-4); `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed 15 tests. Training dropout remains on existing f32 path; no compiled-C++ golden.

## 124. Native f16 channel-normalization inference

- v0.2.097: compact f16 convolution now supports all three convolution-only `NORM_CHAN*` activations through a native half-storage kernel with f32 reductions and half output stores. Actual-Oxide batch-2 outputs matched f32 (relative max errors 4.429e-4, 4.639e-4, 4.418e-4); `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16` passed 15 tests. Inference only; no compiled-C++ golden.

## 125. Native f16 connected inference

- v0.2.098: compact f16 inference now runs Connected through the existing f16 GEMM convolution epilogue after per-sample NCHW flattening. Supports folded rolling-stat BN and mapped scalar activations. Actual-Oxide batch-2 flattened inputs matched f32 for non-BN/BN (relative max errors 2.467e-4/3.655e-4); the full f16 suite passed 17 tests. Inference only; no compiled-C++ golden.

## 126. Native f16 Cost identity and full ResNet101 coverage

- v0.2.099: compact f16 now accepts `[cost]` as an inference identity, allowing the existing ResNet101 f16 parity fixture to run through its final Connected classifier and Cost section instead of stopping at avgpool. The full `real_cfgs_f16_matches_f32` test passed; final ResNet101 relative max error was 9.775e-4. The complete Oxide f16 suite passed 17 tests. Rust compact CPU/f32 references only; no compiled-C++ golden.

## 127. Native f16 Detection and Regression identity paths

- v0.2.100: compact f16 inference now retains `[detection]` and `[regression]` as device-resident identity layers, matching the current CPU/f32 forward contract. Actual-Oxide batch-2 chain parity passed against f32 compact (relative max error 2.152e-4); full Oxide f16 suite passed 18 tests. No compiled-C++ golden.

## 128. Native connected-layer BN GPU backward

- v0.2.101 (`c2c3acd9`): connected layers with BN no longer force whole-network CPU backward when the activation is supported. Oxide computes the activation/BN adjoint, raw connected input/weight gradients, and BN scale gradients; rolling statistics are updated from the pre-update raw output using the CPU rolling coefficient. BN scale/bias optimizer accumulation and rolling-stat synchronization remain host-managed. Actual-Oxide parity across two Muon updates passed in `gpu_muon_connected_batch_norm_matches_cpu_across_steps`; all five `gpu_muon_connected` Oxide tests and the focused host test passed. Rust CPU-reference parity only; no compiled-C++ golden or speedup claim.

## 129. Connected BN GPU backward activation parity

- v0.2.102 (`93c56f17`): expanded the actual-Oxide two-step connected-BN GPU Muon parity fixture to cover both `linear` and nonlinear `leaky` activation, including outputs, weights, biases, scales and rolling stats. The focused test and all five `gpu_muon_connected` runtime-chain tests passed. Rust CPU-reference parity; no compiled-C++ golden.

## 130. Connected BN GPU backward SGD/Adam parity

- v0.2.103 (`97374ebc`): added actual-Oxide two-step Connected BN backward and optimizer parity for both SGD and Adam with `leaky` activation. Outputs, weights, biases, BN scales, rolling means and rolling variances match the CPU reference. `MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain gpu_connected_bn_backward_matches_cpu_sgd_and_adam` passed. This verifies Rust CPU-reference parity only; no compiled-C++ golden.

## 131. Route source/group axis pin

- v0.1.032 (`6a98ce9b`): hand-derived from `Jev_laya:src-lib/route_layer.cpp:91-139`; named test `route_axis_layout_pins_named_source_order_group_slice_and_delta_scatter` uses batch-2 non-square NCHW sentinel tensors to pin relative named-source order, group selection, channel concatenation and backward delta scatter. `MEM=3G ./tools/test_host.sh route_axis_layout` passed; full `MEM=3G ./tools/test_host.sh axis_layout` passed 28 tests (945 filtered). No compiled-C++ golden.

## 132. Upsample axis pin and ragged reverse extent

- v0.1.033 (`f20bce66`): hand-derived from `Jev_laya:src-lib/upsample_layer.cpp:6-28` and `blas.cpp:669-695`. The named B=2 non-square axis test pins forward repeat/scale, reverse stride-block summation, adjoints, NCHW axes, integer-floor output extent and ignored ragged trailing cells. It exposed a CPU reverse-mode panic on non-divisible dimensions; forward and adjoint now process only the complete stride-covered region, leaving the ragged tail zero in the adjoint. `MEM=3G ./tools/test_host.sh upsample` passed 6 tests; `MEM=3G ./tools/test_host.sh axis_layout` passed 29. Hand-derived, no compiled-C++ golden.

## 133. Reorg forward axis pin and reverse-source finding

- v0.1.034 (`4842e926`): hand-derived the normal `[reorg]` forward/adjoint from `Jev_laya:src-lib/reorg_layer.cpp:112-128` and `blas.cpp:69-100`. The batch-2 NCHW fixture pins offset-channel packing and both spatial offsets with explicit sentinels; `MEM=3G ./tools/test_host.sh reorg_axis_layout` passed and `MEM=3G ./tools/test_host.sh axis_layout` passed 30 tests. C++ reverse-mode is not pinned: source inspection found its `out_c`/`in_c` divisor defect (documented as axis ledger F2). No compiled-C++ golden.

## 134. ChannelShuffle axis pin

- v0.1.035 (`be9738c2`): hand-derived from `Jev_laya:src-lib/channel_shuffle_layer.cpp:11-20,95-113`; named batch-2 non-square NCHW sentinel test pins the group-row/group-column transpose and exact inverse adjoint across spatial planes. `MEM=3G ./tools/test_host.sh shuffle_axis_layout` passed; full `MEM=3G ./tools/test_host.sh axis_layout` passed 31 tests. No compiled-C++ golden.

## 135. Softmax grouped-reduction axis pin

- v0.1.036 (`8a75e857`): hand-derived from `Jev_laya:src-lib/softmax_layer.cpp:75-85` and `blas.cpp:655-665`. The named batch-2 non-square test pins contiguous per-sample flat groups (not a whole-sample or per-channel reduction), stable probabilities, and the matching grouped adjoint. `MEM=3G ./tools/test_host.sh softmax` passed 12; `MEM=3G ./tools/test_host.sh axis_layout` passed 32. No compiled-C++ golden.

## 136. Dropout element-mask layout evidence and DropBlock finding

- v0.1.037 (`5b9006b5`): added a focused batch-2 non-square NCHW test for standard Dropout's per-element mask, inference identity and same-mask adjoint; `MEM=3G ./tools/test_host.sh dropout_element_mask_and_adjoint_preserve_nchw_slots` passed. This is partial axis evidence, not a closed pin. `axis-coverage.md` F3 records the unresolved DropBlock GPU index expression; no compiled-C++ golden or full DropBlock source parity is claimed.

## 137. ScaleChannels source and broadcast-axis pin

- v0.1.038 (`189cebae`): hand-derived from `Jev_laya:src-lib/scale_channels_layer.cpp:12-35,73-135` and `blas_kernels.cu:1664-1676`. A named `from=0` source with a batch-2 non-square NCHW sentinel pins both channel-vector broadcast (`scale_wh=0`) and spatial-plane broadcast (`scale_wh=1`), plus input/source adjoint accumulation. `MEM=3G ./tools/test_host.sh scale_channels_axis_layout` passed; full axis filter passed 33. No compiled-C++ golden.

## 138. SAM elementwise axis pin

- v0.1.039 (`3e674de4`): hand-derived from `Jev_laya:src-lib/sam_layer.cpp:69-105` and `blas_kernels.cu:1752-1798`. A batch-2, non-square NCHW sentinel checks the complete elementwise named-source product, input adjoint, and source adjoint at each `(n,c,h,w)`. `MEM=3G ./tools/test_host.sh sam_axis_layout` passed; full axis filter passed 34. No compiled-C++ golden.

## Not implemented / open (pointer)

See `tasks/rust/todo.md` for the deduplicated open list (P8/P9 follow-ups,
fp16 remainder, hardware-blocked wgmma/tcgen05/NVFP4/multi-GPU, and Stage-2 GPU
Muon), `layers.md` for Phases 1b–4 (remaining layers),
`layer-support.md` for the per-`LayerKind` CPU/GPU/f16/axis matrix and the
current gap lists (GPU hard rejections, forward-but-CPU-backward layers,
partial device paths), `fp8_fp4.md` for FP8 step 5, and `gui_plan.md` for the
GUI plan (partly implemented in v0.1.019–v0.1.021); ONNX export is complete.

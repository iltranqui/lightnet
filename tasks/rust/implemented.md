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
- [31. C++ enum boundary config regression](#31-c-enum-boundary-config-regression)
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
- SAM still uses the CPU backward path. No-BN RNN/CRNN/LSTM backward uses
  backend ops; BN RNN/LSTM/CRNN backward also uses backend ops with a host
  forward/cache upload and host optimizer.

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

## Not implemented / open (pointer)

See `tasks/rust/todo.md` for the deduplicated open list (P8/P9 follow-ups,
fp16 remainder, hardware-blocked wgmma/tcgen05/NVFP4/multi-GPU, and Stage-2 GPU
Muon), `layers.md` for Phases 1b–4 (remaining layers),
`fp8_fp4.md` for FP8 step 5, and `gui_plan.md` for the deferred GUI plan; ONNX
export is complete.

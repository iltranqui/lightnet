# Todo (branch rust)

Plan: ~/.claude/plans/https-github-com-nvidia-cuda-rust-port-t-imperative-pumpkin.md
Goal: cuda-oxide backend `src/darknet-cuda-oxide` implementing `F32TensorOpsBackend` with Rust kernels ported from `src-lib/*.cu`.

> **All implemented work has moved to [`implemented.md`](implemented.md).** This
> file tracks only open / partial / deferred items. Phase tables live in
> [`layers.md`](layers.md); the FP8/FP4 plan is [`fp8_fp4.md`](fp8_fp4.md).

## Contents

- [Build/test (one command at a time, repo root)](#buildtest-one-command-at-a-time-repo-root)
- [Hardware-blocked (cannot be compiled/tested on this sm_89 GPU)](#hardware-blocked-cannot-be-compiledtested-on-this-sm_89-gpu)
- [Backend / kernel follow-ups](#backend--kernel-follow-ups)
- [Gradients](#gradients)
- [P8/P9 training follow-ups](#p8p9-training-follow-ups)
- [Data pipeline](#data-pipeline)
- [FP8 / FP4 (`fp8_fp4.md`)](#fp8--fp4-fp8_fp4md)
- [Layer port (Phases 1b–4 — see `layers.md` phase tables)](#layer-port-phases-1b4--see-layersmd-phase-tables)
- [Active and unstarted plans](#active-and-unstarted-plans)
- [ONNX export](#onnx-export)
- [Known evidence gaps (not user questions)](#known-evidence-gaps-not-user-questions)
- [Not implemented by design / no scanned cfg needs it](#not-implemented-by-design--no-scanned-cfg-needs-it)
- [Known small diffs vs C++](#known-small-diffs-vs-c)
- [Version rewrite v0.0.001 – v0.0.246](#version-rewrite-v00001--v00246)

## Build/test (one command at a time, repo root)

```
cargo check -j 4 --workspace && cargo build -j 4 --workspace && cargo run -j 4 -p darknet-cli
MEM=3G ./tools/test_host.sh                      # darknet-core + darknet-cli
MEM=3G ./tools/test_host.sh <filter>             # one filter, much faster
MEM=3G ./tools/test_oxide.sh                      # ~15 min, all suites
systemd-run --user --scope -p MemoryHigh=3G --setenv=RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
  target/debug/cargo-oxide oxide test -- --manifest-path src/darknet-cuda-oxide/Cargo.toml -j 4
```

Run `cargo test -p darknet-core --test cfg_corpus -j 2` alone before believing a failure. The full isolated corpus run exceeded a 40-minute timeout during v69 with no final result; do not record that as a pass or a test failure.

## Hardware-blocked (cannot be compiled/tested on this sm_89 GPU)

- [ ] wgmma (sm_90) / tcgen05 (sm_100 Blackwell) GEMM kernels and per-arch
  dispatch (the `GemmArgs` + `GemmMode` seam is in place).
- [ ] Native NVFP4 `mma` (needs sm_120a/121a); the sm_89 path is the `__dp4a`
  emulation.
- [ ] Multi-GPU (2 GPUs planned); keep GEMM/optimizer behind interfaces.

## Backend / kernel follow-ups

- [ ] CUDA graph capture: skipped — cuda-core only has raw begin/end capture (no
  instantiate/launch wrapper) and pool addresses are not static.
- [x] fp16 convolution dgrad/wgrad: Oxide AMP stores convolution inputs/gradients in f16, uses f32 accumulation for dgrad col2im and wgrad, and returns f32 weight gradients. Verified by `MEM=3G ./tools/test_oxide.sh --test oxide_amp` (11/11 passed, including grouped and multi-chunk convolution cases). Half2 elementwise remains unnecessary.
  Standalone f16 batchnorm uses the channel-affine kernel, standalone bias +
  activation and reverse upsample forward are implemented, and all three have
  `oxide_half` / runtime f16 parity coverage.
- [ ] Implicit-GEMM conv as default: implemented (`DARKNET_OXIDE_CONV=implicit`)
  but ~5% slower (materialised B tile uses 16-byte `cp.async`, implicit only
  4-byte scalar copies); stays opt-in, `Auto` keeps the materialised path.
- [ ] f32 GEMM epilogue bias/activation and training-forward BN-apply +
  activation fusion: blocked because backward needs the pre-activation.

## Gradients

- [~] Analytical gradients everywhere: the C++-shortcut audit is done and
  `DARKNET_GRAD_MODE=analytical|cpp` gates softmax and scale_channels (see
  `implemented.md` §12). v65 added host/Oxide `NORM_CHAN*` primitives and v66
  integrated them into convolution CPU/f32-GPU forward/backward, with exact
  default derivatives and explicit C++ CUDA shortcuts. Other layer types and
  f16 inference reject these activations; no compiled-C++ golden is available.
  Other parity gaps include conv `reverse`/`deform`. Convolution `grad_centr` matches the C++ BN gate,
  per-filter mean subtraction, and `f_size % 32` skip;
  `gpu_grad_centr_runtime_matches_cpu_and_respects_batchnorm_gate` and
  `gradient_centralization_matches_reference_and_is_noop_off_warp_sizes` pass
  on host/FakeBackend and real Oxide. YOLO IoU/GIoU geometry and
  objectness/class sigmoid-BCE deltas now have finite-difference
  coverage; the box test records Darknet's legacy 0.5 width/height comparison
  factor. DIoU/CIoU deltas have source-derived parity coverage for contained,
  edge-crossing/tied, and horizontal/vertical disjoint boxes. Do not force
  DIoU/CIoU deltas to equal naive finite differences:
  C++ `box_diou` uses `(d/c)^0.6` while `dx_box_iou` supplies the legacy gradient
  of `d/c`; CIoU similarly uses an approximate `alpha` gradient. Overlapping-window
  maxpool with `maxpool_zero_nonmax` also has finite-difference and real Oxide
  runtime-chain forward/backward/SGD parity coverage; the argmax-only kernel
  inherently writes zero gradients to non-argmax inputs.
  Weighted shortcut CPU gradients have finite-difference coverage; GPU input
  and source gradients use backend elementwise multiply/fit operations. Oxide
  assembles raw weight dot products and applies exact None/ReLU/softmax
  normalization-chain derivatives on-device, preserving `GradMode::Cpp` raw-z
  behavior; only the final per-input/per-slot weight-gradient vector crosses to
  host train buffers. Parameter/optimizer updates remain host-side.

## P8/P9 training follow-ups

- [~] Recurrent GPU execution: no-BN RNN projections and CRNN recurrent
  convolutions use backend forward/backward ops. No-BN LSTM gates/cell products,
  BPTT gate derivatives, connected backward, and recurrent/input-gradient
  accumulation use backend ops. Training retains LSTM backward caches on-device
  instead of downloading projection/gate/cell caches; timestep outputs are
  concatenated directly on-device; LSTM backward slices input/output gradients
  and assembles input gradients on-device too. RNN now also slices input/output
  sequences and assembles input gradients on-device, accumulating projection
  gradients on-device across steps before one transfer per block. CRNN forward
  and backward sequence slicing and input-gradient assembly are on-device, and
  convolution parameter gradients accumulate on-device across steps before one
  transfer per block. No-BN RNN and CRNN training pre-activation/state caches
  now stay on-device for backward, avoiding per-timestep cache downloads and
  re-uploads; host-facing forwards still capture the legacy CPU cache. Device
  parameter-gradient transfers, host accumulation/SGD, and some sequence handling
  still cross the host. BN RNN/LSTM backward uses backend projection/BPTT ops;
  BN CRNN backward uses backend convolution/BPTT ops with rolling-stat affine
  input/beta derivatives. BN scale-gradient reductions now use existing backend
  centering, channel-scale, pointwise-multiply, and channel-reduction ops; the
  Oxide path no longer transfers raw projections/convolutions and deltas to host
  just for dscale.
  Recurrent BN forward/cache capture, rolling-stat update, and optimizer remain
  host-side. FakeBackend and real Oxide parity cover parameter, rolling-stat,
  and upstream connected/convolution updates. Dynamic loss-scale finiteness
  checks include host-buffered connected/recurrent gradients. Unsupported
  recurrent activations still use CPU backward.
- [x] Host Adam: connected, standalone batchnorm, local, weighted shortcut,
  MoE fallback, deconvolutional, GRU/RNN/LSTM/CRNN, and ViT host params use
  per-layer rate scaling and persisted host moments. DNOPT3 also persists
  device MoE Adam moments; DNOPT1/2 remain readable, and legacy DNOPT2 MoE
  moments migrate to the device on the first no-BN MoE Adam update. ViT
  preserves weight order, decay groups, gradient clipping, and its reduced mHC
  learning rate.
- [~] weighted shortcut: per-feature/per-channel oxide forward composes
  channel-scale + add; input/source gradients use backend elementwise multiply
  and shortcut-fit ops. Oxide assembles raw weight dot products and applies the
  exact normalization derivative, including `GradMode::Cpp`; only the final
  per-input/per-slot gradient transfers to host. Parameter/SGD updates remain
  host-resident. FakeBackend and real-device runtime tests cover short-source
  prefixes, forward, backward, and SGD parity.
- [ ] `reverse` conv, `deform`, adversarial yolo branch.
- [x] End-to-end real-device `subdivisions=2` deferred-gradient flush matches
  CPU weights/biases through two iterations in
  `oxide_runtime_chain::subdivisions_deferred_gradient_flush_matches_cpu`.
- [x] AMP with dropout/dropblock: real-device end-to-end runtime coverage in
  `oxide_runtime_chain::amp_dropout_and_dropblock_runtime_chain_matches_f32_reference`
  checks layer outputs, masks, backward, and SGD against f32 (these ops use the
  documented f32 boundary path under AMP).
- [ ] Real convergence run: only when the user asks.

## Data pipeline

- [~] mixup, multi-scale `random`, and source-level faithful mosaic geometry are
  implemented. Mosaic follows C++ crop alignment, flip-aware quadrant shifts,
  corner relocation, integer box clipping, and strict filtering; the seeded Rust
  RNG is not C++-bit-identical and compiled-C++ mosaic parity is not established.
  Cutmix (which C++ itself rejects), blur/noise/fog remain deferred/rejected.
- [~] Resume image-order replay is not yet guaranteed; Rust SplitMix64 remains
  deterministic but not C++-bit-identical. Hand-derived mosaic geometry and
  single-image OpenCV fixture tests pass.

## FP8 / FP4 (`fp8_fp4.md`)

- [ ] Step 5: FP8 conv via im2col + LegoGears_v2 mAP vs the C++ FP8 path. Not
  runnable here (no C++ build of `v6-dev-fp8`).

## Layer port (Phases 1b–4 — see `layers.md` phase tables)

Immediately next, in dependency order:

- [x] LAYA family (`laya_scorer`, `laya_marker_gather`, `laya_act_head`,
  `laya_loss`, `laya_encoder`, `laya_head`) — all done; see `implemented.md` §5.
- [~] Phase 3A: `[vit]` reference path is implemented: CPU forward/backward,
  header-aware weights, SGD, and GPU f32 host round-trip. Standard flash
  attention has native Oxide forward and dQ/dK/dV primitives inside the
  whole-network CPU fallback (see `implemented.md` §8). VC source audit confirmed
  the key-centering/LSE and padded-query `sq` risks; Rust avoids the padded query
  and defines the centered/clamped-score derivative contract explicitly.
  v0.0.261–v0.0.266 added and tested the VC quantization, V-Smooth, online-row,
  Q/K-scale, head composition, and B×H packing primitives. v0.0.267 wires strict
  cfg aliases/modes/permutations and CPU/GPU host-reference forward/backward;
  `vc_attention` (16 filtered tests) and `vit` (32) pass. GPU VC deliberately
  falls back to the Rust host reference on all devices/head sizes; no native VC
  kernel or compiled-C++ parity is claimed. Swin/transformer integration is
  complete in `model.rs`, `swin.rs`, `swin_block.rs`, `runtime.rs`, and
  `weights.rs`; all four aliases, CPU exact backward, Rust-defined weights,
  SGD/Adam, and GPU f32 host fallback are covered. `MEM=3G
  ./tools/test_host.sh swin` passed 25 tests, `runtime` passed 166, and `weights`
  passed 29. No native Swin kernel, C++ weights writer, or compiled-C++ parity is
  available; see `swin_plan.md`. v0.0.270 (`ecd4e54e`) completes `[tucker_attention]` on the
  host reference path; `MEM=3G ./tools/test_host.sh tucker_attention` passes 14
  tests, `weights` 32, and `runtime` 168. v0.0.271 completes MambaVision with
  strict config, explicit recurrent state, exact CPU forward/backward, seeded
  initialization, SGD/Adam, source-order `.weights`, optimizer persistence, and
  GPU f32 host fallback. `MEM=3G ./tools/test_host.sh mambavision` passed 16
  tests, `runtime` 171, and `weights` 33. Coverage includes input/all-parameter/
  state finite differences, causal convolution, MSA clip derivatives, weight
  initialization/decay, serialization, Adam-state persistence, and FakeBackend
  parity. No native kernel or compiled-C++ golden. Kimi KDA is integrated in
  v0.0.272; see `implemented.md` §8. MiniMax sparse attention is a shared
  option, not a standalone layer in the
  audited C++ branch; ViT, Tucker, and MambaVision now integrate it according
  to their source contracts.
- [x] Container-conv cache prerequisite: device convolution params are keyed by
  owner layer + expert slot; `warm_device_conv_params` warms each MoE expert,
  and host-updated experts invalidate/re-upload their slots. FakeBackend test
  covers slot separation and rewarm. The generic convolution optimizer/BN
  refresh still handles top-level owners only; MoE router/expert SGD and Adam
  have device-side paths, including BN scales, using owner/expert convolution slots.
  Router parameter values sync to host after each update because the current
  router forward consumes host params; gradients and optimizer moments stay
  device-side. DNOPT3 persists no-BN device MoE Adam moments; DNOPT4 adds BN
  expert scale moments and migrates nested host BN moments to device. DNOPT1-3
  remain readable. BN expert backward uses backend ops; BN MoE SGD/Adam update
  experts/router and BN scales on-device, with SGD momentum retained in device buffers.
- [~] Phase 2B `moe_convolutional`: real-device forward computes router pooling,
  logits, top-k masking, softmax, expert convolutions, and gate blending. For
  no-BN experts, router connected-backward, gate-column selection and gate-dot
  reductions, load-balance plus top-k softmax-Jacobian backward, router-pool
  input-gradient broadcast, and expert convolution backward use backend ops;
  gates and pooled/expert activation/output caches remain device-resident, with
  no gate/dot transfers in the normal Oxide training path. Router/expert
  parameter gradients accumulate on-device across backward calls. For SGD and
  Adam, no-BN router and expert updates run on-device without gradient transfers.
  Router parameter values sync to host after each step for the next forward,
  while expert values support deferred host sync. BN expert backward uses
  backend ops and accumulates scale/beta gradients on device;
  BN expert SGD/Adam update weights, beta, and scale on-device; SGD momentum
  stays in device buffers, while Adam scale moments use DNOPT4 (DNOPT1-3 remain
  compatible). The
  dynamic-loss-scale finiteness check covers dirty router/expert/BN-scale
  buffers (`gpu_moe_device_gradients_participate_in_loss_scale_finiteness_check`);
  FakeBackend
  parity also checks router momentum residency, `dont_update` gradient
  retention, and BN expert backward/SGD parity.
  Verified against CPU in
  `oxide_runtime_chain::moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu`
  (two backward calls accumulated before the first deferred SGD step, followed
  by a second deferred step, including upstream convolution updates) and
  `oxide_runtime_chain::moe_adam_updates_router_and_experts_on_oxide_match_cpu_gradients`,
  `oxide_runtime_chain::moe_batchnorm_expert_device_backward_matches_cpu_on_oxide`,
  and `oxide_runtime_chain::moe_batchnorm_adam_updates_experts_on_oxide_match_cpu_gradients`;
  BN Adam parity checks scales, upstream convolution, and deferred host sync.
  FakeBackend optimizer-state tests cover host BN migration and DNOPT4 scale
  round-trip; router-specific soft/tied top-k parity is in `oxide_layers_extra`.
  CPU path is done (`cdf68d00`). v67 adds source-derived pure forward/backward
  math for the EML two-branch combination (`eml_conv_math.rs`, four host tests).
  v68 adds host-reference and native Oxide f32 branch-combine ops, optional
  identity residual, and the C++ nonpositive clamp/epsilon fallbacks. v69 adds
  `[eml_convolutional]` parsing/shape inference, paired convolution parameters,
  CPU forward/backward, C++ `input_layer` then `self_layer` weight serialization,
  and SGD/Adam updates. Focused host EML tests pass 3/3; EML weights round-trip
  passes in `recurrent_weight_blocks_round_trip_in_darknet_order`. v70 runs both
  EML branch convolutions and the combine through the f32 backend on GPU
  forward, including rolling-stat BN folding; `oxide_runtime_chain` passes
  19/19, including BN/no-BN outputs and cache invalidation after host updates.
  CPU branch caches still feed fallback paths when needed. v71 adds native
  Oxide backward for no-BN EML: backend EML adjoints and both convolution
  input/weight gradients, residual routing, and host accumulation for SGD/Adam.
  v72 adds per-branch batch-stat BN forward/backward caches, exact BN scale/beta
  gradients, rolling-stat EMA propagation, and host train-buffer accumulation;
  BN EML no longer forces a CPU backward fallback. `MEM=3G PKGS=darknet-core
  ./tools/test_host.sh eml_convolutional` passes 3/3 and
  `./tools/test_oxide.sh --test oxide_runtime_chain` passes 19/19, including
  BN rolling-stat inference, finite native BN backward, EMA updates, and host
  optimizer updates. v73 adds per-branch device gradient/momentum buffers,
  native SGD for BN and no-BN EML, and deferred synchronization for both branch
  parameter blocks; SGD accumulation and finiteness checks stay device-side.
  `MEM=3G PKGS=darknet-core ./tools/test_host.sh eml_` passed 9/9 and
  `./tools/test_oxide.sh --test oxide_runtime_chain` passed 19/19, including
  multi-step SGD momentum parity and deferred host sync. v74 adds native Adam
  for both branches and BN scales, per-branch moments, deferred synchronization,
  host fallback for CPU-backward networks, and DNOPT5 sidecar persistence while
  retaining DNOPT1–4 readers and migrating DNOPT3 EML host moments. Host EML
  tests pass 12/12, optimizer-state tests pass 5/5, and `oxide_runtime_chain`
  passes 20/20, including no-BN Adam update parity, BN-scale updates, and
  deferred sync. No compiled-C++ golden is available; CPU EML BN training still
  uses rolling rather than batch statistics, so no CPU/C++ training parity is
  claimed.
- [ ] Phase 2B `dcnv3` blocked: no reference numerics (stub in all five branches).
- [ ] Phase 2C OBB family (unblocked by decision 4; XL — also data loading, NMS,
  mAP). v75 adds legacy front-point rectangle geometry; v76 adds standalone
  CARPN convex-quad IoU; v77 adds typed geometry dispatch and an OBB NMS helper;
  v78 adds standalone legacy `[obb_yolo]` six-coordinate per-head decoding with
  source-derived front-point and integer-letterbox transforms; v79 adds a
  standalone five/seven-field OBB label parser preserving the optional front
  point; v80 adds source-derived label bounds and rotation; v81 adds
  `[net] obb=1` and `[obb_yolo]` parsing, runtime head capture, network-level
  multi-head OBB decode/NMS, and explicit AABB-training/mAP rejection; v82 adds
  separate no-augmentation OBB batch loading with direct image resize and
  preserved optional-front labels; v83 adds standalone AP matching for normalized
  legacy front-point OBB detections; v84 adds an OBB validation dataset/evaluation
  API with network decode and normalized orientation-aware scoring; v85 adds a CPU
  `detector map` CLI route for standard and legacy OBB evaluators; v86 adds the
  source-default oriented `nms_kind=obb` path and rejects unsupported alternatives;
  v87 adds a standalone horizontal-mirror transform for OBB labels; v88 applies
  caller-selected per-image flips to OBB batch tensors and labels together; v89
  adds source-derived direction-free anchor-size matching and best-anchor/cell
  assignment primitives; v90 adds seeded config-driven horizontal flips through
  an OBB-specific batch loader; v91 adds seven-float truth-row packing in the
  source batch/box-slot layout with absent-front placeholders and zero padding;
  v92 enumerates primary and strict-threshold additional per-head positive
  anchors; v93 selects best-truth and ignore/positive gates from precomputed
  source-style prediction scores; v94 adds a standalone CARPN polygon mAP helper
  for normalized quad detections and front-point labels; v96 adds source-style
  per-cell objectness deltas from the v93 match gates, including ignore smoothing
  and positive-target override; v97 adds source CIoU-OBB match-quality scoring for
  decoded geometries; v98 decodes one cell's post-YOLO channels for the source
  CIoU/default score path; v99 composes those per-truth scores with the source
  class-gated ignore and positive-threshold selection for one cell; v100 adds
  source-style class-channel deltas for positive cells; v101 adds standalone
  covariance-W2 loss terms and an `exp(-loss)` match quality for decoded OBB
  geometry; v102 adds a raw-channel covariance-branch match score for one cell;
  v103 adds the source QBBV heatmap and sampled vIoU metric; v104 adds the
  source QBBV blend and covariance-branch integration; v105 adds the source
  legacy angular fallback scorer and one-cell decode/blend; v106 adds decoded
  COBB loss/quality and QBBV blending; v107 adds raw-channel COBB match decoding
  and source branch gates; v108 adds one-cell covariance-first/COBB/angular
  score dispatch; v109 composes branch scores with per-truth cell gates; v110
  enumerates primary/additional positives across batch slots and multiple heads;
  v111 maps caller-selected head/cell predictions to batch-specific truths and
  returns tagged match gates; v112 traverses a full batch-major head tensor in
  source cell/anchor order; v113 emits source objectness deltas for batch/head
  cells with ignore and positive overrides; v114 writes those deltas into entry 6
  of one head's delta tensor without altering other entries; v115 applies
  source best/additional positive-anchor objectness overrides, class mapping and
  smooth-mode preservation; v116 writes source class-channel deltas at those
  positives with existing-target behavior; v117 adds source finite-difference
  COBB/legacy box deltas and positive-anchor writeback; v118 ports source
  covariance-W2 analytic geometry gradients; v119 routes those gradients into
  positive-anchor box-delta writeback; v120 reports `avg_cat` contributions
  from best/additional class assignments; v121 applies mapped class multipliers
  to box-delta normalization; v122 applies truth-threshold cell class and box
  deltas after the all-cell objectness pass; v123 composes ordered delta passes
  for one complete head; v124 dispatches those passes across multiple head
  tensors; v125 adds a standalone center-front-anchor point decoder; v126 routes
  optional anchor offsets through head matching and box deltas; v127 aggregates
  multi-head class and positive-assignment metrics; v128 reports source total-bbox
  counts across threshold and anchor assignments; v129 adds a standalone source
  front-point delta primitive; v130 writes those deltas to best/additional positive
  anchor channels; v131 writes them to truth-threshold-positive cells; v132
  composes front deltas in the one-head pass order; v133 dispatches that path
  across multiple heads; v134 reports source-normalized front-loss sums/counts
  across those passes and heads; v135 parses source `obb_anchors` pair/quadruple
  forms and retains per-anchor direction priors; v136 parses the front-loss
  normalizer and alias/default; v137 builds per-head delta configs from parsed
  OBB specs plus runtime dimensions; v138 supports per-head front-loss configs
  during multi-head dispatch; v139 builds network-layer delta seeds from OBB
  forward heads and labels; v140 propagates those seeds through runtime
  backpropagation; v141 adds a one-subdivision CPU SGD batch step with explicit
  Adam/subdivision rejection; v142 accumulates configured subdivisions and applies
  one normalized SGD update; v143 parses OBB `counters_per_class` into
  assignment class multipliers; v144 parses and routes OBB relative class-map
  files; v145 parses and routes covariance/COBB normalizers and QBBV blending
  options into per-head scoring; v146 wires parsed `iou_loss` to CIoU-OBB score
  dispatch; v147 routes the parsed front-loss normalizer into covariance
  match-score weighting; v148 retains standard `iou_loss` choices in the parsed
  spec and routes them to front-point deltas; v149 retains `iou_thresh_kind` in
  the parsed spec; v150 explicitly rejects `ciou_obb_mul` training until its
  distinct loss/gradient path is implemented; v151 aggregates front-vector angle
  error mean/threshold counts in front-aware stats across passes and subdivisions;
  v152 separates weighted MSE matching scores from the source CIoU-OBB box-delta
  fallback; v153 adds covariance and COBB box-delta contributions independently
  when both weighted branches are active; v154 combines their matching scores by
  configured weights; v155 reports source raw `sum(delta²)` for all-MSE OBB heads;
  v156 reports the objectness-normalized non-box delta-squared component; v157
  allows distinct weighted MSE matching and CIoU-OBB regression configs through
  batch dispatch; v158 aggregates source-style MSE and supported non-MSE loss
  costs across heads, with non-MSE box losses counted only for best/additional
  anchor assignments; v159 routes `detector train` through a bounded CPU OBB
  SGD loop with configured subdivisions, direct resize, seeded horizontal flips,
  checkpoints and loss CSV logging, and explicitly rejects `-map`, Adam, dynamic
  loss scaling, unsupported augmentation, and OBB GPU training. The CLI test
  covers one iteration and the `-map` rejection; all 119 OBB-filtered core tests
  plus the CLI test passed. v160 adds a separate subdivision-aggregated OBB
  metrics CSV (average IoU/GIoU, class output, assignment counts, loss components,
  front loss, and orientation accuracy/error buckets); the CLI schema/count test,
  four detector-train CLI tests, and 119 OBB-filtered core tests pass. v161 routes
  CPU OBB training `-map` through orientation-aware validation and shared cadence;
  the test verifies `-map_every 2` and the final-iteration evaluation, best weights,
  and map log. `detector_train_` passed 4 CLI tests, `map_` passed 4 CLI + 11 core
  tests, and `obb_` passed 119 core + 1 CLI test. v162 routes `detect-image`
  `[net] obb=1` outputs through the orientation-aware decoder and prints legacy
  front-point geometry in human and `det_obb` TSV output; mixed OBB/modern heads
  fail explicitly. The focused CLI tests and `detect_` passed 2 CLI tests; `obb_`
  passed 119 core + 2 CLI tests. v163 accepts source `default` and `cornersnms`
  OBB NMS kinds with AABB-IoU suppression, retaining orientation-aware `obb` and
  source aliases; `obb_` passed 120 core + 2 CLI tests. v164 implements source
  `greedynms` (fixed 0.6 DIoU exponent) and `diounms` (`beta_nms`, default 0.6)
  using AABB box formulas; non-finite betas and mismatched multi-head DIoU betas
  fail explicitly. v0.0.249 adds CPU OBB Adam with subdivision-correct updates and
  DNOPT sidecar moments; v0.0.250 adds photometric augmentation while retaining
  explicit rejection of unsupported geometry; v0.0.251 adds weighted OBB NMS
  (`obb_w`/`obb_wnms`); v0.0.252 defaults omitted OBB jitter to zero; v0.0.253
  implements multiplicative CIoU-OBB loss and its six-coordinate finite-difference
  gradients. The direct `cargo test -p darknet-core -p darknet-cli -j 2 -- obb_` filter
  passed 135 core + 4 CLI tests; after systemd scope availability was confirmed,
  `MEM=3G ./tools/test_host.sh obb_` also passed 135 core + 4 CLI tests.
  No compiled-C++ parity is established.
  v0.0.258 adds sample-level CARPN mAP dispatch through `obb_detector_map`; focused
  `obb_map` (5 core), `carpn_map` (3), and `carpn_polygon_iou` (2) host filters pass.
  v0.0.259 completes CPU CARPN network inference, `detect-image`, and direct
  `detector map` via real cfg/weights-backed fixture tests; `carpn_refine_cfg` (2),
  `carpn_refine_weights` (1), `carpn_refine_forward` (3), `carpn_refine_decode` (5),
  `carpn_map` (6), `carpn_cli` (1), `detect_image_routes_carpn` (1), and
  `detector_train_rejects_carpn` (1) pass. The legacy `obb_` filter passes 138 core
  + 4 CLI tests. CARPN Stage-2 training and scheduled `detector train -map` remain
  deferred. Still open: broader OBB GPU training/validation, remaining cfg/training-
  option wiring, source-faithful geometric augmentation/label transforms, and any
  unimplemented OBB source paths.
- [x] Phase 3B bounded Stage-1 CPU layer ports: `graph_conv` v0.1.003
  (`implemented.md` §20), WMHF v0.1.004 (§21), Clifford v0.1.006 (§23), and
  `recursive_block` v0.1.007 (§24) are source-derived from `Jev_laya`; no
  compiled-C++ golden/parity or GPU path is claimed. CPU Muon is integrated in
  v0.1.005 (§22). This closes the bounded CPU-port task set, not every source
  variant: DSConv multi-template `morph=0/1` remains open, and DCNv1/2 plus
  `[dcnv4]` still use the bounded CPU/reference or host-fallback scope recorded
  in §§17–19. Native GPU support and GPU Muon remain Stage 2.
- [x] CPU Muon: spec and Stage-1 hybrid implementation complete in v0.1.005
  (`implemented.md` §22). Eligible matrices use paper-derived Muon; other
  trainable parameters use AdamW fallback. DNOPT6 persists shape-checked
  momentum while DNOPT1–5 remain readable. `muon_` (13), `weights` (43), and
  `runtime` (204) host tests passed. No training, Oxide tests, or compiled-C++
  golden/parity. GPU Muon is deferred to Stage 2.
- [~] Phase 4 foveation / depth / DETR: Stage-1 CPU `[hyperbolic_foveation]` is complete in v0.1.008 (`implemented.md` §25), `[foveation_film]` in v0.1.009 (§26), `[depth_anything]` in v0.1.010 (§27), and `[dpt_head]` in v0.1.013 (§30); LAYA encoder/head are already implemented. `[yolo_3d]` and DETR remain open. DETR is still broken in C++ (paper only).
- [x] Stage-1 A1 axis/layout audit of every already-ported family named in `AGENTS.md`: completed in v0.1.012 (`implemented.md` §29), with the initial subset recorded in §28. `STATUS.md` tabulates the source contracts and named tests. Per-backlog axis DoDs for missing-layer ports and the remaining v0.1–v0.3 work stay open.

## Active and unstarted plans

- [x] `onnx.md` — `darknet-rs export-onnx`; all 21 implementation, verification, and bookkeeping steps are complete.
- [x] `carpn_map_plan.md` — bounded caller-supplied sample dispatch implemented in v0.0.258.
- [x] `carpn_network_map_plan.md` — v0.0.259 completes CPU network inference, `detect-image`, and direct `detector map`; CARPN training/`detector train -map` remain explicitly deferred.
- [x] `dcnv4_plan.md` — source-audited CPU/reference + GPU f32 host-fallback implementation completed in v0.0.273; v0.0.274 matches the deformable-layer `try_fix_nan` config default. No native kernel or compiled-C++ parity.
- [x] `deform_conv_plan.md` — bounded DCNv1/2 CPU reference and f32 host-fallback implementation completed in v0.1.001. No native kernel or compiled-C++ parity; CPU Muon was added separately in v0.1.005.
- [x] `dsconv_plan.md` — bounded single-template CPU implementation completed in v0.1.002; multi-template modes remain open. CPU Muon was added separately in v0.1.005. No native kernel or compiled-C++ parity.
- [x] `graph_conv_plan.md` — Stage-1 CPU implementation completed in v0.1.003; CPU Muon was added separately in v0.1.005 and native GPU support remains Stage 2. No compiled-C++ parity.
- [x] `wmhf_plan.md` — bounded Stage-1 CPU implementation completed in v0.1.004; source/config/backward/weights audit is in `wmhf_notes.md`. CPU Muon was added in v0.1.005. No compiled-C++ golden.
- [x] `clifford_plan.md` — source-derived Stage-1 CPU implementation and focused verification completed in v0.1.006; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `recursive_block_plan.md` — source-derived Stage-1 CPU implementation and focused verification completed in v0.1.007; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `hyperbolic_foveation_plan.md` — source-derived Stage-1 CPU support and verification completed in v0.1.008; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `foveation_film_plan.md` — source-derived Stage-1 CPU support and focused verification completed in v0.1.009; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `depth_anything_plan.md` — source-derived Stage-1 CPU support and focused verification completed in v0.1.010; no compiled-C++ golden/parity. Native GPU is Stage 2.
- [x] `dpt_head_plan.md` — source-derived Stage-1 CPU support and focused verification completed in v0.1.013; no compiled-C++ golden/parity. Native GPU and GPU Muon remain Stage 2; `[yolo_3d]` remains separate.
- [x] `muon_plan.md` — Stage-1 CPU hybrid Muon implementation and focused verification completed in v0.1.005. GPU Muon remains Stage 2; no compiled-C++ golden/parity.
- [ ] `gui_plan.md` — egui DarkMark replacement (M0–M6, not started).

## Known evidence gaps (not user questions)

- `[dcnv3]`: forward/backward/update are stubs in all five branches that carry it
  (`deform`, `Jev_laya`, `fog`, `jepa`, `depth_3d`); only the `.weights` order is
  portable against C++.
- `[contrastive]` detection branch needs the yolo head's per-box class ids
  (`YoloHeadOutput` only carries the raw output tensor).
- DETR is broken in C++, so Phase 4's `detr_decoder` has no reference numerics.
- No C++ build of any owning branch is available here, so C++ dump parity is
  "NOT RUN" everywhere.

## Not implemented by design / no scanned cfg needs it

- All parsed scalar activations with CPU forward/derivative support now have
  oxide kernels, including `gelu`, `elu`, `selu`, `hard_mish`, `loggy`,
  `hardtan`, `relie`, `lhtan`, `plse`, and `stair`. `normalize_channels*` is
  supported on `[convolutional]` for CPU and f32 GPU runtimes; the parser rejects
  other layer types and f16 inference rejects it explicitly.
- xnor (74 layers), share_index (54), `new_coords=1` yolo (path looks wrong),
  antialiasing, conv grad clamps / `try_fix_nan` clamps,
  `rewritten_bbox`/`total_bbox` logging.

## Known small diffs vs C++

- CPU resize uses `image` Triangle (differs for non-network-size inputs); EXIF
  orientation ignored.
- Dilated conv: oxide needs `pad*dilation` (only resnet152_trident); darknet-core
  output dims ignore dilation.
- relu/relu6 NaN -> 0 on oxide (C++ NaN); mish/swish at -inf -> NaN.
- C++ CUDA reference runs `cudnn_half=1`, so layer dumps differ from pure fp32 by
  up to 5.7e-3 abs (not an oxide error).

## Version rewrite v0.0.001 – v0.0.246

- [x] Snapshot main tree (status, index, refs, stash)
- [x] Backup tag `backup/rust-pre-version` -> `a4c40375` (old tip)
- [x] Guard: 246 commits `e05545c1..a4c40375`, 0 merges, none reachable from other branches
- [x] `git rebase --exec` in temp worktree (branch `rust-versioned`); hook checks each replayed tree equals the original before amending
- [x] Verify: 246/246 rows; parent chain; trees; author + committer name/email/date; raw message bytes after subject; subject sequence `v0.0.001…246`; tip tree == old tip tree; main tree and other refs unchanged
- [x] Swap `refs/heads/rust`: `a4c40375` -> `d1972af5` (compare-and-swap); temp worktree and branch removed
- Map (old -> new per index): `tasks/rust/version-map.tsv` (git-excluded)

Review:
- Only subject lines changed. `rust milestone vN: ` prefix stripped on 163 commits, 83 subjects kept as-is, body bytes identical.
- Committer identity and date restored from the original commit (the amend would otherwise use the current git config identity).
- Not pushed. `rust` is now 246 ahead / 7 behind `origin/rust`; updating the remote needs `git push --force-with-lease origin rust` by hand.
- Left alone: 25 old-hash references in `tasks/rust/` (`implemented.md` 16, `layers.md` 7, `todo.md` 1, `STATUS.md` 1). They still resolve via the backup tag; map above gives replacements.
- Rollback: `git update-ref refs/heads/rust backup/rust-pre-version`

## ONNX export

- [x] `darknet-rs export-onnx` is implemented: protobuf encoder (`v0.0.254`), supported-layer exporter (`v0.0.255`), CLI (`v0.0.256`), and Rust-runtime checker (`v0.0.257`). Focused results: `MEM=3G ./tools/test_host.sh onnx` passed 11 core tests; `MEM=3G ./tools/test_host.sh export_onnx_args` passed 1 CLI test; `cargo build -p darknet-cli -j 2` passed. The CLI exported both fused and unfused models from `build/logs/verify/lego_cur_cpu/run.cfg` and `run_final.weights`.
- [x] `python3 -m py_compile tools/onnx_check.py` passed. ONNX Runtime numeric checks against Rust CPU layer dumps passed at tolerance `1e-4`: fused worst relative error `3.257e-06`; unfused `3.908e-05`; Mish-opset-18 variant `3.831e-06`. The Mish test changed Leaky activations to Mish in a temporary cfg while retaining the same real weights; image was `artwork/dog.jpg`.
- Canonical `/home/kerrigan/src/DarkSuite/datasets/LegoGears_v2` assets are absent in this checkout, so verification used the local matching cfg/weights plus `build/logs/bench/LegoGears.names`; no compiled-C++ parity is claimed. `onnx` Python package is absent, but `onnxruntime`, NumPy, and OpenCV are installed and the emitted models loaded and executed successfully in ONNX Runtime.
- [x] ONNX plan bookkeeping and verification are complete; version map, `implemented.md`, and `STATUS.md` record the milestone evidence. Follow-ups listed in `tasks/rust/onnx.md` remain out of scope.

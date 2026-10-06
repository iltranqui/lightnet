# Todo (branch rust)

Plan: ~/.claude/plans/https-github-com-nvidia-cuda-rust-port-t-imperative-pumpkin.md
Goal: cuda-oxide backend `src/darknet-cuda-oxide` implementing `F32TensorOpsBackend` with Rust kernels ported from `src-lib/*.cu`.

> **All implemented work has moved to [`implemented.md`](implemented.md).** This
> file tracks only open / partial / deferred items. Phase tables live in
> [`layers.md`](layers.md); the FP8/FP4 plan is [`fp8_fp4.md`](fp8_fp4.md).

## Build/test (one command at a time, repo root)

```
cargo check -j 4 --workspace && cargo build -j 4 --workspace && cargo run -j 4 -p darknet-cli
MEM=3G ./tools/test_host.sh                      # darknet-core + darknet-cli
MEM=3G ./tools/test_host.sh <filter>             # one filter, much faster
MEM=3G ./tools/test_oxide.sh                      # ~15 min, all suites
systemd-run --user --scope -p MemoryHigh=3G --setenv=RUSTUP_TOOLCHAIN=nightly-2026-08-28 \
  target/debug/cargo-oxide oxide test -- --manifest-path src/darknet-cuda-oxide/Cargo.toml -j 4
```

`cfg_corpus` needs ~145 s alone; run it on its own before believing a failure.

## Hardware-blocked (cannot be compiled/tested on this sm_89 GPU)

- [ ] wgmma (sm_90) / tcgen05 (sm_100 Blackwell) GEMM kernels and per-arch
  dispatch (the `GemmArgs` + `GemmMode` seam is in place).
- [ ] Native NVFP4 `mma` (needs sm_120a/121a); the sm_89 path is the `__dp4a`
  emulation.
- [ ] Multi-GPU (2 GPUs planned); keep GEMM/optimizer behind interfaces.

## Backend / kernel follow-ups

- [ ] CUDA graph capture: skipped — cuda-core only has raw begin/end capture (no
  instantiate/launch wrapper) and pool addresses are not static.
- [ ] fp16 remainder: conv dgrad/wgrad on f16; half2 elementwise (not needed).
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
  header-aware weights, SGD, and GPU f32 host fallback (see `implemented.md` §8).
  Open: flash/vc attention variants and `swin`/`transformer`, `tucker_attention`,
  `mambavision`, `kimi_kda`, `minimax_sparse_attention` on the shared
  LayerNorm/GEMM set (already built).
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
  SGD/Adam updates, and an explicit GPU host-fallback path. Focused host EML
  tests pass 3/3; EML weights round-trip passes in
  `recurrent_weight_blocks_round_trip_in_darknet_order`. Remaining: native
  Oxide whole-layer execution; CPU/GPU-host BN training still uses the existing
  convolution helper's rolling-stat semantics rather than C++ batch-stat/EMA
  behavior, and no compiled-C++ golden is available. Do not claim full C++
  parity.
- [ ] Phase 2B `dcnv3` blocked: no reference numerics (stub in all five branches).
- [ ] Phase 2C OBB family (unblocked by decision 4; XL — also data loading, NMS,
  mAP).
- [ ] Phase 3B deformable family: start with `dcnv4` (real C++ forward and
  kernels, weights order `weights.cpp:428`); then `deform_conv`, `dsconv`,
  `graph_conv`, `wmhf`, `clifford`, `recursive_block`.
- [ ] Phase 4 foveation / depth / DETR. DETR is still broken in C++ (paper only).

## Two unstarted plans

- [ ] `onnx.md` — `darknet-rs export-onnx` (0/23 steps).
- [ ] `gui_iced_plan.md` — iced DarkMark-compatible GUI (0/84 steps).

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

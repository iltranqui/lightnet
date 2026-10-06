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
  `implemented.md` §12). Remaining parity gaps include `NORM_CHAN*` and conv
  `reverse`/`deform`/`grad_centr`. YOLO IoU/GIoU geometry and
  objectness/class sigmoid-BCE deltas now have finite-difference
  coverage; the box test records Darknet's legacy 0.5 width/height comparison
  factor. DIoU/CIoU deltas have source-derived parity coverage for contained,
  edge-crossing/tied, and horizontal/vertical disjoint boxes. Do not force
  DIoU/CIoU deltas to equal naive finite differences:
  C++ `box_diou` uses `(d/c)^0.6` while `dx_box_iou` supplies the legacy gradient
  of `d/c`; CIoU similarly uses an approximate `alpha` gradient. Overlapping-window
  maxpool with `maxpool_zero_nonmax` also has finite-difference coverage.
  Weighted shortcut CPU gradients have finite-difference coverage; GPU input
  and source gradients use backend elementwise multiply/fit operations. Raw
  weight dot products reduce on Oxide and only small per-input/per-slot results
  cross the host; normalization-chain derivatives and optimizer updates remain
  host-side.

## P8/P9 training follow-ups

- [~] Recurrent GPU execution: no-BN RNN projections and CRNN recurrent
  convolutions use backend forward/backward ops. No-BN LSTM gates/cell products,
  BPTT gate derivatives, connected backward, and recurrent/input-gradient
  accumulation use backend ops. Training retains LSTM backward caches on-device
  instead of downloading projection/gate/cell caches; timestep outputs are
  concatenated directly on-device; LSTM backward slices input/output gradients
  and assembles input gradients on-device too. RNN/CRNN input slicing,
  per-projection parameter-gradient transfers, host accumulation/SGD, and some
  cache handling still cross the host. FakeBackend and real Oxide tests compare
  training updates with CPU, including upstream convolution updates.
  Batch-normalized recurrent layers still use CPU backward.
- [x] Host Adam: connected, standalone batchnorm, local, weighted shortcut,
  MoE, deconvolutional, GRU/RNN/LSTM/CRNN, and ViT host params use per-layer
  rate scaling and persisted host moments in DNOPT2 optimizer sidecars (DNOPT1
  remains readable); ViT preserves weight order, decay groups, gradient
  clipping, and its reduced mHC learning rate.
- [~] weighted shortcut: per-feature/per-channel oxide forward composes
  channel-scale + add; input/source gradients use backend elementwise multiply
  and shortcut-fit ops. Oxide reduces raw weight dot products natively, then
  transfers only per-input/per-slot results for the normalization derivative;
  parameters/SGD remain host-resident. FakeBackend and real-device runtime tests
  cover mismatched source fit, forward, backward, and SGD parity. Fully device
  normalization and parameter updates remain future work.
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

- [~] mixup and multi-scale `random` are done. **Faithful mosaic** (needs the
  single-image crop geometry exposed) and cutmix (which C++ itself rejects) are
  deferred; blur/noise/fog are rejected explicitly at startup.
- [~] Resume image-order replay not yet guaranteed; RNG is deterministic Rust
  SplitMix64, not C++ bit-identical. Mosaic stitch tests remain open.

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
  covers slot separation and rewarm. Device-side parameter update/BN refresh
  remains top-level-conv-only; container parameter accumulation and SGD remain
  host-resident.
- [~] Phase 2B `moe_convolutional`: real-device forward computes router pooling,
  logits, top-k masking, softmax, expert convolutions, and gate blending. For
  no-BN experts, router connected-backward and expert convolution backward use
  backend ops; router softmax/gate derivatives, cache transfers, parameter
  accumulation, and SGD remain host-side. BN experts still use CPU backward.
  Verified against CPU in
  `oxide_runtime_chain::moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu`
  (including upstream convolution updates) and router-specific soft/tied top-k
  parity coverage in `oxide_layers_extra`.
  CPU path is done (`cdf68d00`). `eml_convolutional` not started (save order
  verified 2026-10-05).
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
  `hardtan`, `relie`, `lhtan`, `plse`, and `stair`. `normalize_channels*` remains
  a config error.
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

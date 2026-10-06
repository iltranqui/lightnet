# Porting progress — durable state

Read this first after a context compaction. It records what is verified and what
is next. The authoritative plan is `tasks/rust/layers.md`; the FP8/FP4 plan is
`tasks/rust/fp8_fp4.md`; the **full implemented inventory is
`tasks/rust/implemented.md`** (this file no longer repeats it).

## Ground rules (do not violate)

- Every cargo command uses `-j 4` or fewer. `tools/test_host.sh` and
  `tools/test_oxide.sh` use `-j 2`.
- `src/darknet-cuda-oxide` is **not** a workspace member; build/test it with
  `./tools/test_oxide.sh` (which wraps `cargo oxide`).
- Backward is always the **exact derivative of the Rust forward**. C++
  deviations only under `GradMode::Cpp`.
- Unknown config = error, never silent identity.
- Commit only the files a change names.

## User decisions (2026-10-04) — all Open Questions answered YES

"port everything in rust". Recorded in full in `tasks/rust/layers.md`
("Open questions for the user — RESOLVED 2026-10-04"). Short version:
ship the cfgs, port LAYA, port the `flash_attention`/`vc_attention` variants,
port the whole OBB family, build a dense-truth trainer, keep the legacy layers,
and let Rust define the swin weights order. **Nothing is blocked on a question
any more** — the remaining blockers are evidence gaps only.

## Implemented

Everything delivered so far is consolidated in
[`tasks/rust/implemented.md`](implemented.md): the cuda-oxide backend (P0–P9),
FP8/FP4, layer-port Phases 0 / 1b / 2A / 2B-CPU and the Phase 3A shared
primitives, LegoGears_v2 inference parity, the training loop + data pipeline,
and housekeeping — each with its evidence.

Latest increments:
- Added no-BN LSTM device backward: gate derivatives, connected projection
  gradients, and recurrent/input-gradient accumulation use backend ops. The
  training path retains BPTT caches on-device rather than downloading projection,
  gate, and cell caches, and concatenates timestep outputs directly on-device.
  LSTM and RNN forward/backward sequence slicing and timestep input-gradient
  assembly stay on-device; RNN projection gradients also accumulate on-device
  across steps, and no-BN RNN/CRNN training pre-activation/state caches now
  remain device-resident for backward (the host cache is captured only for
  host-facing forwards). CRNN forward input slicing and output assembly also
  stay on-device. CRNN backward sequence slicing and input-gradient assembly
  also stay on-device; convolution parameter gradients accumulate on-device across
  steps. Recurrent cache/state and parameter-gradient transfers, host
  accumulation/SGD, and some cache handling remain host-side. Targeted host
  RNN/CRNN tests and real-device `oxide_runtime_chain` tests
  `rnn_recurrent_projections_and_training_backward_match_cpu` and
  `crnn_recurrent_convolutions_and_training_backward_match_cpu` pass. LSTM
  device-backward parity also passes, including upstream convolution updates.
- Added an oxide Stair activation kernel matching Darknet's floor-based
  alternating-step formula and parity gradient, including negative odd intervals.
  Verified by the 391-test core library suite, `oxide_parity` (11 passed,
  including finite differences away from integer discontinuities), `oxide_half`
  (13 passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (195 convolution-
  activation pairs), and all 7 real-device `oxide_runtime_chain` tests. Every
  oxide test target also compiled with `MEM=3G ./tools/test_oxide.sh --no-run`.
- Added an oxide PLSE activation kernel with the C++ piecewise-linear sigmoid
  and exact ±4 endpoint gradients. Verified by the 391-test core library suite,
  `oxide_parity` (11 passed, including finite differences and input/output
  derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (189 convolution-activation pairs), and all 7
  real-device `oxide_runtime_chain` tests.
- Added an oxide Lhtan activation kernel with its exact 0.001 tail slopes and
  Darknet's endpoint gradients at 0 and 1. Verified by the 391-test core library
  suite, `oxide_parity` (11 passed, including finite differences and input/output
  derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (183 convolution-activation pairs), and all 7
  real-device `oxide_runtime_chain` tests. Every oxide test target also compiled
  with `MEM=3G ./tools/test_oxide.sh --no-run`.
- Added an oxide Relie activation kernel (leaky ReLU with slope 0.01), including
  the exact zero-point derivative. Verified by `oxide_parity` (11 passed,
  including finite differences and input/output derivative parity), `oxide_half`
  (13 passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (177 convolution-
  activation pairs), and all 7 real-device `oxide_runtime_chain` tests. The
  391-test core library suite also passed.
- Added an oxide HardTanh kernel with Darknet's inclusive clamp and zero
  derivative at exactly -1 and 1. Verified by `oxide_parity` (11 passed,
  including finite differences and endpoint parity), `oxide_half` (13 passed, 1
  ignored), all 13 `oxide_cfg_shapes` tests (171 convolution-activation pairs),
  the real-device chained-convolution activation test, the 391-test core
  library suite, and a no-run build of every oxide test target.
- Added an oxide loggy activation kernel and exact derivative, wiring it through
  cfg parsing, f32/f16 convolution activation paths, and the real-device runtime.
  Verified by `oxide_parity` (11 passed, including finite differences and
  input-vs-output derivative parity), `oxide_half` (13 passed, 1 ignored), all 13
  `oxide_cfg_shapes` tests (165 convolution-activation pairs), all 7 real-device
  `oxide_runtime_chain` tests, and the 391-test core library suite. The oxide
  `--no-run` build compiled every test target.
- Added an oxide Hard Mish activation kernel and exact derivative, including
  Darknet's strict threshold behavior at -2 and 0. Verified by `oxide_parity`
  (11 passed, including finite differences and explicit x=-2), `oxide_half` (13
  passed, 1 ignored), all 13 `oxide_cfg_shapes` tests (159 forward/backward
  convolution-activation pairs), and all 7 real-device `oxide_runtime_chain`
  tests. The core library suite passed (391 tests).
- Added an oxide SELU activation kernel and exact derivative, enabling native
  convolution activation execution. Verified by `oxide_parity` (11 passed,
  including activation finite differences), `oxide_half` (13 passed, 1 ignored,
  including f16 bias-activation), all 13 `oxide_cfg_shapes` tests (including
  153 forward/backward convolution-activation pairs), and all 7 real-device
  `oxide_runtime_chain` tests. `cargo test -j 4 -p darknet-core --lib` passed
  (391 tests).
- Added an oxide ELU activation kernel and exact derivative, enabling native
  convolution activation execution. Verified by `oxide_parity` (11 passed,
  including activation finite differences), `oxide_half` (13 passed, 1 ignored,
  including f16 bias-activation), and the real-device
  `oxide_runtime_chain::chained_conv_activation_stacks_match_cpu_backward` test.
  `MEM=3G ./tools/test_oxide.sh --no-run` also compiled every oxide test target.
- YOLO IoU/GIoU box geometry and objectness/class sigmoid-BCE deltas have
  central finite-difference coverage; the 394-test core library suite passed at
  that milestone. The box test records Darknet's legacy 0.5 width/height
  comparison factor. DIoU/CIoU deltas now have source-derived parity coverage
  for contained, edge-crossing/tied, and horizontal/vertical disjoint boxes;
  all 24 `yolo_`-filtered host tests pass. Do not use naive forward finite
  differences as their C++ reference: `box_diou` uses `(d/c)^0.6`, while `dx_box_iou`
  implements the legacy derivative of `d/c`; CIoU approximates the `alpha`
  contribution.
- f16 standalone batchnorm (via channel-affine), bias + activation, and reverse
  upsample forward are already implemented. Verified by
  `oxide_half::f16_shortcut_avgpool_affine_reverse_match_f64_reference` and
  `oxide_runtime_f16::tiny_net_with_avgpool_local_avgpool_batchnorm_reverse_upsample_matches_f32`
  (13 + 5 tests passed; 1 ignored in the half suite).
- Weighted `[shortcut]` GPU forward composes oxide channel-scale + add. Input and
  source gradients use backend elementwise multiply + shortcut-fit ops. Oxide
  also reduces raw `delta * source` dot products on-device, transferring only
  per-input/per-slot results for host normalization-chain derivatives; parameters
  and SGD remain host-resident. FakeBackend and real-device
  `weighted_shortcut_runtime_matches_cpu_forward_backward_and_sgd` parity pass.
- AMP runtime with dropout and dropblock now has real-device output/mask/backward/
  SGD comparison against f32 (`oxide_runtime_chain` test).
- Deferred optimizer accumulation has a real-device `subdivisions=2` CPU parity
  test across two iterations, including pending-gradient flush.
- Device convolution params support per-owner expert slots for container reuse;
  warm-up and host-update invalidation are covered by the
  `container_convolution_cache_is_keyed_by_owner_and_expert_slot` FakeBackend test.
  MoE computes router pooling/logits/top-k softmax, expert convolutions, and
  per-sample gate blending on Oxide. No-BN router connected-backward,
  device-selected gate columns, expert gate-dot reductions, load-balance plus
  top-k softmax-Jacobian backward, router-pool input-gradient broadcast, and
  expert convolution backward use backend ops. Gates and pooled/expert
  output/activation caches stay device-resident; no gate/dot values cross the
  host in normal Oxide training. Router/expert parameter gradients accumulate
  on-device across backward calls; accumulated weight/bias buffers transfer to
  host only at optimizer flush for SGD/Adam. The dynamic-loss-scale finiteness
  check covers dirty router/expert buffers. BN experts stay on the CPU fallback.
  FakeBackend parity and real-device
  `moe_container_experts_run_oxide_forward_and_device_backward_matches_cpu`
  pass, including two accumulated backward calls and upstream convolution
  updates. The FakeBackend test
  `gpu_moe_device_gradients_participate_in_loss_scale_finiteness_check` verifies
  dirty router/expert accumulators are checked for overflow. `oxide_layers_extra`
  verifies soft/tied top-k, pool and pool-backward, non-multiple gate-dot
  reductions, and router backward parity with load balancing.
- Overlapping-window maxpool backward, including `maxpool_zero_nonmax`, now
  matches central finite differences.
- No-BN RNN and CRNN forward/backward use backend ops: RNN projections and
  CRNN recurrent convolutions, activation gradients, state accumulation, and
  shortcuts. No-BN LSTM forward runs gates/cell products through backend ops,
  including `vector_mul_f32`; its backward now uses a backend gate-derivative
  operation plus connected ops for projection gradients, time-accumulated
  parameters, and recurrent/input-gradient accumulation. Device training keeps
  BPTT caches resident instead of downloading projection/gate/cell caches,
  and timestep outputs are concatenated on-device. RNN/LSTM forward/backward
  sequence slicing and timestep input-gradient assembly also stay on-device;
  RNN projection gradients accumulate on-device across steps; CRNN forward
  input slicing and output assembly also stay on-device. CRNN backward sequence
  slicing and input-gradient assembly also stay on-device, with convolution
  parameter gradients accumulated across steps. Recurrent cache/state and
  parameter-gradient transfers, host accumulation/SGD, and some cache handling
  still cross the host. Batch-normalized recurrent layers remain on CPU backward.
  FakeBackend parity
  and real-device
  `oxide_runtime_chain::lstm_no_batch_norm_forward_and_device_backward_match_cpu`
  compare training updates with CPU, including upstream convolution updates.
- Host Adam now updates standalone batchnorm, local, connected, weighted
  shortcut, MoE router/experts, deconvolutional, GRU/RNN/LSTM/CRNN, and ViT
  parameters with distinct moment slots and per-layer rate scaling. DNOPT2
  sidecars persist host moments while DNOPT1 remains loadable. The ViT reference
  test covers C++ weight order, decay groups, clipping, and reduced mHC rate;
  host resume has a multi-step continuation test.
- `[vit]` reference path: cfg/shape, C++-ordered header-aware weights, seeded init,
  CPU analytical forward/backward + SGD, and GPU f32 host fallback; 23 focused
  tests pass, targeted package check/build pass with `-j 4`.
- Host suites: `darknet-core --lib` 390/390 and `darknet-cli` 18/18 pass. The
  real-device `oxide_runtime_chain` suite passes all 7 tests.
  `cfg_corpus` started but the combined host run hit its 30-minute command timeout;
  this is not evidence of a code failure. Historical `raptor-fp4` C++ commit
  `ceb7c974` was located for MSA and flash/VC source audit.
- `[moe_convolutional]` CPU path (commit `cdf68d00`) — 13 tests green.
- `[laya_marker_gather]` — 10 tests; `[laya_act_head]` — 9 tests;
  `[laya_loss]` — 11 tests; `[laya_encoder]` — 9 tests; `[laya_head]` — 8 tests;
  `darknet-core --lib` 354/354.
- Docs reconciled: `todo.md` now tracks open work only; `eml_convolutional`
  save order verified; `onnx.md` / `gui_iced_plan.md` marked not started.

## Verification commands

```
cargo check -j 4 --workspace && cargo build -j 4 --workspace && cargo run -j 4 -p darknet-cli
MEM=3G ./tools/test_host.sh                      # darknet-core + darknet-cli
MEM=3G ./tools/test_host.sh <filter>             # one filter, much faster
MEM=3G cargo test -j 4 -p darknet-core --lib segmentation   # the fast inner loop
MEM=3G ./tools/test_oxide.sh --test oxide_gemm_f8  # ~50 s
MEM=3G ./tools/test_oxide.sh                      # ~15 min, 18 suites
```

`cfg_corpus` needs ~145 s **alone**; five unrelated cfgs hit their 300 s
per-cfg timeout when the suite shares the machine, so run it on its own before
believing a `cfg_corpus` failure.

## NEXT (in dependency order, nothing blocked)

1. The LAYA family is **complete** (`laya_scorer`, `laya_marker_gather`,
   `laya_act_head`, `laya_loss`, `laya_encoder`, `laya_head`). The `[vit]`
   reference path is now wired; remaining Phase 3A work is flash/vc-attention
   fast paths and `swin`/`transformer`, `tucker_attention`, `mambavision`,
   `kimi_kda`, `minimax_sparse_attention` on the shared LayerNorm/GEMM set.
   **Order matters:** `[laya_scorer]`'s `weights=` is only usable together with
   `[laya_encoder]` + `[laya_head]` (the shared-file offset is computed past
   their blobs), so those two come next even though the plan files them under
   Phase 4. They also need the Phase 3A prerequisite kernel set.
2. Phase 2A is done, including `detect::decode_modern_detections` and the
   `detect-image` merge. Phase 2B `moe_convolutional` has a verified CPU path
   and partial Oxide forward/backward; device gradient accumulation is in place,
   while optimizer-flush transfers, host SGD/Adam, and BN expert backward remain.
   Remaining across the phases:
   `cfg/LegoGears_centernet.cfg` end to end (the cfg lives on `Jev_laya`), then
   2B's `eml_convolutional`, 2C (OBB), 3A/3B (transformer + deformable families,
   shared LayerNorm/GEMM kernel set first) and 4.
3. **Phase 3A prerequisite (2) is complete**: LayerNorm fwd/bwd, the fused
   residual/bias epilogue, the row-softmax backward and the batched-GEMM wrapper
   (`linear_f32`) are all in, each with a host default plus GPU parity tests.
   That unblocks `vit`, `swin`/`transformer`, `tucker_attention`, `mambavision`,
   `kimi_kda`, `minimax_sparse_attention` and the LAYA encoder/head. Still to
   decide when writing those layers: whether the `flash_attention` /
   `vc_attention` fast paths (decision 3) get their own kernels or reuse the
   reference attention path.
4. Phase 2B `moe_convolutional` (weights order verified) has partial Oxide
   training support; optimizer-flush parameter transfers, host SGD/Adam, and BN
   expert backward remain. Then port
   `eml_convolutional` (an activation, not a `LayerKind`; save order **re-read
   2026-10-05**: `input_layer` then `self_layer`, two full conv blocks).
5. Phase 3B deformable family — start with `dcnv4` (real C++ forward and
   kernels; weights order `weights.cpp:428`).
6. Phase 2C OBB, then Phase 4.

## Known evidence gaps (not user questions)

- `[dcnv3]`: forward/backward/update are stubs in **all five** branches that
  carry it (`deform`, `Jev_laya`, `fog`, `jepa`, `depth_3d`), and
  `docs/dcnv3_layer.md` says so. Only its `.weights` order can be ported
  against C++; the numerics would have to come from InternImage. Details and
  the evidence are in `tasks/rust/layers.md`.
- `[contrastive]` detection branch needs the yolo head's per-box class ids;
  `YoloHeadOutput` only carries the raw output tensor.
- DETR is still broken in C++, so Phase 4's `detr_decoder` also has no
  reference numerics.
- No C++ build of any owning branch is available here, so C++ dump parity is
  "NOT RUN" everywhere.

## Task plans not started

Two self-contained task plans in this directory have **no implementation**:

- `tasks/rust/onnx.md` (`darknet-rs export-onnx`): 0/23 task steps done. No
  `src/darknet-core/src/onnx/` module and no `export-onnx` CLI arm exist.
- `tasks/rust/gui_iced_plan.md` (iced DarkMark-compatible GUI): 0/84 task
  steps done. None of `darknet-gui-project` / `darknet-gui-engine` /
  `darknet-gui` exist, and there is no `detect-server` CLI arm.

Both are greenfield; the layer-port work above does not depend on them.

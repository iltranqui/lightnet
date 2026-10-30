# Device completion priority

## Contents

- [Priority and constraints](#priority-and-constraints)
- [Project tree](#project-tree)
- [Components](#components)
- [Implementation sequence](#implementation-sequence)
- [Review](#review)
- [Muon, AMP and GRU continuation](#muon-amp-and-gru-continuation)
- [Lifecycle continuation and next residency contract](#lifecycle-continuation-and-next-residency-contract)
- [ViT residency continuation](#vit-residency-continuation)
- [Swin residency continuation](#swin-residency-continuation)

## Priority and constraints

Operator priority: full backward, native GPU updates, FP16 and every-layer
completion precede the separate CPU detector-versus-/usr/bin/darknet milestone.
Original scope was source only. The user now authorizes Cargo checks/tests,
at most two jobs/test threads and one invocation at a time, under MemoryHigh=3G.
Training and unrelated executable probes remain outside scope.
Make local WIP commits after source-complete slices, with no verified release
tags. Preserve unrelated Latin corpus/example/Cargo.lock work. Work directly
without subagents under tasks/rust/AGENTS.md.

## Project tree

```text
src/
  lightnet-core/src/
    layers/mambavision/{cpu.rs,gpu.rs}
    layers/kimi_kda/{cpu.rs,gpu.rs}
    layers/tucker_attention/{cpu.rs,gpu.rs}
    runtime/
      mod.rs
      gpu_resident_params.rs
      gpu_types.rs
      gpu_state.rs
      gpu_update.rs
      gpu_backward_support.rs
      gpu_f16.rs
  lightnet-cuda/src/lib.rs
  lightnet-cuda-oxide/
    src/{lib.rs,half.rs,kernels/optim.rs}
    tests/{oxide_mambavision.rs,oxide_resident_params.rs,oxide_kimi_kda.rs,oxide_tucker_attention.rs}
tasks/
  todo.md
  lessons.md
  rust/{device_completion_plan.md,source-wip.md,layer-support.md}
```

## Components

- MambaVision CPU defines serialized ranges, decay groups and checkpoint order.
  GPU glue slices resident parameters and concatenates gradients on device in
  that order. Backward must not download parameter gradients.
- KimiKDA uses a biases-first serialized bank with root bias/weight DNOPT
  moments. Many projection/LayerNorm gradient blocks must cover the two
  optimizer arrays exactly once. SGD/AdamW exclude biases from decay; coupled
  Adam includes them, matching the host helper.
- TuckerAttention uses the same biases-first checkpoint slots and decay rules.
  Its eight source factors are packed into connected matrices on device using
  token-layout permutations, with independent head batches for Q/K/V cores.
  Backward applies inverse permutations before resident accumulation.
- gpu_resident_params owns device weights, descending update accumulators,
  Adam moments and decay masks. Host snapshots detect public edits; deferred
  updates keep device values authoritative until explicit synchronization.
  Coverage is native Mamba/Kimi/Tucker SGD/Adam. Muon and networks selecting CPU
  backward retain the existing explicitly reported hybrid path.
- Runtime types/state owns bank allocation, reset, diagnostics and sidecar
  materialization/restoration. DNOPT children/ranges must round-trip; saving
  without prior host sync must still serialize current device moments.
- gpu_update performs one update per eligible layer, preserving learning-rate,
  loss-scale and update gates and skipping the host arm. Momentum/moments and
  resident weights survive forwards and deferred updates.
- Existing route, vector and optimizer backend ops supply slicing, packing,
  decay and updates. Native Oxide finiteness reduction returns a scalar flag;
  backend host defaults remain references, not evidence of device execution.
- Unrun source fixtures cover accumulated gradients, two updates, decay modes,
  deferred sync, loss scale, update gates, sidecar resume and public host edits.
- F16TensorOpsBackend exposes mandatory device widening/narrowing methods;
  Oxide overrides them with existing conversion kernels. gpu_f16 dispatches
  supported Mamba/Kimi/Tucker through explicit f32 arithmetic islands with half
  activation boundaries. Master parameters/moments remain f32. These are not
  native half arithmetic or f16 training; inference never replaces training
  caches. Other half-dispatch gaps remain open.

## Implementation sequence

- [x] Read live source, WIP commits and execution constraints.
- [x] Write flat resident parameters/gradients/moments and native Mamba SGD/Adam integration (uncompiled).
- [x] Write reset, diagnostics, host synchronization and DNOPT state restoration.
- [x] Write native finiteness reduction and unrun end-to-end fixtures.
- [x] Review source diffs, document limits and create local WIP commit `c9feefb` (no Cargo validation).
- [x] Write Kimi/Tucker resident SGD/Adam, separate coupled decay masks and explicit root/child checkpoint slots (uncompiled).
- [x] Write Tucker device factor packing and inverse gradient permutations (uncompiled).
- [x] Write Mamba/Kimi/Tucker half-storage inference with native device casts and explicit f32 arithmetic (uncompiled).
- [x] Write unrun deferred update, gradient, decay, sidecar, cache, gate, reset and cast-boundary fixtures.
- [x] Inspect continuation diffs and record a source-only local WIP checkpoint without release tags (no Cargo validation).
- [ ] Continue full native updates/Muon, remaining backward variants and FP16 coverage.
- [ ] Deferred by user: compile and execute host/Oxide parity gates before certification.

## Review

Source written for Mamba/Kimi/Tucker native SGD/Adam residency, including eager/deferred
steps and sidecar moments. Corrected CPU Mamba accumulation from mathematical
gradients to descending optimizer directions. Updated unrun optimizer fixtures
to select CPU Adam/Muon APIs explicitly and account for loss scaling. Native
Oxide finiteness reduction now downloads one flag. Source diff review only;
no compilation, tests, performance or C++ golden evidence. Muon and CPU-fallback
networks retain their existing hybrid paths. All-layer completion remains open.
Kimi/Tucker retain root bias/weight moment slots; Mamba retains per-block child
weight slots. Native Kimi finiteness guards still transfer scalar flags, not
gradient tensors. Tucker source layouts and their inverse permutations were
reviewed algebraically; no numeric proof is available. Half-storage inference
uses f32 arithmetic islands and preserves the training cache. A resident update
marks host mirrors stale immediately after each layer publication, including
when a subsequent layer fails.

## Muon, AMP and GRU continuation

The operator requested implementing the remaining Muon/FP16/layer work.
The user subsequently authorized serial check/test execution with two jobs.
This source continuation supersedes
the earlier Muon-hybrid-only scope above.

```text
src/
|-- lightnet-core/src/
|   |-- layers/
|   |   |-- gru/
|   |   |   |-- mod.rs
|   |   |   |-- cpu.rs
|   |   |   `-- gpu.rs
|   |   |-- mambavision/gpu.rs
|   |   `-- tucker_attention/gpu.rs
|   |-- muon_runtime.rs
|   `-- runtime/
|       |-- exec_backward.rs
|       |-- exec_accumulate.rs
|       |-- gpu_backward.rs
|       |-- gpu_backward_support.rs
|       |-- gpu_forward.rs
|       |-- gpu_forward_compact.rs
|       |-- gpu_f16.rs
|       |-- gpu_types.rs
|       |-- gpu_state.rs
|       |-- gpu_update.rs
|       |-- gpu_resident_params.rs
|       `-- tests/yolo_3d_runtime_tests.rs
`-- lightnet-cuda-oxide/
    |-- src/
    |   |-- lib.rs
    |   |-- mixed.rs
    |   `-- half.rs
    `-- tests/
        |-- oxide_resident_params.rs
        |-- oxide_connected_amp.rs
        |-- oxide_gru_training.rs
        `-- oxide_runtime_chain.rs
```

- Muon mapping defines serialized matrix views in the existing CPU visitor and
  DNOPT6 order. Its unrun host fixture compares both values and shapes. Views
  must use source storage, not the packed forward-GEMM layout.
- Resident banks own persistent native matrix momentum and AdamW fallback
  state. Updates mask matrix Adam gradients but preserve resumed moments.
  Every layer is published only after staging its complete device result.
  Finite-value diagnostics still transfer scalar flags. Mixed hybrid families
  require eager synchronization; all-resident networks can defer Muon updates.
- gpu_state materializes resident Muon momentum for DNOPT6 saves and explicit
  host synchronization. Loaded state is validated and uploaded lazily. GRU
  uses six child bias/weight moment slots, matching existing host optimizer IO.
- mixed.rs implements dense half-operand tensor-core forward/dgrad/wgrad;
  lib.rs selects it only under AMP. Forward/dX use half storage; gradients of
  parameters and optimizer/master state stay f32. half.rs widening handles AMP
  tensors before the explicit half-inference narrowing boundary.
- GRU host/device code adds exact non-BN BPTT for the source's sigmoid candidate,
  with a forward-owned device tape. Runtime dispatch, gradient accumulation,
  native parameter residency and all three optimizers are wired. Compact and
  half-storage inference read resident masters without replacing training state.
  BN training remains rejected; legacy rolling-stat BN inference is preserved.
- Fixtures cover CPU finite differences, repeated BPTT, native/deferred updates,
  AMP, DNOPT resume, gates/reset and cast overflow. The historical GRU rejection
  test now asserts only the BN restriction. The unrelated YOLO-3D fixture only
  receives the new empty cache field required by its struct constructor.

Implementation sequence and source status:

- [x] Define serialized Muon views and unrun CPU visitor-order fixture.
- [x] Write device-resident Muon momentum, AdamW fallback and DNOPT6 materialization.
- [x] Write native dense AMP training forward/dgrad/wgrad and unrun primitive fixtures.
- [x] Write non-BN GRU exact host/native BPTT, resident SGD/Adam/Muon and compact/half inference.
- [x] Write unrun integration, gradient, gate/reset and resume fixtures.
- [ ] Finish source/diff review and local WIP checkpoint, without a release tag.
- [x] Host compilation and focused GRU (4)/Muon (26) tests passed with two jobs.
- [ ] Resolve broad runtime failures and run the GPU-enabled check/Oxide suites serially.

Remaining full-completion work includes other families' residency/Muon,
RecursiveBlock controller/lattice/BN/body variants, other documented ViT and
Clifford variants, GRU BN training, dense-loss gaps, DSConv variants and YOLO-3D.
Dense AMP supplies real half arithmetic for projections; it does not establish
native half coverage of all scan/attention/normalization operations or all layers.

## Lifecycle continuation and next residency contract

Checkpoint `2ee6840` contains the Muon/AMP/GRU slice, with focused verification
recorded in source-wip.md. The full core invocation passed 1,016 unit tests
and was interrupted during cfg-corpus integration at the user's stop request.
No task-owned Cargo/corpus process remains; resume only when asked. Latest
source edits are checkpointed at `0e93bcb` and uncompiled: Mamba/Kimi/Tucker device tapes move to the returned
GpuForwardState, and resident bank epochs no longer reset on retirement/reload.
Overlapping-forward and stale-reload device regressions are queued for the
next serial test slot. Do not attribute the checkpoint's green evidence to
these later edits.
The tape migration also covers DeformConv, ViT, Swin, WMHF, Clifford, DPT,
DepthAnything and RecursiveBlock, including WMHF/Clifford rolling-stat consumers.
New `oxide_forward_tapes` isolates the lifecycle property from numeric CPU
parity by comparing interleaved and isolated GPU input adjoints/BN statistics.

The DPT source contract needs more than a parameter enum arm. The audit below
guided the partial SGD/Adam residency slice that follows; its remaining
requirements are still open:

- `layers/dpt_head/cpu.rs` declares 66 serialized slots: 32 active matrix
  slots and 28 active bias slots; 12/13 are empty, and 28/29/36/37 contain
  serialized parameters unused by the executed graph and optimizer.
- Native banks must preserve inactive parameters and update accumulators,
  including explicitly populated host accumulators. A zero decay mask alone
  does not freeze an SGD momentum accumulator. Device gradient packing must
  cover the full arena while keeping unused slots zero on backward.
- `runtime/param_updates.rs::{update_dpt_head_sgd,apply_host_adam_dpt}` and
  `runtime/host_adam.rs::validate_dpt_head_adam_state` define per-slot child
  weights/biases moments, not root arrays. Empty/inactive slots must not
  acquire checkpoint moments.
- DPT's Muon host fallback skips matrix Adam updates/moments, unlike the
  common mixer fallback that advances resumed matrix moments with zero new
  gradient. Preserve this distinction; test switching/resume explicitly.
- Before the partial residency slice, forward uploaded every slot, backward
  downloaded each conv parameter gradient, and updates were hybrid. The slice
  removes the SGD/Adam parameter transfers while preserving transpose-conv
storage, source order and update gates. Explicit child-state checkpoint
round-trips are covered. Adam-to-Muon switching now has a named atomic
rejection test for sidecars containing matrix Adam moments.

### DPT SGD/Adam residency slice, 2026-10-26

Implemented and partially validated: resident DPT forward slices the full
serialized arena, backward packs slot gradients without host downloads, and
native SGD/Adam updates use per-child moment mappings. Empty slots 12/13 and
inactive slots 28/29/36/37 are masked from updates and moment materialization;
inactive weights and seeded accumulators are preserved by the update mask.
DPT Muon uses its 32 CPU visitor matrix views, skips matrix Adam moments, and
updates active bias moments on device.

Evidence: host check passed, `runtime::` passed 325/325, gpu-check passed,
focused DPT runtime tests passed 1/1, and `oxide_dpt_head` passed 1/1 (65.57s)
with resident SGD/Adam/Muon. Full `oxide_resident_params` passed 17/17
(1117.58s) after the shared Muon change. The first runtime rerun exposed a
tensor-shape mismatch (323/324); the focused test and full runtime rerun passed
after preserving convolution gradient shapes until flat packing. The Oxide
test checks gradients through an explicit diagnostic and updated weights. A
host regression round-trips Adam and Muon child moments through DNOPT, including
the absence of inactive and Muon matrix moments. A separate regression now
verifies that loading an Adam sidecar containing matrix moments into Muon mode
is rejected atomically. Full layer completion remains open.

## ViT residency continuation

Next family: ViT, following the transformer order in AGENTS.md. First slice:
native supported SGD/Adam residency, then its FP16 dispatch coverage; Muon and
MSA/VC variants retain their existing paths until separately implemented.

```text
src/lightnet-core/src/
  layers/vit/{cpu.rs,gpu.rs}
  runtime/{gpu_resident_params.rs,gpu_forward.rs,gpu_forward_compact.rs}
src/lightnet-cuda/src/lib.rs
src/lightnet-cuda-oxide/
  src/{lib.rs,kernels/vit.rs}
  tests/oxide_vit.rs
```

ViT CPU owns the 16-array serialization order and optimizer reference. Layer
GPU glue owns forward slices, generation-pinned tapes and device gradient
packing. Runtime owns resident state and per-array optimizer publication.
Backend operations own mHC logits/adjoints and sanitize/L2 clipping, including
frozen sinusoidal positions, mHC's 0.1 rate and [-8,8] parameter clamp. Oxide
tests compare CPU/device gradients, eager/deferred updates and checkpoints.

- [x] Implement the 16-array bank and device mHC/gradient clipping operations.
- [x] Wire native full/compact forward and device gradient accumulation.
- [x] Run check, gpu-check, runtime:: and full oxide_vit serially with caps.
- [x] Add ViT half-storage compact dispatch around f32 arithmetic and test before/after deferred resident updates.
- [x] Record evidence and retry the named-path WIP checkpoint; `.git/index.lock` creation is blocked by read-only `.git`.
- [ ] Extend native backward/residency to ViT MSA and VC attention variants; current support gate rejects both.

Validation on the resumed tree: host check passed; gpu-check passed; core
`runtime::` passed **325/325** (corpus filter selected zero); `oxide_vit`
passed **7/7** (364.28s), including dense and flash backward comparisons,
resident SGD/Adam/AdamW and half-storage inference before and after deferred
updates. A focused hybrid Muon test passed **1/1** (23.91s), comparing
host-staged ViT gradients and GPU orthogonalization with CPU; resident ViT
Muon is still open. `oxide_forward_tapes` passed **1/1** (297.20s) before the
half-storage edit; it remains lifecycle-equivalence evidence. No C++ golden or
all-variant completion claim is made. The next transformer family in the queue is Swin.

Resident ViT Muon continuation: the native forward uses the resident f32
master bank for Muon as well as SGD/Adam/AdamW. Five serialized matrix views
match the CPU Muon visitor. Device updates apply the layer-wide gradient
sanitization/clipping and mHC's 0.1 AdamW rate plus `[-8, 8]` clamp. Host check
passed (5.82s), gpu-check passed (5.05s), core `runtime::` passed **326/326**
(22.98s; corpus selected zero), the focused resident Muon case passed **1/1**
(25.50s), and full `oxide_vit` passed **8/8** (458.47s). MSA/VC attention
remain unsupported; this does not complete ViT.

## Swin residency continuation

Supported Swin Muon now retains its resident bank and CPU-ordered matrix views,
including the optional projection. Global gradient clipping and sanitization,
AdamW fallback and the mHC clamp are preserved. Following the documentation
reorganization, host check, gpu-check and the focused Muon Oxide regression
passed; the latter covered deferred updates and DNOPT resume with and without
projection. Fresh post-reorganization full `oxide_swin` passed 9/9 (545.15s),
host `swin` 28/28 and host `runtime::` 332/332.
No C++ golden or full-layer completion is claimed; changes remain uncommitted.

Swin is the next transformer-family residency slice after ViT. Preserve its
variable serialized block order, optional residual projection, relative
position bias and 12-value mHC block. First scope native SGD/Adam residency
for variants already accepted by `layers::swin::gpu::supported`; keep Muon
and unsupported activation fallbacks separate. Add resident diagnostics,
deferred updates and checkpoint coverage before reporting a green slice.

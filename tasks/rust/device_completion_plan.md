# Device completion priority

## Contents

- [Priority and constraints](#priority-and-constraints)
- [Project tree](#project-tree)
- [Components](#components)
- [Implementation sequence](#implementation-sequence)
- [Review](#review)

## Priority and constraints

Operator priority: full backward, native GPU updates, FP16 and every-layer
completion precede the separate CPU detector-versus-/usr/bin/darknet milestone.
Source only: no Cargo check/build/test/run, executable probes or training.
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

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
    runtime/
      mod.rs
      gpu_resident_params.rs
      gpu_types.rs
      gpu_state.rs
      gpu_update.rs
      gpu_backward_support.rs
  lightnet-cuda/src/lib.rs
  lightnet-cuda-oxide/
    src/{lib.rs,kernels/optim.rs}
    tests/{oxide_mambavision.rs,oxide_resident_params.rs}
tasks/
  todo.md
  lessons.md
  rust/{device_completion_plan.md,source-wip.md,layer-support.md}
```

## Components

- MambaVision CPU defines serialized ranges, decay groups and checkpoint order.
  GPU glue slices resident parameters and concatenates gradients on device in
  that order. Backward must not download parameter gradients.
- gpu_resident_params owns device weights, descending update accumulators,
  Adam moments and decay masks. Host snapshots detect public edits; deferred
  updates keep device values authoritative until explicit synchronization.
  Initial coverage is native Mamba SGD/Adam. Muon and networks selecting CPU
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

## Implementation sequence

- [x] Read live source, WIP commits and execution constraints.
- [x] Write flat resident parameters/gradients/moments and native Mamba SGD/Adam integration (uncompiled).
- [x] Write reset, diagnostics, host synchronization and DNOPT state restoration.
- [x] Write native finiteness reduction and unrun end-to-end fixtures.
- [x] Review source diffs, document limits and create local WIP commit `c9feefb` (no Cargo validation).
- [ ] Continue full native updates/Muon, remaining backward variants and FP16 coverage.
- [ ] Deferred by user: compile and execute host/Oxide parity gates before certification.

## Review

Source written for Mamba native SGD/Adam residency, including eager/deferred
steps and sidecar moments. Corrected CPU Mamba accumulation from mathematical
gradients to descending optimizer directions. Updated unrun optimizer fixtures
to select CPU Adam/Muon APIs explicitly and account for loss scaling. Native
Oxide finiteness reduction now downloads one flag. Source diff review only;
no compilation, tests, performance or C++ golden evidence. Muon and CPU-fallback
networks retain their existing hybrid paths. All-layer completion remains open.

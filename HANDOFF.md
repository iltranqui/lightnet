# Device completion handoff

Date: 2026-10-25. Repository: `/home/kerrigan/src/DarkSuite/lightnet`.
Branch at handoff: `main`. Latest implementation commit: `c9feefb`.

## Contents

- [User priority and execution restrictions](#user-priority-and-execution-restrictions)
- [Read first](#read-first)
- [Implementation checkpoints](#implementation-checkpoints)
- [Current implementation](#current-implementation)
- [Code map](#code-map)
- [Next work](#next-work)
- [Source review concerns](#source-review-concerns)
- [Verification status](#verification-status)
- [Unrelated working tree](#unrelated-working-tree)

## User priority and execution restrictions

The current priority is **full backward, native GPU updates, FP16 and
every-layer completion**, per `tasks/rust/AGENTS.md`. The separate milestone
described as a pure-Rust CPU detector with `/usr/bin/darknet` reference
detections/parity comes later. Do not divert back to that oracle milestone.

Standing user instructions:

- **Write source only. Do not run Cargo check, build, test, run, training or
  runtime probes.** The latest priority change did not lift this restriction.
- Make local **WIP commits** when a feature slice is source-complete, clearly
  recording the missing validation. The user explicitly authorized these
  commits despite the AGENTS.md release-on-green rule. Do not make release tags
  or version-map entries for untested WIP. Do not push.
- Work directly without subagents under the Rust handbook's operator override.
- Preserve unrelated changes listed below; stage named paths only.
- Source/diff inspection is allowed. `git diff --check` was used; it does not
  establish compilation, correctness, GPU execution or parity.
- If execution is later authorized: one build/test command at a time, at most
  four jobs and `MemoryHigh=3G`. The session's 3G requirement is stricter than
  older 4G examples in the Rust handbook. Oxide uses the repository wrapper.

The user's latest request was to prepare this Markdown handoff. No additional
implementation should be inferred as having happened after `c9feefb`.

## Read first

1. `tasks/rust/AGENTS.md`: layer completion contract, remaining variants and
   policy constraints. Some version/status paragraphs predate the WIP commits.
2. `src/lightnet-core/src/layers/CONTEXT.md`: layer layout and runtime wiring.
3. `tasks/rust/source-wip.md`: written source and explicit limitations.
4. `tasks/rust/device_completion_plan.md`: current device-first continuation.
5. `tasks/rust/layer-support.md`, `layer-map.md`, `gpu_layers_plan.md`: historical
   support matrix, file locations and remaining queue. Do not mistake old
   tested counts or matrix rows for validation of these new WIP changes.
6. `tasks/rust/lessons.md`, plus local `tasks/lessons.md` and `tasks/todo.md`.
   The latter two also contain independent Latin preparation work.

## Implementation checkpoints

All entries below are **local, uncompiled and untested WIP**.

| Commit | Source scope |
|---|---|
| `c9feefb` | Mamba device-resident SGD/Adam parameters, gradients and moments; deferred sync and DNOPT integration; native scalar finiteness diagnostic; CPU gradient-sign correction and parity-fixture corrections |
| `9713aad` | Bounded static RecursiveBlock full/compact f32 forward/backward and nested optimizer integration |
| `3c39d9a` | Compact f32 inference dispatch for ViT, Swin, TuckerAttention, MambaVision, KimiKDA, DepthAnything, DPT, WMHF and Clifford |
| `f66caf8` | Native transformer/depth runtime continuation: Mamba/Kimi/DPT/DepthAnything, pending Tucker implementation, caches, backward wiring and source fixtures |
| `048c302` | Earlier native WMHF/Clifford and Swin backend work; also untested |

Earlier ViT/Swin WIP exists in history. It is not certified complete by the
later changes. The next agent should inspect `git log` and `git status` before
editing or staging.

## Current implementation

### MambaVision

Native f32 forward/backward source covers depthwise SiLU branches, selective
scan, optional block-top-k MSA, LayerNorm, residuals and FFN. Preserve the CPU
reference's unusual **flat BDT selective-projection indexing**; inserting a
transpose changes the function. Runtime starts from zero scan state; primitive
APIs additionally expose nonzero incoming-state and final-state adjoints.

`c9feefb` adds the first resident-parameter consumer:

- Forward slices weights from one serialized device master buffer.
- Backward packs mathematical gradients on device and accumulates descending
  directions without downloading parameter gradients.
- SGD momentum, Adam moments and updated weights remain on device across
  deferred steps. Masks preserve Mamba's source decay groups.
- Eager updates or `sync_host_weights_from_device()` refresh public host
  parameters. `synchronize_resident_parameter_gradients()` is an explicit
  diagnostic; ordinary native backward does not refresh host train buffers.
- DNOPT save materializes current device Adam moments; restore invalidates the
  resident cache so loaded moments are uploaded on reuse.
- Generation checks reject backward using parameters from before an update.
  Editing a stale host mirror is rejected rather than silently discarding the
  device update. Synchronize before changing public parameters or execution
  mode after deferred updates.
- **Muon remains hybrid.** Networks requiring CPU backward use the existing
  host path; the resident path is enabled only for native SGD/Adam execution.
- Scan/MSA derivative-boundary flags still synchronize with the host.

CPU Mamba accumulation previously added mathematical derivatives to additive
optimizer buffers. It now negates them. A new unrun host fixture pins the
relationship between `BackwardResult.param_gradients` and update directions.

### Other layer source

- **TuckerAttention:** native padded-window factor projections and dense/sparse
  attention forward/backward; parameter gradients/updates remain host-staged.
- **KimiKDA:** native causal delta-rule scan, piecewise key normalization,
  deterministic index-tied top-k gates, projections, shared/routed experts,
  load-balance adjoint and runtime backward. CPU residual input adjoint was
  corrected. Final parameter updates remain host-staged.
- **DPT:** native four-tap neck/head composition and reverse device tape for
  convolutions, transposed convolutions, resizing, residuals and activations.
  Parameter gradients stage to existing SGD/Adam/hybrid-Muon arms.
- **DepthAnything:** frozen native f32 encoder and full input adjoint, including
  channel-major patch convolution, RGB normalization, CLS/position embedding,
  flash attention, exact-erf GELU and repeated/reordered taps. Head dimensions
  above 128 retain CPU fallback. Frozen position interpolation is host weight
  preprocessing. Parameters stay frozen; preserve Cpp-mode no-op backward.
- **RecursiveBlock:** native static recurrence for non-BN convolution/connected
  bodies, shared gradients over all loops, injection and residual scaling.
  Controllers, lattice, BN and other body types remain CPU fallbacks. Host math
  is still in `runtime/exec_accumulate.rs`; layer co-location is incomplete.
- **Compact f32:** the nine layer families named in the checkpoint table now
  reuse inference paths with last-use output release. This is independent of
  half-storage inference and must not overwrite training caches.
- **FP16:** existing supported-layer inference and convolution AMP training
  infrastructure remain. No new every-layer FP16 completion was implemented
  in this continuation. New transformer/depth paths still have half-dispatch
  gaps. Ops that widen to f32 must not be reported as native half arithmetic.

## Code map

```text
HANDOFF.md
src/lightnet-core/src/
  layers/
    mambavision/{cpu.rs,gpu.rs}
    kimi_kda/{cpu.rs,gpu.rs}
    tucker_attention/{cpu.rs,gpu.rs}
    dpt_head/{cpu.rs,gpu.rs}
    depth_anything/{cpu.rs,gpu.rs}
    recursive_block/{mod.rs,gpu.rs}
    vit/gpu.rs
    swin/gpu.rs
    wmhf/gpu.rs
    clifford/gpu.rs
  runtime/
    gpu_resident_params.rs
    gpu_forward.rs
    gpu_forward_compact.rs
    gpu_backward.rs
    gpu_backward_support.rs
    gpu_update.rs
    gpu_state.rs
    gpu_types.rs
    gpu_params.rs
    gpu_f16.rs
    exec_accumulate.rs
    param_updates.rs
src/lightnet-cuda/src/
  lib.rs
  mambavision.rs
  kimi_kda.rs
  host_ops/tucker_attention.rs
src/lightnet-cuda-oxide/
  src/
    lib.rs
    mambavision.rs
    kimi_kda.rs
    tucker_attention.rs
    kernels/{mambavision.rs,kimi_kda.rs,tucker_attention.rs,optim.rs,vit.rs}
  tests/
    oxide_resident_params.rs
    oxide_mambavision.rs
    oxide_kimi_kda.rs
    oxide_tucker_attention.rs
    oxide_dpt_head.rs
    oxide_depth_anything.rs
    oxide_recursive_block.rs
```

## Next work

1. Review the newest resident update code by source inspection. Keep execution
   prohibited. The plan's source review/commit checkbox is complete at
   `c9feefb`; numeric validation is not.
2. Finish native parameter/optimizer residency and Muon integration across the
   remaining families. Extend the resident abstraction only after defining
   each layer's serialization, decay rules and optimizer-state mapping.
3. Complete remaining native backward variants: RecursiveBlock controller,
   lattice, BN and other bodies; documented ViT variants, Clifford Fourier,
   act-head training/loading, GRU and other remaining layer gaps in AGENTS.md.
   Keep any fallback explicit and do not mark the whole layer complete.
4. Complete FP16 coverage with FP32 master/optimizer state and the proper
   device forward/backward/update lifecycle. Distinguish half storage,
   mixed-precision f32 islands and actual half/tensor-core math.
5. Resolve remaining CPU contracts needed for every-layer completion, including
   YOLO-3D loss/decode/evaluator and DSConv multi-template variants. The handbook
   also records missing reference numerics/broken C++ for some blocked kinds.
6. Continue local WIP checkpoints. Update the source ledger without promoting
   unrun fixtures to passed evidence. No release tags or pushing.

## Source review concerns

- The flat resident bank currently materializes only Mamba parameters and its
  per-block DNOPT child `weights` moments. It is **not yet a general adapter for
  Kimi, DPT, Tucker, ViT or Swin**.
- Tucker factors require packing/unpacking permutations. Kimi has separate
  bias/weight arrays. DPT has fixed slot indices and unused slots. Do not apply
  Mamba's range/sidecar assumptions to those layers unchanged.
- Decay rules differ: Mamba non-weight blocks explicitly disable decay;
  `apply_host_adam_non_weight` elsewhere only disables decoupled decay. Preserve
  the relevant CPU rule for SGD, coupled Adam and AdamW separately.
- CPU seeded backward does not automatically apply GPU `loss_scale`. The new
  resident fixtures explicitly scale CPU seeds and adjust CPU SGD arguments.
  CPU Adam/Muon require their dedicated entry points; `update_weights()` is
  SGD. Earlier unrun optimizer fixtures were corrected accordingly.
- Resident update gates, gradient reset versus SGD momentum, moment-sidecar
  resume, public host edits and cache generations are all source-written but
  unproven. Inspect their RefCell/borrow lifetimes and failure behavior.
- A native primitive is insufficient: verify the backend trait override and
  runtime dispatch actually select it. Host defaults can hide missing wiring.
- The Oxide finiteness implementation scans device data and downloads one flag;
  it still synchronizes. No speed claim was made.
- The old support matrix and version numbers are historical. Use the current
  source and WIP ledger; no full CPU/GPU/C++ parity claim is justified.

## Verification status

**No Cargo checks, builds, tests, runtime probes, training or benchmarks ran for
these WIP commits.** Source review and staged `git diff --check` passed. That is
the entire current verification claim.

Unrun fixtures cover forward/input/parameter gradients, repeated accumulation,
SGD/Adam/Muon update selection, compact inference between forward/backward,
resident deferred updates, loss scaling, update gates, reset, stale state,
DNOPT resume and finiteness tails.

Only after the user lifts the restriction, follow the repository's host and
Oxide wrappers and execute the relevant filters/suites with the memory/job
caps. Do not paste old passing test counts into this continuation.

`/usr/bin/darknet` exists as a file on this machine; it was not executed or
identified as a compatible oracle. C++ detection fixtures/goldens are not part
of the current evidence and are lower priority than device completion.

## Unrelated working tree

At handoff, the implementation commits were clean; these independent edits
remained. Do not discard or include them in layer commits:

```text
 M Cargo.lock
 M src/lightnet-examples/CONTEXT.md
 M src/lightnet-examples/Cargo.toml
?? datasets/
?? docs/LLM_PLAN.md
?? src/lightnet-examples/examples/
?? tasks/lessons.md
?? tasks/todo.md
?? tools/latin_pretraining.md
?? tools/prepare_latin.py
?? tools/test_prepare_latin.py
```

The local tasks files include session priority/lesson notes as well as Latin
work; retain them. Refresh `git status --short` before continuing, since another
session may change the working tree.

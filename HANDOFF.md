# Device completion handoff

Updated: **2026-10-26**. Branch: `main`. Latest code: **`627d115`**.

## Stop handoff contents

- [Stopped state](#stopped-state)
- [Latest checkpoints and results](#latest-checkpoints-and-results)
- [First work on explicit resume](#first-work-on-explicit-resume)
- [Resume commands](#resume-commands)
- [Historical checkpoint notes](#historical-checkpoint-notes)

## Stopped state

**Explicitly resumed by the user, 2026-10-26.** The earlier stopped
Cargo/cfg-corpus session `61558` is historical; do not poll that handle.
Current resume evidence at `0e93bcb`: host check passed for core/CLI;
`PKGS=lightnet-core` host `runtime::` passed **324 tests, 0 failures**;
`gpu-check` passed. No compile fixes were needed. The runtime filter selected
zero corpus tests; it does not establish corpus or full-suite success.
Both resident lifecycle regressions passed across Mamba/Kimi/Tucker:
`resident_overlapping_forwards_keep_their_own_backward_tapes` **1/1** (84.86s)
and `resident_reload_cannot_reuse_an_old_forward_generation` **1/1** (72.91s).
`oxide_forward_tapes` passed **1/1** (333.34s) across the other eight native
families (lifecycle equivalence, not independent numeric parity). Commands
remain serial with two build jobs, one GPU test thread and MemoryHigh=3G.
Full CLI tests passed **36/36** (0.44s). Full `oxide_resident_params` passed
**17/17** (1278.44s), a fresh aggregate result at `0e93bcb` superseding the
earlier partial resident reruns. `oxide_mambavision` passed **2/2** (98.50s).
`oxide_kimi_kda` passed **3/3** (100.85s); `oxide_tucker_attention` passed
**3/3** (175.91s). `oxide_vit` passed **5/5** (258.35s); `oxide_swin` passed
**5/5** (224.26s); `oxide_wmhf` passed **8/8** (247.76s).
Initial `oxide_clifford`: **4 passed, 1 failed** (220.09s);
`clifford_native_adam_update_matches_cpu` reported updated parameter 0
`1.1428108 != 1.1332897`. One root-cause pass found its fixture called CPU
SGD unconditionally while the GPU selected Adam from config. Corrected only
`src/lightnet-cuda-oxide/tests/oxide_clifford.rs` to call CPU Adam for Adam
cases; tolerances and production code are unchanged. Full suite rerun:
**5/5 passed** (216.17s). The fix is validated but uncommitted: named-path
`git add` failed with `.git/index.lock: Read-only file system`. No commit/tag
was created; this environment has a read-only `.git` and no approval escalation.
`oxide_dpt_head` passed **1/1** (59.33s), covering the original host-staged
SGD/Adam/Muon path. The next DPT SGD/Adam resident slice also passes
`oxide_dpt_head` **1/1** (63.58s) after host check, `runtime::` **324/324**,
and gpu-check. It compares explicit resident gradient diagnostics and updated
weights across SGD/Adam/Muon. A host regression also seeds inactive-slot
accumulators and verifies bitwise preservation and the absence of matrix Adam
moments in Muon. The same host regression saves and reloads Adam and Muon
child moments through DNOPT, comparing all restored DPT child slots. The full
suite was rerun after resident Muon changes:
`oxide_resident_params` passed **17/17** (1117.58s). Explicit DPT DNOPT
At the time of this earlier checkpoint, Adam-to-Muon switching remained open;
DPT is not complete. The first
runtime rerun found a gradient tensor shape mismatch (323/324); preserving
convolution tensor shapes until flat arena packing fixed it, and the focused
DPT runtime test passed 1/1. `oxide_depth_anything`
passed **2/2** (63.13s). Initial `oxide_recursive_block` failed **0/1**
(22.47s): shared parameter accumulation element 0 was
`-0.008424982 vs 0.008424982`. One source pass identified an extra native
negation: nested CPU convolution/connected buffers add the returned gradient,
but native recurrence subtracted it. Changed the two accumulation loops in
`src/lightnet-core/src/layers/recursive_block/gpu.rs` to add, preserving CPU
semantics and tolerances. Verification restarted at host check, then runtime::,
gpu-check and the full recursive suite. Host check passed (4.44s), runtime::
passed (324/324), gpu-check passed (4.41s), and the full recursive suite
passed **1/1** (59.48s). `git diff --check` also passed. DPT SGD/Adam residency
resident SGD/Adam/Muon is now partially implemented and validated; explicit
DPT optimizer checkpoint save/reload coverage remains open.
Remaining numeric suites/corpus stay pending, corpus last. Commit remains
blocked by read-only `.git`; no bypass attempted.

Latest explicit-resume continuation: at session entry `git diff --stat
0e93bcb` reported 26 tracked files, **1,059 insertions and 168 deletions**,
before the ViT half-storage and evidence-note edits. Fresh `oxide_forward_tapes` passed
**1/1** (297.20s). ViT supported native SGD/Adam/AdamW residency and compact
f16 dispatch were added; host check and gpu-check passed, `runtime::` passed
**325/325** (cfg-corpus selected zero), and full `oxide_vit` passed **7/7**
(364.28s), including f16 storage-boundary checks before/after deferred updates.
The half path uses f32 ViT arithmetic and resident f32 masters. MSA/VC backward,
resident ViT Muon and all-layer completion remain open. The existing hybrid
ViT Muon path later passed its focused CPU-comparison test (1/1, 23.91s). The named-path DPT commit retry
again failed at `git add` with `.git/index.lock: Read-only file system`; no
lock was inspected or removed. Full CLI/layer reruns after this continuation,
cfg-corpus and diff whitespace validation remain outstanding.

At `627d115`, DPT Adam-to-Muon DNOPT switching is covered as an intentional
rejection when the Adam sidecar contains matrix moments. The host regression
verifies that the failed load leaves Muon weights, optimizer step, Adam state
and Muon state unchanged. Host check passed (4.95s), gpu-check passed (3.96s),
`PKGS=lightnet-core runtime::` passed **326/326** (20.03s; corpus filter
selected zero), and `oxide_dpt_head` passed **1/1** (68.88s). A requested fresh
`oxide_forward_tapes` run passed **1/1** (324.59s). Hybrid ViT Muon coverage:
`cargo xtask oxide --test oxide_vit
vit_gpu_muon_hybrid_backward_and_adamw_update_match_cpu` passed **1/1**
(23.91s), comparing host-staged parameter gradients and GPU orthogonalization
against CPU. It does not establish resident ViT Muon updates.

The substantive objective remains full native GPU backward/updates, Muon,
FP16 training and every-layer completion under `tasks/rust/AGENTS.md`.
It is not complete. The standalone CPU detector/oracle milestone stays lower
priority. No training, benchmarks, release tags or pushing occurred.

`627d115` is now HEAD; the orchestrator committed the DPT and ViT residency
slice. Preserve the unrelated root
Cargo.lock, examples, Latin/dataset files and untracked task notes listed at
the bottom.

## Latest checkpoints and results

| Checkpoint | Scope and validation boundary |
|---|---|
| `0e93bcb` | Eleven native cache families now own tapes in GpuForwardState; WMHF/Clifford rolling-stat readers use that forward; resident generations are monotonic across retirement/reload. Host check, runtime filter (324/324), GPU-enabled check and two focused resident lifecycle GPU tests passed on explicit resume. Other gates pending. |
| `2ee6840` | Resident Muon, dense FP16/BF16 AMP, exact non-BN GRU CPU/native BPTT and resident SGD/Adam/Muon, plus verification-driven runtime/Mamba/Swin fixes. Evidence below. |
| `eaeb292` | Earlier Kimi/Tucker residency and mixer half-storage source. |

Historical passing results below precede `0e93bcb`; current resume results
are recorded above and must be kept separate:

- Host check passed for core/CLI; GPU-enabled check passed during the preceding implementation.
- Host GRU filter: 4 passed. Muon: 26 passed. Runtime filter: 324 passed.
- Swin two-/three-head finite-difference regression: 1 passed.
- Mamba host filter after its CPU derivative correction: 19 passed.
- Oxide dense AMP: 3 passed (f16/bf16 operands, ragged GEMMs, f32 gradients/master/state, refreshed operands and overflow).
- Oxide GRU training: 1 passed across SGD/Adam/Muon, AMP off/on, repeated BPTT, deferred steps and resume.
- Initial resident suite: 12 passed, 3 failed. After fixing the CPU Mamba reference, focused Muon rerun passed 3/3 and the separate SGD/Adam gate rerun passed 1/1. All three original failures now pass; the twelve earlier passing cases were not rerun after that correction. Do not report a second full green suite.
- Full core invocation at `2ee6840`: **1,016 unit tests passed**. Subsequent cfg-corpus integration was **interrupted by user request**, without an aggregate result. Its termination is not a numeric test failure or a pass. Full host-suite green remains unproven. Full CLI tests have not run.
- Source whitespace checks passed before the latest checkpoint. No compiled-C++ golden, convergence or speed evidence exists.

Verification fixed missing lazy device training-buffer allocation/invalidation,
fake routing metadata, fallback-specific fixtures and two CPU adjoint bugs.
Swin Q/K/V caches now use compact per-head strides. Mamba scan backward now
reads C from `cache.x_proj`, the saved per-token selective projection, rather
than the parameter blob. Its new regression failed before the fix (0.10833
numerical derivative versus zero analytic derivative); GPU scan math was
already correct. No tolerances were loosened.

## First work on explicit resume

Read `tasks/rust/AGENTS.md`, `tasks/rust/source-wip.md`,
`tasks/rust/device_completion_plan.md` and the layer CONTEXT file first.
Work directly without subagents. Preserve **at most two build jobs/test
threads, MemoryHigh=3G, one Cargo check/test command at a time**. GPU suites
use one test thread; cap Rayon at two. The earlier source-only restriction
is superseded by the user's check/test authorization, but the current stop
request takes precedence until an explicit resume.

1. Compile `0e93bcb` first, then rerun host runtime checks. It changes cache
   fields, signatures, constructors and generation allocation across eleven
   native families. The interrupted corpus binary was compiled before these edits.
2. Run new resident lifecycle regressions:
   `resident_overlapping_forwards_keep_their_own_backward_tapes` and
   `resident_reload_cannot_reuse_an_old_forward_generation` in
   `oxide_resident_params`. They cover input/parameter gradients, repeated
   accumulation and stale-reload rejection for Mamba/Kimi/Tucker.
3. Run `oxide_forward_tapes`, a new suite covering DeformConv, ViT, Swin,
   WMHF, Clifford, DPT, DepthAnything and RecursiveBlock. It compares
   interleaved versus isolated GPU input adjoints and WMHF/Clifford BN stats.
   This is lifecycle equivalence, not independent CPU/C++ numeric parity.
4. Run the relevant full layer/resident suites and full CLI tests serially.
   Schedule a separate full core/corpus rerun; the handbook records prior
   corpus OOM/timeout cases and warns it may take over an hour. Do not reuse
   a narrow result to claim a green full suite.
5. Continue the actual remaining layer/FP16/residency queue. DPT resident
   SGD/Adam/Muon and child-state save/reload are covered. At `627d115`, an
   Adam-to-Muon mode switch has a named test: loading is rejected when the
   Adam sidecar contains matrix moments, with receiver state unchanged. The
   DPT residency audit in device_completion_plan.md identifies 66 serialized
   slots, 32 active matrices, 28 active bias slots, empty 12/13 and inactive
   28/29/36/37. Preserve inactive parameters, accumulators and per-slot DNOPT
   moments; DPT remains partial, not complete.

Other open requirements include other families' residency, RecursiveBlock
controllers/lattice/BN/body variants, documented ViT/Clifford variants, GRU BN,
dense-loss gaps, broader half arithmetic/dispatch, DSConv multi-template and
YOLO-3D loss/decode/evaluator/device work. Keep act-head/reference gaps explicit.
Dense AMP is real half-operand projection math with f32 master/optimizer state;
scan/attention/normalization islands and half storage do not establish
every-layer native-half completion.

## Resume commands

Only after explicit resume. Run each command to completion before the next.
These shell functions retain explicit caps even when invoking the xtask launcher:

```sh
ln_host() {
  systemd-run --user --scope -p MemoryHigh=3G env JOBS=2 CARGO_BUILD_JOBS=2 MEM=3G RUST_TEST_THREADS=2 RAYON_NUM_THREADS=2 cargo run --quiet --package xtask -j 2 -- "$@"
}
ln_gpu() {
  systemd-run --user --scope -p MemoryHigh=3G env JOBS=2 CARGO_BUILD_JOBS=2 MEM=3G RUST_TEST_THREADS=1 RAYON_NUM_THREADS=2 cargo run --quiet --package xtask -j 2 -- "$@"
}
ln_host check
PKGS=lightnet-core ln_host test runtime:: --test-threads=2
ln_host gpu-check
ln_gpu oxide --test oxide_resident_params -- resident_overlapping_forwards
ln_gpu oxide --test oxide_resident_params -- resident_reload_cannot
ln_gpu oxide --test oxide_forward_tapes
PKGS=lightnet-cli ln_host test --test-threads=2
```

The explicit `cargo run --package xtask -j 2 -- ...` launcher avoids a hook
that previously misread the alias's job cap. Oxide still runs through the
repository wrapper. Its standalone lockfile can acquire generated ordering
changes; inspect them without touching the unrelated root Cargo.lock change.

## Historical checkpoint notes

The material below preserves earlier source-only checkpoints and code context.
The stop state, authorization and evidence above supersede historical claims
that verification was prohibited or that all WIP was untested.

Date: 2026-10-25. Repository: `/home/kerrigan/src/DarkSuite/lightnet`.
Branch at handoff: `main`. Original implementation checkpoint: `c9feefb`.
Later source continuations are described below; consult `git log` for the latest WIP.

## Contents

- [User priority and execution restrictions](#user-priority-and-execution-restrictions)
- [Read first](#read-first)
- [Continuation after this handoff](#continuation-after-this-handoff)
- [Muon, AMP and GRU source continuation](#muon-amp-and-gru-source-continuation)
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

- **Latest authorization:** the user now permits Cargo checks and tests, with
  at most **two threads** and only **one Cargo xtask check/test command at a
  time**. Keep MemoryHigh=3G. The source-only restriction recorded below is
  historical; training and unrelated executable probes remain outside scope.
- Make local **WIP commits** when a feature slice is source-complete, clearly
  recording the missing validation. The user explicitly authorized these
  commits despite the AGENTS.md release-on-green rule. Do not make release tags
  or version-map entries for untested WIP. Do not push.
- Work directly without subagents under the Rust handbook's operator override.
- Preserve unrelated changes listed below; stage named paths only.
- Source/diff inspection is allowed. `git diff --check` was used; it does not
  establish compilation, correctness, GPU execution or parity.
- For authorized verification: one check/test command at a time, at most
  two jobs and `MemoryHigh=3G`. The session's 3G requirement is stricter than
  older 4G examples in the Rust handbook. Oxide uses the repository wrapper.

The original handoff ended at `c9feefb`. The user subsequently requested source
implementation of the device-completion priority; see the continuation below.

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

## Continuation after this handoff

2026-10-25, source-only WIP following the user's explicit implementation request:

- KimiKDA and TuckerAttention now use resident serialized parameters,
  descending gradient accumulators, SGD momentum and Adam moments on the native
  SGD/Adam paths. Both map DNOPT moments to root biases/weights, while Mamba
  retains per-block child weights. Separate coupled-decay masks preserve each
  CPU optimizer's bias decay rules.
- Tucker factors and gradients are packed/unpacked on device using token-layout
  permutations, including head-major Q/K/V cores. Native backward no longer
  downloads parameter gradients on these resident paths. Kimi finiteness guards
  retain scalar flag synchronization. Muon remains explicitly hybrid.
- Mamba/Kimi/Tucker now support half-storage inference around device f32
  arithmetic islands. Mandatory backend casts call existing Oxide kernels;
  parameters/moments stay f32 and inference does not replace training caches.
  Native half arithmetic and FP16 training remain open.
- Unrun fixtures cover repeated gradients/updates, coupled versus decoupled
  decay, deferred inference, Kimi DNOPT resume, stale caches/edits, gates/reset,
  Tucker multi-head/asymmetric-factor permutations and half conversion edges.
- A layer publication marks host parameters stale immediately, even if a later
  resident layer fails. Source inspection and `git diff --check` only; no Cargo,
  build, test, runtime, training or benchmark execution. Check `git log` for the
  local WIP checkpoint; no release tag or version-map entry is appropriate.

The original per-layer descriptions below describe the earlier checkpoint;
use this continuation and `tasks/rust/source-wip.md` for current source scope.

## Muon, AMP and GRU source continuation

The follow-up implementation request extends the resident banks to native
Muon for Mamba/Kimi/Tucker and non-BN GRU. Matrix momentum, Newton-Schulz,
AdamW fallback and final parameters remain on device. DNOPT6 saves materialize
current device momentum, including before host-weight synchronization. Fully
resident networks can defer Muon; other families retain eager hybrid updates.
Scalar finiteness guards still synchronize.

Oxide dense forward/dgrad/wgrad now use half operands and tensor-core GEMMs
under AMP. Forward/dX use half storage; parameter gradients, master weights
and optimizer state remain f32. Mixer scans and other unsupported operations
can remain f32 islands. This is real dense AMP training source, not all-layer
native-half completion or measured tensor-core execution.

Non-BN GRU now has exact CPU BPTT, native device training forward/backward,
forward-owned caches, resident SGD/Adam/Muon, compact and half-storage inference.
Its sigmoid candidate and time-major sequence semantics match existing source.
GRU BN training remains rejected; rolling-stat BN inference is preserved.

At the source-only checkpoint, the new fixtures were unrun. CPU finite differences cover every GRU input/parameter;
Oxide sources cover dense AMP, Muon order/gates/resume, full GRU accumulation,
deferred steps and optimizer resume. No Cargo/check/build/test/runtime/training
or benchmark execution occurred. Source review and `git diff --check` are the
only verification. See `tasks/rust/device_completion_plan.md` for the exact map
and remaining queue. Keep local WIP commits untagged and preserve unrelated work.

The user subsequently authorized check/test execution with two jobs, serially.
Host `cargo xtask check` passed for lightnet-core/lightnet-cli. Focused tests
passed: GRU 4/4, Muon 26/26 and the broad runtime filter 324/324. Fixes include
lazy training-buffer allocation/invalidation, fake backend routing metadata,
explicit fallback fixtures and Swin's multi-head CPU cache stride. Device
validation subsequently stopped by user request; see the latest evidence above.

Latest GPU evidence: `gpu-check` passed; `oxide_connected_amp` passed 3/3,
and `oxide_gru_training` passed 1/1 across all optimizer/AMP combinations.
The initial resident mixer suite passed 12/15. Its failures exposed a CPU
Mamba scan-adjoint bug (C read from model parameters instead of the cached
token projection). A finite-difference regression caught the error before the
fix; all 19 Mamba host tests and the three focused resident Muon GPU tests
passed after the correction. The separate SGD/Adam gate rerun also passed 1/1.
All three original failures now pass; this is focused rerun evidence, not a
second full resident-suite run.
Verification remains serialized,
with two build jobs, one GPU test thread and MemoryHigh=3G.

Local WIP checkpoint: `2ee6840`. Its full core unit suite passed 1,016 tests;
cfg-corpus integration was subsequently interrupted by the user's stop request;
CLI full tests remain pending.
After that checkpoint, an uncompiled lifecycle continuation moves the three
resident mixer tapes into each GpuForwardState and adds a network-wide
monotonic bank generation. Added device regressions cover interleaved forward
tapes and stale generations after sidecar reload. Run those after the active
corpus invocation ends; do not overlap Cargo commands.
The ownership change also covers the other eight network-global native tape
families, including WMHF/Clifford BN-stat reads. New `oxide_forward_tapes`
checks interleaving against isolated GPU execution for all eight. This later
source remains uncompiled and unrun; it is not part of checkpoint `2ee6840`.

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

1. On explicit resume, follow the latest verification queue above. The
   historical source review/commit checkbox was complete at
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

- The flat resident bank now materializes Mamba parameters/per-block DNOPT
  child moments and Kimi/Tucker biases-first arrays/root DNOPT moments. It is
  **not yet an adapter for DPT, ViT or Swin**.
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

At the original source-only checkpoint, only source review and staged
`git diff --check` had run. Later authorized evidence and its exact checkpoint
boundaries are recorded at the top of this handoff; this historical paragraph
must not be used as the current verification status.

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

# Native transformer and depth source continuation

## Contents

- [Execution boundary](#execution-boundary)
- [Written source](#written-source)
- [Resident Muon, dense AMP and GRU training](#resident-muon-dense-amp-and-gru-training)
- [Evidence and remaining work](#evidence-and-remaining-work)
- [ViT residency and half-storage follow-up](#vit-residency-and-half-storage-follow-up)

## Execution boundary

Latest authorization update: the user permits Cargo checks/tests with at most
two threads, one check/test command at a time, under MemoryHigh=3G. Historical
source-only statements below describe earlier checkpoints. Training remains
outside scope. Host cargo xtask check passed for lightnet-core/lightnet-cli;
focused test and GPU evidence will be recorded separately as it becomes available.

2026-10-25: user requested host/device code per AGENTS.md, then explicitly
prohibited Cargo check, build and run and requested source only. No compilation,
tests, runtime, training or benchmarks were run. Later the user authorized local
WIP commits for features without Cargo validation. No release tag or version-map
entry is appropriate. The unrelated Latin corpus, examples and Cargo.lock edits
remain outside this checkpoint.

## Written source

- TuckerAttention: retain the pending native padded-window/factor/attention
  forward and backward implementation and its backend overrides. Uncompiled.
- MambaVision: connect pending device depthwise/SiLU, selective scan and MSA
  kernels to the backend trait; compose both depthwise branches, scan, residual,
  LayerNorm and FFN in the runtime, preserving the CPU flat-BDT selective
  projection. Backward covers all serialized parameter blocks, activation/norm,
  causal/centered convolution and scan derivatives. Boundary flags are read
  before differentiating. Runtime initial state is zero, as before; nonzero
  state and final-state adjoints are available at the primitive boundary.
- KimiKDA: add host primitive references, native causal scan forward/backward,
  piecewise normalized keys, and index-stable top-k gates. Compose projections,
  shared/routed MLPs, load-balance router derivative, residuals and LayerNorm.
  Fix the CPU input adjoint to include the LN2/expert derivative through
  pre_expert's identity edge. Add an isolated finite-difference fixture.
- DPT: implement native four-tap projection/reassembly/neck/fusion/head forward
  and a reverse device tape for conv, transposed conv, residual, ReLU, tap
  routing and align-corners bilinear gather. Stage complete 66-slot parameter
  gradients for existing SGD/Adam and hybrid device-Muon. Unused source slots
  stay zero. Preserve the Cpp-mode backward no-op.
- DepthAnything: native channel-major patch projection via convolution, RGB
  normalization, CLS insertion, positional embedding, flash attention,
  layer scales, exact-erf GELU, repeated/reordered taps, and full input adjoint.
  Frozen positional weights are bicubically interpolated on upload using the
  CPU weight-only helper. Native flash head dimensions are <=128; larger heads
  use the explicit CPU fallback. Encoder weights remain frozen under SGD, Adam
  and Muon. Preserve Cpp-mode no-op backward and the CPU inclusive clamp mask,
  including NaN/boundary behavior, through a dedicated tap kernel.
- Runtime: add/cache/initialize forward and backward dispatch consistently;
  retain CPU caches when another layer selects whole-network host backward.
  Initialize gradient buffers by array lengths, fixing repeated backward
  accumulation for networks with no convolution-owned buffers. Propagate
  errors from host-staged SGD dispatch instead of returning errors from a
  unit-returning function.
- Compact f32 continuation: reuse inference-only layer paths for ViT, Swin,
  TuckerAttention, MambaVision, KimiKDA, frozen DepthAnything, DPT, WMHF and
  Clifford. Keep output last-use release accounting and full-forward fallback
  for documented unsupported variants. Half-storage guards remain separate.
  Source fixtures compare CPU/full/compact outputs and place compact inference
  between training forward/backward to pin preservation of training caches.
- RecursiveBlock continuation: add bounded native full/compact f32 forward and
  backward for static non-BN convolution/connected bodies, including input
  injection, residual/body scales and all shared-weight loop adjoints. Stage
  gradients into existing nested SGD/Adam/hybrid-Muon arms. Controllers,
  lattice, BN and other body layers now use an explicit host fallback rather
  than a hard GPU rejection. Their native coverage remains open. Host math
  still lives in runtime/exec_accumulate.rs; co-location is not complete.
  oxide_recursive_block fixtures cover three loops, two backward accumulations,
  two updates, compact inference and both convolution and connected bodies.
- Source fixtures: oxide_mambavision, oxide_kimi_kda, oxide_dpt_head and
  oxide_depth_anything exercise layout, state adjoints, routing, repeated
  accumulation, input/parameter gradients and updates. Existing depth host
  dispatch tests now expect native f32 support and keep half inference rejected.
- Residency continuation: native Mamba SGD/Adam use a flat serialized device
  parameter bank, device-sliced forward weights and device-packed gradient
  accumulation. Weights, SGD momentum and Adam moments survive deferred steps.
  Eager steps and explicit synchronization materialize host mirrors; saving
  DNOPT materializes current device moments, and loading state refreshes caches.
  Public edits to stale mirrors and backward after a parameter-generation
  change return errors. Muon keeps its existing hybrid path. Corrected CPU
  Mamba accumulation to negate mathematical gradients before additive updates.
  Existing unrun parity fixtures now select the CPU Adam/Muon entry points.
  Added oxide_resident_params fixtures for loss-scaled accumulated backward,
  SGD/Adam/AdamW, update gates, deferred inference, sidecar resume, reset,
  stale mirrors and stale training caches. Oxide finiteness diagnostics reduce
  on device and transfer one scalar flag, not the inspected gradient tensor.
- Kimi/Tucker residency continuation: native SGD/Adam now slice serialized
  resident parameters and accumulate mathematical gradients as descending
  device directions. Kimi packs all projections and four LayerNorm arrays;
  Tucker performs source-factor packing and inverse gradient permutations on
  device using existing token-layout kernels. Both preserve biases-first
  serialization and root bias/weight DNOPT slots, independent of Mamba's child
  moment slots. Separate coupled-decay masks preserve bias L2 under Adam while
  excluding bias decay for SGD/AdamW. Deferred/eager updates, explicit gradient
  diagnostics, host mirrors and sidecar resume use the shared bank lifecycle.
  Muon and whole-network CPU backward retain their hybrid paths. Kimi scalar
  finiteness checks still synchronize; no throughput claim is made.
- Half-storage continuation: add required device widening/narrowing methods to
  F16TensorOpsBackend and connect Oxide's existing conversion kernels. Supported
  MambaVision/KimiKDA/TuckerAttention inference now crosses half activation
  boundaries around device f32 arithmetic islands with f32 master parameters.
  Training caches are preserved. This is neither native half mixer arithmetic
  nor FP16 training completion. Unrun fixtures cover half conversion shape,
  NaN/overflow, inference between forward/backward and updated resident masters.
- Additional unrun Kimi/Tucker fixtures cover asymmetric Tucker ranks and
  multiple heads, padded dense/sparse windows, repeated accumulation, loss
  scaling, three deferred optimizer steps, decay modes, stale-cache/host-edit
  rejection, update gates, reset and Kimi root-slot DNOPT resume. Existing Kimi
  gradient comparisons now request explicit resident diagnostics. Host staleness
  is recorded on each resident layer publication, even if a later layer fails.

## Resident Muon, dense AMP and GRU training

- Resident Mamba/Kimi/Tucker/non-BN-GRU Muon: explicit serialized matrix views
  match CPU visitor and DNOPT6 order. Momentum/lookahead, Newton-Schulz,
  AdamW non-matrix updates and final parameters stay on device. Matrix Adam
  gradients are masked to zero while existing moments still advance, preserving
  resumed AdamW/Muon behavior. Scalar finiteness guards synchronize. Eager mixed
  networks retain hybrid updates for other families; deferred Muon requires all
  trainable families to be resident. Per-layer publication is staged before
  installation, not a transaction across the entire network.
- DNOPT6 saves materialize current resident matrix momentum without requiring
  host-weight sync. Explicit synchronization materializes gradients, Adam moments,
  Muon momentum and parameters. Load retires banks, validates CPU matrix order
  and lazily uploads loaded state on reuse. New GRU slots use six child
  bias/weight pairs in the existing serialized projection order.
- Native dense AMP training: Oxide connected forward/dgrad/wgrad use real
  binary16/bfloat16 operands via the existing tensor-core GEMM driver when AMP
  is enabled. Forward and dX round to half storage; dW/dB, master parameters and
  optimizer state remain f32. Each use rounds the current master, avoiding stale
  cached operands. Bias is f32 and added before output rounding. No saturation
  hides half overflow. Non-dense mixer operations can remain f32 islands; this
  does not establish all-layer native-half training or performance parity.
- GRU: exact CPU BPTT through all sequence steps and six projections, including
  the source's sigmoid candidate and time-major batching. Native device BPTT
  consumes a forward-owned tape, so inference cannot replace training caches.
  Gradients accumulate on device into resident banks; SGD/Adam/Muon, full/compact
  f32 and half-storage inference read updated masters. Non-BN only; BN training
  remains explicitly rejected. Legacy rolling-stat BN inference is retained.
  CPU gradient accumulation negates mathematical derivatives to match additive
  optimizers. Added the missing GPU host-SGD arm for whole-network fallback.
- Unrun fixtures: CPU matrix-view values/order against the Muon visitor; mixer
  deferred/resume/gate/reset Muon; ragged dense AMP forward/dgrad/wgrad against
  rounded operands, f32 state and half overflow; mixer AMP updates; GRU CPU
  finite differences over every input/parameter; three-step GRU repeated BPTT,
  deferred SGD/Adam/Muon, AMP, inference/cache isolation and optimizer resume.
  The historical GRU training-rejection fixture now pins the BN-only restriction.

## Evidence and remaining work

Explicit resume, 2026-10-26, at `0e93bcb`: host check passed for core/CLI
(1m 07s), `PKGS=lightnet-core` host `runtime::` passed **324 tests, 0 failures**
(51.17s test time), and `gpu-check` passed (19.91s). No compile fixes were
needed. All commands ran serially under MemoryHigh=3G with two build jobs,
two host test threads and Rayon capped at two. Existing unused-import warnings
remain. The runtime filter selected zero cfg-corpus tests. Resident lifecycle
GPU tests are in progress; neither device lifecycle nor full-suite success is
established yet. This paragraph supersedes historical uncompiled/stop wording
below for those three verification gates only.

Checkpoint `2ee6840` records the Muon/AMP/GRU implementation and fixes covered
below. A later lifecycle continuation moves Mamba/Kimi/Tucker
training tapes into GpuForwardState and makes resident generations monotonic
across bank retirement/reload. Two new device regressions cover overlapping
forwards (input and parameter adjoints) and stale sidecar-reload tapes.
The same source change now covers DeformConv, ViT, Swin, WMHF, Clifford, DPT,
DepthAnything and RecursiveBlock; WMHF/Clifford rolling-stat accumulation
also reads the supplied forward state. `oxide_forward_tapes` adds isolated
versus interleaved GPU input-adjoint/BN-stat checks across those eight families.
These later edits are checkpointed at `0e93bcb` and remain **uncompiled/unrun**.
The prior checkpoint's full core invocation was interrupted during cfg-corpus
integration at the user's stop request. The test scope was stopped and no
task-owned Cargo/corpus process remained. Do not apply older passing results
to this continuation or resume work without an explicit user request.

The two focused resident lifecycle GPU regressions also passed at `0e93bcb`:
overlapping forwards **1/1** (84.86s), stale reload generation **1/1** (72.91s),
each across Mamba/Kimi/Tucker. Built/executed through the Oxide wrapper, one
test thread and command at a time under MemoryHigh=3G. Full resident suite and
the other eight families' forward-tape suite were pending at that point.
The subsequent `oxide_forward_tapes` invocation passed **1/1** (333.34s),
covering DeformConv, ViT, Swin, WMHF, Clifford, DPT, DepthAnything and bounded
RecursiveBlock lifecycle equivalence. It compares isolated/interleaved GPU
input adjoints and WMHF/Clifford BN statistics, not independent numeric parity.
Full CLI tests subsequently passed **36/36** (0.44s). Full numeric
layer/corpus gates are pending. Full `oxide_resident_params` subsequently
passed **17/17** (1278.44s) at `0e93bcb`: a fresh full-suite aggregate rather
than the earlier partial reruns. `oxide_mambavision` passed **2/2** (98.50s),
including runtime accumulation/two updates and nonzero scan/final-state
adjoints. `oxide_kimi_kda` passed **3/3** (100.85s), including runtime
outputs/gradients/two updates, scan/projection adjoints and top-k ties.
`oxide_tucker_attention` passed **3/3** (175.91s), including compact padded
dense/sparse windows, half casts and resident factor-gradient/deferred
optimizers. `oxide_vit` passed **5/5** (258.35s), covering backward/elementwise
kernels, patch layout, native forward and training backward/update against
Rust references. `oxide_swin` passed **5/5** (224.26s), including native
forward/training backward/update, mHC and window-layout adjoints and the
documented fallback. `oxide_wmhf` passed **8/8** (247.76s), including native
inference/training backward/update and primitive references. Initial
`oxide_clifford` initially finished **4 passed, 1 failed** (220.09s), in
`clifford_native_adam_update_matches_cpu`: updated parameter 0 was
`1.1428108 != 1.1332897`. Its forward/input/parameter gradient checks passed.
Root cause found in one pass: the fixture called CPU SGD unconditionally
while GPU selected configured Adam. The fixture now calls CPU Adam for Adam
cases; no tolerance/production change. Full corrected rerun passed **5/5**
(216.17s). Named-path staging failed with `.git/index.lock: Read-only file
system`, so the validated fixture fix remains uncommitted; no commit/tag was
created. `oxide_dpt_head` passed **1/1** (59.33s), across the existing
host-staged SGD/Adam/Muon path before the DPT residency slice below.
`oxide_depth_anything` passed **2/2** (63.13s): clamp/NaN boundary and frozen
encoder repeated-tap/patch-layout/input-adjoint references.
Initial `oxide_recursive_block` failed **0/1** (22.47s), with shared
parameter accumulation element 0 `-0.008424982 vs 0.008424982` after input
adjoints passed. One source pass found an extra native negation: nested CPU
convolution/connected accumulators add the returned gradient. Corrected the
two native loops to add in `layers/recursive_block/gpu.rs`. CPU conventions
and tolerances are unchanged. Restarted serial verification: host check passed
(4.44s), runtime:: passed (324/324), gpu-check passed (4.41s), and full
`oxide_recursive_block` rerun passed **1/1** (59.48s). `git diff --check`
passed. Next source slice is DPT SGD/Adam residency; Muon remains separate
because DPT host fallback skips matrix Adam moments while generic resident
Muon advances them.

DPT continuation: resident SGD/Adam now slices its 66-slot serialized arena,
packs slot gradients on device, and applies per-slot child moment mappings.
Empty slots 12/13 and optimizer-inactive slots 28/29/36/37 are masked from
updates and moment materialization; inactive weights and preloaded update
accumulators are blended through unchanged. DPT Muon uses its source visitor's
32 matrix views, masks matrix Adam moments, and retains Adam moments for active
biases. Host check passed, `runtime::` passed 325/325, gpu-check passed,
`dpt_head_resident_optimizers_preserve_inactive_slots_and_moments` passed 1/1,
and `oxide_dpt_head` passed 1/1 (65.57s) across resident SGD/Adam/Muon. The
regression saves and reloads DPT Adam and Muon child moments through DNOPT and
compares restored child slots. The full shared `oxide_resident_params` suite
passed 17/17 (1117.58s) after the generic resident Muon policy change. The
first full runtime rerun exposed a shape mismatch (323/324);
preserving native gradient shapes until flat arena packing fixed it. Remaining
gaps: switching a serialized Adam DPT state into Muon and broader layer/family
evidence. This is a partial optimizer-residency slice, not DPT completion. No
tolerance changes or C++ parity claims.
Other numeric layer/corpus gates remain pending.

Current authorized verification uses at most two jobs/test threads and a 3G
soft RAM cap, one Cargo invocation at a time:

- Host `cargo xtask check`: passed for lightnet-core and lightnet-cli.
- `cargo xtask test gru --test-threads=2`: 4 passed, including exact GRU input
  and parameter finite differences.
- `cargo xtask test muon --test-threads=2`: 26 passed, including serialized
  resident matrix values/shapes/DNOPT order against the CPU visitor.
- First broad runtime filter: 305 passed, 19 failed. Root-cause investigation
  found missing lazy training-buffer initialization (constructor vectors had
  correct lengths but empty slots), a fake routing backend that inferred batch
  from nominal token shape instead of layout metadata, stale fallback fixtures,
  and a Swin CPU multi-head cache stride mismatch. Buffer allocation/invalidation,
  routing metadata and explicit fallback-fixture fixes are now verified by the
  runtime filter: **324 passed, 0 failed**. The Swin CPU cache uses compact
  per-head storage, fixing its exact adjoint; the direct two-/three-head
  finite-difference test passed (1 test).
- GPU-enabled `cargo xtask gpu-check`: passed for lightnet-cli with Oxide.
- Oxide `oxide_connected_amp`: **3 passed, 0 failed** on the GPU, including
  f16/bf16 ragged dense forward/dgrad/wgrad, f32 parameter gradients/master and
  Adam state, refreshed operands after updates, AMP-off behavior and visible
  half overflow. Built through the repository wrapper with two build jobs and
  one test thread under MemoryHigh=3G.
- Oxide `oxide_gru_training`: **1 passed, 0 failed** on the GPU. The fixture
  exercises all six combinations of SGD/Adam/Muon and AMP off/on over three
  sequence steps, repeated BPTT, deferred updated-master inference, explicit
  synchronization and Adam/Muon checkpoint resume. Completed in 209.68s;
  this is suite wall time, not training/throughput evidence.
- Initial Oxide `oxide_resident_params`: **12 passed, 3 failed**. The failures
  exposed a CPU Mamba scan-adjoint error: the C coefficient was read from the
  parameter blob instead of the saved per-token selective projection. A new
  finite-difference regression failed before the one-line correction
  (0.10833 numerical derivative versus zero analytic derivative). After the
  fix, all **19 Mamba host tests passed**, including the new regression.
  Focused `resident_muon` device rerun: **3 passed, 0 failed**, covering gates,
  momentum reset, AdamW-to-Muon state preservation and repeated updates/DNOPT6
  resume across all three mixer families. The separate SGD/Adam gate rerun
  also **passed (1/1)**. All three original failures now pass; the earlier
  twelve passing cases have not been rerun after this correction. No
  tolerances were changed.
- Full core host invocation: **1,016 unit tests passed**. Its cfg-corpus
  integration test was interrupted at the user's stop request, without an
  aggregate result. This is neither a pass nor a numeric failure; the full
  host suite remains unproven. Full CLI host tests remain pending.

Further Oxide numeric suites are pending until an explicit resume. Host backend
reference tests do not prove GPU execution or device parity. References are
hand-derived Rust/C++ source, no compiled-C++ golden. Whitespace validation
uses git diff --check.

This is not every layer completed. Remaining RecursiveBlock variants, yolo_3d CPU loss
and device path, remaining native variants/controller paths, remaining compact f32/f16
coverage, host optimizer transfers/residency, native Muon eligibility and
the rest of the AGENTS.md queue remain open. Existing ViT/Swin/WMHF/Clifford
WIP commits are also unvalidated and do not certify completion.

DPT parameters/gradients remain host-staged at the optimizer boundary.
Mamba/Kimi/Tucker/non-BN-GRU retain resident parameters/gradients/optimizer
state for SGD/Adam/Muon. Other families and CPU-fallback networks retain their
documented hybrid paths. Mamba clamp/MSA flags, Kimi gradient finiteness and
resident Muon finiteness still synchronize scalar flags. No speed claim is
made. Half-storage inference exists for these four resident families; other
half dispatch and native arithmetic coverage remain open.

Next work follows the AGENTS.md device queue and the now-authorized host/Oxide
verification gates, serially with two jobs. Keep WIP commits local and do not
tag them as verified releases before all required coverage passes.

## ViT residency and half-storage follow-up

On the resumed `0e93bcb` tree, ViT now has native supported SGD/Adam/AdamW
parameter residency, device gradient accumulation and its 16-array serialized
order. Its mHC reductions/updates and frozen sinusoidal position handling stay
on device. The support gate still excludes MSA and VC attention; those
backward/residency variants remain open, as do ViT Muon and full layer
completion.

Validation: host check passed; gpu-check passed; `PKGS=lightnet-core`
`runtime::` passed **325/325** (zero cfg-corpus cases selected);
`oxide_vit` passed **7/7** (364.28s), including dense/flash backward CPU
comparisons, resident optimizer/deferred-update/checkpoint checks, and f16
compact inference compared at half-storage boundaries before and after updates.
`oxide_forward_tapes` freshly passed **1/1** (297.20s) before the f16 dispatch
edit, establishing lifecycle equivalence only. The named-path DPT commit retry
again failed because `.git/index.lock` creation reports `Read-only file
system`; no lock was removed and no commit/tag was created. Diff whitespace,
full CLI/layer reruns after these edits and cfg-corpus remain unverified.

The next family for native residency is Swin. Keep its optional residual
projection and Rust-defined variable blob order explicit; do not combine this
with ViT's unsupported attention variants in one change.

DPT checkpoint-mode switch follow-up at `627d115`: a new host regression saves
an actual DPT Adam DNOPT sidecar containing matrix moments, then verifies that
a Muon-configured DPT executable rejects it without changing weights, step,
Adam state or Muon state. Host check and gpu-check passed; the core `runtime::`
filter passed **326/326** (20.03s, zero corpus tests selected), and
`oxide_dpt_head` passed **1/1** (68.88s). A fresh `oxide_forward_tapes` run
passed **1/1** (324.59s). The policy is rejection, not implicit moment
discard; full DPT completion remains open.

Optional ViT Muon coverage was added at `627d115`: focused
`oxide_vit::vit_gpu_muon_hybrid_backward_and_adamw_update_match_cpu` passed
**1/1** (23.91s). It verifies the existing hybrid path—native ViT forward and
backward, host-staged parameter gradients, GPU Muon orthogonalization, and
AdamW fallback—against CPU. Resident ViT Muon updates remain unimplemented and
unverified; this test is not evidence of native resident Muon.

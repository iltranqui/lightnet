# Native transformer and depth source continuation

## Contents

- [Execution boundary](#execution-boundary)
- [Written source](#written-source)
- [Resident Muon, dense AMP and GRU training](#resident-muon-dense-amp-and-gru-training)
- [Evidence and remaining work](#evidence-and-remaining-work)

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
  integration test is still running serially; the handbook records prior
  corpus memory/timeout issues, so this is not yet a green full host suite.

The Oxide numeric suites are in progress. Host backend
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

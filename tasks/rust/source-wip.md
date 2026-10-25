# Native transformer and depth source continuation

## Contents

- [Execution boundary](#execution-boundary)
- [Written source](#written-source)
- [Evidence and remaining work](#evidence-and-remaining-work)

## Execution boundary

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

## Evidence and remaining work

Source diff inspection only; whitespace validation uses git diff --check.
All new fixtures are unrun. No successful compilation or numeric parity is
claimed. References are hand-derived Rust/C++ source, no compiled-C++ golden.

This is not every layer completed. Remaining RecursiveBlock variants, yolo_3d CPU loss
and device path, remaining native variants/controller paths, remaining compact f32/f16
coverage, host optimizer transfers/residency, native Muon eligibility and
the rest of the AGENTS.md queue remain open. Existing ViT/Swin/WMHF/Clifford
WIP commits are also unvalidated and do not certify completion.

Kimi/DPT parameters are uploaded for forward and parameter gradients
downloaded for the existing optimizer boundary; their SGD/Adam state/final
updates are host-managed and Muon uses the existing hybrid path. Mamba native
SGD/Adam now retain device state, while Muon and CPU-fallback networks keep
their host-managed update path. Kimi key-floor
diagnostics, Mamba clamp/MSA flags and Kimi gradient-finiteness inspection
still synchronize with the host. These are explicit residency gaps, with no
speed claim. Compact half inference is still rejected for these new paths.

Next source work follows the AGENTS.md device queue; restore the named host
and Oxide gates only after the user lifts the execution restriction. Keep
WIP commits local and do not tag them as a verified release.

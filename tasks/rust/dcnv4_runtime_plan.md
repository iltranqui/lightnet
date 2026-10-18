# DCNv4 runtime follow-ups

## Contents

- [Current boundary](#current-boundary)
- [Files](#files)
- [Open gates](#open-gates)
- [Constraints and verification](#constraints-and-verification)

## Current boundary

Bounded f32 runtime math is wired in v0.2.013; see `implemented.md` §50 for the
complete requirement-to-artifact audit, test-first evidence, resolved compiler
issues and the full runtime-chain timeout. Do not repeat trait migration or
no-BN forward/backward wiring. Native eligibility is no BN, <=25 active points,
supported elementwise activation and no other CPU fallback/host-only deform_conv.

Parameter gradients and SGD/Adam still cross to host buffers. Backward recomputes
offsets from that forward state's own input and unchanged current parameters;
no mutable last-forward cache or public GpuForwardState shape change. Projection
weights/biases upload again, and redundant aggregation recomputation is accepted
but disclosed. No speedup/residency claim. GPU Muon remains absent.

## Files

- `src/darknet-core/src/runtime/dcnv4_gpu.rs`: eligibility and family math helpers.
- `src/darknet-core/src/runtime.rs`: shared forward/backward path selection and
  existing host optimizer plumbing; keep dispatch changes small.
- `src/darknet-cuda/src/lib.rs`, `src/darknet-cuda/src/host_ops/dcnv4.rs`,
  `host_ops/dcnv4_composition.rs`: generic surface and host defaults.
- `src/darknet-cuda-oxide/src/lib.rs`, `src/darknet-cuda-oxide/src/dcnv4_native.rs`:
  native backend surface/overrides; kernels already implement raw math.
- `src/darknet-core/src/runtime/tests/dcnv4_gpu_tests.rs` and
  `src/darknet-cuda-oxide/tests/oxide_runtime_chain.rs`: path/cache, multiple
  forward states, accumulation, parameter motion and two-step optimizer parity.
- Handbook/status/todo/implemented/support matrix/version map: honest scope.

## Open gates

- [ ] **Next: native DCNv4 rolling-stat BN affine/adjoint.** Audit
  `forward_dcnv4_with_cache` and CPU DCNv4 backward: stored rolling mean/variance,
  scale, no beta addition and no batch-stat training substitution. Verify forward,
  input/scale/predictor gradients, update gates and two steps before removing BN
  from `needs_host`. Retain mixed-network caches.
- [ ] Native NormalizeChannels*/unsupported activation and >25-point paths.
- [ ] Compact/f16 inference and precision-specific tests, preserving current
  explicit rejection until each path is implemented.
- [ ] Device-resident gradients, optimizer state/updates and reusable weights;
  no performance claim without actual measurements.
- [ ] Investigate full runtime-chain timeout separately: the 600-second run
  reached ViT flash-attention after 25 preceding tests reported ok, but yielded
  no final result. Focused DCNv4 runtime suite passes; do not call the full suite
  green or infer the timeout's cause.

## Constraints and verification

Direct work only, one build/test at a time, -j 2 (max 4) in MemoryHigh=3G user
scopes; Oxide only via test_oxide.sh/cargo-oxide. No dataset training, subagents,
new host fallback disguised as native, unrelated/parked edits or compiled-C++
parity claims. Preserve every open CPU/source/axis/GPU/Muon/sparse-MoE requirement.

Before each next slice, write its precise contract/test checklist. After runtime
changes run `test_host.sh runtime::` and `dcnv4`; verify matched counts. Run
host-cuda and focused `oxide_dcnv4_chain`, `oxide_dcnv4_aggregate` and runtime
DCNv4 tests. Test both native selection and retained host caches, signed seeds,
B>1/groups, all affected parameter blocks, actual GPU forward derivatives and
optimizer motion. A full-suite timeout is not a pass. Stop on unresolved math/
cache mismatch with evidence and keep the affected fallback gate intact.

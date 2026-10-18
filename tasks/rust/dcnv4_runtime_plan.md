# DCNv4 runtime follow-ups

## Contents

- [Current boundary](#current-boundary)
- [Files](#files)
- [Open gates](#open-gates)
- [Constraints and verification](#constraints-and-verification)

## Current boundary

Bounded f32 runtime math is wired in v0.2.013 and rolling-stat BN/adjoint/EMA
is native in v0.2.014; audits in `implemented.md` §§50–51. Do not repeat trait
migration, forward/backward wiring or BN math. Native eligibility is <=25 active
points, supported elementwise activation, optional BN and no other CPU fallback/
host-only deform_conv. BN normalizes with old rolling stats; 0.9 EMA runs after
backward, including unseeded/gated layers, not in forward.

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
- `src/darknet-core/src/runtime/tests/dcnv4_gpu_tests.rs`,
  `src/darknet-cuda-oxide/tests/oxide_dcnv4_bn.rs` and
  `src/darknet-cuda-oxide/tests/oxide_runtime_chain.rs`: path/cache, multiple
  forward states, accumulation, parameter motion and two-step optimizer parity.
- Handbook/status/todo/implemented/support matrix/version map: honest scope.

## Open gates

- [ ] **Next: native DCNv4 NormalizeChannels* activations/adjoints.** Use existing
  backend normalization primitives after optional BN affine, pin gradient-mode
  semantics and zero/tie cases, verify CPU/GPU derivatives/two-step updates before
  relaxing activation gates. Retain other unsupported/mixed host caches.
- [ ] Other unsupported activations and >25-point paths.
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

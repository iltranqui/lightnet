# Native GPU layer work

## Contents

- [Priority and boundaries](#priority-and-boundaries)
- [Resume point](#resume-point)
- [Act-head contract finding](#act-head-contract-finding)
- [Next independent slice: native act-head forward](#next-independent-slice-native-act-head-forward)
- [Verification gates](#verification-gates)
- [Remaining scope](#remaining-scope)

## Priority and boundaries

The operator explicitly requested GPU layers now, overriding CPU-first sequencing.
CPU completeness is not implied. Implement directly without subagents. Pending,
uncommitted YOLO-3D activation math is parked: do not discard or include it in GPU
commits (`darknet-core/src/lib.rs`, `model/yolo_3d_config.rs`, `yolo_3d.rs` and its
tests). Preserve unrelated user edits. One build/test at a time, max four build
threads, MemoryHigh=3G; Oxide builds/tests only through `tools/test_oxide.sh` /
cargo-oxide. No training runs. Native math, not new host fallback, is the GPU
deliverable. Sparse MoE remains unstarted.

## Resume point

Marker gather is native in f32 full/compact forward/backward and f16 compact
inference (v0.2.001–002). Standalone act-head feature construction (v0.2.003) and
exact-erf GELU forward/backward (v0.2.004) are native f32. Evidence/limits are in
`implemented.md` §§41–44. Do not repeat these implementations. Exact erf uses
the already-supported device-FFI/libdevice route; no compiler change was needed.
Act-head runtime still uses host fallback and still rejects f16. Next GPU version
is v0.2.005; recheck the ledger before allocating.

## Act-head contract finding

The source audit resolved facts, not the operator's requested behavior:
`Jev_laya@4af013d2:src-lib/laya_act_head_layer.cpp` accumulates and updates all
four weight/bias blocks, while Rust runtime drops the pure helper's parameter
gradients and retains only dcls. Existing module comments call weights frozen.
The question about including this training fix has not been answered; automatic
continuation is not treated as approval to change existing training behavior.
Preserve it while advancing independent native math/forward execution. This is
an outstanding source-training gap, not a completed frozen source contract.

Do not remove the CPU-backward fallback until all required backward/update paths
are resolved and actually implemented. Keep scorer/marker summary features
detached and route the head adjoint to CLS only. Source feature edge cases and
exact activation contracts are recorded and tested in §§43–44.

## Next independent slice: native act-head forward

Forward math is independent of the unanswered training-eligibility question.
Replace its per-operator host feature/MLP execution while retaining CPU backward.
This would be native forward only, not a completed native layer.

- [ ] Inspect current full/compact GPU act-head arms and dense/GEMM/bias APIs,
  precision/AMP behavior, source-resolution helpers and compact use counts.
  Read runtime_modularization_plan before editing dispatch; thin arms only.
- [ ] Use the existing native feature primitive to create `[B,H+4,1,1]`, then
  fc1 linear, dedicated exact-erf GELU, then fc2 linear. Never substitute the
  fused tanh-GELU kind. Confirm matrix orientation/output metadata against the
  existing CPU source loops before choosing the backend composition.
- [ ] Validate actual layouts, positive F/A, checked parameter products and all
  four blob lengths: fc1_w `[F,H+4]`, fc1_b `[F]`, fc2_w `[A,F]`, fc2_b `[A]`.
  Define supported precision/dtype limits rather than silently enabling modes.
- [ ] Audit parameter upload/cache lifetime and invalidation under blob reload,
  params replacement and batch/shape rebuilds. Correct uncached upload is safer
  than stale reuse; claim residency/speedup only if actually implemented/tested.
  Retain any CPU weight copy required by the existing backward fallback.
- [ ] Test-first hand-computed and CPU/device MLP forward cases, B>1, nontrivial
  H/K/T/F/A, masked/all-masked features and exact-GELU-sensitive values. Test
  positive/relative/aliased source references and compact source lifetimes.
- [ ] Wire full and compact f32 forward only after direct native composition
  tests pass. Keep fallback selection and frozen/update behavior unchanged;
  explicitly document which work still executes on CPU. Keep f16 rejected.
- [ ] Prove actual runtime calls reach native ops, not host feature/MLP math.
  Test weight reload/mutation behavior, invalid parameter storage errors and
  relevant regressions. Preserve all unknown-config error behavior.

## Verification gates

Use test-first, serial test threads, script `-j 2` and the 3G cap. Add focused new
full/compact act-head runtime tests; generic backend defaults do not demonstrate
native Oxide execution. Existing relevant regressions:

```
MEM=3G ./tools/test_host.sh laya_act_head
MEM=3G ./tools/test_oxide.sh --test oxide_gelu_erf
MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_features
MEM=3G ./tools/test_oxide.sh --test oxide_grad_methods
MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather_f16
```

Targeted Rustfmt/diff checks, requirement-to-artifact audit, implemented/STATUS/
todo/version-map updates and disposable-artifact cleanup gate each release.
Distinguish native primitives/forward from full native layers, selected tests
from full-suite coverage and CPU-reference agreement from compiled-C++ parity.
No benchmark/performance claims without actual measurements.

## Remaining scope

Full act-head backward/update and other LAYA/transformer layers, DCNv4/deform/
deconvolution host fallbacks, CPU-only layer GPU paths, and GPU Muon remain open.
Each layer needs actual device forward/backward/update tests as applicable and
explicit limits. CPU/config/axis gaps remain open. No green full-host/corpus/
full-Oxide suite or compiled-C++ parity is established; sparse MoE stays unstarted.

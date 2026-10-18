# Native GPU layer work

## Contents

- [Priority and boundaries](#priority-and-boundaries)
- [Resume point](#resume-point)
- [Act-head contract finding](#act-head-contract-finding)
- [Next independent slice: exact erf-GELU](#next-independent-slice-exact-erf-gelu)
- [Verification gates](#verification-gates)
- [Remaining scope](#remaining-scope)

## Priority and boundaries

The operator explicitly requested GPU layers now, overriding CPU-first sequencing.
CPU completeness is not implied. Implement directly without subagents. Pending,
uncommitted YOLO-3D activation math is parked: do not discard or include it in GPU
commits (`darknet-core/src/lib.rs`, `model/yolo_3d_config.rs`, `yolo_3d.rs` and its
tests). Preserve unrelated user edits. One build/test at a time, max four build
threads, MemoryHigh=3G; Oxide builds/tests only through `tools/test_oxide.sh` /
cargo-oxide. No training runs. Native kernels, not new host fallback, are the
GPU deliverable. Sparse MoE remains unstarted.

## Resume point

Marker gather is native in f32 full/compact forward/backward and f16 compact
inference (v0.2.001–002). The standalone detached act-head feature primitive is
native f32 in v0.2.003, but not wired into runtime. Evidence/limits/source contract
are in `implemented.md` §§41–43. Do not repeat these implementations. Act-head
still uses host fallback and still rejects f16. Next GPU version is v0.2.004;
recheck the ledger before allocating.

## Act-head contract finding

The source audit resolved facts, not the operator's requested behavior:
`Jev_laya@4af013d2:src-lib/laya_act_head_layer.cpp` accumulates and updates all
four weight/bias blocks, while Rust runtime drops the pure helper's parameter
gradients and retains only dcls. Existing module comments call weights frozen.
The question about including this training fix has not been answered; automatic
continuation is not treated as approval to change existing training behavior.
Preserve it while advancing independent native math. This is an outstanding
source-training gap, not a completed frozen source contract.

For full act-head integration, still needed: native MLP and cache lifetime/
invalidation, detached scorer/marker behavior, CLS adjoint scatter, resolution
of parameter eligibility/buffers/update/state behavior, and real-device full/
compact/backward/update tests. Do not remove fallback before all required paths
are implemented. Existing source feature math (including all-masked and finite
-1e4 edge cases) is already recorded and tested in §43.

## Next independent slice: exact erf-GELU

Existing GPU `F32ActivationKind::Gelu` is the tanh approximation. LAYA CPU/source
uses exact erf-GELU; substituting tanh would change its forward/backward.
This math prerequisite is independent of the unresolved frozen/trainable choice.

- [ ] Inspect CPU `laya_scorer::{gelu_erf,gelu_erf_grad}`, native activation/
  CUDA-device math conventions, feature gates and existing native erf support.
  Do not assume Rust std erf lowers correctly through the custom backend.
- [ ] Prefer dedicated checked f32 forward/backward primitives to avoid changing
  existing Gelu enum/cfg semantics or accidentally enabling unsupported f16.
  No approximation or host execution in the Oxide override.
- [ ] Source formula: y=.5*x*(1+erf(x/sqrt(2))); derivative is normal CDF plus
  x*normal PDF. Differentiate the actual Rust forward exactly; preserve the
  existing tanh activation behavior and all other public paths.
- [ ] Test first: hand values (notably +/-1 distinguish variants), nonunit seed
  signs, central finite differences away from saturation, non-square batched
  layouts and ragged lengths, extreme/nonfinite policy matching the chosen
  documented Rust contract, malformed shape/storage errors before unsafe launch.
- [ ] Observe missing-operation failures, implement actual device kernels, then
  run relevant activation/gradient and existing LAYA primitive regressions.
- [ ] If native erf/codegen cannot be supported defensibly, report the actual
  blocker rather than silently adding a tanh/host fallback. Other independent
  native GPU tasks can then be selected explicitly.

## Verification gates

Use serial test threads, script `-j 2`, and the 3G cap. Add focused direct exact-
erf forward/backward tests before considering act-head runtime integration.
Existing relevant regressions:

```
MEM=3G ./tools/test_host.sh laya_act_head
MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_features
MEM=3G ./tools/test_oxide.sh --test oxide_layers_extra
MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather_f16
```

Targeted Rustfmt/diff checks, requirement-to-artifact audit, implemented/STATUS/
todo/version-map updates and disposable-artifact cleanup gate each release.
Distinguish a native primitive from a native layer, generic host defaults from
Oxide overrides, and selected-suite evidence from full-suite completeness.
No benchmark/performance or compiled-C++ parity claims without actual evidence.

## Remaining scope

Full act-head and other LAYA/transformer layers, DCNv4/deform/deconvolution host
fallbacks, CPU-only layer GPU paths, and GPU Muon remain open. Each layer needs
actual device forward/backward/update tests as applicable and explicit limits.
CPU/config/axis gaps remain open. No green full-host/corpus/full-Oxide suite or
compiled-C++ parity is established; sparse MoE stays unstarted.

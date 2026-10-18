# Native GPU layer work

## Contents

- [Priority and boundaries](#priority-and-boundaries)
- [Resume point](#resume-point)
- [Next bounded slice: native LAYA act-head](#next-bounded-slice-native-laya-act-head)
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

Marker gather is native in f32 full/compact forward/backward (v0.2.001) and f16
compact inference (v0.2.002). Evidence and limits are in `implemented.md` §§41–42;
do not repeat those implementations. No new full-state f16 or f16 training API
was added. The next GPU version is v0.2.003; recheck the ledger before allocating.

## Next bounded slice: native LAYA act-head

The current CPU module describes a CLS-pooled two-layer exact-erf-GELU MLP with
four summary features from masked scorer probabilities. Its feature inputs are
detached in backward. The module describes frozen weights but exposes pure
parameter gradients; do not infer optimizer eligibility from that alone.

- [ ] Read the CPU module, model spec, runtime forward/backward/update arms and
  optimizer membership. Refresh C++ paths with git grep/show (expected
  `Jev_laya:src-lib/laya_act_head_layer.cpp`) and pin the actual source SHA.
  Resolve the frozen/gradient/update contract before writing device code.
- [ ] Write a concrete formula/layout contract for the four detached features:
  top1, top1-top2, normalized entropy and active-option count; inspect mask,
  all-masked/nonfinite, numerical floors, ties and temperature semantics rather
  than assuming the module summary is complete. Confirm all three source shapes.
- [ ] Inventory existing native GEMM/bias/exact-GELU and reduction primitives.
  Keep features, cached activations, input adjoints and required parameter data
  on device. Define cache lifetime/invalidation behavior before integration.
- [ ] Test-first native f32 forward and exact backward with detached scorer/
  marker features fixed. Preserve frozen/update semantics; no invented parameter
  updates or new optimizer eligibility. Do not remove the CPU-backward fallback
  until all required source/parameter gradient routing is actually implemented.
- [ ] Replace full/compact host-forward math and the fallback gate with thin
  runtime arms only after direct kernel tests pass. Keep f16 explicitly rejected
  until independently implemented/tested; do not add an unverified dtype path.
- [ ] Map every contract requirement to hand math, CPU/device comparison, fixed-
  feature finite differences and runtime tests; validate all layouts/storage and
  error paths before unsafe launches. Record coverage gaps honestly.

## Verification gates

Use test-first, serial test threads, script `-j 2`, and the 3G cap. Add named
LAYA act-head direct and runtime tests for batch >1, nontrivial H/K/T axes,
masked options, source aliasing/lifetimes and the resolved update contract.
Do not assume generic backend host defaults prove native Oxide execution.

Existing marker-gather regressions:

```
MEM=3G ./tools/test_host.sh marker_gather
MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather
MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather_f16
MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- marker_gather
MEM=3G ./tools/test_oxide.sh --test oxide_runtime_f16
```

Use focused new-layer filters first, then relevant regressions. Targeted
Rustfmt/diff checks, artifact/test audit, implemented/STATUS/todo/version-map
updates and disposable-artifact cleanup gate each release. No performance claim
without a benchmark; no full-suite claim from selected suites. Hand-derived from
source, no compiled-C++ golden.

## Remaining scope

Other LAYA/transformer layers, DCNv4/deform/deconvolution host fallbacks, CPU-only
layer GPU paths, and GPU Muon remain open. Each selected layer needs actual device
forward/backward/update tests as applicable and explicit unsupported-mode limits.
CPU/config/axis gaps remain open. No green full-host/corpus/full-Oxide suite or
compiled-C++ parity is established; sparse MoE stays unstarted.

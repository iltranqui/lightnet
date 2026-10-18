# Native GPU layer work

## Contents

- [Priority and boundaries](#priority-and-boundaries)
- [Resume point](#resume-point)
- [Act-head contract finding](#act-head-contract-finding)
- [Next independent slice: native act-head gradient math](#next-independent-slice-native-act-head-gradient-math)
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
inference (v0.2.001–002). Act-head features/exact-erf primitives are native f32
(v0.2.003–004), and full/compact act-head f32 forward is native in v0.2.005.
Evidence/limits are in `implemented.md` §§41–45. Do not repeat these ports.
Act-head uses four uncached weight uploads per forward; no residency/speedup
claim. Runtime backward still uses CPU, act params remain frozen, f16 rejected.
Next GPU version is v0.2.006; recheck the ledger before allocating.

## Act-head contract finding

`Jev_laya@4af013d2:src-lib/laya_act_head_layer.cpp` accumulates/updates all four
weight/bias blocks, while Rust runtime drops pure helper parameter gradients and
retains only dcls. Existing module comments call weights frozen. The question
about including this training fix has not been answered; automatic continuation
is not approval to change existing training behavior. Preserve it while doing
independent native math. Do not remove CPU-backward fallback or start applying
new updates until the full required contract is resolved and implemented.

Shared-file loading is another evidence gap: tests cover in-memory blob
replacement, not a file loader. The runtime hint names load_laya_act_head_blob,
but no such core loader implementation was found. Do not accept the hint as proof.
Native forward precision inherits existing connected/GEMM settings; only explicit
Fp32 mode was validated for parity. No new global AMP/precision behavior was added.

## Next independent slice: native act-head gradient math

Pure gradients are independent of optimizer eligibility: CPU's standalone
act_head_backward already computes all four parameter derivatives even though
runtime ignores them. Advance equivalent device math without wiring updates.

- [ ] Inspect native connected_backward_f32, row/column layout conventions,
  exact-erf backward and device route/slicing primitives. Define a checked,
  testable operand/result API before implementation, preferably backend math
  returning typed device tensors rather than host downloads or a test-only API.
- [ ] Compose/recompute features, fc1 preactivation, exact-erf hidden activation,
  fc2 backward, seeded exact-erf backward, then fc1 backward. Return positive
  parameter gradients in source order plus `[B,H]` CLS adjoints. Validate checked
  dimensions/storage before each unsafe consumer; do not invent batch averaging.
- [ ] Scorer/marker summaries are detached constants. Only the first H columns
  of the feature gradient belong to CLS; discard the last four input derivatives.
  Existing native route/slice operations may suffice for CLS extraction. Full
  sequence-column scatter/routing is a later runtime integration requirement.
- [ ] Test first against CPU act_head_backward for all four blocks and dcls,
  B>1, nontrivial H/K/T/F/A, masked/all-masked inputs and signed/nonunit seeds.
  Actual GPU forward finite differences must hold scorer/marker inputs fixed;
  do not differentiate the detached feature path or confuse alias roles.
- [ ] Cover malformed shapes/parameter/seed storage and independent input-buffer
  lifetimes. Prove actual device primitive execution rather than generic host
  references. Recompute/upload is acceptable initially if correct and disclosed;
  no cache/speedup claim without evidence.
- [ ] Keep runtime fallback selection, act-head training buffers, parameter
  updates/optimizer state and f16 rejection unchanged in this independent slice.
  A native gradient primitive is not native runtime backward or full training.

## Verification gates

Use test-first, serial test threads, script `-j 2` and the 3G cap. Add focused real
GPU gradient-math tests; existing related regressions are:

```
MEM=3G ./tools/test_host.sh laya_act_head
MEM=3G ./tools/test_host.sh runtime::
MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_forward
MEM=3G ./tools/test_oxide.sh --test oxide_gelu_erf
MEM=3G ./tools/test_oxide.sh --test oxide_laya_act_features
```

Targeted Rustfmt/diff checks, requirement-to-artifact audit, implemented/STATUS/
todo/version-map updates and disposable-artifact cleanup gate each release.
Distinguish native forward/gradient primitives from complete native layers,
selected tests from full-suite coverage and Rust CPU agreement from C++ parity.
No performance/compiled-C++ claim without actual measurements/golden evidence.

## Remaining scope

Act-head runtime backward/update/file loading and f16, other LAYA/transformer
layers, DCNv4/deform/deconvolution host fallbacks, CPU-only layer GPU paths and
GPU Muon remain open. Each layer needs device forward/backward/update tests as
applicable and explicit limits. CPU/config/axis gaps remain open. No green full-
host/corpus/full-Oxide suite is established; sparse MoE stays unstarted.

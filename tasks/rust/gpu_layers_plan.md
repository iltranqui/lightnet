# Native GPU layer work

## Contents

- [Priority and boundaries](#priority-and-boundaries)
- [Resume point](#resume-point)
- [Next bounded slice: marker gather f16 inference](#next-bounded-slice-marker-gather-f16-inference)
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

Native f32 marker gather was delivered in v0.2.001 (`f56342ca`); actual artifacts,
source identity, tests and limitations are in `implemented.md` §41. Do not repeat
that implementation. The operator is weightless; no optimizer state was added.
The next GPU version is v0.2.002; recheck `version-map.tsv` before allocating it.

## Next bounded slice: marker gather f16 inference

- [ ] Inspect `F16TensorOpsBackend`, existing device half conversion primitives,
  Oxide f16 tensor/launch conventions, and runtime's explicit gather-f16 rejection.
- [ ] Extend the source-backed f32 routing contract to native f16 inference:
  copy half feature values exactly, decode half marker positions on device,
  then round half away from zero and clamp. Read channel `2*j`, column zero,
  independently per batch; ignore masks/extra columns.
- [ ] Keep metadata/feature buffers on device, preserve exact f16 bits when
  gathering features, and validate all layouts/storage before unsafe launches.
- [ ] Replace the runtime f16 rejection only after direct operator parity tests
  pass. No unverified f16 path, implicit host round trip or global AMP change.
- [ ] Test f16 full/compact inference (as supported by the existing f16 API),
  duplicate/per-sample/boundary/extreme indices and malformed shapes/storage.
  Reference rounding must operate on the quantized half marker value, not its
  pre-quantization f32 input. Include half subnormals and feature bit preservation.
- [ ] Review source/math/error paths, update evidence/status/todo/version ledger,
  and remove disposable logs/source copies. Do not claim all-GPU completeness.

## Verification gates

Run test-first with the existing script's serial test threads and 3G cap.
Existing regression commands (all passed for the f32 increment):

```
MEM=3G ./tools/test_host.sh marker_gather
MEM=3G ./tools/test_oxide.sh --test oxide_marker_gather
MEM=3G ./tools/test_oxide.sh --test oxide_runtime_chain -- marker_gather
MEM=3G ./tools/test_oxide.sh --test oxide_layers_extra
```

Add focused f16 tests using the actual existing f16 suite/API. Run targeted
Rustfmt/diff checks. Distinguish generic-backend host defaults from native Oxide
overrides, and ordinary debug/output downloads from per-operator host math.
No performance claim without a benchmark. Source evidence remains hand-derived
from source, no compiled-C++ golden.

## Remaining scope

After marker gather, audit and expand a bounded plan for native LAYA act-head
forward/backward and its parameter updates, then other LAYA/transformer layers,
DCNv4/deform/deconvolution host fallbacks, and CPU-only layer GPU paths. GPU Muon
remains open. Each selected layer needs actual device tests for forward/backward,
parameter routing/updates where applicable, and explicit unsupported-mode limits.
CPU/config/axis gaps remain open. No green full-host/corpus/full-Oxide suite or
compiled-C++ parity is established.

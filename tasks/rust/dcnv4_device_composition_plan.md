# DCNv4 native composition

## Contents

- [Contract](#contract)
- [Files](#files)
- [Phases](#phases)
- [Findings](#findings)
- [Verification and stop condition](#verification-and-stop-condition)

## Contract

Compose native f32 coarse projection, border-clamped offset interpolation,
aggregation and their adjoints; compare raw outputs, input and all offset
parameter gradients with the real CPU runtime. Preserve runtime fallback,
optimizer eligibility/updates, f16 rejection, unrelated YOLO-3D and user edits.
No training runs or subagents. This slice is not runtime integration or a
complete GPU layer. Hand-derived from source, no compiled-C++ golden.

## Files

- `src/darknet-cuda/src/host_ops/dcnv4.rs`: correct active-point count and
  eight-aligned row width, checked composition descriptor.
- `src/darknet-cuda-oxide/src/kernels/dcnv4.rs`: correct aggregation packing.
- `src/darknet-cuda-oxide/src/kernels/dcnv4_projection.rs`: native coarse
  predictor, upsample, and their adjoints; register in `kernels/mod.rs`.
- `src/darknet-cuda-oxide/src/dcnv4_native.rs`: checked native composition
  entry points/types; module load/export registration in `src/lib.rs`.
- `src/darknet-cuda-oxide/tests/oxide_dcnv4_aggregate.rs`: retain and strengthen
  aggregation/FD tests for actual packed rows.
- `src/darknet-cuda-oxide/tests/oxide_dcnv4_chain.rs`: real CPU output/gradient
  oracle, including groups, remove_center, softmax, coarse/odd grids and dilation.
- Rust handoff documents and `version-map.tsv`: evidence and open wiring gates.

## Phases

- [x] Inspect latest code/ledger and CPU layout/projection/interpolation/backward.
- [x] Add failing actual-layout test (27 vs expected 32) before changing math.
- [x] Correct aggregation packing and validate descriptor arithmetic.
- [x] Implement native predictor/interpolation and adjoints, no host downloads.
- [x] Compare with real CPU runtime input/parameter gradients (18 cases) and
  actual-device finite differences for all three operand groups.
- [x] Device chain 3, aggregate 6, host-cuda 36, host dcnv4 17, Oxide runtime
  fallback 1 passed; targeted formatting/diff checks and §49 audit recorded.
- [ ] Next slice: trait/default migration, coordinated runtime cache/forward/
  backward selection, epilogue, and two-step optimizer/weight parity. Keep
  fallback until those tests pass.

## Findings

- v0.2.009–011 only compare a decomposed host oracle with device code. They
  share a layout bug: `K=size²` and `row=groups*3*K`, unlike CPU `K=size²-
  remove_center`, `row=ceil(groups*3*K/8)*8`. Existing tests therefore establish
  internal consistency, not real CPU-layer agreement. Fix before composition.
- CPU projection is a dense convolution with explicit ceil-divided coarse
  extent and strides multiplied by d_stride. Stock conv output geometry is not
  interchangeable for dilation/coarse extents. Direct native kernels may use
  explicit extents initially; no speed claim without measurements.
- Offsets are NHWC, input/output NCHW. Weight order `[P,C,ky,kx]`. Upsample
  samples at `(y/d_stride,x/d_stride)` with border replication, not align-corners.
- Full layer epilogue/BN/activation and optimizer wiring remain later gates.

## Verification and stop condition

One build/test at a time, -j 2 and MemoryHigh=3G; Oxide only through test_oxide.sh.
Use actual CPU runtime gradients (`BackwardResult.input_gradient` exposes input
adjoints directly), not fabricated host stage outputs. Completed evidence and
resolved compiler/fixture errors are in `implemented.md` §49. If layout/math
mismatches cannot be resolved, leave fallback intact and record failing evidence.
Overall handbook objective stays open while any CPU/GPU/Muon/sparse-MoE gap exists.

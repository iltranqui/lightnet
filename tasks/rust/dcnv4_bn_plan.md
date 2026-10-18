# DCNv4 BN follow-ups

## Contents

- [Current boundary](#current-boundary)
- [Open work](#open-work)
- [Files and verification](#files-and-verification)

## Current boundary

The rolling-stat BN increment is implemented/verified in v0.2.014. Its complete
contract-to-artifact checklist, red selection test, CPU/GPU/FD results and
source boundaries live in `implemented.md` §51. Do not repeat affine/scale
adjoint or post-backward EMA work.

Forward/gradient normalization holds rolling stats fixed for the pass. EMA is
0.9 with unbiased variance after all gradients, including unseeded/gated layers.
Neither ordinary bias nor BN beta contributes. This is the existing Rust CPU
contract, not compiled-C++ training-BN parity (C++ CPU backward is a stub).

## Open work

- [ ] Compose native NormalizeChannels* activation and adjoint after optional BN
  affine. Specify analytical/Cpp modes and zero/tie fixtures before dispatch;
  test real CPU/GPU outputs, derivatives and two optimizer steps before relaxing
  these activation gates. Current bounded path excludes unsupported activations.
- [ ] Native large-kernel, compact/f16 paths and parameter-update/state residency.
- [ ] Avoid repeated raw/projection recomputation and unused batch-BN outputs in
  stats reduction only after a safe per-forward cache/residency contract exists.
  Do not introduce a mutable last-forward cache; preserve the unchanged-parameter
  forward/backward pairing (including rolling values).
- [ ] C++ training/BN evidence and full-suite completion remain unestablished;
  do not replace missing goldens with CPU/GPU agreement or the old ViT timeout.

## Files and verification

Keep family math in `src/darknet-core/src/runtime/dcnv4_gpu.rs`, shared hooks in
`runtime.rs` minimal, and update `runtime/tests/dcnv4_gpu_tests.rs`,
`src/darknet-cuda-oxide/tests/oxide_dcnv4_bn.rs` and `oxide_runtime_chain.rs`.
Use serial -j 2 (max4), MemoryHigh=3G, test_oxide.sh only for Oxide. No dataset
training/subagents/unrelated edits or speedup/C++ parity claims. After runtime
changes run runtime:: and dcnv4 host filters, relevant real-device derivatives,
raw composition/aggregate and optimizer/fallback regressions. Update all version
ledgers and scope docs; overall CPU/GPU/Muon/sparse-MoE goal stays open.

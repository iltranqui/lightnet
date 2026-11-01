# 04_bench — measure and tune

One job: honest performance numbers for the kernel, and tuning that survives measurement.

## Contents

- [Inputs](#inputs)
- [Process](#process)
- [Outputs](#outputs)
- [Human check](#human-check)

## Inputs

- Working (this run): `runs/<name>/02_kernel.md`, `runs/<name>/03_parity.md`.
- Reference (every run): `_shared/benchmarking.md`, `_shared/rules.md`.

Do NOT load: other runs' numbers as targets, `crates/hip_darknet`.

## Process

1. Add or extend a bench subcommand in `apps/runner/src/main.rs` (pattern: `gemm --bench`, `conv-bench`).
2. Measure per `_shared/benchmarking.md`: idle CPU, warm-up, best of N, twice. Compare with a reference (rocBLAS or the previous config) in the same run.
3. Try tuning (tile config, BK, prefetch). Keep a change only if it is faster in two runs and stage 03 still passes.
4. If a number becomes a project reference, update `_shared/benchmarking.md` (one home).

## Outputs

- `runs/<name>/04_bench.md`: config table (GFLOP/s or GB/s, VGPRs), what was kept and why, exact commands.

## Human check

Do the two measurement runs agree within ~10%, and were tests re-run after the last tuning change?

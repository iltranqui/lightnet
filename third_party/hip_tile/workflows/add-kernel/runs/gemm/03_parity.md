---
status: done
stage: workflows/add-kernel/03_parity
date: 2026-10-25
---

# gemm — parity

Contract: [`workflows/add-kernel/03_parity/CONTEXT.md`](../../03_parity/CONTEXT.md).

## Contents

- [Tests](#tests)
- [Worst error per shape](#worst-error-per-shape)
- [Test output](#test-output)

## Tests

`apps/runner/tests/gemm.rs` (f64 CPU and rocBLAS), `apps/runner/tests/rocblas.rs` (smoke).

## Worst error per shape

Max relative error vs f64 CPU: 1.72e-6 at N=1024, 2.44e-6 at N=2048; all ragged shapes under 1e-5.

## Test output

`tests/gemm.rs`: 5 passed; `tests/rocblas.rs`: 1 passed.

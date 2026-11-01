---
status: done
stage: workflows/add-kernel/01_spec
date: 2026-10-25
---

# gemm — spec

Contract: [`workflows/add-kernel/01_spec/CONTEXT.md`](../../01_spec/CONTEXT.md).

## Contents

- [Math](#math)
- [Shapes](#shapes)
- [Tile config and LDS](#tile-config-and-lds)
- [Launcher](#launcher)
- [Reference and tolerance](#reference-and-tolerance)

## Math

`C[m×n] = A[m×k] · B[k×n]`, f32, row-major `Tensor` views.

## Shapes

Square 1024, 2048, 1000, 77; rectangular 130×77×300 and 129×130×5.

## Tile config and LDS

Block tile 128×128×16 (8×8 accumulator per thread, 110 VGPRs). LDS `mma_lds_bytes::<f32>(128, 128, 16)`.

## Launcher

`kernels::gemm::gemm::launch(module, stream, (ceil(n/128), ceil(m/128)), a, b, c)`; helper `runner::launch_gemm`.

## Reference and tolerance

f64 CPU reference and rocBLAS `sgemm`; max relative error ≤ 1e-5.

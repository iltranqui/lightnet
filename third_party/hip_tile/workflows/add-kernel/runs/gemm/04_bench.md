---
status: done
stage: workflows/add-kernel/04_bench
date: 2026-10-25
---

# gemm — bench

Contract: [`workflows/add-kernel/04_bench/CONTEXT.md`](../../04_bench/CONTEXT.md).

## Contents

- [Configs](#configs)
- [Kept and why](#kept-and-why)
- [Commands](#commands)

## Configs

N=1024, all correct at 1.75e-6 (scratch harness, before the asm-barrier revert):

| Config | GFLOP/s | VGPRs |
|---|---|---|
| spike 64×64×16 (hand-written) | 1355 | 57 |
| tile API 64×64×16 | 1453 | 63 |
| 64×64×8 / 64×64×32 | 1271 / 1516 | 66 / 61 |
| 128×64×16 / 64×128×16 | 1647 / 1723 | 81 / 68 |
| 64×256×16 / 256×64×16 | 1866 / 1663 | — |
| 128×128×8 | 1805 | 127 |
| **128×128×16 (kept)** | 1941–1966 | 110 |
| 128×128×32 | 1984 | — |

Final, pure-Rust barrier, CPU idle, through `runner gemm --bench`: N=1024 **2.05 TFLOP/s** (rocBLAS 2.69); N=2048 **2.28 TFLOP/s** (rocBLAS 1.91).

## Kept and why

128×128×16: best measured config at reasonable LDS. Register prefetch of the next k tile gave no gain; BK=32 gave ~2% for double the LDS. The asm barrier (+5%) was reverted per `_shared/rules.md`.

## Commands

`cargo run -p runner --release -- gemm --n 1024 --bench` (method: `_shared/benchmarking.md`).

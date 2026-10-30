# lowprec

fp16, bf16, fp8 and fp4 work. ROADMAP_FP8.md is the goal document with a versioned todo and user gates.

## Table of Contents

- [Files](#files)

## Files

| File | Holds |
|---|---|
| [ROADMAP_FP8.md](ROADMAP_FP8.md) | Low-precision goal: state snapshot, loop, versioned roadmap todo (`v0.2.224`+), delegation and user gates |
| [plan_fp16.md](plan_fp16.md) | `-fp16` mixed-precision training for LegoGears_v2 |
| [plan_bf16.md](plan_bf16.md) | `-bf16` training via the shared `HalfFormat` machinery |
| [plan_fp8.md](plan_fp8.md) | fp8 e4m3/e5m2 faster than f16 AMP on sm_89, or proof it cannot be |
| [fp8_fp4.md](fp8_fp4.md) | FP8 GEMM and FP4 emulation design, sm_89 |

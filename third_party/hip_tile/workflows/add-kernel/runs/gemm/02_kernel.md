---
status: done
stage: workflows/add-kernel/02_kernel
date: 2026-10-25
---

# gemm — kernel

Contract: [`workflows/add-kernel/02_kernel/CONTEXT.md`](../../02_kernel/CONTEXT.md).

## Contents

- [Files](#files)
- [Launcher path](#launcher-path)
- [ISA](#isa)

## Files

`kernels/src/gemm.rs`; tile ops `load_tile`, `mma`, `store_tile` from `crates/hip-tile`.

## Launcher path

`kernels::gemm::gemm::launch`.

## ISA

110 VGPRs, `private_segment_fixed_size: 0`, no spills; 64 `v_fmac_f32` per k step, LDS reads as `ds_read2_b32`, 2 `s_barrier` per `mma`.

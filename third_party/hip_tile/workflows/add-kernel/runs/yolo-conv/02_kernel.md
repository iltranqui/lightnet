---
status: done
stage: workflows/add-kernel/02_kernel
date: 2026-10-25
---

# yolo-conv — kernel

Contract: [`workflows/add-kernel/02_kernel/CONTEXT.md`](../../02_kernel/CONTEXT.md).

## Contents

- [Files](#files)
- [Launcher path](#launcher-path)
- [ISA](#isa)

## Files

`kernels/src/yolo/{conv,maxpool,upsample,concat}.rs`. One `#[kernel]` per conv config via `macro_rules!` around an `#[inline(always)] conv_body<BM, BN>`; B tile built with `Tile::from_fn`; bias + activation fused through `map`.

## Launcher path

`kernels::yolo::{conv::launch_conv, maxpool::launch_maxpool, upsample::launch_upsample, concat::launch_concat}`; symbols `conv_64x64`, `conv_128x64`, `conv_32x128`, `conv_64x32`, `conv_32x32`, `maxpool`, `upsample`, `concat`.

## ISA

Not recorded in this run.

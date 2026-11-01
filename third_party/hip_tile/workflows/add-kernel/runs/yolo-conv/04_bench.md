---
status: done
stage: workflows/add-kernel/04_bench
date: 2026-10-25
---

# yolo-conv — bench

Contract: [`workflows/add-kernel/04_bench/CONTEXT.md`](../../04_bench/CONTEXT.md).

## Contents

- [Configs](#configs)
- [Kept and why](#kept-and-why)
- [Commands](#commands)

## Configs

Auto-chosen config, best of 5×20:

| Layer | Config | GFLOP/s |
|---|---|---|
| 32→64 k3 s2, 56×40 out | C64x64 | 850 |
| 64→64 k3, 56×40 | C64x64 | 970 |
| 128→128 k3, 28×20 | C64x64 | 880 |
| 256→256 k3, 14×10 | C64x32 | 590 |
| 512→512 k3, 7×5 | C32x32 | 306 |
| 512→256 k1, 7×5 | C32x32 | 169 |
| 512→30 k1, 7×5 (head) | C32x32 | 24 |

## Kept and why

`Config::choose` thresholds tuned from these runs. The 7×5 layers have few blocks and a long K loop; split-K is the next step (not done).

## Commands

`cargo run -p runner --release -- conv-bench`. Measured once with an idle CPU; not repeated for clock variance.

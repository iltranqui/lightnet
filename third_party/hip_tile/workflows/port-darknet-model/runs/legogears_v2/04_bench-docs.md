---
status: done
stage: workflows/port-darknet-model/04_bench-docs
date: 2026-10-25
---

# legogears_v2 — bench and docs

Contract: [`workflows/port-darknet-model/04_bench-docs/CONTEXT.md`](../../04_bench-docs/CONTEXT.md).

## Contents

- [Speed](#speed)
- [Annotated sample](#annotated-sample)
- [README](#readme)

## Speed

`--bench 200` on `00000082.png`, CPU idle:

| Path | Run 1 | Run 2 |
|---|---|---|
| GPU forward (upload + kernels + sync) | 4.83 ms | 4.91 ms |
| GPU end to end (resize, upload, download, decode) | 5.78 ms | 5.52 ms |
| CPU forward | 443 ms | 449 ms |

## Annotated sample

`docs/legogears_gpu.png`: medium gear, small gear, center, pin boxed correctly.

## README

Results table and demo image updated (`README.md`).

---
status: done
stage: workflows/port-darknet-model/02_gpu-ops
date: 2026-10-25
---

# legogears_v2 — GPU ops

Contract: [`workflows/port-darknet-model/02_gpu-ops/CONTEXT.md`](../../02_gpu-ops/CONTEXT.md).

## Contents

- [Layer to kernel map](#layer-to-kernel-map)
- [Test output](#test-output)

## Layer to kernel map

| Layer type | Kernel | Run |
|---|---|---|
| convolutional (k1/k3, s1/s2, leaky/linear) | `kernels::yolo::conv` | [yolo-conv](../../../add-kernel/runs/yolo-conv/) |
| maxpool | `kernels::yolo::maxpool` | yolo-conv |
| route concat | `kernels::yolo::concat` | yolo-conv |
| route groups | zero-copy `Tensor` view (no kernel) | yolo-conv |
| upsample | `kernels::yolo::upsample` | yolo-conv |
| yolo | CPU decode after download (no kernel) | — |

## Test output

`apps/runner/tests/yolo_ops.rs`: 14 passed.

---
status: done
stage: workflows/add-kernel/03_parity
date: 2026-10-25
---

# yolo-conv — parity

Contract: [`workflows/add-kernel/03_parity/CONTEXT.md`](../../03_parity/CONTEXT.md).

## Contents

- [Tests](#tests)
- [Worst error per shape](#worst-error-per-shape)
- [Test output](#test-output)

## Tests

`apps/runner/tests/yolo_ops.rs`: conv at the real model shapes, ragged shapes, all 5 configs, channel-slice writes with poison checks; maxpool (2/2, 3/2, 3/1 pad 2, pad 0); upsample 128×7×5→14×10; concat incl. a route-group view.

## Worst error per shape

All conv shapes ≤ 1e-4 relative; maxpool, upsample, concat exact.

## Test output

`tests/yolo_ops.rs`: 14 passed.

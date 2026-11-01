---
status: done
stage: workflows/port-darknet-model/03_executor
date: 2026-10-25
---

# legogears_v2 — executor

Contract: [`workflows/port-darknet-model/03_executor/CONTEXT.md`](../../03_executor/CONTEXT.md).

## Contents

- [Per-layer parity](#per-layer-parity)
- [Detection parity](#detection-parity)
- [Test output](#test-output)

## Per-layer parity

3 images, 38 layers: max abs error ≤ 1.9e-5; max relative error 1.7e-4 at worst (layer 32), all layers under 1e-3. Full table: `cargo test -p hip_darknet --release --test gpu -- --nocapture`.

## Detection parity

10 labelled images: GPU detections identical to CPU (count, class, IoU ≥ 0.99, confidence within 1e-3). Ground truth: recall 1.000, precision 1.000, 0 class errors.

## Test output

`cargo test -p hip_darknet --release`: 25 unit + 1 detect + 2 gpu passed; `--no-default-features`: 25 + 1 passed.

---
status: template
stage: workflows/train-darknet-model/03_gpu-backward
---

# {model} — GPU backward

Contract: [`workflows/train-darknet-model/03_gpu-backward/CONTEXT.md`](../../03_gpu-backward/CONTEXT.md). Set `status: done` when the human check has passed.

## Contents

- [New tile ops](#new-tile-ops)
- [Op to kernel map](#op-to-kernel-map)
- [Parity](#parity)
- [Test output](#test-output)

## New tile ops

| Op | Why it cannot live in a kernel | Location |
|---|---|---|

## Op to kernel map

| Op | Kernel | add-kernel run | Batched? |
|---|---|---|---|

## Parity

| Kernel | GPU vs CPU (tolerance, measured) | Finite differences |
|---|---|---|

## Test output

{`test result` lines, plus the batch-1 inference suites re-run unchanged}

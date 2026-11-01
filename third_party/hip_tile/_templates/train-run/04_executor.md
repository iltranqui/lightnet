---
status: template
stage: workflows/train-darknet-model/04_executor
---

# {model} — GPU training step

Contract: [`workflows/train-darknet-model/04_executor/CONTEXT.md`](../../04_executor/CONTEXT.md). Set `status: done` when the human check has passed.

## Contents

- [Step shape](#step-shape)
- [Parity](#parity)
- [Timing probe](#timing-probe)
- [Test output](#test-output)

## Step shape

{buffers, batch size, accumulation, what is device-resident, syncs per iteration}

## Parity

| Quantity | CPU reference | GPU | Error |
|---|---|---|---|

## Timing probe

{200 iterations: ms/iteration, images/s, projected full-run wall clock}

## Test output

{`test result` lines}

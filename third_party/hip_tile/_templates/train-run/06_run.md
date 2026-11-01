---
status: template
stage: workflows/train-darknet-model/06_run
---

# {model} — the run

Contract: [`workflows/train-darknet-model/06_run/CONTEXT.md`](../../06_run/CONTEXT.md). Set `status: done` when the human check has passed.

## Contents

- [Command](#command)
- [Loss](#loss)
- [mAP](#map)
- [Round trip](#round-trip)
- [Verdict](#verdict)

## Command

{full command line, seed, iteration count, cfg overrides, wall clock}

## Loss

| Images seen | Reference loss | Ours | Reference lr | Our lr |
|---|---|---|---|---|

## mAP

| Iteration | Reference mAP@0.5 | Ours mAP@0.5 | Ours mAP@0.5:0.95 |
|---|---|---|---|

## Round trip

{final weights loaded back through inference: ground-truth table}

## Verdict

{acceptance numbers met or not; if not, the diagnosis order followed}

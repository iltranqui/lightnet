# DCNv4 device adjoint continuation

## Contents

- [Contract](#contract)
- [Checklist](#checklist)
- [Evidence and findings](#evidence-and-findings)
- [Status](#status)

## Contract

Finish the pending native DCNv4 aggregation adjoint (offsets given), verified
against the Rust host reference and actual-device finite differences. Preserve
runtime fallback, optimizer behavior, unrelated YOLO-3D/user edits and all open
CPU/GPU requirements. No training runs, subagents or compiled-C++ parity claim.
Compose the full DCNv4 predictor/backward and wire runtime only in later slices.
Use serial, memory-capped builds through `tools/test_oxide.sh` for Oxide.

## Checklist

- [x] Inspect current artifacts and version ledger: v0.2.010 forward exists;
  three uncommitted Oxide files contain the pending adjoint.
- [x] Reproduce failing adjoint test before editing device math (batch 1 input
  adjoint 0 vs 0.67130804; 1 failed/2 passed).
- [x] Validate batches/groups, all softmax/center modes, signed seeds,
  padding/stride/dilation, malformed storage/geometry and overflow rejection.
- [x] Check every input and offset derivative with actual-device finite
  differences using half-cell coordinates and epsilon=0.01.
- [x] Device suite 6, host-cuda 35, host dcnv4 17, Oxide runtime fallback 1;
  targeted Rustfmt and diff checks passed.
- [x] Requirement-to-artifact audit in `implemented.md` §48; code committed
  as v0.2.011 (`8fe32a09`); docs and version map updated.

## Evidence and findings

- Pending backward uses `channels * spatial / cpg` as its thread bound, which
  excludes every batch after batch zero. Forward correctly derives its bound
  from the full output length. Existing parity fixture already has batch=2.
- Rust skill knowledge directory `~/.Codex/skills/rust-dev-knowledge` is absent;
  repository implementations and tests are the available local references.
- Root `task_plan.md` and `notes.md` describe earlier CARPN work; preserve them.

## Status

Bounded adjoint slice complete. Next open work is full predictor/upsample and
aggregation forward/backward composition against real CPU parameter gradients,
then trait/runtime wiring and optimizer parity. Keep the runtime fallback until
those gates pass. Overall handbook objective remains open: Stage 1 missing
variants/axis pins, Stage 2 native layers/GPU Muon and Stage 3 sparse MoE are not
complete; this primitive's six tests cannot establish those deliverables.
Full audit/evidence is in `implemented.md` §48; resume in `gpu_layers_plan.md`.

No new weights, sidecars or logs were created; existing unrelated build trees
and compiler IR/PTX were inspected and retained (not this slice's artifacts).

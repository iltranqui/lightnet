# train-darknet-model — pipeline

Trains a Darknet model from scratch (`.cfg` + a labelled dataset) through `crates/hip_darknet`, in pure Rust on the AMD GPU, and proves the result against the dataset's own reference training history. The unit of work is one model+dataset pair; each run is `runs/<name>/`.

Goal, decisions, acceptance numbers and risks live in [`docs/ROADMAP.md`](../../docs/ROADMAP.md) — one home. This file only says how a run moves.

## Contents

- [Stages](#stages)
- [Starting a run](#starting-a-run)
- [Status](#status)

## Stages

| Stage | Job | Writes |
|---|---|---|
| `01_train-semantics/` | cfg training keys, dataset + augmentation, loss scaling; the facts the rest of the run rests on | `_shared/darknet-semantics.md` Training sections + `runs/<name>/01_train-semantics.md` |
| `02_cpu-backward/` | training-mode forward, loss + deltas, backward, SGD on the CPU (f32), gradient-checked | `crates/hip_darknet/src/train/` + `runs/<name>/02_cpu-backward.md` |
| `03_gpu-backward/` | tile ops in `hip-tile` and kernels for every backward pass, incl. batched forward | `crates/hip-tile/src/ops/*`, `kernels/src/`, `apps/runner/tests/*` + `runs/<name>/03_gpu-backward.md` |
| `04_executor/` | one `TrainModel` stepping forward→loss→backward→update on the GPU, parity with stage 02 | `crates/hip_darknet/src/train/gpu.rs` + `runs/<name>/04_executor.md` |
| `05_cli-checkpoints/` | `train` subcommand, Darknet-compatible `.weights` writer, `loss.csv`/`map.csv`, resume | `src/main.rs`, `src/metrics.rs`, `src/train/checkpoint.rs` + `runs/<name>/05_cli-checkpoints.md` |
| `06_run/` | the full training run, compared to the dataset's reference history | trained `.weights`, `loss.csv`, `map.csv` + `runs/<name>/06_run.md` |
| `07_bench-docs/` | speed, memory, docs | `runs/<name>/07_bench-docs.md` + CONTEXT/README updates |

Every stage reads `_shared/rules.md` first: no Darknet binary ever, no inline asm, pure Rust, GPU tests before a stage counts as done. Training math is f32 end to end and the optimizer is SGD with momentum — see `docs/ROADMAP.md` Decisions.

Nothing moves to the next stage until a person has read the previous stage's run file.

## Starting a run

`cp -r _templates/train-run workflows/train-darknet-model/runs/<name>`, fill in the model and dataset paths at the top of `01_train-semantics.md`, then open `01_train-semantics/CONTEXT.md`.

## Status

A run's stage is the highest-numbered `0N_*.md` in `runs/<name>/` whose frontmatter says `status: done`. `status: review` means written and waiting for the human read; `status: template` means untouched.

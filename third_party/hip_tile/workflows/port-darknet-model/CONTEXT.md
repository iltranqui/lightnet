# port-darknet-model — pipeline

Runs one Darknet model (`.cfg` + `.weights`) on the AMD GPU through `crates/hip_darknet`, as a direct Rust port verified against a CPU reference and the dataset labels. The unit of work is one model; each run is `runs/<model-name>/`.

## Contents

- [Stages](#stages)
- [Starting a run](#starting-a-run)
- [Status](#status)

## Stages

| Stage | Job | Writes |
|---|---|---|
| `01_cfg-cpu/` | parse, load, CPU reference forward, decode; match ground truth | `crates/hip_darknet` + `runs/<model>/01_cfg-cpu.md` |
| `02_gpu-ops/` | GPU kernels for every layer type the model uses | `kernels/src/yolo/` via `add-kernel` runs + `runs/<model>/02_gpu-ops.md` |
| `03_executor/` | whole network on the GPU, layer-by-layer parity | `crates/hip_darknet/src/gpu.rs` + `runs/<model>/03_executor.md` |
| `04_bench-docs/` | speed, annotated output, README | `runs/<model>/04_bench-docs.md` + README |

Every stage reads `_shared/rules.md`: no Darknet binary, ever. A model that only uses already-supported layers skips most of stages 01–03 (record that in the run file).

## Starting a run

`cp -r _templates/port-run workflows/port-darknet-model/runs/<model-name>`, fill in the model paths at the top of `01_cfg-cpu.md`, then open `01_cfg-cpu/CONTEXT.md`.

## Status

A run's stage is the highest-numbered `0N_*.md` in `runs/<model>/` whose frontmatter says `status: done`.

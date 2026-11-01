# docs — README assets and the training roadmap

Two jobs: hold images referenced by `README.md`, and hold `ROADMAP.md`, the training goal with its decisions, acceptance numbers, stages and status.

## Contents

- [Files](#files)
- [Rules for changes](#rules-for-changes)

## Files

- `ROADMAP.md`: training `hip_darknet` end to end (acceptance, decisions, stages, status, risks). Read by `workflows/train-darknet-model/`; update its status table when a stage finishes.
- `legogears_gpu.png`: GPU detections on LegoGears validation image `00000082.png` (from `hip_darknet --gpu --out`).

## Rules for changes

New images come from stage `04_bench-docs` of a model port, named `<model>_gpu.png`.

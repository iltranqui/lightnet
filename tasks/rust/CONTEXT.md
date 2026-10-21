# tasks/rust

Roadmap, plans and status for the Rust port. Entry point is `AGENTS.md`; this file only routes.

## Table of Contents

- [Read first](#read-first)
- [Shelves](#shelves)
- [Constraints](#constraints)

## Read first

`AGENTS.md` (rules, commands, versioning, file map), then `STATUS.md` after a context reset.

## Shelves

| Kind | Files |
|---|---|
| State | `STATUS.md`, `todo.md`, `implemented.md`, `version-map.tsv` |
| Coverage | `layer-support.md`, `layers.md`, `axis-coverage.md` |
| Plans | `*_plan.md`, `plan_fp16.md`, `plan_bf16.md`, `plan_fp8.md`, `AGENTS_fp8.md` (goal for the fp16/bf16/fp8 work), `fp8_fp4.md`, `onnx.md`, `plan-open-items.md` |
| CARPN history | `carpn_task_plan.md`, `carpn_notes.md` |
| Lessons | `lessons.md` |

## Constraints

- Completed plans stay: they are the roadmap.
- Only `STATUS.md`, `todo.md`, `layers.md`, `lessons.md`, `fp8_fp4.md` are tracked; others need `git add -f`.
- This is the branch-level task folder for branch `rust`; do not duplicate it under `tasks/<branch>/`.

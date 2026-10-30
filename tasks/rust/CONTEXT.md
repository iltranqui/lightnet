# tasks/rust

Roadmap, plans and status for the Rust port (branch `rust`). Entry point is [AGENTS.md](AGENTS.md); this file is the folder map.

## Table of Contents

- [Workflow](#workflow)
- [Folder map](#folder-map)
- [Constraints](#constraints)

## Workflow

`01_rules` (read first) -> pick an item in `todo.md` / `02_state/open-work.md` -> follow its plan in `03_plans/` and the contract in `04_reference/` -> build and verify -> record the result in `02_state/`. State is what exists in `02_state/`: if `implemented.md`, `STATUS.md` and `version-map.tsv` have no row for a version, the version is not done.

## Folder map

| Path | Role | Kind |
|---|---|---|
| `AGENTS.md` | Entry, goal loop and the ordered agent todo list | catalog |
| `todo.md` | Long backlog by area with history (global convention path `tasks/<branch>/todo.md`) | state |
| `lessons.md` | Rules from past corrections (global convention path) | factory |
| [01_rules/](01_rules/CONTEXT.md) | Rules, commands, versioning, layer contract | factory |
| [02_state/](02_state/CONTEXT.md) | Status, done ledger, layer matrices, version map | product |
| [03_plans/](03_plans/CONTEXT.md) | Plans by area: `gpu/`, `cpu-layers/`, `lowprec/`, `llm/`, `misc/` | plans |
| [04_reference/](04_reference/CONTEXT.md) | Layer Port Contract, source notes, history | factory |

## Constraints

- Completed plans stay: they are the roadmap. Nothing is archived or deleted.
- All files here are tracked; keep one home per fact and link instead of copying.
- This is the branch-level task folder; do not duplicate it under `tasks/<branch>/`.
- Each `.md` needs a TOC.

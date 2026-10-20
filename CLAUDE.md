# lightnet

Rust port of Darknet/YOLO (branch `rust`) with CPU, cuTile and cuda-oxide GPU backends. This file only routes; read the file it points to for the task at hand.

## Table of Contents

- [Where things live](#where-things-live)
- [Which file for which task](#which-file-for-which-task)
- [Hard rules](#hard-rules)
- [Learned rules](#learned-rules)

## Where things live

| Zone | Path | Entry file |
|---|---|---|
| Product (Darknet port) | `src/darknet-*` | `src/<crate>/CONTEXT.md` |
| cuda-oxide (submodule) | `third_party/cuda-oxide` | `third_party/CONTEXT.md` |
| Roadmap and plans | `tasks/rust/` | `tasks/rust/CONTEXT.md` |
| Reference docs | `docs/` | `docs/CONTEXT.md` |
| Model configs | `cfg/` | `cfg/CONTEXT.md` |
| Scripts | `tools/` | `tools/CONTEXT.md` |

Whole-repo map: [CONTEXT.md](CONTEXT.md).

## Which file for which task

- Port or change a layer: `tasks/rust/AGENTS.md`, then `tasks/rust/layer-support.md` and `tasks/rust/layers.md`.
- GPU layer work: `tasks/rust/gpu_layers_plan.md`.
- Current state after a context reset: `tasks/rust/STATUS.md`.
- Touch a crate: its `CONTEXT.md` first.
- Touch cuda-oxide: read `third_party/CONTEXT.md` first; it is a pinned submodule, not editable here.

## Hard rules

- Max 4 build threads, wrap builds in `systemd-run --user --scope -p MemoryHigh=3G`.
- `src/darknet-cuda-oxide` is not a workspace member; build it only with `./tools/test_oxide.sh`.
- Commit only the files a task names, never `git add -A`.

Full rule set lives in `tasks/rust/AGENTS.md`.

<!-- claude-reflect: learned rules (saved from the band; edit freely) -->
## Learned rules
- Delete model weights and other big files from the repository when they are no longer needed.
- Do not archive or remove completed plans in tasks/rust/; keep them, since they are the project roadmap needed for development.
<!-- end claude-reflect -->

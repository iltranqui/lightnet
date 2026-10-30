# lightnet: repository map

One-screen description of the repo. Details live in the linked files, not here.

## Table of Contents

- [Zones](#zones)
- [Crate graph](#crate-graph)
- [State and roadmap](#state-and-roadmap)
- [Factory vs product](#factory-vs-product)

## Zones

```
lightnet/
├─ CLAUDE.md             routing table
├─ CONTEXT.md            this file
├─ src/                  PRODUCT: Darknet Rust port (9 crates)
├─ third_party/          SUBMODULES: cuda-oxide (pinned 728539f65), local patches
├─ docs/                 reference shelf: architecture, parity, benchmarks
├─ cfg/                  factory: model .cfg files
├─ tools/                scripts: oxide test/bench/parity helpers
└─ tasks/rust/           roadmap, plans, status (see below)
```

## Crate graph

```
lightnet-cli ──► lightnet-core ──► lightnet-cuda
     │                                ▲
     ├─(feature oxide)─► lightnet-cuda-oxide ──► cuda-oxide crates cuda-device, cuda-host
     │                       (not a workspace member)
lightnet-cuda-cutile ───────────────────┘   (not a workspace member)
lightnet-examples ──► lightnet-core
lightnet-lm ──► lightnet-cuda
lightnet-gui ──► lightnet-gui-project, lightnet-gui-engine ──(child process)──► lightnet-rs detect-server
```

Each crate's role, responsibilities and constraints are in `src/<crate>/CONTEXT.md`.

## State and roadmap

Most files in `tasks/rust/` are tracked. Check with `git ls-files tasks/rust/<file>`; only an untracked file needs `git add -f`.

| Need | File |
|---|---|
| Rules, commands, versioning, file map | `tasks/rust/AGENTS.md` |
| Verified state after a reset | `tasks/rust/02_state/STATUS.md` |
| What is implemented | `tasks/rust/02_state/implemented.md` |
| Per-layer CPU/GPU/f16/axis status | `tasks/rust/02_state/layer-support.md` |
| Open work | `tasks/rust/todo.md`, `tasks/rust/03_plans/cpu-layers/plan-open-items.md` |
| Feature plans | `tasks/rust/*_plan.md`, `fp8_fp4.md`, `onnx.md` |
| Version to commit map | `tasks/rust/02_state/version-map.tsv` |

Completed plans stay in place: they are the roadmap.

## Factory vs product

- Factory (stable across runs): `cfg/`, `tools/`, `third_party/`, `tasks/rust/AGENTS.md`.
- Product (changes with the work): `src/`, `tasks/rust/02_state/STATUS.md`, `todo.md`, `implemented.md`.

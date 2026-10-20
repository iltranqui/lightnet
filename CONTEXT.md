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
├─ bridges/              VENDORED: NVlabs/cuda-oxide crates, treated as opaque
├─ docs/                 reference shelf: architecture, parity, benchmarks
├─ cfg/                  factory: model .cfg files
├─ tools/                scripts: oxide test/bench/parity helpers
└─ tasks/rust/           roadmap, plans, status (see below)
```

## Crate graph

```
darknet-cli ──► darknet-core ──► darknet-cuda
     │                                ▲
     ├─(feature oxide)─► darknet-cuda-oxide ──► bridges/cuda-device, cuda-host
     │                       (not a workspace member)
darknet-cuda-cutile ───────────────────┘   (not a workspace member)
darknet-examples ──► darknet-core
darknet-gui ──► darknet-gui-project, darknet-gui-engine ──(child process)──► darknet-rs detect-server
```

Each crate's role, responsibilities and constraints are in `src/<crate>/CONTEXT.md`.

## State and roadmap

`tasks/` is listed in `.git/info/exclude`: only `STATUS.md`, `todo.md`, `layers.md`, `lessons.md`, `fp8_fp4.md` are tracked; other files are local and need `git add -f`.

| Need | File |
|---|---|
| Rules, commands, versioning, file map | `tasks/rust/AGENTS.md` |
| Verified state after a reset | `tasks/rust/STATUS.md` |
| What is implemented | `tasks/rust/implemented.md` |
| Per-layer CPU/GPU/f16/axis status | `tasks/rust/layer-support.md` |
| Open work | `tasks/rust/todo.md`, `tasks/rust/plan-open-items.md` |
| Feature plans | `tasks/rust/*_plan.md`, `fp8_fp4.md`, `onnx.md` |
| Version to commit map | `tasks/rust/version-map.tsv` |

Completed plans stay in place: they are the roadmap.

## Factory vs product

- Factory (stable across runs): `cfg/`, `tools/`, `bridges/`, `tasks/rust/AGENTS.md`.
- Product (changes with the work): `src/`, `tasks/rust/STATUS.md`, `todo.md`, `implemented.md`.

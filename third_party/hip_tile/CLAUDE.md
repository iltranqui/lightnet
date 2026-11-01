# hip-tile

cuTile-style GPU kernels for AMD in pure Rust, plus `hip_darknet`, a Rust port of Darknet YOLO that runs on them. What leaves this repo: Rust crates, kernels, and verified model ports.

Built on ICM: folders carry sequencing, hierarchy carries context, files carry state. If something needs explaining, it goes in that folder's `CONTEXT.md`.

## Contents

- [Where things live](#where-things-live)
- [Route by task](#route-by-task)
- [The one rule](#the-one-rule)
- [Learned rules](#learned-rules)

## Where things live

| Folder | What it holds |
|---|---|
| `CONTEXT.md` | workspace map: crates, dependency graph, workflows |
| `_shared/` | factory: hard rules, tile model, Darknet semantics, benchmarking method |
| `workflows/` | the two pipelines (`add-kernel`, `port-darknet-model`); `runs/` hold each run's record |
| `_templates/` | starters: copy, don't start blank |
| `crates/`, `kernels/`, `apps/` | the code; each folder has its own `CONTEXT.md` |
| `tasks/main/` | todo + lessons for branch `main` |

## Route by task

| If | Go to | Then stop at |
|---|---|---|
| any change at all | `_shared/rules.md` | — (read once) |
| new kernel or tile op | `workflows/add-kernel/CONTEXT.md` | human reads each stage's run file |
| new Darknet model or layer | `workflows/port-darknet-model/CONTEXT.md` | human reads each stage's run file |
| training a Darknet model from scratch | `docs/ROADMAP.md`, then `workflows/train-darknet-model/CONTEXT.md` | human reads each stage's run file |
| fixing code in one crate | that folder's `CONTEXT.md` | tests pass, output shown |
| status of a workflow | scan `workflows/*/runs/*/` | report which stage files exist |
| performance question | `_shared/benchmarking.md` | numbers measured per its method |

## The one rule

Nothing moves to the next stage until a person has read the output of the last one.

<!-- claude-reflect: learned rules (saved from the band; edit freely) -->
## Learned rules
- Keep kernels in pure Rust and let the LLVM backend handle codegen; avoid hand-written AMDGCN inline asm.
<!-- end claude-reflect -->

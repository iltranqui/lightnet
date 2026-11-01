# add-kernel — pipeline

Adds one GPU kernel (or one new tile op in `crates/hip-tile`) from a spec to a measured, tested launcher. The unit of work is one kernel; each run is `runs/<kernel-name>/`.

## Contents

- [Stages](#stages)
- [Starting a run](#starting-a-run)
- [Status](#status)

## Stages

| Stage | Job | Writes |
|---|---|---|
| `01_spec/` | define the math, shapes, tile config, LDS | `runs/<name>/01_spec.md` |
| `02_kernel/` | write the `#[kernel]` and host helper | code in `kernels/src/` + `runs/<name>/02_kernel.md` |
| `03_parity/` | GPU vs CPU reference tests | `apps/runner/tests/*.rs` + `runs/<name>/03_parity.md` |
| `04_bench/` | measure, tune, record | `runs/<name>/04_bench.md` |

Every stage reads `_shared/rules.md`. Stop after each stage for a human read.

## Starting a run

`cp -r _templates/kernel workflows/add-kernel/runs/<name>`, then open `01_spec/CONTEXT.md`.

## Status

A run's stage is the highest-numbered `0N_*.md` in `runs/<name>/` whose frontmatter says `status: done` (copied templates say `status: template` until written).

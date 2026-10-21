# Low-Precision Training Goal: fp16, bf16, fp8

Goal document for an agent (the "lead") that delegates the implementation of three plans. Read this file first, then only the plan of the phase you are working on. It extends [AGENTS.md](AGENTS.md) for this one goal.

## Table of Contents

- [Start here](#start-here)
- [Goal](#goal)
- [Done when](#done-when)
- [Plans](#plans)
- [Current state](#current-state)
- [Rules](#rules)
- [Delegation protocol](#delegation-protocol)
- [User-gated actions](#user-gated-actions)
- [Reporting](#reporting)
- [Documentation updates](#documentation-updates)

## Start here

Begin at [plan_fp16.md](plan_fp16.md) Task 1. The next free version number is `v0.2.160` (`version-map.tsv` ends at `v0.2.159`); take later numbers from that file. Earlier v0.2 work that `AGENTS.md` lists as unversioned in the working tree keeps its place: do not renumber it, and never reuse a number. Work phases in order and tasks in plan order.

## Goal

Make LegoGears_v2 training faster with lower-precision arithmetic on the sm_89 GPU, selectable from the command line:

```
cargo xtask train_legogears -fp16
cargo xtask train_legogears -bf16
cargo xtask train_legogears -fp8
```

The three phases run in this order, each one finished and recorded before the next starts:

1. **fp16**: harden the existing mixed-precision (AMP) mode.
2. **bf16**: add a bf16 mixed-precision mode on the same machinery.
3. **fp8**: remove the overhead that makes fp8 slower than fp16 today, tune the kernel, combine it with 16-bit activation storage, and decide with measurements whether it is a speed feature.

## Done when

- Phase 1 ([plan_fp16.md](plan_fp16.md)): all tasks done; a recorded fp32-versus-fp16 convergence run meets the pass criterion in its Task 5.
- Phase 2 ([plan_bf16.md](plan_bf16.md)): all tasks done; a recorded bf16 run meets the pass criterion in its Task 8; every f16 test still passes unmodified.
- Phase 3 ([plan_fp8.md](plan_fp8.md)): all tasks done; the Task 9 gate is evaluated and its verdict recorded. A "numerics only, not faster" verdict is a valid finish; an unmeasured claim of speed is not.
- Every task's tests pass, `cargo xtask clippy` is clean for touched crates, docs are updated as in [Documentation updates](#documentation-updates), and the `CLAUDE.md` Commands table matches the xtask flags.
- No parity claim without a golden. Results are compared against the Rust fp32 run; there is no C++ FP8 build.

## Plans

| Phase | Plan | Depends on |
|---|---|---|
| 1 fp16 | [plan_fp16.md](plan_fp16.md) | none |
| 2 bf16 | [plan_bf16.md](plan_bf16.md) | Phase 1 Tasks 1 to 4 (the scaler and xtask return shape) |
| 3 fp8 | [plan_fp8.md](plan_fp8.md) | Phase 2 Tasks 1, 2 and 6 for fp8 Tasks 4 and 6; fp8 Tasks 1, 2, 3, 5 only need the current code |

Related reading: [fp8_fp4.md](fp8_fp4.md) (earlier fp8 design and the Phase D numbers), `docs/OXIDE_BENCH.md` section 7.2 (AMP measurements).

## Current state

Verified by reading the code:

- fp16 AMP exists (`DARKNET_OXIDE_AMP=1`, `src/darknet-cuda-oxide/src/mixed.rs`); all LegoGears_v2 layers except the yolo loss have f16 paths; measured 1.3x faster at batch 16; no convergence run; a dynamic loss scaler exists inline in the CLI (`-dyn_scale`) without tests.
- bf16: only the tensor-core GEMM (f32 output) and the conversion kernels exist.
- fp8: `DARKNET_OXIDE_GEMM=fp8` works but is slower than fp16 (21 LegoGears convs: 24.2 ms against 3.87 ms) because of per-call host readbacks, per-call weight quantisation and an untuned kernel.
- `cargo xtask train_legogears` maps `-fp8` to `DARKNET_OXIDE_GEMM=fp8` and `-fp16` to `DARKNET_OXIDE_AMP=1`; `-bf16` is rejected until Phase 2.

## Rules

- Use `cargo xtask` for every check, build, test and training command (`check`, `build`, `clippy`, `test`, `oxide`, `fmt`, `train_legogears`); do not type raw `cargo` or `systemd-run`.
- One build or test command at a time. Build threads are capped at 4 (`train_legogears` runs 8, operator-approved); memory is capped at `MemoryHigh=3G`, applied by `cargo xtask` (it warns and runs uncapped when there is no user systemd bus).
- `src/darknet-cuda-oxide` is not a workspace member: run its tests with `cargo xtask oxide --test <file> [-- <filter>]`.
- Commit only the files a task names; never `git add -A`. `src/darknet-core/src/runtime/mod.rs` can carry unrelated edits from the user: stage only your hunk with `git add -p`.
- Commit and tag on green, as in [AGENTS.md](AGENTS.md#versioning) ("Commit and tag on green"): every plan task whose tests pass is committed and tagged at once, by whoever implemented it (Claude or Codex), without asking. Each such task takes the next free `v0.2.NNN` number from `version-map.tsv`; the commit message is `v0.2.NNN: <what>` where `<what>` is the plan step's suggested message, the commit body lists the tests run and their results, and the commit is tagged `v0.2.NNN` (local, lightweight). A task whose tests fail, time out or were not run gets neither a commit nor a tag. Measurement-only or doc-only steps (convergence records) are `docs(rust): <what>` with no tag. Never push.
- Never edit `third_party/` or `src/cutile-rs-aot` (pinned submodules).
- Delete weights, backups, logs and other big files a task produces once the evidence is recorded.
- Do not archive or remove completed plans in `tasks/rust/`; they are the roadmap.
- Keep existing comments in files you edit.
- This goal explicitly overrides the "no subagents" operator override in [AGENTS.md](AGENTS.md) for this work: delegation is expected.

## Delegation protocol

The lead reads this file and the current plan, then runs the plan with `superpowers:subagent-driven-development`.

1. One subagent per plan task (one tack each), in plan order. Give the subagent the full task text from the plan (the plan's Interfaces block is how it learns names and types), the Global Constraints and Review Focus sections of that plan, and the file paths it will touch. Do not ask it to read other plans.
2. Implementation subagents run on Sonnet. Exploration subagents (read-only fact finding) also run on Sonnet. The lead verifies the facts that a task depends on by reading the cited lines itself, because earlier exploration summaries contained at least one wrong claim (they reported no dynamic loss scaler when `-dyn_scale` existed).
3. After each task a fresh reviewer subagent checks the diff against the task text and the tests; the lead merges nothing that has failing tests or unexplained changes outside the named files.
4. Phase boundaries: after the last task of a phase, run the whole touched test files (`cargo xtask oxide --test oxide_amp`, `oxide_half`, `oxide_gemm_f8`, `oxide_runtime_f16`, and `cargo xtask test`), then a whole-phase review on the most capable available model before moving on.
5. If a measurement contradicts the plan (a kernel change does not speed up, a pass criterion fails), stop that thread, record the result in the task's commit message and the docs, and re-plan before continuing; do not keep pushing.
6. Subagents never start training runs, push, or touch files outside their task.

## User-gated actions

Ask the user before each of these; do not infer approval from an earlier one:

- Starting any training run (plan_fp16 Task 5, plan_bf16 Task 8, plan_fp8 Task 9). The user chooses the iteration count.
- Pushing, opening a pull request, or rewriting history.
- Changing the default of a user-visible mode (for example the default fp8 scaling mode in plan_fp8 Task 9).
- Bumping a submodule pin.

## Reporting

At the end of each task, one line: task, tests run and their result, files changed, commit hash and tag (`v0.2.NNN`), or why there is no commit (failing, timed-out or unrun tests). At the end of each phase, a short summary with the measurements (step time, loss and mAP against the baseline) and the verdict against the "Done when" criteria. Report failures with the command output; never describe a failing run as passing.

## Documentation updates

For every version-mapped increment (next free `v0.2.NNN` in `version-map.tsv`; never reuse a number): add the evidence line to `implemented.md`, an entry in `STATUS.md` "Latest increments", tick `todo.md`, and add the row to `version-map.tsv`. Record benchmark and convergence results in `docs/OXIDE_BENCH.md` (the AMP section 7.2 for fp16 and bf16; a new fp8 subsection for Phase 3). Keep the `CLAUDE.md` Commands table and `xtask help` in sync with every flag change. Every `.md` file keeps its table of contents in sync with its headings.

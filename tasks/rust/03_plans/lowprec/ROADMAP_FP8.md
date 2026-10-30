# Low-Precision Roadmap: fp16, bf16, fp8

Goal document and ordered work list for an agent (the "lead") on the low-precision training line. Read this file first, then only the plan of the phase you are working on. It extends [AGENTS.md](../../AGENTS.md) for this one goal; the global todo there and this roadmap share the `v0.2.NNN` number line.

## Table of Contents

- [Goal](#goal)
- [Done when](#done-when)
- [State snapshot](#state-snapshot)
- [Plans](#plans)
- [Running this file as a goal](#running-this-file-as-a-goal)
  - [Loop](#loop)
  - [Definition of done for one item](#definition-of-done-for-one-item)
  - [Stop conditions](#stop-conditions)
- [Roadmap Todo](#roadmap-todo)
  - [Phase 0: Baseline](#phase-0-baseline)
  - [Phase 1: fp16](#phase-1-fp16)
  - [Phase 2: bf16](#phase-2-bf16)
  - [Phase 3: fp8](#phase-3-fp8)
  - [Phase 4: Follow-ups](#phase-4-follow-ups)
  - [Blocked or out of scope](#blocked-or-out-of-scope)
- [Rules](#rules)
- [Delegation protocol](#delegation-protocol)
- [User-gated actions](#user-gated-actions)
- [Reporting](#reporting)
- [Documentation updates](#documentation-updates)
- [Keeping this file current](#keeping-this-file-current)

## Goal

Make LegoGears_v2 training faster with lower-precision arithmetic on the sm_89 GPU, selectable from the command line:

```
cargo xtask train_legogears -fp16
cargo xtask train_legogears -bf16
cargo xtask train_legogears -fp8
```

The three phases run in this order, each one finished and recorded before the next starts:

1. **Phase 1: fp16**: harden the existing mixed-precision (AMP) mode.
2. **Phase 2: bf16**: add a bf16 mixed-precision mode on the same machinery.
3. **Phase 3: fp8**: remove the overhead that makes fp8 slower than fp16 today, tune the kernel, combine it with 16-bit activation storage, and decide with measurements whether it is a speed feature.

## Done when

- Phase 1 ([plan_fp16.md](plan_fp16.md)): all tasks done; a recorded fp32-versus-fp16 convergence run meets the pass criterion in its Task 5.
- Phase 2 ([plan_bf16.md](plan_bf16.md)): all tasks done; a recorded bf16 run meets the pass criterion in its Task 8; every f16 test still passes unmodified.
- Phase 3 ([plan_fp8.md](plan_fp8.md)): all tasks done; the Task 9 gate is evaluated and its verdict recorded. A "numerics only, not faster" verdict is a valid finish; an unmeasured claim of speed is not.
- Every task's tests pass, `cargo xtask clippy` is clean for touched crates, docs are updated as in [Documentation updates](#documentation-updates), and the `CLAUDE.md` Commands table matches the xtask flags.
- No parity claim without a golden. Results are compared against the Rust fp32 run; there is no C++ FP8 build.

## State snapshot

State on 2026-10-27, from a static read of the plans, `version-map.tsv`, `xtask/src/main.rs` and `docs/OXIDE_BENCH.md` (no cargo run):

- **Phase 1 closed.** fp16 AMP (`LIGHTNET_OXIDE_AMP=1`, `src/lightnet-cuda-oxide/src/mixed.rs`), the public `LossScaler`, `-loss_scale`, overflow coverage and xtask defaults are done (`v0.2.161`–`v0.2.165`). Recorded in `docs/OXIDE_BENCH.md` §7.2.1: seed 42, 3000 iterations, fp16 loss 0.0525 against fp32 0.0597, mAP 99.63% against 99.55%. Accuracy gate passes; fp16 is 67.5 ms against fp32 48.7 ms, so it is not a speed feature.
- **Phase 2 closed.** bf16 is done (`v0.2.170`–`v0.2.176`): `LIGHTNET_OXIDE_AMP=bf16` and `-bf16` train on bf16 storage. `docs/OXIDE_BENCH.md` §10 records 138.6 s, loss 0.0524, mAP 99.63% against fp32 147.1 s. Open: `gemm_bf16_parity_all_transposes_alignments_alpha_beta` failed once on `OxideBackend::new()` and passed in 10 later runs; the cause is unknown.
- **Phase 3 open at Task 9 only.** Tasks 1 to 8 and the delayed-history fixes are done (`v0.2.166`–`v0.2.192`): device-resident scales, cached fp8 weights, delayed scaling, quantise from 16-bit storage, kernel tuning (2048^3 at 0.613 ms, 28000 GF/s), fp8 on f16/bf16 storage, e5m2 gradients, `-fp8`. Default 3000-iteration fp8 run: 47.2 ms / 142.5 s against fp16 67.5 ms / 208.8 s, loss 0.0578, mAP 99.80% (`docs/OXIDE_BENCH.md` §10). Task 9 still needs the delayed-scaling run, the default decision and the recorded verdict; no verdict exists yet. CUDA Graphs (overhead step 6) stays open and conditional on measured launch overhead.
- **Plan checkboxes are stale.** `plan_fp8.md` Tasks 1 to 4 and 6 to 8, and `plan_fp16.md` steps, still show `[ ]` although the work shipped. Progress tables and `implemented.md` §§238–256 are authoritative.
- **xtask flags** (verified in `xtask/src/main.rs`): `-fp8` sets `LIGHTNET_OXIDE_GEMM=fp8` and `LIGHTNET_OXIDE_GEMM_BWD=fp8` (f32 storage, inherited `LIGHTNET_OXIDE_AMP` unset); `-fp16` sets `LIGHTNET_OXIDE_AMP=1` and adds `-dyn_scale -loss_scale 1024` unless supplied; `-bf16` sets `LIGHTNET_OXIDE_AMP=bf16` with no scaler flags; `-fp8 -fp16` and `-fp8 -bf16` combine fp8 convolution GEMMs with 16-bit storage.
- **Version numbers.** Latest GPU increment is `v0.2.202`; the next free number is `v0.2.203`. `AGENTS.md` plans `v0.2.203`–`v0.2.223` for the layer list, so the items below start at `v0.2.224`. Planned numbers are provisional: take the next free number in [version-map.tsv](../../02_state/version-map.tsv) at commit time.
- **Baseline.** The default `cargo xtask test` compiles again (commit `37ec4b5`) and the default `cargo xtask clippy` completes with warnings only. Four `cfg_corpus` cases are unresolved; do not claim a green host suite.

## Plans

| Phase | Plan | Depends on |
|---|---|---|
| 1 fp16 | [plan_fp16.md](plan_fp16.md) | none |
| 2 bf16 | [plan_bf16.md](plan_bf16.md) | Phase 1 Tasks 1 to 4 (the scaler and xtask return shape) |
| 3 fp8 | [plan_fp8.md](plan_fp8.md) | Phase 2 Tasks 1, 2 and 6 for fp8 Tasks 4 and 6; fp8 Tasks 1, 2, 3, 5 only need the current code |

Related reading: [fp8_fp4.md](fp8_fp4.md) (earlier fp8 design and the Phase D numbers), `docs/OXIDE_BENCH.md` section 7.2 (AMP measurements) and section 10 (fp8 and bf16 against fp32).

## Running this file as a goal

Use it as `/goal tasks/rust/03_plans/lowprec/ROADMAP_FP8.md`.

### Loop

1. Read `lessons.md`, `git status`, `git log -5`, this file's [State snapshot](#state-snapshot) and the plan of the phase you work on.
2. Take the first unchecked, unblocked item in [Roadmap Todo](#roadmap-todo). An item marked user-gated or blocked is reported, not worked around.
3. Run one command at a time through `cargo xtask` (rules in [Rules](#rules)); the delegation protocol below applies to plan tasks.
4. On green, commit and tag per [01_rules/versioning.md](../../01_rules/versioning.md) and update the docs in the same pass.
5. Tick the item below, then go back to step 2.

### Definition of done for one item

1. The item's "Done when" line is true, with a named test or recorded measurement. A timeout, a filtered-out test or an unrun test is not a pass; say which ran and which did not.
2. `cargo xtask check` and the touched crate's `cargo xtask clippy` pass.
3. The code commit is `v0.2.NNN: <what>` with the tests and results in the body, followed by a local lightweight tag `v0.2.NNN`. Commit only the files the item names. A measurement-only or doc-only item is `docs(rust): <what>` with no tag.
4. The docs listed in [Documentation updates](#documentation-updates) are updated in the same pass.
5. The item is ticked here and in the plan task it came from.
6. Results are compared against the Rust fp32 run; no claim of C++ FP8 parity.

### Stop conditions

Stop and report, without continuing, when:

- every phase meets [Done when](#done-when);
- the next item is user-gated (see [User-gated actions](#user-gated-actions));
- a covered test fails, times out or cannot run, and the cause is not clear after one root-cause pass;
- a measurement contradicts the plan (a kernel change does not speed up, a pass criterion fails): record the result and re-plan, do not keep tuning without a profile.

## Roadmap Todo

Ordered checklist. Each item starts with the version it will ship as. `--` means no release (a check, a measurement or a docs-only record). Planned numbers are provisional; see [State snapshot](#state-snapshot).

### Phase 0: Baseline

- [ ] -- Read `lessons.md`, `git status`, `git log -5`; confirm the `-fp8`, `-fp16`, `-bf16` mapping in `xtask/src/main.rs` still matches the `CLAUDE.md` Commands table (`cargo xtask help`).
- [ ] -- `JOBS=4 MEM=4G cargo xtask check`, then `PKGS=xtask cargo xtask test`. Done when both pass.
- [ ] -- Tick the stale checkboxes in `plan_fp16.md`, `plan_bf16.md` and `plan_fp8.md` (Tasks 1 to 8) from the version rows `v0.2.161`–`v0.2.192` in `version-map.tsv`; leave Task 9 open. Docs-only commit `docs(rust): tick shipped low-precision plan steps`.

### Phase 1: fp16

Closed (`v0.2.161`–`v0.2.165`, `docs/OXIDE_BENCH.md` §7.2.1). Nothing open; a regression here blocks every later item.

- [x] v0.2.165 fp16 hardening complete: scaler, `-loss_scale`, overflow coverage, convergence record.

### Phase 2: bf16

Closed (`v0.2.170`–`v0.2.176`, `docs/OXIDE_BENCH.md` §10). One defect is open.

- [x] v0.2.176 bf16 mixed-precision mode and `-bf16`, convergence record.
- [ ] v0.2.224 Root-cause the flaky `oxide_half` test `gemm_bf16_parity_all_transposes_alignments_alpha_beta` (failed once on `OxideBackend::new()`, passed 10 later runs). Run it in a loop, one command at a time (`cargo xtask oxide --test oxide_half -- gemm_bf16_parity`), capture the failing output. Done when the cause is named and either fixed (new version) or documented as environmental with the evidence. If no failure reproduces, record the loop count and stay `--`.

### Phase 3: fp8

Tasks 1 to 8 shipped (`v0.2.166`–`v0.2.192`). Task 9 is the only open plan task.

- [x] v0.2.192 fp8 Tasks 1 to 8 and delayed-history fixes.
- [ ] -- User-gated. `plan_fp8.md` Task 9 steps 1 and 2: run `cargo xtask train_legogears -fp8 -map_every 100` and once with `LIGHTNET_OXIDE_FP8_SCALING=delayed`, same seed and iteration count as the fp32 and fp16 runs. Ask the user for the iteration count first. Gate: batch-16 step time below fp16, loss within 10% of fp32, mAP within 2 points of fp16.
- [ ] v0.2.225 `plan_fp8.md` Task 9 step 3: decide the default scaling mode from the two runs. If the default changes, this is a code commit with a tag; if it stays, record the verdict as `docs(rust)` with no tag. Record the table in `docs/OXIDE_BENCH.md`, update `implemented.md`, `STATUS.md`, `todo.md`, `version-map.tsv`, delete weights and logs. Needs the user's approval before the default changes.
- [ ] -- `plan_fp8.md` Task 9 step 4 and the verdict in `fp8_fp4.md`: `docs(rust): record fp8 accuracy and speed gate`. A "numerics only, not faster" verdict is valid and ends the phase.

### Phase 4: Follow-ups

Only after Phase 3 is closed.

- [ ] v0.2.226 CUDA Graphs (overhead step 6), conditional: first profile launch overhead (`nsys` or the bench tests) and record the share of step time. Start only if launch overhead is a named bottleneck; cuda-core has only raw begin/end capture, so state the wrapper design before coding. If the share is small, record that and close as `--`.
- [ ] -- Reconcile the `AGENTS.md` items `v0.2.221` (GPU Muon residency) and `v0.2.223` (native half-storage transformer arithmetic) with this line: they use the same `HalfFormat` machinery. Write the shared constraint into both files; no code.

### Blocked or out of scope

- Native NVFP4 `mma` needs sm_120a/121a; the sm_89 path is the `__dp4a` emulation (`fp8_fp4.md`). Hardware-blocked.
- wgmma (sm_90) and tcgen05 (sm_100) GEMM kernels. Hardware-blocked.
- fp8 conv via im2col against the C++ FP8 path (`v6-dev-fp8`): no C++ build here, so no golden. Not runnable.
- Extending fp8 beyond convolution GEMMs (connected, transformer GEMMs): not planned; ask the user before adding scope.

## Rules

- Use `cargo xtask` for every check, build, test and training command (`check`, `build`, `clippy`, `test`, `oxide`, `train_legogears`); do not type raw `cargo` or `systemd-run`. Do not run `cargo xtask fmt`: it reformats unrelated files.
- One build or test command at a time. Build threads are capped at 4; `train_legogears` defaults to 4 and `JOBS` is clamped to 4. Memory is capped at `MemoryHigh=3G`, applied by `cargo xtask` (it warns and runs uncapped when there is no user systemd bus).
- `src/lightnet-cuda-oxide` is not a workspace member: run its tests with `cargo xtask oxide --test <file> [filter]`. The harness already adds `--nocapture`; do not pass it again. Revert `src/lightnet-cuda-oxide/Cargo.lock` after oxide runs.
- Commit only the files a task names; never `git add -A`. `src/lightnet-core/src/runtime/mod.rs` can carry unrelated edits from the user: stage only your hunk with `git add -p`.
- Commit and tag on green, as in [01_rules/versioning.md](../../01_rules/versioning.md) ("Commit and tag on green"): every plan task whose tests pass is committed and tagged at once, by whoever implemented it (Claude or Codex), without asking. Each such task takes the next free `v0.2.NNN` number from [version-map.tsv](../../02_state/version-map.tsv); the commit message is `v0.2.NNN: <what>` where `<what>` is the plan step's suggested message, the commit body lists the tests run and their results, and the commit is tagged `v0.2.NNN` (local, lightweight). A task whose tests fail, time out or were not run gets neither a commit nor a tag. Measurement-only or doc-only steps (convergence records) are `docs(rust): <what>` with no tag. Never push.
- Never edit `third_party/` or `src/cutile-rs-aot` (pinned submodules).
- Delete weights, backups, logs and other big files a task produces once the evidence is recorded.
- Do not archive or remove completed plans in `tasks/rust/`; they are the roadmap.
- Keep existing comments in files you edit.
- This goal explicitly overrides the "no subagents" operator override in [AGENTS.md](../../AGENTS.md) for plan-task work: delegation is expected here. Edits outside plan tasks (docs, ticking boxes) are made directly.

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
- Adding scope beyond the three phases (fp8 outside convolution GEMMs, new precisions).

## Reporting

At the end of each task, one line: task, tests run and their result, files changed, commit hash and tag (`v0.2.NNN`), or why there is no commit (failing, timed-out or unrun tests). At the end of each phase, a short summary with the measurements (step time, loss and mAP against the baseline) and the verdict against the [Done when](#done-when) criteria. Report failures with the command output; never describe a failing run as passing.

## Documentation updates

For every version-mapped increment (next free `v0.2.NNN` in [version-map.tsv](../../02_state/version-map.tsv); currently `v0.2.203`; never reuse a number): add the evidence line to [implemented.md](../../02_state/implemented.md), an entry in [STATUS.md](../../02_state/STATUS.md) "Latest increments", tick [todo.md](../../todo.md) and this roadmap, update the layer or mode row in [layer-support.md](../../02_state/layer-support.md) if one exists, and add the row to `version-map.tsv`. Record benchmark and convergence results in `docs/OXIDE_BENCH.md` (the AMP section 7.2 for fp16 and bf16; section 10 for fp8). Keep the `CLAUDE.md` Commands table and `xtask help` in sync with every flag change. Every `.md` file keeps its table of contents in sync with its headings.

## Keeping this file current

- Tick an item only after its definition of done holds. Add a new item at the end of its phase with the next planned version; a ticked item keeps the version it shipped as.
- Update the [State snapshot](#state-snapshot) whenever a version lands or a measurement is recorded.
- Finished work goes in `02_state/implemented.md`, not here. Open work stays here and in `todo.md`.
- After a user correction, add the rule to `lessons.md`.

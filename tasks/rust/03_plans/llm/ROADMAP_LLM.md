# LLM Goal: Latin decoder, encoder-decoder and Cicero chat

Goal document for an agent (the "lead") that delegates the implementation of four plans. Read this file first, then only the plan of the phase you are working on. It extends [AGENTS.md](../../AGENTS.md) for this one goal, in the same way as [AGENTS_fp8.md](../lowprec/AGENTS_fp8.md).

## Table of Contents

- [Start here](#start-here)
- [Goal](#goal)
- [Done when](#done-when)
- [Plans](#plans)
- [Current state](#current-state)
- [Rules](#rules)
- [Known plan conflicts](#known-plan-conflicts)
- [Delegation protocol](#delegation-protocol)
- [User-gated actions](#user-gated-actions)
- [Reporting](#reporting)
- [Documentation updates](#documentation-updates)

## Start here

Begin at [plan_llm_decoder.md](plan_llm_decoder.md) Task 1. The crate `src/lightnet-lm` already exists as a workspace member (scaffold: `Cargo.toml`, `CONTEXT.md`, a docs-only `src/lib.rs`), so Task 1 fills it in; it does not create it. Read `src/lightnet-lm/CONTEXT.md` and the CONTEXT file of every other crate you touch before editing.

All four plans were drafted by `gpt-6.1-sol` and are **unreviewed**. Treat every file path, API name, test count and number in them as a proposal until you have read the cited code. Earlier exploration summaries in this repo contained wrong claims, so the lead verifies each fact a task depends on by reading the cited lines.

## Goal

Train and run a small Latin language model in Rust on this repo, then talk to it as a disclosed simulation of Cicero:

```
cargo xtask test lm_            # CPU reference, tokenizer, loader, trainer
cargo xtask oxide --test oxide_lm_train   # resident bf16 GPU training
```

The phases run in this order, each one finished and recorded before the next starts:

1. **Decoder foundation** (CPU first): contracts, byte-fallback BPE, differentiable ops, trainable decoder with tied head, token shards, trainer with checkpoint resume, KV-cache and sampling.
2. **GPU decoder** (oxide, bf16): resident forward, backward, optimizer, then causal/GQA flash with chunked cross-entropy, optional fp8, optional cuTile inference, and the sparse-attention family (sliding window, DSA-style, NSA-style).
3. **Latin pilot and recipe**: dataset inventory and whole-work splits, pretrain, mid-train, chat post-train, evaluation, chat CLI.
4. **Encoder and DeepSeek-style variants** (optional experiments): encoder stacks, cross-attention, the causal encoder-decoder split, MLA / low-rank MQA, MoE, MTP.

## Done when

- Phase 1 ([plan_llm_decoder.md](plan_llm_decoder.md) Tasks 1 to 7): all tasks done; cached logits equal full-prefix logits at nonzero positions; every parameter family passes its finite-difference check; the descent-sign cross-entropy assertions pass.
- Phase 2 (Tasks 8 to 16 of the same plan): all tasks done or explicitly marked optional-and-skipped with a reason; oxide numerics agree with the CPU reference within the plan's stated tolerances; no speed claim without a separately authorized, matched timing.
- Phase 3 ([plan_llm_training_recipe.md](plan_llm_training_recipe.md) Tasks R1 to R15): R1 to R8 and R12 done as code and tests; each user-gated run (R9 to R11) is either recorded with its evidence or still waiting on the user, stated plainly. Persona rules in R5 hold: the model is disclosed as a simulation and never presents invented quotations as real.
- Phase 4 ([plan_llm_encdec.md](plan_llm_encdec.md) Tasks 17 to 40): only tasks the user selects. Each one that lands passes its equivalence-to-dense or cache-versus-full-prefix test.
- Every task's tests pass, `cargo xtask clippy` is clean for touched crates, docs are updated as in [Documentation updates](#documentation-updates), and the `CLAUDE.md` Commands table matches any new xtask subcommand.
- No quality, speed or parity claim without a recorded measurement. Results are compared against the repo's own CPU reference; there is no external golden.

## Plans

| Phase | Plan | Tasks | Depends on |
|---|---|---|---|
| 1 and 2 | [plan_llm_decoder.md](plan_llm_decoder.md) | 1 to 16 | none (1 to 7 CPU, 8 to 9 oxide, 10 pilot, 11 to 13 optimisation, 14 to 16 sparse attention) |
| 3 | [plan_llm_training_recipe.md](plan_llm_training_recipe.md) | R1 to R15 | decoder Tasks 1 to 7 for R1 to R8 and R12; Tasks 8 to 9 for R13 |
| 3 (data) | [latin_datasets.md](latin_datasets.md) | source list, not tasks | feeds R1 and decoder Task 10 |
| 4 | [plan_llm_encdec.md](plan_llm_encdec.md) | 17 to 40 | decoder Tasks 1 to 9; sparse tasks 29 and 30 need decoder Tasks 14 and 15 |

Related reading: [layers.md](../../04_reference/layers.md) and [layer-support.md](../../02_state/layer-support.md) for the detector layer ports the decoder must not disturb, [plan_bf16.md](../lowprec/plan_bf16.md) and [plan_fp8.md](../lowprec/plan_fp8.md) for the reusable low-precision machinery, `docs/OXIDE_BENCH.md` for measurement style.

## Current state

Verified at the time of writing (re-check before relying on it):

- `src/lightnet-lm` CPU Tasks 1 to 7 are implemented and versioned as v0.1.042: strict `[decoder]` config/tokens, byte-fallback BPE, exact CPU ops and causal GQA decoder, token shards, AdamW/checkpoints, KV-cache and sampling, and the `lightnet-rs lm` CLI. `lm_` passed 52 of 52 in `lightnet-lm`, `lm_cli_` 4 of 4; no corpus training or evaluation was measured. Tasks 8 to 16 remain unimplemented, so later plans still depend on open GPU work.
- `[transformer]` in cfg is the Swin-style windowed attention in `lightnet-core`; the LAYA encoder is frozen and bidirectional. Neither is a text decoder and neither is to be reinterpreted. The decoder lives in `lightnet-lm` behind its own `[decoder]` schema.
- Decoder CPU primitives now exist for RMSNorm, causal masking, GQA, sparse next-token CE, the trainable embedding scatter-add gradient, tied head, gated-MLP backward, global-norm clipping, token LR schedules, the token loader, BPE, KV-cache inference and sampling. The resident 16-bit/fp8 transformer path and the decoupled oxide/cuTile decoder remain missing.
- The existing flash attention is non-causal and clamps Q/K to +-128 and scores to +-10. A decoder needs a separate unclamped kernel family.
- fp16 and bf16 exist for GEMM and conv only; there is no resident 16-bit or fp8 transformer training.
- cuTile has forward-only vendored decoder kernels that are not integrated; the adapter may not build against the current enum.
- Losses seed the backward pass as target minus prediction (descent direction). Any new cross-entropy keeps that sign, and no compensating negation is added at the optimizer boundary.
- `cfg/unsupported` is gone and the `darknet-*` crates are renamed `lightnet-*` (`LIGHTNET_*` env vars); use the new names everywhere.
- DeepSeek V4.1 Flash is reported (several secondary sources, fetched as summaries) to be one causal LM split into a 20-layer encoder and a 20-layer decoder, not a T5-style model. Treat the specs in `plan_llm_encdec.md` as UNVERIFIED unless the plan marks them VERIFIED with a URL.

## Rules

- Use `cargo xtask` for every check, build, test and training command; do not type raw `cargo` or `systemd-run`. New subcommands are added as arms in `xtask/src/main.rs` and to the `CLAUDE.md` Commands table.
- One build or test command at a time. Build threads are capped at 4. Memory is capped at `MemoryHigh=3G`, applied by `cargo xtask` (it warns and runs uncapped when there is no user systemd bus; record that instead of claiming a capped run).
- `src/lightnet-cuda-oxide` and `src/lightnet-cuda-cutile` are not workspace members. Run oxide tests with `cargo xtask oxide --test <file> [-- <filter>]`. The oxide bridge needs `lightnet-lm` as a normal dependency (a manifest change in the plan); `lightnet-lm` itself depends on neither oxide nor `lightnet-core`.
- A zero-test filter, a timeout, a skipped GPU test or an unavailable device is not a pass.
- Each task follows red test, observed failure, implementation, observed green checks, local commit and tag, as the plan states.
- Commit only the files a task names; never `git add -A`. `tasks/` is git-excluded, so task documents need `git add -f` and a named path.
- Commit and tag on green, as in [AGENTS.md](../../AGENTS.md#versioning): every task whose named tests pass is committed and tagged at once by whoever implemented it, without asking. Message `v0.M.NNN: <what>`. Take the next free number from `version-map.tsv` for the line you are in (CPU increments use the `v0.1` line, GPU increments the `v0.2` line, as `plan_llm_decoder.md` states); never reuse, move or delete a tag. Then append the `version-map.tsv` row in a separate docs-only commit. Do not allocate numbers in advance.
- Use the version-line numbering you find in `version-map.tsv` at the time; the plans contain `v0.M.NNN` placeholders on purpose.
- Corpus training, tiny-corpus overfit runs, pilot evaluation runs and long benchmarks are user-gated (see below). Bounded synthetic tests are fine.
- Corpus licences are mostly unconfirmed (see `latin_datasets.md`). Do not download, vendor or commit corpus text; do not train on a source marked UNVERIFIED or noncommercial without the user's say-so. Record provenance for every text used.
- Never edit `third_party/` or `src/cutile-rs-aot` (pinned submodules).
- Delete weights, shards, logs and other big files a task produces once the evidence is recorded; keep referenced fixtures.
- Do not archive or remove completed plans in `tasks/rust/`; they are the roadmap.
- Keep existing comments in files you edit.
- Each edited Markdown file needs an updated TOC.
- Exact resume (RNG, data cursor, schedule, gradients, optimizer state) is required within one deterministic backend and configuration. CPU to GPU, or different GPU reduction orders, use tolerances, never bitwise claims.

## Known plan conflicts

Resolve these before or at the task that hits them; record the decision in that task's commit message.

- `plan_llm_decoder.md` calls `lightnet-lm` a new crate and lists `lightnet-core` cfg and error modules as reusable, while also saying `lightnet-lm` must not depend on `lightnet-core`. Settle the crate boundary (own error type and small cfg parser, or a shared module) in Task 1.
- Decoder Task 2 fixes token IDs 0 to 3 and bytes at 4 to 259, merges from 260. The recipe plan (R2) wants chat control tokens in a versioned reserved range at the top of the vocabulary and a bumped tokenizer artifact version. Make Task 2 leave room for this.
- Decoder Task 6's `TokenSchedule` is a single warmup plus cosine. R3 needs WSD, re-warm and continuation; extend it there.
- The 3e-4 / 3e-5 learning-rate defaults in decoder Task 10 are unmeasured; treat them as sweep candidates.
- Latin u/v and i/j folding is lossy; default it to off and ablate it.
- Decoder Tasks 8 and 9 have no per-token eval readback or device KV-cache for chat; recipe tasks R4 and R13 add them.
- The `v0.M.NNN` placeholders, and the CPU/GPU version-line choice, are policy questions the user may want to confirm once before Task 1 lands.
- `AGENTS.md` names `gpt-6-luna` for Codex; the plans were drafted with `gpt-6.1-sol`. Use the model the user names for the session.

## Delegation protocol

The lead reads this file and the current plan, then runs the plan with `superpowers:subagent-driven-development`.

1. One subagent per plan task (one tack each), in plan order. Give the subagent the full task text (the Interfaces block is how it learns names and types), the plan's Global Constraints and Review Focus, and the file paths it will touch. Do not ask it to read other plans.
2. Implementation subagents run on Sonnet. Read-only exploration runs on Sonnet or Haiku. The lead re-reads the cited lines for every fact a task depends on.
3. After each task a fresh reviewer subagent checks the diff against the task text and the tests; the lead merges nothing with failing tests or unexplained changes outside the named files.
4. Phase boundaries: after the last task of a phase, run the whole touched test files (`cargo xtask test lm_`, the touched `cargo xtask oxide --test oxide_lm_*` suites, and `cargo xtask check lightnet-lm`), then a whole-phase review on the most capable available model before moving on.
5. If a measurement or a verified fact contradicts the plan (a kernel does not match the CPU reference, a pass criterion fails, a cited path is wrong), stop that thread, record the result in the commit message and the docs, and re-plan before continuing.
6. Subagents never start training runs, push, download corpora or touch files outside their task.
7. Codex (`codex exec -m <model> -s read-only`) is allowed for read-only review and drafting only. It never edits the workspace; the lead or an implementation subagent applies changes.

## User-gated actions

Ask the user before each of these; do not infer approval from an earlier one:

- Starting any training run (R9, R10, R11), a tiny-corpus overfit demonstration, a pilot evaluation run, or a long benchmark. The user chooses the iteration or token count.
- Downloading or storing any corpus, and using any source whose licence is unconfirmed or noncommercial.
- Pushing, opening a pull request, or rewriting history.
- Changing the default of a user-visible mode.
- Bumping a submodule pin.
- Publishing or distributing a trained checkpoint or a Cicero chat demo.
- Enabling any Phase 4 experiment (MoE, MTP, MLA, causal split, sparse attention beyond sliding window).

## Reporting

At the end of each task, one line: task, tests run and their result, files changed, commit hash and tag, or why there is no commit (failing, timed-out or unrun tests). At the end of each phase, a short summary with the measurements (loss, perplexity, workspace bytes, step time where measured) against the plan's gate, and the verdict against the "Done when" criteria. Report failures with the command output; never describe a failing, skipped or unrun test as passing. Mark anything not measured as unverified.

## Documentation updates

For every version-mapped increment: add the evidence line to `implemented.md`, an entry in `STATUS.md` "Latest increments", tick `todo.md`, and add the row to `version-map.tsv` in the separate docs-only commit. Record data provenance in `llm_data.md`, evaluation results in `llm_eval.md` (both created by the decoder plan) and training or benchmark numbers in `docs/OXIDE_BENCH.md`. Keep the `CLAUDE.md` Commands table in step with any new xtask subcommand, and update this file's [Current state](#current-state) when a phase finishes. After a user correction, add the lesson to `tasks/<branch>/lessons.md`.

# LLM Roadmap: Latin decoder, encoder-decoder and Cicero chat

Goal document and ordered work list for an agent (the "lead") that delegates the implementation of five plans. Read this file first, then only the plan of the phase you are working on. It extends [AGENTS.md](../../AGENTS.md) for this one goal, in the same way as [ROADMAP_FP8.md](../lowprec/ROADMAP_FP8.md); the three documents share one `v0.M.NNN` number line.

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
  - [Phase 1: Decoder foundation (CPU)](#phase-1-decoder-foundation-cpu)
  - [Phase 2: GPU decoder](#phase-2-gpu-decoder)
  - [Phase 3: Latin pilot and recipe](#phase-3-latin-pilot-and-recipe)
  - [Phase 4: Encoder, DeepSeek-style and recursive variants](#phase-4-encoder-deepseek-style-and-recursive-variants)
  - [Blocked or out of scope](#blocked-or-out-of-scope)
- [Rules](#rules)
- [Known plan conflicts](#known-plan-conflicts)
- [Delegation protocol](#delegation-protocol)
- [User-gated actions](#user-gated-actions)
- [Reporting](#reporting)
- [Documentation updates](#documentation-updates)
- [Keeping this file current](#keeping-this-file-current)

## Goal

Train and run a small Latin language model in Rust on this repo, then talk to it as a disclosed simulation of Cicero:

```
cargo xtask test lm_            # CPU reference, tokenizer, loader, trainer
cargo xtask oxide --test oxide_lm_train   # resident bf16 GPU training (planned, not yet present)
```

The phases run in this order, each one finished and recorded before the next starts:

1. **Phase 1: Decoder foundation** (CPU first): contracts, byte-fallback BPE, differentiable ops, trainable decoder with tied head, token shards, trainer with checkpoint resume, KV-cache and sampling.
2. **Phase 2: GPU decoder** (oxide, bf16): resident forward, backward, optimizer, then causal/GQA flash with chunked cross-entropy, optional fp8, optional cuTile inference, and the sparse-attention family (sliding window, DSA-style, NSA-style).
3. **Phase 3: Latin pilot and recipe**: dataset inventory and whole-work splits, pretrain, mid-train, chat post-train, evaluation, chat CLI.
4. **Phase 4: Encoder, DeepSeek-style and recursive variants** (optional experiments): encoder stacks, cross-attention, the causal encoder-decoder split, MLA / low-rank MQA, MoE, MTP, and the weight-shared recursive decoder.

## Done when

- Phase 1 ([plan_llm_decoder.md](plan_llm_decoder.md) Tasks 1 to 7): all tasks done; cached logits equal full-prefix logits at nonzero positions; every parameter family passes its finite-difference check; the descent-sign cross-entropy assertions pass.
- Phase 2 (Tasks 8 to 16 of the same plan): all tasks done or explicitly marked optional-and-skipped with a reason; oxide numerics agree with the CPU reference within the plan's stated tolerances; no speed claim without a separately authorized, matched timing.
- Phase 3 ([plan_llm_training_recipe.md](plan_llm_training_recipe.md) Tasks R1 to R15): R1 to R8 and R12 done as code and tests; each user-gated run (R9 to R11) is either recorded with its evidence or still waiting on the user, stated plainly. Persona rules in R5 hold: the model is disclosed as a simulation and never presents invented quotations as real.
- Phase 4 ([plan_llm_encdec.md](plan_llm_encdec.md) Tasks 17 to 40 and [plan_llm_recursive.md](plan_llm_recursive.md) Tasks L1 to L22): only tasks the user selects. Each one that lands passes its equivalence-to-dense or cache-versus-full-prefix test.
- Every task's tests pass, `cargo xtask clippy` is clean for touched crates, docs are updated as in [Documentation updates](#documentation-updates), and the `CLAUDE.md` Commands table matches any new xtask subcommand.
- No quality, speed or parity claim without a recorded measurement. Results are compared against the repo's own CPU reference; there is no external golden.

## State snapshot

State on 2026-10-27, from a static read of the code, plans and `version-map.tsv` (no cargo run). Treat plan facts as proposals until you read the cited code.

- **Phase 1 done as `v0.1.042`** (`2794f2c`, decoder Tasks 1 to 7). `src/lightnet-lm` has 13 source files (about 5.6k lines: `config`, `tokenizer`, `ops`, `attention`, `decoder`, `data`, `optim`, `checkpoint`, `train`, `cache`, `sampling`, `types`, `lib`) and 7 test files (`lm_contract`, `lm_data`, `lm_decoder`, `lm_generate`, `lm_ops`, `lm_tokenizer`, `lm_train`). At the last record `lm_` passed 52 of 52 and `lm_cli_` 4 of 4. The `lightnet-rs lm` CLI arm exists in `lightnet-cli`. No corpus training or evaluation was measured.
- **Phase 2 not started.** There is no `oxide_lm_*` test file and no resident 16-bit decoder. The existing flash attention is non-causal and clamps Q/K to +-128 and scores to +-10, so a decoder needs a separate unclamped kernel family. cuTile has forward-only vendored decoder kernels that are not integrated and may not build against the current enum.
- **Phase 3 is partly prepared outside the plan, uncommitted.** Untracked user work: `datasets/latin/` (about 2.4 GB: `raw/`, `prepared/`, `sources.tsv`, `SHA256SUMS`, `README.md`), `tools/prepare_latin.py`, `tools/test_prepare_latin.py`, `tools/latin_pretraining.md`, `docs/LLM_PLAN.md` and `src/lightnet-examples/examples/llm_prepare_corpus.rs` (feature `lightnet-examples/llm`: fits a train-only BPE sample and writes token shards). They overlap recipe tasks R1 and R7 and decoder Task 10. `docs/LLM_PLAN.md` is a second plan for the same line; reconcile it with `plan_llm_training_recipe.md` so each fact has one home. Do not commit `datasets/`.
- **Plans are five, all unreviewed.** `plan_llm_decoder.md` (Tasks 1 to 16), `plan_llm_training_recipe.md` (R1 to R15), `plan_llm_encdec.md` (Tasks 17 to 40), `plan_llm_recursive.md` (L1 to L22, not in the earlier four-plan list) and the data notes `llm_data.md`, `latin_datasets.md`. `llm_eval.md` does not exist yet (R4 creates it).
- `[transformer]` in cfg is the Swin-style windowed attention in `lightnet-core`; the LAYA encoder is frozen and bidirectional. Neither is a text decoder and neither is to be reinterpreted. The decoder lives in `lightnet-lm` behind its own `[decoder]` schema; `lightnet-lm` depends on neither oxide nor `lightnet-core`.
- Losses seed the backward pass as target minus prediction (descent direction). Any new cross-entropy keeps that sign, and no compensating negation is added at the optimizer boundary.
- `cfg/unsupported` is gone and the `darknet-*` crates are renamed `lightnet-*` (`LIGHTNET_*` env vars); use the new names everywhere.
- DeepSeek V4.1 Flash is reported (secondary sources, fetched as summaries) to be one causal LM split into a 20-layer encoder and a 20-layer decoder, not a T5-style model. Treat the specs in `plan_llm_encdec.md` as UNVERIFIED unless the plan marks them VERIFIED with a URL.
- Low-precision machinery to reuse: bf16 AMP and fp8 convolution GEMMs exist for convolution only ([ROADMAP_FP8.md](../lowprec/ROADMAP_FP8.md)); there is no resident 16-bit or fp8 transformer training.
- **Version numbers.** Latest CPU increment `v0.1.044`, latest GPU increment `v0.2.202`. `AGENTS.md` plans `v0.1.045`–`v0.1.066` and `v0.2.203`–`v0.2.223`, `ROADMAP_FP8.md` plans `v0.2.224`–`v0.2.226`, so the items below start at `v0.1.067` and `v0.2.227`. Planned numbers are provisional: take the next free number in [version-map.tsv](../../02_state/version-map.tsv) at commit time.

## Plans

| Phase | Plan | Tasks | Depends on |
|---|---|---|---|
| 1 and 2 | [plan_llm_decoder.md](plan_llm_decoder.md) | 1 to 16 | none (1 to 7 CPU, 8 to 9 oxide, 10 pilot, 11 to 13 optimisation, 14 to 16 sparse attention) |
| 3 | [plan_llm_training_recipe.md](plan_llm_training_recipe.md) | R1 to R15 | decoder Tasks 1 to 7 for R1 to R8 and R12; Tasks 8 to 9 for R13 |
| 3 (data) | [latin_datasets.md](latin_datasets.md), [llm_data.md](llm_data.md) | source list and tokenizer contract, not tasks | feed R1 and decoder Task 10 |
| 4 | [plan_llm_encdec.md](plan_llm_encdec.md) | 17 to 40 | decoder Tasks 1 to 9; sparse tasks 29 and 30 need decoder Tasks 14 and 15 |
| 4 | [plan_llm_recursive.md](plan_llm_recursive.md) | L1 to L22 | decoder Tasks 1 to 9 |

Related reading: [layers.md](../../04_reference/layers.md) and [layer-support.md](../../02_state/layer-support.md) for the detector layer ports the decoder must not disturb, [plan_bf16.md](../lowprec/plan_bf16.md) and [plan_fp8.md](../lowprec/plan_fp8.md) for the reusable low-precision machinery, `docs/OXIDE_BENCH.md` for measurement style, `tools/latin_pretraining.md` for the data-preparation commands.

## Running this file as a goal

Use it as `/goal tasks/rust/03_plans/llm/ROADMAP_LLM.md`.

### Loop

1. Read `lessons.md`, `git status`, `git log -5`, this file's [State snapshot](#state-snapshot), `src/lightnet-lm/CONTEXT.md`, and the plan of the phase you work on.
2. Take the first unchecked, unblocked item in [Roadmap Todo](#roadmap-todo). An item marked user-gated or optional is reported or skipped, not worked around.
3. Verify every fact the task depends on by reading the cited lines: the plans are unreviewed and earlier exploration summaries contained wrong claims.
4. Follow red test, observed failure, implementation, observed green checks, then commit and tag per [01_rules/versioning.md](../../01_rules/versioning.md) and update the docs in the same pass.
5. Tick the item below, then go back to step 2.

### Definition of done for one item

1. The item's "Done when" line is true, with a named test that ran and passed. A zero-test filter, a timeout, a skipped GPU test or an unavailable device is not a pass; say which ran and which did not.
2. `cargo xtask check lightnet-lm` and `cargo xtask clippy` for the touched crate pass.
3. The code commit is `v0.M.NNN: <what>` (CPU on the `v0.1` line, GPU on the `v0.2` line) with the tests and results in the body, followed by a local lightweight tag. Commit only the files the item names. Docs-only or measurement-only items are `docs(rust): <what>` with no tag.
4. The docs in [Documentation updates](#documentation-updates) are updated in the same pass.
5. The item is ticked here and in the plan task it came from.
6. No quality, speed or parity claim without a recorded measurement.

### Stop conditions

Stop and report, without continuing, when:

- every phase meets [Done when](#done-when), or the next item is user-gated or an unselected Phase 4 experiment;
- a covered test fails, times out or cannot run, and the cause is not clear after one root-cause pass;
- a measurement or a verified fact contradicts the plan (a kernel does not match the CPU reference, a pass criterion fails, a cited path is wrong): record it and re-plan;
- an action touches the corpus, a training run, a licence question or anything under [User-gated actions](#user-gated-actions).

## Roadmap Todo

Ordered checklist. Each item starts with the version it will ship as. `--` means no release (a check, a user-gated run or a docs-only record). Items for unselected Phase 4 experiments are shown with their provisional versions but stay blocked until the user selects them. Planned numbers are provisional; see [State snapshot](#state-snapshot).

### Phase 0: Baseline

- [ ] -- Read `lessons.md`, `git status`, `git log -5`; re-read `src/lightnet-lm/CONTEXT.md`.
- [ ] -- `JOBS=4 MEM=4G cargo xtask check lightnet-lm`, then `cargo xtask test lm_`. Done when both pass and the counts match the snapshot (52 of 52, `lm_cli_` 4 of 4) or the drift is explained.
- [ ] -- Reconcile `docs/LLM_PLAN.md` and `tools/latin_pretraining.md` with `plan_llm_training_recipe.md` R1/R7 and `llm_data.md`: one home per fact, links instead of copies. Docs-only commit `docs(rust): reconcile the Latin data-preparation notes`. The untracked data files stay untracked unless the user asks otherwise.
- [ ] -- Fix the stale statements in this plan family found by the audit: `tasks/` is no longer git-excluded (drop the `git add -f` advice in the plans), and `llm_eval.md` does not exist yet.

### Phase 1: Decoder foundation (CPU)

Closed as `v0.1.042`.

- [x] v0.1.042 Decoder Tasks 1 to 7: contracts, byte-fallback BPE, CPU ops and causal GQA decoder, token shards, AdamW and checkpoints, KV-cache and sampling, `lightnet-rs lm` CLI.

### Phase 2: GPU decoder

Plan: `plan_llm_decoder.md` Tasks 8 to 16. The oxide bridge needs `lightnet-lm` as a normal dependency of `lightnet-cuda-oxide` (a manifest change in the plan). Run oxide tests with `cargo xtask oxide --test <file> [-- <filter>]`.

- [ ] v0.2.227 Decoder Task 8: resident oxide bf16 forward. Suite `oxide_lm_train` (new). Done when oxide logits agree with the CPU reference within the plan's tolerance.
- [ ] v0.2.228 Decoder Task 9: resident oxide backward and updates. Done when gradients and one optimizer step agree with the CPU reference.
- [ ] -- Decoder Task 10: Latin pilot and evaluation. User-gated (training run, corpus). Prepare the code path only after R1 to R8 land.
- [ ] v0.2.229 Decoder Task 11: causal GQA flash and chunked cross-entropy, unclamped kernel family.
- [ ] v0.2.230 Decoder Task 12: opt-in decoder fp8 (reuse the machinery in `ROADMAP_FP8.md`). Optional.
- [ ] v0.2.231 Decoder Task 13: optional cuTile inference. Check first that the adapter builds against the current enum.
- [ ] v0.2.232 Decoder Task 14: causal sliding-window attention.
- [ ] v0.2.233 Decoder Task 15: optional DSA-style indexer top-k attention. Needs user selection.
- [ ] v0.2.234 Decoder Task 16: optional NSA-style three-branch attention. Needs user selection.

### Phase 3: Latin pilot and recipe

Plan: `plan_llm_training_recipe.md`. R1 to R8 and R12 are code and bounded synthetic tests (CPU line); R9 to R11 are user-gated runs. Persona rules in R5 are a hard requirement.

- [ ] v0.1.067 R1: corpus provenance, inventory and whole-work splits (reuse the existing `tools/prepare_latin.py` and `llm_prepare_corpus.rs` work where it fits).
- [ ] v0.1.068 R2: versioned chat tokens and `ChatTemplate`. Leave room in the vocabulary as described in [Known plan conflicts](#known-plan-conflicts).
- [ ] v0.1.069 R3: recipe configuration, schedules (WSD, re-warm, continuation), replay and resume.
- [ ] v0.1.070 R4: Latin evaluation and diagnostic suite; creates `llm_eval.md`.
- [ ] v0.1.071 R5: persona policy and verified quotation lookup.
- [ ] v0.1.072 R6: shifted assistant masks and SFT packing.
- [ ] v0.1.073 R7: real and synthetic SFT data preparation.
- [ ] v0.1.074 R8: bounded run orchestration and stage gates.
- [ ] -- User-gated. R9: stage-A recipe and pretraining run. Ask for the token count first.
- [ ] -- User-gated. R10: stage-B recipe and continuation run.
- [ ] -- User-gated. R11: stage-C recipe and SFT run.
- [ ] v0.1.075 R12: CPU chat session and CLI.
- [ ] v0.2.235 R13: oxide KV-cache chat integration (needs decoder Tasks 8 and 9).
- [ ] v0.1.076 R14: optional preference tuning. Needs user selection.
- [ ] -- R15: final acceptance evidence and model card. Publishing or distributing a checkpoint or demo is user-gated.

### Phase 4: Encoder, DeepSeek-style and recursive variants

Only tasks the user selects. Ask which experiments to enable before starting any item; each landed task passes its equivalence-to-dense or cache-versus-full-prefix test. The ranges below reserve provisional numbers per task in plan order.

- [ ] v0.1.077 to v0.1.095 `plan_llm_encdec.md` Tasks 17 to 35 (CPU: composition contracts, masked attention, RoPE/YaRN, GQA and low-rank MQA, MLA, encoder and decoder stacks, cross-attention, causal split, sparse reference, router, MoE FFN, balancing, MTP, checkpoint lifecycle). Blocked until selected.
- [ ] v0.2.236 to v0.2.239 `plan_llm_encdec.md` Tasks 36 to 39 (oxide encoder and cross-attention, latent and sparse caches, MoE and MTP, projection-scoped fp8). Blocked until selected.
- [ ] -- `plan_llm_encdec.md` Task 40: pilot configurations and acceptance matrix.
- [ ] v0.1.096 to v0.1.110 `plan_llm_recursive.md` Tasks L1 to L10 and L16 to L20 (CPU: loop configuration, block boundaries, fixed-loop forward, BPTT, truncated adjoints, sampled depth, caches, shared-first KV, checkpoints, schedules, exit distributions, adaptive execution, Mixture-of-Recursions). Blocked until selected.
- [ ] v0.2.240 to v0.2.245 `plan_llm_recursive.md` Tasks L11 to L15 and L21 (oxide fixed-loop forward and backward, sliding window, DSA and NSA across loops, resident adaptive execution). Blocked until selected.
- [ ] v0.1.111 `plan_llm_recursive.md` Task L22: CLI, acceptance matrix and evidence.

### Blocked or out of scope

- MoE, MTP, MLA, the causal split and sparse attention beyond sliding window: enabled only on the user's say-so.
- External golden or quality comparison: none exists; results compare against the repo's own CPU reference.
- Corpora whose licence is unconfirmed or noncommercial: do not train on them without the user's say-so (`latin_datasets.md`).

## Rules

- Use `cargo xtask` for every check, build, test and training command; do not type raw `cargo` or `systemd-run`. New subcommands are added as arms in `xtask/src/main.rs` and to the `CLAUDE.md` Commands table.
- One build or test command at a time. Build threads are capped at 4. Memory is capped at `MemoryHigh=3G`, applied by `cargo xtask` (it warns and runs uncapped when there is no user systemd bus; record that instead of claiming a capped run).
- `src/lightnet-cuda-oxide` and `src/lightnet-cuda-cutile` are not workspace members. Run oxide tests with `cargo xtask oxide --test <file> [-- <filter>]`.
- A zero-test filter, a timeout, a skipped GPU test or an unavailable device is not a pass.
- Each task follows red test, observed failure, implementation, observed green checks, local commit and tag, as the plan states.
- Commit only the files a task names; never `git add -A`. All files under `tasks/rust/` are tracked; a new file under an ignored path still needs `git add -f` with a named path (check with `git check-ignore`).
- Commit and tag on green, as in [01_rules/versioning.md](../../01_rules/versioning.md): every task whose named tests pass is committed and tagged at once by whoever implemented it, without asking. Message `v0.M.NNN: <what>`. Take the next free number from [version-map.tsv](../../02_state/version-map.tsv) for the line you are in (CPU increments use the `v0.1` line, GPU increments the `v0.2` line); never reuse, move or delete a tag. The numbers in the todo above are provisional; allocate the real one at commit time. Append the `version-map.tsv` row in a separate docs-only commit.
- Corpus training, tiny-corpus overfit runs, pilot evaluation runs and long benchmarks are user-gated. Bounded synthetic tests are fine.
- Corpus licences are mostly unconfirmed (see `latin_datasets.md`). Do not download, vendor or commit corpus text, and never commit `datasets/`; do not train on a source marked UNVERIFIED or noncommercial without the user's say-so. Record provenance for every text used. The 2.4 GB `datasets/latin/` is the user's: do not delete it under the big-file cleanup rule, since `docs/LLM_PLAN.md` and `tools/latin_pretraining.md` reference it.
- Never edit `third_party/` or `src/cutile-rs-aot` (pinned submodules).
- Delete weights, shards, logs and other big files a task produces once the evidence is recorded; keep referenced fixtures.
- Do not archive or remove completed plans in `tasks/rust/`; they are the roadmap.
- Keep existing comments in files you edit.
- Each edited Markdown file needs an updated TOC.
- Exact resume (RNG, data cursor, schedule, gradients, optimizer state) is required within one deterministic backend and configuration. CPU to GPU, or different GPU reduction orders, use tolerances, never bitwise claims.

## Known plan conflicts

Resolve these before or at the task that hits them; record the decision in that task's commit message.

- `plan_llm_decoder.md` calls `lightnet-lm` a new crate and lists `lightnet-core` cfg and error modules as reusable, while also saying `lightnet-lm` must not depend on `lightnet-core`. Task 1 settled this in `v0.1.042` (own types in `lightnet-lm`); confirm the plan text matches the code.
- Decoder Task 2 fixes token IDs 0 to 3 and bytes at 4 to 259, merges from 260. The recipe plan (R2) wants chat control tokens in a versioned reserved range at the top of the vocabulary and a bumped tokenizer artifact version. R2 must stay compatible with shards already written by the existing `llm_prepare_corpus.rs`.
- Decoder Task 6's `TokenSchedule` is a single warmup plus cosine. R3 needs WSD, re-warm and continuation; extend it there.
- The 3e-4 / 3e-5 learning-rate defaults in decoder Task 10 are unmeasured; treat them as sweep candidates.
- Latin u/v and i/j folding is lossy; default it to off and ablate it.
- Decoder Tasks 8 and 9 have no per-token eval readback or device KV-cache for chat; recipe tasks R4 and R13 add them.
- `docs/LLM_PLAN.md` and `plan_llm_training_recipe.md` describe the same recipe; the Phase 0 reconcile item decides which is authoritative.
- Codex model: use the model the user names for the session. The current `CLAUDE.md` rule is `gpt-6.1-terra` at medium reasoning effort; the plans were drafted with `gpt-6.1-sol`.

## Delegation protocol

The lead reads this file and the current plan, then runs the plan with `superpowers:subagent-driven-development`. This goal overrides the "no subagents" rule in [AGENTS.md](../../AGENTS.md) for plan-task work only; docs edits and ticking boxes are made directly.

1. One subagent per plan task (one tack each), in plan order. Give the subagent the full task text (the Interfaces block is how it learns names and types), the plan's Global Constraints and Review Focus, and the file paths it will touch. Do not ask it to read other plans.
2. Implementation subagents run on Sonnet. Read-only exploration runs on Sonnet or Haiku. The lead re-reads the cited lines for every fact a task depends on.
3. After each task a fresh reviewer subagent checks the diff against the task text and the tests; the lead merges nothing with failing tests or unexplained changes outside the named files.
4. Phase boundaries: after the last task of a phase, run the whole touched test files (`cargo xtask test lm_`, the touched `cargo xtask oxide --test oxide_lm_*` suites, and `cargo xtask check lightnet-lm`), then a whole-phase review on the most capable available model before moving on.
5. If a measurement or a verified fact contradicts the plan, stop that thread, record the result in the commit message and the docs, and re-plan before continuing.
6. Subagents never start training runs, push, download corpora or touch files outside their task.
7. Codex (`codex exec -m <model> -s read-only`) is allowed for read-only review and drafting only. It never edits the workspace; the lead or an implementation subagent applies changes.

## User-gated actions

Ask the user before each of these; do not infer approval from an earlier one:

- Starting any training run (R9, R10, R11), a tiny-corpus overfit demonstration, a pilot evaluation run, or a long benchmark. The user chooses the iteration or token count.
- Downloading, moving, deleting or storing any corpus, and using any source whose licence is unconfirmed or noncommercial.
- Pushing, opening a pull request, or rewriting history.
- Changing the default of a user-visible mode.
- Bumping a submodule pin.
- Publishing or distributing a trained checkpoint or a Cicero chat demo.
- Enabling any Phase 4 experiment (MoE, MTP, MLA, causal split, recursive decoder, sparse attention beyond sliding window).

## Reporting

At the end of each task, one line: task, tests run and their result, files changed, commit hash and tag, or why there is no commit (failing, timed-out or unrun tests). At the end of each phase, a short summary with the measurements (loss, perplexity, workspace bytes, step time where measured) against the plan's gate, and the verdict against the [Done when](#done-when) criteria. Report failures with the command output; never describe a failing, skipped or unrun test as passing. Mark anything not measured as unverified.

## Documentation updates

For every version-mapped increment: add the evidence line to [implemented.md](../../02_state/implemented.md), an entry in [STATUS.md](../../02_state/STATUS.md) "Latest increments", tick [todo.md](../../todo.md) and this roadmap, and add the row to `version-map.tsv` in the separate docs-only commit. Record data provenance in [llm_data.md](llm_data.md), evaluation results in `llm_eval.md` (created by R4) and training or benchmark numbers in `docs/OXIDE_BENCH.md`. Keep the `CLAUDE.md` Commands table in step with any new xtask subcommand, and update this file's [State snapshot](#state-snapshot) when a phase finishes. After a user correction, add the lesson to [lessons.md](../../lessons.md).

## Keeping this file current

- Tick an item only after its definition of done holds. Add a new item at the end of its phase with the next planned version; a ticked item keeps the version it shipped as.
- Update the [State snapshot](#state-snapshot) whenever a version lands, a measurement is recorded, or the user's data-preparation work is committed.
- Finished work goes in `02_state/implemented.md`, not here. Open work stays here and in `todo.md`.
- Each edited `.md` keeps its table of contents in sync with its headings.

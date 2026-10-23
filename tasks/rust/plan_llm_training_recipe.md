# Latin Pretrain, Mid-train and Cicero Chat Recipe Implementation Plan

> Drafted by gpt-6.1-sol (Codex CLI, reasoning effort high, read-only). Unreviewed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The writing-plans workflow also offers superpowers:subagent-driven-development, but the repository instruction to work inline without subagents takes precedence. Steps use checkbox syntax. This request authorizes drafting only; do not implement or execute this plan.

**Goal:** Build a reproducible pretraining, mid-training, and chat post-training recipe for a 30–60M-parameter Latin decoder, then expose a clearly disclosed Cicero simulation through an interactive Latin chat CLI.

**Architecture:** Extend the decoder plan rather than introduce another model or trainer. Stage A learns broad Latin; stage B continues on curated Latin and eligible Cicero works with replay; stage C learns dialogue formatting, persona behavior, and termination through supervised fine-tuning. Keep corpus provenance, whole-work splits, tokenizer identity, schedules, loader state, evaluation, and stage transitions reproducible.

**Tech Stack:** Existing Rust 2024 workspace, `nightly-2026-08-28`, `lightnet-lm`, thin `lightnet-cli` integration, and the excluded `lightnet-cuda-oxide` bridge. Reuse decoder `TokenBatch`, sparse CE, `Trainer`, `LmEngine`, `UpdateOptions`, checkpoints, CPU KV cache, and sampling. Use the serialization/hash dependencies introduced by decoder Task 2; do not add a Python training framework.

**Spec:** The user request; [decoder plan](plan_llm_decoder.md), especially Global Constraints, Tasks 1–11 and 14–16, and Open issues; the research notes supplied in the request. `tasks/rust/latin_datasets.md` and `tasks/rust/plan_llm_encdec.md` were checked and are absent. Repository instructions were read from `CLAUDE.md`, `tasks/rust/AGENTS.md`, and the relevant `CONTEXT.md` files.

**Global Constraints:** The section below applies to every task and run recipe.

**Review Focus:** Five cross-component failure modes below each have a named test in their owning task.

This document is proposed content for `tasks/rust/plan_llm_training_recipe.md`. No files were modified. No builds, tests, downloads, network requests, generator calls, or training runs were performed. Proposed APIs, commands, numerical defaults, and acceptance thresholds are specifications, not measured results.

## Table of Contents

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Verified vs unverified research basis](#verified-vs-unverified-research-basis)
  - [Repository evidence](#repository-evidence)
  - [Research evidence supplied with the request](#research-evidence-supplied-with-the-request)
  - [Findings against the existing decoder plan](#findings-against-the-existing-decoder-plan)
- [What a 30–60M model can and cannot do](#what-a-3060m-model-can-and-cannot-do)
  - [Expected capability boundary](#expected-capability-boundary)
  - [What scale-up would change](#what-scale-up-would-change)
- [Stage map and gates](#stage-map-and-gates)
  - [Decoder prerequisites](#decoder-prerequisites)
  - [Stage A: broad Latin pretraining](#stage-a-broad-latin-pretraining)
  - [Stage B: curated continuation and Cicero](#stage-b-curated-continuation-and-cicero)
  - [Stage C: chat post-training](#stage-c-chat-post-training)
  - [Gate criteria and authorization](#gate-criteria-and-authorization)
- [File structure](#file-structure)
  - [Relevant project tree](#relevant-project-tree)
  - [Component breakdown](#component-breakdown)
- [Tasks](#tasks)
  - [Task R1: corpus provenance, inventory, and whole-work splits](#task-r1-corpus-provenance-inventory-and-whole-work-splits)
  - [Task R2: versioned chat tokens and ChatTemplate](#task-r2-versioned-chat-tokens-and-chattemplate)
  - [Task R3: recipe configuration, schedules, replay, and resume](#task-r3-recipe-configuration-schedules-replay-and-resume)
  - [Task R4: Latin evaluation and diagnostic suite](#task-r4-latin-evaluation-and-diagnostic-suite)
  - [Task R5: persona policy and verified quotation lookup](#task-r5-persona-policy-and-verified-quotation-lookup)
  - [Task R6: shifted assistant masks and SFT packing](#task-r6-shifted-assistant-masks-and-sft-packing)
  - [Task R7: real and synthetic SFT data preparation](#task-r7-real-and-synthetic-sft-data-preparation)
  - [Task R8: bounded run orchestration and stage gates](#task-r8-bounded-run-orchestration-and-stage-gates)
  - [Task R9: stage-A recipe and user-gated pretraining](#task-r9-stage-a-recipe-and-user-gated-pretraining)
  - [Task R10: stage-B recipe and user-gated continuation](#task-r10-stage-b-recipe-and-user-gated-continuation)
  - [Task R11: stage-C recipe and user-gated SFT](#task-r11-stage-c-recipe-and-user-gated-sft)
  - [Task R12: CPU chat session and CLI](#task-r12-cpu-chat-session-and-cli)
  - [Task R13: oxide KV-cache chat integration](#task-r13-oxide-kv-cache-chat-integration)
  - [Task R14: optional preference tuning](#task-r14-optional-preference-tuning)
  - [Task R15: final acceptance evidence and model card](#task-r15-final-acceptance-evidence-and-model-card)
- [Progress table](#progress-table)
  - [Draft self-review](#draft-self-review)
- [Open issues](#open-issues)

## Global Constraints

The following repository constraints are reproduced verbatim from the request:

> use `cargo xtask` only (check, gpu-check, build, clippy, test, oxide, fmt, and the planned `lm` forwarder); never raw cargo or systemd-run; max 4 build threads; builds under MemoryHigh=3G (xtask applies it; record if it fell back uncapped); one build/test command at a time; stage only files named by a task, never `git add -A`; tasks/ is git-excluded so use `git add -f` for task docs; commit+tag on green as `v0.M.NNN: <what>` per tasks/rust/AGENTS.md Versioning, do not allocate numbers in the draft; common version files implemented.md, STATUS.md, todo.md and version-map.tsv follow the decoder plan's convention; CE keeps descent-direction sign (target minus prediction, summed loss, normalised once by active targets); unknown config/keys/special tokens are errors, no silent fallback; no C++ parity claims; long training runs, corpus downloads and generator-model calls are USER-GATED (plan and unit-test only, never run); delete large weights/logs after recording evidence; every edited Markdown file needs a TOC; read each affected crate's CONTEXT.md before implementing; DO NOT download corpora in any task without explicit user request; each task: red test -> observed failure -> implement -> observed green -> commit.

Additional application rules:

- Work inline, without subagents. Preserve unrelated changes and the detector graph.
- The current request authorizes only this document. Implementation requires a subsequent request.
- Every implementation task includes these common files in its **Files** scope: `tasks/rust/implemented.md`, `tasks/rust/STATUS.md`, `tasks/rust/todo.md`, `tasks/rust/version-map.tsv`, and this plan’s progress entries.
- Follow the decoder convention: feature commit with evidence and local tag; then a separate docs-only version-map commit containing the resulting full SHA. Preserve the live mapping format, including its existing trailing field.
- Docs-only increments use `docs(rust): ...`, without a version tag. Do not assign release numbers in advance.
- Use `PKGS=lightnet-lm cargo xtask test FILTER` for LM host tests: the inspected xtask defaults select `lightnet-core lightnet-cli`, not `lightnet-lm`.
- Run CLI tests separately with `PKGS=lightnet-cli`. Run bridge tests through `cargo xtask oxide --test SUITE`.
- A missing API/compiler error is acceptable initial red evidence. A zero-test selection, skip, timeout, or unavailable GPU is never green evidence.
- Corpus runs include tokenizer experiments, pilot sweeps, SFT, preference tuning, and model-based evaluation. Each requires explicit user authorization. Unit tests use small authored fixtures without network access.
- Unknown structural role names and unknown artifact IDs are errors. Literal marker spellings within message text remain ordinary text.
- Preserve PAD=0, BOS=1, EOS=2, UNK=3, byte IDs 4–259, and learned merges beginning at 260.
- Preserve `TokenBatch { batch, time, input_ids, targets, positions, segments, valid }`; assistant masking uses existing `targets=-100`.
- Default training is oxide bf16 working storage with f32 masters, gradients, moments, and reductions. Unsupported execution modes fail explicitly.
- Record actual successful-update tokens separately from input tokens, assistant targets, source exposure, and requested budget.
- Evaluate tokenizer variants on the same documents. Token NLL is not a cross-tokenizer ranking metric; normalization changes also alter the bits-per-byte denominator.
- Do not claim reliable factuality, instruction following, safety, or Latin fluency from synthetic unit tests.
- Preserve the selected deployable checkpoint and required resume artifacts at user-selected storage outside the repository. Record evidence before deleting disposable checkpoints, weights, and large logs; never delete an artifact still referenced by the final model card or a pending resume.

## Review Focus

| Failure mode | Required behavior | Owning test |
|---|---|---|
| An alternate edition or synthetic prompt reproduces an evaluation-only Cicero passage | Whole-work grouping and passage checks reject training admission, even with a different document ID | R1 `recipe_corpus_alias_leakage`; R7 `recipe_sft_synthetic_holdout_leakage` |
| Assistant mask is shifted against input positions instead of predicted tokens | Only assistant content and its end-of-turn target contribute; masked rows have exactly zero loss and seed | R6 `recipe_sft_shifted_mask` |
| Resume occurs at a replay boundary or LR transition | Next source, batch, LR, counters, and same-backend update match uninterrupted execution | R3 `recipe_schedule_mixer_resume_boundary` |
| A user types role-token spellings or chat truncates cached history | Text cannot inject control IDs; truncation rebuilds the cache from the retained system prompt and complete turns | R2 `recipe_chat_literal_injection`; R12 `recipe_chat_truncation_rebuild` |
| The model fabricates a citation or a judge rewards verbose imitation | Only verified lookup output can be presented as a quotation; shuffled and length-controlled judging retains human Latinist review | R5 `recipe_policy_forged_citation`; R4 `recipe_eval_judge_controls` |

## Verified vs unverified research basis

### Repository evidence

**VERIFIED by local inspection:**

- `src/lightnet-lm/src/lib.rs` contains only crate documentation. Its manifest depends only on `lightnet-cuda`. `CONTEXT.md` calls it a scaffold.
- `lightnet-lm` is already listed in workspace members and default members.
- The CLI manifest currently lacks a `lightnet-lm` dependency. No LM dispatch or xtask `lm` arm was found in the inspected entry points.
- The decoder plan specifies, rather than implements, the required model, tokenizer, loader, trainer, checkpoint, evaluation, and generation APIs.
- The decoder plan’s pilot is 12 layers, hidden 512, intermediate 1536, eight query heads, four KV heads, tied embeddings, and context 512 or 1024.
- Its exact parameter counts are 45,628,928 at vocabulary 8192 and 49,823,232 at vocabulary 16384.
- Decoder Task 6 defines warmup plus one token-count cosine schedule and exact checkpoint/loader restoration.
- Decoder Task 8 explicitly does not require native generation-cache integration.
- The optional Latin dataset and encoder-decoder plans are absent.
- No current build, device allocation, corpus inventory, tokenizer fertility, throughput, or training quality was verified.

### Research evidence supplied with the request

Here **VERIFIED** preserves the supplied notes’ label: the source was reportedly read during that research session, generally at abstract/page or search-summary level. These sources were **not independently revisited in this offline audit**.

| Topic | Basis and consequence |
|---|---|
| Compute-optimal scaling | **VERIFIED:** [Chinchilla](https://arxiv.org/abs/2203.15556) says model size and training tokens should scale equally. Its abstract supplies no tokens-per-parameter ratio. **UNVERIFIED:** “20 tokens per parameter”; never use it as this recipe’s budget. |
| Repeated data | **VERIFIED:** [data-constrained scaling](https://arxiv.org/abs/2305.16264) reports negligible loss change for up to four epochs relative to unique data, with diminishing returns beyond that. The experiments were larger than this model; transfer to 30–60M is **UNVERIFIED**. |
| Smaller models trained longer | **VERIFIED:** [Beyond Chinchilla-optimal](https://arxiv.org/abs/2401.00448) reports improvements at high token/model ratios when inference cost matters. This permits investigating longer training; it does not establish a Latin repetition optimum. |
| Continuous schedules | **VERIFIED:** [MiniCPM](https://arxiv.org/abs/2404.06395) introduces Warmup-Stable-Decay. Exact decay shape/fraction is **UNVERIFIED** here. |
| Vocabulary | **VERIFIED:** [vocabulary scaling](https://arxiv.org/abs/2407.13623) studies 33M–3B models and finds vocabulary should grow with compute. No optimal 8192/16384 choice for this corpus is established. |
| Tiny-model fluency | **VERIFIED:** [TinyStories](https://arxiv.org/abs/2305.07759) shows narrow, clean data can enable fluent output from small models. Transfer from simplified English stories to Latin conversation is **UNVERIFIED**. |
| Continued pretraining | **VERIFIED:** [continual pretraining](https://arxiv.org/abs/2403.08763) studies LR re-warming, re-decaying, and replay at substantially larger scales. [Domain/task adaptation](https://arxiv.org/abs/2004.10964) supports second-phase pretraining. |
| Changing stage mixes | **VERIFIED:** [OLMo 2](https://arxiv.org/abs/2501.00656) describes specialized mid-training data; [SmolLM2](https://arxiv.org/abs/2502.02737) describes multi-stage mixtures refined through ablations. Their proportions and scales do not specify this recipe. |
| Latin web data | **VERIFIED:** [CC100 dataset page](https://huggingface.co/datasets/statmt/cc100) lists Latin as 609M in the page’s size unit, not a measured token count, with license “unknown” and Common Crawl terms. Quality and usable quantity here are **UNVERIFIED**. |
| Classical text rights | **VERIFIED:** [canonical-latinLit](https://github.com/PerseusDL/canonical-latinLit) describes CC-BY-SA 4.0 unless noted and warns headers are not all checked. Individual works, editions, rights, and share-alike obligations require admission review. |
| Latin Library | **VERIFIED via supplied search summary:** [about page](https://thelatinlibrary.com/about.html) describes intended public-domain/personal or educational use, without an explicit general reuse license. Per-text reuse rights remain **UNVERIFIED**. |
| Larger Latin corpus precedent | **VERIFIED:** [Latin BERT](https://arxiv.org/abs/2009.10053) reports approximately 642.7M words across periods. Availability and licensing for this recipe are **UNVERIFIED**; these are not this recipe’s token counts. |
| Morphological tokenization | **VERIFIED via supplied search summary:** [arXiv 2511.09709](https://arxiv.org/abs/2511.09709) reports benefits for Latin encoder tokenization. Transfer to decoder byte BPE is **UNVERIFIED** and outside the initial tokenizer implementation. |
| Evaluation design | **VERIFIED via supplied search summary:** [EvaLatin 2022](https://aclanthology.org/2022.lt4hala-1.29/) evaluates lemmatization, PoS, and morphology across unseen author, genre, and time. Decoder probes require adaptation; that adaptation is **UNVERIFIED**. |
| Chat serialization | **VERIFIED:** [Transformers chat templating](https://huggingface.co/docs/transformers/main/en/chat_templating) describes control-token serialization and inference-only assistant openers. [TRL SFT](https://huggingface.co/docs/trl/sft_trainer) describes shifted CE, `-100` masking, assistant masks, packing, and termination alignment. |
| SFT limits | **VERIFIED:** [LIMA](https://arxiv.org/abs/2305.11206) studies curated SFT on a 65B model. It does not establish that small SFT repairs weak pretraining at 30–60M. |
| Prompt-loss ablation | **VERIFIED:** [Instruction Modelling](https://arxiv.org/abs/2405.14394) reports benefits from instruction-token loss in some settings. Its benefit here is **UNVERIFIED**; assistant-only remains default. |
| Synthetic instructions | **VERIFIED:** [Self-Instruct](https://arxiv.org/abs/2212.10560) uses a capable generator and filtering. A 30–60M Latin model is not a credible generator for its own chat curriculum. Specific external-generator terms are **UNVERIFIED**. |
| Preferences | **VERIFIED:** [DPO](https://arxiv.org/abs/2305.18290), [ORPO](https://arxiv.org/abs/2403.07691), and [KTO](https://arxiv.org/abs/2402.01306) describe different preference objectives. Benefit at 30–60M Latin scale is **UNVERIFIED**. |
| Judging | **VERIFIED:** [LLM-as-judge](https://arxiv.org/abs/2306.05685) reports strong agreement in its evaluated settings and position, verbosity, and self-enhancement biases. Latin competence of any proposed judge is **UNVERIFIED**. |
| Historical personas | **VERIFIED:** [Character-LLM](https://arxiv.org/abs/2310.10158) studies historical-character simulation and interview evaluation. Reliable factuality or safety for this model does not follow. |

**UNVERIFIED defaults or design judgments:** all proposed optimizer values, LR ranges, mixture proportions, replay fractions, stopping thresholds, Latin-fold benefits, cloze/probe effectiveness, tiny-model chat capability, persona rules, and safety effectiveness.

No public Latin decoder-LM benchmark was identified in the supplied research; the search was limited. Do not present this as proof that none exists.

### Findings against the existing decoder plan

| Verified finding | Recipe consequence |
|---|---|
| Task 2 reserves only IDs 0–3, fixes bytes at 4–259, and starts merges at 260 | R2 adds an explicit versioned reserved-control range at the vocabulary’s high end. It cannot insert controls before byte or merge IDs. |
| Plain `ByteBpe::encode` treats marker spellings as text | Preserve that contract. Only typed `ChatTemplate` serialization inserts role IDs. |
| Tasks 1–2 have no `ChatMessage`, chat template, or assistant mask | R2 defines serialization; R6 converts its predicted-token mask into `TokenBatch.targets`. |
| Task 3 supports `-100` and summed, descent-sign CE, but no weighted objective | Assistant-only requires no new CE kernel. An optional prompt-weight experiment requires an explicit extension and separate tests. |
| Latin folding replaces distinctions, including `v/u`, `j/i`, and ligatures | R2 defaults to fold off; retain original bytes and compare fold variants without claiming lossless reconstruction. |
| Task 6 has a single warmup/cosine `TokenSchedule` | R3 adds checkpointed WSD, continuation, and explicit stage re-warm behavior; it must preserve legacy schedule loading. |
| Task 10’s `3e-4`/`3e-5` LR values are pilot defaults | Treat them as unmeasured candidates, not validated tiny-model optima. |
| Task 8 permits quadratic attention and excludes generation-cache integration | Memory fit may require Task 11. R13 separately supplies device KV-cache chat. |
| Tasks 8–9 do not specify a per-token evaluation readback API | R4 adds a loss-only evaluation API without changing training residency claims. |
| Open issues describe single-device execution and unverified VRAM fit | Accumulation is the initial scaling mechanism. Distributed training is outside this plan. |
| Tasks 14–16 are optional sparse-attention work | They are not prerequisites at context 512–1024 and do not enter this recipe. |
| The live crate is a scaffold and CLI wiring is absent | None of the decoder-plan interfaces may be assumed implemented or green. |

## What a 30–60M model can and cannot do

### Expected capability boundary

These are **UNVERIFIED expectations**, not promised outcomes.

A clean, narrow curriculum may produce short, locally fluent Latin, plausible morphology, recognizable Ciceronian phrasing, and brief dialogue-shaped replies. That would be a useful outcome.

Fluent short Latin is different from real conversation. This scale is unlikely to provide dependable factual answers, sustained reasoning, broad instruction following, long-range coherence, reliable multilingual understanding, or robust memory across many turns. Repeated exposure to a small corpus increases the risk of reproducing passages rather than composing useful new replies.

SFT teaches formatting and behavior on top of existing knowledge. It cannot reliably create knowledge, reasoning, or Latin competence absent from pretraining. A 30–60M model must not generate its own synthetic instruction curriculum.

The product must disclose that it simulates Cicero. Generated Latin must never be presented as an authenticated quotation. The policy shim and test suite reduce specific failures; they cannot guarantee semantic safety or historical accuracy.

### What scale-up would change

More parameters could increase capacity, but would also increase optimizer state, activation memory, and data requirements. More diverse, rights-cleared Latin could matter more than adding parameters to an exhausted corpus.

Bilingual instruction following would require an explicit bilingual curriculum and evaluation, rather than assuming Latin pretraining understands English or Italian. Longer usable conversation would require training and testing at longer contexts, not merely increasing a cache allocation.

Larger pretrained models may make instruction and preference tuning more effective. No parameter threshold or quality gain is established here. Multi-GPU training, imported pretrained models, retrieval-based factual answering beyond verified quotations, and long-context sparse attention require separate plans.

## Stage map and gates

```text
Decoder correctness + rights/split/tokenizer gate G0
                         |
                         v
A: broad Latin pretraining
                         |
                  gate GAB + user approval
                         v
B: curated Latin + eligible Cicero continuation + A replay
                         |
                  gate GBC + user approval
                         v
C: assistant-masked Latin chat SFT
                         |
                  gate GCHAT + user approval
                         v
Disclosed Cicero chat CLI
                         |
              optional preference experiment
                         |
                  evaluation-win gate
                         v
Retain candidate only if it improves the accepted SFT model
```

### Decoder prerequisites

Before **any stage-A corpus run**, decoder Tasks **1–10 must be green**, including:

| Decoder task | Required evidence |
|---|---|
| 1 | Strict geometry, integer IDs, masks, and exact pilot parameter count |
| 2 | Deterministic BPE, normalization, corrupt-artifact rejection, literal-marker safety |
| 3 | Exact CPU adjoints; summed CE; seeds `one_hot-softmax`; ignored rows zero |
| 4 | Complete trainable CPU decoder, tied storage, causality, dropout replay |
| 5 | Whole-work manifests, shifted packing, segment isolation, exact loader restore |
| 6 | AdamW, active-target normalization, clipping, atomic rejection, exact CPU resume, capped LM forwarder |
| 7 | CPU cached/full-prefix agreement, deterministic sampling, excluded tokens, EOS |
| 8 | Actual-device bf16 forward, packed causality, transfer accounting |
| 9 | Actual-device backward/update, ignored-target behavior, residency, deterministic device resume |
| 10 | Pilot configuration, aggregate evaluation, bits-per-byte, corpus validation, forced choice |

“Task 10 green” means its implementation and bounded tests; its user-gated corpus pilot is not an implicit prerequisite run.

Decoder Task **11 is additionally mandatory** when the proposed context/microbatch configuration does not fit the measured memory allowance using Tasks 8–9, or when the recipe selects flash attention, chunked head/CE, or recomputation. Require its chunking, mask, gradient, workspace, and RNG tests. Otherwise keep the baseline mode.

Recipe CPU-only tasks may be implemented once their specific decoder dependencies are green. GPU corpus runs wait for the complete G0 evidence bundle.

Decoder Tasks 12–16 are not required. FP8, cuTile, sliding-window, DSA, and NSA remain outside the baseline.

### Stage A: broad Latin pretraining

Hold **all Cicero text out of stage A**. This gives stage B a meaningful adaptation target and keeps the curriculum interpretable. Deduplication must also catch Cicero passages quoted inside later texts.

Initial mixture proposal, **UNVERIFIED and subject to measured availability and short authorized ablations**:

| Stratum | Proposed sampling share | Constraint | Accepted bytes | Unique eligible targets |
|---|---:|---|---|---|
| Classical prose excluding Cicero | 0.50 | Rights-cleared editions; author/work diversity | to be measured | to be measured |
| Classical verse | 0.20 | Preserve verse boundaries; report separately | to be measured | to be measured |
| Medieval Latin | 0.15 | Explicit period/genre labels | to be measured | to be measured |
| Neo-Latin | 0.10 | Explicit period and rights review | to be measured | to be measured |
| Vulgate | 0.03 | Hard proposed cap 0.05 | to be measured | to be measured |
| Filtered CC100 `la` | 0.02 | Hard proposed cap 0.10; unresolved rights excluded | to be measured | to be measured |

These shares are starting hypotheses, not inferred corpus sizes. An absent stratum is an error for this configuration: explicitly approve another mixture rather than silently renormalizing.

Split whole works before normalization fitting, tokenizer training, filtering threshold tuning, or shard construction. Group alternate editions under one canonical work ID. Exact and near duplicates crossing splits must be resolved before admission.

Let:

- \(U_i\): measured unique eligible next-token targets in admitted training shards for source \(i\);
- \(q_i\): configured target sampling fraction, with \(\sum_i q_i=1\);
- \(E_i\): proposed maximum exposure, initially at most 4;
- \(B_{\text{user}}\): explicitly authorized successful-update token ceiling.

Choose:

\[
B_A=\left\lfloor\min\left(B_{\text{user}},
\sum_i E_iU_i,
\min_{i:q_i>0}\frac{E_iU_i}{q_i}\right)\right\rfloor.
\]

The loader also enforces actual per-work/source exposure limits; the formula is an initial feasibility bound, not permission to oversample a small work.

Four epochs is the only verified repetition datapoint in the supplied notes, and it comes from larger models. It is a conservative experimental ceiling here, not an optimality claim. Exceeding it requires a separate user-authorized experiment and explicit overfitting/memorization evidence.

Tokenizer experiments:

- Total vocabulary: 8192 versus 16384, **including four chat controls**.
- Latin fold: off versus on; off is the baseline.
- Fit BPE only on the stage-A training split, using a bounded deterministic sample.
- Measure tokens per whitespace-delimited word, tokens per normalized byte, achieved merge count, and representative morphological splits on development works.
- Use matched, bounded LM pilots to select among viable tokenizers. Lower fertility alone does not establish better LM quality.
- Keep final evaluation works out of tokenizer fitting and experiment selection.

Optimizer and batching proposals are **UNVERIFIED sweep defaults**:

| Parameter | Starting experiment |
|---|---|
| Precision | bf16 work buffers; f32 masters/gradients/moments/reductions |
| Optimizer | AdamW; β1=0.9, β2=0.95, ε=`1e-8` |
| Peak LR | candidates `3e-4`, `6e-4`, `1e-3` |
| Final LR | initially 0.1 × peak |
| Schedule | matched-budget cosine versus WSD |
| Warmup | candidates 1% and 3% of the authorized token budget |
| WSD fractions | warmup/stable/decay = 0.03/0.77/0.20 initially |
| Effective active targets/update | start 32,768; compare 131,072 and 262,144 only when budget permits |
| Weight decay | start 0.1; compare 0 and 0.1 |
| Global norm cap | start 1.0 |
| Context | 512 first; 1024 is a separate matched experiment |
| Dropout | decoder pilot value 0 initially |

Choose the lowest validation NLL/bits-per-byte candidate satisfying finite-update, memory, and throughput constraints at matched successful-update tokens. Break statistically indistinguishable results in favor of the simpler/lower-memory configuration. Record accumulation and microbatch size separately.

Evaluate every configured token interval; proposed default is approximately 1% of the authorized budget, rounded up to a complete update. Keep at least five evaluations when the budget supports them. Record per-work/author/period loss, fixed-seed generations, source exposure, memory, and steady-state tokens/second.

Stop on budget exhaustion, exposure exhaustion, non-finite updates, or the configured plateau/overfit rule. A proposed plateau is improvement below 0.01 nats/token for four successive evaluations. A proposed overfit alarm is a train–validation gap growing by more than 0.10 nats/token from its best checkpoint across three evaluations. These are tunable operational thresholds, not research-established values.

### Stage B: curated continuation and Cicero

Continue from the selected A checkpoint with the **same tokenizer and vocabulary**. Do not retrain BPE or resize tied embeddings.

Use:

- cleaner, manually reviewed Latin;
- eligible Cicero dialogues and letters;
- replay sampled from stage A’s admitted distribution.

Reserve `Laelius de Amicitia` and the whole `Epistulae ad Atticum` collection as evaluation-only canonical work groups. Their availability, edition rights, and extraction remain unverified. If unavailable, the stage is blocked until the user explicitly selects another whole dialogue and another whole letter collection and freezes the replacement split.

Candidate training material includes `De Oratore`, `Tusculanae Disputationes`, `De Re Publica`, `Cato Maior de Senectute`, and eligible non-held-out letter collections. Work presence and rights are checked during ingestion, not assumed from titles.

Replay fraction is a sweep parameter: proposed candidates 0.05, 0.15, and 0.25. Replay earlier Latin, not English or Italian. Select the fraction using Cicero validation improvement and stage-A forgetting.

Re-warm to a lower peak: proposed candidates 0.1 and 0.3 times the selected stage-A peak. Compare lower-peak cosine with a WSD continuation/decay. Preserve the absolute successful-update token counter; record a new stage-local schedule origin.

Apply the same measured-budget rule using remaining exposure allowance. Record cumulative A+B exposures so replay cannot silently bypass stage-A repetition limits.

Define forgetting:

\[
F_A=NLL_{\text{A-val}}(\text{B checkpoint})
-NLL_{\text{A-val}}(\text{selected A checkpoint}).
\]

Report it globally and by author/period. The proposed gate allows at most +0.05 nats/token aggregate forgetting while requiring at least 0.02 nats/token improvement on the frozen Cicero development set. These are engineering acceptance proposals.

### Stage C: chat post-training

Use typed chat serialization, assistant-only shifted targets, and packed-example segment isolation. Train assistant content and its end-of-turn token. Do not train user text merely because it occupies an input position.

Default supervised curriculum:

1. Provenance-preserving adaptations of eligible real dialogue passages.
2. Explicitly labeled letter-to-dialogue adaptations.
3. Template/rule conversations.
4. Optionally, externally generated conversations admitted only after user authorization, rights review, Latin-quality filtering, and holdout checks.

A small model cannot reliably create this curriculum itself. External-generator prompts must exclude held-out work text, excerpts, translations, and close paraphrases.

Start with 1–3 epochs over accepted SFT examples, selecting by held-out assistant NLL and the human-reviewed chat rubric. These are **UNVERIFIED defaults**. Count assistant targets, prompt tokens, example exposures, and total input tokens separately.

Optional prompt loss uses weight 0.05 as an explicitly tested ablation; assistant-only remains the baseline. Optional preference tuning follows SFT and is retained only after an evaluation win.

### Gate criteria and authorization

All numerical quality thresholds below are proposed product gates, not measured outcomes.

| Gate | Measurable criteria | Who authorizes | Recorded evidence |
|---|---|---|---|
| G0 → A | Decoder Tasks 1–10 green; Task 11 green if selected/needed; recipe tests green; zero unresolved split/rights violations; exact tokenizer/config hashes; measured memory fit | Engineer verifies software; user explicitly authorizes corpus preparation and each training/sweep budget | Test commands/results, device, cap status, rights decisions, inventory, splits, tokenizer experiment, run specification |
| A → B | Finite training; valid resumable checkpoint; A validation NLL at least 0.05 below same-model initialization; budget/exposure compliance; no unresolved regurgitation alert | Engineer records gate result; user authorizes B run | Selected A SHA/hash, curves, fixed generations, exposure report, review decisions |
| B → C | Cicero development NLL improves ≥0.02; aggregate A forgetting ≤0.05; both evaluation-only Cicero work groups untouched; no unresolved leakage alert | Engineer and Latinist review; user authorizes SFT data/generation and C run | A/B comparison, per-work losses, replay choice, provenance and holdout audit |
| C → chat | On 40 frozen prompts, ≥80% score at least 3/5 for Latin quality and persona fidelity; zero fabricated quotations presented as verified; refusal suite ≥95%; no unresolved memorization alert; cache/termination tests green | Latinist signs quality assessment; user accepts model and chat execution | Blind rubric sheets, regression output, policy results, hashes, limitations |
| SFT → preference candidate | Rights-cleared labels, frozen SFT reference, baseline report, explicit budget | User authorizes optional experiment | Reference/cache identity, label provenance, run specification |
| Preference candidate → retained | Human-reviewed blinded win rate >0.50, with no regression beyond 0.05 assistant nats/token and no new critical policy failure; inconclusive results retain SFT | Latinist reviews; user chooses retained artifact | Paired judgments, uncertainty/sample size, length controls, loss/policy comparison |

Gate failure stops advancement. A later user request can authorize a revised experiment; it must not silently turn failed evidence into a pass.

## File structure

### Relevant project tree

`[M]` means modify after decoder prerequisites exist; `[N]` means create; `[R]` means read-only dependency/reference. Files supplied by the decoder plan remain decoder-owned prerequisites.

```text
lightnet/
├── CLAUDE.md                                           [M: command table]
├── Cargo.lock                                          [M: only if dependencies change]
├── xtask/
│   └── src/main.rs                                     [M: planned lm forwarding]
├── cfg/
│   ├── CONTEXT.md                                      [R]
│   ├── llm_latin.cfg                                   [R: decoder Task 10]
│   ├── lm_latin_a.json                                 [N]
│   ├── lm_latin_b.json                                 [N]
│   ├── lm_latin_sft.json                               [N]
│   └── lm_cicero_persona.json                          [N]
├── src/
│   ├── lightnet-lm/
│   │   ├── CONTEXT.md                                 [M]
│   │   ├── Cargo.toml                                 [M: only required deps]
│   │   ├── src/
│   │   │   ├── lib.rs                                 [M: exports]
│   │   │   ├── types.rs                               [R: TokenBatch/LossSummary]
│   │   │   ├── tokenizer.rs                           [M: v2 control reservation]
│   │   │   ├── chat.rs                                [N: template/session]
│   │   │   ├── corpus.rs                              [N: provenance/splits]
│   │   │   ├── recipe.rs                              [N: strict configuration]
│   │   │   ├── schedule.rs                            [N: WSD/continuation]
│   │   │   ├── replay.rs                              [N: deterministic mixing]
│   │   │   ├── data.rs                                [M: shard metadata]
│   │   │   ├── optim.rs                               [M: schedule dispatch]
│   │   │   ├── train.rs                               [M: recipe lifecycle]
│   │   │   ├── checkpoint.rs                          [M: versioned extensions]
│   │   │   ├── eval.rs                                [M: existing eval reuse]
│   │   │   ├── recipe_eval.rs                         [N: grouped diagnostics]
│   │   │   ├── persona.rs                             [N: policy/quotation lookup]
│   │   │   ├── sft.rs                                 [N: masks/loader]
│   │   │   ├── sft_data.rs                            [N: extraction/filtering]
│   │   │   ├── run.rs                                 [N: cadence/gates]
│   │   │   ├── preference.rs                          [N: optional DPO]
│   │   │   ├── cache.rs                               [R: decoder Task 7]
│   │   │   └── sampling.rs                            [R: decoder Task 7]
│   │   └── tests/
│   │       ├── recipe_corpus.rs                       [N]
│   │       ├── recipe_chat.rs                         [N]
│   │       ├── recipe_schedule.rs                     [N]
│   │       ├── recipe_eval.rs                         [N]
│   │       ├── recipe_policy.rs                       [N]
│   │       ├── recipe_sft.rs                          [N]
│   │       ├── recipe_sft_data.rs                     [N]
│   │       ├── recipe_run.rs                          [N]
│   │       ├── recipe_stages.rs                       [N]
│   │       └── recipe_preference.rs                   [N]
│   ├── lightnet-cli/
│   │   ├── CONTEXT.md                                 [M]
│   │   └── src/
│   │       ├── main.rs                                [R: decoder dispatch]
│   │       ├── lm.rs                                  [M: thin subcommands]
│   │       └── lm_recipe.rs                           [N: recipe CLI parsing]
│   └── lightnet-cuda-oxide/
│       ├── CONTEXT.md                                 [M]
│       ├── src/
│       │   ├── lm.rs                                  [M: eval/weighted/DPO hooks]
│       │   ├── lm_chat.rs                             [N: device KV cache]
│       │   ├── lib.rs                                 [M: exports]
│       │   └── kernels/
│       │       └── lm.rs                              [M: required device kernels]
│       └── tests/
│           ├── oxide_lm_recipe_eval.rs                 [N]
│           ├── oxide_lm_sft.rs                         [N]
│           ├── oxide_lm_chat.rs                        [N]
│           └── oxide_lm_preference.rs                  [N: optional]
├── tools/
│   ├── CONTEXT.md                                      [M]
│   ├── lib_cap.sh                                      [R]
│   ├── test_oxide.sh                                   [R]
│   └── lm_recipe.sh                                    [N: capped forwarding only]
└── tasks/rust/
    ├── AGENTS.md                                       [R]
    ├── CONTEXT.md                                      [R]
    ├── plan_llm_decoder.md                             [R]
    ├── plan_llm_training_recipe.md                     [N: this document]
    ├── llm_data.md                                     [M: decoder document]
    ├── llm_eval.md                                     [M: decoder document]
    ├── llm_recipe_evidence.md                          [N: compact run evidence]
    ├── llm_cicero_model_card.md                        [N]
    ├── implemented.md                                 [M: common version file]
    ├── STATUS.md                                      [M: common version file]
    ├── todo.md                                        [M: common version file]
    └── version-map.tsv                                [M: separate mapping commit]
```

Runtime corpora, shards, generated datasets, checkpoints, and large logs live under user-supplied external directories. Their hashes and compact reports enter evidence documents; their contents do not become repository fixtures.

### Component breakdown

| Component/files | Role and responsibilities | Depends on / depended on by | Constraints or open questions |
|---|---|---|---|
| `corpus.rs`, `data.rs` | Edition provenance, canonical works, split registry, exposure inventory, shard metadata | Decoder Tasks 2/5/10; consumed by all stages and SFT | Actual sources, licenses, and quantities unmeasured |
| `tokenizer.rs`, `chat.rs` template portion | Reserved controls, versioned artifacts, safe typed chat serialization | Decoder Tasks 1/2; consumed by SFT and chat | Preserve byte/merge IDs; no literal injection |
| `recipe.rs`, JSON configs | Strict stage settings and required measured inputs | Decoder config; consumed by run/schedule/mixer | Unknown keys fail; experiment values labeled unverified |
| `schedule.rs`, `optim.rs` | Cosine/WSD, stage-local origin, re-warm/decay | Decoder Task 6; consumed by trainer | Successful-update tokens only; explicit transition semantics |
| `replay.rs`, `checkpoint.rs`, `train.rs` | Deterministic source selection, exposure limits, complete resume | Decoder Tasks 5/6/9; consumed by A/B/C runner | No hidden reset of RNG, counters, moments, or loader |
| `recipe_eval.rs`, `eval.rs` | Per-group loss, cloze/probes, generations, judge controls, overlap reports | Decoder Tasks 7/10; consumed by gates/model card | Adapted probes are experimental; human Latinist mandatory |
| `persona.rs`, persona JSON | Latin disclosure, fixed policy responses, verified quotations | Corpus registry; consumed by SFT/chat/eval | Policy cannot guarantee semantic factuality |
| `sft.rs` | Correct shift, assistant masks, packed segments, resumable examples | Chat template, CE, loader contracts | Assistant EOT supervised; adjacent examples isolated |
| `sft_data.rs` | Dialogue/letter adaptation, synthetic export/import, provenance filters | Corpus/split registry and persona policy | Edition adapters need supplied source formats; generator terms unresolved |
| `run.rs` | Cadence, stop rules, gate reports, stage run specifications | Trainer/evaluation/source APIs | Does not grant user authorization |
| `chat.rs` session portion | Multi-turn history, sampling constraints, overflow/cache lifecycle | Decoder Task 7 and persona/template | Keep system prompt; drop whole old turns; rebuild cache |
| `lm_chat.rs`, bridge kernels | Device prefill/decode and cache agreement | Decoder Task 8; consumed by oxide CLI | Not already provided by Tasks 8–9 |
| `preference.rs`, optional bridge hooks | DPO reference cache and objective | Accepted SFT engine/checkpoint | Optional; additional memory/compute; tiny-scale benefit unverified |
| CLI files | Parse strict proposed commands and delegate semantics | LM crate and optional oxide bridge | No corpus/model semantics in argument parser |
| `tools/lm_recipe.sh`, xtask arm | Route CPU/oxide execution through capped runners | Existing cap/codegen tools | No direct script invocation in recipes; maximum four threads |
| `CONTEXT.md` files, `CLAUDE.md`, task docs | Keep component maps, commands, evidence, limitations current | Every owning task | Every edited Markdown file needs a TOC |
| Common version files | Local feature/tag evidence and SHA mapping | Every completed implementation task | Explicit path staging; no preallocated version numbers |

## Tasks

All recipe signatures below use `lightnet_lm::recipe::Result<T>`, defined by R1 as `std::result::Result<T, RecipeError>`. Map decoder errors explicitly; do not require a new `lightnet-core` dependency.

All **Files** lists additionally include the common version files named in Global Constraints. A task owns export additions in `src/lightnet-lm/src/lib.rs` and map updates in the affected crate’s `CONTEXT.md`.

Tasks are ordered so implementation can proceed inline. User-gated runs remain separate from their software increments. R9–R11 software tests do not depend on having trained checkpoints.

### Task R1: corpus provenance, inventory, and whole-work splits

**Files**

- Create: `src/lightnet-lm/src/corpus.rs`, `src/lightnet-lm/src/recipe.rs`, `src/lightnet-lm/tests/recipe_corpus.rs`.
- Modify: `src/lightnet-lm/src/data.rs`, `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, `tasks/rust/llm_data.md`.

**Interfaces**

Consumes:

- `validate_corpus_manifest(documents: &[CorpusDocument]) -> decoder::Result<()>`.
- `write_token_shards(documents: &[CorpusDocument], tokenizer: &ByteBpe, out: &Path, shard_tokens: usize) -> decoder::Result<ShardManifest>`.

Produces:

- `validate_recipe_corpus(documents: &[RecipeDocument], split: &SplitRegistry) -> Result<CorpusInventory>`.
- `measure_shards(manifest: &ShardManifest, documents: &[RecipeDocument]) -> Result<CorpusInventory>`.
- `reject_holdout_overlap(text: &str, split: &SplitRegistry) -> Result<()>`.

Define:

- `RecipeError`: typed invalid-config, provenance, rights, leakage, artifact, unsupported, and I/O errors; error messages include the offending field/work.
- `Stage::{A, B, C}`.
- `RecipeDocument { base: CorpusDocument, author_id: String, period: String, genre: String, edition_id: String, canonical_work_id: String, source_url: String, rights_status: RightsStatus, rights_evidence: String, attribution: String, extraction_version: String, original_sha256: String }`.
- `RightsStatus::{Approved, Rejected, Unresolved}`; only `Approved` is trainable.
- `SplitRegistry { aliases: BTreeMap<String,String>, assignments: BTreeMap<String,Split>, eval_only: BTreeSet<String>, holdout_passage_hashes: BTreeSet<String> }`.
- `Split::{Train, Development, Test}`.
- `CorpusInventory { manifest_hash: String, source_targets: BTreeMap<String,u64>, work_targets: BTreeMap<String,u64>, normalized_bytes: BTreeMap<String,u64>, original_bytes: BTreeMap<String,u64> }`.

The registry permanently identifies evaluation-only works, even when files arrive under new editions. Record normalized exact hashes and the decoder’s five-word-shingle/Jaccard ≥0.8 audit. Cross-work passage quotations require exclusion/review, not just work-ID checks.

- [ ] Write `recipe_corpus_alias_leakage`: editions `laelius-ed1` and `laelius-ed2` map to `cicero-laelius`; assigning one to training and one to test returns `Leakage`.
- [ ] Write `recipe_corpus_rights_and_inventory`: unresolved rights are rejected; two authored shard fixtures with 3 and 5 eligible targets report exactly 8, without counting PAD or ignored targets.
- [ ] Write `recipe_corpus_stage_a_excludes_cicero`: author `cicero` is rejected from A; a quoted held-out passage under a different author also fails.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_corpus_`; record missing APIs as red.
- [ ] Implement the records and three functions in the named files, reusing decoder manifest validation.
- [ ] Run the same filter; require every named test to pass with nonzero selection.
- [ ] Commit/tag the named files as `v0.M.NNN: add Latin recipe provenance and work splits`.

**GPU (oxide) needed:** none. CPU-only schema, hashing, shard inventory, and admission validation. No downloads.

### Task R2: versioned chat tokens and ChatTemplate

**Files**

- Create: `src/lightnet-lm/src/chat.rs`, `src/lightnet-lm/tests/recipe_chat.rs`.
- Modify: `src/lightnet-lm/src/tokenizer.rs`, `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, `tasks/rust/llm_data.md`.

**Interfaces**

Consumes:

- `ByteBpe::train(documents: &[Vec<u8>], vocab_size: usize, normalization: NormalizationSpec) -> decoder::Result<Self>`.
- `ByteBpe::encode(&self, text: &str) -> decoder::Result<Vec<u32>>`.
- `ByteBpe::artifact_hash(&self) -> String`.

Produces:

- `ByteBpe::train_chat(documents: &[Vec<u8>], vocab_size: usize, normalization: NormalizationSpec) -> Result<Self>`.
- `ByteBpe::chat_tokens(&self) -> Result<ChatTokenIds>`.
- `ByteBpe::defined_ids(&self) -> Vec<u32>`.
- `ChatTemplate::new(tokens: ChatTokenIds) -> Result<Self>`.
- `ChatTemplate::encode_chat(&self, tokenizer: &ByteBpe, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<EncodedChat>`.

Define:

- `Role::{System, User, Assistant}`.
- `ChatMessage { role: Role, content: String }`.
- `ChatTokenIds { system: u32, user: u32, assistant: u32, end_turn: u32 }`.
- `EncodedChat { input_ids: Vec<u32>, assistant_mask: Vec<bool> }`.

Artifact decision:

- Chat tokenizer schema is version **2**; version 1 remains readable for plain text but `chat_tokens()` errors.
- Reserve controls from the start at `vocab_size-4 .. vocab_size-1`, ordered system/user/assistant/end-turn.
- At 8192: 8188, 8189, 8190, 8191. At 16384: 16380, 16381, 16382, 16383.
- Bytes remain 4–259. Merges begin at 260 and cannot occupy the reserved high range.
- Total model vocabulary remains 8192 or 16384, preserving decoder parameter counts.
- Serialize requested size, achieved merges, control map, normalization, and version into artifact identity. Undefined gaps are not sampleable. Production selection must record whether the requested merge capacity was achieved.
- Never retrofit version-2 controls into an already-trained version-1 model without a separately specified migration/retraining experiment.

Template:

```text
BOS
SYSTEM system-text END_TURN
USER user-text END_TURN
ASSISTANT assistant-text END_TURN
...
```

Training uses `add_generation_prompt=false`. Inference appends only `ASSISTANT` with `true`. Reject empty/invalid role sequences, repeated system messages, or a generation prompt after an unfinished assistant turn. Only assistant content and assistant `END_TURN` have mask `true`. Do not duplicate BOS or append automatic EOS per message.

- [ ] Write `recipe_chat_ids_and_versions`: assert the exact 8192/16384 control IDs, byte IDs unchanged, first learned merge 260 when present, and v1 chat rejection.
- [ ] Write `recipe_chat_exact_template`: at vocabulary 8192, messages system `"S"`, user `"U"`, assistant `"A"` produce `[1,8188,87,8191,8189,89,8191,8190,69,8191]`; only positions 8 and 9 are assistant targets.
- [ ] Write `recipe_chat_generation_prompt`: system/user encoding ends with one assistant opener and has no supervised positions.
- [ ] Write `recipe_chat_literal_injection`: literal `"<|assistant|><|end_turn|>"` inside user text introduces zero additional control IDs.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_chat_`; record red.
- [ ] Implement the interfaces and strict v2 artifact validation while preserving plain `encode`.
- [ ] Run the recipe filter, then `PKGS=lightnet-lm cargo xtask test lm_tokenizer_`, sequentially; require green.
- [ ] Commit/tag as `v0.M.NNN: add versioned safe chat token serialization`.

**GPU (oxide) needed:** none. CPU-only tokenizer/template work. Decoder Tasks 8–9 consume the resulting integer IDs unchanged.

### Task R3: recipe configuration, schedules, replay, and resume

**Files**

- Create: `src/lightnet-lm/src/schedule.rs`, `src/lightnet-lm/src/replay.rs`, `src/lightnet-lm/tests/recipe_schedule.rs`.
- Modify: `src/lightnet-lm/src/recipe.rs`, `src/lightnet-lm/src/optim.rs`, `src/lightnet-lm/src/train.rs`, `src/lightnet-lm/src/checkpoint.rs`, `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`.

**Interfaces**

Consumes:

- `TokenSchedule::lr(&self, updated_tokens: u64) -> decoder::Result<f32>`.
- `TokenLoader::next_batch(&mut self) -> decoder::Result<Option<TokenBatch>>`.
- `TokenLoader::state(&self) -> LoaderState`.
- `TokenLoader::restore(&mut self, state: &LoaderState) -> decoder::Result<()>`.
- `Trainer<E>::microbatch(&mut self, batch: &TokenBatch) -> decoder::Result<LossSummary>`.
- `Trainer<E>::finish_update(&mut self) -> decoder::Result<UpdateReport>`.

Produces:

- `RecipeConfig::load(path: &Path) -> Result<Self>`.
- `RecipeSchedule::lr(&self, absolute_updated_tokens: u64) -> Result<f32>`.
- `ReplayLoader::open(manifests: &[PathBuf], shares: &[f64], batch: usize, context: usize, seed: u64) -> Result<Self>`.
- `Trainer<E>::from_recipe(engine: E, config: &RecipeConfig) -> Result<Self>`.
- `Trainer<E>::begin_stage(&mut self, stage: Stage, schedule: RecipeSchedule, reset_moments: bool) -> Result<()>`.
- `Trainer<E>::engine_mut(&mut self) -> &mut E`.
- `Trainer<E>::recipe_snapshot(&self, source: &dyn RecipeSource) -> Result<TrainingSnapshot>`.
- `Trainer<E>::from_recipe_snapshot(engine: E, snapshot: TrainingSnapshot) -> Result<Self>`.

Define:

- `RecipeSource` with `next_batch(&mut self) -> Result<Option<TokenBatch>>`, `state(&self) -> RecipeLoaderState`, and `restore(&mut self, state: &RecipeLoaderState) -> Result<()>`.
- `impl RecipeSource for ReplayLoader`.
- `RecipeLoaderState { kind: LoaderKind, manifest_hash: String, children: Vec<LoaderState>, rng: RngState, order: Vec<u64>, epoch: u64, cursor: u64, consumed_targets: Vec<u64>, pending: Option<TokenBatch> }`.
- `LoaderKind::{Replay, Sft}`.
- `RecipeSchedule { origin_tokens: u64, warmup_tokens: u64, stable_tokens: u64, decay_tokens: u64, start_lr: f64, peak_lr: f64, final_lr: f64, decay: DecayShape }`.
- `DecayShape::{Cosine, Linear}`.
- `LearningRateSchedule::{LegacyCosine(TokenSchedule), Recipe(RecipeSchedule)}`.
- `TrainingLoaderState::{Token(LoaderState), Recipe(RecipeLoaderState)}`.

Extend `TrainingSnapshot` to store these schedule/loader variants, stage, recipe hash, and cumulative source exposures. Bump its implemented schema version by one; keep a tested loader for the preceding decoder schema. Preserve f32 masters, moments, pending summed gradients, RNG, and accumulation counts.

`RecipeConfig` is strict JSON with these required groups: stage, model/tokenizer/manifest identities, optimizer values, context/microbatch/effective targets, schedule, per-source shares/exposure ceilings, run budget, evaluation interval, checkpoint interval, stop rules, and parent-checkpoint identity. Stage-A parent is explicitly absent; B/C require it. Unknown fields fail.

Schedule semantics:

- Linear warmup from `start_lr` to `peak_lr`.
- Optional stable interval.
- Cosine or linear decay to `final_lr`.
- Clamp at final LR after completion.
- No silent extension of an already-decayed schedule. Continuation requires an explicit new schedule segment.
- Stage changes require an update boundary with no pending gradients. Keep moments by default; reset is an explicit recorded experiment.
- Rejected/non-finite/all-ignored updates do not advance counters or LR.

Replay selection uses a deterministic token-deficit rule against configured shares. Checkpoint child loader states, deficits/exposures, RNG, and pending batch. Do not select by document count when the recipe requests token fractions.

- [ ] Write `recipe_schedule_wsd_values`: origin 100, warmup 10, stable 20, decay 10, start 0, peak 1, final 0.1, linear decay gives LR 0, 1, 1, 0.55, 0.1 at absolute tokens 100, 110, 130, 135, 140.
- [ ] Write `recipe_schedule_legacy_cosine`: preserve decoder assertions 0, 1, 0.55, 0.1 at 0, 10, 20, 30.
- [ ] Write `recipe_schedule_mixer_resume_boundary`: equal four-target fixture batches with shares 0.75/0.25 produce counts 12/4 after four batches; save before the next selection and assert identical next source, batch, LR, exposures, and CPU update after restore.
- [ ] Write `recipe_schedule_transition_guards`: pending accumulation rejects a stage transition; recipe/hash mismatch rejects resume; a rejected update leaves token count unchanged.
- [ ] Write `recipe_schedule_strict_config`: key `"peak_lrr"` and shares totaling 0.9 return errors.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_schedule_`; record red.
- [ ] Implement the interfaces and checkpoint migration in the named files.
- [ ] Run the recipe filter, then `PKGS=lightnet-lm cargo xtask test lm_train_`, sequentially; require green.
- [ ] Commit/tag as `v0.M.NNN: add resumable recipe schedules and token replay`.

**GPU (oxide) needed:** no new kernels. Decoder Task 9 engine export/import and deterministic update are required for later device resume. Task 11 execution settings remain part of checkpoint identity when selected.

### Task R4: Latin evaluation and diagnostic suite

**Files**

- Create: `src/lightnet-lm/src/recipe_eval.rs`, `src/lightnet-lm/tests/recipe_eval.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_recipe_eval.rs`.
- Modify: `src/lightnet-lm/src/eval.rs`, `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `src/lightnet-cuda-oxide/CONTEXT.md`, `tasks/rust/llm_eval.md`.

**Interfaces**

Consumes:

- `evaluate_cpu(model: &CpuDecoder, loader: &mut TokenLoader, normalized_bytes: u64) -> decoder::Result<EvalReport>`.
- `sparse_ce(logits: &[f32], targets: &[i32], vocab: usize) -> decoder::Result<(LossSummary,Vec<f32>)>`.

Produces:

- `RecipeEngine: LmEngine` with `eval_token_nll(&mut self, batch: &TokenBatch) -> Result<Vec<f32>>`.
- Implement `RecipeEngine` for `CpuEngine` and `OxideDecoder`.
- `evaluate_recipe<E: RecipeEngine, I: Iterator<Item=Result<EvalBatch>>>(engine: &mut E, batches: I) -> Result<RecipeEvalReport>`.
- `score_choices(nll_sums: &[f64], token_counts: &[u64]) -> Result<usize>`.
- `ngram_overlap(generated: &[String], training: &[String], n: usize) -> Result<f64>`.
- `prepare_judge_pairs(candidates: &[(String,String)], seed: u64) -> Result<Vec<JudgePair>>`.

Define:

- `EvalBatch { tokens: TokenBatch, work_id: String, author_id: String, period: String, normalized_bytes: u64, original_bytes: u64 }`.
- `GroupEval { nll_sum: f64, active_tokens: u64, normalized_bytes: u64, original_bytes: u64, perplexity: f64, bits_per_byte: f64 }`.
- `RecipeEvalReport { overall: GroupEval, by_work: BTreeMap<String,GroupEval>, by_author: BTreeMap<String,GroupEval>, by_period: BTreeMap<String,GroupEval> }`.
- `JudgePair { case_id: String, left: String, right: String, left_candidate: String, token_lengths: [usize;2] }`.

Per-token loss API returns one row loss per `TokenBatch` position, zero at ignored targets, without backward, RNG advancement, or optimizer mutation. Aggregate sums/counts; never average batch means.

Record normalized-text bits-per-byte per decoder convention. For fold on/off comparisons also report NLL per **original UTF-8 byte** on identical source documents and retain the differing normalized denominators. Do not treat normalized-byte figures across destructive normalizations as directly interchangeable.

Evaluation datasets contain:

- per-work/author/period language modeling;
- Latin cloze and length-normalized forced-choice agreement;
- adapted lemma, PoS, and morphological-feature probes;
- unseen-author, cross-genre, and cross-time partitions inspired by EvaLatin;
- fixed-seed continuation and multi-turn prompts;
- anachronism, refusal, quotation, and out-of-persona cases;
- generation/training-text n-gram overlap.

Probe records store prompt, choices, correct index, author/work/period/genre, provenance, and rights. These are experimental decoder probes, not EvaLatin tagger scores.

Judge rubric: Latin grammar/morphology, coherence, persona/style, answer relevance, historical restraint, and quotation honesty, each 1–5. Randomize candidate side, blind model identity, compare length-controlled subsets, and require human Latinist review of all gate-critical cases plus at least 20% of remaining judged cases. External judge calls remain user-gated.

- [ ] Write `recipe_eval_aggregate`: losses `ln(2)` and `3*ln(2)` over 1 and 3 active targets give overall NLL/token `ln(2)` and perplexity 2, not an average of unrelated batch means.
- [ ] Write `recipe_eval_bytes_and_choices`: NLL `8*ln(2)` over 4 normalized bytes gives 2 bits/byte; choice sums `[4,3]` and counts `[2,1]` select index 0.
- [ ] Write `recipe_eval_mask_and_state`: ignored rows return exact zero; evaluation leaves exported engine state unchanged.
- [ ] Write `recipe_eval_overlap`: generated five-word sequence identical to training gives five-gram overlap 1.0; disjoint sequences give 0.0.
- [ ] Write `recipe_eval_judge_controls`: two repeated exports with seed 7 are identical; every pair retains candidate mapping and token lengths; the report rejects missing human review for a critical case.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_eval_`; record red.
- [ ] Implement CPU evaluation, grouped diagnostics, probe schemas, rubric export, and overlap reporting.
- [ ] Run the host filter; require green.
- [ ] Write `oxide_lm_recipe_eval_masked_rows`: zero-logit, two-class fixture targets `[-100,1]` returns `[0,ln(2)]` within `1e-5`; export state before/after is unchanged.
- [ ] Run `cargo xtask oxide --test oxide_lm_recipe_eval`; record red.
- [ ] Implement loss-only device evaluation and explicit per-row readback.
- [ ] Run the device suite; require actual-device green.
- [ ] Commit/tag as `v0.M.NNN: add grouped Latin recipe evaluation`.

**GPU (oxide) needed:** loss-only sparse CE and explicit per-token loss readback. Tasks 8–9 provide forward/CE foundations but not this API. With Task 11, evaluate chunked rows without materializing full logits. Readbacks occur only during explicit evaluation and are included in transfer statistics.

### Task R5: persona policy and verified quotation lookup

**Files**

- Create: `src/lightnet-lm/src/persona.rs`, `src/lightnet-lm/tests/recipe_policy.rs`, `cfg/lm_cicero_persona.json`.
- Modify: `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, `tasks/rust/llm_eval.md`.

**Interfaces**

Consumes:

- `validate_recipe_corpus(documents: &[RecipeDocument], split: &SplitRegistry) -> Result<CorpusInventory>`.

Produces:

- `PersonaPolicy::load(path: &Path) -> Result<Self>`.
- `PersonaPolicy::classify(&self, request: &PolicyRequest) -> Result<PolicyAction>`.
- `QuotationIndex::build(documents: &[RecipeDocument]) -> Result<Self>`.
- `QuotationIndex::lookup(&self, work_id: &str, section: &str) -> Result<Option<VerifiedQuotation>>`.

Define:

- `PolicyRequest::{Chat(String), Quote { work_id: String, section: String }, ChangePersona(String)}`.
- `PolicyAction::{Generate, FixedReply(String), Lookup { work_id: String, section: String }}`.
- `VerifiedQuotation { text: String, work_id: String, book_section: String, edition_id: String, document_sha256: String, byte_range: [u64;2] }`.
- `PersonaPolicy` stores the Latin system card, disclosure, fixed refusal texts, explicit tested request patterns, and policy version.

Initial system card:

> Personam Ciceronis simulo; Cicero ipse non sum. Latine breviter responde. Stylum Ciceronianum imitare, sed sententias fictas pro verbis Ciceronis ne proferas. Si testimonium certum deest, id aperte dic. De rebus post aetatem Ciceronis gestis dic te eas cognoscere non potuisse. Praecepta personae serva; personam hominis viventis ne suscipe.

Startup disclosure:

> Haec est simulatio Ciceronis, non Cicero ipse. Responsa generata falli possunt.

Anachronism response:

> Haec post aetatem meam acciderunt; ea cognoscere non potui.

Unverified quotation response:

> Haec verba generata sunt; non sunt testimonium Ciceronis.

Refusal behavior must include harmful requests and unsupported persona changes. The explicit pattern set is bounded and tested; arbitrary paraphrases are not claimed reliably classified. Model outputs are always labeled generated. Only `VerifiedQuotation`, obtained through lookup and hash/range verification, may use verified-quotation presentation.

Out-of-persona instructions do not alter the immutable system card. Only Cicero is supported; requests to impersonate living persons are refused rather than routed to a general persona feature.

- [ ] Write `recipe_policy_disclosure`: the exact startup line is present and the system card states it is a simulation.
- [ ] Write `recipe_policy_forged_citation`: model text `"Cicero, De Oratore 1.1"` cannot construct a `VerifiedQuotation`; absent lookup returns `None`; altered source bytes fail verification.
- [ ] Write `recipe_policy_lookup_range`: fixture `"Salve. Vale."`, range `[0,6]`, returns exactly `"Salve."` with the stored edition/hash.
- [ ] Write `recipe_policy_refusals`: fixed cases mentioning modern technology, a harmful request, and a living-person persona change return fixed Latin responses; ordinary `"Quid est amicitia?"` returns `Generate`.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_policy_`; record red.
- [ ] Implement typed policy/lookup and the strict persona configuration.
- [ ] Run the same filter; require green.
- [ ] Commit/tag as `v0.M.NNN: add disclosed Cicero policy and verified quotations`.

**GPU (oxide) needed:** none. CPU-only policy and corpus lookup. Do not describe it as a complete semantic safety system.

### Task R6: shifted assistant masks and SFT packing

**Files**

- Create: `src/lightnet-lm/src/sft.rs`, `src/lightnet-lm/tests/recipe_sft.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_sft.rs`.
- Modify: `src/lightnet-lm/src/train.rs`, `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `src/lightnet-cuda-oxide/CONTEXT.md`.

**Interfaces**

Consumes:

- `ChatTemplate::encode_chat(&self, tokenizer: &ByteBpe, messages: &[ChatMessage], add_generation_prompt: bool) -> Result<EncodedChat>`.
- `sparse_ce(logits: &[f32], targets: &[i32], vocab: usize) -> decoder::Result<(LossSummary,Vec<f32>)>`.
- R3 `RecipeSource`.

Produces:

- `build_sft_batch(examples: &[EncodedChat], batch: usize, context: usize) -> Result<TokenBatch>`.
- `SftLoader::open(manifest: &Path, batch: usize, context: usize, seed: u64) -> Result<Self>`.
- `impl RecipeSource for SftLoader`.
- Optional weighted extension: `weighted_sparse_ce(logits: &[f32], targets: &[i32], weights: &[f32], vocab: usize) -> Result<(LossSummary,Vec<f32>)>`.
- `LmEngine::loss_sum_backward_weighted(&mut self, batch: &TokenBatch, weights: &[f32]) -> decoder::Result<LossSummary>`.
- `Trainer<E>::microbatch_weighted(&mut self, batch: &TokenBatch, weights: &[f32]) -> decoder::Result<LossSummary>`.

Default mask rule: at input row `j`, supervise token `input_ids[j+1]` only when `assistant_mask[j+1]` is true. The last row has no next target. Assistant EOT is active; assistant opener, system, user, and padding are ignored.

Different conversations have distinct segment IDs. All turns within one conversation share its segment. Never insert EOS→next-example targets. Reject an overlong example during preparation; do not silently cut an assistant answer. Preparation may explicitly create shorter provenance-linked examples.

Optional prompt-weight mode uses 0.05 for prompt targets and 1.0 for assistant targets; ignored/padded rows still have weight zero. Sum weighted losses and seeds, then normalize once by the count of positive-weight active targets. Record this denominator and objective identity; do not compare its training loss directly with assistant-only loss. Weight-zero rows do not advance active-target counts.

- [ ] Write `recipe_sft_shifted_mask`: R2’s ten-ID fixture produces nine shifted targets `[-100,-100,-100,-100,-100,-100,-100,69,8191]`.
- [ ] Write `recipe_sft_zero_masked_gradient`: logits `[0,0,0,0]`, targets `[-100,1]`, vocabulary 2 give loss `ln(2)`, active count 1, seeds `[0,0,-0.5,0.5]`.
- [ ] Write `recipe_sft_pack_isolation`: modifying every token of packed example two leaves example-one logits unchanged within the decoder CPU tolerance.
- [ ] Write `recipe_sft_resume`: saved example order/cursor and pending row reproduce the next batch exactly; all-ignored examples do not advance an update.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_sft_`; record red.
- [ ] Implement assistant-only batching, strict overflow rejection, and resumable SFT loading.
- [ ] Run the host filter; require green.
- [ ] Write `recipe_sft_prompt_weight`: two zero-logit binary rows, targets `[1,1]`, weights `[0.05,1]` give sum `1.05*ln(2)`, active count 2, seeds `[-0.025,0.025,-0.5,0.5]`.
- [ ] Run the host filter; record the weighted API’s red result.
- [ ] Implement the optional CPU weighted path and explicit unsupported error for engines without it.
- [ ] Run the host filter; require green.
- [ ] Write device tests `oxide_lm_sft_mask_and_segments` and `oxide_lm_sft_weighted`: assert the same ignored rows/weights, segment isolation, and CPU/device gradients within decoder Task 9’s 0.08 normalized maximum error.
- [ ] Run `cargo xtask oxide --test oxide_lm_sft`; record red.
- [ ] Implement or wire device masking/weighting without host gradient fallback.
- [ ] Run the device suite, then `cargo xtask oxide --test oxide_lm_train`, sequentially; require green.
- [ ] Commit/tag as `v0.M.NNN: add assistant-masked packed SFT training`.

**GPU (oxide) needed:** existing Task 9 ignored-target sparse CE and Task 8 packed segment attention; Task 11 must preserve both when selected. Only optional weighted CE is a new objective path. Masking does not require changing `TokenBatch`.

### Task R7: real and synthetic SFT data preparation

**Files**

- Create: `src/lightnet-lm/src/sft_data.rs`, `src/lightnet-lm/tests/recipe_sft_data.rs`.
- Modify: `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, `src/lightnet-cli/src/lm.rs`, `src/lightnet-cli/src/lm_recipe.rs`, `tasks/rust/llm_data.md`.

**Interfaces**

Consumes:

- `reject_holdout_overlap(text: &str, split: &SplitRegistry) -> Result<()>`.
- R2 `ChatMessage`; R5 persona policy; R6 batch preparation.

Produces:

- `extract_dialogue(document: &RecipeDocument, passages: &[SourceTurn], split: &SplitRegistry) -> Result<Vec<SftExample>>`.
- `export_synthetic_requests(specs: &[SyntheticRequest], split: &SplitRegistry, out: &Path) -> Result<()>`.
- `import_synthetic_responses(path: &Path, split: &SplitRegistry) -> Result<Vec<SftExample>>`.
- `filter_sft(examples: &[SftExample], policy: &SftFilterPolicy) -> Result<SftFilterReport>`.
- `write_sft_manifest(examples: &[SftExample], tokenizer: &ByteBpe, out: &Path) -> Result<()>`.

Define:

- `SourceTurn { speaker: String, text: String, book_section: String, byte_range: [u64;2] }`.
- `SftExample { id: String, messages: Vec<ChatMessage>, provenance: SftProvenance }`.
- `SftProvenance { kind: SftKind, source_work_ids: Vec<String>, sections: Vec<String>, editions: Vec<String>, licenses: Vec<String>, source_hashes: Vec<String>, generator: Option<GeneratorProvenance> }`.
- `SftKind::{DialogueAdaptation, LetterAdaptation, RuleTemplate, ExternalSynthetic}`.
- `GeneratorProvenance { name: String, revision: String, prompt: String, date: String, terms_url: String, terms_status: RightsStatus, response_hash: String }`.
- `SyntheticRequest { id: String, prompt: String, source_work_ids: Vec<String> }`.
- `SftFilterPolicy { max_tokens: usize, min_latin_score: f64, max_duplicate_jaccard: f64 }`.
- `SftFilterReport { accepted: Vec<SftExample>, rejected: BTreeMap<String,Vec<String>> }`.

Use edition-specific, explicit speaker/section exports; do not infer turns from punctuation alone. Preserve work, book/section, edition, rights, and source ranges. Mark letter-derived conversations as adaptations: an invented interlocutor prompt is not an original Cicero exchange. Do not attribute every dialogue speaker’s words directly to Cicero.

Required source coverage is audited for `De Oratore`, `Tusculanae Disputationes`, `De Re Publica`, `Laelius de Amicitia`, `Cato Maior de Senectute`, and `Epistulae`. The selected held-out works are evaluation-only, never training examples.

Latin-quality filtering combines supplied Latin word-list/function-word coverage, script checks, markup/translation rejection, length limits, and human review. Thresholds are development-set parameters; heuristics are not a Latin correctness proof.

The synthetic harness exports requests and imports responses; it does not silently call a remote model. Templates/rules work offline. Any stronger-generator call is an external user-authorized action with recorded terms. Unresolved generator terms block training admission.

- [ ] Write `recipe_sft_data_source_ranges`: two fixture speaker turns retain exact `"1.2"` sections and source ranges; missing edition/license is rejected.
- [ ] Write `recipe_sft_data_letter_label`: letter adaptations retain `LetterAdaptation` and label the prompt as constructed.
- [ ] Write `recipe_sft_synthetic_holdout_leakage`: a request naming `cicero-laelius`, an alternate-edition excerpt, and a copied held-out five-word fixture are all rejected.
- [ ] Write `recipe_sft_data_terms`: absent generator/prompt/date/terms evidence or unresolved terms rejects an external example.
- [ ] Write `recipe_sft_data_filters`: identical examples yield one accepted item; `"the and of"` fails the authored Latin filter fixture while `"amicitia virtutem sequitur"` passes its fixture threshold.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_sft_data_`; record red.
- [ ] Implement preparation/filtering and proposed `lm sft-prepare`, `lm synth-export`, and `lm synth-import` parsing.
- [ ] Run the host filter, then `PKGS=lightnet-cli cargo xtask test lm_recipe_sft_`, sequentially; require green.
- [ ] Commit/tag as `v0.M.NNN: add provenance-preserving Latin SFT preparation`.

**GPU (oxide) needed:** none. CPU-only extraction, filters, hashing, template serialization, and manifest creation.

**USER-GATED data recipe:** corpus ingestion and generator use are not part of the unit-test cycle. Proposed command shapes:

```text
cargo xtask lm sft-prepare --corpus "$CORPUS_MANIFEST" --splits "$SPLITS" --tokenizer "$TOKENIZER" --persona cfg/lm_cicero_persona.json --output "$SFT_DIR"
cargo xtask lm synth-export --requests "$REQUESTS" --splits "$SPLITS" --output "$GENERATOR_INPUT"
cargo xtask lm synth-import --responses "$GENERATOR_OUTPUT" --splits "$SPLITS" --output "$SYNTHETIC_DIR"
```

The external generator invocation is deliberately not prescribed before provider/model/terms authorization. Record request/response hashes, model revision, prompt, date, terms URL/status, filtering decisions, and human Latin review.

### Task R8: bounded run orchestration and stage gates

**Files**

- Create: `src/lightnet-lm/src/run.rs`, `src/lightnet-lm/tests/recipe_run.rs`, `src/lightnet-cli/src/lm_recipe.rs`, `tools/lm_recipe.sh`, `tasks/rust/llm_recipe_evidence.md`.
- Modify: `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, `src/lightnet-cli/src/lm.rs`, `src/lightnet-cli/CONTEXT.md`, `xtask/src/main.rs`, `CLAUDE.md`, `tools/CONTEXT.md`.

If R7 implementation precedes this task, create `lm_recipe.rs` in R7 and modify it here; do not duplicate the module.

**Interfaces**

Consumes:

- R3 trainer/snapshot/source APIs.
- `evaluate_recipe<E,I>(engine: &mut E, batches: I) -> Result<RecipeEvalReport>`.
- `save_checkpoint(path: &Path, state: &TrainingSnapshot) -> decoder::Result<()>`.

Produces:

- `run_stage<E: RecipeEngine, S: RecipeSource>(engine: E, source: S, spec: &StageRunSpec, out: &Path) -> Result<RunEvidence>`.
- `evaluate_gate(kind: GateKind, evidence: &RunEvidence) -> Result<GateDecision>`.
- `parse_recipe_command(args: &[String]) -> Result<RecipeCommand>`.

Define:

- `GateKind::{G0, AtoB, BtoC, Chat, Preference}`.
- `GateDecision { passed: bool, failures: Vec<String> }`.
- `StageRunSpec { stage: Stage, config: RecipeConfig, inventory: CorpusInventory, parent_checkpoint: Option<PathBuf>, approval_record: PathBuf }`.
- `RunEvidence` contains run/config/tokenizer/manifest/checkpoint hashes, stage, successful tokens/updates, source exposures, evaluation series, stop reason, device/precision/memory, cap status, throughput window, gate measurements, and human-review references.
- `RecipeCommand` strictly represents the proposed `recipe-run`, `recipe-check`, `sft-train`, `sft-prepare`, synthetic, and evaluation subcommands.
- `StopReason::{Budget, Exposure, Plateau, Overfit, NonFinite, UserInterrupted}`.

Record both total wall time and steady-state successful targets/second. Exclude build, initialization, checkpoint, and evaluation intervals from steady-state throughput; also report end-to-end throughput. Do not invent speeds.

`tools/lm_recipe.sh` only selects the existing CPU or oxide codegen route and applies the existing cap. `cargo xtask lm ...` is the only public invocation. Preserve the decoder forwarder if already implemented.

Authorization records identify the user request, allowed data/actions, stage, and budget. They document authorization; creating a file is not itself permission.

- [ ] Write `recipe_run_cadence`: a fake engine with four active targets/update, budget 12, eval interval 4 records evaluations at 4, 8, 12 and exactly three successful updates.
- [ ] Write `recipe_run_rejected_update`: one rejected update adds zero tokens and triggers no false budget completion.
- [ ] Write `recipe_run_stops`: authored sequences exercise four-evaluation plateau, three-evaluation gap alarm, and immediate non-finite stop.
- [ ] Write `recipe_run_gate_failures`: forgetting 0.051 fails B→C; 0.050 passes that criterion; missing human review fails chat acceptance.
- [ ] Write CLI tests `lm_recipe_strict_args` and `lm_recipe_cap_plan`: unknown flags fail; forwarded build jobs never exceed 4; cap fallback is represented as uncapped evidence.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_run_`; record red.
- [ ] Implement run orchestration and typed gate evaluation using a fake-engine unit-test path.
- [ ] Run the host filter; require green.
- [ ] Run `PKGS=lightnet-cli cargo xtask test lm_recipe_`; record red for the new CLI cases.
- [ ] Implement thin CLI parsing and capped forwarding.
- [ ] Run the CLI filter, then `cargo xtask check lightnet-cli`, sequentially; require green.
- [ ] Commit/tag as `v0.M.NNN: add gated Latin recipe orchestration`.

**GPU (oxide) needed:** no new kernels. Actual oxide runs require decoder Tasks 8–9, R4 evaluation, R6 for SFT, and Task 11 when selected. Fake-engine tests are CPU-only.

### Task R9: stage-A recipe and user-gated pretraining

**Files**

- Create: `cfg/lm_latin_a.json`, `src/lightnet-lm/tests/recipe_stages.rs`.
- Modify: `src/lightnet-lm/src/recipe.rs`, `tasks/rust/llm_recipe_evidence.md`, `tasks/rust/llm_data.md`.

**Interfaces**

Consumes:

- `RecipeConfig::load(path: &Path) -> Result<Self>`.
- `measure_shards(manifest: &ShardManifest, documents: &[RecipeDocument]) -> Result<CorpusInventory>`.

Produces:

- `stage_a_budget(inventory: &CorpusInventory, shares: &BTreeMap<String,f64>, max_exposures: &BTreeMap<String,f64>, user_ceiling: u64) -> Result<u64>`.
- `stage_a_spec(config: &RecipeConfig, inventory: &CorpusInventory, approval_record: &Path) -> Result<StageRunSpec>`.

The checked-in configuration contains the proposed mixture/optimizer defaults, explicit experiment labels, and required-input declarations. It must not contain fabricated counts or a runnable guessed budget. A missing measured inventory or approval record fails recipe validation.

- [ ] Write `recipe_stage_a_budget`: source counts 100 and 50, shares 0.75/0.25, exposure ceilings 4/4, user ceiling 1000 yield budget 533; user ceiling 400 yields 400.
- [ ] Write `recipe_stage_a_caps`: Vulgate share 0.06 and CC100 share 0.11 each fail; a positive-share empty source fails.
- [ ] Write `recipe_stage_a_identity`: total vocabularies 8192/16384 retain parameter counts 45,628,928/49,823,232 and include reserved chat controls.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_stage_a_`; record red.
- [ ] Implement the pure budget/spec builders and strict configuration.
- [ ] Run the same filter; require green.
- [ ] Commit/tag as `v0.M.NNN: specify measured-budget Latin pretraining`.

**GPU (oxide) needed:** none for recipe tests. The run requires resident bf16 forward/backward/update from Tasks 8–9, R4 evaluation, and Task 11 if memory fit requires it.

**USER-GATED run recipe**

Before execution, obtain explicit authorization for corpus preparation, tokenizer fitting, each short sweep, and the selected main run. Supply measured inventory and an external output directory.

Proposed command shapes:

```text
cargo xtask lm recipe-check --stage a --recipe cfg/lm_latin_a.json --inventory "$INVENTORY" --tokenizer "$TOKENIZER" --approval "$APPROVAL"
cargo xtask lm recipe-run --stage a --recipe cfg/lm_latin_a.json --inventory "$INVENTORY" --tokenizer "$TOKENIZER" --approval "$APPROVAL" --backend oxide --precision bf16 --output "$A_RUN_DIR"
```

Bounded sweep default: at most 200 successful updates and at most `min(1_000_000, floor(B_A/20))` active targets per candidate, whichever stops first. These bounds are proposed defaults requiring authorization. Use the same documents, seeds, budget, and evaluation points; if fewer than five updates fit, report insufficient pilot evidence.

Expected artifacts: resolved recipe, tokenizer artifact, corpus/shard manifests, checkpoint plus loader/optimizer state, evaluation JSON, generations, exposure report, timing/memory report, and compact G0/A evidence.

Record failures and early stops. Do not claim an optimum from an inconclusive sweep or promote a checkpoint with unresolved leakage/regurgitation alerts.

### Task R10: stage-B recipe and user-gated continuation

**Files**

- Create: `cfg/lm_latin_b.json`.
- Modify: `src/lightnet-lm/src/recipe.rs`, `src/lightnet-lm/tests/recipe_stages.rs`, `tasks/rust/llm_recipe_evidence.md`.

**Interfaces**

Consumes:

- `RecipeSchedule::lr(&self, absolute_updated_tokens: u64) -> Result<f32>`.
- `stage_a_budget(...) -> Result<u64>` from R9.
- `evaluate_gate(kind: GateKind, evidence: &RunEvidence) -> Result<GateDecision>`.

Produces:

- `stage_b_spec(config: &RecipeConfig, inventory: &CorpusInventory, parent: &RunEvidence, approval_record: &Path) -> Result<StageRunSpec>`.
- `forgetting_delta(before: &RecipeEvalReport, after: &RecipeEvalReport) -> Result<f64>`.

Reuse the budget calculation with remaining exposure capacity rather than resetting repeated-data accounting. Parent tokenizer/config identity must match. Default is preserved AdamW moments and a new explicit schedule origin at the parent’s successful-update token count.

- [ ] Write `recipe_stage_b_forgetting`: before NLL/token 2.00 and after 2.04 give `0.04` within `1e-12`; missing/mismatched validation identity errors.
- [ ] Write `recipe_stage_b_parent_and_holdouts`: changing tokenizer hash or admitting `cicero-laelius`/`cicero-ad-atticum` rejects the spec.
- [ ] Write `recipe_stage_b_rewarm`: parent count 100 and peak LR 0.001, selected factor 0.1, create peak 0.0001 with origin 100; absolute counters do not reset.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_stage_b_`; record red.
- [ ] Implement stage-B validation/configuration and forgetting calculation.
- [ ] Run the same filter; require green.
- [ ] Commit/tag as `v0.M.NNN: specify replay-controlled Cicero continuation`.

**GPU (oxide) needed:** no new kernels. The run reuses Tasks 8–9, R3 resume/mixer, R4 evaluation, and Task 11 where needed.

**USER-GATED run recipe**

Authorize replay/LR ablations before the selected continuation. Proposed command:

```text
cargo xtask lm recipe-run --stage b --recipe cfg/lm_latin_b.json --inventory "$B_INVENTORY" --tokenizer "$TOKENIZER" --resume "$A_CHECKPOINT" --approval "$B_APPROVAL" --backend oxide --precision bf16 --output "$B_RUN_DIR"
```

Expected artifacts: resolved replay fractions and schedule, parent hash, cumulative exposures, B checkpoint/snapshot, Cicero/A validation reports, fixed generations, and GBC evidence.

Record the complete replay sweep, including forgetting. If the forgetting criterion fails, increase replay or reduce re-warm/continuation budget in another authorized experiment; do not cherry-pick a single improved Cicero loss while hiding broad-Latin regression.

### Task R11: stage-C recipe and user-gated SFT

**Files**

- Create: `cfg/lm_latin_sft.json`.
- Modify: `src/lightnet-lm/src/recipe.rs`, `src/lightnet-lm/tests/recipe_stages.rs`, `src/lightnet-cli/src/lm_recipe.rs`, `tasks/rust/llm_recipe_evidence.md`.

**Interfaces**

Consumes:

- `SftLoader::open(manifest: &Path, batch: usize, context: usize, seed: u64) -> Result<Self>`.
- `stage_b_spec(...) -> Result<StageRunSpec>`.
- R6 default/optional weighted training interfaces.

Produces:

- `stage_c_spec(config: &RecipeConfig, sft_manifest: &Path, parent: &RunEvidence, approval_record: &Path) -> Result<StageRunSpec>`.

Proposed **UNVERIFIED** defaults:

- assistant-only loss;
- LR candidates `1e-5`, `3e-5`, `1e-4`;
- warmup 3% of successful assistant-target budget;
- 1, 2, or 3 epochs, with early selection before the authorized ceiling;
- effective active targets/update 8192 initially;
- matrix decay 0 or 0.01 as an ablation;
- clip 1.0; bf16/f32 storage as before;
- validation each tenth of an epoch, rounded to complete updates;
- stop after three evaluations without assistant-NLL improvement and retain the best rubric-compatible checkpoint.

Epoch and active-target counts are both recorded. Prefer the smaller LR/earlier checkpoint when quality is indistinguishable. Prompt-weight 0.05 is a separate matched ablation, never silently enabled.

- [ ] Write `recipe_stage_c_requires_chat_artifact`: version-1 tokenizer or mismatched chat-template hash rejects the run.
- [ ] Write `recipe_stage_c_counts`: fixture with 10 input tokens and two assistant targets advances the default schedule by 2, not 10.
- [ ] Write `recipe_stage_c_no_empty_update`: all-prompt/no-assistant examples cannot advance optimizer or LR.
- [ ] Write `recipe_stage_c_stop_selection`: fixture validation losses `[2.0,1.9,1.91,1.92,1.93]` retain evaluation two and stop after the third subsequent non-improvement.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_stage_c_`; record red.
- [ ] Implement strict C spec construction and proposed `lm sft-train` delegation.
- [ ] Run the host filter, then `PKGS=lightnet-cli cargo xtask test lm_recipe_sft_`, sequentially; require green.
- [ ] Commit/tag as `v0.M.NNN: specify assistant-masked Cicero SFT`.

**GPU (oxide) needed:** Task 9 masked CE/backward/update, R6 packed SFT and optional weighted CE, R4 evaluation, and Task 11 where needed. No preference objective is implied.

**USER-GATED run recipe**

```text
cargo xtask lm sft-train --recipe cfg/lm_latin_sft.json --sft-manifest "$SFT_MANIFEST" --tokenizer "$TOKENIZER" --resume "$B_CHECKPOINT" --approval "$C_APPROVAL" --backend oxide --precision bf16 --output "$C_RUN_DIR"
```

Expected artifacts: SFT manifest/provenance/filter hashes, template/persona version, selected objective, parent hash, resumable checkpoint, assistant-validation curves, fixed chat generations, blind rubric export, human Latinist review, refusal/quotation/overlap reports, and GCHAT evidence.

Do not continue epochs merely because training loss decreases when held-out responses, Latin quality, or regurgitation worsen.

### Task R12: CPU chat session and CLI

**Files**

- Modify: `src/lightnet-lm/src/chat.rs`, `src/lightnet-lm/tests/recipe_chat.rs`, `src/lightnet-cli/src/lm.rs`, `src/lightnet-cli/src/lm_recipe.rs`, `src/lightnet-cli/CONTEXT.md`, `tasks/rust/llm_eval.md`.

**Interfaces**

Consumes:

- `CpuDecoder::prefill(&self, ids: &[u32], cache: &mut KvCache) -> decoder::Result<Vec<f32>>`.
- `CpuDecoder::decode(&self, ids: &[u32], cache: &mut KvCache) -> decoder::Result<Vec<f32>>`.
- `sample_token(logits: &[f32], options: &SamplingOptions, rng: &mut RngState) -> decoder::Result<u32>`.
- R2 template and R5 policy/quotation interfaces.

Produces:

- `ChatSession::new(tokenizer: ByteBpe, template: ChatTemplate, policy: PersonaPolicy, context: usize) -> Result<Self>`.
- `ChatSession::prepare_turn(&mut self, user: &str, max_new_tokens: usize) -> Result<PreparedTurn>`.
- `ChatSession::finish_turn(&mut self, generated: &[u32], reason: FinishReason) -> Result<()>`.
- `run_cpu_chat(model: &CpuDecoder, session: &mut ChatSession, options: &ChatOptions, input: &mut dyn BufRead, output: &mut dyn Write) -> Result<()>`.

Define:

- `ChatOptions { sampling: SamplingOptions, max_new_tokens: usize, stop_tokens: Vec<u32>, transcript_path: Option<PathBuf> }`.
- `PreparedTurn { retained_ids: Vec<u32>, append_ids: Vec<u32>, rebuild_cache: bool }`.
- `FinishReason::{EndTurn, Eos, TokenLimit}`.

Behavior:

- Interactive multi-turn history; print the Latin disclosure at startup.
- Reuse the existing cache only when retained history is an exact prefix extension.
- Reserve room for `max_new_tokens`.
- On overflow, keep the system message and newest user turn, drop oldest complete user/assistant pairs, reserialize retained history, reset and prefill the cache.
- If system plus newest user plus generation allowance cannot fit, error without mutating history/cache.
- On a token-limit finish, serialize a structural EOT before a later turn; it must enter the cache before the next user message.
- EOS and EOT stop output and are not printed as token spellings.
- Prevent sampling PAD/BOS/UNK, system/user/assistant controls, and undefined tokenizer IDs. Permit EOT and EOS. Apply this allowlist before calling existing sampling.
- No repetition penalty initially. An unknown repetition option is an error; add it only with a separate quality/regression test.
- Transcripts are off by default. Explicit logging records the disclosure, policy/template/model hashes, seed, sampling settings, finish reason, and turns to the user-selected path.
- `/quote WORK SECTION` uses verified lookup. Other generated text remains labeled generated.
- `/reset` clears conversation/cache/RNG according to the recorded seed; `/quit` exits.

- [ ] Write `recipe_chat_truncation_rebuild`: a capacity-16 token fixture with system length 3, two completed pairs of length 4 each, newest user/opener length 3, and generation allowance 4 drops the oldest pair; retained prompt length is 10 and `rebuild_cache=true`.
- [ ] Write `recipe_chat_oversized_turn_atomic`: system length 3 plus newest turn length 10 plus allowance 4 exceeds 16 and leaves history/cache unchanged.
- [ ] Write `recipe_chat_append_cache_agreement`: cached two-turn logits match complete re-prefill within `1e-5 + 1e-5*abs(reference)`.
- [ ] Write `recipe_chat_stop_and_allowlist`: larger logits on forbidden controls never select them; EOT/EOS emit zero visible marker text; a token-limit turn appends one structural EOT.
- [ ] Write `recipe_chat_cli_seed_and_logging`: seed 7 reproduces output with the scripted backend; logging is absent by default; unknown flags fail.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_chat_`; record new session tests red.
- [ ] Implement history/cache planning and CPU generation loop.
- [ ] Run the host filter; require green.
- [ ] Run `PKGS=lightnet-cli cargo xtask test lm_recipe_chat_`; record red.
- [ ] Implement proposed `lm chat` parsing and delegation.
- [ ] Run the CLI filter; require green.
- [ ] Commit/tag as `v0.M.NNN: add disclosed multi-turn Cicero chat`.

**GPU (oxide) needed:** none for CPU chat. Decoder Task 7 supplies CPU KV cache and sampling. Device chat is R13, not an assumed feature of Task 8.

Proposed user-invoked command:

```text
cargo xtask lm chat --checkpoint "$C_CHECKPOINT" --tokenizer "$TOKENIZER" --persona cfg/lm_cicero_persona.json --backend cpu --temperature 0.7 --top-p 0.9 --top-k 40 --seed 7 --max-new-tokens 128
```

Sampling values are user-facing defaults, **UNVERIFIED for output quality**. Stop tokens default to artifact EOT plus EOS.

### Task R13: oxide KV-cache chat integration

**Files**

- Create: `src/lightnet-cuda-oxide/src/lm_chat.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_chat.rs`.
- Modify: `src/lightnet-cuda-oxide/src/lib.rs`, `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `src/lightnet-cuda-oxide/CONTEXT.md`, `src/lightnet-cli/src/lm_recipe.rs`.

**Interfaces**

Consumes:

- Decoder Task 8 `OxideDecoder` and R12 `PreparedTurn`/chat lifecycle.

Produces:

- `OxideKvCache::new(model: &OxideDecoder, context: usize) -> Result<Self>`.
- `OxideKvCache::reset(&mut self)`.
- `OxideDecoder::prefill_chat(&mut self, ids: &[u32], cache: &mut OxideKvCache) -> Result<Vec<f32>>`.
- `OxideDecoder::decode_chat(&mut self, id: u32, cache: &mut OxideKvCache) -> Result<Vec<f32>>`.

Cache stores rotated K/unrotated V on device with length/model revision. Validate before writes. Only the next-token vocabulary logits are explicitly downloaded for CPU sampling; KV tensors and intermediate activations remain on device. Count this readback and do not claim transfer-free generation.

- [ ] Write `oxide_lm_chat_prefix_agreement`: MHA/GQA prefill and at least three decode steps agree with bf16 reference logits within normalized maximum error 0.05.
- [ ] Write `oxide_lm_chat_overflow_atomic`: overflow, stale revision, and invalid ID leave device cache length/content unchanged.
- [ ] Write `oxide_lm_chat_truncation`: R12’s retained-history rebuild produces the same logits as a fresh device cache.
- [ ] Write `oxide_lm_chat_residency`: decode performs no KV/activation readback; explicit next-token-logit readback is counted.
- [ ] Run `cargo xtask oxide --test oxide_lm_chat`; record red.
- [ ] Implement native cache writes and incremental decoder execution.
- [ ] Run the device suite, then `cargo xtask oxide --test oxide_lm_forward`, sequentially; require green.
- [ ] Commit/tag as `v0.M.NNN: add device KV-cache Cicero chat`.

**GPU (oxide) needed:** new device cache allocation/writes, nonzero-position RoPE, cached causal GQA, and incremental projections/head. Task 8 supplies native primitives but explicitly leaves this integration open. Task 9 supplies checkpoint parameters; Task 11 is not automatically required for batch-one short-context decode.

The R12 command may use `--backend oxide` only after this task is green. Unsupported device chat errors rather than falling back to CPU.

### Task R14: optional preference tuning

**Files**

- Create: `src/lightnet-lm/src/preference.rs`, `src/lightnet-lm/tests/recipe_preference.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_preference.rs`.
- Modify: `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/src/train.rs`, `src/lightnet-lm/CONTEXT.md`, `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `src/lightnet-cli/src/lm_recipe.rs`, `tasks/rust/llm_recipe_evidence.md`.

This task is optional and does not block chat. Implement DPO only. ORPO is reference-free and KTO accepts binary labels, but selecting either requires another objective specification; do not expose unsupported names.

**Interfaces**

Consumes:

- Accepted SFT checkpoint, R6 assistant masks, R4 paired judging, and R8 gates.

Produces:

- `cache_reference_logprobs<E: RecipeEngine>(reference: &mut E, pairs: &[PreferencePair], identity: &ReferenceIdentity) -> Result<ReferenceCache>`.
- `dpo_loss(policy_logps: &[[f64;2]], reference_logps: &[[f64;2]], beta: f64) -> Result<(f64,Vec<[f64;2]>)>`.
- `PreferenceEngine: RecipeEngine` with `dpo_backward(&mut self, batch: &PreferenceBatch, reference: &ReferenceCache, beta: f64) -> Result<LossSummary>`.
- Implement `PreferenceEngine` for CPU and oxide engines.

Define:

- `PreferencePair { id: String, chosen: Vec<ChatMessage>, rejected: Vec<ChatMessage>, provenance: String }`.
- `ReferenceIdentity { checkpoint_hash: String, tokenizer_hash: String, template_hash: String, dataset_hash: String, objective_version: u32 }`.
- `ReferenceCache { identity: ReferenceIdentity, logps: BTreeMap<String,[f64;2]> }`.
- `PreferenceBatch { pairs: Vec<PreferencePair>, active_targets: u64 }`.

Completion log probabilities sum over assistant content/EOT, not prompts or padding. DPO uses:

\[
z=\beta[(\log\pi(y_w|x)-\log\pi(y_l|x))
-(\log\pi_{\rm ref}(y_w|x)-\log\pi_{\rm ref}(y_l|x))],
\quad L=-\log\sigma(z).
\]

`dpo_loss` returns summed pair loss and descent seeds with respect to chosen/rejected sequence log probabilities. Engine integration scales accumulated pair-mean gradients into the trainer’s active-target normalization contract exactly once; test equivalence across unequal pair/token microbatches.

Cache reference log probabilities to avoid keeping a second model resident. This still requires an authorized reference forward pass and cache storage. Reject cache identity mismatches. No preference run may use the policy itself as a moving reference.

Proposed beta sweep 0.05/0.1 and one epoch are **UNVERIFIED**. Watch for degenerate short replies and length-driven wins.

- [ ] Write `recipe_preference_dpo_sign`: equal policy/reference log probabilities and beta 0.1 give loss `ln(2)` and sequence descent seeds `[0.05,-0.05]`.
- [ ] Write `recipe_preference_reference_identity`: changing checkpoint, tokenizer, template, or dataset hash rejects the cache.
- [ ] Write `recipe_preference_accumulation`: one batch versus unequal-token microbatches produces the same pair-mean update within CPU `1e-6`.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_preference_`; record red.
- [ ] Implement CPU DPO/reference caching and objective-specific trainer integration.
- [ ] Run the host filter; require green.
- [ ] Write `oxide_lm_preference_gradient`: CPU/device sequence scores and parameter gradients agree within Task 9’s 0.08 normalized maximum error; prompts have zero objective seed.
- [ ] Run `cargo xtask oxide --test oxide_lm_preference`; record red.
- [ ] Implement device sequence-logprob reduction and supplied-sequence-seed backward without host gradient fallback.
- [ ] Run the device suite; require green.
- [ ] Commit/tag as `v0.M.NNN: add optional cached-reference Latin DPO`.

**GPU (oxide) needed:** new assistant sequence-logprob reduction and DPO seed backward. Tasks 8–9 provide decoder forward/backward but sparse CE alone is not a DPO implementation. Task 11’s chunked head must support the objective when selected.

**USER-GATED run recipe**

```text
cargo xtask lm preference-cache --reference "$C_CHECKPOINT" --pairs "$PAIRS" --tokenizer "$TOKENIZER" --approval "$PREFERENCE_APPROVAL" --output "$REFERENCE_CACHE"
cargo xtask lm preference-train --objective dpo --resume "$C_CHECKPOINT" --pairs "$PAIRS" --reference-cache "$REFERENCE_CACHE" --approval "$PREFERENCE_APPROVAL" --backend oxide --precision bf16 --output "$PREFERENCE_RUN_DIR"
```

Record label provenance, annotator/judge identities, cache hashes, reference cost, objective settings, response lengths, paired human judgments, and the preference gate. Retain SFT if the win is absent or inconclusive. Effectiveness at this scale remains unverified until measured.

### Task R15: final acceptance evidence and model card

**Files**

- Create: `tasks/rust/llm_cicero_model_card.md`.
- Modify: `src/lightnet-lm/src/run.rs`, `src/lightnet-lm/tests/recipe_run.rs`, `tasks/rust/llm_recipe_evidence.md`, `tasks/rust/llm_eval.md`, `tasks/rust/plan_llm_training_recipe.md`.

**Interfaces**

Consumes:

- `evaluate_gate(kind: GateKind, evidence: &RunEvidence) -> Result<GateDecision>`.
- R4 evaluation, R5 policy, R12/R13 cache evidence.

Produces:

- `validate_release_evidence(evidence: &RunEvidence, card: &ModelCardRecord) -> Result<GateDecision>`.
- `ModelCardRecord { checkpoint_hash: String, tokenizer_hash: String, template_hash: String, persona_version: String, training_evidence: Vec<String>, rights_summary: String, evaluation_summary: String, limitations: Vec<String>, retained_artifacts: Vec<PathBuf>, user_acceptance_reference: String }`.

The model card contains a TOC, exact architecture/parameter count, training-stage identities, corpus/licensing summary, tokenizer normalization/control IDs, exposure budgets, hardware/cap status, evaluation and human-review evidence, sampling defaults, disclosure, known failures, and intended Latin-only use.

List failed prompts and uncertainty. State that generations may be wrong, memorized, anachronistic, or linguistically defective. Distinguish verified quotations from generated text.

- [ ] Write `recipe_run_release_evidence`: missing human Latin review, missing work-level split evidence, absent device-cache tests for an oxide release, or mismatched hashes each fails acceptance.
- [ ] Write `recipe_run_release_no_fabricated_results`: an unmeasured quality field cannot satisfy a threshold; CPU-only evidence cannot satisfy an oxide claim.
- [ ] Run `PKGS=lightnet-lm cargo xtask test recipe_run_release_`; record red.
- [ ] Implement release-evidence validation and a model-card template with explicit unmeasured fields.
- [ ] Run the same filter; require green.
- [ ] Commit/tag the validator as `v0.M.NNN: validate Cicero release evidence`; commit later evidence-only updates as `docs(rust): record Cicero recipe acceptance`.

**GPU (oxide) needed:** none for evidence validation. An oxide model-card claim requires actual green evidence from Tasks 8–9 and R13, plus Task 11 if selected.

**USER-GATED final checklist**

- [ ] Obtain authorization for final checkpoint evaluation and interactive acceptance.
- [ ] Execute the frozen evaluation set once for the selected candidate.
- [ ] Obtain the Latinist’s rubric and quotation/hallucination review.
- [ ] Evaluate GCHAT and record the user’s acceptance or rejection.
- [ ] Record the retained checkpoint and supporting artifact hashes outside the repository.
- [ ] Record compact timing, memory, losses, generations, and failure evidence.
- [ ] Delete disposable large checkpoints/logs only after retained artifacts and evidence are verified.
- [ ] Commit the evidence/model-card documentation without claiming unresolved gates passed.

## Progress table

| Task | Deliverable | Draft status | Completion evidence |
|---|---|---|---|
| R1 | Provenance and whole-work registry | Planned | Rights, inventory, aliases, held-out passage tests |
| R2 | Chat tokenizer v2 and template | Planned | Exact IDs, literal safety, serialization |
| R3 | Schedules/replay/resume | Planned | LR values, boundary resume, strict config |
| R4 | Latin evaluation | Planned | Grouped math, probes, judge controls, device loss readback |
| R5 | Persona/quotation policy | Planned | Disclosure, verified ranges, forged-citation rejection |
| R6 | SFT masking/packing | Planned | Shift, zero seeds, segments, weighted/device tests |
| R7 | SFT preparation | Planned; data actions user-gated | Source provenance, generator terms, holdout/filter tests |
| R8 | Run/gate infrastructure | Planned | Fake-engine cadence, stops, cap/CLI checks |
| R9 | Stage A | Software planned; run user-gated | Budget/cap tests; later measured A evidence |
| R10 | Stage B | Software planned; run user-gated | Parent/re-warm tests; later forgetting evidence |
| R11 | Stage C | Software planned; run user-gated | Assistant counters/selection; later SFT/rubric evidence |
| R12 | CPU chat | Planned | Cache/history, stops, injection, logging tests |
| R13 | Oxide chat | Planned | Actual-device cache/logit/residency agreement |
| R14 | Preference tuning | Optional; run user-gated | DPO sign/cache/gradient tests; later evaluation win |
| R15 | Acceptance/model card | Planned; final evaluation user-gated | Evidence validator, Latinist review, user acceptance |

### Draft self-review

- **Spec coverage:** A/B/C recipes, measured budgets, tokenizer experiments, provenance, whole-work leakage, schedules/resume, masks, real/synthetic SFT, optional preferences, chat, evaluation, persona rules, gates, and model card each have an owning task.
- **Step scan:** Implementation tasks specify named red tests, an observed-red command, exact interface work, observed-green commands, and commit steps. Corpus runs are separate user-gated recipes. No function bodies are prescribed.
- **Type consistency:** `EncodedChat.assistant_mask` describes tokens; R6 shifts it onto targets. `RecipeSource` owns resumable input state. `LearningRateSchedule` extends, rather than silently replaces, decoder cosine behavior. Evaluation-only readback and preference objectives have explicit engine extensions.
- **Review Focus:** All five listed failure modes have named tests in R1/R7, R6, R3, R2/R12, and R5/R4 respectively.
- **Proportion:** Reused decoder mathematics are referenced rather than reimplemented. New detail concentrates on recipe boundaries, state, provenance, masks, and measurable acceptance.
- **Evidence boundary:** This review checks the draft’s internal specification. It does not establish compilation, device correctness, corpus rights, performance, or model quality.

## Open issues

- **Decoder implementation:** Tasks 1–10 are not demonstrated green by the scaffold. This is the primary execution blocker.
- **Corpus supply:** No admitted corpus snapshot, edition list, work-level inventory, or usable token count was measured. All corpus acquisition requires an explicit user request.
- **Rights:** Perseus edition headers, share-alike implications, Latin Library reuse, CC100 terms, and generator output rights require per-source decisions. Unresolved material remains excluded.
- **Held-out availability:** `Laelius` and `Ad Atticum` are proposed fixed evaluation-only groups. Missing rights-cleared editions require an explicit replacement decision before data preparation.
- **Extraction formats:** Dialogue speaker boundaries and letter metadata depend on supplied editions. Do not claim title-level source availability or parse reliability from general knowledge.
- **Hardware fit:** Actual VRAM and workspace consumption are unmeasured. Decoder Task 11 becomes a prerequisite if baseline allocations do not fit the selected microbatch/context.
- **Hyperparameters and thresholds:** Mixtures, LR ranges, replay shares, epochs, warmup/decay fractions, and quality gates are unmeasured proposals. Authorized bounded experiments must determine whether they are useful.
- **Orthography:** Folding destroys distinctions. Its effect on Latin fluency and period transfer is unknown; preserve originals and keep artifact identities distinct.
- **Evaluation validity:** Cloze and morphology probes are adaptations, not established Latin decoder benchmarks. Judge Latin competence is unverified; human Latinist review remains mandatory.
- **Safety boundary:** Fixed refusals and typed quotation lookup cannot guarantee arbitrary generated statements are historically accurate or safe. The disclosed scope must remain narrow.
- **Prompt-weight objective:** Optional weighted loss adds a new backend path. Assistant-only must remain usable without enabling it.
- **Preference value:** DPO may add cost, shorten responses, or provide no benefit. ORPO/KTO are research alternatives, not implemented fallback modes.
- **Versioning:** Select the live next free CPU/GPU version at implementation time, preserve the current mapping format, and keep tags local.
- **Final outcome:** A fluent short Latin imitator is a plausible experiment. A reliable conversational Cicero is not an established consequence of this parameter budget.
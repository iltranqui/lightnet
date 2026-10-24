# Recursive / Looped Depth Decoder Implementation Plan

> Drafted by gpt-6.1-sol (reasoning effort high), unreviewed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. The repository instruction in `tasks/rust/AGENTS_LLM.md` governs delegation, including its requirement to respect the current operator override against subagents.

**Goal:** Add a weight-shared recurrent decoder with exact CPU reference gradients, reproducible depth sampling, correct generation caches, resident oxide execution, and optional causal adaptive depth.

**Architecture:** Split the decoder into a unique prelude, a shared multi-layer recurrent core, and a unique coda. Establish fixed-depth equivalence and full BPTT first, then add sampled training depth, explicit cache policies, and recipe integration. Keep adaptive exits and token routing optional, with their own attention semantics and correctness gates.

**Tech Stack:** Rust 2024, pinned `nightly-2026-08-28`, independent `lightnet-lm`, excluded `lightnet-cuda-oxide`, CPU f32 references, bf16 device working storage, and f32 masters, gradients, moments, and reductions.

**Spec:** This request; `tasks/rust/plan_llm_decoder.md` Tasks 1–16, Progress, and Open issues; `tasks/rust/plan_llm_training_recipe.md` stages A/B/C and R1–R15; `tasks/rust/plan_llm_encdec.md` Tasks 17–40; `tasks/rust/AGENTS_LLM.md`; `CLAUDE.md`; `src/lightnet-lm/CONTEXT.md`; detector recurrence references identified below.

This read-only draft applies `superpowers:writing-plans`. The required repository files were read directly. No files were edited, and no builds, tests, training, corpus processing, or benchmarks were run. Every added file, symbol, configuration field, and acceptance result below is a **proposal**, unless explicitly identified as inspected repository evidence.

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Research basis](#research-basis)
- [Project tree](#project-tree)
- [Component breakdown](#component-breakdown)
  - [Ownership and dependency boundaries](#ownership-and-dependency-boundaries)
  - [Layout, recurrence, and initialization](#layout-recurrence-and-initialization)
  - [Exact parameter arithmetic](#exact-parameter-arithmetic)
  - [Objectives and gradient strategy](#objectives-and-gradient-strategy)
  - [RoPE and cache policies](#rope-and-cache-policies)
  - [Sparse attention and adaptive visibility](#sparse-attention-and-adaptive-visibility)
  - [Training stages and stability](#training-stages-and-stability)
- [Mapping onto existing tasks](#mapping-onto-existing-tasks)
- [Oxide GPU piece inventory](#oxide-gpu-piece-inventory)
- [Staged implementation tasks](#staged-implementation-tasks)
  - [Task L1: Strict loop configuration and parameter counts](#task-l1-strict-loop-configuration-and-parameter-counts)
  - [Task L2: Reusable CPU block boundaries](#task-l2-reusable-cpu-block-boundaries)
  - [Task L3: Fixed-loop forward and unrolled equivalence](#task-l3-fixed-loop-forward-and-unrolled-equivalence)
  - [Task L4: Full BPTT and per-loop supervision](#task-l4-full-bptt-and-per-loop-supervision)
  - [Task L5: Explicit truncated adjoints](#task-l5-explicit-truncated-adjoints)
  - [Task L6: Sampled depth and training engine](#task-l6-sampled-depth-and-training-engine)
  - [Task L7: Exact per-iteration CPU caches](#task-l7-exact-per-iteration-cpu-caches)
  - [Task L8: Shared-first KV architecture](#task-l8-shared-first-kv-architecture)
  - [Task L9: Last-loop storage with exact recomputation](#task-l9-last-loop-storage-with-exact-recomputation)
  - [Task L10: Versioned checkpoints and exact resume](#task-l10-versioned-checkpoints-and-exact-resume)
  - [Task L11: Resident oxide fixed-loop forward](#task-l11-resident-oxide-fixed-loop-forward)
  - [Task L12: Resident oxide backward and updates](#task-l12-resident-oxide-backward-and-updates)
  - [Task L13: Sliding windows across loops](#task-l13-sliding-windows-across-loops)
  - [Task L14: DSA-style attention across loops](#task-l14-dsa-style-attention-across-loops)
  - [Task L15: NSA-style attention across loops](#task-l15-nsa-style-attention-across-loops)
  - [Task L16: A/B/C loop schedules and recipe state](#task-l16-abc-loop-schedules-and-recipe-state)
  - [Task L17: Exit distributions and gate objective](#task-l17-exit-distributions-and-gate-objective)
  - [Task L18: Causal adaptive execution and batched caches](#task-l18-causal-adaptive-execution-and-batched-caches)
  - [Task L19: Token-choice Mixture-of-Recursions](#task-l19-token-choice-mixture-of-recursions)
  - [Task L20: Optional shared-core MoE composition](#task-l20-optional-shared-core-moe-composition)
  - [Task L21: Resident adaptive oxide execution](#task-l21-resident-adaptive-oxide-execution)
  - [Task L22: CLI, acceptance matrix, and evidence](#task-l22-cli-acceptance-matrix-and-evidence)
- [What is worthwhile at the 30-60M Latin pilot vs what only pays off at larger scale](#what-is-worthwhile-at-the-30-60m-latin-pilot-vs-what-only-pays-off-at-larger-scale)
- [Unproven reasoning claims](#unproven-reasoning-claims)
- [Progress table](#progress-table)
- [Open issues](#open-issues)
- [Self-Review](#self-review)

## Global Constraints

- Use `cargo xtask` only: `check`, `gpu-check`, `build`, `clippy`, `test`, `oxide`, `fmt`. Never invoke raw Cargo build/check/test commands or `systemd-run`.
- Maximum four build threads; one build/test command at a time. xtask applies `MemoryHigh=3G`; when no user systemd bus exists it warns and runs uncapped. Record that condition.
- `src/lightnet-cuda-oxide` is **not a workspace member**. GPU tests run through `cargo xtask oxide --test <suite>`.
- New cross-entropy and every loop-level loss retain the repository’s descent-direction sign: backward is seeded with `one_hot(target) - softmax`. Add no compensating negation at the optimizer boundary.
- Per-loop deep-supervision losses sum weighted seeds in that same sign. Generic adjoints propagate their supplied seed linearly.
- Training runs, corpus runs, overfit demonstrations, and benchmarks are **USER-GATED**. Implementation runs only bounded synthetic tests. No speed or quality claim without a recorded, separately authorized measurement.
- Stage/commit only files named by the task, using explicit `git add` paths. `tasks/` is git-excluded: use `git add -f` with named paths. Never use `git add -A`.
- Commit and tag on green using `v0.M.NNN: <what>`; these are placeholders, not allocated numbers. Select the live next unused number only at implementation time.
- Update `tasks/rust/implemented.md`, `tasks/rust/STATUS.md`, `tasks/rust/todo.md`, and `tasks/rust/version-map.tsv` in a **separate docs-only commit**, as requested here.
- That documentation placement differs from the handbook’s placement of three ledgers in the code commit. This request governs this plan; preserve the live version-map row format.
- Unknown cfg keys/sections and unsupported combinations return errors. Never silently substitute identity execution or CPU fallback.
- A zero-test filter, timeout, skipped GPU test, or unavailable device is not a pass.
- Each edited Markdown file needs a TOC. This document’s TOC immediately follows its introduction.
- Label every external claim **VERIFIED (URL)** or **UNVERIFIED**. Never invent external specifications. Never claim C++ parity.
- Do not rename or reinterpret detector-side `[recursive_block]`. `lightnet-lm` must not depend on `lightnet-core`; reuse its ideas only.
- Repository claims must come from inspected files. APIs in existing plans remain proposals until implemented and verified.
- Preserve unrelated changes, existing detector behavior, pinned submodules, and legacy detector checkpoint formats.
- Exact resume means one deterministic backend, precision, geometry, execution policy, and reduction order. CPU/device comparisons use numerical tolerances.
- Do not change the plain decoder’s user-visible default. Selecting recursive, sparse experimental, MoE, or adaptive modes remains explicit.
- This document authorizes no implementation or measurement; it specifies the work for later execution.

## Review Focus

1. **Shared weights accidentally get separate optimizer slots.** L3/L4 pin explicit tied unrolling; L6/L12 assert one gradient, moment pair, clipping contribution, decay application, and update per unique tensor.
2. **Truncation silently changes the claimed derivative.** L5 compares against a frozen-boundary numerical surrogate and requires bitwise full-BPTT equality whenever truncation depth covers all loops.
3. **A cache aliases equal weights with unequal hidden states.** L7–L9 compare every iteration’s logits at nonzero positions; L8 separately tests its changed forward semantics.
4. **Routing leaks future-token information or leaves missing KV entries ambiguous.** L18/L19 perturb suffix tokens, targets, and neighboring batch rows; active-key visibility and absent entries are explicit.
5. **Resume re-samples depth or a rejected update partially commits state.** L10/L12/L16/L21 test mid-accumulation checkpoints, RNG streams, pending gradients, stage boundaries, and atomic rejection.

## Research basis

Labels preserve the distinction between reading a source and proving its transfer to this implementation.

| Evidence | Label and source | Consequence |
|---|---|---|
| Universal Transformer | **VERIFIED**: recurrent shared transformations and per-position halting. [Primary paper](https://arxiv.org/abs/1807.03819) | Supports considering shared depth and causal token exits. |
| ACT | **VERIFIED**: deterministic differentiable computation allocation; its character-LM gains were limited. [Primary paper](https://arxiv.org/abs/1603.08983) | A halting mechanism does not establish a quality improvement. |
| ALBERT | **VERIFIED**: parameter reduction includes cross-layer sharing. [Primary paper](https://arxiv.org/abs/1909.11942) | Sharing changes unique storage, independently of executed depth. |
| PonderNet | **VERIFIED**: learned computation depth trades prediction and computation. [Primary paper](https://arxiv.org/abs/2107.05407) | Motivation only; this plan does not implement its variational objective. |
| Block-Recurrent Transformers | **VERIFIED**: recurrence operates along sequence blocks. [Primary paper](https://arxiv.org/abs/2203.07852) | Different recurrence axis; do not equate it with depth looping. |
| Latent-thought looped Transformers | **VERIFIED**: synthetic reasoning results and theoretical depth results. [Primary paper](https://arxiv.org/abs/2502.17416) | No inference about 30–60M Latin quality follows. “75% fewer parameters” is **UNVERIFIED** here. |
| Huginn | **VERIFIED**: 3.5B parameters, 800B tokens; prelude/core/coda, input injection, sampled recurrence, truncated gradients, and normalization sensitivity. [Primary paper, §§3–4](https://arxiv.org/html/2502.05171v2) | Architectural precedent, not a small-model recipe. Its 2/4/2 geometry and last-eight-step training are not adopted as defaults. |
| Ouro | **VERIFIED**: 1.4B/2.6B models, 7.7T tokens, learned exits, and cache-sharing experiments. [Primary paper](https://arxiv.org/html/2510.25741v2) | Last-step cache reuse was an empirical approximation; it does not prove canonical cache equivalence. |
| Mixture-of-Recursions | **VERIFIED**: token/expert-choice depth routing and recursion-wise/shared caching; the 135M-base isoFLOP arm trailed vanilla. [Primary paper, §§2–3](https://arxiv.org/html/2507.10524v3) | Token routing is optional. Cross-token expert-choice ranking is excluded from the causal pilot design. |
| HRM | **VERIFIED**: two recurrent modules and small supervised puzzle experiments. [Primary abstract](https://arxiv.org/abs/2506.21734) | Its detailed gradient approximation remains **UNVERIFIED** in this audit. |
| TRM | **VERIFIED**: a small recurrent puzzle solver. [Primary paper](https://arxiv.org/abs/2510.04871) | Supervised puzzle results do not establish autoregressive-LM results. |
| GPT-6 Astra looping | **UNVERIFIED** press hypothesis in the supplied notes. [Secondary discussion](https://magazine.sebastianraschka.com/p/gpt-6-astra-looped-transformers-and) | Not a design source; adopt no alleged dimensions, loops, halting, or recipe. |
| DeepSeek V4 layer reuse | **UNVERIFIED**; no primary layer-reuse evidence supplied or established | Do not cite V4 as evidence for recurrence. |
| Other named follow-ups | **UNVERIFIED** in primary sources for this draft | Nanbeige, SMELT, Virtual Logic Depth, and community reconstructions define no implementation requirement. |
| Live LM starting point | **VERIFIED, local**: `src/lightnet-lm/{Cargo.toml,CONTEXT.md,src/lib.rs}` | Existing workspace scaffold; `lib.rs` contains two documentation lines. |
| Existing decoder contracts | **VERIFIED, local**: decoder plan Tasks 1–16 and Progress | All are planned; none supplies an implemented baseline today. |
| Detector recurrence | **VERIFIED, local**: `model.rs:1562`, `exec_accumulate.rs:1337,1515` | Shared body, deterministic per-step RNG, injection, saved steps, and full Rust BPTT are useful ideas. |

The detector recurrence initializes hidden state from input, reuses its body, and accumulates body gradients across reversed steps. Its weights writer stores the body; inspected GPU forward, compact, and update paths explicitly reject this layer. None of that establishes LM cache behavior or permits an LM dependency on the detector graph.

## Project tree

Markers describe the current repository: `[new]` includes files that prerequisite plans must create before this extension modifies them. `[modified]` marks existing files changed by a task below. Reference-only entries remain `[existing]`.

```text
lightnet/
|-- Cargo.toml                                      [existing] workspace boundary
|-- CLAUDE.md                                       [modified] only if CLI commands change
|-- rust-toolchain.toml                             [existing]
|-- xtask/
|   `-- src/main.rs                                 [existing] capped runner; no new arm required here
|-- tools/
|   |-- test_host.sh                                [existing] runner reference
|   |-- test_oxide.sh                               [existing] runner reference
|   `-- lib_cap.sh                                  [existing] cap reference
|-- cfg/
|   |-- llm_smoke.cfg                               [new] decoder prerequisite
|   |-- llm_latin.cfg                               [new] prerequisite; correct count documentation
|   |-- llm_loop_smoke.cfg                          [new] L1
|   |-- llm_loop_latin.cfg                          [new] L1
|   |-- lm_latin_a.json                             [new] R9 prerequisite; extended L16
|   |-- lm_latin_b.json                             [new] R10 prerequisite; extended L16
|   `-- lm_latin_sft.json                           [new] R11 prerequisite; extended L16
|-- tasks/rust/
|   |-- AGENTS.md                                   [existing] reference
|   |-- AGENTS_LLM.md                               [existing] reference
|   |-- lessons.md                                 [existing] reference
|   |-- plan_llm_decoder.md                         [modified] L1 arithmetic correction
|   |-- plan_llm_training_recipe.md                 [modified] L1 arithmetic correction
|   |-- plan_llm_encdec.md                          [existing] optional integration contracts
|   |-- plan_llm_looped_depth.md                    [new] intended eventual home of this draft
|   |-- llm_loop_evidence.md                        [new] L22 acceptance matrix
|   |-- llm_eval.md                                 [new] prerequisite; extended L22
|   |-- llm_recipe_evidence.md                      [new] recipe prerequisite; extended L16
|   |-- implemented.md                             [modified] separate evidence commit
|   |-- STATUS.md                                  [modified] separate evidence commit
|   |-- todo.md                                    [modified] separate evidence commit
|   `-- version-map.tsv                            [modified] separate mapping commit
|-- src/lightnet-core/
|   `-- src/
|       |-- model.rs                               [existing] detector reference only
|       |-- weights.rs                             [existing] detector reference only
|       `-- runtime/
|           |-- exec_accumulate.rs                 [existing] recurrence reference only
|           |-- gpu_forward.rs                     [existing] explicit rejection reference
|           |-- gpu_forward_compact.rs             [existing] explicit rejection reference
|           `-- gpu_update.rs                      [existing] explicit rejection reference
|-- src/lightnet-cuda/
|   `-- src/host_ops.rs                             [existing] AdamW reference
|-- src/lightnet-lm/
|   |-- Cargo.toml                                  [existing] no core/backend dependency
|   |-- CONTEXT.md                                  [modified] module map
|   |-- src/
|   |   |-- lib.rs                                  [modified] exports
|   |   |-- config.rs                               [new] decoder prerequisite
|   |   |-- types.rs                                [new] prerequisite; loop state extensions
|   |   |-- ops.rs                                  [new] decoder prerequisite
|   |   |-- attention.rs                            [new] prerequisite; loop visibility adapters
|   |   |-- decoder.rs                              [new] prerequisite; block extraction L2
|   |   |-- cache.rs                                [new] prerequisite; shared cache utilities
|   |   |-- optim.rs                                [new] decoder prerequisite
|   |   |-- train.rs                                [new] prerequisite; engine integration
|   |   |-- checkpoint.rs                           [new] prerequisite; version changes
|   |   |-- recipe.rs                               [new] recipe prerequisite; depth schedule
|   |   |-- recipe_eval.rs                          [new] recipe prerequisite; evaluation integration
|   |   |-- run.rs                                  [new] recipe prerequisite; evidence integration
|   |   |-- sparse_dsa.rs                           [new] optional decoder Task 15
|   |   |-- sparse_nsa.rs                           [new] optional decoder Task 16
|   |   |-- loop_config.rs                          [new] L1
|   |   |-- loop_decoder.rs                         [new] L3-L5
|   |   |-- loop_train.rs                           [new] L6
|   |   |-- loop_cache.rs                           [new] L7-L9
|   |   |-- loop_recipe.rs                          [new] L16
|   |   |-- loop_halting.rs                         [new] L17-L18
|   |   |-- loop_routing.rs                         [new] L19
|   |   |-- loop_moe.rs                             [new] L20
|   |   `-- stacks/
|   |       |-- mod.rs                              [new] optional encdec prerequisite
|   |       |-- contract.rs                         [new] optional encdec Task 17
|   |       |-- tape.rs                             [new] optional encdec prerequisite
|   |       |-- router.rs                           [new] optional Task 31
|   |       |-- moe.rs                              [new] optional Task 32
|   |       `-- balance.rs                          [new] optional Task 33
|   `-- tests/
|       |-- lm_loop_contract.rs                     [new] L1
|       |-- lm_loop_blocks.rs                       [new] L2
|       |-- lm_loop_forward.rs                      [new] L3
|       |-- lm_loop_bptt.rs                         [new] L4
|       |-- lm_loop_truncated.rs                    [new] L5
|       |-- lm_loop_sampling.rs                     [new] L6
|       |-- lm_loop_cache.rs                        [new] L7
|       |-- lm_loop_shared_kv.rs                    [new] L8
|       |-- lm_loop_last_cache.rs                   [new] L9
|       |-- lm_loop_checkpoint.rs                   [new] L10
|       |-- lm_loop_window.rs                       [new] L13
|       |-- lm_loop_dsa.rs                          [new] L14
|       |-- lm_loop_nsa.rs                          [new] L15
|       |-- lm_loop_recipe.rs                       [new] L16
|       |-- lm_loop_halting.rs                      [new] L17
|       |-- lm_loop_adaptive.rs                     [new] L18
|       |-- lm_loop_routing.rs                      [new] L19
|       |-- lm_loop_moe.rs                          [new] L20
|       `-- lm_loop_acceptance.rs                   [new] L22
|-- src/lightnet-cuda-oxide/                         [existing] excluded workspace
|   |-- Cargo.toml                                  [existing] Task 8 adds normal LM dependency
|   |-- CONTEXT.md                                  [modified] device module map
|   |-- src/
|   |   |-- lib.rs                                  [modified] module export
|   |   |-- lm.rs                                   [new] Tasks 8-12 prerequisite
|   |   |-- lm_chat.rs                              [new] R13 prerequisite
|   |   |-- lm_loop.rs                              [new] L11-L12, L21
|   |   `-- kernels/
|   |       |-- mod.rs                              [modified] kernel module export
|   |       |-- lm.rs                               [new] decoder prerequisite
|   |       |-- lm_loop.rs                          [new] L11-L12, L21
|   |       `-- lm_experts.rs                       [new] optional encdec Task 38
|   `-- tests/
|       |-- oxide_lm_loop_forward.rs                [new] L11
|       |-- oxide_lm_loop_train.rs                  [new] L12
|       |-- oxide_lm_loop_window.rs                 [new] L13
|       |-- oxide_lm_loop_dsa.rs                    [new] L14
|       |-- oxide_lm_loop_nsa.rs                    [new] L15
|       `-- oxide_lm_loop_adaptive.rs               [new] L21
`-- src/lightnet-cli/
    |-- CONTEXT.md                                  [modified] L22
    `-- src/
        |-- lm.rs                                   [new] prerequisite; L22 parsing
        `-- lm_recipe.rs                            [new] recipe prerequisite; L22 parsing
```

## Component breakdown

### Ownership and dependency boundaries

| Component | Role and responsibilities | Depends on / consumers | Constraints and open questions |
|---|---|---|---|
| Foundation contracts | Tokens, masks, parameters, optimizer, plain decoder | Decoder Tasks 1–7 / all CPU loop work | Must exist and be green before extension execution. |
| `loop_config.rs` | Strict depth layout and unique-count arithmetic | `DecoderConfig` / every loop component | `layers` means unique blocks; effective depth is separate. |
| `decoder.rs` block adapter | Exposes existing block, embedding, and readout adjoints | Tasks 3–4 / fixed loops and CPU oracles | Preserve plain decoder operation order and parameter IDs. |
| `loop_decoder.rs` | Recurrence, tapes, injection, full/truncated backward | Block adapter / engines, evaluation, caches | One registry; per-invocation activations. |
| `loop_train.rs` | Depth sampling, unique updates, model RNG | Task 6 / recipe and checkpoints | No resampling during recomputation. |
| `loop_cache.rs` | Iteration-specific KV ownership and exact alternatives | Task 7 / generation, adaptive depth, oxide oracle | Policy is architecture/session identity. |
| Sparse adapters | Recurrent windows, selections, compression state | Tasks 14–16 / optional long-context loop execution | No cross-iteration selection reuse by weight identity. |
| `loop_recipe.rs` | Stage-local depth policy and state transitions | R3/R8–R11 / run orchestration | LR tokens and compute counters remain distinct. |
| `loop_halting.rs` | Exit distribution, gate gradients, causal exits | Fixed loops / optional adaptive execution | Gate sees current causal features only. |
| `loop_routing.rs` | Token-choice depth and deterministic dispatch | Prelude features / optional MoR | No ranking against future tokens. |
| `loop_moe.rs` | Shared-core expert FFNs | Encdec 31–33 / optional combined experiments | Depth routing and expert routing are separate. |
| Oxide loop bridge | Native repeated execution, device tapes and reductions | Tasks 8–11 / optional device training/chat | No host activation/gradient fallback. |
| CLI and evidence | Explicit selection, rejection matrix, reports | Implemented capabilities / operator | Command availability grants no run authorization. |

`lightnet-lm` already depends on `lightnet-cuda`; it does not gain a detector or concrete-backend dependency. The existing decoder plan’s references to core parser/error reuse conflict with that boundary; this extension uses its own loop parser and error type.

### Layout, recurrence, and initialization

**Decision:** retain ordinary pre-RMSNorm, causal GQA, RoPE, SwiGLU, bias-free blocks and the tied head from decoder Tasks 3–4.

Let `P`, `C`, and `Z` denote prelude, core, and coda block counts:

```text
integer IDs -> tied embedding -> P unique blocks -> e
                                                  |
                                             h0 = e
                                                  |
                           repeat C shared blocks K times
                                                  |
                                       Z unique blocks
                                                  |
                                     final norm -> tied head
```

The exact recurrence is:

```text
h0 = e
u1 = h0
ui = h(i-1)                         when i > 1 and injection_scale == 0
ui = h(i-1) + injection_scale * e   when i > 1 and injection_scale != 0
hi = Core(ui), i = 1..K
final_logits = Head(FinalNorm(Coda(hK)))
```

**Decision:** first-pass injection is supplied by initialization; additive re-injection begins with pass two. This preserves loop-count-one plain-decoder equivalence even when later-pass injection is configured.

**Decision:** no random initial state, previous-token warm start, learned depth embedding, outer residual, or extra recurrent norm in the first implementation. Injection defaults to zero; `0.1` is an explicit ablation value.

**Decision:** keep the base Task 4 initialization rule based on the number of unique blocks. Do not silently reinitialize shared weights when K changes.

**Decision:** pilot layout is `P=2,C=5,Z=2`, not the below-budget 8192-vocabulary `2/4/2` example. Fixed `K=2` has fourteen executed blocks and nine unique blocks.

**Decision:** sampled training uses one K per microbatch initially. Adaptive inference uses per-token exits only after L18; inference never samples K.

**Open question:** whether learned concatenation injection or sandwich norms improve this pilot requires a separate architecture/count/equivalence revision and authorized measurements.

### Exact parameter arithmetic

For hidden width 512, intermediate width 1536, Q/KV heads 8/4, and head dimension 64:

```text
Q = 512*512 = 262,144
K = 512*256 = 131,072
V = 512*256 = 131,072
O = 512*512 = 262,144
attention = 786,432

gate + up + down = 3*512*1536 = 2,359,296
two RMSNorm scales = 2*512 = 1,024
L = parameters per unique block = 3,146,752

embedding + final norm:
8192*512 + 512  = 4,194,816
16384*512 + 512 = 8,389,120
```

No projection biases, extra adapter, duplicated head, or learned loop embedding is included.

| Unique blocks | Example layout | Parameters, vocab 8192 | Parameters, vocab 16384 |
|---:|---|---:|---:|
| 7 | 1/5/1 | 26,222,080 | 30,416,384 |
| 8 | 2/4/2 | 29,368,832 | 33,563,136 |
| 9 | **2/5/2 pilot** | **32,515,584** | **36,709,888** |
| 12 | Plain baseline | **41,955,840** | **46,150,144** |
| 14 | Plain compute-depth comparator for pilot K=2 | 48,249,344 | 52,443,648 |

The smoke geometry has `L=2,336`, embedding plus final norm `4,176`, and two unique blocks total **8,848**.

**Decision:** assert these counts directly against registered scalar storage. Loop count changes effective depth `P+K*C+Z`, not the unique count.

**Conflict:** decoder Task 10 and recipe R9 repeat **45,628,928 / 49,823,232**. Those exceed the stated geometry by **3,673,088** at either vocabulary. L1 corrects the specified assertions; no unnamed tensor is invented to fill the difference.

Full BPTT stores K core activation sets without recomputation. Core execution costs K passes; weight sharing does not divide those costs.

Optional scalar halting adds `512+1=513` parameters. A token-choice router over K depths adds `K*(512+1)`, hence **1,026** at K=2. Their optimizer state is additional too.

### Objectives and gradient strategy

**Decision:** full BPTT is the correctness reference and pilot default. Truncation is opt-in and changes the adjoint, not the forward.

**Decision:** intermediate supervision uses `Head(FinalNorm(hi))` before the coda. It reuses the final norm and embedding; it adds no parameters or per-loop coda caches.

For auxiliary weight `a`, K probes, and the same active-target mask:

```text
objective = (1-a)*CE(final_logits)
          + (a/K)*sum_i CE(loop_probe_logits[i])
```

Use `a=0` for equivalence and primary-only training; proposed deep-supervision ablation uses `a=0.5`. Count active targets once, regardless of K. Report final-output NLL separately from the weighted training objective.

Every seed has the repository sign. Binary zero logits, target 1, weight 0.25 produce `[-0.125,+0.125]`.

**Decision:** truncate separately for each loss endpoint t. With depth d, retain at most the final d recurrent passes ending at t. Detach the entering recurrent state when `t>d`; direct prelude injection within retained passes remains differentiable.

Shared-first KV producers outside that retained interval are detached too. Shared-first producers inside it receive all retained consumer contributions.

**Decision:** `d>=K` dispatches to the full-BPTT implementation, preserving exact arithmetic order. L5 specifies numerical differentiation of the stopped surrogate for shorter horizons; it must not compare against the unstopped objective.

### RoPE and cache policies

**Decision:** loop index never changes token position. Recompute Q/K from the current invocation’s input; rotate fresh Q/K once at the token’s absolute position. Cache rotated K and unrotated V, as decoder Task 7 specifies.

| Policy | Forward semantics | Persistent core banks | Equivalence requirement |
|---|---|---:|---|
| `PerIteration` | Ordinary depth-unrolled decoder | K*C | Canonical full-prefix equivalence at every loop |
| `SharedFirst` | Later loops reuse each core sublayer’s first-loop K/V | C | Equality to this explicitly shared-KV full-prefix architecture |
| `LastOnlyRecompute` | Canonical recurrence; reconstruct discarded banks from retained prefix on each append | C | Canonical full-prefix equivalence, with recomputation cost |
| Approximate last-step substitution | Earlier-loop queries consume historical final-loop K/V | Not implemented | Reject; empirical quality similarity is not equivalence |

Prelude and final-coda banks remain separate. The fixed-loop per-iteration payload is:

```text
2 * B * Hkv * Dhead * element_bytes * context * (P + K*C + Z)
```

For the 2/5/2 pilot, K=2, batch one, bf16, this is **7,340,032 bytes** at context 512 and **14,680,064 bytes** at 1024, excluding metadata and scratch.

`LastOnlyRecompute` reduces persistent banks, not necessarily peak allocation. It retains the complete bounded token/position/segment prefix and may reconstruct a full transient tape. Never claim a constant-memory or speed improvement from its persistent-bank count.

**Decision:** cache identity includes model revision, layout, K/budget, policy, sparse settings, and controller identity. Changing any requires reset; errors precede mutation.

**Open question:** approximate final-loop substitution may be investigated only through a separately specified policy and authorized quality comparison. It is not an alternative implementation of an exact policy.

### Sparse attention and adaptive visibility

**Decision:** decoder Task 14’s `full_every` indexes **unique block ordinal**, including the prelude offset. The same core sublayer is full/windowed on every invocation.

Per-iteration window rings are independent. Absolute positions, document segments, and iteration identity survive wraparound.

**Decision:** DSA selections are keyed by invocation, query position, segment, model revision, and policy. Shared weights do not imply shared selected indices. Freeze discrete selection for finite differences.

**Decision:** NSA compression buffers, completed-block markers, branch gates, and selected block sets belong to an invocation’s KV source. Incomplete blocks remain invisible.

**Decision:** L18’s adaptive `PerIteration` mode uses recursion-wise visibility: loop i computes only active queries and attends only to valid keys whose tokens also execute loop i. Exited tokens retain their final hidden state for the coda but provide no invented deeper-loop KV.

This is a different attention graph from fixed-K dense recurrence. Its full-prefix oracle uses the same active-key mask. The final coda runs once over each token’s selected state.

**Decision:** adaptive+sparse combinations initially return `Unsupported`. Fixed-K sparse correctness does not establish correctness of depth-dependent sparse candidate sets.

### Training stages and stability

All choices here are proposals; their quality and stability remain unmeasured.

| Stage | Depth policy | Recipe interaction |
|---|---|---|
| A | First 3% of successful-target budget: K=1. Then uniform K in `{1,2}`. Full BPTT; optional `a=0.5` probes. | Preserve Cicero exclusion, context 512 first, 32,768 active targets/update, β1=.9/β2=.95, and four-epoch exposure cap. |
| B | Continue A’s depth distribution and objective unless explicitly changed | Preserve replay candidates .05/.15/.25, cumulative exposures, and re-warm factors .1/.3 of A peak. |
| C | Preserve sampled `{1,2}` or explicitly select fixed K=2 | Assistant-only shifted mask applies to final and every probe loss; retain 1–3-epoch candidate ceiling. |
| Gate calibration | Optional later, decoder frozen | Separate user-gated experiment; cannot masquerade as stage B or C completion. |

The 3% depth warmup is independent of the LR schedule; R3 still owns cosine/WSD and stage-local LR origins. Peak LR candidates `3e-4/6e-4/1e-3` remain unmeasured; looping does not justify increasing them.

**Decision:** retain selected moments across A→B→C by default. Change depth/objective only at an update boundary; record the new configuration identity and stage-local schedule origin.

**Decision:** replay selection counts real targets, not loop exposures. Record additional `core_token_passes` for compute accounting; never multiply successful-target or source-exposure counters by K.

Stage-B forgetting remains `F_A <= +0.05` nats/token with Cicero development gain at least `0.02`, evaluated at the **same fixed depth and policy** for parent and candidate.

**Decision:** evaluate fixed K=1 and K=2 separately; do not average a sampled validation objective into the main acceptance NLL.

**Decision:** stop on non-finite updates; inspect per-loop state norms and losses in authorized runs. Do not quietly clip hidden states, change norms, shorten K, or reset moments to conceal instability.

## Mapping onto existing tasks

“Reused” means a prerequisite contract, not existing implemented functionality.

| Existing task | Status | Recursive extension / owner |
|---|---|---|
| Decoder 1: contracts | Changed | Strict `[loop_depth]`, unique/effective depth distinction, exact counts; L1 |
| Decoder 2: tokenizer | Reused | Same tokenizer IDs, hashes, and reserved chat controls |
| Decoder 3: ops | Reused | Same RMSNorm/RoPE/GQA/SwiGLU/CE; L2/L4 |
| Decoder 4: model | Changed | Extract block/readout boundaries; shared invocations; L2–L5 |
| Decoder 5: loader | Reused | Same packing, segments, cursor, target coverage |
| Decoder 6: trainer/state | Changed | Depth RNG, loop metadata, unique accumulation; L6/L10 |
| Decoder 7: generation | Changed | Loop-aware cache policies; L7–L9 |
| Decoder 8: bf16 forward | Changed | Resident recurrent scheduling/tapes; L11 |
| Decoder 9: backward/update | Changed | Repeated-use accumulation and loop state; L12 |
| Decoder 10: pilot | Changed | Correct counts and separate loop arm; L1/L22 |
| Decoder 11: flash/chunked CE | Reused/changed | Invoke per core pass; chunk probe heads; deterministic recomputation; L11/L12 |
| Decoder 12: FP8 | Deferred | Unsupported loop FP8 initially; shared weight identity and invocation activation histories need separate work |
| Decoder 13: cuTile | Deferred | Recursive mode returns unsupported |
| Decoder 14: window | Changed | Independent invocation rings, unique-ordinal pattern; L13 |
| Decoder 15: DSA | Changed, optional | Loop-local selection/source identity; L14 |
| Decoder 16: NSA | Changed, optional | Loop-local compression/branches; L15 |
| R1: corpus registry | Reused | No new data admission or split policy |
| R2: chat tokens | Reused | Vocabulary size includes controls |
| R3: schedules/replay | Changed | Loop schedule, RNG/state, stage-boundary validation; L16 |
| R4: evaluation | Changed | Fixed-depth final NLL; per-loop diagnostics separate; L16/L22 |
| R5: persona/quotes | Reused | No architecture-based factuality claim |
| R6: SFT mask | Changed | Identical shifted mask at every readout; L16 |
| R7: SFT preparation | Reused | No new synthetic-data generation |
| R8: orchestration | Changed | Depth identity, compute counters, unsupported-mode gates; L16/L22 |
| R9: stage A | Changed | Correct count and proposed depth warmup; L1/L16 |
| R10: stage B | Changed | Replay unchanged; depth-matched forgetting; L16 |
| R11: stage C | Changed | Assistant-target normalization across probe losses; L16 |
| R12: CPU chat | Changed | Explicit recursive cache/session policy; L22 |
| R13: oxide chat | Changed | Device loop banks and next-logit readback; L11/L22 |
| R14: preferences | Deferred | Recursive DPO returns unsupported until its sequence-seed path is specified |
| R15: acceptance/card | Changed | Unique count, executed depth, policies, evidence; L22 |
| Encdec 17: contracts | Reused optionally | Do not create a second incompatible loop tensor/error system |
| Encdec 18: masks | Reused optionally | Adaptive visibility is an additional mask, not bidirectional attention |
| Encdec 19: RoPE/YaRN | Deferred combination | Ordinary split-half RoPE first; no inferred iteration position |
| Encdec 20: projections | Deferred combination | GQA first; low-rank MQA requires its own loop/cache matrix |
| Encdec 21: MLA | Deferred combination | No automatic latent-cache sharing across loops |
| Encdec 22: encoder | Unchanged | No recurrent encoder deliverable |
| Encdec 23: cross bank | Unchanged | No looped cross-attention deliverable |
| Encdec 24: decoder stack | Reuse boundary only | Preserve plain-decoder oracle; avoid forcing stack prerequisites |
| Encdec 25: seq2seq | Deferred combination | Not required for Latin decoder |
| Encdec 26: decoder-only reuse | Reused concept | Same revision/ownership discipline; L2/L7 |
| Encdec 27: causal split | Deferred combination | Different composition axis |
| Encdec 28: split cache | Deferred combination | No inferred global-bank reuse |
| Encdec 29: compression | Deferred combination | Distinct from decoder NSA; not renamed |
| Encdec 30: sparse schedules | Reused optionally | Invocation identity extends schedule ownership; L14/L15 |
| Encdec 31: router | Reused optionally | Stable selection/dispatch ideas; depth axis remains separate; L19/L20 |
| Encdec 32: experts | Changed, optional | One expert bank per unique core block; L20 |
| Encdec 33: balance | Changed, optional | Counts aggregate actual expert uses across loops; L20 |
| Encdec 34: MTP | Deferred combination | Loop probes predict the same next token, not future horizons |
| Encdec 35: state | Reused optionally | One versioned optional-state envelope; L10/L20 |
| Encdec 36: oxide stacks | Unchanged | No encoder/cross GPU dependency |
| Encdec 37: sparse/latent GPU | Reused optionally | Only selected sparse primitives, not latent/split behavior |
| Encdec 38: device experts | Prerequisite if selected | Loop MoE GPU remains unsupported until explicitly wired |
| Encdec 39: FP8 roles | Deferred combination | No implicit recursive FP8 support |
| Encdec 40: pilot configs | Changed documentation | Its 8/2-head, width-1360 geometry differs; do not mix count formulas |

## Oxide GPU piece inventory

“Reused” below means supplied by prerequisite decoder work. It does not mean present in today’s scaffold.

| Piece | Reused / new | Buffers and ownership | Acceptance owner |
|---|---|---|---|
| Embedding and scatter-add | Reused | One embedding master/gradient; repeated IDs reduce deterministically | L11/L12 |
| RMSNorm, SwiGLU, projections | Reused | Unique weights, invocation-specific activations | L11/L12 |
| RoPE | Reused | Absolute positions; fresh Q/K each invocation | L11 |
| Causal GQA/flash | Reused | Per-invocation Q, LSE, K/V and backward scratch | L11/L12 |
| Injection | New small bf16 elementwise kernel | `h`, immutable prelude `e`, output `u`; no extra weights | L11 |
| Injection adjoint | New orchestration plus f32 scaled accumulation | Separate recurrent and prelude seeds | L12 |
| Shared gradient addition | Reused reduction or new deterministic add kernel | One f32 destination per parameter ID | L12 |
| Per-loop probes | Reused tied projection/CE | Chunk scratch reused serially; no mandatory `[K,B,T,V]` allocation | L12 |
| Full-BPTT tapes | New ownership | K core tapes; prelude/coda once | L11/L12 |
| Truncated/recomputed tapes | New ownership | Retained suffixes and saved RNG starts; shared-KV source lifetime explicit | L12 |
| Iteration caches | New indexing | Bank identity `(region, unique_layer, iteration)` | L11 |
| Shared-first KV | New source references | One producer tape/bank, multiple consumer adjoints | L11/L12 |
| Last-only exact cache | New replay orchestration | Retained prefix, final banks, transient reconstruction | L11 |
| Window/DSA/NSA | Reused kernels, new ownership adapters | Rings, index selections, compressed buffers per invocation/source | L13–L15 |
| Halting | New f32 sigmoid/survival/CDF and reductions | Per-token active flags, exit depth, survival; gate parameters | L21 |
| Active-row dispatch | New or reused expert prefix-sum/gather | Stable row indices and per-bank validity | L21 |
| Adaptive KV append | New | Only active tokens append to corresponding depth bank | L21 |
| Expert FFN | Optional prerequisite kernels | Unique expert weights; invocation dispatch scratch | Deferred device composition |
| Optimizer/checkpoint transfers | Reused, metadata extended | f32 masters/moments/pending gradients and named RNG states | L12/L21 |

No new tensor-core GEMM family is required merely to share depth. CPU scheduling of launches is allowed; downloading intermediate features to decide exits is not the resident implementation.

## Staged implementation tasks

All CPU tasks require decoder Tasks 1–7 green first; GPU tasks additionally require Tasks 8–9 and Task 11 when selected. L16 requires the named recipe infrastructure. Optional prerequisites are stated at their owning tasks.

The prerequisite APIs are the proposed `DecoderConfig`, `TokenBatch`, `RngState { state: u64 }`, `Parameter`, `DecoderParams`, `DecoderGrads`, `LossSummary`, `CpuDecoder`, `EngineState`, `TrainingSnapshot`, `LmEngine`, and checkpoint functions from decoder Tasks 1/4/6.

All new loop APIs return `LoopResult<T>` defined in L1. Map prerequisite errors explicitly. Do not adopt the old core-specific `Result` references.

Verification conventions:

- CPU finite differences: centered perturbation `1e-3`; tolerance `2e-3 + 2e-2*abs(reference)`.
- CPU cached/prefix logits: `1e-5 + 1e-5*abs(reference)`.
- Oxide normalized maximum error: `max(abs(actual-reference))/max(1,max(abs(reference)))`.
- Oxide primitive forward tolerance 0.02, complete smoke logits 0.05, parameter gradients 0.08.
- Freeze discrete selections and stopped boundary values in the applicable numerical oracle.
- Every green command requires a nonzero matching-test count and zero failures.

Every task additionally names `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/CONTEXT.md`, this eventual plan’s progress table, and the four common ledgers. Oxide tasks also name `src/lightnet-cuda-oxide/CONTEXT.md`.

For each task, the final **Commit** step means these separate actions after its named tests pass:

1. Stage its explicit implementation/test paths; stage export/map edits only when changed.
2. Commit with that task’s `v0.M.NNN: ...` message and actual test evidence.
3. Create the unused local lightweight `v0.M.NNN` tag.
4. Update TOCs, progress, and evidence ledgers; explicitly stage named `tasks/rust/` paths with `git add -f`.
5. Commit documentation as `docs(rust): record loop-depth increment`, including the actual SHA mapping.

Never run the version placeholders literally. No task permits broad directory staging.

### Task L1: Strict loop configuration and parameter counts

**Files:** create `src/lightnet-lm/src/loop_config.rs`, `src/lightnet-lm/tests/lm_loop_contract.rs`, `cfg/llm_loop_smoke.cfg`, `cfg/llm_loop_latin.cfg`; modify the two existing decoder/recipe plan documents to correct counts.

**Interfaces**

- Consumes: `DecoderConfig::parameter_count(&self) -> Result<u64>` from decoder Task 1.
- Produces: `LoopError::{Config(String), Shape(String), State(String), NonFinite(String), Unsupported(String), Io(std::io::Error)}`; `LoopResult<T> = std::result::Result<T, LoopError>`.
- Produces: `LoopConfig { prelude: usize, core: usize, coda: usize, max_loops: usize, injection_scale: f32 }`.
- Produces: `CachePolicy::{PerIteration, SharedFirst, LastOnlyRecompute}` and `Adjoint::{Full, Last { depth: usize }}`.
- Produces: `LoopConfig::from_cfg(text: &str, base: &DecoderConfig) -> LoopResult<Self>`.
- Produces: `LoopConfig::parameter_count(&self, base: &DecoderConfig) -> LoopResult<u64>` and `effective_depth(&self, loops: usize) -> LoopResult<usize>`.

- [ ] Write `lm_loop_contract_counts`: assert smoke two-block count 8,848; pilot nine-block counts 32,515,584/36,709,888; twelve-block counts 41,955,840/46,150,144; eight-block 8192 count 29,368,832 fails pilot minimum.
- [ ] Write `lm_loop_contract_strict`: `[loop_depth]` accepts only `prelude_layers,core_layers,coda_layers,max_loops,injection_scale`; reject `loops=0`, core zero, nonfinite scale, unknown keys/sections, overflow, and `base.layers != P+C+Z`.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_contract_`; expect missing `LoopConfig` or intended count assertions to fail.
- [ ] Implement the interfaces and cfgs: smoke `0/2/0,max_loops=3`; pilot `2/5/2,max_loops=2`, scale zero; correct both plans’ inherited count assertions and TOCs.
- [ ] Run the same filter; expect all named tests green, including `effective_depth(2)==14` for the pilot.
- [ ] Stage the six named implementation/test/cfg/plan paths explicitly; commit as `v0.M.NNN: define recursive decoder contracts and exact counts`.

### Task L2: Reusable CPU block boundaries

**Files:** modify `src/lightnet-lm/src/decoder.rs`; create `src/lightnet-lm/tests/lm_loop_blocks.rs`.

**Interfaces**

- Consumes: Task 4 `CpuDecoder::forward`, `backward`, `parameters`, `set_parameters`.
- Produces opaque `BlockTape`, `ReadoutTape`; `BlockPass { output: Vec<f32>, tape: BlockTape }`; `ReadoutPass { logits: Vec<f32>, tape: ReadoutTape }`.
- Produces `CpuDecoder::embed_tokens(&self, batch: &TokenBatch) -> LoopResult<Vec<f32>>`.
- Produces `block_forward(&self, block: usize, x: &[f32], batch: &TokenBatch, rng: Option<&mut RngState>) -> LoopResult<BlockPass>`.
- Produces `block_backward(&self, pass: &BlockPass, seed: &[f32]) -> LoopResult<(Vec<f32>, DecoderGrads)>`.
- Produces `readout(&self, x: &[f32], batch: &TokenBatch) -> LoopResult<ReadoutPass>` and `readout_backward(&self, pass: &ReadoutPass, seed: &[f32]) -> LoopResult<(Vec<f32>, DecoderGrads)>`.
- Produces `CpuDecoder::revision(&self) -> u64`; parameter replacement advances revision.

- [ ] Write `lm_loop_blocks_plain_order`: compose embedding, blocks 0/1, and readout; assert bitwise equality to Task 4 forward with dropout zero.
- [ ] Write `lm_loop_blocks_adjoint`: composed adjoints equal plain gradients; readout and input gather contributions reach one embedding ID.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_blocks_`; expect missing block APIs.
- [ ] Extract existing operations without changing arithmetic order, initialization, parameter order, or dropout masks; opaque tapes own model revision and invocation data.
- [ ] Run the same filter, then `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_decoder_`, sequentially; expect green.
- [ ] Stage `src/lightnet-lm/src/decoder.rs` and the named test file; commit as `v0.M.NNN: expose reusable decoder block adjoints`.

### Task L3: Fixed-loop forward and unrolled equivalence

**Files:** create `src/lightnet-lm/src/loop_decoder.rs`, `src/lightnet-lm/tests/lm_loop_forward.rs`.

**Interfaces**

- Consumes: L1 configuration and L2 block/readout APIs.
- Produces `LoopRun { loops: usize, cache_policy: CachePolicy, probe_loops: bool }`.
- Produces opaque `LoopTape`; `LoopForward { logits: Vec<f32>, loop_logits: Vec<Vec<f32>>, tape: LoopTape }`.
- Produces `LoopDecoder::new(base: DecoderConfig, depth: LoopConfig, seed: u64) -> LoopResult<Self>`.
- Produces `LoopDecoder::from_plain(model: CpuDecoder, depth: LoopConfig) -> LoopResult<Self>`.
- Produces `forward(&self, batch: &TokenBatch, run: &LoopRun, rng: Option<&mut RngState>) -> LoopResult<LoopForward>`.
- Produces `parameters(&self) -> &DecoderParams`, `set_parameters(&mut self, params: DecoderParams) -> LoopResult<()>`, `revision(&self) -> u64`.

- [ ] Write `lm_loop_forward_one_plain`: P/C/Z=`1/1/1`, three smoke-width unique blocks, K=1; assert bitwise plain logits for injection scales 0 and 0.1.
- [ ] Write `lm_loop_forward_tied_unrolled`: P/C/Z=`1/2/1`, K=2; independently execute block IDs `[0,1,2,1,2,3]` with shared references and assert bitwise final/probe logits.
- [ ] Write `lm_loop_forward_injection`: scalar recurrence fixture `Core(x)=2*x,e=1,scale=.1,K=3` gives states `[2,4.2,8.6]`; inference consumes no RNG.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_forward_`; expect missing loop forward.
- [ ] Implement the declared recurrence; skip zero-scale addition entirely; reject unimplemented cache policies; store distinct tapes per invocation.
- [ ] Run the same filter; expect green and exactly one parameter allocation per unique block.
- [ ] Stage the two named files; commit as `v0.M.NNN: implement fixed shared-depth decoder forward`.

### Task L4: Full BPTT and per-loop supervision

**Files:** modify `src/lightnet-lm/src/loop_decoder.rs`; create `src/lightnet-lm/tests/lm_loop_bptt.rs`.

**Interfaces**

- Consumes: L3 forward and Task 3 `sparse_ce(logits: &[f32], targets: &[i32], vocab: usize) -> Result<(LossSummary, Vec<f32>)>`.
- Produces `LoopObjective { auxiliary_weight: f32 }`.
- Produces `LoopLoss { main: LossSummary, probes: Vec<LossSummary>, objective_sum: f64, grads: DecoderGrads }`.
- Produces `LoopDecoder::backward(&self, pass: &LoopForward, final_seed: &[f32], probe_seeds: &[Vec<f32>], adjoint: Adjoint) -> LoopResult<DecoderGrads>`.
- Produces `loss_backward(&self, batch: &TokenBatch, run: &LoopRun, objective: &LoopObjective, adjoint: Adjoint, rng: &mut RngState) -> LoopResult<LoopLoss>`.

- [ ] Write `lm_loop_bptt_finite_difference`: K=3, scale 0.1; check every block family, shared norms, embedding, and final norm against negative numerical objective derivatives.
- [ ] Write `lm_loop_bptt_tied_sum`: explicit unrolling with tied IDs yields the same accumulated shared gradient in reverse invocation order; no second tensor or gradient slot exists.
- [ ] Write `lm_loop_bptt_probe_sign`: weight .25 gives binary seed `[-.125,.125]`; `a=0` reproduces primary-only gradients bitwise; ignored rows contribute zero at every readout.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_bptt_`; expect missing backward.
- [ ] Implement reverse recurrence, direct injection adjoints, probe seeds, coda/readout adjoints, and unique-ID summation; reject stale tapes.
- [ ] Run the same filter; expect finite-difference tolerance, exact tied-slot assertions, and one active-target denominator.
- [ ] Stage the two named files; commit as `v0.M.NNN: add full recurrent BPTT and loop supervision`.

### Task L5: Explicit truncated adjoints

**Files:** modify `src/lightnet-lm/src/loop_decoder.rs`; create `src/lightnet-lm/tests/lm_loop_truncated.rs`.

**Interfaces**

- Consumes: L4 `backward` and `loss_backward`; L1 `Adjoint`.
- Produces support for `Adjoint::Last { depth }`; existing signatures remain unchanged.
- Produces no changed forward semantics.

The stopped surrogate for a loss ending at t is:

```text
b = max(0, t-depth)
if b > 0: replace h_b by a constant saved at the unperturbed parameters
recompute passes b+1..t using perturbed parameters
retain direct e -> injection paths in those passes
differentiate this surrogate, not the original full recurrence
```

- [ ] Write `lm_loop_truncated_covers_full`: K=1/2/3, d=K and d=K+2; assert bitwise full-BPTT gradients with probes enabled.
- [ ] Write `lm_loop_truncated_stop_adjoint`: scalar `h_i=w*h_(i-1),h0=1,w=2,K=3` has full derivative 12; last-one derivative 4; last-two derivative 8.
- [ ] Write `lm_loop_truncated_finite_difference`: freeze each loss endpoint’s boundary separately; check all retained parameter families and direct injection paths with the standard tolerance.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_truncated_`; expect unsupported truncation or intended assertions.
- [ ] Implement endpoint-specific stop boundaries, reject depth zero, and dispatch d≥K directly to full BPTT.
- [ ] Run the same filter; expect green; record that truncation is an explicitly approximate gradient for the unstopped objective.
- [ ] Stage the two named files; commit as `v0.M.NNN: define and verify truncated recurrent adjoints`.

### Task L6: Sampled depth and training engine

**Files:** create `src/lightnet-lm/src/loop_train.rs`, `src/lightnet-lm/tests/lm_loop_sampling.rs`; modify `src/lightnet-lm/src/types.rs`, `src/lightnet-lm/src/train.rs`.

**Interfaces**

- Consumes: L4/L5 loss APIs and Task 6 `LmEngine`.
- Produces `LoopSampling::{Fixed(usize), Uniform { min: usize, max: usize }}`.
- Produces `LoopTrainSpec { sampling: LoopSampling, adjoint: Adjoint, objective: LoopObjective, cache_policy: CachePolicy }`.
- Produces `LoopTrainState { spec: LoopTrainSpec, count_rng: RngState, dropout_rng: RngState, microbatch_ordinal: u64, core_token_passes: u64 }`.
- Produces `sample_loops(policy: &LoopSampling, rng: &mut RngState) -> LoopResult<usize>`.
- Produces `LoopEngine::new(model: LoopDecoder, spec: LoopTrainSpec, seed: u64) -> LoopResult<Self>` and `impl LmEngine for LoopEngine`.
- Extends `EngineState` with optional loop configuration and `LoopTrainState`; plain state remains representable.

- [ ] Write `lm_loop_sampling_replay`: cloned SplitMix64 count states yield identical 32-draw sequences in `[1,3]`; fixed K consumes no count RNG.
- [ ] Write `lm_loop_sampling_no_redraw`: recomputation leaves the selected K and final RNG states unchanged; all-ignored batches draw nothing.
- [ ] Write `lm_loop_sampling_unique_update`: repeated shared uses produce one clipping entry, one decay application, and one AdamW update; counts remain actual targets.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_sampling_`; expect missing loop engine.
- [ ] Implement unbiased bounded-integer sampling, one draw per nonempty supervised microbatch, independent count/dropout `RngState`s, and unique-registry updates.
- [ ] Run the same filter; expect green, including in-memory export/import continuation.
- [ ] Stage the four named files; commit as `v0.M.NNN: add deterministic sampled-depth training`.

### Task L7: Exact per-iteration CPU caches

**Files:** create `src/lightnet-lm/src/loop_cache.rs`, `src/lightnet-lm/tests/lm_loop_cache.rs`; modify `src/lightnet-lm/src/loop_decoder.rs`.

**Interfaces**

- Consumes: L3 model and decoder Task 7 cache semantics.
- Produces opaque `LoopCache`; `LoopCache::new(model: &LoopDecoder, run: LoopRun, batch: usize) -> LoopResult<Self>`, `reset(&mut self)`.
- Produces `LoopCacheStats { bank_count: usize, payload_bytes: u64, history_tokens: usize, computed_core_rows: u64 }` and `LoopCache::stats(&self) -> LoopCacheStats`.
- Produces `LoopDecoder::prefill_loop(&self, batch: &TokenBatch, cache: &mut LoopCache) -> LoopResult<LoopForward>`.
- Produces `decode_loop(&self, step: &TokenBatch, cache: &mut LoopCache) -> LoopResult<LoopForward>`.
- `decode_loop` requires time one; initial cache support is batch one. Generation targets must be ignored.

- [ ] Write `lm_loop_cache_each_iteration`: tokens `[1,5,6,7,8,9]`, K=1/2/3, MHA and GQA; compare every probe and final logit at positions 1–5 against full-prefix forward.
- [ ] Write `lm_loop_cache_atomic`: stale revision, changed K/policy, invalid ID, nonempty prefill, overflow, and batch two leave cache bytes/length unchanged.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_cache_`; expect missing cache APIs.
- [ ] Implement separate `(iteration,core_sublayer)` banks, prelude/coda banks, checked byte arithmetic, rotated-K/unrotated-V writes, and validation before mutation.
- [ ] Run the same filter; expect CPU cache tolerance at every nonzero position and exact payload counts.
- [ ] Stage the three named files; commit as `v0.M.NNN: add exact iteration-specific decoder caches`.

### Task L8: Shared-first KV architecture

**Files:** modify `src/lightnet-lm/src/decoder.rs`, `src/lightnet-lm/src/loop_decoder.rs`, `src/lightnet-lm/src/loop_cache.rs`; create `src/lightnet-lm/tests/lm_loop_shared_kv.rs`.

**Interfaces**

- Consumes: L2 block tapes and L4 backward.
- Produces `KvPair { key: Vec<f32>, value: Vec<f32> }`.
- Produces `CpuDecoder::block_forward_with_kv(&self, block: usize, x: &[f32], batch: &TokenBatch, borrowed: Option<&KvPair>, rng: Option<&mut RngState>) -> LoopResult<(BlockPass, KvPair)>`.
- Produces `block_backward_with_kv(&self, pass: &BlockPass, seed: &[f32], produced_kv_seed: Option<&KvPair>) -> LoopResult<(Vec<f32>, DecoderGrads, Option<KvPair>)>`.
- Final optional return contains borrowed-bank descent seeds; producer seeds are supplied only to fresh-KV passes.

- [ ] Write `lm_loop_shared_kv_prefix`: K=3; all nonzero-position loop probes/final logits equal an independent first-bank-reuse full-prefix oracle.
- [ ] Write `lm_loop_shared_kv_adjoint`: later-loop dK/dV contributions sum into loop-one producers; finite-difference full BPTT and stopped shared-bank boundaries.
- [ ] Write `lm_loop_shared_kv_is_distinct`: deterministic nondegenerate fixture differs from `PerIteration` at K>1; K=1 is bitwise equal.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_shared_kv_`; expect unsupported policy.
- [ ] Implement first-loop source references and nonlocal producer adjoints; detach producers outside each retained truncation interval.
- [ ] Run the same filter; expect green, one physical core bank per unique core sublayer, and no re-rotation.
- [ ] Stage the four named files; commit as `v0.M.NNN: add explicit shared-first recursive KV semantics`.

### Task L9: Last-loop storage with exact recomputation

**Files:** modify `src/lightnet-lm/src/loop_cache.rs`; create `src/lightnet-lm/tests/lm_loop_last_cache.rs`.

**Interfaces**

- Consumes: L7 prefix/cache APIs and canonical L3 `PerIteration` forward.
- Produces `LastOnlyRecompute` support through unchanged prefill/decode signatures.
- Cache retains bounded IDs, positions, segments, validity, and final-loop core banks; scratch is transient.

- [ ] Write `lm_loop_last_cache_prefix`: K=3 and positions 1–5; every probe/final logit matches canonical full-prefix output within CPU cache tolerance.
- [ ] Write `lm_loop_last_cache_reconstruction`: decode after discarding nonfinal banks reconstructs them; persistent core banks equal C while computed-core-row accounting increases.
- [ ] Write `lm_loop_last_cache_no_substitution`: a fixture where final-bank substitution changes logits still matches canonical recurrence under recomputation.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_last_cache_`; expect unsupported policy.
- [ ] Implement exact prefix reconstruction and transactional append; retain full bounded history even when attention later uses windows.
- [ ] Run the same filter; expect green; report transient and persistent allocations separately.
- [ ] Stage the two named files; commit as `v0.M.NNN: add exact last-loop cache reconstruction`.

### Task L10: Versioned checkpoints and exact resume

**Files:** modify `src/lightnet-lm/src/checkpoint.rs`, `src/lightnet-lm/src/types.rs`, `src/lightnet-lm/src/loop_train.rs`; create `src/lightnet-lm/tests/lm_loop_checkpoint.rs`.

**Interfaces**

- Consumes: Task 6 `save_checkpoint(path: &Path, state: &TrainingSnapshot) -> Result<()>` and `load_checkpoint(path: &Path) -> Result<TrainingSnapshot>`.
- Produces serialization of L1/L6 state and `pub const LOOP_STATE_SCHEMA: u32 = 1`.
- Increments the implemented outer checkpoint format by exactly one; retains a tested reader for the preceding plain format.
- Reuses existing checksums, unique f32 payloads, and atomic publication.

- [ ] Write `lm_loop_checkpoint_roundtrip`: all loop metadata, count/dropout RNGs, pending gradients, moments, counters, loader state, and schedule round-trip exactly.
- [ ] Write `lm_loop_checkpoint_resume`: save after unequal microbatches, restore, and compare the next sampled K, masks, batches, gradients, moments, and successful CPU update bitwise.
- [ ] Write `lm_loop_checkpoint_versions`: preceding plain fixture loads with loop state absent; new format version equals preceding+1; schema 999, bad hashes, duplicate shared tensors, and policy mismatch fail before import.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_checkpoint_`; expect missing loop serialization.
- [ ] Implement the versioned extension; keep a small frozen preceding-format byte fixture inside the test source; serialize each shared tensor once.
- [ ] Run the same filter, then `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_train_`, sequentially; expect green.
- [ ] Stage the four named files; commit as `v0.M.NNN: persist loop-depth state and exact resume`.

### Task L11: Resident oxide fixed-loop forward

**Files:** create `src/lightnet-cuda-oxide/src/lm_loop.rs`, `src/lightnet-cuda-oxide/src/kernels/lm_loop.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_loop_forward.rs`; modify bridge `src/lib.rs` and `src/kernels/mod.rs`.

**Interfaces**

- Consumes: Tasks 8/11 `OxideBackend`, native LM primitives, `LmTransferStats`, and L10 snapshots.
- Produces opaque `DeviceLoopForward`; `OxideLoopDecoder::upload(backend: OxideBackend, state: &EngineState) -> LoopResult<Self>`.
- Produces `forward_loop(&mut self, batch: &TokenBatch, run: &LoopRun) -> LoopResult<DeviceLoopForward>`.
- Produces `download_logits(&self, pass: &DeviceLoopForward) -> LoopResult<Vec<f32>>`, `download_loop_logits(&self, pass: &DeviceLoopForward) -> LoopResult<Vec<Vec<f32>>>`.
- Produces `prefill_loop(&mut self, batch: &TokenBatch, run: &LoopRun) -> LoopResult<DeviceLoopForward>` and `decode_loop(&mut self, step: &TokenBatch) -> LoopResult<DeviceLoopForward>`.
- Produces `transfer_stats(&self) -> LmTransferStats`, `workspace_bytes(&self) -> usize`.
- New kernel proposal: `unsafe fn lm_loop_inject(hidden: *const u16, prelude: *const u16, out: *mut u16, len: usize, scale: f32)`.

- [ ] Write `oxide_lm_loop_forward_reference`: K=1/2/3, all three cache policies, and nonzero-position decode match their bf16-rounded CPU oracles.
- [ ] Write `oxide_lm_loop_forward_resident`: no intermediate D2H transfer; each loop’s explicit probe download is counted; repeated weights have one device working allocation.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_loop_forward`; expect missing bridge.
- [ ] Implement repeated native launches, device tapes, injection, policy-specific banks, and native full-prefix reconstruction for last-only mode.
- [ ] Run the same suite; expect primitive error ≤.02, complete/probe logits ≤.05, and actual-device nonzero test selection.
- [ ] Stage the five named source/test/module files; commit as `v0.M.NNN: add resident bf16 recurrent decoder forward`.

### Task L12: Resident oxide backward and updates

**Files:** modify bridge `src/lm_loop.rs`, `src/kernels/lm_loop.rs`; create `src/lightnet-cuda-oxide/tests/oxide_lm_loop_train.rs`.

**Interfaces**

- Consumes: L11 device tape, L4/L5 adjoint rules, Task 9 update kernels.
- Produces `impl LmEngine for OxideLoopDecoder`.
- Produces `OxideLoopDecoder::download_gradients(&self) -> LoopResult<DecoderGrads>`.
- New kernel proposal: `unsafe fn lm_loop_add_f32(source: *const f32, destination: *mut f32, len: usize)`.
- Supports full and truncated BPTT, shared-first producer reductions, probe CE, and deterministic recomputation.

- [ ] Write `oxide_lm_loop_train_gradients`: every unique family agrees with CPU within .08; full and truncated modes use their respective references.
- [ ] Write `oxide_lm_loop_train_sign_and_sharing`: weighted binary seeds retain target-minus-prediction; tied/core gradients, clipping, decay, and AdamW update occur once.
- [ ] Write `oxide_lm_loop_train_resume_atomic`: same-backend mid-window continuation is exact; NaN changes no committed master, moment, update counter, or weight generation.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_loop_train`; expect missing backward.
- [ ] Implement serialized deterministic shared reductions, f32 accumulation, chunked probe heads, saved-RNG replay, export/import, and device-wide finite-update decisions.
- [ ] Run the suite, then `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_loop_forward`, sequentially; expect green and zero activation/gradient readbacks.
- [ ] Stage the three named files; commit as `v0.M.NNN: add resident recurrent backward and exact updates`.

### Task L13: Sliding windows across loops

**Files:** modify LM `src/loop_cache.rs`, `src/loop_decoder.rs`, `src/attention.rs`; bridge `src/lm_loop.rs`; create CPU `tests/lm_loop_window.rs` and oxide `tests/oxide_lm_loop_window.rs`.

**Interfaces**

- Consumes: decoder Task 14 `window/full_every`, L7–L12 execution.
- Produces `loop_uses_full_attention(depth: &LoopConfig, unique_block: usize, full_every: usize) -> LoopResult<bool>`.
- Extends existing loop forward/cache methods; no new cache policy.

- [ ] Write `lm_loop_window_pattern`: P/C/Z=`1/2/1`, `full_every=3`; unique block 2 is full on every invocation, block 1 is windowed.
- [ ] Write `lm_loop_window_prefix_gradient`: window 3 over six tokens wraps each independent ring; all three policies match their masked-prefix oracle; finite-difference K=2 Q/K/V.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_window_`; expect unsupported window integration.
- [ ] Implement unique-ordinal selection and invocation/source-specific rings with absolute positions.
- [ ] Run the host filter; expect green.
- [ ] Write `oxide_lm_loop_window_reference`: the same masks/gradients match CPU and wholly excluded tiles do not execute.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_loop_window`; observe red, then implement native loop-window dispatch.
- [ ] Run the device suite; expect green; commit the six named files as `v0.M.NNN: preserve window semantics across recurrent depth`.

### Task L14: DSA-style attention across loops

**Files:** modify LM `src/sparse_dsa.rs`, `src/loop_decoder.rs`, `src/loop_cache.rs`; bridge `src/lm_loop.rs`; create CPU `tests/lm_loop_dsa.rs` and oxide `tests/oxide_lm_loop_dsa.rs`.

**Interfaces**

- Consumes: decoder Task 15’s implemented score/select/selected-attention functions; L13.
- Produces `LoopSelectionKey { iteration: usize, unique_block: usize, position: u32, segment: u32, revision: u64 }`.
- Produces `validate_loop_selection(expected: &LoopSelectionKey, actual: &LoopSelectionKey) -> LoopResult<()>`.
- Extends fixed-loop execution only; adaptive DSA remains unsupported.

- [ ] Write `lm_loop_dsa_dense_limit`: `sparse_topk>=prefix_length` matches the corresponding dense loop architecture for all three cache policies.
- [ ] Write `lm_loop_dsa_identity_gradient`: swapped iteration/source selections error; ties use ascending absolute key index; frozen-selection finite differences pass.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_dsa_`; expect unsupported integration.
- [ ] Implement loop/source keys and separately accumulated shared indexer gradients; shared-first K/index keys may share a producer, query selections remain invocation-specific.
- [ ] Run the host filter; expect green.
- [ ] Write `oxide_lm_loop_dsa_reference`: per-loop indices, logits, indexer-loss sign, and gradients match CPU; scratch excludes a full K*T*T score allocation.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_loop_dsa`; observe red, then implement the native ownership adapter.
- [ ] Run the device suite; expect green; commit the six named files as `v0.M.NNN: add invocation-safe recurrent DSA attention`.

### Task L15: NSA-style attention across loops

**Files:** modify LM `src/sparse_nsa.rs`, `src/loop_decoder.rs`, `src/loop_cache.rs`; bridge `src/lm_loop.rs`; create CPU `tests/lm_loop_nsa.rs` and oxide `tests/oxide_lm_loop_nsa.rs`.

**Interfaces**

- Consumes: decoder Task 16’s implemented compression/selection/window/gate operations; L14 selection identity.
- Produces `LoopCompressionKey { iteration: usize, unique_block: usize, segment: u32, completed_end: u32, revision: u64 }`.
- Produces `validate_loop_compression(expected: &LoopCompressionKey, actual: &LoopCompressionKey) -> LoopResult<()>`.
- Extends fixed-loop execution only.

- [ ] Write `lm_loop_nsa_causal_prefix`: block 2, stride 2, window 3; incomplete blocks cannot affect earlier queries; each policy matches its same-representation full-prefix oracle.
- [ ] Write `lm_loop_nsa_branch_gradient`: all branch/gate/compression families pass frozen-selection finite differences; window-only configuration equals L13.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_nsa_`; expect unsupported integration.
- [ ] Implement invocation/source-owned compression state; query branch gates remain separate even when K/V producers are shared.
- [ ] Run the host filter; expect green.
- [ ] Write `oxide_lm_loop_nsa_reference`: group-shared selections, nonzero-position block emissions, logits, and gradients agree with CPU.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_loop_nsa`; observe red, then implement native state ownership.
- [ ] Run the device suite; expect green; commit the six named files as `v0.M.NNN: add invocation-safe recurrent NSA attention`.

### Task L16: A/B/C loop schedules and recipe state

**Files:** create LM `src/loop_recipe.rs`, `tests/lm_loop_recipe.rs`; modify `src/recipe.rs`, `src/recipe_eval.rs`, `src/run.rs`, `src/checkpoint.rs`, the three stage JSON cfgs, and `tasks/rust/llm_recipe_evidence.md`.

**Interfaces**

- Consumes: R1 `Stage`, R3 `RecipeConfig/RecipeSchedule`, R4 `RecipeEngine`, and L6 engine.
- Produces `LoopStageSpec { warmup_targets: u64, warmup_loops: usize, after: LoopSampling, objective: LoopObjective, adjoint: Adjoint, eval_loops: Vec<usize> }`.
- Produces `loop_policy_at(spec: &LoopStageSpec, successful_stage_targets: u64) -> LoopResult<LoopSampling>`.
- Produces `validate_loop_transition(previous: &LoopStageSpec, next: &LoopStageSpec, pending_targets: u64) -> LoopResult<()>`.
- Produces `impl RecipeEngine for LoopEngine` with `eval_token_nll(&mut self, batch: &TokenBatch) -> recipe::Result<Vec<f32>>`.
- Checkpoints include loop-stage origin and spec; evaluation runs fixed K without consuming training RNG.

- [ ] Write `lm_loop_recipe_schedule`: stage budget 1000 gives 30-target depth warmup; counts 29/30 select Fixed(1)/Uniform{1,2}; B/C inherit unless explicitly changed.
- [ ] Write `lm_loop_recipe_sft_mask`: R6’s shifted targets supervise exactly the same assistant/EOT rows at every probe; ten inputs/two targets advance schedule by two.
- [ ] Write `lm_loop_recipe_resume_replay`: resume preserves next replay source, sampled K, LR, exposures, and update; changing depth with pending gradients errors.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_recipe_`; expect missing recipe depth fields.
- [ ] Implement strict optional depth groups, stage-boundary transitions, fixed-depth evaluation, and separate target/core-pass counters; leave baseline stage cfg depth disabled.
- [ ] Run the same filter; expect green, including B forgetting comparisons rejecting differing evaluation depths/policies.
- [ ] Stage every named source/test/cfg path explicitly; commit as `v0.M.NNN: integrate recurrent depth with Latin recipe stages`.

### Task L17: Exit distributions and gate objective

**Files:** create `src/lightnet-lm/src/loop_halting.rs`, `src/lightnet-lm/tests/lm_loop_halting.rs`.

**Interfaces**

- Consumes: L1 errors and L4 loop-probe losses.
- Produces `ExitDistribution { mass: Vec<f64>, cdf: Vec<f64>, expected_loops: f64 }`.
- Produces `exit_distribution(logits: &[f32], budget: usize) -> LoopResult<ExitDistribution>`.
- Produces `exit_depth(distribution: &ExitDistribution, quantile: f64) -> LoopResult<usize>`.
- Produces `gate_objective(logits: &[f32], detached_losses: &[f64], compute_penalty: f64, budget: usize) -> LoopResult<(f64, Vec<f32>)>`.
- Returned gate seeds are negative derivatives of `sum_i p_i*(detached_loss_i + compute_penalty*i)`.

The recurrence, using one-based loops, is:

```text
S0 = 1
lambda_i = sigmoid(logit_i), i < budget
p_i = S(i-1)*lambda_i
S_i = S(i-1)*(1-lambda_i)
p_budget = S(budget-1)
CDF_i = sum_{j<=i} p_j
exit = first i with CDF_i >= quantile, forced to budget if necessary
```

- [ ] Write `lm_loop_halting_mass`: zero logits, budget 4 give `[.5,.25,.125,.125]`; sum is 1 within `1e-12`; quantile .75 exits at 2.
- [ ] Write `lm_loop_halting_budget`: fixed logits/quantile and budgets 1–4 give nondecreasing exit depth; budget one has mass `[1]`.
- [ ] Write `lm_loop_halting_gradient`: finite-difference gate seeds; reject zero budget, invalid quantiles, nonfinite logits/losses, and negative penalty.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_halting_`; expect missing distribution APIs.
- [ ] Implement stable sigmoid/survival and forced remainder; use penalty .01 only as an explicitly selected calibration proposal.
- [ ] Run the same filter; expect green; commit the two named files as `v0.M.NNN: specify deterministic recurrent exit distributions`.

### Task L18: Causal adaptive execution and batched caches

**Files:** modify LM `src/loop_halting.rs`, `src/loop_decoder.rs`, `src/loop_cache.rs`, `src/loop_train.rs`, `src/checkpoint.rs`; create `tests/lm_loop_adaptive.rs`.

**Interfaces**

- Consumes: L17 distributions and L7 per-iteration cache.
- Produces `HaltingGate { weights: Vec<f32>, bias: f32 }`, `AdaptiveRun { budget: usize, quantile: f64 }`.
- Produces `HaltingGate::logits(&self, hidden: &[f32], rows: usize) -> LoopResult<Vec<f32>>` and `backward(&self, hidden: &[f32], logit_seed: &[f32]) -> LoopResult<Vec<f32>>`; gradient order is weights, then bias.
- Produces `AdaptiveForward { forward: LoopForward, exit_depths: Vec<usize>, distributions: Vec<ExitDistribution>, core_rows: u64 }`.
- Produces `LoopDecoder::forward_adaptive(&self, batch: &TokenBatch, gate: &HaltingGate, run: &AdaptiveRun) -> LoopResult<AdaptiveForward>`.
- Produces `prefill_adaptive(&self, batch: &TokenBatch, gate: &HaltingGate, run: &AdaptiveRun, cache: &mut LoopCache) -> LoopResult<AdaptiveForward>` and `decode_adaptive(&self, step: &TokenBatch, gate: &HaltingGate, run: &AdaptiveRun, cache: &mut LoopCache) -> LoopResult<AdaptiveForward>`.
- Adds `ControlSnapshot { kind: String, parameters: Vec<f32>, moments: [Vec<f32>;2], pending_gradient: Vec<f32>, step: u64, rng: RngState }` as optional checkpoint state; strict kind is `halting_v1`.

- [ ] Write `lm_loop_adaptive_no_leakage`: changing future IDs, all targets, or another batch row leaves earlier features, gate logits, and exits unchanged.
- [ ] Write `lm_loop_adaptive_cache_batch`: two rows with differing exits match separate-row/full-prefix masked oracles; absent deeper KV stays invalid; padded lanes append nothing.
- [ ] Write `lm_loop_adaptive_budget_compute`: fresh caches for budgets 1–4, fixed gate/quantile/input; each token’s executed depth and total core rows are nondecreasing.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_adaptive_`; expect missing adaptive execution.
- [ ] Implement recursion-wise active-key masks, frozen exited states, one final coda, batch-isolated metadata, and a +513-parameter gate; support only dense `PerIteration` caches.
- [ ] Run the same filter; expect deterministic exits, unit exit mass, exact controller-state round trip with another format bump, and atomic rejection of unsupported combinations.
- [ ] Stage the six named files; commit as `v0.M.NNN: add causal token exits and consistent adaptive caches`.

Gate calibration freezes the decoder and optimizes L17’s detached probe-loss objective. Probe-based calibration versus final-coda deployment is an acknowledged surrogate, not a quality guarantee. A zero-weight/zero-bias gate is the deterministic initial gate.

### Task L19: Token-choice Mixture-of-Recursions

**Files:** create LM `src/loop_routing.rs`, `tests/lm_loop_routing.rs`; modify `src/loop_decoder.rs`, `src/loop_cache.rs`, `src/checkpoint.rs`.

**Interfaces**

- Consumes: L18 active-row execution and optional encdec Task 31 stable-selection ideas.
- Produces `DepthRouter { weights: Vec<f32>, bias: Vec<f32>, hidden: usize, max_loops: usize }`.
- Produces `DepthRoute { depths: Vec<usize>, counts: Vec<u64> }`.
- Produces `DepthRouter::route(&self, prelude: &[f32], valid: &[bool], budget: usize) -> LoopResult<DepthRoute>`.
- Produces `depth_router_objective(logits: &[f32], detached_losses: &[f64], rows: usize, loops: usize, compute_penalty: f64) -> LoopResult<(f64, Vec<f32>)>`.
- Produces `LoopDecoder::forward_routed(&self, batch: &TokenBatch, router: &DepthRouter, budget: usize) -> LoopResult<AdaptiveForward>`.
- Extends `ControlSnapshot.kind` with `token_choice_v1`; gate and token-choice controller are mutually exclusive.

- [ ] Write `lm_loop_routing_ties_budget`: equal logits choose depth 1; logits favoring depths `[1,3,2]` produce those depths at budget 3 and `[1,2,2]` at budget 2.
- [ ] Write `lm_loop_routing_causal_dispatch`: suffix/target changes cannot alter earlier routing; active-query/key masking equals dense masked execution; no token is dropped by capacity.
- [ ] Write `lm_loop_routing_surrogate_gradient`: softmax expected detached-loss/compute seeds pass finite differences; hard argmax itself has no claimed derivative.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_routing_`; expect missing router.
- [ ] Implement causal prelude-only token choice, stable dispatch, and optional controller-state serialization; reject expert-choice ranking across sequence tokens.
- [ ] Run the same filter; expect green, +1,026 parameters at hidden 512/K=2, deterministic resume, and cache/full-prefix equality under the routed mask.
- [ ] Stage the five named files; commit as `v0.M.NNN: add causal token-choice recursive depth routing`.

Expert-choice top-k across future tokens is deliberately excluded. Encdec Task 33’s expert balancing is not automatically a valid depth-distribution objective; this version uses an explicit compute penalty and reports depth collapse without hiding it.

### Task L20: Optional shared-core MoE composition

**Files:** create LM `src/loop_moe.rs`, `tests/lm_loop_moe.rs`; modify `src/loop_decoder.rs`, `src/loop_train.rs`, `src/checkpoint.rs`.

**Interfaces**

- Prerequisites: encdec Tasks 31–33 and their tensor/tape dependencies green; user selects this optional experiment.
- Consumes their exact `MoeSpec`, `RouterState`, `BalanceSpec`, `moe_ffn`, and `RouterState::observe/finish` interfaces.
- Produces `LoopMoeConfig { shared_experts: usize, routed_experts: usize, top_k: usize, expert_width: usize }`.
- Produces `loop_moe_parameter_count(base: &DecoderConfig, depth: &LoopConfig, moe: &LoopMoeConfig) -> LoopResult<u64>`.
- Produces `LoopDecoder::enable_core_moe(&mut self, spec: LoopMoeConfig, seed: u64) -> LoopResult<()>`.
- Shared expert/router tensors are unique per core sublayer and reused on every loop.

- [ ] Write `lm_loop_moe_count`: pilot core replacement with shared=1, routed=4, top-k=2, width=256 gives exactly **30,559,744** unique parameters at vocab 8192, controllers disabled.
- [ ] Write `lm_loop_moe_tied_dense_reference`: sparse dispatch equals dense-masked execution; repeated-loop expert gradients sum; unselected experts receive zero; all tokens may choose expert zero without drops.
- [ ] Write `lm_loop_moe_balance_transaction`: counts include every actual routed invocation, exclude shared experts/padding, and commit once after a successful update; rejection clears pending counts without changing bias.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_moe_`; expect missing loop MoE adapter.
- [ ] Implement core-only replacement and shared registry/state mapping; maintain separate depth-controller and FFN-expert routing axes.
- [ ] Run the same filter; expect green and schema-bumped round trip; reject device MoE until separately integrated.
- [ ] Stage the five named files; commit as `v0.M.NNN: compose shared-core experts with recurrent depth`.

Exact replacement arithmetic per core block is `786,432+1,024+5*(3*512*256)+512*4 = 2,755,584`; four outer dense blocks plus five such core blocks plus 4,194,816 gives 30,559,744.

### Task L21: Resident adaptive oxide execution

**Files:** modify bridge `src/lm_loop.rs`, `src/kernels/lm_loop.rs`; create `src/lightnet-cuda-oxide/tests/oxide_lm_loop_adaptive.rs`.

**Interfaces**

- Consumes: L18/L19 CPU semantics, L12 engine, native stable gather/prefix-sum primitives when available.
- Produces `OxideLoopDecoder::forward_adaptive(&mut self, batch: &TokenBatch, control: &ControlSnapshot, budget: usize) -> LoopResult<DeviceLoopForward>`.
- Produces `download_exit_depths(&self, pass: &DeviceLoopForward) -> LoopResult<Vec<usize>>`.
- New kernel proposal: `unsafe fn lm_loop_gather_rows(source: *const u16, rows: *const u32, destination: *mut u16, row_count: usize, hidden: usize)`.
- New kernel proposal: `unsafe fn lm_loop_active_mask(depths: *const u32, valid: *const u8, active: *mut u8, rows: usize, iteration: usize)`.
- Device gate/CDF kernels implement L17; controller outputs and active sets stay resident.

- [ ] Write `oxide_lm_loop_adaptive_reference`: halting and token-choice outputs/exits match CPU on fixtures away from thresholds; exact ties follow the declared rule.
- [ ] Write `oxide_lm_loop_adaptive_cache`: batched differing exits, nonzero positions, no-future leakage, and monotone budget row counts match the masked oracle.
- [ ] Write `oxide_lm_loop_adaptive_resume`: controller parameters/moments/RNG and pending state resume exactly on the same deterministic backend.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_loop_adaptive`; expect unsupported adaptive device mode.
- [ ] Implement native gate/projection, stable active-row compaction, recursion-wise KV validity, batched coda, and controller-state import/export.
- [ ] Run the same suite; expect actual-device green, numerical limits from the verification conventions, and no hidden-feature/exit-decision host readback.
- [ ] Stage the three named files; commit as `v0.M.NNN: add resident causal adaptive-depth execution`.

Adaptive+sparse, loop MoE GPU, FP8, cuTile, and recursive preference objectives remain explicit unsupported combinations.

### Task L22: CLI, acceptance matrix, and evidence

**Files:** create LM `tests/lm_loop_acceptance.rs`, `tasks/rust/llm_loop_evidence.md`; modify CLI `src/lm.rs`, `src/lm_recipe.rs`, `CONTEXT.md`; LM `src/run.rs`; `tasks/rust/llm_eval.md`; `CLAUDE.md` only if documented command syntax changes.

**Interfaces**

- Consumes: implemented L1–L21 capabilities and R8/R15 evidence validation.
- Produces `LoopAcceptance { unique_parameters: u64, fixed_depths: Vec<usize>, supported_policies: Vec<CachePolicy>, controller_kind: Option<String> }`.
- Produces `validate_loop_acceptance(record: &LoopAcceptance) -> LoopResult<()>`.
- Proposed CLI flags: `--depth-mode plain|loop`, `--loops <positive integer>`, `--loop-cache per_iteration|shared_first|last_only_recompute`, `--loop-budget <positive integer>`.
- Training sampling comes from the strict recipe, not generation flags. Quantile is checkpoint/config identity, not the monotonic compute knob.
- Existing `cargo xtask lm` forwarding is a prerequisite supplied by decoder Task 6; this task adds no raw execution path.

- [ ] Write `lm_loop_acceptance_counts_modes`: assert nine-block pilot counts; reject false full-prefix equivalence for shared-first against canonical semantics and reject unsupported cross-products.
- [ ] Write CLI `lm_recipe_loop_args`: unknown flags, conflicting fixed/adaptive settings, zero budget, unsupported backend, and controller/hash mismatch error before allocation/cache mutation.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_acceptance_`; expect missing acceptance validator.
- [ ] Implement the validator, strict CLI selection, and TOC-bearing evidence matrix with every unrun measurement marked unverified.
- [ ] Run the host filter, then `MEM=3G JOBS=4 PKGS=lightnet-cli cargo xtask test lm_recipe_loop_`, sequentially; expect green.
- [ ] Run `MEM=3G JOBS=4 cargo xtask check lightnet-lm`, then `MEM=3G JOBS=4 cargo xtask clippy lightnet-lm`, then `cargo xtask fmt`, sequentially; expect zero exits and no unrelated staged formatting.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_loop_`; expect all implemented CPU loop suites green with nonzero selection.
- [ ] Run each implemented oxide loop suite individually through its exact `cargo xtask oxide --test` command; record unavailable/unrun suites explicitly.
- [ ] Stage only the named changed paths; commit as `v0.M.NNN: expose recursive decoder modes and acceptance evidence`.

The evidence matrix records implementation status independently for fixed depth, sampling, truncation, each cache policy, window/DSA/NSA, controllers, CPU, and oxide. Optional unselected tasks are marked unselected, never passed.

## What is worthwhile at the 30-60M Latin pilot vs what only pays off at larger scale

Loop-count-one equivalence, weight-tied unrolled equivalence, fixed K, sampled training K, and per-loop deep supervision are cheap and worthwhile mainly as an **engineering/ablation arm**.

The chosen nine-unique-block pilot has 32,515,584 parameters at vocab 8192. K=2 executes fourteen blocks. Compare it separately with:

- A nine-block plain decoder for matched unique parameters.
- A fourteen-block plain decoder for matched block depth, then measured FLOPs.
- The original twelve-block baseline for continuity with the decoder recipe.

Matching block depth alone does not match total training FLOPs when probes, recomputation, cache policies, or truncation differ. Record those costs, tokenizer, target budget, data exposures, and precision explicitly.

The reported MoR small-scale result trailed vanilla at a 135M base scale; Ouro/Huginn results used at least 1.4B parameters and 7.7T/0.8T tokens. These are **VERIFIED external results**, with sources in the Research basis. No quality gain from looping is expected or established at 30–60M.

A matched-FLOPs comparison and a matched-unique-parameter comparison against the plain decoder are the only honest experimental claims. Neither comparison is authorized by implementing this plan.

Adaptive halting and per-token routing add substantial complexity for a small model. Keep L17–L21 optional and late. DSA/NSA, large expert banks, compressed caches, and deep recurrence need a measured problem before becoming priorities.

GPT-6 Astra’s alleged looping is an **UNVERIFIED** press report and is not a design source.

## Unproven reasoning claims

- **UNVERIFIED:** a 30–60M Latin LM develops useful “latent reasoning” merely by sharing depth.
- **UNVERIFIED:** additional loops monotonically improve perplexity, reasoning, factuality, or Latin grammar. Only the budget’s executed-compute monotonicity is specified.
- **UNVERIFIED:** a halting probability estimates correctness or semantic difficulty. It is a trained exit score.
- **UNVERIFIED:** puzzle-solver results transfer to autoregressive Latin generation.
- **UNVERIFIED:** truncated gradients or probe supervision preserve the full-BPTT optimum.
- **UNVERIFIED:** post-hoc gate calibration on pre-coda probe losses preserves final-coda quality.
- **UNVERIFIED:** approximate KV substitution preserves canonical logits or quality.
- **UNVERIFIED:** parameter sharing improves wall-clock speed. Sequential passes, probes, launch overhead, and replay may increase cost.
- **UNVERIFIED:** recurrent depth improves historical accuracy, quotation honesty, safety, or chain-of-thought monitorability.
- **UNVERIFIED:** hardware fit, usable corpus size, stable maximum depth, best LR, and the preferred vocabulary.

Synthetic tests establish the declared computation and adjoints. They do not establish these claims.

## Progress table

| Task | Deliverable | Status | Required evidence |
|---|---|---|---|
| Foundation 1–7 | CPU decoder prerequisites | Unimplemented scaffold | Original task gates |
| Foundation 8–10 | Resident baseline and pilot software | Unimplemented | Actual-device and bounded tests |
| L1 | Strict cfg and exact counts | Planned | Count/parser assertions |
| L2 | Block boundaries | Planned | Bitwise plain reconstruction |
| L3 | Fixed recurrence | Planned | K=1 and tied-unroll forward |
| L4 | Full BPTT/probes | Planned | All-family finite differences |
| L5 | Truncated adjoint | Planned | Stopped surrogate and d≥K exactness |
| L6 | Sampled-depth engine | Planned | RNG replay and unique updates |
| L7 | Iteration cache | Planned | Every-loop cached/prefix logits |
| L8 | Shared-first architecture | Planned optional policy | Own-oracle logits and producer gradients |
| L9 | Last-only reconstruction | Planned optional policy | Canonical equivalence and allocation accounting |
| L10 | Checkpoint/resume | Planned | Version bump, round trip, exact continuation |
| L11 | Oxide forward/cache | Planned | Device numerics/residency |
| L12 | Oxide backward/update | Planned | Shared gradients, atomicity, resume |
| L13 | Window integration | Planned after Task 14 | Ring wrap and masked equivalence |
| L14 | DSA integration | Optional | Selection identity and gradients |
| L15 | NSA integration | Optional | Block causality and branch gradients |
| L16 | Recipe integration | Planned | A/B/C schedules, replay, SFT masks |
| L17 | Exit distribution | Optional late task | Unit mass, deterministic exits, gradients |
| L18 | Adaptive CPU/cache | Optional late task | Causality, batch consistency, budget compute |
| L19 | Token-choice MoR | Optional late task | Routing determinism and masked oracle |
| L20 | Recurrent MoE | Optional, user-selected | Exact count, sparse equivalence, balancing |
| L21 | Adaptive oxide | Optional late task | Resident dispatch/cache/controller resume |
| L22 | CLI and acceptance | Planned | Capability matrix and focused suites |

No implementation or runtime gate passed during drafting. Corpus runs, quality comparisons, hardware-fit measurements, and benchmarks remain unrun and user-gated.

## Open issues

- **Parameter-count discrepancy:** decoder Task 10 and recipe R9 assert 45,628,928/49,823,232. The specified tensors give 41,955,840/46,150,144, a difference of 3,673,088 at either vocabulary. L1 owns correction; no extra parameters are presumed.
- **Scaffold dependency:** all decoder Tasks 1–16 remain unimplemented. This plan does not replace their tokenizer, primitive, loader, optimizer, or baseline GPU work.
- **Plan boundary conflict:** old decoder core parser/error references conflict with LM independence. Foundation Task 1 must resolve that; loop errors do not import core.
- **Documentation/version conflict:** this request places the common ledgers in a separate docs-only commit; the handbook places three in the code commit. Follow this request for these increments.
- **Pilot minimum:** 2/4/2 at vocab 8192 is 29,368,832, below 30M. The selected 2/5/2 pilot fixes that without changing hidden/intermediate geometry.
- **Alternative encdec geometry:** encdec Task 40 uses two KV heads and intermediate 1360. Its counts cannot be substituted into the four-KV-head, width-1536 arithmetic here.
- **Hardware fit:** persistent weights alone do not establish capacity. Full-BPTT tapes, probe heads, optimizer state, attention scratch, and exact-replay scratch require later allocation evidence.
- **Sparse/adaptive composition:** its active-key and compression rules are not implemented by fixed-loop sparse tests. Keep the combination rejected until separately specified.
- **Gate calibration mismatch:** pre-coda probe losses are a surrogate for adaptive final-coda output. Whether calibration remains useful after active-key changes is open and measurement-gated.
- **Depth-router collapse:** no quality or balanced-depth guarantee is made. Uniform expert-count balancing is not silently imposed on depth choices.
- **Approximate last-step caching:** excluded from exact policies. A future experiment needs its own name, oracle, checkpoint identity, and quality gate.
- **Normalization/injection:** sandwich norms, learned adapters, random initial state, and warm starts remain separate architecture questions; adopting them changes counts and equivalence contracts.
- **FP8 and combined experiments:** loop-specific scaling-history ownership, MoE GPU, MTP, MLA, causal split, cuTile, and DPO combinations remain unsupported until their own tests land.
- **Authorization and data:** corpus rights, inventories, token budgets, G0/GAB/GBC/GCHAT approvals, and Latinist evaluation remain the recipe’s unresolved operational gates.
- **Delegation:** implementation must re-read `tasks/rust/AGENTS_LLM.md` and the live operator override. This drafting request did not dispatch implementation workers.

## Self-Review

- **Spec coverage:** layout, initialization, injection, depth policies, halting, compute budget, BPTT/truncation, supervision, exact counts, RoPE/cache policies, sparse attention, MoR/MoE, stages A/B/C, checkpointing, and all required test classes have owners.
- **Type consistency:** baseline types are prerequisite proposals; loop-owned types precede their consumers. Shared-first KV adjoints and controller snapshots have explicit interfaces.
- **Step scan:** tasks name files, signatures, pinned assertions, red commands, implementation actions, green commands, and explicit-path commit procedures.
- **Placeholder scan:** no undecided implementation placeholders are used; only the required unallocated `v0.M.NNN` version notation remains.
- **Evidence boundary:** this is an unreviewed implementation plan, not compilation, correctness, performance, training, or quality evidence.
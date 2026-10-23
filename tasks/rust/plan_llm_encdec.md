# Encoder, Decoder, and Causal Split Implementation Plan

> Drafted by gpt-6.1-sol (reasoning effort high, Codex CLI, read-only). Unreviewed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. The current repository instruction favors direct execution without subagents; use superpowers:executing-plans unless that instruction changes.

**Goal:** Extend `src/lightnet-lm` with independently usable encoder and decoder stacks, classic encoder-decoder composition, decoder-only reuse, and an explicitly experimental causal encoder-decoder split, with CPU references and separately gated oxide implementations.

**Architecture:** Share differentiable attention, normalization, FFN, parameter, and masking contracts between stacks while keeping their visibility rules explicit. Support three distinct compositions: decoder-only causal LM; standard bidirectional encoder plus causal decoder with cross-attention; and a causal lower stack feeding final-state global K/V to an upper causal stack. Keep GQA and dense SwiGLU as the pilot baseline, with latent attention, sparse attention, MoE, MTP, and FP8 enabled independently.

**Tech Stack:** Rust 2024, repository-pinned `nightly-2026-08-28`, `lightnet-lm`, backend-neutral `lightnet-cuda`, and the excluded `lightnet-cuda-oxide` bridge. Reuse Tasks 1-16 from the existing decoder plan and existing oxide GEMM/precision facilities.

**Spec:** [Existing Tasks 1-16](plan_llm_decoder.md), the user-supplied “DeepSeek V4.1 Flash research notes (2026-10-23),” and repository instructions in `CLAUDE.md` and `tasks/rust/AGENTS.md`. This document adds Tasks 17-40 without renumbering Tasks 1-16.

This is a read-only draft. Repository files were inspected; no files were changed and no builds, checks, tests, downloads, benchmarks, or training were run. Research labels below preserve the supplied notes' evidence level; cited pages and report bodies were not independently re-read for this draft.

## Table of Contents

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Research basis](#research-basis)
  - [Evidence labels](#evidence-labels)
  - [Verified and unverified claims](#verified-and-unverified-claims)
- [Repository starting point](#repository-starting-point)
- [Architecture decisions](#architecture-decisions)
  - [Three composition modes](#three-composition-modes)
  - [Attention families](#attention-families)
  - [Sparse attention and position encoding](#sparse-attention-and-position-encoding)
  - [Normalization, FFN, routing, and prediction](#normalization-ffn-routing-and-prediction)
  - [Precision and cache ownership](#precision-and-cache-ownership)
  - [Pilot scale and larger-scale gates](#pilot-scale-and-larger-scale-gates)
- [Project tree](#project-tree)
- [Component breakdown](#component-breakdown)
- [Mapping to Tasks 1-16](#mapping-to-tasks-1-16)
- [GPU and oxide kernel inventory](#gpu-and-oxide-kernel-inventory)
- [Implementation tasks](#implementation-tasks)
  - [Task 17: Strict composition and tensor contracts](#task-17-strict-composition-and-tensor-contracts)
  - [Task 18: General masked attention and its adjoint](#task-18-general-masked-attention-and-its-adjoint)
  - [Task 19: Partial RoPE, YaRN, and compressed positions](#task-19-partial-rope-yarn-and-compressed-positions)
  - [Task 20: GQA and low-rank MQA projections](#task-20-gqa-and-low-rank-mqa-projections)
  - [Task 21: MLA reference and latent-cache equivalence](#task-21-mla-reference-and-latent-cache-equivalence)
  - [Task 22: Bidirectional and causal encoder stacks](#task-22-bidirectional-and-causal-encoder-stacks)
  - [Task 23: Cross-attention memory projection and ownership](#task-23-cross-attention-memory-projection-and-ownership)
  - [Task 24: Decoder stack with optional cross-attention](#task-24-decoder-stack-with-optional-cross-attention)
  - [Task 25: Classic encoder-decoder composition](#task-25-classic-encoder-decoder-composition)
  - [Task 26: Decoder-only reuse and generation cache](#task-26-decoder-only-reuse-and-generation-cache)
  - [Task 27: Causal split and final-encoder global K/V](#task-27-causal-split-and-final-encoder-global-kv)
  - [Task 28: Incremental causal split and bounded local cache](#task-28-incremental-causal-split-and-bounded-local-cache)
  - [Task 29: Compressed sparse attention reference](#task-29-compressed-sparse-attention-reference)
  - [Task 30: Full, Reindex, and Reuse dependency schedules](#task-30-full-reindex-and-reuse-dependency-schedules)
  - [Task 31: Deterministic top-k router](#task-31-deterministic-top-k-router)
  - [Task 32: Shared and routed expert FFN](#task-32-shared-and-routed-expert-ffn)
  - [Task 33: Explicit auxiliary-loss-free balancing policy](#task-33-explicit-auxiliary-loss-free-balancing-policy)
  - [Task 34: Multi-token prediction objective](#task-34-multi-token-prediction-objective)
  - [Task 35: Composition checkpoint and update lifecycle](#task-35-composition-checkpoint-and-update-lifecycle)
  - [Task 36: Resident oxide encoder and cross-attention](#task-36-resident-oxide-encoder-and-cross-attention)
  - [Task 37: Oxide latent, sparse, and split caches](#task-37-oxide-latent-sparse-and-split-caches)
  - [Task 38: Oxide MoE and MTP](#task-38-oxide-moe-and-mtp)
  - [Task 39: Projection-scoped FP8 mixed precision](#task-39-projection-scoped-fp8-mixed-precision)
  - [Task 40: Pilot configurations and acceptance matrix](#task-40-pilot-configurations-and-acceptance-matrix)
- [Open issues](#open-issues)

## Global Constraints

- This drafting session is read-only: do not modify files, build, test, train, or create commits.
- Future checks/builds/tests/clippy/fmt/oxide commands use `cargo xtask` only. Never invoke raw Cargo build/test/check commands or `systemd-run`.
- Maximum four build threads. Use `JOBS=4` or less; one build/test command at a time.
- Keep `MEM=3G`, which xtask applies as `MemoryHigh=3G`, a soft cap. If its existing runner falls back to uncapped execution, record that fact.
- `src/lightnet-cuda-oxide` is not a workspace member. GPU suites run through `cargo xtask oxide --test <suite>`.
- Host LM filters require `PKGS=lightnet-lm`; the current xtask test default selects core/CLI, not LM.
- Backward is the exact adjoint of the differentiable Rust reference forward, applied to the supplied seed.
- CE seeds are `one_hot(target)-softmax`. Router supervision, MTP, and any optional auxiliary objective use the same descent-direction convention. Add no compensating negation.
- FP8 is an approximate numerical implementation of reference operations. Do not claim that GEMM backward differentiates bit-level rounding or clipping exactly.
- Unknown configuration keys, architectures, attention modes, and unsupported combinations return errors. Never silently substitute identity, dense attention, or CPU execution.
- No C++ parity claims. CPU/device tests establish agreement with the Rust reference.
- No training runs unless the user asks. Synthetic forward/backward and single-update tests are permitted during authorized implementation.
- Preserve detector configuration semantics, frozen LAYA behavior, legacy checkpoint formats, and unrelated working-tree changes.
- Stage only files named by a task. Never use `git add -A`.
- Use commit message `v0.M.NNN: <what>`; this draft allocates no version numbers. Follow repository version selection and local tagging rules during implementation.
- Each task's **Files** also includes the evidence set: `tasks/rust/implemented.md`, `tasks/rust/STATUS.md`, `tasks/rust/todo.md`, and `tasks/rust/version-map.tsv`.
- Update the first three evidence files in the feature commit; update `version-map.tsv` in the separate mapping commit required by the repository handbook.
- `tasks/` is git-excluded. Stage local documents with explicit `git add -f -- <named-path>`.
- Every edited `.md` needs a TOC. Preserve the existing Tasks 1-16 numbering.
- Read affected crate `CONTEXT.md` files before implementation. Keep the pinned cuda-oxide submodule unchanged.
- No implementation is authorized by this draft alone; review the plan before beginning implementation.
- A missing device, skipped test, zero-test filter, timeout, or compilation failure is not a passing verification result.

## Review Focus

1. **Future leakage through compressed/global memory:** causal queries must not see future encoder states, incomplete compression blocks, or reused selections from another query. Owned by Tasks 27, 29, and 30.
2. **Padding and source identity:** all-padding sources produce finite zero attention; changing source/model revision invalidates cross K/V before mutation. Owned by Tasks 18 and 23.
3. **Shared gradient ownership:** shared global K/V, tied embeddings, grouped projections, and selected experts accumulate every contribution exactly once. Owned by Tasks 21, 25, 27, and 32.
4. **Discrete routing and checkpoint replay:** ties are deterministic; frozen-selection gradients are tested away from seams; rejected updates preserve router counts, biases, RNG, and MTP state. Owned by Tasks 31, 33, and 35.
5. **Positions and cache wrap:** absolute positions survive local-ring wrap, YaRN uses identical prefill/decode frequencies, and stale/overflow errors leave caches unchanged. Owned by Tasks 19, 26, and 28.

## Research basis

### Evidence labels

- **VERIFIED:** stated by a cited source according to the supplied notes.
- **VERIFIED-VIA-FETCH:** extracted by the notes' summarizing fetch tool; not independently checked against raw configuration or PDF text.
- **UNVERIFIED:** absent, conflicting, suspicious, or dependent on unread report bodies.
- **DESIGN:** a lightnet decision, not a claim about DeepSeek implementation.

Use these labels in implementation documentation too. “Inspired by” does not establish architectural equivalence.

### Verified and unverified claims

| ID | Label | Claim and planning consequence | Source |
|---|---|---|---|
| R1 | VERIFIED | The supplied notes identify DeepSeek-V4.1-Flash, a September 10, 2026 release with open weights and MIT licensing. This does not establish checkpoint compatibility with lightnet. | [HF model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [launch coverage](https://thenextweb.com/news/deepseek-v4-1-flash-launch-v4-pro-retired-price-cut) |
| R2 | VERIFIED-VIA-FETCH | CED comprises 20 causal encoder layers followed by 20 decoder layers of one causal LM; decoder global K/V comes from final encoder hidden states, rather than a separate projection at every decoder layer. | [V4.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R3 | UNVERIFIED | Exact CED attention equations, residual placement, projection sharing boundaries, local/global mixing, and training schedule were not established. A T5-style bidirectional source encoder with cross-attention is not established by these sources. | [V4.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R4 | VERIFIED-VIA-FETCH | CSA2 names Full, Reindex, and Reuse modes, shared K/V/indexer keys, reused top-k indices, and a hierarchical indexer candidate pool. Exact algorithms remain UNVERIFIED. | [V4.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R5 | VERIFIED-VIA-FETCH | Config reports 40 layers, hidden 5120, 64 query heads, one KV head, head dimension 512, rotary dimension 64, Q rank 1280, O rank 1024, and eight O groups. These fields support a low-rank MQA design family, not an inferred MLA implementation. | [V4.1 card/config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R6 | VERIFIED-VIA-FETCH | Config reports RoPE theta 10000, YaRN factor 16, beta-fast 32, beta-slow 1, original context 65536, compressed theta 160000, sliding window 128, and maximum context 1048576. Exact compressed-position equations remain UNVERIFIED. | [V4.1 card/config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R7 | VERIFIED-VIA-FETCH | Config/card report one shared and 384 routed experts, six selected experts/token, `sqrtsoftplus`, `noaux_tc`, normalized top-k weights, and routed scale 1.5. The precise bias-update mechanism is UNVERIFIED. | [V4.1 card/config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R8 | VERIFIED-VIA-FETCH / UNVERIFIED | Three next-token prediction layers appear in config. Their architecture, loss weights, and relation to DSpark are UNVERIFIED; config presence does not establish an MTP training recipe. | [V4.1 card/config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R9 | VERIFIED-VIA-FETCH / UNVERIFIED | Config reports bf16 dtype, FP8 dynamic activations, 32x32 weight blocks, UE8M0 scales, and FP4 experts. An FP8 training recipe is UNVERIFIED. Pre/post-norm placement is UNVERIFIED; fetched epsilon `1e-20` is suspicious and UNVERIFIED. | [V4.1 card/config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R10 | VERIFIED-VIA-FETCH | Card describes FP4 main KV with E2M1 values and one E4M3 scale/16 channels, about 890 global-cache bytes/token, and SWA bounded replay. Exact replay dependencies and codec implementation remain UNVERIFIED. | [V4.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |
| R11 | VERIFIED-VIA-FETCH / UNVERIFIED | Card describes 196B Engram conditional memory and mHC. Total parameter counts conflict across hub/press; exact mHC kernels and memory mechanisms remain UNVERIFIED here. | [V4.1 card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), [Register coverage](https://www.theregister.com/ai-and-ml/2026/09/11/deepseeks-new-model-sets-a-template-for-powerful-llms-that-run-lean/5295715) |
| R12 | VERIFIED / UNVERIFIED | The supplied V4 abstract notes CSA/HCA, mHC, and million-token context. Internal equations were not read; the reported date/arXiv-prefix discrepancy remains unresolved. | [V4 abstract](https://arxiv.org/abs/2606.19348) |
| R13 | VERIFIED | V3 establishes MLA, DeepSeekMoE, auxiliary-loss-free balancing, and an MTP objective. Detailed training mechanics were not verified by the supplied abstract-only reading. | [V3 report](https://arxiv.org/abs/2412.19437) |
| R14 | VERIFIED / UNVERIFIED | V3.2 describes DSA with a lightning indexer, ReLU-weighted scores, fine-grained selection, and MLA's shared latent/MQA mode. Top-k 2048 and warm-up token counts are UNVERIFIED in these notes. | [V3.2 report](https://arxiv.org/abs/2512.02556), [secondary implementation discussion](https://vllm.ai/blog/deepseek-v3-2) |
| R15 | VERIFIED | NSA combines coarse compression, fine selection, and sliding windows with hardware-oriented trainability. Exact implementation choices still require paper-body review before claiming NSA fidelity. | [NSA paper](https://arxiv.org/abs/2502.11089) |
| R16 | UNVERIFIED | No directly read official V4.1 technical report or confirmed report identifier was obtained. The cited official news URL redirected to chat. | [Official URL from notes](https://www.deepseek.com/en/news/deepseek-v4-1-flash/) |

## Repository starting point

The inspected `src/lightnet-lm` contains `Cargo.toml`, `CONTEXT.md`, and a documentation-only `src/lib.rs`. It is already a workspace member and depends on `lightnet-cuda`; the decoder APIs in Tasks 1-16 are planned interfaces, not implemented facts.

The existing plan's statement that the crate itself is new is stale. Its tokenizer, operations, model, cache, trainer, checkpoint, and GPU modules remain prerequisites. Do not repeat workspace creation or assume any task is complete from its presence in the plan.

`lightnet-lm` must remain independent of `lightnet-core` and concrete GPU backends. Resolve the old plan's core-specific parser/error references at the crate boundary instead of creating a dependency cycle.

The excluded oxide crate currently has `lightnet-core` as a dev-dependency and lacks a normal `lightnet-lm` dependency. Task 8 owns adding the LM dependency; this extension consumes that boundary.

## Architecture decisions

### Three composition modes

**DESIGN: decoder-only reuse**

```text
target IDs -> shared embedding -> causal decoder stack -> final norm -> tied head
```

Reuse the decoder stack with cross-attention disabled. Require logits and gradients to match the original Tasks 3-7 reference after parameter mapping.

**DESIGN: classic encoder-decoder, standard Transformer practice**

```text
source IDs -> embedding -> bidirectional encoder -> final source states
                                                    |
                                      per-decoder-layer projected K/V
                                                    |
shifted target IDs -> embedding -> causal self-attention + cross-attention -> head
```

The source and target are different sequences. Source attention is bidirectional within a valid segment; target self-attention is causal; cross-attention sees all valid source keys assigned to that target segment. Cross K/V is projected once per source/model revision and decoder layer during generation.

**DESIGN: causal encoder + decoder split**

```text
one token stream -> causal lower stack -> FINAL lower hidden states
                           |                         |
                           |                 shared global K/V projection
                           |                         |
                           +----> upper causal stack + causal global-memory attention
                                                       |
                                                      head
```

This preserves the verified CED distinction, **VERIFIED-VIA-FETCH R2**, without claiming the unverified equations in R3. Both halves process the same stream; upper query position `i` sees only global entries with token index `j <= i`. The proposed upper block sums separately projected local-window and global-memory branches before its attention residual; this mixing rule is **DESIGN**, not a verified DeepSeek detail.

### Attention families

| Family | DESIGN contract | Cache representation | Default |
|---|---|---|---|
| GQA | Query heads share ordinary K/V heads; baseline head geometry from Tasks 1-4. | Rotated K and unrotated V per layer. | Pilot default. |
| Low-rank MQA | One KV head, factorized Q projection, grouped O down-projections followed by one O expansion. | One K/V stream, plus local rings where configured. | Optional comparison. |
| MLA | Shared latent `c`, decoupled rotary key, per-head content-key/value up-projections. | Latent `c` plus rotary key; avoid expanded per-head cache. | Larger-context experiment. |

MLA derives from **VERIFIED R13**, while the low-rank MQA configuration family derives from **VERIFIED-VIA-FETCH R5**. They are distinct parameterizations; do not label the V4-family configuration “MLA” merely because both reduce cache cost.

For MLA, retain an expanded dense oracle and an absorbed computation. Content scores use projected queries against cached latents; weighted latents are expanded through value/output factors. Prove equivalence before enabling latent storage.

### Sparse attention and position encoding

Task 14 remains the sliding-window foundation; Tasks 15-16 remain the DSA-style and NSA-style branches. Correct their evidence wording: DSA's existence/mechanism is **VERIFIED R14**, while the specific 2048 selection count is **UNVERIFIED**; NSA's three-part concept is **VERIFIED R15**.

The new compressed path is **DESIGN**, motivated by CSA naming in **VERIFIED R12** and CSA2 configuration in **VERIFIED-VIA-FETCH R4**. It pools completed blocks, selects compressed entries, and combines them with a local window. It is not a faithful CSA2 or HCA reproduction.

Use split-half partial RoPE, with separate content and rotary coordinates. Implement a declared YaRN formula and distinct ordinary/compressed theta values; the reported values are **VERIFIED-VIA-FETCH R6**, but the chosen compression-position rule remains **DESIGN**. Source and target use independent positions; classic cross-attention starts without RoPE across unrelated source/target coordinates.

### Normalization, FFN, routing, and prediction

Use pre-RMSNorm for every attention and FFN branch, plus terminal stack norms. Classic decoder order is self-attention, cross-attention, FFN, each with its own pre-norm and residual. This placement is **DESIGN**; V4.1 placement is **UNVERIFIED R9**.

Dense FFNs use bias-free SwiGLU. Optional MoE adds always-active shared experts and selected routed experts, with deterministic top-k selection and normalized selected weights. Counts and scoring names in **VERIFIED-VIA-FETCH R7** motivate configurable experiments, not pilot defaults.

Implement a local, explicitly specified auxiliary-loss-free balancing policy. Do not identify its bias update as DeepSeek's exact `noaux_tc` algorithm: that mechanism is **UNVERIFIED R7**.

MTP is optional training functionality with explicit shifted targets and weights. Its existence in V3 is **VERIFIED R13**, and three configuration modules in V4.1 are **VERIFIED-VIA-FETCH R8**; the proposed module architecture and weights are **DESIGN**.

### Precision and cache ownership

Retain f32 master parameters, gradients, optimizer moments, losses, normalization reductions, and softmax/LSE. Use bf16 working weights/activations first; enable FP8 only for enumerated projection GEMMs after resident bf16 correctness.

For batch `B`, tokens `T`, layers `L`, element bytes `s`, approximate payloads excluding metadata are:

- GQA: `B*T*L*Hkv*(Dk+Dv)*s`.
- MLA: `B*T*L*(Rkv+Drope)*s`.
- Shared causal-split global bank: `B*T*(Dk+Dv)*s`, plus lower-stack caches and upper local rings.
- Classic cross bank: `B*S*Ldecoder*Hkv*(Dk+Dv)*s`, independent of generated target length.

Use checked byte accounting and bounded allocation. Quantized KV, paging, SSD offload, and replay are independent features, not implied by FP8 projection support.

The bounded replay claim is **VERIFIED-VIA-FETCH R10**, but its exact reconstruction dependencies are **UNVERIFIED**. This plan retains bounded local rings and explicitly rejects `bounded_replay`; replay cannot be assumed correct from token IDs alone when hidden states depend on omitted context.

### Pilot scale and larger-scale gates

For the 30-60M Latin pilot, prioritize dense SwiGLU, GQA, pre-RMSNorm, tied embeddings, ordinary RoPE, bf16 residency, exact masks, and cache correctness. Build the classic encoder path when a source-conditioned Latin task is actually required.

A useful baseline is vocabulary 8192, hidden 512, twelve layers, eight query heads, two KV heads, and FFN width 1360: approximately 37.1M parameters with tied embeddings. Six encoder plus six cross-attending decoder layers is approximately 41.1M with shared embeddings; Task 40 checks the exact registry count.

Causal split, small MoE, MTP, MLA, and low-rank MQA are architecture experiments at this scale. Sliding windows and sparse attention generally do not repay their overhead at 512-1024 tokens; FP8 needs measured end-to-end benefit.

Most V4.1 Flash features do not pay off at 30-60M: 384 experts, 196B Engram, FP4 KV, 1M context, and mHC are inappropriate pilot targets. Their reported existence is **VERIFIED-VIA-FETCH R6-R11**; this scale judgment is **DESIGN**. Do not implement Engram, mHC, FP4 KV, vision, or DSpark in this plan.

## Project tree

`[existing]` means present during inspection. `[new; Tn]` means an absent prerequisite file owned by original Task n. Other `[new]` entries belong to this extension; directories are shown when they contain involved files.

```text
lightnet/                                                   [existing]
|-- Cargo.toml                                              [existing; reference]
|-- Cargo.lock                                              [existing; dependency changes only]
|-- CLAUDE.md                                               [existing; reference]
|-- rust-toolchain.toml                                     [existing; reference]
|-- .cargo/config.toml                                      [existing; xtask alias]
|-- xtask/src/main.rs                                       [existing; runner reference]
|-- tools/
|   |-- test_oxide.sh                                       [existing; runner reference]
|   `-- lib_cap.sh                                          [existing; cap reference]
|-- cfg/
|   |-- llm_latin.cfg                                       [new; T10]
|   |-- llm_latin_seq2seq.cfg                               [new]
|   `-- llm_latin_split.cfg                                 [new]
|-- tasks/rust/
|   |-- AGENTS.md                                           [existing; reference]
|   |-- lessons.md                                          [existing; reference]
|   |-- plan_llm_decoder.md                                 [existing; Tasks 1-16]
|   |-- plan_llm_stacks.md                                  [new; optional saved document]
|   |-- llm_eval.md                                         [new; T10]
|   |-- implemented.md                                      [existing; evidence]
|   |-- STATUS.md                                           [existing; evidence]
|   |-- todo.md                                             [existing; evidence]
|   `-- version-map.tsv                                     [existing; evidence]
|-- src/lightnet-lm/
|   |-- Cargo.toml                                          [existing]
|   |-- CONTEXT.md                                          [existing]
|   |-- src/
|   |   |-- lib.rs                                          [existing; module exports]
|   |   |-- config.rs                                       [new; T1]
|   |   |-- types.rs                                        [new; T1]
|   |   |-- ops.rs                                          [new; T3]
|   |   |-- attention.rs                                    [new; T3]
|   |   |-- decoder.rs                                      [new; T4]
|   |   |-- cache.rs                                        [new; T7]
|   |   |-- optim.rs                                        [new; T6]
|   |   |-- train.rs                                        [new; T6]
|   |   |-- checkpoint.rs                                   [new; T6]
|   |   |-- sparse_dsa.rs                                   [new; T15]
|   |   |-- sparse_nsa.rs                                   [new; T16]
|   |   `-- stacks/                                         [new]
|   |       |-- mod.rs                                      [new]
|   |       |-- contract.rs                                 [new; shapes/config/errors]
|   |       |-- tape.rs                                     [new; reference adjoints]
|   |       |-- masked.rs                                   [new]
|   |       |-- rope.rs                                     [new]
|   |       |-- projection.rs                               [new]
|   |       |-- mla.rs                                      [new]
|   |       |-- encoder.rs                                  [new]
|   |       |-- cross.rs                                    [new]
|   |       |-- decoder.rs                                  [new]
|   |       |-- model.rs                                    [new]
|   |       |-- cache.rs                                    [new]
|   |       |-- split.rs                                    [new]
|   |       |-- compressed.rs                               [new]
|   |       |-- schedule.rs                                 [new]
|   |       |-- router.rs                                   [new]
|   |       |-- moe.rs                                      [new]
|   |       |-- balance.rs                                  [new]
|   |       |-- mtp.rs                                      [new]
|   |       `-- state.rs                                    [new]
|   `-- tests/                                              [new suites]
|       |-- lm_stack_contract.rs                            [new]
|       |-- lm_stack_attention.rs                           [new]
|       |-- lm_stack_rope.rs                                [new]
|       |-- lm_stack_projection.rs                          [new]
|       |-- lm_stack_mla.rs                                 [new]
|       |-- lm_stack_encoder.rs                             [new]
|       |-- lm_stack_cross.rs                               [new]
|       |-- lm_stack_decoder.rs                             [new]
|       |-- lm_stack_seq2seq.rs                             [new]
|       |-- lm_stack_generate.rs                            [new]
|       |-- lm_stack_split.rs                               [new]
|       |-- lm_stack_split_cache.rs                         [new]
|       |-- lm_stack_compressed.rs                          [new]
|       |-- lm_stack_schedule.rs                            [new]
|       |-- lm_stack_router.rs                              [new]
|       |-- lm_stack_moe.rs                                 [new]
|       |-- lm_stack_balance.rs                             [new]
|       |-- lm_stack_mtp.rs                                 [new]
|       |-- lm_stack_state.rs                               [new]
|       `-- lm_stack_pilot.rs                               [new]
`-- src/lightnet-cuda-oxide/
    |-- Cargo.toml                                          [existing; dependency added by T8]
    |-- CONTEXT.md                                          [existing; reference]
    |-- src/
    |   |-- lib.rs                                          [existing]
    |   |-- lm.rs                                           [new; T8]
    |   |-- lm_stacks.rs                                    [new; device orchestration]
    |   |-- kernels/mod.rs                                  [existing]
    |   |-- kernels/lm.rs                                   [new; T8]
    |   |-- kernels/lm_stacks.rs                            [new]
    |   |-- kernels/lm_sparse.rs                            [new]
    |   `-- kernels/lm_experts.rs                           [new]
    `-- tests/
        |-- oxide_lm_stacks.rs                              [new]
        |-- oxide_lm_stack_sparse.rs                        [new]
        |-- oxide_lm_stack_experts.rs                       [new]
        `-- oxide_lm_stack_fp8.rs                           [new]
```

Tokenizer, data, and sampling modules from Tasks 2, 5, and 7 remain unchanged dependencies. Their full file sets remain defined in the original plan.

## Component breakdown

| Component | Role and responsibilities | Depends on / depended on by | Constraints and open questions |
|---|---|---|---|
| Contract and tape | Validate shapes, architecture, positions, and errors; record exact reference adjoints. | Tasks 1, 3, 4 / every new CPU operation. | No backend dependency; explicit input-gradient ordering. |
| Masks and attention | Implement causal, bidirectional, and rectangular cross visibility; stable softmax and shared-KV adjoints. | Contract / every stack and device oracle. | Empty valid-key sets return zero; packed segment correspondence is explicit. |
| RoPE | Rotate only rotary coordinates; compute YaRN frequencies and compressed positions. | Contract / projection, MLA, cache. | Matching absolute positions in prefill/decode; no inferred DeepSeek formula. |
| Projection | Materialize GQA or low-rank MQA Q/K/V and grouped O. | Ops, tape / stacks and GPU projections. | Parameter shapes are explicit; no checkpoint compatibility claim. |
| MLA | Provide expanded oracle and absorbed latent implementation. | Projection, RoPE / optional self/global attention. | Latent cache must actually avoid expanded persistent K/V. |
| Encoder | Bidirectional or causal self-attention blocks and trainable FFNs. | Masks, projections, norm/FFN ops / classic or split model. | Final norm included; causal encoder is not a bidirectional source encoder. |
| Cross memory | Project source K/V per decoder layer; validate source/model identity; backpropagate to source. | Encoder, projections / classic decoder. | Immutable during target generation; source replacement invalidates all projections. |
| Decoder | Causal self-attention, optional cross-attention, dense or expert FFN. | Attention, cross memory / all three compositions. | Disabled cross branch allocates no parameters or extra norm. |
| Model composition | Embeddings, stack wiring, tied head, unique parameter registry. | Encoder, decoder / objectives, cache, checkpoint. | Shared gradients reduced once; separate source/target sequence semantics. |
| Causal split | Project one global bank from final causal encoder states; causal upper consumption. | Model, masks, cache / split generation. | Local/global mixing is a lightnet design; no T5 attribution. |
| Cache | Own lower per-layer, upper local, global, latent, and cross stores. | Model revision, RoPE / generation and device cache. | Checked capacity and byte counts; transactional validation. |
| Sparse/compressed | Pool complete blocks, select candidates, apply source/reuse schedules. | Tasks 14-16, masks / optional long-context stacks. | Dense-equivalence claims specify the same representation and mask. |
| Router/MoE/balance | Deterministic scoring, dispatch/combine, shared experts, staged bias updates. | Tape, update lifecycle / optional FFNs. | No token drops; actual sparse dispatch; exact `noaux_tc` mechanism unverified. |
| MTP | Build future-token targets, independent prediction modules, weighted descent seeds. | Model, CE / optional training objective. | Padding/document boundaries apply at every horizon. |
| State/training | Serialize configuration, masters, moments, pending gradients/counts, RNG, and precision state. | Task 6, all optional modules / resume and update. | Rejected update mutates no committed state; schema mismatch errors. |
| Oxide | Execute native stack operations and gradients with resident state. | Tasks 8-12, CPU contracts / device engine. | No activation/gradient host fallback; unsupported modes error. |
| Pilot/evaluation | Declare model sizes and evidence gates. | All required baseline tasks / operator decisions. | No quality, capacity, or speed claim from static planning. |

## Mapping to Tasks 1-16

| Existing task | Reused / changed / new dependency |
|---|---|
| 1: contracts | Reuse integer tokens and validation; add composition contracts in Task 17. Do not recreate the existing crate. |
| 2: tokenizer | Reuse artifact and normalization identity for source and target; no tokenizer change. |
| 3: operations | Reuse RMSNorm, SwiGLU, embedding, and CE; generalize attention through Tasks 18-21. |
| 4: CPU decoder | Keep as decoder-only oracle; reuse unique parameter conventions and tied head. |
| 5: data | Reuse single-stream packing; Task 25 adds paired-batch validation, not an unrequested corpus pipeline. |
| 6: training/checkpoint | Reuse optimizer and accumulation; Task 35 adds composition/optional-state schema and engine adapter. |
| 7: generation | Reuse sampling; Tasks 23, 26, and 28 add cross/global/latent cache ownership. |
| 8: bf16 forward | Required before Task 36; reuse native primitives and transfer counters. |
| 9: backward/update | Required before device training extension; maintain f32 unique gradient ownership. |
| 10: Latin pilot | Reuse evaluation discipline; Task 40 adds configurations without running training. |
| 11: flash/CE | Reuse tiled attention and chunked tied-head CE; extend rectangular and bidirectional masks. |
| 12: FP8 | Required before Task 39; extend projection roles and checkpoint histories. |
| 13: cuTile | Unchanged optional decoder-only inference; new stack modes return unsupported until separately implemented. |
| 14: window | Required for split local rings and sparse windows; causal and bidirectional windows remain distinct. |
| 15: DSA | Reuse indexer/top-k machinery; fix evidence labels and expose typed adapter in Task 30. |
| 16: NSA | Retain three-branch implementation; Task 29 adds a separate compressed variant, not an NSA rename. |

Baseline execution can begin after Tasks 1-7. Sparse, device, and FP8 stages additionally require their listed original tasks; completing Tasks 1-16 is not a prerequisite for every new CPU task.

## GPU and oxide kernel inventory

| Kernel family | Required operations | Owner |
|---|---|---|
| Existing LM primitives | Embedding/scatter-add, RMSNorm/adjoint, SwiGLU/adjoint, residual/dropout, tied head and chunked CE. | Tasks 8-11; reused by 36. |
| General attention | Bidirectional/causal/rectangular flash forward, LSE, backward, padding/segment masks, shared-KV reduction. | Task 36. |
| Rotary | Partial split-half rotations and transpose; YaRN frequency application; separate compressed positions. | Task 36. |
| Projection | Low-rank Q, MQA K/V, grouped O down/expand, all GEMM adjoints. | Task 37. |
| Latent attention | Latent score contraction, rotary score addition, latent value reduction, output expansion, latent/rotary cache append. | Task 37. |
| Split/cross cache | Per-source projection, immutable bank reads, shared-global gradient reduction, per-layer local-ring read/write. | Tasks 36-37. |
| Compressed attention | Block pool/adjoint, completed-block emission, tiled index scores, deterministic top-k, gather/scatter, sparse attention. | Task 37. |
| Sparse schedules | Candidate filtering, absolute-index validation, source/reuse dependency application. | Task 37. |
| Experts | Stable score/top-k, histogram/prefix-sum dispatch, gather, grouped GEMMs, weighted combine, inverse scatter, router adjoint. | Task 38. |
| Balancing/MTP | Device counts and commit/discard, bias update, horizon target masks, MTP projections and CE seeds. | Task 38. |
| FP8 | Per-role amax/scales, quantization, eligible GEMMs, generation invalidation, device non-finite flags. | Task 39. |

Prefer existing GEMM/conversion facilities. Introduce no FP4 KV, Engram, mHC, or speculative-decoding kernels.

## Implementation tasks

The following proposed APIs are not existing implementation claims. `DecoderConfig`, `TokenBatch`, `DecoderParams`, `DecoderGrads`, `LossSummary`, `RngState`, `EngineState`, and `UpdateOptions` are prerequisites defined by original Tasks 1, 4, and 6.

Reference gradient checks use centered differences with `epsilon=1e-3` and tolerance `2e-3 + 2e-2*abs(reference)`. Freeze discrete selections and keep fixtures away from top-k ties and piecewise seams. For an objective loss, compare descent seeds against the negative numerical derivative; for an arbitrary supplied seed, compare adjoints against the derivative of its inner product with the output.

Every commit step stages exactly that task's named implementation/test files and the Global Constraints evidence set, then follows the local tag/mapping procedure. No version is allocated here.

### Task 17: Strict composition and tensor contracts

**Files**
- Create: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/contract.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Modify: `src/lightnet-lm/src/lib.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_contract.rs`.

**Interfaces**
- Consumes: `TokenBatch::validate(&self, cfg: &DecoderConfig) -> Result<()>`.
- Produces: the following public types; `LmError` implements `Error`, and `LmResult<T> = std::result::Result<T, LmError>`.

```rust
pub enum LmError { Config(String), Shape(String), State(String), NonFinite(String), Unsupported(String) }
pub struct Tensor { pub shape: Vec<usize>, pub data: Vec<f32> }
pub struct TokenLayout {
    pub batch: usize, pub time: usize, pub valid: Vec<bool>,
    pub segments: Vec<u32>, pub positions: Vec<u32>,
}
pub enum Architecture { DecoderOnly, EncoderDecoder, CausalSplit }
pub enum StackRole { BidirectionalEncoder, CausalEncoder, Decoder }
pub enum Visibility { Causal, Bidirectional, Cross }
pub struct HeadSpec {
    pub q_heads: usize, pub kv_heads: usize,
    pub content_dim: usize, pub rope_dim: usize, pub value_dim: usize,
}
pub enum AttentionSpec {
    Gqa(HeadSpec),
    LowRankMqa { heads: HeadSpec, q_rank: usize, o_rank: usize, o_groups: usize },
    Mla { heads: HeadSpec, kv_rank: usize },
}
pub struct YarnSpec {
    pub factor: f32, pub original_context: usize,
    pub beta_fast: f32, pub beta_slow: f32,
}
pub struct RopeSpec {
    pub theta: f32, pub compress_theta: f32, pub yarn: Option<YarnSpec>,
}
pub struct StackConfig {
    pub base: DecoderConfig, pub role: StackRole,
    pub attention: AttentionSpec, pub rope: RopeSpec,
}
pub struct ModelConfig {
    pub base: DecoderConfig, pub architecture: Architecture,
    pub encoder_layers: usize, pub decoder_layers: usize,
    pub attention: AttentionSpec, pub rope: RopeSpec,
}
pub struct ModelInput { pub source: Option<TokenBatch>, pub target: TokenBatch }
pub struct Eval { pub value: Tensor, pub tape: Tape }
pub struct Tape { /* private owned operation graph */ }
```

- Produces: `Tensor::new(shape: Vec<usize>, data: Vec<f32>) -> LmResult<Self>`, `ModelConfig::from_cfg(text: &str) -> LmResult<Self>`, and `Tape::backward(&self, seed: &Tensor) -> LmResult<Vec<Tensor>>`.
- Tape input ordering is declared by each operation; implement a private identity node here and extend its closed operation enum in subsequent tasks.

- [ ] Write `lm_arch17_contract`: assert `[2,3]` with six values succeeds; five values, overflowing dimensions, unknown `architecture=ced_t5`, `kv_heads=0`, and odd rotary width 3 error.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch17_`; expect missing `stacks` imports/APIs.
- [ ] Implement the signatures above; validate architecture layer counts, positive finite theta/epsilon, head divisibility, O groups/ranks, and exact accepted keys. Decoder-only requires zero encoder layers.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch17_`; expect PASS with nonzero selected-test count.
- [ ] Commit the named files as `v0.M.NNN: define strict stack composition contracts`.

### Task 18: General masked attention and its adjoint

**Files**
- Create: `src/lightnet-lm/src/stacks/masked.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_attention.rs`.

**Interfaces**
- Consumes: `Tensor::new(...)`, `Tape::backward(...)`.
- Produces: `Mask { pub query: TokenLayout, pub key: TokenLayout, pub visibility: Visibility, pub window: Option<usize> }`.
- Produces: `masked_attention(q: &Tensor, k: &Tensor, v: &Tensor, heads: &HeadSpec, mask: &Mask) -> LmResult<Eval>`.
- Adjoint order: `[dQ,dK,dV]`; Q/K dimensions use `content_dim+rope_dim`, while V uses `value_dim`.

- [ ] Write `lm_arch18_visibility`: zero Q/K, V `[2,6,10]` gives causal outputs `[2,4,6]` and bidirectional outputs `[6,6,6]`; cross with validity `[true,false,true]` gives 6.
- [ ] Add `lm_arch18_padding_gradient`: all-invalid keys yield exactly zero finite outputs/gradients; changing a padded value from 6 to 600 changes no valid output. Finite-difference Q/K/V with two Q heads sharing one KV head.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch18_`; expect missing mask/attention APIs.
- [ ] Implement stable unclamped attention, segment equality, causal token-order masking, separate rotary positions, and shared-KV gradient summation.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch18_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add bidirectional causal and cross attention references`.

### Task 19: Partial RoPE, YaRN, and compressed positions

**Files**
- Create: `src/lightnet-lm/src/stacks/rope.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_rope.rs`.

**Interfaces**
- Consumes: `RopeSpec`, `YarnSpec`, `HeadSpec`, and `Tape::backward(...)`.
- Produces: `rope_frequencies(rotary_dim: usize, spec: &RopeSpec, compressed: bool) -> LmResult<Vec<f32>>`.
- Produces: `apply_rope(x: &Tensor, positions: &[u32], content_dim: usize, frequencies: &[f32]) -> LmResult<Eval>`; adjoint order `[dX]`.

- [ ] Write `lm_arch19_rope`: width 4/theta 10000 gives unscaled frequencies `[1,0.01]`; position zero preserves input; content coordinates `[7,9]` remain unchanged.
- [ ] Add `lm_arch19_yarn_cache_positions`: factor 1 equals ordinary RoPE; full positions `[127,128,129]` equal concatenated one-token rotations; compressed theta 160000 gives second frequency `0.0025` for width 4.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch19_`; expect missing rotary functions.
- [ ] Implement split-half rotation and transpose. For YaRN, blend base and factor-divided frequencies with a dimension ramp between correction indices `d*ln(N/(2*pi*beta))/(2*ln(theta))`; clamp indices and use fast-to-slow interpolation. Keep attention magnitude scaling explicitly 1 in this local variant.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch19_`; expect PASS including finite differences at position 129.
- [ ] Commit the named files as `v0.M.NNN: add explicit partial rope and yarn frequency policies`.

### Task 20: GQA and low-rank MQA projections

**Files**
- Create: `src/lightnet-lm/src/stacks/projection.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_projection.rs`.

**Interfaces**
- Consumes: `AttentionSpec`, `Tensor`, `Tape::backward(...)`.
- Produces: `project_qkv(x: &Tensor, weights: &[Tensor], spec: &AttentionSpec) -> LmResult<Vec<Eval>>`.
- Produces: `project_output(context: &Tensor, weights: &[Tensor], spec: &AttentionSpec, hidden: usize) -> LmResult<Eval>`.
- GQA weights are `[Wq,Wk,Wv]` and `[Wo]`; low-rank MQA weights are `[Wq_down,Wq_up,Wk,Wv]` and `[group_O_down...,O_up]`. Each tape returns input gradient followed by supplied-weight gradients.

- [ ] Write `lm_arch20_factorization`: factors `[[1],[2]]` and `[[3,4]]` materialize `[[3,4],[6,8]]`; factorized Q equals dense Q within `1e-6`.
- [ ] Add `lm_arch20_grouped_o`: two groups with rank 1 equal a materialized dense O; finite-difference every factor and reject three groups for four heads.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch20_`; expect missing projection APIs.
- [ ] Implement GQA and low-rank MQA only; MLA returns explicit unsupported here. Group heads contiguously, down-project each group to `o_rank/o_groups`, concatenate, then expand to hidden.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch20_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add gqa and factorized mqa projections`.

### Task 21: MLA reference and latent-cache equivalence

**Files**
- Create: `src/lightnet-lm/src/stacks/mla.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_mla.rs`.

**Interfaces**
- Consumes: `masked_attention(...)`, `apply_rope(...)`, and Task 20 Q/output projection conventions.
- Produces: `MlaKv { pub latent: Tensor, pub rotary_key: Tensor }`.
- Produces: `mla_project(x: &Tensor, weights: &[Tensor], rank: usize, rotary_dim: usize) -> LmResult<(MlaKv, Vec<Eval>)>`.
- Produces: `mla_attention(q: &Tensor, kv: &MlaKv, up_k: &Tensor, up_v: &Tensor, mask: &Mask, expanded: bool) -> LmResult<Eval>`; adjoint order `[dQ,dLatent,dRotaryKey,dUpK,dUpV]`.

- [ ] Write `lm_arch21_absorption`: two query heads, latent rank 2, content width 2, rotary width 2, value width 3; expanded and absorbed outputs agree within `1e-5`.
- [ ] Add `lm_arch21_gradients_cache`: compare all five adjoints against finite differences; cached latent-plus-rotary storage is `T*(2+2)` values rather than expanded `T*2*(4+3)`.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch21_`; expect missing MLA types/functions.
- [ ] Implement `c=x*Wdown`, separate rotary keys, per-head key/value expansion, and absorbed content-score/value contractions. Retain identical `1/sqrt(content_dim+rope_dim)` score scaling in both paths.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch21_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add mla latent reference and absorption checks`.

### Task 22: Bidirectional and causal encoder stacks

**Files**
- Create: `src/lightnet-lm/src/stacks/encoder.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_encoder.rs`.

**Interfaces**
- Consumes: Tasks 18-21 and original Task 3 `rms_norm(...)`/`swiglu(...)` and adjoints.
- Produces: `Encoder::new(cfg: StackConfig, seed: u64) -> LmResult<Self>`, `Encoder::forward(&self, x: &Tensor, layout: &TokenLayout) -> LmResult<Eval>`.
- Produces: `Encoder::parameters(&self) -> &DecoderParams`, `Encoder::set_parameters(&mut self, params: DecoderParams) -> LmResult<()>`.
- Stack tape returns `[dX, parameter gradients in registry order...]`; constructor accepts only encoder roles and gives every parameter a stable name.

- [ ] Write `lm_arch22_encoder_visibility`: changing token 2 changes token 0 in a bidirectional fixture; it leaves token 0 unchanged in causal mode.
- [ ] Add `lm_arch22_encoder_gradient`: hidden 8, two layers, two Q heads/one KV head; finite-difference each parameter family and assert padded rows remain zero after final norm.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch22_`; expect missing `Encoder`.
- [ ] Implement pre-RMSNorm self-attention/residual and pre-RMSNorm SwiGLU/residual, terminal norm, explicit padding-zero restoration, and deterministic initialization from Task 4.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch22_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add trainable bidirectional and causal encoder stacks`.

### Task 23: Cross-attention memory projection and ownership

**Files**
- Create: `src/lightnet-lm/src/stacks/cross.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_cross.rs`.

**Interfaces**
- Consumes: `Tensor`, `TokenLayout`, `HeadSpec`, and `masked_attention(...)`.
- Produces: `CrossBank::build(source: &Tensor, layout: &TokenLayout, kv_weights: &[Tensor], heads: &HeadSpec, source_revision: u64, model_revision: u64) -> LmResult<Self>`.
- Produces: `CrossBank::layer(&self, layer: usize) -> LmResult<(&Tensor, &Tensor)>`, `CrossBank::validate(&self, source_revision: u64, model_revision: u64) -> LmResult<()>`.
- Produces: `CrossBank::adjoint(&self, kv_seeds: &[Tensor]) -> LmResult<Vec<Tensor>>`; weights/seeds order is `[K0,V0,K1,V1,...]`, output order `[dSource,dWk0,dWv0,...]`.

- [ ] Write `lm_arch23_cross_bank`: source length 3, target length 2, two decoder layers produce four cached projection tensors; repeated target calls perform no new source projections.
- [ ] Add `lm_arch23_cross_identity`: source revision 7/model revision 11 validates; revisions 8 or 12 error without mutation. Two layer seeds sum into the same source gradient; finite-difference source and weights.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch23_`; expect missing `CrossBank`.
- [ ] Implement immutable projected memory, stored source layout, checked layer indexing, and one summed source adjoint. Use no source/target RoPE coupling.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch23_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add revision checked cross attention memory`.

### Task 24: Decoder stack with optional cross-attention

**Files**
- Create: `src/lightnet-lm/src/stacks/decoder.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_decoder.rs`.

**Interfaces**
- Consumes: Task 22 block conventions and `CrossBank::build(...)`/`adjoint(...)`.
- Produces: `Decoder::new(cfg: StackConfig, cross_attention: bool, seed: u64) -> LmResult<Self>`.
- Produces: `Decoder::forward(&self, x: &Tensor, layout: &TokenLayout, source: Option<(&Tensor, &TokenLayout)>) -> LmResult<Eval>`.
- Produces: `Decoder::parameters(&self) -> &DecoderParams`, `Decoder::set_parameters(&mut self, params: DecoderParams) -> LmResult<()>`.
- Tape ordering is `[dX,parameters...]` without source, `[dX,dSource,parameters...]` with source.

- [ ] Write `lm_arch24_decoder`: future target changes leave earlier logits unchanged; a valid source change affects decoder output; changing padded source entries does not.
- [ ] Add `lm_arch24_disabled_cross`: `cross_attention=false` owns zero cross parameters; supplying a source errors. Enabled mode with no source errors; all cross parameter families pass finite differences.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch24_`; expect missing `Decoder`.
- [ ] Implement self-attention, optional cross-attention, and FFN with separate pre-norms/residuals. Cross attention initially supports ordinary GQA K/V; unsupported MLA-cross configurations error.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch24_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add decoder stack with optional cross attention`.

### Task 25: Classic encoder-decoder composition

**Files**
- Create: `src/lightnet-lm/src/stacks/model.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_seq2seq.rs`.

**Interfaces**
- Consumes: `Encoder::forward(...)`, `Decoder::forward(...)`, original embedding and tied-head parameter contracts.
- Produces: `CpuModel::new(cfg: ModelConfig, seed: u64) -> LmResult<Self>`, `CpuModel::forward(&self, input: &ModelInput) -> LmResult<Eval>`.
- Produces: `CpuModel::parameters(&self) -> &DecoderParams`, `CpuModel::set_parameters(&mut self, params: DecoderParams) -> LmResult<()>`, `CpuModel::revision(&self) -> u64`.
- Model tape returns only unique parameter gradients; source/target token IDs are nondifferentiable.

- [ ] Write `lm_arch25_seq2seq_gradient`: vocabulary 16, hidden 8, one encoder and one decoder layer; finite-difference encoder, cross, decoder, embedding, and tied-head contributions.
- [ ] Add `lm_arch25_pairs`: source segments `[4,4,9]` and target segments `[4,9]` preserve correspondence; source length 3/target length 2 succeeds; target segment 7 with no valid source errors.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch25_`; expect missing `CpuModel`.
- [ ] Implement shared source/target embeddings and tied output storage, BOS-shift validation, unique registry remapping, and complete encoder-to-loss backpropagation. Initially accept only `EncoderDecoder`.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch25_`; expect PASS and exactly one embedding parameter.
- [ ] Commit the named files as `v0.M.NNN: compose standard encoder decoder model`.

### Task 26: Decoder-only reuse and generation cache

**Files**
- Create: `src/lightnet-lm/src/stacks/cache.rs`.
- Modify: `src/lightnet-lm/src/stacks/model.rs`, `src/lightnet-lm/src/stacks/decoder.rs`, `src/lightnet-lm/src/stacks/mod.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_generate.rs`.

**Interfaces**
- Consumes: `CpuModel::forward(...)`, original `CpuDecoder::forward(...)`, Task 7 and Task 14 cache semantics.
- Produces: `ModelCache::new(cfg: &ModelConfig, capacity: usize) -> LmResult<Self>`, `ModelCache::reset(&mut self)`, `ModelCache::bytes(&self) -> u64`.
- Produces: `CpuModel::prefill(&self, input: &ModelInput, cache: &mut ModelCache) -> LmResult<Tensor>`, `CpuModel::decode(&self, token: u32, cache: &mut ModelCache) -> LmResult<Tensor>`.
- Generation is batch one; classic mode stores one CrossBank and independent growing target self-cache.

- [ ] Write `lm_arch26_decoder_reuse`: map a two-layer GQA Task 4 model into decoder-only mode; logits and every parameter gradient agree within `1e-5`.
- [ ] Add `lm_arch26_cache_prefix`: tokens `[1,5,6,7,8,9]`, capacity 8, local window 3; cached last logits match full-prefix logits through ring wrap. Repeat for classic mode with source `[4,5,0]`, last source token padded.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch26_`; expect missing cache APIs.
- [ ] Implement decoder-only architecture and both modes' incremental projections; cache source/model identity, absolute positions, and explicit reset. Validate capacity, token IDs, and revisions before writes.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch26_`; expect PASS, including stale-cache and capacity-9 append rejection without mutation.
- [ ] Commit the named files as `v0.M.NNN: reuse decoder stack with checked incremental caches`.

### Task 27: Causal split and final-encoder global K/V

**Files**
- Create: `src/lightnet-lm/src/stacks/split.rs`.
- Modify: `src/lightnet-lm/src/stacks/model.rs`, `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_split.rs`.

**Interfaces**
- Consumes: causal `Encoder::forward(...)`, Task 18 causal masks, Task 20 projection operations.
- Produces: `GlobalBank { pub key: Tensor, pub value: Tensor, pub layout: TokenLayout }`.
- Produces: `project_global(final_encoder: &Tensor, layout: &TokenLayout, wk: &Tensor, wv: &Tensor) -> LmResult<(GlobalBank, Vec<Eval>)>`.
- Produces: `split_forward(model: &CpuModel, batch: &TokenBatch) -> LmResult<Eval>`; model tape follows unique parameter order.

- [ ] Write `lm_arch27_split_source`: two lower/two upper layers own exactly one global Wk and one global Wv; projection consumes final normalized lower states, not layer-0 or upper states.
- [ ] Add `lm_arch27_split_causal_gradient`: changing token 3 leaves positions 0-2 unchanged; sum upper-layer bank seeds into one projection adjoint; finite-difference lower, shared-bank, upper-query, and output parameters.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch27_`; expect unsupported `CausalSplit`.
- [ ] Implement same-stream composition, causal global masks, and upper local/global attention sum before the attention residual. Reject separate `source` input and incompatible upper head geometry.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch27_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add causal split with final encoder global memory`.

### Task 28: Incremental causal split and bounded local cache

**Files**
- Modify: `src/lightnet-lm/src/stacks/cache.rs`, `src/lightnet-lm/src/stacks/split.rs`, `src/lightnet-lm/src/stacks/model.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_split_cache.rs`.

**Interfaces**
- Consumes: `CpuModel::prefill(...)`/`decode(...)`, `GlobalBank`, Task 14 absolute-position rings.
- Produces: `CacheStats { pub global_entries: usize, pub lower_entries: Vec<usize>, pub upper_local_entries: Vec<usize>, pub payload_bytes: u64 }`.
- Produces: `ModelCache::stats(&self) -> CacheStats`; extends existing generation methods to `CausalSplit`.

- [ ] Write `lm_arch28_split_cache`: decode nine tokens with window 3; global entries equal 9, each upper local store holds 3, and lower full-cache layers hold 9.
- [ ] Add `lm_arch28_split_prefix`: each cached next-token logit agrees with full-prefix split evaluation within `1e-5`; positions 127-130 remain absolute across ring wrap.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch28_`; expect split cache unsupported.
- [ ] Implement lower incremental forward, one new final-state global projection/token, upper global reads and local ring writes. Reject `bounded_replay` and atomically reject stale/overflow operations.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch28_`; expect PASS, including unchanged snapshots after invalid token/revision/capacity requests.
- [ ] Commit the named files as `v0.M.NNN: add causal split incremental global and local caches`.

### Task 29: Compressed sparse attention reference

**Files**
- Create: `src/lightnet-lm/src/stacks/compressed.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`, `src/lightnet-lm/src/stacks/cache.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_compressed.rs`.

**Interfaces**
- Consumes: Task 14 windows, Task 19 compressed frequencies, Task 18 masked attention.
- Produces: `CompressionSpec { pub block: usize, pub stride: usize, pub selected: usize, pub window: usize }`.
- Produces: `CompressedMemory { pub states: Tensor, pub layout: TokenLayout, pub block_ends: Vec<u32> }`.
- Produces: `compress_blocks(x: &Tensor, layout: &TokenLayout, logits: &Tensor, spec: &CompressionSpec) -> LmResult<(CompressedMemory, Eval)>`; tape order `[dX,dLogits]`.
- Produces: `compressed_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Mask, spec: &CompressionSpec, pool_logits: &Tensor) -> LmResult<Eval>`; tape order `[dQ,dK,dV,dPoolLogits]`.

- [ ] Write `lm_arch29_pool`: block 2/stride 2 and zero pool logits turn states `[2,6,10,14]` into `[4,12]` with ends `[1,3]`; query 0 sees neither completed block.
- [ ] Add `lm_arch29_dense_reference`: block 1, window 0, selected at least T, equal theta values reproduce dense causal attention within `1e-6`; block 2 matches dense materialization of the compressed representation, not original-token dense attention.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch29_`; expect missing compression APIs.
- [ ] Implement learned softmax-weighted block pooling, no cross-segment blocks, endpoint compressed positions, deterministic selection, and joint softmax over selected compressed entries plus raw-window entries. Deduplicate identical raw entries; test gradients with frozen selection.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch29_`; expect PASS, including incomplete-block future perturbations and cache-vs-prefix after two block emissions.
- [ ] Commit the named files as `v0.M.NNN: add explicit compressed sparse attention reference`.

### Task 30: Full, Reindex, and Reuse dependency schedules

**Files**
- Create: `src/lightnet-lm/src/stacks/schedule.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/model.rs`, `src/lightnet-lm/src/sparse_dsa.rs`, `src/lightnet-lm/src/sparse_nsa.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_schedule.rs`.

**Interfaces**
- Consumes: Tasks 15-16 indexer/selection behavior and Task 29 compressed memory.
- Produces: `SourceMode { Full, Reindex { kv_layer: usize }, Reuse { kv_layer: usize, index_layer: usize } }`.
- Produces: `SparseSchedule { pub layers: Vec<SourceMode>, pub candidate_layer: Option<usize>, pub candidate_blocks: usize }`.
- Produces: `SparseSchedule::validate(&self) -> LmResult<()>`, `select_indices(scores: &[f32], allowed: &[bool], k: usize) -> LmResult<Vec<u32>>`.
- Produces: `SelectionCache::new() -> Self`, `SelectionCache::get(&self, layer: usize, position: u32, segment: u32, revision: u64) -> LmResult<&[u32]>`.

- [ ] Write `lm_arch30_schedule`: `[Full,Reindex{kv_layer:0},Reuse{kv_layer:0,index_layer:1}]` validates; self-reference, future-reference, and candidate source layer 9 in a three-layer schedule error.
- [ ] Add `lm_arch30_topk_identity`: scores `[2,2,1]`, all allowed, k=2 select `[0,1]`; changing query position, segment, or revision rejects cached selection. k covering every allowed key reproduces dense masked attention.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch30_`; expect missing schedule APIs.
- [ ] Implement acyclic source ownership, per-query selection identity, candidate-pool restriction, and typed adapters for Tasks 15-16. Full computes banks/index keys/selections; Reindex shares banks and recomputes query selection; Reuse shares banks and prior same-query selection.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch30_`; expect PASS, including frozen-selection indexer-loss descent-sign checks.
- [ ] Commit the named files as `v0.M.NNN: add explicit sparse source and selection schedules`.

### Task 31: Deterministic top-k router

**Files**
- Create: `src/lightnet-lm/src/stacks/router.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_router.rs`.

**Interfaces**
- Consumes: `Tensor`, `select_indices(...)`, `Tape::backward(...)`.
- Produces: `RouterScore { Softmax, SqrtSoftplus }`, `RouterSpec { pub experts: usize, pub top_k: usize, pub score: RouterScore, pub routed_scale: f32 }`.
- Produces: `Route { pub experts: Vec<u32>, pub weights: Tensor, pub counts: Vec<u64>, pub tape: Tape }`.
- Produces: `route(logits: &Tensor, selection_bias: &[f32], valid: &[bool], spec: &RouterSpec) -> LmResult<Route>`; route tape returns `[dLogits]`.

- [ ] Write `lm_arch31_topk`: three equal logits/top-k 2 select experts `[0,1]`; scale 1 gives weights `[0.5,0.5]`; scale 1.5 gives `[0.75,0.75]`.
- [ ] Add `lm_arch31_score_gradient`: `sqrt(softplus(0))=sqrt(ln(2))`; selection bias changes selected IDs but not selected weighting scores; padded tokens have zero counts and gradients. Finite-difference tie-free fixed selections.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch31_`; expect missing router APIs.
- [ ] Implement stable softplus, softmax scoring, ascending expert-ID tie breaks, selection-only biases, selected-score normalization, and explicit rejection of top-k 0 or greater than expert count.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch31_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add deterministic normalized top k routing`.

### Task 32: Shared and routed expert FFN

**Files**
- Create: `src/lightnet-lm/src/stacks/moe.rs`.
- Modify: `src/lightnet-lm/src/stacks/model.rs`, `src/lightnet-lm/src/stacks/encoder.rs`, `src/lightnet-lm/src/stacks/decoder.rs`, `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_moe.rs`.

**Interfaces**
- Consumes: `route(...)` and original `swiglu(...)`/adjoints.
- Produces: `MoeSpec { pub router: RouterSpec, pub shared_experts: usize, pub expert_width: usize }`.
- Produces: `moe_ffn(x: &Tensor, router_weight: &Tensor, experts: &[Tensor], selection_bias: &[f32], valid: &[bool], spec: &MoeSpec, dense_reference: bool) -> LmResult<Eval>`.
- Produces: `CpuModel::enable_moe(&mut self, spec: MoeSpec, seed: u64) -> LmResult<()>`; tape order `[dX,dRouterWeight,expert gradients...]`, experts ordered shared first, then routed, each `[gate,up,down]`.

- [ ] Write `lm_arch32_sparse_dense`: one shared/three routed/top-k 2 gives identical sparse-dispatch and dense-masked outputs within `1e-6`; only two routed experts/token execute in sparse mode.
- [ ] Add `lm_arch32_expert_gradient`: finite-difference shared, selected, router, and input tensors; unselected expert gradients equal zero. Repeated routes sum contributions; no token is dropped when all tokens select expert 0.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch32_`; expect missing MoE APIs.
- [ ] Implement always-active summed shared experts plus weighted routed SwiGLU experts, stable dispatch/inverse combine, unique registry entries, and explicit model revision changes on structural replacement.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch32_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: add shared and sparse routed expert feed forward`.

### Task 33: Explicit auxiliary-loss-free balancing policy

**Files**
- Create: `src/lightnet-lm/src/stacks/balance.rs`.
- Modify: `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/model.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_balance.rs`.

**Interfaces**
- Consumes: `Route::counts`, original update success/failure lifecycle.
- Produces: `RouterState { pub bias: Vec<f32>, pub pending_counts: Vec<u64> }`.
- Produces: `BalanceSpec { pub rate: f32, pub max_abs_bias: f32 }`.
- Produces: `RouterState::observe(&mut self, counts: &[u64]) -> LmResult<()>`, `RouterState::finish(&mut self, applied: bool, spec: &BalanceSpec) -> LmResult<()>`.

- [ ] Write `lm_arch33_balance`: counts `[8,0]`, zero bias, rate 0.1 produce bias `[-0.1,0.1]`; counts `[4,4]` produce no change.
- [ ] Add `lm_arch33_transaction`: failed update preserves committed bias and clears the rejected window's counts; successful update clears counts after applying once; zero total assignments leave bias unchanged.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch33_`; expect missing balancing APIs.
- [ ] Implement the DESIGN rule `bias += rate*sign(mean_count-count)`, then subtract mean bias and symmetrically clip. Counts span one accumulation window, exclude padding/shared experts, and never enter the differentiable loss.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch33_`; expect PASS, including pending-count overflow rejection.
- [ ] Commit the named files as `v0.M.NNN: add transactional auxiliary loss free router balancing`.

### Task 34: Multi-token prediction objective

**Files**
- Create: `src/lightnet-lm/src/stacks/mtp.rs`.
- Modify: `src/lightnet-lm/src/stacks/model.rs`, `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/stacks/tape.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_mtp.rs`.

**Interfaces**
- Consumes: `CpuModel::forward(...)`, original `sparse_ce(...)`, tied parameter registry.
- Produces: `MtpSpec { pub weights: Vec<f32> }`, `ObjectiveResult { pub main: LossSummary, pub auxiliary: Vec<LossSummary>, pub grads: DecoderGrads }`.
- Produces: `mtp_targets(batch: &TokenBatch, extra_horizon: usize) -> LmResult<Vec<i32>>`.
- Produces: `CpuModel::loss_backward(&self, input: &ModelInput, mtp: &MtpSpec) -> LmResult<ObjectiveResult>`.

- [ ] Write `lm_arch34_targets`: base targets `[5,6,2,-100]`, extra horizon 1 produce `[6,2,-100,-100]`; a segment boundary before index 2 masks any target crossing it.
- [ ] Add `lm_arch34_descent`: zero binary logits/target 1/weight 0.25 give auxiliary seed `[-0.125,0.125]`; zero auxiliary weights reproduce base gradients exactly.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch34_`; expect missing MTP APIs.
- [ ] Implement one pre-RMSNorm-plus-linear residual prediction module per extra horizon, reading the final causal hidden state without future-token inputs and using the tied vocabulary head. Sum weighted loss seeds; normalize main and auxiliary sums once by main active-token count.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch34_`; expect PASS, including finite differences for module, backbone, and tied-head parameters.
- [ ] Commit the named files as `v0.M.NNN: add explicit multi token prediction objective`.

### Task 35: Composition checkpoint and update lifecycle

**Files**
- Create: `src/lightnet-lm/src/stacks/state.rs`.
- Modify: `src/lightnet-lm/src/stacks/model.rs`, `src/lightnet-lm/src/stacks/mod.rs`, `src/lightnet-lm/src/checkpoint.rs`, `src/lightnet-lm/src/train.rs`.
- Test: Create `src/lightnet-lm/tests/lm_stack_state.rs`.

**Interfaces**
- Consumes: `EngineState`, `UpdateOptions`, `ObjectiveResult`, `RouterState`, existing checkpoint checksums/atomic publication.
- Produces: `StackState { pub schema: u32, pub model: ModelConfig, pub engine: EngineState, pub routers: Vec<RouterState>, pub mtp: MtpSpec, pub moe: Option<MoeSpec>, pub sparse: Option<SparseSchedule> }`.
- Produces: `save_stack_state(path: &std::path::Path, state: &StackState) -> LmResult<()>`, `load_stack_state(path: &std::path::Path) -> LmResult<StackState>`.
- Produces: `CpuStackEngine::new(model: CpuModel, mtp: MtpSpec) -> LmResult<Self>`, `CpuStackEngine::microbatch(&mut self, input: &ModelInput) -> LmResult<LossSummary>`, `CpuStackEngine::apply_update(&mut self, options: &UpdateOptions) -> LmResult<bool>`, `CpuStackEngine::export_state(&self) -> LmResult<StackState>`, `CpuStackEngine::import_state(&mut self, state: &StackState) -> LmResult<()>`.

- [ ] Write `lm_arch35_resume`: save mid-accumulation after microbatch sizes 2 and 3; resumed parameters, moments, pending gradients/counts, RNG, and next successful update match uninterrupted CPU execution exactly.
- [ ] Add `lm_arch35_reject_atomic`: inject NaN into a shared-expert gradient; parameters, moments, router biases, update count, and model revision remain unchanged. Truncated payload, changed architecture, or schema 999 errors.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch35_`; expect missing stack-state APIs.
- [ ] Implement schema-versioned composition identity, optional-state serialization, single normalization/clip/update transaction, and rejection/discard rules. Do not serialize ephemeral generation caches; invalidate them after successful updates/imports.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch35_`; expect PASS.
- [ ] Commit the named files as `v0.M.NNN: persist stack objectives routing and update state`.

### Task 36: Resident oxide encoder and cross-attention

**Files**
- Create: `src/lightnet-cuda-oxide/src/lm_stacks.rs`, `src/lightnet-cuda-oxide/src/kernels/lm_stacks.rs`.
- Modify: `src/lightnet-cuda-oxide/src/lib.rs`, `src/lightnet-cuda-oxide/src/kernels/mod.rs`.
- Test: Create `src/lightnet-cuda-oxide/tests/oxide_lm_stacks.rs`.

**Interfaces**
- Consumes: original `OxideBackend`, `LmTransferStats`, Tasks 8-11 native primitives, and `CpuModel`/`StackState`.
- Produces: `OxideStacks::upload(backend: OxideBackend, state: &StackState) -> LmResult<Self>`.
- Produces: `OxideStacks::forward(&mut self, input: &ModelInput) -> LmResult<()>`, `OxideStacks::loss_backward(&mut self, input: &ModelInput) -> LmResult<LossSummary>`.
- Produces: `OxideStacks::download_logits(&self) -> LmResult<Tensor>`, `OxideStacks::download_gradients(&self) -> LmResult<DecoderGrads>`, `OxideStacks::transfer_stats(&self) -> LmTransferStats`.

- [ ] Write `oxide_arch36_reference`: bidirectional and rectangular cross attention agree with bf16-rounded CPU operands; normalized max error `max(abs(gpu-ref))/max(1,max(abs(ref)))` is at most 0.03 forward/0.08 backward.
- [ ] Add `oxide_arch36_residency`: source length 5/target length 3, partial tiles and all-padding source rows; zero activation/gradient readbacks before explicit downloads.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stacks`; expect missing device stack APIs.
- [ ] Implement native GQA encoder/decoder/cross forward and adjoints, pre-norms, source-gradient reduction, partial RoPE/YaRN, and existing chunked tied-head CE. Unsupported optional modes error.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stacks`; expect PASS on an actual device with finite zero masked rows.
- [ ] Commit the named files as `v0.M.NNN: add resident oxide encoder and cross attention`.

### Task 37: Oxide latent, sparse, and split caches

**Files**
- Create: `src/lightnet-cuda-oxide/src/kernels/lm_sparse.rs`.
- Modify: `src/lightnet-cuda-oxide/src/lm_stacks.rs`, `src/lightnet-cuda-oxide/src/kernels/lm_stacks.rs`, `src/lightnet-cuda-oxide/src/kernels/mod.rs`.
- Test: Create `src/lightnet-cuda-oxide/tests/oxide_lm_stack_sparse.rs`.

**Interfaces**
- Consumes: Tasks 20-21, 27-30 CPU contracts; original Task 15-16 device primitives.
- Produces: `OxideStacks::prefill(&mut self, input: &ModelInput, capacity: usize) -> LmResult<()>`, `OxideStacks::decode(&mut self, token: u32) -> LmResult<()>`, `OxideStacks::cache_stats(&self) -> LmResult<CacheStats>`.
- Produces: `OxideStacks::workspace_bytes(&self) -> u64`; extends upload/forward/backward dispatch to explicit latent, compressed, and causal-split modes.

- [ ] Write `oxide_arch37_sparse_reference`: low-rank MQA, expanded/absorbed MLA, compression, and scheduled selection match their respective CPU references within 0.03 forward/0.08 gradient normalized error.
- [ ] Add `oxide_arch37_cache_prefix`: nine-token split/window-3 decoding agrees with device full-prefix evaluation within 0.03; latent cache byte count excludes expanded K/V; stale source indices error before launch.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stack_sparse`; expect unsupported optional device modes.
- [ ] Implement grouped projections, latent contractions, pooling/adjoints, tiled selection, cache append/rings, and shared-global reduction. Bound scratch by tiles/selections; never allocate dense T-by-T index scores.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stack_sparse`; expect PASS, including doubled-context workspace accounting and zero intermediate host transfers.
- [ ] Commit the named files as `v0.M.NNN: add oxide latent sparse and causal split execution`.

### Task 38: Oxide MoE and MTP

**Files**
- Create: `src/lightnet-cuda-oxide/src/kernels/lm_experts.rs`.
- Modify: `src/lightnet-cuda-oxide/src/lm_stacks.rs`, `src/lightnet-cuda-oxide/src/kernels/mod.rs`.
- Test: Create `src/lightnet-cuda-oxide/tests/oxide_lm_stack_experts.rs`.

**Interfaces**
- Consumes: Tasks 31-35 contracts and original Task 9 native f32 accumulation/update kernels.
- Produces: `OxideStacks::apply_update(&mut self, options: &UpdateOptions) -> LmResult<bool>`, `OxideStacks::export_state(&self) -> LmResult<StackState>`, `OxideStacks::import_state(&mut self, state: &StackState) -> LmResult<()>`.
- Extends `loss_backward(...)` to shared/routed FFNs and MTP without changing descent-sign or normalization contracts.

- [ ] Write `oxide_arch38_experts`: one shared/four routed/top-k 2 matches dense-masked CPU reference within 0.03 forward/0.08 gradients; exact ties select IDs `[0,1]`.
- [ ] Add `oxide_arch38_transaction`: all tokens route to expert 0 without drops; rejected NaN update changes no master/moment/bias; MTP weight 0.25 preserves the binary seed sign from Task 34.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stack_experts`; expect unsupported device MoE/MTP.
- [ ] Implement deterministic dispatch/prefix sums, selected-expert grouped GEMMs, weighted combine/adjoints, device counts/bias commit, MTP CE, and unique updates. Keep normal forward/backward/update resident.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stack_experts`; expect PASS, including same-backend mid-window state resume and transfer accounting.
- [ ] Commit the named files as `v0.M.NNN: add resident oxide expert and multi token objectives`.

### Task 39: Projection-scoped FP8 mixed precision

**Files**
- Modify: `src/lightnet-cuda-oxide/src/lm_stacks.rs`, `src/lightnet-cuda-oxide/src/kernels/lm_stacks.rs`, `src/lightnet-cuda-oxide/src/kernels/lm_experts.rs`, `src/lightnet-lm/src/stacks/state.rs`.
- Test: Create `src/lightnet-cuda-oxide/tests/oxide_lm_stack_fp8.rs`.

**Interfaces**
- Consumes: original Task 12 `DecoderPrecision`, FP8 role histories/scaling, and Task 38 update lifecycle.
- Produces: `ProjectionRole { EncoderQ, EncoderKv, EncoderO, DecoderQ, DecoderKv, DecoderO, CrossQ, CrossKv, CrossO, GlobalKv, FfnGate, FfnUp, FfnDown, ExpertGate, ExpertUp, ExpertDown }`.
- Produces: `OxideStacks::set_precision(&mut self, precision: DecoderPrecision) -> LmResult<()>`; checkpoint role identity additionally includes layer, group, and expert index.
- MTP modules, embedding/head, rotary/indexer scores, attention, and reductions remain bf16/f32 initially.

- [ ] Write `oxide_arch39_quantized_reference`: each enabled projection matches fixed dequantized-operand GEMM within normalized error 0.02; zero amax yields finite zero output.
- [ ] Add `oxide_arch39_lifecycle`: encoder-Q and cross-Q histories remain separate; successful update invalidates quantized copies once, rejected update does not advance generation, and resume reproduces histories.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stack_fp8`; expect missing stack precision dispatch.
- [ ] Implement enumerated E4M3 forward roles and supported E5M2 backward roles, device scaling/non-finite decisions, f32 masters/gradients/moments, and complete role serialization. Reject unsupported geometry explicitly.
- [ ] Run `MEM=3G JOBS=4 cargo xtask oxide --test oxide_lm_stack_fp8`; expect PASS against the numerical operator reference, not bit-quantizer finite differences.
- [ ] Commit the named files as `v0.M.NNN: extend opt in fp8 to explicit stack projection roles`.

### Task 40: Pilot configurations and acceptance matrix

**Files**
- Create: `cfg/llm_latin_seq2seq.cfg`, `cfg/llm_latin_split.cfg`, `src/lightnet-lm/tests/lm_stack_pilot.rs`.
- Modify: `cfg/llm_latin.cfg`, `src/lightnet-lm/CONTEXT.md`, `tasks/rust/llm_eval.md`.

**Interfaces**
- Consumes: `ModelConfig::from_cfg(...)`, `CpuModel::parameters(...)`, all three composition/cache contracts.
- Produces: three strictly parsed configurations and a documented acceptance matrix; no additional runtime API.

- [ ] Write `lm_arch40_pilot_geometry`: vocabulary 8192, hidden 512, FFN 1360, Q/KV heads 8/2; decoder-only has 12 layers, classic and split have 6+6; all have 30-60M unique registered parameters.
- [ ] Add `lm_arch40_config_modes`: defaults are dense SwiGLU/GQA/bf16, context 1024, theta 10000, RMS epsilon `1e-5`, no YaRN/MoE/MTP/sparsity; `fp4_kv`, `engram`, `mhc`, and `bounded_replay` error.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch40_`; expect missing configurations or unsupported composition parsing.
- [ ] Implement configs and TOC-bearing documentation. Define separate future ablations for classic conditioning, split, MLA/MQA, sparse context, small MoE, MTP, and FP8; change one axis at a time.
- [ ] Run `MEM=3G JOBS=4 PKGS=lightnet-lm cargo xtask test lm_arch40_`; then run `MEM=3G JOBS=4 cargo xtask check lightnet-lm`, `MEM=3G JOBS=4 cargo xtask clippy lightnet-lm`, and `cargo xtask fmt`, sequentially. Expect PASS/zero exit; stage formatting changes only within named files.
- [ ] Commit the named files as `v0.M.NNN: define bounded latin stack pilots and acceptance gates`.

Future operator-authorized experiments must record corpus/tokenizer identity, unique/active parameter counts, matched token budgets, held-out main-token NLL, and source-conditioned task metrics where applicable. Report calibration, warmup, steady-state throughput, cache/workspace bytes, and host transfers separately.

FP8 quality acceptance retains the original plan's maximum 0.02 nats/token held-out degradation relative to matched bf16. Sparse/compressed attention quality is an experiment; agreement with its masked/compressed dense oracle does not establish full-attention quality parity.

## Open issues

- **Research fidelity:** exact CSA2/HCA, CED local/global equations, mHC, bounded replay, and `noaux_tc` mechanics remain unverified. Faithful reproduction requires raw implementation/report review; the explicit local variants above can proceed independently.
- **YaRN fidelity:** the declared frequency-only variant is implementable and testable, but matching an external checkpoint may require its exact correction and attention-magnitude conventions.
- **Cross-attention variants:** classic cross-attention initially uses ordinary GQA projections. MLA cross-memory absorption needs a separate design and test cycle if source-cache size becomes a measured bottleneck.
- **Pilot purpose:** choose the source-conditioned Latin task and paired-data provenance before requesting classic encoder-decoder training. A decoder-only corpus does not by itself justify that mode.
- **Hardware and economics:** available VRAM, supported FP8 shapes, useful context lengths, and actual MoE/latent-attention speedups are unmeasured. No larger-scale feature is justified by configuration presence alone.
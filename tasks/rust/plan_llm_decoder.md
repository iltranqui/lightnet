# Trainable LLM Decoder Implementation Plan

> Drafted by gpt-6.1-sol (Codex CLI, read-only). Unreviewed.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The repository’s current instruction to work directly without subagents takes precedence; use inline execution unless that instruction changes.

**Goal:** Add a trainable decoder-only language model with CPU correctness references, resumable token training, cached generation, resident oxide bf16 training, a reproducible Latin pilot, and separately gated attention/FP8 optimizations.

**Architecture:** Introduce an independent `lightnet_lm` crate (new workspace member `src/lightnet-lm`) with an API using integer token IDs and contiguous `[batch, time, hidden]` activations. Implement pre-RMSNorm causal GQA blocks with RoPE, SwiGLU, residual connections, sparse next-token cross-entropy, and one shared embedding/output parameter. Keep the existing detector graph, `[transformer]` meaning, and frozen LAYA contracts intact; implement oxide execution against the new decoder contract without host-operation fallbacks.

**Tech Stack:** Existing Rust 2024 workspace and pinned `nightly-2026-08-28`; new `lightnet-lm`, plus `lightnet-cli`, `lightnet-cuda`, and the excluded `lightnet-cuda-oxide` bridge. Reuse existing AdamW mathematics and oxide GEMM/precision facilities. Proposed additions are workspace `serde`, `serde_json`, and `sha2` dependencies in `lightnet-lm`, plus `unicode-normalization = "0.1"`; the latter dependency and its resolved version are **unverified** until implementation.

**Spec:** The user-provided scope and staged order, formalized by the contracts below. Intended destination: `tasks/rust/plan_llm_decoder.md`, a proposed file absent during this audit. Repository constraints come from `CLAUDE.md`, `tasks/rust/AGENTS.md`, and the Global Constraints in `tasks/rust/plan_fp16.md`; structure follows `tasks/rust/plan_fp8.md`. `/home/kerrigan/.claude/jobs/c4ec0b42/tmp/llm_latin_gap.md` was read as advisory material and its implementation claims were checked against source.

This document is a read-only implementation plan. No files were changed; no builds, tests, benchmarks, downloads, or training runs were performed. Proposed files, symbols, commands, and acceptance thresholds below are specifications, **unverified as implemented behavior**.

## Table of Contents

- [Global Constraints](#global-constraints)
- [Review Focus](#review-focus)
- [Verified starting state](#verified-starting-state)
  - [Existing implementation boundaries](#existing-implementation-boundaries)
  - [Evidence limits](#evidence-limits)
- [File structure](#file-structure)
  - [Relevant project tree](#relevant-project-tree)
  - [Component breakdown](#component-breakdown)
- [Tasks](#tasks)
  - [Task 1: Decoder and token contracts](#task-1-decoder-and-token-contracts)
  - [Task 2: Byte-fallback BPE and normalization](#task-2-byte-fallback-bpe-and-normalization)
  - [Task 3: CPU differentiable decoder operations](#task-3-cpu-differentiable-decoder-operations)
  - [Task 4: CPU trainable decoder and tied head](#task-4-cpu-trainable-decoder-and-tied-head)
  - [Task 5: Token shards and deterministic packing](#task-5-token-shards-and-deterministic-packing)
  - [Task 6: Token training and checkpoint resume](#task-6-token-training-and-checkpoint-resume)
  - [Task 7: CPU KV-cache and sampling](#task-7-cpu-kv-cache-and-sampling)
  - [Task 8: Resident oxide bf16 forward](#task-8-resident-oxide-bf16-forward)
  - [Task 9: Resident oxide backward and updates](#task-9-resident-oxide-backward-and-updates)
  - [Task 10: Latin pilot and evaluation](#task-10-latin-pilot-and-evaluation)
  - [Task 11: Causal GQA flash and chunked cross-entropy](#task-11-causal-gqa-flash-and-chunked-cross-entropy)
  - [Task 12: Opt-in decoder FP8](#task-12-opt-in-decoder-fp8)
  - [Task 13: Optional cuTile inference](#task-13-optional-cutile-inference)
  - [Task 14: Causal sliding-window attention](#task-14-causal-sliding-window-attention)
  - [Task 15: Optional DSA-style indexer top-k attention](#task-15-optional-dsa-style-indexer-top-k-attention)
  - [Task 16: Optional NSA-style three-branch attention](#task-16-optional-nsa-style-three-branch-attention)
- [Progress](#progress)
- [Open issues](#open-issues)

## Global Constraints

- Use `cargo xtask` only for checks, builds, tests, formatting, and later authorized execution: `check`, `gpu-check`, `build`, `clippy`, `test`, `oxide`, `fmt`. Never invoke raw Cargo build/test/check commands or `systemd-run` directly.
- Maximum **4 build threads**, including new commands. One build/test command at a time.
- Keep `MemoryHigh=3G`; xtask applies the soft cap. Existing runners warn and continue uncapped without a user systemd bus; record that condition rather than claiming a capped run.
- `src/lightnet-cuda-oxide` is **not** a workspace member. Its GPU tests run through `cargo xtask oxide --test <suite>`.
- The current request authorizes planning only. During implementation, normal bounded synthetic tests are permitted; corpus training, tiny-corpus overfit runs, and long benchmarks require a user request.
- Unknown sections, keys, precision modes, token IDs, and unsupported execution modes return errors. Never silently substitute an identity layer or CPU execution.
- Backward operations compute exact adjoints of their Rust forward. Sparse CE seeds backward with **target minus prediction**, consistent with existing descent-direction updates.
- Never claim C++ parity. CPU/device comparisons establish Rust-reference agreement only.
- Preserve existing detector cfg semantics, checkpoint formats, precision selectors, and unrelated working-tree changes.
- Stage only files named by a task, including its common version files. Never use `git add -A`.
- Commit and tag only after the task’s named checks pass. Use `v0.M.NNN: <what>`, selecting the next unused number under `tasks/rust/AGENTS.md` Versioning. CPU increments use the CPU line; GPU increments use the GPU line. Do not allocate numbers in this draft.
- Every version-mapped increment includes `tasks/rust/implemented.md`, `tasks/rust/STATUS.md`, and `tasks/rust/todo.md`; append the resulting full SHA to `tasks/rust/version-map.tsv` in the separate docs-only mapping commit prescribed by AGENTS.md.
- `tasks/` is git-excluded; use explicit `git add -f` for local-only task documents.
- Each edited Markdown file needs an updated TOC.
- Keep commits and tags local. Record evidence before deleting disposable weights, sidecars, logs, or benchmark artifacts; preserve referenced fixtures.
- Read each affected crate’s `CONTEXT.md` before implementation.
- All tasks below include the four common version files named above and this plan’s progress checkboxes in their **Files** scope.
- Each task follows red test → observed failure → implementation → observed green checks → local commit/tag. A zero-test filter, timeout, skipped GPU test, or unavailable device is not a pass.

## Review Focus

1. **Repeated IDs and tied storage:** repeated embedding indices accumulate every contribution; input embedding and output-head contributions reach one parameter, counted once by clipping, decay, optimization, and serialization. Owned by Tasks 3, 4, 6, and 9.
2. **Padding and packed boundaries:** future tokens and neighboring documents cannot influence a prediction; ignored positions contribute neither loss nor gradient; all-ignored batches cannot advance an optimizer step. Owned by Tasks 3–6 and 11.
3. **Partial accumulation and interrupted resume:** unequal microbatch token counts, mid-window checkpoints, and rejected updates preserve normalization and restore the next batch, RNG, schedule, gradients, and optimizer state. Owned by Tasks 5, 6, 9, and 12.
4. **Cache offsets and invalid generation input:** cached logits match full-prefix logits at nonzero positions; overflow, stale caches, empty prompts, invalid probabilities, and unsupported settings fail predictably without corrupting cache state. Owned by Tasks 7, 8, and 13.
5. **Non-finite and precision-boundary values:** non-finite gradients cannot produce a partial optimizer update; wide logits use stable CE without vision clamps; unsupported device shapes are rejected explicitly. Owned by Tasks 3, 6, 8, 9, 11, and 12.

## Verified starting state

### Existing implementation boundaries

| Area | Source checked | Verified finding and planning consequence |
|---|---|---|
| Workspace/toolchain | `Cargo.toml`, `rust-toolchain.toml` | Rust edition 2024 and `nightly-2026-08-28`; oxide and cuTile bridges are excluded from the parent workspace. |
| Vision transformer | `src/lightnet-core/src/model.rs` | `SwinSpec` explicitly represents `[swin]`/`[transformer]` windowed attention. Do not reinterpret that configuration as a text decoder. |
| LAYA | `src/lightnet-core/src/laya_encoder.rs`, `src/lightnet-core/src/runtime/exec_backward.rs` | Frozen embedding/attention/GeGLU encoder; input IDs are represented as floats in its existing channel-major contract. Its encoder backward arm stops gradient propagation. |
| CPU transformer primitives | `src/lightnet-core/src/vit.rs` | `layer_norm_forward` and `layer_norm_backward` include affine gradients; linear and attention derivatives exist. These are useful references, not a complete trainable causal decoder. |
| Existing attention | `src/lightnet-core/src/vit.rs`, `src/lightnet-cuda/src/lib.rs`, `src/lightnet-cuda-oxide/src/kernels/layers_extra.rs` | Existing standard flash interfaces use fused equal-width QKV. The inspected oxide forward loops over every key, uses fixed 128-element local arrays, clamps Q/K to ±128, and scores to ±10. New decoder semantics need separate kernels. |
| GPU training fallback | `src/lightnet-core/src/runtime/gpu_backward_support.rs` | `requires_cpu_backward_fallback` includes ViT, Swin, and other attention families. Existing native suboperations do not prove resident decoder training. |
| Loss direction | `src/lightnet-core/src/jev_ce.rs` | `jev_ce_loss` accepts soft targets and writes `(t - probs[idx]) / temperature`. New sparse CE must preserve the descent direction. |
| AdamW | `src/lightnet-cuda/src/host_ops.rs`, `src/lightnet-core/src/runtime/param_updates.rs` | `AdamParams` supports decoupled decay; `adam_update` adds the supplied descent direction to weights. Reuse this math with decoder parameter registration. |
| Existing schedules | `src/lightnet-core/src/runtime/exec_accumulate.rs` | `compute_learning_rate` is batch/iteration based, including burn-in and SGDR. Token-count warmup/cosine needs its own explicit contract. |
| Training data | `src/lightnet-core/src/train.rs` | Inspected training batches own images and detector labels. Introduce a separate token loader. |
| Checkpoints | `src/lightnet-core/src/weights.rs`, `src/lightnet-core/src/runtime/exec_core.rs` | Legacy f32 `.weights` and DNOPT optimizer state exist. They do not define the proposed decoder’s tokenizer, packing cursor, RNG, or token schedule state. |
| Precision | `src/lightnet-cuda-oxide/src/mixed.rs`, `src/lightnet-cuda-oxide/src/half.rs`, `src/lightnet-cuda-oxide/src/lib.rs` | `HalfFormat` includes f16/bf16; convolution AMP, selected half operations, and FP8 GEMM facilities exist. “Only GEMM and convolution support reduced precision” is too broad: selected other operations also support it. Resident transformer training remains unestablished. |
| Oxide dependency boundary | `src/lightnet-cuda-oxide/Cargo.toml` | `lightnet-core` is currently a **dev-dependency**. The decoder types live in `lightnet-lm`, so the bridge needs `lightnet-lm` as a normal dependency (explicit manifest change); it does not need `lightnet-core`. |
| cuTile integration | `src/lightnet-cuda-cutile/Cargo.toml`, `src/lightnet-cuda-cutile/src/ops_elementwise.rs`, `src/lightnet-cuda/src/lib.rs` | The adapter does not depend on `cutile-kernels`; inspected activation matches omit current enum variants. Buildability is unverified. |
| Vendored cuTile assets | `src/cutile-rs-aot/cutile-kernels/src/{attention,embeddings,norms,positional,kv_cache,pointwise,argmax}.rs` | Forward source exists for embedding, RMSNorm, RoPE, causal/GQA attention, cache writes, gating, and head reduction. RoPE uses split halves; one inspected cache-write entry assumes position zero. Wrappers must respect those contracts. |
| Runner | `xtask/src/main.rs`, `tools/test_oxide.sh`, `tools/lib_cap.sh` | Existing xtask enforces the thread ceiling and wraps normal checks; oxide tests use the special codegen runner. No generic LM execution or cuTile command currently exists. |

Source searches did not locate a product decoder, BPE tokenizer, RMSNorm decoder API, or KV-cache API under the inspected product Rust sources. This is a source-search result, not proof that every vendored dependency lacks such code.

### Evidence limits

- No current build, test, runtime, convergence, throughput, or memory measurement was performed.
- Historical baseline results in `tasks/rust/STATUS.md` are not fresh verification. That document also records unresolved full-suite evidence gaps.
- Corpus availability, editions, licenses, download sizes, token counts, and extraction quality are **unverified**. No earlier gap-analysis corpus statistics are adopted here.
- Device model, available VRAM, resident model capacity, and cuTile toolchain compatibility are **unverified**.
- All proposed files below were absent during the audit; proposed APIs are design decisions, not existing interfaces.

## File structure

### Relevant project tree

`[existing]` means inspected repository material; `[new]` means proposed and unverified. This tree includes the implementation scope and its direct reference components, not unrelated detector modules.

```text
lightnet/
├── Cargo.toml                                      [existing]
├── Cargo.lock                                      [existing; dependency resolution only]
├── CLAUDE.md                                       [existing]
├── rust-toolchain.toml                             [existing; reference only]
├── cfg/
│   ├── llm_smoke.cfg                               [new]
│   └── llm_latin.cfg                               [new]
├── tasks/rust/
│   ├── AGENTS.md                                   [existing; reference only]
│   ├── plan_fp16.md                                [existing; reference only]
│   ├── plan_fp8.md                                 [existing; reference only]
│   ├── plan_llm_decoder.md                         [new; this plan]
│   ├── llm_data.md                                 [new]
│   ├── llm_eval.md                                 [new]
│   ├── implemented.md                             [existing]
│   ├── STATUS.md                                  [existing]
│   ├── todo.md                                    [existing]
│   └── version-map.tsv                             [existing]
├── src/
│   ├── lightnet-core/
│   │   ├── Cargo.toml                              [existing]
│   │   ├── CONTEXT.md                              [existing]
│   │   ├── src/
│   │   │   ├── lib.rs                              [existing]
│   │   │   ├── cfg.rs                              [existing; reuse parser]
│   │   │   ├── error.rs                            [existing; reuse errors]
│   │   │   ├── model.rs                            [existing; reference only]
│   │   │   ├── vit.rs                              [existing; reference only]
│   │   │   ├── jev_ce.rs                           [existing; reference only]
│   │   │   ├── laya_encoder.rs                     [existing; reference only]
│   │   │   ├── train.rs                            [existing; reference only]
│   │   │   ├── weights.rs                          [existing; reference only]
│   │   │   ├── runtime/
│   │   │   │   ├── types.rs                        [existing; reference only]
│   │   │   │   ├── exec_core.rs                    [existing; reference only]
│   │   │   │   ├── exec_backward.rs                [existing; reference only]
│   │   │   │   ├── exec_accumulate.rs              [existing; reference only]
│   │   │   │   ├── gpu_backward_support.rs         [existing; reference only]
│   │   │   │   └── param_updates.rs                [existing; reference only]
│   │   │   └── (no change: decoder code lives in lightnet-lm)
│   │   └── (tests unchanged)
│   ├── lightnet-lm/                                 [new crate, workspace member]
│   │   ├── Cargo.toml                              deps: lightnet-cuda (AdamW), serde, serde_json, sha2, unicode-normalization
│   │   ├── CONTEXT.md                              crate map
│   │   ├── src/
│   │   │   ├── lib.rs                              public exports
│   │   │   ├── config.rs                           decoder schema
│   │   │   ├── types.rs                            token/parameter/state types
│   │   │   ├── tokenizer.rs                        normalization and byte BPE
│   │   │   ├── ops.rs                              CPU differentiable operations
│   │   │   ├── attention.rs                        causal/GQA CPU reference
│   │   │   ├── decoder.rs                          CPU block composition
│   │   │   ├── data.rs                             manifests, shards, packing
│   │   │   ├── optim.rs                            AdamW, clipping, token LR
│   │   │   ├── train.rs                            engine contract and trainer
│   │   │   ├── checkpoint.rs                       decoder checkpoint format
│   │   │   ├── cache.rs                            CPU KV state
│   │   │   ├── sampling.rs                         deterministic token sampling
│   │   │   └── eval.rs                             loss/byte and diagnostics
│   │   └── tests/                                  lm_contract, lm_tokenizer, lm_ops, lm_decoder, lm_data, lm_train, lm_generate, lm_eval
│   ├── lightnet-cli/
│   │   ├── Cargo.toml                              [existing]
│   │   ├── CONTEXT.md                              [existing]
│   │   └── src/
│   │       ├── main.rs                             [existing]
│   │       └── lm.rs                               [new; CLI and inline tests]
│   ├── lightnet-cuda/
│   │   └── src/
│   │       ├── lib.rs                              [existing; reference only]
│   │       └── host_ops.rs                          [existing; reuse AdamW]
│   ├── lightnet-cuda-oxide/
│   │   ├── Cargo.toml                              [existing]
│   │   ├── CONTEXT.md                              [existing]
│   │   ├── src/
│   │   │   ├── lib.rs                              [existing]
│   │   │   ├── mixed.rs                            [existing; reuse]
│   │   │   ├── half.rs                             [existing; reuse]
│   │   │   ├── lm.rs                               [new; device engine]
│   │   │   └── kernels/
│   │   │       ├── mod.rs                          [existing]
│   │   │       ├── layers_extra.rs                 [existing; reference only]
│   │   │       └── lm.rs                           [new; decoder kernels]
│   │   └── tests/
│   │       ├── oxide_amp.rs                        [existing; regression]
│   │       ├── oxide_gemm_f8.rs                    [existing; regression]
│   │       ├── oxide_lm_forward.rs                 [new]
│   │       ├── oxide_lm_train.rs                   [new]
│   │       ├── oxide_lm_flash.rs                   [new]
│   │       └── oxide_lm_fp8.rs                     [new]
│   ├── lightnet-cuda-cutile/
│   │   ├── Cargo.toml                              [existing; optional]
│   │   └── src/
│   │       ├── lib.rs                              [existing; optional]
│   │       ├── ops_elementwise.rs                  [existing; optional]
│   │       ├── trait_impl.rs                       [existing; reference only]
│   │       └── lm.rs                               [new; optional, inline tests]
│   └── cutile-rs-aot/cutile-kernels/
│       ├── Cargo.toml                              [existing; reference only]
│       └── src/
│           ├── attention.rs                        [existing; reuse only]
│           ├── embeddings.rs                       [existing; reuse only]
│           ├── norms.rs                            [existing; reuse only]
│           ├── positional.rs                       [existing; reuse only]
│           ├── kv_cache.rs                         [existing; reuse only]
│           ├── pointwise.rs                        [existing; reuse only]
│           └── argmax.rs                           [existing; reuse only]
├── xtask/src/main.rs                               [existing]
└── tools/
    ├── test_host.sh                                [existing; reference only]
    ├── test_oxide.sh                               [existing; preserve runner]
    └── lib_cap.sh                                  [existing; reference only]
```

### Component breakdown

| Component | Responsibilities and interfaces | Dependencies and consumers | Constraints/open questions |
|---|---|---|---|
| Configuration/types | Validate decoder geometry; define integer tokens, positions, masks, parameter identity, and state schemas. | Reuses core cfg/error modules; consumed by every LM component. | Independent cfg entry point; checked size arithmetic; unknown keys fail. |
| Tokenizer | Normalize text, train deterministic byte BPE, encode/decode, hash artifacts. | Core serialization/hash dependencies; used by data, CLI, generation, and checkpoints. | Preserve arbitrary byte coverage; normalization policy belongs to artifact identity. |
| CPU operations/attention | RMSNorm, gather/scatter-add, RoPE and adjoint, causal GQA, gated MLP, sparse CE. | Plain f32 arrays and contract types; consumed by CPU decoder and device-reference tests. | No vision clamps; masked softmax must be well defined. |
| CPU decoder | Compose trainable blocks, caches for backward, tied output head, seeded dropout. | CPU operations; consumed by training, cached inference, and evaluation. | One owner per parameter; no legacy `LayerKind` additions. |
| Data | Validate provenance/splits, write token shards, deterministic packing and resumable cursor. | Tokenizer and contract types; consumed by trainer/evaluator. | Bound memory; prevent cross-document targets and split leakage. |
| Optimization/training | Global clipping, token-count LR, accumulation, AdamW groups, backend-neutral update lifecycle. | Data/model and existing AdamW math; consumed by CLI and oxide engine. | Normalize once by active targets; failed updates cannot partially mutate state. |
| Checkpoint | Store unique f32 master parameters, optimizer/accumulation state, RNG, cursor, and hashes atomically. | Training state; consumed by resume and generation. | New explicit format; legacy `.weights`/DNOPT remain unchanged. |
| Cache/sampling | CPU prefill/decode, absolute offsets, greedy/temperature/top-k/top-p, EOS. | Decoder/tokenizer; consumed by generation CLI and optional device inference. | Reject overflow and stale state before mutation. |
| CLI/xtask | Thin argument parsing and capped execution routing. | Core and optional device bridge. | Training authorization remains external to command availability. |
| Oxide decoder | Resident bf16 forward/backward, f32 master gradients/moments, native reductions and updates. | Core LM contract plus existing oxide facilities. | Promote core dependency explicitly; no host fallback. |
| Optimized attention/FP8 | Tiled causal GQA, chunked head/CE, recomputation, opt-in FP8 projections. | Correct resident bf16 engine. | Separate correctness, memory, accuracy, and speed gates. |
| Latin configuration/evaluation | Data requirements, pilot geometry, held-out evaluation, reproducibility reports. | Tokenizer, loader, trainer, generation. | Actual corpus rights, counts, training budget, and hardware fit remain unverified. |
| Optional cuTile | Repair build boundary and adapt existing forward kernels. | Vendored kernel API and decoder inference contract. | Inference only; no implied training backward or hardware compatibility. |
| Ledgers/version files | Record task evidence and version mappings. | Every completed increment. | Follow explicit path staging and existing mapping format. |

## Tasks

### Task 1: Decoder and token contracts

**Files:**

- Create: `src/lightnet-lm/src/{lib,config,types}.rs`, `src/lightnet-lm/tests/lm_contract.rs`, `cfg/llm_smoke.cfg`.
- Modify: `Cargo.toml` (workspace member `src/lightnet-lm`), `src/lightnet-lm/Cargo.toml`, `Cargo.lock`.
- Common version files from Global Constraints.

**Interfaces:**

- Consumes: existing `ConfigFile`, `RawSection`, and `crate::error::Result<T>`.
- Produces:
  - `DecoderConfig::from_cfg(cfg: &ConfigFile) -> Result<Self>`.
  - `DecoderConfig::parameter_count(&self) -> Result<u64>`.
  - `DecoderConfig { vocab_size, context, hidden, intermediate, layers, q_heads, kv_heads: usize, rope_theta, rms_eps, dropout: f32, tie_embeddings: bool }`.
  - `TokenBatch { batch: usize, time: usize, input_ids: Vec<u32>, targets: Vec<i32>, positions: Vec<u32>, segments: Vec<u32>, valid: Vec<bool> }`.
  - `TokenBatch::validate(&self, cfg: &DecoderConfig) -> Result<()>`.
  - `LossSummary { sum: f64, active_tokens: u64 }`.
  - `RngState { state: u64 }`, using a documented SplitMix64 sequence.
- Row indexing is `b*time+t`; activation indexing is `(b*time+t)*hidden+d`.
- Attention permits a key only when both positions are valid, segments match, and key time is no later than query time. Positions control RoPE; time order controls causality.
- `targets=-100` means ignored. Every other target and every input ID must be within vocabulary bounds.
- `[decoder]` accepts exactly the named fields. Require tied embeddings for this implementation; reject `tie_embeddings=0`.
- Require positive dimensions, `hidden % q_heads == 0`, `q_heads % kv_heads == 0`, even head dimension, finite positive epsilon/theta, and `0 <= dropout < 1`.
- Smoke geometry: vocabulary 260, context 8, hidden 16, intermediate 32, two layers, four Q heads, two KV heads, theta 10000, epsilon `1e-5`, dropout zero.

- [x] **Step 1: Write contract tests.** `lm_contract_accepts_smoke_geometry` asserts the exact parsed fields; `lm_contract_rejects_invalid_geometry` rejects zero dimensions, odd head dimension, invalid head ratios, NaN epsilon, and overflow; `lm_contract_rejects_unknown_options` rejects a misspelled key and `[transformer]`; `lm_contract_validates_integer_batches` rejects mismatched lengths, out-of-range IDs/targets, invalid positions, and sequence length above context.
- [ ] **Step 2: Run** `cargo xtask test lm_contract_`. Expect missing-module/API failures; confirm the intended tests are selected once compilation succeeds.
- [x] **Step 3: Implement** the interfaces, strict section/key validation, checked allocation arithmetic, and smoke cfg. Keep legacy model parsing untouched.
- [x] **Step 4: Run** `cargo xtask test lm_contract_`, then `cargo xtask check lightnet-lm`, sequentially. Expect all named tests to pass and check exit status zero.
- [ ] **Step 5: Record evidence and commit/tag** the named files as `v0.M.NNN: define decoder token and configuration contracts`.

### Task 2: Byte-fallback BPE and normalization

**Files:**

- Create: `src/lightnet-lm/src/tokenizer.rs`, `src/lightnet-lm/tests/lm_tokenizer.rs`, `tasks/rust/llm_data.md`.
- Modify: `src/lightnet-lm/src/lib.rs`, `src/lightnet-lm/Cargo.toml`, `Cargo.lock`.
- Common version files.

**Interfaces:**

- Consumes: Task 1 errors/RNG contract.
- Produces:
  - `NormalizationSpec { latin_fold: bool }`.
  - `normalize_text(text: &str, spec: &NormalizationSpec) -> String`.
  - `ByteBpe::train(documents: &[Vec<u8>], vocab_size: usize, normalization: NormalizationSpec) -> Result<Self>`.
  - `ByteBpe::encode(&self, text: &str) -> Result<Vec<u32>>`.
  - `ByteBpe::decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>>`.
  - `ByteBpe::save(&self, path: &Path) -> Result<()>`.
  - `ByteBpe::load(path: &Path) -> Result<Self>`.
  - `ByteBpe::artifact_hash(&self) -> String`.
- IDs: PAD=0, BOS=1, EOS=2, reserved UNK=3; byte values 0–255 map to IDs 4–259. Learned merges start at 260.
- Train BPE within documents, choosing highest-frequency adjacent pair and breaking ties by ascending pair IDs. Stop when no pair repeats; distinguish requested and achieved vocabulary size.
- Normalization always applies NFC and canonical line endings. Optional Latin folding expands `æ/Æ/œ/Œ` and folds `v/V→u/U`, `j/J→i/I`; preserve case otherwise. Preserve macrons.
- Plain encoding does not interpret special-token spellings as control tokens. Loading validates merge references, ranks, IDs, and artifact version.

- [x] **Step 1: Write tokenizer tests.** `lm_tokenizer_roundtrips_bytes` covers all 256 byte tokens; `lm_tokenizer_preserves_unicode` roundtrips normalized Latin, Greek, combining marks, emoji, and unseen characters; `lm_tokenizer_latin_policy` asserts `"VJV æ œ" → "UIU ae oe"` with folding and NFC composition; `lm_tokenizer_deterministic_merges` asserts identical artifacts under repeated training and document-order permutation; `lm_tokenizer_rejects_corrupt_artifacts` rejects forward merge references, duplicate IDs, and unknown versions; `lm_tokenizer_special_literals_are_text` asserts literal marker text creates no BOS/EOS.
- [ ] **Step 2: Run** `cargo xtask test lm_tokenizer_`. Expect missing tokenizer APIs.
- [x] **Step 3: Implement** the tokenizer and canonical serialization. Add the proposed dependencies, lock their resolution, and document a bounded, explicitly selected tokenizer-training sample rather than loading an entire general corpus into RAM.
- [x] **Step 4: Run** `cargo xtask test lm_tokenizer_`, then `cargo xtask check lightnet-lm`. Expect all assertions to pass; saved/reloaded hash must match.
- [ ] **Step 5: Record evidence and commit/tag** as `v0.M.NNN: add reproducible byte fallback BPE`.

### Task 3: CPU differentiable decoder operations

**Files:**

- Create: `src/lightnet-lm/src/{ops,attention}.rs`, `src/lightnet-lm/tests/lm_ops.rs`.
- Modify: `src/lightnet-lm/src/lib.rs`.
- Common version files.

**Interfaces:**

- Consumes: `DecoderConfig`, `TokenBatch`, `LossSummary`.
- Produces:
  - `rms_norm(x: &[f32], gamma: &[f32], rows: usize, eps: f32) -> Result<Vec<f32>>`.
  - `rms_norm_backward(x: &[f32], gamma: &[f32], dy: &[f32], rows: usize, eps: f32) -> Result<(Vec<f32>, Vec<f32>)>`.
  - `embedding(table: &[f32], ids: &[u32], hidden: usize) -> Result<Vec<f32>>`.
  - `embedding_backward(ids: &[u32], dy: &[f32], vocab: usize, hidden: usize) -> Result<Vec<f32>>`.
  - `rope_in_place(x: &mut [f32], positions: &[u32], heads: usize, head_dim: usize, theta: f32, inverse: bool) -> Result<()>`.
  - `causal_gqa(q: &[f32], k: &[f32], v: &[f32], batch: &TokenBatch, cfg: &DecoderConfig) -> Result<Vec<f32>>`.
  - `causal_gqa_backward(q: &[f32], k: &[f32], v: &[f32], dy: &[f32], batch: &TokenBatch, cfg: &DecoderConfig) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)>`.
  - `swiglu(gate: &[f32], up: &[f32]) -> Result<Vec<f32>>`.
  - `swiglu_backward(gate: &[f32], up: &[f32], dy: &[f32]) -> Result<(Vec<f32>, Vec<f32>)>`.
  - `sparse_ce(logits: &[f32], targets: &[i32], vocab: usize) -> Result<(LossSummary, Vec<f32>)>`.
- Q is `[B,T,Hq,Dh]`; K/V are `[B,T,Hkv,Dh]`; Q head `h` maps to KV head `h/(Hq/Hkv)`.
- Use split-half RoPE. Backward applies its transpose rotation.
- CE returns **summed** loss and unnormalized descent seeds `one_hot(target)-softmax`. Ignored rows return zero seed. Stable log-sum-exp computes loss without probability flooring.
- Invalid/padded query rows return zero attention output; no allowed-key reduction is attempted. Non-finite active logits are errors.

- [x] **Step 1: Write operation tests.** `lm_ops_rmsnorm_gradient` checks input/scale finite differences; `lm_ops_embedding_scatter_add` uses IDs `[5,5,7]` and asserts row 5 receives two contributions; `lm_ops_rope_adjoint` checks nonzero positions and inverse rotation; `lm_ops_gqa_gradient` checks Q/K/V derivatives and shared-KV summation; `lm_ops_swiglu_gradient` checks both branches; `lm_ops_causality_and_segments` changes future/other-segment values and asserts earlier outputs are unchanged.
- [x] **Step 2: Add CE assertions.** `lm_ops_ce_descent_sign` asserts zero logits with target 1 yield seeds `[-0.5,0.5]` and loss `ln(2)`; `lm_ops_ce_matches_negative_finite_difference` compares seeds against the negative numerical loss derivative; `lm_ops_ce_mask_and_extremes` asserts ignored rows are zero, all-ignored loss/count are zero, logits `[1000,-1000]` targeting class 1 yield approximately 2000 loss, and NaN/Inf active logits error.
- [ ] **Step 3: Run** `cargo xtask test lm_ops_`. Expect missing-operation failures.
- [x] **Step 4: Implement** the interfaces without Q/K or score clamps. Use centered differences with perturbation `1e-3` and tolerance `2e-3 + 2e-2*abs(reference)` on small nonsaturated fixtures.
- [x] **Step 5: Run** `cargo xtask test lm_ops_`. Expect every named test to pass, including exact sign and boundary assertions.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: implement differentiable causal decoder operations`.

### Task 4: CPU trainable decoder and tied head

**Files:**

- Create: `src/lightnet-lm/src/decoder.rs`, `src/lightnet-lm/tests/lm_decoder.rs`.
- Modify: `src/lightnet-lm/src/{lib,types}.rs`.
- Common version files.

**Interfaces:**

- Consumes: Tasks 1 and 3.
- Produces:
  - `Parameter { id: u32, name: String, shape: Vec<usize>, values: Vec<f32>, decay: bool }`.
  - `DecoderParams { tensors: Vec<Parameter> }`.
  - `DecoderGrads { values: Vec<Vec<f32>> }`, indexed by unique parameter ID.
  - `CpuDecoder::new(cfg: DecoderConfig, seed: u64) -> Result<Self>`.
  - `CpuDecoder::forward(&self, batch: &TokenBatch, training_rng: Option<&mut RngState>) -> Result<CpuForward>`.
  - `CpuDecoder::backward(&self, forward: &CpuForward, logits_seed: &[f32]) -> Result<DecoderGrads>`.
  - `CpuForward { logits: Vec<f32>, tape: DecoderTape }`; `DecoderTape` owns opaque backward intermediates.
  - `CpuDecoder::parameters(&self) -> &DecoderParams`.
  - `CpuDecoder::set_parameters(&mut self, params: DecoderParams) -> Result<()>`.
- Block: RMSNorm → Q/K/V projections → RoPE → causal GQA → output projection → residual → RMSNorm → SwiGLU gate/up/down projections → residual. Apply final RMSNorm before the tied head.
- Bias-free projections; gamma initialized to one. Matrix initialization uses deterministic normal samples with standard deviation 0.02; attention-output and MLP-down matrices additionally scale by `1/sqrt(2*layers)`.
- Canonical parameter order: embedding; each block’s attention norm, Q, K, V, O, MLP norm, gate, up, down; final norm. Head references the embedding ID and owns no second tensor.
- Matrices decay; RMSNorm scales do not. Dropout applies to attention/MLP branch outputs before residual addition, using inverted scaling and saved masks. Inference disables dropout without consuming RNG.

- [x] **Step 1: Write decoder tests.** `lm_decoder_full_parameter_gradient` checks every parameter family against negative CE finite differences; `lm_decoder_tied_gradient` asserts the single embedding gradient equals independently calculated gather and head contributions; `lm_decoder_parameter_registry` asserts unique names/IDs, stable order, one embedding allocation, and exact count agreement.
- [x] **Step 2: Add behavioral assertions.** `lm_decoder_prefix_causality` asserts prefix logits are invariant to future IDs; `lm_decoder_padding_isolation` asserts valid logits match an unpadded run; `lm_decoder_dropout_replay` asserts equal RNG states reproduce outputs/gradients and inference leaves RNG unchanged; `lm_decoder_seed_step_lowers_loss` applies a small descent update and asserts lower synthetic CE.
- [ ] **Step 3: Run** `cargo xtask test lm_decoder_`. Expect missing decoder APIs.
- [x] **Step 4: Implement** the complete forward/backward and parameter registry. Preserve a single sign convention through every adjoint; add no compensating negation at the optimizer boundary.
- [x] **Step 5: Run** `cargo xtask test lm_decoder_`, then `cargo xtask test lm_ops_`. Expect all tests to pass. The single synthetic descent assertion is a unit test, not corpus training evidence.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: compose trainable CPU decoder with tied embeddings`.

### Task 5: Token shards and deterministic packing

**Files:**

- Create: `src/lightnet-lm/src/data.rs`, `src/lightnet-lm/tests/lm_data.rs`.
- Modify: `src/lightnet-lm/src/lib.rs`, `tasks/rust/llm_data.md`.
- Common version files.

**Interfaces:**

- Consumes: tokenizer artifact/hash and `TokenBatch`.
- Produces:
  - `CorpusDocument { id, work_id, split, source, license, sha256: String, text_path: PathBuf }`.
  - `write_token_shards(documents: &[CorpusDocument], tokenizer: &ByteBpe, out: &Path, shard_tokens: usize) -> Result<ShardManifest>`.
  - `TokenLoader::open(manifest: &Path, batch: usize, context: usize, seed: u64) -> Result<Self>`.
  - `TokenLoader::next_batch(&mut self) -> Result<Option<TokenBatch>>`.
  - `TokenLoader::state(&self) -> LoaderState`.
  - `TokenLoader::restore(&mut self, state: &LoaderState) -> Result<()>`.
- `ShardManifest` records format/version, tokenizer hash, split, ordered shard hashes, document offsets/lengths, and source metadata.
- `LoaderState` records manifest hash, shuffle RNG/epoch/order, document cursor, long-document offset, and any pending packing fragment.
- Shards use little-endian u32 tokens and explicit document boundaries. Load only bounded shard/window buffers.
- Form per-document `[BOS, body, EOS]` and independently shifted pairs. Never create EOS→next-document targets. Long-document windows preserve each eligible next-token target exactly once.
- Pack document fragments into rows with distinct segment IDs and positions reset at each fragment. Final padding uses PAD input, ignored target, `valid=false`.
- Shuffle documents deterministically; validation uses stable order.

- [x] **Step 1: Write data tests.** `lm_data_shift_and_pack` asserts exact IDs, targets, positions, segments, and padding for two short documents; `lm_data_long_document_coverage` asserts every body/EOS target occurs exactly once across windows; `lm_data_resume_cursor` asserts restored batches equal uninterrupted batches after a partial document and partially filled row.
- [x] **Step 2: Add manifest assertions.** `lm_data_rejects_corruption` rejects truncated shards, invalid IDs, checksum failures, and tokenizer mismatch; `lm_data_split_leakage` rejects the same work or normalized exact duplicate across splits; `lm_data_empty_corpus` errors on no documents while accepting an empty document as BOS→EOS.
- [ ] **Step 3: Run** `cargo xtask test lm_data_`. Expect missing loader/shard APIs.
- [x] **Step 4: Implement** streaming shard reads, deterministic packing, and restoration validation. Require supplied UTF-8 plaintext exports; document TEI/HTML extraction requirements rather than silently training on markup.
- [x] **Step 5: Run** `cargo xtask test lm_data_`, then `cargo xtask test lm_decoder_`. Expect all assertions to pass, including packed-segment isolation.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: add resumable token shards and document packing`.

### Task 6: Token training and checkpoint resume

**Files:**

- Create: `src/lightnet-lm/src/{optim,train,checkpoint}.rs`, `src/lightnet-lm/tests/lm_train.rs`, `src/lightnet-cli/src/lm.rs`.
- Modify: `src/lightnet-lm/src/{lib,types}.rs`, `src/lightnet-cli/src/main.rs`, `xtask/src/main.rs`, `CLAUDE.md`.
- Common version files.

**Interfaces:**

- Consumes: CPU decoder, shard loader, unique parameter registry, existing `lightnet_cuda::host_ops::adam_update` and `AdamParams`.
- Produces:
  - `TokenSchedule { warmup_tokens: u64, total_tokens: u64, peak_lr: f64, min_lr: f64 }`.
  - `TokenSchedule::lr(&self, updated_tokens: u64) -> Result<f32>`.
  - `UpdateOptions { lr, beta1, beta2, eps, weight_decay, clip_norm: f32, active_tokens: u64, step: u32 }`.
  - `UpdateReport { applied: bool, global_norm: f64, clip_factor: f32 }`.
  - `clip_global_norm(grads: &mut DecoderGrads, max_norm: f32) -> Result<(f64, f32)>`.
  - `LmEngine` with:
    - `loss_sum_backward(&mut self, batch: &TokenBatch) -> Result<LossSummary>`.
    - `apply_update(&mut self, options: &UpdateOptions) -> Result<UpdateReport>`.
    - `export_state(&self) -> Result<EngineState>`.
    - `import_state(&mut self, state: &EngineState) -> Result<()>`.
  - `CpuEngine`, implementing `LmEngine`.
  - `Trainer<E: LmEngine>::microbatch(&mut self, batch: &TokenBatch) -> Result<LossSummary>`.
  - `Trainer<E: LmEngine>::finish_update(&mut self) -> Result<UpdateReport>`.
  - `save_checkpoint(path: &Path, state: &TrainingSnapshot) -> Result<()>`.
  - `load_checkpoint(path: &Path) -> Result<TrainingSnapshot>`.
- `EngineState` contains config, unique f32 masters, f32 moments, pending summed gradients, model RNG, and precision-specific state.
- `TrainingSnapshot` adds schema version, tokenizer/manifest hashes, train options, loader state, pending active-token/microbatch counts, successful-update count, updated-token count, and schedule.
- Normalize accumulated gradients once by the total active targets; unscale before normalization where scaling exists. Then check finiteness, compute one global norm over unique parameters, clip, and update with AdamW using `batch=1`.
- Reject non-finite gradients before modifying any parameter/moment. All-ignored accumulation produces no update. Successful updates alone advance optimizer and schedule counters.
- Warmup: `peak_lr*t/warmup_tokens` for `t<warmup_tokens`; afterward a single cosine reaches `min_lr` at `total_tokens`, then stays there. Require `total_tokens>warmup_tokens`; warmup zero starts at peak.
- New checkpoint format uses a versioned manifest and checksummed little-endian f32 payload. Atomic publication uses a temporary sibling and rename. Serialize tied storage once; preserve pending accumulation.
- Proposed `cargo xtask lm ...` forwards to the LM CLI under the existing capped runner; allow training only through explicit `lm train`. Preserve existing commands.

- [x] **Step 1: Write optimizer tests.** `lm_train_token_normalization` compares unequal-size microbatch accumulation with one concatenated effective batch; `lm_train_global_clip_unique_params` asserts gradients `[3,4]` have norm 5 and become `[0.6,0.8]` at cap 1; `lm_train_adamw_groups` matches existing AdamW math, applies no norm-scale decay, and updates tied storage once.
- [x] **Step 2: Write schedule/resume tests.** `lm_train_token_schedule` asserts LR values 0, 1, 0.55, and 0.1 at token counts 0, 10, 20, and 30 for warmup 10/total 30/peak 1/min 0.1; `lm_train_resume_mid_accumulation` asserts bit-identical CPU next batches, dropout results, gradients, moments, counters, and parameters after save/resume; `lm_train_checkpoint_validation` rejects mismatched hashes, tensor shapes, versions, truncation, and checksums without mutating the live engine.
- [x] **Step 3: Add failure/CLI tests.** `lm_train_rejects_nonfinite_atomically` injects NaN into a norm-scale gradient and asserts all parameters/moments/counters remain unchanged; `lm_train_all_ignored_no_step` asserts no update; `lm_train_step_overflow` rejects advancement beyond `u32::MAX`; inline `lm_cli_*` tests assert strict subcommand/flag parsing and reject training options on evaluation commands.
- [ ] **Step 4: Run** `cargo xtask test lm_train_`, then `cargo xtask test lm_cli_`. Expect missing trainer/checkpoint/CLI APIs.
- [x] **Step 5: Implement** training, snapshot validation/publication, and thin CLI commands for tokenizer, shard preparation, CPU train, and resume. An incomplete checkpoint must never replace a valid checkpoint.
- [x] **Step 6: Run** the two filters again, then `cargo xtask check lightnet-cli`, sequentially. Expect all named tests to pass and check exit status zero.
- [ ] **Step 7: Record evidence and commit/tag** as `v0.M.NNN: add token training and exact CPU checkpoint resume`. Keep any corpus-overfit demonstration explicitly user-gated.

### Task 7: CPU KV-cache and sampling

**Files:**

- Create: `src/lightnet-lm/src/{cache,sampling}.rs`, `src/lightnet-lm/tests/lm_generate.rs`.
- Modify: `src/lightnet-lm/src/{lib,decoder}.rs`, `src/lightnet-cli/src/lm.rs`.
- Common version files.

**Interfaces:**

- Consumes: decoder parameters/config, tokenizer, RNG, decoder checkpoints.
- Produces:
  - `KvCache::new(cfg: &DecoderConfig, batch: usize) -> Result<Self>`.
  - `KvCache::reset(&mut self)`.
  - `CpuDecoder::prefill(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>>`.
  - `CpuDecoder::decode(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>>`.
  - `SamplingOptions { temperature: f32, top_k: usize, top_p: f32, seed: u64 }`.
  - `sample_token(logits: &[f32], options: &SamplingOptions, rng: &mut RngState) -> Result<u32>`.
- Generation initially supports batch one; reject other sizes explicitly. Prefill requires an empty cache; decode appends one token.
- Per-layer K/V cache layout is `[batch, kv_heads, context, head_dim]`, storing rotated K and unrotated V. Cache carries length and model revision; parameter replacement invalidates it.
- Empty text starts with BOS; an empty explicit prefill array errors.
- Temperature zero means greedy with lowest-ID tie break. Otherwise apply temperature, top-k, then smallest descending-probability nucleus reaching top-p; ties sort by token ID. Top-k zero disables that filter. Require finite nonnegative temperature and `0<top_p<=1`.
- Never sample PAD/BOS/reserved UNK. EOS stops generation and is not printed as text.

- [x] **Step 1: Write cache tests.** `lm_generate_cached_matches_prefix` checks every next-token logit against full-prefix forward for MHA and GQA within `1e-5 + 1e-5*abs(reference)`; `lm_generate_nonzero_rope_offset` checks multiple decode steps; `lm_generate_cache_validation` rejects overflow, stale model revision, invalid IDs, and nonempty-cache prefill without changing length/content.
- [x] **Step 2: Write sampling tests.** `lm_generate_greedy_ties` selects the lowest allowed ID; `lm_generate_seed_replay` reproduces a sampled sequence from equal RNG state; `lm_generate_filters` asserts excluded tokens never occur and a top-p=0.6 nucleus over probabilities `[0.5,0.3,0.2]` keeps the first two; `lm_generate_invalid_distribution` rejects NaN, Inf, invalid settings, and no allowed candidates; `lm_generate_eos_and_empty_prompt` asserts BOS insertion and EOS stopping.
- [ ] **Step 3: Run** `cargo xtask test lm_generate_`. Expect missing cache/sampling APIs.
- [x] **Step 4: Implement** incremental projections, cache validation before writes, sampling, and proposed `lm generate` CLI dispatch.
- [x] **Step 5: Run** `cargo xtask test lm_generate_`, then `cargo xtask test lm_cli_`. Expect all named assertions to pass.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: add cached CPU generation and seeded sampling`.

#### Phase 1 evidence (Tasks 1-7)

Implemented and verified on 2026-10-24. The named suites pass: `lm_` in
`lightnet-lm` 52 of 52 (contract 6, tokenizer 7, ops 8, decoder 7, data 6,
train 10, generate 8), `lm_cli_` 4 of 4, and `cargo xtask check lightnet-cli`
is clean. `cargo xtask clippy lightnet-lm` is clean for this crate; the CLI's
new `lm.rs` is clean, while the crate keeps pre-existing `main.rs` warnings.
`systemd-run --user` was unavailable, so the runs were uncapped; `-j 2` held.

The red phase was not separately observed for Tasks 1-5: their tests and code
were first compiled together. Tasks 6-7 also went straight to the failing
compile/test output recorded above rather than a per-test red run. Only the
final green runs are evidence.

Deviations recorded rather than hidden:

- Task 6 names `train.rs`; the engine/trainer was initially written as
  `engine.rs` and then renamed to `train.rs`, so the tree matches the plan.
- No `cargo xtask lm` wrapper was added; the plan calls it "proposed". The
  commands are reached through `lightnet-rs lm ...`, so `xtask/src/main.rs` and
  `CLAUDE.md` were not changed for Task 6.
- `KvCache` stores `batch` as a scalar and requires one, so its buffers are
  `[kv_heads, context, head_dim]` (the plan's `[batch, kv_heads, context,
  head_dim]` at batch one).
- Task 4's `lm_decoder_tied_gradient` originally compared the gather term with a
  final-norm-only adjoint, which is not the full `d_stream`; the test now checks
  the head term and the exact `head + gather == embedding gradient` split, while
  the finite-difference test covers the combined gradient.
- Task 2's "no repeated pair" fixture used raw bytes 0-255, which are not valid
  UTF-8 documents; it now uses printable ASCII one-byte documents.
- The Task 10 pilot parameter counts in this plan were arithmetically wrong.
  Against the implemented registry they are 41,955,840 at vocabulary 8192 and
  46,150,144 at vocabulary 16384, not 45,628,928 and 49,823,232.

### Task 8: Resident oxide bf16 forward

**Files:**

- Create: `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_forward.rs`.
- Modify: `src/lightnet-cuda-oxide/Cargo.toml`, `src/lightnet-cuda-oxide/src/lib.rs`, `src/lightnet-cuda-oxide/src/kernels/mod.rs`.
- Common version files.

**Interfaces:**

- Consumes: CPU decoder types/parameters and existing oxide bf16 conversions/GEMM.
- Produces:
  - `OxideDecoder::upload(backend: OxideBackend, cfg: DecoderConfig, params: &DecoderParams) -> Result<Self>`.
  - `OxideDecoder::forward(&mut self, batch: &TokenBatch, training: bool) -> Result<DeviceDecoderForward>`.
  - `OxideDecoder::download_logits(&self, forward: &DeviceDecoderForward) -> Result<Vec<f32>>`.
  - `OxideDecoder::transfer_stats(&self) -> LmTransferStats`.
  - `LmTransferStats { host_to_device_bytes: u64, device_to_host_bytes: u64 }`.
- Add `lightnet-lm` as a normal dependency of the excluded bridge. `lightnet-lm` does not depend on oxide or `lightnet-core`.
- `DeviceDecoderForward` owns logits and backward intermediates on device. Use bf16 working weights/activations, f32 master weights, and f32 normalization/softmax reductions.
- Implement native embedding, RMSNorm, RoPE, SwiGLU, residual/dropout, tied head, and baseline causal GQA. An initially quadratic attention workspace is permitted and reported explicitly.
- Transfer counters cover every LM-owned transfer, including explicit readback APIs. No trait host defaults are used.
- Native generation cache integration is not a prerequisite for forward training correctness; CPU generation remains the required baseline.

- [ ] **Step 1: Write device tests.** `oxide_lm_forward_reference` compares each primitive and complete logits against CPU calculations using bf16-rounded operands; require normalized maximum error at most 0.02 per primitive and 0.05 for complete smoke logits. `oxide_lm_forward_causality` asserts future/other-segment changes do not affect earlier valid outputs.
- [ ] **Step 2: Add residency/shape assertions.** `oxide_lm_forward_resident` snapshots transfer counters after input upload and asserts no device-to-host transfer before explicit logits download; `oxide_lm_forward_tail_shapes` covers non-tile-multiple dimensions and padded rows; `oxide_lm_forward_rejects_unsupported` asserts a clear error before allocation/launch for unsupported geometry.
- [ ] **Step 3: Run** `cargo xtask oxide --test oxide_lm_forward`. Expect missing bridge APIs.
- [ ] **Step 4: Implement** complete native forward and transfer accounting. Validate input and allocation sizes before launching kernels; prefix new kernel names with `lm_`.
- [ ] **Step 5: Run** `cargo xtask oxide --test oxide_lm_forward`, then `cargo xtask oxide --test oxide_amp`, sequentially. Expect all named actual-device tests to pass; record device, format, errors, and allocations.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: add resident bf16 decoder forward`.

### Task 9: Resident oxide backward and updates

**Files:**

- Create: `src/lightnet-cuda-oxide/tests/oxide_lm_train.rs`.
- Modify: `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `src/lightnet-cli/src/lm.rs`, `xtask/src/main.rs`, `CLAUDE.md`.
- Common version files.

**Interfaces:**

- Consumes: Task 6 `LmEngine`, `EngineState`, `UpdateOptions`, `UpdateReport`; Task 8 device forward.
- Produces:
  - `impl LmEngine for OxideDecoder`.
  - `OxideDecoder::download_gradients(&self) -> Result<DecoderGrads>` for explicit test/debug use.
  - Native adjoints for every forward operation, device sparse CE, unique f32 gradient accumulation, global-norm reduction, and AdamW updates.
- Keep gradients, moments, clipping, and updates device-resident. Normal batch uploads and explicitly requested checkpoint exports are allowed; intermediate activation/gradient downloads are not.
- bf16 uses scale 1. Reject non-finite gradients through a device-wide decision before any update. Return only small scalar reports during normal training.
- Accumulate repeated token and shared-KV contributions with a deterministic reduction order; avoid nondeterministic atomic scatter in the exact-resume mode.
- Export/import the common checkpoint state; same-backend deterministic resume is the exactness gate. CPU↔GPU migration is numerically compared, not claimed bit-identical.
- Expose explicit `--backend oxide --precision bf16`; unsupported precision/backend combinations fail.

- [ ] **Step 1: Write gradient/update tests.** `oxide_lm_train_all_parameter_gradients` compares every parameter family with the CPU reference, requiring normalized maximum error at most 0.08 on the smoke fixture; `oxide_lm_train_tied_and_repeated` checks repeated IDs plus tied-head accumulation; `oxide_lm_train_accumulation_clip_adamw` compares unequal microbatch accumulation, global clipping, and one AdamW step with the CPU reference.
- [ ] **Step 2: Add failure/residency tests.** `oxide_lm_train_no_host_fallback` asserts zero activation/gradient readbacks during backward/update; `oxide_lm_train_nonfinite_atomic` injects NaN into a norm gradient and asserts masters, moments, and counters are unchanged; `oxide_lm_train_resume` asserts same-backend deterministic continuation after a mid-accumulation checkpoint, including dropout and loader state.
- [ ] **Step 3: Run** `cargo xtask oxide --test oxide_lm_train`. Expect missing backward/engine implementation.
- [ ] **Step 4: Implement** native backward/update, checkpoint export/import, and oxide CLI dispatch through capped xtask execution.
- [ ] **Step 5: Run** `cargo xtask oxide --test oxide_lm_train`, then `cargo xtask oxide --test oxide_lm_forward`, then `cargo xtask test lm_train_`, sequentially. Expect all named suites to pass.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: add resident bf16 decoder training and resume`. Corpus convergence and performance remain unverified until separately requested runs.

### Task 10: Latin pilot and evaluation

**Files:**

- Create: `cfg/llm_latin.cfg`, `src/lightnet-lm/src/eval.rs`, `src/lightnet-lm/tests/lm_eval.rs`, `tasks/rust/llm_eval.md`.
- Modify: `src/lightnet-lm/src/{lib,data}.rs`, `src/lightnet-cli/src/lm.rs`, `tasks/rust/llm_data.md`.
- Common version files.

**Interfaces:**

- Consumes: tokenizer, shard manifest, trained/resumed decoder, generation API.
- Produces:
  - `EvalReport { nll_sum: f64, active_tokens: u64, normalized_bytes: u64, nats_per_token: f64, bits_per_byte: f64 }`.
  - `evaluate_cpu(model: &CpuDecoder, loader: &mut TokenLoader, normalized_bytes: u64) -> Result<EvalReport>`.
  - `validate_corpus_manifest(documents: &[CorpusDocument]) -> Result<()>`.
  - CLI preparation/evaluation commands and a user-gated pilot recipe.
- Pilot cfg: 12 layers, hidden 512, intermediate 1536, eight Q heads, four KV heads, head dimension 64, tied embedding, RMS epsilon `1e-5`, theta 10000, dropout zero.
- Vocabulary comparisons: 8192 and 16384 total IDs. Context comparisons: 512 and 1024.
- With the Task 4 parameter registry, the derived exact counts are **41,955,840** at vocabulary 8192 and **46,150,144** at vocabulary 16384. Assert both counts in Task 10; remain within the requested 30–60M range. The earlier 45,628,928 / 49,823,232 figures in this plan were wrong.
- Pilot defaults for a later authorized experiment: AdamW β1=0.9, β2=0.95, ε=`1e-8`, matrix decay 0.1, global clip 1.0, peak LR `3e-4`, minimum LR `3e-5`. Token budget and warmup token count must be supplied explicitly and recorded; do not invent a measured optimum.

**Data to provide:**

| Use | Required input | Acceptance requirements |
|---|---|---|
| General corpus | A selected mixture of prose/web, reference material, books, and optional code/math, supplied as plaintext documents plus manifests. | Source/version/rights metadata; language/quality filtering; normalization and dedup reports; held-out works/domains; actual byte/token counts. Instruction/dialogue data is a separate later dataset. |
| Perseus Latin | Latin-only plaintext exported from identified editions. | Preserve work/author/edition attribution; remove translations, apparatus, and TEI markup; verify rights per supplied edition. |
| Latin Library | Identified snapshot exported to plaintext. | Preserve provenance; remove page furniture; inspect scanning errors; deduplicate against other classical sources. |
| Vulgate | One explicitly identified edition and rights record. | Exclude commentary/glosses unless intentionally classified; keep biblical text from dominating the mixture; deduplicate repeated quotations. |
| CC100 Latin | Identified Latin release/subset. | Verify source terms, language quality, boilerplate, duplicates, and contamination; cap sampling contribution explicitly. |

Split by whole work before tokenizer fitting or shard construction. Group alternate editions/translations of the same work; audit normalized exact duplicates and near-duplicate passages across splits. Train BPE only on the training split. Retain original text and the normalization policy; compare optional u/v and i/j folding without silently rewriting medieval orthography.

- [ ] **Step 1: Write evaluation tests.** `lm_eval_parameter_budget` asserts both exact counts above; `lm_eval_weighted_loss` asserts aggregate NLL/count rather than average-of-batch-means; `lm_eval_bits_per_byte` asserts `nll_sum/(ln(2)*normalized_bytes)`; `lm_eval_ignores_padding` asserts padding adds neither loss nor count.
- [ ] **Step 2: Add corpus/diagnostic tests.** `lm_eval_rejects_work_leakage` rejects shared work IDs, edition groups, and duplicate normalized content across splits; `lm_eval_tokenizer_train_split_only` excludes held-out text from BPE fitting; `lm_eval_empty_denominator` rejects zero-token/zero-byte reports; `lm_eval_forced_choice` verifies length-normalized scoring on a small hand-computed agreement fixture.
- [ ] **Step 3: Run** `cargo xtask test lm_eval_`. Expect missing evaluator/config fixtures.
- [ ] **Step 4: Implement** evaluation, exact parameter-count cfgs, manifest validation, and reproducible recipes. Add a near-duplicate audit using normalized five-word shingles, flagging work pairs with Jaccard similarity at least 0.8 for review; record the reviewed exclusions.
- [ ] **Step 5: Run** `cargo xtask test lm_eval_`, then `cargo xtask test lm_data_`, sequentially. Expect all assertions to pass.
- [ ] **Step 6: Write the pilot report template** with corpus hashes/rights, author/work splits, tokenizer fertility, source mixture, parameter count, context, token budget, train/validation loss, bits/byte, fixed generation prompts, morphology/agreement diagnostics, and human Latin-quality review. Leave unmeasured cells marked **unverified**.
- [ ] **Step 7: Record evidence and commit/tag** as `v0.M.NNN: add Latin pilot configuration and evaluation`. Execute corpus training and held-out evaluation only after a user request; do not mark pilot quality established from unit tests.

### Task 11: Causal GQA flash and chunked cross-entropy

**Files:**

- Create: `src/lightnet-cuda-oxide/tests/oxide_lm_flash.rs`.
- Modify: `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `tasks/rust/llm_eval.md`.
- Common version files.

**Interfaces:**

- Consumes: resident bf16 engine and CPU causal/GQA/CE references.
- Produces:
  - Proposed `AttentionMode::{Reference, Flash}`.
  - `OxideDecoder::set_attention_mode(&mut self, mode: AttentionMode) -> Result<()>`.
  - `OxideDecoder::set_ce_chunk_tokens(&mut self, tokens: usize) -> Result<()>`.
  - `OxideDecoder::set_recompute(&mut self, enabled: bool)`.
  - `OxideDecoder::workspace_bytes(&self) -> usize`.
- Flash uses online softmax/LSE with causal and document masks; backward recomputes probabilities and sums GQA contributions into shared K/V.
- Use a separate kernel family from existing clamped vision flash.
- Chunk the **tied output projection and CE together** over token rows. Never materialize full `[B*T,V]` logits during chunked training; reuse chunk scratch and accumulate hidden/embedding gradients.
- Recomputation restores the block’s saved RNG start state without advancing the trainer RNG twice. Checkpoint identity includes execution settings.

- [ ] **Step 1: Write flash tests.** `oxide_lm_flash_matches_reference` checks forward/dQ/dK/dV for MHA, GQA, packed segments, and partial tiles; require normalized maximum error at most 0.03 for forward and 0.08 for gradients. `oxide_lm_flash_no_vision_clamps` compares finite Q/K above 128 and scores above 10 with the unclamped CPU reference.
- [ ] **Step 2: Write memory/CE tests.** `oxide_lm_flash_workspace` asserts attention scratch grows linearly when context doubles; `oxide_lm_chunked_ce_equivalence` compares loss, hidden seeds, and tied-weight gradients for chunk sizes 1, 3, and all rows; `oxide_lm_chunked_ce_no_full_logits` asserts scratch uses chunk rows rather than total token rows; `oxide_lm_flash_masked_rows` asserts all-padding rows stay finite with zero seed.
- [ ] **Step 3: Add recomputation test.** `oxide_lm_recompute_dropout` asserts identical masks, gradients, and final RNG state with recomputation enabled/disabled.
- [ ] **Step 4: Run** `cargo xtask oxide --test oxide_lm_flash`. Expect missing optimized-mode APIs.
- [ ] **Step 5: Implement** tiled causal/GQA forward/backward, token-chunked tied head/CE, and block recomputation. Keep reference mode selectable for correctness comparisons.
- [ ] **Step 6: Run** `cargo xtask oxide --test oxide_lm_flash`, then `cargo xtask oxide --test oxide_lm_train`, sequentially. Expect all tests to pass; record actual peak workspace.
- [ ] **Step 7: Record evidence and commit/tag** as `v0.M.NNN: optimize causal GQA and chunked token loss`. Claim speed improvements only after matched, separately authorized timing; workspace assertions alone establish no throughput gain.

### Task 12: Opt-in decoder FP8

**Files:**

- Create: `src/lightnet-cuda-oxide/tests/oxide_lm_fp8.rs`.
- Modify: `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`, `src/lightnet-lm/src/checkpoint.rs`, `src/lightnet-cli/src/lm.rs`, `tasks/rust/llm_eval.md`.
- Common version files.

**Interfaces:**

- Consumes: stable bf16 engine, optimized attention/CE, existing `Fp8Format`, `Fp8Scaling`, and FP8 GEMM facilities.
- Produces:
  - Proposed `DecoderPrecision::{Bf16, Fp8}`.
  - `OxideDecoder::set_precision(&mut self, precision: DecoderPrecision) -> Result<()>`.
  - Checkpointed per-logical-role FP8 history, update generation, and scaling mode.
- FP8 mode keeps bf16 resident activations/working copies and f32 masters/gradients/moments. Initially quantize eligible projection GEMMs only: E4M3 forward operands and optional E5M2 backward operands.
- RMSNorm, softmax/LSE, loss, clipping, and optimizer reductions remain f32; attention inputs remain bf16. Do not describe this as every tensor being FP8.
- Keep embedding/tied output head bf16 initially. Map projection roles explicitly; never infer eligibility from pointer identity alone.
- Invalidate quantized copies after every successful update/import. Skipped updates do not advance weight generations or training counters.
- Use current scaling by default; delayed scaling is opt-in and its history must survive resume. Preserve non-finite detection.

- [ ] **Step 1: Write FP8 tests.** `oxide_lm_fp8_quantized_reference` compares each GEMM against dequantized-operand f32 reference within the existing FP8 accumulation error convention; `oxide_lm_fp8_zero_spike_nonfinite` asserts finite zero output, saturation on spikes, and update rejection on non-finite gradients.
- [ ] **Step 2: Add lifecycle tests.** `oxide_lm_fp8_cache_refresh` asserts a successful update changes the next quantized forward; `oxide_lm_fp8_role_history` asserts different logical projection roles retain independent histories; `oxide_lm_fp8_resume` compares uninterrupted/resumed history and updates; `oxide_lm_fp8_residency` asserts no host scale readbacks.
- [ ] **Step 3: Run** `cargo xtask oxide --test oxide_lm_fp8`. Expect missing precision-mode APIs.
- [ ] **Step 4: Implement** explicit role-based FP8 dispatch, cache generations, checkpoint state, and CLI selection. Use the existing sm_89-compatible FP8 instruction family rather than introducing newer-architecture instructions.
- [ ] **Step 5: Run** `cargo xtask oxide --test oxide_lm_fp8`, then `cargo xtask oxide --test oxide_gemm_f8`, then `cargo xtask oxide --test oxide_lm_train`, sequentially. Expect all named suites to pass.
- [ ] **Step 6: Document the user-gated acceptance experiment.** Compare bf16/FP8 from identical checkpoints, token batches, and budgets. Require no non-finite updates and held-out NLL no worse than bf16 by more than 0.02 nats/token. Record calibration, warmup, steady-state tokens/second, device time, transfers, and peak memory separately; call FP8 faster only if matched end-to-end throughput improves.
- [ ] **Step 7: Record evidence and commit/tag** as `v0.M.NNN: add opt-in FP8 decoder projections`. Keep FP8 opt-in even when synthetic tests pass.

### Task 13: Optional cuTile inference

**Files:**

- Create: `src/lightnet-cuda-cutile/src/lm.rs`.
- Modify: `src/lightnet-cuda-cutile/Cargo.toml`, `src/lightnet-cuda-cutile/src/lib.rs`, `src/lightnet-cuda-cutile/src/ops_elementwise.rs`, `xtask/src/main.rs`, `CLAUDE.md`.
- Common version files.
- Reuse vendored kernels without editing their source.

**Interfaces:**

- Consumes: decoder geometry/parameters and existing vendored forward kernels.
- Produces:
  - `CutileDecoder::upload(cfg: DecoderConfig, params: &DecoderParams) -> Result<Self>`.
  - `CutileDecoder::prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>>`.
  - `CutileDecoder::decode(&mut self, id: u32) -> Result<Vec<f32>>`.
  - `CutileDecoder::reset(&mut self)`.
  - Proposed capped runner `cargo xtask cutile [test-filter]`, implemented before using it.
- Add normal dependencies on core and `cutile-kernels` only within the excluded adapter.
- Reuse f16 embedding/RMSNorm/RoPE/causal-GQA/gating/cache/head assets with explicit shape and architecture checks. Convert f32 checkpoint parameters explicitly; do not claim bf16 support from f16 kernels.
- Keep this deliverable inference only. Training/backward requests return an unsupported error.
- Repair incomplete activation matches using explicit unsupported errors for unimplemented variants. Never add an identity catch-all.

- [ ] **Step 1: Add runner and failing tests.** Inline `cutile_lm_*` tests assert prefill/decode agree with CPU logits within normalized maximum error 0.05, nonzero RoPE offsets work, GQA maps correctly, cache overflow preserves state, unsupported activations return errors, and training is rejected.
- [ ] **Step 2: Run** `cargo xtask cutile cutile_lm_`. Expect the current build failure or missing decoder API; record the actual first failure instead of assuming the activation match is the only blocker.
- [ ] **Step 3: Implement** adapter build repairs and inference wrappers. Validate the vendored cache entry’s zero-position restriction; use the suitable dynamic-position path for decoding. Reject unsupported shapes/architectures before unsafe kernel calls.
- [ ] **Step 4: Run** `cargo xtask cutile cutile_lm_`. Expect actual-device tests to pass. If toolchain/device support is unavailable, mark this task blocked and **unverified** without weakening mandatory CPU/oxide gates.
- [ ] **Step 5: Record evidence and commit/tag** as `v0.M.NNN: integrate optional cuTile decoder inference` only when the named tests are green.

### Task 14: Causal sliding-window attention

Sparse-attention family, stage 1. Context at the Latin pilot (512–1024) does not need it for speed; it is the cheap, exact baseline that Tasks 15 and 16 build on and the way to reach longer contexts later. Prior art in the repo: `lightnet-cuda/src/lib.rs` (`window_radius` mask) and `lightnet-cuda-oxide/src/kernels/layers_extra.rs` (`sliding`) for the bidirectional LAYA encoder; reuse the idea, not the code (that path is non-causal and clamped).

**Files:**

- Create: `src/lightnet-lm/tests/lm_window.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_window.rs`.
- Modify: `src/lightnet-lm/src/{config,types,attention,decoder}.rs`, `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`.
- Common version files.

**Interfaces:**

- Consumes: Tasks 3, 4, 7, 11.
- Produces:
  - `DecoderConfig` gains `window: usize` (0 = full causal) and `full_every: usize` (every N-th layer keeps full attention, 0 = none); both are accepted by `[decoder]` and rejected when `window == 0 && full_every != 0`.
  - `causal_gqa` honours the window: key `j` is allowed for query `i` only when `i - window < j <= i` (plus the existing segment and validity rules).
  - KV-cache keeps at most `window` entries for windowed layers (ring buffer); positions, not slots, drive RoPE.
  - Oxide flash skips tiles wholly outside the window (the saving is real compute, not only a mask).

- [ ] **Step 1: Write CPU tests.** `lm_window_matches_masked_full` (windowed output equals full attention with the window mask applied, MHA and GQA); `lm_window_gradient` (finite differences on Q/K/V); `lm_window_full_every_layers` (layer pattern, e.g. `full_every=3`); `lm_window_cache_matches_prefix` (ring-buffer cached decode equals full-prefix windowed logits past one full wrap).
- [ ] **Step 2: Run** `cargo xtask test lm_window_`. Expect missing config fields.
- [ ] **Step 3: Implement** config, CPU mask, windowed cache.
- [ ] **Step 4: Write GPU tests.** `oxide_lm_window_matches_cpu` (forward and dQ/dK/dV, partial tiles, packed segments); `oxide_lm_window_skips_tiles` (tile counter or workspace shows work proportional to `T * window`, not `T^2`).
- [ ] **Step 5: Run** `cargo xtask oxide --test oxide_lm_window`. Expect failures, implement the tile skipping, re-run until green.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: add causal sliding-window attention`. Speed claims need a separately authorized timing at a context where the window is smaller than the sequence.

### Task 15: Optional DSA-style indexer top-k attention

Reference: DeepSeek Sparse Attention (DeepSeek-V3.2): a lightweight "lightning indexer" scores keys per query with ReLU dot products of small separate query/key projections, a top-k selector keeps about `k` keys (reported 2,048), and attention runs only on those, so core cost is O(L·k). DeepSeek introduced it by continued training of a dense model, which also fits this repo (train dense, switch on, continue). Everything here is **unverified** against the paper; read the V3.2 report before implementing and record any deviation.

**Files:**

- Create: `src/lightnet-lm/src/sparse_dsa.rs`, `src/lightnet-lm/tests/lm_dsa.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_dsa.rs`.
- Modify: `src/lightnet-lm/src/{lib,config,types,attention,decoder,checkpoint}.rs`, `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`.
- Common version files.

**Interfaces:**

- Consumes: Tasks 3, 4, 6, 9, 11, 14.
- Produces:
  - `DecoderConfig` gains `sparse_topk: usize` (0 = off) and `indexer_heads`, `indexer_dim`.
  - Per-layer indexer parameters (`Wq_idx`, `Wk_idx`, `w_head`), registered like any other parameter with decay.
  - `dsa_scores(q_idx, k_idx, w_head, batch, cfg) -> Result<Vec<f32>>`: `I[i,j] = sum_h w_h * relu(q_h[i] . k[j])`, causal and segment masked.
  - `dsa_select(scores, k) -> Result<Vec<u32>>`: deterministic top-k per query (ties broken by ascending key index; fewer than `k` allowed keys keeps them all).
  - `causal_gqa_selected(q, k, v, selected, ...)` and its backward over the selected keys only.
  - An indexer training loss (the indexer is trained to match the dense attention distribution, for example KL on the dense attention scores, with the main loss not backpropagating into the indexer through the selection).
- Top-k selection is non-differentiable; gradients flow only through the attention over the selected set and through the indexer's own loss.
- Checkpoint format version bump: indexer tensors and `sparse_topk` are part of checkpoint identity.

- [ ] **Step 1: Write CPU tests.** `lm_dsa_scores_causal` (no future keys scored); `lm_dsa_topk_deterministic_ties`; `lm_dsa_equals_dense_when_k_covers_all` (`k >= T` reproduces dense causal attention bitwise or within `1e-6`); `lm_dsa_gradient_selected_only` (finite differences on Q/K/V with the selection frozen); `lm_dsa_indexer_loss_decreases` (a few synthetic steps lower the KL on a toy fixture); `lm_dsa_checkpoint_roundtrip`.
- [ ] **Step 2: Run** `cargo xtask test lm_dsa_`. Expect missing APIs.
- [ ] **Step 3: Implement** the CPU reference (dense scoring, then selection, then gathered attention).
- [ ] **Step 4: Write GPU tests.** `oxide_lm_dsa_matches_cpu` (scores, selected indices on tie-free data, forward, gradients); `oxide_lm_dsa_workspace` (attention scratch scales with `T * k`, scoring tiled without a full `[T,T]` buffer).
- [ ] **Step 5: Run** `cargo xtask oxide --test oxide_lm_dsa`, implement tiled scoring, top-k and gathered attention, re-run.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: add DSA-style indexer top-k attention`. State plainly that no quality or speed claim holds below the context length where `k` is smaller than the sequence.

### Task 16: Optional NSA-style three-branch attention

Reference: Native Sparse Attention (arXiv 2502.11089, ACL 2025): keys/values organised in blocks; three branches (compressed block tokens, top-k selected blocks, sliding window) combined by learned gates, with kernels aligned to GQA groups so all query heads of a group share the selected blocks. Natively trainable (used from pretraining). Hardest of the three; do it only after Tasks 14 and 15 are green and a context length where it matters is a real goal. **Unverified**: read the paper first and record every deviation.

**Files:**

- Create: `src/lightnet-lm/src/sparse_nsa.rs`, `src/lightnet-lm/tests/lm_nsa.rs`, `src/lightnet-cuda-oxide/tests/oxide_lm_nsa.rs`.
- Modify: `src/lightnet-lm/src/{lib,config,types,attention,decoder,checkpoint}.rs`, `src/lightnet-cuda-oxide/src/lm.rs`, `src/lightnet-cuda-oxide/src/kernels/lm.rs`.
- Common version files.

**Interfaces:**

- Consumes: Tasks 3, 4, 9, 11, 14, 15.
- Produces:
  - `DecoderConfig` gains `nsa_block`, `nsa_stride`, `nsa_select_blocks`, `nsa_window` (all zero = off).
  - Block compression: a small learned MLP per layer maps each key/value block to one compressed key/value.
  - Compressed branch: attention over compressed tokens (causal at block granularity); its scores also rank blocks for the selection branch.
  - Selection branch: top-`n` blocks per query group, shared across the GQA group; attention over the tokens of the selected blocks.
  - Window branch: Task 14 sliding window.
  - Gates: per-query sigmoid gate over the three branch outputs (learned linear from the hidden state).
  - `nsa_attention` and `nsa_attention_backward` CPU reference.

- [ ] **Step 1: Write CPU tests.** `lm_nsa_compression_shapes`; `lm_nsa_block_causality` (a block still being filled is never visible to earlier queries); `lm_nsa_selection_group_shared` (all query heads of a KV group use one block set); `lm_nsa_gradient` (finite differences with selection frozen, all three branches and the gates); `lm_nsa_reduces_to_window_when_other_branches_off`; `lm_nsa_checkpoint_roundtrip`.
- [ ] **Step 2: Run** `cargo xtask test lm_nsa_`. Expect missing APIs.
- [ ] **Step 3: Implement** the CPU reference.
- [ ] **Step 4: Write GPU tests.** `oxide_lm_nsa_matches_cpu` (forward and gradients on tie-free fixtures); `oxide_lm_nsa_workspace` (scratch scales with compressed length plus `n * block`, not `T^2`).
- [ ] **Step 5: Run** `cargo xtask oxide --test oxide_lm_nsa`, implement the three kernels with GQA-group-shared loads, re-run.
- [ ] **Step 6: Record evidence and commit/tag** as `v0.M.NNN: add NSA-style three-branch attention`. No quality parity or speed claim without separately authorized runs at matched context.

## Progress

| Task | Stage | Deliverable | Status | Required evidence |
|---|---|---|---|---|
| 1 | Decoder contract/tokenizer | Strict configuration and integer token layout | Implemented (CPU), verified | Contract tests 6 of 6 |
| 2 | Decoder contract/tokenizer | Byte-fallback BPE and normalization | Implemented (CPU), verified | Tokenizer tests 7 of 7 |
| 3 | CPU trainable decoder | Exact primitive adjoints and descent-sign CE | Implemented (CPU), verified | Ops tests 8 of 8 |
| 4 | CPU trainable decoder | Complete model with tied head | Implemented (CPU), verified | Decoder tests 7 of 7 |
| 5 | Loader/training/resume | Token shards and document packing | Implemented (CPU), verified | Data tests 6 of 6 |
| 6 | Loader/training/resume | AdamW, clipping, token LR, checkpoints | Implemented (CPU), verified | Train tests 10 of 10; CLI 4 of 4 |
| 7 | CPU inference | KV-cache and sampling | Implemented (CPU), verified | Generate tests 8 of 8 |
| 8 | Oxide resident bf16 | Native forward | Planned | Actual-device numerics and transfers |
| 9 | Oxide resident bf16 | Backward, optimizer, resume | Planned | All-gradient agreement and residency |
| 10 | Latin pilot/eval | Data contract, cfg, evaluator | Planned; corpus run user-gated | Unit checks; later measured pilot report |
| 11 | Causal/GQA flash and CE | Tiled attention, chunking, recomputation | Planned | Numerics and workspace gates |
| 12 | FP8 | Opt-in projection precision | Planned; accuracy/speed runs user-gated | Kernel/lifecycle tests; later matched runs |
| 13 | Optional cuTile | Cached f16 inference | Optional; buildability unverified | Actual-device adapter tests |
| 14 | Sparse attention | Causal sliding window (CPU, cache, oxide tile skipping) | Planned | Masked-full equivalence, tile-skip evidence |
| 15 | Sparse attention | DSA-style indexer top-k | Optional; paper details unverified | Dense equivalence at `k >= T`, selected-set gradients |
| 16 | Sparse attention | NSA-style compressed/selected/window branches | Optional; paper details unverified | Group sharing, block causality, gradients |

Tasks 1-7 (CPU reference path) are implemented, verified by their named unit tests, and versioned as v0.1.042. Tasks 8-16 remain unimplemented. No corpus training, evaluation, or performance result has been measured, so all runtime and performance claims remain **unverified**.

## Open issues

- **Corpus provision:** actual snapshots, edition rights, plaintext exports, work/author identifiers, and held-out lists must be supplied or separately researched. The plan deliberately adopts no unverified corpus-size or licensing claims.
- **Training authorization:** corpus training, overfit demonstrations, pilot evaluation runs, and long performance comparisons have not been requested. Their recipes can be implemented and reviewed before execution.
- **Hardware fit:** 30–60M parameters does not establish that bf16 masters, gradients, Adam moments, activations, and scratch fit the available device. Measure allocations before selecting batch size or accumulation.
- **General-model scale:** the implementation is initially single-device. Multi-GPU communication, optimizer sharding, fault-tolerant distributed loading, imported-model conversion, and large-scale instruction tuning require separate plans.
- **Release-line policy:** AGENTS.md assigns CPU/GPU minor lines to the existing port roadmap, while this is a new feature. Use those CPU/GPU lines consistently and check current tags/mapping before each increment; do not revive the closed line or use the MoE line.
- **Version-map format:** the inspected mapping tail contains an additional trailing field beyond the handbook’s four-field description. Preserve the live file’s established format when appending; do not rewrite historical rows.
- **Exactness boundary:** exact resume is required within the same deterministic backend/configuration. CPU↔GPU migration and different GPU reduction implementations use numerical tolerances, not bitwise claims.
- **Tokenizer dependency (resolved):** `unicode-normalization 0.1` and `sha2 0.10` are resolved in `Cargo.lock`; the pinned toolchain built and passed the `lm_` tests.
- **Baseline health:** existing full-suite evidence gaps remain separate from this feature’s focused tests. Do not report the entire repository green from LM filters.
- **Sparse attention scope:** Tasks 14–16 are researched from search summaries (DSA from the DeepSeek-V3.2 write-ups, NSA from arXiv 2502.11089); nothing was read in full. MoBA (block routing) and Trainable Dynamic Mask Sparse Attention (arXiv 2508.02124) are candidates left out; no 2026 papers were found. At the 512–1024 pilot context none of these speeds up training; they pay off only when `k` or the window is smaller than the sequence.
- **cuTile compatibility:** source assets are available, but adapter build, kernel architecture support, and driver/toolchain compatibility remain unverified. Optional integration must not block the CPU/oxide sequence.
- **Scope coverage:** Tasks 1–13 cover every requested stage and missing operation; Tasks 14–16 add the sparse-attention family. Resident fp16 decoder training is not implied by the bf16 deliverable; adding it requires explicit precision support, overflow/scaler state, and its own end-to-end tests.
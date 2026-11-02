# Latin LLM: pretraining and post-training plan

This plan takes the downloaded Latin sources to a reproducible base language model, then to a Latin instruction/chat model. It uses the existing `lightnet-lm` decoder and the opt-in `lightnet-examples/llm` preparation example. Optional domain continuation and preference tuning are separate experiments.

**Execution boundary:** documentation and source preparation only. No build, corpus processing, tokenizer fitting, model training, synthetic-data generation or external-model calls are authorized by this document. The user's current instruction is **no build yet**. Commands below are for later execution.

**Evidence date:** 2026-10-25, local source inspection. Proposed settings are starting experiments, not measured results or promises of Latin fluency.

## Table of contents

- [1. Current implementation and gaps](#1-current-implementation-and-gaps)
- [2. Relevant project tree](#2-relevant-project-tree)
- [3. Component breakdown](#3-component-breakdown)
- [4. Latin tokenizer](#4-latin-tokenizer)
- [5. Data and split contract](#5-data-and-split-contract)
- [6. Pretraining](#6-pretraining)
- [7. Optional continued pretraining](#7-optional-continued-pretraining)
- [8. Supervised post-training](#8-supervised-post-training)
- [9. Optional preference tuning](#9-optional-preference-tuning)
- [10. Evaluation and acceptance](#10-evaluation-and-acceptance)
- [11. Implementation sequence](#11-implementation-sequence)
- [12. Later execution and deliverables](#12-later-execution-and-deliverables)
- [13. Open decisions and references](#13-open-decisions-and-references)

## 1. Current implementation and gaps

| Area | Present in source | Remaining work or evidence |
|---|---|---|
| CPU decoder | Pre-RMSNorm causal GQA, RoPE, SwiGLU, tied embeddings/output head | Future validation must use the live source; no new build or runtime proof in this task |
| Tokenizer | Trainable byte-fallback BPE, NFC, optional Latin orthographic folding, version-1 artifacts | No corpus-fitted tokenizer; chat controls/version 2 not implemented |
| Pretraining data | Native token shards, manifests, document-aware packing, resumable loader | No completed real-corpus preparation or token counts |
| Preparation code | XML/CoNLL-U extraction, provenance, work deduplication, overlap screening, sample selection | Five extraction fixtures passed earlier; full pipeline remains unvalidated |
| Preparation example | `llm_prepare_corpus`, gated by `lightnet-examples/llm` | Source added; Rust example and feature wiring uncompiled |
| Optimization and resume | CPU AdamW, token-count warmup/cosine schedule, training snapshots | Need later real-data resume and stability evidence |
| Generation | CPU KV cache and sampling; `lightnet-rs lm generate` | Chat session serialization, history truncation and termination need implementation |
| Native GPU LM training | Detailed decoder plan exists | Resident LM forward/backward/update and parity evidence remain prerequisites; detector/ViT kernels do not establish an LM training engine |
| Post-training | Recipe design exists | Chat controls, assistant masking, SFT dataset/loader, stage orchestration and preference objective remain to implement |

The earlier corpus-processing attempt was stopped during the overlap audit. An existing `datasets/latin/prepared/` directory is not evidence of a ready dataset. A tokenizer artifact or model checkpoint must not be claimed to exist until successfully produced and verified.

## 2. Relevant project tree

`[existing]` means present in source, not necessarily compiled or validated. `[planned]` means a proposed addition. Generated artifacts do not exist as completed outputs yet.

```text
lightnet/
|-- Cargo.toml                                      [existing] workspace
|-- docs/
|   `-- LLM_PLAN.md                                 this roadmap
|-- datasets/latin/
|   |-- sources.tsv                                [existing] pinned source revisions
|   |-- SHA256SUMS                                 [existing] download archive hashes
|   |-- raw/                                       [existing] original snapshots
|   |   |-- perseus/
|   |   |-- lascivaroma/
|   |   |-- ud-llct/
|   |   `-- ud-circse/
|   |-- pretraining-v1/                            [generated later]
|   |   |-- documents.tsv                          corpus registry
|   |   |-- text/                                  normalized Latin surface text
|   |   |-- tokenizer-sample.tsv
|   |   |-- tokenizer-sample.json
|   |   |-- sample/                                train-only text samples
|   |   |-- provenance.jsonl
|   |   |-- excluded.json
|   |   |-- overlap-audit.json
|   |   |-- report.json
|   |   |-- tokenizer-8192.lmbt
|   |   |-- tokenization.tsv
|   |   `-- shards/<split>/                        tokenizer, manifest, .lmt shards
|   `-- posttraining-v1/                           [planned generated outputs]
|       |-- conversations.jsonl
|       |-- preferences.jsonl                      optional
|       |-- provenance.jsonl
|       `-- report.json
|-- tools/
|   |-- prepare_latin.py                            [existing] extraction/audit
|   |-- test_prepare_latin.py                       [existing] extraction fixtures
|   `-- latin_pretraining.md                        [existing] preparation usage
|-- cfg/
|   |-- llm_smoke.cfg                               [existing] synthetic tiny model
|   |-- llm_latin_base.cfg                          [planned] pilot geometry
|   `-- llm/                                       [planned] run recipes
|       |-- pretrain.json
|       |-- continue.json                          optional
|       `-- sft.json
|-- src/
|   |-- lightnet-lm/
|   |   |-- Cargo.toml
|   |   |-- CONTEXT.md
|   |   |-- src/
|   |   |   |-- config.rs                          [existing] strict decoder config
|   |   |   |-- tokenizer.rs                       [existing] version 1; extend
|   |   |   |-- data.rs                            [existing] shards and packing
|   |   |   |-- types.rs                           [existing] integer token batches
|   |   |   |-- ops.rs                             [existing] differentiable ops
|   |   |   |-- attention.rs                       [existing] CPU causal attention
|   |   |   |-- decoder.rs                         [existing] CPU decoder
|   |   |   |-- optim.rs                           [existing] AdamW/schedule support
|   |   |   |-- train.rs                           [existing] engine/trainer
|   |   |   |-- checkpoint.rs                      [existing] training snapshots
|   |   |   |-- cache.rs                           [existing] KV cache
|   |   |   |-- sampling.rs                        [existing] token generation
|   |   |   |-- chat.rs                            [planned] versioned template
|   |   |   |-- sft.rs                             [planned] assistant masks/loader
|   |   |   |-- recipe.rs                          [planned] stages/configuration
|   |   |   |-- replay.rs                          [planned] resumable source mixing
|   |   |   |-- recipe_eval.rs                     [planned] evaluation/reporting
|   |   |   `-- preference.rs                      [planned, optional] DPO
|   |   `-- tests/
|   |       |-- lm_contract.rs
|   |       |-- lm_tokenizer.rs
|   |       |-- lm_data.rs
|   |       |-- lm_ops.rs
|   |       |-- lm_decoder.rs
|   |       |-- lm_train.rs
|   |       |-- lm_generate.rs
|   |       `-- recipe_*.rs                        [planned] post-training tests
|   |-- lightnet-examples/
|   |   |-- Cargo.toml                             [existing] optional llm feature
|   |   `-- examples/llm_prepare_corpus.rs          [existing] native preparation
|   |-- lightnet-cli/src/
|   |   |-- lm.rs                                  [existing] tokenizer/shards/train/
|   |   |                                          resume/eval/generate commands
|   |   `-- lm_chat.rs                             [planned] interactive chat
|   `-- lightnet-cuda-oxide/
|       |-- Cargo.toml                             excluded from workspace
|       |-- src/lm*.rs                             [planned] resident LM engine
|       `-- tests/oxide_lm*.rs                     [planned] CPU/device comparisons
`-- tasks/rust/
    |-- llm_data.md                                artifact/data contract
    |-- latin_datasets.md                          source inventory
    |-- plan_llm_decoder.md                        decoder implementation detail
    `-- plan_llm_training_recipe.md                A/B/C recipe implementation detail
```

## 3. Component breakdown

| Component | Role and responsibilities | Dependencies and consumers | Constraints/open questions |
|---|---|---|---|
| Workspace and crate manifests | Select crates, optional dependencies and example targets | `lightnet-examples/llm` enables `lightnet-lm` and hashing for the example | Feature gates the example; it does not make every LM dependency across the workspace optional |
| `raw/`, `sources.tsv`, download checksums | Preserve original editions, licences and immutable snapshot identity | Inputs to the extractor and provenance audit | Corpus sizes are not unique Latin text/token counts |
| `prepare_latin.py`, extraction tests | Extract surface text, normalize, group works, exclude duplicates, split and audit | Raw snapshots in; document TSV, provenance and sample out | Additional original-source rights and validation/test overlap need review; never silently repair malformed XML |
| Prepared text, TSVs and reports | Declare the admitted corpus and train-only sample | Consumed by the Rust preparation example and LM CLI | Seven TSV fields: ID, work ID, split, source, licence, SHA-256, text path |
| `tokenizer.rs` and sample registry | Fit, save, validate and apply a Latin-trained tokenizer | Training-only sample in; tied embedding dimensions and every shard depend on its identity | Freeze normalization and chat-control IDs before a reusable base-model run |
| `data.rs`, `types.rs`, native shards | Produce integer next-token batches with correct boundaries | Tokenizer and document registry in; CPU/device trainer out | BOS/EOS per document; no cross-document targets; padding is loss-free |
| `config.rs`, model cfg | Define strict decoder geometry and exact parameter count | Used by decoder, checkpoint validation and generation | Match the achieved tokenizer vocabulary; no unknown keys or unsupported modes |
| `ops.rs`, `attention.rs`, `decoder.rs` | Reference forward/backward and tied parameter storage | Consume config/batches; used by engine and device parity tests | Exact derivatives, causality and accumulated repeated-token embedding gradients |
| `optim.rs`, `train.rs`, `checkpoint.rs` | Apply updates and resume with matching optimizer/data/RNG state | Decoder, loader and schedules; consumed by all stages | Normalize once by active targets; rejected updates must not advance successful-update counters |
| `cache.rs`, `sampling.rs` | Cached generation from accepted checkpoints | Decoder and tokenizer; used by generation/chat | Cached/full-prefix agreement; explicit context overflow behavior |
| `chat.rs`, `sft.rs` | Serialize typed conversations and construct assistant-only targets | Version-2 tokenizer and native batch contract; consumed by SFT/chat | Literal role spellings cannot inject controls; supervise assistant end-of-turn |
| `recipe.rs`, `replay.rs`, recipe JSON | Stage budgets, source mixing, transitions and resume | Engine, datasets and checkpoint metadata; consumed by run orchestration | Proposed JSON schema, not current CLI support; record reset/continuation policy explicitly |
| `recipe_eval.rs`, evaluation fixtures | Frozen LM and instruction evaluation | Held-out manifests, prompts and checkpoints; stage gate reports out | No test-driven model selection; Latin human review for chat claims |
| Optional `preference.rs` | Preference objective after an accepted SFT baseline | Human-reviewed pairs, SFT initialization and frozen reference | Separate implementation and compute budget; no promised benefit |
| Examples and CLI | Keep preparation and training interfaces thin | Public `lightnet-lm` API and optional device engine | Existing commands must not be confused with proposed recipe/chat commands |
| Oxide LM bridge/tests | Resident GPU forward/backward/update plus parity tests | Reference contracts and supported GPU primitives | Native detector primitives are insufficient; no silent CPU fallback |
| This roadmap and task plans | Keep sequence, evidence boundaries and acceptance criteria explicit | Refer to source and detailed task plans | Older plan descriptions of scaffolds may be stale; inspect live code before implementation |

## 4. Latin tokenizer

### What exists

There is a tokenizer implementation suitable for Latin: `lightnet_lm::ByteBpe`. It is a trainable byte-level BPE, not an already-fitted Latin vocabulary. Byte fallback can represent any valid UTF-8 Latin input, including inflected forms not seen during fitting.

Version 1 reserves PAD=0, BOS=1, EOS=2 and UNK=3. Byte IDs are 4–259; learned merges start at 260. A requested vocabulary can finish smaller if no pair appears twice. Record requested and achieved vocabulary rather than assuming they match.

Default normalization is canonical line endings and NFC. Latin folding is off: retain macrons, historical spelling, u/v and i/j. An optional ablation expands ligatures and folds v/j to u/i, but is lossy. Source extraction also normalizes layout whitespace; record this separately from tokenizer normalization.

### Fitting and selection

1. Freeze admitted training, validation and test works before fitting.
2. Select an explicit, deterministic, bounded sample from training only. The current helper caps it at 512 KiB total and 4096 UTF-8 bytes per work, round-robin by author. Measure whether this pilot sample represents the admitted corpus; do not assume optimality.
3. Start with an 8192-ID target. Compare 16384 only as a separate, budgeted experiment; the current example hardcodes 8192.
4. Record sample IDs/hashes, source shares, normalization, artifact version, achieved merge count and artifact SHA-256.
5. Evaluate held-out token fertility, bytes/token, morphological examples and byte round trips. Compare different vocabularies on the same text using bits per byte or matched-task evaluation; raw token perplexities are not directly comparable across tokenizers.
6. Freeze the selected tokenizer for pretraining, continuation, SFT, preferences and generation. Store its hash in every manifest/checkpoint.

### Chat-compatible vocabulary is a prerequisite

The existing preparation example writes a **version-1 plain-text tokenizer**. That is suitable for disposable pipeline tests, but the proposed chat recipe requires a version-2 artifact before training the reusable base model.

Follow recipe task R2: preserve low IDs and reserve four controls at the top of the total vocabulary for system, user, assistant and end-of-turn. At total vocabulary 8192 those proposed IDs are 8188–8191. Merges must never occupy them. Artifact identity covers the control map; loaders, model sampling and validation must reject undefined IDs if fitting stops early.

Implement and test this first, then update the preparation example to select version 2 for the chat-capable base model. Do not reinterpret existing merge IDs as chat controls or silently add controls to a pretrained version-1 checkpoint. A migration would be a separate design and experiment.

## 5. Data and split contract

Use admitted Perseus Latin as the primary source. Additional lascivaroma material is conditional on original-text rights and unique content after overlap removal. The downloaded corpus contains annotation overlays and alternate editions; the two repositories are not additive measures of text volume.

The existing preparation policy is:

- One retained edition per known CTS work; prefer canonical Perseus to annotated mirrors.
- Latin surface text only: exclude translations, apparatus, notes and annotation-only duplicated enclitics. Preserve verse/paragraph boundaries where available.
- Seneca held out for validation, Tacitus for test; remaining authors deterministically grouped with nominal 90/5/5 assignment by author count. Actual bytes/tokens per split must be measured.
- LLCT charters and CIRCSE literary texts remain separate diagnostics. Their published train/dev/test labels do not determine LM training admission.
- Original and folded five-word shingle audits screen training against held-out works. The current heuristic quarantines a training work at at least 20 shared shingles and 20% containment of the smaller work. Lower overlaps are reported, not automatically declared harmless.

Before freezing a production dataset, extend/review the audit across validation and test, group known author/edition aliases, inspect excerpts and translations, and adjudicate unresolved overlap. CIRCSE diagnostic text may overlap held-out Seneca/Tacitus editions: report it as a diagnostic overlay, not an independent test population. Record unknown authorship rather than claiming author independence.

Every admitted document needs stable ID, canonical work ID, split, source URL/revision, original-file hash, extracted-text hash, licence and text path. Retain raw inputs. Record every exclusion and extraction error. Do not convert a repository annotation licence into a claim about all underlying texts.

No fixed token budget follows from archive sizes. Measure unique eligible next-token targets **after** extraction, deduplication, splits and tokenizer selection. BOS/EOS, padding, packed input positions and active prediction targets are distinct counts.

## 6. Pretraining

### Objective and baseline

Train causal next-token prediction on admitted Latin. The result is a base completion model; instruction following and chat behavior are separate post-training objectives.

Use the existing dense decoder with tied embeddings, RMSNorm, RoPE, GQA and SwiGLU. Start with a small pilot; the smoke cfg is for correctness tests, not a useful Latin model. Proposed geometry for the pilot:

```ini
[decoder]
vocab_size=8192
context=512
hidden=512
intermediate=1536
layers=10
q_heads=8
kv_heads=2
rope_theta=10000
rms_eps=0.00001
dropout=0
tie_embeddings=1
```

This uses current parser key names and targets roughly 34M parameters. Confirm the exact count with `DecoderConfig::parameter_count()` and set `vocab_size` from the selected artifact's defined ID space before any run. Geometry, memory fit and throughput are unmeasured.

Keep context 512 for the first experiment. Evaluate 1024 later with matched documents and token budgets. Do not introduce recursive layers, MoE, sparse attention, FP8 or architecture changes into the first baseline.

### Proposed optimizer and accounting

| Item | Initial experiment | Required record |
|---|---|---|
| Optimizer | AdamW, beta1=0.9, beta2=0.95, epsilon=1e-8 | Exact update and decay policy; tied tensor updated once |
| Learning rate | Start with a bounded pilot around 3e-4; tune on validation only | Peak/min LR and successful-update token counter |
| Schedule | Existing token-count warmup plus cosine; proposed warmup 3%, final LR 0.1 of peak | Actual token budget and stage-local origin |
| Weight decay | Proposed 0.1 baseline | Explicit eligibility by parameter family |
| Clipping | Proposed global norm 1.0 | Preclip norm and skipped/non-finite updates |
| Effective batch | Proposed target 32,768 active prediction targets/update | Actual active targets, microbatch size and accumulation |
| Execution | CPU reference for correctness; resident GPU engine for practical corpus runs after its acceptance | Backend, precision, hardware and measured peak memory |
| GPU precision | Proposed bf16 work buffers with f32 master state/gradients/moments/reductions | Unsupported formats/shapes must fail explicitly |

These are unvalidated experimental defaults. Tune fewer variables at first. The current CPU trainer exists; the resident GPU LM engine remains implementation work. Never label a CPU fallback as GPU training.

Let `U` be the measured unique training prediction targets. Choose the authorized total successful-update budget from `U` and an explicit exposure cap. A proposed first ceiling is `4 * U`, not a requirement to run four epochs or a claim of optimality. For mixed sources also enforce per-source and per-work exposure bounds; a small source must not be repeated arbitrarily to meet a fixed mixture ratio.

Evaluate at regular successful-update token intervals, recording per-work NLL, bits/byte, train/validation gap, source exposure and fixed-seed continuations. Measure steady-state throughput separately from loading, fitting, warmup, evaluation and checkpoint I/O. Stop at the authorized token/exposure budget or a predeclared plateau/overfit rule. Keep the best validation checkpoint separately from the latest resumable checkpoint.

### Resume contract

A resume must reproduce the next batch, shuffle state, accumulation position, schedule point and subsequent same-backend update. Preserve tokenizer/config/manifest hashes, model parameters, Adam moments, gradients if accumulating, RNG, document/window offsets and successful-update counters. Rejected or all-masked updates must not silently advance the schedule. Stage changes are explicit transitions, not ordinary resumes with new data substituted under an old manifest hash.

## 7. Optional continued pretraining

Continue next-token training on a curated domain only if the base model has a measured capability gap worth addressing. Keep the tokenizer and decoder geometry fixed. Mix earlier training text through resumable replay and measure both domain improvement and broad-Latin forgetting.

The existing detailed recipe proposes a **Cicero adaptation** experiment. That requires a different admission policy from the generic helper: all Cicero must be excluded from stage-A data and tokenizer fitting, with complete Cicero works reserved for stage-B development/test. The current helper does not implement that stage-specific exclusion. Add it before adopting this curriculum; do not claim adaptation to an unseen author if Cicero already appeared in the base corpus.

For that experiment, proposed evaluation-only groups are `Laelius de Amicitia` and the whole `Epistulae ad Atticum` collection, subject to actual work mapping and rights checks. Assign distinct development/test roles before selection. Other Cicero works may be stage-B training candidates after review. Evaluation is then work-held-out within Cicero, not author-held-out.

Proposed experiments use replay fractions 5%, 15% or 25% and a peak LR below the selected pretraining peak. These are hypotheses. Track cumulative A+B exposure and compare with an unchanged base checkpoint. Skip this stage for the first general Latin instruction model unless there is an explicit domain goal.

## 8. Supervised post-training

### Dataset and tasks

Create a separate, versioned instruction dataset after the base tokenizer and held-out works are frozen. Do not treat literary prose as if it were instruction-response supervision.

Include reviewed Latin tasks such as question answering grounded in supplied passages, grammatical explanation, inflection, paraphrase, summarization, instruction following and short multi-turn dialogue. Translation tasks are optional and require suitable bilingual examples; Latin-only pretraining does not establish translation quality. Separate genuine literary dialogue, editorial adaptations, rule-generated examples and external-model-generated examples in provenance.

Proposed JSONL record contract:

```json
{"id":"sft-000001","group_id":"authored-example-001","split":"train","source":"human-authored","license":"recorded-explicitly","messages":[{"role":"user","content":"Responde Latine: quid est liber?"},{"role":"assistant","content":"Liber est opus scriptum quod legere possumus."}]}
```

This is a proposed post-training schema, not an existing CLI input format. Add schema validation and provenance fields for source works, revisions, transformations and content hashes. Group alternate prompts, revisions, shared passages and adaptation variants before splitting. Never derive SFT training material from base validation/test works, including translations or paraphrases. Synthetic-data generators must not receive held-out passages.

Start with a small, human-reviewed seed set; scale only after quality and leakage audits. External generation requires a separate request, generator/version provenance, admissible terms, filtering and reviewer evidence. No fixed example count is justified by the current sources.

### Template and loss

Use typed roles and the versioned controls:

```text
BOS SYSTEM system_text END_TURN
USER user_text END_TURN
ASSISTANT assistant_text END_TURN
```

The template is proposed until `chat.rs` and artifact version 2 exist. Literal spellings inside user text must stay ordinary text bytes; only the serializer inserts controls. Do not duplicate BOS or append EOS after every message.

At input position `j`, predict token `j+1`. Activate the loss only if **the predicted token** belongs to assistant content or the assistant end-of-turn token. System/user text, assistant role openers and padding use `targets=-100`. Masking the input position instead is an off-by-one error. Preserve segment isolation between packed conversations and normalize accumulated gradients once by actual assistant targets.

Report input tokens, assistant targets, examples, exposures and successful optimizer-update tokens separately. Fully masked examples are rejected or skipped without an optimizer step. Use a lower LR than pretraining and initially compare one to three passes as bounded experiments; select by held-out assistant loss and instruction evaluation rather than training loss alone.

### Chat serving

Generation appends the assistant opener, stops on end-of-turn/EOS and enforces context limits. Truncation retains the required system policy and complete recent turns, then rebuilds the cache consistently. Test full-prefix versus cached logits after truncation and at nonzero offsets.

A Cicero-style persona is optional, clearly disclosed as a simulation, and evaluated separately from general Latin instruction following. Stylistic imitation does not establish historical truth or authentic quotations. If quotations are offered as verified, use a source lookup that returns an exact reviewed passage and provenance.

## 9. Optional preference tuning

Only consider preferences after an accepted SFT baseline and a useful held-out rubric exist. This is not required for the first working model.

A proposed DPO experiment consumes grouped prompt/chosen/rejected records with licences, provenance and reviewer rationale. Split prompt families before training. Pin a frozen SFT reference, mask prompt tokens, sum response token log-probabilities consistently, and test policy/reference identity and chosen/rejected orientation. Sequence normalization and beta are explicit experimental choices. DPO is additional implementation work; the current sparse next-token CE path is not a preference trainer.

Compare against the unmodified SFT checkpoint on blinded, order-balanced, length-aware evaluations. Retain preferences only if they improve the targeted behavior without unacceptable Latin-quality or instruction regressions. Do not add RL or claim its benefit without a separate design, reward data and compute budget.

## 10. Evaluation and acceptance

| Gate | Checks required before advancing |
|---|---|
| Corpus ready | Complete provenance, explicit exclusions, reviewed work aliases and overlap, frozen splits, no held-out tokenizer inputs, measured source/split sizes |
| Tokenizer ready | Round-trip UTF-8/macrons, artifact/hash validation, deterministic fitting, requested/achieved vocab, literal-control protection, version-2 controls for chat path |
| Training engine ready | Causality, segment isolation, ignored-target zero gradients, tied updates, uneven accumulation, interrupted resume, finite-update handling; device parity if using GPU |
| Base model accepted | Validation improves over initialization under an explicit budget; per-domain losses/generations and repetition/memorization review; test not used for selection |
| Continued model accepted | Domain development improves against base, broad-Latin regression measured and within a predeclared bound; cumulative exposures recorded |
| SFT ready | Validated message schema, grouped split, source leakage audit, shifted assistant/EOT mask tests, packed examples isolated, empty-target behavior tested |
| Chat model accepted | Frozen instruction and multi-turn suite, Latin grammar/coherence review, grounded answer checks, termination/context handling, comparison with base/continued model |
| Optional preferences accepted | Improvement over SFT on the target rubric, checked regressions and memorization, reproducible reference/objective/pair provenance |

Use validation/development sets for model and recipe selection; run the frozen test set once the selection is fixed. Report LLCT and CIRCSE separately. LM loss on treebank surface text is not a parsing, tagging or morphology benchmark score without a task-specific evaluator.

Prefer per-work/author/domain results to a single corpus aggregate. Use bits/byte for cross-tokenizer comparisons under the same normalization and text denominator. Record all exclusions and uncertainty. Human-reviewed Latin fluency, useful instruction following and historical accuracy cannot be inferred from passing synthetic unit tests.

## 11. Implementation sequence

Each step produces reviewable code before any authorized execution. Read live crate context files and the applicable detailed plan before implementation. These checkboxes describe roadmap acceptance, not work completed by writing this document.

- [ ] **P1 — Validate preparation code:** source review and later fixture/compile checks for extractor and `llm_prepare_corpus`; check failure/partial-output behavior. Preserve the `llm` feature gate.
- [ ] **P2 — Freeze the curriculum:** choose generic Latin baseline or the Cicero A/B experiment; implement any stage-specific exclusions before sample selection. Freeze work aliases, split maps and diagnostics.
- [ ] **P3 — Complete chat-compatible tokenizer:** implement version 2, reserved controls, template and ID validation; keep version 1 readable; update the example and tests. Do this before reusable base-model training.
- [ ] **P4 — Prepare and audit data when requested:** run extraction, audit validation/test overlap, review exclusions, fit the selected tokenizer, write native shards and verify loader batches. Record actual corpus/token statistics.
- [ ] **P5 — Complete training prerequisites:** exact parameter/config checks, resume/accumulation tests, and resident GPU LM engine if selected. Verify CPU/device agreement and memory fit before a corpus run.
- [ ] **P6 — Implement run recipes and evaluation:** strict configuration, token budgets, deterministic source/replay selection, stage transitions, immutable identities, checkpoint/evaluation intervals and exposure counters.
- [ ] **P7 — Execute a bounded pretraining pilot when requested:** select on validation, record stable throughput/memory and preserve best plus resumable checkpoints. Scale only after reviewing evidence.
- [ ] **P8 — Optional domain continuation:** implement/validate replay and stage-state policy, then compare with the base checkpoint under a separate budget.
- [ ] **P9 — Implement SFT preparation and loss:** typed schema, grouping, provenance, serializer, shifted assistant/EOT masks, packing and resumable loader; add mask/gradient/leakage fixtures.
- [ ] **P10 — Execute bounded SFT when requested:** validate assistant-target accounting, select on development, evaluate against the base model and freeze chat behavior.
- [ ] **P11 — Implement chat and optional preferences:** cache/termination/context tests first; preferences require their own baseline comparison and objective tests.
- [ ] **P12 — Package accepted artifacts:** checkpoint, matching tokenizer/template/config, data/run hashes, evaluation report, licences/attributions, model card and reproduction/resume commands.

Keep architecture optimizations and recursive/encoder-decoder extensions out of this baseline until their separate contracts and experiments are accepted.

## 12. Later execution and deliverables

Builds/checks must use at most four workers and the required soft RAM cap. Keep build/test commands sequential. Preserve the user's current no-build boundary; nothing below is a request to run now.

The current preparation entry points, for a later authorized **version-1 plain-text pipeline test**, are:

```sh
systemd-run --user --scope -p MemoryHigh=3G \
  python3 tools/prepare_latin.py --out datasets/latin/pretraining-v1

systemd-run --user --scope -p MemoryHigh=3G \
  cargo run -p lightnet-examples --release -j 4 \
  --features llm --example llm_prepare_corpus -- datasets/latin/pretraining-v1
```

Use a fresh output directory; do not reuse an interrupted preparation. For a reusable chat-capable base model, complete P3 and update these entry points to the chosen artifact contract first. Future recipe JSON files, `lm chat` and preference commands in this roadmap are proposed; they are not executable interfaces today.

Expected artifact chain:

```text
pinned sources -> admitted works/splits -> frozen tokenizer -> native token shards
  -> base checkpoint -> optional continued checkpoint -> SFT checkpoint
  -> optional preference checkpoint -> tokenizer/template-matched chat package
```

Every model package needs exact configuration, tokenizer/template version, data/manifest hashes, checkpoint identity, objective and stage history, exposure counters, evaluation results and known limits. Keep the selected checkpoint and its required resume state; do not delete referenced artifacts as disposable scratch files.

## 13. Open decisions and references

Decisions still requiring evidence or a future experiment:

- Whether additional lascivaroma sources add enough admissible unique text beyond Perseus.
- Generic Latin pretraining versus a curriculum that excludes Cicero from stage A.
- Representative tokenizer sample size, 8192 versus 16384, and whether folding helps the selected tasks.
- Exact per-domain mixture and repetition limits after measured extraction/tokenization.
- GPU LM backend readiness, precision support and measured context/batch memory fit.
- Post-training task distribution, source material, Latin reviewer coverage and optional bilingual tasks.
- Numeric acceptance thresholds and run budgets, fixed before examining final test results.
- Whether optional domain continuation, persona behavior or preferences justify their additional work.

Local references:

- [Latin data/tokenizer contract](../tasks/rust/03_plans/llm/llm_data.md)
- [Source inventory and rights notes](../tasks/rust/03_plans/llm/latin_datasets.md)
- [Decoder implementation plan](../tasks/rust/03_plans/llm/plan_llm_decoder.md)
- [Detailed A/B/C recipe plan](../tasks/rust/03_plans/llm/plan_llm_training_recipe.md)
- [Preparation usage and current limitations](../tools/latin_pretraining.md)
- [Current LM component context](../src/lightnet-lm/CONTEXT.md)

Those detailed plans contain historical starting-state descriptions and proposed APIs. This roadmap uses the live source as the implementation inventory and does not promote older proposals to implemented features.

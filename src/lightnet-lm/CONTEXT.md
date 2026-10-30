# lightnet-lm

Decoder-only LLM training and inference (pre-RMSNorm causal GQA, RoPE, SwiGLU, tied head). v0.1.042 implements CPU Tasks 1-7, covered by 52 passing `lm_` tests plus 4 CLI parser tests. The plan is `tasks/rust/03_plans/llm/plan_llm_decoder.md`.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Independent of the detector graph in `lightnet-core`. The `[transformer]` layer (Swin-style) and the frozen LAYA encoder stay in `lightnet-core`.

## Responsibilities

- Decoder config and token types, byte-fallback BPE (`tokenizer.rs`), CPU ops (`ops.rs`, `attention.rs`), decoder (`decoder.rs`), token shards/loader (`data.rs`), optimizer (`optim.rs`), engine/trainer (`train.rs`), checkpoints (`checkpoint.rs`), and KV-cache/sampling (`cache.rs`, `sampling.rs`).
- Public API is re-exported from `lib.rs`; the CLI entry is `lightnet-rs lm` (`src/lightnet-cli/src/lm.rs`).
- The CPU code is the reference that the oxide and cuTile device engines check against.

## Dependencies

- Depends on `lightnet-cuda` (AdamW math). Does not depend on `lightnet-core` or any GPU backend.
- `lightnet-cuda-oxide` and `lightnet-cuda-cutile` depend on this crate for the decoder contract.

## Constraints

- New cross-entropy keeps the repo's descent-direction sign (target minus prediction).
- Build and test with `cargo xtask check lightnet-lm` and `PKGS=lightnet-lm cargo xtask test lm_`.

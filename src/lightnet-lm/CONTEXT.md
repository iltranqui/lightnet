# lightnet-lm

Decoder-only LLM training and inference (pre-RMSNorm causal GQA, RoPE, SwiGLU, tied head). Scaffold only; the work is planned in `tasks/rust/plan_llm_decoder.md`.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Independent of the detector graph in `lightnet-core`. The `[transformer]` layer (Swin-style) and the frozen LAYA encoder stay in `lightnet-core`.

## Responsibilities

- Decoder config and token types, byte-fallback BPE, CPU differentiable ops and decoder, token shards, trainer, checkpoints, KV-cache and sampling (planned Tasks 1 to 7).
- The CPU code is the reference that the oxide and cuTile device engines check against.

## Dependencies

- Depends on `lightnet-cuda` (AdamW math). Does not depend on `lightnet-core` or any GPU backend.
- `lightnet-cuda-oxide` and `lightnet-cuda-cutile` depend on this crate for the decoder contract.

## Constraints

- New cross-entropy keeps the repo's descent-direction sign (target minus prediction).
- Build and test with `cargo xtask check lightnet-lm` and `cargo xtask test`.

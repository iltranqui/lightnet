# lightnet-examples

Rust equivalents of Darknet's `src-examples` binaries.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Keeps the original example split (one binary per example in `src/bin/`) with shared helpers in `lib.rs`.

## Responsibilities

- Model loading, image annotation, detection formatting, output naming.
- ffmpeg and ffplay process handling for video examples.
- Opt-in LLM corpus preparation in `examples/llm_prepare_corpus.rs` under the `llm` feature.

## Dependencies

- Depends on: `lightnet-core`.
- Feature `llm` additionally enables optional `lightnet-lm` and `sha2` dependencies.
- Depended on by: nothing.

## Constraints

- Detector examples call `lightnet-core`; LLM examples call `lightnet-lm`. No new model logic here.
- `llm_prepare_corpus` requires `--features llm`; normal builds exclude it.
- Video examples need ffmpeg or ffplay on `PATH`.

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

## Dependencies

- Depends on: `lightnet-core`.
- Depended on by: nothing.

## Constraints

- Examples call `lightnet-core` only; no new model logic here.
- Video examples need ffmpeg or ffplay on `PATH`.

# lightnet-cli

The `lightnet-rs` command-line binary.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Thin front end: parses commands, loads models and data, prints Darknet-style summaries.

## Responsibilities

- Commands for train, detect, evaluate and others, delegating all semantics to `lightnet-core`.
- `detect-server` mode: line-based JSON protocol used by `lightnet-gui-engine`.
- Optional GPU path via feature `oxide`.

## Dependencies

- Depends on: `lightnet-core`, `lightnet-cuda`, optional `lightnet-cuda-oxide`.
- Depended on by: `lightnet-gui-engine` runs it as a child process (no crate dependency).

## Constraints

- Keep logic out of the CLI; graph and training semantics belong in `lightnet-core`.
- Changing the `detect-server` protocol breaks the GUI: update `lightnet-gui-engine` together.
- No training runs unless the user asks.

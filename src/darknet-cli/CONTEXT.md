# darknet-cli

The `darknet-rs` command-line binary.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Thin front end: parses commands, loads models and data, prints Darknet-style summaries.

## Responsibilities

- Commands for train, detect, evaluate and others, delegating all semantics to `darknet-core`.
- `detect-server` mode: line-based JSON protocol used by `darknet-gui-engine`.
- Optional GPU path via feature `oxide`.

## Dependencies

- Depends on: `darknet-core`, `darknet-cuda`, optional `darknet-cuda-oxide`.
- Depended on by: `darknet-gui-engine` runs it as a child process (no crate dependency).

## Constraints

- Keep logic out of the CLI; graph and training semantics belong in `darknet-core`.
- Changing the `detect-server` protocol breaks the GUI: update `darknet-gui-engine` together.
- No training runs unless the user asks.

# lightnet-gui-engine

Child-process bridge from the GUI to `lightnet-rs detect-server`.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Keeps model inference out of the GUI process. One request per line on stdin, one JSON reply per line on stdout.

## Responsibilities

- Spawn and supervise the child; expose `DetectWorker` and `Prediction`.
- Turn any child failure, malformed reply or timeout into `Err` so the UI shows a banner and keeps editing.

## Dependencies

- Depends on: `serde`, `serde_json` only. Runs `lightnet-cli` (`lightnet-rs`) at runtime.
- Depended on by: `lightnet-gui`.

## Constraints

- Never panic on model output.
- The line protocol is a contract with `lightnet-cli detect-server`; change both together.

# cfg

Model configuration files, label lists and `.data` files. Factory material: stable inputs to `lightnet-core`'s cfg parser.

## Table of Contents

- [Contents](#contents)
- [Constraints](#constraints)

## Contents

- `*.cfg`: Darknet network definitions (YOLO, CSP, darknet53, efficientnet, and others).
- `*.data`, `*.names`, `*.labels`, `*.map`: dataset descriptors and class lists (for example `coco.data`, `coco.names`).

Parsed by `src/lightnet-core/src/cfg.rs`. Layer coverage per kind: `tasks/rust/layer-support.md`.

## Constraints

- Never commit model weights here; `*.weights` is gitignored.
- An unknown config key must fail in the parser, not be edited out of a cfg.

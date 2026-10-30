# 01_rules

Stable rules that apply to every run. Change only when the operator changes a rule.

## Table of Contents

- [Files](#files)
- [Reading order](#reading-order)

## Files

| File | Holds |
|---|---|
| [rules.md](rules.md) | Hard rules (threads, memory, edits, commits) and doc hygiene |
| [commands.md](commands.md) | `cargo xtask` commands |
| [versioning.md](versioning.md) | Version lines, commit and tag on green, per-version docs |
| [layer-contract.md](layer-contract.md) | Layout and the Layer Completion Contract |

## Reading order

`rules.md` and `commands.md` before any build; `versioning.md` before any commit; `layer-contract.md` before touching a layer.

# lessons.md — wip-gpu-parity

Corrections from the operator, newest last. Read this before starting work on the
branch.

## 2026-10-31 — GUI work: write the whole feature set, then show it

Operator: "do this, write all the code for all the features, and I will see them
once the code is written."

Rule: when the operator asks for a whole feature set (here the DarkMark editor
layout), write the code for all of it in one pass, keeping `cargo xtask check`
and the crate tests green, and commit. Do not stop after every item to ask for a
visual review; the operator looks at the result once it is written. Report what
could not be verified instead of iterating on screenshots.

Still true: one writer in the main session, no subagents, `cargo xtask` only,
commit only the files the work names.

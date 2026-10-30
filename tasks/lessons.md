# Lessons

## Contents

- [Scope](#scope)

## Scope

- When the user names a reference branch, inspect that branch directly before deriving feature scope or behavior from the currently checked-out branch.
- Detection and segmentation port scope includes all associated layers, data, training, inference, and export behavior from Darknet `raptor-fp4`; do not narrow it to ONNX.

- Latest operator update, 2026-10-25: Cargo checks and tests are now authorized. Use at most two build threads, one cargo xtask check/test invocation at a time, with MemoryHigh=3G. This supersedes the earlier source-only check/test restriction; training remains outside scope.

- A handoff followed by "do this, write code" authorizes implementation of its current priority. Continue the documented source-only workflow and write a concrete feature slice rather than repeating the handoff summary. User correction, 2026-10-25.

- Prioritize full backward, native GPU updates, FP16 and every-layer completion before the standalone CPU detection-parity milestone. Preserve the source-only restriction and local WIP-commit workflow. User steering, 2026-10-25.

- When the user asks to prepare code for pretraining data, implement the reusable pipeline and validate it with fixtures. Do not start processing the real corpus or fitting a tokenizer unless execution is requested. User correction, 2026-10-25.
- "No build yet" excludes Cargo build/check/run/test commands. Finish source changes and document validation as pending; do not request build approval. User correction, 2026-10-25.
- For the all-layer host/device task, write source only: no Cargo check, build, run, test, training or runtime execution. Keep verification pending. User subsequently authorized local WIP commits without Cargo validation; do not create release tags. User corrections, 2026-10-25.

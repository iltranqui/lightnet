# Lessons (branch rust)

## Contents

- [Lessons](#lessons)

## Lessons

- Run builds/checks ONE package/command at a time; do not chain several -p crates or bundle temp-edit+build+restore in one command. (User correction 2026-09-30)
- Model naming: user means codex model **gpt-6-luna** (NOT gpt-5.6-luna). Do not translate model names to the skill's listed variants; use exactly what the user types. (2026-09-30)
- Plan source paths can be stale: use `git grep` on the named reference branch to find moved routines before stopping for a missing reference. The weighted OBB NMS code is in `src-lib/box.cpp`, not `yolo_layer.cpp` (2026-10-16).
- `f32::NAN` is not equal to itself; NaN-preservation tests should assert `.is_nan()` rather than compare whole values with `PartialEq` (2026-10-16).

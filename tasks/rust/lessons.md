# Lessons (branch rust)

## Contents

- [Lessons](#lessons)

## Lessons

- Run builds/checks ONE package/command at a time; do not chain several -p crates or bundle temp-edit+build+restore in one command. (User correction 2026-09-30)
- Model naming: user means codex model **gpt-6-luna** (NOT gpt-5.6-luna). Do not translate model names to the skill's listed variants; use exactly what the user types. (2026-09-30)
- Plan source paths can be stale: use `git grep` on the named reference branch to find moved routines before stopping for a missing reference. The weighted OBB NMS code is in `src-lib/box.cpp`, not `yolo_layer.cpp` (2026-10-16).
- `f32::NAN` is not equal to itself; NaN-preservation tests should assert `.is_nan()` rather than compare whole values with `PartialEq` (2026-10-16).
- When splitting one OBB loss across box-channel and front-channel passes, each pass must own only its partial gradient; do not write the same front-anchor center-chain term into both passes (2026-10-16).
- Test filters must match Rust test-name prefixes: `carpn_map` did not select newly named `obb_map_*` dispatcher tests. Check matched-test counts before treating a focused filter as coverage (2026-10-17).
- Axis fixtures for batch/sequence stacks must account for every normalization stage; the LAYA encoder test's terminal LayerNorm follows the embedding LayerNorm, so its hand expectation must include both (2026-10-17).
- A bulk text replacement in a 45k-line file can land on the wrong test: replacing `assert!(gpu.requires_cpu_backward_fallback())` intended for the `[deform_conv]` test silently weakened the `[vit]` guard instead. Grep the modified assertion afterwards, and run the filtered suite that owns it (`test_host.sh runtime::`) before committing. (2026-10-18)
- Do not commit a runtime change without running the `runtime::` host filter from the plan's gate list; the `deform_conv` filter alone does not cover guards in other tests. (2026-10-18)
- Fixtures that hit a piecewise seam make a *correct* adjoint look wrong. This has now happened five times: integer sample coordinates (floor-cell slope), `scaled == 0` in the `lhtan` map, `|logit| == 80` in the sigmoid clamp, and offsets landing exactly on an integer base. Before "fixing" an analytic gradient that disagrees with finite differences, check whether the perturbation straddles a seam; if it does, move the fixture, not the code. (2026-10-18)
- "Update the todo list" can mean the whole `tasks/rust` folder: when the request is short, check the scope in one question before editing only one file.
- A `sed` that replaces a table row by its first cell also hits every other table that lists the same name (the layer-support rows for `Wmhf` and `Vit` appear in four tables). Replace by line number after reading the table's header.
- Docs drift behind WIP commits: check `git log <last-tag>..HEAD` and the runtime dispatch arms before trusting a "none" or "host" cell.

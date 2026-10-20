# Task Plan: CARPN End-to-End mAP Integration Plan

## Table of Contents

- [Goal](#goal)
- [Phases](#phases)
- [Key Questions](#key-questions)
- [Decisions Made](#decisions-made)
- [Errors Encountered](#errors-encountered)
- [Status](#status)

## Goal
Preserve the completed caller-supplied CARPN mAP dispatch and expand the remaining CARPN network inference/validation integration into a source-grounded, testable plan before implementation.

## Phases
- [x] Phase 1: Inspect current Rust evaluator, existing CARPN helpers, tests, and reference C++ behavior.
- [x] Phase 2: Identify compatibility decisions, edge cases, and environment-dependent validation gates; record evidence in `carpn_notes.md`.
- [x] Phase 3: Write `tasks/rust/carpn_map_plan.md` with sequenced tasks, test gates, file scope, and blocked conditions.
- [x] Phase 4: Review the plan against the source and project handbook; update the Rust task handoff.
- [x] Phase 5: Add failing dispatcher tests, implement the bounded API dispatch, and run focused host tests.
- [x] Phase 6: Complete version/status bookkeeping, commit only task files, and record the remaining network-level boundary.
- [x] Phase 7: Audit Rust cfg/model/runtime/detection support and the reference CARPN stage-2 model/inference path; findings recorded in `carpn_notes.md`.
- [x] Phase 8: Define bounded implementation waves, interfaces, deterministic tests, and separate data/C++-parity gates in `tasks/rust/carpn_network_map_plan.md`.
- [x] Phase 9: Review plan dependencies and update the Rust handbook/backlog to point to the new plan.

## Key Questions
1. Which geometries does CARPN mAP consume, and how does the reference construct and compare them? Answered in `carpn_notes.md` and `carpn_network_map_plan.md`.
2. Which missing model/config/runtime interfaces block real Rust network decoding? Answered by the source audit; see the implementation waves.
3. Which deterministic fixtures can validate end-to-end network mAP without canonical assets? A tiny fixed-weight executable network plus generated image/label fixture is specified.
4. Which claims require unavailable data or a compiled-C++ oracle? Dataset smoke and C++ parity are explicitly separate external gates.

## Decisions Made
- Keep caller-supplied sample dispatch as the completed v0.0.258 boundary; the new plan targets CPU network inference, `detect-image`, and direct `detector map`.
- Defer CARPN training and scheduled `detector train -map` until Stage-2 assignment/loss/backward/update support exists; reject that training path rather than silently training Stage 1 only.
- Preserve all existing user changes and keep the unrelated `.gitignore` edit untouched.

## Errors Encountered
- `MEM=3G ./tools/test_host.sh carpn_map` did not match the new `obb_map_*` tests; switched to the precise `obb_map` filter.
- The first `obb_map` run failed the two new tests as intended: current code rejected CARPN records and lacked mixed-geometry diagnostics. Implemented the dispatch and reran successfully.

## Status
**Planning task complete; implementation not started.** Prior sample-level API dispatch is complete in `v0.0.258`. Source audit and bounded CPU inference/direct-map plan are recorded in `carpn_notes.md` and `tasks/rust/carpn_network_map_plan.md`. CARPN `detector train -map` remains explicitly deferred until Stage-2 training support exists.

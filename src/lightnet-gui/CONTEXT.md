# lightnet-gui

egui/eframe annotation GUI: draw detection boxes and save YOLO labels.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Presentation layer only. File semantics live in `lightnet-gui-project`; model inference lives in `lightnet-gui-engine`.

## Responsibilities

- Top bar, class and image panels, canvas, modal windows (launcher, new project, class editor, help).
- Wire UI events to `lightnet-gui-project` actions and `DetectWorker` predictions.

## Dependencies

- Depends on: `lightnet-gui-project`, `lightnet-gui-engine`.
- Depended on by: nothing.

## Constraints

- Opt-in: build with `-p lightnet-gui`; not in `default-members` because it pulls egui and eframe.
- Keep logic that can be tested without a window in `lightnet-gui-project`.
- Plan: `tasks/rust/03_plans/misc/gui_plan.md` (partly implemented).

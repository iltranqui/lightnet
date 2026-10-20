# darknet-gui

egui/eframe annotation GUI: draw detection boxes and save YOLO labels.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Presentation layer only. File semantics live in `darknet-gui-project`; model inference lives in `darknet-gui-engine`.

## Responsibilities

- Top bar, class and image panels, canvas, modal windows (launcher, new project, class editor, help).
- Wire UI events to `darknet-gui-project` actions and `DetectWorker` predictions.

## Dependencies

- Depends on: `darknet-gui-project`, `darknet-gui-engine`.
- Depended on by: nothing.

## Constraints

- Opt-in: build with `-p darknet-gui`; not in `default-members` because it pulls egui and eframe.
- Keep logic that can be tested without a window in `darknet-gui-project`.
- Plan: `tasks/rust/gui_plan.md` (partly implemented).

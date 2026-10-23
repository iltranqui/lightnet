# lightnet-gui-project

Pure, UI-free project and annotation logic for the GUI.

## Table of Contents

- [Role](#role)
- [Responsibilities](#responsibilities)
- [Dependencies](#dependencies)
- [Constraints](#constraints)

## Role

Everything the annotation GUI does to files and state, testable without a window.

## Responsibilities

- Sidecar IO (`annotation`), image scan and project index (`project`), filter and navigation (`filter`, `nav`), undo (`undo`).
- DarkMark-style mouse and edit model (`edit`), hotkeys, bulk `actions`, project `create`, atomic class remap (`classes`).

## Dependencies

- Depends on: no crate in this repo.
- Depended on by: `lightnet-gui`.

## Constraints

- No UI types and no model inference here.
- Class remapping must stay atomic: never leave labels half-rewritten.

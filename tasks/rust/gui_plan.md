# darknet-gui (egui, DarkMark replacement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status: IMPLEMENTED in v0.1.019 (`fcd23bbd`), with the gaps listed below.**
Supersedes `archive/gui_iced_plan.md` and `archive/gui_code_phase_plan.md`.

| Milestone | State |
|---|---|
| M0 window spike | done (replaced by the real shell) |
| M1 files/filters/undo | done: annotation JSON, project scan, filter/sort/nav, undo |
| M2 canvas editing | done: `Editor` model, cache, zoom/pan, save policy, unsaved flag |
| M3 hotkeys/actions/views | done: key map, nudge/scale/copy/interpolate/clipboard/snap/homography, six views; screenshot open |
| M4 launcher/creation | done: recents (capped at 20), new project; delete-project open |
| M5 class editing | partial: rename with dry run + atomic apply + backup; add/delete from the dialog open (engine supports both) |
| M6 inference overlay | done: `detect-server`, worker with timeout, `p`/`a`/`A`/`o`, threshold keys; heatmaps and `j` jump open |

Still open on top of the milestones: delete-image/undo-delete, screenshot save,
heatmaps, jump, the train runner, and `T1.5` (file diff against a real DarkMark
install).

**Goal:** A desktop app in Rust that replaces DarkMark's detector-mode labeling: open or create a project, edit boxes with every DarkMark detector hotkey, edit classes, and overlay predictions from the Rust port.

**Architecture:** Three crates. `darknet-gui-project` (pure logic, no UI, fully unit-tested), `darknet-gui-engine` (spawns `darknet-rs detect-server` as a child process, JSON lines), `darknet-gui` (eframe/egui shell, `glow` renderer). Files on disk are byte-compatible with DarkMark so both tools can share a project.

**Tech Stack:** Rust nightly-2026-08-28, edition 2024; `eframe`/`egui` with the `glow` feature only (pin an exact version at least 2 weeks old, verify on crates.io in Task 0.1); `serde`, `serde_json` (no `preserve_order`), `image`, `imageproc` (Canny only), `tempfile` (dev).

**Spec:** Decisions from the 2026-10-17 design interview (recorded below). DarkMark behaviour spec: tables G1–G17 and D1–D16 in `archive/gui_code_phase_plan.md` ("DarkMark ground truth", "Deviations"); they remain authoritative and are cited here as `Gn`/`Dn`. DarkMark source: `/home/kerrigan/src/DarkSuite/DarkMark` (hotkeys in `src-tools/HotkeyWnd.cpp`, menu in `src-darkmark/DMContent.cpp`).

## Table of Contents

- [Global Constraints](#global-constraints)
- [Scope](#scope)
- [Review Focus](#review-focus)
- [File Structure](#file-structure)
- [M0 — WSLg spike](#m0--wslg-spike)
- [M1 — Pure crates: files, filters, undo](#m1--pure-crates-files-filters-undo)
- [M2 — Canvas editing](#m2--canvas-editing)
- [M3 — Remaining hotkeys and views](#m3--remaining-hotkeys-and-views)
- [M4 — Launcher and project creation](#m4--launcher-and-project-creation)
- [M5 — Class editing](#m5--class-editing)
- [M6 — Inference overlay](#m6--inference-overlay)
- [Out of scope](#out-of-scope)
- [Self-review](#self-review)

## Global Constraints

- Builds: max 4 threads (`-j 2`), `MEM=3G` / `systemd-run --user --scope -p MemoryHigh=3G`, one cargo command at a time. Pure crates: `MEM=3G cargo test -j 2 -p darknet-gui-project`.
- `darknet-gui-project` and `darknet-gui-engine` are default workspace members; `darknet-gui` is opt-in (`-p darknet-gui`, not in `default-members`).
- Detector mode only. Non-detector JSON (string or integer `annotation_mode`, kinds `obb`/`segmentor`/`cuboid`) errors and is never overwritten (G6, D3). Unknown JSON fields survive a round trip.
- Model files come from the project dir (cfg, weights, `.names`). GUI settings live in a separate GUI-owned file; nothing shared with DarkMark's config.
- Designed for 50,000 images per project: lazy decode, LRU image cache, annotation-count index built on load, no SQLite.
- Hotkeys are identical to DarkMark's. Deviations only where a key has no meaning in v1.
- Version commits `v0.0.NNN: <what>` per `AGENTS.md` "Versioning"; stage by path, never `git add -A`. Edited `.md` files keep a TOC.
- No training runs. Parity claims are source-derived plus file diff vs real DarkMark; never "C++ parity".

## Scope

In v1: draw/move/resize, class keys, zoom/pan, navigation, sort, filters (regex, empty, unannotated, annotation age, per class), snap, copy from neighbour (`y`/`Y`), mark-empty (`n`), clear, nudge/scale, clipboard, merge/interpolate, homography, B&W/Canny/dot/bold/shade/label views, zoom-review, screenshot, save `.json`+`.txt`, project-load validation, per-image undo/redo, launcher with create/open/recent/delete, class editing with remap, inference overlay with prediction hotkeys (`a`/`A`, `p`, Up/Down, `h`/`H`, `o`).

## Review Focus

1. Class remap interrupted halfway (power loss, error on file N): originals must be recoverable and no label may point at the wrong class (T5.1).
2. Image folder with 50,000 files, mixed case extensions, `chart*.png`, `*_mask.png` and cache dirs: listing ignores G2 exclusions and stays responsive (T1.2).
3. Click on a corner or a collapsed drag must not delete or create a degenerate mark (G9, D8) (T2.1).
4. Saving an image with no marks and no empty flag deletes `.json` and `.txt` and never leaves an empty file (G4, G5) (T1.1).
5. Child process dies, hangs, or returns malformed JSON during inference: UI stays usable and shows an error (T6.2).

## File Structure

```
src/darknet-gui-project/src/
  annotation.rs   NormRect, Mark, Annotation; JSON IO; delete-vs-write rules   (T1.1)
  yolo.rs         .txt export, 10 decimals                                      (T1.2)
  project.rs      DataFile, Project load, image scan, mark index, validation    (T1.2)
  filter.rs       Filter, SortKey, apply                                        (T1.3)
  nav.rs          step_index, find_unmarked (PageUp/Down)                       (T1.3)
  undo.rs         UndoStack<Snapshot> per image                                 (T1.4)
  edit.rs         Editor: DarkMark mouse model, predictions in marks            (T2.1)
  hotkeys.rs      Key, KeyInput, Action, map_key                                (T3.1)
  actions.rs      snap, nudge, scale, copy, clipboard, interpolate, homography  (T3.2)
  create.rs       create_project (writes .names/.data/.cfg refs)                (T4.1)
  classes.rs      ClassEdit, plan_remap, apply_remap (atomic, backup)           (T5.1)
src/darknet-gui-engine/src/
  detect_worker.rs  spawn detect-server, request/response JSON lines            (T6.2)
src/darknet-gui/src/
  main.rs app.rs canvas.rs cache.rs views.rs launcher.rs settings.rs
src/darknet-cli/src/main.rs     `detect-server` arm                              (T6.1)
```

Tests: one integration file per module under each crate's `tests/`; fixtures in `src/darknet-gui-project/tests/fixtures/` (copy a real LegoGears_v2 `.json`/`.txt` pair).

## M0 — WSLg spike

### Task 0.1: Window opens and draws an image with a box

**Files:** Create `src/darknet-gui/{Cargo.toml,src/main.rs}`; modify root `Cargo.toml` (add members, keep `darknet-gui` out of `default-members`).

**Produces:** a window showing a PNG texture and one rectangle, run via `cargo run -p darknet-gui -j 2`.

- [ ] **Step 1:** Look up the newest `eframe` release at least 2 weeks old; pin it exactly with `default-features = false, features = ["glow", "default_fonts"]`.
- [ ] **Step 2:** Implement `main()` that loads `sample.png` as an egui texture and paints one `Rect` outline.
- [ ] **Step 3:** `MEM=3G cargo run -p darknet-gui -j 2`. Pass = window appears, image and box visible, pan by drag works. Fail = record the error; fallback order: `LIBGL_ALWAYS_SOFTWARE=1`, then a native Windows build, then ask the user.
- [ ] **Step 4:** Record result and RSS in `STATUS.md`; commit `Cargo.toml` and `src/darknet-gui`.

## M1 — Pure crates: files, filters, undo

### Task 1.1: Annotation JSON IO and save rules

**Files:** Create `annotation.rs`; test `tests/annotation_roundtrip.rs`.

**Produces:** `Annotation::from_json(&str) -> Result<Annotation>`, `Annotation::to_json(&self) -> String` (`dump(1,'\t')+"\n"`, G3), `fn save(ann: &Annotation, json: &Path, txt: &Path) -> Result<()>` applying G4/G5.

- [ ] **Step 1:** Tests: `roundtrip_preserves_unknown_fields`; `rejects_obb_string_and_integer_mode` (G6); `rejects_polygon_with_5_points` (D4); `no_marks_unflagged_deletes_json_and_txt`; `flagged_empty_writes_empty_txt_and_completely_empty_true`; `txt_uses_10_decimals_unclamped` (G5).
- [ ] **Step 2:** Run, expect FAIL (types missing). Implement per G3–G7, D1–D4, D16.
- [ ] **Step 3:** Run, expect PASS. Commit.

### Task 1.2: Project load, image scan, `.txt` export, index

**Files:** Create `project.rs`, `yolo.rs`; test `tests/project_yolo.rs`.

**Produces:** `Project::load(dir: &Path) -> Result<Project>` (validates files against detector mode, D5), `Project::images() -> &[ImageEntry]` sorted alphabetically, `Project::mark_count(idx) -> usize`.

- [ ] **Step 1:** Tests: `scan_ignores_chart_mask_cache_dirs` (G2, D6); `prefers_dirname_data_file` (D5); `load_rejects_non_detector_json`; `index_counts_marks_json_flag_counts_one` (G13); `scan_50k_images_under_2s` (generated empty files in a tempdir; allow 5 s on this machine).
- [ ] **Step 2:** Run FAIL, implement, run PASS. Commit.

### Task 1.3: Filters, sort, PageUp/PageDown navigation

**Files:** Create `filter.rs`, `nav.rs`; test `tests/nav.rs`.

**Produces:** `Filter { include: Option<Regex>, exclude: Option<Regex>, empty: Tri, unannotated: Tri, min_age: Option<Duration>, class: Option<usize> }`, `fn apply(&Project, &Filter, SortKey) -> Vec<usize>`, `fn find_unmarked(&Project, from: usize, dir: Dir) -> usize`.

- [ ] **Step 1:** Tests: `left_right_clamp_no_wrap` (G13); `pagedown_lands_on_boundary_image_even_if_marked`; `filter_per_class_and_regex_combine`; `sort_by_mark_count_is_stable`.
- [ ] **Step 2:** Run FAIL, implement, PASS. Commit.

### Task 1.4: Per-image undo/redo

**Files:** Create `undo.rs`; test `tests/undo.rs`.

**Produces:** `UndoStack::push(&mut self, before: Annotation)`, `undo(&mut self, current: Annotation) -> Option<Annotation>`, `redo(...)`. Capped at 100 entries per image; cleared on image change.

- [ ] **Step 1:** Tests: `undo_then_redo_restores`; `push_after_undo_clears_redo`; `cap_drops_oldest`.
- [ ] **Step 2:** Run FAIL, implement, PASS. Commit.

### Task 1.5: File diff against real DarkMark

- [ ] **Step 1:** User (or script) loads one shared project in real DarkMark and in a throwaway `examples/roundtrip.rs` that load-saves with no edits. `diff -r` of `.json`/`.txt` must be empty except `timestamp`/`version` (D16).
- [ ] **Step 2:** Record result in `STATUS.md`. Gate: M2 does not start with a non-empty diff.

## M2 — Canvas editing

### Task 2.1: Editor mouse model

**Files:** Create `edit.rs`; test `tests/edit.rs`.

**Produces:** `Editor::new(Annotation, class_count) -> Editor`; `press/drag/release(&mut self, p: NormPoint, mods: Mods)`; `select_next/prev` (Tab, G12); `set_class(&mut self, digit: u8, mods: Mods) -> Result<()>` (additive Ctrl/Alt, refuses out-of-range, G11, D9); `marks()` includes predictions (D10).

- [ ] **Step 1:** Tests: `press_selects_smallest_area_mark`; `corner_press_inside_rect_redrags_from_opposite_corner` (G9); `collapse_restores_original` (D8); `new_mark_clamped_to_unit_square` (G10); `class_beyond_list_is_refused` (G11); `tab_cycles_including_predictions` (G12).
- [ ] **Step 2:** Run FAIL, implement, PASS. Every mutation pushes onto `UndoStack`. Commit.

### Task 2.2: egui canvas, image cache, save policy

**Files:** Create `canvas.rs`, `cache.rs`, `app.rs` in `darknet-gui`.

**Produces:** `ImageCache::get(&mut self, path) -> TextureHandle` (LRU, 64 images or 1 GiB decoded), canvas with zoom (`+`/`-` 10%, wheel), Ctrl-drag pan, SPACE auto/manual zoom; save on image change, close, Ctrl+Shift+S, with an unsaved indicator (G8, D12).

- [ ] **Step 1:** Unit-test `ImageCache` eviction (`evicts_least_recently_used_over_budget`) with fake loaders.
- [ ] **Step 2:** Implement canvas and wire `Editor`; run `cargo run -p darknet-gui -j 2`.
- [ ] **Step 3:** Manual smoke by the user: label 5 images on a real project, quit, reload; compare with real DarkMark (Task 1.5 diff again). Commit.

## M3 — Remaining hotkeys and views

### Task 3.1: Key map

**Files:** Create `hotkeys.rs`; test `tests/hotkeys.rs`.

**Produces:** `enum Action` with one variant per detector-mode key in `HotkeyWnd.cpp`; `fn map_key(KeyInput) -> Option<Action>`.

- [ ] **Step 1:** Extract the detector-mode key list from `src-tools/HotkeyWnd.cpp` into a table in the test (`every_documented_detector_key_maps`), excluding OBB/seg/cuboid keys and menu-only bulk operations.
- [ ] **Step 2:** Run FAIL, implement, PASS. Commit.

### Task 3.2: Editing actions

**Files:** Create `actions.rs`; test `tests/actions.rs`.

**Produces:** `snap(&mut Editor, &RgbaImage, which: SnapScope)`, `nudge_all(&mut Editor, dx: i32, dy: i32)`, `scale_all(&mut Editor, pct: f32)`, `copy_from_neighbour(&mut Editor, &Annotation)` (skips identical marks, G16), `clipboard_{copy,paste}` (JSON), `interpolate(a: &Annotation, b: &Annotation, t: f32) -> Annotation` (merge mode), `homography(points: [Pt; 4], &RgbaImage) -> RgbaImage` (saved as `*_HOMO`).

- [ ] **Step 1:** Tests per function from DarkMark's `DMContent.cpp` behaviour with citations: `copy_skips_same_class_same_corners`; `interpolate_midpoint_halfway`; `scale_one_percent_keeps_center`; `snap_moves_edge_to_strongest_gradient`; `homography_identity_is_noop`.
- [ ] **Step 2:** Run FAIL, implement, PASS. Commit.

### Task 3.3: View toggles and screenshot

**Files:** Create `views.rs` in `darknet-gui`.

- [ ] **Step 1:** Implement B&W, Canny (`imageproc::edges::canny`), dot, bold, shade, label modes, zoom-review, screenshot save. Display-only: never write to the image on disk.
- [ ] **Step 2:** Unit-test the pure image transforms (`bw_is_luma`, `canny_marks_a_step_edge`); manual smoke for the rest. Commit.

## M4 — Launcher and project creation

### Task 4.1: `create_project`

**Files:** Create `create.rs`; test `tests/create.rs`.

**Produces:** `create_project(dir: &Path, name: &str, class_names: &[String]) -> Result<Project>` writing `<name>.names` and `<name>.data` with DarkMark's layout (G1); refuses an existing project.

- [ ] **Step 1:** Tests: `creates_files_darkmark_can_load` (loads with `Project::load`); `refuses_existing_project`; `rejects_empty_or_duplicate_class_names`.
- [ ] **Step 2:** Run FAIL, implement, PASS. Commit.

### Task 4.2: Launcher UI

**Files:** Create `launcher.rs`, `settings.rs`.

- [ ] **Step 1:** New (folder picker, class list), open, recent list (persisted in the GUI settings file), delete (confirmation, deletes only DarkMark sidecar files, never images).
- [ ] **Step 2:** Test `settings` serde round trip and `recent_list_caps_at_20`; manual smoke. Commit.

## M5 — Class editing

### Task 5.1: Remap plan and atomic apply

**Files:** Create `classes.rs`; test `tests/classes.rs`.

**Produces:** `enum ClassEdit { Rename(usize, String), Add(String), Delete(usize), Move { from: usize, to: usize } }`, `plan_remap(&Project, &[ClassEdit]) -> RemapPlan` (per-class counts before/after, files touched), `apply_remap(&Project, &RemapPlan, backup_dir: &Path) -> Result<()>`.

- [ ] **Step 1:** Tests: `plan_does_not_touch_disk`; `apply_copies_originals_to_backup_first`; `failure_on_file_n_restores_all` (inject a write error on the Nth file; every file and `.names` byte-identical after); `delete_class_removes_its_marks_and_shifts_higher_ids`; `move_class_remaps_json_and_txt_consistently`.
- [ ] **Step 2:** Run FAIL, implement: validate everything in memory, write temp files, rename all, `.names` last. PASS. Commit.

### Task 5.2: Class editor UI

- [ ] **Step 1:** Dialog: edit list, show the dry-run table (counts per class), Apply button; result shown with the backup path. Manual smoke on a copy of a real project. Commit.

## M6 — Inference overlay

### Task 6.1: `detect-server` in the CLI

**Files:** Modify `src/darknet-cli/src/main.rs` (new arm); test in `src/darknet-cli/tests/`.

**Produces:** `darknet-rs detect-server <cfg> <weights>`: reads `{"image": "<path>", "thresh": f32}` lines on stdin, writes `{"boxes":[{"class":usize,"conf":f32,"cx":f32,"cy":f32,"w":f32,"h":f32}]}` per request, honours `DARKNET_BACKEND=cpu|oxide`; decodes modern heads.

- [ ] **Step 1:** Test: `detect_server_answers_one_request_per_line` with a tiny cfg on the CPU backend; `malformed_line_returns_error_object_and_continues`.
- [ ] **Step 2:** Run FAIL, implement, PASS (`MEM=3G ./tools/test_host.sh detect_server`). Commit.

### Task 6.2: Engine worker

**Files:** Create `detect_worker.rs`; test `tests/fake_server.rs`.

**Produces:** `DetectWorker::spawn(bin: &Path, cfg, weights) -> Result<DetectWorker>`, `detect(&mut self, image: &Path, thresh: f32) -> Result<Vec<Prediction>>` with a 30 s timeout.

- [ ] **Step 1:** Tests against a fake server script: `returns_predictions`; `child_exit_is_error_not_panic`; `timeout_kills_child`; `malformed_json_is_error`.
- [ ] **Step 2:** Run FAIL, implement, PASS. Commit.

### Task 6.3: Overlay and prediction hotkeys

- [ ] **Step 1:** Wire `p` (toggle), Up/Down (threshold ±0.05 in [0.05, 0.95], G15), `a`/`A` (G14), `o` (mass-delete predictions), `h`/`H` (per-class heatmaps from prediction confidences). Unit-test `accept_all_only_when_no_accepted_marks` and `threshold_clamps` in `edit.rs` tests.
- [ ] **Step 2:** Manual smoke on a project with weights; failures show an error banner and editing keeps working. Commit.

## Out of scope

OBB, segmentation and cuboid modes; PDF and video import; bulk rotate/flip/reload-resave/duplicate finder; parquet; Darknet cfg/anchor/tiling generation; training launch and monitoring (v2); SQLite index; screenshot automation.

## Self-review

- Coverage: every Scope item maps to a task (hotkeys T3.1–3.3, prediction keys T6.3, undo T1.4, create T4.1, class edit T5.1–5.2).
- Types: `Editor`, `Annotation`, `UndoStack`, `Project`, `Filter`, `ClassEdit` are each defined once and referenced by name.
- Known weakness: `HotkeyWnd.cpp` was not read while planning; T3.1 Step 1 extracts the key list and must stop and report if it finds keys outside the stated exclusions. Behaviours in T3.2 are described by name; the implementer must read `DMContent.cpp` for exact semantics and cite lines in the tests.
- Proportion: signatures, test names and assertions only; DarkMark behaviour is by reference to `Gn`/`Dn`.

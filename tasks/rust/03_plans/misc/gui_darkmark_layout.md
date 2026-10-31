# lightnet-gui: DarkMark layout plan

Reference: [darkmark_gui_behavior.md](../../04_reference/darkmark_gui_behavior.md). Goal: the editor window looks and behaves like DarkMark's editor. Today `src/lightnet-gui/src/main.rs` shows an egui top toolbar, a left panel (filter, classes, image list) and a bottom model bar around the canvas. DarkMark shows only the canvas plus a thin image-navigation strip. Decision (user, 2026-10-31): full DarkMark layout.

## Table of Contents

- [DarkMark layout reference](#darkmark-layout-reference)
- [Project tree](#project-tree)
- [Component breakdown](#component-breakdown)
- [Phase 1: small independent fixes](#phase-1-small-independent-fixes)
- [Implementation plan](#implementation-plan)
- [Open questions](#open-questions)

## DarkMark layout reference

- `DMContent::resized` (`DarkMark/src-darkmark/DMContent.cpp:413-474`): the canvas fills the window. The scrollfield is a vertical strip on the right edge, `scrollfield_width` px wide (0 disables it), with a 2 px spacer. The image is fitted to `window_width - spacer - scrollfield_width`.
- `ScrollField` (`src-wnd/ScrollField.cpp`): a vertical strip on the right edge, one row per image (top is the first image), resampled to the window height. The field is 200 px wide and is split into one column per class (`line_width = 200 / number_of_classes`, `:92-93`). A mark draws a horizontal segment in its class column, in the class colour (`annotation_colours[class_idx % size]`, `:212-219`). Row base colours: black = has a `.json` (marks drawn over it); grey (32,32,32) = no `.json` (`:180-184`); red = JSON parse error, or a `.json` with zero marks that is not `completely_empty` (`:202-206`, `:233-240`). `completely_empty` images get a fake mark in the `* empty image *` class colour. A full-width white line marks the current image (`draw_marker_at_current_image`, `:435-446`, drawn on a clone). White triangles on the left edge mark the first image of each directory, only when the sort is alphabetical; hovering a triangle shows the directory name (`:108-140`, `:284-321`). Width 0 disables the strip.
- Scrollfield interactions: left-click and drag jump with a quick load (no marks), mouse-up does the full load (`:247-281`); mouse-up on a triangle jumps to that directory's first image (`:332-345`); right-click opens the popup menu. The wheel steps one image with a quick load and a 500 ms full-load timer (`CrosshairComponent.cpp:310-355`). Row index is `round(y / (height / N))`. After a save, one row is rewritten (`DMContent.cpp:3024-3027`).
- Scrollfield build and size: built on a worker thread from each image's `.json` on disk (`:96-106`), gated on `!images_are_loading` and non-zero width. Cfg `scrollfield_width` defaults to 100 (`Cfg.cpp:74`; settings slider 0-200 step 10); `scrollfield_marker_size` (triangle size) defaults to 7. The canvas shrinks by the width plus a 2 px spacer (`DMContent.cpp:448-474`).
- Status information is drawn on the canvas, not in panels. Classes, sort, filter and view options are in the right-click popup menu. Other windows (filter, jump, stats, settings) open separately.
- Marks are drawn on the image with a class colour. Labels are `on`, `off` or `auto`. A crosshair follows the mouse: a thin red vertical and horizontal line through the cursor, spanning the whole canvas, shown only while the mouse is over the image (`src-tools/CrosshairComponent.cpp`; the user supplied a reference screenshot on 2026-10-31).

## Project tree

```
src/
├── lightnet-gui/
│   └── src/
│       ├── main.rs            editor shell: update(), canvas(), top_bar(), side_panel(), modals()  (edit)
│       ├── scrollfield.rs     NEW: right-edge strip widget and its colour cache
│       ├── overlay.rs         NEW: on-canvas status text and crosshair
│       └── popup.rs           NEW: right-click menu (classes, sort, filter, view, actions)
├── lightnet-gui-project/
│   └── src/
│       ├── project.rs         mark_count(), image list (read only here; supplies row colours)
│       └── filter.rs          filter state reused by the popup and the filter window
└── lightnet-gui-engine/       detect-server client (unchanged; model bar moves into a window)
```

## Component breakdown

- **Editor shell (`main.rs`)**
  - Role: owns `App` state and the egui frame.
  - Changes: `update()` drops `top_bar`, `side_panel` and the bottom model panel. It lays out `CentralPanel` (canvas) plus a right `SidePanel` of fixed width for the scrollfield.
  - Depends on: `scrollfield`, `overlay`, `popup`, `lightnet-gui-project`.
  - Constraint: every control that disappears from a panel must stay reachable by hotkey or the popup menu. No feature may be lost.
- **Scrollfield (`scrollfield.rs`)**
  - Role: DarkMark's right-edge strip, ported faithfully. One row per image, one column per class (width 100 px by default, configurable, 0 disables). Row colours: black base with class-coloured segments per mark, grey for no sidecar, red for a corrupt sidecar or an empty sidecar not flagged `completely_empty`, class colour of `* empty image *` for flagged-empty images. White full-width line for the current image. Directory-start triangles with a hover tooltip when sorted alphabetically.
  - Interaction: click or drag jumps with a quick load, release does the full load; clicking a triangle jumps to the first image of that directory; right-click opens the popup. Mouse wheel on the canvas steps one image with a 500 ms full-load timer.
  - Depends on: image count, the per-image mark classes (new `Project` accessor, since `mark_count` alone gives no class), current index, class colours (`class_color`).
  - Constraint: build off the UI thread, once per project load, and rewrite one row after a save. With thousands of images this must not block egui. The screenshot in the user's message shows the strip with green clusters in a middle column and red/grey lines in a left column, which matches class columns plus the red and grey row states.
- **Overlay (`overlay.rs`)**
  - Role: draws filename, `index/total`, zoom, mark and prediction counts, dirty flag, and the crosshair on the canvas. Crosshair: 1 px red lines (class colour of `most_recent_class`, as DarkMark does) through the pointer, full width and height of the canvas, drawn only while `response.hovered()` and not during a pan; hide the system cursor over the image so the cross is the pointer. Must not repaint the whole image: request a repaint on pointer move only.
  - Depends on: `App` fields only.
- **Popup (`popup.rs`)**
  - Role: right-click menu replacing the left panel: class list (click sets class, as digits do), sort, filter (opens the filter window), view (normal, B&W, Canny), labels, predictions, class editor, load model, projects.
  - Depends on: `Action` and `run_action` in `main.rs`, so menu items and hotkeys share one path.
- **Windows (`modals()`)**
  - Role: filter, class editor, model (cfg and weights, Load model), help, new project. These become floating `egui::Window`s opened from the popup or a hotkey.
- **Launcher**
  - Role today: first window with project path field, Open, Open folder... and Recent. The legacy-sidecar fix (missing `annotation_mode` accepted) is already in `annotation.rs`.
  - DarkMark reference (user screenshot, "DarkMark v1.5.19-2 Launcher"): a tab per project across the top; a table for the selected project (name, project directory, size of directory and cache size, last opened, image files, markup files with percentage and negative samples, number of classes, number of marks with per-class and per-image averages, newest and oldest markup with relative age, exclusion regex, network dimensions, darknet template, configuration, weights, classes/names paths); a file list (filename, type, size, timestamp with relative age, newest rows highlighted green) for `.cfg`, `.weights` and `.names` with a "hide extra .weights files" checkbox; a thumbnail; and the buttons Add..., Delete..., Import video..., Open folder..., Refresh, Load..., Cancel.
  - Target: rebuild the launcher as project tabs plus the info table, file list and these buttons. Stats come from the sidecars (`Project` scan; cache them, a large project must not block the UI). Import video is out of scope for v1 and may be a disabled button.

## Phase 1: small independent fixes

Done first and committed one item at a time, because they are independent of the layout. Verification for each: `JOBS=4 MEM=4G cargo xtask check lightnet-gui`, `PKGS=lightnet-gui-project JOBS=4 MEM=4G cargo xtask test`, and a look at `cargo xtask gui <project-dir>` (the app takes an optional project directory so the editor can be opened without clicking through the launcher).

- [x] a. Hover crosshair: 1 px red vertical and horizontal lines through the pointer over the whole canvas, only while the pointer is over the image, hidden while a drag or a pan is active; the system cursor is hidden over the image (`canvas()` in `main.rs`). DarkMark tints the crosshair with the most-recent-class colour instead of red; step 3 of the layout plan can revisit that.
- [x] b. Hotkey parity for `p`, `m`, `a`, `A` (see [Hotkeys p, m, a, A](../../04_reference/darkmark_gui_behavior.md#hotkeys-p-m-a-a)). `p` is Off -> Auto -> Off with `On` only in the model-bar menu; both toggles persist in `~/.config/lightnet-gui/settings.json`; `p` re-runs the detector when a model is loaded; Auto honours `m`; the crosshair rule lives in `edit::predictions_are_visible`; `a`/`A` report DarkMark's refusal reasons. Verified live on an Xvfb display (XTest cannot give the app keyboard focus under WSLg XWayland): `p` off/auto/off with messages and persisted state, `m` hidden/visible with messages, `a` and `A` refusal messages, `A` with no selection falling back to accept-all. Not verified live: the re-detect on `p` (no weights loaded) and the Auto hide rule (needs predictions).
- [x] c. Save pipeline fixes (see [Save pipeline](../../04_reference/darkmark_gui_behavior.md#save-pipeline)): temp file plus rename, `f64` coordinates, non-finite marks refused, `description` and unknown fields kept, real `image.scale`. Also fixed on the way: DarkMark's `points`/`rect` are normalized, not pixels, so the port was collapsing every loaded mark into the top-left corner and would have written new marks in pixel space.

## Implementation plan

Verification for every step: `JOBS=4 MEM=4G cargo xtask check lightnet-gui`. Launch with `cargo xtask gui` and look at the window (use `WAYLAND_DISPLAY=` on WSLg if the Wayland connection resets).

- [x] 1. Extract `Action` dispatch so menu and hotkeys share `run_action` (already true for keys; confirm each panel button maps to an `Action` or add one). The panels are gone; the popup, the windows and the keys all call `run_action`, except the pure display setters (`view`, `label_mode`, `sort` by mark count) and the window flags, which are single-path.
- [x] 2. Add `scrollfield.rs` and render it as a right `SidePanel` with fixed width. Keep the old panels behind a `--classic` flag only if needed during the transition; remove after step 6. No `--classic` flag was needed.
- [x] 3. Add `overlay.rs`: status text on the canvas and the full-canvas hover crosshair (see the overlay component), plus DarkMark's white bubble message (`show_message`, four seconds).
- [x] 4. Add `popup.rs`: right-click menu with classes, sort, filter, view, labels, predictions, projects, class editor, load model. Also accepts/saves/undoes and opens the windows; the canvas and the scrollfield share it.
- [x] 5. Move filter, class editor and model bar into floating windows opened from the popup and hotkeys (`e` settings, `f` filter, `j` jump, F2 help). The settings window carries the scrollfield width and marker size, the threshold and snapping.
- [x] 6. Remove `top_bar` and `side_panel`. Canvas fills the window. Verify fit-to-window zoom matches DarkMark (auto fit, Space toggles, `+`/`-` steps of 0.1). `zoom` is now `Option<f32>` (None = fit), the wheel leaves manual zoom instead of navigating, and the status overlay replaces the title-bar text.
- [x] 7. Quick-load and 500 ms full-load for the scrollfield and the mouse wheel, matching DarkMark's two-stage load. `go_to(.., LoadKind::Quick)` followed by `tick_pending_load`; a full load also re-runs the detector when predictions are not off.
- [ ] 8. Manual check against DarkMark side by side on `datasets/LegoGears_v2`: layout, colours, labels, popup contents. Partly done from a rendered window (canvas plus scrollfield, row colours, current-image line, directory triangles, status overlay, crosshair, popup contents, launcher info table and file list); a side-by-side comparison against a running DarkMark and the user's own look are still open.
- [x] 9. Commit with the feature tag required by `tasks/rust/AGENTS.md`, only after the manual check.

## Open questions

- Scrollfield width: DarkMark defaults to 100 px (`Cfg.cpp:74`, slider 0-200). Use 100 and keep it configurable. Resolved by reading DarkMark.
- Should the old left panel remain available as an optional toggle (`F3`) for users who prefer it?
- `predictions.jpg` and other non-dataset images in the project root appear in the image list (the first entry in the screenshot). DarkMark lists them too, so keep parity unless you want them excluded by the project regex.
- Prediction mode is re-run only when `p`, the menu or an image load changes it. The load hook is in `load_current(LoadKind::Full)`, matching `DMContent.cpp:2537`: navigating re-detects when a model is loaded and predictions are not off. Not verified live (no weights in the test project).
- The scrollfield follows the visible (filtered and sorted) list and rebuilds its rows on a worker thread when that list changes. DarkMark rebuilds after sort and filter apply; this port keeps the strip in sync every frame instead.
- `image.scale` is written from the canvas fit/zoom at save time. DarkMark records the same number, but this port does not keep DarkMark's per-project cache database, so the launcher stats are recomputed from the sidecars on demand.

## Not done in this pass

Kept here so nothing looks finished that is not:

- Launcher rows DarkMark shows that this port does not: directory size and cache size, last opened, exclusion regex, network dimensions from the cfg, darknet template. The thumbnail is missing too. The file list, the markup/negative/class/mark statistics and the buttons are there.
- Hotkeys that are still mapped-but-unimplemented: mass delete (`o`), homography (`P`), merge (backtick), heatmaps (`h`/`H`), screenshots (`s`/`S`/F12), tiling (`t`, `[`, `]`, `(`, `)`), delete/undo-delete image (Shift+Del, Shift+U). They report their name in the status bubble instead of doing something.
- DarkMark windows that have no equivalent yet: statistics, review, review-IoU, jump slider markers. Segmentor, OBB, cuboid and 3D projects remain unsupported by the sidecar layer.
- Sidecar format is still JSON only; YAML stays an open question.

# DarkMark GUI behavior and parity audit

Reference for the egui replacement in `lightnet-gui` / `lightnet-gui-project` (plan: [gui_plan.md](../03_plans/misc/gui_plan.md)). The source is the C++ DarkMark in `/home/kerrigan/src/DarkSuite/DarkMark`. The first three sections were produced by two read-only subagents on 2026-10-31; later sections compare them with the Rust crate, record the hotkey, save, scrollfield and launcher studies, and note the fixes made. Citations are `file:line` in the C++ tree unless noted. `DMC` is `src-darkmark/DMContent.cpp`, `DMCv` is `src-darkmark/DMCanvas.cpp`, `Xh` is `src-tools/CrosshairComponent.cpp`.

## Table of Contents

- [State space](#state-space)
  - [Structure](#structure)
  - [Persistent and session state](#persistent-and-session-state)
  - [Modes and transitions](#modes-and-transitions)
  - [Data on disk](#data-on-disk)
  - [Invariants and hazards](#invariants-and-hazards)
- [Interaction logic](#interaction-logic)
  - [Keys](#keys)
  - [Mouse](#mouse)
  - [Redraw, save and reload order](#redraw-save-and-reload-order)
  - [Algorithms to port exactly](#algorithms-to-port-exactly)
  - [Quirks the port must replicate](#quirks-the-port-must-replicate)
- [Rust parity audit](#rust-parity-audit)
  - [Matches](#matches)
  - [Deliberate deviations](#deliberate-deviations)
  - [Open findings](#open-findings)
  - [Not yet verified](#not-yet-verified)
- [Launcher folder picker](#launcher-folder-picker)
- [Hotkeys p, m, a, A](#hotkeys-p-m-a-a)
- [Save pipeline](#save-pipeline)
- [Legacy sidecars (fixed)](#legacy-sidecars-fixed)
- [ScrollField](#scrollfield)
- [Launcher](#launcher)
- [Searches that found nothing](#searches-that-found-nothing)
- [GUI environment notes](#gui-environment-notes)
- [Coverage gaps of the C++ study](#coverage-gaps-of-the-c-study)

## State space

### Structure

One `DMContent` (`DMContent.hpp:55`) owns the model (`marks`, the image list, the flags), the `DMCanvas` (input and tool state) and the `ScrollField`. The UI thread mutates nearly everything. There is no mode enum: state is the product of a fixed project mode and several orthogonal flags.

The project mode `annotation_mode_` is one of Detector, OBB, Segmentor, Cuboid, Detector3D. It is read once from the per-project Cfg key `<prefix>default_annotation_mode` and is immutable for the editor's lifetime, so changing it means reopening the project. Segmentor changes the digit-key mapping (0 turns on mask-delete, 1..k map to classes 0..k-1, `DMC:1103-1126`) and never writes `.txt`. In Detector3D, `g` toggles 2D box and 3D cuboid creation.

### Persistent and session state

Preferences live in the global Cfg (`~/.config/DarkMark/*.cfg`, a JUCE PropertiesFile) under a per-project key prefix. Editing state is not persisted.

Persisted preferences: `sort_order` (12 values), `show_labels` (Off/On/Auto), `show_predictions` (Off/On/Auto), `show_marks`, the display flags (`shade_rectangles`, `all_marks_are_bold`, `show_dots`, `alpha_blend_percentage`, `corner_size`), the B&W and Canny filter flags (mutually exclusive) and their parameters, `snapping_enabled` with its tolerances, the heatmap settings, and `scrollfield_width`.

Session state, not persisted:

- **Image list and position:** `image_filenames`, `image_filename_index`. A background thread scans the rest of the list after the first 100 images are found; the merge runs via `callAsync` and re-finds the index by filename.
- **Current image:** cached paths, the cv::Mat variants (`original_image`, `scaled_image`, `heatmap_image`, B&W, Canny), `scaled_image_size`, `current_zoom_factor`.
- **Annotations:** `marks` holds user marks and predictions mixed in one vector. Also `names`, `image_is_completely_empty` (negative-sample flag), `preserve_source_text_annotation`.
- **Selection and dirty tracking:** `selected_mark` (-1 or an index below `marks.size()`) and `need_to_save`.
- **Class memory:** `most_recent_class_idx`, `most_recent_size`, `crosshair_colour`. Only class and size are reused when a mark is created by double-click.
- **Derived counters** (`marks_are_shown`, `number_of_marks`, `number_of_predictions` and so on) are recomputed in `DMCanvas::rebuild_cache_image` (`DMCv:291-334`) and go stale until the next rebuild.
- **Zoom:** `user_specified_zoom_factor` (-1 is auto, otherwise 0.01 to 5.0), `previous_zoom_factor`, `zoom_point_of_interest`, `canvas.zoom_image_offset`.
- **Mode flags:** `show_obb_annotations`, `merge_mode_active`, `mass_delete_mode_active`, `mask_delete_mode_active`, `homography_mode_active`, `homography_points`, `zoom_review_marks_remaining`, `filter_use_this_subset_of_class_ids`.
- **Undo-delete:** `deleted_images_cache`, a deque capped near 10 entries and 200 MB, holding image bytes plus `.txt` and `.json` contents. Shift+U restores the newest.
- **Canvas:** `is_panning`, the three tools (`obb_tool`, `seg_tool`, `cuboid_tool`), the OBB edit sub-state (`obb_edit_mode`, grabbed corner, start snapshot), and the Crosshair mouse fields.

### Modes and transitions

- **Zoom** has two states, auto and manual. `+`, `-` and Space enter manual; Space and Esc return to auto. Space remembers the zoom in `previous_zoom_factor`. Home, End, `r`, `R`, Shift+Delete, Shift+U and `P` leave manual zoom first. Left, Right, PgUp and PgDn do not, so manual zoom persists across navigation. `z` (zoom review) forces manual zoom on its first press.
- **Predictions** cycle Off and Auto with `p` (On exists but is skipped). Auto shows predictions only when no real mark is visible (`DMCv:316-318`), and hides them while a drag rectangle is active (`DMCv:324-329`).
- **Exclusive modes** are merge (backtick), mass-delete (`o`), homography (`P`) and mask-delete. Esc cancels the first active one in this order: OBB in progress, predicted mask, manual zoom, merge, mass-delete, homography, mask-delete. With none active, Esc closes the window (`DMC:1297-1353`).
- **Visibility** follows `mark_is_visible_in_mode` (`src-tools/Mark.cpp:146`): the mark must fit the project mode, OBB marks follow `m`, predictions follow the prediction toggle independently.
- **Navigation:** Scrollfield dragging and the Jump slider use a quick load (`full_load=false`, no JSON or predictions); the full load follows on mouse-up or after a 500 ms timer.

### Data on disk

- `<img>.json` is the native format: `annotation_mode`, `mark[]`, `image{scale,width,height}`, `timestamp`, `version`, `completely_empty`, and a `predictions` block only after IoU data was written. Each mark has `class_idx`, `name`, `description`, `is_prediction` (always false when saved), `annotation_kind` and geometry (`points` plus `rect` for detectors; `xywh_front` or `quad` for OBB; `point_state` for cuboids). `int_x`/`int_y` are written for third-party readers and ignored on load.
- `<img>.txt` is the Darknet label file. An empty `.txt` means a negative sample. It is deleted when the image has no real marks and is not flagged empty (`DMC:2833`) and never written in Segmentor mode.
- Segmentation uses `<image>_mask.png` (0 is background, k+1 is class k) plus a manifest and `classes.txt`.
- Project files in the project directory: `.names` (with a trailing `* empty image *` entry), `.data`, `.cfg`, train and valid lists. `.darkmark_index.sqlite3` caches per-image stats for the launcher and is not a source of truth.
- **Load precedence** (`load_image`, `DMC:2380`): `.json` first, then `.txt`. If exactly one of the two exists, `need_to_save` is set so the pair is rebuilt. A malformed `.txt` clears the marks and sets `preserve_source_text_annotation` so the file is not overwritten. An unknown class index aborts the load with a message.

### Invariants and hazards

- `selected_mark` is reset to -1 in `load_image` (`DMC:2391`) and after Delete. Marks are re-sorted on every full load by `(round(15*y), round(15*x))` of the midpoint, so indices differ between loads and sessions.
- **Saving is lazy.** `need_to_save` is set on every edit and cleared only by a successful `save_json`. Saves run at `load_image` entry (`DMC:2384`), on `f`, on Ctrl+Shift+S, on exit and at review, statistics and IoU entry points. There is no autosave and no undo stack except image deletion. `save_json` and `save_text` are always called as a pair and re-set `need_to_save` on failure.
- **Corner click deletes.** A corner grab erases the mark at mouse-down; it is recreated only if the drag exceeds 2 px in both directions (`Xh:266`). A bare click on a corner deletes the mark.
- **Empty list.** `load_image` clamps the index to `size()-1` and uses `.at()`. With an empty list this underflows or throws. Deleting the last image or an empty filter result can reach it. FilterWnd forbids Apply on an empty result, but `delete_current_image` does not. Home, End and PgDn use `size()-1` unchecked.
- **Filter Apply** writes `image_filenames` without the mutex, then calls `load_image(0)` and re-sorts alphabetically, which discards the user's sort order.
- **Hard delete.** `C` (`erase_all_marks`) deletes both sidecars immediately with no confirmation or undo (`DMC:3997`).
- **Threading.** The scan thread, the Scrollfield worker, FilterWnd's worker and the filename sorter run beside the UI thread. Only the scan-merge path takes `image_filenames_mutex`.
- The Jump slider range and markers are fixed at construction and go stale if the list changes. `ScrollField::update_index` silently returns when the row count does not match the list.

## Interaction logic

### Keys

`DMContent::keyPressed` (`DMC:1013-2078`) is an if/else-if chain: the first match wins. An unmatched key shows "ignoring unknown key" and returns false (`DMC:2074`). `o`, `e` and `f` return false and fall out of the chain.

| Key | Effect |
|---|---|
| Tab / Shift+Tab | Cycle `selected_mark`, wrapping, skipping invisible marks (`:1041`). Sets `most_recent_class_idx`, `most_recent_size`, crosshair colour. |
| 0-9 | Reclassify the selected mark. Ctrl adds 10, Alt adds 20. Guard `class_idx >= names.size()-1` because the last name is `* empty image *`. Always updates `most_recent_class_idx` (`:1103`, `set_class` `:2081`). |
| Home / End | Leave zoom, load first/last image (`:1138`). |
| Left / Right | Previous/next image, no wrap, zoom stays (`:1158`). |
| PgUp / PgDn | Previous/next image whose `.json` has 0 marks; if none, ends on the first/last image (`:1184`). |
| Up / Down | Prediction threshold plus/minus 0.05, clamped 0.05-0.95, then a full `load_image` (`:1233`). |
| Shift+Del / Shift+U | Delete / undo-delete the image, leaving zoom first (`:1251`, `:1261`). |
| Del / Backspace | Seg tool first; otherwise erase `selected_mark`, clear the selection, set `need_to_save` (`:1271`). |
| Esc | Cancel the first active mode (see above), else close (`:1297`). |
| F1 / F2 | About / hotkey window. |
| Ctrl+C / Ctrl+V | Copy all non-prediction marks (or the selected one) as JSON to the clipboard; paste validates, sets `is_prediction=false`, appends (`:1373`, `:1407`). |
| Ctrl+Shift+S | `save_json` plus `save_text` (`:1461`). |
| Ctrl+W/A/S/D | Nudge every non-prediction mark by 1 image pixel (`nudge_all_marks`, `:736`). |
| `-` / `+` | Zoom steps of 0.1; Shift+`+` jumps to 5.0 (`:1475`, `:1496`). |
| Space | Toggle zoom against auto-fit, recentre on the mouse (`:1525`). |
| `r` / `R` | Random / alphabetical sort (`:1552`, `:1563`). |
| `a` / `A` | Accept all predictions, only when no accepted mark exists / accept the selected one, or all if none selected (`:1574`, `:3893`, `:3936`). |
| `,` / `.` | Shrink/grow every non-prediction mark by 1% of image size; the delta is added to normalized width and height, not a ratio (`:1606`, `Mark.cpp:180`). Shift rotates a selected OBB 1 degree, Ctrl 5. |
| `p` | Predictions Off/Auto (`:1663`). |
| `m` | Toggle marks visible (`:1713`). |
| `o` | Mass-delete mode (`:1738`). |
| `P` | Homography mode (`:1752`). |
| Backtick | Merge/interpolation mode (`:1765`). |
| `l`, `b`, `B` | Labels on/off/auto, bold, shade. |
| `C` | Erase all marks and delete both sidecars immediately (`:1800`, `:3997`). |
| `c` / Enter | Class popup menu. |
| `d` / `D` | Snap selected mark (or all real marks) / toggle auto-snap, persisted (`:1810`, `:1856`). |
| `h` / `H` | Cycle / toggle heatmap. |
| `j` | Jump window. |
| `n` | Mark the image negative. Refused when `number_of_marks != 0` with the message "delete annotations before marking the image as empty" (`:1905`). |
| `s` / `S`, F12 | Screenshot, with markup / window screenshot. |
| `t`, `[` `]`, `(` `)` | Tiling toggle, tile edge factor 0.01-1 step 0.05, tile rect factor 1-2 step 0.05. |
| `w`, `q` | Black-and-white, Canny. |
| `y` / `Y` | Copy marks from the previous / next annotated image in alphabetical order; duplicates (same class and corners) are skipped (`:2022`, `:3701-3890`). |
| `z` | Zoom review (`:4335`). |
| `e`, `f` | Settings; save if dirty, then filter window. |

### Mouse

`DMCanvas::mouseDown` (`DMCv:849`) checks, in order: right-click cancels an OBB or cuboid in progress; Ctrl+click starts panning and clears the selection; the seg tool; homography point; OBB edit/select (grab radius 8 px: front point rotates, nearest corner resizes, inside translates); OBB and cuboid creation; Detector selection (`:935-1029`).

- **Detector hit test:** among marks whose rect contains the point, a non-prediction, non-OBB mark whose nearest corner is within `corner_size` starts a corner drag. Otherwise the smallest-area visible containing mark is selected. A right-click also selects, then opens the popup menu.
- **Corner drag:** erase the mark, set `mouse_drag_rectangle` to start at the opposite corner, offset by the click, then rebuild through the normal create-by-drag path on release.
- **Drag:** panning is `zoom_image_offset -= delta`, clamped to `0..(image-canvas+5)`. Otherwise the rubber band grows from the down position.
- **Mouse-up** (`Xh:248`): normalize the rectangle; if width and height are both above 2 it calls `mouseDragFinished`.
- **`mouseDragFinished`** (`DMCv:1403`) order: panning end, OBB, cuboid, Segmentor, mass-delete, then Detector creation. The centre is the rect centre plus offset, divided by the cached image size. Snap runs when `snapping_enabled` XOR Shift is held.
- **Double-click** (`DMCv:1084`): seg tool, OBB, cuboid, merge key frame; otherwise create a mark of `most_recent_size` and `most_recent_class_idx` centred on the click, select it and snap if enabled. Shift is not checked here.
- **Wheel** (`Xh:310`): in zoom it only exits zoom. Otherwise it loads index plus or minus 1 with `full_load=false` and starts a 500 ms timer for the full load.
- There is no whole-mark drag in Detector mode; moving marks is Ctrl+WASD nudge only.

### Redraw, save and reload order

`rebuild_image_and_repaint()` (`DMC:683`) sets `need_to_rebuild_cache_image`, repaints and redraws the Scrollfield marker. The crosshair repaints narrow strips on mouse move. `load_image` (`DMC:2380`) saves the previous image if dirty, clears marks, mask, selection and zoom-review state, reads the image, loads `.json` (falling back to `.txt`), runs predictions when enabled, generates heatmaps, then sorts the marks. On error it shows a red 32x32 image. Every prediction-related toggle (`p`, threshold, tiling, heatmap) must call `load_image`, not just repaint. `save_json` and `save_text` refuse and keep `need_to_save=true` if validation fails. Predictions are never serialized.

### Algorithms to port exactly

- **Snap** (`snap_annotation`, `DMC:4825`): grayscale, adaptive Gaussian threshold, optional dilate/erode modes 0-4 (`:4959-5008`), invert. Then loop: grow the ROI by `min(attempt, tol_h/tol_v)` per side clamped to the image, take `boundingRect(findNonZero(roi))`. If the rect changed, reset `attempt=0`; if it grew past 3x the original area, abort and keep the original. Stop when the rect is unchanged and `attempt >= max(tol_h, tol_v)`. Apply only if changed and both sides are at least 10 px; otherwise report "not snapped".
- **Interpolation** (`interpolateMarks`, `DMC:5039`): linear in centre and size with `t=(i-start)/(end-start)`, forwards or backwards. Needs the same class at both ends and at least one frame between, counted by index in the current sort order. Each frame is fully loaded, the mark is appended (nothing is replaced), `need_to_save` is set, and the save happens on the next `load_image`. Only the selected mark of each key frame is used. Coordinates truncate to integer pixels.
- **IoU review** (`DMContentReviewIoU.cpp:60`): per image with a `.json`, predict. For each annotation, consider unconsumed predictions with a probability entry for that class, compute IoU (exact polygon IoU for OBB), take the maximum above 0 and consume it, else count the annotation as unmatched. Average IoU is the sum divided by `max(#annotations, #predictions)`. An empty image scores 1.0. Differences are unmatched predictions plus unmatched annotations. Results are written into the `.json` `predictions` block. Segmentation and cuboid projects are skipped.
- **Mass-delete** (`DMC:5197-5311`): convert the drag rect to normalized coordinates with the zoom offset, ask for a class (popup index is class+1), then ask for N extra frames. Predictions are forced off. From the current image forward N+1 frames, delete marks of that class fully inside the rect, then reload the start frame. Cancelling aborts with no change.
- **Homography** (`apply_homography_from_points`, `DMC:776`; `src-tools/Tools.cpp:215-330`): order the four points by angle around the centroid, rotated so the minimum x+y point is first. The destination is the quad's bounding-box size, capped at 4x the image. Validation: points more than 5 px apart, no three nearly collinear (`|cross| > 0.001*bbox area`), convex. `warpPerspective`, then write `<stem>_HOMO[N].<ext>` (PNG compression 9, JPEG quality 75). No annotation sidecar is written. The image list is refreshed and the mode state is always reset.
- **Cuboid** (`src-cuboid/CuboidTool.cpp`): eight sequential left-clicks, 1-4 base corners and 5-8 the tops (i+4 above i), each rejected outside 0-1. Right-click cancels a sequence in progress. Visibility is toggled later with `v` / `V`.
- **Zoom review** (`DMC:4335`): marks sorted largest first; each press centres on the next remaining mark and drops marks fully inside the view. A class filter applies. It advances to the next image whose `.txt` is larger than 10 bytes.
- **Batch tools:** rotate marks by sin/cos about (0.5, 0.5) and swap w/h unless 180 degrees (`DMContentRotateImages.cpp:410-440`); resize TL/TR forces 45x45 px and keeps the right edge for TR; move-empty moves image and sidecars whose `.txt` is 0 bytes into `<project>/empty_images`; delete-rotated/flipped removes files whose stem ends in `_r090`, `_r180`, `_r270`, `_fh` or `_fv` to the trash.

### Quirks the port must replicate

- The last class name is `* empty image *`, so valid classes are `0..names-2`.
- Predictions can be selected and Tab-cycled but are never saved. Only `a` and `A` promote them.
- Hit tests use scaled-image coordinates plus `zoom_image_offset`; normalization divides by `scaled_image_size`, but `mouseDragFinished` divides by the cached image size, and the two can differ when zoomed.
- Wheel and Left/Right save a dirty image and discard the selection. The wheel's marks and predictions appear 500 ms later.
- Interpolation follows the current sort order; `y` / `Y` always use alphabetical order.
- Shift on double-click is ignored, while Shift on drag-create inverts snap.
- The corner grab radius is `corner_size` from config. The OBB grab radius is a hard-coded 8 px.
- `n` counts `number_of_marks` from the loaded file.

## Rust parity audit

Compared against `src/lightnet-gui-project/src/{edit,nav,undo,hotkeys}.rs`.

### Matches

- `nav.rs`: Left/Right/Home/End clamp; PgUp/PgDn return the boundary image even if annotated; an empty list returns 0, so the C++ empty-list underflow does not occur.
- `edit.rs`: smallest-area hit test; corner grab erases and anchors at the opposite corner; digit class selection with Ctrl +10 and Alt +20; `,` / `.` apply a fraction of the image (`edit.rs:392`); `a` only when no real mark exists; `A` only the selected prediction; double-click ignores Shift; threshold clamped to `[0.05, 0.95]`.

### Deliberate deviations

- A per-image undo/redo stack capped at 100 (`undo.rs`). DarkMark has none apart from image-delete undo.
- A corner re-drag that collapses restores the original mark (`edit.rs:228`, rule `D8`). DarkMark leaves the mark deleted.

### Open findings

1. **`accept_all` ignores the threshold** (`edit.rs:412`; DarkMark also has no threshold at accept time, its stored predictions already passed DarkHelp's, so the real gap is that a later threshold change does not re-filter stored predictions). It promotes every prediction, including ones below the confidence threshold, instead of using `visible_predictions()`. Raising the threshold hides predictions without removing them, so hidden ones can be accepted. Fix: filter by `confidence >= threshold`.
2. **`set_class` does not update the remembered class** when a mark is selected. DarkMark always updates `most_recent_class_idx`, so the next drag-created or double-clicked mark gets the new class. In Rust it keeps the old one.
3. **Class guard bound.** Rust checks `class_id >= class_count`; DarkMark checks `names.size()-1` because of the trailing `* empty image *` name. Verify that `class_count` excludes that entry.
4. **`nudge` and `scale` push an undo entry even when nothing changes.** `nudge` also skips marks that would leave the image; check this against DarkMark's `nudge_all_marks` (`DMC:736`).
5. **Dead code with a stale comment.** `delete_predictions` (`edit.rs:446`) is documented as `o`, but `o` maps to `MassDelete` in `hotkeys.rs`, and the function is unused in `lightnet-gui`.
6. **`set_empty_flag` refuses silently** when marks exist. DarkMark shows "delete annotations before marking the image as empty".

### Not yet verified

- The save policy: lazy save on image change, `.txt` deletion when empty, single-sidecar repair.
- The two-stage wheel load with its 500 ms timer.
- The mark re-sort on load.
- `actions.rs` against snap, interpolation, homography and mass-delete above.
- A confirmation for the `C` hard delete.

## Launcher folder picker

Added 2026-10-31 by a Haiku subagent. `src/lightnet-gui/src/main.rs:1006` adds an "Open folder..." button to the launcher. It calls `rfd::FileDialog::new().pick_folder()`, fills `new_dir` and calls the existing `open_project()`. Cancel does nothing. The dependency is `rfd = "=0.17.2"` with only the `xdg-portal` feature (no GTK). `cargo xtask check lightnet-gui` passes; the dialog itself was not tested. On WSL2 the dialog needs an `xdg-desktop-portal` service; without one `pick_folder` returns `None` and the button does nothing. The typed path field and the Open button still work. Fallback: enable the `gtk3` feature or install a portal. The change is uncommitted.

## Hotkeys p, m, a, A

Compared on 2026-10-31 by a Haiku subagent. Cites are DarkMark (`DMContent.cpp` unless noted) versus Rust (`lightnet-gui/src/main.rs`, `lightnet-gui-project/src/{edit,hotkeys}.rs`).

DarkMark:

- `EToggle` is `kOff=0, kOn=1, kAuto=2` (`src-tools/Cfg.hpp:12-18`). `p` (`:1663-1712`) cycles `(v+1)%3` but skips `kOn`, so Off, Auto, Off. `kOn` is reachable only from the View menu (`:4128-4130`). `p` calls `toggle_show_predictions` (`:2239`), which saves Cfg `show_predictions` and calls `load_image`, so every press re-runs Darknet when the result is not Off (`:2537`). Guards: "neural network not loaded" (`:1685`) and OBB messages. Message: "predictions: on|off|auto".
- Visibility is decided in `DMCanvas::rebuild_cache_image` (`DMCanvas.cpp:291-330`): `marks_are_shown` comes from `show_marks`; `predictions_are_shown = show_predictions != kOff`. Auto hides predictions only when a real mark is actually shown (so `m` off lets them appear). Predictions are also hidden while a drag rectangle is active.
- `m` (`:1713`) flips `show_marks`, saves Cfg `show_marks`, repaints. Message "user marks: visible|hidden". In OBB mode it toggles OBB annotations instead.
- `a` (`:1574`) calls `accept_all_marks` (`:3922-3961`): does nothing if any real mark exists; validates the batch and shows "Cannot accept predictions: ..." on failure. There is no confidence threshold at accept time.
- `A` (`:1590`) calls `accept_current_mark` (`:3893`): accepts a selected prediction; a selected real mark does nothing; with no selection it falls back to `accept_all_marks`.

Rust and differences:

1. `p` cycles `prediction_mode` off, on, auto, off (`main.rs:406`), so On is reachable by key. The doc comment at `hotkeys.rs:76` says "on -> auto -> off", which is wrong. Default is auto.
2. `p` never detects. Detection runs only from the Detect button (`main.rs:1261`); `detect()` returns early when the mode is off (`:579`). Turning predictions on with none loaded shows nothing.
3. Auto hides predictions whenever any real mark exists, even with `m` off (`main.rs:768` versus `:774`). DarkMark honours `m`.
4. Rust does not hide predictions during a drag.
5. `prediction_mode` and `show_marks` are not persisted; DarkMark writes both to Cfg.
6. Rust shows no message for `p`, `m` or a rejected `a`.
7. `A` with no selection is a no-op in Rust (`edit.rs:442`); DarkMark accepts all.
8. Both `a` implementations accept every prediction and no-op when a real mark exists. DarkMark's stored predictions already passed DarkHelp's threshold; in Rust the threshold only filters at detect time (`main.rs:588`), and raising it later does not remove stored predictions, so `a` can accept ones the user can no longer see (see finding 1 in the parity audit).

## Save pipeline

Compared by a Haiku subagent on 2026-10-31.

DarkMark:

- **Triggers:** destructor (`DMContent.cpp:159`, only if `need_to_save`), `load_image` entry (`:2384`), `f` (`:2041`), Ctrl+Shift+S (`:1461`), and the menus, statistics and review entry points.
- **`save_json`** (`:2909`): validates with `validate_annotation_batch`; on failure sets `need_to_save`, shows a message and writes nothing. Builds `annotation_mode`, `mark[]`, `rect` with `int_x/int_y/int_w/int_h` (`:2958`), `points`, `image{scale,width,height}`, `timestamp`, `version`, `completely_empty` (`:2973-2981`). Writes `dump(1,'\t') + "\n"` when there are marks or the image is flagged empty (`:2985`), else deletes the `.json` (`:3002`). Then upserts or removes the row in the SQLite index (`:3007-3025`).
- **`save_text`** (`:2833`): skipped in segmentation mode; deletes the `.txt` unless the image is flagged empty or has a real mark (`:2869-2901`). OBB lines have 6 fields, cuboid 25.
- **`write_validated_annotation_file`** (`AnnotationPersistence.cpp:209`): validates, then truncates and writes. No temp file and no rename: not atomic.
- **Batch resave:** `DMContentReloadResave.cpp` reloads every sidecar, sets `need_to_save`, and also converts `.txt`-only images to `.json`.
- **Delete image** (`:3510`): caches the image bytes and the two sidecar strings in RAM (limit 200 MB, at most 10), moves the files to the trash, overflow unlinks the oldest; Shift+U (`:3585`) rewrites from RAM. The destructor permanently deletes cached files (`:166`).
- **Screenshots and homography:** `s` saves `<stem>_annotated.png` to the Desktop via a chooser, `S` at full size (PNG level 9, JPEG q75, `:4635`); homography writes `<stem>_HOMO<ext>` beside the image with no sidecar.

Rust (`lightnet-gui-project/src/annotation.rs`, `project.rs`, `lightnet-gui/src/main.rs`):

- Detector only. `to_json` (`annotation.rs:258`) keeps unknown keys, sets `timestamp`, `version`, `image.scale = 1`, pretty-prints with two spaces then rewrites the indent to tabs. `Annotation::save` (`:359`) removes both files when there are no real marks and the image is not `completely_empty`, otherwise writes `.json` then `.txt` with `std::fs::write` (not atomic). `.txt` uses `{cx:.10}`.
- Saves on Ctrl+Shift+S, the Save button, navigation (`load_current`, `main.rs:290`) and exit (`:987`). There is no delete, undo-delete, screenshot, homography or index database.

Differences and risks to fix:

- **Not atomic** in either tool. Rust should write to a temp file and rename.
- **f32 noise:** marks store f32, so `0.1` is written as `0.10000000149011612` and every save rewrites all coordinates with noise. DarkMark uses double.
- **No finiteness check:** NaN is written as `null`; DarkMark refuses such marks.
- **`description` is lost:** accepting or merging a mark overwrites `description` with `name`.
- **`image.scale`** is always 1 in Rust; DarkMark writes the real scale.
- **Failed save before navigation** discards the edits (`load_current` ignores the failure and `dirty` is reset).
- **Fewer save points** than DarkMark, so a crash loses edits since the last navigation.
- **Whole-project rejection:** `validate_modes` rejects a project containing any non-detector sidecar; DarkMark supports them.
- Key order should match because both serializers sort keys, but verify with a byte diff on a real sidecar.

## Legacy sidecars (fixed)

The LegoGears_v2 sidecars (238 files) have no `annotation_mode` key. The launcher showed "missing annotation_mode" and refused to open the project. DarkMark accepts a missing key (`AnnotationJson.cpp:729`). `Annotation::from_json` (`lightnet-gui-project/src/annotation.rs`, around line 222) now treats a missing key as detector, with the test `missing_annotation_mode_is_legacy_detector`. `lightnet-gui-project` passes 40 of 40 tests. The next save writes `"detector"`. Uncommitted.

## ScrollField

Found by a Haiku subagent on 2026-10-31. Files are in `src-wnd/ScrollField.cpp`; the layout code is in `src-darkmark/DMContent.cpp`.

- **Rows and columns:** a vertical strip on the right edge, one row per image (top is the first image), resampled to the window height. The field is 200 px wide and split into one column per class (`line_width = 200 / number_of_classes`, `ScrollField.cpp:92-93`). A mark draws a horizontal segment in its class column in the class colour (`annotation_colours[class_idx % size]`, `:212-219`).
- **Row base colours:** black = has a `.json` (marks drawn over it); grey (32,32,32) = no `.json` (`:180-184`); red = JSON parse error, or a `.json` with zero marks that is not `completely_empty` (`:202-206`, `:233-240`). Images flagged `completely_empty` get a fake mark in the `* empty image *` class colour (`:195-199`, in memory only).
- **Overlays:** a full-width white line marks the current image (`draw_marker_at_current_image`, `:435-446`, drawn on a clone). White triangles on the left edge mark the first image of each directory, only when the sort is alphabetical (`:108-140`); hovering one shows the directory name (`:284-321`).
- **Build:** `run()` on a worker thread builds an N by 200 `cv::Mat`, one row per image, reading each `.json` from disk (`:96-106`). It is gated on `!images_are_loading` and a non-zero width. After a save one row is rewritten (`DMContent.cpp:3024-3027`).
- **Mouse:** left-click and drag jump with a quick load (`full_load=false`, `:257-281`); mouse-up does the full load (`:247-254`); mouse-up on a triangle jumps to that directory's first image (`:332-345`); right-click opens the popup menu. Row index is `round(y / (height / N))`. The wheel on the canvas steps one image with a quick load and a 500 ms full-load timer (`CrosshairComponent.cpp:310-355`).
- **Sizing and config:** Cfg `scrollfield_width` defaults to 100 (`src-tools/Cfg.cpp:74`; settings slider 0-200 step 10); `scrollfield_marker_size` (triangle size) defaults to 7. The canvas shrinks by the width plus a 2 px spacer (`DMContent.cpp:448-474`). Width 0 disables everything. Several batch tools temporarily set the width to 0 and rebuild afterwards.

## Launcher

DarkMark v1.5.19-2 launcher, from a user screenshot on 2026-10-31:

- A tab per project across the top.
- A table for the selected project: name, project directory, size of directory and cache size, last opened, image files, markup files (count, percentage, negative samples), number of classes, number of marks (per class and per image), newest and oldest markup with relative age, exclusion regex, network dimensions, darknet template, configuration, weights, classes/names paths.
- A file list (filename, type, size, timestamp with relative age; newest rows highlighted green) for `.cfg`, `.weights`, `.names`, with a "hide extra .weights files" checkbox and a thumbnail.
- Buttons: Add..., Delete..., Import video..., Open folder..., Refresh, Load..., Cancel.

The Rust launcher (`lightnet-gui/src/main.rs`, `launcher()`) has a project path field, Open, Open folder..., New project... and a Recent list.

## Searches that found nothing

On 2026-10-31 Haiku subagents searched for a crate named `obb_annotator` or `annotator`. None exists in `lightnet/src`, the workspace `Cargo.toml` or its `.claude/worktrees/cuda-split-plan` copy, in `/home/kerrigan/src/burn_dataset/burn_dataset`, or in `/home/kerrigan/src/kizauela` (17 crates). The nearest matches are generators, not annotation UIs: `ocr_syth_gen` in burn_dataset (rotated digits with Darknet labels, `image` and `imageproc` only), and `synthtiger-template` in kizauela (`obb_digit.rs`, `obb_arrow.rs`). All three repositories use the same OBB label line, `class cx cy w h front_x front_y`, normalized. `kizauela` GUI crates (`homography-viz`, `crane-game`) use bevy 0.19 with `bevy_egui` and annotate nothing.

## GUI environment notes

- `cargo xtask gui` (alias `g`) runs `cargo run --release -p lightnet-gui` under the 4-thread and memory cap; arguments go to the app. Added to `xtask/src/main.rs` and the `CLAUDE.md` command table.
- On WSLg the GUI can exit with `Io error: Connection reset by peer` and `WinitEventLoop(ExitFailure(1))` when the Wayland compositor connection resets. It could not be reproduced on 2026-10-31 (the same binary ran with and without `WAYLAND_DISPLAY=`). Workaround: `WAYLAND_DISPLAY= cargo xtask gui`, or `wsl --shutdown`.
- The folder picker needs an `xdg-desktop-portal` service (see the folder picker section).
- Open user decision: whether sidecars should be YAML instead of JSON. Replacing JSON breaks sharing a project with DarkMark, which only reads `.json`. Recommended: keep JSON, optionally read and write YAML as an extra format, or use YAML only for GUI settings. Nothing implemented.

## Coverage gaps of the C++ study

The agents did not read in depth: the OBB and Seg tool internals (only entry points), `DMReviewIoUWnd`, `DMReviewWnd`, `src-wnd/DMReviewCanvas.cpp` (only `cellDoubleClicked`), and the DarknetWnd and CfgHandler training-config code. Treat any claim touching those as unchecked.

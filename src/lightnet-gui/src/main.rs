//! Darknet annotation GUI (egui/eframe), laid out like DarkMark.
//!
//! The window is the canvas plus a right-edge scrollfield
//! (`gui_darkmark_layout.md`): no top bar, no left panel. Status is drawn on the
//! canvas, the right-click popup carries classes, sort, filter, view, labels and
//! predictions, and the filter, class editor, model bar, settings, jump and help
//! live in floating windows. All file semantics live in `lightnet-gui-project`;
//! all model inference lives in `lightnet-gui-engine`.
//!
//! `lightnet-gui [--project] <dir>` opens that project instead of the launcher.

mod colors;
mod overlay;
mod popup;
mod scrollfield;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use lightnet_gui_engine::{DetectWorker, Prediction};
use lightnet_gui_project::classes::{ClassEdit, apply_remap, plan_remap};
use lightnet_gui_project::create::create_project;
use lightnet_gui_project::filter::{Filter, SortKey, Tri};
use lightnet_gui_project::hotkeys::{self, Action, Key, KeyInput, map_key};
use lightnet_gui_project::{Editor, Mark, Mods, Point, Project, ProjectStats, actions, edit, nav};
use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use image::{GrayImage, RgbaImage};

use colors::{PREDICTION_COLOR, class_color};
use scrollfield::ScrollField;

/// DarkMark puts a 2 px spacer between the canvas and the scrollfield
/// (`DMContent.cpp:448-474`).
const SCROLLFIELD_SPACER: f32 = 2.0;
/// Two-stage load: a quick load, then a full load 500 ms later
/// (`CrosshairComponent.cpp:310-355`).
const FULL_LOAD_DELAY: Duration = Duration::from_millis(500);

fn main() -> eframe::Result<()> {
    // `lightnet-gui [--project] <dir>` opens that project instead of showing the
    // launcher; `cargo xtask gui <dir>` forwards its arguments here.
    let project = project_argument(std::env::args().skip(1));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Darknet Annotator"),
        ..Default::default()
    };
    eframe::run_native(
        "Darknet Annotator",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, project)))),
    )
}

fn project_argument(mut args: impl Iterator<Item = String>) -> Option<String> {
    let mut project = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--project" | "-p" => project = args.next(),
            other if !other.starts_with('-') => project = Some(other.to_owned()),
            _ => {}
        }
    }
    project
}

/// Load depth. DarkMark's `load_image(..., full_load)`: a quick load shows the
/// image without reading the sidecar (`CrosshairComponent.cpp:310-355`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoadKind {
    Quick,
    Full,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
enum View {
    #[default]
    Normal,
    BlackWhite,
    Canny,
    Dot,
    Bold,
    Shade,
}

impl View {
    fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::BlackWhite => "B&W",
            Self::Canny => "Canny",
            Self::Dot => "dot",
            Self::Bold => "bold",
            Self::Shade => "shade",
        }
    }
}

fn default_prediction_mode() -> u8 {
    2
}

fn default_show_marks() -> bool {
    true
}

fn default_label_mode() -> u8 {
    1
}

fn default_scrollfield_width() -> f32 {
    100.0
}

fn default_scrollfield_marker_size() -> f32 {
    7.0
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Settings {
    #[serde(default)]
    recent: Vec<String>,
    /// DarkMark's `show_predictions` (`EToggle`: 0 off, 1 on, 2 auto).
    #[serde(default = "default_prediction_mode")]
    prediction_mode: u8,
    /// DarkMark's `show_marks`.
    #[serde(default = "default_show_marks")]
    show_marks: bool,
    /// DarkMark's `show_labels`.
    #[serde(default = "default_label_mode")]
    label_mode: u8,
    /// `scrollfield_width` (`Cfg.cpp:74`, default 100, 0 disables).
    #[serde(default = "default_scrollfield_width")]
    scrollfield_width: f32,
    /// `scrollfield_marker_size`, the directory triangle size (`Cfg.cpp:75`).
    #[serde(default = "default_scrollfield_marker_size")]
    scrollfield_marker_size: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            recent: Vec::new(),
            prediction_mode: default_prediction_mode(),
            show_marks: default_show_marks(),
            label_mode: default_label_mode(),
            scrollfield_width: default_scrollfield_width(),
            scrollfield_marker_size: default_scrollfield_marker_size(),
        }
    }
}

impl Settings {
    fn path() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join(".config/lightnet-gui/settings.json"))
    }

    fn load() -> Self {
        Self::path()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }

    fn save(&self) {
        let Some(path) = Self::path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, text);
        }
    }

    fn remember(&mut self, root: &Path) {
        let text = root.to_string_lossy().into_owned();
        self.recent.retain(|entry| entry != &text);
        self.recent.insert(0, text);
        self.recent.truncate(20);
        self.save();
    }
}

struct Texture {
    handle: egui::TextureHandle,
    gray: GrayImage,
    rgba: RgbaImage,
    last_used: u64,
}

#[derive(Default)]
struct ImageCache {
    entries: HashMap<usize, Texture>,
    tick: u64,
}

impl ImageCache {
    const CAP: usize = 64;

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn get(&mut self, ctx: &egui::Context, index: usize, path: &Path) -> Result<(), String> {
        self.tick += 1;
        let tick = self.tick;
        if let Some(entry) = self.entries.get_mut(&index) {
            entry.last_used = tick;
            return Ok(());
        }
        let decoded = image::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let rgba = decoded.to_rgba8();
        let gray = image::imageops::colorops::grayscale(&rgba);
        let size = [rgba.width() as usize, rgba.height() as usize];
        let color = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
        let handle = ctx.load_texture(path.to_string_lossy(), color, egui::TextureOptions::LINEAR);
        if self.entries.len() >= Self::CAP
            && let Some(oldest) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(index, _)| *index)
        {
            self.entries.remove(&oldest);
        }
        self.entries.insert(
            index,
            Texture {
                handle,
                gray,
                rgba,
                last_used: tick,
            },
        );
        Ok(())
    }
}

struct LauncherFile {
    name: String,
    kind: &'static str,
    size: u64,
    modified: Option<SystemTime>,
}

struct App {
    settings: Settings,
    project: Option<Project>,
    filter: Filter,
    filter_text: String,
    filter_exclude: String,
    sort: SortKey,
    view: View,
    index: usize,
    editor: Option<Editor>,
    cache: ImageCache,
    /// `None` = fit to window (DarkMark's `user_specified_zoom_factor == -1`).
    zoom: Option<f32>,
    pan: Vec2,
    dirty: bool,
    status: String,
    status_at: Option<Instant>,
    error: bool,
    show_launcher: bool,
    show_new_project: bool,
    show_classes: bool,
    show_help: bool,
    show_filter: bool,
    show_model: bool,
    show_settings: bool,
    show_jump: bool,
    new_dir: String,
    new_names: String,
    class_edit: Vec<String>,
    class_plan: Option<String>,
    cfg_path: String,
    weights_path: String,
    worker: Option<DetectWorker>,
    clipboard: Option<arboard::Clipboard>,
    /// 0 = off, 1 = on, 2 = auto (`p` cycles Off and Auto only).
    prediction_mode: u8,
    /// 0 = off, 1 = on, 2 = auto (`l`).
    label_mode: u8,
    bold: bool,
    shade: bool,
    snapping: bool,
    random_order: Option<Vec<usize>>,
    show_marks: bool,
    threshold: f32,
    drag_start: Option<Pos2>,
    drag_now: Option<Pos2>,
    pan_start: Option<(Vec2, Pos2)>,
    /// Last canvas scale (fit or manual zoom), written to `image.scale` on save.
    canvas_scale: f64,
    /// Cached filtered texture for the non-normal views (`view` + image index).
    view_cache: Option<(String, egui::TextureHandle)>,
    scrollfield: ScrollField,
    /// Pending 500 ms full load after a quick load.
    pending_full: Option<Instant>,
    launcher_selected: Option<usize>,
    launcher_project: Option<Project>,
    launcher_stats: Option<ProjectStats>,
    launcher_files: Vec<LauncherFile>,
    launcher_hide_extra_weights: bool,
    launcher_stale: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, project: Option<String>) -> Self {
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        cc.egui_ctx.set_style(style);
        let settings = Settings::load();
        let prediction_mode = settings.prediction_mode.min(2);
        let label_mode = settings.label_mode.min(2);
        let show_marks = settings.show_marks;
        let scrollfield = ScrollField::new(
            settings.scrollfield_width,
            settings.scrollfield_marker_size,
        );
        let launcher_selected = if settings.recent.is_empty() { None } else { Some(0) };
        let mut app = Self {
            settings,
            project: None,
            filter: Filter::default(),
            filter_text: String::new(),
            filter_exclude: String::new(),
            sort: SortKey::Name,
            view: View::Normal,
            index: 0,
            editor: None,
            cache: ImageCache::default(),
            zoom: None,
            pan: Vec2::ZERO,
            dirty: false,
            status: "Open a project to start annotating".to_owned(),
            status_at: None,
            error: false,
            show_launcher: true,
            show_new_project: false,
            show_classes: false,
            show_help: false,
            show_filter: false,
            show_model: false,
            show_settings: false,
            show_jump: false,
            new_dir: String::new(),
            new_names: "person\ncar\n".to_owned(),
            class_edit: Vec::new(),
            class_plan: None,
            cfg_path: String::new(),
            weights_path: String::new(),
            worker: None,
            clipboard: arboard::Clipboard::new().ok(),
            prediction_mode,
            label_mode,
            bold: false,
            shade: false,
            snapping: true,
            random_order: None,
            show_marks,
            threshold: 0.25,
            drag_start: None,
            drag_now: None,
            pan_start: None,
            canvas_scale: 1.0,
            view_cache: None,
            scrollfield,
            pending_full: None,
            launcher_selected,
            launcher_project: None,
            launcher_stats: None,
            launcher_files: Vec::new(),
            launcher_hide_extra_weights: false,
            launcher_stale: true,
        };
        if let Some(project) = project {
            app.open_project(Path::new(&project));
        }
        app
    }

    fn fail(&mut self, error: String) {
        self.status = error;
        self.error = true;
        self.status_at = Some(Instant::now());
    }

    fn ok(&mut self, message: impl Into<String>) {
        self.status = message.into();
        self.error = false;
        self.status_at = Some(Instant::now());
    }

    fn open_project(&mut self, root: &Path) {
        match Project::load(root) {
            Ok(project) => {
                self.settings.remember(&project.root);
                if let Some(cfg) = &project.cfg_file {
                    self.cfg_path = cfg.to_string_lossy().into_owned();
                }
                if let Some(weights) = &project.weights_file {
                    self.weights_path = weights.to_string_lossy().into_owned();
                }
                self.scrollfield = ScrollField::new(
                    self.settings.scrollfield_width,
                    self.settings.scrollfield_marker_size,
                );
                self.project = Some(project);
                self.cache.clear();
                self.editor = None;
                self.index = 0;
                self.dirty = false;
                self.zoom = None;
                self.pan = Vec2::ZERO;
                self.pending_full = None;
                self.show_launcher = false;
                self.load_current(LoadKind::Full);
            }
            Err(error) => self.fail(error),
        }
    }

    /// Saves the current image and returns whether it reached disk. The caller
    /// keeps the edits in memory when this returns `false`.
    fn save(&mut self) -> bool {
        let (Some(project), Some(editor)) = (self.project.as_mut(), self.editor.as_ref()) else {
            return false;
        };
        let mut annotation = editor.to_annotation();
        annotation.scale = self.canvas_scale;
        let boxes = annotation.real_marks().count();
        let index = self.index;
        match project.save_annotation(index, &annotation) {
            Ok(()) => {
                self.dirty = false;
                self.ok(format!("saved {boxes} box(es)"));
                // One scrollfield row is rewritten on save
                // (`DMContent.cpp:3024-3027`).
                let position = self
                    .visible_indices()
                    .iter()
                    .position(|value| *value == index);
                if let Some(position) = position
                    && let Some(project) = self.project.as_ref()
                {
                    self.scrollfield.refresh_row(position, project, index);
                }
                true
            }
            Err(error) => {
                self.fail(error);
                false
            }
        }
    }

    /// Loads the current image. `Quick` skips the sidecar and the detector, like
    /// DarkMark's `load_image(..., full_load = false)`. The caller must have
    /// saved the edits of the image it is leaving; see [`App::go_to`].
    fn load_current(&mut self, kind: LoadKind) {
        let Some(project) = &self.project else {
            return;
        };
        let Some(entry) = project.images.get(self.index) else {
            return;
        };
        let classes = project.classes.clone();
        if kind == LoadKind::Full {
            match project.load_annotation(self.index) {
                Ok(annotation) => {
                    self.editor = Some(Editor::from_annotation(&annotation, classes));
                    self.dirty = false;
                }
                Err(error) => {
                    self.fail(error);
                    return;
                }
            }
        } else {
            let (width, height) = entry.dimensions().unwrap_or((1, 1));
            self.editor = Some(Editor::new(width, height, classes));
            self.dirty = false;
        }
        self.pan = Vec2::ZERO;
        // Every prediction-related reload runs Darknet again when it is not off
        // (`DMContent.cpp:2537`).
        if kind == LoadKind::Full && self.prediction_mode != 0 && self.worker.is_some() {
            let _ = self.detect();
        }
    }

    /// Jumps to `index`. The edits of the image being left are saved first: a
    /// save writes to `self.index`, so moving the index before saving would put
    /// them in the next image's sidecar. A failed save keeps the edits and
    /// cancels the jump. A quick load schedules the full load 500 ms later.
    fn go_to(&mut self, index: usize, kind: LoadKind) {
        if self.project.is_none() {
            return;
        }
        if self.dirty && !self.save() {
            return;
        }
        self.index = index;
        self.load_current(kind);
        self.pending_full = match kind {
            LoadKind::Quick => Some(Instant::now()),
            LoadKind::Full => None,
        };
    }

    fn tick_pending_load(&mut self, ctx: &egui::Context) {
        let Some(started) = self.pending_full else {
            return;
        };
        if started.elapsed() >= FULL_LOAD_DELAY {
            self.pending_full = None;
            // Through `go_to` so an edit made during the quick load is saved
            // instead of being overwritten by the full load.
            self.go_to(self.index, LoadKind::Full);
        } else {
            ctx.request_repaint_after(FULL_LOAD_DELAY - started.elapsed());
        }
    }

    fn visible_indices(&self) -> Vec<usize> {
        let Some(project) = &self.project else {
            return Vec::new();
        };
        let mut filter = self.filter.clone();
        if !self.filter_text.trim().is_empty() {
            filter.include =
                regex::Regex::new(&format!("(?i){}", regex::escape(&self.filter_text))).ok();
        }
        if !self.filter_exclude.trim().is_empty() {
            filter.exclude = regex::Regex::new(&self.filter_exclude).ok();
        }
        let mut indices = filter.apply(project, self.sort);
        if let Some(order) = &self.random_order {
            let rank = |index: usize| {
                order
                    .iter()
                    .position(|value| *value == index)
                    .unwrap_or(usize::MAX)
            };
            indices.sort_by_key(|index| rank(*index));
        }
        indices
    }

    fn step_image(&mut self, delta: isize, kind: LoadKind) {
        let visible = self.visible_indices();
        if visible.is_empty() {
            return;
        }
        let position = visible
            .iter()
            .position(|index| *index == self.index)
            .unwrap_or(0);
        let next = (position as isize + delta).clamp(0, visible.len() as isize - 1) as usize;
        if visible[next] != self.index || kind == LoadKind::Full {
            self.go_to(visible[next], kind);
        }
    }

    fn page_image(&mut self, dir: nav::Dir) {
        let Some(project) = &self.project else {
            return;
        };
        if let Some(next) = nav::find_unmarked(project, self.index, dir)
            && next != self.index
        {
            self.go_to(next, LoadKind::Full);
        }
    }

    fn set_label_mode(&mut self, mode: u8) {
        self.label_mode = mode.min(2);
        self.settings.label_mode = self.label_mode;
        self.settings.save();
    }

    fn open_class_editor(&mut self) {
        self.class_edit = self
            .editor
            .as_ref()
            .map(|editor| (0..editor.class_count).map(|id| editor.class_name(id)).collect())
            .unwrap_or_default();
        self.class_plan = None;
        self.show_classes = true;
    }

    fn run_action(&mut self, action: Action) {
        // These need `App` state as well as the editor, so they run before the
        // editor borrow below.
        match action {
            Action::AcceptAll => {
                self.accept_all_marks();
                return;
            }
            Action::AcceptSelected => {
                self.accept_mark();
                return;
            }
            Action::TogglePredictions => {
                self.set_prediction_mode(hotkeys::next_prediction_mode(self.prediction_mode));
                return;
            }
            Action::SetPredictions(mode) => {
                self.set_prediction_mode(mode);
                return;
            }
            Action::SetClass(class_id) => {
                self.set_class(class_id);
                return;
            }
            Action::ToggleMarks => {
                self.toggle_marks();
                return;
            }
            Action::Settings => {
                if self.dirty {
                    self.save();
                }
                self.show_settings = true;
                return;
            }
            Action::Filter => {
                if self.dirty {
                    self.save();
                }
                self.show_filter = true;
                return;
            }
            Action::Jump => {
                self.show_jump = true;
                return;
            }
            Action::Help => {
                self.show_help = true;
                return;
            }
            Action::ClassMenu => {
                self.open_class_editor();
                return;
            }
            _ => {}
        }
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        match action {
            Action::ClassDigit(digit) => {
                if let Err(error) = editor.set_class(digit as usize, Mods::default()) {
                    self.fail(error);
                } else {
                    self.dirty = true;
                }
            }
            Action::PrevImage => self.step_image(-1, LoadKind::Full),
            Action::NextImage => self.step_image(1, LoadKind::Full),
            Action::FirstImage => {
                self.zoom = None;
                self.go_to(0, LoadKind::Full);
            }
            Action::LastImage => {
                self.zoom = None;
                if let Some(project) = &self.project {
                    let last = project.images.len().saturating_sub(1);
                    self.go_to(last, LoadKind::Full);
                }
            }
            Action::PrevUnmarked => self.page_image(nav::Dir::Previous),
            Action::NextUnmarked => self.page_image(nav::Dir::Next),
            Action::CycleMark(backward) => editor.cycle(backward),
            Action::ThresholdUp => {
                self.threshold = (self.threshold + 0.05).min(0.95);
                editor.set_threshold(self.threshold);
            }
            Action::ThresholdDown => {
                self.threshold = (self.threshold - 0.05).max(0.05);
                editor.set_threshold(self.threshold);
            }
            Action::DeleteMark => {
                if editor.delete_selected() {
                    self.dirty = true;
                }
            }
            Action::ClearMarks => {
                editor.clear_marks();
                self.dirty = true;
            }
            Action::ToggleEmpty => {
                let flagged = !editor.completely_empty;
                if flagged && editor.has_real_marks() {
                    self.fail("delete annotations before marking the image as empty".to_owned());
                } else {
                    editor.set_empty_flag(flagged);
                    self.dirty = true;
                }
            }
            Action::CopyPrev => self.copy_neighbour(nav::Dir::Previous),
            Action::CopyNext => self.copy_neighbour(nav::Dir::Next),
            Action::ToggleLabels => {
                let next = (self.label_mode + 1) % 3;
                self.set_label_mode(next);
            }
            Action::ToggleBold => self.bold = !self.bold,
            Action::ToggleShade => self.shade = !self.shade,
            Action::ToggleSnapping => {
                self.snapping = !self.snapping;
                self.ok(format!(
                    "snapping: {}",
                    if self.snapping { "on" } else { "off" }
                ));
            }
            // Handled before the editor borrow, above.
            Action::AcceptAll
            | Action::AcceptSelected
            | Action::TogglePredictions
            | Action::SetPredictions(_)
            | Action::SetClass(_)
            | Action::ToggleMarks
            | Action::Settings
            | Action::Filter
            | Action::Jump
            | Action::Help
            | Action::ClassMenu => unreachable!("handled before the editor borrow"),
            Action::Scale(delta) => {
                editor.scale(delta);
                self.dirty = true;
            }
            Action::Nudge(dx, dy) => {
                editor.nudge(dx as f64, dy as f64);
                self.dirty = true;
            }
            Action::Undo => {
                if editor.undo() {
                    self.dirty = true;
                }
            }
            Action::Redo => {
                if editor.redo() {
                    self.dirty = true;
                }
            }
            Action::Save => {
                self.save();
            }
            Action::SnapAll => {
                let Some(image) = self.cache.entries.get(&self.index).map(|entry| entry.gray.clone())
                else {
                    return;
                };
                let mut annotation = editor.to_annotation();
                let moved = actions::snap(&mut annotation, &image, 6, actions::SnapScope::All);
                if moved > 0 {
                    editor.marks = annotation.marks;
                    self.dirty = true;
                }
                self.ok(format!("snapped {moved} edge(s)"));
            }
            Action::ClipboardCopy => {
                let text = actions::clipboard_copy(&editor.to_annotation());
                match self.clipboard.as_mut() {
                    Some(clipboard) => match clipboard.set_text(text) {
                        Ok(()) => self.ok("copied marks to the clipboard"),
                        Err(error) => self.fail(format!("clipboard: {error}")),
                    },
                    None => self.fail("no system clipboard available".to_owned()),
                }
            }
            Action::ClipboardPaste => {
                let classes = editor.class_names.clone();
                let text = match self.clipboard.as_mut() {
                    Some(clipboard) => match clipboard.get_text() {
                        Ok(text) => text,
                        Err(error) => {
                            self.fail(format!("clipboard: {error}"));
                            return;
                        }
                    },
                    None => {
                        self.fail("no system clipboard available".to_owned());
                        return;
                    }
                };
                match actions::clipboard_paste(&text, editor.width, editor.height, &classes) {
                    Ok(marks) => {
                        for mark in marks {
                            editor.marks.push(mark);
                        }
                        self.dirty = true;
                    }
                    Err(error) => self.fail(format!("paste: {error}")),
                }
            }
            Action::ZoomIn => {
                let next = (self.canvas_scale as f32 + 0.1).min(5.0);
                self.zoom = Some(next);
            }
            Action::ZoomOut => {
                let next = (self.canvas_scale as f32 - 0.1).max(0.01);
                self.zoom = Some(next);
            }
            Action::ZoomToggle => {
                self.zoom = match self.zoom {
                    Some(_) => None,
                    None => Some(self.canvas_scale as f32),
                };
            }
            Action::SortRandom => {
                let visible = self.visible_indices();
                let mut order = visible.clone();
                let mut seed = 0x2545_f491_4f6c_dd1du64;
                for index in (1..order.len()).rev() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    order.swap(index, (seed % (index as u64 + 1)) as usize);
                }
                self.random_order = Some(order);
            }
            Action::SortAlphabetical => {
                self.random_order = None;
                self.sort = SortKey::Name;
            }
            Action::Cancel => {
                self.zoom = None;
                self.pan = Vec2::ZERO;
                if let Some(editor) = self.editor.as_mut() {
                    editor.selected = None;
                }
            }
            Action::DeleteImage
            | Action::UndoDeleteImage
            | Action::MassDelete
            | Action::Homography
            | Action::Merge
            | Action::HeatmapCycle
            | Action::ToggleHeatmap
            | Action::Screenshot(_)
            | Action::WindowScreenshot
            | Action::ToggleTiling
            | Action::TileEdge(_)
            | Action::TileRect(_) => {
                self.ok(format!(
                    "{action:?} is mapped but not implemented in this build"
                ));
            }
        }
    }

    fn set_class(&mut self, class_id: usize) {
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        match editor.set_class(class_id, Mods::default()) {
            Ok(()) => self.dirty = true,
            Err(error) => self.fail(error),
        }
    }

    fn copy_neighbour(&mut self, dir: nav::Dir) {
        let Some(project) = &self.project else {
            return;
        };
        let len = project.images.len();
        let mut cursor = self.index;
        let source = loop {
            let next = match dir {
                nav::Dir::Previous if cursor == 0 => break None,
                nav::Dir::Next if cursor + 1 >= len => break None,
                nav::Dir::Previous => cursor - 1,
                nav::Dir::Next => cursor + 1,
            };
            cursor = next;
            if project.mark_count(cursor) > 0 {
                break Some(project.load_annotation(cursor).ok());
            }
        };
        let Some(Some(source)) = source else {
            self.fail("no neighbouring image with marks".to_owned());
            return;
        };
        if let Some(editor) = self.editor.as_mut() {
            let added = editor.copy_from(&source);
            self.dirty |= added > 0;
            self.ok(format!("copied {added} box(es)"));
        }
    }

    fn detect(&mut self) -> Result<usize, String> {
        let Some(worker) = self.worker.as_mut() else {
            return Err("no model is loaded".to_owned());
        };
        let (Some(project), Some(editor)) = (&self.project, self.editor.as_mut()) else {
            return Err("no image is open".to_owned());
        };
        let Some(entry) = project.images.get(self.index) else {
            return Err("no image is open".to_owned());
        };
        let predictions = worker.detect(&entry.path, self.threshold)?;
        let count = predictions.len();
        let marks: Vec<Mark> = predictions
            .into_iter()
            .map(|Prediction { class, conf, cx, cy, w, h }| {
                let (width, height) = (editor.width as f64, editor.height as f64);
                let (cx, cy, w, h) = (cx as f64, cy as f64, w as f64, h as f64);
                Mark::prediction(
                    class,
                    editor.class_name(class),
                    (cx - w * 0.5) * width,
                    (cy - h * 0.5) * height,
                    (cx + w * 0.5) * width,
                    (cy + h * 0.5) * height,
                    conf,
                )
            })
            .collect();
        editor.set_predictions(marks);
        Ok(count)
    }

    /// `p` and the prediction menu: store the new `show_predictions` value, then
    /// re-run Darknet. DarkMark reloads the image for every change so the
    /// predictions match the current mode (`DMC:2239-2252`).
    fn set_prediction_mode(&mut self, mode: u8) {
        self.prediction_mode = mode.min(2);
        self.settings.prediction_mode = self.prediction_mode;
        self.settings.save();
        let label = prediction_label(self.prediction_mode);
        if self.prediction_mode == 0 {
            // Turning predictions off drops the ones in memory: they are never
            // saved, and DarkMark rebuilds the image without them.
            if let Some(editor) = self.editor.as_mut() {
                editor.set_predictions(Vec::new());
            }
            self.ok(format!("predictions: {label}"));
            return;
        }
        match self.worker.is_some().then(|| self.detect()) {
            Some(Err(error)) => self.fail(format!("predictions: {label}; {error}")),
            _ => self.ok(format!("predictions: {label}")),
        }
    }

    /// `m`: DarkMark's `toggle_show_marks` (`DMC:1713-1736`).
    fn toggle_marks(&mut self) {
        self.show_marks = !self.show_marks;
        self.settings.show_marks = self.show_marks;
        self.settings.save();
        self.ok(format!(
            "user marks: {}",
            if self.show_marks { "visible" } else { "hidden" }
        ));
    }

    /// `a`: accept every prediction (`DMC:3922-3961`).
    fn accept_all_marks(&mut self) {
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        match editor.accept_all() {
            Ok(accepted) => {
                self.dirty = true;
                self.ok(format!("accepted {accepted} prediction(s)"));
            }
            Err(error) => self.fail(format!("cannot accept predictions: {error}")),
        }
    }

    /// `A`: accept the selected prediction; with nothing selected accept all
    /// (`DMC:3893-3920`).
    fn accept_mark(&mut self) {
        let Some(editor) = self.editor.as_mut() else {
            return;
        };
        if editor.selected.is_none() {
            self.accept_all_marks();
            return;
        }
        match editor.accept_selected() {
            Ok(()) => {
                self.dirty = true;
                self.ok("accepted the selected prediction");
            }
            Err(error) => self.fail(format!("cannot accept the prediction: {error}")),
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        // Never steal keys from a focused text field (filter, paths, class names).
        if ctx.wants_keyboard_input() {
            return;
        }
        let mut actions = Vec::new();
        ctx.input(|input| {
            let mods = Mods {
                ctrl: input.modifiers.command,
                alt: input.modifiers.alt,
                shift: input.modifiers.shift,
            };
            let mut push = |key: Key| {
                if let Some(action) = map_key(KeyInput { key, mods }) {
                    actions.push(action);
                }
            };
            for (digit, key) in [
                (0, egui::Key::Num0),
                (1, egui::Key::Num1),
                (2, egui::Key::Num2),
                (3, egui::Key::Num3),
                (4, egui::Key::Num4),
                (5, egui::Key::Num5),
                (6, egui::Key::Num6),
                (7, egui::Key::Num7),
                (8, egui::Key::Num8),
                (9, egui::Key::Num9),
            ] {
                if input.key_pressed(key) {
                    push(Key::Digit(digit));
                }
            }
            for (key, mapped) in [
                (egui::Key::ArrowLeft, Key::Left),
                (egui::Key::ArrowRight, Key::Right),
                (egui::Key::ArrowUp, Key::Up),
                (egui::Key::ArrowDown, Key::Down),
                (egui::Key::Home, Key::Home),
                (egui::Key::End, Key::End),
                (egui::Key::PageUp, Key::PageUp),
                (egui::Key::PageDown, Key::PageDown),
                (egui::Key::Tab, Key::Tab),
                (egui::Key::Delete, Key::Delete),
                (egui::Key::Backspace, Key::Backspace),
                (egui::Key::Enter, Key::Enter),
                (egui::Key::Space, Key::Space),
                (egui::Key::Escape, Key::Escape),
                (egui::Key::F1, Key::F1),
                (egui::Key::F2, Key::F2),
                (egui::Key::Plus, Key::Plus),
                (egui::Key::Minus, Key::Minus),
            ] {
                if input.key_pressed(key) {
                    push(mapped);
                }
            }
            for (key, ch) in [
                (egui::Key::A, 'a'),
                (egui::Key::C, 'c'),
                (egui::Key::D, 'd'),
                (egui::Key::E, 'e'),
                (egui::Key::F, 'f'),
                (egui::Key::H, 'h'),
                (egui::Key::J, 'j'),
                (egui::Key::M, 'm'),
                (egui::Key::N, 'n'),
                (egui::Key::O, 'o'),
                (egui::Key::P, 'p'),
                (egui::Key::R, 'r'),
                (egui::Key::S, 's'),
                (egui::Key::U, 'u'),
                (egui::Key::V, 'v'),
                (egui::Key::W, 'w'),
                (egui::Key::Y, 'y'),
                (egui::Key::Z, 'z'),
            ] {
                if input.key_pressed(key) {
                    let ch = if input.modifiers.shift {
                        ch.to_ascii_uppercase()
                    } else {
                        ch
                    };
                    push(Key::Char(ch));
                }
            }
            if input.key_pressed(egui::Key::Period) {
                push(Key::Char('.'));
            }
            if input.key_pressed(egui::Key::Comma) {
                push(Key::Char(','));
            }
        });
        for action in actions {
            self.run_action(action);
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let Some(project) = &self.project else {
            ui.centered_and_justified(|ui| ui.label("No project open"));
            return;
        };
        let Some(entry) = project.images.get(self.index) else {
            ui.centered_and_justified(|ui| ui.label("No image"));
            return;
        };
        let path = entry.path.clone();
        let relative = entry.relative.clone();
        if let Err(error) = self.cache.get(ui.ctx(), self.index, &path) {
            self.fail(error);
            return;
        }
        let (width, height, view_image) = {
            let Some(texture) = self.cache.entries.get(&self.index) else {
                return;
            };
            let size = (texture.rgba.width() as f32, texture.rgba.height() as f32);
            // The filtered views are expensive (Canny especially); keep the last
            // one per image instead of re-uploading it every frame.
            let handle = match self.view {
                View::Normal => texture.handle.clone(),
                view => {
                    let key = format!("{}-{}", view.label(), self.index);
                    let cached = match &self.view_cache {
                        Some((cached, handle)) if *cached == key => Some(handle.clone()),
                        _ => None,
                    };
                    match cached {
                        Some(handle) => handle,
                        None => {
                            let filtered = apply_view(view, &texture.gray, &texture.rgba);
                            let size = [filtered.width() as usize, filtered.height() as usize];
                            let color =
                                egui::ColorImage::from_rgba_unmultiplied(size, filtered.as_raw());
                            let handle = ui.ctx().load_texture(
                                key.clone(),
                                color,
                                egui::TextureOptions::LINEAR,
                            );
                            self.view_cache = Some((key, handle.clone()));
                            handle
                        }
                    }
                }
            };
            (size.0, size.1, handle)
        };

        let response = ui.allocate_response(ui.available_size(), Sense::click_and_drag());
        let rect = response.rect;
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(18));
        // Fit to the canvas (which already excludes the scrollfield) unless the
        // user set a manual zoom with `+`/`-`/Space (`DMContent::resized`).
        let fit = (rect.width() / width).min(rect.height() / height).max(0.01);
        let scale = self.zoom.unwrap_or(fit).clamp(0.01, 8.0);
        self.canvas_scale = scale as f64;
        let image_rect =
            Rect::from_center_size(rect.center() + self.pan, Vec2::new(width, height) * scale);
        painter.image(
            view_image.id(),
            image_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );

        let to_screen = |p: Point| {
            Pos2::new(
                image_rect.left() + (p.x / width as f64) as f32 * image_rect.width(),
                image_rect.top() + (p.y / height as f64) as f32 * image_rect.height(),
            )
        };
        let to_image = |p: Pos2| {
            Point::new(
                ((p.x - image_rect.left()) / image_rect.width().max(1.0)) as f64 * width as f64,
                ((p.y - image_rect.top()) / image_rect.height().max(1.0)) as f64 * height as f64,
            )
        };

        let mut real_count = 0;
        let mut prediction_count = 0;
        if let Some(editor) = &self.editor {
            real_count = editor.real_marks().count();
            prediction_count = editor.marks.iter().filter(|mark| mark.is_prediction).count();
            // `DMCanvas::rebuild_cache_image` (`DMCanvas.cpp:289-330`).
            let marks_shown = self.show_marks && editor.has_real_marks();
            let predictions_visible = edit::predictions_are_visible(
                self.prediction_mode,
                marks_shown,
                self.drag_start.is_some(),
            );
            for mark in &editor.marks {
                if mark.is_prediction && !predictions_visible {
                    continue;
                }
                if !mark.is_prediction && !self.show_marks {
                    continue;
                }
                let color = if mark.is_prediction {
                    PREDICTION_COLOR
                } else {
                    class_color(mark.class_id)
                };
                let selected = editor
                    .selected
                    .and_then(|index| editor.marks.get(index))
                    .is_some_and(|selected| std::ptr::eq(selected, mark));
                let line = if selected || self.bold { 3.0_f32 } else { 2.0_f32 };
                let stroke = Stroke::new(line, color);
                let corners: Vec<Pos2> = mark.points.iter().map(|p| to_screen(*p)).collect();
                if self.shade {
                    painter.rect_filled(
                        Rect::from_two_pos(corners[0], corners[2]),
                        0.0,
                        Color32::from_rgba_unmultiplied(0, 0, 0, 96),
                    );
                }
                for index in 0..4 {
                    painter.line_segment([corners[index], corners[(index + 1) % 4]], stroke);
                }
                let labels_visible = match self.label_mode {
                    1 => true,
                    2 => selected || mark.is_prediction,
                    _ => false,
                };
                if labels_visible {
                    let text = if mark.is_prediction {
                        format!("{} {:.2}", editor.class_name(mark.class_id), mark.confidence)
                    } else {
                        editor.class_name(mark.class_id)
                    };
                    overlay::mark_label(&painter, corners[0], text, color);
                }
            }
            if let Some((a, b)) = editor.preview() {
                let a = to_screen(a);
                let b = to_screen(b);
                let stroke = Stroke::new(1.0_f32, Color32::WHITE);
                for (p, q) in [
                    (a, Pos2::new(b.x, a.y)),
                    (Pos2::new(b.x, a.y), b),
                    (b, Pos2::new(a.x, b.y)),
                    (Pos2::new(a.x, b.y), a),
                ] {
                    painter.line_segment([p, q], stroke);
                }
            }
        }

        // DarkMark hides the system pointer over the canvas and paints a 1 px
        // crosshair through the mouse instead (`CrosshairComponent.cpp:40-49`,
        // `:119-137`). A rubber band or a pan replaces it.
        let dragging = self.drag_start.is_some();
        let panning = self.pan_start.is_some();
        if let Some(pointer) = response.hover_pos()
            && image_rect.contains(pointer)
        {
            ui.ctx().set_cursor_icon(egui::CursorIcon::None);
            if !dragging && !panning {
                overlay::crosshair(&painter, rect, pointer);
            }
        }

        // Status on the canvas: filename, position, zoom, counts, dirty flag.
        let mut lines = vec![
            relative,
            format!(
                "{}/{}  {:.0}%  {real_count} marks  {prediction_count} predictions{}",
                self.index + 1,
                project.images.len(),
                scale * 100.0,
                if self.dirty { "  *unsaved" } else { "" }
            ),
        ];
        if self.zoom.is_some() {
            lines.push("manual zoom: Space or Esc returns to fit".to_owned());
        }
        overlay::status(&painter, rect, &lines);
        if let Some(at) = self.status_at
            && at.elapsed().as_secs_f32() < overlay::MESSAGE_SECONDS
        {
            overlay::bubble(&painter, rect, &self.status, self.error);
            ui.ctx().request_repaint_after(Duration::from_millis(120));
        }

        let mods = ui.input(|input| Mods {
            ctrl: input.modifiers.command,
            alt: input.modifiers.alt,
            shift: input.modifiers.shift,
        });
        if response.drag_started() {
            // `drag_started()` only fires once the pointer has moved past the drag
            // threshold, so the current position is not the press point; ask for
            // it, otherwise the first few pixels of every box are lost.
            let pressed = ui
                .input(|input| input.pointer.press_origin())
                .or_else(|| response.interact_pointer_pos());
            if let Some(pos) = pressed {
                if mods.ctrl {
                    self.pan_start = Some((self.pan, pos));
                } else {
                    self.drag_start = Some(pos);
                    self.drag_now = Some(pos);
                    if let Some(editor) = self.editor.as_mut()
                        && editor.press(to_image(pos), mods, false)
                    {
                        self.dirty = true;
                    }
                }
            }
        }
        if response.dragged() {
            if let (Some((pan, start)), Some(pos)) =
                (self.pan_start, response.interact_pointer_pos())
            {
                self.pan = pan + (pos - start);
            } else if let Some(pos) = response.interact_pointer_pos() {
                self.drag_now = Some(pos);
                if let Some(editor) = self.editor.as_mut() {
                    editor.drag(to_image(pos));
                }
            }
        }
        if response.drag_stopped() {
            if mods.ctrl && self.pan_start.take().is_some() {
                // pan ended
            } else {
                // The release position is newer than the last drag frame's.
                let position = ui
                    .input(|input| input.pointer.latest_pos())
                    .or_else(|| response.interact_pointer_pos())
                    .or(self.drag_now);
                if let (Some(_start), Some(pos)) = (self.drag_start.take(), position)
                    && let Some(editor) = self.editor.as_mut()
                    && editor.release(to_image(pos))
                {
                    self.dirty = true;
                }
            }
        } else if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
            && let Some(editor) = self.editor.as_mut()
        {
            let point = to_image(pos);
            let changed = editor.press(point, mods, false);
            editor.drag(point);
            if editor.release(point) || changed {
                self.dirty = true;
            }
        }
        if response.double_clicked()
            && let Some(pos) = response.interact_pointer_pos()
            && let Some(editor) = self.editor.as_mut()
            && editor.press(to_image(pos), mods, true)
        {
            self.dirty = true;
        }

        // DarkMark's wheel: in manual zoom it leaves zoom, otherwise it steps one
        // image with a quick load and a 500 ms full-load timer
        // (`CrosshairComponent.cpp:310-355`).
        let scroll = ui.input(|input| input.raw_scroll_delta.y);
        if response.hovered() && scroll != 0.0 {
            if self.zoom.is_some() {
                self.zoom = None;
            } else {
                let delta = if scroll < 0.0 { 1 } else { -1 };
                self.step_image(delta, LoadKind::Quick);
            }
        }

        response.context_menu(|ui| popup::menu(ui, self));
    }

    fn scrollfield_panel(&mut self, ctx: &egui::Context) {
        let width = self.settings.scrollfield_width;
        if width < 1.0 || self.project.is_none() {
            return;
        }
        let visible = self.visible_indices();
        let current = visible.iter().position(|index| *index == self.index);
        let alphabetical = self.sort == SortKey::Name && self.random_order.is_none();
        let mut quick = None;
        let mut full = None;
        let mut response = None;
        {
            let Some(project) = self.project.as_ref() else {
                return;
            };
            let scrollfield = &mut self.scrollfield;
            scrollfield.width = width;
            scrollfield.marker_size = self.settings.scrollfield_marker_size;
            scrollfield.sync(ctx, project, &visible);
            egui::SidePanel::right("scrollfield")
                .exact_width(width + SCROLLFIELD_SPACER)
                .resizable(false)
                .frame(egui::Frame::NONE.fill(Color32::from_gray(32)))
                .show(ctx, |ui| {
                    let interaction = scrollfield.show(ui, current, alphabetical);
                    quick = interaction.quick.map(|position| visible[position]);
                    full = interaction.full.map(|position| visible[position]);
                    response = Some(interaction.response);
                });
        }
        // Right-click opens the same popup as the canvas (`ScrollField.cpp:332-345`).
        if let Some(response) = response {
            response.context_menu(|ui| popup::menu(ui, self));
        }
        if let Some(index) = quick {
            self.go_to(index, LoadKind::Quick);
        }
        if let Some(index) = full {
            self.go_to(index, LoadKind::Full);
        }
    }

    fn windows(&mut self, ctx: &egui::Context) {
        self.help_window(ctx);
        self.class_window(ctx);
        self.filter_window(ctx);
        self.settings_window(ctx);
        self.jump_window(ctx);
        self.model_window(ctx);
        if self.show_new_project {
            self.new_project_window(ctx);
        }
    }

    fn help_window(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let mut open = true;
        egui::Window::new("Hotkeys (F2)").open(&mut open).show(ctx, |ui| {
            ui.label("digits class; Ctrl/Alt +10/+20");
            ui.label("arrows / Home / End / PageUp / PageDown navigate");
            ui.label("Tab cycle; Del delete; C clear; n empty flag");
            ui.label("a accept all; A accept selected; p predictions; m marks");
            ui.label("l labels; b bold; B shade; r random; R alphabetical");
            ui.label("y / Y copy from previous / next; Ctrl+WASD nudge; , . scale");
            ui.label("Ctrl+Z undo; Ctrl+Shift+S save; Ctrl+C/V clipboard");
            ui.label("Space fit/manual zoom; + / - zoom by 0.1; wheel steps image");
            ui.label("e settings; f filter; j jump; c class editor; F2 this window");
            ui.label("right-click: classes, sort, view, filter, windows");
        });
        self.show_help = open;
    }

    fn class_window(&mut self, ctx: &egui::Context) {
        if !self.show_classes {
            return;
        }
        let mut open = self.show_classes;
        egui::Window::new("Classes")
            .open(&mut open)
            .collapsible(false)
            .show(ctx, |ui| {
                let mut remove = None;
                for (index, name) in self.class_edit.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{index}:"));
                        ui.add(egui::TextEdit::singleline(name).desired_width(220.0));
                        if ui.button("✕").clicked() {
                            remove = Some(index);
                        }
                    });
                }
                if let Some(index) = remove {
                    self.class_edit.remove(index);
                }
                if ui.button("Add class").clicked() {
                    self.class_edit.push(format!("class {}", self.class_edit.len()));
                }
                if ui.button("Dry run").clicked() {
                    self.class_plan = Some(self.local_plan());
                }
                if let Some(plan) = &self.class_plan {
                    ui.separator();
                    ui.label(plan);
                }
                if ui.button("Apply").clicked() {
                    let result = self.apply_classes();
                    match result {
                        Ok(message) => self.ok(message),
                        Err(error) => self.fail(error),
                    }
                }
            });
        self.show_classes = open;
    }

    fn filter_window(&mut self, ctx: &egui::Context) {
        if !self.show_filter {
            return;
        }
        let mut open = true;
        egui::Window::new("Filter").open(&mut open).show(ctx, |ui| {
            ui.label("Include (case-insensitive literal):");
            ui.add(egui::TextEdit::singleline(&mut self.filter_text).desired_width(320.0));
            ui.label("Exclude (regex):");
            ui.add(egui::TextEdit::singleline(&mut self.filter_exclude).desired_width(320.0));
            ui.separator();
            tri_row(ui, "Unannotated:", &mut self.filter.unannotated);
            tri_row(ui, "Empty flag:", &mut self.filter.empty);
            ui.separator();
            ui.label("Class:");
            let names: Vec<String> = self
                .project
                .as_ref()
                .map(|project| project.classes.clone())
                .unwrap_or_default();
            let mut class = self.filter.class;
            egui::ComboBox::from_id_salt("filter-class")
                .selected_text(
                    class
                        .and_then(|index| names.get(index).cloned())
                        .unwrap_or_else(|| "any".to_owned()),
                )
                .show_ui(ui, |ui| {
                    if ui.selectable_label(class.is_none(), "any").clicked() {
                        class = None;
                    }
                    for (index, name) in names.iter().enumerate() {
                        if ui
                            .selectable_label(class == Some(index), format!("{index}: {name}"))
                            .clicked()
                        {
                            class = Some(index);
                        }
                    }
                });
            self.filter.class = class;
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Reset").clicked() {
                    self.filter = Filter::default();
                    self.filter_text.clear();
                    self.filter_exclude.clear();
                }
                if ui.button("Apply").clicked() {
                    self.show_filter = false;
                }
            });
            let visible = self.visible_indices().len();
            let total = self.project.as_ref().map(|project| project.images.len()).unwrap_or(0);
            ui.label(format!("{visible} of {total} image(s) visible"));
        });
        self.show_filter = open;
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.show_settings {
            return;
        }
        let mut open = true;
        egui::Window::new("Settings").open(&mut open).show(ctx, |ui| {
            let mut changed = false;
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.scrollfield_width, 0.0..=200.0)
                        .step_by(10.0)
                        .text("Scrollfield width (0 disables)"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut self.settings.scrollfield_marker_size, 3.0..=20.0)
                        .text("Directory marker size"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut self.threshold, 0.05..=0.95).text("Prediction threshold"))
                .changed();
            changed |= ui.checkbox(&mut self.snapping, "Snapping").changed();
            let mut marks = self.show_marks;
            if ui.checkbox(&mut marks, "Show user marks").changed() {
                self.toggle_marks();
            }
            ui.label("Predictions and labels also persist; change them from the popup.");
            if changed {
                self.settings.save();
                self.scrollfield.width = self.settings.scrollfield_width;
                self.scrollfield.marker_size = self.settings.scrollfield_marker_size;
            }
        });
        self.show_settings = open;
    }

    fn jump_window(&mut self, ctx: &egui::Context) {
        if !self.show_jump {
            return;
        }
        let last = self
            .project
            .as_ref()
            .map(|project| project.images.len())
            .unwrap_or(1)
            .max(1);
        let mut open = true;
        let mut target = (self.index + 1).clamp(1, last);
        egui::Window::new("Jump").open(&mut open).show(ctx, |ui| {
            ui.add(egui::Slider::new(&mut target, 1..=last));
            ui.horizontal(|ui| {
                if ui.button("Go").clicked() {
                    self.go_to(target - 1, LoadKind::Full);
                    self.show_jump = false;
                }
                if ui.button("Cancel").clicked() {
                    self.show_jump = false;
                }
            });
        });
        self.show_jump = open && self.show_jump;
    }

    fn model_window(&mut self, ctx: &egui::Context) {
        if !self.show_model {
            return;
        }
        let mut open = true;
        egui::Window::new("Model").open(&mut open).show(ctx, |ui| {
            ui.label("cfg:");
            ui.add(egui::TextEdit::singleline(&mut self.cfg_path).desired_width(420.0));
            ui.label("weights:");
            ui.add(egui::TextEdit::singleline(&mut self.weights_path).desired_width(420.0));
            ui.horizontal(|ui| {
                if ui.button("Load model").clicked() {
                    // `cargo run -p lightnet-gui` puts the GUI binary here, not
                    // the CLI, so prefer an explicit path.
                    let bin = std::env::var_os("LIGHTNET_RS_BIN")
                        .map(PathBuf::from)
                        .or_else(|| std::env::current_exe().ok())
                        .unwrap_or_else(|| PathBuf::from("lightnet-rs"));
                    if bin.file_name().and_then(|name| name.to_str()) == Some("lightnet-gui") {
                        self.fail(
                            "set LIGHTNET_RS_BIN to the lightnet-rs binary (cargo build -p lightnet-cli)"
                                .to_owned(),
                        );
                    } else {
                        match DetectWorker::spawn(
                            &bin,
                            Path::new(&self.cfg_path),
                            Path::new(&self.weights_path),
                        ) {
                            Ok(worker) => {
                                self.worker = Some(worker);
                                self.ok("model loaded");
                            }
                            Err(error) => self.fail(error),
                        }
                    }
                }
                ui.label(if self.worker.is_some() {
                    "loaded"
                } else {
                    "no model loaded"
                });
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("predictions:");
                for (mode, name) in [(0u8, "off"), (1, "on"), (2, "auto")] {
                    if ui
                        .selectable_label(self.prediction_mode == mode, name)
                        .clicked()
                    {
                        self.run_action(Action::SetPredictions(mode));
                    }
                }
            });
            ui.add(egui::Slider::new(&mut self.threshold, 0.05..=0.95).text("threshold"));
            if ui.button("Detect").clicked()
                && self.worker.is_some()
            {
                match self.detect() {
                    Ok(count) => self.ok(format!("{count} prediction(s)")),
                    Err(error) => self.fail(error),
                }
            }
        });
        self.show_model = open;
    }

    fn new_project_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_new_project;
        egui::Window::new("New project")
            .open(&mut open)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.label("Folder (created empty):");
                ui.add(egui::TextEdit::singleline(&mut self.new_dir).desired_width(420.0));
                ui.label("Class names, one per line:");
                ui.add(
                    egui::TextEdit::multiline(&mut self.new_names)
                        .desired_width(420.0)
                        .desired_rows(8),
                );
                if ui.button("Create").clicked() {
                    let names: Vec<String> = self
                        .new_names
                        .lines()
                        .map(str::trim)
                        .filter(|line| !line.is_empty())
                        .map(str::to_owned)
                        .collect();
                    let dir = PathBuf::from(self.new_dir.clone());
                    match create_project(&dir, &names) {
                        Ok(_) => {
                            self.show_new_project = false;
                            self.open_project(&dir);
                        }
                        Err(error) => self.fail(error),
                    }
                }
            });
        self.show_new_project = open && self.show_new_project;
    }

    fn local_plan(&self) -> String {
        let Some(project) = &self.project else {
            return "no project".to_owned();
        };
        let edits: Vec<ClassEdit> = self
            .class_edit
            .iter()
            .enumerate()
            .filter(|(index, name)| project.classes.get(*index) != Some(*name))
            .map(|(index, name)| ClassEdit::Rename(index, name.clone()))
            .collect();
        match plan_remap(project, &edits) {
            Ok(plan) => format!(
                "{} -> {} classes; {} images touched",
                plan.old_names.len(),
                plan.new_names.len(),
                plan.files.len()
            ),
            Err(error) => error,
        }
    }

    fn apply_classes(&mut self) -> Result<String, String> {
        let Some(project) = &self.project else {
            return Err("no project".to_owned());
        };
        let edits: Vec<ClassEdit> = self
            .class_edit
            .iter()
            .enumerate()
            .filter(|(index, name)| project.classes.get(*index) != Some(*name))
            .map(|(index, name)| ClassEdit::Rename(index, name.clone()))
            .collect();
        if self.class_edit.len() != project.classes.len() {
            return Err("adding or removing classes is not wired in this build".to_owned());
        }
        let plan = plan_remap(project, &edits)?;
        let backup = project
            .root
            .join("backup")
            .join(format!("classes-{}", std::process::id()));
        apply_remap(project, &plan, &backup)?;
        let root = project.root.clone();
        self.open_project(&root);
        Ok(format!("classes applied; backup at {}", backup.display()))
    }

    // ---- launcher ---------------------------------------------------------

    fn refresh_launcher(&mut self) {
        self.launcher_stale = false;
        let Some(path) = self
            .launcher_selected
            .and_then(|index| self.settings.recent.get(index))
            .cloned()
        else {
            self.launcher_project = None;
            self.launcher_stats = None;
            self.launcher_files.clear();
            return;
        };
        match Project::load(&path) {
            Ok(project) => {
                self.launcher_stats = Some(project.stats());
                self.launcher_files = project_files(&project.root);
                self.launcher_project = Some(project);
            }
            Err(error) => {
                self.launcher_project = None;
                self.launcher_stats = None;
                self.launcher_files.clear();
                self.fail(error);
            }
        }
    }

    fn launcher_info(&self) -> Vec<(String, String)> {
        let mut rows = Vec::new();
        let Some(project) = &self.launcher_project else {
            return rows;
        };
        rows.push(("Name".to_owned(), project.name.clone()));
        rows.push((
            "Project directory".to_owned(),
            project.root.display().to_string(),
        ));
        if let Some(stats) = &self.launcher_stats {
            rows.push(("Image files".to_owned(), stats.images.to_string()));
            let percentage = if stats.images > 0 {
                stats.annotated as f64 * 100.0 / stats.images as f64
            } else {
                0.0
            };
            rows.push((
                "Markup files".to_owned(),
                format!(
                    "{} ({percentage:.1}%), {} negative sample(s)",
                    stats.annotated, stats.negative
                ),
            ));
            rows.push((
                "Number of classes".to_owned(),
                project.classes.len().to_string(),
            ));
            let per_image = if stats.annotated > 0 {
                stats.marks as f64 / stats.annotated as f64
            } else {
                0.0
            };
            rows.push((
                "Number of marks".to_owned(),
                format!("{} ({per_image:.2} per annotated image)", stats.marks),
            ));
            let per_class: Vec<String> = project
                .classes
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    format!("{name}: {}", stats.per_class.get(index).copied().unwrap_or(0))
                })
                .collect();
            rows.push(("Marks per class".to_owned(), per_class.join(", ")));
            rows.push((
                "Newest markup".to_owned(),
                stats
                    .newest_markup
                    .map(relative_age)
                    .unwrap_or_else(|| "-".to_owned()),
            ));
            rows.push((
                "Oldest markup".to_owned(),
                stats
                    .oldest_markup
                    .map(relative_age)
                    .unwrap_or_else(|| "-".to_owned()),
            ));
        }
        rows.push((
            "Configuration".to_owned(),
            project
                .cfg_file
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "(none)".to_owned()),
        ));
        rows.push((
            "Weights".to_owned(),
            project
                .weights_file
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "(none)".to_owned()),
        ));
        rows.push((
            "Classes file".to_owned(),
            classes_file(project).display().to_string(),
        ));
        rows
    }

    fn launcher(&mut self, ctx: &egui::Context) {
        if self.launcher_stale {
            self.refresh_launcher();
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("DarkMark");
            ui.horizontal_wrapped(|ui| {
                let recent = self.settings.recent.clone();
                for (index, entry) in recent.iter().enumerate() {
                    let label = Path::new(entry)
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| entry.clone());
                    if ui
                        .selectable_label(self.launcher_selected == Some(index), label)
                        .clicked()
                        && self.launcher_selected != Some(index)
                    {
                        self.launcher_selected = Some(index);
                        self.launcher_stale = true;
                    }
                }
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Project:");
                ui.add(egui::TextEdit::singleline(&mut self.new_dir).desired_width(520.0));
                if ui.button("Open").clicked() {
                    let path = PathBuf::from(self.new_dir.clone());
                    self.open_project(&path);
                }
                if ui.button("Open folder…").clicked()
                    && let Some(folder) = rfd::FileDialog::new().pick_folder()
                {
                    self.new_dir = folder.to_string_lossy().into_owned();
                    self.open_project(&folder);
                }
                if ui.button("New project…").clicked() {
                    self.show_new_project = true;
                }
            });
            ui.separator();
            let info = self.launcher_info();
            if info.is_empty() {
                ui.label("Pick a project tab, type a folder, or open one.");
            } else {
                egui::Grid::new("launcher-info")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        for (label, value) in info {
                            ui.label(label);
                            ui.label(value);
                            ui.end_row();
                        }
                    });
            }
            ui.separator();
            ui.heading("Files");
            ui.checkbox(
                &mut self.launcher_hide_extra_weights,
                "Hide extra .weights files",
            );
            let hidden = self.launcher_hide_extra_weights;
            let mut seen_weights = 0usize;
            egui::ScrollArea::vertical()
                .max_height(220.0)
                .show(ui, |ui| {
                    egui::Grid::new("launcher-files")
                        .num_columns(4)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("File");
                            ui.label("Type");
                            ui.label("Size");
                            ui.label("Modified");
                            ui.end_row();
                            for file in &self.launcher_files {
                                if hidden && file.kind == "weights" {
                                    seen_weights += 1;
                                    if seen_weights > 1 {
                                        continue;
                                    }
                                }
                                let newest = file.modified.is_some()
                                    && file.modified == self.launcher_files.first().and_then(|f| f.modified);
                                let name = if newest {
                                    egui::RichText::new(&file.name).color(Color32::LIGHT_GREEN)
                                } else {
                                    egui::RichText::new(&file.name)
                                };
                                ui.label(name);
                                ui.label(file.kind);
                                ui.label(format_size(file.size));
                                ui.label(
                                    file.modified
                                        .map(relative_age)
                                        .unwrap_or_else(|| "-".to_owned()),
                                );
                                ui.end_row();
                            }
                        });
                });
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Add…").clicked()
                    && let Some(folder) = rfd::FileDialog::new().pick_folder()
                {
                    self.settings.remember(&folder);
                    self.launcher_selected = Some(0);
                    self.launcher_stale = true;
                }
                if ui.button("Delete…").clicked()
                    && let Some(index) = self.launcher_selected
                {
                    self.settings.recent.remove(index);
                    self.settings.save();
                    self.launcher_selected = if self.settings.recent.is_empty() {
                        None
                    } else {
                        Some(0)
                    };
                    self.launcher_stale = true;
                }
                ui.add_enabled(false, egui::Button::new("Import video…"));
                if ui.button("Open folder…").clicked()
                    && let Some(folder) = rfd::FileDialog::new().pick_folder()
                {
                    self.new_dir = folder.to_string_lossy().into_owned();
                    self.open_project(&folder);
                }
                if ui.button("Refresh").clicked() {
                    self.launcher_stale = true;
                }
                if ui.button("Load…").clicked()
                    && let Some(path) = self
                        .launcher_selected
                        .and_then(|index| self.settings.recent.get(index))
                        .cloned()
                {
                    self.open_project(Path::new(&path));
                }
                if ui.button("Cancel").clicked() && self.project.is_some() {
                    self.show_launcher = false;
                }
            });
            ui.separator();
            let color = if self.error {
                Color32::LIGHT_RED
            } else {
                Color32::GRAY
            };
            ui.colored_label(color, &self.status);
        });
        if self.show_new_project {
            self.new_project_window(ctx);
        }
    }
}

fn prediction_label(mode: u8) -> &'static str {
    match mode {
        1 => "on",
        2 => "auto",
        _ => "off",
    }
}

fn tri_row(ui: &mut egui::Ui, label: &str, value: &mut Tri) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.selectable_value(value, Tri::Any, "any");
        ui.selectable_value(value, Tri::Yes, "yes");
        ui.selectable_value(value, Tri::No, "no");
    });
}

fn classes_file(project: &Project) -> PathBuf {
    let preferred = project.root.join(format!("{}.names", project.name));
    if preferred.is_file() {
        return preferred;
    }
    project
        .root
        .read_dir()
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.extension().and_then(|value| value.to_str()) == Some("names"))
        .unwrap_or_else(|| project.root.join("(no .names file)"))
}

fn project_files(root: &Path) -> Vec<LauncherFile> {
    let mut files = Vec::new();
    let Ok(entries) = std::fs::read_dir(root) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let kind = match extension.as_str() {
            "cfg" => "cfg",
            "weights" => "weights",
            "names" => "names",
            _ => continue,
        };
        let metadata = entry.metadata().ok();
        files.push(LauncherFile {
            name: path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            kind,
            size: metadata.as_ref().map(|value| value.len()).unwrap_or(0),
            modified: metadata.and_then(|value| value.modified().ok()),
        });
    }
    files.sort_by(|a, b| b.modified.cmp(&a.modified).then_with(|| a.name.cmp(&b.name)));
    files
}

fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn relative_age(time: SystemTime) -> String {
    let seconds = SystemTime::now()
        .duration_since(time)
        .map(|delta| delta.as_secs())
        .unwrap_or(0);
    match seconds {
        0..=59 => format!("{seconds}s ago"),
        60..=3599 => format!("{}m ago", seconds / 60),
        3600..=86399 => format!("{}h ago", seconds / 3600),
        _ => format!("{}d ago", seconds / 86400),
    }
}

fn apply_view(view: View, gray: &GrayImage, rgba: &RgbaImage) -> RgbaImage {
    match view {
        View::Normal => rgba.clone(),
        View::BlackWhite => {
            let mut out = RgbaImage::new(gray.width(), gray.height());
            for (x, y, pixel) in out.enumerate_pixels_mut() {
                let value = gray.get_pixel(x, y)[0];
                *pixel = image::Rgba([value, value, value, 255]);
            }
            out
        }
        View::Canny => {
            let edges = imageproc::edges::canny(gray, 50.0, 100.0);
            let mut out = rgba.clone();
            for (x, y, pixel) in out.enumerate_pixels_mut() {
                if edges.get_pixel(x, y)[0] > 0 {
                    *pixel = image::Rgba([255, 64, 64, 255]);
                } else {
                    let value = (gray.get_pixel(x, y)[0] as f32 * 0.35) as u8;
                    *pixel = image::Rgba([value, value, value, 255]);
                }
            }
            out
        }
        View::Dot => {
            let mut out = rgba.clone();
            for (x, y, pixel) in out.enumerate_pixels_mut() {
                let value = gray.get_pixel(x, y)[0];
                let radius = (value as u32 / 64).clamp(0, 3) as i64;
                let inside =
                    (x as i64 % 4 - 2).abs() <= radius && (y as i64 % 4 - 2).abs() <= radius;
                *pixel = if inside {
                    image::Rgba([255, 255, 255, 255])
                } else {
                    image::Rgba([0, 0, 0, 255])
                };
            }
            out
        }
        View::Bold => {
            let mut out = rgba.clone();
            for (x, y, pixel) in out.enumerate_pixels_mut() {
                let value = gray.get_pixel(x, y)[0];
                let boosted = if value > 127 { 255 } else { 0 };
                *pixel = image::Rgba([boosted, boosted, boosted, 255]);
            }
            out
        }
        View::Shade => {
            let mut out = rgba.clone();
            for (x, y, pixel) in out.enumerate_pixels_mut() {
                let value = gray.get_pixel(x, y)[0];
                let p = pixel.0;
                *pixel = image::Rgba([
                    (p[0] as f32 * value as f32 / 255.0) as u8,
                    (p[1] as f32 * value as f32 / 255.0) as u8,
                    (p[2] as f32 * value as f32 / 255.0) as u8,
                    255,
                ]);
            }
            out
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.show_launcher {
            self.launcher(ctx);
            return;
        }
        self.handle_keys(ctx);
        self.tick_pending_load(ctx);
        self.scrollfield_panel(ctx);
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::from_gray(18)))
            .show(ctx, |ui| self.canvas(ui));
        self.windows(ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if self.dirty {
            self.save();
        }
    }
}

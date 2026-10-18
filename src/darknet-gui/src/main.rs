//! Darknet annotation GUI (egui/eframe).
//!
//! Layout: a top bar (project, save state, zoom), a left panel (classes, filter,
//! image list), the canvas, and modal windows for the launcher, new project,
//! class editor and help. All file semantics live in `darknet-gui-project`; all
//! model inference lives in `darknet-gui-engine`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use darknet_gui_engine::{DetectWorker, Prediction};
use darknet_gui_project::classes::{ClassEdit, apply_remap, plan_remap};
use darknet_gui_project::create::create_project;
use darknet_gui_project::filter::{Filter, SortKey, Tri};
use darknet_gui_project::hotkeys::{Action, Key, KeyInput, map_key};
use darknet_gui_project::{Editor, Mark, Mods, Point, Project, actions, nav};
use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use image::{GrayImage, RgbaImage};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Darknet Annotator"),
        ..Default::default()
    };
    eframe::run_native(
        "Darknet Annotator",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
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

#[derive(Default, serde::Serialize, serde::Deserialize)]
struct Settings {
    recent: Vec<String>,
}

impl Settings {
    fn path() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        Some(PathBuf::from(home).join(".config/darknet-gui/settings.json"))
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

struct App {
    settings: Settings,
    project: Option<Project>,
    filter: Filter,
    filter_text: String,
    sort: SortKey,
    view: View,
    index: usize,
    editor: Option<Editor>,
    cache: ImageCache,
    zoom: f32,
    pan: Vec2,
    dirty: bool,
    status: String,
    error: bool,
    show_launcher: bool,
    show_new_project: bool,
    show_classes: bool,
    show_help: bool,
    new_dir: String,
    new_names: String,
    class_edit: Vec<String>,
    class_plan: Option<String>,
    cfg_path: String,
    weights_path: String,
    worker: Option<DetectWorker>,
    clipboard: Option<arboard::Clipboard>,
    /// 0 = off, 1 = on, 2 = auto (`p` cycles in that order, like DarkMark).
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
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals = egui::Visuals::dark();
        cc.egui_ctx.set_style(style);
        Self {
            settings: Settings::load(),
            project: None,
            filter: Filter::default(),
            filter_text: String::new(),
            sort: SortKey::Name,
            view: View::Normal,
            index: 0,
            editor: None,
            cache: ImageCache::default(),
            zoom: 1.0,
            pan: Vec2::ZERO,
            dirty: false,
            status: "Open a project to start annotating".to_owned(),
            error: false,
            show_launcher: true,
            show_new_project: false,
            show_classes: false,
            show_help: false,
            new_dir: String::new(),
            new_names: "person\ncar\n".to_owned(),
            class_edit: Vec::new(),
            class_plan: None,
            cfg_path: String::new(),
            weights_path: String::new(),
            worker: None,
            clipboard: arboard::Clipboard::new().ok(),
            prediction_mode: 2,
            label_mode: 1,
            bold: false,
            shade: false,
            snapping: true,
            random_order: None,
            show_marks: true,
            threshold: 0.25,
            drag_start: None,
            drag_now: None,
            pan_start: None,
        }
    }

    fn fail(&mut self, error: String) {
        self.status = error;
        self.error = true;
    }

    fn ok(&mut self, message: impl Into<String>) {
        self.status = message.into();
        self.error = false;
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
                self.project = Some(project);
                self.cache.clear();
                self.editor = None;
                self.index = 0;
                self.dirty = false;
                self.show_launcher = false;
                self.load_current();
            }
            Err(error) => self.fail(error),
        }
    }

    fn save(&mut self) {
        let (Some(project), Some(editor)) = (self.project.as_mut(), self.editor.as_ref()) else {
            return;
        };
        let annotation = editor.to_annotation();
        match project.save_annotation(self.index, &annotation) {
            Ok(()) => {
                self.dirty = false;
                self.ok(format!(
                    "saved {} box(es)",
                    annotation.real_marks().count()
                ));
            }
            Err(error) => self.fail(error),
        }
    }

    fn load_current(&mut self) {
        if self.dirty {
            self.save();
        }
        let Some(project) = &self.project else {
            return;
        };
        let Some(entry) = project.images.get(self.index) else {
            return;
        };
        let entry_path = entry.path.clone();
        match project.load_annotation(self.index) {
            Ok(annotation) => {
                self.editor = Some(Editor::from_annotation(&annotation, project.classes.clone()));
                self.dirty = false;
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
                self.ok(entry_path.file_name().unwrap_or_default().to_string_lossy().to_string());
            }
            Err(error) => self.fail(error),
        }
    }

    fn visible_indices(&self) -> Vec<usize> {
        let Some(project) = &self.project else {
            return Vec::new();
        };
        let mut filter = self.filter.clone();
        if !self.filter_text.trim().is_empty() {
            filter.include = regex::Regex::new(&format!("(?i){}", regex::escape(&self.filter_text))).ok();
        }
        let mut indices = filter.apply(project, self.sort);
        if let Some(order) = &self.random_order {
            let rank = |index: usize| order.iter().position(|value| *value == index).unwrap_or(usize::MAX);
            indices.sort_by_key(|index| rank(*index));
        }
        indices
    }

    fn step_image(&mut self, delta: isize) {
        let visible = self.visible_indices();
        if visible.is_empty() {
            return;
        }
        let position = visible.iter().position(|index| *index == self.index).unwrap_or(0);
        let next = (position as isize + delta).clamp(0, visible.len() as isize - 1) as usize;
        if visible[next] != self.index {
            self.index = visible[next];
            self.load_current();
        }
    }

    fn page_image(&mut self, dir: nav::Dir) {
        let Some(project) = &self.project else {
            return;
        };
        if let Some(next) = nav::find_unmarked(project, self.index, dir)
            && next != self.index
        {
            self.index = next;
            self.load_current();
        }
    }

    fn run_action(&mut self, action: Action) {
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
            Action::PrevImage => self.step_image(-1),
            Action::NextImage => self.step_image(1),
            Action::FirstImage => {
                self.index = 0;
                self.load_current();
            }
            Action::LastImage => {
                if let Some(project) = &self.project {
                    self.index = project.images.len().saturating_sub(1);
                }
                self.load_current();
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
                editor.set_empty_flag(flagged);
                self.dirty = true;
            }
            Action::CopyPrev => self.copy_neighbour(nav::Dir::Previous),
            Action::CopyNext => self.copy_neighbour(nav::Dir::Next),
            Action::ToggleMarks => self.show_marks = !self.show_marks,
            Action::TogglePredictions => self.prediction_mode = (self.prediction_mode + 1) % 3,
            Action::ToggleLabels => self.label_mode = (self.label_mode + 1) % 3,
            Action::ToggleBold => self.bold = !self.bold,
            Action::ToggleShade => self.shade = !self.shade,
            Action::ToggleSnapping => self.snapping = !self.snapping,
            Action::AcceptAll => {
                if editor.accept_all() {
                    self.dirty = true;
                }
            }
            Action::AcceptSelected => {
                if editor.accept_selected() {
                    self.dirty = true;
                }
            }
            Action::Scale(delta) => {
                editor.scale(delta);
                self.dirty = true;
            }
            Action::Nudge(dx, dy) => {
                editor.nudge(dx, dy);
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
            Action::Save => self.save(),
            Action::ClassMenu => self.show_classes = true,
            Action::SnapAll => {
                let Some(image) = self.cache.entries.get(&self.index).map(|entry| entry.gray.clone()) else {
                    return;
                };
                let mut annotation = editor.to_annotation();
                let moved = actions::snap(
                    &mut annotation,
                    &image,
                    6,
                    actions::SnapScope::All,
                );
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
            Action::ZoomIn => self.zoom = (self.zoom * 1.1).min(16.0),
            Action::ZoomOut => self.zoom = (self.zoom / 1.1).max(0.1),
            Action::ZoomToggle => {
                if (self.zoom - 1.0).abs() < f32::EPSILON {
                    self.zoom = 2.0;
                } else {
                    self.zoom = 1.0;
                    self.pan = Vec2::ZERO;
                }
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
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
                if let Some(editor) = self.editor.as_mut() {
                    editor.selected = None;
                }
            }
            Action::Help => self.show_help = true,
            Action::DeleteImage
            | Action::UndoDeleteImage
            | Action::MassDelete
            | Action::Homography
            | Action::Merge
            | Action::HeatmapCycle
            | Action::ToggleHeatmap
            | Action::Jump
            | Action::Screenshot(_)
            | Action::WindowScreenshot
            | Action::ToggleTiling
            | Action::TileEdge(_)
            | Action::TileRect(_) => {
                self.status = format!("{action:?} is mapped but not implemented in this build");
                self.error = false;
            }
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

    fn detect(&mut self) {
        let (Some(project), Some(editor)) = (&self.project, self.editor.as_mut()) else {
            return;
        };
        if self.prediction_mode == 0 {
            return;
        }
        let Some(worker) = self.worker.as_mut() else {
            return;
        };
        let Some(entry) = project.images.get(self.index) else {
            return;
        };
        match worker.detect(&entry.path, self.threshold) {
            Ok(predictions) => {
                let marks: Vec<Mark> = predictions
                    .into_iter()
                    .map(|Prediction { class, conf, cx, cy, w, h }| {
                        let (width, height) = (editor.width as f32, editor.height as f32);
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
                self.ok("predictions loaded");
            }
            Err(error) => self.fail(error),
        }
    }

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let mut actions = Vec::new();
        ctx.input(|input| {
            let mods = Mods {
                ctrl: input.modifiers.command,
                alt: input.modifiers.alt,
                shift: input.modifiers.shift,
            };
            let mut push = |key: Key| {
                if input.key_pressed(egui::Key::Backspace) && key == Key::Backspace {
                    // handled below by the generic path
                }
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
                    let ch = if input.modifiers.shift { ch.to_ascii_uppercase() } else { ch };
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
        if let Err(error) = self.cache.get(ui.ctx(), self.index, &path) {
            self.fail(error);
            return;
        }
        let Some(texture) = self.cache.entries.get(&self.index) else {
            return;
        };
        let (width, height, view_image) = (
            texture.rgba.width() as f32,
            texture.rgba.height() as f32,
            match self.view {
                View::Normal => texture.handle.clone(),
                other => {
                    let filtered = apply_view(other, &texture.gray, &texture.rgba);
                    let size = [filtered.width() as usize, filtered.height() as usize];
                    let color = egui::ColorImage::from_rgba_unmultiplied(size, filtered.as_raw());
                    ui.ctx().load_texture(
                        format!("view-{}-{}", other.label(), self.index),
                        color,
                        egui::TextureOptions::LINEAR,
                    )
                }
            },
        );

        let available = ui.available_size();
        let response = ui.allocate_response(available, Sense::click_and_drag());
        let rect = response.rect;
        let painter = ui.painter_at(rect);
        let scale = ((rect.width() / width).min(rect.height() / height) * self.zoom).max(0.01);
        let image_rect = Rect::from_center_size(rect.center() + self.pan, Vec2::new(width, height) * scale);
        painter.image(
            view_image.id(),
            image_rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );

        let to_screen = |p: Point| {
            Pos2::new(
                image_rect.left() + p.x / width * image_rect.width(),
                image_rect.top() + p.y / height * image_rect.height(),
            )
        };
        let to_image = |p: Pos2| {
            Point::new(
                (p.x - image_rect.left()) / image_rect.width().max(1.0) * width,
                (p.y - image_rect.top()) / image_rect.height().max(1.0) * height,
            )
        };

        if let Some(editor) = &self.editor {
            for mark in &editor.marks {
                let predictions_visible = match self.prediction_mode {
                    1 => true,
                    2 => !editor.has_real_marks(),
                    _ => false,
                };
                if mark.is_prediction && !predictions_visible {
                    continue;
                }
                if !mark.is_prediction && !self.show_marks {
                    continue;
                }
                let color = if mark.is_prediction {
                    Color32::from_rgb(0xE8, 0xC8, 0x4E)
                } else {
                    class_color(mark.class_id)
                };
                let selected = editor
                    .selected
                    .and_then(|index| editor.marks.get(index))
                    .is_some_and(|selected| std::ptr::eq(selected, mark));
                let width = if selected || self.bold { 3.0_f32 } else { 2.0_f32 };
                let stroke = Stroke::new(width, color);
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
                    painter.text(
                        corners[0] + Vec2::new(2.0, -2.0),
                        egui::Align2::LEFT_BOTTOM,
                        text,
                        egui::FontId::proportional(13.0),
                        color,
                    );
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

        let mods = ui.input(|input| Mods {
            ctrl: input.modifiers.command,
            alt: input.modifiers.alt,
            shift: input.modifiers.shift,
        });
        if response.drag_started() {
            if let Some(pos) = response.interact_pointer_pos() {
                if mods.ctrl {
                    self.pan_start = Some((self.pan, pos));
                } else {
                    self.drag_start = Some(pos);
                    self.drag_now = Some(pos);
                    if let Some(editor) = self.editor.as_mut() {
                        editor.press(to_image(pos), mods, false);
                    }
                }
            }
        }
        if response.dragged() {
            if let (Some((pan, start)), Some(pos)) = (self.pan_start, response.interact_pointer_pos()) {
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
            } else if let (Some(_start), Some(pos)) = (self.drag_start.take(), self.drag_now.take()) {
                if let Some(editor) = self.editor.as_mut() {
                    editor.release(to_image(pos));
                    self.dirty = true;
                }
            }
        } else if response.clicked()
            && let Some(pos) = response.interact_pointer_pos()
            && let Some(editor) = self.editor.as_mut()
        {
            let point = to_image(pos);
            editor.press(point, mods, false);
            editor.drag(point);
            editor.release(point);
        }
        if response.double_clicked()
            && let Some(pos) = response.interact_pointer_pos()
            && let Some(editor) = self.editor.as_mut()
        {
            editor.press(to_image(pos), mods, true);
            self.dirty = true;
        }
        let scroll = ui.input(|input| input.raw_scroll_delta.y);
        if response.hovered() && scroll != 0.0 {
            self.zoom = (self.zoom * (1.0 + scroll * 0.001)).clamp(0.1, 16.0);
        }
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
                let inside = (x as i64 % 4 - 2).abs() <= radius && (y as i64 % 4 - 2).abs() <= radius;
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

fn class_color(class_id: usize) -> Color32 {
    const PALETTE: [Color32; 10] = [
        Color32::from_rgb(0x4E, 0x9A, 0xE8),
        Color32::from_rgb(0xE8, 0x6A, 0x4E),
        Color32::from_rgb(0x5E, 0xC8, 0x7A),
        Color32::from_rgb(0xE8, 0xC8, 0x4E),
        Color32::from_rgb(0xB8, 0x6A, 0xE8),
        Color32::from_rgb(0x4E, 0xE8, 0xD8),
        Color32::from_rgb(0xE8, 0x4E, 0x9A),
        Color32::from_rgb(0x9A, 0xE8, 0x4E),
        Color32::from_rgb(0xD8, 0x8E, 0x4E),
        Color32::from_rgb(0x8E, 0x9A, 0xE8),
    ];
    PALETTE[class_id % PALETTE.len()]
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.show_launcher {
            self.launcher(ctx);
            return;
        }
        self.handle_keys(ctx);
        self.top_bar(ctx);
        self.side_panel(ctx);
        egui::CentralPanel::default().show(ctx, |ui| self.canvas(ui));
        self.modals(ctx);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if self.dirty {
            self.save();
        }
    }
}

impl App {
    fn launcher(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Darknet Annotator");
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Project folder:");
                ui.add(egui::TextEdit::singleline(&mut self.new_dir).desired_width(460.0));
                if ui.button("Open").clicked() {
                    let path = PathBuf::from(self.new_dir.clone());
                    self.open_project(&path);
                }
                if ui.button("New project…").clicked() {
                    self.show_new_project = true;
                }
            });
            ui.separator();
            ui.heading("Recent");
            let recent = self.settings.recent.clone();
            for entry in recent {
                if ui.selectable_label(false, &entry).clicked() {
                    self.open_project(Path::new(&entry));
                }
            }
            if !self.status.is_empty() {
                ui.separator();
                let color = if self.error { Color32::LIGHT_RED } else { Color32::GRAY };
                ui.colored_label(color, &self.status);
            }
        });
        if self.show_new_project {
            self.new_project_window(ctx);
        }
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
                ui.add(egui::TextEdit::multiline(&mut self.new_names).desired_width(420.0).desired_rows(8));
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

    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Projects").clicked() {
                    self.show_launcher = true;
                }
                if ui.button("Save").clicked() {
                    self.save();
                }
                if ui.button("Undo").clicked() {
                    self.run_action(Action::Undo);
                }
                if self.dirty {
                    ui.colored_label(Color32::YELLOW, "● unsaved");
                }
                ui.separator();
                if let Some(project) = &self.project {
                    ui.label(&project.name);
                    ui.separator();
                    ui.label(format!("{}/{}", self.index + 1, project.images.len()));
                }
                ui.separator();
                if ui.button("−").clicked() {
                    self.zoom = (self.zoom / 1.1).max(0.1);
                }
                ui.label(format!("{:.0}%", self.zoom * 100.0));
                if ui.button("+").clicked() {
                    self.zoom = (self.zoom * 1.1).min(16.0);
                }
                egui::ComboBox::from_id_salt("view")
                    .selected_text(self.view.label())
                    .show_ui(ui, |ui| {
                        for view in [View::Normal, View::BlackWhite, View::Canny, View::Dot, View::Bold, View::Shade] {
                            ui.selectable_value(&mut self.view, view, view.label());
                        }
                    });
                let labels = match self.label_mode {
                    1 => "labels: on",
                    2 => "labels: auto",
                    _ => "labels: off",
                };
                if ui.button(labels).clicked() {
                    self.run_action(Action::ToggleLabels);
                }
            });
            let color = if self.error { Color32::LIGHT_RED } else { Color32::GRAY };
            ui.colored_label(color, &self.status);
        });
    }

    fn side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("side").min_width(240.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Filter:");
                ui.add(egui::TextEdit::singleline(&mut self.filter_text).desired_width(140.0));
                if ui.button("sort").clicked() {
                    self.sort = if self.sort == SortKey::Name { SortKey::MarkCount } else { SortKey::Name };
                }
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.filter.unannotated, Tri::Any, "all");
                ui.selectable_value(&mut self.filter.unannotated, Tri::Yes, "unannotated");
                ui.selectable_value(&mut self.filter.unannotated, Tri::No, "annotated");
                ui.selectable_value(&mut self.filter.empty, Tri::Yes, "empty");
            });
            ui.separator();
            let (names, current, real_count, prediction_count) = match &self.editor {
                Some(editor) => (
                    (0..editor.class_count)
                        .map(|id| editor.class_name(id))
                        .collect::<Vec<_>>(),
                    editor.class_id,
                    editor.real_marks().count(),
                    editor.marks.iter().filter(|mark| mark.is_prediction).count(),
                ),
                None => (Vec::new(), 0, 0, 0),
            };
            ui.heading("Classes");
            let mut picked = None;
            for (id, name) in names.iter().enumerate() {
                if ui
                    .selectable_label(current == id, format!("{id}: {name}"))
                    .clicked()
                {
                    picked = Some(id);
                }
            }
            if let Some(id) = picked
                && let Some(editor) = self.editor.as_mut()
            {
                editor.class_id = id;
            }
            if ui.button("Edit classes…").clicked() {
                self.class_edit = names;
                self.class_plan = None;
                self.show_classes = true;
            }
            ui.separator();
            ui.label(format!("{real_count} real, {prediction_count} predictions"));
            ui.checkbox(&mut self.dirty, "unsaved flag");
            ui.separator();
            ui.heading("Images");
            egui::ScrollArea::vertical().show(ui, |ui| {
                let visible = self.visible_indices();
                let names: Vec<(usize, String, usize)> = self
                    .project
                    .as_ref()
                    .map(|project| {
                        visible
                            .iter()
                            .map(|index| {
                                (
                                    *index,
                                    project.images[*index].relative.clone(),
                                    project.mark_count(*index),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                for (index, name, marks) in names {
                    let label = if marks > 0 { format!("{name}  [{marks}]") } else { name };
                    if ui.selectable_label(self.index == index, label).clicked() && self.index != index {
                        self.index = index;
                        self.load_current();
                    }
                }
            });
        });
    }

    fn modals(&mut self, ctx: &egui::Context) {
        if self.show_help {
            let mut open = true;
            egui::Window::new("Hotkeys").open(&mut open).show(ctx, |ui| {
                ui.label("digits class; Ctrl/Alt +10/+20");
                ui.label("arrows / Home / End / PageUp / PageDown navigate");
                ui.label("Tab cycle; Del delete; C clear; n empty flag");
                ui.label("a accept all; A accept selected; o drop predictions; p toggle predictions");
                ui.label("y / Y copy from previous / next; Ctrl+WASD nudge; , . scale");
                ui.label("Ctrl+Z undo; Ctrl+Shift+S save; Ctrl+C/V clipboard");
                ui.label("Space reset zoom; + / - zoom; F1 help");
            });
            self.show_help = open;
        }
        if self.show_classes {
            self.class_window(ctx);
        }
        if self.worker.is_none() {
            egui::TopBottomPanel::bottom("model").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("cfg:");
                    ui.add(egui::TextEdit::singleline(&mut self.cfg_path).desired_width(240.0));
                    ui.label("weights:");
                    ui.add(egui::TextEdit::singleline(&mut self.weights_path).desired_width(240.0));
                    if ui.button("Load model").clicked() {
                        // `cargo run -p darknet-gui` puts the GUI binary here, not
                        // the CLI, so prefer an explicit path.
                        let bin = std::env::var_os("DARKNET_RS_BIN")
                            .map(PathBuf::from)
                            .or_else(|| std::env::current_exe().ok())
                            .unwrap_or_else(|| PathBuf::from("darknet-rs"));
                        if bin.file_name().and_then(|n| n.to_str()) == Some("darknet-gui") {
                            self.fail(
                                "set DARKNET_RS_BIN to the darknet-rs binary (cargo build -p darknet-cli)"
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
                });
            });
        } else {
            egui::TopBottomPanel::bottom("model").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let label = match self.prediction_mode {
                        1 => "predictions: on",
                        2 => "predictions: auto",
                        _ => "predictions: off",
                    };
                    if ui.button(label).clicked() {
                        self.run_action(Action::TogglePredictions);
                    }
                    ui.label(format!("threshold {:.2}", self.threshold));
                    if ui.button("Detect").clicked() {
                        self.detect();
                    }
                });
            });
        }
    }

    fn class_window(&mut self, ctx: &egui::Context) {
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
}

//! DarkMark's right-edge scrollfield.
//!
//! One row per image (the first image at the top), one column per class, drawn
//! as a texture and resampled to the window height. Transcribed from
//! `ScrollField.cpp`:
//!
//! * row base colours: black = a `.json` with marks, grey (32,32,32) = no
//!   `.json`, red = a JSON parse error or a `.json` with no marks that is not
//!   flagged `completely_empty` (`:180-206`);
//! * a mark paints the horizontal segment of its class column in the class
//!   colour, `line_width = width / number_of_classes` (`:92-93`, `:212-219`);
//! * a `completely_empty` image gets a fake mark in the `* empty image *` class
//!   colour (`:195-199`);
//! * a full-width white line marks the current image (`:435-446`);
//! * white triangles on the left edge mark the first image of each directory,
//!   with the directory name on hover, only while the sort is alphabetical
//!   (`:108-140`, `:284-321`).
//!
//! Left-click and drag jump with a quick load; the full load happens on release
//! (main.rs owns the two-stage load, `CrosshairComponent.cpp:310-355`). The rows
//! are built on a worker thread from the sidecars on disk (`:96-106`), so a
//! large project never blocks the UI.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use eframe::egui::{
    self, Color32, ColorImage, Pos2, Rect, Response, Sense, Stroke, TextureHandle,
    TextureOptions, Ui, Vec2,
};
use lightnet_gui_project::{Annotation, Project};

use crate::colors::{ROW_BLACK, ROW_GREY, ROW_RED, class_color};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowState {
    /// A `.json` sidecar with marks.
    #[default]
    Black,
    /// No `.json` sidecar.
    Grey,
    /// Parse error, or a sidecar with no marks that is not flagged empty.
    Red,
}

#[derive(Clone, Debug, Default)]
pub struct Row {
    pub state: RowState,
    /// Classes of the real marks, in sidecar order. A `completely_empty` image
    /// carries one entry with the `* empty image *` class index instead.
    pub classes: Vec<usize>,
    /// Directory name when this row starts a new directory.
    pub directory: Option<String>,
}

/// What the user did on the strip; positions are indices into the visible list.
#[derive(Debug)]
pub struct ScrollInteraction {
    pub response: Response,
    pub quick: Option<usize>,
    pub full: Option<usize>,
}

pub struct ScrollField {
    /// Strip width in points; 0 disables the strip (main.rs skips it).
    pub width: f32,
    /// Directory triangle size (`scrollfield_marker_size`, default 7).
    pub marker_size: f32,
    rows: Vec<Row>,
    /// The visible list the rows were built for.
    visible: Vec<usize>,
    class_count: usize,
    texture: Option<TextureHandle>,
    texture_width: usize,
    dirty_texture: bool,
    loader: Option<Receiver<Vec<Row>>>,
}

impl ScrollField {
    pub fn new(width: f32, marker_size: f32) -> Self {
        Self {
            width,
            marker_size,
            rows: Vec::new(),
            visible: Vec::new(),
            class_count: 0,
            texture: None,
            texture_width: 0,
            dirty_texture: true,
            loader: None,
        }
    }

    /// Starts a rebuild when the project or the visible list changed, and picks
    /// up a finished worker (`ScrollField.cpp:96-106`).
    pub fn sync(&mut self, ctx: &egui::Context, project: &Project, visible: &[usize]) {
        if let Some(loader) = &self.loader
            && let Ok(rows) = loader.try_recv()
        {
            self.rows = rows;
            self.loader = None;
            self.dirty_texture = true;
        }
        let class_count = project.classes.len();
        let stale = self.rows.len() != visible.len()
            || self.visible != visible
            || self.class_count != class_count;
        if stale && self.loader.is_none() {
            let entries: Vec<(PathBuf, String)> = visible
                .iter()
                .map(|index| {
                    let entry = &project.images[*index];
                    (entry.json_path.clone(), entry.relative.clone())
                })
                .collect();
            self.visible = visible.to_vec();
            self.class_count = class_count;
            self.loader = Some(spawn_builder(entries, class_count));
            ctx.request_repaint();
        }
        // Keep the strip texture in step with the configured width (the settings
        // slider can change it at any time).
        if self.width >= 1.0 && self.width.round() as usize != self.texture_width {
            self.dirty_texture = true;
        }
        if self.dirty_texture && self.loader.is_none() {
            self.rebuild_texture(ctx);
            self.dirty_texture = false;
        }
    }

    /// Rewrites one row after a save (`DMContent.cpp:3024-3027`).
    pub fn refresh_row(&mut self, position: usize, project: &Project, index: usize) {
        let Some(entry) = project.images.get(index) else {
            return;
        };
        if position >= self.rows.len() {
            return;
        }
        let directory = self.rows[position].directory.clone();
        self.rows[position] = build_row(&entry.json_path, self.class_count);
        self.rows[position].directory = directory;
        self.dirty_texture = true;
    }

    /// Number of class columns; the trailing one is the `* empty image *` class.
    fn columns(&self) -> usize {
        self.class_count + 1
    }

    fn rebuild_texture(&mut self, ctx: &egui::Context) {
        if self.rows.is_empty() || self.width < 1.0 {
            self.texture = None;
            self.texture_width = 0;
            return;
        }
        let width = self.width.round().max(1.0) as usize;
        let height = self.rows.len();
        let columns = self.columns().max(1);
        let line_width = width as f32 / columns as f32;
        let mut image = ColorImage::filled([width, height], ROW_BLACK);
        for (y, row) in self.rows.iter().enumerate() {
            let base = match row.state {
                RowState::Black => ROW_BLACK,
                RowState::Grey => ROW_GREY,
                RowState::Red => ROW_RED,
            };
            for x in 0..width {
                image[(x, y)] = base;
            }
            if row.state != RowState::Black {
                continue;
            }
            for class in &row.classes {
                let column = (*class).min(columns - 1);
                let start = (column as f32 * line_width).round() as usize;
                let end = (((column + 1) as f32 * line_width).round() as usize).min(width);
                for x in start..end.max(start).min(width) {
                    image[(x, y)] = class_color(*class);
                }
            }
        }
        match &mut self.texture {
            Some(texture) => texture.set(image, TextureOptions::LINEAR),
            None => {
                self.texture =
                    Some(ctx.load_texture("scrollfield", image, TextureOptions::LINEAR));
            }
        }
        self.texture_width = width;
    }

    /// Draws the strip. `current` is the position of the current image in the
    /// visible list; `alphabetical` gates the directory triangles.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        current: Option<usize>,
        alphabetical: bool,
    ) -> ScrollInteraction {
        let width = self.width.max(1.0);
        let (rect, response) = ui.allocate_exact_size(
            Vec2::new(width, ui.available_height().max(1.0)),
            Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, ROW_GREY);
        if let Some(texture) = &self.texture {
            painter.image(
                texture.id(),
                rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }
        if self.rows.is_empty() {
            return ScrollInteraction {
                response,
                quick: None,
                full: None,
            };
        }
        let count = self.rows.len();
        let row_height = rect.height() / count as f32;

        // Current image: a full-width white line (`:435-446`).
        if let Some(position) = current
            && position < count
        {
            let y = rect.top() + (position as f32 + 0.5) * row_height;
            painter.line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                Stroke::new(1.0_f32, Color32::WHITE),
            );
        }

        // Directory triangles (`:108-140`), alphabetical sort only.
        let triangle = self.marker_size.max(3.0).min(row_height.max(3.0));
        let mut hovered_directory: Option<String> = None;
        if alphabetical {
            let pointer = response.hover_pos();
            for (position, row) in self.rows.iter().enumerate() {
                let Some(directory) = &row.directory else {
                    continue;
                };
                let y = rect.top() + position as f32 * row_height;
                let points = vec![
                    Pos2::new(rect.left(), y),
                    Pos2::new(rect.left() + triangle, y + triangle * 0.5),
                    Pos2::new(rect.left(), y + triangle),
                ];
                painter.add(egui::Shape::convex_polygon(
                    points,
                    Color32::WHITE,
                    Stroke::NONE,
                ));
                if let Some(pointer) = pointer
                    && pointer.x <= rect.left() + triangle
                    && pointer.y >= y
                    && pointer.y < y + triangle
                {
                    hovered_directory = Some(directory.clone());
                }
            }
        }
        if let Some(directory) = hovered_directory {
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                ui.layer_id(),
                egui::Id::new("scrollfield-directory"),
                egui::PopupAnchor::Pointer,
            )
            .show(|ui| {
                ui.label(directory);
            });
        }

        // Row under the pointer.
        let position_at = |y: f32| -> Option<usize> {
            let relative = (y - rect.top()) / rect.height().max(1.0);
            if !(0.0..1.0).contains(&relative) {
                return None;
            }
            Some(((relative * count as f32) as usize).min(count - 1))
        };
        let pointer = response.interact_pointer_pos();
        let mut quick = None;
        let mut full = None;
        if (response.drag_started() || response.dragged())
            && let Some(position) = pointer.and_then(|p| position_at(p.y))
        {
            quick = Some(position);
        }
        if response.drag_stopped()
            && let Some(position) = pointer.and_then(|p| position_at(p.y))
        {
            full = Some(position);
        }
        if response.clicked()
            && let Some(position) = pointer.and_then(|p| position_at(p.y))
        {
            full = Some(position);
        }
        ScrollInteraction {
            response,
            quick,
            full,
        }
    }
}

/// Builds every row for `entries` in a worker thread.
fn spawn_builder(entries: Vec<(PathBuf, String)>, class_count: usize) -> Receiver<Vec<Row>> {
    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        let mut rows = Vec::with_capacity(entries.len());
        let mut previous: Option<String> = None;
        for (json_path, relative) in entries {
            let mut row = build_row(&json_path, class_count);
            let directory = relative_directory(&relative);
            if previous.as_deref() != Some(directory.as_str()) {
                row.directory = Some(directory.clone());
            }
            previous = Some(directory);
            rows.push(row);
        }
        let _ = sender.send(rows);
    });
    receiver
}

fn relative_directory(relative: &str) -> String {
    match relative.rfind('/') {
        Some(index) => relative[..index].to_owned(),
        None => String::new(),
    }
}

/// Reads one sidecar and turns it into a row (`ScrollField.cpp:180-206`).
fn build_row(json_path: &Path, class_count: usize) -> Row {
    let empty_class = class_count;
    match std::fs::read_to_string(json_path) {
        Ok(text) => match Annotation::from_json(&text) {
            Ok(annotation) => {
                if annotation.completely_empty {
                    return Row {
                        state: RowState::Black,
                        classes: vec![empty_class],
                        directory: None,
                    };
                }
                let classes: Vec<usize> = annotation
                    .real_marks()
                    .map(|mark| mark.class_id)
                    .collect();
                if classes.is_empty() {
                    Row {
                        state: RowState::Red,
                        classes,
                        directory: None,
                    }
                } else {
                    Row {
                        state: RowState::Black,
                        classes,
                        directory: None,
                    }
                }
            }
            Err(_) => Row {
                state: RowState::Red,
                classes: Vec::new(),
                directory: None,
            },
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Row {
            state: RowState::Grey,
            classes: Vec::new(),
            directory: None,
        },
        Err(_) => Row {
            state: RowState::Red,
            classes: Vec::new(),
            directory: None,
        },
    }
}

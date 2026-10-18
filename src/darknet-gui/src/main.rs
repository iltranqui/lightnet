//! Darknet annotation GUI.
//!
//! One window, one project directory. Draw detection boxes over an image, press
//! a digit to pick the class, and the YOLO `<stem>.txt` sidecar is written next
//! to the image. An empty image deletes its sidecar instead of writing a blank
//! file, so "no marks" and "not annotated yet" stay distinguishable.
//!
//! Keys: digits select the class, arrows move between images, `Delete` removes
//! the selected box, `Ctrl+Z` undoes, `Ctrl+S` saves. Dragging on the canvas
//! creates a box; clicking one selects it.

mod annotation;
mod project;

use annotation::{Annotation, Mark};
use eframe::egui;
use egui::{Color32, Pos2, Rect, Sense, Stroke, Vec2};
use project::{ImageEntry, Project};

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
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

#[derive(Default)]
struct App {
    folder: String,
    project: Option<Project>,
    index: usize,
    texture: Option<egui::TextureHandle>,
    image_size: Vec2,
    annotation: Annotation,
    undo: Vec<Annotation>,
    selected: Option<usize>,
    class_id: usize,
    dirty: bool,
    status: String,
    error: bool,
    drag_start: Option<Pos2>,
    drag_now: Option<Pos2>,
    /// Decoded image waiting for the first draw, when a texture can be made.
    pending_image: Option<egui::ColorImage>,
}

impl App {
    fn open_project(&mut self) {
        match Project::open(&self.folder) {
            Ok(project) => {
                self.project = Some(project);
                self.class_id = 0;
                self.index = 0;
                self.status = "project loaded".to_owned();
                self.error = false;
                self.load_current();
            }
            Err(error) => self.fail(error),
        }
    }

    fn fail(&mut self, error: String) {
        self.status = error;
        self.error = true;
    }

    fn entry(&self) -> Option<&ImageEntry> {
        self.project.as_ref()?.images.get(self.index)
    }

    fn classes(&self) -> usize {
        self.project.as_ref().map(Project::class_count).unwrap_or(1)
    }

    /// Saves the current annotation when it is dirty, then loads the image.
    fn load_current(&mut self) {
        if self.dirty {
            self.save();
        }
        let Some(entry) = self.entry().cloned() else {
            return;
        };
        let classes = self.classes();
        match project::load_annotation(&entry, classes) {
            Ok(annotation) => {
                self.annotation = annotation;
                self.undo.clear();
                self.selected = None;
                self.dirty = false;
                if let Some(project) = &mut self.project {
                    project.images[self.index].marks = Some(self.annotation.marks.len());
                }
            }
            Err(error) => self.fail(error),
        }
        match image::open(&entry.path) {
            Ok(image) => {
                let rgba = image.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color = egui::ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                self.image_size = Vec2::new(size[0] as f32, size[1] as f32);
                // The texture is (re)created through the context on first draw.
                self.texture = None;
                self.pending_image = Some(color);
                self.status = format!(
                    "{} — {}x{}",
                    entry.path.file_name().unwrap_or_default().to_string_lossy(),
                    size[0],
                    size[1]
                );
                self.error = false;
            }
            Err(error) => self.fail(format!("{}: {error}", entry.path.display())),
        }
    }

    fn save(&mut self) {
        let Some(entry) = self.entry().cloned() else {
            return;
        };
        match project::save_annotation(&entry, &self.annotation) {
            Ok(exists) => {
                if let Some(project) = &mut self.project {
                    project.images[self.index].marks =
                        if exists { Some(self.annotation.marks.len()) } else { None };
                }
                self.dirty = false;
                self.status = if exists {
                    format!("saved {} box(es) to {}", self.annotation.marks.len(), entry.label_path.display())
                } else {
                    "saved: no boxes, label file removed".to_owned()
                };
                self.error = false;
            }
            Err(error) => self.fail(error),
        }
    }

    fn step(&mut self, delta: isize) {
        let Some(project) = &self.project else {
            return;
        };
        let len = project.images.len();
        if len == 0 {
            return;
        }
        let next = (self.index as isize + delta).rem_euclid(len as isize) as usize;
        if next != self.index {
            self.index = next;
            self.load_current();
        }
    }

    fn push_undo(&mut self) {
        self.undo.push(self.annotation.clone());
        if self.undo.len() > 100 {
            self.undo.remove(0);
        }
        self.dirty = true;
    }

    fn undo(&mut self) {
        if let Some(previous) = self.undo.pop() {
            self.annotation = previous;
            self.selected = None;
            self.dirty = true;
        }
    }

    fn delete_selected(&mut self) {
        if let Some(index) = self.selected.take() {
            if index < self.annotation.marks.len() {
                self.push_undo();
                self.annotation.marks.remove(index);
            }
        }
    }

    fn add_box(&mut self, a: Pos2, b: Pos2, image_rect: Rect) {
        if image_rect.width() <= 0.0 || image_rect.height() <= 0.0 {
            return;
        }
        let norm = |p: Pos2| {
            (
                ((p.x - image_rect.left()) / image_rect.width()).clamp(0.0, 1.0),
                ((p.y - image_rect.top()) / image_rect.height()).clamp(0.0, 1.0),
            )
        };
        let (x0, y0) = norm(a);
        let (x1, y1) = norm(b);
        if let Some(mark) = Mark::from_corners(self.class_id, x0, y0, x1, y1) {
            self.push_undo();
            self.annotation.marks.push(mark);
            self.selected = Some(self.annotation.marks.len() - 1);
        }
    }
}

/// The decoded image is stashed here between the load and the first draw, when
/// a texture can finally be created from the context.
impl App {
    fn ensure_texture(&mut self, ctx: &egui::Context) {
        if self.texture.is_none() {
            if let Some(image) = self.pending_image.take() {
                self.texture = Some(ctx.load_texture("image", image, egui::TextureOptions::LINEAR));
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.handle_keys(ctx);
        self.top_panel(ctx);
        self.side_panel(ctx);
        self.canvas(ctx);
    }
}

impl App {
    fn handle_keys(&mut self, ctx: &egui::Context) {
        ctx.input(|input| {
            let command = input.modifiers.command;
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
                if input.key_pressed(key) && digit < self.classes() {
                    self.class_id = digit;
                }
            }
            if input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace) {
                self.delete_selected();
            }
            if input.key_pressed(egui::Key::ArrowRight) && !command {
                self.step(1);
            }
            if input.key_pressed(egui::Key::ArrowLeft) && !command {
                self.step(-1);
            }
            if command && input.key_pressed(egui::Key::Z) {
                self.undo();
            }
            if command && input.key_pressed(egui::Key::S) {
                self.save();
            }
        });
    }

    fn top_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Project:");
                ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width(420.0));
                if ui.button("Open").clicked() {
                    self.open_project();
                }
                if ui.button("Save").clicked() {
                    self.save();
                }
                if ui.button("Reload").clicked() {
                    self.dirty = false;
                    self.load_current();
                }
                ui.separator();
                if ui.button("◀").clicked() {
                    self.step(-1);
                }
                if ui.button("▶").clicked() {
                    self.step(1);
                }
                let total = self.project.as_ref().map_or(0, |p| p.images.len());
                ui.label(format!("{}/{}", if total == 0 { 0 } else { self.index + 1 }, total));
                if self.dirty {
                    ui.colored_label(Color32::YELLOW, "● unsaved");
                }
            });
            let color = if self.error { Color32::LIGHT_RED } else { Color32::GRAY };
            ui.colored_label(color, &self.status);
        });
    }

    fn side_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("side").min_width(220.0).show(ctx, |ui| {
            ui.heading("Classes");
            let names: Vec<String> = (0..self.classes())
                .map(|id| self.project.as_ref().map_or_else(|| format!("class {id}"), |p| p.class_name(id)))
                .collect();
            for (id, name) in names.iter().enumerate() {
                if ui.selectable_label(self.class_id == id, format!("{id}: {name}")).clicked() {
                    self.class_id = id;
                }
            }
            ui.separator();
            ui.heading("Images");
            egui::ScrollArea::vertical().show(ui, |ui| {
                let items: Vec<(usize, String, Option<usize>)> = self
                    .project
                    .as_ref()
                    .map(|project| {
                        project
                            .images
                            .iter()
                            .enumerate()
                            .map(|(index, entry)| {
                                (
                                    index,
                                    entry
                                        .path
                                        .file_name()
                                        .unwrap_or_default()
                                        .to_string_lossy()
                                        .into_owned(),
                                    entry.marks,
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                for (index, name, marks) in items {
                    let label = match marks {
                        Some(0) | None => name,
                        Some(count) => format!("{name}  [{count}]"),
                    };
                    if ui.selectable_label(self.index == index, label).clicked() && self.index != index {
                        self.index = index;
                        self.load_current();
                    }
                }
            });
        });
    }

    fn canvas(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.ensure_texture(ctx);
            let available = ui.available_size();
            let response = ui.allocate_response(available, Sense::click_and_drag());
            let rect = response.rect;
            let painter = ui.painter_at(rect);
            let Some(texture) = self.texture.clone() else {
                painter.text(
                    rect.center(),
                    egui::Align2::CENTER_CENTER,
                    "Open a project folder to start annotating",
                    egui::FontId::proportional(18.0),
                    Color32::GRAY,
                );
                return;
            };

            let size = self.image_size;
            let scale = (rect.width() / size.x).min(rect.height() / size.y).max(0.0);
            let image_rect = Rect::from_center_size(rect.center(), size * scale);
            painter.image(
                texture.id(),
                image_rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );

            let to_screen = |x: f32, y: f32| {
                Pos2::new(
                    image_rect.left() + x * image_rect.width(),
                    image_rect.top() + y * image_rect.height(),
                )
            };

            for (index, mark) in self.annotation.marks.iter().enumerate() {
                let selected = self.selected == Some(index);
                let color = class_color(mark.class_id);
                let stroke = Stroke::new(if selected { 3.0_f32 } else { 2.0_f32 }, color);
                let min = to_screen(mark.left(), mark.top());
                let max = to_screen(mark.right(), mark.bottom());
                for (a, b) in [
                    (min, Pos2::new(max.x, min.y)),
                    (Pos2::new(max.x, min.y), max),
                    (max, Pos2::new(min.x, max.y)),
                    (Pos2::new(min.x, max.y), min),
                ] {
                    painter.line_segment([a, b], stroke);
                }
                let label = self
                    .project
                    .as_ref()
                    .map_or_else(|| format!("{}", mark.class_id), |p| p.class_name(mark.class_id));
                painter.text(
                    Pos2::new(min.x + 2.0, min.y - 2.0),
                    egui::Align2::LEFT_BOTTOM,
                    label,
                    egui::FontId::proportional(13.0),
                    color,
                );
            }

            if let (Some(start), Some(now)) = (self.drag_start, self.drag_now) {
                let clipped = |p: Pos2| Pos2::new(
                    p.x.clamp(image_rect.left(), image_rect.right()),
                    p.y.clamp(image_rect.top(), image_rect.bottom()),
                );
                let a = clipped(start);
                let b = clipped(now);
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

            if response.drag_started() {
                self.drag_start = response.interact_pointer_pos();
                self.drag_now = self.drag_start;
            }
            if response.dragged() {
                self.drag_now = response.interact_pointer_pos();
            }
            if response.drag_stopped() {
                if let (Some(start), Some(now)) = (self.drag_start.take(), self.drag_now.take()) {
                    self.add_box(start, now, image_rect);
                }
            } else if response.clicked() {
                if let Some(pos) = response.interact_pointer_pos() {
                    let x = (pos.x - image_rect.left()) / image_rect.width().max(1.0);
                    let y = (pos.y - image_rect.top()) / image_rect.height().max(1.0);
                    self.selected = self
                        .annotation
                        .marks
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, mark)| mark.contains(x, y))
                        .map(|(index, _)| index);
                }
            }
        });
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

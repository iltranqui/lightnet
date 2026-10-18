//! DarkMark's detector mouse and edit model (`G9`-`G14`, `D7`-`D11`).
//!
//! Points are image pixels. The editor owns both real marks and predictions
//! (`D10`), so selection, Tab, acceptance and deletion apply to both. Every
//! mutation records the previous mark list on an [`UndoStack`].

use crate::annotation::{Annotation, Mark, Point};
use crate::undo::UndoStack;

/// Smallest box side, in pixels, that a drag may create.
pub const MIN_SIZE: f32 = 2.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mods {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
}

#[derive(Clone, Debug)]
enum Drag {
    New { anchor: Point },
    /// Corner re-drag: the original mark was removed and is restored if the
    /// drag collapses (`D8`).
    Corner { original: Mark, anchor: Point, press: Point },
}

#[derive(Clone, Debug)]
pub struct Editor {
    pub marks: Vec<Mark>,
    pub selected: Option<usize>,
    pub class_count: usize,
    /// Class used for the next created mark.
    pub class_id: usize,
    pub class_names: Vec<String>,
    pub most_recent_size: (f32, f32),
    pub most_recent_class: usize,
    /// Corner hit radius in pixels (`G9` default 10).
    pub corner_size: f32,
    pub width: u32,
    pub height: u32,
    /// Prediction confidence threshold (`G15`).
    pub threshold: f32,
    pub completely_empty: bool,
    drag: Option<Drag>,
    preview: Option<(Point, Point)>,
    undo: UndoStack<Vec<Mark>>,
}

impl Editor {
    pub fn new(width: u32, height: u32, class_names: Vec<String>) -> Self {
        Self {
            marks: Vec::new(),
            selected: None,
            class_count: class_names.len().max(1),
            class_id: 0,
            class_names,
            most_recent_size: (0.0, 0.0),
            most_recent_class: 0,
            corner_size: 10.0,
            width: width.max(1),
            height: height.max(1),
            threshold: 0.25,
            completely_empty: false,
            drag: None,
            preview: None,
            undo: UndoStack::new(),
        }
    }

    pub fn from_annotation(annotation: &Annotation, class_names: Vec<String>) -> Self {
        let mut editor = Self::new(annotation.image_width, annotation.image_height, class_names);
        editor.marks = annotation.marks.clone();
        editor.completely_empty = annotation.completely_empty;
        editor
    }

    pub fn to_annotation(&self) -> Annotation {
        let mut annotation = Annotation::new(self.width, self.height);
        annotation.marks = self.marks.clone();
        annotation.completely_empty = self.completely_empty && !self.has_real_marks();
        annotation
    }

    pub fn real_marks(&self) -> impl Iterator<Item = &Mark> {
        self.marks.iter().filter(|mark| !mark.is_prediction)
    }

    pub fn has_real_marks(&self) -> bool {
        self.marks.iter().any(|mark| !mark.is_prediction)
    }

    pub fn predictions(&self) -> impl Iterator<Item = &Mark> {
        self.marks.iter().filter(|mark| mark.is_prediction)
    }

    /// Predictions at or above the current threshold, in mark order.
    pub fn visible_predictions(&self) -> Vec<usize> {
        self.marks
            .iter()
            .enumerate()
            .filter(|(_, mark)| mark.is_prediction && mark.confidence >= self.threshold)
            .map(|(index, _)| index)
            .collect()
    }

    pub fn class_name(&self, class_id: usize) -> String {
        self.class_names
            .get(class_id)
            .cloned()
            .unwrap_or_else(|| format!("class {class_id}"))
    }

    fn clamp_point(&self, point: Point) -> Point {
        Point::new(
            point.x.clamp(0.0, self.width as f32),
            point.y.clamp(0.0, self.height as f32),
        )
    }

    fn push_undo(&mut self) {
        self.undo.push(self.marks.clone());
    }

    pub fn undo(&mut self) -> bool {
        if let Some(marks) = self.undo.undo(self.marks.clone()) {
            self.marks = marks;
            self.selected = None;
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if let Some(marks) = self.undo.redo(self.marks.clone()) {
            self.marks = marks;
            self.selected = None;
            true
        } else {
            false
        }
    }

    /// Smallest-area mark containing `point`.
    pub fn hit_test(&self, point: Point) -> Option<usize> {
        self.marks
            .iter()
            .enumerate()
            .filter(|(_, mark)| {
                mark.contains(point) && (!mark.is_prediction || mark.confidence >= self.threshold)
            })
            .min_by(|(_, a), (_, b)| a.area().total_cmp(&b.area()))
            .map(|(index, _)| index)
    }

    /// Nearest real-mark corner within `corner_size` of `point`, if the point is
    /// inside that mark's rect (`G9`).
    fn corner_hit(&self, point: Point) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, f32)> = None;
        for (index, mark) in self.marks.iter().enumerate() {
            if mark.is_prediction || !mark.contains(point) {
                continue;
            }
            for (corner, candidate) in mark.points.iter().enumerate() {
                let distance = ((candidate.x - point.x).powi(2) + (candidate.y - point.y).powi(2)).sqrt();
                if distance <= self.corner_size
                    && best.is_none_or(|(_, _, best_distance)| distance < best_distance)
                {
                    best = Some((index, corner, distance));
                }
            }
        }
        best.map(|(index, corner, _)| (index, corner))
    }

    pub fn press(&mut self, point: Point, mods: Mods, double_click: bool) {
        let point = self.clamp_point(point);
        if double_click && self.most_recent_size.0 >= MIN_SIZE && self.most_recent_size.1 >= MIN_SIZE {
            self.push_undo();
            let class_id = self.most_recent_class.min(self.class_count.saturating_sub(1));
            let (w, h) = self.most_recent_size;
            let mark = self.make_mark(
                class_id,
                point.x - w * 0.5,
                point.y - h * 0.5,
                point.x + w * 0.5,
                point.y + h * 0.5,
            );
            if let Some(mark) = mark {
                self.marks.push(mark);
                self.selected = Some(self.marks.len() - 1);
            }
            return;
        }

        self.selected = self.hit_test(point);
        if mods.ctrl {
            return; // Ctrl+drag pans; the GUI handles it.
        }
        if let Some((index, corner)) = self.corner_hit(point) {
            self.push_undo();
            let original = self.marks.remove(index);
            let opposite = original.points[(corner + 2) % 4];
            self.selected = None;
            self.drag = Some(Drag::Corner {
                original,
                anchor: opposite,
                press: point,
            });
            self.preview = Some((opposite, point));
            return;
        }
        self.drag = Some(Drag::New { anchor: point });
        self.preview = Some((point, point));
    }

    pub fn drag(&mut self, point: Point) {
        let point = self.clamp_point(point);
        if let Some(drag) = &self.drag {
            let anchor = match drag {
                Drag::New { anchor } | Drag::Corner { anchor, .. } => *anchor,
            };
            self.preview = Some((anchor, point));
        }
    }

    pub fn release(&mut self, point: Point) -> bool {
        let point = self.clamp_point(point);
        self.preview = None;
        let Some(drag) = self.drag.take() else {
            return false;
        };
        match drag {
            Drag::New { anchor } => {
                let class_id = self.class_id.min(self.class_count.saturating_sub(1));
                if let Some(mark) = self.make_mark(class_id, anchor.x, anchor.y, point.x, point.y) {
                    self.push_undo();
                    self.most_recent_size = (mark.width(), mark.height());
                    self.most_recent_class = mark.class_id;
                    self.marks.push(mark);
                    self.selected = Some(self.marks.len() - 1);
                    true
                } else {
                    false
                }
            }
            Drag::Corner { original, anchor, press } => {
                // A collapsed re-drag restores the original mark (`D8`).
                let collapsed = (point.x - press.x).abs() < MIN_SIZE
                    && (point.y - press.y).abs() < MIN_SIZE;
                if collapsed {
                    self.marks.push(original);
                    return true;
                }
                if let Some(mark) = self.make_mark(original.class_id, anchor.x, anchor.y, point.x, point.y) {
                    self.marks.push(mark);
                    self.selected = Some(self.marks.len() - 1);
                } else {
                    self.marks.push(original);
                }
                true
            }
        }
    }

    pub fn preview(&self) -> Option<(Point, Point)> {
        self.preview
    }

    fn make_mark(&self, class_id: usize, x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Mark> {
        let a = self.clamp_point(Point::new(x0, y0));
        let b = self.clamp_point(Point::new(x1, y1));
        let mark = Mark::from_corners(
            class_id,
            self.class_name(class_id),
            a.x,
            a.y,
            b.x,
            b.y,
        );
        (mark.width() >= MIN_SIZE && mark.height() >= MIN_SIZE).then_some(mark)
    }

    /// Tab / Shift+Tab cycle all marks including predictions (`G12`).
    pub fn cycle(&mut self, backward: bool) {
        if self.marks.is_empty() {
            self.selected = None;
            return;
        }
        self.selected = match (self.selected, backward) {
            (None, false) => Some(0),
            (None, true) => Some(self.marks.len() - 1),
            (Some(index), false) => Some((index + 1) % self.marks.len()),
            (Some(index), true) => Some((index + self.marks.len() - 1) % self.marks.len()),
        };
    }

    /// Digit class selection, additive with Ctrl (+10) and Alt (+20) (`G11`,
    /// `D14`). Refuses a class outside the class list (`D9`).
    pub fn set_class(&mut self, digit: usize, mods: Mods) -> Result<(), String> {
        let class_id = digit + usize::from(mods.ctrl) * 10 + usize::from(mods.alt) * 20;
        if class_id >= self.class_count {
            return Err(format!(
                "class {class_id} is outside the {}-class list",
                self.class_count
            ));
        }
        match self.selected {
            Some(index) if index < self.marks.len() => {
                self.push_undo();
                let name = self.class_name(class_id);
                if let Some(mark) = self.marks.get_mut(index) {
                    mark.class_id = class_id;
                    mark.name = name;
                }
            }
            _ => self.class_id = class_id,
        }
        Ok(())
    }

    pub fn delete_selected(&mut self) -> bool {
        let Some(index) = self.selected.take() else {
            return false;
        };
        if index >= self.marks.len() {
            return false;
        }
        self.push_undo();
        self.marks.remove(index);
        true
    }

    pub fn clear_marks(&mut self) {
        if self.marks.is_empty() {
            return;
        }
        self.push_undo();
        self.marks.clear();
        self.selected = None;
    }

    pub fn set_empty_flag(&mut self, flagged: bool) {
        if flagged && self.has_real_marks() {
            return;
        }
        self.completely_empty = flagged;
    }

    /// Ctrl+W/A/S/D: move every real mark by one pixel; a mark that would leave
    /// the image is left untouched (`G16`).
    pub fn nudge(&mut self, dx: f32, dy: f32) {
        self.push_undo();
        for mark in &mut self.marks {
            if mark.is_prediction {
                continue;
            }
            let (left, right) = (mark.left() + dx, mark.right() + dx);
            let (top, bottom) = (mark.top() + dy, mark.bottom() + dy);
            if left < 0.0 || top < 0.0 || right > self.width as f32 || bottom > self.height as f32 {
                continue;
            }
            for point in &mut mark.points {
                point.x += dx;
                point.y += dy;
            }
        }
    }

    /// `,`/`.`: scale every real mark by `factor` around its centre; shrink is
    /// refused below `MIN_SIZE` (`G16`).
    pub fn scale(&mut self, factor: f32) {
        self.push_undo();
        for mark in &mut self.marks {
            if mark.is_prediction {
                continue;
            }
            let center = mark.center();
            let (width, height) = (mark.width() * factor, mark.height() * factor);
            if width < MIN_SIZE || height < MIN_SIZE {
                continue;
            }
            for point in &mut mark.points {
                point.x = center.x + (point.x - center.x) * factor;
                point.y = center.y + (point.y - center.y) * factor;
            }
        }
    }

    /// `a`: accept every prediction, but only when no accepted mark exists (`G14`).
    pub fn accept_all(&mut self) -> bool {
        if self.has_real_marks() {
            return false;
        }
        if !self.marks.iter().any(|mark| mark.is_prediction) {
            return false;
        }
        self.push_undo();
        let names = self.class_names.clone();
        for mark in &mut self.marks {
            mark.is_prediction = false;
            mark.name = names
                .get(mark.class_id)
                .cloned()
                .unwrap_or_else(|| format!("class {}", mark.class_id));
        }
        true
    }

    /// `A`: accept the selected prediction, or all when nothing is selected (`G14`).
    pub fn accept_selected(&mut self) -> bool {
        match self.selected {
            Some(index) if index < self.marks.len() && self.marks[index].is_prediction => {
                self.push_undo();
                let name = self.class_name(self.marks[index].class_id);
                let mark = &mut self.marks[index];
                mark.is_prediction = false;
                mark.name = name;
                true
            }
            Some(_) => false,
            None => self.accept_all(),
        }
    }

    /// `o`: delete predictions only.
    pub fn delete_predictions(&mut self) {
        if !self.marks.iter().any(|mark| mark.is_prediction) {
            return;
        }
        self.push_undo();
        self.marks.retain(|mark| !mark.is_prediction);
        self.selected = None;
    }

    /// Up/Down threshold, clamped to `[0.05, 0.95]` (`G15`).
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold.clamp(0.05, 0.95);
    }

    pub fn bump_threshold(&mut self, delta: f32) {
        self.set_threshold(self.threshold + delta);
    }

    pub fn set_predictions(&mut self, predictions: Vec<Mark>) {
        if predictions.is_empty() && !self.marks.iter().any(|mark| mark.is_prediction) {
            return;
        }
        self.push_undo();
        self.marks.retain(|mark| !mark.is_prediction);
        self.selected = None;
        for mut mark in predictions {
            mark.is_prediction = true;
            self.marks.push(mark);
        }
    }

    /// Copies marks that are not already present as the same class with the same
    /// corners (`G16`).
    pub fn copy_from(&mut self, source: &Annotation) -> usize {
        let mut added = 0;
        for mark in source.real_marks() {
            let duplicate = self.marks.iter().any(|existing| {
                existing.class_id == mark.class_id && existing.points == mark.points
            });
            if duplicate {
                continue;
            }
            if added == 0 {
                self.push_undo();
            }
            let mut copy = mark.clone();
            copy.name = self.class_name(copy.class_id);
            self.marks.push(copy);
            added += 1;
        }
        added
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn editor() -> Editor {
        Editor::new(100, 100, vec!["a".into(), "b".into(), "c".into()])
    }

    fn mark(class_id: usize, x0: f32, y0: f32, x1: f32, y1: f32) -> Mark {
        Mark::from_corners(class_id, "a", x0, y0, x1, y1)
    }

    #[test]
    fn press_selects_smallest_area_mark() {
        let mut editor = editor();
        editor.marks = vec![mark(0, 0.0, 0.0, 80.0, 80.0), mark(1, 10.0, 10.0, 40.0, 40.0)];
        editor.press(Point::new(20.0, 20.0), Mods::default(), false);
        assert_eq!(editor.selected, Some(1));
    }

    #[test]
    fn corner_press_inside_rect_redrags_from_the_opposite_corner() {
        let mut editor = editor();
        editor.marks = vec![mark(0, 10.0, 10.0, 50.0, 50.0)];
        editor.press(Point::new(11.0, 11.0), Mods::default(), false);
        assert!(editor.marks.is_empty(), "corner press removes the mark");
        editor.release(Point::new(80.0, 80.0));
        assert_eq!(editor.marks.len(), 1);
        let redragged = &editor.marks[0];
        assert_eq!((redragged.left(), redragged.top()), (50.0, 50.0));
        assert_eq!((redragged.right(), redragged.bottom()), (80.0, 80.0));
    }

    #[test]
    fn collapsing_a_corner_drag_restores_the_original() {
        let mut editor = editor();
        let original = mark(0, 10.0, 10.0, 50.0, 50.0);
        editor.marks = vec![original.clone()];
        editor.press(Point::new(11.0, 11.0), Mods::default(), false);
        editor.release(Point::new(11.0, 11.0));
        assert_eq!(editor.marks, vec![original]);
    }

    #[test]
    fn new_marks_are_clamped_to_the_image() {
        let mut editor = editor();
        editor.press(Point::new(-5.0, -5.0), Mods::default(), false);
        editor.release(Point::new(500.0, 500.0));
        let mark = &editor.marks[0];
        assert_eq!((mark.left(), mark.top()), (0.0, 0.0));
        assert_eq!((mark.right(), mark.bottom()), (100.0, 100.0));
    }

    #[test]
    fn class_beyond_the_list_is_refused() {
        let mut editor = editor();
        assert!(editor.set_class(5, Mods::default()).is_err());
        assert!(editor.set_class(1, Mods::default()).is_ok());
        assert_eq!(editor.class_id, 1);
        assert!(editor.set_class(1, Mods { ctrl: true, ..Mods::default() }).is_err());
    }

    #[test]
    fn tab_cycles_including_predictions() {
        let mut editor = editor();
        let mut prediction = mark(0, 60.0, 60.0, 90.0, 90.0);
        prediction.is_prediction = true;
        editor.marks = vec![mark(0, 0.0, 0.0, 10.0, 10.0), prediction];
        editor.cycle(false);
        assert_eq!(editor.selected, Some(0));
        editor.cycle(false);
        assert_eq!(editor.selected, Some(1));
        editor.cycle(false);
        assert_eq!(editor.selected, Some(0));
        editor.selected = None;
        editor.cycle(true);
        assert_eq!(editor.selected, Some(1));
    }

    #[test]
    fn accept_all_only_when_no_real_marks() {
        let mut editor = editor();
        let mut prediction = mark(0, 0.0, 0.0, 10.0, 10.0);
        prediction.is_prediction = true;
        editor.marks = vec![prediction];
        assert!(editor.accept_all());
        assert!(editor.has_real_marks());
        assert!(!editor.accept_all());
    }

    #[test]
    fn threshold_clamps() {
        let mut editor = editor();
        editor.bump_threshold(-10.0);
        assert_eq!(editor.threshold, 0.05);
        editor.bump_threshold(10.0);
        assert_eq!(editor.threshold, 0.95);
    }

    #[test]
    fn copy_skips_identical_class_and_corners() {
        let mut editor = editor();
        editor.marks = vec![mark(1, 10.0, 10.0, 20.0, 20.0)];
        let mut source = Annotation::new(100, 100);
        source.marks = vec![mark(1, 10.0, 10.0, 20.0, 20.0), mark(2, 30.0, 30.0, 40.0, 40.0)];
        assert_eq!(editor.copy_from(&source), 1);
    }
}

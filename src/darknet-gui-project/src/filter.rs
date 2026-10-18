//! Image filters and sort keys (`G13`, `D13`).

use std::time::{Duration, SystemTime};

use regex::Regex;

use crate::project::Project;

/// Three-state filter flag so "unset" and "no" stay distinct.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tri {
    #[default]
    Any,
    Yes,
    No,
}

impl Tri {
    pub fn matches(self, value: bool) -> bool {
        match self {
            Self::Any => true,
            Self::Yes => value,
            Self::No => !value,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Filter {
    /// Substring/regex matched against the relative path.
    pub include: Option<Regex>,
    pub exclude: Option<Regex>,
    /// `Yes` keeps images flagged `completely_empty`; `No` keeps the rest.
    pub empty: Tri,
    /// `Yes` keeps images with no sidecar at all.
    pub unannotated: Tri,
    /// Keeps images whose sidecar is at least this old.
    pub min_age: Option<Duration>,
    /// Keeps images that contain at least one mark of this class.
    pub class: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SortKey {
    #[default]
    Name,
    MarkCount,
}

impl Filter {
    pub fn matches(&self, project: &Project, index: usize) -> bool {
        let Some(entry) = project.images.get(index) else {
            return false;
        };
        if let Some(include) = &self.include
            && !include.is_match(&entry.relative)
        {
            return false;
        }
        if let Some(exclude) = &self.exclude
            && exclude.is_match(&entry.relative)
        {
            return false;
        }

        let annotated = entry.json_path.is_file() || entry.txt_path.is_file();
        if !self.unannotated.matches(!annotated) {
            return false;
        }
        if self.empty != Tri::Any || self.class.is_some() {
            let annotation = project.load_annotation(index).ok();
            let flagged_empty = annotation.as_ref().map(|a| a.completely_empty).unwrap_or(false);
            if !self.empty.matches(flagged_empty) {
                return false;
            }
            if let Some(class) = self.class {
                let Some(annotation) = annotation else {
                    return false;
                };
                if !annotation.real_marks().any(|mark| mark.class_id == class) {
                    return false;
                }
            }
        }
        if let Some(min_age) = self.min_age {
            let path = if entry.json_path.is_file() {
                &entry.json_path
            } else {
                &entry.txt_path
            };
            let Ok(modified) = std::fs::metadata(path).and_then(|meta| meta.modified()) else {
                return false;
            };
            if SystemTime::now().duration_since(modified).unwrap_or_default() < min_age {
                return false;
            }
        }
        true
    }

    /// Indices in display order. `SortKey::MarkCount` is a stable sort, so equal
    /// counts keep alphabetical order (`G13`).
    pub fn apply(&self, project: &Project, sort: SortKey) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..project.images.len())
            .filter(|index| self.matches(project, *index))
            .collect();
        if sort == SortKey::MarkCount {
            indices.sort_by_key(|index| project.mark_count(*index));
        }
        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::{Annotation, Mark};
    use std::path::PathBuf;

    fn temp(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("darknet-gui-filter-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn touch_image(root: &std::path::Path, name: &str) {
    /// A valid 1x1 PNG, so `image::image_dimensions` succeeds.
    fn write_png(path: &std::path::Path) {
        const PNG: [u8; 70] = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0, 0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99,
            0x3d, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        std::fs::write(path, PNG).unwrap();
    }
        write_png(&root.join(name));
    }

    #[test]
    fn regex_and_class_filters_combine() {
        let root = temp("combine");
        for name in ["cat_1.png", "cat_2.png", "dog_1.png"] {
            touch_image(&root, name);
        }
        let mut project = Project::load(&root).unwrap();
        let mut annotation = Annotation::new(1, 1);
        annotation.marks.push(Mark::from_corners(2, "dog", 0.0, 0.0, 1.0, 1.0));
        project.save_annotation(2, &annotation).unwrap();

        let filter = Filter {
            include: Some(Regex::new(r"(?i)cat|dog").unwrap()),
            class: Some(2),
            ..Filter::default()
        };
        let selected = filter.apply(&project, SortKey::Name);
        assert_eq!(selected, vec![2]);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sort_by_mark_count_is_stable() {
        let root = temp("sort");
        for name in ["a.png", "b.png", "c.png", "d.png"] {
            touch_image(&root, name);
        }
        let mut project = Project::load(&root).unwrap();
        let mut two = Annotation::new(1, 1);
        two.marks = vec![
            Mark::from_corners(0, "a", 0.0, 0.0, 1.0, 1.0),
            Mark::from_corners(0, "a", 0.0, 0.0, 1.0, 1.0),
        ];
        project.save_annotation(1, &two).unwrap();
        let indices = Filter::default().apply(&project, SortKey::MarkCount);
        assert_eq!(indices, vec![0, 2, 3, 1], "stable: equal counts stay alphabetical");
        std::fs::remove_dir_all(root).unwrap();
    }
}

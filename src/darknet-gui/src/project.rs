//! Project directory access: image listing, class names, and per-image labels.
//!
//! The on-disk contract is deliberately small and Darknet-native:
//!
//! * images live directly in the project directory;
//! * class names come from the first `*.names` file, else `classes.txt`, else
//!   numeric labels;
//! * labels live next to the image as `<stem>.txt` in YOLO format.
//!
//! Any I/O error is surfaced to the UI. A failed read never produces an empty
//! annotation, because saving that would destroy the user's labels.

use std::fs;
use std::path::{Path, PathBuf};

use crate::annotation::Annotation;

/// Extensions we list, compared case-insensitively.
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "webp"];

/// Directories that hold images but never annotations.
const SKIPPED_DIRS: &[&str] = &["cache", ".git", "target"];

#[derive(Clone, Debug)]
pub struct ImageEntry {
    pub path: PathBuf,
    pub label_path: PathBuf,
    /// Number of boxes, filled in when the image is opened. `None` means "not
    /// read yet", so the list does not parse every file up front.
    pub marks: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Project {
    pub images: Vec<ImageEntry>,
    pub classes: Vec<String>,
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| IMAGE_EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

impl Project {
    /// Scans `root` for images and class names. Unknown directories are skipped
    /// rather than recursed into, so a 50k-image project stays a flat listing.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        if !root.is_dir() {
            return Err(format!("{} is not a directory", root.display()));
        }

        let classes = load_classes(&root);
        let mut images = Vec::new();
        let entries = fs::read_dir(&root).map_err(|error| format!("{}: {error}", root.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| error.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if SKIPPED_DIRS.contains(&name) {
                    continue;
                }
                continue;
            }
            if !is_image(&path) {
                continue;
            }
            images.push(ImageEntry {
                label_path: path.with_extension("txt"),
                path,
                marks: None,
            });
        }
        images.sort_by(|a, b| a.path.cmp(&b.path));
        if images.is_empty() {
            return Err(format!("no images found in {}", root.display()));
        }
        Ok(Self { images, classes })
    }

    pub fn class_name(&self, class_id: usize) -> String {
        self.classes
            .get(class_id)
            .cloned()
            .unwrap_or_else(|| format!("class {class_id}"))
    }

    /// Class count for parsing: at least the number of names, at least 1.
    pub fn class_count(&self) -> usize {
        self.classes.len().max(1)
    }
}

fn load_classes(root: &Path) -> Vec<String> {
    let discover = |candidates: &[&str]| -> Option<PathBuf> {
        for entry in fs::read_dir(root).ok()? {
            let path = entry.ok()?.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if candidates.iter().any(|candidate| {
                candidate.starts_with("*.")
                    && name.to_ascii_lowercase().ends_with(&candidate[1..].to_ascii_lowercase())
            }) || candidates.contains(&name)
            {
                return Some(path);
            }
        }
        None
    };

    let Some(path) = discover(&["*.names", "classes.txt"]) else {
        return Vec::new();
    };
    match fs::read_to_string(&path) {
        Ok(text) => text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Reads the label file for `entry`, treating a missing file as "unannotated".
pub fn load_annotation(entry: &ImageEntry, classes: usize) -> Result<Annotation, String> {
    match fs::read_to_string(&entry.label_path) {
        Ok(text) => Annotation::parse(&text, classes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Annotation::default()),
        Err(error) => Err(format!("{}: {error}", entry.label_path.display())),
    }
}

/// Writes the label file, deleting it when the annotation is empty. Returns
/// whether the file exists afterwards.
pub fn save_annotation(entry: &ImageEntry, annotation: &Annotation) -> Result<bool, String> {
    if annotation.is_empty() {
        match fs::remove_file(&entry.label_path) {
            Ok(()) => Ok(false),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(format!("{}: {error}", entry.label_path.display())),
        }
    } else {
        fs::write(&entry.label_path, annotation.to_yolo())
            .map_err(|error| format!("{}: {error}", entry.label_path.display()))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::Mark;

    fn temp_project(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("darknet-gui-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn lists_images_case_insensitively_and_reads_names() {
        let root = temp_project("list");
        fs::write(root.join("a.JPG"), b"x").unwrap();
        fs::write(root.join("b.png"), b"x").unwrap();
        fs::write(root.join("notes.txt"), b"x").unwrap();
        fs::write(root.join("obj.names"), "person\ncar\n").unwrap();
        let project = Project::open(&root).unwrap();
        assert_eq!(project.images.len(), 2);
        assert_eq!(project.classes, vec!["person", "car"]);
        assert_eq!(project.class_name(1), "car");
        assert_eq!(project.class_name(7), "class 7");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn empty_annotation_deletes_the_label_file() {
        let root = temp_project("save");
        fs::write(root.join("a.jpg"), b"x").unwrap();
        let project = Project::open(&root).unwrap();
        let entry = &project.images[0];
        let mut annotation = Annotation::default();
        annotation.marks.push(Mark { class_id: 0, cx: 0.5, cy: 0.5, w: 0.2, h: 0.2 });
        assert!(save_annotation(entry, &annotation).unwrap());
        assert!(entry.label_path.exists());
        assert_eq!(load_annotation(entry, 1).unwrap(), annotation);
        assert!(!save_annotation(entry, &Annotation::default()).unwrap());
        assert!(!entry.label_path.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrupt_label_file_is_an_error_not_an_empty_annotation() {
        let root = temp_project("corrupt");
        fs::write(root.join("a.jpg"), b"x").unwrap();
        fs::write(root.join("a.txt"), "0 1 2\n").unwrap();
        let project = Project::open(&root).unwrap();
        assert!(load_annotation(&project.images[0], 1).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

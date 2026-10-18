//! Project directory: image scan, class names, and the per-image index.
//!
//! Scan rules are DarkMark's (`G2`, `D6`): recursive, the listed extensions only,
//! and `chart*.png`, `*_mask.png`, `darkmark_image_cache/` and
//! `darkmark_parquet/` excluded. Paths are matched relative to the project root.
//!
//! The project name is the directory name (`G1`); class names come from
//! `<name>.names` (else the first `*.names`), the data file from `<name>.data`
//! (else the first `*.data`, `D5`).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;

use crate::annotation::Annotation;

#[derive(Clone, Debug)]
pub struct ImageEntry {
    pub path: PathBuf,
    pub json_path: PathBuf,
    pub txt_path: PathBuf,
    pub relative: String,
    dimensions: OnceLock<(u32, u32)>,
}

impl ImageEntry {
    /// Image size in pixels, read once from the file header.
    pub fn dimensions(&self) -> Result<(u32, u32), String> {
        if let Some(size) = self.dimensions.get() {
            return Ok(*size);
        }
        let size = image::image_dimensions(&self.path)
            .map_err(|error| format!("{}: {error}", self.path.display()))?;
        let _ = self.dimensions.set(size);
        Ok(size)
    }

    pub fn stem(&self) -> String {
        self.path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct Project {
    pub root: PathBuf,
    pub name: String,
    pub images: Vec<ImageEntry>,
    pub classes: Vec<String>,
    pub data_file: Option<PathBuf>,
    /// Real-mark counts per image; `completely_empty` counts as one (`G13`).
    pub mark_counts: Vec<usize>,
}

fn image_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)^.+\.(jpe?g|bmp|png|webp|tiff?|gif)$").expect("valid image regex")
    })
}

/// `G2` exclusions, applied to the path relative to the project root.
fn excluded(relative: &str) -> bool {
    let lower = relative.to_ascii_lowercase();
    if lower.contains("darkmark_image_cache/") || lower.contains("darkmark_parquet/") {
        return true;
    }
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    if name.ends_with("_mask.png") {
        return true;
    }
    name == "chart.png" || name.starts_with("chart_")
}

impl Project {
    pub fn load(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().canonicalize().map_err(|error| format!("{}: {error}", root.as_ref().display()))?;
        if !root.is_dir() {
            return Err(format!("{} is not a directory", root.display()));
        }
        let name = root
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "project".to_owned());

        let classes = load_classes(&root, &name);
        let data_file = load_data(&root, &name);

        let mut images = Vec::new();
        scan(&root, &root, &mut images)?;
        images.sort_by(|a, b| a.relative.cmp(&b.relative));
        if images.is_empty() {
            return Err(format!("no images found in {}", root.display()));
        }

        let mut mark_counts = Vec::with_capacity(images.len());
        for entry in &images {
            mark_counts.push(mark_count(entry, classes.len())?);
        }
        validate_modes(&images)?;
        Ok(Self {
            root,
            name,
            images,
            classes,
            data_file,
            mark_counts,
        })
    }

    pub fn class_name(&self, class_id: usize) -> String {
        self.classes
            .get(class_id)
            .cloned()
            .unwrap_or_else(|| format!("class {class_id}"))
    }

    /// Class count used when refusing out-of-range digits (`G11`); 1 minimum.
    pub fn class_count(&self) -> usize {
        self.classes.len().max(1)
    }

    pub fn mark_count(&self, index: usize) -> usize {
        self.mark_counts.get(index).copied().unwrap_or(0)
    }

    pub fn refresh_counts(&mut self) {
        let classes = self.classes.len();
        for (index, entry) in self.images.iter().enumerate() {
            if let Ok(count) = mark_count(entry, classes) {
                self.mark_counts[index] = count;
            }
        }
    }

    /// Loads the annotation for `index`, preferring the JSON sidecar and falling
    /// back to the `.txt` when only that exists.
    pub fn load_annotation(&self, index: usize) -> Result<Annotation, String> {
        let entry = self
            .images
            .get(index)
            .ok_or_else(|| format!("image index {index} out of range"))?;
        let (width, height) = entry.dimensions()?;
        match fs::read_to_string(&entry.json_path) {
            Ok(text) => {
                let mut annotation = Annotation::from_json(&text)?;
                if annotation.image_width == 0 || annotation.image_height == 0 {
                    annotation.image_width = width;
                    annotation.image_height = height;
                }
                Ok(annotation)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match fs::read_to_string(&entry.txt_path) {
                    Ok(text) => Annotation::from_yolo(&text, width, height, &self.classes),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        Ok(Annotation::new(width, height))
                    }
                    Err(error) => Err(format!("{}: {error}", entry.txt_path.display())),
                }
            }
            Err(error) => Err(format!("{}: {error}", entry.json_path.display())),
        }
    }

    /// Writes both sidecars according to `G4`/`G5` and updates the index.
    pub fn save_annotation(&mut self, index: usize, annotation: &Annotation) -> Result<(), String> {
        let entry = self
            .images
            .get(index)
            .ok_or_else(|| format!("image index {index} out of range"))?;
        annotation.save(&entry.json_path, &entry.txt_path)?;
        if let Some(slot) = self.mark_counts.get_mut(index) {
            *slot = if annotation.has_real_marks() {
                annotation.real_marks().count()
            } else if annotation.completely_empty {
                1
            } else {
                0
            };
        }
        Ok(())
    }
}

fn mark_count(entry: &ImageEntry, classes: usize) -> Result<usize, String> {
    let _ = classes;
    match fs::read_to_string(&entry.json_path) {
        Ok(text) => {
            let annotation = Annotation::from_json(&text)?;
            Ok(if annotation.has_real_marks() {
                annotation.real_marks().count()
            } else if annotation.completely_empty {
                1
            } else {
                0
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match fs::read_to_string(&entry.txt_path) {
                Ok(text) => Ok(text.lines().filter(|line| !line.trim().is_empty()).count()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
                Err(error) => Err(format!("{}: {error}", entry.txt_path.display())),
            }
        }
        Err(error) => Err(format!("{}: {error}", entry.json_path.display())),
    }
}

/// Rejects a project that contains non-detector sidecars (`D5`).
fn validate_modes(images: &[ImageEntry]) -> Result<(), String> {
    for entry in images {
        if let Ok(text) = fs::read_to_string(&entry.json_path) {
            Annotation::from_json(&text).map_err(|error| format!("{}: {error}", entry.json_path.display()))?;
        }
    }
    Ok(())
}

fn scan(root: &Path, dir: &Path, out: &mut Vec<ImageEntry>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if file_type.is_dir() {
            scan(root, &path, out)?;
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        if !image_regex().is_match(&relative) || excluded(&relative) {
            continue;
        }
        out.push(ImageEntry {
            json_path: path.with_extension("json"),
            txt_path: path.with_extension("txt"),
            path,
            relative,
            dimensions: OnceLock::new(),
        });
    }
    Ok(())
}

fn load_classes(root: &Path, name: &str) -> Vec<String> {
    let preferred = root.join(format!("{name}.names"));
    let path = if preferred.is_file() {
        Some(preferred)
    } else {
        first_with_extension(root, "names")
    };
    path.and_then(|path| fs::read_to_string(path).ok())
        .map(|text| {
            text.lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn load_data(root: &Path, name: &str) -> Option<PathBuf> {
    let preferred = root.join(format!("{name}.data"));
    if preferred.is_file() {
        Some(preferred)
    } else {
        first_with_extension(root, "data")
    }
}

fn first_with_extension(dir: &Path, extension: &str) -> Option<PathBuf> {
    let mut matches: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|value| value.to_str()) == Some(extension)
        })
        .collect();
    matches.sort();
    matches.into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::Mark;

    fn temp(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("darknet-gui-proj-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn fake_image(path: &Path) {
        // 1x1 PNG.
        let png: [u8; 70] = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0xf8, 0xcf, 0xc0, 0xf0, 0x1f, 0x00, 0x05, 0x00, 0x01, 0xff, 0x89, 0x99,
            0x3d, 0x1d, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        fs::write(path, png).unwrap();
    }

    #[test]
    fn scan_ignores_charts_masks_and_cache_dirs() {
        let root = temp("scan");
        fake_image(&root.join("keep.png"));
        fake_image(&root.join("chart.png"));
        fake_image(&root.join("chart_loss.png"));
        fake_image(&root.join("keep_mask.png"));
        fs::create_dir_all(root.join("darkmark_image_cache")).unwrap();
        fake_image(&root.join("darkmark_image_cache/cached.png"));
        fake_image(&root.join("skip.txt1"));
        fs::write(root.join("notes.txt"), b"x").unwrap();
        let project = Project::load(&root).unwrap();
        assert_eq!(project.images.len(), 1);
        assert_eq!(project.images[0].relative, "keep.png");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prefers_dirname_data_and_names_files() {
        let root = temp("prefer");
        let dir_name = root.file_name().unwrap().to_string_lossy().into_owned();
        fake_image(&root.join("a.png"));
        fs::write(root.join("aaa.names"), "wrong\n").unwrap();
        fs::write(root.join(format!("{dir_name}.names")), "right\n").unwrap();
        fs::write(root.join("aaa.data"), "wrong\n").unwrap();
        fs::write(root.join(format!("{dir_name}.data")), "right\n").unwrap();
        let project = Project::load(&root).unwrap();
        assert_eq!(project.classes, vec!["right"]);
        assert!(project.data_file.unwrap().ends_with(format!("{dir_name}.data")));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn index_counts_completely_empty_as_one() {
        let root = temp("index");
        fake_image(&root.join("a.png"));
        fake_image(&root.join("b.png"));
        let mut project = Project::load(&root).unwrap();
        assert_eq!(project.mark_count(0), 0);
        let mut annotation = Annotation::new(1, 1);
        annotation.completely_empty = true;
        project.save_annotation(0, &annotation).unwrap();
        assert_eq!(project.mark_count(0), 1);
        annotation.completely_empty = false;
        annotation.marks.push(Mark::from_corners(0, "x", 0.0, 0.0, 1.0, 1.0));
        project.save_annotation(0, &annotation).unwrap();
        assert_eq!(project.mark_count(0), 1);
        assert_eq!(project.mark_count(1), 0);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn load_rejects_non_detector_sidecars() {
        let root = temp("modes");
        fake_image(&root.join("a.png"));
        fs::write(
            root.join("a.json"),
            r#"{"annotation_mode":"obo","image":{"width":1,"height":1},"mark":[]}"#,
        )
        .unwrap();
        assert!(Project::load(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}

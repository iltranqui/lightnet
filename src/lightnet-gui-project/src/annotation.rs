//! DarkMark-compatible annotation sidecars.
//!
//! Implements the detector-mode subset of DarkMark's `.json` format (ground
//! truth `G3`-`G7`, `D1`-`D4`, `D16`) and the `.txt` label file (`G5`).
//!
//! Geometry: DarkMark stores every coordinate normalized to `0.0..1.0`
//! (`encode_annotation_mark_json`, `AnnotationJson.cpp:528-560`;
//! `DMContent::save_json`, `DMContent.cpp:2950-2975`) and repeats it as integer
//! pixels in `int_x`/`int_y` for third-party readers. Marks here are held in
//! image pixels and converted on load and save, so a sidecar round-trips.
//! Coordinates are `f64`, the same precision DarkMark uses, so a save does not
//! turn `0.1` into `0.10000000149011612`.
//!
//! The JSON is parsed through [`serde_json::Value`] so unknown fields survive a
//! load/save round trip: DarkMark writes fields this port does not model, and
//! dropping them would make the two tools incompatible.
//!
//! Save rules (`G4`, `G5`):
//!
//! * no real marks, not flagged empty -> delete `.json` and `.txt`;
//! * flagged empty -> write `.json` with `completely_empty: true` and an
//!   (empty) `.txt`;
//! * otherwise -> write both.
//!
//! File writes go through a sibling temp file and a rename, so a crash or a full
//! disk cannot leave a truncated sidecar.
//!
//! Predictions are never written.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

/// Version string written to new sidecars. DarkMark never reads it (`G17`).
pub const VERSION: &str = concat!("lightnet-gui ", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// One detection box, in image pixels (the space the GUI draws in). DarkMark
/// stores the same corners normalized; [`Annotation`] converts on load and save.
#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub class_id: usize,
    pub name: String,
    /// DarkMark's `description`. Kept verbatim: a save must not overwrite a
    /// description that came from the sidecar with the class name.
    pub description: String,
    /// The four corners in DarkMark's order: TL, TR, BR, BL.
    pub points: [Point; 4],
    pub is_prediction: bool,
    /// Prediction confidence. Real marks carry 1.0; never serialized.
    pub confidence: f32,
}

impl Mark {
    pub fn from_corners(class_id: usize, name: impl Into<String>, x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        let name = name.into();
        let (left, right) = (x0.min(x1), x0.max(x1));
        let (top, bottom) = (y0.min(y1), y0.max(y1));
        Self {
            class_id,
            description: name.clone(),
            name,
            points: [
                Point::new(left, top),
                Point::new(right, top),
                Point::new(right, bottom),
                Point::new(left, bottom),
            ],
            is_prediction: false,
            confidence: 1.0,
        }
    }

    /// A detector prediction. `G3`: predictions are never written to disk.
    pub fn prediction(
        class_id: usize,
        name: impl Into<String>,
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        confidence: f32,
    ) -> Self {
        let mut mark = Self::from_corners(class_id, name, x0, y0, x1, y1);
        mark.is_prediction = true;
        mark.confidence = confidence;
        mark
    }

    pub fn left(&self) -> f64 {
        self.points[0].x
    }

    pub fn top(&self) -> f64 {
        self.points[0].y
    }

    pub fn right(&self) -> f64 {
        self.points[1].x
    }

    pub fn bottom(&self) -> f64 {
        self.points[2].y
    }

    pub fn width(&self) -> f64 {
        self.right() - self.left()
    }

    pub fn height(&self) -> f64 {
        self.bottom() - self.top()
    }

    pub fn area(&self) -> f64 {
        (self.width() * self.height()).abs()
    }

    pub fn center(&self) -> Point {
        Point::new(
            (self.left() + self.right()) * 0.5,
            (self.top() + self.bottom()) * 0.5,
        )
    }

    pub fn contains(&self, p: Point) -> bool {
        p.x >= self.left() && p.x <= self.right() && p.y >= self.top() && p.y <= self.bottom()
    }

    /// Normalized `cx cy w h` against the image size (`G5`).
    pub fn normalized(&self, width: u32, height: u32) -> (f64, f64, f64, f64) {
        let (w, h) = (width.max(1) as f64, height.max(1) as f64);
        (
            (self.left() + self.right()) * 0.5 / w,
            (self.top() + self.bottom()) * 0.5 / h,
            self.width() / w,
            self.height() / h,
        )
    }

    fn to_json(&self, width: u32, height: u32) -> Value {
        let (w, h) = (width.max(1) as f64, height.max(1) as f64);
        let points: Vec<Value> = self
            .points
            .iter()
            .map(|point| {
                json!({
                    "x": point.x / w,
                    "y": point.y / h,
                    "int_x": point.x.round() as i64,
                    "int_y": point.y.round() as i64,
                })
            })
            .collect();
        let (x, y, bw, bh) = (self.left(), self.top(), self.width(), self.height());
        json!({
            "class_idx": self.class_id,
            "name": self.name,
            "description": self.description,
            "is_prediction": self.is_prediction,
            "annotation_kind": "detector",
            "points": points,
            "rect": {
                "x": x / w,
                "y": y / h,
                "w": bw / w,
                "h": bh / h,
                "int_x": x.round() as i64,
                "int_y": y.round() as i64,
                "int_w": bw.round() as i64,
                "int_h": bh.round() as i64,
            },
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Annotation {
    pub marks: Vec<Mark>,
    pub completely_empty: bool,
    pub image_width: u32,
    pub image_height: u32,
    /// `image.scale`: the display scale that was active when DarkMark saved
    /// (`DMContent.cpp:2967`). The GUI records its canvas fit/zoom here.
    pub scale: f64,
    /// Original JSON, kept so unknown fields survive a round trip. `None` for
    /// annotations created in memory (no sidecar existed yet).
    original: Option<Map<String, Value>>,
    /// Original mark objects, aligned with `marks`, so mark-level unknown fields
    /// survive as well.
    original_marks: Vec<Map<String, Value>>,
}

impl Annotation {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            marks: Vec::new(),
            completely_empty: false,
            image_width: width,
            image_height: height,
            scale: 1.0,
            original: None,
            original_marks: Vec::new(),
        }
    }

    pub fn real_marks(&self) -> impl Iterator<Item = &Mark> {
        self.marks.iter().filter(|mark| !mark.is_prediction)
    }

    pub fn has_real_marks(&self) -> bool {
        self.marks.iter().any(|mark| !mark.is_prediction)
    }

    /// Parses a DarkMark sidecar. Rejects non-detector modes (`G6`), polygons
    /// with anything other than four points (`D4`), and marks without a usable
    /// `points` array (`G7`).
    ///
    /// The `image` block carries the size the normalized points are relative to.
    /// Use [`Annotation::from_json_sized`] when the file may lack it.
    pub fn from_json(text: &str) -> Result<Self, String> {
        Self::parse(text, None)
    }

    /// Parses a sidecar and falls back to the real image size when the `image`
    /// block is missing or has no dimensions, so old or third-party files with
    /// normalized points still load correctly.
    pub fn from_json_sized(text: &str, width: u32, height: u32) -> Result<Self, String> {
        Self::parse(text, Some((width, height)))
    }

    fn parse(text: &str, fallback: Option<(u32, u32)>) -> Result<Self, String> {
        let root: Value = serde_json::from_str(text).map_err(|error| format!("invalid JSON: {error}"))?;
        let object = root
            .as_object()
            .ok_or_else(|| "sidecar root must be an object".to_owned())?;

        // `G6`: the mode may be a string or an integer.
        match object.get("annotation_mode") {
            Some(Value::String(mode)) if mode == "detector" => {}
            Some(Value::Number(number)) if number.as_i64() == Some(0) => {}
            Some(other) => {
                return Err(format!(
                    "unsupported annotation_mode {other}; only detector mode is handled"
                ));
            }
            // Pre-mode DarkMark sidecars (e.g. LegoGears_v2) have no key; DarkMark itself
            // accepts them (`AnnotationJson.cpp:729`), so treat them as detector.
            None => {}
        }

        let (json_width, json_height, scale) = match object.get("image") {
            Some(Value::Object(image)) => (
                image.get("width").and_then(Value::as_u64).unwrap_or(0) as u32,
                image.get("height").and_then(Value::as_u64).unwrap_or(0) as u32,
                image.get("scale").and_then(Value::as_f64).unwrap_or(1.0),
            ),
            _ => (0, 0, 1.0),
        };
        let (image_width, image_height) = if json_width > 0 && json_height > 0 {
            (json_width, json_height)
        } else {
            fallback.unwrap_or((json_width, json_height))
        };

        let mut marks = Vec::new();
        let mut original_marks = Vec::new();
        if let Some(Value::Array(entries)) = object.get("mark") {
            for (index, entry) in entries.iter().enumerate() {
                marks.push(parse_mark(entry, index, image_width, image_height)?);
                original_marks.push(entry.as_object().cloned().unwrap_or_default());
            }
        }

        Ok(Self {
            marks,
            completely_empty: object
                .get("completely_empty")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            image_width,
            image_height,
            scale,
            original: Some(object.clone()),
            original_marks,
        })
    }

    /// Serializes with DarkMark's `dump(1, '\t') + "\n"` shape (`G3`) and
    /// normalized geometry (`AnnotationJson.cpp:528-560`).
    pub fn to_json(&self) -> String {
        let mut root = self.original.clone().unwrap_or_default();
        root.insert("annotation_mode".to_owned(), json!("detector"));
        root.insert(
            "image".to_owned(),
            json!({
                "scale": self.scale,
                "width": self.image_width,
                "height": self.image_height,
            }),
        );
        let marks: Vec<Value> = self
            .marks
            .iter()
            .enumerate()
            .filter(|(_, mark)| !mark.is_prediction)
            .map(|(index, mark)| {
                merge_mark(
                    mark,
                    self.original_marks.get(index),
                    self.image_width,
                    self.image_height,
                )
            })
            .collect();
        root.insert("mark".to_owned(), Value::Array(marks));
        root.insert(
            "timestamp".to_owned(),
            json!(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)),
        );
        root.insert("version".to_owned(), json!(VERSION));
        root.insert("completely_empty".to_owned(), json!(self.completely_empty));
        // DarkMark writes `dump(1, '\t') + "\n"` (`G3`): one tab per level.
        // `serde_json`'s pretty printer uses two spaces, so rewrite the leading
        // indentation. Only line prefixes are touched, never string contents.
        let pretty = serde_json::to_string_pretty(&Value::Object(root))
            .unwrap_or_else(|_| "{}".to_owned());
        let mut text = String::with_capacity(pretty.len());
        for line in pretty.lines() {
            let spaces = line.len() - line.trim_start_matches(' ').len();
            for _ in 0..spaces / 2 {
                text.push('\t');
            }
            text.push_str(&line[spaces..]);
            text.push('\n');
        }
        text
    }

    /// YOLO `.txt` body: one `class cx cy w h` line per real mark, `G5`'s ten
    /// decimals, unclamped.
    pub fn to_yolo(&self) -> String {
        let mut out = String::new();
        for mark in self.real_marks() {
            let (cx, cy, w, h) = mark.normalized(self.image_width, self.image_height);
            let _ = writeln!(out, "{} {cx:.10} {cy:.10} {w:.10} {h:.10}", mark.class_id);
        }
        out
    }

    /// Parses a YOLO `.txt` into pixel-space marks. Missing files are handled by
    /// the caller (`project::load`).
    pub fn from_yolo(text: &str, width: u32, height: u32, classes: &[String]) -> Result<Self, String> {
        let (w, h) = (width.max(1) as f64, height.max(1) as f64);
        let mut annotation = Self::new(width, height);
        for (index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 5 {
                return Err(format!("line {}: expected 5 fields, got {}", index + 1, fields.len()));
            }
            let class_id: usize = fields[0]
                .parse()
                .map_err(|_| format!("line {}: `{}` is not a class index", index + 1, fields[0]))?;
            let numbers: Vec<f64> = fields[1..]
                .iter()
                .map(|field| {
                    field
                        .parse::<f64>()
                        .map_err(|_| format!("line {}: `{field}` is not a number", index + 1))
                })
                .collect::<Result<_, _>>()?;
            let (cx, cy, bw, bh) = (numbers[0], numbers[1], numbers[2], numbers[3]);
            if ![cx, cy, bw, bh].iter().all(|value| value.is_finite()) || bw <= 0.0 || bh <= 0.0 {
                return Err(format!("line {}: invalid box geometry", index + 1));
            }
            let (px, py) = (cx * w, cy * h);
            let (pw, ph) = (bw * w, bh * h);
            annotation.marks.push(Mark::from_corners(
                class_id,
                classes.get(class_id).cloned().unwrap_or_else(|| format!("class {class_id}")),
                px - pw * 0.5,
                py - ph * 0.5,
                px + pw * 0.5,
                py + ph * 0.5,
            ));
        }
        Ok(annotation)
    }

    /// Applies `G4`/`G5` to disk. Returns whether a label file exists after the
    /// call. Both files are written through a temp file plus rename.
    pub fn save(&self, json_path: &Path, txt_path: &Path) -> Result<bool, String> {
        self.validate()?;
        if !self.has_real_marks() && !self.completely_empty {
            remove_if_present(json_path)?;
            remove_if_present(txt_path)?;
            return Ok(false);
        }
        write_atomic(json_path, self.to_json().as_bytes())?;
        write_atomic(txt_path, self.to_yolo().as_bytes())?;
        Ok(true)
    }

    /// Refuses what DarkMark refuses: a non-finite coordinate never reaches a
    /// file (`validate_annotation_batch`, `AnnotationPersistence.cpp:93-118`).
    /// `serde_json` would otherwise silently write `null`.
    fn validate(&self) -> Result<(), String> {
        for (index, mark) in self.marks.iter().enumerate() {
            if mark.is_prediction {
                continue;
            }
            if mark.points.iter().any(|point| !point.x.is_finite() || !point.y.is_finite()) {
                return Err(format!("mark #{index} contains a non-finite point"));
            }
        }
        Ok(())
    }
}

/// Writes `mark` over the original object so unknown fields stay.
fn merge_mark(
    mark: &Mark,
    original: Option<&Map<String, Value>>,
    width: u32,
    height: u32,
) -> Value {
    let mut object = original.cloned().unwrap_or_default();
    if let Value::Object(fresh) = mark.to_json(width, height) {
        for (key, value) in fresh {
            object.insert(key, value);
        }
    }
    Value::Object(object)
}

fn parse_mark(
    entry: &Value,
    index: usize,
    image_width: u32,
    image_height: u32,
) -> Result<Mark, String> {
    let object = entry
        .as_object()
        .ok_or_else(|| format!("mark {index}: not an object"))?;
    if let Some(kind) = object.get("annotation_kind")
        && !kind.is_string()
    {
        return Err(format!("mark {index}: annotation_kind must be a string (`G7`)"));
    }
    let class_id = object
        .get("class_idx")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("mark {index}: missing class_idx"))? as usize;
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    // DarkMark falls back to the class name when `description` is absent
    // (`AnnotationJson.cpp:483`).
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| name.clone());
    let points = object
        .get("points")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("mark {index}: missing points"))?;
    if points.len() != 4 {
        return Err(format!(
            "mark {index}: expected exactly 4 detector points, got {} (`D4`)",
            points.len()
        ));
    }
    // Stored normalized; a zero size cannot be denormalized, so the raw value is
    // kept rather than collapsed to zero.
    let (scale_x, scale_y) = (
        image_width.max(1) as f64,
        image_height.max(1) as f64,
    );
    let mut corners = [Point::new(0.0, 0.0); 4];
    for (slot, point) in corners.iter_mut().zip(points) {
        let x = point.get("x").and_then(Value::as_f64);
        let y = point.get("y").and_then(Value::as_f64);
        let (Some(x), Some(y)) = (x, y) else {
            return Err(format!("mark {index}: each point needs numeric x and y"));
        };
        *slot = Point::new(x * scale_x, y * scale_y);
    }
    Ok(Mark {
        class_id,
        name,
        description,
        points: corners,
        is_prediction: object
            .get("is_prediction")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        confidence: 1.0,
    })
}

fn remove_if_present(path: &Path) -> Result<(), String> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}

/// Sibling temp path, so the rename stays inside one filesystem.
fn temp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "sidecar".to_owned());
    path.with_file_name(format!("{name}.{}.tmp", std::process::id()))
}

/// Writes through a temp file and renames it over `path`. DarkMark truncates the
/// real file in place (`write_validated_annotation_file`,
/// `AnnotationPersistence.cpp:209-226`), which can leave a half-written sidecar.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let temporary = temp_path(path);
    std::fs::write(&temporary, bytes).map_err(|error| format!("{}: {error}", temporary.display()))?;
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        format!("{}: {error}", path.display())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
	"annotation_mode": "detector",
	"image": { "scale": 2.05625, "width": 640, "height": 480 },
	"mark": [
		{
			"class_idx": 1,
			"name": "car",
			"description": "front wheel",
			"is_prediction": false,
			"annotation_kind": "detector",
			"points": [
				{"x": 0.1, "y": 0.1, "int_x": 64, "int_y": 48},
				{"x": 0.5, "y": 0.1, "int_x": 320, "int_y": 48},
				{"x": 0.5, "y": 0.5, "int_x": 320, "int_y": 240},
				{"x": 0.1, "y": 0.5, "int_x": 64, "int_y": 240}
			],
			"rect": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4, "int_x": 64, "int_y": 48, "int_w": 256, "int_h": 192},
			"custom_darkmark_field": {"kept": true}
		}
	],
	"timestamp": 1700000000,
	"version": "DarkMark 1.0",
	"completely_empty": false
}"#;

    #[test]
    fn roundtrip_preserves_unknown_fields() {
        let annotation = Annotation::from_json(SAMPLE).unwrap();
        assert_eq!(annotation.marks.len(), 1);
        assert_eq!(annotation.marks[0].class_id, 1);
        let text = annotation.to_json();
        assert!(text.contains("custom_darkmark_field"), "unknown field must survive");
        let again = Annotation::from_json(&text).unwrap();
        assert_eq!(again.marks, annotation.marks);
    }

    #[test]
    fn normalized_points_load_as_image_pixels() {
        let annotation = Annotation::from_json(SAMPLE).unwrap();
        let mark = &annotation.marks[0];
        assert_eq!(mark.points[0], Point::new(64.0, 48.0));
        assert_eq!(mark.points[2], Point::new(320.0, 240.0));
        assert_eq!((mark.width(), mark.height()), (256.0, 192.0));
    }

    #[test]
    fn save_writes_normalized_geometry_like_darkmark() {
        let annotation = Annotation::from_json(SAMPLE).unwrap();
        let text = annotation.to_json();
        let root: Value = serde_json::from_str(&text).unwrap();
        let point = &root["mark"][0]["points"][0];
        assert_eq!(point["x"].as_f64(), Some(0.1), "points are normalized");
        assert_eq!(point["int_x"].as_i64(), Some(64), "int_* stays in pixels");
        assert_eq!(root["mark"][0]["rect"]["w"].as_f64(), Some(0.4));
        assert_eq!(root["mark"][0]["rect"]["int_w"].as_i64(), Some(256));
        assert_eq!(root["image"]["scale"].as_f64(), Some(2.05625));
        assert_eq!(root["mark"][0]["description"].as_str(), Some("front wheel"));
    }

    #[test]
    fn coordinates_are_doubles_not_f32_noise() {
        let mut annotation = Annotation::new(1000, 1000);
        annotation
            .marks
            .push(Mark::from_corners(0, "a", 100.0, 100.0, 200.0, 200.0));
        let text = annotation.to_json();
        assert!(text.contains("\"x\": 0.1"), "{text}");
        assert!(!text.contains("0.10000000149011612"), "{text}");
    }

    #[test]
    fn sized_fallback_denormalizes_without_an_image_block() {
        let text = SAMPLE.replacen(
            "\t\"image\": { \"scale\": 2.05625, \"width\": 640, \"height\": 480 },\n",
            "",
            1,
        );
        assert!(!text.contains("\"image\""));
        let annotation = Annotation::from_json_sized(&text, 640, 480).unwrap();
        assert_eq!(annotation.marks[0].points[0], Point::new(64.0, 48.0));
        assert_eq!(annotation.scale, 1.0);
    }

    #[test]
    fn non_finite_marks_are_refused_and_nothing_is_written() {
        let root = temp_dir("nonfinite");
        let (json_path, txt_path) = (root.join("a.json"), root.join("a.txt"));
        let mut annotation = Annotation::new(100, 100);
        let mut mark = Mark::from_corners(0, "a", 1.0, 1.0, 10.0, 10.0);
        mark.points[1].x = f64::NAN;
        annotation.marks.push(mark);
        let error = annotation.save(&json_path, &txt_path).unwrap_err();
        assert!(error.contains("non-finite"), "{error}");
        assert!(!json_path.exists() && !txt_path.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn saves_leave_no_temp_files() {
        let root = temp_dir("atomic");
        let (json_path, txt_path) = (root.join("a.json"), root.join("a.txt"));
        let mut annotation = Annotation::new(100, 100);
        annotation
            .marks
            .push(Mark::from_corners(0, "a", 10.0, 10.0, 50.0, 50.0));
        annotation.save(&json_path, &txt_path).unwrap();
        let names: Vec<String> = std::fs::read_dir(&root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names.iter().all(|name| !name.ends_with(".tmp")), "{names:?}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn json_uses_tab_indentation_like_darkmark() {
        let annotation = Annotation::from_json(SAMPLE).unwrap();
        let text = annotation.to_json();
        assert!(text.contains("\n\t\"annotation_mode\""), "{text}");
        assert!(
            text.lines().all(|line| !line.starts_with(' ')),
            "no space indentation may remain"
        );
    }

    #[test]
    fn missing_annotation_mode_is_legacy_detector() {
        let text = SAMPLE.replacen("\t\"annotation_mode\": \"detector\",\n", "", 1);
        assert!(!text.contains("annotation_mode"));
        assert!(Annotation::from_json(&text).is_ok());
    }

    #[test]
    fn integer_mode_zero_is_detector_and_other_modes_are_rejected() {
        assert!(Annotation::from_json(&SAMPLE.replacen("\"annotation_mode\": \"detector\"", "\"annotation_mode\": 0", 1)).is_ok());
        for mode in ["\"obb\"", "\"segmentor\"", "\"cuboid\"", "1", "4"] {
            let text = SAMPLE.replacen("\"annotation_mode\": \"detector\"", &format!("\"annotation_mode\": {mode}"), 1);
            assert!(Annotation::from_json(&text).is_err(), "{mode} must be rejected");
        }
    }

    #[test]
    fn polygon_with_five_points_is_rejected() {
        let text = SAMPLE.replace(
            "\t\t\t\t{\"x\": 0.1, \"y\": 0.5, \"int_x\": 64, \"int_y\": 240}",
            "\t\t\t\t{\"x\": 0.1, \"y\": 0.5, \"int_x\": 64, \"int_y\": 240},\n\t\t\t\t{\"x\": 0.0, \"y\": 0.0, \"int_x\": 0, \"int_y\": 0}",
        );
        assert!(Annotation::from_json(&text).is_err());
    }

    #[test]
    fn no_marks_unflagged_deletes_json_and_txt() {
        let root = temp_dir("delete");
        let (json_path, txt_path) = (root.join("a.json"), root.join("a.txt"));
        std::fs::write(&json_path, b"{}").unwrap();
        std::fs::write(&txt_path, b"0 0.5 0.5 0.1 0.1\n").unwrap();
        let annotation = Annotation::new(10, 10);
        assert!(!annotation.save(&json_path, &txt_path).unwrap());
        assert!(!json_path.exists() && !txt_path.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn flagged_empty_writes_empty_txt_and_completely_empty_true() {
        let root = temp_dir("flagged");
        let (json_path, txt_path) = (root.join("a.json"), root.join("a.txt"));
        let mut annotation = Annotation::new(10, 10);
        annotation.completely_empty = true;
        assert!(annotation.save(&json_path, &txt_path).unwrap());
        assert_eq!(std::fs::read_to_string(&txt_path).unwrap(), "");
        let text = std::fs::read_to_string(&json_path).unwrap();
        assert!(text.contains("\"completely_empty\": true"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn txt_uses_ten_decimals_and_is_unclamped() {
        let mut annotation = Annotation::new(100, 100);
        annotation.marks.push(Mark::from_corners(0, "a", -10.0, -10.0, 50.0, 50.0));
        let text = annotation.to_yolo();
        assert_eq!(text, "0 0.2000000000 0.2000000000 0.6000000000 0.6000000000\n");
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("lightnet-gui-ann-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}

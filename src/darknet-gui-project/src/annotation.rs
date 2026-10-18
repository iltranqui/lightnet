//! DarkMark-compatible annotation sidecars.
//!
//! Implements the detector-mode subset of DarkMark's `.json` format (ground
//! truth `G3`-`G7`, `D1`-`D4`, `D16`) and the `.txt` label file (`G5`).
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
//! Predictions are never written.

use std::fmt::Write as _;
use std::path::Path;

use serde_json::{Map, Value, json};

/// Version string written to new sidecars. DarkMark never reads it (`G17`).
pub const VERSION: &str = concat!("darknet-gui ", env!("CARGO_PKG_VERSION"));

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// One detection box, in image pixels (the same space DarkMark stores).
#[derive(Clone, Debug, PartialEq)]
pub struct Mark {
    pub class_id: usize,
    pub name: String,
    /// The four corners in DarkMark's order: TL, TR, BR, BL.
    pub points: [Point; 4],
    pub is_prediction: bool,
    /// Prediction confidence. Real marks carry 1.0; never serialized.
    pub confidence: f32,
}

impl Mark {
    pub fn from_corners(class_id: usize, name: impl Into<String>, x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        let (left, right) = (x0.min(x1), x0.max(x1));
        let (top, bottom) = (y0.min(y1), y0.max(y1));
        Self {
            class_id,
            name: name.into(),
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
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
        confidence: f32,
    ) -> Self {
        let mut mark = Self::from_corners(class_id, name, x0, y0, x1, y1);
        mark.is_prediction = true;
        mark.confidence = confidence;
        mark
    }

    pub fn left(&self) -> f32 {
        self.points[0].x
    }

    pub fn top(&self) -> f32 {
        self.points[0].y
    }

    pub fn right(&self) -> f32 {
        self.points[1].x
    }

    pub fn bottom(&self) -> f32 {
        self.points[2].y
    }

    pub fn width(&self) -> f32 {
        self.right() - self.left()
    }

    pub fn height(&self) -> f32 {
        self.bottom() - self.top()
    }

    pub fn area(&self) -> f32 {
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
    pub fn normalized(&self, width: u32, height: u32) -> (f32, f32, f32, f32) {
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
        (
            (self.left() + self.right()) * 0.5 / w,
            (self.top() + self.bottom()) * 0.5 / h,
            self.width() / w,
            self.height() / h,
        )
    }

    fn to_json(&self) -> Value {
        let points: Vec<Value> = self
            .points
            .iter()
            .map(|point| {
                json!({
                    "x": point.x,
                    "y": point.y,
                    "int_x": point.x.round() as i64,
                    "int_y": point.y.round() as i64,
                })
            })
            .collect();
        let (x, y, w, h) = (self.left(), self.top(), self.width(), self.height());
        json!({
            "class_idx": self.class_id,
            "name": self.name,
            "description": self.name,
            "is_prediction": self.is_prediction,
            "annotation_kind": "detector",
            "points": points,
            "rect": {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "int_x": x.round() as i64,
                "int_y": y.round() as i64,
                "int_w": w.round() as i64,
                "int_h": h.round() as i64,
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
    pub fn from_json(text: &str) -> Result<Self, String> {
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
            None => return Err("missing annotation_mode".to_owned()),
        }

        let (image_width, image_height) = match object.get("image") {
            Some(Value::Object(image)) => (
                image.get("width").and_then(Value::as_u64).unwrap_or(0) as u32,
                image.get("height").and_then(Value::as_u64).unwrap_or(0) as u32,
            ),
            _ => (0, 0),
        };

        let mut marks = Vec::new();
        let mut original_marks = Vec::new();
        if let Some(Value::Array(entries)) = object.get("mark") {
            for (index, entry) in entries.iter().enumerate() {
                marks.push(parse_mark(entry, index)?);
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
            original: Some(object.clone()),
            original_marks,
        })
    }

    /// Serializes with DarkMark's `dump(1, '\t') + "\n"` shape (`G3`).
    pub fn to_json(&self) -> String {
        let mut root = self.original.clone().unwrap_or_default();
        root.insert("annotation_mode".to_owned(), json!("detector"));
        root.insert(
            "image".to_owned(),
            json!({
                "scale": 1,
                "width": self.image_width,
                "height": self.image_height,
            }),
        );
        let marks: Vec<Value> = self
            .marks
            .iter()
            .enumerate()
            .filter(|(_, mark)| !mark.is_prediction)
            .map(|(index, mark)| merge_mark(mark, self.original_marks.get(index)))
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
        let (w, h) = (width.max(1) as f32, height.max(1) as f32);
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
            let numbers: Vec<f32> = fields[1..]
                .iter()
                .map(|field| {
                    field
                        .parse::<f32>()
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
    /// call.
    pub fn save(&self, json_path: &Path, txt_path: &Path) -> Result<bool, String> {
        if !self.has_real_marks() && !self.completely_empty {
            remove_if_present(json_path)?;
            remove_if_present(txt_path)?;
            return Ok(false);
        }
        write(json_path, self.to_json().as_bytes())?;
        write(txt_path, self.to_yolo().as_bytes())?;
        Ok(true)
    }
}

/// Writes `mark` over the original object so unknown fields stay.
fn merge_mark(mark: &Mark, original: Option<&Map<String, Value>>) -> Value {
    let mut object = original.cloned().unwrap_or_default();
    if let Value::Object(fresh) = mark.to_json() {
        for (key, value) in fresh {
            object.insert(key, value);
        }
    }
    Value::Object(object)
}

fn parse_mark(entry: &Value, index: usize) -> Result<Mark, String> {
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
    let mut corners = [Point::new(0.0, 0.0); 4];
    for (slot, point) in corners.iter_mut().zip(points) {
        let x = point.get("x").and_then(Value::as_f64);
        let y = point.get("y").and_then(Value::as_f64);
        let (Some(x), Some(y)) = (x, y) else {
            return Err(format!("mark {index}: each point needs numeric x and y"));
        };
        *slot = Point::new(x as f32, y as f32);
    }
    Ok(Mark {
        class_id,
        name,
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

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
	"annotation_mode": "detector",
	"image": { "scale": 1, "width": 640, "height": 480 },
	"mark": [
		{
			"class_idx": 1,
			"name": "car",
			"description": "car",
			"is_prediction": false,
			"annotation_kind": "detector",
			"points": [
				{"x": 64.0, "y": 48.0, "int_x": 64, "int_y": 48},
				{"x": 320.0, "y": 48.0, "int_x": 320, "int_y": 48},
				{"x": 320.0, "y": 240.0, "int_x": 320, "int_y": 240},
				{"x": 64.0, "y": 240.0, "int_x": 64, "int_y": 240}
			],
			"rect": {"x": 64.0, "y": 48.0, "w": 256.0, "h": 192.0, "int_x": 64, "int_y": 48, "int_w": 256, "int_h": 192},
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
            "\t\t\t\t{\"x\": 64.0, \"y\": 240.0, \"int_x\": 64, \"int_y\": 240}",
            "\t\t\t\t{\"x\": 64.0, \"y\": 240.0, \"int_x\": 64, \"int_y\": 240},\n\t\t\t\t{\"x\": 10.0, \"y\": 10.0, \"int_x\": 10, \"int_y\": 10}",
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
        assert_eq!(text, "0 0.2000000030 0.2000000030 0.6000000238 0.6000000238\n");
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("darknet-gui-ann-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }
}

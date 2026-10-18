//! A detection box in Darknet's normalized centre/size form.
//!
//! Values are stored exactly as the `.txt` sidecar holds them: `class cx cy w h`
//! with every geometry field normalized to `0..=1` against the image size.
//! Clamping happens on construction so a hand-edited file can never produce a
//! box outside the image.

use std::fmt::Write as _;

/// Smallest box side we keep. A click or a degenerate drag must not create a
/// mark that disappears when saved.
pub const MIN_SIDE: f32 = 1.0e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mark {
    pub class_id: usize,
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

impl Mark {
    /// Builds a clamped mark from two corner points in normalized space.
    pub fn from_corners(class_id: usize, x0: f32, y0: f32, x1: f32, y1: f32) -> Option<Self> {
        let (lo_x, hi_x) = (x0.min(x1).clamp(0.0, 1.0), x0.max(x1).clamp(0.0, 1.0));
        let (lo_y, hi_y) = (y0.min(y1).clamp(0.0, 1.0), y0.max(y1).clamp(0.0, 1.0));
        let (w, h) = (hi_x - lo_x, hi_y - lo_y);
        if w < MIN_SIDE || h < MIN_SIDE {
            return None;
        }
        Some(Self {
            class_id,
            cx: (lo_x + hi_x) * 0.5,
            cy: (lo_y + hi_y) * 0.5,
            w,
            h,
        })
    }

    pub fn left(&self) -> f32 {
        self.cx - self.w * 0.5
    }

    pub fn top(&self) -> f32 {
        self.cy - self.h * 0.5
    }

    pub fn right(&self) -> f32 {
        self.cx + self.w * 0.5
    }

    pub fn bottom(&self) -> f32 {
        self.cy + self.h * 0.5
    }

    /// True when the normalized point is inside the box. Used for click-to-select.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.left() && x <= self.right() && y >= self.top() && y <= self.bottom()
    }

    fn line(&self) -> String {
        let mut line = String::new();
        // 10 decimals match the label writer in `darknet-gui-project`'s plan
        // (DarkMark-compatible precision), avoiding 1-ulp churn on round trips.
        let _ = write!(
            line,
            "{} {:.10} {:.10} {:.10} {:.10}",
            self.class_id, self.cx, self.cy, self.w, self.h
        );
        line
    }
}

/// The editable state of one image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Annotation {
    pub marks: Vec<Mark>,
}

impl Annotation {
    pub fn is_empty(&self) -> bool {
        self.marks.is_empty()
    }

    /// Parses a YOLO `.txt` sidecar. Unknown or malformed lines are an error,
    /// not silently dropped: a label file that cannot be read must never be
    /// overwritten with a partial annotation.
    pub fn parse(text: &str, classes: usize) -> Result<Self, String> {
        let mut marks = Vec::new();
        for (index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != 5 {
                return Err(format!(
                    "line {}: expected 5 fields (class cx cy w h), got {}",
                    index + 1,
                    fields.len()
                ));
            }
            let class_id: usize = fields[0]
                .parse()
                .map_err(|_| format!("line {}: class `{}` is not an index", index + 1, fields[0]))?;
            if classes > 0 && class_id >= classes {
                return Err(format!(
                    "line {}: class {class_id} is outside 0..{classes}",
                    index + 1
                ));
            }
            let numbers: Vec<f32> = fields[1..]
                .iter()
                .map(|field| {
                    field
                        .parse::<f32>()
                        .map_err(|_| format!("line {}: `{field}` is not a number", index + 1))
                })
                .collect::<Result<_, _>>()?;
            let (cx, cy, w, h) = (numbers[0], numbers[1], numbers[2], numbers[3]);
            if ![cx, cy, w, h].iter().all(|v| v.is_finite()) {
                return Err(format!("line {}: non-finite coordinate", index + 1));
            }
            if w <= 0.0 || h <= 0.0 {
                return Err(format!("line {}: width and height must be positive", index + 1));
            }
            marks.push(Mark {
                class_id,
                cx: cx.clamp(0.0, 1.0),
                cy: cy.clamp(0.0, 1.0),
                w: w.clamp(0.0, 1.0),
                h: h.clamp(0.0, 1.0),
            });
        }
        Ok(Self { marks })
    }

    /// Serializes to the YOLO sidecar. An empty annotation yields an empty
    /// string, and the caller deletes the file rather than writing a blank one.
    pub fn to_yolo(&self) -> String {
        let mut out = String::new();
        for mark in &self.marks {
            out.push_str(&mark.line());
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_round_trip_through_the_centre_form() {
        let mark = Mark::from_corners(2, 0.1, 0.2, 0.5, 0.8).expect("valid box");
        assert!((mark.cx - 0.3).abs() < 1e-6);
        assert!((mark.cy - 0.5).abs() < 1e-6);
        assert!((mark.w - 0.4).abs() < 1e-6);
        assert!((mark.h - 0.6).abs() < 1e-6);
        assert!(mark.contains(0.2, 0.4));
        assert!(!mark.contains(0.9, 0.4));
    }

    #[test]
    fn degenerate_drags_do_not_create_marks() {
        assert!(Mark::from_corners(0, 0.5, 0.5, 0.5, 0.5).is_none());
        assert!(Mark::from_corners(0, 0.0, 0.0, 0.0001, 0.5).is_none());
    }

    #[test]
    fn yolo_text_round_trips_at_ten_decimals() {
        let annotation = Annotation {
            marks: vec![
                Mark { class_id: 0, cx: 0.25, cy: 0.5, w: 0.5, h: 0.25 },
                Mark { class_id: 3, cx: 0.75, cy: 0.125, w: 0.125, h: 0.75 },
            ],
        };
        let text = annotation.to_yolo();
        assert_eq!(
            text,
            "0 0.2500000000 0.5000000000 0.5000000000 0.2500000000\n\
             3 0.7500000000 0.1250000000 0.1250000000 0.7500000000\n"
        );
        assert_eq!(Annotation::parse(&text, 4).unwrap(), annotation);
    }

    #[test]
    fn malformed_lines_are_errors_not_drops() {
        assert!(Annotation::parse("0 0.5 0.5 0.1\n", 1).is_err());
        assert!(Annotation::parse("x 0.5 0.5 0.1 0.1\n", 1).is_err());
        assert!(Annotation::parse("5 0.5 0.5 0.1 0.1\n", 1).is_err());
        assert!(Annotation::parse("0 0.5 0.5 -0.1 0.1\n", 1).is_err());
        assert!(Annotation::parse("0 0.5 0.5 0.1 0.1\n", 1).is_ok());
    }
}

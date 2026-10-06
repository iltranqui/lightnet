//! Parsing for Darknet OBB label files (`class cx cy w h [front_x front_y]`).
//!
//! This module intentionally does not perform image augmentation or training
//! truth assignment. `has_front` distinguishes an omitted direction from an
//! explicitly annotated front point; absent points use the box center only as
//! a placeholder, matching Darknet's label reader.

use std::{fs, path::Path};

use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObbLabel {
    pub class_id: usize,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Front-point placeholder (the box center when `has_front` is false).
    pub front_x: f32,
    /// Front-point placeholder (the box center when `has_front` is false).
    pub front_y: f32,
    /// Whether the source row actually supplied both front-point coordinates.
    pub has_front: bool,
}

/// Load oriented labels from a Darknet text file.
///
/// Rows contain either five fields (`class cx cy w h`) or seven fields
/// (`class cx cy w h front_x front_y`). Finite coordinates are clamped to
/// `[0, 1]`, matching the source `read_boxes` behavior. If `classes` is given,
/// class IDs must be less than that count.
pub fn load_obb_labels(path: impl AsRef<Path>, classes: Option<usize>) -> Result<Vec<ObbLabel>> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)?;
    parse_obb_labels(&text, path, classes)
}

fn parse_obb_labels(text: &str, path: &Path, classes: Option<usize>) -> Result<Vec<ObbLabel>> {
    let mut labels = Vec::new();
    for (index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 5 && fields.len() != 7 {
            return Err(Error::Parse(format!(
                "failed to parse {} line {}: expected 5 or 7 OBB fields",
                path.display(),
                index + 1
            )));
        }

        let class_id = fields[0].parse::<i64>().map_err(|_| {
            Error::Parse(format!(
                "invalid class id in {} line {}",
                path.display(),
                index + 1
            ))
        })?;
        if class_id < 0 {
            return Err(Error::InvalidConfig(format!(
                "negative class id in {} line {}",
                path.display(),
                index + 1
            )));
        }
        let class_id = usize::try_from(class_id).map_err(|_| {
            Error::InvalidConfig(format!(
                "class id is out of range in {} line {}",
                path.display(),
                index + 1
            ))
        })?;
        if let Some(classes) = classes
            && class_id >= classes
        {
            return Err(Error::InvalidConfig(format!(
                "class id {} exceeds classes={} in {} line {}",
                class_id,
                classes,
                path.display(),
                index + 1
            )));
        }

        let x = parse_clamped_coordinate(fields[1], path, index + 1, "x")?;
        let y = parse_clamped_coordinate(fields[2], path, index + 1, "y")?;
        let w = parse_clamped_coordinate(fields[3], path, index + 1, "w")?;
        let h = parse_clamped_coordinate(fields[4], path, index + 1, "h")?;
        if w <= 0.0 || h <= 0.0 {
            return Err(Error::InvalidConfig(format!(
                "non-positive OBB size in {} line {}",
                path.display(),
                index + 1
            )));
        }

        let has_front = fields.len() == 7;
        let (front_x, front_y) = if has_front {
            (
                parse_clamped_coordinate(fields[5], path, index + 1, "front_x")?,
                parse_clamped_coordinate(fields[6], path, index + 1, "front_y")?,
            )
        } else {
            (x, y)
        };
        labels.push(ObbLabel {
            class_id,
            x,
            y,
            w,
            h,
            front_x,
            front_y,
            has_front,
        });
    }
    Ok(labels)
}

fn parse_clamped_coordinate(value: &str, path: &Path, line: usize, field: &str) -> Result<f32> {
    let value = value
        .parse::<f32>()
        .map_err(|_| Error::Parse(format!("invalid {field} in {} line {line}", path.display())))?;
    if !value.is_finite() {
        return Err(Error::InvalidConfig(format!(
            "non-finite {field} in {} line {line}",
            path.display()
        )));
    }
    Ok(value.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::{ObbLabel, load_obb_labels, parse_obb_labels};
    use crate::error::Error;

    fn parse(text: &str, classes: Option<usize>) -> crate::Result<Vec<ObbLabel>> {
        parse_obb_labels(text, Path::new("labels.txt"), classes)
    }

    #[test]
    fn parses_optional_front_points_and_skips_blank_lines() {
        let labels = parse(
            "\n  0 0.5 0.25 0.1 0.2 0.6 0.3  \n\t\n1 0.25 0.75 0.2 0.4\n",
            Some(2),
        )
        .expect("valid OBB labels");
        assert_eq!(
            labels,
            vec![
                ObbLabel {
                    class_id: 0,
                    x: 0.5,
                    y: 0.25,
                    w: 0.1,
                    h: 0.2,
                    front_x: 0.6,
                    front_y: 0.3,
                    has_front: true,
                },
                ObbLabel {
                    class_id: 1,
                    x: 0.25,
                    y: 0.75,
                    w: 0.2,
                    h: 0.4,
                    front_x: 0.25,
                    front_y: 0.75,
                    has_front: false,
                },
            ]
        );
    }

    #[test]
    fn clamps_finite_fields_like_source_reader() {
        let labels =
            parse("0 -0.2 1.2 1.5 0.5 -0.5 1.4\n", Some(1)).expect("finite OBB fields are clamped");
        assert_eq!(labels[0].x, 0.0);
        assert_eq!(labels[0].y, 1.0);
        assert_eq!(labels[0].w, 1.0);
        assert_eq!(labels[0].h, 0.5);
        assert_eq!(labels[0].front_x, 0.0);
        assert_eq!(labels[0].front_y, 1.0);
        assert!(labels[0].has_front);
    }

    #[test]
    fn empty_files_and_whitespace_only_files_have_no_labels() {
        assert!(parse("", None).expect("empty file").is_empty());
        assert!(
            parse(" \t\n\r\n", None)
                .expect("whitespace-only file")
                .is_empty()
        );
    }

    #[test]
    fn rejects_bad_field_counts_classes_sizes_and_non_finite_values() {
        let cases = [
            ("0 0.5 0.5 0.2 0.2 0.7\n", "expected 5 or 7"),
            ("0 0.5 0.5 0.2 0.2 0.7 0.8 9\n", "expected 5 or 7"),
            ("-1 0.5 0.5 0.2 0.2\n", "negative class id"),
            ("2 0.5 0.5 0.2 0.2\n", "exceeds classes"),
            ("0 NaN 0.5 0.2 0.2\n", "non-finite x"),
            ("0 0.5 0.5 0.0 0.2\n", "non-positive OBB size"),
            ("0 0.5 0.5 0.2 0.2 inf 0.8\n", "non-finite front_x"),
        ];
        for (text, expected) in cases {
            let error = parse(text, Some(2)).expect_err("invalid OBB label must fail");
            assert!(format!("{error}").contains(expected), "{text:?}: {error}");
        }
    }

    #[test]
    fn file_loader_reads_labels_and_reports_io_errors() {
        let path = std::env::temp_dir().join(format!(
            "darknet-obb-labels-{}-{}.txt",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::write(&path, "0 0.5 0.5 0.2 0.3\n").expect("label file writes");
        let labels = load_obb_labels(&path, Some(1)).expect("label file loads");
        fs::remove_file(&path).expect("label file removes");
        assert_eq!(labels.len(), 1);
        assert!(!labels[0].has_front);

        let missing = path.with_extension("missing");
        assert!(matches!(load_obb_labels(&missing, None), Err(Error::Io(_))));
    }
}

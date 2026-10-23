//! Bulk editing actions: interpolation, clipboard, snapping, homography.

use image::{GrayImage, RgbaImage};

use crate::annotation::{Annotation, Mark, Point};

/// Linear interpolation between two annotations by index (`t = 0` is `a`,
/// `t = 1` is `b`). Marks beyond the shorter list are dropped.
pub fn interpolate(a: &Annotation, b: &Annotation, t: f32) -> Annotation {
    let t = t.clamp(0.0, 1.0);
    let mut out = Annotation::new(a.image_width.max(b.image_width), a.image_height.max(b.image_height));
    let count = a.marks.len().min(b.marks.len());
    for index in 0..count {
        let (first, second) = (&a.marks[index], &b.marks[index]);
        let lerp = |p: Point, q: Point| Point::new(p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t);
        let mut points = first.points;
        for (slot, second_point) in points.iter_mut().zip(second.points) {
            *slot = lerp(*slot, second_point);
        }
        out.marks.push(Mark {
            class_id: first.class_id,
            name: first.name.clone(),
            points,
            is_prediction: false,
            confidence: 1.0,
        });
    }
    out
}

/// Clipboard payload: the mark list as JSON, with the image size so a paste into
/// a differently sized image can be scaled.
pub fn clipboard_copy(annotation: &Annotation) -> String {
    let marks: Vec<serde_json::Value> = annotation
        .real_marks()
        .map(|mark| {
            serde_json::json!({
                "class": mark.class_id,
                "points": mark.points.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
            })
        })
        .collect();
    serde_json::json!({
        "width": annotation.image_width,
        "height": annotation.image_height,
        "marks": marks,
    })
    .to_string()
}

pub fn clipboard_paste(text: &str, width: u32, height: u32, classes: &[String]) -> Result<Vec<Mark>, String> {
    let root: serde_json::Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
    let source_width = root.get("width").and_then(serde_json::Value::as_f64).unwrap_or(width as f64);
    let source_height = root.get("height").and_then(serde_json::Value::as_f64).unwrap_or(height as f64);
    let (sx, sy) = (
        width as f64 / source_width.max(1.0),
        height as f64 / source_height.max(1.0),
    );
    let mut marks = Vec::new();
    for entry in root
        .get("marks")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "clipboard has no marks".to_owned())?
    {
        let class_id = entry
            .get("class")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| "clipboard mark without class".to_owned())? as usize;
        let points = entry
            .get("points")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "clipboard mark without points".to_owned())?;
        let mut corners = [Point::new(0.0, 0.0); 4];
        for (slot, point) in corners.iter_mut().zip(points) {
            let pair = point.as_array().ok_or_else(|| "bad point".to_owned())?;
            let x = pair.first().and_then(serde_json::Value::as_f64).unwrap_or(0.0) * sx;
            let y = pair.get(1).and_then(serde_json::Value::as_f64).unwrap_or(0.0) * sy;
            *slot = Point::new(x as f32, y as f32);
        }
        marks.push(Mark {
            class_id,
            name: classes.get(class_id).cloned().unwrap_or_else(|| format!("class {class_id}")),
            points: corners,
            is_prediction: false,
            confidence: 1.0,
        });
    }
    Ok(marks)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapScope {
    All,
    Horizontal,
    Vertical,
}

/// Translates each box so its strongest edge lands on the strongest gradient
/// within `radius` pixels. Translating the whole box keeps its size, so a snap
/// can never collapse a mark. Boxes with no gradient in range are untouched.
pub fn snap(annotations: &mut Annotation, image: &GrayImage, radius: u32, scope: SnapScope) -> usize {
    let (width, height) = image.dimensions();
    let gradient = |x: u32, y: u32, vertical: bool| -> f32 {
        if vertical {
            let (x0, x1) = (x.saturating_sub(1), (x + 1).min(width - 1));
            (image.get_pixel(x1, y)[0] as f32 - image.get_pixel(x0, y)[0] as f32).abs()
        } else {
            let (y0, y1) = (y.saturating_sub(1), (y + 1).min(height - 1));
            (image.get_pixel(x, y1)[0] as f32 - image.get_pixel(x, y0)[0] as f32).abs()
        }
    };
    let mut moved = 0;
    for mark in annotations.marks.iter_mut() {
        if mark.is_prediction {
            continue;
        }
        let mut best: Option<(f32, f32, bool)> = None; // (gradient, delta, vertical)
        for (edge, vertical) in [(0, true), (1, false), (2, true), (3, false)] {
            let wanted = matches!(
                (scope, vertical),
                (SnapScope::All, _) | (SnapScope::Vertical, true) | (SnapScope::Horizontal, false)
            );
            if !wanted {
                continue;
            }
            let reference = mark.points[edge];
            let (axis, length) = if vertical {
                (reference.x.round() as i64, width)
            } else {
                (reference.y.round() as i64, height)
            };
            let start = (axis - radius as i64).max(0);
            let end = (axis + radius as i64).min(length as i64 - 1);
            for candidate in start..=end {
                let value = if vertical {
                    gradient(candidate as u32, (mark.top() as u32).min(height - 1), true)
                } else {
                    gradient((mark.left() as u32).min(width - 1), candidate as u32, false)
                };
                if value > 0.0
                    && best.is_none_or(|(best_value, _, _)| value > best_value)
                {
                    best = Some((value, (candidate - axis) as f32, vertical));
                }
            }
        }
        if let Some((value, delta, vertical)) = best
            && value > 0.0
            && delta != 0.0
        {
            for point in &mut mark.points {
                if vertical {
                    point.x += delta;
                } else {
                    point.y += delta;
                }
            }
            moved += 1;
        }
    }
    moved
}

/// Four-point projectivity from the unit square to `corners`, applied to the
/// image. Identity corners return a copy unchanged.
pub fn homography(corners: [Point; 4], image: &RgbaImage) -> RgbaImage {
    let identity = [
        Point::new(0.0, 0.0),
        Point::new(image.width() as f32, 0.0),
        Point::new(image.width() as f32, image.height() as f32),
        Point::new(0.0, image.height() as f32),
    ];
    if corners
        .iter()
        .zip(identity)
        .all(|(a, b)| (a.x - b.x).abs() < 1e-3 && (a.y - b.y).abs() < 1e-3)
    {
        return image.clone();
    }
    let Some(matrix) = solve_homography(&identity, &corners) else {
        return image.clone();
    };
    let mut out = RgbaImage::new(image.width(), image.height());
    for y in 0..image.height() {
        for x in 0..image.width() {
            let (sx, sy) = apply(&matrix, x as f32, y as f32);
            let (sx, sy) = (sx.round() as i64, sy.round() as i64);
            if sx < 0 || sy < 0 || sx >= image.width() as i64 || sy >= image.height() as i64 {
                continue;
            }
            out.put_pixel(x, y, *image.get_pixel(sx as u32, sy as u32));
        }
    }
    out
}

fn apply(m: &[f64; 9], x: f32, y: f32) -> (f32, f32) {
    let (x, y) = (x as f64, y as f64);
    let denominator = m[6] * x + m[7] * y + m[8];
    (
        ((m[0] * x + m[1] * y + m[2]) / denominator) as f32,
        ((m[3] * x + m[4] * y + m[5]) / denominator) as f32,
    )
}

/// Solves the 8-unknown projectivity mapping `from` to `to` (Gaussian
/// elimination with partial pivoting).
fn solve_homography(from: &[Point; 4], to: &[Point; 4]) -> Option<[f64; 9]> {
    let mut a = [[0.0f64; 9]; 8];
    for index in 0..4 {
        let (x, y) = (from[index].x as f64, from[index].y as f64);
        let (u, v) = (to[index].x as f64, to[index].y as f64);
        a[index * 2] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[index * 2 + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    for column in 0..8 {
        let pivot = (column..8).max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))?;
        if a[pivot][column].abs() < 1e-12 {
            return None;
        }
        a.swap(column, pivot);
        let scale = a[column][column];
        for slot in a[column].iter_mut() {
            *slot /= scale;
        }
        for row in 0..8 {
            if row == column {
                continue;
            }
            let factor = a[row][column];
            for k in 0..9 {
                a[row][k] -= factor * a[column][k];
            }
        }
    }
    Some([
        a[0][8], a[1][8], a[2][8], a[3][8], a[4][8], a[5][8], a[6][8], a[7][8], 1.0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_midpoint_is_halfway() {
        let mut a = Annotation::new(100, 100);
        let mut b = Annotation::new(100, 100);
        a.marks.push(Mark::from_corners(0, "a", 0.0, 0.0, 10.0, 10.0));
        b.marks.push(Mark::from_corners(0, "a", 20.0, 20.0, 40.0, 40.0));
        let mid = interpolate(&a, &b, 0.5);
        assert_eq!(mid.marks[0].points[0], Point::new(10.0, 10.0));
        assert_eq!(mid.marks[0].points[2], Point::new(25.0, 25.0));
    }

    #[test]
    fn clipboard_round_trips_and_rescales() {
        let mut source = Annotation::new(100, 100);
        source.marks.push(Mark::from_corners(1, "b", 10.0, 10.0, 30.0, 30.0));
        let text = clipboard_copy(&source);
        let marks = clipboard_paste(&text, 200, 200, &["a".into(), "b".into()]).unwrap();
        assert_eq!(marks[0].class_id, 1);
        assert_eq!(marks[0].points[0], Point::new(20.0, 20.0));
    }

    #[test]
    fn homography_identity_is_a_noop() {
        let image = RgbaImage::from_pixel(4, 3, image::Rgba([1, 2, 3, 255]));
        let out = homography(
            [
                Point::new(0.0, 0.0),
                Point::new(4.0, 0.0),
                Point::new(4.0, 3.0),
                Point::new(0.0, 3.0),
            ],
            &image,
        );
        assert_eq!(out, image);
    }

    #[test]
    fn snap_moves_an_edge_to_the_step() {
        let mut image = GrayImage::new(20, 10);
        for y in 0..10 {
            for x in 0..20 {
                image.put_pixel(x, y, image::Luma([if x >= 10 { 255 } else { 0 }]));
            }
        }
        let mut annotation = Annotation::new(20, 10);
        annotation.marks.push(Mark::from_corners(0, "a", 7.0, 2.0, 15.0, 8.0));
        snap(&mut annotation, &image, 5, SnapScope::Vertical);
        assert!((9.0..=10.0).contains(&annotation.marks[0].points[0].x));
    }
}

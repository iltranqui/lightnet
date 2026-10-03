//! Minimal training-curve renderer (`chart.png`).
//!
//! C++ Darknet writes a loss/mAP chart next to the checkpoints.  The Rust
//! trainer writes the same file so existing tooling that reads `chart.png`
//! keeps working.  This renderer is intentionally dependency-light: it uses
//! the already-present `image` crate and a built-in 5x7 bitmap font for the
//! numeric tick labels (no font asset to ship).
use std::path::Path;

use image::{Rgba, RgbaImage};

use crate::error::Result;

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 400;
const LEFT: i32 = 64;
const RIGHT: i32 = 24;
const TOP: i32 = 20;
const BOTTOM: i32 = 36;

const BLACK: Rgba<u8> = Rgba([0, 0, 0, 255]);
const GRID: Rgba<u8> = Rgba([210, 210, 210, 255]);
const LOSS: Rgba<u8> = Rgba([200, 0, 0, 255]);
const MAP: Rgba<u8> = Rgba([0, 140, 0, 255]);
const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);

/// One logged training iteration.
#[derive(Clone, Copy, Debug)]
pub struct ChartPoint {
    pub iteration: usize,
    pub avg_loss: f32,
    pub map: Option<f64>,
}

/// Render `chart.png` with the average-loss curve (red) and, when available,
/// the mAP curve (green, its own right-hand scale).  Numeric tick labels use a
/// built-in 5x7 bitmap font; there is no legend text (curves are colour-coded).
pub fn write_training_chart(path: &Path, points: &[ChartPoint]) -> Result<()> {
    let mut image = RgbaImage::from_pixel(WIDTH, HEIGHT, WHITE);
    draw_frame(&mut image);
    if points.is_empty() {
        return Ok(image.save(path).map_err(chart_error)?);
    }

    let plot_w = (WIDTH as i32 - LEFT - RIGHT).max(1) as f32;
    let plot_h = (HEIGHT as i32 - TOP - BOTTOM).max(1) as f32;

    let first = points.first().map(|p| p.iteration).unwrap_or(0);
    let last = points.last().map(|p| p.iteration).unwrap_or(first);
    let (x0, x1) = (first as f32, (last.max(first + 1)) as f32);

    let (loss_lo, loss_hi) = range(points.iter().map(|p| p.avg_loss as f64));
    draw_grid_labels(&mut image, loss_lo, loss_hi, plot_h);

    let to_x = |iteration: usize| -> f32 {
        LEFT as f32 + (iteration as f32 - x0) / (x1 - x0) * plot_w
    };
    let to_y = |value: f64, lo: f64, hi: f64| -> f32 {
        let t = if hi > lo { (value - lo) / (hi - lo) } else { 0.5 };
        TOP as f32 + (1.0 - t as f32) * plot_h
    };

    let loss_poly: Vec<(f32, f32)> = points
        .iter()
        .map(|p| (to_x(p.iteration), to_y(p.avg_loss as f64, loss_lo, loss_hi)))
        .collect();
    draw_polyline(&mut image, &loss_poly, LOSS);

    let map_points: Vec<ChartPoint> = points.iter().copied().filter(|p| p.map.is_some()).collect();
    if map_points.len() >= 2 {
        let (map_lo, map_hi) = range(map_points.iter().filter_map(|p| p.map));
        let map_poly: Vec<(f32, f32)> = map_points
            .iter()
            .map(|p| (to_x(p.iteration), to_y(p.map.unwrap_or(0.0), map_lo, map_hi)))
            .collect();
        draw_polyline(&mut image, &map_poly, MAP);
    }

    // x axis labels: first and last iteration.
    draw_number(&mut image, LEFT, HEIGHT as i32 - 28, first.to_string().as_str(), 1);
    let last_label = last.to_string();
    let w = text_width(&last_label, 1);
    draw_number(&mut image, WIDTH as i32 - RIGHT - w, HEIGHT as i32 - 28, &last_label, 1);
    image.save(path).map_err(chart_error)?;
    Ok(())
}

fn chart_error(err: image::ImageError) -> crate::error::Error {
    crate::error::Error::Parse(format!("failed to write chart: {err}"))
}

fn draw_frame(image: &mut RgbaImage) {
    for x in LEFT..(WIDTH as i32 - RIGHT) {
        image.put_pixel(x as u32, TOP as u32, BLACK);
        image.put_pixel(x as u32, (HEIGHT as i32 - BOTTOM) as u32, BLACK);
    }
    for y in TOP..(HEIGHT as i32 - BOTTOM) {
        image.put_pixel(LEFT as u32, y as u32, BLACK);
    }
}

fn draw_grid_labels(image: &mut RgbaImage, lo: f64, hi: f64, plot_h: f32) {
    for tick in 0..=4 {
        let t = tick as f64 / 4.0;
        let y = (TOP as f32 + (1.0 - t as f32) * plot_h).round() as i32;
        for x in LEFT..(WIDTH as i32 - RIGHT) {
            image.put_pixel(x as u32, y as u32, GRID);
        }
        let value = lo + t * (hi - lo);
        let label = format!("{value:.3}");
        let w = text_width(&label, 1);
        draw_number(image, LEFT - w - 6, y - 3, &label, 1);
    }
}

fn range(values: impl IntoIterator<Item = f64>) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut any = false;
    for value in values {
        if value.is_finite() {
            any = true;
            lo = lo.min(value);
            hi = hi.max(value);
        }
    }
    if !any {
        return (0.0, 1.0);
    }
    if (hi - lo).abs() < 1e-9 {
        let pad = lo.abs().max(1.0) * 0.05;
        return (lo - pad, hi + pad);
    }
    let pad = (hi - lo) * 0.05;
    (lo - pad, hi + pad)
}

fn draw_polyline(image: &mut RgbaImage, points: &[(f32, f32)], color: Rgba<u8>) {
    for pair in points.windows(2) {
        draw_line(image, pair[0], pair[1], color);
    }
}

fn draw_line(image: &mut RgbaImage, (x0, y0): (f32, f32), (x1, y1): (f32, f32), color: Rgba<u8>) {
    let steps = ((x1 - x0).abs().max((y1 - y0).abs())).ceil().max(1.0) as usize;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let x = (x0 + (x1 - x0) * t).round() as i32;
        let y = (y0 + (y1 - y0) * t).round() as i32;
        if x >= 0 && y >= 0 && (x as u32) < WIDTH && (y as u32) < HEIGHT {
            image.put_pixel(x as u32, y as u32, color);
        }
    }
}

/// 5x7 glyphs for the characters used by tick labels: digits, `.`, `-`, `e`, `+`.
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        _ => [0; 7],
    }
}

fn text_width(text: &str, scale: u32) -> i32 {
    (text.chars().count() as i32) * (6 * scale as i32) - (scale as i32)
}

fn draw_number(image: &mut RgbaImage, x: i32, y: i32, text: &str, scale: u32) {
    let mut cursor = x;
    for c in text.chars() {
        let rows = glyph(c);
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = cursor + (col as u32 * scale + dx) as i32;
                            let py = y + (row as u32 * scale + dy) as i32;
                            if px >= 0 && py >= 0 && (px as u32) < WIDTH && (py as u32) < HEIGHT {
                                image.put_pixel(px as u32, py as u32, BLACK);
                            }
                        }
                    }
                }
            }
        }
        cursor += (6 * scale) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_loss_and_map_chart() {
        let dir = std::env::temp_dir().join(format!("darknet_chart_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chart.png");
        let points = vec![
            ChartPoint { iteration: 1, avg_loss: 10.0, map: None },
            ChartPoint { iteration: 2, avg_loss: 5.0, map: Some(0.1) },
            ChartPoint { iteration: 3, avg_loss: 2.0, map: Some(0.3) },
        ];
        write_training_chart(&path, &points).expect("chart");
        let image = image::open(&path).expect("read chart");
        assert_eq!(image.width(), WIDTH);
        assert_eq!(image.height(), HEIGHT);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn render_empty_chart() {
        let dir = std::env::temp_dir().join(format!("darknet_chart_empty_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("chart.png");
        write_training_chart(&path, &[]).expect("empty chart");
        assert!(path.exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}

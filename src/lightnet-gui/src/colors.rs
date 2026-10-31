//! Colours shared by the canvas, the scrollfield and the popup menu.
//!
//! The class palette mirrors DarkMark's `annotation_colours` idea
//! (`ScrollField.cpp:212-219`): a mark is drawn in `palette[class_idx % len]`.
//! The row colours are the three ScrollField states (`ScrollField.cpp:180-206`).

use eframe::egui::Color32;

/// DarkMark tints class `n` with `annotation_colours[n % size]`; the port uses a
/// fixed 10-colour palette with the same wrap-around rule.
pub const CLASS_PALETTE: [Color32; 10] = [
    Color32::from_rgb(0x4E, 0x9A, 0xE8),
    Color32::from_rgb(0xE8, 0x6A, 0x4E),
    Color32::from_rgb(0x5E, 0xC8, 0x7A),
    Color32::from_rgb(0xE8, 0xC8, 0x4E),
    Color32::from_rgb(0xB8, 0x6A, 0xE8),
    Color32::from_rgb(0x4E, 0xE8, 0xD8),
    Color32::from_rgb(0xE8, 0x4E, 0x9A),
    Color32::from_rgb(0x9A, 0xE8, 0x4E),
    Color32::from_rgb(0xD8, 0x8E, 0x4E),
    Color32::from_rgb(0x8E, 0x9A, 0xE8),
];

pub fn class_color(class_id: usize) -> Color32 {
    CLASS_PALETTE[class_id % CLASS_PALETTE.len()]
}

/// Row base colour: the image has a `.json` with marks (`ScrollField.cpp:180-184`).
pub const ROW_BLACK: Color32 = Color32::BLACK;
/// Row base colour: no `.json` sidecar for this image.
pub const ROW_GREY: Color32 = Color32::from_rgb(32, 32, 32);
/// Row base colour: JSON parse error, or a `.json` with zero marks that is not
/// flagged `completely_empty` (`ScrollField.cpp:202-206`).
pub const ROW_RED: Color32 = Color32::from_rgb(0xC8, 0x20, 0x20);

/// Predicted mark colour: DarkMark draws predictions in a distinct colour from
/// the class palette; this yellow keeps them readable next to class colours.
pub const PREDICTION_COLOR: Color32 = Color32::from_rgb(0xE8, 0xC8, 0x4E);

//! On-canvas status text and the hover crosshair.
//!
//! DarkMark has no status panels: the canvas draws everything
//! (`gui_darkmark_layout.md`, "DarkMark layout reference"). Messages are JUCE
//! bubble messages: white bubble, black text, shown for four seconds
//! (`DMContent::show_message`, `DMContent.cpp:4564-4589`). The crosshair is a
//! 1 px vertical and horizontal line through the pointer
//! (`CrosshairComponent.cpp:119-137`).

use eframe::egui::{self, Align2, Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};

/// Hover crosshair colour. DarkMark tints it with the most-recent-class colour
/// (`crosshair_colour`, `CrosshairComponent.cpp:119-137`); this port keeps the
/// thin red lines the layout plan asks for.
pub const CROSSHAIR_COLOUR: Color32 = Color32::from_rgb(0xF0, 0x20, 0x20);

/// How long a status message stays on the canvas, in seconds
/// (`show_message` passes 4000 ms).
pub const MESSAGE_SECONDS: f32 = 4.0;

/// Draws the crosshair through `at`, spanning the whole canvas.
pub fn crosshair(painter: &Painter, rect: Rect, at: Pos2) {
    let stroke = Stroke::new(1.0_f32, CROSSHAIR_COLOUR);
    painter.line_segment(
        [Pos2::new(rect.left(), at.y), Pos2::new(rect.right(), at.y)],
        stroke,
    );
    painter.line_segment(
        [Pos2::new(at.x, rect.top()), Pos2::new(at.x, rect.bottom())],
        stroke,
    );
}

/// DarkMark-style status block in the canvas corner: filename, position, zoom,
/// mark and prediction counts, dirty flag.
pub fn status(painter: &Painter, rect: Rect, lines: &[String]) {
    let font = FontId::proportional(12.0);
    let mut y = rect.top() + 4.0;
    let x = rect.left() + 6.0;
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let galley = painter.layout_no_wrap(line.clone(), font.clone(), Color32::WHITE);
        let size = galley.size() + Vec2::new(8.0, 3.0);
        let background = Rect::from_min_size(Pos2::new(x - 4.0, y - 1.0), size);
        painter.rect_filled(background, 3.0, Color32::from_black_alpha(150));
        painter.galley(Pos2::new(x, y), galley, Color32::WHITE);
        y += size.y + 3.0;
    }
}

/// DarkMark's bubble message: white rounded bubble with black text, placed at
/// the top centre of the canvas for [`MESSAGE_SECONDS`] seconds. Errors get red
/// text so a refused action is distinguishable from a confirmation.
pub fn bubble(painter: &Painter, rect: Rect, text: &str, error: bool) {
    if text.is_empty() {
        return;
    }
    let colour = if error {
        Color32::from_rgb(0xB0, 0x00, 0x00)
    } else {
        Color32::BLACK
    };
    let galley = painter.layout_no_wrap(text.to_owned(), FontId::proportional(13.0), colour);
    let size = galley.size() + Vec2::new(16.0, 8.0);
    let top_left = Pos2::new((rect.center().x - size.x * 0.5).max(rect.left() + 2.0), rect.top() + 4.0);
    let background = Rect::from_min_size(top_left, size);
    painter.rect_filled(background, 5.0, Color32::WHITE);
    painter.rect_stroke(
        background,
        5.0,
        Stroke::new(1.0_f32, Color32::from_gray(90)),
        egui::StrokeKind::Outside,
    );
    painter.galley(background.min + Vec2::new(8.0, 4.0), galley, colour);
}

/// Small helper used by the canvas to place a label at a mark corner.
pub fn mark_label(painter: &Painter, at: Pos2, text: impl Into<String>, colour: Color32) {
    painter.text(
        at + Vec2::new(2.0, -2.0),
        Align2::LEFT_BOTTOM,
        text.into(),
        FontId::proportional(13.0),
        colour,
    );
}

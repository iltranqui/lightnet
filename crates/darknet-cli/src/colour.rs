//! Darknet-compatible ANSI colour output for the CLI.
//!
//! Mirrors `src-lib/darknet_format_and_colour.{hpp,cpp}` (the `EColour` enum,
//! the `ansi_colours` escape table and `in_colour`) plus `Darknet::format_loss`
//! and `Darknet::format_map_accuracy` from `src-lib/darknet_utils.cpp`, so the
//! `detector train` console line can be coloured exactly like the C++ trainer.
//!
//! Output is plain text unless colour is enabled.  Unlike C++ (which defaults
//! to on), colour is auto-enabled only when stdout is a terminal, so piped
//! output and captured test output stay free of escape codes.  `-colour` and
//! `-nocolour` force it, and `NO_COLOR` disables the auto-enable.

use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, Ordering};

/// Darknet's `EColour`, in the same order as the C++ enum so the index math
/// below stays trivial to audit.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EColour {
    Normal = 0,
    Black,
    Red,
    Green,
    Brown,
    Blue,
    Magenta,
    Cyan,
    LightGrey,
    DarkGrey,
    BrightRed,
    BrightGreen,
    Yellow,
    BrightBlue,
    BrightMagenta,
    BrightCyan,
    BrightWhite,
}

/// Verbatim copy of the `ansi_colours` table in `darknet_format_and_colour.cpp`.
const ANSI: [&str; 17] = [
    "\x1b[0m",    // Normal
    "\x1b[0;30m", // Black
    "\x1b[0;31m", // Red
    "\x1b[0;32m", // Green
    "\x1b[0;33m", // Brown
    "\x1b[0;34m", // Blue
    "\x1b[0;35m", // Magenta
    "\x1b[0;36m", // Cyan
    "\x1b[0;37m", // LightGrey
    "\x1b[1;30m", // DarkGrey
    "\x1b[1;31m", // BrightRed
    "\x1b[1;32m", // BrightGreen
    "\x1b[1;33m", // Yellow
    "\x1b[1;34m", // BrightBlue
    "\x1b[1;35m", // BrightMagenta
    "\x1b[1;36m", // BrightCyan
    "\x1b[1;37m", // BrightWhite
];

static COLOUR_ENABLED: AtomicBool = AtomicBool::new(false);

/// Serializes tests that mutate the process-global colour flag, including
/// tests in `main.rs` (through `colour::TEST_LOCK`).
#[cfg(test)]
pub(crate) static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Choose whether escape codes are emitted.
///
/// `None` auto-detects: enabled only when `NO_COLOR` is unset and stdout is a
/// terminal.  `Some(true)` / `Some(false)` force it on / off (`-colour`,
/// `-nocolour`).
pub fn configure(choice: Option<bool>) {
    let enabled = match choice {
        Some(value) => value,
        None => std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal(),
    };
    COLOUR_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    COLOUR_ENABLED.load(Ordering::Relaxed)
}

/// `Darknet::in_colour`: wrap `msg` in the colour's escape codes when enabled.
pub fn in_colour(colour: EColour, msg: &str) -> String {
    if enabled() {
        format!(
            "{}{}{}",
            ANSI[colour as usize],
            msg,
            ANSI[EColour::Normal as usize]
        )
    } else {
        msg.to_owned()
    }
}

/// Colour selected by `Darknet::format_loss`.
pub fn loss_colour(loss: f64) -> EColour {
    if loss < 0.0 || loss >= 1000.0 || !loss.is_finite() {
        EColour::BrightRed
    } else if loss >= 20.0 {
        EColour::Red
    } else if loss >= 4.0 {
        EColour::Cyan
    } else {
        EColour::BrightCyan
    }
}

/// `Darknet::format_loss`: fixed 3-decimal loss, coloured by magnitude.
pub fn format_loss(loss: f64) -> String {
    in_colour(loss_colour(loss), &format!("{loss:.3}"))
}

/// Colour selected by `Darknet::format_map_accuracy`.  `accuracy` is a 0..1
/// fraction; `-1.0` means "not calculated yet".
pub fn map_accuracy_colour(accuracy: f32) -> EColour {
    if accuracy < 0.5 || !accuracy.is_finite() {
        EColour::BrightRed
    } else if accuracy < 0.6 {
        EColour::Red
    } else if accuracy < 0.7 {
        EColour::Blue
    } else {
        EColour::BrightBlue
    }
}

/// `Darknet::format_map_accuracy`: percentage with 2 decimals, coloured by
/// accuracy.  The special value `-1.0` prints `none` uncoloured.
pub fn format_map_accuracy(accuracy: f32) -> String {
    if accuracy == -1.0 {
        return in_colour(EColour::Normal, "none");
    }
    in_colour(
        map_accuracy_colour(accuracy),
        &format!("{:.2}%", 100.0 * accuracy),
    )
}

/// Right-aligned integer in a fixed colour (`format_in_colour(int, colour, len)`).
pub fn format_int(value: usize, colour: EColour, len: usize) -> String {
    let text = value.to_string();
    format!(
        "{}{}",
        " ".repeat(len.saturating_sub(text.len())),
        in_colour(colour, &text)
    )
}

/// `format_in_colour(float, len)`: 4 decimals, right-aligned, coloured by value
/// (values > 1 are taken as percentages), as in the C++ mAP table.
pub fn format_ratio(value: f32, len: usize) -> String {
    let scale = if value > 1.0 { value / 100.0 } else { value };
    let colour = if scale > 1.0 {
        EColour::Normal
    } else if scale >= 0.85 {
        EColour::BrightGreen
    } else if scale >= 0.70 {
        EColour::BrightCyan
    } else if scale >= 0.55 {
        EColour::BrightBlue
    } else if scale >= 0.40 {
        EColour::BrightMagenta
    } else {
        EColour::BrightRed
    };
    let shown = if value.is_normal() { value } else { 0.0 };
    let text = format!("{shown:.4}");
    format!(
        "{}{}",
        " ".repeat(len.saturating_sub(text.len())),
        in_colour(colour, &text)
    )
}

/// Left-aligned, truncated name (`format_in_colour(string, colour, len)`).
pub fn format_name(name: &str, colour: EColour, len: usize) -> String {
    let mut text: String = name.to_owned();
    if text.chars().count() > len {
        text = text.chars().take(len - 1).collect::<String>() + "+";
    }
    let pad = " ".repeat(len.saturating_sub(text.chars().count()));
    format!("{}{pad}", in_colour(colour, &text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ansi_table_matches_cpp_bytes() {
        assert_eq!(ANSI[EColour::Normal as usize], "\x1b[0m");
        assert_eq!(ANSI[EColour::BrightWhite as usize], "\x1b[1;37m");
        assert_eq!(ANSI[EColour::BrightGreen as usize], "\x1b[1;32m");
        assert_eq!(ANSI[EColour::Cyan as usize], "\x1b[0;36m");
        assert_eq!(ANSI.len(), 17);
    }

    #[test]
    fn in_colour_wraps_only_when_enabled() {
        let _guard = TEST_LOCK.lock().unwrap();
        COLOUR_ENABLED.store(false, Ordering::Relaxed);
        assert_eq!(in_colour(EColour::BrightWhite, "7"), "7");
        COLOUR_ENABLED.store(true, Ordering::Relaxed);
        assert_eq!(in_colour(EColour::BrightWhite, "7"), "\x1b[1;37m7\x1b[0m");
        COLOUR_ENABLED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn format_loss_thresholds_and_precision_match_cpp() {
        let _guard = TEST_LOCK.lock().unwrap();
        COLOUR_ENABLED.store(false, Ordering::Relaxed);
        // Magnitude/colour boundaries from darknet_utils.cpp:format_loss.
        assert_eq!(loss_colour(-1.0), EColour::BrightRed);
        assert_eq!(loss_colour(1000.0), EColour::BrightRed);
        assert_eq!(loss_colour(f64::NAN), EColour::BrightRed);
        assert_eq!(loss_colour(20.0), EColour::Red);
        assert_eq!(loss_colour(19.999), EColour::Cyan);
        assert_eq!(loss_colour(4.0), EColour::Cyan);
        assert_eq!(loss_colour(3.999), EColour::BrightCyan);
        assert_eq!(loss_colour(0.0), EColour::BrightCyan);
        assert_eq!(format_loss(0.4444), "0.444");
        assert_eq!(format_loss(3.9996), "4.000");
        // Disabled output has no escape codes.
        assert!(!format_loss(0.5).contains('\x1b'));
    }

    #[test]
    fn format_map_accuracy_thresholds_and_none() {
        let _guard = TEST_LOCK.lock().unwrap();
        COLOUR_ENABLED.store(false, Ordering::Relaxed);
        assert_eq!(format_map_accuracy(-1.0), "none");
        assert_eq!(map_accuracy_colour(0.49), EColour::BrightRed);
        assert_eq!(map_accuracy_colour(0.5), EColour::Red);
        assert_eq!(map_accuracy_colour(0.6), EColour::Blue);
        assert_eq!(map_accuracy_colour(0.7), EColour::BrightBlue);
        assert_eq!(map_accuracy_colour(f32::NAN), EColour::BrightRed);
        assert_eq!(format_map_accuracy(0.996685), "99.67%");
    }
}

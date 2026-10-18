//! Detector-mode key map.
//!
//! Transcribed from `DarkMark/src-darkmark/DMContent.cpp`, `keyPressed()`
//! (`:1013-2050`), not from the plan's summary table. Every branch below names
//! the DarkMark key that produced it; OBB/segmentor/cuboid-only branches are
//! excluded (they are out of the plan's v1 scope).
//!
//! Where a key only changes a display mode, the action is still mapped so the
//! map is parity-complete; the UI decides whether it has an implementation.

use crate::edit::Mods;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    Char(char),
    Digit(u8),
    Left,
    Right,
    Up,
    Down,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    Delete,
    Backspace,
    Enter,
    Space,
    Escape,
    F1,
    F2,
    F12,
    Plus,
    Minus,
    Insert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyInput {
    pub key: Key,
    pub mods: Mods,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    /// `0`-`9`, additive with Ctrl (+10) and Alt (+20) (`:1106-1120`).
    ClassDigit(u8),
    PrevImage,
    NextImage,
    FirstImage,
    LastImage,
    PrevUnmarked,
    NextUnmarked,
    /// Tab / Shift+Tab; skips marks that are not visible (`:1041-1093`).
    CycleMark(bool),
    ThresholdUp,
    ThresholdDown,
    DeleteMark,
    DeleteImage,
    /// Escape: leave zoom/modes, then clear the selection (`:1297`).
    Cancel,
    Help,
    ZoomIn,
    ZoomOut,
    ZoomToggle,
    /// `r`: random sort (`:1556`).
    SortRandom,
    /// `R`: alphabetical sort (`:1568`).
    SortAlphabetical,
    AcceptAll,
    /// `A`: accept the selected prediction only (`:1586`).
    AcceptSelected,
    /// `,` / `.`: signed normalized size delta applied to every mark (`:1606`).
    Scale(f32),
    /// `p`: on -> auto -> off (`:1672-1712`).
    TogglePredictions,
    ToggleMarks,
    /// `o`: mass-delete click-drag mode (`:1736`).
    MassDelete,
    /// `P`: four-click homography (`:1751`).
    Homography,
    /// `` ` ``: merge mode (`:1768`).
    Merge,
    /// `l`: on -> off -> auto labels (`:1780`).
    ToggleLabels,
    ToggleBold,
    ToggleShade,
    ClearMarks,
    ClassMenu,
    /// `d`: snap every real mark (`:1814`).
    SnapAll,
    /// `D`: snapping on/off (`:1856`).
    ToggleSnapping,
    /// `h`: cycle heatmaps, or enable them (`:1883`).
    HeatmapCycle,
    /// `H`: heatmaps on/off (`:1898`).
    ToggleHeatmap,
    Jump,
    ToggleEmpty,
    /// `s` / `S`: screenshot, with markup for the shifted key (`:1927-1942`).
    Screenshot(bool),
    WindowScreenshot,
    ToggleTiling,
    TileEdge(f32),
    TileRect(f32),
    Nudge(f32, f32),
    ClipboardCopy,
    ClipboardPaste,
    Undo,
    Redo,
    Save,
    /// `y` / `Y`: copy marks from the previous / next annotated image (`:2022`).
    CopyPrev,
    CopyNext,
    /// `Shift+U`: restore the last deleted image (`:1261`).
    UndoDeleteImage,
}

pub fn map_key(input: KeyInput) -> Option<Action> {
    let mods = input.mods;
    match input.key {
        Key::Digit(digit) => Some(Action::ClassDigit(digit)),

        Key::Tab => Some(Action::CycleMark(mods.shift)),
        Key::Home => Some(Action::FirstImage),
        Key::End => Some(Action::LastImage),
        Key::Right => Some(Action::NextImage),
        Key::Left => Some(Action::PrevImage),
        Key::PageUp => Some(Action::PrevUnmarked),
        Key::PageDown => Some(Action::NextUnmarked),

        Key::Up => Some(Action::ThresholdUp),
        Key::Down => Some(Action::ThresholdDown),

        Key::Delete if mods.shift => Some(Action::DeleteImage),
        Key::Delete | Key::Backspace => Some(Action::DeleteMark),
        Key::Escape => Some(Action::Cancel),
        Key::F1 | Key::F2 => Some(Action::Help),
        Key::F12 => Some(Action::WindowScreenshot),

        Key::Minus => Some(Action::ZoomOut),
        Key::Plus => Some(Action::ZoomIn),
        Key::Space => Some(Action::ZoomToggle),

        Key::Insert if mods.ctrl => Some(Action::ClipboardCopy),
        Key::Insert if mods.shift => Some(Action::ClipboardPaste),

        Key::Char('c') if mods.ctrl => Some(Action::ClipboardCopy),
        Key::Char('v') if mods.ctrl => Some(Action::ClipboardPaste),
        Key::Char('s') if mods.ctrl && mods.shift => Some(Action::Save),
        Key::Char('z') if mods.ctrl => Some(Action::Undo),
        Key::Char('y') if mods.ctrl => Some(Action::Redo),
        Key::Char('y') => Some(Action::CopyPrev),
        Key::Char('Y') => Some(Action::CopyNext),
        Key::Char('U') => Some(Action::UndoDeleteImage),
        Key::Char('u') if mods.shift => Some(Action::UndoDeleteImage),
        Key::Char('w') if mods.ctrl => Some(Action::Nudge(0.0, -1.0)),
        Key::Char('s') if mods.ctrl => Some(Action::Nudge(0.0, 1.0)),
        Key::Char('a') if mods.ctrl => Some(Action::Nudge(-1.0, 0.0)),
        Key::Char('d') if mods.ctrl => Some(Action::Nudge(1.0, 0.0)),

        Key::Char('r') => Some(Action::SortRandom),
        Key::Char('R') => Some(Action::SortAlphabetical),
        Key::Char('a') => Some(Action::AcceptAll),
        Key::Char('A') => Some(Action::AcceptSelected),
        Key::Char(',') => Some(Action::Scale(-0.01)),
        Key::Char('.') => Some(Action::Scale(0.01)),
        Key::Char('p') => Some(Action::TogglePredictions),
        Key::Char('m') => Some(Action::ToggleMarks),
        Key::Char('o') => Some(Action::MassDelete),
        Key::Char('P') => Some(Action::Homography),
        Key::Char('`') => Some(Action::Merge),
        Key::Char('l') => Some(Action::ToggleLabels),
        Key::Char('b') => Some(Action::ToggleBold),
        Key::Char('B') => Some(Action::ToggleShade),
        Key::Char('C') => Some(Action::ClearMarks),
        Key::Char('c') | Key::Enter => Some(Action::ClassMenu),
        Key::Char('d') => Some(Action::SnapAll),
        Key::Char('D') => Some(Action::ToggleSnapping),
        Key::Char('h') => Some(Action::HeatmapCycle),
        Key::Char('H') => Some(Action::ToggleHeatmap),
        Key::Char('j') => Some(Action::Jump),
        Key::Char('n') => Some(Action::ToggleEmpty),
        Key::Char('s') => Some(Action::Screenshot(false)),
        Key::Char('S') => Some(Action::Screenshot(true)),
        Key::Char('t') => Some(Action::ToggleTiling),
        Key::Char('[') => Some(Action::TileEdge(-0.05)),
        Key::Char(']') => Some(Action::TileEdge(0.05)),
        Key::Char('(') => Some(Action::TileRect(-0.05)),
        Key::Char(')') => Some(Action::TileRect(0.05)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(key: Key) -> KeyInput {
        KeyInput { key, mods: Mods::default() }
    }

    #[test]
    fn detector_keys_match_dmcontent_keypressed() {
        for (key, expected) in [
            (Key::Char('r'), Action::SortRandom),
            (Key::Char('R'), Action::SortAlphabetical),
            (Key::Char('a'), Action::AcceptAll),
            (Key::Char('A'), Action::AcceptSelected),
            (Key::Char(','), Action::Scale(-0.01)),
            (Key::Char('.'), Action::Scale(0.01)),
            (Key::Char('p'), Action::TogglePredictions),
            (Key::Char('m'), Action::ToggleMarks),
            (Key::Char('o'), Action::MassDelete),
            (Key::Char('P'), Action::Homography),
            (Key::Char('`'), Action::Merge),
            (Key::Char('l'), Action::ToggleLabels),
            (Key::Char('b'), Action::ToggleBold),
            (Key::Char('B'), Action::ToggleShade),
            (Key::Char('C'), Action::ClearMarks),
            (Key::Char('c'), Action::ClassMenu),
            (Key::Char('d'), Action::SnapAll),
            (Key::Char('D'), Action::ToggleSnapping),
            (Key::Char('h'), Action::HeatmapCycle),
            (Key::Char('H'), Action::ToggleHeatmap),
            (Key::Char('j'), Action::Jump),
            (Key::Char('n'), Action::ToggleEmpty),
            (Key::Char('s'), Action::Screenshot(false)),
            (Key::Char('S'), Action::Screenshot(true)),
            (Key::Char('t'), Action::ToggleTiling),
            (Key::Char('['), Action::TileEdge(-0.05)),
            (Key::Char(']'), Action::TileEdge(0.05)),
            (Key::Char('('), Action::TileRect(-0.05)),
            (Key::Char(')'), Action::TileRect(0.05)),
            (Key::Left, Action::PrevImage),
            (Key::Right, Action::NextImage),
            (Key::Home, Action::FirstImage),
            (Key::End, Action::LastImage),
            (Key::PageUp, Action::PrevUnmarked),
            (Key::PageDown, Action::NextUnmarked),
            (Key::Up, Action::ThresholdUp),
            (Key::Down, Action::ThresholdDown),
            (Key::Space, Action::ZoomToggle),
            (Key::Plus, Action::ZoomIn),
            (Key::Minus, Action::ZoomOut),
            (Key::Escape, Action::Cancel),
            (Key::F1, Action::Help),
            (Key::F12, Action::WindowScreenshot),
            (Key::Char('y'), Action::CopyPrev),
            (Key::Char('Y'), Action::CopyNext),
            (Key::Char('U'), Action::UndoDeleteImage),
        ] {
            assert_eq!(map_key(input(key)), Some(expected), "{key:?}");
        }
        assert_eq!(map_key(input(Key::Digit(0))), Some(Action::ClassDigit(0)));
        assert_eq!(map_key(input(Key::Digit(9))), Some(Action::ClassDigit(9)));
        assert_eq!(map_key(input(Key::Tab)), Some(Action::CycleMark(false)));
        assert_eq!(
            map_key(KeyInput { key: Key::Tab, mods: Mods { shift: true, ..Mods::default() } }),
            Some(Action::CycleMark(true))
        );
        assert_eq!(
            map_key(KeyInput { key: Key::Delete, mods: Mods { shift: true, ..Mods::default() } }),
            Some(Action::DeleteImage)
        );
        assert_eq!(
            map_key(KeyInput { key: Key::Char('s'), mods: Mods { ctrl: true, shift: true, ..Mods::default() } }),
            Some(Action::Save)
        );
        assert_eq!(
            map_key(KeyInput { key: Key::Char('w'), mods: Mods { ctrl: true, ..Mods::default() } }),
            Some(Action::Nudge(0.0, -1.0))
        );
    }
}

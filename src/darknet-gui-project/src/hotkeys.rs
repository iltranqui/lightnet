//! Detector-mode key map (`G16`).
//!
//! The `Action` set is the detector subset of DarkMark's hotkey window; OBB,
//! segmentation and cuboid keys are out of scope. Bulk menu-only operations
//! (reload/resave/duplicate finder) are not mapped.

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
    Plus,
    Minus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyInput {
    pub key: Key,
    pub mods: Mods,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    /// Digit class selection, additive with Ctrl (+10) and Alt (+20) (`G11`).
    ClassDigit(u8),
    PrevImage,
    NextImage,
    FirstImage,
    LastImage,
    PrevUnmarked,
    NextUnmarked,
    CycleMark(bool),
    ThresholdUp,
    ThresholdDown,
    DeleteMark,
    ClearMarks,
    ToggleEmpty,
    CopyPrev,
    CopyNext,
    ToggleMarks,
    TogglePredictions,
    AcceptAll,
    AcceptSelected,
    DeletePredictions,
    Nudge(f32, f32),
    Scale(f32),
    Undo,
    Redo,
    Save,
    ClassMenu,
    ClipboardCopy,
    ClipboardPaste,
    ZoomIn,
    ZoomOut,
    ZoomAuto,
    SortNext,
    DeleteImage,
    UndoDelete,
    Heatmap,
    Jump,
    Help,
}

pub fn map_key(input: KeyInput) -> Option<Action> {
    let mods = input.mods;
    match input.key {
        Key::Digit(digit) => Some(Action::ClassDigit(digit)),

        Key::Left if !mods.ctrl && !mods.alt => Some(Action::PrevImage),
        Key::Right if !mods.ctrl && !mods.alt => Some(Action::NextImage),
        Key::Home => Some(Action::FirstImage),
        Key::End => Some(Action::LastImage),
        Key::PageUp => Some(Action::PrevUnmarked),
        Key::PageDown => Some(Action::NextUnmarked),
        Key::Tab => Some(Action::CycleMark(mods.shift)),

        Key::Up => Some(Action::ThresholdUp),
        Key::Down => Some(Action::ThresholdDown),

        Key::Delete if mods.shift => Some(Action::DeleteImage),
        Key::Delete | Key::Backspace => Some(Action::DeleteMark),

        Key::Char('c') if mods.ctrl => Some(Action::ClipboardCopy),
        Key::Char('v') if mods.ctrl => Some(Action::ClipboardPaste),
        Key::Char('s') if mods.ctrl && mods.shift => Some(Action::Save),
        Key::Char('z') if mods.ctrl => Some(Action::Undo),
        Key::Char('y') if mods.ctrl => Some(Action::Redo),
        Key::Char('w') if mods.ctrl => Some(Action::Nudge(0.0, -1.0)),
        Key::Char('s') if mods.ctrl => Some(Action::Nudge(0.0, 1.0)),
        Key::Char('a') if mods.ctrl => Some(Action::Nudge(-1.0, 0.0)),
        Key::Char('d') if mods.ctrl => Some(Action::Nudge(1.0, 0.0)),

        Key::Char('a') if !mods.ctrl => Some(Action::AcceptAll),
        Key::Char('A') if !mods.ctrl => Some(Action::AcceptSelected),
        Key::Char('o') => Some(Action::DeletePredictions),
        Key::Char('C') => Some(Action::ClearMarks),
        Key::Char('n') => Some(Action::ToggleEmpty),
        Key::Char('y') => Some(Action::CopyPrev),
        Key::Char('Y') => Some(Action::CopyNext),
        Key::Char('m') => Some(Action::ToggleMarks),
        Key::Char('p') => Some(Action::TogglePredictions),
        Key::Char('j') => Some(Action::Jump),
        Key::Char('c') | Key::Enter => Some(Action::ClassMenu),
        Key::Char('h') | Key::Char('H') => Some(Action::Heatmap),
        Key::Char('.') => Some(Action::Scale(1.01)),
        Key::Char(',') => Some(Action::Scale(0.99)),
        Key::Char('r') | Key::Char('R') => Some(Action::SortNext),
        Key::Char('U') | Key::Char('u') if mods.shift => Some(Action::UndoDelete),

        Key::Space => Some(Action::ZoomAuto),
        Key::Plus => Some(Action::ZoomIn),
        Key::Minus => Some(Action::ZoomOut),
        Key::F1 | Key::F2 => Some(Action::Help),
        Key::Escape => None,
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
    fn every_documented_detector_key_maps() {
        for (key, expected) in [
            (Key::Char('C'), Action::ClearMarks),
            (Key::Char('n'), Action::ToggleEmpty),
            (Key::Char('y'), Action::CopyPrev),
            (Key::Char('Y'), Action::CopyNext),
            (Key::Char('m'), Action::ToggleMarks),
            (Key::Char('p'), Action::TogglePredictions),
            (Key::Char('a'), Action::AcceptAll),
            (Key::Char('A'), Action::AcceptSelected),
            (Key::Char('o'), Action::DeletePredictions),
            (Key::Char('j'), Action::Jump),
            (Key::Char('h'), Action::Heatmap),
            (Key::Char('.'), Action::Scale(1.01)),
            (Key::Char(','), Action::Scale(0.99)),
            (Key::Char('r'), Action::SortNext),
            (Key::Left, Action::PrevImage),
            (Key::Right, Action::NextImage),
            (Key::Home, Action::FirstImage),
            (Key::End, Action::LastImage),
            (Key::PageUp, Action::PrevUnmarked),
            (Key::PageDown, Action::NextUnmarked),
            (Key::Up, Action::ThresholdUp),
            (Key::Down, Action::ThresholdDown),
            (Key::Space, Action::ZoomAuto),
            (Key::Plus, Action::ZoomIn),
            (Key::Minus, Action::ZoomOut),
            (Key::F1, Action::Help),
        ] {
            assert_eq!(map_key(input(key)), Some(expected), "{key:?}");
        }
        assert_eq!(map_key(input(Key::Digit(0))), Some(Action::ClassDigit(0)));
        assert_eq!(map_key(input(Key::Digit(9))), Some(Action::ClassDigit(9)));
        assert_eq!(map_key(input(Key::Tab)), Some(Action::CycleMark(false)));
    }
}

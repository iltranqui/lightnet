//! Per-image undo/redo, capped at 100 entries.

const CAP: usize = 100;

#[derive(Clone, Debug)]
pub struct UndoStack<T: Clone> {
    undo: Vec<T>,
    redo: Vec<T>,
}

impl<T: Clone> Default for UndoStack<T> {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }
}

impl<T: Clone> UndoStack<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Records the state before a mutation. Any pending redo is discarded.
    pub fn push(&mut self, before: T) {
        self.undo.push(before);
        if self.undo.len() > CAP {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Returns the state to restore, given the current one.
    pub fn undo(&mut self, current: T) -> Option<T> {
        let previous = self.undo.pop()?;
        self.redo.push(current);
        Some(previous)
    }

    pub fn redo(&mut self, current: T) -> Option<T> {
        let next = self.redo.pop()?;
        self.undo.push(current);
        Some(next)
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_then_redo_restores() {
        let mut stack = UndoStack::new();
        stack.push(1);
        stack.push(2);
        assert_eq!(stack.undo(3), Some(2));
        assert_eq!(stack.undo(2), Some(1));
        assert_eq!(stack.undo(1), None);
        assert_eq!(stack.redo(1), Some(2));
    }

    #[test]
    fn push_after_undo_clears_redo() {
        let mut stack = UndoStack::new();
        stack.push(1);
        assert_eq!(stack.undo(2), Some(1));
        stack.push(9);
        assert!(!stack.can_redo());
    }

    #[test]
    fn cap_drops_oldest() {
        let mut stack = UndoStack::new();
        for value in 0..150 {
            stack.push(value);
        }
        // Only the newest 100 survive, so undo walks back to 50.
        let mut current = 150;
        while let Some(previous) = stack.undo(current) {
            current = previous;
        }
        assert_eq!(current, 50);
    }
}

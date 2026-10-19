//! Navigation rules (`G13`).
//!
//! Left/Right and Home/End clamp instead of wrapping. PageUp/PageDown walk to the
//! previous/next image with no marks (`completely_empty` counts as one), which is
//! the "find unannotated" shortcut.

use crate::project::Project;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Previous,
    Next,
}

/// Clamped single-step move; returns `from` at the ends.
pub fn step(from: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    (from as isize + delta).clamp(0, len as isize - 1) as usize
}

pub fn home() -> usize {
    0
}

pub fn end(len: usize) -> usize {
    len.saturating_sub(1)
}

/// Nearest image with zero marks in `dir`; the boundary image is returned even
/// when it is annotated, matching DarkMark.
pub fn find_unmarked(project: &Project, from: usize, dir: Dir) -> Option<usize> {
    let len = project.images.len();
    if len == 0 {
        return None;
    }
    let mut index = from;
    loop {
        let next = match dir {
            Dir::Previous if index == 0 => return Some(0),
            Dir::Next if index + 1 >= len => return Some(len - 1),
            Dir::Previous => index - 1,
            Dir::Next => index + 1,
        };
        index = next;
        if project.mark_count(index) == 0 {
            return Some(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(name: &str, marks: &[usize]) -> Project {
        let root = std::env::temp_dir().join(format!("darknet-gui-nav-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        for (index, count) in marks.iter().enumerate() {
            let path = root.join(format!("{index}.png"));
            std::fs::write(&path, b"x").unwrap();
            if *count > 0 {
                let mut text = String::new();
                for _ in 0..*count {
                    text.push_str("0 0.5000000000 0.5000000000 0.1000000000 0.1000000000\n");
                }
                std::fs::write(path.with_extension("txt"), text).unwrap();
            }
        }
        Project::load(&root).unwrap()
    }

    #[test]
    fn left_right_clamp_no_wrap() {
        assert_eq!(step(0, 5, -1), 0);
        assert_eq!(step(4, 5, 1), 4);
        assert_eq!(step(2, 5, 1), 3);
        assert_eq!(home(), 0);
        assert_eq!(end(5), 4);
    }

    #[test]
    fn page_walk_stops_on_unmarked_and_at_the_boundary() {
        let project = project("walk", &[1, 1, 0, 1, 0]);
        // Forward from 0 skips 1 and lands on 2.
        assert_eq!(find_unmarked(&project, 0, Dir::Next), Some(2));
        // Forward from 4 has no unmarked ahead: boundary image 4.
        assert_eq!(find_unmarked(&project, 4, Dir::Next), Some(4));
        // Backward from 4 lands on 2.
        assert_eq!(find_unmarked(&project, 4, Dir::Previous), Some(2));
        // Backward from 0 is the boundary.
        assert_eq!(find_unmarked(&project, 0, Dir::Previous), Some(0));
        std::fs::remove_dir_all(&project.root).unwrap();
    }
}

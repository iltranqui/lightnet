//! Pure project logic for the Darknet annotation GUI.
//!
//! Everything here is UI-free and file-oriented so it can be unit-tested without
//! a window: sidecar IO ([`annotation`]), image scanning and the project index
//! ([`project`]), filtering and navigation ([`filter`], [`nav`]), undo
//! ([`undo`]), the DarkMark mouse/edit model ([`edit`]), the detector key map
//! ([`hotkeys`]), bulk actions ([`actions`]), project creation ([`create`]), and
//! atomic class remapping ([`classes`]).

pub mod actions;
pub mod annotation;
pub mod classes;
pub mod create;
pub mod edit;
pub mod filter;
pub mod hotkeys;
pub mod nav;
pub mod project;
pub mod undo;

pub use annotation::{Annotation, Mark, Point, VERSION};
pub use edit::{Editor, Mods};
pub use filter::{Filter, SortKey, Tri};
pub use project::{ImageEntry, Project};
pub use undo::UndoStack;

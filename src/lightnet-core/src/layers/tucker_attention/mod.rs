mod cpu;
pub(crate) mod geometry;
pub use cpu::*;
pub(crate) use geometry::{TuckerGeometry, tucker_weight_count, validate_geometry};
pub(crate) mod gpu;

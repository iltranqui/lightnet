//! Errors shared by parsing, execution, I/O, and backend boundaries.
//!
//! C++ Darknet reports many failures through fatal logging and process exits.
//! Rust returns the same validation failures as values so library callers can
//! choose whether to retry, fall back to CPU, or terminate.

use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Parse(String),
    InvalidConfig(String),
    Backend(String),
    InvalidRoute {
        layer_index: usize,
        reference: isize,
    },
    UnsupportedLayer {
        layer_index: usize,
        name: String,
    },
    Weights(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse(message) => write!(f, "{message}"),
            Self::InvalidConfig(message) => write!(f, "{message}"),
            Self::Backend(message) => write!(f, "{message}"),
            Self::InvalidRoute {
                layer_index,
                reference,
            } => write!(
                f,
                "invalid route reference {reference} while resolving layer #{layer_index}"
            ),
            Self::UnsupportedLayer { layer_index, name } => {
                write!(f, "unsupported layer [{name}] at index #{layer_index}")
            }
            Self::Weights(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

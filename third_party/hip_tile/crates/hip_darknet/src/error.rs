use thiserror::Error;

/// Errors from parsing, loading and running a Darknet model.
#[derive(Debug, Error)]
pub enum Error {
    #[error("cfg line {line}: {msg}")]
    Cfg { line: usize, msg: String },
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("model: {0}")]
    Model(String),
    #[error("weights: {0}")]
    Weights(String),
    #[error("{path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("image {path}: {source}")]
    Image {
        path: String,
        #[source]
        source: image::ImageError,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

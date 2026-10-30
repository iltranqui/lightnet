//! Decoder-only language model: tokenizer, CPU reference ops, training and
//! KV-cache inference.  See `tasks/rust/03_plans/llm/plan_llm_decoder.md`.
//!
//! The crate is independent of the detector graph in `lightnet-core`, so it
//! carries its own error type and a small `.cfg` reader (settled in Task 1).

use std::fmt::{Display, Formatter};

pub mod attention;
pub mod cache;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod decoder;
pub mod ops;
pub mod optim;
pub mod sampling;
pub mod tokenizer;
pub mod train;
pub mod types;

pub use attention::{causal_gqa, causal_gqa_backward};
pub use cache::KvCache;
pub use checkpoint::{
    CHECKPOINT_SCHEMA_VERSION, TrainOptions, TrainingSnapshot, load_checkpoint, save_checkpoint,
};
pub use config::{ConfigFile, DecoderConfig, RawSection};
pub use data::{
    CorpusDocument, DocumentEntry, LoaderState, ShardInfo, ShardManifest, StateSlot, TokenLoader,
    read_manifest, write_token_shards,
};
pub use decoder::{CpuDecoder, CpuForward, DecoderGrads, DecoderParams, DecoderTape, Parameter};
pub use ops::{
    embedding, embedding_backward, rms_norm, rms_norm_backward, rope_in_place, sparse_ce, swiglu,
    swiglu_backward,
};
pub use optim::{TokenSchedule, UpdateOptions, UpdateReport, clip_global_norm};
pub use sampling::{SamplingOptions, generate, sample_token};
pub use tokenizer::{
    BASE_VOCAB_SIZE, BOS_ID, BYTE_BASE, ByteBpe, EOS_ID, FIRST_MERGE_ID, NormalizationSpec, PAD_ID,
    UNK_ID, normalize_text,
};
pub use train::{CHECKPOINT_PRECISION, CpuEngine, EngineState, LmEngine, Trainer};
pub use types::{LossSummary, RngState, TokenBatch};

/// Result alias for every fallible LM entry point.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors produced while reading decoder configuration and token data.
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Parse(String),
    InvalidConfig(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse(message) => write!(f, "{message}"),
            Self::InvalidConfig(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Parse(_) | Self::InvalidConfig(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

//! Versioned, checksummed training checkpoints with atomic publication.
//!
//! Layout: magic, schema, UTF-8 metadata, binary f32 payload, SHA-256 over
//! everything before the digest.  Parameter tensors, moments, and pending
//! gradients are stored once each, in canonical registry order.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::optim::TokenSchedule;
use crate::train::CHECKPOINT_PRECISION;
use crate::{
    CpuDecoder, DecoderConfig, DecoderGrads, DecoderParams, Error, LoaderState, Parameter, Result,
    StateSlot,
};

const MAGIC: &[u8; 8] = b"LMSNAP01";
pub const CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Training configuration that must survive a resume.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainOptions {
    pub batch: usize,
    pub context: usize,
    pub accum: u32,
    pub seed: u64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub clip_norm: f32,
    pub schedule: TokenSchedule,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingSnapshot {
    pub schema_version: u32,
    pub tokenizer_hash: String,
    pub manifest_hash: String,
    pub engine: crate::train::EngineState,
    pub loader: LoaderState,
    pub train_options: TrainOptions,
    pub pending_active_tokens: u64,
    pub pending_microbatches: u32,
    pub successful_updates: u32,
    pub updated_tokens: u64,
    pub schedule: TokenSchedule,
}

pub fn save_checkpoint(path: &Path, snapshot: &TrainingSnapshot) -> Result<()> {
    validate_snapshot(snapshot)?;
    let bytes = encode_snapshot(snapshot)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temp = temporary_sibling(path)?;
    let write_result = (|| -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = write_result {
        let _ = fs::remove_file(&temp);
        return Err(err);
    }
    if let Err(err) = fs::rename(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(err.into());
    }
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Result<TrainingSnapshot> {
    let bytes = fs::read(path)?;
    decode_snapshot(&bytes)
}

fn temporary_sibling(path: &Path) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| Error::InvalidConfig("checkpoint path has no file name".to_owned()))?;
    let mut temp = format!(".{}.tmp.{}", name.to_string_lossy(), std::process::id());
    let mut attempt = 0u32;
    while temp_path_exists(path, &temp) {
        attempt = attempt.wrapping_add(1);
        temp = format!(
            ".{}.tmp.{}.{}",
            name.to_string_lossy(),
            std::process::id(),
            attempt
        );
        if attempt > 1024 {
            return Err(Error::InvalidConfig(
                "could not allocate a checkpoint temporary path".to_owned(),
            ));
        }
    }
    Ok(path.with_file_name(temp))
}

fn temp_path_exists(path: &Path, name: &str) -> bool {
    path.with_file_name(name).exists()
}

fn validate_snapshot(snapshot: &TrainingSnapshot) -> Result<()> {
    if snapshot.schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(Error::InvalidConfig(format!(
            "unsupported checkpoint schema {}",
            snapshot.schema_version
        )));
    }
    if !is_hex_digest(&snapshot.tokenizer_hash) {
        return Err(Error::InvalidConfig(
            "tokenizer hash must be 64 hexadecimal characters".to_owned(),
        ));
    }
    if !is_hex_digest(&snapshot.manifest_hash) {
        return Err(Error::InvalidConfig(
            "manifest hash must be 64 hexadecimal characters".to_owned(),
        ));
    }
    if snapshot.loader.manifest_hash != snapshot.manifest_hash {
        return Err(Error::InvalidConfig(
            "loader state and snapshot disagree on the manifest hash".to_owned(),
        ));
    }
    if snapshot.engine.pending_tokens != snapshot.pending_active_tokens {
        return Err(Error::InvalidConfig(
            "engine and trainer disagree on pending tokens".to_owned(),
        ));
    }
    if snapshot.engine.step != snapshot.successful_updates
        || snapshot.engine.updated_tokens != snapshot.updated_tokens
    {
        return Err(Error::InvalidConfig(
            "checkpoint counters disagree with the engine state".to_owned(),
        ));
    }
    snapshot.schedule.lr(0)?;
    if snapshot.schedule != snapshot.train_options.schedule {
        return Err(Error::InvalidConfig(
            "checkpoint schedule disagrees with the train options".to_owned(),
        ));
    }
    if snapshot.train_options.batch == 0
        || snapshot.train_options.context == 0
        || snapshot.train_options.accum == 0
    {
        return Err(Error::InvalidConfig(
            "train options require non-zero batch, context, and accumulation".to_owned(),
        ));
    }
    if snapshot.pending_active_tokens > 0 && snapshot.pending_microbatches == 0 {
        return Err(Error::InvalidConfig(
            "pending tokens require at least one pending microbatch".to_owned(),
        ));
    }
    if snapshot.train_options.context > snapshot.engine.config.context {
        return Err(Error::InvalidConfig(
            "train context exceeds the decoder context".to_owned(),
        ));
    }
    if snapshot.engine.precision != CHECKPOINT_PRECISION {
        return Err(Error::InvalidConfig(
            "only f32 CPU checkpoints are supported".to_owned(),
        ));
    }
    snapshot.engine.config.validate()?;
    validate_parameter_arrays(&snapshot.engine.parameters)?;
    let count = snapshot.engine.parameters.tensors.len();
    if snapshot.engine.first_moment.len() != count
        || snapshot.engine.second_moment.len() != count
        || snapshot.engine.pending.values.len() != count
    {
        return Err(Error::InvalidConfig(
            "checkpoint arrays do not match the parameter registry".to_owned(),
        ));
    }
    for index in 0..count {
        let expected = snapshot.engine.parameters.tensors[index].values.len();
        for (label, values) in [
            ("first moment", &snapshot.engine.first_moment[index]),
            ("second moment", &snapshot.engine.second_moment[index]),
            ("pending gradient", &snapshot.engine.pending.values[index]),
        ] {
            if values.len() != expected {
                return Err(Error::InvalidConfig(format!(
                    "checkpoint {label} {index} has the wrong length"
                )));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidConfig(format!(
                    "checkpoint {label} {index} contains a non-finite value"
                )));
            }
        }
    }
    let slot_count = snapshot.loader.pending.len();
    if snapshot.loader.pending_fragments as usize > slot_count {
        return Err(Error::InvalidConfig(
            "loader pending fragments exceed the pending slots".to_owned(),
        ));
    }
    Ok(())
}

fn validate_parameter_arrays(params: &DecoderParams) -> Result<()> {
    for parameter in &params.tensors {
        if parameter.values.len() != parameter.shape.iter().product::<usize>() {
            return Err(Error::InvalidConfig(format!(
                "parameter {} has {} values for shape {:?}",
                parameter.name,
                parameter.values.len(),
                parameter.shape
            )));
        }
        if parameter.values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidConfig(format!(
                "parameter {} contains a non-finite value",
                parameter.name
            )));
        }
    }
    Ok(())
}

fn encode_snapshot(snapshot: &TrainingSnapshot) -> Result<Vec<u8>> {
    let cfg = &snapshot.engine.config;
    let schedule = &snapshot.schedule;
    let train = &snapshot.train_options;
    let loader = &snapshot.loader;
    let mut meta = String::new();
    meta.push_str(&format!("schema {}\n", snapshot.schema_version));
    meta.push_str(&format!("tokenizer_hash {}\n", snapshot.tokenizer_hash));
    meta.push_str(&format!("manifest_hash {}\n", snapshot.manifest_hash));
    meta.push_str(&format!("precision {}\n", snapshot.engine.precision));
    meta.push_str(&format!(
        "config {} {} {} {} {} {} {} {} {} {} {}\n",
        cfg.vocab_size,
        cfg.context,
        cfg.hidden,
        cfg.intermediate,
        cfg.layers,
        cfg.q_heads,
        cfg.kv_heads,
        cfg.rope_theta,
        cfg.rms_eps,
        cfg.dropout,
        if cfg.tie_embeddings { 1 } else { 0 }
    ));
    meta.push_str(&format!(
        "schedule {} {} {} {}\n",
        schedule.warmup_tokens, schedule.total_tokens, schedule.peak_lr, schedule.min_lr
    ));
    meta.push_str(&format!(
        "train {} {} {} {} {} {} {} {} {}\n",
        train.batch,
        train.context,
        train.accum,
        train.seed,
        train.beta1,
        train.beta2,
        train.eps,
        train.weight_decay,
        train.clip_norm
    ));
    meta.push_str(&format!(
        "counters {} {} {} {} {}\n",
        snapshot.pending_active_tokens,
        snapshot.pending_microbatches,
        snapshot.successful_updates,
        snapshot.updated_tokens,
        snapshot.engine.pending_tokens
    ));
    meta.push_str(&format!(
        "optimizer {} {}\n",
        snapshot.engine.step, snapshot.engine.model_rng.state
    ));
    meta.push_str(&format!("loader_hash {}\n", loader.manifest_hash));
    meta.push_str(&format!(
        "loader {} {} {} {} {} {}\n",
        loader.rng_state,
        loader.epoch,
        loader.order_cursor,
        loader.document.map(u64::from).unwrap_or(u64::MAX),
        loader.window_offset,
        loader.pending_fragments
    ));
    meta.push_str(&format!(
        "loader_counts {} {}\n",
        loader.order.len(),
        loader.pending.len()
    ));

    let mut payload = Vec::new();
    for tensor in &snapshot.engine.parameters.tensors {
        push_f32s(&mut payload, &tensor.values);
    }
    for values in &snapshot.engine.first_moment {
        push_f32s(&mut payload, values);
    }
    for values in &snapshot.engine.second_moment {
        push_f32s(&mut payload, values);
    }
    for values in &snapshot.engine.pending.values {
        push_f32s(&mut payload, values);
    }
    for &index in &loader.order {
        payload.extend_from_slice(&index.to_le_bytes());
    }
    for slot in &loader.pending {
        payload.extend_from_slice(&slot.input.to_le_bytes());
        payload.extend_from_slice(&slot.target.to_le_bytes());
        payload.extend_from_slice(&slot.segment.to_le_bytes());
        payload.extend_from_slice(&slot.position.to_le_bytes());
        payload.push(u8::from(slot.valid));
    }

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&CHECKPOINT_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    out.extend_from_slice(meta.as_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&payload);
    let digest = Sha256::digest(&out);
    out.extend_from_slice(&digest);
    Ok(out)
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for &value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn decode_snapshot(bytes: &[u8]) -> Result<TrainingSnapshot> {
    const HEADER: usize = 8 + 4 + 8;
    if bytes.len() < HEADER + 8 + 32 {
        return Err(Error::Parse("checkpoint is truncated".to_owned()));
    }
    if &bytes[..8] != MAGIC {
        return Err(Error::Parse("checkpoint magic does not match".to_owned()));
    }
    let schema = read_u32(bytes, 8)?;
    if schema != CHECKPOINT_SCHEMA_VERSION {
        return Err(Error::Parse(format!(
            "unsupported checkpoint schema {schema}"
        )));
    }
    let meta_len = read_u64(bytes, 12)? as usize;
    let meta_start = HEADER;
    let meta_end = meta_start
        .checked_add(meta_len)
        .ok_or_else(|| Error::Parse("checkpoint metadata length overflows".to_owned()))?;
    if meta_end + 8 + 32 > bytes.len() {
        return Err(Error::Parse("checkpoint metadata is truncated".to_owned()));
    }
    let payload_len = read_u64(bytes, meta_end)? as usize;
    let payload_start = meta_end + 8;
    let payload_end = payload_start
        .checked_add(payload_len)
        .ok_or_else(|| Error::Parse("checkpoint payload length overflows".to_owned()))?;
    let expected = payload_end + 32;
    if expected != bytes.len() {
        return Err(Error::Parse(
            "checkpoint length does not match its header".to_owned(),
        ));
    }
    let digest = Sha256::digest(&bytes[..payload_end]);
    if digest.as_slice() != &bytes[payload_end..] {
        return Err(Error::Parse(
            "checkpoint checksum does not match".to_owned(),
        ));
    }
    let meta = std::str::from_utf8(&bytes[meta_start..meta_end])
        .map_err(|_| Error::Parse("checkpoint metadata is not UTF-8".to_owned()))?;
    let fields = parse_meta(meta)?;

    let schema_version = fields.u32("schema")?;
    if schema_version != CHECKPOINT_SCHEMA_VERSION {
        return Err(Error::Parse(
            "checkpoint metadata schema disagrees".to_owned(),
        ));
    }
    let tokenizer_hash = fields.digest("tokenizer_hash")?;
    let manifest_hash = fields.digest("manifest_hash")?;
    let precision = fields.string("precision")?.to_owned();
    let config = parse_config(fields.string("config")?)?;
    let schedule = parse_schedule(fields.string("schedule")?)?;
    let train_options = parse_train(fields.string("train")?, schedule)?;
    let counters = fields.u64_list("counters", 5)?;
    let optimizer = fields.u64_list("optimizer", 2)?;
    let loader_hash = fields.digest("loader_hash")?;
    let loader_values = fields.u64_list("loader", 6)?;
    let loader_counts = fields.u64_list("loader_counts", 2)?;

    if loader_hash != manifest_hash {
        return Err(Error::Parse(
            "loader state and checkpoint disagree on the manifest hash".to_owned(),
        ));
    }
    if precision != CHECKPOINT_PRECISION {
        return Err(Error::Parse(format!(
            "checkpoint precision {precision:?} is not supported"
        )));
    }
    let engine_step = u32::try_from(optimizer[0])
        .map_err(|_| Error::Parse("optimizer step does not fit u32".to_owned()))?;
    let pending_microbatches = u32::try_from(counters[1])
        .map_err(|_| Error::Parse("microbatch count does not fit u32".to_owned()))?;
    let successful_updates = u32::try_from(counters[2])
        .map_err(|_| Error::Parse("update count does not fit u32".to_owned()))?;
    if engine_step != successful_updates {
        return Err(Error::Parse(
            "optimizer step and successful updates disagree".to_owned(),
        ));
    }
    if counters[0] != counters[4] {
        return Err(Error::Parse(
            "trainer and engine pending-token counts disagree".to_owned(),
        ));
    }
    let loader_epoch = loader_values[1];
    let order_cursor = usize::try_from(loader_values[2])
        .map_err(|_| Error::Parse("loader cursor does not fit usize".to_owned()))?;
    let document_raw = loader_values[3];
    let document = if document_raw == u64::MAX {
        None
    } else {
        Some(
            u32::try_from(document_raw)
                .map_err(|_| Error::Parse("loader document does not fit u32".to_owned()))?,
        )
    };
    let window_offset = loader_values[4];
    let pending_fragments = u32::try_from(loader_values[5])
        .map_err(|_| Error::Parse("pending fragments do not fit u32".to_owned()))?;
    let order_len = usize::try_from(loader_counts[0])
        .map_err(|_| Error::Parse("loader order length does not fit usize".to_owned()))?;
    let pending_len = usize::try_from(loader_counts[1])
        .map_err(|_| Error::Parse("loader pending length does not fit usize".to_owned()))?;

    let mut cursor = Cursor::new(&bytes[payload_start..payload_end]);
    let prototype = CpuDecoder::new(config.clone(), 0)?;
    let layouts: Vec<usize> = prototype
        .parameters()
        .tensors
        .iter()
        .map(|parameter| parameter.values.len())
        .collect();
    let parameter_count = layouts.len();
    let mut read_block = || -> Result<Vec<Vec<f32>>> {
        let mut block = Vec::with_capacity(parameter_count);
        for &len in &layouts {
            block.push(cursor.read_f32s(len)?);
        }
        Ok(block)
    };
    let parameter_values = read_block()?;
    let first_moment = read_block()?;
    let second_moment = read_block()?;
    let pending = read_block()?;
    let mut order = Vec::with_capacity(order_len);
    for _ in 0..order_len {
        order.push(cursor.read_u32()?);
    }
    let mut loader_pending = Vec::with_capacity(pending_len);
    for _ in 0..pending_len {
        loader_pending.push(StateSlot {
            input: cursor.read_u32()?,
            target: cursor.read_i32()?,
            segment: cursor.read_u32()?,
            position: cursor.read_u32()?,
            valid: cursor.read_u8()? != 0,
        });
    }
    if !cursor.is_empty() {
        return Err(Error::Parse(
            "checkpoint payload has trailing bytes".to_owned(),
        ));
    }

    let mut tensors = Vec::with_capacity(parameter_count);
    for (parameter, values) in prototype.parameters().tensors.iter().zip(parameter_values) {
        tensors.push(Parameter {
            id: parameter.id,
            name: parameter.name.clone(),
            shape: parameter.shape.clone(),
            values,
            decay: parameter.decay,
        });
    }
    let engine = crate::train::EngineState {
        config,
        parameters: DecoderParams { tensors },
        first_moment,
        second_moment,
        pending: DecoderGrads { values: pending },
        pending_tokens: counters[4],
        model_rng: crate::RngState {
            state: optimizer[1],
        },
        precision,
        step: engine_step,
        updated_tokens: counters[3],
    };
    let loader = LoaderState {
        manifest_hash: loader_hash,
        rng_state: loader_values[0],
        epoch: loader_epoch,
        order,
        order_cursor,
        document,
        window_offset,
        pending: loader_pending,
        pending_fragments,
    };
    let snapshot = TrainingSnapshot {
        schema_version,
        tokenizer_hash,
        manifest_hash,
        engine,
        loader,
        train_options,
        pending_active_tokens: counters[0],
        pending_microbatches,
        successful_updates,
        updated_tokens: counters[3],
        schedule,
    };
    validate_snapshot(&snapshot)?;
    Ok(snapshot)
}

fn parse_meta(meta: &str) -> Result<Meta> {
    let mut fields = HashMap::new();
    for line in meta.lines() {
        let (key, value) = line
            .split_once(' ')
            .ok_or_else(|| Error::Parse("checkpoint metadata line has no value".to_owned()))?;
        if fields.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(Error::Parse(format!(
                "duplicate checkpoint metadata key {key}"
            )));
        }
    }
    Ok(Meta { fields })
}

struct Meta {
    fields: HashMap<String, String>,
}

impl Meta {
    fn string(&self, key: &str) -> Result<&str> {
        self.fields
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| Error::Parse(format!("checkpoint metadata is missing {key}")))
    }

    fn digest(&self, key: &str) -> Result<String> {
        let value = self.string(key)?;
        if !is_hex_digest(value) {
            return Err(Error::Parse(format!(
                "checkpoint metadata {key} is not a SHA-256 digest"
            )));
        }
        Ok(value.to_owned())
    }

    fn u32(&self, key: &str) -> Result<u32> {
        self.string(key)?
            .parse()
            .map_err(|_| Error::Parse(format!("checkpoint metadata {key} is not a u32")))
    }

    fn u64_list(&self, key: &str, count: usize) -> Result<Vec<u64>> {
        let values: Vec<u64> = self
            .string(key)?
            .split_whitespace()
            .map(|value| {
                value
                    .parse()
                    .map_err(|_| Error::Parse(format!("checkpoint metadata {key} has a non-u64")))
            })
            .collect::<Result<_>>()?;
        if values.len() != count {
            return Err(Error::Parse(format!(
                "checkpoint metadata {key} has {} values, expected {count}",
                values.len()
            )));
        }
        Ok(values)
    }
}

fn parse_config(text: &str) -> Result<DecoderConfig> {
    let values: Vec<&str> = text.split_whitespace().collect();
    if values.len() != 11 {
        return Err(Error::Parse(
            "checkpoint config does not have 11 fields".to_owned(),
        ));
    }
    let parse_usize = |index: usize| -> Result<usize> {
        values[index]
            .parse()
            .map_err(|_| Error::Parse(format!("checkpoint config field {index} is invalid")))
    };
    let parse_f32 = |index: usize| -> Result<f32> {
        values[index]
            .parse()
            .map_err(|_| Error::Parse(format!("checkpoint config field {index} is invalid")))
    };
    let config = DecoderConfig {
        vocab_size: parse_usize(0)?,
        context: parse_usize(1)?,
        hidden: parse_usize(2)?,
        intermediate: parse_usize(3)?,
        layers: parse_usize(4)?,
        q_heads: parse_usize(5)?,
        kv_heads: parse_usize(6)?,
        rope_theta: parse_f32(7)?,
        rms_eps: parse_f32(8)?,
        dropout: parse_f32(9)?,
        tie_embeddings: match values[10] {
            "0" => false,
            "1" => true,
            _ => {
                return Err(Error::Parse(
                    "checkpoint config tie_embeddings must be 0 or 1".to_owned(),
                ));
            }
        },
    };
    config.validate()?;
    Ok(config)
}

fn parse_schedule(text: &str) -> Result<TokenSchedule> {
    let values: Vec<&str> = text.split_whitespace().collect();
    if values.len() != 4 {
        return Err(Error::Parse(
            "checkpoint schedule does not have 4 fields".to_owned(),
        ));
    }
    Ok(TokenSchedule {
        warmup_tokens: parse_token_u64(values[0], "warmup")?,
        total_tokens: parse_token_u64(values[1], "total")?,
        peak_lr: parse_token_f64(values[2], "peak")?,
        min_lr: parse_token_f64(values[3], "min")?,
    })
}

fn parse_train(text: &str, schedule: TokenSchedule) -> Result<TrainOptions> {
    let values: Vec<&str> = text.split_whitespace().collect();
    if values.len() != 9 {
        return Err(Error::Parse(
            "checkpoint train options do not have 9 fields".to_owned(),
        ));
    }
    let batch = values[0]
        .parse()
        .map_err(|_| Error::Parse("checkpoint batch is invalid".to_owned()))?;
    let context = values[1]
        .parse()
        .map_err(|_| Error::Parse("checkpoint context is invalid".to_owned()))?;
    let accum = values[2]
        .parse()
        .map_err(|_| Error::Parse("checkpoint accumulation is invalid".to_owned()))?;
    let seed = parse_token_u64(values[3], "seed")?;
    let beta1 = parse_token_f32(values[4], "beta1")?;
    let beta2 = parse_token_f32(values[5], "beta2")?;
    let eps = parse_token_f32(values[6], "eps")?;
    let weight_decay = parse_token_f32(values[7], "weight_decay")?;
    let clip_norm = parse_token_f32(values[8], "clip_norm")?;
    Ok(TrainOptions {
        batch,
        context,
        accum,
        seed,
        beta1,
        beta2,
        eps,
        weight_decay,
        clip_norm,
        schedule,
    })
}

fn parse_token_u64(text: &str, label: &str) -> Result<u64> {
    text.parse()
        .map_err(|_| Error::Parse(format!("checkpoint {label} is not a u64")))
}

fn parse_token_f64(text: &str, label: &str) -> Result<f64> {
    text.parse()
        .map_err(|_| Error::Parse(format!("checkpoint {label} is not a number")))
}

fn parse_token_f32(text: &str, label: &str) -> Result<f32> {
    text.parse()
        .map_err(|_| Error::Parse(format!("checkpoint {label} is not a number")))
}

fn is_hex_digest(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::Parse("checkpoint is truncated".to_owned()))?;
    Ok(u32::from_le_bytes(slice.try_into().expect("four bytes")))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| Error::Parse("checkpoint is truncated".to_owned()))?;
    Ok(u64::from_le_bytes(slice.try_into().expect("eight bytes")))
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| Error::Parse("checkpoint cursor overflows".to_owned()))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| Error::Parse("checkpoint payload is truncated".to_owned()))?;
        self.offset = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn read_f32s(&mut self, len: usize) -> Result<Vec<f32>> {
        let slice = self.take(
            len.checked_mul(4)
                .ok_or_else(|| Error::Parse("checkpoint tensor length overflows".to_owned()))?,
        )?;
        let (words, remainder) = slice.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        Ok(words.iter().map(|word| f32::from_le_bytes(*word)).collect())
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

//! `lightnet-rs lm`: tokenizer, shards, CPU train/resume, and evaluation.

use std::error::Error as StdError;
use std::fs;
use std::path::{Path, PathBuf};

use lightnet_lm::{
    ByteBpe, CHECKPOINT_SCHEMA_VERSION, CpuEngine, DecoderConfig, LmEngine, NormalizationSpec,
    TrainOptions, Trainer, TrainingSnapshot, UpdateOptions, load_checkpoint, read_manifest,
    save_checkpoint, write_token_shards,
};

type Result<T> = std::result::Result<T, Box<dyn StdError>>;

#[derive(Clone, Debug, PartialEq)]
pub enum LmCommand {
    Tokenizer(TokenizerArgs),
    Shards(ShardsArgs),
    Train(TrainArgs),
    Resume(ResumeArgs),
    Eval(EvalArgs),
    Generate(GenerateArgs),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TokenizerArgs {
    pub documents: PathBuf,
    pub out: PathBuf,
    pub vocab: usize,
    pub latin_fold: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShardsArgs {
    pub documents: PathBuf,
    pub tokenizer: PathBuf,
    pub out: PathBuf,
    pub split: String,
    pub shard_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainArgs {
    pub cfg: PathBuf,
    pub manifest: PathBuf,
    pub out: PathBuf,
    pub steps: u32,
    pub batch: usize,
    pub context: usize,
    pub seed: u64,
    pub accum: u32,
    pub warmup: u64,
    pub total: u64,
    pub peak_lr: f64,
    pub min_lr: f64,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub clip_norm: f32,
    pub log_every: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResumeArgs {
    pub checkpoint: PathBuf,
    pub manifest: PathBuf,
    pub out: PathBuf,
    pub steps: u32,
    pub log_every: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerateArgs {
    pub checkpoint: PathBuf,
    pub manifest: PathBuf,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvalArgs {
    pub checkpoint: PathBuf,
    pub manifest: PathBuf,
    pub batches: usize,
}

pub fn run(args: Vec<String>) -> Result<()> {
    match parse_lm_args(args)? {
        LmCommand::Tokenizer(args) => run_tokenizer(args),
        LmCommand::Shards(args) => run_shards(args),
        LmCommand::Train(args) => run_train(args),
        LmCommand::Resume(args) => run_resume(args),
        LmCommand::Eval(args) => run_eval(args),
        LmCommand::Generate(args) => run_generate(args),
    }
}

fn run_tokenizer(args: TokenizerArgs) -> Result<()> {
    let documents = read_documents(&args.documents)?;
    if documents.is_empty() {
        return Err("no documents in the tokenizer corpus".into());
    }
    let mut bytes = Vec::with_capacity(documents.len());
    for document in &documents {
        bytes.push(fs::read(&document.text_path)?);
    }
    let tokenizer = ByteBpe::train(
        &bytes,
        args.vocab,
        NormalizationSpec {
            latin_fold: args.latin_fold,
        },
    )?;
    if let Some(parent) = args
        .out
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    tokenizer.save(&args.out)?;
    println!(
        "tokenizer {} vocab {} merges {} hash {}",
        args.out.display(),
        tokenizer.vocab_size(),
        tokenizer.merge_count(),
        tokenizer.artifact_hash()
    );
    Ok(())
}

fn run_shards(args: ShardsArgs) -> Result<()> {
    let tokenizer = ByteBpe::load(&args.tokenizer)?;
    let documents: Vec<_> = read_documents(&args.documents)?
        .into_iter()
        .filter(|document| document.split == args.split)
        .collect();
    if documents.is_empty() {
        return Err(format!("no documents for split {:?}", args.split).into());
    }
    fs::create_dir_all(&args.out)?;
    let manifest = write_token_shards(&documents, &tokenizer, &args.out, args.shard_tokens)?;
    println!(
        "shards {} documents {} tokens {} manifest {}",
        manifest.shards.len(),
        manifest.documents.len(),
        manifest
            .shards
            .iter()
            .map(|shard| shard.token_count)
            .sum::<u64>(),
        manifest.manifest_hash
    );
    Ok(())
}

fn run_train(args: TrainArgs) -> Result<()> {
    let cfg = DecoderConfig::from_cfg(&lightnet_lm::ConfigFile::parse_file(&args.cfg)?)?;
    let context = if args.context == 0 {
        cfg.context
    } else {
        if args.context > cfg.context {
            return Err(format!(
                "train context {} exceeds the decoder context {}",
                args.context, cfg.context
            )
            .into());
        }
        args.context
    };
    let schedule = lightnet_lm::TokenSchedule {
        warmup_tokens: args.warmup,
        total_tokens: args.total,
        peak_lr: args.peak_lr,
        min_lr: args.min_lr,
    };
    schedule.lr(0)?;
    let train_options = TrainOptions {
        batch: args.batch,
        context,
        accum: args.accum,
        seed: args.seed,
        beta1: args.beta1,
        beta2: args.beta2,
        eps: args.eps,
        weight_decay: args.weight_decay,
        clip_norm: args.clip_norm,
        schedule,
    };
    let engine = CpuEngine::new(cfg, args.seed)?;
    let loader = lightnet_lm::TokenLoader::open(&args.manifest, args.batch, context, args.seed)?;
    let mut trainer = Trainer::new(engine);
    train_loop(
        &mut trainer,
        loader,
        &train_options,
        args.steps,
        args.accum,
        &args.out,
        args.log_every,
    )
}

fn run_resume(args: ResumeArgs) -> Result<()> {
    let snapshot = load_checkpoint(&args.checkpoint)?;
    let manifest = read_manifest(&args.manifest)?;
    if snapshot.manifest_hash != manifest.manifest_hash {
        return Err("checkpoint belongs to a different manifest".into());
    }
    if snapshot.tokenizer_hash != manifest.tokenizer_hash {
        return Err("checkpoint belongs to a different tokenizer".into());
    }
    let mut engine = CpuEngine::new(snapshot.engine.config.clone(), snapshot.train_options.seed)?;
    engine.import_state(&snapshot.engine)?;
    let mut loader = lightnet_lm::TokenLoader::open(
        &args.manifest,
        snapshot.train_options.batch,
        snapshot.train_options.context,
        snapshot.train_options.seed,
    )?;
    loader.restore(&snapshot.loader)?;
    let mut trainer = Trainer::with_pending(
        engine,
        snapshot.pending_microbatches,
        snapshot.pending_active_tokens,
    );
    train_loop(
        &mut trainer,
        loader,
        &snapshot.train_options,
        args.steps,
        snapshot.train_options.accum,
        &args.out,
        args.log_every,
    )
}

#[allow(clippy::too_many_arguments)]
fn train_loop(
    trainer: &mut Trainer<CpuEngine>,
    mut loader: lightnet_lm::TokenLoader,
    options: &TrainOptions,
    steps: u32,
    accum: u32,
    out: &Path,
    log_every: u32,
) -> Result<()> {
    let accum = accum.max(1);
    let mut updates = 0u32;
    let mut attempts = 0u32;
    loop {
        if steps != 0 && updates >= steps {
            break;
        }
        if steps == 0 && trainer.engine().updated_tokens() >= options.schedule.total_tokens {
            break;
        }
        let mut loss_sum = 0.0f64;
        let mut active = 0u64;
        let pending = trainer.pending_microbatches();
        if pending > accum {
            return Err(format!(
                "checkpoint holds {pending} pending microbatches but --accum is {accum}"
            )
            .into());
        }
        for _ in pending..accum {
            let batch = loader
                .next_batch()?
                .ok_or("token loader produced no batch")?;
            let summary = trainer.microbatch(&batch)?;
            loss_sum += summary.sum;
            active += summary.active_tokens;
        }
        let lr = options.schedule.lr(trainer.engine().updated_tokens())?;
        let report = trainer.finish_update(UpdateOptions {
            lr,
            beta1: options.beta1,
            beta2: options.beta2,
            eps: options.eps,
            weight_decay: options.weight_decay,
            clip_norm: options.clip_norm,
            active_tokens: 0,
            step: 0,
        })?;
        if !report.applied {
            attempts += 1;
            if attempts > 16 || steps == 0 {
                break;
            }
            continue;
        }
        attempts = 0;
        updates += 1;
        if log_every != 0 && updates.is_multiple_of(log_every) {
            let mean = if active == 0 {
                0.0
            } else {
                loss_sum / active as f64
            };
            println!(
                "lm step {} tokens {} loss {:.6} lr {:.3e} norm {:.4} clip {:.4}",
                trainer.engine().step(),
                trainer.engine().updated_tokens(),
                mean,
                lr,
                report.global_norm,
                report.clip_factor,
            );
        }
        let snapshot = make_snapshot(
            trainer.engine(),
            &loader,
            options,
            trainer.pending_microbatches(),
            trainer.pending_tokens(),
        )?;
        save_checkpoint(out, &snapshot)?;
        if steps == 0 && trainer.engine().updated_tokens() >= options.schedule.total_tokens {
            break;
        }
    }
    Ok(())
}

fn make_snapshot(
    engine: &CpuEngine,
    loader: &lightnet_lm::TokenLoader,
    options: &TrainOptions,
    pending_microbatches: u32,
    pending_tokens: u64,
) -> Result<TrainingSnapshot> {
    let state = engine.export_state()?;
    let manifest = loader.manifest();
    Ok(TrainingSnapshot {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        tokenizer_hash: manifest.tokenizer_hash.clone(),
        manifest_hash: manifest.manifest_hash.clone(),
        engine: state,
        loader: loader.state(),
        train_options: options.clone(),
        pending_active_tokens: pending_tokens,
        pending_microbatches,
        successful_updates: engine.step(),
        updated_tokens: engine.updated_tokens(),
        schedule: options.schedule,
    })
}

fn run_eval(args: EvalArgs) -> Result<()> {
    let snapshot = load_checkpoint(&args.checkpoint)?;
    let manifest = read_manifest(&args.manifest)?;
    if snapshot.manifest_hash != manifest.manifest_hash {
        return Err("checkpoint belongs to a different manifest".into());
    }
    let mut engine = CpuEngine::new(snapshot.engine.config.clone(), snapshot.train_options.seed)?;
    engine.import_state(&snapshot.engine)?;
    let mut loader = lightnet_lm::TokenLoader::open(
        &args.manifest,
        snapshot.train_options.batch,
        snapshot.train_options.context,
        snapshot.train_options.seed,
    )?;
    loader.restore(&snapshot.loader)?;
    let mut sum = 0.0f64;
    let mut tokens = 0u64;
    for _ in 0..args.batches {
        let batch = loader
            .next_batch()?
            .ok_or("token loader produced no batch")?;
        let summary = engine.eval_loss_sum(&batch)?;
        sum += summary.sum;
        tokens += summary.active_tokens;
    }
    let loss = if tokens == 0 {
        0.0
    } else {
        sum / tokens as f64
    };
    println!("lm eval loss {loss:.6} tokens {tokens}");
    Ok(())
}

fn run_generate(args: GenerateArgs) -> Result<()> {
    let snapshot = load_checkpoint(&args.checkpoint)?;
    let manifest = read_manifest(&args.manifest)?;
    if snapshot.manifest_hash != manifest.manifest_hash {
        return Err("checkpoint belongs to a different manifest".into());
    }
    if snapshot.tokenizer_hash != manifest.tokenizer_hash {
        return Err("checkpoint belongs to a different tokenizer".into());
    }
    let dir = args
        .manifest
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let tokenizer = ByteBpe::load(&dir.join(&manifest.tokenizer_file))?;
    if tokenizer.artifact_hash() != manifest.tokenizer_hash {
        return Err("tokenizer artifact hash does not match the manifest".into());
    }
    let mut engine = CpuEngine::new(snapshot.engine.config.clone(), snapshot.train_options.seed)?;
    engine.import_state(&snapshot.engine)?;
    let options = lightnet_lm::SamplingOptions {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };
    let ids = lightnet_lm::generate(
        engine.decoder(),
        &tokenizer,
        &args.prompt,
        &options,
        args.max_new_tokens,
    )?;
    let bytes = tokenizer.decode_bytes(&ids)?;
    println!("{}", String::from_utf8_lossy(&bytes));
    Ok(())
}

fn read_documents(path: &Path) -> Result<Vec<lightnet_lm::CorpusDocument>> {
    let text = fs::read_to_string(path)?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut documents = Vec::new();
    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields[0] == "id" {
            continue;
        }
        if fields.len() != 7 {
            return Err(format!(
                "document line {} has {} tab-separated fields, expected 7",
                line_number + 1,
                fields.len()
            )
            .into());
        }
        let text_path = Path::new(fields[6]);
        let text_path = if text_path.is_absolute() {
            text_path.to_path_buf()
        } else {
            base.join(text_path)
        };
        documents.push(lightnet_lm::CorpusDocument {
            id: fields[0].to_owned(),
            work_id: fields[1].to_owned(),
            split: fields[2].to_owned(),
            source: fields[3].to_owned(),
            license: fields[4].to_owned(),
            sha256: fields[5].to_owned(),
            text_path,
        });
    }
    Ok(documents)
}

pub fn parse_lm_args(args: Vec<String>) -> std::result::Result<LmCommand, String> {
    let mut args = args.into_iter();
    let command = args
        .next()
        .ok_or_else(|| "lm requires a subcommand".to_owned())?;
    match command.as_str() {
        "tokenizer" => {
            let mut documents = None;
            let mut out = None;
            let mut vocab = 4096usize;
            let mut latin_fold = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--documents" => documents = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--out" => out = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--vocab" => {
                        vocab = next_value(&mut args, &arg)?
                            .parse()
                            .map_err(|_| "--vocab must be a positive integer".to_owned())?;
                    }
                    "--latin-fold" => latin_fold = true,
                    other => {
                        return Err(format!("unknown tokenizer flag: {other}"));
                    }
                }
            }
            Ok(LmCommand::Tokenizer(TokenizerArgs {
                documents: documents.ok_or("tokenizer requires --documents")?,
                out: out.ok_or("tokenizer requires --out")?,
                vocab,
                latin_fold,
            }))
        }
        "shards" => {
            let mut documents = None;
            let mut tokenizer = None;
            let mut out = None;
            let mut split = "train".to_owned();
            let mut shard_tokens = 1_000_000usize;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--documents" => documents = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--tokenizer" => tokenizer = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--out" => out = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--split" => split = next_value(&mut args, &arg)?,
                    "--shard-tokens" => {
                        shard_tokens = next_value(&mut args, &arg)?
                            .parse()
                            .map_err(|_| "--shard-tokens must be a positive integer".to_owned())?;
                    }
                    other => {
                        return Err(format!("unknown shards flag: {other}"));
                    }
                }
            }
            if shard_tokens == 0 {
                return Err("--shard-tokens must be positive".to_owned());
            }
            Ok(LmCommand::Shards(ShardsArgs {
                documents: documents.ok_or("shards requires --documents")?,
                tokenizer: tokenizer.ok_or("shards requires --tokenizer")?,
                out: out.ok_or("shards requires --out")?,
                split,
                shard_tokens,
            }))
        }
        "train" => {
            let mut cfg = None;
            let mut manifest = None;
            let mut out = None;
            let mut steps = 1u32;
            let mut batch = 4usize;
            let mut context = 0usize;
            let mut seed = 0x5EEDu64;
            let mut accum = 1u32;
            let mut warmup = 10u64;
            let mut total = 1000u64;
            let mut peak_lr = 1.0e-3f64;
            let mut min_lr = 1.0e-4f64;
            let mut beta1 = 0.9f32;
            let mut beta2 = 0.95f32;
            let mut eps = 1.0e-8f32;
            let mut weight_decay = 0.1f32;
            let mut clip_norm = 1.0f32;
            let mut log_every = 1u32;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--cfg" => cfg = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--manifest" => manifest = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--out" => out = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--steps" => steps = parse_u32(&mut args, &arg)?,
                    "--batch" => batch = parse_usize(&mut args, &arg)?,
                    "--context" => context = parse_usize(&mut args, &arg)?,
                    "--seed" => {
                        seed = next_value(&mut args, &arg)?
                            .parse()
                            .map_err(|_| "--seed must be an unsigned integer".to_owned())?
                    }
                    "--accum" => accum = parse_u32(&mut args, &arg)?,
                    "--warmup" => warmup = parse_u64(&mut args, &arg)?,
                    "--total" => total = parse_u64(&mut args, &arg)?,
                    "--peak-lr" => peak_lr = parse_f64(&mut args, &arg)?,
                    "--min-lr" => min_lr = parse_f64(&mut args, &arg)?,
                    "--beta1" => beta1 = parse_f32(&mut args, &arg)?,
                    "--beta2" => beta2 = parse_f32(&mut args, &arg)?,
                    "--eps" => eps = parse_f32(&mut args, &arg)?,
                    "--weight-decay" => weight_decay = parse_f32(&mut args, &arg)?,
                    "--clip-norm" => clip_norm = parse_f32(&mut args, &arg)?,
                    "--log-every" => log_every = parse_u32(&mut args, &arg)?,
                    other => return Err(format!("unknown train flag: {other}")),
                }
            }
            if batch == 0 {
                return Err("--batch must be positive".to_owned());
            }
            if accum == 0 {
                return Err("--accum must be positive".to_owned());
            }
            Ok(LmCommand::Train(TrainArgs {
                cfg: cfg.ok_or("train requires --cfg")?,
                manifest: manifest.ok_or("train requires --manifest")?,
                out: out.ok_or("train requires --out")?,
                steps,
                batch,
                context,
                seed,
                accum,
                warmup,
                total,
                peak_lr,
                min_lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                clip_norm,
                log_every,
            }))
        }
        "resume" => {
            let mut checkpoint = None;
            let mut manifest = None;
            let mut out = None;
            let mut steps = 1u32;
            let mut log_every = 1u32;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--checkpoint" => {
                        checkpoint = Some(PathBuf::from(next_value(&mut args, &arg)?))
                    }
                    "--manifest" => manifest = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--out" => out = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--steps" => steps = parse_u32(&mut args, &arg)?,
                    "--log-every" => log_every = parse_u32(&mut args, &arg)?,
                    other => return Err(format!("unknown resume flag: {other}")),
                }
            }
            Ok(LmCommand::Resume(ResumeArgs {
                checkpoint: checkpoint.ok_or("resume requires --checkpoint")?,
                manifest: manifest.ok_or("resume requires --manifest")?,
                out: out.ok_or("resume requires --out")?,
                steps,
                log_every,
            }))
        }
        "generate" => {
            let mut checkpoint = None;
            let mut manifest = None;
            let mut prompt = String::new();
            let mut max_new_tokens = 32usize;
            let mut temperature = 1.0f32;
            let mut top_k = 0usize;
            let mut top_p = 1.0f32;
            let mut seed = 0u64;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--checkpoint" => {
                        checkpoint = Some(PathBuf::from(next_value(&mut args, &arg)?))
                    }
                    "--manifest" => manifest = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--prompt" => prompt = next_value(&mut args, &arg)?,
                    "--max-new-tokens" => max_new_tokens = parse_usize(&mut args, &arg)?,
                    "--temperature" => temperature = parse_f32(&mut args, &arg)?,
                    "--top-k" => top_k = parse_usize(&mut args, &arg)?,
                    "--top-p" => top_p = parse_f32(&mut args, &arg)?,
                    "--seed" => seed = parse_u64(&mut args, &arg)?,
                    other => return Err(format!("unknown generate flag: {other}")),
                }
            }
            Ok(LmCommand::Generate(GenerateArgs {
                checkpoint: checkpoint.ok_or("generate requires --checkpoint")?,
                manifest: manifest.ok_or("generate requires --manifest")?,
                prompt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                seed,
            }))
        }
        "eval" => {
            let mut checkpoint = None;
            let mut manifest = None;
            let mut batches = 1usize;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--checkpoint" => {
                        checkpoint = Some(PathBuf::from(next_value(&mut args, &arg)?))
                    }
                    "--manifest" => manifest = Some(PathBuf::from(next_value(&mut args, &arg)?)),
                    "--batches" => batches = parse_usize(&mut args, &arg)?,
                    other => return Err(format!("unknown eval flag: {other}")),
                }
            }
            if batches == 0 {
                return Err("--batches must be positive".to_owned());
            }
            Ok(LmCommand::Eval(EvalArgs {
                checkpoint: checkpoint.ok_or("eval requires --checkpoint")?,
                manifest: manifest.ok_or("eval requires --manifest")?,
                batches,
            }))
        }
        other => Err(format!("unknown lm subcommand: {other}")),
    }
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_u32(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<u32, String> {
    next_value(args, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn parse_u64(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<u64, String> {
    next_value(args, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn parse_usize(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<usize, String> {
    next_value(args, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be an unsigned integer"))
}

fn parse_f32(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<f32, String> {
    next_value(args, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be a number"))
}

fn parse_f64(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> std::result::Result<f64, String> {
    next_value(args, flag)?
        .parse()
        .map_err(|_| format!("{flag} must be a number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn lm_cli_parses_train_defaults() {
        let command = parse_lm_args(strings(&[
            "train",
            "--cfg",
            "cfg/llm_smoke.cfg",
            "--manifest",
            "data/manifest.lmtxt",
            "--out",
            "runs/smoke.lmckpt",
        ]))
        .expect("valid train command");
        match command {
            LmCommand::Train(args) => {
                assert_eq!(args.batch, 4);
                assert_eq!(args.steps, 1);
                assert_eq!(args.context, 0);
                assert_eq!(args.seed, 0x5EED);
                assert_eq!((args.warmup, args.total), (10, 1000));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn lm_cli_rejects_unknown_flags_and_missing_values() {
        assert!(parse_lm_args(strings(&["eval", "--checkpoint", "x", "--lr", "1"])).is_err());
        assert!(parse_lm_args(strings(&["train", "--cfg"])).is_err());
        assert!(parse_lm_args(strings(&["nope"])).is_err());
        assert!(parse_lm_args(strings(&["eval", "--checkpoint", "x"])).is_err());
    }

    #[test]
    fn lm_cli_accepts_generate() {
        assert!(matches!(
            parse_lm_args(strings(&[
                "generate",
                "--checkpoint",
                "run.lmckpt",
                "--manifest",
                "data/manifest.lmtxt",
                "--prompt",
                "arma virumque",
                "--max-new-tokens",
                "8",
                "--temperature",
                "0",
            ])),
            Ok(LmCommand::Generate(GenerateArgs {
                max_new_tokens: 8,
                temperature: 0.0,
                ..
            }))
        ));
        assert!(
            parse_lm_args(strings(&[
                "generate",
                "--checkpoint",
                "x",
                "--manifest",
                "y",
                "--lr",
                "1",
            ]))
            .is_err()
        );
    }

    #[test]
    fn lm_cli_accepts_shards_and_tokenizer() {
        assert!(matches!(
            parse_lm_args(strings(&[
                "tokenizer",
                "--documents",
                "docs.tsv",
                "--out",
                "tok.lmbt",
                "--vocab",
                "512",
                "--latin-fold",
            ])),
            Ok(LmCommand::Tokenizer(TokenizerArgs {
                vocab: 512,
                latin_fold: true,
                ..
            }))
        ));
        assert!(matches!(
            parse_lm_args(strings(&[
                "shards",
                "--documents",
                "docs.tsv",
                "--tokenizer",
                "tok.lmbt",
                "--out",
                "data",
                "--split",
                "train",
                "--shard-tokens",
                "1024",
            ])),
            Ok(LmCommand::Shards(ShardsArgs {
                shard_tokens: 1024,
                ..
            }))
        ));
    }
}

//! Compatibility gate for the repository's complete Darknet configuration corpus.
//!
//! This test deliberately treats every `cfg/**/*.cfg` file as an acceptance
//! case.  A failure names the relative file, the section(s) in that file, and
//! the parser/model error so unsupported layer families cannot hide behind a
//! single aggregate count.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use darknet_core::{ConfigFile, Error, ExecutableNetwork, Network};

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the Darknet repository root must exist")
}

fn cfg_files(directory: &Path, files: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(directory)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()))
        .map(|entry| entry.expect("cfg directory entry must be readable").path())
        .collect();
    entries.sort();

    for path in entries {
        if path.is_dir() {
            cfg_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "cfg") {
            files.push(path);
        }
    }
}

fn section_summary(cfg: &ConfigFile) -> String {
    cfg.sections
        .iter()
        .skip(1)
        .map(|section| format!("[{}]@{}", section.name, section.line))
        .collect::<Vec<_>>()
        .join(", ")
}

fn unsupported_sections(cfg: &ConfigFile) -> String {
    // Keep this list aligned with the section aliases accepted by
    // `model::parse_layer_kind`; aliases are included because legacy cfgs use
    // both spellings (for example, `[conv]` and `[convolutional]`).
    const SUPPORTED: &[&str] = &[
        "net",
        "network",
        "conv",
        "convolutional",
        "conn",
        "connected",
        "max",
        "maxpool",
        "local_avg",
        "local_avgpool",
        "avg",
        "avgpool",
        "route",
        "shortcut",
        "upsample",
        "yolo",
        "gaussian_yolo",
        "region",
        "reorg",
        "reorg3d",
        "soft",
        "softmax",
        "cost",
        "dropout",
        "scale_channels",
        "sam",
        "contrastive",
        "crop",
        "batchnorm",
        "local",
        "detection",
        "gru",
        "rnn",
        "lstm",
        "crnn",
        "channel_slice",
        "channel_shuffle",
        "deconvolutional",
        "logistic",
        "regression",
        "jev_ce",
        "yolox",
        "ppyoloe",
        "yolonas",
        "laya_scorer",
    ];
    cfg.sections
        .iter()
        .filter(|section| !SUPPORTED.contains(&section.name.as_str()))
        .map(|section| format!("[{}]@{}", section.name, section.line))
        .collect::<Vec<_>>()
        .join(", ")
}

fn failure_section(error: &Error, cfg: &ConfigFile) -> String {
    match error {
        Error::UnsupportedLayer { layer_index, .. } => cfg
            .sections
            .get(layer_index + 1)
            .map(|section| format!("[{}]@{}", section.name, section.line))
            .unwrap_or_else(|| format!("layer index #{layer_index}")),
        _ => "no single unsupported layer identified".to_owned(),
    }
}

const CHILD_CFG_ENV: &str = "DARKNET_CFG_CORPUS_CHILD";
const EXECUTION_TIMEOUT_ENV: &str = "DARKNET_CFG_CORPUS_TIMEOUT_SECS";
const DEFAULT_EXECUTION_TIMEOUT_SECS: u64 = 300;

fn execution_timeout() -> Duration {
    let seconds = std::env::var(EXECUTION_TIMEOUT_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(DEFAULT_EXECUTION_TIMEOUT_SECS);
    Duration::from_secs(seconds)
}

fn execute_cfg(path: &Path) -> Result<(), String> {
    let network = Network::from_path(path).map_err(|error| error.to_string())?;
    let executable = ExecutableNetwork::new(network);
    executable
        .forward_inference(&executable.zero_input())
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn execute_cfg_in_child(path: &Path) -> Result<(), String> {
    let mut child = Command::new(std::env::current_exe().map_err(|error| error.to_string())?)
        .arg("--exact")
        .arg("every_cfg_file_parses_constructs_and_executes_cpu_inference")
        .arg("--nocapture")
        .env(CHILD_CFG_ENV, path)
        .spawn()
        .map_err(|error| format!("cannot spawn cfg child: {error}"))?;
    let timeout = execution_timeout();
    let started = Instant::now();
    loop {
        if child
            .try_wait()
            .map_err(|error| error.to_string())?
            .is_some()
        {
            let output = child
                .wait_with_output()
                .map_err(|error| error.to_string())?;
            if output.status.success() {
                return Ok(());
            }
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
        }
        if started.elapsed() >= timeout {
            child
                .kill()
                .map_err(|error| format!("cannot stop timed-out cfg child: {error}"))?;
            let _ = child.wait();
            return Err(format!(
                "CPU inference exceeded {} seconds",
                timeout.as_secs()
            ));
        }
        thread::sleep(Duration::from_millis(25));
    }
}

#[test]
fn every_cfg_file_parses_constructs_and_executes_cpu_inference() {
    if let Some(path) = std::env::var_os(CHILD_CFG_ENV) {
        execute_cfg(Path::new(&path)).unwrap_or_else(|error| panic!("{error}"));
        return;
    }
    let root = repository_root();
    let cfg_root = root.join("cfg");
    let mut files = Vec::new();
    cfg_files(&cfg_root, &mut files);
    assert!(
        !files.is_empty(),
        "the repository cfg corpus must contain at least one configuration"
    );

    let mut failures = Vec::new();
    for path in &files {
        let relative = path
            .strip_prefix(&root)
            .expect("cfg path must be below repository root")
            .display();
        let cfg = match ConfigFile::parse_file(path) {
            Ok(cfg) => cfg,
            Err(error) => {
                failures.push(format!("{relative}: parse failure: {error}"));
                continue;
            }
        };

        let network = match Network::from_path(path) {
            Ok(network) => network,
            Err(error) => {
                let unsupported = unsupported_sections(&cfg);
                failures.push(format!(
                    "{relative}: construction failure at {}: {error}; unsupported sections: {}; all sections: {}",
                    failure_section(&error, &cfg),
                    if unsupported.is_empty() { "none" } else { &unsupported },
                    section_summary(&cfg)
                ));
                continue;
            }
        };
        drop(network);
        eprintln!("cfg corpus executing {relative}");
        if let Err(error) = execute_cfg_in_child(path) {
            failures.push(format!(
                "{relative}: CPU inference failure at {}: {error}; all sections: {}",
                "child execution",
                section_summary(&cfg)
            ));
        }
    }

    if !failures.is_empty() {
        let mut report = String::from("Darknet cfg corpus compatibility failures:\n");
        for (index, failure) in failures.iter().enumerate() {
            writeln!(&mut report, "  {}. {failure}", index + 1).expect("in-memory write");
        }
        panic!("{report}");
    }
}

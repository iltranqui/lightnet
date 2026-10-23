use std::path::Path;

use lightnet_lm::{ConfigFile, DecoderConfig, TokenBatch};

const SMOKE: &str = include_str!("../../../cfg/llm_smoke.cfg");

fn parse(text: &str) -> lightnet_lm::Result<DecoderConfig> {
    DecoderConfig::from_cfg(&ConfigFile::parse_str("test.cfg", text)?)
}

fn replace_in(text: &str, key: &str, value: &str) -> String {
    let mut out = String::new();
    let mut replaced = false;
    for line in text.lines() {
        match line.split_once('=') {
            Some((existing, _)) if existing.trim() == key => {
                out.push_str(key);
                out.push('=');
                out.push_str(value);
                out.push('\n');
                replaced = true;
            }
            _ => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    assert!(replaced, "key {key} is not in the cfg");
    out
}

fn replace_line(key: &str, value: &str) -> String {
    replace_in(SMOKE, key, value)
}

fn replace_lines(pairs: &[(&str, &str)]) -> String {
    let mut text = SMOKE.to_owned();
    for (key, value) in pairs {
        text = replace_in(&text, key, value);
    }
    text
}

fn without_line(key: &str) -> String {
    SMOKE
        .lines()
        .filter(|line| line.split_once('=').is_none_or(|(k, _)| k.trim() != key))
        .map(|line| format!("{line}\n"))
        .collect()
}

fn with_line(line: &str) -> String {
    format!("{SMOKE}{line}\n")
}

#[test]
fn lm_contract_accepts_smoke_geometry() {
    let config = parse(SMOKE).expect("smoke cfg must parse");
    assert_eq!(config.vocab_size, 260);
    assert_eq!(config.context, 8);
    assert_eq!(config.hidden, 16);
    assert_eq!(config.intermediate, 32);
    assert_eq!(config.layers, 2);
    assert_eq!(config.q_heads, 4);
    assert_eq!(config.kv_heads, 2);
    assert_eq!(config.rope_theta, 10_000.0);
    assert_eq!(config.rms_eps, 1e-5);
    assert_eq!(config.dropout, 0.0);
    assert!(config.tie_embeddings);
    assert_eq!(config.head_dim(), 4);

    // embedding 260*16 + 2 layers * 2336 + final norm 16
    assert_eq!(config.parameter_count().expect("count fits"), 8_848);
}

#[test]
fn lm_contract_rejects_invalid_geometry() {
    for (key, value) in [
        ("vocab_size", "0"),
        ("context", "0"),
        ("hidden", "0"),
        ("intermediate", "0"),
        ("layers", "0"),
        ("q_heads", "0"),
        ("kv_heads", "0"),
        ("dropout", "1"),
        ("dropout", "-0.5"),
        ("rms_eps", "0"),
        ("rms_eps", "NaN"),
        ("rope_theta", "0"),
        ("rope_theta", "inf"),
    ] {
        assert!(
            parse(&replace_line(key, value)).is_err(),
            "{key}={value} must be rejected"
        );
    }

    // hidden 12 / q_heads 4 gives an odd head dimension.
    assert!(parse(&replace_line("hidden", "12")).is_err());
    // head ratios must divide.
    assert!(parse(&replace_line("q_heads", "3")).is_err());
    assert!(parse(&replace_line("kv_heads", "3")).is_err());

    let huge = usize::MAX.to_string();
    let overflow = replace_lines(&[("vocab_size", &huge), ("layers", &huge)]);
    match parse(&overflow) {
        Err(_) => {}
        Ok(config) => assert!(
            config.parameter_count().is_err(),
            "overflowing parameter count must be rejected"
        ),
    }
}

#[test]
fn lm_contract_rejects_unknown_options() {
    // Misspelled key.
    assert!(parse(&with_line("hidde=16")).is_err());
    // A detector [transformer] section is not a decoder.
    let transformer = SMOKE.replace("[decoder]", "[transformer]");
    assert!(parse(&transformer).is_err());
    // Unknown non-metadata section.
    assert!(parse(&with_line("[foo]\nx=1")).is_err());
    // Untied embeddings are unsupported.
    assert!(parse(&replace_line("tie_embeddings", "0")).is_err());
    // Only 0/1 are accepted for a flag.
    assert!(parse(&replace_line("tie_embeddings", "2")).is_err());
    assert!(parse(&replace_line("tie_embeddings", "true")).is_err());
    // Missing and non-numeric keys.
    assert!(parse(&without_line("hidden")).is_err());
    assert!(parse(&replace_line("hidden", "sixteen")).is_err());
    // Duplicate keys and malformed files.
    assert!(parse(&with_line("hidden=16")).is_err());
    assert!(parse(&with_line("[decoder]")).is_err());
    assert!(parse("[decoder\nvocab_size=260").is_err());
    assert!(parse("vocab_size=260").is_err());
    assert!(parse("=260").is_err());
}

#[test]
fn lm_contract_accepts_net_metadata() {
    let text = format!("{SMOKE}\n[net]\nwidth=16\n");
    assert!(parse(&text).is_ok());
}

fn valid_batch() -> TokenBatch {
    TokenBatch {
        batch: 2,
        time: 3,
        input_ids: vec![0, 1, 2, 3, 4, 5],
        targets: vec![-100, 1, 2, -100, 4, 5],
        positions: vec![0, 1, 2, 0, 1, 2],
        segments: vec![0, 0, 0, 1, 1, 1],
        valid: vec![true, true, true, false, true, true],
    }
}

#[test]
fn lm_contract_validates_integer_batches() {
    let config = parse(SMOKE).expect("smoke cfg must parse");
    valid_batch().validate(&config).expect("valid batch");

    let mut batch = valid_batch();
    batch.targets.pop();
    assert!(batch.validate(&config).is_err(), "mismatched lengths");

    let mut batch = valid_batch();
    batch.input_ids[0] = 260;
    assert!(batch.validate(&config).is_err(), "input outside vocabulary");

    let mut batch = valid_batch();
    batch.targets[0] = 260;
    assert!(
        batch.validate(&config).is_err(),
        "target outside vocabulary"
    );

    let mut batch = valid_batch();
    batch.targets[0] = -101;
    assert!(
        batch.validate(&config).is_err(),
        "target below ignore sentinel"
    );

    let mut batch = valid_batch();
    batch.targets[0] = -100;
    batch.validate(&config).expect("ignored target is allowed");

    let mut batch = valid_batch();
    batch.positions[0] = 8;
    assert!(batch.validate(&config).is_err(), "position outside context");

    let mut batch = valid_batch();
    batch.batch = 1;
    batch.time = 9;
    batch.input_ids = vec![0; 9];
    batch.targets = vec![-100; 9];
    batch.positions = vec![0; 9];
    batch.segments = vec![0; 9];
    batch.valid = vec![false; 9];
    assert!(batch.validate(&config).is_err(), "time above context");

    let mut batch = valid_batch();
    batch.batch = 0;
    batch.time = 0;
    batch.input_ids.clear();
    batch.targets.clear();
    batch.positions.clear();
    batch.segments.clear();
    batch.valid.clear();
    assert!(batch.validate(&config).is_err(), "empty batch");
}

#[test]
fn lm_contract_config_parse_reads_file_path() {
    let cfg = ConfigFile::parse_str(Path::new("smoke.cfg"), SMOKE).expect("parse");
    assert_eq!(cfg.path.as_path(), Path::new("smoke.cfg"));
    assert_eq!(
        cfg.section("DECODER").map(|s| s.name.as_str()),
        Some("decoder")
    );
}

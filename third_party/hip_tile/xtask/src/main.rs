//! `cargo xtask <task>`: project automation, no dependencies.
//!
//! - `train_hip_legogears`: train the LegoGears_v2 YOLOv4-tiny model from scratch on the AMD GPU
//!   (`ROADMAP.md`): builds `hip_darknet` in release mode, then runs
//!   `hip_darknet detector train CFG DATA -gpu`. Everything the `.cfg` says (batch, iterations,
//!   learning-rate schedule, augmentation) stays as the cfg says unless an option overrides it.

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "\
usage: cargo xtask <task> [options]

tasks:
  train_hip_legogears   train LegoGears_v2 from scratch on the AMD GPU (hip-tile)

train_hip_legogears options:
  --data PATH      the .data file   (default: ~/src/DarkSuite/datasets/LegoGears_v2/LegoGears_v2.data)
  --cfg PATH       the .cfg file    (default: the .data path with a .cfg extension)
  --out DIR        output directory (default: target/legogears_v2_gpu)
  --iters N        optimizer steps  (default: the cfg's max_batches)
  --batch N        images per step  (default: the cfg's batch)
  --subdivisions N passes per step, each batch/N images, to fit in GPU memory (default: 2)
  --seed N         RNG seed
  --weights PATH   start from these Darknet weights instead of from scratch
  --map            evaluate mAP on the validation split as the run goes
  --save-every N   checkpoint interval
  --cpu            train on the CPU reference instead (very slow)
";

struct Train {
    data: PathBuf,
    cfg: Option<PathBuf>,
    out: PathBuf,
    iters: Option<String>,
    batch: Option<String>,
    subdivisions: String,
    seed: Option<String>,
    weights: Option<PathBuf>,
    save_every: Option<String>,
    map: bool,
    cpu: bool,
}

fn workspace_root() -> PathBuf {
    // xtask lives at <root>/xtask.
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("xtask has a parent").to_path_buf()
}

fn parse_train(args: &[String]) -> Result<Train, String> {
    let home = std::env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
    let mut t = Train {
        data: Path::new(&home).join("src/DarkSuite/datasets/LegoGears_v2/LegoGears_v2.data"),
        cfg: None,
        out: workspace_root().join("target/legogears_v2_gpu"),
        iters: None,
        batch: None,
        subdivisions: "2".into(),
        seed: None,
        weights: None,
        save_every: None,
        map: false,
        cpu: false,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |what: &str| it.next().cloned().ok_or(format!("{what} needs a value"));
        match a.as_str() {
            "--data" => t.data = val("--data")?.into(),
            "--cfg" => t.cfg = Some(val("--cfg")?.into()),
            "--out" => t.out = val("--out")?.into(),
            "--iters" => t.iters = Some(val("--iters")?),
            "--batch" => t.batch = Some(val("--batch")?),
            "--subdivisions" => t.subdivisions = val("--subdivisions")?,
            "--seed" => t.seed = Some(val("--seed")?),
            "--weights" => t.weights = Some(val("--weights")?.into()),
            "--save-every" => t.save_every = Some(val("--save-every")?),
            "--map" => t.map = true,
            "--cpu" => t.cpu = true,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown option {other}\n{USAGE}")),
        }
    }
    if !t.data.exists() {
        return Err(format!("{} not found (pass --data)", t.data.display()));
    }
    Ok(t)
}

/// `systemd-run --user --scope -p MemoryHigh=3G <cmd>` when available: a soft cgroup cap that
/// throttles instead of triggering the OOM killer (this machine's rule for builds and runs).
fn capped(program: &str) -> Command {
    let has = Command::new("systemd-run").arg("--version").output().is_ok_and(|o| o.status.success());
    if has {
        let mut c = Command::new("systemd-run");
        c.args(["--user", "--scope", "-q", "-p", "MemoryHigh=3G", program]);
        c
    } else {
        eprintln!("xtask: systemd-run not found, running without the memory cap");
        Command::new(program)
    }
}

fn run(mut cmd: Command, what: &str) -> Result<(), String> {
    eprintln!("xtask: {what}");
    let status = cmd.status().map_err(|e| format!("{what}: {e}"))?;
    if status.success() { Ok(()) } else { Err(format!("{what} failed ({status})")) }
}

fn train_hip_legogears(args: &[String]) -> Result<(), String> {
    let t = parse_train(args)?;
    let root = workspace_root();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());

    let mut build = capped(&cargo);
    build.current_dir(&root).args(["build", "--release", "-j", "4", "-p", "hip_darknet"]);
    run(build, "cargo build --release -p hip_darknet")?;

    let cfg = t.cfg.clone().unwrap_or_else(|| t.data.with_extension("cfg"));
    let bin = root.join("target/release/hip_darknet");
    let mut train = capped(&bin.to_string_lossy());
    train.current_dir(&root).arg("detector").arg("train").arg(&cfg).arg(&t.data);
    if !t.cpu {
        train.arg("-gpu");
    }
    train.arg("-out").arg(&t.out);
    if let Some(v) = &t.batch {
        train.args(["-batch", v]);
    }
    train.args(["-subdivisions", &t.subdivisions]);
    if let Some(v) = &t.iters {
        train.args(["-iters", v]);
    }
    if let Some(v) = &t.seed {
        train.args(["-seed", v]);
    }
    if let Some(v) = &t.save_every {
        train.args(["-save_every", v]);
    }
    if let Some(v) = &t.weights {
        train.arg("-weights").arg(v);
    }
    if t.map {
        train.arg("-map");
    }
    std::fs::create_dir_all(&t.out).map_err(|e| format!("{}: {e}", t.out.display()))?;
    run(train, &format!("hip_darknet detector train {} (output {})", t.data.display(), t.out.display()))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("train_hip_legogears") => train_hip_legogears(&args[1..]),
        Some("-h" | "--help") | None => Err(USAGE.into()),
        Some(other) => Err(format!("unknown task {other}\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

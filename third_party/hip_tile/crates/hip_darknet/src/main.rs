//! `hip_darknet` CLI: run a Darknet YOLO model on images (CPU path) and print Darknet-style lines.

use hip_darknet::{Detection, DetectOptions, Model, detect_cpu, image::Image, load_names};
use std::path::Path;
use std::time::Instant;

const USAGE: &str =
    "usage: hip_darknet --cfg C --weights W --names N [--gpu] [--thresh 0.25] [--nms 0.45] [--out PATH] [--bench N] image...
       hip_darknet train --data D [--cfg C] [--gpu] [--iters N] [--batch N] [--subdivisions N] [--seed S] [--out DIR]
                         [--save-every N] [--eval-every N] [--eval] [--weights W] [--resume STATE]
       hip_darknet detector train CFG DATA [-gpu] [-map] [-dont_show] [-gpus N] [-iters N] [-batch N] [-out DIR]
  --gpu       run the forward pass on the AMD GPU (needs the default `gpu` feature)
  --out PATH  write annotated image(s); a directory when several images are given
  --bench N   time N forwards on the first image (CPU and, with --gpu, GPU)

train: the CPU reference trainer, or the AMD GPU trainer with `--gpu`. `--cfg` defaults to the dataset's own cfg next to the `.data`
file; `--iters` is optimizer steps, `--batch` overrides the cfg's batch.

`detector train CFG DATA` is the Darknet-shaped form: the two paths may be given in either order
(they are told apart by extension), `-map` turns on the periodic mAP, `-dont_show`/`-gpus`/`-clear`
are accepted and ignored, and checkpoints go next to the `.data` file like Darknet's `backup`.";

struct Args {
    cfg: String,
    weights: String,
    names: String,
    opts: DetectOptions,
    images: Vec<String>,
    gpu: bool,
    out: Option<String>,
    bench: Option<usize>,
}

fn parse_args() -> Result<Args, String> {
    let (mut cfg, mut weights, mut names) = (None, None, None);
    let mut opts = DetectOptions::default();
    let mut images = vec![];
    let (mut gpu, mut out, mut bench) = (false, None, None);
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut val = |what: &str| {
            i += 1;
            argv.get(i).cloned().ok_or(format!("{what} needs a value"))
        };
        match a {
            "--cfg" => cfg = Some(val("--cfg")?),
            "--weights" => weights = Some(val("--weights")?),
            "--names" => names = Some(val("--names")?),
            "--thresh" => opts.thresh = val("--thresh")?.parse().map_err(|e| format!("--thresh: {e}"))?,
            "--nms" => opts.nms_thresh = val("--nms")?.parse().map_err(|e| format!("--nms: {e}"))?,
            "--gpu" => gpu = true,
            "--out" => out = Some(val("--out")?),
            "--bench" => bench = Some(val("--bench")?.parse().map_err(|e| format!("--bench: {e}"))?),
            "-h" | "--help" => return Err(USAGE.into()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}")),
            _ => images.push(a.to_string()),
        }
        i += 1;
    }
    let need = |v: Option<String>, n: &str| v.ok_or(format!("missing {n}\n{USAGE}"));
    Ok(Args { cfg: need(cfg, "--cfg")?, weights: need(weights, "--weights")?, names: need(names, "--names")?, opts, images, gpu, out, bench })
}

/// One detector: CPU, or GPU with the weights resident.
enum Backend<'m> {
    Cpu(&'m Model),
    #[cfg(feature = "gpu")]
    Gpu(&'m Model, hip_darknet::gpu::GpuModel<'m>),
}

impl Backend<'_> {
    fn detect(&self, img: &Image, opts: &DetectOptions) -> hip_darknet::Result<Vec<Detection>> {
        match self {
            Backend::Cpu(m) => detect_cpu(m, img, opts),
            #[cfg(feature = "gpu")]
            Backend::Gpu(m, g) => hip_darknet::detect_gpu(m, g, img, opts),
        }
    }
}

#[cfg(feature = "gpu")]
fn make_gpu(model: &Model) -> Result<Backend<'_>, Box<dyn std::error::Error>> {
    Ok(Backend::Gpu(model, hip_darknet::gpu::GpuModel::new(model)?))
}

#[cfg(not(feature = "gpu"))]
fn make_gpu(_: &Model) -> Result<Backend<'_>, Box<dyn std::error::Error>> {
    Err("--gpu: built without the `gpu` feature (rebuild with default features)".into())
}

/// Best-of-rounds mean ms per call (the iGPU clock varies with the shared power budget).
fn best_ms(rounds: usize, n: usize, mut f: impl FnMut()) -> f64 {
    (0..rounds)
        .map(|_| {
            let t = Instant::now();
            (0..n).for_each(|_| f());
            t.elapsed().as_secs_f64() * 1e3 / n as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn bench(model: &Model, gpu: bool, img: &Image, opts: &DetectOptions, n: usize) -> Result<(), Box<dyn std::error::Error>> {
    let sized = img.resize(model.net.width, model.net.height);
    let cpu_n = n.clamp(1, 20);
    detect_cpu(model, img, opts)?;
    let cpu = best_ms(3, cpu_n, || {
        hip_darknet::reference::forward(model, &sized.data).unwrap();
    });
    let cpu_e2e = best_ms(3, cpu_n, || {
        detect_cpu(model, img, opts).unwrap();
    });
    println!("cpu: forward {cpu:.2} ms/frame, end-to-end {cpu_e2e:.2} ms/frame (best of 3 x {cpu_n})");
    if gpu {
        #[cfg(feature = "gpu")]
        {
            let g = hip_darknet::gpu::GpuModel::new(model)?;
            let be = Backend::Gpu(model, g);
            let Backend::Gpu(_, g) = &be else { unreachable!() };
            for _ in 0..10 {
                g.run(&sized.data)?;
                be.detect(img, opts)?;
            }
            let fwd = best_ms(5, n, || g.run(&sized.data).unwrap());
            let e2e = best_ms(5, n, || {
                be.detect(img, opts).unwrap();
            });
            println!(
                "gpu: forward {fwd:.2} ms/frame (upload + kernels + sync), end-to-end {e2e:.2} ms/frame (resize, upload, download, decode; best of 5 x {n}); speedup {:.1}x fwd, {:.1}x e2e",
                cpu / fwd, cpu_e2e / e2e
            );
        }
    }
    Ok(())
}

fn out_path(out: &str, many: bool, img: &str) -> std::path::PathBuf {
    if many {
        let stem = Path::new(img).file_stem().map_or("out".into(), |s| s.to_string_lossy().into_owned());
        Path::new(out).join(format!("{stem}.png"))
    } else {
        out.into()
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args()?;
    let model = Model::load(&a.cfg, &a.weights)?;
    let names = load_names(&a.names)?;
    if let Some(n) = a.bench {
        let first = a.images.first().ok_or("--bench needs an image")?;
        return bench(&model, a.gpu, &Image::load(first)?, &a.opts, n.max(1));
    }
    let backend = if a.gpu { make_gpu(&model)? } else { Backend::Cpu(&model) };
    let many = a.images.len() > 1;
    if let (Some(out), true) = (&a.out, many) {
        std::fs::create_dir_all(out)?;
    }
    for path in &a.images {
        let img = Image::load(path)?;
        let t = Instant::now();
        let dets = backend.detect(&img, &a.opts)?;
        println!("{path}: predicted in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        for d in &dets {
            let (l, t, w, h) = d.pixel_rect(img.w, img.h);
            let name = names.get(d.class).map_or("?", |s| s.as_str());
            println!(
                "{name}: {:.0}% (left_x: {l:4.0} top_y: {t:4.0} width: {w:4.0} height: {h:4.0})",
                d.prob * 100.0
            );
        }
        if let Some(out) = &a.out {
            let p = out_path(out, many, path);
            hip_darknet::draw::save_annotated(&img, &dets, &p)?;
            println!("wrote {}", p.display());
        }
    }
    Ok(())
}

/// `hip_darknet train …`: the CPU reference trainer (see `ROADMAP.md` for the GPU one).
fn run_train(argv: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use hip_darknet::train::{TrainOptions, train};
    let mut data: Option<String> = None;
    let mut cfg_path: Option<String> = None;
    let mut opts = TrainOptions { out_dir: "train_out".into(), ..TrainOptions::default() };
    let mut iters_given = false;
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut val = |what: &str| -> Result<String, String> {
            i += 1;
            argv.get(i).cloned().ok_or(format!("{what} needs a value"))
        };
        match a {
            "--data" => data = Some(val("--data")?),
            "--cfg" => cfg_path = Some(val("--cfg")?),
            "--iters" => {
                opts.iters = val("--iters")?.parse().map_err(|e| format!("--iters: {e}"))?;
                iters_given = true;
            }
            "--batch" => opts.batch = Some(val("--batch")?.parse().map_err(|e| format!("--batch: {e}"))?),
            "--subdivisions" => {
                opts.subdivisions = Some(val("--subdivisions")?.parse().map_err(|e| format!("--subdivisions: {e}"))?)
            }
            "--seed" => opts.seed = val("--seed")?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--out" => opts.out_dir = val("--out")?.into(),
            "--save-every" => opts.save_every = val("--save-every")?.parse().map_err(|e| format!("--save-every: {e}"))?,
            "--eval-every" => opts.eval_every = val("--eval-every")?.parse().map_err(|e| format!("--eval-every: {e}"))?,
            "--eval" => opts.eval = true,
            "--gpu" => opts.gpu = true,
            "--weights" => opts.init_weights = Some(val("--weights")?.into()),
            "--resume" => opts.resume = Some(val("--resume")?.into()),
            "-h" | "--help" => return Err(USAGE.into()),
            s if s.starts_with("--") => return Err(format!("unknown option {s}\n{USAGE}").into()),
            s => return Err(format!("unexpected argument {s}\n{USAGE}").into()),
        }
        i += 1;
    }
    let data = data.ok_or_else(|| format!("train needs --data\n{USAGE}"))?;
    let ds = hip_darknet::data::Dataset::load(&data)?;
    let cfg_text = match cfg_path {
        Some(p) => std::fs::read_to_string(&p)?,
        None => {
            // Default: the dataset's own cfg, i.e. the `.data` path with the cfg extension.
            let p = std::path::Path::new(&data).with_extension("cfg");
            std::fs::read_to_string(&p).map_err(|e| format!("--cfg not given and {}: {e}", p.display()))?
        }
    };
    let cfg = hip_darknet::cfg::Cfg::parse(&cfg_text)?;
    if !iters_given {
        // Default to the cfg's own schedule so a bare `train` does what the cfg says.
        opts.iters = if cfg.train.max_batches > 0 { cfg.train.max_batches } else { opts.iters };
    }
    let summary = train(&cfg, &ds, &opts)?;
    println!(
        "trained {} iterations ({} images seen) in {:.1}s: loss {:.4} -> {:.4} (avg {:.4}){}",
        summary.iterations,
        summary.seen,
        summary.seconds,
        summary.first_loss,
        summary.last_loss,
        summary.avg_loss,
        if opts.eval { format!(", mAP@0.5 {:.4}, mAP@0.5:0.95 {:.4}", summary.map_50, summary.map_50_95) } else { String::new() }
    );
    println!("weights: {}", summary.final_weights.display());
    println!("csv: {} and {}", opts.out_dir.join("loss.csv").display(), opts.out_dir.join("map.csv").display());
    Ok(())
}

/// `hip_darknet detector train CFG DATA …` — Darknet's command shape, translated onto
/// [`run_train`]'s options. Paths are told apart by extension, so either order works.
fn run_detector(argv: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if argv.first().map(String::as_str) != Some("train") {
        return Err(format!("detector: only `train` is implemented\n{USAGE}").into());
    }
    let mut paths: Vec<String> = vec![];
    let mut train_argv: Vec<String> = vec![];
    let mut eval = false;
    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].as_str();
        let mut val = |what: &str| -> Result<String, String> {
            i += 1;
            argv.get(i).cloned().ok_or(format!("{what} needs a value"))
        };
        match a {
            // Darknet flags we accept and ignore (no display, single GPU, no momentum reset).
            "-dont_show" | "-clear" | "-show_imgs" => {}
            "-gpus" | "-i" => {
                let _ = val(a)?;
            }
            "-map" => eval = true,
            "-gpu" => train_argv.push("--gpu".into()),
            // Our own knobs, in Darknet's `-flag value` spelling.
            "-iters" => train_argv.extend(["--iters".into(), val("-iters")?]),
            "-batch" => train_argv.extend(["--batch".into(), val("-batch")?]),
            "-subdivisions" => train_argv.extend(["--subdivisions".into(), val("-subdivisions")?]),
            "-seed" => train_argv.extend(["--seed".into(), val("-seed")?]),
            "-out" => train_argv.extend(["--out".into(), val("-out")?]),
            "-save_every" => train_argv.extend(["--save-every".into(), val("-save_every")?]),
            "-map_every" => train_argv.extend(["--eval-every".into(), val("-map_every")?]),
            "-weights" => train_argv.extend(["--weights".into(), val("-weights")?]),
            s if s.starts_with('-') => {
                return Err(format!("detector train: unknown option {s}\n{USAGE}").into());
            }
            s => paths.push(s.to_string()),
        }
        i += 1;
    }
    let (mut cfg, mut data) = (None, None);
    for p in paths {
        match std::path::Path::new(&p).extension().and_then(|e| e.to_str()) {
            Some("cfg") => cfg = Some(p),
            Some("data") => data = Some(p),
            _ => return Err(format!("detector train: expected a .cfg and a .data file, got {p}\n{USAGE}").into()),
        }
    }
    let data = data.ok_or_else(|| format!("detector train: missing the .data file\n{USAGE}"))?;
    train_argv.extend(["--data".into(), data.clone()]);
    if let Some(cfg) = cfg {
        train_argv.extend(["--cfg".into(), cfg]);
    }
    if eval {
        train_argv.push("--eval".into());
        // mAP every 100 iterations once burn-in is over (`-map_every` overrides).
        if !train_argv.iter().any(|a| a == "--eval-every") {
            train_argv.extend(["--eval-every".into(), "100".into()]);
        }
    }
    // Darknet writes checkpoints to the `.data` file's `backup` directory; ours defaults to the
    // `.data` file's own directory unless `-out` was given.
    if !train_argv.iter().any(|a| a == "--out") {
        let dir = std::path::Path::new(&data).parent().unwrap_or(std::path::Path::new(".")).join("hip_train");
        train_argv.extend(["--out".into(), dir.to_string_lossy().into_owned()]);
    }
    run_train(&train_argv)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("detector") {
        if let Err(e) = run_detector(&argv[1..]) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }
    if argv.first().map(String::as_str) == Some("train") {
        if let Err(e) = run_train(&argv[1..]) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        return;
    }
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

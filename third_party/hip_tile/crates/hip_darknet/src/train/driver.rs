//! The training loop: sample → augment → forward → deltas → backward → SGD, with `loss.csv`,
//! `map.csv`, Darknet-format checkpoints and resume.
//!
//! Two devices share this loop: the CPU reference ([`TrainNet`]) and, with `opts.gpu`, the GPU
//! trainer ([`crate::train::gpu::GpuTrainNet`]). The GPU net is created from the CPU net (so
//! `--weights`/`--resume` apply to both) and copied back into it at every checkpoint and mAP
//! evaluation, so the file formats and the schedule do not change between devices.

use crate::augment::{Augment, Rng};
use crate::cfg::Cfg;
use crate::data::{Dataset, Sample};
use crate::error::{Error, Result};
use crate::train::checkpoint;
use crate::train::net::{learning_rate, TrainNet};
use crate::yolo::DetectOptions;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Formats a loss value, blue when it fell versus `prev` and red when it rose.
/// Plain text when stderr is not a terminal or on the first iteration.
fn trend(value: f32, prev: f32, first: bool) -> String {
    let text = format!("{value:.4}");
    if first || !std::io::stderr().is_terminal() {
        return text;
    }
    let code = if value < prev { 34 } else if value > prev { 31 } else { return text };
    format!("\x1b[{code}m{text}\x1b[0m")
}

/// Per-class TP / FP / FN table (confidence 0.25, IoU 0.5) with a total row, as lightnet prints it.
/// There is no TN column: a detector has no countable true negatives.
fn counts_table(counts: &[crate::metrics::ClassCounts], names: &[String]) -> String {
    use crate::metrics::{ClassCounts, REPORT_CONFIDENCE};
    let (tp, fp, fn_) = (tint("TP", 32), tint("FP", 31), tint("FN", 33));
    let mut out = format!(
        "  {tp}={}, {fp}={}, {fn_}={}, GT=ground truth (conf >= {REPORT_CONFIDENCE}, IoU 0.5)\n  \
         Id Name                     {tp}     {fp}     {fn_}     GT     F1 AvgIoU\n",
        tint("true positive", 32),
        tint("false positive", 31),
        tint("false negative", 33),
    );
    let mut total = ClassCounts::default();
    let mut iou_sum = 0.0f32;
    for (id, c) in counts.iter().enumerate() {
        total.tp += c.tp;
        total.fp += c.fp;
        total.fn_ += c.fn_;
        total.gt += c.gt;
        iou_sum += c.avg_iou * c.tp as f32;
        if c.gt == 0 && c.tp + c.fp == 0 {
            continue;
        }
        let name = names.get(id).map_or("", String::as_str);
        out += &format!(
            "  {id:>2} {name:<20.20} {} {} {} {:>6} {:>5.1}% {:>5.1}%\n",
            paint(c.tp, 32), paint(c.fp, 31), paint(c.fn_, 33), c.gt, 100.0 * c.f1(), 100.0 * c.avg_iou
        );
    }
    total.avg_iou = if total.tp == 0 { 0.0 } else { iou_sum / total.tp as f32 };
    out += &format!(
        "     {:<20} {} {} {} {:>6} {:>5.1}% {:>5.1}%\n",
        "total", paint(total.tp, 32), paint(total.fp, 31), paint(total.fn_, 33), total.gt, 100.0 * total.f1(),
        100.0 * total.avg_iou
    );
    out
}

/// A count right-aligned to 6 columns, in ANSI colour `code` (green TP, red FP, yellow FN) when
/// stderr is a terminal and the count is not zero.
fn paint(n: usize, code: u8) -> String {
    let text = format!("{n:>6}");
    if n == 0 { text } else { tint(&text, code) }
}

/// `text` in ANSI colour `code` when stderr is a terminal.
fn tint(text: &str, code: u8) -> String {
    if std::io::stderr().is_terminal() { format!("\x1b[{code}m{text}\x1b[0m") } else { text.to_string() }
}

/// What a run should do.
#[derive(Debug, Clone)]
pub struct TrainOptions {
    /// Iterations (one optimizer step each, `batch` images per step).
    pub iters: usize,
    /// Images per step; overrides the cfg's `batch` when `Some`.
    pub batch: Option<usize>,
    /// Passes per step, each of `batch / subdivisions` images; overrides the cfg's when `Some`.
    pub subdivisions: Option<usize>,
    pub seed: u64,
    pub save_every: usize,
    pub eval_every: usize,
    pub out_dir: PathBuf,
    /// Start from these weights (unfolded, Darknet format).
    pub init_weights: Option<PathBuf>,
    /// Resume weights + momentum + `seen` (our own state format).
    pub resume: Option<PathBuf>,
    /// Evaluate the validation split (mAP) as the run goes.
    pub eval: bool,
    /// Train on the AMD GPU instead of the CPU reference (needs the `gpu` feature).
    pub gpu: bool,
}

impl Default for TrainOptions {
    fn default() -> Self {
        TrainOptions {
            iters: 100,
            batch: None,
            subdivisions: None,
            seed: 1,
            save_every: 100,
            eval_every: 100,
            out_dir: PathBuf::from("."),
            init_weights: None,
            resume: None,
            eval: false,
            gpu: false,
        }
    }
}

/// What a finished (or partial) run produced.
#[derive(Debug, Clone, Default)]
pub struct TrainSummary {
    pub iterations: usize,
    pub seen: u64,
    pub first_loss: f32,
    pub last_loss: f32,
    pub avg_loss: f32,
    pub seconds: f64,
    pub final_weights: PathBuf,
    pub map_50: f32,
    pub map_50_95: f32,
}

/// Per-iteration numbers, matching the reference csv columns plus a per-image column.
#[derive(Debug, Clone, Copy, Default)]
pub struct LossRow {
    pub iteration: usize,
    pub seen: u64,
    pub loss: f32,
    pub avg_loss: f32,
    pub loss_per_image: f32,
    pub learning_rate: f32,
    pub ms: f32,
}

fn augment_for(cfg: &Cfg) -> Augment {
    // Jitter/resize live on the `[yolo]` heads; the colour keys live on `[net]`. They must agree.
    let mut jitter: Option<f32> = None;
    let mut resize: Option<f32> = None;
    for l in &cfg.layers {
        if let crate::cfg::LayerCfg::Yolo(y) = l {
            match jitter {
                None => jitter = Some(y.train.jitter),
                Some(j) if (j - y.train.jitter).abs() > 1e-6 => {
                    eprintln!("warning: [yolo] heads disagree on `jitter`; using {j}")
                }
                _ => {}
            }
            match resize {
                None => resize = Some(y.train.resize),
                Some(r) if (r - y.train.resize).abs() > 1e-6 => {
                    eprintln!("warning: [yolo] heads disagree on `resize`; using {r}")
                }
                _ => {}
            }
        }
    }
    let t = &cfg.train;
    Augment {
        jitter: jitter.unwrap_or(0.0),
        resize: resize.unwrap_or(1.0),
        flip: t.flip,
        hue: t.hue,
        saturation: t.saturation,
        exposure: t.exposure,
    }
}

fn next_batch(rng: &mut Rng, order: &mut [usize], cursor: &mut usize, batch: usize, n: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(batch);
    while out.len() < batch {
        if *cursor >= order.len() {
            // Fisher-Yates with our own PRNG: a fresh epoch order.
            for i in (1..n).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            *cursor = 0;
        }
        out.push(order[*cursor]);
        *cursor += 1;
    }
    out
}

/// Write the `loss.csv` header the reference dataset uses, plus `loss_per_image`.
pub const LOSS_CSV_HEADER: &str = "iteration,seen,loss,avg_loss,loss_per_image,learning_rate,milliseconds";
/// `map.csv` header, matching the reference dataset's columns.
pub const MAP_CSV_HEADER: &str = "iteration,seen,iou,threshold,map,best_map";

#[cfg(feature = "gpu")]
type GpuHandle = crate::train::gpu::GpuTrainNet;
#[cfg(not(feature = "gpu"))]
type GpuHandle = ();

#[cfg(feature = "gpu")]
fn start_gpu(want: bool, net: &TrainNet) -> Result<Option<GpuHandle>> {
    if !want {
        return Ok(None);
    }
    crate::train::gpu::GpuTrainNet::from_net(net).map(Some)
}

#[cfg(not(feature = "gpu"))]
fn start_gpu(want: bool, _net: &TrainNet) -> Result<Option<GpuHandle>> {
    if want {
        return Err(Error::Unsupported("--gpu: this build has no `gpu` feature".into()));
    }
    Ok(None)
}

/// Forward, loss and backward of one pass on the device; returns the pass loss. The gradients
/// accumulate in the momentum buffers until [`gpu_update`].
#[cfg(feature = "gpu")]
fn gpu_pass(g: &mut GpuHandle, input: &[f32], truths: &[Vec<crate::data::BoxLabel>]) -> Result<f32> {
    g.zero_grads()?;
    g.forward(input)?;
    let loss = g.loss_and_deltas(truths)?;
    g.backward()?;
    Ok(loss)
}

#[cfg(not(feature = "gpu"))]
fn gpu_pass(_: &mut GpuHandle, _: &[f32], _: &[Vec<crate::data::BoxLabel>]) -> Result<f32> {
    unreachable!("start_gpu refuses without the gpu feature")
}

/// The SGD step on the device, once per iteration after all passes.
#[cfg(feature = "gpu")]
fn gpu_update(g: &mut GpuHandle, lr: f32) -> Result<()> {
    g.update(lr)
}

#[cfg(not(feature = "gpu"))]
fn gpu_update(_: &mut GpuHandle, _: f32) -> Result<()> {
    unreachable!("start_gpu refuses without the gpu feature")
}

/// Detector settings for mAP: decode down to confidence 0.005 so AP sees the whole
/// precision-recall curve (lightnet's `map_thresh`; the TP/FP/FN table still counts only
/// `REPORT_CONFIDENCE` and up), and suppress duplicates with CIoU NMS.
fn map_detect_options() -> DetectOptions {
    DetectOptions { thresh: 0.005, nms_kind: Some(crate::cfg::NmsKind::Ciou), ..DetectOptions::default() }
}

/// mAP of `model` on `samples`, with the forward on the GPU when `gpu` (the inference model is
/// built per call and freed after, so it does not stay resident next to the training buffers).
#[cfg(feature = "gpu")]
fn evaluate_map(
    model: &crate::model::Model,
    gpu: bool,
    classes: usize,
    samples: &[Sample],
    load: impl Fn(&Sample) -> Result<crate::image::Image>,
) -> Result<crate::metrics::MapResult> {
    let opts = map_detect_options();
    if gpu {
        let g = crate::gpu::GpuModel::new(model)?;
        crate::metrics::evaluate(classes, samples, load, |img| crate::detect_gpu(model, &g, img, &opts))
    } else {
        crate::metrics::evaluate(classes, samples, load, |img| crate::detect_cpu(model, img, &opts))
    }
}

#[cfg(not(feature = "gpu"))]
fn evaluate_map(
    model: &crate::model::Model,
    _gpu: bool,
    classes: usize,
    samples: &[Sample],
    load: impl Fn(&Sample) -> Result<crate::image::Image>,
) -> Result<crate::metrics::MapResult> {
    let opts = map_detect_options();
    crate::metrics::evaluate(classes, samples, load, |img| crate::detect_cpu(model, img, &opts))
}

/// Load and augment the images at `picks` into one NCHW input and their ground truths.
fn prepare_pass(
    rng: &mut Rng,
    ds: &Dataset,
    aug: &Augment,
    picks: &[usize],
    net_w: usize,
    net_h: usize,
) -> Result<(Vec<f32>, Vec<Vec<crate::data::BoxLabel>>)> {
    let mut input = Vec::with_capacity(picks.len() * 3 * net_h * net_w);
    let mut truths = Vec::with_capacity(picks.len());
    for &idx in picks {
        let sample: &Sample = &ds.train[idx];
        let img = ds.image(sample)?;
        let prep = crate::augment::prepare(rng, &img, &sample.labels, net_w as u32, net_h as u32, aug);
        if prep.data.len() != 3 * net_h * net_w {
            return Err(Error::Model(format!(
                "{}: prepared image has {} floats, expected {} (cfg channels must be 3)",
                sample.image.display(),
                prep.data.len(),
                3 * net_h * net_w
            )));
        }
        input.extend_from_slice(&prep.data);
        truths.push(prep.labels);
    }
    Ok((input, truths))
}

/// Copy the device parameters back into the CPU net (checkpoints, evaluation).
#[cfg(feature = "gpu")]
fn gpu_sync(g: &GpuHandle, net: &mut TrainNet) -> Result<()> {
    g.download_into(net)
}

#[cfg(not(feature = "gpu"))]
fn gpu_sync(_: &GpuHandle, _: &mut TrainNet) -> Result<()> {
    unreachable!("start_gpu refuses without the gpu feature")
}

/// Run the training loop on the CPU, or on the GPU with `opts.gpu`.
pub fn train(cfg: &Cfg, ds: &Dataset, opts: &TrainOptions) -> Result<TrainSummary> {
    let classes = ds.classes;
    // Darknet's `batch` is the images per optimizer step; each pass holds `batch / subdivisions`.
    let full = opts.batch.unwrap_or(cfg.train.batch).max(1);
    let subdiv = opts.subdivisions.unwrap_or(cfg.train.subdivisions).max(1);
    if !full.is_multiple_of(subdiv) {
        return Err(Error::Model(format!("batch {full} is not divisible by subdivisions {subdiv}")));
    }
    let batch = full / subdiv;
    // The net's `effective_batch()` (`batch * subdivisions`) must equal the images per step.
    let mut cfg = cfg.clone();
    cfg.train.batch = batch;
    cfg.train.subdivisions = subdiv;
    let cfg = &cfg;
    let mut net = TrainNet::new(cfg, batch, opts.seed)?;
    let mut seen: u64 = 0;
    if let Some(p) = &opts.resume {
        seen = checkpoint::load_state(p, &mut net)?;
    } else if let Some(p) = &opts.init_weights {
        seen = checkpoint::load_weights(p, &mut net)?;
    }
    let mut gpu = start_gpu(opts.gpu, &net)?;
    let aug = augment_for(cfg);
    let (net_w, net_h) = (net.net_w(), net.net_h());
    let mut rng = Rng::new(opts.seed);
    let mut loss_csv = BufCsv::create(&opts.out_dir.join("loss.csv"), LOSS_CSV_HEADER)?;
    let mut map_csv = BufCsv::create(&opts.out_dir.join("map.csv"), MAP_CSV_HEADER)?;
    let mut order: Vec<usize> = (0..ds.train.len()).collect();
    let mut cursor = 0usize;
    let mut ema = -1.0f32;
    let mut best_map = 0.0f32;
    let start = Instant::now();
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    let mut final_weights = opts.out_dir.join("last.weights");
    let mut map_50 = 0.0f32;
    let mut map_50_95 = 0.0f32;
    let mut eval_model: Option<crate::model::Model> = None;

    // A header, so a run that is working is visibly working: at batch 4 the first iterations take
    // ~20 s each, and silent gaps look like a hang.
    eprintln!(
        "hip_darknet train — device: {}\n  \
         {} training images, batch {batch}, {iters} iterations, {} parameters, seed {}\n  \
         augmentation: jitter {:.2} resize {:.2} flip {} hue {:.2} sat {:.2} exp {:.2}",
        if gpu.is_some() { "AMD GPU (hip-tile)" } else { "CPU (reference trainer)" },
        ds.train.len(),
        net.n_params(),
        opts.seed,
        aug.jitter,
        aug.resize,
        aug.flip,
        aug.hue,
        aug.saturation,
        aug.exposure,
        iters = opts.iters,
    );

    // Darknet's iteration count is `seen / batch`, so weights or a state that already saw images
    // continue the learning-rate schedule (and the burn-in gate) instead of restarting it.
    let base = (seen / full as u64) as usize;
    for step in 1..=opts.iters {
        let iteration = base + step;
        let t0 = Instant::now();
        let lr = learning_rate(&cfg.train, iteration.saturating_sub(1));
        // `subdiv` passes of `batch` images each accumulate into the momentum buffers; one update.
        let mut loss_sum = 0.0f32;
        for _ in 0..subdiv {
            let picks = next_batch(&mut rng, &mut order, &mut cursor, batch, ds.train.len());
            let (input, truths) = prepare_pass(&mut rng, ds, &aug, &picks, net_w, net_h)?;
            loss_sum += if let Some(g) = gpu.as_mut() {
                gpu_pass(g, &input, &truths)?
            } else {
                net.set_input(&input)?;
                net.zero_grads();
                net.forward()?;
                let loss = net.loss_and_deltas(&truths)?;
                net.backward();
                loss
            };
        }
        if let Some(g) = gpu.as_mut() {
            gpu_update(g, lr)?;
        } else {
            net.update(lr);
        }
        // The batch loss is a sum over images, so the step's loss is the sum over the passes.
        let loss = loss_sum;

        seen += (batch * subdiv) as u64;
        let (prev_loss, prev_ema) = (last_loss, ema);
        if step == 1 {
            ema = loss;
            first_loss = loss;
        } else {
            ema = ema * 0.9 + loss * 0.1;
        }
        last_loss = loss;
        let row = LossRow {
            iteration,
            seen,
            loss,
            avg_loss: ema,
            loss_per_image: loss / full as f32,
            learning_rate: lr,
            ms: t0.elapsed().as_secs_f32() * 1e3,
        };
        loss_csv.row(row)?;

        let should_save = opts.save_every > 0 && (iteration.is_multiple_of(opts.save_every) || step == opts.iters);
        // No mAP during burn-in (the rate is still ramping up): evaluate from `burn_in` on.
        let should_eval = opts.eval
            && opts.eval_every > 0
            && iteration >= cfg.train.burn_in
            && iteration.is_multiple_of(opts.eval_every)
            && !ds.valid.is_empty();
        if let Some(g) = gpu.as_ref().filter(|_| should_save || should_eval) {
            gpu_sync(g, &mut net)?;
        }
        if should_save {
            std::fs::create_dir_all(&opts.out_dir)
                .map_err(|e| Error::Io { path: opts.out_dir.display().to_string(), source: e })?;
            let p = opts.out_dir.join(format!("iter_{iteration:06}.weights"));
            checkpoint::save_weights(&p, &net, seen)?;
            checkpoint::save_state(opts.out_dir.join("last.state"), &net, seen)?;
            eprintln!("saved {} and last.state", p.display());
            final_weights = p;
        }
        if should_eval {
            let model = net.to_model()?;
            let (done, total, t_eval) = (std::cell::Cell::new(0usize), ds.valid.len(), Instant::now());
            let res = evaluate_map(&model, gpu.is_some(), classes, &ds.valid, |s| {
                done.set(done.get() + 1);
                eprint!("\r  mAP eval {}/{total} images ({:.0}s)", done.get(), t_eval.elapsed().as_secs_f32());
                ds.image(s)
            })?;
            eprintln!();
            map_50 = res.map_50;
            map_50_95 = res.map_50_95;
            best_map = best_map.max(res.map_50);
            map_csv.row_raw(&format!(
                "{iteration},{seen},0.5,0.005,{:.8},{:.8}",
                res.map_50, best_map
            ))?;
            map_csv.row_raw(&format!("{iteration},{seen},0.5:0.95,0.05,{:.8},{:.8}", res.map_50_95, best_map))?;
            eprintln!(
                "iter {iteration} loss {loss:.4} (avg {ema:.4}) lr {lr:.8} mAP@0.5 {:.4} mAP@0.5:0.95 {:.4}",
                res.map_50, res.map_50_95
            );
            eprint!("{}", counts_table(&res.counts, &ds.names));
            eval_model = Some(model);
        } else {
            // Every iteration: at ~20 s per step a 50-iteration cadence reads as a hang.
            let first = step == 1;
            eprintln!(
                "iter {iteration}/{} loss {} (avg {}) lr {lr:.8} {:.0} ms  [{} images]",
                base + opts.iters,
                trend(loss, prev_loss, first),
                trend(ema, prev_ema, first),
                row.ms,
                seen
            );
        }
    }
    loss_csv.flush()?;
    map_csv.flush()?;
    let _ = eval_model;
    Ok(TrainSummary {
        iterations: opts.iters,
        seen,
        first_loss,
        last_loss,
        avg_loss: ema.max(0.0),
        seconds: start.elapsed().as_secs_f64(),
        final_weights,
        map_50,
        map_50_95,
    })
}

/// A tiny append-only CSV writer, flushed per row so a long run always leaves usable numbers.
struct BufCsv {
    w: std::io::BufWriter<std::fs::File>,
    path: PathBuf,
}

impl BufCsv {
    fn create(path: &Path, header: &str) -> Result<BufCsv> {
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| Error::Io { path: dir.display().to_string(), source: e })?;
        }
        let f = std::fs::File::create(path).map_err(|e| Error::Io { path: path.display().to_string(), source: e })?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "{header}").map_err(|e| Error::Io { path: path.display().to_string(), source: e })?;
        Ok(BufCsv { w, path: path.to_path_buf() })
    }
    fn row(&mut self, r: LossRow) -> Result<()> {
        let line = format!(
            "{},{},{:.6},{:.6},{:.6},{:.8},{:.3}",
            r.iteration, r.seen, r.loss, r.avg_loss, r.loss_per_image, r.learning_rate, r.ms
        );
        self.row_raw(&line)
    }
    fn row_raw(&mut self, line: &str) -> Result<()> {
        writeln!(self.w, "{line}").map_err(|e| Error::Io { path: self.path.display().to_string(), source: e })?;
        self.w.flush().map_err(|e| Error::Io { path: self.path.display().to_string(), source: e })
    }
    fn flush(&mut self) -> Result<()> {
        self.w.flush().map_err(|e| Error::Io { path: self.path.display().to_string(), source: e })
    }
}

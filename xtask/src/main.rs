//! `cargo xtask <cmd>`: repo task runner.
//!
//! Every cargo/script invocation runs under a soft cgroup RAM cap
//! (`systemd-run --user --scope -p MemoryHigh=$MEM`, default 3G) and never
//! exceeds 4 build threads (`JOBS`, default 2, clamped to 4).

use std::env;
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

const USAGE: &str = "\
usage: cargo xtask <cmd> [args...]

  check|c [pkg...]      cargo check (default: darknet-cli darknet-core)
  gpu-check|gc          cargo check -p darknet-cli --features oxide
  build|b [pkg...]      cargo build (default: darknet-cli)
  clippy|l [pkg...]     cargo clippy (default: darknet-cli darknet-core)
  test|host|t [filter...]  host tests, same as tools/test_host.sh (PKGS overrides)
  oxide|o [args...]     ./tools/test_oxide.sh (GPU tests; args pass through)
  train_legogears|train|tr [-fp8|-fp16|-bf16|-fp8 -fp16|-fp8 -bf16] [w] [flags]
                        -fp8: DARKNET_OXIDE_GEMM=fp8 and DARKNET_OXIDE_GEMM_BWD=fp8 (f32 storage; an inherited
                        DARKNET_OXIDE_AMP is unset, as without any flag); -fp16: DARKNET_OXIDE_AMP=1
                        (-fp16 adds -dyn_scale and -loss_scale 1024 unless supplied)
                        -bf16: DARKNET_OXIDE_AMP=bf16, no loss scaling
                        -fp8 -fp16 / -fp8 -bf16: fp8 convolution GEMMs on 16-bit activation
                        storage (both env sets above; the -fp16 scaler rule applies)
                        train LegoGears_v2 on oxide (./tools/train_legogears_oxide.sh;
                        optional initial.weights, extra flags pass to `detector train`)
  fmt|f                 cargo fmt --all
  help                  this text

env: MEM (default 3G), JOBS (default 2, max 4), PKGS (test only)";

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "help".into());
    let rest: Vec<String> = args.collect();
    let mem = env::var("MEM").unwrap_or_else(|_| "3G".into());
    let jobs = env::var("JOBS")
        .ok()
        .and_then(|j| j.parse::<u32>().ok())
        .unwrap_or(2)
        .clamp(1, 4)
        .to_string();

    let pkgs = |defaults: &[&str]| -> Vec<String> {
        let names: Vec<&str> = if rest.is_empty() {
            defaults.to_vec()
        } else {
            rest.iter().map(String::as_str).collect()
        };
        names
            .iter()
            .flat_map(|p| ["-p".to_owned(), (*p).to_owned()])
            .collect()
    };

    let argv: Vec<String> = match cmd.as_str() {
        "check" | "c" => cargo(
            &["check", "-j", &jobs],
            pkgs(&["darknet-cli", "darknet-core"]),
        ),
        "gpu-check" | "gc" => cargo(
            &[
                "check",
                "-j",
                &jobs,
                "-p",
                "darknet-cli",
                "--features",
                "oxide",
            ],
            vec![],
        ),
        "build" | "b" => cargo(&["build", "-j", &jobs], pkgs(&["darknet-cli"])),
        "clippy" | "l" => cargo(
            &["clippy", "-j", &jobs],
            pkgs(&["darknet-cli", "darknet-core"]),
        ),
        "test" | "host" | "t" => {
            let packages = env::var("PKGS").unwrap_or_else(|_| "darknet-core darknet-cli".into());
            let flags = packages
                .split_whitespace()
                .flat_map(|p| ["-p".to_owned(), p.to_owned()])
                .collect();
            let mut v = cargo(&["test", "-j", &jobs], flags);
            v.push("--".into());
            v.extend(rest.clone());
            v
        }
        // test_oxide.sh does its own capping and `-j 2`; MEM reaches it via the env.
        "oxide" | "o" => {
            let mut v = vec!["./tools/test_oxide.sh".to_owned()];
            v.extend(rest.clone());
            return run(v, &mem, false);
        }
        // Release build + `detector train` on LegoGears_v2 (oxide backend); the script
        // caps itself. Overrides: CFG, DATA, SEED, PROFILE. DARKNET_OXIDE_AMP is set or unset from the precision flags.
        "train_legogears" | "train" | "tr" => {
            // Same 4 build thread cap as every other command.
            if env::var_os("JOBS").is_none() {
                // SAFETY: single-threaded at this point; no other thread reads the env.
                unsafe { env::set_var("JOBS", "4") };
            }
            let mut v = vec!["./tools/train_legogears_oxide.sh".to_owned()];
            match precision_env(&rest) {
                Ok((rest, vars, extra)) => {
                    apply_precision_env(&vars, |k, val| {
                        // SAFETY: single-threaded at this point; no other thread reads the env.
                        unsafe {
                            match val {
                                Some(val) => env::set_var(k, val),
                                None => env::remove_var(k),
                            }
                        }
                    });
                    v.extend(rest);
                    v.extend(extra);
                }
                Err(msg) => {
                    eprintln!("xtask: {msg}");
                    return ExitCode::from(2);
                }
            }
            return run(v, &mem, false);
        }
        "fmt" | "f" => vec!["cargo".into(), "fmt".into(), "--all".into()],
        "help" | "-h" | "--help" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        other => {
            eprintln!("unknown command `{other}`\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    run(argv, &mem, !matches!(cmd.as_str(), "fmt" | "f"))
}

/// `cargo <base...> <pkg flags...>`.
fn cargo(base: &[&str], tail: Vec<String>) -> Vec<String> {
    let mut v = vec!["cargo".to_owned()];
    v.extend(base.iter().map(|s| (*s).to_owned()));
    v.extend(tail);
    v
}

/// Runs `argv` from the repo root, optionally inside a soft-capped systemd scope.
fn run(argv: Vec<String>, mem: &str, cap: bool) -> ExitCode {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut full = argv;
    if cap {
        if scope_available(mem) {
            let mut capped = vec![
                "systemd-run".to_owned(),
                "--user".into(),
                "--scope".into(),
                "-p".into(),
                format!("MemoryHigh={mem}"),
            ];
            capped.append(&mut full);
            full = capped;
        } else {
            eprintln!("xtask: systemd-run --user unavailable, running UNCAPPED");
        }
    }
    eprintln!("xtask: {}", full.join(" "));
    let mut cmd = Command::new(&full[0]);
    cmd.args(&full[1..]).env("MEM", mem);
    if let Some(arch) = detected_gpu_arch() {
        cmd.env("CUDA_OXIDE_DEVICE_ARCH", arch);
    }
    match cmd.current_dir(root).status() {
        Ok(s) if s.success() => ExitCode::SUCCESS,
        Ok(s) => ExitCode::from(s.code().unwrap_or(1).clamp(1, 255) as u8),
        Err(e) => {
            eprintln!("xtask: failed to spawn `{}`: {e}", full[0]);
            ExitCode::FAILURE
        }
    }
}

fn scope_available(mem: &str) -> bool {
    Command::new("systemd-run")
        .args([
            "--user",
            "--scope",
            "-q",
            "-p",
            &format!("MemoryHigh={mem}"),
            "true",
        ])
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Local GPU arch (`sm_89`) from `nvidia-smi`, as the advisory `CUDA_OXIDE_DEVICE_ARCH`
/// hint: the backend uses it only when the GPU can run the kernel. `None` when the
/// user already set `CUDA_OXIDE_TARGET` or `CUDA_OXIDE_DEVICE_ARCH`, or there is no GPU.
fn detected_gpu_arch() -> Option<String> {
    if env::var_os("CUDA_OXIDE_TARGET").is_some() || env::var_os("CUDA_OXIDE_DEVICE_ARCH").is_some()
    {
        return None;
    }
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    let cap: String = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .chars()
        .filter(char::is_ascii_digit)
        .collect();
    (!cap.is_empty()).then(|| format!("sm_{cap}"))
}

/// Pulls the precision flags (`-fp8`, `-fp16`, `-bf16`, also with `--`) out of the
/// `train_legogears` args and maps them to the oxide backend's env switches.
/// One flag, or `-fp8` together with one 16-bit flag (`-fp8 -fp16`, `-fp8 -bf16`:
/// fp8 convolutions on 16-bit activation storage); anything else is an error.
/// Everything else passes through to the training script.
type PrecisionEnv = Result<(Vec<String>, Vec<(&'static str, &'static str)>, Vec<String>), String>;

fn precision_env(args: &[String]) -> PrecisionEnv {
    let mut rest = Vec::new();
    let mut fp8 = false;
    let mut half: Option<&str> = None;
    for a in args {
        match a.as_str() {
            "-fp8" | "--fp8" => {
                if fp8 {
                    return Err("-fp8 given twice".into());
                }
                fp8 = true;
            }
            flag @ ("-fp16" | "--fp16" | "-bf16" | "--bf16") => {
                let name = if flag.ends_with("fp16") { "fp16" } else { "bf16" };
                if let Some(prev) = half {
                    return Err(format!(
                        "one 16-bit precision flag only (got {prev} and {name})"
                    ));
                }
                half = Some(name);
            }
            _ => rest.push(a.clone()),
        }
    }
    let mut vars = Vec::new();
    let mut extra = Vec::new();
    if fp8 {
        // e4m3 forward, e5m2 for the incoming gradient (the C++ recipe).  Alone this
        // keeps f32 storage; with a 16-bit flag the conv GEMMs run fp8 on 16-bit
        // activations (`mixed.rs` keeps the fp8 GEMM mode under AMP).
        vars.push(("DARKNET_OXIDE_GEMM", "fp8"));
        vars.push(("DARKNET_OXIDE_GEMM_BWD", "fp8"));
    }
    match half {
        // AMP also selects the f16 tensor-core GEMM (unless fp8 was asked for).
        Some("fp16") => {
            vars.push(("DARKNET_OXIDE_AMP", "1"));
            if !args
                .iter()
                .any(|arg| arg == "-dyn_scale" || arg == "--dyn_scale")
            {
                extra.push("-dyn_scale".to_owned());
            }
            if !args
                .iter()
                .any(|arg| arg == "-loss_scale" || arg == "--loss_scale")
            {
                extra.extend(["-loss_scale".to_owned(), "1024".to_owned()]);
            }
        }
        // bf16 has the f32 exponent range, so no loss scaling flags.
        Some(_) => vars.push(("DARKNET_OXIDE_AMP", "bf16")),
        None => {}
    }
    Ok((rest, vars, extra))
}

/// Applies the precision env to the child: every var in `vars` is set, and
/// `DARKNET_OXIDE_AMP` is removed when no 16-bit flag asked for it, so a value
/// inherited from the caller's shell cannot turn bare `-fp8` (f32 storage) or
/// the fp32 default into AMP storage.
fn apply_precision_env(vars: &[(&'static str, &'static str)], mut set: impl FnMut(&str, Option<&str>)) {
    for (k, val) in vars {
        set(k, Some(val));
    }
    if !vars.iter().any(|(k, _)| *k == AMP_VAR) {
        set(AMP_VAR, None);
    }
}

const AMP_VAR: &str = "DARKNET_OXIDE_AMP";

#[cfg(test)]
mod tests {
    use super::{apply_precision_env, precision_env};
    use std::collections::HashMap;

    /// The child env after `-flags`, starting from an inherited `DARKNET_OXIDE_AMP=bf16`.
    fn child_env(flags: &[&str]) -> HashMap<String, String> {
        let (_, vars, _) = precision_env(&args(flags)).unwrap();
        let mut env = HashMap::from([("DARKNET_OXIDE_AMP".to_owned(), "bf16".to_owned())]);
        apply_precision_env(&vars, |k, v| match v {
            Some(v) => {
                env.insert(k.to_owned(), v.to_owned());
            }
            None => {
                env.remove(k);
            }
        });
        env
    }

    #[test]
    fn bare_fp8_and_default_drop_an_inherited_amp() {
        for flags in [&["-fp8"][..], &[][..]] {
            let env = child_env(flags);
            assert!(!env.contains_key("DARKNET_OXIDE_AMP"), "{flags:?}: {env:?}");
        }
        assert_eq!(child_env(&["-fp8"]).get("DARKNET_OXIDE_GEMM").map(String::as_str), Some("fp8"));
    }

    #[test]
    fn sixteen_bit_flags_override_an_inherited_amp() {
        assert_eq!(child_env(&["-fp16"])["DARKNET_OXIDE_AMP"], "1");
        assert_eq!(child_env(&["-bf16"])["DARKNET_OXIDE_AMP"], "bf16");
        assert_eq!(child_env(&["-fp8", "-fp16"])["DARKNET_OXIDE_AMP"], "1");
        assert_eq!(child_env(&["-fp8", "-bf16"])["DARKNET_OXIDE_AMP"], "bf16");
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn bf16_sets_amp_bf16_without_scaler_flags() {
        let (rest, vars, extra) = precision_env(&args(&["-bf16"])).unwrap();

        assert!(rest.is_empty());
        assert_eq!(vars, [("DARKNET_OXIDE_AMP", "bf16")]);
        assert!(extra.is_empty());
    }

    #[test]
    fn two_precision_flags_are_rejected() {
        assert!(precision_env(&args(&["-fp16", "-bf16"])).is_err());
    }

    #[test]
    fn fp16_adds_dynamic_loss_scale_flags() {
        let (rest, vars, extra) = precision_env(&args(&["-fp16"])).unwrap();

        assert!(rest.is_empty());
        assert_eq!(vars, [("DARKNET_OXIDE_AMP", "1")]);
        assert_eq!(extra, ["-dyn_scale", "-loss_scale", "1024"]);
    }

    #[test]
    fn fp16_respects_user_loss_scale() {
        let (rest, _, extra) =
            precision_env(&args(&["-fp16", "-loss_scale", "256"])).unwrap();

        assert_eq!(rest, ["-loss_scale", "256"]);
        assert_eq!(extra, ["-dyn_scale"]);
    }

    #[test]
    fn fp16_respects_user_dyn_scale() {
        let (rest, _, extra) = precision_env(&args(&["-fp16", "-dyn_scale"])).unwrap();

        assert_eq!(rest, ["-dyn_scale"]);
        assert_eq!(extra, ["-loss_scale", "1024"]);
    }

    #[test]
    fn fp8_adds_no_scaler_flags() {
        let (rest, vars, extra) = precision_env(&args(&["-fp8"])).unwrap();

        assert!(rest.is_empty());
        // e4m3 forward plus the e5m2 gradient recipe.
        assert_eq!(
            vars,
            [("DARKNET_OXIDE_GEMM", "fp8"), ("DARKNET_OXIDE_GEMM_BWD", "fp8")]
        );
        assert!(extra.is_empty());
    }

    #[test]
    fn fp8_with_fp16_sets_amp_and_fp8_gemm_and_scaler_flags() {
        for order in [["-fp8", "-fp16"], ["-fp16", "-fp8"]] {
            let (rest, vars, extra) = precision_env(&args(&order)).unwrap();

            assert!(rest.is_empty());
            assert!(vars.contains(&("DARKNET_OXIDE_GEMM", "fp8")), "{order:?}");
            assert!(vars.contains(&("DARKNET_OXIDE_GEMM_BWD", "fp8")), "{order:?}");
            assert!(vars.contains(&("DARKNET_OXIDE_AMP", "1")), "{order:?}");
            assert_eq!(vars.len(), 3, "{order:?}");
            assert_eq!(extra, ["-dyn_scale", "-loss_scale", "1024"], "{order:?}");
        }
    }

    #[test]
    fn fp8_with_fp16_respects_user_loss_scale() {
        let (rest, _, extra) =
            precision_env(&args(&["-fp8", "-fp16", "-loss_scale", "256"])).unwrap();

        assert_eq!(rest, ["-loss_scale", "256"]);
        assert_eq!(extra, ["-dyn_scale"]);
    }

    #[test]
    fn fp8_with_bf16_sets_amp_bf16_and_fp8_gemm_without_scaler_flags() {
        for order in [["-fp8", "-bf16"], ["--bf16", "--fp8"]] {
            let (rest, vars, extra) = precision_env(&args(&order)).unwrap();

            assert!(rest.is_empty());
            assert!(vars.contains(&("DARKNET_OXIDE_GEMM", "fp8")), "{order:?}");
            assert!(vars.contains(&("DARKNET_OXIDE_GEMM_BWD", "fp8")), "{order:?}");
            assert!(vars.contains(&("DARKNET_OXIDE_AMP", "bf16")), "{order:?}");
            assert_eq!(vars.len(), 3, "{order:?}");
            assert!(extra.is_empty(), "{order:?}");
        }
    }

    #[test]
    fn plain_fp8_keeps_f32_storage() {
        let (_, vars, _) = precision_env(&args(&["-fp8"])).unwrap();

        assert!(vars.iter().all(|(k, _)| *k != "DARKNET_OXIDE_AMP"));
    }

    #[test]
    fn repeated_or_conflicting_16_bit_flags_are_rejected() {
        assert!(precision_env(&args(&["-fp8", "-fp16", "-bf16"])).is_err());
        assert!(precision_env(&args(&["-fp8", "-fp8"])).is_err());
        assert!(precision_env(&args(&["-fp16", "-fp16"])).is_err());
        assert!(precision_env(&args(&["-bf16", "-fp16", "-fp8"])).is_err());
    }
}

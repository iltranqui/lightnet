//! Build-script helper: compile a kernels crate for `amdgcn-amd-amdhsa`.
//!
//! ```ignore
//! // build.rs
//! let elf = hip_tile_build::build_kernels(Path::new("../../kernels"), "gfx1031");
//! println!("cargo:rustc-env=HIP_TILE_KERNELS_ELF={}", elf.display());
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const TARGET: &str = "amdgcn-amd-amdhsa";

/// Env vars that would leak host build settings into the nested device build.
const SCRUBBED_ENV: &[&str] = &[
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_RUSTFLAGS",
    "CARGO_BUILD_TARGET",
    "CARGO_BUILD_TARGET_DIR",
    "CARGO_TARGET_DIR",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER",
    "CARGO_BUILD_RUSTDOCFLAGS",
    "CARGO_ENCODED_RUSTDOCFLAGS",
    "RUSTDOCFLAGS",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_INCREMENTAL",
    "CARGO_PRIMARY_PACKAGE",
    "CARGO_MAKEFLAGS",
];

/// Prefixes of env vars scrubbed likewise (per-target flags, profile overrides, cfg/feature
/// info exported to the outer build script).
const SCRUBBED_PREFIXES: &[&str] = &["CARGO_PROFILE_", "CARGO_CFG_", "CARGO_FEATURE_"];

/// Emit an rpath to the ROCm libraries (`ROCM_PATH`, default `/opt/rocm`) for the package
/// whose build script calls this. Link args from `hip-sys` do not propagate to dependents,
/// so every binary or test package needs this to run without `LD_LIBRARY_PATH`.
pub fn link_rocm() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    let root = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{root}/lib");
}

/// Build `crate_dir` for the GPU and return the path of the resulting `.elf`.
///
/// `arch` is the default GPU architecture; env `HIP_TILE_ARCH` overrides it.
/// Panics with the child's stderr if the build fails.
pub fn build_kernels(crate_dir: &Path, arch: &str) -> PathBuf {
    build_kernels_watching(crate_dir, arch, &[])
}

/// Like [`build_kernels`], with extra paths (files or directories) that trigger a rebuild.
pub fn build_kernels_watching(crate_dir: &Path, arch: &str, extra_watch: &[PathBuf]) -> PathBuf {
    println!("cargo:rerun-if-env-changed=HIP_TILE_ARCH");
    let arch = env::var("HIP_TILE_ARCH").unwrap_or_else(|_| arch.to_owned());

    let crate_dir = crate_dir.canonicalize().unwrap_or_else(|e| {
        panic!("hip-tile-build: kernels crate dir {} not found: {e}", crate_dir.display())
    });
    let manifest = crate_dir.join("Cargo.toml");
    let crate_name = package_name(&manifest);

    // Kernel sources, its manifest, and the device-side library next to it.
    println!("cargo:rerun-if-changed={}", crate_dir.join("src").display());
    println!("cargo:rerun-if-changed={manifest}", manifest = manifest.display());
    if let Some(hip_tile_src) = crate_dir.parent().map(|p| p.join("crates/hip-tile/src"))
        && hip_tile_src.exists() {
            println!("cargo:rerun-if-changed={}", hip_tile_src.display());
        }
    for path in extra_watch {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let target_dir = env::var_os("OUT_DIR")
        .map(|out| PathBuf::from(out).join("target-device"))
        .unwrap_or_else(|| crate_dir.join("target-device"));

    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(&crate_dir)
        .args(["rustc", "--release", "--lib", "--crate-type", "cdylib"])
        .args(["-j", "4", "-Zbuild-std=core", "--target", TARGET])
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target_dir);
    for name in SCRUBBED_ENV {
        cmd.env_remove(name);
    }
    for (name, _) in env::vars_os() {
        let name = name.to_string_lossy();
        if SCRUBBED_PREFIXES.iter().any(|p| name.starts_with(p)) {
            cmd.env_remove(&*name);
        }
    }
    // Set after scrubbing: RUSTFLAGS is both scrubbed and needed.
    cmd.env("RUSTFLAGS", format!("-Ctarget-cpu={arch} -Cpanic=abort"));

    let output = cmd
        .output()
        .unwrap_or_else(|e| panic!("hip-tile-build: failed to spawn cargo: {e}"));
    if !output.status.success() {
        panic!(
            "hip-tile-build: device build of `{crate_name}` for {arch} failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let elf = target_dir.join(TARGET).join("release").join(format!("{}.elf", crate_name.replace('-', "_")));
    if !elf.exists() {
        panic!("hip-tile-build: build succeeded but {} does not exist", elf.display());
    }
    elf
}

/// `name = "..."` of the `[package]` table; enough for a plain manifest.
fn package_name(manifest: &Path) -> String {
    let text = fs::read_to_string(manifest)
        .unwrap_or_else(|e| panic!("hip-tile-build: cannot read {}: {e}", manifest.display()));
    let mut in_package = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
        } else if in_package
            && let Some(rest) = line.strip_prefix("name")
                && let Some(value) = rest.trim_start().strip_prefix('=') {
                    return value.trim().trim_matches('"').to_owned();
                }
    }
    panic!("hip-tile-build: no [package] name in {}", manifest.display());
}

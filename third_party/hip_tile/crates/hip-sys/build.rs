use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    let root = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let lib = format!("{root}/lib");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-lib=amdhip64");
    if env::var_os("CARGO_FEATURE_ROCBLAS").is_some() {
        println!("cargo:rustc-link-lib=rocblas");
    }
}

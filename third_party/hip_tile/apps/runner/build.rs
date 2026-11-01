use std::path::Path;

fn main() {
    hip_tile_build::link_rocm();
    let elf = hip_tile_build::build_kernels(Path::new("../../kernels"), "gfx1031");
    println!("cargo:rustc-env=HIP_TILE_KERNELS_ELF={}", elf.display());
}

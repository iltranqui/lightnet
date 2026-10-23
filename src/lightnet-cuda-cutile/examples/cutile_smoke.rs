//! Minimal cuTile smoke test: upload two vectors, run a device add, download,
//! and print the result.  CUDA execution is intentionally separate from the
//! CPU/C++ differential tests because a device is optional.

use lightnet_cuda_cutile::{CutileBackend, DEFAULT_TILE_1D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = CutileBackend::new(0)?;
    backend.hello_world((1, 1, 1))?;

    if std::env::var_os("CUTILE_SMOKE_HELLO_ONLY").is_some() {
        return Ok(());
    }

    if std::env::var_os("CUTILE_SMOKE_SGD_ONLY").is_some() {
        let weights = backend.upload_f32(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0])?;
        let updates = backend.upload_f32(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0])?;
        let (weights, updates) =
            backend.sgd_update_f32(&weights, &updates, 0.1, 0.9, 0.0, 1, DEFAULT_TILE_1D)?;
        println!("weights={:?}", backend.download_f32(&weights)?);
        println!("updates={:?}", backend.download_f32(&updates)?);
        return Ok(());
    }

    if std::env::var_os("CUTILE_SMOKE_CROP_ONLY").is_some() {
        let input = backend.upload_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0])?;
        let source_x = backend.upload_f32(&[1], &[0.5])?;
        let source_y = backend.upload_f32(&[1], &[0.5])?;
        let scales = backend.upload_f32(&[1], &[1.0])?;
        let offsets = backend.upload_f32(&[1], &[0.0])?;
        let output = backend.crop_bilinear_f32(
            &input,
            &[1, 1, 1, 1],
            1,
            2,
            2,
            &source_x,
            &source_y,
            &scales,
            &offsets,
            DEFAULT_TILE_1D,
        )?;
        let output_grad = backend.upload_f32(&[1, 1, 1, 1], &[1.0])?;
        let input_grad = backend.crop_bilinear_backward_f32(
            &output_grad,
            &[1, 1, 2, 2],
            1,
            2,
            2,
            &source_x,
            &source_y,
            &scales,
            DEFAULT_TILE_1D,
        )?;
        println!("crop={:?}", backend.download_f32(&output)?);
        println!("crop_grad={:?}", backend.download_f32(&input_grad)?);
        return Ok(());
    }

    let x = backend.upload_f32(&[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    let y = backend.upload_f32(&[8], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])?;
    let z = backend.vector_add_f32(&x, &y, DEFAULT_TILE_1D)?;
    let host = backend.download_f32(&z)?;
    println!("{host:?}");
    Ok(())
}

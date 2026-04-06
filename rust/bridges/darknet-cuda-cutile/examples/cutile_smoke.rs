use darknet_cuda_cutile::{CutileBackend, DEFAULT_TILE_1D};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = CutileBackend::new(0)?;
    backend.hello_world((1, 1, 1))?;

    let x = backend.upload_f32(&[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;
    let y = backend.upload_f32(&[8], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])?;
    let z = backend.vector_add_f32(&x, &y, DEFAULT_TILE_1D)?;
    let host = backend.download_f32(&z)?;
    println!("{host:?}");
    Ok(())
}

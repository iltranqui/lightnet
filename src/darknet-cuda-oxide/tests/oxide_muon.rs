//! Native GPU Newton–Schulz orthogonalization parity for Muon.

use darknet_cuda::{F32TensorOpsBackend, host_ops};
use darknet_cuda_oxide::OxideBackend;

fn close(name: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance * w.abs().max(1.0),
            "{name}[{i}]: got {g}, want {w}, tolerance {tolerance}"
        );
    }
}

#[test]
fn oxide_muon_orthogonalization_matches_host_reference_across_aspect_ratios() {
    let backend = OxideBackend::new().expect("backend");
    for &(rows, cols) in &[
        (1usize, 1usize),
        (2, 3),
        (3, 2),
        (4, 8),
        (8, 4),
        (5, 5),
        (17, 31),
        (31, 17),
    ] {
        let input = (0..rows * cols)
            .map(|i| ((i * 17 + 3) as f32).sin() * 0.7)
            .collect::<Vec<_>>();
        let tensor = backend.upload_f32(&[rows, cols], &input).unwrap();
        let actual = F32TensorOpsBackend::muon_orthogonalize_f32(
            &backend,
            &tensor,
            &[rows, cols],
            rows,
            cols,
        )
        .unwrap();
        let actual = backend.download_f32(&actual).unwrap();
        let expected = host_ops::muon::orthogonalize(&input, rows, cols).unwrap();
        close(&format!("Muon {rows}x{cols}"), &actual, &expected, 3e-4);
    }
}

#[test]
fn oxide_muon_momentum_and_final_update_match_host_references() {
    let backend = OxideBackend::new().expect("backend");
    let shape = [2, 3];
    let updates = [0.4, -0.2, 0.15, -0.5, 0.3, 0.1];
    let old_momentum = [0.1, 0.3, -0.25, 0.2, 0.05, -0.1];
    let updates_gpu = backend.upload_f32(&shape, &updates).unwrap();
    let momentum_gpu = backend.upload_f32(&shape, &old_momentum).unwrap();
    let (next_gpu, lookahead_gpu) = backend
        .muon_momentum_lookahead_f32(&updates_gpu, &momentum_gpu, &shape, 2, 4.0, 0.9)
        .unwrap();
    let (next_expected, lookahead_expected) =
        host_ops::muon::momentum_lookahead(&updates, &old_momentum, 2, 4.0, 0.9).unwrap();
    close(
        "Muon next momentum",
        &backend.download_f32(&next_gpu).unwrap(),
        &next_expected,
        2e-6,
    );
    close(
        "Muon lookahead",
        &backend.download_f32(&lookahead_gpu).unwrap(),
        &lookahead_expected,
        2e-6,
    );

    let direction = F32TensorOpsBackend::muon_orthogonalize_f32(
        &backend,
        &lookahead_gpu,
        &shape,
        2,
        3,
    )
    .unwrap();
    let weights = [0.2, -0.1, 0.3, 0.4, -0.25, 0.15];
    let weights_gpu = backend.upload_f32(&shape, &weights).unwrap();
    let updated_gpu = backend
        .muon_apply_update_f32(&weights_gpu, &direction, &shape, 0.02, 0.0005, 0.2 * 3.0f32.sqrt())
        .unwrap();
    let direction_host = host_ops::muon::orthogonalize(&lookahead_expected, 2, 3).unwrap();
    let updated_expected = host_ops::muon::apply_update(
        &weights,
        &direction_host,
        0.02,
        0.0005,
        0.2 * 3.0f32.sqrt(),
    )
    .unwrap();
    close(
        "Muon device-applied weights",
        &backend.download_f32(&updated_gpu).unwrap(),
        &updated_expected,
        3e-4,
    );
}

#[test]
fn oxide_muon_orthogonalization_preserves_zero_and_rejects_bad_shapes() {
    let backend = OxideBackend::new().expect("backend");
    let zero = backend.zeros_f32(&[2, 3]).unwrap();
    let output =
        F32TensorOpsBackend::muon_orthogonalize_f32(&backend, &zero, &[2, 3], 2, 3).unwrap();
    assert_eq!(backend.download_f32(&output).unwrap(), vec![0.0; 6]);

    assert!(F32TensorOpsBackend::muon_orthogonalize_f32(&backend, &zero, &[2, 2], 2, 2,).is_err());
    assert!(F32TensorOpsBackend::muon_orthogonalize_f32(&backend, &zero, &[0, 3], 0, 3,).is_err());
}

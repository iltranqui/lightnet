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
fn oxide_muon_orthogonalization_preserves_zero_and_rejects_bad_shapes() {
    let backend = OxideBackend::new().expect("backend");
    let zero = backend.zeros_f32(&[2, 3]).unwrap();
    let output =
        F32TensorOpsBackend::muon_orthogonalize_f32(&backend, &zero, &[2, 3], 2, 3).unwrap();
    assert_eq!(backend.download_f32(&output).unwrap(), vec![0.0; 6]);

    assert!(F32TensorOpsBackend::muon_orthogonalize_f32(&backend, &zero, &[2, 2], 2, 2,).is_err());
    assert!(F32TensorOpsBackend::muon_orthogonalize_f32(&backend, &zero, &[0, 3], 0, 3,).is_err());
}

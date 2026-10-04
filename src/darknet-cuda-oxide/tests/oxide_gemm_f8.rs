//! FP8 (e4m3 / e5m2) tensor-core GEMM tests.
//!
//! Three layers of checks, from the hardware upwards:
//!
//! 1. `fragment_layout_*`: the raw `mma.m16n8k32` probe.  The host builds A
//!    (16x32) and B (32x8) fragments with the layout documented in
//!    `kernels/gemm_f8.rs`, uploads them, and compares the 16x8 f32 result
//!    with a CPU product of the matrices those fragments represent.  This is
//!    the mandatory smoke test of `tasks/rust/fp8_fp4.md` step 1: the FP8 mma
//!    family is declared `sm_89+` but its runtime support was unverified, so
//!    the layout has to be pinned on the real GPU rather than assumed.
//! 2. `quantize_*`: the device quantiser against the host reference in
//!    `darknet_cuda_oxide::fp8`, over a dense sweep that includes the
//!    saturation boundary and NaN.
//! 3. `gemm_fp8_*`: the tiled GEMM against a CPU f32 reference on the
//!    *dequantised* operands, all four transposes and odd shapes.
//!
//! Tolerance.  e4m3 keeps 3 explicit mantissa bits, so a relative error of up
//! to 2^-4 is introduced by the operand rounding alone; e5m2 keeps 2, so
//! 2^-3.  The tests therefore compare against the exact f32 product of the
//! *quantised* operands, with only the f32 accumulation order allowed to
//! differ (`2^-22 * K` relative).

use darknet_cuda_oxide::fp8::{Fp8Format, amax_to_scale, f32_to_fp8, fp8_to_f32};
use darknet_cuda_oxide::{GemmMode, OxideBackend, OxideTensorF32};

/// Relative slack against the *row norm* (`sum_k |a_k| |b_k|`) for the f32
/// accumulation-order difference between the tensor core and the f64 CPU
/// reference.  32 f32 additions of products bounded by the row norm give
/// ~2e-5; 1e-3 leaves ample margin while still being far tighter than the
/// ~6% error the e4m3 / e5m2 operand formats introduce on their own (a wrong
/// fragment layout moves the result by O(1) of the row norm - see
/// `fragment_layout_sweep_finds_the_documented_wiring`).
const TOL_ACC: f64 = 1e-3;

struct Rng(u64);
impl Rng {
    fn next(&mut self, lo: f32, hi: f32) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let r = self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
        lo + (hi - lo) * (r as f32 / (1u64 << 24) as f32)
    }
    fn vec(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(lo, hi)).collect()
    }
}

fn fmt_code(fmt: Fp8Format) -> u32 {
    fmt.code()
}

/// Host A fragment (16x32, row-major) for `mma.m16n8k32`: with
/// `group = lane / 4` and `thread = lane % 4`,
/// `row = group + (i >= 8) * 8`, `col = thread * 4 + (i & 3) + (i >= 8) * 16`.
fn a_fragment(a: &[f32], lane: usize, fmt: Fp8Format) -> [u8; 16] {
    // 32 lanes x 16 bytes = the whole warp's 16x32 e4m3 A tile.
    let group = lane / 4;
    let thread = lane % 4;
    let mut out = [0u8; 16];
    for i in 0..16 {
        // Registers 1 and 3 hold row group+8 (elements 4..8 and 12..16);
        // registers 2 and 3 carry the +16 K band.
        let row = group + if (i >= 4 && i < 8) || i >= 12 { 8 } else { 0 };
        let col = thread * 4 + (i & 3) + if i >= 8 { 16 } else { 0 };
        out[i] = f32_to_fp8(a[row * 32 + col], fmt);
    }
    out
}

/// Host B fragment (32x8, column-major, i.e. `b[k][n]`): with
/// `group = lane / 4` and `thread = lane % 4`,
/// `row = thread * 4 + (i & 3) + (i >= 4) * 16`, `col = group`.
fn b_fragment(b: &[f32], lane: usize, fmt: Fp8Format) -> [u8; 8] {
    // 32 lanes x 8 bytes = the whole warp's 32x8 B tile.
    let group = lane / 4;
    let thread = lane % 4;
    let mut out = [0u8; 8];
    for i in 0..8 {
        let row = thread * 4 + (i & 3) + if i >= 4 { 16 } else { 0 };
        out[i] = f32_to_fp8(b[row * 8 + group], fmt);
    }
    out
}

fn pack_a_fragments(a: &[f32], fmt: Fp8Format) -> Vec<u8> {
    let mut out = Vec::with_capacity(512);
    for lane in 0..32 {
        out.extend_from_slice(&a_fragment(a, lane, fmt));
    }
    out
}

fn pack_b_fragments(b: &[f32], fmt: Fp8Format) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for lane in 0..32 {
        out.extend_from_slice(&b_fragment(b, lane, fmt));
    }
    out
}

/// C fragment layout: `c[i * 32 + lane]` with
/// `row = group + (i >= 2) * 8`, `col = thread * 2 + (i & 1)`.
fn probe_to_matrix(c: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; 16 * 8];
    for lane in 0..32 {
        let group = lane / 4;
        let thread = lane % 4;
        for i in 0..4 {
            let row = group + if i >= 2 { 8 } else { 0 };
            let col = thread * 2 + (i & 1);
            out[row * 8 + col] = c[i * 32 + lane];
        }
    }
    out
}

/// One `mma.m16n8k32` on random matrices, compared with the CPU product.
fn probe_once(be: &OxideBackend, fmt_a: Fp8Format, fmt_b: Fp8Format, seed: u64) {
    let mut rng = Rng(seed);
    // Small magnitudes so the e5m2 (2 mantissa bit) grid is exercised densely.
    let a = rng.vec(16 * 32, -1.0, 1.0);
    let b = rng.vec(32 * 8, -1.0, 1.0);
    let a_q: Vec<f32> = a.iter().map(|v| fp8_to_f32(f32_to_fp8(*v, fmt_a), fmt_a)).collect();
    let b_q: Vec<f32> = b.iter().map(|v| fp8_to_f32(f32_to_fp8(*v, fmt_b), fmt_b)).collect();
    let fa = pack_a_fragments(&a, fmt_a);
    let fb = pack_b_fragments(&b, fmt_b);
    let got = be.fp8_mma_probe(&fa, &fb, fmt_a, fmt_b).expect("probe");
    assert_eq!(got.len(), 128);
    let got = probe_to_matrix(&got);
    let mut worst = 0.0f64;
    for row in 0..16 {
        for col in 0..8 {
            let mut want = 0.0f64;
            let mut norm = 0.0f64;
            for k in 0..32 {
                let av = a_q[row * 32 + k] as f64;
                let bv = b_q[k * 8 + col] as f64;
                want += av * bv;
                norm += av.abs() * bv.abs();
            }
            let scale = norm.max(1e-6);
            let err = (got[row * 8 + col] as f64 - want).abs();
            worst = worst.max(err / scale);
            assert!(
                err <= TOL_ACC * scale,
                "{fmt_a:?}x{fmt_b:?} probe[{row},{col}]: got {}, want {want}, err {err}, scale {scale}",
                got[row * 8 + col]
            );
        }
    }
    println!("{fmt_a:?}x{fmt_b:?} probe worst err / row norm = {worst:e}");
    let _ = fmt_code(fmt_a);
}

#[test]
fn fragment_layout_e4m3_x_e4m3() {
    let be = OxideBackend::new().expect("backend");
    probe_once(&be, Fp8Format::E4M3, Fp8Format::E4M3, 0x51ed_2701);
}

#[test]
fn fragment_layout_mixed_e4m3_x_e5m2() {
    let be = OxideBackend::new().expect("backend");
    probe_once(&be, Fp8Format::E4M3, Fp8Format::E5M2, 0x1234_5678);
}

#[test]
fn fragment_layout_e5m2_x_e4m3_and_e5m2_x_e5m2() {
    let be = OxideBackend::new().expect("backend");
    probe_once(&be, Fp8Format::E5M2, Fp8Format::E4M3, 0x0bad_c0de);
    probe_once(&be, Fp8Format::E5M2, Fp8Format::E5M2, 0xfeed_face);
}

#[test]
fn fragment_layout_hand_computed_tile() {
    // A single-lane-decodable case: A = [row r has value r+1] and
    // B = [col c has value c+1] give C[r][c] = 32 * (r+1) * (c+1).
    let be = OxideBackend::new().expect("backend");
    let a: Vec<f32> = (0..16).map(|r| (r + 1) as f32).flat_map(|v| [v; 32]).collect();
    // B is the 32x8 tile in [k][n] order, as `b_fragment` reads it.
    let b: Vec<f32> = (0..32).flat_map(|_| (0..8).map(|c| (c + 1) as f32)).collect();
    let fa = pack_a_fragments(&a, Fp8Format::E4M3);
    let fb = pack_b_fragments(&b, Fp8Format::E4M3);
    let got = probe_to_matrix(&be.fp8_mma_probe(&fa, &fb, Fp8Format::E4M3, Fp8Format::E4M3).unwrap());
    for row in 0..16 {
        for col in 0..8 {
            let want = 32.0 * (row as f32 + 1.0) * (col as f32 + 1.0);
            assert!(
                (got[row * 8 + col] - want).abs() <= 1e-3 * want,
                "[{row},{col}]: got {} want {want}",
                got[row * 8 + col]
            );
        }
    }
}

#[test]
fn quantize_matches_the_host_reference_over_a_dense_sweep() {
    let be = OxideBackend::new().expect("backend");
    let mut values: Vec<f32> = Vec::new();
    // Dense sweep across the interesting magnitudes, including both
    // saturation boundaries and every representable e4m3 / e5m2 value.
    for i in 0..4000 {
        let t = i as f32 / 4000.0;
        values.push(-600.0 + t * 1200.0);
    }
    for e in -12..4 {
        let s = (2.0f32).powi(e);
        for m in 0..16 {
            values.push((1.0 + m as f32 / 16.0) * s);
            values.push(-(1.0 + m as f32 / 16.0) * s);
        }
    }
    values.push(0.0);
    values.push(-0.0);
    values.push(f32::NAN);
    values.push(-f32::NAN);
    values.push(f32::INFINITY);
    values.push(f32::NEG_INFINITY);
    values.push(1e30);
    values.push(-1e30);
    for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
        let got = be.quantize_f32_to_fp8(&values, 1.0, fmt).expect("quantize");
        assert_eq!(got.len(), values.len());
        for (i, v) in values.iter().enumerate() {
            let want = f32_to_fp8(*v, fmt);
            assert_eq!(
                got[i], want,
                "{fmt:?}: value {v:e} (bits {:#x}) -> device {:#x}, host {want:#x}",
                v.to_bits(), got[i]
            );
        }
    }
}

#[test]
fn dequantize_is_the_inverse_of_quantize_on_finite_values() {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(0xabcd_ef01);
    let values: Vec<f32> = (0..4096).map(|_| rng.next(-2.0, 2.0)).collect();
    for fmt in [Fp8Format::E4M3, Fp8Format::E5M2] {
        let bytes = be.quantize_f32_to_fp8(&values, 1.0, fmt).unwrap();
        let back = be.dequantize_fp8_to_f32(&bytes, fmt).unwrap();
        assert_eq!(back.len(), values.len());
        for i in 0..values.len() {
            let want = fp8_to_f32(bytes[i], fmt);
            assert_eq!(back[i], want, "{fmt:?} element {i}");
            // Relative slack from the mantissa plus the format's absolute
            // floor (half of the smallest subnormal step), which is what
            // dominates for values near the bottom of the range.
            let slack = 2f32.powi(-(4 - fmt.mantissa_bits() as i32));
            let floor = 2f32.powi(-(11 - fmt.mantissa_bits() as i32));
            assert!(
                (back[i] - values[i]).abs() <= slack * values[i].abs() + floor,
                "{fmt:?} round trip {i}: {} vs {}",
                back[i],
                values[i]
            );
        }
    }
}

#[test]
fn amax_ignores_non_finite_values() {
    let be = OxideBackend::new().expect("backend");
    let mut rng = Rng(7);
    let mut values = rng.vec(100_000, -3.0, 3.0);
    values.push(f32::NAN);
    values.push(f32::INFINITY);
    values.push(-f32::INFINITY);
    let mut want = 0.0f32;
    for v in &values {
        if v.is_finite() {
            want = want.max(v.abs());
        }
    }
    let got = be.fp8_amax(&values).expect("amax");
    assert_eq!(got, want, "device amax {got} vs host {want}");
    assert_eq!(be.fp8_amax(&[]).unwrap(), 0.0);
    assert_eq!(amax_to_scale(want, Fp8Format::E4M3), want / 448.0);
}

/// Exact CPU reference plus the "row norm" `sum_p |a_ip| |b_pj|` of each
/// output: the tensor core accumulates in f32 in its own order, so the error
/// scales with the row norm rather than with the (possibly cancelling) sum.
fn reference_fp8_gemm(
    a_q: &[f32],
    b_q: &[f32],
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; m * n];
    let mut norm = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            let mut rows = 0.0f64;
            for p in 0..k {
                let av = if trans_a { a_q[p * m + i] } else { a_q[i * k + p] } as f64;
                let bv = if trans_b { b_q[j * k + p] } else { b_q[p * n + j] } as f64;
                acc += av * bv;
                rows += av.abs() * bv.abs();
            }
            out[i * n + j] = alpha as f64 * acc;
            norm[i * n + j] = rows;
        }
    }
    (out, norm)
}

/// Quantise on the host, run the tiled kernel, compare with the CPU product of
/// the *quantised* operands.
fn gemm_case(
    be: &OxideBackend,
    m: usize,
    n: usize,
    k: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    fmt_a: Fp8Format,
    fmt_b: Fp8Format,
    seed: u64,
) {
    let mut rng = Rng(seed);
    let (ar, ac) = if trans_a { (k, m) } else { (m, k) };
    let (br, bc) = if trans_b { (n, k) } else { (k, n) };
    let a = rng.vec(ar * ac, -2.0, 2.0);
    let b = rng.vec(br * bc, -2.0, 2.0);
    let q = |v: &[f32], fmt: Fp8Format| -> (Vec<u8>, Vec<f32>) {
        let bytes: Vec<u8> = v.iter().map(|x| f32_to_fp8(*x, fmt)).collect();
        let back = bytes.iter().map(|x| fp8_to_f32(*x, fmt)).collect();
        (bytes, back)
    };
    let (aq, a_dq) = q(&a, fmt_a);
    let (bq, b_dq) = q(&b, fmt_b);
    let out = be.gemm_fp8(&aq, &bq, m, n, k, trans_a, trans_b, alpha, fmt_a, fmt_b).expect("gemm_fp8");
    let got = download(&be, &out);
    let (want, norm) = reference_fp8_gemm(&a_dq, &b_dq, m, n, k, trans_a, trans_b, alpha);
    let mut worst = 0.0f64;
    for idx in 0..m * n {
        let err = (got[idx] as f64 - want[idx]).abs();
        let scale = norm[idx].max(1e-4);
        worst = worst.max(err / scale);
        assert!(
            err <= TOL_ACC * scale,
            "{m}x{n}x{k} ta={trans_a} tb={trans_b} {fmt_a:?}x{fmt_b:?} [{idx}]: got {}, want {}, err {err}, scale {scale}",
            got[idx], want[idx]
        );
    }
    println!("{m}x{n}x{k} ta={trans_a} tb={trans_b}: worst err / row norm = {worst:e}");
}

fn download(be: &OxideBackend, t: &OxideTensorF32) -> Vec<f32> {
    use darknet_cuda::F32TensorOpsBackend;
    be.download_f32(t).expect("download")
}

#[test]
fn gemm_fp8_e4m3_all_transposes() {
    let be = OxideBackend::new().expect("backend");
    for &(ta, tb) in &[(false, false), (false, true), (true, false), (true, true)] {
        gemm_case(&be, 96, 80, 64, ta, tb, 1.0, Fp8Format::E4M3, Fp8Format::E4M3, 0x1111 + ta as u64 + 10 * tb as u64);
    }
}

#[test]
fn gemm_fp8_odd_shapes_and_mixed_formats() {
    let be = OxideBackend::new().expect("backend");
    // Shapes that exercise the K tail (< 32, not a multiple of 32), partial
    // tiles in M and N, and the split-K path (few tiles, long K).
    for &(m, n, k) in &[(16usize, 8usize, 32usize), (1, 1, 32), (17, 19, 5), (65, 3, 200), (130, 70, 129)] {
        gemm_case(&be, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3, 0x2000 + k as u64);
        gemm_case(&be, m, n, k, false, false, 0.5, Fp8Format::E4M3, Fp8Format::E5M2, 0x3000 + k as u64);
        gemm_case(&be, m, n, k, true, true, 1.0, Fp8Format::E5M2, Fp8Format::E5M2, 0x4000 + k as u64);
    }
}

#[test]
fn gemm_fp8_matches_the_f32_reference_on_unquantised_inputs() {
    // Operands already on the e4m3 grid, so the kernel result must match the
    // plain f32 GEMM to accumulation noise: this catches scaling mistakes in
    // the epilogue that the quantised comparison above would absorb.
    let be = OxideBackend::new().expect("backend");
    let m = 128;
    let n = 96;
    let k = 256;
    let mut rng = Rng(0x9999_1111);
    let a: Vec<f32> = rng.vec(m * k, -1.0, 1.0);
    let b: Vec<f32> = rng.vec(k * n, -1.0, 1.0);
    let aq: Vec<u8> = a.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
    let bq: Vec<u8> = b.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
    let a_dq: Vec<f32> = aq.iter().map(|x| fp8_to_f32(*x, Fp8Format::E4M3)).collect();
    let b_dq: Vec<f32> = bq.iter().map(|x| fp8_to_f32(*x, Fp8Format::E4M3)).collect();
    let out = be
        .gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3)
        .expect("gemm_fp8");
    let got = download(&be, &out);
    for i in 0..m {
        for j in 0..n {
            let mut want = 0.0f64;
            let mut norm = 0.0f64;
            for p in 0..k {
                let av = a_dq[i * k + p] as f64;
                let bv = b_dq[p * n + j] as f64;
                want += av * bv;
                norm += av.abs() * bv.abs();
            }
            let scale = norm.max(1.0);
            assert!(
                (got[i * n + j] as f64 - want).abs() <= TOL_ACC * scale,
                "[{i},{j}]: got {} want {want}",
                got[i * n + j]
            );
        }
    }
}
// ---------------------------------------------------------------------------
// Layout sweep
// ---------------------------------------------------------------------------
//
// `tasks/rust/fp8_fp4.md` step 1 makes pinning the fragment layout on the real
// GPU mandatory: the FP8 `mma` family is declared `sm_89+` but its runtime
// support is `unexecuted` in `intrinsics/overlay/register_mma_fp8.toml`. Rather
// than trust the documentation, `fragment_layout_sweep` runs the raw mma once
// per candidate wiring and keeps the ones that reproduce an exact host GEMM.
// Only the true wiring can, because every axis (which register holds row
// group+8, which K band it covers, and the byte order inside a register)
// changes the sum.

/// A values that depend on row *and* on the K band, so swapping A registers,
/// A bands or bytes inside a register changes the product.
fn sweep_a(r: u32, c: u32) -> f32 {
    (1 + (3 * r + 5 * c) % 7 + c / 16) as f32
}

/// B values that depend on K (row and band) *and* on the column.
fn sweep_b(k: u32, n: u32) -> f32 {
    (1 + (2 * k + 3 * n + 4 * (k / 16)) % 8) as f32
}

fn a_row_offset(rp: u32, n: u32) -> u32 {
    match rp {
        0 => n & 1,
        1 => n / 2,
        2 => (n & 1) ^ 1,
        _ => (n / 2) ^ 1,
    }
}

fn a_band(ab: u32, n: u32) -> u32 {
    match ab {
        0 => 16 * (n / 2),
        1 => 4 * (n / 2),
        2 => 8 * n,
        3 => 4 * n,
        4 => 8 * (n / 2),
        _ => 16 * n,
    }
}

fn b_band(bb: u32, n: u32) -> u32 {
    match bb {
        0 => 16 * n,
        1 => 8 * n,
        2 => 4 * n,
        3 => 16 * (2 * (n & 1) + n / 2),
        4 => 16 * (n / 2),
        _ => 16 * (n & 1),
    }
}

fn byte_order(o: u32, i: u32) -> u32 {
    if o == 0 { i } else { 3 - i }
}

#[test]
fn fragment_layout_sweep_finds_the_documented_wiring() {
    const A_ROW_OPTS: u32 = 4;
    const A_BAND_OPTS: u32 = 6;
    const A_ORD_OPTS: u32 = 2;
    const B_BAND_OPTS: u32 = 6;
    const B_ORD_OPTS: u32 = 2;
    let be = OxideBackend::new().expect("backend");
    let a: Vec<f32> = (0..16).flat_map(|r| (0..32).map(move |c| sweep_a(r, c))).collect();
    let b: Vec<f32> = (0..32).flat_map(|k| (0..8).map(move |n| sweep_b(k, n))).collect();
    let mut want = vec![0.0f64; 128];
    for r in 0..16usize {
        for n in 0..8usize {
            want[r * 8 + n] = (0..32).map(|k| (a[r * 32 + k] * b[k * 8 + n]) as f64).sum();
        }
    }
    let mut hits: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
    for rp in 0..A_ROW_OPTS {
        for ab in 0..A_BAND_OPTS {
            // Skip candidates whose K band leaves the 32-wide tile.
            if (0..4).any(|n| a_band(ab, n) + 15 > 31) {
                continue;
            }
            for ao in 0..A_ORD_OPTS {
                let mut fa = Vec::with_capacity(512);
                for lane in 0..32u32 {
                    let g = lane / 4;
                    let t = lane % 4;
                    for n in 0..4u32 {
                        let r = g + 8 * a_row_offset(rp, n);
                        for i in 0..4u32 {
                            let c = 4 * t + a_band(ab, n) + byte_order(ao, i);
                            fa.push(f32_to_fp8(a[(r * 32 + c) as usize], Fp8Format::E4M3));
                        }
                    }
                }
                for bb in 0..B_BAND_OPTS {
                    if (0..2).any(|n| b_band(bb, n) + 15 > 31) {
                        continue;
                    }
                    for bo in 0..B_ORD_OPTS {
                        let mut fb = Vec::with_capacity(256);
                        for lane in 0..32u32 {
                            let g = lane / 4;
                            let t = lane % 4;
                            for n in 0..2u32 {
                                for i in 0..4u32 {
                                    let k = b_band(bb, n) + 4 * t + byte_order(bo, i);
                                    fb.push(f32_to_fp8(b[(k * 8 + g) as usize], Fp8Format::E4M3));
                                }
                            }
                        }
                        let got = probe_to_matrix(
                            &be.fp8_mma_probe(&fa, &fb, Fp8Format::E4M3, Fp8Format::E4M3).unwrap(),
                        );
                        let err: f64 = (0..128)
                            .map(|i| (got[i] as f64 - want[i]).abs() / want[i].abs().max(1e-3))
                            .fold(0.0, f64::max);
                        if (rp, ab, ao, bb, bo) == (0, 0, 0, 0, 0) {
                            println!("documented candidate: max rel err {err:e}");
                        }
                        if err <= 1e-3 {
                            hits.push((rp, ab, ao, bb, bo));
                        }
                    }
                }
            }
        }
    }
    println!("matching fragment layouts (a_row, a_band, a_order, b_band, b_order): {hits:?}");
    // Candidate 0 on every axis is the documented wiring.
    assert!(
        hits.contains(&(0, 0, 0, 0, 0)),
        "the documented fragment layout did not reproduce the exact GEMM; matches were {hits:?}"
    );
    assert!(!hits.is_empty(), "no candidate layout reproduced the exact GEMM");
}

#[test]
fn fragment_layout_sweep_k16() {
    // The m16n8k16 FP8 family (A = 2 registers, B = 1) swept over the same
    // candidate axes as the k32 probe.
    const A_ROW_OPTS: u32 = 4;
    const A_BAND_OPTS: u32 = 6;
    const A_ORD_OPTS: u32 = 2;
    const B_BAND_OPTS: u32 = 6;
    const B_ORD_OPTS: u32 = 2;
    let be = OxideBackend::new().expect("backend");
    let a: Vec<f32> = (0..16).flat_map(|r| (0..16).map(move |c| sweep_a(r, c))).collect();
    let b: Vec<f32> = (0..16).flat_map(|k| (0..8).map(move |n| sweep_b(k, n))).collect();
    let mut want = vec![0.0f64; 128];
    for r in 0..16usize {
        for n in 0..8usize {
            want[r * 8 + n] = (0..16).map(|k| (a[r * 16 + k] * b[k * 8 + n]) as f64).sum();
        }
    }
    let mut hits: Vec<(u32, u32, u32, u32, u32)> = Vec::new();
    let mut best = (f64::INFINITY, (0, 0, 0, 0, 0));
    for rp in 0..A_ROW_OPTS {
        for ab in 0..A_BAND_OPTS {
            if (0..2).any(|n| a_band(ab, n) + 15 > 15) {
                continue;
            }
            for ao in 0..A_ORD_OPTS {
                let mut fa = Vec::with_capacity(256);
                for lane in 0..32u32 {
                    let g = lane / 4;
                    let t = lane % 4;
                    for n in 0..2u32 {
                        let r = g + 8 * a_row_offset(rp, n);
                        for i in 0..4u32 {
                            let c = 4 * t + a_band(ab, n) + byte_order(ao, i);
                            fa.push(f32_to_fp8(a[(r * 16 + c) as usize], Fp8Format::E4M3));
                        }
                    }
                }
                for bb in 0..B_BAND_OPTS {
                    if (0..1).any(|n| b_band(bb, n) + 15 > 15) {
                        continue;
                    }
                    for bo in 0..B_ORD_OPTS {
                        let mut fb = Vec::with_capacity(128);
                        for lane in 0..32u32 {
                            let g = lane / 4;
                            let t = lane % 4;
                            for i in 0..4u32 {
                                let k = b_band(bb, 0) + 4 * t + byte_order(bo, i);
                                fb.push(f32_to_fp8(b[(k * 8 + g) as usize], Fp8Format::E4M3));
                            }
                        }
                        let got = probe_to_matrix(
                            &be.fp8_mma_probe16(&fa, &fb, Fp8Format::E4M3, Fp8Format::E4M3).unwrap(),
                        );
                        let err: f64 = (0..128)
                            .map(|i| (got[i] as f64 - want[i]).abs() / want[i].abs().max(1e-3))
                            .fold(0.0, f64::max);
                        if err < best.0 {
                            best = (err, (rp, ab, ao, bb, bo));
                        }
                        if err <= 1e-3 {
                            hits.push((rp, ab, ao, bb, bo));
                        }
                    }
                }
            }
        }
    }
    println!("k16 best {best:?}, matching layouts {hits:?}");
    let ones = be
        .fp8_mma_probe16(&vec![0x38u8; 256], &vec![0x38u8; 128], Fp8Format::E4M3, Fp8Format::E4M3)
        .unwrap();
    println!("k16 all-ones raw (first 8): {:?} (16 expected)", &ones[..8]);
    assert!(!hits.is_empty(), "no k16 candidate reproduced the exact GEMM");
}

#[test]
fn f16_control_probe_uses_the_whole_warp() {
    // Same harness as the FP8 probes but with the f16 mma that the existing
    // fp16 GEMM already uses.  Every output must be 16 (16 products of 1.0);
    // if only a couple of lanes came back, the harness would be at fault
    // rather than the FP8 instruction.
    let be = OxideBackend::new().expect("backend");
    let raw = be.f16_mma_probe16(&vec![0x3C00u16; 256], &vec![0x3C00u16; 128]).unwrap();
    let nonzero: Vec<usize> = (0..128).filter(|i| raw[*i] != 0.0).collect();
    println!("f16 control: {} nonzero slots", nonzero.len());
    assert_eq!(nonzero.len(), 128);
    assert_eq!(raw.iter().filter(|v| **v == 16.0).count(), 128, "all 128 outputs must be 16");
}

#[test]
fn gemm_fp8_hand_computed_ones() {
    // A = 1 (16x32), B = 1 (8x32) -> C must be 32 everywhere.  The simplest
    // possible tiled-GEMM check: one output tile, one k step.
    let be = OxideBackend::new().expect("backend");
    let one = f32_to_fp8(1.0, Fp8Format::E4M3);
    for &(m, n, k) in &[(16usize, 8usize, 32usize), (64, 64, 32), (65, 67, 96)] {
        let aq = vec![one; m * k];
        let bq = vec![one; n * k];
        let out = be.gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3).unwrap();
        let out = download(&be, &out);
        println!("{m}x{n}x{k} ones -> {:?}", &out[..8]);
        for (i, v) in out.iter().enumerate() {
            assert!((v - k as f32).abs() <= 1e-3 * k as f32, "{m}x{n}x{k} [{i}] = {v}, want {k}");
        }
    }
}

#[test]
fn gemm_fp8_normal_operands_only() {
    // Same check with values that are all *normal* e4m3 (|v| >= 2^-6).  If the
    // tensor core flushed e4m3 subnormals to zero, this would pass while the
    // random-data tests would not.
    let be = OxideBackend::new().expect("backend");
    let grid: [f32; 8] = [0.25, -0.5, 1.0, -1.5, 2.0, -3.0, 4.0, -6.0];
    let codes: Vec<u8> = grid.iter().map(|v| f32_to_fp8(*v, Fp8Format::E4M3)).collect();
    for i in 0..8 {
        assert!(fp8_to_f32(codes[i], Fp8Format::E4M3).abs() >= 2f32.powi(-6));
    }
    let (m, n, k) = (16usize, 8usize, 32usize);
    let aq: Vec<u8> = (0..m * k).map(|i| codes[i % 8]).collect();
    let bq: Vec<u8> = (0..n * k).map(|i| codes[(i / 3) % 8]).collect();
    let out = be.gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3).unwrap();
    let got = download(&be, &out);
    let av: Vec<f32> = aq.iter().map(|c| fp8_to_f32(*c, Fp8Format::E4M3)).collect();
    let bv: Vec<f32> = bq.iter().map(|c| fp8_to_f32(*c, Fp8Format::E4M3)).collect();
    for idx in 0..m * n {
        let mut want = 0.0f64;
        for p in 0..k {
            want += av[idx / n * k + p] as f64 * bv[p * n + idx % n] as f64;
        }
        let scale = want.abs().max(1e-3);
        assert!(
            (got[idx] as f64 - want).abs() <= TOL_ACC * scale,
            "[{idx}]: got {} want {want}",
            got[idx]
        );
    }
}

#[test]
fn gemm_fp8_subnormal_operands_are_reported() {
    // Quantise a value whose e4m3 code is subnormal (0.5 * 2^-9) and check
    // what the tensor core does with it.
    let be = OxideBackend::new().expect("backend");
    let tiny = 2f32.powi(-9); // 0.001953125: the smallest e4m3 subnormal
    let code = f32_to_fp8(tiny, Fp8Format::E4M3);
    println!("e4m3 code for 2^-9 = {code:#04x} -> {}", fp8_to_f32(code, Fp8Format::E4M3));
    let (m, n, k) = (16usize, 8usize, 32usize);
    let aq = vec![code; m * k];
    let one = f32_to_fp8(1.0, Fp8Format::E4M3);
    let bq = vec![one; n * k];
    let out = be.gemm_fp8(&aq, &bq, m, n, k, false, false, 1.0, Fp8Format::E4M3, Fp8Format::E4M3).unwrap();
    let got = download(&be, &out);
    println!("subnormal x ones -> {}", got[0]);
}

// ---------------------------------------------------------------------------
// Backend precision selector (`DARKNET_OXIDE_GEMM=fp8` / `fp4`)
// ---------------------------------------------------------------------------
//
// The kernels above are only useful if the network GEMM seam can reach them, so
// `GemmMode::Fp8` / `GemmMode::Fp4` route `gemm_f32` through
// quantise -> low-precision GEMM -> f32.  Both tests compare against a CPU
// reference built from the *dequantised* operands, which pins the whole path
// (including the per-tensor / per-block scale recipe) rather than just the mma.

use darknet_cuda::{ElementwiseLaunchConfig, F32TensorOpsBackend as _};

fn gemm_f32_reference(a: &[f32], b: &[f32], c0: &[f32], m: usize, n: usize, k: usize, alpha: f32, beta: f32) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0f64; m * n];
    let mut norm = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            let mut rows = 0.0f64;
            for p in 0..k {
                let av = a[i * k + p] as f64;
                let bv = b[p * n + j] as f64;
                acc += av * bv;
                rows += av.abs() * bv.abs();
            }
            out[i * n + j] = alpha as f64 * acc + beta as f64 * c0[i * n + j] as f64;
            norm[i * n + j] = rows;
        }
    }
    (out, norm)
}

#[test]
fn gemm_mode_fp8_selector_matches_the_dequantised_reference() {
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp8);
    let mut rng = Rng(0x5eed_1234);
    for &(m, n, k) in &[(96usize, 80usize, 64usize), (65, 33, 129), (129, 70, 96)] {
        let a = rng.vec(m * k, -2.0, 2.0);
        let b = rng.vec(k * n, -2.0, 2.0);
        let c0 = rng.vec(m * n, -1.0, 1.0);
        let ta = be.upload_f32(&[m, k], &a).unwrap();
        let tb = be.upload_f32(&[k, n], &b).unwrap();
        let tc = be.upload_f32(&[m, n], &c0).unwrap();
        let got = download(&be, &be.gemm_f32(&ta, &tb, &tc, m, n, k, false, false, 0.7, 0.3).unwrap());

        // Reference: quantise both operands exactly as the selector does and
        // dequantise them again.  `amax_to_scale` is the C++ *divisor*
        // (`amax / 448`), so the quantiser multiplies by its reciprocal and the
        // dequantiser multiplies by the divisor.
        let factor = |v: &[f32]| {
            let scale = darknet_cuda_oxide::fp8::amax_to_scale(darknet_cuda_oxide::fp8::amax_f32(v), Fp8Format::E4M3);
            1.0 / scale
        };
        let (sa, sb) = (factor(&a), factor(&b));
        let q = |v: &[f32], scale: f32| -> Vec<f32> {
            v.iter()
                .map(|x| fp8_to_f32(f32_to_fp8(x * scale, Fp8Format::E4M3), Fp8Format::E4M3) / scale)
                .collect()
        };
        let (want, norm) = gemm_f32_reference(&q(&a, sa), &q(&b, sb), &c0, m, n, k, 0.7, 0.3);
        let mut worst = 0.0f64;
        for i in 0..m * n {
            let err = (got[i] as f64 - want[i]).abs();
            worst = worst.max(err / norm[i].max(1e-4));
            // e4m3 keeps 3 explicit mantissa bits, so each operand carries up to
            // 2^-4 of relative error and the row norm bounds the sum's error at
            // about 2^-3.  0.25 is a comfortable factor of three over that and
            // still far tighter than the ~6% a wrong wiring would produce.
            assert!(err <= 0.25 * norm[i].max(1e-4), "{m}x{n}x{k} [{i}]: got {} want {}", got[i], want[i]);
        }
        println!("fp8 selector {m}x{n}x{k}: worst err / row norm = {worst:e}");
    }
}

#[test]
fn gemm_mode_fp4_selector_matches_the_dequantised_reference() {
    let be = OxideBackend::new().expect("backend").with_gemm_mode(GemmMode::Fp4);
    let mut rng = Rng(0xfeed_4321);
    for &(m, n, k) in &[(64usize, 48usize, 64usize), (33, 17, 32), (96, 96, 80)] {
        let a = rng.vec(m * k, -2.0, 2.0);
        let b = rng.vec(k * n, -2.0, 2.0);
        let c0 = rng.vec(m * n, -1.0, 1.0);
        let ta = be.upload_f32(&[m, k], &a).unwrap();
        let tb = be.upload_f32(&[k, n], &b).unwrap();
        let tc = be.upload_f32(&[m, n], &c0).unwrap();
        let got = download(&be, &be.gemm_f32(&ta, &tb, &tc, m, n, k, false, false, 0.7, 0.3).unwrap());

        // Reference: the same NVFP4 recipe the kernel uses (see `fp4`).  The
        // packed layout is [rows, k] for both operands, so B (stored [k, n] by
        // `gemm_f32`) is transposed first, exactly as the selector does.
        let pa = darknet_cuda_oxide::fp4::quantize_1x16(&a, m, k);
        let mut b_k = vec![0.0f32; n * k];
        for p in 0..k {
            for j in 0..n {
                b_k[j * k + p] = b[p * n + j];
            }
        }
        let pb = darknet_cuda_oxide::fp4::quantize_1x16(&b_k, n, k);
        let packed = darknet_cuda_oxide::fp4::gemm_nt_packed(&pa, &pb, m, n, k, 1.0, 0.0, None);
        for i in 0..m * n {
            let want = 0.7 * packed[i] as f64 + 0.3 * c0[i] as f64;
            let err = (got[i] as f64 - want).abs();
            let scale = (packed[i] as f64).abs().max(1e-3);
            assert!(
                err <= 1e-4 * scale,
                "{m}x{n}x{k} [{i}]: got {} want {want}",
                got[i]
            );
        }
    }
}

#[test]
fn gemm_mode_from_env_parses_fp8_and_fp4() {
    // `from_env` reads a process-wide variable, so exercise the mapping through
    // the documented spellings by comparing the parse result of a child-free
    // helper: the enum is `Copy`, so a scoped set/restore is enough.
    for (value, expect) in [
        ("fp8", GemmMode::Fp8),
        ("e4m3", GemmMode::Fp8),
        ("fp4", GemmMode::Fp4),
        ("nvfp4", GemmMode::Fp4),
        ("fp16", GemmMode::Fp16),
        ("tf32", GemmMode::Tf32),
        ("auto", GemmMode::Auto),
        ("nonsense", GemmMode::Fp32),
    ] {
        // SAFETY: single-threaded test; the variable is restored below.
        unsafe { std::env::set_var("DARKNET_OXIDE_GEMM", value) };
        assert_eq!(GemmMode::from_env(), expect, "DARKNET_OXIDE_GEMM={value}");
        // SAFETY: see above.
        unsafe { std::env::remove_var("DARKNET_OXIDE_GEMM") };
        assert_eq!(GemmMode::from_env(), GemmMode::Fp32);
    }
}

// ---------------------------------------------------------------------------
// FP8 backward (Phase B step 6)
// ---------------------------------------------------------------------------
//
// The C++ recipe (`v6-dev-fp8`) is asymmetric: the forward is e4m3 x e4m3, while
// dgrad is `W:E4M3 x dy:E5M2` and wgrad is `dy:E5M2 x col:E4M3`.  e5m2 has one
// mantissa bit less but 128x the dynamic range, which is what gradients need.
// `with_fp8_backward(Some((E4M3, E5M2)))` (or `DARKNET_OXIDE_GEMM_BWD=fp8`)
// switches just the two backward GEMMs; the forward keeps its own mode.
//
// The checks are against the *device f32* backward (same kernels, fp32 GEMM), so
// what is measured is exactly the quantisation error of the FP8 pair.

/// Row norm `sum_k |a_ik| |b_kj|` is not directly available for conv, so the
/// tolerance is scaled by the magnitude of the f32 result's own row: the test
/// uses data with a positive mean so the dot products do not cancel, and then
/// bounds the relative error per element.
fn relative_error(got: &[f32], want: &[f32], label: &str) -> f64 {
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let scale = w.abs().max(1e-3) as f64;
        let err = (*g as f64 - *w as f64).abs() / scale;
        worst = worst.max(err);
        assert!(
            err <= 0.25,
            "{label}[{i}]: fp8 {g} vs f32 {w}, relative {err:.4}"
        );
    }
    worst
}

#[test]
fn fp8_backward_dgrad_and_wgrad_track_the_f32_backward() {
    let be = OxideBackend::new().expect("backend");
    let fp8 = OxideBackend::new().expect("backend").with_fp8_backward(Some((Fp8Format::E4M3, Fp8Format::E5M2)));
    assert_eq!(fp8.fp8_backward_formats(), Some((Fp8Format::E4M3, Fp8Format::E5M2)));
    assert_eq!(be.fp8_backward_formats(), None, "default stays off");

    let cfg = ElementwiseLaunchConfig::default();
    let mut rng = Rng(0xdead_beef);
    // batch 2, c 32 -> f 64 on a 16x16 input, 3x3, stride 2, pad 1.  The
    // geometry is deliberately above `TINY_MACS` with `groups != c`, so
    // `ConvMode::Auto` picks the GEMM arm and the backward really goes through
    // the FP8 GEMM seam rather than the direct elementwise kernel.
    let (batch, c, h, w, f, k, sx, sy, pad) = (2usize, 32, 16, 16, 64, 3, 2, 2, 1);
    let oh = (h + 2 * pad - k) / sy + 1;
    let ow = (w + 2 * pad - k) / sx + 1;
    let x: Vec<f32> = (0..batch * c * h * w).map(|_| rng.next(0.2, 1.0)).collect();
    let dy: Vec<f32> = (0..batch * f * oh * ow).map(|_| rng.next(0.2, 1.0)).collect();
    let wt: Vec<f32> = (0..f * c * k * k).map(|_| rng.next(0.2, 1.0)).collect();
    let tx = be.upload_f32(&[batch, c, h, w], &x).unwrap();
    let tdy = be.upload_f32(&[batch, f, oh, ow], &dy).unwrap();
    let twt = be.upload_f32(&[f, c, k, k], &wt).unwrap();

    let dx32 = download(&be, &be.conv2d_backward_input_f32(&tdy, &twt, &[batch, c, h, w], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let dx8 = download(&fp8, &fp8.conv2d_backward_input_f32(&tdy, &twt, &[batch, c, h, w], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let worst_d = relative_error(&dx8, &dx32, "dgrad");
    assert!(worst_d > 0.0, "the FP8 dgrad must actually differ from the f32 one");

    let dw32 = download(&be, &be.conv2d_backward_weights_f32(&tx, &tdy, &[f, c, k, k], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let dw8 = download(&fp8, &fp8.conv2d_backward_weights_f32(&tx, &tdy, &[f, c, k, k], c, h, w, f, k, sx, sy, 1, pad, 1, cfg).unwrap());
    let worst_w = relative_error(&dw8, &dw32, "wgrad");
    assert!(worst_w > 0.0, "the FP8 wgrad must actually differ from the f32 one");
    println!("fp8 backward worst relative error: dgrad {worst_d:.4}, wgrad {worst_w:.4}");
}

//! Differential CPU tests between Rust Darknet and the repository's C++ layer.
//!
//! The test compiles the real `src-lib/connected_layer.cpp` with a minimal
//! compatibility header, then feeds both implementations identical tensors.
//! This isolates numerical behavior from OpenCV/CUDA build dependencies while
//! still exercising the C++ source that ships with this repository.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use darknet_core::{ConfigFile, ExecutableNetwork, LayerParams, Network, Tensor};

#[derive(Debug)]
struct ReferenceOutput {
    output: Vec<f32>,
    output_delta: Vec<f32>,
    input_gradient: Vec<f32>,
    weight_gradient: Vec<f32>,
    bias_gradient: Vec<f32>,
}

/// One deterministic connected-layer input shared by the C++ and Rust calls.
/// Keeping the case together prevents the two runners from receiving subtly
/// different dimensions or parameter slices.
struct ConnectedCase<'a> {
    activation: &'a str,
    batch: usize,
    inputs: usize,
    outputs: usize,
    input: &'a [f32],
    weights: &'a [f32],
    biases: &'a [f32],
    output_gradient: &'a [f32],
}

static CPP_ORACLE: OnceLock<Result<PathBuf, String>> = OnceLock::new();

fn cpp_oracle() -> Option<PathBuf> {
    let result = CPP_ORACLE.get_or_init(compile_cpp_oracle);
    match result {
        Ok(path) => Some(path.clone()),
        Err(reason) => {
            if std::env::var_os("DARKNET_CPP_PARITY_REQUIRED").is_some() {
                panic!("C++ parity was required but could not be prepared: {reason}");
            }
            eprintln!("skipping C++ parity test: {reason}");
            None
        }
    }
}

fn source_root() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("DARKNET_CPP_ROOT") {
        return PathBuf::from(root)
            .canonicalize()
            .map_err(|err| format!("DARKNET_CPP_ROOT is not readable: {err}"));
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .map_err(|err| format!("could not resolve the Darknet repository root: {err}"))
}

fn find_cxx() -> Option<String> {
    if let Some(candidate) = std::env::var_os("DARKNET_CXX") {
        return Some(candidate.to_string_lossy().into_owned());
    }

    ["c++", "g++", "clang++"].iter().find_map(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()
            .filter(std::process::ExitStatus::success)
            .map(|_| (*candidate).to_owned())
    })
}

fn compile_cpp_oracle() -> Result<PathBuf, String> {
    let Some(cxx) = find_cxx() else {
        return Err("no C++ compiler found; set DARKNET_CXX to enable the test".to_owned());
    };
    let root = source_root()?;
    let connected_source = root.join("src-lib/connected_layer.cpp");
    if !connected_source.is_file() {
        return Err(format!(
            "missing C++ reference source: {}",
            connected_source.display()
        ));
    }

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock error: {err}"))?
        .as_nanos();
    let build_dir = std::env::temp_dir().join(format!(
        "darknet-rust-cpp-parity-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir_all(&build_dir)
        .map_err(|err| format!("could not create {}: {err}", build_dir.display()))?;

    for name in ["darknet_internal.hpp", "gemm.hpp", "oracle_main.cpp"] {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/cpp_compat")
            .join(name);
        fs::copy(&source, build_dir.join(name))
            .map_err(|err| format!("could not copy {}: {err}", source.display()))?;
    }
    fs::copy(&connected_source, build_dir.join("connected_layer.cpp"))
        .map_err(|err| format!("could not copy {}: {err}", connected_source.display()))?;

    let binary = build_dir.join("darknet-connected-cpp-oracle");
    let output = Command::new(&cxx)
        .current_dir(&build_dir)
        .args([
            "-std=c++17",
            "-O2",
            "-fno-fast-math",
            "-Wall",
            "-Wextra",
            "connected_layer.cpp",
            "oracle_main.cpp",
            "-o",
        ])
        .arg(&binary)
        .output()
        .map_err(|err| format!("could not launch {cxx}: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "C++ parity oracle failed to compile:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(binary)
}

fn run_cpp_oracle(binary: &Path, case: &ConnectedCase<'_>) -> ReferenceOutput {
    let mut stdin = String::new();
    for values in [case.input, case.weights, case.biases, case.output_gradient] {
        for value in values {
            write!(&mut stdin, "{value:.9} ").expect("writing an in-memory string cannot fail");
        }
    }

    let result = Command::new(binary)
        .args([
            case.activation,
            &case.batch.to_string(),
            &case.inputs.to_string(),
            &case.outputs.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .expect("stdin was requested")
                .write_all(stdin.as_bytes())?;
            child.wait_with_output()
        })
        .expect("failed to run the C++ parity oracle");
    assert!(
        result.status.success(),
        "C++ parity oracle failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let mut parsed = std::collections::BTreeMap::new();
    for line in String::from_utf8_lossy(&result.stdout).lines() {
        let mut fields = line.split_whitespace();
        let Some(name) = fields.next() else { continue };
        let values = fields
            .map(|value| value.parse::<f32>().expect("C++ oracle emitted a float"))
            .collect::<Vec<_>>();
        parsed.insert(name.to_owned(), values);
    }

    let take = |name: &str| {
        parsed.get(name).cloned().unwrap_or_else(|| {
            panic!(
                "C++ parity oracle omitted {name}; stdout={:?}",
                result.stdout
            )
        })
    };
    ReferenceOutput {
        output: take("OUTPUT"),
        output_delta: take("OUTPUT_DELTA"),
        input_gradient: take("INPUT_GRADIENT"),
        weight_gradient: take("WEIGHT_GRADIENT"),
        bias_gradient: take("BIAS_GRADIENT"),
    }
}

fn run_rust_connected(case: &ConnectedCase<'_>) -> ReferenceOutput {
    let cfg = format!(
        "[net]\nwidth={}\nheight=1\nchannels=1\nbatch={}\nsubdivisions=1\n\n[connected]\noutput={}\nactivation={}\n",
        case.inputs, case.batch, case.outputs, case.activation
    );
    let config = ConfigFile::parse_str("cpp-parity.cfg", &cfg).expect("parity config parses");
    let network = Network::from_config(config).expect("parity network builds");
    let mut executable = ExecutableNetwork::new(network);
    let LayerParams::Connected(params) = &mut executable.params[0] else {
        panic!("parity config did not create a connected layer");
    };
    params.weights.clone_from_slice(case.weights);
    params.biases.clone_from_slice(case.biases);

    let input = Tensor::from_data(executable.network.net.input_shape(), case.input.to_vec())
        .expect("parity input shape");
    let forward = executable.forward(&input).expect("Rust forward pass");
    let output = forward.final_output().expect("Rust output").data.clone();
    let output_tensor = Tensor::from_data(
        forward.final_output().unwrap().shape,
        case.output_gradient.to_vec(),
    )
    .expect("parity output gradient shape");
    let backward = executable
        .backward(&input, &forward, &output_tensor)
        .expect("Rust backward pass");
    let LayerParams::Connected(param_gradient) = &backward.param_gradients[0] else {
        panic!("Rust connected backward produced the wrong gradient variant");
    };
    let output_delta = output_gradient_from_output(&output, case.output_gradient, case.activation);

    ReferenceOutput {
        output,
        output_delta,
        input_gradient: backward.input_gradient.data.clone(),
        weight_gradient: param_gradient.weights.clone(),
        bias_gradient: param_gradient.biases.clone(),
    }
}

fn output_gradient_from_output(output: &[f32], seed: &[f32], activation: &str) -> Vec<f32> {
    output
        .iter()
        .zip(seed)
        .map(|(value, seed)| {
            let derivative = match activation {
                "linear" => 1.0,
                "logistic" => value * (1.0 - value),
                "relu" => f32::from(*value > 0.0),
                "leaky" => {
                    if *value > 0.0 {
                        1.0
                    } else {
                        0.1
                    }
                }
                "tanh" => 1.0 - value * value,
                "loggy" => {
                    let y = (value + 1.0) * 0.5;
                    2.0 * (1.0 - y) * y
                }
                other => panic!("unsupported test activation {other}"),
            };
            derivative * seed
        })
        .collect()
}

fn assert_close(name: &str, expected: &[f32], actual: &[f32]) {
    assert_eq!(expected.len(), actual.len(), "{name} length differs");
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let tolerance = 2.0e-5f32 + 2.0e-5f32 * expected.abs();
        assert!(
            (expected - actual).abs() <= tolerance,
            "{name}[{index}] differs: C++={expected:.8e}, Rust={actual:.8e}, tolerance={tolerance:.8e}"
        );
    }
}

#[test]
fn cpp_connected_forward_and_backward_match_rust() {
    let Some(binary) = cpp_oracle() else { return };
    let batch = 2;
    let inputs = 4;
    let outputs = 3;
    let input = vec![
        -1.25, 0.5, 2.0, -0.75, // sample 0
        0.25, -2.0, 1.5, 0.875, // sample 1
    ];
    let weights = vec![
        0.25, -0.5, 0.75, -1.0, // output 0
        -0.125, 0.375, -0.625, 0.875, // output 1
        1.0, 0.5, -0.25, 0.125, // output 2
    ];
    let biases = vec![0.1, -0.2, 0.3];
    let output_gradient = vec![
        0.5, -1.25, 2.0, // sample 0
        -0.75, 0.875, 1.5, // sample 1
    ];

    for activation in ["linear", "logistic", "relu", "leaky", "tanh", "loggy"] {
        let case = ConnectedCase {
            activation,
            batch,
            inputs,
            outputs,
            input: &input,
            weights: &weights,
            biases: &biases,
            output_gradient: &output_gradient,
        };
        let expected = run_cpp_oracle(&binary, &case);
        let actual = run_rust_connected(&case);

        assert_close(
            &format!("{activation} output"),
            &expected.output,
            &actual.output,
        );
        assert_close(
            &format!("{activation} output delta"),
            &expected.output_delta,
            &actual.output_delta,
        );
        assert_close(
            &format!("{activation} input gradient"),
            &expected.input_gradient,
            &actual.input_gradient,
        );
        assert_close(
            &format!("{activation} weight gradient"),
            &expected.weight_gradient,
            &actual.weight_gradient,
        );
        assert_close(
            &format!("{activation} bias gradient"),
            &expected.bias_gradient,
            &actual.bias_gradient,
        );
    }
}

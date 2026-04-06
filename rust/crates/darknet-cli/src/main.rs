use std::path::Path;

use darknet_core::{ExecutableNetwork, Network, inspect_weights_file, load_names_file};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args();
    let _bin = args.next();
    let Some(command) = args.next() else {
        print_usage();
        return Ok(());
    };

    match command.as_str() {
        "version" => {
            println!("darknet-rs {}", env!("CARGO_PKG_VERSION"));
            println!("cutile reference {}", darknet_cuda::CUTILE_RS_REFERENCE);
        }
        "cfg-validate" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let network = Network::from_path(&cfg)?;
            println!(
                "validated {} with {} layers and {} parameters",
                cfg,
                network.layers.len(),
                network.total_parameter_count()
            );
        }
        "cfg-summary" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let network = Network::from_path(&cfg)?;
            print_network_summary(&network);
        }
        "names" => {
            let names_path = required_arg(args.next(), "names path")?;
            let names = load_names_file(&names_path)?;
            println!("loaded {} class names from {}", names.len(), names_path);
            for (idx, name) in names.iter().take(10).enumerate() {
                println!("{idx:>3}: {name}");
            }
        }
        "weights-inspect" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let weights = required_arg(args.next(), "weights path")?;
            let network = Network::from_path(&cfg)?;
            let inspection = inspect_weights_file(&network, &weights)?;
            println!(
                "weights version {}.{}.{}",
                inspection.header.major, inspection.header.minor, inspection.header.revision
            );
            println!("images seen {}", inspection.header.seen);
            println!(
                "accounted {} bytes of {} bytes (trailing {})",
                inspection.accounted_bytes, inspection.file_size, inspection.trailing_bytes
            );
            for layer in inspection.layers.iter().take(16) {
                let bytes: u64 = layer.arrays.iter().map(|array| array.byte_len).sum();
                println!(
                    "#{:>3} {:<18} {} bytes",
                    layer.layer_index, layer.layer_name, bytes
                );
            }
        }
        "forward-dummy" => {
            let cfg = required_arg(args.next(), "cfg path")?;
            let maybe_weights = args.next();
            let mut executable = ExecutableNetwork::from_path(&cfg)?;
            if let Some(weights) = maybe_weights {
                let header = executable.load_weights_file(&weights)?;
                println!(
                    "loaded weights {}.{}.{} seen={}",
                    header.major, header.minor, header.revision, header.seen
                );
            } else {
                println!("running with zero-initialized weights");
            }

            let input = executable.zero_input();
            let result = executable.forward(&input)?;
            let final_output = result
                .final_output()
                .ok_or("network produced no layer outputs")?;

            println!("layers executed {}", result.layer_outputs.len());
            println!("final output {}", final_output.shape);
            println!("yolo heads {}", result.yolo_heads.len());
            for head in &result.yolo_heads {
                println!("#{:>3} yolo {}", head.layer_index, head.output.shape);
            }
        }
        "help" | "--help" | "-h" => print_usage(),
        other => {
            return Err(format!("unknown command: {other}").into());
        }
    }

    Ok(())
}

fn required_arg(value: Option<String>, name: &str) -> Result<String, Box<dyn std::error::Error>> {
    value.ok_or_else(|| format!("missing required argument: {name}").into())
}

fn print_usage() {
    println!("darknet-rs <command> [args]");
    println!("commands:");
    println!("  version");
    println!("  cfg-validate <cfg>");
    println!("  cfg-summary <cfg>");
    println!("  names <names>");
    println!("  weights-inspect <cfg> <weights>");
    println!("  forward-dummy <cfg> [weights]");
}

fn print_network_summary(network: &Network) {
    println!("cfg {}", display_path(&network.source));
    println!(
        "input {} batch={} subdivisions={}",
        network.net.input_shape(),
        network.net.batch,
        network.net.subdivisions
    );
    println!(
        "layers={} weighted_layers={} parameters={}",
        network.layers.len(),
        network.weighted_layers().count(),
        network.total_parameter_count()
    );

    for layer in &network.layers {
        let params = layer
            .weights()
            .map(|weights| weights.parameter_count())
            .unwrap_or(0);
        println!(
            "#{:>3} {:<18} {:<14} -> {:<14} params={}",
            layer.index,
            layer.kind.name(),
            layer.input_shape,
            layer.output_shape,
            params
        );
    }
}

fn display_path(path: &Path) -> String {
    path.display().to_string()
}

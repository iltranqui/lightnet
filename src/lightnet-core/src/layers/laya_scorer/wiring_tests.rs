    //! End-to-end checks that `[laya_scorer]` parses, builds parameters, runs
    //! on the CPU and the GPU, and loads its real blob.

    use lightnet_runtime::laya_scorer::*;
    use crate::{ConfigFile, ExecutableNetwork, LayerParams, Network, Tensor};

    fn cfg(hidden: usize, max_options: usize) -> String {
        format!(
            "[net]\nwidth={max_options}\nheight=1\nchannels={hidden}\nbatch=2\n\
             [convolutional]\nfilters={hidden}\nsize=1\nactivation=linear\n\
             [laya_scorer]\nhidden={hidden}\nmax_options={max_options}\n"
        )
    }

    #[test]
    fn the_layer_parses_builds_parameters_and_runs() {
        let network =
            Network::from_config(ConfigFile::parse_str("laya.cfg", &cfg(4, 3)).unwrap()).unwrap();
        // The shape contract: [batch, hidden, 1, max_options] -> [batch, 1, 1, max_options].
        assert_eq!(network.layers[1].input_shape.channels, 4);
        assert_eq!(network.layers[1].input_shape.width, 3);
        let out = network.layers[1].output_shape;
        assert_eq!(
            (out.channels, out.height, out.width, out.batch),
            (1, 1, 3, 2)
        );
        assert!(
            network.layers[1].weights().is_none(),
            "the blob is not the sequential stream"
        );

        let exec = ExecutableNetwork::new(network);
        let LayerParams::LayaScorer(_) = &exec.params[1] else {
            panic!("expected scorer params")
        };
        let input_shape = exec.network.net.input_shape();
        let data: Vec<f32> = (0..input_shape.batch * input_shape.element_count())
            .map(|i| (i % 9) as f32 * 0.1 - 0.4)
            .collect();
        let input = Tensor::from_data(input_shape, data.clone()).unwrap();
        let forward = exec.forward_training(&input).unwrap();
        assert_eq!(forward.layer_outputs[1].data.len(), 2 * 3);
        assert!(forward.layer_outputs[1].data.iter().all(|v| v.is_finite()));
        // With all-zero weights LayerNorm gives 0, GELU(0) = 0, so the logit is
        // just fc2_b (0.0).
        assert!(
            forward.layer_outputs[1].data.iter().all(|v| v.abs() < 1e-6),
            "{:?}",
            forward.layer_outputs[1].data
        );
    }

    #[test]
    fn the_backward_reaches_the_previous_layer() {
        let network =
            Network::from_config(ConfigFile::parse_str("laya.cfg", &cfg(4, 3)).unwrap()).unwrap();
        let mut exec = ExecutableNetwork::new(network);
        {
            // The convolution is zero-initialised, which would swallow the
            // gradient before it reaches the network input, so give it content.
            let LayerParams::Convolutional(conv) = &mut exec.params[0] else {
                panic!()
            };
            for (i, v) in conv.weights.iter_mut().enumerate() {
                *v = (i % 5) as f32 * 0.1 - 0.2;
            }
            conv.biases.iter_mut().for_each(|b| *b = 0.0);
            let LayerParams::LayaScorer(weights) = &mut exec.params[1] else {
                panic!()
            };
            // C++ zero-initialises the scorer and expects the loader to fill
            // it, so the *unloaded* LayerNorm gamma is 0 and the whole chain
            // collapses.  A trained-like gamma (near 1) and a dense fc1 are what
            // make the seed reach the input.
            weights.ln_w.fill(1.0);
            for (i, v) in weights.fc1_w.iter_mut().enumerate() {
                *v = ((i * 7 % 9) as f32 - 4.0) / 6.0;
            }
            for (i, v) in weights.fc2_w.iter_mut().enumerate() {
                *v = (i as f32 % 3.0) * 0.5 - 0.25;
            }
            weights.fc2_b = 0.1;
        }
        let input_shape = exec.network.net.input_shape();
        // A *varying* input matters here: LayerNorm's Jacobian annihilates the
        // all-ones direction, so a constant row would legitimately produce a
        // zero input gradient and the test would prove nothing.
        let data: Vec<f32> = (0..input_shape.batch * input_shape.element_count())
            .map(|i| (i % 7) as f32 * 0.3 - 0.9)
            .collect();
        let input = Tensor::from_data(input_shape, data).unwrap();
        let forward = exec.forward_training(&input).unwrap();
        let mut seeds = vec![None; exec.network.layers.len()];
        seeds[1] =
            Some(Tensor::from_data(exec.network.layers[1].output_shape, vec![1.0f32; 6]).unwrap());
        let result = exec
            .backward_accumulate_seeded(&input, &forward, &seeds)
            .unwrap();
        assert_eq!(
            result.input_gradient.data.len(),
            input_shape.batch * input_shape.element_count()
        );
        assert!(
            result.input_gradient.data.iter().any(|v| v.abs() > 1e-6),
            "the seed must reach the convolution: {:?}",
            result.input_gradient.data
        );
    }

    #[test]
    fn the_shape_contract_is_enforced_at_parse_time() {
        let bad_channel = format!(
            "[net]\nwidth=3\nheight=1\nchannels=5\n[convolutional]\nfilters=5\nsize=1\n\
             [laya_scorer]\nhidden=4\nmax_options=3\n"
        );
        let err = Network::from_config(ConfigFile::parse_str("laya.cfg", &bad_channel).unwrap())
            .expect_err("channels must equal hidden");
        assert!(format!("{err}").contains("channels=hidden=4"), "{err}");
        let bad_width = format!(
            "[net]\nwidth=5\nheight=1\nchannels=4\n[convolutional]\nfilters=4\nsize=1\n\
             [laya_scorer]\nhidden=4\nmax_options=3\n"
        );
        let err = Network::from_config(ConfigFile::parse_str("laya.cfg", &bad_width).unwrap())
            .expect_err("width must equal max_options");
        assert!(format!("{err}").contains("width=max_options=3"), "{err}");
    }

    #[test]
    fn the_real_blob_loads_into_the_layer_parameters() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/laya/laya_english.weights");
        let Ok(bytes) = std::fs::read(&path) else {
            eprintln!("skipping: {} is not present", path.display());
            return;
        };
        // The scorer must follow a layer that produces `[batch, hidden, 1,
        // max_options]`, which is exactly what `[laya_marker_gather]` would; a
        // 1x1 convolution stands in here so the test does not need that layer
        // ported first.
        let network =
            Network::from_config(ConfigFile::parse_str("laya.cfg", &cfg(1024, 4)).unwrap())
                .unwrap();
        let mut exec = ExecutableNetwork::new(network);
        let offset = scorer_blob_offset_bytes(1024, 28, 2624, 50368, 1024, 2, 4096, false);
        {
            let LayerParams::LayaScorer(weights) = &mut exec.params[1] else {
                panic!()
            };
            let blob = &bytes[offset..offset + 4 * blob_float_count(1024)];
            *weights = LayaScorerWeights::from_blob(
                &blob
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>(),
            )
            .unwrap();
        }
        let input_shape = exec.network.net.input_shape();
        let input = Tensor::zeros(input_shape);
        let forward = exec.forward_training(&input).unwrap();
        assert!(
            forward.layer_outputs[1].data.iter().all(|v| v.is_finite()),
            "the real scorer must produce finite logits"
        );
    }
//! Unrun compact/full native Tucker source fixture against the Rust CPU.
use lightnet_core::{ConfigFile, ExecutableNetwork, GpuExecutableNetwork, LayerParams, Network, Tensor};
use lightnet_cuda_oxide::OxideBackend;

#[test]
fn compact_tucker_padded_windows_dense_and_sparse_match_cpu() {
    for msa in [0,1] {
        let text = format!("[net]\nbatch=2\nwidth=3\nheight=2\nchannels=4\n[tucker_attention]\nfilters=4\nsize=2\nheads=2\nrank_q=2\nrank_k=3\nrank_v=2\nrank_o=3\nactivation=gelu\nmsa={msa}\nmsa_block_size=2\nmsa_topk=1\n");
        let network = Network::from_config(ConfigFile::parse_str("compact-tucker.cfg",&text).unwrap()).unwrap();
        let mut cpu = ExecutableNetwork::new(network.clone());
        let LayerParams::TuckerAttention(p) = &mut cpu.params[0] else { panic!("Tucker params") };
        for (i,w) in p.weights.iter_mut().enumerate() { *w=(i as f32*0.19).sin()*0.15; }
        for (i,b) in p.biases.iter_mut().enumerate() { *b=(i as f32*0.23).cos()*0.1; }
        let shape = cpu.network.net.input_shape();
        let input = Tensor::from_data(shape,(0..shape.batch*shape.element_count()).map(|i|(i as f32*0.17).sin()).collect()).unwrap();
        let expected = cpu.forward(&input).unwrap().final_output().unwrap().clone();
        let mut gpu = GpuExecutableNetwork::new(network,OxideBackend::new().unwrap());
        gpu.params=cpu.params.clone();
        let full=gpu.forward(&input).unwrap().final_output().unwrap().clone();
        let compact=gpu.forward_compact(&input).unwrap().final_output.unwrap();
        let scale=expected.data.iter().fold(1.0_f32,|a,x|a.max(x.abs()));
        for result in [full,compact] {
            assert_eq!(result.shape,expected.shape);
            for (i,(&a,&b)) in result.data.iter().zip(&expected.data).enumerate() {
                assert!((a-b).abs()<=3e-4*scale,"msa={msa} [{i}]: {a} vs {b}");
            }
        }
    }
}

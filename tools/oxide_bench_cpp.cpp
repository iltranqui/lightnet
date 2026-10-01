// C++ CUDA Darknet latency probe for tools/oxide_bench.py (links the in-tree libdarknet.so).
// Usage: oxide_bench_cpp <cfg> <weights|-> <batch> <iters> <warmup> <cudnn_half 0|1> [train]
// With `train` the probe times synthetic training steps instead (random init weights, random
// input, all-zero truth; train_network_datum_gpu = forward+backward incl. the yolo loss, then
// update_network_gpu = SGD, synchronised). `weights`, `batch` and `cudnn_half` are ignored: the batch
// comes from the cfg, the cfg parser picks cudnn_half for the GPU, and BN is NOT fused.
// Prints one JSON line: per-iteration ms (full network_predict: H2D, forward, D2H sync),
// device memory delta. Input is a fixed pseudo-random tensor (conv timing is data independent).
#include "darknet_internal.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>

int main(int argc, char **argv)
{
	if (argc < 7) { std::fprintf(stderr, "usage: %s cfg weights batch iters warmup cudnn_half\n", argv[0]); return 2; }
	const char *cfg = argv[1], *weights = argv[2];
	const int batch = std::atoi(argv[3]), iters = std::atoi(argv[4]), warmup = std::atoi(argv[5]), half = std::atoi(argv[6]);
	cudaFree(nullptr);
	size_t free0 = 0, total = 0;
	cudaMemGetInfo(&free0, &total);
	// Training-parity mode (docs/OXIDE_BENCH.md section 8):
	//   oxide_bench_cpp <cfg> <weights> 0 0 0 0 parity <input.bin> <truth.bin> <steps> <lr> <dump_prefix> <dump_steps csv> [delta_prefix]
	// Same init weights / fixed input / fixed truth, real yolo loss, constant lr, SGD from the cfg. Prints
	// PARITY step=k loss=.. delta_l2=.. and dumps conv params (biases, [biases, scales, mean, var], weights; f32 LE).
	if (argc > 13 && std::string(argv[7]) == "parity")
	{
		Darknet::Network net = parse_network_cfg_custom(cfg, 0, 0);
		load_weights(&net, weights);
		net.train = 1;
		net.policy = CONSTANT;
		net.burn_in = 0;
		net.learning_rate = static_cast<float>(std::atof(argv[11]));
		const int steps = std::atoi(argv[10]);
		const std::string prefix = argv[12];
		const std::string delta_prefix = argc > 14 ? argv[14] : "";
		if (!delta_prefix.empty())
			for (int i = 0; i < net.n; ++i)
				if (net.layers[i].type == Darknet::ELayerType::YOLO) net.layers[i].onlyforward = 1;
		std::vector<int> dump_steps;
		for (char *t = std::strtok(argv[13], ","); t; t = std::strtok(nullptr, ",")) dump_steps.push_back(std::atoi(t));
		auto slurp = [](const char *p) { std::vector<float> v; FILE *f = std::fopen(p, "rb"); if (!f) { std::perror(p); std::exit(3); }
			float b[4096]; size_t n; while ((n = std::fread(b, 4, 4096, f)) > 0) v.insert(v.end(), b, b + n); std::fclose(f); return v; };
		std::vector<float> x = slurp(argv[8]);
		std::vector<float> tr = slurp(argv[9]);
		const int truths = net.layers[net.n - 1].truths;
		std::vector<float> y(static_cast<size_t>(truths) * net.batch, 0.0f);
		const size_t per = tr.size() / net.batch;
		for (int b = 0; b < net.batch; ++b)
			for (size_t i = 0; i < per && i < static_cast<size_t>(truths); ++i) y[b * truths + i] = tr[b * per + i];
		auto dump = [&](int step) {
			std::vector<float> bytes;
			std::string lens;
			for (int i = 0; i < net.n; ++i)
			{
				Darknet::Layer &l = net.layers[i];
				if (l.type != Darknet::ELayerType::CONVOLUTIONAL) continue;
				pull_convolutional_layer(l);
				auto put = [&](const float *p, int n) { bytes.insert(bytes.end(), p, p + n); };
				put(l.biases, l.n);
				if (l.batch_normalize) { put(l.biases, l.n); put(l.scales, l.n); put(l.rolling_mean, l.n); put(l.rolling_variance, l.n); }
				put(l.weights, l.nweights);
				lens += (lens.empty() ? "" : ",") + std::to_string(i) + ":" + std::to_string(l.n) + ":" + std::to_string(l.batch_normalize ? l.n : 0) + ":" + std::to_string(l.nweights);
			}
			FILE *f = std::fopen((prefix + ".layers").c_str(), "w"); std::fputs(lens.c_str(), f); std::fclose(f);
			f = std::fopen((prefix + ".step" + std::to_string(step) + ".bin").c_str(), "wb"); std::fwrite(bytes.data(), 4, bytes.size(), f); std::fclose(f);
		};
		auto wants = [&](int s) { return std::find(dump_steps.begin(), dump_steps.end(), s) != dump_steps.end(); };
		if (wants(0)) dump(0);
		std::printf("PARITY cfg batch=%d cudnn_half=%d subdivisions=%d momentum=%g decay=%g lr=%g truths=%d\n", net.batch, net.cudnn_half, net.subdivisions, net.momentum, net.decay, net.learning_rate, truths);
		for (int s = 1; s <= steps; ++s)
		{
			float cost = 0;
			double sur = 0;
			if (delta_prefix.empty())
				cost = train_network_datum_gpu(net, x.data(), y.data());
			else
			{
				// fixed synthetic yolo deltas: yolo layers run forward-only (no loss), delta_gpu is overwritten after the
				// forward with <prefix>.<layer>.bin, then backward = forward_backward_network_gpu minus the loss.
				*net.seen += net.batch;
				Darknet::NetworkState state;
				state.index = 0; state.net = net;
				const int x_size = get_network_input_size(net) * net.batch;
				if (!*net.input_gpu) { *net.input_gpu = cuda_make_array(x.data(), x_size); *net.truth_gpu = cuda_make_array(y.data(), y.size()); }
				else cuda_push_array(*net.input_gpu, x.data(), x_size);
				state.input = *net.input_gpu; state.delta = 0; state.truth = *net.truth_gpu; state.train = 1;
				for (int i = 0; i < net.n; ++i)
				{
					Darknet::Layer &l = net.layers[i];
					if (net.cudnn_half && l.type == Darknet::ELayerType::CONVOLUTIONAL && l.weights_gpu && l.weights_gpu16)
						cuda_convert_f32_to_f16(l.weights_gpu, l.nweights, l.weights_gpu16);
				}
				forward_network_gpu(net, state);
				cudaDeviceSynchronize();
				for (int i = 0; i < net.n; ++i)
				{
					Darknet::Layer &l = net.layers[i];
					if (l.type != Darknet::ELayerType::YOLO) continue;
					l.onlyforward = 0; // backward_network_gpu skips onlyforward layers
					std::vector<float> d = slurp((delta_prefix + "." + std::to_string(i) + ".bin").c_str());
					if (d.size() != static_cast<size_t>(l.batch) * l.outputs) { std::fprintf(stderr, "delta size %zu != %d\n", d.size(), l.batch * l.outputs); std::exit(4); }
					cuda_push_array(l.delta_gpu, d.data(), d.size());
					std::memcpy(l.delta, d.data(), d.size() * 4);
					for (size_t k = 0; k < d.size(); ++k) sur += double(d[k]) * l.output[k];
					if (s == 1) { FILE *f = std::fopen((prefix + ".out." + std::to_string(i) + ".bin").c_str(), "wb"); std::fwrite(l.output, 4, d.size(), f); std::fclose(f); }
				}
				backward_network_gpu(net, state);
				cudaDeviceSynchronize();
			}
			double d2 = 0;
			for (int i = 0; i < net.n; ++i)
				if (net.layers[i].type == Darknet::ELayerType::YOLO)
					for (int k = 0; k < net.layers[i].batch * net.layers[i].outputs; ++k) d2 += double(net.layers[i].delta[k]) * net.layers[i].delta[k];
			update_network_gpu(net);
			cudaDeviceSynchronize();
			std::printf("PARITY backend=cpp step=%d loss=%.6f delta_l2=%.6f surrogate=%.6f\n", s, cost, std::sqrt(d2), sur);
			if (wants(s)) dump(s);
		}
		return 0;
	}
	if (argc > 7 && std::string(argv[7]) == "train")
	{
		Darknet::Network net = parse_network_cfg_custom(cfg, 0, 0); // batch 0 = batch/subdivisions from the cfg, training buffers
		// cudnn_half stays as chosen by the cfg parser for this GPU (overriding it after parsing breaks cuDNN descriptors).
		net.train = 1;
		const size_t nx = static_cast<size_t>(get_network_input_size(net)) * net.batch;
		const size_t ny = net.layers[net.n - 1].truths ? static_cast<size_t>(net.layers[net.n - 1].truths) * net.batch
		                                               : static_cast<size_t>(get_network_output_size(net)) * net.batch;
		std::vector<float> x(nx), y(ny, 0.0f);
		unsigned seed = 12345u;
		for (auto &v : x) { seed = seed * 1664525u + 1013904223u; v = (seed >> 8) / 16777216.0f; }
		std::vector<double> ms;
		size_t peak = 0;
		for (int i = 0; i < warmup + iters; ++i)
		{
			auto t0 = std::chrono::steady_clock::now();
			train_network_datum_gpu(net, x.data(), y.data());
			update_network_gpu(net);
			cudaDeviceSynchronize();
			double d = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			size_t f = 0;
			cudaMemGetInfo(&f, &total);
			peak = std::max(peak, free0 - f);
			if (i >= warmup) ms.push_back(d);
		}
		std::printf("{\"ms\":[");
		for (size_t i = 0; i < ms.size(); ++i) std::printf("%s%.4f", i ? "," : "", ms[i]);
		std::printf("],\"device_mem_peak_mb\":%.1f,\"batch\":%d,\"cudnn_half\":%d,\"mode\":\"train\"}\n",
			peak / 1048576.0, net.batch, net.cudnn_half);
		return 0;
	}
	DarknetNetworkPtr ptr = load_network_custom(cfg, std::string(weights) == "-" ? nullptr : weights, 0, batch);
	Darknet::Network *net = reinterpret_cast<Darknet::Network *>(ptr);
	net->cudnn_half = half;
	const size_t n = static_cast<size_t>(get_network_input_size(*net)) * net->batch;
	std::vector<float> in(n);
	unsigned s = 12345u;
	for (auto &v : in) { s = s * 1664525u + 1013904223u; v = (s >> 8) / 16777216.0f; }
	for (int i = 0; i < warmup; ++i) network_predict(*net, in.data());
	std::vector<double> ms;
	for (int i = 0; i < iters; ++i)
	{
		auto t0 = std::chrono::steady_clock::now();
		network_predict(*net, in.data());
		ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
	}
	size_t free1 = 0;
	cudaMemGetInfo(&free1, &total);
	std::printf("{\"ms\":[");
	for (size_t i = 0; i < ms.size(); ++i) std::printf("%s%.4f", i ? "," : "", ms[i]);
	std::printf("],\"device_mem_used_mb\":%.1f,\"batch\":%d,\"cudnn_half\":%d,\"cuda_graph\":%d}\n",
		(free0 - free1) / 1048576.0, net->batch, net->cudnn_half, net->use_cuda_graph);
	return 0;
}

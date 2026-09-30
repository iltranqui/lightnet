#include "darknet_internal.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

Darknet::Layer make_connected_layer(int batch, int steps, int inputs, int outputs,
                                    ACTIVATION activation, int batch_normalize);
void forward_connected_layer(Darknet::Layer &layer, Darknet::NetworkState state);
void backward_connected_layer(Darknet::Layer &layer, Darknet::NetworkState state);

static ACTIVATION parse_activation(const std::string &name)
{
    if (name == "linear") {
        return LINEAR;
    }
    if (name == "logistic") {
        return LOGISTIC;
    }
    if (name == "relu") {
        return RELU;
    }
    if (name == "leaky") {
        return LEAKY;
    }
    if (name == "tanh") {
        return TANH;
    }
    if (name == "loggy") {
        return LOGGY;
    }
    throw std::runtime_error("unsupported parity activation: " + name);
}

static void print_vector(const char *label, const std::vector<float> &values)
{
    std::cout << label;
    for (const float value : values) {
        std::cout << ' ' << std::setprecision(9) << value;
    }
    std::cout << '\n';
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cerr << "usage: oracle <activation> <batch> <inputs> <outputs>\n";
        return 2;
    }

    const ACTIVATION activation = parse_activation(argv[1]);
    const int batch = std::stoi(argv[2]);
    const int inputs = std::stoi(argv[3]);
    const int outputs = std::stoi(argv[4]);

    std::vector<float> input(static_cast<std::size_t>(batch * inputs));
    std::vector<float> weights(static_cast<std::size_t>(outputs * inputs));
    std::vector<float> biases(static_cast<std::size_t>(outputs));
    std::vector<float> output_gradient(static_cast<std::size_t>(batch * outputs));

    for (float &value : input) {
        if (!(std::cin >> value)) {
            return 3;
        }
    }
    for (float &value : weights) {
        if (!(std::cin >> value)) {
            return 3;
        }
    }
    for (float &value : biases) {
        if (!(std::cin >> value)) {
            return 3;
        }
    }
    for (float &value : output_gradient) {
        if (!(std::cin >> value)) {
            return 3;
        }
    }

    // This constructor and both calls below come from the real Darknet
    // connected_layer.cpp copied from src-lib by the Rust integration test.
    Darknet::Layer layer = make_connected_layer(batch, 1, inputs, outputs, activation, 0);
    std::copy(weights.begin(), weights.end(), layer.weights);
    std::copy(biases.begin(), biases.end(), layer.biases);

    std::vector<float> input_gradient(static_cast<std::size_t>(batch * inputs), 0.0f);
    Darknet::NetworkState state{input.data(), input_gradient.data(), false};
    forward_connected_layer(layer, state);

    std::copy(output_gradient.begin(), output_gradient.end(), layer.delta);
    backward_connected_layer(layer, state);

    std::vector<float> output(layer.output, layer.output + batch * outputs);
    std::vector<float> delta(layer.delta, layer.delta + batch * outputs);
    std::vector<float> weight_gradient(layer.weight_updates,
                                       layer.weight_updates + outputs * inputs);
    std::vector<float> bias_gradient(layer.bias_updates, layer.bias_updates + outputs);

    print_vector("OUTPUT", output);
    print_vector("OUTPUT_DELTA", delta);
    print_vector("INPUT_GRADIENT", input_gradient);
    print_vector("WEIGHT_GRADIENT", weight_gradient);
    print_vector("BIAS_GRADIENT", bias_gradient);

    std::free(layer.output);
    std::free(layer.delta);
    std::free(layer.weight_updates);
    std::free(layer.bias_updates);
    std::free(layer.weights);
    std::free(layer.biases);
    return 0;
}

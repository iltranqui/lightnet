#pragma once

// This header deliberately contains only the small CPU surface needed to
// compile Darknet's real connected_layer.cpp in the parity test.  The full
// Darknet header graph pulls in OpenCV, CUDA, protobuf, and the public C API;
// none of those dependencies are relevant to the scalar connected-layer
// equations we want to compare.  Keeping this shim local to the test lets us
// execute the repository's C++ implementation instead of maintaining a
// second, hand-written C++ oracle.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#define TATPARMS 0
#define TAT(...) ((void)0)

enum ACTIVATION {
    LOGISTIC,
    RELU,
    RELU6,
    RELIE,
    LINEAR,
    RAMP,
    TANH,
    PLSE,
    REVLEAKY,
    LEAKY,
    ELU,
    LOGGY,
    STAIR,
    HARDTAN,
    LHTAN,
    SELU,
    GELU,
    SWISH,
    MISH,
    HARD_MISH,
    NORM_CHAN,
    NORM_CHAN_SOFTMAX,
    NORM_CHAN_SOFTMAX_MAXVAL,
};

namespace Darknet {

enum class ELayerType : int { CONNECTED = 0 };

struct Layer;
struct NetworkState {
    float *input = nullptr;
    float *delta = nullptr;
    bool train = false;
};

using ForwardFn = void (*)(Layer &, NetworkState);
using BackwardFn = void (*)(Layer &, NetworkState);
using UpdateFn = void (*)(Layer &, int, float, float, float);

// These are the fields read by the non-GPU functions in connected_layer.cpp.
// The production Layer is much larger; this ABI-independent test double is
// intentionally source-compatible with the CPU implementation's field names.
struct Layer {
    ELayerType type = ELayerType::CONNECTED;
    int inputs = 0;
    int outputs = 0;
    int batch = 0;
    int batch_normalize = 0;
    int h = 0;
    int w = 0;
    int c = 0;
    int out_h = 0;
    int out_w = 0;
    int out_c = 0;
    int n = 0;
    int size = 0;
    int stride = 0;
    int stride_x = 0;
    int stride_y = 0;
    int pad = 0;
    ACTIVATION activation = LINEAR;
    float learning_rate_scale = 1.0f;
    int groups = 1;
    int dilation = 1;

    float *output = nullptr;
    float *delta = nullptr;
    float *weight_updates = nullptr;
    float *bias_updates = nullptr;
    float *weights = nullptr;
    float *biases = nullptr;

    float *scales = nullptr;
    float *scale_updates = nullptr;
    float *mean = nullptr;
    float *mean_delta = nullptr;
    float *variance = nullptr;
    float *variance_delta = nullptr;
    float *rolling_mean = nullptr;
    float *rolling_variance = nullptr;
    float *x = nullptr;
    float *x_norm = nullptr;

    ForwardFn forward = nullptr;
    BackwardFn backward = nullptr;
    UpdateFn update = nullptr;
};

struct CfgAndState {
    std::ostream *output;

    static CfgAndState &get()
    {
        static CfgAndState state{&std::cerr};
        return state;
    }
};

} // namespace Darknet

inline void *xcalloc(const std::size_t count, const std::size_t size)
{
    void *memory = std::calloc(count, size);
    if (memory == nullptr && count != 0 && size != 0) {
        throw std::bad_alloc{};
    }
    return memory;
}

inline void rand_uniform_many(float *values, const int count, const float min,
                              const float max, const float scale)
{
    // Initialization is not part of parity: the harness overwrites every
    // parameter before calling forward_connected_layer.  Deterministic values
    // keep make_connected_layer harmless if the implementation changes.
    const float midpoint = (min + max) * 0.5f * scale;
    for (int index = 0; index < count; ++index) {
        values[index] = midpoint;
    }
}

inline void fill_cpu(const int count, const float value, float *dst, const int increment)
{
    for (int index = 0; index < count; ++index) {
        dst[index * increment] = value;
    }
}

inline void axpy_cpu(const int count, const float alpha, float *x, const int inc_x,
                     float *y, const int inc_y)
{
    for (int index = 0; index < count; ++index) {
        y[index * inc_y] += alpha * x[index * inc_x];
    }
}

inline void scal_cpu(const int count, const float alpha, float *x, const int increment)
{
    for (int index = 0; index < count; ++index) {
        x[index * increment] *= alpha;
    }
}

inline void copy_cpu(const int count, float *x, const int inc_x, float *y, const int inc_y)
{
    for (int index = 0; index < count; ++index) {
        y[index * inc_y] = x[index * inc_x];
    }
}

inline float activation_value(const float x, const ACTIVATION activation)
{
    switch (activation) {
    case LINEAR:
        return x;
    case LOGISTIC:
        return 1.0f / (1.0f + std::exp(-x));
    case RELU:
        return x > 0.0f ? x : 0.0f;
    case LEAKY:
    case REVLEAKY:
        return x > 0.0f ? x : 0.1f * x;
    case TANH:
        return 2.0f / (1.0f + std::exp(-2.0f * x)) - 1.0f;
    case LOGGY:
        return 2.0f / (1.0f + std::exp(-x)) - 1.0f;
    default:
        // The parity cases intentionally use the activations shared by the
        // CPU connected-layer tests.  Other production activations are not
        // silently claimed to be covered by this compatibility shim.
        return x;
    }
}

inline float activation_gradient_from_output(const float output, const ACTIVATION activation)
{
    switch (activation) {
    case LINEAR:
        return 1.0f;
    case LOGISTIC:
        return (1.0f - output) * output;
    case RELU:
        return output > 0.0f ? 1.0f : 0.0f;
    case LEAKY:
    case REVLEAKY:
        return output > 0.0f ? 1.0f : 0.1f;
    case TANH:
        return 1.0f - output * output;
    case LOGGY:
        return 2.0f * (1.0f - (output + 1.0f) / 2.0f) * ((output + 1.0f) / 2.0f);
    default:
        return 1.0f;
    }
}

inline void activate_array(float *values, const int count, const ACTIVATION activation)
{
    for (int index = 0; index < count; ++index) {
        values[index] = activation_value(values[index], activation);
    }
}

inline void gradient_array(const float *values, const int count, const ACTIVATION activation,
                           float *delta)
{
    // Darknet's gradient_array receives the post-activation output.  This is
    // important for logistic, tanh, and the leaky branch, and is why the Rust
    // implementation computes its derivative from the cached output too.
    for (int index = 0; index < count; ++index) {
        delta[index] *= activation_gradient_from_output(values[index], activation);
    }
}

inline void gemm_cpu(const int transpose_a, const int transpose_b, const int rows,
                     const int columns, const int inner, const float alpha, float *a,
                     const int leading_a, float *b, const int leading_b, const float beta,
                     float *c, const int leading_c)
{
    // This is the scalar row-major equivalent of Darknet's GEMM call.  The
    // loop order matches the reference layer's mathematical contract while
    // avoiding a dependency on OpenBLAS in the test process.
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < columns; ++column) {
            float sum = 0.0f;
            for (int index = 0; index < inner; ++index) {
                const float left = transpose_a ? a[index * leading_a + row]
                                                : a[row * leading_a + index];
                const float right = transpose_b ? b[column * leading_b + index]
                                                 : b[index * leading_b + column];
                sum += left * right;
            }
            c[row * leading_c + column] = alpha * sum + beta * c[row * leading_c + column];
        }
    }
}

// The following batch-normalization helpers are linked because the production
// connected-layer translation unit contains both BN and non-BN branches.  The
// parity test uses batch_normalize=0, but these implementations keep the shim
// semantically useful for a future BN parity case.
inline void mean_cpu(float *x, const int batch, const int filters, const int spatial,
                     float *mean)
{
    for (int filter = 0; filter < filters; ++filter) {
        mean[filter] = 0.0f;
        for (int b = 0; b < batch; ++b) {
            for (int position = 0; position < spatial; ++position) {
                mean[filter] += x[(b * filters + filter) * spatial + position];
            }
        }
        mean[filter] /= static_cast<float>(batch * spatial);
    }
}

inline void variance_cpu(float *x, float *mean, const int batch, const int filters,
                         const int spatial, float *variance)
{
    for (int filter = 0; filter < filters; ++filter) {
        variance[filter] = 0.0f;
        for (int b = 0; b < batch; ++b) {
            for (int position = 0; position < spatial; ++position) {
                const float diff = x[(b * filters + filter) * spatial + position] - mean[filter];
                variance[filter] += diff * diff;
            }
        }
        variance[filter] /= static_cast<float>(batch * spatial);
    }
}

inline void normalize_cpu(float *x, float *mean, float *variance, const int batch,
                          const int filters, const int spatial)
{
    for (int b = 0; b < batch; ++b) {
        for (int filter = 0; filter < filters; ++filter) {
            const float inverse_std = 1.0f / std::sqrt(variance[filter] + 0.00001f);
            for (int position = 0; position < spatial; ++position) {
                float &value = x[(b * filters + filter) * spatial + position];
                value = (value - mean[filter]) * inverse_std;
            }
        }
    }
}

inline void scale_bias(float *output, float *scales, const int batch, const int filters,
                       const int spatial)
{
    for (int b = 0; b < batch; ++b) {
        for (int filter = 0; filter < filters; ++filter) {
            for (int position = 0; position < spatial; ++position) {
                output[(b * filters + filter) * spatial + position] *= scales[filter];
            }
        }
    }
}

inline void backward_scale_cpu(float *, float *, int, int, int, float *) {}
inline void mean_delta_cpu(float *, float *, int, int, int, float *) {}
inline void variance_delta_cpu(float *, float *, float *, float *, int, int, int, float *) {}
inline void normalize_delta_cpu(float *, float *, float *, float *, float *, int, int, int,
                                float *)
{
}

inline void print_statistics(float *, int) {}

// Declarations needed before connected_layer.cpp assigns its callbacks.
void forward_connected_layer(Darknet::Layer &, Darknet::NetworkState);
void backward_connected_layer(Darknet::Layer &, Darknet::NetworkState);
void update_connected_layer(Darknet::Layer &, int, float, float, float);

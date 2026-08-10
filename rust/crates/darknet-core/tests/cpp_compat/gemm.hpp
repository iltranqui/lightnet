#pragma once

// connected_layer.cpp includes Darknet's GEMM header.  The parity build uses
// the scalar implementation declared in the compatibility header, so no BLAS
// or CPU feature library is needed to execute the actual C++ layer source.
#include "darknet_internal.hpp"

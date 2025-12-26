/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  NNUE evaluation - GPU-accelerated neural network evaluation.
*/

#include "core/types.h"
#include <string>

namespace MetalFish {
namespace NNUE {

// NNUE evaluation stub
// Full implementation will use Metal GPU kernels

bool load_network(const std::string& path) {
    // TODO: Load NNUE network weights
    return false;
}

Value evaluate() {
    // TODO: Run NNUE inference on GPU
    return VALUE_ZERO;
}

} // namespace NNUE
} // namespace MetalFish


/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Evaluation functions - integrates CPU and GPU evaluation.
*/

#include "core/types.h"

namespace MetalFish {
namespace Eval {

// Classical evaluation stub
Value evaluate_classical() {
    // TODO: Implement classical evaluation as fallback
    return VALUE_ZERO;
}

} // namespace Eval
} // namespace MetalFish


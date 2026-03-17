/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Node and Tree - Optimized for Apple Silicon
  Licensed under GPL-3.0
*/

#include "node.h"
#include <algorithm>
#include <cmath>

// All methods are currently defined inline in node.h.
// This translation unit ensures the header is compiled and linked,
// and serves as the home for any future out-of-line definitions.

namespace MetalFish {
namespace MCTS {

// Force at least one translation unit to instantiate the full Node/NodeTree
// class, catching any ODR or template issues at compile time.
static_assert(sizeof(Node) >= sizeof(double), "Node must contain WL");
static_assert(alignof(Node) == CACHE_LINE_SIZE, "Node alignment mismatch");

} // namespace MCTS
} // namespace MetalFish

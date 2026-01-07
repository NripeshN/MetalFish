/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include <string>

namespace MetalFish {

namespace Eval {

// Evaluate the position using GPU-accelerated NNUE
Value evaluate(const Position &pos);

// Batch evaluation for GPU parallelism
void batch_evaluate(const Position *positions, Value *scores, size_t count);

// Get tracing info for debugging
std::string trace(const Position &pos);

// Initialize evaluation
void init();

// NNUE network file handling
bool load_network(const std::string &path);
bool is_network_loaded();
std::string network_info();

} // namespace Eval

} // namespace MetalFish

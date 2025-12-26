/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
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

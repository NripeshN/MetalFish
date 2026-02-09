/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../core/types.h"
#include <string>

namespace MetalFish {
namespace NN {

// Map UCI move to policy index (0-1857)
int MoveToNNIndex(Move move, int transform = 0);

// Map policy index to UCI move
Move IndexToNNMove(int index, int transform = 0);

// Initialize policy tables
void InitPolicyTables();

} // namespace NN
} // namespace MetalFish

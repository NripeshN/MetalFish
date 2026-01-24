/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>
#include "../core/types.h"

namespace MetalFish {
namespace NN {

// Map UCI move to policy index (0-1857)
int MoveToNNIndex(Move move);

// Map policy index to UCI move  
Move IndexToNNMove(int index);

// Initialize policy tables
void InitPolicyTables();

}  // namespace NN
}  // namespace MetalFish

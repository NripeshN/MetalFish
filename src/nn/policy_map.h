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

int MoveToNNIndex(Move move, int transform = 0);

Move IndexToNNMove(int index, int transform = 0);

void InitPolicyTables();

} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <array>

#include "input_planes.h"

namespace MetalFish {
namespace NN {

struct NetworkOutput {
  std::array<float, kPolicyOutputs> policy{};
  float value = 0.0f;
  float wdl[3] = {};
  bool has_wdl = false;
  float moves_left = 0.0f;
  bool has_moves_left = false;
};

} // namespace NN
} // namespace MetalFish

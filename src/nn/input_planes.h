/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <array>

namespace MetalFish {
namespace NN {

constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
constexpr int kTotalPlanes = 112;
// 1792 regular moves + 66 underpromotions (22 directions * 3 piece types).
constexpr int kPolicyOutputs = 1858;

using InputPlanes = std::array<std::array<float, 64>, kTotalPlanes>;

} // namespace NN
} // namespace MetalFish

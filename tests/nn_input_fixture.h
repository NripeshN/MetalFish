/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "nn/encoder.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace MetalFish::Tests {

struct PackedPlaneCheck {
  int plane = 0;
  std::uint64_t mask = 0;
  float value = 0.0f;
};

struct PackedInputFixture {
  std::string fen;
  NN::InputPlanes planes{};
  std::vector<std::uint64_t> masks;
  std::vector<float> values;
  std::size_t nonzero_planes = 0;
  std::size_t full_mask_planes = 0;
};

constexpr const char *kStartPositionFixtureFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
constexpr std::uint64_t kFullPlaneMask = 0xffffffffffffffffULL;

constexpr std::array<PackedPlaneCheck, 18> kStartPositionPackedPlaneChecks = {{
    {0, 0x000000000000ff00ULL, 1.0f},
    {1, 0x0000000000000042ULL, 1.0f},
    {2, 0x0000000000000024ULL, 1.0f},
    {3, 0x0000000000000081ULL, 1.0f},
    {4, 0x0000000000000008ULL, 1.0f},
    {5, 0x0000000000000010ULL, 1.0f},
    {6, 0x00ff000000000000ULL, 1.0f},
    {7, 0x4200000000000000ULL, 1.0f},
    {8, 0x2400000000000000ULL, 1.0f},
    {9, 0x8100000000000000ULL, 1.0f},
    {10, 0x0800000000000000ULL, 1.0f},
    {11, 0x1000000000000000ULL, 1.0f},
    {12, 0x0000000000000000ULL, 0.0f},
    {NN::kAuxPlaneBase + 0, kFullPlaneMask, 1.0f},
    {NN::kAuxPlaneBase + 1, kFullPlaneMask, 1.0f},
    {NN::kAuxPlaneBase + 2, kFullPlaneMask, 1.0f},
    {NN::kAuxPlaneBase + 3, kFullPlaneMask, 1.0f},
    {NN::kAuxPlaneBase + 7, kFullPlaneMask, 1.0f},
}};

PackedInputFixture BuildStartPositionPackedInputFixture();
bool ValidateStartPositionPackedInputFixture(const PackedInputFixture &fixture,
                                             std::string *error);

} // namespace MetalFish::Tests

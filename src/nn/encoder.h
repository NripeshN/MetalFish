/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <array>
#include <span>
#include <vector>

#include "../core/position.h"
#include "proto/net.pb.h"

namespace MetalFish {
namespace NN {

enum BoardTransform {
  kNoTransform = 0,
  kFlipTransform = 1,     // Horizontal flip
  kMirrorTransform = 2,   // Vertical mirror
  kTransposeTransform = 4 // Diagonal transpose
};

constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
constexpr int kTotalPlanes = 112; // 8 history * 13 planes + 8 auxiliary

// 1792 regular moves + 66 underpromotions (22 directions * 3 piece types)
constexpr int kPolicyOutputs = 1858;

using InputPlanes = std::array<std::array<float, 64>, kTotalPlanes>;

enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };

void EncodePositionForNN(MetalFishNN::NetworkFormat::InputFormat input_format,
                         std::span<const Position *const> position_history,
                         int history_planes,
                         FillEmptyHistory fill_empty_history,
                         InputPlanes &output,
                         int *transform_out = nullptr);

InputPlanes
EncodePositionForNN(MetalFishNN::NetworkFormat::InputFormat input_format,
                    std::span<const Position *const> position_history,
                    int history_planes, FillEmptyHistory fill_empty_history,
                    int *transform_out = nullptr);

InputPlanes
EncodePositionForNN(const Position &pos,
                    MetalFishNN::NetworkFormat::InputFormat input_format =
                        MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE);

bool IsCanonicalFormat(MetalFishNN::NetworkFormat::InputFormat input_format);

int TransformForPosition(MetalFishNN::NetworkFormat::InputFormat input_format,
                         std::span<const Position *const> history);

} // namespace NN
} // namespace MetalFish

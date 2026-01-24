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

// Neural network input constants
constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
constexpr int kTotalPlanes = 112;  // 8 history * 13 planes + 8 auxiliary

// Policy output size (all possible moves in UCI encoding)
constexpr int kPolicyOutputs = 1858;

// Input planes type: 112 planes of 8x8 board
using InputPlanes = std::array<std::array<float, 64>, kTotalPlanes>;

enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };

// Encode position for neural network input
// Returns 112-plane representation compatible with training data
InputPlanes EncodePositionForNN(
    MetalFishNN::NetworkFormat::InputFormat input_format,
    const std::vector<Position>& position_history,
    int history_planes,
    FillEmptyHistory fill_empty_history,
    int* transform_out = nullptr);

// Simpler interface using current position only
InputPlanes EncodePositionForNN(
    const Position& pos,
    MetalFishNN::NetworkFormat::InputFormat input_format = 
        MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE);

// Check if format uses canonicalization
bool IsCanonicalFormat(MetalFishNN::NetworkFormat::InputFormat input_format);

// Get transform to apply for canonicalization
int TransformForPosition(MetalFishNN::NetworkFormat::InputFormat input_format,
                        const std::vector<Position>& history);

}  // namespace NN
}  // namespace MetalFish

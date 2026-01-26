/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/



























#pragma once

#include <span>

#include "../core/position.h"
#include "nn/network.h"
#include "../proto/net.pb.h"

namespace MetalFish::NN {

constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;

enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };

// Returns the transform that would be used in EncodePositionForNN.
int TransformForPosition(pbMetalFish::NN::NetworkFormat::InputFormat input_format,
                         const PositionHistory& history);

// Encodes the last position in history for the neural network request.
InputPlanes EncodePositionForNN(
    pbMetalFish::NN::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out);

InputPlanes EncodePositionForNN(
    pbMetalFish::NN::NetworkFormat::InputFormat input_format,
    std::span<const Position> positions, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out);

bool IsCanonicalFormat(pbMetalFish::NN::NetworkFormat::InputFormat input_format);
bool IsCanonicalArmageddonFormat(
    pbMetalFish::NN::NetworkFormat::InputFormat input_format);
bool IsHectopliesFormat(pbMetalFish::NN::NetworkFormat::InputFormat input_format);
bool Is960CastlingFormat(pbMetalFish::NN::NetworkFormat::InputFormat input_format);

uint16_t MoveToNNIndex(Move move, int transform);
Move MoveFromNNIndex(int idx, int transform);

}  // namespace MetalFish::NN

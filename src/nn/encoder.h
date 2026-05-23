/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <span>
#include <vector>

#include "../core/position.h"
#include "input_planes.h"
#include "proto/net.pb.h"

namespace MetalFish {
namespace NN {

enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };

void EncodePositionForNN(MetalFishNN::NetworkFormat::InputFormat input_format,
                         std::span<const Position *const> position_history,
                         int history_planes,
                         FillEmptyHistory fill_empty_history,
                         InputPlanes &output, int *transform_out = nullptr);

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

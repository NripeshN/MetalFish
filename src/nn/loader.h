/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "proto/net.pb.h"
#include "weights_file.h"

namespace MetalFish {
namespace NN {

using FloatVector = std::vector<float>;
using FloatVectors = std::vector<FloatVector>;

WeightsFile LoadWeightsFromFile(const std::string &filename);

std::optional<WeightsFile> LoadWeights(std::string_view location);

FloatVector DecodeLayer(const MetalFishNN::Weights::Layer &layer);

} // namespace NN
} // namespace MetalFish

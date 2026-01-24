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

namespace MetalFish {
namespace NN {

using FloatVector = std::vector<float>;
using FloatVectors = std::vector<FloatVector>;
using WeightsFile = MetalFishNN::Net;

// Load weights from file (supports .pb and .pb.gz formats)
WeightsFile LoadWeightsFromFile(const std::string& filename);

// Load weights with autodiscovery support
std::optional<WeightsFile> LoadWeights(std::string_view location);

// Discover weights file in common locations
std::string DiscoverWeightsFile();

// Decode layer weights to float vector
FloatVector DecodeLayer(const MetalFishNN::Weights::Layer& layer);

}  // namespace NN
}  // namespace MetalFish

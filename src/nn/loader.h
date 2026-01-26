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

#include "nn/network.h"
#include "../proto/net.pb.h"

namespace MetalFish::NN {

class OptionsDict;
using FloatVector = std::vector<float>;
using FloatVectors = std::vector<FloatVector>;

using WeightsFile = pbMetalFish::NN::Net;

// Read weights file and fill the weights structure.
WeightsFile LoadWeightsFromFile(const std::string& filename);

// Read weights from the "locations", which is one of:
// * "<autodiscover>" -- tries to find a file which looks like a weights file.
// * "<embed>" -- weights are embedded in the binary.
// * filename -- reads weights from the file.
// Returns std::nullopt if no weights file was found in <autodiscover> mode.
std::optional<WeightsFile> LoadWeights(std::string_view location);

// Tries to find a file which looks like a weights file, and located in
// directory of binary_name or one of subdirectories. If there are several such
// files, returns one which has the latest modification date.
std::string DiscoverWeightsFile();

}  // namespace MetalFish::NN

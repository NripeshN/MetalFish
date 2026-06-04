/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>

#include "loader.h"
#include "network_format_types.h"
#include "weights.h"

namespace MetalFish {
namespace NN {

std::string
ActivationToString(MetalFishNN::NetworkFormat_ActivationFunction activation);

NetworkFormatDescriptor DescribeNetworkFormat(const WeightsFile &file);

std::string SelectPolicyHeadName(const MultiHeadWeights &weights);
std::string SelectValueHeadName(const MultiHeadWeights &weights);

} // namespace NN
} // namespace MetalFish

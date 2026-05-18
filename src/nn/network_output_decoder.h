/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <vector>

#include "network.h"
#include "network_tensor_plan.h"

namespace MetalFish {
namespace NN {

std::vector<NetworkOutput>
DecodeNetworkOutputBatch(const NetworkTensorPlan &plan, const float *policy,
                         std::size_t policy_count, const float *value,
                         std::size_t value_count,
                         const float *moves_left,
                         std::size_t moves_left_count, int batch_size);

} // namespace NN
} // namespace MetalFish

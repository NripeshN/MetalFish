/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "network_tensor_plan.h"
#include "weights.h"

namespace MetalFish {
namespace NN {

struct NetworkWeightTensorView {
  std::string name;
  const float *data = nullptr;
  std::size_t elements = 0;
};

struct NetworkWeightInventory {
  std::vector<NetworkWeightTensorView> tensors;

  std::size_t TotalElements() const;
  std::size_t TotalBytes() const;
  bool Contains(std::string_view name) const;
  std::string Summary() const;
};

NetworkWeightInventory CreateNetworkWeightInventory(
    const MultiHeadWeights &weights, const std::string &policy_head,
    const std::string &value_head, const NetworkTensorPlan &plan);

} // namespace NN
} // namespace MetalFish

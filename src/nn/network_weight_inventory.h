/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "network_tensor_plan.h"
#include "weights.h"

namespace MetalFish {
namespace NN {

enum class NetworkWeightTensorKind {
  Generic,
  ConvWeight,
  ConvBias,
  DenseWeight,
  DenseBias,
  NormScale,
  NormBias,
  Gate,
  PositionalEncoding,
};

struct NetworkWeightTensorView {
  std::string name;
  const float *data = nullptr;
  std::size_t elements = 0;
  std::vector<std::uint32_t> dims;
  NetworkWeightTensorKind kind = NetworkWeightTensorKind::Generic;

  std::size_t ShapeElements() const;
  bool ShapeMatchesElements() const;
  std::string ShapeString() const;
};

struct NetworkWeightInventory {
  std::vector<NetworkWeightTensorView> tensors;

  std::size_t TotalElements() const;
  std::size_t TotalBytes() const;
  bool Contains(std::string_view name) const;
  const NetworkWeightTensorView *Find(std::string_view name) const;
  bool AllShapesMatchElements(std::string *error = nullptr) const;
  std::string Summary() const;
};

std::string NetworkWeightTensorKindName(NetworkWeightTensorKind kind);

NetworkWeightInventory CreateNetworkWeightInventory(
    const MultiHeadWeights &weights, const std::string &policy_head,
    const std::string &value_head, const NetworkTensorPlan &plan);

} // namespace NN
} // namespace MetalFish

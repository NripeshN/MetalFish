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

#include "network_format.h"
#include "network_tensor_plan.h"
#include "network_weight_inventory.h"

namespace MetalFish {
namespace NN {

enum class NetworkExecutionOpKind {
  InputPack,
  Convolution,
  Dense,
  LayerNorm,
  Gate,
  PositionalEncoding,
  Attention,
  FeedForward,
  PolicyMap,
  OutputDecode,
};

struct NetworkExecutionStep {
  NetworkExecutionOpKind kind = NetworkExecutionOpKind::Dense;
  std::string name;
  std::vector<std::string> tensors;
};

struct NetworkExecutionValidation {
  std::vector<std::string> errors;

  bool ok() const { return errors.empty(); }
  std::string Summary() const;
};

struct NetworkExecutionPlan {
  NetworkFormatDescriptor format;
  NetworkTensorPlan tensors;
  std::string policy_head;
  std::string value_head;
  std::vector<NetworkExecutionStep> steps;

  bool ContainsStep(std::string_view name) const;
  bool ReferencesTensor(std::string_view name) const;
  std::size_t TensorReferenceCount() const;
  std::string Summary() const;
  NetworkExecutionValidation
  ValidateAgainstInventory(const NetworkWeightInventory &inventory) const;
};

std::string NetworkExecutionOpKindName(NetworkExecutionOpKind kind);

NetworkExecutionPlan CreateNetworkExecutionPlan(
    const NetworkFormatDescriptor &format, const NetworkTensorPlan &tensors,
    const std::string &policy_head, const std::string &value_head,
    const NetworkWeightInventory &inventory);

} // namespace NN
} // namespace MetalFish

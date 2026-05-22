/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cpu_network.h"

#include "../network_tensor_plan.h"
#include "../weights.h"

#include <memory>
#include <sstream>
#include <stdexcept>

namespace MetalFish {
namespace NN {
namespace Cpu {
namespace {

std::string FirstUnsupportedExecutionStep(
    const NetworkResolvedExecutionPlan &plan) {
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::Attention) {
      return "CPU transformer backend does not support attention yet: " +
             step.name;
    }
  }
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::Convolution) {
      return "CPU transformer backend does not support convolution yet: " +
             step.name;
    }
  }
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::PolicyMap) {
      return "CPU transformer backend does not support policy mapping yet: " +
             step.name;
    }
  }
  for (const auto &step : plan.steps) {
    if (step.kind != NetworkExecutionOpKind::InputPack &&
        step.kind != NetworkExecutionOpKind::OutputDecode) {
      return "CPU transformer backend does not support " +
             NetworkExecutionOpKindName(step.kind) + " yet: " + step.name;
    }
  }
  return "CPU transformer backend has no execution kernels yet";
}

} // namespace

CpuNetwork::CpuNetwork(const WeightsFile &weights)
    : format_(DescribeNetworkFormat(weights)),
      tensor_plan_(CreateNetworkTensorPlan(format_)) {
  std::unique_ptr<MultiHeadWeights> decoded_weights;
  std::string policy_head;
  std::string value_head;
  try {
    decoded_weights = std::make_unique<MultiHeadWeights>(weights.weights());
    policy_head = SelectPolicyHeadName(*decoded_weights);
    value_head = SelectValueHeadName(*decoded_weights);
    const auto tensor_validation = ValidateNetworkTensorPlan(
        tensor_plan_, *decoded_weights, policy_head, value_head);
    if (!tensor_validation.ok()) {
      throw std::runtime_error(tensor_validation.Summary());
    }

    const auto inventory = CreateNetworkWeightInventory(
        *decoded_weights, policy_head, value_head, tensor_plan_);
    execution_plan_ = CreateNetworkExecutionPlan(
        format_, tensor_plan_, policy_head, value_head, inventory);
    const auto execution_validation =
        execution_plan_.ValidateAgainstInventory(inventory);
    if (!execution_validation.ok()) {
      throw std::runtime_error(execution_validation.Summary());
    }
    resolved_execution_plan_ =
        ResolveNetworkExecutionPlan(execution_plan_, inventory);
    const auto resolved_inventory =
        CreateResolvedNetworkWeightInventory(inventory,
                                             resolved_execution_plan_);
    weight_bytes_ = resolved_inventory.TotalBytes();
  } catch (const std::exception &e) {
    throw std::runtime_error("CPU transformer backend weight validation failed: " +
                             std::string(e.what()));
  }

  unsupported_execution_reason_ =
      FirstUnsupportedExecutionStep(resolved_execution_plan_);
}

NetworkOutput CpuNetwork::Evaluate(const InputPlanes &) {
  throw std::runtime_error(UnsupportedExecutionMessage());
}

std::vector<NetworkOutput>
CpuNetwork::EvaluateBatch(const std::vector<InputPlanes> &inputs) {
  if (inputs.empty())
    return {};
  throw std::runtime_error(UnsupportedExecutionMessage());
}

std::string CpuNetwork::GetNetworkInfo() const {
  std::ostringstream out;
  out << "CPU transformer backend (validation-only, format: "
      << format_.Summary() << ", tensors: " << tensor_plan_.Summary()
      << ", execution: " << resolved_execution_plan_.Summary()
      << ", weight_bytes=" << weight_bytes_ << ", executor=unsupported)";
  return out.str();
}

std::string CpuNetwork::UnsupportedExecutionMessage() const {
  return unsupported_execution_reason_.empty()
             ? "CPU transformer backend execution is not implemented yet"
             : unsupported_execution_reason_;
}

} // namespace Cpu
} // namespace NN
} // namespace MetalFish

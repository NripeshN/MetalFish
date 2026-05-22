/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network_tensor_plan.h"

#include "network_format.h"
#include "weights.h"

#include <sstream>
#include <string>

namespace MetalFish {
namespace NN {
namespace {

bool HasConvWeights(const BaseWeights::ConvBlock &block) {
  return !block.weights.empty() || !block.biases.empty();
}

bool HasEncoderWeights(const BaseWeights::EncoderLayer &layer) {
  return !layer.mha.q_w.empty() || !layer.mha.k_w.empty() ||
         !layer.mha.v_w.empty() || !layer.mha.dense_w.empty() ||
         !layer.ffn.dense1_w.empty() || !layer.ffn.dense2_w.empty();
}

bool HasPolicyHeadWeights(const MultiHeadWeights::PolicyHead &head) {
  if (!head.ip_pol_w.empty() || !head.ip_pol_b.empty() ||
      !head.ip2_pol_w.empty() || !head.ip3_pol_w.empty() ||
      !head.ip4_pol_w.empty() || HasConvWeights(head.policy1) ||
      HasConvWeights(head.policy)) {
    return true;
  }
  for (const auto &layer : head.pol_encoder) {
    if (HasEncoderWeights(layer))
      return true;
  }
  return false;
}

bool HasValueHeadWeights(const MultiHeadWeights::ValueHead &head) {
  return HasConvWeights(head.value) || !head.ip_val_w.empty() ||
         !head.ip_val_b.empty() || !head.ip1_val_w.empty() ||
         !head.ip2_val_w.empty() || !head.ip_val_err_w.empty();
}

bool HasMovesLeftWeights(const MultiHeadWeights &weights) {
  return HasConvWeights(weights.moves_left) || !weights.ip_mov_w.empty() ||
         !weights.ip1_mov_w.empty() || !weights.ip2_mov_w.empty();
}

void AddError(NetworkTensorValidation &validation, const std::string &error) {
  validation.errors.push_back(error);
}

} // namespace

size_t NetworkTensorPlan::InputMaskEntries(int batch_size) const {
  return static_cast<size_t>(batch_size) * input_planes;
}

size_t NetworkTensorPlan::InputValueEntries(int batch_size) const {
  return static_cast<size_t>(batch_size) * input_planes;
}

size_t NetworkTensorPlan::PolicyEntries(int batch_size) const {
  return static_cast<size_t>(batch_size) * policy_outputs;
}

size_t NetworkTensorPlan::ValueEntries(int batch_size) const {
  return static_cast<size_t>(batch_size) * value_outputs;
}

size_t NetworkTensorPlan::MovesLeftEntries(int batch_size) const {
  return static_cast<size_t>(batch_size) * moves_left_outputs;
}

size_t NetworkTensorPlan::RawPolicyEntries(int batch_size) const {
  return static_cast<size_t>(batch_size) * raw_policy_outputs;
}

std::string NetworkTensorPlan::Summary() const {
  std::ostringstream out;
  out << "input=" << input_planes << "x" << input_squares
      << ", policy=" << policy_outputs << ", value=" << value_outputs
      << ", moves_left=" << moves_left_outputs
      << ", raw_policy=" << raw_policy_outputs;
  return out.str();
}

NetworkTensorPlan
CreateNetworkTensorPlan(const NetworkFormatDescriptor &format) {
  NetworkTensorPlan plan;
  plan.wdl = format.wdl;
  plan.moves_left = format.moves_left;
  plan.conv_policy = format.conv_policy;
  plan.attention_policy = format.attention_policy;
  plan.value_outputs = format.wdl ? 3 : 1;
  plan.moves_left_outputs = format.moves_left ? 1 : 0;
  if (format.attention_policy) {
    plan.raw_policy_outputs = kNetworkAttentionPolicyScratch;
  } else if (format.conv_policy) {
    plan.raw_policy_outputs = kNetworkConvPolicyScratch;
  }
  return plan;
}

std::string NetworkTensorValidation::Summary() const {
  if (errors.empty())
    return "ok";

  std::ostringstream out;
  for (size_t i = 0; i < errors.size(); ++i) {
    if (i > 0)
      out << "; ";
    out << errors[i];
  }
  return out.str();
}

NetworkTensorValidation ValidateNetworkTensorPlan(
    const NetworkTensorPlan &plan, const MultiHeadWeights &weights,
    const std::string &policy_head, const std::string &value_head) {
  NetworkTensorValidation validation;

  if (plan.input_planes != kPackedInputPlaneCount)
    AddError(validation, "unexpected input plane count");
  if (plan.input_squares != kPackedInputSquareCount)
    AddError(validation, "unexpected input square count");
  if (plan.policy_outputs != kNetworkPolicyOutputs)
    AddError(validation, "unexpected policy output count");
  if (plan.value_outputs != (plan.wdl ? 3 : 1))
    AddError(validation, "value output width does not match value format");
  if (plan.moves_left_outputs != (plan.moves_left ? 1 : 0))
    AddError(validation, "moves-left output width does not match format");
  if (plan.attention_policy &&
      plan.raw_policy_outputs != kNetworkAttentionPolicyScratch) {
    AddError(validation, "attention policy scratch width mismatch");
  }
  if (plan.conv_policy &&
      plan.raw_policy_outputs != kNetworkConvPolicyScratch) {
    AddError(validation, "convolution policy scratch width mismatch");
  }
  if (!plan.attention_policy && !plan.conv_policy &&
      plan.raw_policy_outputs != 0) {
    AddError(validation, "classical policy should not allocate raw scratch");
  }

  const auto policy_it = weights.policy_heads.find(policy_head);
  if (policy_head.empty() || policy_it == weights.policy_heads.end()) {
    AddError(validation, "selected policy head is missing: " + policy_head);
  } else if (!HasPolicyHeadWeights(policy_it->second)) {
    AddError(validation, "selected policy head has no weights: " + policy_head);
  }

  const auto value_it = weights.value_heads.find(value_head);
  if (value_head.empty() || value_it == weights.value_heads.end()) {
    AddError(validation, "selected value head is missing: " + value_head);
  } else if (!HasValueHeadWeights(value_it->second)) {
    AddError(validation, "selected value head has no weights: " + value_head);
  }

  if (plan.moves_left && !HasMovesLeftWeights(weights)) {
    AddError(validation, "moves-left head is declared but has no weights");
  }

  return validation;
}

} // namespace NN
} // namespace MetalFish

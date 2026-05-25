/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "input_plane_packing.h"

namespace MetalFish {
namespace NN {

struct MultiHeadWeights;
struct NetworkFormatDescriptor;

constexpr int kNetworkPolicyOutputs = 1858;
constexpr int kNetworkAttentionPolicyScratch = 64 * 64 + 8 * 24;
constexpr int kNetworkConvPolicyScratch = 73 * 64;

struct NetworkTensorPlan {
  int input_planes = kPackedInputPlaneCount;
  int input_squares = kPackedInputSquareCount;
  int policy_outputs = kNetworkPolicyOutputs;
  int value_outputs = 1;
  int moves_left_outputs = 0;
  int raw_policy_outputs = 0;

  bool wdl = false;
  bool moves_left = false;
  bool conv_policy = false;
  bool attention_policy = false;

  size_t InputMaskEntries(int batch_size) const;
  size_t InputValueEntries(int batch_size) const;
  size_t PolicyEntries(int batch_size) const;
  size_t ValueEntries(int batch_size) const;
  size_t MovesLeftEntries(int batch_size) const;
  size_t RawPolicyEntries(int batch_size) const;
  std::string Summary() const;
};

enum class NetworkOutputTarget {
  Policy,
  Value,
  MovesLeft,
  RawPolicy,
};

std::string NetworkOutputTargetName(NetworkOutputTarget target);
int NetworkOutputTargetStride(const NetworkTensorPlan &plan,
                              NetworkOutputTarget target);
bool NetworkOutputTargetEnabled(const NetworkTensorPlan &plan,
                                NetworkOutputTarget target);

NetworkTensorPlan
CreateNetworkTensorPlan(const NetworkFormatDescriptor &format);

struct NetworkTensorValidation {
  std::vector<std::string> errors;

  bool ok() const { return errors.empty(); }
  std::string Summary() const;
};

NetworkTensorValidation ValidateNetworkTensorPlan(
    const NetworkTensorPlan &plan, const MultiHeadWeights &weights,
    const std::string &policy_head, const std::string &value_head);

} // namespace NN
} // namespace MetalFish

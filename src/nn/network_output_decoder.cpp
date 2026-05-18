/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network_output_decoder.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace MetalFish {
namespace NN {
namespace {

void ValidateTensorView(const char *name, const float *data,
                        std::size_t actual, std::size_t expected) {
  if (actual != expected) {
    throw std::runtime_error(std::string("NN output tensor size mismatch: ") +
                             name);
  }
  if (expected > 0 && data == nullptr) {
    throw std::runtime_error(std::string("NN output tensor is null: ") + name);
  }
}

} // namespace

std::vector<NetworkOutput>
DecodeNetworkOutputBatch(const NetworkTensorPlan &plan, const float *policy,
                         std::size_t policy_count, const float *value,
                         std::size_t value_count,
                         const float *moves_left,
                         std::size_t moves_left_count, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("NN output batch size must be positive");
  if (plan.policy_outputs != kPolicyOutputs)
    throw std::runtime_error("NN output policy width is not 1858");

  ValidateTensorView("policy", policy, policy_count,
                     plan.PolicyEntries(batch_size));
  ValidateTensorView("value", value, value_count,
                     plan.ValueEntries(batch_size));
  ValidateTensorView("moves_left", moves_left, moves_left_count,
                     plan.MovesLeftEntries(batch_size));

  std::vector<NetworkOutput> outputs(static_cast<std::size_t>(batch_size));
  for (int b = 0; b < batch_size; ++b) {
    NetworkOutput &out = outputs[static_cast<std::size_t>(b)];
    std::memcpy(out.policy.data(),
                policy + static_cast<std::size_t>(b) * plan.policy_outputs,
                sizeof(float) * plan.policy_outputs);

    if (plan.wdl) {
      const std::size_t offset = static_cast<std::size_t>(b) * 3;
      out.has_wdl = true;
      out.wdl[0] = value[offset + 0];
      out.wdl[1] = value[offset + 1];
      out.wdl[2] = value[offset + 2];
      out.value = out.wdl[0] - out.wdl[2];
    } else {
      out.has_wdl = false;
      out.value = value[b];
      out.wdl[0] = out.wdl[1] = out.wdl[2] = 0.0f;
    }

    if (plan.moves_left) {
      out.has_moves_left = true;
      out.moves_left = moves_left[b];
    } else {
      out.has_moves_left = false;
      out.moves_left = 0.0f;
    }
  }

  return outputs;
}

} // namespace NN
} // namespace MetalFish

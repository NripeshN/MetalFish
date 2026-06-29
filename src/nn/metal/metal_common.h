/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once
#include "../network_tensor_plan.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Metal {

static int kInputPlanes = kPackedInputPlaneCount;

// FP16 compute for the MPSGraph transformer body. Enabled by default via UCI
// option NNMetalFP16 (true) or legacy env var METALFISH_METAL_FP16. FP16 is
// ~14% faster per eval at batch 1 with near-identical policy/value outputs.
inline bool HalfPrecisionEnabled() {
  static bool cached = []() {
    const char *e = std::getenv("METALFISH_METAL_FP16");
    if (e && (e[0] == '0' || e[0] == 'f' || e[0] == 'F' || e[0] == 'n' ||
              e[0] == 'N'))
      return false;
    return true;
  }();
  return cached;
}

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, const NetworkTensorPlan &plan) {
    input_masks_mem_.resize(plan.InputMaskEntries(maxBatchSize));
    input_val_mem_.resize(plan.InputValueEntries(maxBatchSize));
    op_policy_mem_.resize(plan.PolicyEntries(maxBatchSize));
    op_value_mem_.resize(plan.ValueEntries(maxBatchSize));

    if (plan.moves_left) {
      op_moves_left_mem_.resize(plan.MovesLeftEntries(maxBatchSize));
    };
  }
  ~InputsOutputs() {}

  std::vector<float> &OutputBuffer(NetworkOutputTarget target) {
    switch (target) {
    case NetworkOutputTarget::Policy:
      return op_policy_mem_;
    case NetworkOutputTarget::Value:
      return op_value_mem_;
    case NetworkOutputTarget::MovesLeft:
      return op_moves_left_mem_;
    case NetworkOutputTarget::RawPolicy:
      break;
    }
    throw std::runtime_error("Metal decoded output buffer is unavailable");
  }

  std::vector<uint64_t> input_masks_mem_;
  std::vector<float> input_val_mem_;
  std::vector<float> op_policy_mem_;
  std::vector<float> op_value_mem_;
  std::vector<float> op_moves_left_mem_;
};

} // namespace Metal
} // namespace NN
} // namespace MetalFish

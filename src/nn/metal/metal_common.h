/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once
#include "../network_tensor_plan.h"

#include <cstdint>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Metal {

static int kNumOutputPolicy = kNetworkPolicyOutputs;
static int kInputPlanes = kPackedInputPlaneCount;

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, const NetworkTensorPlan &plan) {
    input_masks_mem_.resize(plan.InputMaskEntries(maxBatchSize));
    input_val_mem_.resize(plan.InputValueEntries(maxBatchSize));
    op_policy_mem_.resize(plan.PolicyEntries(maxBatchSize));
    op_value_mem_.resize(plan.ValueEntries(maxBatchSize));

    if (plan.moves_left) {
      op_moves_left_mem_.resize(plan.MovesLeftEntries(maxBatchSize));
    };

    /**
     * @todo policy map implementation has bug in MPSGraph (GatherND not working
     * in graph). Implementation of policy map to be done in CPU for now.
     *
     * Remove this op_policy_raw_mem_ memory allocation when bug is fixed.
     */
    if (plan.raw_policy_outputs > 0) {
      op_policy_raw_mem_.resize(plan.RawPolicyEntries(maxBatchSize));
    }
  }
  ~InputsOutputs() {}

  std::vector<uint64_t> input_masks_mem_;
  std::vector<float> input_val_mem_;
  std::vector<float> op_policy_mem_;
  std::vector<float> op_value_mem_;
  std::vector<float> op_moves_left_mem_;
  std::vector<float> op_policy_raw_mem_;
};

} // namespace Metal
} // namespace NN
} // namespace MetalFish

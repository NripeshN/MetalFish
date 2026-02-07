/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/
#pragma once
#include <cstdint>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Metal {

static constexpr int kNumOutputPolicy = 1858;
static constexpr int kInputPlanes = 112;

struct InputsOutputs {
  InputsOutputs(int maxBatchSize, bool wdl, bool moves_left, bool conv_policy,
                bool attn_policy) {
    input_masks_mem_.resize(maxBatchSize * kInputPlanes);
    input_val_mem_.resize(maxBatchSize * kInputPlanes);
    op_policy_mem_.resize(maxBatchSize * kNumOutputPolicy);
    op_value_mem_.resize(maxBatchSize * (wdl ? 3 : 1));

    if (moves_left) {
      op_moves_left_mem_.resize(maxBatchSize);
    };

    /**
     * @todo policy map implementation has bug in MPSGraph (GatherND not working
     * in graph). Implementation of policy map to be done in CPU for now.
     *
     * Remove this op_policy_raw_mem_ memory allocation when bug is fixed.
     */
    if (attn_policy) {
      op_policy_raw_mem_.resize(maxBatchSize * (64 * 64 + 8 * 24));
    } else if (conv_policy) {
      op_policy_raw_mem_.resize(maxBatchSize * 73 * 64);
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

}  // namespace Metal
}  // namespace NN
}  // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "network_execution_plan.h"

#include <cstddef>
#include <string>

namespace MetalFish {
namespace NN {

constexpr int kAttentionSquares = 64;

struct SmolgenStagePlan {
  bool present = false;
  bool has_global_positional_weights = false;
  int compressed_channels = 0;
  int dense1_width = 0;
  int dense2_width = 0;
  int dense2_width_per_head = 0;
  int global_position_rows = 0;
  int global_position_cols = 0;
};

struct AttentionStagePlan {
  std::string name;
  int heads = 0;
  int squares = kAttentionSquares;
  int input_width = 0;
  int qkv_width = 0;
  int head_depth = 0;
  int output_width = 0;
  SmolgenStagePlan smolgen;
};

const NetworkResolvedExecutionStep *FindGlobalPositionalEncodingStep(
    const NetworkResolvedExecutionPlan &execution_plan);

AttentionStagePlan ResolveAttentionStagePlan(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, int head_count);

} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network_execution_plan.h"

#include <cstddef>
#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {

constexpr int kCudaAttentionSquares = 64;

struct CudaSmolgenStagePlan {
  bool present = false;
  bool has_global_positional_weights = false;
  int compressed_channels = 0;
  int dense1_width = 0;
  int dense2_width = 0;
  int dense2_width_per_head = 0;
  int global_position_rows = 0;
  int global_position_cols = 0;
};

struct CudaAttentionStagePlan {
  std::string name;
  int heads = 0;
  int squares = kCudaAttentionSquares;
  int input_width = 0;
  int qkv_width = 0;
  int head_depth = 0;
  int output_width = 0;
  CudaSmolgenStagePlan smolgen;
};

const NetworkResolvedExecutionStep *FindCudaGlobalPositionalEncodingStep(
    const NetworkResolvedExecutionPlan &execution_plan);

CudaAttentionStagePlan ResolveCudaAttentionStagePlan(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, int head_count);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

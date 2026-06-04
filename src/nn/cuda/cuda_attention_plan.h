/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network_attention_plan.h"

#include <cstddef>

namespace MetalFish {
namespace NN {
namespace Cuda {

constexpr int kCudaAttentionSquares = kAttentionSquares;

using CudaSmolgenStagePlan = SmolgenStagePlan;
using CudaAttentionStagePlan = AttentionStagePlan;

const NetworkResolvedExecutionStep *FindCudaGlobalPositionalEncodingStep(
    const NetworkResolvedExecutionPlan &execution_plan);

CudaAttentionStagePlan ResolveCudaAttentionStagePlan(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, int head_count);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_attention_plan.h"

namespace MetalFish {
namespace NN {
namespace Cuda {

const NetworkResolvedExecutionStep *FindCudaGlobalPositionalEncodingStep(
    const NetworkResolvedExecutionPlan &execution_plan) {
  return FindGlobalPositionalEncodingStep(execution_plan);
}

CudaAttentionStagePlan ResolveCudaAttentionStagePlan(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, int head_count) {
  return ResolveAttentionStagePlan(execution_plan, attention_step_index,
                                   head_count);
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

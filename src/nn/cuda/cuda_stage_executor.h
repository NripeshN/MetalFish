/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network_execution_plan.h"
#include "cuda_execution_tape.h"
#include "cuda_weight_buffers.h"
#include "cuda_workspace.h"

#include <cuda_runtime_api.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {

struct CudaDenseStageOutput {
  float *dense = nullptr;
  float *activation = nullptr;
  float *normalized = nullptr;
  float *output = nullptr;
  int input_width = 0;
  int output_width = 0;
};

struct CudaDenseStageSequenceOutput {
  CudaDenseStageOutput last;
  std::vector<std::pair<std::string, CudaDenseStageOutput>> stages;
  int stage_count = 0;

  const CudaDenseStageOutput *FindStage(std::string_view name) const;
};

CudaDenseStageOutput ExecuteDenseActivationStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size);

CudaDenseStageOutput ExecuteDenseActivationLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size);

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size);

void CopyDeviceFloatRows(float *dst, int dst_stride, const float *src,
                         int src_stride, int rows, int width,
                         std::string_view name, cudaStream_t stream);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

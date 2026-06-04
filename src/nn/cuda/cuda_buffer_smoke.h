/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "cuda_smoke_status.h"

#include <cstddef>
#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {

struct CudaBufferSmokeResult {
  CudaSmokeStatus status = CudaSmokeStatus::RuntimeError;
  std::string message;
  std::size_t allocation_bytes = 0;
};

CudaBufferSmokeResult RunInferenceBufferSmoke();
CudaBufferSmokeResult RunPackedInputUploadSmokeRaw(const float *input);
CudaBufferSmokeResult RunNullExecutorPipelineSmokeRaw(const float *inputs,
                                                      int batch_size);
CudaBufferSmokeResult RunPlanExecutorPipelineSmoke();
CudaBufferSmokeResult RunAttentionProjectionSmoke();
CudaBufferSmokeResult RunDynamicPositionEncodingStageSmoke();
CudaBufferSmokeResult RunStaticPositionEncodingStageSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

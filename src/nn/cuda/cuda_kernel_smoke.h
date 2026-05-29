/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "cuda_smoke_status.h"

#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {

struct CudaKernelSmokeResult {
  CudaSmokeStatus status = CudaSmokeStatus::RuntimeError;
  std::string message;
};

CudaKernelSmokeResult RunDenseAffineKernelSmoke();
CudaKernelSmokeResult RunConvolutionKernelSmoke();
CudaKernelSmokeResult RunLayerNormKernelSmoke();
CudaKernelSmokeResult RunActivationKernelSmoke();
CudaKernelSmokeResult RunGateKernelSmoke();
CudaKernelSmokeResult RunResidualAddKernelSmoke();
CudaKernelSmokeResult RunSqueezeExciteKernelSmoke();
CudaKernelSmokeResult RunResidualLayerNormKernelSmoke();
CudaKernelSmokeResult RunAttentionCoreKernelSmoke();
CudaKernelSmokeResult RunConvolutionPolicyMapKernelSmoke();
CudaKernelSmokeResult RunAttentionPolicyMapKernelSmoke();
CudaKernelSmokeResult RunDynamicPositionEncodingKernelSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

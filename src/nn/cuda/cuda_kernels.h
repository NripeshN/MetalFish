/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>

#include "cuda_input_packing.h"

namespace MetalFish {
namespace NN {
namespace Cuda {

struct CudaKernelSmokeResult {
  CudaSmokeStatus status = CudaSmokeStatus::RuntimeError;
  std::string message;
};

void LaunchDenseAffineKernel(const float *input, const float *weights,
                             const float *bias, float *output, int batch_size,
                             int input_width, int output_width);

void LaunchLayerNormKernel(const float *input, const float *gamma,
                           const float *beta, float *output, int rows,
                           int width, float epsilon);

CudaKernelSmokeResult RunDenseAffineKernelSmoke();
CudaKernelSmokeResult RunLayerNormKernelSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

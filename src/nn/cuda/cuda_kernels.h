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

enum class CudaActivationKind {
  Relu,
  Relu2,
  Tanh,
  Sigmoid,
  Swish,
  Mish,
  Selu,
};

void LaunchDenseAffineKernel(const float *input, const float *weights,
                             const float *bias, float *output, int batch_size,
                             int input_width, int output_width);

void LaunchLayerNormKernel(const float *input, const float *gamma,
                           const float *beta, float *output, int rows,
                           int width, float epsilon);

void LaunchActivationKernel(const float *input, float *output, int elements,
                            CudaActivationKind kind);

CudaKernelSmokeResult RunDenseAffineKernelSmoke();
CudaKernelSmokeResult RunLayerNormKernelSmoke();
CudaKernelSmokeResult RunActivationKernelSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

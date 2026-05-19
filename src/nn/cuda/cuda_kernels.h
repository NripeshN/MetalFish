/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>

#include <cuda_runtime_api.h>

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

enum class CudaGateKind {
  Multiply,
  Add,
};

void LaunchDenseAffineKernel(const float *input, const float *weights,
                             const float *bias, float *output, int batch_size,
                             int input_width, int output_width,
                             cudaStream_t stream = nullptr);

void LaunchLayerNormKernel(const float *input, const float *gamma,
                           const float *beta, float *output, int rows,
                           int width, float epsilon,
                           cudaStream_t stream = nullptr);

void LaunchActivationKernel(const float *input, float *output, int elements,
                            CudaActivationKind kind,
                            cudaStream_t stream = nullptr);

void LaunchGateKernel(const float *input, const float *weights, float *output,
                      int batch_size, int width, CudaGateKind kind,
                      cudaStream_t stream = nullptr);

void LaunchResidualAddKernel(const float *parent, const float *secondary,
                             float *output, int batch_size, int width,
                             float secondary_scale,
                             cudaStream_t stream = nullptr);

CudaKernelSmokeResult RunDenseAffineKernelSmoke();
CudaKernelSmokeResult RunLayerNormKernelSmoke();
CudaKernelSmokeResult RunActivationKernelSmoke();
CudaKernelSmokeResult RunGateKernelSmoke();
CudaKernelSmokeResult RunResidualAddKernelSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

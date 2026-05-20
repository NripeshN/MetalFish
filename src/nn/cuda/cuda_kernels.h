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

void LaunchBiasActivationKernel(float *input, const float *bias, float *output,
                                int rows, int width, CudaActivationKind kind,
                                cudaStream_t stream = nullptr);

void LaunchGateKernel(const float *input, const float *weights, float *output,
                      int rows, int width, int gate_rows, CudaGateKind kind,
                      cudaStream_t stream = nullptr);

void LaunchResidualAddKernel(const float *parent, const float *secondary,
                             float *output, int batch_size, int width,
                             float secondary_scale,
                             cudaStream_t stream = nullptr);

void LaunchResidualLayerNormKernel(
    const float *parent, const float *secondary, const float *gamma,
    const float *beta, float *residual, float *output, int rows, int width,
    float secondary_scale, float epsilon, cudaStream_t stream = nullptr);

void LaunchAttentionScoreKernel(const float *query, const float *key,
                                float *scores, int batch_size, int heads,
                                int squares, int head_depth, int qkv_width,
                                float scale, cudaStream_t stream = nullptr);

void LaunchAttentionBiasAddKernel(float *scores, const float *bias,
                                  int batch_size, int heads, int squares,
                                  cudaStream_t stream = nullptr);

void LaunchAttentionSoftmaxKernel(const float *scores, float *probabilities,
                                  int rows, int width,
                                  cudaStream_t stream = nullptr);

void LaunchAttentionBiasSoftmaxKernel(float *scores, const float *bias,
                                      float *probabilities, int rows,
                                      int width,
                                      cudaStream_t stream = nullptr);

void LaunchAttentionContextKernel(const float *probabilities,
                                  const float *value, float *context,
                                  int batch_size, int heads, int squares,
                                  int head_depth, int qkv_width,
                                  cudaStream_t stream = nullptr);

void LaunchAttentionPolicyMapKernel(const float *query, const float *key,
                                    const float *promotion_weights,
                                    float *raw_policy, float *policy,
                                    int batch_size, int channels,
                                    cudaStream_t stream = nullptr);

void LaunchExpandPackedInputPlanesKernel(const std::uint64_t *masks,
                                         const float *values, float *expanded,
                                         int batch_size, int planes,
                                         int squares,
                                         cudaStream_t stream = nullptr);

void LaunchDynamicPositionEncodingInputKernel(const float *expanded,
                                              float *position_input,
                                              int batch_size, int input_planes,
                                              int position_planes,
                                              int squares,
                                              cudaStream_t stream = nullptr);

void LaunchDynamicPositionEncodingConcatKernel(
    const float *expanded, const float *position_encoding, float *output,
    int batch_size, int input_planes, int position_width, int squares,
    cudaStream_t stream = nullptr);

CudaKernelSmokeResult RunDenseAffineKernelSmoke();
CudaKernelSmokeResult RunLayerNormKernelSmoke();
CudaKernelSmokeResult RunActivationKernelSmoke();
CudaKernelSmokeResult RunGateKernelSmoke();
CudaKernelSmokeResult RunResidualAddKernelSmoke();
CudaKernelSmokeResult RunAttentionCoreKernelSmoke();
CudaKernelSmokeResult RunDynamicPositionEncodingKernelSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

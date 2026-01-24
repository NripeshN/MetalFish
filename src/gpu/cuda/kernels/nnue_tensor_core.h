/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE Tensor Core Kernels Header

  Interface for tensor core accelerated kernels.
*/

#ifndef NNUE_CUDA_TENSOR_CORE_H
#define NNUE_CUDA_TENSOR_CORE_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

using accumulator_t = int32_t;
using layer_weight_t = int8_t;

#ifdef __cplusplus
extern "C" {
#endif

// Check if tensor cores are available
bool cuda_tensor_cores_available(int device_id);

// Check if INT8 tensor cores are available
bool cuda_int8_tensor_cores_available(int device_id);

// FC layer with FP16 tensor cores
void cuda_fc_layer_tensor_core_fp16(
    const half *input, const half *weights, const half *biases,
    half *output, int batch_size, int input_size, int output_size,
    cudaStream_t stream);

// FC0 layer with tensor cores
void cuda_fc0_layer_tensor_core(
    const accumulator_t *accumulators,
    const half *weights_fp16, const half *biases_fp16,
    int8_t *output_sqr, int8_t *output_linear,
    int hidden_dim, int batch_size, cudaStream_t stream);

// Full NNUE forward pass with tensor cores
void cuda_nnue_forward_tensor_core(
    const accumulator_t *accumulators,
    const half *fc0_weights, const half *fc0_biases,
    const half *fc1_weights, const half *fc1_biases,
    const half *fc2_weights, const half *fc2_biases,
    int32_t *output, int hidden_dim, int batch_size, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // NNUE_CUDA_TENSOR_CORE_H

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE Kernel Declarations

  Header file declaring the CUDA kernel launch functions.
*/

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

// Feature transform kernel
void cuda_feature_transform_full(
    const int16_t* weights,
    const int16_t* biases,
    const int32_t* features,
    const uint32_t* feature_counts,
    const uint32_t* feature_offsets,
    int32_t* accumulators,
    int hidden_dim,
    int batch_size,
    cudaStream_t stream);

// Fused NNUE forward pass
void cuda_nnue_forward_fused(
    const int32_t* accumulators,
    const int8_t* fc0_weights,
    const int32_t* fc0_biases,
    const int8_t* fc1_weights,
    const int32_t* fc1_biases,
    const int8_t* fc2_weights,
    const int32_t* fc2_biases,
    int32_t* output,
    int hidden_dim,
    int batch_size,
    cudaStream_t stream);

// PSQT accumulation
void cuda_psqt_accumulate(
    const int32_t* psqt_weights,
    const int32_t* features,
    const uint32_t* feature_counts,
    const uint32_t* feature_offsets,
    int32_t* psqt_output,
    int batch_size,
    cudaStream_t stream);

// Initialize accumulators with biases
void cuda_init_accumulators(
    const int16_t* biases,
    int32_t* accumulators,
    int hidden_dim,
    int batch_size,
    cudaStream_t stream);

} // extern "C"

#endif // USE_CUDA

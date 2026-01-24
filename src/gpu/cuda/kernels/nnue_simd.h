/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE SIMD Kernels Header

  Interface for warp-optimized CUDA kernels.
*/

#ifndef NNUE_CUDA_SIMD_H
#define NNUE_CUDA_SIMD_H

#include <cuda_runtime.h>
#include <cstdint>

using weight_t = int16_t;
using layer_weight_t = int8_t;
using accumulator_t = int32_t;

#ifdef __cplusplus
extern "C" {
#endif

// Feature transform with warp shuffle optimization
void cuda_feature_transform_simd(
    const weight_t *weights, const weight_t *biases,
    const int32_t *features, const uint32_t *feature_counts,
    accumulator_t *accumulators, int hidden_dim, int batch_size,
    int max_features_per_pos, cudaStream_t stream);

// FC layer with warp reduction
void cuda_fc_layer_simd(
    const int8_t *input, const layer_weight_t *weights,
    const int32_t *biases, int8_t *output,
    int input_size, int output_size, int batch_size, cudaStream_t stream);

// Batch evaluation with cooperative groups
void cuda_batch_evaluate_simd(
    const accumulator_t *accumulators,
    const layer_weight_t *fc0_weights, const int32_t *fc0_biases,
    const layer_weight_t *fc1_weights, const int32_t *fc1_biases,
    const layer_weight_t *fc2_weights, const int32_t *fc2_biases,
    int32_t *output, int hidden_dim, int batch_size, cudaStream_t stream);

// PSQT accumulation with warp reduction
void cuda_psqt_accumulate_simd(
    const int32_t *features, const uint32_t *feature_counts,
    const int32_t *psqt_weights, int32_t *psqt_values,
    int batch_size, int max_features, int num_buckets, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // NNUE_CUDA_SIMD_H

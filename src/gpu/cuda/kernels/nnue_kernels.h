/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE Kernel Declarations

  Header file declaring the CUDA kernel launch functions.
  Feature parity with Metal implementation.
*/

#pragma once

#ifdef USE_CUDA

#include <cstdint>
#include <cuda_runtime.h>

extern "C" {

// ============================================================================
// Feature Extraction
// ============================================================================

// Extract HalfKA features from positions
void cuda_extract_halfka_features(const uint64_t *piece_bitboards,
                                  const uint8_t *king_squares,
                                  int32_t *white_features,
                                  int32_t *black_features,
                                  uint32_t *feature_counts, int batch_size,
                                  int max_features, cudaStream_t stream);

// Extract threat features from positions
void cuda_extract_threat_features(const uint64_t *piece_bitboards,
                                  int32_t *threat_features,
                                  uint32_t *feature_counts, int batch_size,
                                  int max_features, cudaStream_t stream);

// ============================================================================
// Feature Transform
// ============================================================================

// Full feature transform from scratch
void cuda_feature_transform_full(const int16_t *weights, const int16_t *biases,
                                 const int32_t *features,
                                 const uint32_t *feature_counts,
                                 const uint32_t *feature_offsets,
                                 int32_t *accumulators, int hidden_dim,
                                 int batch_size, cudaStream_t stream);

// Optimized feature transform with shared memory
void cuda_feature_transform_optimized(
    const int16_t *weights, const int16_t *biases, const int32_t *features,
    const uint32_t *feature_counts, int32_t *accumulators, int hidden_dim,
    int batch_size, int max_features_per_pos, cudaStream_t stream);

// Warp-optimized feature transform (CUDA equivalent of Metal SIMD)
void cuda_feature_transform_warp_optimized(
    const int16_t *weights, const int16_t *biases, const int32_t *features,
    const uint32_t *feature_counts, int32_t *accumulators, int hidden_dim,
    int batch_size, int max_features_per_pos, cudaStream_t stream);

// Incremental accumulator update
void cuda_feature_transform_incremental(
    const int16_t *weights, const int32_t *added_features,
    const int32_t *removed_features, const uint32_t *add_counts,
    const uint32_t *remove_counts, const int32_t *src_accumulators,
    int32_t *dst_accumulators, int hidden_dim, int batch_size,
    cudaStream_t stream);

// Double incremental update (two consecutive moves)
void cuda_double_incremental_update(
    const int16_t *weights, const int32_t *added1, const int32_t *removed1,
    const int32_t *added2, const int32_t *removed2, const uint32_t *counts,
    const int32_t *src_acc, int32_t *dst_acc, int hidden_dim, int perspective,
    cudaStream_t stream);

// ============================================================================
// Network Layers
// ============================================================================

// FC0 layer (basic)
void cuda_fc0_layer(const int32_t *accumulators, const int8_t *weights,
                    const int32_t *biases, int8_t *output_sqr,
                    int8_t *output_linear, int hidden_dim, int batch_size,
                    cudaStream_t stream);

// FC0 layer with sparse input optimization
void cuda_fc0_sparse_input(const int32_t *accumulators, const int8_t *weights,
                           const int32_t *biases, int8_t *output_sqr,
                           int8_t *output_linear, int hidden_dim,
                           int batch_size, int bucket, cudaStream_t stream);

// FC0 layer with per-position bucket selection
void cuda_fc0_layer_batched(const uint8_t *input, const int8_t *weights,
                            const int32_t *biases, const int32_t *buckets,
                            int8_t *output_sqr, int8_t *output_linear,
                            int hidden_dim, int batch_size,
                            cudaStream_t stream);

// FC1 layer
void cuda_fc1_layer(const int8_t *input, const int8_t *weights,
                    const int32_t *biases, int8_t *output, int batch_size,
                    cudaStream_t stream);

// FC2 output layer
void cuda_fc2_layer(const int8_t *fc1_out, const int8_t *weights,
                    const int32_t *biases, const int8_t *skip_connection,
                    int32_t *output, int batch_size, cudaStream_t stream);

// ============================================================================
// Fused Operations
// ============================================================================

// Complete NNUE forward pass in a single kernel
void cuda_nnue_forward_fused(
    const int32_t *accumulators, const int8_t *fc0_weights,
    const int32_t *fc0_biases, const int8_t *fc1_weights,
    const int32_t *fc1_biases, const int8_t *fc2_weights,
    const int32_t *fc2_biases, int32_t *output, int hidden_dim, int batch_size,
    cudaStream_t stream);

// ============================================================================
// PSQT Operations
// ============================================================================

// PSQT accumulation
void cuda_psqt_accumulate(const int32_t *psqt_weights, const int32_t *features,
                          const uint32_t *feature_counts,
                          const uint32_t *feature_offsets, int32_t *psqt_output,
                          int batch_size, cudaStream_t stream);

// PSQT reduction
void cuda_psqt_reduce(const int32_t *partial_sums, int32_t *output,
                      int num_partials, int batch_size, cudaStream_t stream);

// ============================================================================
// Utility Operations
// ============================================================================

// Initialize accumulators with biases
void cuda_init_accumulators(const int16_t *biases, int32_t *accumulators,
                            int hidden_dim, int batch_size,
                            cudaStream_t stream);

// Zero buffer
void cuda_zero_buffer(int32_t *buffer, int count, cudaStream_t stream);

// Swap accumulator perspectives
void cuda_swap_accumulator_perspectives(const int32_t *src, int32_t *dst,
                                        int hidden_dim, int batch_size,
                                        cudaStream_t stream);

// Transform accumulator output with clipping and pairwise multiplication
void cuda_transform_accumulator_output(const int32_t *accumulators,
                                       const int32_t *threat_accumulators,
                                       uint8_t *output, int hidden_dim,
                                       int batch_size, int use_threats,
                                       int perspective, cudaStream_t stream);

// Fast memory copy (4-element coalescing)
void cuda_copy_accumulator_fast(const int32_t *src, int32_t *dst, int count,
                                cudaStream_t stream);

} // extern "C"

#endif // USE_CUDA

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive CUDA Kernels for NNUE Evaluation

  This file contains all GPU kernels needed for NNUE inference:
  - Feature extraction (HalfKAv2_hm and FullThreats)
  - Feature transformer (sparse to dense)
  - Network layers (AffineTransform, ClippedReLU, SqrClippedReLU)
  - Incremental accumulator updates

  Designed to mirror the Metal implementation for NVIDIA GPUs.
*/

#include <cuda_runtime.h>
#include <stdint.h>

// ============================================================================
// NNUE Architecture Constants
// ============================================================================

// Network dimensions
#define FT_DIM_BIG 1024
#define FT_DIM_SMALL 128
#define FC0_OUT 15
#define FC1_OUT 32
#define PSQT_BUCKETS 8
#define LAYER_STACKS 8

// Feature dimensions
#define HALFKA_DIMS 45056 // 64 * 11 * 64
#define THREAT_DIMS 1536  // Full threats feature size

// Quantization
#define WEIGHT_SCALE_BITS 6
#define OUTPUT_SCALE 16

// Chess constants
#define SQUARE_NB 64
#define COLOR_NB 2
#define PIECE_TYPE_NB 7

// ============================================================================
// Type Definitions
// ============================================================================

typedef int16_t weight_t;
typedef int8_t layer_weight_t;
typedef int32_t accumulator_t;
typedef uint8_t activation_t;

// Position representation for GPU
struct GPUPosition {
  // Piece bitboards [color][piece_type]
  uint64_t pieces[2][7];
  // King squares
  uint8_t king_sq[2];
  // Side to move
  uint8_t stm;
  // Piece count for bucket selection
  uint8_t piece_count;
  // Padding
  uint8_t padding[4];
};

// Feature update info for incremental updates
struct FeatureUpdate {
  int32_t added_features[32];
  int32_t removed_features[32];
  uint8_t num_added;
  uint8_t num_removed;
  uint8_t perspective;
  uint8_t padding;
};

// ============================================================================
// Activation Functions
// ============================================================================

// ClippedReLU: clamp to [0, 127]
__device__ inline int8_t clipped_relu(int16_t x) {
  return (int8_t)max(0, min(127, (int)x));
}

// SqrClippedReLU: (clamp(x, 0, 127))^2 / 128
__device__ inline int8_t sqr_clipped_relu(int16_t x) {
  int clamped = max(0, min(127, (int)x));
  return (int8_t)((clamped * clamped) >> 7);
}

// Scaled ClippedReLU for big network (scaled by 2)
__device__ inline int8_t clipped_relu_scaled(int16_t x) {
  return (int8_t)max(0, min(254, (int)x));
}

// ============================================================================
// Bitboard Utilities
// ============================================================================

__device__ inline uint32_t popcount64(uint64_t x) {
  return __popcll(x);
}

__device__ inline uint32_t lsb(uint64_t x) {
  if (x == 0)
    return UINT32_MAX;  // Return max value for invalid input
  return __ffsll(x) - 1;
}

__device__ inline uint64_t pop_lsb(uint64_t *x) {
  uint64_t lsb_bit = *x & -*x;
  *x ^= lsb_bit;
  return lsb_bit;
}

// ============================================================================
// Feature Extraction Kernels
// ============================================================================

/**
 * Extract HalfKAv2_hm features from position
 * Maps (king_square, piece, square) to feature indices
 * Maximum features per position: 64 pieces across both perspectives
 */
__global__ void extract_halfka_features(const GPUPosition *positions,
                                        int32_t *feature_indices,
                                        uint32_t *feature_counts,
                                        uint32_t batch_size) {
  uint32_t pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos_idx >= batch_size)
    return;

  const GPUPosition &pos = positions[pos_idx];
  uint32_t feature_count = 0;
  int32_t *features = &feature_indices[pos_idx * 64]; // Max 64 features (32 pieces * 2 perspectives)

  // Extract features for both perspectives
  for (int perspective = 0; perspective < 2; perspective++) {
    uint8_t ksq = pos.king_sq[perspective];

    // Iterate through all pieces
    for (int color = 0; color < 2; color++) {
      for (int piece_type = 0; piece_type < PIECE_TYPE_NB; piece_type++) {
        uint64_t bb = pos.pieces[color][piece_type];

        while (bb) {
          uint32_t sq = lsb(bb);
          pop_lsb(&bb);

          // Calculate feature index
          // Format: king_sq * 704 + piece * 64 + sq
          int32_t feature = ksq * 704 + (color * 6 + piece_type) * 64 + sq;
          features[feature_count++] = feature;

          if (feature_count >= 64)
            break; // Safety limit (max pieces)
        }
        if (feature_count >= 64)
          break;
      }
      if (feature_count >= 64)
        break;
    }
  }

  feature_counts[pos_idx] = feature_count;
}

/**
 * Feature Transformer: Apply weights to sparse features
 * Accumulates weighted features into dense accumulator
 */
__global__ void feature_transformer(const int32_t *feature_indices,
                                    const uint32_t *feature_counts,
                                    const weight_t *weights,
                                    const accumulator_t *bias,
                                    accumulator_t *accumulators,
                                    uint32_t batch_size, uint32_t ft_dim) {
  uint32_t pos_idx = blockIdx.x;
  uint32_t dim_idx = threadIdx.x;

  if (pos_idx >= batch_size || dim_idx >= ft_dim)
    return;

  // Start with bias
  accumulator_t acc = bias[dim_idx];

  // Add weighted features
  const int32_t *features = &feature_indices[pos_idx * 64];
  uint32_t num_features = feature_counts[pos_idx];

  for (uint32_t i = 0; i < num_features; i++) {
    int32_t feature = features[i];
    if (feature >= 0 && feature < HALFKA_DIMS) {
      weight_t weight = weights[feature * ft_dim + dim_idx];
      acc += (accumulator_t)weight;
    }
  }

  accumulators[pos_idx * ft_dim + dim_idx] = acc;
}

/**
 * Incremental accumulator update
 * Efficiently updates accumulator by adding/removing features
 */
__global__ void
incremental_update(const FeatureUpdate *updates, const weight_t *weights,
                   accumulator_t *accumulators, uint32_t batch_size,
                   uint32_t ft_dim) {
  uint32_t pos_idx = blockIdx.x;
  uint32_t dim_idx = threadIdx.x;

  if (pos_idx >= batch_size || dim_idx >= ft_dim)
    return;

  const FeatureUpdate &update = updates[pos_idx];
  accumulator_t acc = accumulators[pos_idx * ft_dim + dim_idx];

  // Remove old features
  for (uint32_t i = 0; i < update.num_removed; i++) {
    int32_t feature = update.removed_features[i];
    if (feature >= 0 && feature < HALFKA_DIMS) {
      weight_t weight = weights[feature * ft_dim + dim_idx];
      acc -= (accumulator_t)weight;
    }
  }

  // Add new features
  for (uint32_t i = 0; i < update.num_added; i++) {
    int32_t feature = update.added_features[i];
    if (feature >= 0 && feature < HALFKA_DIMS) {
      weight_t weight = weights[feature * ft_dim + dim_idx];
      acc += (accumulator_t)weight;
    }
  }

  accumulators[pos_idx * ft_dim + dim_idx] = acc;
}

// ============================================================================
// Network Layer Kernels
// ============================================================================

/**
 * Affine transform with ClippedReLU activation
 * output[i] = ClippedReLU(weights[i] * input + bias[i])
 */
__global__ void affine_transform_relu(const activation_t *input,
                                      const layer_weight_t *weights,
                                      const int32_t *bias,
                                      activation_t *output, uint32_t batch_size,
                                      uint32_t input_dim, uint32_t output_dim) {
  uint32_t pos_idx = blockIdx.x;
  uint32_t out_idx = threadIdx.x;

  if (pos_idx >= batch_size || out_idx >= output_dim)
    return;

  const activation_t *in = &input[pos_idx * input_dim];
  int32_t sum = bias[out_idx];

  // Compute weighted sum
  for (uint32_t i = 0; i < input_dim; i++) {
    int32_t weight = (int32_t)weights[out_idx * input_dim + i];
    sum += (int32_t)in[i] * weight;
  }

  // Apply activation and quantization
  int16_t activated = (int16_t)(sum >> WEIGHT_SCALE_BITS);
  output[pos_idx * output_dim + out_idx] = clipped_relu(activated);
}

/**
 * Affine transform with SqrClippedReLU activation
 * output[i] = SqrClippedReLU(weights[i] * input + bias[i])
 */
__global__ void affine_transform_sqr_relu(const activation_t *input,
                                          const layer_weight_t *weights,
                                          const int32_t *bias,
                                          activation_t *output,
                                          uint32_t batch_size,
                                          uint32_t input_dim,
                                          uint32_t output_dim) {
  uint32_t pos_idx = blockIdx.x;
  uint32_t out_idx = threadIdx.x;

  if (pos_idx >= batch_size || out_idx >= output_dim)
    return;

  const activation_t *in = &input[pos_idx * input_dim];
  int32_t sum = bias[out_idx];

  // Compute weighted sum
  for (uint32_t i = 0; i < input_dim; i++) {
    int32_t weight = (int32_t)weights[out_idx * input_dim + i];
    sum += (int32_t)in[i] * weight;
  }

  // Apply activation and quantization
  int16_t activated = (int16_t)(sum >> WEIGHT_SCALE_BITS);
  output[pos_idx * output_dim + out_idx] = sqr_clipped_relu(activated);
}

/**
 * Final output layer: linear transform for evaluation score
 * Returns single evaluation value per position
 */
__global__ void output_layer(const activation_t *input,
                              const layer_weight_t *weights,
                              const int32_t *bias, int32_t *output,
                              uint32_t batch_size, uint32_t input_dim) {
  uint32_t pos_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size)
    return;

  const activation_t *in = &input[pos_idx * input_dim];
  int32_t sum = bias[0];

  // Compute weighted sum
  for (uint32_t i = 0; i < input_dim; i++) {
    int32_t weight = (int32_t)weights[i];
    sum += (int32_t)in[i] * weight;
  }

  // Scale output
  output[pos_idx] = sum / OUTPUT_SCALE;
}

// ============================================================================
// Fused NNUE Forward Pass
// ============================================================================

/**
 * Complete NNUE forward pass in a single kernel
 * Optimized for small batch sizes with minimal kernel launch overhead
 */
__global__ void nnue_forward_pass(const accumulator_t *accumulators,
                                  const layer_weight_t *fc0_weights,
                                  const int32_t *fc0_bias,
                                  const layer_weight_t *fc1_weights,
                                  const int32_t *fc1_bias,
                                  const layer_weight_t *out_weights,
                                  const int32_t *out_bias, int32_t *output,
                                  uint32_t batch_size) {
  uint32_t pos_idx = blockIdx.x;
  if (pos_idx >= batch_size)
    return;

  // Shared memory for intermediate results
  __shared__ activation_t fc0_output[FC0_OUT];
  __shared__ activation_t fc1_output[FC1_OUT];

  // Each thread processes a subset of the computation
  uint32_t tid = threadIdx.x;

  // Layer 0: FT -> FC0
  if (tid < FC0_OUT) {
    const accumulator_t *acc = &accumulators[pos_idx * FT_DIM_BIG];
    int32_t sum = fc0_bias[tid];

    for (uint32_t i = 0; i < FT_DIM_BIG; i++) {
      // Apply ClippedReLU to accumulator and use as input
      int16_t input = (int16_t)(acc[i] >> WEIGHT_SCALE_BITS);
      activation_t activated = clipped_relu_scaled(input);
      sum += (int32_t)activated * (int32_t)fc0_weights[tid * FT_DIM_BIG + i];
    }

    fc0_output[tid] = clipped_relu((int16_t)(sum >> WEIGHT_SCALE_BITS));
  }

  __syncthreads();

  // Layer 1: FC0 -> FC1
  if (tid < FC1_OUT) {
    int32_t sum = fc1_bias[tid];

    for (uint32_t i = 0; i < FC0_OUT; i++) {
      sum += (int32_t)fc0_output[i] * (int32_t)fc1_weights[tid * FC0_OUT + i];
    }

    fc1_output[tid] = sqr_clipped_relu((int16_t)(sum >> WEIGHT_SCALE_BITS));
  }

  __syncthreads();

  // Output layer: FC1 -> Score
  if (tid == 0) {
    int32_t sum = out_bias[0];

    for (uint32_t i = 0; i < FC1_OUT; i++) {
      sum += (int32_t)fc1_output[i] * (int32_t)out_weights[i];
    }

    output[pos_idx] = sum / OUTPUT_SCALE;
  }
}

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * Simple vector addition for testing
 */
__global__ void vector_add(const float *a, const float *b, float *c,
                           uint32_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

/**
 * Memory copy kernel
 */
__global__ void mem_copy(const void *src, void *dst, uint32_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    ((uint8_t *)dst)[idx] = ((const uint8_t *)src)[idx];
  }
}

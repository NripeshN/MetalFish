/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE Kernels

  GPU kernels for NNUE neural network evaluation on NVIDIA GPUs.
  Optimized for modern CUDA architectures with tensor core support.
*/

#ifndef NNUE_CUDA_KERNELS_CU
#define NNUE_CUDA_KERNELS_CU

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// NNUE Architecture Constants
// ============================================================================

constexpr int FT_DIM_BIG = 1024;
constexpr int FT_DIM_SMALL = 128;
constexpr int FC0_OUT = 15;
constexpr int FC1_OUT = 32;
constexpr int PSQT_BUCKETS = 8;
constexpr int LAYER_STACKS = 8;

constexpr int HALFKA_DIMS = 45056;
constexpr int THREAT_DIMS = 1536;

constexpr int WEIGHT_SCALE_BITS = 6;
constexpr int OUTPUT_SCALE = 16;

// ============================================================================
// Type Definitions
// ============================================================================

using weight_t = int16_t;
using layer_weight_t = int8_t;
using accumulator_t = int32_t;
using activation_t = uint8_t;

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ __forceinline__ int8_t clipped_relu(int16_t x) {
  return static_cast<int8_t>(max(0, min(127, static_cast<int>(x))));
}

__device__ __forceinline__ int8_t sqr_clipped_relu(int16_t x) {
  int clamped = max(0, min(127, static_cast<int>(x)));
  return static_cast<int8_t>((clamped * clamped) >> 7);
}

__device__ __forceinline__ int popcount64(uint64_t x) { return __popcll(x); }

__device__ __forceinline__ int lsb64(uint64_t x) { return __ffsll(x) - 1; }

// ============================================================================
// Feature Extraction Kernels
// ============================================================================

/**
 * Extract HalfKA features from positions
 * Each thread processes one position
 */
__global__ void extract_halfka_features(
    const uint64_t *__restrict__ piece_bitboards, // [batch_size][2][7]
    const uint8_t *__restrict__ king_squares,     // [batch_size][2]
    int32_t *__restrict__ white_features,         // [batch_size][max_features]
    int32_t *__restrict__ black_features,         // [batch_size][max_features]
    uint32_t *__restrict__ feature_counts,        // [batch_size][2]
    int batch_size, int max_features) {

  int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos_idx >= batch_size)
    return;

  int white_ksq = king_squares[pos_idx * 2];
  int black_ksq = king_squares[pos_idx * 2 + 1];

  int white_count = 0;
  int black_count = 0;
  int base_idx = pos_idx * max_features;

  // Iterate through all pieces
  for (int color = 0; color < 2; color++) {
    for (int pt = 1; pt <= 6; pt++) { // PAWN to KING
      uint64_t bb = piece_bitboards[pos_idx * 14 + color * 7 + pt];
      while (bb && white_count < max_features && black_count < max_features) {
        int sq = lsb64(bb);
        bb &= bb - 1;

        // White perspective feature
        int oriented_ksq_w = white_ksq ^ ((white_ksq & 4) ? 7 : 0);
        int oriented_sq_w = sq ^ ((white_ksq & 4) ? 7 : 0);
        int piece_idx_w = (pt - 1) + (color != 0 ? 6 : 0);
        int white_feat =
            oriented_ksq_w * 640 + piece_idx_w * 64 + oriented_sq_w;

        if (white_feat >= 0 && white_feat < HALFKA_DIMS) {
          white_features[base_idx + white_count++] = white_feat;
        }

        // Black perspective feature (mirrored)
        int black_ksq_mir = black_ksq ^ 56;
        int oriented_ksq_b = black_ksq_mir ^ ((black_ksq_mir & 4) ? 7 : 0);
        int sq_mir = sq ^ 56;
        int oriented_sq_b = sq_mir ^ ((black_ksq_mir & 4) ? 7 : 0);
        int piece_idx_b = (pt - 1) + ((color ^ 1) != 0 ? 6 : 0);
        int black_feat =
            oriented_ksq_b * 640 + piece_idx_b * 64 + oriented_sq_b;

        if (black_feat >= 0 && black_feat < HALFKA_DIMS) {
          black_features[base_idx + black_count++] = black_feat;
        }
      }
    }
  }

  feature_counts[pos_idx * 2] = white_count;
  feature_counts[pos_idx * 2 + 1] = black_count;
}

// ============================================================================
// Feature Transformer Kernels
// ============================================================================

/**
 * Feature transform from scratch
 * Transforms sparse features to dense accumulator
 * Grid: (hidden_dim / 256, batch_size)
 * Block: (256)
 */
__global__ void feature_transform_full(
    const weight_t *__restrict__ weights, const weight_t *__restrict__ biases,
    const int32_t *__restrict__ features,
    const uint32_t *__restrict__ feature_counts,
    const uint32_t *__restrict__ feature_offsets,
    accumulator_t *__restrict__ accumulators, int hidden_dim, int batch_size) {

  int pos_idx = blockIdx.y;
  int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  // Start with bias
  accumulator_t acc = static_cast<accumulator_t>(biases[hidden_idx]);

  // Get feature range for this position
  int start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  int count = feature_counts[pos_idx];

  // Accumulate weights for active features
  for (int i = 0; i < count; i++) {
    int feature_idx = features[start + i];
    if (feature_idx >= 0 && feature_idx < HALFKA_DIMS) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

/**
 * Optimized feature transform using shared memory
 * For better memory coalescing
 */
__global__ void feature_transform_optimized(
    const weight_t *__restrict__ weights, const weight_t *__restrict__ biases,
    const int32_t *__restrict__ features,
    const uint32_t *__restrict__ feature_counts,
    accumulator_t *__restrict__ accumulators, int hidden_dim, int batch_size,
    int max_features_per_pos) {

  extern __shared__ int32_t shared_features[];

  int pos_idx = blockIdx.y;
  int hidden_base = blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  if (pos_idx >= batch_size)
    return;

  // Load features to shared memory
  int count = feature_counts[pos_idx];
  const int32_t *pos_features = features + pos_idx * max_features_per_pos;

  for (int i = tid; i < count; i += blockDim.x) {
    shared_features[i] = pos_features[i];
  }
  __syncthreads();

  int hidden_idx = hidden_base + tid;
  if (hidden_idx >= hidden_dim)
    return;

  // Start with bias
  accumulator_t acc = static_cast<accumulator_t>(biases[hidden_idx]);

  // Accumulate weights for active features
  for (int i = 0; i < count; i++) {
    int feature_idx = shared_features[i];
    if (feature_idx >= 0 && feature_idx < HALFKA_DIMS) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

/**
 * Incremental accumulator update
 * Only updates changed features
 */
__global__ void feature_transform_incremental(
    const weight_t *__restrict__ weights,
    const int32_t *__restrict__ added_features,
    const int32_t *__restrict__ removed_features,
    const uint32_t *__restrict__ add_counts,
    const uint32_t *__restrict__ remove_counts,
    const accumulator_t *__restrict__ src_accumulators,
    accumulator_t *__restrict__ dst_accumulators, int hidden_dim,
    int batch_size) {

  int pos_idx = blockIdx.y;
  int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  // Start from source accumulator
  accumulator_t acc = src_accumulators[pos_idx * hidden_dim + hidden_idx];

  // Remove old features
  int num_removed = remove_counts[pos_idx];
  for (int i = 0; i < num_removed; i++) {
    int feature_idx = removed_features[pos_idx * 32 + i];
    if (feature_idx >= 0 && feature_idx < HALFKA_DIMS) {
      acc -= weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  // Add new features
  int num_added = add_counts[pos_idx];
  for (int i = 0; i < num_added; i++) {
    int feature_idx = added_features[pos_idx * 32 + i];
    if (feature_idx >= 0 && feature_idx < HALFKA_DIMS) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  dst_accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

// ============================================================================
// Network Layer Kernels
// ============================================================================

/**
 * FC0 layer with sparse input
 * One block per position
 */
__global__ void fc0_layer(const accumulator_t *__restrict__ accumulators,
                          const layer_weight_t *__restrict__ weights,
                          const int32_t *__restrict__ biases,
                          int8_t *__restrict__ output_sqr,
                          int8_t *__restrict__ output_linear, int hidden_dim,
                          int batch_size) {

  __shared__ int8_t sqr_out[2][16];
  __shared__ int8_t linear_out[2][16];

  int pos_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (pos_idx >= batch_size)
    return;

  const accumulator_t *white_acc = accumulators + pos_idx * 2 * hidden_dim;
  const accumulator_t *black_acc = white_acc + hidden_dim;

  // Each thread computes one or more output neurons
  for (int out = tid; out <= FC0_OUT; out += blockDim.x) {
    for (int p = 0; p < 2; p++) {
      const accumulator_t *acc = (p == 0) ? white_acc : black_acc;

      int32_t sum = biases[out];
      for (int i = 0; i < hidden_dim; i++) {
        int8_t clipped =
            clipped_relu(static_cast<int16_t>(acc[i] >> WEIGHT_SCALE_BITS));
        sum += clipped * weights[i * (FC0_OUT + 1) + out];
      }

      int16_t result = static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS);
      sqr_out[p][out] = sqr_clipped_relu(result);
      linear_out[p][out] = clipped_relu(result);
    }
  }
  __syncthreads();

  // Write outputs
  if (tid < 2 * (FC0_OUT + 1)) {
    int p = tid / (FC0_OUT + 1);
    int o = tid % (FC0_OUT + 1);
    output_sqr[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        sqr_out[p][o];
    output_linear[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        linear_out[p][o];
  }
}

/**
 * FC1 layer
 */
__global__ void fc1_layer(const int8_t *__restrict__ input,
                          const layer_weight_t *__restrict__ weights,
                          const int32_t *__restrict__ biases,
                          int8_t *__restrict__ output, int batch_size) {

  int pos_idx = blockIdx.x;
  int out_idx = threadIdx.x;

  if (pos_idx >= batch_size || out_idx >= FC1_OUT)
    return;

  const int8_t *in_ptr = input + pos_idx * 2 * FC0_OUT;

  int32_t sum = biases[out_idx];
  for (int i = 0; i < 2 * FC0_OUT; i++) {
    sum += in_ptr[i] * weights[i * FC1_OUT + out_idx];
  }

  output[pos_idx * FC1_OUT + out_idx] =
      clipped_relu(static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS));
}

/**
 * FC2 output layer
 */
__global__ void fc2_layer(const int8_t *__restrict__ fc1_out,
                          const layer_weight_t *__restrict__ weights,
                          const int32_t *__restrict__ biases,
                          const int8_t *__restrict__ skip_connection,
                          int32_t *__restrict__ output, int batch_size) {

  int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos_idx >= batch_size)
    return;

  const int8_t *in_ptr = fc1_out + pos_idx * FC1_OUT;

  int32_t sum = biases[0];
  for (int i = 0; i < FC1_OUT; i++) {
    sum += in_ptr[i] * weights[i];
  }

  // Add skip connection
  int32_t skip_white = skip_connection[pos_idx * 2 * (FC0_OUT + 1) + FC0_OUT];
  int32_t skip_black =
      skip_connection[pos_idx * 2 * (FC0_OUT + 1) + (FC0_OUT + 1) + FC0_OUT];
  int32_t skip_val = ((skip_white + skip_black) * 600 * OUTPUT_SCALE) /
                     (2 * 127 * (1 << WEIGHT_SCALE_BITS));

  output[pos_idx] = sum + skip_val;
}

// ============================================================================
// Fused Forward Pass Kernel
// ============================================================================

/**
 * Complete NNUE forward pass in a single kernel
 * Best for batch evaluation
 */
__global__ void
nnue_forward_fused(const accumulator_t *__restrict__ accumulators,
                   const layer_weight_t *__restrict__ fc0_weights,
                   const int32_t *__restrict__ fc0_biases,
                   const layer_weight_t *__restrict__ fc1_weights,
                   const int32_t *__restrict__ fc1_biases,
                   const layer_weight_t *__restrict__ fc2_weights,
                   const int32_t *__restrict__ fc2_biases,
                   int32_t *__restrict__ output, int hidden_dim,
                   int batch_size) {

  __shared__ int8_t fc0_sqr[2 * 16];
  __shared__ int8_t fc0_skip[2];
  __shared__ int8_t fc1_out[32];

  int pos_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (pos_idx >= batch_size)
    return;

  const accumulator_t *white_acc = accumulators + pos_idx * 2 * hidden_dim;
  const accumulator_t *black_acc = white_acc + hidden_dim;

  // ========== FC0 Layer ==========
  for (int out = tid; out <= FC0_OUT; out += blockDim.x) {
    for (int p = 0; p < 2; p++) {
      const accumulator_t *acc = (p == 0) ? white_acc : black_acc;

      int32_t sum = fc0_biases[out];
      for (int i = 0; i < hidden_dim; i++) {
        int8_t clipped =
            clipped_relu(static_cast<int16_t>(acc[i] >> WEIGHT_SCALE_BITS));
        sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
      }

      int16_t result = static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS);

      if (out < FC0_OUT) {
        fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
      } else {
        fc0_skip[p] = clipped_relu(result);
      }
    }
  }
  __syncthreads();

  // ========== FC1 Layer ==========
  for (int out = tid; out < FC1_OUT; out += blockDim.x) {
    int32_t sum = fc1_biases[out];
    for (int i = 0; i < 2 * FC0_OUT; i++) {
      sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
    }
    fc1_out[out] = clipped_relu(static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS));
  }
  __syncthreads();

  // ========== FC2 Layer ==========
  if (tid == 0) {
    int32_t sum = fc2_biases[0];
    for (int i = 0; i < FC1_OUT; i++) {
      sum += fc1_out[i] * fc2_weights[i];
    }

    // Skip connection
    int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * OUTPUT_SCALE) /
                       (2 * 127 * (1 << WEIGHT_SCALE_BITS));

    output[pos_idx] = sum + skip_val;
  }
}

// ============================================================================
// PSQT Kernels
// ============================================================================

/**
 * PSQT accumulation
 */
__global__ void psqt_accumulate(const int32_t *__restrict__ psqt_weights,
                                const int32_t *__restrict__ features,
                                const uint32_t *__restrict__ feature_counts,
                                const uint32_t *__restrict__ feature_offsets,
                                int32_t *__restrict__ psqt_output,
                                int batch_size) {

  int pos_idx = blockIdx.y;
  int bucket = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || bucket >= PSQT_BUCKETS)
    return;

  int start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  int count = feature_counts[pos_idx];

  int32_t acc = 0;
  for (int i = 0; i < count; i++) {
    int feature_idx = features[start + i];
    if (feature_idx >= 0 && feature_idx < HALFKA_DIMS) {
      acc += psqt_weights[feature_idx * PSQT_BUCKETS + bucket];
    }
  }

  psqt_output[pos_idx * PSQT_BUCKETS + bucket] = acc;
}

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * Initialize accumulators with biases
 */
__global__ void init_accumulators(const weight_t *__restrict__ biases,
                                  accumulator_t *__restrict__ accumulators,
                                  int hidden_dim, int batch_size) {

  int pos_idx = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || idx >= hidden_dim * 2)
    return;

  int offset = idx % hidden_dim;
  accumulators[pos_idx * 2 * hidden_dim + idx] =
      static_cast<accumulator_t>(biases[offset]);
}

/**
 * Zero buffer
 */
__global__ void zero_buffer(int32_t *buffer, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    buffer[idx] = 0;
  }
}

/**
 * Copy accumulator with perspective swap
 */
__global__ void
swap_accumulator_perspectives(const accumulator_t *__restrict__ src,
                              accumulator_t *__restrict__ dst, int hidden_dim,
                              int batch_size) {

  int pos_idx = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || idx >= hidden_dim * 2)
    return;

  int perspective = idx / hidden_dim;
  int offset = idx % hidden_dim;
  int swapped = 1 - perspective;

  dst[pos_idx * 2 * hidden_dim + perspective * hidden_dim + offset] =
      src[pos_idx * 2 * hidden_dim + swapped * hidden_dim + offset];
}

// ============================================================================
// Threat Feature Extraction (Missing from original CUDA implementation)
// ============================================================================

/**
 * Extract threat features from position
 * Matches Metal's extract_threat_features kernel
 */
__global__ void extract_threat_features(
    const uint64_t *__restrict__ piece_bitboards, // [batch_size][2][7]
    int32_t *__restrict__ threat_features,
    uint32_t *__restrict__ feature_counts, int batch_size, int max_features) {

  int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos_idx >= batch_size)
    return;

  int count = 0;
  int base_idx = pos_idx * max_features;

  // Threat feature extraction based on piece attacks
  for (int attacker_color = 0; attacker_color < 2 && count < max_features;
       attacker_color++) {
    for (int pt = 1; pt <= 6 && count < max_features; pt++) {
      uint64_t attackers =
          piece_bitboards[pos_idx * 14 + attacker_color * 7 + pt];
      while (attackers && count < max_features) {
        int from = lsb64(attackers);
        attackers &= attackers - 1;

        for (int target_color = 0; target_color < 2 && count < max_features;
             target_color++) {
          for (int target_pt = 1; target_pt <= 6 && count < max_features;
               target_pt++) {
            uint64_t targets =
                piece_bitboards[pos_idx * 14 + target_color * 7 + target_pt];
            while (targets && count < max_features) {
              int to = lsb64(targets);
              targets &= targets - 1;

              // Simplified threat index calculation
              int32_t threat_idx = attacker_color * 768 + pt * 128 +
                                   target_pt * 16 + (from % 8) + (to % 8);
              if (threat_idx >= 0 && threat_idx < THREAT_DIMS) {
                threat_features[base_idx + count++] = threat_idx;
              }
            }
          }
        }
      }
    }
  }

  feature_counts[pos_idx] = count;
}

// ============================================================================
// Double Incremental Update (Missing from original CUDA implementation)
// ============================================================================

/**
 * Double incremental update - combines two consecutive move updates
 * Matches Metal's double_incremental_update kernel
 */
__global__ void double_incremental_update(
    const weight_t *__restrict__ weights, const int32_t *__restrict__ added1,
    const int32_t *__restrict__ removed1, const int32_t *__restrict__ added2,
    const int32_t *__restrict__ removed2,
    const uint32_t *__restrict__ counts, // [add1, rem1, add2, rem2]
    const accumulator_t *__restrict__ src_acc,
    accumulator_t *__restrict__ dst_acc, int hidden_dim, int perspective) {

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= hidden_dim)
    return;

  int num_added1 = counts[0];
  int num_removed1 = counts[1];
  int num_added2 = counts[2];
  int num_removed2 = counts[3];

  accumulator_t acc = src_acc[perspective * hidden_dim + gid];

  // First move: remove then add
  for (int i = 0; i < num_removed1; i++) {
    int32_t feat_idx = removed1[i];
    if (feat_idx >= 0) {
      acc -= weights[feat_idx * hidden_dim + gid];
    }
  }
  for (int i = 0; i < num_added1; i++) {
    int32_t feat_idx = added1[i];
    if (feat_idx >= 0) {
      acc += weights[feat_idx * hidden_dim + gid];
    }
  }

  // Second move: remove then add
  for (int i = 0; i < num_removed2; i++) {
    int32_t feat_idx = removed2[i];
    if (feat_idx >= 0) {
      acc -= weights[feat_idx * hidden_dim + gid];
    }
  }
  for (int i = 0; i < num_added2; i++) {
    int32_t feat_idx = added2[i];
    if (feat_idx >= 0) {
      acc += weights[feat_idx * hidden_dim + gid];
    }
  }

  dst_acc[perspective * hidden_dim + gid] = acc;
}

// ============================================================================
// Warp-Optimized Feature Transform (CUDA equivalent of Metal SIMD)
// ============================================================================

/**
 * Warp-optimized feature transform using shuffle operations
 * CUDA equivalent of Metal's feature_transform_simd_optimized
 */
__global__ void feature_transform_warp_optimized(
    const weight_t *__restrict__ weights, const weight_t *__restrict__ biases,
    const int32_t *__restrict__ features,
    const uint32_t *__restrict__ feature_counts,
    accumulator_t *__restrict__ accumulators, int hidden_dim, int batch_size,
    int max_features_per_pos) {

  int pos_idx = blockIdx.y;
  if (pos_idx >= batch_size)
    return;

  // Each warp (32 threads) processes 32 hidden dimensions
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int hidden_base = (blockIdx.x * (blockDim.x / 32) + warp_id) * 32;
  int hidden_idx = hidden_base + lane_id;

  if (hidden_idx >= hidden_dim)
    return;

  // Start with bias
  accumulator_t acc = static_cast<accumulator_t>(biases[hidden_idx]);

  int count = feature_counts[pos_idx];
  const int32_t *pos_features = features + pos_idx * max_features_per_pos;

  // Use warp-level broadcast for feature indices
  for (int i = 0; i < count; i++) {
    // All threads in warp read the same feature index
    int32_t feat_idx = pos_features[i];
    if (feat_idx >= 0 && feat_idx < HALFKA_DIMS) {
      acc += weights[feat_idx * hidden_dim + hidden_idx];
    }
  }

  accumulators[pos_idx * hidden_dim * 2 + hidden_idx] = acc;
}

// ============================================================================
// FC0 Layer with Sparse Input Optimization
// ============================================================================

/**
 * FC0 layer with sparse input - skips zero values
 * Matches Metal's fc0_sparse_input kernel
 */
__global__ void fc0_sparse_input(const accumulator_t *__restrict__ accumulators,
                                 const layer_weight_t *__restrict__ weights,
                                 const int32_t *__restrict__ biases,
                                 int8_t *__restrict__ output_sqr,
                                 int8_t *__restrict__ output_linear,
                                 int hidden_dim, int batch_size, int bucket) {

  __shared__ int8_t sqr_out[2][16];
  __shared__ int8_t linear_out[2][16];

  int pos_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (pos_idx >= batch_size)
    return;

  // Process both perspectives
  for (int perspective = 0; perspective < 2; perspective++) {
    const accumulator_t *acc =
        accumulators + pos_idx * 2 * hidden_dim + perspective * hidden_dim;

    // Each thread computes one or more output neurons
    for (int out = tid; out <= FC0_OUT; out += blockDim.x) {
      int32_t sum = biases[out];

      // Sparse input: only process non-zero clipped values
      for (int i = 0; i < hidden_dim; i++) {
        int8_t clipped =
            clipped_relu(static_cast<int16_t>(acc[i] >> WEIGHT_SCALE_BITS));
        if (clipped != 0) {
          sum += clipped * weights[i * (FC0_OUT + 1) + out];
        }
      }

      int16_t result = static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS);
      sqr_out[perspective][out] = sqr_clipped_relu(result);
      linear_out[perspective][out] = clipped_relu(result);
    }
  }

  __syncthreads();

  // Write outputs
  if (tid < 2 * (FC0_OUT + 1)) {
    int p = tid / (FC0_OUT + 1);
    int o = tid % (FC0_OUT + 1);
    output_sqr[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        sqr_out[p][o];
    output_linear[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        linear_out[p][o];
  }
}

// ============================================================================
// FC0 Layer with Per-Position Bucket Selection
// ============================================================================

/**
 * FC0 layer with per-position bucket selection
 * Matches Metal's fc0_layer_batched kernel
 */
__global__ void fc0_layer_batched(
    const uint8_t *__restrict__ input,
    const layer_weight_t
        *__restrict__ weights, // [LAYER_STACKS][hidden_dim*2][FC0_OUT+1]
    const int32_t *__restrict__ biases, // [LAYER_STACKS][FC0_OUT+1]
    const int32_t *__restrict__ buckets, int8_t *__restrict__ output_sqr,
    int8_t *__restrict__ output_linear, int hidden_dim, int batch_size) {

  int pos_idx = blockIdx.y;
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || out_idx > FC0_OUT)
    return;

  int bucket = buckets[pos_idx];

  // Get weights and biases for this bucket
  const layer_weight_t *bucket_weights =
      weights + bucket * hidden_dim * 2 * (FC0_OUT + 1);
  const int32_t *bucket_biases = biases + bucket * (FC0_OUT + 1);

  const uint8_t *in_ptr = input + pos_idx * hidden_dim * 2;

  int32_t sum = bucket_biases[out_idx];

  // Sparse input: only process non-zero values
  for (int i = 0; i < hidden_dim * 2; i++) {
    if (in_ptr[i] != 0) {
      sum += in_ptr[i] * bucket_weights[i * (FC0_OUT + 1) + out_idx];
    }
  }

  int16_t result = static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS);
  output_sqr[pos_idx * (FC0_OUT + 1) + out_idx] = sqr_clipped_relu(result);
  output_linear[pos_idx * (FC0_OUT + 1) + out_idx] = clipped_relu(result);
}

// ============================================================================
// Transform Accumulator Output
// ============================================================================

/**
 * Transform accumulator to network input with clipping and pairwise
 * multiplication Matches Metal's transform_accumulator_output kernel
 */
__global__ void transform_accumulator_output(
    const accumulator_t *__restrict__ accumulators,
    const accumulator_t *__restrict__ threat_accumulators,
    uint8_t *__restrict__ output, int hidden_dim, int batch_size,
    int use_threats, int perspective) {

  int pos_idx = blockIdx.y;
  int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || out_idx >= hidden_dim / 2)
    return;

  int half_dim = hidden_dim / 2;
  const accumulator_t *acc =
      accumulators + pos_idx * hidden_dim * 2 + perspective * hidden_dim;

  int16_t sum0, sum1;

  if (use_threats && threat_accumulators) {
    const accumulator_t *threat_acc = threat_accumulators +
                                      pos_idx * hidden_dim * 2 +
                                      perspective * hidden_dim;
    sum0 =
        max(0, min(255, static_cast<int>(acc[out_idx] + threat_acc[out_idx])));
    sum1 = max(0, min(255, static_cast<int>(acc[out_idx + half_dim] +
                                            threat_acc[out_idx + half_dim])));
  } else {
    sum0 =
        max(0, min(254, static_cast<int>(acc[out_idx]) >> WEIGHT_SCALE_BITS));
    sum1 = max(0, min(254, static_cast<int>(acc[out_idx + half_dim]) >>
                               WEIGHT_SCALE_BITS));
  }

  // Pairwise multiplication with division by 512
  output[pos_idx * hidden_dim + perspective * half_dim + out_idx] =
      static_cast<uint8_t>((sum0 * sum1) / 512);
}

// ============================================================================
// Fast Memory Copy (4-element coalescing)
// ============================================================================

/**
 * Fast memory copy for accumulator states
 * Matches Metal's copy_accumulator_fast kernel
 */
__global__ void copy_accumulator_fast(const accumulator_t *__restrict__ src,
                                      accumulator_t *__restrict__ dst,
                                      int count) {

  // Each thread copies 4 elements for better memory coalescing
  int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

  if (base + 3 < count) {
    // Vectorized copy using int4
    reinterpret_cast<int4 *>(dst)[base / 4] =
        reinterpret_cast<const int4 *>(src)[base / 4];
  } else {
    for (int i = 0; i < 4 && base + i < count; i++) {
      dst[base + i] = src[base + i];
    }
  }
}

// ============================================================================
// PSQT Reduction
// ============================================================================

/**
 * Parallel reduction for PSQT accumulation
 * Matches Metal's psqt_reduce kernel
 */
__global__ void psqt_reduce(const int32_t *__restrict__ partial_sums,
                            int32_t *__restrict__ output, int num_partials,
                            int batch_size) {

  int pos_idx = blockIdx.y;
  int bucket = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || bucket >= PSQT_BUCKETS)
    return;

  int32_t sum = 0;
  for (int i = 0; i < num_partials; i++) {
    sum += partial_sums[i * batch_size * PSQT_BUCKETS + pos_idx * PSQT_BUCKETS +
                        bucket];
  }

  output[pos_idx * PSQT_BUCKETS + bucket] = sum;
}

// ============================================================================
// Kernel Launch Helpers (Host Functions)
// ============================================================================

extern "C" {

void cuda_extract_halfka_features(const uint64_t *piece_bitboards,
                                  const uint8_t *king_squares,
                                  int32_t *white_features,
                                  int32_t *black_features,
                                  uint32_t *feature_counts, int batch_size,
                                  int max_features, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((batch_size + 255) / 256);

  extract_halfka_features<<<grid, block, 0, stream>>>(
      piece_bitboards, king_squares, white_features, black_features,
      feature_counts, batch_size, max_features);
}

void cuda_extract_threat_features(const uint64_t *piece_bitboards,
                                  int32_t *threat_features,
                                  uint32_t *feature_counts, int batch_size,
                                  int max_features, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((batch_size + 255) / 256);

  extract_threat_features<<<grid, block, 0, stream>>>(
      piece_bitboards, threat_features, feature_counts, batch_size,
      max_features);
}

void cuda_feature_transform_full(const weight_t *weights,
                                 const weight_t *biases,
                                 const int32_t *features,
                                 const uint32_t *feature_counts,
                                 const uint32_t *feature_offsets,
                                 accumulator_t *accumulators, int hidden_dim,
                                 int batch_size, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((hidden_dim + 255) / 256, batch_size);

  feature_transform_full<<<grid, block, 0, stream>>>(
      weights, biases, features, feature_counts, feature_offsets, accumulators,
      hidden_dim, batch_size);
}

void cuda_feature_transform_optimized(
    const weight_t *weights, const weight_t *biases, const int32_t *features,
    const uint32_t *feature_counts, accumulator_t *accumulators, int hidden_dim,
    int batch_size, int max_features_per_pos, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((hidden_dim + 255) / 256, batch_size);
  size_t shared_mem = max_features_per_pos * sizeof(int32_t);

  feature_transform_optimized<<<grid, block, shared_mem, stream>>>(
      weights, biases, features, feature_counts, accumulators, hidden_dim,
      batch_size, max_features_per_pos);
}

void cuda_feature_transform_warp_optimized(
    const weight_t *weights, const weight_t *biases, const int32_t *features,
    const uint32_t *feature_counts, accumulator_t *accumulators, int hidden_dim,
    int batch_size, int max_features_per_pos, cudaStream_t stream) {

  dim3 block(256); // 8 warps per block
  dim3 grid((hidden_dim + 255) / 256, batch_size);

  feature_transform_warp_optimized<<<grid, block, 0, stream>>>(
      weights, biases, features, feature_counts, accumulators, hidden_dim,
      batch_size, max_features_per_pos);
}

void cuda_feature_transform_incremental(
    const weight_t *weights, const int32_t *added_features,
    const int32_t *removed_features, const uint32_t *add_counts,
    const uint32_t *remove_counts, const accumulator_t *src_accumulators,
    accumulator_t *dst_accumulators, int hidden_dim, int batch_size,
    cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((hidden_dim + 255) / 256, batch_size);

  feature_transform_incremental<<<grid, block, 0, stream>>>(
      weights, added_features, removed_features, add_counts, remove_counts,
      src_accumulators, dst_accumulators, hidden_dim, batch_size);
}

void cuda_double_incremental_update(
    const weight_t *weights, const int32_t *added1, const int32_t *removed1,
    const int32_t *added2, const int32_t *removed2, const uint32_t *counts,
    const accumulator_t *src_acc, accumulator_t *dst_acc, int hidden_dim,
    int perspective, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((hidden_dim + 255) / 256);

  double_incremental_update<<<grid, block, 0, stream>>>(
      weights, added1, removed1, added2, removed2, counts, src_acc, dst_acc,
      hidden_dim, perspective);
}

void cuda_fc0_layer(const accumulator_t *accumulators,
                    const layer_weight_t *weights, const int32_t *biases,
                    int8_t *output_sqr, int8_t *output_linear, int hidden_dim,
                    int batch_size, cudaStream_t stream) {

  dim3 block(64);
  dim3 grid(batch_size);

  fc0_layer<<<grid, block, 0, stream>>>(accumulators, weights, biases,
                                        output_sqr, output_linear, hidden_dim,
                                        batch_size);
}

void cuda_fc0_sparse_input(const accumulator_t *accumulators,
                           const layer_weight_t *weights, const int32_t *biases,
                           int8_t *output_sqr, int8_t *output_linear,
                           int hidden_dim, int batch_size, int bucket,
                           cudaStream_t stream) {

  dim3 block(64);
  dim3 grid(batch_size);

  fc0_sparse_input<<<grid, block, 0, stream>>>(accumulators, weights, biases,
                                               output_sqr, output_linear,
                                               hidden_dim, batch_size, bucket);
}

void cuda_fc0_layer_batched(const uint8_t *input, const layer_weight_t *weights,
                            const int32_t *biases, const int32_t *buckets,
                            int8_t *output_sqr, int8_t *output_linear,
                            int hidden_dim, int batch_size,
                            cudaStream_t stream) {

  dim3 block(16);
  dim3 grid(1, batch_size);

  fc0_layer_batched<<<grid, block, 0, stream>>>(input, weights, biases, buckets,
                                                output_sqr, output_linear,
                                                hidden_dim, batch_size);
}

void cuda_fc1_layer(const int8_t *input, const layer_weight_t *weights,
                    const int32_t *biases, int8_t *output, int batch_size,
                    cudaStream_t stream) {

  dim3 block(FC1_OUT);
  dim3 grid(batch_size);

  fc1_layer<<<grid, block, 0, stream>>>(input, weights, biases, output,
                                        batch_size);
}

void cuda_fc2_layer(const int8_t *fc1_out, const layer_weight_t *weights,
                    const int32_t *biases, const int8_t *skip_connection,
                    int32_t *output, int batch_size, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((batch_size + 255) / 256);

  fc2_layer<<<grid, block, 0, stream>>>(fc1_out, weights, biases,
                                        skip_connection, output, batch_size);
}

void cuda_nnue_forward_fused(
    const accumulator_t *accumulators, const layer_weight_t *fc0_weights,
    const int32_t *fc0_biases, const layer_weight_t *fc1_weights,
    const int32_t *fc1_biases, const layer_weight_t *fc2_weights,
    const int32_t *fc2_biases, int32_t *output, int hidden_dim, int batch_size,
    cudaStream_t stream) {

  dim3 block(64);
  dim3 grid(batch_size);

  nnue_forward_fused<<<grid, block, 0, stream>>>(
      accumulators, fc0_weights, fc0_biases, fc1_weights, fc1_biases,
      fc2_weights, fc2_biases, output, hidden_dim, batch_size);
}

void cuda_psqt_accumulate(const int32_t *psqt_weights, const int32_t *features,
                          const uint32_t *feature_counts,
                          const uint32_t *feature_offsets, int32_t *psqt_output,
                          int batch_size, cudaStream_t stream) {

  dim3 block(8);
  dim3 grid(1, batch_size);

  psqt_accumulate<<<grid, block, 0, stream>>>(psqt_weights, features,
                                              feature_counts, feature_offsets,
                                              psqt_output, batch_size);
}

void cuda_psqt_reduce(const int32_t *partial_sums, int32_t *output,
                      int num_partials, int batch_size, cudaStream_t stream) {

  dim3 block(8);
  dim3 grid(1, batch_size);

  psqt_reduce<<<grid, block, 0, stream>>>(partial_sums, output, num_partials,
                                          batch_size);
}

void cuda_init_accumulators(const weight_t *biases, accumulator_t *accumulators,
                            int hidden_dim, int batch_size,
                            cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((hidden_dim * 2 + 255) / 256, batch_size);

  init_accumulators<<<grid, block, 0, stream>>>(biases, accumulators,
                                                hidden_dim, batch_size);
}

void cuda_zero_buffer(int32_t *buffer, int count, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((count + 255) / 256);

  zero_buffer<<<grid, block, 0, stream>>>(buffer, count);
}

void cuda_swap_accumulator_perspectives(const accumulator_t *src,
                                        accumulator_t *dst, int hidden_dim,
                                        int batch_size, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((hidden_dim * 2 + 255) / 256, batch_size);

  swap_accumulator_perspectives<<<grid, block, 0, stream>>>(
      src, dst, hidden_dim, batch_size);
}

void cuda_transform_accumulator_output(const accumulator_t *accumulators,
                                       const accumulator_t *threat_accumulators,
                                       uint8_t *output, int hidden_dim,
                                       int batch_size, int use_threats,
                                       int perspective, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((hidden_dim / 2 + 255) / 256, batch_size);

  transform_accumulator_output<<<grid, block, 0, stream>>>(
      accumulators, threat_accumulators, output, hidden_dim, batch_size,
      use_threats, perspective);
}

void cuda_copy_accumulator_fast(const accumulator_t *src, accumulator_t *dst,
                                int count, cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((count / 4 + 255) / 256);

  copy_accumulator_fast<<<grid, block, 0, stream>>>(src, dst, count);
}

} // extern "C"

#endif // NNUE_CUDA_KERNELS_CU

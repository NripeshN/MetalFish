/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE Kernels

  GPU kernels for NNUE neural network evaluation on NVIDIA GPUs.
  Optimized for modern CUDA architectures with tensor core support.
*/

#ifndef NNUE_CUDA_KERNELS_CU
#define NNUE_CUDA_KERNELS_CU

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

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

__device__ __forceinline__ int popcount64(uint64_t x) {
  return __popcll(x);
}

__device__ __forceinline__ int lsb64(uint64_t x) {
  return __ffsll(x) - 1;
}

// ============================================================================
// Feature Extraction Kernels
// ============================================================================

/**
 * Extract HalfKA features from positions
 * Each thread processes one position
 */
__global__ void extract_halfka_features(
    const uint64_t* __restrict__ piece_bitboards,  // [batch_size][2][7]
    const uint8_t* __restrict__ king_squares,      // [batch_size][2]
    int32_t* __restrict__ white_features,          // [batch_size][max_features]
    int32_t* __restrict__ black_features,          // [batch_size][max_features]
    uint32_t* __restrict__ feature_counts,         // [batch_size][2]
    int batch_size,
    int max_features) {
  
  int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos_idx >= batch_size) return;

  int white_ksq = king_squares[pos_idx * 2];
  int black_ksq = king_squares[pos_idx * 2 + 1];

  int white_count = 0;
  int black_count = 0;
  int base_idx = pos_idx * max_features;

  // Iterate through all pieces
  for (int color = 0; color < 2; color++) {
    for (int pt = 1; pt <= 6; pt++) {  // PAWN to KING
      uint64_t bb = piece_bitboards[pos_idx * 14 + color * 7 + pt];
      while (bb && white_count < max_features && black_count < max_features) {
        int sq = lsb64(bb);
        bb &= bb - 1;

        // White perspective feature
        int oriented_ksq_w = white_ksq ^ ((white_ksq & 4) ? 7 : 0);
        int oriented_sq_w = sq ^ ((white_ksq & 4) ? 7 : 0);
        int piece_idx_w = (pt - 1) + (color != 0 ? 6 : 0);
        int white_feat = oriented_ksq_w * 640 + piece_idx_w * 64 + oriented_sq_w;
        
        if (white_feat >= 0 && white_feat < HALFKA_DIMS) {
          white_features[base_idx + white_count++] = white_feat;
        }

        // Black perspective feature (mirrored)
        int black_ksq_mir = black_ksq ^ 56;
        int oriented_ksq_b = black_ksq_mir ^ ((black_ksq_mir & 4) ? 7 : 0);
        int sq_mir = sq ^ 56;
        int oriented_sq_b = sq_mir ^ ((black_ksq_mir & 4) ? 7 : 0);
        int piece_idx_b = (pt - 1) + ((color ^ 1) != 0 ? 6 : 0);
        int black_feat = oriented_ksq_b * 640 + piece_idx_b * 64 + oriented_sq_b;
        
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
    const weight_t* __restrict__ weights,
    const weight_t* __restrict__ biases,
    const int32_t* __restrict__ features,
    const uint32_t* __restrict__ feature_counts,
    const uint32_t* __restrict__ feature_offsets,
    accumulator_t* __restrict__ accumulators,
    int hidden_dim,
    int batch_size) {
  
  int pos_idx = blockIdx.y;
  int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim) return;

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
    const weight_t* __restrict__ weights,
    const weight_t* __restrict__ biases,
    const int32_t* __restrict__ features,
    const uint32_t* __restrict__ feature_counts,
    accumulator_t* __restrict__ accumulators,
    int hidden_dim,
    int batch_size,
    int max_features_per_pos) {
  
  extern __shared__ int32_t shared_features[];

  int pos_idx = blockIdx.y;
  int hidden_base = blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  if (pos_idx >= batch_size) return;

  // Load features to shared memory
  int count = feature_counts[pos_idx];
  const int32_t* pos_features = features + pos_idx * max_features_per_pos;
  
  for (int i = tid; i < count; i += blockDim.x) {
    shared_features[i] = pos_features[i];
  }
  __syncthreads();

  int hidden_idx = hidden_base + tid;
  if (hidden_idx >= hidden_dim) return;

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
    const weight_t* __restrict__ weights,
    const int32_t* __restrict__ added_features,
    const int32_t* __restrict__ removed_features,
    const uint32_t* __restrict__ add_counts,
    const uint32_t* __restrict__ remove_counts,
    const accumulator_t* __restrict__ src_accumulators,
    accumulator_t* __restrict__ dst_accumulators,
    int hidden_dim,
    int batch_size) {
  
  int pos_idx = blockIdx.y;
  int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim) return;

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
__global__ void fc0_layer(
    const accumulator_t* __restrict__ accumulators,
    const layer_weight_t* __restrict__ weights,
    const int32_t* __restrict__ biases,
    int8_t* __restrict__ output_sqr,
    int8_t* __restrict__ output_linear,
    int hidden_dim,
    int batch_size) {
  
  __shared__ int8_t sqr_out[2][16];
  __shared__ int8_t linear_out[2][16];

  int pos_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (pos_idx >= batch_size) return;

  const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
  const accumulator_t* black_acc = white_acc + hidden_dim;

  // Each thread computes one or more output neurons
  for (int out = tid; out <= FC0_OUT; out += blockDim.x) {
    for (int p = 0; p < 2; p++) {
      const accumulator_t* acc = (p == 0) ? white_acc : black_acc;

      int32_t sum = biases[out];
      for (int i = 0; i < hidden_dim; i++) {
        int8_t clipped = clipped_relu(static_cast<int16_t>(acc[i] >> WEIGHT_SCALE_BITS));
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
    output_sqr[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] = sqr_out[p][o];
    output_linear[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] = linear_out[p][o];
  }
}

/**
 * FC1 layer
 */
__global__ void fc1_layer(
    const int8_t* __restrict__ input,
    const layer_weight_t* __restrict__ weights,
    const int32_t* __restrict__ biases,
    int8_t* __restrict__ output,
    int batch_size) {
  
  int pos_idx = blockIdx.x;
  int out_idx = threadIdx.x;

  if (pos_idx >= batch_size || out_idx >= FC1_OUT) return;

  const int8_t* in_ptr = input + pos_idx * 2 * FC0_OUT;

  int32_t sum = biases[out_idx];
  for (int i = 0; i < 2 * FC0_OUT; i++) {
    sum += in_ptr[i] * weights[i * FC1_OUT + out_idx];
  }

  output[pos_idx * FC1_OUT + out_idx] = clipped_relu(static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS));
}

/**
 * FC2 output layer
 */
__global__ void fc2_layer(
    const int8_t* __restrict__ fc1_out,
    const layer_weight_t* __restrict__ weights,
    const int32_t* __restrict__ biases,
    const int8_t* __restrict__ skip_connection,
    int32_t* __restrict__ output,
    int batch_size) {
  
  int pos_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos_idx >= batch_size) return;

  const int8_t* in_ptr = fc1_out + pos_idx * FC1_OUT;

  int32_t sum = biases[0];
  for (int i = 0; i < FC1_OUT; i++) {
    sum += in_ptr[i] * weights[i];
  }

  // Add skip connection
  int32_t skip_white = skip_connection[pos_idx * 2 * (FC0_OUT + 1) + FC0_OUT];
  int32_t skip_black = skip_connection[pos_idx * 2 * (FC0_OUT + 1) + (FC0_OUT + 1) + FC0_OUT];
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
__global__ void nnue_forward_fused(
    const accumulator_t* __restrict__ accumulators,
    const layer_weight_t* __restrict__ fc0_weights,
    const int32_t* __restrict__ fc0_biases,
    const layer_weight_t* __restrict__ fc1_weights,
    const int32_t* __restrict__ fc1_biases,
    const layer_weight_t* __restrict__ fc2_weights,
    const int32_t* __restrict__ fc2_biases,
    int32_t* __restrict__ output,
    int hidden_dim,
    int batch_size) {
  
  __shared__ int8_t fc0_sqr[2 * 16];
  __shared__ int8_t fc0_skip[2];
  __shared__ int8_t fc1_out[32];

  int pos_idx = blockIdx.x;
  int tid = threadIdx.x;

  if (pos_idx >= batch_size) return;

  const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
  const accumulator_t* black_acc = white_acc + hidden_dim;

  // ========== FC0 Layer ==========
  for (int out = tid; out <= FC0_OUT; out += blockDim.x) {
    for (int p = 0; p < 2; p++) {
      const accumulator_t* acc = (p == 0) ? white_acc : black_acc;

      int32_t sum = fc0_biases[out];
      for (int i = 0; i < hidden_dim; i++) {
        int8_t clipped = clipped_relu(static_cast<int16_t>(acc[i] >> WEIGHT_SCALE_BITS));
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
__global__ void psqt_accumulate(
    const int32_t* __restrict__ psqt_weights,
    const int32_t* __restrict__ features,
    const uint32_t* __restrict__ feature_counts,
    const uint32_t* __restrict__ feature_offsets,
    int32_t* __restrict__ psqt_output,
    int batch_size) {
  
  int pos_idx = blockIdx.y;
  int bucket = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || bucket >= PSQT_BUCKETS) return;

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
__global__ void init_accumulators(
    const weight_t* __restrict__ biases,
    accumulator_t* __restrict__ accumulators,
    int hidden_dim,
    int batch_size) {
  
  int pos_idx = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || idx >= hidden_dim * 2) return;

  int offset = idx % hidden_dim;
  accumulators[pos_idx * 2 * hidden_dim + idx] = static_cast<accumulator_t>(biases[offset]);
}

/**
 * Zero buffer
 */
__global__ void zero_buffer(int32_t* buffer, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    buffer[idx] = 0;
  }
}

/**
 * Copy accumulator with perspective swap
 */
__global__ void swap_accumulator_perspectives(
    const accumulator_t* __restrict__ src,
    accumulator_t* __restrict__ dst,
    int hidden_dim,
    int batch_size) {
  
  int pos_idx = blockIdx.y;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos_idx >= batch_size || idx >= hidden_dim * 2) return;

  int perspective = idx / hidden_dim;
  int offset = idx % hidden_dim;
  int swapped = 1 - perspective;

  dst[pos_idx * 2 * hidden_dim + perspective * hidden_dim + offset] =
      src[pos_idx * 2 * hidden_dim + swapped * hidden_dim + offset];
}

// ============================================================================
// Kernel Launch Helpers (Host Functions)
// ============================================================================

extern "C" {

void cuda_feature_transform_full(
    const weight_t* weights,
    const weight_t* biases,
    const int32_t* features,
    const uint32_t* feature_counts,
    const uint32_t* feature_offsets,
    accumulator_t* accumulators,
    int hidden_dim,
    int batch_size,
    cudaStream_t stream) {
  
  dim3 block(256);
  dim3 grid((hidden_dim + 255) / 256, batch_size);
  
  feature_transform_full<<<grid, block, 0, stream>>>(
      weights, biases, features, feature_counts, feature_offsets,
      accumulators, hidden_dim, batch_size);
}

void cuda_nnue_forward_fused(
    const accumulator_t* accumulators,
    const layer_weight_t* fc0_weights,
    const int32_t* fc0_biases,
    const layer_weight_t* fc1_weights,
    const int32_t* fc1_biases,
    const layer_weight_t* fc2_weights,
    const int32_t* fc2_biases,
    int32_t* output,
    int hidden_dim,
    int batch_size,
    cudaStream_t stream) {
  
  dim3 block(64);
  dim3 grid(batch_size);
  
  nnue_forward_fused<<<grid, block, 0, stream>>>(
      accumulators, fc0_weights, fc0_biases, fc1_weights, fc1_biases,
      fc2_weights, fc2_biases, output, hidden_dim, batch_size);
}

void cuda_psqt_accumulate(
    const int32_t* psqt_weights,
    const int32_t* features,
    const uint32_t* feature_counts,
    const uint32_t* feature_offsets,
    int32_t* psqt_output,
    int batch_size,
    cudaStream_t stream) {
  
  dim3 block(8);
  dim3 grid(1, batch_size);
  
  psqt_accumulate<<<grid, block, 0, stream>>>(
      psqt_weights, features, feature_counts, feature_offsets,
      psqt_output, batch_size);
}

void cuda_init_accumulators(
    const weight_t* biases,
    accumulator_t* accumulators,
    int hidden_dim,
    int batch_size,
    cudaStream_t stream) {
  
  dim3 block(256);
  dim3 grid((hidden_dim * 2 + 255) / 256, batch_size);
  
  init_accumulators<<<grid, block, 0, stream>>>(
      biases, accumulators, hidden_dim, batch_size);
}

} // extern "C"

#endif // NNUE_CUDA_KERNELS_CU

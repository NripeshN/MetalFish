/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA NNUE SIMD Kernels - Warp-Optimized

  Advanced CUDA kernels using warp-level primitives for maximum performance.
  Optimized for Volta and later architectures with independent thread scheduling.
*/

#ifndef NNUE_CUDA_SIMD_CU
#define NNUE_CUDA_SIMD_CU

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Cooperative groups for flexible thread synchronization
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// ============================================================================
// Architecture Constants
// ============================================================================

constexpr int FT_DIM_BIG = 1024;
constexpr int FT_DIM_SMALL = 128;
constexpr int FC0_OUT = 15;
constexpr int FC1_OUT = 32;
constexpr int WEIGHT_SCALE_BITS = 6;
constexpr int OUTPUT_SCALE = 16;
constexpr int HALFKA_DIMS = 45056;

using weight_t = int16_t;
using layer_weight_t = int8_t;
using accumulator_t = int32_t;

// ============================================================================
// Warp-Level Reduction Primitives
// ============================================================================

/**
 * Warp-level sum reduction using shuffle operations
 * Much faster than shared memory reduction
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

/**
 * Block-level sum reduction combining warp reductions
 */
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
  static __shared__ T shared[32];  // One element per warp
  
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  
  // Reduce within warp
  val = warp_reduce_sum(val);
  
  // Write reduced value to shared memory
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  
  // First warp reduces across warps
  if (wid == 0) {
    val = (lane < blockDim.x / 32) ? shared[lane] : 0;
    val = warp_reduce_sum(val);
  }
  
  return val;
}

/**
 * Warp-level max reduction using shuffle operations
 */
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// ============================================================================
// Activation Functions
// ============================================================================

__device__ __forceinline__ int8_t clipped_relu(int16_t x) {
  return static_cast<int8_t>(max(0, min(127, static_cast<int>(x))));
}

__device__ __forceinline__ int8_t sqr_clipped_relu(int16_t x) {
  int clamped = max(0, min(127, static_cast<int>(x)));
  return static_cast<int8_t>((clamped * clamped) >> 7);
}

// ============================================================================
// Feature Extraction with Ballot Sync
// ============================================================================

/**
 * Extract HalfKA features using warp ballot for efficient bitboard processing
 * Uses __ballot_sync to find active lanes with pieces
 */
__global__ void extract_halfka_features_simd(
    const uint64_t *__restrict__ piece_bitboards,
    const uint8_t *__restrict__ king_squares,
    int32_t *__restrict__ white_features,
    int32_t *__restrict__ black_features,
    uint32_t *__restrict__ feature_counts,
    int batch_size, int max_features) {
  
  int pos_idx = blockIdx.x;
  if (pos_idx >= batch_size) return;
  
  int lane = threadIdx.x % 32;
  int warp_id = threadIdx.x / 32;
  
  __shared__ int white_count_shared;
  __shared__ int black_count_shared;
  
  if (threadIdx.x == 0) {
    white_count_shared = 0;
    black_count_shared = 0;
  }
  __syncthreads();
  
  int white_ksq = king_squares[pos_idx * 2];
  int black_ksq = king_squares[pos_idx * 2 + 1];
  
  // Each warp processes a subset of piece types
  int color = warp_id / 3;
  int pt = (warp_id % 3) * 2 + 1;
  
  if (color < 2 && pt <= 6) {
    uint64_t bb = piece_bitboards[pos_idx * 14 + color * 7 + pt];
    
    // Each lane processes potential squares
    int sq_base = lane * 2;
    for (int sq_off = 0; sq_off < 2; sq_off++) {
      int sq = sq_base + sq_off;
      if (sq < 64 && (bb & (1ULL << sq))) {
        // White perspective
        int oriented_ksq_w = white_ksq ^ ((white_ksq & 4) ? 7 : 0);
        int oriented_sq_w = sq ^ ((white_ksq & 4) ? 7 : 0);
        int piece_idx_w = (pt - 1) + (color != 0 ? 6 : 0);
        int white_feat = oriented_ksq_w * 640 + piece_idx_w * 64 + oriented_sq_w;
        
        if (white_feat >= 0 && white_feat < HALFKA_DIMS) {
          int idx = atomicAdd(&white_count_shared, 1);
          if (idx < max_features) {
            white_features[pos_idx * max_features + idx] = white_feat;
          }
        }
        
        // Black perspective
        int black_ksq_mir = black_ksq ^ 56;
        int oriented_ksq_b = black_ksq_mir ^ ((black_ksq_mir & 4) ? 7 : 0);
        int sq_mir = sq ^ 56;
        int oriented_sq_b = sq_mir ^ ((black_ksq_mir & 4) ? 7 : 0);
        int piece_idx_b = (pt - 1) + ((color ^ 1) != 0 ? 6 : 0);
        int black_feat = oriented_ksq_b * 640 + piece_idx_b * 64 + oriented_sq_b;
        
        if (black_feat >= 0 && black_feat < HALFKA_DIMS) {
          int idx = atomicAdd(&black_count_shared, 1);
          if (idx < max_features) {
            black_features[pos_idx * max_features + idx] = black_feat;
          }
        }
      }
    }
  }
  __syncthreads();
  
  if (threadIdx.x == 0) {
    feature_counts[pos_idx * 2] = white_count_shared;
    feature_counts[pos_idx * 2 + 1] = black_count_shared;
  }
}

// ============================================================================
// Feature Transform with Warp Shuffle
// ============================================================================

/**
 * Feature transform using advanced warp shuffle for feature broadcast
 * Achieves better memory coalescing than standard approach
 */
__global__ void feature_transform_simd(
    const weight_t *__restrict__ weights,
    const weight_t *__restrict__ biases,
    const int32_t *__restrict__ features,
    const uint32_t *__restrict__ feature_counts,
    accumulator_t *__restrict__ accumulators,
    int hidden_dim, int batch_size, int max_features_per_pos) {
  
  int pos_idx = blockIdx.y;
  if (pos_idx >= batch_size) return;
  
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  
  // Each warp processes 32 hidden dimensions
  int hidden_base = (blockIdx.x * (blockDim.x / 32) + warp_id) * 32;
  int hidden_idx = hidden_base + lane;
  
  if (hidden_idx >= hidden_dim) return;
  
  // Start with bias
  accumulator_t acc = static_cast<accumulator_t>(biases[hidden_idx]);
  
  // Feature counts are stored as [white, black] for each position
  // For now, we process white features (index 0). This should be extended
  // to handle both perspectives or the caller should specify which perspective.
  int count = feature_counts[pos_idx * 2];  // Use white features
  const int32_t *pos_features = features + pos_idx * max_features_per_pos;
  
  // Process features with warp-level cooperation
  // Note: Broadcasting one feature at a time provides good coalesced access
  // to weights. Alternative approaches (shared memory or processing multiple
  // features) trade off register pressure and may not improve performance.
  // This simple approach keeps registers low and allows high occupancy.
  for (int i = 0; i < count; i++) {
    // Lane 0 reads the feature index
    int32_t feat_idx = (lane == 0) ? pos_features[i] : 0;
    
    // Broadcast to all lanes in warp using shuffle
    feat_idx = warp.shfl(feat_idx, 0);
    
    if (feat_idx >= 0 && feat_idx < HALFKA_DIMS) {
      // All lanes read coalesced weight access
      // Each thread reads weights[feat_idx * hidden_dim + hidden_idx]
      // where hidden_idx is unique per thread (hidden_base + lane)
      // This ensures perfect coalescing across the warp
      acc += weights[feat_idx * hidden_dim + hidden_idx];
    }
  }
  
  accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

// ============================================================================
// FC Layer with Warp Reduction
// ============================================================================

/**
 * Fully connected layer using warp-level sum reduction
 * Much faster than atomic operations or shared memory
 */
__global__ void fc_layer_simd(
    const int8_t *__restrict__ input,
    const layer_weight_t *__restrict__ weights,
    const int32_t *__restrict__ biases,
    int8_t *__restrict__ output,
    int input_size, int output_size, int batch_size) {
  
  int pos_idx = blockIdx.x;
  int out_idx = blockIdx.y;
  
  if (pos_idx >= batch_size || out_idx >= output_size) return;
  
  const int8_t *in_ptr = input + pos_idx * input_size;
  const layer_weight_t *w_ptr = weights + out_idx * input_size;
  
  // Each thread processes a subset of inputs
  int32_t partial_sum = 0;
  for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
    partial_sum += static_cast<int32_t>(in_ptr[i]) * w_ptr[i];
  }
  
  // Warp-level reduction
  partial_sum = warp_reduce_sum(partial_sum);
  
  // First thread in each warp writes to shared memory
  __shared__ int32_t warp_sums[32];
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  
  if (lane == 0) {
    warp_sums[wid] = partial_sum;
  }
  __syncthreads();
  
  // Final reduction by first warp
  if (wid == 0) {
    partial_sum = (lane < blockDim.x / 32) ? warp_sums[lane] : 0;
    partial_sum = warp_reduce_sum(partial_sum);
    
    if (lane == 0) {
      partial_sum += biases[out_idx];
      output[pos_idx * output_size + out_idx] = 
          clipped_relu(static_cast<int16_t>(partial_sum >> WEIGHT_SCALE_BITS));
    }
  }
}

// ============================================================================
// Batch Evaluation with Cooperative Groups
// ============================================================================

/**
 * Complete NNUE evaluation using cooperative groups
 * Enables better thread cooperation and grid-wide synchronization
 */
__global__ void batch_evaluate_simd(
    const accumulator_t *__restrict__ accumulators,
    const layer_weight_t *__restrict__ fc0_weights,
    const int32_t *__restrict__ fc0_biases,
    const layer_weight_t *__restrict__ fc1_weights,
    const int32_t *__restrict__ fc1_biases,
    const layer_weight_t *__restrict__ fc2_weights,
    const int32_t *__restrict__ fc2_biases,
    int32_t *__restrict__ output,
    int hidden_dim, int batch_size) {
  
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);
  
  int pos_idx = blockIdx.x;
  if (pos_idx >= batch_size) return;
  
  __shared__ int8_t fc0_sqr[2 * 16];
  __shared__ int8_t fc0_linear[2];
  __shared__ int8_t fc1_out[32];
  
  const accumulator_t *white_acc = accumulators + pos_idx * 2 * hidden_dim;
  const accumulator_t *black_acc = white_acc + hidden_dim;
  
  int lane = threadIdx.x % 32;
  int wid = threadIdx.x / 32;
  
  // FC0 layer - process both perspectives in parallel with warp-level cooperation
  for (int p = 0; p < 2; p++) {
    const accumulator_t *acc = (p == 0) ? white_acc : black_acc;
    
    // Each warp cooperatively computes all FC0 outputs
    for (int out = 0; out <= FC0_OUT; ++out) {
      // Lane 0 starts from bias; other lanes start from 0 to avoid double-counting
      int32_t sum = (lane == 0) ? fc0_biases[out] : 0;
      
      // Warp-level reduction over hidden dims: strided accumulation per lane
      for (int i = lane; i < hidden_dim; i += 32) {
        int8_t clipped = clipped_relu(
            static_cast<int16_t>(acc[i] >> WEIGHT_SCALE_BITS));
        sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
      }
      
      // Reduce partial sums across the warp
      sum = warp_reduce_sum(sum);
      
      if (lane == 0) {
        int16_t result = static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS);
        if (out < FC0_OUT) {
          fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
        } else {
          fc0_linear[p] = clipped_relu(result);
        }
      }
    }
  }
  block.sync();
  
  // FC1 layer
  if (lane < FC1_OUT) {
    int32_t sum = fc1_biases[lane];
    for (int i = 0; i < 2 * FC0_OUT; i++) {
      sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + lane];
    }
    fc1_out[lane] = clipped_relu(static_cast<int16_t>(sum >> WEIGHT_SCALE_BITS));
  }
  block.sync();
  
  // FC2 layer with skip connection
  if (threadIdx.x == 0) {
    int32_t sum = fc2_biases[0];
    for (int i = 0; i < FC1_OUT; i++) {
      sum += fc1_out[i] * fc2_weights[i];
    }
    
    // Add skip connection
    int32_t skip_val = ((fc0_linear[0] + fc0_linear[1]) * 600 * OUTPUT_SCALE) /
                       (2 * 127 * (1 << WEIGHT_SCALE_BITS));
    output[pos_idx] = sum + skip_val;
  }
}

// ============================================================================
// PSQT Accumulation with Warp Reduction
// ============================================================================

/**
 * PSQT (Piece-Square Table) accumulation using warp primitives
 */
__global__ void psqt_accumulate_simd(
    const int32_t *__restrict__ features,
    const uint32_t *__restrict__ feature_counts,
    const int32_t *__restrict__ psqt_weights,
    int32_t *__restrict__ psqt_values,
    int batch_size, int max_features, int num_buckets) {
  
  int pos_idx = blockIdx.x;
  if (pos_idx >= batch_size) return;
  
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  int lane = warp.thread_rank();
  
  int count = feature_counts[pos_idx];
  const int32_t *pos_features = features + pos_idx * max_features;
  
  // Each thread accumulates a subset of features
  int32_t partial_sum = 0;
  for (int i = lane; i < count; i += 32) {
    int feat_idx = pos_features[i];
    if (feat_idx >= 0) {
      partial_sum += psqt_weights[feat_idx];
    }
  }
  
  // Warp-level sum reduction
  partial_sum = warp_reduce_sum(partial_sum);
  
  // Lane 0 writes the result
  if (lane == 0) {
    psqt_values[pos_idx] = partial_sum;
  }
}

// ============================================================================
// Host Interface Functions
// ============================================================================

extern "C" {

void cuda_feature_transform_simd(
    const weight_t *weights, const weight_t *biases,
    const int32_t *features, const uint32_t *feature_counts,
    accumulator_t *accumulators, int hidden_dim, int batch_size,
    int max_features_per_pos, cudaStream_t stream) {
  
  dim3 block(256);  // 8 warps per block
  dim3 grid((hidden_dim + 255) / 256, batch_size);
  
  feature_transform_simd<<<grid, block, 0, stream>>>(
      weights, biases, features, feature_counts, accumulators,
      hidden_dim, batch_size, max_features_per_pos);
}

void cuda_fc_layer_simd(
    const int8_t *input, const layer_weight_t *weights,
    const int32_t *biases, int8_t *output,
    int input_size, int output_size, int batch_size, cudaStream_t stream) {
  
  dim3 block(128);  // 4 warps per block
  dim3 grid(batch_size, output_size);
  
  fc_layer_simd<<<grid, block, 0, stream>>>(
      input, weights, biases, output, input_size, output_size, batch_size);
}

void cuda_batch_evaluate_simd(
    const accumulator_t *accumulators,
    const layer_weight_t *fc0_weights, const int32_t *fc0_biases,
    const layer_weight_t *fc1_weights, const int32_t *fc1_biases,
    const layer_weight_t *fc2_weights, const int32_t *fc2_biases,
    int32_t *output, int hidden_dim, int batch_size, cudaStream_t stream) {
  
  dim3 block(128);
  dim3 grid(batch_size);
  
  batch_evaluate_simd<<<grid, block, 0, stream>>>(
      accumulators, fc0_weights, fc0_biases, fc1_weights, fc1_biases,
      fc2_weights, fc2_biases, output, hidden_dim, batch_size);
}

void cuda_psqt_accumulate_simd(
    const int32_t *features, const uint32_t *feature_counts,
    const int32_t *psqt_weights, int32_t *psqt_values,
    int batch_size, int max_features, int num_buckets, cudaStream_t stream) {
  
  dim3 block(32);  // Single warp
  dim3 grid(batch_size);
  
  psqt_accumulate_simd<<<grid, block, 0, stream>>>(
      features, feature_counts, psqt_weights, psqt_values,
      batch_size, max_features, num_buckets);
}

} // extern "C"

#endif // NNUE_CUDA_SIMD_CU

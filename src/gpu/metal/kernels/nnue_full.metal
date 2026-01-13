/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive Metal Compute Shaders for NNUE Evaluation

  This file contains all GPU kernels needed for NNUE inference:
  - Feature extraction (HalfKAv2_hm and FullThreats)
  - Feature transformer (sparse to dense)
  - Network layers (AffineTransform, ClippedReLU, SqrClippedReLU)
  - Incremental accumulator updates

  Optimized for Apple Silicon unified memory architecture.
*/

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// NNUE Architecture Constants
// ============================================================================

// Network dimensions
constant uint FT_DIM_BIG = 1024;
constant uint FT_DIM_SMALL = 128;
constant uint FC0_OUT = 15;
constant uint FC1_OUT = 32;
constant uint PSQT_BUCKETS = 8;
constant uint LAYER_STACKS = 8;

// Feature dimensions
constant uint HALFKA_DIMS = 45056; // 64 * 11 * 64
constant uint THREAT_DIMS = 1536;  // Full threats feature size

// Quantization
constant int WEIGHT_SCALE_BITS = 6;
constant int OUTPUT_SCALE = 16;

// Chess constants
constant uint SQUARE_NB = 64;
constant uint COLOR_NB = 2;
constant uint PIECE_TYPE_NB = 7;

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
inline int8_t clipped_relu(int16_t x) { return int8_t(clamp(int(x), 0, 127)); }

// SqrClippedReLU: (clamp(x, 0, 127))^2 / 128
inline int8_t sqr_clipped_relu(int16_t x) {
  int clamped = clamp(int(x), 0, 127);
  return int8_t((clamped * clamped) >> 7);
}

// Scaled ClippedReLU for big network (scaled by 2)
inline int8_t clipped_relu_scaled(int16_t x) {
  return int8_t(clamp(int(x), 0, 254));
}

// ============================================================================
// Bitboard Utilities
// ============================================================================

inline uint popcount64(uint64_t x) {
  return popcount(uint(x)) + popcount(uint(x >> 32));
}

inline uint lsb64(uint64_t x) {
  return (x & 0xFFFFFFFF) ? ctz(uint(x)) : 32 + ctz(uint(x >> 32));
}

inline uint64_t pop_lsb64(thread uint64_t &x) {
  uint idx = lsb64(x);
  x &= x - 1;
  return idx;
}

// ============================================================================
// Feature Extraction Kernels
// ============================================================================

// HalfKAv2_hm feature index calculation
inline int32_t make_halfka_index(uint ksq, uint piece_sq, uint piece_type,
                                 uint piece_color, uint perspective) {
  // Orient based on king file
  uint oriented_ksq = (ksq ^ (perspective * 56)) ^ ((ksq & 4) ? 7 : 0);
  uint oriented_sq = (piece_sq ^ (perspective * 56)) ^ ((ksq & 4) ? 7 : 0);

  // Piece index: 0-10 for each piece type and color relative to perspective
  uint piece_idx = piece_type - 1; // PAWN=1 to KING=6 -> 0 to 5
  if (piece_color != perspective) {
    piece_idx += 6; // Enemy pieces
  }

  // Final index
  return int32_t(oriented_ksq * 640 + piece_idx * 64 + oriented_sq);
}

// Extract HalfKA features from position
kernel void
extract_halfka_features(device const GPUPosition *positions [[buffer(0)]],
                        device int32_t *white_features [[buffer(1)]],
                        device int32_t *black_features [[buffer(2)]],
                        device uint32_t *feature_counts [[buffer(3)]],
                        constant uint &batch_size [[buffer(4)]],
                        constant uint &max_features [[buffer(5)]],
                        uint gid [[thread_position_in_grid]]) {

  if (gid >= batch_size)
    return;

  GPUPosition pos = positions[gid];
  uint white_ksq = pos.king_sq[0];
  uint black_ksq = pos.king_sq[1];

  uint white_count = 0;
  uint black_count = 0;
  uint base_idx = gid * max_features;

  // Iterate through all pieces
  for (uint color = 0; color < 2; color++) {
    for (uint pt = 1; pt <= 6; pt++) { // PAWN to KING
      uint64_t bb = pos.pieces[color][pt];
      while (bb && white_count < max_features && black_count < max_features) {
        uint sq = pop_lsb64(bb);

        // White perspective feature
        int32_t white_feat = make_halfka_index(white_ksq, sq, pt, color, 0);
        if (white_feat >= 0 && white_feat < int32_t(HALFKA_DIMS)) {
          white_features[base_idx + white_count++] = white_feat;
        }

        // Black perspective feature (mirrored)
        int32_t black_feat =
            make_halfka_index(black_ksq ^ 56, sq ^ 56, pt, color ^ 1, 1);
        if (black_feat >= 0 && black_feat < int32_t(HALFKA_DIMS)) {
          black_features[base_idx + black_count++] = black_feat;
        }
      }
    }
  }

  // Store counts
  feature_counts[gid * 2] = white_count;
  feature_counts[gid * 2 + 1] = black_count;
}

// ============================================================================
// Feature Transformer Kernels
// ============================================================================

// Full feature transform from scratch
// Transforms sparse features to dense accumulator
kernel void
feature_transform_full(device const weight_t *weights [[buffer(0)]],
                       device const weight_t *biases [[buffer(1)]],
                       device const int32_t *features [[buffer(2)]],
                       device const uint32_t *feature_counts [[buffer(3)]],
                       device const uint32_t *feature_offsets [[buffer(4)]],
                       device accumulator_t *accumulators [[buffer(5)]],
                       constant uint &hidden_dim [[buffer(6)]],
                       constant uint &batch_size [[buffer(7)]],
                       uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint hidden_idx = gid.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  // Start with bias
  accumulator_t acc = accumulator_t(biases[hidden_idx]);

  // Get feature range for this position
  uint start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  uint count = feature_counts[pos_idx];

  // Accumulate weights for active features
  for (uint i = 0; i < count; i++) {
    int32_t feature_idx = features[start + i];
    if (feature_idx >= 0 && feature_idx < int32_t(HALFKA_DIMS)) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

// SIMD-optimized feature transform using simdgroups
kernel void
feature_transform_simd(device const weight_t *weights [[buffer(0)]],
                       device const weight_t *biases [[buffer(1)]],
                       device const int32_t *features [[buffer(2)]],
                       device const uint32_t *feature_counts [[buffer(3)]],
                       device const uint32_t *feature_offsets [[buffer(4)]],
                       device accumulator_t *accumulators [[buffer(5)]],
                       constant uint &hidden_dim [[buffer(6)]],
                       constant uint &batch_size [[buffer(7)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint simd_lane [[thread_index_in_simdgroup]]) {

  uint pos_idx = gid.y;
  uint hidden_base = gid.x * 32; // Process 32 elements per SIMD group

  if (pos_idx >= batch_size)
    return;

  uint hidden_idx = hidden_base + simd_lane;
  if (hidden_idx >= hidden_dim)
    return;

  accumulator_t acc = accumulator_t(biases[hidden_idx]);

  uint start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  uint count = feature_counts[pos_idx];

  for (uint i = 0; i < count; i++) {
    int32_t feature_idx = features[start + i];
    if (feature_idx >= 0 && feature_idx < int32_t(HALFKA_DIMS)) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

// Incremental accumulator update
// Only updates changed features (much faster for move-by-move updates)
kernel void
feature_transform_incremental(device const weight_t *weights [[buffer(0)]],
                              device const FeatureUpdate *updates [[buffer(1)]],
                              device accumulator_t *accumulators [[buffer(2)]],
                              device const accumulator_t *src_accumulators
                              [[buffer(3)]],
                              constant uint &hidden_dim [[buffer(4)]],
                              constant uint &batch_size [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint hidden_idx = gid.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  FeatureUpdate update = updates[pos_idx];
  uint perspective = update.perspective;

  // Start from source accumulator
  accumulator_t acc = src_accumulators[pos_idx * 2 * hidden_dim +
                                       perspective * hidden_dim + hidden_idx];

  // Remove old features
  for (uint i = 0; i < update.num_removed; i++) {
    int32_t feature_idx = update.removed_features[i];
    if (feature_idx >= 0 && feature_idx < int32_t(HALFKA_DIMS)) {
      acc -= weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  // Add new features
  for (uint i = 0; i < update.num_added; i++) {
    int32_t feature_idx = update.added_features[i];
    if (feature_idx >= 0 && feature_idx < int32_t(HALFKA_DIMS)) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  accumulators[pos_idx * 2 * hidden_dim + perspective * hidden_dim +
               hidden_idx] = acc;
}

// ============================================================================
// PSQT (Piece-Square Table) Evaluation
// ============================================================================

kernel void psqt_accumulate(device const int32_t *psqt_weights [[buffer(0)]],
                            device const int32_t *features [[buffer(1)]],
                            device const uint32_t *feature_counts [[buffer(2)]],
                            device const uint32_t *feature_offsets
                            [[buffer(3)]],
                            device int32_t *psqt_output [[buffer(4)]],
                            constant uint &batch_size [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint bucket = gid.x;

  if (pos_idx >= batch_size || bucket >= PSQT_BUCKETS)
    return;

  uint start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  uint count = feature_counts[pos_idx];

  int32_t acc = 0;
  for (uint i = 0; i < count; i++) {
    int32_t feature_idx = features[start + i];
    if (feature_idx >= 0 && feature_idx < int32_t(HALFKA_DIMS)) {
      acc += psqt_weights[feature_idx * PSQT_BUCKETS + bucket];
    }
  }

  psqt_output[pos_idx * PSQT_BUCKETS + bucket] = acc;
}

// ============================================================================
// Network Layer Kernels
// ============================================================================

// AffineTransformSparseInput: First FC layer with sparse input
// Takes clipped accumulator values as input
kernel void fc0_sparse_input(device const accumulator_t *accumulators
                             [[buffer(0)]],
                             device const layer_weight_t *weights [[buffer(1)]],
                             device const int32_t *biases [[buffer(2)]],
                             device int8_t *output_sqr [[buffer(3)]],
                             device int8_t *output_linear [[buffer(4)]],
                             constant uint &hidden_dim [[buffer(5)]],
                             constant uint &batch_size [[buffer(6)]],
                             constant uint &bucket [[buffer(7)]],
                             uint gid [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tg_size [[threads_per_threadgroup]]) {

  uint pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  // Shared memory for intermediate results
  threadgroup int8_t sqr_out[2][16]; // FC0_OUT + 1
  threadgroup int8_t linear_out[2][16];

  // Process both perspectives
  for (uint perspective = 0; perspective < 2; perspective++) {
    device const accumulator_t *acc =
        accumulators + pos_idx * 2 * hidden_dim + perspective * hidden_dim;

    // Each thread computes one or more output neurons
    for (uint out = lid; out <= FC0_OUT; out += tg_size) {
      int32_t sum = biases[out];

      // Sparse input: only process non-zero clipped values
      for (uint i = 0; i < hidden_dim; i++) {
        int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
        if (clipped != 0) {
          sum += clipped * weights[i * (FC0_OUT + 1) + out];
        }
      }

      int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
      sqr_out[perspective][out] = sqr_clipped_relu(result);
      linear_out[perspective][out] = clipped_relu(result);
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write outputs
  if (lid < 2 * (FC0_OUT + 1)) {
    uint p = lid / (FC0_OUT + 1);
    uint o = lid % (FC0_OUT + 1);
    output_sqr[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        sqr_out[p][o];
    output_linear[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        linear_out[p][o];
  }
}

// FC1 Layer: Takes concatenated sqr outputs
kernel void fc1_layer(device const int8_t *input [[buffer(0)]],
                      device const layer_weight_t *weights [[buffer(1)]],
                      device const int32_t *biases [[buffer(2)]],
                      device int8_t *output [[buffer(3)]],
                      constant uint &batch_size [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint out_idx = gid.x;

  if (pos_idx >= batch_size || out_idx >= FC1_OUT)
    return;

  device const int8_t *in_ptr = input + pos_idx * 2 * FC0_OUT;

  int32_t sum = biases[out_idx];

  // Unrolled for better performance
  for (uint i = 0; i < 2 * FC0_OUT; i++) {
    sum += in_ptr[i] * weights[i * FC1_OUT + out_idx];
  }

  output[pos_idx * FC1_OUT + out_idx] =
      clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
}

// FC2 Layer: Final output layer
kernel void fc2_output(device const int8_t *fc1_out [[buffer(0)]],
                       device const layer_weight_t *weights [[buffer(1)]],
                       device const int32_t *biases [[buffer(2)]],
                       device const int8_t *skip_connection [[buffer(3)]],
                       device int32_t *output [[buffer(4)]],
                       constant uint &batch_size [[buffer(5)]],
                       uint gid [[thread_position_in_grid]]) {

  uint pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  device const int8_t *in_ptr = fc1_out + pos_idx * FC1_OUT;

  int32_t sum = biases[0];

  for (uint i = 0; i < FC1_OUT; i++) {
    sum += in_ptr[i] * weights[i];
  }

  // Add skip connection (average of both perspectives)
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

// Complete NNUE forward pass in a single kernel
// Best for batch evaluation where we want to minimize kernel launches
kernel void
nnue_forward_fused(device const accumulator_t *accumulators [[buffer(0)]],
                   device const layer_weight_t *fc0_weights [[buffer(1)]],
                   device const int32_t *fc0_biases [[buffer(2)]],
                   device const layer_weight_t *fc1_weights [[buffer(3)]],
                   device const int32_t *fc1_biases [[buffer(4)]],
                   device const layer_weight_t *fc2_weights [[buffer(5)]],
                   device const int32_t *fc2_biases [[buffer(6)]],
                   device int32_t *output [[buffer(7)]],
                   constant uint &hidden_dim [[buffer(8)]],
                   constant uint &batch_size [[buffer(9)]],
                   uint gid [[threadgroup_position_in_grid]],
                   uint lid [[thread_position_in_threadgroup]],
                   uint tg_size [[threads_per_threadgroup]]) {

  uint pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  // Shared memory
  threadgroup int8_t fc0_sqr[2 * 16];
  threadgroup int8_t fc0_skip[2];
  threadgroup int8_t fc1_out[32];

  device const accumulator_t *white_acc =
      accumulators + pos_idx * 2 * hidden_dim;
  device const accumulator_t *black_acc = white_acc + hidden_dim;

  // ========== FC0 Layer ==========
  for (uint out = lid; out <= FC0_OUT; out += tg_size) {
    for (uint p = 0; p < 2; p++) {
      device const accumulator_t *acc = (p == 0) ? white_acc : black_acc;

      int32_t sum = fc0_biases[out];
      for (uint i = 0; i < hidden_dim; i++) {
        int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
        sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
      }

      int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);

      if (out < FC0_OUT) {
        fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
      } else {
        fc0_skip[p] = clipped_relu(result);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ========== FC1 Layer ==========
  for (uint out = lid; out < FC1_OUT; out += tg_size) {
    int32_t sum = fc1_biases[out];
    for (uint i = 0; i < 2 * FC0_OUT; i++) {
      sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
    }
    fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ========== FC2 Layer ==========
  if (lid == 0) {
    int32_t sum = fc2_biases[0];
    for (uint i = 0; i < FC1_OUT; i++) {
      sum += fc1_out[i] * fc2_weights[i];
    }

    // Skip connection
    int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * OUTPUT_SCALE) /
                       (2 * 127 * (1 << WEIGHT_SCALE_BITS));

    output[pos_idx] = sum + skip_val;
  }
}

// ============================================================================
// Utility Kernels
// ============================================================================

// Copy accumulator with perspective swap
kernel void
swap_accumulator_perspectives(device const accumulator_t *src [[buffer(0)]],
                              device accumulator_t *dst [[buffer(1)]],
                              constant uint &hidden_dim [[buffer(2)]],
                              constant uint &batch_size [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint idx = gid.x;

  if (pos_idx >= batch_size || idx >= hidden_dim * 2)
    return;

  uint perspective = idx / hidden_dim;
  uint offset = idx % hidden_dim;
  uint swapped = 1 - perspective;

  dst[pos_idx * 2 * hidden_dim + perspective * hidden_dim + offset] =
      src[pos_idx * 2 * hidden_dim + swapped * hidden_dim + offset];
}

// Initialize accumulators with biases
kernel void init_accumulators(device const weight_t *biases [[buffer(0)]],
                              device accumulator_t *accumulators [[buffer(1)]],
                              constant uint &hidden_dim [[buffer(2)]],
                              constant uint &batch_size [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint idx = gid.x;

  if (pos_idx >= batch_size || idx >= hidden_dim * 2)
    return;

  uint offset = idx % hidden_dim;
  accumulators[pos_idx * 2 * hidden_dim + idx] = accumulator_t(biases[offset]);
}

// Zero buffer
kernel void zero_buffer(device int32_t *buffer [[buffer(0)]],
                        constant uint &count [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {

  if (gid < count) {
    buffer[gid] = 0;
  }
}

// ============================================================================
// Batch Evaluation Entry Point
// ============================================================================

// Main entry point for batch NNUE evaluation
// Combines feature extraction and network forward pass
kernel void
batch_nnue_evaluate(device const GPUPosition *positions [[buffer(0)]],
                    device const weight_t *ft_weights [[buffer(1)]],
                    device const weight_t *ft_biases [[buffer(2)]],
                    device const int32_t *psqt_weights [[buffer(3)]],
                    device const layer_weight_t *fc0_weights [[buffer(4)]],
                    device const int32_t *fc0_biases [[buffer(5)]],
                    device const layer_weight_t *fc1_weights [[buffer(6)]],
                    device const int32_t *fc1_biases [[buffer(7)]],
                    device const layer_weight_t *fc2_weights [[buffer(8)]],
                    device const int32_t *fc2_biases [[buffer(9)]],
                    device int32_t *psqt_output [[buffer(10)]],
                    device int32_t *positional_output [[buffer(11)]],
                    constant uint &hidden_dim [[buffer(12)]],
                    constant uint &batch_size [[buffer(13)]],
                    uint gid [[threadgroup_position_in_grid]],
                    uint lid [[thread_position_in_threadgroup]],
                    uint tg_size [[threads_per_threadgroup]]) {

  // This is a placeholder for a fully fused kernel
  // In practice, we'll dispatch separate kernels for better occupancy
  // and easier debugging
}

// ============================================================================
// Threat Feature Extraction (FullThreats)
// ============================================================================

// Extract threat features from position
kernel void
extract_threat_features(device const GPUPosition *positions [[buffer(0)]],
                        device int32_t *threat_features [[buffer(1)]],
                        device uint32_t *feature_counts [[buffer(2)]],
                        constant uint &batch_size [[buffer(3)]],
                        constant uint &max_features [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {

  if (gid >= batch_size)
    return;

  GPUPosition pos = positions[gid];
  uint count = 0;
  uint base_idx = gid * max_features;

  // Threat feature extraction based on piece attacks
  // Simplified version - full implementation requires attack tables
  for (uint attacker_color = 0; attacker_color < 2; attacker_color++) {
    for (uint pt = 1; pt <= 6 && count < max_features; pt++) {
      uint64_t attackers = pos.pieces[attacker_color][pt];
      while (attackers && count < max_features) {
        uint from = lsb64(attackers);
        attackers &= attackers - 1;

        // For each attacker, check what it threatens
        for (uint target_color = 0; target_color < 2; target_color++) {
          for (uint target_pt = 1; target_pt <= 6 && count < max_features;
               target_pt++) {
            uint64_t targets = pos.pieces[target_color][target_pt];
            while (targets && count < max_features) {
              uint to = lsb64(targets);
              targets &= targets - 1;

              // Simplified threat index calculation
              int32_t threat_idx =
                  int32_t(attacker_color * 768 + pt * 128 + target_pt * 16 +
                          (from % 8) + (to % 8));
              if (threat_idx >= 0 && threat_idx < int32_t(THREAT_DIMS)) {
                threat_features[base_idx + count++] = threat_idx;
              }
            }
          }
        }
      }
    }
  }

  feature_counts[gid] = count;
}

// ============================================================================
// Double Incremental Update (Optimized for consecutive moves)
// ============================================================================

// Double incremental update - combines two consecutive move updates
kernel void
double_incremental_update(device const weight_t *weights [[buffer(0)]],
                          device const int32_t *added1 [[buffer(1)]],
                          device const int32_t *removed1 [[buffer(2)]],
                          device const int32_t *added2 [[buffer(3)]],
                          device const int32_t *removed2 [[buffer(4)]],
                          device const uint32_t *counts
                          [[buffer(5)]], // [add1, rem1, add2, rem2]
                          device const accumulator_t *src_acc [[buffer(6)]],
                          device accumulator_t *dst_acc [[buffer(7)]],
                          constant uint &hidden_dim [[buffer(8)]],
                          constant uint &perspective [[buffer(9)]],
                          uint gid [[thread_position_in_grid]]) {

  if (gid >= hidden_dim)
    return;

  uint num_added1 = counts[0];
  uint num_removed1 = counts[1];
  uint num_added2 = counts[2];
  uint num_removed2 = counts[3];

  accumulator_t acc = src_acc[perspective * hidden_dim + gid];

  // First move: remove then add
  for (uint i = 0; i < num_removed1; i++) {
    int32_t feat_idx = removed1[i];
    if (feat_idx >= 0) {
      acc -= weights[feat_idx * hidden_dim + gid];
    }
  }
  for (uint i = 0; i < num_added1; i++) {
    int32_t feat_idx = added1[i];
    if (feat_idx >= 0) {
      acc += weights[feat_idx * hidden_dim + gid];
    }
  }

  // Second move: remove then add
  for (uint i = 0; i < num_removed2; i++) {
    int32_t feat_idx = removed2[i];
    if (feat_idx >= 0) {
      acc -= weights[feat_idx * hidden_dim + gid];
    }
  }
  for (uint i = 0; i < num_added2; i++) {
    int32_t feat_idx = added2[i];
    if (feat_idx >= 0) {
      acc += weights[feat_idx * hidden_dim + gid];
    }
  }

  dst_acc[perspective * hidden_dim + gid] = acc;
}

// ============================================================================
// SIMD-Optimized Feature Transform (using simdgroups)
// ============================================================================

// Feature transform using SIMD operations for better throughput
kernel void feature_transform_simd_optimized(
    device const weight_t *weights [[buffer(0)]],
    device const weight_t *biases [[buffer(1)]],
    device const int32_t *features [[buffer(2)]],
    device const uint32_t *feature_counts [[buffer(3)]],
    device accumulator_t *accumulators [[buffer(4)]],
    constant uint &hidden_dim [[buffer(5)]],
    constant uint &batch_size [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {

  uint pos_idx = tg_pos.y;
  if (pos_idx >= batch_size)
    return;

  // Each SIMD group processes 32 hidden dimensions
  uint hidden_base = (tg_pos.x * tg_size.x / 32 + simd_group) * 32;
  uint hidden_idx = hidden_base + simd_lane;

  if (hidden_idx >= hidden_dim)
    return;

  // Start with bias
  accumulator_t acc = accumulator_t(biases[hidden_idx]);

  uint count = feature_counts[pos_idx];
  device const int32_t *pos_features =
      features + pos_idx * 32; // Max 32 features per position

  // Accumulate weights for active features
  for (uint i = 0; i < count; i++) {
    int32_t feat_idx = pos_features[i];
    if (feat_idx >= 0 && feat_idx < int32_t(HALFKA_DIMS)) {
      acc += weights[feat_idx * hidden_dim + hidden_idx];
    }
  }

  accumulators[pos_idx * hidden_dim * 2 + hidden_idx] = acc;
}

// ============================================================================
// Accumulator Clipping and Pairwise Multiplication (Transform Output)
// ============================================================================

// Transform accumulator to network input with clipping and pairwise
// multiplication
kernel void transform_accumulator_output(
    device const accumulator_t *accumulators [[buffer(0)]],
    device const accumulator_t *threat_accumulators [[buffer(1)]],
    device uint8_t *output [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant uint &batch_size [[buffer(4)]],
    constant uint &use_threats [[buffer(5)]],
    constant uint &perspective [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint out_idx = gid.x;

  if (pos_idx >= batch_size || out_idx >= hidden_dim / 2)
    return;

  uint half_dim = hidden_dim / 2;
  device const accumulator_t *acc =
      accumulators + pos_idx * hidden_dim * 2 + perspective * hidden_dim;

  int16_t sum0, sum1;

  if (use_threats) {
    device const accumulator_t *threat_acc = threat_accumulators +
                                             pos_idx * hidden_dim * 2 +
                                             perspective * hidden_dim;
    sum0 = clamp(int(acc[out_idx] + threat_acc[out_idx]), 0, 255);
    sum1 = clamp(int(acc[out_idx + half_dim] + threat_acc[out_idx + half_dim]),
                 0, 255);
  } else {
    sum0 = clamp(int(acc[out_idx]) >> WEIGHT_SCALE_BITS, 0, 254);
    sum1 = clamp(int(acc[out_idx + half_dim]) >> WEIGHT_SCALE_BITS, 0, 254);
  }

  // Pairwise multiplication with division by 512
  output[pos_idx * hidden_dim + perspective * half_dim + out_idx] =
      uint8_t((sum0 * sum1) / 512);
}

// ============================================================================
// Batch FC Layer with Multiple Buckets
// ============================================================================

// FC0 layer with per-position bucket selection
kernel void
fc0_layer_batched(device const uint8_t *input [[buffer(0)]],
                  device const layer_weight_t *weights
                  [[buffer(1)]], // [LAYER_STACKS][hidden_dim*2][FC0_OUT+1]
                  device const int32_t *biases
                  [[buffer(2)]], // [LAYER_STACKS][FC0_OUT+1]
                  device const int32_t *buckets [[buffer(3)]],
                  device int8_t *output_sqr [[buffer(4)]],
                  device int8_t *output_linear [[buffer(5)]],
                  constant uint &hidden_dim [[buffer(6)]],
                  constant uint &batch_size [[buffer(7)]],
                  uint2 gid [[thread_position_in_grid]]) {

  uint pos_idx = gid.y;
  uint out_idx = gid.x;

  if (pos_idx >= batch_size || out_idx > FC0_OUT)
    return;

  int bucket = buckets[pos_idx];

  // Get weights and biases for this bucket
  device const layer_weight_t *bucket_weights =
      weights + bucket * hidden_dim * 2 * (FC0_OUT + 1);
  device const int32_t *bucket_biases = biases + bucket * (FC0_OUT + 1);

  device const uint8_t *in_ptr = input + pos_idx * hidden_dim * 2;

  int32_t sum = bucket_biases[out_idx];

  // Sparse input: only process non-zero values
  for (uint i = 0; i < hidden_dim * 2; i++) {
    if (in_ptr[i] != 0) {
      sum += in_ptr[i] * bucket_weights[i * (FC0_OUT + 1) + out_idx];
    }
  }

  int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
  output_sqr[pos_idx * (FC0_OUT + 1) + out_idx] = sqr_clipped_relu(result);
  output_linear[pos_idx * (FC0_OUT + 1) + out_idx] = clipped_relu(result);
}

// ============================================================================
// Complete Evaluation Pipeline
// ============================================================================

// Full evaluation combining all steps
kernel void
evaluate_position_full(device const GPUPosition *positions [[buffer(0)]],
                       device const weight_t *ft_weights [[buffer(1)]],
                       device const weight_t *ft_biases [[buffer(2)]],
                       device const int8_t *threat_weights [[buffer(3)]],
                       device const int32_t *ft_psqt [[buffer(4)]],
                       device const int32_t *threat_psqt [[buffer(5)]],
                       device const layer_weight_t *fc0_weights [[buffer(6)]],
                       device const int32_t *fc0_biases [[buffer(7)]],
                       device const layer_weight_t *fc1_weights [[buffer(8)]],
                       device const int32_t *fc1_biases [[buffer(9)]],
                       device const layer_weight_t *fc2_weights [[buffer(10)]],
                       device const int32_t *fc2_biases [[buffer(11)]],
                       device int32_t *psqt_output [[buffer(12)]],
                       device int32_t *positional_output [[buffer(13)]],
                       constant uint &hidden_dim [[buffer(14)]],
                       constant uint &batch_size [[buffer(15)]],
                       constant uint &use_threats [[buffer(16)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]]) {

  uint pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  // This kernel orchestrates the full evaluation
  // Individual steps are better done as separate kernel dispatches
  // for flexibility and debugging
}

// ============================================================================
// Memory Copy Utilities
// ============================================================================

// Fast memory copy for accumulator states
kernel void copy_accumulator_fast(device const accumulator_t *src [[buffer(0)]],
                                  device accumulator_t *dst [[buffer(1)]],
                                  constant uint &count [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]],
                                  uint simd_lane
                                  [[thread_index_in_simdgroup]]) {

  // Each thread copies 4 elements for better memory coalescing
  uint base = gid * 4;
  if (base + 3 < count) {
    dst[base] = src[base];
    dst[base + 1] = src[base + 1];
    dst[base + 2] = src[base + 2];
    dst[base + 3] = src[base + 3];
  } else {
    for (uint i = 0; i < 4 && base + i < count; i++) {
      dst[base + i] = src[base + i];
    }
  }
}

// ============================================================================
// Reduction Kernels for PSQT Computation
// ============================================================================

// Parallel reduction for PSQT accumulation
kernel void psqt_reduce(device const int32_t *partial_sums [[buffer(0)]],
                        device int32_t *output [[buffer(1)]],
                        constant uint &num_partials [[buffer(2)]],
                        constant uint &batch_size [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]],
                        uint simd_lane [[thread_index_in_simdgroup]]) {

  uint pos_idx = gid.y;
  uint bucket = gid.x;

  if (pos_idx >= batch_size || bucket >= PSQT_BUCKETS)
    return;

  int32_t sum = 0;
  for (uint i = 0; i < num_partials; i++) {
    sum += partial_sums[i * batch_size * PSQT_BUCKETS + pos_idx * PSQT_BUCKETS +
                        bucket];
  }

  output[pos_idx * PSQT_BUCKETS + bucket] = sum;
}

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Unified NNUE Metal Shaders for Apple Silicon

  All GPU kernels for NNUE inference optimized for unified memory:
  - Feature extraction (HalfKAv2_hm)
  - Feature transformer (sparse to dense)
  - Network layers (FC0, FC1, FC2 with skip connection)
  - Incremental accumulator updates
  - SIMD-optimized variants
*/

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Architecture Constants (matching ReferenceEngine)
// ============================================================================

constant uint FT_DIM_BIG = 1024;
constant uint FT_DIM_SMALL = 128;
constant uint FC0_OUT = 15;
constant uint FC1_OUT = 32;
constant uint PSQT_BUCKETS = 8;
constant uint LAYER_STACKS = 8;
constant uint HALFKA_DIMS = 45056;
constant uint THREAT_DIMS = 1536;
constant uint WEIGHT_SCALE_BITS = 6;
constant uint OUTPUT_SCALE = 16;
constant uint SQUARE_NB = 64;

// ============================================================================
// Type Definitions
// ============================================================================

typedef int16_t weight_t;
typedef int8_t layer_weight_t;
typedef int32_t accumulator_t;
typedef uint8_t activation_t;

// ============================================================================
// Data Structures
// ============================================================================

struct GPUPosition {
  uint64_t pieces[2][7];
  uint8_t king_sq[2];
  uint8_t stm;
  uint8_t piece_count;
  uint8_t padding[4];
};

struct FeatureUpdate {
  int32_t added[32];
  int32_t removed[32];
  uint8_t num_added;
  uint8_t num_removed;
  uint8_t perspective;
  uint8_t is_king_move;
};

// ============================================================================
// Activation Functions
// ============================================================================

inline int8_t clipped_relu(int16_t x) { return int8_t(clamp(int(x), 0, 127)); }

inline int8_t sqr_clipped_relu(int16_t x) {
  int clamped = clamp(int(x), 0, 127);
  return int8_t((clamped * clamped) >> 7);
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
// HalfKAv2_hm Feature Index
// ============================================================================

inline int32_t make_halfka_index(uint ksq, uint piece_sq, uint piece_type,
                                 uint piece_color, uint perspective) {
  uint king_file = ksq & 7;
  uint mirror = (king_file >= 4) ? 7 : 0;
  uint oriented_ksq = (ksq ^ (perspective * 56)) ^ mirror;
  uint oriented_sq = (piece_sq ^ (perspective * 56)) ^ mirror;
  uint piece_idx = piece_type - 1;
  if (piece_color != perspective)
    piece_idx += 6;
  return int32_t(oriented_ksq * 640 + piece_idx * 64 + oriented_sq);
}

// ============================================================================
// Feature Extraction
// ============================================================================

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
  uint white_count = 0, black_count = 0;
  uint base_idx = gid * max_features;

  for (uint color = 0; color < 2; color++) {
    for (uint pt = 1; pt <= 5; pt++) {
      uint64_t bb = pos.pieces[color][pt];
      while (bb && white_count < max_features && black_count < max_features) {
        uint sq = pop_lsb64(bb);
        int32_t wf = make_halfka_index(white_ksq, sq, pt, color, 0);
        if (wf >= 0 && wf < int32_t(HALFKA_DIMS))
          white_features[base_idx + white_count++] = wf;
        int32_t bf =
            make_halfka_index(black_ksq ^ 56, sq ^ 56, pt, color ^ 1, 1);
        if (bf >= 0 && bf < int32_t(HALFKA_DIMS))
          black_features[base_idx + black_count++] = bf;
      }
    }
  }
  feature_counts[gid * 2] = white_count;
  feature_counts[gid * 2 + 1] = black_count;
}

// ============================================================================
// Feature Transformer
// ============================================================================

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

  accumulator_t acc = accumulator_t(biases[hidden_idx]);
  uint start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  uint count = feature_counts[pos_idx];

  for (uint i = 0; i < count; i++) {
    int32_t feat_idx = features[start + i];
    if (feat_idx >= 0 && feat_idx < int32_t(HALFKA_DIMS))
      acc += weights[feat_idx * hidden_dim + hidden_idx];
  }
  accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

kernel void
feature_transform_simd(device const weight_t *weights [[buffer(0)]],
                       device const weight_t *biases [[buffer(1)]],
                       device const int32_t *features [[buffer(2)]],
                       device const uint32_t *feature_counts [[buffer(3)]],
                       device accumulator_t *accumulators [[buffer(4)]],
                       constant uint &hidden_dim [[buffer(5)]],
                       constant uint &batch_size [[buffer(6)]],
                       constant uint &max_features [[buffer(7)]],
                       uint2 tg_pos [[threadgroup_position_in_grid]],
                       uint simd_lane [[thread_index_in_simdgroup]],
                       uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint pos_idx = tg_pos.y;
  if (pos_idx >= batch_size)
    return;

  uint hidden_base = (tg_pos.x * 4 + simd_group) * 32;
  uint hidden_idx = hidden_base + simd_lane;
  if (hidden_idx >= hidden_dim)
    return;

  accumulator_t acc = accumulator_t(biases[hidden_idx]);
  uint count = feature_counts[pos_idx];
  device const int32_t *pos_features = features + pos_idx * max_features;

  for (uint i = 0; i < count; i++) {
    int32_t feat_idx = pos_features[i];
    if (feat_idx >= 0 && feat_idx < int32_t(HALFKA_DIMS))
      acc += weights[feat_idx * hidden_dim + hidden_idx];
  }
  accumulators[pos_idx * hidden_dim * 2 + hidden_idx] = acc;
}

// ============================================================================
// Incremental Updates
// ============================================================================

kernel void incremental_update(device const weight_t *weights [[buffer(0)]],
                               device const FeatureUpdate *updates
                               [[buffer(1)]],
                               device const accumulator_t *src_acc
                               [[buffer(2)]],
                               device accumulator_t *dst_acc [[buffer(3)]],
                               constant uint &hidden_dim [[buffer(4)]],
                               constant uint &batch_size [[buffer(5)]],
                               uint2 gid [[thread_position_in_grid]]) {
  uint pos_idx = gid.y;
  uint hidden_idx = gid.x;
  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  FeatureUpdate update = updates[pos_idx];
  uint perspective = update.perspective;
  accumulator_t acc =
      src_acc[pos_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx];

  for (uint i = 0; i < update.num_removed; i++) {
    int32_t feat_idx = update.removed[i];
    if (feat_idx >= 0 && feat_idx < int32_t(HALFKA_DIMS))
      acc -= weights[feat_idx * hidden_dim + hidden_idx];
  }
  for (uint i = 0; i < update.num_added; i++) {
    int32_t feat_idx = update.added[i];
    if (feat_idx >= 0 && feat_idx < int32_t(HALFKA_DIMS))
      acc += weights[feat_idx * hidden_dim + hidden_idx];
  }
  dst_acc[pos_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx] =
      acc;
}

kernel void
incremental_update_simd(device const weight_t *weights [[buffer(0)]],
                        device const FeatureUpdate *updates [[buffer(1)]],
                        device const accumulator_t *src_acc [[buffer(2)]],
                        device accumulator_t *dst_acc [[buffer(3)]],
                        constant uint &hidden_dim [[buffer(4)]],
                        uint2 tg_pos [[threadgroup_position_in_grid]],
                        uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint pos_idx = tg_pos.y;
  FeatureUpdate update = updates[pos_idx];
  uint perspective = update.perspective;

  uint hidden_base = (tg_pos.x * 4 + simd_group) * 32;
  uint hidden_idx = hidden_base + simd_lane;
  if (hidden_idx >= hidden_dim)
    return;

  accumulator_t acc =
      src_acc[pos_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx];

  for (uint i = 0; i < update.num_removed; i++) {
    int32_t feat_idx = update.removed[i];
    if (feat_idx >= 0)
      acc -= weights[feat_idx * hidden_dim + hidden_idx];
  }
  for (uint i = 0; i < update.num_added; i++) {
    int32_t feat_idx = update.added[i];
    if (feat_idx >= 0)
      acc += weights[feat_idx * hidden_dim + hidden_idx];
  }
  dst_acc[pos_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx] =
      acc;
}

// ============================================================================
// PSQT Accumulation
// ============================================================================

kernel void psqt_accumulate(device const int32_t *psqt_weights [[buffer(0)]],
                            device const int32_t *features [[buffer(1)]],
                            device const uint32_t *feature_counts [[buffer(2)]],
                            device int32_t *psqt_output [[buffer(3)]],
                            constant uint &batch_size [[buffer(4)]],
                            constant uint &max_features [[buffer(5)]],
                            uint2 gid [[thread_position_in_grid]]) {
  uint pos_idx = gid.y;
  uint bucket = gid.x;
  if (pos_idx >= batch_size || bucket >= PSQT_BUCKETS)
    return;

  uint count = feature_counts[pos_idx];
  device const int32_t *pos_features = features + pos_idx * max_features;

  int32_t acc = 0;
  for (uint i = 0; i < count; i++) {
    int32_t feat_idx = pos_features[i];
    if (feat_idx >= 0 && feat_idx < int32_t(HALFKA_DIMS))
      acc += psqt_weights[feat_idx * PSQT_BUCKETS + bucket];
  }
  psqt_output[pos_idx * PSQT_BUCKETS + bucket] = acc;
}

// ============================================================================
// FC Layers
// ============================================================================

kernel void fc0_sparse(device const accumulator_t *accumulators [[buffer(0)]],
                       device const layer_weight_t *weights [[buffer(1)]],
                       device const int32_t *biases [[buffer(2)]],
                       device int8_t *output_sqr [[buffer(3)]],
                       device int8_t *output_linear [[buffer(4)]],
                       constant uint &hidden_dim [[buffer(5)]],
                       constant uint &batch_size [[buffer(6)]],
                       uint gid [[threadgroup_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]],
                       uint tg_size [[threads_per_threadgroup]]) {
  uint pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  threadgroup int8_t sqr_out[2][16];
  threadgroup int8_t linear_out[2][16];

  for (uint perspective = 0; perspective < 2; perspective++) {
    device const accumulator_t *acc =
        accumulators + pos_idx * 2 * hidden_dim + perspective * hidden_dim;

    for (uint out = lid; out <= FC0_OUT; out += tg_size) {
      int32_t sum = biases[out];
      for (uint i = 0; i < hidden_dim; i++) {
        int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
        if (clipped != 0)
          sum += clipped * weights[i * (FC0_OUT + 1) + out];
      }
      int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
      sqr_out[perspective][out] = sqr_clipped_relu(result);
      linear_out[perspective][out] = clipped_relu(result);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < 2 * (FC0_OUT + 1)) {
    uint p = lid / (FC0_OUT + 1);
    uint o = lid % (FC0_OUT + 1);
    output_sqr[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        sqr_out[p][o];
    output_linear[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        linear_out[p][o];
  }
}

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
  for (uint i = 0; i < 2 * FC0_OUT; i++)
    sum += in_ptr[i] * weights[i * FC1_OUT + out_idx];
  output[pos_idx * FC1_OUT + out_idx] =
      clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
}

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
  for (uint i = 0; i < FC1_OUT; i++)
    sum += in_ptr[i] * weights[i];

  int32_t skip_white = skip_connection[pos_idx * 2 * (FC0_OUT + 1) + FC0_OUT];
  int32_t skip_black =
      skip_connection[pos_idx * 2 * (FC0_OUT + 1) + (FC0_OUT + 1) + FC0_OUT];
  int32_t skip_val = ((skip_white + skip_black) * 600 * OUTPUT_SCALE) /
                     (2 * 127 * (1 << WEIGHT_SCALE_BITS));
  output[pos_idx] = sum + skip_val;
}

// ============================================================================
// Fused Forward Pass (All Layers in One Kernel)
// ============================================================================

kernel void
forward_fused(device const accumulator_t *accumulators [[buffer(0)]],
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

  threadgroup int8_t fc0_sqr[2 * 16];
  threadgroup int8_t fc0_skip[2];
  threadgroup int8_t fc1_out[32];

  device const accumulator_t *white_acc =
      accumulators + pos_idx * 2 * hidden_dim;
  device const accumulator_t *black_acc = white_acc + hidden_dim;

  // FC0
  for (uint out = lid; out <= FC0_OUT; out += tg_size) {
    for (uint p = 0; p < 2; p++) {
      device const accumulator_t *acc = (p == 0) ? white_acc : black_acc;
      int32_t sum = fc0_biases[out];
      for (uint i = 0; i < hidden_dim; i++) {
        int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
        sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
      }
      int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
      if (out < FC0_OUT)
        fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
      else
        fc0_skip[p] = clipped_relu(result);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // FC1
  for (uint out = lid; out < FC1_OUT; out += tg_size) {
    int32_t sum = fc1_biases[out];
    for (uint i = 0; i < 2 * FC0_OUT; i++)
      sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
    fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // FC2
  if (lid == 0) {
    int32_t sum = fc2_biases[0];
    for (uint i = 0; i < FC1_OUT; i++)
      sum += fc1_out[i] * fc2_weights[i];
    int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * OUTPUT_SCALE) /
                       (2 * 127 * (1 << WEIGHT_SCALE_BITS));
    output[pos_idx] = sum + skip_val;
  }
}

// ============================================================================
// Utility Kernels
// ============================================================================

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

kernel void zero_buffer(device int32_t *buffer [[buffer(0)]],
                        constant uint &count [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < count)
    buffer[gid] = 0;
}

kernel void copy_accumulator(device const accumulator_t *src [[buffer(0)]],
                             device accumulator_t *dst [[buffer(1)]],
                             constant uint &count [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
  if (gid < count)
    dst[gid] = src[gid];
}

// ============================================================================
// Double Incremental Update (for null-move search optimization)
// Applies two moves worth of updates in a single kernel
// ============================================================================

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
    int32_t feat = removed1[i];
    if (feat >= 0)
      acc -= weights[feat * hidden_dim + gid];
  }
  for (uint i = 0; i < num_added1; i++) {
    int32_t feat = added1[i];
    if (feat >= 0)
      acc += weights[feat * hidden_dim + gid];
  }

  // Second move: remove then add
  for (uint i = 0; i < num_removed2; i++) {
    int32_t feat = removed2[i];
    if (feat >= 0)
      acc -= weights[feat * hidden_dim + gid];
  }
  for (uint i = 0; i < num_added2; i++) {
    int32_t feat = added2[i];
    if (feat >= 0)
      acc += weights[feat * hidden_dim + gid];
  }

  dst_acc[perspective * hidden_dim + gid] = acc;
}

// ============================================================================
// Sparse Input FC0 with Bitmask (ReferenceEngine's find_nnz optimization)
// Uses precomputed bitmask to skip zero activations
// ============================================================================

kernel void fc0_sparse_bitmask(device const accumulator_t *accumulators
                               [[buffer(0)]],
                               device const layer_weight_t *weights
                               [[buffer(1)]],
                               device const int32_t *biases [[buffer(2)]],
                               device const uint64_t *nnz_masks
                               [[buffer(3)]], // Bitmask of non-zero elements
                               device int8_t *output_sqr [[buffer(4)]],
                               device int8_t *output_linear [[buffer(5)]],
                               constant uint &hidden_dim [[buffer(6)]],
                               constant uint &batch_size [[buffer(7)]],
                               uint gid [[threadgroup_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
  uint pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  threadgroup int8_t sqr_out[2][16];
  threadgroup int8_t linear_out[2][16];

  for (uint perspective = 0; perspective < 2; perspective++) {
    device const accumulator_t *acc =
        accumulators + pos_idx * 2 * hidden_dim + perspective * hidden_dim;
    uint64_t mask = nnz_masks[pos_idx * 2 + perspective];

    for (uint out = lid; out <= FC0_OUT; out += tg_size) {
      int32_t sum = biases[out];

      // Process only non-zero elements using bitmask
      uint64_t remaining = mask;
      while (remaining) {
        uint idx = ctz(uint(remaining));
        if (remaining >> 32)
          idx = 32 + ctz(uint(remaining >> 32));
        else
          idx = ctz(uint(remaining));

        int8_t clipped = clipped_relu(int16_t(acc[idx] >> WEIGHT_SCALE_BITS));
        sum += clipped * weights[idx * (FC0_OUT + 1) + out];
        remaining &= remaining - 1;
      }

      int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
      sqr_out[perspective][out] = sqr_clipped_relu(result);
      linear_out[perspective][out] = clipped_relu(result);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lid < 2 * (FC0_OUT + 1)) {
    uint p = lid / (FC0_OUT + 1);
    uint o = lid % (FC0_OUT + 1);
    output_sqr[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        sqr_out[p][o];
    output_linear[pos_idx * 2 * (FC0_OUT + 1) + p * (FC0_OUT + 1) + o] =
        linear_out[p][o];
  }
}

// ============================================================================
// Permuted Weight Feature Transform
// Uses pre-permuted weights for optimal SIMD memory access
// ============================================================================

kernel void
feature_transform_permuted(device const weight_t *permuted_weights
                           [[buffer(0)]], // Permuted for tile access
                           device const weight_t *biases [[buffer(1)]],
                           device const int32_t *features [[buffer(2)]],
                           device const uint32_t *feature_counts [[buffer(3)]],
                           device accumulator_t *accumulators [[buffer(4)]],
                           constant uint &hidden_dim [[buffer(5)]],
                           constant uint &batch_size [[buffer(6)]],
                           constant uint &max_features [[buffer(7)]],
                           constant uint &num_features
                           [[buffer(8)]], // Total feature dimension
                           uint2 tg_pos [[threadgroup_position_in_grid]],
                           uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint pos_idx = tg_pos.y;
  if (pos_idx >= batch_size)
    return;

  // Each simdgroup processes one tile of 32 hidden dimensions
  uint tile = tg_pos.x * 4 + simd_group;
  uint hidden_base = tile * 32;
  uint hidden_idx = hidden_base + simd_lane;

  if (hidden_idx >= hidden_dim)
    return;

  accumulator_t acc = accumulator_t(biases[hidden_idx]);
  uint count = feature_counts[pos_idx];
  device const int32_t *pos_features = features + pos_idx * max_features;

  // Permuted weight layout: weights[tile * num_features * 32 + feature * 32 +
  // lane]
  for (uint i = 0; i < count; i++) {
    int32_t feat_idx = pos_features[i];
    if (feat_idx >= 0 && feat_idx < int32_t(num_features)) {
      // Access permuted weights - consecutive lanes access consecutive memory
      acc += permuted_weights[tile * num_features * 32 + feat_idx * 32 +
                              simd_lane];
    }
  }

  accumulators[pos_idx * hidden_dim * 2 + hidden_idx] = acc;
}

// ============================================================================
// SIMD Optimized Forward Pass with Parallel Reduction
// ============================================================================

kernel void
forward_simd_optimized(device const accumulator_t *accumulators [[buffer(0)]],
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
                       uint tg_size [[threads_per_threadgroup]],
                       uint simd_lane [[thread_index_in_simdgroup]],
                       uint simd_group [[simdgroup_index_in_threadgroup]]) {
  uint pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  threadgroup int8_t fc0_sqr[2 * 16];
  threadgroup int8_t fc0_skip[2];
  threadgroup int8_t fc1_out[32];
  threadgroup int32_t partial_sums[4];

  device const accumulator_t *white_acc =
      accumulators + pos_idx * 2 * hidden_dim;
  device const accumulator_t *black_acc = white_acc + hidden_dim;

  // FC0 with SIMD reduction
  uint elements_per_thread = (hidden_dim + tg_size - 1) / tg_size;

  for (uint out = 0; out <= FC0_OUT; out++) {
    for (uint p = 0; p < 2; p++) {
      device const accumulator_t *acc = (p == 0) ? white_acc : black_acc;

      // Each thread computes partial sum
      int32_t local_sum = 0;
      uint start_idx = lid * elements_per_thread;
      uint end_idx = min(start_idx + elements_per_thread, hidden_dim);

      // 4-way unrolled
      uint i = start_idx;
      for (; i + 3 < end_idx; i += 4) {
        int8_t c0 = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
        int8_t c1 = clipped_relu(int16_t(acc[i + 1] >> WEIGHT_SCALE_BITS));
        int8_t c2 = clipped_relu(int16_t(acc[i + 2] >> WEIGHT_SCALE_BITS));
        int8_t c3 = clipped_relu(int16_t(acc[i + 3] >> WEIGHT_SCALE_BITS));
        local_sum += c0 * fc0_weights[(i) * (FC0_OUT + 1) + out];
        local_sum += c1 * fc0_weights[(i + 1) * (FC0_OUT + 1) + out];
        local_sum += c2 * fc0_weights[(i + 2) * (FC0_OUT + 1) + out];
        local_sum += c3 * fc0_weights[(i + 3) * (FC0_OUT + 1) + out];
      }
      for (; i < end_idx; i++) {
        int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
        local_sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
      }

      // SIMD reduction
      int32_t simd_sum_val = simd_sum(local_sum);

      if (simd_lane == 0 && simd_group < 4) {
        partial_sums[simd_group] = simd_sum_val;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (lid == 0) {
        int32_t total = fc0_biases[out];
        uint num_simd_groups = (tg_size + 31) / 32;
        for (uint s = 0; s < num_simd_groups && s < 4; s++) {
          total += partial_sums[s];
        }

        int16_t result = int16_t(total >> WEIGHT_SCALE_BITS);
        if (out < FC0_OUT) {
          fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
        } else {
          fc0_skip[p] = clipped_relu(result);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // FC1 - parallel across outputs
  for (uint out = lid; out < FC1_OUT; out += tg_size) {
    int32_t sum = fc1_biases[out];
    for (uint i = 0; i < 2 * FC0_OUT; i++) {
      sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
    }
    fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // FC2 with SIMD reduction
  if (simd_group == 0) {
    int32_t local_sum = 0;
    for (uint i = simd_lane; i < FC1_OUT; i += 32) {
      local_sum += fc1_out[i] * fc2_weights[i];
    }

    int32_t sum = simd_sum(local_sum);

    if (simd_lane == 0) {
      sum += fc2_biases[0];
      int32_t skip_val =
          ((fc0_skip[0] + fc0_skip[1]) * 600 * int32_t(OUTPUT_SCALE)) /
          (2 * 127 * (1 << WEIGHT_SCALE_BITS));
      output[pos_idx] = sum + skip_val;
    }
  }
}

// ============================================================================
// Vectorized Feature Transform (int4 loads)
// ============================================================================

kernel void feature_transform_vec4(device const weight_t *weights [[buffer(0)]],
                                   device const weight_t *biases [[buffer(1)]],
                                   device const int32_t *features [[buffer(2)]],
                                   device const uint32_t *feature_counts
                                   [[buffer(3)]],
                                   device accumulator_t *accumulators
                                   [[buffer(4)]],
                                   constant uint &hidden_dim [[buffer(5)]],
                                   constant uint &batch_size [[buffer(6)]],
                                   constant uint &max_features [[buffer(7)]],
                                   uint2 gid [[thread_position_in_grid]]) {
  uint pos_idx = gid.y;
  uint vec_idx = gid.x;
  uint hidden_base = vec_idx * 4;

  if (pos_idx >= batch_size || hidden_base >= hidden_dim)
    return;

  // Load 4 biases
  int4 acc;
  if (hidden_base + 3 < hidden_dim) {
    short4 bias_vec =
        *reinterpret_cast<device const short4 *>(biases + hidden_base);
    acc = int4(bias_vec);
  } else {
    acc = int4(hidden_base < hidden_dim ? biases[hidden_base] : 0,
               hidden_base + 1 < hidden_dim ? biases[hidden_base + 1] : 0,
               hidden_base + 2 < hidden_dim ? biases[hidden_base + 2] : 0,
               hidden_base + 3 < hidden_dim ? biases[hidden_base + 3] : 0);
  }

  uint count = feature_counts[pos_idx];
  device const int32_t *pos_features = features + pos_idx * max_features;

  for (uint i = 0; i < count; i++) {
    int32_t feat_idx = pos_features[i];
    if (feat_idx >= 0 && feat_idx < int32_t(HALFKA_DIMS)) {
      uint weight_base = feat_idx * hidden_dim + hidden_base;
      if (hidden_base + 3 < hidden_dim) {
        short4 w =
            *reinterpret_cast<device const short4 *>(weights + weight_base);
        acc += int4(w);
      } else {
        if (hidden_base < hidden_dim)
          acc.x += weights[weight_base];
        if (hidden_base + 1 < hidden_dim)
          acc.y += weights[weight_base + 1];
        if (hidden_base + 2 < hidden_dim)
          acc.z += weights[weight_base + 2];
        if (hidden_base + 3 < hidden_dim)
          acc.w += weights[weight_base + 3];
      }
    }
  }

  // Store results
  uint out_base = pos_idx * hidden_dim + hidden_base;
  if (hidden_base + 3 < hidden_dim) {
    *reinterpret_cast<device int4 *>(accumulators + out_base) = acc;
  } else {
    if (hidden_base < hidden_dim)
      accumulators[out_base] = acc.x;
    if (hidden_base + 1 < hidden_dim)
      accumulators[out_base + 1] = acc.y;
    if (hidden_base + 2 < hidden_dim)
      accumulators[out_base + 2] = acc.z;
    if (hidden_base + 3 < hidden_dim)
      accumulators[out_base + 3] = acc.w;
  }
}

// ============================================================================
// Finny Table Update Kernel
// Updates Finny table entry after computing accumulator
// ============================================================================

kernel void update_finny_entry(device const accumulator_t *src_acc
                               [[buffer(0)]],
                               device accumulator_t *finny_acc [[buffer(1)]],
                               device int32_t *finny_psqt [[buffer(2)]],
                               device const int32_t *src_psqt [[buffer(3)]],
                               constant uint &hidden_dim [[buffer(4)]],
                               constant uint &king_square [[buffer(5)]],
                               uint gid [[thread_position_in_grid]]) {
  if (gid < hidden_dim) {
    finny_acc[king_square * hidden_dim + gid] = src_acc[gid];
  }
  if (gid < PSQT_BUCKETS) {
    finny_psqt[king_square * PSQT_BUCKETS + gid] = src_psqt[gid];
  }
}

// ============================================================================
// Load From Finny Table Kernel
// Loads cached accumulator from Finny table
// ============================================================================

kernel void load_from_finny(device const accumulator_t *finny_acc [[buffer(0)]],
                            device const int32_t *finny_psqt [[buffer(1)]],
                            device accumulator_t *dst_acc [[buffer(2)]],
                            device int32_t *dst_psqt [[buffer(3)]],
                            constant uint &hidden_dim [[buffer(4)]],
                            constant uint &king_square [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid < hidden_dim) {
    dst_acc[gid] = finny_acc[king_square * hidden_dim + gid];
  }
  if (gid < PSQT_BUCKETS) {
    dst_psqt[gid] = finny_psqt[king_square * PSQT_BUCKETS + gid];
  }
}

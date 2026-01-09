/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Metal Compute Shaders for NNUE Evaluation

  These kernels implement the NNUE neural network inference on GPU.
  Optimized for Apple Silicon's unified memory architecture.
*/

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Types
// ============================================================================

// NNUE architecture dimensions
constant int FT_DIM_BIG = 1024;
constant int FT_DIM_SMALL = 128;
constant int FC0_OUT = 15;
constant int FC1_OUT = 32;
constant int PSQT_BUCKETS = 8;
constant int LAYER_STACKS = 8;

// Quantization parameters
constant int WEIGHT_SCALE = 64; // 2^6
constant int OUTPUT_SCALE = 16;

// Type aliases for clarity
typedef int16_t weight_t;
typedef int8_t activation_t;
typedef int32_t accumulator_t;

// ============================================================================
// Activation Functions
// ============================================================================

// Clipped ReLU: clamp to [0, 127]
inline int8_t clipped_relu(int16_t x) { return int8_t(clamp(int(x), 0, 127)); }

// Squared Clipped ReLU: (clamp(x, 0, 127))^2 / 128
inline int8_t sqr_clipped_relu(int16_t x) {
  int clamped = clamp(int(x), 0, 127);
  return int8_t((clamped * clamped) >> 7);
}

// ============================================================================
// Feature Transformer Kernels
// ============================================================================

/**
 * Full feature transform from scratch
 *
 * Computes accumulator values by summing weights for all active features.
 * This is used when no incremental update is possible.
 *
 * Unified Memory Advantage: Feature weights are accessed directly without
 * explicit CPU->GPU copy. The GPU reads from the same physical memory.
 */
kernel void feature_transform_full(
    device const weight_t *weights [[buffer(0)]], // [num_features x hidden_dim]
    device const weight_t *biases [[buffer(1)]],  // [hidden_dim]
    device const int32_t *features [[buffer(2)]], // Active feature indices
    device const int32_t *feature_offsets
    [[buffer(3)]], // Start offset per position
    device const int32_t *feature_counts [[buffer(4)]], // Count per position
    device accumulator_t *output [[buffer(5)]],         // [batch x hidden_dim]
    constant int &hidden_dim [[buffer(6)]],
    constant int &batch_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tg_size [[threads_per_threadgroup]]) {
  int pos_idx = gid.y;
  int hidden_idx = gid.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  // Start with bias
  accumulator_t acc = accumulator_t(biases[hidden_idx]);

  // Get feature range for this position
  int start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  int end = feature_offsets[pos_idx];

  // Accumulate contributions from active features
  // Each feature adds its corresponding weight column
  for (int i = start; i < end; i++) {
    int feature_idx = features[i];
    acc += weights[feature_idx * hidden_dim + hidden_idx];
  }

  output[pos_idx * hidden_dim + hidden_idx] = acc;
}

/**
 * Incremental feature update
 *
 * When making a move, only a few features change (typically 2-4).
 * This kernel efficiently updates the accumulator by adding/removing
 * only the changed features.
 *
 * This is the key optimization for NNUE - most positions share
 * most features with their parent position.
 */
kernel void feature_transform_incremental(
    device const weight_t *weights [[buffer(0)]],
    device const int32_t *added [[buffer(1)]],   // Features to add
    device const int32_t *removed [[buffer(2)]], // Features to remove
    device const int32_t *update_counts
    [[buffer(3)]], // [num_added, num_removed] per pos
    device accumulator_t *accumulator [[buffer(4)]], // In/out accumulator
    constant int &hidden_dim [[buffer(5)]],
    constant int &batch_size [[buffer(6)]],
    constant int &max_updates [[buffer(7)]], // Max features per update
    uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int hidden_idx = gid.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  int num_added = update_counts[pos_idx * 2];
  int num_removed = update_counts[pos_idx * 2 + 1];

  accumulator_t acc = accumulator[pos_idx * hidden_dim + hidden_idx];

  // Remove old features
  int remove_base = pos_idx * max_updates;
  for (int i = 0; i < num_removed; i++) {
    int feature_idx = removed[remove_base + i];
    if (feature_idx >= 0) {
      acc -= weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  // Add new features
  int add_base = pos_idx * max_updates;
  for (int i = 0; i < num_added; i++) {
    int feature_idx = added[add_base + i];
    if (feature_idx >= 0) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  accumulator[pos_idx * hidden_dim + hidden_idx] = acc;
}

// ============================================================================
// Fully Connected Layer Kernels
// ============================================================================

/**
 * First FC layer with ClippedReLU input transformation
 *
 * Takes the raw accumulator and applies clipping before matrix multiply.
 * Outputs both squared and linear activations for the skip connection.
 */
kernel void fc0_layer(
    device const accumulator_t *input [[buffer(0)]], // [batch x 2 x hidden_dim]
    device const weight_t *weights [[buffer(1)]], // [hidden_dim x (fc0_out+1)]
    device const weight_t *biases [[buffer(2)]],  // [fc0_out+1]
    device int8_t *output_sqr [[buffer(3)]],      // SqrClippedReLU output
    device int8_t *output_clip [[buffer(4)]],     // ClippedReLU output (skip)
    constant int &hidden_dim [[buffer(5)]],
    constant int &batch_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]]) {
  int pos_idx = gid.y;
  int out_idx = gid.x;

  if (pos_idx >= batch_size || out_idx > FC0_OUT)
    return;

  // Process both perspectives (white and black)
  for (int perspective = 0; perspective < 2; perspective++) {
    accumulator_t acc = biases[out_idx];

    device const accumulator_t *in_ptr =
        input + pos_idx * 2 * hidden_dim + perspective * hidden_dim;

    // Dot product with clipped input
    for (int i = 0; i < hidden_dim; i++) {
      int8_t clipped = clipped_relu(int16_t(in_ptr[i] >> 6));
      acc += clipped * weights[i * (FC0_OUT + 1) + out_idx];
    }

    int16_t result = int16_t(acc >> 6);

    int base = pos_idx * 2 * (FC0_OUT + 1) + perspective * (FC0_OUT + 1);
    output_sqr[base + out_idx] = sqr_clipped_relu(result);
    output_clip[base + out_idx] = clipped_relu(result);
  }
}

/**
 * Generic FC layer with ClippedReLU activation
 */
kernel void fc_layer(device const int8_t *input [[buffer(0)]],
                     device const weight_t *weights [[buffer(1)]],
                     device const weight_t *biases [[buffer(2)]],
                     device int8_t *output [[buffer(3)]],
                     constant int &input_dim [[buffer(4)]],
                     constant int &output_dim [[buffer(5)]],
                     constant int &batch_size [[buffer(6)]],
                     uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int out_idx = gid.x;

  if (pos_idx >= batch_size || out_idx >= output_dim)
    return;

  accumulator_t acc = biases[out_idx];

  device const int8_t *in_ptr = input + pos_idx * input_dim;

  for (int i = 0; i < input_dim; i++) {
    acc += in_ptr[i] * weights[i * output_dim + out_idx];
  }

  output[pos_idx * output_dim + out_idx] = clipped_relu(int16_t(acc >> 6));
}

/**
 * Final output layer - produces evaluation score
 */
kernel void output_layer(device const int8_t *fc1_out [[buffer(0)]],
                         device const weight_t *weights [[buffer(1)]],
                         device const weight_t *biases [[buffer(2)]],
                         device const int8_t *skip
                         [[buffer(3)]], // Skip connection from FC0
                         device int32_t *output [[buffer(4)]],
                         constant int &fc1_dim [[buffer(5)]],
                         constant int &batch_size [[buffer(6)]],
                         uint gid [[thread_position_in_grid]]) {
  int pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  accumulator_t acc = biases[0];

  // FC2 dot product
  device const int8_t *in_ptr = fc1_out + pos_idx * fc1_dim;
  for (int i = 0; i < fc1_dim; i++) {
    acc += in_ptr[i] * weights[i];
  }

  // Add skip connection contribution
  // The skip output is scaled differently
  int skip_idx = pos_idx * 2 * (FC0_OUT + 1) + FC0_OUT; // Last element
  int32_t skip_val =
      (skip[skip_idx] * 600 * OUTPUT_SCALE) / (127 * WEIGHT_SCALE);

  output[pos_idx] = acc + skip_val;
}

// ============================================================================
// Fused NNUE Forward Pass
// ============================================================================

/**
 * Complete NNUE forward pass for a batch of positions
 *
 * This kernel fuses all layers for maximum efficiency.
 * Each threadgroup processes one position.
 *
 * Input: Pre-computed accumulators [batch x 2 x hidden_dim]
 * Output: Evaluation scores [batch]
 */
kernel void nnue_forward_fused(device const accumulator_t *accumulators
                               [[buffer(0)]],
                               device const weight_t *fc0_weights [[buffer(1)]],
                               device const weight_t *fc0_biases [[buffer(2)]],
                               device const weight_t *fc1_weights [[buffer(3)]],
                               device const weight_t *fc1_biases [[buffer(4)]],
                               device const weight_t *fc2_weights [[buffer(5)]],
                               device const weight_t *fc2_biases [[buffer(6)]],
                               device int32_t *output [[buffer(7)]],
                               constant int &hidden_dim [[buffer(8)]],
                               constant int &batch_size [[buffer(9)]],
                               uint gid [[threadgroup_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]]) {
  int pos_idx = gid;
  if (pos_idx >= batch_size)
    return;

  // Shared memory for intermediate results
  threadgroup int8_t fc0_sqr[2 * FC0_OUT];
  threadgroup int8_t fc0_skip[2];
  threadgroup int8_t fc1_out[FC1_OUT];

  device const accumulator_t *white_acc =
      accumulators + pos_idx * 2 * hidden_dim;
  device const accumulator_t *black_acc = white_acc + hidden_dim;

  // ========== FC0 Layer ==========
  // Parallel across output neurons
  for (int out = lid; out <= FC0_OUT; out += tg_size) {
    for (int p = 0; p < 2; p++) {
      device const accumulator_t *acc = (p == 0) ? white_acc : black_acc;

      accumulator_t sum = fc0_biases[out];
      for (int i = 0; i < hidden_dim; i++) {
        int8_t clipped = clipped_relu(int16_t(acc[i] >> 6));
        sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
      }

      int16_t result = int16_t(sum >> 6);

      if (out < FC0_OUT) {
        fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
      } else {
        fc0_skip[p] = clipped_relu(result);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ========== FC1 Layer ==========
  // Input is concatenated [sqr_white, sqr_black]
  for (int out = lid; out < FC1_OUT; out += tg_size) {
    accumulator_t sum = fc1_biases[out];

    for (int i = 0; i < 2 * FC0_OUT; i++) {
      sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
    }

    fc1_out[out] = clipped_relu(int16_t(sum >> 6));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ========== FC2 Layer (Output) ==========
  // Only one thread computes final output
  if (lid == 0) {
    accumulator_t sum = fc2_biases[0];

    for (int i = 0; i < FC1_OUT; i++) {
      sum += fc1_out[i] * fc2_weights[i];
    }

    // Add skip connection (average of both perspectives)
    int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * OUTPUT_SCALE) /
                       (2 * 127 * WEIGHT_SCALE);

    output[pos_idx] = sum + skip_val;
  }
}

// ============================================================================
// PSQT Evaluation Kernel
// ============================================================================

/**
 * PSQT (Piece-Square Table) evaluation
 *
 * Computes the material/positional component of the evaluation.
 * This runs in parallel with the positional evaluation.
 */
kernel void psqt_eval(device const int32_t *features [[buffer(0)]],
                      device const int32_t *feature_offsets [[buffer(1)]],
                      device const weight_t *psqt_weights
                      [[buffer(2)]], // [num_features x buckets]
                      device int32_t *output [[buffer(3)]], // [batch x buckets]
                      constant int &num_buckets [[buffer(4)]],
                      constant int &batch_size [[buffer(5)]],
                      uint2 gid [[thread_position_in_grid]]) {
  int pos_idx = gid.y;
  int bucket = gid.x;

  if (pos_idx >= batch_size || bucket >= num_buckets)
    return;

  int start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
  int end = feature_offsets[pos_idx];

  int32_t acc = 0;
  for (int i = start; i < end; i++) {
    int feature_idx = features[i];
    acc += psqt_weights[feature_idx * num_buckets + bucket];
  }

  output[pos_idx * num_buckets + bucket] = acc;
}

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * Copy accumulator with perspective swap
 * Used when the side to move changes
 */
kernel void copy_accumulator_swapped(device const accumulator_t *src
                                     [[buffer(0)]],
                                     device accumulator_t *dst [[buffer(1)]],
                                     constant int &hidden_dim [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]]) {
  int idx = gid;
  if (idx >= hidden_dim * 2)
    return;

  // Swap white and black perspectives
  int perspective = idx / hidden_dim;
  int offset = idx % hidden_dim;
  int swapped_perspective = 1 - perspective;

  dst[perspective * hidden_dim + offset] =
      src[swapped_perspective * hidden_dim + offset];
}

/**
 * Zero-initialize buffer
 */
kernel void zero_buffer(device int32_t *buffer [[buffer(0)]],
                        constant int &count [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < uint(count)) {
    buffer[gid] = 0;
  }
}

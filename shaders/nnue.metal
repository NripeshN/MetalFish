/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  MetalFish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.

  NNUE (Efficiently Updatable Neural Network) evaluation kernels for Metal.

  The NNUE architecture:
  - Input: 45056 features (HalfKAv2_hm) -> 1024 hidden units (big) or 128
  (small)
  - Hidden layers with ClippedReLU activation
  - Output: single scalar evaluation

  For GPU, we batch multiple positions and process them in parallel.
*/

#include <metal_stdlib>
using namespace metal;

// Constants matching Stockfish NNUE architecture
constant int FEATURE_DIM_BIG = 1024;
constant int FEATURE_DIM_SMALL = 128;
constant int FC0_OUT = 15;
constant int FC1_OUT = 32;
constant int WEIGHT_SCALE_BITS = 6;
constant int OUTPUT_SCALE = 16;

// Type aliases
typedef int16_t weight_t; // Quantized weights
typedef int8_t clipped_t; // ClippedReLU output
typedef int32_t acc_t;    // Accumulator type

// Clipped ReLU activation: clamp to [0, 127] then shift
inline int8_t clipped_relu(int16_t x) {
  return int8_t(clamp(int(x) >> WEIGHT_SCALE_BITS, 0, 127));
}

// Squared clipped ReLU: (clamp(x, 0, 127))^2 / 128
inline int8_t sqr_clipped_relu(int16_t x) {
  int clamped = clamp(int(x) >> WEIGHT_SCALE_BITS, 0, 127);
  return int8_t((clamped * clamped) >> 7);
}

/**
 * Feature transformer kernel: Applies sparse feature multiplication
 *
 * This kernel processes the first layer of NNUE which transforms
 * active features into hidden layer activations.
 *
 * On unified memory, the feature weights and accumulators can be
 * accessed directly without explicit copies.
 */
kernel void feature_transform(device const weight_t *weights
                              [[buffer(0)]], // [num_features x hidden_dim]
                              device const int32_t *active_features
                              [[buffer(1)]], // Active feature indices
                              device const int32_t *feature_counts
                              [[buffer(2)]], // Number of features per position
                              device const weight_t *biases
                              [[buffer(3)]], // [hidden_dim]
                              device acc_t *output
                              [[buffer(4)]], // [batch_size x hidden_dim]
                              constant int &hidden_dim [[buffer(5)]],
                              constant int &batch_size [[buffer(6)]],
                              uint2 tid [[thread_position_in_grid]],
                              uint2 tg_size [[threads_per_threadgroup]]) {
  int pos_idx = tid.y;
  int hidden_idx = tid.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  // Start with bias
  acc_t acc = biases[hidden_idx];

  // Get feature range for this position
  int feature_start = (pos_idx > 0) ? feature_counts[pos_idx - 1] : 0;
  int feature_end = feature_counts[pos_idx];

  // Accumulate contributions from active features
  for (int i = feature_start; i < feature_end; i++) {
    int feature_idx = active_features[i];
    acc += weights[feature_idx * hidden_dim + hidden_idx];
  }

  output[pos_idx * hidden_dim + hidden_idx] = acc;
}

/**
 * Incremental feature update kernel
 *
 * When making a move, only a few features change. This kernel
 * efficiently updates the accumulator by adding/removing specific features.
 */
kernel void
incremental_update(device const weight_t *weights [[buffer(0)]],
                   device const int32_t *added_features [[buffer(1)]],
                   device const int32_t *removed_features [[buffer(2)]],
                   device const int32_t *update_counts
                   [[buffer(3)]], // [num_added, num_removed] per position
                   device acc_t *accumulator [[buffer(4)]],
                   constant int &hidden_dim [[buffer(5)]],
                   constant int &batch_size [[buffer(6)]],
                   uint2 tid [[thread_position_in_grid]]) {
  int pos_idx = tid.y;
  int hidden_idx = tid.x;

  if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
    return;

  int num_added = update_counts[pos_idx * 2];
  int num_removed = update_counts[pos_idx * 2 + 1];

  acc_t acc = accumulator[pos_idx * hidden_dim + hidden_idx];

  // Remove old features
  for (int i = 0; i < num_removed; i++) {
    int feature_idx =
        removed_features[pos_idx * 32 + i]; // Max 32 features per update
    if (feature_idx >= 0) {
      acc -= weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  // Add new features
  for (int i = 0; i < num_added; i++) {
    int feature_idx = added_features[pos_idx * 32 + i];
    if (feature_idx >= 0) {
      acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
  }

  accumulator[pos_idx * hidden_dim + hidden_idx] = acc;
}

/**
 * First fully connected layer with sparse input (after feature transform)
 * Uses affine transform with ClippedReLU and SqrClippedReLU outputs
 */
kernel void
fc0_sparse(device const acc_t *input
           [[buffer(0)]], // Transformed features [batch x hidden_dim]
           device const weight_t *weights
           [[buffer(1)]], // [hidden_dim x (fc0_out + 1)]
           device const weight_t *biases [[buffer(2)]], // [fc0_out + 1]
           device int8_t *output_sqr [[buffer(3)]],     // SqrClippedReLU output
           device int8_t *output_clipped [[buffer(4)]], // ClippedReLU output
           constant int &hidden_dim [[buffer(5)]],
           constant int &batch_size [[buffer(6)]],
           uint2 tid [[thread_position_in_grid]],
           uint tg_idx [[threadgroup_position_in_grid]]) {
  int pos_idx = tid.y;
  int out_idx = tid.x;

  if (pos_idx >= batch_size || out_idx >= FC0_OUT + 1)
    return;

  threadgroup int16_t shared_input[FEATURE_DIM_BIG];

  // Collaboratively load input to shared memory
  int input_base = pos_idx * hidden_dim;
  for (int i = tid.x; i < hidden_dim; i += 32) {
    if (i < hidden_dim) {
      shared_input[i] = clipped_relu(input[input_base + i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Compute dot product
  acc_t acc = biases[out_idx];
  for (int i = 0; i < hidden_dim; i++) {
    acc += shared_input[i] * weights[i * (FC0_OUT + 1) + out_idx];
  }

  // Apply activations
  int16_t result = int16_t(acc >> WEIGHT_SCALE_BITS);
  output_sqr[pos_idx * (FC0_OUT + 1) + out_idx] = sqr_clipped_relu(result);
  output_clipped[pos_idx * (FC0_OUT + 1) + out_idx] = clipped_relu(result);
}

/**
 * Standard fully connected layer (FC1 and FC2)
 */
kernel void fc_layer(device const int8_t *input [[buffer(0)]],
                     device const weight_t *weights [[buffer(1)]],
                     device const weight_t *biases [[buffer(2)]],
                     device int8_t *output [[buffer(3)]],
                     constant int &input_dim [[buffer(4)]],
                     constant int &output_dim [[buffer(5)]],
                     constant int &batch_size [[buffer(6)]],
                     constant bool &apply_relu [[buffer(7)]],
                     uint2 tid [[thread_position_in_grid]]) {
  int pos_idx = tid.y;
  int out_idx = tid.x;

  if (pos_idx >= batch_size || out_idx >= output_dim)
    return;

  acc_t acc = biases[out_idx];

  int input_base = pos_idx * input_dim;
  for (int i = 0; i < input_dim; i++) {
    acc += input[input_base + i] * weights[i * output_dim + out_idx];
  }

  if (apply_relu) {
    output[pos_idx * output_dim + out_idx] =
        clipped_relu(int16_t(acc >> WEIGHT_SCALE_BITS));
  } else {
    // Final layer - store raw value (will be read as int32 later)
    output[pos_idx * output_dim + out_idx] = int8_t(acc >> WEIGHT_SCALE_BITS);
  }
}

/**
 * Final output layer - produces evaluation score
 */
kernel void output_layer(device const int8_t *fc1_output [[buffer(0)]],
                         device const weight_t *weights [[buffer(1)]],
                         device const weight_t *biases [[buffer(2)]],
                         device const int16_t *fc0_skip
                         [[buffer(3)]], // Skip connection from FC0
                         device int32_t *output [[buffer(4)]],
                         constant int &input_dim [[buffer(5)]],
                         constant int &batch_size [[buffer(6)]],
                         uint tid [[thread_position_in_grid]]) {
  int pos_idx = tid;
  if (pos_idx >= batch_size)
    return;

  acc_t acc = biases[0];

  // FC2 computation
  int input_base = pos_idx * input_dim;
  for (int i = 0; i < input_dim; i++) {
    acc += fc1_output[input_base + i] * weights[i];
  }

  // Add skip connection from FC0 (the extra output)
  int32_t fwd_out = (fc0_skip[pos_idx] * (600 * OUTPUT_SCALE)) /
                    (127 * (1 << WEIGHT_SCALE_BITS));

  output[pos_idx] = acc + fwd_out;
}

/**
 * Batched NNUE evaluation - complete forward pass for multiple positions
 *
 * This kernel combines all layers for maximum GPU efficiency.
 * Uses shared memory for intermediate results within a threadgroup.
 */
kernel void nnue_forward_batch(
    device const acc_t *transformed
    [[buffer(0)]], // Pre-computed accumulators [batch x 2 x hidden_dim]
    device const weight_t *fc0_weights [[buffer(1)]],
    device const weight_t *fc0_biases [[buffer(2)]],
    device const weight_t *fc1_weights [[buffer(3)]],
    device const weight_t *fc1_biases [[buffer(4)]],
    device const weight_t *fc2_weights [[buffer(5)]],
    device const weight_t *fc2_biases [[buffer(6)]],
    device int32_t *output [[buffer(7)]],
    constant int &hidden_dim [[buffer(8)]],
    constant int &batch_size [[buffer(9)]],
    uint pos_idx [[thread_position_in_grid]],
    uint local_idx [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
  if (pos_idx >= (uint)batch_size)
    return;

  // Allocate shared memory for intermediate results
  threadgroup int8_t fc0_sqr[FC0_OUT * 2];
  threadgroup int8_t fc0_clip[FC0_OUT + 1];
  threadgroup int8_t fc1_out[FC1_OUT];

  // Get perspectives (white and black accumulators)
  device const acc_t *white_acc = transformed + pos_idx * 2 * hidden_dim;
  device const acc_t *black_acc = white_acc + hidden_dim;

  // FC0 layer (parallel across outputs)
  for (int out = local_idx; out < FC0_OUT + 1; out += tg_size) {
    acc_t acc_w = fc0_biases[out];
    acc_t acc_b = fc0_biases[out];

    for (int i = 0; i < hidden_dim; i++) {
      int8_t w_val = clipped_relu(white_acc[i]);
      int8_t b_val = clipped_relu(black_acc[i]);
      acc_w += w_val * fc0_weights[i * (FC0_OUT + 1) + out];
      acc_b += b_val * fc0_weights[i * (FC0_OUT + 1) + out];
    }

    int16_t result_w = int16_t(acc_w >> WEIGHT_SCALE_BITS);
    int16_t result_b = int16_t(acc_b >> WEIGHT_SCALE_BITS);

    if (out < FC0_OUT) {
      fc0_sqr[out] = sqr_clipped_relu(result_w);
      fc0_sqr[FC0_OUT + out] = sqr_clipped_relu(result_b);
    }
    fc0_clip[out] = clipped_relu(result_w);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Concatenate sqr and clip outputs for FC1 input
  // Input is [sqr_white, sqr_black, clip_white, clip_black] = FC0_OUT * 2 * 2

  // FC1 layer
  for (int out = local_idx; out < FC1_OUT; out += tg_size) {
    acc_t acc = fc1_biases[out];

    for (int i = 0; i < FC0_OUT * 2; i++) {
      acc += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
    }

    fc1_out[out] = clipped_relu(int16_t(acc >> WEIGHT_SCALE_BITS));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // FC2 layer (only thread 0 computes final output)
  if (local_idx == 0) {
    acc_t acc = fc2_biases[0];

    for (int i = 0; i < FC1_OUT; i++) {
      acc += fc1_out[i] * fc2_weights[i];
    }

    // Add skip connection
    int32_t fwd_out = (fc0_clip[FC0_OUT] * (600 * OUTPUT_SCALE)) /
                      (127 * (1 << WEIGHT_SCALE_BITS));

    output[pos_idx] = acc + fwd_out;
  }
}

/**
 * PSQT (Piece Square Table) evaluation kernel
 * Computes material and positional bonuses in parallel
 */
kernel void psqt_eval(device const int32_t *active_features [[buffer(0)]],
                      device const int32_t *feature_counts [[buffer(1)]],
                      device const int16_t *psqt_weights
                      [[buffer(2)]], // [num_features x buckets]
                      device int32_t *output [[buffer(3)]],
                      constant int &num_buckets [[buffer(4)]],
                      constant int &batch_size [[buffer(5)]],
                      uint2 tid [[thread_position_in_grid]]) {
  int pos_idx = tid.y;
  int bucket_idx = tid.x;

  if (pos_idx >= batch_size || bucket_idx >= num_buckets)
    return;

  int feature_start = (pos_idx > 0) ? feature_counts[pos_idx - 1] : 0;
  int feature_end = feature_counts[pos_idx];

  int32_t acc = 0;
  for (int i = feature_start; i < feature_end; i++) {
    int feature_idx = active_features[i];
    acc += psqt_weights[feature_idx * num_buckets + bucket_idx];
  }

  output[pos_idx * num_buckets + bucket_idx] = acc;
}

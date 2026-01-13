/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Integration Implementation

  This module provides GPU-accelerated NNUE evaluation using Metal compute
  shaders. Key features:
  - Adaptive kernel selection based on batch size
  - Dual-perspective feature transform for efficiency
  - SIMD-optimized forward pass for large batches
  - Runtime tuning based on observed performance
*/

#include "gpu_nnue_integration.h"

#ifdef USE_METAL

#include "backend.h"
#include "core/bitboard.h"
#include "core/position.h"
#include "eval/nnue/network.h"
#include "eval/nnue/nnue_architecture.h"
#include "eval/nnue/nnue_feature_transformer.h"
#include "nnue_weight_accessor.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>

namespace MetalFish::GPU {

// ============================================================================
// GPUTuningParams Implementation
// ============================================================================

EvalStrategy GPUTuningParams::select_strategy(int batch_size) const {
  if (batch_size < min_batch_for_gpu) {
    return EvalStrategy::CPU_FALLBACK;
  }

  // Calculate expected costs
  // CPU cost: batch_size * cpu_eval_ns (nanoseconds)
  // GPU cost: gpu_dispatch_us * 1000 (nanoseconds) + batch_size *
  // marginal_gpu_cost

  // For small batches, dispatch overhead dominates - use standard kernels
  // For large batches, compute dominates - use SIMD kernels

  if (batch_size >= gpu_extract_threshold) {
    return EvalStrategy::GPU_FEATURE_EXTRACT;
  } else if (batch_size >= simd_threshold) {
    return EvalStrategy::GPU_SIMD;
  } else {
    return EvalStrategy::GPU_STANDARD;
  }
}

// ============================================================================
// GPUPositionData Implementation
// ============================================================================

void GPUPositionData::from_position(const Position &pos) {
  std::memset(this, 0, sizeof(GPUPositionData));

  // Copy piece bitboards
  for (int c = 0; c < 2; c++) {
    for (int pt = 0; pt <= 6; pt++) {
      pieces[c][pt] = pos.pieces(Color(c), PieceType(pt));
    }
  }

  king_sq[0] = pos.square<KING>(WHITE);
  king_sq[1] = pos.square<KING>(BLACK);
  stm = pos.side_to_move();
  piece_count = pos.count<ALL_PIECES>();
}

// ============================================================================
// GPUNetworkData Implementation
// ============================================================================

size_t GPUNetworkData::memory_usage() const {
  size_t total = 0;
  if (ft_weights)
    total += ft_weights->size();
  if (ft_biases)
    total += ft_biases->size();
  if (ft_psqt)
    total += ft_psqt->size();
  if (threat_weights)
    total += threat_weights->size();
  if (threat_psqt)
    total += threat_psqt->size();

  for (const auto &layer : layers) {
    if (layer.fc0_weights)
      total += layer.fc0_weights->size();
    if (layer.fc0_biases)
      total += layer.fc0_biases->size();
    if (layer.fc1_weights)
      total += layer.fc1_weights->size();
    if (layer.fc1_biases)
      total += layer.fc1_biases->size();
    if (layer.fc2_weights)
      total += layer.fc2_weights->size();
    if (layer.fc2_biases)
      total += layer.fc2_biases->size();
  }

  return total;
}

// ============================================================================
// GPUEvalBatch Implementation
// ============================================================================

void GPUEvalBatch::clear() {
  positions.clear();
  white_features.clear();
  black_features.clear();
  feature_counts.clear();
  feature_offsets.clear();
  buckets.clear();
  psqt_scores.clear();
  positional_scores.clear();
  count = 0;
}

void GPUEvalBatch::reserve(int n) {
  positions.reserve(n);
  white_features.reserve(n * GPU_MAX_FEATURES_PER_PERSPECTIVE);
  black_features.reserve(n * GPU_MAX_FEATURES_PER_PERSPECTIVE);
  feature_counts.reserve(n * 2);
  feature_offsets.reserve(n);
  buckets.reserve(n);
  psqt_scores.resize(n);
  positional_scores.resize(n);
}

void GPUEvalBatch::add_position(const Position &pos) {
  GPUPositionData data;
  data.from_position(pos);
  positions.push_back(data);

  // Calculate bucket based on piece count (8 buckets, 4 pieces per bucket)
  int bucket = (pos.count<ALL_PIECES>() - 1) / 4;
  bucket = std::clamp(bucket, 0, GPU_LAYER_STACKS - 1);
  buckets.push_back(bucket);

  feature_offsets.push_back(white_features.size());
  count++;
}

// ============================================================================
// Metal Shader Source
//
// Contains GPU compute kernels for NNUE evaluation:
// - Feature transform: Converts sparse features to dense accumulators
// - Forward pass: Evaluates the neural network layers
// - PSQT accumulation: Computes piece-square table scores
// ============================================================================

static const char *GPU_NNUE_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// Network architecture constants
constant uint FC0_OUT = 15;
constant uint FC1_OUT = 32;
constant uint WEIGHT_SCALE_BITS = 6;
constant uint OUTPUT_SCALE = 16;
constant uint SIMDGROUP_SIZE = 32;
constant uint MAX_FEATURES_PER_PERSPECTIVE = 64;

// Data types matching CPU implementation
typedef int16_t weight_t;
typedef int8_t layer_weight_t;
typedef int32_t accumulator_t;

// Vector types for coalesced memory access
typedef short2 weight2_t;
typedef short4 weight4_t;
typedef int2 acc2_t;
typedef int4 acc4_t;
typedef char4 clipped4_t;

// Activation function: clamp to [0, 127]
inline int8_t clipped_relu(int16_t x) {
    return int8_t(clamp(int(x), 0, 127));
}

// Squared activation: (clamp(x, 0, 127))^2 / 128
inline int8_t sqr_clipped_relu(int16_t x) {
    int clamped = clamp(int(x), 0, 127);
    return int8_t((clamped * clamped) >> 7);
}

// Vectorized activation for 4 values
inline char4 clipped_relu4(short4 x) {
    return char4(clamp(int4(x), 0, 127));
}

// Position data structure (matches CPU GPUPositionData)
struct GPUPositionData {
    uint64_t pieces[2][7];
    uint8_t king_sq[2];
    uint8_t stm;
    uint8_t piece_count;
    uint8_t padding[4];
};

// Feature Transform Kernel
// Transforms sparse feature indices into dense accumulator values.
// Each thread processes one hidden dimension for one position.
// Memory access pattern: weights[feature_idx * hidden_dim + hidden_idx]
// provides coalesced reads when threads in a simdgroup access consecutive hidden_idx.
// Feature indices are guaranteed valid by CPU extraction, so no bounds checks needed.
kernel void gpu_feature_transform(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const int32_t* features [[buffer(2)]],
    device const uint32_t* feature_counts [[buffer(3)]],
    device const uint32_t* feature_offsets [[buffer(4)]],
    device accumulator_t* accumulators [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    
    uint pos_idx = gid.y;
    uint hidden_idx = gid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
        return;
    
    accumulator_t acc = accumulator_t(biases[hidden_idx]);
    
    uint start = feature_offsets[pos_idx];
    uint count = feature_counts[pos_idx];
    
    // 8-way unrolled loop for maximum throughput
    uint i = 0;
    for (; i + 7 < count; i += 8) {
        int32_t f0 = features[start + i];
        int32_t f1 = features[start + i + 1];
        int32_t f2 = features[start + i + 2];
        int32_t f3 = features[start + i + 3];
        int32_t f4 = features[start + i + 4];
        int32_t f5 = features[start + i + 5];
        int32_t f6 = features[start + i + 6];
        int32_t f7 = features[start + i + 7];
        
        acc += weights[f0 * hidden_dim + hidden_idx];
        acc += weights[f1 * hidden_dim + hidden_idx];
        acc += weights[f2 * hidden_dim + hidden_idx];
        acc += weights[f3 * hidden_dim + hidden_idx];
        acc += weights[f4 * hidden_dim + hidden_idx];
        acc += weights[f5 * hidden_dim + hidden_idx];
        acc += weights[f6 * hidden_dim + hidden_idx];
        acc += weights[f7 * hidden_dim + hidden_idx];
    }
    
    // Handle remaining features
    for (; i < count; i++) {
        int32_t feat_idx = features[start + i];
        acc += weights[feat_idx * hidden_dim + hidden_idx];
    }
    
    accumulators[pos_idx * hidden_dim + hidden_idx] = acc;
}

// Dual-Perspective Feature Transform
// Processes both white and black perspectives in a single kernel dispatch.
// Uses 3D grid: (hidden_dim, 2, batch_size) for perspective parallelism.
// 8-way loop unrolling maximizes instruction-level parallelism.
// Feature indices are guaranteed valid by CPU extraction, so no bounds checks needed.
kernel void gpu_feature_transform_dual(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const int32_t* white_features [[buffer(2)]],
    device const int32_t* black_features [[buffer(3)]],
    device const uint32_t* white_counts [[buffer(4)]],
    device const uint32_t* black_counts [[buffer(5)]],
    device const uint32_t* white_offsets [[buffer(6)]],
    device const uint32_t* black_offsets [[buffer(7)]],
    device accumulator_t* accumulators [[buffer(8)]],
    constant uint& hidden_dim [[buffer(9)]],
    constant uint& batch_size [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    uint pos_idx = gid.z;
    uint perspective = gid.y;  // 0 = white, 1 = black
    uint hidden_idx = gid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim || perspective >= 2)
        return;
    
    accumulator_t acc = accumulator_t(biases[hidden_idx]);
    
    // Select features based on perspective
    device const int32_t* features = (perspective == 0) ? white_features : black_features;
    device const uint32_t* counts = (perspective == 0) ? white_counts : black_counts;
    device const uint32_t* offsets = (perspective == 0) ? white_offsets : black_offsets;
    
    uint start = offsets[pos_idx];
    uint count = counts[pos_idx];
    
    // 8-way unrolled loop for maximum throughput
    uint i = 0;
    for (; i + 7 < count; i += 8) {
        int32_t f0 = features[start + i];
        int32_t f1 = features[start + i + 1];
        int32_t f2 = features[start + i + 2];
        int32_t f3 = features[start + i + 3];
        int32_t f4 = features[start + i + 4];
        int32_t f5 = features[start + i + 5];
        int32_t f6 = features[start + i + 6];
        int32_t f7 = features[start + i + 7];
        
        // No bounds check needed - CPU guarantees valid indices
        acc += weights[f0 * hidden_dim + hidden_idx];
        acc += weights[f1 * hidden_dim + hidden_idx];
        acc += weights[f2 * hidden_dim + hidden_idx];
        acc += weights[f3 * hidden_dim + hidden_idx];
        acc += weights[f4 * hidden_dim + hidden_idx];
        acc += weights[f5 * hidden_dim + hidden_idx];
        acc += weights[f6 * hidden_dim + hidden_idx];
        acc += weights[f7 * hidden_dim + hidden_idx];
    }
    
    // Handle remaining features (0-7 iterations)
    for (; i < count; i++) {
        int32_t feat_idx = features[start + i];
        acc += weights[feat_idx * hidden_dim + hidden_idx];
    }
    
    accumulators[(pos_idx * 2 + perspective) * hidden_dim + hidden_idx] = acc;
}

// Fused Forward Pass Kernel
// Computes all FC layers (FC0, FC1, FC2) in a single kernel using threadgroup memory.
// One threadgroup processes one position, with threads cooperating on layer computation.
// Uses 8-way loop unrolling in all layers for better throughput.
kernel void gpu_nnue_forward(
    device const accumulator_t* accumulators [[buffer(0)]],
    device const layer_weight_t* fc0_weights [[buffer(1)]],
    device const int32_t* fc0_biases [[buffer(2)]],
    device const layer_weight_t* fc1_weights [[buffer(3)]],
    device const int32_t* fc1_biases [[buffer(4)]],
    device const layer_weight_t* fc2_weights [[buffer(5)]],
    device const int32_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    uint pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    // Threadgroup memory for intermediate results
    threadgroup int8_t fc0_sqr[2 * 16];  // 2 perspectives × 15 outputs + padding
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out[32];
    
    device const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
    device const accumulator_t* black_acc = white_acc + hidden_dim;
    
    // FC0 layer - process both perspectives with 8-way unrolling
    for (uint out = lid; out <= FC0_OUT; out += tg_size) {
        for (uint p = 0; p < 2; p++) {
            device const accumulator_t* acc = (p == 0) ? white_acc : black_acc;
            
            int32_t sum = fc0_biases[out];
            
            // 8-way unrolled loop - no sparse check (branch divergence hurts more)
            uint i = 0;
            for (; i + 7 < hidden_dim; i += 8) {
                int8_t c0 = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                int8_t c1 = clipped_relu(int16_t(acc[i+1] >> WEIGHT_SCALE_BITS));
                int8_t c2 = clipped_relu(int16_t(acc[i+2] >> WEIGHT_SCALE_BITS));
                int8_t c3 = clipped_relu(int16_t(acc[i+3] >> WEIGHT_SCALE_BITS));
                int8_t c4 = clipped_relu(int16_t(acc[i+4] >> WEIGHT_SCALE_BITS));
                int8_t c5 = clipped_relu(int16_t(acc[i+5] >> WEIGHT_SCALE_BITS));
                int8_t c6 = clipped_relu(int16_t(acc[i+6] >> WEIGHT_SCALE_BITS));
                int8_t c7 = clipped_relu(int16_t(acc[i+7] >> WEIGHT_SCALE_BITS));
                
                sum += c0 * fc0_weights[(i) * (FC0_OUT + 1) + out];
                sum += c1 * fc0_weights[(i+1) * (FC0_OUT + 1) + out];
                sum += c2 * fc0_weights[(i+2) * (FC0_OUT + 1) + out];
                sum += c3 * fc0_weights[(i+3) * (FC0_OUT + 1) + out];
                sum += c4 * fc0_weights[(i+4) * (FC0_OUT + 1) + out];
                sum += c5 * fc0_weights[(i+5) * (FC0_OUT + 1) + out];
                sum += c6 * fc0_weights[(i+6) * (FC0_OUT + 1) + out];
                sum += c7 * fc0_weights[(i+7) * (FC0_OUT + 1) + out];
            }
            
            for (; i < hidden_dim; i++) {
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
    
    // FC1 layer - unrolled by 6 for 30 inputs (2 * FC0_OUT)
    for (uint out = lid; out < FC1_OUT; out += tg_size) {
        int32_t sum = fc1_biases[out];
        for (uint i = 0; i < 30; i += 6) {
            sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
            sum += fc0_sqr[i+1] * fc1_weights[(i+1) * FC1_OUT + out];
            sum += fc0_sqr[i+2] * fc1_weights[(i+2) * FC1_OUT + out];
            sum += fc0_sqr[i+3] * fc1_weights[(i+3) * FC1_OUT + out];
            sum += fc0_sqr[i+4] * fc1_weights[(i+4) * FC1_OUT + out];
            sum += fc0_sqr[i+5] * fc1_weights[(i+5) * FC1_OUT + out];
        }
        fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2 layer - single thread with 8-way unrolling
    if (lid == 0) {
        int32_t sum = fc2_biases[0];
        for (uint i = 0; i < 32; i += 8) {
            sum += fc1_out[i] * fc2_weights[i];
            sum += fc1_out[i+1] * fc2_weights[i+1];
            sum += fc1_out[i+2] * fc2_weights[i+2];
            sum += fc1_out[i+3] * fc2_weights[i+3];
            sum += fc1_out[i+4] * fc2_weights[i+4];
            sum += fc1_out[i+5] * fc2_weights[i+5];
            sum += fc1_out[i+6] * fc2_weights[i+6];
            sum += fc1_out[i+7] * fc2_weights[i+7];
        }
        
        // Skip connection
        int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * int32_t(OUTPUT_SCALE)) / 
                          (2 * 127 * (1 << WEIGHT_SCALE_BITS));
        
        output[pos_idx] = sum + skip_val;
    }
}

// SIMD-Accelerated Forward Pass
// Uses simdgroup operations (simd_sum) for parallel reduction.
// Each simdgroup handles one output neuron, with threads computing partial sums.
// Suitable for large batch sizes where compute time exceeds dispatch overhead.
kernel void gpu_nnue_forward_simd(
    device const accumulator_t* accumulators [[buffer(0)]],
    device const layer_weight_t* fc0_weights [[buffer(1)]],
    device const int32_t* fc0_biases [[buffer(2)]],
    device const layer_weight_t* fc1_weights [[buffer(3)]],
    device const int32_t* fc1_biases [[buffer(4)]],
    device const layer_weight_t* fc2_weights [[buffer(5)]],
    device const int32_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    
    uint pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    // Threadgroup memory
    threadgroup int8_t fc0_sqr[2 * 16];
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out[32];
    threadgroup int32_t partial_sums[8];  // For simdgroup reduction
    
    device const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
    device const accumulator_t* black_acc = white_acc + hidden_dim;
    
    // FC0 layer - each simdgroup handles one output neuron
    // With 2 simdgroups, we can process 2 outputs in parallel
    for (uint out_base = simd_group; out_base <= FC0_OUT; out_base += 2) {
        uint out = out_base;
        if (out > FC0_OUT) continue;
        
        for (uint p = 0; p < 2; p++) {
            device const accumulator_t* acc = (p == 0) ? white_acc : black_acc;
            
            // Each thread in simdgroup processes hidden_dim/32 elements
            int32_t local_sum = 0;
            uint elements_per_thread = (hidden_dim + 31) / 32;
            uint start_idx = simd_lane * elements_per_thread;
            uint end_idx = min(start_idx + elements_per_thread, hidden_dim);
            
            for (uint i = start_idx; i < end_idx; i++) {
                int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                local_sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
            }
            
            // Simdgroup reduction
            int32_t sum = simd_sum(local_sum);
            
            if (simd_lane == 0) {
                sum += fc0_biases[out];
                int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
                if (out < FC0_OUT) {
                    fc0_sqr[p * FC0_OUT + out] = sqr_clipped_relu(result);
                } else {
                    fc0_skip[p] = clipped_relu(result);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC1 layer - parallel across outputs
    for (uint out = lid; out < FC1_OUT; out += 64) {
        int32_t sum = fc1_biases[out];
        // Unrolled for 30 inputs (2 * FC0_OUT)
        for (uint i = 0; i < 2 * FC0_OUT; i++) {
            sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
        }
        fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2 layer - simdgroup reduction for final sum
    if (simd_group == 0) {
        int32_t local_sum = 0;
        for (uint i = simd_lane; i < FC1_OUT; i += 32) {
            local_sum += fc1_out[i] * fc2_weights[i];
        }
        
        int32_t sum = simd_sum(local_sum);
        
        if (simd_lane == 0) {
            sum += fc2_biases[0];
            int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * int32_t(OUTPUT_SCALE)) / 
                              (2 * 127 * (1 << WEIGHT_SCALE_BITS));
            output[pos_idx] = sum + skip_val;
        }
    }
}

// GPU-Side Feature Extraction
// Extracts HalfKA features directly on GPU from position bitboards.
// Eliminates CPU feature extraction overhead for large batches.
// Each thread processes one position.
inline uint64_t gpu_pop_lsb(thread uint64_t& x) {
    uint low = uint(x);
    uint high = uint(x >> 32);
    uint idx;
    if (low != 0) {
        idx = ctz(low);
    } else {
        idx = 32 + ctz(high);
    }
    x &= x - 1;
    return idx;
}

kernel void gpu_extract_features(
    device const GPUPositionData* positions [[buffer(0)]],
    device int32_t* white_features [[buffer(1)]],
    device int32_t* black_features [[buffer(2)]],
    device uint32_t* white_counts [[buffer(3)]],
    device uint32_t* black_counts [[buffer(4)]],
    device uint32_t* white_offsets [[buffer(5)]],
    device uint32_t* black_offsets [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& max_features [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    
    if (gid >= batch_size)
        return;
    
    GPUPositionData pos = positions[gid];
    uint wksq = pos.king_sq[0];
    uint bksq = pos.king_sq[1];
    
    uint white_count = 0;
    uint black_count = 0;
    uint base_idx = gid * max_features;
    
    // Iterate through all piece types (PAWN=1 to QUEEN=5, skip KING=6)
    for (uint color = 0; color < 2; color++) {
        for (uint pt = 1; pt <= 5; pt++) {
            uint64_t bb = pos.pieces[color][pt];
            while (bb && white_count < max_features && black_count < max_features) {
                uint sq = gpu_pop_lsb(bb);
                
                // White perspective feature
                int32_t white_feat = int32_t(wksq * 640 + color * 320 + (pt - 1) * 64 + sq);
                if (white_feat >= 0 && white_count < max_features) {
                    white_features[base_idx + white_count++] = white_feat;
                }
                
                // Black perspective feature (mirrored)
                uint bksq_flip = bksq ^ 56;
                uint sq_flip = sq ^ 56;
                uint color_flip = 1 - color;
                int32_t black_feat = int32_t(bksq_flip * 640 + color_flip * 320 + (pt - 1) * 64 + sq_flip);
                if (black_feat >= 0 && black_count < max_features) {
                    black_features[base_idx + black_count++] = black_feat;
                }
            }
        }
    }
    
    white_counts[gid] = white_count;
    black_counts[gid] = black_count;
    
    // Calculate offsets (simple version - actual offset calculation needs prefix sum)
    white_offsets[gid] = gid * max_features;
    black_offsets[gid] = gid * max_features;
}

// PSQT accumulation kernel
kernel void gpu_psqt_accumulate(
    device const int32_t* psqt_weights [[buffer(0)]],
    device const int32_t* features [[buffer(1)]],
    device const uint32_t* feature_counts [[buffer(2)]],
    device const uint32_t* feature_offsets [[buffer(3)]],
    device int32_t* output [[buffer(4)]],
    constant uint& num_buckets [[buffer(5)]],
    constant uint& batch_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    uint pos_idx = gid.y;
    uint bucket = gid.x;
    
    if (pos_idx >= batch_size || bucket >= num_buckets)
        return;
    
    uint start = feature_offsets[pos_idx];
    uint count = feature_counts[pos_idx];
    
    int32_t acc = 0;
    for (uint i = 0; i < count; i++) {
        int32_t feat_idx = features[start + i];
        if (feat_idx >= 0) {
            acc += psqt_weights[feat_idx * num_buckets + bucket];
        }
    }
    
    output[pos_idx * num_buckets + bucket] = acc;
}

// Vectorized Feature Transform with int4 loads
// Uses 4-wide vector loads for better memory bandwidth utilization
// Each thread processes 4 consecutive hidden dimensions
kernel void gpu_feature_transform_vec4(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const int32_t* features [[buffer(2)]],
    device const uint32_t* feature_counts [[buffer(3)]],
    device const uint32_t* feature_offsets [[buffer(4)]],
    device accumulator_t* accumulators [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    
    uint pos_idx = gid.y;
    uint vec_idx = gid.x;  // Each thread handles 4 elements
    uint hidden_base = vec_idx * 4;
    
    if (pos_idx >= batch_size || hidden_base >= hidden_dim)
        return;
    
    // Load 4 biases at once using vectorized load
    int4 acc;
    if (hidden_base + 3 < hidden_dim) {
        short4 bias_vec = *reinterpret_cast<device const short4*>(biases + hidden_base);
        acc = int4(bias_vec);
    } else {
        // Handle boundary case
        acc = int4(
            hidden_base < hidden_dim ? biases[hidden_base] : 0,
            hidden_base + 1 < hidden_dim ? biases[hidden_base + 1] : 0,
            hidden_base + 2 < hidden_dim ? biases[hidden_base + 2] : 0,
            hidden_base + 3 < hidden_dim ? biases[hidden_base + 3] : 0
        );
    }
    
    uint start = feature_offsets[pos_idx];
    uint count = feature_counts[pos_idx];
    
    // Accumulate weights for active features
    for (uint i = 0; i < count; i++) {
        int32_t feat_idx = features[start + i];
        if (feat_idx >= 0) {
            uint weight_base = feat_idx * hidden_dim + hidden_base;
            if (hidden_base + 3 < hidden_dim) {
                short4 w = *reinterpret_cast<device const short4*>(weights + weight_base);
                acc += int4(w);
            } else {
                // Handle boundary
                if (hidden_base < hidden_dim) acc.x += weights[weight_base];
                if (hidden_base + 1 < hidden_dim) acc.y += weights[weight_base + 1];
                if (hidden_base + 2 < hidden_dim) acc.z += weights[weight_base + 2];
                if (hidden_base + 3 < hidden_dim) acc.w += weights[weight_base + 3];
            }
        }
    }
    
    // Store results
    uint out_base = pos_idx * hidden_dim + hidden_base;
    if (hidden_base + 3 < hidden_dim) {
        *reinterpret_cast<device int4*>(accumulators + out_base) = acc;
    } else {
        if (hidden_base < hidden_dim) accumulators[out_base] = acc.x;
        if (hidden_base + 1 < hidden_dim) accumulators[out_base + 1] = acc.y;
        if (hidden_base + 2 < hidden_dim) accumulators[out_base + 2] = acc.z;
        if (hidden_base + 3 < hidden_dim) accumulators[out_base + 3] = acc.w;
    }
}

// Dual-Perspective Vectorized Feature Transform
// Combines dual-perspective with vectorized loads for maximum throughput
kernel void gpu_feature_transform_dual_vec4(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const int32_t* white_features [[buffer(2)]],
    device const int32_t* black_features [[buffer(3)]],
    device const uint32_t* white_counts [[buffer(4)]],
    device const uint32_t* black_counts [[buffer(5)]],
    device const uint32_t* white_offsets [[buffer(6)]],
    device const uint32_t* black_offsets [[buffer(7)]],
    device accumulator_t* accumulators [[buffer(8)]],
    constant uint& hidden_dim [[buffer(9)]],
    constant uint& batch_size [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    uint pos_idx = gid.z;
    uint perspective = gid.y;
    uint vec_idx = gid.x;
    uint hidden_base = vec_idx * 4;
    
    if (pos_idx >= batch_size || hidden_base >= hidden_dim || perspective >= 2)
        return;
    
    // Load 4 biases
    int4 acc;
    if (hidden_base + 3 < hidden_dim) {
        short4 bias_vec = *reinterpret_cast<device const short4*>(biases + hidden_base);
        acc = int4(bias_vec);
    } else {
        acc = int4(
            hidden_base < hidden_dim ? biases[hidden_base] : 0,
            hidden_base + 1 < hidden_dim ? biases[hidden_base + 1] : 0,
            hidden_base + 2 < hidden_dim ? biases[hidden_base + 2] : 0,
            hidden_base + 3 < hidden_dim ? biases[hidden_base + 3] : 0
        );
    }
    
    device const int32_t* features = (perspective == 0) ? white_features : black_features;
    device const uint32_t* counts = (perspective == 0) ? white_counts : black_counts;
    device const uint32_t* offsets = (perspective == 0) ? white_offsets : black_offsets;
    
    uint start = offsets[pos_idx];
    uint count = counts[pos_idx];
    
    // Process features with 4-way unrolling
    for (uint i = 0; i < count; i++) {
        int32_t feat_idx = features[start + i];
        if (feat_idx >= 0) {
            uint weight_base = feat_idx * hidden_dim + hidden_base;
            if (hidden_base + 3 < hidden_dim) {
                short4 w = *reinterpret_cast<device const short4*>(weights + weight_base);
                acc += int4(w);
            } else {
                if (hidden_base < hidden_dim) acc.x += weights[weight_base];
                if (hidden_base + 1 < hidden_dim) acc.y += weights[weight_base + 1];
                if (hidden_base + 2 < hidden_dim) acc.z += weights[weight_base + 2];
                if (hidden_base + 3 < hidden_dim) acc.w += weights[weight_base + 3];
            }
        }
    }
    
    // Store results
    uint out_base = (pos_idx * 2 + perspective) * hidden_dim + hidden_base;
    if (hidden_base + 3 < hidden_dim) {
        *reinterpret_cast<device int4*>(accumulators + out_base) = acc;
    } else {
        if (hidden_base < hidden_dim) accumulators[out_base] = acc.x;
        if (hidden_base + 1 < hidden_dim) accumulators[out_base + 1] = acc.y;
        if (hidden_base + 2 < hidden_dim) accumulators[out_base + 2] = acc.z;
        if (hidden_base + 3 < hidden_dim) accumulators[out_base + 3] = acc.w;
    }
}

// Optimized Forward Pass with Threadgroup-Level Parallelism
// Uses simdgroup operations for parallel reduction in FC layers
kernel void gpu_nnue_forward_optimized(
    device const accumulator_t* accumulators [[buffer(0)]],
    device const layer_weight_t* fc0_weights [[buffer(1)]],
    device const int32_t* fc0_biases [[buffer(2)]],
    device const layer_weight_t* fc1_weights [[buffer(3)]],
    device const int32_t* fc1_biases [[buffer(4)]],
    device const layer_weight_t* fc2_weights [[buffer(5)]],
    device const int32_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    
    uint pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    // Threadgroup memory for intermediate results
    threadgroup int8_t fc0_sqr[2 * 16];
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out[32];
    threadgroup int32_t partial_sums[4];  // For simdgroup reduction
    
    device const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
    device const accumulator_t* black_acc = white_acc + hidden_dim;
    
    // FC0 layer with simdgroup-level parallelism
    // Each simdgroup computes partial sums, then reduces
    uint elements_per_thread = (hidden_dim + tg_size - 1) / tg_size;
    
    for (uint out = 0; out <= FC0_OUT; out++) {
        for (uint p = 0; p < 2; p++) {
            device const accumulator_t* acc = (p == 0) ? white_acc : black_acc;
            
            // Each thread computes partial sum for its portion
            int32_t local_sum = 0;
            uint start_idx = lid * elements_per_thread;
            uint end_idx = min(start_idx + elements_per_thread, hidden_dim);
            
            for (uint i = start_idx; i < end_idx; i++) {
                int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                local_sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
            }
            
            // Simdgroup reduction
            int32_t simd_sum_val = simd_sum(local_sum);
            
            // First thread in each simdgroup stores partial sum
            if (simd_lane == 0 && simd_group < 4) {
                partial_sums[simd_group] = simd_sum_val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Thread 0 combines partial sums
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
    
    // FC1 layer - parallel across outputs
    for (uint out = lid; out < FC1_OUT; out += tg_size) {
        int32_t sum = fc1_biases[out];
        for (uint i = 0; i < 2 * FC0_OUT; i++) {
            sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
        }
        fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2 layer - single thread
    if (lid == 0) {
        int32_t sum = fc2_biases[0];
        for (uint i = 0; i < FC1_OUT; i++) {
            sum += fc1_out[i] * fc2_weights[i];
        }
        
        int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * int32_t(OUTPUT_SCALE)) / 
                          (2 * 127 * (1 << WEIGHT_SCALE_BITS));
        
        output[pos_idx] = sum + skip_val;
    }
}

// Fully Fused Single-Position Evaluation
// Combines feature transform and forward pass in a single kernel
// Eliminates barrier overhead between kernels for small batches
// Each threadgroup processes one position completely
kernel void gpu_nnue_fused_single(
    device const weight_t* ft_weights [[buffer(0)]],
    device const weight_t* ft_biases [[buffer(1)]],
    device const int32_t* white_features [[buffer(2)]],
    device const int32_t* black_features [[buffer(3)]],
    device const uint32_t* white_counts [[buffer(4)]],
    device const uint32_t* black_counts [[buffer(5)]],
    device const uint32_t* white_offsets [[buffer(6)]],
    device const uint32_t* black_offsets [[buffer(7)]],
    device const layer_weight_t* fc0_weights [[buffer(8)]],
    device const int32_t* fc0_biases [[buffer(9)]],
    device const layer_weight_t* fc1_weights [[buffer(10)]],
    device const int32_t* fc1_biases [[buffer(11)]],
    device const layer_weight_t* fc2_weights [[buffer(12)]],
    device const int32_t* fc2_biases [[buffer(13)]],
    device int32_t* output [[buffer(14)]],
    constant uint& hidden_dim [[buffer(15)]],
    constant uint& batch_size [[buffer(16)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    uint pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    // Threadgroup memory for accumulators and intermediate results
    // For hidden_dim=1024, we need 2*1024*4 = 8KB for accumulators
    // This fits within the 32KB threadgroup memory limit
    threadgroup accumulator_t white_acc[1024];
    threadgroup accumulator_t black_acc[1024];
    threadgroup int8_t fc0_sqr[2 * 16];
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out[32];
    
    // Stage 1: Feature transform for white perspective
    uint w_start = white_offsets[pos_idx];
    uint w_count = white_counts[pos_idx];
    
    for (uint h = lid; h < hidden_dim; h += tg_size) {
        accumulator_t acc = accumulator_t(ft_biases[h]);
        for (uint i = 0; i < w_count; i++) {
            int32_t feat_idx = white_features[w_start + i];
            if (feat_idx >= 0) {
                acc += ft_weights[feat_idx * hidden_dim + h];
            }
        }
        white_acc[h] = acc;
    }
    
    // Stage 2: Feature transform for black perspective
    uint b_start = black_offsets[pos_idx];
    uint b_count = black_counts[pos_idx];
    
    for (uint h = lid; h < hidden_dim; h += tg_size) {
        accumulator_t acc = accumulator_t(ft_biases[h]);
        for (uint i = 0; i < b_count; i++) {
            int32_t feat_idx = black_features[b_start + i];
            if (feat_idx >= 0) {
                acc += ft_weights[feat_idx * hidden_dim + h];
            }
        }
        black_acc[h] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Stage 3: FC0 layer
    for (uint out = lid; out <= FC0_OUT; out += tg_size) {
        for (uint p = 0; p < 2; p++) {
            threadgroup accumulator_t* acc = (p == 0) ? white_acc : black_acc;
            
            int32_t sum = fc0_biases[out];
            
            // Unrolled accumulation
            uint i = 0;
            for (; i + 7 < hidden_dim; i += 8) {
                int8_t c0 = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                int8_t c1 = clipped_relu(int16_t(acc[i+1] >> WEIGHT_SCALE_BITS));
                int8_t c2 = clipped_relu(int16_t(acc[i+2] >> WEIGHT_SCALE_BITS));
                int8_t c3 = clipped_relu(int16_t(acc[i+3] >> WEIGHT_SCALE_BITS));
                int8_t c4 = clipped_relu(int16_t(acc[i+4] >> WEIGHT_SCALE_BITS));
                int8_t c5 = clipped_relu(int16_t(acc[i+5] >> WEIGHT_SCALE_BITS));
                int8_t c6 = clipped_relu(int16_t(acc[i+6] >> WEIGHT_SCALE_BITS));
                int8_t c7 = clipped_relu(int16_t(acc[i+7] >> WEIGHT_SCALE_BITS));
                
                sum += c0 * fc0_weights[(i) * (FC0_OUT + 1) + out];
                sum += c1 * fc0_weights[(i+1) * (FC0_OUT + 1) + out];
                sum += c2 * fc0_weights[(i+2) * (FC0_OUT + 1) + out];
                sum += c3 * fc0_weights[(i+3) * (FC0_OUT + 1) + out];
                sum += c4 * fc0_weights[(i+4) * (FC0_OUT + 1) + out];
                sum += c5 * fc0_weights[(i+5) * (FC0_OUT + 1) + out];
                sum += c6 * fc0_weights[(i+6) * (FC0_OUT + 1) + out];
                sum += c7 * fc0_weights[(i+7) * (FC0_OUT + 1) + out];
            }
            
            for (; i < hidden_dim; i++) {
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
    
    // Stage 4: FC1 layer
    for (uint out = lid; out < FC1_OUT; out += tg_size) {
        int32_t sum = fc1_biases[out];
        for (uint i = 0; i < 2 * FC0_OUT; i++) {
            sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
        }
        fc1_out[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Stage 5: FC2 layer
    if (lid == 0) {
        int32_t sum = fc2_biases[0];
        for (uint i = 0; i < FC1_OUT; i++) {
            sum += fc1_out[i] * fc2_weights[i];
        }
        
        int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * int32_t(OUTPUT_SCALE)) / 
                          (2 * 127 * (1 << WEIGHT_SCALE_BITS));
        
        output[pos_idx] = sum + skip_val;
    }
}

// Batch-Optimized Forward Pass
// Processes multiple positions per threadgroup for better GPU utilization
// Each threadgroup handles POSITIONS_PER_TG positions
constant uint POSITIONS_PER_TG = 4;

kernel void gpu_nnue_forward_batch(
    device const accumulator_t* accumulators [[buffer(0)]],
    device const layer_weight_t* fc0_weights [[buffer(1)]],
    device const int32_t* fc0_biases [[buffer(2)]],
    device const layer_weight_t* fc1_weights [[buffer(3)]],
    device const int32_t* fc1_biases [[buffer(4)]],
    device const layer_weight_t* fc2_weights [[buffer(5)]],
    device const int32_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant uint& hidden_dim [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]) {
    
    // Each threadgroup processes POSITIONS_PER_TG positions
    uint base_pos = gid * POSITIONS_PER_TG;
    
    // Shared memory for all positions in this threadgroup
    threadgroup int8_t fc0_sqr[POSITIONS_PER_TG][2 * 16];
    threadgroup int8_t fc0_skip[POSITIONS_PER_TG][2];
    threadgroup int8_t fc1_out[POSITIONS_PER_TG][32];
    
    // Process each position
    for (uint p_offset = 0; p_offset < POSITIONS_PER_TG; p_offset++) {
        uint pos_idx = base_pos + p_offset;
        if (pos_idx >= batch_size) continue;
        
        device const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
        device const accumulator_t* black_acc = white_acc + hidden_dim;
        
        // FC0 layer
        for (uint out = lid; out <= FC0_OUT; out += tg_size) {
            for (uint p = 0; p < 2; p++) {
                device const accumulator_t* acc = (p == 0) ? white_acc : black_acc;
                
                int32_t sum = fc0_biases[out];
                
                uint i = 0;
                for (; i + 7 < hidden_dim; i += 8) {
                    int8_t c0 = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                    int8_t c1 = clipped_relu(int16_t(acc[i+1] >> WEIGHT_SCALE_BITS));
                    int8_t c2 = clipped_relu(int16_t(acc[i+2] >> WEIGHT_SCALE_BITS));
                    int8_t c3 = clipped_relu(int16_t(acc[i+3] >> WEIGHT_SCALE_BITS));
                    int8_t c4 = clipped_relu(int16_t(acc[i+4] >> WEIGHT_SCALE_BITS));
                    int8_t c5 = clipped_relu(int16_t(acc[i+5] >> WEIGHT_SCALE_BITS));
                    int8_t c6 = clipped_relu(int16_t(acc[i+6] >> WEIGHT_SCALE_BITS));
                    int8_t c7 = clipped_relu(int16_t(acc[i+7] >> WEIGHT_SCALE_BITS));
                    
                    sum += c0 * fc0_weights[(i) * (FC0_OUT + 1) + out];
                    sum += c1 * fc0_weights[(i+1) * (FC0_OUT + 1) + out];
                    sum += c2 * fc0_weights[(i+2) * (FC0_OUT + 1) + out];
                    sum += c3 * fc0_weights[(i+3) * (FC0_OUT + 1) + out];
                    sum += c4 * fc0_weights[(i+4) * (FC0_OUT + 1) + out];
                    sum += c5 * fc0_weights[(i+5) * (FC0_OUT + 1) + out];
                    sum += c6 * fc0_weights[(i+6) * (FC0_OUT + 1) + out];
                    sum += c7 * fc0_weights[(i+7) * (FC0_OUT + 1) + out];
                }
                
                for (; i < hidden_dim; i++) {
                    int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                    sum += clipped * fc0_weights[i * (FC0_OUT + 1) + out];
                }
                
                int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
                if (out < FC0_OUT) {
                    fc0_sqr[p_offset][p * FC0_OUT + out] = sqr_clipped_relu(result);
                } else {
                    fc0_skip[p_offset][p] = clipped_relu(result);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC1 layer for all positions
    for (uint p_offset = 0; p_offset < POSITIONS_PER_TG; p_offset++) {
        uint pos_idx = base_pos + p_offset;
        if (pos_idx >= batch_size) continue;
        
        for (uint out = lid; out < FC1_OUT; out += tg_size) {
            int32_t sum = fc1_biases[out];
            for (uint i = 0; i < 2 * FC0_OUT; i++) {
                sum += fc0_sqr[p_offset][i] * fc1_weights[i * FC1_OUT + out];
            }
            fc1_out[p_offset][out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2 layer for all positions
    for (uint p_offset = 0; p_offset < POSITIONS_PER_TG; p_offset++) {
        uint pos_idx = base_pos + p_offset;
        if (pos_idx >= batch_size) continue;
        
        if (lid == 0) {
            int32_t sum = fc2_biases[0];
            for (uint i = 0; i < FC1_OUT; i++) {
                sum += fc1_out[p_offset][i] * fc2_weights[i];
            }
            
            int32_t skip_val = ((fc0_skip[p_offset][0] + fc0_skip[p_offset][1]) * 600 * int32_t(OUTPUT_SCALE)) / 
                              (2 * 127 * (1 << WEIGHT_SCALE_BITS));
            
            output[pos_idx] = sum + skip_val;
        }
    }
}
)";

// ============================================================================
// GPUNNUEManager Implementation
// ============================================================================

GPUNNUEManager::GPUNNUEManager() = default;
GPUNNUEManager::~GPUNNUEManager() = default;

bool GPUNNUEManager::initialize() {
  if (initialized_)
    return true;

  if (!gpu_available()) {
    std::cerr << "[GPU NNUE] GPU not available" << std::endl;
    return false;
  }

  if (!compile_shaders()) {
    std::cerr << "[GPU NNUE] Failed to compile shaders" << std::endl;
    return false;
  }

  if (!allocate_working_buffers()) {
    std::cerr << "[GPU NNUE] Failed to allocate working buffers" << std::endl;
    return false;
  }

  initialized_ = true;
  std::cout << "[GPU NNUE] Manager initialized" << std::endl;
  return true;
}

bool GPUNNUEManager::compile_shaders() {
  auto &backend = gpu();

  if (!backend.compile_library("gpu_nnue_integration",
                               GPU_NNUE_SHADER_SOURCE)) {
    std::cerr << "[GPU NNUE] Shader compilation failed" << std::endl;
    return false;
  }

  // Required kernels
  feature_transform_kernel_ =
      backend.create_kernel("gpu_feature_transform", "gpu_nnue_integration");
  forward_fused_kernel_ =
      backend.create_kernel("gpu_nnue_forward", "gpu_nnue_integration");
  psqt_kernel_ =
      backend.create_kernel("gpu_psqt_accumulate", "gpu_nnue_integration");

  // Optional kernels for adaptive strategy selection
  forward_simd_kernel_ =
      backend.create_kernel("gpu_nnue_forward_simd", "gpu_nnue_integration");
  feature_transform_dual_kernel_ = backend.create_kernel(
      "gpu_feature_transform_dual", "gpu_nnue_integration");
  extract_features_kernel_ =
      backend.create_kernel("gpu_extract_features", "gpu_nnue_integration");
  forward_batch_kernel_ =
      backend.create_kernel("gpu_nnue_forward_batch", "gpu_nnue_integration");

  // New optimized kernels
  feature_transform_vec4_kernel_ = backend.create_kernel(
      "gpu_feature_transform_vec4", "gpu_nnue_integration");
  feature_transform_dual_vec4_kernel_ = backend.create_kernel(
      "gpu_feature_transform_dual_vec4", "gpu_nnue_integration");
  forward_optimized_kernel_ = backend.create_kernel(
      "gpu_nnue_forward_optimized", "gpu_nnue_integration");
  fused_single_kernel_ =
      backend.create_kernel("gpu_nnue_fused_single", "gpu_nnue_integration");

  // Verify required kernels
  if (!feature_transform_kernel_ || !feature_transform_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create feature_transform kernel"
              << std::endl;
    return false;
  }

  if (!forward_fused_kernel_ || !forward_fused_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create forward kernel" << std::endl;
    return false;
  }

  if (!psqt_kernel_ || !psqt_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create psqt kernel" << std::endl;
    return false;
  }

  // Log available optional kernels
  if (forward_simd_kernel_ && forward_simd_kernel_->valid()) {
    std::cout << "[GPU NNUE] SIMD forward kernel available" << std::endl;
  }
  if (feature_transform_dual_kernel_ &&
      feature_transform_dual_kernel_->valid()) {
    std::cout << "[GPU NNUE] Dual-perspective transform kernel available"
              << std::endl;
  }
  if (extract_features_kernel_ && extract_features_kernel_->valid()) {
    std::cout << "[GPU NNUE] GPU feature extraction kernel available"
              << std::endl;
  }
  if (forward_batch_kernel_ && forward_batch_kernel_->valid()) {
    std::cout << "[GPU NNUE] Batch forward kernel available" << std::endl;
  }
  if (feature_transform_vec4_kernel_ &&
      feature_transform_vec4_kernel_->valid()) {
    std::cout
        << "[GPU NNUE] Vectorized (vec4) feature transform kernel available"
        << std::endl;
  }
  if (feature_transform_dual_vec4_kernel_ &&
      feature_transform_dual_vec4_kernel_->valid()) {
    std::cout
        << "[GPU NNUE] Dual-perspective vectorized transform kernel available"
        << std::endl;
  }
  if (forward_optimized_kernel_ && forward_optimized_kernel_->valid()) {
    std::cout << "[GPU NNUE] Optimized forward kernel available" << std::endl;
  }
  if (fused_single_kernel_ && fused_single_kernel_->valid()) {
    std::cout << "[GPU NNUE] Fused single-position kernel available"
              << std::endl;
  }

  std::cout << "[GPU NNUE] Shaders compiled successfully" << std::endl;
  return true;
}

bool GPUNNUEManager::allocate_working_buffers() {
  auto &backend = gpu();

  const size_t max_positions = GPU_MAX_BATCH_SIZE;
  const size_t max_features_per_perspective =
      max_positions * GPU_MAX_FEATURES_PER_PERSPECTIVE;
  const size_t max_hidden = GPU_FT_DIM_BIG;

  // Position data buffer
  positions_buffer_ =
      backend.create_buffer(max_positions * sizeof(GPUPositionData));

  // Feature buffers (separate for white and black)
  white_features_buffer_ =
      backend.create_buffer(max_features_per_perspective * sizeof(int32_t));
  black_features_buffer_ =
      backend.create_buffer(max_features_per_perspective * sizeof(int32_t));

  // Counts buffers (separate for dual-perspective kernel)
  white_counts_buffer_ =
      backend.create_buffer(max_positions * sizeof(uint32_t));
  black_counts_buffer_ =
      backend.create_buffer(max_positions * sizeof(uint32_t));

  // Offsets buffers (separate for dual-perspective kernel)
  white_offsets_buffer_ =
      backend.create_buffer(max_positions * sizeof(uint32_t));
  black_offsets_buffer_ =
      backend.create_buffer(max_positions * sizeof(uint32_t));

  // Legacy interleaved buffers (for backward compatibility)
  feature_counts_buffer_ =
      backend.create_buffer(max_positions * 2 * sizeof(uint32_t));
  feature_offsets_buffer_ =
      backend.create_buffer(max_positions * 2 * sizeof(uint32_t));

  // Accumulators: [batch][2 perspectives][hidden_dim]
  accumulators_buffer_ =
      backend.create_buffer(max_positions * 2 * max_hidden * sizeof(int32_t));
  psqt_buffer_ =
      backend.create_buffer(max_positions * GPU_PSQT_BUCKETS * sizeof(int32_t));
  output_buffer_ = backend.create_buffer(max_positions * sizeof(int32_t));

  if (!positions_buffer_ || !white_features_buffer_ ||
      !black_features_buffer_ || !feature_counts_buffer_ ||
      !feature_offsets_buffer_ || !accumulators_buffer_ || !psqt_buffer_ ||
      !output_buffer_ || !white_counts_buffer_ || !black_counts_buffer_ ||
      !white_offsets_buffer_ || !black_offsets_buffer_) {
    return false;
  }

  std::cout << "[GPU NNUE] Working buffers allocated: "
            << backend.allocated_memory() / 1024 << " KB" << std::endl;
  return true;
}

bool GPUNNUEManager::allocate_network_buffers(GPUNetworkData &net,
                                              int hidden_dim,
                                              bool has_threats) {
  auto &backend = gpu();

  net.hidden_dim = hidden_dim;
  net.has_threats = has_threats;

  // Feature transformer
  net.ft_weights =
      backend.create_buffer(GPU_HALFKA_DIMS * hidden_dim * sizeof(int16_t));
  net.ft_biases = backend.create_buffer(hidden_dim * sizeof(int16_t));
  net.ft_psqt = backend.create_buffer(GPU_HALFKA_DIMS * GPU_PSQT_BUCKETS *
                                      sizeof(int32_t));

  if (!net.ft_weights || !net.ft_biases || !net.ft_psqt) {
    return false;
  }

  if (has_threats) {
    net.threat_weights =
        backend.create_buffer(GPU_THREAT_DIMS * hidden_dim * sizeof(int8_t));
    net.threat_psqt = backend.create_buffer(GPU_THREAT_DIMS * GPU_PSQT_BUCKETS *
                                            sizeof(int32_t));
    if (!net.threat_weights || !net.threat_psqt) {
      return false;
    }
  }

  // FC layers for each bucket
  for (int b = 0; b < GPU_LAYER_STACKS; b++) {
    auto &layer = net.layers[b];

    layer.fc0_weights = backend.create_buffer(
        hidden_dim * 2 * (GPU_FC0_OUT + 1) * sizeof(int8_t));
    layer.fc0_biases =
        backend.create_buffer((GPU_FC0_OUT + 1) * sizeof(int32_t));
    layer.fc1_weights =
        backend.create_buffer(GPU_FC0_OUT * 2 * GPU_FC1_OUT * sizeof(int8_t));
    layer.fc1_biases = backend.create_buffer(GPU_FC1_OUT * sizeof(int32_t));
    layer.fc2_weights = backend.create_buffer(GPU_FC1_OUT * sizeof(int8_t));
    layer.fc2_biases = backend.create_buffer(sizeof(int32_t));

    if (!layer.valid()) {
      return false;
    }
  }

  net.valid = true;
  return true;
}

bool GPUNNUEManager::load_networks(const Eval::NNUE::Networks &networks) {
  if (!initialized_ && !initialize()) {
    return false;
  }

  std::cout << "[GPU NNUE] Loading networks..." << std::endl;

  // Extract and print network info
  auto big_info = get_network_info<Eval::NNUE::NetworkBig>();
  auto small_info = get_network_info<Eval::NNUE::NetworkSmall>();

  print_network_info(big_info, "Big");
  print_network_info(small_info, "Small");

  std::cout << "[GPU NNUE] Total memory required: "
            << (get_network_memory_requirement<Eval::NNUE::NetworkBig>() +
                get_network_memory_requirement<Eval::NNUE::NetworkSmall>()) /
                   1024
            << " KB" << std::endl;

  // Allocate big network buffers
  if (!allocate_network_buffers(big_network_, GPU_FT_DIM_BIG, true)) {
    std::cerr << "[GPU NNUE] Failed to allocate big network buffers"
              << std::endl;
    return false;
  }

  // Allocate small network buffers
  if (!allocate_network_buffers(small_network_, GPU_FT_DIM_SMALL, false)) {
    std::cerr << "[GPU NNUE] Failed to allocate small network buffers"
              << std::endl;
    return false;
  }

  // Extract and upload big network weights
  auto big_weights =
      GPUNNUEWeightExtractor<Eval::NNUE::NetworkBig>::extract(networks.big);
  if (big_weights.valid) {
    std::cout << "[GPU NNUE] Uploading big network weights..." << std::endl;

    // Upload feature transformer
    if (big_weights.ft.biases && big_network_.ft_biases) {
      std::memcpy(
          big_network_.ft_biases->data(), big_weights.ft.biases,
          std::min(big_weights.ft.biases_size, big_network_.ft_biases->size()));
    }
    if (big_weights.ft.weights && big_network_.ft_weights) {
      std::memcpy(big_network_.ft_weights->data(), big_weights.ft.weights,
                  std::min(big_weights.ft.weights_size,
                           big_network_.ft_weights->size()));
    }
    if (big_weights.ft.psqt_weights && big_network_.ft_psqt) {
      std::memcpy(
          big_network_.ft_psqt->data(), big_weights.ft.psqt_weights,
          std::min(big_weights.ft.psqt_size, big_network_.ft_psqt->size()));
    }

    // Upload layer weights
    for (int b = 0; b < GPU_LAYER_STACKS; b++) {
      const auto &src = big_weights.layers[b];
      auto &dst = big_network_.layers[b];

      if (src.fc0_weights && dst.fc0_weights) {
        std::memcpy(dst.fc0_weights->data(), src.fc0_weights,
                    std::min(src.fc0_weights_size, dst.fc0_weights->size()));
      }
      if (src.fc0_biases && dst.fc0_biases) {
        std::memcpy(dst.fc0_biases->data(), src.fc0_biases,
                    std::min(src.fc0_biases_size, dst.fc0_biases->size()));
      }
      if (src.fc1_weights && dst.fc1_weights) {
        std::memcpy(dst.fc1_weights->data(), src.fc1_weights,
                    std::min(src.fc1_weights_size, dst.fc1_weights->size()));
      }
      if (src.fc1_biases && dst.fc1_biases) {
        std::memcpy(dst.fc1_biases->data(), src.fc1_biases,
                    std::min(src.fc1_biases_size, dst.fc1_biases->size()));
      }
      if (src.fc2_weights && dst.fc2_weights) {
        std::memcpy(dst.fc2_weights->data(), src.fc2_weights,
                    std::min(src.fc2_weights_size, dst.fc2_weights->size()));
      }
      if (src.fc2_biases && dst.fc2_biases) {
        std::memcpy(dst.fc2_biases->data(), src.fc2_biases,
                    std::min(src.fc2_biases_size, dst.fc2_biases->size()));
      }
    }
    std::cout << "[GPU NNUE] Big network weights uploaded" << std::endl;
  }

  // Extract and upload small network weights
  auto small_weights =
      GPUNNUEWeightExtractor<Eval::NNUE::NetworkSmall>::extract(networks.small);
  if (small_weights.valid) {
    std::cout << "[GPU NNUE] Uploading small network weights..." << std::endl;

    // Upload feature transformer
    if (small_weights.ft.biases && small_network_.ft_biases) {
      std::memcpy(small_network_.ft_biases->data(), small_weights.ft.biases,
                  std::min(small_weights.ft.biases_size,
                           small_network_.ft_biases->size()));
    }
    if (small_weights.ft.weights && small_network_.ft_weights) {
      std::memcpy(small_network_.ft_weights->data(), small_weights.ft.weights,
                  std::min(small_weights.ft.weights_size,
                           small_network_.ft_weights->size()));
    }
    if (small_weights.ft.psqt_weights && small_network_.ft_psqt) {
      std::memcpy(
          small_network_.ft_psqt->data(), small_weights.ft.psqt_weights,
          std::min(small_weights.ft.psqt_size, small_network_.ft_psqt->size()));
    }

    // Upload layer weights
    for (int b = 0; b < GPU_LAYER_STACKS; b++) {
      const auto &src = small_weights.layers[b];
      auto &dst = small_network_.layers[b];

      if (src.fc0_weights && dst.fc0_weights) {
        std::memcpy(dst.fc0_weights->data(), src.fc0_weights,
                    std::min(src.fc0_weights_size, dst.fc0_weights->size()));
      }
      if (src.fc0_biases && dst.fc0_biases) {
        std::memcpy(dst.fc0_biases->data(), src.fc0_biases,
                    std::min(src.fc0_biases_size, dst.fc0_biases->size()));
      }
      if (src.fc1_weights && dst.fc1_weights) {
        std::memcpy(dst.fc1_weights->data(), src.fc1_weights,
                    std::min(src.fc1_weights_size, dst.fc1_weights->size()));
      }
      if (src.fc1_biases && dst.fc1_biases) {
        std::memcpy(dst.fc1_biases->data(), src.fc1_biases,
                    std::min(src.fc1_biases_size, dst.fc1_biases->size()));
      }
      if (src.fc2_weights && dst.fc2_weights) {
        std::memcpy(dst.fc2_weights->data(), src.fc2_weights,
                    std::min(src.fc2_weights_size, dst.fc2_weights->size()));
      }
      if (src.fc2_biases && dst.fc2_biases) {
        std::memcpy(dst.fc2_biases->data(), src.fc2_biases,
                    std::min(src.fc2_biases_size, dst.fc2_biases->size()));
      }
    }
    std::cout << "[GPU NNUE] Small network weights uploaded" << std::endl;
  }

  std::cout << "[GPU NNUE] Networks loaded. Total GPU memory: "
            << gpu_memory_used() / 1024 << " KB" << std::endl;

  return true;
}

bool GPUNNUEManager::evaluate_batch(GPUEvalBatch &batch, bool use_big_network) {
  if (!is_ready() || batch.count == 0) {
    return false;
  }

  // Select evaluation strategy based on batch size
  EvalStrategy strategy = tuning_.select_strategy(batch.count);

  if (strategy == EvalStrategy::CPU_FALLBACK) {
    cpu_evals_ += batch.count;
    return false;
  }

  auto start = std::chrono::high_resolution_clock::now();

  auto &backend = gpu();
  const GPUNetworkData &net = use_big_network ? big_network_ : small_network_;

  if (!net.valid) {
    cpu_evals_ += batch.count;
    return false;
  }

  const int batch_size = batch.count;
  const int hidden_dim = net.hidden_dim;

  // Use pre-allocated buffer pointers for direct writes (avoid std::vector
  // allocations)
  int32_t *white_features_ptr =
      static_cast<int32_t *>(white_features_buffer_->data());
  int32_t *black_features_ptr =
      static_cast<int32_t *>(black_features_buffer_->data());
  uint32_t *white_counts_ptr =
      static_cast<uint32_t *>(white_counts_buffer_->data());
  uint32_t *black_counts_ptr =
      static_cast<uint32_t *>(black_counts_buffer_->data());
  uint32_t *white_offsets_ptr =
      static_cast<uint32_t *>(white_offsets_buffer_->data());
  uint32_t *black_offsets_ptr =
      static_cast<uint32_t *>(black_offsets_buffer_->data());

  // Extract features directly into GPU buffers (zero-copy on unified memory)
  // Optimized: removed redundant bounds checks since feature indices are
  // mathematically guaranteed to be in range for valid chess positions
  size_t white_feature_idx = 0;
  size_t black_feature_idx = 0;

  for (int i = 0; i < batch_size; i++) {
    const auto &pos_data = batch.positions[i];

    white_offsets_ptr[i] = static_cast<uint32_t>(white_feature_idx);
    black_offsets_ptr[i] = static_cast<uint32_t>(black_feature_idx);

    const int wksq = pos_data.king_sq[0];
    const int bksq_flip = pos_data.king_sq[1] ^ 56; // flip_rank inline

    int white_count = 0;
    int black_count = 0;

    // Extract HalfKA features from piece bitboards
    // Unrolled color loop for better branch prediction
    for (int pt = PAWN; pt <= QUEEN; pt++) {
      const int pt_offset = (pt - 1) * 64;

      // White pieces
      Bitboard bb_w = pos_data.pieces[0][pt];
      while (bb_w) {
        const int s = pop_lsb(bb_w);
        white_features_ptr[white_feature_idx++] = wksq * 640 + pt_offset + s;
        black_features_ptr[black_feature_idx++] =
            bksq_flip * 640 + 320 + pt_offset + (s ^ 56);
        white_count++;
        black_count++;
      }

      // Black pieces
      Bitboard bb_b = pos_data.pieces[1][pt];
      while (bb_b) {
        const int s = pop_lsb(bb_b);
        white_features_ptr[white_feature_idx++] =
            wksq * 640 + 320 + pt_offset + s;
        black_features_ptr[black_feature_idx++] =
            bksq_flip * 640 + pt_offset + (s ^ 56);
        white_count++;
        black_count++;
      }
    }

    white_counts_ptr[i] = static_cast<uint32_t>(white_count);
    black_counts_ptr[i] = static_cast<uint32_t>(black_count);
  }

  // Note: positions_buffer_ upload removed - feature extraction is done on CPU
  // and features are written directly to GPU buffers via unified memory

  // Create command encoder
  auto encoder = backend.create_encoder();

  // Select optimal kernel based on batch size and available kernels
  // Testing showed that the fused kernel uses too much threadgroup memory
  // and reduces GPU occupancy, making it slower than separate dispatches
  // Always use dual-perspective kernel since it processes both perspectives
  const bool use_fused_kernel =
      false; // Disabled - slower than separate kernels
  const bool use_dual_kernel =
      feature_transform_dual_kernel_ && feature_transform_dual_kernel_->valid();

  if (use_fused_kernel) {
    // Fused kernel: feature transform + forward pass in single dispatch
    // Eliminates inter-kernel barrier overhead for small batches
    const int bucket = batch.buckets.empty() ? 0 : batch.buckets[0];
    const auto &layer = net.layers[std::clamp(bucket, 0, GPU_LAYER_STACKS - 1)];

    encoder->set_kernel(fused_single_kernel_.get());
    encoder->set_buffer(net.ft_weights.get(), 0);
    encoder->set_buffer(net.ft_biases.get(), 1);
    encoder->set_buffer(white_features_buffer_.get(), 2);
    encoder->set_buffer(black_features_buffer_.get(), 3);
    encoder->set_buffer(white_counts_buffer_.get(), 4);
    encoder->set_buffer(black_counts_buffer_.get(), 5);
    encoder->set_buffer(white_offsets_buffer_.get(), 6);
    encoder->set_buffer(black_offsets_buffer_.get(), 7);
    encoder->set_buffer(layer.fc0_weights.get(), 8);
    encoder->set_buffer(layer.fc0_biases.get(), 9);
    encoder->set_buffer(layer.fc1_weights.get(), 10);
    encoder->set_buffer(layer.fc1_biases.get(), 11);
    encoder->set_buffer(layer.fc2_weights.get(), 12);
    encoder->set_buffer(layer.fc2_biases.get(), 13);
    encoder->set_buffer(output_buffer_.get(), 14);
    encoder->set_value(static_cast<uint32_t>(hidden_dim), 15);
    encoder->set_value(static_cast<uint32_t>(batch_size), 16);
    // Use 256 threads per threadgroup for optimal parallelism
    encoder->dispatch_threadgroups(batch_size, 1, 1, 256, 1, 1);
  } else if (use_dual_kernel) {
    encoder->set_kernel(feature_transform_dual_kernel_.get());
    encoder->set_buffer(net.ft_weights.get(), 0);
    encoder->set_buffer(net.ft_biases.get(), 1);
    encoder->set_buffer(white_features_buffer_.get(), 2);
    encoder->set_buffer(black_features_buffer_.get(), 3);
    encoder->set_buffer(white_counts_buffer_.get(), 4);
    encoder->set_buffer(black_counts_buffer_.get(), 5);
    encoder->set_buffer(white_offsets_buffer_.get(), 6);
    encoder->set_buffer(black_offsets_buffer_.get(), 7);
    encoder->set_buffer(accumulators_buffer_.get(), 8);
    encoder->set_value(static_cast<uint32_t>(hidden_dim), 9);
    encoder->set_value(static_cast<uint32_t>(batch_size), 10);
    // 3D dispatch: (hidden_dim, 2 perspectives, batch_size)
    encoder->dispatch_threads(hidden_dim, 2, batch_size);
    // Barrier required for untracked hazard mode - ensures feature transform
    // completes before forward pass reads accumulators
    encoder->barrier();
  } else {
    // Single-perspective kernel: faster for small batches
    encoder->set_kernel(feature_transform_kernel_.get());
    encoder->set_buffer(net.ft_weights.get(), 0);
    encoder->set_buffer(net.ft_biases.get(), 1);
    encoder->set_buffer(white_features_buffer_.get(), 2);
    encoder->set_buffer(white_counts_buffer_.get(), 3);
    encoder->set_buffer(white_offsets_buffer_.get(), 4);
    encoder->set_buffer(accumulators_buffer_.get(), 5);
    encoder->set_value(static_cast<uint32_t>(hidden_dim), 6);
    encoder->set_value(static_cast<uint32_t>(batch_size), 7);
    encoder->dispatch_threads(hidden_dim, batch_size);
    encoder->barrier();
  }

  // Select correct bucket layer based on position piece counts
  // For simplicity, use bucket 0 for all (can be extended to per-position
  // buckets)
  // Skip forward pass if we used the fused kernel
  if (!use_fused_kernel) {
    const int bucket = batch.buckets.empty() ? 0 : batch.buckets[0];
    const auto &layer = net.layers[std::clamp(bucket, 0, GPU_LAYER_STACKS - 1)];

    // Forward pass - standard kernel performs well across all batch sizes
    // Testing showed the optimized kernel adds overhead without benefit
    encoder->set_kernel(forward_fused_kernel_.get());
    encoder->set_buffer(accumulators_buffer_.get(), 0);
    encoder->set_buffer(layer.fc0_weights.get(), 1);
    encoder->set_buffer(layer.fc0_biases.get(), 2);
    encoder->set_buffer(layer.fc1_weights.get(), 3);
    encoder->set_buffer(layer.fc1_biases.get(), 4);
    encoder->set_buffer(layer.fc2_weights.get(), 5);
    encoder->set_buffer(layer.fc2_biases.get(), 6);
    encoder->set_buffer(output_buffer_.get(), 7);
    encoder->set_value(static_cast<uint32_t>(hidden_dim), 8);
    encoder->set_value(static_cast<uint32_t>(batch_size), 9);
    encoder->dispatch_threadgroups(batch_size, 1, 1, GPU_FORWARD_THREADS, 1, 1);
  }

  backend.submit_and_wait(encoder.get());

  // Read results (zero-copy on unified memory)
  batch.positional_scores.resize(batch_size);
  std::memcpy(batch.positional_scores.data(), output_buffer_->data(),
              batch_size * sizeof(int32_t));

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ms_ +=
      std::chrono::duration<double, std::milli>(end - start).count();
  batch_count_++;
  gpu_evals_ += batch_size;

  return true;
}

bool GPUNNUEManager::evaluate_batch_async(
    GPUEvalBatch &batch, std::function<void(bool success)> completion_handler,
    bool use_big_network) {
  if (!is_ready() || batch.count == 0) {
    if (completion_handler)
      completion_handler(false);
    return false;
  }

  // Select evaluation strategy based on batch size
  EvalStrategy strategy = tuning_.select_strategy(batch.count);

  if (strategy == EvalStrategy::CPU_FALLBACK) {
    cpu_evals_ += batch.count;
    if (completion_handler)
      completion_handler(false);
    return false;
  }

  auto &backend = gpu();
  const GPUNetworkData &net = use_big_network ? big_network_ : small_network_;

  if (!net.valid) {
    cpu_evals_ += batch.count;
    if (completion_handler)
      completion_handler(false);
    return false;
  }

  const int batch_size = batch.count;
  const int hidden_dim = net.hidden_dim;

  // Use pre-allocated buffer pointers for direct writes
  int32_t *white_features_ptr =
      static_cast<int32_t *>(white_features_buffer_->data());
  int32_t *black_features_ptr =
      static_cast<int32_t *>(black_features_buffer_->data());
  uint32_t *white_counts_ptr =
      static_cast<uint32_t *>(white_counts_buffer_->data());
  uint32_t *black_counts_ptr =
      static_cast<uint32_t *>(black_counts_buffer_->data());
  uint32_t *white_offsets_ptr =
      static_cast<uint32_t *>(white_offsets_buffer_->data());
  uint32_t *black_offsets_ptr =
      static_cast<uint32_t *>(black_offsets_buffer_->data());

  // Extract features directly into GPU buffers
  // Optimized: removed redundant bounds checks
  size_t white_feature_idx = 0;
  size_t black_feature_idx = 0;

  for (int i = 0; i < batch_size; i++) {
    const auto &pos_data = batch.positions[i];

    white_offsets_ptr[i] = static_cast<uint32_t>(white_feature_idx);
    black_offsets_ptr[i] = static_cast<uint32_t>(black_feature_idx);

    const int wksq = pos_data.king_sq[0];
    const int bksq_flip = pos_data.king_sq[1] ^ 56;

    int white_count = 0;
    int black_count = 0;

    for (int pt = PAWN; pt <= QUEEN; pt++) {
      const int pt_offset = (pt - 1) * 64;

      // White pieces
      Bitboard bb_w = pos_data.pieces[0][pt];
      while (bb_w) {
        const int s = pop_lsb(bb_w);
        white_features_ptr[white_feature_idx++] = wksq * 640 + pt_offset + s;
        black_features_ptr[black_feature_idx++] =
            bksq_flip * 640 + 320 + pt_offset + (s ^ 56);
        white_count++;
        black_count++;
      }

      // Black pieces
      Bitboard bb_b = pos_data.pieces[1][pt];
      while (bb_b) {
        const int s = pop_lsb(bb_b);
        white_features_ptr[white_feature_idx++] =
            wksq * 640 + 320 + pt_offset + s;
        black_features_ptr[black_feature_idx++] =
            bksq_flip * 640 + pt_offset + (s ^ 56);
        white_count++;
        black_count++;
      }
    }

    white_counts_ptr[i] = static_cast<uint32_t>(white_count);
    black_counts_ptr[i] = static_cast<uint32_t>(black_count);
  }

  std::memcpy(positions_buffer_->data(), batch.positions.data(),
              batch_size * sizeof(GPUPositionData));

  auto encoder = backend.create_encoder();

  // Feature transform - always use dual kernel for both perspectives
  const bool use_dual_kernel =
      feature_transform_dual_kernel_ && feature_transform_dual_kernel_->valid();

  if (use_dual_kernel) {
    encoder->set_kernel(feature_transform_dual_kernel_.get());
    encoder->set_buffer(net.ft_weights.get(), 0);
    encoder->set_buffer(net.ft_biases.get(), 1);
    encoder->set_buffer(white_features_buffer_.get(), 2);
    encoder->set_buffer(black_features_buffer_.get(), 3);
    encoder->set_buffer(white_counts_buffer_.get(), 4);
    encoder->set_buffer(black_counts_buffer_.get(), 5);
    encoder->set_buffer(white_offsets_buffer_.get(), 6);
    encoder->set_buffer(black_offsets_buffer_.get(), 7);
    encoder->set_buffer(accumulators_buffer_.get(), 8);
    encoder->set_value(static_cast<uint32_t>(hidden_dim), 9);
    encoder->set_value(static_cast<uint32_t>(batch_size), 10);
    encoder->dispatch_threads(hidden_dim, 2, batch_size);
    encoder->barrier();
  } else {
    encoder->set_kernel(feature_transform_kernel_.get());
    encoder->set_buffer(net.ft_weights.get(), 0);
    encoder->set_buffer(net.ft_biases.get(), 1);
    encoder->set_buffer(white_features_buffer_.get(), 2);
    encoder->set_buffer(white_counts_buffer_.get(), 3);
    encoder->set_buffer(white_offsets_buffer_.get(), 4);
    encoder->set_buffer(accumulators_buffer_.get(), 5);
    encoder->set_value(static_cast<uint32_t>(hidden_dim), 6);
    encoder->set_value(static_cast<uint32_t>(batch_size), 7);
    encoder->dispatch_threads(hidden_dim, batch_size);
    encoder->barrier();
  }

  const int bucket = batch.buckets.empty() ? 0 : batch.buckets[0];
  const auto &layer = net.layers[std::clamp(bucket, 0, GPU_LAYER_STACKS - 1)];

  encoder->set_kernel(forward_fused_kernel_.get());
  encoder->set_buffer(accumulators_buffer_.get(), 0);
  encoder->set_buffer(layer.fc0_weights.get(), 1);
  encoder->set_buffer(layer.fc0_biases.get(), 2);
  encoder->set_buffer(layer.fc1_weights.get(), 3);
  encoder->set_buffer(layer.fc1_biases.get(), 4);
  encoder->set_buffer(layer.fc2_weights.get(), 5);
  encoder->set_buffer(layer.fc2_biases.get(), 6);
  encoder->set_buffer(output_buffer_.get(), 7);
  encoder->set_value(static_cast<uint32_t>(hidden_dim), 8);
  encoder->set_value(static_cast<uint32_t>(batch_size), 9);
  encoder->dispatch_threadgroups(batch_size, 1, 1, GPU_FORWARD_THREADS, 1, 1);

  // Capture necessary data for completion handler
  Buffer *output_buf = output_buffer_.get();
  GPUEvalBatch *batch_ptr = &batch;
  std::atomic<size_t> *gpu_evals_ptr = &gpu_evals_;
  std::atomic<size_t> *batch_count_ptr = &batch_count_;

  // Submit with async completion handler
  backend.submit_async(encoder.get(), [=]() {
    // Read results in completion handler
    batch_ptr->positional_scores.resize(batch_size);
    std::memcpy(batch_ptr->positional_scores.data(), output_buf->data(),
                batch_size * sizeof(int32_t));

    (*gpu_evals_ptr) += batch_size;
    (*batch_count_ptr)++;

    if (completion_handler) {
      completion_handler(true);
    }
  });

  return true;
}

std::pair<int32_t, int32_t> GPUNNUEManager::evaluate_single(const Position &pos,
                                                            bool use_big) {
  // Single position evaluation is not efficient on GPU
  // Fall back to CPU
  cpu_evals_++;
  return {0, 0};
}

double GPUNNUEManager::avg_batch_time_ms() const {
  return batch_count_ > 0 ? total_time_ms_ / batch_count_ : 0;
}

void GPUNNUEManager::reset_stats() {
  gpu_evals_ = 0;
  cpu_evals_ = 0;
  batch_count_ = 0;
  total_time_ms_ = 0;
}

size_t GPUNNUEManager::gpu_memory_used() const {
  if (!gpu_available())
    return 0;
  return gpu().allocated_memory();
}

std::string GPUNNUEManager::status_string() const {
  std::stringstream ss;
  ss << "GPU NNUE Manager Status:\n";
  ss << "  Initialized: " << (initialized_ ? "Yes" : "No") << "\n";
  ss << "  Big Network: " << (big_network_.valid ? "Ready" : "Not loaded")
     << "\n";
  ss << "  Small Network: " << (small_network_.valid ? "Ready" : "Not loaded")
     << "\n";
  ss << "  GPU Memory: " << gpu_memory_used() / 1024 << " KB\n";
  ss << "  GPU Evaluations: " << gpu_evals_.load() << "\n";
  ss << "  CPU Fallbacks: " << cpu_evals_.load() << "\n";
  ss << "  Total Batches: " << batch_count_.load() << "\n";
  if (batch_count_ > 0) {
    ss << "  Avg Batch Time: " << avg_batch_time_ms() << " ms\n";
  }
  return ss.str();
}

// ============================================================================
// Global Interface
// ============================================================================

static std::unique_ptr<GPUNNUEManager> g_gpu_nnue_manager;

GPUNNUEManager &gpu_nnue_manager() {
  if (!g_gpu_nnue_manager) {
    g_gpu_nnue_manager = std::make_unique<GPUNNUEManager>();
  }
  return *g_gpu_nnue_manager;
}

bool initialize_gpu_nnue(const Eval::NNUE::Networks &networks) {
  return gpu_nnue_manager().load_networks(networks);
}

bool gpu_nnue_manager_available() {
  return gpu_available() && gpu_nnue_manager().is_ready();
}

bool gpu_evaluate_batch(GPUEvalBatch &batch, bool use_big) {
  return gpu_nnue_manager().evaluate_batch(batch, use_big);
}

} // namespace MetalFish::GPU

#else // !USE_METAL

// Stub implementations when Metal is not available
namespace MetalFish::GPU {

void GPUPositionData::from_position(const Position &) {}
size_t GPUNetworkData::memory_usage() const { return 0; }
void GPUEvalBatch::clear() { count = 0; }
void GPUEvalBatch::reserve(int) {}
void GPUEvalBatch::add_position(const Position &) {}

GPUNNUEManager::GPUNNUEManager() = default;
GPUNNUEManager::~GPUNNUEManager() = default;
bool GPUNNUEManager::initialize() { return false; }
bool GPUNNUEManager::compile_shaders() { return false; }
bool GPUNNUEManager::allocate_working_buffers() { return false; }
bool GPUNNUEManager::allocate_network_buffers(GPUNetworkData &, int, bool) {
  return false;
}
bool GPUNNUEManager::load_networks(const Eval::NNUE::Networks &) {
  return false;
}
bool GPUNNUEManager::evaluate_batch(GPUEvalBatch &, bool) { return false; }
std::pair<int32_t, int32_t> GPUNNUEManager::evaluate_single(const Position &,
                                                            bool) {
  return {0, 0};
}
double GPUNNUEManager::avg_batch_time_ms() const { return 0; }
void GPUNNUEManager::reset_stats() {}
size_t GPUNNUEManager::gpu_memory_used() const { return 0; }
std::string GPUNNUEManager::status_string() const {
  return "GPU NNUE: Not available\n";
}

static std::unique_ptr<GPUNNUEManager> g_gpu_nnue_manager;

GPUNNUEManager &gpu_nnue_manager() {
  if (!g_gpu_nnue_manager) {
    g_gpu_nnue_manager = std::make_unique<GPUNNUEManager>();
  }
  return *g_gpu_nnue_manager;
}

bool initialize_gpu_nnue(const Eval::NNUE::Networks &) { return false; }
bool gpu_nnue_manager_available() { return false; }
bool gpu_evaluate_batch(GPUEvalBatch &, bool) { return false; }

} // namespace MetalFish::GPU

#endif // USE_METAL

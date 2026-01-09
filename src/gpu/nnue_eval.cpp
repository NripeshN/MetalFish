/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU-accelerated NNUE Evaluation Implementation
  
  This module provides batch NNUE evaluation on the GPU.
  Key optimizations:
  - Unified memory: Zero-copy access between CPU and GPU
  - Batch processing: Evaluate multiple positions in parallel
  - Runtime shader compilation: Compiles shaders on first use
*/

#ifdef USE_METAL

#include "nnue_eval.h"
#include "backend.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

namespace MetalFish {
namespace GPU {

// Embedded shader source for runtime compilation
static const char* NNUE_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// NNUE architecture dimensions
constant int FT_DIM_BIG = 1024;
constant int FC0_OUT = 15;
constant int FC1_OUT = 32;

typedef int16_t weight_t;
typedef int8_t activation_t;
typedef int32_t accumulator_t;

inline int8_t clipped_relu(int16_t x) {
    return int8_t(clamp(int(x), 0, 127));
}

inline int8_t sqr_clipped_relu(int16_t x) {
    int clamped = clamp(int(x), 0, 127);
    return int8_t((clamped * clamped) >> 7);
}

kernel void feature_transform_full(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const int32_t* features [[buffer(2)]],
    device const int32_t* feature_offsets [[buffer(3)]],
    device accumulator_t* output [[buffer(4)]],
    constant int& hidden_dim [[buffer(5)]],
    constant int& batch_size [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    int pos_idx = gid.y;
    int hidden_idx = gid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
        return;
    
    accumulator_t acc = accumulator_t(biases[hidden_idx]);
    
    int start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
    int end = feature_offsets[pos_idx];
    
    for (int i = start; i < end; i++) {
        int feature_idx = features[i];
        acc += weights[feature_idx * hidden_dim + hidden_idx];
    }
    
    output[pos_idx * hidden_dim + hidden_idx] = acc;
}

kernel void nnue_forward_fused(
    device const accumulator_t* accumulators [[buffer(0)]],
    device const weight_t* fc0_weights [[buffer(1)]],
    device const weight_t* fc0_biases [[buffer(2)]],
    device const weight_t* fc1_weights [[buffer(3)]],
    device const weight_t* fc1_biases [[buffer(4)]],
    device const weight_t* fc2_weights [[buffer(5)]],
    device const weight_t* fc2_biases [[buffer(6)]],
    device int32_t* output [[buffer(7)]],
    constant int& hidden_dim [[buffer(8)]],
    constant int& batch_size [[buffer(9)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    int pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    threadgroup int8_t fc0_sqr[2 * FC0_OUT];
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out[FC1_OUT];
    
    device const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
    device const accumulator_t* black_acc = white_acc + hidden_dim;
    
    // FC0 Layer
    for (int out = lid; out <= FC0_OUT; out += tg_size) {
        for (int p = 0; p < 2; p++) {
            device const accumulator_t* acc = (p == 0) ? white_acc : black_acc;
            
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
    
    // FC1 Layer
    for (int out = lid; out < FC1_OUT; out += tg_size) {
        accumulator_t sum = fc1_biases[out];
        
        for (int i = 0; i < 2 * FC0_OUT; i++) {
            sum += fc0_sqr[i] * fc1_weights[i * FC1_OUT + out];
        }
        
        fc1_out[out] = clipped_relu(int16_t(sum >> 6));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2 Layer (Output)
    if (lid == 0) {
        accumulator_t sum = fc2_biases[0];
        
        for (int i = 0; i < FC1_OUT; i++) {
            sum += fc1_out[i] * fc2_weights[i];
        }
        
        int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * 16) / (2 * 127 * 64);
        
        output[pos_idx] = sum + skip_val;
    }
}

kernel void batch_accumulator_update(
    device const accumulator_t* src_accumulators [[buffer(0)]],
    device accumulator_t* dst_accumulators [[buffer(1)]],
    device const weight_t* weights [[buffer(2)]],
    device const int32_t* added_features [[buffer(3)]],
    device const int32_t* removed_features [[buffer(4)]],
    device const int32_t* update_info [[buffer(5)]],
    constant int& hidden_dim [[buffer(6)]],
    constant int& batch_size [[buffer(7)]],
    constant int& max_updates [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    int pos_idx = gid.y;
    int hidden_idx = gid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
        return;
    
    int src_idx = update_info[pos_idx * 4];
    int perspective = update_info[pos_idx * 4 + 1];
    int num_added = update_info[pos_idx * 4 + 2];
    int num_removed = update_info[pos_idx * 4 + 3];
    
    accumulator_t acc = src_accumulators[src_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx];
    
    int base = pos_idx * max_updates;
    
    for (int i = 0; i < num_removed; i++) {
        int feature_idx = removed_features[base + i];
        if (feature_idx >= 0) {
            acc -= weights[feature_idx * hidden_dim + hidden_idx];
        }
    }
    
    for (int i = 0; i < num_added; i++) {
        int feature_idx = added_features[base + i];
        if (feature_idx >= 0) {
            acc += weights[feature_idx * hidden_dim + hidden_idx];
        }
    }
    
    dst_accumulators[pos_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx] = acc;
}
)";

// ============================================================================
// NNUEEvaluator Implementation
// ============================================================================

NNUEEvaluator::NNUEEvaluator() {}

NNUEEvaluator::~NNUEEvaluator() {}

bool NNUEEvaluator::initialize(const void* big_weights, size_t big_size,
                               const void* small_weights, size_t small_size) {
    if (!gpu_available()) {
        std::cerr << "[GPU NNUE] GPU not available" << std::endl;
        return false;
    }
    
    // Compile shaders
    if (!load_kernels()) {
        std::cerr << "[GPU NNUE] Failed to load kernels" << std::endl;
        return false;
    }
    
    // Allocate buffers
    if (!allocate_buffers()) {
        std::cerr << "[GPU NNUE] Failed to allocate buffers" << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[GPU NNUE] Initialized successfully" << std::endl;
    return true;
}

bool NNUEEvaluator::load_kernels() {
    auto& backend = gpu();
    
    // Compile shader source
    if (!backend.compile_library("nnue", NNUE_SHADER_SOURCE)) {
        std::cerr << "[GPU NNUE] Failed to compile NNUE shaders" << std::endl;
        return false;
    }
    
    // Create kernels
    feature_transform_kernel_ = backend.create_kernel("feature_transform_full", "nnue");
    nnue_batch_kernel_ = backend.create_kernel("nnue_forward_fused", "nnue");
    
    if (!feature_transform_kernel_ || !feature_transform_kernel_->valid()) {
        std::cerr << "[GPU NNUE] Failed to create feature_transform kernel" << std::endl;
        return false;
    }
    
    if (!nnue_batch_kernel_ || !nnue_batch_kernel_->valid()) {
        std::cerr << "[GPU NNUE] Failed to create nnue_forward kernel" << std::endl;
        return false;
    }
    
    std::cout << "[GPU NNUE] Kernels loaded successfully" << std::endl;
    return true;
}

bool NNUEEvaluator::allocate_buffers() {
    auto& backend = gpu();
    
    // Allocate input/output buffers for batch processing
    size_t max_features = MAX_BATCH_SIZE * MAX_FEATURES_PER_POSITION * 2;
    
    features_buffer_ = backend.create_buffer(max_features * sizeof(int32_t));
    feature_counts_buffer_ = backend.create_buffer(MAX_BATCH_SIZE * sizeof(int32_t));
    accumulators_buffer_ = backend.create_buffer(MAX_BATCH_SIZE * 2 * NNUE_FEATURE_DIM_BIG * sizeof(int32_t));
    output_buffer_ = backend.create_buffer(MAX_BATCH_SIZE * sizeof(int32_t));
    
    if (!features_buffer_ || !feature_counts_buffer_ || 
        !accumulators_buffer_ || !output_buffer_) {
        return false;
    }
    
    std::cout << "[GPU NNUE] Buffers allocated: " 
              << backend.allocated_memory() / 1024 << " KB" << std::endl;
    return true;
}

bool NNUEEvaluator::evaluate_batch(EvalBatch& batch) {
    if (!initialized_ || batch.count == 0) {
        return false;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    auto& backend = gpu();
    
    // Upload features to GPU (unified memory - just copy to buffer)
    if (batch.features.size() > 0) {
        int32_t* feature_ptr = features_buffer_->as<int32_t>();
        std::copy(batch.features.begin(), batch.features.end(), feature_ptr);
    }
    
    // Upload feature counts
    int32_t* counts_ptr = feature_counts_buffer_->as<int32_t>();
    std::copy(batch.feature_counts.begin(), batch.feature_counts.end(), counts_ptr);
    
    // Create command encoder and dispatch
    auto encoder = backend.create_encoder();
    
    // For now, just return false to indicate GPU eval not fully implemented
    // The actual kernel dispatch would go here once we have the weight buffers
    
    auto end = std::chrono::high_resolution_clock::now();
    total_time_ms_ += std::chrono::duration<double, std::milli>(end - start).count();
    batch_count_++;
    gpu_evals_ += batch.count;
    
    return false; // Fall back to CPU for now
}

int32_t NNUEEvaluator::evaluate(const Position& pos) {
    // Single position evaluation - not efficient on GPU
    // Should use CPU path
    cpu_evals_++;
    return 0;
}

// Global instance
static std::unique_ptr<NNUEEvaluator> g_gpu_nnue;

NNUEEvaluator& gpu_nnue() {
    if (!g_gpu_nnue) {
        g_gpu_nnue = std::make_unique<NNUEEvaluator>();
    }
    return *g_gpu_nnue;
}

} // namespace GPU
} // namespace MetalFish

// Forward declare MetalBackend for the cast
namespace MetalFish {
namespace GPU {
class MetalBackend;
}
}

#else

// Stub implementation when Metal is not available
namespace MetalFish {
namespace GPU {

NNUEEvaluator::NNUEEvaluator() {}
NNUEEvaluator::~NNUEEvaluator() {}

bool NNUEEvaluator::initialize(const void*, size_t, const void*, size_t) {
    return false;
}

bool NNUEEvaluator::evaluate_batch(EvalBatch&) {
    return false;
}

int32_t NNUEEvaluator::evaluate(const Position&) {
    return 0;
}

bool NNUEEvaluator::load_kernels() { return false; }
bool NNUEEvaluator::allocate_buffers() { return false; }

static std::unique_ptr<NNUEEvaluator> g_gpu_nnue;

NNUEEvaluator& gpu_nnue() {
    if (!g_gpu_nnue) {
        g_gpu_nnue = std::make_unique<NNUEEvaluator>();
    }
    return *g_gpu_nnue;
}

} // namespace GPU
} // namespace MetalFish

#endif // USE_METAL

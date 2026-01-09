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

#include "nnue_eval.h"

#ifdef USE_METAL

#include "backend.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

namespace MetalFish::GPU {

// Embedded shader source for runtime compilation
static const char *NNUE_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

// NNUE architecture dimensions (configurable via constants)
constant int FT_DIM = 1024;  // Feature transformer output dimension
constant int FC0_OUT = 15;   // First FC layer output (without skip connection)
constant int FC1_OUT = 32;   // Second FC layer output

// Weight scale for quantization
constant int WEIGHT_SCALE_BITS = 6;

// Type aliases matching CPU implementation
typedef int16_t weight_t;
typedef int8_t activation_t;
typedef int32_t accumulator_t;

// Clipped ReLU activation: clamp to [0, 127]
inline int8_t clipped_relu(int16_t x) {
    return int8_t(clamp(int(x), 0, 127));
}

// Squared clipped ReLU: (clamp(x, 0, 127))^2 >> 7
inline int8_t sqr_clipped_relu(int16_t x) {
    int clamped = clamp(int(x), 0, 127);
    return int8_t((clamped * clamped) >> 7);
}

// Feature transformer kernel
// Transforms sparse input features to dense accumulator
kernel void feature_transform(
    device const weight_t* weights [[buffer(0)]],
    device const weight_t* biases [[buffer(1)]],
    device const weight_t* psqt_weights [[buffer(2)]],
    device const int32_t* features [[buffer(3)]],
    device const int32_t* feature_offsets [[buffer(4)]],
    device accumulator_t* accumulators [[buffer(5)]],
    device int32_t* psqt_output [[buffer(6)]],
    constant int& hidden_dim [[buffer(7)]],
    constant int& batch_size [[buffer(8)]],
    constant int& psqt_buckets [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    int pos_idx = gid.y;
    int hidden_idx = gid.x;
    
    if (pos_idx >= batch_size || hidden_idx >= hidden_dim)
        return;
    
    // Start with bias
    accumulator_t acc_white = accumulator_t(biases[hidden_idx]);
    accumulator_t acc_black = accumulator_t(biases[hidden_idx]);
    
    // Get feature range for this position
    int start = (pos_idx > 0) ? feature_offsets[pos_idx - 1] : 0;
    int end = feature_offsets[pos_idx];
    
    // Accumulate features (features come in pairs: white_feat, black_feat)
    for (int i = start; i < end; i += 2) {
        int white_feat = features[i];
        int black_feat = features[i + 1];
        
        if (white_feat >= 0) {
            acc_white += weights[white_feat * hidden_dim + hidden_idx];
        }
        if (black_feat >= 0) {
            acc_black += weights[black_feat * hidden_dim + hidden_idx];
        }
    }
    
    // Store accumulators (white perspective first, then black)
    accumulators[pos_idx * 2 * hidden_dim + hidden_idx] = acc_white;
    accumulators[pos_idx * 2 * hidden_dim + hidden_dim + hidden_idx] = acc_black;
    
    // Compute PSQT score (only first thread per position)
    if (hidden_idx == 0) {
        int32_t psqt = 0;
        for (int i = start; i < end; i += 2) {
            int white_feat = features[i];
            if (white_feat >= 0) {
                // Sum PSQT values for bucket 0 (simplified)
                psqt += psqt_weights[white_feat * psqt_buckets];
            }
        }
        psqt_output[pos_idx] = psqt;
    }
}

// Fused NNUE forward pass kernel
// Computes FC0 -> SqrClippedReLU -> FC1 -> ClippedReLU -> FC2
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
    constant int& fc0_out [[buffer(10)]],
    constant int& fc1_out [[buffer(11)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    int pos_idx = gid;
    if (pos_idx >= batch_size)
        return;
    
    // Threadgroup shared memory for intermediate results
    threadgroup int8_t fc0_sqr[2 * 16];  // FC0_OUT * 2, padded
    threadgroup int8_t fc0_skip[2];
    threadgroup int8_t fc1_out_buf[32];  // FC1_OUT
    
    device const accumulator_t* white_acc = accumulators + pos_idx * 2 * hidden_dim;
    device const accumulator_t* black_acc = white_acc + hidden_dim;
    
    // FC0 Layer: sparse input to (FC0_OUT + 1) outputs
    // Process both perspectives
    for (int out = lid; out <= fc0_out; out += tg_size) {
        for (int p = 0; p < 2; p++) {
            device const accumulator_t* acc = (p == 0) ? white_acc : black_acc;
            
            accumulator_t sum = fc0_biases[out];
            
            // Sparse matrix-vector multiplication
            for (int i = 0; i < hidden_dim; i++) {
                // Apply clipped ReLU to accumulator value
                int8_t clipped = clipped_relu(int16_t(acc[i] >> WEIGHT_SCALE_BITS));
                sum += clipped * fc0_weights[i * (fc0_out + 1) + out];
            }
            
            int16_t result = int16_t(sum >> WEIGHT_SCALE_BITS);
            
            if (out < fc0_out) {
                // SqrClippedReLU for main outputs
                fc0_sqr[p * fc0_out + out] = sqr_clipped_relu(result);
            } else {
                // Skip connection uses ClippedReLU
                fc0_skip[p] = clipped_relu(result);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC1 Layer: 2*FC0_OUT inputs to FC1_OUT outputs
    for (int out = lid; out < fc1_out; out += tg_size) {
        accumulator_t sum = fc1_biases[out];
        
        for (int i = 0; i < 2 * fc0_out; i++) {
            sum += fc0_sqr[i] * fc1_weights[i * fc1_out + out];
        }
        
        fc1_out_buf[out] = clipped_relu(int16_t(sum >> WEIGHT_SCALE_BITS));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // FC2 Layer: FC1_OUT inputs to 1 output (only one thread needed)
    if (lid == 0) {
        accumulator_t sum = fc2_biases[0];
        
        for (int i = 0; i < fc1_out; i++) {
            sum += fc1_out_buf[i] * fc2_weights[i];
        }
        
        // Add skip connection contribution
        // Skip value is scaled: (skip[0] + skip[1]) * 600 * 16 / (2 * 127 * 64)
        int32_t skip_val = ((fc0_skip[0] + fc0_skip[1]) * 600 * 16) / (2 * 127 * 64);
        
        output[pos_idx] = sum + skip_val;
    }
}

// Incremental accumulator update kernel
// Updates accumulator when pieces move
kernel void accumulator_update(
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
    
    // Update info: [src_idx, perspective, num_added, num_removed]
    int src_idx = update_info[pos_idx * 4];
    int perspective = update_info[pos_idx * 4 + 1];
    int num_added = update_info[pos_idx * 4 + 2];
    int num_removed = update_info[pos_idx * 4 + 3];
    
    // Start from source accumulator
    accumulator_t acc = src_accumulators[src_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx];
    
    int base = pos_idx * max_updates;
    
    // Remove features
    for (int i = 0; i < num_removed; i++) {
        int feature_idx = removed_features[base + i];
        if (feature_idx >= 0) {
            acc -= weights[feature_idx * hidden_dim + hidden_idx];
        }
    }
    
    // Add features
    for (int i = 0; i < num_added; i++) {
        int feature_idx = added_features[base + i];
        if (feature_idx >= 0) {
            acc += weights[feature_idx * hidden_dim + hidden_idx];
        }
    }
    
    // Store updated accumulator
    dst_accumulators[pos_idx * 2 * hidden_dim + perspective * hidden_dim + hidden_idx] = acc;
}
)";

// ============================================================================
// NNUEEvaluator Implementation
// ============================================================================

NNUEEvaluator::NNUEEvaluator() {}

NNUEEvaluator::~NNUEEvaluator() {}

bool NNUEEvaluator::initialize(const void *big_weights, size_t big_size,
                               const void *small_weights, size_t small_size) {
  if (!gpu_available()) {
    std::cerr << "[GPU NNUE] GPU not available" << std::endl;
    return false;
  }

  // Compile shaders
  if (!load_kernels()) {
    std::cerr << "[GPU NNUE] Failed to load kernels" << std::endl;
    return false;
  }

  // Allocate working buffers
  if (!allocate_buffers()) {
    std::cerr << "[GPU NNUE] Failed to allocate buffers" << std::endl;
    return false;
  }

  // Upload network weights (if provided)
  if (big_weights && big_size > 0) {
    if (!upload_weights(big_weights, big_size, big_network_, 
                       NNUE_FEATURE_DIM_BIG, NNUE_L2_BIG, NNUE_L3_BIG)) {
      std::cerr << "[GPU NNUE] Failed to upload big network weights" << std::endl;
      // Continue without big network - will fall back to CPU
    }
  }

  if (small_weights && small_size > 0) {
    if (!upload_weights(small_weights, small_size, small_network_,
                       NNUE_FEATURE_DIM_SMALL, NNUE_L2_SMALL, NNUE_L3_SMALL)) {
      std::cerr << "[GPU NNUE] Failed to upload small network weights" << std::endl;
      // Continue without small network - will fall back to CPU
    }
  }

  initialized_ = true;
  std::cout << "[GPU NNUE] Initialized successfully" << std::endl;
  std::cout << "[GPU NNUE] Big network: " << (big_network_.valid ? "loaded" : "not loaded") << std::endl;
  std::cout << "[GPU NNUE] Small network: " << (small_network_.valid ? "loaded" : "not loaded") << std::endl;
  
  return true;
}

bool NNUEEvaluator::load_kernels() {
  auto &backend = gpu();

  // Compile shader source
  if (!backend.compile_library("nnue", NNUE_SHADER_SOURCE)) {
    std::cerr << "[GPU NNUE] Failed to compile NNUE shaders" << std::endl;
    return false;
  }

  // Create kernels
  feature_transform_kernel_ = backend.create_kernel("feature_transform", "nnue");
  nnue_forward_kernel_ = backend.create_kernel("nnue_forward_fused", "nnue");
  accumulator_update_kernel_ = backend.create_kernel("accumulator_update", "nnue");

  if (!feature_transform_kernel_ || !feature_transform_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create feature_transform kernel" << std::endl;
    return false;
  }

  if (!nnue_forward_kernel_ || !nnue_forward_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create nnue_forward kernel" << std::endl;
    return false;
  }

  if (!accumulator_update_kernel_ || !accumulator_update_kernel_->valid()) {
    std::cerr << "[GPU NNUE] Failed to create accumulator_update kernel" << std::endl;
    return false;
  }

  std::cout << "[GPU NNUE] Kernels loaded successfully" << std::endl;
  return true;
}

bool NNUEEvaluator::allocate_buffers() {
  auto &backend = gpu();

  // Allocate input/output buffers for batch processing
  size_t max_features = MAX_BATCH_SIZE * MAX_FEATURES_PER_POSITION * 2;

  features_buffer_ = backend.create_buffer(max_features * sizeof(int32_t));
  feature_counts_buffer_ = backend.create_buffer(MAX_BATCH_SIZE * sizeof(int32_t));
  
  // Accumulators for both perspectives
  accumulators_buffer_ = backend.create_buffer(
      MAX_BATCH_SIZE * 2 * NNUE_FEATURE_DIM_BIG * sizeof(int32_t));
  
  output_buffer_ = backend.create_buffer(MAX_BATCH_SIZE * sizeof(int32_t));
  psqt_output_buffer_ = backend.create_buffer(MAX_BATCH_SIZE * sizeof(int32_t));

  if (!features_buffer_ || !feature_counts_buffer_ || !accumulators_buffer_ ||
      !output_buffer_ || !psqt_output_buffer_) {
    return false;
  }

  std::cout << "[GPU NNUE] Buffers allocated: "
            << backend.allocated_memory() / 1024 << " KB" << std::endl;
  return true;
}

bool NNUEEvaluator::upload_weights(const void* weights, size_t size, 
                                   GPUNetworkWeights& gpu_weights,
                                   int feature_dim, int l2, int l3) {
  // Note: This is a simplified implementation
  // Full implementation would parse the NNUE file format and extract weights
  // For now, we just allocate buffers - actual weight loading requires
  // understanding the exact binary format of the NNUE files
  
  auto& backend = gpu();
  
  gpu_weights.feature_dim = feature_dim;
  gpu_weights.l2 = l2;
  gpu_weights.l3 = l3;
  
  // Allocate weight buffers
  // Feature transformer: HALFKA_DIMS x feature_dim
  size_t ft_weights_size = HALFKA_DIMS * feature_dim * sizeof(int16_t);
  size_t ft_biases_size = feature_dim * sizeof(int16_t);
  size_t ft_psqt_size = HALFKA_DIMS * PSQT_DIMS * sizeof(int16_t);
  
  // FC layers
  size_t fc0_weights_size = feature_dim * 2 * (l2 + 1) * sizeof(int16_t);
  size_t fc0_biases_size = (l2 + 1) * sizeof(int16_t);
  size_t fc1_weights_size = l2 * 2 * l3 * sizeof(int16_t);
  size_t fc1_biases_size = l3 * sizeof(int16_t);
  size_t fc2_weights_size = l3 * sizeof(int16_t);
  size_t fc2_biases_size = sizeof(int16_t);
  
  gpu_weights.ft_weights = backend.create_buffer(ft_weights_size);
  gpu_weights.ft_biases = backend.create_buffer(ft_biases_size);
  gpu_weights.ft_psqt = backend.create_buffer(ft_psqt_size);
  gpu_weights.fc0_weights = backend.create_buffer(fc0_weights_size);
  gpu_weights.fc0_biases = backend.create_buffer(fc0_biases_size);
  gpu_weights.fc1_weights = backend.create_buffer(fc1_weights_size);
  gpu_weights.fc1_biases = backend.create_buffer(fc1_biases_size);
  gpu_weights.fc2_weights = backend.create_buffer(fc2_weights_size);
  gpu_weights.fc2_biases = backend.create_buffer(fc2_biases_size);
  
  if (!gpu_weights.ft_weights || !gpu_weights.ft_biases || !gpu_weights.ft_psqt ||
      !gpu_weights.fc0_weights || !gpu_weights.fc0_biases ||
      !gpu_weights.fc1_weights || !gpu_weights.fc1_biases ||
      !gpu_weights.fc2_weights || !gpu_weights.fc2_biases) {
    return false;
  }
  
  // TODO: Parse NNUE file format and copy weights to GPU buffers
  // For now, mark as valid but weights are not actually loaded
  // This means GPU evaluation will not produce correct results
  // until proper weight loading is implemented
  gpu_weights.valid = false;  // Set to true once weight loading is implemented
  
  return true;
}

bool NNUEEvaluator::evaluate_batch(EvalBatch &batch, bool use_big) {
  if (!initialized_ || batch.count == 0) {
    return false;
  }
  
  // Check if we have valid weights for the requested network
  const GPUNetworkWeights& weights = use_big ? big_network_ : small_network_;
  if (!weights.valid) {
    // Fall back to CPU
    return false;
  }

  // Check batch size threshold
  if (batch.count < min_batch_size_) {
    cpu_evals_ += batch.count;
    return false;
  }

  auto start = std::chrono::high_resolution_clock::now();

  auto &backend = gpu();

  // Upload features to GPU (unified memory - just copy to buffer)
  if (!batch.features.empty()) {
    int32_t *feature_ptr = features_buffer_->as<int32_t>();
    std::memcpy(feature_ptr, batch.features.data(), 
                batch.features.size() * sizeof(int32_t));
  }

  // Upload feature counts
  int32_t *counts_ptr = feature_counts_buffer_->as<int32_t>();
  std::memcpy(counts_ptr, batch.feature_counts.data(),
              batch.feature_counts.size() * sizeof(int32_t));

  int hidden_dim = weights.feature_dim;
  int fc0_out = weights.l2;
  int fc1_out = weights.l3;
  int psqt_buckets = NNUE_PSQT_BUCKETS;

  // Create command encoder
  auto encoder = backend.create_encoder();

  // 1. Feature transform
  encoder->set_kernel(feature_transform_kernel_.get());
  encoder->set_buffer(weights.ft_weights.get(), 0);
  encoder->set_buffer(weights.ft_biases.get(), 1);
  encoder->set_buffer(weights.ft_psqt.get(), 2);
  encoder->set_buffer(features_buffer_.get(), 3);
  encoder->set_buffer(feature_counts_buffer_.get(), 4);
  encoder->set_buffer(accumulators_buffer_.get(), 5);
  encoder->set_buffer(psqt_output_buffer_.get(), 6);
  encoder->set_value(hidden_dim, 7);
  encoder->set_value(batch.count, 8);
  encoder->set_value(psqt_buckets, 9);
  
  encoder->dispatch_threads(hidden_dim, batch.count);
  encoder->barrier();

  // 2. NNUE forward pass
  encoder->set_kernel(nnue_forward_kernel_.get());
  encoder->set_buffer(accumulators_buffer_.get(), 0);
  encoder->set_buffer(weights.fc0_weights.get(), 1);
  encoder->set_buffer(weights.fc0_biases.get(), 2);
  encoder->set_buffer(weights.fc1_weights.get(), 3);
  encoder->set_buffer(weights.fc1_biases.get(), 4);
  encoder->set_buffer(weights.fc2_weights.get(), 5);
  encoder->set_buffer(weights.fc2_biases.get(), 6);
  encoder->set_buffer(output_buffer_.get(), 7);
  encoder->set_value(hidden_dim, 8);
  encoder->set_value(batch.count, 9);
  encoder->set_value(fc0_out, 10);
  encoder->set_value(fc1_out, 11);
  
  // Dispatch one threadgroup per position
  encoder->dispatch_threadgroups(batch.count, 1, 1, 64, 1, 1);

  // Submit and wait
  backend.submit_and_wait(encoder.get());

  // Read results
  int32_t *psqt_results = psqt_output_buffer_->as<int32_t>();
  int32_t *positional_results = output_buffer_->as<int32_t>();
  
  batch.psqt_scores.resize(batch.count);
  batch.positional_scores.resize(batch.count);
  
  std::memcpy(batch.psqt_scores.data(), psqt_results, 
              batch.count * sizeof(int32_t));
  std::memcpy(batch.positional_scores.data(), positional_results,
              batch.count * sizeof(int32_t));

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ms_ += std::chrono::duration<double, std::milli>(end - start).count();
  batch_count_++;
  gpu_evals_ += batch.count;

  return true;
}

int32_t NNUEEvaluator::evaluate(const Position &pos) {
  // Single position evaluation - not efficient on GPU
  // Should use CPU path
  cpu_evals_++;
  return 0;
}

// Global instance
static std::unique_ptr<NNUEEvaluator> g_gpu_nnue;

bool gpu_nnue_available() {
  return gpu_available() && g_gpu_nnue && g_gpu_nnue->available();
}

NNUEEvaluator &gpu_nnue() {
  if (!g_gpu_nnue) {
    g_gpu_nnue = std::make_unique<NNUEEvaluator>();
  }
  return *g_gpu_nnue;
}

bool init_gpu_nnue(const void* big_weights, size_t big_size,
                   const void* small_weights, size_t small_size) {
  if (!gpu_available()) {
    return false;
  }
  
  return gpu_nnue().initialize(big_weights, big_size, small_weights, small_size);
}

} // namespace MetalFish::GPU

#else // !USE_METAL

// Stub implementation when Metal is not available
namespace MetalFish::GPU {

NNUEEvaluator::NNUEEvaluator() {}
NNUEEvaluator::~NNUEEvaluator() {}

bool NNUEEvaluator::initialize(const void *, size_t, const void *, size_t) {
  return false;
}

bool NNUEEvaluator::evaluate_batch(EvalBatch &, bool) { return false; }

int32_t NNUEEvaluator::evaluate(const Position &) { return 0; }

bool NNUEEvaluator::load_kernels() { return false; }
bool NNUEEvaluator::allocate_buffers() { return false; }
bool NNUEEvaluator::upload_weights(const void*, size_t, GPUNetworkWeights&, int, int, int) { 
  return false; 
}

static std::unique_ptr<NNUEEvaluator> g_gpu_nnue;

bool gpu_nnue_available() { return false; }

NNUEEvaluator &gpu_nnue() {
  if (!g_gpu_nnue) {
    g_gpu_nnue = std::make_unique<NNUEEvaluator>();
  }
  return *g_gpu_nnue;
}

bool init_gpu_nnue(const void*, size_t, const void*, size_t) {
  return false;
}

} // namespace MetalFish::GPU

#endif // USE_METAL

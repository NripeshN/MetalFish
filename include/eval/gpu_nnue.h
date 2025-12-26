/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  GPU NNUE Evaluator
  ==================
  
  This evaluator runs Stockfish's NNUE architecture on Apple Silicon GPU
  using Metal, leveraging unified memory for zero-copy evaluation.
  
  Key advantage over CPU NNUE:
  - Matrix operations parallelized across GPU cores
  - Unified memory means no copy overhead
  - Can evaluate single positions efficiently (unlike discrete GPUs)
  
  Architecture matches Stockfish exactly:
  - Feature Transformer: 45056 -> 1024 (with incremental updates)
  - FC0: 2048 -> 16 (perspectives concatenated)
  - FC1: 30 -> 32
  - FC2: 32 -> 1 + skip connection
*/

#pragma once

#include "core/types.h"
#include "core/position.h"
#include <Metal/Metal.hpp>
#include <memory>
#include <vector>
#include <array>
#include <atomic>

namespace MetalFish {
namespace Eval {

// NNUE dimensions (matching Stockfish big network)
constexpr int FT_IN_DIMS = 45056;        // HalfKAv2_hm features
constexpr int FT_OUT_DIMS = 1024;        // Hidden layer size
constexpr int FC0_OUT = 16;              // First FC output
constexpr int FC1_OUT = 32;              // Second FC output
constexpr int PSQT_BUCKETS = 8;

// Accumulator - stored in unified memory for GPU access
struct alignas(64) GPUAccumulator {
    std::array<int32_t, FT_OUT_DIMS> white;
    std::array<int32_t, FT_OUT_DIMS> black;
    std::array<int32_t, PSQT_BUCKETS> psqt;
    bool computed = false;
    
    void clear() {
        computed = false;
    }
};

// GPU NNUE network weights - loaded once, shared across all evaluations
class GPUNNUEWeights {
public:
    // Feature transformer
    MTL::Buffer* ft_weights = nullptr;      // [FT_IN_DIMS x FT_OUT_DIMS] int16_t
    MTL::Buffer* ft_biases = nullptr;       // [FT_OUT_DIMS] int16_t
    MTL::Buffer* psqt_weights = nullptr;    // [FT_IN_DIMS x PSQT_BUCKETS] int16_t
    
    // FC layers
    MTL::Buffer* fc0_weights = nullptr;     // [FT_OUT_DIMS * 2 x FC0_OUT] int8_t
    MTL::Buffer* fc0_biases = nullptr;      // [FC0_OUT] int32_t
    MTL::Buffer* fc1_weights = nullptr;     // [30 x FC1_OUT] int8_t
    MTL::Buffer* fc1_biases = nullptr;      // [FC1_OUT] int32_t
    MTL::Buffer* fc2_weights = nullptr;     // [FC1_OUT] int8_t
    MTL::Buffer* fc2_bias = nullptr;        // [1] int32_t
    
    ~GPUNNUEWeights();
    bool load(MTL::Device* device, const std::string& path);
    bool is_loaded() const { return loaded_; }
    
private:
    bool loaded_ = false;
};

// Main GPU NNUE Evaluator
class GPUNNUEEvaluator {
public:
    GPUNNUEEvaluator();
    ~GPUNNUEEvaluator();
    
    // Initialize Metal resources
    bool init();
    
    // Load NNUE network
    bool load_network(const std::string& path);
    
    // Evaluate a single position
    // This uses unified memory - accumulator can be in shared memory
    Value evaluate(const Position& pos, GPUAccumulator* acc = nullptr);
    
    // Compute feature transformer (for incremental updates)
    void compute_accumulator(const Position& pos, GPUAccumulator& acc);
    
    // Incremental accumulator update
    void update_accumulator(GPUAccumulator& acc,
                            const std::vector<int>& added,
                            const std::vector<int>& removed,
                            Color perspective);
    
    // Forward pass through FC layers only (accumulator already computed)
    Value forward_pass(const GPUAccumulator& acc, Color stm);
    
    // Get active features for position
    static void get_features(const Position& pos, Color perspective,
                             std::vector<int>& features);
    
    // Statistics
    uint64_t evaluations() const { return evaluations_; }
    uint64_t gpu_time_ns() const { return gpu_time_ns_; }
    bool is_ready() const { return ready_; }
    
private:
    // Metal resources
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    MTL::Library* library_ = nullptr;
    
    // Compute pipelines
    MTL::ComputePipelineState* ft_kernel_ = nullptr;
    MTL::ComputePipelineState* ft_update_kernel_ = nullptr;
    MTL::ComputePipelineState* forward_kernel_ = nullptr;
    
    // Weights
    std::unique_ptr<GPUNNUEWeights> weights_;
    
    // Working buffers (in unified memory)
    MTL::Buffer* acc_buffer_ = nullptr;
    MTL::Buffer* features_buffer_ = nullptr;
    MTL::Buffer* output_buffer_ = nullptr;
    MTL::Buffer* scratch_buffer_ = nullptr;
    
    // State
    bool ready_ = false;
    std::atomic<uint64_t> evaluations_{0};
    std::atomic<uint64_t> gpu_time_ns_{0};
    
    // Fallback CPU evaluation
    Value cpu_evaluate(const Position& pos);
};

// Global GPU NNUE evaluator
GPUNNUEEvaluator& gpu_nnue();

} // namespace Eval
} // namespace MetalFish



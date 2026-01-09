/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Evaluation Header
  
  This module provides GPU-accelerated NNUE evaluation with:
  - Full network inference on GPU
  - Batch evaluation for parallel search
  - Seamless CPU fallback when GPU is unavailable
*/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>

// Include backend for Buffer and ComputeKernel definitions
#include "backend.h"

// Forward declarations
namespace MetalFish {
class Position;
}

namespace MetalFish::GPU {

// NNUE Architecture constants (must match CPU implementation)
constexpr int NNUE_FEATURE_DIM_BIG = 1024;
constexpr int NNUE_FEATURE_DIM_SMALL = 128;
constexpr int NNUE_L2_BIG = 15;
constexpr int NNUE_L3_BIG = 32;
constexpr int NNUE_L2_SMALL = 15;
constexpr int NNUE_L3_SMALL = 32;
constexpr int NNUE_PSQT_BUCKETS = 8;
constexpr int NNUE_LAYER_STACKS = 8;

// Maximum batch size for GPU evaluation
constexpr int MAX_BATCH_SIZE = 256;
constexpr int MAX_FEATURES_PER_POSITION = 32;

// Feature index for HalfKAv2_hm
constexpr int HALFKA_DIMS = 45056;  // 64 * 11 * 64 + 64
constexpr int PSQT_DIMS = 8;

/**
 * Batch of positions for GPU evaluation
 */
struct EvalBatch {
    int count = 0;
    
    // Feature indices for each position (sparse representation)
    std::vector<int32_t> features;
    std::vector<int32_t> feature_counts;  // Cumulative count per position
    
    // Output scores (filled by GPU)
    std::vector<int32_t> psqt_scores;
    std::vector<int32_t> positional_scores;
    
    void clear() {
        count = 0;
        features.clear();
        feature_counts.clear();
        psqt_scores.clear();
        positional_scores.clear();
    }
    
    void reserve(int n) {
        features.reserve(n * MAX_FEATURES_PER_POSITION * 2);
        feature_counts.reserve(n);
        psqt_scores.resize(n);
        positional_scores.resize(n);
    }
};

/**
 * GPU NNUE Network Weights
 * Stores network weights in GPU-friendly format
 */
struct GPUNetworkWeights {
    // Feature transformer weights and biases
    std::unique_ptr<Buffer> ft_weights;      // [HALFKA_DIMS x FEATURE_DIM]
    std::unique_ptr<Buffer> ft_biases;       // [FEATURE_DIM]
    std::unique_ptr<Buffer> ft_psqt;         // [HALFKA_DIMS x PSQT_BUCKETS]
    
    // FC0 layer (sparse input)
    std::unique_ptr<Buffer> fc0_weights;     // [FEATURE_DIM*2 x (L2+1)]
    std::unique_ptr<Buffer> fc0_biases;      // [L2+1]
    
    // FC1 layer
    std::unique_ptr<Buffer> fc1_weights;     // [L2*2 x L3]
    std::unique_ptr<Buffer> fc1_biases;      // [L3]
    
    // FC2 layer (output)
    std::unique_ptr<Buffer> fc2_weights;     // [L3 x 1]
    std::unique_ptr<Buffer> fc2_biases;      // [1]
    
    int feature_dim = 0;
    int l2 = 0;
    int l3 = 0;
    bool valid = false;
};

/**
 * GPU NNUE Evaluator
 * 
 * Provides GPU-accelerated NNUE evaluation with automatic CPU fallback.
 * Uses unified memory for zero-copy data access on Apple Silicon.
 */
class NNUEEvaluator {
public:
    NNUEEvaluator();
    ~NNUEEvaluator();
    
    // Prevent copying
    NNUEEvaluator(const NNUEEvaluator&) = delete;
    NNUEEvaluator& operator=(const NNUEEvaluator&) = delete;
    
    /**
     * Initialize GPU NNUE with network weights
     * @param big_weights Pointer to big network weights
     * @param big_size Size of big network weights
     * @param small_weights Pointer to small network weights  
     * @param small_size Size of small network weights
     * @return true if GPU initialization succeeded
     */
    bool initialize(const void* big_weights, size_t big_size,
                   const void* small_weights, size_t small_size);
    
    /**
     * Check if GPU NNUE is available and initialized
     */
    bool available() const { return initialized_; }
    
    /**
     * Evaluate a batch of positions on GPU
     * @param batch Input/output batch structure
     * @param use_big Use big network (true) or small network (false)
     * @return true if GPU evaluation succeeded, false to fall back to CPU
     */
    bool evaluate_batch(EvalBatch& batch, bool use_big = true);
    
    /**
     * Evaluate a single position
     * Note: For single positions, CPU is usually faster
     * @param pos Position to evaluate
     * @return Evaluation score
     */
    int32_t evaluate(const Position& pos);
    
    /**
     * Get minimum batch size for GPU to be efficient
     * Below this threshold, CPU evaluation is preferred
     */
    int min_efficient_batch_size() const { return min_batch_size_; }
    
    /**
     * Set minimum batch size threshold
     */
    void set_min_batch_size(int size) { min_batch_size_ = size; }
    
    // Statistics
    size_t gpu_evaluations() const { return gpu_evals_; }
    size_t cpu_evaluations() const { return cpu_evals_; }
    double total_gpu_time_ms() const { return total_time_ms_; }
    size_t batch_count() const { return batch_count_; }
    double avg_batch_time_ms() const { 
        return batch_count_ > 0 ? total_time_ms_ / batch_count_ : 0; 
    }
    
    void reset_stats() {
        gpu_evals_ = 0;
        cpu_evals_ = 0;
        total_time_ms_ = 0;
        batch_count_ = 0;
    }

private:
    bool load_kernels();
    bool allocate_buffers();
    bool upload_weights(const void* weights, size_t size, GPUNetworkWeights& gpu_weights, int feature_dim, int l2, int l3);
    
    bool initialized_ = false;
    int min_batch_size_ = 16;  // GPU is efficient above this batch size
    
    // Network weights
    GPUNetworkWeights big_network_;
    GPUNetworkWeights small_network_;
    
    // Compute kernels
    std::unique_ptr<ComputeKernel> feature_transform_kernel_;
    std::unique_ptr<ComputeKernel> nnue_forward_kernel_;
    std::unique_ptr<ComputeKernel> accumulator_update_kernel_;
    
    // Working buffers
    std::unique_ptr<Buffer> features_buffer_;
    std::unique_ptr<Buffer> feature_counts_buffer_;
    std::unique_ptr<Buffer> accumulators_buffer_;
    std::unique_ptr<Buffer> output_buffer_;
    std::unique_ptr<Buffer> psqt_output_buffer_;
    
    // Statistics
    std::atomic<size_t> gpu_evals_{0};
    std::atomic<size_t> cpu_evals_{0};
    double total_time_ms_ = 0;
    size_t batch_count_ = 0;
};

/**
 * Check if GPU NNUE is available
 */
bool gpu_nnue_available();

/**
 * Get global GPU NNUE evaluator instance
 */
NNUEEvaluator& gpu_nnue();

/**
 * Initialize GPU NNUE with network data
 * Call this after loading NNUE networks
 */
bool init_gpu_nnue(const void* big_weights, size_t big_size,
                   const void* small_weights, size_t small_size);

} // namespace MetalFish::GPU

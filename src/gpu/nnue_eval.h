/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Evaluation Header (Legacy interface)
  
  This provides backward compatibility. New code should use gpu_nnue.h
*/

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>

#include "backend.h"

namespace MetalFish {
class Position;
}

namespace MetalFish::GPU {

// Legacy constants
constexpr int NNUE_FEATURE_DIM_BIG = 1024;
constexpr int NNUE_FEATURE_DIM_SMALL = 128;
constexpr int MAX_BATCH_SIZE = 256;
constexpr int MAX_FEATURES_PER_POSITION = 32;
constexpr int HALFKA_DIMS = 45056;
constexpr int PSQT_DIMS = 8;

// Legacy batch structure
struct EvalBatch {
    int count = 0;
    std::vector<int32_t> features;
    std::vector<int32_t> feature_counts;
    std::vector<int32_t> psqt_scores;
    std::vector<int32_t> positional_scores;
    
    void clear();
    void reserve(int n);
};

// Legacy evaluator (uses new GPU NNUE interface internally)
class NNUEEvaluator {
public:
    NNUEEvaluator();
    ~NNUEEvaluator();
    
    NNUEEvaluator(const NNUEEvaluator&) = delete;
    NNUEEvaluator& operator=(const NNUEEvaluator&) = delete;
    
    bool initialize(const void* big_weights, size_t big_size,
                   const void* small_weights, size_t small_size);
    bool available() const { return initialized_; }
    bool evaluate_batch(EvalBatch& batch, bool use_big = true);
    int32_t evaluate(const Position& pos);
    
    int min_efficient_batch_size() const { return min_batch_size_; }
    void set_min_batch_size(int size) { min_batch_size_ = size; }
    
    size_t gpu_evaluations() const { return gpu_evals_; }
    size_t cpu_evaluations() const { return cpu_evals_; }
    double total_gpu_time_ms() const { return total_time_ms_; }
    size_t batch_count() const { return batch_count_; }
    double avg_batch_time_ms() const;
    void reset_stats();

private:
    bool initialized_ = false;
    int min_batch_size_ = 16;
    std::atomic<size_t> gpu_evals_{0};
    std::atomic<size_t> cpu_evals_{0};
    double total_time_ms_ = 0;
    size_t batch_count_ = 0;
};

// Legacy convenience functions
bool gpu_nnue_available();
NNUEEvaluator& gpu_nnue();

} // namespace MetalFish::GPU

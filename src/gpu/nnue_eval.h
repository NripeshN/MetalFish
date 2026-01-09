/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU-accelerated NNUE Evaluation

  This module provides batch NNUE evaluation on the GPU.
  Key optimizations:
  - Unified memory: Zero-copy access between CPU and GPU
  - Batch processing: Evaluate multiple positions in parallel
  - Incremental updates: Only update changed features
*/

#pragma once

#include "backend.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace MetalFish {

// Forward declarations
class Position;

namespace GPU {

// NNUE architecture constants (matching Stockfish)
constexpr int NNUE_FEATURE_DIM_BIG = 1024;
constexpr int NNUE_FEATURE_DIM_SMALL = 128;
constexpr int NNUE_FC0_OUT = 15;
constexpr int NNUE_FC1_OUT = 32;
constexpr int NNUE_PSQT_BUCKETS = 8;
constexpr int NNUE_LAYER_STACKS = 8;

// Maximum batch size for GPU evaluation
constexpr int MAX_BATCH_SIZE = 256;
constexpr int MAX_FEATURES_PER_POSITION = 64;

/**
 * GPU NNUE Accumulator State
 *
 * Stores the accumulated feature transformer output for a position.
 * On unified memory systems, this can be accessed by both CPU and GPU.
 */
struct GPUAccumulator {
  // Accumulator values for white and black perspectives
  alignas(64) int16_t accumulation[2][NNUE_FEATURE_DIM_BIG];

  // PSQT accumulation
  alignas(64) int32_t psqt[2][NNUE_PSQT_BUCKETS];

  // Computed flag
  bool computed = false;
};

/**
 * Batch of positions for GPU evaluation
 */
struct EvalBatch {
  // Number of positions in batch
  int count = 0;

  // Active feature indices for each position (sparse representation)
  // Format: [pos0_white_features..., pos0_black_features...,
  // pos1_white_features..., ...]
  std::vector<int32_t> features;

  // Feature counts per position [white_count, black_count] pairs
  std::vector<int32_t> feature_counts;

  // Piece counts for bucket selection
  std::vector<int32_t> piece_counts;

  // Output scores
  std::vector<int32_t> scores;

  void reserve(int batch_size) {
    features.reserve(batch_size * MAX_FEATURES_PER_POSITION * 2);
    feature_counts.reserve(batch_size * 2);
    piece_counts.reserve(batch_size);
    scores.resize(batch_size);
  }

  void clear() {
    count = 0;
    features.clear();
    feature_counts.clear();
    piece_counts.clear();
  }
};

/**
 * GPU NNUE Evaluator
 *
 * Manages GPU resources for NNUE evaluation and provides
 * batch evaluation functionality.
 */
class NNUEEvaluator {
public:
  NNUEEvaluator();
  ~NNUEEvaluator();

  // Initialize with network weights
  bool initialize(const void *big_weights, size_t big_size,
                  const void *small_weights, size_t small_size);

  // Check if GPU evaluation is available
  bool available() const { return initialized_ && gpu_available(); }

  // Evaluate a batch of positions
  // Returns true if GPU evaluation was used, false if fell back to CPU
  bool evaluate_batch(EvalBatch &batch);

  // Evaluate a single position (less efficient, prefer batching)
  int32_t evaluate(const Position &pos);

  // Get statistics
  size_t gpu_evals() const { return gpu_evals_; }
  size_t cpu_evals() const { return cpu_evals_; }
  double avg_batch_time_ms() const {
    return batch_count_ > 0 ? total_time_ms_ / batch_count_ : 0;
  }

  // Reset statistics
  void reset_stats() {
    gpu_evals_ = cpu_evals_ = batch_count_ = 0;
    total_time_ms_ = 0;
  }

private:
  bool initialized_ = false;

  // GPU buffers for network weights
  std::unique_ptr<Buffer> ft_weights_big_; // Feature transformer weights
  std::unique_ptr<Buffer> ft_biases_big_;  // Feature transformer biases
  std::unique_ptr<Buffer> ft_psqt_big_;    // PSQT weights
  std::unique_ptr<Buffer> fc0_weights_;    // FC0 layer weights
  std::unique_ptr<Buffer> fc0_biases_;     // FC0 layer biases
  std::unique_ptr<Buffer> fc1_weights_;    // FC1 layer weights
  std::unique_ptr<Buffer> fc1_biases_;     // FC1 layer biases
  std::unique_ptr<Buffer> fc2_weights_;    // FC2 layer weights
  std::unique_ptr<Buffer> fc2_biases_;     // FC2 layer biases

  // GPU buffers for batch processing
  std::unique_ptr<Buffer> features_buffer_; // Input features
  std::unique_ptr<Buffer> feature_counts_buffer_;
  std::unique_ptr<Buffer> accumulators_buffer_; // Intermediate accumulators
  std::unique_ptr<Buffer> output_buffer_;       // Output scores

  // Compute kernels
  std::unique_ptr<ComputeKernel> feature_transform_kernel_;
  std::unique_ptr<ComputeKernel> fc_forward_kernel_;
  std::unique_ptr<ComputeKernel> nnue_batch_kernel_;

  // Statistics
  size_t gpu_evals_ = 0;
  size_t cpu_evals_ = 0;
  size_t batch_count_ = 0;
  double total_time_ms_ = 0;

  // Internal methods
  bool load_kernels();
  bool allocate_buffers();
  void upload_weights(const void *weights, size_t size, Buffer *buffer);
};

/**
 * Global GPU NNUE evaluator instance
 */
NNUEEvaluator &gpu_nnue();

/**
 * Check if GPU NNUE evaluation is available
 */
inline bool gpu_nnue_available() {
  return gpu_available() && gpu_nnue().available();
}

} // namespace GPU
} // namespace MetalFish

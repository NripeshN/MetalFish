/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include <Metal/Metal.hpp>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace MetalFish {
namespace NNUE {

// NNUE architecture constants (matching Stockfish's HalfKAv2_hm)
constexpr int FEATURE_DIM_BIG = 1024;   // Big network hidden dimension
constexpr int FEATURE_DIM_SMALL = 128;  // Small network hidden dimension
constexpr int FC0_OUT = 15;             // First FC layer output
constexpr int FC1_OUT = 32;             // Second FC layer output
constexpr int PSQT_BUCKETS = 8;         // PSQT buckets
constexpr int MAX_ACTIVE_FEATURES = 32; // Max features per position
constexpr int INPUT_FEATURES = 45056;   // HalfKAv2_hm input features

// Weight types
using weight_t = int16_t;
using clipped_t = int8_t;
using acc_t = int32_t;

// Feature index for NNUE input
struct FeatureIndex {
  int white_idx;
  int black_idx;
};

// GPU Accumulator for NNUE - stores transformed features
struct GPUAccumulator {
  std::array<acc_t, FEATURE_DIM_BIG> white;
  std::array<acc_t, FEATURE_DIM_BIG> black;
  bool computed = false;
};

// GPU buffers for NNUE evaluation
struct NNUEBuffers {
  // Network weights (constant after loading)
  MTL::Buffer *ft_weights = nullptr;   // Feature transformer weights
  MTL::Buffer *ft_biases = nullptr;    // Feature transformer biases
  MTL::Buffer *fc0_weights = nullptr;  // FC0 weights
  MTL::Buffer *fc0_biases = nullptr;   // FC0 biases
  MTL::Buffer *fc1_weights = nullptr;  // FC1 weights
  MTL::Buffer *fc1_biases = nullptr;   // FC1 biases
  MTL::Buffer *fc2_weights = nullptr;  // FC2 weights
  MTL::Buffer *fc2_biases = nullptr;   // FC2 biases
  MTL::Buffer *psqt_weights = nullptr; // PSQT weights

  // Dynamic buffers for batch evaluation
  MTL::Buffer *active_features = nullptr;
  MTL::Buffer *feature_counts = nullptr;
  MTL::Buffer *accumulators = nullptr;
  MTL::Buffer *output = nullptr;

  ~NNUEBuffers();
  void release();
};

// GPU-accelerated NNUE evaluator
class NNUEEvaluator {
public:
  NNUEEvaluator();
  ~NNUEEvaluator();

  // Initialize GPU resources
  bool init();

  // Load network from file
  bool load_network(const std::string &path);

  // Evaluate a single position
  Value evaluate(const Position &pos);

  // Batch evaluate multiple positions
  void evaluate_batch(const std::vector<const Position *> &positions,
                      std::vector<Value> &results);

  // Get active features for a position
  static void get_active_features(const Position &pos, Color perspective,
                                  std::vector<int> &features);

  // Check if GPU is available
  bool is_gpu_available() const { return gpu_available_; }

  // Get batch size for GPU evaluation
  int get_batch_size() const { return batch_size_; }
  void set_batch_size(int size) { batch_size_ = size; }

private:
  // GPU resources
  MTL::Device *device_ = nullptr;
  MTL::CommandQueue *command_queue_ = nullptr;
  MTL::Library *library_ = nullptr;

  // Compute pipelines
  MTL::ComputePipelineState *ft_kernel_ = nullptr;
  MTL::ComputePipelineState *fc_kernel_ = nullptr;
  MTL::ComputePipelineState *forward_kernel_ = nullptr;
  MTL::ComputePipelineState *psqt_kernel_ = nullptr;

  // Buffers
  std::unique_ptr<NNUEBuffers> buffers_;

  // State
  bool gpu_available_ = false;
  bool network_loaded_ = false;
  int batch_size_ = 64;
  int hidden_dim_ = FEATURE_DIM_BIG;

  // CPU fallback weights (for when GPU isn't available)
  std::vector<weight_t> cpu_ft_weights_;
  std::vector<weight_t> cpu_ft_biases_;
  std::vector<weight_t> cpu_fc0_weights_;
  std::vector<weight_t> cpu_fc1_weights_;
  std::vector<weight_t> cpu_fc2_weights_;

  // Private methods
  bool load_kernels();
  bool create_buffers();
  void dispatch_feature_transform(const std::vector<int> &features,
                                  int num_positions);
  void dispatch_forward_pass(int num_positions);
  Value cpu_evaluate(const Position &pos);
};

// Global GPU NNUE instance
NNUEEvaluator &get_gpu_nnue();

} // namespace NNUE
} // namespace MetalFish

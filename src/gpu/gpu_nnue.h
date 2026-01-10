/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Weight Manager
*/

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "backend.h"
#include "gpu_constants.h"

// Forward declarations
namespace MetalFish {
class Position;
}

namespace MetalFish::Eval::NNUE {
struct Networks;
}

namespace MetalFish::GPU {

/**
 * GPU Network Weights
 */
struct GPUNetworkWeights {
  std::unique_ptr<Buffer> ft_weights;
  std::unique_ptr<Buffer> ft_biases;
  std::unique_ptr<Buffer> ft_psqt;
  std::unique_ptr<Buffer> threat_weights;
  std::unique_ptr<Buffer> threat_psqt;

  struct LayerWeights {
    std::unique_ptr<Buffer> fc0_weights;
    std::unique_ptr<Buffer> fc0_biases;
    std::unique_ptr<Buffer> fc1_weights;
    std::unique_ptr<Buffer> fc1_biases;
    std::unique_ptr<Buffer> fc2_weights;
    std::unique_ptr<Buffer> fc2_biases;
  };
  std::vector<LayerWeights> layers;

  int hidden_dim = 0;
  bool has_threats = false;
  bool valid = false;
};

/**
 * Position batch for GPU evaluation
 */
struct GPUPositionBatch {
  std::vector<int32_t> white_features;
  std::vector<int32_t> black_features;
  std::vector<int32_t> feature_counts;
  std::vector<int32_t> buckets;
  std::vector<int32_t> stm;
  int count = 0;

  void clear();
  void reserve(int n, int features_per_pos = 32);
};

/**
 * GPU evaluation results
 */
struct GPUEvalResults {
  std::vector<int32_t> psqt_scores;
  std::vector<int32_t> positional_scores;
  std::vector<int32_t> final_scores;
};

/**
 * GPU NNUE Weight Manager
 */
class GPUNNUEWeightManager {
public:
  GPUNNUEWeightManager();
  ~GPUNNUEWeightManager();

  bool load_networks(const Eval::NNUE::Networks &networks);
  bool big_network_ready() const { return big_weights_.valid; }
  bool small_network_ready() const { return small_weights_.valid; }
  const GPUNetworkWeights &big_weights() const { return big_weights_; }
  const GPUNetworkWeights &small_weights() const { return small_weights_; }
  size_t gpu_memory_used() const;

private:
  GPUNetworkWeights big_weights_;
  GPUNetworkWeights small_weights_;

  bool allocate_network_buffers(GPUNetworkWeights &weights, int hidden_dim,
                                bool has_threats);
};

/**
 * GPU NNUE Batch Evaluator
 */
class GPUNNUEBatchEvaluator {
public:
  GPUNNUEBatchEvaluator();
  ~GPUNNUEBatchEvaluator();

  bool initialize(GPUNNUEWeightManager &weights);
  bool ready() const { return initialized_; }

  void add_position(const Position &pos);
  void clear_batch() { batch_.clear(); }
  int batch_size() const { return batch_.count; }

  bool evaluate_big(GPUEvalResults &results);
  bool evaluate_small(GPUEvalResults &results);

  int min_efficient_batch() const { return min_batch_size_; }
  void set_min_batch_size(int size) { min_batch_size_ = size; }

  size_t total_gpu_evals() const { return gpu_evals_; }
  size_t total_batches() const { return batch_count_; }
  double avg_batch_time_ms() const;
  void reset_stats();

private:
  bool initialized_ = false;
  int min_batch_size_ = 8;

  GPUNNUEWeightManager *weights_ = nullptr;
  GPUPositionBatch batch_;

  std::unique_ptr<ComputeKernel> ft_kernel_;
  std::unique_ptr<ComputeKernel> forward_kernel_;

  std::unique_ptr<Buffer> features_buffer_;
  std::unique_ptr<Buffer> feature_offsets_buffer_;
  std::unique_ptr<Buffer> accumulators_buffer_;
  std::unique_ptr<Buffer> output_buffer_;

  std::atomic<size_t> gpu_evals_{0};
  std::atomic<size_t> batch_count_{0};
  double total_time_ms_ = 0;

  bool compile_kernels();
  bool allocate_buffers(int max_batch_size);
  bool dispatch_evaluation(const GPUNetworkWeights &weights,
                           GPUEvalResults &results);
};

/**
 * Global GPU NNUE interface
 */
class GPUNNUEInterface {
public:
  static GPUNNUEInterface &instance();

  bool initialize(const Eval::NNUE::Networks &networks);
  bool available() const { return initialized_; }
  GPUNNUEBatchEvaluator &evaluator() { return evaluator_; }
  GPUNNUEWeightManager &weights() { return weights_; }
  std::string status_string() const;

private:
  GPUNNUEInterface() = default;

  bool initialized_ = false;
  GPUNNUEWeightManager weights_;
  GPUNNUEBatchEvaluator evaluator_;
};

// Convenience functions
bool init_gpu_nnue(const Eval::NNUE::Networks &networks);
bool gpu_nnue_ready();
GPUNNUEInterface &gpu_nnue_interface();

} // namespace MetalFish::GPU

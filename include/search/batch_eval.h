/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Batch Evaluation for GPU
  ========================

  Collects leaf node positions during search and evaluates them in batches
  on the GPU for maximum throughput.

  Key optimizations:
  - Collect positions until batch is full or search completes
  - Use unified memory for zero-copy transfer
  - Pipeline CPU search with GPU evaluation
  - Speculative evaluation of likely positions
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include "metal/gpu_ops.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace MetalFish {
namespace Search {

// Configuration for batch evaluation
struct BatchEvalConfig {
  int batch_size = 64;    // Positions per batch
  int max_pending = 256;  // Max positions waiting for eval
  bool async_eval = true; // Use async GPU evaluation
  int prefetch_depth = 2; // Depth to start collecting positions
};

// Pending evaluation request
struct EvalRequest {
  Position *pos;
  int id;
  std::atomic<bool> *ready;
  std::atomic<Value> *result;
};

// Batch evaluator for GPU-accelerated position evaluation
class BatchEvaluator {
public:
  BatchEvaluator();
  ~BatchEvaluator();

  // Initialize with configuration
  void init(const BatchEvalConfig &config = {});

  // Submit position for evaluation
  // Returns immediately, result available when ready flag is set
  int submit(Position &pos, std::atomic<bool> &ready,
             std::atomic<Value> &result);

  // Get evaluation synchronously (blocks until ready)
  Value evaluate_sync(Position &pos);

  // Process pending evaluations (call from search thread)
  void process_batch();

  // Flush all pending evaluations
  void flush();

  // Statistics
  size_t get_batch_count() const { return batch_count_; }
  size_t get_eval_count() const { return eval_count_; }
  double get_avg_batch_size() const {
    return batch_count_ > 0 ? double(eval_count_) / batch_count_ : 0;
  }

  // Check if GPU evaluation is available
  bool is_gpu_available() const { return gpu_available_; }

private:
  // Pending positions
  std::vector<EvalRequest> pending_;
  std::mutex pending_mutex_;

  // Configuration
  BatchEvalConfig config_;
  bool initialized_ = false;
  bool gpu_available_ = false;

  // Statistics
  std::atomic<size_t> batch_count_{0};
  std::atomic<size_t> eval_count_{0};

  // Request ID counter
  std::atomic<int> next_id_{0};

  // Process a batch of positions on GPU
  void evaluate_batch(std::vector<EvalRequest> &batch);
};

// Global batch evaluator
extern BatchEvaluator g_batch_eval;

// Helper for search to use batch evaluation
inline Value lazy_evaluate(Position &pos, BatchEvaluator &evaluator) {
  // For now, use synchronous evaluation
  // In a pipelined implementation, this would check for ready results
  // and return cached values
  return evaluator.evaluate_sync(pos);
}

} // namespace Search
} // namespace MetalFish

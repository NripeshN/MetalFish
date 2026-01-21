/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Batch Evaluator - Unified Memory Optimized

  This module provides efficient batch evaluation for MCTS that leverages
  Apple Silicon's unified memory architecture:

  1. Zero-copy data sharing between CPU and GPU
  2. Persistent buffers that avoid allocation overhead
  3. Asynchronous evaluation with double-buffering
  4. Lock-free batch collection from multiple threads

  The key insight is that on Apple Silicon, CPU and GPU share the same
  physical memory, so we can avoid expensive copies by using shared buffers.

  Licensed under GPL-3.0
*/

#pragma once

#include "../gpu/backend.h"
#include "../gpu/gpu_nnue_integration.h"
#include "stockfish_adapter.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace MetalFish {
namespace MCTS {

// Forward declarations
class HybridNode;

// ============================================================================
// Unified Memory Batch Buffer
// ============================================================================

// A batch buffer that uses unified memory for zero-copy CPU/GPU access
struct UnifiedBatchBuffer {
  // Position data (CPU writes, GPU reads)
  std::vector<GPU::GPUPositionData> positions;

  // Features extracted on CPU (unified memory)
  std::vector<int32_t> white_features;
  std::vector<int32_t> black_features;
  std::vector<uint32_t> feature_counts;
  std::vector<uint32_t> feature_offsets;

  // Bucket indices for layer selection
  std::vector<int32_t> buckets;

  // Results (GPU writes, CPU reads)
  std::vector<int32_t> psqt_scores;
  std::vector<int32_t> positional_scores;

  // Node references for backpropagation
  std::vector<HybridNode *> nodes;
  std::vector<MCTSPosition> mcts_positions;

  // Batch metadata
  int count = 0;
  int capacity = 0;
  bool ready_for_gpu = false;
  bool results_ready = false;

  // Timing for profiling
  std::chrono::steady_clock::time_point submit_time;
  std::chrono::steady_clock::time_point complete_time;

  void allocate(int max_batch_size);
  void clear();
  void add_position(HybridNode *node, const MCTSPosition &pos);
  bool is_full() const { return count >= capacity; }
};

// ============================================================================
// Asynchronous Batch Evaluator
// ============================================================================

// Configuration for the batch evaluator
struct BatchEvaluatorConfig {
  int batch_size = 64;        // Target batch size
  int max_batch_size = 256;   // Maximum batch size
  int min_batch_size = 8;     // Minimum batch size before evaluation
  int batch_timeout_us = 500; // Timeout for batch collection (microseconds)
  int num_buffers = 3;        // Number of buffers for pipelining
  bool use_async = true;      // Use asynchronous evaluation
  bool profile = false;       // Enable profiling
};

// Statistics for the batch evaluator
struct BatchEvaluatorStats {
  std::atomic<uint64_t> total_batches{0};
  std::atomic<uint64_t> total_positions{0};
  std::atomic<uint64_t> timeouts{0};
  std::atomic<uint64_t> full_batches{0};

  double total_gpu_time_ms = 0;
  double total_wait_time_ms = 0;
  double avg_batch_size = 0;
  double avg_gpu_time_us = 0;

  void reset() {
    total_batches = 0;
    total_positions = 0;
    timeouts = 0;
    full_batches = 0;
    total_gpu_time_ms = 0;
    total_wait_time_ms = 0;
    avg_batch_size = 0;
    avg_gpu_time_us = 0;
  }
};

// Callback for when a batch evaluation completes
using BatchCompleteCallback = std::function<void(UnifiedBatchBuffer &batch)>;

// Asynchronous batch evaluator using unified memory
class MCTSBatchEvaluator {
public:
  MCTSBatchEvaluator();
  ~MCTSBatchEvaluator();

  // Initialize with GPU NNUE manager
  bool initialize(GPU::GPUNNUEManager *gpu_manager,
                  const BatchEvaluatorConfig &config);
  bool is_ready() const { return initialized_; }

  // Submit a position for evaluation
  // Returns true if position was added to batch
  // The callback will be called when the batch completes
  bool submit(HybridNode *node, const MCTSPosition &pos);

  // Force evaluation of current batch (even if not full)
  void flush();

  // Wait for all pending evaluations to complete
  void wait_all();

  // Stop the evaluator
  void stop();

  // Set callback for batch completion
  void set_callback(BatchCompleteCallback callback) { callback_ = callback; }

  // Get statistics
  const BatchEvaluatorStats &stats() const { return stats_; }

  // Direct synchronous evaluation (bypasses batching)
  float evaluate_single(const MCTSPosition &pos);

private:
  bool initialized_ = false;
  BatchEvaluatorConfig config_;
  BatchEvaluatorStats stats_;

  GPU::GPUNNUEManager *gpu_manager_ = nullptr;
  BatchCompleteCallback callback_;

  // Buffer pool for double/triple buffering
  std::vector<std::unique_ptr<UnifiedBatchBuffer>> buffers_;
  int current_buffer_idx_ = 0;

  // Synchronization
  std::mutex submit_mutex_;
  std::condition_variable submit_cv_;
  std::atomic<bool> stop_flag_{false};

  // Evaluation thread
  std::thread eval_thread_;

  // Queue of buffers ready for GPU evaluation
  std::queue<UnifiedBatchBuffer *> pending_queue_;
  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;

  // Internal methods
  void eval_thread_main();
  void evaluate_batch(UnifiedBatchBuffer &batch);
  UnifiedBatchBuffer *get_current_buffer();
  void submit_current_buffer();
};

// ============================================================================
// Lock-Free Batch Collector
// ============================================================================

// A lock-free batch collector for multiple search threads
// Uses atomic operations to minimize contention
class LockFreeBatchCollector {
public:
  LockFreeBatchCollector(int max_batch_size = 256);
  ~LockFreeBatchCollector();

  // Add a position to the batch (thread-safe)
  // Returns the index in the batch, or -1 if batch is full
  int add(HybridNode *node, const MCTSPosition &pos);

  // Check if batch is ready for evaluation
  bool is_ready(int min_size) const;

  // Get the current batch and reset
  // Returns the number of positions in the batch
  int take_batch(std::vector<HybridNode *> &nodes,
                 std::vector<MCTSPosition> &positions);

  // Get current size
  int size() const { return size_.load(std::memory_order_relaxed); }

private:
  int max_size_;
  std::atomic<int> size_{0};

  // Pre-allocated storage (protected by mutex)
  std::vector<HybridNode *> nodes_;
  std::vector<MCTSPosition> positions_;
  std::mutex mutex_;
};

// ============================================================================
// Unified Memory Pool
// ============================================================================

// Memory pool that uses Metal's unified memory
// Provides fast allocation for batch buffers
class UnifiedMemoryPool {
public:
  UnifiedMemoryPool();
  ~UnifiedMemoryPool();

  // Initialize with GPU backend
  bool initialize(size_t pool_size_bytes);

  // Allocate from pool
  void *allocate(size_t size, size_t alignment = 16);

  // Free to pool
  void free(void *ptr);

  // Get total allocated
  size_t allocated() const { return allocated_; }

  // Get total pool size
  size_t pool_size() const { return pool_size_; }

private:
  size_t pool_size_ = 0;
  size_t allocated_ = 0;

  // Simple bump allocator for now
  std::unique_ptr<uint8_t[]> pool_;
  size_t offset_ = 0;
  std::mutex mutex_;
};

// ============================================================================
// Global Batch Evaluator
// ============================================================================

// Get the global MCTS batch evaluator
MCTSBatchEvaluator &mcts_batch_evaluator();

// Initialize the global batch evaluator
bool initialize_mcts_batch_evaluator(
    GPU::GPUNNUEManager *gpu_manager,
    const BatchEvaluatorConfig &config = BatchEvaluatorConfig());

// Shutdown the global batch evaluator
void shutdown_mcts_batch_evaluator();

} // namespace MCTS
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Persistent Metal Pipeline

  This module provides persistent Metal command buffers and pipelines
  to minimize dispatch overhead. Key optimizations:

  1. Pre-allocated command buffers that are reused
  2. Persistent compute pipeline states
  3. Double-buffered execution for overlap
  4. Unified memory buffer pools

  Licensed under GPL-3.0
*/

#pragma once

#include "backend.h"
#include "gpu_constants.h"
#include "gpu_nnue_integration.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace MetalFish {
namespace GPU {

// ============================================================================
// Persistent Buffer Pool
// ============================================================================

// A pool of pre-allocated unified memory buffers
class PersistentBufferPool {
public:
  PersistentBufferPool();
  ~PersistentBufferPool();

  // Initialize pool with specified sizes
  bool initialize(size_t small_size, int num_small, size_t medium_size,
                  int num_medium, size_t large_size, int num_large);

  // Acquire a buffer of at least the specified size
  Buffer *acquire(size_t min_size);

  // Release a buffer back to the pool
  void release(Buffer *buffer);

  // Get pool statistics
  size_t total_memory() const { return total_memory_; }
  int buffers_in_use() const { return buffers_in_use_.load(); }

private:
  struct BufferEntry {
    std::unique_ptr<Buffer> buffer;
    bool in_use = false;
  };

  std::vector<BufferEntry> small_buffers_;
  std::vector<BufferEntry> medium_buffers_;
  std::vector<BufferEntry> large_buffers_;

  size_t small_size_ = 0;
  size_t medium_size_ = 0;
  size_t large_size_ = 0;

  size_t total_memory_ = 0;
  std::atomic<int> buffers_in_use_{0};
  std::mutex mutex_;
};

// ============================================================================
// Persistent Command Buffer
// ============================================================================

// A reusable command buffer with pre-encoded commands
class PersistentCommandBuffer {
public:
  PersistentCommandBuffer();
  ~PersistentCommandBuffer();

  // Initialize with kernel
  bool initialize(ComputeKernel *kernel, int max_batch_size);

  // Set input/output buffers (must be called before encode)
  void set_input_buffer(Buffer *buffer, int index);
  void set_output_buffer(Buffer *buffer, int index);
  void set_constant(const void *data, size_t size, int index);

  // Encode commands for batch size
  bool encode(int batch_size);

  // Submit and wait
  void submit_and_wait();

  // Submit without waiting (for pipelining)
  void submit_async();

  // Wait for completion
  void wait();

  // Check if completed
  bool is_complete() const;

  // Reset for reuse
  void reset();

private:
  ComputeKernel *kernel_ = nullptr;
  int max_batch_size_ = 0;

  std::unique_ptr<CommandEncoder> encoder_;
  bool encoded_ = false;
  bool submitted_ = false;

  std::vector<Buffer *> input_buffers_;
  std::vector<Buffer *> output_buffers_;
};

// ============================================================================
// Double-Buffered Pipeline
// ============================================================================

// Double-buffered execution for overlapping CPU prep and GPU execution
class DoubleBufferedPipeline {
public:
  DoubleBufferedPipeline();
  ~DoubleBufferedPipeline();

  // Initialize with kernel and buffer sizes
  bool initialize(ComputeKernel *kernel, size_t input_size, size_t output_size,
                  int max_batch_size);

  // Get current input buffer for CPU to fill
  Buffer *get_input_buffer();

  // Get current output buffer for CPU to read
  Buffer *get_output_buffer();

  // Submit current buffer and swap
  void submit(int batch_size);

  // Wait for previous submission
  void wait_previous();

  // Wait for current submission
  void wait_current();

  // Swap buffers (call after wait_previous)
  void swap();

private:
  int current_idx_ = 0;

  struct BufferSet {
    std::unique_ptr<Buffer> input;
    std::unique_ptr<Buffer> output;
    std::unique_ptr<PersistentCommandBuffer> cmd;
    bool pending = false;
  };

  BufferSet buffers_[2];
  ComputeKernel *kernel_ = nullptr;
};

// ============================================================================
// Persistent NNUE Pipeline
// ============================================================================

// Optimized NNUE evaluation pipeline with persistent resources
class PersistentNNUEPipeline {
public:
  PersistentNNUEPipeline();
  ~PersistentNNUEPipeline();

  // Initialize with GPU backend
  bool initialize(int max_batch_size = 256);

  // Set network weights (call once after loading)
  void set_weights(Buffer *ft_weights, Buffer *ft_biases, Buffer *fc_weights,
                   Buffer *fc_biases);

  // Evaluate batch of positions
  // Input: features buffer, feature counts, feature offsets
  // Output: scores buffer
  bool evaluate(Buffer *features, Buffer *counts, Buffer *offsets,
                Buffer *scores, int batch_size);

  // Async evaluation (returns immediately)
  // Returns false if evaluation cannot be started
  bool evaluate_async(Buffer *features, Buffer *counts, Buffer *offsets,
                      Buffer *scores, int batch_size);

  // Wait for async evaluation
  void wait();

  // Get statistics
  uint64_t total_evaluations() const { return total_evals_; }
  double avg_latency_us() const;

private:
  bool initialized_ = false;
  int max_batch_size_ = 256;

  // Kernels
  std::unique_ptr<ComputeKernel> feature_transform_kernel_;
  std::unique_ptr<ComputeKernel> forward_kernel_;

  // Persistent buffers
  std::unique_ptr<Buffer> accumulator_buffer_;
  std::unique_ptr<Buffer> hidden_buffer_;

  // Weight buffers (external)
  Buffer *ft_weights_ = nullptr;
  Buffer *ft_biases_ = nullptr;
  Buffer *fc_weights_ = nullptr;
  Buffer *fc_biases_ = nullptr;

  // Double-buffered command encoders
  std::unique_ptr<DoubleBufferedPipeline> pipeline_;

  // Statistics
  uint64_t total_evals_ = 0;
  double total_latency_us_ = 0;
  std::mutex stats_mutex_;
};

// ============================================================================
// Persistent Batch Evaluator
// ============================================================================

// High-performance batch evaluator using persistent resources
class PersistentBatchEvaluator {
public:
  PersistentBatchEvaluator();
  ~PersistentBatchEvaluator();

  // Initialize
  bool initialize(int max_batch_size = 256);

  // Set GPU NNUE manager
  void set_gpu_nnue(GPUNNUEManager *manager) { gpu_manager_ = manager; }

  // Evaluate positions
  // positions: array of position data
  // scores: output scores (must be pre-allocated)
  // Returns number of positions evaluated
  int evaluate(const GPUPositionData *positions, int count,
               int32_t *psqt_scores, int32_t *positional_scores);

  // Async evaluation
  void evaluate_async(const GPUPositionData *positions, int count);

  // Get results from async evaluation
  int get_results(int32_t *psqt_scores, int32_t *positional_scores);

  // Wait for async evaluation
  void wait();

  // Statistics
  uint64_t total_batches() const { return total_batches_; }
  uint64_t total_positions() const { return total_positions_; }
  double avg_batch_latency_us() const;

private:
  bool initialized_ = false;
  int max_batch_size_ = 256;

  GPUNNUEManager *gpu_manager_ = nullptr;

  // Persistent input/output buffers
  std::unique_ptr<Buffer> position_buffer_;
  std::unique_ptr<Buffer> feature_buffer_;
  std::unique_ptr<Buffer> count_buffer_;
  std::unique_ptr<Buffer> offset_buffer_;
  std::unique_ptr<Buffer> psqt_buffer_;
  std::unique_ptr<Buffer> positional_buffer_;

  // Async state
  int pending_count_ = 0;
  bool async_pending_ = false;

  // Statistics
  uint64_t total_batches_ = 0;
  uint64_t total_positions_ = 0;
  double total_latency_us_ = 0;
};

// ============================================================================
// Global Persistent Resources
// ============================================================================

// Get the global buffer pool
PersistentBufferPool &persistent_buffer_pool();

// Initialize global persistent resources
bool initialize_persistent_resources();

// Get the global persistent evaluator
PersistentBatchEvaluator &persistent_evaluator();

} // namespace GPU
} // namespace MetalFish

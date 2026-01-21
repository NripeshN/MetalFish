/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Batch Evaluator - Implementation

  Licensed under GPL-3.0
*/

#include "mcts_batch_evaluator.h"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// UnifiedBatchBuffer
// ============================================================================

void UnifiedBatchBuffer::allocate(int max_batch_size) {
  capacity = max_batch_size;

  // Pre-allocate all vectors
  positions.resize(max_batch_size);

  // Features: assume max 64 features per perspective per position
  int max_features = max_batch_size * 64;
  white_features.resize(max_features);
  black_features.resize(max_features);
  feature_counts.resize(max_batch_size * 2); // 2 perspectives
  feature_offsets.resize(max_batch_size * 2);

  buckets.resize(max_batch_size);
  psqt_scores.resize(max_batch_size);
  positional_scores.resize(max_batch_size);

  nodes.resize(max_batch_size);
  mcts_positions.resize(max_batch_size);

  clear();
}

void UnifiedBatchBuffer::clear() {
  count = 0;
  ready_for_gpu = false;
  results_ready = false;
}

void UnifiedBatchBuffer::add_position(HybridNode *node,
                                      const MCTSPosition &pos) {
  if (count >= capacity)
    return;

  nodes[count] = node;
  mcts_positions[count] = pos;

  // Convert to GPU format
  positions[count].from_position(pos.stockfish_position());

  ++count;
}

// ============================================================================
// MCTSBatchEvaluator
// ============================================================================

MCTSBatchEvaluator::MCTSBatchEvaluator() = default;

MCTSBatchEvaluator::~MCTSBatchEvaluator() { stop(); }

bool MCTSBatchEvaluator::initialize(GPU::GPUNNUEManager *gpu_manager,
                                    const BatchEvaluatorConfig &config) {
  if (!gpu_manager || !gpu_manager->is_ready()) {
    return false;
  }

  gpu_manager_ = gpu_manager;
  config_ = config;

  // Allocate buffers
  buffers_.resize(config.num_buffers);
  for (int i = 0; i < config.num_buffers; ++i) {
    buffers_[i] = std::make_unique<UnifiedBatchBuffer>();
    buffers_[i]->allocate(config.max_batch_size);
  }

  // Start evaluation thread if async
  if (config.use_async) {
    stop_flag_ = false;
    eval_thread_ = std::thread(&MCTSBatchEvaluator::eval_thread_main, this);
  }

  initialized_ = true;
  return true;
}

bool MCTSBatchEvaluator::submit(HybridNode *node, const MCTSPosition &pos) {
  if (!initialized_)
    return false;

  std::lock_guard<std::mutex> lock(submit_mutex_);

  UnifiedBatchBuffer *buffer = get_current_buffer();
  buffer->add_position(node, pos);

  // Check if batch is full
  if (buffer->count >= config_.batch_size) {
    submit_current_buffer();
  }

  return true;
}

void MCTSBatchEvaluator::flush() {
  if (!initialized_)
    return;

  std::lock_guard<std::mutex> lock(submit_mutex_);

  UnifiedBatchBuffer *buffer = get_current_buffer();
  if (buffer->count > 0) {
    submit_current_buffer();
  }
}

void MCTSBatchEvaluator::wait_all() {
  flush();

  if (config_.use_async) {
    // Wait for queue to empty
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock,
                   [this] { return pending_queue_.empty() || stop_flag_; });
  }
}

void MCTSBatchEvaluator::stop() {
  stop_flag_ = true;
  submit_cv_.notify_all();
  queue_cv_.notify_all();

  if (eval_thread_.joinable()) {
    eval_thread_.join();
  }

  initialized_ = false;
}

float MCTSBatchEvaluator::evaluate_single(const MCTSPosition &pos) {
  if (!gpu_manager_)
    return 0.0f;

  auto [psqt, score] =
      gpu_manager_->evaluate_single(pos.stockfish_position(), true);

  // Convert to MCTS value in [-1, 1]
  float value = std::tanh(score / 400.0f);

  // Flip for black
  if (pos.is_black_to_move()) {
    value = -value;
  }

  return value;
}

UnifiedBatchBuffer *MCTSBatchEvaluator::get_current_buffer() {
  return buffers_[current_buffer_idx_].get();
}

void MCTSBatchEvaluator::submit_current_buffer() {
  UnifiedBatchBuffer *buffer = get_current_buffer();
  buffer->ready_for_gpu = true;
  buffer->submit_time = std::chrono::steady_clock::now();

  if (config_.use_async) {
    // Add to pending queue
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_queue_.push(buffer);
    }
    queue_cv_.notify_one();

    // Move to next buffer
    current_buffer_idx_ = (current_buffer_idx_ + 1) % config_.num_buffers;

    // Wait if next buffer is still being processed
    UnifiedBatchBuffer *next = get_current_buffer();
    while (next->ready_for_gpu && !next->results_ready && !stop_flag_) {
      std::this_thread::yield();
    }
    next->clear();
  } else {
    // Synchronous evaluation
    evaluate_batch(*buffer);

    // Call callback
    if (callback_) {
      callback_(*buffer);
    }

    buffer->clear();
  }
}

void MCTSBatchEvaluator::eval_thread_main() {
  while (!stop_flag_) {
    UnifiedBatchBuffer *batch = nullptr;

    // Wait for batch
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);

      // Wait with timeout
      bool got_batch = queue_cv_.wait_for(
          lock, std::chrono::microseconds(config_.batch_timeout_us),
          [this] { return !pending_queue_.empty() || stop_flag_; });

      if (stop_flag_)
        break;

      if (!pending_queue_.empty()) {
        batch = pending_queue_.front();
        pending_queue_.pop();
      } else {
        stats_.timeouts++;
        continue;
      }
    }

    if (batch && batch->count > 0) {
      // Evaluate batch on GPU
      evaluate_batch(*batch);

      // Call callback
      if (callback_) {
        callback_(*batch);
      }

      // Mark results ready
      batch->results_ready = true;
    }

    // Notify waiters
    queue_cv_.notify_all();
  }
}

void MCTSBatchEvaluator::evaluate_batch(UnifiedBatchBuffer &batch) {
  if (!gpu_manager_ || batch.count == 0)
    return;

  auto start_time = std::chrono::steady_clock::now();

  // Create GPU eval batch
  GPU::GPUEvalBatch gpu_batch;
  gpu_batch.count = batch.count;
  gpu_batch.reserve(batch.count);

  // Extract features for each position
  for (int i = 0; i < batch.count; ++i) {
    gpu_batch.add_position(batch.mcts_positions[i].stockfish_position());
  }

  // Evaluate on GPU
  bool success = gpu_manager_->evaluate_batch(gpu_batch, true);

  if (success) {
    // Copy results back
    // Note: GPUNNUEManager::evaluate_batch only populates positional_scores,
    // not psqt_scores. Guard against accessing empty vector.
    const bool has_psqt = !gpu_batch.psqt_scores.empty();
    for (int i = 0; i < batch.count; ++i) {
      batch.psqt_scores[i] = has_psqt ? gpu_batch.psqt_scores[i] : 0;
      batch.positional_scores[i] = gpu_batch.positional_scores[i];
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  batch.complete_time = end_time;

  // Update statistics
  double gpu_time_us =
      std::chrono::duration<double, std::micro>(end_time - start_time).count();
  stats_.total_batches++;
  stats_.total_positions += batch.count;
  stats_.total_gpu_time_ms += gpu_time_us / 1000.0;

  if (batch.count >= config_.batch_size) {
    stats_.full_batches++;
  }

  // Update averages
  uint64_t total = stats_.total_batches.load();
  if (total > 0) {
    stats_.avg_batch_size =
        static_cast<double>(stats_.total_positions.load()) / total;
    stats_.avg_gpu_time_us = stats_.total_gpu_time_ms * 1000.0 / total;
  }
}

// ============================================================================
// LockFreeBatchCollector
// ============================================================================

LockFreeBatchCollector::LockFreeBatchCollector(int max_batch_size)
    : max_size_(max_batch_size) {
  nodes_.resize(max_batch_size, nullptr);
  positions_.resize(max_batch_size);
}

LockFreeBatchCollector::~LockFreeBatchCollector() = default;

int LockFreeBatchCollector::add(HybridNode *node, const MCTSPosition &pos) {
  std::lock_guard<std::mutex> lock(mutex_);

  int idx = size_.load(std::memory_order_relaxed);
  if (idx >= max_size_) {
    return -1;
  }

  nodes_[idx] = node;
  positions_[idx] = pos;
  size_.fetch_add(1, std::memory_order_release);

  return idx;
}

bool LockFreeBatchCollector::is_ready(int min_size) const {
  return size_.load(std::memory_order_relaxed) >= min_size;
}

int LockFreeBatchCollector::take_batch(std::vector<HybridNode *> &nodes,
                                       std::vector<MCTSPosition> &positions) {
  std::lock_guard<std::mutex> lock(mutex_);

  int count = size_.exchange(0, std::memory_order_acquire);

  if (count == 0)
    return 0;

  nodes.resize(count);
  positions.resize(count);

  for (int i = 0; i < count; ++i) {
    nodes[i] = nodes_[i];
    nodes_[i] = nullptr;
    positions[i] = std::move(positions_[i]);
  }

  return count;
}

// ============================================================================
// UnifiedMemoryPool
// ============================================================================

UnifiedMemoryPool::UnifiedMemoryPool() = default;

UnifiedMemoryPool::~UnifiedMemoryPool() = default;

bool UnifiedMemoryPool::initialize(size_t pool_size_bytes) {
  pool_size_ = pool_size_bytes;
  pool_ = std::make_unique<uint8_t[]>(pool_size_bytes);
  offset_ = 0;
  allocated_ = 0;
  return true;
}

void *UnifiedMemoryPool::allocate(size_t size, size_t alignment) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Align offset
  size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

  if (aligned_offset + size > pool_size_) {
    // Pool exhausted, wrap around (simple strategy)
    aligned_offset = 0;
  }

  void *ptr = pool_.get() + aligned_offset;
  offset_ = aligned_offset + size;
  allocated_ += size;

  return ptr;
}

void UnifiedMemoryPool::free(void *ptr) {
  // Simple bump allocator doesn't support individual frees
  // Memory is reclaimed when pool wraps around
}

// ============================================================================
// Global Instance
// ============================================================================

static std::unique_ptr<MCTSBatchEvaluator> g_mcts_batch_evaluator;

MCTSBatchEvaluator &mcts_batch_evaluator() {
  if (!g_mcts_batch_evaluator) {
    g_mcts_batch_evaluator = std::make_unique<MCTSBatchEvaluator>();
  }
  return *g_mcts_batch_evaluator;
}

bool initialize_mcts_batch_evaluator(GPU::GPUNNUEManager *gpu_manager,
                                     const BatchEvaluatorConfig &config) {
  return mcts_batch_evaluator().initialize(gpu_manager, config);
}

void shutdown_mcts_batch_evaluator() {
  if (g_mcts_batch_evaluator) {
    g_mcts_batch_evaluator->stop();
    g_mcts_batch_evaluator.reset();
  }
}

} // namespace MCTS
} // namespace MetalFish

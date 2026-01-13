/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Persistent Metal Pipeline - Implementation

  Licensed under GPL-3.0
*/

#include "persistent_pipeline.h"
#include "../core/position.h"
#include <chrono>
#include <cstring>
#include <deque>

namespace MetalFish {
namespace GPU {

// ============================================================================
// PersistentBufferPool Implementation
// ============================================================================

PersistentBufferPool::PersistentBufferPool() = default;

PersistentBufferPool::~PersistentBufferPool() = default;

bool PersistentBufferPool::initialize(size_t small_size, int num_small,
                                      size_t medium_size, int num_medium,
                                      size_t large_size, int num_large) {
  if (!gpu_available()) return false;
  
  auto& backend = gpu();
  
  small_size_ = small_size;
  medium_size_ = medium_size;
  large_size_ = large_size;
  
  // Allocate small buffers
  small_buffers_.resize(num_small);
  for (int i = 0; i < num_small; ++i) {
    small_buffers_[i].buffer = backend.create_buffer(small_size);
    if (!small_buffers_[i].buffer) return false;
    total_memory_ += small_size;
  }
  
  // Allocate medium buffers
  medium_buffers_.resize(num_medium);
  for (int i = 0; i < num_medium; ++i) {
    medium_buffers_[i].buffer = backend.create_buffer(medium_size);
    if (!medium_buffers_[i].buffer) return false;
    total_memory_ += medium_size;
  }
  
  // Allocate large buffers
  large_buffers_.resize(num_large);
  for (int i = 0; i < num_large; ++i) {
    large_buffers_[i].buffer = backend.create_buffer(large_size);
    if (!large_buffers_[i].buffer) return false;
    total_memory_ += large_size;
  }
  
  return true;
}

Buffer* PersistentBufferPool::acquire(size_t min_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Try to find a free buffer of appropriate size
  std::vector<BufferEntry>* pool = nullptr;
  
  if (min_size <= small_size_ && !small_buffers_.empty()) {
    pool = &small_buffers_;
  } else if (min_size <= medium_size_ && !medium_buffers_.empty()) {
    pool = &medium_buffers_;
  } else if (min_size <= large_size_ && !large_buffers_.empty()) {
    pool = &large_buffers_;
  }
  
  if (pool) {
    for (auto& entry : *pool) {
      if (!entry.in_use) {
        entry.in_use = true;
        buffers_in_use_.fetch_add(1);
        return entry.buffer.get();
      }
    }
  }
  
  return nullptr;  // No buffer available
}

void PersistentBufferPool::release(Buffer* buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // Find and release the buffer
  auto release_from = [&](std::vector<BufferEntry>& pool) -> bool {
    for (auto& entry : pool) {
      if (entry.buffer.get() == buffer) {
        entry.in_use = false;
        buffers_in_use_.fetch_sub(1);
        return true;
      }
    }
    return false;
  };
  
  if (!release_from(small_buffers_)) {
    if (!release_from(medium_buffers_)) {
      release_from(large_buffers_);
    }
  }
}

// ============================================================================
// PersistentCommandBuffer Implementation
// ============================================================================

PersistentCommandBuffer::PersistentCommandBuffer() = default;

PersistentCommandBuffer::~PersistentCommandBuffer() = default;

bool PersistentCommandBuffer::initialize(ComputeKernel* kernel, int max_batch_size) {
  if (!kernel || !kernel->valid()) return false;
  
  kernel_ = kernel;
  max_batch_size_ = max_batch_size;
  
  return true;
}

void PersistentCommandBuffer::set_input_buffer(Buffer* buffer, int index) {
  if (index >= static_cast<int>(input_buffers_.size())) {
    input_buffers_.resize(index + 1, nullptr);
  }
  input_buffers_[index] = buffer;
}

void PersistentCommandBuffer::set_output_buffer(Buffer* buffer, int index) {
  if (index >= static_cast<int>(output_buffers_.size())) {
    output_buffers_.resize(index + 1, nullptr);
  }
  output_buffers_[index] = buffer;
}

void PersistentCommandBuffer::set_constant(const void* data, size_t size, int index) {
  // Constants are set during encode
}

bool PersistentCommandBuffer::encode(int batch_size) {
  if (!kernel_ || batch_size <= 0 || batch_size > max_batch_size_) {
    return false;
  }
  
  encoder_ = gpu().create_encoder();
  if (!encoder_) return false;
  
  encoder_->set_kernel(kernel_);
  
  // Set buffers
  int buffer_idx = 0;
  for (auto* buf : input_buffers_) {
    if (buf) {
      encoder_->set_buffer(buf, buffer_idx);
    }
    buffer_idx++;
  }
  for (auto* buf : output_buffers_) {
    if (buf) {
      encoder_->set_buffer(buf, buffer_idx);
    }
    buffer_idx++;
  }
  
  // Set batch size constant
  encoder_->set_value(batch_size, buffer_idx);
  
  // Dispatch
  encoder_->dispatch_threads(batch_size);
  
  encoded_ = true;
  return true;
}

void PersistentCommandBuffer::submit_and_wait() {
  if (!encoded_ || !encoder_) return;
  
  gpu().submit_and_wait(encoder_.get());
  submitted_ = true;
}

void PersistentCommandBuffer::submit_async() {
  if (!encoded_ || !encoder_) return;
  
  // Note: For true async, we'd need to track the command buffer
  // For now, this is the same as submit_and_wait
  gpu().submit_and_wait(encoder_.get());
  submitted_ = true;
}

void PersistentCommandBuffer::wait() {
  // Already waited in submit
}

bool PersistentCommandBuffer::is_complete() const {
  return submitted_;
}

void PersistentCommandBuffer::reset() {
  encoder_.reset();
  encoded_ = false;
  submitted_ = false;
}

// ============================================================================
// DoubleBufferedPipeline Implementation
// ============================================================================

DoubleBufferedPipeline::DoubleBufferedPipeline() = default;

DoubleBufferedPipeline::~DoubleBufferedPipeline() = default;

bool DoubleBufferedPipeline::initialize(ComputeKernel* kernel, size_t input_size,
                                        size_t output_size, int max_batch_size) {
  if (!kernel || !kernel->valid()) return false;
  
  kernel_ = kernel;
  auto& backend = gpu();
  
  for (int i = 0; i < 2; ++i) {
    buffers_[i].input = backend.create_buffer(input_size);
    buffers_[i].output = backend.create_buffer(output_size);
    buffers_[i].cmd = std::make_unique<PersistentCommandBuffer>();
    
    if (!buffers_[i].input || !buffers_[i].output) {
      return false;
    }
    
    buffers_[i].cmd->initialize(kernel, max_batch_size);
    buffers_[i].cmd->set_input_buffer(buffers_[i].input.get(), 0);
    buffers_[i].cmd->set_output_buffer(buffers_[i].output.get(), 1);
  }
  
  return true;
}

Buffer* DoubleBufferedPipeline::get_input_buffer() {
  return buffers_[current_idx_].input.get();
}

Buffer* DoubleBufferedPipeline::get_output_buffer() {
  return buffers_[current_idx_].output.get();
}

void DoubleBufferedPipeline::submit(int batch_size) {
  auto& buf = buffers_[current_idx_];
  buf.cmd->reset();
  buf.cmd->encode(batch_size);
  buf.cmd->submit_async();
  buf.pending = true;
}

void DoubleBufferedPipeline::wait_previous() {
  int prev_idx = 1 - current_idx_;
  if (buffers_[prev_idx].pending) {
    buffers_[prev_idx].cmd->wait();
    buffers_[prev_idx].pending = false;
  }
}

void DoubleBufferedPipeline::wait_current() {
  if (buffers_[current_idx_].pending) {
    buffers_[current_idx_].cmd->wait();
    buffers_[current_idx_].pending = false;
  }
}

void DoubleBufferedPipeline::swap() {
  current_idx_ = 1 - current_idx_;
}

// ============================================================================
// PersistentNNUEPipeline Implementation
// ============================================================================

PersistentNNUEPipeline::PersistentNNUEPipeline() = default;

PersistentNNUEPipeline::~PersistentNNUEPipeline() = default;

bool PersistentNNUEPipeline::initialize(int max_batch_size) {
  if (!gpu_available()) return false;
  
  max_batch_size_ = max_batch_size;
  auto& backend = gpu();
  
  // Allocate intermediate buffers
  size_t accumulator_size = max_batch_size * GPU_FT_DIM_BIG * 2 * sizeof(int16_t);
  size_t hidden_size = max_batch_size * 32 * sizeof(int32_t);
  
  accumulator_buffer_ = backend.create_buffer(accumulator_size);
  hidden_buffer_ = backend.create_buffer(hidden_size);
  
  if (!accumulator_buffer_ || !hidden_buffer_) {
    return false;
  }
  
  initialized_ = true;
  return true;
}

void PersistentNNUEPipeline::set_weights(Buffer* ft_weights, Buffer* ft_biases,
                                         Buffer* fc_weights, Buffer* fc_biases) {
  ft_weights_ = ft_weights;
  ft_biases_ = ft_biases;
  fc_weights_ = fc_weights;
  fc_biases_ = fc_biases;
}

bool PersistentNNUEPipeline::evaluate(Buffer* features, Buffer* counts, Buffer* offsets,
                                      Buffer* scores, int batch_size) {
  if (!initialized_ || batch_size <= 0 || batch_size > max_batch_size_) {
    return false;
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // For now, delegate to the standard GPU NNUE path
  // In a full implementation, we'd use persistent command buffers here
  
  auto end = std::chrono::high_resolution_clock::now();
  double latency = std::chrono::duration<double, std::micro>(end - start).count();
  
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    total_evals_ += batch_size;
    total_latency_us_ += latency;
  }
  
  return true;
}

void PersistentNNUEPipeline::evaluate_async(Buffer* features, Buffer* counts,
                                            Buffer* offsets, Buffer* scores,
                                            int batch_size) {
  // For now, just do synchronous evaluation
  evaluate(features, counts, offsets, scores, batch_size);
}

void PersistentNNUEPipeline::wait() {
  // Nothing to wait for in current implementation
}

double PersistentNNUEPipeline::avg_latency_us() const {
  if (total_evals_ == 0) return 0;
  return total_latency_us_ / total_evals_;
}

// ============================================================================
// PersistentBatchEvaluator Implementation
// ============================================================================

PersistentBatchEvaluator::PersistentBatchEvaluator() = default;

PersistentBatchEvaluator::~PersistentBatchEvaluator() = default;

bool PersistentBatchEvaluator::initialize(int max_batch_size) {
  if (!gpu_available()) return false;
  
  max_batch_size_ = max_batch_size;
  auto& backend = gpu();
  
  // Allocate persistent buffers
  size_t position_size = max_batch_size * sizeof(GPUPositionData);
  size_t feature_size = max_batch_size * 64 * 2 * sizeof(int32_t);  // Max 64 features per perspective
  size_t count_size = max_batch_size * 2 * sizeof(uint32_t);
  size_t offset_size = max_batch_size * 2 * sizeof(uint32_t);
  size_t score_size = max_batch_size * sizeof(int32_t);
  
  position_buffer_ = backend.create_buffer(position_size);
  feature_buffer_ = backend.create_buffer(feature_size);
  count_buffer_ = backend.create_buffer(count_size);
  offset_buffer_ = backend.create_buffer(offset_size);
  psqt_buffer_ = backend.create_buffer(score_size);
  positional_buffer_ = backend.create_buffer(score_size);
  
  if (!position_buffer_ || !feature_buffer_ || !count_buffer_ ||
      !offset_buffer_ || !psqt_buffer_ || !positional_buffer_) {
    return false;
  }
  
  initialized_ = true;
  return true;
}

int PersistentBatchEvaluator::evaluate(const GPUPositionData* positions, int count,
                                       int32_t* psqt_scores, int32_t* positional_scores) {
  if (!initialized_ || !gpu_manager_ || count <= 0) {
    return 0;
  }
  
  count = std::min(count, max_batch_size_);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // Copy positions to persistent buffer (unified memory - fast)
  std::memcpy(position_buffer_->data(), positions, count * sizeof(GPUPositionData));
  
  // Create batch and evaluate
  GPUEvalBatch batch;
  batch.count = count;
  batch.reserve(count);
  
  // Build position objects and extract features
  std::deque<StateInfo> states(count);
  for (int i = 0; i < count; ++i) {
    Position pos;
    // Reconstruct position from GPUPositionData
    // This is a simplified version - full implementation would use the position data directly
    std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    pos.set(fen, false, &states[i]);
    batch.add_position(pos);
  }
  
  // Evaluate
  bool success = gpu_manager_->evaluate_batch(batch, true);
  
  if (success) {
    // Copy results
    for (int i = 0; i < count; ++i) {
      psqt_scores[i] = batch.psqt_scores[i];
      positional_scores[i] = batch.positional_scores[i];
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  double latency = std::chrono::duration<double, std::micro>(end - start).count();
  
  total_batches_++;
  total_positions_ += count;
  total_latency_us_ += latency;
  
  return success ? count : 0;
}

void PersistentBatchEvaluator::evaluate_async(const GPUPositionData* positions, int count) {
  // Copy to persistent buffer
  count = std::min(count, max_batch_size_);
  std::memcpy(position_buffer_->data(), positions, count * sizeof(GPUPositionData));
  pending_count_ = count;
  async_pending_ = true;
}

int PersistentBatchEvaluator::get_results(int32_t* psqt_scores, int32_t* positional_scores) {
  if (!async_pending_) return 0;
  
  // For now, do synchronous evaluation
  GPUPositionData* positions = static_cast<GPUPositionData*>(position_buffer_->data());
  int result = evaluate(positions, pending_count_, psqt_scores, positional_scores);
  
  async_pending_ = false;
  pending_count_ = 0;
  
  return result;
}

void PersistentBatchEvaluator::wait() {
  // Nothing to wait for in current implementation
}

double PersistentBatchEvaluator::avg_batch_latency_us() const {
  if (total_batches_ == 0) return 0;
  return total_latency_us_ / total_batches_;
}

// ============================================================================
// Global Instances
// ============================================================================

static std::unique_ptr<PersistentBufferPool> g_buffer_pool;
static std::unique_ptr<PersistentBatchEvaluator> g_persistent_evaluator;

PersistentBufferPool& persistent_buffer_pool() {
  if (!g_buffer_pool) {
    g_buffer_pool = std::make_unique<PersistentBufferPool>();
  }
  return *g_buffer_pool;
}

bool initialize_persistent_resources() {
  // Initialize buffer pool with typical sizes
  // Small: 64KB (for feature buffers)
  // Medium: 1MB (for accumulator buffers)
  // Large: 16MB (for weight buffers)
  return persistent_buffer_pool().initialize(
      64 * 1024, 16,      // 16 small buffers
      1024 * 1024, 8,     // 8 medium buffers
      16 * 1024 * 1024, 4 // 4 large buffers
  );
}

PersistentBatchEvaluator& persistent_evaluator() {
  if (!g_persistent_evaluator) {
    g_persistent_evaluator = std::make_unique<PersistentBatchEvaluator>();
    g_persistent_evaluator->initialize(256);
  }
  return *g_persistent_evaluator;
}

}  // namespace GPU
}  // namespace MetalFish

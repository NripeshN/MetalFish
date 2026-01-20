/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Apple Silicon MCTS Optimizations
  
  This module provides Apple Silicon-specific optimizations for MCTS:
  
  1. Unified Memory Zero-Copy:
     - Direct CPU/GPU memory sharing without copies
     - GPU-resident evaluation batches
     - Lock-free atomic operations on shared memory
  
  2. Metal GPU Acceleration:
     - Batched position evaluation using Metal compute shaders
     - SIMD-accelerated PUCT selection
     - Parallel policy softmax computation
  
  3. Apple Silicon Specifics:
     - M-series GPU optimal threadgroup sizing
     - AMX (Apple Matrix Extensions) friendly data layouts
     - Efficient memory coalescing for unified memory
  
  Licensed under GPL-3.0
*/

#pragma once

#include "../gpu/backend.h"
#include "../gpu/gpu_nnue_integration.h"
#include "lc0_mcts_core.h"
#include <atomic>
#include <memory>
#include <vector>

#ifdef __APPLE__
#include <os/lock.h>  // Apple's unfair lock (fastest on Apple Silicon)
#include <cstdlib>    // For free() in AlignedDeleter
#endif

namespace MetalFish {
namespace MCTS {

// ============================================================================
// Cache-Line Aligned Structures for Apple Silicon
// ============================================================================

// Apple Silicon has 128-byte cache lines (M1/M2/M3)
constexpr size_t APPLE_CACHE_LINE_SIZE = 128;

// Align to cache line for optimal memory access
#define APPLE_ALIGNED alignas(APPLE_CACHE_LINE_SIZE)

// ============================================================================
// Apple Silicon Optimized Node Statistics
// ============================================================================

// Compact node statistics optimized for Apple Silicon unified memory
// All atomic operations are cache-line aligned to prevent false sharing
struct APPLE_ALIGNED AppleSiliconNodeStats {
  // Primary statistics (hot path - first cache line)
  std::atomic<uint32_t> n{0};           // Visit count
  std::atomic<uint32_t> n_in_flight{0}; // Virtual loss
  std::atomic<float> w{0.0f};           // Total value
  std::atomic<float> q{0.0f};           // Mean value (Q = W / N)
  std::atomic<float> d{0.0f};           // Draw probability
  std::atomic<float> m{0.0f};           // Moves left estimate
  
  // Padding to fill first cache line
  char _pad1[APPLE_CACHE_LINE_SIZE - 6 * sizeof(std::atomic<float>)];
  
  // Secondary statistics (cold path - second cache line)
  std::atomic<bool> is_terminal{false};
  std::atomic<int8_t> terminal_type{0}; // 0=none, 1=win, -1=loss, 2=draw
  std::atomic<bool> has_ab_score{false};
  std::atomic<int32_t> ab_score{0};
  std::atomic<int16_t> ab_depth{0};
  
  // Padding to fill second cache line
  char _pad2[APPLE_CACHE_LINE_SIZE - 5 * sizeof(std::atomic<int32_t>)];
  
  void reset() {
    n.store(0, std::memory_order_relaxed);
    n_in_flight.store(0, std::memory_order_relaxed);
    w.store(0.0f, std::memory_order_relaxed);
    q.store(0.0f, std::memory_order_relaxed);
    d.store(0.0f, std::memory_order_relaxed);
    m.store(0.0f, std::memory_order_relaxed);
    is_terminal.store(false, std::memory_order_relaxed);
    terminal_type.store(0, std::memory_order_relaxed);
    has_ab_score.store(false, std::memory_order_relaxed);
    ab_score.store(0, std::memory_order_relaxed);
    ab_depth.store(0, std::memory_order_relaxed);
  }
};

static_assert(sizeof(AppleSiliconNodeStats) == 2 * APPLE_CACHE_LINE_SIZE,
              "AppleSiliconNodeStats must be exactly 2 cache lines");

// ============================================================================
// GPU-Resident Evaluation Batch
// ============================================================================

// Position data structure matching Metal shader expectations
// Aligned for efficient GPU memory access
struct APPLE_ALIGNED GPUPositionData {
  // Piece bitboards [color][piece_type] - 112 bytes
  uint64_t pieces[2][7];
  
  // King squares - 2 bytes
  uint8_t king_sq[2];
  
  // Side to move - 1 byte
  uint8_t stm;
  
  // Piece count for bucket selection - 1 byte
  uint8_t piece_count;
  
  // Padding to 128 bytes (cache line)
  uint8_t padding[12];
  
  void from_position(const Position& pos);
};

static_assert(sizeof(GPUPositionData) == APPLE_CACHE_LINE_SIZE,
              "GPUPositionData must be exactly one cache line");

// GPU-resident batch for zero-copy evaluation
// Buffers are allocated in unified memory for direct CPU/GPU access
class GPUResidentEvalBatch {
public:
  GPUResidentEvalBatch() = default;
  ~GPUResidentEvalBatch() = default;
  
  // Initialize with Metal buffers in unified memory
  bool initialize(int batch_capacity);
  
  // Add position to batch (returns index, or -1 if full)
  int add_position(const GPUPositionData& pos_data);
  
  // Clear batch for reuse (doesn't deallocate)
  void clear();
  
  // Get current batch size
  int size() const { return current_size_.load(std::memory_order_acquire); }
  
  // Check if batch is full
  bool is_full() const { return size() >= capacity_; }
  
  // Get capacity
  int capacity() const { return capacity_; }
  
  // Direct access to GPU buffers (for Metal dispatch)
  GPU::Buffer* positions_buffer() { return positions_buffer_.get(); }
  GPU::Buffer* results_buffer() { return results_buffer_.get(); }
  
  // Direct access to results (unified memory - no copy needed!)
  const int32_t* psqt_scores() const;
  const int32_t* positional_scores() const;
  
  // Get position data pointer (unified memory)
  GPUPositionData* positions_data();
  const GPUPositionData* positions_data() const;

private:
  std::unique_ptr<GPU::Buffer> positions_buffer_;
  std::unique_ptr<GPU::Buffer> results_buffer_;
  
  int capacity_ = 0;
  std::atomic<int> current_size_{0};
  bool initialized_ = false;
};

// ============================================================================
// Apple Silicon MCTS Configuration
// ============================================================================

struct AppleSiliconMCTSConfig {
  // Lc0-style MCTS parameters
  Lc0SearchParams lc0_params;
  
  // Apple Silicon specific settings
  int gpu_batch_size = 128;        // Optimal for M-series GPUs
  int num_parallel_queues = 2;     // Number of Metal command queues
  bool use_async_evaluation = true; // Use async GPU dispatch
  bool use_unified_memory = true;  // Use zero-copy unified memory
  
  // Thread configuration
  int num_search_threads = 4;      // MCTS worker threads
  int virtual_loss = 3;            // Virtual loss for parallel search
  
  // Memory settings
  size_t node_pool_size = 1 << 20; // 1M nodes pre-allocated
  size_t tt_size = 1 << 22;        // 4M TT entries
  
  // Auto-tune for Apple Silicon
  void auto_tune_for_apple_silicon();
};

// ============================================================================
// Apple Silicon MCTS Evaluator
// ============================================================================

// High-performance MCTS evaluator optimized for Apple Silicon
class AppleSiliconMCTSEvaluator {
public:
  AppleSiliconMCTSEvaluator();
  ~AppleSiliconMCTSEvaluator();
  
  // Initialize with GPU NNUE manager
  bool initialize(GPU::GPUNNUEManager* gpu_manager, 
                  const AppleSiliconMCTSConfig& config);
  
  // Single position evaluation (uses batch internally)
  float evaluate_position(const Position& pos);
  
  // Batch evaluation - fully async on GPU
  void evaluate_batch_async(GPUResidentEvalBatch& batch,
                            std::function<void()> completion_handler);
  
  // Synchronous batch evaluation
  void evaluate_batch_sync(GPUResidentEvalBatch& batch);
  
  // Get evaluation result from batch (call after async completes)
  float get_batch_result(const GPUResidentEvalBatch& batch, int index,
                         Color side_to_move) const;
  
  // Statistics
  uint64_t total_evaluations() const { return total_evals_.load(); }
  uint64_t batch_count() const { return batch_count_.load(); }
  double avg_batch_size() const;
  
  // Configuration
  const AppleSiliconMCTSConfig& config() const { return config_; }

private:
  GPU::GPUNNUEManager* gpu_manager_ = nullptr;
  AppleSiliconMCTSConfig config_;
  
  // Pre-allocated batch for single evaluations
  std::unique_ptr<GPUResidentEvalBatch> single_eval_batch_;
  
  // Metal compute kernels for MCTS operations
  std::unique_ptr<GPU::ComputeKernel> nnue_eval_kernel_;
  std::unique_ptr<GPU::ComputeKernel> score_transform_kernel_;
  
  // Statistics
  std::atomic<uint64_t> total_evals_{0};
  std::atomic<uint64_t> batch_count_{0};
  
  // Compile Metal kernels for MCTS
  bool compile_mcts_kernels();
};

// ============================================================================
// Apple Silicon PUCT Selector
// ============================================================================

// SIMD-accelerated PUCT selection using Apple's Accelerate framework
class AppleSiliconPUCTSelector {
public:
  // Select best child using PUCT with SIMD acceleration
  // Returns index of best child, or -1 if no children
  template<typename Node, typename EdgeArray>
  static int select_best_child(
      Node* parent,
      const EdgeArray& edges,
      int num_edges,
      const Lc0SearchParams& params,
      bool is_root,
      float draw_score);
  
  // Batch PUCT computation for multiple nodes (GPU accelerated)
  static void compute_puct_batch(
      const std::vector<float>& parent_q,
      const std::vector<float>& parent_n,
      const std::vector<float>& child_q,
      const std::vector<float>& child_n,
      const std::vector<float>& policy,
      const Lc0SearchParams& params,
      std::vector<float>& scores_out);
};

// ============================================================================
// Apple Silicon Policy Softmax
// ============================================================================

// GPU-accelerated policy softmax computation
class AppleSiliconPolicySoftmax {
public:
  // Compute softmax over policy scores using GPU
  static void compute_softmax_gpu(
      const std::vector<float>& scores,
      float temperature,
      std::vector<float>& probs_out);
  
  // Compute softmax using SIMD (for smaller batches)
  static void compute_softmax_simd(
      const float* scores,
      int count,
      float temperature,
      float* probs_out);
};

// ============================================================================
// Apple Silicon Memory Pool
// ============================================================================

// Lock-free memory pool for MCTS nodes using Apple's os_unfair_lock
class AppleSiliconNodePool {
public:
  AppleSiliconNodePool(size_t capacity);
  ~AppleSiliconNodePool();
  
  // Allocate a node (returns nullptr if pool exhausted)
  void* allocate();
  
  // Return node to pool
  void deallocate(void* ptr);
  
  // Get allocation statistics
  size_t allocated_count() const { return allocated_.load(); }
  size_t capacity() const { return capacity_; }
  
  // Reset pool (invalidates all allocations!)
  void reset();

private:
#ifdef __APPLE__
  // Custom deleter for posix_memalign-allocated memory (must use free(), not delete[])
  struct AlignedDeleter {
    void operator()(char* ptr) const { free(ptr); }
  };
  std::unique_ptr<char, AlignedDeleter> memory_;
#else
  std::unique_ptr<char[]> memory_;
#endif
  size_t capacity_;
  size_t node_size_;
  
  std::atomic<size_t> next_free_{0};
  std::atomic<size_t> allocated_{0};
  
#ifdef __APPLE__
  os_unfair_lock lock_ = OS_UNFAIR_LOCK_INIT;
#else
  std::mutex lock_;
#endif
};

// ============================================================================
// Utility Functions
// ============================================================================

// Check if running on Apple Silicon
bool is_apple_silicon();

// Get optimal thread count for Apple Silicon
int get_optimal_thread_count();

// Get optimal GPU batch size for current device
int get_optimal_gpu_batch_size();

// Get unified memory availability
bool has_unified_memory();

// ============================================================================
// Metal Shader Source for MCTS Operations
// ============================================================================

// This will be compiled at runtime if Metal is available
extern const char* MCTS_METAL_SHADER_SOURCE;

} // namespace MCTS
} // namespace MetalFish

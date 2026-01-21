/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU NNUE Integration Header

  Provides GPU-accelerated NNUE evaluation with adaptive kernel selection
  based on batch size and runtime performance characteristics.
*/

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "backend.h"
#include "core/types.h"
#include "gpu_constants.h"

namespace MetalFish {
class Position;

namespace Eval::NNUE {
struct Networks;
template <typename Arch, typename Transformer> class Network;
} // namespace Eval::NNUE
} // namespace MetalFish

namespace MetalFish::GPU {

// ============================================================================
// Evaluation Strategy Selection
// ============================================================================

// Strategy for kernel dispatch based on batch characteristics
enum class EvalStrategy {
  CPU_FALLBACK,       // Use CPU evaluation (batch too small)
  GPU_STANDARD,       // Standard GPU kernels
  GPU_SIMD,           // SIMD-optimized kernels (large batches)
  GPU_FEATURE_EXTRACT // GPU-side feature extraction (very large batches)
};

// Runtime tuning parameters learned from observed performance
struct GPUTuningParams {
  int min_batch_for_gpu = 4; // Minimum batch size for GPU path
  int simd_threshold =
      512; // Batch size threshold for SIMD kernels (conservative)
  int gpu_extract_threshold = 2048; // Threshold for GPU feature extraction
  double cpu_eval_ns = 80.0;        // Observed CPU eval time (nanoseconds)
  double gpu_dispatch_us =
      140.0; // Observed GPU dispatch overhead (microseconds)

  // Select optimal strategy based on batch size
  EvalStrategy select_strategy(int batch_size) const;
};

// ============================================================================
// GPU Position Representation
// ============================================================================

struct alignas(16) GPUPositionData {
  uint64_t pieces[2][7]; // [color][piece_type] bitboards
  uint8_t king_sq[2];    // King squares
  uint8_t stm;           // Side to move
  uint8_t piece_count;   // Total pieces for bucket selection
  uint8_t padding[4];

  void from_position(const Position &pos);
};

// ============================================================================
// Feature Update for Incremental Computation
// ============================================================================

struct GPUFeatureUpdate {
  std::array<int32_t, 32> added_features;
  std::array<int32_t, 32> removed_features;
  uint8_t num_added = 0;
  uint8_t num_removed = 0;
  uint8_t perspective = 0;
  uint8_t padding = 0;
};

// ============================================================================
// GPU Network Weights
// ============================================================================

struct GPULayerWeights {
  std::unique_ptr<Buffer> fc0_weights; // [hidden_dim * (FC0_OUT+1)]
  std::unique_ptr<Buffer> fc0_biases;  // [FC0_OUT+1]
  std::unique_ptr<Buffer> fc1_weights; // [FC0_OUT*2 * FC1_OUT]
  std::unique_ptr<Buffer> fc1_biases;  // [FC1_OUT]
  std::unique_ptr<Buffer> fc2_weights; // [FC1_OUT]
  std::unique_ptr<Buffer> fc2_biases;  // [1]

  bool valid() const {
    return fc0_weights && fc0_biases && fc1_weights && fc1_biases &&
           fc2_weights && fc2_biases;
  }
};

struct GPUNetworkData {
  // Feature transformer
  std::unique_ptr<Buffer> ft_weights; // [HALFKA_DIMS * hidden_dim]
  std::unique_ptr<Buffer> ft_biases;  // [hidden_dim]
  std::unique_ptr<Buffer> ft_psqt;    // [HALFKA_DIMS * PSQT_BUCKETS]

  // Threat weights (big network only)
  std::unique_ptr<Buffer> threat_weights;
  std::unique_ptr<Buffer> threat_psqt;

  // Per-bucket FC layers
  std::array<GPULayerWeights, GPU_LAYER_STACKS> layers;

  int hidden_dim = 0;
  bool has_threats = false;
  bool valid = false;

  size_t memory_usage() const;
};

// ============================================================================
// GPU Evaluation Batch
// ============================================================================

struct GPUEvalBatch {
  // Position data
  std::vector<GPUPositionData> positions;

  // Extracted features
  std::vector<int32_t> white_features;
  std::vector<int32_t> black_features;
  std::vector<uint32_t> feature_counts;
  std::vector<uint32_t> feature_offsets;

  // Bucket indices
  std::vector<int32_t> buckets;

  // Results
  std::vector<int32_t> psqt_scores;
  std::vector<int32_t> positional_scores;

  int count = 0;

  void clear();
  void reserve(int n);
  void add_position(const Position &pos);
  void add_position_data(const GPUPositionData &data);
  int get_bucket(int idx) const { return buckets[idx]; }
};

// ============================================================================
// GPU NNUE Manager
// ============================================================================

class GPUNNUEManager {
public:
  GPUNNUEManager();
  ~GPUNNUEManager();

  // Initialization
  bool initialize();
  bool load_networks(const Eval::NNUE::Networks &networks);
  bool is_ready() const {
    return initialized_ && (big_network_.valid || small_network_.valid);
  }

  // Batch evaluation with automatic strategy selection
  // If force_gpu is true, bypasses the min_batch_for_gpu threshold check
  bool evaluate_batch(GPUEvalBatch &batch, bool use_big_network = true,
                      bool force_gpu = false);

  // Asynchronous batch evaluation (returns immediately, calls
  // completion_handler when done) The batch must remain valid until
  // completion_handler is called
  bool
  evaluate_batch_async(GPUEvalBatch &batch,
                       std::function<void(bool success)> completion_handler,
                       bool use_big_network = true);

  // Single position (falls back to CPU if batch size is 1)
  std::pair<int32_t, int32_t> evaluate_single(const Position &pos,
                                              bool use_big = true);

  // Configuration
  int min_batch_size() const { return tuning_.min_batch_for_gpu; }
  void set_min_batch_size(int size) { tuning_.min_batch_for_gpu = size; }

  // Access tuning parameters for runtime adjustment
  GPUTuningParams &tuning() { return tuning_; }
  const GPUTuningParams &tuning() const { return tuning_; }

  // Statistics
  size_t gpu_evaluations() const { return gpu_evals_; }
  size_t cpu_fallback_evaluations() const { return cpu_evals_; }
  size_t total_batches() const { return batch_count_; }
  double avg_batch_time_ms() const;
  double total_gpu_time_ms() const { return total_time_ms_; }
  void reset_stats();

  // Memory
  size_t gpu_memory_used() const;

  // Status
  std::string status_string() const;

private:
  bool initialized_ = false;
  GPUTuningParams tuning_;

  // Network weights
  GPUNetworkData big_network_;
  GPUNetworkData small_network_;

  // Compute kernels - standard
  std::unique_ptr<ComputeKernel> extract_features_kernel_;
  std::unique_ptr<ComputeKernel> feature_transform_kernel_;
  std::unique_ptr<ComputeKernel> feature_transform_dual_kernel_;
  std::unique_ptr<ComputeKernel> psqt_kernel_;
  std::unique_ptr<ComputeKernel> forward_fused_kernel_;

  // Compute kernels - optimized variants
  std::unique_ptr<ComputeKernel> forward_simd_kernel_;
  std::unique_ptr<ComputeKernel> forward_batch_kernel_;
  std::unique_ptr<ComputeKernel> feature_transform_vec4_kernel_;
  std::unique_ptr<ComputeKernel> feature_transform_dual_vec4_kernel_;
  std::unique_ptr<ComputeKernel> forward_optimized_kernel_;
  std::unique_ptr<ComputeKernel> fused_single_kernel_;

  // Working buffers
  std::unique_ptr<Buffer> positions_buffer_;
  std::unique_ptr<Buffer> white_features_buffer_;
  std::unique_ptr<Buffer> black_features_buffer_;
  std::unique_ptr<Buffer> feature_counts_buffer_;
  std::unique_ptr<Buffer> feature_offsets_buffer_;
  std::unique_ptr<Buffer> accumulators_buffer_;
  std::unique_ptr<Buffer> psqt_buffer_;
  std::unique_ptr<Buffer> output_buffer_;

  // Pre-allocated staging buffers (avoid per-call allocations)
  std::unique_ptr<Buffer> white_counts_buffer_;
  std::unique_ptr<Buffer> black_counts_buffer_;
  std::unique_ptr<Buffer> white_offsets_buffer_;
  std::unique_ptr<Buffer> black_offsets_buffer_;

  // Statistics
  std::atomic<size_t> gpu_evals_{0};
  std::atomic<size_t> cpu_evals_{0};
  std::atomic<size_t> batch_count_{0};
  double total_time_ms_ = 0;

  // Thread safety for GPU operations
  mutable std::mutex gpu_mutex_;

  // Internal methods
  bool compile_shaders();
  bool allocate_working_buffers();
  bool allocate_network_buffers(GPUNetworkData &net, int hidden_dim,
                                bool has_threats);

  template <typename Network>
  bool extract_weights(const Network &network, GPUNetworkData &gpu_net,
                       int hidden_dim, bool has_threats);

  bool dispatch_feature_transform(const GPUNetworkData &net, int batch_size);
  bool dispatch_forward_pass(const GPUNetworkData &net, int batch_size,
                             int bucket);

  // Strategy-specific evaluation paths
  bool evaluate_standard(GPUEvalBatch &batch, const GPUNetworkData &net);
  bool evaluate_simd(GPUEvalBatch &batch, const GPUNetworkData &net);
};

// ============================================================================
// Global Interface
// ============================================================================

// Get the global GPU NNUE manager
GPUNNUEManager &gpu_nnue_manager();

// Initialize GPU NNUE with networks
bool initialize_gpu_nnue(const Eval::NNUE::Networks &networks);

// Check if GPU NNUE manager is available
bool gpu_nnue_manager_available();

// Evaluate a batch of positions
bool gpu_evaluate_batch(GPUEvalBatch &batch, bool use_big = true);

// Shutdown GPU NNUE manager - call before program exit to ensure clean cleanup
// This prevents crashes during static destruction
void shutdown_gpu_nnue();

} // namespace MetalFish::GPU

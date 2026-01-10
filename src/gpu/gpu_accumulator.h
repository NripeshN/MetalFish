/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Accumulator Cache

  This module provides GPU-accelerated accumulator management for NNUE
  evaluation. It mirrors the CPU AccumulatorStack but uses GPU memory and
  compute for:
  - Full accumulator computation from scratch
  - Incremental updates for move-by-move evaluation
  - Batch accumulator computation for parallel search

  Key optimizations:
  - Unified memory for zero-copy access
  - Incremental updates minimize computation
  - Batch processing for parallel search threads
*/

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "backend.h"
#include "core/types.h"
#include "gpu_nnue_integration.h"

namespace MetalFish {
class Position;
}

namespace MetalFish::GPU {

// ============================================================================
// GPU Accumulator Entry
// ============================================================================

// Mirrors CPU Accumulator structure for GPU
struct GPUAccumulatorEntry {
  // Accumulation values for both perspectives [color][hidden_dim]
  std::unique_ptr<Buffer> accumulation;
  // PSQT accumulation [color][PSQT_BUCKETS]
  std::unique_ptr<Buffer> psqt_accumulation;
  // Validity flags
  bool computed[2] = {false, false};

  int hidden_dim = 0;
  bool valid = false;

  bool allocate(int hidden_dim);
  void reset();
};

// ============================================================================
// GPU Accumulator Stack
// ============================================================================

// Mirrors CPU AccumulatorStack for GPU-accelerated evaluation
class GPUAccumulatorStack {
public:
  static constexpr int MAX_PLY = 256;

  GPUAccumulatorStack();
  ~GPUAccumulatorStack();

  // Initialize with network hidden dimension
  bool initialize(int hidden_dim, bool has_threats = false);
  bool is_initialized() const { return initialized_; }

  // Stack operations
  void reset();
  void push();
  void pop();
  int size() const { return size_; }

  // Get current accumulator
  GPUAccumulatorEntry &current();
  const GPUAccumulatorEntry &current() const;

  // Get accumulator at specific ply
  GPUAccumulatorEntry &at(int ply);
  const GPUAccumulatorEntry &at(int ply) const;

  // Compute accumulator from scratch
  bool compute_full(const Position &pos, const GPUNetworkData &network);

  // Incremental update after a move
  bool compute_incremental(const Position &pos, const GPUNetworkData &network,
                           const GPUFeatureUpdate &update);

  // Copy accumulator from another entry
  bool copy_from(int src_ply, int dst_ply);

  // Statistics
  size_t full_computations() const { return full_computes_; }
  size_t incremental_updates() const { return incremental_updates_; }
  double total_compute_time_ms() const { return total_time_ms_; }
  void reset_stats();

private:
  bool initialized_ = false;
  int hidden_dim_ = 0;
  bool has_threats_ = false;
  int size_ = 1;

  std::array<GPUAccumulatorEntry, MAX_PLY> entries_;

  // Compute kernels
  std::unique_ptr<ComputeKernel> full_transform_kernel_;
  std::unique_ptr<ComputeKernel> incremental_kernel_;
  std::unique_ptr<ComputeKernel> copy_kernel_;

  // Working buffers
  std::unique_ptr<Buffer> features_buffer_;
  std::unique_ptr<Buffer> feature_counts_buffer_;
  std::unique_ptr<Buffer> update_buffer_;

  // Statistics
  std::atomic<size_t> full_computes_{0};
  std::atomic<size_t> incremental_updates_{0};
  double total_time_ms_ = 0;

  bool compile_kernels();
  bool allocate_buffers();
};

// ============================================================================
// GPU Accumulator Cache (Finny Tables)
// ============================================================================

// GPU version of AccumulatorCaches for efficient refresh
class GPUAccumulatorCache {
public:
  GPUAccumulatorCache();
  ~GPUAccumulatorCache();

  // Initialize cache for a network
  bool initialize(int hidden_dim, const GPUNetworkData &network);
  bool is_initialized() const { return initialized_; }

  // Get cached entry for a king square
  GPUAccumulatorEntry &get(Square king_sq, Color perspective);
  const GPUAccumulatorEntry &get(Square king_sq, Color perspective) const;

  // Clear all entries (reset to biases)
  void clear(const GPUNetworkData &network);

  // Update cache entry
  bool update(Square king_sq, Color perspective,
              const std::vector<int32_t> &added_features,
              const std::vector<int32_t> &removed_features,
              const GPUNetworkData &network);

private:
  bool initialized_ = false;
  int hidden_dim_ = 0;

  // Cache entries: [king_square][color]
  std::array<std::array<GPUAccumulatorEntry, 2>, 64> entries_;

  // Piece tracking for each entry
  struct PieceState {
    std::array<Piece, 64> pieces;
    Bitboard piece_bb;
  };
  std::array<std::array<PieceState, 2>, 64> piece_states_;

  std::unique_ptr<ComputeKernel> init_kernel_;
  std::unique_ptr<ComputeKernel> update_kernel_;
};

// ============================================================================
// GPU Feature Extractor
// ============================================================================

// Extracts HalfKAv2_hm features for GPU evaluation
class GPUFeatureExtractor {
public:
  GPUFeatureExtractor();
  ~GPUFeatureExtractor();

  bool initialize();
  bool is_initialized() const { return initialized_; }

  // Extract features from position
  bool extract(const Position &pos, std::vector<int32_t> &white_features,
               std::vector<int32_t> &black_features);

  // Extract features for batch of positions
  bool extract_batch(const std::vector<const Position *> &positions,
                     std::vector<int32_t> &features,
                     std::vector<uint32_t> &feature_counts,
                     std::vector<uint32_t> &feature_offsets);

  // Compute feature delta for a move
  bool compute_delta(const Position &pos, Move move,
                     GPUFeatureUpdate &white_update,
                     GPUFeatureUpdate &black_update);

private:
  bool initialized_ = false;

  // Precomputed lookup tables (in GPU memory)
  std::unique_ptr<Buffer> orient_table_;
  std::unique_ptr<Buffer> piece_to_index_table_;

  std::unique_ptr<ComputeKernel> extract_kernel_;
  std::unique_ptr<ComputeKernel> batch_extract_kernel_;
  std::unique_ptr<ComputeKernel> delta_kernel_;

  bool compile_kernels();
  bool create_lookup_tables();
};

// ============================================================================
// Global Interface
// ============================================================================

// Get global GPU feature extractor
GPUFeatureExtractor &gpu_feature_extractor();

// Check if GPU feature extraction is available
bool gpu_features_available();

} // namespace MetalFish::GPU

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file gpu_accumulator_cache.h
 * @brief MetalFish source file.
 */

  GPU Accumulator Cache - Finny Tables Implementation

  This implements Stockfish's Finny Tables optimization for GPU:
  - Caches accumulator states indexed by king square
  - Enables efficient incremental updates during search
  - Tracks which accumulators need refresh vs incremental update
*/

#pragma once

#include "gpu_constants.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace MetalFish::GPU {

// Forward declarations
class Buffer;

// ============================================================================
// Accumulator State Tracking
// ============================================================================

// Tracks the state of a cached accumulator
struct AccumulatorState {
  // The king square this accumulator was computed for
  int king_square = -1;

  // Epoch counter to detect stale entries
  uint32_t epoch = 0;

  // Whether this accumulator is valid
  bool valid = false;

  // Number of pieces when computed (for bucket selection)
  int piece_count = 0;
};

// ============================================================================
// Feature Delta for Incremental Updates
// ============================================================================

// Represents the changes between two positions
struct FeatureDelta {
  // Features to add (max 3: piece moved to, captured piece removed handled
  // separately)
  std::array<int32_t, 4> added;
  int num_added = 0;

  // Features to remove (max 3: piece moved from, captured piece)
  std::array<int32_t, 4> removed;
  int num_removed = 0;

  // Which perspective this delta applies to (0=white, 1=black)
  int perspective = 0;

  // Whether this is a king move (requires full refresh)
  bool is_king_move = false;

  void clear() {
    num_added = 0;
    num_removed = 0;
    is_king_move = false;
  }

  void add_feature(int32_t feature) {
    if (num_added < 4)
      added[num_added++] = feature;
  }

  void remove_feature(int32_t feature) {
    if (num_removed < 4)
      removed[num_removed++] = feature;
  }
};

// ============================================================================
// Finny Table Entry
// ============================================================================

// Single entry in the Finny Table - caches accumulator for one king position
struct FinnyEntry {
  // Cached accumulator values [hidden_dim]
  std::vector<int32_t> accumulator;

  // PSQT values [PSQT_BUCKETS]
  std::array<int32_t, GPU_PSQT_BUCKETS> psqt;

  // State tracking
  AccumulatorState state;

  // Piece bitboard hash when this was computed (for validation)
  uint64_t piece_hash = 0;

  FinnyEntry() : psqt{} {}

  void resize(int hidden_dim) { accumulator.resize(hidden_dim, 0); }

  bool matches(int ksq, uint64_t hash) const {
    return state.valid && state.king_square == ksq && piece_hash == hash;
  }
};

// ============================================================================
// Finny Table - Per-Perspective Accumulator Cache
// ============================================================================

class FinnyTable {
public:
  static constexpr int NUM_SQUARES = 64;

  FinnyTable() = default;

  void initialize(int hidden_dim) {
    hidden_dim_ = hidden_dim;
    for (auto &entry : entries_) {
      entry.resize(hidden_dim);
    }
    epoch_ = 0;
  }

  // Get entry for a king square
  FinnyEntry &get(int king_square) { return entries_[king_square]; }

  const FinnyEntry &get(int king_square) const { return entries_[king_square]; }

  // Check if we have a valid cached accumulator for this king position
  bool has_valid(int king_square, uint64_t piece_hash) const {
    return entries_[king_square].matches(king_square, piece_hash);
  }

  // Invalidate all entries (e.g., after UCI "position" command)
  void invalidate_all() {
    epoch_++;
    for (auto &entry : entries_) {
      entry.state.valid = false;
    }
  }

  // Invalidate entry for a specific king square
  void invalidate(int king_square) {
    entries_[king_square].state.valid = false;
  }

  // Get current epoch
  uint32_t epoch() const { return epoch_; }

private:
  std::array<FinnyEntry, NUM_SQUARES> entries_;
  int hidden_dim_ = 0;
  uint32_t epoch_ = 0;
};

// ============================================================================
// GPU Accumulator Stack - Search-Aware Accumulator Management
// ============================================================================

// Manages accumulator state during search with GPU acceleration
class GPUAccumulatorStack {
public:
  static constexpr int MAX_PLY = 256;

  GPUAccumulatorStack();
  ~GPUAccumulatorStack();

  // Initialize with network dimensions
  bool initialize(int hidden_dim, bool use_big_network = true);

  // Push a new ply (after making a move)
  void push();

  // Pop a ply (after unmaking a move)
  void pop();

  // Get current ply depth
  int ply() const { return ply_; }

  // Mark current accumulator as needing computation
  void mark_dirty(int perspective);

  // Mark current accumulator as computed
  void mark_computed(int perspective, int king_square, uint64_t piece_hash);

  // Check if current accumulator needs refresh
  bool needs_refresh(int perspective) const;

  // Get the feature delta from parent to current position
  FeatureDelta &get_delta(int perspective) {
    return deltas_[ply_][perspective];
  }

  // Clear deltas for current ply
  void clear_deltas();

  // Access Finny tables
  FinnyTable &finny_table(int perspective) {
    return finny_tables_[perspective];
  }

  // Get accumulator data for current ply
  int32_t *accumulator_data(int perspective);
  const int32_t *accumulator_data(int perspective) const;

  // Get PSQT data for current ply
  int32_t *psqt_data(int perspective);

  // Copy accumulator from parent ply
  void copy_from_parent(int perspective);

  // Reset to initial state
  void reset();

  // Get hidden dimension
  int hidden_dim() const { return hidden_dim_; }

private:
  int hidden_dim_ = 0;
  int ply_ = 0;

  // Per-ply accumulator states [MAX_PLY][2 perspectives]
  std::array<std::array<AccumulatorState, 2>, MAX_PLY> states_;

  // Per-ply feature deltas [MAX_PLY][2 perspectives]
  std::array<std::array<FeatureDelta, 2>, MAX_PLY> deltas_;

  // Finny tables for each perspective
  std::array<FinnyTable, 2> finny_tables_;

  // Accumulator storage [MAX_PLY][2 perspectives][hidden_dim]
  std::vector<int32_t> accumulator_storage_;

  // PSQT storage [MAX_PLY][2 perspectives][PSQT_BUCKETS]
  std::vector<int32_t> psqt_storage_;

  // Helper to get storage offset
  size_t acc_offset(int ply, int perspective) const {
    return (ply * 2 + perspective) * hidden_dim_;
  }

  size_t psqt_offset(int ply, int perspective) const {
    return (ply * 2 + perspective) * GPU_PSQT_BUCKETS;
  }
};

// ============================================================================
// Weight Permutation Helper
// ============================================================================

// Permutes weights for optimal SIMD access pattern
// Stockfish permutes weights so that consecutive SIMD lanes access
// consecutive memory locations
class WeightPermuter {
public:
  // Permute feature transformer weights for optimal access
  // Input:  weights[feature_idx * hidden_dim + hidden_idx]
  // Output: weights[hidden_tile * features * tile_size + feature_idx *
  // tile_size
  // + lane]
  static void permute_ft_weights(const int16_t *src, int16_t *dst,
                                 int num_features, int hidden_dim,
                                 int tile_size = 32);

  // Permute FC layer weights for optimal access
  static void permute_fc_weights(const int8_t *src, int8_t *dst, int input_dim,
                                 int output_dim, int tile_size = 32);

  // Inverse permutation for verification
  static void unpermute_ft_weights(const int16_t *src, int16_t *dst,
                                   int num_features, int hidden_dim,
                                   int tile_size = 32);
};

// ============================================================================
// Sparse Input Bitmask Helper
// ============================================================================

// Generates bitmasks for non-zero activations (Stockfish's find_nnz)
class SparseInputHelper {
public:
  // Find non-zero indices in clipped accumulator
  // Returns bitmask where bit i is set if accumulator[i] != 0 after clipping
  static uint64_t find_nonzero_mask(const int32_t *accumulator, int hidden_dim,
                                    int weight_scale_bits = 6);

  // Count non-zero elements
  static int count_nonzero(const int32_t *accumulator, int hidden_dim,
                           int weight_scale_bits = 6);

  // Extract indices of non-zero elements
  static int extract_nonzero_indices(const int32_t *accumulator, int hidden_dim,
                                     uint16_t *indices,
                                     int weight_scale_bits = 6);
};

} // namespace MetalFish::GPU
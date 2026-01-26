/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file gpu_accumulator_cache.cpp
 * @brief MetalFish source file.
 */

  GPU Accumulator Cache Implementation
*/

#include "gpu_accumulator_cache.h"
#include <algorithm>
#include <cstring>

namespace MetalFish::GPU {

// ============================================================================
// GPUAccumulatorStack Implementation
// ============================================================================

GPUAccumulatorStack::GPUAccumulatorStack() { reset(); }

GPUAccumulatorStack::~GPUAccumulatorStack() = default;

bool GPUAccumulatorStack::initialize(int hidden_dim, bool use_big_network) {
  hidden_dim_ = hidden_dim;

  // Allocate accumulator storage
  // [MAX_PLY][2 perspectives][hidden_dim]
  accumulator_storage_.resize(MAX_PLY * 2 * hidden_dim, 0);

  // Allocate PSQT storage
  // [MAX_PLY][2 perspectives][PSQT_BUCKETS]
  psqt_storage_.resize(MAX_PLY * 2 * GPU_PSQT_BUCKETS, 0);

  // Initialize Finny tables
  finny_tables_[0].initialize(hidden_dim);
  finny_tables_[1].initialize(hidden_dim);

  reset();
  return true;
}

void GPUAccumulatorStack::push() {
  if (ply_ < MAX_PLY - 1) {
    ply_++;
    // Mark new ply as needing computation
    states_[ply_][0].valid = false;
    states_[ply_][1].valid = false;
    deltas_[ply_][0].clear();
    deltas_[ply_][1].clear();
  }
}

void GPUAccumulatorStack::pop() {
  if (ply_ > 0) {
    ply_--;
  }
}

void GPUAccumulatorStack::mark_dirty(int perspective) {
  states_[ply_][perspective].valid = false;
}

void GPUAccumulatorStack::mark_computed(int perspective, int king_square,
                                        uint64_t piece_hash) {
  auto &state = states_[ply_][perspective];
  state.valid = true;
  state.king_square = king_square;
  state.epoch = finny_tables_[perspective].epoch();

  // Also update Finny table
  auto &entry = finny_tables_[perspective].get(king_square);
  entry.state = state;
  entry.piece_hash = piece_hash;

  // Copy accumulator to Finny table
  std::memcpy(entry.accumulator.data(), accumulator_data(perspective),
              hidden_dim_ * sizeof(int32_t));
  std::memcpy(entry.psqt.data(), psqt_data(perspective),
              GPU_PSQT_BUCKETS * sizeof(int32_t));
}

bool GPUAccumulatorStack::needs_refresh(int perspective) const {
  return !states_[ply_][perspective].valid;
}

void GPUAccumulatorStack::clear_deltas() {
  deltas_[ply_][0].clear();
  deltas_[ply_][1].clear();
}

int32_t *GPUAccumulatorStack::accumulator_data(int perspective) {
  return accumulator_storage_.data() + acc_offset(ply_, perspective);
}

const int32_t *GPUAccumulatorStack::accumulator_data(int perspective) const {
  return accumulator_storage_.data() + acc_offset(ply_, perspective);
}

int32_t *GPUAccumulatorStack::psqt_data(int perspective) {
  return psqt_storage_.data() + psqt_offset(ply_, perspective);
}

void GPUAccumulatorStack::copy_from_parent(int perspective) {
  if (ply_ > 0) {
    std::memcpy(accumulator_data(perspective),
                accumulator_storage_.data() + acc_offset(ply_ - 1, perspective),
                hidden_dim_ * sizeof(int32_t));
    std::memcpy(psqt_data(perspective),
                psqt_storage_.data() + psqt_offset(ply_ - 1, perspective),
                GPU_PSQT_BUCKETS * sizeof(int32_t));
  }
}

void GPUAccumulatorStack::reset() {
  ply_ = 0;
  for (auto &ply_states : states_) {
    for (auto &state : ply_states) {
      state.valid = false;
      state.king_square = -1;
      state.epoch = 0;
    }
  }
  for (auto &ply_deltas : deltas_) {
    for (auto &delta : ply_deltas) {
      delta.clear();
    }
  }
  finny_tables_[0].invalidate_all();
  finny_tables_[1].invalidate_all();
}

// ============================================================================
// WeightPermuter Implementation
// ============================================================================

void WeightPermuter::permute_ft_weights(const int16_t *src, int16_t *dst,
                                        int num_features, int hidden_dim,
                                        int tile_size) {
  // Permute for optimal SIMD access
  // Original layout: weights[feature * hidden_dim + hidden]
  // Permuted layout: weights[tile * num_features * tile_size + feature *
  // tile_size + lane]

  const int num_tiles = (hidden_dim + tile_size - 1) / tile_size;

  for (int tile = 0; tile < num_tiles; tile++) {
    for (int feature = 0; feature < num_features; feature++) {
      for (int lane = 0; lane < tile_size; lane++) {
        int hidden = tile * tile_size + lane;
        if (hidden < hidden_dim) {
          int src_idx = feature * hidden_dim + hidden;
          int dst_idx =
              tile * num_features * tile_size + feature * tile_size + lane;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

void WeightPermuter::permute_fc_weights(const int8_t *src, int8_t *dst,
                                        int input_dim, int output_dim,
                                        int tile_size) {
  // Permute FC weights for optimal access
  // Original: weights[input * output_dim + output]
  // Permuted: weights[tile * input_dim * tile_size + input * tile_size + lane]

  const int num_tiles = (output_dim + tile_size - 1) / tile_size;

  for (int tile = 0; tile < num_tiles; tile++) {
    for (int input = 0; input < input_dim; input++) {
      for (int lane = 0; lane < tile_size; lane++) {
        int output = tile * tile_size + lane;
        if (output < output_dim) {
          int src_idx = input * output_dim + output;
          int dst_idx = tile * input_dim * tile_size + input * tile_size + lane;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

void WeightPermuter::unpermute_ft_weights(const int16_t *src, int16_t *dst,
                                          int num_features, int hidden_dim,
                                          int tile_size) {
  // Inverse of permute_ft_weights
  const int num_tiles = (hidden_dim + tile_size - 1) / tile_size;

  for (int tile = 0; tile < num_tiles; tile++) {
    for (int feature = 0; feature < num_features; feature++) {
      for (int lane = 0; lane < tile_size; lane++) {
        int hidden = tile * tile_size + lane;
        if (hidden < hidden_dim) {
          int src_idx =
              tile * num_features * tile_size + feature * tile_size + lane;
          int dst_idx = feature * hidden_dim + hidden;
          dst[dst_idx] = src[src_idx];
        }
      }
    }
  }
}

// ============================================================================
// SparseInputHelper Implementation
// ============================================================================

uint64_t SparseInputHelper::find_nonzero_mask(const int32_t *accumulator,
                                              int hidden_dim,
                                              int weight_scale_bits) {
  // For hidden_dim > 64, we return a partial mask
  // This is used for the first 64 elements which covers most sparse patterns
  uint64_t mask = 0;
  const int max_bits = std::min(hidden_dim, 64);

  for (int i = 0; i < max_bits; i++) {
    int16_t clipped = static_cast<int16_t>(
        std::clamp(accumulator[i] >> weight_scale_bits, 0, 127));
    if (clipped != 0) {
      mask |= (1ULL << i);
    }
  }

  return mask;
}

int SparseInputHelper::count_nonzero(const int32_t *accumulator, int hidden_dim,
                                     int weight_scale_bits) {
  int count = 0;
  for (int i = 0; i < hidden_dim; i++) {
    int16_t clipped = static_cast<int16_t>(
        std::clamp(accumulator[i] >> weight_scale_bits, 0, 127));
    if (clipped != 0) {
      count++;
    }
  }
  return count;
}

int SparseInputHelper::extract_nonzero_indices(const int32_t *accumulator,
                                               int hidden_dim,
                                               uint16_t *indices,
                                               int weight_scale_bits) {
  int count = 0;
  for (int i = 0; i < hidden_dim; i++) {
    int16_t clipped = static_cast<int16_t>(
        std::clamp(accumulator[i] >> weight_scale_bits, 0, 127));
    if (clipped != 0) {
      indices[count++] = static_cast<uint16_t>(i);
    }
  }
  return count;
}

} // namespace MetalFish::GPU
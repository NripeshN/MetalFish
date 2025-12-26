/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include "core/types.h"
#include "core/position.h"
#include <Metal/Metal.hpp>
#include <vector>
#include <memory>
#include <atomic>

namespace MetalFish {

// GPU Search configuration
struct GPUSearchConfig {
    int batch_size = 64;           // Number of positions to evaluate at once
    int max_depth = 6;             // Max depth for GPU-parallel search
    bool use_unified_memory = true;
    bool async_evaluation = true;  // Pipeline CPU and GPU work
};

// Position snapshot for GPU evaluation
struct PositionData {
    std::array<int8_t, 64> pieces;   // Piece on each square
    uint64_t occupancy[3];           // WHITE, BLACK, ALL
    uint64_t piece_bb[2][7];         // Bitboards by color and piece type
    Color side_to_move;
    CastlingRights castling;
    Square ep_square;
    int ply;
    
    void from_position(const Position& pos);
};

// Batch of positions for GPU processing
struct PositionBatch {
    std::vector<PositionData> positions;
    std::vector<Move> moves;              // Associated moves that led to positions
    std::vector<int> parent_indices;      // Index of parent position
    std::vector<Value> scores;            // Evaluation scores (output)
    
    void reserve(size_t n) {
        positions.reserve(n);
        moves.reserve(n);
        parent_indices.reserve(n);
        scores.reserve(n);
    }
    
    void clear() {
        positions.clear();
        moves.clear();
        parent_indices.clear();
        scores.clear();
    }
    
    size_t size() const { return positions.size(); }
};

// GPU Search accelerator
class GPUSearch {
public:
    GPUSearch();
    ~GPUSearch();
    
    // Initialize GPU resources
    bool init();
    
    // Evaluate a batch of positions on GPU
    void evaluate_batch(PositionBatch& batch);
    
    // Score and sort moves for a batch of positions
    void score_moves_batch(const std::vector<Position*>& positions,
                           std::vector<std::vector<Move>>& moves,
                           std::vector<std::vector<int>>& scores);
    
    // Run parallel tree exploration on GPU (experimental)
    Value parallel_search(Position& root, int depth, Value alpha, Value beta);
    
    // Configuration
    void set_config(const GPUSearchConfig& config) { config_ = config; }
    GPUSearchConfig& config() { return config_; }
    
    // Statistics
    uint64_t positions_evaluated() const { return positions_evaluated_; }
    uint64_t gpu_batches() const { return gpu_batches_; }
    void reset_stats() { positions_evaluated_ = 0; gpu_batches_ = 0; }
    
    bool is_available() const { return available_; }
    
private:
    // Metal resources
    MTL::Device* device_ = nullptr;
    MTL::CommandQueue* queue_ = nullptr;
    MTL::Library* library_ = nullptr;
    
    // Compute pipelines
    MTL::ComputePipelineState* score_moves_kernel_ = nullptr;
    MTL::ComputePipelineState* material_eval_kernel_ = nullptr;
    MTL::ComputePipelineState* see_kernel_ = nullptr;
    
    // Buffers
    MTL::Buffer* position_buffer_ = nullptr;
    MTL::Buffer* moves_buffer_ = nullptr;
    MTL::Buffer* scores_buffer_ = nullptr;
    MTL::Buffer* params_buffer_ = nullptr;
    
    // Configuration and state
    GPUSearchConfig config_;
    bool available_ = false;
    
    // Statistics
    std::atomic<uint64_t> positions_evaluated_{0};
    std::atomic<uint64_t> gpu_batches_{0};
    
    // Private methods
    bool load_kernels();
    bool create_buffers();
    void dispatch_evaluation(const PositionBatch& batch);
};

// Global GPU search instance
GPUSearch& get_gpu_search();

} // namespace MetalFish



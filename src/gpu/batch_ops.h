/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU-accelerated Batch Operations
  
  This module provides GPU acceleration for batch operations:
  - Batch SEE (Static Exchange Evaluation)
  - Batch move scoring
  - Batch position evaluation
*/

#pragma once

#include "backend.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace MetalFish {

// Forward declarations
class Position;
class Move;

namespace GPU {

// Maximum batch sizes
constexpr int MAX_SEE_BATCH = 512;
constexpr int MAX_SCORE_BATCH = 256;

/**
 * GPU Position representation for batch operations
 * Compact format for efficient GPU transfer
 */
struct alignas(64) GPUPosition {
    uint64_t pieces[2][7];  // [color][piece_type] bitboards
    uint64_t occupied[3];   // [white, black, all]
    int8_t board[64];       // Piece on each square
    int32_t side_to_move;
    int32_t castling_rights;
    int32_t ep_square;
    int32_t rule50;
    
    void from_position(const Position& pos);
};

/**
 * GPU Move representation
 */
struct GPUMove {
    uint16_t data;   // from:6, to:6, type:2, promo:2
    int16_t score;   // Move ordering score
    
    static GPUMove from_move(Move m);
    Move to_move() const;
};

/**
 * Batch SEE (Static Exchange Evaluation) on GPU
 * 
 * SEE is used extensively in move ordering to determine if a capture
 * is likely to gain or lose material. Computing SEE for many moves
 * in parallel on the GPU can significantly speed up move ordering.
 */
class BatchSEE {
public:
    BatchSEE();
    ~BatchSEE();
    
    // Initialize GPU resources
    bool initialize();
    
    // Check if GPU SEE is available
    bool available() const { return initialized_; }
    
    /**
     * Compute SEE for a batch of moves
     * 
     * @param positions Array of positions
     * @param moves Array of moves (one per position)
     * @param thresholds Minimum value threshold for each move
     * @param results Output: true if SEE >= threshold
     * @param count Number of positions
     * @return true if GPU was used, false if fell back to CPU
     */
    bool compute(const Position* positions[], 
                 const Move moves[],
                 const int thresholds[],
                 bool results[],
                 int count);
    
    // Statistics
    size_t gpu_computations() const { return gpu_computations_; }
    size_t cpu_computations() const { return cpu_computations_; }
    double avg_batch_time_ms() const {
        return batch_count_ > 0 ? total_time_ms_ / batch_count_ : 0;
    }
    
private:
    bool initialized_ = false;
    
    // GPU buffers
    std::unique_ptr<Buffer> positions_buffer_;
    std::unique_ptr<Buffer> moves_buffer_;
    std::unique_ptr<Buffer> thresholds_buffer_;
    std::unique_ptr<Buffer> results_buffer_;
    std::unique_ptr<Buffer> attack_tables_buffer_;
    
    // Compute kernel
    std::unique_ptr<ComputeKernel> see_kernel_;
    
    // Statistics
    size_t gpu_computations_ = 0;
    size_t cpu_computations_ = 0;
    size_t batch_count_ = 0;
    double total_time_ms_ = 0;
    
    bool load_kernels();
    bool allocate_buffers();
    void upload_attack_tables();
};

/**
 * Batch Move Scoring on GPU
 * 
 * Scores moves based on history tables and other heuristics.
 * Useful when scoring many moves for multiple positions.
 */
class BatchMoveScorer {
public:
    BatchMoveScorer();
    ~BatchMoveScorer();
    
    bool initialize();
    bool available() const { return initialized_; }
    
    /**
     * Score moves for multiple positions
     * 
     * @param positions Array of positions
     * @param moves Array of move lists
     * @param move_counts Number of moves per position
     * @param history History table pointer
     * @param count Number of positions
     */
    bool score_moves(const Position* positions[],
                     GPUMove* moves[],
                     const int move_counts[],
                     const int16_t* history,
                     int count);
    
private:
    bool initialized_ = false;
    
    std::unique_ptr<Buffer> positions_buffer_;
    std::unique_ptr<Buffer> moves_buffer_;
    std::unique_ptr<Buffer> history_buffer_;
    std::unique_ptr<Buffer> move_counts_buffer_;
    
    std::unique_ptr<ComputeKernel> score_kernel_;
};

/**
 * GPU Operations Manager
 * 
 * Central manager for all GPU batch operations.
 * Handles resource allocation and provides a unified interface.
 */
class GPUOperations {
public:
    static GPUOperations& instance();
    
    // Initialize all GPU operations
    bool initialize();
    
    // Check availability
    bool available() const { return initialized_; }
    bool see_available() const { return batch_see_ && batch_see_->available(); }
    bool scorer_available() const { return move_scorer_ && move_scorer_->available(); }
    
    // Access individual modules
    BatchSEE& see() { return *batch_see_; }
    BatchMoveScorer& scorer() { return *move_scorer_; }
    
    // Memory statistics
    size_t total_gpu_memory() const;
    
private:
    GPUOperations() = default;
    
    bool initialized_ = false;
    std::unique_ptr<BatchSEE> batch_see_;
    std::unique_ptr<BatchMoveScorer> move_scorer_;
};

// Convenience function
inline GPUOperations& gpu_ops() { return GPUOperations::instance(); }

} // namespace GPU
} // namespace MetalFish

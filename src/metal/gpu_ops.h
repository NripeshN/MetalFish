/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Operations Interface
  ========================

  Provides high-performance GPU-accelerated chess operations:
  - Batch move generation
  - Batch move scoring (MVV-LVA + history)
  - Batch SEE evaluation
  - Batch position evaluation
  - Parallel perft

  Uses unified memory for zero-copy CPU-GPU data sharing.
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include "metal/device.h"
#include <Metal/Metal.hpp>
#include <memory>
#include <vector>

namespace MetalFish {
namespace GPU {

// GPU-friendly position representation (mirrors Metal struct)
struct alignas(8) GPUPosition {
  uint64_t pieces[2][7]; // [color][piece_type]
  uint64_t occupied[3];  // [white, black, all]
  int8_t board[64];      // Piece on each square
  int32_t sideToMove;
  int32_t castlingRights;
  int32_t epSquare;
  int32_t halfmoveClock;

  void from_position(const Position &pos);
};

// GPU move representation
struct alignas(4) GPUMove {
  uint16_t data; // from:6, to:6, type:2, promo:2
  int16_t score; // Move ordering score

  Move to_move() const;
  static GPUMove from_move(Move m);
};

// Attack tables for GPU (pre-computed)
struct AttackTables {
  uint64_t pawnAttacks[2][64];
  uint64_t knightAttacks[64];
  uint64_t kingAttacks[64];
  uint64_t bishopMagics[64];
  uint64_t rookMagics[64];
  uint64_t bishopMasks[64];
  uint64_t rookMasks[64];
  int32_t bishopShifts[64];
  int32_t rookShifts[64];
  uint32_t bishopOffsets[64];
  uint32_t rookOffsets[64];
};

// Configuration for GPU operations
struct GPUOpsConfig {
  int max_batch_size = 256;
  int max_moves_per_position = 256;
  bool async_execution = true;
  bool use_unified_memory = true;
};

// Main GPU Operations class
class GPUOps {
public:
  GPUOps();
  ~GPUOps();

  // Initialize with Metal device
  bool init(MTL::Device *device, const GPUOpsConfig &config = {});

  // Check if initialized
  bool is_ready() const { return initialized_; }

  // ==================== Move Generation ====================

  // Generate moves for a batch of positions on GPU
  // Returns number of legal moves per position
  std::vector<int>
  batch_generate_moves(const std::vector<Position *> &positions,
                       std::vector<std::vector<Move>> &moves);

  // ==================== Move Scoring ====================

  // Score moves for ordering using GPU
  void batch_score_moves(const std::vector<Position *> &positions,
                         std::vector<std::vector<Move>> &moves,
                         const int16_t *history);

  // ==================== SEE Evaluation ====================

  // Compute SEE for a batch of moves
  std::vector<int> batch_see(const std::vector<Position *> &positions,
                             const std::vector<Move> &moves);

  // ==================== Position Evaluation ====================

  // Evaluate positions on GPU (classical + NNUE)
  std::vector<Value> batch_evaluate(const std::vector<Position *> &positions);

  // ==================== Attack Detection ====================

  // Check if squares are attacked
  std::vector<bool> batch_is_attacked(const std::vector<Position *> &positions,
                                      const std::vector<Square> &squares,
                                      const std::vector<Color> &attackers);

  // ==================== Zobrist Hashing ====================

  // Compute Zobrist hashes on GPU
  std::vector<Key> batch_compute_hash(const std::vector<Position *> &positions);

  // ==================== Perft ====================

  // GPU-accelerated perft
  uint64_t gpu_perft(Position &pos, int depth);

  // ==================== Utility ====================

  // Prepare positions for GPU (convert to GPU format)
  void prepare_positions(const std::vector<Position *> &positions);

  // Synchronize GPU operations
  void sync();

  // Get statistics
  size_t get_gpu_memory_used() const;
  double get_average_batch_time() const;

private:
  // Metal resources
  MTL::Device *device_ = nullptr;
  MTL::CommandQueue *queue_ = nullptr;
  MTL::Library *library_ = nullptr;

  // Kernels
  MTL::ComputePipelineState *pawn_movegen_kernel_ = nullptr;
  MTL::ComputePipelineState *knight_movegen_kernel_ = nullptr;
  MTL::ComputePipelineState *king_movegen_kernel_ = nullptr;
  MTL::ComputePipelineState *score_moves_kernel_ = nullptr;
  MTL::ComputePipelineState *see_kernel_ = nullptr;
  MTL::ComputePipelineState *eval_kernel_ = nullptr;
  MTL::ComputePipelineState *attack_kernel_ = nullptr;
  MTL::ComputePipelineState *hash_kernel_ = nullptr;
  MTL::ComputePipelineState *filter_legal_kernel_ = nullptr;
  MTL::ComputePipelineState *sort_kernel_ = nullptr;

  // Buffers (persistent, unified memory)
  MTL::Buffer *positions_buffer_ = nullptr;
  MTL::Buffer *moves_buffer_ = nullptr;
  MTL::Buffer *move_counts_buffer_ = nullptr;
  MTL::Buffer *attack_tables_buffer_ = nullptr;
  MTL::Buffer *results_buffer_ = nullptr;
  MTL::Buffer *history_buffer_ = nullptr;

  // Magic bitboard tables
  MTL::Buffer *bishop_table_buffer_ = nullptr;
  MTL::Buffer *rook_table_buffer_ = nullptr;

  // Zobrist keys
  MTL::Buffer *zobrist_pieces_buffer_ = nullptr;
  MTL::Buffer *zobrist_castling_buffer_ = nullptr;
  MTL::Buffer *zobrist_ep_buffer_ = nullptr;

  // Configuration
  GPUOpsConfig config_;
  bool initialized_ = false;

  // Statistics
  mutable double total_time_ = 0;
  mutable size_t batch_count_ = 0;

  // Helper functions
  bool load_kernels();
  bool allocate_buffers();
  void upload_attack_tables();
  void upload_zobrist_keys();
};

// Global GPU operations instance
extern std::unique_ptr<GPUOps> gpu_ops;

// Initialize global GPU operations
bool init_gpu_ops();

} // namespace GPU
} // namespace MetalFish

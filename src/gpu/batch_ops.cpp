/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU-accelerated Batch Operations Implementation
*/

#include "batch_ops.h"
#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include <chrono>
#include <cstring>
#include <iostream>

namespace MetalFish {
namespace GPU {

// ============================================================================
// GPUPosition Implementation
// ============================================================================

void GPUPosition::from_position(const Position &pos) {
  // Copy piece bitboards
  for (int c = 0; c < 2; c++) {
    for (int pt = 0; pt < 7; pt++) {
      pieces[c][pt] = pos.pieces(Color(c), PieceType(pt));
    }
  }

  // Occupancy
  occupied[0] = pos.pieces(WHITE);
  occupied[1] = pos.pieces(BLACK);
  occupied[2] = pos.pieces();

  // Board array
  for (int sq = 0; sq < 64; sq++) {
    Piece p = pos.piece_on(Square(sq));
    board[sq] = static_cast<int8_t>(p);
  }

  // State
  side_to_move = pos.side_to_move();
  castling_rights = pos.can_castle(ANY_CASTLING);
  ep_square = pos.ep_square();
  rule50 = pos.rule50_count();
}

// ============================================================================
// GPUMove Implementation
// ============================================================================

GPUMove GPUMove::from_move(Move m) {
  GPUMove gm;
  int from = m.from_sq();
  int to = m.to_sq();
  int type = 0;
  int promo = 0;

  MoveType mt = m.type_of();
  if (mt == PROMOTION) {
    type = 1;
    promo = m.promotion_type() - KNIGHT;
  } else if (mt == EN_PASSANT) {
    type = 2;
  } else if (mt == CASTLING) {
    type = 3;
  }

  gm.data = (from & 0x3F) | ((to & 0x3F) << 6) | ((type & 0x3) << 12) |
            ((promo & 0x3) << 14);
  gm.score = 0;
  return gm;
}

Move GPUMove::to_move() const {
  int from = data & 0x3F;
  int to = (data >> 6) & 0x3F;
  int type = (data >> 12) & 0x3;
  int promo = (data >> 14) & 0x3;

  if (type == 0) {
    return Move(Square(from), Square(to));
  } else if (type == 1) {
    return Move::make<PROMOTION>(Square(from), Square(to),
                                 PieceType(promo + KNIGHT));
  } else if (type == 2) {
    return Move::make<EN_PASSANT>(Square(from), Square(to));
  } else {
    return Move::make<CASTLING>(Square(from), Square(to));
  }
}

// ============================================================================
// BatchSEE Implementation
// ============================================================================

// SEE shader source
static const char *SEE_SHADER_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

typedef uint64_t Bitboard;

constant int PIECE_VALUES[7] = {0, 100, 320, 330, 500, 900, 20000};

struct GPUPosition {
    uint64_t pieces[2][7];
    uint64_t occupied[3];
    int8_t board[64];
    int side_to_move;
    int castling_rights;
    int ep_square;
    int rule50;
};

struct GPUMove {
    uint16_t data;
    int16_t score;
};

struct AttackTables {
    uint64_t pawnAttacks[2][64];
    uint64_t knightAttacks[64];
    uint64_t kingAttacks[64];
};

inline int lsb(uint64_t b) {
    return ctz(b);
}

inline uint64_t lsb_bb(uint64_t b) {
    return b & -b;
}

inline int popcount(uint64_t b) {
    return metal::popcount(b);
}

inline int piece_type(int piece) {
    return piece & 7;
}

inline int piece_color(int piece) {
    return piece >> 3;
}

// Simplified SEE - just checks captures
kernel void batch_see(
    device const GPUPosition* positions [[buffer(0)]],
    device const GPUMove* moves [[buffer(1)]],
    device const int* thresholds [[buffer(2)]],
    device const AttackTables* tables [[buffer(3)]],
    device int* results [[buffer(4)]],
    constant int& batch_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uint(batch_size))
        return;
    
    GPUPosition pos = positions[gid];
    GPUMove move = moves[gid];
    int threshold = thresholds[gid];
    
    int from = move.data & 0x3F;
    int to = (move.data >> 6) & 0x3F;
    
    int captured_piece = pos.board[to];
    int moving_piece = pos.board[from];
    
    // Simple SEE approximation
    int value = 0;
    if (captured_piece != 0) {
        value = PIECE_VALUES[piece_type(captured_piece)];
    }
    
    // Subtract moving piece value as potential loss
    value -= PIECE_VALUES[piece_type(moving_piece)];
    
    // If we're already above threshold, it's good
    if (value >= threshold) {
        results[gid] = 1;
        return;
    }
    
    // Check if the destination is defended
    int us = pos.side_to_move;
    int them = 1 - us;
    
    // Check pawn attacks
    if (tables->pawnAttacks[them][to] & pos.pieces[them][1]) {
        results[gid] = value >= threshold ? 1 : 0;
        return;
    }
    
    // Check knight attacks
    if (tables->knightAttacks[to] & pos.pieces[them][2]) {
        value -= PIECE_VALUES[2];
    }
    
    results[gid] = value >= threshold ? 1 : 0;
}
)";

BatchSEE::BatchSEE() {}

BatchSEE::~BatchSEE() {}

bool BatchSEE::initialize() {
  if (!gpu_available()) {
    return false;
  }

  if (!load_kernels()) {
    return false;
  }

  if (!allocate_buffers()) {
    return false;
  }

  upload_attack_tables();

  initialized_ = true;
  std::cout << "[GPU SEE] Initialized successfully" << std::endl;
  return true;
}

bool BatchSEE::load_kernels() {
#ifdef USE_METAL
  auto &backend = gpu();

  if (!backend.compile_library("see", SEE_SHADER_SOURCE)) {
    std::cerr << "[GPU SEE] Failed to compile SEE shader" << std::endl;
    return false;
  }

  see_kernel_ = backend.create_kernel("batch_see", "see");

  if (!see_kernel_ || !see_kernel_->valid()) {
    std::cerr << "[GPU SEE] Failed to create batch_see kernel" << std::endl;
    return false;
  }

  return true;
#else
  return false;
#endif
}

bool BatchSEE::allocate_buffers() {
  auto &backend = gpu();

  positions_buffer_ =
      backend.create_buffer(MAX_SEE_BATCH * sizeof(GPUPosition));
  moves_buffer_ = backend.create_buffer(MAX_SEE_BATCH * sizeof(GPUMove));
  thresholds_buffer_ = backend.create_buffer(MAX_SEE_BATCH * sizeof(int32_t));
  results_buffer_ = backend.create_buffer(MAX_SEE_BATCH * sizeof(int32_t));

  // Attack tables
  struct AttackTables {
    uint64_t pawnAttacks[2][64];
    uint64_t knightAttacks[64];
    uint64_t kingAttacks[64];
  };
  attack_tables_buffer_ = backend.create_buffer(sizeof(AttackTables));

  return positions_buffer_ && moves_buffer_ && thresholds_buffer_ &&
         results_buffer_ && attack_tables_buffer_;
}

void BatchSEE::upload_attack_tables() {
  if (!attack_tables_buffer_)
    return;

  struct AttackTables {
    uint64_t pawnAttacks[2][64];
    uint64_t knightAttacks[64];
    uint64_t kingAttacks[64];
  };

  AttackTables *tables = attack_tables_buffer_->as<AttackTables>();

  // Copy attack tables from CPU
  // PseudoAttacks[WHITE/BLACK][sq] contains pawn attacks
  for (int sq = 0; sq < 64; sq++) {
    tables->pawnAttacks[WHITE][sq] = PseudoAttacks[WHITE][Square(sq)];
    tables->pawnAttacks[BLACK][sq] = PseudoAttacks[BLACK][Square(sq)];
    tables->knightAttacks[sq] = PseudoAttacks[KNIGHT][Square(sq)];
    tables->kingAttacks[sq] = PseudoAttacks[KING][Square(sq)];
  }
}

bool BatchSEE::compute(const Position *positions[], const Move moves[],
                       const int thresholds[], bool results[], int count) {
  if (!initialized_ || count == 0) {
    // CPU fallback
    for (int i = 0; i < count; i++) {
      results[i] = positions[i]->see_ge(moves[i], Value(thresholds[i]));
    }
    cpu_computations_ += count;
    return false;
  }

  // For small batches, CPU is faster
  if (count < 16) {
    for (int i = 0; i < count; i++) {
      results[i] = positions[i]->see_ge(moves[i], Value(thresholds[i]));
    }
    cpu_computations_ += count;
    return false;
  }

  auto start = std::chrono::high_resolution_clock::now();

  int batch_size = std::min(count, MAX_SEE_BATCH);

  // Upload positions
  GPUPosition *gpu_positions = positions_buffer_->as<GPUPosition>();
  for (int i = 0; i < batch_size; i++) {
    gpu_positions[i].from_position(*positions[i]);
  }

  // Upload moves
  GPUMove *gpu_moves = moves_buffer_->as<GPUMove>();
  for (int i = 0; i < batch_size; i++) {
    gpu_moves[i] = GPUMove::from_move(moves[i]);
  }

  // Upload thresholds
  int32_t *gpu_thresholds = thresholds_buffer_->as<int32_t>();
  std::copy(thresholds, thresholds + batch_size, gpu_thresholds);

  // Dispatch kernel
  auto &backend = gpu();
  auto encoder = backend.create_encoder();

  encoder->set_kernel(see_kernel_.get());
  encoder->set_buffer(positions_buffer_.get(), 0);
  encoder->set_buffer(moves_buffer_.get(), 1);
  encoder->set_buffer(thresholds_buffer_.get(), 2);
  encoder->set_buffer(attack_tables_buffer_.get(), 3);
  encoder->set_buffer(results_buffer_.get(), 4);
  encoder->set_value(batch_size, 5);

  encoder->dispatch_threads(batch_size);

  backend.submit_and_wait(encoder.get());

  // Read results
  int32_t *gpu_results = results_buffer_->as<int32_t>();
  for (int i = 0; i < batch_size; i++) {
    results[i] = gpu_results[i] != 0;
  }

  auto end = std::chrono::high_resolution_clock::now();
  total_time_ms_ +=
      std::chrono::duration<double, std::milli>(end - start).count();
  batch_count_++;
  gpu_computations_ += batch_size;

  // Handle remaining positions with CPU
  if (count > batch_size) {
    for (int i = batch_size; i < count; i++) {
      results[i] = positions[i]->see_ge(moves[i], Value(thresholds[i]));
    }
    cpu_computations_ += (count - batch_size);
  }

  return true;
}

// ============================================================================
// BatchMoveScorer Implementation
// ============================================================================

BatchMoveScorer::BatchMoveScorer() {}

BatchMoveScorer::~BatchMoveScorer() {}

bool BatchMoveScorer::initialize() {
  // TODO: Implement move scoring kernel
  return false;
}

bool BatchMoveScorer::score_moves(const Position *positions[], GPUMove *moves[],
                                  const int move_counts[],
                                  const int16_t *history, int count) {
  // TODO: Implement GPU move scoring
  return false;
}

// ============================================================================
// GPUOperations Implementation
// ============================================================================

GPUOperations &GPUOperations::instance() {
  static GPUOperations instance;
  return instance;
}

bool GPUOperations::initialize() {
  if (initialized_)
    return true;

  if (!gpu_available()) {
    std::cout << "[GPU Ops] GPU not available" << std::endl;
    return false;
  }

  batch_see_ = std::make_unique<BatchSEE>();
  if (!batch_see_->initialize()) {
    std::cerr << "[GPU Ops] Failed to initialize BatchSEE" << std::endl;
  }

  move_scorer_ = std::make_unique<BatchMoveScorer>();
  if (!move_scorer_->initialize()) {
    // Not critical, move scoring can fall back to CPU
  }

  initialized_ = true;
  std::cout << "[GPU Ops] Initialized" << std::endl;
  return true;
}

size_t GPUOperations::total_gpu_memory() const {
  if (!gpu_available())
    return 0;
  return gpu().allocated_memory();
}

} // namespace GPU
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Lc0 Policy Tables Implementation
*/

#include "policy_tables.h"
#include "../core/position.h"
#include "../core/movegen.h"

#include <cmath>
#include <cstdlib>

namespace MetalFish {
namespace NN {

namespace PolicyTables {

static bool initialized = false;

// Direction vectors for queen moves (N, NE, E, SE, S, SW, W, NW)
static const int kQueenDirections[8][2] = {
  {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}
};

// Knight move offsets
static const int kKnightOffsets[8][2] = {
  {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}
};

void initialize() {
  // Policy tables are computed on-the-fly for now
  // In a full implementation, would pre-compute and store mapping tables
  initialized = true;
}

bool is_initialized() {
  return initialized;
}

int get_queen_direction(int from_sq, int to_sq) {
  int from_file = from_sq % 8;
  int from_rank = from_sq / 8;
  int to_file = to_sq % 8;
  int to_rank = to_sq / 8;
  
  int df = to_file - from_file;
  int dr = to_rank - from_rank;
  
  // Normalize to get direction
  int dir_f = (df == 0) ? 0 : (df > 0 ? 1 : -1);
  int dir_r = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
  
  // Map to direction index
  for (int i = 0; i < 8; ++i) {
    if (kQueenDirections[i][0] == dir_f && kQueenDirections[i][1] == dir_r) {
      return i;
    }
  }
  
  return -1; // Not a queen-like move
}

int get_knight_direction(int from_sq, int to_sq) {
  // TODO: Verify this matches Lc0's exact knight move policy encoding
  // The current implementation may not produce identical indices to Lc0
  // See issue noted in src/nn/README.md - needs comparison with Lc0 reference
  
  int from_file = from_sq % 8;
  int from_rank = from_sq / 8;
  int to_file = to_sq % 8;
  int to_rank = to_sq / 8;
  
  int df = to_file - from_file;
  int dr = to_rank - from_rank;
  
  for (int i = 0; i < 8; ++i) {
    if (kKnightOffsets[i][0] == df && kKnightOffsets[i][1] == dr) {
      return i;
    }
  }
  
  return -1;
}

bool is_underpromotion(Move move) {
  if (move.type_of() != PROMOTION) return false;
  
  PieceType promo = move.promotion_type();
  return promo != QUEEN; // Knight, Bishop, or Rook promotion
}

} // namespace PolicyTables

int move_to_policy_index(Move move, const Position& pos) {
  if (!PolicyTables::is_initialized()) {
    PolicyTables::initialize();
  }
  
  Square from = move.from_sq();
  Square to = move.to_sq();
  
  // Flip squares if black to move (Lc0 convention)
  bool flip = (pos.side_to_move() == BLACK);
  if (flip) {
    from = Square(int(from) ^ 56); // Flip rank
    to = Square(int(to) ^ 56);
  }
  
  int from_idx = static_cast<int>(from);
  int to_idx = static_cast<int>(to);
  
  // Handle promotions
  if (move.type_of() == PROMOTION) {
    PieceType promo = move.promotion_type();
    
    if (promo == QUEEN) {
      // Queen promotion treated as normal queen move
      int dir = PolicyTables::get_queen_direction(from_idx, to_idx);
      if (dir < 0) return -1;
      
      int distance = std::max(std::abs(file_of(from) - file_of(to)),
                             std::abs(rank_of(from) - rank_of(to)));
      
      return from_idx * 56 + dir * 7 + (distance - 1);
    } else {
      // Underpromotion (Knight, Bishop, Rook)
      // Simplified index calculation - full implementation would need
      // proper underpromotion table
      int promo_type = (promo == KNIGHT) ? 0 : (promo == BISHOP ? 1 : 2);
      int dir_offset = (file_of(to) > file_of(from)) ? 1 : 0;
      
      // Base offset for underpromotions (after queen + knight moves)
      int base = 64 * 56 + 64 * 8;
      return base + promo_type * 128 + from_idx * 2 + dir_offset;
    }
  }
  
  // Check if it's a knight move
  int knight_dir = PolicyTables::get_knight_direction(from_idx, to_idx);
  if (knight_dir >= 0) {
    // Knight moves: base offset + from_square * 8 + direction
    return 64 * 56 + from_idx * 8 + knight_dir;
  }
  
  // Otherwise, it's a queen-like move (including pawns, castling)
  int queen_dir = PolicyTables::get_queen_direction(from_idx, to_idx);
  if (queen_dir < 0) return -1;
  
  int distance = std::max(std::abs(file_of(from) - file_of(to)),
                         std::abs(rank_of(from) - rank_of(to)));
  
  // Queen moves: from_square * 56 + direction * 7 + (distance - 1)
  return from_idx * 56 + queen_dir * 7 + (distance - 1);
}

Move policy_index_to_move(int index, const Position& pos) {
  // Placeholder - full implementation would reverse the mapping
  // This requires maintaining legal move generation
  return Move::none();
}

std::vector<std::pair<Move, int>> get_legal_moves_with_indices(const Position& pos) {
  std::vector<std::pair<Move, int>> result;
  
  MoveList<LEGAL> moves(pos);
  for (const auto& move : moves) {
    int policy_idx = move_to_policy_index(move, pos);
    if (policy_idx >= 0) {
      result.emplace_back(move, policy_idx);
    }
  }
  
  return result;
}

} // namespace NN
} // namespace MetalFish

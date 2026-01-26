/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file encoder.cpp
 * @brief MetalFish source file.
 */

  Lc0 Position Encoder Implementation
*/

#include "encoder.h"
#include "../core/position.h"
#include "../core/types.h"

namespace MetalFish {
namespace NN {

void Lc0PositionEncoder::encode(const std::vector<Position>& positions, 
                                 EncodedPosition& output) const {
  output.planes.fill(0.0f);
  
  if (positions.empty()) return;
  
  const Position& current = positions[0];
  bool flip_board = (current.side_to_move() == BLACK);
  
  // Encode up to 8 history positions
  int num_positions = std::min(static_cast<int>(positions.size()), HISTORY_POSITIONS);
  
  for (int hist_idx = 0; hist_idx < num_positions; ++hist_idx) {
    const Position& pos = positions[hist_idx];
    
    // Calculate repetition count (simplified - would need full history)
    int repetitions = 0; // TODO: Implement proper repetition detection
    
    encode_history_position(pos, hist_idx, repetitions, output);
  }
  
  // Encode auxiliary planes
  encode_auxiliary(current, output);
}

void Lc0PositionEncoder::encode(const Position& pos, EncodedPosition& output) const {
  output.planes.fill(0.0f);
  
  // Encode single position at history index 0
  bool flip_board = (pos.side_to_move() == BLACK);
  int repetitions = 0; // No history, so no repetitions
  
  encode_history_position(pos, 0, repetitions, output);
  
  // Encode auxiliary planes
  encode_auxiliary(pos, output);
}

void Lc0PositionEncoder::encode_history_position(const Position& pos, 
                                                   int history_idx,
                                                   int repetitions, 
                                                   EncodedPosition& output) const {
  bool flip_board = (pos.side_to_move() == BLACK);
  int base_plane = history_idx * PLANES_PER_POSITION;
  
  // Encode pieces
  // White pieces: planes 0-5
  for (int pt = PAWN; pt <= KING; ++pt) {
    int plane_idx = base_plane + (pt - PAWN);
    encode_pieces(pos, pt, WHITE, output.get_plane(plane_idx), flip_board);
  }
  
  // Black pieces: planes 6-11
  for (int pt = PAWN; pt <= KING; ++pt) {
    int plane_idx = base_plane + 6 + (pt - PAWN);
    encode_pieces(pos, pt, BLACK, output.get_plane(plane_idx), flip_board);
  }
  
  // Repetition plane (plane 12 of this history position)
  float* rep_plane = output.get_plane(base_plane + 12);
  if (repetitions > 0) {
    std::fill_n(rep_plane, 64, repetitions >= 2 ? 1.0f : 0.5f);
  }
}

void Lc0PositionEncoder::encode_pieces(const Position& pos, int piece_type, 
                                        int color, float* plane, 
                                        bool flip_board) const {
  // Get bitboard for this piece type and color
  Bitboard pieces = pos.pieces(static_cast<Color>(color), static_cast<PieceType>(piece_type));
  
  // Set plane bits for each piece
  while (pieces) {
    Square sq = pop_lsb(pieces);
    int flipped_sq = flip_square(static_cast<int>(sq), flip_board);
    plane[flipped_sq] = 1.0f;
  }
}

void Lc0PositionEncoder::encode_auxiliary(const Position& pos, 
                                           EncodedPosition& output) const {
  int base_plane = HISTORY_POSITIONS * PLANES_PER_POSITION; // Plane 104
  
  // Plane 104: Side to move (all 1s if white, all 0s if black)
  float* stm_plane = output.get_plane(base_plane + 0);
  if (pos.side_to_move() == WHITE) {
    std::fill_n(stm_plane, 64, 1.0f);
  }
  
  // Plane 105: Rule50 counter normalized to [0, 1]
  float* rule50_plane = output.get_plane(base_plane + 1);
  float rule50_value = std::min(pos.rule50_count(), 99) / 99.0f;
  std::fill_n(rule50_plane, 64, rule50_value);
  
  // Plane 106: Zeroed (legacy)
  // Already zeroed by initialization
  
  // Plane 107: All 1s plane
  float* ones_plane = output.get_plane(base_plane + 3);
  std::fill_n(ones_plane, 64, 1.0f);
  
  // Planes 108-111: Castling rights
  // Note: Lc0 encodes castling from side-to-move's perspective
  bool flip = (pos.side_to_move() == BLACK);
  
  // For white to move: WK, WQ, BK, BQ
  // For black to move: BK, BQ, WK, WQ (flipped)
  auto set_castling = [&](int plane_offset, bool can_castle) {
    if (can_castle) {
      float* plane = output.get_plane(base_plane + 4 + plane_offset);
      std::fill_n(plane, 64, 1.0f);
    }
  };
  
  if (!flip) {
    set_castling(0, pos.can_castle(WHITE_OO));
    set_castling(1, pos.can_castle(WHITE_OOO));
    set_castling(2, pos.can_castle(BLACK_OO));
    set_castling(3, pos.can_castle(BLACK_OOO));
  } else {
    set_castling(0, pos.can_castle(BLACK_OO));
    set_castling(1, pos.can_castle(BLACK_OOO));
    set_castling(2, pos.can_castle(WHITE_OO));
    set_castling(3, pos.can_castle(WHITE_OOO));
  }
}

} // namespace NN
} // namespace MetalFish
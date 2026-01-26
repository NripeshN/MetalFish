/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file encoder.h
 * @brief MetalFish source file.
 */

  Lc0 Position Encoder - 112-Plane Input Format
  
  Encodes chess positions into Lc0's 112-plane neural network input format:
  - 8 history positions × 13 planes (6 piece types × 2 colors + castling/enpassant)
  - 8 auxiliary planes (side to move, rule50, etc.)
  
  Handles board flipping for black-to-move positions to maintain
  side-to-move perspective consistency with Lc0.
  
  Licensed under GPL-3.0
*/

#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace MetalFish {

class Position;

namespace NN {

// Lc0's 112-plane input encoding
// 8 history positions × 13 planes + 8 auxiliary planes = 112 planes
constexpr int INPUT_PLANES = 112;
constexpr int HISTORY_POSITIONS = 8;
constexpr int PLANES_PER_POSITION = 13; // 12 pieces + repetition
constexpr int AUXILIARY_PLANES = 8;

// Input plane layout:
// Planes 0-103: History (8 positions × 13 planes)
//   For each position:
//     0-5:   White pieces (P,N,B,R,Q,K)
//     6-11:  Black pieces (P,N,B,R,Q,K)
//     12:    Repetition count (1 or 2+)
// Planes 104-111: Auxiliary
//   104: Side to move (all 1s if white, all 0s if black)
//   105: Rule50 move counter / 99
//   106: Zeroed plane (historically used)
//   107: All 1s plane
//   108-111: Castling rights (WK, WQ, BK, BQ)

struct EncodedPosition {
  // 112 planes × 64 squares (8×8 board)
  // Stored in row-major order from white's perspective
  // Each plane is 64 floats (0.0 or 1.0)
  std::array<float, INPUT_PLANES * 64> planes;
  
  EncodedPosition() {
    planes.fill(0.0f);
  }
  
  // Access plane data
  float* get_plane(int plane_idx) {
    return &planes[plane_idx * 64];
  }
  
  const float* get_plane(int plane_idx) const {
    return &planes[plane_idx * 64];
  }
};

// Lc0 Position Encoder
class Lc0PositionEncoder {
public:
  Lc0PositionEncoder() = default;
  
  // Encode a position with its history
  // positions[0] is the current position, positions[1] is 1 ply back, etc.
  // If fewer than 8 positions available, earlier positions are zeroed
  void encode(const std::vector<Position>& positions, 
              EncodedPosition& output) const;
  
  // Encode single position (no history)
  void encode(const Position& pos, EncodedPosition& output) const;
  
  // Encode position at specific history index (0 = current, 1 = -1 ply, etc.)
  void encode_history_position(const Position& pos, int history_idx, 
                               int repetitions, EncodedPosition& output) const;
  
private:
  // Encode a single piece type for one color
  void encode_pieces(const Position& pos, int piece_type, int color,
                     float* plane, bool flip_board) const;
  
  // Set auxiliary planes
  void encode_auxiliary(const Position& pos, EncodedPosition& output) const;
  
  // Convert square index accounting for board flip
  // Lc0 always encodes from side-to-move's perspective
  int flip_square(int sq, bool flip) const {
    if (flip) {
      // Vertical flip: rank 0 ↔ 7, rank 1 ↔ 6, etc.
      int file = sq % 8;
      int rank = sq / 8;
      return file + (7 - rank) * 8;
    }
    return sq;
  }
};

} // namespace NN
} // namespace MetalFish
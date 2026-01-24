/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "encoder.h"

#include <cstring>

namespace MetalFish {
namespace NN {

namespace {

// Extract bitboard for a specific piece type and color
uint64_t GetPieceBitboard(const Position& pos, PieceType pt, Color c) {
  Bitboard bb = pos.pieces(c, pt);
  return bb;
}

// Fill a plane from a bitboard
void FillPlaneFromBitboard(std::array<float, 64>& plane, uint64_t bitboard) {
  for (int sq = 0; sq < 64; ++sq) {
    plane[sq] = (bitboard & (1ULL << sq)) ? 1.0f : 0.0f;
  }
}

// Set all values in a plane
void SetPlane(std::array<float, 64>& plane, float value) {
  for (int i = 0; i < 64; ++i) {
    plane[i] = value;
  }
}

}  // namespace

bool IsCanonicalFormat(MetalFishNN::NetworkFormat::InputFormat input_format) {
  using IF = MetalFishNN::NetworkFormat;
  return input_format == IF::INPUT_112_WITH_CANONICALIZATION ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2 ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}

int TransformForPosition(MetalFishNN::NetworkFormat::InputFormat input_format,
                         const std::vector<Position>& history) {
  // For now, no canonicalization transform
  // Full implementation would compute optimal board orientation
  return 0;
}

InputPlanes EncodePositionForNN(
    MetalFishNN::NetworkFormat::InputFormat input_format,
    const std::vector<Position>& position_history,
    int history_planes,
    FillEmptyHistory fill_empty_history,
    int* transform_out) {
  
  InputPlanes result{};
  
  if (position_history.empty()) {
    return result;
  }
  
  // Get side to move
  const Position& current_pos = position_history.back();
  Color us = current_pos.side_to_move();
  Color them = ~us;
  
  // Encode position history (8 positions, 13 planes each)
  int history_size = std::min(static_cast<int>(position_history.size()), 
                              std::min(history_planes, kMoveHistory));
  
  for (int i = 0; i < history_size; ++i) {
    // Get position from history (most recent first)
    int history_idx = position_history.size() - 1 - i;
    const Position& pos = position_history[history_idx];
    
    // Determine perspective (always from current side to move)
    Color perspective_us = us;
    Color perspective_them = them;
    
    // If this is an old position where opponent moved, flip perspective
    if (i % 2 == 1) {
      std::swap(perspective_us, perspective_them);
    }
    
    int base = i * kPlanesPerBoard;
    
    // Encode our pieces (6 planes)
    FillPlaneFromBitboard(result[base + 0], GetPieceBitboard(pos, PAWN, perspective_us));
    FillPlaneFromBitboard(result[base + 1], GetPieceBitboard(pos, KNIGHT, perspective_us));
    FillPlaneFromBitboard(result[base + 2], GetPieceBitboard(pos, BISHOP, perspective_us));
    FillPlaneFromBitboard(result[base + 3], GetPieceBitboard(pos, ROOK, perspective_us));
    FillPlaneFromBitboard(result[base + 4], GetPieceBitboard(pos, QUEEN, perspective_us));
    FillPlaneFromBitboard(result[base + 5], GetPieceBitboard(pos, KING, perspective_us));
    
    // Encode opponent pieces (6 planes)
    FillPlaneFromBitboard(result[base + 6], GetPieceBitboard(pos, PAWN, perspective_them));
    FillPlaneFromBitboard(result[base + 7], GetPieceBitboard(pos, KNIGHT, perspective_them));
    FillPlaneFromBitboard(result[base + 8], GetPieceBitboard(pos, BISHOP, perspective_them));
    FillPlaneFromBitboard(result[base + 9], GetPieceBitboard(pos, ROOK, perspective_them));
    FillPlaneFromBitboard(result[base + 10], GetPieceBitboard(pos, QUEEN, perspective_them));
    FillPlaneFromBitboard(result[base + 11], GetPieceBitboard(pos, KING, perspective_them));
    
    // Repetition plane (1 if position repeats)
    SetPlane(result[base + 12], 0.0f);  // Simplified: no repetition tracking
  }
  
  // Fill auxiliary planes (8 planes starting at index 104)
  int aux_base = kAuxPlaneBase;
  
  // Plane 0-3: Castling rights
  SetPlane(result[aux_base + 0], current_pos.can_castle(WHITE_OO) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 1], current_pos.can_castle(WHITE_OOO) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 2], current_pos.can_castle(BLACK_OO) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 3], current_pos.can_castle(BLACK_OOO) ? 1.0f : 0.0f);
  
  // Plane 4: Color to move (or en passant in canonical format)
  if (IsCanonicalFormat(input_format)) {
    // En passant square
    Square ep_sq = current_pos.ep_square();
    SetPlane(result[aux_base + 4], 0.0f);
    if (ep_sq != SQ_NONE) {
      result[aux_base + 4][ep_sq] = 1.0f;
    }
  } else {
    SetPlane(result[aux_base + 4], us == BLACK ? 1.0f : 0.0f);
  }
  
  // Plane 5: Rule50 counter (halfmove clock)
  using IF = MetalFishNN::NetworkFormat;
  bool is_hectoplies = (input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
                        input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON);
  
  float rule50_value = is_hectoplies ? 
      (current_pos.rule50_count() / 100.0f) : 
      static_cast<float>(current_pos.rule50_count());
  SetPlane(result[aux_base + 5], rule50_value);
  
  // Plane 6: Move count (zeros for now, or armageddon color)
  SetPlane(result[aux_base + 6], 0.0f);
  
  // Plane 7: All ones (helps NN detect board edges)
  SetPlane(result[aux_base + 7], 1.0f);
  
  if (transform_out) {
    *transform_out = 0;  // No transform for now
  }
  
  return result;
}

InputPlanes EncodePositionForNN(
    const Position& pos,
    MetalFishNN::NetworkFormat::InputFormat input_format) {
  
  // Position can't be copied, so we need to pass it by reference
  // For simplicity, just encode current position without history
  std::vector<Position> history;
  // Can't copy position, so we'll just encode it directly
  
  InputPlanes result{};
  
  Color us = pos.side_to_move();
  Color them = ~us;
  
  // Encode current position only (no history)
  int base = 0;
  
  // Our pieces (6 planes)
  FillPlaneFromBitboard(result[base + 0], GetPieceBitboard(pos, PAWN, us));
  FillPlaneFromBitboard(result[base + 1], GetPieceBitboard(pos, KNIGHT, us));
  FillPlaneFromBitboard(result[base + 2], GetPieceBitboard(pos, BISHOP, us));
  FillPlaneFromBitboard(result[base + 3], GetPieceBitboard(pos, ROOK, us));
  FillPlaneFromBitboard(result[base + 4], GetPieceBitboard(pos, QUEEN, us));
  FillPlaneFromBitboard(result[base + 5], GetPieceBitboard(pos, KING, us));
  
  // Their pieces (6 planes)
  FillPlaneFromBitboard(result[base + 6], GetPieceBitboard(pos, PAWN, them));
  FillPlaneFromBitboard(result[base + 7], GetPieceBitboard(pos, KNIGHT, them));
  FillPlaneFromBitboard(result[base + 8], GetPieceBitboard(pos, BISHOP, them));
  FillPlaneFromBitboard(result[base + 9], GetPieceBitboard(pos, ROOK, them));
  FillPlaneFromBitboard(result[base + 10], GetPieceBitboard(pos, QUEEN, them));
  FillPlaneFromBitboard(result[base + 11], GetPieceBitboard(pos, KING, them));
  
  // Repetition plane
  SetPlane(result[base + 12], 0.0f);
  
  // Fill auxiliary planes
  int aux_base = kAuxPlaneBase;
  
  SetPlane(result[aux_base + 0], pos.can_castle(WHITE_OO) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 1], pos.can_castle(WHITE_OOO) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 2], pos.can_castle(BLACK_OO) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 3], pos.can_castle(BLACK_OOO) ? 1.0f : 0.0f);
  
  SetPlane(result[aux_base + 4], us == BLACK ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 5], static_cast<float>(pos.rule50_count()));
  SetPlane(result[aux_base + 6], 0.0f);
  SetPlane(result[aux_base + 7], 1.0f);
  
  return result;
}

}  // namespace NN
}  // namespace MetalFish

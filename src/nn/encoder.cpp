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

// Board transform constants
enum BoardTransform {
  NoTransform = 0,
  FlipTransform = 1,       // Horizontal flip
  MirrorTransform = 2,     // Vertical mirror
  TransposeTransform = 4,  // Diagonal transpose
};

// Get lowest bit position
inline unsigned long GetLowestBit(uint64_t value) {
#if defined(_MSC_VER) && defined(_WIN64)
  unsigned long result;
  _BitScanForward64(&result, value);
  return result;
#elif defined(_MSC_VER)
  unsigned long result;
  if (value & 0xFFFFFFFF) {
    _BitScanForward(&result, value);
  } else {
    _BitScanForward(&result, value >> 32);
    result += 32;
  }
  return result;
#else
  return __builtin_ctzll(value);
#endif
}

// Reverse bits within each byte (horizontal flip)
inline uint64_t ReverseBitsInBytes(uint64_t v) {
  v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
  v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
  v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
  return v;
}

// Reverse bytes (vertical mirror)
inline uint64_t ReverseBytesInBytes(uint64_t v) {
  v = (v & 0x00000000FFFFFFFF) << 32 | (v & 0xFFFFFFFF00000000) >> 32;
  v = (v & 0x0000FFFF0000FFFF) << 16 | (v & 0xFFFF0000FFFF0000) >> 16;
  v = (v & 0x00FF00FF00FF00FF) << 8 | (v & 0xFF00FF00FF00FF00) >> 8;
  return v;
}

// Transpose 8x8 bit matrix (diagonal transpose)
inline uint64_t TransposeBitsInBytes(uint64_t v) {
  v = (v & 0xAA00AA00AA00AA00ULL) >> 9 | (v & 0x0055005500550055ULL) << 9 |
      (v & 0x55AA55AA55AA55AAULL);
  v = (v & 0xCCCC0000CCCC0000ULL) >> 18 | (v & 0x0000333300003333ULL) << 18 |
      (v & 0x3333CCCC3333CCCCULL);
  v = (v & 0xF0F0F0F000000000ULL) >> 36 | (v & 0x000000000F0F0F0FULL) << 36 |
      (v & 0x0F0F0F0FF0F0F0F0ULL);
  return v;
}

// Apply transform to a bitboard
inline uint64_t ApplyTransform(uint64_t bitboard, int transform) {
  if (bitboard == 0 || bitboard == ~0ULL) return bitboard;
  
  uint64_t v = bitboard;
  if ((transform & FlipTransform) != 0) {
    v = ReverseBitsInBytes(v);
  }
  if ((transform & MirrorTransform) != 0) {
    v = ReverseBytesInBytes(v);
  }
  if ((transform & TransposeTransform) != 0) {
    v = TransposeBitsInBytes(v);
  }
  return v;
}

// Compare transposing for canonicalization
int CompareTransposing(uint64_t board, int initial_transform) {
  uint64_t value = board;
  if ((initial_transform & FlipTransform) != 0) {
    value = ReverseBitsInBytes(value);
  }
  if ((initial_transform & MirrorTransform) != 0) {
    value = ReverseBytesInBytes(value);
  }
  auto alternative = TransposeBitsInBytes(value);
  if (value < alternative) return -1;
  if (value > alternative) return 1;
  return 0;
}

// Choose optimal transform for canonicalization
int ChooseTransform(const Position& pos, Color us) {
  // If there are any castling options, no transform is valid
  if (pos.can_castle(ANY_CASTLING)) {
    return NoTransform;
  }
  
  uint64_t our_king = pos.pieces(us, KING);
  int transform = NoTransform;
  
  // Flip horizontally if king on left half
  if ((our_king & 0x0F0F0F0F0F0F0F0FULL) != 0) {
    transform |= FlipTransform;
    our_king = ReverseBitsInBytes(our_king);
  }
  
  // If there are any pawns, only horizontal flip is valid
  if (pos.pieces(PAWN) != 0) {
    return transform;
  }
  
  // Mirror vertically if king on top half
  if ((our_king & 0xFFFFFFFF00000000ULL) != 0) {
    transform |= MirrorTransform;
    our_king = ReverseBytesInBytes(our_king);
  }
  
  // Our king is now in bottom right quadrant
  // Transpose for king in top right triangle, or if on diagonal use comparison
  if ((our_king & 0xE0C08000ULL) != 0) {
    transform |= TransposeTransform;
  } else if ((our_king & 0x10204080ULL) != 0) {
    // Compare all pieces, then ours, then each piece type to choose best transform
    auto outcome = CompareTransposing(pos.pieces(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(pos.pieces(us), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(pos.pieces(KING), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(pos.pieces(QUEEN), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(pos.pieces(ROOK), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(pos.pieces(KNIGHT), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(pos.pieces(BISHOP), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
  }
  
  return transform;
}

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

bool IsHectopliesFormat(MetalFishNN::NetworkFormat::InputFormat input_format) {
  using IF = MetalFishNN::NetworkFormat;
  return input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2 ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}

bool IsCanonicalArmageddonFormat(MetalFishNN::NetworkFormat::InputFormat input_format) {
  using IF = MetalFishNN::NetworkFormat;
  return input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}

int TransformForPosition(MetalFishNN::NetworkFormat::InputFormat input_format,
                         const std::vector<Position>& history) {
  if (!IsCanonicalFormat(input_format) || history.empty()) {
    return 0;
  }
  const Position& pos = history.back();
  Color us = pos.side_to_move();
  return ChooseTransform(pos, us);
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
  
  // Get current position and side to move
  const Position& current_pos = position_history.back();
  Color us = current_pos.side_to_move();
  Color them = ~us;
  
  // Determine if we should use canonicalization
  int transform = NoTransform;
  bool stop_early = IsCanonicalFormat(input_format);
  bool skip_non_repeats = (input_format == MetalFishNN::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
                           input_format == MetalFishNN::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON);
  
  if (stop_early) {
    transform = ChooseTransform(current_pos, us);
  }
  
  // Auxiliary planes (8 planes starting at index 104)
  int aux_base = kAuxPlaneBase;
  
  // Fill castling and en passant auxiliary planes first
  {
    using IF = MetalFishNN::NetworkFormat;
    
    if (input_format == IF::INPUT_CLASSICAL_112_PLANE) {
      // Legacy format: full planes for castling rights (from our perspective)
      CastlingRights our_queenside = (us == WHITE ? WHITE_OOO : BLACK_OOO);
      CastlingRights our_kingside = (us == WHITE ? WHITE_OO : BLACK_OO);
      CastlingRights their_queenside = (them == WHITE ? WHITE_OOO : BLACK_OOO);
      CastlingRights their_kingside = (them == WHITE ? WHITE_OO : BLACK_OO);
      
      SetPlane(result[aux_base + 0], current_pos.can_castle(our_queenside) ? 1.0f : 0.0f);
      SetPlane(result[aux_base + 1], current_pos.can_castle(our_kingside) ? 1.0f : 0.0f);
      SetPlane(result[aux_base + 2], current_pos.can_castle(their_queenside) ? 1.0f : 0.0f);
      SetPlane(result[aux_base + 3], current_pos.can_castle(their_kingside) ? 1.0f : 0.0f);
    } else {
      // Modern format: rook positions for castling (for Chess960 support)
      // Note: MetalFish may not have FRC support yet, so this is simplified
      SetPlane(result[aux_base + 0], 0.0f);
      SetPlane(result[aux_base + 1], 0.0f);
      
      // Set bits for castling rook positions (from our perspective)
      // In standard chess, queenside rook on file A, kingside rook on file H
      // From our perspective: our rooks on rank 1, their rooks on rank 8
      if (us == WHITE) {
        if (current_pos.can_castle(WHITE_OOO)) {
          result[aux_base + 0][0] = 1.0f;  // a1 rook (our queenside)
        }
        if (current_pos.can_castle(WHITE_OO)) {
          result[aux_base + 1][7] = 1.0f;  // h1 rook (our kingside)
        }
        if (current_pos.can_castle(BLACK_OOO)) {
          result[aux_base + 0][56] = 1.0f;  // a8 rook (their queenside)
        }
        if (current_pos.can_castle(BLACK_OO)) {
          result[aux_base + 1][63] = 1.0f;  // h8 rook (their kingside)
        }
      } else {
        // Black's perspective: flip the board
        if (current_pos.can_castle(BLACK_OOO)) {
          result[aux_base + 0][0] = 1.0f;  // a8 rook becomes a1 from black's view
        }
        if (current_pos.can_castle(BLACK_OO)) {
          result[aux_base + 1][7] = 1.0f;  // h8 rook becomes h1 from black's view
        }
        if (current_pos.can_castle(WHITE_OOO)) {
          result[aux_base + 0][56] = 1.0f;  // a1 rook becomes a8 from black's view
        }
        if (current_pos.can_castle(WHITE_OO)) {
          result[aux_base + 1][63] = 1.0f;  // h1 rook becomes h8 from black's view
        }
      }
    }
    
    // Plane 4: En passant or side to move
    if (IsCanonicalFormat(input_format)) {
      Square ep_sq = current_pos.ep_square();
      SetPlane(result[aux_base + 4], 0.0f);
      if (ep_sq != SQ_NONE) {
        result[aux_base + 4][ep_sq] = 1.0f;
      }
    } else {
      SetPlane(result[aux_base + 4], us == BLACK ? 1.0f : 0.0f);
    }
    
    // Plane 5: Rule50 counter
    float rule50_value = IsHectopliesFormat(input_format) ? 
        (current_pos.rule50_count() / 100.0f) : 
        static_cast<float>(current_pos.rule50_count());
    SetPlane(result[aux_base + 5], rule50_value);
    
    // Plane 6: Armageddon side to move (or zeros)
    if (IsCanonicalArmageddonFormat(input_format)) {
      SetPlane(result[aux_base + 6], us == BLACK ? 1.0f : 0.0f);
    } else {
      SetPlane(result[aux_base + 6], 0.0f);
    }
    
    // Plane 7: All ones (helps NN detect board edges)
    SetPlane(result[aux_base + 7], 1.0f);
  }
  
  // Encode position history (up to 8 positions, 13 planes each)
  int initial_castling = current_pos.can_castle(ANY_CASTLING) ? -1 : 0;
  bool flip = false;
  int history_size = std::min(history_planes, kMoveHistory);
  int actual_history = static_cast<int>(position_history.size());
  
  for (int i = 0; i < history_size; ++i) {
    // Calculate history index
    int history_idx = actual_history - 1 - i;
    
    // Check if we should break early for canonical formats
    if (stop_early && history_idx < actual_history - 1) {
      const Position& check_pos = position_history[history_idx >= 0 ? history_idx : 0];
      
      // Break if castling changed
      int cur_castling = check_pos.can_castle(ANY_CASTLING) ? 1 : 0;
      if (initial_castling >= 0 && cur_castling != initial_castling) break;
      
      // Break if en passant and not current position
      if (check_pos.ep_square() != SQ_NONE) break;
    }
    
    // Check if we should skip this position for fill_empty_history
    if (fill_empty_history == FillEmptyHistory::NO && history_idx < -1) {
      break;
    }
    if (fill_empty_history == FillEmptyHistory::NO && history_idx == -1) {
      const Position& check_pos = position_history[0];
      if (check_pos.ep_square() == SQ_NONE) break;
    }
    
    // Get position (use oldest if history_idx < 0 for fill_empty_history)
    const Position& pos = position_history[history_idx >= 0 ? history_idx : 0];
    
    // Check repetitions for v2 canonicalization
    if (skip_non_repeats && i > 0) {
      // Simplified: we don't have repetition tracking yet
      // In full implementation, check if position repeats
      if (pos.rule50_count() == 0) break;
    }
    
    int base = i * kPlanesPerBoard;
    
    // Get piece bitboards from perspective of current side to move
    Color perspective_us = flip ? them : us;
    Color perspective_them = flip ? us : them;
    
    uint64_t our_pieces[6] = {
      GetPieceBitboard(pos, PAWN, perspective_us),
      GetPieceBitboard(pos, KNIGHT, perspective_us),
      GetPieceBitboard(pos, BISHOP, perspective_us),
      GetPieceBitboard(pos, ROOK, perspective_us),
      GetPieceBitboard(pos, QUEEN, perspective_us),
      GetPieceBitboard(pos, KING, perspective_us)
    };
    
    uint64_t their_pieces[6] = {
      GetPieceBitboard(pos, PAWN, perspective_them),
      GetPieceBitboard(pos, KNIGHT, perspective_them),
      GetPieceBitboard(pos, BISHOP, perspective_them),
      GetPieceBitboard(pos, ROOK, perspective_them),
      GetPieceBitboard(pos, QUEEN, perspective_them),
      GetPieceBitboard(pos, KING, perspective_them)
    };
    
    // Fill planes for our pieces
    for (int piece = 0; piece < 6; ++piece) {
      FillPlaneFromBitboard(result[base + piece], our_pieces[piece]);
    }
    
    // Fill planes for their pieces
    for (int piece = 0; piece < 6; ++piece) {
      FillPlaneFromBitboard(result[base + 6 + piece], their_pieces[piece]);
    }
    
    // Repetition plane (simplified - always 0 for now)
    SetPlane(result[base + 12], 0.0f);
    
    // Handle en passant for filled history
    if (history_idx < 0 && pos.ep_square() != SQ_NONE) {
      Square ep_sq = pos.ep_square();
      int ep_idx = static_cast<int>(ep_sq);
      
      // Undo the pawn move for en passant
      if (ep_idx < 8) {  // "Us" pawn
        uint64_t mask = ((0x0000000000000100ULL - 0x0000000001000000ULL) << ep_idx);
        FillPlaneFromBitboard(result[base + 0], our_pieces[0] + mask);
      } else if (ep_idx >= 56) {  // "Them" pawn
        uint64_t mask = ((0x0001000000000000ULL - 0x0000000100000000ULL) << (ep_idx - 56));
        FillPlaneFromBitboard(result[base + 6], their_pieces[0] + mask);
      }
    }
    
    // Alternate perspective for next position
    if (history_idx > 0) flip = !flip;
    
    // Stop early if rule50 was reset (capture or pawn move)
    if (stop_early && pos.rule50_count() == 0) break;
  }
  
  // Apply transform to all planes if canonicalization is enabled
  if (transform != NoTransform) {
    // Transform piece planes and en passant plane
    for (int i = 0; i <= aux_base + 4; ++i) {
      // Convert plane to bitboard
      uint64_t bitboard = 0;
      for (int sq = 0; sq < 64; ++sq) {
        if (result[i][sq] > 0.5f) {
          bitboard |= (1ULL << sq);
        }
      }
      
      // Skip empty and full planes
      if (bitboard == 0 || bitboard == ~0ULL) continue;
      
      // Apply transform
      uint64_t transformed = ApplyTransform(bitboard, transform);
      
      // Convert back to plane
      FillPlaneFromBitboard(result[i], transformed);
    }
  }
  
  if (transform_out) {
    *transform_out = transform;
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
  
  // Castling rights from side-to-move perspective
  CastlingRights our_oo = us == WHITE ? WHITE_OO : BLACK_OO;
  CastlingRights our_ooo = us == WHITE ? WHITE_OOO : BLACK_OOO;
  CastlingRights their_oo = us == WHITE ? BLACK_OO : WHITE_OO;
  CastlingRights their_ooo = us == WHITE ? BLACK_OOO : WHITE_OOO;
  
  SetPlane(result[aux_base + 0], pos.can_castle(our_oo) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 1], pos.can_castle(our_ooo) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 2], pos.can_castle(their_oo) ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 3], pos.can_castle(their_ooo) ? 1.0f : 0.0f);
  
  SetPlane(result[aux_base + 4], us == BLACK ? 1.0f : 0.0f);
  SetPlane(result[aux_base + 5], static_cast<float>(pos.rule50_count()));
  SetPlane(result[aux_base + 6], 0.0f);
  SetPlane(result[aux_base + 7], 1.0f);
  
  return result;
}

}  // namespace NN
}  // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "encoder.h"

#include <cstring>
#include <limits>

namespace MetalFish {
namespace NN {

namespace {

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

inline uint64_t ReverseBitsInBytes(uint64_t v) {
  v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
  v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
  v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
  return v;
}

inline uint64_t ReverseBytesInBytes(uint64_t v) {
  v = (v & 0x00000000FFFFFFFF) << 32 | (v & 0xFFFFFFFF00000000) >> 32;
  v = (v & 0x0000FFFF0000FFFF) << 16 | (v & 0xFFFF0000FFFF0000) >> 16;
  v = (v & 0x00FF00FF00FF00FF) << 8 | (v & 0xFF00FF00FF00FF00) >> 8;
  return v;
}

inline uint64_t TransposeBitsInBytes(uint64_t v) {
  v = (v & 0xAA00AA00AA00AA00ULL) >> 9 | (v & 0x0055005500550055ULL) << 9 |
      (v & 0x55AA55AA55AA55AAULL);
  v = (v & 0xCCCC0000CCCC0000ULL) >> 18 | (v & 0x0000333300003333ULL) << 18 |
      (v & 0x3333CCCC3333CCCCULL);
  v = (v & 0xF0F0F0F000000000ULL) >> 36 | (v & 0x000000000F0F0F0FULL) << 36 |
      (v & 0x0F0F0F0FF0F0F0F0ULL);
  return v;
}

inline uint64_t ApplyTransform(uint64_t bitboard, int transform) {
  if (bitboard == 0 || bitboard == ~0ULL)
    return bitboard;

  uint64_t v = bitboard;
  if ((transform & kFlipTransform) != 0) {
    v = ReverseBitsInBytes(v);
  }
  if ((transform & kMirrorTransform) != 0) {
    v = ReverseBytesInBytes(v);
  }
  if ((transform & kTransposeTransform) != 0) {
    v = TransposeBitsInBytes(v);
  }
  return v;
}

int CompareTransposing(uint64_t board, int initial_transform) {
  uint64_t value = board;
  if ((initial_transform & kFlipTransform) != 0) {
    value = ReverseBitsInBytes(value);
  }
  if ((initial_transform & kMirrorTransform) != 0) {
    value = ReverseBytesInBytes(value);
  }
  auto alternative = TransposeBitsInBytes(value);
  if (value < alternative)
    return -1;
  if (value > alternative)
    return 1;
  return 0;
}

int ChooseTransform(const Position &pos, Color us) {
  if (pos.can_castle(ANY_CASTLING)) {
    return kNoTransform;
  }

  uint64_t our_king = pos.pieces(us, KING);
  int transform = kNoTransform;

  if ((our_king & 0x0F0F0F0F0F0F0F0FULL) != 0) {
    transform |= kFlipTransform;
    our_king = ReverseBitsInBytes(our_king);
  }

  if (pos.pieces(PAWN) != 0) {
    return transform;
  }

  if ((our_king & 0xFFFFFFFF00000000ULL) != 0) {
    transform |= kMirrorTransform;
    our_king = ReverseBytesInBytes(our_king);
  }

  if ((our_king & 0xE0C08000ULL) != 0) {
    transform |= kTransposeTransform;
  } else if ((our_king & 0x10204080ULL) != 0) {
    auto outcome = CompareTransposing(pos.pieces(), transform);
    if (outcome == -1)
      return transform;
    if (outcome == 1)
      return transform | kTransposeTransform;
    outcome = CompareTransposing(pos.pieces(us), transform);
    if (outcome == -1)
      return transform;
    if (outcome == 1)
      return transform | kTransposeTransform;
    outcome = CompareTransposing(pos.pieces(KING), transform);
    if (outcome == -1)
      return transform;
    if (outcome == 1)
      return transform | kTransposeTransform;
    outcome = CompareTransposing(pos.pieces(QUEEN), transform);
    if (outcome == -1)
      return transform;
    if (outcome == 1)
      return transform | kTransposeTransform;
    outcome = CompareTransposing(pos.pieces(ROOK), transform);
    if (outcome == -1)
      return transform;
    if (outcome == 1)
      return transform | kTransposeTransform;
    outcome = CompareTransposing(pos.pieces(KNIGHT), transform);
    if (outcome == -1)
      return transform;
    if (outcome == 1)
      return transform | kTransposeTransform;
    outcome = CompareTransposing(pos.pieces(BISHOP), transform);
    if (outcome == -1)
      return transform;
    if (outcome == 1)
      return transform | kTransposeTransform;
  }

  return transform;
}

uint64_t GetPieceBitboard(const Position &pos, PieceType pt, Color c) {
  Bitboard bb = pos.pieces(c, pt);
  return bb;
}

void FillPlaneFromBitboard(std::array<float, 64> &plane, uint64_t bitboard) {
  plane.fill(0.0f);
  while (bitboard) {
    const int sq = GetLowestBit(bitboard);
    plane[sq] = 1.0f;
    bitboard &= bitboard - 1;
  }
}

void SetBitsInZeroedPlane(std::array<float, 64> &plane, uint64_t bitboard) {
  while (bitboard) {
    const int sq = GetLowestBit(bitboard);
    plane[sq] = 1.0f;
    bitboard &= bitboard - 1;
  }
}

void SetPlane(std::array<float, 64> &plane, float value) {
  plane.fill(value);
}

} // namespace

bool IsCanonicalFormat(MetalFishNN::NetworkFormat::InputFormat input_format) {
  using IF = MetalFishNN::NetworkFormat;
  return input_format == IF::INPUT_112_WITH_CANONICALIZATION ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
         input_format ==
             IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2 ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}

bool IsHectopliesFormat(MetalFishNN::NetworkFormat::InputFormat input_format) {
  using IF = MetalFishNN::NetworkFormat;
  return input_format == IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
         input_format ==
             IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2 ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}

bool IsCanonicalArmageddonFormat(
    MetalFishNN::NetworkFormat::InputFormat input_format) {
  using IF = MetalFishNN::NetworkFormat;
  return input_format ==
             IF::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == IF::INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}

bool IsStartPosition(const Position &pos) {
  constexpr Bitboard kWhitePawns = 0x000000000000FF00ULL;
  constexpr Bitboard kBlackPawns = 0x00FF000000000000ULL;
  constexpr Bitboard kWhiteKnights = 0x0000000000000042ULL;
  constexpr Bitboard kBlackKnights = 0x4200000000000000ULL;
  constexpr Bitboard kWhiteBishops = 0x0000000000000024ULL;
  constexpr Bitboard kBlackBishops = 0x2400000000000000ULL;
  constexpr Bitboard kWhiteRooks = 0x0000000000000081ULL;
  constexpr Bitboard kBlackRooks = 0x8100000000000000ULL;
  constexpr Bitboard kWhiteQueen = 0x0000000000000008ULL;
  constexpr Bitboard kBlackQueen = 0x0800000000000000ULL;
  constexpr Bitboard kWhiteKing = 0x0000000000000010ULL;
  constexpr Bitboard kBlackKing = 0x1000000000000000ULL;

  return pos.side_to_move() == WHITE && pos.game_ply() == 0 &&
         pos.rule50_count() == 0 && pos.ep_square() == SQ_NONE &&
         pos.can_castle(WHITE_OO) && pos.can_castle(WHITE_OOO) &&
         pos.can_castle(BLACK_OO) && pos.can_castle(BLACK_OOO) &&
         pos.pieces(WHITE, PAWN) == kWhitePawns &&
         pos.pieces(BLACK, PAWN) == kBlackPawns &&
         pos.pieces(WHITE, KNIGHT) == kWhiteKnights &&
         pos.pieces(BLACK, KNIGHT) == kBlackKnights &&
         pos.pieces(WHITE, BISHOP) == kWhiteBishops &&
         pos.pieces(BLACK, BISHOP) == kBlackBishops &&
         pos.pieces(WHITE, ROOK) == kWhiteRooks &&
         pos.pieces(BLACK, ROOK) == kBlackRooks &&
         pos.pieces(WHITE, QUEEN) == kWhiteQueen &&
         pos.pieces(BLACK, QUEEN) == kBlackQueen &&
         pos.pieces(WHITE, KING) == kWhiteKing &&
         pos.pieces(BLACK, KING) == kBlackKing;
}

int TransformForPosition(MetalFishNN::NetworkFormat::InputFormat input_format,
                         std::span<const Position *const> history) {
  if (!IsCanonicalFormat(input_format) || history.empty()) {
    return 0;
  }
  const Position &pos = *history.back();
  Color us = pos.side_to_move();
  return ChooseTransform(pos, us);
}

void EncodePositionForNN(MetalFishNN::NetworkFormat::InputFormat input_format,
                         std::span<const Position *const> position_history,
                         int history_planes,
                         FillEmptyHistory fill_empty_history,
                         InputPlanes &result, int *transform_out) {

  result = {};

  if (position_history.empty()) {
    return;
  }

  const Position &current_pos = *position_history.back();
  Color us = current_pos.side_to_move();
  Color them = ~us;

  int transform = kNoTransform;
  bool stop_early = IsCanonicalFormat(input_format);
  bool skip_non_repeats =
      (input_format ==
           MetalFishNN::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
       input_format == MetalFishNN::NetworkFormat::
                           INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON);

  if (stop_early) {
    transform = ChooseTransform(current_pos, us);
  }

  int aux_base = kAuxPlaneBase;

  {
    using IF = MetalFishNN::NetworkFormat;

    if (input_format == IF::INPUT_CLASSICAL_112_PLANE) {
      CastlingRights our_queenside = (us == WHITE ? WHITE_OOO : BLACK_OOO);
      CastlingRights our_kingside = (us == WHITE ? WHITE_OO : BLACK_OO);
      CastlingRights their_queenside = (them == WHITE ? WHITE_OOO : BLACK_OOO);
      CastlingRights their_kingside = (them == WHITE ? WHITE_OO : BLACK_OO);

      SetPlane(result[aux_base + 0],
               current_pos.can_castle(our_queenside) ? 1.0f : 0.0f);
      SetPlane(result[aux_base + 1],
               current_pos.can_castle(our_kingside) ? 1.0f : 0.0f);
      SetPlane(result[aux_base + 2],
               current_pos.can_castle(their_queenside) ? 1.0f : 0.0f);
      SetPlane(result[aux_base + 3],
               current_pos.can_castle(their_kingside) ? 1.0f : 0.0f);
    } else {
      SetPlane(result[aux_base + 0], 0.0f);
      SetPlane(result[aux_base + 1], 0.0f);

      if (us == WHITE) {
        if (current_pos.can_castle(WHITE_OOO))
          result[aux_base + 0][0] = 1.0f;
        if (current_pos.can_castle(WHITE_OO))
          result[aux_base + 1][7] = 1.0f;
        if (current_pos.can_castle(BLACK_OOO))
          result[aux_base + 0][56] = 1.0f;
        if (current_pos.can_castle(BLACK_OO))
          result[aux_base + 1][63] = 1.0f;
      } else {
        if (current_pos.can_castle(BLACK_OOO))
          result[aux_base + 0][0] = 1.0f;
        if (current_pos.can_castle(BLACK_OO))
          result[aux_base + 1][7] = 1.0f;
        if (current_pos.can_castle(WHITE_OOO))
          result[aux_base + 0][56] = 1.0f;
        if (current_pos.can_castle(WHITE_OO))
          result[aux_base + 1][63] = 1.0f;
      }
    }

    if (IsCanonicalFormat(input_format)) {
      Square ep_sq = current_pos.ep_square();
      SetPlane(result[aux_base + 4], 0.0f);
      if (ep_sq != SQ_NONE) {
        result[aux_base + 4][ep_sq] = 1.0f;
      }
    } else {
      SetPlane(result[aux_base + 4], us == BLACK ? 1.0f : 0.0f);
    }

    float rule50_value = IsHectopliesFormat(input_format)
                             ? (current_pos.rule50_count() / 100.0f)
                             : static_cast<float>(current_pos.rule50_count());
    SetPlane(result[aux_base + 5], rule50_value);

    if (IsCanonicalArmageddonFormat(input_format)) {
      SetPlane(result[aux_base + 6], us == BLACK ? 1.0f : 0.0f);
    } else {
      SetPlane(result[aux_base + 6], 0.0f);
    }

    SetPlane(result[aux_base + 7], 1.0f);
  }

  auto castling_mask_for_us = [us](const Position &p) -> uint8_t {
    uint8_t mask = 0;
    if (us == WHITE) {
      if (p.can_castle(WHITE_OO))
        mask |= 1 << 0;
      if (p.can_castle(WHITE_OOO))
        mask |= 1 << 1;
      if (p.can_castle(BLACK_OO))
        mask |= 1 << 2;
      if (p.can_castle(BLACK_OOO))
        mask |= 1 << 3;
    } else {
      if (p.can_castle(BLACK_OO))
        mask |= 1 << 0;
      if (p.can_castle(BLACK_OOO))
        mask |= 1 << 1;
      if (p.can_castle(WHITE_OO))
        mask |= 1 << 2;
      if (p.can_castle(WHITE_OOO))
        mask |= 1 << 3;
    }
    return mask;
  };

  const uint8_t root_castling_mask = castling_mask_for_us(current_pos);
  int history_size = std::min(history_planes, kMoveHistory);
  int actual_history = static_cast<int>(position_history.size());

  for (int i = 0, history_idx = actual_history - 1; i < history_size;
       ++i, --history_idx) {
    const Position &pos = *position_history[history_idx >= 0 ? history_idx : 0];

    // If en passant is possible we can infer one previous move; otherwise stop.
    if (fill_empty_history == FillEmptyHistory::NO &&
        (history_idx < -1 ||
         (history_idx == -1 && pos.ep_square() == SQ_NONE))) {
      break;
    }

    // For FEN-only history, don't synthesize through the start position.
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
        IsStartPosition(pos)) {
      break;
    }

    // Castling or non-current en passant changes cannot be repeated.
    if (stop_early && history_idx < actual_history - 1) {
      if (castling_mask_for_us(pos) != root_castling_mask)
        break;
      if (pos.ep_square() != SQ_NONE)
        break;
    }

    const bool has_repetition =
        pos.is_repetition(std::numeric_limits<int>::max());

    // Canonical v2: keep plane index fixed when skipping non-repeats.
    if (skip_non_repeats && i > 0 && !has_repetition) {
      if (pos.rule50_count() == 0)
        break;
      --i;
      continue;
    }

    int base = i * kPlanesPerBoard;

    uint64_t our_pieces[6] = {
        GetPieceBitboard(pos, PAWN, us),   GetPieceBitboard(pos, KNIGHT, us),
        GetPieceBitboard(pos, BISHOP, us), GetPieceBitboard(pos, ROOK, us),
        GetPieceBitboard(pos, QUEEN, us),  GetPieceBitboard(pos, KING, us)};

    uint64_t their_pieces[6] = {GetPieceBitboard(pos, PAWN, them),
                                GetPieceBitboard(pos, KNIGHT, them),
                                GetPieceBitboard(pos, BISHOP, them),
                                GetPieceBitboard(pos, ROOK, them),
                                GetPieceBitboard(pos, QUEEN, them),
                                GetPieceBitboard(pos, KING, them)};

    if (us == BLACK) {
      for (int piece = 0; piece < 6; ++piece) {
        our_pieces[piece] = ReverseBytesInBytes(our_pieces[piece]);
        their_pieces[piece] = ReverseBytesInBytes(their_pieces[piece]);
      }
    }

    for (int piece = 0; piece < 6; ++piece) {
      SetBitsInZeroedPlane(result[base + piece], our_pieces[piece]);
    }
    for (int piece = 0; piece < 6; ++piece) {
      SetBitsInZeroedPlane(result[base + 6 + piece], their_pieces[piece]);
    }

    SetPlane(result[base + 12], has_repetition ? 1.0f : 0.0f);

    if (history_idx < 0 && pos.ep_square() != SQ_NONE) {
      Square ep_sq = pos.ep_square();
      int ep_idx = static_cast<int>(ep_sq);
      if (ep_idx < 8) {
        uint64_t mask =
            ((0x0000000000000100ULL - 0x0000000001000000ULL) << ep_idx);
        FillPlaneFromBitboard(result[base + 0], our_pieces[0] + mask);
      } else if (ep_idx >= 56) {
        uint64_t mask =
            ((0x0001000000000000ULL - 0x0000000100000000ULL) << (ep_idx - 56));
        FillPlaneFromBitboard(result[base + 6], their_pieces[0] + mask);
      }
    }

    if (stop_early && pos.rule50_count() == 0)
      break;
  }

  if (transform != kNoTransform) {
    for (int i = 0; i <= aux_base + 4; ++i) {
      uint64_t bitboard = 0;
      for (int sq = 0; sq < 64; ++sq) {
        if (result[i][sq] > 0.5f) {
          bitboard |= (1ULL << sq);
        }
      }

      if (bitboard == 0 || bitboard == ~0ULL)
        continue;

      uint64_t transformed = ApplyTransform(bitboard, transform);
      FillPlaneFromBitboard(result[i], transformed);
    }
  }

  if (transform_out) {
    *transform_out = transform;
  }
}

InputPlanes
EncodePositionForNN(MetalFishNN::NetworkFormat::InputFormat input_format,
                    std::span<const Position *const> position_history,
                    int history_planes, FillEmptyHistory fill_empty_history,
                    int *transform_out) {
  InputPlanes result;
  EncodePositionForNN(input_format, position_history, history_planes,
                      fill_empty_history, result, transform_out);
  return result;
}

InputPlanes
EncodePositionForNN(const Position &pos,
                    MetalFishNN::NetworkFormat::InputFormat input_format) {
  std::array<const Position *, 1> history = {&pos};
  return EncodePositionForNN(input_format, history, kMoveHistory,
                             FillEmptyHistory::FEN_ONLY, nullptr);
}

} // namespace NN
} // namespace MetalFish

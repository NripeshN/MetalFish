/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

// Definition of input features Full_Threats of NNUE evaluation function

#ifndef NNUE_FEATURES_FULL_THREATS_INCLUDED
#define NNUE_FEATURES_FULL_THREATS_INCLUDED

#include "../nnue_common.h"
#include "core/misc.h"
#include "core/types.h"

namespace MetalFish {
class Position;
}

namespace MetalFish::Eval::NNUE::Features {

// Pawn diagonal threats only target knights and rooks (pawn-pawn relationships
// are handled by the PP_3Wide feature set), so pawns have 4 valid targets.
static constexpr int numValidTargets[PIECE_NB] = {0, 4, 10, 8, 8, 10, 0, 0,
                                                  0, 4, 10, 8, 8, 10, 0, 0};

class FullThreats {
public:
  // Hash value embedded in the evaluation file
  static constexpr u32 HashValue = 0x2e6b9d04u;

  // Number of feature dimensions
  static constexpr IndexType Dimensions = 59808;

  // clang-format off
    // Orient a square according to perspective (rotates by 180 for black)
    static constexpr i8 OrientTBL[SQUARE_NB] = {
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
        SQ_A1, SQ_A1, SQ_A1, SQ_A1, SQ_H1, SQ_H1, SQ_H1, SQ_H1,
    };

    static constexpr int map[PIECE_TYPE_NB-2][PIECE_TYPE_NB-2] = {
      {-1,  0, -1,  1, -1, -1},
      { 0,  1,  2,  3,  4, -1},
      { 0,  1,  2,  3, -1, -1},
      { 0,  1,  2,  3, -1, -1},
      { 0,  1,  2,  3,  4, -1},
      {-1, -1, -1, -1, -1, -1}
    };
  // clang-format on

  // Maximum number of simultaneously active features.
  using IndexList = ValueList<u16, 256>;
  using DiffType = DirtyThreats;

  static IndexType make_index(Color perspective, Piece attkr, Square from,
                              Square to, Piece attkd, Square ksq);

  // Get a list of indices for active features
  static void append_active_indices(Color perspective, const Position &pos,
                                    IndexList &active);

  // Get a list of indices for recently changed features
  static void
  append_changed_indices(Color perspective, Square ksq, const DiffType &diff,
                         IndexList &removed, IndexList &added,
                         const ThreatWeightType *prefetchBase = nullptr,
                         IndexType prefetchStride = 0);

  static void
  append_changed_indices_both(Square white_ksq, Square black_ksq,
                              const DiffType &diff, IndexList &white_removed,
                              IndexList &white_added, IndexList &black_removed,
                              IndexList &black_added,
                              const ThreatWeightType *prefetchBase = nullptr,
                              IndexType prefetchStride = 0);
};

} // namespace MetalFish::Eval::NNUE::Features

#endif // #ifndef NNUE_FEATURES_FULL_THREATS_INCLUDED

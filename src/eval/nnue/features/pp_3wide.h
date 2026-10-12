/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef NNUE_FEATURES_PP_3WIDE_INCLUDED
#define NNUE_FEATURES_PP_3WIDE_INCLUDED

#include "../nnue_common.h"
#include "core/misc.h"
#include "core/types.h"

namespace MetalFish {
class Position;
}

namespace MetalFish::Eval::NNUE::Features {

class PP_3Wide {
public:
  static constexpr u32 HashValue = 0x86f2b1ddu;

  static constexpr IndexType PawnIds = COLOR_NB * 48;
  static constexpr IndexType Dimensions = PawnIds * (PawnIds - 1) / 2;

  // Pawn pair feature indices are concatenated to threats, so this must equal
  // ThreatFeatureSet::Dimensions; see nnue_feature_transformer.h
  static constexpr IndexType IndexBase = 59808;
  using IndexList = ValueList<u16, 256>;
  using DiffType = DirtyPawnPairs;

  static IndexType make_index(Color perspective, Color color, Square from,
                              Square to, Color pairedColor, Square ksq);

  static void append_active_indices(Color perspective, const Position &pos,
                                    IndexList &active);

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

#endif // #ifndef NNUE_FEATURES_PP_3WIDE_INCLUDED

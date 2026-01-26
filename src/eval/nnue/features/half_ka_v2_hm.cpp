/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file half_ka_v2_hm.cpp
 * @brief MetalFish source file.
 */

  Licensed under GPL-3.0
*/

// Definition of input features HalfKAv2_hm of NNUE evaluation function

#include "half_ka_v2_hm.h"

#include "../nnue_common.h"
#include "core/bitboard.h"
#include "core/position.h"
#include "core/types.h"

namespace MetalFish::Eval::NNUE::Features {

// Index of a feature for a given king position and another piece on some square

IndexType HalfKAv2_hm::make_index(Color perspective, Square s, Piece pc,
                                  Square ksq) {
  const IndexType flip = 56 * perspective;
  return (IndexType(s) ^ OrientTBL[ksq] ^ flip) +
         PieceSquareIndex[perspective][pc] + KingBuckets[int(ksq) ^ flip];
}

// Get a list of indices for active features

void HalfKAv2_hm::append_active_indices(Color perspective, const Position &pos,
                                        IndexList &active) {
  Square ksq = pos.square<KING>(perspective);
  Bitboard bb = pos.pieces();
  while (bb) {
    Square s = pop_lsb(bb);
    active.push_back(make_index(perspective, s, pos.piece_on(s), ksq));
  }
}

// Get a list of indices for recently changed features

void HalfKAv2_hm::append_changed_indices(Color perspective, Square ksq,
                                         const DiffType &diff,
                                         IndexList &removed, IndexList &added) {
  removed.push_back(make_index(perspective, diff.from, diff.pc, ksq));
  if (diff.to != SQ_NONE)
    added.push_back(make_index(perspective, diff.to, diff.pc, ksq));

  if (diff.remove_sq != SQ_NONE)
    removed.push_back(
        make_index(perspective, diff.remove_sq, diff.remove_pc, ksq));

  if (diff.add_sq != SQ_NONE)
    added.push_back(make_index(perspective, diff.add_sq, diff.add_pc, ksq));
}

bool HalfKAv2_hm::requires_refresh(const DiffType &diff, Color perspective) {
  return diff.pc == make_piece(perspective, KING);
}

} // namespace MetalFish::Eval::NNUE::Features
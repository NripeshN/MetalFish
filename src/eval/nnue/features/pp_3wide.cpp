/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "pp_3wide.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "../nnue_common.h"
#include "core/bitboard.h"
#include "core/misc.h"
#include "core/position.h"
#include "core/types.h"
#include "full_threats.h"

namespace MetalFish::Eval::NNUE::Features {

constexpr IndexType make_pawn_id(Color color, Square square) {
  assert(square >= SQ_A2 && square <= SQ_H7);
  return 48 * int(color) + square - SQ_A2;
}

#ifdef USE_AVX512ICL
static inline __m256i pp_idx_epi16(__m256i a, __m256i b) {
  const __m256i hi = _mm256_max_epu16(a, b);
  const __m256i lo = _mm256_min_epu16(a, b);
  const __m256i prod =
      _mm256_mullo_epi16(hi, _mm256_sub_epi16(hi, _mm256_set1_epi16(1)));
  return _mm256_add_epi16(_mm256_add_epi16(_mm256_srli_epi16(prod, 1), lo),
                          _mm256_set1_epi16(i16(PP_3Wide::IndexBase)));
}
#endif

sf_always_inline IndexType PP_3Wide::make_index(Color perspective, Color color,
                                                Square from, Square to,
                                                Color pairedColor, Square ksq) {
  const i8 orientation = FullThreats::OrientTBL[ksq] ^ (56 * perspective);
  unsigned from_oriented = u8(from) ^ orientation;
  unsigned to_oriented = u8(to) ^ orientation;

  Color color_oriented = Color(color ^ perspective);
  Color pairedColor_oriented = Color(pairedColor ^ perspective);

  assert(from_oriented >= SQ_A2 && from_oriented <= SQ_H7);
  assert(to_oriented >= SQ_A2 && to_oriented <= SQ_H7);

  const IndexType idA = make_pawn_id(color_oriented, Square(from_oriented));
  const IndexType idB = make_pawn_id(pairedColor_oriented, Square(to_oriented));
  const IndexType hi = std::max(idA, idB);
  const IndexType lo = std::min(idA, idB);

  return hi * (hi - 1) / 2 + lo + IndexBase;
}

void PP_3Wide::append_active_indices(Color perspective, const Position &pos,
                                     IndexList &active) {
  const Square ksq = pos.square<KING>(perspective);
  const Bitboard white = pos.pieces(WHITE, PAWN);
  const Bitboard black = pos.pieces(BLACK, PAWN);

  Bitboard bb = white;
  while (bb) {
    Square from = pop_lsb(bb);
    const Bitboard band = pawn_pair_bb(from);
    for (Bitboard ww = band & bb; ww;)
      active.push_back(
          make_index(perspective, WHITE, from, pop_lsb(ww), WHITE, ksq));
    for (Bitboard wb = band & black; wb;)
      active.push_back(
          make_index(perspective, WHITE, from, pop_lsb(wb), BLACK, ksq));
  }

  bb = black;
  while (bb) {
    Square from = pop_lsb(bb);
    const Bitboard band = pawn_pair_bb(from);
    for (Bitboard bbk = band & bb; bbk;)
      active.push_back(
          make_index(perspective, BLACK, from, pop_lsb(bbk), BLACK, ksq));
  }
}

void PP_3Wide::append_changed_indices(
    Color perspective, Square ksq, const DiffType &diff, IndexList &removed,
    IndexList &added, [[maybe_unused]] const ThreatWeightType *prefetchBase,
    [[maybe_unused]] IndexType prefetchStride) {

  const Bitboard whiteBefore = diff.before[WHITE];
  const Bitboard blackBefore = diff.before[BLACK];
  const Bitboard whiteAfter = diff.after[WHITE];
  const Bitboard blackAfter = diff.after[BLACK];

  if (whiteBefore == whiteAfter && blackBefore == blackAfter)
    return;

#ifdef USE_AVX512ICL
  const u8 orientation = u8(FullThreats::OrientTBL[ksq]) ^ u8(56 * perspective);
  const __m512i iota = AllSquares;
  const __m512i adjusted =
      _mm512_sub_epi8(_mm512_xor_si512(iota, _mm512_set1_epi8(orientation)),
                      _mm512_set1_epi8(8));

  auto generate = [&](Bitboard updatedW, Bitboard updatedB, Bitboard pawnsW,
                      Bitboard pawnsB, IndexList &out) {
    const Bitboard friendly = perspective == WHITE ? pawnsW : pawnsB;
    const Bitboard enemy = perspective == WHITE ? pawnsB : pawnsW;
    const __m512i ids = _mm512_mask_blend_epi8(
        friendly, _mm512_add_epi8(adjusted, _mm512_set1_epi8(48)), adjusted);

    const Bitboard unchanged = (pawnsW | pawnsB) & ~(updatedW | updatedB);
    for (Bitboard u = updatedW | updatedB; u;) {
      const Square a = pop_lsb(u);
      const Bitboard partners = pawn_pair_bb(a) & (unchanged | u);
      const int n = popcount(partners);
      if (!n)
        continue;

      const u16 colorOff = (enemy & a) ? 48 : 0;
      const u16 aId = u16(((u8(a) ^ orientation) - 8) + colorOff);
      const __m256i pids = _mm256_cvtepu8_epi16(
          _mm512_castsi512_si128(_mm512_maskz_compress_epi8(partners, ids)));
      const __m256i feats = pp_idx_epi16(_mm256_set1_epi16(aId), pids);

      u16 *w = out.make_space(n);
      _mm256_storeu_epi16(w, feats);
    }
  };
#else
  auto generate = [&](Bitboard updatedW, Bitboard updatedB, Bitboard pawnsW,
                      Bitboard pawnsB, IndexList &out) {
    auto push = [&](IndexType index) {
      if (prefetchBase)
        prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(
            reinterpret_cast<const void *>(
                reinterpret_cast<uintptr_t>(prefetchBase) +
                uintptr_t(index) * prefetchStride));
      out.push_back(index);
    };
    const Bitboard unchanged = (pawnsW | pawnsB) & ~(updatedW | updatedB);
    for (Bitboard u = updatedW | updatedB; u;) {
      const Square a = pop_lsb(u);
      const Bitboard mask = pawn_pair_bb(a) & (unchanged | u);
      const Color aCol = (pawnsB & a) ? BLACK : WHITE;
      for (Bitboard pb = pawnsB & mask; pb;)
        push(make_index(perspective, aCol, a, pop_lsb(pb), BLACK, ksq));
      for (Bitboard pw = pawnsW & mask; pw;)
        push(make_index(perspective, aCol, a, pop_lsb(pw), WHITE, ksq));
    }
  };
#endif

  generate(whiteAfter & ~whiteBefore, blackAfter & ~blackBefore, whiteAfter,
           blackAfter, added);
  generate(whiteBefore & ~whiteAfter, blackBefore & ~blackAfter, whiteBefore,
           blackBefore, removed);
}

void PP_3Wide::append_changed_indices_both(
    Square white_ksq, Square black_ksq, const DiffType &diff,
    IndexList &white_removed, IndexList &white_added, IndexList &black_removed,
    IndexList &black_added, const ThreatWeightType *prefetchBase,
    IndexType prefetchStride) {

#ifdef USE_AVX512ICL
  append_changed_indices(WHITE, white_ksq, diff, white_removed, white_added,
                         prefetchBase, prefetchStride);
  append_changed_indices(BLACK, black_ksq, diff, black_removed, black_added,
                         prefetchBase, prefetchStride);
#else
  const Bitboard whiteBefore = diff.before[WHITE];
  const Bitboard blackBefore = diff.before[BLACK];
  const Bitboard whiteAfter = diff.after[WHITE];
  const Bitboard blackAfter = diff.after[BLACK];

  if (whiteBefore == whiteAfter && blackBefore == blackAfter)
    return;

  auto generate = [&](Bitboard updatedW, Bitboard updatedB, Bitboard pawnsW,
                      Bitboard pawnsB, IndexList &white_out,
                      IndexList &black_out) {
    auto push = [&](Color color, Square from, Square to, Color pairedColor) {
      const IndexType white_index =
          make_index(WHITE, color, from, to, pairedColor, white_ksq);
      const IndexType black_index =
          make_index(BLACK, color, from, to, pairedColor, black_ksq);

      if (prefetchBase) {
        prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(
            reinterpret_cast<const void *>(
                reinterpret_cast<uintptr_t>(prefetchBase) +
                uintptr_t(white_index) * prefetchStride));
        prefetch<PrefetchRw::READ, PrefetchLoc::LOW>(
            reinterpret_cast<const void *>(
                reinterpret_cast<uintptr_t>(prefetchBase) +
                uintptr_t(black_index) * prefetchStride));
      }

      white_out.push_back(white_index);
      black_out.push_back(black_index);
    };

    const Bitboard unchanged = (pawnsW | pawnsB) & ~(updatedW | updatedB);
    for (Bitboard updated = updatedW | updatedB; updated;) {
      const Square from = pop_lsb(updated);
      const Bitboard mask = pawn_pair_bb(from) & (unchanged | updated);
      const Color color = (pawnsB & from) ? BLACK : WHITE;

      for (Bitboard paired = pawnsB & mask; paired;)
        push(color, from, pop_lsb(paired), BLACK);
      for (Bitboard paired = pawnsW & mask; paired;)
        push(color, from, pop_lsb(paired), WHITE);
    }
  };

  generate(whiteAfter & ~whiteBefore, blackAfter & ~blackBefore, whiteAfter,
           blackAfter, white_added, black_added);
  generate(whiteBefore & ~whiteAfter, blackBefore & ~blackAfter, whiteBefore,
           blackBefore, white_removed, black_removed);
#endif
}

} // namespace MetalFish::Eval::NNUE::Features

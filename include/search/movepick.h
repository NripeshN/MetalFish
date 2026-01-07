/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

  MovePicker - Sophisticated move ordering
  ========================================

  Orders moves for optimal alpha-beta pruning efficiency.
*/

#pragma once

#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "search/history.h"
#include <algorithm>
#include <limits>

namespace MetalFish {

// MovePicker stages
enum Stages {
  MAIN_TT,
  CAPTURE_INIT,
  GOOD_CAPTURE,
  QUIET_INIT,
  GOOD_QUIET,
  BAD_CAPTURE,
  BAD_QUIET,
  EVASION_TT,
  EVASION_INIT,
  EVASION,
  PROBCUT_TT,
  PROBCUT_INIT,
  PROBCUT,
  QSEARCH_TT,
  QCAPTURE_INIT,
  QCAPTURE
};

// Partial insertion sort
inline void partial_insertion_sort(ExtMove *begin, ExtMove *end, int limit) {
  for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p) {
    if (p->value >= limit) {
      ExtMove tmp = *p, *q;
      *p = *++sortedEnd;
      for (q = sortedEnd; q != begin && (q - 1)->value < tmp.value; --q)
        *q = *(q - 1);
      *q = tmp;
    }
  }
}

class MovePicker {
public:
  // Constructor for main search
  MovePicker(const Position &p, Move ttm, Depth d, const ButterflyHistory *mh,
             const KillerMoves *km, const CounterMoveHistory *cmh,
             const CapturePieceToHistory *cph, const PieceToHistory **ch,
             int pl);

  // Constructor for quiescence/probcut
  MovePicker(const Position &p, Move ttm, Depth d,
             const CapturePieceToHistory *cph);

  Move next_move();
  void skip_quiet_moves() { skipQuiets = true; }

private:
  template <GenType Type> void score();
  ExtMove *select_best(ExtMove *begin, ExtMove *end);

  const Position &pos;
  const ButterflyHistory *mainHistory;
  const KillerMoves *killers;
  const CounterMoveHistory *counterMoves;
  const CapturePieceToHistory *captureHistory;
  const PieceToHistory *const *continuationHistory;

  ExtMove moves[MAX_MOVES];
  ExtMove *cur, *endMoves, *endBadCaptures;

  Move ttMove;
  Depth depth;
  int ply;
  int threshold;
  int stage;
  bool skipQuiets = false;
};

// Inline implementations
inline MovePicker::MovePicker(const Position &p, Move ttm, Depth d,
                              const ButterflyHistory *mh, const KillerMoves *km,
                              const CounterMoveHistory *cmh,
                              const CapturePieceToHistory *cph,
                              const PieceToHistory **ch, int pl)
    : pos(p), mainHistory(mh), killers(km), counterMoves(cmh),
      captureHistory(cph), continuationHistory(ch), ttMove(ttm), depth(d),
      ply(pl) {
  cur = endMoves = endBadCaptures = moves;
  if (pos.checkers())
    stage = EVASION_TT + !(ttm.is_ok() && pos.pseudo_legal(ttm));
  else
    stage = (d > 0 ? MAIN_TT : QSEARCH_TT) +
            !(ttm.is_ok() && pos.pseudo_legal(ttm));
}

inline MovePicker::MovePicker(const Position &p, Move ttm, Depth d,
                              const CapturePieceToHistory *cph)
    : pos(p), mainHistory(nullptr), killers(nullptr), counterMoves(nullptr),
      captureHistory(cph), continuationHistory(nullptr), ttMove(ttm), depth(d),
      ply(0) {
  cur = endMoves = endBadCaptures = moves;
  if (pos.checkers())
    stage = EVASION_TT + !(ttm.is_ok() && pos.pseudo_legal(ttm));
  else
    stage = QSEARCH_TT + !(ttm.is_ok() && pos.pseudo_legal(ttm));
}

template <> inline void MovePicker::score<CAPTURES>() {
  for (ExtMove *it = cur; it != endMoves; ++it) {
    Move m = *it;
    Square to = m.to_sq();
    Piece captured = pos.piece_on(to);
    PieceType capturedType = (captured != NO_PIECE) ? type_of(captured) : PAWN;
    Piece moved = pos.moved_piece(m);
    it->value = int(PieceValue[captured]) * 6 - int(type_of(moved));
    if (captureHistory)
      it->value += (*captureHistory)[moved][to][capturedType] / 8;
  }
}

template <> inline void MovePicker::score<QUIETS>() {
  Color us = pos.side_to_move();

  for (ExtMove *it = cur; it != endMoves; ++it) {
    Move m = *it;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece moved = pos.moved_piece(m);
    int value = 0;

    // History scores (Stockfish weights)
    if (mainHistory)
      value = 2 * (*mainHistory)[us][from * 64 + to];
    if (continuationHistory) {
      if (continuationHistory[0])
        value += (*continuationHistory[0])[moved][to];
      if (continuationHistory[1])
        value += (*continuationHistory[1])[moved][to];
      if (continuationHistory[2])
        value += (*continuationHistory[2])[moved][to];
      if (continuationHistory[3])
        value += (*continuationHistory[3])[moved][to];
    }

    // Killer move bonus
    if (killers && killers->is_killer(ply, m))
      value += 10000;

    it->value = value;
  }
}

template <> inline void MovePicker::score<EVASIONS>() {
  for (ExtMove *it = cur; it != endMoves; ++it) {
    Move m = *it;
    if (pos.capture(m)) {
      Piece captured = pos.piece_on(m.to_sq());
      it->value = int(PieceValue[captured]) + (1 << 28);
    } else {
      Color us = pos.side_to_move();
      it->value =
          mainHistory ? (*mainHistory)[us][m.from_sq() * 64 + m.to_sq()] : 0;
    }
  }
}

inline ExtMove *MovePicker::select_best(ExtMove *begin, ExtMove *end) {
  std::swap(*begin, *std::max_element(begin, end));
  return begin;
}

inline Move MovePicker::next_move() {
top:
  switch (stage) {
  case MAIN_TT:
  case EVASION_TT:
  case PROBCUT_TT:
  case QSEARCH_TT:
    ++stage;
    return ttMove;

  case CAPTURE_INIT:
  case PROBCUT_INIT:
  case QCAPTURE_INIT: {
    cur = endBadCaptures = moves;
    Move tmp[MAX_MOVES];
    Move *end = generate<CAPTURES>(pos, tmp);
    for (Move *m = tmp; m != end; ++m) {
      endMoves->operator=(*m);
      endMoves->value = 0;
      ++endMoves;
    }
    score<CAPTURES>();
    partial_insertion_sort(cur, endMoves, -3000);
    ++stage;
    goto top;
  }

  case GOOD_CAPTURE:
    while (cur < endMoves) {
      ExtMove *best = select_best(cur++, endMoves);
      if (*best != ttMove) {
        if (pos.see_ge(*best, -best->value / 16))
          return *best;
        *endBadCaptures++ = *best;
      }
    }
    ++stage;
    [[fallthrough]];

  case QUIET_INIT:
    if (!skipQuiets) {
      cur = endBadCaptures;
      endMoves = endBadCaptures;
      Move tmp[MAX_MOVES];
      Move *end = generate<QUIETS>(pos, tmp);
      for (Move *m = tmp; m != end; ++m) {
        endMoves->operator=(*m);
        endMoves->value = 0;
        ++endMoves;
      }
      score<QUIETS>();
      partial_insertion_sort(cur, endMoves, -2000);
    }
    ++stage;
    [[fallthrough]];

  case GOOD_QUIET:
    if (!skipQuiets) {
      while (cur < endMoves) {
        Move m = *(cur++);
        if (m != ttMove)
          return m;
      }
    }
    ++stage;
    [[fallthrough]];

  case BAD_CAPTURE:
    cur = moves;
    while (cur < endBadCaptures) {
      Move m = *(cur++);
      if (m != ttMove)
        return m;
    }
    ++stage;
    [[fallthrough]];

  case BAD_QUIET:
    return Move::none();

  case EVASION_INIT: {
    cur = moves;
    endMoves = moves;
    Move tmp[MAX_MOVES];
    Move *end = generate<EVASIONS>(pos, tmp);
    for (Move *m = tmp; m != end; ++m) {
      endMoves->operator=(*m);
      endMoves->value = 0;
      ++endMoves;
    }
    score<EVASIONS>();
    ++stage;
    [[fallthrough]];
  }

  case EVASION:
    while (cur < endMoves) {
      Move m = *select_best(cur++, endMoves);
      if (m != ttMove)
        return m;
    }
    return Move::none();

  case PROBCUT:
    while (cur < endMoves) {
      Move m = *select_best(cur++, endMoves);
      if (m != ttMove && pos.see_ge(m, threshold))
        return m;
    }
    return Move::none();

  case QCAPTURE:
    while (cur < endMoves) {
      Move m = *select_best(cur++, endMoves);
      if (m != ttMove)
        return m;
    }
    return Move::none();
  }
  return Move::none();
}

} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#ifndef MOVEPICK_H_INCLUDED
#define MOVEPICK_H_INCLUDED

#include "core/movegen.h"
#include "core/types.h"
#include "search/history.h"

namespace MetalFish {

class Position;

// The MovePicker class is used to pick one pseudo-legal move at a time from the
// current position. The most important method is next_move(), which emits one
// new pseudo-legal move on every call, until there are no moves left, when
// Move::none() is returned. In order to improve the efficiency of the
// alpha-beta algorithm, MovePicker attempts to return the moves which are most
// likely to get a cut-off first.
class MovePicker {

public:
  MovePicker(const MovePicker &) = delete;
  MovePicker &operator=(const MovePicker &) = delete;
  MovePicker(const Position &, Move, Depth, const ButterflyHistory *,
             const LowPlyHistory *, const CapturePieceToHistory *,
             const PieceToHistory **, const SharedHistories *, int);
  MovePicker(const Position &, Move, int, const CapturePieceToHistory *);
  Move next_move();
  void skip_quiet_moves();

private:
  template <typename Pred> Move select(Pred);
  template <GenType T> ExtMove *score(MoveList<T> &);
  ExtMove *begin() { return cur; }
  ExtMove *end() { return endCur; }

  const Position &pos;
  const ButterflyHistory *mainHistory;
  const LowPlyHistory *lowPlyHistory;
  const CapturePieceToHistory *captureHistory;
  const PieceToHistory **continuationHistory;
  const SharedHistories *sharedHistory;
  Move ttMove;
  ExtMove *cur, *endCur, *endBadCaptures, *endCaptures, *endGenerated;
  int stage;
  int threshold;
  Depth depth;
  int ply;
  bool skipQuiets = false;
  ExtMove moves[MAX_MOVES];
};

} // namespace MetalFish

#endif // #ifndef MOVEPICK_H_INCLUDED

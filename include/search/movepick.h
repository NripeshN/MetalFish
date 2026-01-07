/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

  MovePicker - Sophisticated move ordering matching Stockfish
*/

#pragma once

#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "search/history.h"
#include <algorithm>
#include <limits>

namespace MetalFish {

// MovePicker stages - matching Stockfish exactly
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

constexpr int GOOD_QUIET_THRESHOLD = -14000;

// Partial insertion sort - matching Stockfish
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
  // Constructor for main search and quiescence search
  MovePicker(const Position &p, Move ttm, Depth d, const ButterflyHistory *mh,
             const LowPlyHistory *lph, const CapturePieceToHistory *cph,
             const PieceToHistory **ch, const PawnHistory *ph, int pl);

  // Constructor for ProbCut: captures with SEE >= threshold
  MovePicker(const Position &p, Move ttm, int th,
             const CapturePieceToHistory *cph);

  Move next_move();
  void skip_quiet_moves() { skipQuiets = true; }

private:
  template <GenType Type> ExtMove *score();

  const Position &pos;
  const ButterflyHistory *mainHistory;
  const LowPlyHistory *lowPlyHistory;
  const CapturePieceToHistory *captureHistory;
  const PieceToHistory *const *continuationHistory;
  const PawnHistory *pawnHistory;

  ExtMove moves[MAX_MOVES];
  ExtMove *cur, *endCur, *endBadCaptures, *endCaptures, *endGenerated;

  Move ttMove;
  Depth depth;
  int ply;
  int threshold;
  int stage;
  bool skipQuiets = false;
};

// Main search/qsearch constructor
inline MovePicker::MovePicker(const Position &p, Move ttm, Depth d,
                              const ButterflyHistory *mh,
                              const LowPlyHistory *lph,
                              const CapturePieceToHistory *cph,
                              const PieceToHistory **ch, const PawnHistory *ph,
                              int pl)
    : pos(p), mainHistory(mh), lowPlyHistory(lph), captureHistory(cph),
      continuationHistory(ch), pawnHistory(ph), ttMove(ttm), depth(d), ply(pl),
      threshold(0) {
  cur = endCur = endBadCaptures = endCaptures = endGenerated = moves;

  if (pos.checkers())
    stage = EVASION_TT + !(ttm.is_ok() && pos.pseudo_legal(ttm));
  else
    stage = (d > 0 ? MAIN_TT : QSEARCH_TT) +
            !(ttm.is_ok() && pos.pseudo_legal(ttm));
}

// ProbCut constructor
inline MovePicker::MovePicker(const Position &p, Move ttm, int th,
                              const CapturePieceToHistory *cph)
    : pos(p), mainHistory(nullptr), lowPlyHistory(nullptr), captureHistory(cph),
      continuationHistory(nullptr), pawnHistory(nullptr), ttMove(ttm), depth(0),
      ply(0), threshold(th) {
  cur = endCur = endBadCaptures = endCaptures = endGenerated = moves;

  stage = PROBCUT_TT +
          !(ttm.is_ok() && pos.capture_stage(ttm) && pos.pseudo_legal(ttm));
}

// Score captures - MVV-LVA + capture history
template <> inline ExtMove *MovePicker::score<CAPTURES>() {
  ExtMove *it = cur;
  Move tmp[MAX_MOVES];
  Move *end = generate<CAPTURES>(pos, tmp);

  for (Move *m = tmp; m != end; ++m) {
    Move move = *m;
    Square to = move.to_sq();
    Piece movedPiece = pos.moved_piece(move);
    Piece capturedPiece = pos.piece_on(to);
    PieceType capturedType =
        (capturedPiece != NO_PIECE) ? type_of(capturedPiece) : PAWN;

    it->operator=(move);
    // Stockfish formula: captureHistory + 7 * PieceValue
    it->value = 7 * int(PieceValue[capturedPiece]);
    if (captureHistory)
      it->value += (*captureHistory)[movedPiece][to][capturedType];
    ++it;
  }
  return it;
}

// Score quiet moves - full Stockfish formula with threatByLesser
template <> inline ExtMove *MovePicker::score<QUIETS>() {
  Color us = pos.side_to_move();
  ExtMove *it = cur;
  Move tmp[MAX_MOVES];
  Move *end = generate<QUIETS>(pos, tmp);

  // Compute threatByLesser bitboards for move scoring
  Bitboard threatByLesser[KING + 1];
  threatByLesser[PAWN] = 0;
  threatByLesser[KNIGHT] = threatByLesser[BISHOP] = pos.attacks_by<PAWN>(~us);
  threatByLesser[ROOK] = pos.attacks_by<KNIGHT>(~us) |
                         pos.attacks_by<BISHOP>(~us) | threatByLesser[KNIGHT];
  threatByLesser[QUEEN] = pos.attacks_by<ROOK>(~us) | threatByLesser[ROOK];
  threatByLesser[KING] = pos.attacks_by<QUEEN>(~us) | threatByLesser[QUEEN];

  for (Move *m = tmp; m != end; ++m) {
    Move move = *m;
    Square from = move.from_sq();
    Square to = move.to_sq();
    Piece movedPiece = pos.moved_piece(move);
    PieceType pt = type_of(movedPiece);
    int value = 0;

    // 2x mainHistory weight
    if (mainHistory)
      value = 2 * (*mainHistory)[us][from * 64 + to];

    // Pawn history
    if (pawnHistory) {
      int pawnIdx = pawn_history_index(pos);
      value += 2 * (*pawnHistory)[pawnIdx][movedPiece][to];
    }

    // All 6 continuation histories (indices 0,1,2,3,5 - skipping 4)
    if (continuationHistory) {
      if (continuationHistory[0])
        value += (*continuationHistory[0])[movedPiece][to];
      if (continuationHistory[1])
        value += (*continuationHistory[1])[movedPiece][to];
      if (continuationHistory[2])
        value += (*continuationHistory[2])[movedPiece][to];
      if (continuationHistory[3])
        value += (*continuationHistory[3])[movedPiece][to];
      if (continuationHistory[5])
        value += (*continuationHistory[5])[movedPiece][to];
    }

    // Bonus for checks
    if ((pos.check_squares(pt) & to) && pos.see_ge(move, -75))
      value += 16384;

    // Penalty for moving to a square threatened by lesser piece
    // Bonus for escaping an attack by lesser piece
    int threatBonus =
        (threatByLesser[pt] & to) ? -19 : 20 * bool(threatByLesser[pt] & from);
    value += PieceValue[pt] * threatBonus;

    // Low ply history bonus
    if (lowPlyHistory && ply < LOW_PLY_HISTORY_SIZE)
      value += 8 * (*lowPlyHistory)[ply][from * 64 + to] / (1 + ply);

    it->operator=(move);
    it->value = value;
    ++it;
  }
  return it;
}

// Score evasions
template <> inline ExtMove *MovePicker::score<EVASIONS>() {
  ExtMove *it = cur;
  Move tmp[MAX_MOVES];
  Move *end = generate<EVASIONS>(pos, tmp);

  for (Move *m = tmp; m != end; ++m) {
    Move move = *m;
    it->operator=(move);

    if (pos.capture_stage(move)) {
      Piece capturedPiece = pos.piece_on(move.to_sq());
      it->value = int(PieceValue[capturedPiece]) + (1 << 28);
    } else {
      Color us = pos.side_to_move();
      Piece movedPiece = pos.moved_piece(move);
      it->value = mainHistory
                      ? (*mainHistory)[us][move.from_sq() * 64 + move.to_sq()]
                      : 0;
      if (continuationHistory && continuationHistory[0])
        it->value += (*continuationHistory[0])[movedPiece][move.to_sq()];
    }
    ++it;
  }
  return it;
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
    endCur = endCaptures = score<CAPTURES>();
    partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
    ++stage;
    goto top;
  }

  case GOOD_CAPTURE:
    while (cur < endCur) {
      Move m = *cur;
      if (m != ttMove) {
        // SEE threshold: -value/18 (Stockfish)
        if (pos.see_ge(m, -cur->value / 18)) {
          ++cur;
          return m;
        }
        // Move to bad captures
        std::swap(*endBadCaptures++, *cur);
      }
      ++cur;
    }
    ++stage;
    [[fallthrough]];

  case QUIET_INIT:
    if (!skipQuiets) {
      cur = endCaptures;
      endCur = endGenerated = score<QUIETS>();
      // Stockfish: partial sort with -3560 * depth threshold
      partial_insertion_sort(cur, endCur, -3560 * depth);
    }
    ++stage;
    [[fallthrough]];

  case GOOD_QUIET:
    if (!skipQuiets) {
      while (cur < endCur) {
        if (cur->value <= GOOD_QUIET_THRESHOLD)
          break;
        Move m = *cur++;
        if (m != ttMove)
          return m;
      }
    }
    // Prepare for bad captures
    cur = moves;
    endCur = endBadCaptures;
    ++stage;
    [[fallthrough]];

  case BAD_CAPTURE:
    while (cur < endCur) {
      Move m = *cur++;
      if (m != ttMove)
        return m;
    }
    // Prepare for bad quiets
    cur = endCaptures;
    endCur = endGenerated;
    ++stage;
    [[fallthrough]];

  case BAD_QUIET:
    if (!skipQuiets) {
      while (cur < endCur) {
        Move m = *cur++;
        if (m != ttMove && cur[-1].value <= GOOD_QUIET_THRESHOLD)
          return m;
      }
    }
    return Move::none();

  case EVASION_INIT: {
    cur = moves;
    endCur = endGenerated = score<EVASIONS>();
    partial_insertion_sort(cur, endCur, std::numeric_limits<int>::min());
    ++stage;
    [[fallthrough]];
  }

  case EVASION:
    while (cur < endCur) {
      Move m = *cur++;
      if (m != ttMove)
        return m;
    }
    return Move::none();

  case PROBCUT:
    while (cur < endCur) {
      Move m = *cur++;
      if (m != ttMove && pos.see_ge(m, threshold))
        return m;
    }
    return Move::none();

  case QCAPTURE:
    while (cur < endCur) {
      Move m = *cur++;
      if (m != ttMove)
        return m;
    }
    return Move::none();
  }
  return Move::none();
}

} // namespace MetalFish

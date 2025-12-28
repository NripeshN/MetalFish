/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers
*/

#include "search/movepick.h"
#include "core/bitboard.h"
#include <algorithm>
#include <cstring>

namespace MetalFish {

namespace {

// Piece values for MVV-LVA ordering
constexpr int PieceValueMVV[PIECE_TYPE_NB] = {
    0,    // NO_PIECE_TYPE
    100,  // PAWN
    320,  // KNIGHT
    330,  // BISHOP
    500,  // ROOK
    900,  // QUEEN
    20000 // KING
};

// Partial insertion sort - sort best moves to front
void partial_insertion_sort(ExtMove *begin, ExtMove *end, int limit) {
  for (ExtMove *sortedEnd = begin, *p = begin + 1; p < end; ++p) {
    if (p->value >= limit) {
      ExtMove tmp = *p, *q;
      *p = *++sortedEnd;
      for (q = sortedEnd; q != begin && *(q - 1) < tmp; --q)
        *q = *(q - 1);
      *q = tmp;
    }
  }
}

} // namespace

// Constructor for main search
MovePicker::MovePicker(const Position &p, Move ttm, int d,
                       const ButterflyHistory *mh, const KillerMoves *km,
                       const CounterMoveHistory *cmh,
                       const CapturePieceToHistory *cph,
                       const PieceToHistory **ch, int pl)
    : pos(p), mainHistory(mh), killers(km), counterMoves(cmh),
      captureHistory(cph), continuationHistory(ch), ttMove(ttm), depth(d),
      ply(pl) {
  cur = endMoves = endBadCaptures = moves;

  if (pos.checkers())
    stage = Stages(EVASION_TT + !(ttm.is_ok() && pos.pseudo_legal(ttm)));
  else
    stage = Stages((depth > 0 ? MAIN_TT : QSEARCH_TT) +
                   !(ttm.is_ok() && pos.pseudo_legal(ttm)));
}

// Constructor for quiescence search
MovePicker::MovePicker(const Position &p, Move ttm, int d,
                       const CapturePieceToHistory *cph)
    : pos(p), mainHistory(nullptr), killers(nullptr), counterMoves(nullptr),
      captureHistory(cph), continuationHistory(nullptr), ttMove(ttm), depth(d),
      ply(0) {
  cur = endMoves = endBadCaptures = moves;

  if (pos.checkers())
    stage = Stages(EVASION_TT + !(ttm.is_ok() && pos.pseudo_legal(ttm)));
  else
    stage = Stages(QSEARCH_TT + !(ttm.is_ok() && pos.pseudo_legal(ttm)));
}

// Constructor for probcut
MovePicker::MovePicker(const Position &p, Move ttm, int th,
                       const CapturePieceToHistory *cph, bool)
    : pos(p), mainHistory(nullptr), killers(nullptr), counterMoves(nullptr),
      captureHistory(cph), continuationHistory(nullptr), ttMove(ttm), depth(0),
      ply(0), threshold(th) {
  cur = endMoves = endBadCaptures = moves;
  stage = Stages(PROBCUT_TT +
                 !(ttm.is_ok() && pos.capture(ttm) && pos.pseudo_legal(ttm)));
}

// Generate moves and convert to ExtMove
void MovePicker::generate_moves(GenType type) {
  Move tmp[MAX_MOVES];
  Move *end;

  switch (type) {
  case CAPTURES:
    end = generate<CAPTURES>(pos, tmp);
    break;
  case QUIETS:
    end = generate<QUIETS>(pos, tmp);
    break;
  case EVASIONS:
    end = generate<EVASIONS>(pos, tmp);
    break;
  default:
    end = generate<NON_EVASIONS>(pos, tmp);
    break;
  }

  // Copy to ExtMove array
  for (Move *m = tmp; m != end; ++m) {
    endMoves->operator=(*m);
    endMoves->value = 0;
    ++endMoves;
  }
}

// Score captures for MVV-LVA ordering
template <> void MovePicker::score<CAPTURES>() {
  for (ExtMove *it = cur; it != endMoves; ++it) {
    Move m = *it;
    Square to = m.to_sq();
    Piece captured = pos.piece_on(to);
    PieceType capturedType = (captured != NO_PIECE) ? type_of(captured) : PAWN;

    // MVV-LVA: Most Valuable Victim - Least Valuable Attacker
    Piece moved = pos.moved_piece(m);
    PieceType movedType = type_of(moved);

    int value = PieceValueMVV[capturedType] * 6 - PieceValueMVV[movedType];

    // Add capture history
    if (captureHistory) {
      value += (*captureHistory)[moved][to][capturedType] / 8;
    }

    // Bonus for promotions
    if (m.type_of() == PROMOTION) {
      value += PieceValueMVV[m.promotion_type()] - PieceValueMVV[PAWN];
    }

    it->value = value;
  }
}

// Score quiet moves by history
template <> void MovePicker::score<QUIETS>() {
  Color us = pos.side_to_move();

  for (ExtMove *it = cur; it != endMoves; ++it) {
    Move m = *it;
    Square from = m.from_sq();
    Square to = m.to_sq();
    Piece moved = pos.moved_piece(m);

    int value = 0;

    // Main history
    if (mainHistory) {
      value = (*mainHistory)[us][from * 64 + to];
    }

    // Continuation history
    if (continuationHistory) {
      for (int i = 0; i < 4; ++i) {
        if (continuationHistory[i]) {
          value += (*continuationHistory[i])[moved][to];
        }
      }
    }

    // Killer bonus
    if (killers && killers->is_killer(ply, m)) {
      value += 10000;
    }

    it->value = value;
  }
}

// Score evasion moves
template <> void MovePicker::score<EVASIONS>() {
  for (ExtMove *it = cur; it != endMoves; ++it) {
    Move m = *it;

    if (pos.capture(m)) {
      // Score captures by MVV-LVA
      Square to = m.to_sq();
      Piece captured = pos.piece_on(to);
      PieceType capturedType =
          (captured != NO_PIECE) ? type_of(captured) : PAWN;
      Piece moved = pos.moved_piece(m);

      it->value = PieceValueMVV[capturedType] - type_of(moved) + (1 << 28);
    } else {
      // Score quiets by history
      Color us = pos.side_to_move();
      Square from = m.from_sq();
      Square to = m.to_sq();

      it->value = mainHistory ? (*mainHistory)[us][from * 64 + to] : 0;
    }
  }
}

// Select best move from remaining moves
ExtMove *MovePicker::select_best(ExtMove *begin, ExtMove *end) {
  std::swap(*begin, *std::max_element(begin, end));
  return begin;
}

// Get next move
Move MovePicker::next_move() {
top:
  switch (stage) {

  case MAIN_TT:
  case EVASION_TT:
  case PROBCUT_TT:
  case QSEARCH_TT:
    stage = Stages(stage + 1);
    return ttMove;

  case CAPTURE_INIT:
  case PROBCUT_INIT:
  case QCAPTURE_INIT:
    cur = endBadCaptures = moves;
    endMoves = moves;
    generate_moves(CAPTURES);
    score<CAPTURES>();
    partial_insertion_sort(cur, endMoves, -3000);
    stage = Stages(stage + 1);
    goto top;

  case GOOD_CAPTURE:
    while (cur < endMoves) {
      ExtMove *best = select_best(cur++, endMoves);
      Move m = *best;
      if (m != ttMove) {
        if (pos.see_ge(m, -best->value / 16)) {
          return m;
        }
        // Bad capture - save for later
        endBadCaptures->operator=(m);
        endBadCaptures->value = best->value;
        ++endBadCaptures;
      }
    }
    stage = Stages(stage + 1);
    [[fallthrough]];

  case QUIET_INIT:
    if (!skipQuiets) {
      cur = endBadCaptures;
      endMoves = endBadCaptures;
      generate_moves(QUIETS);
      score<QUIETS>();
      partial_insertion_sort(cur, endMoves, -2000);
    }
    stage = Stages(stage + 1);
    [[fallthrough]];

  case GOOD_QUIET:
    if (!skipQuiets) {
      while (cur < endMoves) {
        Move m = *(cur++);
        if (m != ttMove && (!killers || !killers->is_killer(ply, m)))
          return m;
      }
    }
    stage = Stages(stage + 1);
    [[fallthrough]];

  case BAD_CAPTURE:
    cur = moves;
    while (cur < endBadCaptures) {
      Move m = *(cur++);
      if (m != ttMove)
        return m;
    }
    stage = Stages(stage + 1);
    [[fallthrough]];

  case BAD_QUIET:
    // All quiet moves already returned in GOOD_QUIET
    // BAD_QUIET stage is kept for compatibility but not used
    // since we don't split quiets into good/bad in this implementation
    return Move::none();

  case EVASION_INIT:
    cur = moves;
    endMoves = moves;
    generate_moves(EVASIONS);
    score<EVASIONS>();
    stage = Stages(stage + 1);
    [[fallthrough]];

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

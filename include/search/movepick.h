/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MovePicker - Sophisticated move ordering
  ========================================

  Orders moves for optimal alpha-beta pruning efficiency:
  1. TT move (hash move)
  2. Good captures (MVV-LVA + SEE)
  3. Killer moves
  4. Counter moves
  5. Quiet moves by history
  6. Bad captures
*/

#pragma once

#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include <algorithm>

namespace MetalFish {

// Forward declarations for history tables
template <typename T, int D> class StatsEntry;

// Simple butterfly history (indexed by [color][from*64+to])
using ButterflyHistory = int16_t[COLOR_NB][SQUARE_NB * SQUARE_NB];

// Capture history: [piece][to][captured_type]
using CapturePieceToHistory = int16_t[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB];

// Pawn history: indexed by pawn structure hash, [piece][to]
// Size must be power of 2 for efficient modulo operation
// Reduced size for memory efficiency (128 * 16 * 64 * 2 = 256KB)
constexpr int PAWN_HISTORY_SIZE = 128;
using PawnHistory = int16_t[PAWN_HISTORY_SIZE][PIECE_NB][SQUARE_NB];

// Get pawn history index from position's pawn key
inline int pawn_history_index(const Position &pos) {
  return pos.pawn_key() & (PAWN_HISTORY_SIZE - 1);
}

// Piece to history for continuation
using PieceToHistory = int16_t[PIECE_NB][SQUARE_NB];

// Killer moves structure
struct KillerMoves {
  static constexpr int MAX_PLY = 128;
  static constexpr int NUM_KILLERS = 2;

  Move killers[MAX_PLY][NUM_KILLERS];

  KillerMoves() { clear(); }

  void clear() {
    for (int i = 0; i < MAX_PLY; ++i)
      for (int j = 0; j < NUM_KILLERS; ++j)
        killers[i][j] = Move::none();
  }

  void update(int ply, Move move) {
    if (ply >= MAX_PLY)
      return;
    if (killers[ply][0] != move) {
      killers[ply][1] = killers[ply][0];
      killers[ply][0] = move;
    }
  }

  bool is_killer(int ply, Move move) const {
    if (ply >= MAX_PLY)
      return false;
    return killers[ply][0] == move || killers[ply][1] == move;
  }
};

// Counter moves: [piece][to]
using CounterMoveHistory = Move[PIECE_NB][SQUARE_NB];

// MovePicker stages
enum Stages {
  // Main search
  MAIN_TT,
  CAPTURE_INIT,
  GOOD_CAPTURE,
  QUIET_INIT,
  GOOD_QUIET,
  BAD_CAPTURE,
  BAD_QUIET,

  // Evasion search
  EVASION_TT,
  EVASION_INIT,
  EVASION,

  // Probcut
  PROBCUT_TT,
  PROBCUT_INIT,
  PROBCUT,

  // Quiescence search
  QSEARCH_TT,
  QCAPTURE_INIT,
  QCAPTURE
};

class MovePicker {
public:
  // Constructor for main search
  MovePicker(const Position &p, Move ttm, int depth, const ButterflyHistory *mh,
             const KillerMoves *km, const CounterMoveHistory *cmh,
             const CapturePieceToHistory *cph, const PieceToHistory **ch,
             int ply);

  // Constructor for quiescence search
  MovePicker(const Position &p, Move ttm, int depth,
             const CapturePieceToHistory *cph);

  // Constructor for probcut
  MovePicker(const Position &p, Move ttm, int threshold,
             const CapturePieceToHistory *cph, bool probcut);

  // Get next move
  Move next_move();

  // Skip remaining quiet moves
  void skip_quiet_moves() { skipQuiets = true; }

private:
  // Generate moves into internal array
  void generate_moves(GenType type);

  // Score moves for ordering
  template <GenType> void score();

  // Select best remaining move
  ExtMove *select_best(ExtMove *begin, ExtMove *end);

  // Position reference
  const Position &pos;

  // History tables
  const ButterflyHistory *mainHistory;
  const KillerMoves *killers;
  const CounterMoveHistory *counterMoves;
  const CapturePieceToHistory *captureHistory;
  const PieceToHistory *const *continuationHistory;

  // Move lists
  ExtMove moves[MAX_MOVES];
  ExtMove *cur;
  ExtMove *endMoves;
  ExtMove *endBadCaptures;

  // State
  Move ttMove;
  int depth;
  int ply;
  int threshold;
  Stages stage;
  bool skipQuiets = false;
};

// Helper to check if a move is pseudo-legal
inline bool is_pseudo_legal(const Position &pos, Move m) {
  if (!m.is_ok())
    return false;

  Color us = pos.side_to_move();
  Square from = m.from_sq();

  Piece pc = pos.piece_on(from);
  if (pc == NO_PIECE || color_of(pc) != us)
    return false;

  // Basic validity checks - use pos.pseudo_legal() for full check
  return true;
}

} // namespace MetalFish

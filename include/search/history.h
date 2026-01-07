/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  History Tables - Move ordering heuristics
  =========================================

  Implements Stockfish-style history tables for move ordering.
  Sizes are reduced from Stockfish defaults for memory efficiency.
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include <algorithm>
#include <cstdint>
#include <cstring>

namespace MetalFish {

// History table constants - reduced sizes for memory efficiency
constexpr int PAWN_HISTORY_SIZE = 64;  // Reduced from 8192
constexpr int CORRECTION_HISTORY_SIZE = 256;  // Reduced from 16384  
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
constexpr int LOW_PLY_HISTORY_SIZE = 5;

// Simple butterfly history (indexed by [color][from*64+to])
using ButterflyHistory = int16_t[COLOR_NB][SQUARE_NB * SQUARE_NB];

// Capture history: [piece][to][captured_type]
using CapturePieceToHistory = int16_t[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB];

// Pawn history: indexed by pawn structure hash, [piece][to]
using PawnHistory = int16_t[PAWN_HISTORY_SIZE][PIECE_NB][SQUARE_NB];

// Correction history: adjusts static eval based on search results
using CorrectionHistory = int16_t[CORRECTION_HISTORY_SIZE][COLOR_NB];

// Low ply history: extra weight for moves at low search depths
using LowPlyHistory = int16_t[LOW_PLY_HISTORY_SIZE][SQUARE_NB * SQUARE_NB];

// Piece to history for continuation
using PieceToHistory = int16_t[PIECE_NB][SQUARE_NB];

// Continuation correction history
using ContinuationCorrectionHistory = int16_t[PIECE_NB][SQUARE_NB];

// Counter moves: [piece][to] -> refutation move
using CounterMoveHistory = Move[PIECE_NB][SQUARE_NB];

// TT move history for singular extension
using TTMoveHistory = int16_t;

// Index functions
inline int pawn_history_index(const Position &pos) {
  return pos.pawn_key() & (PAWN_HISTORY_SIZE - 1);
}

inline int correction_history_index(Key key) {
  return key & (CORRECTION_HISTORY_SIZE - 1);
}

// Gravity update for history tables
inline void history_update(int16_t &entry, int bonus, int maxVal = 16384) {
  int clampedBonus = std::clamp(bonus, -maxVal, maxVal);
  entry += clampedBonus - entry * std::abs(clampedBonus) / maxVal;
}

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

  Move get_killer(int ply, int idx) const {
    if (ply >= MAX_PLY || idx >= NUM_KILLERS)
      return Move::none();
    return killers[ply][idx];
  }
};

// Compute correction value
inline int compute_correction_value(const CorrectionHistory &corrHist,
                                    const Position &pos) {
  Color us = pos.side_to_move();
  int idx = correction_history_index(pos.pawn_key());
  return corrHist[idx][us];
}

// Apply correction to static eval
inline Value to_corrected_static_eval(Value v, int correctionValue) {
  return std::clamp(v + correctionValue / 256,
                    VALUE_TB_LOSS_IN_MAX_PLY + 1,
                    VALUE_TB_WIN_IN_MAX_PLY - 1);
}

} // namespace MetalFish

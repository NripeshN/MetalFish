/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  History Tables - Move ordering heuristics
  =========================================

  Implements Stockfish-style history tables for move ordering.
  Includes full correction history system (pawn, minor, nonpawn, continuation).
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace MetalFish {

// History table constants
constexpr int PAWN_HISTORY_SIZE = 512;  // Reduced for memory efficiency
constexpr int CORRECTION_HISTORY_SIZE = 1024;  // Reduced for memory efficiency
constexpr int CORRECTION_HISTORY_LIMIT = 1024;
constexpr int LOW_PLY_HISTORY_SIZE = 5;
constexpr int BUTTERFLY_HISTORY_LIMIT = 7183;
constexpr int CAPTURE_HISTORY_LIMIT = 10692;
constexpr int CONTINUATION_HISTORY_LIMIT = 30000;
constexpr int SEARCHEDLIST_CAPACITY = 32;
constexpr int MAIN_HISTORY_DEFAULT = 68;

// =============================================================================
// SearchedList - Fixed-size list for tracking searched moves
// =============================================================================

template<typename T, std::size_t MaxSize>
class ValueList {
public:
  std::size_t size() const { return size_; }
  int ssize() const { return int(size_); }
  void clear() { size_ = 0; }
  
  void push_back(const T& value) {
    if (size_ < MaxSize)
      values_[size_++] = value;
  }
  
  const T* begin() const { return values_; }
  const T* end() const { return values_ + size_; }
  T* begin() { return values_; }
  T* end() { return values_ + size_; }
  
  const T& operator[](int index) const { return values_[index]; }
  T& operator[](int index) { return values_[index]; }

private:
  T values_[MaxSize];
  std::size_t size_ = 0;
};

using SearchedList = ValueList<Move, SEARCHEDLIST_CAPACITY>;

// Continuation history bonus structure (matching Stockfish)
struct ConthistBonus {
  int ply;
  int weight;
};

// Stockfish continuation history bonuses
constexpr std::array<ConthistBonus, 6> CONTHIST_BONUSES = {{
  {1, 1133}, {2, 683}, {3, 312}, {4, 582}, {5, 149}, {6, 474}
}};

// StatsEntry with gravity update (matching Stockfish)
template<typename T, int D>
struct StatsEntry {
  T value = 0;

  operator T() const { return value; }
  void operator=(T v) { value = v; }

  // Gravity update: bonus decays existing value
  void operator<<(int bonus) {
    int clampedBonus = std::clamp(bonus, -D, D);
    value += clampedBonus - value * std::abs(clampedBonus) / D;
  }
};

// Simple butterfly history (indexed by [color][from*64+to])
using ButterflyHistory = int16_t[COLOR_NB][SQUARE_NB * SQUARE_NB];

// Capture history: [piece][to][captured_type]
using CapturePieceToHistory = int16_t[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB];

// Pawn history: indexed by pawn structure hash, [piece][to]
using PawnHistory = int16_t[PAWN_HISTORY_SIZE][PIECE_NB][SQUARE_NB];

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

// =============================================================================
// Full Correction History System (matching Stockfish)
// =============================================================================

// Correction bundle: stores all correction types for a hash entry
struct CorrectionBundle {
  std::atomic<int16_t> pawn{0};
  std::atomic<int16_t> minor{0};
  std::atomic<int16_t> nonPawnWhite{0};
  std::atomic<int16_t> nonPawnBlack{0};

  void clear() {
    pawn.store(0, std::memory_order_relaxed);
    minor.store(0, std::memory_order_relaxed);
    nonPawnWhite.store(0, std::memory_order_relaxed);
    nonPawnBlack.store(0, std::memory_order_relaxed);
  }

  // Update with gravity
  void update_pawn(int bonus) {
    int16_t val = pawn.load(std::memory_order_relaxed);
    int clampedBonus = std::clamp(bonus, -CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT);
    int16_t newVal = val + clampedBonus - val * std::abs(clampedBonus) / CORRECTION_HISTORY_LIMIT;
    pawn.store(newVal, std::memory_order_relaxed);
  }

  void update_minor(int bonus) {
    int16_t val = minor.load(std::memory_order_relaxed);
    int clampedBonus = std::clamp(bonus, -CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT);
    int16_t newVal = val + clampedBonus - val * std::abs(clampedBonus) / CORRECTION_HISTORY_LIMIT;
    minor.store(newVal, std::memory_order_relaxed);
  }

  void update_nonpawn_white(int bonus) {
    int16_t val = nonPawnWhite.load(std::memory_order_relaxed);
    int clampedBonus = std::clamp(bonus, -CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT);
    int16_t newVal = val + clampedBonus - val * std::abs(clampedBonus) / CORRECTION_HISTORY_LIMIT;
    nonPawnWhite.store(newVal, std::memory_order_relaxed);
  }

  void update_nonpawn_black(int bonus) {
    int16_t val = nonPawnBlack.load(std::memory_order_relaxed);
    int clampedBonus = std::clamp(bonus, -CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT);
    int16_t newVal = val + clampedBonus - val * std::abs(clampedBonus) / CORRECTION_HISTORY_LIMIT;
    nonPawnBlack.store(newVal, std::memory_order_relaxed);
  }
};

// Unified correction history table
struct UnifiedCorrectionHistory {
  CorrectionBundle entries[CORRECTION_HISTORY_SIZE][COLOR_NB];

  void clear() {
    for (int i = 0; i < CORRECTION_HISTORY_SIZE; ++i)
      for (int c = 0; c < COLOR_NB; ++c)
        entries[i][c].clear();
  }

  CorrectionBundle& at(int index, Color c) {
    return entries[index & (CORRECTION_HISTORY_SIZE - 1)][c];
  }

  const CorrectionBundle& at(int index, Color c) const {
    return entries[index & (CORRECTION_HISTORY_SIZE - 1)][c];
  }
};

// Simple correction history (fallback)
using SimpleCorrectionHistory = int16_t[CORRECTION_HISTORY_SIZE][COLOR_NB];

// =============================================================================
// Index functions
// =============================================================================

inline int pawn_history_index(const Position &pos) {
  return pos.pawn_key() & (PAWN_HISTORY_SIZE - 1);
}

inline int correction_history_index(Key key) {
  return key & (CORRECTION_HISTORY_SIZE - 1);
}

// Minor piece key: position of knights and bishops
inline Key minor_piece_key(const Position &pos) {
  return pos.pieces(KNIGHT) ^ (pos.pieces(BISHOP) * 0x9E3779B97F4A7C15ULL);
}

// Non-pawn key for a color
inline Key non_pawn_key(const Position &pos, Color c) {
  return pos.pieces(c, KNIGHT) ^ pos.pieces(c, BISHOP) ^ 
         pos.pieces(c, ROOK) ^ pos.pieces(c, QUEEN);
}

// =============================================================================
// Gravity update for history tables
// =============================================================================

inline void history_update(int16_t &entry, int bonus, int maxVal = 16384) {
  int clampedBonus = std::clamp(bonus, -maxVal, maxVal);
  entry += clampedBonus - entry * std::abs(clampedBonus) / maxVal;
}

// =============================================================================
// Correction value computation (matching Stockfish)
// =============================================================================

// Compute full correction value from all correction history components
inline int compute_full_correction_value(
    const UnifiedCorrectionHistory &corrHist,
    const ContinuationCorrectionHistory *contCorrHist2,
    const ContinuationCorrectionHistory *contCorrHist4,
    const Position &pos,
    Move prevMove) {
  
  Color us = pos.side_to_move();
  
  // Get indices
  int pawnIdx = correction_history_index(pos.pawn_key());
  int minorIdx = correction_history_index(minor_piece_key(pos));
  int nonPawnWhiteIdx = correction_history_index(non_pawn_key(pos, WHITE));
  int nonPawnBlackIdx = correction_history_index(non_pawn_key(pos, BLACK));
  
  // Get correction values
  int pcv = corrHist.at(pawnIdx, us).pawn.load(std::memory_order_relaxed);
  int micv = corrHist.at(minorIdx, us).minor.load(std::memory_order_relaxed);
  int wnpcv = corrHist.at(nonPawnWhiteIdx, us).nonPawnWhite.load(std::memory_order_relaxed);
  int bnpcv = corrHist.at(nonPawnBlackIdx, us).nonPawnBlack.load(std::memory_order_relaxed);
  
  // Continuation correction
  int cntcv = 8; // Default
  if (prevMove.is_ok() && contCorrHist2 && contCorrHist4) {
    Piece pc = pos.piece_on(prevMove.to_sq());
    if (pc != NO_PIECE) {
      Square to = prevMove.to_sq();
      cntcv = (*contCorrHist2)[pc][to] + (*contCorrHist4)[pc][to];
    }
  }
  
  // Stockfish weights
  return 10347 * pcv + 8821 * micv + 11665 * (wnpcv + bnpcv) + 7841 * cntcv;
}

// Apply correction to static eval (matching Stockfish)
inline Value to_corrected_static_eval(Value v, int correctionValue) {
  return std::clamp(v + correctionValue / 131072,
                    VALUE_TB_LOSS_IN_MAX_PLY + 1,
                    VALUE_TB_WIN_IN_MAX_PLY - 1);
}

// Update full correction history (matching Stockfish)
inline void update_full_correction_history(
    UnifiedCorrectionHistory &corrHist,
    ContinuationCorrectionHistory *contCorrHist2,
    ContinuationCorrectionHistory *contCorrHist4,
    const Position &pos,
    Move prevMove,
    int bonus) {
  
  Color us = pos.side_to_move();
  constexpr int nonPawnWeight = 178;
  
  // Get indices
  int pawnIdx = correction_history_index(pos.pawn_key());
  int minorIdx = correction_history_index(minor_piece_key(pos));
  int nonPawnWhiteIdx = correction_history_index(non_pawn_key(pos, WHITE));
  int nonPawnBlackIdx = correction_history_index(non_pawn_key(pos, BLACK));
  
  // Update all correction components with appropriate weights
  corrHist.at(pawnIdx, us).update_pawn(bonus);
  corrHist.at(minorIdx, us).update_minor(bonus * 156 / 128);
  corrHist.at(nonPawnWhiteIdx, us).update_nonpawn_white(bonus * nonPawnWeight / 128);
  corrHist.at(nonPawnBlackIdx, us).update_nonpawn_black(bonus * nonPawnWeight / 128);
  
  // Update continuation correction history
  if (prevMove.is_ok() && contCorrHist2 && contCorrHist4) {
    Piece pc = pos.piece_on(prevMove.to_sq());
    if (pc != NO_PIECE) {
      Square to = prevMove.to_sq();
      history_update((*contCorrHist2)[pc][to], bonus * 127 / 128, CORRECTION_HISTORY_LIMIT);
      history_update((*contCorrHist4)[pc][to], bonus * 59 / 128, CORRECTION_HISTORY_LIMIT);
    }
  }
}

// =============================================================================
// Killer moves structure
// =============================================================================

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

} // namespace MetalFish

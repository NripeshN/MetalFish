/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#pragma once

#include "core/position.h"
#include "core/types.h"
#include "search/history.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace MetalFish {

class TranspositionTable;

namespace Search {

// Different node types for search
enum NodeType { NonPV, PV, Root };

// Stack keeps track of search state at each ply
struct Stack {
  Move *pv;
  int ply;
  Move currentMove;
  Move excludedMove;
  Value staticEval;
  int statScore;
  int moveCount;
  bool inCheck;
  bool ttPv;
  bool ttHit;
  int cutoffCnt;
  int reduction;
  PieceToHistory *continuationHistory;
  ContinuationCorrectionHistory *continuationCorrectionHistory;
};

// RootMove stores information about root moves
struct RootMove {
  explicit RootMove(Move m) : pv(1, m) {}

  bool operator==(const Move &m) const { return pv[0] == m; }
  bool operator<(const RootMove &m) const {
    return m.score != score ? m.score < score : m.previousScore < previousScore;
  }

  uint64_t effort = 0;
  Value score = -VALUE_INFINITE;
  Value previousScore = -VALUE_INFINITE;
  Value averageScore = -VALUE_INFINITE;
  Value uciScore = -VALUE_INFINITE;
  int64_t meanSquaredScore = 0;  // For aspiration window sizing
  bool scoreLowerbound = false;
  bool scoreUpperbound = false;
  int selDepth = 0;
  std::vector<Move> pv;
  int tbRank = 0;
  Value tbScore = VALUE_ZERO;
};

using RootMoves = std::vector<RootMove>;

// Search limits configuration
struct LimitsType {
  LimitsType() {
    time[WHITE] = time[BLACK] = inc[WHITE] = inc[BLACK] = 0;
    movetime = 0;
    movestogo = depth = mate = infinite = 0;
    nodes = 0;
    ponderMode = false;
    multiPV = 1;
  }

  bool use_time_management() const { return time[WHITE] || time[BLACK] || movetime; }

  std::vector<Move> searchmoves;
  int64_t time[COLOR_NB], inc[COLOR_NB], movetime;
  int movestogo, depth, mate, infinite;
  uint64_t nodes;
  bool ponderMode;
  int multiPV;
};

// Time management
using TimePoint = std::chrono::milliseconds::rep;

inline TimePoint now() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class TimeManager {
public:
  void init(const LimitsType &limits, Color us, int ply);
  TimePoint optimum() const { return optimumTime; }
  TimePoint maximum() const { return maximumTime; }
  TimePoint elapsed() const { return now() - startTime; }
  void adjust_time(double ratio);

private:
  TimePoint startTime;
  TimePoint optimumTime;
  TimePoint maximumTime;
};

// Main search worker class
class Worker {
public:
  Worker(size_t threadIdx = 0);
  ~Worker();

  void clear();
  void start_searching(Position &pos, const LimitsType &limits, StateListPtr &states);
  void wait_for_search_finished();
  bool is_main_thread() const { return threadIdx == 0; }

  std::atomic<uint64_t> nodes{0};
  std::atomic<uint64_t> tbHits{0};
  std::atomic<double> bestMoveChanges{0};

  int selDepth = 0;
  int nmpMinPly = 0;
  int pvIdx = 0;
  int pvLast = 0;

  RootMoves rootMoves;
  Depth rootDepth;
  Depth completedDepth;
  Value rootDelta = VALUE_INFINITE;

  Position *rootPos;
  LimitsType limits;
  StateListPtr *states;

  std::atomic<bool> searching{false};
  std::atomic<bool> stopRequested{false};

private:
  void iterative_deepening();

  template <NodeType nodeType>
  Value search(Position &pos, Stack *ss, Value alpha, Value beta, Depth depth, bool cutNode);

  template <NodeType nodeType>
  Value qsearch(Position &pos, Stack *ss, Value alpha, Value beta);

  Value evaluate(const Position &pos);

  void update_quiet_stats(Stack *ss, Move move, int bonus);
  void update_capture_stats(Piece piece, Square to, PieceType captured, int bonus);
  void update_continuation_histories(Stack *ss, Piece pc, Square to, int bonus);
  void update_quiet_histories(const Position &pos, Stack *ss, Move move, int bonus);
  void update_all_stats(const Position &pos, Stack *ss, Move bestMove, Square prevSq,
                        SearchedList &quietsSearched, SearchedList &capturesSearched,
                        Depth depth, Move ttMove, int moveCount);

  size_t threadIdx;
  TimeManager timeManager;

  int reductions[MAX_MOVES];

  // History tables (stack allocated - smaller tables)
  ButterflyHistory mainHistory;
  LowPlyHistory lowPlyHistory;
  CapturePieceToHistory captureHistory;
  PawnHistory pawnHistory;
  
  // Full correction history system (heap allocated due to size)
  std::unique_ptr<UnifiedCorrectionHistory> correctionHistory;
  
  // Continuation histories
  PieceToHistory continuationHistoryTable[PIECE_NB][SQUARE_NB];
  ContinuationCorrectionHistory continuationCorrectionHistory[PIECE_NB][SQUARE_NB];
  
  TTMoveHistory ttMoveHistory;
  KillerMoves killers;
  CounterMoveHistory counterMoves;
  
  // Optimism values for evaluation blending
  Value optimism[COLOR_NB] = {VALUE_ZERO, VALUE_ZERO};
};

// Global search control
extern std::atomic<bool> Signals_stop;
extern std::atomic<bool> Signals_ponder;
extern TranspositionTable *TT;

void init();
void clear();

void set_thread_count(size_t count);
size_t thread_count();
Worker *main_thread();
Worker *best_thread();
uint64_t total_nodes();

void start_search(Position &pos, const LimitsType &limits, StateListPtr &states);
void stop_search();
void wait_for_search();
void clear_threads();

// Helper functions
inline Value value_to_tt(Value v, int ply) {
  return v >= VALUE_TB_WIN_IN_MAX_PLY  ? v + ply
       : v <= VALUE_TB_LOSS_IN_MAX_PLY ? v - ply
       : v;
}

inline Value value_from_tt(Value v, int ply, int r50c) {
  if (v == VALUE_NONE) return VALUE_NONE;
  if (v >= VALUE_TB_WIN_IN_MAX_PLY) {
    if (v >= VALUE_MATE_IN_MAX_PLY && VALUE_MATE - v > 100 - r50c)
      return VALUE_TB_WIN_IN_MAX_PLY - 1;
    if (VALUE_TB - v > 100 - r50c)
      return VALUE_TB_WIN_IN_MAX_PLY - 1;
    return v - ply;
  }
  if (v <= VALUE_TB_LOSS_IN_MAX_PLY) {
    if (v <= VALUE_MATED_IN_MAX_PLY && VALUE_MATE + v > 100 - r50c)
      return VALUE_TB_LOSS_IN_MAX_PLY + 1;
    if (VALUE_TB + v > 100 - r50c)
      return VALUE_TB_LOSS_IN_MAX_PLY + 1;
    return v + ply;
  }
  return v;
}

// =============================================================================
// Skill Level - Playing strength handicap
// =============================================================================
// Skill 0..19 covers CCRL Blitz Elo from 1320 to 3190
// Reference: Stockfish skill level implementation

struct Skill {
  static constexpr int LowestElo = 1320;
  static constexpr int HighestElo = 3190;

  Skill(int skillLevel, int uciElo) {
    if (uciElo) {
      double e = double(uciElo - LowestElo) / (HighestElo - LowestElo);
      level = std::clamp((((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438), 0.0, 19.0);
    } else {
      level = double(skillLevel);
    }
  }

  bool enabled() const { return level < 20.0; }
  bool time_to_pick(Depth depth) const { return depth == 1 + int(level); }
  Move pick_best(const RootMoves &rootMoves, size_t multiPV);

  double level = 20.0;
  Move best = Move::none();
};

} // namespace Search
} // namespace MetalFish

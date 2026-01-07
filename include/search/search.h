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
#include "search/movepick.h" // For history tables
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
  Move killers[2];
  Value staticEval;
  int statScore;
  int moveCount;
  bool inCheck;
  bool ttPv;
  bool ttHit;
  int cutoffCnt;
  PieceToHistory *continuationHistory; // For continuation heuristic
};

// RootMove stores information about root moves
struct RootMove {
  explicit RootMove(Move m) : pv(1, m) {}

  bool operator==(const Move &m) const { return pv[0] == m; }
  bool operator<(const RootMove &m) const {
    return m.score != score ? m.score < score : m.previousScore < previousScore;
  }

  Value score = -VALUE_INFINITE;
  Value previousScore = -VALUE_INFINITE;
  Value averageScore = -VALUE_INFINITE;
  int selDepth = 0;
  std::vector<Move> pv;

  // Tablebase information
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
  void start_searching(Position &pos, const LimitsType &limits,
                       StateListPtr &states);
  void wait_for_search_finished();

  bool is_main_thread() const { return threadIdx == 0; }

  // Search statistics
  std::atomic<uint64_t> nodes{0};
  std::atomic<uint64_t> tbHits{0};

  int selDepth = 0;
  int nmpMinPly = 0;
  int pvIdx = 0;  // Current MultiPV index being searched

  RootMoves rootMoves;
  Depth rootDepth;
  Depth completedDepth;

  // Search state
  Position *rootPos;
  LimitsType limits;
  StateListPtr *states;

  // Thread control
  std::atomic<bool> searching{false};
  std::atomic<bool> stopRequested{false};

private:
  void iterative_deepening();

  template <NodeType nodeType>
  Value search(Position &pos, Stack *ss, Value alpha, Value beta, Depth depth,
               bool cutNode);

  template <NodeType nodeType>
  Value qsearch(Position &pos, Stack *ss, Value alpha, Value beta);

  Value evaluate(const Position &pos);

  // Update history on beta cutoff
  void update_quiet_stats(Stack *ss, Move move, int bonus);
  void update_capture_stats(Piece piece, Square to, PieceType captured,
                            int bonus);

  size_t threadIdx;
  TimeManager timeManager;

  // Reductions table
  int reductions[MAX_MOVES];

  // History tables for move ordering
  ButterflyHistory mainHistory;
  KillerMoves killers;
  CounterMoveHistory counterMoves;
  CapturePieceToHistory captureHistory;
  PawnHistory pawnHistory;
  CorrectionHistory correctionHistory;

  // Continuation history: indexed by [piece][to], for tracking move sequences
  PieceToHistory continuationHistoryTable[PIECE_NB][SQUARE_NB];

  // Low ply history: extra weight for moves near root
  LowPlyHistory lowPlyHistory;
};

// Global search control
extern std::atomic<bool> Signals_stop;
extern std::atomic<bool> Signals_ponder;
extern TranspositionTable *TT;

void init();
void clear();

// Thread management
void set_thread_count(size_t count);
size_t thread_count();
Worker *main_thread();
Worker *best_thread();
uint64_t total_nodes();

// Search control
void start_search(Position &pos, const LimitsType &limits,
                  StateListPtr &states);
void stop_search();
void wait_for_search();
void clear_threads();

} // namespace Search

} // namespace MetalFish

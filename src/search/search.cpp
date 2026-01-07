/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "search/search.h"
#include "core/movegen.h"
#include "eval/evaluate.h"
#include "search/tt.h"
#include "search/movepick.h"
#include "uci/uci.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <array>

namespace MetalFish {

namespace Search {

std::atomic<bool> Signals_stop{false};
std::atomic<bool> Signals_ponder{false};
TranspositionTable *TT = &MetalFish::TT;

namespace {

// Futility margin
Value futility_margin(Depth d, bool noTtCutNode, bool improving,
                      bool oppWorsening) {
  Value futilityMult = 118 - 45 * noTtCutNode;
  Value improvingDeduction = 58 * improving * futilityMult / 32;
  Value worseningDeduction = 19 * oppWorsening * futilityMult / 32;
  return futilityMult * d - improvingDeduction - worseningDeduction;
}

// Late move reductions table
int Reductions[MAX_MOVES];

// Add a small random component to draw evaluations to avoid 3-fold blindness
// This prevents the engine from being overly repetition-prone
Value value_draw(uint64_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }

// Detect shuffling moves (repetitive back-and-forth moves)
// Used to reduce extension for shuffling positions
bool is_shuffling(Move move, Stack *ss, const Position &pos) {
  if (pos.capture(move) || pos.rule50_count() < 10)
    return false;
  if (pos.state()->pliesFromNull <= 6 || ss->ply < 20)
    return false;
  return move.from_sq() == (ss - 2)->currentMove.to_sq() &&
         (ss - 2)->currentMove.from_sq() == (ss - 4)->currentMove.to_sq();
}

void init_reductions() {
  Reductions[0] = 0;
  for (int i = 1; i < MAX_MOVES; ++i)
    Reductions[i] = int(19.41 * std::log(i));
}

} // anonymous namespace

void init() { init_reductions(); }

void clear() { TT->clear(); }

// Time manager implementation
void TimeManager::init(const LimitsType &limits, Color us, int ply) {
  startTime = now();

  TimePoint moveOverhead = 30; // Default move overhead

  if (limits.movetime) {
    optimumTime = limits.movetime - moveOverhead;
    maximumTime = limits.movetime - moveOverhead;
    return;
  }

  if (limits.time[us] == 0) {
    optimumTime = maximumTime = 0;
    return;
  }

  TimePoint time = limits.time[us];
  TimePoint inc = limits.inc[us];
  int movestogo = limits.movestogo ? limits.movestogo : 40;

  // Calculate time budget
  TimePoint timeLeft =
      std::max(TimePoint(1),
               time + inc * (movestogo - 1) - moveOverhead * (2 + movestogo));

  double mtg = movestogo;
  double optScale = std::min(0.0120 + std::pow(ply + 2.00, 0.400) * 0.0039,
                             0.2 * time / double(timeLeft)) *
                    1.15;
  double maxScale = std::min(6.5, 1.5 + 0.11 * mtg);

  optimumTime = TimePoint(optScale * timeLeft);
  maximumTime =
      TimePoint(std::min(0.825 * time - moveOverhead, maxScale * optimumTime));
}

void TimeManager::adjust_time(double ratio) {
  optimumTime = TimePoint(optimumTime * ratio);
}

// Worker constructor
Worker::Worker(size_t idx) : threadIdx(idx) {
  // Initialize history tables
  std::memset(mainHistory, 0, sizeof(mainHistory));
  std::memset(lowPlyHistory, 0, sizeof(lowPlyHistory));
  std::memset(captureHistory, 0, sizeof(captureHistory));
  std::memset(pawnHistory, 0, sizeof(pawnHistory));
  
  // Allocate correction history on heap
  correctionHistory = std::make_unique<UnifiedCorrectionHistory>();
  correctionHistory->clear();
  
  std::memset(continuationHistoryTable, 0, sizeof(continuationHistoryTable));
  std::memset(continuationCorrectionHistory, 0, sizeof(continuationCorrectionHistory));
  std::memset(counterMoves, 0, sizeof(counterMoves));
  ttMoveHistory = 0;

  for (int i = 0; i < MAX_MOVES; ++i)
    reductions[i] = int(19.41 * std::log(i + 1));
}

Worker::~Worker() {}

void Worker::clear() {
  nodes = 0;
  tbHits = 0;
  bestMoveChanges = 0;
  selDepth = 0;
  nmpMinPly = 0;
  pvIdx = 0;
  pvLast = 0;
  rootMoves.clear();
  rootDepth = 0;
  completedDepth = 0;
  rootDelta = VALUE_INFINITE;

  std::memset(mainHistory, 0, sizeof(mainHistory));
  std::memset(lowPlyHistory, 0, sizeof(lowPlyHistory));
  std::memset(captureHistory, 0, sizeof(captureHistory));
  std::memset(pawnHistory, 0, sizeof(pawnHistory));
  if (correctionHistory)
    correctionHistory->clear();
  std::memset(continuationHistoryTable, 0, sizeof(continuationHistoryTable));
  std::memset(continuationCorrectionHistory, 0, sizeof(continuationCorrectionHistory));
  std::memset(counterMoves, 0, sizeof(counterMoves));
  ttMoveHistory = 0;
  killers.clear();
}

// Update quiet move history on beta cutoff
void Worker::update_quiet_stats(Stack *ss, Move move, int bonus) {
  bonus = std::clamp(bonus, -1200, 1200);
  Color us = rootPos->side_to_move();
  Piece movedPiece = rootPos->moved_piece(move);
  
  // Update main history
  int idx = move.from_sq() * 64 + move.to_sq();
  history_update(mainHistory[us][idx], bonus);

  // Update low ply history
  if (ss->ply < LOW_PLY_HISTORY_SIZE)
    history_update(lowPlyHistory[ss->ply][idx], bonus * 805 / 1024);

  // Update continuation histories
  update_continuation_histories(ss, movedPiece, move.to_sq(), bonus * 896 / 1024);

  // Update pawn history
  int pIdx = pawn_history_index(*rootPos);
  history_update(pawnHistory[pIdx][movedPiece][move.to_sq()], bonus);

  // Update killer moves
  killers.update(ss->ply, move);

  // Update counter moves
  if (ss->ply >= 1 && (ss - 1)->currentMove.is_ok()) {
    Piece prevPiece = rootPos->piece_on((ss - 1)->currentMove.to_sq());
    if (prevPiece != NO_PIECE) {
      counterMoves[prevPiece][(ss - 1)->currentMove.to_sq()] = move;
    }
  }
}

// Update continuation histories with Stockfish's weighted bonuses
void Worker::update_continuation_histories(Stack *ss, Piece pc, Square to, int bonus) {
  // Stockfish continuation history weights: {ply_offset, weight}
  static constexpr std::array<std::pair<int, int>, 6> conthist_bonuses = {{
    {1, 1133}, {2, 683}, {3, 312}, {4, 582}, {5, 149}, {6, 474}
  }};

  for (const auto& [i, weight] : conthist_bonuses) {
    // Only update the first 2 continuation histories if we are in check
    if (ss->inCheck && i > 2)
      break;

    if (ss->ply >= i && (ss - i)->currentMove.is_ok() && (ss - i)->continuationHistory) {
      int weightedBonus = (bonus * weight / 1024) + 88 * (i < 2);
      history_update((*(ss - i)->continuationHistory)[pc][to], weightedBonus);
    }
  }
}

// Update capture history on beta cutoff
void Worker::update_capture_stats(Piece piece, Square to, PieceType captured, int bonus) {
  bonus = std::clamp(bonus, -1200, 1200);
  history_update(captureHistory[piece][to][captured], bonus);
}

// Update quiet histories (matching Stockfish)
void Worker::update_quiet_histories(const Position &pos, Stack *ss, Move move, int bonus) {
  Color us = pos.side_to_move();
  int idx = move.from_sq() * 64 + move.to_sq();
  
  // Update main history
  history_update(mainHistory[us][idx], bonus);
  
  // Update low ply history
  if (ss->ply < LOW_PLY_HISTORY_SIZE)
    history_update(lowPlyHistory[ss->ply][idx], bonus * 805 / 1024);
  
  // Update continuation histories
  update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(), bonus * 896 / 1024);
  
  // Update pawn history with asymmetric bonus (higher for positive)
  int pIdx = pawn_history_index(pos);
  history_update(pawnHistory[pIdx][pos.moved_piece(move)][move.to_sq()], 
                 bonus * (bonus > 0 ? 905 : 505) / 1024);
}

// Update all statistics after a move (matching Stockfish)
void Worker::update_all_stats(const Position &pos, Stack *ss, Move bestMove, Square prevSq,
                              SearchedList &quietsSearched, SearchedList &capturesSearched,
                              Depth depth, Move ttMove, int moveCount) {
  
  Piece movedPiece = pos.moved_piece(bestMove);
  
  // Calculate bonus and malus based on depth and other factors
  int bonus = std::min(116 * depth - 81, 1515) + 347 * (bestMove == ttMove) + 
              (ss - 1)->statScore / 32;
  int malus = std::min(848 * depth - 207, 2446) - 17 * moveCount;
  
  if (!pos.capture(bestMove)) {
    // Best move is quiet - update quiet histories
    update_quiet_histories(pos, ss, bestMove, bonus * 910 / 1024);
    
    // Update killer and counter moves
    killers.update(ss->ply, bestMove);
    if (ss->ply >= 1 && (ss - 1)->currentMove.is_ok()) {
      Piece prevPiece = pos.piece_on((ss - 1)->currentMove.to_sq());
      if (prevPiece != NO_PIECE)
        counterMoves[prevPiece][(ss - 1)->currentMove.to_sq()] = bestMove;
    }
    
    // Decrease stats for all non-best quiet moves
    int i = 0;
    for (Move move : quietsSearched) {
      i++;
      int actualMalus = malus * 1085 / 1024;
      if (i > 5)
        actualMalus -= actualMalus * (i - 5) / i;
      update_quiet_histories(pos, ss, move, -actualMalus);
    }
  } else {
    // Best move is capture - update capture history
    PieceType capturedPiece = type_of(pos.piece_on(bestMove.to_sq()));
    history_update(captureHistory[movedPiece][bestMove.to_sq()][capturedPiece], 
                   bonus * 1395 / 1024);
  }
  
  // Extra penalty for a quiet early move that was not a TT move in
  // previous ply when it gets refuted
  if (prevSq != SQ_NONE && ((ss - 1)->moveCount == 1 + (ss - 1)->ttHit) && 
      !pos.captured_piece()) {
    Piece pc = pos.piece_on(prevSq);
    if (pc != NO_PIECE)
      update_continuation_histories(ss - 1, pc, prevSq, -malus * 602 / 1024);
  }
  
  // Decrease stats for all non-best capture moves
  for (Move move : capturesSearched) {
    Piece mp = pos.moved_piece(move);
    PieceType capturedPiece = type_of(pos.piece_on(move.to_sq()));
    history_update(captureHistory[mp][move.to_sq()][capturedPiece], -malus * 1448 / 1024);
  }
}

// Start searching
void Worker::start_searching(Position &pos, const LimitsType &lim,
                             StateListPtr &st) {
  rootPos = &pos;
  limits = lim;
  states = &st;

  searching = true;
  stopRequested = false;

  // Initialize time manager
  timeManager.init(limits, pos.side_to_move(), pos.game_ply());

  // Generate root moves
  rootMoves.clear();
  for (const auto &m : MoveList<LEGAL>(pos)) {
    if (limits.searchmoves.empty() ||
        std::find(limits.searchmoves.begin(), limits.searchmoves.end(), m) !=
            limits.searchmoves.end())
      rootMoves.emplace_back(m);
  }

  // Start search
  iterative_deepening();

  searching = false;
}

void Worker::wait_for_search_finished() {
  while (searching.load())
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// Iterative deepening loop
void Worker::iterative_deepening() {
  Move pv[MAX_PLY + 1];
  Stack stack[MAX_PLY + 10], *ss = stack + 7;

  std::memset(ss - 7, 0, 10 * sizeof(Stack));

  for (int i = 0; i <= MAX_PLY + 2; ++i) {
    (ss + i)->ply = i;
    (ss + i)->continuationHistory =
        &continuationHistoryTable[NO_PIECE][SQ_A1]; // Default empty
    (ss + i)->continuationCorrectionHistory =
        &continuationCorrectionHistory[NO_PIECE][SQ_A1];
    (ss + i)->cutoffCnt = 0;
  }

  // Set up initial continuation history for root
  for (int i = -7; i < 0; ++i) {
    (ss + i)->continuationHistory = &continuationHistoryTable[NO_PIECE][SQ_A1];
    (ss + i)->continuationCorrectionHistory = &continuationCorrectionHistory[NO_PIECE][SQ_A1];
  }

  ss->pv = pv;

  TT->new_search();

  Value bestValue = -VALUE_INFINITE;
  Value alpha = -VALUE_INFINITE;
  Value beta = VALUE_INFINITE;
  int delta = 0;
  int failedHighCnt = 0;

  // Best move stability tracking for time management
  Move lastBestMove = Move::none();
  int bestMoveChanges = 0;
  int stableBestMoveCount = 0;
  Value lastScore = -VALUE_INFINITE;

  // MultiPV support
  int multiPV = std::min(limits.multiPV, int(rootMoves.size()));
  multiPV = std::max(1, multiPV);

  // Iterative deepening
  int searchAgainCounter = 0;
  bool increaseDepth = true;
  
  for (rootDepth = 1; rootDepth <= (limits.depth ? limits.depth : MAX_PLY);
       ++rootDepth) {
    // Check for stop conditions
    if (stopRequested || Signals_stop)
      break;

    // Save previous scores for all root moves
    for (auto &rm : rootMoves)
      rm.previousScore = rm.score;
    
    // Hindsight depth adjustment: if time allows, search again at same depth
    if (!increaseDepth)
      searchAgainCounter++;

    // MultiPV loop
    for (pvIdx = 0; pvIdx < multiPV && !stopRequested && !Signals_stop;
         ++pvIdx) {
      // Reset fail high counter and aspiration window for new depth
      failedHighCnt = 0;
      selDepth = 0;

      // Aspiration window search
      if (rootDepth >= 4) {
        // Use average score for more stable aspiration windows
        Value avg = rootMoves[pvIdx].averageScore != -VALUE_INFINITE
                        ? rootMoves[pvIdx].averageScore
                        : (pvIdx == 0 ? bestValue : rootMoves[pvIdx].score);
        
        // Use meanSquaredScore for delta calculation (matching Stockfish)
        int64_t mss = rootMoves[pvIdx].meanSquaredScore;
        delta = 5 + threadIdx % 8 + std::abs(mss) / 9000;
        alpha = std::max(avg - delta, -VALUE_INFINITE);
        beta = std::min(avg + delta, VALUE_INFINITE);
        
        // Calculate optimism based on average score (matching Stockfish)
        Color us = rootPos->side_to_move();
        optimism[us] = 142 * avg / (std::abs(avg) + 91);
        optimism[~us] = -optimism[us];
      } else {
        alpha = -VALUE_INFINITE;
        beta = VALUE_INFINITE;
        // Reset optimism for shallow depths
        optimism[WHITE] = optimism[BLACK] = VALUE_ZERO;
      }

      // Main search with hindsight depth adjustment
      while (true) {
        // Adjust depth based on fail-high count and searchAgain counter
        // Ensure at least one effective increment for every four searchAgain steps
        Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt - 
                                       3 * (searchAgainCounter + 1) / 4);
        bestValue =
            search<Root>(*rootPos, ss, alpha, beta, adjustedDepth, false);

        // Sort root moves by score (only moves from pvIdx onwards)
        std::stable_sort(rootMoves.begin() + pvIdx, rootMoves.end());

        if (stopRequested || Signals_stop)
          break;

        // Aspiration window handling with improved fail-high logic
        if (bestValue <= alpha) {
          // Fail low: expand window downward, reset fail high count
          beta = (alpha + beta) / 2;
          alpha = std::max(bestValue - delta, -VALUE_INFINITE);
          failedHighCnt = 0;
        } else if (bestValue >= beta) {
          // Fail high: expand window upward, increment fail high count
          alpha = std::max(beta - delta, alpha);
          beta = std::min(bestValue + delta, VALUE_INFINITE);
          ++failedHighCnt;
        } else
          break;

        delta += delta / 3;
      }

      // Update average score (exponential moving average) and mean squared score
      if (pvIdx < int(rootMoves.size())) {
        Value prevAvg = rootMoves[pvIdx].averageScore;
        if (prevAvg == -VALUE_INFINITE)
          rootMoves[pvIdx].averageScore = bestValue;
        else
          rootMoves[pvIdx].averageScore = (prevAvg * 2 + bestValue) / 3;
        
        // Update mean squared score for aspiration window sizing
        int64_t diff = bestValue - rootMoves[pvIdx].averageScore;
        rootMoves[pvIdx].meanSquaredScore = 
            (rootMoves[pvIdx].meanSquaredScore * 3 + diff * diff) / 4;
      }

      // Output search info for this PV line
      if (is_main_thread() && pvIdx < int(rootMoves.size())) {
        TimePoint elapsed = timeManager.elapsed();
        uint64_t nodeCount = nodes.load();

        std::cout << "info"
                  << " depth " << rootDepth << " seldepth " << selDepth;

        if (multiPV > 1)
          std::cout << " multipv " << (pvIdx + 1);

        std::cout << " score cp " << rootMoves[pvIdx].score << " nodes "
                  << nodeCount
                  << " nps " << (elapsed > 0 ? nodeCount * 1000 / elapsed : 0)
                  << " time " << elapsed << " hashfull " << TT->hashfull()
                  << " pv";

        for (const auto &m : rootMoves[pvIdx].pv)
          std::cout << " " << UCI::move_to_uci(m, false);

        std::cout << std::endl;
      }
    }

    completedDepth = rootDepth;

    // Track best move stability for time management (only for first PV)
    if (!rootMoves.empty()) {
      Move currentBestMove = rootMoves[0].pv[0];
      if (currentBestMove != lastBestMove) {
        lastBestMove = currentBestMove;
        bestMoveChanges++;
        stableBestMoveCount = 0;
      } else {
        stableBestMoveCount++;
      }

      // Track score changes (falling eval)
      Value currentScore = rootMoves[0].score;
      Value scoreDiff = currentScore - lastScore;
      lastScore = currentScore;

      // Time management with stability and falling eval considerations
      // Skip time management in ponder mode
      if (!limits.infinite && !limits.ponderMode && limits.use_time_management()) {
        TimePoint elapsed = timeManager.elapsed();
        TimePoint optimum = timeManager.optimum();
        TimePoint maximum = timeManager.maximum();

        // Decide whether to increase depth or search again at same depth
        // Based on Stockfish: increaseDepth = ponder || elapsed <= totalTime * 0.50
        increaseDepth = limits.ponderMode || elapsed <= maximum * 0.50;
        
        // Calculate effort-based adjustment (Stockfish formula)
        uint64_t totalNodes = nodes.load();
        uint64_t nodesEffort = rootMoves[0].effort * 100000 / std::max(uint64_t(1), totalNodes);
        double highBestMoveEffort = nodesEffort >= 93340 ? 0.76 : 1.0;

        // Early termination if best move is stable and score is not falling
        if (stableBestMoveCount >= 3 && scoreDiff >= -20) {
          // Reduce time if very stable (adjust by 0.8x)
          if (elapsed > optimum * 0.8 * highBestMoveEffort)
            break;
        }

        // Extend time if best move changed recently or score is falling
        double timeScale = 1.0;
        if (bestMoveChanges > 2 && stableBestMoveCount < 2)
          timeScale = 1.3; // Unstable, use more time
        else if (scoreDiff < -50)
          timeScale = 1.2; // Falling eval, use more time
        else if (stableBestMoveCount >= 5)
          timeScale = 0.75; // Very stable, use less time

        if (elapsed > optimum * timeScale * highBestMoveEffort)
          break;
      }
    }

    // Node limit
    if (limits.nodes && nodes.load() >= limits.nodes)
      break;
  }
}

// Main search function
template <NodeType nodeType>
Value Worker::search(Position &pos, Stack *ss, Value alpha, Value beta,
                     Depth depth, bool cutNode) {
  constexpr bool PvNode = nodeType != NonPV;
  constexpr bool rootNode = nodeType == Root;

  // Check for stop conditions
  if (stopRequested || Signals_stop)
    return VALUE_ZERO;

  // Drop into quiescence search at horizon
  if (depth <= 0)
    return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta);

  // Node count
  nodes++;

  // Check node limit and time limit every 64 nodes for efficiency
  if ((nodes.load() & 63) == 0) {
    if (limits.nodes && nodes.load() >= limits.nodes) {
      stopRequested = true;
      return VALUE_ZERO;
    }
    // Check time limit
    if (limits.use_time_management() && timeManager.elapsed() >= timeManager.maximum()) {
      stopRequested = true;
      return VALUE_ZERO;
    }
  }

  // Update selective depth
  if (PvNode && ss->ply > selDepth)
    selDepth = ss->ply;

  // Draw detection with randomization to avoid 3-fold blindness
  if (!rootNode && pos.is_draw(ss->ply))
    return value_draw(nodes);

  // Check if we have an upcoming move that draws by repetition
  // If so, adjust alpha to avoid repetition blindness
  if (!rootNode && alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply)) {
    alpha = value_draw(nodes);
    if (alpha >= beta)
      return alpha;
  }

  // Max ply check
  if (ss->ply >= MAX_PLY)
    return pos.checkers() ? value_draw(nodes) : evaluate(pos);

  // Reset cutoffCnt for child nodes
  (ss + 2)->cutoffCnt = 0;

  // Mate distance pruning
  // Even if we mate at the next move, our score would be at best mate_in(ss->ply+1).
  // If alpha is already bigger because a shorter mate was found upward in the tree,
  // there is no need to search because we will never beat current alpha.
  if (!rootNode) {
    alpha = std::max(mated_in(ss->ply), alpha);
    beta = std::min(mate_in(ss->ply + 1), beta);
    if (alpha >= beta)
      return alpha;
  }

  // Transposition table lookup
  Key posKey = pos.key();
  bool ttHit;
  TTEntry *tte = TT->probe(posKey, ttHit);
  Value ttValue = ttHit ? value_from_tt(tte->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
  Move ttMove = ttHit ? tte->move() : Move::none();

  ss->ttHit = ttHit;
  ss->ttPv = PvNode || (ttHit && tte->is_pv());

  // TT cutoff (not in PV nodes)
  // For high rule50 counts don't produce transposition table cutoffs
  if (!PvNode && ttHit && tte->depth() >= depth && pos.rule50_count() < 90) {
    if (ttValue >= beta ? tte->bound() & BOUND_LOWER
                        : tte->bound() & BOUND_UPPER)
      return ttValue;
  }

  // Internal Iterative Reductions (IIR)
  // Reduce depth for PV/Cut nodes without a TT move
  if (depth >= 4 && !ttMove && !pos.checkers())
    depth--;

  // Static evaluation
  Value eval;
  bool improving;
  
  // Track if we're in check for continuation history updates
  ss->inCheck = pos.checkers();

  if (ss->inCheck) {
    ss->staticEval = eval = VALUE_NONE;
    improving = false;
  } else {
    if (ttHit) {
      ss->staticEval = eval = tte->eval();
      if (eval == VALUE_NONE)
        ss->staticEval = eval = evaluate(pos);
    } else {
      ss->staticEval = eval = evaluate(pos);
    }

    // Apply full correction history adjustment (matching Stockfish)
    Move prevMove = ss->ply >= 1 ? (ss - 1)->currentMove : Move::none();
    ContinuationCorrectionHistory *contCorr2 = ss->ply >= 2 ? (ss - 2)->continuationCorrectionHistory : nullptr;
    ContinuationCorrectionHistory *contCorr4 = ss->ply >= 4 ? (ss - 4)->continuationCorrectionHistory : nullptr;
    
    int correctionValue = compute_full_correction_value(
        *correctionHistory, contCorr2, contCorr4, pos, prevMove);
    eval = to_corrected_static_eval(eval, correctionValue);
    ss->staticEval = eval;

    improving = ss->ply >= 2 && ss->staticEval > (ss - 2)->staticEval;
  }

  // Razoring
  if (!PvNode && depth <= 4 && eval + 290 * depth <= alpha)
    return qsearch<NonPV>(pos, ss, alpha, beta);

  // Futility pruning
  if (!PvNode && !pos.checkers() && depth < 11 && eval >= beta &&
      eval - futility_margin(depth, cutNode, improving, false) >= beta)
    return eval;

  // Null move pruning
  if (!PvNode && !pos.checkers() && eval >= beta && eval >= ss->staticEval &&
      ss->ply >= nmpMinPly && pos.non_pawn_material(pos.side_to_move())) {

    Depth R = std::min(int(eval - beta) / 155, 6) + depth / 3 + 4;

    StateInfo st;
    pos.do_null_move(st);

    Value nullValue =
        -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, !cutNode);

    pos.undo_null_move();

    if (nullValue >= beta)
      return nullValue < VALUE_TB_WIN_IN_MAX_PLY ? nullValue : beta;
  }

  // ProbCut
  // If we have a good enough capture and a reduced search returns a value
  // much above beta, we can prune the previous move safely.
  Value probCutBeta = beta + 200 - 50 * improving;
  if (!PvNode && depth >= 4 && !pos.checkers() &&
      std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY) {

    Move probCutMoves[MAX_MOVES];
    Move *probCutEnd = generate<CAPTURES>(pos, probCutMoves);

    for (Move *pm = probCutMoves; pm != probCutEnd; ++pm) {
      Move probCutMove = *pm;

      if (!pos.legal(probCutMove))
        continue;

      // Skip captures with bad SEE
      if (!pos.see_ge(probCutMove, probCutBeta - ss->staticEval))
        continue;

      StateInfo st;
      pos.do_move(probCutMove, st);

      // Verify with qsearch first
      Value value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

      // If qsearch doesn't refute, do a reduced search
      if (value >= probCutBeta)
        value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1,
                               depth - 4, !cutNode);

      pos.undo_move(probCutMove);

      if (value >= probCutBeta)
        return value > VALUE_TB_WIN_IN_MAX_PLY ? probCutBeta : value;
    }
  }

  // Move generation and ordering
  Move pv[MAX_PLY + 1];
  ss->pv = pv;
  pv[0] = Move::none();

  Move move;
  int moveCount = 0;
  Value bestValue = -VALUE_INFINITE;
  Move bestMove = Move::none();
  
  // Track searched moves for history updates
  SearchedList quietsSearched;
  SearchedList capturesSearched;
  Square prevSq = ss->ply >= 1 && (ss - 1)->currentMove.is_ok() 
                  ? (ss - 1)->currentMove.to_sq() : SQ_NONE;

  // Generate moves - at root, use rootMoves; otherwise generate all
  Move moves[MAX_MOVES];
  Move *end;
  
  if (rootNode) {
    // At root, only search moves in rootMoves
    end = moves;
    for (const auto &rm : rootMoves) {
      if (!rm.pv.empty())
        *end++ = rm.pv[0];
    }
  } else {
    end = pos.checkers() ? generate<EVASIONS>(pos, moves)
                         : generate<NON_EVASIONS>(pos, moves);
  }

  // Move ordering: TT move first
  if (ttMove != Move::none()) {
    for (Move *m = moves; m != end; ++m) {
      if (*m == ttMove) {
        std::swap(*m, moves[0]);
        break;
      }
    }
  }

  // Search moves
  uint64_t prevNodes = nodes.load(); // For effort tracking at root
  for (Move *m = moves; m != end; ++m) {
    move = *m;

    // Skip excluded move (for singular extension search)
    if (move == ss->excludedMove)
      continue;

    // For MultiPV at root, skip moves already searched in previous PVs
    if (rootNode && pvIdx > 0) {
      bool skipMove = false;
      for (int i = 0; i < pvIdx; ++i) {
        if (!rootMoves[i].pv.empty() && rootMoves[i].pv[0] == move) {
          skipMove = true;
          break;
        }
      }
      if (skipMove)
        continue;
    }

    // Root moves are already legal; non-root need legality check
    if (!rootNode && !pos.legal(move))
      continue;

    moveCount++;

    bool isCapture = pos.capture(move);
    bool givesCheck = pos.gives_check(move);

    // Pruning at shallow depths
    if (!rootNode && bestValue > VALUE_MATED_IN_MAX_PLY &&
        pos.non_pawn_material(pos.side_to_move())) {

      // Late move pruning: skip quiet moves after enough have been searched
      if (depth <= 6 && moveCount > (3 + depth * depth) / (2 - improving))
        continue;

      // Reduced depth for pruning decisions
      int lmrDepth = std::max(
          0, depth - 1 - reductions[std::min(moveCount, MAX_MOVES - 1)] / 16);

      if (isCapture) {
        // Futility pruning for captures
        if (!givesCheck && lmrDepth < 6) {
          Piece captured = pos.piece_on(move.to_sq());
          PieceType capturedType =
              captured != NO_PIECE ? type_of(captured) : PAWN;
          Value futilityValue = ss->staticEval + 200 + 200 * lmrDepth +
                                PieceValue[make_piece(WHITE, capturedType)];
          if (futilityValue <= alpha)
            continue;
        }

        // SEE-based pruning for captures
        if (!pos.see_ge(move, -200 * depth))
          continue;
      } else {
        // Futility pruning for quiet moves
        if (!givesCheck && lmrDepth < 10 &&
            ss->staticEval + 100 + 150 * lmrDepth <= alpha)
          continue;

        // History-based pruning: skip quiet moves with very bad history
        if (!givesCheck && depth <= 5) {
          Piece movedPc = pos.moved_piece(move);
          if (movedPc != NO_PIECE) {
            int histIdx = move.from_sq() * 64 + move.to_sq();
            Color us = pos.side_to_move();
            int histScore = mainHistory[us][histIdx];

            // Prune moves with very negative history
            if (histScore < -4000 * depth)
              continue;

            // Also check pawn history
            int pawnIdx = pawn_history_index(pos);
            int pawnHistScore = pawnHistory[pawnIdx][movedPc][move.to_sq()];
            if (histScore + pawnHistScore < -6000 * depth)
              continue;
          }
        }

        // SEE-based pruning for quiet moves
        if (!pos.see_ge(move, -30 * lmrDepth * lmrDepth))
          continue;
      }
    }

    // Extensions
    Depth extension = 0;

    // Singular extension search
    // If the TT move is significantly better than all other moves,
    // extend its search
    if (!rootNode && depth >= 6 && move == ttMove && ttHit &&
        ss->excludedMove == Move::none() && (tte->bound() & BOUND_LOWER) &&
        tte->depth() >= depth - 3 && std::abs(ttValue) < VALUE_TB_WIN_IN_MAX_PLY) {

      Value singularBeta = ttValue - 2 * depth;
      Depth singularDepth = (depth - 1) / 2;

      ss->excludedMove = move;
      Value singularValue =
          search<NonPV>(pos, ss, singularBeta - 1, singularBeta, singularDepth,
                        cutNode);
      ss->excludedMove = Move::none();

      if (singularValue < singularBeta) {
        extension = 1;
        
        // Double extension if singularValue is much lower
        // But not if we're shuffling
        if (!is_shuffling(move, ss, pos) && singularValue < singularBeta - 20)
          extension = 2;
      } else if (singularBeta >= beta) {
        // Multi-cut pruning: singular beta >= beta means all moves fail high
        return singularBeta;
      } else if (ttValue >= beta) {
        // Negative extension if TT value >= beta but no singular move
        extension = -1;
      } else if (cutNode) {
        // Reduce extension in cut nodes
        extension = -1;
      }
    }

    // Make the move
    ss->currentMove = move;

    // Set up continuation history for this ply
    Piece movedPc = pos.moved_piece(move);
    (ss + 1)->continuationHistory =
        &continuationHistoryTable[movedPc][move.to_sq()];
    (ss + 1)->continuationCorrectionHistory =
        &continuationCorrectionHistory[movedPc][move.to_sq()];

    StateInfo st;
    pos.do_move(move, st, givesCheck);

    // Check extension - extend when giving check (if not already extended)
    if (givesCheck && extension <= 0)
      extension = 1;

    // Passed pawn extension - extend for pawns pushing to 7th rank
    if (extension <= 0 && type_of(pos.moved_piece(move)) == PAWN) {
      Rank r = relative_rank(pos.side_to_move(), rank_of(move.to_sq()));
      if (r == RANK_7)
        extension = 1;
    }

    // New depth with extension
    Depth newDepth = depth - 1 + extension;
    Value value;

    if (depth >= 2 && moveCount > 1 + (PvNode ? 1 : 0)) {
      int reduction =
          reductions[std::min(moveCount, MAX_MOVES - 1)] * depth / 16;

      // Factor 1: Decrease for checks
      if (givesCheck)
        reduction--;

      // Factor 2: Decrease for captures
      if (isCapture)
        reduction--;

      // Factor 3: Decrease for killer moves
      if (killers.is_killer(ss->ply, move))
        reduction--;

      // Factor 4: Increase for cut nodes
      if (cutNode)
        reduction += 2;

      // Factor 5: Increase for non-improving positions
      if (!improving)
        reduction++;

      // Factor 6: Decrease for TT move
      if (move == ttMove)
        reduction--;

      // Factor 7: Decrease for PV nodes
      if (PvNode)
        reduction--;

      // Factor 8: Increase for quiet moves after many moves
      if (!isCapture && moveCount > 4)
        reduction++;

      // Factor 9: Decrease for pawn pushes to 6th/7th rank
      if (!isCapture && type_of(pos.moved_piece(move)) == PAWN) {
        Rank r = relative_rank(pos.side_to_move(), rank_of(move.to_sq()));
        if (r >= RANK_6)
          reduction -= 2;
        else if (r >= RANK_5)
          reduction--;
      }

      // Factor 10: Decrease for TT PV moves
      if (ss->ttPv)
        reduction--;

      // Factor 11: Increase if next ply has a lot of fail highs (cutoffCnt)
      if ((ss + 1)->cutoffCnt > 1)
        reduction += 1 + ((ss + 1)->cutoffCnt > 2) + ((ss + 1)->cutoffCnt > 3);

      // Factor 11: Decrease for promotions
      if (move.type_of() == PROMOTION)
        reduction -= 2;

      // Factor 12: Increase for very late moves
      if (moveCount > 10)
        reduction++;

      // Factor 13: Decrease for moves to central squares
      {
        File f = file_of(move.to_sq());
        Rank r = rank_of(move.to_sq());
        if (f >= FILE_C && f <= FILE_F && r >= RANK_3 && r <= RANK_6)
          reduction--;
      }

      // Factor 14: Decrease for recaptures (capturing on same square as previous)
      if (isCapture && ss->ply >= 1 && (ss - 1)->currentMove.is_ok() &&
          move.to_sq() == (ss - 1)->currentMove.to_sq())
        reduction--;

      reduction = std::max(0, std::min(reduction, newDepth - 1));

      value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha,
                             newDepth - reduction, true);

      if (value > alpha && reduction > 0)
        value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth,
                               !cutNode);
    } else {
      value = alpha + 1; // Force full search
    }

    // Full window search for PV nodes
    if (PvNode && (moveCount == 1 || value > alpha)) {
      (ss + 1)->pv = nullptr;
      value = -search<PV>(pos, ss + 1, -beta, -alpha, newDepth, false);
    }

    pos.undo_move(move);

    // Check for stop
    if (stopRequested || Signals_stop)
      return VALUE_ZERO;

    // Track searched moves for history updates (before checking if it improves)
    if (move != bestMove) {
      if (pos.capture(move)) {
        capturesSearched.push_back(move);
      } else {
        quietsSearched.push_back(move);
      }
    }

    // Update best value
    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        bestMove = move;

        if (PvNode && !rootNode) {
          // Update PV
          ss->pv[0] = move;
          int i = 1;
          if ((ss + 1)->pv)
            for (; (ss + 1)->pv[i - 1] != Move::none(); ++i)
              ss->pv[i] = (ss + 1)->pv[i - 1];
          ss->pv[i] = Move::none();
        }

        if (value >= beta) {
          // Beta cutoff - use update_all_stats for comprehensive history updates
          update_all_stats(pos, ss, move, prevSq, quietsSearched, capturesSearched,
                           depth, ttMove, moveCount);
          
          // Increment cutoffCnt for LMR adjustment in parent
          ss->cutoffCnt += (extension < 2) || PvNode;
          
          break;
        }

        alpha = value;
      }
    }
  }

  // Checkmate/stalemate detection
  if (moveCount == 0)
    bestValue = pos.checkers() ? mated_in(ss->ply) : value_draw(nodes);

  // Update full correction history when search result differs from static eval
  if (!pos.checkers() && bestMove && !pos.capture(bestMove) &&
      std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY) {
    int diff = bestValue - ss->staticEval;
    int bonus = std::clamp(diff * depth / 8, -CORRECTION_HISTORY_LIMIT / 4,
                           CORRECTION_HISTORY_LIMIT / 4);
    
    Move prevMove = ss->ply >= 1 ? (ss - 1)->currentMove : Move::none();
    ContinuationCorrectionHistory *contCorr2 = ss->ply >= 2 ? (ss - 2)->continuationCorrectionHistory : nullptr;
    ContinuationCorrectionHistory *contCorr4 = ss->ply >= 4 ? (ss - 4)->continuationCorrectionHistory : nullptr;
    
    update_full_correction_history(*correctionHistory, contCorr2, contCorr4, pos, prevMove, bonus);
  }

  // Update TT with adjusted value for storage
  if (!ss->excludedMove) {
    tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
              bestValue >= beta    ? BOUND_LOWER
              : PvNode && bestMove ? BOUND_EXACT
                                   : BOUND_UPPER,
              depth, bestMove, ss->staticEval, TT->generation());
  }

  // Update root move scores and effort
  if (rootNode) {
    // Find the move to update - use bestMove if found, otherwise first root move
    Move moveToUpdate = bestMove ? bestMove : (rootMoves.empty() ? Move::none() : rootMoves[pvIdx].pv[0]);
    
    if (moveToUpdate != Move::none()) {
      auto it = std::find(rootMoves.begin(), rootMoves.end(), moveToUpdate);
      if (it != rootMoves.end()) {
        RootMove &rm = *it;
        
        // Track effort (nodes spent on this move)
        rm.effort += nodes.load() - prevNodes;
        
        rm.score = bestValue;
        rm.selDepth = selDepth;

        if (bestMove) {
          rm.pv.clear();
          rm.pv.push_back(bestMove);
          for (int i = 0; ss->pv[i + 1] != Move::none(); ++i)
            rm.pv.push_back(ss->pv[i + 1]);
        }
      }
    }
  }

  return bestValue;
}

// Quiescence search
template <NodeType nodeType>
Value Worker::qsearch(Position &pos, Stack *ss, Value alpha, Value beta) {
  constexpr bool PvNode = nodeType != NonPV;

  nodes++;

  // Draw detection with randomization
  if (pos.is_draw(ss->ply))
    return value_draw(nodes);

  // Max ply
  if (ss->ply >= MAX_PLY)
    return pos.checkers() ? value_draw(nodes) : evaluate(pos);

  // TT probe
  Key posKey = pos.key();
  bool ttHit;
  TTEntry *tte = TT->probe(posKey, ttHit);
  Value ttValue = ttHit ? value_from_tt(tte->value(), ss->ply, pos.rule50_count()) : VALUE_NONE;
  Move ttMove = ttHit ? tte->move() : Move::none();
  (void)ttMove; // Unused for now, could be used for move ordering in qsearch

  // TT cutoff
  if (!PvNode && ttHit && tte->depth() >= DEPTH_QS && pos.rule50_count() < 90) {
    if (ttValue >= beta ? tte->bound() & BOUND_LOWER
                        : tte->bound() & BOUND_UPPER)
      return ttValue;
  }

  // Static evaluation
  Value bestValue;

  if (pos.checkers()) {
    ss->staticEval = VALUE_NONE;
    bestValue = -VALUE_INFINITE;
  } else {
    if (ttHit) {
      ss->staticEval = bestValue = tte->eval();
      if (bestValue == VALUE_NONE)
        ss->staticEval = bestValue = evaluate(pos);
    } else {
      ss->staticEval = bestValue = evaluate(pos);
    }

    if (bestValue >= beta)
      return bestValue;

    if (bestValue > alpha)
      alpha = bestValue;
  }

  // Generate captures or evasions
  Move moves[MAX_MOVES];
  Move *end = pos.checkers() ? generate<EVASIONS>(pos, moves)
                             : generate<CAPTURES>(pos, moves);

  Move bestMove = Move::none();

  // Search captures
  for (Move *m = moves; m != end; ++m) {
    Move move = *m;

    if (!pos.legal(move))
      continue;

    // Futility pruning for captures
    if (!pos.checkers() && !pos.see_ge(move))
      continue;

    StateInfo st;
    pos.do_move(move, st);

    Value value = -qsearch<nodeType>(pos, ss + 1, -beta, -alpha);

    pos.undo_move(move);

    if (value > bestValue) {
      bestValue = value;

      if (value > alpha) {
        bestMove = move;

        if (value >= beta)
          break;

        alpha = value;
      }
    }
  }

  // Checkmate detection
  if (pos.checkers() && bestValue == -VALUE_INFINITE)
    return mated_in(ss->ply);

  // Update TT with adjusted value for storage
  tte->save(posKey, value_to_tt(bestValue, ss->ply), false,
            bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
            ss->staticEval, TT->generation());

  return bestValue;
}

// Evaluate position using GPU-accelerated NNUE with optimism blending
Value Worker::evaluate(const Position &pos) { 
  Value rawEval = Eval::evaluate(pos);
  
  // Blend optimism with evaluation based on material
  int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
  int opt = optimism[pos.side_to_move()];
  
  // Stockfish formula: (nnue * (77871 + material) + optimism * (7191 + material)) / 77871
  Value v = (rawEval * (77871 + material) + opt * (7191 + material)) / 77871;
  
  // Damp down evaluation linearly when shuffling (rule50)
  v -= v * pos.rule50_count() / 199;
  
  // Guarantee evaluation does not hit tablebase range
  return std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
}

// Explicit template instantiations
template Value Worker::search<Root>(Position &, Stack *, Value, Value, Depth,
                                    bool);
template Value Worker::search<PV>(Position &, Stack *, Value, Value, Depth,
                                  bool);
template Value Worker::search<NonPV>(Position &, Stack *, Value, Value, Depth,
                                     bool);
template Value Worker::qsearch<PV>(Position &, Stack *, Value, Value);
template Value Worker::qsearch<NonPV>(Position &, Stack *, Value, Value);

} // namespace Search

} // namespace MetalFish

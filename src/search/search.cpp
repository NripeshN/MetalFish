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
#include "uci/uci.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

namespace MetalFish {

namespace Search {

std::atomic<bool> Signals_stop{false};
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
  std::memset(counterMoves, 0, sizeof(counterMoves));
  std::memset(captureHistory, 0, sizeof(captureHistory));

  for (int i = 0; i < MAX_MOVES; ++i)
    reductions[i] = Reductions[i];
}

Worker::~Worker() {}

void Worker::clear() {
  nodes = 0;
  tbHits = 0;
  selDepth = 0;
  nmpMinPly = 0;
  rootMoves.clear();
  rootDepth = 0;
  completedDepth = 0;

  // Clear history tables
  std::memset(mainHistory, 0, sizeof(mainHistory));
  std::memset(counterMoves, 0, sizeof(counterMoves));
  std::memset(captureHistory, 0, sizeof(captureHistory));
  killers.clear();
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

  for (int i = 0; i <= MAX_PLY + 2; ++i)
    (ss + i)->ply = i;

  ss->pv = pv;

  TT->new_search();

  Value bestValue = -VALUE_INFINITE;
  Value alpha = -VALUE_INFINITE;
  Value beta = VALUE_INFINITE;
  Value delta = 0;

  // Iterative deepening
  for (rootDepth = 1; rootDepth <= (limits.depth ? limits.depth : MAX_PLY);
       ++rootDepth) {
    // Check for stop conditions
    if (stopRequested || Signals_stop)
      break;

    // Aspiration window search
    if (rootDepth >= 4) {
      delta = 16 + bestValue * bestValue / 21367;
      alpha = std::max(bestValue - delta, -VALUE_INFINITE);
      beta = std::min(bestValue + delta, VALUE_INFINITE);
    }

    // Main search
    while (true) {
      bestValue = search<Root>(*rootPos, ss, alpha, beta, rootDepth, false);

      // Sort root moves by score
      std::stable_sort(rootMoves.begin(), rootMoves.end());

      if (stopRequested || Signals_stop)
        break;

      // Aspiration window handling
      if (bestValue <= alpha) {
        beta = (alpha + beta) / 2;
        alpha = std::max(bestValue - delta, -VALUE_INFINITE);
      } else if (bestValue >= beta) {
        beta = std::min(bestValue + delta, VALUE_INFINITE);
      } else
        break;

      delta += delta / 3;
    }

    completedDepth = rootDepth;

    // Output search info
    if (is_main_thread()) {
      TimePoint elapsed = timeManager.elapsed();
      uint64_t nodeCount = nodes.load();

      std::cout << "info"
                << " depth " << rootDepth << " seldepth " << selDepth
                << " score cp " << bestValue << " nodes " << nodeCount
                << " nps " << (elapsed > 0 ? nodeCount * 1000 / elapsed : 0)
                << " time " << elapsed << " hashfull " << TT->hashfull()
                << " pv";

      for (const auto &m : rootMoves[0].pv)
        std::cout << " " << UCI::move_to_uci(m, false);

      std::cout << std::endl;
    }

    // Time management
    if (!limits.infinite && limits.use_time_management()) {
      if (timeManager.elapsed() > timeManager.optimum())
        break;
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

  // Update selective depth
  if (PvNode && ss->ply > selDepth)
    selDepth = ss->ply;

  // Draw detection
  if (!rootNode && pos.is_draw(ss->ply))
    return VALUE_DRAW;

  // Max ply check
  if (ss->ply >= MAX_PLY)
    return pos.checkers() ? VALUE_DRAW : evaluate(pos);

  // Transposition table lookup
  Key posKey = pos.key();
  bool ttHit;
  TTEntry *tte = TT->probe(posKey, ttHit);
  Value ttValue = ttHit ? tte->value() : VALUE_NONE;
  Move ttMove = ttHit ? tte->move() : Move::none();

  ss->ttHit = ttHit;
  ss->ttPv = PvNode || (ttHit && tte->is_pv());

  // TT cutoff (not in PV nodes)
  if (!PvNode && ttHit && tte->depth() >= depth) {
    if (ttValue >= beta ? tte->bound() & BOUND_LOWER
                        : tte->bound() & BOUND_UPPER)
      return ttValue;
  }

  // Static evaluation
  Value eval;
  bool improving;

  if (pos.checkers()) {
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

  // ProbCut (~10 Elo)
  // If a reduced search (depth - 4) with a raised beta (beta + 189)
  // already finds a value >= beta, we can prune this node
  if (!PvNode && depth > 4 && !pos.checkers() &&
      std::abs(beta) < VALUE_TB_WIN_IN_MAX_PLY) {

    Value probCutBeta = beta + 189 - 44 * improving;

    Move probCutMoves[MAX_MOVES];
    Move *probCutEnd = generate<CAPTURES>(pos, probCutMoves);

    for (Move *m = probCutMoves; m != probCutEnd; ++m) {
      Move probCutMove = *m;

      if (!pos.legal(probCutMove))
        continue;

      // Only consider captures with good SEE
      if (!pos.see_ge(probCutMove, probCutBeta - ss->staticEval))
        continue;

      StateInfo st;
      pos.do_move(probCutMove, st);

      // Verify with qsearch
      Value value = -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

      // If qsearch doesn't refute, do a proper reduced search
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

  // Generate moves
  Move moves[MAX_MOVES];
  Move *end = pos.checkers() ? generate<EVASIONS>(pos, moves)
                             : generate<NON_EVASIONS>(pos, moves);

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
  for (Move *m = moves; m != end; ++m) {
    move = *m;

    if (!pos.legal(move))
      continue;

    moveCount++;

    // Late move pruning
    if (!rootNode && bestValue > VALUE_MATED_IN_MAX_PLY) {
      if (depth <= 6 && moveCount > (3 + 2 * depth * depth))
        continue;
    }

    // Extensions
    Depth extension = 0;

    // Singular extension search:
    // If the TT move is significantly better than all other moves,
    // extend its search
    bool singularQuietLMR = false;
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
        singularQuietLMR = !pos.capture(move);
      } else if (singularBeta >= beta) {
        // Multi-cut pruning
        return singularBeta;
      } else if (ttValue >= beta) {
        // Negative extension if TT value >= beta but singular search didn't
        extension = -1;
      }
    }

    // Check extension - extend search when giving check
    bool givesCheck = pos.gives_check(move);
    if (givesCheck && extension < 1)
      extension = 1;

    // Make the move
    ss->currentMove = move;
    StateInfo st;
    pos.do_move(move, st, givesCheck);

    // New depth with extension
    Depth newDepth = depth - 1 + extension;
    Value value;

    bool isCapture = pos.capture(move);

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
      if (move == ss->killers[0] || move == ss->killers[1])
        reduction--;

      // Factor 4: Increase for cut nodes
      if (cutNode)
        reduction += 2;

      // Factor 5: Increase for non-improving nodes
      if (!improving)
        reduction++;

      // Factor 6: Decrease for TT move
      if (move == ttMove)
        reduction--;

      // Factor 7: Decrease for PV nodes
      if (PvNode)
        reduction--;

      // Factor 8: Increase for quiet moves after many quiet moves
      if (!isCapture && moveCount > 4)
        reduction++;

      // Factor 9: Decrease for passed pawn pushes
      if (!isCapture && type_of(pos.moved_piece(move)) == PAWN) {
        Square to = move.to_sq();
        // Rank bonus for advanced pawns
        Rank relRank = relative_rank(pos.side_to_move(), rank_of(to));
        if (relRank >= RANK_6)
          reduction -= 2;
        else if (relRank >= RANK_5)
          reduction--;
      }

      // Factor 16: Decrease for singular extension candidate that failed
      if (singularQuietLMR)
        reduction--;

      // Factor 10: Decrease for recaptures
      if (isCapture && ss->ply >= 1 && move.to_sq() == (ss - 1)->currentMove.to_sq())
        reduction--;

      // Factor 11: Decrease for TT PV moves
      if (ss->ttPv)
        reduction--;

      // Factor 12: Increase when previous move was a null move (opponent passed)
      if ((ss - 1)->currentMove == Move::null())
        reduction++;

      // Factor 13: Decrease for promotions
      if (move.type_of() == PROMOTION)
        reduction -= 2;

      // Factor 14: Increase more for very late moves
      if (moveCount > 10)
        reduction++;

      // Factor 15: Decrease for moves to central squares
      {
        Square to = move.to_sq();
        int f = file_of(to);
        int r = rank_of(to);
        if (f >= FILE_C && f <= FILE_F && r >= RANK_3 && r <= RANK_6)
          reduction--;
      }

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
          // Beta cutoff - update statistics for quiet moves
          if (!pos.capture(move)) {
            // Update killer moves
            if (move != ss->killers[0]) {
              ss->killers[1] = ss->killers[0];
              ss->killers[0] = move;
            }

            // Update counter moves (refutation of opponent's previous move)
            if (ss->ply >= 1 && (ss - 1)->currentMove.is_ok()) {
              Piece prevPiece = pos.piece_on((ss - 1)->currentMove.to_sq());
              if (prevPiece != NO_PIECE) {
                counterMoves[prevPiece][(ss - 1)->currentMove.to_sq()] = move;
              }
            }

            // Update main history
            int bonus = std::min(16 * depth * depth, 1200);
            Color us = pos.side_to_move();
            int idx = move.from_sq() * 64 + move.to_sq();
            int16_t &entry = mainHistory[us][idx];
            entry += bonus - entry * std::abs(bonus) / 16384;
          } else {
            // Update capture history
            Piece moved = pos.moved_piece(move);
            Square to = move.to_sq();
            PieceType captured = type_of(pos.piece_on(to));
            int bonus = std::min(16 * depth * depth, 1200);
            int16_t &entry = captureHistory[moved][to][captured];
            entry += bonus - entry * std::abs(bonus) / 16384;
          }
          break;
        }

        alpha = value;
      }
    }
  }

  // Checkmate/stalemate detection
  if (moveCount == 0)
    bestValue = pos.checkers() ? mated_in(ss->ply) : VALUE_DRAW;

  // Update TT
  tte->save(posKey, bestValue, ss->ttPv,
            bestValue >= beta    ? BOUND_LOWER
            : PvNode && bestMove ? BOUND_EXACT
                                 : BOUND_UPPER,
            depth, bestMove, ss->staticEval, TT->generation());

  // Update root move scores
  if (rootNode) {
    RootMove &rm = *std::find(rootMoves.begin(), rootMoves.end(),
                              bestMove ? bestMove : moves[0]);
    rm.score = bestValue;
    rm.selDepth = selDepth;

    if (bestMove) {
      rm.pv.clear();
      rm.pv.push_back(bestMove);
      for (int i = 0; ss->pv[i + 1] != Move::none(); ++i)
        rm.pv.push_back(ss->pv[i + 1]);
    }
  }

  return bestValue;
}

// Quiescence search
template <NodeType nodeType>
Value Worker::qsearch(Position &pos, Stack *ss, Value alpha, Value beta) {
  constexpr bool PvNode = nodeType != NonPV;

  nodes++;

  // Draw detection
  if (pos.is_draw(ss->ply))
    return VALUE_DRAW;

  // Max ply
  if (ss->ply >= MAX_PLY)
    return pos.checkers() ? VALUE_DRAW : evaluate(pos);

  // TT probe
  Key posKey = pos.key();
  bool ttHit;
  TTEntry *tte = TT->probe(posKey, ttHit);
  Value ttValue = ttHit ? tte->value() : VALUE_NONE;
  Move ttMove = ttHit ? tte->move() : Move::none();

  // TT cutoff
  if (!PvNode && ttHit && tte->depth() >= DEPTH_QS) {
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

  // Update TT
  tte->save(posKey, bestValue, false,
            bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
            ss->staticEval, TT->generation());

  return bestValue;
}

// Evaluate position using GPU-accelerated NNUE
Value Worker::evaluate(const Position &pos) { return Eval::evaluate(pos); }

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

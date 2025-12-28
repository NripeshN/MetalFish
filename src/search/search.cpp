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

// Add a small random component to draw evaluations to avoid 3-fold blindness
// This prevents the engine from being overly repetition-prone
Value value_draw(uint64_t nodes) { return VALUE_DRAW - 1 + Value(nodes & 0x2); }

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
  std::memset(pawnHistory, 0, sizeof(pawnHistory));
  std::memset(correctionHistory, 0, sizeof(correctionHistory));
  std::memset(continuationHistoryTable, 0, sizeof(continuationHistoryTable));
  std::memset(lowPlyHistory, 0, sizeof(lowPlyHistory));

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
  std::memset(pawnHistory, 0, sizeof(pawnHistory));
  std::memset(correctionHistory, 0, sizeof(correctionHistory));
  std::memset(continuationHistoryTable, 0, sizeof(continuationHistoryTable));
  std::memset(lowPlyHistory, 0, sizeof(lowPlyHistory));
  killers.clear();
}

// Update quiet move history on beta cutoff
void Worker::update_quiet_stats(Stack *ss, Move move, int bonus) {
  // Clamp bonus
  bonus = std::clamp(bonus, -1200, 1200);

  // Update main history with gravity formula
  Color us = rootPos->side_to_move();
  int idx = move.from_sq() * 64 + move.to_sq();
  int16_t &entry = mainHistory[us][idx];
  entry += bonus - entry * std::abs(bonus) / 16384;

  // Update pawn history (indexed by pawn structure)
  int pawnIdx = pawn_history_index(*rootPos);
  Piece movedPiece = rootPos->piece_on(move.from_sq());
  if (movedPiece != NO_PIECE) {
    int16_t &pawnEntry = pawnHistory[pawnIdx][movedPiece][move.to_sq()];
    pawnEntry += bonus - pawnEntry * std::abs(bonus) / 16384;
  }

  // Update continuation history (previous moves -> this move)
  if (movedPiece != NO_PIECE && ss->continuationHistory) {
    Square to = move.to_sq();
    // Update continuation history for this move relative to previous moves
    for (int i : {1, 2, 4}) {
      if (ss->ply >= i && (ss - i)->continuationHistory) {
        int16_t &contEntry = (*(ss - i)->continuationHistory)[movedPiece][to];
        contEntry += bonus - contEntry * std::abs(bonus) / 16384;
      }
    }
  }

  // Update low ply history for moves near root
  if (ss->ply < LOW_PLY_HISTORY_SIZE) {
    int moveIdx = move.from_sq() * 64 + move.to_sq();
    int16_t &lowEntry = lowPlyHistory[ss->ply][moveIdx];
    lowEntry += bonus - lowEntry * std::abs(bonus) / 16384;
  }

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

// Update capture history on beta cutoff
void Worker::update_capture_stats(Piece piece, Square to, PieceType captured,
                                  int bonus) {
  bonus = std::clamp(bonus, -1200, 1200);
  int16_t &entry = captureHistory[piece][to][captured];
  entry += bonus - entry * std::abs(bonus) / 16384;
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
  }

  // Set up initial continuation history for root
  for (int i = -7; i < 0; ++i)
    (ss + i)->continuationHistory = &continuationHistoryTable[NO_PIECE][SQ_A1];

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

  // Iterative deepening
  for (rootDepth = 1; rootDepth <= (limits.depth ? limits.depth : MAX_PLY);
       ++rootDepth) {
    // Check for stop conditions
    if (stopRequested || Signals_stop)
      break;

    // Reset fail high counter and aspiration window for new depth
    failedHighCnt = 0;

    // Aspiration window search
    if (rootDepth >= 4 && !rootMoves.empty()) {
      // Use average score for more stable aspiration windows
      Value avg = rootMoves[0].averageScore != -VALUE_INFINITE
                      ? rootMoves[0].averageScore
                      : bestValue;
      delta = 16 + std::abs(avg) * std::abs(avg) / 21367;
      alpha = std::max(avg - delta, -VALUE_INFINITE);
      beta = std::min(avg + delta, VALUE_INFINITE);
    }

    // Main search
    while (true) {
      // Reduce depth after multiple fail-highs (like Stockfish)
      Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt);
      bestValue = search<Root>(*rootPos, ss, alpha, beta, adjustedDepth, false);

      // Sort root moves by score
      std::stable_sort(rootMoves.begin(), rootMoves.end());

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
        // Keep alpha stable on fail-high (Stockfish behavior)
        alpha = std::max(beta - delta, alpha);
        beta = std::min(bestValue + delta, VALUE_INFINITE);
        ++failedHighCnt;
      } else
        break;

      delta += delta / 3;
    }

    // Update average score (exponential moving average)
    if (!rootMoves.empty()) {
      Value prevAvg = rootMoves[0].averageScore;
      if (prevAvg == -VALUE_INFINITE)
        rootMoves[0].averageScore = bestValue;
      else
        rootMoves[0].averageScore = (prevAvg * 2 + bestValue) / 3;
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

    // Track best move stability for time management
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
      if (!limits.infinite && limits.use_time_management()) {
        TimePoint elapsed = timeManager.elapsed();
        TimePoint optimum = timeManager.optimum();

        // Early termination if best move is stable and score is not falling
        if (stableBestMoveCount >= 3 && scoreDiff >= -20) {
          // Reduce time if very stable (adjust by 0.8x)
          if (elapsed > optimum * 0.8)
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

        if (elapsed > optimum * timeScale)
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

  // Internal Iterative Reductions (IIR)
  // Reduce depth for PV/Cut nodes without a TT move
  if (depth >= 4 && !ttMove && !pos.checkers())
    depth--;

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

    // Apply correction history adjustment
    int corrIdx = correction_history_index(pos.pawn_key());
    int correction = correctionHistory[corrIdx][pos.side_to_move()];
    eval += correction / 2; // Scale down correction
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

    // Skip excluded move (for singular extension search)
    if (move == ss->excludedMove)
      continue;

    if (!pos.legal(move))
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
      } else if (singularBeta >= beta) {
        // Multi-cut pruning: singular beta >= beta means all moves fail high
        return singularBeta;
      } else if (ttValue >= beta) {
        // Negative extension if TT value >= beta but no singular move
        extension = -1;
      }
    }

    // Make the move
    ss->currentMove = move;

    // Set up continuation history for this ply
    Piece movedPc = pos.moved_piece(move);
    (ss + 1)->continuationHistory =
        &continuationHistoryTable[movedPc][move.to_sq()];

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
          // Beta cutoff - update history statistics
          int bonus = std::min(16 * depth * depth, 1200);

          if (!pos.capture(move)) {
            // Update quiet move statistics
            update_quiet_stats(ss, move, bonus);
          } else {
            // Update capture statistics
            Piece moved = pos.moved_piece(move);
            Square to = move.to_sq();
            Piece captured = pos.piece_on(to);
            PieceType capturedType =
                captured != NO_PIECE ? type_of(captured) : PAWN;
            update_capture_stats(moved, to, capturedType, bonus);
          }
          break;
        }

        alpha = value;
      }
    }
  }

  // Checkmate/stalemate detection
  if (moveCount == 0)
    bestValue = pos.checkers() ? mated_in(ss->ply) : value_draw(nodes);

  // Update correction history when search result differs from static eval
  if (!pos.checkers() && bestMove && !pos.capture(bestMove) &&
      std::abs(bestValue) < VALUE_TB_WIN_IN_MAX_PLY) {
    int corrIdx = correction_history_index(pos.pawn_key());
    Color us = pos.side_to_move();
    int diff = bestValue - ss->staticEval;
    int bonus = std::clamp(diff * depth / 8, -CORRECTION_HISTORY_LIMIT / 4,
                           CORRECTION_HISTORY_LIMIT / 4);
    int16_t &entry = correctionHistory[corrIdx][us];
    entry = std::clamp(entry + bonus - entry * std::abs(bonus) / 16384,
                       -CORRECTION_HISTORY_LIMIT, CORRECTION_HISTORY_LIMIT);
  }

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
  Value ttValue = ttHit ? tte->value() : VALUE_NONE;
  Move ttMove = ttHit ? tte->move() : Move::none();
  (void)ttMove; // Unused for now, could be used for move ordering in qsearch

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

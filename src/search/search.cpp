/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0

*/

#include "search/search.h"
#include "core/movegen.h"
#include "eval/evaluate.h"
#include "search/movepick.h"
#include "search/tt.h"
#include "uci/uci.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>

namespace MetalFish {

namespace Search {

std::atomic<bool> Signals_stop{false};
std::atomic<bool> Signals_ponder{false};
TranspositionTable *TT = &MetalFish::TT;

namespace {

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
  // Initialize history tables with Stockfish default values
  constexpr int mainHistoryDefault = 68;
  for (int c = 0; c < 2; ++c)
    for (int i = 0; i < 4096; ++i)
      mainHistory[c][i] = mainHistoryDefault;
  
  std::memset(lowPlyHistory, 0, sizeof(lowPlyHistory));
  
  // Stockfish initializes captureHistory to -689
  for (int p = 0; p < 16; ++p)
    for (int s = 0; s < 64; ++s)
      for (int pt = 0; pt < 8; ++pt)
        captureHistory[p][s][pt] = -689;
  
  // Stockfish initializes pawnHistory to -1238
  for (int i = 0; i < PAWN_HISTORY_SIZE; ++i)
    for (int p = 0; p < 16; ++p)
      for (int s = 0; s < 64; ++s)
        pawnHistory[i][p][s] = -1238;

  // Allocate correction history on heap
  correctionHistory = std::make_unique<UnifiedCorrectionHistory>();
  correctionHistory->clear();

  // Stockfish initializes continuationHistory to -529
  for (int p = 0; p < 16; ++p)
    for (int s = 0; s < 64; ++s)
      for (int p2 = 0; p2 < 16; ++p2)
        for (int s2 = 0; s2 < 64; ++s2)
          continuationHistoryTable[p][s][p2][s2] = -529;
  
  // Stockfish initializes continuationCorrectionHistory to 8
  for (int p = 0; p < 16; ++p)
    for (int s = 0; s < 64; ++s)
      for (int p2 = 0; p2 < 16; ++p2)
        for (int s2 = 0; s2 < 64; ++s2)
          continuationCorrectionHistory[p][s][p2][s2] = 8;
  
  std::memset(counterMoves, 0, sizeof(counterMoves));
  ttMoveHistory = 0;

  for (int i = 0; i < MAX_MOVES; ++i)
    reductions[i] = int(2747 / 128.0 * std::log(i + 1));

  // Initialize time management tracking
  iterValue.fill(VALUE_ZERO);
  previousTimeReduction = 0.85;
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

  // Initialize history tables with Stockfish default values
  constexpr int mainHistoryDefault = 68;
  for (int c = 0; c < 2; ++c)
    for (int i = 0; i < 4096; ++i)
      mainHistory[c][i] = mainHistoryDefault;
  
  std::memset(lowPlyHistory, 0, sizeof(lowPlyHistory));
  
  // Stockfish initializes captureHistory to -689
  for (int p = 0; p < 16; ++p)
    for (int s = 0; s < 64; ++s)
      for (int pt = 0; pt < 8; ++pt)
        captureHistory[p][s][pt] = -689;
  
  // Stockfish initializes pawnHistory to -1238
  for (int i = 0; i < PAWN_HISTORY_SIZE; ++i)
    for (int p = 0; p < 16; ++p)
      for (int s = 0; s < 64; ++s)
        pawnHistory[i][p][s] = -1238;
  
  if (correctionHistory)
    correctionHistory->clear();
  
  // Stockfish initializes continuationHistory to -529
  for (int p = 0; p < 16; ++p)
    for (int s = 0; s < 64; ++s)
      for (int p2 = 0; p2 < 16; ++p2)
        for (int s2 = 0; s2 < 64; ++s2)
          continuationHistoryTable[p][s][p2][s2] = -529;
  
  // Stockfish initializes continuationCorrectionHistory to 8
  for (int p = 0; p < 16; ++p)
    for (int s = 0; s < 64; ++s)
      for (int p2 = 0; p2 < 16; ++p2)
        for (int s2 = 0; s2 < 64; ++s2)
          continuationCorrectionHistory[p][s][p2][s2] = 8;
  
  std::memset(counterMoves, 0, sizeof(counterMoves));
  ttMoveHistory = 0;
  killers.clear();

  // Reset time management tracking
  iterValue.fill(VALUE_ZERO);
  previousTimeReduction = 0.85;
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
  update_continuation_histories(ss, movedPiece, move.to_sq(),
                                bonus * 896 / 1024);

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
void Worker::update_continuation_histories(Stack *ss, Piece pc, Square to,
                                           int bonus) {
  // Stockfish continuation history weights: {ply_offset, weight}
  static constexpr std::array<std::pair<int, int>, 6> conthist_bonuses = {
      {{1, 1133}, {2, 683}, {3, 312}, {4, 582}, {5, 149}, {6, 474}}};

  for (const auto &[i, weight] : conthist_bonuses) {
    // Only update the first 2 continuation histories if we are in check
    if (ss->inCheck && i > 2)
      break;

    if (ss->ply >= i && (ss - i)->currentMove.is_ok() &&
        (ss - i)->continuationHistory) {
      int weightedBonus = (bonus * weight / 1024) + 88 * (i < 2);
      history_update((*(ss - i)->continuationHistory)[pc][to], weightedBonus);
    }
  }
}

// Update capture history on beta cutoff
void Worker::update_capture_stats(Piece piece, Square to, PieceType captured,
                                  int bonus) {
  bonus = std::clamp(bonus, -1200, 1200);
  history_update(captureHistory[piece][to][captured], bonus);
}

// Update quiet histories (matching Stockfish)
void Worker::update_quiet_histories(const Position &pos, Stack *ss, Move move,
                                    int bonus) {
  Color us = pos.side_to_move();
  int idx = move.from_sq() * 64 + move.to_sq();

  // Update main history
  history_update(mainHistory[us][idx], bonus);

  // Update low ply history
  if (ss->ply < LOW_PLY_HISTORY_SIZE)
    history_update(lowPlyHistory[ss->ply][idx], bonus * 805 / 1024);

  // Update continuation histories
  update_continuation_histories(ss, pos.moved_piece(move), move.to_sq(),
                                bonus * 896 / 1024);

  // Update pawn history with asymmetric bonus (higher for positive)
  int pIdx = pawn_history_index(pos);
  history_update(pawnHistory[pIdx][pos.moved_piece(move)][move.to_sq()],
                 bonus * (bonus > 0 ? 905 : 505) / 1024);
}

// Update all statistics after a move (matching Stockfish)
void Worker::update_all_stats(const Position &pos, Stack *ss, Move bestMove,
                              Square prevSq, SearchedList &quietsSearched,
                              SearchedList &capturesSearched, Depth depth,
                              Move ttMove, int moveCount) {

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
    history_update(captureHistory[mp][move.to_sq()][capturedPiece],
                   -malus * 1448 / 1024);
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
    (ss + i)->continuationCorrectionHistory =
        &continuationCorrectionHistory[NO_PIECE][SQ_A1];
    (ss + i)->staticEval = VALUE_NONE;
  }

  ss->pv = pv;

  TT->new_search();

  // Initialize lowPlyHistory (matching Stockfish)
  for (int i = 0; i < LOW_PLY_HISTORY_SIZE; ++i)
    for (int j = 0; j < 4096; ++j)
      lowPlyHistory[i][j] = 97;

  // Age mainHistory (matching Stockfish)
  constexpr int mainHistoryDefault = 68;
  Color us = rootPos->side_to_move();
  for (Color c : {WHITE, BLACK})
    for (int i = 0; i < 4096; ++i)
      mainHistory[c][i] = (mainHistory[c][i] - mainHistoryDefault) * 3 / 4 + mainHistoryDefault;

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

  // Lazy SMP: helper threads skip some depths for diversity
  // Main thread (idx 0) searches all depths, helpers skip based on their index
  const int skipSize[] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                          3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
  const int skipPhase[] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3,
                           4, 5, 0, 1, 2, 3, 4, 5, 6, 7};

  for (rootDepth = 1; rootDepth <= (limits.depth ? limits.depth : MAX_PLY);
       ++rootDepth) {
    // Check for stop conditions
    if (stopRequested || Signals_stop)
      break;

    // Lazy SMP: skip some depths for helper threads
    if (threadIdx > 0) {
      int idx = std::min(size_t(19), threadIdx - 1);
      if ((rootDepth + skipPhase[idx]) % skipSize[idx] != 0)
        continue;
    }

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
        // Ensure at least one effective increment for every four searchAgain
        // steps
        Depth adjustedDepth = std::max(1, rootDepth - failedHighCnt -
                                              3 * (searchAgainCounter + 1) / 4);
        
        // Set rootDelta before search (needed for reduction formula)
        rootDelta = beta - alpha;
        
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

      // Update average score (exponential moving average) and mean squared
      // score
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
                  << nodeCount << " nps "
                  << (elapsed > 0 ? nodeCount * 1000 / elapsed : 0) << " time "
                  << elapsed << " hashfull " << TT->hashfull() << " pv";

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
      if (!limits.infinite && !limits.ponderMode &&
          limits.use_time_management()) {
        TimePoint elapsed = timeManager.elapsed();
        TimePoint optimum = timeManager.optimum();
        TimePoint maximum = timeManager.maximum();

        // Decide whether to increase depth or search again at same depth
        // Based on Stockfish: increaseDepth = ponder || elapsed <= totalTime *
        // 0.50
        increaseDepth = limits.ponderMode || elapsed <= maximum * 0.50;

        // Calculate effort-based adjustment (Stockfish formula)
        uint64_t totalNodes = nodes.load();
        uint64_t nodesEffort =
            rootMoves[0].effort * 100000 / std::max(uint64_t(1), totalNodes);
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
  const bool allNode = !(PvNode || cutNode); // ALL node flag for LMR scaling

  // Check for stop conditions
  if (stopRequested || Signals_stop)
    return VALUE_ZERO;

  // Drop into quiescence search at horizon
  if (depth <= 0)
    return qsearch<PvNode ? PV : NonPV>(pos, ss, alpha, beta);

  // Limit depth to avoid overflow
  depth = std::min(depth, MAX_PLY - 1);

  // Node count
  nodes++;

  // Check node limit and time limit every 64 nodes for efficiency
  if ((nodes.load() & 63) == 0) {
    if (limits.nodes && nodes.load() >= limits.nodes) {
      stopRequested = true;
      return VALUE_ZERO;
    }
    // Check time limit
    if (limits.use_time_management() &&
        timeManager.elapsed() >= timeManager.maximum()) {
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

  // Initialize node
  Square prevSq = ((ss - 1)->currentMove).is_ok()
                      ? ((ss - 1)->currentMove).to_sq()
                      : SQ_NONE;
  Move bestMove = Move::none();
  int priorReduction = (ss - 1)->reduction;
  (ss - 1)->reduction = 0;
  ss->statScore = 0;
  (ss + 2)->cutoffCnt = 0;

  // Mate distance pruning
  // Even if we mate at the next move, our score would be at best
  // mate_in(ss->ply+1). If alpha is already bigger because a shorter mate was
  // found upward in the tree, there is no need to search because we will never
  // beat current alpha.
  if (!rootNode) {
    alpha = std::max(mated_in(ss->ply), alpha);
    beta = std::min(mate_in(ss->ply + 1), beta);
    if (alpha >= beta)
      return alpha;
  }

  // Transposition table lookup
  Move excludedMove = ss->excludedMove;
  Key posKey = pos.key();
  bool ttHit;
  TTEntry *tte = TT->probe(posKey, ttHit);
  Value ttValue = ttHit
                      ? value_from_tt(tte->value(), ss->ply, pos.rule50_count())
                      : VALUE_NONE;
  Move ttMove =
      rootNode ? rootMoves[pvIdx].pv[0] : (ttHit ? tte->move() : Move::none());
  bool ttCapture = ttMove.is_ok() && pos.capture(ttMove);

  ss->ttHit = ttHit;
  ss->ttPv = excludedMove ? ss->ttPv : (PvNode || (ttHit && tte->is_pv()));

  // Track if we're in check
  ss->inCheck = pos.checkers();
  bool priorCapture = pos.captured_piece() != NO_PIECE;

  // TT cutoff (not in PV nodes)
  // At non-PV nodes we check for an early TT cutoff
  if (!PvNode && !excludedMove && ttHit &&
      tte->depth() >= depth - (ttValue <= beta) && ttValue != VALUE_NONE &&
      (tte->bound() & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER)) &&
      (cutNode == (ttValue >= beta) || depth > 5)) {

    // If ttMove is quiet, update move sorting heuristics on TT hit
    if (ttMove.is_ok() && ttValue >= beta) {
      if (!ttCapture) {
        int bonus = std::min(132 * depth - 72, 985);
        update_quiet_histories(pos, ss, ttMove, bonus);
      }
      // Extra penalty for early quiet moves of the previous ply
      if (prevSq != SQ_NONE && (ss - 1)->moveCount < 4 && !priorCapture) {
        update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq, -2060);
      }
    }
    
    // Graph history interaction workaround
    // For high rule50 counts don't produce transposition table cutoffs
    if (pos.rule50_count() < 96)
      return ttValue;
  }

  // Static evaluation
  Value eval;
  Value unadjustedStaticEval = VALUE_NONE;
  bool improving;
  bool opponentWorsening;

  if (ss->inCheck) {
    ss->staticEval = eval = (ss - 2)->staticEval;
    improving = false;
    opponentWorsening = false;
  } else if (excludedMove) {
    unadjustedStaticEval = eval = ss->staticEval;
    improving = ss->ply >= 2 && ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;
  } else if (ttHit) {
    unadjustedStaticEval = tte->eval();
    if (unadjustedStaticEval == VALUE_NONE)
      unadjustedStaticEval = evaluate(pos);

    // Apply full correction history adjustment (matching Stockfish)
    Move prevMove = ss->ply >= 1 ? (ss - 1)->currentMove : Move::none();
    ContinuationCorrectionHistory *contCorr2 =
        ss->ply >= 2 ? (ss - 2)->continuationCorrectionHistory : nullptr;
    ContinuationCorrectionHistory *contCorr4 =
        ss->ply >= 4 ? (ss - 4)->continuationCorrectionHistory : nullptr;

    int correctionValue = compute_full_correction_value(
        *correctionHistory, contCorr2, contCorr4, pos, prevMove);
    ss->staticEval = eval =
        to_corrected_static_eval(unadjustedStaticEval, correctionValue);

    // ttValue can be used as a better position evaluation
    if (ttValue != VALUE_NONE &&
        (tte->bound() & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
      eval = ttValue;

    improving = ss->ply >= 2 && ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;
  } else {
    unadjustedStaticEval = evaluate(pos);

    // Apply full correction history adjustment
    Move prevMove = ss->ply >= 1 ? (ss - 1)->currentMove : Move::none();
    ContinuationCorrectionHistory *contCorr2 =
        ss->ply >= 2 ? (ss - 2)->continuationCorrectionHistory : nullptr;
    ContinuationCorrectionHistory *contCorr4 =
        ss->ply >= 4 ? (ss - 4)->continuationCorrectionHistory : nullptr;

    int correctionValue = compute_full_correction_value(
        *correctionHistory, contCorr2, contCorr4, pos, prevMove);
    ss->staticEval = eval =
        to_corrected_static_eval(unadjustedStaticEval, correctionValue);

    // Save static evaluation to TT
    tte->save(posKey, VALUE_NONE, ss->ttPv, BOUND_NONE, DEPTH_QS, Move::none(),
              unadjustedStaticEval, TT->generation());

    improving = ss->ply >= 2 && ss->staticEval > (ss - 2)->staticEval;
    opponentWorsening = ss->staticEval > -(ss - 1)->staticEval;
  }

  // Hindsight adjustment of reductions based on static evaluation difference
  if (priorReduction >= 3 && !opponentWorsening)
    depth++;
  if (priorReduction >= 2 && depth >= 2 &&
      ss->staticEval + (ss - 1)->staticEval > 173)
    depth--;

  // Declare variables before potential goto
  Color us = pos.side_to_move();
  Value probCutBeta;

  // Skip early pruning when in check
  if (ss->inCheck)
    goto moves_loop;

  // Use static evaluation difference to improve quiet move ordering
  if (((ss - 1)->currentMove).is_ok() && !(ss - 1)->inCheck && !priorCapture) {
    int evalDiff =
        std::clamp(-int((ss - 1)->staticEval + ss->staticEval), -209, 167) + 59;
    int histIdx = ((ss - 1)->currentMove).from_sq() * 64 +
                  ((ss - 1)->currentMove).to_sq();
    history_update(mainHistory[~us][histIdx], evalDiff * 9);
    
    // Also update pawn history if not a pawn move and not a promotion
    if (!ttHit && type_of(pos.piece_on(prevSq)) != PAWN &&
        ((ss - 1)->currentMove).type_of() != PROMOTION) {
      int pawnIdx = pawn_history_index(pos);
      history_update(pawnHistory[pawnIdx][pos.piece_on(prevSq)][prevSq], evalDiff * 13);
    }
  }

  // Razoring
  if (!PvNode && eval < alpha - 485 - 281 * depth * depth)
    return qsearch<NonPV>(pos, ss, alpha, beta);

  // Futility pruning (Stockfish formula)
  {
    auto futilityMargin = [&](Depth d) {
      Value futilityMult = 76 - 23 * !ss->ttHit;
      return futilityMult * d -
             (2474 * improving + 331 * opponentWorsening) * futilityMult / 1024;
    };

    if (!ss->ttPv && depth < 14 && eval - futilityMargin(depth) >= beta &&
        eval >= beta && (!ttMove.is_ok() || ttCapture) && !is_loss(beta) &&
        !is_win(eval))
      return (2 * beta + eval) / 3;
  }

  // Null move pruning
  if (cutNode && ss->staticEval >= beta - 18 * depth + 350 && !excludedMove &&
      pos.non_pawn_material(us) && ss->ply >= nmpMinPly && !is_loss(beta)) {

    Depth R = 7 + depth / 3;

    StateInfo st;
    pos.do_null_move(st);
    ss->currentMove = Move::null();
    ss->continuationHistory = &continuationHistoryTable[0][0];

    Value nullValue =
        -search<NonPV>(pos, ss + 1, -beta, -beta + 1, depth - R, false);

    pos.undo_null_move();

    // Do not return unproven mate or TB scores
    if (nullValue >= beta && !is_win(nullValue)) {
      if (nmpMinPly || depth < 16)
        return nullValue;

      // Do verification search at high depths
      nmpMinPly = ss->ply + 3 * (depth - R) / 4;
      Value v = search<NonPV>(pos, ss, beta - 1, beta, depth - R, false);
      nmpMinPly = 0;

      if (v >= beta)
        return nullValue;
    }
  }

  // Update improving flag if static eval >= beta
  improving |= ss->staticEval >= beta;

  // Internal Iterative Reductions (IIR)
  // At sufficient depth, reduce depth for PV/Cut nodes without a TTMove
  if (!allNode && depth >= 6 && !ttMove.is_ok() && priorReduction <= 3)
    depth--;

  // ProbCut
  // If we have a good enough capture and a reduced search returns a value
  // much above beta, we can prune the previous move safely.
  probCutBeta = beta + 235 - 63 * improving;
  if (depth >= 3 && !is_decisive(beta) &&
      !(ttValue != VALUE_NONE && ttValue < probCutBeta)) {

    Move probCutMoves[MAX_MOVES];
    Move *probCutEnd = generate<CAPTURES>(pos, probCutMoves);
    Depth probCutDepth =
        std::clamp(depth - 5 - (ss->staticEval - beta) / 315, 0, depth);

    for (Move *pm = probCutMoves; pm != probCutEnd; ++pm) {
      Move probCutMove = *pm;

      if (probCutMove == excludedMove || !pos.legal(probCutMove))
        continue;

      // Skip captures with bad SEE
      if (!pos.see_ge(probCutMove, probCutBeta - ss->staticEval))
        continue;

      StateInfo st;
      pos.do_move(probCutMove, st);
      ss->currentMove = probCutMove;
      ss->continuationHistory =
          &continuationHistoryTable[pos.moved_piece(probCutMove)]
                                   [probCutMove.to_sq()];

      // Verify with qsearch first
      Value value =
          -qsearch<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1);

      // If qsearch doesn't refute, do a reduced search
      if (value >= probCutBeta && probCutDepth > 0)
        value = -search<NonPV>(pos, ss + 1, -probCutBeta, -probCutBeta + 1,
                               probCutDepth, !cutNode);

      pos.undo_move(probCutMove);

      if (value >= probCutBeta) {
        // Save ProbCut data into transposition table
        tte->save(posKey, value_to_tt(value, ss->ply), ss->ttPv, BOUND_LOWER,
                  probCutDepth + 1, probCutMove, unadjustedStaticEval,
                  TT->generation());
        if (!is_decisive(value))
          return value - (probCutBeta - beta);
      }
    }
  }

moves_loop: // When in check, search starts here

  // Small ProbCut idea: TT-based pruning before move loop
  probCutBeta = beta + 418;
  if ((tte->bound() & BOUND_LOWER) && tte->depth() >= depth - 4 &&
      ttValue >= probCutBeta && !is_decisive(beta) && ttValue != VALUE_NONE &&
      !is_decisive(ttValue))
    return probCutBeta;

  // Move generation and ordering
  Move pv[MAX_PLY + 1];
  ss->pv = pv;
  pv[0] = Move::none();

  Move move;
  int moveCount = 0;
  Value bestValue = -VALUE_INFINITE;
  bestMove = Move::none();

  // Track searched moves for history updates
  SearchedList quietsSearched;
  SearchedList capturesSearched;

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
      // Note: Only skip quiet moves, not captures
      bool skipQuiets = (depth <= 6 && moveCount > (3 + depth * depth) / (2 - improving));

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
        // Skip quiet moves if we've searched enough
        if (skipQuiets)
          continue;
          
        // Futility pruning for quiet moves (less aggressive)
        if (!givesCheck && lmrDepth < 8 &&
            ss->staticEval + 150 + 200 * lmrDepth <= alpha)
          continue;

        // SEE-based pruning for quiet moves
        if (lmrDepth > 0 && !pos.see_ge(move, -30 * lmrDepth * lmrDepth))
          continue;
      }
    }

    // Extensions
    Depth extension = 0;

    // Singular extension search
    // If the TT move is significantly better than all other moves,
    // extend its search
    if (!rootNode && depth >= 6 + ss->ttPv && move == ttMove && ttHit &&
        ss->excludedMove == Move::none() && (tte->bound() & BOUND_LOWER) &&
        tte->depth() >= depth - 3 && !is_decisive(ttValue) &&
        !is_shuffling(move, ss, pos)) {

      Value singularBeta = ttValue - (53 + 75 * (ss->ttPv && !PvNode)) * depth / 60;
      Depth singularDepth = (depth - 1) / 2;

      ss->excludedMove = move;
      Value singularValue = search<NonPV>(pos, ss, singularBeta - 1,
                                          singularBeta, singularDepth, cutNode);
      ss->excludedMove = Move::none();

      if (singularValue < singularBeta) {
        // Calculate correction value for extension margins
        Move prevMove = ss->ply >= 1 ? (ss - 1)->currentMove : Move::none();
        ContinuationCorrectionHistory *contCorr2 =
            ss->ply >= 2 ? (ss - 2)->continuationCorrectionHistory : nullptr;
        ContinuationCorrectionHistory *contCorr4 =
            ss->ply >= 4 ? (ss - 4)->continuationCorrectionHistory : nullptr;
        int corrVal = compute_full_correction_value(*correctionHistory, contCorr2, contCorr4, pos, prevMove);
        int corrValAdj = std::abs(corrVal) / 230673;
        
        int doubleMargin = -4 + 199 * PvNode - 201 * !ttCapture - corrValAdj
                         - 897 * ttMoveHistory / 127649 - (ss->ply > rootDepth) * 42;
        int tripleMargin = 73 + 302 * PvNode - 248 * !ttCapture + 90 * ss->ttPv - corrValAdj
                         - (ss->ply * 2 > rootDepth * 3) * 50;
        
        extension = 1 + (singularValue < singularBeta - doubleMargin) 
                      + (singularValue < singularBeta - tripleMargin);
        depth++;
      }
      // Multi-cut pruning
      else if (singularValue >= beta && !is_decisive(singularValue)) {
        history_update(ttMoveHistory, std::max(-400 - 100 * depth, -4000));
        return singularValue;
      }
      // Negative extensions
      else if (ttValue >= beta)
        extension = -3;
      else if (cutNode)
        extension = -2;
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

    // Calculate statScore for LMR adjustment
    Color us = pos.side_to_move();
    int moveIdx = move.from_sq() * 64 + move.to_sq();

    if (isCapture) {
      Piece captured = pos.piece_on(move.to_sq());
      PieceType capturedType = captured != NO_PIECE ? type_of(captured) : PAWN;
      ss->statScore = 868 * PieceValue[make_piece(WHITE, capturedType)] / 128 +
                      captureHistory[movedPc][move.to_sq()][capturedType];
    } else {
      ss->statScore = 2 * mainHistory[us][moveIdx];
      if (ss->ply >= 1 && (ss - 1)->continuationHistory)
        ss->statScore +=
            (*(ss - 1)->continuationHistory)[movedPc][move.to_sq()];
      if (ss->ply >= 2 && (ss - 2)->continuationHistory)
        ss->statScore +=
            (*(ss - 2)->continuationHistory)[movedPc][move.to_sq()];
    }

    if (depth >= 2 && moveCount > 1 + (PvNode ? 1 : 0)) {
      // Calculate delta for reduction formula
      int delta = beta - alpha;
      
      // Use Stockfish's reduction formula
      Depth r = reduction(improving, depth, moveCount, delta);
      
      // Clamp and convert to depth (simpler formula to avoid issues)
      Depth d = std::max(1, newDepth - r);
      
      ss->reduction = newDepth - d;
      value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, d, true);
      ss->reduction = 0;

      // Do a full-depth search when reduced LMR search fails high
      if (value > alpha && d < newDepth) {
        // Adjust full-depth search based on LMR results
        bool doDeeperSearch = value > bestValue + 50;
        bool doShallowerSearch = value < bestValue + 9;
        
        Depth adjustedDepth = newDepth + doDeeperSearch - doShallowerSearch;
        
        if (adjustedDepth > d)
          value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, adjustedDepth, !cutNode);
        
        // Post LMR continuation history updates
        update_continuation_histories(ss, movedPc, move.to_sq(), 1365);
      }
    } else if (!PvNode || moveCount > 1) {
      // Full-depth search when LMR is skipped
      value = -search<NonPV>(pos, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
    }

    // Full window search for PV nodes
    if (PvNode && (moveCount == 1 || value > alpha)) {
      Move childPv[MAX_PLY + 1];
      childPv[0] = Move::none();
      (ss + 1)->pv = childPv;
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
          // Increment cutoffCnt for LMR adjustment in parent
          ss->cutoffCnt += (extension < 2) || PvNode;
          break;
        }
        
        // Reduce other moves if we have found at least one score improvement
        if (depth > 2 && depth < 14 && !is_decisive(value))
          depth -= 2;

        alpha = value;
      }
    }
    
    // Track searched moves for history updates
    if (move != bestMove && moveCount <= SEARCHEDLIST_CAPACITY) {
      if (pos.capture(move)) {
        capturesSearched.push_back(move);
      } else {
        quietsSearched.push_back(move);
      }
    }
  }

  // Checkmate/stalemate detection
  if (moveCount == 0)
    bestValue = excludedMove ? alpha : (pos.checkers() ? mated_in(ss->ply) : value_draw(nodes));
  
  // Adjust best value for fail high cases
  if (bestValue >= beta && !is_decisive(bestValue) && !is_decisive(alpha))
    bestValue = (bestValue * depth + beta) / (depth + 1);

  // If there is a move that produces search value greater than alpha,
  // we update the stats of searched moves
  if (bestMove) {
    update_all_stats(pos, ss, bestMove, prevSq, quietsSearched,
                     capturesSearched, depth, ttMove, moveCount);
    if (!PvNode)
      history_update(ttMoveHistory, (bestMove == ttMove ? 809 : -865));
  }
  // Bonus for prior quiet countermove that caused the fail low
  else if (!priorCapture && prevSq != SQ_NONE) {
    int bonusScale = -215;
    bonusScale -= (ss - 1)->statScore / 100;
    bonusScale += std::min(56 * depth, 489);
    bonusScale += 184 * ((ss - 1)->moveCount > 8);
    bonusScale += 147 * (!ss->inCheck && bestValue <= ss->staticEval - 107);
    bonusScale += 156 * (!(ss - 1)->inCheck && bestValue <= -(ss - 1)->staticEval - 65);
    bonusScale = std::max(bonusScale, 0);
    
    int scaledBonus = std::min(141 * depth - 87, 1351) * bonusScale;
    
    update_continuation_histories(ss - 1, pos.piece_on(prevSq), prevSq,
                                  scaledBonus * 406 / 32768);
    
    int histIdx = ((ss - 1)->currentMove).from_sq() * 64 + ((ss - 1)->currentMove).to_sq();
    history_update(mainHistory[~us][histIdx], scaledBonus * 243 / 32768);
  }
  // Bonus for prior capture countermove that caused the fail low
  else if (priorCapture && prevSq != SQ_NONE) {
    Piece capturedPiece = pos.captured_piece();
    if (capturedPiece != NO_PIECE)
      history_update(captureHistory[pos.piece_on(prevSq)][prevSq][type_of(capturedPiece)], 1012);
  }
  
  // For PV nodes, clamp bestValue to maxValue (from TB)
  Value maxValue = VALUE_INFINITE;
  if (PvNode)
    bestValue = std::min(bestValue, maxValue);
  
  // If no good move is found and the previous position was ttPv, then the previous
  // opponent move is probably good and the new position is added to the search tree
  if (bestValue <= alpha)
    ss->ttPv = ss->ttPv || (ss - 1)->ttPv;

  // Update full correction history when search result differs from static eval
  if (!ss->inCheck && !(bestMove && pos.capture(bestMove)) &&
      (bestValue > ss->staticEval) == bool(bestMove)) {
    int diff = bestValue - ss->staticEval;
    int bonus = std::clamp(diff * depth / (bestMove ? 10 : 8), -CORRECTION_HISTORY_LIMIT / 4,
                           CORRECTION_HISTORY_LIMIT / 4);

    Move prevMove = ss->ply >= 1 ? (ss - 1)->currentMove : Move::none();
    ContinuationCorrectionHistory *contCorr2 =
        ss->ply >= 2 ? (ss - 2)->continuationCorrectionHistory : nullptr;
    ContinuationCorrectionHistory *contCorr4 =
        ss->ply >= 4 ? (ss - 4)->continuationCorrectionHistory : nullptr;

    update_full_correction_history(*correctionHistory, contCorr2, contCorr4,
                                   pos, prevMove, bonus);
  }

  // Update TT with adjusted value for storage
  // Note: static evaluation is saved as it was before correction history
  if (!ss->excludedMove && !(rootNode && pvIdx > 0)) {
    tte->save(posKey, value_to_tt(bestValue, ss->ply), ss->ttPv,
              bestValue >= beta    ? BOUND_LOWER
              : PvNode && bestMove ? BOUND_EXACT
                                   : BOUND_UPPER,
              moveCount != 0 ? depth : std::min(MAX_PLY - 1, depth + 6),
              bestMove, unadjustedStaticEval, TT->generation());
  }

  // Update root move scores and effort
  if (rootNode) {
    // Find the move to update - use bestMove if found, otherwise first root
    // move
    Move moveToUpdate =
        bestMove ? bestMove
                 : (rootMoves.empty() ? Move::none() : rootMoves[pvIdx].pv[0]);

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

  // Check if we have an upcoming move that draws by repetition
  if (alpha < VALUE_DRAW && pos.upcoming_repetition(ss->ply)) {
    alpha = value_draw(nodes);
    if (alpha >= beta)
      return alpha;
  }

  // Draw detection with randomization
  if (pos.is_draw(ss->ply))
    return value_draw(nodes);

  // Max ply
  if (ss->ply >= MAX_PLY)
    return pos.checkers() ? value_draw(nodes) : evaluate(pos);

  // Initialize
  ss->inCheck = pos.checkers();
  Move bestMove = Move::none();
  Value bestValue;
  int moveCount = 0;

  // TT probe
  Key posKey = pos.key();
  bool ttHit;
  TTEntry *tte = TT->probe(posKey, ttHit);
  Value ttValue = ttHit
                      ? value_from_tt(tte->value(), ss->ply, pos.rule50_count())
                      : VALUE_NONE;
  bool pvHit = ttHit && tte->is_pv();

  // TT cutoff at non-PV nodes
  if (!PvNode && ttHit && tte->depth() >= DEPTH_QS &&
      ttValue != VALUE_NONE &&
      (tte->bound() & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER)))
    return ttValue;

  // Static evaluation
  Value unadjustedStaticEval = VALUE_NONE;
  Value futilityBase;
  
  if (ss->inCheck) {
    bestValue = futilityBase = -VALUE_INFINITE;
  } else {
    // Compute correction value for static eval adjustment
    Move prevMove = ss->ply >= 1 ? (ss - 1)->currentMove : Move::none();
    ContinuationCorrectionHistory *contCorr2 =
        ss->ply >= 2 ? (ss - 2)->continuationCorrectionHistory : nullptr;
    ContinuationCorrectionHistory *contCorr4 =
        ss->ply >= 4 ? (ss - 4)->continuationCorrectionHistory : nullptr;
    int correctionValue = compute_full_correction_value(
        *correctionHistory, contCorr2, contCorr4, pos, prevMove);
    
    if (ttHit) {
      unadjustedStaticEval = tte->eval();
      if (unadjustedStaticEval == VALUE_NONE)
        unadjustedStaticEval = evaluate(pos);
      ss->staticEval = bestValue = 
          to_corrected_static_eval(unadjustedStaticEval, correctionValue);
      
      // ttValue can be used as a better position evaluation
      if (ttValue != VALUE_NONE && !is_decisive(ttValue) &&
          (tte->bound() & (ttValue > bestValue ? BOUND_LOWER : BOUND_UPPER)))
        bestValue = ttValue;
    } else {
      unadjustedStaticEval = evaluate(pos);
      ss->staticEval = bestValue = 
          to_corrected_static_eval(unadjustedStaticEval, correctionValue);
    }

    // Stand pat
    if (bestValue >= beta) {
      if (!is_decisive(bestValue))
        bestValue = (bestValue + beta) / 2;
      
      if (!ttHit)
        tte->save(posKey, value_to_tt(bestValue, ss->ply), false, BOUND_LOWER,
                  DEPTH_QS, Move::none(), unadjustedStaticEval, TT->generation());
      return bestValue;
    }

    if (bestValue > alpha)
      alpha = bestValue;

    futilityBase = ss->staticEval + 351;
  }

  // Generate captures or evasions
  Move moves[MAX_MOVES];
  Move *end = ss->inCheck ? generate<EVASIONS>(pos, moves)
                          : generate<CAPTURES>(pos, moves);

  // Search moves
  for (Move *m = moves; m != end; ++m) {
    Move move = *m;

    if (!pos.legal(move))
      continue;

    bool givesCheck = pos.gives_check(move);
    moveCount++;

    // Pruning (not in check and not losing)
    if (bestValue > VALUE_MATED_IN_MAX_PLY && !ss->inCheck) {
      // Futility pruning
      if (!givesCheck && move.type_of() != PROMOTION && moveCount > 2)
        continue;
      
      // SEE pruning
      if (!pos.see_ge(move, -80))
        continue;
    }

    // Make the move
    StateInfo st;
    ss->currentMove = move;
    pos.do_move(move, st, givesCheck);

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
  if (ss->inCheck && bestValue == -VALUE_INFINITE)
    return mated_in(ss->ply);

  // Adjust bestValue for fail high
  if (!is_decisive(bestValue) && bestValue > beta)
    bestValue = (bestValue + beta) / 2;

  // Update TT
  tte->save(posKey, value_to_tt(bestValue, ss->ply), pvHit,
            bestValue >= beta ? BOUND_LOWER : BOUND_UPPER, DEPTH_QS, bestMove,
            unadjustedStaticEval, TT->generation());

  return bestValue;
}

// Evaluate position using GPU-accelerated NNUE with optimism blending
Value Worker::evaluate(const Position &pos) {
  Value rawEval = Eval::evaluate(pos);

  // Blend optimism with evaluation based on material
  int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
  int opt = optimism[pos.side_to_move()];

  // Stockfish formula: (nnue * (77871 + material) + optimism * (7191 +
  // material)) / 77871
  Value v = (rawEval * (77871 + material) + opt * (7191 + material)) / 77871;

  // Damp down evaluation linearly when shuffling (rule50)
  v -= v * pos.rule50_count() / 199;

  // Guarantee evaluation does not hit tablebase range
  return std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1,
                    VALUE_TB_WIN_IN_MAX_PLY - 1);
}

// Reduction function - simplified for stability
Depth Worker::reduction(bool improving, Depth depth, int moveCount, int /*delta*/) const {
  // Simple logarithmic reduction
  int r = reductions[std::min(depth, MAX_PLY - 1)] * reductions[std::min(moveCount, MAX_MOVES - 1)] / 1024;
  
  // Adjust for improving
  if (!improving)
    r += 1;
  
  return std::max(0, std::min(r, depth - 1));
}

// =============================================================================
// Skill Level Implementation
// =============================================================================
// When playing with strength handicap, choose the best move among a set of
// RootMoves using a statistical rule dependent on 'level'. Idea by Heinz van
// Saanen.

Move Skill::pick_best(const RootMoves &rootMoves, size_t multiPV) {
  // Simple PRNG for randomization
  static uint64_t seed = now();
  auto rand = [&]() {
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    return seed * 0x2545F4914F6CDD1DULL;
  };

  if (rootMoves.empty())
    return Move::none();

  // RootMoves are already sorted by score in descending order
  Value topScore = rootMoves[0].score;
  int delta = std::min(
      topScore - rootMoves[std::min(multiPV - 1, rootMoves.size() - 1)].score,
      int(PawnValue));
  int maxScore = -VALUE_INFINITE;
  double weakness = 120 - 2 * level;

  // Choose best move. For each move score we add two terms, both dependent on
  // weakness. One is deterministic and bigger for weaker levels, and one is
  // random. Then we choose the move with the resulting highest score.
  for (size_t i = 0; i < std::min(multiPV, rootMoves.size()); ++i) {
    // This is the magic formula from Stockfish
    int push = int(weakness * int(topScore - rootMoves[i].score) +
                   delta * (rand() % int(weakness + 1))) /
               128;

    if (rootMoves[i].score + push >= maxScore) {
      maxScore = rootMoves[i].score + push;
      best = rootMoves[i].pv[0];
    }
  }

  return best;
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

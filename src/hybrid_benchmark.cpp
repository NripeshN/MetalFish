/*
  MetalFish - Hybrid Search Benchmark Suite
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive benchmarking for hybrid MCTS-AB search validation.
  Includes: match playing, profiling, classifier analysis, ablations.

  Licensed under GPL-3.0
*/

#include "core/bitboard.h"
#include "core/misc.h"
#include "core/position.h"
#include "eval/evaluate.h"
#include "gpu/gpu_nnue_integration.h"
#include "mcts/ab_integration.h"
#include "mcts/enhanced_hybrid_search.h"
#include "mcts/hybrid_search.h"
#include "mcts/position_classifier.h"
#include "search/search.h"
#include "search/tt.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

using namespace MetalFish;

// Test positions for various phases
static const char *TEST_POSITIONS[] = {
    // Opening positions
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    // Middlegame positions
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 5",
    "r2qkb1r/ppp1pppp/2n2n2/3p1b2/3P1B2/2N2N2/PPP1PPPP/R2QKB1R w KQkq - 4 5",
    "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    // Tactical positions
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r2qk2r/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 0 8",
    "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 5 6",
    // Endgame positions
    "8/8/8/8/8/5K2/8/4k2R w - - 0 1",
    "8/8/8/4k3/8/8/4K3/4Q3 w - - 0 1",
    "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1",
    "8/5k2/8/8/8/8/5PP1/5K2 w - - 0 1",
    "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
};
static const int NUM_TEST_POSITIONS =
    sizeof(TEST_POSITIONS) / sizeof(TEST_POSITIONS[0]);

// Match result
struct MatchResult {
  int wins = 0;
  int draws = 0;
  int losses = 0;
  double elo_diff = 0.0;
  double elo_error = 0.0;
};

// Calculate Elo difference from score
double calculate_elo(int wins, int draws, int losses) {
  int total = wins + draws + losses;
  if (total == 0)
    return 0.0;
  double score = (wins + 0.5 * draws) / total;
  if (score <= 0.0)
    return -400.0;
  if (score >= 1.0)
    return 400.0;
  return -400.0 * std::log10(1.0 / score - 1.0);
}

// Classifier distribution analysis
struct ClassifierStats {
  int counts[5] = {0, 0, 0, 0, 0}; // For each PositionType
  int total = 0;
  int stability_tests = 0;
  int stability_flips = 0;

  void add(MCTS::PositionType type) {
    counts[static_cast<int>(type)]++;
    total++;
  }

  void print() const {
    std::cout << "\n=== Classifier Distribution ===\n";
    const char *names[] = {"HIGHLY_TACTICAL", "TACTICAL", "BALANCED",
                           "STRATEGIC", "HIGHLY_STRATEGIC"};
    for (int i = 0; i < 5; i++) {
      double pct = total > 0 ? 100.0 * counts[i] / total : 0.0;
      std::cout << "  " << std::setw(18) << names[i] << ": " << std::setw(6)
                << counts[i] << " (" << std::fixed << std::setprecision(1)
                << pct << "%)\n";
    }
    std::cout << "  Total positions: " << total << "\n";
    if (stability_tests > 0) {
      double flip_rate = 100.0 * stability_flips / stability_tests;
      std::cout << "  Stability: " << stability_flips << "/" << stability_tests
                << " flips (" << std::fixed << std::setprecision(1) << flip_rate
                << "%)\n";
    }
  }
};

// MCTS profiling results
struct MCTSProfile {
  double selection_pct = 0;
  double expansion_pct = 0;
  double evaluation_pct = 0;
  double backprop_pct = 0;
  double queue_pct = 0;
  uint64_t total_nodes = 0;
  double nps = 0;

  void print() const {
    std::cout << "\n=== MCTS Profiling Breakdown ===\n";
    std::cout << "  Selection:    " << std::fixed << std::setprecision(1)
              << selection_pct << "%\n";
    std::cout << "  Expansion:    " << expansion_pct << "%\n";
    std::cout << "  Evaluation:   " << evaluation_pct << "%\n";
    std::cout << "  Backprop:     " << backprop_pct << "%\n";
    std::cout << "  Queue/Other:  " << queue_pct << "%\n";
    std::cout << "  Total nodes:  " << total_nodes << "\n";
    std::cout << "  NPS:          " << std::fixed << std::setprecision(0) << nps
              << "\n";
  }
};

// Run classifier analysis on positions
ClassifierStats run_classifier_analysis(const std::vector<std::string> &fens) {
  ClassifierStats stats;
  MCTS::PositionClassifier classifier;

  for (const auto &fen : fens) {
    Position pos;
    StateInfo st;
    pos.set(fen, false, &st);

    auto features = classifier.analyze(pos);
    MCTS::PositionType type = classifier.quick_classify(pos);
    stats.add(type);

    // Stability test: make one legal move and check if class changes
    MoveList<LEGAL> moves(pos);
    if (moves.size() > 0) {
      StateInfo st2;
      pos.do_move(*moves.begin(), st2);
      MCTS::PositionType type_after = classifier.quick_classify(pos);
      stats.stability_tests++;
      if (type_after != type) {
        stats.stability_flips++;
      }
      pos.undo_move(*moves.begin());
    }
  }

  return stats;
}

// Run MCTS profiling
MCTSProfile run_mcts_profiling(GPU::GPUNNUEManager *gpu_manager,
                               const std::string &fen, int time_ms) {
  MCTSProfile profile;

  MCTS::HybridSearchConfig config;
  config.num_search_threads = 1;
  auto search = MCTS::create_hybrid_search(gpu_manager, config);

  Position pos;
  StateInfo st;
  pos.set(fen, false, &st);

  MCTS::MCTSPositionHistory history;
  history.reset(fen);

  Search::LimitsType limits;
  limits.movetime = time_ms;

  std::atomic<bool> done{false};
  MCTS::MCTSMove best_move;

  auto start = std::chrono::steady_clock::now();

  search->start_search(
      history, limits,
      [&](MCTS::MCTSMove move, MCTS::MCTSMove ponder) {
        best_move = move;
        done = true;
      },
      nullptr);

  search->wait();

  auto end = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  const auto &stats = search->stats();
  stats.get_profile_breakdown(profile.selection_pct, profile.expansion_pct,
                              profile.evaluation_pct, profile.backprop_pct,
                              profile.queue_pct);

  profile.total_nodes = stats.mcts_nodes.load();
  profile.nps = elapsed_ms > 0 ? profile.total_nodes * 1000.0 / elapsed_ms : 0;

  return profile;
}

// Pure MCTS search (for comparison)
Move run_pure_mcts(GPU::GPUNNUEManager *gpu_manager, const Position &pos,
                   int time_ms, uint64_t &nodes_out) {
  MCTS::HybridSearchConfig config;
  config.num_search_threads = 1;
  auto search = MCTS::create_hybrid_search(gpu_manager, config);

  MCTS::MCTSPositionHistory history;
  history.reset(pos.fen());

  Search::LimitsType limits;
  limits.movetime = time_ms;

  MCTS::MCTSMove best_move;
  std::atomic<bool> done{false};

  search->start_search(
      history, limits,
      [&](MCTS::MCTSMove move, MCTS::MCTSMove ponder) {
        best_move = move;
        done = true;
      },
      nullptr);

  search->wait();
  nodes_out = search->stats().mcts_nodes.load();

  return best_move.to_stockfish();
}

// Pure Alpha-Beta search (for comparison)
Move run_pure_ab(TranspositionTable *tt, const Position &pos, int depth,
                 uint64_t &nodes_out) {
  MCTS::ABSearcher searcher;
  searcher.initialize(tt);

  MCTS::ABSearchConfig config;
  config.max_depth = depth;
  config.use_tt = true;
  searcher.set_config(config);

  auto result = searcher.search(pos, depth);
  nodes_out = searcher.nodes_searched();

  return result.best_move;
}

// Hybrid search (classifier + verifier)
Move run_hybrid(GPU::GPUNNUEManager *gpu_manager, const Position &pos,
                int time_ms, uint64_t &mcts_nodes, uint64_t &ab_nodes) {
  auto search = MCTS::create_enhanced_hybrid_search(gpu_manager);
  if (!search) {
    return Move::none();
  }

  Search::LimitsType limits;
  limits.movetime = time_ms;

  Move best_move = Move::none();
  std::atomic<bool> done{false};

  search->start_search(
      pos, limits,
      [&](Move move, Move ponder) {
        best_move = move;
        done = true;
      },
      nullptr);

  search->wait();

  mcts_nodes = search->stats().mcts_nodes.load();
  ab_nodes = search->stats().ab_nodes.load();

  return best_move;
}

// Simple game result from position evaluation
enum class SimpleResult { WHITE_WIN, DRAW, BLACK_WIN, ONGOING };

SimpleResult get_simple_result(const Position &pos) {
  if (pos.is_draw(0))
    return SimpleResult::DRAW;

  MoveList<LEGAL> moves(pos);
  if (moves.size() == 0) {
    if (pos.checkers()) {
      return pos.side_to_move() == WHITE ? SimpleResult::BLACK_WIN
                                         : SimpleResult::WHITE_WIN;
    }
    return SimpleResult::DRAW;
  }

  return SimpleResult::ONGOING;
}

// Play a single game between two search functions
// Returns: 1 for white win, 0 for draw, -1 for black win
int play_game(GPU::GPUNNUEManager *gpu_manager, TranspositionTable *tt,
              bool white_is_hybrid, bool black_is_hybrid, int move_time_ms,
              int max_moves = 200) {
  Position pos;
  StateInfo states[512];
  int state_idx = 0;
  pos.set(StartFEN, false, &states[state_idx++]);

  for (int move_num = 0; move_num < max_moves; move_num++) {
    SimpleResult result = get_simple_result(pos);
    if (result != SimpleResult::ONGOING) {
      if (result == SimpleResult::WHITE_WIN)
        return 1;
      if (result == SimpleResult::BLACK_WIN)
        return -1;
      return 0;
    }

    bool is_white = pos.side_to_move() == WHITE;
    bool use_hybrid = is_white ? white_is_hybrid : black_is_hybrid;

    Move move = Move::none();
    uint64_t nodes1 = 0, nodes2 = 0;

    if (use_hybrid) {
      move = run_hybrid(gpu_manager, pos, move_time_ms, nodes1, nodes2);
    } else {
      move = run_pure_mcts(gpu_manager, pos, move_time_ms, nodes1);
    }

    if (move == Move::none()) {
      // No legal move found, game over
      break;
    }

    pos.do_move(move, states[state_idx++]);
  }

  // Max moves reached, evaluate final position
  Value eval = Eval::simple_eval(pos);
  if (eval > 100)
    return 1;
  if (eval < -100)
    return -1;
  return 0;
}

// Run match between two configurations
MatchResult run_match(GPU::GPUNNUEManager *gpu_manager, TranspositionTable *tt,
                      bool config1_hybrid, bool config2_hybrid, int num_games,
                      int move_time_ms) {
  MatchResult result;

  std::cout << "Playing " << num_games << " games...\n";

  for (int i = 0; i < num_games; i++) {
    // Alternate colors
    bool config1_white = (i % 2 == 0);

    int game_result;
    if (config1_white) {
      game_result = play_game(gpu_manager, tt, config1_hybrid, config2_hybrid,
                              move_time_ms);
    } else {
      game_result = play_game(gpu_manager, tt, config2_hybrid, config1_hybrid,
                              move_time_ms);
      game_result = -game_result; // Flip for config1's perspective
    }

    if (game_result > 0)
      result.wins++;
    else if (game_result < 0)
      result.losses++;
    else
      result.draws++;

    if ((i + 1) % 10 == 0) {
      std::cout << "  Game " << (i + 1) << "/" << num_games << ": +"
                << result.wins << " =" << result.draws << " -" << result.losses
                << "\n";
    }
  }

  result.elo_diff = calculate_elo(result.wins, result.draws, result.losses);

  // Approximate error (simplified)
  int total = result.wins + result.draws + result.losses;
  if (total > 0) {
    double score = (result.wins + 0.5 * result.draws) / total;
    double variance = score * (1 - score) / total;
    result.elo_error = 400.0 * std::sqrt(variance) / std::log(10);
  }

  return result;
}

// Main benchmark function
void run_hybrid_benchmark(GPU::GPUNNUEManager *gpu_manager,
                          TranspositionTable *tt) {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║     MetalFish Hybrid Search Validation Suite             ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  // 1. Classifier Distribution Analysis
  std::cout << "\n=== 1. Classifier Distribution Analysis ===\n";
  std::vector<std::string> test_fens;
  for (int i = 0; i < NUM_TEST_POSITIONS; i++) {
    test_fens.push_back(TEST_POSITIONS[i]);
  }
  auto classifier_stats = run_classifier_analysis(test_fens);
  classifier_stats.print();

  // 2. MCTS Profiling
  std::cout << "\n=== 2. MCTS Profiling (5 second search) ===\n";
  auto profile = run_mcts_profiling(gpu_manager, StartFEN, 5000);
  profile.print();

  // 3. Quick match: Hybrid vs Pure MCTS
  std::cout << "\n=== 3. Quick Match: Hybrid vs Pure MCTS (20 games) ===\n";
  auto match1 = run_match(gpu_manager, tt, true, false, 20, 500);
  std::cout << "\nResult: +" << match1.wins << " =" << match1.draws << " -"
            << match1.losses << "\n";
  std::cout << "Elo: " << std::fixed << std::setprecision(0) << match1.elo_diff
            << " +/- " << match1.elo_error << "\n";

  // 4. Summary
  std::cout << "\n=== Summary ===\n";
  std::cout << "Classifier: " << classifier_stats.total << " positions, "
            << classifier_stats.stability_flips << " stability flips\n";
  std::cout << "MCTS NPS: " << std::fixed << std::setprecision(0) << profile.nps
            << "\n";
  std::cout << "Hybrid vs MCTS: " << match1.elo_diff << " Elo\n";

  std::cout << "\nBenchmark complete.\n";
}

// Entry point for benchmark
extern "C" void hybrid_benchmark_main(GPU::GPUNNUEManager *gpu_manager,
                                      TranspositionTable *tt) {
  run_hybrid_benchmark(gpu_manager, tt);
}

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Test Suite

  Tests all MCTS and hybrid search components including:
  - Stockfish adapter
  - Thread-safe MCTS
*/

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "core/bitboard.h"
#include "core/position.h"
#include "gpu/gpu_nnue_integration.h"
#include "mcts/ab_integration.h"
#include "mcts/position_classifier.h"
#include "mcts/stockfish_adapter.h"
#include "mcts/thread_safe_mcts.h"

using namespace MetalFish;
using namespace MetalFish::MCTS;

namespace {

// Test utilities
static int g_tests_passed = 0;
static int g_tests_failed = 0;

class TestCase {
public:
  TestCase(const char *name) : name_(name), passed_(true) {
    std::cout << "  Testing " << name_ << "... " << std::flush;
  }

  ~TestCase() {
    if (passed_) {
      std::cout << "PASSED" << std::endl;
      g_tests_passed++;
    } else {
      std::cout << "FAILED" << std::endl;
      g_tests_failed++;
    }
  }

  void fail(const char *msg, const char *file, int line) {
    if (passed_) {
      std::cout << "FAILED\n";
      passed_ = false;
    }
    std::cout << "    " << file << ":" << line << ": " << msg << std::endl;
  }

  bool passed() const { return passed_; }

private:
  const char *name_;
  bool passed_;
};

#define EXPECT(tc, cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      tc.fail(#cond, __FILE__, __LINE__);                                      \
    }                                                                          \
  } while (0)

// Test FENs
const char *TEST_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // Starting
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // Endgame
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
};

// ============================================================================
// Stockfish Adapter Tests
// ============================================================================

bool test_mcts_move() {
  TestCase tc("MCTSMove");

  // Create from Stockfish move
  Move sf_move(SQ_E2, SQ_E4);
  MCTSMove mcts_move = MCTSMove::FromStockfish(sf_move);

  EXPECT(tc, mcts_move.to_stockfish() == sf_move);
  EXPECT(tc, mcts_move.from() == SQ_E2);
  EXPECT(tc, mcts_move.to() == SQ_E4);

  // Test promotion
  Move promo_move = Move::make<PROMOTION>(SQ_E7, SQ_E8, QUEEN);
  MCTSMove mcts_promo = MCTSMove::FromStockfish(promo_move);
  EXPECT(tc, mcts_promo.to_stockfish() == promo_move);

  // Test null move
  MCTSMove null_move;
  EXPECT(tc, null_move.to_stockfish() == Move::none());

  return tc.passed();
}

bool test_mcts_position() {
  TestCase tc("MCTSPosition");

  MCTSPosition mcts_pos;
  mcts_pos.set_from_fen(TEST_FENS[0]);

  EXPECT(tc, mcts_pos.side_to_move() == WHITE);
  EXPECT(tc, mcts_pos.hash() != 0);

  return tc.passed();
}

// ============================================================================
// Position Classifier Tests
// ============================================================================

bool test_position_classifier_basic() {
  TestCase tc("PositionClassifierBasic");

  PositionClassifier classifier;

  Position pos;
  StateInfo st;
  pos.set(TEST_FENS[0], false, &st);

  auto features = classifier.analyze(pos);

  // Starting position should not be in check
  EXPECT(tc, !features.in_check);

  // Should have some tactical score
  EXPECT(tc, features.tactical_score >= 0.0f);
  EXPECT(tc, features.tactical_score <= 1.0f);

  return tc.passed();
}

bool test_position_classifier_tactical() {
  TestCase tc("PositionClassifierTactical");

  PositionClassifier classifier;

  Position pos;
  StateInfo st;
  pos.set(TEST_FENS[1], false, &st); // Kiwipete - complex tactical

  auto features = classifier.analyze(pos);

  // Tactical positions should have higher complexity
  EXPECT(tc, features.complexity >= 0.0f);

  return tc.passed();
}

// ============================================================================
// AB Integration Tests
// ============================================================================

bool test_ab_search_result() {
  TestCase tc("ABSearchResult");

  ABSearchResult result;
  result.best_move = Move(SQ_E2, SQ_E4);
  result.score = 50;
  result.depth = 10;
  result.nodes_searched = 10000;

  EXPECT(tc, result.best_move == Move(SQ_E2, SQ_E4));
  EXPECT(tc, result.score == 50);
  EXPECT(tc, result.depth == 10);
  EXPECT(tc, result.nodes_searched == 10000);

  return tc.passed();
}

bool test_ab_search_config() {
  TestCase tc("ABSearchConfig");

  ABSearchConfig config;
  config.max_depth = 20;
  config.use_tt = true;
  config.use_lmr = true;

  EXPECT(tc, config.max_depth == 20);
  EXPECT(tc, config.use_tt);
  EXPECT(tc, config.use_lmr);

  return tc.passed();
}

bool test_hybrid_search_bridge() {
  TestCase tc("HybridSearchBridge");

  HybridSearchBridge bridge;

  // Test without initialization
  EXPECT(tc, !bridge.has_engine());

  // Get stats (should be zero)
  auto stats = bridge.get_stats();
  EXPECT(tc, stats.verifications == 0);
  EXPECT(tc, stats.overrides == 0);

  return tc.passed();
}

// ============================================================================
// Thread-Safe MCTS Tests
// ============================================================================

bool test_thread_safe_mcts_config() {
  TestCase tc("ThreadSafeMCTSConfig");

  ThreadSafeMCTSConfig config;

  // Test default values
  EXPECT(tc, config.cpuct > 0.0f);
  EXPECT(tc, config.fpu_reduction >= 0.0f);
  EXPECT(tc, config.num_threads >= 0);

  // Test auto-tune
  config.auto_tune(4);
  EXPECT(tc, config.min_batch_size >= 1);
  EXPECT(tc, config.max_batch_size >= config.min_batch_size);

  return tc.passed();
}

bool test_thread_safe_mcts() {
  TestCase tc("ThreadSafeMCTS");

  // Configure with 2 threads for testing
  ThreadSafeMCTSConfig config;
  config.num_threads = 2;
  config.cpuct = 2.5f;
  config.add_dirichlet_noise = false; // Deterministic for testing

  // Create MCTS without GPU (uses simple eval)
  auto mcts = std::make_unique<ThreadSafeMCTS>(config);
  EXPECT(tc, mcts != nullptr);

  // Search from starting position for 500ms
  Search::LimitsType limits;
  limits.movetime = 500;

  bool got_best_move = false;
  Move best_move = Move::none();

  auto best_cb = [&](Move best, Move ponder) {
    got_best_move = true;
    best_move = best;
  };

  mcts->start_search("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                     limits, best_cb, nullptr);
  mcts->wait();

  EXPECT(tc, got_best_move);
  EXPECT(tc, best_move != Move::none());

  // Check that we got some nodes
  const auto &stats = mcts->stats();
  EXPECT(tc, stats.total_nodes > 0);

  std::cout << "\n    Nodes: " << stats.total_nodes.load();
  std::cout << "\n    Iterations: " << stats.total_iterations.load();
  std::cout << std::endl << "  ";

  return tc.passed();
}

// ============================================================================
// Main Test Runner
// ============================================================================

} // namespace

bool test_mcts() {
  std::cout << "\n=== MCTS Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[Stockfish Adapter]" << std::endl;
  test_mcts_move();
  test_mcts_position();

  std::cout << "\n[Position Classifier]" << std::endl;
  test_position_classifier_basic();
  test_position_classifier_tactical();

  std::cout << "\n[AB Integration]" << std::endl;
  test_ab_search_result();
  test_ab_search_config();
  test_hybrid_search_bridge();

  std::cout << "\n[Thread-Safe MCTS]" << std::endl;
  test_thread_safe_mcts_config();
  test_thread_safe_mcts();

  std::cout << "\n=== MCTS Test Summary ===" << std::endl;
  std::cout << "  Passed: " << g_tests_passed << std::endl;
  std::cout << "  Failed: " << g_tests_failed << std::endl;

  if (g_tests_failed > 0) {
    std::cout << "  SOME TESTS FAILED!" << std::endl;
    return false;
  }

  std::cout << "All MCTS tests passed!" << std::endl;
  return true;
}

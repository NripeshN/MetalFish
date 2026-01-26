/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Hybrid Search Tests - Position Classifier, AB Integration, Parallel Search
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "mcts/ab_integration.h"
#include "mcts/parallel_hybrid_search.h"
#include "mcts/position_classifier.h"
#include "mcts/position_adapter.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace MetalFish;
using namespace MetalFish::MCTS;

namespace {

static int g_tests_passed = 0;
static int g_tests_failed = 0;

class TestCase {
public:
  TestCase(const char *name) : name_(name), passed_(true) {
    std::cout << "  " << name_ << "... " << std::flush;
  }
  ~TestCase() {
    if (passed_) {
      std::cout << "OK" << std::endl;
      g_tests_passed++;
    } else {
      g_tests_failed++;
    }
  }
  void fail(const char *msg, int line) {
    if (passed_) {
      std::cout << "FAILED\n";
      passed_ = false;
    }
    std::cout << "    Line " << line << ": " << msg << std::endl;
  }
  bool passed() const { return passed_; }

private:
  const char *name_;
  bool passed_;
};

#define EXPECT(tc, cond)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      tc.fail(#cond, __LINE__);                                                \
    }                                                                          \
  } while (0)

// ============================================================================
// Position Classifier Tests
// ============================================================================

void test_classifier() {
  {
    TestCase tc("Features default values");
    PositionFeatures features;

    EXPECT(tc, !features.in_check);
    EXPECT(tc, features.num_captures == 0);
    EXPECT(tc, features.tactical_score >= 0.0f);
    EXPECT(tc, features.strategic_score >= 0.0f);
  }
  {
    TestCase tc("Starting position analysis");
    PositionClassifier classifier;
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    auto features = classifier.analyze(pos);

    EXPECT(tc, !features.in_check);
    EXPECT(tc, features.tactical_score >= 0.0f);
    EXPECT(tc, features.tactical_score <= 1.0f);
    EXPECT(tc, features.strategic_score >= 0.0f);
    EXPECT(tc, features.strategic_score <= 1.0f);
  }
  {
    TestCase tc("Tactical position analysis");
    PositionClassifier classifier;
    StateInfo st;
    Position pos;
    pos.set(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        false, &st);

    auto features = classifier.analyze(pos);
    EXPECT(tc, features.complexity >= 0.0f);
  }
  {
    TestCase tc("Endgame detection");
    PositionClassifier classifier;
    StateInfo st;
    Position pos;
    pos.set("8/8/4k3/8/8/4K3/8/8 w - - 0 1", false, &st);

    auto features = classifier.analyze(pos);
    EXPECT(tc, features.is_endgame);
  }
  {
    TestCase tc("Quick classification");
    PositionClassifier classifier;
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    PositionType type = classifier.quick_classify(pos);

    EXPECT(tc, type == PositionType::BALANCED ||
                   type == PositionType::STRATEGIC ||
                   type == PositionType::HIGHLY_STRATEGIC);
  }
  {
    TestCase tc("Quiet position check");
    PositionClassifier classifier;
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    EXPECT(tc, classifier.is_quiet(pos));
  }
}

// ============================================================================
// Strategy Selector Tests
// ============================================================================

void test_strategy() {
  {
    TestCase tc("Strategy selection");
    StrategySelector selector;
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    SearchStrategy strategy = selector.get_strategy(pos);

    EXPECT(tc, strategy.mcts_weight >= 0.0f && strategy.mcts_weight <= 1.0f);
    EXPECT(tc, strategy.ab_weight >= 0.0f && strategy.ab_weight <= 1.0f);
    EXPECT(tc, strategy.cpuct > 0.0f);
    EXPECT(tc, strategy.ab_depth > 0);
  }
  {
    TestCase tc("Time adjustment");
    StrategySelector selector;
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    SearchStrategy strategy = selector.get_strategy(pos);
    selector.adjust_for_time(strategy, 1000, 0);

    EXPECT(tc, strategy.time_multiplier > 0.0f);
  }
}

// ============================================================================
// AB Integration Tests
// ============================================================================

void test_ab_integration() {
  {
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
  }
  {
    TestCase tc("ABSearchConfig");
    ABSearchConfig config;
    config.max_depth = 20;
    config.use_tt = true;
    config.use_lmr = true;
    config.use_null_move = true;

    EXPECT(tc, config.max_depth == 20);
    EXPECT(tc, config.use_tt);
    EXPECT(tc, config.use_lmr);
    EXPECT(tc, config.use_null_move);
  }
  {
    TestCase tc("HybridBridgeStats");
    HybridSearchBridge bridge;
    auto stats = bridge.get_stats();

    EXPECT(tc, stats.verifications == 0);
    EXPECT(tc, stats.overrides == 0);
    EXPECT(tc, stats.ab_nodes == 0);
  }
}

// ============================================================================
// Position Adapter Tests
// ============================================================================

void test_adapter() {
  {
    TestCase tc("Move conversion");
    Move sf_move(SQ_E2, SQ_E4);
    MCTSMove mcts_move = MCTSMove::FromInternal(sf_move);

    EXPECT(tc, mcts_move.to_internal() == sf_move);
    EXPECT(tc, mcts_move.from() == SQ_E2);
    EXPECT(tc, mcts_move.to() == SQ_E4);
  }
  {
    TestCase tc("Promotion conversion");
    Move promo = Move::make<PROMOTION>(SQ_E7, SQ_E8, QUEEN);
    MCTSMove mcts_promo = MCTSMove::FromInternal(promo);

    EXPECT(tc, mcts_promo.to_internal() == promo);
    EXPECT(tc, mcts_promo.to_internal().promotion_type() == QUEEN);
  }
  {
    TestCase tc("Position adapter");
    MCTSPosition mcts_pos;
    mcts_pos.set_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    EXPECT(tc, mcts_pos.side_to_move() == WHITE);
    EXPECT(tc, mcts_pos.hash() != 0);
  }
}

// ============================================================================
// Shared State Tests
// ============================================================================

void test_shared_state() {
  {
    TestCase tc("ABSharedState reset");
    ABSharedState state;
    state.reset();

    EXPECT(tc, state.best_move_raw.load() == 0);
    EXPECT(tc, state.best_score.load() == 0);
    EXPECT(tc, state.completed_depth.load() == 0);
    EXPECT(tc, !state.ab_running.load());
    EXPECT(tc, !state.has_result.load());
  }
  {
    TestCase tc("ABSharedState update");
    ABSharedState state;
    state.reset();

    Move m(SQ_E2, SQ_E4);
    state.set_best_move(m, 50, 10, 1000);

    EXPECT(tc, state.get_best_move() == m);
    EXPECT(tc, state.best_score.load() == 50);
    EXPECT(tc, state.completed_depth.load() == 10);
    EXPECT(tc, state.has_result.load());
  }
  {
    TestCase tc("ABSharedState move scores");
    ABSharedState state;
    state.reset();

    Move m1(SQ_E2, SQ_E4);
    Move m2(SQ_D2, SQ_D4);

    state.update_move_score(m1, 50, 5);
    state.update_move_score(m2, 30, 5);

    EXPECT(tc, state.num_scored_moves.load() == 2);
  }
  {
    TestCase tc("MCTSSharedState reset");
    MCTSSharedState state;
    state.reset();

    EXPECT(tc, state.best_move_raw.load() == 0);
    EXPECT(tc, state.best_q.load() == 0.0f);
    EXPECT(tc, state.best_visits.load() == 0);
    EXPECT(tc, !state.mcts_running.load());
  }
}

// ============================================================================
// Config Tests
// ============================================================================

void test_hybrid_config() {
  {
    TestCase tc("Config defaults");
    ParallelHybridConfig config;

    EXPECT(tc, config.mcts_threads >= 1);
    EXPECT(tc, config.ab_min_depth > 0);
    EXPECT(tc, config.ab_max_depth > config.ab_min_depth);
    EXPECT(tc, config.ab_policy_weight >= 0.0f);
    EXPECT(tc, config.ab_policy_weight <= 1.0f);
  }
  {
    TestCase tc("Decision modes");
    ParallelHybridConfig config;

    config.decision_mode = ParallelHybridConfig::DecisionMode::MCTS_PRIMARY;
    EXPECT(tc, config.decision_mode ==
                   ParallelHybridConfig::DecisionMode::MCTS_PRIMARY);

    config.decision_mode = ParallelHybridConfig::DecisionMode::AB_PRIMARY;
    EXPECT(tc, config.decision_mode ==
                   ParallelHybridConfig::DecisionMode::AB_PRIMARY);

    config.decision_mode = ParallelHybridConfig::DecisionMode::VOTE_WEIGHTED;
    EXPECT(tc, config.decision_mode ==
                   ParallelHybridConfig::DecisionMode::VOTE_WEIGHTED);

    config.decision_mode = ParallelHybridConfig::DecisionMode::DYNAMIC;
    EXPECT(tc,
           config.decision_mode == ParallelHybridConfig::DecisionMode::DYNAMIC);
  }
  {
    TestCase tc("GPU settings");
    ParallelHybridConfig config;

    EXPECT(tc, config.gpu_batch_size > 0);
    EXPECT(tc, config.gpu_batch_timeout_us > 0);
  }
}

// ============================================================================
// Stats Tests
// ============================================================================

void test_parallel_stats() {
  {
    TestCase tc("Stats reset");
    ParallelSearchStats stats;
    stats.mcts_nodes = 1000;
    stats.ab_nodes = 500;

    stats.reset();

    EXPECT(tc, stats.mcts_nodes == 0);
    EXPECT(tc, stats.ab_nodes == 0);
    EXPECT(tc, stats.policy_updates == 0);
    EXPECT(tc, stats.move_agreements == 0);
  }
}

} // namespace

bool test_hybrid_module() {
  std::cout << "\n=== Hybrid Search Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[Classifier]" << std::endl;
  test_classifier();

  std::cout << "\n[Strategy]" << std::endl;
  test_strategy();

  std::cout << "\n[AB Integration]" << std::endl;
  test_ab_integration();

  std::cout << "\n[Adapter]" << std::endl;
  test_adapter();

  std::cout << "\n[Shared State]" << std::endl;
  test_shared_state();

  std::cout << "\n[Hybrid Config]" << std::endl;
  test_hybrid_config();

  std::cout << "\n[Parallel Stats]" << std::endl;
  test_parallel_stats();

  std::cout << "\n--- Hybrid Results: " << g_tests_passed << " passed, "
            << g_tests_failed << " failed ---" << std::endl;

  return g_tests_failed == 0;
}

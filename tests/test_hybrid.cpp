/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Hybrid Search Tests - Position Classifier, AB Integration, Parallel Search
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "hybrid/classifier.h"
#include "hybrid/hybrid_search.h"
#include "hybrid/position_adapter.h"
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
}

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
    TestCase tc("ABSharedState PV publish");
    ABSharedState state;
    state.reset();

    std::vector<Move> pv = {Move(SQ_E2, SQ_E4), Move(SQ_E7, SQ_E5)};
    state.publish_pv(pv, 8);

    EXPECT(tc, state.pv_length.load() == 2);
    EXPECT(tc, state.pv_depth.load() == 8);
  }
  {
    TestCase tc("MCTSSharedState reset");
    MCTSSharedState state;
    state.num_top_moves.store(1);
    state.top_moves[0].move_raw.store(Move(SQ_E2, SQ_E4).raw());
    state.top_moves[0].policy.store(0.25f);
    state.top_moves[0].visits.store(42);
    state.top_moves[0].current_visits.store(12);
    state.top_moves[0].q.store(0.5f);
    state.best_current_visits.store(12);
    state.total_current_nodes.store(24);
    state.reset();

    EXPECT(tc, state.best_move_raw.load() == 0);
    EXPECT(tc, state.best_q.load() == 0.0f);
    EXPECT(tc, state.best_visits.load() == 0);
    EXPECT(tc, state.best_current_visits.load() == 0);
    EXPECT(tc, state.total_current_nodes.load() == 0);
    EXPECT(tc, state.num_top_moves.load() == 0);
    EXPECT(tc, state.top_moves[0].move_raw.load() == 0);
    EXPECT(tc, state.top_moves[0].policy.load() == 0.0f);
    EXPECT(tc, state.top_moves[0].visits.load() == 0);
    EXPECT(tc, state.top_moves[0].current_visits.load() == 0);
    EXPECT(tc, state.top_moves[0].q.load() == 0.0f);
    EXPECT(tc, !state.mcts_running.load());
  }
}

void test_hybrid_config() {
  {
    TestCase tc("Config defaults");
    ParallelHybridConfig config;

    EXPECT(tc, config.mcts_threads >= 1);
    EXPECT(tc, config.ab_min_depth > 0);
    EXPECT(tc, config.ab_max_depth > config.ab_min_depth);
    EXPECT(tc, config.ab_policy_weight >= 0.0f);
    EXPECT(tc, config.ab_policy_weight <= 1.0f);
    EXPECT(tc, config.ab_root_reject_mcts);
    EXPECT(tc, config.mcts_root_reject);
    EXPECT(tc, config.mcts_ab_root_hints);
    EXPECT(tc, config.mcts_ab_root_hint_delay_ms == 0);
    EXPECT(tc, config.mcts_ab_root_hint_count == 8);
    EXPECT(tc, config.ab_candidate_verify_ms == 120);
    EXPECT(tc, config.ab_candidate_verify_count == 4);
    EXPECT(tc, config.root_pawn_lever_tiebreak);
    EXPECT(tc, !config.ane_root_probe);
    EXPECT(tc, !config.ane_root_hints);
    EXPECT(tc, config.ane_only_pawn_endgames);
    EXPECT(tc, config.ane_compute_units == "cpu-ne");
    EXPECT(tc, config.ane_root_hint_count == 10);
    EXPECT(tc, config.ane_root_hint_wait_ms == 0);
    EXPECT(tc, config.ane_min_budget_ms == 0);
  }
  {
    TestCase tc("Transformer batching settings");
    ParallelHybridConfig config;

    EXPECT(tc, config.transformer_batch_size > 0);
    EXPECT(tc, config.transformer_batch_timeout_us > 0);
  }
  {
    TestCase tc("Coordinator budgets keep MCTS alive");
    ::MetalFish::Search::LimitsType limits;

    limits.movetime = 1000;
    EXPECT(tc, HybridShouldContinueMCTSAfterAB(limits));
    EXPECT(tc, !HybridCanStopEarlyOnAgreement(limits));
    EXPECT(tc, HybridHasMCTSDecisionBudget(limits, 0, false));

    limits.movetime = 0;
    limits.nodes = 1024;
    EXPECT(tc, HybridShouldContinueMCTSAfterAB(limits));
    EXPECT(tc, !HybridCanStopEarlyOnAgreement(limits));
    EXPECT(tc, HybridHasMCTSDecisionBudget(limits, 0, false));

    limits.nodes = 0;
    limits.infinite = 1;
    EXPECT(tc, HybridShouldContinueMCTSAfterAB(limits));
    EXPECT(tc, !HybridCanStopEarlyOnAgreement(limits));
    EXPECT(tc, HybridHasMCTSDecisionBudget(limits, 0, false));

    limits.infinite = 0;
    limits.depth = 8;
    EXPECT(tc, !HybridShouldContinueMCTSAfterAB(limits));
    EXPECT(tc, !HybridCanStopEarlyOnAgreement(limits));
    EXPECT(tc, !HybridHasMCTSDecisionBudget(limits, 1500, false));

    limits.depth = 0;
    limits.mate = 2;
    EXPECT(tc, !HybridShouldContinueMCTSAfterAB(limits));
    EXPECT(tc, !HybridCanStopEarlyOnAgreement(limits));
    EXPECT(tc, !HybridHasMCTSDecisionBudget(limits, 1500, false));

    limits.mate = 0;
    limits.searchmoves = {"e2e4"};
    limits.time[WHITE] = 30000;
    limits.inc[WHITE] = 1000;
    EXPECT(tc, HybridShouldContinueMCTSAfterAB(limits));
    EXPECT(tc, HybridCanStopEarlyOnAgreement(limits));
    EXPECT(tc, HybridHasMCTSDecisionBudget(limits, 1500, false));
    EXPECT(tc, !HybridHasMCTSDecisionBudget(limits, 0, false));
    auto mcts_limits = HybridBuildMCTSLimits(limits, 1500, false);
    EXPECT(tc, mcts_limits.movetime == 1500);
    EXPECT(tc, mcts_limits.time[WHITE] == 0);
    EXPECT(tc, mcts_limits.inc[WHITE] == 0);
    EXPECT(tc, mcts_limits.searchmoves.size() == 1);

    limits.movetime = 1000;
    EXPECT(tc, HybridShouldContinueMCTSAfterAB(limits));
    EXPECT(tc, !HybridCanStopEarlyOnAgreement(limits));
    mcts_limits = HybridBuildMCTSLimits(limits, 1500, false);
    EXPECT(tc, mcts_limits.movetime == 1000);

    limits.movetime = 0;
    limits.ponderMode = true;
    EXPECT(tc, !HybridHasMCTSDecisionBudget(limits, 1500, false));
    EXPECT(tc, HybridHasMCTSDecisionBudget(limits, 1500, true));
    mcts_limits = HybridBuildMCTSLimits(limits, 1500, true);
    EXPECT(tc, mcts_limits.infinite == 0);
    EXPECT(tc, mcts_limits.ponderMode);
    EXPECT(tc, mcts_limits.time[WHITE] == 30000);
    EXPECT(tc, mcts_limits.inc[WHITE] == 1000);
    EXPECT(tc, mcts_limits.searchmoves.size() == 1);
  }
  {
    TestCase tc("AB candidate verification budget");
    ::MetalFish::Search::LimitsType limits;

    limits.movetime = 5000;
    EXPECT(tc,
           HybridABCandidateVerifyBudgetMs(limits, 5000, 150, false) == 150);
    EXPECT(tc,
           HybridABCandidateVerifyBudgetMs(limits, 5000, 1000, false) == 833);
    EXPECT(tc, HybridABCandidateVerifyBudgetMs(limits, 5000, 150, true) == 0);

    limits = ::MetalFish::Search::LimitsType();
    limits.time[WHITE] = 30000;
    EXPECT(tc,
           HybridABCandidateVerifyBudgetMs(limits, 4000, 600, false) == 500);
    EXPECT(tc, HybridABCandidateVerifyBudgetMs(limits, 999, 100, false) == 0);

    limits.nodes = 1000;
    EXPECT(tc, HybridABCandidateVerifyBudgetMs(limits, 4000, 100, false) == 0);
  }
  {
    TestCase tc("Fixed-budget decisive MCTS override predicate");

    EXPECT(tc, HybridMCTSDecisiveFixedBudgetOverride(true, true, 456, 313,
                                                     0.686f, 117));
    EXPECT(tc, HybridMCTSDecisiveFixedBudgetOverride(true, true, 359, 237,
                                                     0.660f, 449));
    EXPECT(tc, !HybridMCTSDecisiveFixedBudgetOverride(false, true, 456, 313,
                                                      0.686f, 117));
    EXPECT(tc, !HybridMCTSDecisiveFixedBudgetOverride(true, false, 456, 313,
                                                      0.686f, 117));
    EXPECT(tc, !HybridMCTSDecisiveFixedBudgetOverride(true, true, 299, 250,
                                                      0.836f, 117));
    EXPECT(tc, !HybridMCTSDecisiveFixedBudgetOverride(true, true, 456, 209,
                                                      0.686f, 117));
    EXPECT(tc, !HybridMCTSDecisiveFixedBudgetOverride(true, true, 456, 313,
                                                      0.599f, 117));
    EXPECT(tc, !HybridMCTSDecisiveFixedBudgetOverride(true, true, 456, 313,
                                                      0.686f, 89));
  }
  {
    TestCase tc("Fixed-budget no-clear AB MCTS override predicate");

    EXPECT(tc,
           HybridMCTSNoClearFixedBudgetOverride(true, true, 293, 0.590f, 39));
    EXPECT(tc,
           !HybridMCTSNoClearFixedBudgetOverride(false, true, 293, 0.590f, 39));
    EXPECT(tc,
           !HybridMCTSNoClearFixedBudgetOverride(true, false, 293, 0.590f, 39));
    EXPECT(tc,
           !HybridMCTSNoClearFixedBudgetOverride(true, true, 224, 0.590f, 39));
    EXPECT(tc,
           !HybridMCTSNoClearFixedBudgetOverride(true, true, 293, 0.579f, 39));
    EXPECT(tc,
           !HybridMCTSNoClearFixedBudgetOverride(true, true, 293, 0.590f, 29));
  }
  {
    TestCase tc("Fixed-budget root-dominant MCTS override predicate");

    EXPECT(tc, HybridMCTSRootDominantFixedBudgetOverride(true, true, 275, 211,
                                                         0.767f, 179, 99));
    EXPECT(tc, !HybridMCTSRootDominantFixedBudgetOverride(false, true, 275, 211,
                                                          0.767f, 179, 99));
    EXPECT(tc, !HybridMCTSRootDominantFixedBudgetOverride(true, false, 275, 211,
                                                          0.767f, 179, 99));
    EXPECT(tc, !HybridMCTSRootDominantFixedBudgetOverride(true, true, 249, 211,
                                                          0.767f, 179, 99));
    EXPECT(tc, !HybridMCTSRootDominantFixedBudgetOverride(true, true, 275, 199,
                                                          0.767f, 179, 99));
    EXPECT(tc, !HybridMCTSRootDominantFixedBudgetOverride(true, true, 275, 211,
                                                          0.739f, 179, 99));
    EXPECT(tc, !HybridMCTSRootDominantFixedBudgetOverride(true, true, 275, 211,
                                                          0.767f, 149, 99));
    EXPECT(tc, !HybridMCTSRootDominantFixedBudgetOverride(true, true, 275, 211,
                                                          0.767f, 179, 79));
  }
  {
    TestCase tc("Fixed-budget tactical-gap MCTS override predicate");

    EXPECT(tc, HybridMCTSTacticalGapFixedBudgetOverride(true, 344, 64, 0.186f,
                                                        0.240f, 366, 307));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(false, 344, 64, 0.186f,
                                                         0.240f, 366, 307));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(true, 249, 64, 0.186f,
                                                         0.240f, 366, 307));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(true, 344, 54, 0.186f,
                                                         0.240f, 366, 307));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(true, 344, 64, 0.159f,
                                                         0.240f, 366, 307));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(true, 344, 64, 0.186f,
                                                         0.119f, 366, 307));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(true, 344, 64, 0.186f,
                                                         0.240f, 299, 307));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(true, 344, 64, 0.186f,
                                                         0.240f, 366, 249));
    EXPECT(tc, !HybridMCTSTacticalGapFixedBudgetOverride(true, 295, 109, 0.369f,
                                                         0.011f, 499, 311));
  }
  {
    TestCase tc("Fixed-budget root-confidence MCTS override predicate");

    EXPECT(tc, HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 277, 186, 0.671f, 0.195f, 233, 123));
    EXPECT(tc, HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 242, 184, 0.760f, 0.080f, 182, 76));
    EXPECT(tc, HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 380, 260, 0.684f, 0.137f, 222, 65));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   false, true, 277, 186, 0.671f, 0.195f, 233, 123));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, false, 277, 186, 0.671f, 0.195f, 233, 123));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 229, 186, 0.671f, 0.195f, 233, 123));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 277, 179, 0.671f, 0.195f, 233, 123));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 277, 186, 0.649f, 0.195f, 233, 123));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 277, 186, 0.671f, 0.119f, 233, 123));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 277, 186, 0.671f, 0.195f, 169, 123));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 277, 186, 0.671f, 0.195f, 233, 109));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 380, 239, 0.684f, 0.137f, 222, 65));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 380, 260, 0.639f, 0.137f, 222, 65));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 380, 260, 0.684f, 0.119f, 222, 65));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 380, 260, 0.684f, 0.137f, 199, 65));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 380, 260, 0.684f, 0.137f, 222, 59));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 262, 137, 0.523f, 0.066f, 20, 20));
  }
  {
    TestCase tc("Low-node root-confidence MCTS override stays narrow");

    EXPECT(tc, HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 73, 53, 0.726f, 0.354f, 239, 236, 15,
                   -1, 4, -32001, false, false, 23, 3, 9, 0.242f, 0.663f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   false, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, false, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, true, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 49, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 39, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.699f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.299f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 179, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 149, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, 30,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 1, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 13, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, 0.190f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 49999, 2, 5, -0.007f, 0.579f));
  }
  {
    TestCase tc("Short-root tactical MCTS override");

    EXPECT(tc, HybridMCTSShortRootTacticalOverride(
                   true, true, true, 172, 104, 0.605f, 0.173f, 234, 67, 598,
                   563, 2, -32001, false, 520856, 5, 10, 0.381f, 0.653f));
    EXPECT(tc, HybridMCTSShortRootTacticalOverride(
                   true, true, false, 172, 104, 0.605f, 0.173f, 234, 80, 572,
                   565, 2, -32001, false, 965531, 5, 10, 0.381f, 0.653f));
    EXPECT(tc, HybridMCTSShortRootTacticalOverride(
                   true, true, false, 172, 104, 0.605f, 0.173f, 234, 68, 601,
                   579, 2, -32001, false, 1051908, 5, 10, 0.381f, 0.653f));
    EXPECT(tc, HybridMCTSShortRootTacticalOverride(
                   true, true, false, 161, 98, 0.609f, 0.177f, 236, 80, 579,
                   555, 2, -32001, false, 1144870, 4, 10, 0.381f, 0.657f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 172, 94, 0.605f, 0.173f, 234, 67, 598, 563,
                   2, -32001, false, 520856, 5, 10, 0.381f, 0.653f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 172, 104, 0.605f, 0.159f, 234, 67, 598,
                   563, 2, -32001, false, 520856, 5, 10, 0.381f, 0.653f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 172, 104, 0.605f, 0.173f, 234, 67, 634,
                   563, 2, -32001, false, 520856, 5, 10, 0.381f, 0.653f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, false, 172, 104, 0.605f, 0.173f, 234, 80, 591,
                   565, 2, -32001, false, 965531, 5, 10, 0.381f, 0.653f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 172, 104, 0.605f, 0.173f, 234, 67, 598,
                   563, 2, -32001, false, 520856, 3, 10, 0.381f, 0.653f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 172, 104, 0.605f, 0.173f, 234, 67, 598,
                   563, 2, -32001, false, 520856, 5, 10, 0.420f, 0.653f));
  }
  {
    TestCase tc("MCTS AB-lowerbound confirmed override predicate");

    EXPECT(tc, HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.374f, 296, 2, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 117, 47, 0.770f, 0.336f, 27, 2, 283, true,
                   391824, 3, 6));
    EXPECT(tc, HybridMCTSABLowerBoundConfirmedOverride(
                   true, false, 2080, 2009, 0.966f, 0.551f, -11, 2, 577, true,
                   103741, 3, 21));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   false, true, 63, 56, 0.889f, 0.374f, 296, 2, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, false, 63, 56, 0.889f, 0.374f, 296, 2, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, false, 2080, 2009, 0.949f, 0.551f, -11, 2, 577, true,
                   103741, 3, 21));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, false, 2080, 2009, 0.966f, 0.499f, -11, 2, 577, true,
                   103741, 3, 21));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, false, 2080, 2009, 0.966f, 0.551f, 101, 2, 577, true,
                   103741, 3, 21));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 49, 56, 0.889f, 0.374f, 296, 2, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 44, 0.889f, 0.374f, 296, 2, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.749f, 0.374f, 296, 2, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.299f, 296, 2, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.374f, 296, 1, 1117, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.374f, 296, 2, 1117, false,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.374f, 296, 2, 545, true,
                   242416, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.374f, 296, 2, 1117, true,
                   99999, 2, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.374f, 296, 2, 1117, true,
                   242416, 1, 4));
    EXPECT(tc, !HybridMCTSABLowerBoundConfirmedOverride(
                   true, true, 63, 56, 0.889f, 0.374f, 296, 2, 1117, true,
                   242416, 2, 9));
  }
  {
    TestCase tc("Fixed-budget root policy tie-break predicate");

    EXPECT(tc, HybridRootPolicyTieBreak(true, 698, 339, 0.747f, 0.170f, 300,
                                        0.705f, 0.410f));
    EXPECT(tc, !HybridRootPolicyTieBreak(false, 698, 339, 0.747f, 0.170f, 300,
                                         0.705f, 0.410f));
    EXPECT(tc, !HybridRootPolicyTieBreak(true, 599, 339, 0.747f, 0.170f, 300,
                                         0.705f, 0.410f));
    EXPECT(tc, !HybridRootPolicyTieBreak(true, 698, 255, 0.747f, 0.170f, 300,
                                         0.705f, 0.410f));
    EXPECT(tc, !HybridRootPolicyTieBreak(true, 698, 339, 0.747f, 0.170f, 255,
                                         0.705f, 0.410f));
    EXPECT(tc, !HybridRootPolicyTieBreak(true, 698, 339, 0.747f, 0.170f, 277,
                                         0.705f, 0.410f));
    EXPECT(tc, !HybridRootPolicyTieBreak(true, 698, 339, 0.747f, 0.170f, 300,
                                         0.690f, 0.410f));
    EXPECT(tc, !HybridRootPolicyTieBreak(true, 698, 339, 0.747f, 0.170f, 300,
                                         0.705f, 0.299f));
    EXPECT(tc, !HybridRootPolicyTieBreak(true, 698, 339, 0.747f, 0.210f, 300,
                                         0.705f, 0.410f));
  }
  {
    TestCase tc("Fixed-budget root Q-conflict tie-break predicate");

    EXPECT(tc, HybridRootQConflictTieBreak(true, true, 57, 24, -0.349f, 0.349f,
                                           23, 0.206f, 0.108f, -245, -278));
    EXPECT(tc,
           !HybridRootQConflictTieBreak(false, true, 57, 24, -0.349f, 0.349f,
                                        23, 0.206f, 0.108f, -245, -278));
    EXPECT(tc,
           !HybridRootQConflictTieBreak(true, false, 57, 24, -0.349f, 0.349f,
                                        23, 0.206f, 0.108f, -245, -278));
    EXPECT(tc, !HybridRootQConflictTieBreak(true, true, 57, 24, -0.349f, 0.349f,
                                            17, 0.206f, 0.108f, -245, -278));
    EXPECT(tc, !HybridRootQConflictTieBreak(true, true, 57, 24, 0.051f, 0.349f,
                                            23, 0.560f, 0.108f, -245, -278));
    EXPECT(tc, !HybridRootQConflictTieBreak(true, true, 57, 24, -0.349f, 0.349f,
                                            23, 0.090f, 0.108f, -245, -278));
    EXPECT(tc, !HybridRootQConflictTieBreak(true, true, 57, 24, -0.349f, 0.349f,
                                            23, 0.206f, 0.250f, -245, -278));
    EXPECT(tc, !HybridRootQConflictTieBreak(true, true, 57, 24, -0.349f, 0.349f,
                                            23, 0.206f, 0.108f, -245, -320));
  }
  {
    TestCase tc("MCTS root rejection of AB predicate");

    EXPECT(tc, HybridMCTSRootRejectsAB(true, true, true, false, 198, 21, 0.137f,
                                       0.007f, 0.808f, 31));
    EXPECT(tc, HybridMCTSRootRejectsAB(true, true, true, false, 210, 52, 0.142f,
                                       0.020f, 0.680f, 32));
    EXPECT(tc, HybridMCTSRootRejectsAB(true, true, false, true, 260, 40, 0.750f,
                                       0.490f, 0.500f, 0));
    EXPECT(tc, !HybridMCTSRootRejectsAB(false, true, true, false, 198, 21,
                                        0.137f, 0.007f, 0.808f, 31));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, false, true, false, 198, 21,
                                        0.137f, 0.007f, 0.808f, 31));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, true, false, false, 198, 21,
                                        0.137f, 0.007f, 0.808f, 31));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, true, true, true, 198, 21, 0.137f,
                                        0.007f, 0.808f, 31));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, true, true, false, 179, 21,
                                        0.137f, 0.007f, 0.808f, 31));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, true, true, false, 210, 53,
                                        0.142f, 0.020f, 0.680f, 32));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, true, true, false, 198, 21,
                                        0.126f, 0.007f, 0.808f, 31));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, true, true, false, 198, 21,
                                        0.137f, 0.007f, 0.649f, 31));
    EXPECT(tc, !HybridMCTSRootRejectsAB(true, true, true, false, 198, 21,
                                        0.137f, 0.007f, 0.808f, 24));
  }
  {
    TestCase tc("MCTS root-reject Q-gap override predicate");

    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   false, true, true, 62, 28, 0.452f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, false, true, 62, 28, 0.452f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, false, 62, 28, 0.452f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 24, 0.387f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.590f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.640f, 79, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.640f, 89, 149, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 66, 32, 0.485f, 0.667f, 98, 163, 3,
                   -VALUE_INFINITE, 1055, 2, 24, -0.349f, 0.318f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.640f, 89, 160, 4,
                   -VALUE_INFINITE, 1366, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 2501, 2, 24, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 20, -0.349f, 0.291f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 62, 28, 0.452f, 0.640f, 89, 160, 2,
                   -VALUE_INFINITE, 1366, 2, 24, -0.258f, 0.291f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 190, 155, 0.816f, 0.991f, 228, 284, 2,
                   -VALUE_INFINITE, 413, 2, 24, -0.349f, 0.642f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 190, 155, 0.590f, 0.991f, 228, 284, 2,
                   -VALUE_INFINITE, 413, 2, 24, -0.349f, 0.642f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, false, true, 2192, 2113, 0.964f, 1.149f, 336, 396, 3,
                   -VALUE_INFINITE, 1862, 2, 27, -0.341f, 0.808f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, false, true, 2192, 2113, 0.964f, 1.149f, 336, 396, 3,
                   -VALUE_INFINITE, 2601, 2, 27, -0.341f, 0.808f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.963f, 1.463f, 603, 685, 2,
                   -VALUE_INFINITE, 55, 2, 1, -0.498f, 0.965f));
  }
  {
    TestCase tc("Leading MCTS root hint survives inconclusive AB verify");

    EXPECT(tc,
           HybridPreserveLeadingHintAfterABVerify(
               true, true, true, 50, 48, 0.960f, 0.755f, 0.325f, 0.423f, 0.409f,
               173, 162, 172, false, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, HybridPreserveLeadingHintAfterABVerify(
                   true, true, true, 18, 17, 0.944f, 0.840f, 0.268f, 0.423f,
                   0.409f, 1, 84, 70, false, 108, true, false, 381844));
    EXPECT(tc,
           !HybridPreserveLeadingHintAfterABVerify(
               true, true, false, 50, 48, 0.960f, 0.755f, 0.325f, 0.423f,
               0.409f, 173, 162, 172, false, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc,
           !HybridPreserveLeadingHintAfterABVerify(
               true, true, true, 50, 48, 0.960f, 0.755f, 0.380f, 0.423f, 0.409f,
               173, 162, 172, false, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc,
           !HybridPreserveLeadingHintAfterABVerify(
               true, true, true, 50, 48, 0.960f, 0.755f, 0.325f, 0.340f, 0.409f,
               173, 162, 172, false, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc,
           !HybridPreserveLeadingHintAfterABVerify(
               true, true, true, 50, 48, 0.960f, 0.755f, 0.325f, 0.423f, 0.409f,
               250, 162, 172, false, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc,
           !HybridPreserveLeadingHintAfterABVerify(
               true, true, true, 50, 48, 0.960f, 0.755f, 0.325f, 0.423f, 0.409f,
               173, 162, 600, true, -VALUE_INFINITE, false, false, 4));
  }
  {
    TestCase tc("Discovered pawn-push MCTS override predicate");

    EXPECT(tc, HybridMCTSDiscoveredPawnPushOverride(
                   true, true, true, 50, 48, 2, 0.960f, 0.430f, 295, 249, 46,
                   173, 162, 2, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, HybridMCTSDiscoveredPawnPushOverride(
                   true, true, true, 18, 17, 1, 0.944f, 0.572f, 366, 347, 19, 1,
                   84, 2, 108, true, false, 381844));
    EXPECT(tc, !HybridMCTSDiscoveredPawnPushOverride(
                   false, true, true, 50, 48, 2, 0.960f, 0.430f, 295, 249, 46,
                   173, 162, 2, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, !HybridMCTSDiscoveredPawnPushOverride(
                   true, true, false, 50, 48, 2, 0.960f, 0.430f, 295, 249, 46,
                   173, 162, 2, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, !HybridMCTSDiscoveredPawnPushOverride(
                   true, true, true, 50, 48, 5, 0.960f, 0.430f, 295, 249, 46,
                   173, 162, 2, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, !HybridMCTSDiscoveredPawnPushOverride(
                   true, true, true, 50, 48, 2, 0.960f, 0.390f, 295, 249, 46,
                   173, 162, 2, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, !HybridMCTSDiscoveredPawnPushOverride(
                   true, true, true, 50, 48, 2, 0.960f, 0.430f, 279, 249, 46,
                   173, 162, 2, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, !HybridMCTSDiscoveredPawnPushOverride(
                   true, true, true, 50, 48, 2, 0.960f, 0.430f, 295, 249, 46,
                   230, 162, 2, -VALUE_INFINITE, false, false, 4));
    EXPECT(tc, !HybridMCTSDiscoveredPawnPushOverride(
                   true, true, true, 50, 48, 2, 0.960f, 0.430f, 295, 249, 46,
                   173, 162, 2, 50, false, false, 4));
  }
  {
    TestCase tc("ANE-confirmed MCTS override predicate");

    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 102, 93,
                                          0.912f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 106, 100,
                                          0.943f, 1.058f, 336, 306, 0.100f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                          0.981f, 0.918f, 424, 437, 0.401f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 55, 47,
                                          0.855f, 0.584f, 139, 139, 0.248f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 52, 45,
                                          0.865f, 0.807f, 826, 746, 0.497f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 56, 50,
                                          0.893f, 0.678f, 471, 252, 0.127f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(false, true, true, true, 102, 93,
                                           0.912f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, false, true, true, 102, 93,
                                           0.912f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, false, true, 102, 93,
                                           0.912f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, false, 102, 93,
                                           0.912f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 79, 93,
                                           0.912f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 102, 63,
                                           0.912f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 102, 93,
                                           0.699f, 0.817f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 102, 93,
                                           0.912f, 0.199f, 242, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 102, 93,
                                           0.912f, 0.817f, 119, 242, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 102, 93,
                                           0.912f, 0.817f, 242, 59, 0.100f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 102, 93,
                                           0.912f, 0.817f, 242, 242, 0.019f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 203, 170,
                                           0.837f, 0.262f, 123, 112, 0.047f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                           0.899f, 0.918f, 424, 437, 0.401f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 55, 44,
                                           0.855f, 0.584f, 139, 139, 0.248f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 49, 45,
                                           0.865f, 0.807f, 826, 746, 0.497f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 55, 47,
                                           0.849f, 0.584f, 139, 139, 0.248f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 55, 47,
                                           0.855f, 0.549f, 139, 139, 0.248f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 55, 47,
                                           0.855f, 0.584f, 129, 139, 0.248f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 55, 47,
                                           0.855f, 0.584f, 139, 119, 0.248f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 55, 47,
                                           0.855f, 0.584f, 139, 139, 0.199f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 56, 50,
                                           0.879f, 0.678f, 471, 252, 0.127f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 56, 50,
                                           0.893f, 0.599f, 471, 252, 0.127f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 56, 50,
                                           0.893f, 0.678f, 471, 252, 0.119f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                           0.981f, 0.499f, 424, 437, 0.401f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                           0.981f, 0.918f, 199, 437, 0.401f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                           0.981f, 0.918f, 424, 149, 0.401f));
  }
  {
    TestCase tc("Pawn-only ANE/MCTS override predicate");

    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 32, 25, 32, 25,
                   3, 0.781f, 0.443f, 0.456f, 91, 91, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 30, 23, 30, 23,
                   3, 0.767f, 0.490f, 0.503f, 106, 106, -1, -1, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 64, 42, 64, 42,
                   1, 0.657f, 0.720f, 0.720f, 233, 233, 0, 0, 0.401f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 64, 42, 64, 42,
                   1, 0.780f, 0.720f, 0.720f, 233, 233, 0, 0, 0.401f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 263, 256, 23, 23,
                   0, 1.000f, 0.000f, 0.875f, 361, 361, 0, -1, 0.401f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 82, 79, 46, 46, 0,
                   1.000f, 0.000f, 0.765f, 257, 273, -75, -140, 0.401f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                             false, false, 2090, 2054, 2090,
                                             2054, 13, 0.983f, 1.192f, 1.191f,
                                             498, 498, -1, -1, 1.022f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                             false, true, 10931, 8542, 2136,
                                             2066, 11, 0.967f, 0.289f, 0.509f,
                                             619, 565, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, false, 10931, 8542, 2136,
                                              2066, 11, 0.967f, 0.289f, 0.509f,
                                              619, 565, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, true, 10931, 8542, 2136,
                                              2066, 11, 0.967f, 0.249f, 0.509f,
                                              619, 565, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, true, 10931, 8542, 2136,
                                              2066, 11, 0.967f, 0.289f, 0.249f,
                                              619, 565, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, true, 10931, 8542, 2136,
                                              2066, 11, 0.967f, 0.289f, 0.509f,
                                              499, 565, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, true, 10931, 8542, 2136,
                                              2066, 11, 0.967f, 0.289f, 0.509f,
                                              619, 449, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, true, 10931, 8542, 2136,
                                              2066, 11, 0.967f, 0.289f, 0.509f,
                                              619, 565, 1, 163, 0.299f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, true, 10931, 8542, 2136,
                                              2066, 81, 0.967f, 0.289f, 0.509f,
                                              619, 565, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, true, 10931, 8542, 1499,
                                              1499, 11, 0.967f, 0.289f, 0.509f,
                                              619, 565, 1, 163, 0.348f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 242, 235, 17, 15,
                   1, 0.882f, 0.871f, 0.871f, 355, 355, 0, 0, 0.401f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 432, 417, 16, 16,
                   0, 1.000f, 0.000f, 0.900f, 396, 415, -75, -80, 0.401f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 242, 235, 14, 14,
                   1, 0.882f, 0.871f, 0.871f, 355, 355, 0, 0, 0.401f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 263, 256, 19, 19,
                   0, 1.000f, 0.000f, 0.875f, 361, 361, 0, -1, 0.401f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 29, 25, 29, 25,
                   3, 0.862f, 0.443f, 0.456f, 91, 91, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, false, 2090, 2054, 1499,
                                              1499, 13, 0.983f, 1.192f, 1.191f,
                                              498, 498, -1, -1, 1.022f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, false, 2090, 2054, 2090,
                                              2054, 13, 0.949f, 1.192f, 1.191f,
                                              498, 498, -1, -1, 1.022f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, false, 2090, 2054, 2090,
                                              2054, 13, 0.983f, 0.899f, 1.191f,
                                              498, 498, -1, -1, 1.022f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, false, 2090, 2054, 2090,
                                              2054, 13, 0.983f, 1.192f, 0.899f,
                                              498, 498, -1, -1, 1.022f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(true, true, true, false, true,
                                              false, false, 2090, 2054, 2090,
                                              2054, 13, 0.983f, 1.192f, 1.191f,
                                              498, 498, -1, -1, 0.749f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, false, true, true, false, 2090, 2054, 2090,
                   2054, 13, 0.983f, 1.192f, 1.191f, 498, 498, -1, -1, 1.022f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   false, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, false, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, false, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, false, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, false, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   10, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.740f, 0.484f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.240f, 0.497f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.000f, 0.499f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.240f, 104, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 79, 104, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 79, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 120, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 47, 39, 47, 39,
                   3, 0.830f, 0.484f, 0.497f, 104, 104, 0, 0, 0.149f));
  }
  {
    TestCase tc("Pawn lever root tie-break classifier");

    EXPECT(tc,
           HybridRootPawnLeverAgreementTieBreak(true, true, 0.549f, 0.079f));
    EXPECT(tc,
           !HybridRootPawnLeverAgreementTieBreak(false, true, 0.549f, 0.079f));
    EXPECT(tc,
           !HybridRootPawnLeverAgreementTieBreak(true, false, 0.549f, 0.079f));
    EXPECT(tc,
           !HybridRootPawnLeverAgreementTieBreak(true, true, 0.550f, 0.079f));
    EXPECT(tc,
           !HybridRootPawnLeverAgreementTieBreak(true, true, 0.549f, 0.080f));

    EXPECT(tc, HybridRootPawnLeverCandidate(-507, -555, 7869, 3, 37, 1, -0.200f,
                                            0.200f, -0.200f, -0.230f, 0.250f));
    EXPECT(tc,
           HybridRootPawnLeverCandidate(-521, -548, 705292, 3, 38, 1, -0.200f,
                                        0.200f, -0.200f, -0.240f, 0.250f));
    EXPECT(tc,
           HybridRootPawnLeverCandidate(-522, -510, 369245, 2, 38, 1, -0.433f,
                                        0.208f, -0.433f, -0.630f, 0.260f));
    EXPECT(tc, HybridRootPawnLeverCandidate(-477, -537, 111, 3, 38, 6, -0.504f,
                                            0.044f, -0.504f, -0.630f, 0.260f));
    EXPECT(tc, HybridRootPawnLeverCandidate(-474, -546, 914, 3, 38, 6, -0.504f,
                                            0.044f, -0.434f, -0.630f, 0.260f));
    EXPECT(tc, HybridRootPawnLeverCandidate(929, 899, 3233, 7, 13, 1, 0.892f,
                                            0.188f, 0.892f, 0.878f, 0.050f));
    EXPECT(tc, HybridRootPawnLeverCandidate(899, 884, 1399, 6, 7, 1, 0.928f,
                                            0.206f, 0.928f, 0.888f, 0.050f));
    EXPECT(tc, HybridRootPawnLeverCandidate(932, 895, 303, 6, 6, 1, 0.927f,
                                            0.206f, 0.927f, 0.885f, 0.050f));
    EXPECT(tc, HybridRootPawnLeverCandidate(907, 902, 1742, 6, 3, 2, 0.918f,
                                            0.188f, 0.930f, 0.888f, 0.050f));
    EXPECT(tc, HybridRootPawnLeverCandidate(-245, -254, 5066, 5, 4, 1, -0.193f,
                                            0.314f, -0.193f, -0.257f, 0.038f));
    EXPECT(tc, HybridRootPawnLeverCandidate(930, 900, 192, 7, 13, 3, 0.890f,
                                            0.188f, 0.931f, 0.878f, 0.050f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(166, 156, 59965, 3, 28, 2, 0.195f,
                                             0.279f, 0.195f, 0.139f, 0.088f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(111, 112, 10443, 6, 15, 1, 0.216f,
                                             0.048f, 0.216f, 0.097f, 0.078f));
    EXPECT(tc,
           !HybridRootPawnLeverCandidate(-507, -568, 7869, 3, 37, 1, -0.200f,
                                         0.200f, -0.200f, -0.230f, 0.240f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-474, -555, 914, 3, 38, 6, -0.504f,
                                             0.044f, -0.434f, -0.630f, 0.260f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-507, -555, 149, 3, 37, 1, -0.200f,
                                             0.220f, -0.200f, -0.230f, 0.240f));
    EXPECT(tc, HybridRootPawnLeverCandidate(-507, -555, 0, 3, 37, 1, -0.200f,
                                            0.200f, -0.200f, -0.230f, 0.250f));
    EXPECT(tc,
           !HybridRootPawnLeverCandidate(-507, -555, 7869, 9, 37, 1, -0.200f,
                                         0.200f, -0.200f, -0.230f, 0.250f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-507, -555, 7869, 3, 7, 1, -0.200f,
                                             0.200f, -0.200f, -0.230f, 0.250f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-47, -77, 373, 5, 13, 1, -0.038f,
                                             0.220f, -0.038f, -0.112f, 0.050f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(932, 891, 303, 6, 6, 1, 0.927f,
                                             0.206f, 0.927f, 0.885f, 0.050f));
    EXPECT(tc, HybridRootPawnLeverCandidate(907, 902, 1742, 6, 3, 3, 0.918f,
                                            0.188f, 0.930f, 0.888f, 0.050f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(907, 902, 1742, 6, 3, 4, 0.918f,
                                             0.188f, 0.930f, 0.888f, 0.050f));
    EXPECT(tc, HybridRootPawnLeverCandidate(-245, -254, 5066, 5, 3, 1, -0.193f,
                                            0.314f, -0.193f, -0.257f, 0.038f));
    EXPECT(tc, HybridRootPawnLeverCandidate(-257, -289, 934, 5, 2, 1, -0.189f,
                                            0.314f, -0.189f, -0.259f, 0.038f));
    EXPECT(tc, HybridRootPawnLeverCandidate(-261, -299, 2316, 5, 3, 1, -0.189f,
                                            0.314f, -0.189f, -0.270f, 0.038f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-245, -254, 5066, 5, 2, 1, -0.193f,
                                             0.314f, -0.193f, -0.257f, 0.038f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-257, -298, 934, 5, 2, 1, -0.189f,
                                             0.314f, -0.189f, -0.259f, 0.038f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-257, -289, 899, 5, 2, 1, -0.189f,
                                             0.314f, -0.189f, -0.259f, 0.038f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-257, -289, 934, 5, 1, 1, -0.189f,
                                             0.314f, -0.189f, -0.259f, 0.038f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-257, -289, 934, 5, 2, 2, -0.189f,
                                             0.314f, -0.189f, -0.259f, 0.038f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-257, -289, 934, 5, 2, 1, -0.189f,
                                             0.314f, -0.189f, -0.281f, 0.038f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-245, -254, 5066, 5, 4, 1, -0.193f,
                                             0.314f, -0.193f, -0.265f, 0.038f));
    EXPECT(tc,
           !HybridRootPawnLeverCandidate(-305, -315, 8978, 2, 66, 3, -0.422f,
                                         0.142f, -0.422f, -0.372f, 0.086f));
    EXPECT(tc,
           !HybridRootPawnLeverCandidate(-521, -546, 13286, 4, 19, 3, -0.490f,
                                         0.079f, -0.490f, -0.698f, 0.218f));
    EXPECT(tc,
           !HybridRootPawnLeverCandidate(-505, -505, 19076, 3, 36, 1, -0.495f,
                                         0.181f, -0.495f, -0.636f, 0.240f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-1, -3, 23050, 2, 75, 1, -0.045f,
                                             0.053f, -0.045f, -0.134f, 0.240f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-1, -1, 57350, 3, 74, 1, -0.047f,
                                             0.070f, -0.047f, -0.123f, 0.219f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(-1, -1, 57350, 3, 74, 1, -0.047f,
                                             0.070f, -0.084f, -0.123f, 0.219f));

    EXPECT(tc, HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                               -258, -282, 214, 1, -0.181f, 5,
                                               2, -0.259f, 0.038f));
    EXPECT(tc, HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                               -246, -267, 83, 1, -0.189f, 5, 3,
                                               -0.270f, 0.038f));
    EXPECT(tc, HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                               -256, -300, 4207, 1, -0.189f, 5,
                                               3, -0.270f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(false, 4, -0.297f, 2, -0.271f,
                                                -258, -282, 214, 1, -0.181f, 5,
                                                2, -0.259f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 3, -0.271f,
                                                -258, -282, 214, 1, -0.181f, 5,
                                                2, -0.259f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 2, -0.297f, 4, -0.271f,
                                                -258, -282, 214, 1, -0.181f, 5,
                                                2, -0.259f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.283f,
                                                -258, -282, 214, 1, -0.181f, 5,
                                                2, -0.259f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                                -258, -282, 214, 1, -0.181f, 5,
                                                1, -0.259f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                                -258, -282, 214, 1, -0.181f, 5,
                                                2, -0.259f, 0.034f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                                -258, -309, 214, 1, -0.181f, 5,
                                                2, -0.259f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                                -258, -282, 74, 1, -0.181f, 5,
                                                2, -0.259f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                                -246, -267, 74, 1, -0.189f, 5,
                                                3, -0.270f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                                -256, -307, 4207, 1, -0.189f, 5,
                                                3, -0.270f, 0.038f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(true, 4, -0.297f, 2, -0.271f,
                                                -258, -282, 214, 1, -0.181f, 5,
                                                2, -0.272f, 0.038f));

    Position pos;
    StateInfo st;
    pos.set("2q1rr1k/3bbnnp/p2p1pp1/2pPp3/PpP1P1P1/1P2BNNP/2BQ1PRK/7R b - -",
            false, &st);
    EXPECT(tc, HybridIsPawnLever(pos, UCIEngine::to_move(pos, "f6f5")));
    EXPECT(tc, HybridIsKingsidePawnLever(pos, UCIEngine::to_move(pos, "f6f5")));
    EXPECT(tc, HybridIsKingsidePawnPush(pos, UCIEngine::to_move(pos, "f6f5")));
    EXPECT(tc, !HybridIsPawnLever(pos, UCIEngine::to_move(pos, "e7d8")));
    EXPECT(tc, !HybridIsPawnLever(pos, UCIEngine::to_move(pos, "a6a5")));
    EXPECT(tc,
           !HybridIsKingsidePawnLever(pos, UCIEngine::to_move(pos, "a6a5")));
    EXPECT(tc, !HybridIsKingsidePawnPush(pos, UCIEngine::to_move(pos, "a6a5")));
    EXPECT(tc, HybridRootPawnLeverCanChallengeSelected(
                   pos, UCIEngine::to_move(pos, "e7d8"), true));
    EXPECT(tc, !HybridRootPawnLeverCanChallengeSelected(
                   pos, UCIEngine::to_move(pos, "f6f5"), true));
    EXPECT(tc, HybridHighPolicyRootLeverHint(
                   pos, UCIEngine::to_move(pos, "f6f5"), 0.260f, 0.208f));
    EXPECT(tc, !HybridHighPolicyRootLeverHint(
                   pos, UCIEngine::to_move(pos, "f6f5"), 0.240f, 0.208f));

    pos.set(
        "r2q1rk1/1ppnbppp/p2p1nb1/3Pp3/2P1P1P1/2N2N1P/PPB1QP2/R1B2RK1 b - -",
        false, &st);
    EXPECT(tc, HybridIsPawnLever(pos, UCIEngine::to_move(pos, "h7h5")));
    EXPECT(tc, HybridIsKingsidePawnLever(pos, UCIEngine::to_move(pos, "h7h5")));
    EXPECT(tc, HybridIsKingsidePawnPush(pos, UCIEngine::to_move(pos, "h7h5")));
    EXPECT(tc, !HybridIsPawnLever(pos, UCIEngine::to_move(pos, "c7c6")));
    EXPECT(tc,
           !HybridIsKingsidePawnLever(pos, UCIEngine::to_move(pos, "c7c6")));
    EXPECT(tc, !HybridIsKingsidePawnPush(pos, UCIEngine::to_move(pos, "c7c6")));

    pos.set("r2rn1k1/1p1nbp2/2p1p1p1/p2p3p/2P5/1P1PPNPP/P1R2PB1/1RB3K1 w - -",
            false, &st);
    EXPECT(tc, HybridIsKingsidePawnPush(pos, UCIEngine::to_move(pos, "h3h4")));
    EXPECT(tc, HybridIsKingsidePawnLever(pos, UCIEngine::to_move(pos, "g3g4")));
    EXPECT(tc, !HybridIsPawnDiscoveredKingAttack(
                   pos, UCIEngine::to_move(pos, "h3h4")));

    pos.set("2Q5/6p1/3n3k/4K3/1P3p1P/6qP/2Q5/8 b - - 4 62", false, &st);
    EXPECT(tc, HybridIsKingsidePawnPush(pos, UCIEngine::to_move(pos, "f4f3")));
    EXPECT(tc, HybridIsPawnDiscoveredKingAttack(
                   pos, UCIEngine::to_move(pos, "f4f3")));
    EXPECT(tc, !HybridIsPawnDiscoveredKingAttack(
                   pos, UCIEngine::to_move(pos, "d6c8")));

    pos.set("r1b2rk1/pppn1pp1/3ppq1p/b7/3PP3/P1N2N2/1PPQ1PPP/1K1R1B1R b - -",
            false, &st);
    EXPECT(tc, HybridIsPawnLever(pos, UCIEngine::to_move(pos, "e6e5")));
    EXPECT(tc,
           !HybridIsKingsidePawnLever(pos, UCIEngine::to_move(pos, "e6e5")));

    pos.set("r1bqk2r/4npbp/2pp2p1/p1p1p3/4P3/PPNP1N2/2P2PPP/"
            "1RBQ1RK1 b kq -",
            false, &st);
    const Move castle = UCIEngine::to_move(pos, "e8g8");
    EXPECT(tc, castle.type_of() == CASTLING);
    EXPECT(tc, HybridIsKingsidePawnLever(pos, UCIEngine::to_move(pos, "f7f5")));
    EXPECT(tc, !HybridRootPawnLeverCanChallengeSelected(pos, castle, true));
    EXPECT(tc, !HybridHighPolicyRootLeverHint(
                   pos, UCIEngine::to_move(pos, "f7f5"), 0.147f, 0.326f));

    pos.set("4r1k1/p3brp1/1p1p4/7p/1P1B4/P1R1P1P1/5P2/2R3K1 b - -", false, &st);
    EXPECT(tc, !HybridHighPolicyRootLeverHint(
                   pos, UCIEngine::to_move(pos, "h5h4"), 0.218f, 0.181f));
    EXPECT(tc, !HybridIsRookEndgame(pos));

    pos.set("8/p5r1/1p6/1k1P4/4P2K/8/8/R7 w - - 8 78", false, &st);
    EXPECT(tc, HybridIsRookEndgame(pos));
    EXPECT(tc,
           HybridIsQuietCentralPawnPush(pos, UCIEngine::to_move(pos, "e4e5")));
    EXPECT(tc,
           !HybridIsQuietCentralPawnPush(pos, UCIEngine::to_move(pos, "a1d1")));
    EXPECT(tc,
           !HybridIsQuietCentralPawnPush(pos, UCIEngine::to_move(pos, "h4h5")));

    pos.set("8/8/1p1k2pp/p2p4/3K1PPP/1PP5/8/8 w - - 1 38", false, &st);
    EXPECT(tc, HybridIsPawnOnlyEndgame(pos));
    EXPECT(tc, HybridIsPawnLever(pos, UCIEngine::to_move(pos, "c3c4")));
    EXPECT(tc, HybridIsPawnOnlyMCTSANECandidate(
                   pos, UCIEngine::to_move(pos, "h4h5"),
                   UCIEngine::to_move(pos, "c3c4")));

    pos.set("8/8/1p1k2pp/p7/2pK1PPP/1P6/8/8 w - - 0 39", false, &st);
    EXPECT(tc, HybridIsPawnOnlyKingRecaptureCandidate(
                   pos, UCIEngine::to_move(pos, "b3c4"),
                   UCIEngine::to_move(pos, "d4c4")));
    EXPECT(tc, HybridIsPawnOnlyMCTSANECandidate(
                   pos, UCIEngine::to_move(pos, "b3c4"),
                   UCIEngine::to_move(pos, "d4c4")));

    pos.set("2b2k2/2q2p2/p1p2Ppr/4p3/2r1B3/5Q2/P1P4P/R2R3K w - - 0 25", false,
            &st);
    EXPECT(tc,
           HybridIsQuietCentralQueenMove(pos, UCIEngine::to_move(pos, "f3e3")));
    EXPECT(tc, !HybridIsQuietCentralQueenMove(pos,
                                              UCIEngine::to_move(pos, "f3e2")));
    EXPECT(tc, !HybridIsQuietCentralQueenMove(pos,
                                              UCIEngine::to_move(pos, "a1b1")));

    pos.set("1kq1r2r/p1p2pp1/1pPb4/3P4/Q1B2nbp/4B3/P2N1PPP/1R2R1K1 w - - 0 21",
            false, &st);
    EXPECT(tc,
           HybridIsQuietMinorMajorAttack(pos, UCIEngine::to_move(pos, "c4a6")));
    EXPECT(tc, !HybridIsQuietMinorMajorAttack(pos,
                                              UCIEngine::to_move(pos, "e3f4")));
    EXPECT(tc, !HybridIsQuietMinorMajorAttack(pos,
                                              UCIEngine::to_move(pos, "c4f1")));
    EXPECT(tc, !HybridIsQuietMinorMajorAttack(pos,
                                              UCIEngine::to_move(pos, "a4a5")));

    pos.set("8/1pp5/8/PP4p1/5pP1/3k4/1b1B4/5K2 w - - 0 43", false, &st);
    EXPECT(tc, HybridIsBishopOnlyEndgame(pos));
    EXPECT(tc, HybridIsQuietBishopBackRankRetreat(
                   pos, UCIEngine::to_move(pos, "d2e1")));
    EXPECT(tc, !HybridIsQuietBishopBackRankRetreat(
                   pos, UCIEngine::to_move(pos, "d2f4")));
    EXPECT(tc, !HybridIsQuietBishopBackRankRetreat(
                   pos, UCIEngine::to_move(pos, "a5a6")));

    pos.set("8/8/1p1k2pp/p2p4/3K1PPP/1PP5/8/8 w - - 1 38", false, &st);
    EXPECT(tc, HybridIsPawnOnlyEndgame(pos));
    EXPECT(tc, HybridANEProbeAllowedForPosition(pos, true));
    EXPECT(tc, HybridANEProbeAllowedForPosition(pos, false));
    EXPECT(tc, HybridIsPawnOnlyMCTSANECandidate(
                   pos, UCIEngine::to_move(pos, "h4h5"),
                   UCIEngine::to_move(pos, "c3c4")));
    EXPECT(tc, !HybridIsPawnOnlyKingRecaptureCandidate(
                   pos, UCIEngine::to_move(pos, "h4h5"),
                   UCIEngine::to_move(pos, "c3c4")));
    pos.set("8/8/1p1k2pp/p7/2pK1PPP/1P6/8/8 w - - 0 39", false, &st);
    EXPECT(tc, HybridIsPawnOnlyEndgame(pos));
    EXPECT(tc, HybridIsPawnOnlyMCTSANECandidate(
                   pos, UCIEngine::to_move(pos, "b3c4"),
                   UCIEngine::to_move(pos, "d4c4")));
    EXPECT(tc, HybridIsPawnOnlyKingRecaptureCandidate(
                   pos, UCIEngine::to_move(pos, "b3c4"),
                   UCIEngine::to_move(pos, "d4c4")));
    EXPECT(tc, !HybridIsPawnOnlyPawnCaptureCandidate(
                   pos, UCIEngine::to_move(pos, "b3c4"),
                   UCIEngine::to_move(pos, "d4c4")));
    pos.set("8/8/p7/1ppp2k1/4p1P1/P2P3K/1PP4P/8 b - - 1 41", false, &st);
    EXPECT(tc, HybridIsPawnOnlyEndgame(pos));
    EXPECT(tc, HybridIsPawnOnlyMCTSANECandidate(
                   pos, UCIEngine::to_move(pos, "e4e3"),
                   UCIEngine::to_move(pos, "e4d3")));
    EXPECT(tc, HybridIsPawnOnlyPawnCaptureCandidate(
                   pos, UCIEngine::to_move(pos, "e4e3"),
                   UCIEngine::to_move(pos, "e4d3")));
    EXPECT(tc, !HybridIsPawnOnlyPawnCaptureCandidate(
                   pos, UCIEngine::to_move(pos, "e4d3"),
                   UCIEngine::to_move(pos, "e4e3")));
    pos.set("4k3/8/8/8/2pK4/1P6/8/4N3 w - - 0 39", false, &st);
    EXPECT(tc, !HybridIsPawnOnlyEndgame(pos));
    EXPECT(tc, !HybridANEProbeAllowedForPosition(pos, true));
    EXPECT(tc, HybridANEProbeAllowedForPosition(pos, false));
    EXPECT(tc, !HybridIsPawnOnlyMCTSANECandidate(
                   pos, UCIEngine::to_move(pos, "b3c4"),
                   UCIEngine::to_move(pos, "d4c4")));
    EXPECT(tc, !HybridIsPawnOnlyKingRecaptureCandidate(
                   pos, UCIEngine::to_move(pos, "b3c4"),
                   UCIEngine::to_move(pos, "d4c4")));
  }
  {
    TestCase tc("MCTS visit evidence handles cache-heavy playouts");

    EXPECT(tc, HybridMCTSVisitEvidenceSane(270, 260, 270, 140));
    EXPECT(tc, HybridMCTSVisitEvidenceSane(1024, 260, 1024, 512));
    EXPECT(tc, HybridMCTSVisitEvidenceSane(12645, 234, 12644, 12493));
    EXPECT(tc, !HybridMCTSVisitEvidenceSane(260, 260, 340, 200));
    EXPECT(tc, !HybridMCTSVisitEvidenceSane(260, 260, 270, 300));
    EXPECT(tc, !HybridMCTSVisitEvidenceSane(0, 0, 10, 1));
  }
  {
    TestCase tc("AB root rejection blocks low-effort MCTS blunders");

    EXPECT(tc,
           HybridABRootRejectsMCTS(true, 1, 5, -410, -447, 2523397, 649, -447));
    EXPECT(tc, HybridABRootRejectsMCTS(true, 1, 5, -410, -447, 2523397, 649,
                                       -32001));
    EXPECT(tc, HybridABRootRejectsMCTS(true, 1, 4, -432, -436, 25349399, 1663,
                                       -32001));
    EXPECT(tc, !HybridABRootRejectsMCTS(false, 1, 5, -410, -447, 2523397, 649,
                                        -32001));
    EXPECT(tc, !HybridABRootRejectsMCTS(true, 1, 1, -410, -410, 2523397,
                                        2523397, -401));
    EXPECT(tc, !HybridABRootRejectsMCTS(true, 1, 4, -410, -425, 2523397,
                                        2200000, -420));
  }
  {
    TestCase tc("Fixed-budget cross-root MCTS override predicate");

    EXPECT(tc, HybridMCTSCrossRootConfidenceOverride(
                   true, true, 316, 203, 0.642f, 0.141f, 224, 61, 606, 2,
                   -32001, 585, 1339424, 5, 16, 0.372f, 0.634f));
    EXPECT(tc, HybridMCTSCrossRootConfidenceOverride(
                   true, true, 265, 175, 0.660f, 0.145f, 226, 55, 619, 2,
                   -32001, 564, 2095896, 5, 16, 0.372f, 0.638f));
    EXPECT(tc, !HybridMCTSCrossRootConfidenceOverride(
                   true, true, 265, 169, 0.660f, 0.145f, 226, 55, 619, 2,
                   -32001, 564, 2095896, 5, 16, 0.372f, 0.638f));
    EXPECT(tc, !HybridMCTSCrossRootConfidenceOverride(
                   true, true, 316, 203, 0.642f, 0.141f, 224, 61, 606, 2, 580,
                   585, 1339424, 5, 16, 0.372f, 0.634f));
    EXPECT(tc, !HybridMCTSCrossRootConfidenceOverride(
                   true, true, 316, 203, 0.642f, 0.141f, 224, 61, 700, 2,
                   -32001, 585, 1339424, 5, 16, 0.372f, 0.634f));
    EXPECT(tc, !HybridMCTSCrossRootConfidenceOverride(
                   true, true, 316, 203, 0.642f, 0.141f, 224, 61, 606, 2,
                   -32001, 585, 1339424, 2, 80, 0.520f, 0.634f));
  }
  {
    TestCase tc(
        "Root-confidence MCTS can bypass unbounded low-effort AB reject");

    EXPECT(tc, HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 3292, -232, -352, 2, 24,
                   -0.349f, 0.717f));
    EXPECT(tc, HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 6159, -227, -270, 2, 20,
                   -0.349f, 0.736f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   false, true, 2, -32001, false, false, 3292, -232, -352, 2,
                   24, -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, false, 2, -32001, false, false, 3292, -232, -352, 2,
                   24, -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 4, -32001, false, false, 3292, -232, -352, 2, 24,
                   -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, true, 3292, -232, -352, 2, 24,
                   -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 10001, -232, -352, 2,
                   24, -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 3292, -220, -350, 2, 24,
                   -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 3292, -232, -352, 2, 9,
                   -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 3292, -232, -352, 2, 24,
                   0.130f, 0.717f));
  }
  {
    TestCase tc("Low-material pawn push can bypass stale AB root reject");

    EXPECT(tc, HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   false, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, false, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, false, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, false, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, false, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 699, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 499, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.670f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.440f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   129, 0, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 40, 0, 2, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 1, -32001, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, 331, 33049, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 19999, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 80001, 2, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 3, 248, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 119, -0.038f, 0.476f));
    EXPECT(tc, !HybridMCTSRootRejectLowMaterialPushOverride(
                   true, true, true, true, true, 868, 609, 0.702f, 0.514f, 155,
                   155, 0, 0, 2, -32001, 33049, 2, 248, 0.030f, 0.476f));
  }
  {
    TestCase tc(
        "Rook endgame central pawn push can bypass stale AB root reject");

    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   318, 4, -32001, 0, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 58, 52, 0.897f, 0.681f, 476,
                   310, 2, -32001, 2009, 2, 3, 0.169f, 0.920f));
    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 51, 46, 0.902f, 0.676f, 467,
                   333, 3, -32001, 39, 3, 2, 0.178f, 0.915f));
    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, false, true, true, true, 2055, 1984, 0.965f, 0.550f,
                   483, 400, 2, -32001, 819, 2, 27, 0.287f, 0.923f));
    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 1174, 1124, 0.957f, 0.632f,
                   512, 447, 6, -32001, 0, 2, 18, 0.235f, 0.936f));
    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 102, 93, 0.912f, 0.708f, 472,
                   412, 5, -32001, 1297, 2, 4, 0.193f, 0.918f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   false, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, false, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, false, true, true, true, 2055, 1984, 0.949f, 0.550f,
                   483, 400, 2, -32001, 819, 2, 27, 0.287f, 0.923f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, false, true, true, true, 2055, 1984, 0.965f, 0.499f,
                   483, 400, 2, -32001, 819, 2, 27, 0.287f, 0.923f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, false, true, true, true, 2055, 1984, 0.965f, 0.550f,
                   483, 379, 2, -32001, 819, 2, 27, 0.287f, 0.923f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, false, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, false, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, false, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 44, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 39, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.879f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.590f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 429,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   299, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 1, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 7, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, 388, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 20001, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 1, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 0, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 33, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.330f, 0.916f));
  }
  {
    TestCase tc("Quiet queen move can bypass stale low-effort AB root reject");

    EXPECT(tc, HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 55, 47, 0.855f, 1.117f, 354, 374, 6,
                   -32001, false, false, 88, 3, 1, -0.290f, 0.828f));
    EXPECT(tc, HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 972, 948, 0.975f, 1.294f, 483, 500,
                   2, -32001, false, false, 8, 4, 3, -0.811f, 0.483f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   false, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, false, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, false, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, false, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 49, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 46, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.849f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.049f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 329, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 329, 5,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 1,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 7,
                   -32001, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   120, false, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, true, false, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, true, 4, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 513, 3, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 1, 1, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 0, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 5, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 47, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 4, -0.290f, 0.831f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 56, 48, 0.857f, 1.121f, 357, 373, 5,
                   -32001, false, false, 4, 3, 1, -0.200f, 0.831f));
  }
  {
    TestCase tc("Reused root MCTS confidence stays tightly gated");

    EXPECT(tc, HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, HybridMCTSReusedRootConfidenceOverride(
                   true, 933, 914, 29, 29, 0.980f, 1.000f, 430, 445, -59, -110,
                   2, -32001, 120884, 2, 0, -0.027f, 0.893f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   false, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 100, 90, 0.971f, 0.900f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 199, 180, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 23, 23, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.899f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.899f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 299, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 299, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -50, -130,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 250001, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 3, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, 0.090f, 0.828f));
  }
  {
    TestCase tc("Reused root current confirmation stays narrowly gated");

    EXPECT(tc, HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 541, 480, 52, 52, 0.887f, 1.000f,
                   0.000f, 353, 340, 2, 369, true, false, 488229, 3, 0, 0.243f,
                   0.827f));
    EXPECT(tc, HybridMCTSReusedRootCurrentOverride(
                   true, true, false, true, 758, 685, 40, 39, 0.904f, 0.975f,
                   0.831f, 370, 367, 4, -32001, false, false, 60, 2, 0, 0.311f,
                   0.844f));
    EXPECT(tc, HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 2309, 2203, 27, 26, 0.954f, 0.963f,
                   0.710f, 429, 429, 2, 56, true, false, 359378, 3, 0, 0.273f,
                   0.892f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   false, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, false, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, true, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 349, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 23, 23, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.839f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 0.919f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 299, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 299, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 249999, 2, 0, 0.311f,
                   0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 2,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.330f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, true, 758, 685, 40, 39, 0.904f, 0.975f,
                   0.740f, 370, 367, 4, -32001, false, false, 60, 2, 0, 0.311f,
                   0.844f));
  }
  {
    TestCase tc("Bishop endgame retreat override stays narrowly gated");

    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc, HybridMCTSBishopEndgameRetreatOverride(
                   true, true, false, true, true, 1305, 1224, 32, 32, 0.938f,
                   1.000f, 399, 394, 2, 31, true, false, 289466, 3, 0, 0.311f,
                   0.870f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, true, true, true, 622, 556, 38, 37, 0.894f, 0.974f,
               353, 228, 2, 475, true, false, 579529, 3, 0, 0.243f, 0.827f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               false, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, false, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, false, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, false, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 199, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 179, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 23, 23, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.749f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 0.939f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               279, 294, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 209, 4, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 5, -32001, false, false, 144, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 10001, 3, 0, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 2, 0.243f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 265, 206, 80, 80, 0.777f, 1.000f,
               300, 294, 4, -32001, false, false, 144, 3, 0, 0.270f, 0.763f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, true, true, true, 622, 556, 38, 37, 0.894f, 0.974f,
               353, 219, 2, 475, true, false, 579529, 3, 0, 0.243f, 0.827f));
  }
  {
    TestCase tc("Compact fixed-budget MCTS override stays narrowly gated");

    EXPECT(tc, HybridMCTSCompactFixedBudgetOverride(true, true, false, 65, 56,
                                                    0.862f, 0.660f, 169, 169,
                                                    -1, -1));
    EXPECT(tc, HybridMCTSCompactFixedBudgetOverride(true, true, false, 71, 66,
                                                    0.930f, 1.004f, 294, 294,
                                                    -1, -1));
    EXPECT(tc, HybridMCTSCompactFixedBudgetOverride(true, true, false, 58, 49,
                                                    0.845f, 0.607f, 148, 150,
                                                    -29, -115));
    EXPECT(tc, !HybridMCTSCompactFixedBudgetOverride(true, true, true, 70, 36,
                                                     0.514f, 0.731f, 120, 178,
                                                     -225, -272));
    EXPECT(tc, !HybridMCTSCompactFixedBudgetOverride(true, true, false, 143, 72,
                                                     0.503f, 0.000f, 266, 257,
                                                     1, 30));
    EXPECT(tc, !HybridMCTSCompactFixedBudgetOverride(true, true, false, 190,
                                                     120, 0.632f, 0.500f, 220,
                                                     180, 0, 0));
  }
  {
    TestCase tc("Compact clear-preference MCTS override stays narrowly gated");

    EXPECT(tc, HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, HybridMCTSCompactClearPreferenceOverride(
                   true, true, 101, 97, 0.960f, 0.793f, 282, 307, -25, 2,
                   -32001, false, true, 142554, 2, 1, -0.057f, 0.736f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   false, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, false, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 80, 46, 0.575f, 0.589f, 73, 134, -61, 3, -32001,
                   false, false, 1234, 2, 24, -0.349f, 0.240f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.899f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.749f, 291, 318, -27, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 279, 318, -27, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 299, -27, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -41, 2,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, false, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 19999, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 200001, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 1,
                   -32001, false, true, 28629, 2, 1, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 28629, 2, 3, -0.057f, 0.750f));
    EXPECT(tc, !HybridMCTSCompactClearPreferenceOverride(
                   true, true, 113, 109, 0.965f, 0.806f, 291, 318, -27, 2,
                   -32001, false, true, 28629, 2, 1, 0.010f, 0.750f));
  }
  {
    TestCase tc("Root Q gap ignores unvisited placeholders");

    const uint32_t visits[] = {141, 96, 0, 0};
    const float qs[] = {-0.402f, -0.466f, 0.0f, 0.0f};
    const float gap = HybridVisitedRootQGap(-0.360f, visits, qs, 4);
    EXPECT(tc, gap > 0.041f && gap < 0.043f);

    const uint32_t unvisited[] = {0, 0};
    const float placeholder_qs[] = {0.0f, 0.0f};
    EXPECT(tc, HybridVisitedRootQGap(-0.250f, unvisited, placeholder_qs, 2) ==
                   0.0f);
  }
  {
    TestCase tc("Low-node Hybrid uses MCTS-primary budget");

    MetalFish::Search::LimitsType limits;
    limits.nodes = 50;
    EXPECT(tc, HybridUseMCTSPrimaryForFixedNodeBudget(limits));
    EXPECT(tc, HybridLowNodeABProbeNodes(limits.nodes) == 12);
    EXPECT(tc, HybridLowNodeMCTSPrimaryReady(true, limits.nodes, 38, 5, true));
    EXPECT(tc,
           !HybridLowNodeMCTSPrimaryReady(true, limits.nodes, 37, 4, false));

    limits.nodes = 512;
    EXPECT(tc, HybridUseMCTSPrimaryForFixedNodeBudget(limits));
    EXPECT(tc, HybridLowNodeABProbeNodes(limits.nodes) == 64);

    limits.nodes = 1024;
    EXPECT(tc, HybridUseMCTSPrimaryForFixedNodeBudget(limits));
    EXPECT(tc, HybridLowNodeABProbeNodes(limits.nodes) == 64);

    limits.nodes = 1025;
    EXPECT(tc, !HybridUseMCTSPrimaryForFixedNodeBudget(limits));

    limits.nodes = 50;
    limits.movetime = 1000;
    EXPECT(tc, !HybridUseMCTSPrimaryForFixedNodeBudget(limits));

    limits.movetime = 0;
    limits.ponderMode = true;
    EXPECT(tc, !HybridUseMCTSPrimaryForFixedNodeBudget(limits));
  }
  {
    TestCase tc("Subsearch stop grace stays under bot stop timeout");

    EXPECT(tc, HybridSubsearchJoinGraceMs(true, 30000) == 2500);
    EXPECT(tc, HybridSubsearchJoinGraceMs(false, 0) == 1000);
    EXPECT(tc, HybridSubsearchJoinGraceMs(false, 2000) == 1000);
    EXPECT(tc, HybridSubsearchJoinGraceMs(false, 30000) == 5000);
  }
}

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

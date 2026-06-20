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
    TestCase tc("AB root history reuse requires matching position");
    const std::string root = "8/8/1pk5/3p2p1/K2Pn1N1/4P3/8/8 b - - 17 73";

    EXPECT(tc, HybridCanReuseABPositionHistory(root, root));
    EXPECT(tc, !HybridCanReuseABPositionHistory("", root));
    EXPECT(tc, !HybridCanReuseABPositionHistory(
                   "8/8/1pk5/3p2p1/K2Pn1N1/4P3/8/8 w - - 17 73", root));
  }
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
    EXPECT(tc, !config.mcts_config.high_policy_root_lever_selection);
    EXPECT(tc, !config.mcts_config.low_policy_root_lever_selection);
    EXPECT(tc, !config.mcts_config.low_visit_q_override_rescan);
    EXPECT(tc, config.mcts_ab_root_hint_delay_ms == 0);
    EXPECT(tc, config.mcts_ab_root_hint_count == 8);
    EXPECT(tc, config.ab_candidate_verify_ms == 120);
    EXPECT(tc, config.ab_candidate_verify_count == 5);
    EXPECT(tc, config.root_pawn_lever_tiebreak);
    EXPECT(tc, !config.ane_root_probe);
    EXPECT(tc, !config.ane_root_hints);
    EXPECT(tc, !config.ane_only_pawn_endgames);
    EXPECT(tc, config.ane_compute_units == "cpu-ne");
    EXPECT(tc, config.ane_root_hint_count == 6);
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
    limits.movetime = 999;
    EXPECT(tc, HybridABCandidateVerifyBudgetMs(limits, 999, 150, false) == 0);
    limits.movetime = 1000;
    EXPECT(tc,
           HybridABCandidateVerifyBudgetMs(limits, 1000, 150, false) == 150);

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
                   true, true, 380, 260, 0.684f, 0.137f, 222, 54));
    EXPECT(tc, !HybridMCTSRootConfidenceFixedBudgetOverride(
                   true, true, 262, 137, 0.523f, 0.066f, 20, 20));
  }
  {
    TestCase tc("Low-node root-confidence MCTS override stays narrow");

    EXPECT(tc, HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 73, 53, 0.726f, 0.354f, 239, 236, 15, -1,
                   4, -32001, false, false, 23, 3, 9, 0.242f, 0.663f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   false, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, false, false, 52, 42, 0.808f, 0.586f, 198, 192, -11,
                   -7, 2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, true, 52, 42, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 49, 42, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 39, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.699f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.299f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 179, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 149, -11, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, 30, -7,
                   2, -9, true, false, 71601, 2, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 1, 5, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 13, -0.007f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 71601, 2, 5, 0.190f, 0.579f));
    EXPECT(tc, !HybridMCTSLowNodeRootConfidenceOverride(
                   true, true, false, 52, 42, 0.808f, 0.586f, 198, 192, -11, -7,
                   2, -9, true, false, 49999, 2, 5, -0.007f, 0.579f));
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
    EXPECT(tc, HybridMCTSShortRootTacticalOverride(
                   true, true, true, 294, 187, 0.636f, 0.180f, 229, 60, 622,
                   563, 2, -32001, false, 1867440, 2, 32, 0.421f, 0.644f));
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
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 294, 187, 0.636f, 0.180f, 229, 60, 633,
                   563, 2, -32001, false, 1867440, 2, 32, 0.421f, 0.644f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 361, 187, 0.636f, 0.180f, 229, 60, 622,
                   563, 2, -32001, false, 1867440, 2, 32, 0.421f, 0.644f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 294, 187, 0.636f, 0.180f, 229, 60, 622,
                   563, 2, -32001, false, 1867440, 2, 65, 0.421f, 0.644f));
    EXPECT(tc, !HybridMCTSShortRootTacticalOverride(
                   true, true, true, 294, 187, 0.636f, 0.180f, 229, 60, 622,
                   563, 2, -32001, false, 1867440, 2, 32, 0.445f, 0.644f));
  }
  {
    TestCase tc("Verified AB hint supports MCTS override");

    EXPECT(tc,
           HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 268, 167, 0.623f, 0.187f, 232, 62, 623, 561, 2,
               -VALUE_INFINITE, false, true, 332488, 2, 32, 0.421f, 0.651f));
    EXPECT(tc,
           HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 255, 158, 0.620f, 0.184f, 232, 72, 589, 564, 2,
               -VALUE_INFINITE, false, true, 2018929, 2, 32, 0.421f, 0.649f));
    EXPECT(tc,
           !HybridMCTSVerifiedHintSupportOverride(
               true, true, false, 268, 167, 0.623f, 0.187f, 232, 62, 623, 561,
               2, -VALUE_INFINITE, false, true, 332488, 2, 32, 0.421f, 0.651f));
    EXPECT(tc,
           !HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 268, 167, 0.623f, 0.159f, 232, 62, 623, 561, 2,
               -VALUE_INFINITE, false, true, 332488, 2, 32, 0.421f, 0.651f));
    EXPECT(tc,
           !HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 268, 167, 0.623f, 0.187f, 232, 62, 632, 561, 2,
               -VALUE_INFINITE, false, true, 332488, 2, 32, 0.421f, 0.651f));
    EXPECT(tc,
           !HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 268, 167, 0.623f, 0.187f, 232, 62, 623, 561, 2,
               -VALUE_INFINITE, false, false, 332488, 2, 32, 0.421f, 0.651f));
    EXPECT(tc,
           !HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 268, 167, 0.623f, 0.187f, 232, 62, 623, 561, 2,
               -VALUE_INFINITE, false, true, 2500001, 2, 32, 0.421f, 0.651f));
    EXPECT(tc,
           !HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 268, 167, 0.623f, 0.187f, 232, 62, 623, 561, 2,
               -VALUE_INFINITE, false, true, 332488, 2, 65, 0.421f, 0.651f));
    EXPECT(tc,
           !HybridMCTSVerifiedHintSupportOverride(
               true, true, true, 268, 167, 0.623f, 0.187f, 232, 62, 623, 561, 2,
               -VALUE_INFINITE, false, true, 332488, 2, 32, 0.455f, 0.651f));
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
    TestCase tc("Ultra-low root-confidence MCTS override stays narrow");

    EXPECT(tc, HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 35, 30, 0.857f, 0.671f, 458, 265, 193, 688, -4,
                   4, -VALUE_INFINITE, 103, 3, 2, 0.178f, 0.910f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   false, true, 29, 22, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, false, 29, 22, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 41, 22, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 19, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.730f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.540f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.566f, 169, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.566f, 215, 169, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.566f, 215, 230, 221, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 35, 30, 0.857f, 0.671f, 458, 265, 193, 720, -4,
                   4, -VALUE_INFINITE, 103, 3, 2, 0.178f, 0.910f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   5, -VALUE_INFINITE, 492, 2, 4, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 5, 0.049f, 0.616f));
    EXPECT(tc, !HybridMCTSUltraLowNodeRootConfidenceOverride(
                   true, true, 29, 22, 0.759f, 0.566f, 215, 230, -15, -49, -52,
                   3, -VALUE_INFINITE, 492, 2, 4, 0.200f, 0.616f));
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
                   true, true, true, 38, 25, 0.658f, 1.049f, 298, 329, 4,
                   -VALUE_INFINITE, 4, 4, 1, -0.290f, 0.759f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 20, 18, 0.900f, 0.750f, 311, 312, 3,
                   -VALUE_INFINITE, 394, 2, 1, 0.028f, 0.778f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 13, 10, 0.769f, 0.322f, 83, 83, 2,
                   -VALUE_INFINITE, 1860, 2, 1, -0.239f, 0.083f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 28, 24, 0.857f, 0.774f, 766, 743, 3,
                   -VALUE_INFINITE, 7587, 2, 3, -0.155f, 0.619f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 13, 10, 0.769f, 0.299f, 83, 83, 2,
                   -VALUE_INFINITE, 1860, 2, 1, -0.239f, 0.083f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 13, 10, 0.769f, 0.322f, 74, 83, 2,
                   -VALUE_INFINITE, 1860, 2, 1, -0.239f, 0.083f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 13, 10, 0.769f, 0.322f, 83, 74, 2,
                   -VALUE_INFINITE, 1860, 2, 1, -0.239f, 0.083f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 13, 10, 0.769f, 0.322f, 83, 83, 7,
                   -VALUE_INFINITE, 1860, 2, 1, -0.239f, 0.083f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 13, 10, 0.769f, 0.322f, 83, 83, 2,
                   -VALUE_INFINITE, 10001, 2, 1, -0.239f, 0.083f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 13, 10, 0.769f, 0.322f, 83, 83, 2,
                   -VALUE_INFINITE, 1860, 2, 4, -0.239f, 0.083f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 38, 25, 0.658f, 0.899f, 298, 329, 4,
                   -VALUE_INFINITE, 4, 4, 1, -0.290f, 0.759f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 38, 25, 0.658f, 1.049f, 298, 329, 4,
                   -VALUE_INFINITE, 4, 4, 3, -0.290f, 0.759f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 38, 25, 0.658f, 1.049f, 298, 329, 4, -500,
                   4, 4, 1, -0.290f, 0.759f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, false, true, 2192, 2113, 0.964f, 1.149f, 336, 396, 3,
                   -VALUE_INFINITE, 1862, 2, 27, -0.341f, 0.808f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, false, true, 2192, 2113, 0.964f, 1.149f, 336, 396, 3,
                   -VALUE_INFINITE, 2601, 2, 27, -0.341f, 0.808f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.963f, 1.463f, 603, 685, 2,
                   -VALUE_INFINITE, 55, 2, 1, -0.498f, 0.965f));
    EXPECT(tc, HybridMCTSRootRejectQGapOverride(
                   true, true, true, 56, 51, 0.911f, 1.462f, 601, 686, 3,
                   -VALUE_INFINITE, 61, 2, 1, -0.498f, 0.964f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.879f, 1.463f, 603, 685, 2,
                   -VALUE_INFINITE, 55, 2, 1, -0.498f, 0.965f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.963f, 1.199f, 603, 685, 2,
                   -VALUE_INFINITE, 55, 2, 1, -0.498f, 0.965f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.963f, 1.463f, 499, 685, 2,
                   -VALUE_INFINITE, 55, 2, 1, -0.498f, 0.965f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.963f, 1.463f, 603, 599, 2,
                   -VALUE_INFINITE, 55, 2, 1, -0.498f, 0.965f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.963f, 1.463f, 603, 685, 2,
                   -VALUE_INFINITE, 101, 2, 1, -0.498f, 0.965f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 54, 52, 0.963f, 1.463f, 603, 685, 2,
                   -VALUE_INFINITE, 55, 2, 3, -0.498f, 0.965f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 198, 158, 0.798f, 1.157f, 397, 408, 5,
                   -VALUE_INFINITE, 69, 4, 1, -0.290f, 0.868f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 40, 11, 0.275f, 0.072f, 111, 111, 3,
                   -VALUE_INFINITE, 631, 4, 2, 0.002f, 0.356f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 18, 13, 0.722f, 0.350f, 83, 83, 2,
                   -VALUE_INFINITE, 802, 2, 2, -0.078f, 0.272f));
    EXPECT(tc, !HybridMCTSRootRejectQGapOverride(
                   true, true, true, 18, 13, 0.699f, 0.350f, 83, 83, 2,
                   -VALUE_INFINITE, 802, 2, 2, -0.078f, 0.272f));
  }
  {
    TestCase tc("MCTS root-reject kingside pawn-push predicate");

    EXPECT(tc, HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   false, true, true, true, 18, 11, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, false, 18, 11, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 14, 11, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 9, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 15, 10, 0.667f, 0.323f, 265, 209, 2,
                   -VALUE_INFINITE, 23641, 2, 1, 0.387f, 0.710f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.590f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.299f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.314f, 239, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.314f, 257, 189, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.314f, 257, 207, 3,
                   -VALUE_INFINITE, 3052, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 25001, 2, 3, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 5, 0.382f, 0.696f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 18, 11, 0.611f, 0.314f, 257, 207, 2,
                   -VALUE_INFINITE, 3052, 2, 3, 0.397f, 0.696f));
    EXPECT(tc, HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 19, 19, 1.000f, 0.000f, 272, 249, 2,
                   -VALUE_INFINITE, 1283, 2, 0, 0.386f, 0.721f));
    EXPECT(tc, HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 17, 10, 0.588f, 0.296f, 247, 210, 2,
                   -VALUE_INFINITE, 18535, 2, 3, 0.382f, 0.678f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 19, 17, 1.000f, 0.000f, 272, 249, 2,
                   -VALUE_INFINITE, 1283, 2, 0, 0.386f, 0.721f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 19, 19, 1.000f, 0.000f, 269, 249, 2,
                   -VALUE_INFINITE, 1283, 2, 0, 0.386f, 0.721f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 19, 19, 1.000f, 0.000f, 272, 214, 2,
                   -VALUE_INFINITE, 1283, 2, 0, 0.386f, 0.721f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 19, 19, 1.000f, 0.000f, 272, 249, 2,
                   -VALUE_INFINITE, 20001, 2, 0, 0.386f, 0.721f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 19, 19, 1.000f, 0.000f, 272, 249, 2,
                   -VALUE_INFINITE, 1283, 2, 1, 0.386f, 0.721f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 19, 19, 1.000f, 0.001f, 272, 249, 2,
                   -VALUE_INFINITE, 1283, 2, 0, 0.386f, 0.721f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 17, 10, 0.588f, 0.239f, 247, 210, 2,
                   -VALUE_INFINITE, 18535, 2, 3, 0.382f, 0.678f));
    EXPECT(tc, !HybridMCTSRootRejectKingsidePawnPushOverride(
                   true, true, true, true, 17, 10, 0.588f, 0.296f, 219, 210, 2,
                   -VALUE_INFINITE, 18535, 2, 3, 0.382f, 0.678f));
  }
  {
    TestCase tc("Clock-managed MCTS root-reject Q-gap override predicate");

    EXPECT(tc, HybridMCTSClockRootRejectQGapOverride(
                   true, true, true, 198, 158, 0.798f, 1.157f, 397, 408, 5,
                   -VALUE_INFINITE, 69, 4, 1, -0.290f, 0.868f));
    EXPECT(tc, HybridMCTSClockRootRejectQGapOverride(
                   true, true, true, 124, 85, 0.685f, 1.118f, 354, 373, 5,
                   -VALUE_INFINITE, 387, 4, 1, -0.290f, 0.828f));
    EXPECT(tc, !HybridMCTSClockRootRejectQGapOverride(
                   false, true, true, 198, 158, 0.798f, 1.157f, 397, 408, 5,
                   -VALUE_INFINITE, 69, 4, 1, -0.290f, 0.868f));
    EXPECT(tc, !HybridMCTSClockRootRejectQGapOverride(
                   true, true, true, 198, 158, 0.798f, 1.157f, 397, 408, 5,
                   -VALUE_INFINITE, 1001, 4, 1, -0.290f, 0.868f));
    EXPECT(tc, !HybridMCTSClockRootRejectQGapOverride(
                   true, true, true, 198, 158, 0.798f, 1.157f, 397, 408, 5,
                   -VALUE_INFINITE, 69, 4, 5, -0.290f, 0.868f));
    EXPECT(tc, !HybridMCTSClockRootRejectQGapOverride(
                   true, true, true, 198, 158, 0.798f, 0.899f, 397, 408, 5,
                   -VALUE_INFINITE, 69, 4, 1, -0.290f, 0.868f));
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
    EXPECT(tc, HybridANEConfirmedMCTSOverride(true, true, true, true, 68, 60,
                                              0.882f, 0.431f, 91, 89, 0.599f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 47, 25,
                                          0.532f, 1.049f, 298, 317, 0.182f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 20, 16,
                                          0.800f, 0.772f, 742, 681, 0.409f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 35, 31,
                                          0.886f, 0.947f, 223, 223, 1.022f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 43, 38,
                                          0.884f, 0.685f, 485, 329, 0.127f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 13,
                                          0.812f, 0.443f, 394, 322, 0.127f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 33, 28,
                                          0.848f, 0.712f, 783, 737, 0.409f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 34, 29,
                                          0.853f, 0.712f, 783, 764, 0.409f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 52, 43,
                                          0.827f, 0.849f, 570, 586, 0.019f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 57, 57,
                                          1.000f, 0.000f, 1009, 1009, 1.663f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 24, 20,
                                          0.833f, 0.844f, 176, 176, 1.022f));
    EXPECT(tc,
           HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 12,
                                          0.750f, 0.768f, 709, 652, 0.409f));
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
    EXPECT(tc, !HybridANEConfirmedMCTSOverride(true, true, true, true, 68, 60,
                                               0.859f, 0.431f, 91, 89, 0.599f));
    EXPECT(tc, !HybridANEConfirmedMCTSOverride(true, true, true, true, 68, 60,
                                               0.882f, 0.399f, 91, 89, 0.599f));
    EXPECT(tc, !HybridANEConfirmedMCTSOverride(true, true, true, true, 68, 60,
                                               0.882f, 0.431f, 79, 89, 0.599f));
    EXPECT(tc, !HybridANEConfirmedMCTSOverride(true, true, true, true, 68, 60,
                                               0.882f, 0.431f, 91, 79, 0.599f));
    EXPECT(tc, !HybridANEConfirmedMCTSOverride(true, true, true, true, 68, 60,
                                               0.882f, 0.431f, 91, 89, 0.449f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                           0.981f, 0.499f, 424, 437, 0.401f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                           0.981f, 0.918f, 199, 437, 0.401f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 54, 53,
                                           0.981f, 0.918f, 424, 149, 0.401f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 47, 25,
                                           0.439f, 1.049f, 298, 317, 0.182f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 47, 25,
                                           0.532f, 0.999f, 298, 317, 0.182f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 47, 25,
                                           0.532f, 1.049f, 279, 317, 0.182f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 47, 25,
                                           0.532f, 1.049f, 298, 299, 0.182f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 47, 25,
                                           0.532f, 1.049f, 298, 317, 0.149f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 20, 16,
                                           0.800f, 0.699f, 742, 681, 0.409f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 35, 31,
                                           0.886f, 0.899f, 223, 223, 1.022f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 43, 38,
                                           0.884f, 0.685f, 449, 329, 0.127f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 13,
                                           0.812f, 0.399f, 394, 322, 0.127f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 13,
                                           0.812f, 0.443f, 379, 322, 0.127f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 13,
                                           0.812f, 0.443f, 394, 259, 0.127f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 13,
                                           0.812f, 0.443f, 394, 322, 0.099f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 33, 28,
                                           0.839f, 0.712f, 783, 737, 0.409f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 33, 28,
                                           0.848f, 0.699f, 783, 737, 0.409f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 35, 29,
                                           0.853f, 0.712f, 783, 764, 0.409f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 52, 43,
                                           0.827f, 0.799f, 570, 586, 0.019f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 52, 43,
                                           0.827f, 0.849f, 570, 586, 0.014f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 57, 57,
                                           0.949f, 0.000f, 1009, 1009, 1.663f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 57, 57,
                                           1.000f, 0.000f, 1009, 1009, 0.999f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 24, 20,
                                           0.819f, 0.844f, 176, 176, 1.022f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 24, 20,
                                           0.833f, 0.789f, 176, 176, 1.022f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 24, 20,
                                           0.833f, 0.844f, 149, 176, 1.022f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 12,
                                           0.749f, 0.768f, 709, 652, 0.409f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 12,
                                           0.750f, 0.699f, 709, 652, 0.409f));
    EXPECT(tc,
           !HybridANEConfirmedMCTSOverride(true, true, true, true, 16, 12,
                                           0.750f, 0.768f, 649, 652, 0.409f));
  }
  {
    TestCase tc("ANE Q-supported root override predicate");

    EXPECT(tc, HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 53, 2, 28, 0.785f, 0.134f,
                   32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 58, 2, 25, 0.759f, 0.134f,
                   27, -0.842f, 4, 1, -0.290f, 6, -VALUE_INFINITE, -550, false,
                   0, -71));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   false, true, true, true, 1, 0.182f, 53, 2, 28, 0.785f,
                   0.134f, 32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164,
                   true, 182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, false, 1, 0.182f, 53, 2, 28, 0.785f,
                   0.134f, 32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164,
                   true, 182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 2, 0.182f, 53, 2, 28, 0.785f, 0.134f,
                   32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.149f, 53, 2, 28, 0.785f, 0.134f,
                   32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 121, 2, 28, 0.785f,
                   0.134f, 32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164,
                   true, 182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 53, 3, 28, 0.785f, 0.134f,
                   32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 53, 2, 19, 0.785f, 0.134f,
                   32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 53, 2, 28, 0.540f, 0.134f,
                   32, -0.803f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 53, 2, 28, 0.785f, 0.134f,
                   32, 0.100f, 4, 1, -0.290f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 53, 2, 28, 0.785f, 0.134f,
                   32, -0.803f, 4, 1, 0.100f, 2, -VALUE_INFINITE, -164, true,
                   182, -65));
    EXPECT(tc, !HybridANEQSupportedRootOverride(
                   true, true, true, true, 1, 0.182f, 53, 2, 28, 0.785f, 0.134f,
                   32, -0.803f, 4, 1, -0.290f, 2, -100, -200, true, 182, 0));
  }
  {
    TestCase tc("ANE root hints require clear margin for AB ordering");

    EXPECT(tc, HybridANERootHintMarginClear(0, 0.0f, 0.0f));
    EXPECT(tc, HybridANERootHintMarginClear(1, 0.438f, 0.436f));
    EXPECT(tc, HybridANERootHintMarginClear(2, 0.438f, 0.350f));
    EXPECT(tc, !HybridANERootHintMarginClear(2, 0.234f, 0.000f));
    EXPECT(tc, !HybridANERootHintMarginClear(2, 0.438f, 0.436f));
    EXPECT(tc, !HybridANERootHintMarginClear(2, 0.438f, 0.424f));
    EXPECT(tc, !HybridANERootHintMarginClear(2, 0.528f, 0.487f));
    EXPECT(tc, !HybridANERootHintMarginClear(
                   2, std::numeric_limits<float>::infinity(), 0.424f));
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
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 3,
                   0.778f, 0.517f, 0.530f, 115, 119, 0, -40, 0.171f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 263, 256, 23, 23,
                   0, 1.000f, 0.000f, 0.875f, 361, 361, 0, -1, 0.401f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 82, 79, 46, 46, 0,
                   1.000f, 0.000f, 0.765f, 257, 273, -75, -140, 0.401f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 36, 33, 23, 22, 1,
                   0.957f, 0.446f, 0.446f, 119, 145, -94, -94, 0.444f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 35, 32, 22, 21, 1,
                   0.955f, 0.426f, 0.426f, 112, 125, -105, -105, 0.444f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 49, 46, 31, 30, 1,
                   0.968f, 0.608f, 0.607f, 180, 205, -93, -99, 0.444f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 62, 59, 35, 34, 1,
                   0.971f, 0.691f, 0.666f, 219, 237, -20, -132, 0.444f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 14, 9, 14, 9, 3,
                   0.643f, 0.098f, 0.125f, -11, -11, 0, -1, 0.171f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 16, 15, 16, 15, 1,
                   0.938f, 0.497f, 0.550f, 119, 155, -66, -70, 0.444f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 15, 11, 15, 11,
                   1, 0.733f, 0.746f, 0.746f, 137, 155, -76, -195, 1.022f));
    EXPECT(tc, HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 29, 25, 29, 25,
                   1, 0.862f, 0.938f, 0.938f, 218, 218, 0, -1, 1.022f));
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
                   true, true, true, true, true, true, false, 36, 33, 23, 22, 1,
                   0.957f, 0.399f, 0.446f, 119, 145, -94, -94, 0.444f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 36, 33, 23, 19, 1,
                   0.957f, 0.446f, 0.446f, 119, 145, -94, -94, 0.444f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 36, 33, 23, 22, 2,
                   0.957f, 0.446f, 0.446f, 119, 145, -94, -94, 0.444f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 36, 33, 23, 22, 1,
                   0.957f, 0.446f, 0.446f, 119, 145, -94, -94, 0.399f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 35, 32, 22, 21, 1,
                   0.955f, 0.426f, 0.426f, 112, 119, -105, -105, 0.444f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 35, 32, 22, 21, 1,
                   0.955f, 0.426f, 0.426f, 112, 125, -105, -105, 0.439f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 62, 59, 35, 34, 1,
                   0.971f, 0.691f, 0.666f, 219, 237, -20, -141, 0.444f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 62, 59, 35, 34, 1,
                   0.971f, 0.691f, 0.666f, 209, 237, -20, -132, 0.444f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 29, 25, 29, 25,
                   3, 0.862f, 0.443f, 0.456f, 91, 91, 0, 0, 0.248f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 25, 18, 25, 18, 3,
                   0.778f, 0.517f, 0.530f, 115, 119, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 17, 36, 17, 3,
                   0.778f, 0.517f, 0.530f, 115, 119, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 3,
                   0.689f, 0.517f, 0.530f, 115, 119, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 3,
                   0.778f, 0.339f, 0.530f, 115, 119, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 3,
                   0.778f, 0.517f, 0.359f, 115, 119, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 3,
                   0.778f, 0.517f, 0.530f, 59, 119, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 3,
                   0.778f, 0.517f, 0.530f, 115, 59, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 3,
                   0.778f, 0.517f, 0.530f, 115, 119, 0, -40, 0.149f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 36, 28, 36, 28, 4,
                   0.778f, 0.517f, 0.530f, 115, 119, 0, -40, 0.171f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, true, 14, 9, 14, 9, 3,
                   0.643f, 0.098f, 0.125f, -11, -11, 0, -1, 0.149f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, true, false, 16, 15, 16, 15, 1,
                   0.938f, 0.497f, 0.550f, 119, 99, -66, -70, 0.444f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 15, 11, 15, 11,
                   1, 0.733f, 0.746f, 0.746f, 129, 155, -76, -195, 1.022f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 15, 11, 15, 11,
                   1, 0.733f, 0.746f, 0.746f, 137, 155, -76, -195, 0.999f));
    EXPECT(tc, !HybridPawnOnlyANEMCTSOverride(
                   true, true, true, true, true, false, false, 29, 25, 29, 25,
                   1, 0.862f, 0.938f, 0.938f, 218, 218, 0, -1, 0.749f));
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
    EXPECT(tc, HybridRootPawnLeverCandidate(904, 892, 1255, 7, 11, 1, 0.930f,
                                            0.206f, 0.930f, 0.875f, 0.050f));
    EXPECT(tc, !HybridRootPawnLeverCandidate(904, 892, 1255, 7, 11, 1, 0.930f,
                                             0.206f, 0.930f, 0.869f, 0.050f));
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

    EXPECT(tc, HybridMCTSRootSelectorConfirmsPawnLever(402, 201, 0.179f,
                                                       -0.434f, -530, -538, 3,
                                                       35, -0.621f, 0.217f));
    EXPECT(tc, HybridMCTSRootSelectorConfirmsPawnLever(396, 195, 0.179f,
                                                       -0.432f, -479, -539, 3,
                                                       35, -0.621f, 0.217f));
    EXPECT(tc, !HybridMCTSRootSelectorConfirmsPawnLever(396, 195, 0.179f,
                                                        -0.432f, -479, -540, 3,
                                                        35, -0.621f, 0.217f));
    EXPECT(tc, !HybridMCTSRootSelectorConfirmsPawnLever(402, 201, 0.179f,
                                                        -0.434f, -530, -538, 5,
                                                        35, -0.621f, 0.217f));
    EXPECT(tc, !HybridMCTSRootSelectorConfirmsPawnLever(402, 201, 0.179f,
                                                        -0.434f, -530, -538, 3,
                                                        35, -0.640f, 0.217f));
    EXPECT(tc, !HybridMCTSRootSelectorConfirmsPawnLever(402, 201, 0.179f,
                                                        -0.434f, -530, -538, 3,
                                                        35, -0.621f, 0.199f));

    EXPECT(tc, HybridRootQuietMinorMajorAttackCandidate(
                   102, 117, 3734, -1, 0, 0.000f, 1, 19, 0.404f, 0.135f));
    EXPECT(tc, HybridRootQuietMinorMajorAttackCandidate(
                   124, 73, 4212, 5, 1, -0.005f, 1, 18, 0.371f, 0.135f));
    EXPECT(tc, HybridRootQuietMinorMajorAttackCandidate(
                   167, 163, 1922, 1, 10, 0.357f, 2, 9, 0.118f, 0.135f));
    EXPECT(tc, HybridRootQuietMinorMajorAttackCandidate(
                   118, 115, 2006, 5, 1, -0.005f, 1, 15, 0.274f, 0.135f));
    EXPECT(tc, HybridRootQuietMinorMajorAttackCandidate(
                   123, 69, 7038, 4, 2, 0.021f, 2, 18, 0.371f, 0.135f));
    EXPECT(tc, HybridRootQuietMinorMajorAttackCandidate(
                   96, 138, 3816, 1, 21, 0.355f, 2, 9, 0.118f, 0.135f));
    EXPECT(tc, !HybridRootQuietMinorMajorAttackCandidate(
                   253, 252, 1171, 1, 10, 0.357f, 2, 3, 0.223f, 0.135f));
    EXPECT(tc, !HybridRootQuietMinorMajorAttackCandidate(
                   124, 73, 4212, 5, 1, -0.005f, 1, 18, 0.371f, 0.099f));
    EXPECT(tc, !HybridRootQuietMinorMajorAttackCandidate(
                   124, 63, 4212, 5, 1, -0.005f, 1, 18, 0.371f, 0.135f));
    EXPECT(tc, !HybridRootQuietMinorMajorAttackCandidate(
                   482, 359, 677, 1, 21, 0.355f, 2, 5, 0.144f, 0.135f));
    EXPECT(tc, !HybridRootQuietMinorMajorAttackCandidate(
                   167, 163, 1922, 1, 10, 0.370f, 2, 9, 0.118f, 0.135f));
    EXPECT(tc, !HybridRootQuietMinorMajorAttackCandidate(
                   123, 69, 7038, 4, 3, 0.021f, 2, 18, 0.371f, 0.135f));
    EXPECT(tc, HybridRootQuietAttackTieBreakAllowed(Move(SQ_E3, SQ_F4),
                                                    Move(SQ_H5, SQ_F6)));
    EXPECT(tc, !HybridRootQuietAttackTieBreakAllowed(Move(SQ_H5, SQ_F6),
                                                     Move(SQ_H5, SQ_F6)));
    EXPECT(tc, !HybridRootQuietAttackTieBreakAllowed(Move::none(),
                                                     Move(SQ_H5, SQ_F6)));

    const auto ane_lever =
        [](bool ane_root_probe, int selected_ane_rank, float selected_ane_score,
           int candidate_ane_rank, float candidate_ane_score,
           int selected_average_score, int candidate_average_score,
           uint64_t candidate_effort, int selected_mcts_rank,
           float selected_mcts_q, int candidate_mcts_rank,
           uint32_t candidate_mcts_current_visits, float candidate_mcts_q,
           float candidate_mcts_policy) {
          return HybridANERootPawnLeverCandidate(
              ane_root_probe, selected_ane_rank, selected_ane_score,
              candidate_ane_rank, candidate_ane_score, selected_average_score,
              candidate_average_score, candidate_effort, selected_mcts_rank,
              selected_mcts_q, 0.314f, selected_mcts_q, candidate_mcts_rank,
              candidate_mcts_current_visits, candidate_mcts_q,
              candidate_mcts_policy);
        };

    EXPECT(tc, ane_lever(true, 4, -0.297f, 2, -0.271f, -258, -282, 214, 1,
                         -0.181f, 5, 2, -0.259f, 0.038f));
    EXPECT(tc, ane_lever(true, 4, -0.297f, 2, -0.271f, -246, -267, 83, 1,
                         -0.189f, 5, 3, -0.270f, 0.038f));
    EXPECT(tc, ane_lever(true, 4, -0.297f, 2, -0.271f, -256, -300, 4207, 1,
                         -0.189f, 5, 3, -0.270f, 0.038f));
    EXPECT(tc, !ane_lever(false, 4, -0.297f, 2, -0.271f, -258, -282, 214, 1,
                          -0.181f, 5, 2, -0.259f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 3, -0.271f, -258, -282, 214, 1,
                          -0.181f, 5, 2, -0.259f, 0.038f));
    EXPECT(tc, !ane_lever(true, 2, -0.297f, 4, -0.271f, -258, -282, 214, 1,
                          -0.181f, 5, 2, -0.259f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.283f, -258, -282, 214, 1,
                          -0.181f, 5, 2, -0.259f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.271f, -258, -282, 214, 1,
                          -0.181f, 5, 1, -0.259f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.271f, -258, -282, 214, 1,
                          -0.181f, 5, 2, -0.259f, 0.034f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.271f, -258, -309, 214, 1,
                          -0.181f, 5, 2, -0.259f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.271f, -258, -282, 74, 1,
                          -0.181f, 5, 2, -0.259f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.271f, -246, -267, 74, 1,
                          -0.189f, 5, 3, -0.270f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.271f, -256, -307, 4207, 1,
                          -0.189f, 5, 3, -0.270f, 0.038f));
    EXPECT(tc, !ane_lever(true, 4, -0.297f, 2, -0.271f, -258, -282, 214, 1,
                          -0.181f, 5, 2, -0.272f, 0.038f));

    EXPECT(tc, HybridANERootPawnLeverCandidate(
                   true, 7, -0.602f, 1, -0.520f, -496, -531, 83268, 6, -0.491f,
                   0.048f, -0.440f, 3, 27, -0.629f, 0.217f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(
                   true, 7, -0.602f, 1, -0.545f, -496, -531, 83268, 6, -0.491f,
                   0.048f, -0.440f, 3, 27, -0.629f, 0.217f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(
                   true, 7, -0.602f, 1, -0.520f, -496, -531, 83268, 1, -0.491f,
                   0.048f, -0.440f, 3, 27, -0.629f, 0.217f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(
                   true, 7, -0.602f, 1, -0.520f, -496, -531, 83268, 6, -0.491f,
                   0.080f, -0.440f, 3, 27, -0.629f, 0.217f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(
                   true, 7, -0.602f, 1, -0.520f, -496, -531, 83268, 6, -0.491f,
                   0.048f, -0.440f, 3, 23, -0.629f, 0.217f));
    EXPECT(tc, !HybridANERootPawnLeverCandidate(
                   true, 7, -0.602f, 1, -0.520f, -496, -537, 83268, 6, -0.491f,
                   0.048f, -0.440f, 3, 27, -0.629f, 0.217f));

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

    pos.set("r3r3/pp3pk1/3p1q1p/1b3P2/5R2/P3B2Q/2p2KPP/R7 w - - 0 29", false,
            &st);
    EXPECT(tc, HybridIsQuietMajorCheck(pos, UCIEngine::to_move(pos, "f4g4")));
    EXPECT(tc, HybridIsQuietMajorCheck(pos, UCIEngine::to_move(pos, "h3g4")));
    EXPECT(tc, !HybridIsQuietMajorCheck(pos, UCIEngine::to_move(pos, "e3d4")));
    EXPECT(tc, !HybridIsQuietMajorCheck(pos, UCIEngine::to_move(pos, "f4f6")));

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

    pos.set("2b2k2/2q2p2/p1p2Ppr/4p3/2r1B3/5Q2/P1P4P/R2R3K w - - 0 25", false,
            &st);
    EXPECT(tc, !HybridIsPawnOnlyEndgame(pos));
    EXPECT(tc, HybridANEProbeAllowedForPosition(pos, true));
    EXPECT(tc, HybridANEProbeAllowedForPosition(pos, false));

    pos.set("8/8/5pk1/8/4p1pR/2K4p/1P6/8 b - - 1 54", false, &st);
    EXPECT(tc, !HybridIsPawnOnlyEndgame(pos));
    EXPECT(tc, HybridIsRookEndgame(pos));
    EXPECT(tc, !HybridANEProbeAllowedForPosition(pos, true));
    EXPECT(tc, HybridANEProbeAllowedForPosition(pos, false));
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
    EXPECT(tc, HybridMCTSCrossRootConfidenceOverride(
                   true, true, 306, 198, 0.647f, 0.167f, 223, 53, 634, 2,
                   -32001, 611, 1019286, 2, 32, 0.421f, 0.632f));
    EXPECT(tc, HybridMCTSCrossRootConfidenceOverride(
                   true, true, 418, 276, 0.660f, 0.127f, 216, 45, 636, 2,
                   -32001, 619, 1436766, 3, 32, 0.421f, 0.619f));
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
    EXPECT(tc, !HybridMCTSCrossRootConfidenceOverride(
                   true, true, 418, 276, 0.660f, 0.127f, 216, 45, 636, 2,
                   -32001, 619, 1436766, 3, 32, 0.445f, 0.619f));
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
    EXPECT(tc, HybridMCTSRootConfidenceRejectOverride(
                   true, true, 10, -32001, false, false, 7349, 107, 38, 4, 7,
                   -0.039f, 0.880f));
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
                   true, true, 11, -32001, false, false, 7349, 107, 38, 4, 7,
                   -0.039f, 0.880f));
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
                   true, true, 10, -32001, false, false, 7349, 129, 38, 4, 7,
                   -0.039f, 0.880f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 3292, -232, -352, 2, 9,
                   -0.349f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 10, -32001, false, false, 7349, 107, 38, 5, 7,
                   -0.039f, 0.880f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 10, -32001, false, false, 7349, 107, 38, 4, 17,
                   -0.039f, 0.880f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 2, -32001, false, false, 3292, -232, -352, 2, 24,
                   0.130f, 0.717f));
    EXPECT(tc, !HybridMCTSRootConfidenceRejectOverride(
                   true, true, 10, -32001, false, false, 7349, 107, 38, 4, 7,
                   0.140f, 0.880f));
  }
  {
    TestCase tc("Mid-root tactical Q gap can bypass AB rejection");

    EXPECT(tc, HybridMCTSMidRootTacticalQGapOverride(
                   true, true, true, 141, 82, 0.582f, 0.869f, 187, 229, 2,
                   -32001, false, false, 0, 2, 48, -0.315f, 0.555f));
    EXPECT(tc, HybridMCTSMidRootTacticalQGapOverride(
                   true, true, true, 147, 88, 0.599f, 0.894f, 198, 253, 2,
                   -32001, false, false, 1426, 2, 48, -0.315f, 0.580f));
    EXPECT(tc, !HybridMCTSMidRootTacticalQGapOverride(
                   true, true, false, 141, 82, 0.582f, 0.869f, 187, 229, 2,
                   -32001, false, false, 0, 2, 48, -0.315f, 0.555f));
    EXPECT(tc, !HybridMCTSMidRootTacticalQGapOverride(
                   true, true, true, 141, 82, 0.54f, 0.869f, 187, 229, 2,
                   -32001, false, false, 0, 2, 48, -0.315f, 0.555f));
    EXPECT(tc, !HybridMCTSMidRootTacticalQGapOverride(
                   true, true, true, 141, 82, 0.582f, 0.79f, 187, 229, 2,
                   -32001, false, false, 0, 2, 48, -0.315f, 0.555f));
    EXPECT(tc, !HybridMCTSMidRootTacticalQGapOverride(
                   true, true, true, 141, 82, 0.582f, 0.869f, 187, 229, 2,
                   -32001, false, false, 7000, 2, 48, -0.315f, 0.555f));
    EXPECT(tc, !HybridMCTSMidRootTacticalQGapOverride(
                   true, true, true, 141, 82, 0.582f, 0.869f, 187, 229, 2,
                   -32001, false, false, 0, 2, 24, -0.315f, 0.555f));
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
    EXPECT(tc, HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.673f, 461,
                   263, 6, -32001, 5, 3, 2, 0.178f, 0.912f));
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
                   true, true, true, true, true, 29, 31, 0.861f, 0.673f, 461,
                   263, 6, -32001, 5, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 39, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 27, 0.861f, 0.673f, 461,
                   263, 6, -32001, 5, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.879f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.839f, 0.673f, 461,
                   263, 6, -32001, 5, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.590f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.659f, 461,
                   263, 6, -32001, 5, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 429,
                   366, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.673f, 439,
                   263, 6, -32001, 5, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   299, 3, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.673f, 461,
                   239, 6, -32001, 5, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 1, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 7, -32001, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.673f, 461,
                   263, 7, -32001, 5, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, 388, 2429, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 20001, 2, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.673f, 461,
                   263, 6, -32001, 65, 3, 2, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 1, 3, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 0, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.673f, 461,
                   263, 6, -32001, 5, 3, 3, 0.178f, 0.912f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 33, 0.169f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 53, 47, 0.887f, 0.677f, 468,
                   366, 3, -32001, 2429, 2, 3, 0.330f, 0.916f));
    EXPECT(tc, !HybridMCTSRootRejectRookEndgamePawnPushOverride(
                   true, true, true, true, true, 36, 31, 0.861f, 0.673f, 461,
                   263, 6, -32001, 5, 3, 2, 0.270f, 0.912f));
  }
  {
    TestCase tc("Rook endgame central pawn push lower bound stays narrow");

    EXPECT(tc, HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 57, 46, 0.807f, 0.583f, 196, 166, 30,
                   2, 114, true, false, 277616, 2, 5, -0.007f, 0.576f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   false, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, false, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, false, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 23, 18, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.699f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.549f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 179, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 159, 0,
                   2, -20, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, false, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, true, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -51, true, false, 58884, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 49999, 2, 4, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 3, 0.049f, 0.615f));
    EXPECT(tc, !HybridMCTSRookEndgamePawnPushLowerBoundOverride(
                   true, true, true, true, 32, 24, 0.750f, 0.565f, 214, 214, 0,
                   2, -20, true, false, 58884, 2, 4, 0.116f, 0.615f));
  }
  {
    TestCase tc("Rook endgame quiet rook MCTS override stays narrow");

    EXPECT(tc, HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 12, 12, 12, 12, 1.000f, 1.585f, 321,
                   369, 5, -VALUE_INFINITE, 14, 2, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   false, true, true, true, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, false, true, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, false, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 11, 11, 11, 11, 1.000f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 0.970f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 1.000f, 1.540f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 1.000f, 1.658f, 299,
                   380, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   299, 6, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   380, 3, -VALUE_INFINITE, 27, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 101, 3, 0));
    EXPECT(tc, !HybridMCTSRookEndgameQuietRookOverride(
                   true, true, true, true, 17, 17, 17, 17, 1.000f, 1.658f, 354,
                   380, 6, -VALUE_INFINITE, 27, 3, 1));
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
    EXPECT(tc, HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 14, 11, 0.786f, 1.325f, 254, 291, 7,
                   -32001, false, false, 25, 6, 0, -0.338f, 0.691f));
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
                   true, true, true, true, 6, 5, 0.833f, 1.216f, 199, 243, 5,
                   -32001, false, true, 97, 6, 0, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.690f, 1.216f, 199, 243, 5,
                   -32001, false, true, 97, 6, 0, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.727f, 1.216f, 189, 243, 5,
                   -32001, false, true, 97, 6, 0, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.727f, 1.216f, 199, 229, 5,
                   -32001, false, true, 97, 6, 0, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.727f, 1.216f, 199, 243, 8,
                   -32001, false, true, 97, 6, 0, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.727f, 1.216f, 199, 243, 5,
                   -32001, false, true, 513, 6, 0, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.727f, 1.216f, 199, 243, 5,
                   -32001, false, true, 97, 4, 0, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.727f, 1.216f, 199, 243, 5,
                   -32001, false, true, 97, 6, 1, -0.177f, 0.581f));
    EXPECT(tc, !HybridMCTSRootRejectQuietQueenMoveOverride(
                   true, true, true, true, 11, 8, 0.727f, 1.216f, 199, 243, 5,
                   -32001, false, true, 97, 6, 0, -0.100f, 0.581f));
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
    TestCase tc(
        "Quiet minor major attack can bypass stale AB root reject narrowly");

    EXPECT(tc,
           HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 63, 35, 0.556f, 0.210f, 191, 107,
               5, -VALUE_INFINITE, false, true, 2288, 7, 0, -0.446f, 0.563f));
    EXPECT(tc,
           HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 58, 30, 0.517f, 0.197f, 185, 78, 6,
               -VALUE_INFINITE, false, true, 2088, 2, 22, 0.353f, 0.550f));
    EXPECT(tc,
           HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 60, 32, 0.533f, 0.224f, 197, 164,
               8, -VALUE_INFINITE, false, false, 5627, 5, 2, 0.021f, 0.577f));
    EXPECT(tc,
           HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 66, 37, 0.561f, 0.234f, 201, 180,
               11, -VALUE_INFINITE, false, false, 7117, 5, 2, 0.021f, 0.586f));
    EXPECT(tc,
           HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 66, 37, 0.561f, 0.234f, 201, 179,
               6, -VALUE_INFINITE, false, false, 9972, 2, 22, 0.353f, 0.586f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               false, true, true, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, false, true, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, false, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, false, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, false, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 54, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 29, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.499f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.189f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.244f, 179, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.244f, 206, 74, 7,
               -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               4, -VALUE_INFINITE, false, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, true, true, 6468, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 10001, 2, 22, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 67, 38, 0.567f, 0.244f, 206, 107,
               7, -VALUE_INFINITE, false, true, 6468, 2, 25, 0.353f, 0.597f));
    EXPECT(tc,
           !HybridMCTSRootRejectQuietMinorMajorAttackOverride(
               true, true, true, true, true, 63, 35, 0.556f, 0.210f, 191, 107,
               5, -VALUE_INFINITE, false, true, 2288, 7, 5, -0.446f, 0.563f));
  }
  {
    TestCase tc("Reused root MCTS confidence stays tightly gated");

    EXPECT(tc, HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, HybridMCTSReusedRootConfidenceOverride(
                   true, 933, 914, 29, 29, 0.980f, 1.000f, 430, 445, -59, -110,
                   2, -32001, 120884, 2, 0, -0.027f, 0.893f));
    EXPECT(tc, HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 1.000f, 429, 452, -92, -119,
                   2, -32001, 97079, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, HybridMCTSReusedRootConfidenceOverride(
                   true, 440, 425, 22, 22, 0.966f, 1.000f, 400, 430, -91, -120,
                   2, -32001, 103291, 2, 0, -0.033f, 0.870f));
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
                   true, 657, 640, 17, 17, 0.974f, 1.000f, 429, 452, -92, -119,
                   2, -32001, 97079, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 399, 425, 22, 22, 0.966f, 1.000f, 400, 430, -91, -120,
                   2, -32001, 103291, 2, 0, -0.033f, 0.870f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 440, 425, 19, 19, 0.966f, 1.000f, 400, 430, -91, -120,
                   2, -32001, 103291, 2, 0, -0.033f, 0.870f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.899f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.959f, 1.000f, 429, 452, -92, -119,
                   2, -32001, 97079, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.899f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 0.959f, 429, 452, -92, -119,
                   2, -32001, 97079, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 299, 378, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 1.000f, 399, 452, -92, -119,
                   2, -32001, 97079, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 440, 425, 22, 22, 0.966f, 1.000f, 399, 430, -91, -120,
                   2, -32001, 103291, 2, 0, -0.033f, 0.870f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 299, -83, -77,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 1.000f, 429, 399, -92, -119,
                   2, -32001, 97079, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -50, -130,
                   2, -32001, 129846, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 1.000f, 429, 452, -43, -119,
                   2, -32001, 97079, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 250001, 2, 1, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 1.000f, 429, 452, -92, -119,
                   2, -32001, 250001, 2, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 440, 425, 22, 22, 0.966f, 1.000f, 400, 430, -91, -120,
                   2, -32001, 150001, 2, 0, -0.033f, 0.870f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 3, -0.042f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 1.000f, 429, 452, -92, -119,
                   2, -32001, 97079, 3, 0, -0.027f, 0.892f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 241, 234, 65, 63, 0.971f, 0.969f, 354, 378, -83, -77,
                   2, -32001, 129846, 2, 1, 0.090f, 0.828f));
    EXPECT(tc, !HybridMCTSReusedRootConfidenceOverride(
                   true, 657, 640, 23, 23, 0.974f, 1.000f, 429, 452, -92, -119,
                   2, -32001, 97079, 2, 0, 0.050f, 0.892f));
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
    EXPECT(tc, HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 732, 714, 30, 29, 0.975f, 0.967f,
                   0.918f, 423, 426, 2, -13, true, false, 596724, 2, 1, -0.030f,
                   0.888f));
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
                   true, true, false, false, 732, 714, 30, 29, 0.949f, 0.967f,
                   0.918f, 423, 426, 2, -13, true, false, 596724, 2, 1, -0.030f,
                   0.888f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 0.919f,
                   0.000f, 323, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 732, 714, 30, 29, 0.975f, 0.949f,
                   0.918f, 423, 426, 2, -13, true, false, 596724, 2, 1, -0.030f,
                   0.888f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 299, 318, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 732, 714, 30, 29, 0.975f, 0.967f,
                   0.918f, 399, 426, 2, -13, true, false, 596724, 2, 1, -0.030f,
                   0.888f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 414, 354, 59, 59, 0.855f, 1.000f,
                   0.000f, 323, 299, 2, 1070, true, false, 1501742, 2, 0,
                   0.311f, 0.793f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 732, 714, 30, 29, 0.975f, 0.967f,
                   0.918f, 423, 399, 2, -13, true, false, 596724, 2, 1, -0.030f,
                   0.888f));
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
                   true, true, false, false, 732, 714, 30, 29, 0.975f, 0.967f,
                   0.918f, 423, 426, 2, -51, true, false, 596724, 2, 1, -0.030f,
                   0.888f));
    EXPECT(tc, !HybridMCTSReusedRootCurrentOverride(
                   true, true, false, false, 732, 714, 30, 29, 0.975f, 0.967f,
                   0.918f, 423, 426, 2, -13, true, false, 596724, 2, 1, 0.050f,
                   0.888f));
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
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 356, 296, 39, 39, 0.831f, 1.000f,
               315, 314, 4, -32001, false, false, 13, 2, 0, 0.309f, 0.782f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 908, 832, 24, 23, 0.916f, 0.958f,
               378, 378, 3, -32001, false, false, 3924, 3, 0, 0.244f, 0.852f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 961, 884, 27, 27, 0.920f, 1.000f,
               386, 383, 5, -32001, false, false, 91, 3, 0, 0.309f, 0.859f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 1044, 966, 32, 31, 0.925f, 0.969f,
               392, 390, 2, 188, true, false, 174884, 2, 1, 0.229f, 0.864f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 86, 31, 25, 20, 0.360f, 0.800f,
               200, 191, 3, -32001, false, false, 47, 3, 2, 0.258f, 0.584f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 117, 62, 17, 17, 0.530f, 1.000f,
               226, 226, 3, -32001, false, false, 274, 4, 0, 0.012f, 0.638f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 107, 48, 68, 40, 0.449f, 0.588f,
               211, 207, 3, -32001, false, false, 6, 2, 12, 0.309f, 0.607f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 104, 45, 60, 36, 0.433f, 0.600f,
               216, 215, 4, -32001, false, false, 102, 3, 11, 0.242f, 0.617f));
    EXPECT(tc,
           HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 100, 41, 55, 32, 0.410f, 0.582f,
               225, 218, 2, 150, true, false, 234308, 3, 11, 0.242f, 0.635f));
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
               true, true, false, true, true, 1044, 966, 32, 31, 0.925f, 0.969f,
               392, 390, 2, 149, true, false, 174884, 2, 1, 0.229f, 0.864f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, true, true, true, 622, 556, 38, 37, 0.894f, 0.974f,
               353, 219, 2, 475, true, false, 579529, 3, 0, 0.243f, 0.827f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 79, 31, 25, 20, 0.360f, 0.800f,
               200, 191, 3, -32001, false, false, 47, 3, 2, 0.258f, 0.584f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 86, 31, 25, 11, 0.360f, 0.440f,
               200, 191, 3, -32001, false, false, 47, 3, 2, 0.258f, 0.584f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 86, 31, 25, 20, 0.360f, 0.800f,
               189, 191, 3, -32001, false, false, 47, 3, 2, 0.258f, 0.584f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 86, 31, 25, 20, 0.360f, 0.800f,
               200, 191, 3, -32001, false, false, 47, 2, 2, 0.258f, 0.584f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 86, 31, 25, 20, 0.360f, 0.800f,
               200, 191, 3, -32001, false, false, 47, 3, 13, 0.258f, 0.584f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 107, 48, 68, 40, 0.429f, 0.588f,
               211, 207, 3, -32001, false, false, 6, 2, 12, 0.309f, 0.607f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 107, 48, 68, 40, 0.449f, 0.579f,
               211, 207, 3, -32001, false, false, 6, 2, 12, 0.309f, 0.607f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 107, 48, 68, 40, 0.449f, 0.588f,
               211, 207, 3, -32001, false, false, 6, 2, 13, 0.309f, 0.607f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, true, true, true, 107, 48, 68, 40, 0.449f, 0.588f,
               211, 207, 3, -32001, false, false, 6, 2, 12, 0.309f, 0.607f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 100, 41, 55, 32, 0.410f, 0.582f,
               219, 218, 2, 150, true, false, 234308, 3, 11, 0.242f, 0.635f));
    EXPECT(tc,
           !HybridMCTSBishopEndgameRetreatOverride(
               true, true, false, true, true, 100, 41, 55, 32, 0.410f, 0.582f,
               225, 209, 2, 150, true, false, 234308, 3, 11, 0.242f, 0.635f));
  }
  {
    TestCase tc("Pawn-only MCTS override predicate");

    EXPECT(tc, HybridPawnOnlyMCTSOverride(true, true, true, true, false, 31, 29,
                                          19, 19, 0, 0.935f, 0.000f, 0.449f,
                                          111, 133, -1, -1, 2, -VALUE_INFINITE,
                                          false, 62385, 2));
    EXPECT(tc, HybridPawnOnlyMCTSOverride(true, true, true, true, false, 20, 18,
                                          14, 13, 1, 0.900f, 0.463f, 0.464f,
                                          116, 130, -1, -1, 2, -VALUE_INFINITE,
                                          true, 22922, 2));
    EXPECT(tc, HybridPawnOnlyMCTSOverride(true, true, true, true, false, 25, 22,
                                          18, 16, 2, 0.880f, 0.463f, 0.464f,
                                          124, 146, -1, -1, 2, -VALUE_INFINITE,
                                          true, 45755, 2));
    EXPECT(tc, HybridPawnOnlyMCTSOverride(true, true, true, true, false, 45, 42,
                                          31, 30, 1, 0.968f, 0.565f, 0.565f,
                                          163, 163, 0, 0, 2, -VALUE_INFINITE,
                                          true, 78414, 2));
    EXPECT(tc, HybridPawnOnlyMCTSOverride(true, true, true, false, true, 22, 15,
                                          22, 15, 3, 0.682f, 0.226f, 0.239f, 23,
                                          23, -46, -46, 2, -VALUE_INFINITE,
                                          true, 84043, 3));
    EXPECT(tc, HybridPawnOnlyMCTSOverride(true, true, true, false, true, 37, 29,
                                          37, 29, 3, 0.784f, 0.538f, 0.551f,
                                          123, 123, 0, 0, 2, -VALUE_INFINITE,
                                          true, 97305, 3));
    EXPECT(tc, HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23, 23,
                                          13, 13, 0, 1.000f, 0.000f, 1.078f,
                                          724, 657, 210, 30, 2, -VALUE_INFINITE,
                                          false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 12, 7,
                                           12, 7, 2, 0.583f, 0.092f, 0.100f,
                                           -12, 0, -45, -45, 2, -VALUE_INFINITE,
                                           true, 41775, 3));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23,
                                           23, 11, 11, 0, 1.000f, 0.000f,
                                           1.078f, 724, 657, 210, 30, 2,
                                           -VALUE_INFINITE, false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23,
                                           23, 13, 12, 0, 0.923f, 0.000f,
                                           1.078f, 724, 657, 210, 30, 2,
                                           -VALUE_INFINITE, false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23,
                                           23, 13, 13, 0, 1.000f, 0.001f,
                                           1.078f, 724, 657, 210, 30, 2,
                                           -VALUE_INFINITE, false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23,
                                           23, 13, 13, 0, 1.000f, 0.000f,
                                           0.999f, 724, 657, 210, 30, 2,
                                           -VALUE_INFINITE, false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23,
                                           23, 13, 13, 0, 1.000f, 0.000f,
                                           1.078f, 649, 657, 210, 30, 2,
                                           -VALUE_INFINITE, false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23,
                                           23, 13, 13, 0, 1.000f, 0.000f,
                                           1.078f, 724, 599, 210, 30, 2,
                                           -VALUE_INFINITE, false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 23,
                                           23, 13, 13, 1, 1.000f, 0.000f,
                                           1.078f, 724, 657, 210, 30, 2,
                                           -VALUE_INFINITE, false, 1373, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, true, false, 31,
                                           29, 19, 19, 2, 0.935f, 0.000f,
                                           0.400f, 111, 133, -1, -1, 2,
                                           -VALUE_INFINITE, false, 62385, 2));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 22,
                                           15, 22, 15, 3, 0.682f, 0.226f,
                                           0.239f, 23, 23, -46, -46, 2,
                                           -VALUE_INFINITE, false, 84043, 3));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, true, true, false, true, 37,
                                           29, 37, 29, 3, 0.784f, 0.538f,
                                           0.551f, 123, 123, 0, 0, 2,
                                           -VALUE_INFINITE, false, 97305, 3));
    EXPECT(tc, !HybridPawnOnlyMCTSOverride(true, false, true, true, false, 31,
                                           29, 19, 19, 0, 0.935f, 0.000f,
                                           0.449f, 111, 133, -1, -1, 2,
                                           -VALUE_INFINITE, false, 62385, 2));
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
    TestCase tc("Compact pawn endgame MCTS override stays narrowly gated");

    EXPECT(tc,
           HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 36, 32, 0.889f, 0.959f, 229, 229, -2,
               -30, 2, -VALUE_INFINITE, false, 328718, 2, 1, -0.316f, 0.643f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               false, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, false, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, false, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, true, 47, 43, 0.915f, 1.021f, 263, 263, -1, -1,
               2, -VALUE_INFINITE, false, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 34, 32, 0.889f, 0.959f, 229, 229, -2,
               -30, 2, -VALUE_INFINITE, false, 328718, 2, 1, -0.316f, 0.643f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 36, 31, 0.889f, 0.959f, 229, 229, -2,
               -30, 2, -VALUE_INFINITE, false, 328718, 2, 1, -0.316f, 0.643f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 36, 32, 0.879f, 0.959f, 229, 229, -2,
               -30, 2, -VALUE_INFINITE, false, 328718, 2, 1, -0.316f, 0.643f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 36, 32, 0.889f, 0.899f, 229, 229, -2,
               -30, 2, -VALUE_INFINITE, false, 328718, 2, 1, -0.316f, 0.643f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 36, 32, 0.889f, 0.959f, 219, 229, -2,
               -30, 2, -VALUE_INFINITE, false, 328718, 2, 1, -0.316f, 0.643f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 36, 32, 0.889f, 0.959f, 229, 219, -2,
               -30, 2, -VALUE_INFINITE, false, 328718, 2, 1, -0.316f, 0.643f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, 100,
               -1, 2, -VALUE_INFINITE, false, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 1, -VALUE_INFINITE, false, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, true, 141849, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 99999, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 600001, 2, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 141849, 3, 1, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 141849, 2, 3, -0.316f, 0.705f));
    EXPECT(tc,
           !HybridMCTSCompactPawnEndgameOverride(
               true, true, true, false, 47, 43, 0.915f, 1.021f, 263, 263, -1,
               -1, 2, -VALUE_INFINITE, false, 141849, 2, 1, -0.190f, 0.705f));
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

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
    EXPECT(tc, config.mcts_root_reject);
    EXPECT(tc, config.mcts_ab_root_hints);
    EXPECT(tc, config.mcts_ab_root_hint_delay_ms == 25);
    EXPECT(tc, config.mcts_ab_root_hint_count == 8);
    EXPECT(tc, config.ab_candidate_verify_ms == 120);
    EXPECT(tc, config.ab_candidate_verify_count == 4);
    EXPECT(tc, config.root_pawn_lever_tiebreak);
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

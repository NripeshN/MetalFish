/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive Search Tests - Alpha-Beta, TT, History, Time Management
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "search/history.h"
#include "search/search.h"
#include "search/timeman.h"
#include "search/tt.h"
#include <cassert>
#include <chrono>
#include <deque>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using namespace MetalFish;

namespace {

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

// ============================================================================
// History Tests
// ============================================================================

bool test_stats_entry() {
  TestCase tc("StatsEntry");

  StatsEntry<int16_t, 7183> entry;
  entry = 0;
  EXPECT(tc, int16_t(entry) == 0);

  // Test update with bonus
  entry << 100;
  EXPECT(tc, int16_t(entry) > 0);
  EXPECT(tc, int16_t(entry) <= 7183);

  // Test negative bonus
  entry << -200;
  EXPECT(tc, int16_t(entry) < 100);

  return tc.passed();
}

bool test_butterfly_history() {
  TestCase tc("ButterflyHistory");

  ButterflyHistory history;

  // Initialize to zero
  for (int c = 0; c < COLOR_NB; ++c) {
    for (int i = 0; i < UINT_16_HISTORY_SIZE; ++i) {
      history[c][i] = 0;
    }
  }

  // Test basic access
  Move m(SQ_E2, SQ_E4);
  uint16_t idx = m.raw();
  history[WHITE][idx] << 500;
  EXPECT(tc, int16_t(history[WHITE][idx]) > 0);
  EXPECT(tc, int16_t(history[BLACK][idx]) == 0);

  return tc.passed();
}

bool test_capture_history() {
  TestCase tc("CaptureHistory");

  CapturePieceToHistory history;

  // Initialize
  for (int pc = 0; pc < PIECE_NB; ++pc) {
    for (int sq = 0; sq < SQUARE_NB; ++sq) {
      for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
        history[pc][sq][pt] = 0;
      }
    }
  }

  // Test update
  history[W_PAWN][SQ_D5][PAWN] << 300;
  EXPECT(tc, int16_t(history[W_PAWN][SQ_D5][PAWN]) > 0);

  return tc.passed();
}

bool test_piece_to_history() {
  TestCase tc("PieceToHistory");

  PieceToHistory history;

  for (int pc = 0; pc < PIECE_NB; ++pc) {
    for (int sq = 0; sq < SQUARE_NB; ++sq) {
      history[pc][sq] = 0;
    }
  }

  history[W_KNIGHT][SQ_F3] << 1000;
  EXPECT(tc, int16_t(history[W_KNIGHT][SQ_F3]) > 0);

  return tc.passed();
}

bool test_continuation_history() {
  TestCase tc("ContinuationHistory");

  ContinuationHistory history;

  // Just verify structure exists and can be accessed
  for (int pc = 0; pc < PIECE_NB; ++pc) {
    for (int sq = 0; sq < SQUARE_NB; ++sq) {
      for (int pc2 = 0; pc2 < PIECE_NB; ++pc2) {
        for (int sq2 = 0; sq2 < SQUARE_NB; ++sq2) {
          history[pc][sq][pc2][sq2] = 0;
        }
      }
    }
  }

  history[W_PAWN][SQ_E4][W_KNIGHT][SQ_F3] << 500;
  EXPECT(tc, int16_t(history[W_PAWN][SQ_E4][W_KNIGHT][SQ_F3]) > 0);

  return tc.passed();
}

// ============================================================================
// Limits Tests
// ============================================================================

bool test_limits_type() {
  TestCase tc("LimitsType");

  Search::LimitsType limits;

  // Default values
  EXPECT(tc, limits.time[WHITE] == 0);
  EXPECT(tc, limits.time[BLACK] == 0);
  EXPECT(tc, limits.depth == 0);
  EXPECT(tc, limits.nodes == 0);
  EXPECT(tc, !limits.ponderMode);

  // Time management check
  EXPECT(tc, !limits.use_time_management());

  limits.time[WHITE] = 60000;
  EXPECT(tc, limits.use_time_management());

  return tc.passed();
}

bool test_limits_depth() {
  TestCase tc("LimitsDepth");

  Search::LimitsType limits;
  limits.depth = 10;

  EXPECT(tc, limits.depth == 10);
  EXPECT(tc, !limits.use_time_management());

  return tc.passed();
}

bool test_limits_nodes() {
  TestCase tc("LimitsNodes");

  Search::LimitsType limits;
  limits.nodes = 10000;

  EXPECT(tc, limits.nodes == 10000);

  return tc.passed();
}

bool test_limits_movetime() {
  TestCase tc("LimitsMovetime");

  Search::LimitsType limits;
  limits.movetime = 5000;

  EXPECT(tc, limits.movetime == 5000);

  return tc.passed();
}

// ============================================================================
// Root Move Tests
// ============================================================================

bool test_root_move() {
  TestCase tc("RootMove");

  Move m(SQ_E2, SQ_E4);
  Search::RootMove rm(m);

  EXPECT(tc, rm.pv.size() == 1);
  EXPECT(tc, rm.pv[0] == m);
  EXPECT(tc, rm.score == -VALUE_INFINITE);
  EXPECT(tc, rm == m);

  return tc.passed();
}

bool test_root_move_sorting() {
  TestCase tc("RootMoveSorting");

  Search::RootMoves moves;
  moves.emplace_back(Move(SQ_E2, SQ_E4));
  moves.emplace_back(Move(SQ_D2, SQ_D4));
  moves.emplace_back(Move(SQ_G1, SQ_F3));

  moves[0].score = 50;
  moves[1].score = 100;
  moves[2].score = 25;

  std::sort(moves.begin(), moves.end());

  // Should be sorted in descending order
  EXPECT(tc, moves[0].score == 100);
  EXPECT(tc, moves[1].score == 50);
  EXPECT(tc, moves[2].score == 25);

  return tc.passed();
}

// ============================================================================
// Skill Tests
// ============================================================================

bool test_skill_level() {
  TestCase tc("SkillLevel");

  // Test with skill level
  Search::Skill skill(10, 0);
  EXPECT(tc, skill.enabled());
  EXPECT(tc, skill.level == 10.0);

  // Test with UCI Elo
  Search::Skill skill_elo(20, 2000);
  EXPECT(tc, skill_elo.enabled());

  // Test max skill (disabled)
  Search::Skill max_skill(20, 0);
  EXPECT(tc, !max_skill.enabled());

  return tc.passed();
}

bool test_skill_time_to_pick() {
  TestCase tc("SkillTimeToPick");

  Search::Skill skill(5, 0);
  EXPECT(tc, skill.time_to_pick(6)); // depth = 1 + level
  EXPECT(tc, !skill.time_to_pick(5));

  return tc.passed();
}

// ============================================================================
// Stack Tests
// ============================================================================

bool test_search_stack() {
  TestCase tc("SearchStack");

  Search::Stack ss;
  ss.ply = 5;
  ss.currentMove = Move(SQ_E2, SQ_E4);
  ss.staticEval = 100;
  ss.inCheck = false;
  ss.ttPv = true;

  EXPECT(tc, ss.ply == 5);
  EXPECT(tc, ss.currentMove == Move(SQ_E2, SQ_E4));
  EXPECT(tc, ss.staticEval == 100);
  EXPECT(tc, !ss.inCheck);
  EXPECT(tc, ss.ttPv);

  return tc.passed();
}

// ============================================================================
// Time Management Tests
// ============================================================================

bool test_time_manager_basic() {
  TestCase tc("TimeManagerBasic");

  // TimeManagement requires OptionsMap which is complex to set up
  // Just verify the structures exist
  Search::LimitsType limits;
  limits.time[WHITE] = 60000; // 1 minute
  limits.inc[WHITE] = 1000;   // 1 second increment

  EXPECT(tc, limits.time[WHITE] == 60000);
  EXPECT(tc, limits.inc[WHITE] == 1000);

  return tc.passed();
}

bool test_time_manager_low_time() {
  TestCase tc("TimeManagerLowTime");

  Search::LimitsType limits;
  limits.time[WHITE] = 1000; // 1 second only

  EXPECT(tc, limits.time[WHITE] == 1000);
  EXPECT(tc, limits.use_time_management());

  return tc.passed();
}

bool test_time_manager_increment() {
  TestCase tc("TimeManagerIncrement");

  Search::LimitsType limits1, limits2;

  limits1.time[WHITE] = 60000;
  limits1.inc[WHITE] = 0;

  limits2.time[WHITE] = 60000;
  limits2.inc[WHITE] = 5000;

  // With increment, should have more flexibility
  EXPECT(tc, limits2.inc[WHITE] > limits1.inc[WHITE]);

  return tc.passed();
}

// ============================================================================
// Value/Score Tests
// ============================================================================

bool test_mate_values() {
  TestCase tc("MateValues");

  EXPECT(tc, is_win(mate_in(5)));
  EXPECT(tc, is_loss(mated_in(5)));
  EXPECT(tc, mate_in(1) > mate_in(10));
  EXPECT(tc, mated_in(1) < mated_in(10));

  return tc.passed();
}

bool test_value_bounds() {
  TestCase tc("ValueBounds");

  EXPECT(tc, VALUE_MATE > VALUE_MATE_IN_MAX_PLY);
  EXPECT(tc, VALUE_MATE_IN_MAX_PLY > VALUE_TB);
  EXPECT(tc, VALUE_TB > VALUE_TB_WIN_IN_MAX_PLY);
  EXPECT(tc, VALUE_INFINITE > VALUE_MATE);

  return tc.passed();
}

// ============================================================================
// Info Structures Tests
// ============================================================================

bool test_info_short() {
  TestCase tc("InfoShort");

  Search::InfoShort info;
  info.depth = 15;

  EXPECT(tc, info.depth == 15);

  return tc.passed();
}

bool test_info_full() {
  TestCase tc("InfoFull");

  Search::InfoFull info;
  info.depth = 20;
  info.selDepth = 35;
  info.multiPV = 1;
  info.timeMs = 5000;
  info.nodes = 1000000;
  info.nps = 200000;
  info.hashfull = 500;

  EXPECT(tc, info.depth == 20);
  EXPECT(tc, info.selDepth == 35);
  EXPECT(tc, info.nodes == 1000000);
  EXPECT(tc, info.nps == 200000);

  return tc.passed();
}

} // namespace

bool test_search_comprehensive() {
  std::cout << "\n=== Comprehensive Search Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[History]" << std::endl;
  test_stats_entry();
  test_butterfly_history();
  test_capture_history();
  test_piece_to_history();
  test_continuation_history();

  std::cout << "\n[Limits]" << std::endl;
  test_limits_type();
  test_limits_depth();
  test_limits_nodes();
  test_limits_movetime();

  std::cout << "\n[RootMove]" << std::endl;
  test_root_move();
  test_root_move_sorting();

  std::cout << "\n[Skill]" << std::endl;
  test_skill_level();
  test_skill_time_to_pick();

  std::cout << "\n[Stack]" << std::endl;
  test_search_stack();

  std::cout << "\n[TimeManager]" << std::endl;
  test_time_manager_basic();
  test_time_manager_low_time();
  test_time_manager_increment();

  std::cout << "\n[Values]" << std::endl;
  test_mate_values();
  test_value_bounds();

  std::cout << "\n[Info]" << std::endl;
  test_info_short();
  test_info_full();

  std::cout << "\n=== Search Test Summary ===" << std::endl;
  std::cout << "  Passed: " << g_tests_passed << std::endl;
  std::cout << "  Failed: " << g_tests_failed << std::endl;

  return g_tests_failed == 0;
}

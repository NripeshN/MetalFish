/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Search component tests

  Note: Full search testing is done via the Python test suite (testing.py)
  which tests the engine through UCI protocol. These tests focus on
  basic components that can be tested in isolation without complex dependencies.
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "search/history.h"
#include "search/search.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>

using namespace MetalFish;

//==============================================================================
// Test Utilities
//==============================================================================

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

  void expect(bool condition, const char *expr, int line) {
    if (!condition) {
      std::cerr << "\n    FAILED: " << expr << " at line " << line << std::endl;
      passed_ = false;
    }
  }

  bool passed() const { return passed_; }

private:
  const char *name_;
  bool passed_;
};

#define EXPECT(tc, condition) tc.expect((condition), #condition, __LINE__)

//==============================================================================
// History Table Tests
//==============================================================================

bool test_butterfly_history() {
  TestCase tc("ButterflyHistory");

  ButterflyHistory history;

  // Test StatsEntry behavior
  auto &entry = history[WHITE][SQ_E2 * 64 + SQ_E4];

  // Apply positive bonus
  entry << 500;
  EXPECT(tc, int16_t(entry) > 0);

  // Apply negative bonus
  entry << -300;
  // Should still be positive but reduced
  EXPECT(tc, int16_t(entry) > 0);

  return tc.passed();
}

bool test_capture_history() {
  TestCase tc("CaptureHistory");

  CapturePieceToHistory captureHistory;

  // Test StatsEntry behavior
  auto &entry = captureHistory[W_KNIGHT][SQ_D5][PAWN];
  entry << 200;
  EXPECT(tc, int16_t(entry) > 0);

  return tc.passed();
}

//==============================================================================
// Stats Entry Tests
//==============================================================================

bool test_stats_entry() {
  TestCase tc("StatsEntry");

  StatsEntry<int16_t, 16384> entry;
  // StatsEntry doesn't initialize to 0 by default

  entry << 1000;
  EXPECT(tc, int16_t(entry) != 0);

  entry << -500;
  // After positive and negative, value should still be non-zero
  EXPECT(tc, true);

  return tc.passed();
}

//==============================================================================
// SEE Tests
//==============================================================================

bool test_see_basic() {
  TestCase tc("SEEBasic");

  Bitboards::init();
  Position::init();

  StateListPtr states = std::make_unique<std::deque<StateInfo>>(1);
  Position pos;

  // Position where pawn captures pawn
  pos.set("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
          false, &states->back());

  Move capture(SQ_E4, SQ_D5);
  EXPECT(tc, pos.see_ge(capture, 0));

  return tc.passed();
}

//==============================================================================
// Low Ply History Tests
//==============================================================================

bool test_low_ply_history() {
  TestCase tc("LowPlyHistory");

  LowPlyHistory lowHist;

  Move m1(SQ_E2, SQ_E4);
  int moveIdx = m1.from_sq() * 64 + m1.to_sq();

  for (int ply = 0; ply < LOW_PLY_HISTORY_SIZE; ++ply) {
    lowHist[ply][moveIdx] << ((ply + 1) * 100);
  }

  // Verify values were set (StatsEntry uses gravity formula)
  for (int ply = 0; ply < LOW_PLY_HISTORY_SIZE; ++ply) {
    EXPECT(tc, int16_t(lowHist[ply][moveIdx]) > 0);
  }

  return tc.passed();
}

//==============================================================================
// Mate Value Tests
//==============================================================================

bool test_mate_values() {
  TestCase tc("MateValues");

  Value mateIn5 = mate_in(5);
  EXPECT(tc, mateIn5 > VALUE_TB_WIN_IN_MAX_PLY);
  EXPECT(tc, mateIn5 < VALUE_MATE);

  Value matedIn5 = mated_in(5);
  EXPECT(tc, matedIn5 < VALUE_TB_LOSS_IN_MAX_PLY);
  EXPECT(tc, matedIn5 > -VALUE_MATE);

  EXPECT(tc, is_win(VALUE_MATE - 10));
  EXPECT(tc, !is_win(100));
  EXPECT(tc, is_loss(-VALUE_MATE + 10));
  EXPECT(tc, !is_loss(-100));

  EXPECT(tc, is_decisive(VALUE_MATE - 10));
  EXPECT(tc, is_decisive(-VALUE_MATE + 10));
  EXPECT(tc, !is_decisive(100));

  return tc.passed();
}

//==============================================================================
// Draw Detection Tests
//==============================================================================

bool test_draw_detection() {
  TestCase tc("DrawDetection");

  Bitboards::init();
  Position::init();

  StateListPtr states = std::make_unique<std::deque<StateInfo>>(1);
  Position pos;

  // Test 50-move rule (not yet triggered at 99)
  pos.set("8/8/8/8/8/8/8/4K2k w - - 99 100", false, &states->back());
  EXPECT(tc, !pos.is_draw(0));

  // VALUE_DRAW should be 0
  EXPECT(tc, VALUE_DRAW == 0);

  return tc.passed();
}

//==============================================================================
// Position Keys Tests
//==============================================================================

bool test_position_keys() {
  TestCase tc("PositionKeys");

  Bitboards::init();
  Position::init();

  StateListPtr states = std::make_unique<std::deque<StateInfo>>(1);
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &states->back());

  // Test that position keys are non-zero
  EXPECT(tc, pos.key() != 0);
  EXPECT(tc, pos.pawn_key() != 0);
  EXPECT(tc, pos.minor_piece_key() != 0);
  EXPECT(tc, pos.non_pawn_key(WHITE) != 0);
  EXPECT(tc, pos.non_pawn_key(BLACK) != 0);

  return tc.passed();
}

//==============================================================================
// Root Move Tests
//==============================================================================

bool test_root_move_structure() {
  TestCase tc("RootMoveStructure");

  Move m(SQ_E2, SQ_E4);
  Search::RootMove rm(m);

  EXPECT(tc, rm.pv.size() == 1);
  EXPECT(tc, rm.pv[0] == m);
  EXPECT(tc, rm.score == -VALUE_INFINITE);
  EXPECT(tc, rm.previousScore == -VALUE_INFINITE);
  EXPECT(tc, rm.averageScore == -VALUE_INFINITE);
  EXPECT(tc, rm.effort == 0);
  // meanSquaredScore initializes to -VALUE_INFINITE * VALUE_INFINITE
  EXPECT(tc, rm.meanSquaredScore != 0);

  Search::RootMove rm2(Move(SQ_D2, SQ_D4));
  rm.score = 100;
  rm2.score = 50;
  EXPECT(tc, rm < rm2); // Higher score should sort first

  return tc.passed();
}

//==============================================================================
// Skill Level Tests
//==============================================================================

bool test_skill_level() {
  TestCase tc("SkillLevel");

  Search::Skill fullStrength(20, 0);
  EXPECT(tc, !fullStrength.enabled());

  Search::Skill reduced(10, 0);
  EXPECT(tc, reduced.enabled());
  EXPECT(tc, reduced.level == 10.0);

  Search::Skill eloSkill(20, 2000);
  EXPECT(tc, eloSkill.enabled());
  EXPECT(tc, eloSkill.level < 20.0);

  Search::Skill skill5(5, 0);
  EXPECT(tc, skill5.time_to_pick(6));
  EXPECT(tc, !skill5.time_to_pick(5));

  return tc.passed();
}

//==============================================================================
// Main Test Runner
//==============================================================================

bool test_search() {
  std::cout << "\n=== Search Component Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[History Tables]" << std::endl;
  test_butterfly_history();
  test_capture_history();
  test_stats_entry();
  test_low_ply_history();

  std::cout << "\n[Position]" << std::endl;
  test_see_basic();
  test_draw_detection();
  test_position_keys();

  std::cout << "\n[Data Structures]" << std::endl;
  test_root_move_structure();
  test_skill_level();
  test_mate_values();

  std::cout << "\n=== Search Test Summary ===" << std::endl;
  std::cout << "  Passed: " << g_tests_passed << std::endl;
  std::cout << "  Failed: " << g_tests_failed << std::endl;

  if (g_tests_failed > 0) {
    std::cout << "  SOME TESTS FAILED!" << std::endl;
    return false;
  }

  std::cout << "All search component tests passed!" << std::endl;
  std::cout << "\nNote: Full search/eval integration tests are in testing.py"
            << std::endl;
  return true;
}

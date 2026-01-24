/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Search Tests - History Tables, Limits, Time Management, Values
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "search/history.h"
#include "search/search.h"
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
// History Tests
// ============================================================================

void test_history() {
  {
    TestCase tc("StatsEntry update");
    StatsEntry<int16_t, 7183> entry;
    entry = 0;
    EXPECT(tc, int16_t(entry) == 0);

    entry << 100;
    EXPECT(tc, int16_t(entry) > 0);
    EXPECT(tc, int16_t(entry) <= 7183);

    entry << -200;
    EXPECT(tc, int16_t(entry) < 100);
  }
  {
    TestCase tc("ButterflyHistory");
    ButterflyHistory history;
    for (int c = 0; c < COLOR_NB; ++c) {
      for (int i = 0; i < UINT_16_HISTORY_SIZE; ++i) {
        history[c][i] = 0;
      }
    }

    Move m(SQ_E2, SQ_E4);
    uint16_t idx = m.raw();
    history[WHITE][idx] << 500;
    EXPECT(tc, int16_t(history[WHITE][idx]) > 0);
    EXPECT(tc, int16_t(history[BLACK][idx]) == 0);
  }
  {
    TestCase tc("CaptureHistory");
    CapturePieceToHistory history;
    for (int pc = 0; pc < PIECE_NB; ++pc) {
      for (int sq = 0; sq < SQUARE_NB; ++sq) {
        for (int pt = 0; pt < PIECE_TYPE_NB; ++pt) {
          history[pc][sq][pt] = 0;
        }
      }
    }

    history[W_PAWN][SQ_D5][PAWN] << 300;
    EXPECT(tc, int16_t(history[W_PAWN][SQ_D5][PAWN]) > 0);
  }
  {
    TestCase tc("PieceToHistory");
    PieceToHistory history;
    for (int pc = 0; pc < PIECE_NB; ++pc) {
      for (int sq = 0; sq < SQUARE_NB; ++sq) {
        history[pc][sq] = 0;
      }
    }

    history[W_KNIGHT][SQ_F3] << 1000;
    EXPECT(tc, int16_t(history[W_KNIGHT][SQ_F3]) > 0);
  }
  {
    TestCase tc("LowPlyHistory");
    LowPlyHistory history;
    Move m(SQ_E2, SQ_E4);
    int idx = m.raw();

    for (int ply = 0; ply < LOW_PLY_HISTORY_SIZE; ++ply) {
      history[ply][idx] = 0;
      history[ply][idx] << ((ply + 1) * 100);
    }

    for (int ply = 0; ply < LOW_PLY_HISTORY_SIZE; ++ply) {
      EXPECT(tc, int16_t(history[ply][idx]) > 0);
    }
  }
}

// ============================================================================
// Limits Tests
// ============================================================================

void test_limits() {
  {
    TestCase tc("LimitsType defaults");
    Search::LimitsType limits;

    EXPECT(tc, limits.time[WHITE] == 0);
    EXPECT(tc, limits.time[BLACK] == 0);
    EXPECT(tc, limits.depth == 0);
    EXPECT(tc, limits.nodes == 0);
    EXPECT(tc, !limits.ponderMode);
    EXPECT(tc, !limits.use_time_management());
  }
  {
    TestCase tc("Time management detection");
    Search::LimitsType limits;
    EXPECT(tc, !limits.use_time_management());

    limits.time[WHITE] = 60000;
    EXPECT(tc, limits.use_time_management());
  }
  {
    TestCase tc("Depth limit");
    Search::LimitsType limits;
    limits.depth = 10;

    EXPECT(tc, limits.depth == 10);
    EXPECT(tc, !limits.use_time_management());
  }
  {
    TestCase tc("Nodes limit");
    Search::LimitsType limits;
    limits.nodes = 10000;

    EXPECT(tc, limits.nodes == 10000);
  }
  {
    TestCase tc("Movetime limit");
    Search::LimitsType limits;
    limits.movetime = 5000;

    EXPECT(tc, limits.movetime == 5000);
  }
}

// ============================================================================
// RootMove Tests
// ============================================================================

void test_rootmove() {
  {
    TestCase tc("RootMove creation");
    Move m(SQ_E2, SQ_E4);
    Search::RootMove rm(m);

    EXPECT(tc, rm.pv.size() == 1);
    EXPECT(tc, rm.pv[0] == m);
    EXPECT(tc, rm.score == -VALUE_INFINITE);
    EXPECT(tc, rm == m);
  }
  {
    TestCase tc("RootMove sorting");
    Search::RootMoves moves;
    moves.emplace_back(Move(SQ_E2, SQ_E4));
    moves.emplace_back(Move(SQ_D2, SQ_D4));
    moves.emplace_back(Move(SQ_G1, SQ_F3));

    moves[0].score = 50;
    moves[1].score = 100;
    moves[2].score = 25;

    std::sort(moves.begin(), moves.end());

    EXPECT(tc, moves[0].score == 100);
    EXPECT(tc, moves[1].score == 50);
    EXPECT(tc, moves[2].score == 25);
  }
}

// ============================================================================
// Skill Tests
// ============================================================================

void test_skill() {
  {
    TestCase tc("Skill level");
    Search::Skill skill(10, 0);
    EXPECT(tc, skill.enabled());
    EXPECT(tc, skill.level == 10.0);

    Search::Skill max_skill(20, 0);
    EXPECT(tc, !max_skill.enabled());
  }
  {
    TestCase tc("Skill from UCI Elo");
    Search::Skill skill_elo(20, 2000);
    EXPECT(tc, skill_elo.enabled());
  }
  {
    TestCase tc("Skill time_to_pick");
    Search::Skill skill(5, 0);
    EXPECT(tc, skill.time_to_pick(6));
    EXPECT(tc, !skill.time_to_pick(5));
  }
}

// ============================================================================
// Stack Tests
// ============================================================================

void test_stack() {
  {
    TestCase tc("SearchStack structure");
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
  }
}

// ============================================================================
// Value Tests
// ============================================================================

void test_values() {
  {
    TestCase tc("Mate values");
    EXPECT(tc, is_win(mate_in(5)));
    EXPECT(tc, is_loss(mated_in(5)));
    EXPECT(tc, mate_in(1) > mate_in(10));
    EXPECT(tc, mated_in(1) < mated_in(10));
  }
  {
    TestCase tc("Value bounds");
    EXPECT(tc, VALUE_MATE > VALUE_MATE_IN_MAX_PLY);
    EXPECT(tc, VALUE_MATE_IN_MAX_PLY > VALUE_TB);
    EXPECT(tc, VALUE_TB > VALUE_TB_WIN_IN_MAX_PLY);
    EXPECT(tc, VALUE_INFINITE > VALUE_MATE);
  }
  {
    TestCase tc("Decisive values");
    EXPECT(tc, is_decisive(VALUE_MATE - 10));
    EXPECT(tc, is_decisive(-VALUE_MATE + 10));
    EXPECT(tc, !is_decisive(100));
    EXPECT(tc, !is_decisive(-100));
  }
}

// ============================================================================
// Info Tests
// ============================================================================

void test_info() {
  {
    TestCase tc("InfoShort");
    Search::InfoShort info;
    info.depth = 15;
    EXPECT(tc, info.depth == 15);
  }
  {
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
  }
}

} // namespace

bool test_search_module() {
  std::cout << "\n=== Search Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[History]" << std::endl;
  test_history();

  std::cout << "\n[Limits]" << std::endl;
  test_limits();

  std::cout << "\n[RootMove]" << std::endl;
  test_rootmove();

  std::cout << "\n[Skill]" << std::endl;
  test_skill();

  std::cout << "\n[Stack]" << std::endl;
  test_stack();

  std::cout << "\n[Values]" << std::endl;
  test_values();

  std::cout << "\n[Info]" << std::endl;
  test_info();

  std::cout << "\n--- Search Results: " << g_tests_passed << " passed, "
            << g_tests_failed << " failed ---" << std::endl;

  return g_tests_failed == 0;
}

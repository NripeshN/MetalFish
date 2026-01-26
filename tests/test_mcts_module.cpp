/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file test_mcts_module.cpp
 * @brief MetalFish source file.
 */

  MCTS Tests - Thread-Safe MCTS, Nodes, Tree, Algorithms
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "mcts/thread_safe_mcts.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

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

#define EXPECT_NEAR(tc, a, b, eps)                                             \
  do {                                                                         \
    if (std::abs((a) - (b)) > (eps)) {                                         \
      tc.fail(#a " != " #b, __LINE__);                                         \
    }                                                                          \
  } while (0)

// ============================================================================
// TSEdge Tests
// ============================================================================

void test_edge() {
  {
    TestCase tc("Edge creation");
    TSEdge edge;
    EXPECT(tc, edge.move == Move::none());
    EXPECT(tc, edge.child.load() == nullptr);
  }
  {
    TestCase tc("Edge policy");
    TSEdge edge(Move(SQ_E2, SQ_E4), 0.5f);

    EXPECT(tc, edge.move == Move(SQ_E2, SQ_E4));
    EXPECT_NEAR(tc, edge.GetPolicy(), 0.5f, 0.01f);

    edge.SetPolicy(0.25f);
    EXPECT_NEAR(tc, edge.GetPolicy(), 0.25f, 0.01f);
  }
  {
    TestCase tc("Policy compression");
    float test_values[] = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f};

    for (float val : test_values) {
      TSEdge edge;
      edge.SetPolicy(val);
      float recovered = edge.GetPolicy();
      EXPECT_NEAR(tc, recovered, val, 0.02f);
    }
  }
}

// ============================================================================
// ThreadSafeNode Tests
// ============================================================================

void test_node() {
  {
    TestCase tc("Node creation");
    ThreadSafeNode node;

    EXPECT(tc, node.GetN() == 0);
    EXPECT(tc, node.GetNInFlight() == 0);
    EXPECT(tc, node.GetQ() == 0.0f);
    EXPECT(tc, !node.has_children());
    EXPECT(tc, !node.IsTerminal());
  }
  {
    TestCase tc("Virtual loss");
    ThreadSafeNode node;

    node.add_virtual_loss(3);
    EXPECT(tc, node.GetNInFlight() == 3);

    node.add_virtual_loss(2);
    EXPECT(tc, node.GetNInFlight() == 5);

    node.remove_virtual_loss(3);
    EXPECT(tc, node.GetNInFlight() == 2);
  }
  {
    TestCase tc("Score update");
    ThreadSafeNode node;

    node.FinalizeScoreUpdate(0.5f, 0.1f, 30.0f);
    EXPECT(tc, node.GetN() == 1);
    EXPECT_NEAR(tc, node.GetWL(), 0.5f, 0.01f);
    EXPECT_NEAR(tc, node.GetD(), 0.1f, 0.01f);

    node.FinalizeScoreUpdate(0.3f, 0.2f, 25.0f);
    EXPECT(tc, node.GetN() == 2);
    EXPECT_NEAR(tc, node.GetWL(), 0.4f, 0.05f);
  }
  {
    TestCase tc("Terminal node");
    ThreadSafeNode node;

    EXPECT(tc, !node.IsTerminal());
    EXPECT(tc, node.terminal_type() == ThreadSafeNode::Terminal::NonTerminal);

    node.MakeTerminal(ThreadSafeNode::Terminal::EndOfGame, 1.0f, 0.0f, 0.0f);
    EXPECT(tc, node.IsTerminal());
    EXPECT(tc, node.terminal_type() == ThreadSafeNode::Terminal::EndOfGame);
  }
  {
    TestCase tc("Children creation");
    ThreadSafeNode node;
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);

    MoveList<LEGAL> moves(pos);
    node.create_edges(moves);

    EXPECT(tc, node.has_children());
    EXPECT(tc, node.num_edges() == 20);
  }
  {
    TestCase tc("N started calculation");
    ThreadSafeNode node;

    node.add_virtual_loss(2);
    EXPECT(tc, node.GetNInFlight() == 2);
    EXPECT(tc, node.GetN() == 0);
    EXPECT(tc, node.GetNStarted() == 2);

    node.FinalizeScoreUpdate(0.5f, 0.0f, 0.0f);
    EXPECT(tc, node.GetN() == 1);
  }
  {
    TestCase tc("Concurrent updates");
    ThreadSafeNode node;
    const int NUM_THREADS = 4;
    const int UPDATES_PER_THREAD = 100;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
      threads.emplace_back([&]() {
        for (int j = 0; j < UPDATES_PER_THREAD; ++j) {
          node.FinalizeScoreUpdate(0.5f, 0.1f, 25.0f);
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    uint32_t n = node.GetN();
    EXPECT(tc, n > 0);
    EXPECT(tc, n <= NUM_THREADS * UPDATES_PER_THREAD);

    float q = node.GetWL();
    EXPECT(tc, q > 0.4f && q < 0.6f);
  }
}

// ============================================================================
// ThreadSafeTree Tests
// ============================================================================

void test_tree() {
  {
    TestCase tc("Tree creation");
    ThreadSafeTree tree;
    tree.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    EXPECT(tc, tree.root() != nullptr);
    EXPECT(tc, tree.node_count() >= 1);
  }
  {
    TestCase tc("Tree reset");
    ThreadSafeTree tree;
    tree.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    std::string fen1 = tree.root_fen();

    tree.reset(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

    std::string fen2 = tree.root_fen();
    EXPECT(tc, fen1 != fen2);
  }
  {
    TestCase tc("Node allocation");
    ThreadSafeTree tree;
    tree.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    size_t initial_count = tree.node_count();

    for (int i = 0; i < 10; ++i) {
      tree.allocate_node(tree.root(), i);
    }

    EXPECT(tc, tree.node_count() > initial_count);
  }
}

// ============================================================================
// WorkerContext Tests
// ============================================================================

void test_worker_context() {
  {
    TestCase tc("Context initialization");
    WorkerContext ctx;

    ctx.set_root_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    EXPECT(tc, ctx.pos.side_to_move() == WHITE);
    EXPECT(tc, ctx.move_stack.empty());
    EXPECT(tc, ctx.state_stack.empty());
  }
  {
    TestCase tc("Context do_move");
    WorkerContext ctx;
    ctx.set_root_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    ctx.do_move(Move(SQ_E2, SQ_E4));

    EXPECT(tc, ctx.pos.side_to_move() == BLACK);
    EXPECT(tc, ctx.move_stack.size() == 1);
    EXPECT(tc, ctx.state_stack.size() == 1);
  }
  {
    TestCase tc("Context reset");
    WorkerContext ctx;
    ctx.set_root_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    ctx.do_move(Move(SQ_E2, SQ_E4));
    ctx.do_move(Move(SQ_E7, SQ_E5));

    ctx.reset_to_root();

    EXPECT(tc, ctx.pos.side_to_move() == WHITE);
    EXPECT(tc, ctx.move_stack.empty());
  }
}

// ============================================================================
// MCTSConfig Tests
// ============================================================================

void test_config() {
  {
    TestCase tc("Config defaults");
    ThreadSafeMCTSConfig config;

    EXPECT(tc, config.cpuct > 0.0f);
    EXPECT(tc, config.cpuct_base > 0.0f);
    EXPECT(tc, config.fpu_reduction >= 0.0f);
    EXPECT(tc, config.virtual_loss > 0);
  }
  {
    TestCase tc("Config auto-tune");
    ThreadSafeMCTSConfig config;

    config.auto_tune(2);
    int batch_2 = config.max_batch_size;

    config.auto_tune(8);
    int batch_8 = config.max_batch_size;

    EXPECT(tc, batch_8 >= batch_2);
    EXPECT(tc, config.min_batch_size >= 1);
  }
  {
    TestCase tc("Config thread count");
    ThreadSafeMCTSConfig config;
    config.num_threads = 0;

    int auto_threads = config.get_num_threads();
    EXPECT(tc, auto_threads >= 2);
    EXPECT(tc, auto_threads <= 4);

    config.num_threads = 8;
    EXPECT(tc, config.get_num_threads() == 8);
  }
}

// ============================================================================
// MCTSStats Tests
// ============================================================================

void test_stats() {
  {
    TestCase tc("Stats reset");
    ThreadSafeMCTSStats stats;
    stats.reset();

    EXPECT(tc, stats.total_nodes == 0);
    EXPECT(tc, stats.total_iterations == 0);
    EXPECT(tc, stats.cache_hits == 0);
  }
  {
    TestCase tc("Stats NPS calculation");
    ThreadSafeMCTSStats stats;
    stats.reset();

    stats.total_nodes = 1000;
    stats.total_iterations = 100;

    EXPECT(tc, stats.nps(1.0) == 1000);
    EXPECT(tc, stats.nps(0.5) == 2000);
  }
  {
    TestCase tc("Stats avg batch size");
    ThreadSafeMCTSStats stats;
    stats.reset();

    stats.total_batch_size = 1000;
    stats.batch_count = 10;

    EXPECT_NEAR(tc, stats.avg_batch_size(), 100.0, 0.01);
  }
}

// ============================================================================
// ThreadSafeMCTS Tests
// ============================================================================

void test_mcts() {
  {
    TestCase tc("MCTS creation");
    ThreadSafeMCTSConfig config;
    config.num_threads = 1;

    auto mcts = std::make_unique<ThreadSafeMCTS>(config);
    EXPECT(tc, mcts != nullptr);
  }
  {
    TestCase tc("Short search");
    ThreadSafeMCTSConfig config;
    config.num_threads = 1;
    config.add_dirichlet_noise = false;

    auto mcts = std::make_unique<ThreadSafeMCTS>(config);

    Search::LimitsType limits;
    limits.movetime = 100;

    bool got_result = false;
    Move best = Move::none();

    mcts->start_search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
        [&](Move b, Move p) {
          got_result = true;
          best = b;
        });

    mcts->wait();

    EXPECT(tc, got_result);
    EXPECT(tc, best != Move::none());
  }
  {
    TestCase tc("Stop search");
    ThreadSafeMCTSConfig config;
    config.num_threads = 2;

    auto mcts = std::make_unique<ThreadSafeMCTS>(config);

    Search::LimitsType limits;
    limits.infinite = 1;

    std::atomic<bool> got_result{false};

    mcts->start_search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
        [&](Move b, Move p) { got_result = true; });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    mcts->stop();
    mcts->wait();

    EXPECT(tc, got_result);
  }
  {
    TestCase tc("Stats after search");
    ThreadSafeMCTSConfig config;
    config.num_threads = 2;
    config.add_dirichlet_noise = false;

    auto mcts = std::make_unique<ThreadSafeMCTS>(config);

    Search::LimitsType limits;
    limits.movetime = 200;

    mcts->start_search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
        [](Move, Move) {});

    mcts->wait();

    const auto &stats = mcts->stats();
    EXPECT(tc, stats.total_nodes > 0);
    EXPECT(tc, stats.total_iterations > 0);
  }
  {
    TestCase tc("Best Q value");
    ThreadSafeMCTSConfig config;
    config.num_threads = 1;

    auto mcts = std::make_unique<ThreadSafeMCTS>(config);

    Search::LimitsType limits;
    limits.movetime = 100;

    mcts->start_search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
        [](Move, Move) {});

    mcts->wait();

    float q = mcts->get_best_q();
    EXPECT(tc, q >= -1.0f && q <= 1.0f);
  }
}

} // namespace

bool test_mcts_module() {
  std::cout << "\n=== MCTS Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[Edge]" << std::endl;
  test_edge();

  std::cout << "\n[Node]" << std::endl;
  test_node();

  std::cout << "\n[Tree]" << std::endl;
  test_tree();

  std::cout << "\n[WorkerContext]" << std::endl;
  test_worker_context();

  std::cout << "\n[Config]" << std::endl;
  test_config();

  std::cout << "\n[Stats]" << std::endl;
  test_stats();

  std::cout << "\n[MCTS]" << std::endl;
  test_mcts();

  std::cout << "\n--- MCTS Results: " << g_tests_passed << " passed, "
            << g_tests_failed << " failed ---" << std::endl;

  return g_tests_failed == 0;
}
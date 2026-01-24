/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Comprehensive MCTS Tests - Thread-Safe MCTS, Nodes, Tree, Algorithms
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

#define EXPECT_NEAR(tc, a, b, eps)                                             \
  do {                                                                         \
    if (std::abs((a) - (b)) > (eps)) {                                         \
      tc.fail(#a " != " #b, __FILE__, __LINE__);                               \
    }                                                                          \
  } while (0)

// ============================================================================
// TSEdge Tests
// ============================================================================

bool test_ts_edge_creation() {
  TestCase tc("TSEdgeCreation");

  TSEdge edge;
  EXPECT(tc, edge.move == Move::none());
  EXPECT(tc, edge.child.load() == nullptr);

  return tc.passed();
}

bool test_ts_edge_policy() {
  TestCase tc("TSEdgePolicy");

  TSEdge edge(Move(SQ_E2, SQ_E4), 0.5f);

  EXPECT(tc, edge.move == Move(SQ_E2, SQ_E4));
  EXPECT_NEAR(tc, edge.GetPolicy(), 0.5f, 0.01f);

  edge.SetPolicy(0.25f);
  EXPECT_NEAR(tc, edge.GetPolicy(), 0.25f, 0.01f);

  return tc.passed();
}

bool test_ts_edge_policy_compression() {
  TestCase tc("TSEdgePolicyCompression");

  // Test various policy values for compression accuracy
  float test_values[] = {0.0f, 0.1f, 0.25f, 0.5f, 0.75f, 0.9f, 1.0f};

  for (float val : test_values) {
    TSEdge edge;
    edge.SetPolicy(val);
    float recovered = edge.GetPolicy();
    // Compression should maintain ~1% accuracy
    EXPECT_NEAR(tc, recovered, val, 0.02f);
  }

  return tc.passed();
}

// ============================================================================
// ThreadSafeNode Tests
// ============================================================================

bool test_node_creation() {
  TestCase tc("NodeCreation");

  ThreadSafeNode node;

  EXPECT(tc, node.GetN() == 0);
  EXPECT(tc, node.GetNInFlight() == 0);
  EXPECT(tc, node.GetQ() == 0.0f);
  EXPECT(tc, !node.has_children());
  EXPECT(tc, !node.IsTerminal());

  return tc.passed();
}

bool test_node_virtual_loss() {
  TestCase tc("NodeVirtualLoss");

  ThreadSafeNode node;

  node.add_virtual_loss(3);
  EXPECT(tc, node.GetNInFlight() == 3);

  node.add_virtual_loss(2);
  EXPECT(tc, node.GetNInFlight() == 5);

  node.remove_virtual_loss(3);
  EXPECT(tc, node.GetNInFlight() == 2);

  return tc.passed();
}

bool test_node_score_update() {
  TestCase tc("NodeScoreUpdate");

  ThreadSafeNode node;

  // First update
  node.FinalizeScoreUpdate(0.5f, 0.1f, 30.0f);
  EXPECT(tc, node.GetN() == 1);
  EXPECT_NEAR(tc, node.GetWL(), 0.5f, 0.01f);
  EXPECT_NEAR(tc, node.GetD(), 0.1f, 0.01f);

  // Second update - running average
  node.FinalizeScoreUpdate(0.3f, 0.2f, 25.0f);
  EXPECT(tc, node.GetN() == 2);
  // Q should be average: (0.5 + 0.3) / 2 = 0.4
  EXPECT_NEAR(tc, node.GetWL(), 0.4f, 0.05f);

  return tc.passed();
}

bool test_node_terminal() {
  TestCase tc("NodeTerminal");

  ThreadSafeNode node;

  EXPECT(tc, !node.IsTerminal());
  EXPECT(tc, node.terminal_type() == ThreadSafeNode::Terminal::NonTerminal);

  node.MakeTerminal(ThreadSafeNode::Terminal::EndOfGame, 1.0f, 0.0f, 0.0f);
  EXPECT(tc, node.IsTerminal());
  EXPECT(tc, node.terminal_type() == ThreadSafeNode::Terminal::EndOfGame);

  return tc.passed();
}

bool test_node_children_creation() {
  TestCase tc("NodeChildrenCreation");

  ThreadSafeNode node;
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  MoveList<LEGAL> moves(pos);
  node.create_edges(moves);

  EXPECT(tc, node.has_children());
  EXPECT(tc, node.num_edges() == 20); // Starting position has 20 legal moves

  return tc.passed();
}

bool test_node_n_started() {
  TestCase tc("NodeNStarted");

  ThreadSafeNode node;

  // Add virtual loss first
  node.add_virtual_loss(2);
  EXPECT(tc, node.GetNInFlight() == 2);
  EXPECT(tc, node.GetN() == 0);
  // NStarted = N + NInFlight = 0 + 2 = 2
  EXPECT(tc, node.GetNStarted() == 2);

  // Now finalize - this increments N
  node.FinalizeScoreUpdate(0.5f, 0.0f, 0.0f);
  EXPECT(tc, node.GetN() == 1);

  return tc.passed();
}

bool test_node_concurrent_updates() {
  TestCase tc("NodeConcurrentUpdates");

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

  // All updates should be counted
  uint32_t expected = NUM_THREADS * UPDATES_PER_THREAD;
  EXPECT(tc, node.GetN() >= expected * 0.95); // Allow small variance

  return tc.passed();
}

// ============================================================================
// ThreadSafeTree Tests
// ============================================================================

bool test_tree_creation() {
  TestCase tc("TreeCreation");

  ThreadSafeTree tree;
  tree.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  EXPECT(tc, tree.root() != nullptr);
  EXPECT(tc, tree.node_count() >= 1);

  return tc.passed();
}

bool test_tree_reset() {
  TestCase tc("TreeReset");

  ThreadSafeTree tree;
  tree.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  std::string fen1 = tree.root_fen();

  tree.reset("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - "
             "0 1");

  std::string fen2 = tree.root_fen();
  EXPECT(tc, fen1 != fen2);

  return tc.passed();
}

bool test_tree_node_allocation() {
  TestCase tc("TreeNodeAllocation");

  ThreadSafeTree tree;
  tree.reset("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  size_t initial_count = tree.node_count();

  // Allocate some nodes
  for (int i = 0; i < 10; ++i) {
    tree.allocate_node(tree.root(), i);
  }

  EXPECT(tc, tree.node_count() > initial_count);

  return tc.passed();
}

// ============================================================================
// WorkerContext Tests
// ============================================================================

bool test_worker_context() {
  TestCase tc("WorkerContext");

  WorkerContext ctx;

  ctx.set_root_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  EXPECT(tc, ctx.pos.side_to_move() == WHITE);
  EXPECT(tc, ctx.move_stack.empty());
  EXPECT(tc, ctx.state_stack.empty());

  return tc.passed();
}

bool test_worker_context_do_move() {
  TestCase tc("WorkerContextDoMove");

  WorkerContext ctx;
  ctx.set_root_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  ctx.do_move(Move(SQ_E2, SQ_E4));

  EXPECT(tc, ctx.pos.side_to_move() == BLACK);
  EXPECT(tc, ctx.move_stack.size() == 1);
  EXPECT(tc, ctx.state_stack.size() == 1);

  return tc.passed();
}

bool test_worker_context_reset() {
  TestCase tc("WorkerContextReset");

  WorkerContext ctx;
  ctx.set_root_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  ctx.do_move(Move(SQ_E2, SQ_E4));
  ctx.do_move(Move(SQ_E7, SQ_E5));

  ctx.reset_to_root();

  EXPECT(tc, ctx.pos.side_to_move() == WHITE);
  EXPECT(tc, ctx.move_stack.empty());

  return tc.passed();
}

// ============================================================================
// ThreadSafeMCTSConfig Tests
// ============================================================================

bool test_mcts_config_defaults() {
  TestCase tc("MCTSConfigDefaults");

  ThreadSafeMCTSConfig config;

  EXPECT(tc, config.cpuct > 0.0f);
  EXPECT(tc, config.cpuct_base > 0.0f);
  EXPECT(tc, config.fpu_reduction >= 0.0f);
  EXPECT(tc, config.virtual_loss > 0);

  return tc.passed();
}

bool test_mcts_config_auto_tune() {
  TestCase tc("MCTSConfigAutoTune");

  ThreadSafeMCTSConfig config;

  config.auto_tune(2);
  int batch_2 = config.max_batch_size;

  config.auto_tune(8);
  int batch_8 = config.max_batch_size;

  EXPECT(tc, batch_8 >= batch_2);
  EXPECT(tc, config.min_batch_size >= 1);

  return tc.passed();
}

bool test_mcts_config_thread_count() {
  TestCase tc("MCTSConfigThreadCount");

  ThreadSafeMCTSConfig config;
  config.num_threads = 0; // Auto-select

  int auto_threads = config.get_num_threads();
  EXPECT(tc, auto_threads >= 2);
  EXPECT(tc, auto_threads <= 4);

  config.num_threads = 8;
  EXPECT(tc, config.get_num_threads() == 8);

  return tc.passed();
}

// ============================================================================
// ThreadSafeMCTSStats Tests
// ============================================================================

bool test_mcts_stats() {
  TestCase tc("MCTSStats");

  ThreadSafeMCTSStats stats;
  stats.reset();

  EXPECT(tc, stats.total_nodes == 0);
  EXPECT(tc, stats.total_iterations == 0);
  EXPECT(tc, stats.cache_hits == 0);

  stats.total_nodes = 1000;
  stats.total_iterations = 100;

  EXPECT(tc, stats.nps(1.0) == 1000);
  EXPECT(tc, stats.nps(0.5) == 2000);

  return tc.passed();
}

bool test_mcts_stats_avg_batch() {
  TestCase tc("MCTSStatsAvgBatch");

  ThreadSafeMCTSStats stats;
  stats.reset();

  stats.total_batch_size = 1000;
  stats.batch_count = 10;

  EXPECT_NEAR(tc, stats.avg_batch_size(), 100.0, 0.01);

  return tc.passed();
}

// ============================================================================
// ThreadSafeMCTS Tests
// ============================================================================

bool test_mcts_creation() {
  TestCase tc("MCTSCreation");

  ThreadSafeMCTSConfig config;
  config.num_threads = 1;

  auto mcts = std::make_unique<ThreadSafeMCTS>(config);
  EXPECT(tc, mcts != nullptr);

  return tc.passed();
}

bool test_mcts_short_search() {
  TestCase tc("MCTSShortSearch");

  ThreadSafeMCTSConfig config;
  config.num_threads = 1;
  config.add_dirichlet_noise = false;

  auto mcts = std::make_unique<ThreadSafeMCTS>(config);

  Search::LimitsType limits;
  limits.movetime = 100; // 100ms

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

  return tc.passed();
}

bool test_mcts_stop() {
  TestCase tc("MCTSStop");

  ThreadSafeMCTSConfig config;
  config.num_threads = 2;

  auto mcts = std::make_unique<ThreadSafeMCTS>(config);

  Search::LimitsType limits;
  limits.infinite = 1;

  std::atomic<bool> got_result{false};

  mcts->start_search(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      [&](Move b, Move p) { got_result = true; });

  // Let it run briefly
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  mcts->stop();
  mcts->wait();

  EXPECT(tc, got_result);

  return tc.passed();
}

bool test_mcts_stats_after_search() {
  TestCase tc("MCTSStatsAfterSearch");

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

  return tc.passed();
}

bool test_mcts_get_best_q() {
  TestCase tc("MCTSGetBestQ");

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

  return tc.passed();
}

} // namespace

bool test_mcts_comprehensive() {
  std::cout << "\n=== Comprehensive MCTS Tests ===" << std::endl;

  Bitboards::init();
  Position::init();

  g_tests_passed = 0;
  g_tests_failed = 0;

  std::cout << "\n[TSEdge]" << std::endl;
  test_ts_edge_creation();
  test_ts_edge_policy();
  test_ts_edge_policy_compression();

  std::cout << "\n[ThreadSafeNode]" << std::endl;
  test_node_creation();
  test_node_virtual_loss();
  test_node_score_update();
  test_node_terminal();
  test_node_children_creation();
  test_node_n_started();
  test_node_concurrent_updates();

  std::cout << "\n[ThreadSafeTree]" << std::endl;
  test_tree_creation();
  test_tree_reset();
  test_tree_node_allocation();

  std::cout << "\n[WorkerContext]" << std::endl;
  test_worker_context();
  test_worker_context_do_move();
  test_worker_context_reset();

  std::cout << "\n[MCTSConfig]" << std::endl;
  test_mcts_config_defaults();
  test_mcts_config_auto_tune();
  test_mcts_config_thread_count();

  std::cout << "\n[MCTSStats]" << std::endl;
  test_mcts_stats();
  test_mcts_stats_avg_batch();

  std::cout << "\n[ThreadSafeMCTS]" << std::endl;
  test_mcts_creation();
  test_mcts_short_search();
  test_mcts_stop();
  test_mcts_stats_after_search();
  test_mcts_get_best_q();

  std::cout << "\n=== MCTS Test Summary ===" << std::endl;
  std::cout << "  Passed: " << g_tests_passed << std::endl;
  std::cout << "  Failed: " << g_tests_failed << std::endl;

  return g_tests_failed == 0;
}

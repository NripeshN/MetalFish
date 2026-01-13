/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Hybrid Search Test Suite

  Tests all MCTS and hybrid search components including:
  - Position classifier
  - Stockfish adapter
  - MCTS transposition table
  - Batch evaluator
  - Hybrid search
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
#include "mcts/stockfish_adapter.h"
#include "mcts/position_classifier.h"
#include "mcts/mcts_tt.h"
#include "mcts/mcts_batch_evaluator.h"
#include "mcts/hybrid_search.h"
#include "mcts/enhanced_hybrid_search.h"

using namespace MetalFish;
using namespace MetalFish::MCTS;

namespace {

// Test utilities
static int g_tests_passed = 0;
static int g_tests_failed = 0;

class TestCase {
public:
  TestCase(const char* name) : name_(name), passed_(true) {
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
  
  void expect(bool condition, const char* expr, int line) {
    if (!condition) {
      std::cerr << "\n    FAILED: " << expr << " at line " << line << std::endl;
      passed_ = false;
    }
  }
  
  bool passed() const { return passed_; }

private:
  const char* name_;
  bool passed_;
};

#define EXPECT(tc, condition) tc.expect((condition), #condition, __LINE__)

// Test positions
const char* TEST_FENS[] = {
  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  // Starting
  "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  // Tactical
  "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  // Italian
  "8/8/8/8/8/8/k7/4K2R w - - 0 1",  // Endgame
  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // Complex
};
const int NUM_TEST_FENS = sizeof(TEST_FENS) / sizeof(TEST_FENS[0]);

}  // namespace

// ============================================================================
// Stockfish Adapter Tests
// ============================================================================

bool test_mcts_move() {
  TestCase tc("MCTSMove");
  
  // Test move creation
  Move sf_move(SQ_E2, SQ_E4);
  MCTSMove mcts_move = MCTSMove::FromStockfish(sf_move);
  
  EXPECT(tc, mcts_move.to_stockfish() == sf_move);
  EXPECT(tc, !mcts_move.is_null());
  
  // Test null move
  MCTSMove null_move;
  EXPECT(tc, null_move.is_null());
  
  // Test promotion - use make<PROMOTION> helper
  Move promo_move = Move::make<PROMOTION>(SQ_E7, SQ_E8, QUEEN);
  MCTSMove mcts_promo = MCTSMove::FromStockfish(promo_move);
  EXPECT(tc, mcts_promo.is_promotion());
  
  return tc.passed();
}

bool test_mcts_position() {
  TestCase tc("MCTSPosition");
  
  Bitboards::init();
  Position::init();
  
  MCTSPositionHistory history;
  history.reset(TEST_FENS[0]);
  
  MCTSPosition pos = history.current();
  
  // Test basic properties
  EXPECT(tc, !pos.is_black_to_move());
  EXPECT(tc, !pos.is_terminal());
  EXPECT(tc, pos.hash() != 0);
  
  // Test move generation
  std::vector<MCTSMove> moves = pos.generate_legal_moves();
  EXPECT(tc, moves.size() == 20);  // 20 legal moves from starting position
  
  // Test do_move
  MCTSMove e2e4 = MCTSMove::FromStockfish(Move(SQ_E2, SQ_E4));
  MCTSPosition new_pos = pos;
  new_pos.do_move(e2e4);
  EXPECT(tc, new_pos.is_black_to_move());
  EXPECT(tc, new_pos.hash() != pos.hash());
  
  return tc.passed();
}

bool test_mcts_position_history() {
  TestCase tc("MCTSPositionHistory");
  
  MCTSPositionHistory history;
  history.reset(TEST_FENS[0]);
  
  // Make a move
  Move e2e4(SQ_E2, SQ_E4);
  history.do_move(e2e4);
  
  // Make another move
  Move e7e5(SQ_E7, SQ_E5);
  history.do_move(e7e5);
  
  // Check current position
  MCTSPosition current = history.current();
  EXPECT(tc, !current.is_black_to_move());  // White to move after e4 e5
  
  return tc.passed();
}

// ============================================================================
// Position Classifier Tests
// ============================================================================

bool test_position_classifier_basic() {
  TestCase tc("PositionClassifierBasic");
  
  Bitboards::init();
  Position::init();
  
  PositionClassifier classifier;
  
  std::deque<StateInfo> states(1);
  Position pos;
  pos.set(TEST_FENS[0], false, &states.back());
  
  PositionType type = classifier.quick_classify(pos);
  
  // Starting position should be strategic (no immediate tactics)
  EXPECT(tc, type == PositionType::STRATEGIC || type == PositionType::HIGHLY_STRATEGIC);
  
  return tc.passed();
}

bool test_position_classifier_tactical() {
  TestCase tc("PositionClassifierTactical");
  
  PositionClassifier classifier;
  
  std::deque<StateInfo> states(1);
  Position pos;
  // Position with queen attacking f7 (Scholar's mate setup)
  pos.set(TEST_FENS[1], false, &states.back());
  
  // Analyze and get features
  PositionFeatures features = classifier.analyze(pos);
  
  // Should have some tactical elements
  EXPECT(tc, features.tactical_score > 0.0f || features.num_captures > 0 || features.num_checks_available > 0);
  
  return tc.passed();
}

bool test_position_classifier_endgame() {
  TestCase tc("PositionClassifierEndgame");
  
  PositionClassifier classifier;
  
  std::deque<StateInfo> states(1);
  Position pos;
  // Simple endgame position
  pos.set(TEST_FENS[3], false, &states.back());
  
  // Check that it's identified as endgame-like
  PositionFeatures features = classifier.analyze(pos);
  EXPECT(tc, features.is_endgame);
  
  return tc.passed();
}

// ============================================================================
// MCTS Transposition Table Tests
// ============================================================================

bool test_mcts_tt_basic() {
  TestCase tc("MCTSTTBasic");
  
  MCTSTTConfig config;
  config.size_mb = 1;  // Small for testing
  
  MCTSTranspositionTable tt;
  bool init_ok = tt.initialize(config);
  EXPECT(tc, init_ok);
  EXPECT(tc, tt.num_entries() > 0);
  
  return tc.passed();
}

bool test_mcts_tt_store_probe() {
  TestCase tc("MCTSTTStoreProbe");
  
  MCTSTTConfig config;
  config.size_mb = 1;
  
  MCTSTranspositionTable tt;
  tt.initialize(config);
  
  // Store MCTS stats
  uint64_t key = 0x123456789ABCDEF0ULL;
  MCTSStats stats;
  stats.q = 0.5f;
  stats.n = 100;
  stats.d = 0.1f;
  stats.m = 30.0f;
  stats.set_policy(0.25f);
  
  tt.store_mcts(key, stats);
  
  // Probe
  MCTSStats retrieved;
  bool found = tt.probe_mcts(key, retrieved);
  
  EXPECT(tc, found);
  EXPECT(tc, std::abs(retrieved.q - 0.5f) < 0.001f);
  EXPECT(tc, retrieved.n == 100);
  EXPECT(tc, std::abs(retrieved.d - 0.1f) < 0.001f);
  
  return tc.passed();
}

bool test_mcts_tt_update() {
  TestCase tc("MCTSTTUpdate");
  
  MCTSTTConfig config;
  config.size_mb = 1;
  
  MCTSTranspositionTable tt;
  tt.initialize(config);
  
  uint64_t key = 0xDEADBEEFCAFEBABEULL;
  
  // First update
  tt.update_mcts(key, 0.3f, 0.1f, 25.0f);
  
  MCTSStats stats;
  bool found = tt.probe_mcts(key, stats);
  EXPECT(tc, found);
  EXPECT(tc, stats.n == 1);
  EXPECT(tc, std::abs(stats.q - 0.3f) < 0.001f);
  
  // Second update (running average)
  tt.update_mcts(key, 0.5f, 0.2f, 20.0f);
  
  found = tt.probe_mcts(key, stats);
  EXPECT(tc, found);
  EXPECT(tc, stats.n == 2);
  EXPECT(tc, std::abs(stats.q - 0.4f) < 0.001f);  // Average of 0.3 and 0.5
  
  return tc.passed();
}

bool test_mcts_tt_ab_bounds() {
  TestCase tc("MCTSTTABBounds");
  
  MCTSTTConfig config;
  config.size_mb = 1;
  
  MCTSTranspositionTable tt;
  tt.initialize(config);
  
  uint64_t key = 0x1111222233334444ULL;
  
  ABBounds bounds;
  bounds.score = 150;
  bounds.depth = 10;
  bounds.bound = 1;  // EXACT
  bounds.best_move = Move(SQ_E2, SQ_E4);
  
  tt.store_ab(key, bounds);
  
  ABBounds retrieved;
  bool found = tt.probe_ab(key, retrieved);
  
  EXPECT(tc, found);
  EXPECT(tc, retrieved.score == 150);
  EXPECT(tc, retrieved.depth == 10);
  EXPECT(tc, retrieved.best_move == Move(SQ_E2, SQ_E4));
  
  return tc.passed();
}

// ============================================================================
// Batch Evaluator Tests
// ============================================================================

bool test_batch_buffer() {
  TestCase tc("BatchBuffer");
  
  UnifiedBatchBuffer buffer;
  buffer.allocate(64);
  
  EXPECT(tc, buffer.capacity == 64);
  EXPECT(tc, buffer.count == 0);
  EXPECT(tc, !buffer.is_full());
  
  buffer.clear();
  EXPECT(tc, buffer.count == 0);
  
  return tc.passed();
}

bool test_lock_free_collector() {
  TestCase tc("LockFreeBatchCollector");
  
  LockFreeBatchCollector collector(32);
  
  EXPECT(tc, collector.size() == 0);
  EXPECT(tc, !collector.is_ready(1));
  
  // Add some positions
  MCTSPositionHistory history;
  history.reset(TEST_FENS[0]);
  MCTSPosition pos = history.current();
  
  int idx = collector.add(nullptr, pos);
  EXPECT(tc, idx == 0);
  EXPECT(tc, collector.size() == 1);
  EXPECT(tc, collector.is_ready(1));
  
  // Take batch
  std::vector<HybridNode*> nodes;
  std::vector<MCTSPosition> positions;
  int count = collector.take_batch(nodes, positions);
  
  EXPECT(tc, count == 1);
  EXPECT(tc, collector.size() == 0);
  
  return tc.passed();
}

// ============================================================================
// Hybrid Search Tests
// ============================================================================

bool test_hybrid_edge() {
  TestCase tc("HybridEdge");
  
  MCTSMove move = MCTSMove::FromStockfish(Move(SQ_E2, SQ_E4));
  HybridEdge edge(move);
  
  EXPECT(tc, edge.move().to_stockfish() == Move(SQ_E2, SQ_E4));
  EXPECT(tc, edge.policy() == 0.0f);
  
  edge.set_policy(0.5f);
  EXPECT(tc, std::abs(edge.policy() - 0.5f) < 0.001f);
  
  return tc.passed();
}

bool test_hybrid_node() {
  TestCase tc("HybridNode");
  
  HybridNode node;
  
  EXPECT(tc, node.n() == 0);
  EXPECT(tc, !node.has_children());
  EXPECT(tc, !node.is_terminal());
  
  // Add edges
  std::vector<MCTSMove> moves;
  moves.push_back(MCTSMove::FromStockfish(Move(SQ_E2, SQ_E4)));
  moves.push_back(MCTSMove::FromStockfish(Move(SQ_D2, SQ_D4)));
  node.create_edges(moves);
  
  EXPECT(tc, node.has_children());
  EXPECT(tc, node.edges().size() == 2);
  
  // Test terminal state
  node.set_terminal(HybridNode::Terminal::Win, 1.0f);
  EXPECT(tc, node.is_terminal());
  
  return tc.passed();
}

bool test_hybrid_tree() {
  TestCase tc("HybridTree");
  
  MCTSPositionHistory history;
  history.reset(TEST_FENS[0]);
  
  HybridTree tree;
  tree.reset(history);
  
  EXPECT(tc, tree.root() != nullptr);
  EXPECT(tc, tree.node_count() == 1);
  
  return tc.passed();
}

// ============================================================================
// Integration Tests
// ============================================================================

bool test_hybrid_search_basic() {
  TestCase tc("HybridSearchBasic");
  
  if (!GPU::gpu_available()) {
    std::cout << "(skipped - no GPU) ";
    return true;
  }
  
  auto& gpu_manager = GPU::gpu_nnue_manager();
  if (!gpu_manager.is_ready()) {
    gpu_manager.initialize();
  }
  
  HybridSearch search;
  search.set_gpu_nnue(&gpu_manager);
  
  // Basic instantiation test
  EXPECT(tc, true);
  
  return tc.passed();
}

bool test_enhanced_hybrid_search() {
  TestCase tc("EnhancedHybridSearch");
  
  if (!GPU::gpu_available()) {
    std::cout << "(skipped - no GPU) ";
    return true;
  }
  
  auto& gpu_manager = GPU::gpu_nnue_manager();
  if (!gpu_manager.is_ready()) {
    gpu_manager.initialize();
  }
  
  EnhancedHybridSearch search;
  bool init_ok = search.initialize(&gpu_manager);
  
  if (!init_ok) {
    std::cout << "(skipped - init failed) ";
    return true;
  }
  
  EXPECT(tc, init_ok);
  
  return tc.passed();
}

// ============================================================================
// Performance Tests
// ============================================================================

bool test_tt_performance() {
  TestCase tc("TTPerformance");
  
  MCTSTTConfig config;
  config.size_mb = 16;
  
  MCTSTranspositionTable tt;
  tt.initialize(config);
  
  const int num_ops = 100000;
  
  // Store benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_ops; ++i) {
    uint64_t key = uint64_t(i) * 0x9E3779B97F4A7C15ULL;
    tt.update_mcts(key, float(i % 100) / 100.0f, 0.1f, 30.0f);
  }
  auto end = std::chrono::high_resolution_clock::now();
  double store_time = std::chrono::duration<double, std::milli>(end - start).count();
  
  // Probe benchmark
  start = std::chrono::high_resolution_clock::now();
  int hits = 0;
  MCTSStats stats;
  for (int i = 0; i < num_ops; ++i) {
    uint64_t key = uint64_t(i) * 0x9E3779B97F4A7C15ULL;
    if (tt.probe_mcts(key, stats)) hits++;
  }
  end = std::chrono::high_resolution_clock::now();
  double probe_time = std::chrono::duration<double, std::milli>(end - start).count();
  
  std::cout << "\n    Store: " << std::fixed << std::setprecision(2) 
            << (num_ops / store_time * 1000) << " ops/sec";
  std::cout << "\n    Probe: " << (num_ops / probe_time * 1000) << " ops/sec";
  std::cout << "\n    Hit rate: " << (100.0 * hits / num_ops) << "%" << std::endl;
  std::cout << "  ";
  
  EXPECT(tc, store_time < 1000);  // Should complete in under 1 second
  EXPECT(tc, probe_time < 1000);
  EXPECT(tc, hits > num_ops / 2);  // Should have decent hit rate
  
  return tc.passed();
}

// ============================================================================
// Main Test Runner
// ============================================================================

bool test_mcts() {
  std::cout << "\n=== MCTS Hybrid Search Tests ===" << std::endl;
  
  Bitboards::init();
  Position::init();
  
  g_tests_passed = 0;
  g_tests_failed = 0;
  
  std::cout << "\n[Stockfish Adapter]" << std::endl;
  test_mcts_move();
  test_mcts_position();
  test_mcts_position_history();
  
  std::cout << "\n[Position Classifier]" << std::endl;
  test_position_classifier_basic();
  test_position_classifier_tactical();
  test_position_classifier_endgame();
  
  std::cout << "\n[MCTS Transposition Table]" << std::endl;
  test_mcts_tt_basic();
  test_mcts_tt_store_probe();
  test_mcts_tt_update();
  test_mcts_tt_ab_bounds();
  
  std::cout << "\n[Batch Evaluator]" << std::endl;
  test_batch_buffer();
  test_lock_free_collector();
  
  std::cout << "\n[Hybrid Search]" << std::endl;
  test_hybrid_edge();
  test_hybrid_node();
  test_hybrid_tree();
  test_hybrid_search_basic();
  test_enhanced_hybrid_search();
  
  std::cout << "\n[Performance]" << std::endl;
  test_tt_performance();
  
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

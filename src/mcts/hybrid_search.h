/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Hybrid Search: Combines MCTS tree search with alpha-beta pruning.

  The key insight is that MCTS excels at strategic evaluation with neural
  networks, while alpha-beta excels at tactical calculation with fast
  evaluation.

  This hybrid approach:
  1. Uses MCTS for the root and early tree exploration (strategic planning)
  2. Uses alpha-beta for leaf evaluation when depth is sufficient (tactical
  verification)
  3. Batches neural network evaluations on GPU for throughput
  4. Falls back to alpha-beta for time-critical situations

  Key optimizations:
  - Lock-free evaluation request pool (zero allocation overhead)
  - Arena-based node allocation (reduced contention)
  - Cache-line aligned data structures
  - Batched GPU evaluation with dedicated thread
  - Sampled profiling to reduce chrono overhead
  - Batched stop checks to reduce atomic reads

  Licensed under GPL-3.0
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "../gpu/gpu_nnue_integration.h"
#include "../search/search.h"
#include "mcts_batch_evaluator.h"
#include "mcts_tt.h"
#include "stockfish_adapter.h"

using namespace MetalFish;

namespace MetalFish {
namespace MCTS {

// Forward declarations
class HybridNode;
class HybridTree;
class HybridSearch;
class HybridBatchedEvaluator;

// Configuration for hybrid search
struct HybridSearchConfig {
  // MCTS parameters - tuned for better play
  float cpuct = 1.5f;            // Reduced from 2.5 for less exploration, more exploitation
  float fpu_value = 0.0f;        // First Play Urgency base value (neutral)
  float fpu_reduction = 0.2f;    // Reduction from parent Q for unexplored moves
  float policy_softmax_temp = 1.0f;
  bool add_dirichlet_noise = true;
  float dirichlet_alpha = 0.3f;
  float dirichlet_epsilon = 0.25f;
  int virtual_loss = 3;

  // Alpha-beta integration parameters
  int ab_depth_threshold = 6;
  int ab_node_threshold = 1000;
  float ab_confidence_threshold = 0.9f;
  bool use_ab_for_tactics = true;

  // Batching parameters - auto-tuned based on thread count
  int min_batch_size = 1; // Process immediately if available
  int max_batch_size = 256;
  int batch_timeout_us = 25; // Very short timeout
  bool use_batched_eval = true;

  // Threading - supports -1 for auto (all cores)
  int num_search_threads = 1;

  int get_num_threads() const {
    if (num_search_threads <= 0) {
      int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
      return hw_threads > 0 ? hw_threads : 4;
    }
    return num_search_threads;
  }

  void auto_tune(int actual_threads) {
    if (actual_threads <= 2) {
      min_batch_size = 1;
      batch_timeout_us = 10;
      max_batch_size = 32;
    } else if (actual_threads <= 4) {
      min_batch_size = 1;
      batch_timeout_us = 25;
      max_batch_size = 64;
    } else {
      min_batch_size = 2;
      batch_timeout_us = 50;
      max_batch_size = 128;
    }
  }

  // Time management
  bool use_smart_pruning = true;
  float time_curve_peak = 0.3f;
  float time_curve_left_width = 0.2f;
  float time_curve_right_width = 0.3f;

  // Profiling control
  static constexpr int PROFILE_SAMPLE_RATE = 64;
  static constexpr int STOP_CHECK_INTERVAL = 64;
};

// Statistics for hybrid search
struct HybridSearchStats {
  std::atomic<uint64_t> mcts_nodes{0};
  std::atomic<uint64_t> ab_nodes{0};
  std::atomic<uint64_t> nn_evaluations{0};
  std::atomic<uint64_t> nn_batches{0};
  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> cache_misses{0};
  std::atomic<uint64_t> tactical_positions{0};
  std::atomic<uint64_t> ab_overrides{0};

  // Profiling breakdown (in microseconds)
  std::atomic<uint64_t> selection_time_us{0};
  std::atomic<uint64_t> expansion_time_us{0};
  std::atomic<uint64_t> evaluation_time_us{0};
  std::atomic<uint64_t> backprop_time_us{0};
  std::atomic<uint64_t> batch_wait_time_us{0};
  std::atomic<uint64_t> total_iterations{0};
  std::atomic<uint64_t> total_batch_size{0};
  std::atomic<uint64_t> batch_count{0};

  void reset() {
    mcts_nodes = 0;
    ab_nodes = 0;
    nn_evaluations = 0;
    nn_batches = 0;
    cache_hits = 0;
    cache_misses = 0;
    tactical_positions = 0;
    ab_overrides = 0;
    selection_time_us = 0;
    expansion_time_us = 0;
    evaluation_time_us = 0;
    backprop_time_us = 0;
    batch_wait_time_us = 0;
    total_iterations = 0;
    total_batch_size = 0;
    batch_count = 0;
  }

  double avg_batch_size() const {
    uint64_t count = batch_count.load(std::memory_order_relaxed);
    return count > 0 ? static_cast<double>(
                           total_batch_size.load(std::memory_order_relaxed)) /
                           count
                     : 0;
  }

  void get_profile_breakdown(double &selection_pct, double &expansion_pct,
                             double &eval_pct, double &backprop_pct) const {
    uint64_t total = selection_time_us + expansion_time_us +
                     evaluation_time_us + backprop_time_us;
    if (total == 0) {
      selection_pct = expansion_pct = eval_pct = backprop_pct = 0.0;
      return;
    }
    selection_pct = 100.0 * selection_time_us / total;
    expansion_pct = 100.0 * expansion_time_us / total;
    eval_pct = 100.0 * evaluation_time_us / total;
    backprop_pct = 100.0 * backprop_time_us / total;
  }
};

// Cache-line aligned edge for optimal memory access
struct alignas(64) HybridEdge {
  MCTSMove move_;
  std::atomic<float> policy_{0.0f};
  std::atomic<HybridNode *> child_{nullptr};

  HybridEdge() = default;

  void init(MCTSMove m, float p = 0.0f) {
    move_ = m;
    policy_.store(p, std::memory_order_relaxed);
    child_.store(nullptr, std::memory_order_relaxed);
  }

  MCTSMove move() const { return move_; }
  float policy() const { return policy_.load(std::memory_order_relaxed); }
  void set_policy(float p) { policy_.store(p, std::memory_order_relaxed); }

  HybridNode *child() const { return child_.load(std::memory_order_acquire); }
  void set_child(HybridNode *node) {
    child_.store(node, std::memory_order_release);
  }
};

// Cache-line aligned node for optimal memory access
class alignas(64) HybridNode {
public:
  enum class Terminal : uint8_t { NonTerminal, Win, Draw, Loss, Tablebase };

  HybridNode() = default;
  HybridNode(HybridNode *parent, int edge_index);
  ~HybridNode() = default;

  HybridNode *parent() const { return parent_; }
  int edge_index() const { return edge_index_; }
  bool has_children() const {
    return num_edges_.load(std::memory_order_acquire) > 0;
  }

  int num_edges() const { return num_edges_.load(std::memory_order_acquire); }
  HybridEdge *edges() { return edges_.get(); }
  const HybridEdge *edges() const { return edges_.get(); }
  void create_edges(const MCTSMoveList &moves);

  uint32_t n() const { return n_.load(std::memory_order_acquire); }
  uint32_t n_in_flight() const {
    return n_in_flight_.load(std::memory_order_acquire);
  }
  float q() const { return q_.load(std::memory_order_acquire); }
  float d() const { return d_.load(std::memory_order_acquire); }
  float m() const { return m_.load(std::memory_order_acquire); }

  void add_virtual_loss(int count = 1) {
    n_in_flight_.fetch_add(count, std::memory_order_acq_rel);
  }
  void remove_virtual_loss(int count = 1) {
    n_in_flight_.fetch_sub(count, std::memory_order_acq_rel);
  }
  void update_stats(float value, float draw_prob, float moves_left);

  Terminal terminal_type() const {
    return terminal_type_.load(std::memory_order_acquire);
  }
  bool is_terminal() const { return terminal_type() != Terminal::NonTerminal; }
  void set_terminal(Terminal type, float value);

  bool has_ab_score() const {
    return has_ab_score_.load(std::memory_order_acquire);
  }
  int ab_score() const { return ab_score_.load(std::memory_order_acquire); }
  int ab_depth() const { return ab_depth_.load(std::memory_order_acquire); }
  void set_ab_score(int score, int depth);

  float get_u(float cpuct, float parent_n_sqrt) const;
  float get_puct(float cpuct, float parent_n_sqrt, float fpu) const;

  void reset_parent() {
    parent_ = nullptr;
    edge_index_ = -1;
  }

  void init(HybridNode *parent, int edge_index) {
    parent_ = parent;
    edge_index_ = edge_index;
    edges_.reset();
    num_edges_.store(0, std::memory_order_relaxed);
    n_.store(0, std::memory_order_relaxed);
    n_in_flight_.store(0, std::memory_order_relaxed);
    w_.store(0.0f, std::memory_order_relaxed);
    q_.store(0.0f, std::memory_order_relaxed);
    d_.store(0.0f, std::memory_order_relaxed);
    m_.store(0.0f, std::memory_order_relaxed);
    terminal_type_.store(Terminal::NonTerminal, std::memory_order_relaxed);
    has_ab_score_.store(false, std::memory_order_relaxed);
    ab_score_.store(0, std::memory_order_relaxed);
    ab_depth_.store(0, std::memory_order_relaxed);
  }

private:
  HybridNode *parent_ = nullptr;
  int edge_index_ = -1;

  std::unique_ptr<HybridEdge[]> edges_;
  std::atomic<int> num_edges_{0};

  // Hot path statistics - grouped for cache efficiency
  alignas(64) std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};
  std::atomic<float> w_{0.0f};
  std::atomic<float> q_{0.0f};
  std::atomic<float> d_{0.0f};
  std::atomic<float> m_{0.0f};

  std::atomic<Terminal> terminal_type_{Terminal::NonTerminal};
  std::atomic<bool> has_ab_score_{false};
  std::atomic<int> ab_score_{0};
  std::atomic<int> ab_depth_{0};
};

// Tree with arena-based allocation
class HybridTree {
public:
  HybridTree();
  ~HybridTree();

  void reset(const MCTSPositionHistory &history);
  bool apply_move(MCTSMove move);

  HybridNode *root() { return root_; }
  const HybridNode *root() const { return root_; }

  const MCTSPositionHistory &history() const { return history_; }
  MCTSPositionHistory &history() { return history_; }

  std::string root_fen() const {
    std::shared_lock<std::shared_mutex> lock(fen_mutex_);
    return root_fen_;
  }

  HybridNode *allocate_node(HybridNode *parent, int edge_index);
  size_t node_count() const {
    return node_count_.load(std::memory_order_relaxed);
  }

private:
  HybridNode *root_ = nullptr;
  MCTSPositionHistory history_;
  std::string root_fen_;
  mutable std::shared_mutex fen_mutex_;
  std::atomic<size_t> node_count_{0};

  // Arena-based allocation
  static constexpr size_t ARENA_SIZE = 4096;
  struct NodeArena {
    std::unique_ptr<HybridNode[]> nodes;
    std::atomic<size_t> next{0};
    NodeArena() : nodes(std::make_unique<HybridNode[]>(ARENA_SIZE)) {}
  };

  std::vector<std::unique_ptr<NodeArena>> arenas_;
  std::atomic<size_t> current_arena_{0};
  std::mutex arena_mutex_;
};

// Lock-free evaluation request
struct alignas(64) HybridEvalRequest {
  GPU::GPUPositionData pos_data;
  uint64_t position_key = 0;
  Color side_to_move = WHITE;
  float result = 0.0f;
  std::atomic<bool> ready{false};
  std::atomic<bool> completed{false};
};

// Pre-allocated request pool
class HybridEvalRequestPool {
public:
  static constexpr size_t POOL_SIZE = 4096;

  HybridEvalRequestPool() {
    requests_ = std::make_unique<HybridEvalRequest[]>(POOL_SIZE);
  }

  HybridEvalRequest *acquire() {
    uint64_t idx =
        next_idx_.fetch_add(1, std::memory_order_relaxed) % POOL_SIZE;
    HybridEvalRequest *req = &requests_[idx];
    while (req->ready.load(std::memory_order_acquire) &&
           !req->completed.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    req->ready.store(false, std::memory_order_relaxed);
    req->completed.store(false, std::memory_order_relaxed);
    return req;
  }

  void release(HybridEvalRequest *req) {
    req->completed.store(true, std::memory_order_release);
  }

private:
  std::unique_ptr<HybridEvalRequest[]> requests_;
  std::atomic<uint64_t> next_idx_{0};
};

// Batched GPU evaluator for hybrid search
class HybridBatchedEvaluator {
public:
  HybridBatchedEvaluator(GPU::GPUNNUEManager *gpu_manager,
                         HybridSearchStats *stats, int min_batch_size,
                         int max_batch_size, int batch_timeout_us);
  ~HybridBatchedEvaluator();

  void start();
  void stop();

  float evaluate(const MCTSPosition &pos, uint64_t &cache_hits,
                 uint64_t &cache_misses);

  void new_search() { current_age_.fetch_add(1, std::memory_order_relaxed); }

  static constexpr size_t TT_SIZE = 1 << 22; // 4M entries (~64MB)

private:
  void eval_thread_main();
  void process_batch(std::vector<HybridEvalRequest *> &batch);

  GPU::GPUNNUEManager *gpu_manager_;
  HybridSearchStats *stats_;
  int min_batch_size_;
  int max_batch_size_;
  int batch_timeout_us_;

  HybridEvalRequestPool request_pool_;
  std::vector<HybridEvalRequest *> pending_requests_;
  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;

  std::thread eval_thread_;
  std::atomic<bool> running_{false};

  struct alignas(16) TTEntry {
    uint64_t key = 0;
    float value = 0.0f;
    uint32_t age = 0;
  };
  std::vector<TTEntry> tt_;
  std::atomic<uint32_t> current_age_{0};
};

// Worker context for thread-local state
struct HybridWorkerContext {
  std::string cached_root_fen;
  std::mt19937 rng;

  uint64_t iterations = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;

  // Profiling accumulators
  uint64_t selection_time_acc = 0;
  uint64_t expansion_time_acc = 0;
  uint64_t evaluation_time_acc = 0;
  uint64_t backprop_time_acc = 0;

  // Pre-allocated PUCT scores
  std::vector<float> puct_scores;

  HybridWorkerContext() : rng(std::random_device{}()) {
    puct_scores.reserve(256);
  }

  void set_root_fen(const std::string &fen) { cached_root_fen = fen; }
};

// Main hybrid search class
class HybridSearch {
public:
  using BestMoveCallback = std::function<void(MCTSMove, MCTSMove)>;
  using InfoCallback = std::function<void(const std::string &)>;

  HybridSearch(const HybridSearchConfig &config = HybridSearchConfig());
  ~HybridSearch();

  void set_neural_network(std::shared_ptr<MCTSNeuralNetwork> nn);
  void set_gpu_nnue(GPU::GPUNNUEManager *gpu_nnue);

  void start_search(const MCTSPositionHistory &history,
                    const Search::LimitsType &limits,
                    BestMoveCallback best_move_cb,
                    InfoCallback info_cb = nullptr);

  void stop();
  void wait();

  MCTSMove get_best_move() const;
  std::vector<MCTSMove> get_pv() const;
  float get_best_move_q() const;

  const HybridSearchStats &stats() const { return stats_; }
  const HybridSearchConfig &config() const { return config_; }
  void set_config(const HybridSearchConfig &config) { config_ = config; }

  HybridTree *tree() { return tree_.get(); }
  const HybridTree *tree() const { return tree_.get(); }

private:
  void search_thread_main(int thread_id);

  HybridNode *select_node(HybridNode *node, MCTSPosition &pos,
                          HybridWorkerContext &ctx);
  int select_child_puct(HybridNode *node, float cpuct,
                        HybridWorkerContext &ctx);
  void expand_node(HybridNode *node, const MCTSPosition &pos,
                   HybridWorkerContext &ctx);
  void backpropagate(HybridNode *node, float value, float draw_prob,
                     float moves_left);

  float evaluate_position(const MCTSPosition &pos, HybridWorkerContext &ctx);
  float evaluate_position_batched(const MCTSPosition &pos,
                                  HybridWorkerContext &ctx);
  float evaluate_position_direct(const MCTSPosition &pos,
                                 HybridWorkerContext &ctx);

  bool should_use_alphabeta(HybridNode *node, const MCTSPosition &pos);
  int alphabeta_verify(const MCTSPosition &pos, int depth, int alpha, int beta);

  bool should_stop() const;
  int64_t get_time_budget_ms() const;

  MCTSMove select_best_move() const;
  void update_info();

  HybridSearchConfig config_;
  HybridSearchStats stats_;

  std::unique_ptr<HybridTree> tree_;
  std::shared_ptr<MCTSNeuralNetwork> neural_network_;
  GPU::GPUNNUEManager *gpu_nnue_ = nullptr;

  std::vector<std::thread> search_threads_;
  std::vector<std::unique_ptr<HybridWorkerContext>> worker_contexts_;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> running_{false};

  std::unique_ptr<HybridBatchedEvaluator> batched_evaluator_;
  std::mutex gpu_mutex_;

  // Simple TT for direct evaluation mode
  struct SimpleTTEntry {
    uint64_t key = 0;
    float value = 0;
  };
  std::vector<SimpleTTEntry> simple_tt_;
  static constexpr size_t SIMPLE_TT_SIZE = 1 << 20;

  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;

  Search::LimitsType limits_;
  std::chrono::steady_clock::time_point search_start_;
  int64_t time_budget_ms_ = 0;
};

std::unique_ptr<HybridSearch>
create_hybrid_search(GPU::GPUNNUEManager *gpu_nnue,
                     const HybridSearchConfig &config = HybridSearchConfig());

} // namespace MCTS
} // namespace MetalFish

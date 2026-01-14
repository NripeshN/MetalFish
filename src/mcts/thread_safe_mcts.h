/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Thread-Safe MCTS Implementation - Optimized for Apple Silicon

  This module provides a fully thread-safe MCTS implementation optimized for
  Apple Silicon's unified memory architecture and Metal GPU compute.

  Key optimizations:
  1. Lock-free tree traversal with virtual loss
  2. Thread-local position management (no shared Position objects)
  3. High-performance batched GPU evaluation with lock-free queue
  4. Arena-based node allocation to reduce contention
  5. Unified memory optimization - zero-copy GPU access
  6. Adaptive strategy selection based on thread count

  Licensed under GPL-3.0
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "../core/movegen.h"
#include "../core/position.h"
#include "../core/types.h"
#include "../gpu/gpu_nnue_integration.h"
#include "../search/search.h"

namespace MetalFish {
namespace MCTS {

// Forward declarations
class ThreadSafeNode;
class ThreadSafeTree;
class ThreadSafeMCTS;
class BatchedGPUEvaluator;

// ============================================================================
// Thread-Safe Node - Cache-line aligned for optimal memory access
// ============================================================================

struct alignas(64) TSEdge {
  Move move = Move::none();
  std::atomic<float> policy{0.0f};
  std::atomic<ThreadSafeNode *> child{nullptr};
  char padding[64 - sizeof(Move) - sizeof(std::atomic<float>) -
               sizeof(std::atomic<ThreadSafeNode *>)];

  TSEdge() = default;
  TSEdge(Move m, float p) : move(m), policy(p), child(nullptr) {}

  TSEdge(const TSEdge &) = delete;
  TSEdge &operator=(const TSEdge &) = delete;

  TSEdge(TSEdge &&other) noexcept
      : move(other.move), policy(other.policy.load(std::memory_order_relaxed)),
        child(other.child.load(std::memory_order_relaxed)) {}
};

class alignas(64) ThreadSafeNode {
public:
  enum class Terminal : uint8_t {
    NonTerminal = 0,
    Win = 1,
    Draw = 2,
    Loss = 3
  };

  ThreadSafeNode(ThreadSafeNode *parent = nullptr, int edge_idx = -1);
  ~ThreadSafeNode() = default;

  ThreadSafeNode(const ThreadSafeNode &) = delete;
  ThreadSafeNode &operator=(const ThreadSafeNode &) = delete;

  ThreadSafeNode *parent() const { return parent_; }
  int edge_index() const { return edge_index_; }

  bool has_children() const {
    return num_edges_.load(std::memory_order_acquire) > 0;
  }
  int num_edges() const { return num_edges_.load(std::memory_order_acquire); }
  TSEdge *edges() { return edges_.get(); }
  const TSEdge *edges() const { return edges_.get(); }

  void create_edges(const MoveList<LEGAL> &moves);

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

  std::mutex &mutex() { return mutex_; }

  void reset_parent() {
    parent_ = nullptr;
    edge_index_ = -1;
  }

private:
  ThreadSafeNode *parent_;
  int edge_index_;

  std::unique_ptr<TSEdge[]> edges_;
  std::atomic<int> num_edges_{0};

  // Hot path statistics - cache-line aligned
  alignas(64) std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};
  std::atomic<float> q_{0.0f};
  std::atomic<float> d_{0.0f};
  std::atomic<float> m_{0.0f};
  std::atomic<float> w_{0.0f};

  std::atomic<Terminal> terminal_type_{Terminal::NonTerminal};
  mutable std::mutex mutex_;
};

// ============================================================================
// Thread-Safe Tree with Arena Allocation
// ============================================================================

class ThreadSafeTree {
public:
  ThreadSafeTree();
  ~ThreadSafeTree();

  void reset(const std::string &fen);

  ThreadSafeNode *root() { return root_.get(); }
  const ThreadSafeNode *root() const { return root_.get(); }

  std::string root_fen() const {
    std::shared_lock<std::shared_mutex> lock(fen_mutex_);
    return root_fen_;
  }

  ThreadSafeNode *allocate_node(ThreadSafeNode *parent, int edge_idx);

  size_t node_count() const {
    return node_count_.load(std::memory_order_relaxed);
  }

private:
  std::unique_ptr<ThreadSafeNode> root_;
  std::string root_fen_;
  mutable std::shared_mutex fen_mutex_;

  std::atomic<size_t> node_count_{0};

  // Arena-based allocation for reduced contention
  static constexpr size_t ARENA_SIZE = 4096;
  struct NodeArena {
    std::unique_ptr<ThreadSafeNode[]> nodes;
    std::atomic<size_t> next{0};

    NodeArena() : nodes(std::make_unique<ThreadSafeNode[]>(ARENA_SIZE)) {}
  };

  std::vector<std::unique_ptr<NodeArena>> arenas_;
  std::atomic<size_t> current_arena_{0};
  std::mutex arena_mutex_;
};

// ============================================================================
// Worker Context - Thread-local state
// ============================================================================

struct WorkerContext {
  Position pos;
  StateInfo root_st;
  std::vector<StateInfo> state_stack;
  std::vector<Move> move_stack;
  std::mt19937 rng;

  uint64_t iterations = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;

  // Pre-allocated score buffer for PUCT
  std::vector<float> puct_scores;

  // Cached root FEN to avoid repeated string copies
  std::string cached_root_fen;

  // Profiling samples (reduce chrono overhead by sampling)
  static constexpr int PROFILE_SAMPLE_RATE = 64;
  uint64_t selection_time_acc = 0;
  uint64_t expansion_time_acc = 0;
  uint64_t evaluation_time_acc = 0;
  uint64_t backprop_time_acc = 0;

  WorkerContext() : rng(std::random_device{}()) {
    state_stack.reserve(256);
    move_stack.reserve(256);
    puct_scores.reserve(256);
  }

  void set_root_fen(const std::string &fen) {
    cached_root_fen = fen;
    reset_position(fen);
  }

  void reset_position(const std::string &fen) {
    state_stack.clear();
    move_stack.clear();
    pos.set(fen, false, &root_st);
  }

  void reset_to_cached_root() {
    state_stack.clear();
    move_stack.clear();
    pos.set(cached_root_fen, false, &root_st);
  }

  void do_move(Move m) {
    state_stack.emplace_back();
    pos.do_move(m, state_stack.back());
    move_stack.push_back(m);
  }

  void reset_to_root() {
    while (!move_stack.empty()) {
      pos.undo_move(move_stack.back());
      move_stack.pop_back();
      state_stack.pop_back();
    }
  }
};

// ============================================================================
// MCTS Configuration with Auto-Tuning
// ============================================================================

struct ThreadSafeMCTSConfig {
  float cpuct = 2.5f;
  float fpu_value = -1.0f;
  float policy_softmax_temp = 1.0f;
  bool add_dirichlet_noise = true;
  float dirichlet_alpha = 0.3f;
  float dirichlet_epsilon = 0.25f;

  int num_threads = 4;
  int virtual_loss = 3;

  int get_num_threads() const {
    if (num_threads <= 0) {
      // Auto-select optimal thread count based on benchmarks
      // Testing shows 2-4 threads is optimal for GPU batching
      int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
      // Use 2-4 threads for best GPU batching efficiency
      return std::min(std::max(2, hw_threads / 4), 4);
    }
    return num_threads;
  }

  // Batching parameters - auto-tuned based on thread count
  int min_batch_size = 1; // Process immediately if available
  int max_batch_size = 256;
  int batch_timeout_us = 25; // Very short timeout - process quickly
  bool use_batched_eval = true;

  // Auto-tune batch parameters based on thread count
  void auto_tune(int actual_threads) {
    if (actual_threads <= 2) {
      // Low thread count: minimize latency
      min_batch_size = 1;
      batch_timeout_us = 10;
      max_batch_size = 32;
    } else if (actual_threads <= 4) {
      // Medium thread count: balance latency and throughput
      min_batch_size = 1;
      batch_timeout_us = 25;
      max_batch_size = 64;
    } else {
      // High thread count: more requests arrive, can batch more
      min_batch_size = 2;
      batch_timeout_us = 50;
      max_batch_size = 128;
    }
  }

  int64_t max_time_ms = 0;
  int64_t max_nodes = 0;
};

// ============================================================================
// Thread-Safe MCTS Statistics
// ============================================================================

struct ThreadSafeMCTSStats {
  std::atomic<uint64_t> total_nodes{0};
  std::atomic<uint64_t> total_iterations{0};
  std::atomic<uint64_t> cache_hits{0};
  std::atomic<uint64_t> cache_misses{0};
  std::atomic<uint64_t> nn_evaluations{0};
  std::atomic<uint64_t> nn_batches{0};

  std::atomic<uint64_t> selection_time_us{0};
  std::atomic<uint64_t> expansion_time_us{0};
  std::atomic<uint64_t> evaluation_time_us{0};
  std::atomic<uint64_t> backprop_time_us{0};

  std::atomic<uint64_t> batch_wait_time_us{0};
  std::atomic<uint64_t> total_batch_size{0};
  std::atomic<uint64_t> batch_count{0};

  void reset() {
    total_nodes = 0;
    total_iterations = 0;
    cache_hits = 0;
    cache_misses = 0;
    nn_evaluations = 0;
    nn_batches = 0;
    selection_time_us = 0;
    expansion_time_us = 0;
    evaluation_time_us = 0;
    backprop_time_us = 0;
    batch_wait_time_us = 0;
    total_batch_size = 0;
    batch_count = 0;
  }

  uint64_t nps(double elapsed_s) const {
    return elapsed_s > 0 ? static_cast<uint64_t>(total_nodes / elapsed_s) : 0;
  }

  double avg_batch_size() const {
    uint64_t count = batch_count.load(std::memory_order_relaxed);
    return count > 0 ? static_cast<double>(
                           total_batch_size.load(std::memory_order_relaxed)) /
                           count
                     : 0;
  }
};

// ============================================================================
// High-Performance Batched GPU Evaluator
// ============================================================================

// Lock-free evaluation request using atomic flag
struct alignas(64) EvalRequest {
  GPU::GPUPositionData pos_data;
  uint64_t position_key = 0;
  Color side_to_move = WHITE;
  float result = 0.0f;
  std::atomic<bool> ready{false};
  std::atomic<bool> completed{false};
};

// Pre-allocated request pool for zero-allocation evaluation
class EvalRequestPool {
public:
  static constexpr size_t POOL_SIZE = 8192; // Increased for better distribution

  EvalRequestPool() { requests_ = std::make_unique<EvalRequest[]>(POOL_SIZE); }

  EvalRequest *acquire() {
    // Use thread-local hint for better cache locality
    thread_local uint64_t local_hint = 0;
    uint64_t start_idx = local_hint;

    // Try a few slots before falling back to atomic increment
    for (int tries = 0; tries < 4; ++tries) {
      uint64_t idx = (start_idx + tries) % POOL_SIZE;
      EvalRequest *req = &requests_[idx];

      // Fast path: check if slot is available
      if (!req->ready.load(std::memory_order_acquire) ||
          req->completed.load(std::memory_order_acquire)) {
        // Try to claim it
        bool expected_ready = false;
        if (req->ready.compare_exchange_strong(expected_ready, false,
                                               std::memory_order_acquire)) {
          req->completed.store(false, std::memory_order_relaxed);
          local_hint = idx + 1;
          return req;
        }
      }
    }

    // Fallback: use atomic counter
    uint64_t idx =
        next_idx_.fetch_add(1, std::memory_order_relaxed) % POOL_SIZE;
    EvalRequest *req = &requests_[idx];

    // Wait if slot is still in use (rare with larger pool)
    int spin = 0;
    while (req->ready.load(std::memory_order_acquire) &&
           !req->completed.load(std::memory_order_acquire)) {
      if (++spin > 100) {
        std::this_thread::yield();
        spin = 0;
      }
    }

    req->ready.store(false, std::memory_order_relaxed);
    req->completed.store(false, std::memory_order_relaxed);
    local_hint = idx + 1;
    return req;
  }

  void release(EvalRequest *req) {
    req->completed.store(true, std::memory_order_release);
  }

private:
  std::unique_ptr<EvalRequest[]> requests_;
  std::atomic<uint64_t> next_idx_{0};
};

class BatchedGPUEvaluator {
public:
  BatchedGPUEvaluator(GPU::GPUNNUEManager *gpu_manager,
                      ThreadSafeMCTSStats *stats, int min_batch_size = 1,
                      int max_batch_size = 256, int batch_timeout_us = 50);
  ~BatchedGPUEvaluator();

  void start();
  void stop();

  // Synchronous evaluation (blocks until result ready)
  float evaluate(const Position &pos, WorkerContext &ctx);

  // Asynchronous evaluation with callback
  // Returns immediately, callback is invoked when evaluation completes
  // The callback receives the evaluation value
  using EvalCallback = std::function<void(float value)>;
  void evaluate_async(const Position &pos, uint64_t key, Color stm,
                      EvalCallback callback);

  // Check if async mode is enabled
  bool async_enabled() const { return use_async_mode_; }
  void set_async_mode(bool enabled) { use_async_mode_ = enabled; }

  // Increment age for TT replacement policy
  void new_search() { current_age_.fetch_add(1, std::memory_order_relaxed); }

  // Get number of pending async evaluations
  size_t pending_async_count() const {
    return pending_async_count_.load(std::memory_order_relaxed);
  }

private:
  void eval_thread_main();
  void process_batch(std::vector<EvalRequest *> &batch);
  void process_batch_async(std::vector<EvalRequest *> &batch);

  GPU::GPUNNUEManager *gpu_manager_;
  ThreadSafeMCTSStats *stats_;

  int min_batch_size_;
  int max_batch_size_;
  int batch_timeout_us_;
  bool use_async_mode_ = false; // Enable async GPU submission

  // Lock-free request submission
  EvalRequestPool request_pool_;
  std::vector<EvalRequest *> pending_requests_;
  std::mutex pending_mutex_;
  std::condition_variable pending_cv_;

  std::thread eval_thread_;
  std::atomic<bool> running_{false};

  // Double-buffering: prepare next batch while current processes
  std::vector<EvalRequest *> prefetch_buffer_;
  std::mutex prefetch_mutex_;

  // Async evaluation support
  struct AsyncEvalRequest {
    GPU::GPUPositionData pos_data;
    uint64_t position_key;
    Color side_to_move;
    EvalCallback callback;
  };
  std::vector<AsyncEvalRequest> async_requests_;
  std::mutex async_mutex_;
  std::atomic<size_t> pending_async_count_{0};

  // Multiple in-flight GPU batches for true async
  static constexpr int MAX_INFLIGHT_BATCHES = 4;
  std::atomic<int> inflight_batches_{0};

  // TT for caching evaluations - larger cache for better hit rates
  struct alignas(16) TTEntry {
    uint64_t key = 0;
    float value = 0.0f;
    uint32_t age = 0; // For replacement policy
  };
  static constexpr size_t TT_SIZE = 1 << 22; // 4M entries (~64MB)
  std::vector<TTEntry> tt_;
  std::atomic<uint32_t> current_age_{0};
};

// ============================================================================
// Thread-Safe MCTS Search
// ============================================================================

class ThreadSafeMCTS {
public:
  using BestMoveCallback = std::function<void(Move best, Move ponder)>;
  using InfoCallback = std::function<void(const std::string &)>;

  ThreadSafeMCTS(const ThreadSafeMCTSConfig &config = ThreadSafeMCTSConfig());
  ~ThreadSafeMCTS();

  void set_gpu_manager(GPU::GPUNNUEManager *gpu) { gpu_manager_ = gpu; }

  void start_search(const std::string &fen, const Search::LimitsType &limits,
                    BestMoveCallback best_move_cb = nullptr,
                    InfoCallback info_cb = nullptr);

  void stop();
  void wait();

  Move get_best_move() const;
  std::vector<Move> get_pv() const;
  const ThreadSafeMCTSStats &stats() const { return stats_; }
  float get_best_q() const;

private:
  void worker_thread(int thread_id);
  void run_iteration(WorkerContext &ctx);

  ThreadSafeNode *select_leaf(WorkerContext &ctx);
  void expand_node(ThreadSafeNode *node, WorkerContext &ctx);

  float evaluate_position(WorkerContext &ctx);
  float evaluate_position_batched(WorkerContext &ctx);
  float evaluate_position_direct(WorkerContext &ctx);

  void backpropagate(ThreadSafeNode *node, float value, float draw,
                     float moves_left);

  // Optimized PUCT selection
  int select_child_puct(ThreadSafeNode *node, float cpuct, WorkerContext &ctx);

  void add_dirichlet_noise(ThreadSafeNode *root);
  bool should_stop() const;
  void send_info();
  int64_t calculate_time_budget() const;

  ThreadSafeMCTSConfig config_;
  std::unique_ptr<ThreadSafeTree> tree_;
  GPU::GPUNNUEManager *gpu_manager_ = nullptr;

  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> running_{false};
  Search::LimitsType limits_;
  std::chrono::steady_clock::time_point search_start_;
  int64_t time_budget_ms_ = 0;

  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;

  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<WorkerContext>> worker_contexts_;

  ThreadSafeMCTSStats stats_;

  std::unique_ptr<BatchedGPUEvaluator> batched_evaluator_;
  std::mutex gpu_mutex_;

  struct SimpleTTEntry {
    uint64_t key = 0;
    float value = 0;
  };
  std::vector<SimpleTTEntry> simple_tt_;
  static constexpr size_t SIMPLE_TT_SIZE = 1 << 20;
};

std::unique_ptr<ThreadSafeMCTS> create_thread_safe_mcts(
    GPU::GPUNNUEManager *gpu_manager,
    const ThreadSafeMCTSConfig &config = ThreadSafeMCTSConfig());

} // namespace MCTS
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Thread-Safe MCTS Implementation

  This module provides a fully thread-safe MCTS implementation optimized for
  Apple Silicon's unified memory architecture and Metal GPU compute.

  Key features:
  1. Lock-free tree traversal with virtual loss
  2. Thread-local position management (no shared Position objects)
  3. Batched GPU evaluation with async dispatch
  4. Fine-grained locking only for tree modifications
  5. Unified memory optimization - zero-copy GPU access

  Licensed under GPL-3.0
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
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
class MCTSWorker;

// ============================================================================
// Thread-Safe Node
// ============================================================================

// Edge stores move and policy prior
struct alignas(8) TSEdge {
  Move move = Move::none();
  std::atomic<float> policy{0.0f};
  std::atomic<ThreadSafeNode*> child{nullptr};
  
  TSEdge() = default;
  TSEdge(Move m, float p) : move(m), policy(p), child(nullptr) {}
  
  // Non-copyable due to atomics
  TSEdge(const TSEdge&) = delete;
  TSEdge& operator=(const TSEdge&) = delete;
  
  // Move constructor for vector resize
  TSEdge(TSEdge&& other) noexcept 
    : move(other.move), 
      policy(other.policy.load(std::memory_order_relaxed)),
      child(other.child.load(std::memory_order_relaxed)) {}
};

// Node with lock-free statistics updates
class ThreadSafeNode {
public:
  enum class Terminal : uint8_t { 
    NonTerminal = 0, 
    Win = 1, 
    Draw = 2, 
    Loss = 3 
  };

  ThreadSafeNode(ThreadSafeNode* parent = nullptr, int edge_idx = -1);
  ~ThreadSafeNode() = default;
  
  // Non-copyable
  ThreadSafeNode(const ThreadSafeNode&) = delete;
  ThreadSafeNode& operator=(const ThreadSafeNode&) = delete;
  
  // Tree structure
  ThreadSafeNode* parent() const { return parent_; }
  int edge_index() const { return edge_index_; }
  
  // Edge management (requires mutex for modification)
  bool has_children() const { return num_edges_.load(std::memory_order_acquire) > 0; }
  int num_edges() const { return num_edges_.load(std::memory_order_acquire); }
  TSEdge* edges() { return edges_.get(); }
  const TSEdge* edges() const { return edges_.get(); }
  
  // Create edges (must be called under lock)
  void create_edges(const MoveList<LEGAL>& moves);
  
  // Lock-free statistics access
  uint32_t n() const { return n_.load(std::memory_order_acquire); }
  uint32_t n_in_flight() const { return n_in_flight_.load(std::memory_order_acquire); }
  float q() const { return q_.load(std::memory_order_acquire); }
  float d() const { return d_.load(std::memory_order_acquire); }
  float m() const { return m_.load(std::memory_order_acquire); }
  
  // Virtual loss for multi-threading
  void add_virtual_loss(int count = 1) {
    n_in_flight_.fetch_add(count, std::memory_order_acq_rel);
  }
  
  void remove_virtual_loss(int count = 1) {
    n_in_flight_.fetch_sub(count, std::memory_order_acq_rel);
  }
  
  // Lock-free statistics update using CAS
  void update_stats(float value, float draw_prob, float moves_left);
  
  // Terminal state
  Terminal terminal_type() const { return terminal_type_.load(std::memory_order_acquire); }
  bool is_terminal() const { return terminal_type() != Terminal::NonTerminal; }
  void set_terminal(Terminal type, float value);
  
  // Get mutex for tree modifications
  std::mutex& mutex() { return mutex_; }
  
  // Reset for tree reuse
  void reset_parent() { parent_ = nullptr; edge_index_ = -1; }
  
private:
  ThreadSafeNode* parent_;
  int edge_index_;
  
  // Edges stored as unique_ptr to array
  std::unique_ptr<TSEdge[]> edges_;
  std::atomic<int> num_edges_{0};
  
  // Lock-free statistics
  std::atomic<uint32_t> n_{0};
  std::atomic<uint32_t> n_in_flight_{0};
  std::atomic<float> q_{0.0f};
  std::atomic<float> d_{0.0f};
  std::atomic<float> m_{0.0f};
  std::atomic<float> w_{0.0f};  // Sum of values for averaging
  
  std::atomic<Terminal> terminal_type_{Terminal::NonTerminal};
  
  // Mutex only for tree modifications
  mutable std::mutex mutex_;
};

// ============================================================================
// Thread-Safe Tree
// ============================================================================

class ThreadSafeTree {
public:
  ThreadSafeTree();
  ~ThreadSafeTree();
  
  // Initialize tree with position
  void reset(const std::string& fen);
  
  // Get root
  ThreadSafeNode* root() { return root_.get(); }
  const ThreadSafeNode* root() const { return root_.get(); }
  
  // Get root FEN (thread-safe)
  std::string root_fen() const {
    std::shared_lock<std::shared_mutex> lock(fen_mutex_);
    return root_fen_;
  }
  
  // Allocate new node (thread-safe)
  ThreadSafeNode* allocate_node(ThreadSafeNode* parent, int edge_idx);
  
  // Statistics
  size_t node_count() const { return node_count_.load(std::memory_order_relaxed); }
  
private:
  std::unique_ptr<ThreadSafeNode> root_;
  std::string root_fen_;
  mutable std::shared_mutex fen_mutex_;
  
  std::atomic<size_t> node_count_{0};
  
  // Node pool for efficient allocation
  std::vector<std::unique_ptr<ThreadSafeNode>> node_pool_;
  std::mutex pool_mutex_;
};

// ============================================================================
// MCTS Worker Thread
// ============================================================================

// Thread-local workspace for each worker
struct WorkerContext {
  // Thread-local position (recreated from FEN each iteration)
  Position pos;
  StateInfo root_st;
  std::vector<StateInfo> state_stack;
  std::vector<Move> move_stack;
  
  // Random number generator
  std::mt19937 rng;
  
  // Statistics
  uint64_t iterations = 0;
  uint64_t cache_hits = 0;
  uint64_t cache_misses = 0;
  
  WorkerContext() : rng(std::random_device{}()) {
    state_stack.reserve(256);
    move_stack.reserve(256);
  }
  
  // Reset position to FEN
  void reset_position(const std::string& fen) {
    state_stack.clear();
    move_stack.clear();
    pos.set(fen, false, &root_st);
  }
  
  // Make move
  void do_move(Move m) {
    state_stack.emplace_back();
    pos.do_move(m, state_stack.back());
    move_stack.push_back(m);
  }
  
  // Undo all moves back to root
  void reset_to_root() {
    while (!move_stack.empty()) {
      pos.undo_move(move_stack.back());
      move_stack.pop_back();
      state_stack.pop_back();
    }
  }
};

// ============================================================================
// MCTS Configuration
// ============================================================================

struct ThreadSafeMCTSConfig {
  // MCTS parameters
  float cpuct = 2.5f;
  float fpu_value = -1.0f;
  float policy_softmax_temp = 1.0f;
  bool add_dirichlet_noise = true;
  float dirichlet_alpha = 0.3f;
  float dirichlet_epsilon = 0.25f;
  
  // Threading
  int num_threads = 4;
  int virtual_loss = 3;  // Virtual loss per visit
  
  // Batching for GPU
  int min_batch_size = 8;
  int max_batch_size = 256;
  int batch_timeout_us = 1000;
  
  // Time management
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
  
  // Profiling (microseconds)
  std::atomic<uint64_t> selection_time_us{0};
  std::atomic<uint64_t> expansion_time_us{0};
  std::atomic<uint64_t> evaluation_time_us{0};
  std::atomic<uint64_t> backprop_time_us{0};
  
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
  }
  
  uint64_t nps(double elapsed_s) const {
    return elapsed_s > 0 ? static_cast<uint64_t>(total_nodes / elapsed_s) : 0;
  }
};

// ============================================================================
// Thread-Safe MCTS Search
// ============================================================================

class ThreadSafeMCTS {
public:
  using BestMoveCallback = std::function<void(Move best, Move ponder)>;
  using InfoCallback = std::function<void(const std::string&)>;
  
  ThreadSafeMCTS(const ThreadSafeMCTSConfig& config = ThreadSafeMCTSConfig());
  ~ThreadSafeMCTS();
  
  // Initialize with GPU manager
  void set_gpu_manager(GPU::GPUNNUEManager* gpu) { gpu_manager_ = gpu; }
  
  // Start search
  void start_search(const std::string& fen, 
                    const Search::LimitsType& limits,
                    BestMoveCallback best_move_cb = nullptr,
                    InfoCallback info_cb = nullptr);
  
  // Stop search
  void stop();
  
  // Wait for search to complete
  void wait();
  
  // Get best move
  Move get_best_move() const;
  
  // Get PV
  std::vector<Move> get_pv() const;
  
  // Get statistics
  const ThreadSafeMCTSStats& stats() const { return stats_; }
  
  // Get Q value of best move
  float get_best_q() const;
  
private:
  // Worker thread function
  void worker_thread(int thread_id);
  
  // MCTS iteration (called by worker)
  void run_iteration(WorkerContext& ctx);
  
  // Selection phase - traverse tree to leaf
  ThreadSafeNode* select_leaf(WorkerContext& ctx);
  
  // Expansion phase - add children to leaf
  void expand_node(ThreadSafeNode* node, WorkerContext& ctx);
  
  // Evaluation phase - get value from GPU NNUE
  float evaluate_position(WorkerContext& ctx);
  
  // Backpropagation phase - update statistics
  void backpropagate(ThreadSafeNode* node, float value, float draw, float moves_left);
  
  // Select best child using PUCT
  int select_child_puct(ThreadSafeNode* node, float cpuct);
  
  // Add Dirichlet noise to root
  void add_dirichlet_noise(ThreadSafeNode* root);
  
  // Check if search should stop
  bool should_stop() const;
  
  // Send UCI info
  void send_info();
  
  // Calculate time budget
  int64_t calculate_time_budget() const;
  
  ThreadSafeMCTSConfig config_;
  std::unique_ptr<ThreadSafeTree> tree_;
  GPU::GPUNNUEManager* gpu_manager_ = nullptr;
  
  // Search state
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> running_{false};
  Search::LimitsType limits_;
  std::chrono::steady_clock::time_point search_start_;
  int64_t time_budget_ms_ = 0;
  
  // Callbacks
  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;
  
  // Worker threads
  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<WorkerContext>> worker_contexts_;
  
  // Statistics
  ThreadSafeMCTSStats stats_;
  
  // GPU mutex for thread-safe evaluation
  std::mutex gpu_mutex_;
  
  // Transposition table for caching evaluations
  struct TTEntry {
    uint64_t key = 0;
    float value = 0;
    float draw = 0;
    float moves_left = 0;
  };
  std::vector<TTEntry> tt_;
  static constexpr size_t TT_SIZE = 1 << 20;  // 1M entries
};

// Factory function
std::unique_ptr<ThreadSafeMCTS> create_thread_safe_mcts(
    GPU::GPUNNUEManager* gpu_manager,
    const ThreadSafeMCTSConfig& config = ThreadSafeMCTSConfig());

} // namespace MCTS
} // namespace MetalFish

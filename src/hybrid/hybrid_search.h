/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Parallel Hybrid Search - MCTS and Alpha-Beta running simultaneously
  on Apple Silicon with unified memory for zero-copy data sharing.

  Licensed under GPL-3.0
*/

#pragma once

#include "../eval/gpu_backend.h"
#include "../eval/gpu_integration.h"
#include "../mcts/gpu_backend.h"
#include "../mcts/search.h"
#include "../search/search.h"
#include "../search/tt.h"
#include "classifier.h"
#include "shared_tt.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace MetalFish {

class Engine;

namespace MCTS {

constexpr size_t APPLE_CACHE_LINE_SIZE = 128;

// Lock-free structure for AB to communicate best moves to MCTS
struct alignas(APPLE_CACHE_LINE_SIZE) ABSharedState {
  std::atomic<uint32_t> best_move_raw{0};
  std::atomic<int32_t> best_score{0};
  std::atomic<int32_t> completed_depth{0};
  std::atomic<uint64_t> nodes_searched{0};

  std::atomic<uint64_t> update_counter{0};
  std::atomic<bool> ab_running{false};
  std::atomic<bool> has_result{false};

  void reset() {
    best_move_raw.store(0, std::memory_order_relaxed);
    best_score.store(0, std::memory_order_relaxed);
    completed_depth.store(0, std::memory_order_relaxed);
    nodes_searched.store(0, std::memory_order_relaxed);
    update_counter.store(0, std::memory_order_relaxed);
    ab_running.store(false, std::memory_order_relaxed);
    has_result.store(false, std::memory_order_relaxed);
    pv_length.store(0, std::memory_order_relaxed);
    pv_depth.store(0, std::memory_order_relaxed);
    for (int i = 0; i < MAX_PV; ++i) {
      pv_moves[i].store(0, std::memory_order_relaxed);
    }
  }

  Move get_best_move() const {
    uint32_t raw = best_move_raw.load(std::memory_order_acquire);
    return Move(raw);
  }

  void set_best_move(Move m, int score, int depth, uint64_t nodes) {
    best_move_raw.store(m.raw(), std::memory_order_relaxed);
    best_score.store(score, std::memory_order_relaxed);
    completed_depth.store(depth, std::memory_order_relaxed);
    nodes_searched.store(nodes, std::memory_order_relaxed);
    has_result.store(true, std::memory_order_release);
    update_counter.fetch_add(1, std::memory_order_release);
  }

  // PV from AB iterative deepening, injected into the MCTS tree in real-time
  static constexpr int MAX_PV = 16;
  std::atomic<uint16_t> pv_moves[MAX_PV]{};
  std::atomic<int> pv_length{0};
  std::atomic<int> pv_depth{0};

  void publish_pv(const std::vector<Move> &pv, int depth) {
    int len = std::min(static_cast<int>(pv.size()), MAX_PV);
    for (int i = 0; i < len; ++i) {
      pv_moves[i].store(pv[i].raw(), std::memory_order_relaxed);
    }
    pv_depth.store(depth, std::memory_order_relaxed);
    pv_length.store(len, std::memory_order_release);
  }
};

struct alignas(APPLE_CACHE_LINE_SIZE) MCTSSharedState {
  std::atomic<uint32_t> best_move_raw{0};
  std::atomic<float> best_q{0.0f};
  std::atomic<uint32_t> best_visits{0};
  std::atomic<uint64_t> total_nodes{0};

  static constexpr int MAX_TOP_MOVES = 10;
  struct TopMove {
    std::atomic<uint32_t> move_raw{0};
    std::atomic<float> policy{0.0f};
    std::atomic<uint32_t> visits{0};
    std::atomic<float> q{0.0f};
  };
  TopMove top_moves[MAX_TOP_MOVES];
  std::atomic<int> num_top_moves{0};

  std::atomic<uint64_t> update_counter{0};
  std::atomic<bool> mcts_running{false};
  std::atomic<bool> has_result{false};

  void reset() {
    best_move_raw.store(0, std::memory_order_relaxed);
    best_q.store(0.0f, std::memory_order_relaxed);
    best_visits.store(0, std::memory_order_relaxed);
    total_nodes.store(0, std::memory_order_relaxed);
    num_top_moves.store(0, std::memory_order_relaxed);
    update_counter.store(0, std::memory_order_relaxed);
    mcts_running.store(false, std::memory_order_relaxed);
    has_result.store(false, std::memory_order_relaxed);
    for (int i = 0; i < MAX_TOP_MOVES; ++i) {
      top_moves[i].move_raw.store(0, std::memory_order_relaxed);
      top_moves[i].policy.store(0.0f, std::memory_order_relaxed);
      top_moves[i].visits.store(0, std::memory_order_relaxed);
      top_moves[i].q.store(0.0f, std::memory_order_relaxed);
    }
  }

  Move get_best_move() const {
    uint32_t raw = best_move_raw.load(std::memory_order_acquire);
    return Move(raw);
  }
};

// ============================================================================
// Statistics
// ============================================================================

struct ParallelSearchStats {
  std::atomic<uint64_t> mcts_nodes{0};
  std::atomic<uint64_t> mcts_iterations{0};
  std::atomic<uint64_t> gpu_evaluations{0};
  std::atomic<uint64_t> gpu_batches{0};

  std::atomic<uint64_t> ab_nodes{0};
  std::atomic<uint64_t> ab_depth{0};
  std::atomic<uint64_t> ab_iterations{0};

  std::atomic<uint64_t> policy_updates{0};
  std::atomic<uint64_t> move_agreements{0};
  std::atomic<uint64_t> ab_overrides{0};
  std::atomic<uint64_t> mcts_overrides{0};

  // Timing
  double mcts_time_ms = 0;
  double ab_time_ms = 0;
  double total_time_ms = 0;

  void reset() {
    mcts_nodes = 0;
    mcts_iterations = 0;
    gpu_evaluations = 0;
    gpu_batches = 0;
    ab_nodes = 0;
    ab_depth = 0;
    ab_iterations = 0;
    policy_updates = 0;
    move_agreements = 0;
    ab_overrides = 0;
    mcts_overrides = 0;
    mcts_time_ms = 0;
    ab_time_ms = 0;
    total_time_ms = 0;
  }
};

// ============================================================================
// Configuration
// ============================================================================

struct ParallelHybridConfig {
  SearchParams mcts_config;
  int mcts_threads = 4;
  int ab_threads = 4;

  int ab_min_depth = 8;
  int ab_max_depth = 64;
  bool ab_use_time = true;

  float ab_policy_weight = 0.3f;
  float agreement_threshold = 0.3f;
  float override_threshold = 1.0f;
  int policy_update_interval_ms = 50;

  float time_fraction = 0.05f;
  float max_time_fraction = 0.20f;
  float increment_usage = 0.75f;

  bool use_position_classifier = true;

  // GPU batch evaluation
  int gpu_batch_size = 128;
  int gpu_batch_timeout_us = 200;
  bool use_async_gpu_eval = true;

  // Unified memory optimizations
  bool use_gpu_resident_batches = true;
  bool prefetch_positions = true;

  // Metal-specific
  bool use_simd_kernels = true;
  int metal_threadgroup_size = 256;

  enum class DecisionMode {
    MCTS_PRIMARY,  // Trust MCTS unless AB strongly disagrees
    AB_PRIMARY,    // Trust AB unless MCTS strongly disagrees
    VOTE_WEIGHTED, // Weighted combination based on confidence
    DYNAMIC        // Choose based on position type
  };
  DecisionMode decision_mode = DecisionMode::DYNAMIC;
};

// ============================================================================
// Parallel Hybrid Search Engine
// ============================================================================

class ParallelHybridSearch {
public:
  using BestMoveCallback = std::function<void(Move, Move)>;
  using InfoCallback = std::function<void(const std::string &)>;

  ParallelHybridSearch();
  ~ParallelHybridSearch();

  ParallelHybridSearch(const ParallelHybridSearch &) = delete;
  ParallelHybridSearch &operator=(const ParallelHybridSearch &) = delete;
  ParallelHybridSearch(ParallelHybridSearch &&) = delete;
  ParallelHybridSearch &operator=(ParallelHybridSearch &&) = delete;

  bool initialize(GPU::GPUNNUEManager *gpu_manager, Engine *engine);
  bool is_ready() const { return initialized_; }

  // Configuration
  void set_config(const ParallelHybridConfig &config) { config_ = config; }
  const ParallelHybridConfig &config() const { return config_; }

  void start_search(const Position &pos, const ::MetalFish::Search::LimitsType &limits,
                    BestMoveCallback best_move_cb,
                    InfoCallback info_cb = nullptr);
  void stop();
  void wait();
  bool is_searching() const {
    return searching_.load(std::memory_order_acquire);
  }

  // Results
  const ParallelSearchStats &stats() const { return stats_; }
  Move get_best_move() const;
  Move get_ponder_move() const;

  void new_game();

private:
  bool initialized_ = false;
  ParallelHybridConfig config_;
  ParallelSearchStats stats_;

  // MCTS search engine
  std::unique_ptr<Search> mcts_search_;
  std::unique_ptr<GPU::GPUMCTSBackend> gpu_backend_;
  std::unique_ptr<SharedTTReader> shared_tt_reader_;
  GPU::GPUNNUEManager *gpu_manager_ = nullptr;
  Engine *engine_ = nullptr;

  PositionClassifier classifier_;
  StrategySelector strategy_selector_;
  SearchStrategy current_strategy_;

  std::unordered_map<uint16_t, float> nn_policy_hints_;

  // Shared state (cache-line aligned)
  ABSharedState ab_state_;
  MCTSSharedState mcts_state_;

  // Thread management
  enum class ThreadState { IDLE, RUNNING, STOPPING };

  // Centralized thread control
  std::mutex thread_mutex_;
  std::condition_variable thread_cv_;
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> searching_{false};
  std::atomic<bool> shutdown_requested_{false};

  std::thread mcts_thread_;
  std::thread ab_thread_;
  std::thread coordinator_thread_;

  std::atomic<ThreadState> mcts_thread_state_{ThreadState::IDLE};
  std::atomic<ThreadState> ab_thread_state_{ThreadState::IDLE};
  std::atomic<ThreadState> coordinator_thread_state_{ThreadState::IDLE};

  std::atomic<bool> mcts_thread_done_{true};
  std::atomic<bool> ab_thread_done_{true};
  std::atomic<bool> coordinator_thread_done_{true};

  // Search state
  std::string root_fen_;
  ::MetalFish::Search::LimitsType limits_;
  std::chrono::steady_clock::time_point search_start_;
  int time_budget_ms_ = 0;

  // Final result
  std::atomic<uint32_t> final_best_move_{0};
  std::atomic<uint32_t> final_ponder_move_{0};
  std::vector<Move> final_pv_;
  mutable std::mutex pv_mutex_;

  std::mutex callback_mutex_;
  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;
  std::atomic<bool> callback_invoked_{false};

  // Thread functions
  void mcts_thread_main();
  void ab_thread_main();
  void coordinator_thread_main();

  void update_mcts_policy_from_ab();
  void publish_mcts_state();

  void run_ab_search();
  void publish_ab_state(Move best, int score, int depth, uint64_t nodes);

  Move make_final_decision();
  void refresh_final_state(Move final_move);
  int calculate_time_budget() const;
  bool should_stop() const;

  void join_all_threads();
  void signal_thread_done(std::atomic<bool> &done_flag);
  bool all_threads_done() const;

  void send_info(int depth, int score, uint64_t nodes, int time_ms,
                 const std::vector<Move> &pv, const std::string &source);
  void send_info_string(const std::string &msg);

  void invoke_best_move_callback(Move best, Move ponder);
};

// Factory function
std::unique_ptr<ParallelHybridSearch> create_parallel_hybrid_search(
    GPU::GPUNNUEManager *gpu_manager, Engine *engine,
    const ParallelHybridConfig &config = ParallelHybridConfig());

// Returns true when the coordinator, not AB's time manager, owns the outer
// search budget and should keep MCTS running after AB has produced a result.
bool HybridShouldContinueMCTSAfterAB(
    const ::MetalFish::Search::LimitsType &limits);

} // namespace MCTS
} // namespace MetalFish

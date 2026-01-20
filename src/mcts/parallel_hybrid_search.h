/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Parallel Hybrid Search - MCTS and Alpha-Beta running simultaneously
  
  Optimized for Apple Silicon with unified memory architecture:
  - Zero-copy data sharing between CPU and GPU
  - Lock-free atomic communication using cache-line aligned structures
  - Async GPU evaluation with completion handlers
  - Metal-accelerated batch operations

  Architecture:
  1. MCTS threads continuously explore the tree using GPU NNUE
  2. AB thread runs iterative deepening in parallel
  3. AB results dynamically update MCTS policy priors
  4. Both contribute to the final move selection
  5. Lock-free communication via atomic shared state in unified memory

  Key innovations:
  - AB search informs MCTS policy in real-time
  - MCTS exploration guides AB move ordering
  - Shared transposition table between both searches
  - Dynamic time allocation based on agreement
  - GPU-resident evaluation batches for zero-copy

  Licensed under GPL-3.0
*/

#pragma once

#include "../gpu/backend.h"
#include "../gpu/gpu_mcts_backend.h"
#include "../gpu/gpu_nnue_integration.h"
#include "../search/search.h"
#include "../search/tt.h"
#include "ab_integration.h"
#include "hybrid_search.h"
#include "position_classifier.h"
#include "stockfish_adapter.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace MetalFish {

class Engine;

namespace MCTS {

// ============================================================================
// Apple Silicon Unified Memory Optimizations
// ============================================================================

// Cache line size for optimal alignment (Apple Silicon uses 128-byte cache lines)
constexpr size_t APPLE_CACHE_LINE_SIZE = 128;

// GPU-resident evaluation batch for zero-copy on unified memory
struct GPUResidentBatch {
  // Positions stored in GPU-accessible buffer (unified memory = zero copy)
  std::unique_ptr<GPU::Buffer> positions_buffer;
  std::unique_ptr<GPU::Buffer> results_buffer;
  
  // Metadata (small, kept on CPU)
  std::vector<uint32_t> position_indices;
  int count = 0;
  int capacity = 0;
  
  bool initialized = false;
  
  // Initialize with capacity
  bool initialize(int batch_capacity);
  
  // Clear for reuse (doesn't deallocate)
  void clear() { count = 0; position_indices.clear(); }
  
  // Check if batch is full
  bool full() const { return count >= capacity; }
};

// ============================================================================
// Shared State for Parallel Communication (Cache-line aligned)
// ============================================================================

// Lock-free structure for AB to communicate best moves to MCTS
// Aligned to 128 bytes (Apple Silicon cache line) to prevent false sharing
struct alignas(APPLE_CACHE_LINE_SIZE) ABSharedState {
  // Current best move from AB (updated atomically)
  std::atomic<uint32_t> best_move_raw{0};  // Move encoded as uint32_t
  std::atomic<int32_t> best_score{0};
  std::atomic<int32_t> completed_depth{0};
  std::atomic<uint64_t> nodes_searched{0};
  
  // Policy updates from AB (move scores for MCTS policy adjustment)
  static constexpr int MAX_MOVES = 256;
  struct MoveScore {
    std::atomic<uint32_t> move_raw{0};
    std::atomic<int32_t> score{-32001};  // VALUE_NONE equivalent
    std::atomic<int32_t> depth{0};
  };
  MoveScore move_scores[MAX_MOVES];
  std::atomic<int> num_scored_moves{0};
  
  // Synchronization
  std::atomic<uint64_t> update_counter{0};  // Incremented on each AB update
  std::atomic<bool> ab_running{false};
  std::atomic<bool> has_result{false};
  
  void reset() {
    best_move_raw.store(0, std::memory_order_relaxed);
    best_score.store(0, std::memory_order_relaxed);
    completed_depth.store(0, std::memory_order_relaxed);
    nodes_searched.store(0, std::memory_order_relaxed);
    num_scored_moves.store(0, std::memory_order_relaxed);
    update_counter.store(0, std::memory_order_relaxed);
    ab_running.store(false, std::memory_order_relaxed);
    has_result.store(false, std::memory_order_relaxed);
    for (int i = 0; i < MAX_MOVES; ++i) {
      move_scores[i].move_raw.store(0, std::memory_order_relaxed);
      move_scores[i].score.store(-32001, std::memory_order_relaxed);
      move_scores[i].depth.store(0, std::memory_order_relaxed);
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
  
  void update_move_score(Move m, int score, int depth) {
    int idx = num_scored_moves.load(std::memory_order_relaxed);
    if (idx >= MAX_MOVES) return;
    
    // Check if move already exists
    for (int i = 0; i < idx; ++i) {
      if (move_scores[i].move_raw.load(std::memory_order_relaxed) == m.raw()) {
        // Update if deeper
        if (depth > move_scores[i].depth.load(std::memory_order_relaxed)) {
          move_scores[i].score.store(score, std::memory_order_relaxed);
          move_scores[i].depth.store(depth, std::memory_order_release);
        }
        return;
      }
    }
    
    // Add new move
    int new_idx = num_scored_moves.fetch_add(1, std::memory_order_relaxed);
    if (new_idx < MAX_MOVES) {
      move_scores[new_idx].move_raw.store(m.raw(), std::memory_order_relaxed);
      move_scores[new_idx].score.store(score, std::memory_order_relaxed);
      move_scores[new_idx].depth.store(depth, std::memory_order_release);
    }
  }
};

// Lock-free structure for MCTS to communicate to AB
// Lock-free structure for MCTS to communicate to AB
// Aligned to cache line to prevent false sharing
struct alignas(APPLE_CACHE_LINE_SIZE) MCTSSharedState {
  // Best move from MCTS (by visit count)
  std::atomic<uint32_t> best_move_raw{0};
  std::atomic<float> best_q{0.0f};
  std::atomic<uint32_t> best_visits{0};
  std::atomic<uint64_t> total_nodes{0};
  
  // Top moves for AB to prioritize
  static constexpr int MAX_TOP_MOVES = 10;
  struct TopMove {
    std::atomic<uint32_t> move_raw{0};
    std::atomic<float> policy{0.0f};
    std::atomic<uint32_t> visits{0};
    std::atomic<float> q{0.0f};
  };
  TopMove top_moves[MAX_TOP_MOVES];
  std::atomic<int> num_top_moves{0};
  
  // Synchronization
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
// Statistics for Parallel Search
// ============================================================================

struct ParallelSearchStats {
  // MCTS stats
  std::atomic<uint64_t> mcts_nodes{0};
  std::atomic<uint64_t> mcts_iterations{0};
  std::atomic<uint64_t> gpu_evaluations{0};
  std::atomic<uint64_t> gpu_batches{0};
  
  // AB stats
  std::atomic<uint64_t> ab_nodes{0};
  std::atomic<uint64_t> ab_depth{0};
  std::atomic<uint64_t> ab_iterations{0};
  
  // Interaction stats
  std::atomic<uint64_t> policy_updates{0};      // Times AB updated MCTS policy
  std::atomic<uint64_t> move_agreements{0};     // Times MCTS and AB agreed
  std::atomic<uint64_t> ab_overrides{0};        // Times AB overrode MCTS
  std::atomic<uint64_t> mcts_overrides{0};      // Times MCTS overrode AB
  
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
  // MCTS configuration
  HybridSearchConfig mcts_config;
  int mcts_threads = 1;
  
  // AB configuration  
  int ab_min_depth = 8;       // Minimum depth for AB to search
  int ab_max_depth = 64;      // Maximum depth (iterative deepening)
  bool ab_use_time = true;    // Use time management for AB
  
  // Parallel coordination
  float ab_policy_weight = 0.3f;        // Weight of AB scores in MCTS policy
  float agreement_threshold = 0.3f;     // Pawns - if within this, moves agree
  float override_threshold = 1.0f;      // Pawns - AB overrides MCTS if exceeds
  int policy_update_interval_ms = 50;   // How often to update MCTS policy from AB
  
  // Time allocation
  float time_fraction = 0.05f;          // Base fraction of remaining time
  float max_time_fraction = 0.20f;      // Maximum fraction
  float increment_usage = 0.75f;        // How much of increment to use
  
  // Position-based strategy
  bool use_position_classifier = true;
  
  // =========================================================================
  // Apple Silicon / GPU Optimizations
  // =========================================================================
  
  // GPU batch evaluation settings
  int gpu_batch_size = 128;             // Optimal batch size for M-series GPUs
  int gpu_batch_timeout_us = 200;       // Max wait time to fill batch (microseconds)
  bool use_async_gpu_eval = true;       // Use async GPU evaluation with callbacks
  
  // Unified memory optimizations
  bool use_gpu_resident_batches = true; // Keep batches in unified memory
  bool prefetch_positions = true;       // Prefetch next batch while evaluating
  
  // Metal-specific optimizations
  bool use_simd_kernels = true;         // Use SIMD-optimized Metal kernels
  int metal_threadgroup_size = 256;     // Threads per threadgroup
  
  // Final decision
  enum class DecisionMode {
    MCTS_PRIMARY,     // Trust MCTS unless AB strongly disagrees
    AB_PRIMARY,       // Trust AB unless MCTS strongly disagrees
    VOTE_WEIGHTED,    // Weighted combination based on confidence
    DYNAMIC           // Choose based on position type
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

  // Initialization
  bool initialize(GPU::GPUNNUEManager *gpu_manager, Engine *engine);
  bool is_ready() const { return initialized_; }

  // Configuration
  void set_config(const ParallelHybridConfig &config) { config_ = config; }
  const ParallelHybridConfig &config() const { return config_; }

  // Main search interface
  void start_search(const Position &pos, const Search::LimitsType &limits,
                    BestMoveCallback best_move_cb,
                    InfoCallback info_cb = nullptr);
  void stop();
  void wait();
  bool is_searching() const { return searching_.load(std::memory_order_acquire); }

  // Results
  const ParallelSearchStats &stats() const { return stats_; }
  Move get_best_move() const;
  Move get_ponder_move() const;

  // Tree management
  void new_game();
  void apply_move(Move move);

private:
  bool initialized_ = false;
  ParallelHybridConfig config_;
  ParallelSearchStats stats_;

  // Components
  std::unique_ptr<HybridSearch> mcts_search_;
  std::unique_ptr<GPU::GPUMCTSBackend> gpu_backend_;
  GPU::GPUNNUEManager *gpu_manager_ = nullptr;
  Engine *engine_ = nullptr;

  // Position classifier
  PositionClassifier classifier_;
  StrategySelector strategy_selector_;
  SearchStrategy current_strategy_;

  // Shared state for parallel communication (cache-line aligned)
  ABSharedState ab_state_;
  MCTSSharedState mcts_state_;

  // =========================================================================
  // Apple Silicon GPU Optimization State
  // =========================================================================
  
  // GPU-resident evaluation batches (double-buffered for async)
  GPUResidentBatch gpu_batch_[2];
  std::atomic<int> current_batch_{0};
  std::atomic<bool> batch_pending_{false};
  
  // Async evaluation state
  std::mutex async_mutex_;
  std::condition_variable async_cv_;
  std::atomic<int> pending_evaluations_{0};
  
  // Unified memory info
  bool has_unified_memory_ = false;

  // Threading
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> searching_{false};
  std::thread mcts_thread_;
  std::thread ab_thread_;
  std::thread coordinator_thread_;

  // Search state
  std::string root_fen_;
  Search::LimitsType limits_;
  std::chrono::steady_clock::time_point search_start_;
  int time_budget_ms_ = 0;

  // Final result
  std::atomic<uint32_t> final_best_move_{0};
  std::atomic<uint32_t> final_ponder_move_{0};
  std::vector<Move> final_pv_;
  mutable std::mutex pv_mutex_;

  // Callbacks
  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;

  // Thread functions
  void mcts_thread_main();
  void ab_thread_main();
  void coordinator_thread_main();

  // MCTS helpers
  void update_mcts_policy_from_ab();
  void publish_mcts_state();

  // AB helpers
  void run_ab_search();
  void publish_ab_state(Move best, int score, int depth, uint64_t nodes);

  // Coordination
  Move make_final_decision();
  int calculate_time_budget() const;
  bool should_stop() const;
  
  // =========================================================================
  // GPU Optimization Helpers
  // =========================================================================
  
  // Initialize GPU-resident batches for zero-copy evaluation
  bool initialize_gpu_batches();
  
  // Submit batch for async GPU evaluation
  void submit_gpu_batch_async(int batch_idx, 
                              std::function<void(bool)> completion_handler);
  
  // Swap to next batch (double-buffering)
  int swap_batch();
  
  // Wait for pending GPU evaluations
  void wait_gpu_evaluations();

  // Info output
  void send_info(int depth, int score, uint64_t nodes, int time_ms,
                 const std::vector<Move> &pv, const std::string &source);
  void send_info_string(const std::string &msg);
};

// Factory function
std::unique_ptr<ParallelHybridSearch>
create_parallel_hybrid_search(GPU::GPUNNUEManager *gpu_manager, Engine *engine,
                              const ParallelHybridConfig &config = ParallelHybridConfig());

} // namespace MCTS
} // namespace MetalFish

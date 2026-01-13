/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Enhanced Hybrid Search
  
  This is the core hybrid search that combines:
  1. MCTS for strategic move selection at the root
  2. Alpha-beta for tactical verification of MCTS moves
  3. Dynamic strategy switching based on position type
  4. GPU batch evaluation for throughput
  5. Transposition table sharing between MCTS and AB

  The key innovation is using alpha-beta to verify MCTS selections,
  catching tactical oversights while preserving strategic understanding.

  Licensed under GPL-3.0
*/

#pragma once

#include "stockfish_adapter.h"
#include "position_classifier.h"
#include "hybrid_search.h"
#include "../search/search.h"
#include "../search/tt.h"
#include "../gpu/gpu_nnue_integration.h"
#include "../gpu/gpu_mcts_backend.h"
#include <atomic>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace MetalFish {
namespace MCTS {

// Forward declarations
class EnhancedHybridSearch;

// Result from alpha-beta verification
struct ABVerifyResult {
  Move best_move;
  int score;
  int depth;
  bool is_mate = false;
  int mate_in = 0;
  std::vector<Move> pv;
  
  // Comparison with MCTS result
  bool agrees_with_mcts = true;
  float score_difference = 0.0f;  // In pawns
};

// Combined evaluation result
struct HybridEvalResult {
  MCTSMove mcts_best;
  float mcts_score;        // Q value from MCTS
  uint32_t mcts_visits;
  
  Move ab_best;
  int ab_score;            // Centipawns from AB
  int ab_depth;
  
  Move final_best;         // Combined decision
  float confidence;        // How confident we are (0-1)
  
  std::string pv_string;
  SearchStrategy strategy;
};

// Statistics for enhanced hybrid search
struct EnhancedSearchStats {
  std::atomic<uint64_t> mcts_nodes{0};
  std::atomic<uint64_t> ab_nodes{0};
  std::atomic<uint64_t> ab_verifications{0};
  std::atomic<uint64_t> ab_overrides{0};      // Times AB overrode MCTS
  std::atomic<uint64_t> tactical_fallbacks{0}; // Times we fell back to AB
  std::atomic<uint64_t> gpu_batches{0};
  std::atomic<uint64_t> gpu_positions{0};
  std::atomic<uint64_t> tt_hits{0};
  std::atomic<uint64_t> tt_misses{0};
  
  double mcts_time_ms = 0;
  double ab_time_ms = 0;
  double total_time_ms = 0;
  
  void reset() {
    mcts_nodes = 0;
    ab_nodes = 0;
    ab_verifications = 0;
    ab_overrides = 0;
    tactical_fallbacks = 0;
    gpu_batches = 0;
    gpu_positions = 0;
    tt_hits = 0;
    tt_misses = 0;
    mcts_time_ms = 0;
    ab_time_ms = 0;
    total_time_ms = 0;
  }
};

// Configuration for enhanced hybrid search
struct EnhancedHybridConfig {
  // Base MCTS config
  HybridSearchConfig mcts_config;
  
  // Alpha-beta verification
  bool enable_ab_verify = true;
  int ab_verify_depth = 6;
  int ab_verify_nodes = 10000;
  float ab_override_threshold = 0.5f;  // Pawns difference to override
  
  // Position-based strategy
  bool use_position_classifier = true;
  bool dynamic_strategy = true;
  
  // GPU batching
  int gpu_batch_size = 64;
  int gpu_batch_timeout_us = 500;
  
  // Time management
  float base_time_fraction = 0.025f;  // 2.5% of remaining time
  float max_time_fraction = 0.10f;    // Max 10% of remaining time
  float increment_factor = 0.8f;      // How much of increment to use
  
  // Transposition table
  bool share_tt = true;
  
  // Threading
  int mcts_threads = 1;
  int ab_threads = 1;
};

// Enhanced hybrid search engine
class EnhancedHybridSearch {
public:
  using BestMoveCallback = std::function<void(Move, Move)>;  // bestmove, ponder
  using InfoCallback = std::function<void(const std::string&)>;
  
  EnhancedHybridSearch();
  ~EnhancedHybridSearch();
  
  // Initialization
  bool initialize(GPU::GPUNNUEManager* gpu_manager);
  bool is_ready() const { return initialized_; }
  
  // Configuration
  void set_config(const EnhancedHybridConfig& config) { config_ = config; }
  const EnhancedHybridConfig& config() const { return config_; }
  
  // Main search interface
  void start_search(const Position& pos,
                    const Search::LimitsType& limits,
                    BestMoveCallback best_move_cb,
                    InfoCallback info_cb = nullptr);
  
  void stop();
  void wait();
  
  // Get results
  HybridEvalResult get_result() const { return result_; }
  const EnhancedSearchStats& stats() const { return stats_; }
  
  // Transposition table access (for sharing with main search)
  void set_tt(TranspositionTable* tt) { tt_ = tt; }
  
private:
  bool initialized_ = false;
  EnhancedHybridConfig config_;
  EnhancedSearchStats stats_;
  HybridEvalResult result_;
  
  // Components
  std::unique_ptr<HybridSearch> mcts_search_;
  std::unique_ptr<GPU::GPUMCTSBackend> gpu_backend_;
  GPU::GPUNNUEManager* gpu_manager_ = nullptr;
  TranspositionTable* tt_ = nullptr;
  
  // Position classifier
  PositionClassifier classifier_;
  StrategySelector strategy_selector_;
  
  // Threading
  std::atomic<bool> stop_flag_{false};
  std::atomic<bool> searching_{false};
  std::thread search_thread_;
  
  // Callbacks
  BestMoveCallback best_move_callback_;
  InfoCallback info_callback_;
  
  // Current search state
  std::string root_fen_;
  Search::LimitsType limits_;
  SearchStrategy current_strategy_;
  
  // Internal methods
  void search_thread_main();
  
  // MCTS phase
  MCTSMove run_mcts_phase(const MCTSPositionHistory& history, int time_ms);
  
  // Alpha-beta verification
  ABVerifyResult verify_with_alphabeta(const Position& pos, MCTSMove mcts_move, int depth);
  
  // Combined decision
  Move make_final_decision(MCTSMove mcts_move, const ABVerifyResult& ab_result);
  
  // Time management
  int calculate_time_budget(const Position& pos) const;
  int calculate_mcts_time(int total_time) const;
  int calculate_ab_time(int total_time) const;
  
  // Info output
  void send_info(int depth, int score, uint64_t nodes, int time_ms, 
                 const std::vector<Move>& pv);
  void send_info_string(const std::string& msg);
};

// Factory function
std::unique_ptr<EnhancedHybridSearch> create_enhanced_hybrid_search(
    GPU::GPUNNUEManager* gpu_manager,
    const EnhancedHybridConfig& config = EnhancedHybridConfig());

}  // namespace MCTS
}  // namespace MetalFish

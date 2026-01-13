/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Alpha-Beta Integration for Hybrid Search

  This module provides full integration between MCTS and Stockfish's
  alpha-beta search. It enables:

  1. Using AB search for tactical verification of MCTS moves
  2. Full-depth AB search with proper move ordering
  3. Quiescence search for tactical positions
  4. Integration with Stockfish's transposition table
  5. History heuristics from AB to improve MCTS policy

  Licensed under GPL-3.0
*/

#pragma once

#include "../core/position.h"
#include "../core/types.h"
#include "../eval/evaluate.h"
#include "../gpu/gpu_nnue_integration.h"
#include "../search/history.h"
#include "../search/movepick.h"
#include "../search/search.h"
#include "../search/tt.h"
#include "stockfish_adapter.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace MetalFish {
namespace MCTS {

// Result from alpha-beta search
struct ABSearchResult {
  Move best_move = Move::none();
  Value score = VALUE_NONE;
  Depth depth = 0;
  int nodes_searched = 0;
  bool is_mate = false;
  int mate_in = 0;
  std::vector<Move> pv;
  Bound bound = BOUND_NONE;

  // For comparison with MCTS
  float normalized_score() const {
    if (is_mate) {
      return mate_in > 0 ? 1.0f : -1.0f;
    }
    return std::tanh(static_cast<float>(score) / 400.0f);
  }
};

// Configuration for AB search in hybrid context
struct ABSearchConfig {
  int max_depth = 12;         // Maximum search depth
  int quiescence_depth = 6;   // Quiescence search depth
  int aspiration_window = 25; // Initial aspiration window
  bool use_tt = true;         // Use transposition table
  bool use_null_move = true;  // Use null move pruning
  bool use_lmr = true;        // Use late move reductions
  bool use_futility = true;   // Use futility pruning
  bool use_history = true;    // Use history heuristics
  int64_t max_nodes = 0;      // Node limit (0 = unlimited)
  int64_t max_time_ms = 0;    // Time limit (0 = unlimited)
  bool verify_only = false;   // Only verify a specific move
  Move move_to_verify = Move::none();
};

// Standalone alpha-beta searcher for MCTS verification
class ABSearcher {
public:
  ABSearcher();
  ~ABSearcher();

  // Initialize with transposition table
  void initialize(TranspositionTable *tt);

  // Set configuration
  void set_config(const ABSearchConfig &config) { config_ = config; }

  // Main search entry point
  ABSearchResult search(const Position &pos, int depth);

  // Search with aspiration windows
  ABSearchResult search_aspiration(const Position &pos, int depth,
                                   Value prev_score = VALUE_NONE);

  // Verify a specific move (returns true if move is best or near-best)
  bool verify_move(const Position &pos, Move move, int depth,
                   Value *out_score = nullptr);

  // Get move ordering scores for MCTS policy
  std::vector<std::pair<Move, float>> get_move_scores(const Position &pos);

  // Stop search
  void stop() { stop_flag_ = true; }

  // Statistics
  uint64_t nodes_searched() const { return nodes_; }
  uint64_t tt_hits() const { return tt_hits_; }
  uint64_t tt_cutoffs() const { return tt_cutoffs_; }

  // Reset statistics
  void reset_stats() {
    nodes_ = 0;
    tt_hits_ = 0;
    tt_cutoffs_ = 0;
  }

private:
  // Main search function (PV and non-PV nodes)
  template <bool PvNode>
  Value search_internal(Position &pos, int depth, Value alpha, Value beta,
                        int ply, Move *pv, int &pv_length);

  // Quiescence search
  Value qsearch(Position &pos, Value alpha, Value beta, int ply);

  // Move ordering
  void score_moves(const Position &pos, MoveList<LEGAL> &moves, Move tt_move);

  // Pruning helpers
  bool can_futility_prune(const Position &pos, int depth, Value alpha,
                          Value static_eval) const;
  bool can_null_move_prune(const Position &pos, int depth, Value beta,
                           Value static_eval) const;
  int late_move_reduction(int depth, int move_count, bool improving) const;

  // Evaluation
  Value evaluate(const Position &pos);

  // Time/node management
  bool should_stop() const;

  ABSearchConfig config_;
  TranspositionTable *tt_ = nullptr;

  // Search state
  std::atomic<bool> stop_flag_{false};
  std::chrono::steady_clock::time_point start_time_;

  // Statistics
  std::atomic<uint64_t> nodes_{0};
  std::atomic<uint64_t> tt_hits_{0};
  std::atomic<uint64_t> tt_cutoffs_{0};

  // History tables for move ordering (simplified)
  int16_t main_history_[COLOR_NB][SQUARE_NB][SQUARE_NB];
  int16_t capture_history_[PIECE_NB][SQUARE_NB][PIECE_TYPE_NB];

  // Killer moves
  static constexpr int MAX_PLY = 128;
  Move killers_[MAX_PLY][2];

  // Counter moves
  Move counter_moves_[PIECE_NB][SQUARE_NB];

  // State stack for search
  std::vector<StateInfo> state_stack_;

  // Per-ply static evaluation for improving flag calculation
  Value static_eval_stack_[MAX_PLY];
};

// Policy generator using AB search heuristics
class ABPolicyGenerator {
public:
  ABPolicyGenerator();

  // Initialize with history tables
  void initialize(const ButterflyHistory *main_history,
                  const CapturePieceToHistory *capture_history,
                  TranspositionTable *tt);

  // Generate policy distribution for a position
  std::vector<std::pair<Move, float>> generate_policy(const Position &pos);

  // Update history based on search results
  void update_history(const Position &pos, Move best_move,
                      const std::vector<Move> &searched_moves, int depth);

private:
  const ButterflyHistory *main_history_ = nullptr;
  const CapturePieceToHistory *capture_history_ = nullptr;
  TranspositionTable *tt_ = nullptr;

  // Local history for MCTS-specific patterns (simplified)
  int16_t mcts_history_[COLOR_NB][SQUARE_NB][SQUARE_NB];
};

// Tactical analyzer using quiescence search
class TacticalAnalyzer {
public:
  TacticalAnalyzer();

  // Initialize
  void initialize(TranspositionTable *tt);

  // Analyze position for tactical content
  struct TacticalInfo {
    bool in_check = false;
    bool has_hanging_pieces = false;
    bool has_threats = false;
    bool has_forcing_moves = false;
    int num_captures = 0;
    int num_checks = 0;
    Value tactical_score = VALUE_ZERO;
    Value quiescence_score = VALUE_ZERO;
  };

  TacticalInfo analyze(const Position &pos);

  // Check if position requires deep tactical search
  bool needs_deep_search(const Position &pos);

  // Get SEE scores for all captures
  std::vector<std::pair<Move, Value>> get_capture_scores(const Position &pos);

private:
  TranspositionTable *tt_ = nullptr;
  ABSearcher qsearcher_;
};

// Bridge between MCTS and AB search
class HybridSearchBridge {
public:
  HybridSearchBridge();
  ~HybridSearchBridge();

  // Initialize with shared resources
  void initialize(TranspositionTable *tt, GPU::GPUNNUEManager *gpu_manager);

  // Verify MCTS best move with AB search
  struct VerificationResult {
    bool verified = false;      // MCTS move is verified as good
    bool override_mcts = false; // AB found significantly better move
    Move mcts_move = Move::none();
    Move ab_move = Move::none();
    Value mcts_score = VALUE_NONE;
    Value ab_score = VALUE_NONE;
    float score_difference = 0.0f; // In pawns
    int ab_depth = 0;
    std::vector<Move> ab_pv;
  };

  VerificationResult verify_mcts_move(const Position &pos, Move mcts_move,
                                      int verification_depth,
                                      float override_threshold = 0.5f);

  // Get AB-enhanced policy for MCTS
  std::vector<std::pair<MCTSMove, float>>
  get_enhanced_policy(const Position &pos);

  // Run full AB search for comparison
  ABSearchResult run_ab_search(const Position &pos, int depth, int time_ms = 0);

  // Tactical verification
  bool is_tactical_position(const Position &pos);
  Value get_tactical_score(const Position &pos);

  // Statistics
  struct BridgeStats {
    uint64_t verifications = 0;
    uint64_t overrides = 0;
    uint64_t ab_nodes = 0;
    uint64_t policy_generations = 0;
    double avg_verification_time_ms = 0;
  };

  BridgeStats get_stats() const { return stats_; }
  void reset_stats() { stats_ = BridgeStats(); }

private:
  bool initialized_ = false;
  TranspositionTable *tt_ = nullptr;
  GPU::GPUNNUEManager *gpu_manager_ = nullptr;

  std::unique_ptr<ABSearcher> ab_searcher_;
  std::unique_ptr<ABPolicyGenerator> policy_generator_;
  std::unique_ptr<TacticalAnalyzer> tactical_analyzer_;

  BridgeStats stats_;
  std::mutex stats_mutex_;
};

// Global hybrid search bridge
HybridSearchBridge &hybrid_bridge();
bool initialize_hybrid_bridge(TranspositionTable *tt,
                              GPU::GPUNNUEManager *gpu_manager);

} // namespace MCTS
} // namespace MetalFish

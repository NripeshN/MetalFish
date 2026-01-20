/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Stockfish Search Wrapper for Hybrid MCTS

  This module provides a clean interface to use the full Stockfish alpha-beta
  search implementation within the hybrid MCTS search. Instead of reimplementing
  a simplified AB search, we leverage the battle-tested, highly optimized
  Stockfish search with all its features:

  - Full NNUE evaluation
  - Principal Variation Search (PVS) with aspiration windows
  - Iterative deepening with transposition table
  - Late Move Reductions (LMR) and Late Move Pruning
  - Null Move Pruning, Futility Pruning, Razoring
  - Singular Extensions and Check Extensions
  - History heuristics (butterfly, capture, continuation, pawn)
  - Killer moves and counter moves
  - MVV-LVA move ordering

  Licensed under GPL-3.0
*/

#pragma once

#include "../core/position.h"
#include "../core/types.h"
#include "../search/search.h"
#include "../search/thread.h"
#include "../search/tt.h"
#include "../uci/ucioption.h"
#include "stockfish_adapter.h"
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace MetalFish {

// Forward declarations
class Engine;
class ThreadPool;

namespace MCTS {

// Result from Stockfish search
struct StockfishSearchResult {
  Move best_move = Move::none();
  Move ponder_move = Move::none();
  Value score = VALUE_NONE;
  Depth depth = 0;
  uint64_t nodes_searched = 0;
  bool is_mate = false;
  int mate_in = 0;
  std::vector<Move> pv;

  // For comparison with MCTS
  float normalized_score() const {
    if (is_mate) {
      return mate_in > 0 ? 1.0f : -1.0f;
    }
    // Use Stockfish's WDL model scaling
    return std::tanh(static_cast<float>(score) / 400.0f);
  }

  // Convert to MCTSMove
  MCTSMove best_mcts_move() const {
    return MCTSMove::FromStockfish(best_move);
  }
};

// Configuration for Stockfish search in hybrid context
struct StockfishSearchConfig {
  int max_depth = 12;       // Maximum search depth (0 = use time management)
  int64_t max_time_ms = 0;  // Time limit in ms (0 = unlimited)
  uint64_t max_nodes = 0;   // Node limit (0 = unlimited)
  int num_threads = 1;      // Number of search threads
  bool use_nnue = true;     // Use NNUE evaluation (always true for best play)
  int multi_pv = 1;         // Number of principal variations to search
};

// Wrapper around the full Stockfish search engine
// This provides a clean interface for the hybrid search to use
class StockfishSearchWrapper {
public:
  StockfishSearchWrapper();
  ~StockfishSearchWrapper();

  // Initialize with shared resources
  // Note: This creates its own Engine instance for searching
  bool initialize();

  // Check if initialized
  bool is_ready() const { return initialized_; }

  // Run a search on the given position
  // This uses the full Stockfish search with all optimizations
  StockfishSearchResult search(const Position &pos,
                               const StockfishSearchConfig &config);

  // Run a quick search for move verification
  // Returns the best move and score at the given depth
  StockfishSearchResult quick_search(const Position &pos, int depth,
                                     int time_ms = 0);

  // Verify if a move is the best or near-best
  // Returns true if the move is within threshold of the best move
  bool verify_move(const Position &pos, Move move, int depth,
                   float threshold_pawns = 0.5f, Value *out_score = nullptr);

  // Get move ordering from Stockfish's perspective
  // Uses history heuristics and TT to score moves
  std::vector<std::pair<Move, float>> get_move_policy(const Position &pos,
                                                      int search_depth = 4);

  // Get the score for a specific move
  // Searches the position after making the move
  Value get_move_score(const Position &pos, Move move, int depth);

  // Stop any ongoing search
  void stop();

  // Statistics
  uint64_t total_nodes() const { return total_nodes_.load(); }
  uint64_t total_searches() const { return total_searches_.load(); }
  double avg_search_time_ms() const;

  // Reset statistics
  void reset_stats();

private:
  bool initialized_ = false;

  // We create a lightweight search infrastructure
  // This avoids interfering with the main engine's search
  std::unique_ptr<TranspositionTable> tt_;
  std::unique_ptr<OptionsMap> options_;

  // Mutex for thread safety
  mutable std::mutex search_mutex_;

  // Statistics
  std::atomic<uint64_t> total_nodes_{0};
  std::atomic<uint64_t> total_searches_{0};
  std::atomic<double> total_time_ms_{0};

  // Helper to run search synchronously
  StockfishSearchResult run_search_internal(const std::string &fen,
                                            const Search::LimitsType &limits);
};

// Global Stockfish search wrapper instance
StockfishSearchWrapper &stockfish_search();

// Initialize the global wrapper
bool initialize_stockfish_search();

} // namespace MCTS
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Transposition Table

  This module provides a transposition table optimized for MCTS that can
  share entries with Stockfish's alpha-beta transposition table.

  Key features:
  1. Lock-free access for multiple search threads
  2. Stores both MCTS statistics (Q, N) and AB bounds
  3. Bucket-based replacement strategy
  4. Unified memory layout for GPU access

  Licensed under GPL-3.0
*/

#pragma once

#include "../core/types.h"
#include "../search/tt.h"
#include "stockfish_adapter.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// MCTS TT Entry
// ============================================================================

// Entry type for MCTS-specific data
enum class MCTSTTEntryType : uint8_t {
  NONE = 0,
  MCTS_ONLY = 1, // Only MCTS statistics
  AB_ONLY = 2,   // Only AB bounds
  HYBRID = 3     // Both MCTS and AB data
};

// MCTS statistics stored in TT
struct MCTSStats {
  float q = 0.0f;      // Average value (Q)
  float d = 0.0f;      // Draw probability
  float m = 0.0f;      // Moves left estimate
  uint32_t n = 0;      // Visit count (N)
  uint16_t policy = 0; // Policy prior (fixed point, 0-65535)

  // Convert policy to float
  float policy_float() const { return policy / 65535.0f; }
  void set_policy(float p) { policy = static_cast<uint16_t>(p * 65535.0f); }
};

// Alpha-beta bounds stored in TT
struct ABBounds {
  int16_t score = 0;             // Best score found
  int8_t depth = 0;              // Search depth
  uint8_t bound = 0;             // Bound type (EXACT, LOWER, UPPER)
  Move best_move = Move::none(); // Best move
};

// Combined MCTS TT entry (32 bytes aligned)
struct alignas(32) MCTSTTEntry {
  uint64_t key = 0;                             // Position hash (8 bytes)
  MCTSStats mcts;                               // MCTS statistics (14 bytes)
  ABBounds ab;                                  // AB bounds (8 bytes)
  MCTSTTEntryType type = MCTSTTEntryType::NONE; // Entry type (1 byte)
  uint8_t generation = 0; // Generation for replacement (1 byte)

  bool matches(uint64_t hash) const {
    return key == hash && type != MCTSTTEntryType::NONE;
  }

  bool has_mcts() const {
    return type == MCTSTTEntryType::MCTS_ONLY ||
           type == MCTSTTEntryType::HYBRID;
  }

  bool has_ab() const {
    return type == MCTSTTEntryType::AB_ONLY || type == MCTSTTEntryType::HYBRID;
  }
};

// ============================================================================
// MCTS Transposition Table
// ============================================================================

// Configuration for MCTS TT
struct MCTSTTConfig {
  size_t size_mb = 64;       // Table size in MB
  int num_buckets = 4;       // Entries per bucket
  bool share_with_ab = true; // Share entries with AB search
};

// Statistics for MCTS TT
struct MCTSTTStats {
  std::atomic<uint64_t> lookups{0};
  std::atomic<uint64_t> hits{0};
  std::atomic<uint64_t> mcts_hits{0};
  std::atomic<uint64_t> ab_hits{0};
  std::atomic<uint64_t> stores{0};
  std::atomic<uint64_t> overwrites{0};

  double hit_rate() const {
    uint64_t l = lookups.load();
    return l > 0 ? static_cast<double>(hits.load()) / l : 0.0;
  }

  void reset() {
    lookups = 0;
    hits = 0;
    mcts_hits = 0;
    ab_hits = 0;
    stores = 0;
    overwrites = 0;
  }
};

class MCTSTranspositionTable {
public:
  MCTSTranspositionTable();
  ~MCTSTranspositionTable();

  // Initialize table
  bool initialize(const MCTSTTConfig &config);

  // Clear table
  void clear();

  // New generation (for replacement)
  void new_generation() { ++generation_; }

  // Probe for MCTS entry
  bool probe_mcts(uint64_t key, MCTSStats &stats) const;

  // Probe for AB entry
  bool probe_ab(uint64_t key, ABBounds &bounds) const;

  // Probe for combined entry
  bool probe(uint64_t key, MCTSTTEntry &entry) const;

  // Store MCTS statistics
  void store_mcts(uint64_t key, const MCTSStats &stats);

  // Store AB bounds
  void store_ab(uint64_t key, const ABBounds &bounds);

  // Store combined entry
  void store(uint64_t key, const MCTSTTEntry &entry);

  // Update MCTS statistics (atomic update for visit count and Q)
  void update_mcts(uint64_t key, float value, float draw, float moves_left);

  // Get statistics
  const MCTSTTStats &stats() const { return stats_; }

  // Get table info
  size_t size_bytes() const { return table_size_ * sizeof(MCTSTTEntry); }
  size_t num_entries() const { return table_size_; }
  double usage_percent() const;

  // Link to Stockfish's TT for sharing
  void link_stockfish_tt(TranspositionTable *tt) { stockfish_tt_ = tt; }

private:
  std::vector<MCTSTTEntry> table_;
  size_t table_size_ = 0;
  int num_buckets_ = 4;
  uint8_t generation_ = 0;

  mutable MCTSTTStats stats_;

  // Linked Stockfish TT
  TranspositionTable *stockfish_tt_ = nullptr;

  // Get bucket index
  size_t bucket_index(uint64_t key) const {
    return (key % (table_size_ / num_buckets_)) * num_buckets_;
  }

  // Find or create entry in bucket
  MCTSTTEntry *find_or_create(uint64_t key);
  const MCTSTTEntry *find(uint64_t key) const;
};

// ============================================================================
// TT-Integrated Move Ordering
// ============================================================================

// Move with TT-derived score for ordering
struct TTScoredMove {
  MCTSMove move;
  float score = 0.0f;

  // Score components
  float policy = 0.0f;   // From TT or NN
  float q_bonus = 0.0f;  // From TT Q value
  float ab_bonus = 0.0f; // From TT AB score
  uint32_t visits = 0;   // From TT visit count

  bool operator<(const TTScoredMove &other) const {
    return score > other.score; // Higher score first
  }
};

// Order moves using TT information
class TTMoveOrderer {
public:
  TTMoveOrderer(const MCTSTranspositionTable *tt = nullptr);

  // Order moves for a position
  std::vector<TTScoredMove> order_moves(const MCTSPosition &pos,
                                        const std::vector<MCTSMove> &moves);

  // Get score for a single move
  float score_move(const MCTSPosition &pos, MCTSMove move);

private:
  const MCTSTranspositionTable *tt_;

  // Heuristic scoring
  float capture_score(const MCTSPosition &pos, MCTSMove move);
  float promotion_score(MCTSMove move);
  float center_score(MCTSMove move);
};

// ============================================================================
// Global MCTS TT
// ============================================================================

// Get the global MCTS TT
MCTSTranspositionTable &mcts_tt();

// Initialize the global MCTS TT
bool initialize_mcts_tt(const MCTSTTConfig &config = MCTSTTConfig());

// Shutdown the global MCTS TT
void shutdown_mcts_tt();

} // namespace MCTS
} // namespace MetalFish

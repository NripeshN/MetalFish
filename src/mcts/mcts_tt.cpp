/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS Transposition Table - Implementation

  Licensed under GPL-3.0
*/

#include "mcts_tt.h"
#include <algorithm>
#include <cstring>

namespace MetalFish {
namespace MCTS {

// ============================================================================
// MCTSTranspositionTable
// ============================================================================

MCTSTranspositionTable::MCTSTranspositionTable() = default;

MCTSTranspositionTable::~MCTSTranspositionTable() = default;

bool MCTSTranspositionTable::initialize(const MCTSTTConfig &config) {
  num_buckets_ = config.num_buckets;

  // Calculate table size (round down to multiple of buckets)
  size_t entry_size = sizeof(MCTSTTEntry);
  size_t num_entries = (config.size_mb * 1024 * 1024) / entry_size;
  table_size_ = (num_entries / num_buckets_) * num_buckets_;

  if (table_size_ == 0) {
    return false;
  }

  // Allocate table
  table_.resize(table_size_);
  clear();

  return true;
}

void MCTSTranspositionTable::clear() {
  std::memset(table_.data(), 0, table_.size() * sizeof(MCTSTTEntry));
  generation_ = 0;
  stats_.reset();
}

bool MCTSTranspositionTable::probe_mcts(uint64_t key, MCTSStats &stats) const {
  stats_.lookups++;

  const MCTSTTEntry *entry = find(key);
  if (entry && entry->has_mcts()) {
    stats = entry->mcts;
    stats_.hits++;
    stats_.mcts_hits++;
    return true;
  }

  return false;
}

bool MCTSTranspositionTable::probe_ab(uint64_t key, ABBounds &bounds) const {
  stats_.lookups++;

  const MCTSTTEntry *entry = find(key);
  if (entry && entry->has_ab()) {
    bounds = entry->ab;
    stats_.hits++;
    stats_.ab_hits++;
    return true;
  }

  // Also check Stockfish's TT if linked
  if (stockfish_tt_) {
    auto [found, data, writer] = stockfish_tt_->probe(key);
    if (found) {
      bounds.score = data.value;
      bounds.depth = data.depth;
      bounds.bound = static_cast<uint8_t>(data.bound);
      bounds.best_move = data.move;
      stats_.hits++;
      stats_.ab_hits++;
      return true;
    }
  }

  return false;
}

bool MCTSTranspositionTable::probe(uint64_t key, MCTSTTEntry &entry) const {
  stats_.lookups++;

  const MCTSTTEntry *found = find(key);
  if (found) {
    entry = *found;
    stats_.hits++;
    if (found->has_mcts())
      stats_.mcts_hits++;
    if (found->has_ab())
      stats_.ab_hits++;
    return true;
  }

  return false;
}

void MCTSTranspositionTable::store_mcts(uint64_t key, const MCTSStats &stats) {
  MCTSTTEntry *entry = find_or_create(key);
  if (!entry)
    return;

  bool overwrite = entry->key != 0 && entry->key != key;

  entry->key = key;
  entry->mcts = stats;
  entry->generation = generation_;

  if (entry->type == MCTSTTEntryType::AB_ONLY) {
    entry->type = MCTSTTEntryType::HYBRID;
  } else if (entry->type == MCTSTTEntryType::NONE) {
    entry->type = MCTSTTEntryType::MCTS_ONLY;
  }

  stats_.stores++;
  if (overwrite)
    stats_.overwrites++;
}

void MCTSTranspositionTable::store_ab(uint64_t key, const ABBounds &bounds) {
  MCTSTTEntry *entry = find_or_create(key);
  if (!entry)
    return;

  bool overwrite = entry->key != 0 && entry->key != key;

  entry->key = key;
  entry->ab = bounds;
  entry->generation = generation_;

  if (entry->type == MCTSTTEntryType::MCTS_ONLY) {
    entry->type = MCTSTTEntryType::HYBRID;
  } else if (entry->type == MCTSTTEntryType::NONE) {
    entry->type = MCTSTTEntryType::AB_ONLY;
  }

  stats_.stores++;
  if (overwrite)
    stats_.overwrites++;
}

void MCTSTranspositionTable::store(uint64_t key, const MCTSTTEntry &entry) {
  MCTSTTEntry *slot = find_or_create(key);
  if (!slot)
    return;

  bool overwrite = slot->key != 0 && slot->key != key;

  *slot = entry;
  slot->generation = generation_;

  stats_.stores++;
  if (overwrite)
    stats_.overwrites++;
}

void MCTSTranspositionTable::update_mcts(uint64_t key, float value, float draw,
                                         float moves_left) {
  MCTSTTEntry *entry = find_or_create(key);
  if (!entry)
    return;

  if (entry->key == key && entry->has_mcts()) {
    // Incremental update
    uint32_t n = entry->mcts.n + 1;
    float old_q = entry->mcts.q;
    float old_d = entry->mcts.d;
    float old_m = entry->mcts.m;

    // Running average update
    entry->mcts.q = old_q + (value - old_q) / n;
    entry->mcts.d = old_d + (draw - old_d) / n;
    entry->mcts.m = old_m + (moves_left - old_m) / n;
    entry->mcts.n = n;
  } else {
    // New entry
    entry->key = key;
    entry->mcts.q = value;
    entry->mcts.d = draw;
    entry->mcts.m = moves_left;
    entry->mcts.n = 1;
    entry->type = MCTSTTEntryType::MCTS_ONLY;
  }

  entry->generation = generation_;
}

double MCTSTranspositionTable::usage_percent() const {
  size_t used = 0;
  size_t sample_size = std::min(table_size_, size_t(10000));

  for (size_t i = 0; i < sample_size; ++i) {
    size_t idx = (i * table_size_) / sample_size;
    if (table_[idx].type != MCTSTTEntryType::NONE) {
      ++used;
    }
  }

  return 100.0 * used / sample_size;
}

MCTSTTEntry *MCTSTranspositionTable::find_or_create(uint64_t key) {
  size_t idx = bucket_index(key);
  MCTSTTEntry *bucket = &table_[idx];

  // Look for existing entry or empty slot
  MCTSTTEntry *replace = nullptr;
  int min_score = INT32_MAX;

  for (int i = 0; i < num_buckets_; ++i) {
    MCTSTTEntry *entry = &bucket[i];

    // Exact match
    if (entry->key == key) {
      return entry;
    }

    // Empty slot
    if (entry->type == MCTSTTEntryType::NONE) {
      return entry;
    }

    // Replacement scoring
    int score = 0;

    // Prefer to replace old entries
    int age = (generation_ - entry->generation) & 0xFF;
    score += age * 100;

    // Prefer to replace entries with fewer visits
    if (entry->has_mcts()) {
      score -= std::min(entry->mcts.n, 1000u);
    }

    // Prefer to replace shallow AB entries
    if (entry->has_ab()) {
      score -= entry->ab.depth * 10;
    }

    if (score < min_score) {
      min_score = score;
      replace = entry;
    }
  }

  return replace;
}

const MCTSTTEntry *MCTSTranspositionTable::find(uint64_t key) const {
  size_t idx = bucket_index(key);
  const MCTSTTEntry *bucket = &table_[idx];

  for (int i = 0; i < num_buckets_; ++i) {
    if (bucket[i].matches(key)) {
      return &bucket[i];
    }
  }

  return nullptr;
}

// ============================================================================
// TTMoveOrderer
// ============================================================================

TTMoveOrderer::TTMoveOrderer(const MCTSTranspositionTable *tt) : tt_(tt) {}

std::vector<TTScoredMove>
TTMoveOrderer::order_moves(const MCTSPosition &pos,
                           const std::vector<MCTSMove> &moves) {
  std::vector<TTScoredMove> scored;
  scored.reserve(moves.size());

  for (const auto &move : moves) {
    TTScoredMove sm;
    sm.move = move;
    sm.score = score_move(pos, move);
    scored.push_back(sm);
  }

  // Sort by score (highest first)
  std::sort(scored.begin(), scored.end());

  return scored;
}

float TTMoveOrderer::score_move(const MCTSPosition &pos, MCTSMove move) {
  float score = 0.0f;

  // TT lookup for resulting position
  if (tt_) {
    MCTSPosition next = pos;
    next.do_move(move);

    MCTSStats stats;
    if (tt_->probe_mcts(next.hash(), stats)) {
      // Bonus for visited positions
      score += std::log1p(stats.n) * 0.5f;

      // Bonus for high Q value (from opponent's perspective, so negate)
      score -= stats.q * 2.0f;

      // Policy prior
      score += stats.policy_float() * 3.0f;
    }

    ABBounds bounds;
    if (tt_->probe_ab(next.hash(), bounds)) {
      // Bonus for good AB score (negated for opponent)
      score -= bounds.score / 100.0f;
    }
  }

  // Heuristic bonuses
  score += capture_score(pos, move);
  score += promotion_score(move);
  score += center_score(move);

  return score;
}

float TTMoveOrderer::capture_score(const MCTSPosition &pos, MCTSMove move) {
  const Position &sf_pos = pos.stockfish_position();
  Move m = move.to_stockfish();

  if (!sf_pos.capture(m))
    return 0.0f;

  // MVV-LVA style scoring
  static const float piece_values[] = {0, 1, 3, 3, 5, 9, 0};

  PieceType captured = type_of(sf_pos.piece_on(m.to_sq()));
  PieceType attacker = type_of(sf_pos.piece_on(m.from_sq()));

  // Value of captured - value of attacker / 10
  return piece_values[captured] - piece_values[attacker] * 0.1f;
}

float TTMoveOrderer::promotion_score(MCTSMove move) {
  if (!move.is_promotion())
    return 0.0f;

  Move m = move.to_stockfish();
  PieceType promo = m.promotion_type();

  if (promo == QUEEN)
    return 8.0f;
  if (promo == ROOK)
    return 4.0f;
  if (promo == BISHOP || promo == KNIGHT)
    return 2.0f;

  return 0.0f;
}

float TTMoveOrderer::center_score(MCTSMove move) {
  Move m = move.to_stockfish();
  Square to = m.to_sq();

  int file = file_of(to);
  int rank = rank_of(to);

  // Distance from center
  float file_dist = std::abs(file - 3.5f);
  float rank_dist = std::abs(rank - 3.5f);

  // Small bonus for center control
  return (7.0f - file_dist - rank_dist) * 0.05f;
}

// ============================================================================
// Global Instance
// ============================================================================

static std::unique_ptr<MCTSTranspositionTable> g_mcts_tt;

MCTSTranspositionTable &mcts_tt() {
  if (!g_mcts_tt) {
    g_mcts_tt = std::make_unique<MCTSTranspositionTable>();
    MCTSTTConfig config;
    g_mcts_tt->initialize(config);
  }
  return *g_mcts_tt;
}

bool initialize_mcts_tt(const MCTSTTConfig &config) {
  if (!g_mcts_tt) {
    g_mcts_tt = std::make_unique<MCTSTranspositionTable>();
  }
  return g_mcts_tt->initialize(config);
}

} // namespace MCTS
} // namespace MetalFish

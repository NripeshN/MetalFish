/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Backend Adapter - Wraps NN evaluator for batch computation
  Licensed under GPL-3.0
*/

#include "backend_adapter.h"

#include "../core/movegen.h"

#include <algorithm>
#include <iostream>

namespace MetalFish {
namespace MCTS {

namespace {

constexpr uint64_t kFNVOffset = 1469598103934665603ULL;
constexpr uint64_t kFNVPrime = 1099511628211ULL;

inline uint64_t Mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

inline void MixInto(uint64_t &key, uint64_t value) {
  key ^= Mix64(value);
  key *= kFNVPrime;
}

inline void MixPositionState(uint64_t &key, int history_index,
                             uint64_t raw_key, int rule50_count,
                             int repetition_distance) {
  MixInto(key, raw_key);
  MixInto(key,
          static_cast<uint64_t>(static_cast<uint32_t>(rule50_count)) |
              (static_cast<uint64_t>(history_index) << 32));
  MixInto(key,
          static_cast<uint64_t>(static_cast<int64_t>(repetition_distance)));
}

} // namespace

uint64_t ComputeNNCacheKey(const Position *const *history, int count) {
  if (count <= 0 || !history)
    return kFNVOffset;

  uint64_t key = kFNVOffset;
  MixInto(key, static_cast<uint64_t>(count));
  for (int i = 0; i < count; ++i) {
    const Position *pos = history[i];
    if (!pos) {
      MixInto(key, 0x6a09e667f3bcc909ULL ^ static_cast<uint64_t>(i));
      continue;
    }

    MixPositionState(key, i, pos->raw_key(), pos->rule50_count(),
                     pos->repetition_distance());
  }

  return key;
}

uint64_t ComputeNNCacheKeyFromState(const uint64_t *raw_keys,
                                    const int *rule50_counts,
                                    const int *repetition_distances,
                                    int count) {
  if (count <= 0 || !raw_keys || !rule50_counts || !repetition_distances)
    return kFNVOffset;

  uint64_t key = kFNVOffset;
  MixInto(key, static_cast<uint64_t>(count));
  for (int i = 0; i < count; ++i) {
    MixPositionState(key, i, raw_keys[i], rule50_counts[i],
                     repetition_distances[i]);
  }

  return key;
}

NNCache::NNCache(size_t size) : entries_(size), size_(size) {}

bool NNCache::Lookup(uint64_t key, int expected_moves,
                     EvaluationResult &out) const {
  const Entry &e = entries_[key % size_];

  uint64_t gen1 = e.generation.load(std::memory_order_acquire);
  if (gen1 & 1) return false;

  if (!e.occupied || e.key != key) return false;
  if (expected_moves >= 0 && e.legal_moves != expected_moves) return false;

  out.value = e.value;
  out.has_wdl = e.has_wdl;
  if (e.has_wdl) {
    out.wdl[0] = e.wdl[0];
    out.wdl[1] = e.wdl[1];
    out.wdl[2] = e.wdl[2];
  }
  out.has_moves_left = e.has_moves_left;
  out.moves_left = e.moves_left;

  out.policy_priors.clear();
  out.policy_priors.reserve(e.num_moves);
  for (int i = 0; i < e.num_moves; ++i) {
    Move m = Move(e.moves[i].move_raw);
    out.policy_priors.emplace_back(m, e.moves[i].policy);
  }

  uint64_t gen2 = e.generation.load(std::memory_order_acquire);
  if (gen1 != gen2) return false;

  return true;
}

void NNCache::Insert(uint64_t key, const EvaluationResult &result) {
  Entry &e = entries_[key % size_];
  uint64_t gen = e.generation.load(std::memory_order_relaxed);
  e.generation.store(gen + 1, std::memory_order_release);

  if (result.policy_priors.size() > Entry::MAX_MOVES) {
    e.occupied = false;
    e.key = 0;
    e.num_moves = 0;
    e.legal_moves = 0;
    e.generation.store(gen + 2, std::memory_order_release);
    return;
  }

  e.key = key;
  e.value = result.value;
  e.draw = result.has_wdl ? result.wdl[1] : 0.0f;
  e.moves_left = result.moves_left;
  e.has_wdl = result.has_wdl;
  if (result.has_wdl) {
    e.wdl[0] = result.wdl[0];
    e.wdl[1] = result.wdl[1];
    e.wdl[2] = result.wdl[2];
  }
  e.has_moves_left = result.has_moves_left;

  int n = static_cast<int>(result.policy_priors.size());
  e.num_moves = n;
  e.legal_moves = static_cast<uint16_t>(result.policy_priors.size());
  for (int i = 0; i < n; ++i) {
    e.moves[i].move_raw = result.policy_priors[i].first.raw();
    e.moves[i].policy = result.policy_priors[i].second;
  }
  e.occupied = true;

  e.generation.store(gen + 2, std::memory_order_release);
}

void NNCache::Clear() {
  for (auto &e : entries_) {
    e.generation.store(0, std::memory_order_relaxed);
    e.occupied = false;
    e.key = 0;
    e.num_moves = 0;
    e.legal_moves = 0;
  }
}

BackendComputation::BackendComputation(NNMCTSEvaluator *eval, NNCache *cache)
    : evaluator_(eval), cache_(cache) {}

BackendComputation::AddInputResult
BackendComputation::AddInput(const Position &pos, uint64_t key) {
  const Position *history[1] = {&pos};
  return AddInputWithHistory(history, 1, key);
}

BackendComputation::AddInputResult
BackendComputation::AddInputWithHistory(
    const std::vector<const Position *> &history, uint64_t key) {
  return AddInputWithHistory(history.data(), static_cast<int>(history.size()),
                             key);
}

BackendComputation::AddInputResult
BackendComputation::AddInputWithHistory(const Position *const *history,
                                        int history_depth, uint64_t key,
                                        int expected_moves) {
  return AddInputWithHistory(history, history_depth, key, nullptr,
                             expected_moves);
}

BackendComputation::AddInputResult
BackendComputation::AddInputWithHistory(const Position *const *history,
                                        int history_depth, uint64_t key,
                                        const Move *legal_moves,
                                        int legal_move_count) {
  int idx = total_inputs_++;
  results_.emplace_back();
  from_cache_.push_back(false);

  int expected_moves = legal_move_count;
  EvaluationResult cached;
  if (expected_moves < 0 && history_depth > 0) {
    MoveList<LEGAL> moves(*history[history_depth - 1]);
    expected_moves = static_cast<int>(moves.size());
  }
  if (cache_ && cache_->Lookup(key, expected_moves, cached)) {
    results_[idx] = std::move(cached);
    from_cache_[idx] = true;
    return CACHE_HIT;
  }

  PendingInput pending;
  pending.key = key;
  pending.result_idx = idx;
  pending.history_depth =
      std::min(history_depth, static_cast<int>(pending.history.size()));
  pending.legal_move_count = -1;
  if (legal_moves && legal_move_count >= 0) {
    pending.legal_move_count =
        std::min(legal_move_count, static_cast<int>(pending.legal_moves.size()));
    for (int i = 0; i < pending.legal_move_count; ++i) {
      pending.legal_moves[i] = legal_moves[i];
    }
  }
  const int start = history_depth - pending.history_depth;
  for (int i = 0; i < pending.history_depth; ++i) {
    pending.history[i] = history[start + i];
  }
  pending_.push_back(pending);
  return QUEUED;
}

void BackendComputation::ComputeBlocking() {
  if (pending_.empty()) return;

  std::vector<NNMCTSEvaluator::PositionHistoryView> histories;
  histories.reserve(pending_.size());
  std::vector<NNMCTSEvaluator::LegalMovesView> legal_moves;
  legal_moves.reserve(pending_.size());
  bool all_have_legal_moves = true;
  for (const auto &p : pending_) {
    histories.emplace_back(p.history.data(), p.history_depth);
    if (p.legal_move_count >= 0) {
      legal_moves.emplace_back(p.legal_moves.data(), p.legal_move_count);
    } else {
      all_have_legal_moves = false;
    }
  }

  auto batch_results = all_have_legal_moves
                           ? evaluator_->EvaluateBatchWithHistoryViews(
                                 histories, legal_moves)
                           : evaluator_->EvaluateBatchWithHistoryViews(
                                 histories);

  for (size_t i = 0; i < pending_.size(); ++i) {
    int idx = pending_[i].result_idx;
    results_[idx] = std::move(batch_results[i]);
    if (cache_) {
      cache_->Insert(pending_[i].key, results_[idx]);
    }
  }

  pending_.clear();
}

const EvaluationResult &BackendComputation::GetResult(int idx) const {
  return results_[idx];
}

int BackendComputation::UsedBatchSize() const {
  return static_cast<int>(pending_.size());
}

int BackendComputation::TotalInputs() const {
  return total_inputs_;
}

Backend::Backend(const std::string &weights_path, size_t cache_entries)
    : cache_(cache_entries) {
  try {
    evaluator_ = std::make_unique<NNMCTSEvaluator>(weights_path);
    std::cerr << "info string Backend loaded weights: " << weights_path
              << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "info string Backend failed to load weights: " << e.what()
              << std::endl;
    throw;
  }
}

std::unique_ptr<BackendComputation> Backend::CreateComputation() {
  return std::make_unique<BackendComputation>(evaluator_.get(), &cache_);
}

bool Backend::HasWDL() const { return true; }

bool Backend::HasMovesLeft() const { return true; }

} // namespace MCTS
} // namespace MetalFish

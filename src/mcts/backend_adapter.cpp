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

// ============================================================================
// NNCache
// ============================================================================

NNCache::NNCache(size_t size) : entries_(size), size_(size) {}

bool NNCache::Lookup(uint64_t key, int expected_moves,
                     EvaluationResult &out) const {
  const Entry &e = entries_[key % size_];
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
  return true;
}

void NNCache::Insert(uint64_t key, const EvaluationResult &result) {
  Entry &e = entries_[key % size_];
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

  int n = std::min(static_cast<int>(result.policy_priors.size()),
                   Entry::MAX_MOVES);
  e.num_moves = n;
  e.legal_moves = static_cast<uint16_t>(result.policy_priors.size());
  for (int i = 0; i < n; ++i) {
    e.moves[i].move_raw = result.policy_priors[i].first.raw();
    e.moves[i].policy = result.policy_priors[i].second;
  }
  e.occupied = true;
}

void NNCache::Clear() {
  for (auto &e : entries_) {
    e.occupied = false;
    e.key = 0;
    e.num_moves = 0;
    e.legal_moves = 0;
  }
}

// ============================================================================
// BackendComputation
// ============================================================================

BackendComputation::BackendComputation(NNMCTSEvaluator *eval, NNCache *cache)
    : evaluator_(eval), cache_(cache) {}

BackendComputation::AddInputResult
BackendComputation::AddInput(const Position &pos, uint64_t key) {
  return AddInputWithHistory({&pos}, key);
}

BackendComputation::AddInputResult
BackendComputation::AddInputWithHistory(
    const std::vector<const Position *> &history, uint64_t key) {
  int idx = total_inputs_++;
  results_.emplace_back();
  from_cache_.push_back(false);

  EvaluationResult cached;
  int expected_moves = -1;
  if (!history.empty()) {
    MoveList<LEGAL> moves(*history.back());
    expected_moves = static_cast<int>(moves.size());
  }
  if (cache_ && cache_->Lookup(key, expected_moves, cached)) {
    results_[idx] = std::move(cached);
    from_cache_[idx] = true;
    return CACHE_HIT;
  }

  pending_.push_back({history, key, idx});
  return QUEUED;
}

void BackendComputation::ComputeBlocking() {
  if (pending_.empty()) return;

  std::vector<std::vector<const Position *>> histories;
  histories.reserve(pending_.size());
  for (const auto &p : pending_) {
    histories.push_back(p.history);
  }

  auto batch_results = evaluator_->EvaluateBatchWithHistory(histories);

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

// ============================================================================
// Backend
// ============================================================================

Backend::Backend(const std::string &weights_path) {
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

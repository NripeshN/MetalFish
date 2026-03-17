/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Backend Adapter - Wraps NN evaluator for batch computation
  Licensed under GPL-3.0
*/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "evaluator.h"

namespace MetalFish {
namespace MCTS {

class NNCache {
public:
  explicit NNCache(size_t size = 1 << 21);
  bool Lookup(uint64_t key, int expected_moves, EvaluationResult &out) const;
  void Insert(uint64_t key, const EvaluationResult &result);
  void Clear();

private:
  struct Entry {
    uint64_t key = 0;
    float value = 0, draw = 0, moves_left = 0;
    bool has_wdl = false;
    float wdl[3] = {};
    bool has_moves_left = false;

        static constexpr int MAX_MOVES = 218;
    struct CachedMove {
      uint16_t move_raw = 0;
      float policy = 0;
    };
    CachedMove moves[MAX_MOVES];
    int num_moves = 0;
    uint16_t legal_moves = 0;
    bool occupied = false;
  };

  std::vector<Entry> entries_;
  size_t size_;
};

class BackendComputation {
public:
  enum AddInputResult { QUEUED = 0, CACHE_HIT = 1 };

  BackendComputation(NNMCTSEvaluator *eval, NNCache *cache);

  AddInputResult AddInput(const Position &pos, uint64_t key);
  AddInputResult AddInputWithHistory(
      const std::vector<const Position *> &history, uint64_t key);
  void ComputeBlocking();
  const EvaluationResult &GetResult(int idx) const;
    int UsedBatchSize() const;
    int TotalInputs() const;

private:
  NNMCTSEvaluator *evaluator_;
  NNCache *cache_;

  struct PendingInput {
    std::vector<const Position *> history;
    uint64_t key;
    int result_idx;
  };

  std::vector<PendingInput> pending_;
  std::vector<EvaluationResult> results_;
  int total_inputs_ = 0;
  std::vector<bool> from_cache_;
};

class Backend {
public:
  explicit Backend(const std::string &weights_path);

  std::unique_ptr<BackendComputation> CreateComputation();
  NNCache &Cache() { return cache_; }
  bool HasWDL() const;
  bool HasMovesLeft() const;

private:
  std::unique_ptr<NNMCTSEvaluator> evaluator_;
  NNCache cache_;
};

} // namespace MCTS
} // namespace MetalFish

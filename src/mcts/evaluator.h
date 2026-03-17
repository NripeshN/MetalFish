/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <vector>

#include "../core/position.h"
#include "../nn/encoder.h"
#include "../nn/network.h"

namespace MetalFish {
namespace MCTS {

// MCTS evaluation result from neural network
struct EvaluationResult {
  float value;
  bool has_wdl;
  float wdl[3];
  bool has_moves_left = false;
  float moves_left = 0.0f;
  std::vector<std::pair<Move, float>> policy_priors;

  EvaluationResult() : value(0.0f), has_wdl(false), wdl{0.0f, 0.0f, 0.0f} {}

  float get_policy(Move move) const {
    for (const auto& [m, p] : policy_priors) {
      if (m == move) return p;
    }
    return 0.0f;
  }
};

// Neural network evaluator for MCTS
class NNMCTSEvaluator {
public:
  explicit NNMCTSEvaluator(const std::string &weights_path);
  ~NNMCTSEvaluator();

  // Evaluate single position with game history for accurate NN encoding.
  // The history vector should contain the last 8 board states (or fewer).
  // history.back() is the current position to evaluate.
  EvaluationResult Evaluate(const Position &pos);
  EvaluationResult EvaluateWithHistory(
      const std::vector<const Position *> &history);

  // Batch evaluation with per-position history
  std::vector<EvaluationResult> EvaluateBatch(const Position *const *positions,
                                              size_t count);
  std::vector<EvaluationResult> EvaluateBatchWithHistory(
      const std::vector<std::vector<const Position *>> &histories);

  // Get network information
  std::string GetNetworkInfo() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace MCTS
} // namespace MetalFish

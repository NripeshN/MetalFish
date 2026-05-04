/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <span>
#include <vector>

#include "../core/position.h"
#include "../nn/encoder.h"
#include "../nn/network.h"

namespace MetalFish {
namespace MCTS {

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

class NNMCTSEvaluator {
public:
  using PositionHistoryView = std::span<const Position *const>;
  using LegalMovesView = std::span<const Move>;

  explicit NNMCTSEvaluator(const std::string &weights_path);
  ~NNMCTSEvaluator();

  EvaluationResult Evaluate(const Position &pos);
  EvaluationResult EvaluateWithHistory(PositionHistoryView history);
  EvaluationResult EvaluateWithHistoryAndMoves(PositionHistoryView history,
                                               LegalMovesView legal_moves);
  EvaluationResult EvaluateWithHistory(
      const std::vector<const Position *> &history) {
    return EvaluateWithHistory(PositionHistoryView(history.data(),
                                                  history.size()));
  }

  // Batch evaluation with per-position history
  std::vector<EvaluationResult> EvaluateBatch(const Position *const *positions,
                                              size_t count);
  std::vector<EvaluationResult> EvaluateBatchWithHistory(
      const std::vector<std::vector<const Position *>> &histories);
  std::vector<EvaluationResult> EvaluateBatchWithHistoryViews(
      const std::vector<PositionHistoryView> &histories);
  std::vector<EvaluationResult> EvaluateBatchWithHistoryViews(
      const std::vector<PositionHistoryView> &histories,
      const std::vector<LegalMovesView> &legal_moves);

  // Get network information
  std::string GetNetworkInfo() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace MCTS
} // namespace MetalFish

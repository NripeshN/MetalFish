/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <array>
#include <memory>
#include <vector>

#include "../core/position.h"
#include "../nn/network.h"
#include "../nn/encoder.h"

namespace MetalFish {
namespace MCTS {

// MCTS evaluation result from neural network
struct EvaluationResult {
  float value;  // Q value from side to move perspective
  bool has_wdl;
  float wdl[3];  // win/draw/loss probabilities
  bool has_moves_left = false;
  float moves_left = 0.0f;
  std::vector<std::pair<Move, float>> policy_priors;  // Move → policy probability pairs

  // O(1) policy lookup table indexed by move.raw() (max 4096 encoded moves).
  // Populated during Evaluate() to avoid O(n) linear scans during PUCT.
  static constexpr int kPolicyTableSize = 4096;
  std::array<float, kPolicyTableSize> policy_table{};

  EvaluationResult() : value(0.0f), has_wdl(false), wdl{0.0f, 0.0f, 0.0f} {
    policy_table.fill(0.0f);
  }

  // Build the O(1) lookup table from policy_priors.
  // Must be called after policy_priors is populated.
  void build_policy_table() {
    for (const auto& [m, p] : policy_priors) {
      uint16_t idx = m.raw() & (kPolicyTableSize - 1);
      policy_table[idx] = p;
    }
  }

  // O(1) policy lookup for a move
  float get_policy(Move move) const {
    return policy_table[move.raw() & (kPolicyTableSize - 1)];
  }
};

// Neural network evaluator for MCTS
class NNMCTSEvaluator {
public:
  explicit NNMCTSEvaluator(const std::string& weights_path);
  ~NNMCTSEvaluator();
  
  // Evaluate single position
  EvaluationResult Evaluate(const Position& pos);
  
  // Batch evaluation for multiple positions
  std::vector<EvaluationResult> EvaluateBatch(const std::vector<Position>& positions);
  
  // Get network information
  std::string GetNetworkInfo() const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace MCTS
}  // namespace MetalFish

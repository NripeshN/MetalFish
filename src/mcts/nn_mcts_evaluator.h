/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

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
  std::vector<std::pair<Move, float>> policy_priors;  // Move → policy probability pairs
  
  EvaluationResult() : value(0.0f), has_wdl(false), wdl{0.0f, 0.0f, 0.0f} {}
  
  // Helper to find policy for a move
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

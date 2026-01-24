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
struct NNEvaluation {
  std::vector<float> policy;  // Move probabilities
  float value;                 // Position value from NN
  
  NNEvaluation() : value(0.0f) {}
};

// Neural network evaluator for MCTS
class NNMCTSEvaluator {
public:
  explicit NNMCTSEvaluator(const std::string& weights_path);
  ~NNMCTSEvaluator();
  
  // Evaluate single position
  NNEvaluation Evaluate(const Position& pos);
  
  // Batch evaluation for multiple positions
  std::vector<NNEvaluation> EvaluateBatch(const std::vector<Position>& positions);
  
  // Get network information
  std::string GetNetworkInfo() const;

private:
  std::unique_ptr<NN::Network> network_;
};

}  // namespace MCTS
}  // namespace MetalFish

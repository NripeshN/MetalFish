/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "nn_mcts_evaluator.h"

#include <stdexcept>

namespace MetalFish {
namespace MCTS {

NNMCTSEvaluator::NNMCTSEvaluator(const std::string& weights_path) {
  try {
    network_ = NN::CreateNetwork(weights_path);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to initialize NN evaluator: " + std::string(e.what()));
  }
}

NNMCTSEvaluator::~NNMCTSEvaluator() = default;

NNEvaluation NNMCTSEvaluator::Evaluate(const Position& pos) {
  // Encode position to NN input format
  NN::InputPlanes input = NN::EncodePositionForNN(pos);
  
  // Evaluate through network
  NN::NetworkOutput nn_output = network_->Evaluate(input);
  
  // Convert to MCTS evaluation format
  NNEvaluation result;
  result.policy = std::move(nn_output.policy);
  result.value = nn_output.value;
  
  return result;
}

std::vector<NNEvaluation> NNMCTSEvaluator::EvaluateBatch(
    const std::vector<Position>& positions) {
  
  // Encode all positions
  std::vector<NN::InputPlanes> inputs;
  inputs.reserve(positions.size());
  
  for (const auto& pos : positions) {
    inputs.push_back(NN::EncodePositionForNN(pos));
  }
  
  // Batch evaluate
  std::vector<NN::NetworkOutput> nn_outputs = network_->EvaluateBatch(inputs);
  
  // Convert results
  std::vector<NNEvaluation> results;
  results.reserve(nn_outputs.size());
  
  for (auto& nn_out : nn_outputs) {
    NNEvaluation eval;
    eval.policy = std::move(nn_out.policy);
    eval.value = nn_out.value;
    results.push_back(std::move(eval));
  }
  
  return results;
}

std::string NNMCTSEvaluator::GetNetworkInfo() const {
  return network_->GetNetworkInfo();
}

}  // namespace MCTS
}  // namespace MetalFish

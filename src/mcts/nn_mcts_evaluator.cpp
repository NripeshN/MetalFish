/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "nn_mcts_evaluator.h"
#include "../nn/policy_map.h"
#include "../core/movegen.h"

#include <stdexcept>

namespace MetalFish {
namespace MCTS {

class NNMCTSEvaluator::Impl {
public:
  Impl(const std::string& weights_path) {
    network_ = NN::CreateNetwork(weights_path, "auto");
  }
  
  EvaluationResult Evaluate(const Position& pos) {
    // 1. Encode position (use simple overload that doesn't require copying)
    auto planes = NN::EncodePositionForNN(
        pos, MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE);
    
    // 2. Run neural network
    auto output = network_->Evaluate(planes);
    
    // 3. Convert to MCTS evaluation result
    EvaluationResult result;
    result.value = output.value;
    result.has_wdl = output.has_wdl;
    if (output.has_wdl) {
      result.wdl[0] = output.wdl[0];  // win
      result.wdl[1] = output.wdl[1];  // draw
      result.wdl[2] = output.wdl[2];  // loss
    }
    
    // 4. Map policy outputs to legal moves
    MoveList<LEGAL> moves(pos);
    result.policy_priors.reserve(moves.size());
    for (const auto& move : moves) {
      int policy_idx = NN::MoveToNNIndex(move);
      if (policy_idx >= 0 && policy_idx < static_cast<int>(output.policy.size())) {
        result.policy_priors.emplace_back(move, output.policy[policy_idx]);
      }
    }
    
    return result;
  }
  
  std::vector<EvaluationResult> EvaluateBatch(
      const std::vector<Position>& positions) {
    // Batch encoding
    std::vector<NN::InputPlanes> planes_batch;
    planes_batch.reserve(positions.size());
    
    for (const auto& pos : positions) {
      auto planes = NN::EncodePositionForNN(
          pos, MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE);
      planes_batch.push_back(planes);
    }
    
    // Batch inference
    auto outputs = network_->EvaluateBatch(planes_batch);
    
    // Convert to results
    std::vector<EvaluationResult> results;
    results.reserve(outputs.size());
    
    for (size_t i = 0; i < outputs.size(); ++i) {
      EvaluationResult result;
      result.value = outputs[i].value;
      result.has_wdl = outputs[i].has_wdl;
      if (outputs[i].has_wdl) {
        result.wdl[0] = outputs[i].wdl[0];
        result.wdl[1] = outputs[i].wdl[1];
        result.wdl[2] = outputs[i].wdl[2];
      }
      
      // Map policy
      MoveList<LEGAL> moves(positions[i]);
      result.policy_priors.reserve(moves.size());
      for (const auto& move : moves) {
        int policy_idx = NN::MoveToNNIndex(move);
        if (policy_idx >= 0 && policy_idx < static_cast<int>(outputs[i].policy.size())) {
          result.policy_priors.emplace_back(move, outputs[i].policy[policy_idx]);
        }
      }
      
      results.push_back(result);
    }
    
    return results;
  }
  
  std::string GetNetworkInfo() const {
    return network_->GetNetworkInfo();
  }
  
private:
  std::unique_ptr<NN::Network> network_;
};

NNMCTSEvaluator::NNMCTSEvaluator(const std::string& weights_path)
    : impl_(std::make_unique<Impl>(weights_path)) {}

NNMCTSEvaluator::~NNMCTSEvaluator() = default;

EvaluationResult NNMCTSEvaluator::Evaluate(const Position& pos) {
  return impl_->Evaluate(pos);
}

std::vector<EvaluationResult> NNMCTSEvaluator::EvaluateBatch(
    const std::vector<Position>& positions) {
  return impl_->EvaluateBatch(positions);
}

std::string NNMCTSEvaluator::GetNetworkInfo() const {
  return impl_->GetNetworkInfo();
}

}  // namespace MCTS
}  // namespace MetalFish

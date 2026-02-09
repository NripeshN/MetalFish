/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "evaluator.h"
#include "../nn/policy_map.h"
#include "../nn/loader.h"
#include "../core/movegen.h"
#include <algorithm>

#include <vector>
#include <stdexcept>

namespace MetalFish {
namespace MCTS {

class NNMCTSEvaluator::Impl {
public:
  Impl(const std::string& weights_path) {
    auto weights_opt = NN::LoadWeights(weights_path);
    if (!weights_opt.has_value()) {
      throw std::runtime_error("Could not load network weights");
    }
    weights_ = std::move(weights_opt.value());
    input_format_ = weights_.format().has_network_format()
                        ? weights_.format().network_format().input()
                        : MetalFishNN::NetworkFormat::INPUT_CLASSICAL_112_PLANE;
    network_ = NN::CreateNetwork(weights_, "auto");
  }
  
  EvaluationResult Evaluate(const Position& pos) {
    // 1. Encode position with transform (canonical if requested by network)
    std::vector<const Position*> history = {&pos};
    int transform = 0;
    auto planes = NN::EncodePositionForNN(
        input_format_, history, NN::kMoveHistory,
        NN::FillEmptyHistory::FEN_ONLY, &transform);
    
    // 2. Run neural network
    auto output = network_->Evaluate(planes);
    
    // 3. Convert to MCTS evaluation result
    EvaluationResult result;
    // Use raw network value (already from side-to-move perspective).
    result.value = output.value;
    result.has_wdl = output.has_wdl;
    if (output.has_wdl) {
      result.wdl[0] = output.wdl[0];  // win
      result.wdl[1] = output.wdl[1];  // draw
      result.wdl[2] = output.wdl[2];  // loss
    }
    result.has_moves_left = output.has_moves_left;
    result.moves_left = output.moves_left;
    
    // 4. Map policy outputs to legal moves
    MoveList<LEGAL> moves(pos);
    result.policy_priors.reserve(moves.size());
    for (const auto& move : moves) {
      int policy_idx = NN::MoveToNNIndex(move, transform);
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
    std::vector<int> transforms;
    transforms.reserve(positions.size());
    
    for (const auto& pos : positions) {
      std::vector<const Position*> history = {&pos};
      int transform = 0;
      auto planes = NN::EncodePositionForNN(
          input_format_, history, NN::kMoveHistory,
          NN::FillEmptyHistory::FEN_ONLY, &transform);
      planes_batch.push_back(planes);
      transforms.push_back(transform);
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
      result.has_moves_left = outputs[i].has_moves_left;
      result.moves_left = outputs[i].moves_left;
      
      // Map policy
      MoveList<LEGAL> moves(positions[i]);
      result.policy_priors.reserve(moves.size());
      for (const auto& move : moves) {
        int policy_idx = NN::MoveToNNIndex(move, transforms[i]);
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
  MetalFishNN::NetworkFormat::InputFormat input_format_;
  NN::WeightsFile weights_;
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

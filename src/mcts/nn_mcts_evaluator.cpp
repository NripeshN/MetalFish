/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  STUB IMPLEMENTATION - Neural Network Evaluator for MCTS
*/

#include "nn_mcts_evaluator.h"
#include "../core/position.h"
#include "../core/movegen.h"
#include <cmath>

namespace MetalFish {
namespace NN {

std::unique_ptr<Lc0NNEvaluator> Lc0NNEvaluator::create(const std::string& weights_path) {
    auto evaluator = std::unique_ptr<Lc0NNEvaluator>(new Lc0NNEvaluator());
    
    // TODO: Implement network loading
    // This would involve:
    // 1. Loading .pb file using loader from src/nn/loader.cpp
    // 2. Parsing protobuf weights (proto/net.pb.h)
    // 3. Initializing Metal backend with weights
    // 4. Setting up position encoder
    
    // For now, just mark as not ready
    evaluator->ready_ = false;
    
    return evaluator;
}

NNEvaluation Lc0NNEvaluator::evaluate(const Position& pos) {
    // TODO: Implement actual neural network evaluation
    // Steps would be:
    // 1. Encode position to 112 planes using src/nn/encoder.cpp
    // 2. Run inference through Metal backend
    // 3. Decode policy output (1858 logits) to legal moves
    // 4. Extract value/WDL from network output
    
    // STUB: Return placeholder evaluation
    NNEvaluation result;
    result.value = 0.0f;  // Draw
    result.win_prob = 0.33f;
    result.draw_prob = 0.34f;
    result.loss_prob = 0.33f;
    result.moves_left = 50.0f;
    
    // Generate uniform policy for all legal moves
    StateInfo si;
    MoveList<LEGAL> moves(pos);
    float uniform_prob = 1.0f / static_cast<float>(moves.size());
    
    for (const auto& m : moves) {
        result.policy.emplace_back(static_cast<Move>(m), uniform_prob);
    }
    
    return result;
}

std::vector<NNEvaluation> Lc0NNEvaluator::evaluate_batch(
    const std::vector<Position*>& positions) {
    
    // TODO: Implement batched evaluation
    // This is more efficient as it:
    // 1. Encodes all positions at once
    // 2. Runs inference on full batch through Metal
    // 3. Decodes all outputs
    
    // STUB: Just evaluate one by one
    std::vector<NNEvaluation> results;
    results.reserve(positions.size());
    
    for (const auto* pos : positions) {
        results.push_back(evaluate(*pos));
    }
    
    return results;
}

} // namespace NN
} // namespace MetalFish

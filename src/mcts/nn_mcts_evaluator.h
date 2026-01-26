/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
  
  STUB IMPLEMENTATION - Neural Network Evaluator for MCTS
  
  This file provides the integration point between MetalFish's MCTS
  and neural network evaluation. The actual NN inference implementation
  requires substantial additional work (see src/nn/README.md).
*/

#pragma once

#include "../core/position.h"
#include "../core/movegen.h"
#include <vector>
#include <memory>

namespace MetalFish {
namespace NN {

// Neural network evaluation result
struct NNEvaluation {
    // Policy: probabilities for each legal move
    std::vector<std::pair<Move, float>> policy;
    
    // Value evaluation (win probability from side-to-move perspective)
    float value;
    
    // Win-Draw-Loss probabilities (optional, for WDL heads)
    float win_prob;
    float draw_prob;
    float loss_prob;
    
    // Moves-left prediction (optional)
    float moves_left;
};

// Neural network evaluator interface for MCTS
class NNEvaluator {
public:
    virtual ~NNEvaluator() = default;
    
    // Evaluate a single position
    virtual NNEvaluation evaluate(const Position& pos) = 0;
    
    // Batch evaluation for multiple positions (more efficient)
    virtual std::vector<NNEvaluation> evaluate_batch(
        const std::vector<Position*>& positions) = 0;
    
    // Check if network is loaded and ready
    virtual bool is_ready() const = 0;
};

// STUB: Lc0-compatible NN evaluator using Metal backend
// TODO: Implement full neural network inference
// Current status: Returns placeholder values
class Lc0NNEvaluator : public NNEvaluator {
public:
    // Load network weights from .pb file
    static std::unique_ptr<Lc0NNEvaluator> create(const std::string& weights_path);
    
    ~Lc0NNEvaluator() override = default;
    
    NNEvaluation evaluate(const Position& pos) override;
    std::vector<NNEvaluation> evaluate_batch(
        const std::vector<Position*>& positions) override;
    
    bool is_ready() const override { return ready_; }
    
private:
    Lc0NNEvaluator() = default;
    bool ready_ = false;
    
    // TODO: Add Metal backend, position encoder, policy decoder
    // These would be implemented using code adapted from lc0:
    // - MetalBackend: src/neural/backends/metal/network_metal.h/cc
    // - PositionEncoder: src/neural/encoder.h/cc
    // - PolicyDecoder: src/neural/tables/policy_map.h
};

} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

/**
 * @file nn_mcts_evaluator.h
 * @brief MetalFish source file.
 */

  Neural Network MCTS Evaluator
  
  Bridges neural network evaluation with MCTS search.
  Provides policy and value evaluation for MCTS leaf nodes.
  
  Licensed under GPL-3.0
*/

#pragma once

#include "encoder.h"
#include "loader.h"
#include "../core/position.h"
#include "../core/types.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace MetalFish {
namespace NN {

// Neural network evaluation result
struct NNEvaluation {
  // Policy distribution over legal moves
  std::vector<std::pair<Move, float>> policy;
  
  // Value head outputs (Win-Draw-Loss)
  float win_prob = 0.0f;
  float draw_prob = 0.0f;
  float loss_prob = 0.0f;
  
  // Derived Q value (from side-to-move perspective)
  float q_value = 0.0f;
  
  // Moves left head
  float moves_left = 0.0f;
  
  // Calculate Q from WDL
  void calculate_q(float draw_score = 0.0f) {
    q_value = win_prob - loss_prob + draw_score * draw_prob;
  }
};

// Neural Network Evaluator Interface
class NNEvaluator {
public:
  virtual ~NNEvaluator() = default;
  
  // Evaluate a position
  virtual bool evaluate(const Position& pos, NNEvaluation& result) = 0;
  
  // Batch evaluation (for better GPU utilization)
  virtual bool evaluate_batch(const std::vector<Position>& positions,
                              std::vector<NNEvaluation>& results) = 0;
  
  // Check if evaluator is ready
  virtual bool is_ready() const = 0;
};

// Lc0 Neural Network Evaluator with Metal Backend
class Lc0NNEvaluator : public NNEvaluator {
public:
  Lc0NNEvaluator();
  ~Lc0NNEvaluator() override;
  
  // Load network from file
  bool load_network(const std::string& path);
  
  // NNEvaluator interface
  bool evaluate(const Position& pos, NNEvaluation& result) override;
  bool evaluate_batch(const std::vector<Position>& positions,
                     std::vector<NNEvaluation>& results) override;
  bool is_ready() const override { return network_loaded_; }
  
  // Get network info
  const TransformerConfig& config() const;
  
private:
  bool network_loaded_ = false;
  std::unique_ptr<Lc0NetworkWeights> weights_;
  Lc0PositionEncoder encoder_;
  
  // Inference backend (Metal or CPU fallback)
  struct InferenceBackend;
  std::unique_ptr<InferenceBackend> backend_;
  
  // Run inference on encoded position
  bool run_inference(const EncodedPosition& input, NNEvaluation& output);
};

// MCTS Evaluator - integrates NN with MCTS
class NNMCTSEvaluator {
public:
  NNMCTSEvaluator(std::shared_ptr<NNEvaluator> evaluator);
  ~NNMCTSEvaluator() = default;
  
  // Evaluate position for MCTS
  // Returns Q value and fills policy for legal moves
  float evaluate_for_mcts(const Position& pos, 
                          std::vector<std::pair<Move, float>>& policy,
                          float& draw_prob,
                          float& moves_left);
  
  // Batch evaluation for MCTS
  bool evaluate_batch_for_mcts(const std::vector<Position>& positions,
                               std::vector<float>& q_values,
                               std::vector<std::vector<std::pair<Move, float>>>& policies,
                               std::vector<float>& draw_probs,
                               std::vector<float>& moves_lefts);
  
  // Transposition table for caching evaluations
  struct TTEntry {
    uint64_t key = 0;
    float q_value = 0.0f;
    float draw_prob = 0.0f;
    float moves_left = 0.0f;
    std::vector<std::pair<Move, float>> policy;
    uint32_t age = 0;
  };
  
  // Cache lookup
  bool probe_cache(uint64_t key, TTEntry& entry);
  void store_cache(uint64_t key, const TTEntry& entry);
  
  // Clear cache
  void clear_cache();
  
  // New search (increment age for replacement policy)
  void new_search();
  
private:
  std::shared_ptr<NNEvaluator> evaluator_;
  
  // Evaluation cache
  static constexpr size_t CACHE_SIZE = 1 << 20; // 1M entries
  std::vector<TTEntry> cache_;
  uint32_t current_age_ = 0;
};

} // namespace NN
} // namespace MetalFish
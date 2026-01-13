/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Neural Network Backend for MCTS

  This file provides a neural network backend for MCTS that leverages
  our existing GPU NNUE infrastructure. It converts NNUE scores to
  WDL (Win/Draw/Loss) probabilities and provides policy priors.

  Licensed under GPL-3.0
*/

#pragma once

#include "../mcts/stockfish_adapter.h"
#include "gpu_nnue_integration.h"
#include <memory>
#include <vector>

namespace MetalFish {
namespace GPU {

// GPU-accelerated neural network backend for MCTS
// Uses our existing GPU NNUE infrastructure for evaluation
class GPUMCTSBackend : public MCTS::MCTSNeuralNetwork {
public:
  GPUMCTSBackend();
  ~GPUMCTSBackend() override = default;

  // Initialize with GPU NNUE manager
  bool initialize(GPUNNUEManager *manager);

  // MCTSNeuralNetwork interface
  MCTS::MCTSEvaluation evaluate(const MCTS::MCTSPosition &pos) override;
  std::vector<MCTS::MCTSEvaluation> evaluate_batch(
      const std::vector<const MCTS::MCTSPosition *> &positions) override;
  int optimal_batch_size() const override { return optimal_batch_size_; }

  // Configuration
  void set_optimal_batch_size(int size) { optimal_batch_size_ = size; }
  void set_use_big_network(bool use_big) { use_big_network_ = use_big; }

  // WDL conversion parameters
  void set_wdl_rescale(float win_rate, float draw_rate);

  // Statistics
  size_t total_evaluations() const { return total_evals_; }
  size_t batch_evaluations() const { return batch_evals_; }
  double avg_batch_size() const;

private:
  GPUNNUEManager *gpu_manager_ = nullptr;

  int optimal_batch_size_ = 64;
  bool use_big_network_ = true;

  // WDL conversion parameters (derived from Stockfish's win rate model)
  float wdl_a_ = 0.0f; // Win rate at eval=0
  float wdl_b_ = 1.0f; // Scaling factor

  // Statistics
  size_t total_evals_ = 0;
  size_t batch_evals_ = 0;
  size_t total_positions_ = 0;

  // Convert NNUE centipawn score to WDL probabilities
  void score_to_wdl(int score, float &win, float &draw, float &loss);

  // Generate policy priors from legal moves
  // Since NNUE doesn't provide policy, we use a heuristic based on move
  // ordering
  std::vector<std::pair<MCTS::MCTSMove, float>>
  generate_policy(const MCTS::MCTSPosition &pos);
};

// Factory function
std::unique_ptr<GPUMCTSBackend>
create_gpu_mcts_backend(GPUNNUEManager *manager);

} // namespace GPU
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  GPU Neural Network Backend for MCTS - Implementation

  Licensed under GPL-3.0
*/

#include "gpu_mcts_backend.h"
#include "../core/movegen.h"
#include "../search/movepick.h"
#include "gpu_nnue_integration.h"
#include <algorithm>
#include <cmath>

namespace MetalFish {
namespace GPU {

GPUMCTSBackend::GPUMCTSBackend() {
  // Default WDL parameters based on Stockfish's win rate model
  // These convert centipawn scores to win probabilities
  // win_rate = 1 / (1 + exp(-score / scale))
  wdl_a_ = 0.5f;   // Base win rate at score=0
  wdl_b_ = 200.0f; // Scaling factor (centipawns)
}

bool GPUMCTSBackend::initialize(GPUNNUEManager *manager) {
  if (!manager || !manager->is_ready()) {
    return false;
  }
  gpu_manager_ = manager;
  return true;
}

void GPUMCTSBackend::score_to_wdl(int score, float &win, float &draw,
                                  float &loss) {
  // Convert centipawn score to win probability using logistic function
  // This is based on the win rate model

  // Clamp extreme scores
  score = std::clamp(score, -10000, 10000);

  // Logistic function with bias: P(win) = 1 / (1 + exp(-(score/scale + bias)))
  // The bias term shifts the curve so that P(win|score=0) = wdl_a_
  // bias = log(wdl_a_ / (1 - wdl_a_)) is the inverse logit of wdl_a_
  float x = static_cast<float>(score) / wdl_b_;

  // Compute bias from wdl_a_ (clamp to avoid log(0) or division by zero)
  float clamped_a = std::clamp(wdl_a_, 0.001f, 0.999f);
  float bias = std::log(clamped_a / (1.0f - clamped_a));

  float win_prob = 1.0f / (1.0f + std::exp(-(x + bias)));

  // Estimate draw probability based on score magnitude
  // Higher magnitude = lower draw probability
  float score_mag = std::abs(static_cast<float>(score)) / 100.0f;
  float draw_prob = std::max(0.0f, 0.4f - 0.1f * score_mag);
  draw_prob = std::min(draw_prob, 0.4f);

  // Distribute remaining probability (after draw) between win and loss
  win = win_prob * (1.0f - draw_prob);
  loss = (1.0f - win_prob) * (1.0f - draw_prob);
  draw = draw_prob;

  // Ensure they sum to 1
  float sum = win + draw + loss;
  if (sum > 0.0f) {
    win /= sum;
    draw /= sum;
    loss /= sum;
  } else {
    win = 0.33f;
    draw = 0.34f;
    loss = 0.33f;
  }
}

std::vector<std::pair<MCTS::MCTSMove, float>>
GPUMCTSBackend::generate_policy(const MCTS::MCTSPosition &pos) {

  std::vector<std::pair<MCTS::MCTSMove, float>> policy;
  MCTS::MCTSMoveList moves = pos.generate_legal_moves();

  if (moves.empty()) {
    return policy;
  }

  // Since NNUE doesn't provide policy, we generate priors based on:
  // 1. Captures (higher priority)
  // 2. Checks (higher priority)
  // 3. Center control
  // 4. Development moves

  const Position &stockfish_pos = pos.internal_position();
  std::vector<float> scores(moves.size());
  float total_score = 0.0f;

  for (size_t i = 0; i < moves.size(); ++i) {
    Move m = moves[i].to_stockfish();
    float score = 1.0f; // Base score

    // Captures get bonus based on captured piece value
    if (stockfish_pos.capture(m)) {
      // For en passant, the captured pawn is not on the destination square
      PieceType captured = m.type_of() == EN_PASSANT
                               ? PAWN
                               : type_of(stockfish_pos.piece_on(m.to_sq()));
      static const float piece_values[] = {0, 1, 3, 3, 5, 9, 0}; // PNBRQK
      score += piece_values[captured] * 0.5f;
    }

    // Promotions get high bonus
    if (m.type_of() == PROMOTION) {
      PieceType promo = m.promotion_type();
      if (promo == QUEEN)
        score += 5.0f;
      else
        score += 2.0f;
    }

    // Moves to center get small bonus
    Square to = m.to_sq();
    int file = file_of(to);
    int rank = rank_of(to);
    float center_dist = std::abs(file - 3.5f) + std::abs(rank - 3.5f);
    score += (7.0f - center_dist) * 0.1f;

    // Castling gets bonus
    if (m.type_of() == CASTLING) {
      score += 1.5f;
    }

    scores[i] = score;
    total_score += score;
  }

  // Normalize to probabilities
  policy.reserve(moves.size());
  for (size_t i = 0; i < moves.size(); ++i) {
    float prob =
        (total_score > 0.0f) ? scores[i] / total_score : 1.0f / moves.size();
    policy.emplace_back(moves[i], prob);
  }

  // Sort by probability descending
  std::sort(policy.begin(), policy.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

  return policy;
}

MCTS::MCTSEvaluation GPUMCTSBackend::evaluate(const MCTS::MCTSPosition &pos) {
  MCTS::MCTSEvaluation result;

  if (!gpu_manager_) {
    // Fallback: return neutral evaluation
    result.wdl[0] = 0.33f;
    result.wdl[1] = 0.34f;
    result.wdl[2] = 0.33f;
    result.q = 0.0f;
    result.m = 30.0f;
    result.policy = generate_policy(pos);
    return result;
  }

  // Use GPU NNUE for evaluation
  auto [psqt, positional] =
      gpu_manager_->evaluate_single(pos.internal_position(), use_big_network_);

  int score = positional; // Use positional score

  // Convert to WDL
  float win, draw, loss;
  score_to_wdl(score, win, draw, loss);

  // Flip if black to move (NNUE gives score from white's perspective)
  if (pos.is_black_to_move()) {
    std::swap(win, loss);
    score = -score;
  }

  result.wdl[0] = win;
  result.wdl[1] = draw;
  result.wdl[2] = loss;
  result.q = win - loss; // Q in [-1, 1]
  result.m = 30.0f;      // Estimate moves left (could be improved)
  result.policy = generate_policy(pos);

  ++total_evals_;

  return result;
}

std::vector<MCTS::MCTSEvaluation> GPUMCTSBackend::evaluate_batch(
    const std::vector<const MCTS::MCTSPosition *> &positions) {

  std::vector<MCTS::MCTSEvaluation> results;
  results.reserve(positions.size());

  if (positions.empty()) {
    return results;
  }

  if (!gpu_manager_) {
    // Fallback: evaluate each position individually
    for (const auto *pos : positions) {
      results.push_back(evaluate(*pos));
    }
    return results;
  }

  // Prepare GPU batch
  GPUEvalBatch batch;
  batch.reserve(static_cast<int>(positions.size()));

  for (const auto *pos : positions) {
    batch.add_position(pos->internal_position());
  }

  // Evaluate on GPU
  bool success = gpu_manager_->evaluate_batch(batch, use_big_network_);

  if (!success) {
    // Fallback to individual evaluation
    for (const auto *pos : positions) {
      results.push_back(evaluate(*pos));
    }
    return results;
  }

  // Convert results
  for (size_t i = 0; i < positions.size(); ++i) {
    MCTS::MCTSEvaluation eval;

    int score = batch.positional_scores[i];

    // Convert to WDL
    float win, draw, loss;
    score_to_wdl(score, win, draw, loss);

    // Flip if black to move
    if (positions[i]->is_black_to_move()) {
      std::swap(win, loss);
      score = -score;
    }

    eval.wdl[0] = win;
    eval.wdl[1] = draw;
    eval.wdl[2] = loss;
    eval.q = win - loss;
    eval.m = 30.0f;
    eval.policy = generate_policy(*positions[i]);

    results.push_back(std::move(eval));
  }

  ++batch_evals_;
  total_positions_ += positions.size();
  total_evals_ += positions.size();

  return results;
}

void GPUMCTSBackend::set_wdl_rescale(float win_rate, float draw_rate) {
  wdl_a_ = win_rate;
  // Clamp draw_rate to [0, 0.99] to prevent division by zero or near-zero
  // in score_to_wdl. A draw_rate of 1.0 would make wdl_b_ = 0, and values
  // close to 1.0 would make the logistic function infinitely sensitive.
  draw_rate = std::clamp(draw_rate, 0.0f, 0.99f);
  // Adjust scaling based on draw rate
  wdl_b_ = 200.0f * (1.0f - draw_rate);
}

double GPUMCTSBackend::avg_batch_size() const {
  if (batch_evals_ == 0)
    return 0.0;
  return static_cast<double>(total_positions_) / batch_evals_;
}

std::unique_ptr<GPUMCTSBackend>
create_gpu_mcts_backend(GPUNNUEManager *manager) {
  auto backend = std::make_unique<GPUMCTSBackend>();
  if (manager && backend->initialize(manager)) {
    return backend;
  }
  return nullptr;
}

} // namespace GPU
} // namespace MetalFish

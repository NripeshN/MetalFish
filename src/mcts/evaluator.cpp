/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "evaluator.h"
#include "../core/movegen.h"
#include "../nn/loader.h"
#include "../nn/policy_map.h"
#include <algorithm>

#include <stdexcept>
#include <vector>

namespace MetalFish {
namespace MCTS {

namespace {

inline Square MirrorSquareVertical(Square sq) {
  return make_square(file_of(sq), Rank(7 - rank_of(sq)));
}

inline Move MirrorMoveVertical(Move move) {
  const Square from = MirrorSquareVertical(move.from_sq());
  const Square to = MirrorSquareVertical(move.to_sq());
  if (move.type_of() == PROMOTION) {
    return Move::make<PROMOTION>(from, to, move.promotion_type());
  }
  return Move(from, to);
}

} // namespace

class NNMCTSEvaluator::Impl {
public:
  Impl(const std::string &weights_path) {
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

  EvaluationResult Evaluate(const Position &pos) {
    std::vector<const Position *> history = {&pos};
    return EvaluateWithHistory(history);
  }

  EvaluationResult EvaluateWithHistory(
      const std::vector<const Position *> &history) {
    if (history.empty())
      return EvaluationResult{};
    const Position &pos = *history.back();

    int transform = 0;
    auto planes =
        NN::EncodePositionForNN(input_format_, history, NN::kMoveHistory,
                                NN::FillEmptyHistory::FEN_ONLY, &transform);

    // 2. Run neural network
    auto output = network_->Evaluate(planes);

    // 3. Convert to MCTS evaluation result
    EvaluationResult result;
    // Use raw network value (already from side-to-move perspective).
    result.value = output.value;
    result.has_wdl = output.has_wdl;
    if (output.has_wdl) {
      result.wdl[0] = output.wdl[0]; // win
      result.wdl[1] = output.wdl[1]; // draw
      result.wdl[2] = output.wdl[2]; // loss
    }
    result.has_moves_left = output.has_moves_left;
    result.moves_left = output.moves_left;

    // 4. Map policy outputs to legal moves
    MoveList<LEGAL> moves(pos);
    result.policy_priors.reserve(moves.size());
    for (const auto &move : moves) {
      Move policy_move = move;
      if (pos.side_to_move() == BLACK) {
        policy_move = MirrorMoveVertical(policy_move);
      }
      int policy_idx = NN::MoveToNNIndex(policy_move, transform);
      if (policy_idx >= 0 &&
          policy_idx < static_cast<int>(output.policy.size())) {
        result.policy_priors.emplace_back(move, output.policy[policy_idx]);
      }
    }
    result.build_policy_table();

    return result;
  }

  std::vector<EvaluationResult> EvaluateBatch(const Position *const *positions,
                                              size_t count) {
    // Batch encoding
    std::vector<NN::InputPlanes> planes_batch;
    planes_batch.reserve(count);
    std::vector<int> transforms;
    transforms.reserve(count);

    for (size_t idx = 0; idx < count; ++idx) {
      const Position &pos = *positions[idx];
      std::vector<const Position *> history = {&pos};
      int transform = 0;
      auto planes =
          NN::EncodePositionForNN(input_format_, history, NN::kMoveHistory,
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
      MoveList<LEGAL> moves(*positions[i]);
      result.policy_priors.reserve(moves.size());
      for (const auto &move : moves) {
        Move policy_move = move;
        if (positions[i]->side_to_move() == BLACK) {
          policy_move = MirrorMoveVertical(policy_move);
        }
        int policy_idx = NN::MoveToNNIndex(policy_move, transforms[i]);
        if (policy_idx >= 0 &&
            policy_idx < static_cast<int>(outputs[i].policy.size())) {
          result.policy_priors.emplace_back(move,
                                            outputs[i].policy[policy_idx]);
        }
      }
      result.build_policy_table();

      results.push_back(result);
    }

    return results;
  }

  std::vector<EvaluationResult> EvaluateBatchWithHistory(
      const std::vector<std::vector<const Position *>> &histories) {
    size_t count = histories.size();
    std::vector<NN::InputPlanes> planes_batch;
    planes_batch.reserve(count);
    std::vector<int> transforms;
    transforms.reserve(count);

    for (size_t idx = 0; idx < count; ++idx) {
      int transform = 0;
      auto planes = NN::EncodePositionForNN(
          input_format_, histories[idx], NN::kMoveHistory,
          NN::FillEmptyHistory::FEN_ONLY, &transform);
      planes_batch.push_back(planes);
      transforms.push_back(transform);
    }

    auto outputs = network_->EvaluateBatch(planes_batch);

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

      const Position &pos = *histories[i].back();
      MoveList<LEGAL> moves(pos);
      result.policy_priors.reserve(moves.size());
      for (const auto &move : moves) {
        Move policy_move = move;
        if (pos.side_to_move() == BLACK) {
          policy_move = MirrorMoveVertical(policy_move);
        }
        int policy_idx = NN::MoveToNNIndex(policy_move, transforms[i]);
        if (policy_idx >= 0 &&
            policy_idx < static_cast<int>(outputs[i].policy.size())) {
          result.policy_priors.emplace_back(move, outputs[i].policy[policy_idx]);
        }
      }
      result.build_policy_table();
      results.push_back(result);
    }
    return results;
  }

  std::string GetNetworkInfo() const { return network_->GetNetworkInfo(); }

private:
  MetalFishNN::NetworkFormat::InputFormat input_format_;
  NN::WeightsFile weights_;
  std::unique_ptr<NN::Network> network_;
};

NNMCTSEvaluator::NNMCTSEvaluator(const std::string &weights_path)
    : impl_(std::make_unique<Impl>(weights_path)) {}

NNMCTSEvaluator::~NNMCTSEvaluator() = default;

EvaluationResult NNMCTSEvaluator::Evaluate(const Position &pos) {
  return impl_->Evaluate(pos);
}

EvaluationResult NNMCTSEvaluator::EvaluateWithHistory(
    const std::vector<const Position *> &history) {
  return impl_->EvaluateWithHistory(history);
}

std::vector<EvaluationResult>
NNMCTSEvaluator::EvaluateBatch(const Position *const *positions, size_t count) {
  return impl_->EvaluateBatch(positions, count);
}

std::vector<EvaluationResult>
NNMCTSEvaluator::EvaluateBatchWithHistory(
    const std::vector<std::vector<const Position *>> &histories) {
  return impl_->EvaluateBatchWithHistory(histories);
}

std::string NNMCTSEvaluator::GetNetworkInfo() const {
  return impl_->GetNetworkInfo();
}

} // namespace MCTS
} // namespace MetalFish

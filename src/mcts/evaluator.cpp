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

#include <array>
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
    std::array<const Position *, 1> history = {&pos};
    return EvaluateWithHistory(history);
  }

  EvaluationResult
  EvaluateWithHistory(NNMCTSEvaluator::PositionHistoryView history) {
    if (history.empty())
      return EvaluationResult{};
    const Position &pos = *history.back();

    int transform = 0;
    NN::InputPlanes planes;
    NN::EncodePositionForNN(input_format_, history, NN::kMoveHistory,
                            NN::FillEmptyHistory::FEN_ONLY, planes,
                            &transform);

    auto output = network_->Evaluate(planes);

    return BuildResult(output, pos, transform, nullptr);
  }

  EvaluationResult EvaluateWithHistoryAndMoves(
      NNMCTSEvaluator::PositionHistoryView history,
      NNMCTSEvaluator::LegalMovesView legal_moves) {
    if (history.empty())
      return EvaluationResult{};
    const Position &pos = *history.back();

    int transform = 0;
    NN::InputPlanes planes;
    NN::EncodePositionForNN(input_format_, history, NN::kMoveHistory,
                            NN::FillEmptyHistory::FEN_ONLY, planes,
                            &transform);

    auto output = network_->Evaluate(planes);
    return BuildResult(output, pos, transform, &legal_moves);
  }

  std::vector<EvaluationResult> EvaluateBatch(const Position *const *positions,
                                              size_t count) {
    std::vector<NN::InputPlanes> planes_batch(count);
    std::vector<int> transforms(count);

    for (size_t idx = 0; idx < count; ++idx) {
      const Position &pos = *positions[idx];
      std::array<const Position *, 1> history = {&pos};
      NN::EncodePositionForNN(input_format_, history, NN::kMoveHistory,
                              NN::FillEmptyHistory::FEN_ONLY,
                              planes_batch[idx], &transforms[idx]);
    }

    auto outputs = network_->EvaluateBatch(planes_batch);

    std::vector<EvaluationResult> results;
    results.reserve(outputs.size());

    for (size_t i = 0; i < outputs.size(); ++i) {
      results.push_back(BuildResult(outputs[i], *positions[i], transforms[i],
                                    nullptr));
    }

    return results;
  }

  std::vector<EvaluationResult> EvaluateBatchWithHistoryViews(
      const std::vector<NNMCTSEvaluator::PositionHistoryView> &histories) {
    return EvaluateBatchWithHistoryViews(histories, {});
  }

  std::vector<EvaluationResult> EvaluateBatchWithHistoryViews(
      const std::vector<NNMCTSEvaluator::PositionHistoryView> &histories,
      const std::vector<NNMCTSEvaluator::LegalMovesView> &legal_moves) {
    if (!legal_moves.empty() && legal_moves.size() != histories.size()) {
      throw std::invalid_argument(
          "legal move view count must match history count");
    }

    size_t count = histories.size();
    std::vector<NN::InputPlanes> planes_batch(count);
    std::vector<int> transforms(count);

    for (size_t idx = 0; idx < count; ++idx) {
      NN::EncodePositionForNN(
          input_format_, histories[idx], NN::kMoveHistory,
          NN::FillEmptyHistory::FEN_ONLY, planes_batch[idx],
          &transforms[idx]);
    }

    auto outputs = network_->EvaluateBatch(planes_batch);

    std::vector<EvaluationResult> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      const Position &pos = *histories[i].back();
      const auto *moves = legal_moves.empty() ? nullptr : &legal_moves[i];
      results.push_back(BuildResult(outputs[i], pos, transforms[i], moves));
    }
    return results;
  }

  std::string GetNetworkInfo() const { return network_->GetNetworkInfo(); }

private:
  EvaluationResult BuildResult(const NN::NetworkOutput &output,
                               const Position &pos, int transform,
                               const NNMCTSEvaluator::LegalMovesView
                                   *legal_moves) const {
    EvaluationResult result;
    result.value = output.value;
    result.has_wdl = output.has_wdl;
    if (output.has_wdl) {
      result.wdl[0] = output.wdl[0];
      result.wdl[1] = output.wdl[1];
      result.wdl[2] = output.wdl[2];
    }
    result.has_moves_left = output.has_moves_left;
    result.moves_left = output.moves_left;

    auto append_move = [&](Move move) {
      Move policy_move = move;
      if (pos.side_to_move() == BLACK) {
        policy_move = MirrorMoveVertical(policy_move);
      }
      int policy_idx = NN::MoveToNNIndex(policy_move, transform);
      float prior = 0.0f;
      if (policy_idx >= 0 &&
          policy_idx < static_cast<int>(output.policy.size())) {
        prior = output.policy[policy_idx];
      }
      result.policy_priors.emplace_back(move, prior);
    };

    if (legal_moves) {
      result.policy_priors.reserve(legal_moves->size());
      for (Move move : *legal_moves) {
        append_move(move);
      }
    } else {
      MoveList<LEGAL> moves(pos);
      result.policy_priors.reserve(moves.size());
      for (Move move : moves) {
        append_move(move);
      }
    }

    return result;
  }

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
    PositionHistoryView history) {
  return impl_->EvaluateWithHistory(history);
}

EvaluationResult NNMCTSEvaluator::EvaluateWithHistoryAndMoves(
    PositionHistoryView history, LegalMovesView legal_moves) {
  return impl_->EvaluateWithHistoryAndMoves(history, legal_moves);
}

std::vector<EvaluationResult>
NNMCTSEvaluator::EvaluateBatch(const Position *const *positions, size_t count) {
  return impl_->EvaluateBatch(positions, count);
}

std::vector<EvaluationResult>
NNMCTSEvaluator::EvaluateBatchWithHistory(
    const std::vector<std::vector<const Position *>> &histories) {
  std::vector<PositionHistoryView> views;
  views.reserve(histories.size());
  for (const auto &history : histories) {
    views.emplace_back(history.data(), history.size());
  }
  return impl_->EvaluateBatchWithHistoryViews(views);
}

std::vector<EvaluationResult>
NNMCTSEvaluator::EvaluateBatchWithHistoryViews(
    const std::vector<PositionHistoryView> &histories) {
  return impl_->EvaluateBatchWithHistoryViews(histories);
}

std::vector<EvaluationResult>
NNMCTSEvaluator::EvaluateBatchWithHistoryViews(
    const std::vector<PositionHistoryView> &histories,
    const std::vector<LegalMovesView> &legal_moves) {
  return impl_->EvaluateBatchWithHistoryViews(histories, legal_moves);
}

std::string NNMCTSEvaluator::GetNetworkInfo() const {
  return impl_->GetNetworkInfo();
}

} // namespace MCTS
} // namespace MetalFish

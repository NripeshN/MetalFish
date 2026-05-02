/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Neural-network and encoder smoke test for parity workflows.
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "mcts/evaluator.h"
#include "nn/encoder.h"
#include "nn/policy_map.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <vector>

using namespace MetalFish;

namespace {

bool test_encoder_policy_roundtrip() {
  std::cout << "  Encoder/policy roundtrip..." << std::endl;
  StateInfo st;
  Position pos;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);

  const Move m = Move(SQ_G1, SQ_F3);
  const int idx = NN::MoveToNNIndex(m, 0);
  if (idx < 0) {
    std::cout << "    FAIL: invalid policy index for g1f3" << std::endl;
    return false;
  }
  const Move back = NN::IndexToNNMove(idx, 0);
  if (back != m) {
    std::cout << "    FAIL: policy roundtrip mismatch" << std::endl;
    return false;
  }
  std::cout << "    PASS" << std::endl;
  return true;
}

struct HistoryFixture {
  std::vector<std::unique_ptr<Position>> positions;
  std::vector<std::unique_ptr<StateInfo>> root_states;
  std::vector<std::vector<StateInfo>> stacks;
  std::vector<const Position *> ptrs;
};

HistoryFixture build_history(const std::vector<Move> &moves) {
  HistoryFixture h;
  const int total_plies = static_cast<int>(moves.size());
  h.positions.reserve(total_plies + 1);
  h.root_states.reserve(total_plies + 1);
  h.stacks.resize(total_plies + 1);
  h.ptrs.reserve(total_plies + 1);

  for (int ply = 0; ply <= total_plies; ++ply) {
    h.positions.push_back(std::make_unique<Position>());
    h.root_states.push_back(std::make_unique<StateInfo>());
    h.positions.back()->set(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
        h.root_states.back().get());
    h.stacks[ply].reserve(ply);
    for (int i = 0; i < ply; ++i) {
      h.stacks[ply].emplace_back();
      h.positions.back()->do_move(moves[i], h.stacks[ply].back());
    }
    h.ptrs.push_back(h.positions.back().get());
  }
  return h;
}

bool test_encoder_repetition_plane() {
  std::cout << "  Encoder repetition plane..." << std::endl;
  // Nf3 Nf6 Ng1 Ng8 returns to start position with one repetition.
  std::vector<Move> moves = {
      Move(SQ_G1, SQ_F3),
      Move(SQ_G8, SQ_F6),
      Move(SQ_F3, SQ_G1),
      Move(SQ_F6, SQ_G8),
  };
  auto history = build_history(moves);
  const Position &end = *history.ptrs.back();
  if (!end.is_repetition(std::numeric_limits<int>::max())) {
    std::cout << "    FAIL: expected repeated end position" << std::endl;
    return false;
  }

  int transform = 0;
  auto planes = NN::EncodePositionForNN(
      MetalFishNN::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2,
      history.ptrs, NN::kMoveHistory, NN::FillEmptyHistory::FEN_ONLY,
      &transform);

  bool any_zero = false;
  for (int sq = 0; sq < 64; ++sq) {
    if (planes[12][sq] < 0.5f) {
      any_zero = true;
      break;
    }
  }
  if (any_zero) {
    std::cout << "    FAIL: repetition plane should be set for repeated node"
              << std::endl;
    return false;
  }

  std::cout << "    PASS" << std::endl;
  return true;
}

bool test_mcts_evaluator_optional() {
  std::cout << "  MCTS evaluator smoke..." << std::endl;
  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return true;
  }

  try {
    MCTS::NNMCTSEvaluator eval(weights_path);
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
            &st);
    auto result = eval.Evaluate(pos);
    if (result.policy_priors.empty()) {
      std::cout << "    FAIL: empty policy output" << std::endl;
      return false;
    }
    std::cout << "    PASS: policy_priors=" << result.policy_priors.size()
              << " value=" << result.value << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cout << "    FAIL: exception: " << e.what() << std::endl;
    return false;
  }
}

bool close_enough(float a, float b, float tolerance) {
  return std::fabs(a - b) <= tolerance;
}

bool compare_eval_result(const MCTS::EvaluationResult &single,
                         const MCTS::EvaluationResult &batched,
                         const std::string &label) {
  constexpr float kValueTolerance = 1e-3f;
  constexpr float kPolicyTolerance = 2e-3f;

  if (!close_enough(single.value, batched.value, kValueTolerance)) {
    std::cout << "    FAIL: " << label << " value mismatch single="
              << single.value << " batch=" << batched.value << std::endl;
    return false;
  }

  if (single.has_wdl != batched.has_wdl ||
      single.has_moves_left != batched.has_moves_left) {
    std::cout << "    FAIL: " << label << " head presence mismatch"
              << std::endl;
    return false;
  }

  if (single.has_wdl) {
    for (int i = 0; i < 3; ++i) {
      if (!close_enough(single.wdl[i], batched.wdl[i], kValueTolerance)) {
        std::cout << "    FAIL: " << label << " WDL[" << i
                  << "] mismatch" << std::endl;
        return false;
      }
    }
  }

  if (single.has_moves_left &&
      !close_enough(single.moves_left, batched.moves_left, kValueTolerance)) {
    std::cout << "    FAIL: " << label << " moves-left mismatch"
              << std::endl;
    return false;
  }

  if (single.policy_priors.size() != batched.policy_priors.size()) {
    std::cout << "    FAIL: " << label << " policy size mismatch single="
              << single.policy_priors.size()
              << " batch=" << batched.policy_priors.size() << std::endl;
    return false;
  }

  for (size_t i = 0; i < single.policy_priors.size(); ++i) {
    if (single.policy_priors[i].first != batched.policy_priors[i].first) {
      std::cout << "    FAIL: " << label << " policy move mismatch at " << i
                << std::endl;
      return false;
    }
    if (!close_enough(single.policy_priors[i].second,
                      batched.policy_priors[i].second, kPolicyTolerance)) {
      std::cout << "    FAIL: " << label << " policy logit mismatch at " << i
                << " single=" << single.policy_priors[i].second
                << " batch=" << batched.policy_priors[i].second << std::endl;
      return false;
    }
  }

  return true;
}

bool test_mcts_evaluator_batch_parity_optional() {
  std::cout << "  MCTS evaluator batch parity..." << std::endl;
  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return true;
  }

  try {
    MCTS::NNMCTSEvaluator eval(weights_path);
    const std::vector<std::vector<Move>> lines = {
        {},
        {Move(SQ_E2, SQ_E4)},
        {Move(SQ_D2, SQ_D4)},
        {Move(SQ_G1, SQ_F3)},
        {Move(SQ_C2, SQ_C4)},
        {Move(SQ_E2, SQ_E4), Move(SQ_E7, SQ_E5)},
        {Move(SQ_D2, SQ_D4), Move(SQ_D7, SQ_D5)},
        {Move(SQ_G1, SQ_F3), Move(SQ_G8, SQ_F6)},
        {Move(SQ_C2, SQ_C4), Move(SQ_E7, SQ_E5)},
        {Move(SQ_E2, SQ_E4), Move(SQ_C7, SQ_C5)},
        {Move(SQ_D2, SQ_D4), Move(SQ_G8, SQ_F6), Move(SQ_C2, SQ_C4)},
        {Move(SQ_E2, SQ_E4), Move(SQ_E7, SQ_E5), Move(SQ_G1, SQ_F3),
         Move(SQ_B8, SQ_C6)},
        {Move(SQ_C2, SQ_C4), Move(SQ_E7, SQ_E5), Move(SQ_B1, SQ_C3),
         Move(SQ_G8, SQ_F6)},
        {Move(SQ_G1, SQ_F3), Move(SQ_D7, SQ_D5), Move(SQ_D2, SQ_D4),
         Move(SQ_G8, SQ_F6)},
        {Move(SQ_E2, SQ_E4), Move(SQ_E7, SQ_E6), Move(SQ_D2, SQ_D4),
         Move(SQ_D7, SQ_D5)},
        {Move(SQ_D2, SQ_D4), Move(SQ_D7, SQ_D5), Move(SQ_C2, SQ_C4),
         Move(SQ_E7, SQ_E6), Move(SQ_B1, SQ_C3), Move(SQ_G8, SQ_F6)},
    };

    std::vector<HistoryFixture> fixtures;
    fixtures.reserve(33);
    std::vector<std::vector<const Position *>> histories;
    histories.reserve(33);
    for (int i = 0; i < 33; ++i) {
      fixtures.push_back(build_history(lines[i % lines.size()]));
      histories.push_back(fixtures.back().ptrs);
    }

    std::vector<MCTS::EvaluationResult> singles;
    singles.reserve(histories.size());
    for (const auto &history : histories) {
      singles.push_back(eval.EvaluateWithHistory(history));
    }

    auto batched = eval.EvaluateBatchWithHistory(histories);
    if (batched.size() != singles.size()) {
      std::cout << "    FAIL: batch result count mismatch" << std::endl;
      return false;
    }

    for (size_t i = 0; i < singles.size(); ++i) {
      std::ostringstream label;
      label << "entry " << i;
      if (!compare_eval_result(singles[i], batched[i], label.str()))
        return false;
    }

    std::cout << "    PASS: checked " << histories.size()
              << " varied positions" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cout << "    FAIL: exception: " << e.what() << std::endl;
    return false;
  }
}

} // namespace

int main() {
  Bitboards::init();
  Position::init();
  NN::InitPolicyTables();

  std::cout << "=== NN Comparison Smoke ===" << std::endl;
  const bool ok1 = test_encoder_policy_roundtrip();
  const bool ok2 = test_encoder_repetition_plane();
  const bool ok3 = test_mcts_evaluator_optional();
  const bool ok4 = test_mcts_evaluator_batch_parity_optional();
  return (ok1 && ok2 && ok3 && ok4) ? 0 : 1;
}

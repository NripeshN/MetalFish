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
#include <iostream>
#include <limits>
#include <memory>
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

} // namespace

int main() {
  Bitboards::init();
  Position::init();
  NN::InitPolicyTables();

  std::cout << "=== NN Comparison Smoke ===" << std::endl;
  const bool ok1 = test_encoder_policy_roundtrip();
  const bool ok2 = test_encoder_repetition_plane();
  const bool ok3 = test_mcts_evaluator_optional();
  return (ok1 && ok2 && ok3) ? 0 : 1;
}

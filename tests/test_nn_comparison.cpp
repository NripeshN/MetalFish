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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace MetalFish;

namespace {

std::string move_to_string(Move move) {
  std::string text;
  text.push_back(static_cast<char>('a' + file_of(move.from_sq())));
  text.push_back(static_cast<char>('1' + rank_of(move.from_sq())));
  text.push_back(static_cast<char>('a' + file_of(move.to_sq())));
  text.push_back(static_cast<char>('1' + rank_of(move.to_sq())));
  if (move.type_of() == PROMOTION) {
    char promotion = 'q';
    if (move.promotion_type() == ROOK)
      promotion = 'r';
    else if (move.promotion_type() == BISHOP)
      promotion = 'b';
    else if (move.promotion_type() == KNIGHT)
      promotion = 'n';
    text.push_back(promotion);
  }
  return text;
}

std::vector<std::pair<Move, float>>
sorted_priors(const MCTS::EvaluationResult &result) {
  std::vector<std::pair<Move, float>> priors = result.policy_priors;
  std::sort(priors.begin(), priors.end(), [](const auto &lhs,
                                             const auto &rhs) {
    return lhs.second > rhs.second;
  });
  return priors;
}

bool should_dump_nn_debug() {
  const char *dump = std::getenv("METALFISH_NN_DEBUG_DUMP");
  return dump && dump[0] != '\0' && std::string(dump) != "0";
}

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
    if (should_dump_nn_debug()) {
      const auto priors = sorted_priors(result);
      std::cout << std::fixed << std::setprecision(6);
      std::cout << "    DEBUG backend: " << eval.GetNetworkInfo() << std::endl;
      std::cout << "    DEBUG value=" << result.value;
      if (result.has_wdl) {
        std::cout << " wdl=[" << result.wdl[0] << "," << result.wdl[1]
                  << "," << result.wdl[2] << "]";
      }
      if (result.has_moves_left)
        std::cout << " moves_left=" << result.moves_left;
      std::cout << std::endl;

      std::cout << "    DEBUG top_policy:";
      const size_t limit = std::min<size_t>(priors.size(), 8);
      for (size_t i = 0; i < limit; ++i) {
        std::cout << " " << move_to_string(priors[i].first) << "="
                  << priors[i].second;
      }
      std::cout << std::endl;
    }
    std::cout << "    PASS: policy_priors=" << result.policy_priors.size()
              << " value=" << result.value << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cout << "    FAIL: exception: " << e.what() << std::endl;
    return false;
  }
}

struct ReferenceFenCase {
  const char *name;
  const char *fen;
  float value;
  std::array<float, 3> wdl;
  float moves_left;
  size_t legal_moves;
  std::array<std::pair<const char *, float>, 5> top_policy;
};

bool close_enough(float a, float b, float tolerance) {
  return std::fabs(a - b) <= tolerance;
}

struct ReferenceTolerances {
  float value = 3e-2f;
  float wdl = 3e-2f;
  float moves_left = 2.5e-1f;
  float policy = 6e-2f;
};

bool test_bt4_reference_outputs_optional() {
  std::cout << "  BT4 fixed-position reference outputs..." << std::endl;
  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return true;
  }

  const std::array<ReferenceFenCase, 5> cases = {{
      {"start",
       "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
       0.024781f,
       {0.190430f, 0.643920f, 0.165650f},
       198.672256f,
       20,
       {{{"d2d4", 1.631536f},
         {"g1f3", 1.593553f},
         {"c2c4", 1.429961f},
         {"g2g3", 1.142768f},
         {"e2e3", 1.060148f}}}},
      {"after-e4-black",
       "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
       -0.017762f,
       {0.167213f, 0.647812f, 0.184975f},
       196.408340f,
       20,
       {{{"e7e5", 3.334562f},
         {"c7c6", 0.555071f},
         {"c7c5", 0.270597f},
         {"b8c6", -0.813998f},
         {"e7e6", -0.823254f}}}},
      {"bk07",
       "1nk1r1r1/pp2n1pp/4p3/q2pPp1N/b1pP1P2/B1P2R2/2P1B1PP/"
       "R2Q2K1 w - - 0 1",
       0.514738f,
       {0.609163f, 0.296411f, 0.094425f},
       118.599846f,
       34,
       {{{"a3d6", 2.325019f},
         {"f3g3", 2.061997f},
         {"a3b4", 1.848635f},
         {"d1c1", 1.614317f},
         {"h5f6", 0.444747f}}}},
      {"castling-rights",
       "r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/"
       "R1BQ1RK1 w kq - 0 8",
       0.985659f,
       {0.989455f, 0.006749f, 0.003796f},
       79.126686f,
       36,
       {{{"c1g5", 1.983281f},
         {"c3d5", 1.477554f},
         {"a2a4", 1.432842f},
         {"h2h3", 1.250058f},
         {"c4b5", 1.065558f}}}},
      {"rook-endgame",
       "8/2k5/8/8/8/8/4K3/6R1 w - - 0 1",
       0.999972f,
       {0.999975f, 0.000023f, 0.000002f},
       22.327986f,
       22,
       {{{"g1d1", 1.443181f},
         {"e2d3", 1.430409f},
         {"e2e3", 1.318428f},
         {"g1g6", 1.297542f},
         {"e2f3", 1.032590f}}}},
  }};

  try {
    MCTS::NNMCTSEvaluator eval(weights_path);
    ReferenceTolerances tolerances;
    const std::string network_info = eval.GetNetworkInfo();
    if (network_info.find("CUDA transformer backend") != std::string::npos) {
      tolerances.value = 5e-3f;
      tolerances.wdl = 5e-3f;
      tolerances.moves_left = 1e-1f;
      tolerances.policy = 2e-2f;
    }
    for (const auto &test_case : cases) {
      StateInfo st;
      Position pos;
      pos.set(test_case.fen, false, &st);
      const auto result = eval.Evaluate(pos);
      const auto priors = sorted_priors(result);
      if (result.policy_priors.empty()) {
        std::cout << "    FAIL: " << test_case.name
                  << " empty policy output" << std::endl;
        return false;
      }
      if (!result.has_wdl || !result.has_moves_left) {
        std::cout << "    FAIL: " << test_case.name
                  << " missing WDL or moves-left head" << std::endl;
        return false;
      }
      if (result.policy_priors.size() != test_case.legal_moves) {
        std::cout << "    FAIL: " << test_case.name
                  << " legal count mismatch actual="
                  << result.policy_priors.size()
                  << " expected=" << test_case.legal_moves << std::endl;
        return false;
      }
      if (!close_enough(result.value, test_case.value, tolerances.value)) {
        std::cout << "    FAIL: " << test_case.name
                  << " value drift actual=" << result.value
                  << " expected=" << test_case.value << std::endl;
        return false;
      }
      for (size_t i = 0; i < test_case.wdl.size(); ++i) {
        if (!close_enough(result.wdl[i], test_case.wdl[i], tolerances.wdl)) {
          std::cout << "    FAIL: " << test_case.name << " WDL[" << i
                    << "] drift actual=" << result.wdl[i]
                    << " expected=" << test_case.wdl[i] << std::endl;
          return false;
        }
      }
      if (!close_enough(result.moves_left, test_case.moves_left,
                        tolerances.moves_left)) {
        std::cout << "    FAIL: " << test_case.name
                  << " moves-left drift actual=" << result.moves_left
                  << " expected=" << test_case.moves_left << std::endl;
        return false;
      }
      for (size_t i = 0; i < test_case.top_policy.size(); ++i) {
        const auto actual_move = move_to_string(priors[i].first);
        const auto &[expected_move, expected_logit] = test_case.top_policy[i];
        if (actual_move != expected_move) {
          std::cout << "    FAIL: " << test_case.name << " top policy " << i
                    << " move drift actual=" << actual_move
                    << " expected=" << expected_move << std::endl;
          return false;
        }
        if (!close_enough(priors[i].second, expected_logit,
                          tolerances.policy)) {
          std::cout << "    FAIL: " << test_case.name << " top policy " << i
                    << " logit drift actual=" << priors[i].second
                    << " expected=" << expected_logit << std::endl;
          return false;
        }
      }
      if (should_dump_nn_debug()) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "    DEBUG reference " << test_case.name
                  << " value=" << result.value << " wdl=[" << result.wdl[0]
                  << "," << result.wdl[1] << "," << result.wdl[2]
                  << "] moves_left=" << result.moves_left
                  << " legal=" << result.policy_priors.size() << " top:";
        const size_t limit = std::min<size_t>(priors.size(), 5);
        for (size_t i = 0; i < limit; ++i) {
          std::cout << " " << move_to_string(priors[i].first) << "="
                    << priors[i].second;
        }
        std::cout << std::endl;
      }
    }

    std::cout << "    PASS: checked " << cases.size()
              << " fixed positions" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cout << "    FAIL: exception: " << e.what() << std::endl;
    return false;
  }
}

struct EvalTolerances {
  float value = 1e-2f;
  float moves_left = 5e-2f;
  float policy = 5e-3f;
};

bool compare_eval_result(const MCTS::EvaluationResult &single,
                         const MCTS::EvaluationResult &batched,
                         const std::string &label,
                         const EvalTolerances &tolerances) {
  if (!close_enough(single.value, batched.value, tolerances.value)) {
    std::cout << "    FAIL: " << label
              << " value mismatch single=" << single.value
              << " batch=" << batched.value;
    if (single.has_wdl && batched.has_wdl) {
      std::cout << " single_wdl=[" << single.wdl[0] << ","
                << single.wdl[1] << "," << single.wdl[2]
                << "] batch_wdl=[" << batched.wdl[0] << ","
                << batched.wdl[1] << "," << batched.wdl[2] << "]";
    }
    std::cout << std::endl;
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
      if (!close_enough(single.wdl[i], batched.wdl[i], tolerances.value)) {
        std::cout << "    FAIL: " << label << " WDL[" << i << "] mismatch"
                  << std::endl;
        return false;
      }
    }
  }

  if (single.has_moves_left &&
      !close_enough(single.moves_left, batched.moves_left,
                    tolerances.moves_left)) {
    std::cout << "    FAIL: " << label
              << " moves-left mismatch single=" << single.moves_left
              << " batch=" << batched.moves_left
              << " delta=" << std::fabs(single.moves_left - batched.moves_left)
              << std::endl;
    return false;
  }

  if (single.policy_priors.size() != batched.policy_priors.size()) {
    std::cout << "    FAIL: " << label
              << " policy size mismatch single=" << single.policy_priors.size()
              << " batch=" << batched.policy_priors.size() << std::endl;
    return false;
  }

  for (size_t i = 0; i < single.policy_priors.size(); ++i) {
    if (single.policy_priors[i].first != batched.policy_priors[i].first) {
      std::cout << "    FAIL: " << label << " policy move mismatch at " << i
                << " single=" << move_to_string(single.policy_priors[i].first)
                << " batch=" << move_to_string(batched.policy_priors[i].first)
                << std::endl;
      return false;
    }
    if (!close_enough(single.policy_priors[i].second,
                      batched.policy_priors[i].second, tolerances.policy)) {
      std::cout << "    FAIL: " << label << " policy logit mismatch at " << i
                << " move=" << move_to_string(single.policy_priors[i].first)
                << " single=" << single.policy_priors[i].second
                << " batch=" << batched.policy_priors[i].second
                << " delta="
                << std::fabs(single.policy_priors[i].second -
                             batched.policy_priors[i].second)
                << std::endl;
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
    MCTS::NNMCTSEvaluator single_eval(weights_path);
    MCTS::NNMCTSEvaluator batch_eval(weights_path);
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
      singles.push_back(single_eval.EvaluateWithHistory(history));
    }

    EvalTolerances tolerances;
    const std::string network_info = single_eval.GetNetworkInfo();
    if (network_info.find("CUDA transformer backend") != std::string::npos) {
      tolerances.value = 2.5e-1f;
      tolerances.moves_left = 5.0f;
      tolerances.policy = 2.5e-1f;
    }

    const std::array<size_t, 6> batch_sizes = {1, 2, 4, 8, 16,
                                               histories.size()};
    for (size_t batch_size : batch_sizes) {
      std::vector<std::vector<const Position *>> prefix(
          histories.begin(), histories.begin() + batch_size);
      auto batched = batch_eval.EvaluateBatchWithHistory(prefix);
      if (batched.size() != batch_size) {
        std::cout << "    FAIL: batch result count mismatch for size "
                  << batch_size << std::endl;
        return false;
      }

      for (size_t i = 0; i < batch_size; ++i) {
        std::ostringstream label;
        label << "batch " << batch_size << " entry " << i;
        if (batch_size == histories.size())
          label << " line " << (i % lines.size());
        if (!compare_eval_result(singles[i], batched[i], label.str(),
                                 tolerances))
          return false;
      }
    }

    std::cout << "    PASS: checked " << histories.size()
              << " varied positions across batch sizes" << std::endl;
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
  const bool ok4 = test_bt4_reference_outputs_optional();
  const bool ok5 = test_mcts_evaluator_batch_parity_optional();
  return (ok1 && ok2 && ok3 && ok4 && ok5) ? 0 : 1;
}

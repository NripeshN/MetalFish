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
#include "nn/network.h"
#include "nn/policy_map.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
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

std::string format_float(double value, int precision = 6) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(precision) << value;
  return out.str();
}

std::string single_line(std::string text) {
  std::string result;
  result.reserve(text.size());
  for (char ch : text) {
    if (ch == '\n' || ch == '\r') {
      if (!result.empty() && result.back() != ' ')
        result += "; ";
    } else if (ch == '|') {
      result += ';';
    } else {
      result.push_back(ch);
    }
  }
  return result;
}

std::string result_top_policy_string(const MCTS::EvaluationResult &result,
                                     size_t limit) {
  const auto priors = sorted_priors(result);
  std::ostringstream out;
  const size_t count = std::min(limit, priors.size());
  for (size_t i = 0; i < count; ++i) {
    if (i != 0)
      out << ", ";
    out << move_to_string(priors[i].first) << "="
        << format_float(priors[i].second);
  }
  return out.str();
}

bool should_dump_nn_debug() {
  const char *dump = std::getenv("METALFISH_NN_DEBUG_DUMP");
  return dump && dump[0] != '\0' && std::string(dump) != "0";
}

bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] != '\0' && std::string(value) != "0";
}

int env_int_or_default(const char *name, int fallback, int min_value,
                       int max_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  try {
    const int parsed = std::stoi(value);
    return std::clamp(parsed, min_value, max_value);
  } catch (...) {
    return fallback;
  }
}

struct ParityReport {
  std::string output_path;
  std::string weights_path;
  std::string backend_info;
  std::ostringstream fixed_rows;
  std::ostringstream batch_rows;
  std::string fixed_note;
  std::string batch_note;
  bool has_fixed_rows = false;
  bool has_batch_rows = false;
};

std::unique_ptr<ParityReport> create_parity_report() {
  const char *path = std::getenv("METALFISH_NN_PARITY_REPORT");
  if (!path || path[0] == '\0')
    return nullptr;

  auto report = std::make_unique<ParityReport>();
  report->output_path = path;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  report->weights_path = weights ? weights : "";
  return report;
}

bool write_parity_report(const ParityReport &report) {
  if (report.output_path.empty())
    return true;

  std::ofstream out(report.output_path);
  if (!out) {
    std::cout << "  FAIL: could not write parity report: "
              << report.output_path << std::endl;
    return false;
  }

  out << "# MetalFish NN Parity Report\n\n";
  out << "- Weights: "
      << (report.weights_path.empty() ? "not set" : report.weights_path)
      << "\n";
  out << "- Backend: "
      << (report.backend_info.empty() ? "not resolved"
                                      : single_line(report.backend_info))
      << "\n\n";

  out << "## Fixed BT4 Reference\n\n";
  if (!report.fixed_note.empty())
    out << report.fixed_note << "\n\n";
  if (report.has_fixed_rows) {
    out << "| Case | Value actual/expected/delta | Max WDL delta | Moves-left actual/expected/delta | Max top-policy delta | Actual top policy |\n";
    out << "| --- | ---: | ---: | ---: | ---: | --- |\n";
    out << report.fixed_rows.str();
    out << "\n";
  }

  out << "## Batch Parity\n\n";
  if (!report.batch_note.empty())
    out << report.batch_note << "\n\n";
  if (report.has_batch_rows) {
    out << "| Batch | Entries | Max value delta | Max WDL delta | Max moves-left delta | Max policy delta | Worst field |\n";
    out << "| ---: | ---: | ---: | ---: | ---: | ---: | --- |\n";
    out << report.batch_rows.str();
    out << "\n";
  }

  return true;
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

bool test_bt4_reference_outputs_optional(ParityReport *report) {
  std::cout << "  BT4 fixed-position reference outputs..." << std::endl;
  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    if (report)
      report->fixed_note = "Skipped because `METALFISH_NN_WEIGHTS` is not set.";
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
      tolerances.moves_left = 2e-1f;
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
      const float value_delta = std::fabs(result.value - test_case.value);
      if (!close_enough(result.value, test_case.value, tolerances.value)) {
        std::cout << "    FAIL: " << test_case.name
                  << " value drift actual=" << result.value
                  << " expected=" << test_case.value << std::endl;
        return false;
      }
      float max_wdl_delta = 0.0f;
      for (size_t i = 0; i < test_case.wdl.size(); ++i) {
        max_wdl_delta =
            std::max(max_wdl_delta, std::fabs(result.wdl[i] - test_case.wdl[i]));
        if (!close_enough(result.wdl[i], test_case.wdl[i], tolerances.wdl)) {
          std::cout << "    FAIL: " << test_case.name << " WDL[" << i
                    << "] drift actual=" << result.wdl[i]
                    << " expected=" << test_case.wdl[i] << std::endl;
          return false;
        }
      }
      const float moves_left_delta =
          std::fabs(result.moves_left - test_case.moves_left);
      if (!close_enough(result.moves_left, test_case.moves_left,
                        tolerances.moves_left)) {
        std::cout << "    FAIL: " << test_case.name
                  << " moves-left drift actual=" << result.moves_left
                  << " expected=" << test_case.moves_left << std::endl;
        return false;
      }
      float max_top_policy_delta = 0.0f;
      for (size_t i = 0; i < test_case.top_policy.size(); ++i) {
        const auto actual_move = move_to_string(priors[i].first);
        const auto &[expected_move, expected_logit] = test_case.top_policy[i];
        if (actual_move != expected_move) {
          std::cout << "    FAIL: " << test_case.name << " top policy " << i
                    << " move drift actual=" << actual_move
                    << " expected=" << expected_move << std::endl;
          return false;
        }
        max_top_policy_delta =
            std::max(max_top_policy_delta,
                     std::fabs(priors[i].second - expected_logit));
        if (!close_enough(priors[i].second, expected_logit,
                          tolerances.policy)) {
          std::cout << "    FAIL: " << test_case.name << " top policy " << i
                    << " logit drift actual=" << priors[i].second
                    << " expected=" << expected_logit << std::endl;
          return false;
        }
      }
      if (report) {
        report->backend_info = network_info;
        report->has_fixed_rows = true;
        report->fixed_rows
            << "| " << test_case.name << " | "
            << format_float(result.value) << " / "
            << format_float(test_case.value) << " / "
            << format_float(value_delta) << " | "
            << format_float(max_wdl_delta) << " | "
            << format_float(result.moves_left) << " / "
            << format_float(test_case.moves_left) << " / "
            << format_float(moves_left_delta) << " | "
            << format_float(max_top_policy_delta) << " | "
            << result_top_policy_string(result, test_case.top_policy.size())
            << " |\n";
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

struct EvalDiffMetrics {
  float value_delta = 0.0f;
  float wdl_delta = 0.0f;
  float moves_left_delta = 0.0f;
  float policy_delta = 0.0f;
  std::string worst_field = "none";
};

EvalDiffMetrics measure_eval_result(const MCTS::EvaluationResult &single,
                                    const MCTS::EvaluationResult &batched,
                                    const std::string &label) {
  EvalDiffMetrics metrics;
  auto update_worst = [&](float delta, const std::string &field) {
    const float current_max =
        std::max({metrics.value_delta, metrics.wdl_delta,
                  metrics.moves_left_delta, metrics.policy_delta});
    if (delta >= current_max)
      metrics.worst_field = label + " " + field;
  };

  metrics.value_delta = std::fabs(single.value - batched.value);
  update_worst(metrics.value_delta, "value");

  if (single.has_wdl && batched.has_wdl) {
    for (int i = 0; i < 3; ++i) {
      const float delta = std::fabs(single.wdl[i] - batched.wdl[i]);
      if (delta > metrics.wdl_delta) {
        metrics.wdl_delta = delta;
        update_worst(delta, "wdl" + std::to_string(i));
      }
    }
  }

  if (single.has_moves_left && batched.has_moves_left) {
    metrics.moves_left_delta =
        std::fabs(single.moves_left - batched.moves_left);
    update_worst(metrics.moves_left_delta, "moves_left");
  }

  if (single.policy_priors.size() != batched.policy_priors.size()) {
    metrics.policy_delta = std::numeric_limits<float>::infinity();
    metrics.worst_field = label + " policy_size";
    return metrics;
  }

  for (size_t i = 0; i < single.policy_priors.size(); ++i) {
    if (single.policy_priors[i].first != batched.policy_priors[i].first) {
      metrics.policy_delta = std::numeric_limits<float>::infinity();
      metrics.worst_field = label + " policy_move";
      return metrics;
    }
    const float delta = std::fabs(single.policy_priors[i].second -
                                  batched.policy_priors[i].second);
    if (delta > metrics.policy_delta) {
      metrics.policy_delta = delta;
      metrics.worst_field = label + " policy " +
                            move_to_string(single.policy_priors[i].first);
    }
  }

  return metrics;
}

void merge_eval_metrics(EvalDiffMetrics &target, const EvalDiffMetrics &source) {
  if (source.value_delta > target.value_delta)
    target.value_delta = source.value_delta;
  if (source.wdl_delta > target.wdl_delta)
    target.wdl_delta = source.wdl_delta;
  if (source.moves_left_delta > target.moves_left_delta)
    target.moves_left_delta = source.moves_left_delta;
  if (source.policy_delta > target.policy_delta)
    target.policy_delta = source.policy_delta;

  const float target_max =
      std::max({target.value_delta, target.wdl_delta, target.moves_left_delta,
                target.policy_delta});
  const float source_max =
      std::max({source.value_delta, source.wdl_delta, source.moves_left_delta,
                source.policy_delta});
  if (source_max >= target_max && !source.worst_field.empty())
    target.worst_field = source.worst_field;
}

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

bool test_mcts_evaluator_batch_parity_optional(ParityReport *report) {
  std::cout << "  MCTS evaluator batch parity..." << std::endl;
  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    if (report)
      report->batch_note = "Skipped because `METALFISH_NN_WEIGHTS` is not set.";
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

    EvalTolerances tolerances;
    const std::string network_info = single_eval.GetNetworkInfo();
    if (report && report->backend_info.empty())
      report->backend_info = network_info;
    if (network_info.find("CUDA transformer backend") != std::string::npos) {
      tolerances.value = 2e-3f;
      tolerances.moves_left = 1.25e-1f;
      tolerances.policy = 7.5e-2f;
    }

    if (env_flag_enabled("METALFISH_NN_BATCH_TRACE_PAIR")) {
      const int trace_batch_size = env_int_or_default(
          "METALFISH_NN_BATCH_TRACE_BATCH", 32, 1,
          static_cast<int>(histories.size()));
      const int target =
          env_int_or_default("METALFISH_NN_BATCH_TRACE_INDEX",
                             std::min(6, trace_batch_size - 1), 0,
                             trace_batch_size - 1);
      std::vector<std::vector<const Position *>> trace_batch(
          histories.begin(), histories.begin() + trace_batch_size);
      MCTS::NNMCTSEvaluator trace_single_eval(weights_path);
      MCTS::NNMCTSEvaluator trace_batch_eval(weights_path);
      auto single_target =
          trace_single_eval.EvaluateWithHistory(histories[target]);
      auto batched_targets =
          trace_batch_eval.EvaluateBatchWithHistory(trace_batch);
      if (batched_targets.size() != static_cast<size_t>(trace_batch_size)) {
        std::cout << "    FAIL: trace pair batch result count mismatch"
                  << std::endl;
        return false;
      }
      const auto metrics = measure_eval_result(
          single_target, batched_targets[static_cast<size_t>(target)],
          "trace pair entry " + std::to_string(target));
      std::cout << "    TRACE_PAIR: batch=" << trace_batch_size
                << " entry=" << target
                << " value_delta=" << format_float(metrics.value_delta)
                << " moves_left_delta="
                << format_float(metrics.moves_left_delta)
                << " policy_delta=" << format_float(metrics.policy_delta)
                << " worst=" << metrics.worst_field << std::endl;
    }

    const size_t shrink_batch_size = std::min<size_t>(32, histories.size());
    if (shrink_batch_size > 1) {
      std::vector<std::vector<const Position *>> warm_batch(
          histories.begin(), histories.begin() + shrink_batch_size);
      (void)batch_eval.EvaluateBatchWithHistory(warm_batch);
      std::vector<std::vector<const Position *>> small_batch = {
          histories.front()};
      auto shrink_outputs = batch_eval.EvaluateBatchWithHistory(small_batch);
      if (shrink_outputs.size() != 1) {
        std::cout << "    FAIL: batch shrink result count mismatch"
                  << std::endl;
        return false;
      }
      const auto shrink_single =
          single_eval.EvaluateWithHistory(histories.front());
      if (!compare_eval_result(shrink_single, shrink_outputs.front(),
                               "batch shrink reuse entry 0", tolerances)) {
        return false;
      }
    }

    std::vector<MCTS::EvaluationResult> singles;
    singles.reserve(histories.size());
    for (const auto &history : histories) {
      singles.push_back(single_eval.EvaluateWithHistory(history));
    }

    const std::array<size_t, 7> batch_sizes = {1, 2, 4, 8, 16, 32,
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

      EvalDiffMetrics batch_max;
      for (size_t i = 0; i < batch_size; ++i) {
        std::ostringstream label;
        label << "batch " << batch_size << " entry " << i;
        if (batch_size == histories.size())
          label << " line " << (i % lines.size());
        merge_eval_metrics(batch_max,
                           measure_eval_result(singles[i], batched[i],
                                               label.str()));
        if (!compare_eval_result(singles[i], batched[i], label.str(),
                                 tolerances))
          return false;
      }
      if (report) {
        report->has_batch_rows = true;
        report->batch_rows << "| " << batch_size << " | " << batch_size
                           << " | " << format_float(batch_max.value_delta)
                           << " | " << format_float(batch_max.wdl_delta)
                           << " | "
                           << format_float(batch_max.moves_left_delta)
                           << " | " << format_float(batch_max.policy_delta)
                           << " | " << batch_max.worst_field << " |\n";
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

bool test_mcts_evaluator_first_use_stress_optional() {
  std::cout << "  MCTS evaluator first-use stress..." << std::endl;
  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return true;
  }

  try {
    StateInfo st;
    Position pos;
    pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            false, &st);
    auto fixture = build_history({});
    const std::vector<std::vector<const Position *>> one_history = {
        fixture.ptrs};

    MCTS::NNMCTSEvaluator reference_eval(weights_path);
    const auto reference = reference_eval.Evaluate(pos);
    const std::string network_info = reference_eval.GetNetworkInfo();

    EvalTolerances tolerances;
    if (network_info.find("CUDA transformer backend") != std::string::npos) {
      tolerances.value = 2e-3f;
      tolerances.moves_left = 1.25e-1f;
      tolerances.policy = 7.5e-2f;
    }

    const int iterations = env_int_or_default(
        "METALFISH_NN_FIRST_USE_STRESS_ITERS",
        network_info.find("CUDA transformer backend") != std::string::npos ? 3
                                                                            : 1,
        1, 32);

    for (int iter = 0; iter < iterations; ++iter) {
      MCTS::NNMCTSEvaluator first_eval(weights_path);
      MCTS::NNMCTSEvaluator second_eval(weights_path);

      const auto first = first_eval.Evaluate(pos);
      if (!compare_eval_result(reference, first,
                               "first-use direct iter " +
                                   std::to_string(iter),
                               tolerances)) {
        return false;
      }

      const auto second_batch =
          second_eval.EvaluateBatchWithHistory(one_history);
      if (second_batch.size() != 1) {
        std::cout << "    FAIL: first-use batch iter " << iter
                  << " returned " << second_batch.size() << " outputs"
                  << std::endl;
        return false;
      }
      if (!compare_eval_result(reference, second_batch.front(),
                               "first-use batch iter " +
                                   std::to_string(iter),
                               tolerances)) {
        return false;
      }
    }

    std::cout << "    PASS: checked " << iterations
              << " fresh evaluator pairs" << std::endl;
    return true;
  } catch (const std::exception &e) {
    std::cout << "    FAIL: exception: " << e.what() << std::endl;
    return false;
  }
}

bool benchmark_nn_batch_optional() {
  std::cout << "  NN backend batch benchmark..." << std::endl;
  if (!env_flag_enabled("METALFISH_NN_BATCH_BENCH")) {
    std::cout << "    SKIP: METALFISH_NN_BATCH_BENCH not set" << std::endl;
    return true;
  }

  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return true;
  }

  try {
    const int max_batch =
        env_int_or_default("METALFISH_NN_BENCH_MAX_BATCH", 32, 1, 128);
    const int iterations =
        env_int_or_default("METALFISH_NN_BENCH_ITERS", 2, 1, 20);

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
    fixtures.reserve(max_batch);
    std::vector<NN::InputPlanes> planes;
    planes.reserve(max_batch);
    for (int i = 0; i < max_batch; ++i) {
      fixtures.push_back(build_history(lines[i % lines.size()]));
      int transform = 0;
      planes.push_back(NN::EncodePositionForNN(
          MetalFishNN::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2,
          fixtures.back().ptrs, NN::kMoveHistory,
          NN::FillEmptyHistory::FEN_ONLY, &transform));
    }

    auto network = NN::CreateNetwork(weights_path);
    std::cout << "    backend: " << network->GetNetworkInfo() << std::endl;
    std::cout << "    batches:";
    double checksum = 0.0;
    for (int batch_size = 1; batch_size <= max_batch; batch_size *= 2) {
      std::vector<NN::InputPlanes> batch(planes.begin(),
                                         planes.begin() + batch_size);
      network->EvaluateBatch(batch);
      const auto start = std::chrono::steady_clock::now();
      for (int iter = 0; iter < iterations; ++iter) {
        const auto outputs = network->EvaluateBatch(batch);
        if (outputs.size() != static_cast<size_t>(batch_size)) {
          std::cout << std::endl
                    << "    FAIL: benchmark batch " << batch_size
                    << " returned " << outputs.size() << " outputs"
                    << std::endl;
          return false;
        }
        checksum += outputs.front().value + outputs.back().policy[0];
      }
      const auto end = std::chrono::steady_clock::now();
      const double total_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      const double batch_ms = total_ms / static_cast<double>(iterations);
      const double eval_ms = batch_ms / static_cast<double>(batch_size);
      std::cout << " b" << batch_size << "=" << std::fixed
                << std::setprecision(3) << batch_ms << "ms/"
                << std::setprecision(4) << eval_ms << "ms_eval";
    }
    std::cout << " checksum=" << std::setprecision(6) << checksum
              << std::endl;
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
  auto parity_report = create_parity_report();

  std::cout << "=== NN Comparison Smoke ===" << std::endl;
  const bool ok1 = test_encoder_policy_roundtrip();
  const bool ok2 = test_encoder_repetition_plane();
  const bool ok3 = test_mcts_evaluator_optional();
  const bool ok4 = test_bt4_reference_outputs_optional(parity_report.get());
  const bool ok5 =
      test_mcts_evaluator_batch_parity_optional(parity_report.get());
  const bool ok6 = test_mcts_evaluator_first_use_stress_optional();
  const bool ok7 = benchmark_nn_batch_optional();
  const bool ok8 =
      !parity_report || write_parity_report(*parity_report);
  return (ok1 && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8) ? 0 : 1;
}

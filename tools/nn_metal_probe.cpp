/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "core/bitboard.h"
#include "core/position.h"
#include "nn/encoder.h"
#include "nn/loader.h"
#include "nn/network.h"
#include "nn/network_format.h"
#include "nn/policy_map.h"
#include "search/tt.h"
#include "syzygy/tbprobe.h"
#include "uci/uci.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace MetalFish;

namespace {

struct Options {
  std::string weights;
  std::string backend = "metal";
  std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  int top = 8;
  bool full_input = false;
  bool full_policy = false;
};

std::string JsonEscape(const std::string &input) {
  std::ostringstream out;
  for (unsigned char c : input) {
    switch (c) {
    case '"':
      out << "\\\"";
      break;
    case '\\':
      out << "\\\\";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      if (std::iscntrl(c)) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(c) << std::dec << std::setfill(' ');
      } else {
        out << c;
      }
      break;
    }
  }
  return out.str();
}

std::string SquareString(Square square) {
  return std::string{static_cast<char>('a' + file_of(square)),
                     static_cast<char>('1' + rank_of(square))};
}

std::string MoveString(Move move) {
  if (move == Move::none())
    return "(none)";
  if (move == Move::null())
    return "0000";

  std::string out = SquareString(move.from_sq()) + SquareString(move.to_sq());
  if (move.type_of() == PROMOTION)
    out += " pnbrqk"[move.promotion_type()];
  return out;
}

void PrintUsage(const char *argv0) {
  std::cerr << "Usage: " << argv0
            << " --weights <file.pb[.gz]> [--backend metal|auto]"
               " [--fen <fen>] [--top n] [--full-input] [--full-policy]\n";
}

Options ParseArgs(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc)
        throw std::runtime_error(std::string("Missing value for ") + name);
      return argv[++i];
    };

    if (arg == "--weights") {
      options.weights = require_value("--weights");
    } else if (arg == "--backend") {
      options.backend = require_value("--backend");
    } else if (arg == "--fen") {
      options.fen = require_value("--fen");
    } else if (arg == "--top") {
      options.top = std::stoi(require_value("--top"));
    } else if (arg == "--full-input") {
      options.full_input = true;
    } else if (arg == "--full-policy") {
      options.full_policy = true;
    } else if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.weights.empty())
    throw std::runtime_error("--weights is required");
  if (options.top < 0)
    throw std::runtime_error("--top must be non-negative");
  if (options.backend.empty())
    throw std::runtime_error("--backend must be non-empty");
  return options;
}

void PrintPolicyArray(const std::array<float, NN::kPolicyOutputs> &policy) {
  std::cout << ",\"policy\":[";
  for (int i = 0; i < NN::kPolicyOutputs; ++i) {
    if (i != 0)
      std::cout << ',';
    std::cout << policy[i];
  }
  std::cout << ']';
}

void PrintInputArray(const NN::InputPlanes &planes) {
  std::cout << ",\"input\":[";
  bool first = true;
  for (int sq = 0; sq < 64; ++sq) {
    for (int plane = 0; plane < NN::kTotalPlanes; ++plane) {
      if (!first)
        std::cout << ',';
      first = false;
      std::cout << planes[plane][sq];
    }
  }
  std::cout << ']';
}

void PrintTopPolicy(const std::array<float, NN::kPolicyOutputs> &policy,
                    int transform, int top) {
  const int count = std::min(top, NN::kPolicyOutputs);
  std::vector<int> indices(NN::kPolicyOutputs);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                    [&](int lhs, int rhs) { return policy[lhs] > policy[rhs]; });

  std::cout << ",\"policy_top\":[";
  for (int rank = 0; rank < count; ++rank) {
    if (rank != 0)
      std::cout << ',';
    const int index = indices[rank];
    std::cout << "{\"index\":" << index << ",\"move\":\""
              << JsonEscape(MoveString(NN::IndexToNNMove(index, transform)))
              << "\",\"logit\":" << policy[index] << '}';
  }
  std::cout << ']';
}

void RunProbe(const Options &options) {
  Bitboards::init();
  Position::init();
  NN::InitPolicyTables();

  NN::WeightsFile weights = NN::LoadWeightsFromFile(options.weights);
  const auto input_format = weights.format().network_format().input();
  const auto descriptor = NN::DescribeNetworkFormat(weights);

  StateInfo state;
  Position position;
  position.set(options.fen, false, &state);

  const Position *history_storage[] = {&position};
  std::span<const Position *const> history(history_storage, 1);
  int transform = 0;
  const auto planes = NN::EncodePositionForNN(
      input_format, history, NN::kMoveHistory, NN::FillEmptyHistory::FEN_ONLY,
      &transform);

  auto network = NN::CreateNetwork(weights, options.backend);
  const auto output = network->Evaluate(planes);

  std::cout << std::setprecision(9);
  std::cout << '{';
  std::cout << "\"fen\":\"" << JsonEscape(options.fen) << '"';
  std::cout << ",\"weights\":\"" << JsonEscape(options.weights) << '"';
  std::cout << ",\"backend\":\"" << JsonEscape(options.backend) << '"';
  std::cout << ",\"network_info\":\""
            << JsonEscape(network->GetNetworkInfo()) << '"';
  std::cout << ",\"format\":\"" << JsonEscape(descriptor.Summary()) << '"';
  std::cout << ",\"input_format\":" << static_cast<int>(input_format);
  std::cout << ",\"transform\":" << transform;
  std::cout << ",\"value\":" << output.value;
  std::cout << ",\"has_wdl\":" << (output.has_wdl ? "true" : "false");
  std::cout << ",\"wdl\":[" << output.wdl[0] << ',' << output.wdl[1] << ','
            << output.wdl[2] << ']';
  std::cout << ",\"has_moves_left\":"
            << (output.has_moves_left ? "true" : "false");
  std::cout << ",\"moves_left\":" << output.moves_left;
  PrintTopPolicy(output.policy, transform, options.top);
  if (options.full_input)
    PrintInputArray(planes);
  if (options.full_policy)
    PrintPolicyArray(output.policy);
  std::cout << "}\n";
}

} // namespace

namespace MetalFish {

std::string UCIEngine::square(Square square) { return SquareString(square); }

TTEntry *TranspositionTable::first_entry(const Key) const { return nullptr; }

namespace Tablebases {

int MaxCardinality = 0;

WDLScore probe_wdl(Position &, ProbeState *result) {
  if (result != nullptr)
    *result = FAIL;
  return WDLDraw;
}

int probe_dtz(Position &, ProbeState *result) {
  if (result != nullptr)
    *result = FAIL;
  return 0;
}

} // namespace Tablebases

} // namespace MetalFish

int main(int argc, char **argv) {
  try {
    RunProbe(ParseArgs(argc, argv));
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "metalfish_nn_probe: " << e.what() << '\n';
    PrintUsage(argv[0]);
    return 1;
  }
}

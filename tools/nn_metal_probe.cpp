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
#include "nn/network_execution_plan.h"
#include "nn/network_format.h"
#include "nn/network_tensor_plan.h"
#include "nn/network_weight_inventory.h"
#include "nn/policy_map.h"
#include "search/tt.h"
#include "syzygy/tbprobe.h"
#include "uci/uci.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace MetalFish;

namespace {

struct Options {
  std::string weights;
  std::string backend = "metal";
  std::string coreml_model;
  std::string coreml_compute_units = "cpu-ne";
  std::string fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  int top = 8;
  int batch_size = 1;
  int warmup = 1;
  int iterations = 1;
  bool full_input = false;
  bool full_policy = false;
  bool metadata_only = false;
  std::string ready_file;
  std::string start_file;
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
  std::cerr
      << "Usage: " << argv0
      << " --weights <file.pb[.gz]> [--backend auto|metal|cuda|cpu|coreml]"
         " [--coreml-model model.mlpackage]"
         " [--coreml-compute-units cpu|cpu-gpu|cpu-ne|all]"
         " [--fen <fen>] [--top n] [--batch-size n] [--warmup n]"
         " [--iterations n] [--full-input] [--full-policy]"
         " [--metadata-only]"
         " [--ready-file path] [--start-file path]\n";
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
    } else if (arg == "--coreml-model") {
      options.coreml_model = require_value("--coreml-model");
    } else if (arg == "--coreml-compute-units") {
      options.coreml_compute_units = require_value("--coreml-compute-units");
    } else if (arg == "--fen") {
      options.fen = require_value("--fen");
    } else if (arg == "--top") {
      options.top = std::stoi(require_value("--top"));
    } else if (arg == "--batch-size") {
      options.batch_size = std::stoi(require_value("--batch-size"));
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(require_value("--warmup"));
    } else if (arg == "--iterations") {
      options.iterations = std::stoi(require_value("--iterations"));
    } else if (arg == "--full-input") {
      options.full_input = true;
    } else if (arg == "--full-policy") {
      options.full_policy = true;
    } else if (arg == "--metadata-only") {
      options.metadata_only = true;
    } else if (arg == "--ready-file") {
      options.ready_file = require_value("--ready-file");
    } else if (arg == "--start-file") {
      options.start_file = require_value("--start-file");
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
  if (options.batch_size < 1)
    throw std::runtime_error("--batch-size must be positive");
  if (options.warmup < 0)
    throw std::runtime_error("--warmup must be non-negative");
  if (options.iterations < 1)
    throw std::runtime_error("--iterations must be positive");
  if (options.backend.empty())
    throw std::runtime_error("--backend must be non-empty");
  if (!options.metadata_only && options.backend == "coreml" &&
      options.coreml_model.empty())
    throw std::runtime_error("--coreml-model is required for backend coreml");
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
  std::partial_sort(
      indices.begin(), indices.begin() + count, indices.end(),
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

double Mean(const std::vector<double> &values) {
  double sum = 0.0;
  for (double value : values)
    sum += value;
  return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

double Median(std::vector<double> values) {
  if (values.empty())
    return 0.0;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void TouchFile(const std::string &path) {
  std::ofstream file(path);
  if (!file)
    throw std::runtime_error("Could not create signal file: " + path);
}

void WaitForFile(const std::string &path) {
  while (!std::filesystem::exists(path))
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

void PrintMetadataOnly(const Options &options, const NN::WeightsFile &weights,
                       const NN::NetworkFormatDescriptor &descriptor) {
  const auto tensor_plan = NN::CreateNetworkTensorPlan(descriptor);
  NN::MultiHeadWeights decoded_weights(weights.weights());
  const std::string policy_head = NN::SelectPolicyHeadName(decoded_weights);
  const std::string value_head = NN::SelectValueHeadName(decoded_weights);

  const auto tensor_validation = NN::ValidateNetworkTensorPlan(
      tensor_plan, decoded_weights, policy_head, value_head);
  if (!tensor_validation.ok())
    throw std::runtime_error("tensor plan validation failed: " +
                             tensor_validation.Summary());

  const auto inventory = NN::CreateNetworkWeightInventory(
      decoded_weights, policy_head, value_head, tensor_plan);
  std::string shape_error;
  if (!inventory.AllShapesMatchElements(&shape_error))
    throw std::runtime_error("weight inventory shape validation failed: " +
                             shape_error);

  const auto execution_plan = NN::CreateNetworkExecutionPlan(
      descriptor, tensor_plan, policy_head, value_head, inventory);
  const auto execution_validation =
      execution_plan.ValidateAgainstInventory(inventory);
  if (!execution_validation.ok())
    throw std::runtime_error("execution plan validation failed: " +
                             execution_validation.Summary());

  const auto resolved_plan =
      NN::ResolveNetworkExecutionPlan(execution_plan, inventory);

  std::cout << std::setprecision(9);
  std::cout << '{';
  std::cout << "\"weights\":\"" << JsonEscape(options.weights) << '"';
  std::cout << ",\"backend\":\"" << JsonEscape(options.backend) << '"';
  std::cout << ",\"metadata_only\":true";
  std::cout << ",\"format\":\"" << JsonEscape(descriptor.Summary()) << '"';
  std::cout << ",\"input_format\":"
            << static_cast<int>(weights.format().network_format().input());
  std::cout << ",\"tensor_plan\":\""
            << JsonEscape(tensor_plan.Summary()) << '"';
  std::cout << ",\"policy_head\":\"" << JsonEscape(policy_head) << '"';
  std::cout << ",\"value_head\":\"" << JsonEscape(value_head) << '"';
  std::cout << ",\"inventory\":\"" << JsonEscape(inventory.Summary())
            << '"';
  std::cout << ",\"execution_plan\":\""
            << JsonEscape(resolved_plan.Summary()) << '"';
  std::cout << ",\"tensor_count\":" << inventory.tensors.size();
  std::cout << ",\"parameter_elements\":" << inventory.TotalElements();
  std::cout << ",\"parameter_bytes\":" << inventory.TotalBytes();
  std::cout << ",\"steps\":" << resolved_plan.steps.size();
  std::cout << ",\"attention_steps\":"
            << resolved_plan.StepCount(NN::NetworkExecutionOpKind::Attention);
  std::cout << ",\"feed_forward_steps\":"
            << resolved_plan.StepCount(NN::NetworkExecutionOpKind::FeedForward);
  std::cout << "}\n";
}

void RunProbe(const Options &options) {
  Bitboards::init();
  Position::init();
  NN::InitPolicyTables();

  NN::WeightsFile weights = NN::LoadWeightsFromFile(options.weights);
  const auto input_format = weights.format().network_format().input();
  const auto descriptor = NN::DescribeNetworkFormat(weights);

  if (options.metadata_only) {
    PrintMetadataOnly(options, weights, descriptor);
    return;
  }

  StateInfo state;
  Position position;
  position.set(options.fen, false, &state);

  const Position *history_storage[] = {&position};
  std::span<const Position *const> history(history_storage, 1);
  int transform = 0;
  const auto planes =
      NN::EncodePositionForNN(input_format, history, NN::kMoveHistory,
                              NN::FillEmptyHistory::FEN_ONLY, &transform);

  auto network =
      NN::CreateNetwork(weights, options.backend, options.coreml_model,
                        options.coreml_compute_units);
  const std::vector<NN::InputPlanes> batch_inputs(options.batch_size, planes);
  for (int i = 0; i < options.warmup; ++i)
    (void)network->EvaluateBatch(batch_inputs);
  if (!options.ready_file.empty())
    TouchFile(options.ready_file);
  if (!options.start_file.empty())
    WaitForFile(options.start_file);

  NN::NetworkOutput output;
  std::vector<double> latencies;
  latencies.reserve(options.iterations);
  for (int i = 0; i < options.iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    const auto outputs = network->EvaluateBatch(batch_inputs);
    const auto end = std::chrono::steady_clock::now();
    output = outputs.front();
    latencies.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }

  std::cout << std::setprecision(9);
  std::cout << '{';
  std::cout << "\"fen\":\"" << JsonEscape(options.fen) << '"';
  std::cout << ",\"weights\":\"" << JsonEscape(options.weights) << '"';
  std::cout << ",\"backend\":\"" << JsonEscape(options.backend) << '"';
  std::cout << ",\"network_info\":\"" << JsonEscape(network->GetNetworkInfo())
            << '"';
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
  std::cout << ",\"latency\":{\"warmup\":" << options.warmup
            << ",\"iterations\":" << options.iterations
            << ",\"batch_size\":" << options.batch_size
            << ",\"median_ms\":" << Median(latencies)
            << ",\"median_positions_per_second\":"
            << (1000.0 * static_cast<double>(options.batch_size) /
                Median(latencies))
            << ",\"mean_ms\":" << Mean(latencies) << ",\"min_ms\":"
            << *std::min_element(latencies.begin(), latencies.end())
            << ",\"max_ms\":"
            << *std::max_element(latencies.begin(), latencies.end()) << '}';
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

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MCTS smoke tests for the current Search/Node pipeline.
*/

#include "core/bitboard.h"
#include "core/movegen.h"
#include "core/position.h"
#include "mcts/backend_adapter.h"
#include "mcts/core.h"
#include "mcts/evaluator.h"
#include "mcts/search.h"
#include "nn/loader.h"
#include "nn/network.h"
#include "nn/network_execution_plan.h"
#include "nn/network_format.h"
#include "nn/network_output_decoder.h"
#include "nn/network_tensor_plan.h"
#include "nn/network_weight_inventory.h"
#include "nn_input_fixture.h"
#ifdef USE_CUDA
#include "nn/cuda/cuda_attention_plan.h"
#include "nn/cuda/cuda_buffers.h"
#include "nn/cuda/cuda_execution_schedule.h"
#include "nn/cuda/cuda_execution_tape.h"
#include "nn/cuda/cuda_input_packing.h"
#include "nn/cuda/cuda_kernels.h"
#include "nn/cuda/cuda_output_mapping.h"
#include "nn/cuda/cuda_plan_analysis.h"
#include "nn/cuda/cuda_runtime_probe.h"
#include "nn/cuda/cuda_weight_buffers.h"
#include "nn/cuda/cuda_workspace.h"
#endif
#include "syzygy/tbprobe.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace MetalFish;
using namespace MetalFish::MCTS;
using namespace MetalFish::Tests;

namespace {

struct TestCounter {
  int passed = 0;
  int failed = 0;
};

void expect(bool cond, const char *msg, TestCounter &tc) {
  if (cond) {
    tc.passed++;
  } else {
    tc.failed++;
    std::cout << "    FAIL: " << msg << std::endl;
  }
}

void set_test_float_layer(MetalFishNN::Weights::Layer *layer) {
  const float value = 1.0f;
  std::string bytes(sizeof(value), '\0');
  std::memcpy(bytes.data(), &value, sizeof(value));
  layer->set_encoding(MetalFishNN::Weights::Layer::FLOAT32);
  layer->set_params(bytes);
  layer->add_dims(1);
}

#ifdef USE_CUDA
std::size_t
find_resolved_step_index(const NN::NetworkResolvedExecutionPlan &plan,
                         const std::string &name) {
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    if (plan.steps[i].name == name)
      return i;
  }
  throw std::runtime_error("missing resolved execution step: " + name);
}
#endif

NN::WeightsFile make_minimal_attention_weights_file() {
  NN::WeightsFile file;
  auto *nf = file.mutable_format()->mutable_network_format();
  nf->set_network(
      MetalFishNN::
          NetworkFormat_NetworkStructure_NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT);
  nf->set_policy(MetalFishNN::NetworkFormat_PolicyFormat_POLICY_ATTENTION);
  nf->set_value(MetalFishNN::NetworkFormat_ValueFormat_VALUE_WDL);
  nf->set_moves_left(MetalFishNN::NetworkFormat_MovesLeftFormat_MOVES_LEFT_V1);
  nf->set_input_embedding(
      MetalFishNN::NetworkFormat_InputEmbeddingFormat_INPUT_EMBEDDING_PE_DENSE);
  nf->set_default_activation(
      MetalFishNN::NetworkFormat_DefaultActivation_DEFAULT_ACTIVATION_MISH);
  nf->set_smolgen_activation(
      MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_DEFAULT);
  nf->set_ffn_activation(
      MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_SWISH);

  auto *weights = file.mutable_weights();
  set_test_float_layer(
      weights->mutable_policy_heads()->mutable_vanilla()->mutable_ip_pol_w());
  set_test_float_layer(
      weights->mutable_value_heads()->mutable_winner()->mutable_ip_val_w());
  set_test_float_layer(weights->mutable_moves_left()->mutable_weights());
  return file;
}

void test_node_basics(TestCounter &tc) {
  std::cout << "  Node basics..." << std::endl;
  constexpr size_t node_size_budget =
      CACHE_LINE_SIZE == 128 ? CACHE_LINE_SIZE : 3 * CACHE_LINE_SIZE;
  expect(alignof(Node) == CACHE_LINE_SIZE, "node cache-line alignment", tc);
  expect(sizeof(Node) <= node_size_budget, "node fits size budget", tc);

  Node n;
  expect(n.GetN() == 0, "initial N", tc);
  expect(n.GetNInFlight() == 0, "initial N_in_flight", tc);
  expect(!n.IsTerminal(), "initial non-terminal", tc);

  expect(n.TryStartScoreUpdate(2), "start score update", tc);
  expect(n.GetNInFlight() >= 2, "virtual visits set", tc);
  n.FinalizeScoreUpdate(0.5f, 0.1f, 20.0f, 2);
  expect(n.GetN() == 2, "finalized multivisit", tc);
  expect(n.GetNInFlight() == 0, "in-flight cleared", tc);
}

void test_tablebase_wdl_conversion(TestCounter &tc) {
  std::cout << "  Tablebase WDL conversion..." << std::endl;

  expect(TablebaseWDLToParentWL(Tablebases::WDLWin) == -1.0f,
         "side-to-move TB win is parent loss", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLLoss) == 1.0f,
         "side-to-move TB loss is parent win", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLDraw) == 0.0f,
         "TB draw has neutral WL", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLCursedWin) == 0.0f,
         "cursed TB win is draw-equivalent", tc);
  expect(TablebaseWDLToParentWL(Tablebases::WDLBlessedLoss) == 0.0f,
         "blessed TB loss is draw-equivalent", tc);

  expect(TablebaseWDLToDraw(Tablebases::WDLWin) == 0.0f,
         "decisive TB win has no draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLLoss) == 0.0f,
         "decisive TB loss has no draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLDraw) == 1.0f,
         "TB draw has full draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLCursedWin) == 1.0f,
         "cursed TB win has full draw mass", tc);
  expect(TablebaseWDLToDraw(Tablebases::WDLBlessedLoss) == 1.0f,
         "blessed TB loss has full draw mass", tc);
}

void test_root_search_smoke(TestCounter &tc) {
  std::cout << "  Search smoke..." << std::endl;
  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path.clear(); // Smoke path without NN backend.
  auto search = CreateSearch(params);
  expect(static_cast<bool>(search), "search object created", tc);

  MetalFish::Search::LimitsType limits;
  limits.nodes = 8;
  search->StartSearch(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      nullptr, nullptr);
  search->Wait();

  const auto &stats = search->Stats();
  expect(stats.total_nodes.load() > 0, "search produced visits", tc);
}

void test_pv_boost_respects_weight(TestCounter &tc) {
  std::cout << "  PV boost weight..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights || std::string(weights).empty()) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 8;

  auto search = CreateSearch(params);
  search->StartSearch(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      nullptr, nullptr);
  search->Wait();

  auto root_stats = search->GetRootMoveStats();
  expect(!root_stats.empty(), "root stats should be available for PV boost",
         tc);
  if (root_stats.empty())
    return;

  const Move target = root_stats.front().move;
  auto policy_for = [&](Move move) {
    for (const auto &entry : search->GetRootMoveStats()) {
      if (entry.move == move)
        return entry.policy;
    }
    return -1.0f;
  };
  auto policy_sum = [&]() {
    float total = 0.0f;
    for (const auto &entry : search->GetRootMoveStats())
      total += entry.policy;
    return total;
  };

  const float before = policy_for(target);
  const float total_before = policy_sum();
  Move pv[] = {target};

  search->InjectPVBoost(pv, 1, 20, 0.0f);
  expect(std::abs(policy_for(target) - before) < 1e-6f,
         "zero PV boost weight should leave policy unchanged", tc);

  search->InjectPVBoost(pv, 1, 20, 0.5f);
  const float boosted = policy_for(target);
  const float total_after = policy_sum();
  expect(boosted > before, "positive PV boost weight should raise PV policy",
         tc);
  expect(std::abs(total_after - total_before) < 1e-4f,
         "PV boost should preserve root policy mass", tc);
}

void test_search_params_defaults(TestCounter &tc) {
  std::cout << "  Search params..." << std::endl;
  SearchParams params;
  expect(params.fpu_reduction_at_root == 0.33f, "root FPU default aligned", tc);
  expect(params.smart_pruning_factor == 1.33f, "smart pruning default aligned",
         tc);
  expect(params.kld_gain_min == 0.00005f, "KLD stopper tactical default", tc);
  expect(params.cache_history_length == 0,
         "classic cache history default uses current position", tc);
  expect(params.nn_cache_size == 2000000, "NN cache default aligned", tc);
  expect(params.moves_left_max_effect == 0.0345f,
         "moves-left max effect default aligned", tc);
  expect(params.moves_left_scaled_factor == 1.6521f,
         "moves-left scaled factor default aligned", tc);
  expect(params.moves_left_quadratic_factor == -0.6521f,
         "moves-left quadratic factor default aligned", tc);
  expect(params.temp_winpct_cutoff == 100.0f,
         "temperature value cutoff default aligned", tc);
  expect(params.GetCpuctBase(true) == params.cpuct_base_at_root,
         "root cpuct base getter", tc);
  expect(params.GetFpuValue(true) == params.fpu_value_at_root,
         "root fpu value getter", tc);
}

void test_network_format_descriptor(TestCounter &tc) {
  std::cout << "  Network format descriptor..." << std::endl;

  NN::WeightsFile file = make_minimal_attention_weights_file();

  const auto descriptor = NN::DescribeNetworkFormat(file);
  expect(descriptor.attention_body, "attention body should be detected", tc);
  expect(descriptor.attention_policy, "attention policy should be detected",
         tc);
  expect(descriptor.wdl, "WDL value head should be detected", tc);
  expect(descriptor.moves_left, "moves-left head should be detected", tc);
  expect(descriptor.input_embedding == NN::INPUT_EMBEDDING_PE_DENSE,
         "input embedding should be preserved", tc);
  expect(descriptor.activations.default_activation == "mish",
         "default activation should decode", tc);
  expect(descriptor.activations.smolgen_activation == "mish",
         "default smolgen activation should inherit default", tc);
  expect(descriptor.activations.ffn_activation == "swish",
         "FFN activation should decode", tc);
  expect(descriptor.Summary().find("policy=attention") != std::string::npos,
         "descriptor summary should include policy type", tc);

  const auto tensor_plan = NN::CreateNetworkTensorPlan(descriptor);
  expect(tensor_plan.input_planes == NN::kTotalPlanes,
         "tensor plan should use 112 input planes", tc);
  expect(tensor_plan.policy_outputs == NN::kPolicyOutputs,
         "tensor plan should use 1858 policy outputs", tc);
  expect(tensor_plan.value_outputs == 3,
         "tensor plan should expose WDL value width", tc);
  expect(tensor_plan.moves_left_outputs == 1,
         "tensor plan should expose moves-left width", tc);
  expect(tensor_plan.raw_policy_outputs == NN::kNetworkAttentionPolicyScratch,
         "tensor plan should expose attention policy scratch width", tc);

  MetalFishNN::Weights minimal_proto = file.weights();
  NN::MultiHeadWeights minimal_weights(minimal_proto);
  const auto valid = NN::ValidateNetworkTensorPlan(
      tensor_plan, minimal_weights, NN::SelectPolicyHeadName(minimal_weights),
      NN::SelectValueHeadName(minimal_weights));
  expect(valid.ok(), "tensor plan validation should accept required heads", tc);

  minimal_proto.clear_moves_left();
  NN::MultiHeadWeights missing_moves_left(minimal_proto);
  const auto invalid = NN::ValidateNetworkTensorPlan(
      tensor_plan, missing_moves_left,
      NN::SelectPolicyHeadName(missing_moves_left),
      NN::SelectValueHeadName(missing_moves_left));
  expect(!invalid.ok(),
         "tensor plan validation should reject missing moves-left weights", tc);
}

void test_shared_nn_input_fixture(TestCounter &tc) {
  std::cout << "  Shared NN input fixture..." << std::endl;

  const auto fixture = BuildStartPositionPackedInputFixture();
  std::string error;
  const bool valid = ValidateStartPositionPackedInputFixture(fixture, &error);
  if (!valid)
    std::cout << "    " << error << std::endl;
  expect(valid, "start position packed input fixture should match constants",
         tc);
  expect(fixture.masks[0] == 0x000000000000ff00ULL,
         "fixture should encode white pawns on plane 0", tc);
  expect(fixture.masks[NN::kAuxPlaneBase + 7] == kFullPlaneMask,
         "fixture should encode classical constant plane", tc);
}

void test_network_weight_inventory(TestCounter &tc) {
  std::cout << "  Network weight inventory..." << std::endl;

  const auto file = make_minimal_attention_weights_file();
  const auto descriptor = NN::DescribeNetworkFormat(file);
  const auto tensor_plan = NN::CreateNetworkTensorPlan(descriptor);
  NN::MultiHeadWeights weights(file.weights());
  const std::string policy_head = NN::SelectPolicyHeadName(weights);
  const std::string value_head = NN::SelectValueHeadName(weights);

  const auto validation = NN::ValidateNetworkTensorPlan(
      tensor_plan, weights, policy_head, value_head);
  expect(validation.ok(), "minimal tensor plan should validate", tc);

  const auto inventory = NN::CreateNetworkWeightInventory(
      weights, policy_head, value_head, tensor_plan);
  std::string shape_error;
  expect(inventory.AllShapesMatchElements(&shape_error),
         "minimal fixture tensor shapes should match element counts", tc);
  expect(inventory.Contains("policy.vanilla.ip_pol_w"),
         "inventory should include selected policy head", tc);
  expect(inventory.Contains("value.winner.ip_val_w"),
         "inventory should include selected value head", tc);
  expect(inventory.Contains("moves_left.weights"),
         "inventory should include declared moves-left tensors", tc);
  const auto *policy_tensor = inventory.Find("policy.vanilla.ip_pol_w");
  expect(policy_tensor &&
             policy_tensor->kind == NN::NetworkWeightTensorKind::DenseWeight,
         "inventory should tag dense policy weights", tc);
  expect(policy_tensor && policy_tensor->ShapeString() == "1",
         "inventory should preserve layer dimensions", tc);
  const auto *moves_left_tensor = inventory.Find("moves_left.weights");
  expect(moves_left_tensor &&
             moves_left_tensor->kind == NN::NetworkWeightTensorKind::ConvWeight,
         "inventory should tag convolution weights", tc);
  expect(inventory.tensors.size() == 3,
         "minimal fixture should inventory exactly three tensors", tc);
  expect(inventory.TotalElements() == 3,
         "minimal fixture should inventory exactly three float elements", tc);
  expect(inventory.TotalBytes() == 3 * sizeof(float),
         "inventory byte accounting should match float elements", tc);

  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path || std::string(weights_path).empty()) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  const auto loaded = NN::LoadWeightsFromFile(weights_path);
  const auto loaded_descriptor = NN::DescribeNetworkFormat(loaded);
  const auto loaded_plan = NN::CreateNetworkTensorPlan(loaded_descriptor);
  NN::MultiHeadWeights loaded_weights(loaded.weights());
  const std::string loaded_policy_head =
      NN::SelectPolicyHeadName(loaded_weights);
  const std::string loaded_value_head = NN::SelectValueHeadName(loaded_weights);
  const auto loaded_validation = NN::ValidateNetworkTensorPlan(
      loaded_plan, loaded_weights, loaded_policy_head, loaded_value_head);
  expect(loaded_validation.ok(),
         "loaded BT4 tensor plan should validate selected heads", tc);

  const auto loaded_inventory = NN::CreateNetworkWeightInventory(
      loaded_weights, loaded_policy_head, loaded_value_head, loaded_plan);
  std::cout << "    Loaded inventory: " << loaded_inventory.Summary()
            << std::endl;
  std::string loaded_shape_error;
  expect(loaded_inventory.AllShapesMatchElements(&loaded_shape_error),
         "loaded BT4 inventory tensor shapes should match element counts", tc);
  expect(!loaded_inventory.tensors.empty(),
         "loaded BT4 inventory should expose tensors", tc);
  expect(loaded_inventory.TotalElements() > 1000000,
         "loaded BT4 inventory should include full transformer weights", tc);
  expect(
      loaded_inventory.Contains("policy." + loaded_policy_head + ".ip_pol_w"),
      "loaded inventory should include selected policy head output", tc);
  expect(loaded_inventory.Contains("value." + loaded_value_head + ".ip_val_w"),
         "loaded inventory should include selected value head output", tc);
}

void test_network_execution_plan(TestCounter &tc) {
  std::cout << "  Network execution plan..." << std::endl;

  const auto file = make_minimal_attention_weights_file();
  const auto descriptor = NN::DescribeNetworkFormat(file);
  const auto tensor_plan = NN::CreateNetworkTensorPlan(descriptor);
  NN::MultiHeadWeights weights(file.weights());
  const std::string policy_head = NN::SelectPolicyHeadName(weights);
  const std::string value_head = NN::SelectValueHeadName(weights);
  const auto inventory =
      NN::CreateNetworkWeightInventory(weights, policy_head, value_head,
                                       tensor_plan);
  const auto execution_plan = NN::CreateNetworkExecutionPlan(
      descriptor, tensor_plan, policy_head, value_head, inventory);
  const auto validation = execution_plan.ValidateAgainstInventory(inventory);
  const auto resolved_plan =
      NN::ResolveNetworkExecutionPlan(execution_plan, inventory);
  if (!validation.ok())
    std::cout << "    " << validation.Summary() << std::endl;
  expect(validation.ok(), "minimal execution plan should cover all tensors",
         tc);
  expect(execution_plan.ContainsStep("input.pack"),
         "execution plan should begin with input packing", tc);
  expect(execution_plan.ContainsStep("policy.vanilla.output"),
         "execution plan should include selected policy output", tc);
  expect(execution_plan.ContainsStep("value.winner.output"),
         "execution plan should include selected value output", tc);
  expect(execution_plan.ReferencesTensor("moves_left.weights"),
         "execution plan should reference moves-left weights", tc);
  expect(execution_plan.TensorReferenceCount() == inventory.tensors.size(),
         "minimal execution plan should reference each tensor once", tc);
  expect(resolved_plan.TotalParameterElements() == inventory.TotalElements(),
         "resolved execution plan should account for all parameters", tc);
  expect(resolved_plan.TotalParameterBytes() == inventory.TotalBytes(),
         "resolved execution plan byte accounting should match inventory", tc);
  expect(resolved_plan.StepCount(NN::NetworkExecutionOpKind::Dense) == 2,
         "minimal resolved plan should expose dense policy/value outputs", tc);

  const char *weights_path = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights_path || std::string(weights_path).empty()) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  const auto loaded = NN::LoadWeightsFromFile(weights_path);
  const auto loaded_descriptor = NN::DescribeNetworkFormat(loaded);
  const auto loaded_tensor_plan =
      NN::CreateNetworkTensorPlan(loaded_descriptor);
  NN::MultiHeadWeights loaded_weights(loaded.weights());
  const std::string loaded_policy_head =
      NN::SelectPolicyHeadName(loaded_weights);
  const std::string loaded_value_head =
      NN::SelectValueHeadName(loaded_weights);
  const auto loaded_inventory = NN::CreateNetworkWeightInventory(
      loaded_weights, loaded_policy_head, loaded_value_head,
      loaded_tensor_plan);
  const auto loaded_execution_plan = NN::CreateNetworkExecutionPlan(
      loaded_descriptor, loaded_tensor_plan, loaded_policy_head,
      loaded_value_head, loaded_inventory);
  const auto loaded_validation =
      loaded_execution_plan.ValidateAgainstInventory(loaded_inventory);
  const auto loaded_resolved_plan =
      NN::ResolveNetworkExecutionPlan(loaded_execution_plan, loaded_inventory);
  if (!loaded_validation.ok())
    std::cout << "    " << loaded_validation.Summary() << std::endl;
  std::cout << "    Loaded execution: " << loaded_resolved_plan.Summary()
            << std::endl;
  expect(loaded_validation.ok(),
         "loaded BT4 execution plan should cover all tensors", tc);
  expect(loaded_execution_plan.steps.size() > 20,
         "loaded BT4 execution plan should expose transformer stages", tc);
  expect(loaded_execution_plan.ContainsStep("body.encoder.0.mha"),
         "loaded BT4 execution plan should include body attention", tc);
  expect(loaded_execution_plan.ContainsStep("body.encoder.0.ffn"),
         "loaded BT4 execution plan should include body FFN", tc);
  expect(loaded_execution_plan.ReferencesTensor(
             "policy." + loaded_policy_head + ".ip_pol_w"),
         "loaded execution plan should reference selected policy output", tc);
  expect(loaded_execution_plan.ReferencesTensor(
             "value." + loaded_value_head + ".ip_val_w"),
         "loaded execution plan should reference selected value output", tc);
  expect(loaded_resolved_plan.TotalParameterElements() ==
             loaded_inventory.TotalElements(),
         "loaded resolved plan should account for all BT4 parameters", tc);
  expect(loaded_resolved_plan.TotalParameterBytes() ==
             loaded_inventory.TotalBytes(),
         "loaded resolved plan byte accounting should match BT4 inventory", tc);
  expect(loaded_resolved_plan.StepCount(NN::NetworkExecutionOpKind::Attention) >
             0,
         "loaded resolved plan should expose attention operators", tc);
  expect(
      loaded_resolved_plan.StepCount(NN::NetworkExecutionOpKind::FeedForward) >
          0,
      "loaded resolved plan should expose feed-forward operators", tc);
  expect(loaded_resolved_plan.steps.front().kind ==
             NN::NetworkExecutionOpKind::InputPack,
         "resolved execution plan should start with input packing", tc);
#ifdef USE_CUDA
  const auto loaded_cuda_schedule =
      NN::Cuda::CreateCudaExecutionSchedule(loaded_resolved_plan);
  expect(loaded_cuda_schedule.positional_encoding_stage_count > 0,
         "loaded CUDA schedule should classify smolgen positional weights", tc);
  expect(loaded_cuda_schedule.FirstUnsupported() &&
             loaded_cuda_schedule.FirstUnsupported()->op_kind ==
                 NN::NetworkExecutionOpKind::Attention,
         "loaded CUDA schedule should now reach attention as first unsupported "
         "operator",
         tc);
  const auto first_body_attention = find_resolved_step_index(
      loaded_resolved_plan, "body.encoder.0.mha");
  const auto loaded_attention_plan = NN::Cuda::ResolveCudaAttentionStagePlan(
      loaded_resolved_plan, first_body_attention,
      loaded_weights.encoder_head_count);
  expect(loaded_attention_plan.heads == loaded_weights.encoder_head_count,
         "loaded CUDA attention plan should preserve body head count", tc);
  expect(loaded_attention_plan.squares == NN::Cuda::kCudaAttentionSquares,
         "loaded CUDA attention plan should use 64 board squares", tc);
  expect(loaded_attention_plan.qkv_width ==
             loaded_attention_plan.input_width,
         "loaded CUDA attention QKV width should match model width", tc);
  expect(loaded_attention_plan.head_depth > 0,
         "loaded CUDA attention plan should derive per-head depth", tc);
  expect(loaded_attention_plan.smolgen.present,
         "loaded CUDA attention plan should detect smolgen branch", tc);
  expect(loaded_attention_plan.smolgen.has_global_positional_weights,
         "loaded CUDA attention plan should attach global smolgen weights",
         tc);
#endif
}

void test_network_output_decoder(TestCounter &tc) {
  std::cout << "  Network output decoder..." << std::endl;

  NN::NetworkFormatDescriptor descriptor;
  descriptor.wdl = true;
  descriptor.moves_left = true;
  descriptor.attention_policy = true;
  const auto wdl_plan = NN::CreateNetworkTensorPlan(descriptor);
  std::vector<float> policy(wdl_plan.PolicyEntries(2), 0.0f);
  policy[0] = 0.25f;
  policy[NN::kPolicyOutputs - 1] = -0.75f;
  policy[NN::kPolicyOutputs] = 0.50f;
  policy[2 * NN::kPolicyOutputs - 1] = 1.25f;
  std::vector<float> value = {0.70f, 0.20f, 0.10f, 0.10f, 0.30f, 0.60f};
  std::vector<float> moves_left = {11.0f, 22.0f};

  const auto decoded = NN::DecodeNetworkOutputBatch(
      wdl_plan, policy.data(), policy.size(), value.data(), value.size(),
      moves_left.data(), moves_left.size(), 2);
  expect(decoded.size() == 2, "decoder should emit one output per batch item",
         tc);
  expect(decoded[0].has_wdl && decoded[1].has_wdl,
         "decoder should mark WDL outputs", tc);
  expect(std::abs(decoded[0].value - 0.60f) < 1e-6f,
         "WDL value should decode as win-loss", tc);
  expect(std::abs(decoded[1].value - -0.50f) < 1e-6f,
         "second WDL value should decode as win-loss", tc);
  expect(decoded[0].has_moves_left && decoded[1].has_moves_left,
         "decoder should mark moves-left outputs", tc);
  expect(decoded[0].moves_left == 11.0f && decoded[1].moves_left == 22.0f,
         "decoder should copy moves-left outputs", tc);
  expect(decoded[0].policy[0] == 0.25f &&
             decoded[0].policy[NN::kPolicyOutputs - 1] == -0.75f &&
             decoded[1].policy[0] == 0.50f &&
             decoded[1].policy[NN::kPolicyOutputs - 1] == 1.25f,
         "decoder should copy policy logits per batch item", tc);

  NN::NetworkFormatDescriptor scalar_descriptor;
  const auto scalar_plan = NN::CreateNetworkTensorPlan(scalar_descriptor);
  std::vector<float> scalar_policy(scalar_plan.PolicyEntries(1), 0.0f);
  scalar_policy[42] = 3.0f;
  std::vector<float> scalar_value = {-0.25f};
  const auto scalar_decoded = NN::DecodeNetworkOutputBatch(
      scalar_plan, scalar_policy.data(), scalar_policy.size(),
      scalar_value.data(), scalar_value.size(), nullptr, 0, 1);
  expect(scalar_decoded.size() == 1, "scalar decoder should emit one output",
         tc);
  expect(!scalar_decoded[0].has_wdl,
         "scalar decoder should not mark WDL outputs", tc);
  expect(scalar_decoded[0].value == -0.25f,
         "scalar decoder should copy scalar value", tc);
  expect(scalar_decoded[0].wdl[0] == 0.0f && scalar_decoded[0].wdl[1] == 0.0f &&
             scalar_decoded[0].wdl[2] == 0.0f,
         "scalar decoder should clear WDL tuple", tc);
  expect(!scalar_decoded[0].has_moves_left &&
             scalar_decoded[0].moves_left == 0.0f,
         "scalar decoder should clear absent moves-left output", tc);

  bool threw = false;
  try {
    NN::DecodeNetworkOutputBatch(wdl_plan, policy.data(), policy.size() - 1,
                                 value.data(), value.size(), moves_left.data(),
                                 moves_left.size(), 2);
  } catch (const std::exception &) {
    threw = true;
  }
  expect(threw, "decoder should reject mismatched tensor sizes", tc);
}

void test_nn_backend_selector_contract(TestCounter &tc) {
  std::cout << "  NN backend selector..." << std::endl;

#ifdef USE_CUDA
  std::cout << "    " << NN::Cuda::RuntimeCudaDeviceSummary() << std::endl;
  expect(NN::Cuda::CompiledCudaRuntimeVersion() > 0,
         "CUDA build should compile and link CUDA runtime probe", tc);
  expect(!NN::Cuda::RuntimeCudaDeviceSummary().empty(),
         "CUDA build should expose a runtime device summary", tc);
#endif

  NN::WeightsFile empty_weights;
  auto stub = NN::CreateNetwork(empty_weights, "stub");
  expect(static_cast<bool>(stub), "explicit stub backend should construct", tc);
  expect(stub->GetNetworkInfo().find("Stub network") != std::string::npos,
         "explicit stub backend should not select a platform backend", tc);
  auto path_stub = NN::CreateNetwork("", "stub");
  expect(static_cast<bool>(path_stub),
         "explicit stub backend should not require a weights path", tc);

  Position pos;
  StateInfo st;
  pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false,
          &st);
  NNMCTSEvaluator stub_evaluator("", "stub");
  const auto stub_eval = stub_evaluator.Evaluate(pos);
  expect(stub_eval.policy_priors.size() == 20,
         "stub evaluator should generate legal root priors without weights",
         tc);

  auto expect_throws = [&](const std::function<void()> &fn, const char *msg) {
    bool threw = false;
    try {
      fn();
    } catch (const std::exception &) {
      threw = true;
    }
    expect(threw, msg, tc);
  };

  expect_throws([&]() { NN::CreateNetwork(empty_weights, "not-a-backend"); },
                "unknown backend should fail loudly");

  bool explicit_metal_returned_stub = false;
  bool explicit_metal_threw = false;
  try {
    auto metal = NN::CreateNetwork(empty_weights, "metal");
    explicit_metal_returned_stub =
        metal->GetNetworkInfo().find("Stub network") != std::string::npos;
  } catch (const std::exception &) {
    explicit_metal_threw = true;
  }
  expect(explicit_metal_threw || !explicit_metal_returned_stub,
         "explicit metal backend must not fall back to stub", tc);

  bool explicit_cuda_returned_stub = false;
  bool explicit_cuda_threw = false;
  std::string explicit_cuda_error;
  try {
    auto cuda = NN::CreateNetwork(empty_weights, "cuda");
    explicit_cuda_returned_stub =
        cuda->GetNetworkInfo().find("Stub network") != std::string::npos;
  } catch (const std::exception &e) {
    explicit_cuda_threw = true;
    explicit_cuda_error = e.what();
  }
  expect(explicit_cuda_threw || !explicit_cuda_returned_stub,
         "explicit cuda backend must not fall back to stub", tc);
#ifdef USE_CUDA
  expect(explicit_cuda_error.find("CUDA runtime") != std::string::npos,
         "explicit cuda failure should include runtime diagnostics", tc);
#endif
}

#ifdef USE_CUDA
void test_cuda_input_packing(TestCounter &tc) {
  std::cout << "  CUDA input packing..." << std::endl;

  const auto fixture = BuildStartPositionPackedInputFixture();
  std::vector<std::uint64_t> fixture_masks;
  std::vector<float> fixture_values;
  NN::Cuda::PackInputPlanesHostRaw(fixture.planes[0].data(), 1, fixture_masks,
                                   fixture_values);
  expect(fixture_masks == fixture.masks,
         "CUDA host pack should match shared fixture masks", tc);
  expect(fixture_values == fixture.values,
         "CUDA host pack should match shared fixture values", tc);

  NN::InputPlanes input{};
  input[0][0] = 1.0f;
  input[0][7] = 1.0f;
  input[1][3] = 0.5f;
  input[NN::kPlanesPerBoard - 1][0] = 1.0f;
  input[NN::kAuxPlaneBase + 2][0] = 1.0f;
  input[NN::kAuxPlaneBase + 4][11] = 0.25f;

  std::vector<std::uint64_t> masks;
  std::vector<float> values;
  const float *raw_input = input[0].data();
  NN::Cuda::PackInputPlanesHostRaw(raw_input, 1, masks, values);

  expect(masks.size() == NN::kTotalPlanes,
         "host pack should emit one mask per plane", tc);
  expect(values.size() == NN::kTotalPlanes,
         "host pack should emit one value per plane", tc);
  expect(masks[0] == ((1ULL << 0) | (1ULL << 7)),
         "piece plane mask should include occupied squares", tc);
  expect(values[0] == 1.0f, "piece plane value should use first nonzero", tc);
  expect(masks[1] == (1ULL << 3), "non-uniform plane should pack sparse mask",
         tc);
  expect(values[1] == 0.5f, "non-uniform plane should preserve value", tc);
  expect(masks[NN::kPlanesPerBoard - 1] == ~0ULL,
         "uniform board plane should expand to a full mask", tc);
  expect(masks[NN::kAuxPlaneBase + 2] == ~0ULL,
         "uniform auxiliary plane should expand to a full mask", tc);
  expect(masks[NN::kAuxPlaneBase + 4] == (1ULL << 11),
         "non-uniform auxiliary plane should stay sparse", tc);

  auto smoke = NN::Cuda::RunInputPackingSmokeRaw(fixture.planes[0].data());
  std::cout << "    " << smoke.message << std::endl;
  expect(smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA input packing should pass or skip when no device is present",
         tc);
}

void test_cuda_inference_buffers(TestCounter &tc) {
  std::cout << "  CUDA inference buffers..." << std::endl;

  NN::Cuda::CudaBufferLayout scalar_layout;
  scalar_layout.max_batch_size = 2;
  expect(scalar_layout.InputPlaneEntries() ==
             2U * static_cast<size_t>(NN::kTotalPlanes),
         "scalar layout should allocate packed input planes", tc);
  expect(scalar_layout.PolicyEntries() == 2U * NN::kPolicyOutputs,
         "scalar layout should allocate policy outputs", tc);
  expect(scalar_layout.ValueEntries() == 2U,
         "scalar layout should allocate one value per batch item", tc);
  expect(scalar_layout.MovesLeftEntries() == 0U,
         "scalar layout should skip absent moves-left head", tc);
  expect(scalar_layout.RawPolicyEntries() == 0U,
         "scalar layout should skip absent raw policy scratch", tc);

  NN::NetworkFormatDescriptor descriptor;
  descriptor.wdl = true;
  descriptor.moves_left = true;
  descriptor.attention_policy = true;
  const auto tensor_plan = NN::CreateNetworkTensorPlan(descriptor);
  const auto bt4_layout = NN::Cuda::LayoutFromTensorPlan(tensor_plan, 4);
  const auto derived_layout = NN::Cuda::LayoutFromNetworkFormat(descriptor, 4);
  expect(derived_layout.TotalBytes() == bt4_layout.TotalBytes(),
         "format-derived CUDA layout should match manual BT4 layout", tc);
  expect(bt4_layout.ValueEntries() == 12U,
         "WDL layout should allocate three values per batch item", tc);
  expect(bt4_layout.MovesLeftEntries() == 4U,
         "moves-left layout should allocate one output per batch item", tc);
  expect(bt4_layout.RawPolicyEntries() ==
             4U * NN::Cuda::kCudaAttentionPolicyScratch,
         "attention policy layout should allocate raw policy scratch", tc);
  expect(bt4_layout.TotalBytes() > scalar_layout.TotalBytes(),
         "BT4 layout should require more device memory than scalar layout", tc);

  auto smoke = NN::Cuda::RunInferenceBufferSmoke();
  std::cout << "    " << smoke.message << std::endl;
  expect(smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA inference buffer smoke should pass or skip without a device",
         tc);

  const auto fixture = BuildStartPositionPackedInputFixture();
  auto upload_smoke =
      NN::Cuda::RunPackedInputUploadSmokeRaw(fixture.planes[0].data());
  std::cout << "    " << upload_smoke.message << std::endl;
  expect(upload_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             upload_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA packed input upload should pass or skip without a device", tc);

  auto workspace_smoke = NN::Cuda::RunExecutionWorkspaceSmoke();
  std::cout << "    " << workspace_smoke.message << std::endl;
  expect(workspace_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             workspace_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA execution workspace should pass or skip without a device", tc);

  auto tape_smoke = NN::Cuda::RunExecutionTapeSmoke();
  std::cout << "    " << tape_smoke.message << std::endl;
  expect(tape_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             tape_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA execution tape should pass or skip without a device", tc);

  std::array<NN::InputPlanes, 2> null_executor_inputs = {fixture.planes,
                                                         fixture.planes};
  auto null_executor_smoke = NN::Cuda::RunNullExecutorPipelineSmokeRaw(
      null_executor_inputs[0][0].data(), 2);
  std::cout << "    " << null_executor_smoke.message << std::endl;
  expect(null_executor_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             null_executor_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA null executor pipeline should pass or skip without a device",
         tc);

  auto plan_executor_smoke = NN::Cuda::RunPlanExecutorPipelineSmoke();
  std::cout << "    " << plan_executor_smoke.message << std::endl;
  expect(plan_executor_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             plan_executor_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA plan executor pipeline should pass or skip without a device",
         tc);
}

void test_cuda_execution_schedule(TestCounter &tc) {
  std::cout << "  CUDA execution schedule..." << std::endl;

  NN::NetworkResolvedExecutionPlan plan;
  plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::InputPack, "input.pack", {}});
  plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Dense, "smoke.dense", {}});
  plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::LayerNorm, "smoke.norm", {}});
  plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::OutputDecode, "output.decode", {}});

  const auto schedule = NN::Cuda::CreateCudaExecutionSchedule(plan);
  expect(schedule.FullySupported(),
         "input/dense-layernorm/output schedule should be supported", tc);
  expect(schedule.boundary_count == 2,
         "CUDA schedule should count boundary steps", tc);
  expect(schedule.dense_activation_stage_count == 0,
         "CUDA schedule should not count dense-only stages in fused plan", tc);
  expect(schedule.dense_layernorm_stage_count == 1,
         "CUDA schedule should count dense/layernorm stages", tc);
  expect(schedule.gate_stage_count == 0,
         "CUDA schedule should not count absent gate stages", tc);
  expect(schedule.feed_forward_stage_count == 0,
         "CUDA schedule should not count absent feed-forward stages", tc);
  expect(schedule.feed_forward_layernorm_stage_count == 0,
         "CUDA schedule should not count absent feed-forward/layernorm stages",
         tc);
  expect(schedule.positional_encoding_stage_count == 0,
         "CUDA schedule should not count absent positional encoding stages",
         tc);
  expect(schedule.unsupported_count == 0,
         "CUDA schedule should not report unsupported stages", tc);
  expect(schedule.entries.size() == 3,
         "CUDA schedule should fuse dense/layernorm into one entry", tc);

  NN::NetworkResolvedExecutionPlan unsupported = plan;
  unsupported.steps.insert(
      unsupported.steps.begin() + 1,
      NN::NetworkResolvedExecutionStep{NN::NetworkExecutionOpKind::Attention,
                                       "body.encoder.0.mha",
                                       {}});
  const auto unsupported_schedule =
      NN::Cuda::CreateCudaExecutionSchedule(unsupported);
  expect(!unsupported_schedule.FullySupported(),
         "attention schedule should be unsupported until kernels exist", tc);
  expect(unsupported_schedule.unsupported_count == 1,
         "CUDA schedule should count unsupported stages", tc);
  expect(unsupported_schedule.FirstUnsupported() &&
             unsupported_schedule.FirstUnsupported()->op_kind ==
                 NN::NetworkExecutionOpKind::Attention,
         "CUDA schedule should preserve first unsupported op kind", tc);
  expect(unsupported_schedule.Summary().find("body.encoder.0.mha") !=
             std::string::npos,
         "CUDA schedule summary should name first unsupported step", tc);

  NN::NetworkResolvedExecutionPlan dense_only;
  dense_only.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Dense, "orphan.dense", {}});
  const auto dense_only_schedule =
      NN::Cuda::CreateCudaExecutionSchedule(dense_only);
  expect(dense_only_schedule.FullySupported(),
         "dense without layernorm should run as dense/activation", tc);
  expect(dense_only_schedule.dense_activation_stage_count == 1,
         "CUDA schedule should count dense/activation stages", tc);
  expect(dense_only_schedule.entries.size() == 1 &&
             dense_only_schedule.entries[0].kind ==
                 NN::Cuda::CudaExecutionScheduleKind::DenseActivationStage,
         "CUDA schedule should classify dense-only stages explicitly", tc);

  NN::NetworkResolvedExecutionPlan gated = plan;
  gated.steps.insert(
      gated.steps.begin() + 3,
      NN::NetworkResolvedExecutionStep{NN::NetworkExecutionOpKind::Gate,
                                       "body.input_embedding_gates",
                                       {}});
  const auto gated_schedule = NN::Cuda::CreateCudaExecutionSchedule(gated);
  expect(gated_schedule.FullySupported(),
         "gate schedule should be supported", tc);
  expect(gated_schedule.gate_stage_count == 1,
         "CUDA schedule should count gate stages", tc);
  expect(gated_schedule.entries.size() == 4 &&
             gated_schedule.entries[2].kind ==
                 NN::Cuda::CudaExecutionScheduleKind::GateStage,
         "CUDA schedule should classify gates explicitly", tc);

  NN::NetworkResolvedExecutionPlan ffn = gated;
  ffn.steps.insert(
      ffn.steps.begin() + 3,
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::FeedForward,
          "body.input_embedding_ffn",
          {}});
  ffn.steps.insert(
      ffn.steps.begin() + 4,
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::LayerNorm,
          "body.input_embedding_ffn_norm",
          {}});
  const auto ffn_schedule = NN::Cuda::CreateCudaExecutionSchedule(ffn);
  expect(ffn_schedule.FullySupported(),
         "feed-forward/layernorm schedule should be supported", tc);
  expect(ffn_schedule.feed_forward_layernorm_stage_count == 1,
         "CUDA schedule should fuse feed-forward/layernorm stages", tc);
  expect(ffn_schedule.entries[2].kind ==
             NN::Cuda::CudaExecutionScheduleKind::FeedForwardLayerNormStage,
         "CUDA schedule should classify fused feed-forward stages explicitly",
         tc);

  NN::NetworkResolvedExecutionPlan ffn_only;
  ffn_only.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::FeedForward, "body.ffn_only", {}});
  const auto ffn_only_schedule =
      NN::Cuda::CreateCudaExecutionSchedule(ffn_only);
  expect(ffn_only_schedule.FullySupported(),
         "feed-forward without layernorm should be supported", tc);
  expect(ffn_only_schedule.feed_forward_stage_count == 1,
         "CUDA schedule should count standalone feed-forward stages", tc);

  NN::NetworkResolvedExecutionPlan positional = ffn;
  positional.steps.insert(
      positional.steps.begin() + 5,
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::PositionalEncoding,
          "body.smolgen_positional",
          {
              {0, "body.smolgen_w", 64 * 64, {64, 64},
               NN::NetworkWeightTensorKind::PositionalEncoding},
          }});
  const auto positional_schedule =
      NN::Cuda::CreateCudaExecutionSchedule(positional);
  expect(positional_schedule.FullySupported(),
         "positional encoding metadata schedule should be supported", tc);
  expect(positional_schedule.positional_encoding_stage_count == 1,
         "CUDA schedule should count positional encoding stages", tc);
  expect(positional_schedule.entries[3].kind ==
             NN::Cuda::CudaExecutionScheduleKind::PositionalEncodingStage,
         "CUDA schedule should classify positional encoding explicitly", tc);

  auto tensor = [](std::size_t index, const std::string &name,
                   std::size_t elements,
                   std::vector<std::uint32_t> dims,
                   NN::NetworkWeightTensorKind kind) {
    return NN::NetworkResolvedTensorRef{index, name, elements,
                                        std::move(dims), kind};
  };
  NN::NetworkResolvedExecutionPlan attention_plan;
  attention_plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::PositionalEncoding,
      "body.smolgen_positional",
      {tensor(0, "body.smolgen_w", 4096 * 4, {4096, 4},
              NN::NetworkWeightTensorKind::PositionalEncoding)}});
  attention_plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Attention,
      "body.encoder.0.mha",
      {
          tensor(1, "body.encoder.0.mha.q_w", 64, {8, 8},
                 NN::NetworkWeightTensorKind::DenseWeight),
          tensor(2, "body.encoder.0.mha.q_b", 8, {8},
                 NN::NetworkWeightTensorKind::DenseBias),
          tensor(3, "body.encoder.0.mha.k_w", 64, {8, 8},
                 NN::NetworkWeightTensorKind::DenseWeight),
          tensor(4, "body.encoder.0.mha.k_b", 8, {8},
                 NN::NetworkWeightTensorKind::DenseBias),
          tensor(5, "body.encoder.0.mha.v_w", 64, {8, 8},
                 NN::NetworkWeightTensorKind::DenseWeight),
          tensor(6, "body.encoder.0.mha.v_b", 8, {8},
                 NN::NetworkWeightTensorKind::DenseBias),
          tensor(7, "body.encoder.0.mha.dense_w", 64, {8, 8},
                 NN::NetworkWeightTensorKind::DenseWeight),
          tensor(8, "body.encoder.0.mha.dense_b", 8, {8},
                 NN::NetworkWeightTensorKind::DenseBias),
      }});
  attention_plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Dense,
      "body.encoder.0.mha.smolgen.dense",
      {
          tensor(9, "body.encoder.0.mha.smolgen.compress", 32, {4, 8},
                 NN::NetworkWeightTensorKind::DenseWeight),
          tensor(10, "body.encoder.0.mha.smolgen.dense1_w", 1536,
                 {6, 256}, NN::NetworkWeightTensorKind::DenseWeight),
          tensor(11, "body.encoder.0.mha.smolgen.dense1_b", 6, {6},
                 NN::NetworkWeightTensorKind::DenseBias),
          tensor(12, "body.encoder.0.mha.smolgen.dense2_w", 48, {8, 6},
                 NN::NetworkWeightTensorKind::DenseWeight),
          tensor(13, "body.encoder.0.mha.smolgen.dense2_b", 8, {8},
                 NN::NetworkWeightTensorKind::DenseBias),
      }});
  attention_plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::LayerNorm,
      "body.encoder.0.mha.smolgen.norm",
      {
          tensor(14, "body.encoder.0.mha.smolgen.ln1_gammas", 6, {6},
                 NN::NetworkWeightTensorKind::NormScale),
          tensor(15, "body.encoder.0.mha.smolgen.ln1_betas", 6, {6},
                 NN::NetworkWeightTensorKind::NormBias),
          tensor(16, "body.encoder.0.mha.smolgen.ln2_gammas", 8, {8},
                 NN::NetworkWeightTensorKind::NormScale),
          tensor(17, "body.encoder.0.mha.smolgen.ln2_betas", 8, {8},
                 NN::NetworkWeightTensorKind::NormBias),
      }});
  const auto resolved_attention =
      NN::Cuda::ResolveCudaAttentionStagePlan(attention_plan, 1, 2);
  expect(resolved_attention.heads == 2,
         "CUDA attention plan should preserve synthetic head count", tc);
  expect(resolved_attention.head_depth == 4,
         "CUDA attention plan should derive synthetic head depth", tc);
  expect(resolved_attention.smolgen.present,
         "CUDA attention plan should detect synthetic smolgen", tc);
  expect(resolved_attention.smolgen.dense2_width_per_head == 4,
         "CUDA attention plan should derive smolgen per-head width", tc);
  expect(resolved_attention.smolgen.has_global_positional_weights,
         "CUDA attention plan should validate global smolgen dimensions", tc);
  bool bad_heads_rejected = false;
  try {
    (void)NN::Cuda::ResolveCudaAttentionStagePlan(attention_plan, 1, 3);
  } catch (const std::exception &) {
    bad_heads_rejected = true;
  }
  expect(bad_heads_rejected,
         "CUDA attention plan should reject incompatible head counts", tc);
}

void test_cuda_output_mapping(TestCounter &tc) {
  std::cout << "  CUDA output mapping..." << std::endl;

  NN::NetworkTensorPlan tensor_plan;
  tensor_plan.value_outputs = 3;
  tensor_plan.moves_left_outputs = 1;
  tensor_plan.wdl = true;
  tensor_plan.moves_left = true;
  tensor_plan.raw_policy_outputs = NN::kNetworkAttentionPolicyScratch;

  NN::NetworkResolvedExecutionPlan plan;
  plan.tensors = tensor_plan;
  plan.policy_head = "smoke";
  plan.value_head = "smoke";
  plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Dense,
      "policy.smoke.output",
      {
          {0, "policy.smoke.ip_pol_w", 8, {2, 4},
           NN::NetworkWeightTensorKind::DenseWeight},
          {1, "policy.smoke.ip_pol_b", 2, {2},
           NN::NetworkWeightTensorKind::DenseBias},
      }});
  plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Dense,
      "value.smoke.dense2",
      {
          {2, "value.smoke.ip2_val_w", 6, {3, 2},
           NN::NetworkWeightTensorKind::DenseWeight},
          {3, "value.smoke.ip2_val_b", 3, {3},
           NN::NetworkWeightTensorKind::DenseBias},
      }});
  plan.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Dense,
      "moves_left.output",
      {
          {4, "moves_left.ip2_mov_w", 3, {1, 3},
           NN::NetworkWeightTensorKind::DenseWeight},
          {5, "moves_left.ip2_mov_b", 1, {1},
           NN::NetworkWeightTensorKind::DenseBias},
      }});

  const auto schedule = NN::Cuda::CreateCudaExecutionSchedule(plan);
  expect(NN::Cuda::ClassifyCudaPlanStage(plan, "policy.smoke.output") ==
             NN::Cuda::CudaPlanStageGroup::Policy,
         "CUDA plan analysis should classify selected policy head stages", tc);
  expect(NN::Cuda::ClassifyCudaPlanStage(plan, "value.smoke.dense2") ==
             NN::Cuda::CudaPlanStageGroup::Value,
         "CUDA plan analysis should classify selected value head stages", tc);
  expect(NN::Cuda::ClassifyCudaPlanStage(plan, "moves_left.output") ==
             NN::Cuda::CudaPlanStageGroup::MovesLeft,
         "CUDA plan analysis should classify moves-left stages", tc);
  expect(NN::Cuda::ClassifyCudaPlanStage(plan, "policy.other.output") ==
             NN::Cuda::CudaPlanStageGroup::Other,
         "CUDA plan analysis should reject non-selected policy heads", tc);
  expect(NN::Cuda::CudaDenseStageWidth(plan, schedule.entries[0]) == 2,
         "CUDA plan analysis should expose dense stage output width", tc);
  NN::Cuda::CudaOutputMappingOptions options;
  options.allow_partial_policy_rows = true;
  options.allow_partial_raw_policy_rows = true;
  const auto mapping =
      NN::Cuda::CreateCudaOutputMapping(tensor_plan, plan, schedule, options);
  expect(mapping.ok(), "CUDA output mapping should accept smoke outputs", tc);
  expect(mapping.bindings.size() == 4,
         "CUDA output mapping should bind policy/value/moves/raw outputs", tc);
  expect(mapping.Find(NN::Cuda::CudaOutputTarget::Policy) &&
             mapping.Find(NN::Cuda::CudaOutputTarget::Policy)->source_width ==
                 2,
         "CUDA output mapping should retain policy source width", tc);
  expect(mapping.Find(NN::Cuda::CudaOutputTarget::Value) &&
             mapping.Find(NN::Cuda::CudaOutputTarget::Value)->source_width ==
                 3,
         "CUDA output mapping should bind WDL value width", tc);
  expect(mapping.Find(NN::Cuda::CudaOutputTarget::MovesLeft),
         "CUDA output mapping should bind moves-left output", tc);
  expect(mapping.Find(NN::Cuda::CudaOutputTarget::RawPolicy),
         "CUDA output mapping should bind raw policy scratch source", tc);

  NN::NetworkResolvedExecutionPlan renamed = plan;
  renamed.steps[0].name = "policy.smoke.primary_logits";
  renamed.steps[1].name = "value.smoke.wdl_logits";
  renamed.steps[2].name = "moves_left.final_logits";
  const auto renamed_mapping = NN::Cuda::CreateCudaOutputMapping(
      tensor_plan, renamed, NN::Cuda::CreateCudaExecutionSchedule(renamed),
      options);
  expect(renamed_mapping.ok(),
         "CUDA output mapping should derive sources from head prefixes", tc);
  expect(renamed_mapping.Find(NN::Cuda::CudaOutputTarget::Policy) &&
             renamed_mapping.Find(NN::Cuda::CudaOutputTarget::Policy)
                     ->source_stage == "policy.smoke.primary_logits",
         "CUDA output mapping should bind renamed policy source", tc);
  expect(renamed_mapping.Find(NN::Cuda::CudaOutputTarget::Value) &&
             renamed_mapping.Find(NN::Cuda::CudaOutputTarget::Value)
                     ->source_stage == "value.smoke.wdl_logits",
         "CUDA output mapping should bind renamed value source", tc);
  expect(renamed_mapping.Find(NN::Cuda::CudaOutputTarget::MovesLeft) &&
             renamed_mapping.Find(NN::Cuda::CudaOutputTarget::MovesLeft)
                     ->source_stage == "moves_left.final_logits",
         "CUDA output mapping should bind renamed moves-left source", tc);

  const auto no_body_inputs =
      NN::Cuda::CreateCudaStageInputBindings(plan, schedule);
  expect(no_body_inputs.Size() == 0,
         "CUDA stage input derivation should not bind head-only plans", tc);

  NN::Cuda::CudaStageInputBindings stage_inputs;
  stage_inputs.Add("policy.smoke.output", "body.smoke.dense");
  stage_inputs.Add("value.smoke.dense2", "body.smoke.dense");
  expect(stage_inputs.Size() == 2,
         "CUDA stage input bindings should count explicit sources", tc);
  expect(stage_inputs.FindSource("policy.smoke.output") &&
             *stage_inputs.FindSource("policy.smoke.output") ==
                 "body.smoke.dense",
         "CUDA stage input bindings should return policy source", tc);
  bool duplicate_rejected = false;
  try {
    stage_inputs.Add("policy.smoke.output", "other.source");
  } catch (const std::exception &) {
    duplicate_rejected = true;
  }
  expect(duplicate_rejected,
         "CUDA stage input bindings should reject duplicate stages", tc);

  NN::NetworkResolvedExecutionPlan branched = plan;
  branched.steps.insert(
      branched.steps.begin(),
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::LayerNorm,
          "body.smoke.norm",
          {
              {7, "body.smoke.norm_gammas", 4, {4},
               NN::NetworkWeightTensorKind::NormScale},
              {8, "body.smoke.norm_betas", 4, {4},
               NN::NetworkWeightTensorKind::NormBias},
          }});
  branched.steps.insert(
      branched.steps.begin(),
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::Dense,
          "body.smoke.dense",
          {
              {6, "body.smoke.dense_w", 16, {4, 4},
               NN::NetworkWeightTensorKind::DenseWeight},
              {7, "body.smoke.dense_b", 4, {4},
               NN::NetworkWeightTensorKind::DenseBias},
          }});
  branched.steps.insert(
      branched.steps.begin() + 3,
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::Gate,
          "body.smoke.gates",
          {
              {8, "body.ip_mult_gate", 4, {4},
               NN::NetworkWeightTensorKind::Gate},
              {9, "body.ip_add_gate", 4, {4},
               NN::NetworkWeightTensorKind::Gate},
          }});
  branched.steps.insert(
      branched.steps.begin() + 4,
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::FeedForward,
          "body.input_embedding_ffn",
          {
              {10, "body.ip_emb_ffn.dense1_w", 24, {6, 4},
               NN::NetworkWeightTensorKind::DenseWeight},
              {11, "body.ip_emb_ffn.dense1_b", 6, {6},
               NN::NetworkWeightTensorKind::DenseBias},
              {12, "body.ip_emb_ffn.dense2_w", 24, {4, 6},
               NN::NetworkWeightTensorKind::DenseWeight},
              {13, "body.ip_emb_ffn.dense2_b", 4, {4},
               NN::NetworkWeightTensorKind::DenseBias},
          }});
  branched.steps.insert(
      branched.steps.begin() + 5,
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::LayerNorm,
          "body.input_embedding_ffn_norm",
          {
              {14, "body.ip_emb_ffn_ln_gammas", 4, {4},
               NN::NetworkWeightTensorKind::NormScale},
              {15, "body.ip_emb_ffn_ln_betas", 4, {4},
               NN::NetworkWeightTensorKind::NormBias},
          }});
  branched.steps.insert(
      branched.steps.begin() + 6,
      NN::NetworkResolvedExecutionStep{
          NN::NetworkExecutionOpKind::Dense,
          "policy.smoke.dense2",
          {
              {16, "policy.smoke.ip2_pol_w", 4, {2, 2},
               NN::NetworkWeightTensorKind::DenseWeight},
              {17, "policy.smoke.ip2_pol_b", 2, {2},
               NN::NetworkWeightTensorKind::DenseBias},
          }});
  const auto branched_schedule = NN::Cuda::CreateCudaExecutionSchedule(branched);
  const auto derived_inputs =
      NN::Cuda::CreateCudaStageInputBindings(branched, branched_schedule);
  expect(derived_inputs.Size() == 3,
         "CUDA stage input derivation should bind first policy/value/moves "
         "head stages",
         tc);
  expect(derived_inputs.FindSource("policy.smoke.output") &&
             *derived_inputs.FindSource("policy.smoke.output") ==
                 "body.input_embedding_ffn",
         "CUDA stage input derivation should branch policy from last body output",
         tc);
  expect(!derived_inputs.FindSource("policy.smoke.dense2"),
         "CUDA stage input derivation should not rebind later policy stages",
         tc);
  expect(derived_inputs.FindSource("value.smoke.dense2") &&
             *derived_inputs.FindSource("value.smoke.dense2") ==
                 "body.input_embedding_ffn",
         "CUDA stage input derivation should branch value from body", tc);
  expect(derived_inputs.FindSource("moves_left.output") &&
             *derived_inputs.FindSource("moves_left.output") ==
                 "body.input_embedding_ffn",
         "CUDA stage input derivation should branch moves-left from body", tc);

  NN::NetworkResolvedExecutionPlan value_with_error = plan;
  value_with_error.steps.push_back(NN::NetworkResolvedExecutionStep{
      NN::NetworkExecutionOpKind::Dense,
      "value.smoke.error",
      {
          {10, "value.smoke.ip_val_err_w", 3, {3, 3},
           NN::NetworkWeightTensorKind::DenseWeight},
          {11, "value.smoke.ip_val_err_b", 3, {3},
           NN::NetworkWeightTensorKind::DenseBias},
      }});
  const auto value_with_error_mapping = NN::Cuda::CreateCudaOutputMapping(
      tensor_plan, value_with_error,
      NN::Cuda::CreateCudaExecutionSchedule(value_with_error), options);
  expect(value_with_error_mapping.ok(),
         "CUDA output mapping should ignore value error heads", tc);
  expect(value_with_error_mapping.Find(NN::Cuda::CudaOutputTarget::Value) &&
             value_with_error_mapping.Find(NN::Cuda::CudaOutputTarget::Value)
                     ->source_stage == "value.smoke.dense2",
         "CUDA output mapping should keep primary value source before error",
         tc);

  NN::NetworkResolvedExecutionPlan missing_value = plan;
  missing_value.steps.erase(missing_value.steps.begin() + 1);
  const auto incomplete = NN::Cuda::CreateCudaOutputMapping(
      tensor_plan, missing_value,
      NN::Cuda::CreateCudaExecutionSchedule(missing_value), options);
  expect(!incomplete.ok(),
         "CUDA output mapping should reject missing value output", tc);
  expect(incomplete.Summary().find("value") != std::string::npos,
         "CUDA output mapping summary should name missing value source", tc);
}

void test_cuda_weight_upload(TestCounter &tc) {
  std::cout << "  CUDA weight upload..." << std::endl;

  const auto file = make_minimal_attention_weights_file();
  const auto descriptor = NN::DescribeNetworkFormat(file);
  const auto tensor_plan = NN::CreateNetworkTensorPlan(descriptor);
  NN::MultiHeadWeights weights(file.weights());
  const auto inventory = NN::CreateNetworkWeightInventory(
      weights, NN::SelectPolicyHeadName(weights),
      NN::SelectValueHeadName(weights), tensor_plan);

  auto smoke = NN::Cuda::RunWeightUploadSmoke(inventory);
  std::cout << "    " << smoke.message << std::endl;
  expect(smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA weight upload should pass or skip without a device", tc);
  if (smoke.status == NN::Cuda::CudaSmokeStatus::Success) {
    expect(smoke.allocation_bytes == inventory.TotalBytes(),
           "CUDA weight upload should allocate exact inventory bytes", tc);
    expect(smoke.tensor_count == inventory.tensors.size(),
           "CUDA weight upload should preserve tensor count", tc);
  }
}

void test_cuda_dense_kernels(TestCounter &tc) {
  std::cout << "  CUDA dense kernels..." << std::endl;

  auto smoke = NN::Cuda::RunDenseAffineKernelSmoke();
  std::cout << "    " << smoke.message << std::endl;
  expect(smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA dense affine kernel should pass or skip without a device", tc);

  auto layernorm_smoke = NN::Cuda::RunLayerNormKernelSmoke();
  std::cout << "    " << layernorm_smoke.message << std::endl;
  expect(layernorm_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             layernorm_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA layernorm kernel should pass or skip without a device", tc);

  auto activation_smoke = NN::Cuda::RunActivationKernelSmoke();
  std::cout << "    " << activation_smoke.message << std::endl;
  expect(activation_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             activation_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA activation kernel should pass or skip without a device", tc);

  auto gate_smoke = NN::Cuda::RunGateKernelSmoke();
  std::cout << "    " << gate_smoke.message << std::endl;
  expect(gate_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             gate_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA gate kernel should pass or skip without a device", tc);

  auto residual_smoke = NN::Cuda::RunResidualAddKernelSmoke();
  std::cout << "    " << residual_smoke.message << std::endl;
  expect(residual_smoke.status == NN::Cuda::CudaSmokeStatus::Success ||
             residual_smoke.status == NN::Cuda::CudaSmokeStatus::NoDevice,
         "CUDA residual add kernel should pass or skip without a device", tc);
}
#endif

void test_lc0_stoppers(TestCounter &tc) {
  std::cout << "  Lc0 stopper behavior..." << std::endl;

  KLDGainStopper kld(1.0f, 1);
  StoppersHints hints;
  hints.UpdateEstimatedRemainingTimeMs(1000);
  hints.UpdateEstimatedRemainingTimeMs(1500);
  hints.UpdateEstimatedRemainingPlayouts(100);
  hints.UpdateEstimatedRemainingPlayouts(200);
  expect(hints.GetEstimatedRemainingTimeMs() == 1000,
         "stopper hints keep the tightest time bound", tc);
  expect(hints.GetEstimatedRemainingPlayouts() == 100,
         "stopper hints keep the tightest playout bound", tc);

  SearchStats stats;
  stats.total_nodes = 2;
  stats.nodes_since_movestart = 2;
  stats.edge_n = {1, 0};
  expect(!kld.ShouldStop(stats, &hints), "KLD needs a prior window", tc);
  stats.total_nodes = 3;
  stats.nodes_since_movestart = 3;
  stats.edge_n = {2, 0};
  expect(kld.ShouldStop(stats, &hints), "KLD stops on low distribution gain",
         tc);

  SmartPruningStopper smart(1.33f, 0);
  SearchStats sp_stats;
  sp_stats.total_nodes = 10;
  sp_stats.nodes_since_movestart = 10;
  sp_stats.time_since_movestart_ms = 1;
  sp_stats.edge_n = {100, 1};
  StoppersHints sp_hints;
  sp_hints.UpdateEstimatedRemainingTimeMs(0);
  sp_hints.UpdateEstimatedRemainingPlayouts(0);
  expect(!smart.ShouldStop(sp_stats, &sp_hints),
         "smart pruning waits for tolerance window", tc);

  sp_stats.total_nodes = 400;
  sp_stats.nodes_since_movestart = 400;
  sp_stats.time_since_movestart_ms = 250;
  expect(smart.ShouldStop(sp_stats, &sp_hints),
         "smart pruning stops when second move cannot overtake", tc);
}

void test_solid_tree_repairs_child_parents(TestCounter &tc) {
  std::cout << "  Solid tree parent repair..." << std::endl;

  Node root;
  Move root_moves[] = {Move(SQ_E2, SQ_E4), Move(SQ_D2, SQ_D4)};
  root.CreateEdges(root_moves, 2);

  auto child = std::make_unique<Node>(&root, 0);
  root.Edges()[0].child.store(child.get(), std::memory_order_release);

  Move child_moves[] = {Move(SQ_E7, SQ_E5)};
  child->CreateEdges(child_moves, 1);
  auto grandchild = std::make_unique<Node>(child.get(), 0);
  child->Edges()[0].child.store(grandchild.get(), std::memory_order_release);

  expect(root.MakeSolid(), "root should solidify when no visits are in flight",
         tc);
  Node *solid_child = root.Edges()[0].child.load(std::memory_order_acquire);
  Node *solid_grandchild =
      solid_child->Edges()[0].child.load(std::memory_order_acquire);
  expect(solid_child->Parent() == &root,
         "solid child should point at solidified parent", tc);
  expect(solid_grandchild->Parent() == solid_child,
         "grandchild parent should be repaired after edge transfer", tc);
}

void test_nn_cache_policy_capacity(TestCounter &tc) {
  std::cout << "  NN cache policy capacity..." << std::endl;

  NNCache cache(8);
  EvaluationResult small;
  small.value = 0.25f;
  for (int i = 0; i < 8; ++i) {
    small.policy_priors.emplace_back(Move(static_cast<std::uint16_t>(100 + i)),
                                     0.01f * static_cast<float>(i + 1));
  }

  cache.Insert(1234, small);
  EvaluationResult small_out;
  bool small_hit = cache.Lookup(1234, 8, small_out);
  expect(small_hit, "small policy entry should be cached", tc);
  expect(small_out.policy_priors.size() == small.policy_priors.size(),
         "small policy entry should round-trip all moves", tc);

  EvaluationResult large;
  large.value = -0.10f;
  for (int i = 0; i < 128; ++i) {
    large.policy_priors.emplace_back(Move(static_cast<std::uint16_t>(500 + i)),
                                     0.001f * static_cast<float>(i + 1));
  }

  cache.Insert(5678, large);
  EvaluationResult large_out;
  bool large_hit = cache.Lookup(5678, 128, large_out);
  expect(!large_hit ||
             large_out.policy_priors.size() == large.policy_priors.size(),
         "cache must not return truncated policy entries", tc);
}

void test_history_buffer_ownership(TestCounter &tc) {
  std::cout << "  History buffer ownership..." << std::endl;

  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const std::vector<Move> line = {
      Move(SQ_E2, SQ_E4),
      Move(SQ_E7, SQ_E5),
      Move(SQ_G1, SQ_F3),
      Move(SQ_B8, SQ_C6),
  };

  SearchWorkerCtx ctx;
  ctx.SetRootFen(fen);
  for (Move move : line) {
    ctx.DoMove(move);
  }

  std::vector<std::unique_ptr<SearchWorkerCtx::HistoryBuffer>> keepalive;
  keepalive.push_back(ctx.BuildHistory());

  const auto &history = *keepalive.back();
  expect(history.depth == static_cast<int>(line.size()) + 1,
         "history should include root plus played moves", tc);
  expect(ctx.CurrentNNCacheKey() ==
             ComputeNNCacheKey(history.ptrs, history.depth),
         "incremental cache key should match rebuilt history", tc);

  Position replay;
  StateInfo root_state;
  replay.set(fen, false, &root_state);
  expect(history.ptrs[0]->raw_key() == replay.raw_key(),
         "root history state remains valid after owner move", tc);

  std::vector<StateInfo> states(line.size());
  for (size_t i = 0; i < line.size(); ++i) {
    replay.do_move(line[i], states[i]);
    expect(history.ptrs[i + 1]->raw_key() == replay.raw_key(),
           "played history state remains valid after owner move", tc);
  }
}

void test_history_buffer_tail_replay(TestCounter &tc) {
  std::cout << "  History buffer tail replay..." << std::endl;

  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const std::vector<Move> line = {
      Move(SQ_E2, SQ_E4), Move(SQ_E7, SQ_E5), Move(SQ_G1, SQ_F3),
      Move(SQ_B8, SQ_C6), Move(SQ_F1, SQ_C4), Move(SQ_G8, SQ_F6),
      Move(SQ_D2, SQ_D3), Move(SQ_F8, SQ_C5), Move(SQ_C2, SQ_C3),
      Move(SQ_D7, SQ_D6), Move(SQ_B1, SQ_D2), Move(SQ_A7, SQ_A6),
  };

  SearchWorkerCtx ctx;
  ctx.SetRootFen(fen);
  for (Move move : line) {
    ctx.DoMove(move);
  }

  const Key leaf_key = ctx.pos.raw_key();
  const uint64_t leaf_cache_key = ctx.CurrentNNCacheKey();

  SearchWorkerCtx::HistoryBuffer history;
  ctx.BuildHistory(history);

  expect(ctx.pos.raw_key() == leaf_key,
         "history build should restore leaf position", tc);
  expect(ctx.CurrentNNCacheKey() == leaf_cache_key,
         "history build should restore incremental cache state", tc);
  expect(history.depth == SearchWorkerCtx::HistoryBuffer::kMaxHistory,
         "long paths should keep exactly the NN history tail", tc);
  expect(ctx.CurrentNNCacheKey() ==
             ComputeNNCacheKey(history.ptrs, history.depth),
         "tail history should match incremental cache key", tc);

  const Position *current_only[] = {history.ptrs[history.depth - 1]};
  expect(ctx.CurrentNNCacheKey(0) == ComputeNNCacheKey(current_only, 1),
         "cache history length 0 should hash only current position", tc);
  expect(ctx.CurrentNNCacheKey(7) ==
             ComputeNNCacheKey(history.ptrs, history.depth),
         "cache history length 7 should hash full NN history tail", tc);

  std::deque<StateInfo> replay_states(line.size() + 1);
  Position replay;
  replay.set(fen, false, &replay_states[0]);
  const int start_ply = static_cast<int>(line.size()) -
                        (SearchWorkerCtx::HistoryBuffer::kMaxHistory - 1);

  int history_idx = 0;
  for (int ply = 0; ply <= static_cast<int>(line.size()); ++ply) {
    if (ply >= start_ply) {
      expect(history.ptrs[history_idx]->raw_key() == replay.raw_key(),
             "tail history position should match replayed line", tc);
      expect(history.ptrs[history_idx]->rule50_count() == replay.rule50_count(),
             "tail history rule50 should match replayed line", tc);
      ++history_idx;
    }

    if (ply < static_cast<int>(line.size())) {
      replay.do_move(line[ply], replay_states[ply + 1]);
    }
  }
}

void test_nn_cache_key_tracks_encoded_state(TestCounter &tc) {
  std::cout << "  NN cache key encoded state..." << std::endl;

  Position rule50_zero;
  Position rule50_twenty;
  StateInfo st_zero;
  StateInfo st_twenty;
  rule50_zero.set("8/8/8/8/8/8/4K3/7k w - - 0 1", false, &st_zero);
  rule50_twenty.set("8/8/8/8/8/8/4K3/7k w - - 20 11", false, &st_twenty);

  const Position *zero_history[] = {&rule50_zero};
  const Position *twenty_history[] = {&rule50_twenty};
  expect(ComputeNNCacheKey(zero_history, 1) !=
             ComputeNNCacheKey(twenty_history, 1),
         "cache key should include rule-50 state", tc);

  const std::string start_fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  Position root;
  Position current;
  StateInfo root_st;
  StateInfo current_root_st;
  StateInfo e4_st;
  root.set(start_fen, false, &root_st);
  current.set(start_fen, false, &current_root_st);
  current.do_move(Move(SQ_E2, SQ_E4), e4_st);

  const Position *short_history[] = {&current};
  const Position *full_history[] = {&root, &current};
  expect(ComputeNNCacheKey(short_history, 1) !=
             ComputeNNCacheKey(full_history, 2),
         "cache key should include encoded history depth and boards", tc);
}

void test_deterministic_repro(TestCounter &tc) {
  std::cout << "  Deterministic reproducibility..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;
  params.out_of_order_eval = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 128;
  const std::string fen =
      "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3";

  auto s1 = CreateSearch(params);
  s1->StartSearch(fen, limits, nullptr, nullptr);
  s1->Wait();
  Move b1 = s1->GetBestMove();

  auto s2 = CreateSearch(params);
  s2->StartSearch(fen, limits, nullptr, nullptr);
  s2->Wait();
  Move b2 = s2->GetBestMove();

  expect(b1 == b2, "bestmove should match across deterministic runs", tc);
}

void test_evaluator_legal_move_view_parity(TestCounter &tc) {
  std::cout << "  Evaluator legal-move view parity..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  Position pos;
  StateInfo st;
  pos.set("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
          false, &st);

  std::array<const Position *, 1> history = {&pos};
  MoveList<LEGAL> moves(pos);
  NNMCTSEvaluator evaluator(weights);

  auto generated = evaluator.EvaluateWithHistory(history);
  auto provided = evaluator.EvaluateWithHistoryAndMoves(
      history, NNMCTSEvaluator::LegalMovesView(moves.begin(), moves.size()));

  expect(generated.policy_priors.size() == provided.policy_priors.size(),
         "provided move list should preserve policy size", tc);
  expect(std::abs(generated.value - provided.value) < 1e-4f,
         "provided move list should preserve value", tc);

  size_t common =
      std::min(generated.policy_priors.size(), provided.policy_priors.size());
  for (size_t i = 0; i < common; ++i) {
    expect(generated.policy_priors[i].first == provided.policy_priors[i].first,
           "provided move list should preserve move order", tc);
    expect(std::abs(generated.policy_priors[i].second -
                    provided.policy_priors[i].second) < 1e-4f,
           "provided move list should preserve policy logits", tc);
  }
}

void test_nodes_limit_with_callback(TestCounter &tc) {
  std::cout << "  Nodes limit with callback..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 16;

  bool callback_called = false;
  int info_lines = 0;
  auto search = CreateSearch(params);
  search->StartSearch(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      [&](Move best, Move) {
        callback_called = true;
        expect(best != Move::none(), "callback best move should be legal", tc);
      },
      [&](const std::string &) { ++info_lines; });
  search->Wait();

  const uint64_t sync_nodes = search->Stats().total_nodes.load();
  const auto sync_best = search->GetBestMoveStats();
  expect(callback_called, "bestmove callback should fire", tc);
  expect(info_lines > 0, "final MCTS info callback should fire", tc);
  expect(sync_nodes >= limits.nodes,
         "MCTS should honor node limit before callback", tc);
  expect(sync_best.move != Move::none(), "best move stats should name a move",
         tc);
  expect(sync_best.visits > 0, "best move stats should include child visits",
         tc);
  expect(sync_best.visits <= sync_nodes,
         "best move child visits should not exceed total visits", tc);

  callback_called = false;
  auto async_search = CreateSearch(params);
  async_search->StartSearch(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      [&](Move best, Move) {
        callback_called = true;
        expect(best != Move::none(), "async callback best move should be legal",
               tc);
      },
      [](const std::string &) {});
  std::thread waiter([&]() { async_search->Wait(); });
  waiter.join();

  const uint64_t async_nodes = async_search->Stats().total_nodes.load();
  const auto async_best = async_search->GetBestMoveStats();
  expect(callback_called, "async bestmove callback should fire", tc);
  expect(async_nodes >= limits.nodes,
         "MCTS should honor node limit with asynchronous waiter", tc);
  expect(async_best.move != Move::none(),
         "async best move stats should name a move", tc);
  expect(async_best.visits > 0,
         "async best move stats should include child visits", tc);
  expect(async_best.visits <= async_nodes,
         "async best move child visits should not exceed total visits", tc);
}

void test_searchmoves_restrict_root(TestCounter &tc) {
  std::cout << "  Searchmoves restrict root..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 32;
  limits.searchmoves = {"h2h3"};

  auto search = CreateSearch(params);
  search->StartSearch(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", limits,
      nullptr, nullptr);
  search->Wait();

  Move expected(SQ_H2, SQ_H3);
  expect(search->GetBestMove() == expected,
         "MCTS best move should obey root searchmoves", tc);
  auto pv = search->GetPV();
  expect(!pv.empty() && pv.front() == expected,
         "MCTS PV should start with the allowed searchmove", tc);
}

void test_empty_searchmoves_filter_blocks_root(TestCounter &tc) {
  std::cout << "  Empty searchmoves filter blocks root..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 8;
  limits.searchmoves = {"a1a2"};

  auto search = CreateSearch(params);
  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();

  expect(search->GetBestMove() == Move::none(),
         "MCTS should not fall back to unrestricted moves for an empty filter",
         tc);
  expect(search->GetPV().empty(),
         "MCTS PV should stay empty when searchmoves resolves to no moves", tc);

  MetalFish::Search::LimitsType unrestricted;
  unrestricted.nodes = 8;
  search->StartSearch(fen, unrestricted, nullptr, nullptr);
  search->Wait();

  expect(search->GetBestMove() != Move::none(),
         "same-root unrestricted search should reset an empty root filter", tc);
}

void test_same_root_search_reuses_tree(TestCounter &tc) {
  std::cout << "  Same-root tree reuse..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 16;
  limits.searchmoves = {"h2h3"};

  auto search = CreateSearch(params);
  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();
  const auto first = search->GetBestMoveStats();

  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();
  const auto second = search->GetBestMoveStats();

  Move expected(SQ_H2, SQ_H3);
  expect(first.move == expected && second.move == expected,
         "same-root reuse should preserve the root searchmove", tc);
  expect(second.visits > first.visits,
         "same-root search should continue the existing MCTS tree", tc);
}

void test_new_game_resets_same_root_tree(TestCounter &tc) {
  std::cout << "  New game resets same-root tree..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 16;
  limits.searchmoves = {"h2h3"};

  auto search = CreateSearch(params);
  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();
  const auto first = search->GetBestMoveStats();

  search->NewGame();
  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();
  const auto second = search->GetBestMoveStats();

  Move expected(SQ_H2, SQ_H3);
  expect(first.move == expected && second.move == expected,
         "new game should preserve legal searchmove handling", tc);
  expect(second.visits <= first.visits + 1,
         "new game should not continue the previous MCTS root", tc);
}

uint64_t run_endgame_eval_count(const char *weights, const std::string &fen) {
  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 24;

  auto search = CreateSearch(params);
  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();
  return search->Stats().nn_evaluations.load();
}

void test_mating_material_adjudication(TestCounter &tc) {
  std::cout << "  Mating material adjudication..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  const uint64_t bare_bishop =
      run_endgame_eval_count(weights, "8/8/8/4k3/8/8/2K2B2/8 w - - 0 1");
  expect(bare_bishop == 1,
         "K+B vs K should be adjudicated as insufficient material", tc);

  const uint64_t two_knights =
      run_endgame_eval_count(weights, "8/8/8/3k4/8/5N2/2K2N2/8 w - - 0 1");
  expect(two_knights > 1, "K+NN vs K should keep searching as mating material",
         tc);

  const uint64_t same_color_bishops =
      run_endgame_eval_count(weights, "8/8/8/4k3/8/4b3/5B2/2K5 w - - 0 1");
  expect(same_color_bishops == 1,
         "same-color bishop-only endings should be insufficient material", tc);

  const uint64_t opposite_bishops =
      run_endgame_eval_count(weights, "8/8/8/4k3/8/2K2b2/5B2/8 w - - 0 1");
  expect(opposite_bishops > 1,
         "opposite-colored bishop-only endings should keep searching", tc);
}

void test_node_limited_search_uses_tight_eval_budget(TestCounter &tc) {
  std::cout << "  Node-limited eval budget..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 2;
  params.minibatch_size = 8;
  params.max_prefetch = 16;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;
  params.smart_pruning_factor = 0.0f;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 32;

  auto search = CreateSearch(params);
  search->StartSearch(
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
      limits, nullptr, nullptr);
  search->Wait();

  const auto &stats = search->Stats();
  const uint64_t nodes = stats.total_nodes.load();
  const uint64_t evals = stats.nn_evaluations.load();
  std::cout << "    Nodes: " << nodes << ", NN evals: " << evals << std::endl;
  expect(nodes >= limits.nodes, "MCTS should honor node limit", tc);
  expect(nodes <= limits.nodes + static_cast<uint64_t>(params.num_threads),
         "node-limited MCTS should avoid large batch overshoot", tc);
  expect(evals <= nodes,
         "node-limited MCTS should not spend NN evals on speculative prefetch",
         tc);
}

void test_cache_hit_rate(TestCounter &tc) {
  std::cout << "  Cache hit rate..." << std::endl;
  const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
  if (!weights) {
    std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
    return;
  }

  SearchParams params;
  params.num_threads = 1;
  params.nn_weights_path = weights;
  params.add_dirichlet_noise = false;
  params.out_of_order_eval = false;

  MetalFish::Search::LimitsType limits;
  limits.nodes = 256;
  const std::string fen =
      "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3";

  auto search = CreateSearch(params);
  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();

  MetalFish::Search::LimitsType reset_limits;
  reset_limits.nodes = 1;
  search->StartSearch(
      "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
      reset_limits, nullptr, nullptr);
  search->Wait();

  search->StartSearch(fen, limits, nullptr, nullptr);
  search->Wait();

  const auto &stats = search->Stats();
  uint64_t hits = stats.cache_hits.load();
  uint64_t misses = stats.cache_misses.load();
  uint64_t total_lookups = hits + misses;
  double hit_rate = total_lookups > 0 ? 100.0 * hits / total_lookups : 0.0;
  std::cout << "    Cache hits: " << hits << " / " << total_lookups
            << " lookups (" << hit_rate << "%)" << std::endl;
  expect(hits > 0, "cache should reuse exact NN inputs across searches", tc);
  expect(hit_rate > 5.0, "exact-input cache hit rate > 5%", tc);
}

} // namespace

bool test_mcts_all() {
  std::cout << "\n[MCTS]" << std::endl;
  TestCounter tc;

  test_node_basics(tc);
  test_tablebase_wdl_conversion(tc);
  test_search_params_defaults(tc);
  test_network_format_descriptor(tc);
  test_shared_nn_input_fixture(tc);
  test_network_weight_inventory(tc);
  test_network_execution_plan(tc);
  test_network_output_decoder(tc);
  test_nn_backend_selector_contract(tc);
#ifdef USE_CUDA
  test_cuda_input_packing(tc);
  test_cuda_inference_buffers(tc);
  test_cuda_execution_schedule(tc);
  test_cuda_output_mapping(tc);
  test_cuda_weight_upload(tc);
  test_cuda_dense_kernels(tc);
#endif
  test_lc0_stoppers(tc);
  test_solid_tree_repairs_child_parents(tc);
  test_nn_cache_policy_capacity(tc);
  test_history_buffer_ownership(tc);
  test_history_buffer_tail_replay(tc);
  test_nn_cache_key_tracks_encoded_state(tc);
  test_root_search_smoke(tc);
  test_pv_boost_respects_weight(tc);
  test_evaluator_legal_move_view_parity(tc);
  test_deterministic_repro(tc);
  test_nodes_limit_with_callback(tc);
  test_searchmoves_restrict_root(tc);
  test_empty_searchmoves_filter_blocks_root(tc);
  test_same_root_search_reuses_tree(tc);
  test_new_game_resets_same_root_tree(tc);
  test_mating_material_adjudication(tc);
  test_node_limited_search_uses_tight_eval_budget(tc);
  test_cache_hit_rate(tc);

  std::cout << "  Passed: " << tc.passed << ", Failed: " << tc.failed
            << std::endl;
  return tc.failed == 0;
}

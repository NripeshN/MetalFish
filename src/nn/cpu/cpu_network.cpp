/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cpu_network.h"

#include "../input_plane_packing.h"
#include "../network_attention_plan.h"
#include "../network_format.h"
#include "../network_output_decoder.h"
#include "../network_tensor_plan.h"
#include "../tables/attention_policy_map.h"
#include "../weights.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace MetalFish {
namespace NN {
namespace Cpu {
namespace {

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         std::equal(prefix.begin(), prefix.end(), value.begin());
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

const NetworkResolvedTensorRef *
FindTensorKind(const NetworkResolvedExecutionStep &step,
               NetworkWeightTensorKind kind) {
  for (const auto &tensor : step.tensors) {
    if (tensor.kind == kind)
      return &tensor;
  }
  return nullptr;
}

const NetworkResolvedTensorRef *
FindTensorSuffix(const NetworkResolvedExecutionStep &step,
                 std::string_view suffix) {
  for (const auto &tensor : step.tensors) {
    if (EndsWith(tensor.name, suffix))
      return &tensor;
  }
  return nullptr;
}

bool IsMultiplyGate(std::string_view name) {
  return name.find("mult_gate") != std::string_view::npos;
}

bool IsAddGate(std::string_view name) {
  return name.find("add_gate") != std::string_view::npos;
}

bool IsDynamicPositionPreprocessName(std::string_view name) {
  return name == "body.input_embedding_preprocess";
}

bool IsAttentionSmolgenDenseName(std::string_view name) {
  return EndsWith(name, ".mha.smolgen.dense");
}

bool IsAttentionSmolgenNormName(std::string_view name) {
  return EndsWith(name, ".mha.smolgen.norm");
}

bool IsAttentionPolicyMapStep(const NetworkResolvedExecutionPlan &plan,
                              const NetworkResolvedExecutionStep &step) {
  return plan.format.attention_policy &&
         step.kind == NetworkExecutionOpKind::PolicyMap &&
         !step.tensors.empty();
}

int AttentionHeadCount(const NetworkResolvedExecutionPlan &plan,
                       std::string_view name) {
  if (StartsWith(name, "body.encoder."))
    return plan.format.body_attention_heads;
  if (StartsWith(name, "policy."))
    return plan.format.policy_attention_heads;
  return 0;
}

const std::array<int, kNetworkPolicyOutputs> &AttentionPolicyGatherMap() {
  static const auto gather = [] {
    std::array<int, kNetworkPolicyOutputs> indices{};
    indices.fill(-1);
    for (int raw = 0; raw < kNetworkAttentionPolicyScratch; ++raw) {
      const short mapped = Tables::kAttnPolicyMap[raw];
      if (mapped >= 0)
        indices[static_cast<std::size_t>(mapped)] = raw;
    }
    return indices;
  }();
  return gather;
}

bool IsSimpleDenseStep(const NetworkResolvedExecutionStep &step) {
  if (!FindTensorKind(step, NetworkWeightTensorKind::DenseWeight) ||
      !FindTensorKind(step, NetworkWeightTensorKind::DenseBias)) {
    return false;
  }
  for (const auto &tensor : step.tensors) {
    if (tensor.kind != NetworkWeightTensorKind::DenseWeight &&
        tensor.kind != NetworkWeightTensorKind::DenseBias) {
      return false;
    }
  }
  return true;
}

bool IsSimpleLayerNormStep(const NetworkResolvedExecutionStep &step) {
  if (!FindTensorKind(step, NetworkWeightTensorKind::NormScale) ||
      !FindTensorKind(step, NetworkWeightTensorKind::NormBias)) {
    return false;
  }
  for (const auto &tensor : step.tensors) {
    if (tensor.kind != NetworkWeightTensorKind::NormScale &&
        tensor.kind != NetworkWeightTensorKind::NormBias) {
      return false;
    }
  }
  return true;
}

bool IsSimpleGateStep(const NetworkResolvedExecutionStep &step) {
  if (step.tensors.empty())
    return false;
  for (const auto &tensor : step.tensors) {
    if (tensor.kind != NetworkWeightTensorKind::Gate ||
        (!IsMultiplyGate(tensor.name) && !IsAddGate(tensor.name))) {
      return false;
    }
  }
  return true;
}

bool IsSimpleFeedForwardStep(const NetworkResolvedExecutionStep &step) {
  return FindTensorSuffix(step, ".dense1_w") &&
         FindTensorSuffix(step, ".dense1_b") &&
         FindTensorSuffix(step, ".dense2_w") &&
         FindTensorSuffix(step, ".dense2_b");
}

bool IsSimplePositionalEncodingStep(const NetworkResolvedExecutionStep &step) {
  if (step.tensors.empty())
    return false;
  for (const auto &tensor : step.tensors) {
    if (tensor.kind != NetworkWeightTensorKind::PositionalEncoding ||
        tensor.elements == 0 || tensor.dims.empty()) {
      return false;
    }
  }
  return true;
}

std::string
FirstUnsupportedExecutionStep(const NetworkResolvedExecutionPlan &plan) {
  for (std::size_t index = 0; index < plan.steps.size(); ++index) {
    const auto &step = plan.steps[index];
    if (step.kind == NetworkExecutionOpKind::Attention) {
      try {
        (void)ResolveAttentionStagePlan(plan, index,
                                        AttentionHeadCount(plan, step.name));
      } catch (const std::exception &e) {
        return "CPU transformer backend does not support attention stage " +
               step.name + ": " + e.what();
      }
    }
  }
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::Convolution) {
      return "CPU transformer backend does not support convolution yet: " +
             step.name;
    }
  }
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::PolicyMap) {
      if (IsAttentionPolicyMapStep(plan, step))
        continue;
      return "CPU transformer backend does not support policy mapping yet: " +
             step.name;
    }
  }
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::Attention) {
      continue;
    }
    if (step.kind == NetworkExecutionOpKind::PolicyMap &&
        IsAttentionPolicyMapStep(plan, step)) {
      continue;
    }
    if (step.kind == NetworkExecutionOpKind::Dense) {
      if (IsAttentionSmolgenDenseName(step.name))
        continue;
      if (!IsSimpleDenseStep(step)) {
        return "CPU transformer backend does not support compound dense "
               "stage yet: " +
               step.name;
      }
      continue;
    }
    if (step.kind == NetworkExecutionOpKind::LayerNorm) {
      if (IsAttentionSmolgenNormName(step.name))
        continue;
      if (!IsSimpleLayerNormStep(step)) {
        return "CPU transformer backend does not support compound layernorm "
               "stage yet: " +
               step.name;
      }
      continue;
    }
    if (step.kind == NetworkExecutionOpKind::Gate) {
      if (!IsSimpleGateStep(step)) {
        return "CPU transformer backend does not support compound gate "
               "stage yet: " +
               step.name;
      }
      continue;
    }
    if (step.kind == NetworkExecutionOpKind::FeedForward) {
      if (!IsSimpleFeedForwardStep(step)) {
        return "CPU transformer backend does not support compound "
               "feed-forward stage yet: " +
               step.name;
      }
      continue;
    }
    if (step.kind == NetworkExecutionOpKind::PositionalEncoding) {
      if (!IsSimplePositionalEncodingStep(step)) {
        return "CPU transformer backend does not support malformed positional "
               "metadata stage: " +
               step.name;
      }
      continue;
    }
    if (step.kind != NetworkExecutionOpKind::InputPack &&
        step.kind != NetworkExecutionOpKind::OutputDecode) {
      return "CPU transformer backend does not support " +
             NetworkExecutionOpKindName(step.kind) + " yet: " + step.name;
    }
  }
  return {};
}

enum class CpuActivationKind {
  Relu,
  Relu2,
  Tanh,
  Sigmoid,
  Swish,
  Mish,
  Selu,
};

struct DenseStageActivation {
  enum class Mode {
    Linear,
    Elementwise,
    Softmax,
  };

  Mode mode = Mode::Linear;
  CpuActivationKind kind = CpuActivationKind::Relu;
};

DenseStageActivation ActivationFromName(const std::string &activation) {
  if (activation.empty())
    return {};
  if (activation == "softmax")
    return {DenseStageActivation::Mode::Softmax, CpuActivationKind::Relu};
  if (activation == "relu_2")
    return {DenseStageActivation::Mode::Elementwise, CpuActivationKind::Relu2};
  if (activation == "tanh")
    return {DenseStageActivation::Mode::Elementwise, CpuActivationKind::Tanh};
  if (activation == "sigmoid") {
    return {DenseStageActivation::Mode::Elementwise,
            CpuActivationKind::Sigmoid};
  }
  if (activation == "swish")
    return {DenseStageActivation::Mode::Elementwise, CpuActivationKind::Swish};
  if (activation == "mish")
    return {DenseStageActivation::Mode::Elementwise, CpuActivationKind::Mish};
  if (activation == "selu")
    return {DenseStageActivation::Mode::Elementwise, CpuActivationKind::Selu};
  return {DenseStageActivation::Mode::Elementwise, CpuActivationKind::Relu};
}

DenseStageActivation
DenseStageActivationForName(const NetworkResolvedExecutionPlan &plan,
                            std::string_view name) {
  const std::string policy_prefix = "policy." + plan.policy_head + ".";
  if (StartsWith(name, policy_prefix)) {
    if (name == policy_prefix + "dense2" || name == policy_prefix + "dense3")
      return {};
    if (name == policy_prefix + "output") {
      if (!plan.format.attention_policy)
        return {};
      return ActivationFromName(plan.format.attention_body
                                    ? plan.format.activations.default_activation
                                    : std::string("selu"));
    }
  }

  const std::string value_prefix = "value." + plan.value_head + ".";
  if (StartsWith(name, value_prefix)) {
    if (name == value_prefix + "dense2")
      return ActivationFromName(plan.format.wdl ? "softmax" : "tanh");
    if (name == value_prefix + "output" || name == value_prefix + "dense1")
      return ActivationFromName(plan.format.activations.default_activation);
  }

  if (name == "moves_left.output")
    return ActivationFromName("relu");
  if (name == "moves_left.dense0" || name == "moves_left.dense1")
    return ActivationFromName(plan.format.activations.default_activation);

  if (name == "body.input_embedding_preprocess")
    return {};
  if (name == "body.input_embedding")
    return ActivationFromName(plan.format.activations.default_activation);

  return ActivationFromName(plan.format.activations.ffn_activation);
}

float ActivationValue(float value, CpuActivationKind kind) {
  switch (kind) {
  case CpuActivationKind::Relu:
    return std::max(value, 0.0f);
  case CpuActivationKind::Relu2: {
    const float relu = std::max(value, 0.0f);
    return relu * relu;
  }
  case CpuActivationKind::Tanh:
    return std::tanh(value);
  case CpuActivationKind::Sigmoid:
    return 1.0f / (1.0f + std::exp(-value));
  case CpuActivationKind::Swish:
    return value / (1.0f + std::exp(-value));
  case CpuActivationKind::Mish:
    return value * std::tanh(std::log1p(std::exp(value)));
  case CpuActivationKind::Selu:
    return 1.05070098f *
           (value > 0.0f ? value : 1.67326324f * (std::exp(value) - 1.0f));
  }
  return value;
}

void ApplyDenseActivation(std::vector<float> &values, int rows, int width,
                          const DenseStageActivation &activation) {
  if (activation.mode == DenseStageActivation::Mode::Linear)
    return;

  if (activation.mode == DenseStageActivation::Mode::Elementwise) {
    for (float &value : values)
      value = ActivationValue(value, activation.kind);
    return;
  }

  for (int row = 0; row < rows; ++row) {
    float *base = values.data() + static_cast<std::size_t>(row) * width;
    float max_value = -std::numeric_limits<float>::infinity();
    for (int col = 0; col < width; ++col)
      max_value = std::max(max_value, base[col]);
    float sum = 0.0f;
    for (int col = 0; col < width; ++col) {
      base[col] = std::exp(base[col] - max_value);
      sum += base[col];
    }
    if (sum == 0.0f)
      continue;
    const float inv_sum = 1.0f / sum;
    for (int col = 0; col < width; ++col)
      base[col] *= inv_sum;
  }
}

struct CpuBuffer {
  std::vector<float> values;
  int width = 0;
};

std::size_t
BodyEncoderLayerCount(const NetworkResolvedExecutionPlan &execution_plan) {
  std::size_t max_layer = 0;
  bool found = false;
  constexpr std::string_view prefix = "body.encoder.";
  for (const auto &step : execution_plan.steps) {
    if (!StartsWith(step.name, prefix))
      continue;
    const std::string_view suffix =
        std::string_view(step.name).substr(prefix.size());
    const std::size_t dot = suffix.find('.');
    if (dot == std::string_view::npos)
      continue;
    std::size_t layer = 0;
    bool has_digit = false;
    for (char ch : suffix.substr(0, dot)) {
      if (ch < '0' || ch > '9') {
        has_digit = false;
        break;
      }
      has_digit = true;
      layer = layer * 10 + static_cast<std::size_t>(ch - '0');
    }
    if (!has_digit)
      continue;
    max_layer = std::max(max_layer, layer + 1);
    found = true;
  }
  return found ? max_layer : 0;
}

float FeedForwardResidualScale(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name) {
  if (!StartsWith(stage_name, "body.input_embedding_ffn") &&
      !StartsWith(stage_name, "body.encoder.")) {
    return 1.0f;
  }
  const std::size_t layer_count = BodyEncoderLayerCount(execution_plan);
  if (layer_count == 0)
    return 1.0f;
  return std::pow(2.0f * static_cast<float>(layer_count), -0.25f);
}

float FeedForwardLayerNormEpsilon(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name) {
  if (StartsWith(stage_name, "body.input_embedding_ffn"))
    return 1e-3f;
  if (StartsWith(stage_name, "body.encoder.")) {
    return execution_plan.format.input_embedding == INPUT_EMBEDDING_PE_DENSE
               ? 1e-3f
               : 1e-6f;
  }
  return 1e-5f;
}

float DenseLayerNormEpsilon(const NetworkResolvedExecutionPlan &execution_plan,
                            std::string_view stage_name) {
  if (stage_name == "body.input_embedding_norm" &&
      execution_plan.format.input_embedding == INPUT_EMBEDDING_PE_DENSE) {
    return 1e-3f;
  }
  return 1e-5f;
}

float AttentionLayerNormEpsilon(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name) {
  if (StartsWith(stage_name, "body.encoder.")) {
    return execution_plan.format.input_embedding == INPUT_EMBEDDING_PE_DENSE
               ? 1e-3f
               : 1e-6f;
  }
  if (StartsWith(stage_name, "policy."))
    return 1e-6f;
  return 1e-5f;
}

void DenseAffine(const float *input, int rows, int input_width,
                 const std::vector<float> &weight,
                 const std::vector<float> *bias, int output_width,
                 std::vector<float> &output) {
  output.assign(static_cast<std::size_t>(rows) *
                    static_cast<std::size_t>(output_width),
                0.0f);
  for (int row = 0; row < rows; ++row) {
    const float *input_row =
        input + static_cast<std::size_t>(row) * input_width;
    float *output_row =
        output.data() + static_cast<std::size_t>(row) * output_width;
    for (int out = 0; out < output_width; ++out) {
      const float *weight_row =
          weight.data() + static_cast<std::size_t>(out) * input_width;
      float sum = bias ? (*bias)[static_cast<std::size_t>(out)] : 0.0f;
      for (int in = 0; in < input_width; ++in)
        sum += input_row[in] * weight_row[in];
      output_row[out] = sum;
    }
  }
}

void ApplySoftmaxRows(std::vector<float> &values, int rows, int width) {
  for (int row = 0; row < rows; ++row) {
    float *base = values.data() + static_cast<std::size_t>(row) * width;
    float max_value = -std::numeric_limits<float>::infinity();
    for (int col = 0; col < width; ++col)
      max_value = std::max(max_value, base[col]);
    float sum = 0.0f;
    for (int col = 0; col < width; ++col) {
      base[col] = std::exp(base[col] - max_value);
      sum += base[col];
    }
    if (sum == 0.0f)
      continue;
    const float inv_sum = 1.0f / sum;
    for (int col = 0; col < width; ++col)
      base[col] *= inv_sum;
  }
}

void ApplyLayerNorm(const float *input, const float *residual_parent,
                    float residual_scale, const std::vector<float> &scale,
                    const std::vector<float> &bias, std::vector<float> &output,
                    int rows, int width, float epsilon) {
  for (int row = 0; row < rows; ++row) {
    const float *input_row = input + static_cast<std::size_t>(row) * width;
    const float *parent_row =
        residual_parent
            ? residual_parent + static_cast<std::size_t>(row) * width
            : nullptr;
    float *output_row = output.data() + static_cast<std::size_t>(row) * width;

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int col = 0; col < width; ++col) {
      const float value =
          parent_row ? parent_row[col] + input_row[col] * residual_scale
                     : input_row[col];
      sum += value;
      square_sum += value * value;
    }

    const float mean = sum / static_cast<float>(width);
    float variance = square_sum / static_cast<float>(width) - mean * mean;
    if (variance < 0.0f)
      variance = 0.0f;
    const float inv_std = 1.0f / std::sqrt(variance + epsilon);

    for (int col = 0; col < width; ++col) {
      const float value =
          parent_row ? parent_row[col] + input_row[col] * residual_scale
                     : input_row[col];
      const std::size_t idx = static_cast<std::size_t>(col);
      output_row[col] = (value - mean) * inv_std * scale[idx] + bias[idx];
    }
  }
}

const CpuBuffer &
RequireBuffer(const std::unordered_map<std::string, CpuBuffer> &outputs,
              const std::string &name) {
  const auto it = outputs.find(name);
  if (it == outputs.end())
    throw std::runtime_error("CPU transformer backend missing stage output: " +
                             name);
  return it->second;
}

} // namespace

CpuNetwork::CpuNetwork(const WeightsFile &weights)
    : format_(DescribeNetworkFormat(weights)),
      tensor_plan_(CreateNetworkTensorPlan(format_)) {
  std::unique_ptr<MultiHeadWeights> decoded_weights;
  std::string policy_head;
  std::string value_head;
  try {
    decoded_weights = std::make_unique<MultiHeadWeights>(weights.weights());
    policy_head = SelectPolicyHeadName(*decoded_weights);
    value_head = SelectValueHeadName(*decoded_weights);
    const auto tensor_validation = ValidateNetworkTensorPlan(
        tensor_plan_, *decoded_weights, policy_head, value_head);
    if (!tensor_validation.ok()) {
      throw std::runtime_error(tensor_validation.Summary());
    }

    const auto inventory = CreateNetworkWeightInventory(
        *decoded_weights, policy_head, value_head, tensor_plan_);
    execution_plan_ = CreateNetworkExecutionPlan(
        format_, tensor_plan_, policy_head, value_head, inventory);
    const auto execution_validation =
        execution_plan_.ValidateAgainstInventory(inventory);
    if (!execution_validation.ok()) {
      throw std::runtime_error(execution_validation.Summary());
    }
    resolved_execution_plan_ =
        ResolveNetworkExecutionPlan(execution_plan_, inventory);
    const auto resolved_inventory = CreateResolvedNetworkWeightInventory(
        inventory, resolved_execution_plan_);
    weight_bytes_ = resolved_inventory.TotalBytes();
    tensors_.reserve(resolved_inventory.tensors.size());
    for (const auto &tensor : resolved_inventory.tensors) {
      CpuTensor copy;
      copy.name = tensor.name;
      copy.dims = tensor.dims;
      copy.kind = tensor.kind;
      if (tensor.elements > 0) {
        if (!tensor.data) {
          throw std::runtime_error("resolved tensor has null data: " +
                                   tensor.name);
        }
        copy.data.assign(tensor.data, tensor.data + tensor.elements);
      }
      tensors_.push_back(std::move(copy));
    }
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "CPU transformer backend weight validation failed: " +
        std::string(e.what()));
  }

  unsupported_execution_reason_ =
      FirstUnsupportedExecutionStep(resolved_execution_plan_);
}

NetworkOutput CpuNetwork::Evaluate(const InputPlanes &input) {
  std::vector<InputPlanes> batch;
  batch.push_back(input);
  return RunBatch(batch).front();
}

std::vector<NetworkOutput>
CpuNetwork::EvaluateBatch(const std::vector<InputPlanes> &inputs) {
  if (inputs.empty())
    return {};
  return RunBatch(inputs);
}

std::string CpuNetwork::GetNetworkInfo() const {
  std::ostringstream out;
  out << "CPU transformer backend (format: " << format_.Summary()
      << ", tensors: " << tensor_plan_.Summary()
      << ", execution: " << resolved_execution_plan_.Summary()
      << ", weight_bytes=" << weight_bytes_ << ", executor="
      << (unsupported_execution_reason_.empty()
              ? "dense-layernorm-gate-ffn-positional-dynamic-pe-attention"
              : "unsupported")
      << ")";
  return out.str();
}

std::string CpuNetwork::UnsupportedExecutionMessage() const {
  return unsupported_execution_reason_.empty()
             ? "CPU transformer backend execution is not implemented yet"
             : unsupported_execution_reason_;
}

const CpuNetwork::CpuTensor &CpuNetwork::TensorAt(std::size_t index) const {
  if (index >= tensors_.size())
    throw std::runtime_error(
        "CPU transformer backend tensor index out of range");
  return tensors_[index];
}

std::vector<NetworkOutput>
CpuNetwork::RunBatch(const std::vector<InputPlanes> &inputs) const {
  if (!unsupported_execution_reason_.empty())
    throw std::runtime_error(UnsupportedExecutionMessage());

  const int batch_size = static_cast<int>(inputs.size());
  std::vector<float> raw_inputs(static_cast<std::size_t>(batch_size) *
                                kPackedInputPlaneCount *
                                kPackedInputSquareCount);
  const std::size_t input_plane_floats =
      static_cast<std::size_t>(kPackedInputPlaneCount) *
      kPackedInputSquareCount;
  for (int b = 0; b < batch_size; ++b) {
    std::memcpy(raw_inputs.data() +
                    static_cast<std::size_t>(b) * input_plane_floats,
                inputs[static_cast<std::size_t>(b)][0].data(),
                input_plane_floats * sizeof(float));
  }

  std::vector<std::uint64_t> masks;
  std::vector<float> packed_values;
  PackInputPlanesRaw(raw_inputs.data(), batch_size, masks, packed_values);

  CpuBuffer base;
  base.values = packed_values;
  base.width = kPackedInputPlaneCount;

  std::unordered_map<std::string, CpuBuffer> outputs;
  std::string last_body;
  std::string last_policy;
  std::string last_value;
  std::string last_moves_left;
  std::string last_executed_step;
  std::string last_feed_forward_step;
  std::string last_attention_step;
  std::unordered_map<std::string, CpuBuffer> feed_forward_sources;
  std::unordered_map<std::string, CpuBuffer> attention_sources;
  std::unordered_map<std::string, CpuBuffer> attention_policy_sources;

  const auto source_for = [&](const std::string &name) -> const CpuBuffer & {
    const std::string policy_prefix =
        "policy." + resolved_execution_plan_.policy_head + ".";
    const std::string value_prefix =
        "value." + resolved_execution_plan_.value_head + ".";

    if (StartsWith(name, policy_prefix)) {
      if (resolved_execution_plan_.format.attention_policy &&
          name == policy_prefix + "dense3") {
        const auto branch_it = attention_policy_sources.find(policy_prefix);
        if (branch_it != attention_policy_sources.end())
          return branch_it->second;
      }
      if (!last_policy.empty())
        return RequireBuffer(outputs, last_policy);
      if (!last_body.empty())
        return RequireBuffer(outputs, last_body);
      return base;
    }
    if (StartsWith(name, value_prefix)) {
      if (!last_value.empty())
        return RequireBuffer(outputs, last_value);
      if (!last_body.empty())
        return RequireBuffer(outputs, last_body);
      return base;
    }
    if (StartsWith(name, "moves_left.")) {
      if (!last_moves_left.empty())
        return RequireBuffer(outputs, last_moves_left);
      if (!last_body.empty())
        return RequireBuffer(outputs, last_body);
      return base;
    }
    if (!last_body.empty())
      return RequireBuffer(outputs, last_body);
    return base;
  };

  const auto remember_output = [&](const NetworkResolvedExecutionStep &step,
                                   CpuBuffer output) {
    const std::string policy_prefix =
        "policy." + resolved_execution_plan_.policy_head + ".";
    const std::string value_prefix =
        "value." + resolved_execution_plan_.value_head + ".";

    outputs[step.name] = std::move(output);
    if (StartsWith(step.name, policy_prefix)) {
      last_policy = step.name;
    } else if (StartsWith(step.name, value_prefix)) {
      last_value = step.name;
    } else if (StartsWith(step.name, "moves_left.")) {
      last_moves_left = step.name;
    } else if (StartsWith(step.name, "body.")) {
      last_body = step.name;
    }
  };

  for (std::size_t step_index = 0;
       step_index < resolved_execution_plan_.steps.size(); ++step_index) {
    const auto &step = resolved_execution_plan_.steps[step_index];
    if (step.kind == NetworkExecutionOpKind::InputPack ||
        step.kind == NetworkExecutionOpKind::OutputDecode) {
      continue;
    }

    if ((step.kind == NetworkExecutionOpKind::Dense &&
         IsAttentionSmolgenDenseName(step.name)) ||
        (step.kind == NetworkExecutionOpKind::LayerNorm &&
         IsAttentionSmolgenNormName(step.name))) {
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Dense &&
        IsDynamicPositionPreprocessName(step.name)) {
      const auto *weight_ref =
          FindTensorKind(step, NetworkWeightTensorKind::DenseWeight);
      const auto *bias_ref =
          FindTensorKind(step, NetworkWeightTensorKind::DenseBias);
      if (!weight_ref || !bias_ref) {
        throw std::runtime_error("CPU dynamic PE stage has missing tensors: " +
                                 step.name);
      }
      const auto &weight = TensorAt(weight_ref->inventory_index);
      const auto &bias = TensorAt(bias_ref->inventory_index);
      constexpr int kPositionPlanes = 12;
      const int input_planes = resolved_execution_plan_.tensors.input_planes;
      const int squares = resolved_execution_plan_.tensors.input_squares;
      if (resolved_execution_plan_.format.input_embedding !=
              INPUT_EMBEDDING_PE_DENSE ||
          input_planes <= 0 || squares <= 0 || squares > 64 ||
          input_planes < kPositionPlanes ||
          masks.size() != static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(input_planes) ||
          packed_values.size() != masks.size()) {
        throw std::runtime_error("CPU dynamic PE tensor plan is invalid: " +
                                 step.name);
      }
      if (weight.dims.size() != 2 || bias.dims.size() != 1) {
        throw std::runtime_error("CPU dynamic PE tensor shape is invalid: " +
                                 step.name);
      }
      const int out_width = static_cast<int>(weight.dims[0]);
      const int in_width = static_cast<int>(weight.dims[1]);
      if (in_width != squares * kPositionPlanes || out_width <= 0 ||
          out_width % squares != 0 ||
          bias.data.size() != static_cast<std::size_t>(out_width) ||
          weight.data.size() != static_cast<std::size_t>(out_width) *
                                    static_cast<std::size_t>(in_width)) {
        throw std::runtime_error("CPU dynamic PE tensor dimensions mismatch: " +
                                 step.name);
      }

      std::vector<float> position_input(static_cast<std::size_t>(batch_size) *
                                            static_cast<std::size_t>(in_width),
                                        0.0f);
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int square = 0; square < squares; ++square) {
          const std::uint64_t bit = 1ULL << square;
          for (int plane = 0; plane < kPositionPlanes; ++plane) {
            const std::size_t packed_index =
                static_cast<std::size_t>(batch) * input_planes + plane;
            position_input[static_cast<std::size_t>(batch) * in_width +
                           static_cast<std::size_t>(square) * kPositionPlanes +
                           plane] = (masks[packed_index] & bit)
                                        ? packed_values[packed_index]
                                        : 0.0f;
          }
        }
      }

      std::vector<float> position_output(
          static_cast<std::size_t>(batch_size) *
              static_cast<std::size_t>(out_width),
          0.0f);
      for (int batch = 0; batch < batch_size; ++batch) {
        const float *input_row =
            position_input.data() + static_cast<std::size_t>(batch) * in_width;
        float *output_row = position_output.data() +
                            static_cast<std::size_t>(batch) * out_width;
        for (int out = 0; out < out_width; ++out) {
          const float *weight_row =
              weight.data.data() + static_cast<std::size_t>(out) * in_width;
          float sum = bias.data[static_cast<std::size_t>(out)];
          for (int in = 0; in < in_width; ++in)
            sum += input_row[in] * weight_row[in];
          output_row[out] = sum;
        }
      }

      const int pe_width = out_width / squares;
      CpuBuffer output;
      output.width = input_planes + pe_width;
      output.values.assign(static_cast<std::size_t>(batch_size) *
                               static_cast<std::size_t>(squares) *
                               static_cast<std::size_t>(output.width),
                           0.0f);
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int square = 0; square < squares; ++square) {
          const std::uint64_t bit = 1ULL << square;
          float *output_row =
              output.values.data() +
              (static_cast<std::size_t>(batch) * squares + square) *
                  output.width;
          for (int plane = 0; plane < input_planes; ++plane) {
            const std::size_t packed_index =
                static_cast<std::size_t>(batch) * input_planes + plane;
            output_row[plane] = (masks[packed_index] & bit)
                                    ? packed_values[packed_index]
                                    : 0.0f;
          }
          const float *pe_row = position_output.data() +
                                static_cast<std::size_t>(batch) * out_width +
                                static_cast<std::size_t>(square) * pe_width;
          for (int col = 0; col < pe_width; ++col)
            output_row[input_planes + col] = pe_row[col];
        }
      }

      remember_output(step, std::move(output));
      last_executed_step = step.name;
      continue;
    }

    const CpuBuffer &source = source_for(step.name);
    if (source.width <= 0 || source.values.size() % source.width != 0) {
      throw std::runtime_error(
          "CPU transformer backend source shape mismatch: " + step.name);
    }
    const int rows = static_cast<int>(source.values.size() / source.width);

    if (step.kind == NetworkExecutionOpKind::PositionalEncoding) {
      for (const auto &tensor_ref : step.tensors) {
        const auto &tensor = TensorAt(tensor_ref.inventory_index);
        if (tensor.kind != NetworkWeightTensorKind::PositionalEncoding ||
            tensor.data.empty() || tensor.dims.empty()) {
          throw std::runtime_error(
              "CPU positional metadata tensor is invalid: " + step.name);
        }
      }
      last_executed_step = step.name;
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Dense) {
      const auto *weight_ref =
          FindTensorKind(step, NetworkWeightTensorKind::DenseWeight);
      const auto *bias_ref =
          FindTensorKind(step, NetworkWeightTensorKind::DenseBias);
      if (!weight_ref || !bias_ref)
        throw std::runtime_error("CPU dense stage has missing tensors: " +
                                 step.name);
      const auto &weight = TensorAt(weight_ref->inventory_index);
      const auto &bias = TensorAt(bias_ref->inventory_index);
      if (weight.dims.size() != 2 || bias.dims.size() != 1) {
        throw std::runtime_error(
            "CPU dense stage has unresolved tensor shape: " + step.name);
      }
      const int out_width = static_cast<int>(weight.dims[0]);
      const int in_width = static_cast<int>(weight.dims[1]);
      const CpuBuffer *dense_source = &source;
      CpuBuffer flattened_source;
      int dense_rows = rows;
      if (in_width != source.width) {
        if (batch_size <= 0 || rows % batch_size != 0 ||
            in_width != (rows / batch_size) * source.width) {
          throw std::runtime_error("CPU dense stage shape mismatch: " +
                                   step.name);
        }
        flattened_source.width = in_width;
        flattened_source.values = source.values;
        dense_source = &flattened_source;
        dense_rows = batch_size;
      }
      if (bias.data.size() != static_cast<std::size_t>(out_width) ||
          weight.data.size() != static_cast<std::size_t>(out_width) *
                                    static_cast<std::size_t>(in_width)) {
        throw std::runtime_error("CPU dense stage shape mismatch: " +
                                 step.name);
      }
      if (resolved_execution_plan_.format.attention_policy) {
        const std::string policy_prefix =
            "policy." + resolved_execution_plan_.policy_head + ".";
        if (step.name == policy_prefix + "dense2")
          attention_policy_sources[policy_prefix] = *dense_source;
      }

      CpuBuffer output;
      output.width = out_width;
      output.values.assign(static_cast<std::size_t>(dense_rows) *
                               static_cast<std::size_t>(out_width),
                           0.0f);
      for (int row = 0; row < dense_rows; ++row) {
        const float *input_row = dense_source->values.data() +
                                 static_cast<std::size_t>(row) * in_width;
        float *output_row =
            output.values.data() + static_cast<std::size_t>(row) * out_width;
        for (int out = 0; out < out_width; ++out) {
          const float *weight_row =
              weight.data.data() + static_cast<std::size_t>(out) * in_width;
          float sum = bias.data[static_cast<std::size_t>(out)];
          for (int in = 0; in < in_width; ++in)
            sum += input_row[in] * weight_row[in];
          output_row[out] = sum;
        }
      }
      ApplyDenseActivation(
          output.values, dense_rows, output.width,
          DenseStageActivationForName(resolved_execution_plan_, step.name));
      remember_output(step, std::move(output));
      last_executed_step = step.name;
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::PolicyMap) {
      if (!IsAttentionPolicyMapStep(resolved_execution_plan_, step)) {
        throw std::runtime_error("CPU policy map stage is unsupported: " +
                                 step.name);
      }
      constexpr std::string_view kSuffix = ".policy_map";
      if (!EndsWith(step.name, kSuffix)) {
        throw std::runtime_error("CPU policy map stage name is invalid: " +
                                 step.name);
      }
      const std::string policy_prefix =
          step.name.substr(0, step.name.size() - kSuffix.size());
      const CpuBuffer &query =
          RequireBuffer(outputs, policy_prefix + ".dense2");
      const CpuBuffer &key = RequireBuffer(outputs, policy_prefix + ".dense3");
      if (query.width <= 0 || query.width != key.width ||
          query.values.size() != key.values.size() ||
          query.values.size() != static_cast<std::size_t>(batch_size) *
                                     kPackedInputSquareCount *
                                     static_cast<std::size_t>(query.width)) {
        std::ostringstream out;
        out << "CPU attention policy Q/K shape mismatch: " << step.name
            << " query_width=" << query.width << " key_width=" << key.width
            << " query_values=" << query.values.size()
            << " key_values=" << key.values.size() << " expected="
            << static_cast<std::size_t>(batch_size) * kPackedInputSquareCount *
                   static_cast<std::size_t>(std::max(0, query.width));
        throw std::runtime_error(out.str());
      }
      const auto &promotion = TensorAt(step.tensors[0].inventory_index);
      if (promotion.data.size() != static_cast<std::size_t>(4 * query.width)) {
        throw std::runtime_error(
            "CPU attention policy promotion tensor dimensions mismatch: " +
            step.name);
      }

      std::vector<float> raw_policy(static_cast<std::size_t>(batch_size) *
                                        kNetworkAttentionPolicyScratch,
                                    0.0f);
      const float scale = 1.0f / std::sqrt(static_cast<float>(query.width));
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int query_square = 0; query_square < kPackedInputSquareCount;
             ++query_square) {
          const float *query_row =
              query.values.data() +
              (static_cast<std::size_t>(batch) * kPackedInputSquareCount +
               query_square) *
                  query.width;
          for (int key_square = 0; key_square < kPackedInputSquareCount;
               ++key_square) {
            const float *key_row =
                key.values.data() +
                (static_cast<std::size_t>(batch) * kPackedInputSquareCount +
                 key_square) *
                    key.width;
            float sum = 0.0f;
            for (int channel = 0; channel < query.width; ++channel)
              sum += query_row[channel] * key_row[channel];
            raw_policy[static_cast<std::size_t>(batch) *
                           kNetworkAttentionPolicyScratch +
                       static_cast<std::size_t>(query_square) *
                           kPackedInputSquareCount +
                       key_square] = sum * scale;
          }
        }
      }

      constexpr int kPromotionCount =
          kNetworkAttentionPolicyScratch -
          kPackedInputSquareCount * kPackedInputSquareCount;
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int promo = 0; promo < kPromotionCount; ++promo) {
          // Mirrors the Metal graph reshape: QK uses the 3x64 view while the
          // learned promotion offset keeps the 8x8x3 flattened order.
          const int promotion_row = promo % 3;
          const int promotion_key_square = 56 + (promo % 24) / 3;
          const int square_pair = promo % kPackedInputSquareCount;
          const int query_square = 48 + square_pair / 8;
          const int key_square = 56 + square_pair % 8;
          const float *query_row =
              query.values.data() +
              (static_cast<std::size_t>(batch) * kPackedInputSquareCount +
               query_square) *
                  query.width;
          const float *key_row =
              key.values.data() +
              (static_cast<std::size_t>(batch) * kPackedInputSquareCount +
               key_square) *
                  key.width;
          const float *promotion_key_row =
              key.values.data() +
              (static_cast<std::size_t>(batch) * kPackedInputSquareCount +
               promotion_key_square) *
                  key.width;
          float value = 0.0f;
          for (int channel = 0; channel < query.width; ++channel) {
            value += query_row[channel] * key_row[channel] * scale;
            value += promotion_key_row[channel] *
                     (promotion.data[static_cast<std::size_t>(promotion_row) *
                                         query.width +
                                     channel] +
                      promotion.data[static_cast<std::size_t>(3) * query.width +
                                     channel]);
          }
          raw_policy[static_cast<std::size_t>(batch) *
                         kNetworkAttentionPolicyScratch +
                     kPackedInputSquareCount * kPackedInputSquareCount +
                     promo] = value;
        }
      }

      CpuBuffer output;
      output.width = kNetworkPolicyOutputs;
      output.values.assign(
          static_cast<std::size_t>(batch_size) * kNetworkPolicyOutputs, 0.0f);
      const auto &gather = AttentionPolicyGatherMap();
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int policy = 0; policy < kNetworkPolicyOutputs; ++policy) {
          const int raw = gather[static_cast<std::size_t>(policy)];
          if (raw >= 0) {
            output.values[static_cast<std::size_t>(batch) *
                              kNetworkPolicyOutputs +
                          policy] =
                raw_policy[static_cast<std::size_t>(batch) *
                               kNetworkAttentionPolicyScratch +
                           raw];
          }
        }
      }

      remember_output(step, std::move(output));
      last_executed_step = step.name;
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Attention) {
      const auto attention = ResolveAttentionStagePlan(
          resolved_execution_plan_, step_index,
          AttentionHeadCount(resolved_execution_plan_, step.name));
      if (source.width != attention.input_width ||
          rows != batch_size * attention.squares) {
        throw std::runtime_error("CPU attention source shape mismatch: " +
                                 step.name);
      }

      const auto tensor_by_suffix =
          [&](std::string_view suffix) -> const CpuNetwork::CpuTensor & {
        const auto *ref = FindTensorSuffix(step, suffix);
        if (!ref)
          throw std::runtime_error("CPU attention stage has missing tensor: " +
                                   step.name);
        return TensorAt(ref->inventory_index);
      };
      const auto &q_w = tensor_by_suffix(".q_w");
      const auto &q_b = tensor_by_suffix(".q_b");
      const auto &k_w = tensor_by_suffix(".k_w");
      const auto &k_b = tensor_by_suffix(".k_b");
      const auto &v_w = tensor_by_suffix(".v_w");
      const auto &v_b = tensor_by_suffix(".v_b");
      const auto &dense_w = tensor_by_suffix(".dense_w");
      const auto &dense_b = tensor_by_suffix(".dense_b");

      std::vector<float> query;
      std::vector<float> key;
      std::vector<float> value;
      DenseAffine(source.values.data(), rows, attention.input_width, q_w.data,
                  &q_b.data, attention.qkv_width, query);
      DenseAffine(source.values.data(), rows, attention.input_width, k_w.data,
                  &k_b.data, attention.qkv_width, key);
      DenseAffine(source.values.data(), rows, attention.input_width, v_w.data,
                  &v_b.data, attention.qkv_width, value);

      std::vector<float> attention_bias;
      if (attention.smolgen.present) {
        if (!attention.smolgen.has_global_positional_weights) {
          throw std::runtime_error(
              "CPU attention smolgen requires global positional weights: " +
              step.name);
        }
        const auto smolgen_dense_name = step.name + ".smolgen.dense";
        const auto smolgen_norm_name = step.name + ".smolgen.norm";
        const auto *smolgen_dense_step = [&]() {
          for (const auto &candidate : resolved_execution_plan_.steps) {
            if (candidate.name == smolgen_dense_name)
              return &candidate;
          }
          return static_cast<const NetworkResolvedExecutionStep *>(nullptr);
        }();
        const auto *smolgen_norm_step = [&]() {
          for (const auto &candidate : resolved_execution_plan_.steps) {
            if (candidate.name == smolgen_norm_name)
              return &candidate;
          }
          return static_cast<const NetworkResolvedExecutionStep *>(nullptr);
        }();
        const auto *global_step =
            FindGlobalPositionalEncodingStep(resolved_execution_plan_);
        if (!smolgen_dense_step || !smolgen_norm_step || !global_step) {
          throw std::runtime_error("CPU attention smolgen steps missing: " +
                                   step.name);
        }

        const auto smolgen_tensor =
            [&](const NetworkResolvedExecutionStep &owner,
                std::string_view suffix) -> const CpuNetwork::CpuTensor & {
          const auto *ref = FindTensorSuffix(owner, suffix);
          if (!ref) {
            throw std::runtime_error(
                "CPU attention smolgen stage has missing tensor: " +
                owner.name);
          }
          return TensorAt(ref->inventory_index);
        };
        const auto &compress = smolgen_tensor(*smolgen_dense_step, ".compress");
        const auto &dense1_w = smolgen_tensor(*smolgen_dense_step, ".dense1_w");
        const auto &dense1_b = smolgen_tensor(*smolgen_dense_step, ".dense1_b");
        const auto &dense2_w = smolgen_tensor(*smolgen_dense_step, ".dense2_w");
        const auto &dense2_b = smolgen_tensor(*smolgen_dense_step, ".dense2_b");
        const auto &ln1_gamma =
            smolgen_tensor(*smolgen_norm_step, ".ln1_gammas");
        const auto &ln1_beta = smolgen_tensor(*smolgen_norm_step, ".ln1_betas");
        const auto &ln2_gamma =
            smolgen_tensor(*smolgen_norm_step, ".ln2_gammas");
        const auto &ln2_beta = smolgen_tensor(*smolgen_norm_step, ".ln2_betas");
        const auto &global = smolgen_tensor(*global_step, "body.smolgen_w");

        std::vector<float> compressed;
        DenseAffine(source.values.data(), rows, attention.input_width,
                    compress.data, nullptr,
                    attention.smolgen.compressed_channels, compressed);

        const int flattened_width =
            attention.squares * attention.smolgen.compressed_channels;
        std::vector<float> dense1;
        DenseAffine(compressed.data(), batch_size, flattened_width,
                    dense1_w.data, nullptr, attention.smolgen.dense1_width,
                    dense1);
        for (int row = 0; row < batch_size; ++row) {
          for (int col = 0; col < attention.smolgen.dense1_width; ++col) {
            const std::size_t idx =
                static_cast<std::size_t>(row) * attention.smolgen.dense1_width +
                col;
            dense1[idx] += dense1_b.data[static_cast<std::size_t>(col)];
          }
        }
        ApplyDenseActivation(
            dense1, batch_size, attention.smolgen.dense1_width,
            ActivationFromName(resolved_execution_plan_.format.activations
                                   .smolgen_activation));

        std::vector<float> norm1(dense1.size(), 0.0f);
        ApplyLayerNorm(dense1.data(), nullptr, 1.0f, ln1_gamma.data,
                       ln1_beta.data, norm1, batch_size,
                       attention.smolgen.dense1_width, 1e-3f);

        std::vector<float> dense2;
        DenseAffine(norm1.data(), batch_size, attention.smolgen.dense1_width,
                    dense2_w.data, nullptr, attention.smolgen.dense2_width,
                    dense2);
        for (int row = 0; row < batch_size; ++row) {
          for (int col = 0; col < attention.smolgen.dense2_width; ++col) {
            const std::size_t idx =
                static_cast<std::size_t>(row) * attention.smolgen.dense2_width +
                col;
            dense2[idx] += dense2_b.data[static_cast<std::size_t>(col)];
          }
        }
        ApplyDenseActivation(
            dense2, batch_size, attention.smolgen.dense2_width,
            ActivationFromName(resolved_execution_plan_.format.activations
                                   .smolgen_activation));

        std::vector<float> norm2(dense2.size(), 0.0f);
        ApplyLayerNorm(dense2.data(), nullptr, 1.0f, ln2_gamma.data,
                       ln2_beta.data, norm2, batch_size,
                       attention.smolgen.dense2_width, 1e-3f);

        DenseAffine(norm2.data(), batch_size * attention.heads,
                    attention.smolgen.dense2_width_per_head, global.data,
                    nullptr, attention.squares * attention.squares,
                    attention_bias);
      }

      const int score_rows = batch_size * attention.heads * attention.squares;
      std::vector<float> scores(
          static_cast<std::size_t>(score_rows) * attention.squares, 0.0f);
      const float scale =
          1.0f / std::sqrt(static_cast<float>(attention.head_depth));
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int head = 0; head < attention.heads; ++head) {
          for (int query_square = 0; query_square < attention.squares;
               ++query_square) {
            const int score_row =
                ((batch * attention.heads + head) * attention.squares) +
                query_square;
            for (int key_square = 0; key_square < attention.squares;
                 ++key_square) {
              float sum = 0.0f;
              for (int depth = 0; depth < attention.head_depth; ++depth) {
                const int channel = head * attention.head_depth + depth;
                const std::size_t query_index =
                    (static_cast<std::size_t>(batch) * attention.squares +
                     query_square) *
                        attention.qkv_width +
                    channel;
                const std::size_t key_index =
                    (static_cast<std::size_t>(batch) * attention.squares +
                     key_square) *
                        attention.qkv_width +
                    channel;
                sum += query[query_index] * key[key_index];
              }
              sum *= scale;
              if (!attention_bias.empty()) {
                const std::size_t bias_index =
                    (static_cast<std::size_t>(batch) * attention.heads + head) *
                        attention.squares * attention.squares +
                    static_cast<std::size_t>(query_square) * attention.squares +
                    key_square;
                sum += attention_bias[bias_index];
              }
              scores[static_cast<std::size_t>(score_row) * attention.squares +
                     key_square] = sum;
            }
          }
        }
      }
      ApplySoftmaxRows(scores, score_rows, attention.squares);

      std::vector<float> context(
          static_cast<std::size_t>(rows) * attention.qkv_width, 0.0f);
      for (int batch = 0; batch < batch_size; ++batch) {
        for (int query_square = 0; query_square < attention.squares;
             ++query_square) {
          float *context_row =
              context.data() +
              (static_cast<std::size_t>(batch) * attention.squares +
               query_square) *
                  attention.qkv_width;
          for (int head = 0; head < attention.heads; ++head) {
            const int score_row =
                ((batch * attention.heads + head) * attention.squares) +
                query_square;
            const float *prob_row =
                scores.data() +
                static_cast<std::size_t>(score_row) * attention.squares;
            for (int depth = 0; depth < attention.head_depth; ++depth) {
              const int channel = head * attention.head_depth + depth;
              float sum = 0.0f;
              for (int key_square = 0; key_square < attention.squares;
                   ++key_square) {
                const std::size_t value_index =
                    (static_cast<std::size_t>(batch) * attention.squares +
                     key_square) *
                        attention.qkv_width +
                    channel;
                sum += prob_row[key_square] * value[value_index];
              }
              context_row[channel] = sum;
            }
          }
        }
      }

      CpuBuffer output;
      output.width = attention.output_width;
      DenseAffine(context.data(), rows, attention.qkv_width, dense_w.data,
                  &dense_b.data, attention.output_width, output.values);

      attention_sources[step.name] = source;
      last_attention_step = step.name;
      remember_output(step, std::move(output));
      last_executed_step = step.name;
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Gate) {
      CpuBuffer output;
      output.width = source.width;
      output.values = source.values;

      for (const auto &tensor_ref : step.tensors) {
        const auto &tensor = TensorAt(tensor_ref.inventory_index);
        int gate_rows = 0;
        if (tensor.data.size() == static_cast<std::size_t>(source.width)) {
          gate_rows = 1;
        } else if (tensor.data.size() %
                       static_cast<std::size_t>(source.width) ==
                   0) {
          gate_rows = static_cast<int>(tensor.data.size() /
                                       static_cast<std::size_t>(source.width));
        }
        if (gate_rows <= 0 || rows % gate_rows != 0) {
          throw std::runtime_error(
              "CPU gate stage tensor dimensions mismatch: " + step.name);
        }

        std::vector<float> next(output.values.size(), 0.0f);
        for (int row = 0; row < rows; ++row) {
          const int gate_row = gate_rows == 1 ? 0 : row % gate_rows;
          for (int col = 0; col < source.width; ++col) {
            const float gate =
                gate_rows == 1
                    ? tensor.data[static_cast<std::size_t>(col)]
                    : tensor.data[static_cast<std::size_t>(col) * gate_rows +
                                  gate_row];
            const std::size_t idx =
                static_cast<std::size_t>(row) * source.width + col;
            if (IsMultiplyGate(tensor_ref.name)) {
              next[idx] = output.values[idx] * gate;
            } else if (IsAddGate(tensor_ref.name)) {
              next[idx] = output.values[idx] + gate;
            } else {
              throw std::runtime_error("CPU gate stage tensor is unknown: " +
                                       tensor_ref.name);
            }
          }
        }
        output.values = std::move(next);
      }

      remember_output(step, std::move(output));
      last_executed_step = step.name;
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::FeedForward) {
      const auto *dense1_w_ref = FindTensorSuffix(step, ".dense1_w");
      const auto *dense1_b_ref = FindTensorSuffix(step, ".dense1_b");
      const auto *dense2_w_ref = FindTensorSuffix(step, ".dense2_w");
      const auto *dense2_b_ref = FindTensorSuffix(step, ".dense2_b");
      if (!dense1_w_ref || !dense1_b_ref || !dense2_w_ref || !dense2_b_ref) {
        throw std::runtime_error(
            "CPU feed-forward stage has missing tensors: " + step.name);
      }

      const auto &dense1_w = TensorAt(dense1_w_ref->inventory_index);
      const auto &dense1_b = TensorAt(dense1_b_ref->inventory_index);
      const auto &dense2_w = TensorAt(dense2_w_ref->inventory_index);
      const auto &dense2_b = TensorAt(dense2_b_ref->inventory_index);
      if (dense1_w.dims.size() != 2 || dense1_b.dims.size() != 1 ||
          dense2_w.dims.size() != 2 || dense2_b.dims.size() != 1) {
        throw std::runtime_error("CPU feed-forward tensor shape is invalid: " +
                                 step.name);
      }

      const int input_width = static_cast<int>(dense1_w.dims[1]);
      const int hidden_width = static_cast<int>(dense1_w.dims[0]);
      const int dense2_input_width = static_cast<int>(dense2_w.dims[1]);
      const int output_width = static_cast<int>(dense2_w.dims[0]);
      if (input_width != source.width || dense2_input_width != hidden_width ||
          dense1_b.data.size() != static_cast<std::size_t>(hidden_width) ||
          dense2_b.data.size() != static_cast<std::size_t>(output_width)) {
        throw std::runtime_error(
            "CPU feed-forward tensor dimensions mismatch: " + step.name);
      }

      std::vector<float> hidden(static_cast<std::size_t>(rows) *
                                    static_cast<std::size_t>(hidden_width),
                                0.0f);
      for (int row = 0; row < rows; ++row) {
        const float *input_row =
            source.values.data() + static_cast<std::size_t>(row) * input_width;
        float *hidden_row =
            hidden.data() + static_cast<std::size_t>(row) * hidden_width;
        for (int out = 0; out < hidden_width; ++out) {
          const float *weight_row = dense1_w.data.data() +
                                    static_cast<std::size_t>(out) * input_width;
          float sum = dense1_b.data[static_cast<std::size_t>(out)];
          for (int in = 0; in < input_width; ++in)
            sum += input_row[in] * weight_row[in];
          hidden_row[out] = sum;
        }
      }
      ApplyDenseActivation(
          hidden, rows, hidden_width,
          ActivationFromName(
              resolved_execution_plan_.format.activations.ffn_activation));

      CpuBuffer output;
      output.width = output_width;
      output.values.assign(static_cast<std::size_t>(rows) *
                               static_cast<std::size_t>(output_width),
                           0.0f);
      for (int row = 0; row < rows; ++row) {
        const float *hidden_row =
            hidden.data() + static_cast<std::size_t>(row) * hidden_width;
        float *output_row =
            output.values.data() + static_cast<std::size_t>(row) * output_width;
        for (int out = 0; out < output_width; ++out) {
          const float *weight_row =
              dense2_w.data.data() +
              static_cast<std::size_t>(out) * hidden_width;
          float sum = dense2_b.data[static_cast<std::size_t>(out)];
          for (int in = 0; in < hidden_width; ++in)
            sum += hidden_row[in] * weight_row[in];
          output_row[out] = sum;
        }
      }

      feed_forward_sources[step.name] = source;
      last_feed_forward_step = step.name;
      remember_output(step, std::move(output));
      last_executed_step = step.name;
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::LayerNorm) {
      const auto *scale_ref =
          FindTensorKind(step, NetworkWeightTensorKind::NormScale);
      const auto *bias_ref =
          FindTensorKind(step, NetworkWeightTensorKind::NormBias);
      if (!scale_ref || !bias_ref)
        throw std::runtime_error("CPU layernorm stage has missing tensors: " +
                                 step.name);
      const auto &scale = TensorAt(scale_ref->inventory_index);
      const auto &bias = TensorAt(bias_ref->inventory_index);
      if (scale.data.size() != static_cast<std::size_t>(source.width) ||
          bias.data.size() != static_cast<std::size_t>(source.width)) {
        throw std::runtime_error("CPU layernorm stage shape mismatch: " +
                                 step.name);
      }

      CpuBuffer output;
      output.width = source.width;
      output.values.assign(source.values.size(), 0.0f);

      const CpuBuffer *residual_parent = nullptr;
      float residual_scale = 1.0f;
      float epsilon =
          DenseLayerNormEpsilon(resolved_execution_plan_, step.name);
      if (!last_feed_forward_step.empty() &&
          last_executed_step == last_feed_forward_step) {
        const auto parent_it =
            feed_forward_sources.find(last_feed_forward_step);
        if (parent_it != feed_forward_sources.end() &&
            parent_it->second.width == source.width &&
            parent_it->second.values.size() == source.values.size()) {
          residual_parent = &parent_it->second;
          residual_scale = FeedForwardResidualScale(resolved_execution_plan_,
                                                    last_feed_forward_step);
          epsilon = FeedForwardLayerNormEpsilon(resolved_execution_plan_,
                                                last_feed_forward_step);
        }
      } else if (!last_attention_step.empty() &&
                 last_executed_step == last_attention_step) {
        const auto parent_it = attention_sources.find(last_attention_step);
        if (parent_it != attention_sources.end() &&
            parent_it->second.width == source.width &&
            parent_it->second.values.size() == source.values.size()) {
          residual_parent = &parent_it->second;
          residual_scale =
              FeedForwardResidualScale(resolved_execution_plan_, step.name);
          epsilon =
              AttentionLayerNormEpsilon(resolved_execution_plan_, step.name);
        }
      }

      ApplyLayerNorm(source.values.data(),
                     residual_parent ? residual_parent->values.data() : nullptr,
                     residual_scale, scale.data, bias.data, output.values, rows,
                     source.width, epsilon);
      remember_output(step, std::move(output));
      last_executed_step = step.name;
      continue;
    }

    throw std::runtime_error("CPU transformer backend reached unsupported "
                             "execution step: " +
                             step.name);
  }

  const std::string policy_prefix =
      "policy." + resolved_execution_plan_.policy_head + ".";
  const std::string value_prefix =
      "value." + resolved_execution_plan_.value_head + ".";
  const std::string policy_stage = outputs.count(policy_prefix + "policy_map")
                                       ? policy_prefix + "policy_map"
                                       : policy_prefix + "output";
  const std::string value_stage =
      outputs.count(value_prefix + "dense2")
          ? value_prefix + "dense2"
          : (outputs.count(value_prefix + "output") ? value_prefix + "output"
                                                    : last_value);
  const std::string moves_stage = outputs.count("moves_left.output")
                                      ? "moves_left.output"
                                      : last_moves_left;

  const CpuBuffer &policy = RequireBuffer(outputs, policy_stage);
  const CpuBuffer &value = RequireBuffer(outputs, value_stage);
  const CpuBuffer *moves_left = nullptr;
  if (tensor_plan_.moves_left)
    moves_left = &RequireBuffer(outputs, moves_stage);

  return DecodeNetworkOutputBatch(
      tensor_plan_, policy.values.data(), policy.values.size(),
      value.values.data(), value.values.size(),
      moves_left ? moves_left->values.data() : nullptr,
      moves_left ? moves_left->values.size() : 0, batch_size);
}

} // namespace Cpu
} // namespace NN
} // namespace MetalFish

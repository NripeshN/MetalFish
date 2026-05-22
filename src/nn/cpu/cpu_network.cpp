/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cpu_network.h"

#include "../input_plane_packing.h"
#include "../network_output_decoder.h"
#include "../network_tensor_plan.h"
#include "../weights.h"

#include <algorithm>
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
         value.substr(0, prefix.size()) == prefix;
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

std::string FirstUnsupportedExecutionStep(
    const NetworkResolvedExecutionPlan &plan) {
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::Attention) {
      return "CPU transformer backend does not support attention yet: " +
             step.name;
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
      return "CPU transformer backend does not support policy mapping yet: " +
             step.name;
    }
  }
  for (const auto &step : plan.steps) {
    if (step.kind == NetworkExecutionOpKind::Dense) {
      if (!IsSimpleDenseStep(step)) {
        return "CPU transformer backend does not support compound dense "
               "stage yet: " +
               step.name;
      }
      continue;
    }
    if (step.kind == NetworkExecutionOpKind::LayerNorm) {
      if (!IsSimpleLayerNormStep(step)) {
        return "CPU transformer backend does not support compound layernorm "
               "stage yet: " +
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
      return ActivationFromName(
          plan.format.attention_body ? plan.format.activations.default_activation
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

const CpuBuffer &RequireBuffer(const std::unordered_map<std::string, CpuBuffer>
                                   &outputs,
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
    const auto resolved_inventory =
        CreateResolvedNetworkWeightInventory(inventory,
                                             resolved_execution_plan_);
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
    throw std::runtime_error("CPU transformer backend weight validation failed: " +
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
  out << "CPU transformer backend (format: "
      << format_.Summary() << ", tensors: " << tensor_plan_.Summary()
      << ", execution: " << resolved_execution_plan_.Summary()
      << ", weight_bytes=" << weight_bytes_
      << ", executor="
      << (unsupported_execution_reason_.empty() ? "dense-layernorm"
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
    throw std::runtime_error("CPU transformer backend tensor index out of range");
  return tensors_[index];
}

std::vector<NetworkOutput>
CpuNetwork::RunBatch(const std::vector<InputPlanes> &inputs) const {
  if (!unsupported_execution_reason_.empty())
    throw std::runtime_error(UnsupportedExecutionMessage());

  const int batch_size = static_cast<int>(inputs.size());
  std::vector<float> raw_inputs(
      static_cast<std::size_t>(batch_size) * kPackedInputPlaneCount *
      kPackedInputSquareCount);
  const std::size_t input_plane_floats =
      static_cast<std::size_t>(kPackedInputPlaneCount) *
      kPackedInputSquareCount;
  for (int b = 0; b < batch_size; ++b) {
    std::memcpy(raw_inputs.data() + static_cast<std::size_t>(b) *
                                      input_plane_floats,
                inputs[static_cast<std::size_t>(b)][0].data(),
                input_plane_floats * sizeof(float));
  }

  std::vector<std::uint64_t> masks;
  std::vector<float> values;
  PackInputPlanesRaw(raw_inputs.data(), batch_size, masks, values);

  CpuBuffer base;
  base.values = std::move(values);
  base.width = kPackedInputPlaneCount;

  std::unordered_map<std::string, CpuBuffer> outputs;
  std::string last_body;
  std::string last_policy;
  std::string last_value;
  std::string last_moves_left;

  const auto source_for = [&](const std::string &name) -> const CpuBuffer & {
    const std::string policy_prefix =
        "policy." + resolved_execution_plan_.policy_head + ".";
    const std::string value_prefix =
        "value." + resolved_execution_plan_.value_head + ".";

    if (StartsWith(name, policy_prefix)) {
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

  for (const auto &step : resolved_execution_plan_.steps) {
    if (step.kind == NetworkExecutionOpKind::InputPack ||
        step.kind == NetworkExecutionOpKind::OutputDecode) {
      continue;
    }

    const CpuBuffer &source = source_for(step.name);
    if (source.width <= 0 || source.values.size() % source.width != 0) {
      throw std::runtime_error("CPU transformer backend source shape mismatch: " +
                               step.name);
    }
    const int rows = static_cast<int>(source.values.size() / source.width);

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
        throw std::runtime_error("CPU dense stage has unresolved tensor shape: " +
                                 step.name);
      }
      const int out_width = static_cast<int>(weight.dims[0]);
      const int in_width = static_cast<int>(weight.dims[1]);
      if (in_width != source.width ||
          bias.data.size() != static_cast<std::size_t>(out_width) ||
          weight.data.size() !=
              static_cast<std::size_t>(out_width) *
                  static_cast<std::size_t>(in_width)) {
        throw std::runtime_error("CPU dense stage shape mismatch: " +
                                 step.name);
      }

      CpuBuffer output;
      output.width = out_width;
      output.values.assign(
          static_cast<std::size_t>(rows) * static_cast<std::size_t>(out_width),
          0.0f);
      for (int row = 0; row < rows; ++row) {
        const float *input_row =
            source.values.data() + static_cast<std::size_t>(row) * in_width;
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
      ApplyDenseActivation(output.values, rows, output.width,
                           DenseStageActivationForName(resolved_execution_plan_,
                                                       step.name));
      remember_output(step, std::move(output));
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
      for (int row = 0; row < rows; ++row) {
        const float *input_row =
            source.values.data() + static_cast<std::size_t>(row) * source.width;
        float *output_row =
            output.values.data() + static_cast<std::size_t>(row) * source.width;
        float mean = 0.0f;
        for (int col = 0; col < source.width; ++col)
          mean += input_row[col];
        mean /= static_cast<float>(source.width);

        float variance = 0.0f;
        for (int col = 0; col < source.width; ++col) {
          const float delta = input_row[col] - mean;
          variance += delta * delta;
        }
        variance /= static_cast<float>(source.width);
        const float inv_std = 1.0f / std::sqrt(variance + 1e-5f);

        for (int col = 0; col < source.width; ++col) {
          const std::size_t idx = static_cast<std::size_t>(col);
          output_row[col] =
              (input_row[col] - mean) * inv_std * scale.data[idx] +
              bias.data[idx];
        }
      }
      remember_output(step, std::move(output));
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
  const std::string policy_stage =
      outputs.count(policy_prefix + "policy_map") ? policy_prefix + "policy_map"
                                                  : policy_prefix + "output";
  const std::string value_stage =
      outputs.count(value_prefix + "dense2")
          ? value_prefix + "dense2"
          : (outputs.count(value_prefix + "output") ? value_prefix + "output"
                                                    : last_value);
  const std::string moves_stage =
      outputs.count("moves_left.output") ? "moves_left.output"
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

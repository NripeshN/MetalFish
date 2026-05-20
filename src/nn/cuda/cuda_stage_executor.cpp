/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_stage_executor.h"

#include "cuda_attention_plan.h"
#include "cuda_execution_schedule.h"
#include "cuda_kernels.h"
#include "cuda_plan_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

std::string CudaErrorMessage(std::string_view op, cudaError_t status) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorString(status);
  return out.str();
}

CudaActivationKind ActivationFromString(const std::string &activation) {
  if (activation == "relu_2")
    return CudaActivationKind::Relu2;
  if (activation == "tanh")
    return CudaActivationKind::Tanh;
  if (activation == "sigmoid")
    return CudaActivationKind::Sigmoid;
  if (activation == "swish")
    return CudaActivationKind::Swish;
  if (activation == "mish")
    return CudaActivationKind::Mish;
  if (activation == "selu")
    return CudaActivationKind::Selu;
  return CudaActivationKind::Relu;
}

struct DenseStageActivation {
  enum class Mode {
    Linear,
    Elementwise,
    Softmax,
  };

  Mode mode = Mode::Linear;
  CudaActivationKind kind = CudaActivationKind::Relu;
};

void RequireDenseTensors(const NetworkResolvedExecutionStep &dense) {
  if (dense.kind != NetworkExecutionOpKind::Dense || dense.tensors.size() < 2) {
    throw std::runtime_error("CUDA dense stage has missing tensors");
  }
}

void RequireLayerNormTensors(const NetworkResolvedExecutionStep &norm) {
  if (norm.kind != NetworkExecutionOpKind::LayerNorm ||
      norm.tensors.size() < 2) {
    throw std::runtime_error("CUDA layernorm stage has missing tensors");
  }
}

void RequireGateTensors(const NetworkResolvedExecutionStep &gate) {
  if (gate.kind != NetworkExecutionOpKind::Gate || gate.tensors.empty()) {
    throw std::runtime_error("CUDA gate stage has missing tensors");
  }
}

void RequireFeedForwardTensors(const NetworkResolvedExecutionStep &ffn) {
  if (ffn.kind != NetworkExecutionOpKind::FeedForward ||
      ffn.tensors.size() < 4) {
    throw std::runtime_error("CUDA feed-forward stage has missing tensors");
  }
}

void RequirePolicyMapTensors(const NetworkResolvedExecutionStep &policy_map) {
  if (policy_map.kind != NetworkExecutionOpKind::PolicyMap ||
      policy_map.tensors.empty()) {
    throw std::runtime_error("CUDA policy-map stage has missing tensors");
  }
}

bool IsMultiplyGate(std::string_view name) {
  return name.find("mult_gate") != std::string_view::npos;
}

bool IsAddGate(std::string_view name) {
  return name.find("add_gate") != std::string_view::npos;
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.ends_with(suffix);
}

DenseStageActivation ActivationFromName(const std::string &activation) {
  if (activation.empty())
    return {};
  if (activation == "softmax")
    return {DenseStageActivation::Mode::Softmax, CudaActivationKind::Relu};
  return {DenseStageActivation::Mode::Elementwise,
          ActivationFromString(activation)};
}

DenseStageActivation
DenseStageActivationForName(const NetworkResolvedExecutionPlan &execution_plan,
                            std::string_view name) {
  const std::string policy_prefix =
      "policy." + execution_plan.policy_head + ".";
  if (StartsWith(name, policy_prefix)) {
    if (name == policy_prefix + "dense2" || name == policy_prefix + "dense3") {
      return {};
    }
    if (name == policy_prefix + "output") {
      if (!execution_plan.format.attention_policy)
        return {};
      return ActivationFromName(
          execution_plan.format.attention_body
              ? execution_plan.format.activations.default_activation
              : std::string("selu"));
    }
  }

  const std::string value_prefix = "value." + execution_plan.value_head + ".";
  if (StartsWith(name, value_prefix)) {
    if (name == value_prefix + "dense2") {
      return ActivationFromName(execution_plan.format.wdl ? "softmax" : "tanh");
    }
    if (name == value_prefix + "output" || name == value_prefix + "dense1") {
      return ActivationFromName(
          execution_plan.format.activations.default_activation);
    }
  }

  if (name == "moves_left.output")
    return ActivationFromName("relu");
  if (name == "moves_left.dense0" || name == "moves_left.dense1") {
    return ActivationFromName(
        execution_plan.format.activations.default_activation);
  }

  if (name == "body.input_embedding_preprocess")
    return {};
  if (name == "body.input_embedding") {
    return ActivationFromName(
        execution_plan.format.activations.default_activation);
  }

  return ActivationFromName(execution_plan.format.activations.ffn_activation);
}

void CreateTimingEvent(cudaEvent_t *event) {
  const cudaError_t status = cudaEventCreateWithFlags(event, cudaEventDefault);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaEventCreate", status));
}

class CudaStageTimer {
public:
  CudaStageTimer(CudaStageTimingCollector *collector, std::string name,
                 CudaExecutionScheduleKind kind, cudaStream_t stream)
      : collector_(collector), name_(std::move(name)), kind_(kind),
        stream_(stream) {
    if (!collector_)
      return;
    CreateTimingEvent(&start_);
    CreateTimingEvent(&stop_);
    const cudaError_t status = cudaEventRecord(start_, stream_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventRecord(start)", status));
  }

  CudaStageTimer(const CudaStageTimer &) = delete;
  CudaStageTimer &operator=(const CudaStageTimer &) = delete;

  ~CudaStageTimer() {
    if (start_)
      cudaEventDestroy(start_);
    if (stop_)
      cudaEventDestroy(stop_);
  }

  void Stop() {
    if (!collector_ || stopped_)
      return;
    cudaError_t status = cudaEventRecord(stop_, stream_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventRecord(stop)", status));
    status = cudaEventSynchronize(stop_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventSynchronize", status));
    float millis = 0.0f;
    status = cudaEventElapsedTime(&millis, start_, stop_);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaEventElapsedTime", status));
    collector_->Add(name_, kind_, millis);
    stopped_ = true;
  }

private:
  CudaStageTimingCollector *collector_ = nullptr;
  std::string name_;
  CudaExecutionScheduleKind kind_ = CudaExecutionScheduleKind::Unsupported;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  bool stopped_ = false;
};

bool IsDynamicPositionPreprocessName(std::string_view name) {
  return name == "body.input_embedding_preprocess";
}

int AttentionHeadCount(const NetworkResolvedExecutionPlan &plan,
                       std::string_view name) {
  if (StartsWith(name, "body.encoder."))
    return plan.format.body_attention_heads;
  if (StartsWith(name, "policy."))
    return plan.format.policy_attention_heads;
  return 0;
}

const NetworkResolvedTensorRef &
RequireTensorSuffix(const NetworkResolvedExecutionStep &step,
                    std::string_view suffix) {
  for (const auto &tensor : step.tensors) {
    if (EndsWith(tensor.name, suffix))
      return tensor;
  }
  throw std::runtime_error("CUDA stage is missing tensor " +
                           std::string(suffix) + " in " + step.name);
}

CudaDeviceTensorView TensorBySuffix(const NetworkResolvedExecutionStep &step,
                                    const CudaWeightBuffers &weights,
                                    std::string_view suffix) {
  const auto &ref = RequireTensorSuffix(step, suffix);
  return weights.TensorAt(ref.inventory_index);
}

void RequireTapeShape(const CudaExecutionBufferBinding &binding, int rows,
                      int width, std::string_view role) {
  if (binding.rows != rows || binding.width != width ||
      binding.entries != static_cast<std::size_t>(rows) * width) {
    throw std::runtime_error("CUDA " + std::string(role) +
                             " tape size mismatch");
  }
}

std::string ShapeString(int rows, int width) {
  return std::to_string(rows) + "x" + std::to_string(width);
}

std::string BindingShapeString(const CudaExecutionBufferBinding &binding) {
  return ShapeString(binding.rows, binding.width);
}

int DenseStageInputWidth(const NetworkResolvedExecutionStep &dense) {
  if (dense.kind != NetworkExecutionOpKind::Dense || dense.tensors.empty() ||
      dense.tensors[0].dims.size() != 2) {
    return 0;
  }
  return static_cast<int>(dense.tensors[0].dims[1]);
}

void MaybeFlattenSquareRowsForDenseStage(
    const NetworkResolvedExecutionStep &step, CudaExecutionScheduleKind kind,
    int batch_size, int &stage_input_rows, int &stage_input_width) {
  if (!IsCudaDenseScheduleEntry(kind) ||
      IsDynamicPositionPreprocessName(step.name) || batch_size <= 0 ||
      stage_input_rows <= 0 || stage_input_width <= 0 ||
      stage_input_rows % batch_size != 0) {
    return;
  }

  const int expected_width = DenseStageInputWidth(step);
  if (expected_width <= 0 || expected_width == stage_input_width)
    return;

  const int rows_per_batch = stage_input_rows / batch_size;
  if (expected_width == rows_per_batch * stage_input_width) {
    stage_input_rows = batch_size;
    stage_input_width = expected_width;
  }
}

struct CudaFeedForwardTensors {
  CudaDeviceTensorView dense1_weight;
  CudaDeviceTensorView dense1_bias;
  CudaDeviceTensorView dense2_weight;
  CudaDeviceTensorView dense2_bias;
  int input_width = 0;
  int hidden_width = 0;
  int output_width = 0;
};

struct CudaSmolgenStageOutput {
  float *compressed = nullptr;
  float *dense1 = nullptr;
  float *activation1 = nullptr;
  float *norm1 = nullptr;
  float *dense2 = nullptr;
  float *activation2 = nullptr;
  float *norm2 = nullptr;
  float *global_bias = nullptr;
};

CudaFeedForwardTensors
ResolveFeedForwardTensors(const NetworkResolvedExecutionStep &ffn,
                          const CudaWeightBuffers &weights) {
  RequireFeedForwardTensors(ffn);
  CudaFeedForwardTensors resolved;
  resolved.dense1_weight = weights.TensorAt(ffn.tensors[0].inventory_index);
  resolved.dense1_bias = weights.TensorAt(ffn.tensors[1].inventory_index);
  resolved.dense2_weight = weights.TensorAt(ffn.tensors[2].inventory_index);
  resolved.dense2_bias = weights.TensorAt(ffn.tensors[3].inventory_index);
  if (resolved.dense1_weight.dims.size() != 2 ||
      resolved.dense1_bias.dims.size() != 1 ||
      resolved.dense2_weight.dims.size() != 2 ||
      resolved.dense2_bias.dims.size() != 1) {
    throw std::runtime_error("CUDA feed-forward tensor shape is invalid");
  }

  resolved.hidden_width = static_cast<int>(resolved.dense1_weight.dims[0]);
  resolved.input_width = static_cast<int>(resolved.dense1_weight.dims[1]);
  resolved.output_width = static_cast<int>(resolved.dense2_weight.dims[0]);
  const int dense2_input_width =
      static_cast<int>(resolved.dense2_weight.dims[1]);
  if (resolved.input_width <= 0 || resolved.hidden_width <= 0 ||
      resolved.output_width <= 0 ||
      dense2_input_width != resolved.hidden_width ||
      resolved.dense1_bias.elements !=
          static_cast<std::size_t>(resolved.hidden_width) ||
      resolved.dense2_bias.elements !=
          static_cast<std::size_t>(resolved.output_width)) {
    throw std::runtime_error("CUDA feed-forward tensor dimensions mismatch");
  }
  return resolved;
}

std::size_t
BodyEncoderLayerCount(const NetworkResolvedExecutionPlan &execution_plan) {
  std::size_t max_layer = 0;
  bool found = false;
  constexpr std::string_view kPrefix = "body.encoder.";
  for (const auto &step : execution_plan.steps) {
    if (!CudaStageNameStartsWith(step.name, kPrefix))
      continue;
    const std::string_view rest(step.name.data() + kPrefix.size(),
                                step.name.size() - kPrefix.size());
    const std::size_t dot = rest.find('.');
    if (dot == std::string_view::npos || dot == 0)
      continue;
    bool numeric = true;
    for (std::size_t i = 0; i < dot; ++i) {
      if (rest[i] < '0' || rest[i] > '9') {
        numeric = false;
        break;
      }
    }
    if (!numeric)
      continue;
    const std::size_t layer =
        static_cast<std::size_t>(std::stoul(std::string(rest.substr(0, dot))));
    max_layer = std::max(max_layer, layer + 1);
    found = true;
  }
  return found ? max_layer : 0;
}

float FeedForwardResidualScale(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name) {
  if (!CudaStageNameStartsWith(stage_name, "body.input_embedding_ffn") &&
      !CudaStageNameStartsWith(stage_name, "body.encoder.")) {
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
  if (CudaStageNameStartsWith(stage_name, "body.input_embedding_ffn"))
    return 1e-3f;
  if (CudaStageNameStartsWith(stage_name, "body.encoder.")) {
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
  if (CudaStageNameStartsWith(stage_name, "body.encoder.")) {
    return execution_plan.format.input_embedding == INPUT_EMBEDDING_PE_DENSE
               ? 1e-3f
               : 1e-6f;
  }
  if (CudaStageNameStartsWith(stage_name, "policy."))
    return 1e-6f;
  return 1e-5f;
}

const NetworkResolvedExecutionStep *
FindStep(const NetworkResolvedExecutionPlan &execution_plan,
         std::string_view name) {
  for (const auto &step : execution_plan.steps) {
    if (step.name == name)
      return &step;
  }
  return nullptr;
}

} // namespace

void CopyDeviceFloatRows(float *dst, int dst_stride, const float *src,
                         int src_stride, int rows, int width,
                         std::string_view name, cudaStream_t stream) {
  if (rows <= 0 || width <= 0)
    return;
  if (!dst || !src)
    throw std::runtime_error("CUDA row copy missing buffer: " +
                             std::string(name));
  if (dst_stride < width || src_stride < width)
    throw std::runtime_error("CUDA row copy stride too small: " +
                             std::string(name));

  cudaError_t status = cudaSuccess;
  if (stream) {
    status = cudaMemcpy2DAsync(
        dst, static_cast<std::size_t>(dst_stride) * sizeof(float), src,
        static_cast<std::size_t>(src_stride) * sizeof(float),
        static_cast<std::size_t>(width) * sizeof(float), rows,
        cudaMemcpyDeviceToDevice, stream);
  } else {
    status =
        cudaMemcpy2D(dst, static_cast<std::size_t>(dst_stride) * sizeof(float),
                     src, static_cast<std::size_t>(src_stride) * sizeof(float),
                     static_cast<std::size_t>(width) * sizeof(float), rows,
                     cudaMemcpyDeviceToDevice);
  }
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

const CudaDenseStageOutput *
CudaDenseStageSequenceOutput::FindStage(std::string_view name) const {
  for (const auto &stage : stages) {
    if (stage.first == name)
      return &stage.second;
  }
  return nullptr;
}

void CudaStageTimingCollector::Add(std::string name,
                                   CudaExecutionScheduleKind kind,
                                   float millis) {
  records_.push_back(CudaStageTimingRecord{std::move(name), kind, millis});
}

void CudaStageInputBindings::Add(std::string stage_name,
                                 std::string source_stage_name) {
  if (stage_name.empty())
    throw std::runtime_error("CUDA stage input binding has empty stage name");
  if (FindSource(stage_name)) {
    throw std::runtime_error("CUDA stage input binding is duplicated: " +
                             stage_name);
  }
  bindings_.push_back(CudaStageInputBinding{std::move(stage_name),
                                            std::move(source_stage_name)});
}

const std::string *
CudaStageInputBindings::FindSource(std::string_view stage_name) const {
  for (const auto &binding : bindings_) {
    if (binding.stage_name == stage_name)
      return &binding.source_stage_name;
  }
  return nullptr;
}

CudaStageInputBindings
CreateCudaStageInputBindings(const NetworkResolvedExecutionPlan &execution_plan,
                             const CudaExecutionSchedule &schedule) {
  CudaStageInputBindings bindings;
  const std::string body_stage = LastCudaOutputStageInGroup(
      execution_plan, schedule, CudaPlanStageGroup::Body);
  if (body_stage.empty())
    return bindings;

  bool policy_bound = false;
  bool value_bound = false;
  bool moves_left_bound = false;
  for (const auto &entry : schedule.entries) {
    if (!IsCudaOutputScheduleEntry(entry.kind) ||
        entry.first_step >= execution_plan.steps.size()) {
      continue;
    }
    const auto &step = execution_plan.steps[entry.first_step];
    const CudaPlanStageGroup group =
        ClassifyCudaPlanStage(execution_plan, step.name);
    bool *seen = nullptr;
    if (group == CudaPlanStageGroup::Policy)
      seen = &policy_bound;
    else if (group == CudaPlanStageGroup::Value)
      seen = &value_bound;
    else if (group == CudaPlanStageGroup::MovesLeft)
      seen = &moves_left_bound;

    if (seen && !*seen) {
      if (step.name != body_stage)
        bindings.Add(step.name, body_stage);
      *seen = true;
    }
  }
  const std::string policy_prefix =
      CudaPlanStagePrefix(execution_plan, CudaPlanStageGroup::Policy);
  if (!policy_prefix.empty()) {
    const std::string policy_embedding = policy_prefix + "output";
    if (FindCudaStageEntry(execution_plan, schedule, policy_embedding)) {
      const std::string query_stage = policy_prefix + "dense2";
      const std::string key_stage = policy_prefix + "dense3";
      if (FindCudaStageEntry(execution_plan, schedule, query_stage) &&
          !bindings.FindSource(query_stage)) {
        bindings.Add(query_stage, policy_embedding);
      }
      if (FindCudaStageEntry(execution_plan, schedule, key_stage) &&
          !bindings.FindSource(key_stage)) {
        bindings.Add(key_stage, policy_embedding);
      }
    }
  }
  return bindings;
}

CudaDenseStageOutput
ExecuteDenseActivationStage(const NetworkResolvedExecutionPlan &execution_plan,
                            const NetworkResolvedExecutionStep &dense,
                            const CudaWeightBuffers &weights,
                            const float *input, const CudaExecutionTape &tape,
                            CudaExecutionWorkspace &workspace, int rows) {
  if (rows <= 0)
    throw std::runtime_error("CUDA dense stage received empty batch");
  if (!input)
    throw std::runtime_error("CUDA dense stage input is missing");
  RequireDenseTensors(dense);

  const auto dense_weight = weights.TensorAt(dense.tensors[0].inventory_index);
  const auto dense_bias = weights.TensorAt(dense.tensors[1].inventory_index);
  if (dense_weight.dims.size() != 2 || dense_bias.dims.size() != 1) {
    throw std::runtime_error("CUDA dense stage tensor shape is invalid");
  }

  const int output_width = static_cast<int>(dense_weight.dims[0]);
  const int input_width = static_cast<int>(dense_weight.dims[1]);
  if (input_width <= 0 || output_width <= 0 ||
      dense_bias.elements != static_cast<std::size_t>(output_width)) {
    throw std::runtime_error("CUDA dense stage tensor dimensions mismatch");
  }

  const auto &dense_binding = tape.RequireName(dense.name + ".dense");
  const auto &activation_binding = tape.RequireName(dense.name + ".activation");
  const std::size_t entries =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(output_width);
  if (dense_binding.rows != rows || dense_binding.width != output_width ||
      dense_binding.entries != entries || activation_binding.rows != rows ||
      activation_binding.width != output_width ||
      activation_binding.entries != entries) {
    throw std::runtime_error(
        "CUDA dense stage tape size mismatch: " + dense.name +
        " binding=" + BindingShapeString(dense_binding) +
        " activation=" + BindingShapeString(activation_binding) +
        " actual=" + ShapeString(rows, output_width));
  }

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense_binding);
  output.input_width = input_width;
  output.output_width = output_width;
  output.rows = rows;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, dense_weight.data, dense_bias.data,
                          output.dense, rows, input_width, output_width,
                          stream);
  const DenseStageActivation activation =
      DenseStageActivationForName(execution_plan, dense.name);
  if (activation.mode == DenseStageActivation::Mode::Linear) {
    output.activation = output.dense;
    output.output = output.dense;
  } else {
    output.activation = tape.Reserve(workspace, activation_binding);
    output.output = output.activation;
    if (activation.mode == DenseStageActivation::Mode::Softmax) {
      LaunchAttentionSoftmaxKernel(output.dense, output.activation, rows,
                                   output_width, stream);
    } else {
      LaunchActivationKernel(output.dense, output.activation,
                             static_cast<int>(entries), activation.kind,
                             stream);
    }
  }
  return output;
}

CudaDenseStageOutput ExecuteDenseActivationLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int rows) {
  RequireLayerNormTensors(norm);

  CudaDenseStageOutput output = ExecuteDenseActivationStage(
      execution_plan, dense, weights, input, tape, workspace, rows);
  const auto gamma = weights.TensorAt(norm.tensors[0].inventory_index);
  const auto beta = weights.TensorAt(norm.tensors[1].inventory_index);
  if (gamma.dims.size() != 1 || beta.dims.size() != 1) {
    throw std::runtime_error("CUDA layernorm stage tensor shape is invalid");
  }

  if (gamma.elements != static_cast<std::size_t>(output.output_width) ||
      beta.elements != static_cast<std::size_t>(output.output_width)) {
    throw std::runtime_error("CUDA layernorm stage tensor dimensions mismatch");
  }

  const auto &norm_binding = tape.RequireName(norm.name + ".normalized");
  const std::size_t entries = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(output.output_width);
  if (norm_binding.entries != entries) {
    throw std::runtime_error("CUDA layernorm stage tape size mismatch");
  }

  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;

  cudaStream_t stream = workspace.Stream();
  LaunchLayerNormKernel(output.activation, gamma.data, beta.data,
                        output.normalized, rows, output.output_width,
                        DenseLayerNormEpsilon(execution_plan, norm.name),
                        stream);
  return output;
}

CudaDenseStageOutput ExecuteDynamicPositionEncodingStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense, const CudaWeightBuffers &weights,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA dynamic PE stage received empty batch");
  if (!input_masks || !input_values)
    throw std::runtime_error("CUDA dynamic PE stage input is missing");
  if (!IsDynamicPositionPreprocessName(dense.name))
    throw std::runtime_error("CUDA dynamic PE stage name is invalid");
  RequireDenseTensors(dense);

  constexpr int kPositionPlanes = 12;
  const int input_planes = execution_plan.tensors.input_planes;
  const int squares = execution_plan.tensors.input_squares;
  if (input_planes <= 0 || squares <= 0 || input_planes < kPositionPlanes)
    throw std::runtime_error("CUDA dynamic PE tensor plan is invalid");

  const auto dense_weight = weights.TensorAt(dense.tensors[0].inventory_index);
  const auto dense_bias = weights.TensorAt(dense.tensors[1].inventory_index);
  if (dense_weight.dims.size() != 2 || dense_bias.dims.size() != 1)
    throw std::runtime_error("CUDA dynamic PE dense tensor shape is invalid");

  const int output_width = static_cast<int>(dense_weight.dims[0]);
  const int input_width = static_cast<int>(dense_weight.dims[1]);
  if (input_width != squares * kPositionPlanes || output_width <= 0 ||
      output_width % squares != 0 ||
      dense_bias.elements != static_cast<std::size_t>(output_width)) {
    throw std::runtime_error(
        "CUDA dynamic PE dense tensor dimensions mismatch");
  }
  const int pe_width = output_width / squares;
  const int concat_width = input_planes + pe_width;
  const int square_rows = batch_size * squares;

  const auto &expanded_binding = tape.RequireName(dense.name + ".expanded");
  const auto &position_input_binding =
      tape.RequireName(dense.name + ".position_input");
  const auto &dense_binding = tape.RequireName(dense.name + ".dense");
  const auto &concat_binding = tape.RequireName(dense.name + ".concat");
  RequireTapeShape(expanded_binding, square_rows, input_planes,
                   "dynamic PE expanded input");
  RequireTapeShape(position_input_binding, batch_size, input_width,
                   "dynamic PE dense input");
  RequireTapeShape(dense_binding, batch_size, output_width,
                   "dynamic PE dense output");
  RequireTapeShape(concat_binding, square_rows, concat_width,
                   "dynamic PE concat output");

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense_binding);
  output.expanded_input = tape.Reserve(workspace, expanded_binding);
  output.position_input = tape.Reserve(workspace, position_input_binding);
  output.normalized = tape.Reserve(workspace, concat_binding);
  output.output = output.normalized;
  output.input_width = input_width;
  output.output_width = concat_width;
  output.rows = square_rows;

  cudaStream_t stream = workspace.Stream();
  LaunchExpandPackedInputPlanesKernel(input_masks, input_values,
                                      output.expanded_input, batch_size,
                                      input_planes, squares, stream);
  LaunchDynamicPositionEncodingInputKernel(
      output.expanded_input, output.position_input, batch_size, input_planes,
      kPositionPlanes, squares, stream);
  LaunchDenseAffineKernel(output.position_input, dense_weight.data,
                          dense_bias.data, output.dense, batch_size,
                          input_width, output_width, stream);
  LaunchDynamicPositionEncodingConcatKernel(
      output.expanded_input, output.dense, output.output, batch_size,
      input_planes, pe_width, squares, stream);
  return output;
}

CudaDenseStageOutput ExecuteGateStage(const NetworkResolvedExecutionStep &gate,
                                      const CudaWeightBuffers &weights,
                                      const float *input, int input_width,
                                      const CudaExecutionTape &tape,
                                      CudaExecutionWorkspace &workspace,
                                      int rows) {
  if (rows <= 0)
    throw std::runtime_error("CUDA gate stage received empty batch");
  if (!input || input_width <= 0)
    throw std::runtime_error("CUDA gate stage input is missing");
  RequireGateTensors(gate);

  const auto &gate_binding = tape.RequireName(gate.name + ".gated");
  const std::size_t entries =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(input_width);
  if (gate_binding.rows != rows || gate_binding.width != input_width ||
      gate_binding.entries != entries) {
    std::ostringstream out;
    out << "CUDA gate stage tape size mismatch: binding=" << gate_binding.rows
        << "x" << gate_binding.width << ", actual=" << rows << "x"
        << input_width;
    throw std::runtime_error(out.str());
  }

  CudaDenseStageOutput output;
  output.gated = tape.Reserve(workspace, gate_binding);
  output.output = output.gated;
  output.input_width = input_width;
  output.output_width = input_width;
  output.rows = rows;

  const float *source = input;
  cudaStream_t stream = workspace.Stream();
  bool launched = false;
  for (const auto &tensor : gate.tensors) {
    const auto gate_weights = weights.TensorAt(tensor.inventory_index);
    int gate_rows = 0;
    if (gate_weights.elements == static_cast<std::size_t>(input_width)) {
      gate_rows = 1;
    } else if (gate_weights.elements % static_cast<std::size_t>(input_width) ==
               0) {
      gate_rows = static_cast<int>(gate_weights.elements /
                                   static_cast<std::size_t>(input_width));
    }
    if (gate_rows <= 0 || rows % gate_rows != 0) {
      throw std::runtime_error("CUDA gate stage tensor dimensions mismatch");
    }
    if (IsMultiplyGate(tensor.name)) {
      LaunchGateKernel(source, gate_weights.data, output.gated, rows,
                       input_width, gate_rows, CudaGateKind::Multiply, stream);
    } else if (IsAddGate(tensor.name)) {
      LaunchGateKernel(source, gate_weights.data, output.gated, rows,
                       input_width, gate_rows, CudaGateKind::Add, stream);
    } else {
      throw std::runtime_error("CUDA gate stage tensor is unknown: " +
                               tensor.name);
    }
    source = output.gated;
    launched = true;
  }
  if (!launched)
    throw std::runtime_error("CUDA gate stage launched no operations");
  return output;
}

CudaDenseStageOutput
ExecuteFeedForwardStage(const NetworkResolvedExecutionPlan &execution_plan,
                        const NetworkResolvedExecutionStep &ffn,
                        const CudaWeightBuffers &weights, const float *input,
                        const CudaExecutionTape &tape,
                        CudaExecutionWorkspace &workspace, int rows) {
  if (rows <= 0)
    throw std::runtime_error("CUDA feed-forward stage received empty batch");
  if (!input)
    throw std::runtime_error("CUDA feed-forward stage input is missing");
  const auto tensors = ResolveFeedForwardTensors(ffn, weights);

  const auto &dense1_binding = tape.RequireName(ffn.name + ".dense1");
  const auto &activation_binding = tape.RequireName(ffn.name + ".activation");
  const auto &dense2_binding = tape.RequireName(ffn.name + ".dense2");
  const std::size_t hidden_entries =
      static_cast<std::size_t>(rows) *
      static_cast<std::size_t>(tensors.hidden_width);
  const std::size_t output_entries =
      static_cast<std::size_t>(rows) *
      static_cast<std::size_t>(tensors.output_width);
  if (dense1_binding.entries != hidden_entries ||
      activation_binding.entries != hidden_entries ||
      dense2_binding.entries != output_entries) {
    throw std::runtime_error("CUDA feed-forward stage tape size mismatch");
  }

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense1_binding);
  output.activation = tape.Reserve(workspace, activation_binding);
  output.feed_forward = tape.Reserve(workspace, dense2_binding);
  output.output = output.feed_forward;
  output.input_width = tensors.input_width;
  output.output_width = tensors.output_width;
  output.rows = rows;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, tensors.dense1_weight.data, nullptr,
                          output.dense, rows, tensors.input_width,
                          tensors.hidden_width, stream);
  LaunchBiasActivationKernel(
      output.dense, tensors.dense1_bias.data, output.activation, rows,
      tensors.hidden_width,
      ActivationFromString(execution_plan.format.activations.ffn_activation),
      stream);
  LaunchDenseAffineKernel(output.activation, tensors.dense2_weight.data,
                          tensors.dense2_bias.data, output.feed_forward, rows,
                          tensors.hidden_width, tensors.output_width, stream);
  return output;
}

CudaDenseStageOutput ExecuteFeedForwardLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &ffn,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int rows) {
  RequireLayerNormTensors(norm);

  CudaDenseStageOutput output = ExecuteFeedForwardStage(
      execution_plan, ffn, weights, input, tape, workspace, rows);
  if (output.input_width != output.output_width) {
    throw std::runtime_error("CUDA feed-forward residual width mismatch");
  }

  const auto gamma = weights.TensorAt(norm.tensors[0].inventory_index);
  const auto beta = weights.TensorAt(norm.tensors[1].inventory_index);
  if (gamma.dims.size() != 1 || beta.dims.size() != 1 ||
      gamma.elements != static_cast<std::size_t>(output.output_width) ||
      beta.elements != static_cast<std::size_t>(output.output_width)) {
    throw std::runtime_error(
        "CUDA feed-forward layernorm tensor dimensions mismatch");
  }

  const auto &residual_binding = tape.RequireName(norm.name + ".residual");
  const auto &norm_binding = tape.RequireName(norm.name + ".normalized");
  const std::size_t entries = static_cast<std::size_t>(rows) *
                              static_cast<std::size_t>(output.output_width);
  if (residual_binding.entries != entries || norm_binding.entries != entries) {
    throw std::runtime_error("CUDA feed-forward layernorm tape size mismatch");
  }

  output.residual = tape.Reserve(workspace, residual_binding);
  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;

  cudaStream_t stream = workspace.Stream();
  LaunchResidualLayerNormKernel(
      input, output.feed_forward, gamma.data, beta.data, output.residual,
      output.normalized, rows, output.output_width,
      FeedForwardResidualScale(execution_plan, ffn.name),
      FeedForwardLayerNormEpsilon(execution_plan, ffn.name), stream);
  return output;
}

CudaDenseStageOutput ExecuteAttentionPolicyMapStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &policy_map,
    const CudaWeightBuffers &weights,
    const CudaDenseStageSequenceOutput &sequence, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA policy-map stage received empty batch");
  if (!execution_plan.format.attention_policy) {
    throw std::runtime_error(
        "CUDA policy-map stage requires attention policy format");
  }
  RequirePolicyMapTensors(policy_map);

  const std::string suffix = ".policy_map";
  if (!EndsWith(policy_map.name, suffix)) {
    throw std::runtime_error("CUDA policy-map stage name is invalid");
  }
  const std::string policy_prefix =
      policy_map.name.substr(0, policy_map.name.size() - suffix.size());
  const CudaDenseStageOutput *query_stage =
      sequence.FindStage(policy_prefix + ".dense2");
  const CudaDenseStageOutput *key_stage =
      sequence.FindStage(policy_prefix + ".dense3");
  if (!query_stage || !query_stage->output || !key_stage ||
      !key_stage->output) {
    throw std::runtime_error("CUDA policy-map Q/K stages are missing");
  }
  if (query_stage->output_width <= 0 ||
      query_stage->output_width != key_stage->output_width) {
    throw std::runtime_error("CUDA policy-map Q/K dimensions mismatch");
  }

  const auto promotion_weights =
      weights.TensorAt(policy_map.tensors[0].inventory_index);
  const int channels = query_stage->output_width;
  if (promotion_weights.elements != static_cast<std::size_t>(4 * channels)) {
    throw std::runtime_error(
        "CUDA policy-map promotion tensor dimensions mismatch");
  }

  const auto &raw_binding = tape.RequireName(policy_map.name + ".raw");
  const auto &mapped_binding = tape.RequireName(policy_map.name + ".mapped");
  RequireTapeShape(raw_binding, batch_size, kNetworkAttentionPolicyScratch,
                   "attention policy raw map");
  RequireTapeShape(mapped_binding, batch_size, kNetworkPolicyOutputs,
                   "attention policy mapped logits");

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, raw_binding);
  output.activation = tape.Reserve(workspace, mapped_binding);
  output.output = output.activation;
  output.input_width = channels;
  output.output_width = kNetworkPolicyOutputs;
  output.rows = batch_size;

  LaunchAttentionPolicyMapKernel(
      query_stage->output, key_stage->output, promotion_weights.data,
      output.dense, output.output, batch_size, channels, workspace.Stream());
  return output;
}

CudaAttentionProjectionOutput ExecuteAttentionInputProjectionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention projection received empty batch");
  if (!input)
    throw std::runtime_error("CUDA attention projection input is missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention projection index is out of range");

  const auto &step = execution_plan.steps[attention_step_index];
  const auto attention = ResolveCudaAttentionStagePlan(
      execution_plan, attention_step_index,
      AttentionHeadCount(execution_plan, step.name));
  const int rows = batch_size * attention.squares;

  const auto q_weight = TensorBySuffix(step, weights, ".q_w");
  const auto q_bias = TensorBySuffix(step, weights, ".q_b");
  const auto k_weight = TensorBySuffix(step, weights, ".k_w");
  const auto k_bias = TensorBySuffix(step, weights, ".k_b");
  const auto v_weight = TensorBySuffix(step, weights, ".v_w");
  const auto v_bias = TensorBySuffix(step, weights, ".v_b");
  const auto &q_binding = tape.RequireName(step.name + ".q");
  const auto &k_binding = tape.RequireName(step.name + ".k");
  const auto &v_binding = tape.RequireName(step.name + ".v");
  RequireTapeShape(q_binding, rows, attention.qkv_width, "query");
  RequireTapeShape(k_binding, rows, attention.qkv_width, "key");
  RequireTapeShape(v_binding, rows, attention.qkv_width, "value");

  CudaAttentionProjectionOutput output;
  output.query = tape.Reserve(workspace, q_binding);
  output.key = tape.Reserve(workspace, k_binding);
  output.value = tape.Reserve(workspace, v_binding);
  output.rows = rows;
  output.input_width = attention.input_width;
  output.qkv_width = attention.qkv_width;
  output.output_width = attention.output_width;
  output.heads = attention.heads;
  output.head_depth = attention.head_depth;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, q_weight.data, q_bias.data, output.query, rows,
                          attention.input_width, attention.qkv_width, stream);
  LaunchDenseAffineKernel(input, k_weight.data, k_bias.data, output.key, rows,
                          attention.input_width, attention.qkv_width, stream);
  LaunchDenseAffineKernel(input, v_weight.data, v_bias.data, output.value, rows,
                          attention.input_width, attention.qkv_width, stream);
  return output;
}

CudaAttentionProjectionOutput ExecuteAttentionOutputProjectionStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaWeightBuffers &weights,
    const float *context, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention output received empty batch");
  if (!context)
    throw std::runtime_error("CUDA attention output context is missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention output index is out of range");

  const auto &step = execution_plan.steps[attention_step_index];
  const auto attention = ResolveCudaAttentionStagePlan(
      execution_plan, attention_step_index,
      AttentionHeadCount(execution_plan, step.name));
  const int rows = batch_size * attention.squares;
  const auto dense_weight = TensorBySuffix(step, weights, ".dense_w");
  const auto dense_bias = TensorBySuffix(step, weights, ".dense_b");
  const auto &projection_binding = tape.RequireName(step.name + ".projection");
  RequireTapeShape(projection_binding, rows, attention.output_width,
                   "attention output projection");

  CudaAttentionProjectionOutput output;
  output.projection = tape.Reserve(workspace, projection_binding);
  output.rows = rows;
  output.input_width = attention.input_width;
  output.qkv_width = attention.qkv_width;
  output.output_width = attention.output_width;
  output.heads = attention.heads;
  output.head_depth = attention.head_depth;

  LaunchDenseAffineKernel(context, dense_weight.data, dense_bias.data,
                          output.projection, rows, attention.qkv_width,
                          attention.output_width, workspace.Stream());
  return output;
}

CudaSmolgenStageOutput ExecuteAttentionSmolgenStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index, const CudaAttentionStagePlan &attention,
    const CudaWeightBuffers &weights, const float *parent,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size) {
  if (!attention.smolgen.present)
    return {};
  if (!attention.smolgen.has_global_positional_weights) {
    throw std::runtime_error(
        "CUDA attention smolgen requires global positional weights");
  }
  if (!parent)
    throw std::runtime_error("CUDA attention smolgen parent is missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention smolgen index is out of range");

  const auto &attention_step = execution_plan.steps[attention_step_index];
  const auto *dense =
      FindStep(execution_plan, attention_step.name + ".smolgen.dense");
  const auto *norm =
      FindStep(execution_plan, attention_step.name + ".smolgen.norm");
  const auto *global_step =
      FindCudaGlobalPositionalEncodingStep(execution_plan);
  if (!dense || !norm || !global_step) {
    throw std::runtime_error("CUDA attention smolgen steps are incomplete");
  }

  const auto compress = TensorBySuffix(*dense, weights, ".compress");
  const auto dense1_weight = TensorBySuffix(*dense, weights, ".dense1_w");
  const auto dense1_bias = TensorBySuffix(*dense, weights, ".dense1_b");
  const auto dense2_weight = TensorBySuffix(*dense, weights, ".dense2_w");
  const auto dense2_bias = TensorBySuffix(*dense, weights, ".dense2_b");
  const auto ln1_gamma = TensorBySuffix(*norm, weights, ".ln1_gammas");
  const auto ln1_beta = TensorBySuffix(*norm, weights, ".ln1_betas");
  const auto ln2_gamma = TensorBySuffix(*norm, weights, ".ln2_gammas");
  const auto ln2_beta = TensorBySuffix(*norm, weights, ".ln2_betas");
  const auto global = TensorBySuffix(*global_step, weights, "body.smolgen_w");

  const int square_rows = batch_size * attention.squares;
  const int flattened_width =
      attention.squares * attention.smolgen.compressed_channels;
  const int global_rows = batch_size * attention.heads;
  const int global_width = attention.squares * attention.squares;

  const auto &compress_binding =
      tape.RequireName(attention.name + ".smolgen.compress");
  const auto &dense1_binding =
      tape.RequireName(attention.name + ".smolgen.dense1");
  const auto &activation1_binding =
      tape.RequireName(attention.name + ".smolgen.activation1");
  const auto &norm1_binding =
      tape.RequireName(attention.name + ".smolgen.norm1");
  const auto &dense2_binding =
      tape.RequireName(attention.name + ".smolgen.dense2");
  const auto &activation2_binding =
      tape.RequireName(attention.name + ".smolgen.activation2");
  const auto &norm2_binding =
      tape.RequireName(attention.name + ".smolgen.norm2");
  const auto &global_binding =
      tape.RequireName(attention.name + ".smolgen.global");
  RequireTapeShape(compress_binding, square_rows,
                   attention.smolgen.compressed_channels, "smolgen compress");
  RequireTapeShape(dense1_binding, batch_size, attention.smolgen.dense1_width,
                   "smolgen dense1");
  RequireTapeShape(activation1_binding, batch_size,
                   attention.smolgen.dense1_width, "smolgen activation1");
  RequireTapeShape(norm1_binding, batch_size, attention.smolgen.dense1_width,
                   "smolgen norm1");
  RequireTapeShape(dense2_binding, batch_size, attention.smolgen.dense2_width,
                   "smolgen dense2");
  RequireTapeShape(activation2_binding, batch_size,
                   attention.smolgen.dense2_width, "smolgen activation2");
  RequireTapeShape(norm2_binding, batch_size, attention.smolgen.dense2_width,
                   "smolgen norm2");
  RequireTapeShape(global_binding, global_rows, global_width, "smolgen global");

  CudaSmolgenStageOutput output;
  output.compressed = tape.Reserve(workspace, compress_binding);
  output.dense1 = tape.Reserve(workspace, dense1_binding);
  output.activation1 = tape.Reserve(workspace, activation1_binding);
  output.norm1 = tape.Reserve(workspace, norm1_binding);
  output.dense2 = tape.Reserve(workspace, dense2_binding);
  output.activation2 = tape.Reserve(workspace, activation2_binding);
  output.norm2 = tape.Reserve(workspace, norm2_binding);
  output.global_bias = tape.Reserve(workspace, global_binding);

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(parent, compress.data, nullptr, output.compressed,
                          square_rows, attention.input_width,
                          attention.smolgen.compressed_channels, stream);
  LaunchDenseAffineKernel(output.compressed, dense1_weight.data, nullptr,
                          output.dense1, batch_size, flattened_width,
                          attention.smolgen.dense1_width, stream);
  LaunchBiasActivationKernel(
      output.dense1, dense1_bias.data, output.activation1, batch_size,
      attention.smolgen.dense1_width,
      ActivationFromString(
          execution_plan.format.activations.smolgen_activation),
      stream);
  LaunchLayerNormKernel(output.activation1, ln1_gamma.data, ln1_beta.data,
                        output.norm1, batch_size,
                        attention.smolgen.dense1_width, 1e-3f, stream);
  LaunchDenseAffineKernel(
      output.norm1, dense2_weight.data, nullptr, output.dense2, batch_size,
      attention.smolgen.dense1_width, attention.smolgen.dense2_width, stream);
  LaunchBiasActivationKernel(
      output.dense2, dense2_bias.data, output.activation2, batch_size,
      attention.smolgen.dense2_width,
      ActivationFromString(
          execution_plan.format.activations.smolgen_activation),
      stream);
  LaunchLayerNormKernel(output.activation2, ln2_gamma.data, ln2_beta.data,
                        output.norm2, batch_size,
                        attention.smolgen.dense2_width, 1e-3f, stream);
  LaunchDenseAffineKernel(
      output.norm2, global.data, nullptr, output.global_bias, global_rows,
      attention.smolgen.dense2_width_per_head, global_width, stream);
  return output;
}

CudaDenseStageOutput ExecuteAttentionResidualLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &norm, const float *parent,
    const CudaAttentionProjectionOutput &attention_output,
    const CudaWeightBuffers &weights, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention layernorm received empty batch");
  if (!parent)
    throw std::runtime_error("CUDA attention layernorm parent is missing");
  if (!attention_output.projection)
    throw std::runtime_error("CUDA attention layernorm projection is missing");
  RequireLayerNormTensors(norm);

  const int rows = batch_size * kCudaAttentionSquares;
  const int width = attention_output.output_width;
  if (width <= 0 || attention_output.rows != rows ||
      attention_output.input_width != width) {
    throw std::runtime_error(
        "CUDA attention layernorm projection shape mismatch");
  }

  const auto gamma = weights.TensorAt(norm.tensors[0].inventory_index);
  const auto beta = weights.TensorAt(norm.tensors[1].inventory_index);
  if (gamma.dims.size() != 1 || beta.dims.size() != 1 ||
      gamma.elements != static_cast<std::size_t>(width) ||
      beta.elements != static_cast<std::size_t>(width)) {
    throw std::runtime_error(
        "CUDA attention layernorm tensor dimensions mismatch");
  }

  const auto &residual_binding =
      tape.RequireName(norm.name + ".attention_residual");
  const auto &norm_binding = tape.RequireName(norm.name + ".normalized");
  RequireTapeShape(residual_binding, rows, width, "attention residual");
  RequireTapeShape(norm_binding, rows, width, "attention layernorm");

  CudaDenseStageOutput output;
  output.residual = tape.Reserve(workspace, residual_binding);
  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;
  output.input_width = width;
  output.output_width = width;
  output.rows = rows;

  cudaStream_t stream = workspace.Stream();
  LaunchResidualLayerNormKernel(
      parent, attention_output.projection, gamma.data, beta.data,
      output.residual, output.normalized, rows, width,
      FeedForwardResidualScale(execution_plan, norm.name),
      AttentionLayerNormEpsilon(execution_plan, norm.name), stream);
  return output;
}

CudaAttentionCoreOutput ExecuteAttentionCoreStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::size_t attention_step_index,
    const CudaAttentionProjectionOutput &projections,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaWeightBuffers *weights, const float *parent) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA attention core received empty batch");
  if (!projections.query || !projections.key || !projections.value)
    throw std::runtime_error("CUDA attention core projections are missing");
  if (attention_step_index >= execution_plan.steps.size())
    throw std::runtime_error("CUDA attention core index is out of range");

  const auto &step = execution_plan.steps[attention_step_index];
  const auto attention = ResolveCudaAttentionStagePlan(
      execution_plan, attention_step_index,
      AttentionHeadCount(execution_plan, step.name));
  const int square_rows = batch_size * attention.squares;
  const int score_rows = batch_size * attention.heads * attention.squares;
  if (projections.rows != square_rows ||
      projections.qkv_width != attention.qkv_width ||
      projections.heads != attention.heads ||
      projections.head_depth != attention.head_depth) {
    throw std::runtime_error("CUDA attention core projection shape mismatch");
  }

  const auto &scores_binding = tape.RequireName(step.name + ".scores");
  const auto &probabilities_binding =
      tape.RequireName(step.name + ".probabilities");
  const auto &context_binding = tape.RequireName(step.name + ".context");
  RequireTapeShape(scores_binding, score_rows, attention.squares,
                   "attention scores");
  RequireTapeShape(probabilities_binding, score_rows, attention.squares,
                   "attention probabilities");
  RequireTapeShape(context_binding, square_rows, attention.qkv_width,
                   "attention context");

  CudaAttentionCoreOutput output;
  output.scores = tape.Reserve(workspace, scores_binding);
  output.probabilities = tape.Reserve(workspace, probabilities_binding);
  output.context = tape.Reserve(workspace, context_binding);
  output.score_rows = score_rows;
  output.score_width = attention.squares;
  output.rows = square_rows;
  output.qkv_width = attention.qkv_width;
  output.heads = attention.heads;
  output.head_depth = attention.head_depth;

  cudaStream_t stream = workspace.Stream();
  LaunchAttentionScoreKernel(
      projections.query, projections.key, output.scores, batch_size,
      attention.heads, attention.squares, attention.head_depth,
      attention.qkv_width,
      1.0f / std::sqrt(static_cast<float>(attention.head_depth)), stream);
  bool applied_bias_with_softmax = false;
  if (attention.smolgen.present) {
    if (!weights || !parent) {
      throw std::runtime_error(
          "CUDA attention smolgen requires weights and parent input");
    }
    const auto smolgen = ExecuteAttentionSmolgenStage(
        execution_plan, attention_step_index, attention, *weights, parent, tape,
        workspace, batch_size);
    output.attention_bias = smolgen.global_bias;
    output.smolgen_dense1 = smolgen.dense1;
    output.smolgen_norm1 = smolgen.norm1;
    output.smolgen_dense2 = smolgen.dense2;
    output.smolgen_activation2 = smolgen.activation2;
    output.smolgen_norm2 = smolgen.norm2;
    LaunchAttentionBiasSoftmaxKernel(output.scores, smolgen.global_bias,
                                     output.probabilities, score_rows,
                                     attention.squares, stream);
    applied_bias_with_softmax = true;
  }
  if (!applied_bias_with_softmax) {
    LaunchAttentionSoftmaxKernel(output.scores, output.probabilities,
                                 score_rows, attention.squares, stream);
  }
  LaunchAttentionContextKernel(output.probabilities, projections.value,
                               output.context, batch_size, attention.heads,
                               attention.squares, attention.head_depth,
                               attention.qkv_width, stream);
  return output;
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, CudaStageTimingCollector *timings) {
  const CudaStageInputBindings input_bindings;
  return ExecuteDenseActivationLayerNormSequence(execution_plan, weights, input,
                                                 tape, workspace, batch_size,
                                                 input_bindings, timings);
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    CudaStageTimingCollector *timings) {
  return ExecuteDenseActivationLayerNormSequence(
      execution_plan, weights, input, nullptr, nullptr, tape, workspace,
      batch_size, input_bindings, timings);
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    CudaStageTimingCollector *timings) {
  const auto schedule = CreateCudaExecutionSchedule(execution_plan);
  return ExecuteDenseActivationLayerNormSequence(
      execution_plan, weights, input, input_masks, input_values, tape,
      workspace, batch_size, input_bindings, schedule, timings);
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const std::uint64_t *input_masks, const float *input_values,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings,
    const CudaExecutionSchedule &schedule, CudaStageTimingCollector *timings) {
  if (!input)
    throw std::runtime_error("CUDA dense stage sequence input is missing");

  if (!schedule.FullySupported()) {
    throw std::runtime_error("CUDA dense stage sequence is unsupported: " +
                             schedule.Summary());
  }

  CudaDenseStageSequenceOutput sequence;
  const float *current_input = input;
  int current_width = 0;
  int current_rows = batch_size;
  for (const auto &entry : schedule.entries) {
    if (entry.kind != CudaExecutionScheduleKind::DenseLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::DenseActivationStage &&
        entry.kind != CudaExecutionScheduleKind::GateStage &&
        entry.kind != CudaExecutionScheduleKind::AttentionLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::FeedForwardStage &&
        entry.kind != CudaExecutionScheduleKind::FeedForwardLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::PolicyMapStage) {
      continue;
    }

    const auto &step = execution_plan.steps[entry.first_step];
    const float *stage_input = current_input;
    int stage_input_width = current_width;
    int stage_input_rows = current_rows;
    if (const std::string *source = input_bindings.FindSource(step.name)) {
      if (source->empty()) {
        stage_input = input;
        stage_input_width = 0;
        stage_input_rows = batch_size;
      } else {
        const CudaDenseStageOutput *source_stage = sequence.FindStage(*source);
        if (!source_stage || !source_stage->output) {
          throw std::runtime_error("CUDA stage input source is missing for " +
                                   step.name + ": " + *source);
        }
        stage_input = source_stage->output;
        stage_input_width = source_stage->output_width;
        stage_input_rows = source_stage->rows;
      }
    }
    MaybeFlattenSquareRowsForDenseStage(step, entry.kind, batch_size,
                                        stage_input_rows, stage_input_width);

    CudaDenseStageOutput stage;
    CudaStageTimer timer(timings, step.name, entry.kind, workspace.Stream());
    if (entry.kind == CudaExecutionScheduleKind::DenseActivationStage &&
        IsDynamicPositionPreprocessName(step.name)) {
      stage = ExecuteDynamicPositionEncodingStage(execution_plan, step, weights,
                                                  input_masks, input_values,
                                                  tape, workspace, batch_size);
    } else if (entry.kind == CudaExecutionScheduleKind::GateStage) {
      stage = ExecuteGateStage(step, weights, stage_input, stage_input_width,
                               tape, workspace, stage_input_rows);
    } else if (entry.kind ==
               CudaExecutionScheduleKind::AttentionLayerNormStage) {
      const auto input_projection = ExecuteAttentionInputProjectionStage(
          execution_plan, entry.first_step, weights, stage_input, tape,
          workspace, batch_size);
      const auto core = ExecuteAttentionCoreStage(
          execution_plan, entry.first_step, input_projection, tape, workspace,
          batch_size, &weights, stage_input);
      const auto output_projection = ExecuteAttentionOutputProjectionStage(
          execution_plan, entry.first_step, weights, core.context, tape,
          workspace, batch_size);
      stage = ExecuteAttentionResidualLayerNormStage(
          execution_plan, execution_plan.steps[entry.second_step], stage_input,
          output_projection, weights, tape, workspace, batch_size);
    } else if (entry.kind ==
               CudaExecutionScheduleKind::FeedForwardLayerNormStage) {
      stage = ExecuteFeedForwardLayerNormStage(
          execution_plan, step, execution_plan.steps[entry.second_step],
          weights, stage_input, tape, workspace, stage_input_rows);
    } else if (entry.kind == CudaExecutionScheduleKind::FeedForwardStage) {
      stage =
          ExecuteFeedForwardStage(execution_plan, step, weights, stage_input,
                                  tape, workspace, stage_input_rows);
    } else if (entry.kind == CudaExecutionScheduleKind::PolicyMapStage) {
      stage = ExecuteAttentionPolicyMapStage(
          execution_plan, step, weights, sequence, tape, workspace, batch_size);
    } else if (entry.kind == CudaExecutionScheduleKind::DenseLayerNormStage) {
      stage = ExecuteDenseActivationLayerNormStage(
          execution_plan, step, execution_plan.steps[entry.second_step],
          weights, stage_input, tape, workspace, stage_input_rows);
    } else {
      stage = ExecuteDenseActivationStage(execution_plan, step, weights,
                                          stage_input, tape, workspace,
                                          stage_input_rows);
    }
    timer.Stop();
    if (stage_input_width != 0 && stage.input_width != stage_input_width) {
      throw std::runtime_error(
          "CUDA dense stage sequence input width mismatch");
    }

    sequence.last = stage;
    sequence.stages.push_back({step.name, stage});
    ++sequence.stage_count;
    current_input = stage.output;
    current_width = stage.output_width;
    current_rows = stage.rows;
  }

  if (sequence.stage_count == 0)
    throw std::runtime_error("CUDA dense stage sequence found no stages");
  return sequence;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

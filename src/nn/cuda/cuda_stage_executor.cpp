/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_stage_executor.h"

#include "cuda_execution_schedule.h"
#include "cuda_kernels.h"
#include "cuda_plan_analysis.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

void RequireDenseTensors(const NetworkResolvedExecutionStep &dense) {
  if (dense.kind != NetworkExecutionOpKind::Dense ||
      dense.tensors.size() < 2) {
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

bool IsMultiplyGate(std::string_view name) {
  return name.find("mult_gate") != std::string_view::npos;
}

bool IsAddGate(std::string_view name) {
  return name.find("add_gate") != std::string_view::npos;
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

CudaFeedForwardTensors ResolveFeedForwardTensors(
    const NetworkResolvedExecutionStep &ffn, const CudaWeightBuffers &weights) {
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

std::size_t BodyEncoderLayerCount(
    const NetworkResolvedExecutionPlan &execution_plan) {
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
    status = cudaMemcpy2D(
        dst, static_cast<std::size_t>(dst_stride) * sizeof(float), src,
        static_cast<std::size_t>(src_stride) * sizeof(float),
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

CudaStageInputBindings CreateCudaStageInputBindings(
    const NetworkResolvedExecutionPlan &execution_plan,
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
  return bindings;
}

CudaDenseStageOutput ExecuteDenseActivationStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
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
  const auto &activation_binding =
      tape.RequireName(dense.name + ".activation");
  const std::size_t entries = static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(output_width);
  if (dense_binding.entries != entries ||
      activation_binding.entries != entries) {
    throw std::runtime_error("CUDA dense stage tape size mismatch");
  }

  CudaDenseStageOutput output;
  output.dense = tape.Reserve(workspace, dense_binding);
  output.activation = tape.Reserve(workspace, activation_binding);
  output.output = output.activation;
  output.input_width = input_width;
  output.output_width = output_width;

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, dense_weight.data, dense_bias.data,
                          output.dense, batch_size, input_width, output_width,
                          stream);
  LaunchActivationKernel(
      output.dense, output.activation, static_cast<int>(entries),
      ActivationFromString(execution_plan.format.activations.ffn_activation),
      stream);
  return output;
}

CudaDenseStageOutput ExecuteDenseActivationLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &dense,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  RequireLayerNormTensors(norm);

  CudaDenseStageOutput output = ExecuteDenseActivationStage(
      execution_plan, dense, weights, input, tape, workspace, batch_size);
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
  const std::size_t entries = static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(output.output_width);
  if (norm_binding.entries != entries) {
    throw std::runtime_error("CUDA layernorm stage tape size mismatch");
  }

  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;

  cudaStream_t stream = workspace.Stream();
  LaunchLayerNormKernel(output.activation, gamma.data, beta.data,
                        output.normalized, batch_size, output.output_width,
                        1e-5f, stream);
  return output;
}

CudaDenseStageOutput ExecuteGateStage(
    const NetworkResolvedExecutionStep &gate, const CudaWeightBuffers &weights,
    const float *input, int input_width, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA gate stage received empty batch");
  if (!input || input_width <= 0)
    throw std::runtime_error("CUDA gate stage input is missing");
  RequireGateTensors(gate);

  const auto &gate_binding = tape.RequireName(gate.name + ".gated");
  const std::size_t entries = static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(input_width);
  if (gate_binding.entries != entries) {
    throw std::runtime_error("CUDA gate stage tape size mismatch");
  }

  CudaDenseStageOutput output;
  output.gated = tape.Reserve(workspace, gate_binding);
  output.output = output.gated;
  output.input_width = input_width;
  output.output_width = input_width;

  const float *source = input;
  cudaStream_t stream = workspace.Stream();
  bool launched = false;
  for (const auto &tensor : gate.tensors) {
    const auto gate_weights = weights.TensorAt(tensor.inventory_index);
    if (gate_weights.elements != static_cast<std::size_t>(input_width)) {
      throw std::runtime_error("CUDA gate stage tensor dimensions mismatch");
    }
    if (IsMultiplyGate(tensor.name)) {
      LaunchGateKernel(source, gate_weights.data, output.gated, batch_size,
                       input_width, CudaGateKind::Multiply, stream);
    } else if (IsAddGate(tensor.name)) {
      LaunchGateKernel(source, gate_weights.data, output.gated, batch_size,
                       input_width, CudaGateKind::Add, stream);
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

CudaDenseStageOutput ExecuteFeedForwardStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &ffn, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA feed-forward stage received empty batch");
  if (!input)
    throw std::runtime_error("CUDA feed-forward stage input is missing");
  const auto tensors = ResolveFeedForwardTensors(ffn, weights);

  const auto &dense1_binding = tape.RequireName(ffn.name + ".dense1");
  const auto &activation_binding =
      tape.RequireName(ffn.name + ".activation");
  const auto &dense2_binding = tape.RequireName(ffn.name + ".dense2");
  const std::size_t hidden_entries = static_cast<std::size_t>(batch_size) *
                                     static_cast<std::size_t>(
                                         tensors.hidden_width);
  const std::size_t output_entries = static_cast<std::size_t>(batch_size) *
                                     static_cast<std::size_t>(
                                         tensors.output_width);
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

  cudaStream_t stream = workspace.Stream();
  LaunchDenseAffineKernel(input, tensors.dense1_weight.data,
                          tensors.dense1_bias.data, output.dense, batch_size,
                          tensors.input_width, tensors.hidden_width, stream);
  LaunchActivationKernel(
      output.dense, output.activation, static_cast<int>(hidden_entries),
      ActivationFromString(execution_plan.format.activations.ffn_activation),
      stream);
  LaunchDenseAffineKernel(output.activation, tensors.dense2_weight.data,
                          tensors.dense2_bias.data, output.feed_forward,
                          batch_size, tensors.hidden_width,
                          tensors.output_width, stream);
  return output;
}

CudaDenseStageOutput ExecuteFeedForwardLayerNormStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    const NetworkResolvedExecutionStep &ffn,
    const NetworkResolvedExecutionStep &norm, const CudaWeightBuffers &weights,
    const float *input, const CudaExecutionTape &tape,
    CudaExecutionWorkspace &workspace, int batch_size) {
  RequireLayerNormTensors(norm);

  CudaDenseStageOutput output = ExecuteFeedForwardStage(
      execution_plan, ffn, weights, input, tape, workspace, batch_size);
  if (output.input_width != output.output_width) {
    throw std::runtime_error(
        "CUDA feed-forward residual width mismatch");
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
  const std::size_t entries = static_cast<std::size_t>(batch_size) *
                              static_cast<std::size_t>(output.output_width);
  if (residual_binding.entries != entries || norm_binding.entries != entries) {
    throw std::runtime_error(
        "CUDA feed-forward layernorm tape size mismatch");
  }

  output.residual = tape.Reserve(workspace, residual_binding);
  output.normalized = tape.Reserve(workspace, norm_binding);
  output.output = output.normalized;

  cudaStream_t stream = workspace.Stream();
  LaunchResidualAddKernel(input, output.feed_forward, output.residual,
                          batch_size, output.output_width,
                          FeedForwardResidualScale(execution_plan, ffn.name),
                          stream);
  LaunchLayerNormKernel(output.residual, gamma.data, beta.data,
                        output.normalized, batch_size, output.output_width,
                        FeedForwardLayerNormEpsilon(execution_plan, ffn.name),
                        stream);
  return output;
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size) {
  const CudaStageInputBindings input_bindings;
  return ExecuteDenseActivationLayerNormSequence(
      execution_plan, weights, input, tape, workspace, batch_size,
      input_bindings);
}

CudaDenseStageSequenceOutput ExecuteDenseActivationLayerNormSequence(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaWeightBuffers &weights, const float *input,
    const CudaExecutionTape &tape, CudaExecutionWorkspace &workspace,
    int batch_size, const CudaStageInputBindings &input_bindings) {
  if (!input)
    throw std::runtime_error("CUDA dense stage sequence input is missing");

  const auto schedule = CreateCudaExecutionSchedule(execution_plan);
  if (!schedule.FullySupported()) {
    throw std::runtime_error("CUDA dense stage sequence is unsupported: " +
                             schedule.Summary());
  }

  CudaDenseStageSequenceOutput sequence;
  const float *current_input = input;
  int current_width = 0;
  for (const auto &entry : schedule.entries) {
    if (entry.kind != CudaExecutionScheduleKind::DenseLayerNormStage &&
        entry.kind != CudaExecutionScheduleKind::DenseActivationStage &&
        entry.kind != CudaExecutionScheduleKind::GateStage &&
        entry.kind != CudaExecutionScheduleKind::FeedForwardStage &&
        entry.kind !=
            CudaExecutionScheduleKind::FeedForwardLayerNormStage) {
      continue;
    }

    const auto &step = execution_plan.steps[entry.first_step];
    const float *stage_input = current_input;
    int stage_input_width = current_width;
    if (const std::string *source = input_bindings.FindSource(step.name)) {
      if (source->empty()) {
        stage_input = input;
        stage_input_width = 0;
      } else {
        const CudaDenseStageOutput *source_stage =
            sequence.FindStage(*source);
        if (!source_stage || !source_stage->output) {
          throw std::runtime_error("CUDA stage input source is missing for " +
                                   step.name + ": " + *source);
        }
        stage_input = source_stage->output;
        stage_input_width = source_stage->output_width;
      }
    }

    CudaDenseStageOutput stage;
    if (entry.kind == CudaExecutionScheduleKind::GateStage) {
      stage = ExecuteGateStage(step, weights, stage_input, stage_input_width,
                               tape, workspace, batch_size);
    } else if (entry.kind ==
               CudaExecutionScheduleKind::FeedForwardLayerNormStage) {
      stage = ExecuteFeedForwardLayerNormStage(
          execution_plan, step, execution_plan.steps[entry.second_step],
          weights, stage_input, tape, workspace, batch_size);
    } else if (entry.kind == CudaExecutionScheduleKind::FeedForwardStage) {
      stage = ExecuteFeedForwardStage(execution_plan, step, weights,
                                      stage_input, tape, workspace,
                                      batch_size);
    } else if (entry.kind == CudaExecutionScheduleKind::DenseLayerNormStage) {
      stage = ExecuteDenseActivationLayerNormStage(
          execution_plan, step, execution_plan.steps[entry.second_step],
          weights, stage_input, tape, workspace, batch_size);
    } else {
      stage = ExecuteDenseActivationStage(execution_plan, step, weights,
                                          stage_input, tape, workspace,
                                          batch_size);
    }
    if (stage_input_width != 0 && stage.input_width != stage_input_width) {
      throw std::runtime_error(
          "CUDA dense stage sequence input width mismatch");
    }

    sequence.last = stage;
    sequence.stages.push_back({step.name, stage});
    ++sequence.stage_count;
    current_input = stage.output;
    current_width = stage.output_width;
  }

  if (sequence.stage_count == 0)
    throw std::runtime_error("CUDA dense stage sequence found no stages");
  return sequence;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

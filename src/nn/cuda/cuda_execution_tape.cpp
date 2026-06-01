/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_execution_tape.h"

#include "cuda_attention_plan.h"
#include "cuda_runtime_probe.h"
#include "cuda_workspace.h"
#include "cuda_workspace_smoke.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

const NetworkResolvedExecutionStep &
FindStep(const NetworkResolvedExecutionPlan &execution_plan,
         NetworkExecutionOpKind kind) {
  for (const auto &step : execution_plan.steps) {
    if (step.kind == kind)
      return step;
  }
  throw std::runtime_error("CUDA execution tape is missing required step");
}

int DenseOutputWidth(const NetworkResolvedExecutionStep &dense) {
  return NetworkDenseStageOutputWidth(dense);
}

int ConvolutionOutputChannels(const NetworkResolvedExecutionStep &convolution) {
  return NetworkConvolutionStageOutputChannels(convolution);
}

int LayerNormWidth(const NetworkResolvedExecutionStep &norm) {
  return NetworkLayerNormStageWidth(norm);
}

int GateWidth(const NetworkResolvedExecutionStep &gate) {
  return NetworkGateStageWidth(gate);
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool EndsWith(std::string_view value, std::string_view suffix) {
  return value.ends_with(suffix);
}

bool IsSmolgenDenseName(std::string_view name) {
  return EndsWith(name, ".smolgen.dense");
}

bool IsSmolgenNormName(std::string_view name) {
  return EndsWith(name, ".smolgen.norm");
}

std::string ResidualBlockName(std::string_view name) {
  if (EndsWith(name, ".conv1"))
    return std::string(name.substr(0, name.size() - 6));
  if (EndsWith(name, ".conv2"))
    return std::string(name.substr(0, name.size() - 6));
  return {};
}

bool IsResidualConv1Name(std::string_view name) {
  return StartsWith(name, "body.residual.") && EndsWith(name, ".conv1");
}

bool IsResidualConv2For(const NetworkResolvedExecutionStep &step,
                        std::string_view block_name) {
  return step.kind == NetworkExecutionOpKind::Convolution &&
         step.name == std::string(block_name) + ".conv2";
}

bool IsResidualSqueezeExciteFor(const NetworkResolvedExecutionStep &step,
                                std::string_view block_name) {
  return step.kind == NetworkExecutionOpKind::Dense &&
         step.name == std::string(block_name) + ".se";
}

bool IsDynamicPositionPreprocessName(std::string_view name) {
  return name == "body.input_embedding_preprocess";
}

bool IsStaticPositionEmbeddingName(const NetworkResolvedExecutionPlan &plan,
                                   std::string_view name) {
  return plan.format.input_embedding == INPUT_EMBEDDING_PE_MAP &&
         name == "body.input_embedding";
}

} // namespace

std::string CudaExecutionBufferRoleName(CudaExecutionBufferRole role) {
  switch (role) {
  case CudaExecutionBufferRole::DenseOutput:
    return "dense_output";
  case CudaExecutionBufferRole::ConvolutionOutput:
    return "convolution_output";
  case CudaExecutionBufferRole::ActivationOutput:
    return "activation_output";
  case CudaExecutionBufferRole::NormalizedOutput:
    return "normalized_output";
  case CudaExecutionBufferRole::GateOutput:
    return "gate_output";
  case CudaExecutionBufferRole::FeedForwardHiddenOutput:
    return "feed_forward_hidden_output";
  case CudaExecutionBufferRole::FeedForwardOutput:
    return "feed_forward_output";
  case CudaExecutionBufferRole::ResidualOutput:
    return "residual_output";
  case CudaExecutionBufferRole::SqueezeExcitePoolOutput:
    return "squeeze_excite_pool_output";
  case CudaExecutionBufferRole::SqueezeExciteHiddenOutput:
    return "squeeze_excite_hidden_output";
  case CudaExecutionBufferRole::SqueezeExciteOutput:
    return "squeeze_excite_output";
  case CudaExecutionBufferRole::AttentionQuery:
    return "attention_query";
  case CudaExecutionBufferRole::AttentionKey:
    return "attention_key";
  case CudaExecutionBufferRole::AttentionValue:
    return "attention_value";
  case CudaExecutionBufferRole::AttentionScores:
    return "attention_scores";
  case CudaExecutionBufferRole::AttentionProbabilities:
    return "attention_probabilities";
  case CudaExecutionBufferRole::AttentionContext:
    return "attention_context";
  case CudaExecutionBufferRole::AttentionOutputProjection:
    return "attention_output_projection";
  case CudaExecutionBufferRole::AttentionSmolgenBias:
    return "attention_smolgen_bias";
  case CudaExecutionBufferRole::AttentionResidualOutput:
    return "attention_residual_output";
  case CudaExecutionBufferRole::InputPlaneExpanded:
    return "input_plane_expanded";
  case CudaExecutionBufferRole::StaticPositionOutput:
    return "static_position_output";
  case CudaExecutionBufferRole::DynamicPositionInput:
    return "dynamic_position_input";
  case CudaExecutionBufferRole::DynamicPositionOutput:
    return "dynamic_position_output";
  case CudaExecutionBufferRole::PolicyMapRawOutput:
    return "policy_map_raw_output";
  case CudaExecutionBufferRole::PolicyMapOutput:
    return "policy_map_output";
  }
  return "unknown";
}

const CudaExecutionBufferBinding *
CudaExecutionTape::FindName(std::string_view name) const {
  for (const auto &binding : bindings_) {
    if (binding.name == name)
      return &binding;
  }
  return nullptr;
}

const CudaExecutionBufferBinding &
CudaExecutionTape::RequireName(std::string_view name) const {
  const CudaExecutionBufferBinding *binding = FindName(name);
  if (!binding)
    throw std::runtime_error("CUDA execution tape is missing buffer: " +
                             std::string(name));
  return *binding;
}

const CudaExecutionBufferBinding &
CudaExecutionTape::RequireRole(CudaExecutionBufferRole role) const {
  for (const auto &binding : bindings_) {
    if (binding.role == role)
      return binding;
  }
  throw std::runtime_error("CUDA execution tape is missing buffer role: " +
                           CudaExecutionBufferRoleName(role));
}

float *
CudaExecutionTape::Reserve(CudaExecutionWorkspace &workspace,
                           const CudaExecutionBufferBinding &binding) const {
  return workspace.ReserveNamedFloats(binding.name, binding.entries);
}

std::size_t CudaExecutionTape::CountRole(CudaExecutionBufferRole role) const {
  std::size_t count = 0;
  for (const auto &binding : bindings_) {
    if (binding.role == role)
      ++count;
  }
  return count;
}

std::size_t CudaExecutionTape::TotalEntries() const {
  std::size_t total = 0;
  for (const auto &binding : bindings_)
    total += binding.entries;
  return total;
}

std::string CudaExecutionTape::Summary() const {
  std::ostringstream out;
  out << bindings_.size() << " bindings, " << TotalEntries()
      << " intermediate floats";
  return out.str();
}

void CudaExecutionTape::Add(std::string name, CudaExecutionBufferRole role,
                            int rows, int width) {
  if (name.empty() || rows <= 0 || width <= 0)
    throw std::runtime_error("CUDA execution tape binding is invalid");
  if (FindName(name))
    throw std::runtime_error("CUDA execution tape duplicate binding: " + name);
  bindings_.push_back(CudaExecutionBufferBinding{
      std::move(name), role, static_cast<std::size_t>(rows) * width, rows,
      width});
}

CudaExecutionTape
CreateResolvedExecutionTape(const NetworkResolvedExecutionPlan &plan,
                            int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA execution tape batch size is invalid");

  CudaExecutionTape tape;
  int current_rows = batch_size;
  int current_width = 0;
  for (std::size_t step_index = 0; step_index < plan.steps.size();
       ++step_index) {
    const auto &step = plan.steps[step_index];
    switch (step.kind) {
    case NetworkExecutionOpKind::Convolution: {
      if (IsResidualConv1Name(step.name)) {
        const std::string block_name = ResidualBlockName(step.name);
        const std::size_t conv2_index = step_index + 1;
        const std::size_t se_index = conv2_index + 1;
        if (conv2_index < plan.steps.size() &&
            IsResidualConv2For(plan.steps[conv2_index], block_name)) {
          const int conv1_channels = ConvolutionOutputChannels(step);
          const int conv2_channels =
              ConvolutionOutputChannels(plan.steps[conv2_index]);
          tape.Add(step.name + ".convolution",
                   CudaExecutionBufferRole::ConvolutionOutput,
                   batch_size * conv1_channels, kCudaAttentionSquares);
          tape.Add(plan.steps[conv2_index].name + ".convolution",
                   CudaExecutionBufferRole::ConvolutionOutput,
                   batch_size * conv2_channels, kCudaAttentionSquares);
          tape.Add(block_name + ".residual",
                   CudaExecutionBufferRole::ResidualOutput,
                   batch_size * conv2_channels, kCudaAttentionSquares);
          if (se_index < plan.steps.size() &&
              IsResidualSqueezeExciteFor(plan.steps[se_index], block_name)) {
            const auto &se = plan.steps[se_index];
            const auto se_widths = NetworkSqueezeExciteStageWidthsFor(se);
            tape.Add(se.name + ".pool",
                     CudaExecutionBufferRole::SqueezeExcitePoolOutput,
                     batch_size, conv2_channels);
            tape.Add(se.name + ".fc1.dense",
                     CudaExecutionBufferRole::SqueezeExciteHiddenOutput,
                     batch_size, se_widths.hidden);
            tape.Add(se.name + ".fc1.activation",
                     CudaExecutionBufferRole::SqueezeExciteHiddenOutput,
                     batch_size, se_widths.hidden);
            tape.Add(se.name + ".fc2.dense",
                     CudaExecutionBufferRole::SqueezeExciteOutput, batch_size,
                     se_widths.output);
            step_index = se_index;
          } else {
            step_index = conv2_index;
          }
          tape.Add(block_name + ".activation",
                   CudaExecutionBufferRole::ActivationOutput,
                   batch_size * conv2_channels, kCudaAttentionSquares);
          current_rows = batch_size * conv2_channels;
          current_width = kCudaAttentionSquares;
          break;
        }
      }
      const int width = ConvolutionOutputChannels(step);
      if (step.name == "body.input") {
        tape.Add(step.name + ".expanded",
                 CudaExecutionBufferRole::InputPlaneExpanded,
                 batch_size * plan.tensors.input_planes,
                 plan.tensors.input_squares);
      }
      tape.Add(step.name + ".convolution",
               CudaExecutionBufferRole::ConvolutionOutput, batch_size * width,
               kCudaAttentionSquares);
      current_rows = batch_size * width;
      current_width = kCudaAttentionSquares;
      break;
    }
    case NetworkExecutionOpKind::Dense: {
      if (IsSmolgenDenseName(step.name))
        break;
      const int width = DenseOutputWidth(step);
      if (IsStaticPositionEmbeddingName(plan, step.name)) {
        const auto geometry = ResolveStaticPositionEncodingGeometry(plan, step);
        tape.Add(step.name + ".static_pe_concat",
                 CudaExecutionBufferRole::StaticPositionOutput,
                 batch_size * geometry.input_squares, geometry.concat_width);
      }
      if (IsDynamicPositionPreprocessName(step.name)) {
        const auto geometry =
            ResolveDynamicPositionEncodingGeometry(plan, step);
        tape.Add(step.name + ".expanded",
                 CudaExecutionBufferRole::InputPlaneExpanded,
                 batch_size * geometry.input_squares, geometry.input_planes);
        tape.Add(step.name + ".position_input",
                 CudaExecutionBufferRole::DynamicPositionInput, batch_size,
                 geometry.dense_input_width);
        tape.Add(step.name + ".dense", CudaExecutionBufferRole::DenseOutput,
                 batch_size, geometry.dense_output_width);
        tape.Add(step.name + ".concat",
                 CudaExecutionBufferRole::DynamicPositionOutput,
                 batch_size * geometry.input_squares, geometry.concat_width);
        current_rows = batch_size * geometry.input_squares;
        current_width = geometry.concat_width;
        break;
      }
      const int rows = NetworkDenseLikeRows(plan, step.name, batch_size);
      tape.Add(step.name + ".dense", CudaExecutionBufferRole::DenseOutput, rows,
               width);
      tape.Add(step.name + ".activation",
               CudaExecutionBufferRole::ActivationOutput, rows, width);
      current_rows = rows;
      current_width = width;
      break;
    }
    case NetworkExecutionOpKind::LayerNorm: {
      if (IsSmolgenNormName(step.name))
        break;
      const int width = LayerNormWidth(step);
      const int rows = NetworkDenseLikeRows(plan, step.name, batch_size);
      tape.Add(step.name + ".normalized",
               CudaExecutionBufferRole::NormalizedOutput, rows, width);
      current_rows = rows;
      current_width = width;
      break;
    }
    case NetworkExecutionOpKind::Gate: {
      int width = GateWidth(step);
      if (current_width > 0 && !step.tensors.empty() &&
          step.tensors[0].elements % static_cast<std::size_t>(current_width) ==
              0) {
        width = current_width;
      }
      const int rows = current_width > 0
                           ? current_rows
                           : NetworkDenseLikeRows(plan, step.name, batch_size);
      tape.Add(step.name + ".gated", CudaExecutionBufferRole::GateOutput, rows,
               width);
      current_rows = rows;
      current_width = width;
      break;
    }
    case NetworkExecutionOpKind::FeedForward: {
      const auto widths = NetworkFeedForwardStageWidthsFor(step);
      const int rows = NetworkDenseLikeRows(plan, step.name, batch_size);
      tape.Add(step.name + ".dense1",
               CudaExecutionBufferRole::FeedForwardHiddenOutput, rows,
               widths.hidden);
      tape.Add(step.name + ".activation",
               CudaExecutionBufferRole::ActivationOutput, rows, widths.hidden);
      tape.Add(step.name + ".dense2",
               CudaExecutionBufferRole::FeedForwardOutput, rows, widths.output);
      current_rows = rows;
      current_width = widths.output;
      break;
    }
    case NetworkExecutionOpKind::Attention: {
      const int head_count = NetworkAttentionHeadCount(plan, step.name);
      const auto attention =
          ResolveCudaAttentionStagePlan(plan, step_index, head_count);
      const int square_rows = batch_size * attention.squares;
      tape.Add(step.name + ".q", CudaExecutionBufferRole::AttentionQuery,
               square_rows, attention.qkv_width);
      tape.Add(step.name + ".k", CudaExecutionBufferRole::AttentionKey,
               square_rows, attention.qkv_width);
      tape.Add(step.name + ".v", CudaExecutionBufferRole::AttentionValue,
               square_rows, attention.qkv_width);
      tape.Add(step.name + ".scores", CudaExecutionBufferRole::AttentionScores,
               batch_size * attention.heads * attention.squares,
               attention.squares);
      tape.Add(step.name + ".probabilities",
               CudaExecutionBufferRole::AttentionProbabilities,
               batch_size * attention.heads * attention.squares,
               attention.squares);
      tape.Add(step.name + ".context",
               CudaExecutionBufferRole::AttentionContext, square_rows,
               attention.qkv_width);
      tape.Add(step.name + ".projection",
               CudaExecutionBufferRole::AttentionOutputProjection, square_rows,
               attention.output_width);
      if (attention.smolgen.present) {
        tape.Add(step.name + ".smolgen.compress",
                 CudaExecutionBufferRole::DenseOutput, square_rows,
                 attention.smolgen.compressed_channels);
        tape.Add(step.name + ".smolgen.dense1",
                 CudaExecutionBufferRole::DenseOutput, batch_size,
                 attention.smolgen.dense1_width);
        tape.Add(step.name + ".smolgen.activation1",
                 CudaExecutionBufferRole::ActivationOutput, batch_size,
                 attention.smolgen.dense1_width);
        tape.Add(step.name + ".smolgen.norm1",
                 CudaExecutionBufferRole::NormalizedOutput, batch_size,
                 attention.smolgen.dense1_width);
        tape.Add(step.name + ".smolgen.dense2",
                 CudaExecutionBufferRole::DenseOutput, batch_size,
                 attention.smolgen.dense2_width);
        tape.Add(step.name + ".smolgen.activation2",
                 CudaExecutionBufferRole::ActivationOutput, batch_size,
                 attention.smolgen.dense2_width);
        tape.Add(step.name + ".smolgen.norm2",
                 CudaExecutionBufferRole::NormalizedOutput, batch_size,
                 attention.smolgen.dense2_width);
        tape.Add(step.name + ".smolgen.global",
                 CudaExecutionBufferRole::AttentionSmolgenBias,
                 batch_size * attention.heads,
                 attention.squares * attention.squares);
      }
      current_rows = square_rows;
      current_width = attention.output_width;
      break;
    }
    case NetworkExecutionOpKind::PolicyMap: {
      if (!plan.format.attention_policy && !plan.format.conv_policy)
        break;
      if (plan.format.attention_policy) {
        tape.Add(step.name + ".raw",
                 CudaExecutionBufferRole::PolicyMapRawOutput, batch_size,
                 kNetworkAttentionPolicyScratch);
      }
      tape.Add(step.name + ".mapped", CudaExecutionBufferRole::PolicyMapOutput,
               batch_size, kNetworkPolicyOutputs);
      break;
    }
    default:
      break;
    }
  }

  for (std::size_t i = 1; i < plan.steps.size(); ++i) {
    const auto &step = plan.steps[i];
    if (step.kind != NetworkExecutionOpKind::LayerNorm ||
        plan.steps[i - 1].kind != NetworkExecutionOpKind::FeedForward) {
      continue;
    }
    const int width = LayerNormWidth(step);
    const int rows = NetworkDenseLikeRows(plan, step.name, batch_size);
    tape.Add(step.name + ".residual", CudaExecutionBufferRole::ResidualOutput,
             rows, width);
  }
  for (std::size_t i = 1; i < plan.steps.size(); ++i) {
    const auto &step = plan.steps[i];
    if (step.kind != NetworkExecutionOpKind::LayerNorm ||
        !NetworkIsAttentionLayerNormStage(plan, step.name)) {
      continue;
    }
    const int width = LayerNormWidth(step);
    tape.Add(step.name + ".attention_residual",
             CudaExecutionBufferRole::AttentionResidualOutput,
             NetworkDenseLikeRows(plan, step.name, batch_size), width);
  }
  return tape;
}

CudaExecutionTape
CreatePlanSmokeExecutionTape(const NetworkTensorPlan &tensor_plan,
                             const NetworkResolvedExecutionPlan &plan,
                             int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA execution tape batch size is invalid");

  const auto &dense = FindStep(plan, NetworkExecutionOpKind::Dense);
  const auto &norm = FindStep(plan, NetworkExecutionOpKind::LayerNorm);
  const int output_width = DenseOutputWidth(dense);
  if (output_width > tensor_plan.policy_outputs ||
      output_width > tensor_plan.raw_policy_outputs) {
    throw std::runtime_error("CUDA execution tape output width exceeds head");
  }

  return CreateResolvedExecutionTape(plan, batch_size);
}

CudaWorkspaceSmokeResult RunExecutionTapeSmoke() {
  CudaWorkspaceSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  try {
    NetworkFormatDescriptor format;
    format.wdl = true;
    format.moves_left = true;
    format.attention_policy = true;
    const auto tensor_plan = CreateNetworkTensorPlan(format);

    NetworkResolvedExecutionPlan plan;
    plan.tensors = tensor_plan;
    NetworkResolvedExecutionStep dense;
    dense.kind = NetworkExecutionOpKind::Dense;
    dense.name = "smoke.dense";
    NetworkResolvedTensorRef dense_weight;
    dense_weight.name = "smoke.dense_w";
    dense_weight.elements = 6;
    dense_weight.dims = {2, 3};
    dense.tensors.push_back(dense_weight);

    NetworkResolvedExecutionStep norm;
    norm.kind = NetworkExecutionOpKind::LayerNorm;
    norm.name = "smoke.norm";
    NetworkResolvedTensorRef gamma;
    gamma.name = "smoke.gamma";
    gamma.elements = 2;
    gamma.dims = {2};
    norm.tensors.push_back(gamma);

    plan.steps.push_back(dense);
    plan.steps.push_back(norm);

    const auto tape = CreatePlanSmokeExecutionTape(tensor_plan, plan, 3);
    if (tape.BindingCount() != 3 || tape.TotalEntries() != 18 ||
        tape.RequireRole(CudaExecutionBufferRole::DenseOutput).width != 2 ||
        tape.RequireRole(CudaExecutionBufferRole::NormalizedOutput).rows != 3) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA execution tape binding mismatch";
      return result;
    }

    CudaExecutionWorkspace workspace;
    for (const auto &binding : tape.Bindings()) {
      if (!tape.Reserve(workspace, binding)) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA execution tape reserve returned null";
        return result;
      }
    }
    result.allocation_bytes = workspace.TotalBytes();
    if (workspace.NamedBufferCount() != 3 ||
        workspace.TotalCapacityFloats() != tape.TotalEntries()) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA execution tape workspace mismatch";
      return result;
    }

    NetworkResolvedExecutionStep dense2;
    dense2.kind = NetworkExecutionOpKind::Dense;
    dense2.name = "smoke.dense2";
    NetworkResolvedTensorRef dense2_weight;
    dense2_weight.name = "smoke.dense2_w";
    dense2_weight.elements = 10;
    dense2_weight.dims = {5, 2};
    dense2.tensors.push_back(dense2_weight);

    NetworkResolvedExecutionStep norm2;
    norm2.kind = NetworkExecutionOpKind::LayerNorm;
    norm2.name = "smoke.norm2";
    NetworkResolvedTensorRef gamma2;
    gamma2.name = "smoke.gamma2";
    gamma2.elements = 5;
    gamma2.dims = {5};
    norm2.tensors.push_back(gamma2);

    NetworkResolvedExecutionPlan stacked = plan;
    stacked.steps.push_back(dense2);
    stacked.steps.push_back(norm2);
    const auto stacked_tape = CreateResolvedExecutionTape(stacked, 2);
    if (stacked_tape.BindingCount() != 6 ||
        stacked_tape.CountRole(CudaExecutionBufferRole::DenseOutput) != 2 ||
        stacked_tape.CountRole(CudaExecutionBufferRole::ActivationOutput) !=
            2 ||
        stacked_tape.CountRole(CudaExecutionBufferRole::NormalizedOutput) !=
            2 ||
        stacked_tape.RequireName("smoke.dense2.activation").width != 5 ||
        stacked_tape.TotalEntries() != 42) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA resolved execution tape sequence mismatch";
      return result;
    }

    CudaExecutionWorkspace stacked_workspace;
    for (const auto &binding : stacked_tape.Bindings()) {
      if (!stacked_tape.Reserve(stacked_workspace, binding)) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA resolved execution tape reserve returned null";
        return result;
      }
    }
    if (stacked_workspace.NamedBufferCount() != stacked_tape.BindingCount() ||
        stacked_workspace.TotalCapacityFloats() !=
            stacked_tape.TotalEntries()) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA resolved execution tape workspace mismatch";
      return result;
    }
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

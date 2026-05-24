/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_execution_schedule.h"

#include <sstream>
#include <utility>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

bool IsBoundary(NetworkExecutionOpKind kind) {
  return kind == NetworkExecutionOpKind::InputPack ||
         kind == NetworkExecutionOpKind::OutputDecode;
}

CudaExecutionScheduleEntry
BoundaryEntry(const NetworkResolvedExecutionPlan &plan,
              std::size_t step_index) {
  const auto &step = plan.steps[step_index];
  return CudaExecutionScheduleEntry{CudaExecutionScheduleKind::Boundary,
                                    step_index,
                                    std::numeric_limits<std::size_t>::max(),
                                    step.kind,
                                    step.name,
                                    "handled by CUDA buffer plumbing"};
}

CudaExecutionScheduleEntry
ConvolutionEntry(const NetworkResolvedExecutionPlan &plan,
                 std::size_t convolution_index) {
  const auto &convolution = plan.steps[convolution_index];
  return CudaExecutionScheduleEntry{CudaExecutionScheduleKind::ConvolutionStage,
                                    convolution_index,
                                    std::numeric_limits<std::size_t>::max(),
                                    convolution.kind,
                                    convolution.name,
                                    "standalone convolution"};
}

CudaExecutionScheduleEntry
DenseLayerNormEntry(const NetworkResolvedExecutionPlan &plan,
                    std::size_t dense_index, std::size_t norm_index) {
  const auto &dense = plan.steps[dense_index];
  return CudaExecutionScheduleEntry{
      CudaExecutionScheduleKind::DenseLayerNormStage,
      dense_index,
      norm_index,
      dense.kind,
      dense.name + " -> " + plan.steps[norm_index].name,
      "dense activation plus layernorm"};
}

CudaExecutionScheduleEntry
DenseActivationEntry(const NetworkResolvedExecutionPlan &plan,
                     std::size_t dense_index) {
  const auto &dense = plan.steps[dense_index];
  return CudaExecutionScheduleEntry{
      CudaExecutionScheduleKind::DenseActivationStage,
      dense_index,
      std::numeric_limits<std::size_t>::max(),
      dense.kind,
      dense.name,
      "dense activation"};
}

CudaExecutionScheduleEntry GateEntry(const NetworkResolvedExecutionPlan &plan,
                                     std::size_t gate_index) {
  const auto &gate = plan.steps[gate_index];
  return CudaExecutionScheduleEntry{CudaExecutionScheduleKind::GateStage,
                                    gate_index,
                                    std::numeric_limits<std::size_t>::max(),
                                    gate.kind,
                                    gate.name,
                                    "elementwise gate"};
}

CudaExecutionScheduleEntry
AttentionLayerNormEntry(const NetworkResolvedExecutionPlan &plan,
                        std::size_t attention_index, std::size_t norm_index) {
  const auto &attention = plan.steps[attention_index];
  return CudaExecutionScheduleEntry{
      CudaExecutionScheduleKind::AttentionLayerNormStage,
      attention_index,
      norm_index,
      attention.kind,
      attention.name + " -> " + plan.steps[norm_index].name,
      "attention core plus residual layernorm"};
}

CudaExecutionScheduleEntry
FeedForwardLayerNormEntry(const NetworkResolvedExecutionPlan &plan,
                          std::size_t ffn_index, std::size_t norm_index) {
  const auto &ffn = plan.steps[ffn_index];
  return CudaExecutionScheduleEntry{
      CudaExecutionScheduleKind::FeedForwardLayerNormStage,
      ffn_index,
      norm_index,
      ffn.kind,
      ffn.name + " -> " + plan.steps[norm_index].name,
      "feed-forward residual plus layernorm"};
}

CudaExecutionScheduleEntry
FeedForwardEntry(const NetworkResolvedExecutionPlan &plan,
                 std::size_t ffn_index) {
  const auto &ffn = plan.steps[ffn_index];
  return CudaExecutionScheduleEntry{CudaExecutionScheduleKind::FeedForwardStage,
                                    ffn_index,
                                    std::numeric_limits<std::size_t>::max(),
                                    ffn.kind,
                                    ffn.name,
                                    "feed-forward block"};
}

CudaExecutionScheduleEntry
PolicyMapEntry(const NetworkResolvedExecutionPlan &plan,
               std::size_t policy_map_index) {
  const auto &policy_map = plan.steps[policy_map_index];
  return CudaExecutionScheduleEntry{CudaExecutionScheduleKind::PolicyMapStage,
                                    policy_map_index,
                                    std::numeric_limits<std::size_t>::max(),
                                    policy_map.kind,
                                    policy_map.name,
                                    "attention policy scratch mapping"};
}

CudaExecutionScheduleEntry
PositionalEncodingEntry(const NetworkResolvedExecutionPlan &plan,
                        std::size_t positional_index) {
  const auto &positional = plan.steps[positional_index];
  return CudaExecutionScheduleEntry{
      CudaExecutionScheduleKind::PositionalEncodingStage,
      positional_index,
      std::numeric_limits<std::size_t>::max(),
      positional.kind,
      positional.name,
      "global positional encoding weights"};
}

CudaExecutionScheduleEntry
UnsupportedEntry(const NetworkResolvedExecutionPlan &plan,
                 std::size_t step_index, std::string reason) {
  const auto &step = plan.steps[step_index];
  return CudaExecutionScheduleEntry{CudaExecutionScheduleKind::Unsupported,
                                    step_index,
                                    std::numeric_limits<std::size_t>::max(),
                                    step.kind,
                                    step.name,
                                    std::move(reason)};
}

bool IsSmolgenDenseFor(const NetworkResolvedExecutionStep &step,
                       const NetworkResolvedExecutionStep &attention) {
  return step.kind == NetworkExecutionOpKind::Dense &&
         step.name == attention.name + ".smolgen.dense";
}

bool IsSmolgenNormFor(const NetworkResolvedExecutionStep &step,
                      const NetworkResolvedExecutionStep &attention) {
  return step.kind == NetworkExecutionOpKind::LayerNorm &&
         step.name == attention.name + ".smolgen.norm";
}

bool IsResidualConvolution(const NetworkResolvedExecutionStep &step) {
  return step.kind == NetworkExecutionOpKind::Convolution &&
         step.name.rfind("body.residual.", 0) == 0;
}

void AddEntry(CudaExecutionSchedule &schedule,
              CudaExecutionScheduleEntry entry) {
  switch (entry.kind) {
  case CudaExecutionScheduleKind::Boundary:
    ++schedule.boundary_count;
    break;
  case CudaExecutionScheduleKind::ConvolutionStage:
    ++schedule.convolution_stage_count;
    break;
  case CudaExecutionScheduleKind::DenseActivationStage:
    ++schedule.dense_activation_stage_count;
    break;
  case CudaExecutionScheduleKind::DenseLayerNormStage:
    ++schedule.dense_layernorm_stage_count;
    break;
  case CudaExecutionScheduleKind::GateStage:
    ++schedule.gate_stage_count;
    break;
  case CudaExecutionScheduleKind::AttentionLayerNormStage:
    ++schedule.attention_layernorm_stage_count;
    break;
  case CudaExecutionScheduleKind::FeedForwardStage:
    ++schedule.feed_forward_stage_count;
    break;
  case CudaExecutionScheduleKind::FeedForwardLayerNormStage:
    ++schedule.feed_forward_layernorm_stage_count;
    break;
  case CudaExecutionScheduleKind::PolicyMapStage:
    ++schedule.policy_map_stage_count;
    break;
  case CudaExecutionScheduleKind::PositionalEncodingStage:
    ++schedule.positional_encoding_stage_count;
    break;
  case CudaExecutionScheduleKind::Unsupported:
    ++schedule.unsupported_count;
    break;
  }
  schedule.entries.push_back(std::move(entry));
}

} // namespace

std::string CudaExecutionScheduleKindName(CudaExecutionScheduleKind kind) {
  switch (kind) {
  case CudaExecutionScheduleKind::Boundary:
    return "boundary";
  case CudaExecutionScheduleKind::ConvolutionStage:
    return "convolution_stage";
  case CudaExecutionScheduleKind::DenseActivationStage:
    return "dense_activation_stage";
  case CudaExecutionScheduleKind::DenseLayerNormStage:
    return "dense_layernorm_stage";
  case CudaExecutionScheduleKind::GateStage:
    return "gate_stage";
  case CudaExecutionScheduleKind::AttentionLayerNormStage:
    return "attention_layernorm_stage";
  case CudaExecutionScheduleKind::FeedForwardStage:
    return "feed_forward_stage";
  case CudaExecutionScheduleKind::FeedForwardLayerNormStage:
    return "feed_forward_layernorm_stage";
  case CudaExecutionScheduleKind::PolicyMapStage:
    return "policy_map_stage";
  case CudaExecutionScheduleKind::PositionalEncodingStage:
    return "positional_encoding_stage";
  case CudaExecutionScheduleKind::Unsupported:
    return "unsupported";
  }
  return "unknown";
}

const CudaExecutionScheduleEntry *
CudaExecutionSchedule::FirstUnsupported() const {
  for (const auto &entry : entries) {
    if (entry.kind == CudaExecutionScheduleKind::Unsupported)
      return &entry;
  }
  return nullptr;
}

std::string CudaExecutionSchedule::Summary() const {
  std::ostringstream out;
  out << entries.size() << " CUDA schedule entries, "
      << convolution_stage_count << " convolution stages, "
      << dense_activation_stage_count << " dense/activation stages, "
      << dense_layernorm_stage_count << " dense/layernorm stages, "
      << gate_stage_count << " gate stages, " << attention_layernorm_stage_count
      << " attention/layernorm stages, " << feed_forward_stage_count
      << " feed-forward stages, " << feed_forward_layernorm_stage_count
      << " feed-forward/layernorm stages, " << policy_map_stage_count
      << " policy-map stages, " << positional_encoding_stage_count
      << " positional encoding stages, " << boundary_count << " boundaries, "
      << unsupported_count << " unsupported";
  if (const auto *unsupported = FirstUnsupported()) {
    out << "; first unsupported: " << unsupported->name << " ("
        << NetworkExecutionOpKindName(unsupported->op_kind)
        << "): " << unsupported->reason;
  }
  return out.str();
}

CudaExecutionSchedule
CreateCudaExecutionSchedule(const NetworkResolvedExecutionPlan &plan) {
  CudaExecutionSchedule schedule;
  for (std::size_t i = 0; i < plan.steps.size(); ++i) {
    const auto &step = plan.steps[i];
    if (IsBoundary(step.kind)) {
      AddEntry(schedule, BoundaryEntry(plan, i));
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Convolution) {
      if (IsResidualConvolution(step)) {
        AddEntry(schedule,
                 UnsupportedEntry(plan, i,
                                  "CUDA residual convolution block not "
                                  "implemented yet"));
      } else {
        AddEntry(schedule, ConvolutionEntry(plan, i));
      }
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Dense) {
      const std::size_t norm_index = i + 1;
      if (norm_index < plan.steps.size() &&
          plan.steps[norm_index].kind == NetworkExecutionOpKind::LayerNorm) {
        AddEntry(schedule, DenseLayerNormEntry(plan, i, norm_index));
        i = norm_index;
        continue;
      }
      AddEntry(schedule, DenseActivationEntry(plan, i));
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Gate) {
      AddEntry(schedule, GateEntry(plan, i));
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::Attention) {
      const std::size_t norm_index = i + 1;
      if (norm_index < plan.steps.size() &&
          plan.steps[norm_index].kind == NetworkExecutionOpKind::LayerNorm) {
        AddEntry(schedule, AttentionLayerNormEntry(plan, i, norm_index));
        i = norm_index;
        continue;
      }
      const std::size_t smolgen_dense_index = i + 1;
      const std::size_t smolgen_norm_index = i + 2;
      const std::size_t layernorm_index = i + 3;
      if (layernorm_index < plan.steps.size() &&
          IsSmolgenDenseFor(plan.steps[smolgen_dense_index], step) &&
          IsSmolgenNormFor(plan.steps[smolgen_norm_index], step) &&
          plan.steps[layernorm_index].kind ==
              NetworkExecutionOpKind::LayerNorm) {
        AddEntry(schedule, AttentionLayerNormEntry(plan, i, layernorm_index));
        i = layernorm_index;
        continue;
      }
      AddEntry(schedule,
               UnsupportedEntry(plan, i,
                                "CUDA attention requires adjacent layernorm"));
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::FeedForward) {
      const std::size_t norm_index = i + 1;
      if (norm_index < plan.steps.size() &&
          plan.steps[norm_index].kind == NetworkExecutionOpKind::LayerNorm) {
        AddEntry(schedule, FeedForwardLayerNormEntry(plan, i, norm_index));
        i = norm_index;
        continue;
      }
      AddEntry(schedule, FeedForwardEntry(plan, i));
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::PositionalEncoding) {
      AddEntry(schedule, PositionalEncodingEntry(plan, i));
      continue;
    }

    if (step.kind == NetworkExecutionOpKind::PolicyMap) {
      if (plan.format.attention_policy) {
        AddEntry(schedule, PolicyMapEntry(plan, i));
      } else {
        AddEntry(schedule,
                 UnsupportedEntry(
                     plan, i,
                     "CUDA policy map currently supports attention policy"));
      }
      continue;
    }

    AddEntry(schedule,
             UnsupportedEntry(plan, i, "CUDA kernel not implemented yet"));
  }
  return schedule;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

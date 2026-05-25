/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_plan_analysis.h"

#include <stdexcept>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

CudaPlanStageGroup ToCudaStageGroup(NetworkPlanStageGroup group) {
  switch (group) {
  case NetworkPlanStageGroup::Body:
    return CudaPlanStageGroup::Body;
  case NetworkPlanStageGroup::Policy:
    return CudaPlanStageGroup::Policy;
  case NetworkPlanStageGroup::Value:
    return CudaPlanStageGroup::Value;
  case NetworkPlanStageGroup::MovesLeft:
    return CudaPlanStageGroup::MovesLeft;
  case NetworkPlanStageGroup::Other:
    return CudaPlanStageGroup::Other;
  }
  return CudaPlanStageGroup::Other;
}

NetworkPlanStageGroup ToNetworkStageGroup(CudaPlanStageGroup group) {
  switch (group) {
  case CudaPlanStageGroup::Body:
    return NetworkPlanStageGroup::Body;
  case CudaPlanStageGroup::Policy:
    return NetworkPlanStageGroup::Policy;
  case CudaPlanStageGroup::Value:
    return NetworkPlanStageGroup::Value;
  case CudaPlanStageGroup::MovesLeft:
    return NetworkPlanStageGroup::MovesLeft;
  case CudaPlanStageGroup::Other:
    return NetworkPlanStageGroup::Other;
  }
  return NetworkPlanStageGroup::Other;
}

} // namespace

bool IsCudaDenseScheduleEntry(CudaExecutionScheduleKind kind) {
  return kind == CudaExecutionScheduleKind::DenseActivationStage ||
         kind == CudaExecutionScheduleKind::DenseLayerNormStage;
}

bool IsCudaOutputScheduleEntry(CudaExecutionScheduleKind kind) {
  return kind == CudaExecutionScheduleKind::ConvolutionStage ||
         kind == CudaExecutionScheduleKind::ResidualConvolutionStage ||
         IsCudaDenseScheduleEntry(kind) ||
         kind == CudaExecutionScheduleKind::GateStage ||
         kind == CudaExecutionScheduleKind::AttentionLayerNormStage ||
         kind == CudaExecutionScheduleKind::FeedForwardStage ||
         kind == CudaExecutionScheduleKind::FeedForwardLayerNormStage ||
         kind == CudaExecutionScheduleKind::PolicyMapStage;
}

bool CudaStageNameStartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool CudaStageNameEndsWith(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

std::string
CudaPlanStagePrefix(const NetworkResolvedExecutionPlan &execution_plan,
                    CudaPlanStageGroup group) {
  return NetworkPlanStagePrefix(execution_plan, ToNetworkStageGroup(group));
}

CudaPlanStageGroup
ClassifyCudaPlanStage(const NetworkResolvedExecutionPlan &execution_plan,
                      std::string_view stage_name) {
  return ToCudaStageGroup(ClassifyNetworkPlanStage(execution_plan, stage_name));
}

bool IsCudaValueErrorStage(std::string_view stage_name) {
  return IsNetworkValueErrorStage(stage_name);
}

const CudaExecutionScheduleEntry *
FindCudaStageEntry(const NetworkResolvedExecutionPlan &execution_plan,
                   const CudaExecutionSchedule &schedule,
                   std::string_view stage_name) {
  for (const auto &entry : schedule.entries) {
    if (!IsCudaOutputScheduleEntry(entry.kind))
      continue;
    if (entry.first_step >= execution_plan.steps.size())
      continue;
    if (execution_plan.steps[entry.first_step].name == stage_name)
      return &entry;
  }
  return nullptr;
}

int PreviousCudaOutputWidth(const NetworkResolvedExecutionPlan &execution_plan,
                            std::size_t step_index) {
  for (std::size_t i = step_index; i-- > 0;) {
    const auto &step = execution_plan.steps[i];
    if (step.kind == NetworkExecutionOpKind::Dense)
      return NetworkDenseStageOutputWidth(step);
    if (step.kind == NetworkExecutionOpKind::Convolution)
      return NetworkConvolutionStageOutputChannels(step);
    if (step.kind == NetworkExecutionOpKind::LayerNorm)
      return NetworkLayerNormStageWidth(step);
    if (step.kind == NetworkExecutionOpKind::Attention)
      return NetworkAttentionStageOutputWidth(step);
    if (step.kind == NetworkExecutionOpKind::FeedForward)
      return NetworkFeedForwardStageWidthsFor(step).output;
    if (step.kind == NetworkExecutionOpKind::Gate && !step.tensors.empty()) {
      return static_cast<int>(step.tensors[0].elements);
    }
  }
  return 0;
}

int CudaDenseStageWidth(const NetworkResolvedExecutionPlan &execution_plan,
                        const CudaExecutionScheduleEntry &entry) {
  if (entry.first_step >= execution_plan.steps.size())
    throw std::runtime_error("CUDA dense stage index is invalid");
  const auto &step = execution_plan.steps[entry.first_step];
  if (step.kind != NetworkExecutionOpKind::Dense) {
    throw std::runtime_error("CUDA dense stage tensor is invalid");
  }
  return NetworkDenseStageOutputWidth(step);
}

int CudaOutputStageWidth(const NetworkResolvedExecutionPlan &execution_plan,
                         const CudaExecutionScheduleEntry &entry) {
  if (entry.first_step >= execution_plan.steps.size())
    throw std::runtime_error("CUDA output stage index is invalid");
  const auto &step = execution_plan.steps[entry.first_step];
  if (entry.kind == CudaExecutionScheduleKind::ConvolutionStage) {
    if (step.kind != NetworkExecutionOpKind::Convolution) {
      throw std::runtime_error("CUDA convolution stage tensor is invalid");
    }
    return NetworkConvolutionStageOutputChannels(step);
  }
  if (entry.kind == CudaExecutionScheduleKind::ResidualConvolutionStage) {
    if (entry.second_step >= execution_plan.steps.size())
      throw std::runtime_error(
          "CUDA residual convolution stage index is invalid");
    const auto &conv2 = execution_plan.steps[entry.second_step];
    if (conv2.kind != NetworkExecutionOpKind::Convolution) {
      throw std::runtime_error(
          "CUDA residual convolution stage tensor is invalid");
    }
    return NetworkConvolutionStageOutputChannels(conv2);
  }
  if (IsCudaDenseScheduleEntry(entry.kind))
    return CudaDenseStageWidth(execution_plan, entry);
  if (entry.kind == CudaExecutionScheduleKind::GateStage) {
    if (step.kind != NetworkExecutionOpKind::Gate || step.tensors.empty())
      throw std::runtime_error("CUDA gate stage tensor is invalid");
    const int previous_width =
        PreviousCudaOutputWidth(execution_plan, entry.first_step);
    if (previous_width > 0 &&
        step.tensors[0].elements % static_cast<std::size_t>(previous_width) ==
            0) {
      return previous_width;
    }
    return NetworkGateStageWidth(step);
  }
  if (entry.kind == CudaExecutionScheduleKind::AttentionLayerNormStage) {
    if (step.kind != NetworkExecutionOpKind::Attention) {
      throw std::runtime_error("CUDA attention stage tensor is invalid");
    }
    return NetworkAttentionStageOutputWidth(step);
  }
  if (entry.kind == CudaExecutionScheduleKind::FeedForwardStage ||
      entry.kind == CudaExecutionScheduleKind::FeedForwardLayerNormStage) {
    if (step.kind != NetworkExecutionOpKind::FeedForward) {
      throw std::runtime_error("CUDA feed-forward stage tensor is invalid");
    }
    return NetworkFeedForwardStageWidthsFor(step).output;
  }
  if (entry.kind == CudaExecutionScheduleKind::PolicyMapStage) {
    if (step.kind != NetworkExecutionOpKind::PolicyMap)
      throw std::runtime_error("CUDA policy-map stage tensor is invalid");
    return kNetworkPolicyOutputs;
  }
  throw std::runtime_error("CUDA output stage kind has no width");
}

std::string
LastCudaDenseStageInGroup(const NetworkResolvedExecutionPlan &execution_plan,
                          const CudaExecutionSchedule &schedule,
                          CudaPlanStageGroup group) {
  std::string last_stage;
  for (const auto &entry : schedule.entries) {
    if (!IsCudaDenseScheduleEntry(entry.kind) ||
        entry.first_step >= execution_plan.steps.size()) {
      continue;
    }
    const auto &step = execution_plan.steps[entry.first_step];
    if (ClassifyCudaPlanStage(execution_plan, step.name) == group)
      last_stage = step.name;
  }
  return last_stage;
}

std::string
LastCudaOutputStageInGroup(const NetworkResolvedExecutionPlan &execution_plan,
                           const CudaExecutionSchedule &schedule,
                           CudaPlanStageGroup group) {
  std::string last_stage;
  for (const auto &entry : schedule.entries) {
    if (!IsCudaOutputScheduleEntry(entry.kind) ||
        entry.first_step >= execution_plan.steps.size()) {
      continue;
    }
    const auto &step = execution_plan.steps[entry.first_step];
    if (ClassifyCudaPlanStage(execution_plan, step.name) == group)
      last_stage = step.name;
  }
  return last_stage;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

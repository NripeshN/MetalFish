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
  switch (group) {
  case CudaPlanStageGroup::Body:
    return "body.";
  case CudaPlanStageGroup::Policy:
    return "policy." + execution_plan.policy_head + ".";
  case CudaPlanStageGroup::Value:
    return "value." + execution_plan.value_head + ".";
  case CudaPlanStageGroup::MovesLeft:
    return "moves_left.";
  case CudaPlanStageGroup::Other:
    return {};
  }
  return {};
}

CudaPlanStageGroup
ClassifyCudaPlanStage(const NetworkResolvedExecutionPlan &execution_plan,
                      std::string_view stage_name) {
  if (CudaStageNameStartsWith(
          stage_name,
          CudaPlanStagePrefix(execution_plan, CudaPlanStageGroup::Body))) {
    return CudaPlanStageGroup::Body;
  }
  if (CudaStageNameStartsWith(
          stage_name,
          CudaPlanStagePrefix(execution_plan, CudaPlanStageGroup::Policy))) {
    return CudaPlanStageGroup::Policy;
  }
  if (CudaStageNameStartsWith(
          stage_name,
          CudaPlanStagePrefix(execution_plan, CudaPlanStageGroup::Value))) {
    return CudaPlanStageGroup::Value;
  }
  if (CudaStageNameStartsWith(
          stage_name,
          CudaPlanStagePrefix(execution_plan, CudaPlanStageGroup::MovesLeft))) {
    return CudaPlanStageGroup::MovesLeft;
  }
  return CudaPlanStageGroup::Other;
}

bool IsCudaValueErrorStage(std::string_view stage_name) {
  return CudaStageNameEndsWith(stage_name, ".error");
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
    if (step.kind == NetworkExecutionOpKind::Dense && !step.tensors.empty() &&
        step.tensors[0].dims.size() == 2) {
      return static_cast<int>(step.tensors[0].dims[0]);
    }
    if (step.kind == NetworkExecutionOpKind::Convolution &&
        !step.tensors.empty() && step.tensors[0].dims.size() == 4) {
      return static_cast<int>(step.tensors[0].dims[0]);
    }
    if (step.kind == NetworkExecutionOpKind::LayerNorm &&
        !step.tensors.empty() && step.tensors[0].dims.size() == 1) {
      return static_cast<int>(step.tensors[0].dims[0]);
    }
    if (step.kind == NetworkExecutionOpKind::Attention &&
        step.tensors.size() >= 8 && step.tensors[6].dims.size() == 2) {
      return static_cast<int>(step.tensors[6].dims[0]);
    }
    if (step.kind == NetworkExecutionOpKind::FeedForward &&
        step.tensors.size() >= 4 && step.tensors[2].dims.size() == 2) {
      return static_cast<int>(step.tensors[2].dims[0]);
    }
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
  if (step.kind != NetworkExecutionOpKind::Dense || step.tensors.empty() ||
      step.tensors[0].dims.size() != 2) {
    throw std::runtime_error("CUDA dense stage tensor is invalid");
  }
  return static_cast<int>(step.tensors[0].dims[0]);
}

int CudaOutputStageWidth(const NetworkResolvedExecutionPlan &execution_plan,
                         const CudaExecutionScheduleEntry &entry) {
  if (entry.first_step >= execution_plan.steps.size())
    throw std::runtime_error("CUDA output stage index is invalid");
  const auto &step = execution_plan.steps[entry.first_step];
  if (entry.kind == CudaExecutionScheduleKind::ConvolutionStage) {
    if (step.kind != NetworkExecutionOpKind::Convolution ||
        step.tensors.empty() || step.tensors[0].dims.size() != 4) {
      throw std::runtime_error("CUDA convolution stage tensor is invalid");
    }
    return static_cast<int>(step.tensors[0].dims[0]);
  }
  if (entry.kind == CudaExecutionScheduleKind::ResidualConvolutionStage) {
    if (entry.second_step >= execution_plan.steps.size())
      throw std::runtime_error("CUDA residual convolution stage index is invalid");
    const auto &conv2 = execution_plan.steps[entry.second_step];
    if (conv2.kind != NetworkExecutionOpKind::Convolution ||
        conv2.tensors.empty() || conv2.tensors[0].dims.size() != 4) {
      throw std::runtime_error(
          "CUDA residual convolution stage tensor is invalid");
    }
    return static_cast<int>(conv2.tensors[0].dims[0]);
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
    return static_cast<int>(step.tensors[0].elements);
  }
  if (entry.kind == CudaExecutionScheduleKind::AttentionLayerNormStage) {
    if (step.kind != NetworkExecutionOpKind::Attention ||
        step.tensors.size() < 8 || step.tensors[6].dims.size() != 2) {
      throw std::runtime_error("CUDA attention stage tensor is invalid");
    }
    return static_cast<int>(step.tensors[6].dims[0]);
  }
  if (entry.kind == CudaExecutionScheduleKind::FeedForwardStage ||
      entry.kind == CudaExecutionScheduleKind::FeedForwardLayerNormStage) {
    if (step.kind != NetworkExecutionOpKind::FeedForward ||
        step.tensors.size() < 4 || step.tensors[2].dims.size() != 2) {
      throw std::runtime_error("CUDA feed-forward stage tensor is invalid");
    }
    return static_cast<int>(step.tensors[2].dims[0]);
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

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_output_mapping.h"

#include "cuda_plan_analysis.h"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

float *TargetPointer(CudaInferenceBuffers &buffers, CudaOutputTarget target) {
  switch (target) {
  case CudaOutputTarget::Policy:
    return buffers.policy;
  case CudaOutputTarget::Value:
    return buffers.value;
  case CudaOutputTarget::MovesLeft:
    return buffers.moves_left;
  case CudaOutputTarget::RawPolicy:
    return buffers.raw_policy;
  }
  return nullptr;
}

int TargetStride(const NetworkTensorPlan &tensor_plan,
                 CudaOutputTarget target) {
  switch (target) {
  case CudaOutputTarget::Policy:
    return tensor_plan.policy_outputs;
  case CudaOutputTarget::Value:
    return tensor_plan.value_outputs;
  case CudaOutputTarget::MovesLeft:
    return tensor_plan.moves_left_outputs;
  case CudaOutputTarget::RawPolicy:
    return tensor_plan.raw_policy_outputs;
  }
  return 0;
}

bool AllowsPartialRows(CudaOutputTarget target,
                       const CudaOutputMappingOptions &options) {
  switch (target) {
  case CudaOutputTarget::Policy:
    return options.allow_partial_policy_rows;
  case CudaOutputTarget::RawPolicy:
    return options.allow_partial_raw_policy_rows;
  case CudaOutputTarget::Value:
  case CudaOutputTarget::MovesLeft:
    return false;
  }
  return false;
}

bool StageWidthFitsTarget(int source_width, int target_stride,
                          CudaOutputTarget target,
                          const CudaOutputMappingOptions &options) {
  if (target_stride <= 0 || source_width <= 0)
    return false;
  if (source_width > target_stride)
    return false;
  return AllowsPartialRows(target, options) || source_width == target_stride;
}

std::string SelectCompatibleStage(
    const NetworkTensorPlan &tensor_plan,
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaExecutionSchedule &schedule,
    const CudaOutputMappingOptions &options, CudaOutputTarget target,
    CudaPlanStageGroup group, bool first_match) {
  const int target_stride = TargetStride(tensor_plan, target);
  std::string selected;
  for (const auto &entry : schedule.entries) {
    if (!IsCudaOutputScheduleEntry(entry.kind) ||
        entry.first_step >= execution_plan.steps.size()) {
      continue;
    }
    const auto &step = execution_plan.steps[entry.first_step];
    if (ClassifyCudaPlanStage(execution_plan, step.name) != group)
      continue;
    if (target == CudaOutputTarget::Value && IsCudaValueErrorStage(step.name))
      continue;

    const int source_width = CudaOutputStageWidth(execution_plan, entry);
    if (!StageWidthFitsTarget(source_width, target_stride, target, options))
      continue;

    selected = step.name;
    if (first_match)
      break;
  }
  return selected;
}

std::string SelectPolicyStage(const NetworkTensorPlan &tensor_plan,
                              const NetworkResolvedExecutionPlan &execution_plan,
                              const CudaExecutionSchedule &schedule,
                              const CudaOutputMappingOptions &options) {
  for (const auto &entry : schedule.entries) {
    if (entry.kind != CudaExecutionScheduleKind::PolicyMapStage ||
        entry.first_step >= execution_plan.steps.size()) {
      continue;
    }
    const auto &step = execution_plan.steps[entry.first_step];
    if (ClassifyCudaPlanStage(execution_plan, step.name) ==
        CudaPlanStageGroup::Policy) {
      return step.name;
    }
  }
  return SelectCompatibleStage(tensor_plan, execution_plan, schedule, options,
                               CudaOutputTarget::Policy,
                               CudaPlanStageGroup::Policy, true);
}

void AddError(CudaOutputMapping &mapping, const std::string &error) {
  mapping.errors.push_back(error);
}

void AddBinding(CudaOutputMapping &mapping,
                const NetworkTensorPlan &tensor_plan,
                const NetworkResolvedExecutionPlan &execution_plan,
                const CudaExecutionSchedule &schedule,
                const CudaOutputMappingOptions &options,
                CudaOutputTarget target, const std::string &source_stage,
                bool required) {
  const int target_stride = TargetStride(tensor_plan, target);
  if (target_stride == 0) {
    if (required)
      AddError(mapping, CudaOutputTargetName(target) + " output is disabled");
    return;
  }

  if (source_stage.empty()) {
    if (required) {
      AddError(mapping, "missing CUDA output source for " +
                            CudaOutputTargetName(target));
    }
    return;
  }

  const auto *entry =
      FindCudaStageEntry(execution_plan, schedule, source_stage);
  if (!entry) {
    if (required) {
      AddError(mapping, "missing CUDA output source for " +
                            CudaOutputTargetName(target) + ": " +
                            source_stage);
    }
    return;
  }

  const int source_width = CudaOutputStageWidth(execution_plan, *entry);
  const bool partial = AllowsPartialRows(target, options);
  if (source_width > target_stride ||
      (!partial && source_width != target_stride)) {
    std::ostringstream out;
    out << "CUDA output source width mismatch for "
        << CudaOutputTargetName(target) << ": " << source_stage << " has "
        << source_width << ", target stride is " << target_stride;
    AddError(mapping, out.str());
    return;
  }

  mapping.bindings.push_back(
      CudaOutputBinding{target, source_stage, source_width, target_stride});
}

} // namespace

std::string CudaOutputTargetName(CudaOutputTarget target) {
  switch (target) {
  case CudaOutputTarget::Policy:
    return "policy";
  case CudaOutputTarget::Value:
    return "value";
  case CudaOutputTarget::MovesLeft:
    return "moves_left";
  case CudaOutputTarget::RawPolicy:
    return "raw_policy";
  }
  return "unknown";
}

const CudaOutputBinding *
CudaOutputMapping::Find(CudaOutputTarget target) const {
  for (const auto &binding : bindings) {
    if (binding.target == target)
      return &binding;
  }
  return nullptr;
}

std::string CudaOutputMapping::Summary() const {
  std::ostringstream out;
  out << bindings.size() << " output bindings";
  if (!errors.empty()) {
    out << ", " << errors.size() << " errors";
    for (std::size_t i = 0; i < errors.size(); ++i) {
      out << (i == 0 ? ": " : "; ") << errors[i];
    }
  }
  return out.str();
}

CudaOutputMapping CreateCudaOutputMapping(
    const NetworkTensorPlan &tensor_plan,
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaExecutionSchedule &schedule,
    CudaOutputMappingOptions options) {
  CudaOutputMapping mapping;
  if (!schedule.FullySupported()) {
    AddError(mapping, "CUDA output mapping requires a fully supported schedule");
    return mapping;
  }

  const std::string policy_source =
      SelectPolicyStage(tensor_plan, execution_plan, schedule, options);
  const std::string value_source = SelectCompatibleStage(
      tensor_plan, execution_plan, schedule, options, CudaOutputTarget::Value,
      CudaPlanStageGroup::Value, false);
  AddBinding(mapping, tensor_plan, execution_plan, schedule, options,
             CudaOutputTarget::Policy, policy_source, true);
  AddBinding(mapping, tensor_plan, execution_plan, schedule, options,
             CudaOutputTarget::Value, value_source, true);
  if (tensor_plan.moves_left) {
    const std::string moves_left_source = SelectCompatibleStage(
        tensor_plan, execution_plan, schedule, options,
        CudaOutputTarget::MovesLeft, CudaPlanStageGroup::MovesLeft, false);
    AddBinding(mapping, tensor_plan, execution_plan, schedule, options,
               CudaOutputTarget::MovesLeft, moves_left_source, true);
  }
  const auto *policy_entry =
      FindCudaStageEntry(execution_plan, schedule, policy_source);
  if (tensor_plan.raw_policy_outputs > 0 && policy_entry &&
      policy_entry->kind != CudaExecutionScheduleKind::PolicyMapStage) {
    AddBinding(mapping, tensor_plan, execution_plan, schedule, options,
               CudaOutputTarget::RawPolicy, policy_source, true);
  }
  return mapping;
}

void CopyMappedOutputs(const CudaOutputMapping &mapping,
                       const CudaDenseStageSequenceOutput &sequence,
                       CudaInferenceBuffers &buffers,
                       CudaExecutionWorkspace &workspace, int batch_size) {
  if (!mapping.ok()) {
    throw std::runtime_error("CUDA output mapping is incomplete: " +
                             mapping.Summary());
  }
  if (batch_size <= 0)
    throw std::runtime_error("CUDA output mapping received empty batch");

  cudaStream_t stream = workspace.Stream();
  for (const auto &binding : mapping.bindings) {
    const auto *stage = sequence.FindStage(binding.source_stage);
    if (!stage || !stage->output) {
      throw std::runtime_error("CUDA output source was not executed: " +
                               binding.source_stage);
    }
    if (stage->output_width != binding.source_width) {
      throw std::runtime_error("CUDA output source width changed: " +
                               binding.source_stage);
    }
    CopyDeviceFloatRows(TargetPointer(buffers, binding.target),
                        binding.target_stride, stage->output,
                        stage->output_width, batch_size, binding.source_width,
                        "cudaMemcpy(" + CudaOutputTargetName(binding.target) +
                            ")",
                        stream);
  }
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_stage_bindings.h"

#include "cuda_plan_analysis.h"

#include <stdexcept>
#include <utility>

namespace MetalFish {
namespace NN {
namespace Cuda {

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

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network_execution_plan.h"
#include "cuda_execution_schedule.h"

#include <string>
#include <string_view>

namespace MetalFish {
namespace NN {
namespace Cuda {

enum class CudaPlanStageGroup {
  Other,
  Body,
  Policy,
  Value,
  MovesLeft,
};

bool IsCudaDenseScheduleEntry(CudaExecutionScheduleKind kind);
bool CudaStageNameStartsWith(std::string_view value,
                             std::string_view prefix);
bool CudaStageNameEndsWith(std::string_view value, std::string_view suffix);
std::string CudaPlanStagePrefix(
    const NetworkResolvedExecutionPlan &execution_plan,
    CudaPlanStageGroup group);
CudaPlanStageGroup ClassifyCudaPlanStage(
    const NetworkResolvedExecutionPlan &execution_plan,
    std::string_view stage_name);
bool IsCudaValueErrorStage(std::string_view stage_name);
const CudaExecutionScheduleEntry *FindCudaStageEntry(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaExecutionSchedule &schedule, std::string_view stage_name);
int CudaDenseStageWidth(const NetworkResolvedExecutionPlan &execution_plan,
                        const CudaExecutionScheduleEntry &entry);
std::string LastCudaDenseStageInGroup(
    const NetworkResolvedExecutionPlan &execution_plan,
    const CudaExecutionSchedule &schedule, CudaPlanStageGroup group);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

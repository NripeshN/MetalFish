/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace MetalFish {
namespace NN {
struct NetworkResolvedExecutionPlan;

namespace Cuda {
struct CudaExecutionSchedule;

struct CudaStageInputBinding {
  std::string stage_name;
  std::string source_stage_name;
};

class CudaStageInputBindings {
public:
  void Add(std::string stage_name, std::string source_stage_name);
  const std::string *FindSource(std::string_view stage_name) const;
  std::size_t Size() const { return bindings_.size(); }

private:
  std::vector<CudaStageInputBinding> bindings_;
};

CudaStageInputBindings
CreateCudaStageInputBindings(const NetworkResolvedExecutionPlan &execution_plan,
                             const CudaExecutionSchedule &schedule);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

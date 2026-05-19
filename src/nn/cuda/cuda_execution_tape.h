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

#include "../network_execution_plan.h"
#include "../network_tensor_plan.h"
#include "cuda_workspace.h"

namespace MetalFish {
namespace NN {
namespace Cuda {

enum class CudaExecutionBufferRole {
  DenseOutput,
  ActivationOutput,
  NormalizedOutput,
  GateOutput,
};

std::string CudaExecutionBufferRoleName(CudaExecutionBufferRole role);

struct CudaExecutionBufferBinding {
  std::string name;
  CudaExecutionBufferRole role = CudaExecutionBufferRole::DenseOutput;
  std::size_t entries = 0;
  int rows = 0;
  int width = 0;
};

class CudaExecutionTape {
public:
  const std::vector<CudaExecutionBufferBinding> &Bindings() const {
    return bindings_;
  }
  const CudaExecutionBufferBinding *FindName(std::string_view name) const;
  const CudaExecutionBufferBinding &RequireName(std::string_view name) const;
  const CudaExecutionBufferBinding &
  RequireRole(CudaExecutionBufferRole role) const;
  float *Reserve(CudaExecutionWorkspace &workspace,
                 const CudaExecutionBufferBinding &binding) const;
  std::size_t BindingCount() const { return bindings_.size(); }
  std::size_t CountRole(CudaExecutionBufferRole role) const;
  std::size_t TotalEntries() const;
  std::string Summary() const;

private:
  friend CudaExecutionTape
  CreateResolvedExecutionTape(const NetworkResolvedExecutionPlan &plan,
                              int batch_size);
  friend CudaExecutionTape
  CreatePlanSmokeExecutionTape(const NetworkTensorPlan &tensor_plan,
                               const NetworkResolvedExecutionPlan &plan,
                               int batch_size);

  void Add(std::string name, CudaExecutionBufferRole role, int rows, int width);

  std::vector<CudaExecutionBufferBinding> bindings_;
};

CudaExecutionTape CreateResolvedExecutionTape(
    const NetworkResolvedExecutionPlan &plan, int batch_size);

CudaExecutionTape
CreatePlanSmokeExecutionTape(const NetworkTensorPlan &tensor_plan,
                             const NetworkResolvedExecutionPlan &plan,
                             int batch_size);

CudaWorkspaceSmokeResult RunExecutionTapeSmoke();

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

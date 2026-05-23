/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <string>

namespace MetalFish {
namespace NN {
struct NetworkResolvedExecutionPlan;
struct NetworkTensorPlan;

namespace Cuda {

struct CudaExecutionSchedule;
struct CudaOutputMapping;
class CudaInferenceBuffers;
class CudaExecutionWorkspace;
class CudaWeightBuffers;

class CudaExecutor {
public:
  virtual ~CudaExecutor() = default;

  virtual void Execute(const NetworkTensorPlan &tensor_plan,
                       const NetworkResolvedExecutionPlan &execution_plan,
                       const CudaWeightBuffers &weights,
                       CudaInferenceBuffers &buffers,
                       CudaExecutionWorkspace &workspace, int batch_size) = 0;
  virtual std::string Name() const = 0;
};

class CudaProfileSuppressionScope {
public:
  CudaProfileSuppressionScope();
  ~CudaProfileSuppressionScope();

  CudaProfileSuppressionScope(const CudaProfileSuppressionScope &) = delete;
  CudaProfileSuppressionScope &
  operator=(const CudaProfileSuppressionScope &) = delete;

private:
  bool previous_ = false;
};

std::unique_ptr<CudaExecutor> CreateMissingCudaExecutor();
std::unique_ptr<CudaExecutor> CreateNullCudaExecutorForSmoke();
std::unique_ptr<CudaExecutor> CreatePlanSmokeCudaExecutor();
std::unique_ptr<CudaExecutor>
CreateResolvedCudaExecutor(CudaExecutionSchedule schedule,
                           CudaOutputMapping output_mapping);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

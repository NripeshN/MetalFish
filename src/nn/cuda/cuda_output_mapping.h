/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network_tensor_plan.h"

#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
struct NetworkResolvedExecutionPlan;

namespace Cuda {

struct CudaDenseStageSequenceOutput;
struct CudaExecutionSchedule;
class CudaInferenceBuffers;
class CudaExecutionWorkspace;

struct CudaOutputMappingOptions {
  bool allow_partial_policy_rows = false;
  bool allow_partial_raw_policy_rows = false;
};

struct CudaOutputBinding {
  NetworkOutputTarget target = NetworkOutputTarget::Policy;
  std::string source_stage;
  int source_width = 0;
  int target_stride = 0;
};

struct CudaOutputMapping {
  std::vector<CudaOutputBinding> bindings;
  std::vector<std::string> errors;

  bool ok() const { return errors.empty(); }
  const CudaOutputBinding *Find(NetworkOutputTarget target) const;
  std::string Summary() const;
};

CudaOutputMapping
CreateCudaOutputMapping(const NetworkTensorPlan &tensor_plan,
                        const NetworkResolvedExecutionPlan &execution_plan,
                        const CudaExecutionSchedule &schedule,
                        CudaOutputMappingOptions options = {});

void CopyMappedOutputs(const CudaOutputMapping &mapping,
                       const CudaDenseStageSequenceOutput &sequence,
                       CudaInferenceBuffers &buffers,
                       CudaExecutionWorkspace &workspace, int batch_size);

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

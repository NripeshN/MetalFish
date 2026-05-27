/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "cuda_buffers.h"
#include "cuda_executor.h"
#include "cuda_weight_buffers.h"
#include "cuda_workspace.h"

#include "../network.h"
#include "../network_execution_plan.h"

#include <memory>
#include <mutex>
#include <span>

namespace MetalFish {
namespace NN {
namespace Cuda {

class CudaNetwork : public Network {
public:
  explicit CudaNetwork(const WeightsFile &weights,
                       BackendConfig config = BackendConfig{});

  NetworkOutput Evaluate(const InputPlanes &input) override;
  std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) override;
  std::string GetNetworkInfo() const override;
  bool HasWDL() const override;
  bool HasMovesLeft() const override;
  BackendCapabilities GetBackendCapabilities() const override;

private:
  void WarmupExecution();
  std::vector<NetworkOutput> RunBatch(std::span<const InputPlanes> inputs);

  NetworkFormatDescriptor format_;
  BackendConfig config_;
  NetworkTensorPlan tensor_plan_;
  NetworkExecutionPlan execution_plan_;
  NetworkResolvedExecutionPlan resolved_execution_plan_;
  CudaBufferLayout buffer_layout_;
  CudaInferenceBuffers buffers_;
  CudaWeightBuffers weight_buffers_;
  CudaExecutionWorkspace workspace_;
  int workspace_batch_size_ = 0;
  int selected_cuda_device_ = -1;
  std::string device_selection_summary_;
  std::unique_ptr<CudaExecutor> executor_;
  std::mutex execution_mutex_;
};

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

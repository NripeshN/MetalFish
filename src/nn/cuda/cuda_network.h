/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "cuda_buffers.h"
#include "cuda_executor.h"
#include "cuda_weight_buffers.h"

#include "../network.h"
#include "../network_format.h"

#include <memory>
#include <mutex>

namespace MetalFish {
namespace NN {
namespace Cuda {

class CudaNetwork : public Network {
public:
  explicit CudaNetwork(const WeightsFile &weights);

  NetworkOutput Evaluate(const InputPlanes &input) override;
  std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) override;
  std::string GetNetworkInfo() const override;

private:
  NetworkFormatDescriptor format_;
  NetworkTensorPlan tensor_plan_;
  CudaBufferLayout buffer_layout_;
  CudaInferenceBuffers buffers_;
  CudaWeightBuffers weight_buffers_;
  std::unique_ptr<CudaExecutor> executor_;
  std::mutex execution_mutex_;
};

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

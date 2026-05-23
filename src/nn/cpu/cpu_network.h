/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network.h"
#include "../network_execution_plan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cpu {

class CpuNetwork : public Network {
public:
  explicit CpuNetwork(const WeightsFile &weights);

  NetworkOutput Evaluate(const InputPlanes &input) override;
  std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) override;
  std::string GetNetworkInfo() const override;

private:
  struct CpuTensor {
    std::string name;
    std::vector<float> data;
    std::vector<std::uint32_t> dims;
    NetworkWeightTensorKind kind = NetworkWeightTensorKind::Generic;
  };

  std::string UnsupportedExecutionMessage() const;
  const CpuTensor &TensorAt(std::size_t index) const;
  std::vector<NetworkOutput>
  RunBatch(const std::vector<InputPlanes> &inputs) const;

  NetworkFormatDescriptor format_;
  NetworkTensorPlan tensor_plan_;
  NetworkExecutionPlan execution_plan_;
  NetworkResolvedExecutionPlan resolved_execution_plan_;
  std::vector<CpuTensor> tensors_;
  std::size_t weight_bytes_ = 0;
  std::string unsupported_execution_reason_;
};

} // namespace Cpu
} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_executor.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

std::string CudaErrorMessage(const char *op, cudaError_t status) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorString(status);
  return out.str();
}

void UploadDeviceFloats(float *ptr, const std::vector<float> &host,
                        const char *name) {
  if (host.empty())
    return;
  if (!ptr) {
    throw std::runtime_error(std::string("CUDA output buffer is missing: ") +
                             name);
  }
  const cudaError_t status =
      cudaMemcpy(ptr, host.data(), host.size() * sizeof(float),
                 cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

class MissingCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &, CudaInferenceBuffers &,
               int) override {
    throw std::runtime_error(
        "CUDA transformer executor is not implemented yet");
  }

  std::string Name() const override { return "missing"; }
};

class NullCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &plan, CudaInferenceBuffers &buffers,
               int batch_size) override {
    std::vector<float> policy(plan.PolicyEntries(batch_size), 0.0f);
    std::vector<float> value(plan.ValueEntries(batch_size), 0.0f);
    std::vector<float> moves_left(plan.MovesLeftEntries(batch_size), 0.0f);
    std::vector<float> raw_policy(plan.RawPolicyEntries(batch_size), 0.0f);

    for (int b = 0; b < batch_size; ++b) {
      const size_t policy_offset =
          static_cast<size_t>(b) * plan.policy_outputs;
      policy[policy_offset] = 0.25f + static_cast<float>(b);
      policy[policy_offset + plan.policy_outputs - 1] =
          -0.75f - static_cast<float>(b);

      const size_t value_offset = static_cast<size_t>(b) * 3;
      value[value_offset + 0] = 0.70f;
      value[value_offset + 1] = 0.20f;
      value[value_offset + 2] = 0.10f + 0.05f * static_cast<float>(b);

      moves_left[static_cast<size_t>(b)] = 12.0f + static_cast<float>(b);
      if (!raw_policy.empty()) {
        const size_t raw_offset =
            static_cast<size_t>(b) * plan.raw_policy_outputs;
        raw_policy[raw_offset] = 3.0f + static_cast<float>(b);
      }
    }

    UploadDeviceFloats(buffers.policy, policy, "cudaMemcpy(policy)");
    UploadDeviceFloats(buffers.value, value, "cudaMemcpy(value)");
    UploadDeviceFloats(buffers.moves_left, moves_left,
                       "cudaMemcpy(moves_left)");
    UploadDeviceFloats(buffers.raw_policy, raw_policy,
                       "cudaMemcpy(raw_policy)");
  }

  std::string Name() const override { return "null-smoke"; }
};

} // namespace

std::unique_ptr<CudaExecutor> CreateMissingCudaExecutor() {
  return std::make_unique<MissingCudaExecutor>();
}

std::unique_ptr<CudaExecutor> CreateNullCudaExecutorForSmoke() {
  return std::make_unique<NullCudaExecutor>();
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

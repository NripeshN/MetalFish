/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_executor.h"

#include "cuda_kernels.h"

#include <cuda_runtime_api.h>

#include <cstddef>
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
  const cudaError_t status = cudaMemcpy(
      ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

void FreeDevice(float *ptr) {
  if (ptr)
    cudaFree(ptr);
}

float *AllocateDeviceFloats(std::size_t entries, const char *name) {
  if (entries == 0)
    throw std::runtime_error(std::string("CUDA scratch size is zero: ") + name);
  float *ptr = nullptr;
  const cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(&ptr), entries * sizeof(float));
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
  return ptr;
}

void CopyDeviceFloats(float *dst, const float *src, std::size_t entries,
                      const char *name) {
  if (entries == 0)
    return;
  if (!dst || !src)
    throw std::runtime_error(std::string("CUDA device copy missing buffer: ") +
                             name);
  const cudaError_t status =
      cudaMemcpy(dst, src, entries * sizeof(float), cudaMemcpyDeviceToDevice);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

CudaActivationKind ActivationFromString(const std::string &activation) {
  if (activation == "relu_2")
    return CudaActivationKind::Relu2;
  if (activation == "tanh")
    return CudaActivationKind::Tanh;
  if (activation == "sigmoid")
    return CudaActivationKind::Sigmoid;
  if (activation == "swish")
    return CudaActivationKind::Swish;
  if (activation == "mish")
    return CudaActivationKind::Mish;
  if (activation == "selu")
    return CudaActivationKind::Selu;
  return CudaActivationKind::Relu;
}

const NetworkResolvedExecutionStep &
FindStep(const NetworkResolvedExecutionPlan &execution_plan,
         NetworkExecutionOpKind kind) {
  for (const auto &step : execution_plan.steps) {
    if (step.kind == kind)
      return step;
  }
  throw std::runtime_error("CUDA smoke execution plan is missing step");
}

class MissingCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &, const NetworkResolvedExecutionPlan &,
               const CudaWeightBuffers &, CudaInferenceBuffers &, int) override {
    throw std::runtime_error(
        "CUDA transformer executor is not implemented yet");
  }

  std::string Name() const override { return "missing"; }
};

class NullCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &plan, const NetworkResolvedExecutionPlan &,
               const CudaWeightBuffers &, CudaInferenceBuffers &buffers,
               int batch_size) override {
    std::vector<float> policy(plan.PolicyEntries(batch_size), 0.0f);
    std::vector<float> value(plan.ValueEntries(batch_size), 0.0f);
    std::vector<float> moves_left(plan.MovesLeftEntries(batch_size), 0.0f);
    std::vector<float> raw_policy(plan.RawPolicyEntries(batch_size), 0.0f);

    for (int b = 0; b < batch_size; ++b) {
      const size_t policy_offset = static_cast<size_t>(b) * plan.policy_outputs;
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

class PlanSmokeCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &plan,
               const NetworkResolvedExecutionPlan &execution_plan,
               const CudaWeightBuffers &weights, CudaInferenceBuffers &buffers,
               int batch_size) override {
    if (batch_size != 1)
      throw std::runtime_error("CUDA plan smoke executor supports batch size 1");
    if (!buffers.input_values || !buffers.policy)
      throw std::runtime_error("CUDA plan smoke executor received no buffers");

    const auto &dense = FindStep(execution_plan, NetworkExecutionOpKind::Dense);
    const auto &norm =
        FindStep(execution_plan, NetworkExecutionOpKind::LayerNorm);
    if (dense.tensors.size() < 2 || norm.tensors.size() < 2)
      throw std::runtime_error("CUDA smoke execution plan has missing tensors");

    const auto dense_weight =
        weights.TensorAt(dense.tensors[0].inventory_index);
    const auto dense_bias = weights.TensorAt(dense.tensors[1].inventory_index);
    const auto gamma = weights.TensorAt(norm.tensors[0].inventory_index);
    const auto beta = weights.TensorAt(norm.tensors[1].inventory_index);
    if (dense_weight.dims.size() != 2 || dense_bias.elements == 0)
      throw std::runtime_error("CUDA smoke dense tensor shape is invalid");

    const int output_width = static_cast<int>(dense_weight.dims[0]);
    const int input_width = static_cast<int>(dense_weight.dims[1]);
    if (dense_bias.elements != static_cast<std::size_t>(output_width) ||
        gamma.elements != static_cast<std::size_t>(output_width) ||
        beta.elements != static_cast<std::size_t>(output_width) ||
        output_width > plan.policy_outputs) {
      throw std::runtime_error("CUDA smoke tensor dimensions are inconsistent");
    }

    float *dense_output = nullptr;
    float *activation_output = nullptr;
    try {
      const std::size_t scratch_entries =
          static_cast<std::size_t>(batch_size) * output_width;
      dense_output = AllocateDeviceFloats(scratch_entries,
                                          "cudaMalloc(smoke_dense_output)");
      activation_output = AllocateDeviceFloats(
          scratch_entries, "cudaMalloc(smoke_activation_output)");

      LaunchDenseAffineKernel(buffers.input_values, dense_weight.data,
                              dense_bias.data, dense_output, batch_size,
                              input_width, output_width);
      LaunchActivationKernel(
          dense_output, activation_output, static_cast<int>(scratch_entries),
          ActivationFromString(execution_plan.format.activations.ffn_activation));
      LaunchLayerNormKernel(activation_output, gamma.data, beta.data,
                            buffers.policy, batch_size, output_width, 1e-5f);

      if (buffers.raw_policy)
        CopyDeviceFloats(buffers.raw_policy, activation_output, scratch_entries,
                         "cudaMemcpy(smoke_raw_policy)");
    } catch (...) {
      FreeDevice(dense_output);
      FreeDevice(activation_output);
      throw;
    }

    FreeDevice(dense_output);
    FreeDevice(activation_output);
  }

  std::string Name() const override { return "plan-smoke"; }
};

} // namespace

std::unique_ptr<CudaExecutor> CreateMissingCudaExecutor() {
  return std::make_unique<MissingCudaExecutor>();
}

std::unique_ptr<CudaExecutor> CreateNullCudaExecutorForSmoke() {
  return std::make_unique<NullCudaExecutor>();
}

std::unique_ptr<CudaExecutor> CreatePlanSmokeCudaExecutor() {
  return std::make_unique<PlanSmokeCudaExecutor>();
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

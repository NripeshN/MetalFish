/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_executor.h"

#include "cuda_execution_tape.h"
#include "cuda_output_mapping.h"
#include "cuda_stage_executor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

class MissingCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &, const NetworkResolvedExecutionPlan &,
               const CudaWeightBuffers &, CudaInferenceBuffers &,
               CudaExecutionWorkspace &, int) override {
    throw std::runtime_error(
        "CUDA transformer executor is not implemented yet");
  }

  std::string Name() const override { return "missing"; }
};

class NullCudaExecutor final : public CudaExecutor {
public:
  void Execute(const NetworkTensorPlan &plan, const NetworkResolvedExecutionPlan &,
               const CudaWeightBuffers &, CudaInferenceBuffers &buffers,
               CudaExecutionWorkspace &, int batch_size) override {
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
               CudaExecutionWorkspace &workspace, int batch_size) override {
    if (batch_size <= 0)
      throw std::runtime_error("CUDA plan smoke executor received empty batch");
    if (!buffers.input_values || !buffers.policy)
      throw std::runtime_error("CUDA plan smoke executor received no buffers");

    const auto tape =
        CreatePlanSmokeExecutionTape(plan, execution_plan, batch_size);
    const auto schedule = CreateCudaExecutionSchedule(execution_plan);
    const auto stage_inputs =
        CreateCudaStageInputBindings(execution_plan, schedule);
    const auto sequence = ExecuteDenseActivationLayerNormSequence(
        execution_plan, weights, buffers.input_values, tape, workspace,
        batch_size, stage_inputs);
    CudaOutputMappingOptions options;
    options.allow_partial_policy_rows = true;
    options.allow_partial_raw_policy_rows = true;
    const auto mapping =
        CreateCudaOutputMapping(plan, execution_plan, schedule, options);
    CopyMappedOutputs(mapping, sequence, buffers, workspace, batch_size);
    workspace.Synchronize();
  }

  std::string Name() const override { return "plan-smoke"; }
};

class ResolvedCudaExecutor final : public CudaExecutor {
public:
  ResolvedCudaExecutor(CudaExecutionSchedule schedule,
                       CudaOutputMapping output_mapping)
      : schedule_(std::move(schedule)),
        output_mapping_(std::move(output_mapping)) {}

  void Execute(const NetworkTensorPlan &,
               const NetworkResolvedExecutionPlan &execution_plan,
               const CudaWeightBuffers &weights, CudaInferenceBuffers &buffers,
               CudaExecutionWorkspace &workspace, int batch_size) override {
    if (batch_size <= 0)
      throw std::runtime_error("CUDA resolved executor received empty batch");
    if (!buffers.input_masks || !buffers.input_values || !buffers.policy)
      throw std::runtime_error("CUDA resolved executor received no buffers");
    if (!schedule_.FullySupported()) {
      throw std::runtime_error("CUDA resolved executor schedule is unsupported: " +
                               schedule_.Summary());
    }
    if (!output_mapping_.ok()) {
      throw std::runtime_error("CUDA resolved executor output mapping failed: " +
                               output_mapping_.Summary());
    }

    const auto tape = CreateResolvedExecutionTape(execution_plan, batch_size);
    const auto stage_inputs =
        CreateCudaStageInputBindings(execution_plan, schedule_);
    const auto sequence = ExecuteDenseActivationLayerNormSequence(
        execution_plan, weights, buffers.input_values, buffers.input_masks,
        buffers.input_values, tape, workspace, batch_size, stage_inputs);
    CopyMappedOutputs(output_mapping_, sequence, buffers, workspace,
                      batch_size);
    workspace.Synchronize();
  }

  std::string Name() const override { return "resolved"; }

private:
  CudaExecutionSchedule schedule_;
  CudaOutputMapping output_mapping_;
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

std::unique_ptr<CudaExecutor>
CreateResolvedCudaExecutor(CudaExecutionSchedule schedule,
                           CudaOutputMapping output_mapping) {
  return std::make_unique<ResolvedCudaExecutor>(std::move(schedule),
                                                std::move(output_mapping));
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_execution_tape.h"

#include "cuda_runtime_probe.h"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

const NetworkResolvedExecutionStep &
FindStep(const NetworkResolvedExecutionPlan &execution_plan,
         NetworkExecutionOpKind kind) {
  for (const auto &step : execution_plan.steps) {
    if (step.kind == kind)
      return step;
  }
  throw std::runtime_error("CUDA execution tape is missing required step");
}

int DenseOutputWidth(const NetworkResolvedExecutionStep &dense) {
  if (dense.tensors.empty() || dense.tensors[0].dims.size() != 2)
    throw std::runtime_error("CUDA execution tape dense tensor is invalid");
  const auto width = dense.tensors[0].dims[0];
  if (width == 0)
    throw std::runtime_error("CUDA execution tape dense width is zero");
  return static_cast<int>(width);
}

} // namespace

std::string CudaExecutionBufferRoleName(CudaExecutionBufferRole role) {
  switch (role) {
  case CudaExecutionBufferRole::DenseOutput:
    return "dense_output";
  case CudaExecutionBufferRole::ActivationOutput:
    return "activation_output";
  case CudaExecutionBufferRole::NormalizedOutput:
    return "normalized_output";
  }
  return "unknown";
}

const CudaExecutionBufferBinding &
CudaExecutionTape::RequireRole(CudaExecutionBufferRole role) const {
  for (const auto &binding : bindings_) {
    if (binding.role == role)
      return binding;
  }
  throw std::runtime_error("CUDA execution tape is missing buffer role: " +
                           CudaExecutionBufferRoleName(role));
}

float *CudaExecutionTape::Reserve(
    CudaExecutionWorkspace &workspace,
    const CudaExecutionBufferBinding &binding) const {
  return workspace.ReserveNamedFloats(binding.name, binding.entries);
}

std::size_t CudaExecutionTape::TotalEntries() const {
  std::size_t total = 0;
  for (const auto &binding : bindings_)
    total += binding.entries;
  return total;
}

std::string CudaExecutionTape::Summary() const {
  std::ostringstream out;
  out << bindings_.size() << " bindings, " << TotalEntries()
      << " intermediate floats";
  return out.str();
}

void CudaExecutionTape::Add(std::string name, CudaExecutionBufferRole role,
                            int rows, int width) {
  if (name.empty() || rows <= 0 || width <= 0)
    throw std::runtime_error("CUDA execution tape binding is invalid");
  bindings_.push_back(CudaExecutionBufferBinding{
      std::move(name), role, static_cast<std::size_t>(rows) * width, rows,
      width});
}

CudaExecutionTape
CreatePlanSmokeExecutionTape(const NetworkTensorPlan &tensor_plan,
                             const NetworkResolvedExecutionPlan &plan,
                             int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA execution tape batch size is invalid");

  const auto &dense = FindStep(plan, NetworkExecutionOpKind::Dense);
  const auto &norm = FindStep(plan, NetworkExecutionOpKind::LayerNorm);
  const int output_width = DenseOutputWidth(dense);
  if (output_width > tensor_plan.policy_outputs ||
      output_width > tensor_plan.raw_policy_outputs) {
    throw std::runtime_error("CUDA execution tape output width exceeds head");
  }

  CudaExecutionTape tape;
  tape.Add(dense.name + ".dense", CudaExecutionBufferRole::DenseOutput,
           batch_size, output_width);
  tape.Add(dense.name + ".activation",
           CudaExecutionBufferRole::ActivationOutput, batch_size,
           output_width);
  tape.Add(norm.name + ".normalized",
           CudaExecutionBufferRole::NormalizedOutput, batch_size,
           output_width);
  return tape;
}

CudaWorkspaceSmokeResult RunExecutionTapeSmoke() {
  CudaWorkspaceSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  try {
    NetworkFormatDescriptor format;
    format.wdl = true;
    format.moves_left = true;
    format.attention_policy = true;
    const auto tensor_plan = CreateNetworkTensorPlan(format);

    NetworkResolvedExecutionPlan plan;
    plan.tensors = tensor_plan;
    NetworkResolvedExecutionStep dense;
    dense.kind = NetworkExecutionOpKind::Dense;
    dense.name = "smoke.dense";
    NetworkResolvedTensorRef dense_weight;
    dense_weight.name = "smoke.dense_w";
    dense_weight.elements = 6;
    dense_weight.dims = {2, 3};
    dense.tensors.push_back(dense_weight);

    NetworkResolvedExecutionStep norm;
    norm.kind = NetworkExecutionOpKind::LayerNorm;
    norm.name = "smoke.norm";
    NetworkResolvedTensorRef gamma;
    gamma.name = "smoke.gamma";
    gamma.elements = 2;
    gamma.dims = {2};
    norm.tensors.push_back(gamma);

    plan.steps.push_back(dense);
    plan.steps.push_back(norm);

    const auto tape = CreatePlanSmokeExecutionTape(tensor_plan, plan, 3);
    if (tape.BindingCount() != 3 || tape.TotalEntries() != 18 ||
        tape.RequireRole(CudaExecutionBufferRole::DenseOutput).width != 2 ||
        tape.RequireRole(CudaExecutionBufferRole::NormalizedOutput).rows != 3) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA execution tape binding mismatch";
      return result;
    }

    CudaExecutionWorkspace workspace;
    for (const auto &binding : tape.Bindings()) {
      if (!tape.Reserve(workspace, binding)) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA execution tape reserve returned null";
        return result;
      }
    }
    result.allocation_bytes = workspace.TotalBytes();
    if (workspace.NamedBufferCount() != 3 ||
        workspace.TotalCapacityFloats() != tape.TotalEntries()) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA execution tape workspace mismatch";
      return result;
    }
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

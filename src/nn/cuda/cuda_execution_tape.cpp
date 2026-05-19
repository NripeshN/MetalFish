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
  if (dense.kind != NetworkExecutionOpKind::Dense || dense.tensors.empty() ||
      dense.tensors[0].dims.size() != 2)
    throw std::runtime_error("CUDA execution tape dense tensor is invalid");
  const auto width = dense.tensors[0].dims[0];
  if (width == 0)
    throw std::runtime_error("CUDA execution tape dense width is zero");
  return static_cast<int>(width);
}

int LayerNormWidth(const NetworkResolvedExecutionStep &norm) {
  if (norm.kind != NetworkExecutionOpKind::LayerNorm || norm.tensors.empty() ||
      norm.tensors[0].dims.size() != 1)
    throw std::runtime_error("CUDA execution tape layernorm tensor is invalid");
  const auto width = norm.tensors[0].dims[0];
  if (width == 0)
    throw std::runtime_error("CUDA execution tape layernorm width is zero");
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

const CudaExecutionBufferBinding *
CudaExecutionTape::FindName(std::string_view name) const {
  for (const auto &binding : bindings_) {
    if (binding.name == name)
      return &binding;
  }
  return nullptr;
}

const CudaExecutionBufferBinding &
CudaExecutionTape::RequireName(std::string_view name) const {
  const CudaExecutionBufferBinding *binding = FindName(name);
  if (!binding)
    throw std::runtime_error("CUDA execution tape is missing buffer: " +
                             std::string(name));
  return *binding;
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

std::size_t CudaExecutionTape::CountRole(CudaExecutionBufferRole role) const {
  std::size_t count = 0;
  for (const auto &binding : bindings_) {
    if (binding.role == role)
      ++count;
  }
  return count;
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
  if (FindName(name))
    throw std::runtime_error("CUDA execution tape duplicate binding: " + name);
  bindings_.push_back(CudaExecutionBufferBinding{
      std::move(name), role, static_cast<std::size_t>(rows) * width, rows,
      width});
}

CudaExecutionTape CreateResolvedExecutionTape(
    const NetworkResolvedExecutionPlan &plan, int batch_size) {
  if (batch_size <= 0)
    throw std::runtime_error("CUDA execution tape batch size is invalid");

  CudaExecutionTape tape;
  for (const auto &step : plan.steps) {
    switch (step.kind) {
    case NetworkExecutionOpKind::Dense: {
      const int width = DenseOutputWidth(step);
      tape.Add(step.name + ".dense", CudaExecutionBufferRole::DenseOutput,
               batch_size, width);
      tape.Add(step.name + ".activation",
               CudaExecutionBufferRole::ActivationOutput, batch_size, width);
      break;
    }
    case NetworkExecutionOpKind::LayerNorm: {
      const int width = LayerNormWidth(step);
      tape.Add(step.name + ".normalized",
               CudaExecutionBufferRole::NormalizedOutput, batch_size, width);
      break;
    }
    default:
      break;
    }
  }
  return tape;
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

  return CreateResolvedExecutionTape(plan, batch_size);
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

    NetworkResolvedExecutionStep dense2;
    dense2.kind = NetworkExecutionOpKind::Dense;
    dense2.name = "smoke.dense2";
    NetworkResolvedTensorRef dense2_weight;
    dense2_weight.name = "smoke.dense2_w";
    dense2_weight.elements = 10;
    dense2_weight.dims = {5, 2};
    dense2.tensors.push_back(dense2_weight);

    NetworkResolvedExecutionStep norm2;
    norm2.kind = NetworkExecutionOpKind::LayerNorm;
    norm2.name = "smoke.norm2";
    NetworkResolvedTensorRef gamma2;
    gamma2.name = "smoke.gamma2";
    gamma2.elements = 5;
    gamma2.dims = {5};
    norm2.tensors.push_back(gamma2);

    NetworkResolvedExecutionPlan stacked = plan;
    stacked.steps.push_back(dense2);
    stacked.steps.push_back(norm2);
    const auto stacked_tape = CreateResolvedExecutionTape(stacked, 2);
    if (stacked_tape.BindingCount() != 6 ||
        stacked_tape.CountRole(CudaExecutionBufferRole::DenseOutput) != 2 ||
        stacked_tape.CountRole(CudaExecutionBufferRole::ActivationOutput) !=
            2 ||
        stacked_tape.CountRole(CudaExecutionBufferRole::NormalizedOutput) !=
            2 ||
        stacked_tape.RequireName("smoke.dense2.activation").width != 5 ||
        stacked_tape.TotalEntries() != 42) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA resolved execution tape sequence mismatch";
      return result;
    }

    CudaExecutionWorkspace stacked_workspace;
    for (const auto &binding : stacked_tape.Bindings()) {
      if (!stacked_tape.Reserve(stacked_workspace, binding)) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA resolved execution tape reserve returned null";
        return result;
      }
    }
    if (stacked_workspace.NamedBufferCount() != stacked_tape.BindingCount() ||
        stacked_workspace.TotalCapacityFloats() != stacked_tape.TotalEntries()) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA resolved execution tape workspace mismatch";
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

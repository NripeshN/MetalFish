/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_workspace.h"

#include "cuda_runtime_probe.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

std::string CudaErrorMessage(const char *op, cudaError_t status) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorString(status);
  return out.str();
}

std::size_t SlotIndex(CudaWorkspaceSlot slot) {
  const auto index = static_cast<std::size_t>(slot);
  if (index >= static_cast<std::size_t>(CudaWorkspaceSlot::Count))
    throw std::runtime_error("CUDA workspace slot is out of range");
  return index;
}

void FreeDevice(float *ptr) {
  if (ptr)
    cudaFree(ptr);
}

} // namespace

CudaExecutionWorkspace::~CudaExecutionWorkspace() { Release(); }

float *CudaExecutionWorkspace::ReserveFloats(CudaWorkspaceSlot slot,
                                             std::size_t entries) {
  if (entries == 0)
    return nullptr;

  const std::size_t index = SlotIndex(slot);
  if (capacities_[index] >= entries)
    return buffers_[index];

  float *next = nullptr;
  const cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(&next), entries * sizeof(float));
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaMalloc(workspace)", status));

  FreeDevice(buffers_[index]);
  buffers_[index] = next;
  capacities_[index] = entries;
  return buffers_[index];
}

std::size_t
CudaExecutionWorkspace::CapacityFloats(CudaWorkspaceSlot slot) const {
  return capacities_[SlotIndex(slot)];
}

std::size_t CudaExecutionWorkspace::TotalCapacityFloats() const {
  std::size_t total = 0;
  for (std::size_t capacity : capacities_)
    total += capacity;
  return total;
}

std::size_t CudaExecutionWorkspace::TotalBytes() const {
  return TotalCapacityFloats() * sizeof(float);
}

void CudaExecutionWorkspace::Release() {
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    FreeDevice(buffers_[i]);
    buffers_[i] = nullptr;
    capacities_[i] = 0;
  }
}

CudaWorkspaceSmokeResult RunExecutionWorkspaceSmoke() {
  CudaWorkspaceSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  try {
    CudaExecutionWorkspace workspace;
    float *dense = workspace.ReserveFloats(CudaWorkspaceSlot::Dense, 8);
    float *activation =
        workspace.ReserveFloats(CudaWorkspaceSlot::Activation, 4);
    float *dense_reused = workspace.ReserveFloats(CudaWorkspaceSlot::Dense, 4);
    float *norm = workspace.ReserveFloats(CudaWorkspaceSlot::Norm, 16);

    if (!dense || !activation || !norm || dense != dense_reused ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Dense) != 8 ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Activation) != 4 ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Norm) != 16) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA workspace capacity reuse mismatch";
      return result;
    }

    result.allocation_bytes = workspace.TotalBytes();
    if (result.allocation_bytes != (8 + 4 + 16) * sizeof(float)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA workspace allocation byte mismatch";
      return result;
    }

    workspace.Release();
    if (workspace.TotalBytes() != 0) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA workspace release mismatch";
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

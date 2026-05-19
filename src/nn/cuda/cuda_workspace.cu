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
#include <string_view>

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

void DestroyStream(cudaStream_t stream) {
  if (stream)
    cudaStreamDestroy(stream);
}

CudaNamedWorkspaceBuffer *
FindNamedBuffer(std::vector<CudaNamedWorkspaceBuffer> &buffers,
                std::string_view name) {
  for (auto &buffer : buffers) {
    if (buffer.name == name)
      return &buffer;
  }
  return nullptr;
}

const CudaNamedWorkspaceBuffer *FindNamedBuffer(
    const std::vector<CudaNamedWorkspaceBuffer> &buffers,
    std::string_view name) {
  for (const auto &buffer : buffers) {
    if (buffer.name == name)
      return &buffer;
  }
  return nullptr;
}

} // namespace

CudaExecutionWorkspace::~CudaExecutionWorkspace() {
  try {
    Release();
  } catch (...) {
  }
}

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

float *CudaExecutionWorkspace::ReserveNamedFloats(std::string_view name,
                                                  std::size_t entries) {
  if (name.empty())
    throw std::runtime_error("CUDA workspace named buffer is empty");
  if (entries == 0)
    return nullptr;

  CudaNamedWorkspaceBuffer *buffer = FindNamedBuffer(named_buffers_, name);
  if (!buffer) {
    named_buffers_.push_back(
        CudaNamedWorkspaceBuffer{std::string(name), nullptr, 0});
    buffer = &named_buffers_.back();
  }
  if (buffer->capacity >= entries)
    return buffer->buffer;

  float *next = nullptr;
  const cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(&next), entries * sizeof(float));
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("cudaMalloc(named_workspace)", status));

  FreeDevice(buffer->buffer);
  buffer->buffer = next;
  buffer->capacity = entries;
  return buffer->buffer;
}

cudaStream_t CudaExecutionWorkspace::Stream() {
  if (stream_)
    return stream_;

  const cudaError_t status =
      cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaStreamCreate", status));
  return stream_;
}

void CudaExecutionWorkspace::Synchronize() {
  if (!stream_)
    return;
  const cudaError_t status = cudaStreamSynchronize(stream_);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaStreamSynchronize", status));
}

std::size_t
CudaExecutionWorkspace::CapacityFloats(CudaWorkspaceSlot slot) const {
  return capacities_[SlotIndex(slot)];
}

std::size_t
CudaExecutionWorkspace::NamedCapacityFloats(std::string_view name) const {
  const CudaNamedWorkspaceBuffer *buffer =
      FindNamedBuffer(named_buffers_, name);
  return buffer ? buffer->capacity : 0;
}

std::size_t CudaExecutionWorkspace::NamedBufferCount() const {
  return named_buffers_.size();
}

std::size_t CudaExecutionWorkspace::TotalCapacityFloats() const {
  std::size_t total = 0;
  for (std::size_t capacity : capacities_)
    total += capacity;
  for (const auto &buffer : named_buffers_)
    total += buffer.capacity;
  return total;
}

std::size_t CudaExecutionWorkspace::TotalBytes() const {
  return TotalCapacityFloats() * sizeof(float);
}

void CudaExecutionWorkspace::Release() {
  if (stream_)
    cudaStreamSynchronize(stream_);
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    FreeDevice(buffers_[i]);
    buffers_[i] = nullptr;
    capacities_[i] = 0;
  }
  for (auto &buffer : named_buffers_) {
    FreeDevice(buffer.buffer);
    buffer.buffer = nullptr;
    buffer.capacity = 0;
  }
  named_buffers_.clear();
  DestroyStream(stream_);
  stream_ = nullptr;
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
    float *named = workspace.ReserveNamedFloats("smoke.step.output", 6);
    float *named_reused = workspace.ReserveNamedFloats("smoke.step.output", 2);
    cudaStream_t stream = workspace.Stream();

    if (!dense || !activation || !norm || dense != dense_reused ||
        !named || named != named_reused || !stream ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Dense) != 8 ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Activation) != 4 ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Norm) != 16 ||
        workspace.NamedCapacityFloats("smoke.step.output") != 6 ||
        workspace.NamedBufferCount() != 1) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA workspace capacity reuse mismatch";
      return result;
    }

    result.allocation_bytes = workspace.TotalBytes();
    if (result.allocation_bytes != (8 + 4 + 16 + 6) * sizeof(float)) {
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

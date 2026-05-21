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

void FreeDeviceBytes(void *ptr) {
  if (ptr)
    cudaFree(ptr);
}

void DestroyStream(cudaStream_t stream) {
  if (stream)
    cudaStreamDestroy(stream);
}

void ClearDeviceBytes(void *ptr, std::size_t bytes, const char *name,
                      cudaStream_t stream) {
  if (!ptr || bytes == 0)
    return;

  cudaError_t status = cudaSuccess;
  if (stream) {
    status = cudaMemsetAsync(ptr, 0, bytes, stream);
  } else {
    status = cudaMemset(ptr, 0, bytes);
  }
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
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

const CudaNamedWorkspaceBuffer *
FindNamedBuffer(const std::vector<CudaNamedWorkspaceBuffer> &buffers,
                std::string_view name) {
  for (const auto &buffer : buffers) {
    if (buffer.name == name)
      return &buffer;
  }
  return nullptr;
}

CudaNamedByteWorkspaceBuffer *
FindNamedByteBuffer(std::vector<CudaNamedByteWorkspaceBuffer> &buffers,
                    std::string_view name) {
  for (auto &buffer : buffers) {
    if (buffer.name == name)
      return &buffer;
  }
  return nullptr;
}

const CudaNamedByteWorkspaceBuffer *
FindNamedByteBuffer(const std::vector<CudaNamedByteWorkspaceBuffer> &buffers,
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
  ++generation_;
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
  ++generation_;
  return buffer->buffer;
}

void *CudaExecutionWorkspace::ReserveNamedBytes(std::string_view name,
                                                std::size_t bytes) {
  if (name.empty())
    throw std::runtime_error("CUDA workspace named byte buffer is empty");
  if (bytes == 0)
    return nullptr;

  CudaNamedByteWorkspaceBuffer *buffer =
      FindNamedByteBuffer(named_byte_buffers_, name);
  if (!buffer) {
    named_byte_buffers_.push_back(
        CudaNamedByteWorkspaceBuffer{std::string(name), nullptr, 0});
    buffer = &named_byte_buffers_.back();
  }
  if (buffer->capacity_bytes >= bytes)
    return buffer->buffer;

  void *next = nullptr;
  const cudaError_t status = cudaMalloc(&next, bytes);
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("cudaMalloc(named_byte_workspace)", status));

  FreeDeviceBytes(buffer->buffer);
  buffer->buffer = next;
  buffer->capacity_bytes = bytes;
  ++generation_;
  return buffer->buffer;
}

cudaStream_t CudaExecutionWorkspace::Stream() {
  if (stream_)
    return stream_;

  const cudaError_t status =
      cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaStreamCreate", status));
  ++generation_;
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

std::size_t
CudaExecutionWorkspace::NamedCapacityBytes(std::string_view name) const {
  const CudaNamedByteWorkspaceBuffer *buffer =
      FindNamedByteBuffer(named_byte_buffers_, name);
  return buffer ? buffer->capacity_bytes : 0;
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
  std::size_t total = TotalCapacityFloats() * sizeof(float);
  for (const auto &buffer : named_byte_buffers_)
    total += buffer.capacity_bytes;
  return total;
}

std::uint64_t CudaExecutionWorkspace::Generation() const {
  return generation_;
}

void CudaExecutionWorkspace::Clear(cudaStream_t stream) {
  for (std::size_t i = 0; i < buffers_.size(); ++i) {
    ClearDeviceBytes(buffers_[i], capacities_[i] * sizeof(float),
                     "cudaMemset(workspace)", stream);
  }
  for (auto &buffer : named_buffers_) {
    ClearDeviceBytes(buffer.buffer, buffer.capacity * sizeof(float),
                     "cudaMemset(named_workspace)", stream);
  }
  for (auto &buffer : named_byte_buffers_) {
    ClearDeviceBytes(buffer.buffer, buffer.capacity_bytes,
                     "cudaMemset(named_byte_workspace)", stream);
  }
}

void CudaExecutionWorkspace::Release() {
  const bool had_state = stream_ != nullptr || TotalBytes() > 0 ||
                         !named_buffers_.empty() ||
                         !named_byte_buffers_.empty();
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
  for (auto &buffer : named_byte_buffers_) {
    FreeDeviceBytes(buffer.buffer);
    buffer.buffer = nullptr;
    buffer.capacity_bytes = 0;
  }
  named_byte_buffers_.clear();
  DestroyStream(stream_);
  stream_ = nullptr;
  if (had_state)
    ++generation_;
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
    void *byte_buffer = workspace.ReserveNamedBytes("smoke.ptrs", 96);
    void *byte_buffer_reused = workspace.ReserveNamedBytes("smoke.ptrs", 48);
    cudaStream_t stream = workspace.Stream();

    if (!dense || !activation || !norm || dense != dense_reused || !named ||
        named != named_reused || !byte_buffer ||
        byte_buffer != byte_buffer_reused || !stream ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Dense) != 8 ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Activation) != 4 ||
        workspace.CapacityFloats(CudaWorkspaceSlot::Norm) != 16 ||
        workspace.NamedCapacityFloats("smoke.step.output") != 6 ||
        workspace.NamedCapacityBytes("smoke.ptrs") != 96 ||
        workspace.NamedBufferCount() != 1) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA workspace capacity reuse mismatch";
      return result;
    }

    result.allocation_bytes = workspace.TotalBytes();
    if (result.allocation_bytes != (8 + 4 + 16 + 6) * sizeof(float) + 96) {
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

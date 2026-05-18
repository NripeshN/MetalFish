/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_buffers.h"

#include "cuda_runtime_probe.h"

#include <cuda_runtime_api.h>

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

template <typename T>
void AllocateDevice(T **ptr, size_t entries, const char *name) {
  *ptr = nullptr;
  if (entries == 0)
    return;

  const cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(ptr), entries * sizeof(T));
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(name, status));
  }
}

template <typename T> void FreeDevice(T *ptr) {
  if (ptr)
    cudaFree(ptr);
}

void ValidateBatchSize(const CudaBufferLayout &layout, int batch_size) {
  if (batch_size <= 0 || batch_size > layout.max_batch_size)
    throw std::runtime_error("CUDA buffer batch size is out of range");
}

void ClearDeviceFloats(float *ptr, size_t entries, const char *name) {
  if (entries == 0)
    return;
  if (!ptr)
    throw std::runtime_error(std::string("CUDA output buffer is missing: ") +
                             name);
  const cudaError_t status = cudaMemset(ptr, 0, entries * sizeof(float));
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

void DownloadDeviceFloats(float *ptr, size_t entries,
                          std::vector<float> &host, const char *name) {
  host.assign(entries, 0.0f);
  if (entries == 0)
    return;
  if (!ptr)
    throw std::runtime_error(std::string("CUDA output buffer is missing: ") +
                             name);
  const cudaError_t status =
      cudaMemcpy(host.data(), ptr, entries * sizeof(float),
                 cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

bool AllZero(const std::vector<float> &values) {
  for (float value : values) {
    if (value != 0.0f)
      return false;
  }
  return true;
}

} // namespace

size_t CudaBufferLayout::InputPlaneEntries() const {
  return tensor_plan.InputMaskEntries(max_batch_size);
}

size_t CudaBufferLayout::PolicyEntries() const {
  return tensor_plan.PolicyEntries(max_batch_size);
}

size_t CudaBufferLayout::ValueEntries() const {
  return tensor_plan.ValueEntries(max_batch_size);
}

size_t CudaBufferLayout::MovesLeftEntries() const {
  return tensor_plan.MovesLeftEntries(max_batch_size);
}

size_t CudaBufferLayout::RawPolicyEntries() const {
  return tensor_plan.RawPolicyEntries(max_batch_size);
}

size_t CudaBufferLayout::TotalBytes() const {
  return InputPlaneEntries() * (sizeof(std::uint64_t) + sizeof(float)) +
         PolicyEntries() * sizeof(float) + ValueEntries() * sizeof(float) +
         MovesLeftEntries() * sizeof(float) +
         RawPolicyEntries() * sizeof(float);
}

CudaBufferLayout LayoutFromTensorPlan(const NetworkTensorPlan &plan,
                                      int max_batch_size) {
  CudaBufferLayout layout;
  layout.max_batch_size = max_batch_size;
  layout.tensor_plan = plan;
  layout.wdl = plan.wdl;
  layout.moves_left = plan.moves_left;
  layout.conv_policy = plan.conv_policy;
  layout.attention_policy = plan.attention_policy;
  return layout;
}

CudaBufferLayout LayoutFromNetworkFormat(const NetworkFormatDescriptor &format,
                                         int max_batch_size) {
  return LayoutFromTensorPlan(CreateNetworkTensorPlan(format), max_batch_size);
}

CudaInferenceBuffers::CudaInferenceBuffers(
    CudaInferenceBuffers &&other) noexcept {
  *this = std::move(other);
}

CudaInferenceBuffers &
CudaInferenceBuffers::operator=(CudaInferenceBuffers &&other) noexcept {
  if (this == &other)
    return *this;

  Release();

  layout_ = other.layout_;
  allocation_bytes_ = other.allocation_bytes_;
  input_masks = other.input_masks;
  input_values = other.input_values;
  policy = other.policy;
  value = other.value;
  moves_left = other.moves_left;
  raw_policy = other.raw_policy;

  other.layout_ = {};
  other.allocation_bytes_ = 0;
  other.input_masks = nullptr;
  other.input_values = nullptr;
  other.policy = nullptr;
  other.value = nullptr;
  other.moves_left = nullptr;
  other.raw_policy = nullptr;

  return *this;
}

CudaInferenceBuffers::~CudaInferenceBuffers() { Release(); }

void CudaInferenceBuffers::Allocate(const CudaBufferLayout &layout) {
  if (layout.max_batch_size <= 0) {
    throw std::runtime_error("CUDA buffer max batch size must be positive");
  }

  CudaInferenceBuffers next;
  next.layout_ = layout;
  next.allocation_bytes_ = layout.TotalBytes();

  try {
    AllocateDevice(&next.input_masks, layout.InputPlaneEntries(),
                   "cudaMalloc(input_masks)");
    AllocateDevice(&next.input_values, layout.InputPlaneEntries(),
                   "cudaMalloc(input_values)");
    AllocateDevice(&next.policy, layout.PolicyEntries(),
                   "cudaMalloc(policy)");
    AllocateDevice(&next.value, layout.ValueEntries(), "cudaMalloc(value)");
    AllocateDevice(&next.moves_left, layout.MovesLeftEntries(),
                   "cudaMalloc(moves_left)");
    AllocateDevice(&next.raw_policy, layout.RawPolicyEntries(),
                   "cudaMalloc(raw_policy)");
  } catch (...) {
    next.Release();
    throw;
  }

  *this = std::move(next);
}

void CudaInferenceBuffers::UploadPackedInputs(
    const std::vector<std::uint64_t> &masks, const std::vector<float> &values,
    int batch_size) {
  if (!input_masks || !input_values)
    throw std::runtime_error("CUDA inference buffers are not allocated");
  ValidateBatchSize(layout_, batch_size);

  const size_t entries =
      static_cast<size_t>(batch_size) * layout_.tensor_plan.input_planes;
  if (masks.size() != entries || values.size() != entries)
    throw std::runtime_error("CUDA input upload size mismatch");

  cudaError_t status =
      cudaMemcpy(input_masks, masks.data(), entries * sizeof(std::uint64_t),
                 cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaMemcpy(input_masks)",
                                             status));

  status = cudaMemcpy(input_values, values.data(), entries * sizeof(float),
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("cudaMemcpy(input_values)",
                                             status));
}

void CudaInferenceBuffers::ClearOutputs(int batch_size) {
  ValidateBatchSize(layout_, batch_size);
  ClearDeviceFloats(policy, layout_.tensor_plan.PolicyEntries(batch_size),
                    "cudaMemset(policy)");
  ClearDeviceFloats(value, layout_.tensor_plan.ValueEntries(batch_size),
                    "cudaMemset(value)");
  ClearDeviceFloats(moves_left,
                    layout_.tensor_plan.MovesLeftEntries(batch_size),
                    "cudaMemset(moves_left)");
  ClearDeviceFloats(raw_policy,
                    layout_.tensor_plan.RawPolicyEntries(batch_size),
                    "cudaMemset(raw_policy)");
}

CudaOutputDownload CudaInferenceBuffers::DownloadOutputs(
    int batch_size) const {
  ValidateBatchSize(layout_, batch_size);
  CudaOutputDownload output;
  DownloadDeviceFloats(policy, layout_.tensor_plan.PolicyEntries(batch_size),
                       output.policy, "cudaMemcpy(policy)");
  DownloadDeviceFloats(value, layout_.tensor_plan.ValueEntries(batch_size),
                       output.value, "cudaMemcpy(value)");
  DownloadDeviceFloats(moves_left,
                       layout_.tensor_plan.MovesLeftEntries(batch_size),
                       output.moves_left, "cudaMemcpy(moves_left)");
  DownloadDeviceFloats(raw_policy,
                       layout_.tensor_plan.RawPolicyEntries(batch_size),
                       output.raw_policy, "cudaMemcpy(raw_policy)");
  return output;
}

void CudaInferenceBuffers::Release() {
  FreeDevice(input_masks);
  FreeDevice(input_values);
  FreeDevice(policy);
  FreeDevice(value);
  FreeDevice(moves_left);
  FreeDevice(raw_policy);

  input_masks = nullptr;
  input_values = nullptr;
  policy = nullptr;
  value = nullptr;
  moves_left = nullptr;
  raw_policy = nullptr;
  layout_ = {};
  allocation_bytes_ = 0;
}

CudaBufferSmokeResult RunInferenceBufferSmoke() {
  CudaBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  CudaBufferLayout layout;
  NetworkFormatDescriptor format;
  format.wdl = true;
  format.moves_left = true;
  format.attention_policy = true;
  layout = LayoutFromNetworkFormat(format, 4);

  try {
    CudaInferenceBuffers buffers;
    buffers.Allocate(layout);
    buffers.ClearOutputs(4);
    result.allocation_bytes = buffers.AllocationBytes();
    if (!buffers.input_masks || !buffers.input_values || !buffers.policy ||
        !buffers.value || !buffers.moves_left || !buffers.raw_policy ||
        buffers.AllocationBytes() != layout.TotalBytes()) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA inference buffer layout mismatch";
      return result;
    }
    const auto output = buffers.DownloadOutputs(4);
    if (output.policy.size() != layout.PolicyEntries() ||
        output.value.size() != layout.ValueEntries() ||
        output.moves_left.size() != layout.MovesLeftEntries() ||
        output.raw_policy.size() != layout.RawPolicyEntries() ||
        !AllZero(output.policy) || !AllZero(output.value) ||
        !AllZero(output.moves_left) || !AllZero(output.raw_policy)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA output buffer clear/download mismatch";
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

CudaBufferSmokeResult RunPackedInputUploadSmokeRaw(const float *input) {
  CudaBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  std::vector<std::uint64_t> masks;
  std::vector<float> values;
  PackInputPlanesHostRaw(input, 1, masks, values);

  CudaBufferLayout layout;
  NetworkFormatDescriptor format;
  format.wdl = true;
  format.moves_left = true;
  format.attention_policy = true;
  layout = LayoutFromNetworkFormat(format, 2);

  try {
    CudaInferenceBuffers buffers;
    buffers.Allocate(layout);
    buffers.UploadPackedInputs(masks, values, 1);
    result.allocation_bytes = buffers.AllocationBytes();

    std::vector<std::uint64_t> actual_masks(masks.size());
    std::vector<float> actual_values(values.size());
    cudaError_t status =
        cudaMemcpy(actual_masks.data(), buffers.input_masks,
                   actual_masks.size() * sizeof(std::uint64_t),
                   cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      result.status = CudaSmokeStatus::RuntimeError;
      result.message = CudaErrorMessage("cudaMemcpy(actual_masks)", status);
      return result;
    }

    status = cudaMemcpy(actual_values.data(), buffers.input_values,
                        actual_values.size() * sizeof(float),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      result.status = CudaSmokeStatus::RuntimeError;
      result.message = CudaErrorMessage("cudaMemcpy(actual_values)", status);
      return result;
    }

    if (actual_masks != masks || actual_values != values) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA packed input upload mismatch";
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

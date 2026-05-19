/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_weight_buffers.h"

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

std::string CudaErrorMessage(const std::string &op, cudaError_t status) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorString(status);
  return out.str();
}

void FreeDevice(float *ptr) {
  if (ptr)
    cudaFree(ptr);
}

std::size_t ShapeElements(const std::vector<std::uint32_t> &dims) {
  if (dims.empty())
    return 0;
  std::size_t elements = 1;
  for (std::uint32_t dim : dims)
    elements *= dim;
  return elements;
}

} // namespace

CudaWeightBuffers::CudaWeightBuffers(CudaWeightBuffers &&other) noexcept {
  *this = std::move(other);
}

CudaWeightBuffers &
CudaWeightBuffers::operator=(CudaWeightBuffers &&other) noexcept {
  if (this == &other)
    return *this;

  Release();
  device_tensors_ = std::move(other.device_tensors_);
  allocation_bytes_ = other.allocation_bytes_;
  other.allocation_bytes_ = 0;
  return *this;
}

CudaWeightBuffers::~CudaWeightBuffers() { Release(); }

CudaDeviceTensorView CudaWeightBuffers::TensorAt(std::size_t index) const {
  if (index >= device_tensors_.size())
    throw std::out_of_range("CUDA weight tensor index is out of range");
  const auto &tensor = device_tensors_[index];
  return CudaDeviceTensorView{tensor.data, tensor.elements, tensor.dims,
                              tensor.kind};
}

void CudaWeightBuffers::Upload(const NetworkWeightInventory &inventory) {
  CudaWeightBuffers next;
  next.device_tensors_.reserve(inventory.tensors.size());

  try {
    for (const auto &tensor : inventory.tensors) {
      if (tensor.elements == 0)
        continue;
      const std::size_t shape_elements = ShapeElements(tensor.dims);
      if (shape_elements != 0 && shape_elements != tensor.elements) {
        throw std::runtime_error("CUDA weight tensor shape mismatch: " +
                                 tensor.name);
      }
      if (tensor.data == nullptr)
        throw std::runtime_error("CUDA weight tensor has null host data: " +
                                 tensor.name);

      DeviceTensor device_tensor;
      device_tensor.elements = tensor.elements;
      device_tensor.dims = tensor.dims;
      device_tensor.kind = tensor.kind;
      const cudaError_t alloc_status =
          cudaMalloc(reinterpret_cast<void **>(&device_tensor.data),
                     tensor.elements * sizeof(float));
      if (alloc_status != cudaSuccess) {
        throw std::runtime_error(
            CudaErrorMessage("cudaMalloc(" + tensor.name + ")", alloc_status));
      }

      const cudaError_t copy_status =
          cudaMemcpy(device_tensor.data, tensor.data,
                     tensor.elements * sizeof(float), cudaMemcpyHostToDevice);
      if (copy_status != cudaSuccess) {
        FreeDevice(device_tensor.data);
        throw std::runtime_error(
            CudaErrorMessage("cudaMemcpy(" + tensor.name + ")", copy_status));
      }

      next.allocation_bytes_ += tensor.elements * sizeof(float);
      next.device_tensors_.push_back(device_tensor);
    }
  } catch (...) {
    next.Release();
    throw;
  }

  *this = std::move(next);
}

bool CudaWeightBuffers::DownloadMatches(const NetworkWeightInventory &inventory,
                                        std::string *error) const {
  auto fail = [&](const std::string &message) {
    if (error)
      *error = message;
    return false;
  };

  if (device_tensors_.size() != inventory.tensors.size())
    return fail("CUDA weight tensor count mismatch");

  for (std::size_t i = 0; i < inventory.tensors.size(); ++i) {
    const auto &host = inventory.tensors[i];
    const auto &device = device_tensors_[i];
    if (device.elements != host.elements)
      return fail("CUDA weight tensor element count mismatch: " + host.name);
    if (device.dims != host.dims)
      return fail("CUDA weight tensor shape metadata mismatch: " + host.name);
    if (device.kind != host.kind)
      return fail("CUDA weight tensor kind metadata mismatch: " + host.name);
    if (host.elements == 0)
      continue;

    std::vector<float> downloaded(host.elements, 0.0f);
    cudaError_t status =
        cudaMemcpy(downloaded.data(), device.data,
                   host.elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
      return fail(
          CudaErrorMessage("cudaMemcpy(" + host.name + ")", status));
    for (std::size_t j = 0; j < host.elements; ++j) {
      if (downloaded[j] != host.data[j]) {
        return fail("CUDA weight value mismatch: " + host.name +
                    " at " + std::to_string(j));
      }
    }
  }

  return true;
}

void CudaWeightBuffers::Release() {
  for (auto &tensor : device_tensors_) {
    FreeDevice(tensor.data);
    tensor.data = nullptr;
    tensor.elements = 0;
  }
  device_tensors_.clear();
  allocation_bytes_ = 0;
}

CudaWeightBufferSmokeResult
RunWeightUploadSmoke(const NetworkWeightInventory &inventory) {
  CudaWeightBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  try {
    CudaWeightBuffers buffers;
    buffers.Upload(inventory);
    result.allocation_bytes = buffers.AllocationBytes();
    result.tensor_count = buffers.TensorCount();

    if (result.allocation_bytes != inventory.TotalBytes() ||
        result.tensor_count != inventory.tensors.size()) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA weight buffer allocation mismatch";
      return result;
    }

    std::string mismatch;
    if (!buffers.DownloadMatches(inventory, &mismatch)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = mismatch;
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

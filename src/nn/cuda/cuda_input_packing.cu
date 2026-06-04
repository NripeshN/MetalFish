/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_input_packing.h"

#include "cuda_runtime_probe.h"

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

__global__ void PackInputPlanesKernel(const float *input, std::uint64_t *masks,
                                      float *values, int batch_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_planes = batch_size * kCudaInputPlanes;
  if (idx >= total_planes)
    return;

  const int plane = idx % kCudaInputPlanes;
  const float *src = input + idx * kCudaSquares;
  PackInputPlaneRaw(src, plane, masks[idx], values[idx]);
}

std::string CudaErrorMessage(const char *op, cudaError_t status) {
  std::ostringstream out;
  out << op << " failed: " << cudaGetErrorString(status);
  return out.str();
}

} // namespace

void PackInputPlanesHostRaw(const float *inputs, int batch_size,
                            std::vector<std::uint64_t> &masks,
                            std::vector<float> &values) {
  PackInputPlanesRaw(inputs, batch_size, masks, values);
}

void PackInputPlaneBatchHostRaw(const std::vector<const float *> &inputs,
                                std::vector<std::uint64_t> &masks,
                                std::vector<float> &values) {
  if (inputs.empty()) {
    masks.clear();
    values.clear();
    return;
  }

  const size_t total_planes = inputs.size() * kCudaInputPlanes;
  masks.assign(total_planes, 0);
  values.assign(total_planes, 0.0f);

  for (size_t batch = 0; batch < inputs.size(); ++batch) {
    const float *planes = inputs[batch];
    if (!planes)
      throw std::runtime_error("CUDA host batch pack received null input");
    for (int plane = 0; plane < kCudaInputPlanes; ++plane) {
      const size_t index =
          batch * kCudaInputPlanes + static_cast<size_t>(plane);
      PackInputPlaneRaw(planes + static_cast<size_t>(plane) * kCudaSquares,
                        plane, masks[index], values[index]);
    }
  }
}

CudaInputPackingSmokeResult RunInputPackingSmokeRaw(const float *input) {
  CudaInputPackingSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  std::vector<std::uint64_t> expected_masks;
  std::vector<float> expected_values;
  PackInputPlanesHostRaw(input, 1, expected_masks, expected_values);

  std::vector<float> flat_input(input, input + kCudaInputPlanes * kCudaSquares);

  float *device_input = nullptr;
  std::uint64_t *device_masks = nullptr;
  float *device_values = nullptr;
  const size_t input_bytes = flat_input.size() * sizeof(float);
  const size_t packed_bytes = expected_masks.size() * sizeof(std::uint64_t);
  const size_t values_bytes = expected_values.size() * sizeof(float);

  auto cleanup = [&]() {
    if (device_input)
      cudaFree(device_input);
    if (device_masks)
      cudaFree(device_masks);
    if (device_values)
      cudaFree(device_values);
  };

  cudaError_t status = cudaMalloc(&device_input, input_bytes);
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("cudaMalloc(input)", status);
    cleanup();
    return result;
  }
  status = cudaMalloc(&device_masks, packed_bytes);
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("cudaMalloc(masks)", status);
    cleanup();
    return result;
  }
  status = cudaMalloc(&device_values, values_bytes);
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("cudaMalloc(values)", status);
    cleanup();
    return result;
  }

  status = cudaMemcpy(device_input, flat_input.data(), input_bytes,
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("cudaMemcpy(input)", status);
    cleanup();
    return result;
  }

  const int total_planes = kCudaInputPlanes;
  const int block = 128;
  const int grid = (total_planes + block - 1) / block;
  PackInputPlanesKernel<<<grid, block>>>(device_input, device_masks,
                                         device_values, 1);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("PackInputPlanesKernel", status);
    cleanup();
    return result;
  }
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("cudaDeviceSynchronize", status);
    cleanup();
    return result;
  }

  std::vector<std::uint64_t> actual_masks(expected_masks.size());
  std::vector<float> actual_values(expected_values.size());
  status = cudaMemcpy(actual_masks.data(), device_masks, packed_bytes,
                      cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("cudaMemcpy(masks)", status);
    cleanup();
    return result;
  }
  status = cudaMemcpy(actual_values.data(), device_values, values_bytes,
                      cudaMemcpyDeviceToHost);
  cleanup();
  if (status != cudaSuccess) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = CudaErrorMessage("cudaMemcpy(values)", status);
    return result;
  }

  if (actual_masks != expected_masks || actual_values != expected_values) {
    result.status = CudaSmokeStatus::Mismatch;
    result.message = "CUDA input packing mismatch";
    return result;
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

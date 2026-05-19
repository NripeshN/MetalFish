/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_kernels.h"

#include "cuda_runtime_probe.h"

#include <cuda_runtime_api.h>

#include <cmath>
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

template <typename T>
void AllocateDevice(T **ptr, std::size_t entries, const char *name) {
  *ptr = nullptr;
  if (entries == 0)
    return;
  const cudaError_t status =
      cudaMalloc(reinterpret_cast<void **>(ptr), entries * sizeof(T));
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

template <typename T> void FreeDevice(T *ptr) {
  if (ptr)
    cudaFree(ptr);
}

void UploadFloats(float *device, const std::vector<float> &host,
                  const char *name) {
  if (host.empty())
    return;
  const cudaError_t status = cudaMemcpy(
      device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

void DownloadFloats(std::vector<float> &host, const float *device,
                    const char *name) {
  if (host.empty())
    return;
  const cudaError_t status =
      cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                 cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

__global__ void DenseAffineKernel(const float *input, const float *weights,
                                  const float *bias, float *output,
                                  int input_width, int output_width,
                                  int total_outputs) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_outputs)
    return;

  const int out = index % output_width;
  const int batch = index / output_width;
  const float *input_row = input + static_cast<std::size_t>(batch) * input_width;
  const float *weight_row = weights + static_cast<std::size_t>(out) * input_width;

  float sum = bias ? bias[out] : 0.0f;
  for (int i = 0; i < input_width; ++i)
    sum += input_row[i] * weight_row[i];
  output[index] = sum;
}

} // namespace

void LaunchDenseAffineKernel(const float *input, const float *weights,
                             const float *bias, float *output, int batch_size,
                             int input_width, int output_width) {
  if (!input || !weights || !output)
    throw std::runtime_error("CUDA dense affine kernel received null buffer");
  if (batch_size <= 0 || input_width <= 0 || output_width <= 0)
    throw std::runtime_error("CUDA dense affine kernel dimensions are invalid");

  const int total_outputs = batch_size * output_width;
  constexpr int kThreads = 256;
  const int blocks = (total_outputs + kThreads - 1) / kThreads;
  DenseAffineKernel<<<blocks, kThreads>>>(input, weights, bias, output,
                                          input_width, output_width,
                                          total_outputs);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("DenseAffineKernel launch",
                                             status));
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("DenseAffineKernel synchronize", status));
}

CudaKernelSmokeResult RunDenseAffineKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kInput = 3;
  constexpr int kOutput = 4;
  const std::vector<float> input = {
      1.0f, 2.0f, 3.0f,
      -1.0f, 0.5f, 2.0f,
  };
  const std::vector<float> weights = {
      1.0f, 0.0f, -1.0f,
      0.5f, 0.5f, 0.5f,
      -2.0f, 1.0f, 0.25f,
      0.0f, -1.0f, 2.0f,
  };
  const std::vector<float> bias = {0.25f, -1.0f, 0.5f, 2.0f};
  std::vector<float> actual(kBatch * kOutput, 0.0f);
  std::vector<float> expected(kBatch * kOutput, 0.0f);

  for (int b = 0; b < kBatch; ++b) {
    for (int o = 0; o < kOutput; ++o) {
      float sum = bias[o];
      for (int i = 0; i < kInput; ++i) {
        sum += input[static_cast<std::size_t>(b) * kInput + i] *
               weights[static_cast<std::size_t>(o) * kInput + i];
      }
      expected[static_cast<std::size_t>(b) * kOutput + o] = sum;
    }
  }

  float *device_input = nullptr;
  float *device_weights = nullptr;
  float *device_bias = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_input, input.size(), "cudaMalloc(dense_input)");
    AllocateDevice(&device_weights, weights.size(), "cudaMalloc(dense_weights)");
    AllocateDevice(&device_bias, bias.size(), "cudaMalloc(dense_bias)");
    AllocateDevice(&device_output, actual.size(), "cudaMalloc(dense_output)");

    UploadFloats(device_input, input, "cudaMemcpy(dense_input)");
    UploadFloats(device_weights, weights, "cudaMemcpy(dense_weights)");
    UploadFloats(device_bias, bias, "cudaMemcpy(dense_bias)");

    LaunchDenseAffineKernel(device_input, device_weights, device_bias,
                            device_output, kBatch, kInput, kOutput);
    DownloadFloats(actual, device_output, "cudaMemcpy(dense_output)");
  } catch (const std::exception &e) {
    FreeDevice(device_input);
    FreeDevice(device_weights);
    FreeDevice(device_bias);
    FreeDevice(device_output);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_input);
  FreeDevice(device_weights);
  FreeDevice(device_bias);
  FreeDevice(device_output);

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-5f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dense affine kernel output mismatch";
      return result;
    }
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

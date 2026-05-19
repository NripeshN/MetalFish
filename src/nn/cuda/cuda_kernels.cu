/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_kernels.h"

#include "cuda_runtime_probe.h"

#include <cuda_runtime_api.h>

#include <algorithm>
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

__global__ void LayerNormKernel(const float *input, const float *gamma,
                                const float *beta, float *output, int width,
                                float epsilon) {
  extern __shared__ float reductions[];
  float *sum_storage = reductions;
  float *square_storage = reductions + blockDim.x;

  const int row = blockIdx.x;
  const float *input_row = input + static_cast<std::size_t>(row) * width;
  float *output_row = output + static_cast<std::size_t>(row) * width;

  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = input_row[col];
    sum += value;
    square_sum += value * value;
  }

  sum_storage[threadIdx.x] = sum;
  square_storage[threadIdx.x] = square_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sum_storage[threadIdx.x] += sum_storage[threadIdx.x + stride];
      square_storage[threadIdx.x] += square_storage[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float mean = sum_storage[0] / static_cast<float>(width);
  float variance = square_storage[0] / static_cast<float>(width) - mean * mean;
  if (variance < 0.0f)
    variance = 0.0f;
  const float inv_std = rsqrtf(variance + epsilon);

  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float normalized = (input_row[col] - mean) * inv_std;
    output_row[col] = normalized * gamma[col] + beta[col];
  }
}

__device__ float ApplyActivationValue(float value, CudaActivationKind kind) {
  switch (kind) {
  case CudaActivationKind::Relu:
    return fmaxf(value, 0.0f);
  case CudaActivationKind::Relu2: {
    const float relu = fmaxf(value, 0.0f);
    return relu * relu;
  }
  case CudaActivationKind::Tanh:
    return tanhf(value);
  case CudaActivationKind::Sigmoid:
    return 1.0f / (1.0f + expf(-value));
  case CudaActivationKind::Swish:
    return value / (1.0f + expf(-value));
  case CudaActivationKind::Mish:
    return value * tanhf(log1pf(expf(value)));
  case CudaActivationKind::Selu:
    return 1.05070098f *
           (value > 0.0f ? value : 1.67326324f * (expf(value) - 1.0f));
  }
  return value;
}

__global__ void ActivationKernel(const float *input, float *output,
                                 int elements, CudaActivationKind kind) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements)
    return;
  output[index] = ApplyActivationValue(input[index], kind);
}

__global__ void GateKernel(const float *input, const float *weights,
                           float *output, int width, int total,
                           CudaGateKind kind) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int column = index % width;
  const float gate = weights[column];
  const float value = input[index];
  output[index] =
      kind == CudaGateKind::Add ? value + gate : value * gate;
}

} // namespace

void LaunchDenseAffineKernel(const float *input, const float *weights,
                             const float *bias, float *output, int batch_size,
                             int input_width, int output_width,
                             cudaStream_t stream) {
  if (!input || !weights || !output)
    throw std::runtime_error("CUDA dense affine kernel received null buffer");
  if (batch_size <= 0 || input_width <= 0 || output_width <= 0)
    throw std::runtime_error("CUDA dense affine kernel dimensions are invalid");

  const int total_outputs = batch_size * output_width;
  constexpr int kThreads = 256;
  const int blocks = (total_outputs + kThreads - 1) / kThreads;
  DenseAffineKernel<<<blocks, kThreads, 0, stream>>>(
      input, weights, bias, output, input_width, output_width, total_outputs);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("DenseAffineKernel launch",
                                             status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("DenseAffineKernel synchronize", status));
}

void LaunchLayerNormKernel(const float *input, const float *gamma,
                           const float *beta, float *output, int rows,
                           int width, float epsilon, cudaStream_t stream) {
  if (!input || !gamma || !beta || !output)
    throw std::runtime_error("CUDA layernorm kernel received null buffer");
  if (rows <= 0 || width <= 0 || epsilon <= 0.0f)
    throw std::runtime_error("CUDA layernorm kernel dimensions are invalid");

  constexpr int kThreads = 256;
  const std::size_t shared_bytes = 2 * kThreads * sizeof(float);
  LayerNormKernel<<<rows, kThreads, shared_bytes, stream>>>(
      input, gamma, beta, output, width, epsilon);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("LayerNormKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("LayerNormKernel synchronize", status));
}

void LaunchActivationKernel(const float *input, float *output, int elements,
                            CudaActivationKind kind, cudaStream_t stream) {
  if (!input || !output)
    throw std::runtime_error("CUDA activation kernel received null buffer");
  if (elements <= 0)
    throw std::runtime_error("CUDA activation kernel dimensions are invalid");

  constexpr int kThreads = 256;
  const int blocks = (elements + kThreads - 1) / kThreads;
  ActivationKernel<<<blocks, kThreads, 0, stream>>>(input, output, elements,
                                                   kind);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("ActivationKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("ActivationKernel synchronize", status));
}

void LaunchGateKernel(const float *input, const float *weights, float *output,
                      int batch_size, int width, CudaGateKind kind,
                      cudaStream_t stream) {
  if (!input || !weights || !output)
    throw std::runtime_error("CUDA gate kernel received null buffer");
  if (batch_size <= 0 || width <= 0)
    throw std::runtime_error("CUDA gate kernel dimensions are invalid");

  const int total = batch_size * width;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  GateKernel<<<blocks, kThreads, 0, stream>>>(input, weights, output, width,
                                             total, kind);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("GateKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("GateKernel synchronize",
                                             status));
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

CudaKernelSmokeResult RunActivationKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  const std::vector<float> input = {-3.0f, -1.0f, -0.25f, 0.0f,
                                    0.25f, 1.0f, 3.0f};
  const std::vector<CudaActivationKind> activations = {
      CudaActivationKind::Relu,    CudaActivationKind::Relu2,
      CudaActivationKind::Tanh,    CudaActivationKind::Sigmoid,
      CudaActivationKind::Swish,   CudaActivationKind::Mish,
      CudaActivationKind::Selu,
  };

  float *device_input = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_input, input.size(), "cudaMalloc(activation_input)");
    AllocateDevice(&device_output, input.size(),
                   "cudaMalloc(activation_output)");
    UploadFloats(device_input, input, "cudaMemcpy(activation_input)");

    for (CudaActivationKind activation : activations) {
      std::vector<float> actual(input.size(), 0.0f);
      std::vector<float> expected(input.size(), 0.0f);

      for (std::size_t i = 0; i < input.size(); ++i) {
        const float value = input[i];
        switch (activation) {
        case CudaActivationKind::Relu:
          expected[i] = std::max(value, 0.0f);
          break;
        case CudaActivationKind::Relu2: {
          const float relu = std::max(value, 0.0f);
          expected[i] = relu * relu;
          break;
        }
        case CudaActivationKind::Tanh:
          expected[i] = std::tanh(value);
          break;
        case CudaActivationKind::Sigmoid:
          expected[i] = 1.0f / (1.0f + std::exp(-value));
          break;
        case CudaActivationKind::Swish:
          expected[i] = value / (1.0f + std::exp(-value));
          break;
        case CudaActivationKind::Mish:
          expected[i] = value * std::tanh(std::log1p(std::exp(value)));
          break;
        case CudaActivationKind::Selu:
          expected[i] =
              1.05070098f *
              (value > 0.0f ? value : 1.67326324f * (std::exp(value) - 1.0f));
          break;
        }
      }

      LaunchActivationKernel(device_input, device_output,
                             static_cast<int>(input.size()), activation);
      DownloadFloats(actual, device_output, "cudaMemcpy(activation_output)");

      for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > 1e-5f) {
          FreeDevice(device_input);
          FreeDevice(device_output);
          result.status = CudaSmokeStatus::Mismatch;
          result.message = "CUDA activation kernel output mismatch";
          return result;
        }
      }
    }
  } catch (const std::exception &e) {
    FreeDevice(device_input);
    FreeDevice(device_output);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_input);
  FreeDevice(device_output);

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunLayerNormKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kRows = 3;
  constexpr int kWidth = 5;
  constexpr float kEpsilon = 1e-5f;
  const std::vector<float> input = {
      1.0f, 2.0f, 4.0f, 8.0f, 16.0f,
      -3.0f, -1.0f, 0.0f, 1.0f, 3.0f,
      0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
  };
  const std::vector<float> gamma = {1.0f, 0.5f, -1.5f, 2.0f, 0.25f};
  const std::vector<float> beta = {0.0f, -0.25f, 0.5f, 1.0f, -1.0f};
  std::vector<float> actual(kRows * kWidth, 0.0f);
  std::vector<float> expected(kRows * kWidth, 0.0f);

  for (int row = 0; row < kRows; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * kWidth;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int col = 0; col < kWidth; ++col) {
      const float value = input[row_offset + col];
      sum += value;
      square_sum += value * value;
    }
    const float mean = sum / static_cast<float>(kWidth);
    float variance = square_sum / static_cast<float>(kWidth) - mean * mean;
    if (variance < 0.0f)
      variance = 0.0f;
    const float inv_std = 1.0f / std::sqrt(variance + kEpsilon);
    for (int col = 0; col < kWidth; ++col) {
      const float normalized = (input[row_offset + col] - mean) * inv_std;
      expected[row_offset + col] = normalized * gamma[col] + beta[col];
    }
  }

  float *device_input = nullptr;
  float *device_gamma = nullptr;
  float *device_beta = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_input, input.size(), "cudaMalloc(layernorm_input)");
    AllocateDevice(&device_gamma, gamma.size(), "cudaMalloc(layernorm_gamma)");
    AllocateDevice(&device_beta, beta.size(), "cudaMalloc(layernorm_beta)");
    AllocateDevice(&device_output, actual.size(),
                   "cudaMalloc(layernorm_output)");

    UploadFloats(device_input, input, "cudaMemcpy(layernorm_input)");
    UploadFloats(device_gamma, gamma, "cudaMemcpy(layernorm_gamma)");
    UploadFloats(device_beta, beta, "cudaMemcpy(layernorm_beta)");

    LaunchLayerNormKernel(device_input, device_gamma, device_beta,
                          device_output, kRows, kWidth, kEpsilon);
    DownloadFloats(actual, device_output, "cudaMemcpy(layernorm_output)");
  } catch (const std::exception &e) {
    FreeDevice(device_input);
    FreeDevice(device_gamma);
    FreeDevice(device_beta);
    FreeDevice(device_output);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_input);
  FreeDevice(device_gamma);
  FreeDevice(device_beta);
  FreeDevice(device_output);

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-5f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA layernorm kernel output mismatch";
      return result;
    }
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunGateKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kWidth = 4;
  const std::vector<float> input = {
      1.0f, -2.0f, 0.5f, 3.0f,
      -1.0f, 0.25f, 2.0f, -0.5f,
  };
  const std::vector<float> mult_gate = {2.0f, -0.5f, 4.0f, 0.25f};
  const std::vector<float> add_gate = {0.5f, 1.0f, -2.0f, 3.0f};
  std::vector<float> actual(kBatch * kWidth, 0.0f);
  std::vector<float> expected(kBatch * kWidth, 0.0f);

  for (int b = 0; b < kBatch; ++b) {
    for (int i = 0; i < kWidth; ++i) {
      const std::size_t index = static_cast<std::size_t>(b) * kWidth + i;
      expected[index] = input[index] * mult_gate[static_cast<std::size_t>(i)] +
                        add_gate[static_cast<std::size_t>(i)];
    }
  }

  float *device_input = nullptr;
  float *device_mult = nullptr;
  float *device_add = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_input, input.size(), "cudaMalloc(gate_input)");
    AllocateDevice(&device_mult, mult_gate.size(), "cudaMalloc(gate_mult)");
    AllocateDevice(&device_add, add_gate.size(), "cudaMalloc(gate_add)");
    AllocateDevice(&device_output, actual.size(), "cudaMalloc(gate_output)");
    UploadFloats(device_input, input, "cudaMemcpy(gate_input)");
    UploadFloats(device_mult, mult_gate, "cudaMemcpy(gate_mult)");
    UploadFloats(device_add, add_gate, "cudaMemcpy(gate_add)");

    LaunchGateKernel(device_input, device_mult, device_output, kBatch, kWidth,
                     CudaGateKind::Multiply);
    LaunchGateKernel(device_output, device_add, device_output, kBatch, kWidth,
                     CudaGateKind::Add);
    DownloadFloats(actual, device_output, "cudaMemcpy(gate_output)");

    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA gate kernel output mismatch";
        break;
      }
    }
    if (result.message.empty())
      result.status = CudaSmokeStatus::Success;
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
  }

  FreeDevice(device_input);
  FreeDevice(device_mult);
  FreeDevice(device_add);
  FreeDevice(device_output);

  if (result.message.empty())
    result.message = RuntimeCudaDeviceSummary();
  return result;
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

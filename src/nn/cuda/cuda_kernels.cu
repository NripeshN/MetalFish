/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_kernels.h"

#include "../network_tensor_plan.h"
#include "../tables/attention_policy_map.h"
#include "../tables/conv_policy_map.h"
#include "cuda_kernel_smoke.h"
#include "cuda_runtime_probe.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
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

const char *CublasStatusName(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "success";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "not_initialized";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "alloc_failed";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "invalid_value";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "arch_mismatch";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "mapping_error";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "execution_failed";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "internal_error";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "not_supported";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "license_error";
  }
  return "unknown";
}

std::string CublasErrorMessage(const char *op, cublasStatus_t status) {
  std::ostringstream out;
  out << op << " failed: " << CublasStatusName(status);
  return out.str();
}

class ThreadLocalCublasHandle {
public:
  ThreadLocalCublasHandle() {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(CublasErrorMessage("cublasCreate", status));

    status = cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(
          CublasErrorMessage("cublasSetPointerMode", status));

#if CUDART_VERSION >= 11000
    status = cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH);
#else
    status = cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH);
#endif
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error(CublasErrorMessage("cublasSetMathMode", status));

    status = cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(
          CublasErrorMessage("cublasSetAtomicsMode", status));
    }
  }

  ThreadLocalCublasHandle(const ThreadLocalCublasHandle &) = delete;
  ThreadLocalCublasHandle &operator=(const ThreadLocalCublasHandle &) = delete;

  ~ThreadLocalCublasHandle() {
    if (handle_)
      cublasDestroy(handle_);
  }

  cublasHandle_t Get() const { return handle_; }

private:
  cublasHandle_t handle_ = nullptr;
};

cublasHandle_t CublasHandle() {
  static thread_local ThreadLocalCublasHandle handle;
  return handle.Get();
}

const std::array<int, kNetworkPolicyOutputs> &AttentionPolicyGatherMap() {
  static const auto gather = [] {
    std::array<int, kNetworkPolicyOutputs> indices{};
    indices.fill(-1);
    for (int raw = 0; raw < kNetworkAttentionPolicyScratch; ++raw) {
      const short mapped = Tables::kAttnPolicyMap[raw];
      if (mapped >= 0)
        indices[static_cast<std::size_t>(mapped)] = raw;
    }
    return indices;
  }();
  return gather;
}

const std::array<int, kNetworkPolicyOutputs> &ConvolutionPolicyGatherMap() {
  static const auto gather = [] {
    std::array<int, kNetworkPolicyOutputs> indices{};
    indices.fill(-1);
    for (int raw = 0; raw < kNetworkConvPolicyScratch; ++raw) {
      const short mapped = Tables::kConvPolicyMap[raw];
      if (mapped >= 0)
        indices[static_cast<std::size_t>(mapped)] = raw;
    }
    return indices;
  }();
  return gather;
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
  const cudaError_t status = cudaMemcpy(
      host.data(), device, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage(name, status));
}

__global__ void BiasAddKernel(float *output, const float *bias,
                              int output_width, int total_outputs) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total_outputs)
    return;
  output[index] += bias[index % output_width];
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

__global__ void BiasActivationKernel(float *input, const float *bias,
                                     float *output, int width, int total,
                                     CudaActivationKind kind) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;
  const float value = input[index] + bias[index % width];
  input[index] = value;
  output[index] = ApplyActivationValue(value, kind);
}

__global__ void Convolution2DKernel(const float *input, const float *weights,
                                    const float *bias, float *output,
                                    int squares, int input_channels,
                                    int output_channels, int kernel_size,
                                    CudaActivationKind activation,
                                    bool apply_activation, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  constexpr int kBoardSide = 8;
  const int square = index % squares;
  const int out_channel = (index / squares) % output_channels;
  const int batch = index / (output_channels * squares);
  const int rank = square / kBoardSide;
  const int file = square % kBoardSide;
  const int radius = kernel_size / 2;

  float sum = bias ? bias[out_channel] : 0.0f;
  for (int in_channel = 0; in_channel < input_channels; ++in_channel) {
    for (int ky = 0; ky < kernel_size; ++ky) {
      const int src_rank = rank + ky - radius;
      if (src_rank < 0 || src_rank >= kBoardSide)
        continue;
      for (int kx = 0; kx < kernel_size; ++kx) {
        const int src_file = file + kx - radius;
        if (src_file < 0 || src_file >= kBoardSide)
          continue;
        const int src_square = src_rank * kBoardSide + src_file;
        const std::size_t input_index =
            (static_cast<std::size_t>(batch) * input_channels + in_channel) *
                squares +
            src_square;
        const std::size_t weight_index =
            (((static_cast<std::size_t>(out_channel) * input_channels +
               in_channel) *
                  kernel_size +
              ky) *
                 kernel_size +
             kx);
        sum += input[input_index] * weights[weight_index];
      }
    }
  }
  output[index] =
      apply_activation ? ApplyActivationValue(sum, activation) : sum;
}

__global__ void GateKernel(const float *input, const float *weights,
                           float *output, int width, int total, int gate_rows,
                           CudaGateKind kind) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int row = index / width;
  const int column = index % width;
  const int gate_row = gate_rows == 1 ? 0 : row % gate_rows;
  const float gate =
      gate_rows == 1 ? weights[column] : weights[column * gate_rows + gate_row];
  const float value = input[index];
  output[index] = kind == CudaGateKind::Add ? value + gate : value * gate;
}

__global__ void ResidualAddKernel(const float *parent, const float *secondary,
                                  float *output, int total,
                                  float secondary_scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;
  output[index] = parent[index] + secondary[index] * secondary_scale;
}

__global__ void GlobalAveragePoolNchwKernel(const float *input, float *output,
                                            int channels, int squares,
                                            int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;
  const int batch = index / channels;
  const int channel = index % channels;
  const std::size_t offset =
      (static_cast<std::size_t>(batch) * channels + channel) * squares;
  float sum = 0.0f;
  for (int square = 0; square < squares; ++square)
    sum += input[offset + square];
  output[index] = sum / static_cast<float>(squares);
}

__global__ void
SqueezeExciteResidualKernel(const float *skip, const float *convolution,
                            const float *se_output, float *residual,
                            float *output, int channels, int squares, int total,
                            CudaActivationKind activation) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;
  const int channel = (index / squares) % channels;
  const int batch = index / (channels * squares);
  const int se_offset = batch * channels * 2 + channel;
  const float gamma = 1.0f / (1.0f + expf(-se_output[se_offset]));
  const float beta = se_output[se_offset + channels];
  const float value = skip[index] + convolution[index] * gamma + beta;
  residual[index] = value;
  output[index] = ApplyActivationValue(value, activation);
}

__global__ void ResidualLayerNormKernel(const float *parent,
                                        const float *secondary,
                                        const float *gamma, const float *beta,
                                        float *residual, float *output,
                                        int width, float secondary_scale,
                                        float epsilon) {
  extern __shared__ float reductions[];
  float *sum_storage = reductions;
  float *square_storage = reductions + blockDim.x;

  const int row = blockIdx.x;
  const std::size_t row_offset = static_cast<std::size_t>(row) * width;
  const float *parent_row = parent + row_offset;
  const float *secondary_row = secondary + row_offset;
  float *residual_row = residual + row_offset;
  float *output_row = output + row_offset;

  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = parent_row[col] + secondary_row[col] * secondary_scale;
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
    const float value = parent_row[col] + secondary_row[col] * secondary_scale;
    residual_row[col] = value;
    const float normalized = (value - mean) * inv_std;
    output_row[col] = normalized * gamma[col] + beta[col];
  }
}

__global__ void
ResidualBiasLayerNormKernel(const float *parent, const float *secondary,
                            const float *secondary_bias, const float *gamma,
                            const float *beta, float *biased_secondary,
                            float *residual, float *output, int width,
                            float secondary_scale, float epsilon) {
  extern __shared__ float reductions[];
  float *sum_storage = reductions;
  float *square_storage = reductions + blockDim.x;

  const int row = blockIdx.x;
  const std::size_t row_offset = static_cast<std::size_t>(row) * width;
  const float *parent_row = parent + row_offset;
  const float *secondary_row = secondary + row_offset;
  float *biased_secondary_row = biased_secondary + row_offset;
  float *residual_row = residual + row_offset;
  float *output_row = output + row_offset;

  float sum = 0.0f;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float biased = secondary_row[col] + secondary_bias[col];
    const float value = parent_row[col] + biased * secondary_scale;
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
    const float biased = secondary_row[col] + secondary_bias[col];
    const float value = parent_row[col] + biased * secondary_scale;
    biased_secondary_row[col] = biased;
    residual_row[col] = value;
    const float normalized = (value - mean) * inv_std;
    output_row[col] = normalized * gamma[col] + beta[col];
  }
}

__global__ void AttentionBiasAddKernel(float *scores, const float *bias,
                                       int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;
  scores[index] += bias[index];
}

__global__ void AttentionSoftmaxKernel(const float *scores,
                                       float *probabilities, int width) {
  extern __shared__ float reductions[];
  const int row = blockIdx.x;
  const float *score_row = scores + static_cast<std::size_t>(row) * width;
  float *probability_row =
      probabilities + static_cast<std::size_t>(row) * width;

  float max_value = -3.4028234663852886e+38F;
  for (int col = threadIdx.x; col < width; col += blockDim.x)
    max_value = fmaxf(max_value, score_row[col]);
  reductions[threadIdx.x] = max_value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] =
          fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  max_value = reductions[0];

  float sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = expf(score_row[col] - max_value);
    probability_row[col] = value;
    sum += value;
  }
  reductions[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  sum = reductions[0];

  for (int col = threadIdx.x; col < width; col += blockDim.x)
    probability_row[col] = sum > 0.0f ? probability_row[col] / sum : 0.0f;
}

__global__ void AttentionBiasSoftmaxKernel(float *scores, const float *bias,
                                           float *probabilities, int width) {
  extern __shared__ float reductions[];
  const int row = blockIdx.x;
  float *score_row = scores + static_cast<std::size_t>(row) * width;
  const float *bias_row = bias + static_cast<std::size_t>(row) * width;
  float *probability_row =
      probabilities + static_cast<std::size_t>(row) * width;

  for (int col = threadIdx.x; col < width; col += blockDim.x)
    score_row[col] += bias_row[col];
  __syncthreads();

  float max_value = -3.4028234663852886e+38F;
  for (int col = threadIdx.x; col < width; col += blockDim.x)
    max_value = fmaxf(max_value, score_row[col]);
  reductions[threadIdx.x] = max_value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] =
          fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  max_value = reductions[0];

  float sum = 0.0f;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = expf(score_row[col] - max_value);
    probability_row[col] = value;
    sum += value;
  }
  reductions[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  sum = reductions[0];

  for (int col = threadIdx.x; col < width; col += blockDim.x)
    probability_row[col] = sum > 0.0f ? probability_row[col] / sum : 0.0f;
}

__device__ double WarpMax(double value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmax(value, __shfl_down_sync(0xffffffffu, value, offset));
  return __shfl_sync(0xffffffffu, value, 0);
}

__device__ double WarpSum(double value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void AttentionSoftmaxDeterministic64Kernel(const float *scores,
                                                      float *probabilities) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  const float *score_row = scores + static_cast<std::size_t>(row) * 64;
  float *probability_row = probabilities + static_cast<std::size_t>(row) * 64;

  const float score0 = score_row[lane];
  const float score1 = score_row[lane + 32];
  const double max_value =
      WarpMax(fmax(static_cast<double>(score0), static_cast<double>(score1)));

  const float value0 = expf(score0 - static_cast<float>(max_value));
  const float value1 = expf(score1 - static_cast<float>(max_value));
  const double sum =
      WarpSum(static_cast<double>(value0) + static_cast<double>(value1));
  const float scale = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;

  probability_row[lane] = value0 * scale;
  probability_row[lane + 32] = value1 * scale;
}

__global__ void
AttentionBiasSoftmaxDeterministic64Kernel(float *scores, const float *bias,
                                          float *probabilities) {
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  float *score_row = scores + static_cast<std::size_t>(row) * 64;
  const float *bias_row = bias + static_cast<std::size_t>(row) * 64;
  float *probability_row = probabilities + static_cast<std::size_t>(row) * 64;

  const float score0 = score_row[lane] + bias_row[lane];
  const float score1 = score_row[lane + 32] + bias_row[lane + 32];
  score_row[lane] = score0;
  score_row[lane + 32] = score1;

  const double max_value =
      WarpMax(fmax(static_cast<double>(score0), static_cast<double>(score1)));
  const float value0 = expf(score0 - static_cast<float>(max_value));
  const float value1 = expf(score1 - static_cast<float>(max_value));
  const double sum =
      WarpSum(static_cast<double>(value0) + static_cast<double>(value1));
  const float scale = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;

  probability_row[lane] = value0 * scale;
  probability_row[lane + 32] = value1 * scale;
}

__global__ void AttentionSoftmaxDeterministicKernel(const float *scores,
                                                    float *probabilities,
                                                    int width) {
  extern __shared__ double softmax_deterministic_reductions[];
  const int row = blockIdx.x;
  const float *score_row = scores + static_cast<std::size_t>(row) * width;
  float *probability_row =
      probabilities + static_cast<std::size_t>(row) * width;

  double max_value = -1.7976931348623157e+308;
  for (int col = threadIdx.x; col < width; col += blockDim.x)
    max_value = fmax(max_value, static_cast<double>(score_row[col]));
  softmax_deterministic_reductions[threadIdx.x] = max_value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      softmax_deterministic_reductions[threadIdx.x] =
          fmax(softmax_deterministic_reductions[threadIdx.x],
               softmax_deterministic_reductions[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  max_value = softmax_deterministic_reductions[0];

  double sum = 0.0;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = expf(score_row[col] - static_cast<float>(max_value));
    probability_row[col] = value;
    sum += static_cast<double>(value);
  }
  softmax_deterministic_reductions[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      softmax_deterministic_reductions[threadIdx.x] +=
          softmax_deterministic_reductions[threadIdx.x + stride];
    __syncthreads();
  }
  sum = softmax_deterministic_reductions[0];

  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    probability_row[col] =
        sum > 0.0 ? static_cast<float>(
                        static_cast<double>(probability_row[col]) / sum)
                  : 0.0f;
  }
}

__global__ void AttentionBiasSoftmaxDeterministicKernel(float *scores,
                                                        const float *bias,
                                                        float *probabilities,
                                                        int width) {
  extern __shared__ double bias_softmax_deterministic_reductions[];
  const int row = blockIdx.x;
  float *score_row = scores + static_cast<std::size_t>(row) * width;
  const float *bias_row = bias + static_cast<std::size_t>(row) * width;
  float *probability_row =
      probabilities + static_cast<std::size_t>(row) * width;

  for (int col = threadIdx.x; col < width; col += blockDim.x)
    score_row[col] += bias_row[col];
  __syncthreads();

  double max_value = -1.7976931348623157e+308;
  for (int col = threadIdx.x; col < width; col += blockDim.x)
    max_value = fmax(max_value, static_cast<double>(score_row[col]));
  bias_softmax_deterministic_reductions[threadIdx.x] = max_value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      bias_softmax_deterministic_reductions[threadIdx.x] =
          fmax(bias_softmax_deterministic_reductions[threadIdx.x],
               bias_softmax_deterministic_reductions[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  max_value = bias_softmax_deterministic_reductions[0];

  double sum = 0.0;
  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    const float value = expf(score_row[col] - static_cast<float>(max_value));
    probability_row[col] = value;
    sum += static_cast<double>(value);
  }
  bias_softmax_deterministic_reductions[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      bias_softmax_deterministic_reductions[threadIdx.x] +=
          bias_softmax_deterministic_reductions[threadIdx.x + stride];
    __syncthreads();
  }
  sum = bias_softmax_deterministic_reductions[0];

  for (int col = threadIdx.x; col < width; col += blockDim.x) {
    probability_row[col] =
        sum > 0.0 ? static_cast<float>(
                        static_cast<double>(probability_row[col]) / sum)
                  : 0.0f;
  }
}

__device__ __constant__ int kAttentionPolicyGatherDevice[kNetworkPolicyOutputs];
__device__ __constant__ int
    kConvolutionPolicyGatherDevice[kNetworkPolicyOutputs];
__device__ __constant__ float
    kStaticPositionEncodingDevice[kPackedInputSquareCount *
                                  Tables::kNumPosEncodingChannels];

void EnsureAttentionPolicyGatherMapUploaded() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("cudaGetDevice(policy_gather)", status));
  }

  static std::mutex mutex;
  static std::vector<int> uploaded_devices;

  std::lock_guard<std::mutex> lock(mutex);
  if (std::find(uploaded_devices.begin(), uploaded_devices.end(), device) !=
      uploaded_devices.end()) {
    return;
  }

  const auto &gather_map = AttentionPolicyGatherMap();
  status = cudaMemcpyToSymbol(kAttentionPolicyGatherDevice, gather_map.data(),
                              gather_map.size() * sizeof(int), 0,
                              cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("AttentionPolicyGather table upload", status));
  }
  uploaded_devices.push_back(device);
}

void EnsureConvolutionPolicyGatherMapUploaded() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("cudaGetDevice(conv_policy_gather)", status));
  }

  static std::mutex mutex;
  static std::vector<int> uploaded_devices;

  std::lock_guard<std::mutex> lock(mutex);
  if (std::find(uploaded_devices.begin(), uploaded_devices.end(), device) !=
      uploaded_devices.end()) {
    return;
  }

  const auto &gather_map = ConvolutionPolicyGatherMap();
  status = cudaMemcpyToSymbol(kConvolutionPolicyGatherDevice, gather_map.data(),
                              gather_map.size() * sizeof(int), 0,
                              cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ConvolutionPolicyGather table upload", status));
  }
  uploaded_devices.push_back(device);
}

void EnsureStaticPositionEncodingUploaded() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("cudaGetDevice(static_position_encoding)", status));
  }

  static std::mutex mutex;
  static std::vector<int> uploaded_devices;

  std::lock_guard<std::mutex> lock(mutex);
  if (std::find(uploaded_devices.begin(), uploaded_devices.end(), device) !=
      uploaded_devices.end()) {
    return;
  }

  status = cudaMemcpyToSymbol(
      kStaticPositionEncodingDevice, &Tables::kPosEncoding[0][0],
      kPackedInputSquareCount * Tables::kNumPosEncodingChannels * sizeof(float),
      0, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("static position encoding table upload", status));
  }
  uploaded_devices.push_back(device);
}

__global__ void AttentionPolicyPromotionKernel(const float *query,
                                               const float *key,
                                               const float *promotion_weights,
                                               float *raw_policy, int channels,
                                               int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  constexpr int kSquares = kPackedInputSquareCount;
  constexpr int kPromotionCount =
      kNetworkAttentionPolicyScratch -
      kPackedInputSquareCount * kPackedInputSquareCount;
  const int promo_index = index % kPromotionCount;
  const int batch = index / kPromotionCount;
  // Mirrors the Metal graph reshape: QK uses the 3x64 view while the learned
  // promotion offset keeps the 8x8x3 flattened order.
  const int promotion_row = promo_index % 3;
  const int promotion_key_square = 56 + (promo_index % 24) / 3;
  const int square_pair = promo_index % kSquares;
  const int query_square = 48 + square_pair / 8;
  const int key_square = 56 + square_pair % 8;
  const float *query_row =
      query +
      (static_cast<std::size_t>(batch) * kSquares + query_square) * channels;
  const float *key_row =
      key +
      (static_cast<std::size_t>(batch) * kSquares + key_square) * channels;
  const float *promotion_key_row =
      key +
      (static_cast<std::size_t>(batch) * kSquares + promotion_key_square) *
          channels;

  float value = 0.0f;
  const float scale = rsqrtf(static_cast<float>(channels));
  for (int channel = 0; channel < channels; ++channel) {
    value += query_row[channel] * key_row[channel] * scale;
    value += promotion_key_row[channel] *
             (promotion_weights[promotion_row * channels + channel] +
              promotion_weights[3 * channels + channel]);
  }

  raw_policy[static_cast<std::size_t>(batch) * kNetworkAttentionPolicyScratch +
             kSquares * kSquares + promo_index] = value;
}

__global__ void AttentionPolicyGatherKernel(const float *raw_policy,
                                            float *policy, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int policy_index = index % kNetworkPolicyOutputs;
  const int batch = index / kNetworkPolicyOutputs;
  const int raw_index = kAttentionPolicyGatherDevice[policy_index];
  policy[index] = raw_index >= 0
                      ? raw_policy[static_cast<std::size_t>(batch) *
                                       kNetworkAttentionPolicyScratch +
                                   raw_index]
                      : 0.0f;
}

__global__ void ConvolutionPolicyGatherKernel(const float *raw_policy,
                                              float *policy, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int policy_index = index % kNetworkPolicyOutputs;
  const int batch = index / kNetworkPolicyOutputs;
  const int raw_index = kConvolutionPolicyGatherDevice[policy_index];
  policy[index] = raw_index >= 0 ? raw_policy[static_cast<std::size_t>(batch) *
                                                  kNetworkConvPolicyScratch +
                                              raw_index]
                                 : 0.0f;
}

__global__ void ExpandPackedInputPlanesKernel(const std::uint64_t *masks,
                                              const float *values,
                                              float *expanded, int planes,
                                              int squares, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int plane = index % planes;
  const int square = (index / planes) % squares;
  const int batch = index / (planes * squares);
  const int packed_index = batch * planes + plane;
  const std::uint64_t bit = 1ULL << square;
  expanded[index] = (masks[packed_index] & bit) ? values[packed_index] : 0.0f;
}

__global__ void ExpandPackedInputPlanesNchwKernel(const std::uint64_t *masks,
                                                  const float *values,
                                                  float *expanded, int planes,
                                                  int squares, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int square = index % squares;
  const int plane = (index / squares) % planes;
  const int batch = index / (planes * squares);
  const int packed_index = batch * planes + plane;
  const std::uint64_t bit = 1ULL << square;
  expanded[index] = (masks[packed_index] & bit) ? values[packed_index] : 0.0f;
}

__global__ void ExpandPackedInputPlanesWithPositionInputKernel(
    const std::uint64_t *masks, const float *values, float *expanded,
    float *position_input, int planes, int position_planes, int squares,
    int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int plane = index % planes;
  const int square = (index / planes) % squares;
  const int batch = index / (planes * squares);
  const int packed_index = batch * planes + plane;
  const std::uint64_t bit = 1ULL << square;
  const float value = (masks[packed_index] & bit) ? values[packed_index] : 0.0f;
  expanded[index] = value;
  if (plane < position_planes) {
    position_input[(static_cast<std::size_t>(batch) * squares + square) *
                       position_planes +
                   plane] = value;
  }
}

__global__ void DynamicPositionEncodingInputKernel(const float *expanded,
                                                   float *position_input,
                                                   int input_planes,
                                                   int position_planes,
                                                   int squares, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int channel = index % position_planes;
  const int square = (index / position_planes) % squares;
  const int batch = index / (position_planes * squares);
  position_input[index] =
      expanded[(static_cast<std::size_t>(batch) * squares + square) *
                   input_planes +
               channel];
}

__global__ void DynamicPositionEncodingSparseDenseKernel(
    const std::uint64_t *masks, const float *values, const float *weights,
    const float *bias, float *output, int input_planes, int position_planes,
    int squares, int input_width, int output_width, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int output_channel = index % output_width;
  const int batch = index / output_width;
  const float *weight_row =
      weights + static_cast<std::size_t>(output_channel) * input_width;
  float sum = bias ? bias[output_channel] : 0.0f;
  const std::uint64_t valid_squares =
      squares >= 64 ? ~0ULL : ((1ULL << squares) - 1ULL);
  for (int plane = 0; plane < position_planes; ++plane) {
    const int packed_index = batch * input_planes + plane;
    std::uint64_t mask = masks[packed_index] & valid_squares;
    if (!mask)
      continue;
    const float value = values[packed_index];
    while (mask) {
      const int square = __ffsll(static_cast<long long>(mask)) - 1;
      const int input_channel = square * position_planes + plane;
      sum += value * weight_row[input_channel];
      mask &= mask - 1;
    }
  }
  output[index] = sum;
}

__global__ void DynamicPositionEncodingConcatKernel(
    const float *expanded, const float *position_encoding, float *output,
    int input_planes, int position_width, int squares, int output_width,
    int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int channel = index % output_width;
  const int square = (index / output_width) % squares;
  const int batch = index / (output_width * squares);
  if (channel < input_planes) {
    output[index] =
        expanded[(static_cast<std::size_t>(batch) * squares + square) *
                     input_planes +
                 channel];
    return;
  }

  const int pe_channel = channel - input_planes;
  output[index] =
      position_encoding[(static_cast<std::size_t>(batch) * squares + square) *
                            position_width +
                        pe_channel];
}

__global__ void
StaticPositionEncodingConcatKernel(const std::uint64_t *masks,
                                   const float *values, float *output,
                                   int input_planes, int position_width,
                                   int squares, int output_width, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int channel = index % output_width;
  const int square = (index / output_width) % squares;
  const int batch = index / (output_width * squares);
  if (channel < input_planes) {
    const int packed_index = batch * input_planes + channel;
    const std::uint64_t bit = 1ULL << square;
    output[index] = (masks[packed_index] & bit) ? values[packed_index] : 0.0f;
    return;
  }

  const int pe_channel = channel - input_planes;
  output[index] =
      kStaticPositionEncodingDevice[square * position_width + pe_channel];
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
  cublasHandle_t handle = CublasHandle();
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        CublasErrorMessage("cublasSetStream", cublas_status));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublas_status =
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, output_width, batch_size,
                  input_width, &alpha, weights, input_width, input, input_width,
                  &beta, output, output_width);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        CublasErrorMessage("cublasSgemm(dense_affine)", cublas_status));
  }

  constexpr int kThreads = 256;
  if (bias) {
    const int blocks = (total_outputs + kThreads - 1) / kThreads;
    BiasAddKernel<<<blocks, kThreads, 0, stream>>>(output, bias, output_width,
                                                   total_outputs);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("BiasAddKernel launch", status));
  }
  if (stream)
    return;
  cudaError_t status = cudaDeviceSynchronize();
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

void LaunchBiasActivationKernel(float *input, const float *bias, float *output,
                                int rows, int width, CudaActivationKind kind,
                                cudaStream_t stream) {
  if (!input || !bias || !output)
    throw std::runtime_error(
        "CUDA bias activation kernel received null buffer");
  if (rows <= 0 || width <= 0)
    throw std::runtime_error(
        "CUDA bias activation kernel dimensions are invalid");

  constexpr int kThreads = 256;
  const int total = rows * width;
  const int blocks = (total + kThreads - 1) / kThreads;
  BiasActivationKernel<<<blocks, kThreads, 0, stream>>>(input, bias, output,
                                                        width, total, kind);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("BiasActivationKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("BiasActivationKernel synchronize", status));
}

void LaunchGateKernel(const float *input, const float *weights, float *output,
                      int rows, int width, int gate_rows, CudaGateKind kind,
                      cudaStream_t stream) {
  if (!input || !weights || !output)
    throw std::runtime_error("CUDA gate kernel received null buffer");
  if (rows <= 0 || width <= 0 || gate_rows <= 0 || rows % gate_rows != 0)
    throw std::runtime_error("CUDA gate kernel dimensions are invalid");

  const int total = rows * width;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  GateKernel<<<blocks, kThreads, 0, stream>>>(input, weights, output, width,
                                              total, gate_rows, kind);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(CudaErrorMessage("GateKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("GateKernel synchronize", status));
}

void LaunchResidualAddKernel(const float *parent, const float *secondary,
                             float *output, int batch_size, int width,
                             float secondary_scale, cudaStream_t stream) {
  if (!parent || !secondary || !output)
    throw std::runtime_error("CUDA residual add kernel received null buffer");
  if (batch_size <= 0 || width <= 0)
    throw std::runtime_error("CUDA residual add kernel dimensions are invalid");

  const int total = batch_size * width;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  ResidualAddKernel<<<blocks, kThreads, 0, stream>>>(parent, secondary, output,
                                                     total, secondary_scale);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("ResidualAddKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("ResidualAddKernel synchronize", status));
}

void LaunchGlobalAveragePoolNchwKernel(const float *input, float *output,
                                       int batch_size, int channels,
                                       int squares, cudaStream_t stream) {
  if (!input || !output)
    throw std::runtime_error(
        "CUDA global average pool kernel received null buffer");
  if (batch_size <= 0 || channels <= 0 || squares <= 0)
    throw std::runtime_error(
        "CUDA global average pool kernel dimensions are invalid");

  const int total = batch_size * channels;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  GlobalAveragePoolNchwKernel<<<blocks, kThreads, 0, stream>>>(
      input, output, channels, squares, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("GlobalAveragePoolNchwKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("GlobalAveragePoolNchwKernel synchronize", status));
}

void LaunchSqueezeExciteResidualKernel(
    const float *skip, const float *convolution, const float *se_output,
    float *residual, float *output, int batch_size, int channels, int squares,
    CudaActivationKind activation, cudaStream_t stream) {
  if (!skip || !convolution || !se_output || !residual || !output)
    throw std::runtime_error(
        "CUDA squeeze-excite residual kernel received null buffer");
  if (batch_size <= 0 || channels <= 0 || squares <= 0)
    throw std::runtime_error(
        "CUDA squeeze-excite residual kernel dimensions are invalid");

  const int total = batch_size * channels * squares;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  SqueezeExciteResidualKernel<<<blocks, kThreads, 0, stream>>>(
      skip, convolution, se_output, residual, output, channels, squares, total,
      activation);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("SqueezeExciteResidualKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("SqueezeExciteResidualKernel synchronize", status));
}

void LaunchResidualLayerNormKernel(const float *parent, const float *secondary,
                                   const float *gamma, const float *beta,
                                   float *residual, float *output, int rows,
                                   int width, float secondary_scale,
                                   float epsilon, cudaStream_t stream) {
  if (!parent || !secondary || !gamma || !beta || !residual || !output) {
    throw std::runtime_error(
        "CUDA residual layernorm kernel received null buffer");
  }
  if (rows <= 0 || width <= 0 || epsilon <= 0.0f) {
    throw std::runtime_error(
        "CUDA residual layernorm kernel dimensions are invalid");
  }

  constexpr int kThreads = 256;
  const std::size_t shared_bytes = 2 * kThreads * sizeof(float);
  ResidualLayerNormKernel<<<rows, kThreads, shared_bytes, stream>>>(
      parent, secondary, gamma, beta, residual, output, width, secondary_scale,
      epsilon);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ResidualLayerNormKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ResidualLayerNormKernel synchronize", status));
  }
}

void LaunchConvolution2DKernel(const float *input, const float *weights,
                               const float *bias, float *output, int batch_size,
                               int squares, int input_channels,
                               int output_channels, int kernel_size,
                               CudaActivationKind activation,
                               bool apply_activation, cudaStream_t stream) {
  if (!input || !weights || !output)
    throw std::runtime_error("CUDA convolution kernel received null buffer");
  if (batch_size <= 0 || squares != 64 || input_channels <= 0 ||
      output_channels <= 0 || (kernel_size != 1 && kernel_size != 3)) {
    throw std::runtime_error("CUDA convolution dimensions are invalid");
  }

  const int total = batch_size * squares * output_channels;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  Convolution2DKernel<<<blocks, kThreads, 0, stream>>>(
      input, weights, bias, output, squares, input_channels, output_channels,
      kernel_size, activation, apply_activation, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("Convolution2DKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("Convolution2DKernel synchronize", status));
  }
}

void LaunchResidualBiasLayerNormKernel(
    const float *parent, const float *secondary, const float *secondary_bias,
    const float *gamma, const float *beta, float *biased_secondary,
    float *residual, float *output, int rows, int width, float secondary_scale,
    float epsilon, cudaStream_t stream) {
  if (!parent || !secondary || !secondary_bias || !gamma || !beta ||
      !biased_secondary || !residual || !output) {
    throw std::runtime_error(
        "CUDA residual bias layernorm kernel received null buffer");
  }
  if (rows <= 0 || width <= 0 || epsilon <= 0.0f) {
    throw std::runtime_error(
        "CUDA residual bias layernorm kernel dimensions are invalid");
  }

  constexpr int kThreads = 256;
  const std::size_t shared_bytes = 2 * kThreads * sizeof(float);
  ResidualBiasLayerNormKernel<<<rows, kThreads, shared_bytes, stream>>>(
      parent, secondary, secondary_bias, gamma, beta, biased_secondary,
      residual, output, width, secondary_scale, epsilon);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ResidualBiasLayerNormKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ResidualBiasLayerNormKernel synchronize", status));
  }
}

void LaunchAttentionScoreKernel(const float *query, const float *key,
                                float *scores, int batch_size, int heads,
                                int squares, int head_depth, int qkv_width,
                                float scale, cudaStream_t stream) {
  if (!query || !key || !scores)
    throw std::runtime_error(
        "CUDA attention score kernel received null buffer");
  if (batch_size <= 0 || heads <= 0 || squares <= 0 || head_depth <= 0 ||
      qkv_width <= 0 || qkv_width != heads * head_depth || scale <= 0.0f) {
    throw std::runtime_error("CUDA attention score dimensions are invalid");
  }

  cublasHandle_t handle = CublasHandle();
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        CublasErrorMessage("cublasSetStream", cublas_status));
  }

  const float beta = 0.0f;
  const long long head_stride = head_depth;
  const long long score_stride = static_cast<long long>(squares) * squares;
  const std::size_t qkv_batch_stride =
      static_cast<std::size_t>(squares) * qkv_width;
  const std::size_t score_batch_stride =
      static_cast<std::size_t>(heads) * squares * squares;
  for (int batch = 0; batch < batch_size; ++batch) {
    const float *query_base = query + batch * qkv_batch_stride;
    const float *key_base = key + batch * qkv_batch_stride;
    float *score_base = scores + batch * score_batch_stride;
    cublas_status = cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, squares, squares, head_depth, &scale,
        key_base, qkv_width, head_stride, query_base, qkv_width, head_stride,
        &beta, score_base, squares, score_stride, heads);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(CublasErrorMessage(
          "cublasSgemmStridedBatched(attention_score)", cublas_status));
    }
  }
  if (stream)
    return;
  cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionScoreKernel synchronize", status));
}

void LaunchAttentionBiasAddKernel(float *scores, const float *bias,
                                  int batch_size, int heads, int squares,
                                  cudaStream_t stream) {
  if (!scores || !bias)
    throw std::runtime_error(
        "CUDA attention bias add kernel received null buffer");
  if (batch_size <= 0 || heads <= 0 || squares <= 0)
    throw std::runtime_error("CUDA attention bias add dimensions are invalid");

  const int total = batch_size * heads * squares * squares;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  AttentionBiasAddKernel<<<blocks, kThreads, 0, stream>>>(scores, bias, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionBiasAddKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionBiasAddKernel synchronize", status));
}

void LaunchAttentionSoftmaxKernel(const float *scores, float *probabilities,
                                  int rows, int width, cudaStream_t stream,
                                  bool deterministic) {
  if (!scores || !probabilities)
    throw std::runtime_error(
        "CUDA attention softmax kernel received null buffer");
  if (rows <= 0 || width <= 0)
    throw std::runtime_error("CUDA attention softmax dimensions are invalid");

  if (deterministic) {
    if (width == 64) {
      AttentionSoftmaxDeterministic64Kernel<<<rows, 32, 0, stream>>>(
          scores, probabilities);
    } else {
      const int threads = width <= 64 ? 64 : 128;
      const std::size_t shared_bytes = threads * sizeof(double);
      AttentionSoftmaxDeterministicKernel<<<rows, threads, shared_bytes,
                                            stream>>>(scores, probabilities,
                                                      width);
    }
  } else {
    constexpr int kThreads = 128;
    const std::size_t shared_bytes = kThreads * sizeof(float);
    AttentionSoftmaxKernel<<<rows, kThreads, shared_bytes, stream>>>(
        scores, probabilities, width);
  }
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionSoftmaxKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionSoftmaxKernel synchronize", status));
}

void LaunchAttentionBiasSoftmaxKernel(float *scores, const float *bias,
                                      float *probabilities, int rows, int width,
                                      cudaStream_t stream, bool deterministic) {
  if (!scores || !bias || !probabilities)
    throw std::runtime_error(
        "CUDA attention bias softmax kernel received null buffer");
  if (rows <= 0 || width <= 0)
    throw std::runtime_error(
        "CUDA attention bias softmax dimensions are invalid");

  if (deterministic) {
    if (width == 64) {
      AttentionBiasSoftmaxDeterministic64Kernel<<<rows, 32, 0, stream>>>(
          scores, bias, probabilities);
    } else {
      const int threads = width <= 64 ? 64 : 128;
      const std::size_t shared_bytes = threads * sizeof(double);
      AttentionBiasSoftmaxDeterministicKernel<<<rows, threads, shared_bytes,
                                                stream>>>(scores, bias,
                                                          probabilities, width);
    }
  } else {
    constexpr int kThreads = 128;
    const std::size_t shared_bytes = kThreads * sizeof(float);
    AttentionBiasSoftmaxKernel<<<rows, kThreads, shared_bytes, stream>>>(
        scores, bias, probabilities, width);
  }
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionBiasSoftmaxKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionBiasSoftmaxKernel synchronize", status));
}

void LaunchAttentionContextKernel(const float *probabilities,
                                  const float *value, float *context,
                                  int batch_size, int heads, int squares,
                                  int head_depth, int qkv_width,
                                  cudaStream_t stream) {
  if (!probabilities || !value || !context)
    throw std::runtime_error(
        "CUDA attention context kernel received null buffer");
  if (batch_size <= 0 || heads <= 0 || squares <= 0 || head_depth <= 0 ||
      qkv_width <= 0 || qkv_width != heads * head_depth) {
    throw std::runtime_error("CUDA attention context dimensions are invalid");
  }

  cublasHandle_t handle = CublasHandle();
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        CublasErrorMessage("cublasSetStream", cublas_status));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const long long head_stride = head_depth;
  const long long probability_stride =
      static_cast<long long>(squares) * squares;
  const std::size_t qkv_batch_stride =
      static_cast<std::size_t>(squares) * qkv_width;
  const std::size_t probability_batch_stride =
      static_cast<std::size_t>(heads) * squares * squares;
  for (int batch = 0; batch < batch_size; ++batch) {
    const float *value_base = value + batch * qkv_batch_stride;
    const float *probability_base =
        probabilities + batch * probability_batch_stride;
    float *context_base = context + batch * qkv_batch_stride;
    cublas_status = cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, head_depth, squares, squares, &alpha,
        value_base, qkv_width, head_stride, probability_base, squares,
        probability_stride, &beta, context_base, qkv_width, head_stride, heads);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(CublasErrorMessage(
          "cublasSgemmStridedBatched(attention_context)", cublas_status));
    }
  }
  if (stream)
    return;
  cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionContextKernel synchronize", status));
}

void LaunchAttentionPolicyMapKernel(const float *query, const float *key,
                                    const float *promotion_weights,
                                    float *raw_policy, float *policy,
                                    int batch_size, int channels,
                                    cudaStream_t stream) {
  if (!query || !key || !promotion_weights || !raw_policy || !policy) {
    throw std::runtime_error(
        "CUDA attention policy map kernel received null buffer");
  }
  if (batch_size <= 0 || channels <= 0) {
    throw std::runtime_error(
        "CUDA attention policy map dimensions are invalid");
  }

  EnsureAttentionPolicyGatherMapUploaded();

  cublasHandle_t handle = CublasHandle();
  cublasStatus_t cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        CublasErrorMessage("cublasSetStream", cublas_status));
  }

  constexpr int kSquares = kPackedInputSquareCount;
  const float alpha = 1.0f / std::sqrt(static_cast<float>(channels));
  const float beta = 0.0f;
  const long long input_stride = static_cast<long long>(kSquares) * channels;
  const long long raw_policy_stride = kNetworkAttentionPolicyScratch;
  cublas_status = cublasSgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, kSquares, kSquares, channels, &alpha,
      key, channels, input_stride, query, channels, input_stride, &beta,
      raw_policy, kSquares, raw_policy_stride, batch_size);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(CublasErrorMessage(
        "cublasSgemmStridedBatched(attention_policy_map)", cublas_status));
  }

  constexpr int kPromotionCount =
      kNetworkAttentionPolicyScratch - kSquares * kSquares;
  const int total = batch_size * kPromotionCount;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  AttentionPolicyPromotionKernel<<<blocks, kThreads, 0, stream>>>(
      query, key, promotion_weights, raw_policy, channels, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("AttentionPolicyPromotionKernel launch", status));
  }
  const int policy_total = batch_size * kNetworkPolicyOutputs;
  const int gather_blocks = (policy_total + kThreads - 1) / kThreads;
  AttentionPolicyGatherKernel<<<gather_blocks, kThreads, 0, stream>>>(
      raw_policy, policy, policy_total);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("AttentionPolicyGatherKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("AttentionPolicyGatherKernel synchronize", status));
  }
}

void LaunchConvolutionPolicyMapKernel(const float *raw_policy, float *policy,
                                      int batch_size, cudaStream_t stream) {
  if (!raw_policy || !policy) {
    throw std::runtime_error(
        "CUDA convolution policy map kernel received null buffer");
  }
  if (batch_size <= 0) {
    throw std::runtime_error(
        "CUDA convolution policy map dimensions are invalid");
  }

  EnsureConvolutionPolicyGatherMapUploaded();

  constexpr int kThreads = 256;
  const int total = batch_size * kNetworkPolicyOutputs;
  const int blocks = (total + kThreads - 1) / kThreads;
  ConvolutionPolicyGatherKernel<<<blocks, kThreads, 0, stream>>>(raw_policy,
                                                                 policy, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ConvolutionPolicyGatherKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ConvolutionPolicyGatherKernel synchronize", status));
  }
}

void LaunchExpandPackedInputPlanesKernel(const std::uint64_t *masks,
                                         const float *values, float *expanded,
                                         int batch_size, int planes,
                                         int squares, cudaStream_t stream) {
  if (!masks || !values || !expanded)
    throw std::runtime_error(
        "CUDA input expansion kernel received null buffer");
  if (batch_size <= 0 || planes <= 0 || squares <= 0 || squares > 64) {
    throw std::runtime_error("CUDA input expansion dimensions are invalid");
  }

  const int total = batch_size * squares * planes;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  ExpandPackedInputPlanesKernel<<<blocks, kThreads, 0, stream>>>(
      masks, values, expanded, planes, squares, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ExpandPackedInputPlanesKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ExpandPackedInputPlanesKernel synchronize", status));
  }
}

void LaunchExpandPackedInputPlanesNchwKernel(const std::uint64_t *masks,
                                             const float *values,
                                             float *expanded, int batch_size,
                                             int planes, int squares,
                                             cudaStream_t stream) {
  if (!masks || !values || !expanded)
    throw std::runtime_error(
        "CUDA NCHW input expansion kernel received null buffer");
  if (batch_size <= 0 || planes <= 0 || squares <= 0 || squares > 64) {
    throw std::runtime_error(
        "CUDA NCHW input expansion dimensions are invalid");
  }

  const int total = batch_size * planes * squares;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  ExpandPackedInputPlanesNchwKernel<<<blocks, kThreads, 0, stream>>>(
      masks, values, expanded, planes, squares, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("ExpandPackedInputPlanesNchwKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "ExpandPackedInputPlanesNchwKernel synchronize", status));
  }
}

void LaunchExpandPackedInputPlanesWithPositionInputKernel(
    const std::uint64_t *masks, const float *values, float *expanded,
    float *position_input, int batch_size, int planes, int position_planes,
    int squares, cudaStream_t stream) {
  if (!masks || !values || !expanded || !position_input)
    throw std::runtime_error(
        "CUDA input expansion with position input received null buffer");
  if (batch_size <= 0 || planes <= 0 || position_planes <= 0 ||
      position_planes > planes || squares <= 0 || squares > 64) {
    throw std::runtime_error(
        "CUDA input expansion with position input dimensions are invalid");
  }

  const int total = batch_size * squares * planes;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  ExpandPackedInputPlanesWithPositionInputKernel<<<blocks, kThreads, 0,
                                                   stream>>>(
      masks, values, expanded, position_input, planes, position_planes, squares,
      total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "ExpandPackedInputPlanesWithPositionInputKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "ExpandPackedInputPlanesWithPositionInputKernel synchronize", status));
  }
}

void LaunchDynamicPositionEncodingSparseDenseKernel(
    const std::uint64_t *masks, const float *values, const float *weights,
    const float *bias, float *output, int batch_size, int input_planes,
    int position_planes, int squares, int input_width, int output_width,
    cudaStream_t stream) {
  if (!masks || !values || !weights || !output) {
    throw std::runtime_error(
        "CUDA dynamic position sparse dense kernel received null buffer");
  }
  if (batch_size <= 0 || input_planes <= 0 || position_planes <= 0 ||
      position_planes > input_planes || squares <= 0 || squares > 64 ||
      input_width != squares * position_planes || output_width <= 0) {
    throw std::runtime_error(
        "CUDA dynamic position sparse dense dimensions are invalid");
  }

  const int total = batch_size * output_width;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  DynamicPositionEncodingSparseDenseKernel<<<blocks, kThreads, 0, stream>>>(
      masks, values, weights, bias, output, input_planes, position_planes,
      squares, input_width, output_width, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "DynamicPositionEncodingSparseDenseKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "DynamicPositionEncodingSparseDenseKernel synchronize", status));
  }
}

void LaunchStaticPositionEncodingConcatKernel(const std::uint64_t *masks,
                                              const float *values,
                                              float *output, int batch_size,
                                              int input_planes,
                                              int position_width, int squares,
                                              cudaStream_t stream) {
  if (!masks || !values || !output) {
    throw std::runtime_error(
        "CUDA static position concat kernel received null buffer");
  }
  if (batch_size <= 0 || input_planes <= 0 || position_width <= 0 ||
      position_width > Tables::kNumPosEncodingChannels || squares <= 0 ||
      squares > kPackedInputSquareCount) {
    throw std::runtime_error(
        "CUDA static position concat dimensions are invalid");
  }

  EnsureStaticPositionEncodingUploaded();
  const int output_width = input_planes + position_width;
  const int total = batch_size * squares * output_width;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  StaticPositionEncodingConcatKernel<<<blocks, kThreads, 0, stream>>>(
      masks, values, output, input_planes, position_width, squares,
      output_width, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("StaticPositionEncodingConcatKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "StaticPositionEncodingConcatKernel synchronize", status));
  }
}

void LaunchDynamicPositionEncodingInputKernel(const float *expanded,
                                              float *position_input,
                                              int batch_size, int input_planes,
                                              int position_planes, int squares,
                                              cudaStream_t stream) {
  if (!expanded || !position_input) {
    throw std::runtime_error(
        "CUDA dynamic position input kernel received null buffer");
  }
  if (batch_size <= 0 || input_planes <= 0 || position_planes <= 0 ||
      position_planes > input_planes || squares <= 0) {
    throw std::runtime_error(
        "CUDA dynamic position input dimensions are invalid");
  }

  const int total = batch_size * squares * position_planes;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  DynamicPositionEncodingInputKernel<<<blocks, kThreads, 0, stream>>>(
      expanded, position_input, input_planes, position_planes, squares, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("DynamicPositionEncodingInputKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "DynamicPositionEncodingInputKernel synchronize", status));
  }
}

void LaunchDynamicPositionEncodingConcatKernel(const float *expanded,
                                               const float *position_encoding,
                                               float *output, int batch_size,
                                               int input_planes,
                                               int position_width, int squares,
                                               cudaStream_t stream) {
  if (!expanded || !position_encoding || !output) {
    throw std::runtime_error(
        "CUDA dynamic position concat kernel received null buffer");
  }
  if (batch_size <= 0 || input_planes <= 0 || position_width <= 0 ||
      squares <= 0) {
    throw std::runtime_error(
        "CUDA dynamic position concat dimensions are invalid");
  }

  const int output_width = input_planes + position_width;
  const int total = batch_size * squares * output_width;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  DynamicPositionEncodingConcatKernel<<<blocks, kThreads, 0, stream>>>(
      expanded, position_encoding, output, input_planes, position_width,
      squares, output_width, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("DynamicPositionEncodingConcatKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "DynamicPositionEncodingConcatKernel synchronize", status));
  }
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
      1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f,
  };
  const std::vector<float> weights = {
      1.0f,  0.0f, -1.0f, 0.5f, 0.5f,  0.5f,
      -2.0f, 1.0f, 0.25f, 0.0f, -1.0f, 2.0f,
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
    AllocateDevice(&device_weights, weights.size(),
                   "cudaMalloc(dense_weights)");
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
                                    0.25f, 1.0f,  3.0f};
  const std::vector<CudaActivationKind> activations = {
      CudaActivationKind::Relu,  CudaActivationKind::Relu2,
      CudaActivationKind::Tanh,  CudaActivationKind::Sigmoid,
      CudaActivationKind::Swish, CudaActivationKind::Mish,
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

CudaKernelSmokeResult RunConvolutionKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 1;
  constexpr int kSquares = 64;
  constexpr int kInputChannels = 1;
  constexpr int kOutputChannels = 1;
  constexpr int kKernel = 3;
  const std::vector<float> input(kBatch * kInputChannels * kSquares, 1.0f);
  const std::vector<float> weights(
      kOutputChannels * kInputChannels * kKernel * kKernel, 1.0f);
  const std::vector<float> bias(kOutputChannels, 0.0f);
  std::vector<float> actual(kBatch * kOutputChannels * kSquares, 0.0f);
  std::vector<float> expected(actual.size(), 0.0f);

  for (int square = 0; square < kSquares; ++square) {
    const int rank = square / 8;
    const int file = square % 8;
    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
      const int src_rank = rank + dy;
      if (src_rank < 0 || src_rank >= 8)
        continue;
      for (int dx = -1; dx <= 1; ++dx) {
        const int src_file = file + dx;
        if (src_file >= 0 && src_file < 8)
          ++count;
      }
    }
    expected[static_cast<std::size_t>(square)] = static_cast<float>(count);
  }

  float *device_input = nullptr;
  float *device_weights = nullptr;
  float *device_bias = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_input, input.size(), "cudaMalloc(conv_input)");
    AllocateDevice(&device_weights, weights.size(), "cudaMalloc(conv_weights)");
    AllocateDevice(&device_bias, bias.size(), "cudaMalloc(conv_bias)");
    AllocateDevice(&device_output, actual.size(), "cudaMalloc(conv_output)");

    UploadFloats(device_input, input, "cudaMemcpy(conv_input)");
    UploadFloats(device_weights, weights, "cudaMemcpy(conv_weights)");
    UploadFloats(device_bias, bias, "cudaMemcpy(conv_bias)");

    LaunchConvolution2DKernel(device_input, device_weights, device_bias,
                              device_output, kBatch, kSquares, kInputChannels,
                              kOutputChannels, kKernel,
                              CudaActivationKind::Relu, false);
    DownloadFloats(actual, device_output, "cudaMemcpy(conv_output)");
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
      result.message = "CUDA convolution kernel output mismatch";
      return result;
    }
  }

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
      1.0f, 2.0f, 4.0f,  8.0f,  16.0f, -3.0f, -1.0f, 0.0f,
      1.0f, 3.0f, 0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
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

  constexpr int kRows = 4;
  constexpr int kWidth = 4;
  const std::vector<float> input = {
      1.0f,  -2.0f,  0.5f, 3.0f, -1.0f,  0.25f, 2.0f,  -0.5f,
      0.75f, -1.25f, 1.5f, 0.0f, -0.75f, 1.25f, -1.5f, 0.5f,
  };
  const std::vector<float> mult_gate = {
      2.0f, -0.5f, 4.0f, 0.25f, -1.0f, 0.75f, 0.5f, 3.0f,
  };
  const std::vector<float> add_gate = {
      0.5f, 1.0f, -2.0f, 3.0f, -0.25f, 0.5f, 1.5f, -1.0f,
  };
  std::vector<float> actual(kRows * kWidth, 0.0f);
  std::vector<float> expected(kRows * kWidth, 0.0f);

  for (int b = 0; b < kRows; ++b) {
    for (int i = 0; i < kWidth; ++i) {
      const std::size_t index = static_cast<std::size_t>(b) * kWidth + i;
      const std::size_t gate_index = static_cast<std::size_t>(i) * 2 + (b % 2);
      expected[index] =
          input[index] * mult_gate[gate_index] + add_gate[gate_index];
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

    LaunchGateKernel(device_input, device_mult, device_output, kRows, kWidth, 2,
                     CudaGateKind::Multiply);
    LaunchGateKernel(device_output, device_add, device_output, kRows, kWidth, 2,
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

CudaKernelSmokeResult RunResidualAddKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kWidth = 4;
  constexpr float kScale = 0.375f;
  const std::vector<float> parent = {
      1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 0.25f, 2.0f, -0.5f,
  };
  const std::vector<float> secondary = {
      2.0f, 0.5f, -4.0f, 1.5f, -2.5f, 3.0f, 0.25f, -1.0f,
  };
  std::vector<float> actual(kBatch * kWidth, 0.0f);
  std::vector<float> expected(kBatch * kWidth, 0.0f);
  for (std::size_t i = 0; i < expected.size(); ++i)
    expected[i] = parent[i] + secondary[i] * kScale;

  float *device_parent = nullptr;
  float *device_secondary = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_parent, parent.size(),
                   "cudaMalloc(residual_parent)");
    AllocateDevice(&device_secondary, secondary.size(),
                   "cudaMalloc(residual_secondary)");
    AllocateDevice(&device_output, actual.size(),
                   "cudaMalloc(residual_output)");
    UploadFloats(device_parent, parent, "cudaMemcpy(residual_parent)");
    UploadFloats(device_secondary, secondary, "cudaMemcpy(residual_secondary)");

    LaunchResidualAddKernel(device_parent, device_secondary, device_output,
                            kBatch, kWidth, kScale);
    DownloadFloats(actual, device_output, "cudaMemcpy(residual_output)");

    for (std::size_t i = 0; i < actual.size(); ++i) {
      if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA residual add kernel output mismatch";
        break;
      }
    }
    if (result.message.empty())
      result.status = CudaSmokeStatus::Success;
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
  }

  FreeDevice(device_parent);
  FreeDevice(device_secondary);
  FreeDevice(device_output);

  if (result.message.empty())
    result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunSqueezeExciteKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kChannels = 2;
  constexpr int kSquares = 4;
  const std::vector<float> skip = {
      0.25f, -0.5f,  1.0f, 0.75f, -1.0f, 0.5f, 0.25f,  -0.25f,
      1.25f, -0.75f, 0.0f, 0.5f,  -0.5f, 1.5f, -1.25f, 0.25f,
  };
  const std::vector<float> convolution = {
      1.0f,  2.0f,  -1.0f, 0.0f,  0.5f, -0.5f, 1.5f, 2.5f,
      -1.5f, 0.25f, 0.75f, 1.25f, 2.0f, -2.0f, 0.5f, -0.5f,
  };
  const std::vector<float> se_output = {
      0.0f, 1.0f, 0.25f, -0.5f, -1.0f, 0.5f, 0.75f, 0.125f,
  };
  std::vector<float> actual_pool(kBatch * kChannels, 0.0f);
  std::vector<float> actual_residual(skip.size(), 0.0f);
  std::vector<float> actual_output(skip.size(), 0.0f);
  std::vector<float> expected_pool(actual_pool.size(), 0.0f);
  std::vector<float> expected_residual(skip.size(), 0.0f);
  std::vector<float> expected_output(skip.size(), 0.0f);

  for (int batch = 0; batch < kBatch; ++batch) {
    for (int channel = 0; channel < kChannels; ++channel) {
      const std::size_t pool_index =
          static_cast<std::size_t>(batch) * kChannels + channel;
      const std::size_t plane_offset = pool_index * kSquares;
      float sum = 0.0f;
      for (int square = 0; square < kSquares; ++square)
        sum += convolution[plane_offset + square];
      expected_pool[pool_index] = sum / static_cast<float>(kSquares);

      const std::size_t gamma_index =
          static_cast<std::size_t>(batch) * kChannels * 2 + channel;
      const float gamma = 1.0f / (1.0f + std::exp(-se_output[gamma_index]));
      const float beta = se_output[gamma_index + kChannels];
      for (int square = 0; square < kSquares; ++square) {
        const std::size_t index = plane_offset + square;
        const float value = skip[index] + convolution[index] * gamma + beta;
        expected_residual[index] = value;
        expected_output[index] = std::max(value, 0.0f);
      }
    }
  }

  float *device_skip = nullptr;
  float *device_convolution = nullptr;
  float *device_se_output = nullptr;
  float *device_pool = nullptr;
  float *device_residual = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_skip, skip.size(), "cudaMalloc(se_skip)");
    AllocateDevice(&device_convolution, convolution.size(),
                   "cudaMalloc(se_convolution)");
    AllocateDevice(&device_se_output, se_output.size(),
                   "cudaMalloc(se_output)");
    AllocateDevice(&device_pool, actual_pool.size(), "cudaMalloc(se_pool)");
    AllocateDevice(&device_residual, actual_residual.size(),
                   "cudaMalloc(se_residual)");
    AllocateDevice(&device_output, actual_output.size(),
                   "cudaMalloc(se_activation)");
    UploadFloats(device_skip, skip, "cudaMemcpy(se_skip)");
    UploadFloats(device_convolution, convolution, "cudaMemcpy(se_convolution)");
    UploadFloats(device_se_output, se_output, "cudaMemcpy(se_output)");

    LaunchGlobalAveragePoolNchwKernel(device_convolution, device_pool, kBatch,
                                      kChannels, kSquares);
    LaunchSqueezeExciteResidualKernel(
        device_skip, device_convolution, device_se_output, device_residual,
        device_output, kBatch, kChannels, kSquares, CudaActivationKind::Relu);
    DownloadFloats(actual_pool, device_pool, "cudaMemcpy(se_pool)");
    DownloadFloats(actual_residual, device_residual, "cudaMemcpy(se_residual)");
    DownloadFloats(actual_output, device_output, "cudaMemcpy(se_activation)");

    for (std::size_t i = 0; i < expected_pool.size(); ++i) {
      if (std::fabs(actual_pool[i] - expected_pool[i]) > 1e-6f) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA squeeze-excite pool output mismatch";
        break;
      }
    }
    for (std::size_t i = 0;
         result.message.empty() && i < expected_residual.size(); ++i) {
      if (std::fabs(actual_residual[i] - expected_residual[i]) > 1e-6f ||
          std::fabs(actual_output[i] - expected_output[i]) > 1e-6f) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA squeeze-excite residual output mismatch";
        break;
      }
    }
    if (result.message.empty())
      result.status = CudaSmokeStatus::Success;
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
  }

  FreeDevice(device_skip);
  FreeDevice(device_convolution);
  FreeDevice(device_se_output);
  FreeDevice(device_pool);
  FreeDevice(device_residual);
  FreeDevice(device_output);

  if (result.message.empty())
    result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunResidualLayerNormKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kRows = 2;
  constexpr int kWidth = 4;
  constexpr float kScale = 0.5f;
  constexpr float kEpsilon = 1e-5f;
  const std::vector<float> parent = {
      1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 0.25f, 2.0f, -0.5f,
  };
  const std::vector<float> secondary = {
      2.0f, 0.5f, -4.0f, 1.5f, -2.5f, 3.0f, 0.25f, -1.0f,
  };
  const std::vector<float> bias = {0.25f, -0.5f, 1.0f, 0.125f};
  const std::vector<float> gamma = {1.0f, 0.5f, -1.25f, 2.0f};
  const std::vector<float> beta = {0.0f, -0.25f, 0.5f, 1.0f};
  std::vector<float> expected_biased(kRows * kWidth, 0.0f);
  std::vector<float> expected_residual(kRows * kWidth, 0.0f);
  std::vector<float> expected_output(kRows * kWidth, 0.0f);
  for (int row = 0; row < kRows; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * kWidth;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int col = 0; col < kWidth; ++col) {
      const std::size_t index = row_offset + col;
      const float biased = secondary[index] + bias[col];
      const float value = parent[index] + biased * kScale;
      expected_biased[index] = biased;
      expected_residual[index] = value;
      sum += value;
      square_sum += value * value;
    }
    const float mean = sum / static_cast<float>(kWidth);
    float variance = square_sum / static_cast<float>(kWidth) - mean * mean;
    if (variance < 0.0f)
      variance = 0.0f;
    const float inv_std = 1.0f / std::sqrt(variance + kEpsilon);
    for (int col = 0; col < kWidth; ++col) {
      const std::size_t index = row_offset + col;
      const float normalized = (expected_residual[index] - mean) * inv_std;
      expected_output[index] = normalized * gamma[col] + beta[col];
    }
  }

  std::vector<float> actual_biased(kRows * kWidth, 0.0f);
  std::vector<float> actual_residual(kRows * kWidth, 0.0f);
  std::vector<float> actual_output(kRows * kWidth, 0.0f);

  float *device_parent = nullptr;
  float *device_secondary = nullptr;
  float *device_bias = nullptr;
  float *device_gamma = nullptr;
  float *device_beta = nullptr;
  float *device_residual = nullptr;
  float *device_output = nullptr;
  try {
    AllocateDevice(&device_parent, parent.size(),
                   "cudaMalloc(residual_norm_parent)");
    AllocateDevice(&device_secondary, secondary.size(),
                   "cudaMalloc(residual_norm_secondary)");
    AllocateDevice(&device_bias, bias.size(), "cudaMalloc(residual_norm_bias)");
    AllocateDevice(&device_gamma, gamma.size(),
                   "cudaMalloc(residual_norm_gamma)");
    AllocateDevice(&device_beta, beta.size(), "cudaMalloc(residual_norm_beta)");
    AllocateDevice(&device_residual, actual_residual.size(),
                   "cudaMalloc(residual_norm_residual)");
    AllocateDevice(&device_output, actual_output.size(),
                   "cudaMalloc(residual_norm_output)");
    UploadFloats(device_parent, parent, "cudaMemcpy(residual_norm_parent)");
    UploadFloats(device_secondary, secondary,
                 "cudaMemcpy(residual_norm_secondary)");
    UploadFloats(device_bias, bias, "cudaMemcpy(residual_norm_bias)");
    UploadFloats(device_gamma, gamma, "cudaMemcpy(residual_norm_gamma)");
    UploadFloats(device_beta, beta, "cudaMemcpy(residual_norm_beta)");

    LaunchResidualBiasLayerNormKernel(
        device_parent, device_secondary, device_bias, device_gamma, device_beta,
        device_secondary, device_residual, device_output, kRows, kWidth, kScale,
        kEpsilon);
    DownloadFloats(actual_biased, device_secondary,
                   "cudaMemcpy(residual_norm_biased)");
    DownloadFloats(actual_residual, device_residual,
                   "cudaMemcpy(residual_norm_residual)");
    DownloadFloats(actual_output, device_output,
                   "cudaMemcpy(residual_norm_output)");
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
  }

  FreeDevice(device_parent);
  FreeDevice(device_secondary);
  FreeDevice(device_bias);
  FreeDevice(device_gamma);
  FreeDevice(device_beta);
  FreeDevice(device_residual);
  FreeDevice(device_output);

  if (!result.message.empty()) {
    return result;
  }

  for (std::size_t i = 0; i < expected_output.size(); ++i) {
    if (std::fabs(actual_biased[i] - expected_biased[i]) > 1e-6f ||
        std::fabs(actual_residual[i] - expected_residual[i]) > 1e-6f ||
        std::fabs(actual_output[i] - expected_output[i]) > 1e-5f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA residual bias layernorm kernel output mismatch";
      return result;
    }
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunAttentionCoreKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kHeads = 2;
  constexpr int kSquares = 4;
  constexpr int kHeadDepth = 3;
  constexpr int kQkv = kHeads * kHeadDepth;
  constexpr float kScale = 0.57735026919f;
  const int score_entries = kBatch * kHeads * kSquares * kSquares;
  const int qkv_entries = kBatch * kSquares * kQkv;

  std::vector<float> query(qkv_entries, 0.0f);
  std::vector<float> key(qkv_entries, 0.0f);
  std::vector<float> value(qkv_entries, 0.0f);
  for (std::size_t i = 0; i < query.size(); ++i) {
    query[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
    key[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.0625f;
    value[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.03125f;
  }

  std::vector<float> expected_scores(score_entries, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int head = 0; head < kHeads; ++head) {
      for (int query_square = 0; query_square < kSquares; ++query_square) {
        for (int key_square = 0; key_square < kSquares; ++key_square) {
          float dot = 0.0f;
          for (int depth = 0; depth < kHeadDepth; ++depth) {
            const int column = head * kHeadDepth + depth;
            dot +=
                query[(static_cast<std::size_t>(batch) * kSquares +
                       query_square) *
                          kQkv +
                      column] *
                key[(static_cast<std::size_t>(batch) * kSquares + key_square) *
                        kQkv +
                    column];
          }
          expected_scores[((batch * kHeads + head) * kSquares + query_square) *
                              kSquares +
                          key_square] = dot * kScale;
        }
      }
    }
  }

  std::vector<float> expected_probabilities(score_entries, 0.0f);
  const int softmax_rows = kBatch * kHeads * kSquares;
  for (int row = 0; row < softmax_rows; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * kSquares;
    float max_value = expected_scores[offset];
    for (int col = 1; col < kSquares; ++col)
      max_value = std::max(max_value, expected_scores[offset + col]);
    float sum = 0.0f;
    for (int col = 0; col < kSquares; ++col) {
      const float probability =
          std::exp(expected_scores[offset + col] - max_value);
      expected_probabilities[offset + col] = probability;
      sum += probability;
    }
    for (int col = 0; col < kSquares; ++col)
      expected_probabilities[offset + col] /= sum;
  }

  std::vector<float> expected_context(qkv_entries, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int query_square = 0; query_square < kSquares; ++query_square) {
      for (int column = 0; column < kQkv; ++column) {
        const int head = column / kHeadDepth;
        const std::size_t probability_offset =
            static_cast<std::size_t>((batch * kHeads + head) * kSquares +
                                     query_square) *
            kSquares;
        float sum = 0.0f;
        for (int key_square = 0; key_square < kSquares; ++key_square) {
          sum +=
              expected_probabilities[probability_offset + key_square] *
              value[(static_cast<std::size_t>(batch) * kSquares + key_square) *
                        kQkv +
                    column];
        }
        expected_context[(static_cast<std::size_t>(batch) * kSquares +
                          query_square) *
                             kQkv +
                         column] = sum;
      }
    }
  }

  std::vector<float> actual_scores(score_entries, 0.0f);
  std::vector<float> actual_probabilities(score_entries, 0.0f);
  std::vector<float> actual_context(qkv_entries, 0.0f);
  float *device_query = nullptr;
  float *device_key = nullptr;
  float *device_value = nullptr;
  float *device_scores = nullptr;
  float *device_probabilities = nullptr;
  float *device_context = nullptr;
  try {
    AllocateDevice(&device_query, query.size(), "cudaMalloc(attention_query)");
    AllocateDevice(&device_key, key.size(), "cudaMalloc(attention_key)");
    AllocateDevice(&device_value, value.size(), "cudaMalloc(attention_value)");
    AllocateDevice(&device_scores, actual_scores.size(),
                   "cudaMalloc(attention_scores)");
    AllocateDevice(&device_probabilities, actual_probabilities.size(),
                   "cudaMalloc(attention_probabilities)");
    AllocateDevice(&device_context, actual_context.size(),
                   "cudaMalloc(attention_context)");
    UploadFloats(device_query, query, "cudaMemcpy(attention_query)");
    UploadFloats(device_key, key, "cudaMemcpy(attention_key)");
    UploadFloats(device_value, value, "cudaMemcpy(attention_value)");

    LaunchAttentionScoreKernel(device_query, device_key, device_scores, kBatch,
                               kHeads, kSquares, kHeadDepth, kQkv, kScale);
    LaunchAttentionSoftmaxKernel(device_scores, device_probabilities,
                                 softmax_rows, kSquares);
    LaunchAttentionContextKernel(device_probabilities, device_value,
                                 device_context, kBatch, kHeads, kSquares,
                                 kHeadDepth, kQkv);
    DownloadFloats(actual_scores, device_scores,
                   "cudaMemcpy(attention_scores)");
    DownloadFloats(actual_probabilities, device_probabilities,
                   "cudaMemcpy(attention_probabilities)");
    DownloadFloats(actual_context, device_context,
                   "cudaMemcpy(attention_context)");
  } catch (const std::exception &e) {
    FreeDevice(device_query);
    FreeDevice(device_key);
    FreeDevice(device_value);
    FreeDevice(device_scores);
    FreeDevice(device_probabilities);
    FreeDevice(device_context);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_query);
  FreeDevice(device_key);
  FreeDevice(device_value);
  FreeDevice(device_scores);
  FreeDevice(device_probabilities);
  FreeDevice(device_context);

  for (std::size_t i = 0; i < expected_scores.size(); ++i) {
    if (std::fabs(actual_scores[i] - expected_scores[i]) > 1e-5f ||
        std::fabs(actual_probabilities[i] - expected_probabilities[i]) >
            1e-5f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention score/softmax output mismatch";
      return result;
    }
  }
  for (std::size_t i = 0; i < expected_context.size(); ++i) {
    if (std::fabs(actual_context[i] - expected_context[i]) > 1e-5f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention context output mismatch";
      return result;
    }
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunConvolutionPolicyMapKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  std::vector<float> raw_policy(kBatch * kNetworkConvPolicyScratch, 0.0f);
  for (std::size_t i = 0; i < raw_policy.size(); ++i)
    raw_policy[i] =
        static_cast<float>(static_cast<int>(i % 997) - 498) * 0.00390625f;

  std::vector<float> expected(kBatch * kNetworkPolicyOutputs, 0.0f);
  const auto &gather = ConvolutionPolicyGatherMap();
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int policy = 0; policy < kNetworkPolicyOutputs; ++policy) {
      const int raw = gather[static_cast<std::size_t>(policy)];
      expected[static_cast<std::size_t>(batch) * kNetworkPolicyOutputs +
               policy] = raw >= 0 ? raw_policy[static_cast<std::size_t>(batch) *
                                                   kNetworkConvPolicyScratch +
                                               raw]
                                  : 0.0f;
    }
  }

  std::vector<float> actual(expected.size(), 0.0f);
  float *device_raw = nullptr;
  float *device_policy = nullptr;
  try {
    AllocateDevice(&device_raw, raw_policy.size(),
                   "cudaMalloc(conv_policy_raw)");
    AllocateDevice(&device_policy, actual.size(),
                   "cudaMalloc(conv_policy_mapped)");
    UploadFloats(device_raw, raw_policy, "cudaMemcpy(conv_policy_raw)");

    LaunchConvolutionPolicyMapKernel(device_raw, device_policy, kBatch);
    DownloadFloats(actual, device_policy, "cudaMemcpy(conv_policy_mapped)");
  } catch (const std::exception &e) {
    FreeDevice(device_raw);
    FreeDevice(device_policy);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_raw);
  FreeDevice(device_policy);

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA convolution policy map output mismatch";
      return result;
    }
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunAttentionPolicyMapKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kSquares = kPackedInputSquareCount;
  constexpr int kChannels = 1;
  constexpr int kPromotionCount =
      kNetworkAttentionPolicyScratch - kSquares * kSquares;
  std::vector<float> query(kBatch * kSquares * kChannels, 0.0f);
  std::vector<float> key(kBatch * kSquares * kChannels, 0.0f);
  std::vector<float> promotion_weights(4 * kChannels, 0.0f);
  promotion_weights[0] = 1.0f;
  promotion_weights[1] = 3.0f;
  promotion_weights[2] = 5.0f;

  for (int batch = 0; batch < kBatch; ++batch) {
    query[static_cast<std::size_t>(batch) * kSquares + 0] =
        2.0f + static_cast<float>(batch);
    key[static_cast<std::size_t>(batch) * kSquares + 1] = 5.0f;
    query[static_cast<std::size_t>(batch) * kSquares + 48] =
        3.0f + static_cast<float>(batch);
    key[static_cast<std::size_t>(batch) * kSquares + 56] = 7.0f;
    key[static_cast<std::size_t>(batch) * kSquares + 57] = 11.0f;
  }

  std::vector<float> raw_policy(kBatch * kNetworkAttentionPolicyScratch, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int query_square = 0; query_square < kSquares; ++query_square) {
      for (int key_square = 0; key_square < kSquares; ++key_square) {
        raw_policy[static_cast<std::size_t>(batch) *
                       kNetworkAttentionPolicyScratch +
                   query_square * kSquares + key_square] =
            query[static_cast<std::size_t>(batch) * kSquares + query_square] *
            key[static_cast<std::size_t>(batch) * kSquares + key_square];
      }
    }
    for (int promo = 0; promo < kPromotionCount; ++promo) {
      const int promotion_row = promo % 3;
      const int promotion_key_square = 56 + (promo % 24) / 3;
      const int square_pair = promo % kSquares;
      const int query_square = 48 + square_pair / 8;
      const int key_square = 56 + square_pair % 8;
      raw_policy[static_cast<std::size_t>(batch) *
                     kNetworkAttentionPolicyScratch +
                 kSquares * kSquares + promo] =
          query[static_cast<std::size_t>(batch) * kSquares + query_square] *
              key[static_cast<std::size_t>(batch) * kSquares + key_square] +
          key[static_cast<std::size_t>(batch) * kSquares +
              promotion_key_square] *
              (promotion_weights[static_cast<std::size_t>(promotion_row)] +
               promotion_weights[3]);
    }
  }

  std::vector<float> expected(kBatch * kNetworkPolicyOutputs, 0.0f);
  const auto &gather = AttentionPolicyGatherMap();
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int policy = 0; policy < kNetworkPolicyOutputs; ++policy) {
      const int raw = gather[static_cast<std::size_t>(policy)];
      expected[static_cast<std::size_t>(batch) * kNetworkPolicyOutputs +
               policy] = raw >= 0
                             ? raw_policy[static_cast<std::size_t>(batch) *
                                              kNetworkAttentionPolicyScratch +
                                          raw]
                             : 0.0f;
    }
  }

  std::vector<float> actual(expected.size(), 0.0f);
  float *device_query = nullptr;
  float *device_key = nullptr;
  float *device_promotion = nullptr;
  float *device_raw = nullptr;
  float *device_policy = nullptr;
  try {
    AllocateDevice(&device_query, query.size(),
                   "cudaMalloc(attention_policy_query)");
    AllocateDevice(&device_key, key.size(), "cudaMalloc(attention_policy_key)");
    AllocateDevice(&device_promotion, promotion_weights.size(),
                   "cudaMalloc(attention_policy_promotion)");
    AllocateDevice(&device_raw, raw_policy.size(),
                   "cudaMalloc(attention_policy_raw)");
    AllocateDevice(&device_policy, actual.size(),
                   "cudaMalloc(attention_policy_mapped)");
    UploadFloats(device_query, query, "cudaMemcpy(attention_policy_query)");
    UploadFloats(device_key, key, "cudaMemcpy(attention_policy_key)");
    UploadFloats(device_promotion, promotion_weights,
                 "cudaMemcpy(attention_policy_promotion)");

    LaunchAttentionPolicyMapKernel(device_query, device_key, device_promotion,
                                   device_raw, device_policy, kBatch,
                                   kChannels);
    DownloadFloats(actual, device_policy,
                   "cudaMemcpy(attention_policy_mapped)");
  } catch (const std::exception &e) {
    FreeDevice(device_query);
    FreeDevice(device_key);
    FreeDevice(device_promotion);
    FreeDevice(device_raw);
    FreeDevice(device_policy);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_query);
  FreeDevice(device_key);
  FreeDevice(device_promotion);
  FreeDevice(device_raw);
  FreeDevice(device_policy);

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > 1e-6f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention policy map output mismatch";
      return result;
    }
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaKernelSmokeResult RunDynamicPositionEncodingKernelSmoke() {
  CudaKernelSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kPlanes = kPackedInputPlaneCount;
  constexpr int kSquares = kPackedInputSquareCount;
  constexpr int kPositionPlanes = 12;
  constexpr int kPositionWidth = 5;
  constexpr int kOutputWidth = kPlanes + kPositionWidth;
  constexpr int kStaticPositionWidth = Tables::kNumPosEncodingChannels;
  constexpr int kStaticOutputWidth = kPlanes + kStaticPositionWidth;

  std::vector<std::uint64_t> masks(kBatch * kPlanes, 0);
  std::vector<float> values(kBatch * kPlanes, 0.0f);
  auto set_plane = [&](int batch, int plane, std::uint64_t mask, float value) {
    const std::size_t index = static_cast<std::size_t>(batch) * kPlanes + plane;
    masks[index] = mask;
    values[index] = value;
  };
  set_plane(0, 0, (1ULL << 1) | (1ULL << 3), 2.0f);
  set_plane(0, 1, ~0ULL, -1.0f);
  set_plane(0, 13, 1ULL << 7, 4.0f);
  set_plane(1, 0, 1ULL << 5, 0.5f);
  set_plane(1, 11, 1ULL << 63, 3.0f);
  set_plane(1, 12, ~0ULL, 8.0f);

  std::vector<float> expected_expanded(kBatch * kSquares * kPlanes, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int square = 0; square < kSquares; ++square) {
      for (int plane = 0; plane < kPlanes; ++plane) {
        const std::size_t packed =
            static_cast<std::size_t>(batch) * kPlanes + plane;
        const std::size_t expanded =
            (static_cast<std::size_t>(batch) * kSquares + square) * kPlanes +
            plane;
        expected_expanded[expanded] =
            (masks[packed] & (1ULL << square)) ? values[packed] : 0.0f;
      }
    }
  }

  std::vector<float> expected_position_input(
      kBatch * kSquares * kPositionPlanes, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int square = 0; square < kSquares; ++square) {
      for (int plane = 0; plane < kPositionPlanes; ++plane) {
        expected_position_input[(static_cast<std::size_t>(batch) * kSquares +
                                 square) *
                                    kPositionPlanes +
                                plane] = expected_expanded
            [(static_cast<std::size_t>(batch) * kSquares + square) * kPlanes +
             plane];
      }
    }
  }

  std::vector<float> position_encoding(kBatch * kSquares * kPositionWidth,
                                       0.0f);
  for (std::size_t i = 0; i < position_encoding.size(); ++i) {
    position_encoding[i] =
        static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
  }

  constexpr int kSparseDenseInputWidth = kSquares * kPositionPlanes;
  constexpr int kSparseDenseOutputWidth = 7;
  std::vector<float> sparse_weights(kSparseDenseOutputWidth *
                                        kSparseDenseInputWidth,
                                    0.0f);
  std::vector<float> sparse_bias(kSparseDenseOutputWidth, 0.0f);
  for (std::size_t i = 0; i < sparse_weights.size(); ++i) {
    sparse_weights[i] =
        static_cast<float>(static_cast<int>(i % 23) - 11) * 0.03125f;
  }
  for (std::size_t i = 0; i < sparse_bias.size(); ++i) {
    sparse_bias[i] = static_cast<float>(i) * 0.125f - 0.25f;
  }

  std::vector<float> expected_sparse_dense(
      kBatch * kSparseDenseOutputWidth, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int output = 0; output < kSparseDenseOutputWidth; ++output) {
      float sum = sparse_bias[output];
      for (int plane = 0; plane < kPositionPlanes; ++plane) {
        const std::size_t packed =
            static_cast<std::size_t>(batch) * kPlanes + plane;
        for (int square = 0; square < kSquares; ++square) {
          if ((masks[packed] & (1ULL << square)) == 0)
            continue;
          const int input_index = square * kPositionPlanes + plane;
          sum += values[packed] *
                 sparse_weights[static_cast<std::size_t>(output) *
                                    kSparseDenseInputWidth +
                                input_index];
        }
      }
      expected_sparse_dense[static_cast<std::size_t>(batch) *
                                kSparseDenseOutputWidth +
                            output] = sum;
    }
  }

  std::vector<float> expected_output(kBatch * kSquares * kOutputWidth, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int square = 0; square < kSquares; ++square) {
      for (int channel = 0; channel < kOutputWidth; ++channel) {
        const std::size_t output_index =
            (static_cast<std::size_t>(batch) * kSquares + square) *
                kOutputWidth +
            channel;
        if (channel < kPlanes) {
          expected_output[output_index] = expected_expanded
              [(static_cast<std::size_t>(batch) * kSquares + square) * kPlanes +
               channel];
        } else {
          expected_output[output_index] =
              position_encoding[(static_cast<std::size_t>(batch) * kSquares +
                                 square) *
                                    kPositionWidth +
                                channel - kPlanes];
        }
      }
    }
  }

  std::vector<float> actual_expanded(expected_expanded.size(), 0.0f);
  std::vector<float> actual_position_input(expected_position_input.size(),
                                           0.0f);
  std::vector<float> actual_sparse_dense(expected_sparse_dense.size(), 0.0f);
  std::vector<float> actual_output(expected_output.size(), 0.0f);
  std::vector<float> expected_static_output(
      kBatch * kSquares * kStaticOutputWidth, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int square = 0; square < kSquares; ++square) {
      for (int channel = 0; channel < kStaticOutputWidth; ++channel) {
        const std::size_t output_index =
            (static_cast<std::size_t>(batch) * kSquares + square) *
                kStaticOutputWidth +
            channel;
        if (channel < kPlanes) {
          expected_static_output[output_index] = expected_expanded
              [(static_cast<std::size_t>(batch) * kSquares + square) * kPlanes +
               channel];
        } else {
          expected_static_output[output_index] =
              Tables::kPosEncoding[square][channel - kPlanes];
        }
      }
    }
  }
  std::vector<float> actual_static_output(expected_static_output.size(), 0.0f);
  std::uint64_t *device_masks = nullptr;
  float *device_values = nullptr;
  float *device_expanded = nullptr;
  float *device_position_input = nullptr;
  float *device_position_encoding = nullptr;
  float *device_sparse_weights = nullptr;
  float *device_sparse_bias = nullptr;
  float *device_sparse_output = nullptr;
  float *device_output = nullptr;
  float *device_static_output = nullptr;

  try {
    AllocateDevice(&device_masks, masks.size(), "cudaMalloc(dynamic_masks)");
    AllocateDevice(&device_values, values.size(), "cudaMalloc(dynamic_values)");
    AllocateDevice(&device_expanded, actual_expanded.size(),
                   "cudaMalloc(dynamic_expanded)");
    AllocateDevice(&device_position_input, actual_position_input.size(),
                   "cudaMalloc(dynamic_position_input)");
    AllocateDevice(&device_position_encoding, position_encoding.size(),
                   "cudaMalloc(dynamic_position_encoding)");
    AllocateDevice(&device_sparse_weights, sparse_weights.size(),
                   "cudaMalloc(dynamic_sparse_weights)");
    AllocateDevice(&device_sparse_bias, sparse_bias.size(),
                   "cudaMalloc(dynamic_sparse_bias)");
    AllocateDevice(&device_sparse_output, actual_sparse_dense.size(),
                   "cudaMalloc(dynamic_sparse_output)");
    AllocateDevice(&device_output, actual_output.size(),
                   "cudaMalloc(dynamic_position_output)");
    AllocateDevice(&device_static_output, actual_static_output.size(),
                   "cudaMalloc(static_position_output)");

    cudaError_t status = cudaMemcpy(device_masks, masks.data(),
                                    masks.size() * sizeof(std::uint64_t),
                                    cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      throw std::runtime_error(
          CudaErrorMessage("cudaMemcpy(dynamic_masks)", status));
    UploadFloats(device_values, values, "cudaMemcpy(dynamic_values)");
    UploadFloats(device_position_encoding, position_encoding,
                 "cudaMemcpy(dynamic_position_encoding)");
    UploadFloats(device_sparse_weights, sparse_weights,
                 "cudaMemcpy(dynamic_sparse_weights)");
    UploadFloats(device_sparse_bias, sparse_bias,
                 "cudaMemcpy(dynamic_sparse_bias)");

    LaunchExpandPackedInputPlanesWithPositionInputKernel(
        device_masks, device_values, device_expanded, device_position_input,
        kBatch, kPlanes, kPositionPlanes, kSquares);
    LaunchDynamicPositionEncodingSparseDenseKernel(
        device_masks, device_values, device_sparse_weights, device_sparse_bias,
        device_sparse_output, kBatch, kPlanes, kPositionPlanes, kSquares,
        kSparseDenseInputWidth, kSparseDenseOutputWidth);
    LaunchDynamicPositionEncodingConcatKernel(
        device_expanded, device_position_encoding, device_output, kBatch,
        kPlanes, kPositionWidth, kSquares);
    LaunchStaticPositionEncodingConcatKernel(
        device_masks, device_values, device_static_output, kBatch, kPlanes,
        kStaticPositionWidth, kSquares);

    DownloadFloats(actual_expanded, device_expanded,
                   "cudaMemcpy(dynamic_expanded)");
    DownloadFloats(actual_position_input, device_position_input,
                   "cudaMemcpy(dynamic_position_input)");
    DownloadFloats(actual_sparse_dense, device_sparse_output,
                   "cudaMemcpy(dynamic_sparse_output)");
    DownloadFloats(actual_output, device_output,
                   "cudaMemcpy(dynamic_position_output)");
    DownloadFloats(actual_static_output, device_static_output,
                   "cudaMemcpy(static_position_output)");
  } catch (const std::exception &e) {
    FreeDevice(device_masks);
    FreeDevice(device_values);
    FreeDevice(device_expanded);
    FreeDevice(device_position_input);
    FreeDevice(device_position_encoding);
    FreeDevice(device_sparse_weights);
    FreeDevice(device_sparse_bias);
    FreeDevice(device_sparse_output);
    FreeDevice(device_output);
    FreeDevice(device_static_output);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_masks);
  FreeDevice(device_values);
  FreeDevice(device_expanded);
  FreeDevice(device_position_input);
  FreeDevice(device_position_encoding);
  FreeDevice(device_sparse_weights);
  FreeDevice(device_sparse_bias);
  FreeDevice(device_sparse_output);
  FreeDevice(device_output);
  FreeDevice(device_static_output);

  for (std::size_t i = 0; i < expected_expanded.size(); ++i) {
    if (actual_expanded[i] != expected_expanded[i]) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic input expansion mismatch";
      return result;
    }
  }
  for (std::size_t i = 0; i < expected_position_input.size(); ++i) {
    if (actual_position_input[i] != expected_position_input[i]) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic position input mismatch";
      return result;
    }
  }
  for (std::size_t i = 0; i < expected_sparse_dense.size(); ++i) {
    if (std::fabs(actual_sparse_dense[i] - expected_sparse_dense[i]) > 1e-6f) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic sparse position dense mismatch";
      return result;
    }
  }
  for (std::size_t i = 0; i < expected_output.size(); ++i) {
    if (actual_output[i] != expected_output[i]) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic position concat mismatch";
      return result;
    }
  }
  for (std::size_t i = 0; i < expected_static_output.size(); ++i) {
    if (actual_static_output[i] != expected_static_output[i]) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA static position concat mismatch";
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

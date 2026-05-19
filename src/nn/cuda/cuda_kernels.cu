/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_kernels.h"

#include "cuda_runtime_probe.h"
#include "../metal/tables/attention_policy_map.h"
#include "../network_tensor_plan.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
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

const std::array<int, kNetworkPolicyOutputs> &AttentionPolicyGatherMap() {
  static const auto gather = [] {
    std::array<int, kNetworkPolicyOutputs> indices{};
    indices.fill(-1);
    for (int raw = 0; raw < kNetworkAttentionPolicyScratch; ++raw) {
      const short mapped = Metal::kAttnPolicyMap[raw];
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
                           int gate_rows, CudaGateKind kind) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int row = index / width;
  const int column = index % width;
  const int gate_row = gate_rows == 1 ? 0 : row % gate_rows;
  const float gate = gate_rows == 1 ? weights[column]
                                    : weights[column * gate_rows + gate_row];
  const float value = input[index];
  output[index] =
      kind == CudaGateKind::Add ? value + gate : value * gate;
}

__global__ void ResidualAddKernel(const float *parent, const float *secondary,
                                  float *output, int total,
                                  float secondary_scale) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;
  output[index] = parent[index] + secondary[index] * secondary_scale;
}

__global__ void AttentionScoreKernel(const float *query, const float *key,
                                     float *scores, int heads, int squares,
                                     int head_depth, int qkv_width,
                                     float scale, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int key_square = index % squares;
  const int query_square = (index / squares) % squares;
  const int head = (index / (squares * squares)) % heads;
  const int batch = index / (heads * squares * squares);
  const int head_offset = head * head_depth;
  const float *query_row =
      query + (static_cast<std::size_t>(batch) * squares + query_square) *
                  qkv_width +
      head_offset;
  const float *key_row =
      key + (static_cast<std::size_t>(batch) * squares + key_square) *
                qkv_width +
      head_offset;

  float dot = 0.0f;
  for (int i = 0; i < head_depth; ++i)
    dot += query_row[i] * key_row[i];
  scores[index] = dot * scale;
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

__global__ void AttentionContextKernel(const float *probabilities,
                                       const float *value, float *context,
                                       int heads, int squares, int head_depth,
                                       int qkv_width, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int column = index % qkv_width;
  const int query_square = (index / qkv_width) % squares;
  const int batch = index / (squares * qkv_width);
  const int head = column / head_depth;
  const int probability_row =
      ((batch * heads + head) * squares + query_square) * squares;

  float sum = 0.0f;
  for (int key_square = 0; key_square < squares; ++key_square) {
    const float probability = probabilities[probability_row + key_square];
    const float value_cell =
        value[(static_cast<std::size_t>(batch) * squares + key_square) *
                  qkv_width +
              column];
    sum += probability * value_cell;
  }
  context[index] = sum;
}

__device__ __constant__ int kAttentionPolicyGatherDevice[kNetworkPolicyOutputs];

__global__ void AttentionPolicyMapKernel(const float *query, const float *key,
                                         const float *promotion_weights,
                                         float *raw_policy, int channels,
                                         int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int raw_index = index % kNetworkAttentionPolicyScratch;
  const int batch = index / kNetworkAttentionPolicyScratch;
  constexpr int kSquares = kPackedInputSquareCount;
  float value = 0.0f;

  if (raw_index < kSquares * kSquares) {
    const int query_square = raw_index / kSquares;
    const int key_square = raw_index % kSquares;
    const float *query_row =
        query + (static_cast<std::size_t>(batch) * kSquares + query_square) *
                    channels;
    const float *key_row =
        key + (static_cast<std::size_t>(batch) * kSquares + key_square) *
                  channels;
    for (int channel = 0; channel < channels; ++channel)
      value += query_row[channel] * key_row[channel];
    value *= rsqrtf(static_cast<float>(channels));
  } else {
    const int promo_index = raw_index - kSquares * kSquares;
    const int query_square = 48 + promo_index / 24;
    const int key_square = 56 + (promo_index % 24) / 3;
    const int promotion_row = promo_index % 3;
    const float *query_row =
        query + (static_cast<std::size_t>(batch) * kSquares + query_square) *
                    channels;
    const float *key_row =
        key + (static_cast<std::size_t>(batch) * kSquares + key_square) *
                  channels;
    for (int channel = 0; channel < channels; ++channel) {
      value += query_row[channel] * key_row[channel] *
               rsqrtf(static_cast<float>(channels));
      value += key_row[channel] *
               (promotion_weights[promotion_row * channels + channel] +
                promotion_weights[3 * channels + channel]);
    }
  }

  raw_policy[index] = value;
}

__global__ void AttentionPolicyGatherKernel(const float *raw_policy,
                                            float *policy, int total) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total)
    return;

  const int policy_index = index % kNetworkPolicyOutputs;
  const int batch = index / kNetworkPolicyOutputs;
  const int raw_index = kAttentionPolicyGatherDevice[policy_index];
  policy[index] =
      raw_index >= 0
          ? raw_policy[static_cast<std::size_t>(batch) *
                           kNetworkAttentionPolicyScratch +
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
    throw std::runtime_error(CudaErrorMessage("GateKernel synchronize",
                                             status));
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
  ResidualAddKernel<<<blocks, kThreads, 0, stream>>>(
      parent, secondary, output, total, secondary_scale);
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

void LaunchAttentionScoreKernel(const float *query, const float *key,
                                float *scores, int batch_size, int heads,
                                int squares, int head_depth, int qkv_width,
                                float scale, cudaStream_t stream) {
  if (!query || !key || !scores)
    throw std::runtime_error("CUDA attention score kernel received null buffer");
  if (batch_size <= 0 || heads <= 0 || squares <= 0 || head_depth <= 0 ||
      qkv_width <= 0 || qkv_width != heads * head_depth || scale <= 0.0f) {
    throw std::runtime_error("CUDA attention score dimensions are invalid");
  }

  const int total = batch_size * heads * squares * squares;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  AttentionScoreKernel<<<blocks, kThreads, 0, stream>>>(
      query, key, scores, heads, squares, head_depth, qkv_width, scale, total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionScoreKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
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
                                  int rows, int width, cudaStream_t stream) {
  if (!scores || !probabilities)
    throw std::runtime_error(
        "CUDA attention softmax kernel received null buffer");
  if (rows <= 0 || width <= 0)
    throw std::runtime_error("CUDA attention softmax dimensions are invalid");

  constexpr int kThreads = 128;
  const std::size_t shared_bytes = kThreads * sizeof(float);
  AttentionSoftmaxKernel<<<rows, kThreads, shared_bytes, stream>>>(
      scores, probabilities, width);
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

  const int total = batch_size * squares * qkv_width;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  AttentionContextKernel<<<blocks, kThreads, 0, stream>>>(
      probabilities, value, context, heads, squares, head_depth, qkv_width,
      total);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(
        CudaErrorMessage("AttentionContextKernel launch", status));
  if (stream)
    return;
  status = cudaDeviceSynchronize();
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
    throw std::runtime_error("CUDA attention policy map dimensions are invalid");
  }

  const auto &gather_map = AttentionPolicyGatherMap();
  cudaError_t status = cudaMemcpyToSymbolAsync(
      kAttentionPolicyGatherDevice, gather_map.data(),
      gather_map.size() * sizeof(int), 0, cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("AttentionPolicyGather table upload", status));
  }

  const int total = batch_size * kNetworkAttentionPolicyScratch;
  constexpr int kThreads = 256;
  const int blocks = (total + kThreads - 1) / kThreads;
  AttentionPolicyMapKernel<<<blocks, kThreads, 0, stream>>>(
      query, key, promotion_weights, raw_policy, channels, total);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(
        CudaErrorMessage("AttentionPolicyMapKernel launch", status));
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

void LaunchExpandPackedInputPlanesKernel(const std::uint64_t *masks,
                                         const float *values, float *expanded,
                                         int batch_size, int planes,
                                         int squares, cudaStream_t stream) {
  if (!masks || !values || !expanded)
    throw std::runtime_error("CUDA input expansion kernel received null buffer");
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
    throw std::runtime_error(CudaErrorMessage(
        "ExpandPackedInputPlanesKernel synchronize", status));
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
    throw std::runtime_error(CudaErrorMessage(
        "DynamicPositionEncodingInputKernel launch", status));
  }
  if (stream)
    return;
  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    throw std::runtime_error(CudaErrorMessage(
        "DynamicPositionEncodingInputKernel synchronize", status));
  }
}

void LaunchDynamicPositionEncodingConcatKernel(
    const float *expanded, const float *position_encoding, float *output,
    int batch_size, int input_planes, int position_width, int squares,
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
    throw std::runtime_error(CudaErrorMessage(
        "DynamicPositionEncodingConcatKernel launch", status));
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

  constexpr int kRows = 4;
  constexpr int kWidth = 4;
  const std::vector<float> input = {
      1.0f, -2.0f, 0.5f, 3.0f,
      -1.0f, 0.25f, 2.0f, -0.5f,
      0.75f, -1.25f, 1.5f, 0.0f,
      -0.75f, 1.25f, -1.5f, 0.5f,
  };
  const std::vector<float> mult_gate = {
      2.0f, -0.5f, 4.0f, 0.25f,
      -1.0f, 0.75f, 0.5f, 3.0f,
  };
  const std::vector<float> add_gate = {
      0.5f, 1.0f, -2.0f, 3.0f,
      -0.25f, 0.5f, 1.5f, -1.0f,
  };
  std::vector<float> actual(kRows * kWidth, 0.0f);
  std::vector<float> expected(kRows * kWidth, 0.0f);

  for (int b = 0; b < kRows; ++b) {
    for (int i = 0; i < kWidth; ++i) {
      const std::size_t index = static_cast<std::size_t>(b) * kWidth + i;
      const std::size_t gate_index =
          static_cast<std::size_t>(i) * 2 + (b % 2);
      expected[index] = input[index] * mult_gate[gate_index] +
                        add_gate[gate_index];
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

    LaunchGateKernel(device_input, device_mult, device_output, kRows, kWidth,
                     2, CudaGateKind::Multiply);
    LaunchGateKernel(device_output, device_add, device_output, kRows, kWidth,
                     2, CudaGateKind::Add);
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
      1.0f, -2.0f, 0.5f, 3.0f,
      -1.0f, 0.25f, 2.0f, -0.5f,
  };
  const std::vector<float> secondary = {
      2.0f, 0.5f, -4.0f, 1.5f,
      -2.5f, 3.0f, 0.25f, -1.0f,
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
    UploadFloats(device_secondary, secondary,
                 "cudaMemcpy(residual_secondary)");

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
            dot += query[(static_cast<std::size_t>(batch) * kSquares +
                          query_square) *
                             kQkv +
                         column] *
                   key[(static_cast<std::size_t>(batch) * kSquares +
                        key_square) *
                           kQkv +
                       column];
          }
          expected_scores[((batch * kHeads + head) * kSquares +
                           query_square) *
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
      const float probability = std::exp(expected_scores[offset + col] -
                                         max_value);
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
          sum += expected_probabilities[probability_offset + key_square] *
                 value[(static_cast<std::size_t>(batch) * kSquares +
                        key_square) *
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
                                plane] =
            expected_expanded[(static_cast<std::size_t>(batch) * kSquares +
                               square) *
                                  kPlanes +
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

  std::vector<float> expected_output(kBatch * kSquares * kOutputWidth, 0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int square = 0; square < kSquares; ++square) {
      for (int channel = 0; channel < kOutputWidth; ++channel) {
        const std::size_t output_index =
            (static_cast<std::size_t>(batch) * kSquares + square) *
                kOutputWidth +
            channel;
        if (channel < kPlanes) {
          expected_output[output_index] =
              expected_expanded[(static_cast<std::size_t>(batch) * kSquares +
                                 square) *
                                    kPlanes +
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
  std::vector<float> actual_output(expected_output.size(), 0.0f);
  std::uint64_t *device_masks = nullptr;
  float *device_values = nullptr;
  float *device_expanded = nullptr;
  float *device_position_input = nullptr;
  float *device_position_encoding = nullptr;
  float *device_output = nullptr;

  try {
    AllocateDevice(&device_masks, masks.size(), "cudaMalloc(dynamic_masks)");
    AllocateDevice(&device_values, values.size(), "cudaMalloc(dynamic_values)");
    AllocateDevice(&device_expanded, actual_expanded.size(),
                   "cudaMalloc(dynamic_expanded)");
    AllocateDevice(&device_position_input, actual_position_input.size(),
                   "cudaMalloc(dynamic_position_input)");
    AllocateDevice(&device_position_encoding, position_encoding.size(),
                   "cudaMalloc(dynamic_position_encoding)");
    AllocateDevice(&device_output, actual_output.size(),
                   "cudaMalloc(dynamic_position_output)");

    cudaError_t status =
        cudaMemcpy(device_masks, masks.data(),
                   masks.size() * sizeof(std::uint64_t),
                   cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
      throw std::runtime_error(CudaErrorMessage("cudaMemcpy(dynamic_masks)",
                                               status));
    UploadFloats(device_values, values, "cudaMemcpy(dynamic_values)");
    UploadFloats(device_position_encoding, position_encoding,
                 "cudaMemcpy(dynamic_position_encoding)");

    LaunchExpandPackedInputPlanesKernel(device_masks, device_values,
                                        device_expanded, kBatch, kPlanes,
                                        kSquares);
    LaunchDynamicPositionEncodingInputKernel(
        device_expanded, device_position_input, kBatch, kPlanes,
        kPositionPlanes, kSquares);
    LaunchDynamicPositionEncodingConcatKernel(
        device_expanded, device_position_encoding, device_output, kBatch,
        kPlanes, kPositionWidth, kSquares);

    DownloadFloats(actual_expanded, device_expanded,
                   "cudaMemcpy(dynamic_expanded)");
    DownloadFloats(actual_position_input, device_position_input,
                   "cudaMemcpy(dynamic_position_input)");
    DownloadFloats(actual_output, device_output,
                   "cudaMemcpy(dynamic_position_output)");
  } catch (const std::exception &e) {
    FreeDevice(device_masks);
    FreeDevice(device_values);
    FreeDevice(device_expanded);
    FreeDevice(device_position_input);
    FreeDevice(device_position_encoding);
    FreeDevice(device_output);
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  FreeDevice(device_masks);
  FreeDevice(device_values);
  FreeDevice(device_expanded);
  FreeDevice(device_position_input);
  FreeDevice(device_position_encoding);
  FreeDevice(device_output);

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
  for (std::size_t i = 0; i < expected_output.size(); ++i) {
    if (actual_output[i] != expected_output[i]) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic position concat mismatch";
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

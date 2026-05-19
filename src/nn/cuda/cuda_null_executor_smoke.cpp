/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_buffers.h"

#include "../network_output_decoder.h"
#include "../network_weight_inventory.h"
#include "cuda_attention_plan.h"
#include "cuda_executor.h"
#include "cuda_kernels.h"
#include "cuda_runtime_probe.h"
#include "cuda_stage_executor.h"
#include "cuda_workspace.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

void UploadFloats(float *device, const std::vector<float> &host,
                  cudaStream_t stream, const char *name) {
  if (host.empty())
    return;
  const cudaError_t status = cudaMemcpyAsync(
      device, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice,
      stream);
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

std::vector<float> DenseAffineHost(const std::vector<float> &input,
                                   const std::vector<float> &weights,
                                   const std::vector<float> &bias, int rows,
                                   int input_width, int output_width) {
  std::vector<float> output(static_cast<std::size_t>(rows) * output_width,
                            0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int out = 0; out < output_width; ++out) {
      float sum = bias[static_cast<std::size_t>(out)];
      for (int in = 0; in < input_width; ++in) {
        sum += input[static_cast<std::size_t>(row) * input_width + in] *
               weights[static_cast<std::size_t>(out) * input_width + in];
      }
      output[static_cast<std::size_t>(row) * output_width + out] = sum;
    }
  }
  return output;
}

std::vector<float> AttentionScoresHost(const std::vector<float> &query,
                                       const std::vector<float> &key,
                                       int batch_size, int heads, int squares,
                                       int head_depth, int qkv_width,
                                       float scale) {
  std::vector<float> scores(
      static_cast<std::size_t>(batch_size) * heads * squares * squares, 0.0f);
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int head = 0; head < heads; ++head) {
      for (int query_square = 0; query_square < squares; ++query_square) {
        for (int key_square = 0; key_square < squares; ++key_square) {
          float dot = 0.0f;
          for (int depth = 0; depth < head_depth; ++depth) {
            const int column = head * head_depth + depth;
            dot += query[(static_cast<std::size_t>(batch) * squares +
                          query_square) *
                             qkv_width +
                         column] *
                   key[(static_cast<std::size_t>(batch) * squares +
                        key_square) *
                           qkv_width +
                       column];
          }
          scores[((batch * heads + head) * squares + query_square) * squares +
                 key_square] = dot * scale;
        }
      }
    }
  }
  return scores;
}

std::vector<float> SoftmaxRowsHost(const std::vector<float> &scores, int rows,
                                   int width) {
  std::vector<float> probabilities(scores.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * width;
    float max_value = scores[offset];
    for (int col = 1; col < width; ++col)
      max_value = std::max(max_value, scores[offset + col]);
    float sum = 0.0f;
    for (int col = 0; col < width; ++col) {
      const float probability = std::exp(scores[offset + col] - max_value);
      probabilities[offset + col] = probability;
      sum += probability;
    }
    for (int col = 0; col < width; ++col)
      probabilities[offset + col] /= sum;
  }
  return probabilities;
}

std::vector<float> LayerNormRowsHost(const std::vector<float> &input,
                                     const std::vector<float> &gamma,
                                     const std::vector<float> &beta, int rows,
                                     int width, float epsilon) {
  std::vector<float> output(input.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * width;
    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int col = 0; col < width; ++col) {
      const float value = input[offset + col];
      sum += value;
      square_sum += value * value;
    }
    const float mean = sum / static_cast<float>(width);
    float variance = square_sum / static_cast<float>(width) - mean * mean;
    if (variance < 0.0f)
      variance = 0.0f;
    const float inv_std = 1.0f / std::sqrt(variance + epsilon);
    for (int col = 0; col < width; ++col) {
      const float normalized = (input[offset + col] - mean) * inv_std;
      output[offset + col] = normalized * gamma[static_cast<std::size_t>(col)] +
                             beta[static_cast<std::size_t>(col)];
    }
  }
  return output;
}

float ActivationValueHost(float value, CudaActivationKind kind) {
  switch (kind) {
  case CudaActivationKind::Relu:
    return std::max(value, 0.0f);
  case CudaActivationKind::Relu2: {
    const float relu = std::max(value, 0.0f);
    return relu * relu;
  }
  case CudaActivationKind::Tanh:
    return std::tanh(value);
  case CudaActivationKind::Sigmoid:
    return 1.0f / (1.0f + std::exp(-value));
  case CudaActivationKind::Swish:
    return value / (1.0f + std::exp(-value));
  case CudaActivationKind::Mish:
    return value * std::tanh(std::log1p(std::exp(value)));
  case CudaActivationKind::Selu:
    return 1.05070098f *
           (value > 0.0f ? value
                          : 1.67326324f * (std::exp(value) - 1.0f));
  }
  return value;
}

std::vector<float> ActivationHost(const std::vector<float> &input,
                                  CudaActivationKind kind) {
  std::vector<float> output(input.size(), 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i)
    output[i] = ActivationValueHost(input[i], kind);
  return output;
}

std::vector<float> AttentionContextHost(
    const std::vector<float> &probabilities, const std::vector<float> &value,
    int batch_size, int heads, int squares, int head_depth, int qkv_width) {
  std::vector<float> context(
      static_cast<std::size_t>(batch_size) * squares * qkv_width, 0.0f);
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int query_square = 0; query_square < squares; ++query_square) {
      for (int column = 0; column < qkv_width; ++column) {
        const int head = column / head_depth;
        const std::size_t probability_offset =
            static_cast<std::size_t>((batch * heads + head) * squares +
                                     query_square) *
            squares;
        float sum = 0.0f;
        for (int key_square = 0; key_square < squares; ++key_square) {
          sum += probabilities[probability_offset + key_square] *
                 value[(static_cast<std::size_t>(batch) * squares +
                        key_square) *
                           qkv_width +
                       column];
        }
        context[(static_cast<std::size_t>(batch) * squares + query_square) *
                    qkv_width +
                column] = sum;
      }
    }
  }
  return context;
}

bool AlmostEqual(const std::vector<float> &actual,
                 const std::vector<float> &expected, float tolerance) {
  if (actual.size() != expected.size())
    return false;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::fabs(actual[i] - expected[i]) > tolerance)
      return false;
  }
  return true;
}

} // namespace

CudaBufferSmokeResult RunNullExecutorPipelineSmokeRaw(const float *inputs,
                                                      int batch_size) {
  CudaBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }
  if (inputs == nullptr || batch_size <= 0) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = "CUDA null executor smoke received invalid input";
    return result;
  }

  NetworkFormatDescriptor format;
  format.wdl = true;
  format.moves_left = true;
  format.attention_policy = true;
  const auto plan = CreateNetworkTensorPlan(format);
  const auto layout = LayoutFromTensorPlan(plan, batch_size);

  std::vector<std::uint64_t> masks;
  std::vector<float> values;
  PackInputPlanesHostRaw(inputs, batch_size, masks, values);

  try {
    CudaInferenceBuffers buffers;
    buffers.Allocate(layout);
    buffers.UploadPackedInputs(masks, values, batch_size);
    buffers.ClearOutputs(batch_size);
    auto executor = CreateNullCudaExecutorForSmoke();
    NetworkResolvedExecutionPlan resolved_plan;
    CudaWeightBuffers weights;
    CudaExecutionWorkspace workspace;
    executor->Execute(plan, resolved_plan, weights, buffers, workspace,
                      batch_size);

    result.allocation_bytes = buffers.AllocationBytes();
    const auto downloaded = buffers.DownloadOutputs(batch_size);
    const float *decoded_moves_left =
        downloaded.moves_left.empty() ? nullptr : downloaded.moves_left.data();
    const auto decoded = DecodeNetworkOutputBatch(
        plan, downloaded.policy.data(), downloaded.policy.size(),
        downloaded.value.data(), downloaded.value.size(), decoded_moves_left,
        downloaded.moves_left.size(), batch_size);

    if (decoded.size() != static_cast<size_t>(batch_size)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA null executor decoded batch size mismatch";
      return result;
    }
    for (int b = 0; b < batch_size; ++b) {
      const auto &out = decoded[static_cast<size_t>(b)];
      const float expected_value = 0.60f - 0.05f * static_cast<float>(b);
      if (!out.has_wdl || !out.has_moves_left ||
          std::fabs(out.value - expected_value) > 1e-6f ||
          out.moves_left != 12.0f + static_cast<float>(b) ||
          out.policy[0] != 0.25f + static_cast<float>(b) ||
          out.policy[plan.policy_outputs - 1] !=
              -0.75f - static_cast<float>(b)) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA null executor decoded output mismatch";
        return result;
      }
      if (!downloaded.raw_policy.empty()) {
        const size_t raw_offset =
            static_cast<size_t>(b) * plan.raw_policy_outputs;
        if (downloaded.raw_policy[raw_offset] != 3.0f + static_cast<float>(b)) {
          result.status = CudaSmokeStatus::Mismatch;
          result.message = "CUDA null executor raw policy download mismatch";
          return result;
        }
      }
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

CudaBufferSmokeResult RunAttentionProjectionSmoke() {
  CudaBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 2;
  constexpr int kRows = kBatch * kCudaAttentionSquares;
  constexpr int kInput = 3;
  constexpr int kQkv = 4;
  constexpr int kHeads = 2;
  constexpr int kHeadDepth = kQkv / kHeads;
  constexpr int kOutput = 3;
  constexpr int kSmolgenCompressed = 2;
  constexpr int kSmolgenDense1 = 5;
  constexpr int kSmolgenPerHead = 3;
  constexpr int kSmolgenDense2 = kHeads * kSmolgenPerHead;
  constexpr int kSmolgenGlobal = kCudaAttentionSquares * kCudaAttentionSquares;
  std::vector<float> input(static_cast<std::size_t>(kRows) * kInput, 0.0f);
  for (std::size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.125f;

  const std::vector<float> q_weight = {
      0.25f, -0.50f, 0.75f,
      -0.10f, 0.30f, 0.20f,
      0.60f, -0.15f, -0.35f,
      0.40f, 0.10f, -0.20f,
  };
  const std::vector<float> k_weight = {
      -0.20f, 0.15f, 0.50f,
      0.35f, -0.25f, 0.45f,
      0.10f, 0.55f, -0.40f,
      -0.30f, 0.20f, 0.25f,
  };
  const std::vector<float> v_weight = {
      0.50f, 0.25f, -0.10f,
      -0.45f, 0.15f, 0.35f,
      0.20f, -0.60f, 0.30f,
      0.05f, 0.40f, -0.25f,
  };
  const std::vector<float> projection_weight = {
      0.30f, -0.20f, 0.10f, 0.45f,
      -0.35f, 0.50f, 0.25f, -0.15f,
      0.20f, 0.05f, -0.40f, 0.60f,
  };
  const std::vector<float> q_bias = {0.10f, -0.20f, 0.30f, -0.40f};
  const std::vector<float> k_bias = {-0.05f, 0.15f, -0.25f, 0.35f};
  const std::vector<float> v_bias = {0.20f, 0.00f, -0.10f, 0.05f};
  const std::vector<float> projection_bias = {0.25f, -0.15f, 0.05f};
  const std::vector<float> ln_gamma = {1.20f, -0.70f, 0.50f};
  const std::vector<float> ln_beta = {0.10f, -0.20f, 0.30f};
  const std::vector<float> smolgen_compress = {
      0.20f, -0.35f, 0.45f,
      -0.10f, 0.25f, 0.15f,
  };
  const std::vector<float> smolgen_dense1_weight = {
      0.15f, -0.25f, 0.35f, 0.05f, -0.10f, 0.20f,
      -0.30f, 0.40f, 0.10f, -0.15f, 0.25f, -0.05f,
      0.20f, 0.05f, -0.35f, 0.30f, -0.25f, 0.15f,
      -0.10f, 0.45f, -0.20f, 0.10f, 0.05f, -0.30f,
      0.35f, -0.05f, 0.25f, -0.40f, 0.15f, 0.20f,
  };
  const std::vector<float> smolgen_dense1_bias = {0.10f, -0.20f, 0.05f,
                                                  0.15f, -0.10f};
  const std::vector<float> smolgen_ln1_gamma = {1.00f, -0.50f, 0.75f,
                                                1.25f, 0.40f};
  const std::vector<float> smolgen_ln1_beta = {0.05f, -0.10f, 0.15f, 0.00f,
                                               -0.05f};
  const std::vector<float> smolgen_dense2_weight = {
      0.25f, -0.10f, 0.15f, 0.30f, -0.20f,
      -0.35f, 0.20f, 0.10f, -0.15f, 0.25f,
      0.05f, 0.30f, -0.25f, 0.10f, -0.05f,
      0.40f, -0.30f, 0.20f, -0.10f, 0.15f,
      -0.15f, 0.25f, 0.35f, -0.20f, 0.05f,
      0.10f, -0.05f, 0.30f, 0.20f, -0.25f,
  };
  const std::vector<float> smolgen_dense2_bias = {0.20f, -0.05f, 0.10f,
                                                  -0.15f, 0.05f, -0.10f};
  const std::vector<float> smolgen_ln2_gamma = {0.80f, -1.10f, 0.60f,
                                                1.30f, 0.50f, -0.70f};
  const std::vector<float> smolgen_ln2_beta = {-0.05f, 0.20f, -0.15f,
                                               0.10f, 0.05f, -0.25f};
  std::vector<float> smolgen_global(
      static_cast<std::size_t>(kSmolgenGlobal) * kSmolgenPerHead, 0.0f);
  for (std::size_t i = 0; i < smolgen_global.size(); ++i) {
    smolgen_global[i] =
        static_cast<float>(static_cast<int>(i % 13) - 6) * 0.0025f;
  }

  NetworkWeightInventory inventory;
  inventory.tensors = {
      {"body.encoder.0.mha.q_w", q_weight.data(), q_weight.size(),
       {kQkv, kInput}, NetworkWeightTensorKind::DenseWeight},
      {"body.encoder.0.mha.q_b", q_bias.data(), q_bias.size(), {kQkv},
       NetworkWeightTensorKind::DenseBias},
      {"body.encoder.0.mha.k_w", k_weight.data(), k_weight.size(),
       {kQkv, kInput}, NetworkWeightTensorKind::DenseWeight},
      {"body.encoder.0.mha.k_b", k_bias.data(), k_bias.size(), {kQkv},
       NetworkWeightTensorKind::DenseBias},
      {"body.encoder.0.mha.v_w", v_weight.data(), v_weight.size(),
       {kQkv, kInput}, NetworkWeightTensorKind::DenseWeight},
      {"body.encoder.0.mha.v_b", v_bias.data(), v_bias.size(), {kQkv},
       NetworkWeightTensorKind::DenseBias},
      {"body.encoder.0.mha.dense_w", projection_weight.data(),
       projection_weight.size(), {kOutput, kQkv},
       NetworkWeightTensorKind::DenseWeight},
      {"body.encoder.0.mha.dense_b", projection_bias.data(),
       projection_bias.size(), {kOutput}, NetworkWeightTensorKind::DenseBias},
      {"body.encoder.0.ln1_gammas", ln_gamma.data(), ln_gamma.size(),
       {kOutput}, NetworkWeightTensorKind::NormScale},
      {"body.encoder.0.ln1_betas", ln_beta.data(), ln_beta.size(),
       {kOutput}, NetworkWeightTensorKind::NormBias},
      {"body.smolgen_w", smolgen_global.data(), smolgen_global.size(),
       {kSmolgenGlobal, kSmolgenPerHead},
       NetworkWeightTensorKind::PositionalEncoding},
      {"body.encoder.0.mha.smolgen.compress", smolgen_compress.data(),
       smolgen_compress.size(), {kSmolgenCompressed, kInput},
       NetworkWeightTensorKind::DenseWeight},
      {"body.encoder.0.mha.smolgen.dense1_w",
       smolgen_dense1_weight.data(), smolgen_dense1_weight.size(),
       {kSmolgenDense1, kCudaAttentionSquares * kSmolgenCompressed},
       NetworkWeightTensorKind::DenseWeight},
      {"body.encoder.0.mha.smolgen.dense1_b", smolgen_dense1_bias.data(),
       smolgen_dense1_bias.size(), {kSmolgenDense1},
       NetworkWeightTensorKind::DenseBias},
      {"body.encoder.0.mha.smolgen.dense2_w",
       smolgen_dense2_weight.data(), smolgen_dense2_weight.size(),
       {kSmolgenDense2, kSmolgenDense1}, NetworkWeightTensorKind::DenseWeight},
      {"body.encoder.0.mha.smolgen.dense2_b", smolgen_dense2_bias.data(),
       smolgen_dense2_bias.size(), {kSmolgenDense2},
       NetworkWeightTensorKind::DenseBias},
      {"body.encoder.0.mha.smolgen.ln1_gammas",
       smolgen_ln1_gamma.data(), smolgen_ln1_gamma.size(), {kSmolgenDense1},
       NetworkWeightTensorKind::NormScale},
      {"body.encoder.0.mha.smolgen.ln1_betas", smolgen_ln1_beta.data(),
       smolgen_ln1_beta.size(), {kSmolgenDense1},
       NetworkWeightTensorKind::NormBias},
      {"body.encoder.0.mha.smolgen.ln2_gammas",
       smolgen_ln2_gamma.data(), smolgen_ln2_gamma.size(), {kSmolgenDense2},
       NetworkWeightTensorKind::NormScale},
      {"body.encoder.0.mha.smolgen.ln2_betas", smolgen_ln2_beta.data(),
       smolgen_ln2_beta.size(), {kSmolgenDense2},
       NetworkWeightTensorKind::NormBias},
  };

  NetworkResolvedExecutionPlan execution_plan;
  execution_plan.format.input_embedding = INPUT_EMBEDDING_PE_DENSE;
  execution_plan.format.activations.smolgen_activation = "swish";
  execution_plan.format.body_attention_heads = kHeads;
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::PositionalEncoding,
      "body.smolgen_positional",
      {
          {10, "body.smolgen_w", smolgen_global.size(),
           {kSmolgenGlobal, kSmolgenPerHead},
           NetworkWeightTensorKind::PositionalEncoding},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Attention,
      "body.encoder.0.mha",
      {
          {0, "body.encoder.0.mha.q_w", q_weight.size(), {kQkv, kInput},
           NetworkWeightTensorKind::DenseWeight},
          {1, "body.encoder.0.mha.q_b", q_bias.size(), {kQkv},
           NetworkWeightTensorKind::DenseBias},
          {2, "body.encoder.0.mha.k_w", k_weight.size(), {kQkv, kInput},
           NetworkWeightTensorKind::DenseWeight},
          {3, "body.encoder.0.mha.k_b", k_bias.size(), {kQkv},
           NetworkWeightTensorKind::DenseBias},
          {4, "body.encoder.0.mha.v_w", v_weight.size(), {kQkv, kInput},
           NetworkWeightTensorKind::DenseWeight},
          {5, "body.encoder.0.mha.v_b", v_bias.size(), {kQkv},
           NetworkWeightTensorKind::DenseBias},
          {6, "body.encoder.0.mha.dense_w", projection_weight.size(),
           {kOutput, kQkv}, NetworkWeightTensorKind::DenseWeight},
          {7, "body.encoder.0.mha.dense_b", projection_bias.size(),
           {kOutput}, NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "body.encoder.0.mha.smolgen.dense",
      {
          {11, "body.encoder.0.mha.smolgen.compress",
           smolgen_compress.size(), {kSmolgenCompressed, kInput},
           NetworkWeightTensorKind::DenseWeight},
          {12, "body.encoder.0.mha.smolgen.dense1_w",
           smolgen_dense1_weight.size(),
           {kSmolgenDense1, kCudaAttentionSquares * kSmolgenCompressed},
           NetworkWeightTensorKind::DenseWeight},
          {13, "body.encoder.0.mha.smolgen.dense1_b",
           smolgen_dense1_bias.size(), {kSmolgenDense1},
           NetworkWeightTensorKind::DenseBias},
          {14, "body.encoder.0.mha.smolgen.dense2_w",
           smolgen_dense2_weight.size(), {kSmolgenDense2, kSmolgenDense1},
           NetworkWeightTensorKind::DenseWeight},
          {15, "body.encoder.0.mha.smolgen.dense2_b",
           smolgen_dense2_bias.size(), {kSmolgenDense2},
           NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "body.encoder.0.mha.smolgen.norm",
      {
          {16, "body.encoder.0.mha.smolgen.ln1_gammas",
           smolgen_ln1_gamma.size(), {kSmolgenDense1},
           NetworkWeightTensorKind::NormScale},
          {17, "body.encoder.0.mha.smolgen.ln1_betas",
           smolgen_ln1_beta.size(), {kSmolgenDense1},
           NetworkWeightTensorKind::NormBias},
          {18, "body.encoder.0.mha.smolgen.ln2_gammas",
           smolgen_ln2_gamma.size(), {kSmolgenDense2},
           NetworkWeightTensorKind::NormScale},
          {19, "body.encoder.0.mha.smolgen.ln2_betas",
           smolgen_ln2_beta.size(), {kSmolgenDense2},
           NetworkWeightTensorKind::NormBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "body.encoder.0.ln1",
      {
          {8, "body.encoder.0.ln1_gammas", ln_gamma.size(), {kOutput},
           NetworkWeightTensorKind::NormScale},
          {9, "body.encoder.0.ln1_betas", ln_beta.size(), {kOutput},
           NetworkWeightTensorKind::NormBias},
      }});

  const auto expected_q =
      DenseAffineHost(input, q_weight, q_bias, kRows, kInput, kQkv);
  const auto expected_k =
      DenseAffineHost(input, k_weight, k_bias, kRows, kInput, kQkv);
  const auto expected_v =
      DenseAffineHost(input, v_weight, v_bias, kRows, kInput, kQkv);
  const float attention_scale =
      1.0f / std::sqrt(static_cast<float>(kHeadDepth));
  auto expected_scores = AttentionScoresHost(
      expected_q, expected_k, kBatch, kHeads, kCudaAttentionSquares,
      kHeadDepth, kQkv, attention_scale);
  const std::vector<float> zero_compress_bias(kSmolgenCompressed, 0.0f);
  const auto expected_smolgen_compress =
      DenseAffineHost(input, smolgen_compress, zero_compress_bias, kRows,
                      kInput, kSmolgenCompressed);
  const auto expected_smolgen_dense1 = DenseAffineHost(
      expected_smolgen_compress, smolgen_dense1_weight, smolgen_dense1_bias,
      kBatch, kCudaAttentionSquares * kSmolgenCompressed, kSmolgenDense1);
  const auto expected_smolgen_activation1 =
      ActivationHost(expected_smolgen_dense1, CudaActivationKind::Swish);
  const auto expected_smolgen_norm1 = LayerNormRowsHost(
      expected_smolgen_activation1, smolgen_ln1_gamma, smolgen_ln1_beta,
      kBatch, kSmolgenDense1, 1e-3f);
  const auto expected_smolgen_dense2 = DenseAffineHost(
      expected_smolgen_norm1, smolgen_dense2_weight, smolgen_dense2_bias,
      kBatch, kSmolgenDense1, kSmolgenDense2);
  const auto expected_smolgen_activation2 =
      ActivationHost(expected_smolgen_dense2, CudaActivationKind::Swish);
  const auto expected_smolgen_norm2 = LayerNormRowsHost(
      expected_smolgen_activation2, smolgen_ln2_gamma, smolgen_ln2_beta,
      kBatch, kSmolgenDense2, 1e-3f);
  const std::vector<float> zero_global_bias(kSmolgenGlobal, 0.0f);
  const auto expected_smolgen_global = DenseAffineHost(
      expected_smolgen_norm2, smolgen_global, zero_global_bias, kBatch * kHeads,
      kSmolgenPerHead, kSmolgenGlobal);
  for (int batch = 0; batch < kBatch; ++batch) {
    for (int head = 0; head < kHeads; ++head) {
      const std::size_t row =
          static_cast<std::size_t>(batch * kHeads + head) * kSmolgenGlobal;
      for (int index = 0; index < kSmolgenGlobal; ++index)
        expected_scores[row + index] += expected_smolgen_global[row + index];
    }
  }
  const auto expected_probabilities = SoftmaxRowsHost(
      expected_scores, kBatch * kHeads * kCudaAttentionSquares,
      kCudaAttentionSquares);
  const auto expected_context = AttentionContextHost(
      expected_probabilities, expected_v, kBatch, kHeads, kCudaAttentionSquares,
      kHeadDepth, kQkv);
  const auto expected_projection = DenseAffineHost(
      expected_context, projection_weight, projection_bias, kRows, kQkv,
      kOutput);
  std::vector<float> expected_residual(expected_projection.size(), 0.0f);
  for (std::size_t i = 0; i < expected_residual.size(); ++i)
    expected_residual[i] = input[i] + expected_projection[i];
  const auto expected_normalized = LayerNormRowsHost(
      expected_residual, ln_gamma, ln_beta, kRows, kOutput, 1e-3f);

  try {
    CudaExecutionWorkspace workspace;
    cudaStream_t stream = workspace.Stream();
    float *device_input =
        workspace.ReserveNamedFloats("attention.input", input.size());
    UploadFloats(device_input, input, stream, "cudaMemcpy(attention_input)");

    CudaWeightBuffers weights;
    weights.Upload(inventory);
    const auto tape = CreateResolvedExecutionTape(execution_plan, kBatch);
    const auto input_output = ExecuteAttentionInputProjectionStage(
        execution_plan, 1, weights, device_input, tape, workspace, kBatch);
    const auto core_output = ExecuteAttentionCoreStage(
        execution_plan, 1, input_output, tape, workspace, kBatch, &weights,
        device_input);
    const auto projection_output = ExecuteAttentionOutputProjectionStage(
        execution_plan, 1, weights, core_output.context, tape, workspace,
        kBatch);
    const auto norm_output = ExecuteAttentionResidualLayerNormStage(
        execution_plan, execution_plan.steps[4], device_input,
        projection_output, weights, tape, workspace, kBatch);
    workspace.Synchronize();

    if (input_output.rows != kRows || input_output.qkv_width != kQkv ||
        input_output.heads != kHeads || input_output.head_depth != kHeadDepth ||
        core_output.score_rows != kBatch * kHeads * kCudaAttentionSquares ||
        core_output.score_width != kCudaAttentionSquares ||
        core_output.rows != kRows || core_output.qkv_width != kQkv ||
        projection_output.output_width != kOutput ||
        norm_output.output_width != kOutput) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention metadata mismatch";
      return result;
    }

    std::vector<float> actual_q(expected_q.size(), 0.0f);
    std::vector<float> actual_k(expected_k.size(), 0.0f);
    std::vector<float> actual_v(expected_v.size(), 0.0f);
    std::vector<float> actual_scores(expected_scores.size(), 0.0f);
    std::vector<float> actual_probabilities(expected_probabilities.size(),
                                            0.0f);
    std::vector<float> actual_context(expected_context.size(), 0.0f);
    std::vector<float> actual_projection(expected_projection.size(), 0.0f);
    std::vector<float> actual_residual(expected_residual.size(), 0.0f);
    std::vector<float> actual_normalized(expected_normalized.size(), 0.0f);
    DownloadFloats(actual_q, input_output.query,
                   "cudaMemcpy(attention_q)");
    DownloadFloats(actual_k, input_output.key, "cudaMemcpy(attention_k)");
    DownloadFloats(actual_v, input_output.value,
                   "cudaMemcpy(attention_v)");
    DownloadFloats(actual_scores, core_output.scores,
                   "cudaMemcpy(attention_scores)");
    DownloadFloats(actual_probabilities, core_output.probabilities,
                   "cudaMemcpy(attention_probabilities)");
    DownloadFloats(actual_context, core_output.context,
                   "cudaMemcpy(attention_context)");
    DownloadFloats(actual_projection, projection_output.projection,
                   "cudaMemcpy(attention_projection)");
    DownloadFloats(actual_residual, norm_output.residual,
                   "cudaMemcpy(attention_residual)");
    DownloadFloats(actual_normalized, norm_output.normalized,
                   "cudaMemcpy(attention_normalized)");
    if (!AlmostEqual(actual_q, expected_q, 1e-5f) ||
        !AlmostEqual(actual_k, expected_k, 1e-5f) ||
        !AlmostEqual(actual_v, expected_v, 1e-5f) ||
        !AlmostEqual(actual_scores, expected_scores, 1e-5f) ||
        !AlmostEqual(actual_probabilities, expected_probabilities, 1e-5f) ||
        !AlmostEqual(actual_context, expected_context, 1e-5f) ||
        !AlmostEqual(actual_projection, expected_projection, 1e-5f) ||
        !AlmostEqual(actual_residual, expected_residual, 1e-5f) ||
        !AlmostEqual(actual_normalized, expected_normalized, 1e-5f)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention output mismatch";
      return result;
    }

    CudaExecutionWorkspace sequence_workspace;
    float *sequence_input = sequence_workspace.ReserveNamedFloats(
        "attention.sequence.input", input.size());
    UploadFloats(sequence_input, input, sequence_workspace.Stream(),
                 "cudaMemcpy(attention_sequence_input)");
    const auto sequence = ExecuteDenseActivationLayerNormSequence(
        execution_plan, weights, sequence_input, tape, sequence_workspace,
        kBatch);
    sequence_workspace.Synchronize();
    const auto *sequence_stage = sequence.FindStage("body.encoder.0.mha");
    if (!sequence_stage || !sequence_stage->output ||
        sequence_stage->output_width != kOutput) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention sequence metadata mismatch";
      return result;
    }
    std::vector<float> actual_sequence(expected_normalized.size(), 0.0f);
    DownloadFloats(actual_sequence, sequence_stage->output,
                   "cudaMemcpy(attention_sequence_output)");
    if (!AlmostEqual(actual_sequence, expected_normalized, 1e-5f)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention sequence output mismatch";
      return result;
    }
    result.allocation_bytes = workspace.TotalBytes() + weights.AllocationBytes();
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaBufferSmokeResult RunDynamicPositionEncodingStageSmoke() {
  CudaBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  constexpr int kBatch = 1;
  constexpr int kPlanes = kPackedInputPlaneCount;
  constexpr int kSquares = kPackedInputSquareCount;
  constexpr int kPositionPlanes = 12;
  constexpr int kPositionWidth = 3;
  constexpr int kPositionInput = kSquares * kPositionPlanes;
  constexpr int kPositionOutput = kSquares * kPositionWidth;
  constexpr int kConcatWidth = kPlanes + kPositionWidth;
  constexpr int kEmbeddingWidth = 4;
  constexpr int kFfnHidden = 5;

  std::vector<std::uint64_t> masks(kBatch * kPlanes, 0);
  std::vector<float> values(kBatch * kPlanes, 0.0f);
  auto set_plane = [&](int plane, std::uint64_t mask, float value) {
    masks[plane] = mask;
    values[plane] = value;
  };
  set_plane(0, (1ULL << 0) | (1ULL << 5), 2.0f);
  set_plane(11, (1ULL << 0) | (1ULL << 63), -1.5f);
  set_plane(12, ~0ULL, 7.0f);

  std::vector<float> preproc_weights(
      static_cast<std::size_t>(kPositionOutput) * kPositionInput, 0.0f);
  std::vector<float> preproc_bias(kPositionOutput, 0.0f);
  for (int square = 0; square < kSquares; ++square) {
    const int output_offset = square * kPositionWidth;
    const int input_offset = square * kPositionPlanes;
    preproc_weights[static_cast<std::size_t>(output_offset) *
                        kPositionInput +
                    input_offset] = 0.5f;
    preproc_weights[static_cast<std::size_t>(output_offset + 1) *
                        kPositionInput +
                    input_offset + 11] = 2.0f;
    preproc_bias[output_offset + 2] =
        1.0f + static_cast<float>(square) * 0.01f;
  }
  std::vector<float> embedding_weights(
      static_cast<std::size_t>(kEmbeddingWidth) * kConcatWidth, 0.0f);
  std::vector<float> embedding_bias = {0.25f, -0.5f, 0.75f, 0.0f};
  embedding_weights[0 * kConcatWidth + 0] = 0.5f;
  embedding_weights[0 * kConcatWidth + kPlanes] = 1.0f;
  embedding_weights[1 * kConcatWidth + 11] = -1.0f;
  embedding_weights[1 * kConcatWidth + kPlanes + 1] = 0.25f;
  embedding_weights[2 * kConcatWidth + 12] = 0.1f;
  embedding_weights[2 * kConcatWidth + kPlanes + 2] = 0.5f;
  embedding_weights[3 * kConcatWidth + 0] = -0.25f;
  embedding_weights[3 * kConcatWidth + 12] = 0.2f;
  const std::vector<float> embedding_gamma = {1.10f, -0.75f, 0.50f, 1.25f};
  const std::vector<float> embedding_beta = {0.20f, -0.10f, 0.35f, -0.40f};
  const std::vector<float> mult_gate = {1.50f, -0.25f, 0.75f, 1.10f};
  const std::vector<float> add_gate = {-0.20f, 0.45f, 0.10f, -0.30f};
  const std::vector<float> ffn1_weights = {
      0.30f, -0.20f, 0.50f, 0.10f,
      -0.40f, 0.25f, 0.15f, -0.35f,
      0.60f, -0.10f, 0.20f, 0.45f,
      0.05f, 0.70f, -0.30f, 0.20f,
      -0.25f, 0.15f, 0.35f, -0.50f,
  };
  const std::vector<float> ffn1_bias = {0.10f, -0.05f, 0.20f, -0.15f,
                                        0.30f};
  const std::vector<float> ffn2_weights = {
      0.40f, -0.10f, 0.25f, 0.35f, -0.20f,
      -0.30f, 0.55f, -0.15f, 0.10f, 0.45f,
      0.15f, 0.20f, -0.50f, 0.60f, 0.05f,
      -0.45f, 0.30f, 0.10f, -0.25f, 0.50f,
  };
  const std::vector<float> ffn2_bias = {0.05f, -0.20f, 0.15f, 0.25f};
  const std::vector<float> ffn_gamma = {0.90f, 1.20f, -0.80f, 0.65f};
  const std::vector<float> ffn_beta = {-0.05f, 0.25f, 0.10f, -0.35f};

  NetworkTensorPlan tensor_plan;
  tensor_plan.input_planes = kPlanes;
  tensor_plan.input_squares = kSquares;

  NetworkResolvedExecutionPlan execution_plan;
  execution_plan.format.input_embedding = INPUT_EMBEDDING_PE_DENSE;
  execution_plan.tensors = tensor_plan;
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "body.input_embedding_preprocess",
      {
          {0, "body.ip_emb_preproc_w", preproc_weights.size(),
           {kPositionOutput, kPositionInput},
           NetworkWeightTensorKind::DenseWeight},
          {1, "body.ip_emb_preproc_b", preproc_bias.size(),
           {kPositionOutput}, NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "body.input_embedding",
      {
          {2, "body.input_embedding_w", embedding_weights.size(),
           {kEmbeddingWidth, kConcatWidth},
           NetworkWeightTensorKind::DenseWeight},
          {3, "body.input_embedding_b", embedding_bias.size(),
           {kEmbeddingWidth}, NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "body.input_embedding_norm",
      {
          {4, "body.ip_emb_ln_gammas", embedding_gamma.size(),
           {kEmbeddingWidth}, NetworkWeightTensorKind::NormScale},
          {5, "body.ip_emb_ln_betas", embedding_beta.size(),
           {kEmbeddingWidth}, NetworkWeightTensorKind::NormBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Gate,
      "body.input_embedding_gates",
      {
          {6, "body.ip_mult_gate", mult_gate.size(), {kEmbeddingWidth},
           NetworkWeightTensorKind::Gate},
          {7, "body.ip_add_gate", add_gate.size(), {kEmbeddingWidth},
           NetworkWeightTensorKind::Gate},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::FeedForward,
      "body.input_embedding_ffn",
      {
          {8, "body.ip_emb_ffn.dense1_w", ffn1_weights.size(),
           {kFfnHidden, kEmbeddingWidth}, NetworkWeightTensorKind::DenseWeight},
          {9, "body.ip_emb_ffn.dense1_b", ffn1_bias.size(), {kFfnHidden},
           NetworkWeightTensorKind::DenseBias},
          {10, "body.ip_emb_ffn.dense2_w", ffn2_weights.size(),
           {kEmbeddingWidth, kFfnHidden}, NetworkWeightTensorKind::DenseWeight},
          {11, "body.ip_emb_ffn.dense2_b", ffn2_bias.size(),
           {kEmbeddingWidth}, NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "body.input_embedding_ffn_norm",
      {
          {12, "body.ip_emb_ffn_ln_gammas", ffn_gamma.size(),
           {kEmbeddingWidth}, NetworkWeightTensorKind::NormScale},
          {13, "body.ip_emb_ffn_ln_betas", ffn_beta.size(),
           {kEmbeddingWidth}, NetworkWeightTensorKind::NormBias},
      }});

  NetworkWeightInventory inventory;
  inventory.tensors = {
      {"body.ip_emb_preproc_w", preproc_weights.data(),
       preproc_weights.size(), {kPositionOutput, kPositionInput},
       NetworkWeightTensorKind::DenseWeight},
      {"body.ip_emb_preproc_b", preproc_bias.data(), preproc_bias.size(),
       {kPositionOutput}, NetworkWeightTensorKind::DenseBias},
      {"body.input_embedding_w", embedding_weights.data(),
       embedding_weights.size(), {kEmbeddingWidth, kConcatWidth},
       NetworkWeightTensorKind::DenseWeight},
      {"body.input_embedding_b", embedding_bias.data(),
       embedding_bias.size(), {kEmbeddingWidth},
       NetworkWeightTensorKind::DenseBias},
      {"body.ip_emb_ln_gammas", embedding_gamma.data(),
       embedding_gamma.size(), {kEmbeddingWidth},
       NetworkWeightTensorKind::NormScale},
      {"body.ip_emb_ln_betas", embedding_beta.data(),
       embedding_beta.size(), {kEmbeddingWidth},
       NetworkWeightTensorKind::NormBias},
      {"body.ip_mult_gate", mult_gate.data(), mult_gate.size(),
       {kEmbeddingWidth}, NetworkWeightTensorKind::Gate},
      {"body.ip_add_gate", add_gate.data(), add_gate.size(),
       {kEmbeddingWidth}, NetworkWeightTensorKind::Gate},
      {"body.ip_emb_ffn.dense1_w", ffn1_weights.data(),
       ffn1_weights.size(), {kFfnHidden, kEmbeddingWidth},
       NetworkWeightTensorKind::DenseWeight},
      {"body.ip_emb_ffn.dense1_b", ffn1_bias.data(), ffn1_bias.size(),
       {kFfnHidden}, NetworkWeightTensorKind::DenseBias},
      {"body.ip_emb_ffn.dense2_w", ffn2_weights.data(),
       ffn2_weights.size(), {kEmbeddingWidth, kFfnHidden},
       NetworkWeightTensorKind::DenseWeight},
      {"body.ip_emb_ffn.dense2_b", ffn2_bias.data(), ffn2_bias.size(),
       {kEmbeddingWidth}, NetworkWeightTensorKind::DenseBias},
      {"body.ip_emb_ffn_ln_gammas", ffn_gamma.data(), ffn_gamma.size(),
       {kEmbeddingWidth}, NetworkWeightTensorKind::NormScale},
      {"body.ip_emb_ffn_ln_betas", ffn_beta.data(), ffn_beta.size(),
       {kEmbeddingWidth}, NetworkWeightTensorKind::NormBias},
  };

  std::vector<float> expected(kSquares * kConcatWidth, 0.0f);
  for (int square = 0; square < kSquares; ++square) {
    for (int plane = 0; plane < kPlanes; ++plane) {
      const float expanded =
          (masks[plane] & (1ULL << square)) ? values[plane] : 0.0f;
      expected[static_cast<std::size_t>(square) * kConcatWidth + plane] =
          expanded;
    }
    const float plane0 = (masks[0] & (1ULL << square)) ? values[0] : 0.0f;
    const float plane11 = (masks[11] & (1ULL << square)) ? values[11] : 0.0f;
    const std::size_t pe_offset =
        static_cast<std::size_t>(square) * kConcatWidth + kPlanes;
    expected[pe_offset + 0] = plane0 * 0.5f;
    expected[pe_offset + 1] = plane11 * 2.0f;
    expected[pe_offset + 2] = 1.0f + static_cast<float>(square) * 0.01f;
  }

  try {
    CudaInferenceBuffers buffers;
    buffers.Allocate(LayoutFromTensorPlan(tensor_plan, kBatch));
    CudaExecutionWorkspace workspace;
    buffers.UploadPackedInputs(masks, values, kBatch, workspace.Stream());

    CudaWeightBuffers weights;
    weights.Upload(inventory);
    const auto tape = CreateResolvedExecutionTape(execution_plan, kBatch);
    const auto output = ExecuteDynamicPositionEncodingStage(
        execution_plan, execution_plan.steps.front(), weights,
        buffers.input_masks, buffers.input_values, tape, workspace, kBatch);
    workspace.Synchronize();

    if (!output.output || output.rows != kSquares ||
        output.output_width != kConcatWidth) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic PE stage metadata mismatch";
      return result;
    }

    std::vector<float> actual(expected.size(), 0.0f);
    DownloadFloats(actual, output.output, "cudaMemcpy(dynamic_stage_output)");
    if (!AlmostEqual(actual, expected, 1e-5f)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic PE stage output mismatch";
      return result;
    }

    const auto expected_embedding_dense = DenseAffineHost(
        expected, embedding_weights, embedding_bias, kSquares, kConcatWidth,
        kEmbeddingWidth);
    const auto expected_embedding =
        ActivationHost(expected_embedding_dense, CudaActivationKind::Relu);
    const auto expected_embedding_norm =
        LayerNormRowsHost(expected_embedding, embedding_gamma, embedding_beta,
                          kSquares, kEmbeddingWidth, 1e-3f);
    std::vector<float> expected_gated(expected_embedding_norm.size(), 0.0f);
    for (int square = 0; square < kSquares; ++square) {
      const std::size_t offset =
          static_cast<std::size_t>(square) * kEmbeddingWidth;
      for (int channel = 0; channel < kEmbeddingWidth; ++channel) {
        expected_gated[offset + channel] =
            expected_embedding_norm[offset + channel] *
                mult_gate[static_cast<std::size_t>(channel)] +
            add_gate[static_cast<std::size_t>(channel)];
      }
    }
    const auto expected_ffn_dense1 =
        DenseAffineHost(expected_gated, ffn1_weights, ffn1_bias, kSquares,
                        kEmbeddingWidth, kFfnHidden);
    const auto expected_ffn_activation =
        ActivationHost(expected_ffn_dense1, CudaActivationKind::Relu);
    const auto expected_ffn_dense2 =
        DenseAffineHost(expected_ffn_activation, ffn2_weights, ffn2_bias,
                        kSquares, kFfnHidden, kEmbeddingWidth);
    std::vector<float> expected_ffn_residual(expected_ffn_dense2.size(),
                                             0.0f);
    for (std::size_t i = 0; i < expected_ffn_residual.size(); ++i)
      expected_ffn_residual[i] = expected_gated[i] + expected_ffn_dense2[i];
    const auto expected_ffn_norm =
        LayerNormRowsHost(expected_ffn_residual, ffn_gamma, ffn_beta, kSquares,
                          kEmbeddingWidth, 1e-3f);

    CudaExecutionWorkspace sequence_workspace;
    const CudaStageInputBindings input_bindings;
    const auto sequence = ExecuteDenseActivationLayerNormSequence(
        execution_plan, weights, buffers.input_values, buffers.input_masks,
        buffers.input_values, tape, sequence_workspace, kBatch,
        input_bindings);
    sequence_workspace.Synchronize();
    const auto *preprocess_stage =
        sequence.FindStage("body.input_embedding_preprocess");
    const auto *embedding_stage = sequence.FindStage("body.input_embedding");
    const auto *gate_stage = sequence.FindStage("body.input_embedding_gates");
    const auto *ffn_stage = sequence.FindStage("body.input_embedding_ffn");
    if (sequence.stage_count != 4 || !preprocess_stage ||
        !preprocess_stage->output ||
        preprocess_stage->output_width != kConcatWidth ||
        !embedding_stage || !embedding_stage->activation ||
        !embedding_stage->output ||
        embedding_stage->rows != kSquares ||
        embedding_stage->output_width != kEmbeddingWidth || !gate_stage ||
        !gate_stage->output || gate_stage->rows != kSquares ||
        gate_stage->output_width != kEmbeddingWidth || !ffn_stage ||
        !ffn_stage->feed_forward || !ffn_stage->residual ||
        !ffn_stage->output || ffn_stage->rows != kSquares ||
        ffn_stage->output_width != kEmbeddingWidth) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic PE sequence metadata mismatch";
      return result;
    }
    std::vector<float> actual_sequence(expected.size(), 0.0f);
    std::vector<float> actual_embedding(expected_embedding.size(), 0.0f);
    std::vector<float> actual_embedding_norm(expected_embedding_norm.size(),
                                             0.0f);
    std::vector<float> actual_gated(expected_gated.size(), 0.0f);
    std::vector<float> actual_ffn_residual(expected_ffn_residual.size(), 0.0f);
    std::vector<float> actual_ffn_norm(expected_ffn_norm.size(), 0.0f);
    DownloadFloats(actual_sequence, preprocess_stage->output,
                   "cudaMemcpy(dynamic_sequence_preprocess)");
    DownloadFloats(actual_embedding, embedding_stage->activation,
                   "cudaMemcpy(dynamic_sequence_embedding)");
    DownloadFloats(actual_embedding_norm, embedding_stage->output,
                   "cudaMemcpy(dynamic_sequence_embedding_norm)");
    DownloadFloats(actual_gated, gate_stage->output,
                   "cudaMemcpy(dynamic_sequence_gate)");
    DownloadFloats(actual_ffn_residual, ffn_stage->residual,
                   "cudaMemcpy(dynamic_sequence_ffn_residual)");
    DownloadFloats(actual_ffn_norm, ffn_stage->output,
                   "cudaMemcpy(dynamic_sequence_ffn_norm)");
    if (!AlmostEqual(actual_sequence, expected, 1e-5f) ||
        !AlmostEqual(actual_embedding, expected_embedding, 1e-5f) ||
        !AlmostEqual(actual_embedding_norm, expected_embedding_norm, 1e-5f) ||
        !AlmostEqual(actual_gated, expected_gated, 1e-5f) ||
        !AlmostEqual(actual_ffn_residual, expected_ffn_residual, 1e-5f) ||
        !AlmostEqual(actual_ffn_norm, expected_ffn_norm, 1e-5f)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA dynamic PE sequence output mismatch";
      return result;
    }

    result.allocation_bytes =
        buffers.AllocationBytes() + workspace.TotalBytes() +
        sequence_workspace.TotalBytes() +
        weights.AllocationBytes();
  } catch (const std::exception &e) {
    result.status = CudaSmokeStatus::RuntimeError;
    result.message = e.what();
    return result;
  }

  result.status = CudaSmokeStatus::Success;
  result.message = RuntimeCudaDeviceSummary();
  return result;
}

CudaBufferSmokeResult RunPlanExecutorPipelineSmoke() {
  CudaBufferSmokeResult result;

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    result.status = CudaSmokeStatus::NoDevice;
    result.message = RuntimeCudaDeviceSummary();
    return result;
  }

  NetworkTensorPlan plan;
  plan.input_planes = 3;
  plan.policy_outputs = kNetworkPolicyOutputs;
  plan.value_outputs = 3;
  plan.moves_left_outputs = 1;
  plan.wdl = true;
  plan.moves_left = true;
  plan.raw_policy_outputs = kNetworkAttentionPolicyScratch;
  NetworkFormatDescriptor format;
  format.wdl = true;
  format.moves_left = true;
  format.attention_policy = true;
  format.activations.ffn_activation = "relu_2";

  constexpr int kBatch = 3;
  constexpr int kInput = 3;
  constexpr int kHidden = 4;
  constexpr int kFfnHidden = 5;
  constexpr int kPolicy = 2;
  constexpr int kValue = 3;
  constexpr int kMovesLeft = 1;
  const std::vector<float> input_values = {
      1.0f, -2.0f, 0.5f,
      -0.5f, 2.0f, 1.5f,
      0.25f, -0.75f, 3.0f,
  };
  const std::vector<std::uint64_t> input_masks(static_cast<size_t>(kBatch) *
                                                   kInput,
                                               ~0ULL);
  const std::vector<float> dense_weights = {
      1.0f, 0.0f, 0.5f,
      -0.5f, 1.0f, 0.25f,
      0.0f, -1.0f, 0.5f,
      2.0f, 0.5f, -1.0f,
  };
  const std::vector<float> dense_bias = {0.25f, -0.75f, 0.0f, 1.0f};
  const std::vector<float> gamma = {1.0f, 0.5f, -1.0f, 2.0f};
  const std::vector<float> beta = {0.0f, -0.25f, 0.5f, 1.0f};
  const std::vector<float> mult_gate = {1.5f, -0.5f, 2.0f, 0.25f};
  const std::vector<float> add_gate = {0.25f, 1.0f, -0.75f, 0.5f};
  const std::vector<float> ffn1_weights = {
      0.20f, -0.10f, 0.50f, 0.75f,
      -0.40f, 0.30f, 0.10f, -0.20f,
      0.60f, 0.25f, -0.35f, 0.15f,
      -0.10f, 0.90f, 0.05f, -0.45f,
      0.30f, -0.60f, 0.20f, 0.40f,
  };
  const std::vector<float> ffn1_bias = {0.10f, -0.20f, 0.05f, 0.30f,
                                        -0.15f};
  const std::vector<float> ffn2_weights = {
      0.50f, -0.25f, 0.10f, 0.20f, -0.40f,
      -0.30f, 0.60f, 0.15f, -0.10f, 0.25f,
      0.20f, 0.35f, -0.45f, 0.50f, 0.10f,
      -0.15f, 0.05f, 0.30f, -0.35f, 0.70f,
  };
  const std::vector<float> ffn2_bias = {0.05f, -0.10f, 0.20f, -0.25f};
  const std::vector<float> ffn_gamma = {1.10f, -0.60f, 0.80f, 1.40f};
  const std::vector<float> ffn_beta = {0.20f, 0.05f, -0.30f, 0.45f};
  const std::vector<float> positional(static_cast<size_t>(kHidden) * kHidden,
                                      0.0f);
  const std::vector<float> dense2_weights = {
      0.25f, -0.5f, 1.0f, 0.75f,
      -1.0f, 0.5f, 0.25f, -0.25f,
  };
  const std::vector<float> dense2_bias = {0.1f, -0.2f};
  const std::vector<float> gamma2 = {1.25f, -0.75f};
  const std::vector<float> beta2 = {0.05f, 0.30f};
  const std::vector<float> dense3_weights = {
      0.5f, -0.25f, 0.75f, 0.10f,
      -1.0f, 0.75f, 0.20f, -0.30f,
      0.25f, 1.5f, -0.50f, 0.40f,
  };
  const std::vector<float> dense3_bias = {-0.1f, 0.2f, 0.05f};
  const std::vector<float> moves_weights = {0.5f, -0.25f, 1.25f, 0.75f};
  const std::vector<float> moves_bias = {0.4f};

  NetworkWeightInventory inventory;
  inventory.tensors = {
      {"smoke.dense_w", dense_weights.data(), dense_weights.size(),
       {kHidden, kInput}, NetworkWeightTensorKind::DenseWeight},
      {"smoke.dense_b", dense_bias.data(), dense_bias.size(), {kHidden},
       NetworkWeightTensorKind::DenseBias},
      {"smoke.norm_gammas", gamma.data(), gamma.size(), {kHidden},
       NetworkWeightTensorKind::NormScale},
      {"smoke.norm_betas", beta.data(), beta.size(), {kHidden},
       NetworkWeightTensorKind::NormBias},
      {"policy.smoke.ip_pol_w", dense2_weights.data(), dense2_weights.size(),
       {kPolicy, kHidden}, NetworkWeightTensorKind::DenseWeight},
      {"policy.smoke.ip_pol_b", dense2_bias.data(), dense2_bias.size(), {kPolicy},
       NetworkWeightTensorKind::DenseBias},
      {"policy.smoke.norm_gammas", gamma2.data(), gamma2.size(), {kPolicy},
       NetworkWeightTensorKind::NormScale},
      {"policy.smoke.norm_betas", beta2.data(), beta2.size(), {kPolicy},
       NetworkWeightTensorKind::NormBias},
      {"value.smoke.ip2_val_w", dense3_weights.data(), dense3_weights.size(),
       {kValue, kHidden}, NetworkWeightTensorKind::DenseWeight},
      {"value.smoke.ip2_val_b", dense3_bias.data(), dense3_bias.size(), {kValue},
       NetworkWeightTensorKind::DenseBias},
      {"moves_left.ip2_mov_w", moves_weights.data(), moves_weights.size(),
       {kMovesLeft, kHidden}, NetworkWeightTensorKind::DenseWeight},
      {"moves_left.ip2_mov_b", moves_bias.data(), moves_bias.size(), {kMovesLeft},
       NetworkWeightTensorKind::DenseBias},
      {"body.ip_mult_gate", mult_gate.data(), mult_gate.size(), {kHidden},
       NetworkWeightTensorKind::Gate},
      {"body.ip_add_gate", add_gate.data(), add_gate.size(), {kHidden},
       NetworkWeightTensorKind::Gate},
      {"body.ip_emb_ffn.dense1_w", ffn1_weights.data(), ffn1_weights.size(),
       {kFfnHidden, kHidden}, NetworkWeightTensorKind::DenseWeight},
      {"body.ip_emb_ffn.dense1_b", ffn1_bias.data(), ffn1_bias.size(),
       {kFfnHidden}, NetworkWeightTensorKind::DenseBias},
      {"body.ip_emb_ffn.dense2_w", ffn2_weights.data(), ffn2_weights.size(),
       {kHidden, kFfnHidden}, NetworkWeightTensorKind::DenseWeight},
      {"body.ip_emb_ffn.dense2_b", ffn2_bias.data(), ffn2_bias.size(),
       {kHidden}, NetworkWeightTensorKind::DenseBias},
      {"body.ip_emb_ffn_ln_gammas", ffn_gamma.data(), ffn_gamma.size(),
       {kHidden}, NetworkWeightTensorKind::NormScale},
      {"body.ip_emb_ffn_ln_betas", ffn_beta.data(), ffn_beta.size(),
       {kHidden}, NetworkWeightTensorKind::NormBias},
      {"body.smolgen_w", positional.data(), positional.size(),
       {kHidden, kHidden}, NetworkWeightTensorKind::PositionalEncoding},
  };

  NetworkResolvedExecutionPlan execution_plan;
  execution_plan.format = format;
  execution_plan.tensors = plan;
  execution_plan.policy_head = "smoke";
  execution_plan.value_head = "smoke";
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "body.smoke.dense",
      {
          {0, "smoke.dense_w", dense_weights.size(), {kHidden, kInput},
           NetworkWeightTensorKind::DenseWeight},
          {1, "smoke.dense_b", dense_bias.size(), {kHidden},
           NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "body.smoke.norm",
      {
          {2, "smoke.norm_gammas", gamma.size(), {kHidden},
           NetworkWeightTensorKind::NormScale},
          {3, "smoke.norm_betas", beta.size(), {kHidden},
           NetworkWeightTensorKind::NormBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Gate,
      "body.smoke.gates",
      {
          {12, "body.ip_mult_gate", mult_gate.size(), {kHidden},
           NetworkWeightTensorKind::Gate},
          {13, "body.ip_add_gate", add_gate.size(), {kHidden},
           NetworkWeightTensorKind::Gate},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::FeedForward,
      "body.input_embedding_ffn",
      {
          {14, "body.ip_emb_ffn.dense1_w", ffn1_weights.size(),
           {kFfnHidden, kHidden}, NetworkWeightTensorKind::DenseWeight},
          {15, "body.ip_emb_ffn.dense1_b", ffn1_bias.size(), {kFfnHidden},
           NetworkWeightTensorKind::DenseBias},
          {16, "body.ip_emb_ffn.dense2_w", ffn2_weights.size(),
           {kHidden, kFfnHidden}, NetworkWeightTensorKind::DenseWeight},
          {17, "body.ip_emb_ffn.dense2_b", ffn2_bias.size(), {kHidden},
           NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "body.input_embedding_ffn_norm",
      {
          {18, "body.ip_emb_ffn_ln_gammas", ffn_gamma.size(), {kHidden},
           NetworkWeightTensorKind::NormScale},
          {19, "body.ip_emb_ffn_ln_betas", ffn_beta.size(), {kHidden},
           NetworkWeightTensorKind::NormBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::PositionalEncoding,
      "body.smolgen_positional",
      {
          {20, "body.smolgen_w", positional.size(), {kHidden, kHidden},
           NetworkWeightTensorKind::PositionalEncoding},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "policy.smoke.output",
      {
          {4, "policy.smoke.ip_pol_w", dense2_weights.size(), {kPolicy, kHidden},
           NetworkWeightTensorKind::DenseWeight},
          {5, "policy.smoke.ip_pol_b", dense2_bias.size(), {kPolicy},
           NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "policy.smoke.norm",
      {
          {6, "policy.smoke.norm_gammas", gamma2.size(), {kPolicy},
           NetworkWeightTensorKind::NormScale},
          {7, "policy.smoke.norm_betas", beta2.size(), {kPolicy},
           NetworkWeightTensorKind::NormBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "value.smoke.dense2",
      {
          {8, "value.smoke.ip2_val_w", dense3_weights.size(), {kValue, kHidden},
           NetworkWeightTensorKind::DenseWeight},
          {9, "value.smoke.ip2_val_b", dense3_bias.size(), {kValue},
           NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "moves_left.output",
      {
          {10, "moves_left.ip2_mov_w", moves_weights.size(),
           {kMovesLeft, kHidden}, NetworkWeightTensorKind::DenseWeight},
          {11, "moves_left.ip2_mov_b", moves_bias.size(), {kMovesLeft},
           NetworkWeightTensorKind::DenseBias},
      }});

  std::vector<float> hidden_activated(static_cast<size_t>(kBatch) * kHidden,
                                      0.0f);
  std::vector<float> hidden_normalized(static_cast<size_t>(kBatch) * kHidden,
                                       0.0f);
  std::vector<float> hidden_gated(static_cast<size_t>(kBatch) * kHidden,
                                  0.0f);
  std::vector<float> ffn_activated(static_cast<size_t>(kBatch) * kFfnHidden,
                                   0.0f);
  std::vector<float> ffn_output(static_cast<size_t>(kBatch) * kHidden, 0.0f);
  std::vector<float> hidden_ffn_normalized(
      static_cast<size_t>(kBatch) * kHidden, 0.0f);
  std::vector<float> activated(static_cast<size_t>(kBatch) * kPolicy, 0.0f);
  std::vector<float> policy_expected(static_cast<size_t>(kBatch) * kPolicy,
                                     0.0f);
  std::vector<float> value_expected(static_cast<size_t>(kBatch) * kValue,
                                    0.0f);
  std::vector<float> moves_expected(static_cast<size_t>(kBatch) * kMovesLeft,
                                    0.0f);
  for (int batch = 0; batch < kBatch; ++batch) {
    const size_t hidden_offset = static_cast<size_t>(batch) * kHidden;
    for (int out = 0; out < kHidden; ++out) {
      float dense = dense_bias[static_cast<size_t>(out)];
      for (int in = 0; in < kInput; ++in) {
        dense += input_values[static_cast<size_t>(batch) * kInput + in] *
                 dense_weights[static_cast<size_t>(out) * kInput + in];
      }
      const float relu = std::max(dense, 0.0f);
      hidden_activated[hidden_offset + out] = relu * relu;
    }

    float sum = 0.0f;
    float square_sum = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
      const float value = hidden_activated[hidden_offset + i];
      sum += value;
      square_sum += value * value;
    }
    const float mean = sum / static_cast<float>(kHidden);
    float variance = square_sum / static_cast<float>(kHidden) - mean * mean;
    if (variance < 0.0f)
      variance = 0.0f;
    const float inv_std = 1.0f / std::sqrt(variance + 1e-5f);
    for (int i = 0; i < kHidden; ++i) {
      const float normalized =
          (hidden_activated[hidden_offset + i] - mean) * inv_std;
      hidden_normalized[hidden_offset + i] =
          normalized * gamma[static_cast<size_t>(i)] +
          beta[static_cast<size_t>(i)];
      hidden_gated[hidden_offset + i] =
          hidden_normalized[hidden_offset + i] *
              mult_gate[static_cast<size_t>(i)] +
          add_gate[static_cast<size_t>(i)];
    }

    const size_t ffn_offset = static_cast<size_t>(batch) * kFfnHidden;
    for (int out = 0; out < kFfnHidden; ++out) {
      float dense = ffn1_bias[static_cast<size_t>(out)];
      for (int in = 0; in < kHidden; ++in) {
        dense += hidden_gated[hidden_offset + in] *
                 ffn1_weights[static_cast<size_t>(out) * kHidden + in];
      }
      const float relu = std::max(dense, 0.0f);
      ffn_activated[ffn_offset + out] = relu * relu;
    }
    for (int out = 0; out < kHidden; ++out) {
      float dense = ffn2_bias[static_cast<size_t>(out)];
      for (int in = 0; in < kFfnHidden; ++in) {
        dense += ffn_activated[ffn_offset + in] *
                 ffn2_weights[static_cast<size_t>(out) * kFfnHidden + in];
      }
      ffn_output[hidden_offset + out] = hidden_gated[hidden_offset + out] +
                                        dense;
    }

    sum = 0.0f;
    square_sum = 0.0f;
    for (int i = 0; i < kHidden; ++i) {
      const float value = ffn_output[hidden_offset + i];
      sum += value;
      square_sum += value * value;
    }
    const float ffn_mean = sum / static_cast<float>(kHidden);
    float ffn_variance =
        square_sum / static_cast<float>(kHidden) - ffn_mean * ffn_mean;
    if (ffn_variance < 0.0f)
      ffn_variance = 0.0f;
    const float ffn_inv_std = 1.0f / std::sqrt(ffn_variance + 1e-3f);
    for (int i = 0; i < kHidden; ++i) {
      const float normalized =
          (ffn_output[hidden_offset + i] - ffn_mean) * ffn_inv_std;
      hidden_ffn_normalized[hidden_offset + i] =
          normalized * ffn_gamma[static_cast<size_t>(i)] +
          ffn_beta[static_cast<size_t>(i)];
    }

    const size_t policy_offset = static_cast<size_t>(batch) * kPolicy;
    for (int out = 0; out < kPolicy; ++out) {
      float dense = dense2_bias[static_cast<size_t>(out)];
      for (int in = 0; in < kHidden; ++in) {
        dense += hidden_ffn_normalized[hidden_offset + in] *
                 dense2_weights[static_cast<size_t>(out) * kHidden + in];
      }
      const float relu = std::max(dense, 0.0f);
      activated[policy_offset + out] = relu * relu;
    }

    sum = 0.0f;
    square_sum = 0.0f;
    for (int i = 0; i < kPolicy; ++i) {
      const float value = activated[policy_offset + i];
      sum += value;
      square_sum += value * value;
    }
    const float output_mean = sum / static_cast<float>(kPolicy);
    float output_variance =
        square_sum / static_cast<float>(kPolicy) - output_mean * output_mean;
    if (output_variance < 0.0f)
      output_variance = 0.0f;
    const float output_inv_std = 1.0f / std::sqrt(output_variance + 1e-5f);
    for (int i = 0; i < kPolicy; ++i) {
      const float normalized =
          (activated[policy_offset + i] - output_mean) * output_inv_std;
      policy_expected[policy_offset + i] =
          normalized * gamma2[static_cast<size_t>(i)] +
          beta2[static_cast<size_t>(i)];
    }

    const size_t value_offset = static_cast<size_t>(batch) * kValue;
    for (int out = 0; out < kValue; ++out) {
      float dense = dense3_bias[static_cast<size_t>(out)];
      for (int in = 0; in < kHidden; ++in) {
        dense += hidden_ffn_normalized[hidden_offset + in] *
                 dense3_weights[static_cast<size_t>(out) * kHidden + in];
      }
      const float relu = std::max(dense, 0.0f);
      value_expected[value_offset + out] = relu * relu;
    }

    for (int out = 0; out < kMovesLeft; ++out) {
      float dense = moves_bias[static_cast<size_t>(out)];
      for (int in = 0; in < kHidden; ++in) {
        dense += hidden_ffn_normalized[hidden_offset + in] *
                 moves_weights[static_cast<size_t>(out) * kHidden + in];
      }
      const float relu = std::max(dense, 0.0f);
      moves_expected[static_cast<size_t>(batch) * kMovesLeft + out] =
          relu * relu;
    }
  }

  try {
    CudaInferenceBuffers buffers;
    buffers.Allocate(LayoutFromTensorPlan(plan, kBatch));
    CudaExecutionWorkspace workspace;
    cudaStream_t stream = workspace.Stream();
    buffers.UploadPackedInputs(input_masks, input_values, kBatch, stream);
    buffers.ClearOutputs(kBatch, stream);

    CudaWeightBuffers weights;
    weights.Upload(inventory);
    auto executor = CreatePlanSmokeCudaExecutor();
    executor->Execute(plan, execution_plan, weights, buffers, workspace,
                      kBatch);

    result.allocation_bytes = buffers.AllocationBytes();
    const auto downloaded = buffers.DownloadOutputs(kBatch, stream);
    if (downloaded.policy.size() != plan.PolicyEntries(kBatch) ||
        downloaded.value.size() != plan.ValueEntries(kBatch) ||
        downloaded.moves_left.size() != plan.MovesLeftEntries(kBatch) ||
        downloaded.raw_policy.size() != plan.RawPolicyEntries(kBatch)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA plan executor output size mismatch";
      return result;
    }

    for (int batch = 0; batch < kBatch; ++batch) {
      const size_t expected_policy_offset = static_cast<size_t>(batch) * kPolicy;
      const size_t actual_policy_offset =
          static_cast<size_t>(batch) * plan.policy_outputs;
      const size_t raw_policy_offset =
          static_cast<size_t>(batch) * plan.raw_policy_outputs;
      for (int i = 0; i < kPolicy; ++i) {
        if (std::fabs(downloaded.policy[actual_policy_offset + i] -
                      policy_expected[expected_policy_offset + i]) > 1e-5f ||
            std::fabs(downloaded.raw_policy[raw_policy_offset + i] -
                      policy_expected[expected_policy_offset + i]) > 1e-5f) {
          result.status = CudaSmokeStatus::Mismatch;
          result.message = "CUDA plan executor pipeline output mismatch";
          return result;
        }
      }
      if (downloaded.policy[actual_policy_offset + kPolicy] != 0.0f ||
          downloaded.raw_policy[raw_policy_offset + kPolicy] != 0.0f) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA plan executor row stride overwrite";
        return result;
      }

      const size_t expected_value_offset = static_cast<size_t>(batch) * kValue;
      const size_t actual_value_offset =
          static_cast<size_t>(batch) * plan.value_outputs;
      for (int i = 0; i < kValue; ++i) {
        if (std::fabs(downloaded.value[actual_value_offset + i] -
                      value_expected[expected_value_offset + i]) > 1e-5f) {
          result.status = CudaSmokeStatus::Mismatch;
          result.message = "CUDA plan executor value output mismatch";
          return result;
        }
      }

      const size_t expected_moves_offset =
          static_cast<size_t>(batch) * kMovesLeft;
      if (std::fabs(downloaded.moves_left[static_cast<size_t>(batch)] -
                    moves_expected[expected_moves_offset]) > 1e-5f) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA plan executor moves-left output mismatch";
        return result;
      }
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

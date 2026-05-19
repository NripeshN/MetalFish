/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_buffers.h"

#include "../network_output_decoder.h"
#include "../network_weight_inventory.h"
#include "cuda_executor.h"
#include "cuda_runtime_probe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
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
    executor->Execute(plan, resolved_plan, weights, buffers, batch_size);

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
  plan.raw_policy_outputs = kNetworkAttentionPolicyScratch;
  NetworkFormatDescriptor format;
  format.attention_policy = true;
  format.activations.ffn_activation = "relu_2";

  constexpr int kBatch = 1;
  constexpr int kInput = 3;
  constexpr int kOutput = 4;
  const std::vector<float> input_values = {1.0f, -2.0f, 0.5f};
  const std::vector<std::uint64_t> input_masks = {~0ULL, ~0ULL, ~0ULL};
  const std::vector<float> dense_weights = {
      1.0f, 0.0f, 0.5f,
      -0.5f, 1.0f, 0.25f,
      0.0f, -1.0f, 0.5f,
      2.0f, 0.5f, -1.0f,
  };
  const std::vector<float> dense_bias = {0.25f, -0.75f, 0.0f, 1.0f};
  const std::vector<float> gamma = {1.0f, 0.5f, -1.0f, 2.0f};
  const std::vector<float> beta = {0.0f, -0.25f, 0.5f, 1.0f};

  NetworkWeightInventory inventory;
  inventory.tensors = {
      {"smoke.dense_w", dense_weights.data(), dense_weights.size(),
       {kOutput, kInput}, NetworkWeightTensorKind::DenseWeight},
      {"smoke.dense_b", dense_bias.data(), dense_bias.size(), {kOutput},
       NetworkWeightTensorKind::DenseBias},
      {"smoke.norm_gammas", gamma.data(), gamma.size(), {kOutput},
       NetworkWeightTensorKind::NormScale},
      {"smoke.norm_betas", beta.data(), beta.size(), {kOutput},
       NetworkWeightTensorKind::NormBias},
  };

  NetworkResolvedExecutionPlan execution_plan;
  execution_plan.format = format;
  execution_plan.tensors = plan;
  execution_plan.policy_head = "smoke";
  execution_plan.value_head = "smoke";
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::Dense,
      "smoke.dense",
      {
          {0, "smoke.dense_w", dense_weights.size(), {kOutput, kInput},
           NetworkWeightTensorKind::DenseWeight},
          {1, "smoke.dense_b", dense_bias.size(), {kOutput},
           NetworkWeightTensorKind::DenseBias},
      }});
  execution_plan.steps.push_back(NetworkResolvedExecutionStep{
      NetworkExecutionOpKind::LayerNorm,
      "smoke.norm",
      {
          {2, "smoke.norm_gammas", gamma.size(), {kOutput},
           NetworkWeightTensorKind::NormScale},
          {3, "smoke.norm_betas", beta.size(), {kOutput},
           NetworkWeightTensorKind::NormBias},
      }});

  std::vector<float> dense(kOutput, 0.0f);
  std::vector<float> activated(kOutput, 0.0f);
  for (int out = 0; out < kOutput; ++out) {
    float sum = dense_bias[static_cast<size_t>(out)];
    for (int in = 0; in < kInput; ++in) {
      sum += input_values[static_cast<size_t>(in)] *
             dense_weights[static_cast<size_t>(out) * kInput + in];
    }
    dense[static_cast<size_t>(out)] = sum;
    const float relu = std::max(sum, 0.0f);
    activated[static_cast<size_t>(out)] = relu * relu;
  }

  float sum = 0.0f;
  float square_sum = 0.0f;
  for (float value : activated) {
    sum += value;
    square_sum += value * value;
  }
  const float mean = sum / static_cast<float>(kOutput);
  float variance = square_sum / static_cast<float>(kOutput) - mean * mean;
  if (variance < 0.0f)
    variance = 0.0f;
  const float inv_std = 1.0f / std::sqrt(variance + 1e-5f);
  std::vector<float> expected(kOutput, 0.0f);
  for (int i = 0; i < kOutput; ++i) {
    const float normalized =
        (activated[static_cast<size_t>(i)] - mean) * inv_std;
    expected[static_cast<size_t>(i)] =
        normalized * gamma[static_cast<size_t>(i)] +
        beta[static_cast<size_t>(i)];
  }

  try {
    CudaInferenceBuffers buffers;
    buffers.Allocate(LayoutFromTensorPlan(plan, kBatch));
    buffers.UploadPackedInputs(input_masks, input_values, kBatch);
    buffers.ClearOutputs(kBatch);

    CudaWeightBuffers weights;
    weights.Upload(inventory);
    auto executor = CreatePlanSmokeCudaExecutor();
    executor->Execute(plan, execution_plan, weights, buffers, kBatch);

    result.allocation_bytes = buffers.AllocationBytes();
    const auto downloaded = buffers.DownloadOutputs(kBatch);
    if (downloaded.policy.size() < expected.size() ||
        downloaded.raw_policy.size() < activated.size()) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA plan executor output size mismatch";
      return result;
    }

    for (std::size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(downloaded.policy[i] - expected[i]) > 1e-5f ||
          std::fabs(downloaded.raw_policy[i] - activated[i]) > 1e-5f) {
        result.status = CudaSmokeStatus::Mismatch;
        result.message = "CUDA plan executor pipeline output mismatch";
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

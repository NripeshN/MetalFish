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
  };

  NetworkResolvedExecutionPlan execution_plan;
  execution_plan.format.body_attention_heads = kHeads;
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

  const auto expected_q =
      DenseAffineHost(input, q_weight, q_bias, kRows, kInput, kQkv);
  const auto expected_k =
      DenseAffineHost(input, k_weight, k_bias, kRows, kInput, kQkv);
  const auto expected_v =
      DenseAffineHost(input, v_weight, v_bias, kRows, kInput, kQkv);
  const float attention_scale =
      1.0f / std::sqrt(static_cast<float>(kHeadDepth));
  const auto expected_scores = AttentionScoresHost(
      expected_q, expected_k, kBatch, kHeads, kCudaAttentionSquares,
      kHeadDepth, kQkv, attention_scale);
  const auto expected_probabilities = SoftmaxRowsHost(
      expected_scores, kBatch * kHeads * kCudaAttentionSquares,
      kCudaAttentionSquares);
  const auto expected_context = AttentionContextHost(
      expected_probabilities, expected_v, kBatch, kHeads, kCudaAttentionSquares,
      kHeadDepth, kQkv);
  const auto expected_projection = DenseAffineHost(
      expected_context, projection_weight, projection_bias, kRows, kQkv,
      kOutput);

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
        execution_plan, 0, weights, device_input, tape, workspace, kBatch);
    const auto core_output = ExecuteAttentionCoreStage(
        execution_plan, 0, input_output, tape, workspace, kBatch);
    const auto projection_output = ExecuteAttentionOutputProjectionStage(
        execution_plan, 0, weights, core_output.context, tape, workspace,
        kBatch);
    workspace.Synchronize();

    if (input_output.rows != kRows || input_output.qkv_width != kQkv ||
        input_output.heads != kHeads || input_output.head_depth != kHeadDepth ||
        core_output.score_rows != kBatch * kHeads * kCudaAttentionSquares ||
        core_output.score_width != kCudaAttentionSquares ||
        core_output.rows != kRows || core_output.qkv_width != kQkv ||
        projection_output.output_width != kOutput) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention metadata mismatch";
      return result;
    }

    std::vector<float> actual_q(expected_q.size(), 0.0f);
    std::vector<float> actual_k(expected_k.size(), 0.0f);
    std::vector<float> actual_v(expected_v.size(), 0.0f);
    std::vector<float> actual_probabilities(expected_probabilities.size(),
                                            0.0f);
    std::vector<float> actual_context(expected_context.size(), 0.0f);
    std::vector<float> actual_projection(expected_projection.size(), 0.0f);
    DownloadFloats(actual_q, input_output.query,
                   "cudaMemcpy(attention_q)");
    DownloadFloats(actual_k, input_output.key, "cudaMemcpy(attention_k)");
    DownloadFloats(actual_v, input_output.value,
                   "cudaMemcpy(attention_v)");
    DownloadFloats(actual_probabilities, core_output.probabilities,
                   "cudaMemcpy(attention_probabilities)");
    DownloadFloats(actual_context, core_output.context,
                   "cudaMemcpy(attention_context)");
    DownloadFloats(actual_projection, projection_output.projection,
                   "cudaMemcpy(attention_projection)");
    if (!AlmostEqual(actual_q, expected_q, 1e-5f) ||
        !AlmostEqual(actual_k, expected_k, 1e-5f) ||
        !AlmostEqual(actual_v, expected_v, 1e-5f) ||
        !AlmostEqual(actual_probabilities, expected_probabilities, 1e-5f) ||
        !AlmostEqual(actual_context, expected_context, 1e-5f) ||
        !AlmostEqual(actual_projection, expected_projection, 1e-5f)) {
      result.status = CudaSmokeStatus::Mismatch;
      result.message = "CUDA attention output mismatch";
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

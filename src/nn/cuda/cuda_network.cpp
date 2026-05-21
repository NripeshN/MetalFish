/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_network.h"

#include "../network_execution_plan.h"
#include "../network_output_decoder.h"
#include "../network_weight_inventory.h"
#include "cuda_execution_schedule.h"
#include "cuda_output_mapping.h"
#include "cuda_runtime_probe.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

constexpr int kDefaultMaxBatchSize = 256;
constexpr int kMaxStableExecutionBatchSize = 16;

bool EnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return false;
  return !(value[0] == '0' && value[1] == '\0');
}

int EnvIntOrDefault(const char *name, int fallback, int min_value,
                    int max_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  try {
    return std::clamp(std::stoi(value), min_value, max_value);
  } catch (...) {
    return fallback;
  }
}

bool IsFinitePolicy(const std::array<float, kPolicyOutputs> &policy) {
  for (float value : policy) {
    if (!std::isfinite(value))
      return false;
  }
  return true;
}

bool IsValidCudaOutput(const NetworkOutput &output,
                       const NetworkTensorPlan &plan) {
  if (!std::isfinite(output.value) || !IsFinitePolicy(output.policy))
    return false;

  if (plan.wdl) {
    if (!output.has_wdl)
      return false;
    const float wdl_sum = output.wdl[0] + output.wdl[1] + output.wdl[2];
    for (float value : output.wdl) {
      if (!std::isfinite(value))
        return false;
    }
    if (wdl_sum < 0.5f || wdl_sum > 1.5f)
      return false;
  }

  if (plan.moves_left) {
    if (!output.has_moves_left || !std::isfinite(output.moves_left))
      return false;
  }

  return true;
}

bool OutputsAreValid(const std::vector<NetworkOutput> &outputs,
                     const NetworkTensorPlan &plan) {
  for (const auto &output : outputs) {
    if (!IsValidCudaOutput(output, plan))
      return false;
  }
  return true;
}

} // namespace

CudaNetwork::CudaNetwork(const WeightsFile &weights)
    : format_(DescribeNetworkFormat(weights)),
      tensor_plan_(CreateNetworkTensorPlan(format_)),
      buffer_layout_(LayoutFromTensorPlan(tensor_plan_, kDefaultMaxBatchSize)),
      executor_(CreateMissingCudaExecutor()) {
  const auto device_selection = SelectCudaDevice();
  if (!device_selection.ok) {
    throw std::runtime_error(
        "CUDA transformer backend is compiled (" + RuntimeCudaDeviceSummary() +
        "; format: " + format_.Summary() + ") but " + device_selection.message);
  }
  device_selection_summary_ = device_selection.message;

  std::unique_ptr<MultiHeadWeights> decoded_weights;
  std::string policy_head;
  std::string value_head;
  try {
    decoded_weights = std::make_unique<MultiHeadWeights>(weights.weights());
    policy_head = SelectPolicyHeadName(*decoded_weights);
    value_head = SelectValueHeadName(*decoded_weights);
    const auto validation = ValidateNetworkTensorPlan(
        tensor_plan_, *decoded_weights, policy_head, value_head);
    if (!validation.ok()) {
      throw std::runtime_error(validation.Summary());
    }
  } catch (const std::exception &e) {
    throw std::runtime_error("CUDA transformer backend is compiled (" +
                             RuntimeCudaDeviceSummary() +
                             "; format: " + format_.Summary() +
                             ") but weight validation failed: " + e.what());
  }

  try {
    buffers_.Allocate(buffer_layout_);
  } catch (const std::exception &e) {
    throw std::runtime_error("CUDA transformer backend is compiled (" +
                             RuntimeCudaDeviceSummary() +
                             "; format: " + format_.Summary() +
                             ") but buffer allocation failed: " + e.what());
  }

  try {
    const auto inventory = CreateNetworkWeightInventory(
        *decoded_weights, policy_head, value_head, tensor_plan_);
    execution_plan_ = CreateNetworkExecutionPlan(
        format_, tensor_plan_, policy_head, value_head, inventory);
    const auto execution_validation =
        execution_plan_.ValidateAgainstInventory(inventory);
    if (!execution_validation.ok())
      throw std::runtime_error(execution_validation.Summary());
    resolved_execution_plan_ =
        ResolveNetworkExecutionPlan(execution_plan_, inventory);
    const auto resolved_inventory = CreateResolvedNetworkWeightInventory(
        inventory, resolved_execution_plan_);
    weight_buffers_.Upload(resolved_inventory);
  } catch (const std::exception &e) {
    throw std::runtime_error("CUDA transformer backend is compiled (" +
                             RuntimeCudaDeviceSummary() +
                             "; format: " + format_.Summary() +
                             ") but weight upload failed: " + e.what());
  }

  try {
    const auto schedule = CreateCudaExecutionSchedule(resolved_execution_plan_);
    if (!schedule.FullySupported()) {
      throw std::runtime_error(schedule.Summary());
    }
    const auto output_mapping = CreateCudaOutputMapping(
        tensor_plan_, resolved_execution_plan_, schedule);
    if (!output_mapping.ok()) {
      throw std::runtime_error(output_mapping.Summary());
    }
    executor_ = CreateResolvedCudaExecutor(schedule, output_mapping);
    WarmupExecution();
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "CUDA transformer backend is compiled (" + RuntimeCudaDeviceSummary() +
        "; format: " + format_.Summary() +
        "; execution=" + resolved_execution_plan_.Summary() +
        ") but executor setup failed: " + e.what());
  }
}

void CudaNetwork::WarmupExecution() {
  constexpr int kWarmupBatchSize = 1;

  const bool batch_size_changed = workspace_batch_size_ != kWarmupBatchSize;
  if (batch_size_changed) {
    workspace_.Release();
    workspace_batch_size_ = kWarmupBatchSize;
  }

  cudaStream_t stream = workspace_.Stream();
  buffers_.ClearAll(stream);
  std::vector<std::uint64_t> input_masks(
      tensor_plan_.InputMaskEntries(kWarmupBatchSize), 0);
  std::vector<float> input_values(
      tensor_plan_.InputValueEntries(kWarmupBatchSize), 0.0f);
  buffers_.UploadPackedInputs(input_masks, input_values, kWarmupBatchSize,
                              stream);
  buffers_.ClearOutputs(kWarmupBatchSize, stream);

  CudaProfileSuppressionScope suppress_profile;
  executor_->Execute(tensor_plan_, resolved_execution_plan_, weight_buffers_,
                     buffers_, workspace_, kWarmupBatchSize);
  workspace_.Synchronize();
}

NetworkOutput CudaNetwork::Evaluate(const InputPlanes &input) {
  auto outputs = RunBatch(std::span<const InputPlanes>(&input, 1));
  return std::move(outputs.front());
}

std::vector<NetworkOutput>
CudaNetwork::EvaluateBatch(const std::vector<InputPlanes> &inputs) {
  if (inputs.size() > static_cast<size_t>(kMaxStableExecutionBatchSize)) {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(inputs.size());
    for (size_t offset = 0; offset < inputs.size();) {
      const size_t chunk_size = std::min<size_t>(kMaxStableExecutionBatchSize,
                                                 inputs.size() - offset);
      auto chunk = RunBatch(
          std::span<const InputPlanes>(inputs.data() + offset, chunk_size));
      outputs.insert(outputs.end(), std::make_move_iterator(chunk.begin()),
                     std::make_move_iterator(chunk.end()));
      offset += chunk_size;
    }
    return outputs;
  }
  return RunBatch(inputs);
}

std::vector<NetworkOutput>
CudaNetwork::RunBatch(std::span<const InputPlanes> inputs) {
  if (inputs.empty())
    return {};

  const int requested_batch_size = static_cast<int>(inputs.size());
  const int min_execution_batch = EnvIntOrDefault(
      "METALFISH_CUDA_MIN_EXECUTION_BATCH", 1, 1, kMaxStableExecutionBatchSize);
  const int execution_batch_size =
      std::max(requested_batch_size, min_execution_batch);
  if (execution_batch_size > buffer_layout_.max_batch_size) {
    throw std::runtime_error("CUDA batch size exceeds configured max batch");
  }

  std::vector<const float *> input_plane_ptrs;
  input_plane_ptrs.reserve(static_cast<size_t>(execution_batch_size));
  for (const auto &input : inputs)
    input_plane_ptrs.push_back(input[0].data());
  while (static_cast<int>(input_plane_ptrs.size()) < execution_batch_size)
    input_plane_ptrs.push_back(input_plane_ptrs.back());

  std::vector<std::uint64_t> input_masks;
  std::vector<float> input_values;
  PackInputPlaneBatchHostRaw(input_plane_ptrs, input_masks, input_values);

  const bool force_full_buffer_clear =
      EnvFlagEnabled("METALFISH_CUDA_FULL_BUFFER_CLEAR");
  const bool release_workspace_each_run =
      EnvFlagEnabled("METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN");

  auto run_once = [&]() {
    if (release_workspace_each_run) {
      workspace_.Release();
      workspace_batch_size_ = 0;
    }

    const bool batch_size_changed =
        workspace_batch_size_ != execution_batch_size;
    if (batch_size_changed) {
      workspace_.Release();
      workspace_batch_size_ = execution_batch_size;
    }
    cudaStream_t stream = workspace_.Stream();
    if (batch_size_changed || force_full_buffer_clear)
      buffers_.ClearAll(stream);
    buffers_.UploadPackedInputs(input_masks, input_values,
                                execution_batch_size, stream);
    buffers_.ClearOutputs(execution_batch_size, stream);
    executor_->Execute(tensor_plan_, resolved_execution_plan_, weight_buffers_,
                       buffers_, workspace_, execution_batch_size);

    const auto downloaded =
        buffers_.DownloadOutputs(execution_batch_size, stream);
    return DecodeNetworkOutputBatch(
        tensor_plan_, downloaded.policy.data(), downloaded.policy.size(),
        downloaded.value.data(), downloaded.value.size(),
        downloaded.moves_left.data(), downloaded.moves_left.size(),
        execution_batch_size);
  };

  std::lock_guard<std::mutex> lock(execution_mutex_);
  auto outputs = run_once();
  if (OutputsAreValid(outputs, tensor_plan_)) {
    outputs.resize(static_cast<size_t>(requested_batch_size));
    return outputs;
  }

  workspace_.Release();
  workspace_batch_size_ = 0;
  outputs = run_once();
  if (OutputsAreValid(outputs, tensor_plan_)) {
    outputs.resize(static_cast<size_t>(requested_batch_size));
    return outputs;
  }

  throw std::runtime_error(
      "CUDA transformer backend produced invalid network output");
}

std::string CudaNetwork::GetNetworkInfo() const {
  std::ostringstream out;
  out << "CUDA transformer backend (" << RuntimeCudaDeviceSummary()
      << ", selection=" << device_selection_summary_
      << ", format: " << format_.Summary()
      << ", tensors: " << tensor_plan_.Summary()
      << ", execution: " << resolved_execution_plan_.Summary()
      << ", buffer_bytes=" << buffers_.AllocationBytes()
      << ", weight_bytes=" << weight_buffers_.AllocationBytes()
      << ", executor=" << executor_->Name() << ")";
  return out.str();
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

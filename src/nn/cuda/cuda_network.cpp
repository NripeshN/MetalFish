/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_network.h"

#include "../network_execution_plan.h"
#include "../network_format.h"
#include "../network_output_decoder.h"
#include "../network_weight_inventory.h"
#include "cuda_execution_schedule.h"
#include "cuda_input_packing.h"
#include "cuda_output_mapping.h"
#include "cuda_runtime_probe.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

constexpr int kDefaultMaxBatchSize = 256;
constexpr int kDefaultStableExecutionBatchSize = 16;

bool EnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return false;
  return !(value[0] == '0' && value[1] == '\0');
}

bool EnvFlagOrDefault(const char *name, bool fallback) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  return !(value[0] == '0' && value[1] == '\0');
}

int EnvIntOrDefault(const char *name, int fallback, int min_value,
                    int max_value) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0')
    return fallback;
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (!end || *end != '\0')
    return fallback;
  return std::clamp(static_cast<int>(parsed), min_value, max_value);
}

int StableExecutionBatchSize(const BackendConfig &config) {
  if (config.cuda_stable_execution_batch_size > 0) {
    return std::clamp(config.cuda_stable_execution_batch_size, 1,
                      kDefaultMaxBatchSize);
  }
  return EnvIntOrDefault("METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE",
                         kDefaultStableExecutionBatchSize, 1,
                         kDefaultMaxBatchSize);
}

const char *BoolText(bool value) { return value ? "true" : "false"; }

bool CudaGraphEffective(const std::string &executor_name) {
  if (executor_name.rfind("resolved+graph", 0) != 0)
    return false;
  return executor_name.find("fallback") == std::string::npos &&
         executor_name.find("incompatible") == std::string::npos;
}

bool FullBufferClearEffective(const BackendConfig &config) {
  return config.cuda_full_buffer_clear &&
         EnvFlagOrDefault("METALFISH_CUDA_FULL_BUFFER_CLEAR", true);
}

struct TensorTopEntry {
  int index = -1;
  float value = 0.0f;
};

TensorTopEntry TopEntry(const std::vector<float> &values, std::size_t offset,
                        int width) {
  TensorTopEntry top;
  if (width <= 0 || offset >= values.size())
    return top;
  const std::size_t available = values.size() - offset;
  const int count = std::min<int>(width, static_cast<int>(available));
  float best = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < count; ++i) {
    const float value = values[offset + static_cast<std::size_t>(i)];
    if (value > best) {
      best = value;
      top.index = i;
      top.value = value;
    }
  }
  return top;
}

double TensorSum(const std::vector<float> &values, std::size_t offset,
                 int width) {
  if (width <= 0 || offset >= values.size())
    return 0.0;
  const std::size_t available = values.size() - offset;
  const int count = std::min<int>(width, static_cast<int>(available));
  double sum = 0.0;
  for (int i = 0; i < count; ++i)
    sum += values[offset + static_cast<std::size_t>(i)];
  return sum;
}

void TraceRawOutputs(const CudaOutputDownload &downloaded,
                     const NetworkTensorPlan &plan, int batch_size) {
  if (!EnvFlagEnabled("METALFISH_CUDA_TRACE_RAW_OUTPUTS"))
    return;

  static std::atomic<int> run_counter{0};
  static std::atomic<int> trace_counter{0};
  const int run = run_counter.fetch_add(1, std::memory_order_relaxed);
  const int limit =
      EnvIntOrDefault("METALFISH_CUDA_TRACE_RAW_LIMIT", 64, 1, 100000);
  const int target_entry = EnvIntOrDefault("METALFISH_CUDA_TRACE_RAW_ENTRY", 0,
                                           -1, std::max(0, batch_size - 1));

  for (int b = 0; b < batch_size; ++b) {
    if (target_entry >= 0 && b != target_entry)
      continue;
    const int trace = trace_counter.fetch_add(1, std::memory_order_relaxed);
    if (trace >= limit)
      return;

    const std::size_t value_offset =
        static_cast<std::size_t>(b) * plan.value_outputs;
    const std::size_t policy_offset =
        static_cast<std::size_t>(b) * plan.policy_outputs;
    const std::size_t raw_policy_offset =
        static_cast<std::size_t>(b) * plan.raw_policy_outputs;
    const TensorTopEntry policy_top =
        TopEntry(downloaded.policy, policy_offset, plan.policy_outputs);
    const TensorTopEntry raw_policy_top = TopEntry(
        downloaded.raw_policy, raw_policy_offset, plan.raw_policy_outputs);

    std::ostringstream out;
    out << std::fixed << std::setprecision(6) << "CUDA_RAW_TRACE run=" << run
        << " trace=" << trace << " batch=" << batch_size << " entry=" << b;
    if (plan.wdl && value_offset + 2 < downloaded.value.size()) {
      out << " wdl=[" << downloaded.value[value_offset] << ','
          << downloaded.value[value_offset + 1] << ','
          << downloaded.value[value_offset + 2] << ']';
    } else if (value_offset < downloaded.value.size()) {
      out << " value=" << downloaded.value[value_offset];
    }
    if (plan.moves_left &&
        static_cast<std::size_t>(b) < downloaded.moves_left.size()) {
      out << " moves_left=" << downloaded.moves_left[static_cast<size_t>(b)];
    }
    out << " policy_top=" << policy_top.index << ':' << policy_top.value
        << " policy_sum="
        << TensorSum(downloaded.policy, policy_offset, plan.policy_outputs);
    if (plan.raw_policy_outputs > 0) {
      out << " raw_policy_top=" << raw_policy_top.index << ':'
          << raw_policy_top.value << " raw_policy_sum="
          << TensorSum(downloaded.raw_policy, raw_policy_offset,
                       plan.raw_policy_outputs);
    }
    std::cerr << out.str() << std::endl;
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

CudaNetwork::CudaNetwork(const WeightsFile &weights, BackendConfig config)
    : format_(DescribeNetworkFormat(weights)),
      config_(std::move(config)),
      tensor_plan_(CreateNetworkTensorPlan(format_)),
      buffer_layout_(LayoutFromTensorPlan(tensor_plan_, kDefaultMaxBatchSize)),
      executor_(CreateMissingCudaExecutor()) {
  const auto device_selection = SelectCudaDevice(config_.cuda_device);
  if (!device_selection.ok) {
    throw std::runtime_error(
        "CUDA transformer backend is compiled (" + RuntimeCudaDeviceSummary() +
        "; format: " + format_.Summary() + ") but " + device_selection.message);
  }
  device_selection_summary_ = device_selection.message;
  selected_cuda_device_ = device_selection.device;

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
    CudaStageExecutionOptions stage_options;
    stage_options.deterministic_attention_softmax =
        config_.cuda_deterministic_attention_softmax;
    executor_ = CreateResolvedCudaExecutor(
        schedule, output_mapping, config_.cuda_graph_execution, stage_options);
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

  if (workspace_batch_size_ != kWarmupBatchSize)
    workspace_batch_size_ = kWarmupBatchSize;

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
  const auto max_stable_batch =
      static_cast<size_t>(StableExecutionBatchSize(config_));
  if (inputs.size() > max_stable_batch) {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(inputs.size());
    for (size_t offset = 0; offset < inputs.size();) {
      const size_t chunk_size =
          std::min<size_t>(max_stable_batch, inputs.size() - offset);
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

  const int batch_size = static_cast<int>(inputs.size());
  if (batch_size > buffer_layout_.max_batch_size) {
    throw std::runtime_error("CUDA batch size exceeds configured max batch");
  }

  std::vector<const float *> input_plane_ptrs;
  input_plane_ptrs.reserve(inputs.size());
  for (const auto &input : inputs)
    input_plane_ptrs.push_back(input[0].data());

  std::vector<std::uint64_t> input_masks;
  std::vector<float> input_values;
  PackInputPlaneBatchHostRaw(input_plane_ptrs, input_masks, input_values);

  const bool force_full_buffer_clear =
      config_.cuda_full_buffer_clear &&
      EnvFlagOrDefault("METALFISH_CUDA_FULL_BUFFER_CLEAR", true);
  const bool release_workspace_each_run =
      EnvFlagEnabled("METALFISH_CUDA_RELEASE_WORKSPACE_EACH_RUN");
  const bool release_single_workspace_each_run =
      batch_size == 1 &&
      EnvFlagEnabled("METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN");

  auto run_once = [&]() {
    if (release_workspace_each_run) {
      workspace_.Release();
      workspace_batch_size_ = 0;
    }

    const bool batch_size_changed = workspace_batch_size_ != batch_size;
    if (batch_size_changed)
      workspace_batch_size_ = batch_size;
    cudaStream_t stream = workspace_.Stream();
    if (batch_size_changed || force_full_buffer_clear)
      buffers_.ClearAll(stream);
    buffers_.UploadPackedInputs(input_masks, input_values, batch_size, stream);
    buffers_.ClearOutputs(batch_size, stream);
    executor_->Execute(tensor_plan_, resolved_execution_plan_, weight_buffers_,
                       buffers_, workspace_, batch_size);

    const bool trace_raw_outputs =
        EnvFlagEnabled("METALFISH_CUDA_TRACE_RAW_OUTPUTS");
    const auto downloaded =
        buffers_.DownloadOutputs(batch_size, stream, trace_raw_outputs);
    TraceRawOutputs(downloaded, tensor_plan_, batch_size);
    return DecodeNetworkOutputBatch(
        tensor_plan_, downloaded.policy.data(), downloaded.policy.size(),
        downloaded.value.data(), downloaded.value.size(),
        downloaded.moves_left.data(), downloaded.moves_left.size(), batch_size);
  };

  auto release_single_workspace_if_requested = [&]() {
    if (!release_single_workspace_each_run)
      return;
    workspace_.Release();
    workspace_batch_size_ = 0;
  };

  std::lock_guard<std::mutex> lock(execution_mutex_);
  auto outputs = run_once();
  if (OutputsAreValid(outputs, tensor_plan_)) {
    release_single_workspace_if_requested();
    return outputs;
  }

  workspace_.Release();
  workspace_batch_size_ = 0;
  outputs = run_once();
  if (OutputsAreValid(outputs, tensor_plan_)) {
    release_single_workspace_if_requested();
    return outputs;
  }

  throw std::runtime_error(
      "CUDA transformer backend produced invalid network output");
}

std::string CudaNetwork::GetNetworkInfo() const {
  std::ostringstream out;
  const std::string executor_name = executor_->Name();
  out << "CUDA transformer backend (" << RuntimeCudaDeviceSummary()
      << ", selection=" << device_selection_summary_
      << ", cuda_device_config=" << config_.cuda_device
      << ", cuda_device_selected=" << selected_cuda_device_
      << ", cuda_graph_config=" << BoolText(config_.cuda_graph_execution)
      << ", cuda_graph_effective=" << BoolText(CudaGraphEffective(executor_name))
      << ", cuda_stable_execution_batch_config="
      << config_.cuda_stable_execution_batch_size
      << ", cuda_stable_execution_batch_effective="
      << StableExecutionBatchSize(config_)
      << ", cuda_deterministic_attention_softmax="
      << BoolText(config_.cuda_deterministic_attention_softmax)
      << ", cuda_full_buffer_clear_config="
      << BoolText(config_.cuda_full_buffer_clear)
      << ", cuda_full_buffer_clear_effective="
      << BoolText(FullBufferClearEffective(config_))
      << ", format: " << format_.Summary()
      << ", tensors: " << tensor_plan_.Summary()
      << ", execution: " << resolved_execution_plan_.Summary()
      << ", buffer_bytes=" << buffers_.AllocationBytes()
      << ", weight_bytes=" << weight_buffers_.AllocationBytes()
      << ", executor=" << executor_name << ")";
  return out.str();
}

bool CudaNetwork::HasWDL() const { return tensor_plan_.wdl; }

bool CudaNetwork::HasMovesLeft() const { return tensor_plan_.moves_left; }

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "cuda_network.h"

#include "../network_execution_plan.h"
#include "../network_output_decoder.h"
#include "../network_weight_inventory.h"
#include "cuda_runtime_probe.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Cuda {
namespace {

constexpr int kDefaultMaxBatchSize = 256;

} // namespace

CudaNetwork::CudaNetwork(const WeightsFile &weights)
    : format_(DescribeNetworkFormat(weights)),
      tensor_plan_(CreateNetworkTensorPlan(format_)),
      buffer_layout_(LayoutFromTensorPlan(tensor_plan_, kDefaultMaxBatchSize)),
      executor_(CreateMissingCudaExecutor()) {
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

  const int device_count = RuntimeCudaDeviceCount();
  if (device_count <= 0) {
    throw std::runtime_error(
        "CUDA transformer backend is compiled (" + RuntimeCudaDeviceSummary() +
        "; format: " + format_.Summary() + ") but no CUDA device is available");
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
    weight_buffers_.Upload(inventory);
  } catch (const std::exception &e) {
    throw std::runtime_error("CUDA transformer backend is compiled (" +
                             RuntimeCudaDeviceSummary() +
                             "; format: " + format_.Summary() +
                             ") but weight upload failed: " + e.what());
  }

  throw std::runtime_error(
      "CUDA transformer backend is compiled (" + RuntimeCudaDeviceSummary() +
      "; format: " + format_.Summary() +
      "; execution=" + resolved_execution_plan_.Summary() +
      "; buffer_bytes=" + std::to_string(buffers_.AllocationBytes()) +
      "; weight_bytes=" + std::to_string(weight_buffers_.AllocationBytes()) +
      ") but inference is not implemented yet");
}

NetworkOutput CudaNetwork::Evaluate(const InputPlanes &input) {
  std::vector<InputPlanes> batch;
  batch.push_back(input);
  auto outputs = EvaluateBatch(batch);
  return std::move(outputs.front());
}

std::vector<NetworkOutput>
CudaNetwork::EvaluateBatch(const std::vector<InputPlanes> &inputs) {
  if (inputs.empty())
    return {};

  const int batch_size = static_cast<int>(inputs.size());
  if (batch_size > buffer_layout_.max_batch_size) {
    throw std::runtime_error("CUDA batch size exceeds configured max batch");
  }

  std::vector<std::uint64_t> input_masks;
  std::vector<float> input_values;
  PackInputPlanesHostRaw(inputs.front()[0].data(), batch_size, input_masks,
                         input_values);

  std::lock_guard<std::mutex> lock(execution_mutex_);
  buffers_.UploadPackedInputs(input_masks, input_values, batch_size);
  buffers_.ClearOutputs(batch_size);
  executor_->Execute(tensor_plan_, resolved_execution_plan_, weight_buffers_,
                     buffers_, workspace_, batch_size);

  const auto downloaded = buffers_.DownloadOutputs(batch_size);
  const float *moves_left =
      downloaded.moves_left.empty() ? nullptr : downloaded.moves_left.data();
  return DecodeNetworkOutputBatch(
      tensor_plan_, downloaded.policy.data(), downloaded.policy.size(),
      downloaded.value.data(), downloaded.value.size(), moves_left,
      downloaded.moves_left.size(), batch_size);
}

std::string CudaNetwork::GetNetworkInfo() const {
  std::ostringstream out;
  out << "CUDA transformer backend (" << RuntimeCudaDeviceSummary()
      << ", format: " << format_.Summary()
      << ", tensors: " << tensor_plan_.Summary()
      << ", execution: " << resolved_execution_plan_.Summary()
      << ", buffer_bytes=" << buffers_.AllocationBytes()
      << ", weight_bytes=" << weight_buffers_.AllocationBytes()
      << ", executor=" << executor_->Name() << ", inference not implemented)";
  return out.str();
}

} // namespace Cuda
} // namespace NN
} // namespace MetalFish

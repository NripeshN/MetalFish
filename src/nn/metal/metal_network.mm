/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "metal_network.h"

#import <Foundation/Foundation.h>

#include "../input_plane_packing.h"
#include "../network_format.h"
#include "../network_output_decoder.h"

#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace MetalFish {
namespace NN {
namespace Metal {

MetalNetwork::MetalNetwork(const WeightsFile &file, int gpu_id, int max_batch,
                           int batch)
    : max_batch_size_(max_batch), batch_size_(batch) {
  const auto descriptor = DescribeNetworkFormat(file);
  wdl_ = descriptor.wdl;
  moves_left_ = descriptor.moves_left;
  conv_policy_ = descriptor.conv_policy;
  attn_policy_ = descriptor.attention_policy;
  tensor_plan_ = CreateNetworkTensorPlan(descriptor);
  decoded_output_targets_ = NetworkDecodedOutputTargets(tensor_plan_);

  MultiHeadWeights weights(file.weights());

  builder_ = std::make_unique<MetalNetworkBuilder>();
  device_name_ = builder_->init(gpu_id);

  Activations activations;
  activations.default_activation = descriptor.activations.default_activation;
  activations.smolgen_activation = descriptor.activations.smolgen_activation;
  activations.ffn_activation = descriptor.activations.ffn_activation;

  std::string policy_head = SelectPolicyHeadName(weights);
  std::string value_head = SelectValueHeadName(weights);
  const auto validation =
      ValidateNetworkTensorPlan(tensor_plan_, weights, policy_head, value_head);
  if (!validation.ok()) {
    throw std::runtime_error("Network tensor validation failed: " +
                             validation.Summary());
  }

  builder_->build(kInputPlanes, weights, descriptor.input_embedding,
                  descriptor.attention_body, attn_policy_, conv_policy_, wdl_,
                  moves_left_, activations, policy_head, value_head,
                  decoded_output_targets_);
}

MetalNetwork::~MetalNetwork() = default;

InputsOutputs *MetalNetwork::AcquireIO() {
#ifdef __APPLE__
  os_unfair_lock_lock(&io_pool_lock_);
#else
  std::lock_guard<std::mutex> lock(io_pool_mutex_);
#endif
  InputsOutputs *io = nullptr;
  if (!io_pool_.empty()) {
    io = io_pool_.back().release();
    io_pool_.pop_back();
  }
#ifdef __APPLE__
  os_unfair_lock_unlock(&io_pool_lock_);
#endif
  if (!io) {
    io = new InputsOutputs(max_batch_size_, tensor_plan_);
  }
  return io;
}

void MetalNetwork::ReleaseIO(InputsOutputs *io) {
#ifdef __APPLE__
  os_unfair_lock_lock(&io_pool_lock_);
  io_pool_.emplace_back(io);
  os_unfair_lock_unlock(&io_pool_lock_);
#else
  std::lock_guard<std::mutex> lock(io_pool_mutex_);
  io_pool_.emplace_back(io);
#endif
}

NetworkOutput MetalNetwork::Evaluate(const InputPlanes &input) {
  std::vector<NetworkOutput> outputs(1);
  RunBatch(std::span<const InputPlanes>(&input, 1), outputs);
  return std::move(outputs.front());
}

std::vector<NetworkOutput>
MetalNetwork::EvaluateBatch(const std::vector<InputPlanes> &inputs) {
  if (inputs.empty())
    return {};
  std::vector<NetworkOutput> outputs(inputs.size());
  RunBatch(inputs, outputs);
  return outputs;
}

void MetalNetwork::RunBatch(std::span<const InputPlanes> inputs,
                            std::vector<NetworkOutput> &outputs) {
  if (inputs.empty())
    return;

  const int batch = static_cast<int>(inputs.size());
  if (batch > max_batch_size_) {
    throw std::runtime_error("Batch size exceeds configured max batch size");
  }

  InputsOutputs *io = AcquireIO();

  for (int b = 0; b < batch; ++b) {
    const int base = b * kInputPlanes;
    for (int p = 0; p < kInputPlanes; ++p) {
      uint64_t mask;
      float value;
      PackInputPlaneRaw(inputs[b][p].data(), p, mask, value);
      io->input_masks_mem_[base + p] = mask;
      io->input_val_mem_[base + p] = value;
    }
  }

  {
    std::vector<float *> output_mems;
    output_mems.reserve(decoded_output_targets_.size());
    for (NetworkOutputTarget target : decoded_output_targets_) {
      output_mems.push_back(io->OutputBuffer(target).data());
    }

    std::lock_guard<std::mutex> lock(gpu_mutex_);
    builder_->forwardEval(&io->input_val_mem_[0], &io->input_masks_mem_[0],
                          batch, std::move(output_mems));
  }

  const float *moves_left = moves_left_ && !io->op_moves_left_mem_.empty()
                                ? io->op_moves_left_mem_.data()
                                : nullptr;
  auto decoded = DecodeNetworkOutputBatch(
      tensor_plan_, io->op_policy_mem_.data(),
      tensor_plan_.PolicyEntries(batch), io->op_value_mem_.data(),
      tensor_plan_.ValueEntries(batch), moves_left,
      tensor_plan_.MovesLeftEntries(batch), batch);
  for (int b = 0; b < batch; ++b) {
    outputs[b] = std::move(decoded[static_cast<size_t>(b)]);
  }

  ReleaseIO(io);
}

std::string MetalNetwork::GetNetworkInfo() const {
  std::ostringstream oss;
  oss << "Metal (MPSGraph) backend\n";
  oss << "Device: " << device_name_ << "\n";
  oss << "Policy: "
      << (attn_policy_ ? "attention" : (conv_policy_ ? "conv" : "classical"))
      << "\n";
  oss << "Value head: " << (wdl_ ? "WDL" : "scalar") << "\n";
  oss << "Moves left: " << (moves_left_ ? "yes" : "no") << "\n";
  oss << "Precision: " << (HalfPrecisionEnabled() ? "FP16" : "FP32");
  return oss.str();
}

bool MetalNetwork::HasWDL() const { return wdl_; }

bool MetalNetwork::HasMovesLeft() const { return moves_left_; }

BackendCapabilities MetalNetwork::GetBackendCapabilities() const {
  BackendCapabilities capabilities;
  capabilities.actual_backend = "metal";
  capabilities.has_wdl = wdl_;
  capabilities.has_moves_left = moves_left_;
  capabilities.max_batch_size = max_batch_size_;
  capabilities.stable_execution_batch_size = batch_size_;
  capabilities.device_name = device_name_;
  return capabilities;
}

} // namespace Metal
} // namespace NN
} // namespace MetalFish

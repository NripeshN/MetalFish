/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "metal_network.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace MetalFish {
namespace NN {
namespace Metal {

namespace {

std::string ActivationToString(
    MetalFishNN::NetworkFormat_ActivationFunction act) {
  switch (act) {
    case MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_RELU:
      return "relu";
    case MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_MISH:
      return "mish";
    case MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_SWISH:
      return "swish";
    case MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_RELU_2:
      return "relu_2";
    case MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_SELU:
      return "selu";
    case MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_TANH:
      return "tanh";
    case MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_SIGMOID:
      return "sigmoid";
    default:
      return "relu";
  }
}

}  // namespace

MetalNetwork::MetalNetwork(const WeightsFile& file, int gpu_id,
                           int max_batch, int batch)
    : wdl_(file.format().network_format().value() ==
           MetalFishNN::NetworkFormat_ValueFormat_VALUE_WDL),
      moves_left_(file.format().network_format().moves_left() ==
                  MetalFishNN::NetworkFormat_MovesLeftFormat_MOVES_LEFT_V1),
      conv_policy_(file.format().network_format().policy() ==
                   MetalFishNN::NetworkFormat_PolicyFormat_POLICY_CONVOLUTION),
      attn_policy_(file.format().network_format().policy() ==
                   MetalFishNN::NetworkFormat_PolicyFormat_POLICY_ATTENTION),
      max_batch_size_(max_batch),
      batch_size_(batch) {
  // Build weights representation.
  MultiHeadWeights weights(file.weights());

  // Initialize Metal builder.
  builder_ = std::make_unique<MetalNetworkBuilder>();
  device_name_ = builder_->init(gpu_id, max_batch_size_);

  // Activation selection.
  const auto& nf = file.format().network_format();
  Activations activations;
  activations.default_activation =
      (nf.default_activation() ==
       MetalFishNN::NetworkFormat_DefaultActivation_DEFAULT_ACTIVATION_MISH)
          ? "mish"
          : "relu";
  activations.smolgen_activation =
      ActivationToString(nf.smolgen_activation());
  if (activations.smolgen_activation == "relu" &&
      nf.smolgen_activation() ==
          MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_DEFAULT) {
    activations.smolgen_activation = activations.default_activation;
  }
  activations.ffn_activation = ActivationToString(nf.ffn_activation());
  if (activations.ffn_activation == "relu" &&
      nf.ffn_activation() ==
          MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_DEFAULT) {
    activations.ffn_activation = activations.default_activation;
  }

  // Policy/value head selection.
  std::string policy_head = "vanilla";
  if (weights.policy_heads.count(policy_head) == 0) {
    if (!weights.policy_heads.empty()) {
      policy_head = weights.policy_heads.begin()->first;
    }
  }
  std::string value_head = "winner";
  if (weights.value_heads.count(value_head) == 0) {
    if (!weights.value_heads.empty()) {
      value_head = weights.value_heads.begin()->first;
    }
  }

  const bool attn_body =
      nf.network() ==
          MetalFishNN::NetworkFormat_NetworkStructure_NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
      nf.network() ==
          MetalFishNN::NetworkFormat_NetworkStructure_NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;

  auto embedding =
      static_cast<InputEmbedding>(nf.input_embedding());

  builder_->build(kInputPlanes, weights, embedding, attn_body, attn_policy_,
                  conv_policy_, wdl_, moves_left_, activations, policy_head,
                  value_head);
}

MetalNetwork::~MetalNetwork() = default;

NetworkOutput MetalNetwork::Evaluate(const InputPlanes& input) {
  auto outputs = EvaluateBatch({input});
  return outputs.front();
}

std::vector<NetworkOutput> MetalNetwork::EvaluateBatch(
    const std::vector<InputPlanes>& inputs) {
  std::vector<NetworkOutput> outputs(inputs.size());
  RunBatch(inputs, outputs);
  return outputs;
}

void MetalNetwork::RunBatch(const std::vector<InputPlanes>& inputs,
                            std::vector<NetworkOutput>& outputs) {
  const int batch = static_cast<int>(inputs.size());
  if (batch > max_batch_size_) {
    throw std::runtime_error("Batch size exceeds configured max batch size");
  }

  // Allocate buffers at max batch size and pad with zeros for stability.
  InputsOutputs io(max_batch_size_, wdl_, moves_left_, conv_policy_, attn_policy_);

  // Pack inputs into mask/value representation.
  for (int b = 0; b < batch; ++b) {
    for (int p = 0; p < kInputPlanes; ++p) {
      const auto& plane = inputs[b][p];
      uint64_t mask = 0;
      float value = 0.0f;
      for (int sq = 0; sq < 64; ++sq) {
        float v = plane[sq];
        if (v != 0.0f) {
          mask |= (1ULL << sq);
          value = v;
        }
      }
      io.input_masks_mem_[b * kInputPlanes + p] = mask;
      io.input_val_mem_[b * kInputPlanes + p] = value;
    }
  }
  // Pad remaining entries to avoid uninitialized data when batch < max.
  for (int b = batch; b < max_batch_size_; ++b) {
    for (int p = 0; p < kInputPlanes; ++p) {
      io.input_masks_mem_[b * kInputPlanes + p] = 0;
      io.input_val_mem_[b * kInputPlanes + p] = 0.0f;
    }
  }

  {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    const int eval_batch = max_batch_size_;
    if (moves_left_) {
      builder_->forwardEval(&io.input_val_mem_[0], &io.input_masks_mem_[0],
                            eval_batch,
                            {&io.op_policy_mem_[0], &io.op_value_mem_[0],
                             &io.op_moves_left_mem_[0]});
    } else {
      builder_->forwardEval(&io.input_val_mem_[0], &io.input_masks_mem_[0],
                            eval_batch,
                            {&io.op_policy_mem_[0], &io.op_value_mem_[0]});
    }
  }

  // Convert outputs.
  for (int b = 0; b < batch; ++b) {
    NetworkOutput out;
    out.policy.resize(kNumOutputPolicy);
    std::memcpy(out.policy.data(),
                &io.op_policy_mem_[b * kNumOutputPolicy],
                sizeof(float) * kNumOutputPolicy);

    if (wdl_) {
      out.has_wdl = true;
      out.wdl[0] = io.op_value_mem_[b * 3 + 0];
      out.wdl[1] = io.op_value_mem_[b * 3 + 1];
      out.wdl[2] = io.op_value_mem_[b * 3 + 2];
      out.value = out.wdl[0] - out.wdl[2];
    } else {
      out.has_wdl = false;
      out.value = io.op_value_mem_[b];
      out.wdl[0] = out.wdl[1] = out.wdl[2] = 0.0f;
    }
    if (moves_left_) {
      out.has_moves_left = true;
      out.moves_left = io.op_moves_left_mem_[b];
    }
    outputs[b] = out;
  }
}

std::string MetalNetwork::GetNetworkInfo() const {
  std::ostringstream oss;
  oss << "Metal (MPSGraph) backend\n";
  oss << "Device: " << device_name_ << "\n";
  oss << "Policy: " << (attn_policy_ ? "attention" : (conv_policy_ ? "conv" : "classical")) << "\n";
  oss << "Value head: " << (wdl_ ? "WDL" : "scalar") << "\n";
  oss << "Moves left: " << (moves_left_ ? "yes" : "no");
  return oss.str();
}

}  // namespace Metal
}  // namespace NN
}  // namespace MetalFish

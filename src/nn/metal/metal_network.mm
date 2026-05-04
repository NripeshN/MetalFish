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
#include <span>
#include <sstream>
#include <stdexcept>

namespace MetalFish {
namespace NN {
namespace Metal {

namespace {

std::string
ActivationToString(MetalFishNN::NetworkFormat_ActivationFunction act) {
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

inline bool IsSchemaUniformPlane(int plane) {
  return (plane < kAuxPlaneBase &&
          plane % kPlanesPerBoard == kPlanesPerBoard - 1) ||
         plane == kAuxPlaneBase + 2 || plane == kAuxPlaneBase + 3 ||
         plane >= kAuxPlaneBase + 5;
}

inline void PackInputPlane(const InputPlanes::value_type &plane, int plane_idx,
                           uint64_t &mask, float &value) {
  if (IsSchemaUniformPlane(plane_idx)) {
    value = plane[0];
    mask = value != 0.0f ? ~0ULL : 0ULL;
    return;
  }

  mask = 0;
  value = 0.0f;
  for (int sq = 0; sq < 64; ++sq) {
    const float v = plane[sq];
    if (v != 0.0f) {
      if (mask == 0)
        value = v;
      mask |= (1ULL << sq);
    }
  }
}

} // namespace

MetalNetwork::MetalNetwork(const WeightsFile &file, int gpu_id, int max_batch,
                           int batch)
    : wdl_(file.format().network_format().value() ==
           MetalFishNN::NetworkFormat_ValueFormat_VALUE_WDL),
      moves_left_(file.format().network_format().moves_left() ==
                  MetalFishNN::NetworkFormat_MovesLeftFormat_MOVES_LEFT_V1),
      conv_policy_(file.format().network_format().policy() ==
                   MetalFishNN::NetworkFormat_PolicyFormat_POLICY_CONVOLUTION),
      attn_policy_(file.format().network_format().policy() ==
                   MetalFishNN::NetworkFormat_PolicyFormat_POLICY_ATTENTION),
      max_batch_size_(max_batch), batch_size_(batch) {
  // Build weights representation.
  MultiHeadWeights weights(file.weights());

  // Initialize Metal builder.
  builder_ = std::make_unique<MetalNetworkBuilder>();
  device_name_ = builder_->init(gpu_id);

  // Activation selection.
  const auto &nf = file.format().network_format();
  Activations activations;
  activations.default_activation =
      (nf.default_activation() ==
       MetalFishNN::NetworkFormat_DefaultActivation_DEFAULT_ACTIVATION_MISH)
          ? "mish"
          : "relu";
  activations.smolgen_activation = ActivationToString(nf.smolgen_activation());
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
          MetalFishNN::
              NetworkFormat_NetworkStructure_NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
      nf.network() ==
          MetalFishNN::
              NetworkFormat_NetworkStructure_NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;

  auto embedding = static_cast<InputEmbedding>(
      nf.has_input_embedding()
          ? nf.input_embedding()
          : MetalFishNN::NetworkFormat::INPUT_EMBEDDING_PE_MAP);

  builder_->build(kInputPlanes, weights, embedding, attn_body, attn_policy_,
                  conv_policy_, wdl_, moves_left_, activations, policy_head,
                  value_head);
}

MetalNetwork::~MetalNetwork() = default;

// ---- Buffer pool: avoids heap allocation on every inference call ----

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
    io = new InputsOutputs(max_batch_size_, wdl_, moves_left_, conv_policy_,
                           attn_policy_);
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

// ---- Inference entry points ----

NetworkOutput MetalNetwork::Evaluate(const InputPlanes &input) {
  std::vector<NetworkOutput> outputs(1);
  RunBatch(std::span<const InputPlanes>(&input, 1), outputs);
  return outputs.front();
}

std::vector<NetworkOutput>
MetalNetwork::EvaluateBatch(const std::vector<InputPlanes> &inputs) {
  std::vector<NetworkOutput> outputs(inputs.size());
  RunBatch(inputs, outputs);
  return outputs;
}

void MetalNetwork::RunBatch(std::span<const InputPlanes> inputs,
                            std::vector<NetworkOutput> &outputs) {
  const int batch = static_cast<int>(inputs.size());
  if (batch > max_batch_size_) {
    throw std::runtime_error("Batch size exceeds configured max batch size");
  }

  // Acquire a pre-allocated IO buffer from the pool (no heap alloc).
  InputsOutputs *io = AcquireIO();

  // Pack inputs into mask/value representation.
  // Optimized: known uniform planes can be represented without scanning 64
  // floats. This is hot in MCTS/Hybrid because every transformer request is
  // repacked into Metal's mask/value layout.
  for (int b = 0; b < batch; ++b) {
    const int base = b * kInputPlanes;
    for (int p = 0; p < kInputPlanes; ++p) {
      uint64_t mask;
      float value;
      PackInputPlane(inputs[b][p], p, mask, value);
      io->input_masks_mem_[base + p] = mask;
      io->input_val_mem_[base + p] = value;
    }
  }

  {
    std::lock_guard<std::mutex> lock(gpu_mutex_);
    if (moves_left_) {
      builder_->forwardEval(&io->input_val_mem_[0], &io->input_masks_mem_[0],
                            batch,
                            {&io->op_policy_mem_[0], &io->op_value_mem_[0],
                             &io->op_moves_left_mem_[0]});
    } else {
      builder_->forwardEval(&io->input_val_mem_[0], &io->input_masks_mem_[0],
                            batch,
                            {&io->op_policy_mem_[0], &io->op_value_mem_[0]});
    }
  }

  // Convert outputs.
  for (int b = 0; b < batch; ++b) {
    NetworkOutput &out = outputs[b];
    out.policy.resize(kNumOutputPolicy);
    std::memcpy(out.policy.data(), &io->op_policy_mem_[b * kNumOutputPolicy],
                sizeof(float) * kNumOutputPolicy);

    if (wdl_) {
      out.has_wdl = true;
      out.wdl[0] = io->op_value_mem_[b * 3 + 0];
      out.wdl[1] = io->op_value_mem_[b * 3 + 1];
      out.wdl[2] = io->op_value_mem_[b * 3 + 2];
      out.value = out.wdl[0] - out.wdl[2];
    } else {
      out.has_wdl = false;
      out.value = io->op_value_mem_[b];
      out.wdl[0] = out.wdl[1] = out.wdl[2] = 0.0f;
    }
    if (moves_left_) {
      out.has_moves_left = true;
      out.moves_left = io->op_moves_left_mem_[b];
    }
  }

  // Return IO buffer to the pool for reuse.
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
  oss << "Moves left: " << (moves_left_ ? "yes" : "no");
  return oss.str();
}

} // namespace Metal
} // namespace NN
} // namespace MetalFish

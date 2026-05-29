/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network_format.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace {

std::string BoolString(bool value) { return value ? "yes" : "no"; }

std::string InputEmbeddingToString(InputEmbedding embedding) {
  switch (embedding) {
  case INPUT_EMBEDDING_NONE:
    return "none";
  case INPUT_EMBEDDING_PE_MAP:
    return "pe_map";
  case INPUT_EMBEDDING_PE_DENSE:
    return "pe_dense";
  }
  return "unknown";
}

template <typename Map>
std::string SelectHeadName(const Map &heads, const std::string &preferred) {
  if (heads.count(preferred) != 0)
    return preferred;
  if (heads.empty())
    return "";

  std::vector<std::string> names;
  names.reserve(heads.size());
  for (const auto &entry : heads)
    names.push_back(entry.first);
  std::sort(names.begin(), names.end());
  return names.front();
}

} // namespace

std::string
ActivationToString(MetalFishNN::NetworkFormat_ActivationFunction activation) {
  switch (activation) {
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

NetworkFormatDescriptor DescribeNetworkFormat(const WeightsFile &file) {
  NetworkFormatDescriptor descriptor;
  const auto &nf = file.format().network_format();

  descriptor.wdl =
      nf.value() == MetalFishNN::NetworkFormat_ValueFormat_VALUE_WDL;
  descriptor.moves_left =
      nf.moves_left() ==
      MetalFishNN::NetworkFormat_MovesLeftFormat_MOVES_LEFT_V1;
  descriptor.conv_policy =
      nf.policy() == MetalFishNN::NetworkFormat_PolicyFormat_POLICY_CONVOLUTION;
  descriptor.attention_policy =
      nf.policy() == MetalFishNN::NetworkFormat_PolicyFormat_POLICY_ATTENTION;
  descriptor.attention_body =
      nf.network() ==
          MetalFishNN::
              NetworkFormat_NetworkStructure_NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
      nf.network() ==
          MetalFishNN::
              NetworkFormat_NetworkStructure_NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;
  descriptor.body_attention_heads =
      static_cast<int>(file.weights().headcount());
  descriptor.policy_attention_heads =
      static_cast<int>(file.weights().pol_headcount());

  descriptor.input_embedding = static_cast<InputEmbedding>(
      nf.has_input_embedding()
          ? nf.input_embedding()
          : MetalFishNN::NetworkFormat::INPUT_EMBEDDING_PE_MAP);

  descriptor.activations.default_activation =
      (nf.default_activation() ==
       MetalFishNN::NetworkFormat_DefaultActivation_DEFAULT_ACTIVATION_MISH)
          ? "mish"
          : "relu";

  descriptor.activations.smolgen_activation =
      ActivationToString(nf.smolgen_activation());
  if (descriptor.activations.smolgen_activation == "relu" &&
      nf.smolgen_activation() ==
          MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_DEFAULT) {
    descriptor.activations.smolgen_activation =
        descriptor.activations.default_activation;
  }

  descriptor.activations.ffn_activation =
      ActivationToString(nf.ffn_activation());
  if (descriptor.activations.ffn_activation == "relu" &&
      nf.ffn_activation() ==
          MetalFishNN::NetworkFormat_ActivationFunction_ACTIVATION_DEFAULT) {
    descriptor.activations.ffn_activation =
        descriptor.activations.default_activation;
  }

  return descriptor;
}

std::string NetworkFormatDescriptor::Summary() const {
  std::ostringstream out;
  out << "attention_body=" << BoolString(attention_body) << ", policy="
      << (attention_policy ? "attention" : (conv_policy ? "conv" : "classical"))
      << ", value=" << (wdl ? "wdl" : "scalar")
      << ", moves_left=" << BoolString(moves_left)
      << ", body_heads=" << body_attention_heads
      << ", policy_heads=" << policy_attention_heads
      << ", input_embedding=" << InputEmbeddingToString(input_embedding)
      << ", activations=" << activations.default_activation << "/"
      << activations.smolgen_activation << "/" << activations.ffn_activation;
  return out.str();
}

std::string SelectPolicyHeadName(const MultiHeadWeights &weights) {
  return SelectHeadName(weights.policy_heads, "vanilla");
}

std::string SelectValueHeadName(const MultiHeadWeights &weights) {
  return SelectHeadName(weights.value_heads, "winner");
}

} // namespace NN
} // namespace MetalFish

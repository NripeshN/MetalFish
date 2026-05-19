/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>

#include "loader.h"
#include "weights.h"

namespace MetalFish {
namespace NN {

struct NetworkActivations {
  std::string default_activation;
  std::string smolgen_activation;
  std::string ffn_activation;
};

struct NetworkFormatDescriptor {
  bool wdl = false;
  bool moves_left = false;
  bool conv_policy = false;
  bool attention_policy = false;
  bool attention_body = false;
  InputEmbedding input_embedding = INPUT_EMBEDDING_PE_MAP;
  NetworkActivations activations;

  std::string Summary() const;
};

std::string
ActivationToString(MetalFishNN::NetworkFormat_ActivationFunction activation);

NetworkFormatDescriptor DescribeNetworkFormat(const WeightsFile &file);

std::string SelectPolicyHeadName(const MultiHeadWeights &weights);
std::string SelectValueHeadName(const MultiHeadWeights &weights);

} // namespace NN
} // namespace MetalFish

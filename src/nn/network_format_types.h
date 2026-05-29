/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>

namespace MetalFish {
namespace NN {

enum InputEmbedding {
  INPUT_EMBEDDING_NONE = 0,
  INPUT_EMBEDDING_PE_MAP = 1,
  INPUT_EMBEDDING_PE_DENSE = 2,
};

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
  int body_attention_heads = 0;
  int policy_attention_heads = 0;
  InputEmbedding input_embedding = INPUT_EMBEDDING_PE_MAP;
  NetworkActivations activations;

  std::string Summary() const;
};

} // namespace NN
} // namespace MetalFish

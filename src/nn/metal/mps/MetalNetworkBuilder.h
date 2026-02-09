/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>
#include <vector>

namespace MetalFish {
namespace NN {
namespace Metal {

using MetalFish::NN::InputEmbedding;
using MetalFish::NN::MultiHeadWeights;

struct Activations {
  std::string default_activation = "relu";
  std::string smolgen_activation = "swish";
  std::string ffn_activation = "relu_2";
};

class MetalNetworkBuilder {
public:
  MetalNetworkBuilder(void);
  ~MetalNetworkBuilder(void);

  std::string init(int gpu_id);

  void build(int kInputPlanes, MultiHeadWeights &weights,
             InputEmbedding embedding, bool attn_body, bool attn_policy,
             bool conv_policy, bool wdl, bool moves_left,
             Activations &activations, std::string &policy_head,
             std::string &value_head);

  void forwardEval(float *values, uint64_t *masks, int batchSize,
                   std::vector<float *> output_mems);

private:
  int gpu_id;
};

} // namespace Metal
} // namespace NN
} // namespace MetalFish

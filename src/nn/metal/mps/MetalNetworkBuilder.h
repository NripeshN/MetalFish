/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../../network_legacy.h"

namespace MetalFish {
namespace NN {
namespace Metal {

struct Activations {
  std::string default_activation = "relu";
  std::string smolgen_activation = "swish";
  std::string ffn_activation = "relu_2";
};

class MetalNetworkBuilder {
 public:
  MetalNetworkBuilder(void);
  ~MetalNetworkBuilder(void);

  std::string init(int gpu_id, int max_batch);

  void build(int kInputPlanes, MultiHeadWeights& weights,
             InputEmbedding embedding, bool attn_body, bool attn_policy,
             bool conv_policy, bool wdl, bool moves_left,
             Activations& activations, std::string& policy_head,
             std::string& value_head);

  void forwardEval(float* values, uint64_t* masks, int batchSize,
                   std::vector<float*> output_mems);

 private:
  int gpu_id;
  int max_batch_size_;
};

}  // namespace Metal
}  // namespace NN
}  // namespace MetalFish

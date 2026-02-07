/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../network.h"
#include "../network_legacy.h"
#include "metal_common.h"
#include "mps/MetalNetworkBuilder.h"

namespace MetalFish {
namespace NN {
namespace Metal {

// Metal backend implementation using MPSGraph and transformer weights.
class MetalNetwork : public Network {
 public:
  explicit MetalNetwork(const WeightsFile& file, int gpu_id = 0,
                        int max_batch = 256, int batch = 64);
  ~MetalNetwork() override;

  NetworkOutput Evaluate(const InputPlanes& input) override;
  std::vector<NetworkOutput> EvaluateBatch(
      const std::vector<InputPlanes>& inputs) override;
  std::string GetNetworkInfo() const override;

 private:
  void RunBatch(const std::vector<InputPlanes>& inputs,
                std::vector<NetworkOutput>& outputs);

  std::unique_ptr<MetalNetworkBuilder> builder_;
  bool wdl_;
  bool moves_left_;
  bool conv_policy_;
  bool attn_policy_;
  int max_batch_size_;
  int batch_size_;
  std::string device_name_;
  std::mutex gpu_mutex_;
};

}  // namespace Metal
}  // namespace NN
}  // namespace MetalFish


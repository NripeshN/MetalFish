/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  Licensed under GPL-3.0
*/

#pragma once

#include "../network.h"
#include "../loader.h"
#include <memory>

namespace MetalFish {
namespace NN {
namespace Metal {

class MetalNetwork : public Network {
public:
  explicit MetalNetwork(const WeightsFile& weights);
  ~MetalNetwork() override;
  
  NetworkOutput Evaluate(const InputPlanes& input) override;
  std::vector<NetworkOutput> EvaluateBatch(
      const std::vector<InputPlanes>& inputs) override;
  std::string GetNetworkInfo() const override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace Metal
}  // namespace NN
}  // namespace MetalFish

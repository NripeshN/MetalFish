/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include "../network.h"

#include <memory>
#include <string>

namespace MetalFish {
namespace NN {
namespace CoreML {

class CoreMLNetwork : public Network {
public:
  CoreMLNetwork(const WeightsFile &file, const std::string &model_path,
                const std::string &compute_units);
  ~CoreMLNetwork() override;

  NetworkOutput Evaluate(const InputPlanes &input) override;
  std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) override;
  std::string GetNetworkInfo() const override;
  bool HasWDL() const override;
  bool HasMovesLeft() const override;
  BackendCapabilities GetBackendCapabilities() const override;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace CoreML
} // namespace NN
} // namespace MetalFish

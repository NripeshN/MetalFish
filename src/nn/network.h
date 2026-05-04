/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "encoder.h"
#include "loader.h"

namespace MetalFish {
namespace NN {

struct NetworkOutput {
  std::vector<float> policy;
  float value;
  float wdl[3];
  bool has_wdl;
  float moves_left = 0.0f;
  bool has_moves_left = false;
};

class Network {
public:
  virtual ~Network() = default;

  virtual NetworkOutput Evaluate(const InputPlanes &input) = 0;

  virtual std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) = 0;

  virtual std::string GetNetworkInfo() const = 0;
};

std::unique_ptr<Network> CreateNetwork(const std::string &weights_path,
                                       const std::string &backend = "auto");
std::unique_ptr<Network> CreateNetwork(const WeightsFile &weights,
                                       const std::string &backend = "auto");

} // namespace NN
} // namespace MetalFish

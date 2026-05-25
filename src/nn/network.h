/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "input_planes.h"
#include "network_output.h"
#include "weights_file.h"

namespace MetalFish {
namespace NN {

class Network {
public:
  virtual ~Network() = default;

  virtual NetworkOutput Evaluate(const InputPlanes &input) = 0;

  virtual std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) = 0;

  virtual std::string GetNetworkInfo() const = 0;
  virtual bool HasWDL() const { return false; }
  virtual bool HasMovesLeft() const { return false; }
};

std::unique_ptr<Network> CreateNetwork(const std::string &weights_path,
                                       const std::string &backend = "auto");
std::unique_ptr<Network> CreateNetwork(const WeightsFile &weights,
                                       const std::string &backend = "auto");
std::unique_ptr<Network>
CreateNetwork(const WeightsFile &weights, const std::string &backend,
              const std::string &model_path,
              const std::string &compute_units = "cpu-ne");

} // namespace NN
} // namespace MetalFish

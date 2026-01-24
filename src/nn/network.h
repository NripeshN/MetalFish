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

// Neural network output structure
struct NetworkOutput {
  std::vector<float> policy;  // 1858 move probabilities
  float value;                 // Position evaluation (-1 to 1)
  float wdl[3];               // Win/Draw/Loss probabilities
  bool has_wdl;
};

// Abstract neural network interface
class Network {
public:
  virtual ~Network() = default;
  
  // Evaluate single position
  virtual NetworkOutput Evaluate(const InputPlanes& input) = 0;
  
  // Batch evaluation
  virtual std::vector<NetworkOutput> EvaluateBatch(
      const std::vector<InputPlanes>& inputs) = 0;
  
  // Get network information
  virtual std::string GetNetworkInfo() const = 0;
};

// Factory function to create network backend
std::unique_ptr<Network> CreateNetwork(const std::string& weights_path,
                                      const std::string& backend = "auto");

}  // namespace NN
}  // namespace MetalFish

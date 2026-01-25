/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network.h"

#ifdef USE_METAL
#include "metal/metal_network.h"
#endif

#include <stdexcept>

namespace MetalFish {
namespace NN {

// Stub implementation of network
class StubNetwork : public Network {
public:
  StubNetwork(const WeightsFile& weights) : weights_(weights) {}
  
  NetworkOutput Evaluate(const InputPlanes& input) override {
    // Stub implementation - returns random-ish policy and neutral value
    NetworkOutput output;
    output.policy.resize(kPolicyOutputs, 1.0f / kPolicyOutputs);
    output.value = 0.0f;
    output.has_wdl = false;
    return output;
  }
  
  std::vector<NetworkOutput> EvaluateBatch(
      const std::vector<InputPlanes>& inputs) override {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto& input : inputs) {
      outputs.push_back(Evaluate(input));
    }
    return outputs;
  }
  
  std::string GetNetworkInfo() const override {
    return "Stub network (not functional)";
  }

private:
  WeightsFile weights_;
};

std::unique_ptr<Network> CreateNetwork(const std::string& weights_path,
                                      const std::string& backend) {
  // Try to load weights
  auto weights_opt = LoadWeights(weights_path);
  
  if (!weights_opt.has_value()) {
    throw std::runtime_error("Could not load network weights from: " + weights_path);
  }
  
#ifdef USE_METAL
  if (backend == "auto" || backend == "metal") {
    try {
      return std::make_unique<Metal::MetalNetwork>(weights_opt.value());
    } catch (const std::exception& e) {
      if (backend == "metal") {
        // If Metal was explicitly requested, propagate error
        throw;
      }
      // Otherwise fall through to stub
    }
  }
#endif
  
  // Fallback to stub implementation
  return std::make_unique<StubNetwork>(weights_opt.value());
}

}  // namespace NN
}  // namespace MetalFish

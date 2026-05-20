/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "network.h"

#ifdef USE_COREML
#include "coreml/coreml_network.h"
#endif
#ifdef USE_METAL
#include "metal/metal_network.h"
#endif

#include <iostream>
#include <stdexcept>

namespace MetalFish {
namespace NN {

class StubNetwork : public Network {
public:
  StubNetwork(const WeightsFile &weights) : weights_(weights) {}

  NetworkOutput Evaluate(const InputPlanes &input) override {
    NetworkOutput output;
    output.policy.fill(1.0f / kPolicyOutputs);
    output.value = 0.0f;
    output.has_wdl = false;
    output.wdl[0] = output.wdl[1] = output.wdl[2] = 0.0f;
    output.has_moves_left = false;
    return output;
  }

  std::vector<NetworkOutput>
  EvaluateBatch(const std::vector<InputPlanes> &inputs) override {
    std::vector<NetworkOutput> outputs;
    outputs.reserve(inputs.size());
    for (const auto &input : inputs) {
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

std::unique_ptr<Network> CreateNetwork(const WeightsFile &weights,
                                       const std::string &backend,
                                       const std::string &model_path,
                                       const std::string &compute_units) {
#ifdef USE_COREML
  if (backend == "coreml") {
    return std::make_unique<CoreML::CoreMLNetwork>(weights, model_path,
                                                   compute_units);
  }
#else
  if (backend == "coreml") {
    throw std::runtime_error("Core ML backend was not compiled into MetalFish");
  }
#endif

#ifdef USE_METAL
  if (backend == "auto" || backend == "metal") {
    try {
      return std::make_unique<Metal::MetalNetwork>(weights);
    } catch (const std::exception &e) {
      std::cerr << "Metal backend unavailable: " << e.what() << std::endl;
      if (backend == "metal") {
        throw;
      }
    }
  }
#endif

  return std::make_unique<StubNetwork>(weights);
}

std::unique_ptr<Network> CreateNetwork(const WeightsFile &weights,
                                       const std::string &backend) {
  return CreateNetwork(weights, backend, "", "cpu-ne");
}

std::unique_ptr<Network> CreateNetwork(const std::string &weights_path,
                                       const std::string &backend) {
  auto weights_opt = LoadWeights(weights_path);

  if (!weights_opt.has_value()) {
    throw std::runtime_error("Could not load network weights from: " +
                             weights_path);
  }

  return CreateNetwork(weights_opt.value(), backend);
}

} // namespace NN
} // namespace MetalFish
